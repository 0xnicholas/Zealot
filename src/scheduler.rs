//! Continuous batching scheduler（逻辑等价 vLLM，ADR-009 / zealot/docs/architecture.md §7）。
//!
//! GPU-free 纯 CPU 逻辑：等待队列（优先级排序）、token/block 预算、KV block
//! 簿记（经 BlockManager）、OOM 抢占（recompute 语义）。模型前向在
//! `crate::engine::ModelRunner` trait 后面，本模块不感知。

use std::cmp::Reverse;

use crate::block_manager::{BlockHandle, BlockManager};
use crate::error::ZealotError;
use crate::sampling::SamplingParams;

/// 调度优先级，与 proto `runtime.v1.Priority` 一一对应（bin 层做转换）。
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Priority {
    Lowest = 0,
    Medium = 1,
    High = 2,
    Highest = 3,
}

impl Default for Priority {
    fn default() -> Self {
        Priority::Medium
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FinishReason {
    /// EOS token 或 stop 条件命中
    Stop,
    /// 达到 max_tokens
    Length,
    /// 用户取消
    Cancel,
}

impl FinishReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::Cancel => "cancel",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SeqStatus {
    Waiting,
    Running,
}

/// 一条推理序列。token 用 i64（对齐 PyTorch 的 int64）。
#[derive(Debug)]
pub struct Sequence {
    pub request_id: String,
    pub prompt_tokens: Vec<i64>,
    pub output_tokens: Vec<i64>,
    pub max_tokens: usize,
    pub priority: Priority,
    pub eos_token_id: Option<i64>,
    arrival: u64,
    status: SeqStatus,
    blocks: Vec<BlockHandle>,
    /// Sampling parameters for this sequence.
    pub sampling_params: SamplingParams,
    /// true = 下一步需要 prefill（初始，或被抢占后需 recompute）
    prefill_pending: bool,
    prefill_pos: usize,
    chunk_size: usize,
}

impl Sequence {
    pub fn new(
        request_id: String,
        prompt_tokens: Vec<i64>,
        max_tokens: usize,
        priority: Priority,
        eos_token_id: Option<i64>,
        arrival: u64,
        sampling_params: SamplingParams,
    ) -> Self {
        Self {
            request_id,
            prompt_tokens,
            output_tokens: Vec::new(),
            max_tokens,
            priority,
            eos_token_id,
            arrival,
            status: SeqStatus::Waiting,
            blocks: Vec::new(),
            prefill_pending: true,
            prefill_pos: 0,
            chunk_size: 0,
            sampling_params,
        }
    }

    /// 总长度（prompt + 已生成）。被抢占的 seq 保留 output_tokens，
    /// 恢复时整体 recompute（vLLM recompute 语义）。
    pub fn len(&self) -> usize {
        if self.prefill_pending {
            self.prefill_pos + self.output_tokens.len() + self.chunk_size
        } else {
            self.prompt_tokens.len() + self.output_tokens.len()
        }
    }

    /// 本步喂给 runner 的输入：prefill 为全部 token，decode 为上一步 token。
    pub fn step_input(&self) -> Vec<i64> {
        if self.prefill_pending {
            let end = (self.prefill_pos + self.chunk_size).min(self.prompt_tokens.len());
            let mut ids: Vec<i64> = self.prompt_tokens[self.prefill_pos..end].to_vec();
            ids.extend_from_slice(&self.output_tokens);
            ids
        } else {
            vec![*self.output_tokens.last().expect("decode step requires generated token")]
        }
    }

    pub fn is_prefill(&self) -> bool {
        self.prefill_pending
    }

    pub fn mark_prefilled(&mut self) {
        self.prefill_pending = false;
    }

    pub fn is_final_chunk(&self) -> bool {
        self.prefill_pos + self.chunk_size >= self.prompt_tokens.len()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SchedulerConfig {
    /// 同时运行的最大序列数（decode 阶段）
    pub max_num_seqs: usize,
    /// KV block 大小（token 数）
    pub block_size: usize,
    /// block 池总容量
    pub num_gpu_blocks: usize,
    /// 单步 prefill 的最大 token 总数（限制 prefilling 的计算量）
    pub max_prefill_tokens: usize,
    pub prefill_chunk_size: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 8,
            block_size: 16,
            num_gpu_blocks: 1024,
            max_prefill_tokens: 2048,
            prefill_chunk_size: 512,
        }
    }
}

fn blocks_for(tokens: usize, block_size: usize) -> usize {
    tokens.div_ceil(block_size)
}

/// schedule() 的返回：本步批次 + 被抢占的 seq（runner 须丢弃其状态）。
pub struct ScheduleOutput<'a> {
    pub batch: Vec<&'a mut Sequence>,
    /// 本步被抢占（recompute 回等待队列）的 request_id
    pub preempted: Vec<String>,
}

pub enum CancelOutcome {
    /// 等待队列中取消（未消耗任何计算），返回 prompt_len
    WasWaiting(usize),
    /// 运行中取消，返回 (prompt_len, output_len)——已生成 token 照常计费
    WasRunning(usize, usize),
    NotFound,
}

pub struct Scheduler {
    cfg: SchedulerConfig,
    bm: BlockManager,
    /// 按 (priority desc, arrival asc) 排序的等待队列
    waiting: Vec<Sequence>,
    /// Prefill 阶段的序列（计算密集，受 max_prefill_tokens 限制）
    prefilling: Vec<Sequence>,
    /// Decode 阶段的序列（内存密集，受 max_num_seqs 限制）
    decoding: Vec<Sequence>,
    next_arrival: u64,
}

impl Scheduler {
    pub fn new(cfg: SchedulerConfig) -> Result<Self, ZealotError> {
        Ok(Self {
            cfg,
            bm: BlockManager::create(cfg.num_gpu_blocks, cfg.block_size)?,
            waiting: Vec::new(),
            prefilling: Vec::new(),
            decoding: Vec::new(),
            next_arrival: 0,
        })
    }

    /// 准入控制：最坏情况（prompt + max_tokens）超过整个 block 池的序列
    /// 直接拒绝，避免上量后永远调不动（vLLM 同款检查）。
    pub fn make_sequence(
        &mut self,
        request_id: String,
        prompt_tokens: Vec<i64>,
        max_tokens: usize,
        priority: Priority,
        eos_token_id: Option<i64>,
        sampling_params: SamplingParams,
    ) -> Result<Sequence, ZealotError> {
        let worst = blocks_for(prompt_tokens.len() + max_tokens, self.cfg.block_size);
        if worst > self.cfg.num_gpu_blocks {
            return Err(ZealotError::SequenceTooLong {
                required: worst,
                capacity: self.cfg.num_gpu_blocks,
            });
        }
        let arrival = self.next_arrival;
        self.next_arrival += 1;
        Ok(Sequence::new(
            request_id,
            prompt_tokens,
            max_tokens,
            priority,
            eos_token_id,
            arrival,
            sampling_params,
        ))
    }

    /// 加入等待队列（按 priority desc、arrival asc 稳定插入）。
    pub fn add(&mut self, seq: Sequence) {
        self.insert_waiting(seq);
    }

    fn insert_waiting(&mut self, seq: Sequence) {
        let key = (Reverse(seq.priority), seq.arrival);
        let pos = self
            .waiting
            .partition_point(|s| (Reverse(s.priority), s.arrival) <= key);
        self.waiting.insert(pos, seq);
    }

    /// 形成本步批次：
    /// 1. 提升 waiting → prefilling（受 max_prefill_tokens 和 block 预算约束）
    /// 2. 为 prefilling + decoding 分配/补充 block（OOM → 抢占）
    /// 批量返回所有活跃 seq（prefilling + decoding）。
    /// Runner 通过 seq.is_prefill() 区分阶段。
    pub fn schedule(&mut self) -> ScheduleOutput<'_> {
        let mut preempted = Vec::new();
        let total_seqs = self.prefilling.len() + self.decoding.len();

        // ── 1. 提升 waiting → prefilling ───────────────────────────────
        //    同时受 max_num_seqs（总运行数）和 max_prefill_tokens 限制。
        let mut prefill_tokens = 0_usize;
        while let Some(seq) = self.waiting.first() {
            if total_seqs + self.prefilling.len() >= self.cfg.max_num_seqs {
                break;
            }
            let tokens = seq.len(); // prompt + 已生成的 output_tokens
            if prefill_tokens + tokens > self.cfg.max_prefill_tokens && prefill_tokens > 0 {
                break; // 本步 prefill token 预算用尽，剩余的 prefill 等下一步
            }
            let need = blocks_for(tokens, self.cfg.block_size);
            if self.bm.available() < need {
                break; // block 不足，等 running 释放或抢占
            }
            let mut seq = self.waiting.remove(0);
            for _ in 0..need {
                let h = self.bm.try_allocate().expect("available checked");
                seq.blocks.push(h);
            }
            seq.status = SeqStatus::Running;
            prefill_tokens += tokens;
            self.prefilling.push(seq);
        }

        // ── 2. 为 prefilling + decoding 补齐 block ────────────────────────
        let mut i = 0;
        while i < self.prefilling.len() {
            i = Self::ensure_blocks(
                &mut self.prefilling,
                &self.cfg,
                &mut self.bm,
                &mut self.waiting,
                i,
                &mut preempted,
            ) + 1;
        }
        let mut i = 0;
        while i < self.decoding.len() {
            i = Self::ensure_blocks(
                &mut self.decoding,
                &self.cfg,
                &mut self.bm,
                &mut self.waiting,
                i,
                &mut preempted,
            ) + 1;
        }

        // ── 3. 组装批次 ──────────────────────────────────────────────────
        let prefilling = &mut self.prefilling;
        let decoding = &mut self.decoding;
        let batch: Vec<&mut Sequence> = prefilling.iter_mut().chain(decoding.iter_mut()).collect();

        ScheduleOutput { batch, preempted }
    }

    /// 确保 queue[idx] 的 block 数覆盖其当前长度。
    /// 返回 requester 在抢占调整后的新索引。
    fn ensure_blocks(
        queue: &mut Vec<Sequence>,
        cfg: &SchedulerConfig,
        bm: &mut BlockManager,
        waiting: &mut Vec<Sequence>,
        idx: usize,
        preempted: &mut Vec<String>,
    ) -> usize {
        let mut idx = idx;
        let need = blocks_for(queue[idx].len(), cfg.block_size);
        while queue[idx].blocks.len() < need {
            match bm.try_allocate() {
                Ok(h) => queue[idx].blocks.push(h),
                Err(_) => {
                    // 抢占 queue 中优先级最低的（除 requester 自身）
                    let victim_idx = queue
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != idx)
                        .min_by_key(|(_, s)| (s.priority, Reverse(s.arrival)))
                        .map(|(i, _)| i);
                    match victim_idx {
                        Some(vi) => {
                            let mut victim = queue.remove(vi);
                            for h in std::mem::take(&mut victim.blocks) {
                                bm.try_free(&h).expect("owned handle");
                            }
                            victim.prefill_pending = true;
                            victim.status = SeqStatus::Waiting;
                            preempted.push(victim.request_id.clone());
                            let key = (Reverse(victim.priority), victim.arrival);
                            let pos = waiting
                                .partition_point(|s| (Reverse(s.priority), s.arrival) <= key);
                            waiting.insert(pos, victim);
                            if vi < idx {
                                idx -= 1;
                            }
                        }
                        None => break, // 只剩自己，等下一轮
                    }
                }
            }
        }
        idx
    }

    /// 完成：从 prefilling 或 decoding 队列移除并释放全部 block。
    pub fn finish(&mut self, request_id: &str) -> Option<(usize, usize)> {
        if let Some(idx) = self
            .prefilling
            .iter()
            .position(|s| s.request_id == request_id)
        {
            let mut seq = self.prefilling.remove(idx);
            for h in std::mem::take(&mut seq.blocks) {
                self.bm.try_free(&h).expect("owned handle");
            }
            return Some((seq.prompt_tokens.len(), seq.output_tokens.len()));
        }
        if let Some(idx) = self
            .decoding
            .iter()
            .position(|s| s.request_id == request_id)
        {
            let mut seq = self.decoding.remove(idx);
            for h in std::mem::take(&mut seq.blocks) {
                self.bm.try_free(&h).expect("owned handle");
            }
            return Some((seq.prompt_tokens.len(), seq.output_tokens.len()));
        }
        None
    }

    /// 将指定 seq 从 prefilling 移至 decoding（Engine 在 prefill 完成后调用）
    pub fn promote_to_decoding(&mut self, request_id: &str) {
        if let Some(idx) = self
            .prefilling
            .iter()
            .position(|s| s.request_id == request_id)
        {
            let mut seq = self.prefilling.remove(idx);
            seq.prefill_pending = false;
            self.decoding.push(seq);
        }
    }

    /// 取消：waiting 中直接移除；running 中按 finish 处理（返回 WasRunning，
    /// 调用方负责发 final 帧——ADR-010 取消语义）。
    pub fn cancel(&mut self, request_id: &str) -> CancelOutcome {
        if let Some(idx) = self.waiting.iter().position(|s| s.request_id == request_id) {
            let seq = self.waiting.remove(idx);
            return CancelOutcome::WasWaiting(seq.prompt_tokens.len());
        }
        if let Some((p, c)) = self.finish(request_id) {
            return CancelOutcome::WasRunning(p, c);
        }
        CancelOutcome::NotFound
    }

    pub fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.prefilling.is_empty() && self.decoding.is_empty()
    }

    #[cfg(test)]
    fn free_blocks(&self) -> usize {
        self.bm.available()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(max_num_seqs: usize, block_size: usize, num_gpu_blocks: usize) -> SchedulerConfig {
        SchedulerConfig {
            max_num_seqs,
            block_size,
            num_gpu_blocks,
            max_prefill_tokens: 1024,
        }
    }

    fn submit(
        sched: &mut Scheduler,
        id: &str,
        prompt_len: usize,
        max_tokens: usize,
        priority: Priority,
    ) {
        let seq = sched
            .make_sequence(
                id.into(),
                vec![1; prompt_len],
                max_tokens,
                priority,
                None,
                SamplingParams::default(),
            )
            .unwrap();
        sched.add(seq);
    }

    #[test]
    fn promotion_respects_max_num_seqs() {
        let mut sched = Scheduler::new(cfg(1, 2, 16)).unwrap();
        submit(&mut sched, "a", 2, 2, Priority::Medium);
        submit(&mut sched, "b", 2, 2, Priority::Medium);

        let out = sched.schedule();
        assert_eq!(out.batch.len(), 1);
        assert_eq!(out.batch[0].request_id, "a");
    }

    #[test]
    fn priority_jumps_queue() {
        let mut sched = Scheduler::new(cfg(1, 2, 16)).unwrap();
        submit(&mut sched, "low", 2, 2, Priority::Lowest);
        submit(&mut sched, "top", 2, 2, Priority::Highest);

        let out = sched.schedule();
        assert_eq!(out.batch[0].request_id, "top");
    }

    #[test]
    fn waiting_when_blocks_exhausted() {
        // 池 2 block，block_size 2：a 占 1，b 占 1 → 池空，c 等待
        let mut sched = Scheduler::new(cfg(8, 2, 2)).unwrap();
        submit(&mut sched, "a", 2, 2, Priority::Medium);
        submit(&mut sched, "b", 2, 2, Priority::Medium);
        submit(&mut sched, "c", 2, 2, Priority::Medium);

        let out = sched.schedule();
        let ids: Vec<_> = out.batch.iter().map(|s| s.request_id.as_str()).collect();
        assert_eq!(ids, ["a", "b"]);
    }

    #[test]
    fn admission_control_rejects_oversized_sequence() {
        let mut sched = Scheduler::new(cfg(8, 2, 2)).unwrap();
        // prompt 2 + max_tokens 8 = 10 token → 5 block > 池 2
        let err = sched
            .make_sequence(
                "huge".into(),
                vec![1; 2],
                8,
                Priority::Medium,
                None,
                SamplingParams::default(),
            )
            .unwrap_err();
        assert!(matches!(err, ZealotError::SequenceTooLong { .. }));
    }

    #[test]
    fn preemption_frees_blocks_and_resumes() {
        // 池 4 block，block_size 2。A: prompt 4（2 block），B: prompt 2（1 block）。
        let mut sched = Scheduler::new(cfg(2, 2, 4)).unwrap();
        submit(&mut sched, "a", 4, 4, Priority::Medium);
        submit(&mut sched, "b", 2, 2, Priority::Medium);

        let out = sched.schedule();
        assert_eq!(out.batch.len(), 2);
        assert_eq!(sched.free_blocks(), 1);

        // A 生成 2 token：len 4→6，需要 3 block（原占 2）→ 补 1 → 池空
        {
            let mut out = sched.schedule();
            let a = out.batch.iter_mut().find(|s| s.request_id == "a").unwrap();
            a.output_tokens.push(10);
            a.output_tokens.push(11);
            a.mark_prefilled();
        }
        sched.schedule();
        assert_eq!(sched.free_blocks(), 0);

        // B 生成 1 token：len 2→3，需要 2 block（原占 1）→ OOM → 抢占 A
        // （除 requester 外唯一候选）。A 释放 3 block，回等待队列；
        // B 补 1 block。0 → +3 → B 占 1 → 空闲 2。
        {
            let mut out = sched.schedule();
            let b = out.batch.iter_mut().find(|s| s.request_id == "b").unwrap();
            b.output_tokens.push(20);
            b.mark_prefilled();
        }
        let out = sched.schedule();
        assert_eq!(out.preempted, vec!["a".to_string()]);
        assert_eq!(sched.free_blocks(), 2);

        // A 在 waiting 中保留已生成的 2 个 token，且恢复时需 re-prefill
        sched.finish("b");
        let mut out = sched.schedule();
        let a = out
            .batch
            .iter_mut()
            .find(|s| s.request_id == "a")
            .expect("a resumed");
        assert!(a.is_prefill(), "preempted seq must re-prefill (recompute)");
        assert_eq!(a.step_input().len(), 6, "prompt 4 + preserved output 2");
    }

    #[test]
    fn cancel_waiting_and_running() {
        let mut sched = Scheduler::new(cfg(1, 2, 16)).unwrap();
        submit(&mut sched, "a", 2, 2, Priority::Medium);
        submit(&mut sched, "b", 2, 2, Priority::Medium);

        sched.schedule(); // a running, b waiting（max_num_seqs=1）
        assert!(matches!(sched.cancel("b"), CancelOutcome::WasWaiting(2)));
        assert!(matches!(sched.cancel("a"), CancelOutcome::WasRunning(2, 0)));
        assert!(matches!(sched.cancel("ghost"), CancelOutcome::NotFound));
        assert!(sched.is_idle());
        assert_eq!(sched.free_blocks(), 16);
    }

    #[test]
    fn finish_releases_all_blocks() {
        let mut sched = Scheduler::new(cfg(8, 2, 4)).unwrap();
        submit(&mut sched, "a", 4, 2, Priority::Medium);
        sched.schedule();
        assert_eq!(sched.free_blocks(), 2);

        let (prompt, output) = sched.finish("a").unwrap();
        assert_eq!((prompt, output), (4, 0));
        assert_eq!(sched.free_blocks(), 4);
        assert!(sched.is_idle());
    }
}
