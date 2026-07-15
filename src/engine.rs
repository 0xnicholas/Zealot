//! Engine：驱动 Scheduler + ModelRunner 的步进循环。
//!
//! 每个 step：schedule → runner 前向一步 → 写回 token → 检查停止条件
//! （EOS / max_tokens）→ finish 并释放 block。纯粹的编排逻辑，
//! 不感知模型实现（dev-mode PyTorch CPU / 目标 CUDA kernel）。

use crate::error::ZealotError;
use crate::scheduler::{FinishReason, Scheduler, Sequence};

/// 模型执行端。dev-mode: PyTorch CPU（`model_runner_py`）；
/// 目标形态: Rust + CUDA kernel。实现必须是 `Send`（Engine 跑在独立线程）。
pub trait ModelRunner: Send {
    /// 对批次中每个 seq 执行一步（seq.is_prefill() 区分 prefill/decode，
    /// 输入取 seq.step_input()）。返回每个 seq 的新 token 及其文本。
    fn step(&mut self, batch: &mut [&mut Sequence]) -> Result<Vec<StepOut>, ZealotError>;

    /// 丢弃 seq 的执行侧状态（被抢占 / 完成 / 取消时调用）。
    /// 例如 PyModelRunner 要释放该 seq 的 past_key_values。
    fn drop_state(&mut self, _request_id: &str) {}
}

pub struct StepOut {
    pub request_id: String,
    pub token: i64,
    /// 增量文本（runner 侧 detokenize）。None 表示 runner 不提供文本。
    pub text: Option<String>,
}

pub struct TokenOut {
    pub request_id: String,
    pub token: i64,
    pub text: Option<String>,
}

pub struct FinishedOut {
    pub request_id: String,
    pub reason: FinishReason,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Default)]
pub struct StepResult {
    pub tokens: Vec<TokenOut>,
    pub finished: Vec<FinishedOut>,
}

pub struct Engine<R: ModelRunner> {
    sched: Scheduler,
    runner: R,
}

impl<R: ModelRunner> Engine<R> {
    pub fn new(sched: Scheduler, runner: R) -> Self {
        Self { sched, runner }
    }

    pub fn scheduler(&self) -> &Scheduler {
        &self.sched
    }

    pub fn scheduler_mut(&mut self) -> &mut Scheduler {
        &mut self.sched
    }

    pub fn runner(&self) -> &R {
        &self.runner
    }

    pub fn runner_mut(&mut self) -> &mut R {
        &mut self.runner
    }

    pub fn is_idle(&self) -> bool {
        self.sched.is_idle()
    }

    /// 取消：释放资源并丢弃 runner 状态。命中时返回 (prompt_len, completion_len)。
    pub fn cancel(&mut self, request_id: &str) -> Option<(usize, usize)> {
        use crate::scheduler::CancelOutcome;
        match self.sched.cancel(request_id) {
            CancelOutcome::WasRunning(p, c) => {
                self.runner.drop_state(request_id);
                Some((p, c))
            }
            CancelOutcome::WasWaiting(p) => Some((p, 0)),
            CancelOutcome::NotFound => None,
        }
    }

    /// 执行一步。空闲时返回空结果。
    pub fn step(&mut self) -> Result<StepResult, ZealotError> {
        let out = self.sched.schedule();
        for id in &out.preempted {
            self.runner.drop_state(id);
        }

        let mut result = StepResult::default();
        let mut batch = out.batch;
        if batch.is_empty() {
            return Ok(result);
        }

        let was_prefill: Vec<bool> = batch.iter().map(|s| s.is_prefill()).collect();
        for step_out in self.runner.step(&mut batch)? {
            let Some(seq) = batch
                .iter_mut()
                .find(|s| s.request_id == step_out.request_id)
            else {
                continue; // runner 返回了未知 seq，忽略
            };
            seq.output_tokens.push(step_out.token);
            result.tokens.push(TokenOut {
                request_id: step_out.request_id,
                token: step_out.token,
                text: step_out.text,
            });
        }
        for (seq, was) in batch.iter_mut().zip(was_prefill) {
            if was {
                seq.mark_prefilled();
            }
        }

        // 停止条件：EOS 优先于 max_tokens
        let finished: Vec<(String, FinishReason)> = batch
            .iter()
            .filter_map(|seq| {
                if seq.eos_token_id.is_some() && seq.output_tokens.last() == seq.eos_token_id.as_ref() {
                    Some((seq.request_id.clone(), FinishReason::Stop))
                } else if seq.output_tokens.len() >= seq.max_tokens {
                    Some((seq.request_id.clone(), FinishReason::Length))
                } else {
                    None
                }
            })
            .collect();
        for (request_id, reason) in finished {
            self.runner.drop_state(&request_id);
            if let Some((prompt, completion)) = self.sched.finish(&request_id) {
                result.finished.push(FinishedOut {
                    request_id,
                    reason,
                    prompt_tokens: prompt,
                    completion_tokens: completion,
                });
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::{Priority, SchedulerConfig};
    use std::collections::VecDeque;

    /// 按脚本逐次发 token 的假 runner。
    struct ScriptRunner {
        tokens: VecDeque<i64>,
    }

    impl ModelRunner for ScriptRunner {
        fn step(&mut self, batch: &mut [&mut Sequence]) -> Result<Vec<StepOut>, ZealotError> {
            Ok(batch
                .iter()
                .map(|s| StepOut {
                    request_id: s.request_id.clone(),
                    token: self.tokens.pop_front().unwrap_or(0),
                    text: None,
                })
                .collect())
        }
    }

    fn engine(tokens: Vec<i64>) -> Engine<ScriptRunner> {
        let sched = Scheduler::new(SchedulerConfig {
            max_num_seqs: 4,
            block_size: 2,
            num_gpu_blocks: 16,
        })
        .unwrap();
        Engine::new(
            sched,
            ScriptRunner {
                tokens: tokens.into(),
            },
        )
    }

    fn submit(engine: &mut Engine<ScriptRunner>, id: &str, max_tokens: usize, eos: Option<i64>) {
        let seq = engine
            .scheduler_mut()
            .make_sequence(id.into(), vec![1, 2], max_tokens, Priority::Medium, eos)
            .unwrap();
        engine.scheduler_mut().add(seq);
    }

    #[test]
    fn finishes_on_max_tokens_and_frees_blocks() {
        let mut engine = engine(vec![10, 11, 12]);
        submit(&mut engine, "a", 3, None);

        let mut total_tokens = 0;
        let mut finished = Vec::new();
        while !engine.is_idle() {
            let res = engine.step().unwrap();
            total_tokens += res.tokens.len();
            finished.extend(res.finished);
        }

        assert_eq!(total_tokens, 3);
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].reason, FinishReason::Length);
        assert_eq!(finished[0].completion_tokens, 3);
    }

    #[test]
    fn finishes_early_on_eos() {
        // 第二个 token 即 EOS（99），max_tokens 远未到
        let mut engine = engine(vec![10, 99, 12, 13]);
        submit(&mut engine, "a", 10, Some(99));

        let mut finished = Vec::new();
        while !engine.is_idle() {
            finished.extend(engine.step().unwrap().finished);
        }

        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].reason, FinishReason::Stop);
        assert_eq!(finished[0].completion_tokens, 2);
    }

    #[test]
    fn batches_multiple_sequences() {
        let mut engine = engine(vec![10, 20, 11, 21]);
        submit(&mut engine, "a", 2, None);
        submit(&mut engine, "b", 2, None);

        let first = engine.step().unwrap();
        assert_eq!(first.tokens.len(), 2, "one batch step covers both seqs");
        while !engine.is_idle() {
            engine.step().unwrap();
        }
    }

    #[test]
    fn cancel_running_seq() {
        let mut engine = engine(vec![10, 11, 12]);
        submit(&mut engine, "a", 3, None);

        engine.step().unwrap();
        assert!(engine.cancel("a").is_some());
        assert!(engine.is_idle());
        assert!(engine.cancel("a").is_none(), "already finished");
    }
}
