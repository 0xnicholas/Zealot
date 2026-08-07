use pyo3::prelude::*;
// §12-2: 同一份同步代码既支持 std 也支持 loom model checker。
// build.rs 在 ZEALOT_LOOM=1 时对本 crate 发射 cfg(loom)（见 loom_tests 模块头注释），
// 运行：ZEALOT_LOOM=1 cargo test --lib loom -- --test-threads=1
#[cfg(loom)]
use loom::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
#[cfg(loom)]
use loom::sync::Mutex;
#[cfg(not(loom))]
use std::sync::Mutex;

use crate::error::ZealotError;

#[pyclass]
#[derive(Clone, Debug)]
pub struct BlockHandle {
    #[pyo3(get)]
    pub block_id: usize,

    #[pyo3(get)]
    pub generation: u64,
}

#[pymethods]
impl BlockHandle {
    #[new]
    fn new(block_id: usize, generation: u64) -> Self {
        Self {
            block_id,
            generation,
        }
    }

    fn __repr__(&self) -> String {
        format!("BlockHandle(id={}, gen={})", self.block_id, self.generation)
    }
}

#[pyclass]
#[derive(Debug)]
pub struct BlockManager {
    num_gpu_blocks: usize,
    block_size: usize,
    free_list: Mutex<Vec<usize>>,
    refcount: Vec<AtomicUsize>,
    generation: Vec<AtomicU64>,
}

#[pymethods]
impl BlockManager {
    #[new]
    pub fn new(num_gpu_blocks: usize, block_size: usize) -> PyResult<Self> {
        Ok(Self::create(num_gpu_blocks, block_size)?)
    }

    pub fn allocate(&self) -> PyResult<BlockHandle> {
        Ok(self.try_allocate()?)
    }

    pub fn free(&self, handle: &BlockHandle) -> PyResult<()> {
        Ok(self.try_free(handle)?)
    }

    fn reference(&self, handle: &BlockHandle) -> PyResult<()> {
        Ok(self.try_reference(handle)?)
    }

    #[getter]
    fn free_blocks(&self) -> PyResult<usize> {
        Ok(self.free_list.lock().unwrap().len())
    }

    #[getter]
    fn num_gpu_blocks(&self) -> usize {
        self.num_gpu_blocks
    }

    #[getter]
    fn block_size(&self) -> usize {
        self.block_size
    }
}

impl BlockManager {
    /// Rust-native 构造路径（Scheduler 用）。[[new]] pymethod 委托到这里。
    pub fn create(num_gpu_blocks: usize, block_size: usize) -> Result<Self, ZealotError> {
        if block_size == 0 {
            return Err(ZealotError::InvalidBlockSize(block_size));
        }

        let mut free_list: Vec<usize> = (0..num_gpu_blocks).collect();
        // LIFO for cache locality: pop from end gives recently-freed blocks
        free_list.reverse();

        Ok(Self {
            num_gpu_blocks,
            block_size,
            free_list: Mutex::new(free_list),
            refcount: (0..num_gpu_blocks).map(|_| AtomicUsize::new(0)).collect(),
            generation: (0..num_gpu_blocks).map(|_| AtomicU64::new(1)).collect(),
        })
    }

    /// Rust-native 分配路径（Scheduler 用）。与 #[pymethods] 版本语义相同，
    /// 但返回 ZealotError 以便直接 match（如 OutOfBlocks 触发抢占）。
    pub fn try_allocate(&self) -> Result<BlockHandle, ZealotError> {
        let mut free = self.free_list.lock().unwrap();
        let block_id = free.pop().ok_or(ZealotError::OutOfBlocks {
            requested: 1,
            available: free.len(),
        })?;
        drop(free);

        let gen = self.generation[block_id].load(Ordering::Acquire);
        self.refcount[block_id].store(1, Ordering::Release);
        Ok(BlockHandle {
            block_id,
            generation: gen,
        })
    }

    /// Rust-native 释放路径（Scheduler 用）。
    pub fn try_free(&self, handle: &BlockHandle) -> Result<(), ZealotError> {
        self.check(handle)?;

        let prev = self.refcount[handle.block_id].fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            self.generation[handle.block_id].fetch_add(1, Ordering::AcqRel);
            let mut free = self.free_list.lock().unwrap();
            free.push(handle.block_id);
        }
        Ok(())
    }

    /// Rust-native 引用路径：增加一次引用计数（共享 block 时调用）。
    /// 必须与持有者的最后一次 free 建立 happens-before 关系（调用方纪律，
    /// 见 loom_tests 模块头注释）；违反时 generation 检查给出 StaleHandle。
    pub fn try_reference(&self, handle: &BlockHandle) -> Result<(), ZealotError> {
        self.check(handle)?;
        self.refcount[handle.block_id].fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    /// 当前空闲 block 数。
    pub fn available(&self) -> usize {
        self.free_list.lock().unwrap().len()
    }

    fn check(&self, handle: &BlockHandle) -> Result<(), ZealotError> {
        if handle.block_id >= self.num_gpu_blocks {
            return Err(ZealotError::BlockNotAllocated(handle.block_id));
        }
        let current_gen = self.generation[handle.block_id].load(Ordering::Acquire);
        if handle.generation != current_gen {
            return Err(ZealotError::StaleHandle {
                block_id: handle.block_id,
                handle_gen: handle.generation,
                current_gen,
            });
        }
        Ok(())
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BlockManager>()?;
    m.add_class::<BlockHandle>()?;
    Ok(())
}

/// 常规单元测试（std 同步原语下运行）。
/// loom 模拟测试见同文件的 `loom_tests` 模块（`--cfg loom` 时启用，此处互斥）。
#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_and_free() {
        let bm = BlockManager::new(4, 16).unwrap();
        assert_eq!(bm.free_blocks().unwrap(), 4);

        let h1 = bm.allocate().unwrap();
        assert_eq!(bm.free_blocks().unwrap(), 3);

        let h2 = bm.allocate().unwrap();
        assert_eq!(bm.free_blocks().unwrap(), 2);

        bm.free(&h2).unwrap();
        assert_eq!(bm.free_blocks().unwrap(), 3);

        bm.free(&h1).unwrap();
        assert_eq!(bm.free_blocks().unwrap(), 4);
    }

    #[test]
    fn test_reference_counting() {
        let bm = BlockManager::new(2, 16).unwrap();
        let h = bm.allocate().unwrap();

        bm.reference(&h).unwrap();
        // free once — refcount goes 2→1, block stays allocated
        bm.free(&h).unwrap();
        assert_eq!(bm.free_blocks().unwrap(), 1);

        // free again — refcount goes 1→0, block returns to pool
        bm.free(&h).unwrap();
        assert_eq!(bm.free_blocks().unwrap(), 2);
    }

    #[test]
    fn test_stale_handle() {
        let bm = BlockManager::new(2, 16).unwrap();
        let h = bm.allocate().unwrap();
        bm.free(&h).unwrap();

        // handle is stale after free
        let err = bm.free(&h).unwrap_err();
        assert!(err.to_string().contains("Stale handle"));
    }

    #[test]
    fn test_out_of_blocks() {
        let bm = BlockManager::new(1, 16).unwrap();
        let _h = bm.allocate().unwrap();
        let err = bm.allocate().unwrap_err();
        assert!(err.to_string().contains("Out of blocks"));
    }

    #[test]
    fn test_invalid_block_size() {
        let err = BlockManager::new(4, 0).unwrap_err();
        assert!(err.to_string().contains("Invalid block size"));
    }

    #[test]
    fn test_reuse_after_free() {
        let bm = BlockManager::new(2, 16).unwrap();
        let h1 = bm.allocate().unwrap();
        let id1 = h1.block_id;
        bm.free(&h1).unwrap();

        let h2 = bm.allocate().unwrap();
        // Should get the same block_id back (LIFO free list)
        assert_eq!(h2.block_id, id1);
        // But generation should be different
        assert!(h2.generation > h1.generation);
    }
}

/// M4 §12-2: Block Manager 并发行为 loom 模拟测试。
///
/// loom 的 model checker 要求本 crate 的同步原语换成 `loom::sync::*`，
/// 通过 `build.rs` 在 `ZEALOT_LOOM=1` 时向**本 crate** 发射
/// `cargo:rustc-cfg=loom` 实现——不用全局 `RUSTFLAGS="--cfg loom"`，
/// 因为后者会波及 tokio 等依赖（其源码含 `#[cfg(not(loom))]` 门控，
/// 直接编译失败）。
///
/// 运行方式：
/// ```text
/// ZEALOT_LOOM=1 cargo test --lib loom -- --test-threads=1
/// ```
/// 普通 `cargo test`（无 `ZEALOT_LOOM`）不编译本模块、不影响任何常规测试；
/// 常规单测见同文件 `mod tests`（与 `cfg(loom)` 互斥）。
///
/// 覆盖目标（issue #7 验收）：
/// - free list 并发分配/释放：任意交错下无双重分配、无 block 丢失/泄漏
///   （`concurrent_alloc_free_unique_ownership`）
/// - refcount 并发：引用计数必须阻止 block 提前回池
///   （`refcount_prevents_premature_reclaim`）
/// - generation 并发：stale handle 必须被拒绝、generation 严格递增
///   （`stale_handle_rejected_after_reallocation`、`generation_monotonic_per_block`）
/// - 池耗尽路径（`exhaustion_returns_out_of_blocks`）
///
/// 调用方纪律：`reference()` 必须在持有者最后一次 `free()` 之前建立——
/// 若与最后一次 free 并发调用，refcount 可能被新分配者覆盖（泄漏）。
/// 这是 caller 的 use-after-free 违约，generation 检查使其成为可诊断的
/// `StaleHandle` 错误而非静默损坏；本模块只模拟合法纪律下的交错。
#[cfg(all(test, loom))]
mod loom_tests {
    use super::*;
    use loom::sync::atomic::AtomicBool;
    use loom::sync::Arc;
    use loom::thread;

    /// 有界探索：无界穷举下，2 线程 × 数十个原子操作的交错数是天文数字
    /// （组合爆炸），必须限制 preemption / permutation 数量——这是 loom
    /// 的标准用法（crossbeam 等同样加界）。Mutex 阻塞切换不计入
    /// preemption，因此 free list 的锁竞争交错仍被充分覆盖。
    fn run_model(f: impl Fn() + Send + Sync + 'static) {
        let mut builder = loom::model::Builder::new();
        builder.preemption_bound = Some(3);
        builder.max_permutations = Some(200_000);
        builder.max_duration = Some(std::time::Duration::from_secs(180));
        builder.check(f);
    }

    /// 并发分配/释放 + 独占所有权检查。
    ///
    /// `in_use[id]` 标记 block 是否被某线程独占持有：分配成功时 `swap(true)`
    /// 必须读到 `false`（否则两个线程同时持有同一 block）；释放前先清除标记。
    /// 结束时所有 block 必须回到 free list（无泄漏/无重复入池）。
    #[test]
    fn concurrent_alloc_free_unique_ownership() {
        run_model(|| {
            const BLOCKS: usize = 2;
            const THREADS: usize = 2;
            const ITERS: usize = 3;

            let bm = Arc::new(BlockManager::create(BLOCKS, 16).unwrap());
            let in_use: Arc<Vec<AtomicBool>> = Arc::new(
                (0..BLOCKS).map(|_| AtomicBool::new(false)).collect(),
            );

            let threads: Vec<_> = (0..THREADS)
                .map(|_| {
                    let bm = bm.clone();
                    let in_use = in_use.clone();
                    thread::spawn(move || {
                        for _ in 0..ITERS {
                            let h = match bm.try_allocate() {
                                Ok(h) => h,
                                Err(_) => continue, // 池被其他线程耗尽
                            };
                            let prev = in_use[h.block_id].swap(true, Ordering::AcqRel);
                            assert!(
                                !prev,
                                "block {} 被两个线程同时持有（双重分配）",
                                h.block_id
                            );
                            // ... 模拟使用 block ...
                            // 先清标记再 free：free 后 block 可被他人分配，
                            // 标记须在此之前清除，否则新持有者的 swap 会误报。
                            in_use[h.block_id].store(false, Ordering::Release);
                            bm.try_free(&h).unwrap();
                        }
                    })
                })
                .collect();

            for t in threads {
                t.join().unwrap();
            }

            // 无泄漏、无双重入池：所有 block 必须回到 free list。
            assert_eq!(bm.available(), BLOCKS);
            for (id, flag) in in_use.iter().enumerate() {
                assert!(!flag.load(Ordering::Acquire), "block {} 仍被标记持有", id);
            }
        });
    }

    /// 引用计数语义：`reference()` 建立的额外引用必须阻止 block 回池。
    ///
    /// 用户线程：分配 → reference（refcount 1→2）→ free（2→1，不得回池）
    /// → free（1→0，回池）。写线程并发分配/释放其他 block。
    /// 任何交错下：带引用的 block 绝不能被其他线程分配（flag 检查），
    /// 且最终无泄漏。
    #[test]
    fn refcount_prevents_premature_reclaim() {
        run_model(|| {
            const BLOCKS: usize = 2;

            let bm = Arc::new(BlockManager::create(BLOCKS, 16).unwrap());
            let in_use: Arc<Vec<AtomicBool>> = Arc::new(
                (0..BLOCKS).map(|_| AtomicBool::new(false)).collect(),
            );

            let user = {
                let bm = bm.clone();
                let in_use = in_use.clone();
                thread::spawn(move || {
                    for _ in 0..3 {
                        let h = match bm.try_allocate() {
                            Ok(h) => h,
                            Err(_) => continue,
                        };
                        let prev = in_use[h.block_id].swap(true, Ordering::AcqRel);
                        assert!(!prev, "block {} 双重分配", h.block_id);

                        bm.reference(&h).unwrap(); // 1→2
                        bm.try_free(&h).unwrap(); // 2→1：不得回池

                        in_use[h.block_id].store(false, Ordering::Release);
                        bm.try_free(&h).unwrap(); // 1→0：回池
                    }
                })
            };
            let writer = {
                let bm = bm.clone();
                let in_use = in_use.clone();
                thread::spawn(move || {
                    for _ in 0..3 {
                        let h = match bm.try_allocate() {
                            Ok(h) => h,
                            Err(_) => continue,
                        };
                        let prev = in_use[h.block_id].swap(true, Ordering::AcqRel);
                        assert!(!prev, "block {} 双重分配", h.block_id);
                        in_use[h.block_id].store(false, Ordering::Release);
                        bm.try_free(&h).unwrap();
                    }
                })
            };

            user.join().unwrap();
            writer.join().unwrap();

            // 若引用泄漏（refcount 未归零）或重复入池，此断言失败。
            assert_eq!(bm.available(), BLOCKS);
            for (id, flag) in in_use.iter().enumerate() {
                assert!(!flag.load(Ordering::Acquire), "block {} 仍被标记持有", id);
            }
        });
    }

    /// generation 防御：block 回池后 generation 递增，旧 handle 的 free
    /// 必须返回 `StaleHandle`，无论与重新分配如何交错。
    #[test]
    fn stale_handle_rejected_after_reallocation() {
        run_model(|| {
            let bm = Arc::new(BlockManager::create(1, 16).unwrap());

            // 线程 A：分配 → 释放 → 对同一（已过期）handle 再释放。
            // 单 block 池中 A 可能抢不到（B 先分配），此时无 stale 场景可验证。
            let a = {
                let bm = bm.clone();
                thread::spawn(move || {
                    let h = match bm.try_allocate() {
                        Ok(h) => h,
                        Err(_) => return None,
                    };
                    bm.try_free(&h).unwrap();
                    Some(bm.try_free(&h)) // 必须失败：generation 已递增
                })
            };
            // 线程 B：竞争同一 block（可能拿到也可能拿不到）。
            let b = {
                let bm = bm.clone();
                thread::spawn(move || match bm.try_allocate() {
                    Ok(h) => bm.try_free(&h).unwrap(),
                    Err(_) => {}
                })
            };

            let stale_result = a.join().unwrap();
            b.join().unwrap();

            if let Some(result) = stale_result {
                assert!(
                    result.is_err(),
                    "过期 handle 必须返回 StaleHandle"
                );
            }
            assert_eq!(bm.available(), 1);
        });
    }

    /// generation 单调性：每个 block 每被重新分配一次，generation 恰好 +1。
    ///
    /// 断言（与交错顺序无关）：
    /// 1. 每个线程内，同一 block 的 generation 序列严格递增；
    /// 2. 全局：`generation[id] == 1 + 该 block 的总分配次数`
    ///    （每次分配必有一次归零 free，恰好一次 bump）；
    /// 3. 无泄漏：全部 block 回到 free list。
    #[test]
    fn generation_monotonic_per_block() {
        run_model(|| {
            const BLOCKS: usize = 2;
            const THREADS: usize = 2;
            const ITERS: usize = 3;

            let bm = Arc::new(BlockManager::create(BLOCKS, 16).unwrap());

            // 每线程本地记录 (block_id, generation)，避免共享日志引入额外同步。
            let threads: Vec<_> = (0..THREADS)
                .map(|_| {
                    let bm = bm.clone();
                    thread::spawn(move || {
                        let mut log: Vec<(usize, u64)> = Vec::new();
                        for _ in 0..ITERS {
                            if let Ok(h) = bm.try_allocate() {
                                log.push((h.block_id, h.generation));
                                bm.try_free(&h).unwrap();
                            }
                        }
                        log
                    })
                })
                .collect();

            // 先收集每线程日志，再做与交错顺序无关的断言。
            let logs: Vec<Vec<(usize, u64)>> =
                threads.into_iter().map(|t| t.join().unwrap()).collect();

            // 1. 每线程内，同一 block 的 generation 严格递增。
            for log in &logs {
                let mut prev: Vec<u64> = vec![0; BLOCKS];
                for &(block_id, gen) in log {
                    assert!(
                        gen > prev[block_id],
                        "block {block_id} generation 未严格递增: {gen} after {}",
                        prev[block_id]
                    );
                    prev[block_id] = gen;
                }
            }

            // 2. 全局：generation[id] == 1 + 该 block 的总分配次数。
            //    （每次分配必有一次归零 free，恰好 bump 一次。）
            let mut total_alloc: Vec<usize> = vec![0; BLOCKS];
            for log in &logs {
                for &(block_id, _) in log {
                    total_alloc[block_id] += 1;
                }
            }
            for block_id in 0..BLOCKS {
                let gen_now = bm.generation[block_id].load(Ordering::Acquire);
                assert_eq!(
                    gen_now as usize,
                    1 + total_alloc[block_id],
                    "block {block_id} generation 与分配次数不一致"
                );
            }

            // 3. 无泄漏。
            assert_eq!(bm.available(), BLOCKS);
        });
    }

    /// 池耗尽路径：并发竞争最后一个 block 时，失败的线程必须收到
    /// `OutOfBlocks`（而非 panic / 双重分配），随后 block 完整归还。
    #[test]
    fn exhaustion_returns_out_of_blocks() {
        run_model(|| {
            let bm = Arc::new(BlockManager::create(1, 16).unwrap());
            let threads: Vec<_> = (0..2)
                .map(|_| {
                    let bm = bm.clone();
                    thread::spawn(move || {
                        for _ in 0..3 {
                            match bm.try_allocate() {
                                Ok(h) => bm.try_free(&h).unwrap(),
                                Err(ZealotError::OutOfBlocks { .. }) => {}
                                Err(other) => panic!("unexpected error: {other}"),
                            }
                        }
                    })
                })
                .collect();

            for t in threads {
                t.join().unwrap();
            }
            assert_eq!(bm.available(), 1);
        });
    }
}

