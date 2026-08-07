//! M4 §8.1 FFI 微基准：Block Manager 单次操作延迟 + GIL 持有时间（issue #8）。
//!
//! 运行：`cargo bench --bench block_manager`
//!
//! 验收阈值（docs/architecture.md §8.1）：
//! - bm_allocate_1    < 500ns   单次 allocate（池有空闲）
//! - bm_allocate_full < 200ns   池满时 allocate 的 Err 返回
//! - bm_free_last_ref < 200ns   单次 free（refcount 归零，回池）
//! - bm_reference     < 100ns   单次 reference
//! - bm_stale_handle  < 200ns   stale handle 的 free 拒绝路径
//! - gil_hold_bench   < 1μs/op  GIL 持有时间（PyO3 pymethod 包装层，往返）
//!
//! 说明：allocate/free 的纯路径测量用 `iter_custom` 分批——setup 阶段
//! （不计时）把池态补回满，计时段连续执行 BATCH 次目标操作。criterion 的
//! `iter_batched` 的 setup:routine 不是 1:1（SmallInput 批 = iters/10），
//! 无法维持池的有界振荡。

use std::cell::RefCell;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, Criterion};
use zealot_engine::block_manager::{BlockHandle, BlockManager};

const BLOCK_SIZE: usize = 16;
/// 池容量：远大于单批 BATCH，保证计时段内池不会耗尽。
const POOL: usize = 4096;
/// 每批（setup+计时循环）内的目标操作次数。Instant 计时开销（~20ns）
/// 摊到 BATCH 次操作上可忽略。iter_custom 的 iters 参数是 u64。
const BATCH: u64 = 256;

/// 纯 allocate（池有空闲）：setup 把上一批持有的 handle 全部归还
/// （池补满，不计时），计时段连续 allocate。
/// iter_custom 契约：闭包须完成恰好 `iters` 次目标操作并返回总时长——
/// 内部按 BATCH 分组摊销 Instant 计时开销。
fn bench_bm_allocate_1(c: &mut Criterion) {
    let bm = BlockManager::create(POOL, BLOCK_SIZE).unwrap();
    let held = RefCell::new(Vec::<BlockHandle>::with_capacity(BATCH as usize));

    c.bench_function("bm_allocate_1", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            let mut remaining = iters;
            while remaining >= BATCH {
                // setup（不计时）：归还上一批 handle，池回到全满
                let hs = std::mem::take(&mut *held.borrow_mut());
                for h in hs {
                    bm.try_free(&h).unwrap();
                }
                // 计时段：BATCH 次纯 allocate
                let t0 = Instant::now();
                for _ in 0..BATCH {
                    let h = bm.try_allocate().unwrap();
                    std::hint::black_box(h.block_id);
                    held.borrow_mut().push(h);
                }
                total += t0.elapsed();
                remaining -= BATCH;
            }
            // 残余不足一批：直接计
            if remaining > 0 {
                let hs = std::mem::take(&mut *held.borrow_mut());
                for h in hs {
                    bm.try_free(&h).unwrap();
                }
                let t0 = Instant::now();
                for _ in 0..remaining {
                    let h = bm.try_allocate().unwrap();
                    std::hint::black_box(h.block_id);
                    held.borrow_mut().push(h);
                }
                total += t0.elapsed();
            }
            total
        });
    });
}

/// 池满时的 allocate 错误返回：全部 block 预先分配出去，计时段只走
/// pop→None→OutOfBlocks 路径（无状态变化）。
fn bench_bm_allocate_full(c: &mut Criterion) {
    let bm = BlockManager::create(POOL, BLOCK_SIZE).unwrap();
    let all: Vec<BlockHandle> = (0..POOL).map(|_| bm.try_allocate().unwrap()).collect();

    c.bench_function("bm_allocate_full", |b| {
        b.iter(|| {
            let r = bm.try_allocate();
            std::hint::black_box(r.is_err());
        });
    });
    std::mem::drop(all);
}

/// 纯 free（refcount 1→0，含 generation bump + 回池 push）：setup 分配
/// 一批 handle（不计时），计时段连续 free。iter_custom 契约同上。
fn bench_bm_free_last_ref(c: &mut Criterion) {
    let bm = BlockManager::create(POOL, BLOCK_SIZE).unwrap();
    let held = RefCell::new(Vec::<BlockHandle>::with_capacity(BATCH as usize));

    c.bench_function("bm_free_last_ref", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            let mut remaining = iters;
            while remaining >= BATCH {
                // setup（不计时）：分配一批 handle，供计时段释放
                for _ in 0..BATCH {
                    let h = bm.try_allocate().unwrap();
                    held.borrow_mut().push(h);
                }
                // 计时段：BATCH 次纯 free（最后引用归零）
                let t0 = Instant::now();
                for _ in 0..BATCH {
                    let h = held.borrow_mut().pop().unwrap();
                    bm.try_free(&h).unwrap();
                    std::hint::black_box(h.block_id);
                }
                total += t0.elapsed();
                remaining -= BATCH;
            }
            if remaining > 0 {
                for _ in 0..remaining {
                    let h = bm.try_allocate().unwrap();
                    held.borrow_mut().push(h);
                }
                let t0 = Instant::now();
                for _ in 0..remaining {
                    let h = held.borrow_mut().pop().unwrap();
                    bm.try_free(&h).unwrap();
                    std::hint::black_box(h.block_id);
                }
                total += t0.elapsed();
            }
            total
        });
    });
}

/// 单次 reference：refcount fetch_add，无状态收敛问题（refcount 单调增长，
/// block 始终不回池，handle 保持有效）。
fn bench_bm_reference(c: &mut Criterion) {
    let bm = BlockManager::create(POOL, BLOCK_SIZE).unwrap();
    let h = bm.try_allocate().unwrap();

    c.bench_function("bm_reference", |b| {
        b.iter(|| {
            bm.try_reference(&h).unwrap();
            std::hint::black_box(&h);
        });
    });
}

/// stale handle 拒绝路径：allocate→free 后 handle 过期，计时段走
/// generation 检查 → StaleHandle（无状态变化）。
fn bench_bm_stale_handle(c: &mut Criterion) {
    let bm = BlockManager::create(POOL, BLOCK_SIZE).unwrap();
    let h = bm.try_allocate().unwrap();
    bm.try_free(&h).unwrap();

    c.bench_function("bm_stale_handle", |b| {
        b.iter(|| {
            let r = bm.try_free(&h);
            debug_assert!(r.is_err(), "stale handle must be rejected");
            std::hint::black_box(r.is_err());
        });
    });
}

/// GIL 持有时间：持有 GIL 时走 PyO3 pymethod 包装层（allocate/free 往返），
/// 测 FFI 面每次操作占用 GIL 的时长。阈值 < 1μs/op——超出则 §11 风险
/// 「PyO3 FFI 延迟过高」需启用 `allow_threads`。
fn bench_gil_hold_bench(c: &mut Criterion) {
    pyo3::prepare_freethreaded_python();

    c.bench_function("gil_hold_bench", |b| {
        pyo3::Python::with_gil(|_py| {
            let bm = BlockManager::new(POOL, BLOCK_SIZE).unwrap();
            b.iter(|| {
                let h = bm.allocate().unwrap();
                std::hint::black_box(h.block_id);
                bm.free(&h).unwrap();
            });
        });
    });
}

criterion_group!(
    benches,
    bench_bm_allocate_1,
    bench_bm_allocate_full,
    bench_bm_free_last_ref,
    bench_bm_reference,
    bench_bm_stale_handle,
    bench_gil_hold_bench
);
criterion_main!(benches);
