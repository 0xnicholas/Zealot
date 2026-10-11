# ADR-0002: Generation-gated BlockHandle 防御跨 FFI 误用

- **状态**: 已接受（M4 落地）
- **日期**: 2026-07-13
- **相关**: 架构文档 §6.1；issue #4/#7/#10

## Context

Rust 编译器保证 Rust 代码内部无悬垂指针/数据竞争，但无法阻止跨 FFI 边界后 Python 侧的逻辑错误：对已释放 handle 重复 `free()`、忘记 `reference()` 导致提前回收、seq 取消后 handle 传给错误的 seq。vLLM Python 原版用手动引用计数，此类错误是静默数据损坏。

## Decision

`allocate()` 不返回裸 `usize`，而是带 generation 的 typed handle `BlockHandle { block_id, generation }`。每 block 维护 `AtomicU64` 版本号，allocate/free 时递增；对过期 handle 的操作在运行时返回 `Err(StaleHandle)`。

## Consequences

- 逻辑错误变成可诊断错误，而非 UB 或跨请求数据损坏。
- 开销：每 block 额外 8 字节，百万级 block 池约 +8MB，可接受。
- 不变量由 proptest（模型对照法）与 loom（并发模拟）保障：无重复分配、stale 拒绝、refcount 归零才回池、generation 递增。
