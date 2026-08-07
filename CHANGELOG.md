# Changelog

All notable changes to this project are documented in this file.

Format: 语义化版本 + 变更类型（Added / Changed / Fixed / Removed）。
发布 tag 命名：`zealot-vX.Y.Z`（与平台 tag 并行，互不干扰）。

## [Unreleased]

- M4 收官：§8.1 FFI 微基准 8 项落地（issue #8）——criterion benches/
  BlockManager 6 项全部达标（allocate 36.8ns / free 38.4ns / reference
  10.1ns / stale 4.4ns / GIL 61ns），decode 2 项超阈值标风险
  （详见 architecture.md §8.1）；§12-3 allocate<500ns 已勾选
- M4 收官：Block Manager `loom` 并发模拟测试（issue #7）——free list
  分配/释放、refcount、generation 并发交错全覆盖，mutation 验证有效；
  运行 `ZEALOT_LOOM=1 cargo test --lib loom -- --test-threads=1`
- 项目独立化：Zealot 从 Ultralisk monorepo 拆分为独立闭源仓库
  （历史完整保留，48 commits + Runtime Interface proto 2 commits）
- Runtime Interface proto 契约随本仓库独立维护（`proto/runtime/v1/runtime.proto`）

## [0.1.0] — 初始开发版本

- Block Manager（generation-gated handles，防御跨请求 use-after-free）
- Constrained Decode（JSON schema → DFA 编译 + CPU 侧 token 校验）
- Scheduler（优先级队列、block 预算、OOM 抢占 recompute、chunked prefill）
- Engine 步进循环 + ModelRunner trait（CPU / PyO3 / CUDA 三种 runner）
- CUDA：tiled FlashAttention、PagedAttention（GQA）、RMS Norm、embedding、
  transpose、SiLU gate-mul + element-wise add kernels
- KV Cache（per-layer GPU 内存池 + block table）、BufferPool
- Sampler（temperature、top-k/p、repetition/frequency/presence penalties）
- zealot-backend：Runtime Interface gRPC server（:9091）+ HTTP OpenAI endpoint
- dev-mode CPU 端到端真实推理（PyO3 嵌入 torch）
