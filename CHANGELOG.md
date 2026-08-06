# Changelog

All notable changes to this project are documented in this file.

Format: 语义化版本 + 变更类型（Added / Changed / Fixed / Removed）。
发布 tag 命名：`zealot-vX.Y.Z`（与平台 tag 并行，互不干扰）。

## [Unreleased]

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
