# Changelog

All notable changes to this project are documented in this file.

Format: 语义化版本 + 变更类型（Added / Changed / Fixed / Removed）。
发布 tag 命名：`zealot-vX.Y.Z`（与平台 tag 并行，互不干扰）。

## [Unreleased]

- M4 收官：schema 缓存真 LRU（issue #12，§12-5）——原 HashMap 随机驱逐
  （`keys().next()`）修正为 `lru` 0.18 `LruCache`（O(1) 命中提升/淘汰，
  cache_size 上限）；`cache_size=0` 明确为禁用缓存；新增命中/未命中/淘汰
  统计 getter；Rust 单测 + pytest 双面覆盖淘汰顺序（A,B,A,C → 必逐 B）
- M4 收官：CI 闸门（issue #13）——clippy `-D warnings` 转硬闸（移除
  continue-on-error，21 条既有告警清零 + 全仓 rustfmt 归一）；criterion
  bench job（§8.1 的 8 项每次 CI 跑，HTML 报告 artifact，阈值不达标不挂 CI）；
  pytest job（Python 3.12 + maturin build + wheel 安装 + pytest tests/）
- M4 关闭（issue #4 判定式全部达成）——§12 八项 + proptest 全勾选；
  architecture.md 双入口正式形态（gRPC Runtime Interface + HTTP OpenAI
  兼容面，不再写"在 Gateway 完成"）；README Block Manager / CD 状态同步
- M4 收官：docs/phase2_lifecycle.md（issue #11，§12-8）——构建 → 启动 →
  模型加载 → gRPC/HTTP 请求 → 流式返回 → 关闭全链路文档 + 实测通过
  （本地 tiny LLaMA，16 token 生成，SSE 6 帧）；发现并跟进 #26
  （model_loader 仅 LLaMA 系 / CPU runner 无 GQA）；README/测试
  的 tiny-random-gpt2 引用修正为 LLaMA 系
- M4 收官：proptest BlockManager 随机序列测试（issue #10）——模型对照法
  （free list 栈 + refcount/generation），逐操作断言无重复分配 / stale 拒绝 /
  refcount 归零才回池 / generation 递增 + 引用守恒；mutation 验证有效
- M4 收官：pytest 集成测试面落地（issue #9）——test_block_manager.py
  （接口一致性/错误转换/StaleHandle/池语义）+ test_constrained_decode.py
  （编译/缓存/边界/嵌套深度/对抗性 10001 属性 schema 拒绝 §12-6/超时守卫），
  22 用例全绿；README 修正 pytest 声明与过期 Python API 文档
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
