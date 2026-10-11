# ADR-0001: 不从 vLLM fork，独立 Rust 主进程

- **状态**: 已接受
- **日期**: 2026-07-13（随项目独立化确立）
- **上游依据**: Ultralisk ADR-013（zealot-standalone-project）、ADR-009（语言栈）

## Context

Ultralisk 平台需要自研推理引擎替代 vLLM。可选路径：fork vLLM 渐进改造，或从零构建独立引擎。vLLM 的 Python 调度器存在 GIL 开销与 GC tail latency；Python 版 Block Manager 用手动引用计数，存在跨请求 use-after-free 风险。

## Decision

Zealot 以独立 Rust 主进程运行（tonic gRPC + HTTP OpenAI endpoint），关键路径（Block Manager → Constrained Decode → Scheduler）直接以 Rust 原生实现。Python 仅通过 PyO3 嵌入用于 HuggingFace Model Loader，且仅在启动时加载权重。不存在「先注入 vLLM 再脱离」的过渡阶段。

## Consequences

- 无 upstream rebase 负担；新模型支持通过升级 HF transformers 依赖获得，Rust 核心无需改动。
- kill criterion 兜底：里程碑不达标则平台回退 vLLM，本项目冻结为研究项目，不绑架平台。
- GPU 到位前以 dev-mode（ADR-0005）验证全链路。
