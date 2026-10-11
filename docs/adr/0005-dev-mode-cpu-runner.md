# ADR-0005: dev-mode CPU runner（GPU 到位前的临时形态）

- **状态**: 已接受（临时，GPU kernel 就位后废止）
- **日期**: 2026-08（M4 全链路实测）
- **相关**: 架构文档 §5.1；`docs/phase2_lifecycle.md`；issue #11/#26

## Context

目标设计是「Python 仅启动时加载权重，decode loop 不涉 Python」（ADR-0001）。但 Phase 2 开发环境无 GPU，CUDA runner 无法运行，Scheduler/Engine/Runtime Interface 全链路仍需要真实推理来验证。

## Decision

GPU 到位前引入 `PyModelRunner`（`src/model_runner_py.rs`）：PyO3 嵌入 torch CPU 前向跑真实推理，Python 在 decode loop 内、每 step 穿越 FFI。这是 dev-mode 妥协，仅用于无 GPU 下验证 Scheduler/Engine/协议链路。Model loader 支持 LLaMA 系（llama/mistral）+ GPT-2 架构，CPU runner 支持 GQA（kv 头展开）。

## Consequences

- 全链路（构建 → 启动 → 模型加载 → gRPC/HTTP → 流式返回 → 关闭）在 GPU-free 环境可复现、可回归（`tests/cpu_infer_e2e.rs`）。
- CPU runner 依赖 pin：torch 2.2.2 / transformers 4.46.3 / numpy 1.26.4（pyo3 0.23 要求 Python ≤3.13，pin 理由见 README）。
- GPU kernel 就位后 runner 换为 Rust+CUDA 实现，Python 回归「仅启动时加载」；届时本 ADR 标记为 Superseded。
