# Zealot — Ultralisk 推理引擎

> 架构设计文档：[docs/architecture.md](docs/architecture.md)

## 定位

Zealot 是 Ultralisk 的自研推理引擎，目标是在 Phase 3 替代 vLLM 成为默认推理引擎，性能达到 Together TIE 的 80%+。

**策略**：独立引擎，不从 vLLM fork 起步。Zealot 以 Rust 主进程运行（tonic gRPC），Python 仅通过 PyO3 嵌入用于 HuggingFace Model Loader。关键路径（Block Manager → Constrained Decode → Scheduler）直接以 Rust 原生实现，无"先注入 vLLM 再脱离"的过渡阶段。

**相关 ADR**：[003](../docs/adr/003-inference-engine-vllm.md)、[009](../docs/adr/009-zealot-language-strategy.md)、[010](../docs/adr/010-backend-runtime.md)

## 语言分工

```
Rust   ──→ 主进程：gRPC Server (tonic) + Scheduler
           + Block Manager + Constrained Decode
Python ──→ Model Loader                 ← PyO3 嵌入，仅启动时加载权重（HF 生态）
CUDA   ──→ Attention / Quant Kernel     ← 改装优化，不重写
```

## 项目结构

```
zealot/
├── Cargo.toml               # lib (cdylib → zealot_engine.abi3.so) + bin (zealot-backend)
├── build.rs                  # tonic-build 编译 ../proto Runtime Interface（protoc vendored）
├── pyproject.toml            # maturin build config
├── README.md
├── src/
│   ├── lib.rs                # PyO3 #[pymodule] 入口
│   ├── error.rs              # 错误类型 + PyErr 转换
│   ├── block_manager.rs      # P1: KV Cache 分页管理
│   ├── constrained_decode/   # P1: 约束解码引擎
│   │   ├── mod.rs
│   │   ├── schema.rs         # JSON schema → DFA 编译
│   │   └── matcher.rs        # 约束匹配（CPU 侧 token 校验）
│   ├── scheduler.rs          # 调度器：优先级队列 + block 预算 + OOM 抢占（recompute）
│   ├── engine.rs             # Engine 步进循环 + ModelRunner trait
│   ├── model_runner_py.rs    # PyModelRunner：PyO3 嵌入 torch CPU 前向（dev-mode）
│   └── bin/
│       └── zealot-backend.rs # Runtime Interface gRPC server（ADR-010，Engine actor，:9091）
└── tests/                    # backend_e2e.rs + cpu_infer_e2e.rs（真实模型 CPU 端到端）+ pytest
```

## 模块设计

### Block Manager（P1，Phase 2 M4）

PagedAttention 内存管理的 Rust 实现。vLLM 的 Python 原版用手动引用计数，存在跨请求 use-after-free 风险。Zealot 用 generation-gated `BlockHandle` 防御：对已释放 handle 的操作返回 `Err(StaleHandle)`，逻辑错误变成可诊断错误而非静默数据损坏。

**Python API**：
```python
import zealot_engine
bm = zealot_engine.BlockManager(num_gpu_blocks=1024, block_size=16)
block_id = bm.allocate()       # → int
bm.reference(block_id)          # refcount++
bm.free(block_id)               # refcount--, 归零时回收
bm.free_blocks                  # → int
```

**状态**：Phase 2 M4（GPU-free 可开发，分页算法用常规堆内存测试）

### Constrained Decode Engine（P1，Phase 2 M4-M5）

对标 SGLang xgrammar。vLLM 用 Python `outlines` 库做 JSON schema 约束，有 GIL 开销。Rust 版在 CPU 侧编译 schema → DFA，每次采样时校验 token。

**Python API**：
```python
compiler = zealot_engine.JsonSchemaCompiler()
grammar = compiler.compile('{"type":"object","properties":{...}}')
grammar.allowed_tokens(state)  # → List[int]
grammar.advance(state, token)  # → int (next state)
```

**状态**：CPU 侧开发 GPU-free，CUDA 侧采样集成需等 GPU 到位

### Scheduler（已实现，dev-mode 验证）

替换 vLLM 的 Python 调度器，消除 GC tail latency。已实现：优先级等待队列、block 预算（经 BlockManager `try_allocate`）、准入控制（`SequenceTooLong`）、OOM 抢占（preempt by recomputation——驱逐最低优先级 seq、回收 block、重入队）。Engine 步进循环通过 `ModelRunner` trait 驱动，与具体 runner 实现解耦。

**状态**：Rust core 完成，dev-mode CPU 端到端验证通过。Prefill-Decode 分离、Speculative Decoding 属 Phase 3。

## Phase 2 → Phase 3 演进

```
Phase 2（M4-M6，当前）
  Zealot Backend（独立 Rust 进程，tonic gRPC）
      ├── Rust: Block Manager, Constrained Decode, Scheduler
      │         （Scheduler 逻辑等价 vLLM continuous batching）
      ├── CUDA: Attention / Quant Kernel
      └── Python (PyO3 嵌入): Model Loader  ← 仅启动时；decode loop 不涉 Python

  ⚠ dev-mode 妥协（GPU 到位前）：PyModelRunner 用 torch CPU 前向跑真实推理，
    Python 在 decode loop 内。这是临时形态，仅用于无 GPU 下验证
    Scheduler/Engine/协议链路；GPU kernel 就位后回归"仅启动时加载"设计。

Phase 3（M7-M12，目标）
  Zealot 成为默认引擎
      ├── Scheduler 增强：Prefill-Decode 分离、Speculative Decoding、RadixAttention
      └── 性能目标：Together TIE 的 80%+
```

## 构建

> **前置**：pyo3 0.23 最高支持 Python 3.13，构建需要 `python3.12` 在 PATH 中
>（`.cargo/config.toml` 已固定 `PYO3_PYTHON=python3.12`）。

```bash
# 开发
cargo build

# gRPC server（Runtime Interface，ADR-010；Engine actor 模式）
cargo run --bin zealot-backend    # 端口 ZEALOT_GRPC_PORT，默认 :9091

# 生成 Python 扩展
maturin develop         # 安装到当前 venv
pip install .           # 或安装 wheel

# 单元测试 + gRPC 端到端测试（tests/backend_e2e.rs 会拉起真实 server）
cargo test

# Python 集成测试
pytest tests/
```

### CPU dev 环境（无 GPU 跑真实推理）

```bash
# venv（必须 python3.12；系统 python3.14 不被 pyo3 0.23 支持）
python3.12 -m venv .venv
.venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  "torch==2.2.2" "transformers==4.46.3" "numpy==1.26.4"
# pin 说明：torch 2.2.2 是 x86_64 macOS 最后有 wheel 的版本；
# transformers ≥4.47 要求 torch≥2.4；torch 2.2 不兼容 numpy 2.x

# CPU 端到端真实推理测试（未设置 ZEALOT_E2E_MODEL 则自动跳过）
ZEALOT_E2E_MODEL=hf-internal-testing/tiny-random-gpt2 \
ZEALOT_SITE_PACKAGES="$PWD/.venv/lib/python3.12/site-packages" \
HF_ENDPOINT=https://hf-mirror.com \
cargo test --test cpu_infer_e2e -- --nocapture
```

环境变量：`ZEALOT_SITE_PACKAGES` 把 venv 的 site-packages 注入嵌入 CPython；`ZEALOT_E2E_MODEL` 指定 e2e 测试模型；`HF_ENDPOINT` 在 huggingface.co 不可达时指向镜像。

## 测试策略

- **Rust 单元测试**：每个 crate 内的 `#[cfg(test)]` 模块，覆盖纯逻辑
- **Python 集成测试**：pytest + `import zealot_engine`，验证 PyO3 接口
- **GPU 集成测试**（Phase 2 后期）：真实 GPU 环境，验证 Block Manager ↔ CUDA 交互
- **Property-based testing**：Block Manager 分配/释放序列的随机测试（proptest）

## 新模型支持

Zealot 不 fork vLLM，无 upstream rebase 负担。新模型支持依赖 HuggingFace transformers：新模型在 transformers 发布后，升级 Python 侧依赖即可获得，Rust 核心无需改动。目标适配速度不慢于 vLLM 社区的 1-2 周（ADR-009 约束）。
