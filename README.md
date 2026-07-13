# Zealot — Ultralisk 推理引擎

> 架构设计文档：[docs/architecture.md](docs/architecture.md)

## 定位

Zealot 是 Ultralisk 的自研推理引擎，目标是在 Phase 3 替代 vLLM 成为默认推理引擎，性能达到 Together TIE 的 80%+。

**策略**：组件级替换，非全量重写。从 vLLM fork 起步，逐步用 Rust 替换关键路径（Block Manager → Constrained Decode → Scheduler），最终脱离 fork 成为独立引擎。

**相关 ADR**：[003](../docs/adr/003-inference-engine-vllm.md)、[009](../docs/adr/009-zealot-language-strategy.md)、[010](../docs/adr/010-backend-runtime.md)

## 语言分工

```
Python ──→ Model Loader + API Server    ← 永久保留（HF 生态 + OpenAI 兼容）
Rust   ──→ Block Manager + Constrained  ← 逐步替换
           Decode + Scheduler
CUDA   ──→ Attention Kernel + Quant     ← 改装优化，不重写
```

## 项目结构

```
zealot/
├── Cargo.toml               # 单 crate: cdylib → zealot_engine.abi3.so
├── pyproject.toml            # maturin build config
├── README.md
├── src/
│   ├── lib.rs                # PyO3 #[pymodule] 入口
│   ├── error.rs              # 错误类型 + PyErr 转换
│   ├── block_manager.rs      # P1: KV Cache 分页管理
│   └── constrained_decode/   # P1: 约束解码引擎
│       ├── mod.rs
│       ├── schema.rs         # JSON schema → DFA 编译
│       └── matcher.rs        # 约束匹配（CPU 侧 token 校验）
└── tests/                    # pytest 集成测试
```

## 模块设计

### Block Manager（P1，Phase 2 M4）

替换 vLLM 的 PagedAttention 内存管理。Python 原版用手动引用计数，存在跨请求 use-after-free 风险。Rust 版用 ownership 在编译期保证安全。

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

### Scheduler（P2，Phase 3）

替换 vLLM 的 Python 调度器，消除 GC tail latency。支持 Prefill-Decode 分离、优先级调度。

**状态**：Phase 3（不在当前范围）

## Phase 2 → Phase 3 演进

```
Phase 2（当前）
  vLLM fork
      ├── Python: API Server, Model Loader, Scheduler
      └── Rust (.so): Block Manager, Constrained Decode  ← import 注入

Phase 3（目标）
  Zealot Backend (独立进程, tonic gRPC)
      ├── Python: Model Loader, API Server  ← 脱离 fork，纳入 Zealot 仓库
      └── Rust (.so): Block Manager, Scheduler, Constrained Decode
```

## 构建

```bash
# 开发
cargo build

# 生成 Python 扩展
maturin develop         # 安装到当前 venv
pip install .           # 或安装 wheel

# 单元测试
cargo test

# Python 集成测试
pytest tests/
```

## 测试策略

- **Rust 单元测试**：每个 crate 内的 `#[cfg(test)]` 模块，覆盖纯逻辑
- **Python 集成测试**：pytest + `import zealot_engine`，验证 PyO3 接口
- **GPU 集成测试**（Phase 2 后期）：真实 GPU 环境，验证 Block Manager ↔ CUDA 交互
- **Property-based testing**：Block Manager 分配/释放序列的随机测试（proptest）

## 与 vLLM 的冲突处理

由于 Zealot 组件替换了 vLLM 的核心逻辑，rebase vLLM upstream 时：
- Model Loader（Python，不动）→ 零冲突
- Block Manager / Scheduler（Rust 替换）→ 零冲突（原 Python 文件已删除）
- Attention kernel（CUDA 改装）→ 手动解决
