# Zealot 推理引擎架构设计

**日期**: 2026-07-13 · **上次修订**: 2026-07-13 (架构评审)
**依赖**: ADR-003（vLLM 基线）、ADR-009（语言栈）、ADR-010（Backend Runtime）

---

## 1. 概述

Zealot 是 Ultralisk 的自研推理引擎，Phase 2 通过 A/B 灰度逐步替代 vLLM，Phase 3 成为默认引擎，性能目标达到 Together TIE 的 80%+。

**策略**：Zealot 以独立 Rust 进程运行，Python 仅作为嵌入解释器用于 HuggingFace Model Loader。不从 vLLM fork 起步，不经过胶水层注入阶段。

```
Phase 2（独立引擎）                Phase 3（默认引擎）
Zealot Backend ──────────────────────────────────────── Zealot Backend  
  │                                                       │
  ├─ tonic gRPC Server                                    ├─ tonic gRPC Server
  ├─ Rust Block Manager                                   ├─ Rust Block Manager
  ├─ Rust Constrained Decode                              ├─ Rust Constrained Decode
  ├─ Rust Scheduler                                       ├─ Rust Scheduler
  ├─ CUDA Attention/Quant Kernel                          ├─ CUDA Attention/Quant Kernel
  └─ PyO3 → Python Model Loader                           └─ PyO3 → Python Model Loader
                                                          (性能优化)
```

---

## 2. 内部分层架构

```
┌─────────────────────────────────────────────────┐
│         Zealot 内部三层                           │
│                                                  │
│  Layer 1: Python Shell（兼容层）                   │
│  ┌────────────────────────────────────────────┐  │
│  │ FastAPI Server        Model Loader         │  │
│  │ (OpenAI 兼容)          (HuggingFace 生态)   │  │
│  └───────────────────┬────────────────────────┘  │
│                      │ PyO3 FFI                  │
│  Layer 2: Rust Core（内核层）                    │
│  ┌────────────────────────────────────────────┐  │
│  │ ┌──────────┐ ┌──────────┐ ┌─────────────┐  │  │
│  │ │ Scheduler │ │  Block   │ │ Constrained │  │  │
│  │ │           │ │ Manager  │ │   Decode    │  │  │
│  │ └─────┬─────┘ └────┬─────┘ └──────┬──────┘  │  │
│  │       │            │              │         │  │
│  │       └────────────┼──────────────┘         │  │
│  │                    │ FFI → CUDA              │  │
│  └────────────────────┼────────────────────────┘  │
│                       │ cudaLaunchKernel         │
│  Layer 3: CUDA Kernel（计算层）                   │
│  ┌────────────────────────────────────────────┐  │
│  │ Attention  │ Quantization │ MLP │ Sampling │  │
│  └────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

**分界原则**：
- Layer 1（Python）不参与 decode loop 计算。一次 decode step 的 GPU 计算（attention + MLP + sampling）耗时 ~35ms，Python 在此路径中仅充当调度入口和 I/O。
- Layer 2（Rust）通过 PyO3 被 Python 调用，通过 FFI 调用 CUDA kernel。编译到 `zealot_engine.abi3.so`。
- Layer 3（CUDA）是 vLLM 原版 kernel 的改装优化，不改架构只调参。

---

## 3. 组件图与数据流

```
Client Request
    │
    ▼
┌──────────────────┐
│  API Server      │  Layer 1 (Python) — 接收 OpenAI 请求，tokenize
│  + Tokenizer     │
└────────┬─────────┘
         │ token IDs, params
         ▼
┌──────────────────┐
│  Scheduler       │  Layer 2 — 消费 ExecutionPlan，决定 prefill/decode 调度
│  (Phase 3: Rust) │
│  (Phase 2: Python)
└────────┬─────────┘
         │ 分配 block、确定 batch 组成
         ▼
┌──────────────────┐     ┌──────────────────┐
│  Block Manager   │◄───►│  KV Cache        │  Layer 2 (Rust) / GPU Memory
│  (Rust)          │     │                  │  页式分配 + generation counter 防误操作
└────────┬─────────┘     └──────────────────┘
         │ 提供 block 地址
         ▼
┌──────────────────┐
│  Attention       │  Layer 3 (CUDA) — FA-3/FA-4 kernel
│  Kernel          │
└────────┬─────────┘
         │ 输出 logits
         ▼
┌──────────────────┐
│  Sampling        │  Layer 3 (CUDA)
│  + Constrained   │  ↓
│  Decode (Rust)   │  拦截 logits → 施加 token 约束 → 采样
└────────┬─────────┘
         │ sampled token
         ▼
    Response (SSE stream)
```

**关键交互**：

| 交互对 | 方向 | 机制 | 延迟目标（M4 实测验证） |
|--------|------|------|------------------------|
| Python → Rust | 调用 Block Manager | PyO3 FFI | < 200ns |
| Rust → CUDA | Block Manager 通知可用地址 | cudaLaunchKernel | < 15μs |
| Sampling → Constrained Decode | 每 step logits 校验 | Rust 内同步调用 | < 10μs (CPU) |
| Python → GPU | Model Loader 写入权重 | PyTorch CUDA stream | 不参与推理热路径 |

> 以上延迟数字为 M4 设计目标值，非实测值。在 M4 交付时通过 §8.1 的 FFI 基准测试验证，以实测值为准。

---

## 4. 请求生命周期

一次 `/v1/chat/completions` 请求在 Zealot 内部的完整路径：

```
1. gRPC Server (tonic, Rust)
   - 接收 InferRequest (Runtime Interface proto)
   - 解析 token IDs 和调度 hints

2. Scheduler (Rust)
   - 根据 ExecutionPlan 决定调度策略（Phase 2: vLLM 等价逻辑，Phase 3: 增强调度）
   - 调用 BlockManager.allocate_blocks(seq_len)

3. Block Manager (Rust)
   - 从 free pool 分配 GPU block
   - 返回 BlockHandle（generation-gated typed handle）

   Python Model Loader 在此前已完成——启动时通过 PyO3 加载 HF 权重到 GPU，
   decode loop 内不再穿越 Python/Rust 边界。

4. Prefill (CUDA)
   - 并行处理全部 prompt tokens
   - 构建 KV cache → 写入 block_table 对应位置
   - 产生第一个 output token

5. Decode Loop（反复执行）:
   a. Attention kernel (CUDA)
      - 通过 block_table 索引读取 KV cache，计算 logits

   b. Constrained Decode (Rust)
      - 如果请求含 json_schema：DFA 校验 token，裁剪 logits

   c. Sampling (CUDA)
      - 输出 token

6. Stream Complete
   - Scheduler.free_seq()
   - BlockManager.drop(handle) → refcount -1，归零回收
   - gRPC stream 返回 usage + finish_reason
```

---

## 5. Python ↔ Rust FFI 设计

### 5.1 绑定模式

Zealot 是 **Rust 主进程**，通过 PyO3 嵌入 CPython 解释器来调用 HuggingFace Model Loader。方向是 Rust → Python，不是 Python → Rust。

```rust
// Zealot Backend 启动时初始化 Python
let py = Python::with_gil(|py| {
    let hf = py.import("transformers")?;
    let model = hf.call_method1("AutoModelForCausalLM", (...))?;
    // 权重加载完成，释放 GIL，进入推理循环
    Ok::<_, PyErr>(model)
})?;

// 推理循环全程 Rust native，不涉及 Python
// Python 解释器保持休眠状态，不在 decode loop 内
```

**Phase 2 立即这样运行**，不存在"先用 vLLM 注入过渡"的阶段。

### 5.2 跨 FFI 的数据流

Python ↔ Rust FFI 仅发生在模型加载阶段。推理热路径中所有数据结构都在 Rust 侧，不跨越 FFI：

| 阶段 | 涉及 FFI | 内容 |
|------|---------|------|
| 启动 | Rust → Python | 调用 HuggingFace 加载权重到 GPU |
| Prefill/Decode | 无 | 全部 Rust native + CUDA |
| 返回结果 | 无 | gRPC 直接序列化，不经过 Python |

### 5.3 GIL 策略

**Zealot 不持有 GIL**。Python 解释器在模型加载完成后释放 GIL 并保持休眠。

| 阶段 | GIL 状态 | 说明 |
|------|---------|------|
| 启动加载权重 | 持有 | Rust 调用 `Python::with_gil()` → HuggingFace 加载 |
| 推理循环 | 已释放 | 全部 Rust + CUDA，Python 解释器空闲 |
| 解码时 | 从未持有 | Block Manager / Constrained Decode / Scheduler 是纯 Rust，不存在 GIL 竞争 |

对比前置设计（Python 调 Rust 插件模式）：该模式下每次 Block Manager 调用都需持有 GIL，batch 并发时产生串行化瓶颈。当前 Rust 主进程模式下此问题不存在。GIL 仅在模型启动时使用一次，之后永不触及。

---

## 6. 模块接口定义

### 6.1 Block Manager

**设计要点**：

- **Generation counter**：`allocate()` 返回的不是裸 `usize`（可能被 Python 侧重复使用），而是带 generation 的 typed handle。对已释放的 handle 调用 `free()` 或 `reference()` 返回 `Err(StaleHandle)`，而非静默操作到错误的 block。这解决了"Rust ownership 无法在编译期阻止 Python 侧逻辑错误"的问题。
- **Free list 同步**：使用 `Mutex<Vec<usize>>` 保护空闲链表。Phase 2 单线程下无实际竞争，Phase 3 多线程时替换为无锁结构（详见 §6.1.1）。
- **Refcount**：`Vec<AtomicUsize>`，原子操作保证线程安全。

```rust
/// GPU 显存的分页管理器。PagedAttention 算法的 Rust 实现。
///
/// # Safety
///
/// Rust 编译器确保 Rust 代码内部无悬垂指针/数据竞争，但无法阻止 Python
/// 侧的逻辑错误（如对同一 handle 重复 free、忘记 reference 导致提前回收）。
/// 因此引入 generation-gated BlockHandle 作为防御层：对已释放的 handle
/// 的后续操作返回 Err(StaleHandle)，而不是 UB。
#[pyclass]
struct BlockManager {
    num_gpu_blocks: usize,
    block_size: usize,
    // 内部：Mutex<Vec<usize>> free_list + Vec<AtomicUsize> refcount
    //       + Vec<AtomicU64> generation (每 block 一个版本号)
}

/// 带 generation 的 block 句柄。Python 侧按值持有。
#[pyclass]
struct BlockHandle {
    block_id: usize,
    generation: u64,      // 分配时的版本号。free 后 generation 递增，
}                         // 旧的 BlockHandle 操作返回 Err(StaleHandle)

#[pymethods]
impl BlockManager {
    #[new]
    fn new(num_gpu_blocks: usize, block_size: usize) -> Self;

    /// 分配一个 free block，返回带 generation 的句柄。
    /// 池空 → Err(OutOfBlocks)
    fn allocate(&self) -> PyResult<BlockHandle>;

    /// 释放 block。引用计数 -1，归零时回 free pool 并递增 generation。
    /// handle 已过期 → Err(StaleHandle)
    fn free(&self, handle: &BlockHandle) -> PyResult<()>;

    /// 增加引用计数。handle 已过期 → Err(StaleHandle)
    fn reference(&self, handle: &BlockHandle) -> PyResult<()>;

    #[getter]
    fn free_blocks(&self) -> usize;
}
```

#### 6.1.1 并发安全与 Free List 同步策略

| 阶段 | 同步原语 | 理由 |
|------|---------|------|
| Phase 2（单线程） | `Mutex<Vec<usize>>` | 无竞争，`Mutex` 开销 ~20ns，可接受 |
| Phase 3（多线程） | 评估中，候选方案见下 | 取决于 M5 benchmark |

**Phase 3 候选方案**：

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| `RwLock<VecDeque>` | 简单，分配和回收分用读/写锁 | 写锁阻塞所有读 | 分配/释放比例 > 10:1 |
| lock-free `crossbeam::SegQueue` | 无锁，低延迟 | 不能批量回收（释放后不能直接 push 到队列头） | 低延迟优先 |
| `Mutex<Vec>`（继续） | 最简单，已验证 | 高竞争时退化为串行 | 实测竞争 < 5% 时保留 |

**决策时机**：M5 交付前通过 `loom` 并发模拟 + 真实 GPU 环境 benchmark 确定。M4 使用 `Mutex<Vec>` 作为起步实现。

#### 6.1.2 BlockHandle 与内存安全

Rust 编译器保证：
- Rust 代码内部无悬垂指针、无数据竞争（编译期检查）

Rust 编译器**不保证**（因为跨越 FFI 边界后 Python 不遵守 Rust 规则）：
- Python 不会对已释放的 handle 重复调用 `free()`
- Python 不会忘记调用 `reference()` 导致提前回收
- Python 不会在 seq 被取消后将 handle 传给错误的 seq

**防御措施**：`BlockHandle` 的 `generation` 字段在每次 `allocate()` 后递增，`free()` 后再递增。对过期 handle 的操作在运行时返回 `Err(StaleHandle)`，逻辑错误变成可诊断的错误而非静默数据损坏或 UB。

---

### 6.2 Constrained Decode

```rust
/// JSON Schema → DFA 编译器。
#[pyclass]
struct JsonSchemaCompiler {
    // 编译缓存：HashMap<SchemaHash, Arc<ConstrainedGrammar>>
    cache: Mutex<LruCache<u64, Arc<ConstrainedGrammar>>>,
    max_states: usize,      // 编译拒绝阈值（见 §6.2.1）
    max_compile_ms: u64,    // 编译超时阈值
}

/// 编译后的约束语法（DFA）。
#[pyclass]
struct ConstrainedGrammar {
    // DFA 内部表示（压缩状态转移表）
}

#[pymethods]
impl JsonSchemaCompiler {
    #[new]
    fn new(cache_size: usize, max_states: usize, max_compile_ms: u64) -> Self;

    /// 编译 JSON schema → 约束语法。
    /// 缓存命中 → O(1)，未命中 → 编译。
    /// 编译超时或状态数超限 → Err(SchemaTooComplex)
    fn compile(&self, schema: &str) -> PyResult<ConstrainedGrammar>;
}

#[pymethods]
impl ConstrainedGrammar {
    fn allowed_tokens(&self, state: usize) -> Vec<i32>;
    fn advance(&self, current_state: usize, token_id: i32) -> PyResult<usize>;
    fn is_valid_final(&self, state: usize) -> bool;
}
```

#### 6.2.1 缓存策略

| 维度 | 策略 |
|------|------|
| **Key** | `SHA256(schema_str)` 的前 8 字节 → `u64` |
| **上限** | 可配置，默认 128 条。LRU 淘汰 |
| **失效** | 进程重启清空。无 TTL——schema 不变则 grammer 不变 |
| **绕过缓存** | `compile(schema_str, bypass_cache=True)` |

#### 6.2.2 对抗性输入防护

JSON schema 是外部传入的、不可信。理论上存在状态数指数增长的对抗性 schema（如深层嵌套 `{"anyOf": [{"anyOf": [...]}]}`）。

**防护措施**：

| 措施 | 默认值 | 行为 |
|------|--------|------|
| `max_states` | 10000 | DFA 状态数超过此阈值 → 拒绝编译，返回 `Err(SchemaTooComplex)` |
| `max_compile_ms` | 500ms | 编译耗时超过此阈值 → 中断编译，返回 `Err(SchemaCompileTimeout)` |
| 嵌套深度限制 | 32 | schema 解析时检查，超过 → 拒绝 |

上述阈值可配置，在 `JsonSchemaCompiler::new()` 中指定。

---

## 7. Phase 2 → Phase 3 演进路径

```
Phase 2（M4-M6）：独立引擎
  - Zealot Backend 独立进程运行
  - Block Manager + Constrained Decode (Rust)
  - Scheduler (Rust, 逻辑等价于 vLLM continuous batching)
  - Attention / Quant kernel (CUDA)
  - Model Loader (Python, PyO3 嵌入)
  - Gateway 通过 Runtime Interface gRPC 路由到 Zealot

Phase 3（M7-M12）：持续优化
  - Prefill-Decode 分离上线（Scheduler 增强）
  - Speculative Decoding 集成
  - RadixAttention 借鉴
  - 性能达标（Together TIE 80%+）
```

| 里程碑 | 标志 | 交付 |
|--------|------|------|
| M4 | Zealot Backend 独立进程 + Block Manager | Generation-gated handle，Runtime Interface gRPC |
| M5 | Constrained Decode + Attention kernel v1 | 缓存方案 + DoS 防护，20-30% 吞吐提升 |
| M6 | Prefill-Decode 分离原型 | GPU 利用率 > 50% |
| M7-M8 | Scheduler 增强 + Speculative Decoding | 追平 vLLM 2x |
| M9-M10 | UIE 1.0 alpha | 内部 benchmark 通过 |
| M11-M12 | UIE 1.0 stable | 默认引擎，性能达标（Together TIE 80%+） |

---

## 8. 测试架构

```
Zealot 测试金字塔：

                    ┌──────────────┐
                    │  GPU 集成测试  │  ← 需 GPU 硬件
                    │ (CUDA kernel) │
                   ┌┴──────────────┴┐
                   │ Python 集成测试  │  ← pytest + import zealot_engine
                   │ (PyO3 接口验证) │
                  ┌┴────────────────┴┐
                  │  Rust 单元测试     │  ← cargo test
                  │ (纯逻辑，GPU-free) │
                  └─────────────────┘
```

| 层级 | 框架 | 覆盖 | 运行环境 |
|------|------|------|---------|
| Rust 单元测试 | `cargo test` | Block Manager 分配/释放算法、DFA 编译、错误路径 | CI（无 GPU） |
| Property-based | `proptest` | 随机 Block Manager 操作序列、schema/JSON 往返 | CI |
| 并发正确性 | `loom` | Block Manager Free List 多线程竞争模拟 | CI |
| Python 集成 | `pytest` | PyO3 接口一致性、错误转换、`StaleHandle` 检测 | CI（无 GPU） |
| FFI 基准测试 | `criterion` | Block Manager 单次操作延迟、GIL 持有时间 | CI |
| GPU 集成 | `pytest + torch` | Block Manager ↔ CUDA 交互、kernel benchmark、schema DoS 测试 | GPU 机器 |

### 8.1 FFI 延迟基准测试

M4 交付时必须包含的 benchmark（`cargo bench` / `criterion`）：

| Benchmark | 测量项 | 验收阈值 |
|-----------|--------|---------|
| `bm_allocate_1` | 单次 allocate (池有空闲) | < 500ns |
| `bm_allocate_full` | 池满时 allocate 的 Err 返回 | < 200ns |
| `bm_free_last_ref` | 单次 free (refcount 归零，回池) | < 200ns |
| `bm_reference` | 单次 reference | < 100ns |
| `bm_stale_handle` | 对已释放 handle 调用 free 的 Err 返回 | < 200ns |
| `gil_hold_bench` | Block Manager 在 `allow_threads` 下的实际 GIL 持有时间 | < 1μs/op |
| `decode_schema_simple` | 编译简单 schema + allowed_tokens | < 5μs |
| `decode_schema_reject` | 对抗性 schema 被 max_states 拒绝的时间 | < 1ms |

---

## 9. OOM 与抢占策略

当 `allocate()` 返回 `Err(OutOfBlocks)` 时，上游 Scheduler 的响应策略。

### 当前策略

Scheduler 维护 swap-out 机制：将 KV cache 从 GPU 迁移到 CPU，释放 GPU block。Zealot Block Manager 提供 `free()` 接口，Scheduler 决定 swap 哪些 seq。如果 swap 后仍无法满足，`OutOfBlocks` 上抛触发抢占（preempt by recomputation）。

### Phase 3 增强（设计预览，M7 细化）

Scheduler 将内置抢占决策，决策维度：

| 维度 | 选项 | 选择 |
|------|------|------|
| 抢占对象 | seq 级别的 KV cache block | 按优先级从低到高驱逐 |
| 驱逐方式 | swap to CPU / 丢弃重算 | 优先 swap（保留计算成果）；swap 池满后丢弃重算 |
| 受害选择 | FIFO / 优先级 / 最小损失 | 默认优先级（Serverless 最低），可配置 |
| 回迁触发 | GPU block 释放后通知 | Scheduler 监听 free_pool_size 事件 |

---

## 10. 构建与发布

```
zealot/                     # 源码仓库
  │ cargo build --release
  ▼
target/release/
  libzealot_engine.dylib    # macOS
  libzealot_engine.so       # Linux

  │ maturin build --release
  ▼
target/wheels/
  zealot_engine-0.1.0-cp310-cp310-manylinux_x86_64.whl
  zealot_engine-0.1.0-cp310-cp310-macosx_11_0_arm64.whl
```

**依赖链**：`zealot_engine.abi3.so` → CUDA (libcuda.so) → 无其他 Rust 运行时依赖（静态链接）。

---

## 11. 风险与缓解

| 风险 | 缓解 |
|------|------|
| vLLM upstream rebase 冲突累积 | 每 2 周 rebase cadence；冲突 > 50 文件暂停优化（见 §11.1） |
| 被替换组件的 Python API 与 vLLM 社区版本分叉 | 保持同名接口签名一致；CI 对比接口变化 |
| CUDA kernel 调试困难 | Phase 2 GPU-free 组件先行，GPU 到位后并行 |
| PyO3 FFI 延迟过高 | §8.1 基准测试持续监控；实测 >1μs 即启用 `allow_threads` |
| Constrained Decode 对抗性 schema DoS | §6.2.2 防护：max_states / max_compile_ms / 嵌套深度限制 |
| BlockHandle 的 generation counter overhead | 每个 block 额外 8 字节（u64），对百万级 block 池增加 ~8MB，可接受 |

### 11.1 Rebase 冲突升级路径

当 rebase 冲突 > 50 文件时，按以下步骤升级：

| 步骤 | 责任人 | 动作 | 时限 |
|------|--------|------|------|
| 1 | Engine 开发 | 暂停新功能开发，专注 rebase | 立即 |
| 2 | Tech Lead | 评估冲突范围：API 变化？语义变化？纯格式？ | 1 天 |
| 3 | Tech Lead | 若为格式冲突（clippy/rustfmt），自动解决 | 1 天 |
| 4 | Engine 开发 | 若为 API 语义变化，手动逐文件解决 | 3-5 天 |
| 5 | Tech Lead | 冲突仍 > 50 文件 → 上报架构决策，评估是否放弃 rebase 转向长期 fork | 第 5 天 |

**监控**：CI pipeline 每日对比 fork vs upstream main 的 diff，报告冲突预判。连续 2 次 rebase 间隔 > 1 个月 → 风险升级，优先处理（ADR-003 红线）。

---

## 12. 约束清单

以下约束在 M4-M5 交付前必须满足：

- [ ] Block Manager 使用 `BlockHandle`（generation counter），非裸 `usize`
- [ ] Block Manager free list 使用 `Mutex<Vec>` 启动，含测试用例（`loom` 并发模拟）
- [ ] `allocate()` 延迟 < 500ns（benchmark 验证）
- [ ] `JsonSchemaCompiler::compile()` 有 `max_states` / `max_compile_ms` 防护
- [ ] `JsonSchemaCompiler` 有 schema hash 缓存 + LRU 淘汰
- [ ] 对抗性 schema 拒绝测试（pytest + 10000 状态 schema）
- [ ] §8.1 的 8 个 FFI benchmark 全部在 CI 中运行
- [ ] `phase2_lifecycle.md` 单独文档，描述 Phase 2 可运行的完整路径
