# CONTEXT — Zealot 领域语境

单一上下文仓库（single-context）。本文档是领域术语表（ubiquitous language）与文档入口；
架构决策记录（ADR）在 [`docs/adr/`](docs/adr/)。

消费方式见 [`docs/agents/domain.md`](docs/agents/domain.md)：探索代码前先读本文档，
再读与改动区域相关的 ADR；输出中命名领域概念时使用本表术语，不要漂移到同义词。

## 系统定位

| 术语 | 定义 |
|------|------|
| **Zealot** | 本仓库。Ultralisk 推理云平台的自研推理引擎，独立闭源，不 fork vLLM。 |
| **Ultralisk** | 唯一消费者（上游推理云平台）。平台级决策记录在其仓库 `docs/adr/`。 |
| **vLLM** | 对照基线与替代对象。Phase 3 目标：成为平台默认引擎，性能达 Together TIE 的 80%+。 |
| **kill criterion** | 里程碑未达 benchmark 目标时平台回退 vLLM、本项目冻结为研究项目的退出条件（Ultralisk ADR-013）。 |
| **Phase 2 / Phase 3** | Phase 2（M4-M6）：独立引擎灰度替代；Phase 3（M7-M12）：默认引擎 + 性能优化。当前处于 Phase 2 M5。 |
| **Runtime Interface** | Zealot 与平台 Gateway 之间的 gRPC 契约（`proto/runtime/`，随本仓库独立维护，Ultralisk ADR-010）。 |

## 核心组件

| 术语 | 定义 | 代码 |
|------|------|------|
| **Block Manager** | KV Cache 分页管理器（PagedAttention 的 Rust 实现）。 | `src/block_manager.rs` |
| **BlockHandle** | 带 generation 的 block 句柄，Python 侧按值持有；不用裸 `usize`。 | `src/block_manager.rs` |
| **generation** | 每 block 一个版本号（`AtomicU64`），allocate/free 时递增。 | `src/block_manager.rs` |
| **StaleHandle** | 对已释放 handle 操作的运行时错误——逻辑错误变成可诊断错误，而非静默数据损坏。 | `src/error.rs` |
| **Constrained Decode (CD)** | 约束解码引擎：JSON schema → DFA 编译 + CPU 侧 token 校验，对标 SGLang xgrammar。 | `src/constrained_decode/` |
| **JsonSchemaCompiler** | schema → DFA 编译器，含真 LRU 编译缓存、`max_states`/`max_compile_ms` 对抗性防护。 | `src/constrained_decode/schema.rs` |
| **ConstrainedGrammar** | 编译产物（字节级自动机）：`allowed_tokens` / `advance` / `is_valid_final`。 | `src/constrained_decode/matcher.rs` |
| **Scheduler** | 调度器：优先级等待队列、block 预算、准入控制、OOM 抢占；Phase 2 逻辑等价 vLLM continuous batching。 | `src/scheduler.rs` |
| **preempt by recomputation** | OOM 抢占策略：驱逐最低优先级 seq、回收 block、重入队重算（非 swap）。 | `src/scheduler.rs` |
| **Engine** | 步进循环，通过 `ModelRunner` trait 驱动，与具体 runner 解耦。 | `src/engine.rs` |
| **ModelRunner** | 前向执行抽象。三种实现：CPU（Rust）、PyO3（dev-mode）、CUDA。 | `src/model_runner_cpu.rs` / `model_runner_py.rs` / `cuda_runner.rs` |
| **dev-mode** | GPU 到位前的临时形态：`PyModelRunner` 用 torch CPU 前向跑真实推理，Python 在 decode loop 内。GPU kernel 就位后回归「Python 仅启动时加载」。 | `docs/phase2_lifecycle.md` |
| **zealot-backend** | 独立 Rust 主进程二进制：gRPC Runtime Interface + HTTP OpenAI 兼容面双入口。 | `src/bin/zealot-backend.rs` |

## 关键设计语义

- **双入口**：gRPC（Runtime Interface，tonic，默认 :9091）+ HTTP OpenAI（`/v1/chat/completions`，axum）。HTTP 面在 Zealot 内完成，不经 Gateway 转发。
- **GIL 策略**：Rust 主进程嵌入 CPython（方向 Rust → Python）。GIL 仅在启动加载权重时持有一次，decode loop 全程 Rust native，永不触及 GIL。
- **unsupported > unsound**：约束解码对不支持的 schema 关键字显式编译报错，不静默降级为 Any（ADR-0004）。
- **测试金字塔**：Rust 单测 → proptest（模型对照）→ loom（并发模拟）→ criterion（FFI 基准）→ pytest（PyO3 接口）→ e2e（gRPC/HTTP/CPU 真实推理）→ GPU 集成（待硬件）。

## 外部依赖文档

- 架构设计：[`docs/architecture.md`](docs/architecture.md)
- Phase 2 运行链路：[`docs/phase2_lifecycle.md`](docs/phase2_lifecycle.md)
- 平台级 ADR（ADR-003 vLLM 基线 / ADR-009 语言栈 / ADR-010 Backend Runtime / ADR-013 独立项目）：Ultralisk 仓库 `docs/adr/`，本仓库 ADR 在引用时标注「Ultralisk ADR-NNN」。
