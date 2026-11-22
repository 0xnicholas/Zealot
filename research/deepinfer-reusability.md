# 评估 DeepInfer 对 Zealot 的可借鉴性

- 票据：`03-deepinfer-reusability`（wayfinder 研究票）
- 研究日期：2026-11-22
- 研究对象：[`deepseek-ai/DeepInfer`](https://github.com/deepseek-ai/DeepInfer)，分析基于 commit [`a679bb8`](https://github.com/deepseek-ai/DeepInfer/commit/a679bb86ce36e883a98a97733492b686e5b9a08e)（2026-11-03，克隆时 main 最新）
- 定位：DeepInfer 仅作为 Zealot 的**参考实现与潜在补丁来源**，不构成第二上游基线（与地图约束一致）

## TL;DR

1. DeepInfer 开源版是 DeepSeek V4.1 Flash 生产推理系统的**离线推理内核**（引擎 + 算子 + 调度器），官方明确不含在线服务封装、KV 缓存存储系统、P/D 分离编排与负载均衡；`LLMEngine` 文档字符串直接写明 "This release does not include an online-serving wrapper"。对 Zealot 首发目标（昇腾 910 单节点、7B–14B Qwen 稠密模型、OpenAI 兼容在线服务）而言，它**不能作为可运行的服务底座**，只能作为设计参考和局部代码来源。
2. 模型面极度特化：模型注册表只有 `DeepseekV41ForCausalLM` 一个架构，注意力路径硬编码 FlashMLA 稀疏 MLA（V4.1 的 CSA/indexer 结构），KV cache 布局围绕 FP4/MXFP8 量化格式与 YOCO/Engram/HC/DFlash 等 V4.1 专属模块构建。**Qwen 稠密模型的标准 GQA 注意力在其算子集中根本不存在**，模型执行层几乎没有可直接复用的部分。
3. 昇腾路径是"真适配"但**验证范围只覆盖 Ascend 950DT + CANN 9.2**：EP 通信库 DeepEP-Ascend 明确要求 950（A5）的 UBMEM/URMA 互联能力，DeepGEMM-Ascend 只在 950 系列验证；代码内仍有 NPU 侧未完成的占位（graph timer 全部 throw、采样 topk 与 ViT attention 的 torch 回退标注 "TODO(NPU) 待移除"）。**950 → 910 的可移植性未经任何官方验证，不能假定成立**。
4. 真正对 Zealot 有高复用价值的是**与模型无关的系统件**：C++ 零开销异步调度器（ZOS）的流水设计、`Platform`/硬件探测层（torch.cuda→torch.npu 映射、ACL host 内存注册、HCCL 环境初始化）、EPLB 专家均衡算法（纯 PyTorch）、xgrammar 结构化输出集成、`deep_cluster` 硬件/拓扑工具。这些多为设计模式级借鉴，直接搬代码的范围有限。
5. 许可证总体友好：DeepInfer 为 Apache-2.0（与 Zealot 相同），多数文件保留 "Derived from vLLM" 与 vLLM/HF 版权头，复用时需保留这些声明；vendor 内核多为 MIT。**例外：DeepEP-Ascend 仓库当前没有任何 LICENSE 文件**（GitHub API `license: null`，已实测），其代码复用权利不明，涉及 EP 通信代码时必须谨慎。
6. DeepInfer 与 vLLM 的关系是"早期 vLLM 的深度改写分叉"：保留 vLLM 0.x 时代的概念骨架（Sequence、BlockManager、SchedulerOutputs 命名），但调度器已 C++ 化、引擎为单进程多 worker 结构、无任何对 vLLM 的运行时依赖（全仓 0 处 `import vllm`）。**它不能被当作 vLLM 插件或可直接合入 vLLM 的补丁包**；Zealot 若以 vLLM 为基线，DeepInfer 的角色是"对照设计文档 + 可剥离组件的代码来源"。

## 研究方法与来源

- 克隆官方仓库全文通读：`README.md`、`doc/inference_deployment_qa.md`（官方部署 Q&A）、`examples/README.md`、`eval/README.md`、`CONTRIBUTING.md`、`pyproject.toml`、`setup.py`、`build.sh`、`scripts/build*.sh`、`Makefile`、`.gitmodules`。
- 源码深读：`deepinfer/platform.py`、`deepinfer/environ.py`、`deepinfer/engine/llm_engine.py`、`csrc/engine/scheduler.hpp`、`csrc/asc_utils.cpp`、`csrc/graph_timer/graph_timer_asc.cpp`、`deepinfer/model_executor/models/deepseek.py`、`models/__init__.py`、`layers/attention_backend.py`、`layers/tilelang_ops/{ops.py,asc/,cuda/}`、`model_executor/eplb/policy.py`、`npu_ops/`、`worker/model_runner.py`、`worker/zos_status.py`、`deep_cluster/hardware/*`、`entrypoints/llm.py`、`structure_output/compiler.py`、`tests/`。
- 外部核查：GitHub API（仓库元数据与许可证探测）、各 vendor 子模块上游仓库的 LICENSE 与 README、TileLang LICENSE、DeepSeek-V4.1-Flash-1017 模型许可证（HF 直连失败，经多个一致二手来源佐证，详见 §7）。
- 每条结论尽量落到具体文件/URL；无法从一手来源确认的标注为**未知**。

## 1. DeepInfer 是什么：官方声明与代码实证

[README](https://github.com/deepseek-ai/DeepInfer/blob/main/README.md) 声明：DeepInfer 是支撑 DeepSeek 大规模生产服务的推理框架，本次开源"提供离线推理及规模化服务所需的核心特性"；支持模型仅 DeepSeek-V4.1-Flash；硬件为 NVIDIA SM100 与 Huawei Ascend 950；构建要求 Linux、GCC≥13、Python 3.12、CUDA 13 或 CANN 9.2。完整推理服务还需要、但**本发布不包含**：KV 缓存系统、VE/Prefill/Decode 分离编排、跨 DP 副本的负载均衡。

代码实证与声明一致，且更具体：

- **入口只有离线 `LLM` 类**：`deepinfer/entrypoints/` 下只有 [`llm.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/entrypoints/llm.py)，无 OpenAI 兼容服务器、无 HTTP 服务端。[`llm_engine.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/engine/llm_engine.py) 类文档注明 "The `LLM` class wraps this class for offline batched inference. This release does not include an online-serving wrapper."
- **生产组件以"虚函数"形式留白**：`LLMEngine` 中 `init_pd_split`、`do_save_kvcache`、`pe_prepare_save_kvcache`、`send_processed_prompts`、`post_schedule` 等均为空钩子（代码注释直接标"虚函数"），P/D 分离与 KV 落盘的真正实现留在 DeepSeek 内部子类。`EngineType` 枚举（`csrc/engine/scheduler.hpp`）保留了 `PROMPT` / `COMPLETION` 引擎类型，说明 P/D 角色划分在引擎层有骨架，但跨引擎传输、目录服务、LB 均缺席。
- **官方系统级文档** [doc/inference_deployment_qa.md](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md) 描述了完整生产形态：EPD 分离 + 3FS KV 存储集群 + 四级负载均衡（Prefill/Decode/VE/EPLB）+ DeepSeek Recipe 协议层，并明确这些组件"与内部基础设施耦合，尚未完全开源"。该文档是 Zealot 理解大规模昇腾/GPU 在线推理系统设计的**高价值一手参考**，但其代码不在仓库内。
- 仓库极年轻：GitHub API 显示创建于 2026-10-16，全部历史约 6 个 commit，无 tag/release；GitHub Issues 处于关闭状态（`has_issues: false`），对外协作通道有限。

## 2. 与 vLLM 的架构关系与差异

**关系**：README 致谢称 "originally built primarily on vLLM"；源码层面，绝大多数 Python 文件头部为 `SPDX-License-Identifier: Apache-2.0` + `Derived from vLLM; modified by DeepSeek.`，个别文件保留更上游的版权链（如 `rotary_embedding.py` 同时保留 "Copyright 2023 The vLLM team." 与 EleutherAI/HuggingFace 声明，`logger.py` 注明 adapted from skypilot）。概念骨架（`Sequence`、`BlockManager`、`SchedulerOutputs`、waiting/running 队列、抢占重算）可辨认出 vLLM 0.x 血统。

**差异**（均已从源码证实，决定"无法直接当 vLLM 补丁用"）：

| 维度 | vLLM（现行 v1） | DeepInfer |
| --- | --- | --- |
| 调度器实现 | Python（`vllm/v1/core/scheduler.py`） | C++ 1329 行 [`csrc/engine/scheduler.hpp`](https://github.com/deepseek-ai/DeepInfer/blob/main/csrc/engine/scheduler.hpp)，pybind 暴露，调度期 `gil_scoped_release` |
| 异步调度 | engine core 进程 + 异步输出处理 | ZOS（zero-overhead scheduling，[`zos.cpp`](https://github.com/deepseek-ai/DeepInfer/blob/main/csrc/engine/zos.cpp)、[`zos_status.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/worker/zos_status.py)）：CPU 在 GPU 执行 forward(N)/sample(N) 时并发做 schedule(N+1) 与 post-process(N−1)，官方称借鉴 SGLang/NanoFlow |
| DP 步型协调 | 各 DP rank 独立调度 | NORMAL 引擎内 prefill/decode 步型由 DP 组 allreduce **投票**统一（ALL 点/ANY 点，`scheduler.hpp::_schedule`） |
| 平台抽象 | `vllm.platforms` 插件机制 | 自研 [`platform.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/platform.py)：NPU 侧用 `torch_npu.contrib.transfer_to_npu` 把 `torch.cuda.*` 整体映射到 `torch.npu.*`，主干代码继续写 cuda 风格 |
| 内核生态 | FlashAttention/FlashInfer/Triton 等 | 全部自研栈：FlashMLA、DeepGEMM、DeepEP、DeepSelect、TileKernels + 仓内 [TileLang](https://github.com/deepseek-ai/DeepInfer/tree/main/deepinfer/model_executor/layers/tilelang_ops) 算子（`cuda/` 与 `asc/` 双后端，经 [`ops.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/ops.py) 按平台分发） |
| 模型支持 | 数百架构 | 注册表仅 `DeepseekV41ForCausalLM`（[`models/__init__.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/models/__init__.py)） |
| 服务层 | OpenAI 兼容 API server | 无 |
| 运行时依赖 | — | 全仓 0 处 `import vllm`（已 grep 验证） |

结论：DeepInfer 是**独立演进的分叉**，不是 vLLM 的可合入分支。Zealot 不能把它当 vLLM 插件或补丁集直接应用；可行路径是选择性移植其自研组件，或以它为参照重写对应模块。

## 3. Ascend 后端适配方式（实证）

- **双栈构建**：`DEEPINFER_ACCELERATOR=nvidia|ascend` 驱动 [`build.sh`](https://github.com/deepseek-ai/DeepInfer/blob/main/build.sh) → [`scripts/build_vendor.sh`](https://github.com/deepseek-ai/DeepInfer/blob/main/scripts/build_vendor.sh) 按平台分别构建 vendor 内核（Ascend 侧为 DeepSelect、FlashMLA、DeepEP-Ascend、DeepGEMM-Ascend，均以 `BUILD_TARGET=ASCEND` 构建并以同名包安装），再由 [`setup.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/setup.py) 用 `torch_npu.utils.cpp_extension.NpuExtension`（`-DWITH_NPU`）编译 C++ 扩展，产物按 `deepinfer/_so/<accelerator>-<arch>/` 分平台存放。
- **NPU 平台层**（`platform.py::PlatformNPU`）：依赖 `torch-npu==2.13.0rc1`（[`requirements-ascend.txt`](https://github.com/deepseek-ai/DeepInfer/blob/main/requirements-ascend.txt)）；设置 HCCL 系列环境变量、`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`；有一个显式 hack `TORCH_NPU_DEVICE_CAPABILITY=9.0`（注释称等待 torch_npu 修复）；通过 ctypes 直调 `libascendcl.so` 的 `aclrtHostRegisterV2` 实现 host 内存 pinning。
- **C++ NPU 适配**：[`csrc/asc_utils.cpp`](https://github.com/deepseek-ai/DeepInfer/blob/main/csrc/asc_utils.cpp) 用 ACL IPC（`aclrtIpcMemImportByKey` 等）实现跨进程显存共享，注释记录了 2MB 对齐与 `ENABLE_PEER_ACCESS` 等 Ascend 特有约束。
- **NPU 内核策略**：仓内 30 个 TileLang Ascend 算子（`tilelang_ops/asc/`，`from tilelang.ascend import language as T`，配合 `tile_kernels` 的 UB/核数查询）与 30 个 CUDA 算子大致对位，但有差异（asc 多 `kv_quant`、`rstd_1ulp`，缺 `outlier_detection`、`quant` 等）；**缺口用 monkey-patch 兜底**：[`npu_ops/`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/npu_ops/patch_imports.py) 通过 import hook 把 DeepSelect 的 `topk` 换成 torch 回退（注释 "TODO(NPU): To be removed when `topk` for sampler is ready"）、把 ViT attention 换成 `torch_npu.npu_fusion_attention`。
- **NPU 侧明显未完成处**：[`csrc/graph_timer/graph_timer_asc.cpp`](https://github.com/deepseek-ai/DeepInfer/blob/main/csrc/graph_timer/graph_timer_asc.cpp) 所有方法直接 `throw std::runtime_error("NPU graph timer primitives are not implemented yet")`——图内计时探针在 NPU 上不可用。
- **测试**：pytest marker 区分 `ci`（GPU）与 `ci_npu`（[`CONTRIBUTING.md`](https://github.com/deepseek-ai/DeepInfer/blob/main/CONTRIBUTING.md)）；[`tests/models/test_offline_inference.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/tests/models/test_offline_inference.py) 注释写明："On Ascend the model does not fit on 4 GPUs, so 2 nodes × 8 GPUs are used (EP16/CP8)"——即官方 Ascend CI 基线是 16 卡 950 双节点，NPU 侧 PPL golden 与 NVIDIA 分开维护。
- **vendor 上游自述的硬件门槛**（关键）：
  - [DeepEP-Ascend README](https://github.com/deepseek-ai/DeepEP-Ascend)：要求 "Ascend 950 (A5) with UBMEM connectivity and UBC_CTP/URMA channels"，验证栈为 950DT + CANN 9.2.0 + torch_npu 2.13.0rc1，并明确 "Kernel support on other Ascend generations or CANN versions has not been established by these measurements"；混合通信、CPU 侧 Engram、graph capture 均不支持。
  - [DeepGEMM-Ascend README](https://github.com/deepseek-ai/DeepGEMM-Ascend)："developed and validated on the Ascend 950 series"，需 CANN 9.20 的 `bisheng` 编译器；其 BF16 峰值表（432 TFLOPS）与 DeepInfer 代码中 `get_bf16_tflops_limit()` 对 ascend 硬编码的 432 一致，说明该数值面向 950DT。

## 4. 异步调度与并行模块的耦合程度

**ZOS 异步调度**：由 `DEEPINFER_ZERO_OVERHEAD_SCHEDULING` 控制，跨层协作——C++ 调度器（含 MTP varlen 的乐观推进/回填记账）+ worker 侧 `MetadataZOS`（设备端 metadata 修正、remap）+ 引擎侧 `DeferredRelease`（延迟一轮释放资源与 GPU overlap）。这套机制在概念上可移植（Q&A 文档给出了精确的流水线划分），但实现与 DeepInfer 的 C++ Sequence/BlockManager、V4.1 的 MTP/状态缓存深度交织：**对 Zealot 是设计模板，不是可搬代码**。注意：Zealot 若基于 vLLM v1，vLLM 自身已有 async scheduling 方向，应先评估差距再决定是否借鉴 ZOS 的具体设计（如 DP 步型投票、延迟释放）。

**并行**（[`parallel_utils/`](https://github.com/deepseek-ai/DeepInfer/tree/main/deepinfer/model_executor/parallel_utils)）：实现 EP（DeepEP all-to-all + userbuffer symm-mem）、attention DP（#DP=#EP）、mirror CP（`cp.py`）；无 TP——官方 Q&A 解释生产不用 TP/CP（CP 实测端到端最多慢 5%）。EPLB（[`eplb/policy.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/eplb/policy.py)）是**纯 PyTorch 张量运算**的贪心重放置算法（冗余专家分配 + 按历史负载的批贪心摆位），设备无关，是仓内少数可评估直接移植的算法件——但只对 MoE 模型有意义，Qwen 稠密首发用不上。

**耦合度总结**：调度/并行代码与 (a) V4.1 模型语义（MTP slot、SWA state manager、engram lookup table 直接进入 C++ 调度器构造参数）、(b) DeepSeek 自研通信/内核栈（DeepEP userbuffer、TileLang 算子）、(c) 双平台抽象层 三重耦合。任何移植都需先剥离 V4.1 专属参数通路。

## 5. DeepSeek-V4.1 / Ascend 950 / CANN 9.2 耦合清单

| 耦合点 | 证据 | 对 Zealot（910 + Qwen 稠密）的含义 |
| --- | --- | --- |
| 模型架构唯一 | `models/__init__.py` 仅注册 `DeepseekV41ForCausalLM` | 增加 Qwen 架构等于重写模型执行层 |
| 注意力内核 | [`attention_backend.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/attention_backend.py) 硬编码 `from flash_mla import ...`，head 数仅支持 64/128，KV cache 假定为 paged FP8 的 MLA 稀疏注意力 | Qwen 的 GQA 注意力路径不存在；FlashMLA 语义与 GQA 不匹配 |
| 量化格式 | 模型加载路径区分 MXFP4 专家权重 / FP4 主 KV / MXFP8（`deepseek.py::load_model`），Q&A 说明生产用 FP4 KV | Qwen BF16 首发与这些格式无关 |
| V4.1 专属模块 | YOCO(CED) 截断、Engram n-gram 嵌入、HC head、DFlash/DSpark MTP（`v4_modules/` 约 5200 行） | 全部不适用；但 scheduler 把它们作为一等参数 |
| Ascend 950 硬件特性 | DeepEP-Ascend 需 UBMEM/URMA（950 A5）；DeepGEMM-Ascend 仅验证 950；README 硬件支持列 950 | **910 上这些库是否可用未知**；UBMEM/URMA 是 950 代互联能力，910 系列无此硬件路径的风险高 |
| CANN 9.2 + torch_npu 2.13.0rc1 | README 与 requirements-ascend.txt | Zealot 验证机的 CANN 版本是否可到 9.2、torch_npu 是否有对应 910 构建，属环境票待确认项（本研究无法验证） |
| NPU 完成度 | graph timer 未实现；sampler topk / ViT attn 为 torch 回退 | 即便在 950 上，NPU 路径也有官方承认的未收尾处 |

## 6. 哪些设计/代码可能适用于 910 上的 Qwen 在线推理（分级）

**A. 可直接评估移植（模型无关、许可证清晰）**
- `deep_cluster/hardware/*`：设备探测（davinci/ubcore sysfs、npu-smi NUMA 解析）、可见设备分配——与模型无关，Apache-2.0，对 910 环境适配直接有用。
- `platform.py` 的 NPU 初始化模式：transfer_to_npu 映射、HCCL 环境变量、ACL host pinning——**模式**可直接借鉴（Zealot 若走 vLLM 平台插件则落实方式不同）。
- `eplb/policy.py`：纯 PyTorch 均衡算法（仅当未来支持 MoE 时）。
- `structure_output/`：基于 xgrammar 0.2.2 的异步编译 + mask 应用，模型无关；对在线服务的结构化输出需求有参考价值。
- `deepinfer/distributed/shm_broadcast.py`、C++ 引擎 glue 的对象池/零拷贝输出（`RequestOutputPool`）等工程技巧。

**B. 概念可借鉴、代码需重写**
- ZOS 异步调度流水线（含 DP 步型投票、DeferredRelease）——参照设计，不按代码搬。
- C++ 调度器的记账细节（乐观推进 + 接受数回填、LCFS 抢占、水位线 0.9 的 prefill 门控）。
- TileLang 双后端算子组织方式（`ops.py` 分发 + asc/cuda 对位目录）——若 Zealot 需要自研 NPU 算子，这是可复制的工程结构。
- 官方 Q&A 中的系统设计（P/D 分离、KV 存储分层、四级 LB、EPLB 参数、TTL 策略）——作为 Zealot 后续阶段的方案参照。

**C. 不能假定可移植（必须重新验证或重写）**
- 一切 V4.1 模型执行代码（attention_backend、v4_modules、DFlash MTP、Engram/YOCO/HC）。
- vendor 内核在 910 的可用性：DeepEP-Ascend/DeepGEMM-Ascend 均只声明 950；FlashMLA 的 Ascend target 是否覆盖 910 **未知**（其仓库 README 未声明 910 支持）。
- NPU graph 能力栈（cuda_graph_runner 依赖 torch.cuda.graph 语义经 transfer_to_npu 映射到 NPU；950 上可用不代表 910 + 对应 torch_npu 版本可用）——**未知**。
- CANN 9.2 / torch_npu 2.13.0rc1 对 910 的支持矩阵——**未知**，需在验证机上实证。

**D. 明确不适用**
- FP4/MXFP4 量化通路、SWA bounded replay、Engram RDMA、batch-invariant（README 注明 NVIDIA only）等与 V4.1/生产规模绑定的能力。

## 7. 许可证、NOTICE 与第三方来源核查

| 组件 | 许可证 | 核查方式 |
| --- | --- | --- |
| DeepInfer 本体 | Apache-2.0（标准 201 行 LICENSE；[GitHub API 确认](https://api.github.com/repos/deepseek-ai/DeepInfer) `spdx_id: Apache-2.0`） | 一手 |
| NOTICE 文件 | **仓库内不存在**（已全文搜索）；多数源文件带 `Derived from vLLM; modified by DeepSeek.` 头，部分保留 vLLM/EleutherAI/HuggingFace/skypilot 版权行 | 一手；复用时须按 Apache-2.0 §4 保留原有版权与许可声明，修改文件建议保留 "Derived from vLLM" 链并追加 Zealot 声明 |
| vendor: FlashMLA / DeepGEMM / DeepEP / DeepSelect / TileKernels / DeepGEMM-Ascend | 均 MIT（各仓库 LICENSE 实测） | 一手（[FlashMLA](https://raw.githubusercontent.com/deepseek-ai/FlashMLA/main/LICENSE)、[DeepGEMM](https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/main/LICENSE)、[DeepEP](https://raw.githubusercontent.com/deepseek-ai/DeepEP/main/LICENSE)、[DeepSelect](https://raw.githubusercontent.com/deepseek-ai/DeepSelect/main/LICENSE)、[TileKernels](https://raw.githubusercontent.com/deepseek-ai/TileKernels/main/LICENSE)、[DeepGEMM-Ascend](https://raw.githubusercontent.com/deepseek-ai/DeepGEMM-Ascend/main/LICENSE)） |
| vendor: **DeepEP-Ascend** | **无 LICENSE 文件**：[GitHub API](https://api.github.com/repos/deepseek-ai/DeepEP-Ascend) 返回 `license: null`，`raw .../LICENSE` 为空，README 无 License 段 | 一手；**复用权利未授予，引用其代码存在合规风险**——涉及昇腾 EP 通信时只能参考其行为描述，或等待上游补许可证 |
| pip 依赖 | tilelang 0.1.15 **MIT**（[LICENSE](https://raw.githubusercontent.com/tile-ai/tilelang/main/LICENSE)，含 2024-12 至 2025-03 期间 Microsoft 协作条款备注）；xgrammar 0.2.2 Apache-2.0；transformers/tokenizers Apache-2.0 | 一手（tilelang）；xgrammar/transformers 按 pyproject 依赖声明，未逐一开仓库核验 |
| 模型权重 DeepSeek-V4.1-Flash-1017 | 多个一致二手来源称 HF 模型卡标注 MIT（如 [ainews 对 HF trending 的记录](https://ainews.met-ai.tech/en/n/mxPWpyGgwnBJ)、[ccleaks](https://ccleaks.com/news/deepseek-v4.1-flash-1017-open-weights-oct-2026)）；**本环境 HF 直连失败，未能一手核验** | 二手一致；以 <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017> 为准 |

其他第三方代码痕迹：`rotary_embedding.py`（vLLM ← HF Transformers ← EleutherAI GPT-NeoX）、`logger.py`（skypilot）——均有版权头，复用时整条链都要保留。`csrc/asc_utils.cpp` 引用 `third_party/acl/inc/acl/acl.h`，该路径不在仓库内，应来自 CANN 安装（`ASCEND_HOME_PATH`/`ASCEND_TOOLKIT_HOME`），即 **NPU 构建依赖本机 CANN 头文件，仓库不自包含**。

## 8. 对 Zealot 的使用建议（参考/复用边界）

1. **定位**：把 DeepInfer 当"设计参照 + 组件矿石"，不当上游。任何引入都走"读懂 → 剥离 V4.1 耦合 → 按 Zealot（vLLM 基线）结构重写/移植 → 保留许可证链"的流程，不 vendor 整个仓库。
2. **首发阶段（910 + Qwen 稠密 7B–14B 在线）**：DeepInfer 几乎不提供可落地代码——它的算子、模型、量化、调度特化方向与首发矩阵正交。首发价值集中在 `deep_cluster` 硬件探测、NPU 平台初始化模式、结构化输出集成方式，以及 Q&A 文档对在线系统（连续批处理之外的）全局设计的阐述。
3. **后续阶段**：当 Zealot 进入 MoE/EP、P/D 分离、KV 分层存储、异步调度强化等议题时，DeepInfer（代码 + Q&A + Zhihu 架构长文 <https://zhuanlan.zhihu.com/p/2095560295930508953>，本环境未能打开，按官方文档引用）是少数公开的生产级昇腾推理参照，值得按模块再立研究票。
4. **跟踪上游**：DeepInfer 承诺后续模型与性能工作继续开源（Q&A 原文）；其 NPU 未完成项（graph timer、sampler topk 等）的补齐进度可作为 950 生态成熟度信号。建议浅关注，不建立同步流程。
5. **风险红线**：不使用 DeepEP-Ascend 代码（无许可证）；不假定任何 950 验证结论适用于 910；不把 "TileLang 算子双后端对位" 误解为 "910 已有等价内核"——asc/ 内核只声明在 950 栈上验证过。

## 9. 明确未知（未能一手确认）清单

- 910 系列上 CANN 9.2 / torch_npu 2.13.0rc1 的可用性与 DeepInfer 构建可行性——需在昇腾验证机实测。
- FlashMLA Ascend target、TileKernels、DeepSelect 对 910 的支持情况——各仓库未声明。
- DeepEP-Ascend 的许可证意图——上游未放置 LICENSE，是否后续补充未知。
- DeepSeek-V4.1-Flash-1017 权重许可证一手文本（HF 不可达；二手一致称 MIT）。
- DeepInfer 内部（未开源）KV 缓存与 LB 组件的接口边界——只有 Q&A 的文字描述。
- 官方 Q&A 引用的一手架构长文（知乎专栏）内容未能直接核验。

## 附：主要一手来源 URL

- 仓库与文档：<https://github.com/deepseek-ai/DeepInfer> · [README](https://github.com/deepseek-ai/DeepInfer/blob/main/README.md) · [部署 Q&A](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md) · [CONTRIBUTING](https://github.com/deepseek-ai/DeepInfer/blob/main/CONTRIBUTING.md)
- 关键源码：[platform.py](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/platform.py) · [scheduler.hpp](https://github.com/deepseek-ai/DeepInfer/blob/main/csrc/engine/scheduler.hpp) · [llm_engine.py](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/engine/llm_engine.py) · [models/__init__.py](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/models/__init__.py) · [attention_backend.py](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/attention_backend.py) · [npu_ops/patch_imports.py](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/npu_ops/patch_imports.py) · [graph_timer_asc.cpp](https://github.com/deepseek-ai/DeepInfer/blob/main/csrc/graph_timer/graph_timer_asc.cpp) · [eplb/policy.py](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/eplb/policy.py) · [test_offline_inference.py](https://github.com/deepseek-ai/DeepInfer/blob/main/tests/models/test_offline_inference.py)
- vendor 上游：<https://github.com/deepseek-ai/DeepEP-Ascend> · <https://github.com/deepseek-ai/DeepGEMM-Ascend> · <https://github.com/deepseek-ai/FlashMLA> · <https://github.com/deepseek-ai/DeepEP> · <https://github.com/deepseek-ai/DeepGEMM> · <https://github.com/deepseek-ai/DeepSelect> · <https://github.com/deepseek-ai/TileKernels>
- 其他：<https://github.com/tile-ai/tilelang> · <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017> · <https://github.com/deepseek-ai/deepseek-recipe>（协议层，Q&A 提及，本研究未展开）
