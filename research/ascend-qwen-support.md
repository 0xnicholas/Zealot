# 昇腾 910 与 Qwen 首发兼容组合调查报告

研究日期：2026-11-22
研究分支：`research/ascend-qwen-support`
源票据：`.scratch/zealot-inference-plan/issues/02-ascend-qwen-support.md`

## 研究问题

> 基于 vllm-ascend 官方资料、测试与问题记录，调查昇腾 910 系列单节点环境运行 7B–14B 级 Qwen 稠密模型 BF16 推理的实际支持情况。识别候选芯片/服务器、CANN/驱动/固件组合、模型版本、并行方式、关键功能支持与已知限制，并区分官方验证、社区报告和推测。给出候选首发支持组合及证据链接，不作未经实测的支持承诺。

## 证据分级约定

- **[官方验证]**：vllm-ascend 官方文档/支持矩阵/CI 配置/华为官方规格页，或 Qwen 官方模型卡。
- **[社区报告]**：GitHub issue 中非官方成员的答复、社区项目、论坛与博客。
- **[推断]**：由官方事实推导（如显存账面估算），未经实测验证。

## 结论摘要

候选首发组合（待 Zealot 验证环境实测后方可写入支持矩阵）：

| 层 | 候选 | 证据等级 |
| --- | --- | --- |
| 硬件 | Atlas 800I A2 推理服务器，64GB HCCS 款（8×昇腾 910B，单 NPU 64GB HBM） | [官方验证] |
| 驱动/固件 | Ascend HDK 26.2.RC1（A2 配套） | [官方验证] |
| CANN | CANN Toolkit + Ops 9.2.1，NNAL 9.2.1 | [官方验证] |
| 引擎 | vLLM 0.30.0 + vllm-ascend v0.30.0（最新稳定版，2026-10-29 发布） | [官方验证] |
| 框架 | PyTorch 2.10.0 + torch-npu 2.10.0.post7 + triton-ascend 3.6.0.post2，Python 3.12 | [官方验证] |
| 模型 | Qwen3-8B（BF16，单 NPU TP1；官方 nightly CI 有精度门禁） | [官方验证] |
| 备选模型 | Qwen3-14B（BF16；Qwen3-Dense 矩阵条目覆盖，但无 14B BF16 专属 CI 条目，须自验） | [官方验证(族)/推断(单卡)] |
| 并行方式 | 单 NPU TP1 起步；同节点 TP2/TP4 可选（8 卡整机 HCCS 全互联） | [官方验证] |
| 部署形态 | 官方容器 `quay.io/ascend/vllm-ascend:v0.30.0`（A2，Ubuntu / `-openeuler` 变体） | [官方验证] |

关键排除项：原始昇腾 910 / 910 Pro B（910A）vllm-ascend 官方明确 **不支持且无计划**；Atlas 300I DUO（310P）为实验性支持且 **不支持 BF16**；Atlas 300I A2（910B PCIe 卡）官方文档未列名，issue 仍 open。

---

## 1. 硬件候选

### 1.1 vllm-ascend 官方设备支持范围 [官方验证]

vllm-ascend 最新 FAQ 明确当前只支持（[FAQs](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html)）：

- Atlas A2 训练系列（Atlas 800T A2、Atlas 900 A2 PoD、Atlas 200T A2 Box16、Atlas 300T A2）
- Atlas 800I A2 推理系列（Atlas 800I A2）
- Atlas A3 训练/推理系列（Atlas 800T A3、Atlas 900 A3 SuperPoD、Atlas 800I A3 等）
- [实验性] Atlas 300I Duo（310P 内核；FAQ 注明 310I Duo 稳定版为 vllm-ascend v0.10.0rc1）

明确 **不支持且无计划**：Atlas 200I A2（310B）、**Ascend 910、Ascend 910 Pro B（cann-kernels-910，即 910A 代）**。

含义：Zealot 首发语境的"昇腾 910 系列"必须收敛到 **910B（Atlas A2）**；910C（Atlas A3）亦受支持但属另一产品代；910A 排除。

### 1.2 Atlas 800I A2 规格 [官方验证]

华为官方用户指南《[Atlas 800I A2 推理服务器 技术规格](https://support.huawei.com/enterprise/zh/doc/EDOC1100360586/bf112cc)》：

- 4U 推理服务器，8 路昇腾 910 AI 处理器；单芯片 280 TFLOPS@FP16。
- 64GB HCCS 款：单 NPU 片上内存 64GB、带宽 1600GB/s；8 NPU 经 HCCS 组成 8P Full mesh，每路 7 条 HCCS 链路、最大理论带宽 392GB/s。
- 另有 32GB 款（HCCS/PCIe，800GB/s）——KV cache 容量差异显著，选型时优先 64GB HCCS 款。

vllm-ascend Qwen3-Dense 教程的 BF16 权重硬件要求亦为"1 Atlas A2 inference products (64GB × 8)"（[Qwen3-Dense.md](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/Qwen3-Dense.md)）。

### 1.3 Atlas 300I A2（910B PCIe 单卡）[社区报告，官方未确认]

- issue《[Does vLLM-Ascend officially support Atlas 300I A2?](https://github.com/vllm-project/vllm-ascend/issues/5651)》（2026-01-06 提出，截至本调查仍 open，无官方答复）。社区成员答复：300I A2 与 800I A2 同芯片（910B），只是 PCIe 互联替代 HCCS，可用 800I A2 镜像。
- 另有已关闭 issue《[使用Atlas300I A2部署Qwen2.5、3、3.5的小模型…卡住](https://github.com/vllm-project/vllm-ascend/issues/10012)》说明有用户在 300I A2 上实际部署过 Qwen 系列。
- 结论：300I A2 可作**备选验证对象**，但在官方列名支持前不进入首发承诺。

### 1.4 Atlas 300I DUO（310P）——不满足 BF16 要求 [官方验证]

v0.30.0 支持矩阵中 Qwen3-Dense 在 Atlas 300I DUO 上 **BF16 ❌**，仅 W8A8（W8A8SC 量化权重 + `--dtype float16`）路径（[supported_models.md](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/docs/source/user_guide/support_matrix/supported_models.md)、[quick_start 300I DUO 节](https://docs.vllm.ai/projects/ascend/en/v0.30.0/quick_start.html)）。且为实验性支持。不满足 Zealot"BF16"首发要求，排除。

### 1.5 昇腾 910A（原始 910/910 Pro B）[官方验证不支持 + 社区报告]

官方 FAQ 明确 unplanned。社区项目 [vllm-ascend-910a](https://github.com/blarock1034/vllm-ascend-910a) 自称在 910A 上适配了 Qwen3-8B / Qwen3-30B-A3B（补算子方式），属社区报告，不构成首发依据。

## 2. 软件栈（CANN/驱动/固件/框架）

### 2.1 版本兼容矩阵 [官方验证]

官方《[Versioning Policy / Release compatibility matrix](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html)》与 README 要求整行对齐、禁止跨行混搭：

| vllm-ascend | vLLM | CANN（稳定） | PyTorch / torch-npu | triton-ascend | 状态 |
| --- | --- | --- | --- | --- | --- |
| v0.30.0 | 0.30.0 | 9.2.1 | 2.10.0 / 2.10.0.post7 | 3.6.0.post2 | 最新稳定（2026-10-29，[release](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.30.0)） |
| v0.23.0 / v0.23.0.post1 | 0.23.0 | 9.1.0 | 2.10.0 / 2.10.0.post4 | 3.2.2 | 稳定维护分支（post1 于 2026-09-22） |

维护中的分支：main、releases/v0.30.0、releases/v0.23.0、releases/v0.18.0、releases/v0.13.0（[README Branch 表](https://github.com/vllm-project/vllm-ascend#readme)）。

### 2.2 v0.30.0 验证栈与驱动/固件 [官方验证]

《[v0.30.0 Installation Guide](https://docs.vllm.ai/projects/ascend/en/v0.30.0/installation.html)》给出的整体验证栈：

- Ascend HDK **26.2.RC1**（A2/A3/300I DUO/200I Pro 的驱动+固件）；CANN Toolkit+Ops **9.2.1**；NNAL **9.2.1**（提供 `libatb.so`）；Python **3.12**；PyTorch 2.10.0；torch-npu 2.10.0.post7；triton-ascend 3.6.0.post2（A2/A3/950DT 使用；300I DUO 不用）；FLA NPU 26.9.1。
- HDK 与 CANN 的精确配套以《[CANN 9.2.1 Release Notes](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/softwareinst/releasenote/9.2.1/release-notes.md)》为准（vllm-ascend README 指引）。
- 官方预构建镜像：`quay.io/ascend/vllm-ascend:v0.30.0`（A2/Ubuntu）、`-openeuler`、`-a3`、`-310p` 等变体（[v0.30.0 Quick Start](https://docs.vllm.ai/projects/ascend/en/v0.30.0/quick_start.html)）。

**建议**：首发基线用 v0.30.0（CANN 9.2.1）；若验证机驱动/固件无法升到 HDK 26.2.RC1 配套组合，退路是 v0.23.0.post1（CANN 9.1.0，对应 HDK 见 CANN 9.1.0 release notes，[v0.23.0 安装文档](https://docs.vllm.ai/projects/ascend/en/v0.23.0/getting_started/installation.html)）。

## 3. 模型候选

### 3.1 Qwen3 稠密系列：当前官方验证主线 [官方验证]

v0.30.0 支持矩阵 A2/A3 条目（[supported_models.md](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/docs/source/user_guide/support_matrix/supported_models.md)，✅=可用于生产）：

- **Qwen3-Dense（覆盖 Qwen3-0.6B/1.7B/4B/8B/14B/32B，见[教程标题](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/Qwen3-Dense.md)）**：✅，BF16 ✅，W8A8 ✅，Chunked Prefill ✅，Automatic Prefix Cache ✅，Async Scheduling ✅，TP ✅，DP ✅，Piecewise/Fullgraph ACLGraph ✅，max-model-len 至 128k；LoRA、投机解码、PD 分离列为空白（未声明支持）。
- 教程注明：Qwen3 Dense 自 v0.8.4rc2 起支持，文档基于 **v0.21.0 验证**，"v0.21.0 及之后版本可稳定运行"；BF16 部署即去掉 `--quantization ascend`。

**Qwen3-8B 的 CI 证据** [官方验证]：

- nightly 精度测试配置 [`tests/e2e/models/configs/Qwen3-8B.yaml`](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/models/configs/Qwen3-8B.yaml)：hardware = "Atlas A2 Series"，TP1、dtype auto（即 BF16）、gsm8k 门禁（exact_match strict ≥0.765 / flexible ≥0.81）。
- nightly 编排 [`nightly_config.yaml`](https://github.com/vllm-project/vllm-ascend/blob/main/.github/workflows/configs/nightly_config.yaml)：accuracy-group-1 在 `linux-aarch64-a2b3-1`（单颗 A2 NPU）runner 上跑 Qwen3-8B（BF16）与 Qwen3-8B-W8A8。
- 官方 FAQ 第 10 条说明测试体系：功能 CI + E2E 模型可用性 + nightly 全量 + 性能/精度报告（[FAQs](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html)）。

**Qwen3-14B**：在 Qwen3-Dense ✅ 条目覆盖范围内（族级官方验证），但 CI 中只有 `Qwen3-14B-w8a8sc-310p`（310P 量化）条目，无 14B BF16/A2 专属 CI 或性能配置。14B BF16 单卡可行性为 [推断]：14.8B 参数 ×2 字节 ≈ 29.6GB 权重，64GB 单卡在 0.9 显存利用率下约余 28GB 给 KV cache——账面可行，须实测。

### 3.2 Qwen3 官方模型事实 [官方验证]

Qwen3-14B 官方模型卡（[ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-14B)，[HuggingFace](https://huggingface.co/Qwen/Qwen3-14B)）：

- 14.8B 参数（非嵌入 13.2B），40 层，GQA（Q 40 头 / KV 8 头），原生上下文 32,768，YaRN 可扩至 131,072；部署建议 vllm≥0.8.5（[Qwen3 仓库 README](https://github.com/QwenLM/Qwen3) 建议 vllm>=0.9.0）、transformers>=4.51.0。
- 权重默认 `torch_dtype="auto"` 即 BF16；Apache-2.0 许可。
- 注意：模型卡默认 `max_position_embeddings=40960`（32768 输出 + 8192 提示预留），非超长上下文场景不建议开 YaRN。

### 3.3 Qwen2.5-7B/14B：历史验证、当前未持续验证 [官方验证(历史)]

- v0.7.3–v0.9.x 时代官方有单 NPU 教程《[Single NPU (Qwen2.5 7B)](https://docs.vllm.ai/projects/ascend/en/v0.8.4rc1/tutorials/single_npu.html)》，并提示 32K 上下文需按 HBM 限制 `--max_model_len`（当时示例 26240）。
- 官方模型跟踪 issue [#1608](https://github.com/vllm-project/vllm-ascend/issues/1608) 中 Qwen2-7B 为 Eager/Graph BF16 ✅、精度 ✅。
- 但 v0.30.0 当前支持矩阵与教程目录已不再包含 Qwen2.5 稠密条目（仅 Qwen2.5-Math-RM-72B 🔴 不维护）。Qwen2.5 架构经 vLLM 上游仍可运行属**理论兼容**，不建议作为首发承诺。

## 4. 关键功能支持与已知限制（首发 BF16 文本生成相关）

### 4.1 功能矩阵 [官方验证]

- Qwen3-Dense（A2/A3）BF16 下：连续批处理（V1 默认）、Chunked Prefill ✅、Automatic Prefix Cache ✅、Async Scheduling ✅、ACLGraph（Piecewise ✅ / Fullgraph ✅）、TP/DP ✅；LoRA 与投机解码未在 Qwen3-Dense 行声明（[v0.30.0 支持矩阵](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/docs/source/user_guide/support_matrix/supported_models.md)）。注：main 分支新版矩阵的 A2/A3 特性列仅适用于 A3 场景，A2 细节以各模型教程为准。
- 互斥关系见 [feature_matrix.md](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/docs/source/user_guide/support_matrix/feature_matrix.md)（如 FULL_DECODE_ONLY 与 Piecewise 互斥）。
- FAQ 第 23 条：单节点放得下就单节点；举例 Qwen3-32B BF16 需 4×64GB 卡 TP（[FAQs](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html)）。

### 4.2 已知限制与运维要点 [官方验证]

- **显存/KV cache**：单 NPU 上 `--max_model_len` 受 HBM 约束，超配会在启动时报 ValueError；OOM 时降 `--max-model-len`、调低 `--gpu-memory-utilization`（默认 0.9）、设 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 缓解碎片（FAQ 第 12 条）。
- **ACLGraph 流资源**：capture 可能因流资源不足失败（FAQ 第 17 条），建议升级 HDK/CANN 或缩减 capture sizes；调试时可 `enforce_eager`。
- **抢占导致 TPOT 抖动**：KV cache 打满触发 preemption 会显著拉高 TPOT（FAQ 第 22 条），容量规划需按 `NPU KV cache size` 日志评估。
- **多卡环境变量**：HCCL 相关（`HCCL_BUFFSIZE`、`HCCL_OP_EXPANSION_MODE=AIV` 等）按教程/SIT 配置（如 [Qwen3-30B-A3B-BF16-A2.yaml](https://github.com/vllm-project/vllm-ascend/blob/main/tests/SIT/cases/models/configs/Qwen/Qwen3-30B-A3B-BF16-A2.yaml)：TP4、FULL_DECODE_ONLY、async-scheduling）。
- **确定性**：需 `temperature=0` + `LCCL_DETERMINISTIC=1`/`HCCL_DETERMINISTIC=true` 等（FAQ 第 15 条）。

### 4.3 issue 记录中的风险信号 [社区报告/官方 issue]

- [#20963](https://github.com/vllm-project/vllm-ascend/issues/20963)（open）：FULL_AND_PIECEWISE 图模式在 A2/A3 上每 prefill step 增加约 5ms host 侧开销（性能类，vllm-ascend 0.23）。
- [#14663](https://github.com/vllm-project/vllm-ascend/issues/14663)（open）：Qwen3-VL-8B-Instruct 在 910B3 中文输入乱码——VL 模型，非本票文本稠密目标，但提示 910B3 变体上有未决个案。
- [#12015](https://github.com/vllm-project/vllm-ascend/issues/12015)、[#12989](https://github.com/vllm-project/vllm-ascend/issues/12989)（closed）：310P 上 Qwen3-14B-W8A8SC 部署报错——310P 路径曾有问题且已修，亦旁证 310P 不成熟。
- 未发现 Qwen3-8B/14B 稠密 BF16 在 A2 上的阻断性 open bug（检索 "Qwen3-8B"、"Qwen3-14B"、"910B BF16"）。

## 5. 候选首发组合与验收建议

### 5.1 候选组合（建议，非承诺）

**主选**：Atlas 800I A2（64GB HCCS 款）单节点 × 1 颗 NPU（TP1）
- 镜像 `quay.io/ascend/vllm-ascend:v0.30.0`（Ubuntu；或 `-openeuler`）
- 宿主：HDK 26.2.RC1 驱动/固件 + CANN 9.2.1 + NNAL 9.2.1（或直接用官方镜像内含用户态栈）
- 模型：Qwen3-8B BF16，`vllm serve`（OpenAI 兼容），按需 `--max-model-len`（≤32K 原生；YaRN 到 128K 属矩阵声明上限，长上下文另验）
- 依据：支持矩阵 ✅ + nightly CI 单卡精度门禁 + 官方教程，是证据链最完整的组合。

**备选模型**：Qwen3-14B BF16 同机 TP1/TP2——族级矩阵覆盖，但无 14B BF16 专属 CI，需 Zealot 实测后决定是否纳入首发矩阵。

**备选基线**：vllm-ascend v0.23.0.post1 + CANN 9.1.0（稳定维护分支），用于驱动升级受限的验证机。

### 5.2 验收手段建议（对齐官方方法）

- 功能：`vllm serve` 启动日志 + `/v1/models` + chat/completions 冒烟（quick start 同款流程）。
- 精度：AISBench / lm-eval gsm8k，参考官方 CI 门禁（Qwen3-8B gsm8k strict ≥0.765）。
- 性能：`vllm bench serve` 与 AISBench perf（TTFT/TPOT/吞吐），观察 preemption 日志。
- 稳定性：连续并发压测 + OOM 边界（`--max-model-len` × 并发）扫描。

### 5.3 风险与开放问题

1. 验证机未确认（地图 Note）：需可远程、可装指定 CANN/HDK 的 Atlas 800I A2 64GB 款；32GB 款 KV cache 预算差异大，验收结论不可直接平移。
2. 910B 芯片变体（B1–B4/B3）在 HBM 容量与算力上有差异（`SOC_VERSION=ascend910b1` 等），验收时记录 `npu-smi info` 实际型号。
3. Qwen3-14B BF16 单卡属推断，须实测；若 KV cache 需求高可走 TP2。
4. Qwen2.5 系列当前不在官方持续验证列表，除非有强制需求，首发不承诺。
5. CANN/HDK/torch-npu 必须整行对齐，混装是社区 issue 高频根因（如 torch 版本错配）。
6. LoRA、投机解码、PD 分离在 Qwen3-Dense BF16 上未获官方声明，首发范围不含这些特性。

## 6. 主要参考链接

- vllm-ascend 仓库与 README：https://github.com/vllm-project/vllm-ascend
- FAQ（设备支持/测试体系/运维限制）：https://docs.vllm.ai/projects/ascend/en/latest/faqs.html
- v0.30.0 安装指南（验证栈）：https://docs.vllm.ai/projects/ascend/en/v0.30.0/installation.html
- v0.30.0 Quick Start：https://docs.vllm.ai/projects/ascend/en/v0.30.0/quick_start.html
- v0.30.0 支持矩阵：https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/docs/source/user_guide/support_matrix/supported_models.md
- v0.30.0 特性矩阵：https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/docs/source/user_guide/support_matrix/feature_matrix.md
- Qwen3-Dense 模型教程：https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/Qwen3-Dense.md
- 版本策略与兼容矩阵：https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html
- Qwen3-8B CI 精度配置：https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/models/configs/Qwen3-8B.yaml
- nightly CI 编排：https://github.com/vllm-project/vllm-ascend/blob/main/.github/workflows/configs/nightly_config.yaml
- 模型支持跟踪 issue #1608：https://github.com/vllm-project/vllm-ascend/issues/1608
- Atlas 300I A2 支持询问 issue #5651：https://github.com/vllm-project/vllm-ascend/issues/5651
- v0.30.0 release：https://github.com/vllm-project/vllm-ascend/releases/tag/v0.30.0
- Atlas 800I A2 官方规格：https://support.huawei.com/enterprise/zh/doc/EDOC1100360586/bf112cc
- Qwen3-14B 模型卡：https://www.modelscope.cn/models/Qwen/Qwen3-14B 、https://huggingface.co/Qwen/Qwen3-14B
- Qwen3 官方仓库：https://github.com/QwenLM/Qwen3
- CANN 9.2.1 Release Notes：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/softwareinst/releasenote/9.2.1/release-notes.md
- 历史单卡教程（Qwen2.5-7B, v0.8.4rc1）：https://docs.vllm.ai/projects/ascend/en/v0.8.4rc1/tutorials/single_npu.html
- 910A 社区适配项目（社区报告）：https://github.com/blarock1034/vllm-ascend-910a
