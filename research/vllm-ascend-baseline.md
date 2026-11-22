# vLLM 与 vllm-ascend 候选稳定基线调查

- **调查日期**：2026-11-22（所有"最新""当前"表述均以此日期为准）
- **对应票据**：`.scratch/zealot-inference-plan/issues/01-vllm-ascend-baseline.md`
- **Zealot 首发场景**：单节点昇腾（优先 910 系列 / Atlas A2）在线推理，一款 7B–14B 级 Qwen 稠密模型，BF16，OpenAI 兼容在线服务；偏好可控 vLLM 下游基线与最小补丁
- **范围声明**：本报告只提供候选组合与事实，不替后续决策选定版本。除标注"推断"的内容外，其余均来自官方来源，附可核查链接。

## 结论速览

面向 Zealot 首发场景的候选基线组合（按推荐调查优先级排序，均为官方正式版，非 rc）：

| 候选 | vLLM + vllm-ascend | 发布日期 | CANN / 驱动（HDK) | PyTorch / torch_npu | Python | 官方维护状态 |
|---|---|---|---|---|---|---|
| A（最新稳定） | v0.30.0 + v0.30.0 | 2026-10-29 | CANN 9.2.1 / HDK 26.2.RC1 | 2.10.0 / 2.10.0.post7 | 官方镜像 3.12（支持 >=3.10,<3.13） | releases/v0.30.0 Maintained |
| B（上一代稳定，已有 post 修复） | v0.23.0 + v0.23.0.post1 | 2026-08-16 / post1 2026-09-21 | CANN 9.1.0 / 见 CANN 9.1.0 Release Notes | 2.10.0 / 2.10.0.post4 | >=3.10,<3.13（镜像 3.12） | releases/v0.23.0 Maintained |
| C（更老稳定，维护窗口将尽） | v0.18.0 + v0.18.0 | 2026-04-30 | CANN 8.5.1（部分特性需手工升 9.0.0） | 2.9.0 / 2.9.0.post1+git4c901a4（需手工装） | >=3.10,<3.12 | releases/v0.18.0 Maintained |

- 三者官方支持矩阵中 **Qwen3-Dense（0.6B/1.7B/4B/8B/14B/32B）BF16 在 A2/A3 均为 ✅ Supported**；Qwen3-8B/14B 正好落在 Zealot 首发档位。
- 若首发模型选 **Qwen2.5 系列**：v0.23.0 矩阵中 Qwen2.5 为 ✅（Extended Compatible），而 v0.30.0 矩阵已移除 Qwen2.5 通用条目——这是两个候选之间最实质的模型覆盖差异。
- 不建议以 rc 版（当前最新为 v0.32.0rc1，2026-11-10）或 v0.11.0 及更早版本（CANN 8.3.RC2 本身是候选版）做稳定基线。

---

## 1. vLLM 上游现状

### 1.1 当前版本与发布节奏（官方 GitHub Releases / PyPI 实测）

| vLLM 版本 | 发布日期（UTC） | 来源 |
|---|---|---|
| v0.34.0（最新正式版） | 2026-11-16 | [GitHub Releases](https://github.com/vllm-project/vllm/releases) |
| v0.33.0 | PyPI 2026-11-02；**无 GitHub Release 页**（仅有 v0.33.0rc3 release，同日；tag 提交日期 2026-10-31） | [PyPI](https://pypi.org/pypi/vllm/json)、GitHub API |
| v0.32.0 | 2026-10-18 | 同上 |
| v0.31.0 | 2026-10-05 | 同上 |
| v0.30.0 | 2026-09-22 | 同上 |
| v0.29.0 | 2026-09-09 | 同上 |
| v0.23.0 | 2026-06-15 | 同上 |
| v0.18.0 | 2026-03-20 | 同上 |

**节奏（实测归纳，非官方承诺）**：2026 年内 vLLM 大约每 2 周发布一个小版本（0.29→0.30→0.31→0.32 间隔均为 13 天左右）。无 LTS 概念。

### 1.2 维护与安全策略（官方事实）

- vLLM 官方 [SECURITY.md](https://github.com/vllm-project/vllm/blob/main/SECURITY.md) 声明：安全修复随新版本发布（"The advisory is published when the fix lands in a vLLM release"），**没有针对旧小版本的回补（backport）渠道**。
- 近期安全公告表明旧版本风险在累积（[GitHub Security Advisories](https://github.com/vllm-project/vllm/security/advisories)）：
  - GHSA-q9v9-qpjp-fc24（high，2026-11-02）：`skip_tokenizer_init` 下未认证 EngineCore 崩溃，影响 `>= 0.28.0, < 0.33.0` —— 即 v0.30.0 在影响范围内、0.33.0 才修复。
  - GHSA-h3rc-j82p-2c3h（high，2026-10-20）：EPS 图像输入触发 DoS/Ghostscript RCE，影响范围记录为 0.30.0，公告中未列出已修复版本（查询日期 2026-11-22）。该漏洞走多模态路径，纯文本部署暴露面较小。
  - v0.32.0 起官方在 release notes 中常态化列出安全修复（如 GHSA-4hhp-h66f-j5j7、GHSA-q97c-p368-75c4 等）。
- 2026-09 至 2026-11 的 vLLM 公告中，多数 DoS 类问题集中在多模态、结构化输出、推测解码路径；纯文本在线服务暴露面相对小，但旧版本不获修复这一结构性事实不变。

### 1.3 近期影响下游的 breaking changes（官方 release notes）

- **v0.32.0**（2026-10-18）：**放弃 Python 3.10**；`transformers>=5.16.1,<5.20.0`、`tokenizers>=0.23.2`；移除 cascade attention 等。
- **v0.34.0**（2026-11-16）：**Model Runner V1 被移除**（`VLLM_USE_V2_MODEL_RUNNER=0` 直接启动失败）；默认 `--block-size` 16→64；可扩展 KV cache 默认开启、`gpu_memory_utilization` 默认 1.0；要求 `huggingface_hub>=2.0.0`。
- 含义（推断）：vllm-ascend 已官宣下一版本全面切到 Model Runner V2（v0.32.0rc1 已默认 MRV2），下游基线若自己维护补丁，应向 MRV2 路径靠拢，避免在 MRV1 上累积补丁。

## 2. vllm-ascend 版本策略与维护状态（官方事实）

来源：[Versioning Policy（latest 文档）](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html)、[Release Notes](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/release_notes.html)、[GitHub Releases](https://github.com/vllm-project/vllm-ascend/releases)。

- **版本对应**：自 vLLM 0.7.x 起，vllm-ascend 版本号与 vLLM **一一对应**（vllm-ascend v0.30.0 ↔ vLLM v0.30.0）。官方明确要求按矩阵整行选用，**不得跨行混装版本**。
- **发布节奏**：正式版通常每 3 个月一个（实测：v0.13.0 2026-02-05 → v0.18.0 2026-04-30 → v0.23.0 2026-08-16 → v0.30.0 2026-10-29）；rc 按需发布；post 版用于正式版缺陷修复（如 v0.23.0.post1、v0.7.3.post1）。
- **维护窗口**：Maintained 状态约覆盖最近 2–3 个小版本线的分支，之后转 Unmaintained（只收 bugfix、不发布、无 CI 承诺）再到 EOL。
- **当前 Maintained 分支**（查询日期 2026-11-22）：`main`（对 vLLM main + v0.33.0 tag 有 CI）、`releases/v0.30.0`、`releases/v0.23.0`、`releases/v0.18.0`、`releases/v0.13.0`、`v0.11.0-dev`、`v0.9.1-dev`、`v0.7.3-dev`；`v0.7.1-dev` 已 Unmaintained。
- **版本检查机制**：vllm-ascend 内置对 vLLM 版本的运行时检查；装错版本会报 `InvalidVersion`，可用环境变量 `VLLM_VERSION` 显式指定。main 分支锚定 vLLM 的具体 commit（`.github/vllm-main-verified.commit`，当前为 `c410543d…`，对应 vLLM v0.33.0）。
- **依赖管理纪律**：torch_npu 每 3 个月出 PyPI 稳定版（正式版可用）、每月 dev 版（仅 rc 可用）、每日 nightly（任何版本不可用）。正式版只使用 PyPI 稳定 torch_npu。
- **硬件覆盖**（安装文档）：Atlas A2（910B 系列，含 Atlas 800I A2）、Atlas A3、Atlas 300I DUO / 200I Pro（310P）、950PR/950DT。单节点 A2 是官方默认路径之一（源码构建时 `SOC_VERSION=ascend910b1`）。

## 3. 官方版本对应矩阵（关键行摘录）

来源：[Versioning Policy — Release compatibility matrix](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html)（查询日期 2026-11-22）。

| vllm-ascend | vLLM | Python | Stable CANN | PyTorch / torch_npu | Triton Ascend |
|---|---|---|---|---|---|
| v0.30.0 | v0.30.0 | >=3.10,<3.13 | 9.2.1 | 2.10.0 / 2.10.0.post7 | 3.6.0.post2 |
| v0.23.0.post1 / v0.23.0 | v0.23.0 | >=3.10,<3.13 | 9.1.0 | 2.10.0 / 2.10.0.post4 | 3.2.2 |
| v0.18.0 | v0.18.0 | >=3.10,<3.12 | 8.5.1 | 2.9.0 / 2.9.0.post1+git4c901a4 | 3.2.0.dev20260322 |
| v0.13.0 | v0.13.0 | >=3.10,<3.12 | 8.5.0 | 2.8.0 / 2.8.0.post2 | 3.2.0 |
| v0.11.0 | v0.11.0 | >=3.9,<3.12 | 8.3.RC2 | 2.7.1 / 2.7.1.post1 | N/A |
| v0.9.1 | v0.9.1 | >=3.9,<3.12 | 8.2.RC1 | 2.5.1 / 2.5.1.post1 | N/A |
| v0.7.3(.post1) | v0.7.3 | >=3.9,<3.12 | 8.1.RC1 | 2.5.1 / 2.5.1 | N/A |

配套 CANN 官方发布日期（[昇腾产品公告](https://www.hiascend.com/productbulletins)）：CANN 9.1.0 发布于 2026-07-30，9.1.1 于 2026-09-01，9.2.0 于 2026-10-17，9.2.1 于 2026-10-28，9.3.0-beta.1 于 2026-11-13。自 9.1.0 起 CANN 社区版与商用版归一（[公告](https://www.hiascend.com/productbulletins/detail/809)）。

## 4. 候选组合详查

### 候选 A：vLLM v0.30.0 + vllm-ascend v0.30.0（最新稳定）

- **发布**：2026-10-29（[GitHub Release](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.30.0)、[PyPI](https://pypi.org/project/vllm-ascend/0.30.0/)）；分支 `releases/v0.30.0` Maintained。
- **验证软件栈**（[v0.30.0 安装文档](https://docs.vllm.ai/projects/ascend/en/v0.30.0/installation.html)）：Python 3.12（镜像基准；支持 >=3.10,<3.13）；Ascend HDK 26.2.RC1（A2/A3/310P/200I Pro 驱动固件；950 系列用 25.6.RC3.1）；CANN Toolkit+Ops 9.2.1 + NNAL 9.2.1；PyTorch 2.10.0 / torch_npu 2.10.0.post7；Triton Ascend 3.6.0.post2（A2/A3/950DT）；FLA NPU 26.9.1；mooncake-transfer-engine-npu 0.3.14-rc1；ray >=2.47.1,<=2.48.0。
- **安装方式**：官方 Docker 镜像 `quay.io/ascend/vllm-ascend:v0.30.0`（A2，另有 `-a3`/`-310p`/`-a5` 与 openEuler 变体）；或 CANN 基础镜像 `quay.io/ascend/cann:9.2.1-910b-ubuntu22.04-py3.12` + pip 安装 `vllm==0.30.0`、`vllm-ascend==0.30.0`（华为云 PyPI 镜像）；或源码（`VLLM_TARGET_DEVICE=empty` 安装 vLLM + `pip install -e .`）。
- **模型覆盖（[v0.30.0 支持矩阵](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/docs/source/user_guide/support_matrix/supported_models.md)）**：Qwen3-Dense ✅（BF16、A2/A3、chunked prefill、APC、TP/PP、P/D 分离、Fullgraph AclGraph、max-model-len 128k）。注意此版本将大量老模型标为 🔴 Unmaintained（DeepSeek V3/3.1/3.2/R1、Qwen3-235B-A22B、Qwen3-Next、GLM-4.x、GLM-5、Qwen2.5-Math-RM-72B 等），**Qwen2.5 通用系列条目已从矩阵中移除**（不代表不能跑，但不再是官方承诺）。
- **已知问题（官方 release notes，节选与 Zealot 相关者）**：默认仍是 Model Runner V1，部分模型需手动切 MRV2（下一版本将默认 MRV2）；Gemma-4-E4B-it 注意力算子故障；KV Pool + Mooncake + SSD offload 在 RoCE 网络下的两个未决问题；若干 W4 量化权重精度低于论文报告值。Qwen3-Dense 未出现在已知问题列表。
- **升级风险**：v0.30.0 移除了 StoreKVBlock、一批旧环境变量与融合配置；若从 v0.23.0 升级需重验配置项。上游 vLLM 0.30.0 之后已有 0.31/0.32/0.34 三个版本，包含对下游不可用的安全修复（见 §1.2）。

### 候选 B：vLLM v0.23.0 + vllm-ascend v0.23.0.post1（上一代稳定，已有 post 修复）

- **发布**：v0.23.0 于 2026-08-16，v0.23.0.post1 于 2026-09-21（[GitHub Release](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.23.0.post1)）；分支 `releases/v0.23.0` Maintained。post1 修复了 DP 对齐 dummy run 的 KV cache 写脏、310P MTP 前缀缓存精度等问题。
- **验证软件栈**（[v0.23.0 安装文档](https://docs.vllm.ai/projects/ascend/en/v0.23.0/installation.html)）：Python >=3.10,<3.13（镜像 3.12）；CANN 9.1.0 + NNAL 9.1.0；Ascend HDK 版本以 CANN 9.1.0 Release Notes 为准；PyTorch 2.10.0 / torch_npu 2.10.0.post4；Triton Ascend 3.2.2（A2/A3/950；310P 不支持 Triton）。
- **安装方式**：Docker `quay.io/ascend/vllm-ascend:v0.23.0.post1`（A2，另有 `-a3`/`-310p`/`-a5` 与 openEuler 变体）；pip `vllm==0.23.0` + `vllm-ascend==0.23.0.post1`；源码。官方提供 uv-wheelnext 增量安装路径。
- **模型覆盖（[v0.23.0 支持矩阵](https://github.com/vllm-project/vllm-ascend/blob/v0.23.0/docs/source/user_guide/support_matrix/supported_models.md)）**：Qwen3-Dense ✅（BF16、A2/A3、128k、MLP Weight Prefetch ✅）；**Qwen2.5 ✅（Extended Compatible，A2/A3）**；DeepSeek 全系、GLM-5/5.1/5.2、Qwen3-235B 等老模型当时仍为 ✅/🔵。
- **已知问题（官方 release notes，节选）**：PP+PCP 组合不支持；Sequence Parallelism 在 v0.23.0 支持矩阵中全线不可用（需迁移到 FlashComm1）；`enable_sparse_c8` 拆分为两个开关；`VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL`、`ASCEND_BUFFER_POOL` 被移除；Qwen3-30B-A3B 单 batch 性能回归（#12337）；DeepSeek V4 系列若干问题。Qwen3-Dense/Qwen2.5 稠密路径无已知问题记录。
- **升级风险（相对候选 A）**：停留在 CANN 9.1.0（2026-07-30 发布），晚于其后的 9.1.1/9.2.0/9.2.1；官方"Ready to Deprecate"清单中的特性（Layer sharding、FlashComm2、Weight prefetch、paged attention 等）在 v0.26.0rc1 已删除，未来升级需一次性跨越这些移除项。

### 候选 C：vLLM v0.18.0 + vllm-ascend v0.18.0（更老稳定）

- **发布**：2026-04-30；分支 `releases/v0.18.0` Maintained。
- **软件栈**：CANN 8.5.1（AscendStoreConnector FabricMem、310P、Qwen3-Omni 需手工升 CANN 9.0.0）；torch 2.9.0 / torch_npu 2.9.0.post1+git4c901a4（**非 PyPI 稳定版，需手工下载 wheel 安装**）；triton-ascend 3.2.0.dev20260322（同样需手工安装）；transformers>=4.57.4；Python >=3.10,<3.12。
- **模型覆盖**：Qwen3-Dense 教程声称"基于 v0.21.0 验证，v0.21.0 及之后版本可稳定运行"（见 §5）；v0.18.0 早于该验证基线，但 Qwen3 系列在 v0.13.0 起即为核心支持对象（推断：风险低，但需实测确认）。
- **已知问题（官方，节选）**：x86 架构存在已知问题、需 CANN 9.0.0 才解决（#7993）；Qwen3.x 在 P/D 分离下有精度问题（#8421）；NPU 软分区 + PIECEWISE 图不支持；EPLB 不支持 minimax_m2 与 W4A8。
- **评价**：手工依赖多、CANN 更旧、维护窗口（2–3 个小版本）接近尾声，仅在 CANN 8.5.x 环境不可变更时作为兜底候选。

### 排除项及理由

- **v0.32.0rc1（2026-11.10）**：预发布版，官方定位是"供早期测试"，不适合做稳定基线；但其"默认 Model Runner V2、支持 MiniMax-M3.1-Flash/Qwen3.8-Flash-Next"预示下一正式版方向，值得跟踪。
- **v0.13.0（2026-02-05）/ v0.11.0（2025-12-16）/ v0.9.1 / v0.7.3**：分支名义 Maintained，但分别距最新 3–6 个正式版，CANN 停留在 8.5.0/8.3.RC2/8.2.RC1/8.1.RC1（后三者为 CANN 候选版而非正式版），Python 上限 <3.12，且已知问题更多（如 v0.13.0 的 transformers 升级破坏部分量化权重、v0.11.0 的 Qwen3-Next OOM 等）。机会成本显著，不建议。

## 5. 与 Zealot 首发场景的直接相关事实

- **Qwen3-Dense 官方教程**（[Qwen3-Dense.md @ v0.30.0](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/docs/source/tutorials/models/Qwen3-Dense.md)）覆盖 Qwen3-0.6B/1.7B/4B/8B/14B/32B BF16，硬件要求写明 "1 Atlas A2 inference products (64GB × 8)"，即单节点 Atlas A2 即可；教程注明"基于 vLLM-Ascend v0.21.0 验证，所有 v0.21.0 及之后版本可稳定运行"。Qwen3-8B/14B 恰好落在 Zealot 首发 7B–14B 档位。
- **Qwen2.5 差异**：若首发模型为 Qwen2.5-7B/14B，候选 B（v0.23.0.post1）的官方矩阵仍列 Qwen2.5 ✅，候选 A（v0.30.0）已移除该条目（矩阵仅承诺列出项；未列出模型官方说法是"may be verified independently"，即自行验证）。
- **单节点 A2（910B）** 是 vllm-ascend 的一等公民场景：默认镜像变体、CANN 镜像 `9.2.1-910b`、`SOC_VERSION=ascend910b1` 均直接对应。
- **版本锁定的可控性（推断）**：vllm-ascend 与 vLLM 严格一一对应、官方禁止跨行混装，意味着 Zealot 的"可控下游基线"应整体冻结一行矩阵（vLLM + vllm-ascend + CANN + torch/torch_npu + triton-ascend + HDK），最小补丁以 cherry-pick 形式叠加在 `releases/vX.Y.Z` 分支之上，而不是跟随 main。

## 6. 上游升级节奏与"选择旧稳定版"的机会成本

**节奏事实**：vLLM 约 2 周一个小版本；vllm-ascend 正式版约 3 个月一个、且只选部分 vLLM 版本对齐（如 0.24–0.29 无对应 vllm-ascend 正式版）。因此 vllm-ascend 稳定版天然落后 vLLM 最新版 2–4 个小版本（当前：vllm-ascend v0.30.0 vs vLLM v0.34.0）。

**选择旧稳定版的机会成本**（事实 + 推断）：

1. **安全修复不可获得**（事实）：vLLM 不回补旧版本；vllm-ascend 的 Maintained 窗口仅 2–3 个小版本线，且 post 版只修自身缺陷，不升级所锚定的 vLLM。
2. **生态位收缩**（事实）：v0.30.0 已将大批老模型标 🔴 Unmaintained；滞留在旧线意味着所持模型组合可能在新版本中被官方放弃，未来被迫跳跃式升级。
3. **升级债累积**（事实）：v0.23.0 → v0.26.0rc1 之间删除了约 10 项特性/配置；v0.30.0 → 下一版将删除 MRV1。跨越版本越多，配置与补丁迁移成本越高。
4. **CANN/驱动锁定**（事实）：候选 B 锁定 CANN 9.1.0（2026-07-30），候选 A 锁定 CANN 9.2.1（2026-10-28，当前最新正式版）。选旧版即放弃新版 CANN 的算子与性能修复（如 v0.13.0 已知问题中"Qwen3-32B 128k 输入性能将在下一 CANN 版本改善"）。
5. **收益侧**（推断）：旧稳定版发布时间长、post 修复到位、社区踩坑记录多；若验证环境 CANN/驱动不可升级，候选 B 或 C 是唯一可行路径。

**给后续决策的输入（不拍板）**：若验证环境允许安装 CANN 9.2.1 + HDK 26.2.RC1，候选 A 在模型覆盖（Qwen3-Dense ✅）、维护窗口（最新线，预期维护期最长）、安全暴露（距最新 vLLM 最近）上综合最优；若环境锁定 CANN 9.1.0 或首发模型为 Qwen2.5，候选 B（v0.23.0.post1）是自然退路。两者都满足"可控下游基线与最小补丁"的诉求：整行冻结 + 在对应 `releases/` 分支上 cherry-pick。

## 7. 事实与推断的区分

- 所有版本号、日期、依赖矩阵、维护状态、已知问题均直接引自 §8 列出的官方页面或官方 API（GitHub Releases API、PyPI JSON API），查询日期 2026-11-22。
- 标注"推断"的段落（§1.3 末、§4 候选 C 评价、§5 版本锁定、§6.5）为基于官方事实的分析，非官方陈述。
- "约 2 周 / 约 3 个月"的节奏为对官方发布日期的实测归纳；官方仅承诺 vllm-ascend 正式版"typically every 3 months"。
- vLLM v0.33.0 的 PyPI 发布与 GitHub Release 缺失是经 API 验证的事实；其原因此处不作推测。

## 8. 来源清单（均为官方或一手来源）

- vllm-ascend Versioning Policy（版本策略 + 完整兼容矩阵 + 分支/EOL 状态）：https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html
- vllm-ascend Release Notes（各版本发布说明、依赖、已知问题）：https://docs.vllm.ai/projects/ascend/en/latest/user_guide/release_notes.html
- vllm-ascend GitHub Releases：https://github.com/vllm-project/vllm-ascend/releases
- vllm-ascend 安装文档（v0.30.0 验证栈 / 镜像 / HDK）：https://docs.vllm.ai/projects/ascend/en/v0.30.0/installation.html
- vllm-ascend 安装文档（v0.23.0）：https://docs.vllm.ai/projects/ascend/en/v0.23.0/installation.html
- vllm-ascend 支持矩阵 @v0.30.0：https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/docs/source/user_guide/support_matrix/supported_models.md
- vllm-ascend 支持矩阵 @v0.23.0：https://github.com/vllm-project/vllm-ascend/blob/v0.23.0/docs/source/user_guide/support_matrix/supported_models.md
- Qwen3-Dense 官方教程 @v0.30.0：https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/docs/source/tutorials/models/Qwen3-Dense.md
- vllm-ascend v0.30.0 Release：https://github.com/vllm-project/vllm-ascend/releases/tag/v0.30.0
- vllm-ascend v0.23.0.post1 Release：https://github.com/vllm-project/vllm-ascend/releases/tag/v0.23.0.post1
- vLLM GitHub Releases：https://github.com/vllm-project/vllm/releases
- vLLM SECURITY.md：https://github.com/vllm-project/vllm/blob/main/SECURITY.md
- vLLM Security Advisories：https://github.com/vllm-project/vllm/security/advisories（GHSA-q9v9-qpjp-fc24、GHSA-h3rc-j82p-2c3h 等）
- PyPI JSON API：https://pypi.org/pypi/vllm/json 、https://pypi.org/pypi/vllm-ascend/json
- 昇腾产品公告（CANN 发布日期）：https://www.hiascend.com/productbulletins ；CANN 版本归一公告：https://www.hiascend.com/productbulletins/detail/809
