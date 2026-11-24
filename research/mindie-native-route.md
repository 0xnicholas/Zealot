# 评估 MindIE 作为昇腾官方原生路线的价值

- 票据：`.scratch/zealot-inference-plan/issues/10-mindie-native-route.md`
- 调查日期：2026-11-25
- 范围：只做路线研究。未下载需额外授权的软件，未在昇腾机器上实测，未实施任何迁移。
- 证据分级：**[官方]** 华为/昇腾官方文档、代码仓、发布页或产品页；**[社区]** 公开 issue、第三方审计或非官方文章；**[推断]** 由已核实事实推出、但官方未直接陈述的判断；**[未知]** 本次未能核实。

## 0. 决策摘要

1. **在当前公开证据下，MindIE 不能作为 Zealot 的替代基线。** 官方文档明确写着：MindIE LLM 组件自 3.1.0 起暂停后续功能演进，现有功能维持维护状态，不再支持新特性和新模型；官方推荐的部署方式是 MindIE Motor + vLLM Ascend。[官方]
2. **“MindIE”已经不是单一引擎，而是分层产品族。** 其中唯一可能替代 vLLM 的是 MindIE LLM 引擎，而它已冻结；MindIE Motor 是服务化/调度层，当前只对接 vLLM-Ascend 与 SGLang，不提供自有推理引擎。[官方]
3. **首发目标 A2 + Qwen3-8B BF16 的官方支持是成立的。** 模型矩阵把 Qwen3-8B 标为“充分验证支持”，Atlas 800I A2 单卡、128k 上下文、LoRA 与 atbgraph 均支持；但 3.1.0 的 CANN 配套只列到 9.1.0，与 vllm-ascend 0.30.0 的 CANN 9.2.1 不在同一版本行。[官方]
4. **功能面与 vLLM 接近，但不等价。** OpenAI 兼容、连续批处理、PagedAttention、SplitFuse（chunked prefill）、Prefix Cache、异步调度都有；但 Qwen 模型上 Prefix Cache 与 SplitFuse 在官方特性矩阵中互斥，Function Call 的官方模型清单不含 Qwen3-8B（与模型 README 冲突），3.1.0 后也不再新增模型或特性。[官方]
5. **建议：vLLM 0.30.0 + vllm-ascend 0.30.0 仍是唯一可持续基线。** MindIE LLM 3.1.0 只作为官方对照与一次性基准参照；MindIE Motor 留作未来多实例/PD 分离时的可选服务层，不进入首发关键路径。

一句话回答票据问题：MindIE 应被定位为**官方原生对照 + 可选的服务化组件来源**，而不是**可替代 vLLM 的基线**；其中“可替代”的那一部分（MindIE LLM）已经被官方自己冻结，并明确让位给 vLLM Ascend。

## 1. 调查方法与资料来源

本次调查先读地图、术语表与三张已解决票据，然后以官方一手资料为主：

- MindIE LLM 官方文档站：<https://mindie-llm-doc.readthedocs.io/zh-cn/latest/>
- MindIE LLM 代码仓（GitCode 镜像，与 GitHub `Ascend/MindIE-LLM` 同源）：<https://gitcode.com/Ascend/MindIE-LLM>
- MindIE Motor 官方文档站：<https://mindie-motor.readthedocs.io/zh-cn/latest/>
- 昇腾社区 MindIE 产品页：<https://www.hiascend.com/developer/software/mindie>
- 昇腾镜像仓库 MindIE 页面：<https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f>

方法说明：`hiascend.com/document` 的正文页由 JavaScript/WAF 渲染，直接抓取只能得到反爬脚本，因此正文优先引用同一份官方文档的 ReadTheDocs 构建站与 GitCode 仓库中的 Markdown 源文件；产品页公告只引用了可核实的标题与链接，公告正文未能读取（见 §12）。所有引用均在 2026-11-25 访问。

## 2. 产品边界：2026-11 的 MindIE 到底是什么

### 2.1 组件清单

官方镜像总览把 MindIE 定义为“Huawei Atlas 的推理加速套件”，包含三个组件：[官方]

- **MindIE LLM**：大语言模型推理。
- **MindIE SD**：Stable Diffusion 图像生成。
- **MindIE Motor**：推理服务编排。

另外还有两个经常被混入“MindIE”讨论、但边界不同的东西：

- **ATB Models / `atb_llm`**：MindIE LLM 的模型库与图后端，仓库内以 `examples/atb_models` 形式存在，不是独立产品。
- **MindCluster / CCAE**：集群使能与运维平台，MindIE Motor 的 K8s 部署依赖 MindCluster，CCAE 是可选的运维可视化平台，不属于推理引擎本身。

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docker/OVERVIEW.md>、<https://mindie-motor.readthedocs.io/zh-cn/latest/architecture/>。

### 2.2 时间线（官方发布页与文档）

| 时间 | 事件 | 来源 |
| --- | --- | --- |
| 2025-12 | MindIE LLM 正式开源 | README “Latest News” |
| 2026-05-08 | MindIE LLM 3.0.0 发布 | GitCode Releases |
| 2026-08-07 | 3.1.0 镜像发布 | `docker/support_tag.md` |
| 2026-08-19 | MindIE LLM 3.1.0 发布 | GitCode Releases |
| 2026-08-21 | 产品页公告“MindIE 3.1.0 版本发布”“MindIE 3.1.0 架构升级与模块更名” | 昇腾社区产品页 |
| 2026-10-15 | MindIE Motor 3.2.0 发布 | Motor 版本说明 |
| 2026-10-19 | 文档 issue 提出补充“暂停演进说明” | GitCode issue #725 |

来源：<https://gitcode.com/Ascend/MindIE-LLM/releases>、<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docker/support_tag.md>、<https://www.hiascend.com/developer/software/mindie>、<https://mindie-motor.readthedocs.io/zh-cn/latest/release_note_motor/>。

### 2.3 关键事实：MindIE LLM 已冻结

MindIE LLM 3.1.0 文档的“简介”页顶部有一条醒目的说明：[官方]

> MindIE LLM 组件自3.1.0版本起暂停后续功能演进，现有功能维持维护状态，不再支持新特性和新模型。推荐使用 MindIE Motor+vLLM Ascend 部署推理服务。关于 vLLM Ascend 的快速部署，请参阅官方文档的快速开始章节。

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/v3.1.0/docs/zh/user_guide/user_manual/introduction.md>

补充核验说明：`v3.1.0` 是**发布分支**（不是不可变 tag），其头部文档包含上述说明；ReadTheDocs 的 `latest` 构建来自 `master`，目前看不到这句话。核验时应直接打开上面的 `v3.1.0` 源文件链接，或对照 issue #725。

同一结论在镜像说明中被再次确认：[官方]

> 以下为 MindIE-LLM 已发布的镜像 Tag。最新镜像为 3.1.0，之后不再发布新镜像。

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docker/support_tag.md>

需要注意两个细节：

- 这条冻结说明出现在 **v3.1.0 发布分支的文档**中；`master` 分支的同名文件没有这句话，`master` 的代码合入也停在 2026-07-21。这说明冻结公告是 3.1.0 发布之后（约 2026-10）补进发布分支的文档变更，而不是 8 月发布时的原始说明。[官方/推断]
- 3.1.0 版本说明标注“维护周期：三个月”。从 2026-08 起算，到 2026-11-25 已到达或略过该窗口末端；冻结说明只说“维持维护状态”，没有承诺安全补丁的期限。[官方/推断]

### 2.4 MindIE Motor 不是引擎，是 vLLM/SGLang 之上的调度层

Motor 架构页的表述非常直接：[官方]

> MindIE Motor 是面向大语言模型（LLM）分布式推理，如 PD 分离推理（Prefill 与 Decode 阶段分离）的请求调度框架。MindIE Motor 通过开放、可扩展的推理服务化平台架构，对接 vLLM-Ascend 和 SGLang 原生运行时。

> MindIE Motor 的软件定位与基本功能保持不变，对接 vLLM-Ascend 与 SGLang 原生推理引擎。

其“支持的推理引擎”页只列出两个引擎：vLLM（已支持）、SGLang（已支持）；没有 MindIE LLM。[官方] 来源：<https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/supported_inference_engines/>。

Motor 3.2.0 的版本说明还明确写了“删除 EngineServer 模块，与推理引擎解耦”，并与 **vLLM v0.30.0 + vLLM Ascend releases/v0.30.0** 配套。[官方] 来源：<https://mindie-motor.readthedocs.io/zh-cn/latest/release_note_motor/>。

这带来一个对 Zealot 很关键的结论：**官方推荐的“MindIE 原生路线”在引擎层已经收敛到 vLLM Ascend**。Motor 提供的是 PD 分离、KV 亲和调度、限流、故障恢复、K8s 编排等“引擎之上”的能力，它不替代 vLLM，也不构成第二条引擎路线。

### 2.5 产品边界结论

| 层 | 组件 | 与 Zealot 的关系 |
| --- | --- | --- |
| 推理引擎 | MindIE LLM 3.1.0 | 唯一可替代 vLLM 的候选；已冻结，不选 |
| 模型库/图后端 | ATB Models、`atb_llm` | MindIE LLM 的内部实现；可作为 Ascend 图优化参考 |
| 服务/调度 | MindIE Motor 3.2.0 | 可选的 vLLM/SGLang 之上的编排层；首发不需要 |
| 图像生成 | MindIE SD | 与 Zealot 无关 |
| 集群/运维 | MindCluster、CCAE | Motor 的部署依赖；首发不需要 |

“官方原生”在 2026-11 的准确含义是：**官方原生引擎（MindIE LLM）暂停功能演进，官方推荐路径改为“官方调度层 + 社区 vLLM Ascend 引擎”**。这对 Zealot 是利好——它验证了 vLLM 路线的方向，而不是推翻它。

## 3. 许可证、分发与商用条件

### 3.1 代码：木兰宽松许可证第 2 版（Mulan PSL v2）

MindIE LLM 与 MindIE Motor 的代码仓 `LICENSE.md` 均为 Mulan PSL v2。[官方]

- 版权许可：永久、全球、免费、非独占、不可撤销，允许复制、使用、修改、分发，不论是否修改。
- 专利许可：每个贡献者授予其贡献必然涉及的专利许可；若对“软件”发起专利诉讼，专利许可自起诉之日终止。
- 商标：不授予商标许可。
- 分发：允许以源码或可执行形式再分发，但必须向接收者提供许可证副本，并保留版权、商标、专利与免责声明。
- 无 copyleft 传染条款；商用与闭源再分发在许可证文本层面是允许的。

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/LICENSE.md>、<https://gitcode.com/Ascend/MindIE-Motor/blob/master/LICENSE.md>。

对 Zealot（Apache-2.0）的含义：

- **可以**阅读、修改、在内部使用与再分发；许可证文本层面没有 copyleft 障碍。把代码片段并入 Apache-2.0 项目时，必须保留 Mulan PSL v2 的许可证与声明；兼容性细节与 NOTICE 归属应由法务确认。[推断]
- **不建议** vendor 整仓：MindIE LLM 已冻结，vendor 会把一个不再演进、且带 C++/Python 混合构建体系的引擎固化进 Zealot。
- Mulan PSL v2 不含商标许可，不能把“MindIE”或华为商标用作 Zealot 的背书。

### 3.2 文档：CC BY-SA 4.0

仓库 README 的免责声明写明：文档的复制、使用、修改及分发受 CC BY-SA 4.0 约束。[官方]

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/README.md>

对 Zealot 的含义：**不要整段复制 MindIE 文档进 Apache-2.0 仓库**。CC BY-SA 4.0 的“相同方式共享”会覆盖被复制的文本；引用事实、改写并以链接溯源是更安全的做法。[推断]

### 3.3 镜像与二进制：分发受账号与华为条款约束

官方镜像安装指南要求：进入昇腾镜像仓库，使用**华为账号登录**，在“镜像版本”页签选择设备形态后下载镜像。[官方] 来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/install/source/image_usage_guide.md>。

本次未能从公开页面读取镜像的许可协议正文：AscendHub 页面是 JS 渲染，未登录时只显示通用页面框架，没有“许可/协议”文本。[未知] 因此：

- **不能确认** MindIE 容器镜像是否允许自由再分发、是否允许在商业产品中随附分发、是否有出口管制或地域限制。
- **不应**把 MindIE 镜像作为 Zealot 的交付物或 CI 基础镜像对外分发；如需使用，应让最终用户自行从官方渠道获取，或由法务审核华为的软件许可协议。
- 代码仓的 Mulan PSL v2 **不等于**镜像内所有二进制的许可；镜像还包含 CANN、驱动、torch_npu、ATB 等组件，各有自己的许可。

### 3.4 分发渠道与内网可行性

- 官方提供镜像仓库的镜像站：`swr.cn-north-4.myhuaweicloud.com/inference`，用于基础镜像预拉取。[官方] 来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docker/OVERVIEW.md>。
- 从源码构建镜像需要访问 Atlas OBS 镜像与华为云 PyPI 镜像，不是完全离线可完成的。[官方] 同上。
- 已有镜像可以 `docker load -i xxx.tar` 导入，再在无外网的机器上运行；权重也需要提前落盘。[官方] 来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/install/source/image_usage_guide.md>。
- 因此“内网安装”在**运行阶段**可行（镜像 tar + 权重 + 驱动/固件），但**获取阶段**仍受华为账号与官方渠道约束；这与 vllm-ascend 使用 `quay.io/ascend/vllm-ascend` 公开镜像的获取方式不同。[官方/推断]

## 4. 支持矩阵：Atlas 800I A2（910B）+ Qwen3-8B BF16

### 4.1 硬件与操作系统

MindIE LLM 3.1.0 的安装说明把 **Atlas 800I A2 推理服务器**列为支持硬件，操作系统（AArch64）包括：[官方]

- Ubuntu 22.04、Ubuntu 24.04 LTS
- openEuler 22.03 LTS、openEuler 24.03 LTS SP1
- Kylin V10/V11、CentOS 7.6、BCLinux、AliOS、UOS 等

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/install/installation_introduction.md>。

注意：虽然安装文档列了 Ubuntu，但 3.1.0 已发布的 910B 镜像 tag 只有 openEuler 24.03 LTS 一个（见 §4.3）。如果 Zealot 的验证机是 Ubuntu，要么走离线 whl/run 安装，要么自行构建镜像。[官方/推断]

### 4.2 模型支持：Qwen3-8B 是“充分验证支持”

MindIE LLM 的模型支持列表用颜色区分“充分验证支持”（绿色）与“仅功能支持”（橙色）。Qwen3-8B 的行为：[官方]

| 项目 | 值 |
| --- | --- |
| Support | 绿色 ●（充分验证支持） |
| 最大上下文长度 | 128k |
| W8A8 / W4A8 / W8A16 | ❌ / ❌ / ❌（即官方矩阵未列量化支持） |
| LoRA | ✅ |
| atbgraph | ✅ |
| aclgraph | ❌ |
| 硬件规格 | Atlas 800I A2：**1 卡**；Atlas 800I A3：1 卡；Atlas 300I Duo：1 卡 |

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/model_support_list.md>。

模型 README 进一步给出精度与特性矩阵：Qwen3-8B 在 800I A2 上 **BF16 ✅、FP16 ✅、Paged Attention ✅、MindIE Service ✅、长序列 ✅、prefix_cache ✅、functioncall ✅、Multi LoRA ✅、FA3 量化 ❌**，Tensor Parallel 仅 **world size 1**。[官方]

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen/README.md>。

这两份官方资料足以确认：**A2 单卡 + Qwen3-8B BF16 是 MindIE LLM 3.1.0 的官方支持组合**，不需要多卡，也不需要量化。

### 4.3 CANN / PTA / Python 配套

MindIE LLM 3.1.0 的版本说明给出的配套关系：[官方]

| 组件 | 版本 |
| --- | --- |
| CANN | 9.0.0 / 9.0.1 / 9.1.0（表中均为 Y） |
| MindCluster | 26.0.0 / 26.1.0 / 7.3.0 |
| Ascend Extension for PyTorch | 26.0.0 |
| Python | 3.11（依赖表） |
| 维护周期 | 三个月 |

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/release_notes.md>、<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/install/source/preparing_software_and_dependencies.md>。

3.1.0 已发布的 910B 镜像 tag 是：[官方]

```text
3.1.0-cann9.0.1-torch_npu2.9.0.post4-910b-openeuler24.03_lts-py3.11
```

也就是说，**A2 官方镜像实际带的是 CANN 9.0.1，不是 9.1.0**；9.1.0 只出现在 950 的镜像 tag 中。来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docker/support_tag.md>。

与 Zealot 候选 vllm-ascend 基线的对比：

| 项目 | MindIE LLM 3.1.0 | vLLM 0.30.0 + vllm-ascend 0.30.0 |
| --- | --- | --- |
| CANN | 9.0.0 / 9.0.1 / 9.1.0 | 9.2.1 |
| HDK | 文档未固定版本，要求装配套驱动/固件 | 26.2.RC1 |
| PyTorch / torch_npu | 2.9.0 / 2.9.0.post4（910B 镜像） | 2.10.0 / 2.10.0.post7 |
| Python | 3.11 | 3.12（官方镜像） |
| 维护状态 | 冻结，仅维护 | 活跃，约 2 周一个小版本 |

vLLM 侧数据来自已解决票据《调查 vLLM 与 vllm-ascend 的候选稳定基线》及其报告：<https://github.com/0xnicholas/Zealot/blob/research/vllm-ascend-baseline/research/vllm-ascend-baseline.md>。

**这是路线比较中的硬约束**：按官方口径，MindIE LLM 3.1.0 的 CANN 支持列到 9.1.0，vllm-ascend 0.30.0 要求 CANN 9.2.1，两者没有交集。如果 Zealot 要在同一台 A2 上同时保留两条路线，需要两套容器/两套 CANN，不能“装一套、随时切换”。[官方/推断]

### 4.4 驱动与固件

安装文档只要求宿主机安装与产品型号匹配的 NPU 驱动和固件，并给出 `Ascend-hdk-<chip_type>-npu-driver` / `-npu-firmware` 的安装命令，没有在 MindIE 文档中固定一个 HDK 版本号。[官方]

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/install/source/image_usage_guide.md>。

因此：**HDK 的精确配套版本是 [未知]**，需要在验证机上以 CANN 9.0.1/9.1.0 对应的官方驱动矩阵为准。这一点与 vllm-ascend 明确写 HDK 26.2.RC1 不同。

## 5. OpenAI 兼容在线服务与首发相关功能

### 5.1 接口面

MindIE LLM 的 Server 通过 Endpoint 模块同时兼容 Triton / OpenAI / TGI / vLLM 的请求格式。与 Zealot 首发相关的接口：[官方]

| 接口 | 方法 | 说明 |
| --- | --- | --- |
| `/v1/chat/completions` | POST | OpenAI 文本/流式推理 |
| `/v1/completions` | POST | vLLM 兼容 OpenAI 文本/流式推理 |
| `/v1/models` | GET | 模型列表 |
| `/v1/load_lora_adapter` | POST | 动态加载 LoRA |
| `/v1/unload_lora_adapter` | POST | 动态卸载 LoRA |
| `/generate` | POST | TGI/vLLM 风格生成 |
| `/infer`、`/infer_token` | POST | 原生接口 |
| `/health`、`/metrics`、`/metrics-json` | GET | 健康与指标 |
| `/stopService` | GET | 优雅退出 |

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/user_manual/service_APIs_usage_guidance.md>。

没有看到 `/v1/embeddings`、`/v1/responses`、`/v1/messages` 等接口；首发只做文本生成，这一点够用。[官方]

### 5.2 “OpenAI 兼容”的实际口径

配置项 `openAiSupport` 的默认值是 `"vllm"`：[官方]

> 取值为 "vllm" 或者配置字段缺失时，代表 `/v1/chat/completions` 接口使用 vLLM 兼容的 OpenAI 接口版本；取值为其他字符时，代表使用原生 OpenAI 接口版本。

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/user_manual/service_parameter_configuration.md>。

这意味着 MindIE 默认并不是“OpenAI 原生语义”，而是“vLLM 兼容方言”。对 Zealot 来说这反而是好事：同一个客户端在 vLLM 与 MindIE 之间迁移时，默认行为更接近；但也说明 MindIE 的 OpenAI 兼容性是通过 vLLM 方言实现的，不能假定它与 OpenAI 官方 API 的边角行为完全一致。[官方/推断]

### 5.3 连续批处理与调度

MindIE LLM 的架构文档明确写了 Continuous Batching、PagedAttention、FlashDecoding，并把调度器描述为“将多条请求在 Prefill 或 Decode 阶段组成 Batch”。[官方]

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/developer_guide/architecture_design/architecture_overview.md>、<https://mindie-llm-doc.readthedocs.io/zh-cn/latest/user_guide/user_manual/introduction/>。

调度相关的关键默认值（`config.json` 的 `ScheduleConfig`）：[官方]

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `maxSeqLen` | 2560 | 最大序列长度；**默认很小**，上 32k/128k 必须显式调大 |
| `maxInputTokenLen` | 2048 | 最大输入 token 数 |
| `maxIterTimes` | 512 | 全局最大输出长度 |
| `maxBatchSize` | 200 | 最大 decode batch size，上限 819200 |
| `maxPrefillBatchSize` | 50 | 最大 prefill batch size |
| `maxPrefillTokens` | 8192 | 单次 prefill 的总 token 上限 |
| `maxQueueDelayMicroseconds` | 5000 | 队列最大等待时间 |
| `maxFirstTokenWaitTime` | 2500 ms | 首 token 最长排队时间 |

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/user_manual/service_parameter_configuration.md>。

这些默认值说明：**默认配置的 `maxSeqLen` 只有 2560**，不是模型支持的 128k。要跑 Zealot 的在线场景，必须改 `maxSeqLen`、`maxInputTokenLen`、`maxIterTimes`，并重新估算 `npuMemSize` 与 `maxBatchSize`。这是部署复杂度的一部分，不是“拉起就能用”。[官方/推断]

### 5.4 Chunked prefill：SplitFuse

MindIE 把 chunked prefill 叫 **SplitFuse**：把长 prompt 拆成小块，在多个 forward step 中调度，最后一块完成后才开始生成；短 prompt 也会被切分以填满 step。[官方]

限制与约束：[官方]

- 仅 Atlas 800I A2 / A3 支持。
- 支持 LLaMA3.1-70B 浮点、Qwen2、Qwen2.5、Qwen3 系列。
- 量化只支持 W8A8。
- **不能与 Multi-LoRA、Function Call、并行解码、MTP、长序列特性同时使用。**
- 支持 `n`、`best_of`、`use_beam_search` 后处理参数。

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/feature/split_fuse.md>。

对 Zealot 的含义：如果首发要“chunked prefill + Function Call + 长序列 + Multi-LoRA”同时开，MindIE 的组合能力不够；vLLM 侧的组合限制不同，需要逐项对照，不能假定等价。[官方/推断]

### 5.5 Prefix Cache

Prefix Cache 通过哈希表保留 session 结束后的 KV Cache，新请求命中相同 token 前缀时复用，减少 prefill 时间。[官方]

限制与约束：[官方]

- Atlas 800I A2 / 300I Duo / A3 支持。
- Qwen2、Qwen2.5、Qwen3、DeepSeek-R1、DeepSeek-V3/V3.1 支持。
- 公共前缀长度必须 ≥ block size 才复用；按 block 粒度（如 128）对齐。
- 支持的量化：W4A8、W8A8、PDMIX、稀疏量化；其他量化不支持。
- **不能和 Multi-LoRA 同时使用。**
- **不能和 SplitFuse、数据并行同时开启**（3.1.0 版本说明进一步限定为 PD 混部场景；PD 分离不受此限制）。
- 不建议在复用率低的场景开启。

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/feature/prefix_cache.md>、<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/release_notes.md>。

注意官方文档内部有表述冲突：Prefix Cache 页既说“可以和 SplitFuse 同时使用”，又说“prefix cache 不支持和 splitfuse 以及数据并行同时开启”；Qwen 特性矩阵则把二者标为 ❌。以**特性矩阵与 3.1.0 版本说明的约束为准**：Qwen 上二者不兼容，PD 混部下还有额外的 DP 限制。[官方/推断]

### 5.6 其他首发相关特性

| 特性 | 状态 | 对 Zealot 的意义 |
| --- | --- | --- |
| 异步调度 | 支持；Qwen 矩阵 ✅ | 用推理耗时掩盖数据准备/返回，降低时延 |
| SLO 感知调度 | 支持；Qwen 矩阵 ✅ | 按 TTFT/TPOT 目标调吞吐，但需要调参 |
| PD 混部 / PD 分离 | 支持 | MindIE LLM 单卡用混部；PD 分离需要多实例/多卡 |
| 并行解码 / MTP | 支持，但与 SplitFuse 等互斥 | 首发不需要 |
| Multi-LoRA | Qwen3-8B 支持 ✅ | 与 Prefix Cache、SplitFuse 互斥 |
| Function Call | 部分模型支持 | 官方模型清单不含 Qwen3-8B（见 §9） |
| 结构化输出 | 3.0.0 起支持 `response_format`（json_object / json_schema） | 对 JSON 输出场景有用 |
| 思考解析 / 思考预算 | 支持 | Qwen3 的 thinking 输出可结构化 |
| KV Cache 池化 | 支持 | 首发不需要 |

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/feature/README.md>、<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/feature/structured_output.md>。

### 5.7 安全与鉴权

- `httpsEnabled` 默认 **true**，默认要求配置 CA、服务证书、私钥、CRL、KMC 密钥库；支持双向 TLS。[官方]
- 在已核对的参数页中没有看到 OpenAI 风格的 `Authorization: Bearer <api-key>` 鉴权配置；安全模型看起来是 mTLS + 网络隔离，而不是 API key。[官方/推断]
- 文档提供了完整的安全加固章节（umask、ASLR/KASLR、Docker/K8s 加固、防火墙、防 DoS 等）。[官方]

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/user_manual/service_parameter_configuration.md>、<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/install/security_management_and_hardening/security_management.md>。

对 Zealot 的含义：如果首发需要“API key + TLS 终止 + 多租户”，MindIE 的 Server 不直接提供 API key 层，需要前置网关；vLLM 的 OpenAI server 则通常由网关或 `--api-key` 处理。这不是决定性差异，但会影响部署拓扑。[推断]

## 6. 单节点容器与内网部署

### 6.1 官方镜像方式

官方推荐镜像安装：镜像内已包含 CANN、FrameworkPTAdapter、MindIE 与 ATB Models，拉起容器即可。[官方]

启动命令的关键点（Atlas 800I A2/A3）：[官方]

- `--net=host`
- `--shm-size=500g`（多模态高并发建议 ≥100g）
- 挂载 `/dev/davinci0..N`、`/dev/davinci_manager`、`/dev/hisi_hdc`、`/dev/devmm_svm`
- 只读挂载 `/usr/local/Ascend/driver`、`/usr/local/dcmi`、`/usr/local/bin/npu-smi`、`/usr/local/sbin`
- 挂载权重目录
- `--device` 权限必须是 `rwm`，否则同卡有其它任务时 `npu-smi` 会报错、`torch.npu.set_device()` 会失败

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/install/source/image_usage_guide.md>。

A2 是 aarch64，容器镜像必须选 arm64；3.1.0 的 910B 官方 tag 只有 openEuler 24.03 LTS。[官方]

### 6.2 离线 / 内网

- 镜像可以导出为 tar，在目标机 `docker load`；这是内网运行的标准做法。[官方]
- 也可以走离线 whl/run 安装：CANN Toolkit、CANN ops、CANN NNAL、torch_npu、apex、MindIE LLM whl、ATB-Models whl、MindIE Motor whl 等，全部从昇腾下载中心获取。[官方]
- 从源码构建镜像需要访问 Atlas OBS 与华为云 PyPI，不适合完全隔离的环境。[官方]
- 权重需要提前下载并挂载；`trustRemoteCode` 默认 false，开启后可能触发 Hugging Face 下载，内网应保持 false。[官方]

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/install/source/preparing_software_and_dependencies.md>、<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/communication_matrix.md>。

### 6.3 单 NPU / 单节点适配度

- Qwen3-8B 的官方 world size 是 1，单卡可跑；单节点单 NPU 是支持场景。[官方]
- MindIE LLM 的 Server 是单进程多线程 + C++ 引擎，单卡不需要 K8s；用 `config.json` + `mindieservice_daemon` 或 `mindie_llm_server` 拉起即可。[官方]
- 但官方快速入门示例用的是 4 卡 Qwen2-7B，单卡 Qwen3-8B 的完整配置需要自己改 `npuDeviceIds`、`worldSize`、`maxSeqLen` 等。[官方/推断]

### 6.4 MindIE Motor 的单节点形态

- Motor 的单容器部署是 **PD 分离**：1 台服务器、1 个容器，容器内同时跑 Prefill 与 Decode，`--devices 0,1`，适用于权重 <100GB 的小模型，不依赖 K8s。[官方]
- Motor 的 PD 混部（union）是 K8s 形态：需要 MindCluster InferServiceSet CRD、controller、参数面 RoCE 互联，示例 `hybrid_pod_npu_num` 为 2。[官方]
- 官方快速开始用 A2 8 卡、Qwen3-8B、P/D 各 1 实例、每实例 TP2，说明 Motor 的目标场景是多卡/多实例，不是单 NPU。[官方]

来源：<https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/deployment/docker/single_container/>、<https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/deployment/k8s/pd_aggregation_deployment/>、<https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/quick_start/>。

**结论**：首发单节点单 NPU 用 MindIE LLM 是合适的；用 Motor 则是过度设计（官方单容器 PD 分离示例为 2 卡，混部还要 K8s + RoCE）。Motor 的价值要到多实例、PD 分离、弹性扩缩容阶段才会出现。

## 7. 公开性能证据

### 7.1 没有找到可复核的 Qwen3-8B BF16 TP1 A2 官方基准

本次检索没有找到华为官方发布的“Atlas 800I A2 单卡 + Qwen3-8B BF16 + MindIE LLM 3.1.0”吞吐/时延基准。[未知]

本次查到的官方文档中，唯一带性能数字的例子是快速入门的 **Qwen2-7B、4 卡、并发 1** 冒烟测试：[官方]

| 指标 | 值 |
| --- | --- |
| E2EL | 2048.3 ms |
| TTFT | 50.3 ms |
| TPOT | 10.7 ms |
| OutputTokenThroughput | 115.9 token/s |
| Max Concurrency | 1 |

来源：<https://mindie-llm-doc.readthedocs.io/zh-cn/latest/user_guide/quick_start/quick_start/>。

这组数字只能证明“服务能跑通”，不能作为容量或吞吐依据：模型不是 Qwen3-8B、卡数是 4、并发是 1，且没有说明输入/输出长度分布。

### 7.2 官方推荐的性能测试方法

- 官方使用 **AISBench** 做精度与性能测试；镜像内已集成，`pip show ais_bench_benchmark` 可查路径。[官方]
- 性能测试关注 TTFT、TPOT、Request Throughput、Output Token Throughput。[官方]
- 性能调优手段：CPU 高性能模式（约 3% TPS 提升）、透明大页、jemalloc、绑核、`maxBatchSize`/`maxPrefillTokens`/`prefillChunkSize` 调参。[官方]

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/quick_start/quick_start.md>、<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/user_manual/performance_tuning.md>。

### 7.3 对 Zealot 的含义

- 不能用“官方说高性能”作为选型依据；必须用同一台 A2、同一模型、同一输入/输出分布，分别跑 vLLM 与 MindIE，才能比较。[推断]
- 如果要做对照，建议用 AISBench 或 Zealot 自己的压测客户端，固定 `maxSeqLen`、`maxBatchSize`、`maxPrefillTokens`、prefix cache 开关、SplitFuse 开关，记录 TTFT/TPOT/P90/P99。[推断]
- 由于 MindIE 3.1.0 已冻结，即使一次对照显示它更快，也只具有“参考”价值，不能作为长期基线。[推断]

## 8. 自定义模型、算子与调度扩展能力

### 8.1 自定义模型

MindIE LLM 的模型后端是 ATB Models。官方提供《Torch-like 组图模型迁移指南》，说明如何用类似 `torch.nn.Module` 的 API 在 ATB 上组图，并已用 Qwen2/Qwen3/Llama 做过迁移实现。[官方]

- 迁移方式：调用 `atb_llm.layer` / `atb_llm.nn` 中的 torch-like API 构建 attention、MLP、decoder layer，最后用 `torch_to_mindie_graph` 转成静态图。
- 这不是“拿 Hugging Face 权重直接跑”：需要理解 ATB 算子、图切分、图融合和权重切分。
- 对于 Qwen3-8B 这类已支持模型，不需要迁移；对于新模型，迁移成本明显高于 vLLM 的 `transformers` 接入。

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/docs/torcklike_model_migration_guide.md>。

### 8.2 自定义算子

官方提供 `mie_ops`（Ascend C 自定义算子）适配指南：[官方]

- 源码目录 `src/kernels/mie_ops`，算子包按芯片编译（A2 为 `mie_ops_ascend910b`）。
- 通过 `torch_ops_extension/ops_def_registration.cpp` 注册 torch 接口，调用方式为 `torch.ops.mie_ops.<name>`。
- 3.1.0 的性能更新里也提到用自定义 NPU 算子替代 PyTorch 软件实现（Top-K/Top-P 采样、Triton RoPE）。

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/developer_guide/migration_and_adaptation_guide/mie_ops_adaptation_guide.md>、<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/release_notes.md>。

这说明 MindIE 在算子层是**可扩展的**，但扩展语言是 Ascend C + ATB，门槛高于 vLLM 的 Python/CUDA 插件模型。

### 8.3 调度扩展

- MindIE LLM 的调度器是 C++ 实现，README 列出 FCFS / PDDS / Layerwise 等调度策略；`config.json` 暴露 `prefillPolicyType`、`decodePolicyType`、`supportSelectBatch`、SLO 参数等。[官方]
- 没有找到“用户自定义引擎调度器插件”的公开接口文档；可调的是策略参数，不是调度算法本身。[官方/未知]
- MindIE Motor 提供“自定义选点插件”（`scheduler_policy_plugin`），但那是**请求到实例的路由选点**，不是引擎内部的 batch 调度。[官方]

来源：<https://gitcode.com/Ascend/MindIE-LLM>、<https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/features/inference_scheduling/scheduler_policy_plugin/>。

### 8.4 服务集成

MindIE LLM 对外提供 C++ 与 Python API（LLM Manager API），可用于把引擎嵌入自有服务；Server 是它的一个内置消费者。[官方]

来源：<https://mindie-llm-doc.readthedocs.io/zh-cn/latest/user_guide/user_manual/introduction/>。

### 8.5 扩展能力结论

MindIE 的扩展点是“ATB 图 + Ascend C 算子 + C++ 调度参数”，适合华为/伙伴做深度芯片优化；对 Zealot 这种以 Python 为主、需要快速迭代模型与调度策略的项目，扩展成本高，且引擎已冻结，投入无法跟随上游演进。[推断]

## 9. 已知限制与风险

### 9.1 冻结是最大的风险

- 3.1.0 之后不再支持新模型、新特性、新镜像。[官方]
- 维护周期标注三个月，冻结说明没有承诺安全补丁期限。[官方]
- 如果 Zealot 首发后想升级到 Qwen3.5、Qwen4 或新的量化格式，MindIE 路线没有官方升级路径。[推断]

### 9.2 并发缺陷（社区报告，未官方确认）

GitCode issue #718 是一份针对 MindIE-LLM 的并发安全审查，声称在 `v3.1.0`（commit `981ea7cf`）上确认 **11 项缺陷（高 3 / 中 5 / 低 3）**，包括：[社区]

- `LiveInferContext::GetInstance` 的破碎双检锁导致 `shared_ptr` 撕裂读/UAF；
- `batch_latency_tracker` 的无锁 `std::deque` 并发读写；
- 响应输出管线的 `seqGroup` 竞态、最后一条响应丢失；
- PD 分离 Python 引擎的 `pull_blocks`/`unlink` 跨临界区问题；
- Ascend C 采样算子的 workspace 布局缺陷。

issue 声称这些问题在 3.1.0 分支上未被修复，并给出最小复现（5/5 段错误）。截至 2026-11-25，issue 状态为 open。[社区]

来源：<https://gitcode.com/Ascend/MindIE-LLM/issues/718>。

评价：这是第三方审计，不是华为官方公告，**不能当作已证实的产品缺陷**；但它的严重级别、可复现描述和版本对应关系足够具体，足以把“高并发稳定性”列为 MindIE 路线的**待验证风险**。如果 Zealot 要认真考虑 MindIE，必须在验证机上做长稳压测与故障注入，而不是只看功能列表。[推断]

### 9.3 官方文档内部不一致

本次核对发现至少四处：

1. **Function Call 模型清单**：`function_call.md` 列出的支持模型是 ChatGLM3-6B、Qwen3-32B、Qwen3-235B-A22B、Qwen3-30B-A3B、DeepSeek-R1-0528、Qwen2.5-Instruct、DeepSeek-V3.1 系列，**不含 Qwen3-8B**；但 Qwen 模型 README 的矩阵把 Qwen3-8B 的 functioncall 标为 ✅。[官方]
2. **Prefix Cache 与 SplitFuse**：同一页面既说“可以和 SplitFuse 同时使用”，又说“不支持和 splitfuse 以及数据并行同时开启”；Qwen 特性矩阵标 ❌，3.1.0 版本说明又限定为 PD 混部场景。[官方]
3. **CANN 配套**：安装文档的“版本配套”表写 MindIE 3.0.0 ↔ CANN 8.5.1，版本说明写 3.0.0 支持 CANN 9.0.0/8.5.1/8.5.0；两处不一致，且安装文档明显滞后。[官方]
4. **Qwen3-8B 的量化**：3.1.0 版本说明称“Qwen3 系列模型支持 LoRA+int8 量化叠加”，但模型矩阵中 Qwen3-8B 的 W8A8/W4A8/W8A16 全部标 ❌。[官方]

对 Zealot 的含义：MindIE 的文档不能当作“合同级”支持承诺；关键能力（尤其是 Qwen3-8B 的 Function Call）必须实测确认。[推断]

### 9.4 特性组合限制

- Qwen 上 Prefix Cache × SplitFuse × DP>1 不能同时开（PD 混部）。[官方]
- SplitFuse 不能与 Multi-LoRA、Function Call、并行解码、MTP、长序列同开。[官方]
- Prefix Cache 不能与 Multi-LoRA 同开。[官方]
- Qwen3-8B 不支持 aclgraph，只支持 atbgraph。[官方]

如果 Zealot 的未来路线需要“长上下文 + 工具调用 + 多 LoRA + chunked prefill”组合，MindIE 的组合空间比 vLLM 窄。[推断]

### 9.5 默认配置与运维复杂度

- `maxSeqLen` 默认 2560、`maxInputTokenLen` 2048、`maxIterTimes` 512，开箱即用的上下文远小于模型能力。[官方]
- `config.json` 参数多，且 `npuMemSize`、`maxBatchSize`、`maxPrefillTokens`、`prefillChunkSize` 之间相互影响；官方建议用 AISBench 压测后调参。[官方]
- HTTPS 默认开启，需要证书、私钥、CRL、KMC 密钥库；首次部署门槛高于 vLLM 的 `--api-key`/HTTP 默认。[官方]

### 9.6 版本行与镜像获取

- 3.1.0 的 910B 官方镜像只有 CANN 9.0.1 + openEuler 24.03 LTS + py3.11；如果验证机是 Ubuntu，需要离线安装或自建镜像。[官方]
- 镜像下载需要华为账号登录；镜像许可条款未公开可核。[官方/未知]
- 3.1.0 之后不再发新镜像，意味着不会再有 CANN 9.2.x 的官方 MindIE LLM 镜像。[官方]

### 9.7 生命周期

- run 包部署方式将于 2027-03 日落，Torch 2.1.0 与 abi0 包随之日落；未来只剩 whl。[官方]
- Qwen2 系列、Qwen1.5 系列等模型将于 2027-03 日落；Qwen3 不在日落清单内，但也不会有新模型加入。[官方]

来源：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/release_notes.md>。

### 9.8 安全维护不确定性

- 冻结说明只写“维持维护状态”，没有说安全漏洞是否继续修复、修复多久。[官方]
- 对比 vLLM：vLLM 有明确的版本升级路径（0.30.0 → 0.33.0+）来处理已知安全公告；MindIE 没有后继版本可升。[推断]

## 10. 与 vLLM + vllm-ascend、DeepInfer 的公平比较

| 维度 | MindIE LLM 3.1.0 | MindIE Motor 3.2.0 | vLLM 0.30.0 + vllm-ascend 0.30.0 | DeepInfer |
| --- | --- | --- | --- | --- |
| 定位 | 昇腾原生 LLM 推理引擎 + Server | vLLM/SGLang 之上的服务编排/调度层 | 通用高吞吐推理引擎 + Ascend 插件 | DeepSeek-V4.1 专用离线推理参考 |
| 是否自有引擎 | 是 | 否（对接 vLLM/SGLang） | 是 | 是（vLLM 深度分叉） |
| A2 / 910B | 官方支持 | 官方支持（A2 32/64GB） | 官方支持，A2 是默认场景 | 仅 950DT 验证，910 未验证 |
| Qwen3-8B BF16 | 官方充分验证，单卡 | 取决于后端 vLLM | 官方 ✅，TP1 | 不支持（仅 DeepSeek-V4.1） |
| OpenAI 兼容 | `/v1/chat/completions`、`/v1/completions`、`/v1/models`，默认 vLLM 方言 | 兼容 OpenAI Chat/Responses/Completion + Anthropic | 内置 OpenAI server | 无在线服务层 |
| 连续批处理 | 支持 | 由 vLLM/SGLang 提供 | 支持 | 离线 batch |
| Chunked prefill | SplitFuse，组合限制多 | 由后端提供 | 支持 | 不适用 |
| Prefix cache | 支持，与 SplitFuse/Multi-LoRA 互斥 | 由后端提供 | 支持 | 不适用 |
| 单 NPU | 支持（world size 1） | 官方单容器 PD 分离示例为 2 卡；混部用 K8s | 支持 | 不适用 |
| PD 分离 | 支持 | 核心能力 | 需外置编排 | 有设计，未开源服务层 |
| 扩展性 | ATB 图 + Ascend C + C++ 参数 | 选点插件、K8s CRD | Python 插件、模型生态、上游社区 | 极深耦合，难复用 |
| 维护状态 | **冻结，仅维护** | 活跃（2026-10 仍在发版） | 活跃，约 2 周一个小版本 | 研究项目，非上游 |
| 许可证 | Mulan PSL v2（代码） | Mulan PSL v2（代码） | Apache-2.0 | Apache-2.0（DeepEP-Ascend 无许可证） |
| 分发 | 昇腾镜像仓库，需华为账号 | 昇腾镜像仓库，需华为账号 | `quay.io/ascend/vllm-ascend` 公开镜像 | GitHub 源码 |
| 对 Zealot 的适配 | 可做一次性对照，不做基线 | 未来多实例/PD 可选 | **基线** | 只作参考，不作上游 |

vLLM 侧版本与支持矩阵来自票据 01/02 及其报告；DeepInfer 侧来自票据 03 及其报告。MindIE 侧来源见 §13。

### 10.1 公平比较的要点

- **不要拿“官方”当“更好”。** MindIE 的官方支持确实覆盖 A2 + Qwen3-8B BF16，但“支持”不等于“更快”或“更稳”；本次没有找到可复核的性能证据，且存在未解决的并发缺陷报告。
- **不要拿“开源”当“可维护”。** MindIE LLM 是 Mulan PSL v2 开源，但已冻结；vLLM 是 Apache-2.0 且活跃。许可证宽松不代表生命周期健康。
- **不要拿“功能列表”当“组合能力”。** MindIE 的功能很多，但 Qwen 上 Prefix Cache、SplitFuse、Multi-LoRA、Function Call 之间存在互斥；vLLM 的组合限制不同，必须逐项对照。
- **Motor 不是第三条引擎路线。** 它和 vLLM 不是竞争关系；Motor 3.2.0 的兼容表包含 vLLM-Ascend v0.30.0（也兼容 v0.18.0/v0.23.0）。把 Motor 当成“MindIE 替代 vLLM”的证据是误读。
- **DeepInfer 与 MindIE 不在同一比较层级。** DeepInfer 是 DeepSeek-V4.1 / 昇腾 950 专用的离线参考实现，没有在线服务层，910 未验证；MindIE 是通用官方引擎，但已冻结。两者都不能替代 vLLM 基线，定位分别是“设计参考”和“官方对照”。

## 11. 结论与建议

### 11.1 对票据问题的直接回答

**在当前公开证据与 2026-11 的产品状态下，MindIE 应作为“原生官方对照 + 可选的服务/调度组件”，不应作为“替代 vLLM + vllm-ascend 的基线”。**

理由按权重排序：

1. 唯一能替代 vLLM 的组件 MindIE LLM 已被官方冻结，官方推荐路径本身就是 Motor + vLLM Ascend。[官方]
2. 按官方口径，MindIE LLM 3.1.0 的 CANN 支持列到 9.1.0，vllm-ascend 0.30.0 要求 CANN 9.2.1，两者没有交集，无法共用一套环境。[官方]
3. 功能组合与文档一致性不足以支撑“首发唯一基线”的承诺；Qwen3-8B 的 Function Call 甚至存在官方资料冲突。[官方]
4. 存在未解决的并发缺陷报告，高并发稳定性需要额外验证。[社区]
5. 没有可复核的 Qwen3-8B BF16 TP1 A2 性能证据，无法证明它比 vLLM 更适合首发。[未知]

### 11.2 建议的落地方式

| 选项 | 建议 | 理由 |
| --- | --- | --- |
| 把 MindIE LLM 3.1.0 作为首发基线 | **不做** | 冻结、CANN 行不一致、无升级路径、并发风险 |
| 把 MindIE LLM 3.1.0 作为官方对照/一次性基准 | **可做，低优先级** | 可校准 vLLM 在 A2 上的性能上界；需华为账号获取镜像，成本可控 |
| 把 MindIE Motor 3.2.0 纳入首发 | **不做** | 单 NPU 用不上；官方单容器 PD 分离示例为 2 卡，混部要 K8s + RoCE |
| 把 MindIE Motor 作为未来多实例/PD 的服务层 | **保留观察** | 兼容表包含 vLLM-Ascend 0.30.0；等 Zealot 需要 PD/多实例时再评估 |
| 参考 MindIE 的支持矩阵/特性文档 | **可做** | 官方对 Ascend 能力边界的描述，可用于校验 Zealot 的支持承诺 |
| 复用 MindIE 的 AISBench 测试方法 | **可做** | 官方工具，适合做 vLLM/MindIE 对照压测 |

### 11.3 如果要做一次对照基准，最小方案

1. 在首发验证机（A2 64GB，单卡）上，用官方 3.1.0 910B 镜像（openEuler 24.03）或离线 whl 安装，跑 Qwen3-8B BF16，`worldSize=1`。
2. 固定 `maxSeqLen=32768`、`maxInputTokenLen=32768`、`maxIterTimes=2048`、`maxBatchSize` 与 `maxPrefillTokens` 按显存调；分别记录 Prefix Cache 开/关、SplitFuse 开/关。
3. 用 AISBench 或 Zealot 压测客户端，固定输入/输出长度分布，记录 TTFT/TPOT/P90/P99/Output Token Throughput。
4. 在**同一台机器、同一模型、同一分布**上跑 vLLM 0.30.0 + vllm-ascend 0.30.0（CANN 9.2.1），重复同样的指标。
5. 只有当中位吞吐差距超过一个事先约定的阈值（例如 20%）且稳定性不劣于 vLLM 时，才值得把 MindIE 重新提上决策桌；即便如此，冻结问题仍不解决。

### 11.4 改变结论的条件

- 华为发布 MindIE LLM 3.2+，恢复新模型/新特性，并支持 CANN 9.2.x 或更高。
- 官方发布可复核的 A2 + Qwen3-8B BF16 基准，显示显著优于 vLLM。
- Zealot 需要 PD 分离/多实例编排，且 Motor 的收益超过其 K8s/RoCE 复杂度。
- 社区并发缺陷被官方确认并修复，或经 Zealot 长稳压测证明不影响首发场景。

## 12. 未知与待验证

| 项 | 状态 | 说明 |
| --- | --- | --- |
| MindIE 镜像的许可条款 | 未知 | AscendHub 需登录，公开页未提供许可正文；不能确认再分发/商用条件 |
| HDK / 驱动精确配套版本 | 未知 | MindIE 文档只要求“配套驱动/固件”，未固定版本号 |
| Qwen3-8B 的 Function Call 是否真的支持 | 冲突 | `function_call.md` 不含 Qwen3-8B，模型 README 标 ✅；需实测 |
| 并发缺陷报告的真实性 | 未确认 | issue #718 为第三方审计，官方未回应；需长稳压测 |
| Qwen3-8B BF16 TP1 A2 的实际性能 | 未知 | 无公开可复核基准；需自测 |
| 冻结后安全补丁策略 | 未知 | 只写“维护状态”，无期限承诺 |
| 华为商用版 MindIE 是否另有路线 | 未知 | 本次只核实社区版文档；不能排除商业版另有演进 |
| MindIE LLM 3.1.0 在 CANN 9.2.1 上能否非官方运行 | 未知 | 官方矩阵未列，未测试 |
| Motor 3.2.0 镜像是否同样需要华为账号 | 推断为是 | 与 MindIE LLM 同一镜像仓库入口，未见公开 registry |

## 13. 来源清单

### 官方文档与代码仓

- MindIE LLM 文档站：<https://mindie-llm-doc.readthedocs.io/zh-cn/latest/>
- MindIE LLM 模型支持列表：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/model_support_list.md>
- MindIE LLM 3.1.0 冻结说明：<https://gitcode.com/Ascend/MindIE-LLM/blob/v3.1.0/docs/zh/user_guide/user_manual/introduction.md>
- MindIE LLM 版本说明：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/release_notes.md>
- MindIE LLM 镜像 tag：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docker/support_tag.md>
- MindIE 镜像总览：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docker/OVERVIEW.md>
- MindIE LLM 许可证：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/LICENSE.md>
- 镜像安装指南：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/install/source/image_usage_guide.md>
- 软件包与依赖：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/install/source/preparing_software_and_dependencies.md>
- 操作系统支持：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/install/installation_introduction.md>
- 服务化接口：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/user_manual/service_APIs_usage_guidance.md>
- 服务参数配置：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/user_manual/service_parameter_configuration.md>
- 特性总览与叠加矩阵：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/feature/README.md>
- Prefix Cache：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/feature/prefix_cache.md>
- SplitFuse：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/feature/split_fuse.md>
- Function Call：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/feature/function_call.md>
- 结构化输出：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/feature/structured_output.md>
- Qwen 模型 README：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen/README.md>
- Torch-like 模型迁移：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/docs/torcklike_model_migration_guide.md>
- mie_ops 自定义算子：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/developer_guide/migration_and_adaptation_guide/mie_ops_adaptation_guide.md>
- 架构设计：<https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/developer_guide/architecture_design/architecture_overview.md>
- 昇腾社区 MindIE 产品页：<https://www.hiascend.com/developer/software/mindie>
- 产品公告（架构升级与模块更名）：<https://www.hiascend.com/productbulletins/detail/814>
- 昇腾镜像仓库 MindIE：<https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f>

### MindIE Motor 官方文档

- 架构：<https://mindie-motor.readthedocs.io/zh-cn/latest/architecture/>
- 版本说明 3.2.0：<https://mindie-motor.readthedocs.io/zh-cn/latest/release_note_motor/>
- 支持的推理引擎：<https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/supported_inference_engines/>
- 特性支持列表：<https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/features/supported_features/>
- 单容器部署：<https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/deployment/docker/single_container/>
- PD 混部部署：<https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/deployment/k8s/pd_aggregation_deployment/>
- 快速开始：<https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/quick_start/>
- 服务接口：<https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/api/service_interfaces/>
- Motor 许可证：<https://gitcode.com/Ascend/MindIE-Motor/blob/master/LICENSE.md>

### 社区报告

- MindIE-LLM 并发缺陷报告（open）：<https://gitcode.com/Ascend/MindIE-LLM/issues/718>
- 暂停演进说明文档 issue（closed）：<https://gitcode.com/Ascend/MindIE-LLM/issues/725>
- 镜像 HTTPS 报错 issue（closed）：<https://gitcode.com/Ascend/MindIE-LLM/issues/726>

### Zealot 内部依据

- vLLM/vllm-ascend 基线票据：<https://github.com/0xnicholas/Zealot/blob/research/vllm-ascend-baseline/research/vllm-ascend-baseline.md>
- 昇腾 910 与 Qwen 支持票据：<https://github.com/0xnicholas/Zealot/blob/research/ascend-qwen-support/research/ascend-qwen-support.md>
- DeepInfer 可借鉴性票据：<https://github.com/0xnicholas/Zealot/blob/research/deepinfer-reusability/research/deepinfer-reusability.md>
