# Phase 2 生命周期：完整可运行链路（M4 §12-8，issue #11）

本文档描述 Zealot 独立进程（`zealot-backend`）从构建到关闭的完整可复现链路。
按本文档操作应能复现一条真实推理链路（dev-mode CPU runner，GPU-free）。

> 链路验证状态：**M4 收官时全链路实测通过**（`/tmp/tiny-llama-local` 本地模型，
> macOS x86_64 + Python 3.12 venv）。验证过程记录于 issue #11。

---

## 0. 链路总览

```
cargo build
   │
   ▼
zealot-backend ── 启动（gRPC :9091 + HTTP :8081，端口可配）
   │
   ▼
gRPC LoadModel（model_id = HF repo id 或本地目录）
   │   └─ PyO3 嵌入 CPython → torch AutoModelForCausalLM.from_pretrained
   ▼
gRPC Infer（双向流） ──┐
   │                   │ 两种请求面
HTTP /v1/chat/completions（非流式 JSON / SSE 流式）──┘
   │
   ▼
gRPC UnloadModel（engine actor 收 Shutdown）
   │
   ▼
Ctrl-C 退出进程
```

## 1. 前置条件

| 依赖 | 版本 | 说明 |
|------|------|------|
| Rust | stable | `protoc` 由 `protoc-bin-vendored` 自动提供，无需手动安装 |
| Python | 3.12（pyo3 0.23 最高支持 3.13，本机默认 3.14 不可用） | `.cargo/config.toml` 已固定 `PYO3_PYTHON=python3.12` |
| torch / transformers | torch 2.2.2 / transformers 4.46.3 / numpy 1.26.4 | 版本 pin 原因：torch 2.2.2 是 x86_64 macOS 最后有 wheel 的版本；transformers ≥4.47 要求 torch≥2.4；torch 2.2 不兼容 numpy 2.x |
| 模型 | LLaMA 系架构（见 §6 限制） | 可用 HF repo id（需网络）或本地目录（离线可用） |

## 2. 构建

```bash
# 开发构建（测试/调试用）
cargo build

# release 构建（benchmark 用）
cargo build --release

# Python 扩展（pytest 集成测试用，可选）
python3.12 -m venv .venv
.venv/bin/pip install maturin pytest
.venv/bin/maturin develop
```

## 3. 准备推理环境（CPU dev-mode）

```bash
# venv 内安装 torch 栈（国内网络建议加镜像参数）
.venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  "torch==2.2.2" "transformers==4.46.3" "numpy==1.26.4"
```

模型两种来源：

```bash
# A. HF repo id（需能访问 huggingface.co；不可达时设 HF_ENDPOINT 镜像）
export ZEALOT_E2E_MODEL=hf-internal-testing/tiny-random-LlamaForCausalLM

# B. 本地目录（离线可复现，推荐验证用）——LLaMA 架构 + tokenizer.json：
#    config.json / model.safetensors / tokenizer.json / tokenizer_config.json
export ZEALOT_E2E_MODEL=/tmp/tiny-llama-local
```

本地 tiny LLaMA 模型可用 transformers 离线生成（`LlamaConfig` + `save_pretrained`，
MHA 配置，`num_key_value_heads` 不设或等于 `num_attention_heads`，见 §6 限制；
tokenizer 用 `PreTrainedTokenizerFast` 包一个 byte-level BPE `tokenizer.json`）。

## 4. 启动 zealot-backend

```bash
export ZEALOT_SITE_PACKAGES="$PWD/.venv/lib/python3.12/site-packages"
cargo run --bin zealot-backend
# 或：./target/debug/zealot-backend
```

启动成功输出（首行格式被测试解析，勿改）：

```
zealot-backend listening on 0.0.0.0:9091 (gRPC) 0.0.0.0:8081 (HTTP)
```

健康检查：

```bash
curl -s http://127.0.0.1:8081/health
# {"model":null,"ready":false,"status":"healthy"}
```

## 5. 完整请求链路

### 5.1 模型加载（gRPC LoadModel）

```bash
# grpcurl 需要 -proto（服务端未开 reflection）
grpcurl -plaintext -proto proto/runtime/v1/runtime.proto \
  -d '{"model_id":"/tmp/tiny-llama-local"}' \
  127.0.0.1:9091 runtime.v1.InferenceRuntime/LoadModel
# → {"modelId":"/tmp/tiny-llama-local","ready":true,
#    "message":"loaded on CPU (dev-mode PyTorch runner)"}
```

首次加载 HF repo id 会触发下载（数分钟）；本地目录即读即用。
加载期间 `LoadModel` 并发调用返回 `failed_precondition`；重复加载返回 `already_exists`。

### 5.2 推理（两个请求面）

**HTTP 非流式**（OpenAI 兼容）：

```bash
curl -s -X POST http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/tmp/tiny-llama-local",
       "messages":[{"role":"user","content":"Hello"}],
       "max_tokens":16,"stream":false}'
# → {"choices":[{"finish_reason":"length","message":{"content":"..."}}],...}
```

**HTTP 流式（SSE）**：

```bash
curl -s -N -X POST http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/tmp/tiny-llama-local",
       "messages":[{"role":"user","content":"Hello"}],
       "max_tokens":16,"stream":true}'
# → 逐帧 data: {...delta...}，收尾 [DONE]
```

**gRPC Infer（双向流）**：客户端发首帧（request_id / model / messages / params），
服务端回推 `StreamDelta` 帧（每 token 一帧，text 或 `<tk:N>` 回退），末帧
`FinalResponse`（usage + finish_reason）。客户端中途断开会自动取消并回收资源。

### 5.3 流式语义

- delta 帧数 == `usage.completion_tokens`（逐 token 一帧）
- 结束：`finish_reason` ∈ {`stop`, `length`, `cancel`}
- Rust 侧 tokenizer（`Tokenizer::load`，取模型目录 `tokenizer.json`）负责
  token → 文本解码；缺失时回退 `<tk:N>` 占位
- 未加载模型时请求返回 `{"error":"no model loaded"}`（HTTP）/ gRPC 错误

### 5.4 卸载与关闭

```bash
grpcurl -plaintext -proto proto/runtime/v1/runtime.proto \
  -d '{"model_id":"/tmp/tiny-llama-local"}' \
  127.0.0.1:9091 runtime.v1.InferenceRuntime/UnloadModel
# → {"unloaded":true}

# 进程退出：Ctrl-C（SIGINT）
```

`UnloadModel` 会向 engine actor 发 `Shutdown` 命令并回收模型槽位；
槽位清空后可再次 `LoadModel`（单模型架构）。

## 6. 当前形态限制（dev-mode，GPU 后形态见 §7）

| 限制 | 说明 | 影响 |
|------|------|------|
| 模型架构 | 仅 LLaMA 系命名（`self_attn.q_proj/k_proj/v_proj/o_proj`、`mlp.gate/up/down_proj`、rms_norm） | GPT-2（`c_attn` Conv1D + LayerNorm）等架构加载失败（`q_proj: AttributeError`），待后续里程碑支持 |
| GQA | loader 已提取 `num_kv_heads`，但 CPU runner 前向按全宽 KV 投影计算 | 必须用 MHA 模型（`num_key_value_heads == num_attention_heads`），GQA 模型会越界 panic |
| 单模型 | 一次只加载一个模型（slot 机制） | 换模型需先 Unload |
| 批处理 | `SubmitBatch` 仍 UNIMPLEMENTED | Scheduler 批语义后续接入 |
| Python 在 decode loop | dev-mode 的 PyModelRunner 每次前向持有 GIL | 临时形态，GPU kernel 落地后移除 |

## 7. GPU 后形态（目标，ADR-009）

- Python 仅在启动时加载权重（Model Loader），decode loop 全 Rust + CUDA
- `CudaModelRunner`（`--features cuda`）优先尝试 GPU，失败回退 CPU；
  CUDA 侧已有 tiled FlashAttention / PagedAttention（GQA）/ RMS Norm /
  embedding / transpose / SiLU gate-mul 等 kernel
- 推理入口不变：gRPC Runtime Interface + HTTP OpenAI 兼容面

## 8. 自动化复现（测试等价物）

| 测试 | 覆盖链路 | 运行条件 |
|------|---------|---------|
| `cargo test --test backend_e2e` | 启动 / health / 错误路径 / Load/Unload 生命周期 | 无 |
| `cargo test --test cpu_infer_e2e` | **完整链路**：LoadModel → Infer 流式 → usage 帧 | `ZEALOT_E2E_MODEL` + `ZEALOT_SITE_PACKAGES`（未设则跳过） |
| `cargo test --test http_e2e` | HTTP /v1/chat/completions（非流式 + SSE） | 外部 server + 模型目录 |

## 9. 环境变量速查

| 变量 | 默认 | 说明 |
|------|------|------|
| `ZEALOT_GRPC_PORT` | 9091 | gRPC Runtime Interface 端口（`0` = 随机） |
| `ZEALOT_HTTP_PORT` | 8081 | HTTP OpenAI 兼容端口（`0` = 随机） |
| `ZEALOT_SITE_PACKAGES` | 无 | venv 的 site-packages 路径，注入嵌入 CPython（torch/transformers 所在） |
| `ZEALOT_E2E_MODEL` | 无 | e2e 测试模型（HF repo id 或本地目录）；未设则 cpu_infer_e2e 跳过 |
| `HF_ENDPOINT` | huggingface.co | HF 下载镜像（如 `https://hf-mirror.com`） |

## 10. 故障排查

| 症状 | 原因 | 处理 |
|------|------|------|
| `import torch (venv? set ZEALOT_SITE_PACKAGES)` | 嵌入 CPython 找不到 torch | 设 `ZEALOT_SITE_PACKAGES` 指向 venv site-packages |
| `q_proj: AttributeError ...` | 非 LLaMA 架构模型 | 换 LLaMA 系模型（§6） |
| 越界 panic（`len is 2048 but the index is 2048`） | GQA 模型 | 换 MHA 配置（§6） |
| `couldn't connect to huggingface.co` | 网络不可达 | 设 `HF_ENDPOINT` 镜像，或用本地模型目录 |
| 端口占用 | 另一服务已监听 | 换 `ZEALOT_GRPC_PORT` / `ZEALOT_HTTP_PORT` |
