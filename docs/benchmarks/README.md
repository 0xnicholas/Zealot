# Benchmarks（M5-⑪，issue #25）

Zealot 对比基准（vs vLLM）的脚本、模板与执行说明。对比执行属平台侧（GPU 到位后），
本目录是交付物。

## 文件

| 文件 | 说明 |
|---|---|
| `../../scripts/bench_openai.py` | OpenAI 兼容面基准脚本（风格对齐 vLLM `benchmark_serving`） |
| `report-template.json` | JSON 报告模板（环境/版本/矩阵/全部指标） |
| `results/` | 报告输出目录（`report-<ts>.json` + `summary-<ts>.md`） |

## 指标口径（对齐 vLLM benchmark_serving）

- **吞吐**：聚合吞吐 = 稳定窗口内全部完成请求的输出 token 总和 / 总耗时（tokens/s）
- **TTFT**：首 token 时延（请求发出 → 首个内容 chunk 到达），P50/P95/P99
- **TPOT**：per-token 时延 = (总耗时 − TTFT) / (输出 token 数 − 1)，P50/P95/P99
- **矩阵**：并发 1/16/64 × 输入长度 ~100/1k/4k+ tokens × 固定输出 256 tokens

## 执行说明（平台侧）

### 前置

1. 同卡：zealot 与 vLLM 用**同一块 GPU**（同一台机器、同一驱动/CUDA 版本）。
2. 锁版本：记录 vLLM commit/tag（写进报告的 `environment.vllm_commit`）与
   zealot `git rev-parse HEAD`（`engine_version`）。
3. 对齐参数：**max_num_seqs 对齐**（zealot 侧以 Scheduler 容量参数对应 vLLM
   `--max-num-seqs`）；同模型同量化、同 `max_tokens`、同温度（基准用贪婪/温度 0）。

### 跑 zealot

```bash
# 1) 构建并启动后端（真实模型路径；dev 冒烟用 mock:tiny）
cargo build --release --bin zealot-backend
ZEALOT_GRPC_PORT=8080 ZEALOT_HTTP_PORT=8081 ./target/release/zealot-backend

# 2) 加载模型（gRPC LoadModel 或 dev HTTP 口）
curl -X POST http://127.0.0.1:8081/admin/load \
     -H 'Content-Type: application/json' -d '{"model":"/path/to/model"}'

# 3) 基准（预热 + 稳定窗口 ≥60s + 3 轮取中位）
python3 scripts/bench_openai.py --base-url http://127.0.0.1:8081 \
    --model /path/to/model \
    --concurrency 1,16,64 --input-len 100,1000,4096 --max-tokens 256 \
    --window 60 --rounds 3 --out-dir docs/benchmarks/results
```

### 跑 vLLM（对照）

```bash
# vLLM 侧：python3 benchmarks/benchmark_serving.py（vLLM 仓库，锁 commit）
python3 -m vllm.entrypoints.openai.api_server --model <model> \
    --max-num-seqs <与 zealot 对齐> ...
python3 benchmarks/benchmark_serving.py --backend openai \
    --base-url http://127.0.0.1:8000 --model <model> \
    --request-rate inf --num-prompts <N> \
    --max-tokens 256 --input-len 100,1000,4096 ...
```

### 汇总

1. 把 vLLM 的吞吐/TTFT/TPOT 填入 `summary-<ts>.md` 对比表，补 `vs vLLM 吞吐` 比值
   （= zealot / vLLM）。
2. 报告 JSON 中 `environment` 字段填全（gpu / engine_version / vllm_commit /
   max_num_seqs_aligned）。
3. 规则：预热充分（≥ 窗口前的满载预热）、稳定窗口 ≥60s、多轮取中位（≥3）、
   记录机器/驱动信息；异常轮次（错误率 > 5%）重跑。

## 无 GPU 冒烟（dev-mode 正确性验证）

```bash
cargo build --bin zealot-backend
ZEALOT_GRPC_PORT=0 ZEALOT_HTTP_PORT=0 ./target/debug/zealot-backend &
curl -s -X POST http://127.0.0.1:<http_port>/admin/load \
     -H 'Content-Type: application/json' -d '{"model":"mock:tiny"}'
python3 scripts/bench_openai.py --base-url http://127.0.0.1:<http_port> \
    --model mock:tiny --concurrency 1 --input-len 100 \
    --max-tokens 8 --window 3 --rounds 1 --out-dir /tmp/bench_out
```

校验：报告 JSON 可解析、三档指标齐全、summary 表生成。指标数值对 mock 无意义。
