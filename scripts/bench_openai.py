#!/usr/bin/env python3
"""OpenAI 兼容面基准脚本（M5-⑪，issue #25）——风格对齐 vLLM benchmark_serving。

矩阵：并发 1/16/64 × 输入长度 ~100/1k/4k+ × 固定输出 256。
指标：聚合吞吐（tokens/s）+ TTFT + TPOT（P50/P95/P99）。
协议：多轮（--rounds）取中位；预热 + 稳定窗口（--window ≥ 60s 建议）。

用法：
  # 先加载模型（gRPC LoadModel 或 HTTP dev 口）：
  curl -X POST http://127.0.0.1:8081/admin/load -H 'Content-Type: application/json' \\
       -d '{"model":"mock:tiny"}'
  python3 scripts/bench_openai.py --base-url http://127.0.0.1:8081 \\
      --model mock:tiny --out-dir docs/benchmarks/results

输出：
  docs/benchmarks/results/report-<ts>.json   —— 完整报告（环境/矩阵/指标/逐轮明细）
  docs/benchmarks/results/summary-<ts>.md    —— 摘要对比表（占位符待 vLLM 列填充）

依赖：requests（.venv 已有）。无 GPU 时 mock:tiny 冒烟验证脚本正确性。
"""

import argparse
import concurrent.futures as cf
import json
import statistics
import sys
import time

import requests

DEFAULT_CONCURRENCY = [1, 16, 64]
DEFAULT_LENGTHS = [100, 1000, 4096]
DEFAULT_MAX_TOKENS = 256
DEFAULT_WINDOW_S = 5.0  # 冒烟用；正式执行按执行说明 ≥ 60s 稳定窗口


def build_prompt(n_tokens: int) -> str:
    """~n_tokens 的英文提示（≈4 字符/token；服务端 tokenizer 实际计数为准）。"""
    word = "benchmark"
    per_token = 4
    words = max(1, n_tokens // per_token)
    return " ".join([word] * words)


def run_request(base_url: str, model: str, prompt: str, max_tokens: int, timeout_s: float):
    """单次流式请求；返回 (ttft_s, n_out_tokens, total_s, status)。"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    t0 = time.monotonic()
    ttft = None
    n_tokens = 0
    try:
        with requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=timeout_s,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    n_tokens += 1
            total = time.monotonic() - t0
            if ttft is None:
                ttft = total  # 无输出（异常）
            return ttft, n_tokens, total, "ok"
    except Exception as e:  # noqa: BLE001
        return None, 0, 0.0, f"error: {e}"


def run_cell(base_url, model, concurrency, length, max_tokens, timeout_s, window_s):
    """一个 (并发, 长度) 单元：预热 + 满负载采集窗口。返回请求级明细。"""
    prompt = build_prompt(length)
    results = []

    def worker(_):
        return run_request(base_url, model, prompt, max_tokens, timeout_s)

    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        # 预热（并发满载）
        list(ex.map(worker, range(concurrency)))
        # 稳定窗口：持续满载提交直到窗口结束
        window_end = time.monotonic() + window_s
        futures = []
        while time.monotonic() < window_end:
            for _ in range(concurrency):
                futures.append(ex.submit(worker, None))
            time.sleep(0.05)
        for f in futures:
            results.append(f.result())
    return results


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return round(xs[k], 4)


def summarize_cell(results, concurrency, length):
    """单元 → 指标（吞吐按窗口总 token/总耗时）。"""
    ok = [r for r in results if r[0] is not None]
    total_tokens = sum(r[1] for r in ok)
    total_s = sum(r[2] for r in ok)
    n = len(ok)
    ttfts = [r[0] for r in ok]
    tpots = []
    for r in ok:
        ttft, nt, total, _status = r
        if nt > 1 and total > ttft:
            tpots.append((total - ttft) / max(1, nt - 1))
    return {
        "concurrency": concurrency,
        "input_length_tokens": length,
        "requests_ok": n,
        "requests_total": len(results),
        "total_output_tokens": total_tokens,
        "throughput_tokens_per_s": round(total_tokens / total_s, 2) if total_s > 0 else 0.0,
        "ttft_s": {"p50": pct(ttfts, 50), "p95": pct(ttfts, 95), "p99": pct(ttfts, 99)},
        "tpot_s": {"p50": pct(tpots, 50), "p95": pct(tpots, 95), "p99": pct(tpots, 99)},
    }


def median_cell(rounds):
    """多轮中位：吞吐取中位，TTFT/TPOT 各分位取中位。"""
    out = dict(rounds[0])
    for key in ["throughput_tokens_per_s"]:
        out[key] = round(statistics.median(r[key] for r in rounds), 2)
    for metric in ["ttft_s", "tpot_s"]:
        for q in ["p50", "p95", "p99"]:
            vals = [r[metric][q] for r in rounds if r[metric][q] is not None]
            if vals:
                out[metric][q] = round(statistics.median(vals), 4)
    out["rounds"] = len(rounds)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8081")
    ap.add_argument("--model", default="mock:tiny")
    ap.add_argument("--concurrency", default=",".join(map(str, DEFAULT_CONCURRENCY)))
    ap.add_argument("--input-len", default=",".join(map(str, DEFAULT_LENGTHS)))
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--rounds", type=int, default=1, help="多轮取中位（正式 ≥3）")
    ap.add_argument("--window", type=float, default=DEFAULT_WINDOW_S, help="稳定窗口秒数（正式 ≥60）")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out-dir", default="docs/benchmarks/results")
    args = ap.parse_args()

    concurrency = [int(x) for x in args.concurrency.split(",")]
    lengths = [int(x) for x in args.input_len.split(",")]
    matrix = []
    for c in concurrency:
        for ln in lengths:
            rounds = []
            for _ in range(max(1, args.rounds)):
                raw = run_cell(args.base_url, args.model, c, ln, args.max_tokens, args.timeout, args.window)
                rounds.append(summarize_cell(raw, c, ln))
            matrix.append(median_cell(rounds))
            m = matrix[-1]
            print(
                f"conc={c:<3} len={ln:<5} tok/s={m['throughput_tokens_per_s']:<10} "
                f"ttft_p50={m['ttft_s']['p50']} tpot_p50={m['tpot_s']['p50']} "
                f"(ok {m['requests_ok']}/{m['requests_total']})",
                flush=True,
            )

    report = {
        "schema": "zealot-bench/v1",
        "environment": {
            "engine": "zealot",
            "model": args.model,
            "gpu": None,  # TODO(platform): 填写（如 NVIDIA A100 80GB）
            "engine_version": None,  # TODO(platform): git rev-parse HEAD
            "vllm_commit": None,  # TODO(platform): 对比用 vLLM commit/tag
            "max_num_seqs_aligned": None,  # TODO(platform): 与 vLLM 对齐的值
        },
        "matrix": {
            "concurrency": concurrency,
            "input_length_tokens": lengths,
            "max_output_tokens": args.max_tokens,
        },
        "window_seconds": args.window,
        "rounds": args.rounds,
        "results": matrix,
    }

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    json_path = f"{args.out_dir}/report-{ts}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md = summary_table(report)
    md_path = f"{args.out_dir}/summary-{ts}.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"\n报告: {json_path}\n摘要: {md_path}")
    print(md)


def summary_table(report):
    """摘要对比表（vs vLLM 比值占位）。"""
    lines = [
        f"# 对比摘要：zealot vs vLLM（{report['environment']['model']}）",
        "",
        f"- 引擎版本: `{report['environment']['engine_version']}`  "
        f"- vLLM commit: `{report['environment']['vllm_commit']}`  "
        f"- GPU: `{report['environment']['gpu']}`  "
        f"- max_num_seqs: `{report['environment']['max_num_seqs_aligned']}`",
        f"- 窗口: {report['window_seconds']}s 稳定段 · 轮数: {report['rounds']}（取中位）",
        "",
        "| 并发 | 输入 len | 引擎 | 吞吐 tok/s | TTFT p50 | TTFT p95 | TTFT p99 | TPOT p50 | TPOT p95 | TPOT p99 | vs vLLM 吞吐 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in report["results"]:
        r = m  # zealot 行
        lines.append(
            f"| {m['concurrency']} | {m['input_length_tokens']} | zealot | "
            f"{m['throughput_tokens_per_s']} | {m['ttft_s']['p50']} | {m['ttft_s']['p95']} | "
            f"{m['ttft_s']['p99']} | {m['tpot_s']['p50']} | {m['tpot_s']['p95']} | "
            f"{m['tpot_s']['p99']} | — |"
        )
    lines += [
        "",
        "<!-- TODO(platform): vLLM 同卡同参跑 benchmark_serving 后，逐格填 vLLM 列并补比值 "
        "（zealot/vllm）；比值 = zealot 吞吐 / vLLM 吞吐。原始 JSON 见 report-*.json。 -->",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
