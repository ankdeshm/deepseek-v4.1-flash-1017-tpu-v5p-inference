#!/usr/bin/env python3
"""Step 0 10-Request Smoke Test & Post-Warmup HBM / DSpark Tau Verification.

Runs 10 diverse prompts (greedy + stochastic) against the live v5p vLLM endpoint,
verifies 10/10 HTTP 200, non-empty coherent completions, and DSpark tau >= 2.0
from /metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
import urllib.request

SERVED_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"

PROMPTS = [
    "Write a concise Python function `topological_sort(graph: dict[str, list[str]]) -> list[str]` using Kahn's algorithm and explain its time complexity.",
    "Derive the closed-form sum of the geometric series S_n = a + ar + ar^2 + ... + ar^(n-1) for r != 1.",
    "Explain why Mixture-of-Experts (MoE) decode is memory-bandwidth bound at small batch sizes and how Expert Parallelism shifts the roofline.",
    "Write a SQL query using a window function to find the top 3 highest-paid employees per department without ties.",
    "Implement a thread-safe TokenBucket rate limiter in Go using sync.Mutex and time.Now().",
    "Compare 2D torus and 3D torus optical interconnects in terms of bisection bandwidth and all-to-all hop count.",
    "Solve for x over the real numbers: 2^(2x) - 5 * 2^x + 4 = 0. Show each step clearly.",
    "Write a Rust function that merges two sorted slices `&[i64]` into a `Vec<i64>` in O(n + m) time.",
    "Explain how speculative decoding with Multi-Token Prediction (MTP) preserves the target model's exact output distribution.",
    "Given a binary tree, write a Python iterative postorder traversal without recursion.",
]


def fetch_text(url: str, timeout: float = 15.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def parse_spec_metrics(text: str) -> tuple[float, float, float]:
    def sum_m(pat: str) -> float:
        tot = 0.0
        for m in re.finditer(rf"^(?:{pat})(\{{[^}}]*\}})?\s+([0-9.eE+-]+)", text, re.M):
            tot += float(m.group(2))
        return tot

    drafts = sum_m(r"vllm:spec_decode_num_drafts_total")
    accepted = sum_m(r"vllm:spec_decode_num_accepted_tokens_total")
    gen = sum_m(r"vllm:generation_tokens_total")
    return drafts, accepted, gen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://<private-ip>:8000")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    m0 = fetch_text(f"{args.base_url.rstrip('/')}/metrics")
    d0, a0, g0 = parse_spec_metrics(m0)

    results = []
    t_all0 = time.perf_counter()
    for idx, prompt in enumerate(PROMPTS):
        temp = 0.0 if idx < 5 else 0.6
        payload = json.dumps(
            {
                "model": SERVED_MODEL,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "min_tokens": args.max_tokens,
                "ignore_eos": True,
                "temperature": temp,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{args.base_url.rstrip('/')}/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=180) as resp:
            status = resp.status
            body = json.loads(resp.read().decode("utf-8"))
        dt_s = time.perf_counter() - t0
        choice = body["choices"][0]
        text = choice.get("text") or ""
        usage = body.get("usage", {})
        comp_toks = int(usage.get("completion_tokens", 0))
        results.append(
            {
                "idx": idx,
                "temperature": temp,
                "status": status,
                "latency_s": round(dt_s, 3),
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": comp_toks,
                "tok_per_s": round(comp_toks / max(1e-6, dt_s), 2),
                "output_preview": text[:280],
            }
        )
        print(f"[smoke {idx+1}/10] status={status} lat={dt_s:.2f}s toks={comp_toks} preview={text[:90]!r}", flush=True)

    total_s = time.perf_counter() - t_all0
    m1 = fetch_text(f"{args.base_url.rstrip('/')}/metrics")
    d1, a1, g1 = parse_spec_metrics(m1)
    d_drafts = d1 - d0
    d_acc = a1 - a0
    d_gen = g1 - g0
    tau = (1.0 + d_acc / d_drafts) if d_drafts > 0 else 1.0

    ok_all = all(r["status"] == 200 and r["completion_tokens"] >= args.max_tokens and len(r["output_preview"].strip()) > 20 for r in results)
    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": args.base_url,
        "num_requests": len(results),
        "all_200_and_coherent": ok_all,
        "total_elapsed_s": round(total_s, 3),
        "spec_drafts_delta": d_drafts,
        "spec_accepted_delta": d_acc,
        "gen_tokens_delta": d_gen,
        "dspark_tau": round(tau, 4),
        "tau_gate_gt_2_pass": bool(tau >= 2.0),
        "requests": results,
    }
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(summary, indent=2))
    print(f"[SMOKE-SUMMARY] all_ok={ok_all} dspark_tau={tau:.4f} gate_pass={tau >= 2.0}", flush=True)
    if not ok_all or tau < 2.0:
        raise SystemExit(f"Smoke gate failed: all_ok={ok_all}, dspark_tau={tau:.4f}")


if __name__ == "__main__":
    main()
