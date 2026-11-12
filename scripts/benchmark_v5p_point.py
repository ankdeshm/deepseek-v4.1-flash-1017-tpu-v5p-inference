#!/usr/bin/env python3
"""Steady-State Closed-Loop Benchmark & Prometheus Telemetry Harness for TPU v5p-32.

Enforces all benchmark measurement rules:
  - Constant concurrency C maintained across >= 300s steady-state window (600s for V4).
  - Warmup pass executed prior to measurement window; only steady-state completions inside
    [t_steady_start, t_steady_end] and steady-state token deltas are reported.
  - Tokenizes natural-language coding/reasoning passages from requests_mocked.jsonl using
    DeepSeek-V4.1-Flash's tokenizer to construct EXACT ISL token lengths (S1=2048, S2=1024,
    S3=8192, V3=64) with unique request nonces (zero artificial prefix-cache hits in B0/V1/V2/V3),
    OR multi-turn shared-prefix agentic sessions for V4.
  - Scrapes /metrics every 5s from colocated or prefill + decode servers to record:
    * DSpark mean acceptance length tau = 1 + delta(accepted_tokens) / delta(drafts)
    * KV cache usage % (mean & max across DP engines)
    * Running / waiting request queues & preemption count
    * Prefix cache hit rate (queries & hits)
  - Persists full JSON summary + per-request JSONL + Prometheus timeseries to out_dir.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
from pathlib import Path
import re
import threading
import time
import urllib.request
import uuid
import aiohttp


SERVED_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"


def fetch_text(url: str, timeout: float = 8.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def parse_prom_metrics(text: str) -> dict:
    def by_engine(metric_regex: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for m in re.finditer(rf"^(?:{metric_regex})(\{{[^}}]*\}})?\s+([0-9.eE+-]+)", text, re.M):
            labels = m.group(1) or ""
            val = float(m.group(2))
            em = re.search(r'engine="([^"]+)"', labels)
            eng = em.group(1) if em else "all"
            out[eng] = out.get(eng, 0.0) + val
        return out

    kv_by_eng = by_engine(r"vllm:(?:kv_cache_usage_perc|gpu_cache_usage_perc)")
    running_by_eng = by_engine(r"vllm:num_requests_running")
    waiting_by_eng = by_engine(r"vllm:num_requests_waiting")
    preempt_by_eng = by_engine(r"vllm:num_preemptions_total")
    prompt_tok_by_eng = by_engine(r"vllm:prompt_tokens_total")
    gen_tok_by_eng = by_engine(r"vllm:generation_tokens_total")
    prefix_queries = sum(by_engine(r"vllm:(?:prefix_cache_queries_total|gpu_prefix_cache_queries_total)").values())
    prefix_hits = sum(by_engine(r"vllm:(?:prefix_cache_hits_total|gpu_prefix_cache_hits_total)").values())
    drafts = sum(by_engine(r"vllm:spec_decode_num_drafts_total").values())
    accepted = sum(by_engine(r"vllm:spec_decode_num_accepted_tokens_total").values())

    kv_vals = list(kv_by_eng.values())
    return {
        "kv_usage_mean": (sum(kv_vals) / len(kv_vals)) if kv_vals else 0.0,
        "kv_usage_max": max(kv_vals) if kv_vals else 0.0,
        "running_total": sum(running_by_eng.values()),
        "waiting_total": sum(waiting_by_eng.values()),
        "preemptions_total": sum(preempt_by_eng.values()),
        "prompt_tokens_total": sum(prompt_tok_by_eng.values()),
        "generation_tokens_total": sum(gen_tok_by_eng.values()),
        "prefix_queries_total": prefix_queries,
        "prefix_hits_total": prefix_hits,
        "spec_num_drafts": drafts,
        "spec_num_accepted": accepted,
    }


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def build_corpus_tokens(tokenizer, dataset_path: Path) -> list[int]:
    texts: list[str] = []
    if dataset_path.exists():
        for line in dataset_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            body = obj.get("body", obj)
            if isinstance(body, str):
                body = json.loads(body)
            for msg in body.get("messages", []):
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > 64:
                    texts.append(content)
    if not texts:
        texts = [
            "Explain the distributed systems trade-offs between tensor parallelism, expert parallelism, "
            "and disaggregated prefill/decode serving on 3D torus optical interconnects. " * 200
        ]
    joined = "\n\n".join(texts)
    toks = tokenizer.encode(joined, add_special_tokens=False)
    while len(toks) < 32768:
        toks = toks + toks
    return toks


def make_prompt_generator(
    tokenizer,
    corpus_toks: list[int],
    isl: int,
    mode: str,
    num_sessions: int = 32,
    shared_prefix_tokens: int = 0,
):
    if mode in ("agentic_shared_prefix", "cached_prefix"):
        if shared_prefix_tokens > 0:
            shared_len = shared_prefix_tokens
        elif mode == "cached_prefix":
            shared_len = max(64, isl - 8)
        else:
            shared_len = max(512, isl - 512)
        turn_len = max(6, isl - shared_len)
        trunks: list[list[int]] = []
        for s in range(num_sessions):
            prefix_str = f"[Session-{s:03d}] "
            prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=False)
            rem = max(8, shared_len - len(prefix_ids))
            offset = (s * 997) % (len(corpus_toks) - rem - 1024)
            trunks.append((prefix_ids + corpus_toks[offset : offset + rem])[:shared_len])

        counter = 0

        def next_agentic() -> tuple[str, str]:
            nonlocal counter
            s = counter % num_sessions
            turn_id = counter // num_sessions
            counter += 1
            turn_hdr = tokenizer.encode(f"\n[Turn-{turn_id:05d}-{uuid.uuid4().hex[:6]}] ", add_special_tokens=False)
            rem = max(2, turn_len - len(turn_hdr))
            off = (counter * 313) % (len(corpus_toks) - rem - 128)
            full_ids = trunks[s] + (turn_hdr + corpus_toks[off : off + rem])[:turn_len]
            return tokenizer.decode(full_ids[: shared_len + turn_len], skip_special_tokens=True), f"sess-{s:03d}"

        return next_agentic

    counter = 0

    def next_fixed() -> tuple[str, str]:
        nonlocal counter
        counter += 1
        hdr = tokenizer.encode(f"[Req-{uuid.uuid4().hex[:12]}-{counter}] ", add_special_tokens=False)
        rem = max(4, isl - len(hdr))
        off = (counter * 521) % (len(corpus_toks) - rem - 128)
        full_ids = hdr + corpus_toks[off : off + rem]
        return tokenizer.decode(full_ids[:isl], skip_special_tokens=True), f"req-{counter}"

    return next_fixed


async def run_one_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    session_id: str,
    osl: int,
    temperature: float,
    ignore_eos: bool = True,
) -> dict:
    payload: dict = {
        "model": SERVED_MODEL,
        "prompt": prompt,
        "max_tokens": osl,
        "ignore_eos": ignore_eos,
        "temperature": temperature,
        "top_p": 0.95 if temperature > 0 else 1.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["min_tokens"] = osl
    headers = {"x-session-id": session_id, "x-user-id": session_id}
    t_start = time.perf_counter()
    t_first: float | None = None
    chunk_times: list[float] = []
    prompt_tokens = 0
    completion_tokens = 0
    err_msg = None
    status_code = 0

    try:
        async with session.post(f"{url.rstrip('/')}/v1/completions", json=payload, headers=headers) as resp:
            status_code = resp.status
            if resp.status != 200:
                body = await resp.text()
                return {
                    "ok": False,
                    "status": status_code,
                    "error": body[:400],
                    "t_start": t_start,
                    "t_end": time.perf_counter(),
                }
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data: "):
                    continue
                data_s = line[6:]
                if data_s == "[DONE]":
                    break
                try:
                    evt = json.loads(data_s)
                except Exception:
                    continue
                now = time.perf_counter()
                choices = evt.get("choices") or []
                if choices and (choices[0].get("text") or choices[0].get("reasoning_content")):
                    if t_first is None:
                        t_first = now
                    chunk_times.append(now)
                usage = evt.get("usage")
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                    completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"

    t_end = time.perf_counter()
    if err_msg or t_first is None or completion_tokens <= 0:
        return {
            "ok": False,
            "status": status_code,
            "error": err_msg or "empty_completion",
            "t_start": t_start,
            "t_end": t_end,
        }

    ttft_s = t_first - t_start
    e2e_s = t_end - t_start
    decode_s = max(1e-6, t_end - t_first)
    dec_toks = max(1, completion_tokens - 1)
    tpot_ms = (decode_s * 1000.0) / dec_toks
    req_decode_tps = dec_toks / decode_s
    return {
        "ok": True,
        "status": 200,
        "session_id": session_id,
        "t_start": t_start,
        "t_first": t_first,
        "t_end": t_end,
        "ttft_ms": round(ttft_s * 1000.0, 3),
        "tpot_ms": round(tpot_ms, 3),
        "req_decode_tok_per_s": round(req_decode_tps, 3),
        "e2e_s": round(e2e_s, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "num_chunks": len(chunk_times),
    }


async def benchmark_loop(
    base_url: str,
    next_prompt_fn,
    concurrency: int,
    warmup_s: float,
    steady_s: float,
    osl: int,
    temperature: float,
    ignore_eos: bool = True,
) -> tuple[list[dict], float, float]:
    timeout = aiohttp.ClientTimeout(total=1200, sock_read=600)
    connector = aiohttp.TCPConnector(limit=max(256, concurrency * 2))
    records: list[dict] = []
    t0 = time.perf_counter()
    t_steady_start = t0 + warmup_s
    t_steady_end = t_steady_start + steady_s

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def worker() -> None:
            while time.perf_counter() < t_steady_end:
                prompt, session_id = next_prompt_fn()
                rec = await run_one_request(
                    session,
                    base_url,
                    prompt,
                    session_id,
                    osl,
                    temperature,
                    ignore_eos=ignore_eos,
                )
                records.append(rec)

        tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.gather(*tasks, return_exceptions=True)

    return records, t_steady_start, t_steady_end


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base-url", required=True, help="Target OpenAI endpoint (router or colocated leader)")
    ap.add_argument("--decode-metrics-url", required=True, help="Decode or colocated leader :port for /metrics")
    ap.add_argument("--prefill-metrics-url", default="", help="Optional prefill leader :8000 for /metrics")
    ap.add_argument("--isl", type=int, required=True)
    ap.add_argument("--osl", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--warmup-s", type=float, default=30.0)
    ap.add_argument("--steady-s", type=float, default=300.0)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--mode", default="fixed", choices=["fixed", "agentic_shared_prefix", "cached_prefix"])
    ap.add_argument("--num-sessions", type=int, default=32)
    ap.add_argument("--shared-prefix-tokens", type=int, default=0)
    ap.add_argument("--ignore-eos", type=int, default=1, help="1=ignore_eos+min_tokens, 0=natural stop")
    ap.add_argument("--decode-chips", type=int, required=True)
    ap.add_argument("--total-chips", type=int, required=True)
    ap.add_argument("--ep-size", type=int, required=True)
    ap.add_argument("--dp-ranks", type=int, default=0, help="Decode DP ranks for batch/rank (0 = ep-size)")
    ap.add_argument("--tokenizer-path", default="/models/models/deepseek-ai/DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--dataset", default="/models/dsv41-stack/requests_mocked.jsonl")
    ap.add_argument("--out-dir", default="/models/dsv41-v5p/results")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    out_dir = Path(args.out_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bench:{args.tag}] Loading tokenizer from {args.tokenizer_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    corpus_toks = build_corpus_tokens(tokenizer, Path(args.dataset))
    next_prompt = make_prompt_generator(
        tokenizer,
        corpus_toks,
        args.isl,
        args.mode,
        args.num_sessions,
        args.shared_prefix_tokens,
    )

    # Scrape /metrics every 5s in background thread; capture baseline right at t_steady_start and t_steady_end
    stop_evt = threading.Event()
    prom_samples: list[dict] = []
    t_perf0 = time.perf_counter()

    def sampler() -> None:
        while not stop_evt.is_set():
            try:
                el = time.perf_counter() - t_perf0
                d_txt = fetch_text(f"{args.decode_metrics_url.rstrip('/')}/metrics")
                dm = parse_prom_metrics(d_txt)
                pm = None
                if args.prefill_metrics_url:
                    p_txt = fetch_text(f"{args.prefill_metrics_url.rstrip('/')}/metrics")
                    pm = parse_prom_metrics(p_txt)
                prom_samples.append(
                    {
                        "elapsed_s": round(el, 2),
                        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "decode": dm,
                        "prefill": pm,
                    }
                )
            except Exception as e:
                prom_samples.append({"elapsed_s": round(time.perf_counter() - t_perf0, 2), "error": str(e)})
            stop_evt.wait(5.0)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    print(
        f"[bench:{args.tag}] Running concurrency={args.concurrency} shape={args.isl}:{args.osl} "
        f"mode={args.mode} shared_prefix={args.shared_prefix_tokens} ignore_eos={bool(args.ignore_eos)} "
        f"warmup={args.warmup_s}s steady={args.steady_s}s against {args.base_url} ...",
        flush=True,
    )
    records, t_ss_start, t_ss_end = asyncio.run(
        benchmark_loop(
            args.base_url,
            next_prompt,
            args.concurrency,
            args.warmup_s,
            args.steady_s,
            args.osl,
            args.temperature,
            ignore_eos=bool(args.ignore_eos),
        )
    )
    stop_evt.set()
    th.join(timeout=10)

    ss_rel_start = t_ss_start - t_perf0
    ss_rel_end = t_ss_end - t_perf0
    steady_ok = [r for r in records if r.get("ok") and t_ss_start <= r["t_end"] <= t_ss_end]
    steady_err = [r for r in records if (not r.get("ok")) and t_ss_start <= r["t_end"] <= t_ss_end]

    # Compute exact server-side Prometheus steady-state token & tau deltas over [ss_rel_start, ss_rel_end]
    valid_prom = [s for s in prom_samples if "decode" in s and ss_rel_start <= s["elapsed_s"] <= ss_rel_end + 2.5]
    prom_summary: dict = {}
    if len(valid_prom) >= 2:
        p0, p1 = valid_prom[0], valid_prom[-1]
        dt_prom = max(1e-3, p1["elapsed_s"] - p0["elapsed_s"])
        d_gen = p1["decode"]["generation_tokens_total"] - p0["decode"]["generation_tokens_total"]
        d_drafts = p1["decode"]["spec_num_drafts"] - p0["decode"]["spec_num_drafts"]
        d_acc = p1["decode"]["spec_num_accepted"] - p0["decode"]["spec_num_accepted"]
        tau = (1.0 + d_acc / d_drafts) if d_drafts > 0 else 1.0
        kv_means = [s["decode"]["kv_usage_mean"] for s in valid_prom]
        kv_maxes = [s["decode"]["kv_usage_max"] for s in valid_prom]
        preempt_delta = p1["decode"]["preemptions_total"] - p0["decode"]["preemptions_total"]
        src_p0 = p0["prefill"] if p0.get("prefill") else p0["decode"]
        src_p1 = p1["prefill"] if p1.get("prefill") else p1["decode"]
        d_prompt = src_p1["prompt_tokens_total"] - src_p0["prompt_tokens_total"]
        pc_q = src_p1["prefix_queries_total"] - src_p0["prefix_queries_total"]
        pc_h = src_p1["prefix_hits_total"] - src_p0["prefix_hits_total"]
        prom_summary = {
            "prom_window_s": round(dt_prom, 2),
            "prom_prompt_tokens_delta": round(d_prompt, 1),
            "prom_input_tok_per_s": round(d_prompt / dt_prom, 2),
            "prom_gen_tokens_delta": round(d_gen, 1),
            "prom_output_tok_per_s": round(d_gen / dt_prom, 2),
            "prom_output_tok_per_s_per_decode_chip": round((d_gen / dt_prom) / args.decode_chips, 2),
            "prom_output_tok_per_s_per_total_chip": round((d_gen / dt_prom) / args.total_chips, 2),
            "dspark_tau": round(tau, 4),
            "decode_kv_usage_pct_mean": round(sum(kv_means) / len(kv_means) * 100.0, 2),
            "decode_kv_usage_pct_max": round(max(kv_maxes) * 100.0, 2),
            "decode_preemptions_delta": int(preempt_delta),
            "prefix_cache_hit_rate_pct": round((pc_h / pc_q * 100.0) if pc_q > 0 else 0.0, 2),
        }

    client_gen_tokens = sum(r["completion_tokens"] for r in steady_ok)
    client_out_tps = client_gen_tokens / args.steady_s
    ttfts = [r["ttft_ms"] for r in steady_ok]
    tpots = [r["tpot_ms"] for r in steady_ok]
    req_tps = [r["req_decode_tok_per_s"] for r in steady_ok]
    e2es = [r["e2e_s"] for r in steady_ok]
    # Rule 4: TTFT / per-user speed over ALL requests of the level (warmup + window + drain)
    all_ok = [r for r in records if r.get("ok")]
    all_err = [r for r in records if not r.get("ok")]
    ttfts_all = [r["ttft_ms"] for r in all_ok]
    req_tps_all = [r["req_decode_tok_per_s"] for r in all_ok]
    dp_ranks = args.dp_ranks or args.ep_size
    if valid_prom:
        run_means = [s["decode"]["running_total"] for s in valid_prom]
        wait_means = [s["decode"]["waiting_total"] for s in valid_prom]
        prom_summary["decode_running_total_mean"] = round(sum(run_means) / len(run_means), 2)
        prom_summary["decode_running_total_max"] = round(max(run_means), 2)
        prom_summary["decode_waiting_total_mean"] = round(sum(wait_means) / len(wait_means), 2)
        prom_summary["decode_batch_per_dp_rank_mean"] = round(sum(run_means) / len(run_means) / max(1, dp_ranks), 2)
        prom_summary["dp_ranks"] = dp_ranks

    summary = {
        "tag": args.tag,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "shape": f"{args.isl}:{args.osl}",
        "mode": args.mode,
        "shared_prefix_tokens": args.shared_prefix_tokens,
        "ignore_eos": bool(args.ignore_eos),
        "concurrency": args.concurrency,
        "ep_size": args.ep_size,
        "decode_chips": args.decode_chips,
        "total_chips": args.total_chips,
        "precision": {
            "attention": "FP8/BF16 (CSA FP8 KV + BF16 Q, SWA BF16)",
            "moe_experts": "MXFP4 W4A8 (4-bit E2M1 + E8M0 in HBM, uint4->FP8 Mosaic dot)",
            "kv_cache": "FP8 (CSA main cache FP8_E4M3 + E8M0 scale; SWA ring BF16)",
        },
        "steady_window_s": args.steady_s,
        "steady_completed_requests": len(steady_ok),
        "steady_failed_requests": len(steady_err),
        "error_rate_pct": round(100.0 * len(steady_err) / max(1, len(steady_ok) + len(steady_err)), 3),
        "client_output_tok_per_s": round(client_out_tps, 2),
        "client_output_tok_per_s_per_decode_chip": round(client_out_tps / args.decode_chips, 2),
        "client_output_tok_per_s_per_total_chip": round(client_out_tps / args.total_chips, 2),
        "ttft_ms_p50": round(percentile(ttfts, 50), 2),
        "ttft_ms_p90": round(percentile(ttfts, 90), 2),
        "ttft_ms_p95": round(percentile(ttfts, 95), 2),
        "ttft_ms_p99": round(percentile(ttfts, 99), 2),
        "ttft_all_ms_p50": round(percentile(ttfts_all, 50), 2),
        "ttft_all_ms_p95": round(percentile(ttfts_all, 95), 2),
        "ttft_all_ms_p99": round(percentile(ttfts_all, 99), 2),
        "tpot_ms_p50": round(percentile(tpots, 50), 2),
        "tpot_ms_p90": round(percentile(tpots, 90), 2),
        "tpot_ms_p95": round(percentile(tpots, 95), 2),
        "tpot_ms_p99": round(percentile(tpots, 99), 2),
        "decode_tok_per_s_p50": round(percentile(req_tps, 50), 2),
        "decode_tok_per_s_p90_low": round(percentile(req_tps, 10), 2),
        "decode_tok_per_s_p95_low": round(percentile(req_tps, 5), 2),
        "decode_tok_per_s_p99_low": round(percentile(req_tps, 1), 2),
        "decode_tok_per_s_all_p50": round(percentile(req_tps_all, 50), 2),
        "decode_tok_per_s_all_p90_low": round(percentile(req_tps_all, 10), 2),
        "decode_tok_per_s_all_p95_low": round(percentile(req_tps_all, 5), 2),
        "all_completed_requests": len(all_ok),
        "all_failed_requests": len(all_err),
        "all_errors_sample": [str(r.get("error", ""))[:200] for r in all_err[:5]],
        "e2e_s_p50": round(percentile(e2es, 50), 2),
        "e2e_s_p99": round(percentile(e2es, 99), 2),
        "prometheus": prom_summary,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "prom_timeseries.json").write_text(json.dumps(prom_samples, indent=2))
    with open(out_dir / "requests.jsonl", "w") as rf:
        for r in records:
            rf.write(json.dumps(r) + "\n")

    print(f"[bench:{args.tag}] SUMMARY:\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
