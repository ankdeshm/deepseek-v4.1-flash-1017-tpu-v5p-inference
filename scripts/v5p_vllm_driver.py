#!/usr/bin/env python3
"""Universal Multi-Host TPU v5p vLLM Driver (Colocated, Prefill, and Decode).

Supports:
  - Roles: colocated | prefill | decode
  - EP / DP sizes: 32 (8 hosts, 2x4x4), 16 (4 hosts, 2x2x4), 8 (2 hosts, 2x2x2)
  - Slice modes:
      * full     -> real GCE metadata + unix:///var/run/tpu-plugin/vbar.sock
      * subslice -> v5p_subslice_metadata.py on 127.0.0.1:8088 + LD_PRELOAD=/workspace/metadata_hook_8088.so
                    (unset VBAR_CONTROL_SERVICE_URL, matching physical torus host order)
  - Automatic file-coordinated host discovery on /models/dsv41-v5p/coord/<run_id>/
    ordering hosts strictly by GCE agent-worker-number so vLLM --data-parallel-start-rank
    matches physical TPU_WORKER_ID (0..num_hosts-1).
  - Automatic tee logging to /models/dsv41-v5p/logs/<run_id>_sub<sub_idx>.log.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

SERVED_MODEL_NAME = "deepseek-ai/DeepSeek-V4.1-Flash"

SUBSLICE_GEOM = {
    32: {
        "topology": "2x4x4",
        "host_bounds": "1,2,4",
        "wrap": "false,false,false",
        "accel_type": "v5p-64",
        "torch_tpu_topology": "2,4,4",
        "num_hosts": 8,
    },
    16: {
        "topology": "2x2x4",
        "host_bounds": "1,1,4",
        "wrap": "false,false,false",
        "accel_type": "v5p-32",
        "torch_tpu_topology": "2,2,4",
        "num_hosts": 4,
    },
    8: {
        "topology": "2x2x2",
        "host_bounds": "1,1,2",
        "wrap": "false,false,false",
        "accel_type": "v5p-16",
        "torch_tpu_topology": "2,2,2",
        "num_hosts": 2,
    },
}


def log(tag: str, msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] [{tag}] {msg}", flush=True)


def get_meta(rel_path: str) -> str:
    req = urllib.request.Request(
        f"http://169.254.169.254/computeMetadata/v1/{rel_path}",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.read().decode("utf-8", errors="replace").strip()


def get_primary_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("<private-ip>", 1))
        return s.getsockname()[0]
    finally:
        s.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--role", required=True, choices=["colocated", "prefill", "decode"])
    ap.add_argument("--ep-size", type=int, required=True, choices=[8, 16, 32])
    ap.add_argument("--subslice-mode", required=True, choices=["full", "subslice"])
    ap.add_argument("--max-num-seqs", type=int, default=48)
    ap.add_argument("--max-num-batched-tokens", type=int, default=288)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=131072)
    ap.add_argument("--enable-dspark", type=int, default=-1, help="-1=auto (1 for colocated/decode, 0 for prefill)")
    ap.add_argument("--dspark-tokens", type=int, default=5)
    ap.add_argument("--enable-prefix-caching", type=int, default=-1, help="-1=auto (1 for prefill/colocated, 0 for decode)")
    ap.add_argument("--async-scheduling", type=int, default=-1, help="-1=auto (1 for colocated/decode, 0 for prefill)")
    ap.add_argument("--port", type=int, default=0, help="0=auto (9400 for decode, 8000 for prefill/colocated)")
    ap.add_argument("--mds-port", type=int, default=8088)
    ap.add_argument("--scripts-dir", default="/models/dsv41-v5p/scripts")
    ap.add_argument("--base-dir", default="/models/dsv41-v5p")
    ap.add_argument("--model-path", default="/dev/shm/DeepSeek-V4.1-Flash")
    ap.add_argument("--peer-worker-ids", default="", help="Optional comma-separated GCE agent-worker-number list (e.g. 0,1,2,3 or 4,5,6,7); auto-derived from role+ep_size if empty")
    ap.add_argument("--skip-setup", action="store_true")
    ap.add_argument(
        "--cache-root",
        default="",
        help="Compile-cache root. Empty = /root/v5p_vllm_cache_<role> on the node boot disk. "
        "Do NOT use /dev/shm on v5p: tmpfs counts against the ~446 GiB host RAM that already "
        "holds the 189 GiB CPU-offloaded engram shards (caused a kubelet memory eviction).",
    )
    ap.add_argument(
        "--disable-compile-cache",
        type=int,
        default=1,
        help="1 = VLLM_DISABLE_COMPILE_CACHE=1 (skip ~2 GiB/bucket/rank pickled artifacts; the "
        "torch_tpu Tier-3 PJRT cache under <cache-root>/xla is still written).",
    )
    ap.add_argument("--api-server-count", type=int, default=4)
    ap.add_argument("--delete-engram-after-load", type=int, default=1, help="1 = unlink /dev/shm engram shards after the final load pass")
    ap.add_argument("--prune-tier2-shm", type=int, default=1, help="1 = drop stale /dev/shm Tier-2 cache binaries (disk Tier-3 keeps them)")
    args = ap.parse_args()

    geom = SUBSLICE_GEOM[args.ep_size]
    num_hosts = geom["num_hosts"]
    dp_size = args.ep_size
    dp_size_local = 4

    # 1. Ensure stack + v5p patch + hybrid /dev/shm checkpoint are staged
    scripts_dir = Path(args.scripts_dir)
    base_dir = Path(args.base_dir)
    if not args.skip_setup:
        subprocess.run(
            [
                sys.executable,
                str(scripts_dir / "in_pod_setup_v5p.py"),
                "--scripts-dir",
                str(scripts_dir),
                "--shm-model-dir",
                args.model_path,
            ],
            check=True,
        )

    # 2. Deterministically resolve participating hosts via GCE metadata + internal DNS (zero GCSFuse ESTALE risk!)
    phys_worker_id = int(get_meta("instance/attributes/agent-worker-number"))
    full_hn = get_meta("instance/hostname")
    domain = full_hn.split(".", 1)[1]
    all_eps = [x.strip() for x in get_meta("instance/attributes/worker-network-endpoints").split(",") if x.strip()]

    if args.peer_worker_ids.strip():
        target_ids = sorted(int(x) for x in args.peer_worker_ids.split(",") if x.strip())
    elif args.ep_size == 32:
        target_ids = list(range(8))
    elif len(all_eps) == 4:
        if args.ep_size == 16:
            target_ids = [0, 1, 2, 3]
        else:
            target_ids = [2, 3] if args.role == "decode" else [0, 1]
    elif args.role == "decode":
        target_ids = [4, 5, 6, 7] if args.ep_size == 16 else [4, 5]
    else:
        target_ids = [0, 1, 2, 3] if args.ep_size == 16 else [0, 1]

    if len(target_ids) != num_hosts or phys_worker_id not in target_ids:
        raise RuntimeError(
            f"Host phys_worker_id={phys_worker_id} not in target_ids={target_ids} (expected {num_hosts} hosts)"
        )

    peers = [
        {
            "phys_worker_id": wid,
            "short_hostname": all_eps[wid],
            "ip": socket.gethostbyname(f"{all_eps[wid]}.{domain}"),
        }
        for wid in target_ids
    ]
    my_ip = peers[target_ids.index(phys_worker_id)]["ip"]
    sub_idx = target_ids.index(phys_worker_id)
    leader_ip = peers[0]["ip"]
    endpoints_csv = ",".join(p["short_hostname"] for p in peers)
    tag = f"{args.run_id}:sub{sub_idx}(phys{phys_worker_id})"
    log(tag, f"Resolved {num_hosts} hosts via GCE DNS: leader_ip={leader_ip} endpoints={endpoints_csv}")

    if sub_idx == 0:
        Path(f"/tmp/cluster_plan_{args.run_id}.json").write_text(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "role": args.role,
                    "ep_size": args.ep_size,
                    "subslice_mode": args.subslice_mode,
                    "leader_ip": leader_ip,
                    "http_port": (args.port or (9400 if args.role == "decode" else 8000)),
                    "peers": peers,
                },
                indent=2,
            )
        )

    # 3. Clean stale IPC/cache state (preserving /dev/shm/DeepSeek-V4.1-Flash)
    cache_root = Path(args.cache_root or f"/root/v5p_vllm_cache_{args.role}")
    for subdir in ("vllm", "xla", "inductor", "tmp", "plan_dump"):
        (cache_root / subdir).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "bash",
            "-lc",
            f"ulimit -c 0; mkdir -p /root/.cache && ln -sfn {cache_root}/vllm /root/.cache/vllm && ln -sfn {cache_root}/xla /tmp/xla_cache && rm -rf /tmp/tpu_logs/st_stage_* /dev/shm/st_stage_* /dev/shm/psm_* /dev/shm/sem.* /tmp/libengram_gather_omp_*.so /core* /workspace/core* || true",
        ],
        check=False,
    )

    # 4. Build environment
    env = os.environ.copy()
    for k in list(env.keys()):
        if k.startswith("TPU_") or k in {
            "VBAR_CONTROL_SERVICE_URL",
            "TORCH_TPU_TOPOLOGY",
            "HOST_BOUNDS",
            "CHIPS_PER_HOST_BOUNDS",
            "LD_PRELOAD",
            "GCE_METADATA_HOST",
            "GCE_METADATA_IP",
            "CLOUD_TPU_TASK_ID",
        }:
            env.pop(k, None)

    mds_proc = None
    if args.subslice_mode == "full":
        if Path("/var/run/tpu-plugin/vbar.sock").is_socket():
            env["VBAR_CONTROL_SERVICE_URL"] = "unix:///var/run/tpu-plugin/vbar.sock"
    else:
        mds_cmd = [
            sys.executable,
            "/workspace/v5p_subslice_metadata.py",
            "--port",
            str(args.mds_port),
            "--worker-id",
            str(sub_idx),
            "--worker-endpoints",
            endpoints_csv,
            "--topology",
            geom["topology"],
            "--host-bounds",
            geom["host_bounds"],
            "--chips-per-host-bounds",
            "2,2,1",
            "--wrap",
            geom["wrap"],
            "--accelerator-type",
            geom["accel_type"],
        ]
        mds_env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
        mds_proc = subprocess.Popen(mds_cmd, env=mds_env)
        for _ in range(30):
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{args.mds_port}/computeMetadata/v1/instance/attributes/healthz",
                    headers={"Metadata-Flavor": "Google"},
                )
                if urllib.request.urlopen(req, timeout=1).read() == b"ok":
                    break
            except Exception:
                time.sleep(0.2)
        env["MOCK_METADATA_PORT"] = str(args.mds_port)
        env["LD_PRELOAD"] = "/workspace/metadata_hook_8088.so"
    env["TORCH_TPU_TOPOLOGY"] = geom["torch_tpu_topology"]

    dp_master_port = 29520 if args.role != "prefill" else 29420
    dp_rpc_port = 29550 if args.role != "prefill" else 29450
    barrier_port = 29596 if args.role != "prefill" else 29496
    http_port = args.port or (9400 if args.role == "decode" else 8000)

    enable_dspark = (0 if args.role == "prefill" else 1) if args.enable_dspark < 0 else args.enable_dspark
    enable_pc = (0 if args.role == "decode" else 1) if args.enable_prefix_caching < 0 else args.enable_prefix_caching
    async_sched = (0 if args.role == "prefill" else 1) if args.async_scheduling < 0 else args.async_scheduling

    env.update(
        {
            "VLLM_ST_STAGE_SESSION": f"{args.run_id}_{int(time.time())}",
            "VLLM_GCS_STAGE_BUCKET": os.environ.get("VLLM_GCS_STAGE_BUCKET", "YOUR_GCS_BUCKET"),
            "TORCH_TPU_PRE_INIT_BARRIER_PORT": str(barrier_port),
            "TPU_RUNTIME_METRICS_PORTS": "8431,8432,8433,8434",
            "VLLM_HOST_IP": my_ip,
            "MASTER_ADDR": leader_ip,
            "MASTER_PORT": str(dp_master_port),
            "TORCH_TPU_DP_MASTER_ADDR": leader_ip,
            "TORCH_TPU_DP_MASTER_PORT": str(dp_master_port),
            "VLLM_CACHE_ROOT": str(cache_root / "vllm"),
            "VLLM_XLA_CACHE_PATH": str(cache_root / "xla"),
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": str(cache_root / "xla"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "inductor"),
            "TMPDIR": str(cache_root / "tmp"),
            "TORCH_DIST_TIMEOUT": "1800",
            "VLLM_RPC_TIMEOUT": "1800000",
            "VLLM_SHM_BROADCAST_TIMEOUT_S": "1800",
            "VLLM_ENGINE_ITERATION_TIMEOUT_S": "1800",
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "1800",
            "VLLM_ENGINE_READY_TIMEOUT_S": "7200",
            "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "120",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "6",
            "GOMP_SPINCOUNT": "1000",
            "USE_PHASED_PROFILER": "false",
            "VLLM_LOG_POST_WARMUP_HBM": "1",
            "VLLM_PLUGINS": "torchtpu,torchtpu_layers",
            "VLLM_SERVER_DEV_MODE": "1",
            "PYTHONPATH": "/workspace/sitecustomize_bench:/workspace/vllm:/workspace/vllm-torchtpu/src",
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
            "VLLM_DISABLE_COMPILE_CACHE": "1" if args.disable_compile_cache else "0",
            "SPEC_WARMUP": "0",
            "VLLM_SPEC_WARMUP": "0",
            "PYTHONFAULTHANDLER": "1",
            "PYTHONUNBUFFERED": "1",
            # DeepSeek-V4.1 Kernel Config (FP8/BF16 Attn, MXFP4 W4A8 MoE via v5p uint4 patch, FP8 CSA KV)
            "TPU_ROPE_CACHE_TRUNCATE": "1",
            "TPU_ROPE_CACHE_ROW_MAJOR": "1",
            "TPU_MOE_COLLECTION_CHUNK_SIZE": "8192",
            "VLLM_TPU_FAST_TOKEN_SUBSTITUTION": "0",
            "ENABLE_CONNECTOR": "0",
            "ENABLE_DECODE_BENCH_CONNECTOR": "0",
            "USE_MOE_FUSED_EP_KERNEL": "1",
            "MOE_FUSED_EP_KERNEL_IMPL": "v2",
            "MOE_FUSED_EP_ENABLE_W4A8": "1",
            "MOE_FUSED_EP_V2_SHARDED_PLAN": "1",
            "MOE_REQUANTIZE_BLOCK_SIZE": "512",
            "TPU_MOE_DENSE_EXPERT_THRESHOLD": "0",
            "VLLM_TPU_MOE_FP8_DISPATCH": "1",
            "VLLM_DSV41_MOE_EP_V2_ALL39": "0",
            "VLLM_DSV41_MOE_EP_V2_MAX_LOCAL_TOKENS": "16384",
            "TPU_MOE_SKIP_PADDED_TOKENS": "0",
            "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS": "false",
            "VLLM_DSV41_MOE_FP4_PATH_C_MAX_TOKENS": "0",
            "VLLM_DSV41_MOE_FP4_GMM1_FP8": "0",
            "VLLM_DSV41_INDEXER_CAND_READ_16K": "0",
            "VLLM_DSV41_DECODER_TP_ATTN_VERIFY": "0",
            "VLLM_DSV41_DECODER_TP_ATTN_MIN_TOKENS": "999999",
            "VLLM_DSV41_CSA_UNION": "0",
            "VLLM_DSV41_CSA_UNION_SC": "0",
            "VLLM_DSV41_CSA_UNION_VERIFY_BQ": "6",
            "VLLM_DSV41_CSA_UNION_BQ": "6",
            "VLLM_DSV41_CSA_UNION_M": "1536",
            "VLLM_DSV41_CSA_UNION_MAX_TOKENS": "0",
            "VLLM_DSV41_SC_UNION_MAX_BLK_SHIFT": "-1",
            "VLLM_DSV41_FAST_GROUPED_ATTN": "1",
            "VLLM_DSV41_REPL_SHARED_EXPERTS": "1",
            "VLLM_DSV41_ENGRAM_H2D_PREFIX": "0",
            "VLLM_DSV41_TP_SHARD_ENGRAM": "0",
            "VLLM_DSV41_CED": "1" if args.role == "prefill" else "0",
            "VLLM_DSV41_HYBRID_CED_DECODE": "0",
            "VLLM_DSV41_PURE_20L_PREFILL": "0",
            "VLLM_DSV41_TP_ATTN": "0",
            "VLLM_DSV41_TP_ATTN_MIN_TOKENS": "999999",
            "VLLM_DSV41_TP_ATTN_LOCAL_TOPK": "1",
            "VLLM_DSV41_TP_INDEXER": "1",
            "VLLM_DSV41_TP_INDEXER_MIN_TOKENS": "0",
            "VLLM_DSV41_TOKEN_SHARD_MIN_TOKENS": "0",
            "VLLM_DSV41_L2_CAND_K_MIN_TOKENS": "0",
            "VLLM_DSV41_ULYSSES_MIN_TOKENS": "0",
            "VLLM_DSV41_HOIST_CSA_GATHER": "0",
            "VLLM_DSV41_HOIST_CSA_MAX_TOKENS": "0",
            "VLLM_DSV41_FUSED_SEAM_QUANT": "1",
            "VLLM_DSV41_FUSED_Q_ROPE": "1",
            "VLLM_DSV41_DSPARK_SHARDED_CTX_KV": "1",
            "TPU_DEEPSEEK_V41_ENGRAM_DEBUG_VERIFY": "0",
            "VLLM_SPLIT_TOKENIZE_MIN_CHARS": "65536",
            "LIBTPU_INIT_ARGS": (
                "--xla_tpu_enable_diagnostic_coordinator=false "
                "--xla_tpu_no_crash_on_oom=false "
                "--xla_tpu_scheduler_percent_shared_memory_limit=100 "
                "--xla_tpu_enable_scheduler_memory_pressure_tracking=true "
                "--xla_tpu_enable_async_collective_fusion=true "
                "--xla_tpu_scheduling_annotation_deannotate_unsupported_groups=true "
                "--xla_tpu_enable_sparse_core_compute_offload=false "
                "--xla_tpu_offload_gather_to_sparsecore=false "
                "--xla_tpu_enable_offloading_gather_to_sparsecore=false "
                "--xla_tpu_offload_all_supported_gathers_to_sparsecore=false "
                "--xla_tpu_enable_offloading_scatter_to_sparsecore=false "
                "--xla_tpu_enable_offloading_sort_to_sparsecore=false "
                "--xla_tpu_enable_offloading_copy_to_sparsecore=false "
                "--xla_tpu_enable_offloading_reshape_to_sparsecore=false "
                "--xla_tpu_enable_offloading_reduce_to_sparsecore=false "
                "--xla_tpu_num_sparse_cores_for_gather_offloading=0 "
                "--xla_tpu_num_sparse_cores_for_scatter_offloading=0 "
                "--xla_tpu_num_sparse_cores_for_sort_offloading=0 "
                "--xla_tpu_enable_sparse_core_collective_offload_all_gather=false "
                "--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=false "
                "--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=false "
                "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false "
                "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false "
                "--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=false "
                "--xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all=false "
                "--xla_tpu_enable_sparse_core_collective_offload_ragged_all_to_all=false "
                "--xla_tpu_enable_sparse_core_async_all_to_all=false "
                "--xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false"
            ),
        }
    )

    if args.role == "prefill":
        max_num_seqs = 64 if args.max_num_seqs == 48 else args.max_num_seqs
        mnbt = 8192 if args.max_num_batched_tokens == 288 else args.max_num_batched_tokens
        compile_sizes = sorted({512, 2048, mnbt})
    elif args.role == "colocated":
        max_num_seqs = args.max_num_seqs
        mnbt = args.max_num_batched_tokens
        compile_sizes = sorted(x for x in {16, 40, 48, 64, 96, 192, 288, 384, 512, 2048, mnbt} if x <= mnbt)
    else:  # decode
        max_num_seqs = args.max_num_seqs
        mnbt = args.max_num_batched_tokens
        compile_sizes = sorted(x for x in {16, 40, 48, 64, 96, 192, 288, 384, mnbt} if x <= mnbt)

    vllm_cmd = ["vllm", "serve", args.model_path]
    if sub_idx == 0:
        vllm_cmd += [
            "--host",
            "0.0.0.0",
            "--port",
            str(http_port),
            "--api-server-count",
            str(args.api_server_count),
            "--renderer-num-workers",
            "4",
        ]
    else:
        vllm_cmd += [
            "--headless",
            "--data-parallel-start-rank",
            str(sub_idx * dp_size_local),
        ]

    vllm_cmd += [
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--trust-remote-code",
        "--prefill-schedule-interval",
        "1",
        "--data-parallel-size",
        str(dp_size),
        "--data-parallel-size-local",
        str(dp_size_local),
        "--data-parallel-address",
        leader_ip,
        "--data-parallel-rpc-port",
        str(dp_rpc_port),
        "--tensor-parallel-size",
        "1",
        "--enable-expert-parallel",
        "--enable-ep-weight-filter",
        "--kv-cache-dtype",
        "fp8",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-num-batched-tokens",
        str(mnbt),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--no-disable-hybrid-kv-cache-manager",
        "--enable-prompt-tokens-details",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "deepseek_v41",
        "--reasoning-parser",
        "deepseek_v41",
        "--compilation-config",
        json.dumps({"compile_sizes": compile_sizes}),
    ]

    if enable_pc:
        retention = "1024" if args.role == "prefill" else "0"
        vllm_cmd += ["--enable-prefix-caching", "--prefix-cache-retention-interval", retention]
    else:
        vllm_cmd += ["--no-enable-prefix-caching"]

    vllm_cmd += ["--async-scheduling" if async_sched else "--no-async-scheduling"]

    if enable_dspark:
        env["VLLM_SPEC_DECODE_TOKENS"] = str(args.dspark_tokens)
        vllm_cmd += [
            "--speculative-config",
            json.dumps({"method": "dspark", "num_speculative_tokens": args.dspark_tokens}),
        ]

    if args.role in {"prefill", "decode"}:
        is_prefill = args.role == "prefill"
        ctrl_port = 27000 if is_prefill else 28000
        kv_xfer_port = 9100 if is_prefill else 9200
        env.update(
            {
                "TPU_USE_RAIDEN_KV_CACHE_MANAGER": "1",
                "TPU_RAIDEN_DSV4_ADMISSION": "1",
                "TPU_KV_RESHARD_TRANSPORT": "raiden",
                "TPU_RAIDEN_TRANSFER_PARALLELISM": "1",
                "TPU_ENABLE_D2H_TRANSFER": "1",
                "TPU_HMA_D2H_COPY_IMPL": "device_put",
                "TPU_HMA_MIN_DONE_RECVING_BATCH": "1",
                "TPU_HMA_MAX_DONE_RECVING_BATCH": "1",
                "TPU_HMA_MAX_INFLIGHT_D2H_SENDS": "1",
                "TPU_HMA_MAX_INFLIGHT_PULLS": "8",
                "TPU_MAX_HOST_KV_BUFFER_SIZE": "64",
                "TPU_P2P_WAIT_PULL_TIMEOUT": "300",
                "TPU_KV_SHM_POOL_GB": "8",
                "TPU_SIDE_CHANNEL_PORT": "9600",
                "TPU_RAIDEN_PLAN_DUMP_DIR": str(cache_root / "plan_dump"),
                "TPU_KV_TRANSFER_NAMESPACE": f"{args.role}_{args.run_id}",
                "TPU_RAIDEN_JOB_NAME": args.role,
                "TPU_RAIDEN_ENGINE_ID": f"{args.role}-engine-{args.run_id}",
                "TPU_RAIDEN_RESHARD_IMPL": "store",
                "TPU_RAIDEN_ADVERTISE_HOST": my_ip,
                "TPU_RAIDEN_RESHARD_PORT_BASE": str(ctrl_port),
                "TPU_RAIDEN_STORE_DISPATCH_PORT_BASE": str(ctrl_port + 100),
                "TPU_KV_TRANSFER_PORT": str(kv_xfer_port),
            }
        )
        kv_cfg = {
            "kv_connector": "TPURaidenConnector",
            "kv_connector_module_path": "vllm_torchtpu.distributed.kv_transfer.tpu_connector",
            "kv_role": "kv_producer" if is_prefill else "kv_consumer",
            "kv_port": 14579,
        }
        vllm_cmd += ["--kv-transfer-config", json.dumps(kv_cfg)]

    logs_dir = Path("/tmp/vllm_logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{args.run_id}_sub{sub_idx}_phys{phys_worker_id}.log"
    log(tag, f"Launching vLLM -> {log_file}: {' '.join(vllm_cmd)}")

    with open(log_file, "w", buffering=1) as lf:
        lf.write(f"# CMD: {' '.join(vllm_cmd)}\n")
        proc = subprocess.Popen(
            vllm_cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def _handle_sig(signum, _frame):
            proc.terminate()
            if mds_proc is not None:
                mds_proc.terminate()

        signal.signal(signal.SIGTERM, _handle_sig)
        signal.signal(signal.SIGINT, _handle_sig)

        def _tier2_janitor() -> None:
            # torch_tpu writes a Tier-2 copy of every compiled executable to /dev/shm/torch_tpu_cache (tmpfs,
            # counts against host RAM). The same binaries live in the disk-backed Tier-3 cache, so stale Tier-2
            # entries are safe to drop; this kept the v5p EP=16 bring-up clear of the kubelet memory-eviction line.
            root = Path("/dev/shm/torch_tpu_cache")
            while proc.poll() is None:
                time.sleep(120)
                cutoff = time.time() - 240
                for f in root.rglob("*.bin"):
                    try:
                        if f.stat().st_mtime < cutoff:
                            f.unlink()
                    except OSError:
                        pass

        if args.prune_tier2_shm:
            import threading

            threading.Thread(target=_tier2_janitor, daemon=True).start()

        assert proc.stdout is not None
        final_pass = 1 if enable_dspark else 0
        shards_deleted = False
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
            if (
                args.delete_engram_after_load
                and not shards_deleted
                and f"[FAST-GCS-STAGE] [pass={final_pass}] Completed all" in line
            ):
                # Engram shards 00047/00048 are copied into process memory at load (no fd / mmap remains),
                # so the 189 GiB tmpfs copy is redundant once the final pass completes on this host.
                def _delete_shards() -> None:
                    time.sleep(30)
                    for idx in (47, 48):
                        try:
                            os.unlink(f"{args.model_path}/model-{idx:05d}-of-00048.safetensors")
                        except OSError:
                            pass
                    Path(f"{args.model_path}/_ENGRAM_SHARDS_DELETED.txt").write_text(
                        f"deleted {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} after load pass {final_pass}\n"
                    )
                    log(tag, "Deleted /dev/shm engram shards 00047-00048 after final load pass (frees ~189 GiB host RAM)")

                import threading as _th

                _th.Thread(target=_delete_shards, daemon=True).start()
                shards_deleted = True
        rc = proc.wait()

    if mds_proc is not None:
        mds_proc.terminate()
    sys.exit(rc)


if __name__ == "__main__":
    main()
