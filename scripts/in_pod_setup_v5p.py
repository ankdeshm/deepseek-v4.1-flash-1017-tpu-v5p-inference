#!/usr/bin/env python3
"""Idempotent per-host TPU v5p environment setup & hybrid checkpoint stager.

Designed specifically for ct5p-hightpu-4t (~448 GiB host RAM, 4 v5p chips/host):
1. Installs stack_27be32f509754e88.tar + libtpu==0.0.48 + whls + whls_pd + editable vllm/vllm-torchtpu.
2. Applies apply_v5p_compat_patch.py (MXFP4 uint4 E2M1 unpack, SparseCore 8-lane -> TC topk/union,
   multi-host DP TCPStore barrier).
3. Compiles metadata_hook_8088.so for multi-host sub-slice metadata redirection.
4. Stages Hybrid /dev/shm/DeepSeek-V4.1-Flash checkpoint (~189.12 GiB in tmpfs):
   - Downloads ONLY model-00047-of-00048.safetensors (94.56 GiB) and
     model-00048-of-00048.safetensors (94.56 GiB) directly into /dev/shm via parallel
     GCS HTTP range requests (~4 GB/s) so all 4 local workers share a single zero-copy
     safetensors mmap for CPU-offloaded engram tables without exceeding 448 GiB host RAM.
   - Copies config/tokenizer JSONs into /dev/shm/DeepSeek-V4.1-Flash.
   - Symlinks model-00001..00046-of-00048.safetensors to GCSFuse /models/models/deepseek-ai/DeepSeek-V4.1-Flash-1017
     (loaded shard-by-shard with --enable-ep-weight-filter).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request

DEFAULT_BUCKET = "YOUR_GCS_BUCKET"
MODEL_GCS_PREFIX = "models/deepseek-ai/DeepSeek-V4.1-Flash-1017"
ENGRAM_SHARDS = [
    "model-00047-of-00048.safetensors",
    "model-00048-of-00048.safetensors",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] [v5p-setup] {msg}", flush=True)


def get_access_token() -> str:
    req = urllib.request.Request(
        "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))["access_token"]


def gcs_head_size(bucket: str, obj_name: str, token: str) -> int:
    meta_url = (
        f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/"
        f"{urllib.parse.quote(obj_name, safe='')}"
    )
    req = urllib.request.Request(meta_url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(json.loads(resp.read().decode("utf-8"))["size"])


def gcs_parallel_download(
    bucket: str,
    obj_name: str,
    dst: Path,
    num_threads: int = 12,
    chunk_mb: int = 64,
) -> float:
    t0 = time.time()
    token = get_access_token()
    total_size = gcs_head_size(bucket, obj_name, token)
    if dst.exists() and not dst.is_symlink() and dst.stat().st_size == total_size:
        log(f"Already staged {dst.name} ({total_size / (1024**3):.2f} GiB)")
        return 0.0

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_suffix(".part")
    with open(tmp_dst, "wb") as f:
        f.truncate(total_size)

    media_url = (
        f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/"
        f"{urllib.parse.quote(obj_name, safe='')}?alt=media"
    )
    part_bytes = chunk_mb * 1024 * 1024
    ranges: list[tuple[int, int]] = []
    pos = 0
    while pos < total_size:
        end = min(pos + part_bytes - 1, total_size - 1)
        ranges.append((pos, end))
        pos = end + 1

    def fetch_range(rng: tuple[int, int]) -> int:
        start, end = rng
        for attempt in range(4):
            try:
                tok = token if attempt == 0 else get_access_token()
                req = urllib.request.Request(
                    media_url,
                    headers={
                        "Authorization": f"Bearer {tok}",
                        "Range": f"bytes={start}-{end}",
                    },
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = resp.read()
                if len(data) != (end - start + 1):
                    raise IOError(f"Short read {len(data)} != {end - start + 1}")
                with open(tmp_dst, "r+b") as wf:
                    wf.seek(start)
                    wf.write(data)
                return len(data)
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(1.5 * (attempt + 1))
        return 0

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futs = [pool.submit(fetch_range, r) for r in ranges]
        for fut in as_completed(futs):
            fut.result()

    tmp_dst.replace(dst)
    dt = time.time() - t0
    gib = total_size / (1024**3)
    log(f"Downloaded {dst.name} ({gib:.2f} GiB) in {dt:.1f}s ({gib / max(dt, 1e-3):.2f} GiB/s)")
    return dt


def install_stack(scripts_dir: Path, stack_tar: Path) -> None:
    marker = Path("/workspace/.v5p_stack_ready_v1")
    if not marker.exists():
        log(f"Extracting {stack_tar} and installing wheels...")
        if not stack_tar.exists():
            stack_tar = Path("/tmp/stack_27be32f509754e88.tar")
            gcs_parallel_download(
                DEFAULT_BUCKET, "dsv41-stack/stack_27be32f509754e88.tar", stack_tar, num_threads=8
            )
        install_cmd = f"""
set -euo pipefail
mkdir -p /tmp/pip_tmp
export TMPDIR=/tmp/pip_tmp
tar --overwrite --keep-directory-symlink -C / -xf "{stack_tar}"
git config --global --add safe.directory '*'
for sc_dst in /usr/local/lib/python3.12/sitecustomize.py /usr/local/lib/python3.12/site-packages/sitecustomize.py /usr/lib/python3.12/sitecustomize.py; do
  mkdir -p "$(dirname "$sc_dst")" && cp -f /workspace/sitecustomize_bench/sitecustomize.py "$sc_dst" || true
done
pip uninstall -y tpu-inference torchax >/dev/null 2>&1 || true
pip install --no-cache-dir --no-deps "libtpu==0.0.48" portpicker==1.6.0 compressed-tensors==0.15.0 sympy aiohttp >/dev/null
pip install --no-cache-dir --no-index --find-links=/workspace/whls --no-deps --force-reinstall /workspace/whls/*.whl >/dev/null
if ls /workspace/whls_pd/*.whl >/dev/null 2>&1; then
  pip install --no-cache-dir --no-index --no-deps --force-reinstall /workspace/whls_pd/*.whl >/dev/null
fi
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.29.0 SETUPTOOLS_SCM_PRETEND_VERSION=0.29.0 VLLM_TARGET_DEVICE=empty pip install --no-cache-dir --no-index --no-deps --no-build-isolation -e /workspace/vllm >/dev/null
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install --no-cache-dir --no-index --no-deps --no-build-isolation -e /workspace/vllm-torchtpu >/dev/null
"""
        subprocess.run(["bash", "-lc", install_cmd], check=True)
        marker.write_text("ok\n")

    # Always apply idempotent v5p compatibility patch and sync any copied site-packages dirs
    patch_script = scripts_dir / "apply_v5p_compat_patch.py"
    subprocess.run(
        [sys.executable, str(patch_script), "--src-root", "/workspace/vllm-torchtpu/src/vllm_torchtpu"],
        check=True,
    )
    sync_cmd = r"""
set -euo pipefail
for d in /usr/local/lib/python3.12/*-packages/vllm_torchtpu; do
  [ -d "$d" ] && [ "$(realpath "$d")" != "$(realpath /workspace/vllm-torchtpu/src/vllm_torchtpu)" ] && cp -rf /workspace/vllm-torchtpu/src/vllm_torchtpu/* "$d/" || true
done
for d in /usr/local/lib/python3.12/*-packages/vllm; do
  [ -d "$d" ] && [ "$(realpath "$d")" != "$(realpath /workspace/vllm/vllm)" ] && cp -rf /workspace/vllm/vllm/* "$d/" || true
done
"""
    subprocess.run(["bash", "-lc", sync_cmd], check=True)

    # Compile metadata_hook_8088.so and copy v5p_subslice_metadata.py into /workspace
    hook_src = scripts_dir / "metadata_hook_8088.c"
    subprocess.run(
        ["gcc", "-shared", "-fPIC", "-O2", str(hook_src), "-o", "/workspace/metadata_hook_8088.so", "-ldl"],
        check=True,
    )
    shutil.copy2(scripts_dir / "v5p_subslice_metadata.py", "/workspace/v5p_subslice_metadata.py")
    log("Stack + TPU v5p compatibility patch + metadata_hook_8088.so ready.")


def stage_hybrid_checkpoint(
    bucket: str,
    gcsfuse_model_dir: Path,
    shm_model_dir: Path,
) -> dict[str, float]:
    t0 = time.time()
    shm_model_dir.mkdir(parents=True, exist_ok=True)
    if not gcsfuse_model_dir.exists():
        raise FileNotFoundError(f"GCSFuse model directory missing: {gcsfuse_model_dir}")

    # 1. Copy small metadata / tokenizer / index files into /dev/shm
    for item in sorted(gcsfuse_model_dir.iterdir()):
        if item.is_file() and not item.name.endswith(".safetensors"):
            dst = shm_model_dir / item.name
            if not dst.exists():
                shutil.copy2(item, dst)

    # 2. Symlink TPU HBM shards 00001..00046 to GCSFuse (loaded with --enable-ep-weight-filter)
    for idx in range(1, 47):
        shard_name = f"model-{idx:05d}-of-00048.safetensors"
        src = gcsfuse_model_dir / shard_name
        dst = shm_model_dir / shard_name
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() and dst.resolve() == src.resolve():
                continue
            dst.unlink()
        dst.symlink_to(src)

    # 3. Download CPU-offloaded engram shards 00047 & 00048 (189.12 GiB total) into /dev/shm
    dl_times: dict[str, float] = {}
    for shard_name in ENGRAM_SHARDS:
        dt = gcs_parallel_download(
            bucket,
            f"{MODEL_GCS_PREFIX}/{shard_name}",
            shm_model_dir / shard_name,
            num_threads=24,
            chunk_mb=64,
        )
        dl_times[shard_name] = round(dt, 2)

    total_s = round(time.time() - t0, 2)
    dl_times["total_hybrid_stage_s"] = total_s
    log(f"Hybrid checkpoint ready at {shm_model_dir} in {total_s:.2f}s: {dl_times}")
    return dl_times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts-dir", default="/models/dsv41-v5p/scripts")
    ap.add_argument("--stack-tar", default="/models/dsv41-stack/stack_27be32f509754e88.tar")
    ap.add_argument("--gcsfuse-model-dir", default="/models/models/deepseek-ai/DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--shm-model-dir", default="/dev/shm/DeepSeek-V4.1-Flash")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--skip-checkpoint", action="store_true")
    ap.add_argument("--summary-out", default="")
    args = ap.parse_args()

    t_start = time.time()
    install_stack(Path(args.scripts_dir), Path(args.stack_tar))
    stage_stats: dict[str, float] = {}
    if not args.skip_checkpoint:
        stage_stats = stage_hybrid_checkpoint(
            args.bucket,
            Path(args.gcsfuse_model_dir),
            Path(args.shm_model_dir),
        )
    summary = {
        "hostname": os.uname().nodename,
        "total_setup_s": round(time.time() - t_start, 2),
        "stage_stats": stage_stats,
    }
    if args.summary_out:
        out_p = Path(args.summary_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(summary, indent=2))
    log(f"Setup complete: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
