# DeepSeek-V4.1-Flash-1017 Inference on Google Cloud TPU v5p

Colocated Expert-Parallel (`EP=16` and `EP=32`) serving benchmarks for [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) on Google Cloud TPU `v5p-16` and `v5p-32` slices with `vllm-tpu`.

**Summary:** With three Pallas/SparseCore compatibility patches, `DeepSeek-V4.1-Flash-1017` (`MXFP4` routed experts + `FP8` attention) runs colocated across 16 and 32 `v5p` chips, reaching **2,087.1 output tok/s (130.4 tok/s/chip) on `v5p-16` (`EP=16`) and 3,964.1 output tok/s (123.9 tok/s/chip) on `v5p-32` (`EP=32`) on `2048:1024` prompts (1.90× total slice throughput, 5.0% lower per-chip throughput at `EP=32` due to 8-host ICI MoE collectives)**, and **4,965.0 output tok/s (155.2 tok/s/chip) on `v5p-32` with cached prefixes**. Because TPU `v5p` lacks native `FP8`/`MXFP4` hardware (unpacking `MXFP4` to `FP8` in software each step) and has `2.8 TB/s` HBM bandwidth, per-chip throughput on `2048:1024` is **24.5× lower than disaggregated TPU `v7x` (`3,195.9 tok/s/chip`)** and per-user speed remains `3.7–9.8 tok/s/user`.

---

## 1. Model, Framework, and Hardware

- **Model:** [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (`DeepseekV4ForCausalLM`, `510.3 GB` across 48 `safetensors` shards; `131,072` max context length).
  - **Architecture & Weights:** 40 transformer layers (`hidden_size = 5,120`), **384 routed `MXFP4` (`W4A8`) experts + 1 shared expert** (`top-6` routing), sparse `CSA` MLA (`FP8`, `index_topk = 512`) + `SWA` latent sliding-window attention (`BF16`), **`189.1 GiB` CPU-offloaded Engram tables** in host RAM, **`FP8` KV cache**, and **`DSpark` speculative decoding** (`K = 5` draft tokens; `128` draft routed experts).
- **Framework:** `vllm/vllm-tpu` + `vllm-torchtpu` (`libtpu` `0.0.48`, `jax` `0.10.2`), patched at startup via [`scripts/apply_v5p_compat_patch.py`](./scripts/apply_v5p_compat_patch.py) to unpack `E2M1` `MXFP4` weights in-kernel on `v5p`, route `8`-lane SparseCore top-k/union ops to TensorCore `lax.top_k`/`lax.sort`, and fit `H=128` SWA tiles inside `63.9 MiB` VMEM.
- **Hardware:** Google Cloud TPU `v5p` (`ct5p-hightpu-4t` hosts, `4` chips/host, `95.7 GiB` HBM2e and `~2.8 TB/s` HBM bandwidth per chip):
  - **`v5p-16` (`2x2x4` ICI):** 4 hosts × 4 chips = **16 chips** (`DP=16, EP=16`)
  - **`v5p-32` (`2x4x4` ICI):** 8 hosts × 4 chips = **32 chips** (`DP=32, EP=32`)

---

## 2. Main Results

### A. Expert-Parallel Scaling (`EP=16` vs. `EP=32` on `2048` Input : `1024` Output) & `v7x` Comparison

All `v5p` runs are colocated (`max_num_seqs = 48`, `max_num_batched_tokens = 288` per DP rank, `DSpark K=5`, `ignore_eos=True`, 60 s warmup + 300 s steady window, **0 failed requests**; [`results/ep16_v5p16/`](./results/ep16_v5p16/), [`results/ep32_v5p32/`](./results/ep32_v5p32/)).

| Configuration | Chips | Best Concurrency | Steady Output `tok/s` | Output `tok/s / Chip` | Median User `tok/s` | `DSpark` $\tau$ | Comparison |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **`v5p-16` Colocated (`EP=16`, uncached)** | 16 | `c = 512` | **2,087.1** | **130.44** | 6.28 (`TTFT p50 83.3 s`) | 3.88 | `v5p` `EP=16` baseline (`1.00×`) |
| **`v5p-32` Colocated (`EP=32`, uncached)** | 32 | `c = 1024` | **3,964.1** | **123.88** | 6.02 (`TTFT p50 87.6 s`) | 3.81 | **1.90× higher slice tok/s**, **5.0% lower tok/s/chip** than `v5p-16` (`EP=16`) |
| **`v5p-32` Colocated (`EP=32`, cached prefix)** | 32 | `c = 1536` | **4,965.0** | **155.16** | 4.90 (`TTFT p50 101.9 s`) | 3.87 | **1.25× higher tok/s/chip** than uncached `v5p-32` (isolates decode from prefill) |
| Reference: TPU `v7x-16` `1P3D` (`EP=8`, uncached) | 16 | `c = 1024` | 51,134.9 | 3,195.93 | 76.53 (`TTFT p50 6.98 s`) | 4.17 | `v5p-16` is **24.5× lower tok/s/chip** than native-`FP8`/`MXFP4` TPU `v7x` ([`v7x` repo](https://github.com/ankdeshm/deepseek-v4.1-flash-1017-tpu-vllm-inference)) |

### B. Peak Throughput by Prompt Shape (`EP=16` on `v5p-16` & `EP=32` on `v5p-32`)

| Slice (`EP`) | Workload (`ISL : OSL`) | Peak Concurrency | Steady Output `tok/s` | Output `tok/s / Chip` | Median User `tok/s` (`c=32..64` Peak → Peak Load) |
| :--- | :--- | ---: | ---: | ---: | :--- |
| `v5p-16` (`EP=16`) | `1024 : 1024` (uncached) | `c = 768` | **2,688.1** | **168.00** | `9.84 → 5.42 tok/s/user` |
| `v5p-16` (`EP=16`) | `2048 : 1024` (uncached) | `c = 512` | **2,087.1** | **130.44** | `9.36 → 6.28 tok/s/user` |
| `v5p-16` (`EP=16`) | `8192 : 1024` (uncached) | `c = 256` | **748.0** | **46.75** | `7.53 → 6.92 tok/s/user` |
| `v5p-32` (`EP=32`) | `2048 : 1024` (uncached) | `c = 1024` | **3,964.1** | **123.88** | `8.53 → 6.02 tok/s/user` |
| `v5p-32` (`EP=32`) | `2048 : 1024` (`2,040`-tok cached prefix) | `c = 1536` | **4,965.0** | **155.16** | `4.33 → 4.90 tok/s/user` |

### C. Measured HBM & `FP8` KV-Cache Capacity

| Slice (`EP`) | Chips | Weights + Runtime / Chip | Available KV HBM / Chip (`0.85` Cap) | `FP8` KV Tokens / Chip | Total Slice KV Capacity |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **`v5p-16` (`EP=16`)** | 16 | **31.84 GiB** (of `95.73 GiB`) | **49.53 GiB** | **6,034,227** | **96.5M tokens** |
| **`v5p-32` (`EP=32`)** | 32 | **20.64 GiB** (of `95.73 GiB`) | **60.73 GiB** | **7,399,833** | **236.8M tokens** (**2.45×** vs. `v5p-16`) |

---

## 3. Reproducing the Main Results

Deploy a `v5p-16` or `v5p-32` `LeaderWorkerSet` using [`k8s/lws_colocated_v5p16_ep16.yaml`](./k8s/lws_colocated_v5p16_ep16.yaml) or [`k8s/lws_colocated_v5p32_ep32.yaml`](./k8s/lws_colocated_v5p32_ep32.yaml) (which run [`scripts/in_pod_setup_v5p.py`](./scripts/in_pod_setup_v5p.py) and [`scripts/apply_v5p_compat_patch.py`](./scripts/apply_v5p_compat_patch.py)).

### 1. Launch Colocated `vLLM` Server (`EP=16` on `v5p-16` or `EP=32` on `v5p-32`)
Script: [`scripts/v5p_vllm_driver.py`](./scripts/v5p_vllm_driver.py):

```bash
# On each pod of a v5p-16 (2x2x4) LeaderWorkerSet (use --ep-size 32 --slice-topology 2x4x4 on v5p-32):
python3 scripts/v5p_vllm_driver.py \
  --run-id coloc_ep16 \
  --role colocated \
  --ep-size 16 \
  --slice-topology 2x2x4 \
  --prune-tier2-shm \
  --delete-engram-after-load
```

### 2. Run Steady-State Concurrency Sweeps
Script: [`scripts/benchmark_v5p_point.py`](./scripts/benchmark_v5p_point.py):

```bash
# Uncached S1 (2048:1024) point (repeat across concurrency 32..1536; pass --mode cached_prefix for cached decode)
python3 scripts/benchmark_v5p_point.py \
  --base-url http://<LEADER_POD_IP>:8000 \
  --isl 2048 \
  --osl 1024 \
  --concurrency 512 \
  --warmup-s 60 \
  --steady-s 300 \
  --out-dir /tmp/dsv41_v5p_s1_c512
```

---

## 4. Detailed Concurrency Ladders, Compatibility Patches, and Raw Logs

- **[`APPENDIX.md`](./APPENDIX.md):** Complete per-concurrency tables for `EP=16` (`S1`, `S2`, `S3`) and `EP=32` (`S1` and cached-prefix decode), `v5p` Pallas/SparseCore compatibility patch details, why intra-slice `P/D` sub-slicing and `EP=24` are unsupported on `v5p`, host-RAM kubelet eviction fixes, and Spot preemption notes.
- **[`results/`](./results/):** Sub-slicing probe JSONs ([`subslice_probe/`](./results/subslice_probe/)) and raw per-concurrency summaries and Prometheus time series under [`ep16_v5p16/`](./results/ep16_v5p16/) and [`ep32_v5p32/`](./results/ep32_v5p32/).
