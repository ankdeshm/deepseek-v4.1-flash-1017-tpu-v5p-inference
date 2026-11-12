# Appendix: Full Concurrency Sweeps, `v5p` Kernel Patches, and Engineering Notes (`DeepSeek-V4.1-Flash-1017` on TPU v5p)

This document preserves all per-concurrency ladders (`EP=16` and `EP=32`), TPU `v5p` kernel compatibility patches, sub-slicing probe findings, and host-memory mitigation details. For headline comparisons and reproduction commands, see [`README.md`](./README.md).

---

## 1. Porting `DeepSeek-V4.1-Flash-1017` Kernels from TPU `v7x` to TPU `v5p`

The upstream `vllm-torchtpu` DeepSeek-V4.1 kernels assume TPU `v7x` hardware (`FP8`/`MXFP4` datapaths, `16`-lane SparseCore, and `v7x` VMEM sizes). Applying [`scripts/apply_v5p_compat_patch.py`](./scripts/apply_v5p_compat_patch.py) resolves three `v5p` incompatibilities without falling back to `BF16`-dequantized HBM weights:

1. **In-kernel `MXFP4` (`E2M1`) → `FP8` unpacking:** Keeps routed expert weights packed as 4-bit `MXFP4` in HBM (`weight_format=fp4, wire_dtype=float8_e4m3fn`) and unpacks `E2M1 → FP32 → FP8` via `uint4` bitcast inside the fused MoE kernel on every step.
2. **SparseCore `8`-lane top-k / union fallback:** Routes `v7x` 16-lane SparseCore top-k and index-union kernels to exact TensorCore `lax.top_k` / `lax.sort` implementations on `v5p`.
3. **Sliding-window attention (`H=128`) VMEM tile sizing:** Resizes the `H=128` SWA Pallas tile to fit inside `v5p`'s `63.9 MiB` VMEM limit. All custom Pallas/Mosaic kernels passed AOT compilation across `2x2x2`, `2x2x4`, and `2x4x4` `v5p` topologies.

### Why Intra-Slice `P/D` Disaggregation and `EP=24` Did Not Run on `v5p`
- **No multi-host sub-slicing inside one GKE `v5p` ICI domain ([`results/subslice_probe/`](./results/subslice_probe/)):** Live probes confirmed `2x2x4 → 2x2x2 + 2x2x2` (`false`) and `2x4x4 → 16+16 / 8+24 / 16+8` (`false`) because `libtpu` / `vbar.sock` binds multi-host ICI rings to the VM slice metadata. Consequently, single-slice runs execute **colocated** (`DP = EP = 16` or `32`), sharing the `288`-token step budget between `DSpark` (`K=5`) decode verification and chunked prefill.
- **`EP=24` is incompatible with `DSpark` draft experts:** The `DSpark` speculative drafter has `dspark_n_routed_experts = 128`, which is not divisible by `24` (`128 % 24 != 0`).
- **`EP=8` requires a dedicated `2x2x2` nodepool:** One 4-chip `v5p` host (`380 GiB` total HBM) cannot hold the checkpoint weights, and sub-slicing an `8`-chip ring out of `v5p-16` or `v5p-32` is blocked by `vbar.sock`.

---

## 2. Full Concurrency Sweeps on `v5p-16` (`EP=16`, `16` Chips, `2x2x4`)

Common settings: `vllm serve` with `--data-parallel-size 16 --data-parallel-size-local 4 --enable-expert-parallel --kv-cache-dtype fp8 --max-model-len 131072 --max-num-seqs 48 --max-num-batched-tokens 288 --gpu-memory-utilization 0.85 --enable-prefix-caching --async-scheduling --speculative-config '{"method":"dspark","num_speculative_tokens":5}'`, `ignore_eos=True`, 60 s warmup + 300 s steady window ([`results/ep16_v5p16/`](./results/ep16_v5p16/)).

### A. `S1 = 2048 : 1024` (Uncached)

| Concurrency | Steady Out `tok/s` | Out `tok/s / Chip` | Per-User `Med / P90_low` (`tok/s`) | TTFT `p50 / p95` (s, All) | TPOT `p50` (ms) | `DSpark` $\tau$ | Batch / Rank | KV Mean / Peak | Failures |
| ---: | ---: | ---: | :--- | :--- | ---: | ---: | ---: | :--- | ---: |
| 32 | 236.29 | 14.77 | 9.36 / 7.82 | 18.21 / 37.79 | 106.84 | 3.87 | 1.97 | 0.32% / 0.81% | 0 / 96 |
| 64 | 441.86 | 27.62 | 8.28 / 7.19 | 11.45 / 29.44 | 120.85 | 3.89 | 3.95 | 0.62% / 1.14% | 0 / 192 |
| 128 | 492.16 | 30.76 | 4.94 / 4.40 | 27.26 / 71.48 | 202.41 | 3.81 | 7.83 | 1.21% / 1.67% | 0 / 255 |
| 256 | 845.78 | 52.86 | 3.66 / 3.38 | 31.03 / 131.49 | 272.95 | 3.85 | 15.41 | 2.29% / 2.78% | 0 / 425 |
| 384 | 1,600.83 | 100.05 | 5.07 / 4.79 | 42.57 / 159.99 | 197.35 | 3.87 | 21.59 | 3.15% / 3.91% | 0 / 758 |
| **512** | **2,087.08** | **130.44** | **6.28 / 6.10** | **83.32 / 193.15** | **159.25** | **3.88** | **23.90** | **3.47% / 4.74%** | **0 / 990** |
| 768 | 2,072.20 | 129.51 | 6.16 / 6.00 | 196.25 / 312.59 | 162.37 | 3.89 | 23.92 | 3.48% / 4.77% | 0 / 1,242 |

### B. `S2 = 1024 : 1024` (Uncached)

| Concurrency | Steady Out `tok/s` | Out `tok/s / Chip` | Per-User `Med / P90_low` (`tok/s`) | TTFT `p50 / p95` (s, All) | TPOT `p50` (ms) | `DSpark` $\tau$ | Batch / Rank | KV Mean / Peak | Failures |
| ---: | ---: | ---: | :--- | :--- | ---: | ---: | ---: | :--- | ---: |
| 32 | 253.77 | 15.86 | 9.32 / 8.35 | 8.65 / 14.89 | 107.25 | 3.72 | 1.95 | 0.28% / 0.71% | 0 / 127 |
| 64 | 510.26 | 31.89 | 9.84 / 7.95 | 7.65 / 12.42 | 101.62 | 3.72 | 3.92 | 0.54% / 1.02% | 0 / 239 |
| 128 | 970.31 | 60.64 | 8.99 / 7.70 | 8.18 / 17.86 | 111.19 | 3.70 | 7.88 | 1.07% / 1.64% | 0 / 384 |
| 256 | 1,368.17 | 85.51 | 5.69 / 5.57 | 11.52 / 39.15 | 175.82 | 3.70 | 15.85 | 2.09% / 2.47% | 0 / 512 |
| 384 | 1,964.53 | 122.78 | 5.61 / 5.49 | 19.96 / 63.15 | 178.34 | 3.70 | 23.54 | 3.10% / 3.67% | 0 / 769 |
| 512 | 2,302.21 | 143.89 | 5.13 / 5.02 | 35.64 / 103.85 | 194.84 | 3.69 | 29.53 | 3.89% / 4.67% | 0 / 1,020 |
| **768** | **2,688.08** | **168.00** | **5.42 / 5.29** | **94.87 / 229.05** | **184.66** | **3.70** | **33.45** | **4.36% / 5.72%** | **0 / 1,380** |

### C. `S3 = 8192 : 1024` (Uncached)

| Concurrency | Steady Out `tok/s` | Out `tok/s / Chip` | Per-User `Med / P90_low` (`tok/s`) | TTFT `p50 / p95` (s, All) | TPOT `p50` (ms) | `DSpark` $\tau$ | Batch / Rank | KV Mean / Peak | Failures |
| ---: | ---: | ---: | :--- | :--- | ---: | ---: | ---: | :--- | ---: |
| 32 | 203.68 | 12.73 | 7.53 / 6.23 | 36.48 / 84.76 | 132.81 | 4.12 | 1.97 | 0.49% / 0.98% | 0 / 76 |
| 64 | 286.10 | 17.88 | 6.77 / 6.62 | 35.06 / 71.02 | 147.71 | 4.13 | 3.98 | 0.96% / 1.68% | 0 / 128 |
| 128 | 548.91 | 34.31 | 5.45 / 5.24 | 53.34 / 168.68 | 183.36 | 4.08 | 7.67 | 1.78% / 2.38% | 0 / 252 |
| **256** | **748.01** | **46.75** | **6.92 / 6.73** | **171.57 / 292.52** | **144.51** | **4.09** | **11.09** | **2.30% / 2.92%** | **0 / 420** |
| 384 | 745.98 | 46.62 | 6.91 / 6.65 | 316.92 / 452.61 | 144.63 | 4.09 | 11.06 | 2.29% / 2.88% | 0 / 547 |

---

## 3. Full Concurrency Sweeps on `v5p-32` (`EP=32`, `32` Chips, `2x4x4`)

Measured on `32` `v5p` chips (`DP=32, EP=32`, `1,536` total decode slots across 32 ranks; [`results/ep32_v5p32/`](./results/ep32_v5p32/)).

### A. `S1 = 2048 : 1024` (Uncached)

| Concurrency | Steady Out `tok/s` | Out `tok/s / Chip` | Per-User `Med / P90_low` (`tok/s`) | TTFT `p50 / p95` (s, All) | TPOT `p50` (ms) | `DSpark` $\tau$ | Batch / Rank | KV Mean / Peak | Failures |
| ---: | ---: | ---: | :--- | :--- | ---: | ---: | ---: | :--- | ---: |
| 32 | 190.93 | 5.97 | 7.35 / 6.27 | 21.18 / 45.54 | 136.12 | 3.79 | 0.98 | 0.14% / 0.55% | 0 / 94 |
| 64 | 349.32 | 10.92 | 8.53 / 5.04 | 23.61 / 49.90 | 117.21 | 3.83 | 1.95 | 0.26% / 0.68% | 0 / 173 |
| 128 | 787.82 | 24.62 | 7.02 / 5.92 | 14.08 / 28.31 | 142.54 | 3.82 | 3.93 | 0.51% / 0.89% | 0 / 380 |
| 256 | 772.54 | 24.14 | 3.37 / 3.06 | 33.30 / 87.37 | 296.81 | 3.79 | 7.95 | 0.97% / 1.32% | 0 / 337 |
| 384 | 1,383.71 | 43.24 | 3.85 / 3.76 | 20.33 / 58.12 | 260.09 | 3.81 | 11.90 | 1.44% / 1.89% | 0 / 765 |
| 512 | 1,666.98 | 52.09 | 3.54 / 3.45 | 26.43 / 81.95 | 282.19 | 3.80 | 15.76 | 1.88% / 2.27% | 0 / 913 |
| 768 | 3,257.00 | 101.78 | 5.08 / 4.96 | 43.95 / 133.49 | 196.73 | 3.80 | 22.18 | 2.63% / 3.18% | 0 / 1,532 |
| **1,024** | **3,964.05** | **123.88** | **6.02 / 5.88** | **87.61 / 204.00** | **166.04** | **3.81** | **23.80** | **2.82% / 3.95%** | **0 / 1,901** |
| 1,536 | 3,956.31 | 123.63 | 6.02 / 5.88 | 205.84 / 325.95 | 166.01 | 3.82 | 23.70 | 2.81% / 3.85% | 0 / 2,414 |

**Why `EP=32` did not improve per-chip decode throughput over `EP=16`:** At matched batch per rank (`~23.8–23.9` active sequences/rank), `EP=16` (`v5p-16` `c=512`) achieved `130.44 tok/s/chip` (`TPOT p50 159.3 ms`), while `EP=32` (`v5p-32` `c=1024`) achieved `123.88 tok/s/chip` (`TPOT p50 166.0 ms`, `-5.0%`). Expanding the MoE all-to-all dispatch/combine across 8 hosts (`2x4x4`) instead of 4 hosts (`2x2x4`) adds ICI collective latency and synchronizes 32 ranks on the slowest rank's XLA compile bucket (`c=256` shows the same bucket-crossing dip seen on `EP=16` at `c=128–256`). What `EP=32` *does* increase is per-chip HBM headroom (`20.64 GiB` vs `31.84 GiB` used for weights/runtime) and aggregate KV capacity (`236.8M` vs `96.5M` `FP8` tokens).

### B. Cached-Prefix Decode Ceiling (`2,040`-Token Shared Prefix, `2048 : 1024`, `EP=32`)

| Concurrency | Realized Prefix Hit | Steady Out `tok/s` | Out `tok/s / Chip` | Per-User `Med / P90_low` (`tok/s`) | TTFT `p50 / p95` (s, All) | TPOT `p50` (ms) | `DSpark` $\tau$ | Batch / Rank | Failures |
| ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 256 | 15.26% | 688.08 | 21.50 | 4.33 / 3.54 | 25.29 / 91.33 | 230.92 | 3.92 | 7.97 | 0 / 315 |
| 384 | 28.04% | 874.41 | 27.33 | 3.41 / 3.06 | 46.70 / 201.73 | 293.43 | 3.93 | 11.76 | 0 / 414 |
| 512 | 39.52% | 1,090.78 | 34.09 | 3.53 / 3.24 | 27.27 / 81.09 | 283.08 | 3.96 | 15.97 | 0 / 571 |
| 768 | 47.87% | 1,947.93 | 60.87 | 3.93 / 3.39 | 33.16 / 174.21 | 254.69 | 3.94 | 21.98 | 0 / 882 |
| 1,024 | 49.79% | 2,869.24 | 89.66 | 3.65 / 3.18 | 39.68 / 132.91 | 273.78 | 3.94 | 30.99 | 0 / 1,499 |
| **1,536** | **49.98%** | **4,965.04** | **155.16** | **4.90 / 4.58** | **101.95 / 248.11** | **204.13** | **3.87** | **32.77** | **0 / 2,673** |

---

## 4. Operational Lessons on TPU `v5p`

1. **Host RAM (`446 GiB` allocatable on `ct5p-hightpu-4t`) is tighter than TPU HBM:** Staging `189.1 GiB` of CPU-offloaded Engram shards into `/dev/shm` alongside `46 GiB` of XLA compile artifacts caused a kubelet memory eviction (`421.8 GiB` working set) on the first `EP=16` launch. Moving compile caches to boot disk (`VLLM_DISABLE_COMPILE_CACHE=1`), pruning stale Tier-2 executables (`--prune-tier2-shm`), and unlinking `/dev/shm` Engram shard files once loaded into process memory (`--delete-engram-after-load` in [`scripts/v5p_vllm_driver.py`](./scripts/v5p_vllm_driver.py)) restored `>200 GiB` free host RAM per node.
2. **Cold-start duration on Spot slices:** Loading 510 GB of weights plus compiling 7 XLA token buckets without a warm compile cache takes `~90–109 minutes` to `/health 200` (`~32 min` weight load + `~60–70 min` XLA compile). On `v5p-32` Spot capacity, five Spot preemptions interrupted bring-up before the uninterrupted window that completed `EP=32` `S1` and cached-prefix sweeps.
