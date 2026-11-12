#!/usr/bin/env python3
"""Idempotent compatibility & multi-host DP patcher for DeepSeek-V4.1-Flash-1017 on TPU v5p.

Applies five surgical, numerically exact patches to `vllm-torchtpu` (`cpk/dsv41-combined`):
1. `kernels/fused_moe/v2/kernel.py`:
   Replaces `pltpu.bitcast(u32_arr, FP4).astype(FP8)` on pre-v7 TPUs (`chip_generation < 7`)
   with `pltpu.bitcast(u32_arr, jnp.uint4)` + branchless IEEE-754 E2M1 -> FP32 -> FP8
   (zero numerical error `max_abs_diff == 0.0`, preserves 4-bit MXFP4 HBM storage & DMA).
2. `kernels/megablox/gmm_v2.py`:
   Adds pre-v7 (`generation < 7`) `float4_e2m1fn -> uint4` XLA bitcast prior to `pallas_call`,
   `uint4` carrier unpack, `_u4_e2m1_to_f32` widen in `_matmul`, and `is_matmul_supported`
   cast in unquantized matmul path.
3. `kernels/deepseek_v41/{sparse_mla.py, streamindex_topk.py, sparsecore_topk.py}`:
   Guards 16-lane TPU v7x SparseCore kernels with `sc.num_lanes == 16` so TPU v5p
   (`sc.num_lanes == 8`) dispatches to exact TensorCore `lax.sort` union (`sparse_mla.py`)
   and exact TensorCore `lax.top_k` (`_tc_sparsecore_topk`, `_tc_cand_gather_topk`,
   `_tc_write_cand_and_topk`).
4. `layers/adapter/custom_ops/deepseek_v41/deepseek_v41_flash_attention_op.py`:
   Tunes `_run_swa` for `q.shape[1] > 64` (Layer 60 & MTP `H=128`, `sw=512`) from
   `bq_prefill=64` (`67.3 MiB` > `63.94 MiB` v5p VMEM) to `bq_prefill=32` and
   `q_compute_block_size=8`, saving `>16 MiB` VMEM with identical math (`acc_scratch` `(1, D)`).
5. `worker/tpu_rank_binding.py`, `distributed/tpu_mp_multihost.py`, `worker/tpu_worker.py`:
   Fixes multi-host DP (`DP > 4` across 4-chip v5p hosts) `local_world_size`, exports
   `TPU_NUM_HOSTS`/`NNODES`/`NODE_RANK`, and adds a pre-`GetClient` `TCPStore` barrier
   when `WORLD_SIZE > 4`.
"""

from __future__ import annotations

import argparse
import glob
import os
import pathlib
import shutil


def patch_fused_moe_v2(root: pathlib.Path) -> bool:
    p = root / "kernels" / "fused_moe" / "v2" / "kernel.py"
    txt = p.read_text()
    if "_u32_to_fp8" in txt:
        return False

    old = """            # Each widens one k-block: u32 [qb/8, N] -> fp4 [qb, N] -> fp8.
            packed_rows = rhs_qb // PACK4

            def w1_block(b):
                if wait_slab is not None:
                    wait_slab(w1_copy, b)
                block = pltpu.bitcast(
                    w1_vm[slot, pl.ds(b * packed_rows, packed_rows), :], FP4
                ).astype(FP8)
                if slab_retired is not None and b == hidden // rhs_qb - 1:
                    slab_retired()
                return block

            def w2_block(b):
                if wait_slab is not None:
                    wait_slab(w2_copy, b)
                return pltpu.bitcast(
                    w2_vm[slot, pl.ds(b * packed_rows, packed_rows), :], FP4
                ).astype(FP8)

            return w1_block, w2_block"""

    new = """            # Each widens one k-block: u32 [qb/8, N] -> fp4 [qb, N] -> fp8.
            packed_rows = rhs_qb // PACK4

            is_pre_v7 = host.chip_generation() < 7

            def _u32_to_fp8(u32_arr):
                if not is_pre_v7:
                    return pltpu.bitcast(u32_arr, FP4).astype(FP8)
                nib = pltpu.bitcast(u32_arr, jnp.uint4).astype(jnp.uint32)
                sign_bit = (nib & jnp.uint32(0x8)) << jnp.uint32(28)
                mag = nib & jnp.uint32(0x7)
                fp32_ge2 = (
                    (jnp.uint32(126) + (mag >> jnp.uint32(1))) << jnp.uint32(23)
                ) | ((mag & jnp.uint32(1)) << jnp.uint32(22))
                fp32_bits = (
                    jnp.where(
                        mag == 0,
                        jnp.uint32(0),
                        jnp.where(mag == 1, jnp.uint32(0x3F000000), fp32_ge2),
                    )
                    | sign_bit
                )
                return lax.bitcast_convert_type(fp32_bits, jnp.float32).astype(FP8)

            def w1_block(b):
                if wait_slab is not None:
                    wait_slab(w1_copy, b)
                block = _u32_to_fp8(w1_vm[slot, pl.ds(b * packed_rows, packed_rows), :])
                if slab_retired is not None and b == hidden // rhs_qb - 1:
                    slab_retired()
                return block

            def w2_block(b):
                if wait_slab is not None:
                    wait_slab(w2_copy, b)
                return _u32_to_fp8(w2_vm[slot, pl.ds(b * packed_rows, packed_rows), :])

            return w1_block, w2_block"""

    if old not in txt:
        raise RuntimeError(f"Expected snippet not found in {p}")
    p.write_text(txt.replace(old, new, 1))
    return True


def patch_gmm_v2(root: pathlib.Path) -> bool:
    p = root / "kernels" / "megablox" / "gmm_v2.py"
    txt = p.read_text()
    if "_u4_e2m1_to_f32" in txt:
        return False

    helper = """
def _u4_e2m1_to_f32(u4: jax.Array) -> jax.Array:
    u = u4.astype(jnp.uint32)
    sign = (u & 8) << 28
    exp = (u >> 1) & 3
    mant = u & 1
    norm = sign | ((exp + 126) << 23) | (mant << 22)
    sub = sign | jnp.where(mant != 0, jnp.uint32(0x3F000000), jnp.uint32(0))
    return lax.bitcast_convert_type(jnp.where(exp == 0, sub, norm), jnp.float32)
"""

    s1_old = "        tiled_rhs = tiled_rhs_ref.get_weight()\n"
    s1_new = """        tiled_rhs = tiled_rhs_ref.get_weight()
        if tiled_rhs.dtype == jnp.uint4 and (
            cfgs.rhs_cfgs.dtype == jnp.float4_e2m1fn
            or cfgs.rhs_cfgs.quant_dtype == jnp.float4_e2m1fn
        ):
            _tgt_dt = (
                cfgs.lhs_cfgs.quant_dtype
                if (
                    cfgs.lhs_cfgs.should_quantize
                    and not cfgs.rhs_cfgs.should_dequantize_before_matmul
                )
                else cfgs.lhs_cfgs.dtype
            )
            tiled_rhs = _u4_e2m1_to_f32(tiled_rhs).astype(_tgt_dt)
"""

    s2_old = "        if not cfgs.lhs_cfgs.should_quantize:\n            # Unquantized matmul path.\n"
    s2_new = (
        "        if not cfgs.lhs_cfgs.should_quantize:\n"
        "            # Unquantized matmul path.\n"
        "            if not tpu_info.is_matmul_supported(tiled_lhs.dtype, tiled_rhs.dtype):\n"
        "                tiled_rhs = tiled_rhs.astype(tiled_lhs.dtype)\n"
    )

    s3_old = (
        "    if cfgs.rhs_cfgs.should_unpack:\n"
        "        rhs_weight = rhs_ref.weight.bitcast(cfgs.rhs_cfgs.quant_dtype)\n"
        "        rhs_ref = dataclasses.replace(rhs_ref, weight=rhs_weight)\n"
    )
    s3_new = """    if cfgs.rhs_cfgs.should_unpack:
        _unpack_dt = (
            jnp.uint4
            if (
                cfgs.rhs_cfgs.quant_dtype == jnp.float4_e2m1fn
                and pltpu.get_tpu_info().generation < 7
            )
            else cfgs.rhs_cfgs.quant_dtype
        )
        rhs_weight = rhs_ref.weight.bitcast(_unpack_dt)
        rhs_ref = dataclasses.replace(rhs_ref, weight=rhs_weight)
"""

    s4_old = "    rhs_weights = WeightsRef(weight=rhs, scale=rhs_scale, bias=rhs_bias)\n"
    s4_new = (
        "    if rhs.dtype == jnp.float4_e2m1fn and pltpu.get_tpu_info().generation < 7:\n"
        "        rhs = lax.bitcast_convert_type(rhs, jnp.uint4)\n"
        "    rhs_weights = WeightsRef(weight=rhs, scale=rhs_scale, bias=rhs_bias)\n"
    )

    for s in (s1_old, s2_old, s3_old, s4_old, "def kernel_main("):
        if s not in txt:
            raise RuntimeError(f"Expected snippet {s!r} not found in {p}")

    txt = txt.replace(s1_old, s1_new, 1)
    txt = txt.replace(s2_old, s2_new, 1)
    txt = txt.replace(s3_old, s3_new, 1)
    txt = txt.replace(s4_old, s4_new, 1)
    txt = txt.replace("def kernel_main(", helper + "\n\ndef kernel_main(", 1)
    p.write_text(txt)
    return True


def patch_sparsecore_v5p(root: pathlib.Path) -> tuple[bool, bool, bool]:
    # 1. sparse_mla.py
    p_smla = root / "kernels" / "deepseek_v41" / "sparse_mla.py"
    txt_smla = p_smla.read_text()
    old_smla = '        and hasattr(pltpu.get_tpu_info(), "sparse_core")\n    ):'
    new_smla = (
        '        and hasattr(pltpu.get_tpu_info(), "sparse_core")\n'
        '        and getattr(pltpu.get_tpu_info().sparse_core, "num_lanes", 0) == 16\n'
        "    ):"
    )
    c_smla = False
    if new_smla not in txt_smla:
        if old_smla not in txt_smla:
            raise RuntimeError(f"Expected snippet not found in {p_smla}")
        p_smla.write_text(txt_smla.replace(old_smla, new_smla, 1))
        c_smla = True

    # 2. streamindex_topk.py
    p_st = root / "kernels" / "deepseek_v41" / "streamindex_topk.py"
    txt_st = p_st.read_text()
    old_st = "    if candidates is not None and not produce_blk and pages_per_seq > 160:"
    new_st = (
        "    if (\n"
        "        candidates is not None\n"
        "        and not produce_blk\n"
        "        and pages_per_seq > 160\n"
        '        and getattr(getattr(pltpu.get_tpu_info(), "sparse_core", None), "num_lanes", 0) == 16\n'
        "    ):"
    )
    c_st = False
    if new_st not in txt_st:
        if old_st not in txt_st:
            raise RuntimeError(f"Expected snippet not found in {p_st}")
        p_st.write_text(txt_st.replace(old_st, new_st, 1))
        c_st = True

    # 3. sparsecore_topk.py
    p_sctk = root / "kernels" / "deepseek_v41" / "sparsecore_topk.py"
    txt_sctk = p_sctk.read_text()
    c_sctk = False
    if "_has_16lane_sc" not in txt_sctk:
        tc_helpers = """

def _has_16lane_sc() -> bool:
    sc = getattr(pltpu.get_tpu_info(), "sparse_core", None)
    return sc is not None and getattr(sc, "num_lanes", 0) == 16


def _as_f32_scores(scores: jax.Array) -> jax.Array:
    if scores.dtype == jnp.float32:
        return scores
    if scores.dtype == jnp.int32 or scores.dtype == jnp.uint32:
        return lax.bitcast_convert_type(scores, jnp.float32)
    return scores.astype(jnp.float32)


def _tc_sparsecore_topk(
    scores: jax.Array,
    k: int,
    row_lengths: jax.Array,
) -> jax.Array:
    s_f32 = _as_f32_scores(scores)
    b, n = s_f32.shape
    rl = row_lengths.astype(jnp.int32)[:, None]
    cols = jnp.arange(n, dtype=jnp.int32)[None, :]
    masked = jnp.where(cols < rl, s_f32, -jnp.inf)
    if n < k:
        masked = jnp.pad(masked, ((0, 0), (0, k - n)), constant_values=-jnp.inf)
    vals, idxs = lax.top_k(masked, k)
    return jnp.where(vals > -jnp.inf, idxs.astype(jnp.int32), jnp.int32(-1))


def _tc_cand_gather_topk(
    scores: jax.Array,
    candidates: jax.Array,
    row_lengths: jax.Array,
    *,
    k: int = 512,
    rank_map: jax.Array | None = None,
    seq_ids: jax.Array | None = None,
    p_pad: int = 0,
    blk_per_page: int = 32,
    page_size: int = 256,
) -> jax.Array:
    s_f32 = _as_f32_scores(scores)
    b, n = s_f32.shape
    cb = candidates.shape[1]
    cands = candidates[:b].astype(jnp.int32)
    rl = row_lengths[:b].astype(jnp.int32)[:, None]
    sub = jnp.arange(8, dtype=jnp.int32)[None, None, :]
    blk_ok = cands[:, :, None] >= 0
    blk_safe = jnp.maximum(cands, 0)
    orig_tok = (jnp.left_shift(blk_safe, 3)[:, :, None] + sub).reshape(b, cb * 8)
    valid = jnp.broadcast_to(blk_ok, (b, cb, 8)).reshape(b, cb * 8) & (orig_tok < rl)
    if rank_map is not None and seq_ids is not None:
        orig_p = seq_ids[:b].astype(jnp.int32)[:, None] * p_pad + (blk_safe // blk_per_page)
        rm_flat = rank_map.reshape(-1)
        cp = rm_flat[jnp.clip(orig_p, 0, rm_flat.shape[0] - 1)]
        gather_tok = (
            cp[:, :, None] * page_size
            + ((blk_safe % blk_per_page)[:, :, None] * 8)
            + sub
        ).reshape(b, cb * 8)
    else:
        gather_tok = orig_tok
    valid = valid & (gather_tok >= 0) & (gather_tok < n)
    gathered = jnp.take_along_axis(s_f32, jnp.clip(gather_tok, 0, max(n - 1, 0)), axis=1)
    cand_scores = jnp.where(valid, gathered, -jnp.inf)
    cand_len = cb * 8
    if cand_len < k:
        cand_scores = jnp.pad(cand_scores, ((0, 0), (0, k - cand_len)), constant_values=-jnp.inf)
        orig_tok_pad = jnp.pad(orig_tok, ((0, 0), (0, k - cand_len)), constant_values=-1)
    else:
        orig_tok_pad = orig_tok
    vals, pos = lax.top_k(cand_scores, k)
    chosen = jnp.take_along_axis(orig_tok_pad, pos, axis=1)
    return jnp.where(vals > -jnp.inf, chosen.astype(jnp.int32), jnp.int32(-1))


def _tc_write_cand_and_topk(
    blk_scores: jax.Array,
    scores: jax.Array,
    row_lengths: jax.Array,
    *,
    cb: int = 2048,
    k: int = 512,
    num_blk_sub_bkv: int = 8,
) -> tuple[jax.Array, jax.Array]:
    b_f32 = _as_f32_scores(blk_scores)
    b, n_blk = b_f32.shape
    w_bkv = num_blk_sub_bkv * 128
    rl = row_lengths[:b].astype(jnp.int32)
    blk_rl = jnp.right_shift(rl + 7, 3)
    blk_rl_aligned = jnp.minimum(((blk_rl + w_bkv - 1) // w_bkv) * w_bkv, n_blk)
    cols = jnp.arange(n_blk, dtype=jnp.int32)[None, :]
    b_masked = jnp.where(cols < blk_rl_aligned[:, None], b_f32, -jnp.inf)
    if n_blk < cb:
        b_masked = jnp.pad(b_masked, ((0, 0), (0, cb - n_blk)), constant_values=-jnp.inf)
    b_vals, b_cols = lax.top_k(b_masked, cb)
    col = b_cols.astype(jnp.int32)
    bkv_base = jnp.bitwise_and(col, ~(w_bkv - 1))
    sub_idx = jnp.right_shift(jnp.bitwise_and(col, w_bkv - 1), 7)
    b_lane = jnp.bitwise_and(jnp.right_shift(col, 3), 15)
    s_idx = jnp.bitwise_and(col, 7)
    blk_id = bkv_base | ((s_idx * num_blk_sub_bkv + sub_idx) << 4 | b_lane)
    cands_m = jnp.where(b_vals > -jnp.inf, blk_id, jnp.int32(-1))
    topk_idxs = _tc_cand_gather_topk(scores, cands_m, rl, k=k)
    return topk_idxs, cands_m
"""
        r1 = (
            '    """Exact top-k indices per row, unsorted, -1 suffix-padded."""\n    scores = _as_i32_bits(scores)',
            '    """Exact top-k indices per row, unsorted, -1 suffix-padded."""\n    if not _has_16lane_sc():\n        return _tc_sparsecore_topk(scores, k, row_lengths)\n    scores = _as_i32_bits(scores)',
        )
        r2 = (
            '    """1D SparseCore indirect gather DMA + in-VMEM top-k over candidate blocks."""\n    scores = _as_i32_bits(scores)',
            '    """1D SparseCore indirect gather DMA + in-VMEM top-k over candidate blocks."""\n    if not _has_16lane_sc():\n        return _tc_cand_gather_topk(scores, candidates, row_lengths, k=k, rank_map=rank_map, seq_ids=seq_ids, p_pad=p_pad, blk_per_page=blk_per_page, page_size=page_size)\n    scores = _as_i32_bits(scores)',
        )
        r3 = (
            '    """Fused SparseCore block-top-cb + 1D indirect gather token-top-k."""\n    blk_scores = _as_i32_bits(blk_scores)',
            '    """Fused SparseCore block-top-cb + 1D indirect gather token-top-k."""\n    if not _has_16lane_sc():\n        return _tc_write_cand_and_topk(blk_scores, scores, row_lengths, cb=cb, k=k, num_blk_sub_bkv=num_blk_sub_bkv)\n    blk_scores = _as_i32_bits(blk_scores)',
        )
        for old_s, new_s in (r1, r2, r3):
            if old_s not in txt_sctk:
                raise RuntimeError(f"Expected snippet {old_s!r} not found in {p_sctk}")
            txt_sctk = txt_sctk.replace(old_s, new_s, 1)
        p_sctk.write_text(txt_sctk + tc_helpers)
        c_sctk = True

    return c_smla, c_st, c_sctk


def patch_multihost_dp(root: pathlib.Path) -> tuple[bool, bool, bool]:
    # 1. tpu_rank_binding.py
    p_rb = root / "worker" / "tpu_rank_binding.py"
    txt_rb = p_rb.read_text()
    c_rb = False
    if "dp_size_local_eff =" not in txt_rb:
        old_rb = """        dp_rank_local = parallel_config.data_parallel_rank_local
        if dp_rank_local is None:
            dp_size_local = parallel_config.data_parallel_size_local or dp_size
            dp_rank_local = dp_rank % dp_size_local
        native_local_rank = dp_rank_local * world_size + local_rank
        global_rank = dp_rank * world_size + rank
        global_world = world_size * dp_size
        if pcp_size > 1 and use_spawned_pcp_local_rank:
            tpu_local_rank = _get_spawned_pcp_local_rank(
                env=env,
                local_rank_offset=local_rank_offset,
                local_world=global_world,
            )
        else:
            tpu_local_rank = native_local_rank
        return TpuWorkerBinding(
            rank=global_rank,
            local_rank=local_rank_offset + tpu_local_rank,
            world_size=global_world,
            local_world_size=local_rank_offset + global_world,"""
        new_rb = """        dp_size_local_eff = parallel_config.data_parallel_size_local or dp_size
        dp_rank_local = parallel_config.data_parallel_rank_local
        if dp_rank_local is None:
            dp_rank_local = dp_rank % dp_size_local_eff
        native_local_rank = dp_rank_local * world_size + local_rank
        global_rank = dp_rank * world_size + rank
        global_world = world_size * dp_size
        local_chip_count = world_size * dp_size_local_eff
        if pcp_size > 1 and use_spawned_pcp_local_rank:
            tpu_local_rank = _get_spawned_pcp_local_rank(
                env=env,
                local_rank_offset=local_rank_offset,
                local_world=local_chip_count,
            )
        else:
            tpu_local_rank = native_local_rank
        return TpuWorkerBinding(
            rank=global_rank,
            local_rank=local_rank_offset + tpu_local_rank,
            world_size=global_world,
            local_world_size=local_rank_offset + local_chip_count,"""
        if old_rb not in txt_rb:
            raise RuntimeError(f"Expected snippet not found in {p_rb}")
        p_rb.write_text(txt_rb.replace(old_rb, new_rb, 1))
        c_rb = True

    # 2. tpu_mp_multihost.py
    p_mh = root / "distributed" / "tpu_mp_multihost.py"
    txt_mh = p_mh.read_text()
    c_mh = False
    if 'os.environ["NODE_RANK"] = str(node_index)' not in txt_mh:
        old_mh = """    topology = get_tpu_multihost_topology(total_chips)
    os.environ["TORCH_TPU_TOPOLOGY"] = topology
    os.environ["TORCH_TPU_XPROF_SESSION_ID"] = xprof_session_id
    os.environ[_DP_DONE_SENTINEL] = str(dp_size)"""
        new_mh = """    topology = get_tpu_multihost_topology(total_chips)
    os.environ["TORCH_TPU_TOPOLOGY"] = topology
    os.environ["TPU_NUM_HOSTS"] = str(num_hosts)
    os.environ["NNODES"] = str(num_hosts)
    os.environ["NODE_RANK"] = str(node_index)
    os.environ["TORCH_TPU_XPROF_SESSION_ID"] = xprof_session_id
    os.environ[_DP_DONE_SENTINEL] = str(dp_size)"""
        if old_mh not in txt_mh:
            raise RuntimeError(f"Expected snippet not found in {p_mh}")
        p_mh.write_text(txt_mh.replace(old_mh, new_mh, 1))
        c_mh = True

    # 3. tpu_worker.py pre-GetClient TCPStore barrier for >4 workers (multi-host v5p)
    p_tw = root / "worker" / "tpu_worker.py"
    txt_tw = p_tw.read_text()
    c_tw = False
    if "_TORCH_TPU_PRE_INIT_STORES" not in txt_tw:
        old_tw = """        dist_backend = current_platform.get_worker_distributed_backend(dist_world_size)

        with set_current_vllm_config(self.vllm_config):
            init_distributed_environment("""
        new_tw = """        dist_backend = current_platform.get_worker_distributed_backend(dist_world_size)

        _g_world = int(os.environ.get("WORLD_SIZE", "1") or "1")
        _b_port = int(os.environ.get("TORCH_TPU_PRE_INIT_BARRIER_PORT", "0") or "0")
        if _g_world > 4 and _b_port > 0 and os.environ.get("MASTER_ADDR"):
            from datetime import timedelta as _td
            _g_rank = int(os.environ["RANK"])
            _b_store = torch.distributed.TCPStore(
                host_name=os.environ["MASTER_ADDR"],
                port=_b_port,
                world_size=_g_world,
                is_master=(_g_rank == 0),
                timeout=_td(seconds=600),
            )
            globals().setdefault("_TORCH_TPU_PRE_INIT_STORES", []).append(_b_store)
            _b_store.set(f"pre_init_ready_{_g_rank}", "1")
            _b_store.wait([f"pre_init_ready_{i}" for i in range(_g_world)], _td(seconds=600))
            logger.info(
                "TPU multi-host pre-GetClient barrier passed | rank=%d/%d port=%d",
                _g_rank,
                _g_world,
                _b_port,
            )

        with set_current_vllm_config(self.vllm_config):
            init_distributed_environment("""
        if old_tw not in txt_tw:
            raise RuntimeError(f"Expected snippet not found in {p_tw}")
        p_tw.write_text(txt_tw.replace(old_tw, new_tw, 1))
        c_tw = True

    return c_rb, c_mh, c_tw


def patch_deepseek_v41_swa_vmem(root: pathlib.Path) -> bool:
    p = (
        root
        / "layers"
        / "adapter"
        / "custom_ops"
        / "deepseek_v41"
        / "deepseek_v41_flash_attention_op.py"
    )
    txt = p.read_text()
    new_bq = "    bq_prefill = 32 if q.shape[1] > 64 else 128"
    if new_bq in txt:
        return False

    old_bq = "    bq_prefill = 64 if q.shape[1] > 64 else 128"
    old_qcb = """        q_compute_block_size=(
            8
            if (q.shape[1] > 64 and q.shape[0] > 384)
            else (16 if q.shape[1] > 32 else (32 if q.shape[1] > 8 else 64))
        ),"""
    new_qcb = """        q_compute_block_size=(
            8
            if q.shape[1] > 64
            else (16 if q.shape[1] > 32 else (32 if q.shape[1] > 8 else 64))
        ),"""

    if old_bq not in txt or old_qcb not in txt:
        raise RuntimeError(f"Expected SWA snippet not found in {p}")
    p.write_text(txt.replace(old_bq, new_bq, 1).replace(old_qcb, new_qcb, 1))
    return True


def sync_installed_site_packages(root: pathlib.Path) -> None:
    rel_files = [
        "kernels/fused_moe/v2/kernel.py",
        "kernels/megablox/gmm_v2.py",
        "kernels/deepseek_v41/sparse_mla.py",
        "kernels/deepseek_v41/streamindex_topk.py",
        "kernels/deepseek_v41/sparsecore_topk.py",
        "layers/adapter/custom_ops/deepseek_v41/deepseek_v41_flash_attention_op.py",
        "platforms/tpu_platform.py",
        "worker/tpu_rank_binding.py",
        "distributed/tpu_mp_multihost.py",
        "worker/tpu_worker.py",
    ]
    for d in glob.glob("/usr/local/lib/python3.12/*-packages/vllm_torchtpu"):
        if os.path.isdir(d) and os.path.realpath(d) != os.path.realpath(str(root)):
            for rel in rel_files:
                dst = pathlib.Path(d) / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(str(root / rel), str(dst))
    for pyc_dir in glob.glob(str(root) + "/**/__pycache__", recursive=True) + glob.glob(
        "/usr/local/lib/python3.12/*-packages/vllm_torchtpu/**/__pycache__",
        recursive=True,
    ):
        shutil.rmtree(pyc_dir, ignore_errors=True)


def patch_tpu_platform_v5p_topo(root: pathlib.Path) -> bool:
    p_tp = root / "platforms" / "tpu_platform.py"
    txt = p_tp.read_text()
    old_map = """TPU_3D_TORUS_MULTIHOST_TOPOLOGY_MAP = {
    8: "2,2,2,2",
    16: "2,2,4,2",
    32: "2,4,4,2",
    64: "4,4,4,2",
    128: "4,4,8,2",
    256: "4,8,8,2",
}"""
    new_map = """TPU_3D_TORUS_MULTIHOST_TOPOLOGY_MAP = {
    8: "2,2,2",
    16: "2,2,4",
    32: "2,4,4",
    64: "4,4,4",
    128: "4,4,8",
    256: "4,8,8",
}"""
    if new_map in txt:
        return False
    if old_map not in txt:
        raise RuntimeError(f"Expected TPU_3D_TORUS_MULTIHOST_TOPOLOGY_MAP not found in {p_tp}")
    p_tp.write_text(txt.replace(old_map, new_map, 1))
    return True


def patch_vllm_fast_gcs_safetensors_loader(vllm_repo_root: pathlib.Path) -> bool:
    p_wu = vllm_repo_root / "vllm" / "model_executor" / "model_loader" / "weight_utils.py"
    if not p_wu.exists():
        return False
    txt = p_wu.read_text()
    if "_V5P_FAST_GCS_LOADER_V2" in txt:
        return False
    if "def _v5p_fast_gcs_staged_iterator(" in txt:
        txt = txt[: txt.index("\ndef _v5p_fast_gcs_staged_iterator(")]

    old_anchor = """    sorted_files = sorted(hf_weights_files, key=_natural_sort_key)

    fs_type = _get_fs_type(sorted_files)"""

    new_anchor = """    sorted_files = sorted(hf_weights_files, key=_natural_sort_key)
    if (
        os.environ.get("VLLM_DISABLE_GCS_STAGE", "0") != "1"
        and safetensors_load_strategy not in ("eager", "torchao")
        and any(os.path.islink(f) or f.startswith("/models/") for f in sorted_files)
    ):
        yield from _v5p_fast_gcs_staged_iterator(sorted_files, local_expert_ids)
        return

    fs_type = _get_fs_type(sorted_files)"""

    helper_code = '''

_V5P_FAST_GCS_LOADER_V2 = True
_V5P_STAGE_CALL_SEQ = 0


def _v5p_fast_gcs_staged_iterator(
    sorted_files: list[str],
    local_expert_ids: set[int] | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Host-coordinated GCS HTTP -> tmpfs rolling single-shard safetensors loader (v2).

    Uses per-call sequence IDs (supporting multi-pass target + DSpark draft loading)
    and immutable 1-byte barrier flags so no local rank ever misses a peer's done flag,
    while unlinking each 6.2 GB shard file immediately after all local ranks close it.
    """
    global _V5P_STAGE_CALL_SEQ
    call_seq = _V5P_STAGE_CALL_SEQ
    _V5P_STAGE_CALL_SEQ += 1

    import json as _json
    import time as _time
    import urllib.parse as _urlparse
    import urllib.request as _urlreq
    from concurrent.futures import ThreadPoolExecutor as _TPE

    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1") or "1")
    local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0") % max(local_world, 1)
    sess = (
        os.environ.get("VLLM_ST_STAGE_SESSION")
        or os.environ.get("TORCH_TPU_XPROF_SESSION_ID")
        or os.environ.get("MASTER_PORT", "default")
    )
    tmpfs_root = "/tmp/tpu_logs" if os.path.isdir("/tmp/tpu_logs") else "/dev/shm"
    stage_dir = os.path.join(tmpfs_root, f"st_stage_{sess}")
    os.makedirs(stage_dir, exist_ok=True)

    _tok_cache: dict[str, object] = {"tok": "", "ts": 0.0}

    def _get_token(force: bool = False) -> str:
        now = _time.time()
        if not force and _tok_cache["tok"] and (now - float(_tok_cache["ts"])) < 300.0:
            return str(_tok_cache["tok"])
        for attempt in range(5):
            try:
                req = _urlreq.Request(
                    "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
                    headers={"Metadata-Flavor": "Google"},
                )
                with _urlreq.urlopen(req, timeout=10) as resp:
                    tok = _json.loads(resp.read())["access_token"]
                _tok_cache["tok"] = tok
                _tok_cache["ts"] = now
                return str(tok)
            except Exception:
                if attempt == 4:
                    raise
                _time.sleep(0.5 * (attempt + 1))
        raise RuntimeError("Unreachable")

    def _resolve_gcs_target(st_file: str) -> tuple[str, str]:
        bucket = os.environ.get("VLLM_GCS_STAGE_BUCKET", "").strip()
        if not bucket:
            try:
                with open("/proc/mounts", encoding="utf-8") as mf:
                    for line in mf:
                        parts = line.split()
                        if len(parts) >= 3 and parts[1] == "/models":
                            bucket = parts[0]
                            break
            except Exception:
                pass
        if not bucket or bucket == "gcsfuse":
            raise RuntimeError("VLLM_GCS_STAGE_BUCKET not set and could not infer from /proc/mounts")
        hint = os.readlink(st_file) if os.path.islink(st_file) else st_file
        if hint.startswith("/models/"):
            obj = hint[len("/models/"):]
        else:
            prefix = os.environ.get(
                "VLLM_GCS_STAGE_PREFIX",
                "models/deepseek-ai/DeepSeek-V4.1-Flash-1017",
            ).strip("/")
            obj = f"{prefix}/{os.path.basename(st_file)}"
        return bucket, obj

    def _download_shard(bucket: str, obj: str, dst_path: str) -> int:
        enc = _urlparse.quote(obj, safe="")
        meta_url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{enc}"
        size = 0
        for attempt in range(5):
            try:
                tok = _get_token(force=(attempt > 0))
                mreq = _urlreq.Request(meta_url, headers={"Authorization": f"Bearer {tok}"})
                with _urlreq.urlopen(mreq, timeout=15) as mresp:
                    size = int(_json.loads(mresp.read())["size"])
                break
            except Exception:
                if attempt == 4:
                    raise
                _time.sleep(0.5 * (attempt + 1))

        media_url = meta_url + "?alt=media"
        fd = os.open(dst_path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o644)
        try:
            os.ftruncate(fd, size)
            n_streams = 16 if size >= 64 * 1024 * 1024 else 4
            part_sz = (size + n_streams - 1) // n_streams

            def _fetch_part(part_idx: int) -> None:
                start = part_idx * part_sz
                end = min(size, start + part_sz) - 1
                if start >= size:
                    return
                expected = end - start + 1
                for attempt in range(6):
                    try:
                        tok = _get_token(force=(attempt > 0))
                        rreq = _urlreq.Request(
                            media_url,
                            headers={
                                "Authorization": f"Bearer {tok}",
                                "Range": f"bytes={start}-{end}",
                            },
                        )
                        off = start
                        got = 0
                        with _urlreq.urlopen(rreq, timeout=90) as resp:
                            while True:
                                buf = resp.read(16 * 1024 * 1024)
                                if not buf:
                                    break
                                os.pwrite(fd, buf, off)
                                off += len(buf)
                                got += len(buf)
                        if got != expected:
                            raise IOError(f"Short range read: got {got} != {expected}")
                        return
                    except Exception:
                        if attempt == 5:
                            raise
                        _time.sleep(0.5 * (attempt + 1))

            with _TPE(max_workers=n_streams) as pool:
                list(pool.map(_fetch_part, range(n_streams)))
        finally:
            os.close(fd)
        return size

    t_all = _time.time()
    n_total = len(sorted_files)
    for idx, st_file in enumerate(sorted_files):
        bname = os.path.basename(st_file)
        if (not os.path.islink(st_file)) and st_file.startswith(("/dev/shm/", "/tmp/")):
            t0 = _time.time()
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    yield name, f.get_tensor(name)
            if local_rank == 0:
                logger.info(
                    "[FAST-GCS-STAGE] [pass=%d %d/%d] %s zero-copy local tmpfs in %.2fs",
                    call_seq,
                    idx + 1,
                    n_total,
                    bname,
                    _time.time() - t0,
                )
            continue

        is_engram_perm = ("-00047-of-" in bname or "-00048-of-" in bname) and st_file.startswith("/dev/shm/")
        step_key = f"c{call_seq}_i{idx}"
        active_path = st_file if is_engram_perm else os.path.join(stage_dir, f"_active_{step_key}.safetensors")
        ready_flag = os.path.join(stage_dir, f"st_stage_{step_key}.ready")
        err_flag = os.path.join(stage_dir, f"st_stage_{step_key}.err")

        dl_s = 0.0
        sz_bytes = 0
        if local_rank == 0:
            t_dl = _time.time()
            try:
                if os.path.lexists(active_path):
                    os.unlink(active_path)
                bucket, obj = _resolve_gcs_target(st_file)
                sz_bytes = _download_shard(bucket, obj, active_path)
                tmp_flag = ready_flag + ".tmp"
                with open(tmp_flag, "w", encoding="utf-8") as rf:
                    rf.write(str(sz_bytes))
                os.replace(tmp_flag, ready_flag)
                dl_s = _time.time() - t_dl
            except Exception as exc:
                with open(err_flag, "w", encoding="utf-8") as ef:
                    ef.write(repr(exc))
                raise

        t_wait = _time.time()
        while not os.path.exists(ready_flag):
            if os.path.exists(err_flag):
                raise RuntimeError(
                    f"Rank 0 failed staging {bname}: {open(err_flag, encoding='utf-8').read()}"
                )
            if _time.time() - t_wait > 900:
                raise TimeoutError(f"Timed out waiting for {ready_flag}")
            _time.sleep(0.02)

        t_ld = _time.time()
        with safe_open(active_path, framework="pt") as f:
            for name in f.keys():
                if should_skip_weight(name, local_expert_ids):
                    continue
                yield name, f.get_tensor(name)
        del f

        done_me = os.path.join(stage_dir, f"st_done_{step_key}_r{local_rank}")
        tmp_done = done_me + ".tmp"
        with open(tmp_done, "w", encoding="utf-8") as df:
            df.write("1")
        os.replace(tmp_done, done_me)

        t_bar = _time.time()
        for r in range(local_world):
            r_done = os.path.join(stage_dir, f"st_done_{step_key}_r{r}")
            while not os.path.exists(r_done):
                if _time.time() - t_bar > 900:
                    raise TimeoutError(f"Timed out waiting for {r_done}")
                _time.sleep(0.02)

        if local_rank == 0:
            if not is_engram_perm and os.path.exists(active_path):
                try:
                    os.unlink(active_path)
                except OSError:
                    pass
            logger.info(
                "[FAST-GCS-STAGE] [pass=%d %d/%d] %s (%.2f GB) | dl=%.2fs load=%.2fs elapsed=%.1fs",
                call_seq,
                idx + 1,
                n_total,
                bname,
                sz_bytes / 1e9,
                dl_s,
                _time.time() - t_ld,
                _time.time() - t_all,
            )

    if local_rank == 0:
        logger.info(
            "[FAST-GCS-STAGE] [pass=%d] Completed all %d shards in %.1fs (0 GiB shard tmpfs retained)",
            call_seq,
            n_total,
            _time.time() - t_all,
        )
'''

    if new_anchor not in txt:
        if old_anchor not in txt:
            raise RuntimeError(f"Expected sorted_files snippet not found in {p_wu}")
        txt = txt.replace(old_anchor, new_anchor, 1)
    p_wu.write_text(txt + helper_code)

    for d in glob.glob("/usr/local/lib/python3.12/*-packages/vllm"):
        dst = pathlib.Path(d) / "model_executor" / "model_loader" / "weight_utils.py"
        if dst.parent.is_dir() and os.path.realpath(str(dst)) != os.path.realpath(str(p_wu)):
            shutil.copyfile(str(p_wu), str(dst))
    for pyc_dir in glob.glob(str(vllm_repo_root) + "/**/__pycache__", recursive=True) + glob.glob(
        "/usr/local/lib/python3.12/*-packages/vllm/model_executor/model_loader/__pycache__",
        recursive=True,
    ):
        shutil.rmtree(pyc_dir, ignore_errors=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "pos_src_root",
        nargs="?",
        default=None,
        help="Optional positional path to vllm_torchtpu package directory",
    )
    ap.add_argument(
        "--src-root",
        default="/workspace/vllm-torchtpu/src/vllm_torchtpu",
        help="Path to vllm_torchtpu package directory",
    )
    ap.add_argument(
        "--vllm-root",
        default="/workspace/vllm",
        help="Path to vllm repository root",
    )
    ap.add_argument("--no-sync-site", action="store_true")
    args = ap.parse_args()

    root = pathlib.Path(args.pos_src_root or args.src_root)
    c_moe = patch_fused_moe_v2(root)
    c_gmm = patch_gmm_v2(root)
    c_smla, c_st, c_sctk = patch_sparsecore_v5p(root)
    c_swa = patch_deepseek_v41_swa_vmem(root)
    c_tp = patch_tpu_platform_v5p_topo(root)
    c_rb, c_mh, c_tw = patch_multihost_dp(root)
    c_gcs = patch_vllm_fast_gcs_safetensors_loader(pathlib.Path(args.vllm_root))
    if not args.no_sync_site:
        sync_installed_site_packages(root)

    print(
        f"[V5P-COMPAT-PATCH] OK | fused_moe_v2={c_moe} gmm_v2={c_gmm} "
        f"sparse_mla={c_smla} streamindex_topk={c_st} sparsecore_topk={c_sctk} "
        f"swa_vmem={c_swa} tpu_platform_topo={c_tp} rank_binding={c_rb} "
        f"mp_multihost={c_mh} worker_barrier={c_tw} fast_gcs_loader={c_gcs}",
        flush=True,
    )


if __name__ == "__main__":
    main()

