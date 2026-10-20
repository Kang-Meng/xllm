# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSeek-V4.1 projected-pooling compressor Triton kernels (K2+K3).

The production entry is ``compressor_from_projected``: the host pipeline
(``Csa2AttentionBackend``) performs the K1 projection GEMM itself and hands
over pre-projected FP32 ``kv`` / ``scores`` rows, so the kernels here cover
only the two stages that follow the projection:

  K2 group pooling (variable length / residual stitching, two-pass softmax; the
     segment-leading residual is read from the state ring)
  K3 state-ring update (writes only the last ``cache_size`` tokens of each
     segment, one writer per slot)
K2 must run before K3: a segment's tail write and a segment's head residual read
hit the same slot, and a concurrent read would observe this segment's write.

xLLM adaptation (ported from vllm-ascend dsv41): the ring state is kept as the
two contiguous planes ``compress_kv_state`` / ``compress_score_state`` (each
``[blocks, ROWS, HEAD_DIM]`` fp32) used by ``Csa2AttentionBackend``, instead of
vllm-ascend's single packed ``[blocks, 32, 2*HEAD_DIM]`` tensor. ``ROWS`` is
fixed (32) to match ``kDsv41CompressStateRows``; the host wrappers take two
state pointers and the kernels read/write ``kv_state_ptr`` and
``score_state_ptr`` separately. All other math is unchanged.
"""

import torch
import triton
import triton.language as tl

# ============================================================================
# K2: group pooling (task_id -> (batch, local_group) O(1) division decode;
#     control values are loaded inside the kernel)
# ============================================================================


@triton.jit
def _pooled_blocked(
    group_idx,
    start_pos,
    seg_row_base,
    cache_row,
    kv_ptr,
    score_ptr,
    kv_cache_ptr,
    score_cache_ptr,
    offs_h,
    CACHE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RATIO: tl.constexpr,
    RATIO_PAD: tl.constexpr,
    CHUNK_ROWS: tl.constexpr,
    HAS_RES,
):
    """Three chunked passes over one group (max / denom / weighted sum).
    With HAS_RES=0 the cache is never accessed.

    A group spans global tokens [group_idx*RATIO, (group_idx+1)*RATIO):
    seg_off >= 0 reads the big projection matrix at (seg_row_base+seg_off);
    seg_off < 0 is the segment-leading residual, read from the cache
    (cache_row, token_pos % CACHE_SIZE)."""
    # Keep masked history lanes inside the projection allocation. Ascend may
    # form a DMA address for a masked lane before applying the load mask.
    # The original masks and ring selection still supply all history values.
    score_max = tl.full((HEAD_DIM,), -float("inf"), dtype=tl.float32)
    for chunk0 in range(0, RATIO_PAD, CHUNK_ROWS):
        rows = chunk0 + tl.arange(0, CHUNK_ROWS)
        row_valid = rows < RATIO
        token_pos = group_idx * RATIO + rows
        seg_off = token_pos - start_pos
        in_seg = seg_off >= 0
        score_seg = tl.load(
            score_ptr + (seg_row_base + tl.maximum(seg_off, 0))[:, None] * HEAD_DIM + offs_h[None, :],
            mask=(row_valid & in_seg)[:, None],
            other=0.0,
        )
        if HAS_RES:
            slot = token_pos % CACHE_SIZE
            score_cache = tl.load(
                score_cache_ptr + (cache_row * CACHE_SIZE + slot[:, None]) * HEAD_DIM + offs_h[None, :],
                mask=(row_valid & (seg_off < 0))[:, None],
                other=0.0,
            )
            score = tl.where(in_seg[:, None], score_seg, score_cache)
        else:
            score = score_seg
        score = tl.where(row_valid[:, None], score, -float("inf"))
        score_max = tl.maximum(score_max, tl.max(score, axis=0))
    exp_sum = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for chunk0 in range(0, RATIO_PAD, CHUNK_ROWS):
        rows = chunk0 + tl.arange(0, CHUNK_ROWS)
        row_valid = rows < RATIO
        token_pos = group_idx * RATIO + rows
        seg_off = token_pos - start_pos
        in_seg = seg_off >= 0
        score_seg = tl.load(
            score_ptr + (seg_row_base + tl.maximum(seg_off, 0))[:, None] * HEAD_DIM + offs_h[None, :],
            mask=(row_valid & in_seg)[:, None],
            other=0.0,
        )
        if HAS_RES:
            slot = token_pos % CACHE_SIZE
            score_cache = tl.load(
                score_cache_ptr + (cache_row * CACHE_SIZE + slot[:, None]) * HEAD_DIM + offs_h[None, :],
                mask=(row_valid & (seg_off < 0))[:, None],
                other=0.0,
            )
            score = tl.where(in_seg[:, None], score_seg, score_cache)
        else:
            score = score_seg
        score = tl.where(row_valid[:, None], score, -float("inf"))
        exp_sum += tl.sum(tl.exp(score - score_max[None, :]), axis=0)
    pooled = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for chunk0 in range(0, RATIO_PAD, CHUNK_ROWS):
        rows = chunk0 + tl.arange(0, CHUNK_ROWS)
        row_valid = rows < RATIO
        token_pos = group_idx * RATIO + rows
        seg_off = token_pos - start_pos
        in_seg = seg_off >= 0
        score_seg = tl.load(
            score_ptr + (seg_row_base + tl.maximum(seg_off, 0))[:, None] * HEAD_DIM + offs_h[None, :],
            mask=(row_valid & in_seg)[:, None],
            other=0.0,
        )
        if HAS_RES:
            slot = token_pos % CACHE_SIZE
            score_cache = tl.load(
                score_cache_ptr + (cache_row * CACHE_SIZE + slot[:, None]) * HEAD_DIM + offs_h[None, :],
                mask=(row_valid & (seg_off < 0))[:, None],
                other=0.0,
            )
            score = tl.where(in_seg[:, None], score_seg, score_cache)
        else:
            score = score_seg
        score = tl.where(row_valid[:, None], score, -float("inf"))
        prob = tl.exp(score - score_max[None, :]) / exp_sum[None, :]
        kv_seg = tl.load(
            kv_ptr + (seg_row_base + tl.maximum(seg_off, 0))[:, None] * HEAD_DIM + offs_h[None, :],
            mask=(row_valid & in_seg)[:, None],
            other=0.0,
        )
        if HAS_RES:
            slot = token_pos % CACHE_SIZE
            kv_cache = tl.load(
                kv_cache_ptr + (cache_row * CACHE_SIZE + slot[:, None]) * HEAD_DIM + offs_h[None, :],
                mask=(row_valid & (seg_off < 0))[:, None],
                other=0.0,
            )
            kv_vals = tl.where(in_seg[:, None], kv_seg, kv_cache)
        else:
            kv_vals = kv_seg
        pooled += tl.sum(prob * kv_vals, axis=0)
    return pooled


@triton.jit
def _pool_kernel(
    out_ptr,
    kv_ptr,
    score_ptr,
    kv_cache_ptr,
    score_cache_ptr,
    norm_w_ptr,
    meta_ptr,  # packed [start_pos | used_len | out_row_offset | seg_row_base]
    block_table_ptr,
    total_group_slots,
    GROUP_SLOTS_PER_BATCH,  # group slots per batch <= ceil(max_used_len/RATIO)
    NUM_BATCH,
    CACHE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RATIO: tl.constexpr,
    RATIO_PAD: tl.constexpr,
    CHUNK_ROWS: tl.constexpr,
    eps: tl.constexpr,
    SINGLE_BLOCK: tl.constexpr,
    NO_PAD: tl.constexpr,
    TOKEN_ALIGNED: tl.constexpr = False,
    NUM_CACHE_BLOCKS: tl.constexpr = 0,
):
    pid = tl.program_id(0)
    offs_h = tl.arange(0, HEAD_DIM)
    if not TOKEN_ALIGNED:
        norm_w = tl.load(norm_w_ptr + offs_h).to(tl.float32)
    for task_id in range(pid, total_group_slots, tl.num_programs(0)):
        batch = task_id // GROUP_SLOTS_PER_BATCH
        local_group = task_id - batch * GROUP_SLOTS_PER_BATCH
        start_pos = tl.load(meta_ptr + batch)  # [0, NUM_BATCH)
        used_len = tl.load(meta_ptr + NUM_BATCH + batch)  # [NUM_BATCH, 2*NUM_BATCH)
        groups_in_batch = (start_pos + used_len) // RATIO - start_pos // RATIO
        cache_row = tl.load(block_table_ptr + batch)
        valid = local_group < groups_in_batch
        if TOKEN_ALIGNED:
            valid = valid & (cache_row > 0) & (cache_row < NUM_CACHE_BLOCKS)
        if valid:
            group_idx = start_pos // RATIO + local_group
            seg_row_base = tl.load(meta_ptr + 3 * NUM_BATCH + batch)  # [3*NUM_BATCH, 4*NUM_BATCH)
            rows = tl.arange(0, CHUNK_ROWS)
            token_pos = group_idx * RATIO + rows
            seg_off = token_pos - start_pos
            # Segment-leading residual token count; only local_group == 0
            # can have it > 0.
            residual = start_pos - group_idx * RATIO
            cache_row = tl.load(block_table_ptr + batch)
            if residual > 0:
                # Residual group (at most one per batch): stitch with cache.
                pooled = _pooled_blocked(
                    group_idx,
                    start_pos,
                    seg_row_base,
                    cache_row,
                    kv_ptr,
                    score_ptr,
                    kv_cache_ptr,
                    score_cache_ptr,
                    offs_h,
                    CACHE_SIZE,
                    HEAD_DIM,
                    RATIO,
                    RATIO_PAD,
                    CHUNK_ROWS,
                    True,
                )
            else:
                # Non-residual group: single-block direct read; the chunked
                # variant is used when RATIO exceeds the single-block capacity.
                if SINGLE_BLOCK:
                    if NO_PAD:
                        score = tl.load(score_ptr + (seg_row_base + seg_off)[:, None] * HEAD_DIM + offs_h[None, :])
                        score_max = tl.max(score, axis=0)
                        e = tl.exp(score - score_max[None, :])
                        exp_sum = tl.sum(e, axis=0)
                        prob = e / exp_sum[None, :]
                        kv_vals = tl.load(kv_ptr + (seg_row_base + seg_off)[:, None] * HEAD_DIM + offs_h[None, :])
                    else:
                        row_valid = rows < RATIO
                        score = tl.load(
                            score_ptr + (seg_row_base + seg_off)[:, None] * HEAD_DIM + offs_h[None, :],
                            mask=row_valid[:, None],
                            other=-float("inf"),
                        )
                        score_max = tl.max(score, axis=0)
                        e = tl.exp(score - score_max[None, :])
                        e = tl.where(row_valid[:, None], e, 0.0)
                        exp_sum = tl.sum(e, axis=0)
                        prob = e / exp_sum[None, :]
                        kv_vals = tl.load(
                            kv_ptr + (seg_row_base + seg_off)[:, None] * HEAD_DIM + offs_h[None, :],
                            mask=row_valid[:, None],
                            other=0.0,
                        )
                    pooled = tl.sum(prob * kv_vals, axis=0)
                else:
                    pooled = _pooled_blocked(
                        group_idx,
                        start_pos,
                        seg_row_base,
                        cache_row,
                        kv_ptr,
                        score_ptr,
                        kv_cache_ptr,
                        score_cache_ptr,
                        offs_h,
                        CACHE_SIZE,
                        HEAD_DIM,
                        RATIO,
                        RATIO_PAD,
                        CHUNK_ROWS,
                        False,
                    )
            if TOKEN_ALIGNED:
                # Keep Aurora's existing RMSNorm outside this kernel. Place
                # each completed pair at its original completion-token row.
                out_row = seg_row_base + (group_idx + 1) * RATIO - 1 - start_pos
                tl.store(out_ptr + out_row * HEAD_DIM + offs_h, pooled.to(tl.bfloat16))
            else:
                pooled_bf16 = pooled.to(tl.bfloat16).to(tl.float32)
                var = tl.sum(pooled_bf16 * pooled_bf16, axis=0) / HEAD_DIM
                norm_out = pooled_bf16 * (1.0 / tl.sqrt(var + eps)) * norm_w
                out_row = tl.load(meta_ptr + 2 * NUM_BATCH + batch) + local_group
                tl.store(out_ptr + out_row * HEAD_DIM + offs_h, norm_out.to(tl.bfloat16))


# ============================================================================
# K3: cache update (writes only each segment's last
# min(used_len, CACHE_SIZE) tokens)
# ============================================================================


@triton.jit
def _cache_update_kernel(
    kv_ptr,
    score_ptr,
    kv_cache_ptr,
    score_cache_ptr,
    meta_ptr,
    block_table_ptr,
    total_write_slots,
    WRITE_SLOTS_PER_BATCH,  # per-batch write slot bound,
                            # min(max_used_len, CACHE_SIZE)
    NUM_BATCH,
    CACHE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PROTECT_NULL: tl.constexpr = False,
    NUM_CACHE_BLOCKS: tl.constexpr = 0,
):
    pid = tl.program_id(0)
    offs_h = tl.arange(0, HEAD_DIM)
    for task_id in range(pid, total_write_slots, tl.num_programs(0)):
        batch = task_id // WRITE_SLOTS_PER_BATCH
        write_idx = task_id - batch * WRITE_SLOTS_PER_BATCH
        used_len = tl.load(meta_ptr + NUM_BATCH + batch)
        tail_count = used_len if used_len < CACHE_SIZE else CACHE_SIZE
        cache_row = tl.load(block_table_ptr + batch)
        valid = write_idx < tail_count
        if PROTECT_NULL:
            valid = valid & (cache_row > 0) & (cache_row < NUM_CACHE_BLOCKS)
        if valid:
            # The write_idx-th among the last tail_count tokens.
            token_in_seg = used_len - tail_count + write_idx
            proj_row = tl.load(meta_ptr + 3 * NUM_BATCH + batch) + token_in_seg
            token_pos = tl.load(meta_ptr + batch) + token_in_seg
            slot = token_pos % CACHE_SIZE
            cache_row = tl.load(block_table_ptr + batch)
            kv_vals = tl.load(kv_ptr + proj_row * HEAD_DIM + offs_h)
            score_vals = tl.load(score_ptr + proj_row * HEAD_DIM + offs_h)
            tl.store(kv_cache_ptr + (cache_row * CACHE_SIZE + slot) * HEAD_DIM + offs_h, kv_vals)
            tl.store(score_cache_ptr + (cache_row * CACHE_SIZE + slot) * HEAD_DIM + offs_h, score_vals)


# ============================================================================
# Host: layout normalization, control packing, launch
# ============================================================================


_CUBE_CORE_NUM = None  # AI core count, cached after the first query


def _cube_core_num():
    """Number of device AI cores (the grid parallel unit).

    multi_processor_count is the vector core count (2x num_aicore) and
    must not be used as the upper bound."""
    global _CUBE_CORE_NUM
    if _CUBE_CORE_NUM is None:
        driver = triton.runtime.driver

        _CUBE_CORE_NUM = driver.active.utils.get_device_properties(torch.npu.current_device())["num_aicore"]
    return _CUBE_CORE_NUM


def compressor_from_projected(kv, scores, kv_state, score_state, metadata, out, *, max_query_len, num_cores):
    """Pool C2 into token-aligned BF16 rows, then update a private FP32 ring.

    metadata is contiguous device INT32 [5, requests]: start positions, used
    lengths, output bases (reserved), input bases, and actual global block IDs.
    Controls must be bounded by the input rows and request table; padded requests
    have used length and block ID zero. No tensor contents are read by the host.

    xLLM adaptation: the ring state is the two contiguous fp32 planes
    ``kv_state`` / ``score_state`` (each [blocks, 32, width]) that
    ``Csa2AttentionBackend`` owns, not vllm-ascend's single packed
    [blocks, 32, 2*width] tensor.
    """
    if kv.dtype != torch.float32 or scores.dtype != torch.float32:
        raise ValueError("Aurora projections must be FP32")
    if kv_state.dtype != torch.float32 or score_state.dtype != torch.float32:
        raise ValueError("Aurora ring state must be FP32")
    if kv.ndim != 2 or scores.shape != kv.shape:
        raise ValueError("Expected matching [tokens, width] projections")
    tokens, width = kv.shape
    if width < 1 or width & (width - 1):
        raise ValueError("Compressor width must be a positive power of two")
    if kv_state.ndim != 3 or kv_state.shape[1:] != (32, width):
        raise ValueError("Expected [blocks, 32, width] kv ring state")
    if score_state.ndim != 3 or score_state.shape[1:] != (32, width):
        raise ValueError("Expected [blocks, 32, width] score ring state")
    if kv_state.shape[0] != score_state.shape[0]:
        raise ValueError("Compressor kv/score ring state must have the same block count")
    if out.dtype != torch.bfloat16 or out.shape != kv.shape:
        raise ValueError("Expected token-aligned BF16 pooled output")
    if metadata.dtype != torch.int32 or metadata.ndim != 2 or metadata.shape[0] != 5:
        raise ValueError("Expected INT32 [5, requests] device metadata")
    tensors = (kv, scores, kv_state, score_state, metadata, out)
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("Compressor requires contiguous projections, ring pages and controls")
    if not all(t.device == kv.device for t in tensors):
        raise ValueError("Compressor tensors must share one device")
    if num_cores <= 0 or not 0 <= max_query_len <= tokens:
        raise ValueError("Invalid compressor launch bounds")
    out.zero_()
    batches = metadata.shape[1]
    if tokens == 0 or batches == 0 or max_query_len == 0:
        return out
    if kv.device.type != "npu":
        raise ValueError("Projected Triton compressor requires an NPU")
    # Compression-ratio clamp point: the supported set is fixed HERE, at
    # this launch -- the kernel is the ratio-2-only path (RATIO=RATIO_PAD=2,
    # groups=(max_query_len+1)//2). A configured ratio outside this set never
    # reaches the launch: the caller-side gate (csa2_attention.
    # _v41_compressor_triton_reason, ``ratio != 2``) clamps it by leaving the
    # layer on the pure-torch fallback path. No runtime check is added here.
    groups = (max_query_len + 1) // 2
    writes = min(max_query_len, 32)
    # Same-stream launch order is required: residual reads precede ring writes.
    _pool_kernel[(min(num_cores, batches * groups),)](
        out,
        kv,
        scores,
        kv_state,
        score_state,
        out,
        metadata,
        metadata[4],
        batches * groups,
        groups,
        batches,
        CACHE_SIZE=32,
        HEAD_DIM=width,
        RATIO=2,
        RATIO_PAD=2,
        CHUNK_ROWS=2,
        eps=0.0,
        SINGLE_BLOCK=True,
        NO_PAD=True,
        TOKEN_ALIGNED=True,
        NUM_CACHE_BLOCKS=kv_state.shape[0],
    )
    _cache_update_kernel[(min(num_cores, batches * writes),)](
        kv,
        scores,
        kv_state,
        score_state,
        metadata,
        metadata[4],
        batches * writes,
        writes,
        batches,
        CACHE_SIZE=32,
        HEAD_DIM=width,
        PROTECT_NULL=True,
        NUM_CACHE_BLOCKS=kv_state.shape[0],
    )
    return out
