# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.language as tl
import triton.language.extra.mlu as mlu


@triton.jit(do_not_specialize=["softmax_scale", "block_table_width"])
def score_pools(
    query: tl.tensor,
    head_weights: tl.tensor,
    compressed_pool_cache: tl.tensor,
    block_table: tl.tensor,
    positions: tl.tensor,
    rows: tl.tensor,
    pool_scores: tl.tensor,
    query_row_stride: tl.int64,
    query_head_stride: tl.int64,
    query_dim_stride: tl.int64,
    weight_row_stride: tl.int64,
    weight_head_stride: tl.int64,
    cache_block_stride: tl.int64,
    cache_head_stride: tl.int64,
    cache_pool_stride: tl.int64,
    cache_dim_stride: tl.int64,
    block_table_row_stride: tl.int64,
    block_table_column_stride: tl.int64,
    score_row_stride: tl.int64,
    score_column_stride: tl.int64,
    softmax_scale: tl.float32,
    capacity: tl.int64,
    cache_blocks: tl.int64,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    POOL_BLOCK_SIZE: tl.constexpr,
    block_table_width: tl.int64,
    INDEX_KPOOL: tl.constexpr,
    BLOCK_POOLS: tl.constexpr,
    NUM_HEAD_PAD: tl.constexpr,
    DENSE_BLOCK: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    CHUNK: tl.constexpr,
) -> None:
    """Score pool tiles directly from the paged compressed cache.

    Each program owns TILES_PER_PROG consecutive pool tiles of one query.
    With DENSE_BLOCK set (cache layout [blocks, 1, POOL_BLOCK_SIZE, HEAD_DIM]
    whose last two dims are contiguous and POOL_BLOCK_SIZE * HEAD_DIM is a
    power of two), one pool block is loaded as a single contiguous row and
    CHUNK tiles are fused into one tensor-engine dot per iteration, so the
    paged-cache read lowers to a row gather. The fallback path keeps the
    original per-tile, fully strided addressing.
    """
    query_id = tl.program_id(0)
    pool_capacity = capacity
    request = tl.load(rows + query_id).to(tl.int64)

    kv_seq_len = tl.load(positions + query_id).to(tl.int64) + 1
    live_pool_count = tl.minimum(
        tl.maximum(kv_seq_len, 0) // INDEX_KPOOL,
        tl.minimum(pool_capacity, block_table_width * POOL_BLOCK_SIZE),
    )

    dim_offsets = tl.arange(0, HEAD_DIM)
    head_offsets = tl.arange(0, NUM_HEAD_PAD)
    head_mask = head_offsets < NUM_HEADS
    query_offsets = (
        query_id * query_row_stride
        + head_offsets[:, None] * query_head_stride
        + dim_offsets[None, :] * query_dim_stride
    )
    query_values = tl.load(
        query + query_offsets,
        mask=head_mask[:, None],
        other=0.0,
    )
    weights = tl.load(
        head_weights + query_id * weight_row_stride + head_offsets * weight_head_stride,
        mask=head_mask,
        other=0.0,
    ).to(tl.float32)
    query_t = tl.trans(query_values)

    if DENSE_BLOCK:
        blocks_per_tile: tl.constexpr = BLOCK_POOLS // POOL_BLOCK_SIZE
        pools_per_iter: tl.constexpr = BLOCK_POOLS * CHUNK
        blocks_per_iter: tl.constexpr = blocks_per_tile * CHUNK
        flat_offsets = tl.arange(0, POOL_BLOCK_SIZE * HEAD_DIM)
        for it in range(0, TILES_PER_PROG // CHUNK):
            first_tile = tl.program_id(1) * TILES_PER_PROG + it * CHUNK
            pool_ids = first_tile * BLOCK_POOLS + tl.arange(0, pools_per_iter)
            pool_in_capacity = pool_ids < pool_capacity
            pool_is_live = pool_in_capacity & (pool_ids < live_pool_count)
            scores = tl.full((pools_per_iter,), float("-inf"), dtype=tl.float32)
            if first_tile * BLOCK_POOLS < live_pool_count:
                block_ids = first_tile * blocks_per_tile + tl.arange(0, blocks_per_iter)
                block_live = block_ids < ((live_pool_count + POOL_BLOCK_SIZE - 1) // POOL_BLOCK_SIZE)
                block_live = block_live & (block_ids < block_table_width)
                physical_blocks = tl.load(
                    block_table + request * block_table_row_stride + block_ids * block_table_column_stride,
                    mask=block_live,
                    other=-1,
                ).to(tl.int64)
                block_live &= (physical_blocks >= 0) & (physical_blocks < cache_blocks)
                pool_is_live &= tl.broadcast_to(block_live[:, None], (blocks_per_iter, POOL_BLOCK_SIZE)).reshape(
                    pools_per_iter
                )
                cache_offsets = physical_blocks[:, None] * cache_block_stride + flat_offsets[None, :]
                keys_flat = tl.load(
                    compressed_pool_cache + cache_offsets,
                    mask=block_live[:, None],
                    other=0.0,
                )
                pooled_keys = tl.reshape(
                    keys_flat,
                    (pools_per_iter, HEAD_DIM),
                )
                per_pool = tl.dot(
                    pooled_keys,
                    query_t,
                    allow_tf32=False,
                ).to(tl.float32)
                per_pool = tl.maximum(per_pool * softmax_scale, 0.0)
                live_scores = tl.sum(per_pool * weights[None, :], axis=1)
                scores = tl.where(pool_is_live, live_scores, float("-inf"))

            tl.store(
                pool_scores + query_id * score_row_stride + pool_ids * score_column_stride,
                scores,
                mask=pool_in_capacity,
            )
    else:
        for tile_id in range(0, TILES_PER_PROG):
            tile = tl.program_id(1) * TILES_PER_PROG + tile_id
            pool_ids = tile * BLOCK_POOLS + tl.arange(0, BLOCK_POOLS)
            pool_in_capacity = pool_ids < pool_capacity
            pool_is_live = pool_in_capacity & (pool_ids < live_pool_count)
            scores = tl.full((BLOCK_POOLS,), float("-inf"), dtype=tl.float32)
            if tile * BLOCK_POOLS < live_pool_count:
                logical_blocks = pool_ids // POOL_BLOCK_SIZE
                physical_blocks = tl.load(
                    block_table + request * block_table_row_stride + logical_blocks * block_table_column_stride,
                    mask=pool_is_live,
                    other=-1,
                ).to(tl.int64)
                pool_is_live &= (physical_blocks >= 0) & (physical_blocks < cache_blocks)
                pool_offsets = pool_ids % POOL_BLOCK_SIZE
                cache_offsets = (
                    physical_blocks[:, None] * cache_block_stride
                    + 0 * cache_head_stride
                    + pool_offsets[:, None] * cache_pool_stride
                    + dim_offsets[None, :] * cache_dim_stride
                )
                pooled_keys = tl.load(
                    compressed_pool_cache + cache_offsets,
                    mask=pool_is_live[:, None],
                    other=0.0,
                )
                per_pool = tl.dot(
                    pooled_keys,
                    query_t,
                    allow_tf32=False,
                ).to(tl.float32)
                per_pool = tl.maximum(per_pool * softmax_scale, 0.0)
                live_scores = tl.sum(per_pool * weights[None, :], axis=1)
                scores = tl.where(pool_is_live, live_scores, float("-inf"))

            tl.store(
                pool_scores + query_id * score_row_stride + pool_ids * score_column_stride,
                scores,
                mask=pool_in_capacity,
            )


@triton.jit
def score_prefill(
    q: tl.tensor,
    weights: tl.tensor,
    keys: tl.tensor,
    table: tl.tensor,
    positions: tl.tensor,
    rows: tl.tensor,
    out: tl.tensor,
    q_stride: tl.int64,
    q_head_stride: tl.int64,
    w_stride: tl.int64,
    table_stride: tl.int64,
    out_stride: tl.int64,
    N: tl.int64,
    scale: tl.float32,
    H: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    POOL_BLOCK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGED: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    h = tl.arange(0, BLOCK_H)
    d = tl.arange(0, D)
    completed = (tl.load(positions + row) + 1) // P
    valid = (n < N) & (n < completed)
    result = tl.full((BLOCK_N,), -float("inf"), tl.float32)
    if tl.program_id(1) * BLOCK_N < completed:
        if PAGED:
            batch = tl.load(rows + row)
            if BLOCK_N % POOL_BLOCK == 0:
                page = tl.arange(0, BLOCK_N // POOL_BLOCK)
                page_offset = tl.arange(0, POOL_BLOCK)
                page_start = tl.program_id(1) * BLOCK_N + page * POOL_BLOCK
                page_base = tl.program_id(1) * (BLOCK_N // POOL_BLOCK)
                block = tl.load(
                    table + batch * table_stride + page_base + page,
                    page_start < N,
                    0,
                )
                slots = (block[:, None] * POOL_BLOCK + page_offset[None, :]).reshape(BLOCK_N)
            else:
                block = tl.load(table + batch * table_stride + n // POOL_BLOCK, valid, 0)
                slots = block * POOL_BLOCK + n % POOL_BLOCK
        else:
            slots = n
        k = tl.load(keys + slots[None, :] * D + d[:, None], valid[None, :], 0)
        result = tl.full((BLOCK_N,), 0, tl.float32)
        for base in range(tl.cdiv(H, BLOCK_H)):
            heads = base * BLOCK_H + h
            query = tl.load(
                q + row * q_stride + heads[:, None] * q_head_stride + d[None, :],
                heads[:, None] < H,
                0,
            )
            w = tl.load(weights + row * w_stride + heads, heads < H, 0)
            dots = tl.dot(query, k).to(tl.float32)
            result += tl.sum(tl.maximum(dots * scale, 0) * w[:, None], 0)
    tl.store(out + row * out_stride + n, tl.where(valid, result, -float("inf")), n < N)


@triton.jit
def gather_prefill_cache(
    cache: tl.tensor,
    table: tl.tensor,
    keys: tl.tensor,
    request: tl.int64,
    N: tl.int64,
    table_stride: tl.int64,
    D: tl.constexpr,
    POOL_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.arange(0, D)
    block = tl.load(
        table + request * table_stride + n // POOL_BLOCK,
        n // POOL_BLOCK < tl.cdiv(N, POOL_BLOCK),
        0,
    )
    # Padding pages can appear in synthetic or partially populated tables.
    # Redirect them to a valid page without adding a mask to the cache load.
    slot = tl.maximum(block, 0) * POOL_BLOCK + n % POOL_BLOCK
    slot = tl.max_contiguous(slot, POOL_BLOCK)
    value = tl.load(cache + slot[:, None] * D + d[None, :], n[:, None] < N, 0)
    tl.store(keys + n[:, None] * D + d[None, :], value, n[:, None] < N)


@triton.jit
def _wide_prefix(mask: tl.tensor, BN: tl.constexpr) -> tl.tensor:
    # Boolean dot products give exact local counts and avoid a serial scan.
    if BN >= 256:
        C: tl.constexpr = 128 if BN >= 2048 else (32 if BN >= 512 else 16)
        r = tl.arange(0, C)
        triangular = (r[:, None] <= r[None, :]).to(tl.bfloat16)
        blocks = mask.to(tl.float32).reshape(BN // C, C)
        local = tl.dot(blocks.to(tl.bfloat16), triangular)
        counts = tl.sum(blocks, 1)
        g = tl.arange(0, BN // C)
        before = tl.sum(tl.where(g[:, None] < g[None, :], counts[:, None], 0.0), 0)
        return (local + before[:, None]).reshape(BN)
    else:
        r = tl.arange(0, BN)
        return tl.sum(
            tl.where(r[:, None] <= r[None, :], mask[:, None].to(tl.float32), 0.0),
            0,
        )


@triton.jit
def select_topk_streaming(
    scores: tl.tensor,
    output: tl.tensor,
    N: tl.int64,
    in_stride: tl.int64,
    K: tl.constexpr,
    BN: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BN)
    bins = tl.arange(0, 16)
    prefix = tl.full((), 0, tl.uint32)
    remaining = tl.minimum(K, N)
    digit = 0
    done = False
    while (digit < 8) & ~done:
        shift = 28 - 4 * digit
        hist = tl.full((16,), 0, tl.int32)
        for start in range(tl.cdiv(N, BN)):
            i = start * BN + offsets
            value = tl.load(scores + row * in_stride + i, i < N, -float("inf"))
            bits = tl.where(
                value == 0,
                0,
                tl.where(value != value, 0x7FC00000, value.to(tl.uint32, bitcast=True)),
            ).to(tl.uint32)
            ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
            same = (ordered >> (shift + 4).to(tl.uint64)) == (prefix >> (shift + 4).to(tl.uint64))
            if digit == 0:
                same = tl.full((BN,), True, tl.int1)
            bucket = tl.where(same & (i < N), ((ordered >> shift) & 15).to(tl.int32), 16)
            hist += tl.histogram(bucket, 16)
        tail = tl.sum(hist, 0) - tl.cumsum(hist, 0) + hist
        chosen = tl.max(tl.where(tail >= remaining, bins, -1), 0)
        remaining -= tl.sum(tl.where(bins > chosen, hist, 0), 0)
        prefix |= chosen.to(tl.uint32) << shift
        chosen_count = tl.sum(tl.where(bins == chosen, hist, 0), 0)
        done = chosen_count == remaining
        digit += 1
    written = 0
    for start in range(tl.cdiv(N, BN)):
        i = start * BN + offsets
        value = tl.load(scores + row * in_stride + i, i < N, -float("inf"))
        bits = tl.where(
            value == 0,
            0,
            tl.where(value != value, 0x7FC00000, value.to(tl.uint32, bitcast=True)),
        ).to(tl.uint32)
        ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
        equal = (ordered == prefix) & (i < N)
        selected = ((ordered > prefix) | (equal & (_wide_prefix(equal, BN).to(tl.int32) <= remaining))) & (i < N)
        rank = _wide_prefix(selected, BN).to(tl.int32) - 1
        result = tl.where((ordered > 0x007FFFFF) & (ordered < 0xFF800000), i, -1)
        tl.store(output + row * K + written + rank, result, selected)
        written += tl.sum(selected.to(tl.int32), 0)
        remaining -= tl.minimum(remaining, tl.sum(equal.to(tl.int32), 0))
    for start in range(tl.cdiv(K, BN)):
        i = start * BN + offsets
        tl.store(output + row * K + i, -1, (i >= N) & (i < K))


@triton.jit
def select_topk(
    scores: tl.tensor,
    output: tl.tensor,
    N: tl.int64,
    in_stride: tl.int64,
    K: tl.constexpr,
    BN: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    i = tl.arange(0, BN)
    raw = tl.load(scores + row * in_stride + i, i < N, -float("inf"))
    value = tl.where(raw != raw, float("inf"), raw)
    finite = (tl.abs(value) != float("inf")) & (i < N)
    infinities = tl.sum(((value == float("inf")) & (i < N)).to(tl.float32), 0).to(tl.int32)
    finite_count = tl.sum(finite.to(tl.float32), 0).to(tl.int32)
    threshold = tl.full((), -float("inf"), tl.float32)
    if infinities >= K:
        threshold = tl.full((), float("inf"), tl.float32)
    elif finite_count + infinities > K:
        low = tl.min(tl.where(finite, value, float("inf")), 0)
        high = tl.max(tl.where(finite, value, -float("inf")), 0)
        count = tl.full((), 0, tl.int32)
        target = K - infinities
        while (low < high) & (count != target):
            pivot = low * 0.5 + high * 0.5
            pivot = tl.where(pivot <= low, high, pivot)
            count = tl.sum(((value >= pivot) & finite).to(tl.float32), 0).to(tl.int32)
            low = tl.where(count >= target, pivot, low)
            bits = pivot.to(tl.uint32, bitcast=True)
            ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32) - 1
            down = (
                tl.where((ordered & 0x80000000) != 0, ordered ^ 0x80000000, ~ordered)
                .to(tl.uint32)
                .to(tl.float32, bitcast=True)
            )
            high = tl.where(count < target, down, high)
        threshold = low
    greater = (value > threshold) & (i < N)
    equal = (value == threshold) & (i < N)
    remaining = K - tl.sum(greater.to(tl.float32), 0)
    selected = greater | (equal & (_wide_prefix(equal, BN) <= remaining))
    rank = _wide_prefix(selected, BN).to(tl.int32) - 1
    result = tl.where(finite, i, -1)
    BK: tl.constexpr = triton.next_power_of_2(K)
    compact = mlu.scatter(tl.full((BK,), -1, tl.int32), result, rank, selected)
    j = tl.arange(0, BK)
    tl.store(output + row * K + j, compact, j < K)
