# Copyright 2026 The xLLM Authors.
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


@triton.jit
def _prefix(mask: tl.tensor, BP: tl.constexpr) -> tl.tensor:
    # Boolean dot products provide exact local ranks without a serial scan.
    if BP >= 256:
        C: tl.constexpr = 32
        r = tl.arange(0, C)
        x = mask.to(tl.float32).reshape(BP // C, C)
        local = tl.dot(x.to(tl.bfloat16), (r[:, None] <= r[None, :]).to(tl.bfloat16))
        counts = tl.sum(x, 1)
        g = tl.arange(0, BP // C)
        before = tl.sum(tl.where(g[:, None] < g[None, :], counts[:, None], 0.0), 0)
        return (local + before[:, None]).reshape(BP).to(tl.int32)
    r = tl.arange(0, BP)
    return tl.sum(tl.where(r[:, None] <= r[None, :], mask[:, None].to(tl.float32), 0.0), 0).to(tl.int32)


@triton.jit
def _read_blocks(
    table: tl.tensor,
    cached_table: tl.tensor,
    blocks: tl.tensor,
    valid: tl.tensor,
    batch: tl.int64,
    row_stride: tl.int64,
    column_stride: tl.int64,
    BT: tl.constexpr,
) -> tl.tensor:
    if BT:
        indices = tl.where(valid, blocks, 0).reshape((blocks.numel,)).to(tl.int32)
        values = tl.gather(cached_table, indices, 0).reshape(blocks.shape)
    else:
        values = tl.load(table + batch * row_stride + blocks * column_stride, valid, 0)
    return values.to(tl.int64)


@triton.jit
def _read_blocks_i32(
    table: tl.tensor,
    cached_table: tl.tensor,
    blocks: tl.tensor,
    valid: tl.tensor,
    batch: tl.int64,
    row_stride: tl.int64,
    column_stride: tl.int64,
    BT: tl.constexpr,
) -> tl.tensor:
    if BT:
        indices = tl.where(valid, blocks, 0).reshape((blocks.numel,)).to(tl.int32)
        return tl.gather(cached_table, indices, 0).reshape(blocks.shape).to(tl.int32)
    return tl.load(table + batch * row_stride + blocks * column_stride, valid, 0).to(tl.int32)


@triton.jit(
    do_not_specialize=[
        "table_width",
        "table_row_stride",
        "table_column_stride",
        "num_queries",
    ]
)
def kpool_expand(
    pool_ids: tl.tensor,
    positions: tl.tensor,
    row_batch: tl.tensor,
    table: tl.tensor,
    slots: tl.tensor,
    context_lens: tl.tensor,
    pool_row_stride: tl.int64,
    pool_column_stride: tl.int64,
    position_stride: tl.int64,
    row_batch_stride: tl.int64,
    table_row_stride: tl.int64,
    table_column_stride: tl.int64,
    table_width: tl.int64,
    num_queries: tl.int64,
    K: tl.constexpr,
    P: tl.constexpr,
    B: tl.constexpr,
    W: tl.constexpr,
    BP: tl.constexpr,
    BM: tl.constexpr,
    BT: tl.constexpr,
    TAIL: tl.constexpr,
    BTAIL: tl.constexpr,
    LOCAL_I32: tl.constexpr,
) -> None:
    """Stably expand pools and tail with bounded scratch and no intermediate tensor."""
    for row in range(tl.program_id(0).to(tl.int64), num_queries, tl.num_programs(0)):
        batch = tl.load(row_batch + row * row_batch_stride).to(tl.int64)
        position = tl.load(positions + row * position_stride)
        if LOCAL_I32:
            position = position.to(tl.int32)
            table_bound = table_width.to(tl.int32)
        else:
            position = position.to(tl.int64)
            table_bound = table_width
        cached_table = table
        if BT:
            table_columns = tl.arange(0, BT)
            if LOCAL_I32:
                cached_table = tl.load(
                    table + batch * table_row_stride + table_columns * table_column_stride,
                    table_columns < table_bound,
                    0,
                    cache_modifier=".ca",
                )
            else:
                cached_table = tl.load(
                    table + batch * table_row_stride + table_columns * table_column_stride,
                    table_columns < table_bound,
                    0,
                )
        offsets = tl.arange(0, BP)
        members = tl.arange(0, BM)
        if LOCAL_I32:
            count = tl.full((), 0, tl.int32)
        else:
            count = tl.full((), 0, tl.int64)
        for start in range(tl.cdiv(K, BP)):
            i = start * BP + offsets
            pool = tl.load(pool_ids + row * pool_row_stride + i * pool_column_stride, i < K, -1)
            if LOCAL_I32:
                # Selected logical slots fit the int32 output-length contract.
                pool = pool.to(tl.int32)
            else:
                pool = pool.to(tl.int64)
            valid = (i < K) & (pool >= 0)
            if LOCAL_I32:
                valid_count = tl.sum(valid.to(tl.float32), 0).to(tl.int32)
            else:
                valid_count = tl.sum(valid.to(tl.float32), 0).to(tl.int64)
            local_rank = _prefix(valid, BP) - 1
            if B % P == 0:
                # An aligned pool occupies consecutive physical slots in one block.
                logical_base = tl.where(valid, pool * P, 0)
                blocks = tl.minimum(logical_base // B, table_bound - 1)
                if LOCAL_I32:
                    physical = _read_blocks_i32(
                        table,
                        cached_table,
                        blocks,
                        valid,
                        batch,
                        table_row_stride,
                        table_column_stride,
                        BT,
                    )
                else:
                    physical = _read_blocks(
                        table,
                        cached_table,
                        blocks,
                        valid,
                        batch,
                        table_row_stride,
                        table_column_stride,
                        BT,
                    )
                mapped = (physical * B + logical_base % B).to(tl.int32)
                physical = mlu.scatter(tl.full((BP,), 0, tl.int32), mapped, local_rank, valid)
            else:
                pool = mlu.scatter(tl.full((BP,), 0, tl.int64), pool, local_rank, valid)
            valid = offsets < valid_count
            if LOCAL_I32:
                ranks = offsets.to(tl.int32) + count
            else:
                ranks = offsets.to(tl.int64) + count
            for member_tile in range(tl.cdiv(P, BM)):
                member = member_tile * BM + members
                mask = valid[:, None] & (member[None, :] < P)
                if B % P == 0:
                    physical_slots = physical[:, None] + member[None, :]
                else:
                    logical = tl.where(mask, pool[:, None] * P + member[None, :], 0)
                    blocks = tl.minimum(logical // B, table_bound - 1)
                    if LOCAL_I32:
                        physical = _read_blocks_i32(
                            table,
                            cached_table,
                            blocks,
                            mask,
                            batch,
                            table_row_stride,
                            table_column_stride,
                            BT,
                        )
                    else:
                        physical = _read_blocks(
                            table,
                            cached_table,
                            blocks,
                            mask,
                            batch,
                            table_row_stride,
                            table_column_stride,
                            BT,
                        )
                    physical_slots = physical * B + logical % B
                output = slots + row * W + ranks[:, None] * P + member[None, :]
                if LOCAL_I32:
                    tl.store(
                        output,
                        physical_slots.to(tl.int32),
                        mask,
                        cache_modifier=".cg",
                    )
                else:
                    tl.store(output, physical_slots.to(tl.int32), mask)
            count += valid_count
        selected = count * P
        tail_count = tl.maximum(0, (position + 1) % P) if TAIL else 0
        tl.store(context_lens + row, (selected + tail_count).to(tl.int32))
        # Fill the tail and all remaining padding; even an all-invalid row is written.
        tail_begin = ((position + 1) // P) * P
        j = tl.arange(0, BTAIL)
        for start in range(tl.cdiv(W - selected, BTAIL)):
            offset = start * BTAIL + j
            logical = tail_begin + offset
            active = offset < tail_count
            blocks = tl.minimum(tl.where(active, logical, 0) // B, table_bound - 1)
            if LOCAL_I32:
                physical = _read_blocks_i32(
                    table,
                    cached_table,
                    blocks,
                    active,
                    batch,
                    table_row_stride,
                    table_column_stride,
                    BT,
                )
            else:
                physical = _read_blocks(
                    table,
                    cached_table,
                    blocks,
                    active,
                    batch,
                    table_row_stride,
                    table_column_stride,
                    BT,
                )
            physical_slots = physical * B + logical % B
            output = slots + row * W + selected + offset
            value = tl.where(active, physical_slots, -1).to(tl.int32)
            mask = selected + offset < W
            if LOCAL_I32:
                tl.store(output, value, mask, cache_modifier=".cg")
            else:
                tl.store(output, value, mask)
