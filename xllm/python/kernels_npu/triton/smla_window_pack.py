# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-launch SMLA ring gather and current-chunk packing."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_window(
    cache,
    tables,
    chunk,
    lengths,
    output,
    SEQ: tl.constexpr,
    Q_LEN: tl.constexpr,
    WIDTH: tl.constexpr,
    WINDOW: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    channel = tl.arange(0, WIDTH)
    context = tl.load(lengths + SEQ)
    previous = context - Q_LEN
    absolute = previous - WINDOW + row
    position = tl.maximum(absolute, 0)
    ring = row < WINDOW
    column = (position // PAGE_SIZE) % COLS
    page = tl.maximum(tl.load(tables + SEQ * TABLE_STRIDE + column * COL_STRIDE, mask=ring, other=0), 0)
    slot = page.to(tl.int64) * PAGE_SIZE + position % PAGE_SIZE
    history = tl.load(
        cache + slot[:, None] * WIDTH + channel[None, :],
        mask=(ring & (absolute >= 0) & (context > 0))[:, None],
        other=0,
    )
    current = tl.load(
        chunk + tl.maximum(row - WINDOW, 0)[:, None] * WIDTH + channel[None, :],
        mask=((row >= WINDOW) & (row < WINDOW + Q_LEN) & (context > 0))[:, None],
        other=0,
    )
    tl.store(
        output + row[:, None] * WIDTH + channel[None, :],
        tl.where(ring[:, None], history, current),
        mask=(row < WINDOW + Q_LEN)[:, None],
    )


def can_pack_smla_window(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    chunk: torch.Tensor,
    context_lengths: torch.Tensor,
    seq: int,
    block_size: int,
    window: int,
) -> bool:
    """Select the supported packing layout using host-visible metadata only."""
    # Only the V4.1 Flash layout (head_dim=512, window=128) is validated.
    return (
        cache.ndim == chunk.ndim == 2
        and cache.size(1) == chunk.size(1) == 512
        and cache.dtype == chunk.dtype == torch.bfloat16
        and cache.is_contiguous()
        and chunk.is_contiguous()
        and cache.device.type == "npu"
        and block_table.ndim == 2
        and block_table.size(1) > 0
        and block_table.dtype in (torch.int32, torch.int64)
        and context_lengths.ndim == 1
        and context_lengths.is_contiguous()
        and context_lengths.dtype in (torch.int32, torch.int64)
        and 0 <= seq < block_table.size(0)
        and seq < context_lengths.numel()
        and all(item.device == cache.device for item in (block_table, chunk, context_lengths))
        and block_size > 0
        and window == 128
    )


def pack_smla_window(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    chunk: torch.Tensor,
    context_lengths: torch.Tensor,
    seq: int,
    block_size: int,
    window: int,
) -> torch.Tensor:
    """Gather valid ring rows and append chunk, zeroing padding/dummy rows.

    The caller must first accept the layout via :func:`can_pack_smla_window`.
    """
    q_len = chunk.size(0)
    output = torch.empty((window + q_len, 512), dtype=chunk.dtype, device=chunk.device)
    _pack_window[(triton.cdiv(window + q_len, 4),)](
        cache,
        block_table,
        chunk,
        context_lengths,
        output,
        SEQ=seq,
        Q_LEN=q_len,
        WIDTH=512,
        WINDOW=window,
        PAGE_SIZE=block_size,
        COLS=block_table.size(1),
        TABLE_STRIDE=block_table.stride(0),
        COL_STRIDE=block_table.stride(1),
        ROWS=4,
    )
    return output
