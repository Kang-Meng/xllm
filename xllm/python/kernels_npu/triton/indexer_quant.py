# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A3 INT8 indexer quantization inspired by DeepInfer's fused transform.

Retain xLLM's FP16 scale rounding and half-to-even INT8 rounding. This is an
independent INT8 implementation, rather than the upstream Ascend 950 FP8/FP4
TileLang kernel. Input finiteness follows the existing indexer contract.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_rows(
    x,
    codes,
    scales,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    column = tl.arange(0, WIDTH)
    value = tl.load(x + row[:, None] * WIDTH + column[None, :], row[:, None] < ROWS, 0).to(tl.float32)
    maximum = tl.max(tl.abs(value), 1)
    # Preserve the intermediate FP16 scale before normalizing the input.
    scale = tl.maximum((maximum / 127.0).to(tl.float16).to(tl.float32), 2.0**-24)
    rounded = tl.extra.cann.libdevice.nearbyint(value / scale[:, None])
    result = tl.minimum(tl.maximum(rounded, -127.0), 127.0).to(tl.int8)
    tl.store(codes + row[:, None] * WIDTH + column[None, :], result, row[:, None] < ROWS)
    tl.store(scales + row, scale.to(tl.float16), row < ROWS)


def quantize_indexer_rows_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize contiguous BF16 [..., 128] rows with graph-stable shapes."""
    if x.device.type != "npu" or x.dtype != torch.bfloat16 or x.ndim < 2:
        raise ValueError("Expected an NPU BF16 matrix or batch of matrices")
    if not x.is_contiguous() or x.size(-1) != 128:
        raise ValueError("Expected contiguous indexer rows with width 128")
    codes = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(x.shape[:-1], dtype=torch.float16, device=x.device)
    rows = x.numel() // 128
    if rows:
        block_rows = 4 if rows <= 256 else 32
        _quantize_rows[(triton.cdiv(rows, block_rows),)](
            x, codes, scales, ROWS=rows, WIDTH=128, BLOCK_ROWS=block_rows, enable_fp_fusion=False
        )
    return codes, scales
