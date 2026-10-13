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

"""Shared CPU reference for the in-repo Engram int8 lookup dequant.

``xllm_ops.engram_int8_lookup_cpu`` (built from
``core/kernels/npu/xllm_ops/engram_int8_lookup_cpu.cpp``) fuses the host gather,
group-32 scale and bf16 cast into one call. Standalone pytest processes never
link the xLLM binary, so the engram tests substitute this bitwise-equivalent
eager reference for the op. It lives in one module -- imported by every test
that needs it -- so a semantics change updates one source instead of silently
desynchronizing duplicated copies.
"""

from __future__ import annotations

import torch


def _reference_engram_lookup(
    weight: torch.Tensor,
    scale: torch.Tensor,
    ids: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """CPU stand-in for ``xllm_ops.engram_int8_lookup_cpu``.

    Mirrors ``core/kernels/npu/xllm_ops/engram_int8_lookup_cpu.cpp``: per-group
    fp32 ``float(int8) * scale`` rounded to bf16, written in place.
    """
    payload = weight.index_select(0, ids).to(torch.float32)
    factor = scale.index_select(0, ids).repeat_interleave(32, dim=-1)
    output.copy_((payload * factor).to(torch.bfloat16))
