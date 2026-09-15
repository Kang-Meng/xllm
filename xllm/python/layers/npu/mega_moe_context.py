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

"""Forward-scoped tensor inputs for NPU Token Owner MegaMoe."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from xllm.python.model_executor.forward_context import get_execution_context


@dataclass(frozen=True, slots=True)
class MegaMoeLayerSpec:
    """Static execution requirements declared by one MegaMoe layer."""

    layer_id: int
    hidden_size: int
    top_k: int
    token_limit: int
    dtype: torch.dtype
    device: torch.device
    dp_size: int
    dp_rank: int
    tp_rank: int


@dataclass(frozen=True, slots=True)
class MegaMoeLayerContext:
    """Tensor addresses supplied to one MegaMoe layer."""

    input_buffer: torch.Tensor | None
    topk_weights_buffer: torch.Tensor | None
    topk_ids_buffer: torch.Tensor | None
    output_buffer: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class MegaMoeContext:
    """Shared and per-layer inputs for one eager forward or graph entry."""

    token_capacity: int
    active_token_mask: torch.Tensor
    layers: dict[int, MegaMoeLayerContext | None]


def get_mega_moe_layer_context(
    layer_id: int,
) -> tuple[MegaMoeContext, MegaMoeLayerContext | None]:
    """Return the inputs prepared for ``layer_id`` or its explicit fallback."""
    context = get_execution_context(MegaMoeContext)
    if context is None:
        raise RuntimeError("Token Owner MegaMoe context is unavailable")
    if layer_id not in context.layers:
        raise RuntimeError(f"Token Owner MegaMoe context for layer {layer_id} is unavailable")
    return context, context.layers[layer_id]
