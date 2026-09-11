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

"""Loaders for optional draft-owned weights in speculative-decoding models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import torch
import torch.nn as nn

from .parallel_load_context import ParallelLoadContext
from .scoped_weight_loader import ScopedWeightLoader


class _DraftModelWithConfig(Protocol):
    cfg: Any  # duck-typed: reads .hidden_size / .vocab_size below
    dtype: torch.dtype
    device: torch.device


def load_own_weight(
    model: object,
    weights: ScopedWeightLoader,
    checkpoint_key: str,
    attr_name: str,
    build_module: Callable[[], nn.Module],
    *,
    context: ParallelLoadContext,
    shard_dim: int,
) -> None:
    """Load a draft-owned weight when the checkpoint ships one, else share the target's.

    When the checkpoint has no own tensor, ``model.<attr_name>`` stays ``None`` and the
    C++ bridge reads the target model's tensor instead.
    """
    if not weights.has(checkpoint_key):
        return
    setattr(model, attr_name, build_module())
    weights.load_tensor(
        getattr(model, attr_name).weight,
        checkpoint_key,
        dim=shard_dim,
        rank=context.tp_rank,
        world_size=context.tp_size,
    )


def load_own_lm_head(
    model: _DraftModelWithConfig,
    weights: ScopedWeightLoader,
    *,
    context: ParallelLoadContext,
) -> None:
    # Imported lazily so this helper stays torch-only at import time. The layers
    # package needs the C++ runtime bootstrap, which is ready by load_weights.
    from xllm.python.layers import ColumnParallelLinear

    cfg = model.cfg
    load_own_weight(
        model,
        weights,
        "lm_head.weight",
        "lm_head",
        lambda: ColumnParallelLinear(
            cfg.hidden_size,
            cfg.vocab_size // context.tp_size,
            context.tp_size,
            gather_output=True,
            dtype=model.dtype,
            device=model.device,
        ),
        context=context,
        shard_dim=0,
    )
