# Copyright 2025-2026 The xLLM Authors.
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

"""Shared base for Python model-executor causal LMs.

A concrete model builds its own graph, lm_head, and config; the wiring that is
identical across models -- the weight-loading entry point and logits
computation -- lives here.

Model forward is driven by the runners owned by Python ModelExecutor;
PyCausalLM no longer calls model.forward() directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from xllm_weight_loader import StateDict


class PyModelBase(nn.Module):
    """Base class for causal LMs the C++ ``PyCausalLM`` bridge drives.

    Subclass contract:
      * build ``self.model`` as the runner's execution module;
      * provide logits and shared-weight access, using ``self.lm_head`` by default;
      * implement ``load_weights()``.
    """

    model: nn.Module
    lm_head: nn.Module | None

    @staticmethod
    def resolve_dtype(dtype: object) -> torch.dtype:
        """Resolve a torch dtype from a ``torch.dtype`` or its string name."""
        if isinstance(dtype, torch.dtype):
            return dtype
        name = dtype if dtype else "bfloat16"
        resolved = getattr(torch, name, None)
        if not isinstance(resolved, torch.dtype):
            raise ValueError(f"Unknown dtype: {dtype!r}")
        return resolved

    def compute_logits(self, hidden: torch.Tensor, selected_idxes: torch.Tensor | None) -> torch.Tensor:
        if self.lm_head is None:
            raise RuntimeError("LM head must be loaded or shared before computing logits")
        if selected_idxes is not None and selected_idxes.numel() > 0:
            hidden = hidden.index_select(0, selected_idxes)
        return self.lm_head(hidden)

    def get_lm_head(self) -> nn.Module | None:
        return self.lm_head

    def get_token_embedding(self) -> nn.Module | None:
        return self.model.embed_tokens

    def share_weights_from(self, source: PyModelBase) -> None:
        """Borrow only missing vocabulary modules, preserving draft-owned weights."""
        if self.lm_head is None:
            self.lm_head = source.get_lm_head()
        if self.model.embed_tokens is None:
            self.model.embed_tokens = source.get_token_embedding()

    # -- weight loading -------------------------------------------------------
    def load_weights(
        self,
        state_dicts: list[StateDict],
        tp_rank: int,
        tp_size: int,
    ) -> None:
        """Load checkpoint weights into this model's parameters.

        Called by the C++ bridge (``PyCausalLM::load_model``).
        The model owns ALL weight transform logic: TP slicing, fused-weight
        concatenation, format conversion, etc.
        """
        raise NotImplementedError

    def supports_device_only_decode_inputs(self, kv_page_size: int) -> bool:
        """Declare whether the complete decode path needs no unready Host inputs.

        Model authors own this contract for input preparation and the selected
        operator and runner implementations. Ready Host shapes and configuration
        are allowed. kv_page_size is the physical tokens-per-page selected by
        the runtime, available before KV cache binding.
        """
        return False

    def adapt_weights_for_reference_model(
        self,
        reference_model_path: str,
    ) -> None:
        """Adapt loaded draft weights to the reference model's runtime basis."""
        del reference_model_path
