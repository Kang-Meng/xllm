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

"""Typed, DP-local source activity for GLM5.2 AllToAllV expert parallelism."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from xllm.python.model_executor.input_batch import InputBatch

if TYPE_CHECKING:
    from xllm.python.attention.backend import AttentionMetadata


@dataclass(slots=True)
class Glm52EpMetadata:
    """Active query rows only; cached prefixes and dummy rows are not sources."""

    local_token_mask: torch.Tensor


class Glm52EpMetadataBuilder:
    """Build a stable source mask from authoritative, pre-CP InputBatch counts."""

    metadata_type = Glm52EpMetadata

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> Glm52EpMetadataBuilder | None:
        if str(config.get("model_type", "")) != "glm_moe_dsa" or int(config.get("ep_size", 1)) <= 1:
            return None
        return cls(int(config.get("dp_size", 1)), int(config.get("dp_rank", 0)))

    def __init__(self, dp_size: int, dp_rank: int) -> None:
        if dp_size <= 0 or not 0 <= dp_rank < dp_size:
            raise ValueError(f"invalid GLM5.2 EP DP geometry: size={dp_size}, rank={dp_rank}")
        self._dp_size = dp_size
        self._dp_rank = dp_rank

    def build(self, input_batch: InputBatch, metadata: AttentionMetadata) -> Glm52EpMetadata:
        mask = torch.zeros(input_batch.num_tokens, dtype=torch.bool, device=input_batch.input_ids.device)
        self._update_mask(mask, input_batch, metadata)
        return Glm52EpMetadata(mask)

    def allocate_persistent(self, input_batch: InputBatch, metadata: AttentionMetadata) -> Glm52EpMetadata:
        capacity = input_batch.num_tokens_after_padding
        if capacity <= 0:
            raise ValueError("GLM5.2 EP graph token capacity must be positive")
        return Glm52EpMetadata(torch.zeros(capacity, dtype=torch.bool, device=input_batch.input_ids.device))

    def update_persistent(
        self,
        persistent_metadata: object,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> None:
        if not isinstance(persistent_metadata, Glm52EpMetadata):
            raise TypeError("GLM5.2 EP received invalid persistent execution metadata")
        self._update_mask(persistent_metadata.local_token_mask, input_batch, metadata)

    def _update_mask(self, mask: torch.Tensor, input_batch: InputBatch, metadata: AttentionMetadata) -> None:
        if mask.ndim != 1 or mask.dtype != torch.bool or mask.device != input_batch.input_ids.device:
            raise ValueError("GLM5.2 EP source mask must be a one-dimensional device bool tensor")
        if input_batch.num_tokens > mask.numel():
            raise ValueError("GLM5.2 EP source mask capacity must cover all execution rows")
        if self._dp_size > 1:
            counts = tuple(int(count) for count in getattr(metadata, "raw_dp_execution_token_counts", ()))
            if len(counts) != self._dp_size:
                raise ValueError("GLM5.2 EP masks require raw execution counts for every DP rank")
            valid_rows = counts[self._dp_rank]
        else:
            valid_rows = 0 if input_batch.is_dummy else input_batch.num_tokens
        if not 0 <= valid_rows <= input_batch.num_tokens:
            raise ValueError("GLM5.2 EP active-token rows exceed the execution batch")
        if input_batch.is_dummy and valid_rows:
            raise ValueError("GLM5.2 EP dummy batches cannot own active source tokens")
        mask.zero_()
        mask[:valid_rows].fill_(True)
        if input_batch.is_padding is not None:
            if input_batch.is_padding.shape != mask.shape or input_batch.is_padding.dtype != torch.bool:
                raise ValueError("GLM5.2 EP graph padding must match the source mask")
            mask.logical_and_(~input_batch.is_padding)
        return
