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

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from xllm.python.attention.backend import AttentionMetadata, LayerCache
from xllm.python.model_executor.input_batch import InputBatch


@dataclass
class Csa2BlockAttentionMetadata:
    """Complete non-causal query blocks with a shared trailing SWA window."""

    query_lens: tuple[int, ...]
    context_lengths: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


class Csa2BlockAttentionMetadataBuilder:
    metadata_type = Csa2BlockAttentionMetadata

    def __init__(self, query_width: int, window_size: int, max_model_len: int) -> None:
        if query_width <= 0 or window_size <= 0 or max_model_len <= 0:
            raise ValueError("CSA2 block attention requires positive query, window and context sizes")
        self._query_width = query_width
        self._block_size = window_size
        self._max_model_len = max_model_len

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> Csa2BlockAttentionMetadataBuilder:
        if int(config.get("cp_size", 1)) != 1:
            raise ValueError("CSA2 block attention does not support context parallelism")
        return cls(
            int(config["draft_query_width"]),
            int(config.get("window_size", 128)),
            int(config["max_position_embeddings"]),
        )

    def bind_layer_caches(self, layer_caches: list[LayerCache]) -> None:
        block_sizes = {int(cache.swa.shape[1]) for cache in layer_caches if cache.swa is not None}
        if len(block_sizes) != 1 or any(cache.swa is None for cache in layer_caches):
            raise ValueError("CSA2 block attention requires SWA caches with a common block size")
        self._block_size = block_sizes.pop()

    def build(self, input_batch: InputBatch, metadata: AttentionMetadata) -> Csa2BlockAttentionMetadata:
        tables = () if input_batch.is_dummy else metadata.multi_block_tables
        table_columns = tables[0].shape[1] if tables and tables[0] is not None else 1
        result = self._allocate(input_batch, table_columns)
        self.update_persistent(result, input_batch, metadata)
        return result

    def allocate_persistent(self, input_batch: InputBatch, metadata: AttentionMetadata) -> Csa2BlockAttentionMetadata:
        """Allocate capacity without reading a request's positions or tables."""
        table_columns = (self._max_model_len + self._block_size - 1) // self._block_size
        return self._allocate(input_batch, table_columns)

    def _allocate(self, input_batch: InputBatch, table_columns: int) -> Csa2BlockAttentionMetadata:
        capacity = input_batch.num_tokens_after_padding
        if capacity <= 0 or capacity % self._query_width:
            raise ValueError("CSA2 capacity must contain complete query blocks")
        num_sequences = capacity // self._query_width

        device = input_batch.input_ids.device
        return Csa2BlockAttentionMetadata(
            query_lens=(self._query_width,) * num_sequences,
            context_lengths=torch.zeros(num_sequences, dtype=torch.int32, device=device),
            block_table=torch.zeros((num_sequences, table_columns), dtype=torch.int32, device=device),
            slot_mapping=torch.full((capacity,), -1, dtype=torch.int32, device=device),
        )

    def update_persistent(
        self,
        persistent_metadata: object,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> None:
        """Refresh fixed addresses, disabling cache writes for capture, dummy and unused queries."""
        if not isinstance(persistent_metadata, Csa2BlockAttentionMetadata):
            raise TypeError("CSA2 block attention received invalid persistent metadata")
        target = persistent_metadata
        if input_batch.num_tokens_after_padding != target.slot_mapping.numel():
            raise ValueError("CSA2 query capacity changed after allocation")
        target.context_lengths.zero_()
        target.block_table.zero_()
        target.slot_mapping.fill_(-1)
        if input_batch.is_dummy or getattr(metadata, "is_graph_capture", False):
            return
        if input_batch.num_scheduled_tokens != (self._query_width,) * input_batch.num_reqs:
            raise ValueError("CSA2 block attention requires one complete query block per request")
        if any(input_batch.is_prefilling):
            raise ValueError("CSA2 block attention expects decode queries")
        tables = metadata.multi_block_tables
        if not tables or tables[0] is None:
            raise ValueError("CSA2 block attention requires the leading SWA block table")
        source = tables[0]
        if source.ndim != 2 or not 0 < source.shape[1] <= target.block_table.shape[1]:
            raise ValueError("CSA2 SWA table does not match the bound cache layout")
        if source.shape[0] == input_batch.num_tokens:
            source = source[:: self._query_width]
        if source.shape[0] != input_batch.num_reqs:
            raise ValueError("CSA2 SWA table does not match the request count")
        num_reqs = input_batch.num_reqs
        ring_blocks = source.shape[1]
        table_columns = torch.arange(target.block_table.shape[1], device=source.device) % ring_blocks
        target.block_table[:num_reqs].copy_(source[:, table_columns], non_blocking=True)
        positions = input_batch.positions[: input_batch.num_tokens].reshape(num_reqs, self._query_width)
        target.context_lengths[:num_reqs].copy_(positions[:, -1] + 1)

        columns = torch.div(positions, self._block_size, rounding_mode="floor") % ring_blocks
        blocks = target.block_table[:num_reqs].gather(1, columns.long())
        slots = blocks * self._block_size + positions % self._block_size
        last_writes = positions >= positions[:, -1:] - ring_blocks * self._block_size + 1
        slots = torch.where((blocks > 0) & last_writes, slots, -1)
        target.slot_mapping[: input_batch.num_tokens].copy_(slots.reshape(-1))
