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

"""Metadata builder for Qwen3.5 NPU sequence-parallel experts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import TYPE_CHECKING

import torch

from xllm.python.layers.npu.qwen3_5.moe_metadata import Qwen3_5MoeMetadata
from xllm.python.layers.sequence_parallel import sequence_parallel_enabled
from xllm.python.model_executor.input_batch import InputBatch

if TYPE_CHECKING:
    from xllm.python.attention.backend import AttentionMetadata


class Qwen3_5MoeMetadataBuilder:
    """Prepare rank-local token validity without owning model intermediates."""

    metadata_type = Qwen3_5MoeMetadata

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> Qwen3_5MoeMetadataBuilder | None:
        if not sequence_parallel_enabled(config):
            return None
        if str(config.get("device", "cuda")).split(":", 1)[0] not in ("npu", "privateuseone"):
            return None
        if int(config.get("num_experts", config.get("n_routed_experts", 0))) <= 0:
            return None

        tp_size = int(config.get("tp_size", 1))
        dp_size = int(config.get("dp_size", 1))
        return cls(
            tp_size=tp_size,
            tp_rank=int(config.get("tp_rank", 0)),
            dp_size=dp_size,
            dp_rank=int(config.get("dp_rank", 0)),
        )

    def __init__(
        self,
        tp_size: int,
        tp_rank: int,
        dp_size: int,
        dp_rank: int,
    ) -> None:
        if tp_size <= 0 or not 0 <= tp_rank < tp_size:
            raise ValueError(f"invalid SP TP geometry: size={tp_size}, rank={tp_rank}")
        if dp_size <= 0 or not 0 <= dp_rank < dp_size:
            raise ValueError(f"invalid SP DP geometry: size={dp_size}, rank={dp_rank}")
        self._tp_size = tp_size
        self._tp_rank = tp_rank
        self._dp_size = dp_size
        self._dp_rank = dp_rank

    def build(
        self,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> Qwen3_5MoeMetadata:
        self._validate_batch(input_batch, graph=False)
        execution_counts, valid_rows = self._token_counts(input_batch, metadata)
        local_capacity = self._shard_capacity(input_batch.num_tokens)
        dispatch_capacity = self._shard_capacity(max(execution_counts))
        mask = torch.zeros(dispatch_capacity, dtype=torch.int8, device=input_batch.input_ids.device)
        local_valid_rows = self._local_valid_rows(valid_rows, local_capacity)
        mask[:local_valid_rows].fill_(1)
        return Qwen3_5MoeMetadata(active_token_mask=mask)

    def allocate_persistent(
        self,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> Qwen3_5MoeMetadata:
        self._validate_batch(input_batch, graph=True)
        # Allocate storage only: capture/replay contents belong to update_persistent.
        return Qwen3_5MoeMetadata(
            active_token_mask=torch.zeros(
                self._shard_capacity(input_batch.num_tokens_after_padding),
                dtype=torch.int8,
                device=input_batch.input_ids.device,
            )
        )

    def update_persistent(
        self,
        persistent_metadata: object,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> None:
        if not isinstance(persistent_metadata, Qwen3_5MoeMetadata):
            raise TypeError("Qwen3.5 SP received invalid persistent metadata")
        self._validate_batch(input_batch, graph=True)
        execution_counts, valid_rows = self._token_counts(input_batch, metadata)
        token_capacity = input_batch.num_tokens_after_padding
        if max(execution_counts) > token_capacity:
            raise RuntimeError(
                "DP execution token counts must fit the Qwen3.5 SP graph capacity: "
                f"counts={execution_counts}, capacity={token_capacity}"
            )
        local_capacity = self._shard_capacity(token_capacity)
        mask = persistent_metadata.active_token_mask
        if mask.shape != (local_capacity,):
            raise ValueError("Qwen3.5 SP mask shape does not match the graph capacity")
        if mask.dtype != torch.int8:
            raise TypeError("Qwen3.5 SP mask must use int8")
        if mask.device != input_batch.input_ids.device or not mask.is_contiguous():
            raise ValueError("Qwen3.5 SP mask must be contiguous and on the input device")

        local_valid_rows = self._local_valid_rows(valid_rows, local_capacity)
        offset = self._tp_rank * local_capacity
        # Authoritative real rows exclude scheduler dummy rows; the graph mask
        # independently excludes its padding. No device value is read on host.
        mask.zero_()
        if local_valid_rows:
            mask[:local_valid_rows].copy_(~input_batch.is_padding[offset : offset + local_valid_rows])

    def _shard_capacity(self, token_count: int) -> int:
        return max(1, (token_count + self._tp_size - 1) // self._tp_size)

    def _local_valid_rows(self, valid_rows: int, local_capacity: int) -> int:
        return min(local_capacity, max(0, valid_rows - self._tp_rank * local_capacity))

    def _token_counts(
        self,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> tuple[tuple[int, ...], int]:
        execution_counts = self._host_counts(metadata, "dp_execution_token_counts")
        raw_counts = self._host_counts(metadata, "raw_dp_execution_token_counts")
        expected_valid_rows = 0 if input_batch.is_dummy else input_batch.num_tokens
        if self._dp_size == 1:
            execution_counts = execution_counts or (input_batch.num_tokens,)
            raw_counts = raw_counts or (expected_valid_rows,)
        if len(execution_counts) != self._dp_size:
            raise RuntimeError(f"expected {self._dp_size} DP execution token counts, got {execution_counts}")
        if len(raw_counts) != self._dp_size:
            raise RuntimeError(f"expected {self._dp_size} raw DP execution token counts, got {raw_counts}")
        if any(count < 0 for count in execution_counts):
            raise RuntimeError("Qwen3.5 SP DP execution token counts must be non-negative")
        if execution_counts[self._dp_rank] != input_batch.num_tokens:
            raise RuntimeError("DP execution token count does not match the local input")
        if any(raw < 0 or raw > count for raw, count in zip(raw_counts, execution_counts)):
            raise RuntimeError("raw DP execution token counts must fit the execution rows")
        valid_rows = raw_counts[self._dp_rank]
        if valid_rows != expected_valid_rows:
            raise RuntimeError("raw DP execution token count disagrees with InputBatch real token semantics")
        return execution_counts, valid_rows

    @staticmethod
    def _host_counts(metadata: AttentionMetadata, name: str) -> tuple[int, ...]:
        counts = getattr(metadata, name, ())
        if not isinstance(counts, Sequence) or any(not isinstance(count, Integral) for count in counts):
            raise TypeError(f"{name} must be a host integer sequence")
        return tuple(int(count) for count in counts)

    @staticmethod
    def _validate_batch(input_batch: InputBatch, *, graph: bool) -> None:
        token_capacity = input_batch.num_tokens_after_padding
        if input_batch.num_tokens < 0 or token_capacity < input_batch.num_tokens:
            raise ValueError("Qwen3.5 SP execution token rows must fit the input capacity")
        if input_batch.input_ids.shape != (token_capacity,):
            raise ValueError("Qwen3.5 SP input rows do not match the InputBatch capacity")
        if not graph:
            if token_capacity != input_batch.num_tokens or input_batch.is_padding is not None:
                raise ValueError("eager Qwen3.5 SP requires an unpadded InputBatch")
            return
        if token_capacity <= 0:
            raise ValueError("Qwen3.5 SP graph token capacity must be positive")
        padding = input_batch.is_padding
        if padding is None or padding.shape != (token_capacity,):
            raise ValueError("Qwen3.5 SP graph execution requires a padding mask matching its capacity")
        if padding.dtype != torch.bool:
            raise TypeError("Qwen3.5 SP graph padding mask must use bool")
        if padding.device != input_batch.input_ids.device:
            raise ValueError("Qwen3.5 SP graph padding mask must be on the input device")
