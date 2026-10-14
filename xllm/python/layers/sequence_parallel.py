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

"""Sequence-parallel metadata admission and tensor row operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import torch
import torch.nn.functional as F

from xllm.python import distributed
from xllm.python.layers.linear import RowParallelLinear
from xllm.python.layers.sequence_parallel_metadata import EpSpTokenLayout, SpMetadata
from xllm.python.model_executor.input_batch import InputBatch

__all__ = (
    "SpMetadata",
    "SpMetadataBuilder",
    "resolve_sequence_parallel",
    "sequence_parallel_enabled",
    "sp_local_rows",
    "sp_shard",
    "sp_all_gather",
    "sp_reduce_scatter",
    "sp_row_parallel_projection",
)


def resolve_sequence_parallel(
    config: Mapping[str, object],
    decoder_layer_cls: type[torch.nn.Module],
) -> bool:
    """Resolve Qwen3.5 SP from decoder capability and runtime policy.

    The instance role is supplied by the PD launcher.  There is deliberately
    no user-facing environment switch here.  The resolved state is the source
    of truth consumed by the model, layers, and metadata builders.
    """
    if not getattr(decoder_layer_cls, "supports_sequence_parallel", False):
        return False
    instance_role = str(config.get("instance_role", "DEFAULT")).strip().upper()
    if instance_role not in ("PREFILL", "DECODE"):
        return False
    return (
        int(config.get("tp_size", 1)) > 1
        and int(config.get("cp_size", 1)) == 1
        and int(config.get("num_speculative_tokens", 0)) == 0
    )


def sequence_parallel_enabled(config: object) -> bool:
    """Read the resolved SP state without re-evaluating runtime policy."""
    if isinstance(config, Mapping):
        return bool(config.get("sequence_parallel_enabled", False))
    return bool(getattr(config, "sequence_parallel_enabled", False))


class SpMetadataBuilder:
    """Build model-visible sequence-parallel row layouts."""

    metadata_type = SpMetadata

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> SpMetadataBuilder | None:
        device = str(config.get("device", "cuda")).split(":", 1)[0]
        tp_size = int(config.get("tp_size", 1))
        if device not in ("npu", "privateuseone") or not sequence_parallel_enabled(config):
            return None
        return cls(
            tp_size=tp_size,
            tp_rank=int(config.get("tp_rank", 0)),
            dp_size=int(config.get("dp_size", 1)),
            dp_rank=int(config.get("dp_rank", 0)),
        )

    def __init__(self, tp_size: int, tp_rank: int, dp_size: int = 1, dp_rank: int = 0) -> None:
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
        metadata: object,
    ) -> SpMetadata:
        self._validate_batch(input_batch, graph=False)
        expected_valid_rows = 0 if input_batch.is_dummy else input_batch.num_tokens
        execution_counts = self._host_counts(metadata, "dp_execution_token_counts")
        raw_counts = self._host_counts(metadata, "raw_dp_execution_token_counts")
        if self._dp_size == 1:
            execution_counts = execution_counts or (input_batch.num_tokens,)
            raw_counts = raw_counts or (expected_valid_rows,)
        ep_layout = EpSpTokenLayout.from_token_counts(
            execution_counts,
            raw_counts,
            tp_size=self._tp_size,
            tp_rank=self._tp_rank,
            dp_size=self._dp_size,
            dp_rank=self._dp_rank,
            compacted=True,
        )
        if execution_counts[self._dp_rank] != input_batch.num_tokens:
            raise RuntimeError("DP execution token count does not match the local input")
        if raw_counts[self._dp_rank] != expected_valid_rows:
            raise RuntimeError("raw DP execution token count disagrees with InputBatch real token semantics")
        return SpMetadata.from_execution_rows(input_batch.num_tokens, self._tp_size, self._tp_rank, ep_layout=ep_layout)

    def allocate_persistent(
        self,
        input_batch: InputBatch,
        metadata: object,
    ) -> SpMetadata:
        del metadata
        self._validate_batch(input_batch, graph=True)
        return self._graph_layout(input_batch.num_tokens_after_padding)

    def _graph_layout(self, token_capacity: int) -> SpMetadata:
        bucket_counts = (token_capacity,) * self._dp_size
        ep_layout = EpSpTokenLayout.from_token_counts(
            bucket_counts,
            bucket_counts,
            tp_size=self._tp_size,
            tp_rank=self._tp_rank,
            dp_size=self._dp_size,
            dp_rank=self._dp_rank,
            compacted=False,
        )
        return SpMetadata.from_execution_rows(token_capacity, self._tp_size, self._tp_rank, ep_layout=ep_layout)

    def update_persistent(
        self,
        persistent_metadata: object,
        input_batch: InputBatch,
        metadata: object,
    ) -> None:
        del metadata
        if not isinstance(persistent_metadata, SpMetadata):
            raise TypeError("Qwen3.5 SP received invalid persistent metadata")
        self._validate_batch(input_batch, graph=True)
        token_capacity = input_batch.num_tokens_after_padding
        ep_layout = persistent_metadata.ep_layout
        expected = SpMetadata.from_execution_rows(token_capacity, self._tp_size, self._tp_rank, ep_layout=ep_layout)
        if (
            persistent_metadata != expected
            or ep_layout is None
            or ep_layout.compacted
            or ep_layout.variable
            or ep_layout.dp_size != self._dp_size
            or ep_layout.dispatch_counts != (expected.local_rows,) * (self._dp_size * self._tp_size)
            or ep_layout.active_rows != expected.valid_rows
            or ep_layout.dispatch_rows != expected.local_rows
            or ep_layout.ep_rank != self._dp_rank * self._tp_size + self._tp_rank
        ):
            raise ValueError("Qwen3.5 SP graph layout does not match its persistent capacity")

    @staticmethod
    def _host_counts(metadata: object, name: str) -> tuple[int, ...]:
        counts = getattr(metadata, name, ())
        if not isinstance(counts, Sequence) or any(not isinstance(count, Integral) for count in counts):
            raise TypeError(f"{name} must be a host integer sequence")
        return tuple(int(count) for count in counts)

    @staticmethod
    def _validate_batch(input_batch: InputBatch, *, graph: bool) -> None:
        token_capacity = input_batch.num_tokens_after_padding
        if input_batch.num_tokens < 0 or token_capacity < input_batch.num_tokens:
            raise ValueError("Qwen3.5 SP execution rows must fit the input capacity")
        if input_batch.input_ids.ndim != 1 or input_batch.input_ids.shape[0] != token_capacity:
            raise ValueError("Qwen3.5 SP input rows do not match the InputBatch capacity")
        if input_batch.positions.ndim not in (1, 2):
            raise ValueError("Qwen3.5 SP positions must be one- or two-dimensional")
        position_rows = (
            input_batch.positions.shape[0] if input_batch.positions.ndim == 1 else input_batch.positions.shape[-1]
        )
        if position_rows != token_capacity:
            raise ValueError("Qwen3.5 SP positions do not match the InputBatch capacity")
        if graph:
            if token_capacity <= 0:
                raise ValueError("Qwen3.5 SP graph token capacity must be positive")
            if input_batch.is_padding is None or input_batch.is_padding.shape != (token_capacity,):
                raise ValueError("Qwen3.5 SP graph execution requires a padding mask matching its capacity")
            if input_batch.is_padding.dtype != torch.bool:
                raise TypeError("Qwen3.5 SP graph padding mask must use bool")
            if input_batch.is_padding.device != input_batch.input_ids.device:
                raise ValueError("Qwen3.5 SP graph padding mask must be on the input device")
        elif input_batch.num_tokens_after_padding != input_batch.num_tokens or input_batch.is_padding is not None:
            raise ValueError("eager Qwen3.5 SP requires an unpadded InputBatch")


def sp_local_rows(num_tokens: int, tp_size: int) -> int:
    """Return the padded number of rows owned by one TP rank."""
    return SpMetadata.from_execution_rows(num_tokens, tp_size, 0).local_rows


def _pad_rows(tensor: torch.Tensor, rows: int) -> torch.Tensor:
    if tensor.shape[0] == rows:
        return tensor.contiguous()
    if tensor.shape[0] > rows:
        raise ValueError(f"cannot shrink {tensor.shape[0]} rows to sequence-parallel capacity {rows}")
    padded = tensor.new_zeros((rows, *tensor.shape[1:]))
    padded[: tensor.shape[0]].copy_(tensor)
    return padded


def _coerce_layout(
    layout_or_execution_rows: SpMetadata | int,
    tp_size: int | None,
    tp_rank: int | None,
) -> SpMetadata:
    if isinstance(layout_or_execution_rows, SpMetadata):
        if tp_size is not None or tp_rank is not None:
            raise TypeError("TP arguments cannot be combined with SpMetadata")
        return layout_or_execution_rows
    if tp_size is None or tp_rank is None:
        raise TypeError("sequence-parallel TP size and rank are required")
    return SpMetadata.from_execution_rows(layout_or_execution_rows, tp_size, tp_rank)


def sp_shard(
    tensor: torch.Tensor,
    layout_or_tp_size: SpMetadata | int | None = None,
    tp_rank: int | None = None,
    *,
    tp_size: int | None = None,
) -> torch.Tensor:
    """Split full token rows across TP ranks and zero-pad the tail.

    The integer form is retained for callers outside the model execution
    path. New code should pass a :class:`SpMetadata` instance.
    """
    if tensor.ndim == 0:
        raise ValueError("sequence-parallel tensors must have a token dimension")
    if layout_or_tp_size is None:
        if tp_size is None:
            raise TypeError("sequence-parallel TP size is required")
        layout_or_tp_size = tp_size
    elif tp_size is not None:
        raise TypeError("TP size cannot be provided twice")
    if isinstance(layout_or_tp_size, SpMetadata):
        layout = _coerce_layout(layout_or_tp_size, None, None)
    else:
        if tp_rank is None:
            raise TypeError("sequence-parallel TP rank is required")
        layout = SpMetadata.from_execution_rows(tensor.shape[0], layout_or_tp_size, tp_rank)
    if tensor.shape[0] > layout.execution_rows:
        raise ValueError(
            f"cannot shard {tensor.shape[0]} rows into sequence-parallel execution capacity {layout.execution_rows}"
        )
    padded = _pad_rows(tensor, layout.padded_rows)
    return padded.narrow(0, layout.local_offset, layout.local_rows).contiguous()


def sp_all_gather(
    local: torch.Tensor,
    layout_or_num_tokens: SpMetadata | int | None = None,
    tp_size: int | None = None,
    *,
    num_tokens: int | None = None,
) -> torch.Tensor:
    """Gather local token rows and trim the structural TP padding."""
    if local.ndim == 0:
        raise ValueError("sequence-parallel tensors must have a token dimension")
    if layout_or_num_tokens is None:
        if num_tokens is None:
            raise TypeError("sequence-parallel token count or metadata is required")
        layout_or_num_tokens = num_tokens
    elif num_tokens is not None:
        raise TypeError("token count cannot be provided twice")
    if isinstance(layout_or_num_tokens, SpMetadata):
        if tp_size is not None:
            raise TypeError("TP size cannot be combined with SpMetadata")
        layout = _coerce_layout(layout_or_num_tokens, None, None)
    else:
        if tp_size is None:
            raise TypeError("sequence-parallel TP size is required")
        layout = SpMetadata.from_execution_rows(layout_or_num_tokens, tp_size, 0)
    if local.shape[0] != layout.local_rows:
        raise ValueError(f"expected {layout.local_rows} local sequence-parallel rows, got {local.shape[0]}")
    if layout.tp_size == 1:
        return local[: layout.execution_rows].contiguous()
    gathered = distributed.tp_all_gather(local.contiguous(), 0, layout.tp_size)
    return gathered[: layout.execution_rows].contiguous()


def sp_reduce_scatter(
    full_partial: torch.Tensor,
    layout_or_tp_size: SpMetadata | int | None = None,
    *,
    tp_size: int | None = None,
) -> torch.Tensor:
    """Reduce full token rows across TP and return each rank's token shard."""
    if full_partial.ndim == 0:
        raise ValueError("sequence-parallel tensors must have a token dimension")
    if layout_or_tp_size is None:
        if tp_size is None:
            raise TypeError("sequence-parallel TP size or metadata is required")
        layout_or_tp_size = tp_size
    elif tp_size is not None:
        raise TypeError("TP size cannot be provided twice")
    if isinstance(layout_or_tp_size, SpMetadata):
        layout = _coerce_layout(layout_or_tp_size, None, None)
    else:
        layout = SpMetadata.from_execution_rows(full_partial.shape[0], layout_or_tp_size, 0)
    if full_partial.shape[0] > layout.execution_rows:
        raise ValueError(
            f"cannot reduce-scatter {full_partial.shape[0]} rows into sequence-parallel execution capacity "
            f"{layout.execution_rows}"
        )
    padded = _pad_rows(full_partial, layout.padded_rows)
    if layout.tp_size == 1:
        return padded[: layout.local_rows].contiguous()
    return distributed.reduce_scatter(padded, layout.tp_size, "tp")


def sp_row_parallel_projection(
    linear: RowParallelLinear,
    value: torch.Tensor,
    layout: SpMetadata,
) -> torch.Tensor:
    """Reduce a row-parallel projection to SP rows before adding its bias."""
    partial = torch.matmul(value, linear.weight) if linear._weight_is_transposed else F.linear(value, linear.weight)
    output = sp_reduce_scatter(partial, layout)
    if linear.bias is not None:
        output = output + linear.bias
    return output
