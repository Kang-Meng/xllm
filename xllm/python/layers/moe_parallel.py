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

"""Host-side token transport shared by the sparse-MoE forward paths.

Three cooperating concerns live here because every routed-MoE forward needs
some combination of them, and splitting them across sibling modules hid that
relationship:

* Data-parallel gather/scatter (:class:`DpScatterState`, :func:`dp_gather_tokens`,
  :func:`reduce_and_scatter`). Dense ``FusedMoE``, NPU Qwen3.5 and the DeepSeek
  family run the experts over the whole DP group's tokens, then slice the result
  back to the rank-local rows. That gather prologue and scatter epilogue are
  identical across backends; only the expert compute between them differs.
* EP/SP token operations (:func:`ep_gather_tokens`, :func:`ep_reduce_scatter`).
  Consume builder-prepared SP metadata after the expert input is validated.
  Qwen3.5 direct EP gathers valid prefixes in eager execution, retains graph
  capacities during capture, and restores the local SP capacity after reduction.
* EPLv2 communication admission (:class:`Eplv2CommPolicy`,
  :func:`mc2_buffer_bytes_per_source_row`). Host-only selection between MegaMoE,
  MC2 and All-to-AllV for the MoE-TP1 A3 EPLv2 path; no device queries in the selector.
* TP/SP source-token ownership (:class:`TokenParallelLayout`). Unique
  source-token ownership inside an attention-TP MoE boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import cast

import torch

from xllm.python import distributed
from xllm.python.layers.sequence_parallel_metadata import EpSpTokenLayout, SpMetadata
from xllm.python.model_executor.forward_context import get_forward_context

# ---------------------------------------------------------------------------
# Data-parallel token gather/scatter
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DpScatterState:
    """Slices a DP-gathered MoE output back to this rank's local tokens."""

    output_offset: int
    local_tokens: int
    enabled: bool

    def scatter(self, output: torch.Tensor) -> torch.Tensor:
        """Return this rank's output rows, or the unchanged output for DP1."""
        if self.enabled:
            return output.narrow(0, self.output_offset, self.local_tokens)
        return output


# Shared no-op state for the single-DP path: its scatter returns the output
# unchanged, so one frozen instance is reused instead of allocating per forward.
_NO_SCATTER = DpScatterState(0, 0, False)


def dp_gather_tokens(
    hidden: torch.Tensor,
    dp_size: int,
    dp_rank: int,
) -> tuple[torch.Tensor, DpScatterState]:
    """All-gather this rank's tokens across the DP group for expert compute.

    Returns the gathered hidden states and the :class:`DpScatterState` that
    slices the computed output back to the local execution rows. Metadata
    already accounts for the dummy row materialized by an empty DP rank. The
    collective is fixed exactly when every rank executes the same row count;
    otherwise it is variable. ``dp_size <= 1`` is a no-op whose ``scatter``
    returns the output unchanged.
    """
    if dp_size <= 1:
        return hidden, _NO_SCATTER
    ctx = get_forward_context()
    execution_token_counts = tuple(ctx.metadata.dp_execution_token_counts)
    if len(execution_token_counts) != dp_size:
        raise RuntimeError(f"expected {dp_size} DP execution token counts, got {execution_token_counts}")
    local_tokens = hidden.shape[0]
    gathered, output_offset = distributed.gather_dp_execution_tokens(
        hidden,
        execution_token_counts,
        dp_rank,
    )
    return gathered, DpScatterState(output_offset, local_tokens, True)


def _ep_source_tokens(hidden: torch.Tensor, metadata: SpMetadata) -> torch.Tensor:
    layout = cast(EpSpTokenLayout, metadata.ep_layout)
    if not layout.compacted or layout.active_rows == metadata.local_rows:
        return hidden
    if layout.active_rows == 0:
        return hidden.new_zeros((1, *hidden.shape[1:]))
    return hidden.narrow(0, 0, layout.active_rows)


def _ep_restore_tokens(output: torch.Tensor, metadata: SpMetadata) -> torch.Tensor:
    layout = cast(EpSpTokenLayout, metadata.ep_layout)
    expected_rows = max(1, layout.active_rows) if layout.compacted else metadata.local_rows
    if output.ndim == 0 or output.shape[0] != expected_rows:
        raise ValueError(
            "SP compacted output rows do not match the valid layout: "
            f"expected {expected_rows}, got {output.shape[0] if output.ndim else 0}"
        )
    if not layout.compacted or layout.active_rows == metadata.local_rows:
        return output
    restored = output.new_zeros((metadata.local_rows, *output.shape[1:]))
    if layout.active_rows:
        restored[: layout.active_rows].copy_(output[: layout.active_rows])
    return restored


def ep_gather_tokens(hidden: torch.Tensor, metadata: SpMetadata) -> torch.Tensor:
    """Gather valid SP prefixes across EP, retaining graph capacities."""
    layout = cast(EpSpTokenLayout, metadata.ep_layout)
    source = _ep_source_tokens(hidden, metadata)
    if not layout.variable:
        return distributed.all_gather(source, dim=0, world_size=len(layout.dispatch_counts), group_name="moe_ep")
    return distributed.all_gather_variable(source, list(layout.dispatch_counts), layout.ep_rank, "moe_ep")


def ep_reduce_scatter(output: torch.Tensor, metadata: SpMetadata) -> torch.Tensor:
    """Sum EP expert contributions and restore this rank's SP capacity."""
    layout = cast(EpSpTokenLayout, metadata.ep_layout)
    expected_rows = sum(layout.dispatch_counts)
    if output.ndim == 0 or output.shape[0] != expected_rows:
        raise ValueError(
            "EP/SP MoE output rows do not match the gathered layout: "
            f"expected {expected_rows}, got {output.shape[0] if output.ndim else 0}"
        )
    if not layout.variable:
        reduced = distributed.reduce_scatter(output, len(layout.dispatch_counts), "moe_ep")
    else:
        padded = output.new_zeros((layout.dispatch_rows * len(layout.dispatch_counts), *output.shape[1:]))
        source_offset = 0
        for rank, row_count in enumerate(layout.dispatch_counts):
            source = output.narrow(0, source_offset, row_count)
            padded.narrow(0, rank * layout.dispatch_rows, row_count).copy_(source)
            source_offset += row_count
        reduced = distributed.reduce_scatter(padded, len(layout.dispatch_counts), "moe_ep")
        reduced = reduced[: layout.dispatch_counts[layout.ep_rank]].contiguous()
    return _ep_restore_tokens(reduced, metadata)


def ep_apply_local(
    hidden: torch.Tensor,
    metadata: SpMetadata,
    operation: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Run local expert work on valid rows and restore the SP capacity."""
    layout = cast(EpSpTokenLayout, metadata.ep_layout)
    source = _ep_source_tokens(hidden, metadata)
    if layout.compacted and layout.active_rows == 0:
        return torch.zeros_like(hidden)
    return _ep_restore_tokens(operation(source), metadata)


def reduce_and_scatter(
    output: torch.Tensor,
    scatter_state: DpScatterState,
    *,
    reduce_results: bool,
    moe_tp_size: int,
    ep_size: int,
) -> torch.Tensor:
    """Reduce a routed-MoE output across its parallel axes, then slice back to local tokens.

    The reduce-then-scatter epilogue shared by the routed-MoE forwards (dense
    ``FusedMoE``, NPU Qwen3.5 experts): when ``reduce_results``, TP-reduce across the
    MoE tensor-parallel group and EP-reduce across the expert-parallel group, then
    slice the DP-gathered rows back to this rank via ``scatter_state``.
    """
    if reduce_results:
        if moe_tp_size > 1:
            distributed.moe_tp_all_reduce(output)
        if ep_size > 1:
            distributed.moe_ep_all_reduce(output)
    return scatter_state.scatter(output)


# ---------------------------------------------------------------------------
# EPLv2 communication admission
# ---------------------------------------------------------------------------


def mc2_buffer_bytes_per_source_row(hidden_size: int, local_experts: int, ep_size: int, topk: int) -> int:
    """Estimate MC2 window bytes per source row, including the shared slot.

    CANN specifies the window layout using two-byte activations even when
    dispatch quantization is enabled. Traffic compression does not halve this
    allocation requirement.
    """
    dispatch_row = ((2 * hidden_size + 64 + 511) // 512) * 512
    combine_row = ((2 * hidden_size + 511) // 512) * 512
    return 2 * (local_experts * ep_size * dispatch_row + (topk + 1) * combine_row)


@dataclass(frozen=True)
class Eplv2CommPolicy:
    """Precompute the enabled backend; over-capacity batches use eager All-to-AllV."""

    mc2_capacity: int
    mega_moe_capacity: int = 0
    graph_token_limit: int = field(init=False)
    _routed_backend: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mc2_capacity < 0 or self.mega_moe_capacity < 0:
            raise ValueError("EPLv2 communication capacities must be nonnegative")
        if self.mega_moe_capacity > 0:
            backend, capacity = "megamoe", self.mega_moe_capacity
        else:
            backend, capacity = "mc2", self.mc2_capacity
        object.__setattr__(self, "_routed_backend", backend)
        object.__setattr__(self, "graph_token_limit", capacity)

    @classmethod
    def from_geometry(
        cls,
        hidden_size: int,
        num_experts: int,
        ep_size: int,
        topk: int,
        buffer_mb: int,
        token_limit: int = 512,
        *,
        mega_moe_capacity: int = 0,
    ) -> Eplv2CommPolicy:
        """Validate EPLv2 geometry and calculate the maximum MC2 source rows."""
        if buffer_mb <= 0 or not 0 < token_limit <= 512:
            raise ValueError("EPLv2 requires positive HCCL buffer MB and a MC2 token limit in [1, 512]")
        if ep_size <= 1 or num_experts <= 0 or num_experts % ep_size or not 0 < topk <= num_experts:
            raise ValueError("Invalid EPLv2 expert geometry")
        row_bytes = mc2_buffer_bytes_per_source_row(hidden_size, num_experts // ep_size, ep_size, topk)
        capacity = min(token_limit, buffer_mb * 1024 * 1024 // row_bytes)
        if not 1024 <= hidden_size <= 8192 or topk > 16:
            capacity = 0
        return cls(capacity, mega_moe_capacity=mega_moe_capacity)

    def select(self, source_rows: int, *, graph: bool = False) -> str:
        """Select a transport; reject Graph execution when All-to-AllV is required."""
        if source_rows <= 0:
            raise ValueError("EPLv2 source rows must include at least one padded row")
        if source_rows <= self.graph_token_limit:
            return self._routed_backend
        if graph:
            capacity_name = "MegaMoE" if self._routed_backend == "megamoe" else "MC2"
            raise ValueError(
                f"EPLv2 All-to-AllV requires eager execution; Graph rows must fit {capacity_name} capacity"
            )
        return "alltoall"


# ---------------------------------------------------------------------------
# TP/SP source-token ownership
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TokenParallelLayout:
    """Own a TP shard of source rows and pad EP dispatch to a common capacity."""

    num_tokens: int
    tp_size: int
    tp_rank: int
    routed_backend: str | None = None
    dispatch_rows: int | None = None

    def __post_init__(self) -> None:
        """Reject invalid shard coordinates or insufficient dispatch capacity."""
        if self.num_tokens < 0 or self.tp_size <= 0 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("Invalid token-parallel layout")
        if self.dispatch_rows is not None and self.dispatch_rows < self.shard_tokens:
            raise ValueError("EP dispatch capacity must cover the local SP rows")

    @classmethod
    def from_dp(
        cls,
        num_tokens: int,
        tp_size: int,
        tp_rank: int,
        dp_size: int,
        dp_rank: int,
        execution_counts: Sequence[int] | None,
    ) -> TokenParallelLayout:
        """Keep TP-local ownership while matching EP dispatch rows across DP ranks."""
        if dp_size <= 0 or not 0 <= dp_rank < dp_size:
            raise ValueError("Invalid data-parallel layout")
        if dp_size == 1:
            return cls(num_tokens, tp_size, tp_rank)
        if isinstance(execution_counts, torch.Tensor):
            raise TypeError("DP execution counts must be host metadata, not a device tensor")
        if execution_counts is None or len(execution_counts) != dp_size:
            raise ValueError("DP EPLv2 requires execution counts for every DP rank")
        counts = tuple(execution_counts)
        if any(not isinstance(count, int) or count <= 0 for count in counts):
            raise ValueError("DP execution counts must include positive materialized dummy rows")
        if counts[dp_rank] != num_tokens:
            raise ValueError("DP execution count does not match local token rows")
        local = cls(num_tokens, tp_size, tp_rank)
        return cls(
            num_tokens, tp_size, tp_rank, dispatch_rows=max(local.shard_tokens, (max(counts) + tp_size - 1) // tp_size)
        )

    @property
    def shard_tokens(self) -> int:
        """Return the padded number of source rows per TP rank (at least one)."""
        return max(1, (self.num_tokens + self.tp_size - 1) // self.tp_size)

    @property
    def valid_tokens(self) -> int:
        """Structural rows owned by this rank, before the runtime activity mask."""
        return max(0, min(self.shard_tokens, self.num_tokens - self.tp_rank * self.shard_tokens))

    @property
    def dispatch_tokens(self) -> int:
        """Return the common EP dispatch capacity, or this rank's TP capacity."""
        return self.shard_tokens if self.dispatch_rows is None else self.dispatch_rows

    def pad_dispatch(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pad a local TP shard to the rows required by EP dispatch."""
        if tensor.ndim == 0 or tensor.shape[0] != self.shard_tokens:
            raise ValueError("Dispatch input must contain this rank's SP rows")
        return self._pad(tensor, self.dispatch_tokens)

    def _pad(self, tensor: torch.Tensor, rows: int) -> torch.Tensor:
        """Copy tensor rows and zero-fill the remaining capacity."""
        if tensor.shape[0] == rows:
            return tensor.contiguous()
        padded = tensor.new_empty((rows, *tensor.shape[1:]))
        padded[: tensor.shape[0]].copy_(tensor)
        padded[tensor.shape[0] :].zero_()
        return padded

    def shard(self, tensor: torch.Tensor) -> torch.Tensor:
        """Extract this rank's source-token rows and zero-pad its TP tail."""
        if tensor.dim() == 0 or tensor.shape[0] != self.num_tokens:
            raise ValueError("Token tensor does not match the full layout")
        start = min(self.tp_rank * self.shard_tokens, self.num_tokens)
        count = min(self.shard_tokens, self.num_tokens - start)
        local = tensor.narrow(0, start, count)
        return self._pad(local, self.shard_tokens)

    def reduce_scatter(self, partial: torch.Tensor, group_name: str = "tp") -> torch.Tensor:
        """Sum full-hidden TP/EP partials and return only this rank's token rows."""
        if partial.dim() == 0 or partial.shape[0] != self.num_tokens:
            raise ValueError("Partial tensor does not match the full token layout")
        padded = self._pad(partial, self.tp_size * self.shard_tokens)
        if self.tp_size == 1:
            return padded
        return distributed.reduce_scatter(padded, self.tp_size, group_name)

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Collect TP-local rows and trim the padded tail to source-token count."""
        if tensor.dim() == 0 or tensor.shape[0] != self.shard_tokens:
            raise ValueError("Token tensor does not match the local layout")
        if self.tp_size == 1:
            return tensor[: self.num_tokens]
        gathered = distributed.tp_all_gather(tensor.contiguous(), 0, self.tp_size)
        return gathered[: self.num_tokens]
