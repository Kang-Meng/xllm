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

"""Execution metadata for tensor-parallel sequence parallelism."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True, slots=True)
class EpSpTokenLayout:
    """EP-wide row layout supplementing the local shard in SpMetadata.

    ``active_rows`` excludes dummy rows in eager execution. Graph layouts
    retain bucket rows; per-step token validity is carried by feature masks.
    """

    dispatch_counts: tuple[int, ...]
    active_rows: int
    ep_rank: int
    dispatch_rows: int
    variable: bool
    compacted: bool
    dp_size: int

    @classmethod
    def from_token_counts(
        cls,
        execution_counts: Sequence[int],
        raw_counts: Sequence[int],
        *,
        tp_size: int,
        tp_rank: int,
        dp_size: int,
        dp_rank: int,
        compacted: bool,
    ) -> EpSpTokenLayout:
        if tp_size <= 0 or dp_size <= 0:
            raise ValueError("EP/SP parallel sizes must be positive")
        if not 0 <= tp_rank < tp_size or not 0 <= dp_rank < dp_size:
            raise ValueError("invalid EP/SP TP or DP coordinates")
        for counts, name in ((execution_counts, "DP execution"), (raw_counts, "raw DP execution")):
            if not isinstance(counts, Sequence) or any(not isinstance(count, Integral) for count in counts):
                raise TypeError(f"{name} token counts must be a host integer sequence")
            if len(counts) != dp_size:
                raise RuntimeError(f"expected {dp_size} {name} token counts, got {tuple(counts)}")
            if any(count < 0 for count in counts):
                raise RuntimeError(f"{name} token counts must be non-negative, got {tuple(counts)}")
        if any(raw > execution for raw, execution in zip(raw_counts, execution_counts)):
            raise RuntimeError("raw DP execution token counts must fit the execution rows")
        capacity_counts = tuple(
            max(1, (int(count) + tp_size - 1) // tp_size) for count in execution_counts for _ in range(tp_size)
        )
        valid_counts = tuple(
            min(capacity, max(0, int(raw_counts[rank // tp_size]) - (rank % tp_size) * capacity))
            for rank, capacity in enumerate(capacity_counts)
        )
        ep_rank = dp_rank * tp_size + tp_rank
        dispatch_counts = tuple(max(1, count) for count in valid_counts) if compacted else capacity_counts
        return cls(
            dispatch_counts=dispatch_counts,
            active_rows=valid_counts[ep_rank],
            ep_rank=ep_rank,
            dispatch_rows=max(dispatch_counts),
            variable=len(set(dispatch_counts)) != 1,
            compacted=compacted,
            dp_size=dp_size,
        )


@dataclass(frozen=True, slots=True)
class SpMetadata:
    """Execution-time row layout for one sequence-parallel TP shard.

    ``execution_rows`` is the full token capacity consumed by the model.  It
    includes ACL graph padding and scheduler dummy rows; request validity is
    represented separately by the global EP layout and feature-specific masks.
    """

    execution_rows: int
    tp_size: int
    tp_rank: int
    local_rows: int
    valid_rows: int
    local_offset: int
    padded_rows: int
    ep_layout: EpSpTokenLayout | None = None

    @classmethod
    def from_execution_rows(
        cls,
        execution_rows: int,
        tp_size: int,
        tp_rank: int,
        *,
        ep_layout: EpSpTokenLayout | None = None,
    ) -> SpMetadata:
        if execution_rows < 0:
            raise ValueError("sequence-parallel execution rows must be non-negative")
        if tp_size <= 0 or not 0 <= tp_rank < tp_size:
            raise ValueError("invalid sequence-parallel TP coordinates")
        local_rows = max(1, (execution_rows + tp_size - 1) // tp_size)
        local_offset = tp_rank * local_rows
        valid_rows = min(local_rows, max(0, execution_rows - local_offset))
        return cls(
            execution_rows=execution_rows,
            tp_size=tp_size,
            tp_rank=tp_rank,
            local_rows=local_rows,
            valid_rows=valid_rows,
            local_offset=local_offset,
            padded_rows=local_rows * tp_size,
            ep_layout=ep_layout,
        )
