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

"""Host-side MoE parallelism policy, separate from numerical kernel tests."""

import pytest
import torch

from xllm.python import distributed
from xllm.python.layers.moe_parallel import (
    Eplv2CommPolicy,
    ep_apply_local,
    ep_gather_tokens,
    ep_reduce_scatter,
    mc2_buffer_bytes_per_source_row,
)
from xllm.python.layers.sequence_parallel_metadata import EpSpTokenLayout, SpMetadata


def test_a3_window_bound_and_capacity() -> None:
    assert mc2_buffer_bytes_per_source_row(4096, 36, 8, 8) == 5160960
    policy = Eplv2CommPolicy.from_geometry(4096, 288, 8, 8, 1280)
    assert policy.mc2_capacity == 260
    assert policy.select(256) == "mc2"
    assert policy.select(261) == "alltoall"
    with pytest.raises(ValueError, match="Graph"):
        policy.select(261, graph=True)


@pytest.mark.parametrize("rows,expected", [(1, "mc2"), (512, "mc2"), (513, "alltoall"), (1024, "alltoall")])
def test_operator_token_limit_independent_of_phase(rows: int, expected: str) -> None:
    policy = Eplv2CommPolicy.from_geometry(4096, 288, 8, 8, 3072)
    assert policy.mc2_capacity == 512
    assert policy.select(rows) == expected


def test_explicit_modes_fail_before_execution_not_after_collectives() -> None:
    mc2 = Eplv2CommPolicy.from_geometry(4096, 288, 8, 8, 200, mode="mc2")
    with pytest.raises(ValueError, match="exceed capacity"):
        mc2.select(512)
    a2a = Eplv2CommPolicy.from_geometry(4096, 288, 8, 8, 200, mode="alltoall")
    assert a2a.select(1) == "alltoall"
    with pytest.raises(ValueError, match="eager"):
        a2a.select(1, graph=True)


@pytest.mark.parametrize("limit", [0, 513, -1])
def test_invalid_token_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="token limit"):
        Eplv2CommPolicy.from_geometry(4096, 288, 8, 8, 3072, token_limit=limit)


def test_invalid_geometry_never_enters_mc2() -> None:
    policy = Eplv2CommPolicy.from_geometry(16, 32, 2, 2, 3072)
    assert policy.select(1) == "alltoall"
    with pytest.raises(ValueError, match="source rows"):
        policy.select(0)


def _layout(
    execution_token_counts: tuple[int, ...],
    raw_token_counts: tuple[int, ...] | None = None,
    *,
    ep_rank: int = 0,
    graph: bool = False,
) -> SpMetadata:
    if raw_token_counts is None:
        raw_token_counts = execution_token_counts
    ep_layout = EpSpTokenLayout.from_token_counts(
        execution_token_counts,
        raw_token_counts,
        tp_size=2,
        tp_rank=ep_rank % 2,
        dp_size=2,
        dp_rank=ep_rank // 2,
        compacted=not graph,
    )
    return SpMetadata.from_execution_rows(execution_token_counts[ep_rank // 2], 2, ep_rank % 2, ep_layout=ep_layout)


def test_ep_gather_tokens_keeps_equal_capacity_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    hidden = torch.zeros(2, 4)
    gathered = torch.arange(8 * 4, dtype=torch.float32).view(8, 4)
    monkeypatch.setattr(distributed, "all_gather", lambda *args, **kwargs: gathered, raising=False)
    monkeypatch.setattr(
        distributed,
        "all_gather_variable",
        lambda *args, **kwargs: pytest.fail("equal EP/SP rows must use fixed all-gather"),
        raising=False,
    )

    monkeypatch.setattr(
        distributed,
        "reduce_scatter",
        lambda value, world_size, group_name: value[: value.shape[0] // world_size],
        raising=False,
    )
    layout = _layout((4, 4))
    full_hidden = ep_gather_tokens(hidden, layout)
    output = ep_reduce_scatter(full_hidden, layout)

    assert full_hidden is gathered
    assert output.shape == (2, 4)


def test_ep_gather_tokens_pads_unequal_capacities_before_reduce_scatter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden = torch.zeros(3, 4)
    gathered = torch.arange(8 * 4, dtype=torch.float32).view(8, 4)
    gather_counts: list[int] = []

    def all_gather_variable(value: torch.Tensor, counts: list[int], rank: int, group_name: str) -> torch.Tensor:
        gather_counts.extend(counts)
        assert rank == 2 and group_name == "moe_ep"
        torch.testing.assert_close(value, hidden)
        return gathered

    monkeypatch.setattr(
        distributed,
        "all_gather_variable",
        all_gather_variable,
        raising=False,
    )
    monkeypatch.setattr(
        distributed,
        "all_gather",
        lambda *args, **kwargs: pytest.fail("unequal EP/SP rows must use variable all-gather"),
        raising=False,
    )
    reduce_inputs: list[torch.Tensor] = []

    def reduce_scatter(value: torch.Tensor, world_size: int, group_name: str) -> torch.Tensor:
        reduce_inputs.append(value)
        return value[2 * 3 : 3 * 3]

    monkeypatch.setattr(distributed, "reduce_scatter", reduce_scatter, raising=False)
    layout = _layout((3, 5), ep_rank=2)
    full_hidden = ep_gather_tokens(hidden, layout)
    output = ep_reduce_scatter(full_hidden, layout)

    assert full_hidden is gathered
    assert gather_counts == [2, 1, 3, 2]
    assert output.shape == (3, 4)
    expected = torch.zeros(12, 4)
    expected[:2] = gathered[:2]
    expected[3] = gathered[2]
    expected[6:9] = gathered[3:6]
    expected[9:11] = gathered[6:8]
    torch.testing.assert_close(reduce_inputs[0], expected)


@pytest.mark.parametrize(("raw_rows", "graph"), [(3, False), (0, False), (3, True), (0, True)])
def test_ep_layout_applies_local_valid_prefix(raw_rows: int, graph: bool) -> None:
    hidden = torch.arange(2 * 4, dtype=torch.float32).view(2, 4)
    computed: list[torch.Tensor] = []

    def apply(source: torch.Tensor) -> torch.Tensor:
        computed.append(source)
        return source + 10

    layout = _layout((4, 8), (raw_rows, 5), ep_rank=1, graph=graph)
    restored = ep_apply_local(hidden, layout, apply)

    valid_rows = 2 if graph else max(0, raw_rows - 2)
    expected = torch.zeros_like(hidden)
    expected[:valid_rows] = hidden[:valid_rows] + 10
    torch.testing.assert_close(restored, expected)
    assert len(computed) == int(valid_rows > 0)
    if computed:
        torch.testing.assert_close(computed[0], hidden[:valid_rows])


def test_ep_gather_tokens_keeps_capacity_in_acl_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    hidden = torch.zeros(2, 4)
    gathered = torch.arange(12 * 4, dtype=torch.float32).view(12, 4)
    gather_counts: list[int] = []

    def all_gather_variable(value: torch.Tensor, counts: list[int], rank: int, group_name: str) -> torch.Tensor:
        gather_counts.extend(counts)
        assert value is hidden
        return gathered

    monkeypatch.setattr(
        distributed,
        "all_gather_variable",
        all_gather_variable,
        raising=False,
    )
    monkeypatch.setattr(
        distributed,
        "reduce_scatter",
        lambda value, world_size, group_name: value[: value.shape[0] // world_size],
        raising=False,
    )
    layout = _layout((4, 8), (3, 5), graph=True)
    full_hidden = ep_gather_tokens(hidden, layout)
    output = ep_reduce_scatter(full_hidden, layout)

    assert gather_counts == [2, 2, 4, 4]
    assert output.shape == (2, 4)


@pytest.mark.parametrize(
    ("execution_counts", "raw_counts", "capacity_counts", "valid_counts", "graph"),
    [
        ((3, 5), (3, 5), (2, 2, 3, 3), (2, 1, 3, 2), False),
        ((4, 8), (3, 5), (2, 2, 4, 4), (2, 1, 4, 1), False),
        ((4, 4), (4, 4), (2, 2, 2, 2), (2, 2, 2, 2), False),
        ((1, 5), (0, 5), (1, 1, 3, 3), (0, 0, 3, 2), False),
        ((0, 0), (0, 0), (1, 1, 1, 1), (0, 0, 0, 0), False),
        ((8, 8), (1, 1), (4, 4, 4, 4), (1, 0, 1, 0), False),
        ((4, 8), (3, 5), (2, 2, 4, 4), (2, 1, 4, 1), True),
        ((4, 4), (1, 3), (2, 2, 2, 2), (1, 0, 2, 1), True),
    ],
)
def test_ep_sp_transport_restores_each_source_rank(
    monkeypatch: pytest.MonkeyPatch,
    execution_counts: tuple[int, ...],
    raw_counts: tuple[int, ...],
    capacity_counts: tuple[int, ...],
    valid_counts: tuple[int, ...],
    graph: bool,
) -> None:
    inputs = [torch.full((capacity, 4), -99.0) for capacity in capacity_counts]
    for rank, valid in enumerate(valid_counts):
        inputs[rank][:valid] = torch.arange(valid * 4).view(valid, 4) + 10 * rank
    sources = [
        hidden if graph else hidden[:valid] if valid else hidden.new_zeros((1, 4))
        for hidden, valid in zip(inputs, valid_counts)
    ]
    counts = [source.shape[0] for source in sources]
    gathered = torch.cat(sources)
    active_rank = 0

    def all_gather(value: torch.Tensor, dim: int, world_size: int, group_name: str) -> torch.Tensor:
        assert dim == 0 and world_size == 4 and group_name == "moe_ep"
        assert len(set(counts)) == 1
        torch.testing.assert_close(value, sources[active_rank])
        return gathered

    def all_gather_variable(value: torch.Tensor, token_counts: list[int], rank: int, group_name: str) -> torch.Tensor:
        assert token_counts == counts and rank == active_rank and group_name == "moe_ep"
        torch.testing.assert_close(value, sources[rank])
        return gathered

    def reduce_scatter(value: torch.Tensor, world_size: int, group_name: str) -> torch.Tensor:
        assert world_size == 4 and group_name == "moe_ep"
        # Simulate four EP experts contributing identical partials; the local
        # result must match a SUM reduction of the original source token rows.
        return value.chunk(world_size)[active_rank] * world_size

    monkeypatch.setattr(distributed, "all_gather", all_gather, raising=False)
    monkeypatch.setattr(distributed, "all_gather_variable", all_gather_variable, raising=False)
    monkeypatch.setattr(distributed, "reduce_scatter", reduce_scatter, raising=False)
    for active_rank, hidden in enumerate(inputs):
        layout = _layout(execution_counts, raw_counts, ep_rank=active_rank, graph=graph)
        full_hidden = ep_gather_tokens(hidden, layout)
        output = ep_reduce_scatter(full_hidden * 2 + 1, layout)
        expected = torch.zeros_like(hidden)
        rows = hidden.shape[0] if graph else valid_counts[active_rank]
        expected[:rows] = (hidden[:rows] * 2 + 1) * 4
        torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    ("execution_counts", "raw_counts", "error", "message"),
    [
        ((4,), (4,), RuntimeError, "expected 2 DP execution"),
        ((4, 4), (4,), RuntimeError, "expected 2 raw DP"),
        ((-1, 4), (0, 4), RuntimeError, "non-negative"),
        ((4, 4), (-1, 4), RuntimeError, "non-negative"),
        ((4, 4), (5, 4), RuntimeError, "fit the execution"),
    ],
)
def test_ep_sp_rejects_invalid_counts_before_communication(
    monkeypatch: pytest.MonkeyPatch,
    execution_counts: tuple[int, ...],
    raw_counts: tuple[int, ...],
    error: type[Exception],
    message: str,
) -> None:
    def unexpected_collective(*args: object, **kwargs: object) -> torch.Tensor:
        pytest.fail("invalid row metadata must be rejected before collectives")

    monkeypatch.setattr(distributed, "all_gather", unexpected_collective, raising=False)
    monkeypatch.setattr(distributed, "all_gather_variable", unexpected_collective, raising=False)
    with pytest.raises(error, match=message):
        ep_gather_tokens(torch.zeros(2, 4), _layout(execution_counts, raw_counts))


@pytest.mark.parametrize(
    ("coordinates", "message"),
    [
        ({"tp_size": 0}, "parallel sizes"),
        ({"dp_size": 0}, "parallel sizes"),
        ({"tp_rank": 2}, "TP or DP coordinates"),
        ({"dp_rank": -1}, "TP or DP coordinates"),
    ],
)
def test_ep_layout_factory_rejects_invalid_coordinates(coordinates: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EpSpTokenLayout.from_token_counts(
            (4, 4),
            (4, 4),
            **{"tp_size": 2, "tp_rank": 0, "dp_size": 2, "dp_rank": 0, **coordinates},
            compacted=True,
        )


@pytest.mark.parametrize("rows", [0, 7, 9])
def test_ep_sp_rejects_wrong_expert_output_rows(monkeypatch: pytest.MonkeyPatch, rows: int) -> None:
    monkeypatch.setattr(distributed, "all_gather", lambda *args, **kwargs: torch.zeros(8, 4), raising=False)
    layout = _layout((4, 4))
    ep_gather_tokens(torch.zeros(2, 4), layout)
    with pytest.raises(ValueError, match="output rows"):
        ep_reduce_scatter(torch.zeros(rows, 4), layout)


@pytest.mark.parametrize("graph", [False, True])
def test_ep_layout_can_be_prepared_without_forward_context(graph: bool) -> None:
    ep_layout = EpSpTokenLayout.from_token_counts(
        (4, 8), (3, 5), tp_size=2, tp_rank=0, dp_size=2, dp_rank=0, compacted=not graph
    )
    assert ep_layout.dispatch_counts == ((2, 2, 4, 4) if graph else (2, 1, 4, 1))
    layout = SpMetadata.from_execution_rows(4, 2, 0, ep_layout=ep_layout)
    assert layout.ep_layout is ep_layout
    assert layout.local_rows == 2
    assert layout.tp_size == 2 and layout.tp_rank == 0
    assert ep_layout.dp_size == 2 and ep_layout.ep_rank == 0
    assert ep_layout.compacted is not graph
