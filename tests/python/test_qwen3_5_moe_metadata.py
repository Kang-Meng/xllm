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

"""CPU behavior tests for Qwen3.5 SP dispatch validity and Graph lifecycle."""

from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from xllm.python.layers.npu.qwen3_5.moe import _NpuQwen3_5Experts
from xllm.python.layers.npu.qwen3_5.moe_metadata import MEGA_MOE_MAX_TOKENS, Qwen3_5MoeMetadata
from xllm.python.layers.npu.qwen3_5.moe_metadata_builder import Qwen3_5MoeMetadataBuilder
from xllm.python.layers.sequence_parallel import SpMetadataBuilder
from xllm.python.layers.sequence_parallel_metadata import EpSpTokenLayout, SpMetadata
from xllm.python.model_executor.forward_context import ForwardContext, forward_context, get_execution_context
from xllm.python.model_executor.input_batch import InputBatch


def _batch(token_count: int, *, dummy: bool = False, bucket: int | None = None) -> InputBatch:
    scheduled = () if dummy or not token_count else (token_count,)
    batch = InputBatch.from_runtime(
        torch.arange(token_count, dtype=torch.int32),
        torch.arange(token_count, dtype=torch.int32),
        SimpleNamespace(
            num_reqs=len(scheduled),
            num_tokens=token_count,
            num_scheduled_tokens=scheduled,
            num_computed_tokens=(0,) * len(scheduled),
            query_start_loc=(0, token_count) if scheduled else (0,),
            is_prefilling=(False,) * len(scheduled),
        ),
        is_dummy=dummy,
    )
    if bucket is None:
        return batch
    return batch.bind_graph_inputs(
        torch.zeros(bucket, dtype=torch.int32),
        torch.zeros(bucket, dtype=torch.int32),
        torch.arange(bucket) >= token_count,
    )


def _metadata(counts: tuple[int, ...], raw: tuple[int, ...] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        dp_execution_token_counts=counts,
        raw_dp_execution_token_counts=counts if raw is None else raw,
    )


def _config(**overrides: object) -> dict[str, object]:
    config = dict(
        device="privateuseone:0",
        num_experts=16,
        enable_mega_moe=True,
        sequence_parallel_enabled=True,
        tp_size=2,
        tp_rank=0,
        dp_size=2,
        dp_rank=0,
        ep_size=4,
        moe_tp_size=1,
        cp_size=1,
        num_speculative_tokens=0,
    )
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    ("counts", "raw_counts"),
    [((5, 1, 8), (5, 1, 8)), ((5, 1, 8), (5, 0, 8)), ((1, 1), (0, 0)), ((0, 0), (0, 0))],
)
def test_eager_dispatches_every_real_token_exactly_once(counts: tuple[int, ...], raw_counts: tuple[int, ...]) -> None:
    tp_size = 4
    expected_capacity = max(1, (max(counts) + tp_size - 1) // tp_size)
    expected_counts = []
    for rows, real_rows in zip(counts, raw_counts):
        capacity = max(1, (rows + tp_size - 1) // tp_size)
        expected_counts.extend(max(1, min(capacity, max(0, real_rows - rank * capacity))) for rank in range(tp_size))
    ownership: list[tuple[int, int]] = []
    for dp_rank, (rows, real_rows) in enumerate(zip(counts, raw_counts)):
        shard_capacity = max(1, (rows + tp_size - 1) // tp_size)
        for tp_rank in range(tp_size):
            builder = Qwen3_5MoeMetadataBuilder(tp_size, tp_rank, len(counts), dp_rank)
            result = builder.build(_batch(rows, dummy=real_rows == 0), _metadata(counts, raw_counts))
            mask = result.active_token_mask
            sp_metadata = SpMetadataBuilder(tp_size, tp_rank, len(counts), dp_rank).build(
                _batch(rows, dummy=real_rows == 0), _metadata(counts, raw_counts)
            )
            layout = sp_metadata.ep_layout
            assert layout.dispatch_counts == tuple(expected_counts)
            assert sp_metadata.local_rows == shard_capacity
            assert layout.active_rows == min(shard_capacity, max(0, real_rows - tp_rank * shard_capacity))
            assert layout.ep_rank == dp_rank * tp_size + tp_rank
            assert layout.compacted
            assert mask.shape == (expected_capacity,)
            assert mask.dtype == torch.int8
            assert mask.device.type == "cpu"
            ownership.extend(
                (dp_rank, tp_rank * shard_capacity + row) for row, active in enumerate(mask.tolist()) if active
            )
    assert sorted(ownership) == [(rank, row) for rank, count in enumerate(raw_counts) for row in range(count)]


def test_global_ep_layout_belongs_to_sp_metadata_not_moe_metadata() -> None:
    assert tuple(field.name for field in fields(Qwen3_5MoeMetadata)) == ("active_token_mask", "active_rows")
    assert "ep_layout" in tuple(field.name for field in fields(SpMetadata))
    assert tuple(field.name for field in fields(EpSpTokenLayout)) == (
        "dispatch_counts",
        "active_rows",
        "ep_rank",
        "dispatch_rows",
        "variable",
        "compacted",
        "dp_size",
    )


def test_sp_builder_owns_global_ep_layout() -> None:
    builder = SpMetadataBuilder.from_config(
        {
            "device": "privateuseone",
            "sequence_parallel_enabled": True,
            "tp_size": 2,
            "tp_rank": 1,
            "dp_size": 2,
            "dp_rank": 0,
        }
    )
    result = builder.build(_batch(3), _metadata((3, 5)))
    assert result.ep_layout.dispatch_counts == (2, 1, 3, 2)
    assert result.local_rows == 2
    assert result.ep_layout.active_rows == 1
    assert result.ep_layout.ep_rank == 1
    assert result.ep_layout.compacted
    assert not any(
        hasattr(result.ep_layout, name) for name in ("gather", "reduce_scatter", "apply_local", "from_context")
    )


def test_eager_layout_is_built_before_forward_and_shared_by_layers() -> None:
    builder = SpMetadataBuilder(tp_size=2, tp_rank=1, dp_size=2, dp_rank=0)
    first = builder.build(_batch(3), _metadata((3, 5)))
    context = ForwardContext(None, torch.device("cpu"), None, [], execution_contexts={SpMetadata: first})
    with forward_context(context):
        for _ in range(3):
            assert get_execution_context(SpMetadata).ep_layout is first.ep_layout
        assert context.layer_shared_cache == {}
    second = builder.build(_batch(1, dummy=True), _metadata((1, 2), (0, 2)))
    assert second.ep_layout is not first.ep_layout
    assert second.ep_layout.dispatch_counts == (1, 1, 1, 1)
    assert second.local_rows == 1
    assert second.valid_rows == 0
    assert second.ep_layout.active_rows == 0
    assert first.ep_layout.dispatch_counts == (2, 1, 3, 2)
    assert first.ep_layout.active_rows == 1


def test_sp_execution_rows_and_ep_active_rows_differ_for_dummy_input() -> None:
    metadata = SpMetadataBuilder(tp_size=2, tp_rank=0).build(_batch(1, dummy=True), SimpleNamespace())
    assert metadata.local_rows == 1
    assert metadata.valid_rows == 1
    assert metadata.ep_layout.active_rows == 0


def test_above_operator_limit_still_builds_validity_for_eplv1() -> None:
    local_capacity = MEGA_MOE_MAX_TOKENS + 1
    token_count = 2 * local_capacity
    builder = Qwen3_5MoeMetadataBuilder(tp_size=2, tp_rank=1, dp_size=2, dp_rank=0)
    result = builder.build(_batch(token_count), _metadata((token_count, 1), (token_count, 0)))

    assert result.active_token_mask.shape == (local_capacity,)
    assert result.active_token_mask.tolist() == [1] * local_capacity


@pytest.mark.parametrize("tp_rank", range(4))
def test_graph_masks_stay_stable_across_active_dummy_and_active_steps(
    tp_rank: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = Qwen3_5MoeMetadataBuilder(tp_size=4, tp_rank=tp_rank, dp_size=2, dp_rank=0)
    first_batch = _batch(3, bucket=5)
    persistent = builder.allocate_persistent(first_batch, _metadata((3, 4)))
    sp_builder = SpMetadataBuilder(tp_size=4, tp_rank=tp_rank, dp_size=2, dp_rank=0)
    sp_metadata = sp_builder.allocate_persistent(first_batch, _metadata((3, 4)))
    mask = persistent.active_token_mask
    data_ptr = mask.data_ptr()
    layout = sp_metadata.ep_layout
    assert layout.dispatch_counts == (2,) * 8
    assert sp_metadata.local_rows == 2
    assert layout.ep_rank == tp_rank
    assert not layout.compacted
    assert mask.tolist() == [0, 0]

    def unexpected_derivation(*args: object, **kwargs: object) -> None:
        pytest.fail("graph replay updates must not rebuild the fixed EP token layout")

    monkeypatch.setattr(EpSpTokenLayout, "from_token_counts", unexpected_derivation)

    for rows, real_rows in ((3, 3), (1, 0), (5, 5)):
        builder.update_persistent(
            persistent,
            _batch(rows, dummy=real_rows == 0, bucket=5),
            _metadata((rows, 4), (real_rows, 4)),
        )
        sp_builder.update_persistent(
            sp_metadata, _batch(rows, dummy=real_rows == 0, bucket=5), _metadata((rows, 4), (real_rows, 4))
        )
        assert persistent.active_token_mask is mask
        assert mask.data_ptr() == data_ptr
        assert sp_metadata.ep_layout is layout
        assert mask.tolist() == [int(tp_rank * 2 + row < real_rows) for row in range(2)]


def test_graph_uses_bucket_boundaries_and_honors_padding() -> None:
    builder = Qwen3_5MoeMetadataBuilder(tp_size=2, tp_rank=1, dp_size=2, dp_rank=0)
    batch = _batch(6, bucket=7)
    batch.is_padding[5] = True
    metadata = _metadata((6, 7))
    persistent = builder.allocate_persistent(batch, metadata)
    builder.update_persistent(persistent, batch, metadata)

    # The boundary is ceil(bucket / TP), not ceil(step tokens / TP).
    assert persistent.active_token_mask.tolist() == [1, 0, 0, 0]


@pytest.mark.parametrize(
    ("counts", "raw_counts", "error"),
    [
        ((), (), "expected 2 DP execution token counts"),
        ((2,), (2,), "expected 2 DP execution token counts"),
        ((2, 3), (), "expected 2 raw DP execution token counts"),
        ((2, 3), (2,), "expected 2 raw DP execution token counts"),
        ((2, -1), (2, 0), "must be non-negative"),
        ((1, 3), (1, 3), "does not match the local input"),
        ((2, 3), (2, -1), "must fit the execution rows|must be non-negative"),
        ((2, 3), (2, 4), "must fit the execution rows"),
        ((2, 3), (1, 3), "disagrees with InputBatch"),
    ],
)
@pytest.mark.parametrize("builder_type", [Qwen3_5MoeMetadataBuilder, SpMetadataBuilder])
def test_inconsistent_authoritative_counts_fail_before_execution(
    counts: tuple[int, ...], raw_counts: tuple[int, ...], error: str, builder_type: type
) -> None:
    builder = builder_type(tp_size=2, tp_rank=0, dp_size=2, dp_rank=0)
    with pytest.raises(RuntimeError, match=error):
        builder.build(_batch(2), _metadata(counts, raw_counts))


def test_dummy_input_cannot_be_marked_as_a_real_token() -> None:
    builder = Qwen3_5MoeMetadataBuilder(tp_size=2, tp_rank=0, dp_size=2, dp_rank=0)
    with pytest.raises(RuntimeError, match="disagrees with InputBatch"):
        builder.build(_batch(1, dummy=True), _metadata((1, 2)))


@pytest.mark.parametrize("builder_type", [Qwen3_5MoeMetadataBuilder, SpMetadataBuilder])
@pytest.mark.parametrize("count_name", ["dp_execution_token_counts", "raw_dp_execution_token_counts"])
def test_host_counts_reject_device_tensors_instead_of_syncing(builder_type: type, count_name: str) -> None:
    builder = builder_type(tp_size=2, tp_rank=0, dp_size=2, dp_rank=0)
    runtime = _metadata((1, 2))
    setattr(runtime, count_name, torch.tensor([1, 2]))
    with pytest.raises(TypeError, match="host integer sequence"):
        builder.build(_batch(1), runtime)


def test_graph_rejects_remote_capacity_overflow_without_mutating_mask() -> None:
    builder = Qwen3_5MoeMetadataBuilder(tp_size=2, tp_rank=0, dp_size=2, dp_rank=0)
    batch = _batch(2, bucket=4)
    persistent = builder.allocate_persistent(batch, _metadata((2, 3)))
    persistent.active_token_mask.fill_(1)
    with pytest.raises(RuntimeError, match="must fit the Qwen3.5 SP graph capacity"):
        builder.update_persistent(persistent, batch, _metadata((2, 5)))
    assert persistent.active_token_mask.tolist() == [1, 1]


@pytest.mark.parametrize(
    "layout_overrides",
    [
        {"compacted": True},
        {"variable": True},
        {"dispatch_counts": (1,) * 4},
        {"dispatch_rows": 1},
        {"ep_rank": 1},
        {"active_rows": 0},
        {"dp_size": 1},
    ],
)
def test_sp_graph_rejects_wrong_layout_without_mutating_it(layout_overrides: dict[str, object]) -> None:
    builder = SpMetadataBuilder(tp_size=2, tp_rank=0, dp_size=2, dp_rank=0)
    batch = _batch(2, bucket=4)
    metadata = _metadata((2, 3))
    persistent = builder.allocate_persistent(batch, metadata)
    layout = persistent.ep_layout
    invalid = replace(persistent, ep_layout=replace(layout, **layout_overrides))
    with pytest.raises(ValueError, match="SP graph layout does not match"):
        builder.update_persistent(invalid, batch, metadata)
    assert persistent.ep_layout is layout


@pytest.mark.parametrize(
    ("mask", "error_type", "error"),
    [
        (torch.zeros(4, dtype=torch.int8), ValueError, "shape"),
        (torch.zeros((2, 1), dtype=torch.int8), ValueError, "shape"),
        (torch.zeros(2, dtype=torch.bool), TypeError, "int8"),
        (torch.zeros(2, dtype=torch.int8, device="meta"), ValueError, "input device"),
        (torch.zeros(4, dtype=torch.int8)[::2], ValueError, "contiguous"),
    ],
)
def test_graph_rejects_invalid_persistent_mask(mask: torch.Tensor, error_type: type[Exception], error: str) -> None:
    builder = Qwen3_5MoeMetadataBuilder(tp_size=2, tp_rank=0, dp_size=2, dp_rank=0)
    batch = _batch(2, bucket=4)
    metadata = _metadata((2, 3))
    persistent = replace(builder.allocate_persistent(batch, metadata), active_token_mask=mask)
    with pytest.raises(error_type, match=error):
        builder.update_persistent(persistent, batch, metadata)


@pytest.mark.parametrize(
    ("padding", "error_type", "error"),
    [
        (None, ValueError, "padding mask matching"),
        (torch.zeros(2, dtype=torch.bool), ValueError, "padding mask matching"),
        (torch.zeros(4, dtype=torch.int8), TypeError, "bool"),
        (torch.zeros(4, dtype=torch.bool, device="meta"), ValueError, "input device"),
    ],
)
def test_graph_rejects_invalid_padding_before_allocation(
    padding: torch.Tensor | None, error_type: type[Exception], error: str
) -> None:
    builder = Qwen3_5MoeMetadataBuilder(tp_size=2, tp_rank=0, dp_size=2, dp_rank=0)
    batch = replace(_batch(2, bucket=4), is_padding=padding)
    with pytest.raises(error_type, match=error):
        builder.allocate_persistent(batch, _metadata((2, 3)))


@pytest.mark.parametrize("tp_rank", range(4))
def test_graph_update_rejects_short_padding_without_mutating_mask(tp_rank: int) -> None:
    builder = Qwen3_5MoeMetadataBuilder(tp_size=4, tp_rank=tp_rank, dp_size=2, dp_rank=0)
    batch = _batch(3, bucket=5)
    persistent = builder.allocate_persistent(batch, _metadata((3, 4)))
    persistent.active_token_mask.fill_(1)
    invalid_batch = replace(batch, is_padding=torch.zeros(3, dtype=torch.bool))
    with pytest.raises(ValueError, match="padding mask matching"):
        builder.update_persistent(persistent, invalid_batch, _metadata((3, 4)))
    assert persistent.active_token_mask.tolist() == [1, 1]


def test_single_dp_can_use_input_batch_authority_without_count_arrays() -> None:
    builder = Qwen3_5MoeMetadataBuilder(tp_size=4, tp_rank=3, dp_size=1, dp_rank=0)
    assert builder.build(_batch(2), SimpleNamespace()).active_token_mask.tolist() == [0]
    assert builder.build(_batch(1, dummy=True), SimpleNamespace()).active_token_mask.tolist() == [0]
    sp_builder = SpMetadataBuilder(tp_size=4, tp_rank=3, dp_size=1, dp_rank=0)
    assert sp_builder.build(_batch(2), SimpleNamespace()).ep_layout.active_rows == 0
    assert sp_builder.build(_batch(1, dummy=True), SimpleNamespace()).ep_layout.active_rows == 0


@pytest.mark.parametrize(
    "overrides",
    [{"sequence_parallel_enabled": False}, {"device": "cpu"}, {"num_experts": 0}],
)
def test_builder_only_activates_for_npu_sp_moe(overrides: dict[str, object]) -> None:
    assert Qwen3_5MoeMetadataBuilder.from_config(_config(**overrides)) is None


def test_moe_builder_does_not_infer_sp_from_tp_or_megamoe() -> None:
    config = _config()
    del config["sequence_parallel_enabled"]
    assert Qwen3_5MoeMetadataBuilder.from_config(config) is None


def test_moe_builder_reads_resolved_state_without_rechecking_tp() -> None:
    config = _config(tp_size=1, ep_size=2, enable_mega_moe=False)
    assert isinstance(Qwen3_5MoeMetadataBuilder.from_config(config), Qwen3_5MoeMetadataBuilder)


@pytest.mark.parametrize(
    "overrides",
    [
        {"tp_size": 1},
        {"dp_size": 1},
        {"world_size": 8},
        {"ep_size": 2},
        {"moe_tp_size": 2},
        {"cp_size": 2},
        {"num_speculative_tokens": 1},
    ],
)
def test_moe_builder_does_not_own_model_or_sp_policy_validation(overrides: dict[str, object]) -> None:
    builder = Qwen3_5MoeMetadataBuilder.from_config(_config(**overrides))
    assert isinstance(builder, Qwen3_5MoeMetadataBuilder)


@pytest.mark.parametrize("sp", [False, True])
def test_lv2_prepares_bool_mask_once_for_eager_and_persistent_batches(sp: bool) -> None:
    builder = Qwen3_5MoeMetadataBuilder.from_config(
        _config(expert_parallel_degree=2, enable_mega_moe=False, sequence_parallel_enabled=sp, tp_rank=1)
    )
    eager = builder.build(_batch(5), _metadata((5, 17)))
    assert eager.active_rows == 2
    assert builder.build(_batch(1, dummy=True), _metadata((1, 17), (0, 17))).active_rows == 0
    mask = eager.active_token_mask
    assert mask.dtype == torch.bool
    assert mask.tolist() == [True, True] + [False] * 7
    assert mask.to(torch.bool).contiguous() is mask

    persistent = builder.allocate_persistent(_batch(17, bucket=18), _metadata((17, 17)))
    assert persistent.active_rows is None
    mask = persistent.active_token_mask
    pointer = mask.data_ptr()
    assert mask.dtype == torch.bool
    for rows, real_rows in ((17, 17), (1, 0), (5, 5)):
        builder.update_persistent(
            persistent,
            _batch(rows, dummy=real_rows == 0, bucket=18),
            _metadata((rows, 17), (real_rows, 17)),
        )
        assert persistent.active_token_mask is mask
        assert mask.data_ptr() == pointer
        assert mask.to(torch.bool).contiguous() is mask
        assert mask.tolist() == [9 + row < real_rows for row in range(9)]


def test_builder_accepts_supported_npu_topology() -> None:
    builder = Qwen3_5MoeMetadataBuilder.from_config(_config())
    assert isinstance(builder, Qwen3_5MoeMetadataBuilder)


def _dispatch_test_experts() -> _NpuQwen3_5Experts:
    """Build the smallest expert object needed to exercise dispatch selection."""
    experts = _NpuQwen3_5Experts.__new__(_NpuQwen3_5Experts)
    nn.Module.__init__(experts)
    experts._sequence_parallel_enabled = True
    experts._comm_policy = None
    experts._mega_moe_enabled = False
    experts._mega_moe_num_max_tokens_per_rank = MEGA_MOE_MAX_TOKENS
    experts.tp_size = 2
    experts.dp_size = 2
    experts.dp_rank = 0
    experts.ep_size = 4
    experts.ep_rank = 0
    experts.w13 = nn.Parameter(torch.empty(1, 4))
    return experts


def _dispatch_context() -> object:
    metadata_fields: dict[str, object] = {
        "dp_execution_token_counts": (2, 2),
    }
    metadata = type(
        "ForwardMetadata",
        (),
        metadata_fields,
    )()
    layout = SpMetadata.from_execution_rows(
        execution_rows=2,
        tp_size=2,
        tp_rank=0,
        ep_layout=EpSpTokenLayout.from_token_counts(
            (2, 2), (2, 2), tp_size=2, tp_rank=0, dp_size=2, dp_rank=0, compacted=True
        ),
    )
    execution_contexts = {
        SpMetadata: layout,
        Qwen3_5MoeMetadata: type("MoeMetadata", (), {"active_token_mask": torch.ones(1, dtype=torch.int8)})(),
    }
    return type(
        "ForwardContext",
        (),
        {
            "metadata": metadata,
            "execution_state": None,
            "acl_graph": None,
            "execution_contexts": execution_contexts,
        },
    )()


def test_sp_direct_ep_does_not_use_an_execution_phase_gate() -> None:
    experts = _dispatch_test_experts()
    hidden = torch.zeros(1, 4)
    direct_calls: list[torch.Tensor] = []
    layouts: list[SpMetadata] = []

    def direct(value: torch.Tensor, layout: SpMetadata) -> torch.Tensor:
        direct_calls.append(value)
        layouts.append(layout)
        return value

    experts._forward_ep_level1_for_sp = direct

    context = _dispatch_context()
    with forward_context(context):
        result = experts(hidden, sp_metadata=context.execution_contexts[SpMetadata])

    assert result is hidden
    assert direct_calls == [hidden]
    assert layouts == [context.execution_contexts[SpMetadata]]


@pytest.mark.parametrize("shape", [(), (2, 4), (1, 3), (1, 4, 1)])
def test_sp_experts_reject_invalid_hidden_before_compute(shape: tuple[int, ...]) -> None:
    experts = _dispatch_test_experts()

    context = _dispatch_context()
    with forward_context(context), pytest.raises(ValueError, match="local SP hidden rows"):
        experts(torch.zeros(shape), sp_metadata=context.execution_contexts[SpMetadata])


@pytest.mark.parametrize("dp_size,ep_size,moe_tp_size", [(1, 1, 2), (1, 2, 1), (2, 2, 2), (2, 4, 1)])
def test_eplv1_builds_sp_validity_without_megamoe_resources(dp_size: int, ep_size: int, moe_tp_size: int) -> None:
    builder = Qwen3_5MoeMetadataBuilder.from_config(
        _config(
            enable_mega_moe=False,
            dp_size=dp_size,
            ep_size=ep_size,
            moe_tp_size=moe_tp_size,
        )
    )
    assert isinstance(builder, Qwen3_5MoeMetadataBuilder)
    batch = _batch(1, dummy=True, bucket=3)
    metadata = _metadata((1,) * dp_size, (0,) * dp_size)
    persistent = builder.allocate_persistent(batch, metadata)
    pointer = persistent.active_token_mask.data_ptr()
    builder.update_persistent(persistent, batch, metadata)
    assert persistent.active_token_mask.tolist() == [0, 0]
    builder.update_persistent(persistent, _batch(3, bucket=3), _metadata((3,) * dp_size))
    assert persistent.active_token_mask.tolist() == [1, 1]
    assert persistent.active_token_mask.data_ptr() == pointer


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"tp_size": 0}, "invalid SP TP geometry"),
        ({"dp_size": 0}, "invalid SP DP geometry"),
        ({"tp_rank": 2}, "invalid SP TP geometry"),
        ({"dp_rank": -1}, "invalid SP DP geometry"),
    ],
)
def test_builder_rejects_invalid_mask_rank_geometry(overrides: dict[str, object], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        Qwen3_5MoeMetadataBuilder.from_config(_config(**overrides))
