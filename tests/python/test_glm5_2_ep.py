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

"""CPU ownership contracts for real GLM5.2 AllToAllV EP and EPLv2."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from xllm.python.layers.moe_parallel import TokenParallelLayout
from xllm.python.layers.npu.glm5_2_metadata import Glm52EpMetadata, Glm52EpMetadataBuilder
from xllm.python.model_executor.cp_utils import cp_shard_rows
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.model_executor.input_batch import InputBatch
from xllm.python.models import deepseek_v32, glm5_2_moe
from xllm.python.models.glm5_2 import Glm52Config
from xllm.python.models.glm5_2_moe import Glm52MoE, validate_glm52_ep_execution_config


def _config(tp: int = 4, cp: int = 2, degree: int = 2, **overrides: object) -> Glm52Config:
    fields = dict(
        hidden_size=8,
        moe_intermediate_size=16,
        n_routed_experts=16,
        num_experts_per_tok=2,
        n_shared_experts=1,
        tp_size=tp,
        cp_size=cp,
        dp_size=1,
        world_size=tp * cp,
        ep_size=tp * cp,
        moe_tp_size=1,
        expert_parallel_degree=degree,
    )
    fields.update(overrides)
    return Glm52Config(**fields)


def _context(mask: torch.Tensor | None, cp: object = None, **fields: object) -> ForwardContext:
    values = dict(
        is_prefill=True, is_chunked_prefill=False, is_dummy=False, dp_is_decode=(), raw_dp_execution_token_counts=()
    )
    values.update(fields)
    return ForwardContext(
        Mock(),
        torch.device("cpu"),
        SimpleNamespace(**values),
        [],
        cp_context=cp,
        execution_contexts={} if mask is None else {Glm52EpMetadata: Glm52EpMetadata(mask)},
    )


def _cp(index: list[int], rows: int, rank: int) -> SimpleNamespace:
    indices = torch.tensor(index)
    return SimpleNamespace(
        cp_size=2,
        cp_rank=rank,
        restore_index=torch.arange(rows),
        shard_gather_index=indices.clamp_min(0),
        shard_valid_mask=indices >= 0,
    )


def _routing(logits: torch.Tensor, *args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
    assert logits.dtype == torch.float32
    assert kwargs["routed_scaling_factor"] == 1.0
    return torch.ones(logits.shape[0], 2), torch.zeros(logits.shape[0], 2, dtype=torch.int32)


@pytest.mark.parametrize("degree", (0, 1, 2))
@pytest.mark.parametrize("chunked", (False, True))
def test_tp4_cp2_ep8_routes_each_query_once_without_reducing_shared(
    degree: int,
    chunked: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two ragged sequences, including a one-token sequence and cached prefix.
    hidden = torch.arange(1, 12 * 8 + 1, dtype=torch.bfloat16).reshape(12, 8)
    indices = ([0, -1, 1, 2, 3, 10, 11, -1], [-1, -1, 4, 5, 6, 7, 8, 9])
    seen: list[float] = []
    forbidden = Mock(side_effect=AssertionError("AllToAllV output/shared must never be TP/EP summed"))
    monkeypatch.setattr(glm5_2_moe.distributed, "all_reduce_", forbidden, raising=False)
    monkeypatch.setattr(glm5_2_moe.distributed, "moe_ep_all_reduce", forbidden, raising=False)
    monkeypatch.setattr(glm5_2_moe.kernels, "grouped_moe", forbidden, raising=False)
    monkeypatch.setattr(glm5_2_moe.kernels, "moe_gate_routing", _routing, raising=False)

    def dispatch(source: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        mask = kwargs["x_active_mask"]
        assert mask.dtype == torch.bool
        assert kwargs["swiglu_mode"] == 0
        assert kwargs["swiglu_limit"] == 0.0
        assert kwargs["ep_size"] == 8
        seen.extend(source[mask, 0].tolist())
        return source * 2

    monkeypatch.setattr(glm5_2_moe.kernels, "ep_moe_w8a8_alltoall", dispatch, raising=False)
    for cp_rank in range(2):
        cp = _cp(list(indices[cp_rank]), hidden.shape[0], cp_rank)
        local = cp_shard_rows(hidden, cp)
        expected = local * 8  # shared*3 + routed*2 * scale2.5
        gathered = torch.cat([TokenParallelLayout(local.shape[0], 4, rank).shard(expected) for rank in range(4)])
        for tp_rank in range(4):
            expected_source = TokenParallelLayout(local.shape[0], 4, tp_rank).shard(expected)

            def gather(
                value: torch.Tensor, *args: object, expected_source=expected_source, gathered=gathered
            ) -> torch.Tensor:
                torch.testing.assert_close(value, expected_source, rtol=0, atol=0)
                return gathered

            monkeypatch.setattr(glm5_2_moe.distributed, "tp_all_gather", gather, raising=False)
            cfg = _config(degree=degree, tp_rank=tp_rank, cp_rank=cp_rank, ep_rank=cp_rank * 4 + tp_rank)
            moe = Glm52MoE(cfg, 0, torch.bfloat16, torch.device("cpu"))
            assert moe.shared_experts.tp == 1
            moe._run_shared_alltoall = Mock(side_effect=lambda value: value * 3)
            before = local.clone()
            with forward_context(
                _context(torch.ones(12, dtype=torch.bool), cp, is_prefill=not chunked, is_chunked_prefill=chunked)
            ):
                torch.testing.assert_close(moe(local), expected, rtol=0, atol=0)
            torch.testing.assert_close(local, before, rtol=0, atol=0)
    assert sorted(seen) == hidden[:, 0].tolist()
    forbidden.assert_not_called()


@pytest.mark.parametrize("tp_rank", range(4))
def test_empty_tp_owners_still_join_alltoall_with_false_mask(tp_rank: int, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config(cp=1, tp_rank=tp_rank, ep_rank=tp_rank)
    moe = Glm52MoE(cfg, 0, torch.bfloat16, torch.device("cpu"))
    hidden = torch.ones(1, 8, dtype=torch.bfloat16)
    dispatch = Mock(side_effect=lambda source, *args, **kwargs: torch.zeros_like(source))
    monkeypatch.setattr(glm5_2_moe.kernels, "moe_gate_routing", _routing, raising=False)
    monkeypatch.setattr(glm5_2_moe.kernels, "ep_moe_w8a8_alltoall", dispatch, raising=False)
    monkeypatch.setattr(glm5_2_moe.distributed, "tp_all_gather", lambda *args: torch.zeros(4, 8), raising=False)
    moe._run_shared_alltoall = Mock(side_effect=torch.zeros_like)
    with forward_context(_context(torch.zeros(1, dtype=torch.bool), is_dummy=True)):
        moe(hidden)
    dispatch.assert_called_once()
    assert not dispatch.call_args.kwargs["x_active_mask"].any()


@pytest.mark.parametrize("degree", (0, 1))
def test_cp1_legacy_decode_keeps_base_path(degree: int, monkeypatch: pytest.MonkeyPatch) -> None:
    moe = Glm52MoE(_config(cp=1, degree=degree), 0, torch.bfloat16, torch.device("cpu"))
    base = Mock(return_value=torch.ones(1, 8))
    monkeypatch.setattr(deepseek_v32.DeepseekV3MoE, "forward", base)
    hidden = torch.ones(1, 8)
    with forward_context(_context(None, is_prefill=False)):
        assert moe(hidden) is base.return_value
    base.assert_called_once_with(hidden)


@pytest.mark.parametrize("graph", (False, True))
@pytest.mark.parametrize("dp_size", (1, 2))
def test_cp1_degree2_mega_pure_decode_preserves_base_and_graph(
    graph: bool, dp_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tp=2, cp=1, dp_size=dp_size, world_size=2 * dp_size, ep_size=2 * dp_size, enable_mega_moe=True)
    cfg.validate()
    validate_glm52_ep_execution_config(
        {"enable_graph": graph, "python_graph_backend": "aclgraph" if graph else "off"}, cfg
    )
    moe = Glm52MoE(cfg, 0, torch.bfloat16, torch.device("cpu"))
    base = Mock(return_value=torch.ones(1, 8))
    monkeypatch.setattr(deepseek_v32.DeepseekV3MoE, "forward", base)
    alltoall = Mock(side_effect=AssertionError("pure legacy Mega decode must not enter AllToAllV"))
    monkeypatch.setattr(moe, "_forward_alltoall", alltoall)
    hidden = torch.ones(1, 8)
    ctx = _context(None, is_prefill=False, dp_is_decode=(1,) * dp_size)
    if graph:
        ctx = replace(ctx, execution_state=object())
    with forward_context(ctx):
        assert not moe._uses_alltoall()
        assert moe(hidden) is base.return_value
    base.assert_called_once_with(hidden)
    alltoall.assert_not_called()


@pytest.mark.parametrize("chunked", (False, True))
@pytest.mark.parametrize("dp_size", (1, 2))
def test_cp1_degree2_mega_prefill_always_selects_alltoall(
    chunked: bool, dp_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tp=2, cp=1, dp_size=dp_size, world_size=2 * dp_size, ep_size=2 * dp_size, enable_mega_moe=True)
    moe = Glm52MoE(cfg, 0, torch.bfloat16, torch.device("cpu"))
    base = Mock(side_effect=AssertionError("Mega configuration must not send prefill to base grouped EP"))
    monkeypatch.setattr(deepseek_v32.DeepseekV3MoE, "forward", base)
    alltoall = Mock(return_value=torch.ones(1, 8))
    monkeypatch.setattr(moe, "_forward_alltoall", alltoall)
    hidden = torch.ones(1, 8)
    # DP2 rank0 may be pure decode, but a peer prefills: all ranks use A2A.
    ctx = _context(
        None,
        is_prefill=not chunked if dp_size == 1 else False,
        is_chunked_prefill=chunked if dp_size == 1 else False,
        dp_is_decode=(1, 0) if dp_size == 2 else (),
    )
    with forward_context(ctx):
        assert moe._uses_alltoall()
        assert moe(hidden) is alltoall.return_value
    alltoall.assert_called_once_with(hidden)
    base.assert_not_called()


def test_mixed_dp_step_selects_alltoall_even_on_decode_rank() -> None:
    moe = Glm52MoE(
        _config(tp=2, cp=1, degree=0, dp_size=2, world_size=4, ep_size=4), 0, torch.bfloat16, torch.device("cpu")
    )
    with forward_context(_context(None, is_prefill=False, dp_is_decode=(1, 0))):
        assert moe._uses_alltoall()
    with forward_context(_context(None, is_prefill=False)), pytest.raises(ValueError, match="replicated DP"):
        moe._uses_alltoall()


@pytest.mark.parametrize("rank", (0, 1))
def test_cp_mixed_dp_real_decode_rejected_before_collectives(rank: int, monkeypatch: pytest.MonkeyPatch) -> None:
    moe = Glm52MoE(
        _config(tp=2, dp_size=2, dp_rank=rank, world_size=8, ep_size=8), 0, torch.bfloat16, torch.device("cpu")
    )
    forbidden = Mock(side_effect=AssertionError("must reject before collective"))
    monkeypatch.setattr(moe, "_uses_alltoall", forbidden)
    with (
        forward_context(_context(None, dp_is_decode=(0, 1), raw_dp_execution_token_counts=(3, 1))),
        pytest.raises(ValueError, match="real decode DP ranks"),
    ):
        moe(torch.ones(1, 8, dtype=torch.bfloat16))
    forbidden.assert_not_called()


@pytest.mark.parametrize("field", ("enable_mega_moe", "enable_eplb"))
def test_eplv2_rejects_incompatible_modes(field: str) -> None:
    cfg = _config(**{field: True})
    with pytest.raises(ValueError, match="does not support"):
        cfg.validate()


@pytest.mark.parametrize("degree", (0, 1))
def test_legacy_ep_degrees_reject_eplb_with_their_own_message(degree: int) -> None:
    """Degree 0/1 configure plain EP, so the EPLB rejection must name their
    legacy grouped path instead of the degree-2 AllToAllV transport."""
    cfg = _config(degree=degree, enable_eplb=True)
    with pytest.raises(ValueError, match="legacy EP .* does not support EPLB expert migration"):
        cfg.validate()


def test_cp2_degree2_mega_is_rejected_before_model_allocation() -> None:
    cfg = _config(enable_mega_moe=True)
    with pytest.raises(ValueError, match="CP AllToAllV.*enable_mega_moe"):
        cfg.validate()
    with pytest.raises(ValueError, match="CP AllToAllV.*enable_mega_moe"):
        validate_glm52_ep_execution_config({"enable_graph": True}, cfg)


def test_eplv2_rejects_graph_before_model_allocation() -> None:
    cfg = _config()
    for config in ({"enable_graph": True}, {"python_graph_backend": "aclgraph"}):
        with pytest.raises(ValueError, match="eager-only"):
            validate_glm52_ep_execution_config(config, cfg)


@pytest.mark.parametrize("case", ("missing_mask", "wrong_dtype", "cp_decode", "graph"))
def test_source_admission_fails_closed(case: str) -> None:
    cfg = _config(cp=1 if case != "cp_decode" else 2)
    moe = Glm52MoE(cfg, 0, torch.bfloat16, torch.device("cpu"))
    ctx = _context(None if case == "missing_mask" else torch.ones(2, dtype=torch.bool))
    hidden = torch.ones(2, 8, dtype=torch.float32 if case == "wrong_dtype" else torch.bfloat16)
    if case == "cp_decode":
        ctx.metadata.is_prefill = False
    if case == "graph":
        ctx = replace(ctx, execution_state=object())
    with forward_context(ctx), pytest.raises((ValueError, TypeError)):
        moe(hidden)


def _batch(tokens: int, dummy: bool = False, capacity: int | None = None) -> InputBatch:
    capacity = tokens if capacity is None else capacity
    return InputBatch(
        torch.zeros(capacity, dtype=torch.long),
        torch.zeros(capacity, dtype=torch.long),
        1,
        tokens,
        capacity,
        (tokens,),
        (100,),
        (0, tokens),
        (True,),
        None,
        dummy,
    )


def test_typed_metadata_counts_queries_not_cached_prefix_and_clears_graph_tail() -> None:
    builder = Glm52EpMetadataBuilder(2, 1)
    metadata = SimpleNamespace(raw_dp_execution_token_counts=(5, 2))
    persistent = builder.allocate_persistent(_batch(3, capacity=8), metadata)
    address = persistent.local_token_mask.data_ptr()
    builder.update_persistent(persistent, _batch(3, capacity=8), metadata)
    assert persistent.local_token_mask.tolist() == [True, True, False, False, False, False, False, False]
    builder.update_persistent(
        persistent, _batch(1, dummy=True, capacity=8), SimpleNamespace(raw_dp_execution_token_counts=(5, 0))
    )
    assert not persistent.local_token_mask.any()
    assert persistent.local_token_mask.data_ptr() == address
    with pytest.raises(TypeError):
        builder.update_persistent(object(), _batch(1), metadata)


def test_shared_alltoall_uses_standard_swiglu_without_tp_reduce(monkeypatch: pytest.MonkeyPatch) -> None:
    moe = Glm52MoE(_config(), 0, torch.bfloat16, torch.device("cpu"))
    moe._fuse_shared_expert = True
    source = torch.ones(2, 8, dtype=torch.bfloat16)
    quantized = torch.zeros(2, 8, dtype=torch.int8)
    input_scale = torch.ones(2)
    gate_up = torch.zeros(2, 32, dtype=torch.int32)
    act = torch.zeros(2, 16, dtype=torch.int8)
    act_scale = torch.ones(2)
    moe.shared_experts.gate_up_proj.forward_accumulated = Mock(return_value=gate_up)
    moe.shared_experts.down_proj.forward_quantized = Mock(return_value=source)
    monkeypatch.setattr(glm5_2_moe.kernels, "dynamic_quant", Mock(return_value=(quantized, input_scale)), raising=False)
    activation = Mock(return_value=(act, act_scale))
    monkeypatch.setattr(glm5_2_moe.kernels, "dequant_swiglu_quant", activation, raising=False)
    forbidden = Mock(side_effect=AssertionError("shared must not be TP-reduced"))
    monkeypatch.setattr(glm5_2_moe.distributed, "tp_all_reduce", forbidden, raising=False)
    assert moe._run_shared_alltoall(source) is source
    assert activation.call_args.kwargs == {"swiglu_mode": 0, "clamp_limit": 0.0}
    moe.shared_experts.down_proj.forward_quantized.assert_called_once_with(act, act_scale)
    forbidden.assert_not_called()


def test_typed_metadata_rejects_missing_or_invalid_raw_counts() -> None:
    builder = Glm52EpMetadataBuilder(2, 0)
    for counts in ((), (4,), (-1, 2), (4, 2)):
        with pytest.raises(ValueError):
            builder.build(_batch(3), SimpleNamespace(raw_dp_execution_token_counts=counts))
