# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.

"""GLM5.3 MegaMoE admission: own cap inside the graph, All-to-AllV outside it."""

from __future__ import annotations

import pytest
import torch

from xllm.python.layers.moe_parallel import Eplv2CommPolicy, TokenParallelLayout
from xllm.python.models import glm5_next


def _config(**kwargs: object) -> glm5_next.Glm5NextConfig:
    values = dict(
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=16,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        tp_size=2,
        ep_size=2,
        moe_tp_size=1,
        moe_tp_rank=0,
        expert_parallel_degree=2,
        enable_mega_moe=True,
        mega_moe_context=torch.zeros(8, dtype=torch.int32),
        mega_moe_ccl_buffer_size=1024,
        mega_moe_num_max_tokens_per_rank=4,
    )
    values.update(kwargs)
    return glm5_next.Glm5NextConfig(**values)


@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("graph", [False, True])
def test_megamoe_accepts_rows_inside_its_own_cap(rows: int, graph: bool) -> None:
    policy = Eplv2CommPolicy(8, mega_moe_capacity=4)
    assert policy.select(rows, graph=graph) == "megamoe"


@pytest.mark.parametrize("rows", [5, 8, 9])
def test_megamoe_over_cap_uses_eager_alltoall(rows: int) -> None:
    policy = Eplv2CommPolicy(8, mega_moe_capacity=4)
    assert policy.select(rows, graph=False) == "alltoall"
    with pytest.raises(ValueError, match="MegaMoE capacity"):
        policy.select(rows, graph=True)


def test_megamoe_off_auto_selects_mc2_or_alltoall() -> None:
    assert Eplv2CommPolicy(8).select(4) == "mc2"
    assert Eplv2CommPolicy(8).select(9) == "alltoall"
    with pytest.raises(ValueError, match="MC2 capacity"):
        Eplv2CommPolicy(8).select(9, graph=True)


def test_megamoe_preserves_fp32_router_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    moe = glm5_next.Glm5NextMoE(_config(), torch.bfloat16, torch.device("cpu"))
    weights = torch.tensor([[0.12345679, 0.87654321]], dtype=torch.float32)
    assert not torch.equal(weights, weights.to(torch.bfloat16).float())
    ids = torch.tensor([[0, 1]], dtype=torch.int32)
    moe.experts_w13 = torch.empty(0, dtype=torch.int8)
    moe.experts_w2 = torch.empty(0, dtype=torch.int8)
    moe.mega_moe_w13_scale = torch.ones(1, dtype=torch.int64)
    moe.mega_moe_w2_scale = torch.ones(1, dtype=torch.int64)
    monkeypatch.setattr(moe, "_route_ep", lambda hidden: (weights, ids))

    def capture_mega_moe(*args: object, **kwargs: object) -> torch.Tensor:
        router_weights = args[3]
        assert isinstance(router_weights, torch.Tensor)
        assert router_weights.dtype == torch.float32
        torch.testing.assert_close(router_weights[0], weights[0], rtol=0, atol=0)
        assert router_weights.shape == (2, 2)
        assert kwargs["x_active_mask"].tolist() == [True, False]
        return torch.ones(2, 8, dtype=torch.bfloat16)

    monkeypatch.setattr(glm5_next.kernels, "mega_moe", capture_mega_moe, raising=False)
    layout = TokenParallelLayout(2, 2, 0, routed_backend="megamoe", dispatch_rows=2)
    result = moe._run_routed_sp(torch.ones(1, 8, dtype=torch.bfloat16), layout, torch.tensor([True]))
    torch.testing.assert_close(result, torch.ones(1, 8, dtype=torch.bfloat16), rtol=0, atol=0)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"expert_parallel_degree": 1}, "degree=2"),
        ({"mega_moe_context": None}, "communication context"),
        ({"mega_moe_context": torch.zeros(2, 2, dtype=torch.int32)}, "communication context"),
        ({"mega_moe_ccl_buffer_size": 0}, "capacities"),
        ({"mega_moe_num_max_tokens_per_rank": 0}, "capacities"),
    ],
)
def test_megamoe_config_rejects_incomplete_resources(change: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        glm5_next.Glm5NextMoE(_config(**change), torch.bfloat16, torch.device("cpu"))


def test_megamoe_defaults_off() -> None:
    cfg = glm5_next.Glm5NextConfig.from_dict({})
    assert not cfg.enable_mega_moe
    assert cfg.mega_moe_context is None
    parsed = glm5_next.Glm5NextConfig.from_dict(vars(_config()))
    assert parsed.mega_moe_num_max_tokens_per_rank == 4
    assert Eplv2CommPolicy(8, mega_moe_capacity=parsed.mega_moe_num_max_tokens_per_rank).select(4) == "megamoe"


@pytest.mark.parametrize("enable_mega_moe", [False, True])
@pytest.mark.parametrize("mc2_capacity", [2, 8])
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("graph", [False, True])
def test_model_policy_automatically_selects_from_feature_and_capacity(
    monkeypatch: pytest.MonkeyPatch, enable_mega_moe: bool, mc2_capacity: int, offset: int, graph: bool
) -> None:
    monkeypatch.setenv("HCCL_BUFFSIZE", "200")
    cfg = glm5_next.Glm5NextConfig.from_dict(
        vars(_config(enable_mega_moe=enable_mega_moe, hidden_size=1024, eplv2_mc2_max_tokens_per_rank=mc2_capacity))
    )
    policy = glm5_next._eplv2_comm_policy(cfg)
    assert policy is not None
    capacity = 4 if enable_mega_moe else mc2_capacity
    assert policy.mc2_capacity == mc2_capacity
    assert policy.graph_token_limit == capacity
    rows = capacity + offset
    if graph and offset > 0:
        with pytest.raises(ValueError, match="requires eager execution"):
            policy.select(rows, graph=True)
    else:
        expected = ("megamoe" if enable_mega_moe else "mc2") if offset <= 0 else "alltoall"
        assert policy.select(rows, graph=graph) == expected


@pytest.mark.parametrize("enable_mega_moe", [False, True])
@pytest.mark.parametrize("legacy_mode", ["mc2", "alltoall", "invalid"])
def test_removed_mode_inputs_cannot_override_feature_selection(
    monkeypatch: pytest.MonkeyPatch, enable_mega_moe: bool, legacy_mode: str
) -> None:
    monkeypatch.setenv("HCCL_BUFFSIZE", "200")
    monkeypatch.setenv("XLLM_EPLV2_COMM_MODE", legacy_mode)
    values = vars(_config(enable_mega_moe=enable_mega_moe, hidden_size=1024)) | {"eplv2_comm_mode": legacy_mode}
    cfg = glm5_next.Glm5NextConfig.from_dict(values)
    assert "eplv2_comm_mode" not in glm5_next.Glm5NextConfig.__dataclass_fields__
    assert "mode" not in Eplv2CommPolicy.__dataclass_fields__
    policy = glm5_next._eplv2_comm_policy(cfg)
    assert policy is not None
    assert policy.select(1) == ("megamoe" if enable_mega_moe else "mc2")
    assert policy.select(policy.graph_token_limit + 1) == "alltoall"


def test_megamoe_off_does_not_require_communication_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HCCL_BUFFSIZE", "200")
    cfg = glm5_next.Glm5NextConfig.from_dict(
        vars(
            _config(
                enable_mega_moe=False,
                hidden_size=1024,
                mega_moe_context=None,
                mega_moe_ccl_buffer_size=0,
                mega_moe_num_max_tokens_per_rank=0,
            )
        )
    )
    policy = glm5_next._eplv2_comm_policy(cfg)
    assert policy is not None
    assert policy.select(1) == "mc2"


@pytest.mark.parametrize("limit", [float("nan"), float("inf"), -1.0, 1e40])
def test_megamoe_config_rejects_invalid_clamp_before_model_execution(limit: float) -> None:
    with pytest.raises(ValueError, match="SwiGLU"):
        glm5_next.Glm5NextConfig.from_dict(vars(_config(swiglu_limit=limit)))


def test_megamoe_config_accepts_float32_max_clamp() -> None:
    limit = torch.finfo(torch.float32).max
    cfg = glm5_next.Glm5NextConfig.from_dict(vars(_config(swiglu_limit=limit)))
    assert cfg.swiglu_limit == limit


def test_megamoe_off_preserves_existing_large_finite_clamp_contract() -> None:
    cfg = glm5_next.Glm5NextConfig.from_dict(vars(_config(enable_mega_moe=False, swiglu_limit=1e40)))
    assert cfg.swiglu_limit == 1e40


@pytest.mark.parametrize("enable_mega_moe", [False, True])
@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("rows", [1, 4])
def test_fixed_backend_pads_all_inputs_and_trims_local_output(
    monkeypatch: pytest.MonkeyPatch, enable_mega_moe: bool, tp_rank: int, rows: int
) -> None:
    cfg = _config(enable_mega_moe=enable_mega_moe, tp_rank=tp_rank, ep_rank=tp_rank)
    moe = glm5_next.Glm5NextMoE(cfg, torch.bfloat16, torch.device("cpu"))
    moe.experts_w13 = torch.empty(0, dtype=torch.int8)
    moe.experts_w2 = torch.empty(0, dtype=torch.int8)
    if enable_mega_moe:
        moe.mega_moe_w13_scale = torch.ones(1, dtype=torch.int64)
        moe.mega_moe_w2_scale = torch.ones(1, dtype=torch.int64)
    else:
        moe.experts_w13_scale = torch.ones(1)
        moe.experts_w2_scale = torch.ones(1)
        moe._ep_dispatch_group_info = ("ep", tp_rank, 2)
    backend = "megamoe" if enable_mega_moe else "mc2"
    layout = TokenParallelLayout(rows, 2, tp_rank, routed_backend=backend, dispatch_rows=3)
    hidden = torch.ones(layout.shard_tokens, 8, dtype=torch.bfloat16)
    mask = torch.arange(layout.shard_tokens) < layout.valid_tokens
    routing_calls: list[torch.Tensor] = []
    native_calls: list[torch.Tensor] = []

    def route(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        routing_calls.append(value)
        return torch.ones(value.shape[0], 2), torch.zeros(value.shape[0], 2, dtype=torch.int32)

    def capture(*args: object, **kwargs: object) -> torch.Tensor:
        if enable_mega_moe:
            context, actual_hidden, ids, weights = args[:4]
            assert context is cfg.mega_moe_context
        else:
            actual_hidden, weights, ids = args[:3]
        actual_mask = kwargs["x_active_mask"]
        assert actual_hidden.shape == (3, 8) and actual_hidden.dtype == torch.bfloat16
        assert weights.shape == ids.shape == (3, 2)
        assert weights.dtype == torch.float32 and ids.dtype == torch.int32
        torch.testing.assert_close(actual_hidden[: layout.shard_tokens], hidden)
        assert not actual_hidden[layout.shard_tokens :].count_nonzero()
        assert not weights[layout.shard_tokens :].count_nonzero()
        assert not ids[layout.shard_tokens :].count_nonzero()
        torch.testing.assert_close(actual_mask[: layout.shard_tokens], mask)
        assert not actual_mask[layout.shard_tokens :].any()
        native_calls.append(actual_hidden)
        return actual_hidden + 7

    monkeypatch.setattr(moe, "_route_ep", route)
    kernel = "mega_moe" if enable_mega_moe else "ep_moe_w8a8"
    monkeypatch.setattr(glm5_next.kernels, kernel, capture, raising=False)
    output = moe._run_routed_sp(hidden, layout, mask)
    torch.testing.assert_close(output, hidden + 7)
    assert len(native_calls) == 1
    assert len(routing_calls) == int(layout.valid_tokens > 0)


def test_megamoe_zero_clamp_input_keeps_config_default() -> None:
    cfg = glm5_next.Glm5NextConfig.from_dict(vars(_config(swiglu_limit=0.0)))
    assert cfg.swiglu_limit == glm5_next.Glm5NextConfig().swiglu_limit


@pytest.mark.parametrize("ep,cp", [(8, 4), (16, 8)])
@pytest.mark.parametrize("hidden,inter,topk", [(4096, 2048, 8), (8, 16, 2)])
def test_megamoe_npu_constructor_has_no_geometry_whitelist(
    monkeypatch: pytest.MonkeyPatch, ep: int, cp: int, hidden: int, inter: int, topk: int
) -> None:
    # Exercise the NPU-specific constructor branch without allocating on an NPU.
    # Real dtype/topology/resource validators still run; only allocations are mocked.
    device = torch.device("privateuseone:0")

    class NpuContext(torch.Tensor):
        @property
        def device(self) -> torch.device:
            return device

    context = torch.Tensor._make_subclass(NpuContext, torch.zeros(8, dtype=torch.int32))
    cfg = _config(
        ep_size=ep,
        cp_size=cp,
        hidden_size=hidden,
        moe_intermediate_size=inter,
        n_routed_experts=288,
        num_experts_per_tok=topk,
        mega_moe_context=context,
    )
    zeros = torch.zeros
    monkeypatch.setattr(glm5_next.torch, "zeros", lambda *args, **kwargs: zeros(*args, **(kwargs | {"device": "cpu"})))
    monkeypatch.setattr(glm5_next.nn, "Linear", lambda *args, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(glm5_next, "Glm5NextMLP", lambda *args, **kwargs: torch.nn.Identity())
    moe = glm5_next.Glm5NextMoE(cfg, torch.bfloat16, device)
    assert moe.num_local_experts == 288 // ep
    assert moe.mega_moe_context is context


@pytest.mark.parametrize(
    "change,match",
    [
        ({"n_routed_experts": 9}, "divisible by ep_size"),
        ({"ep_size": 4}, "EP size times MoE-TP"),
    ],
)
def test_megamoe_keeps_real_expert_geometry_checks(change: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        glm5_next.Glm5NextMoE(_config(**change), torch.bfloat16, torch.device("cpu"))
