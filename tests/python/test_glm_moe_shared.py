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

"""Shared GLM core contracts against the published HF eager MoE equations.

References are pinned in xllm/python/layers/moe.py docstrings. These are small CPU
reference equations and mocked transport tests, not full HF/NPU inference.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from xllm.python import kernels
from xllm.python.layers.moe import MoECore, clamped_silu_and_mul
from xllm.python.layers.moe_parallel import TokenParallelLayout
from xllm.python.models.glm5_2 import Glm52Config
from xllm.python.models.glm5_2_moe import Glm52MoE
from xllm.python.models.glm5_next import Glm5NextConfig, Glm5NextMoE


def _model(kind: str, **overrides: object) -> Glm52MoE | Glm5NextMoE:
    fields = dict(
        hidden_size=8,
        moe_intermediate_size=16,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        tp_size=2,
        ep_size=2,
        moe_tp_size=1,
        expert_parallel_degree=2,
    )
    fields.update(overrides)
    if kind == "glm52":
        model = Glm52MoE(Glm52Config(world_size=2, **fields), 0, torch.bfloat16, torch.device("cpu"))
    else:
        model = Glm5NextMoE(Glm5NextConfig(**fields), torch.bfloat16, torch.device("cpu"))
        model.experts_w13 = torch.empty(0, dtype=torch.int8)
        model.experts_w2 = torch.empty(0, dtype=torch.int8)
        model.experts_w13_scale = torch.ones(4, 32)
        model.experts_w2_scale = torch.ones(4, 8)
    return model


@pytest.mark.parametrize("name", ["_topk", "_route_ep", "_run_routed_sp", "_forward_source"])
def test_models_inherit_identical_core_methods(name: str) -> None:
    assert getattr(Glm52MoE, name) is getattr(Glm5NextMoE, name) is getattr(MoECore, name)


@pytest.mark.parametrize("limit", [0.0, 2.0, 10.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_swiglu_matches_hf_pre_activation_clamp(limit: float, dtype: torch.dtype) -> None:
    gate = torch.tensor([[-20.0, -11.0, -2.0, 0.0, 2.0, 11.0, 20.0]], dtype=dtype)
    up = torch.tensor([[-20.0, 20.0, -1.0, 1.0, 20.0, -20.0, 20.0]], dtype=dtype)
    expected_gate = gate.clamp(max=limit) if limit else gate
    expected_up = up.clamp(-limit, limit) if limit else up
    expected = F.silu(expected_gate) * expected_up
    actual = clamped_silu_and_mul(torch.cat((gate, up), dim=-1), limit)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if limit:
        # Catch symmetric gate clamp, post-SiLU clamp, and ignored up clamp.
        wrong_symmetric = F.silu(gate.clamp(-limit, limit)) * expected_up
        wrong_post = F.silu(gate).clamp(max=limit) * expected_up
        assert not torch.equal(actual, wrong_symmetric)
        # SiLU(10) rounds to 10 in half precision; limit=2 separates
        # pre/post-clamp ordering for every tested dtype.
        if dtype == torch.float32 or limit == 2.0:
            assert not torch.equal(actual, wrong_post)


@pytest.mark.parametrize("kind", ["glm52", "next"])
@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("groups", [1, 2])
def test_shared_router_matches_hf_bias_selection_and_unbiased_weights(kind: str, norm: bool, groups: int) -> None:
    model = _model(kind, norm_topk_prob=norm, n_group=groups)
    hidden = torch.eye(8, dtype=torch.bfloat16)[:3]
    # Deterministic, separated scores: correction forces low-score experts.
    logits = torch.arange(8, dtype=torch.float32) / 4 - 1
    with torch.no_grad():
        model.gate.weight.copy_(logits[:, None].expand(8, 8))
        model.e_score_correction_bias.copy_(torch.tensor([8.0, 7.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    actual_logits, weights, indices = model._topk(hidden)
    scores = F.linear(hidden.float(), model.gate.weight.float()).sigmoid()
    choice = (scores + model.e_score_correction_bias).detach()
    expected_ids = []
    # Independent literal noaux_tc rule, without core's tensor group masking.
    for row in choice:
        group_score = [
            sum(sorted(row[g * (8 // groups) : (g + 1) * (8 // groups)].tolist())[-2:]) for g in range(groups)
        ]
        owner = max(range(groups), key=lambda g: group_score[g])
        allowed = range(owner * (8 // groups), (owner + 1) * (8 // groups))
        expected_ids.append(sorted(allowed, key=lambda i: float(row[i]), reverse=True)[:2])
    expected_ids_tensor = torch.tensor(expected_ids)
    torch.testing.assert_close(indices.sort(-1).values, expected_ids_tensor.sort(-1).values)
    expected_weights = scores.gather(1, indices)
    if norm:
        expected_weights = expected_weights / (expected_weights.sum(-1, keepdim=True) + 1e-20)
    torch.testing.assert_close(weights, expected_weights * 2.5, rtol=0, atol=0)
    assert actual_logits.dtype == weights.dtype == torch.float32
    assert not torch.equal(indices.sort(-1).values, scores.topk(2, dim=-1).indices.sort(-1).values)


@pytest.mark.parametrize("kind,limit,mode", [("glm52", 0.0, 0), ("next", 10.0, 1)])
def test_shared_alltoall_keeps_model_activation_and_scale_contract(
    kind: str, limit: float, mode: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(kind)
    source = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8)
    mask = torch.tensor([True, False])
    layout = TokenParallelLayout(4, 2, 0, routed_backend="alltoall")
    route = Mock(return_value=(torch.ones(2, 2), torch.zeros(2, 2, dtype=torch.int32)))
    dispatch = Mock(return_value=source * 2)

    def shared_compute(packed: torch.Tensor) -> torch.Tensor:
        # The compression path feeds only the active rows; scale whatever
        # row count arrives so the contract holds under either form.
        return packed * 3

    shared = Mock(side_effect=shared_compute)
    monkeypatch.setattr(kernels, "moe_gate_routing", route, raising=False)
    monkeypatch.setattr(kernels, "ep_moe_w8a8_alltoall", dispatch, raising=False)
    monkeypatch.setattr(model, "_run_shared_source", shared)
    actual = model._forward_source(source, layout, mask)
    expected = (source * 8).masked_fill(~mask[:, None], 0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert dispatch.call_args.kwargs["swiglu_limit"] == limit
    assert dispatch.call_args.kwargs["swiglu_mode"] == mode
    assert dispatch.call_args.args[6] is model._ep_w2_scale
    assert route.call_args.args[0].dtype == torch.float32
    assert route.call_args.kwargs["routed_scaling_factor"] == 1.0
    assert dispatch.call_args.kwargs["x_active_mask"] is mask


@pytest.mark.parametrize("kind", ["glm52", "next"])
def test_shared_combine_scales_in_fp32_before_final_cast(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model(kind)
    source = torch.ones(2, 8, dtype=torch.bfloat16)
    routed = torch.full_like(source, 1.0078125)
    shared = torch.full_like(source, -2.5)
    monkeypatch.setattr(model, "_run_routed_sp", lambda *args: routed)
    monkeypatch.setattr(model, "_run_shared_source", lambda *args: shared)
    actual = model._forward_source(source, TokenParallelLayout(4, 2, 0, "alltoall"), torch.ones(2, dtype=torch.bool))
    expected = torch.add(shared.float(), routed.float(), alpha=2.5).to(source.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, shared + routed * 2.5)


def test_glm52_cannot_inherit_clamped_mc2_by_accident(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model("glm52")
    monkeypatch.setattr(model, "_route_ep", Mock())
    with pytest.raises(ValueError, match="Standard GLM SwiGLU"):
        model._run_routed_sp(torch.ones(2, 8), TokenParallelLayout(4, 2, 0, "mc2"), torch.ones(2, dtype=torch.bool))


def test_core_does_not_change_checkpoint_module_names() -> None:
    for kind in ("glm52", "next"):
        model = _model(kind)
        names = set(model.state_dict())
        assert "gate.weight" in names
        assert not any(name.startswith("core.") for name in names)
        assert "shared_experts" in dict(model.named_children())


class _MegaSentinelCore(MoECore):
    """Minimal host exercising _run_routed_sp's megamoe scale sentinel."""

    def __init__(self, w13_scale: torch.Tensor, w2_scale: torch.Tensor) -> None:
        self.cfg = SimpleNamespace(
            norm_topk_prob=True,
            ep_size=1,
            ep_rank=0,
            enable_mega_moe=True,
            mega_moe_num_max_tokens_per_rank=128,
            mega_moe_ccl_buffer_size=0,
        )
        self.topk = 2
        self.hidden = 4
        self.mega_moe_context = torch.zeros(1, dtype=torch.int32)
        self.mega_moe_w13_scale = w13_scale
        self.mega_moe_w2_scale = w2_scale


@pytest.mark.parametrize("empty_form", ["none", "empty_int64"])
def test_megamoe_scale_sentinel_rejects_both_unencoded_forms(empty_form: str) -> None:
    """DeepseekV3MoE registers empty int64 buffers; Glm5NextMoE registers None.

    The megamoe branch must treat either sentinel as "not encoded" instead of
    passing empty scales into the fused kernel.
    """
    if empty_form == "none":
        w13 = w2 = None
    else:
        w13 = w2 = torch.empty(0, dtype=torch.int64)
    core = _MegaSentinelCore(w13, w2)
    # num_tokens=0 keeps valid_tokens=0, so the branch skips router math and
    # hits the megamoe admission checks directly.
    layout = TokenParallelLayout(0, 1, 0, routed_backend="megamoe")
    with pytest.raises(RuntimeError, match="scales must be encoded"):
        core._run_routed_sp(torch.zeros(1, 4), layout, torch.zeros(1, dtype=torch.bool))


def test_acl_graph_keeps_fixed_shape_and_skips_compression(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under ACL graph capture the shared GEMMs must stay fixed-shape.

    ``mask.all()`` is a device->host sync and ``nonzero`` is dynamically
    shaped; the degree-2 MC2/MegaMoE graph path replays a captured bucket
    whose persistent mask carries false padding rows. Under capture the
    helper must consume the FULL source and drop padding via masked_fill,
    exactly like the pre-refactor code -- no compression, no sync.
    """
    from xllm.python.layers import moe as moe_module

    monkeypatch.setattr(moe_module, "in_acl_graph", lambda: True)
    model = _model("next")
    source = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    mask = torch.tensor([True, False])
    layout = TokenParallelLayout(4, 2, 0, "alltoall")
    shared = Mock(return_value=source * 3)
    routed = Mock(return_value=source * 2)
    monkeypatch.setattr(model, "_run_shared_source", shared)
    monkeypatch.setattr(model, "_run_routed_sp", routed)
    actual = model._forward_source(source, layout, mask)
    expected = (source * 8).masked_fill(~mask[:, None], 0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Full-bucket shape: the padding row still ran the shared GEMMs.
    assert shared.call_args.args[0] is source
    assert routed.call_args.args[0] is source
