# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wiring equivalence for the V4.1 fused SwiGLU shared-expert activation.

Extracted from the monolithic ``test_deepseek_v41_fused_ops.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn as nn

import xllm.python.models.deepseek_v41 as v41
from tests.python._deepseek_v41_fused_testkit import (
    _SpyLogger,
)
from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
)
from xllm.python.models.deepseek_v41 import DeepseekV41SharedMLP

# === [5] FUSED_SHARED_ACT: fused SwiGLU ===


_LIMIT = 10.0
_HIDDEN = 8
_INTER = 8
# ON/OFF numeric relationship. Both paths feed the SAME INT8 down projection
# (per-token dynamic quant of the activation); the only difference is the value
# being quantized: OFF narrows the gate-up GEMM output and the activation to
# bf16 first (the eager chain), ON keeps them fp32 through the fused operator.
# The delta is therefore bounded by ONE INT8 activation code propagated through
# the down projection -- ``act_step * ||w2 * w2_scale||_1`` -- not by a bf16 ulp;
# the tests derive that bound from the recorded tensors instead of hard-coding a
# relative tolerance. The clamp semantics themselves must agree tightly (bf16
# level), and when the activation is exactly representable both paths agree
# bitwise.
_ACT_EQUIV_TOL = 2**-6
# Relative gate for the end-to-end ON/OFF difference, ANCHORED AT THE OUTPUT PEAK.
# Per-element ratios are not usable here: the delta is a fixed-size INT8-code
# effect, so dividing by a near-zero output element inflates it arbitrarily.
# Basis: the difference is ONE INT8 activation code on the 1/16 of activation
# elements whose code flips, so relative to the output peak it must stay at the
# INT8 granularity scale of the range (1/127 ~= 7.9e-3); 2/127 is the ceiling.
_ON_OFF_PEAK_RATIO_MAX = 2.0 / 127.0


def _quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric INT8 quant: ``(scale [rows, 1], codes [rows, cols])``.
    The single quantization routine shared by the kernel stand-in and the bound derivation, so the ON/OFF bound is"""
    value = x.float()
    scale = (value.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(2.0**-24)
    codes = torch.round(value / scale).clamp(-127, 127).to(torch.int8)
    return scale, codes


class _CpuW8A8Kernels:
    """CPU stand-ins for the kernel ops a W8A8 MLP uses, with call recording."""

    def __init__(self) -> None:
        self.dequant_calls: list[dict[str, Any]] = []
        self.quant_matmul_dtypes: list[torch.dtype] = []
        # Pre-quantization (dequant -> clamped SwiGLU) activations, fp32.
        self.activations: list[torch.Tensor] = []

    def dynamic_quant(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token symmetric INT8 quant (row-wise, scale floored at 2**-24)."""
        scale, codes = _quantize_rows(x)
        return codes, scale

    def quant_matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        transpose: bool,
        weight_scale: torch.Tensor,
        weight_offset: torch.Tensor | None = None,
        pertoken: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """INT8 x INT8 accumulate, returned raw (int32) or dequantized (bf16)."""
        self.quant_matmul_dtypes.append(out_dtype)
        weight_f = weight.float().t() if transpose else weight.float()
        acc = x.float() @ weight_f
        if out_dtype is torch.int32:
            return acc.to(torch.int32)
        out = acc * weight_scale.float().flatten().unsqueeze(0)
        if pertoken is not None:
            out = out * pertoken.float().reshape(-1, 1)
        if bias is not None:
            out = out + bias.float()
        return out.to(out_dtype)

    def dequant_swiglu_quant(
        self,
        x: torch.Tensor,
        weight_scale: torch.Tensor,
        activation_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        quant_scale: torch.Tensor | None = None,
        quant_offset: torch.Tensor | None = None,
        group_index: torch.Tensor | None = None,
        activate_left: bool = True,
        quant_mode: int = 1,
        swiglu_mode: int = 1,
        clamp_limit: float = 0.0,
        glu_alpha: float = 1.0,
        glu_bias: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequant -> clamped SwiGLU -> INT8 requant, matching the op contract.
        The clamp mirrors ``_activation_reference`` of the golden case in ``tests/core/kernels/npu/glm5_next_moe_cases.py``"""
        del bias, quant_scale, quant_offset, glu_alpha, glu_bias
        self.dequant_calls.append(
            dict(
                clamp_limit=clamp_limit,
                swiglu_mode=swiglu_mode,
                quant_mode=quant_mode,
                activate_left=activate_left,
                group_index=group_index,
                weight_scale_shape=tuple(weight_scale.shape),
                shape=tuple(x.shape),
            )
        )
        deq = x.float() * weight_scale.float().flatten().unsqueeze(0)
        deq = deq * activation_scale.float().reshape(-1, 1)
        gate, up = deq.chunk(2, dim=-1)
        # DEVICE FACT reproduced here: the operator's DENSE branch (no
        # ``group_index``) IGNORES ``clamp_limit``, only the GROUPED branch
        # honours it. Modelling the defect is what turns the clamped-reference
        # assertion below into a real regression test: a dense call with a limit
        # set would come back unclamped and fail it.
        effective_limit = clamp_limit if group_index is not None else 0.0
        if 0.0 < effective_limit < 1_000_000.0:
            gate = gate.clamp_max(effective_limit)
            up = up.clamp(-effective_limit, effective_limit)
        act = torch.nn.functional.silu(gate) * up
        self.activations.append(act.detach().clone())
        scale, codes = _quantize_rows(act)
        return codes, scale

    def silu_and_mul(self, x: torch.Tensor) -> torch.Tensor:
        """Unclamped fused activation (reached only when the limit is disabled)."""
        gate, up = x.chunk(2, dim=-1)
        return (torch.nn.functional.silu(gate.float()) * up.float()).to(x.dtype)


@pytest.fixture
def w8a8(monkeypatch: pytest.MonkeyPatch) -> _CpuW8A8Kernels:
    from xllm.python import kernels as kernels_stub

    stand_in = _CpuW8A8Kernels()
    for name in ("dynamic_quant", "quant_matmul", "dequant_swiglu_quant", "silu_and_mul"):
        monkeypatch.setattr(kernels_stub, name, getattr(stand_in, name), raising=False)
    # The fused tail is NPU-only; stub the platform predicate like the rope tests.
    monkeypatch.setattr(v41, "_fused_shared_act_device_supported", lambda x: True)
    monkeypatch.setattr(v41, "_fused_shared_act_active_logged", False)
    monkeypatch.setattr(v41, "_fused_shared_act_fallback_logged", False)
    return stand_in


def _make_mlp(limit: float = _LIMIT) -> DeepseekV41SharedMLP:
    """A deterministic W8A8 shared expert on CPU with the V4.1 geometry/limit."""
    mlp = DeepseekV41SharedMLP(
        SimpleNamespace(tp_size=1, hidden_size=_HIDDEN),
        _INTER,
        torch.bfloat16,
        "cpu",
        skip_tp_reduce=True,
        swiglu_limit=limit,
    )
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        mlp.gate_up_proj.weight.copy_(
            torch.randint(-8, 9, mlp.gate_up_proj.weight.shape, generator=generator, dtype=torch.int8)
        )
        mlp.down_proj.weight.copy_(
            torch.randint(-8, 9, mlp.down_proj.weight.shape, generator=generator, dtype=torch.int8)
        )
        mlp.gate_up_proj.weight_scale.copy_(torch.full_like(mlp.gate_up_proj.weight_scale, 0.125))
        mlp.down_proj.weight_scale.copy_(torch.full_like(mlp.down_proj.weight_scale, 0.125))
    return mlp


class _Harness(v41.DeepseekV41W8A8MoE):
    """The real class and MRO, so ``super()._run_shared_experts`` resolves."""

    def __init__(self, shared: nn.Module, limit: float) -> None:
        nn.Module.__init__(self)
        self.cfg = SimpleNamespace(swiglu_limit=limit)
        self.shared_experts = shared


class _NoScaleMLP(DeepseekV41SharedMLP):
    """A ``DeepseekV41SharedMLP`` whose projections lack a quantization scale.
    Exercises the weight-scale guard without breaking the eager path the fallback must still run (removing the scale from"""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.swiglu_limit = _LIMIT
        self.gate_up_proj = nn.Identity()
        self.down_proj = nn.Identity()

    def forward(self, x: torch.Tensor, tp_reduce_add: torch.Tensor | None = None) -> torch.Tensor:
        del tp_reduce_add
        return x * 2


class _BadGroupedScaleMLP(_NoScaleMLP):
    """A ``DeepseekV41SharedMLP`` whose gate-up scale cannot express the grouped layout.
    The grouped call presents the scale as ``[1, 2H]`` -- one scale row per gate | up pair -- so it is only legal when"""

    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj.out_features = 2 * _INTER
        self.gate_up_proj.weight_scale = torch.ones(2 * _INTER + 2, dtype=torch.float32)
        self.down_proj.in_features = _INTER
        self.down_proj.weight_scale = torch.ones(_HIDDEN, dtype=torch.float32)


def _hidden(rows: int = 4, scale: float = 1.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    return (torch.randn(rows, _HIDDEN, generator=generator) * scale).to(torch.bfloat16)


def _eager_activation(mlp: DeepseekV41SharedMLP, hidden: torch.Tensor) -> torch.Tensor:
    """The eager chain's pre-down activation, in fp32."""
    gate_up = mlp.gate_up_proj(hidden)
    return v41._swiglu_with_clamp(gate_up, mlp.swiglu_limit).float()


def _boundary_hidden() -> torch.Tensor:
    """Rows whose gate/up cross the clamp limit on both sides."""
    return torch.tensor(
        [
            [6.0, -6.0, 2.0, -2.0, 0.5, -0.5, 0.0, 12.0],
            [-9.0, 9.0, 4.0, -4.0, 1.5, -1.5, 0.25, -12.0],
        ],
        dtype=torch.bfloat16,
    )


def _graph_context(state: AclGraphExecutionState | None) -> ForwardContext:
    """A forward context, optionally carrying an ACL graph execution state.
    Only ``execution_state`` is read by ``get_execution_buffer``; the remaining fields are required by the dataclass and"""
    return ForwardContext(
        attention_backend=cast(Any, None),
        device=torch.device("cpu"),
        metadata=cast(Any, None),
        layer_caches=[],
        execution_state=state,
    )


def test_off_is_bitwise_the_eager_shared_forward(
    monkeypatch: pytest.MonkeyPatch,
    w8a8: _CpuW8A8Kernels,
) -> None:
    """A declined guard must reproduce the pre-change ``self.shared_experts(hidden)``."""
    monkeypatch.setattr(v41, "_fused_shared_act_supported", lambda *a, **k: False)
    mlp = _make_mlp()
    moe = _Harness(mlp, _LIMIT)
    hidden = _hidden()

    out = moe._run_shared_experts(hidden)

    assert torch.equal(out, mlp(hidden))
    assert w8a8.dequant_calls == []  # the fused operator never ran


def test_default_is_on(
    monkeypatch: pytest.MonkeyPatch,
    w8a8: _CpuW8A8Kernels,
) -> None:
    """The guards are satisfied, so the default runs the fused grouped path.
    Default on since 2026-10-01 (TPOT -1.69%/-2.02%, throughput +1.83%/+2.06% on the on-device A/B), matching the"""
    eager_calls: list[float] = []
    original = v41._swiglu_with_clamp

    def spy(x: torch.Tensor, limit: float) -> torch.Tensor:
        eager_calls.append(limit)
        return original(x, limit)

    monkeypatch.setattr(v41, "_swiglu_with_clamp", spy)
    mlp = _make_mlp()
    moe = _Harness(mlp, _LIMIT)

    out = moe._run_shared_experts(_hidden())

    assert eager_calls == []  # the ~7-kernel eager chain never ran
    assert len(w8a8.dequant_calls) == 1
    call = w8a8.dequant_calls[0]
    assert call["clamp_limit"] == _LIMIT
    assert call["group_index"] is not None  # grouped, never the clamp-dropping dense form
    assert int(call["group_index"][0]) == call["shape"][0] == 4
    assert call["weight_scale_shape"] == (1, 2 * _INTER)
    assert out.dtype == torch.bfloat16 and out.shape == (4, _HIDDEN)


def test_pre_existing_dense_fused_callers_keep_the_no_clamp_default(
    monkeypatch: pytest.MonkeyPatch,
    w8a8: _CpuW8A8Kernels,
) -> None:
    """The new kwarg leaves the V3.2 dense caller bit-identical (clamp_limit 0.0)."""
    mlp = _make_mlp()
    mlp.forward_dequant_swiglu_quant(_hidden())  # no clamp_limit, as V3.2 calls it

    assert len(w8a8.dequant_calls) == 1
    assert w8a8.dequant_calls[0]["clamp_limit"] == 0.0  # the operator's no-clamp default


def test_on_uses_the_fused_op_with_the_model_clamp_limit(
    monkeypatch: pytest.MonkeyPatch,
    w8a8: _CpuW8A8Kernels,
) -> None:
    """ON runs the fused op with ``clamp_limit == cfg.swiglu_limit``, not the eager chain."""
    eager_calls: list[float] = []
    original = v41._swiglu_with_clamp

    def spy(x: torch.Tensor, limit: float) -> torch.Tensor:
        eager_calls.append(limit)
        return original(x, limit)

    monkeypatch.setattr(v41, "_swiglu_with_clamp", spy)
    mlp = _make_mlp()
    moe = _Harness(mlp, _LIMIT)

    out = moe._run_shared_experts(_hidden())

    assert len(w8a8.dequant_calls) == 1
    call = w8a8.dequant_calls[0]
    assert call["clamp_limit"] == _LIMIT
    # swiglu_mode MUST stay 1: it is the variant the golden clamped-activation
    # case exercises, and it is what makes a device without
    # aclnnDequantSwigluQuantV2 hit the hard TORCH_CHECK(swiglu_mode == 0) at
    # xllm/core/kernels/npu/xllm_ops/dequant_swiglu_quant.cpp:124 -- a loud abort
    # instead of a silent clamp drop on the V1 fallback path.
    assert call["swiglu_mode"] == 1
    assert call["quant_mode"] == 1
    assert call["activate_left"] is True
    # The call MUST be GROUPED, never dense: on device the dense branch ignores
    # clamp_limit, so a dense call here would be the silent clamp drop itself.
    grouped = call["group_index"]
    assert grouped is not None
    # 1-D int64 per-group row COUNTS summing to the row count of the gate-up
    # input (golden case: counts=[2, 0, 1] over 3 rows). One group, every row.
    assert grouped.dtype == torch.int64
    assert grouped.shape == (1,)
    assert int(grouped[0]) == call["shape"][0] == 4
    # ...and the scale carries the matching single-group leading dimension.
    assert call["weight_scale_shape"] == (1, 2 * _INTER)
    assert eager_calls == []  # the ~7-kernel eager chain never ran
    # ON accumulates gate-up in int32 and only dequantizes for the down GEMM.
    assert w8a8.quant_matmul_dtypes == [torch.int32, torch.bfloat16]
    assert out.dtype == torch.bfloat16 and out.shape == (4, _HIDDEN)


def test_grouped_form_is_what_makes_the_clamp_take_effect(
    monkeypatch: pytest.MonkeyPatch,
    w8a8: _CpuW8A8Kernels,
) -> None:
    """The clamp reaches the device only through the groupED call.
    The stand-in reproduces the measured device defect (the DENSE branch ignores ``clamp_limit``), so this pins both"""
    hidden = _boundary_hidden()
    mlp = _make_mlp()
    moe = _Harness(mlp, _LIMIT)

    moe._run_shared_experts(hidden)
    fused = w8a8.activations[0]

    gate_up = mlp.gate_up_proj(hidden)
    gate, up = gate_up.float().chunk(2, dim=-1)
    clamped = torch.nn.functional.silu(gate.clamp_max(_LIMIT)) * up.clamp(-_LIMIT, _LIMIT)
    unclamped = torch.nn.functional.silu(gate) * up
    assert not torch.allclose(clamped, unclamped)  # the fixture really crosses the limit

    torch.testing.assert_close(fused, clamped, rtol=_ACT_EQUIV_TOL, atol=_ACT_EQUIV_TOL)
    assert not torch.allclose(fused, unclamped, rtol=_ACT_EQUIV_TOL, atol=_ACT_EQUIV_TOL), (
        "the fused activation matches the UNCLAMPED reference: the clamp was dropped"
    )

    # Same operator, same clamp_limit, DENSE dispatch (the pre-fix call): the
    # device ignores the limit, so this is the unclamped reference.
    mlp.forward_dequant_swiglu_quant(hidden, clamp_limit=_LIMIT)
    assert len(w8a8.dequant_calls) == 2
    assert w8a8.dequant_calls[1]["group_index"] is None
    torch.testing.assert_close(w8a8.activations[1], unclamped, rtol=_ACT_EQUIV_TOL, atol=_ACT_EQUIV_TOL)


def test_on_parks_group_index_in_the_graph_buffer(
    monkeypatch: pytest.MonkeyPatch,
    w8a8: _CpuW8A8Kernels,
) -> None:
    """The descriptor must be graph-resident, never built (H2D) per call.
    An ACL graph capture may not perform a host->device copy, so the constant ``group_index`` is parked in a graph-owned"""
    mlp = _make_mlp()
    moe = _Harness(mlp, _LIMIT)
    state = AclGraphExecutionState({})

    with forward_context(_graph_context(state)):
        moe._run_shared_experts(_hidden(rows=4))
        moe._run_shared_experts(_hidden(rows=4))

    first, second = (call["group_index"] for call in w8a8.dequant_calls)
    assert first is second  # same resident tensor, so no per-call host->device copy
    assert first.dtype == torch.int64 and first.shape == (1,) and int(first[0]) == 4
    assert ("SHARED_ACT_GROUP_INDEX", first.device, 4) in state.persistent_buffers

    # A different row count must not reuse that buffer: the key carries ``rows``.
    with forward_context(_graph_context(AclGraphExecutionState({}))):
        moe._run_shared_experts(_hidden(rows=2))
    other = w8a8.dequant_calls[2]["group_index"]
    assert other is not first
    assert other.dtype == torch.int64 and other.shape == (1,) and int(other[0]) == 2


def test_on_builds_group_index_inline_without_an_execution_state(
    monkeypatch: pytest.MonkeyPatch,
    w8a8: _CpuW8A8Kernels,
) -> None:
    """Eager degrades to a fresh, correct descriptor -- with or without a context.
    Two eager shapes must both keep the previous behaviour of one correctly valued tensor per call: an eager forward"""
    mlp = _make_mlp()
    moe = _Harness(mlp, _LIMIT)

    # Context present, nothing captured yet (runner eager step).
    with forward_context(_graph_context(None)):
        moe._run_shared_experts(_hidden(rows=4))
        moe._run_shared_experts(_hidden(rows=4))
    # No forward context at all (standalone harness).
    moe._run_shared_experts(_hidden(rows=4))
    moe._run_shared_experts(_hidden(rows=4))

    first, second, third, fourth = (call["group_index"] for call in w8a8.dequant_calls)
    assert first is not second  # eager under a runner builds inline, nothing parked
    assert third is not fourth  # no context: built inline too
    assert len({id(grouped) for grouped in (first, second, third, fourth)}) == 4
    for grouped in (first, second, third, fourth):
        assert grouped.dtype == torch.int64 and grouped.shape == (1,) and int(grouped[0]) == 4


def test_on_matches_off_within_the_derived_int8_bound(
    monkeypatch: pytest.MonkeyPatch,
    w8a8: _CpuW8A8Kernels,
) -> None:
    """ON and OFF agree inside the INT8-activation bound, and the clamp agrees tightly."""
    hidden = _boundary_hidden()
    mlp = _make_mlp()
    moe = _Harness(mlp, _LIMIT)

    real_supported = v41._fused_shared_act_supported
    monkeypatch.setattr(v41, "_fused_shared_act_supported", lambda *a, **k: False)
    off = moe._run_shared_experts(hidden)
    monkeypatch.setattr(v41, "_fused_shared_act_supported", real_supported)
    on = moe._run_shared_experts(hidden)

    # (a) Semantic equality of the activation itself: the fused operator's
    # pre-quant activation must equal the eager clamped chain to bf16 precision
    # (this is what "the clamp limit is forwarded and applied" means).
    assert len(w8a8.activations) == 1
    fused_activation = w8a8.activations[0]
    eager_activation = _eager_activation(mlp, hidden)
    torch.testing.assert_close(fused_activation, eager_activation, rtol=_ACT_EQUIV_TOL, atol=_ACT_EQUIV_TOL)

    # (b) Tight end-to-end bound. Both paths feed the SAME down projection, so
    # with (s_f, q_f) / (s_e, q_e) the per-row INT8 scale/codes of the two
    # activations the triangle inequality gives
    #   |Δout[t,j]| <= |s_f - s_e| * (|q_e| @ |w|)[t,j]
    #                  + max(s_f, s_e) * (differing @ |w|)[t,j]
    # where `differing` is 1 exactly on the elements whose INT8 codes differ
    # (|Δcode| <= 1 there) and `w` is the down weight with its per-output scale
    # folded in. Unlike a worst-case |w| L1 over every element this uses the
    # ACTUAL differing set, so it needs no fudge factor.
    down = mlp.down_proj
    w = down.weight.float().t() * down.weight_scale.float().flatten().unsqueeze(0)
    s_f, q_f = _quantize_rows(fused_activation)
    s_e, q_e = _quantize_rows(eager_activation)
    differing = (q_f != q_e).to(fused_activation.dtype)
    scale_term = (s_f - s_e).abs() * (q_e.float().abs() @ w.abs())
    code_term = torch.maximum(s_f, s_e) * (differing @ w.abs())
    bound = float((scale_term + code_term).max())
    delta = float((on.float() - off.float()).abs().max())
    differing_fraction = float(differing.mean())
    assert delta <= bound, f"ON/OFF max abs delta {delta:.4e} exceeds the tight INT8 bound {bound:.4e}"

    # (b2) Explicit relative gate for a human-readable number, anchored at the
    # output peak (see ``_ON_OFF_PEAK_RATIO_MAX`` for why per-element ratios are
    # not usable).
    peak_ratio = delta / float(off.float().abs().max())
    assert peak_ratio <= _ON_OFF_PEAK_RATIO_MAX, (
        f"ON/OFF max|Δ| = {delta:.4e} is {peak_ratio:.4e} of the output peak "
        f"({peak_ratio * 127:.2f}/127 vs the {_ON_OFF_PEAK_RATIO_MAX * 127:.2f}/127 gate; "
        f"codes differ on {differing_fraction:.1%} of elements)"
    )

    # (c) The boundary rows really do clamp: without the limit the guard declines
    # and the eager chain lands somewhere else (`_make_mlp` is deterministic, so
    # the weights are identical).
    unclamped = _Harness(_make_mlp(limit=0.0), 0.0)._run_shared_experts(hidden)
    assert not torch.allclose(unclamped.float(), off.float())


def test_on_is_bitwise_off_when_the_activation_is_exactly_representable(
    monkeypatch: pytest.MonkeyPatch,
    w8a8: _CpuW8A8Kernels,
) -> None:
    """A bf16-exact activation makes ON and OFF bitwise identical.
    With the up half of the gate-up weight zeroed the activation is exactly 0, which both paths quantize identically --"""
    mlp = _make_mlp()
    with torch.no_grad():
        mlp.gate_up_proj.weight[_INTER:, :].zero_()
    moe = _Harness(mlp, _LIMIT)
    hidden = _hidden()

    real_supported = v41._fused_shared_act_supported
    monkeypatch.setattr(v41, "_fused_shared_act_supported", lambda *a, **k: False)
    off = moe._run_shared_experts(hidden)
    monkeypatch.setattr(v41, "_fused_shared_act_supported", real_supported)
    on = moe._run_shared_experts(hidden)

    assert torch.equal(on, off)
    assert bool((mlp(hidden) == 0).all())


def test_eager_clamp_matches_the_golden_activation_reference() -> None:
    """The eager chain is the operator's clamp contract, and the clamp is not a no-op."""
    gate_up = torch.cat((_boundary_hidden(), _boundary_hidden().flip(-1)), dim=-1)

    got = v41._swiglu_with_clamp(gate_up, _LIMIT).float()
    gate, up = gate_up[..., :_HIDDEN].float(), gate_up[..., _HIDDEN:].float()
    reference = torch.nn.functional.silu(gate.clamp_max(_LIMIT)) * up.clamp(-_LIMIT, _LIMIT)
    plain = torch.nn.functional.silu(gate) * up

    torch.testing.assert_close(got, reference, rtol=2**-8, atol=2**-8)
    assert not torch.allclose(reference, plain)


@pytest.mark.parametrize(
    "case",
    [
        "rank_3",
        "dtype_fp32",
        "no_limit",
        "limit_too_large",
        "not_a_mlp",
        "no_weight_scale",
        "illegal_grouped_scale",
    ],
)
def test_guard_falls_back_to_eager(
    monkeypatch: pytest.MonkeyPatch,
    w8a8: _CpuW8A8Kernels,
    case: str,
) -> None:
    """Every guard failure keeps the eager path and never calls the fused op."""
    mlp = _make_mlp()
    limit = _LIMIT
    shared: nn.Module = mlp
    hidden = _hidden()
    if case == "rank_3":
        hidden = hidden.unsqueeze(0)
    elif case == "dtype_fp32":
        hidden = hidden.float()
    elif case == "no_limit":
        limit = 0.0
    elif case == "limit_too_large":
        limit = 1_000_000.0
    elif case == "not_a_mlp":
        shared = nn.Identity()
    elif case == "no_weight_scale":
        shared = _NoScaleMLP()
    elif case == "illegal_grouped_scale":
        shared = _BadGroupedScaleMLP()

    moe = _Harness(shared, limit)
    out = moe._run_shared_experts(hidden)

    assert w8a8.dequant_calls == []
    assert torch.equal(out, shared(hidden))


def test_logging_is_one_shot(
    monkeypatch: pytest.MonkeyPatch,
    w8a8: _CpuW8A8Kernels,
) -> None:
    spy = _SpyLogger()
    monkeypatch.setattr(v41, "logger", spy)
    monkeypatch.setattr(v41, "_fused_shared_act_active_logged", False)
    monkeypatch.setattr(v41, "_fused_shared_act_fallback_logged", False)
    moe = _Harness(_make_mlp(), _LIMIT)
    hidden = _hidden()

    # guard failure: exactly one warning, no info.
    monkeypatch.setattr(v41, "_fused_shared_act_device_supported", lambda x: False)
    moe._run_shared_experts(hidden)
    moe._run_shared_experts(hidden)
    assert len(spy.warnings) == 1 and "shared-expert" in spy.warnings[0]
    assert spy.infos == []

    # guards satisfied: exactly one info.
    monkeypatch.setattr(v41, "_fused_shared_act_device_supported", lambda x: True)
    moe._run_shared_experts(hidden)
    moe._run_shared_experts(hidden)
    active = [message for message in spy.infos if "shared expert" in message]
    assert len(active) == 1


def test_base_hook_is_the_eager_forward(
    monkeypatch: pytest.MonkeyPatch,
    w8a8: _CpuW8A8Kernels,
) -> None:
    """``DeepseekV4MoE._run_shared_experts`` stays the plain shared expert."""
    mlp = _make_mlp()
    moe = object.__new__(v41.DeepseekV4MoE)
    object.__setattr__(moe, "shared_experts", mlp)
    hidden = _hidden()

    assert torch.equal(v41.DeepseekV4MoE._run_shared_experts(moe, hidden), mlp(hidden))
    assert w8a8.dequant_calls == []
