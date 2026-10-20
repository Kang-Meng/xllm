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

"""Wiring equivalence for the V4.1 RMSNorm + norm_cast + MoE post-norm.

Extracted from the monolithic ``test_deepseek_v41_fused_ops.py``.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

import xllm.python.models.deepseek_v4 as v4
import xllm.python.models.deepseek_v41 as v41
from tests.python._deepseek_v41_fused_testkit import (
    _npu_ready,
    _SpyLogger,
)
from tests.python.test_deepseek_v41_hc import _make_stubbed_layer, _run_layer
from tests.python.test_deepseek_v41_model import _DEEPSEEK_V41_TINY_CONFIG
from xllm.python.model_executor.forward_context import (
    ForwardContext,
    forward_context,
)

# ---------------------------------------------------------------------------
# R8: rms_norm (module level)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# R9: fused RMSNorm+cast MoE norm (always on)
# ---------------------------------------------------------------------------
#
# The decoder layer delegates the MoE FFN RMSNorm to ``DeepseekV4MoE.forward``
# (``post_norm=``) instead of running it in the layer. ``DeepseekV4MoE`` pads
# and all-gathers ``hidden`` across DP BEFORE it takes the gate/expert input, so
# the norm now runs on the gathered stream: ``DeepseekV41W8A8MoE._apply_post_norm``
# fuses it with the gate's fp32 widening via ``rms_norm_cast`` when the guard
# passes, else runs the eager norm once. RMSNorm is row-wise, so this equals
# "norm then gather" per row, which is why the fusion also holds for
# ``dp_size > 1``. Tests marked ``_npu_ready`` need an NPU + the fused-ops lib;
# the rest run on CPU with a mocked op / mocked all-gather.


class _StopForward(Exception):
    """Sentinel used to stop ``DeepseekV4MoE.forward`` right after the gate input."""


class _RecordingMoeStub(v41.DeepseekV41MoE):
    """MoE-shaped stub: skips the expert init and records its call arguments.
    Subclasses the real ``DeepseekV41MoE`` so the decoder layer's ``isinstance(self.mlp, (DeepseekV41MoE,"""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.calls: list[tuple[Any, ...]] = []

    def forward(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        post_norm: nn.Module | None = None,
    ) -> torch.Tensor:
        self.calls.append((hidden, input_ids, post_norm))
        return hidden * 0.5


class _RecordingW8A8Stub(v41.DeepseekV41W8A8MoE):
    """Same recording stub for the W8A8 tier (a separate ``DeepseekV4MoE`` subclass)."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.calls: list[tuple[Any, ...]] = []

    def forward(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        post_norm: nn.Module | None = None,
    ) -> torch.Tensor:
        self.calls.append((hidden, input_ids, post_norm))
        return hidden * 0.5


def _make_norm_cast_layer(seed: int) -> tuple[Any, dict[str, Any], _RecordingMoeStub]:
    """A CPU decoder layer with a real FFN RMSNorm and a recording MoE stub."""
    layer, inputs = _make_stubbed_layer("cpu", seed=seed)
    norm = v41.DeepseekV41RMSNorm(
        inputs["hidden"].shape[-1],
        _DEEPSEEK_V41_TINY_CONFIG["rms_norm_eps"],
        torch.bfloat16,
    )
    with torch.no_grad():
        norm.weight.normal_(mean=1.0, std=0.2)
    layer.post_attention_layernorm = norm
    moe = _RecordingMoeStub()
    layer.mlp = moe
    return layer, inputs, moe


def test_rms_norm_cast_guard_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fused switch needs bf16 activation/weight + op + NPU."""
    op = lambda x, gamma, epsilon=1e-6: (x, x.float())  # noqa: E731 - registered-op stand-in
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    weight = torch.ones(8, dtype=torch.bfloat16)

    monkeypatch.setattr(v41, "_get_rms_norm_cast_op", lambda: None)
    assert v41._rms_norm_cast_supported(x, weight) is False

    monkeypatch.setattr(v41, "_get_rms_norm_cast_op", lambda: op)
    assert v41._rms_norm_cast_supported(x.to(torch.float32), weight) is False
    assert v41._rms_norm_cast_supported(x, weight.to(torch.float32)) is False
    if not _npu_ready():
        # Every non-platform condition holds; a CPU tensor must still stay eager.
        assert v41._rms_norm_cast_supported(x, weight) is False


class _GuardTensor:
    """Duck-typed tensor stand-in so the guard's NPU clauses run on a CPU host.
    The fused switch is device-gated, so a real CPU tensor can never reach the op/shape clauses."""

    def __init__(self, shape: tuple[int, ...] = (4, 8), device_type: str = "npu") -> None:
        self.shape = shape
        self.dtype = torch.bfloat16
        self.device = SimpleNamespace(type=device_type)

    def dim(self) -> int:
        return len(self.shape)

    def size(self, index: int = 0) -> int:
        return self.shape[index]


def test_rms_norm_cast_guard_is_shape_and_device_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guarded predicate is static (dtype/op/device), never keyed on DP.
    ``dp_size`` is deliberately absent: the norm now runs inside the MoE AFTER the DP all-gather, so a DP-gathering tier"""
    assert "dp_size" not in inspect.signature(v41._rms_norm_cast_supported).parameters

    op = lambda x, gamma, epsilon=1e-6: (x, x.float())  # noqa: E731 - registered-op stand-in
    monkeypatch.setattr(v41, "_get_rms_norm_cast_op", lambda: op)
    weight = torch.ones(8, dtype=torch.bfloat16)

    # Non-vacuity: every non-platform clause passes on the fake NPU tensor.
    assert v41._rms_norm_cast_supported(_GuardTensor(), weight) is True
    # Shape / numel hardening.
    assert v41._rms_norm_cast_supported(_GuardTensor(shape=(8,)), weight) is False  # not 2D/3D
    assert v41._rms_norm_cast_supported(_GuardTensor(shape=(4, 8, 8, 8)), weight) is False  # 4D
    assert v41._rms_norm_cast_supported(_GuardTensor(), torch.ones(16, dtype=torch.bfloat16)) is False
    assert v41._rms_norm_cast_supported(_GuardTensor(), torch.ones(1, 8, dtype=torch.bfloat16)) is False
    # Device set matches the other V4.1 fused guards (privateuseone included).
    assert v41._rms_norm_cast_supported(_GuardTensor(device_type="privateuseone"), weight) is True
    assert v41._rms_norm_cast_supported(_GuardTensor(device_type="cpu"), weight) is False


def test_rms_norm_cast_layer_delegates_the_un_normalised_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The layer hands the raw collapsed input + norm module to the MoE.
    It must NOT normalise itself: that would both bypass the fused op and, under DP, normalise the local rows the MoE is"""
    layer, inputs, moe = _make_norm_cast_layer(seed=91)
    norm = layer.post_attention_layernorm
    original_forward = norm.forward
    norm_calls: list[torch.Tensor] = []

    def recording_forward(x: torch.Tensor) -> torch.Tensor:
        norm_calls.append(x.clone())
        return original_forward(x)

    monkeypatch.setattr(norm, "forward", recording_forward)
    _run_layer(layer, inputs)

    assert len(moe.calls) == 1
    hidden_moe, _, post_norm_moe = moe.calls[0]
    assert post_norm_moe is norm
    assert norm_calls == []
    # Non-vacuity: the delegated tensor is really un-normalised.
    with torch.inference_mode():
        normed = original_forward(hidden_moe)
    assert not torch.equal(normed, hidden_moe)


def test_rms_norm_cast_layer_delegates_for_the_w8a8_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The W8A8 MoE tier takes the same delegation branch as the pure-torch one."""
    layer, inputs, _ = _make_norm_cast_layer(seed=95)
    moe = _RecordingW8A8Stub()
    layer.mlp = moe
    norm = layer.post_attention_layernorm
    _run_layer(layer, inputs)

    assert len(moe.calls) == 1
    assert moe.calls[0][2] is norm


def _make_w8a8_moe(
    dp_size: int,
    dp_rank: int = 0,
    gate_logits: Any = None,
) -> v41.DeepseekV41W8A8MoE:
    """A ``DeepseekV41W8A8MoE`` without the heavy expert init.
    ``__new__`` + ``nn.Module.__init__`` keeps the real ``_apply_post_norm`` override (and its ``super()`` chain), while"""
    moe = v41.DeepseekV41W8A8MoE.__new__(v41.DeepseekV41W8A8MoE)
    nn.Module.__init__(moe)
    moe.dp_size = dp_size
    moe.dp_rank = dp_rank
    moe.hash_layer = False
    if gate_logits is not None:
        moe._gate_logits = gate_logits
    return moe


def _dp_forward_context(counts: list[int]) -> ForwardContext:
    metadata = SimpleNamespace(dp_execution_token_counts=counts)
    return ForwardContext(
        attention_backend=MagicMock(),
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=[],
    )


class _FakeAllGather:
    """Rank-major dense stand-in for ``distributed.all_gather``.
    Every rank holds the same local rows here, so the gathered stream is the local tensor replicated ``world_size`` times"""

    @staticmethod
    def all_gather(tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        world_size = int(kwargs.get("world_size", 2))
        return tensor.repeat(world_size, *([1] * (tensor.dim() - 1)))


@pytest.mark.parametrize("dp_size", [2, 4])
def test_v4_moe_fused_norm_cast_runs_after_the_dp_gather(
    monkeypatch: pytest.MonkeyPatch,
    dp_size: int,
) -> None:
    """dp_size in {2,4} + fused op: the gate routes the GATHERED rows' exact fp32.
    The norm now runs inside the MoE after ``all_gather``, so the op's widened output describes the gathered gate/expert"""
    monkeypatch.setattr(v4, "distributed", _FakeAllGather)

    captured: dict[str, torch.Tensor] = {}

    def capture_gate_input(gate_input: torch.Tensor, gate_input_ids: Any) -> torch.Tensor:
        captured["gate_input"] = gate_input
        raise _StopForward

    # Records ``(input, y, gamma, epsilon)``.
    op_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = []

    def fake_op(x: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6):
        y = (x.float() * 0.5).to(x.dtype)
        op_calls.append((x.clone(), y.clone(), gamma, epsilon))
        return y, y.float()

    monkeypatch.setattr(v41, "_rms_norm_cast_supported", lambda x, w: True)
    monkeypatch.setattr(v41, "_get_rms_norm_cast_op", lambda: fake_op)
    monkeypatch.setattr(v41, "_rms_norm_cast_active_logged", False)

    norm = v41.DeepseekV41RMSNorm(8, 1e-6, torch.bfloat16)
    moe = _make_w8a8_moe(dp_size, gate_logits=capture_gate_input)
    hidden = torch.randn(3, 8, dtype=torch.bfloat16)
    gathered = hidden.repeat(dp_size, 1)

    with forward_context(_dp_forward_context([3] * dp_size)), pytest.raises(_StopForward):
        v4.DeepseekV4MoE.forward(moe, hidden, post_norm=norm)

    # The op saw the GATHERED stream (dp_size * local 3 rows), not the local rows.
    assert len(op_calls) == 1
    op_input, op_y, op_gamma, op_epsilon = op_calls[0]
    assert op_input.shape[0] == 3 * dp_size
    assert torch.equal(op_input, gathered)
    assert torch.equal(op_gamma, norm.weight)
    assert op_epsilon == norm.eps
    # ...and the gate routes exactly the op's widened result over those rows.
    gate_input = captured["gate_input"]
    assert gate_input.shape[0] == 3 * dp_size
    assert torch.equal(gate_input, op_y.float())


def test_v4_moe_without_post_norm_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ``post_norm`` -> the V4 path widens the (gathered) hidden itself.
    This keeps the unchanged V4 callers bit-for-bit: with dp_size=2 the gate widens the gathered bf16, exactly as before"""
    monkeypatch.setattr(v4, "distributed", _FakeAllGather)

    captured: dict[str, torch.Tensor] = {}

    def capture_gate_input(gate_input: torch.Tensor, gate_input_ids: Any) -> torch.Tensor:
        captured["gate_input"] = gate_input
        raise _StopForward

    hidden = torch.randn(3, 8, dtype=torch.bfloat16)

    # dp_size=1: the gate widens the local hidden.
    moe_one = SimpleNamespace(dp_size=1, dp_rank=0, _gate_logits=capture_gate_input)
    with pytest.raises(_StopForward):
        v4.DeepseekV4MoE.forward(moe_one, hidden)
    assert torch.equal(captured["gate_input"], hidden.to(torch.float32))

    # dp_size=2: the gate widens the GATHERED hidden (6 rows), as before.
    captured.clear()
    moe_two = SimpleNamespace(dp_size=2, dp_rank=0, _gate_logits=capture_gate_input)
    with forward_context(_dp_forward_context([3, 3])), pytest.raises(_StopForward):
        v4.DeepseekV4MoE.forward(moe_two, hidden)
    expected = torch.cat([hidden, hidden], dim=0).to(torch.float32)
    assert captured["gate_input"].shape[0] == 6
    assert torch.equal(captured["gate_input"], expected)


def test_v4_moe_fused_norm_cast_falls_back_to_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing op -> the eager norm once, bitwise the same result, with one warning."""
    captured: dict[str, torch.Tensor] = {}

    def capture_gate_input(gate_input: torch.Tensor, gate_input_ids: Any) -> torch.Tensor:
        captured["gate_input"] = gate_input
        raise _StopForward

    norm = v41.DeepseekV41RMSNorm(8, 1e-6, torch.bfloat16)
    original_forward = norm.forward
    norm_calls: list[torch.Tensor] = []

    def recording_forward(x: torch.Tensor) -> torch.Tensor:
        norm_calls.append(x.clone())
        return original_forward(x)

    monkeypatch.setattr(norm, "forward", recording_forward)
    spy = _SpyLogger()
    monkeypatch.setattr(v41, "logger", spy)
    monkeypatch.setattr(v41, "_rms_norm_cast_fallback_logged", False)
    monkeypatch.setattr(v41, "_get_rms_norm_cast_op", lambda: None)
    hidden = torch.randn(3, 8, dtype=torch.bfloat16)
    expected = original_forward(hidden).to(torch.float32)

    moe = _make_w8a8_moe(1, gate_logits=capture_gate_input)
    with pytest.raises(_StopForward):
        v4.DeepseekV4MoE.forward(moe, hidden, post_norm=norm)
    assert torch.equal(captured["gate_input"], expected)
    assert len(norm_calls) == 1
    assert len([message for message in spy.warnings if "RMSNorm+cast" in message]) == 1


def test_v4_moe_fused_norm_cast_is_applied_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard on: the fused op runs once and the eager norm module not at all."""
    captured: dict[str, torch.Tensor] = {}

    def capture_gate_input(gate_input: torch.Tensor, gate_input_ids: Any) -> torch.Tensor:
        captured["gate_input"] = gate_input
        raise _StopForward

    norm = v41.DeepseekV41RMSNorm(8, 1e-6, torch.bfloat16)
    original_forward = norm.forward
    norm_calls: list[torch.Tensor] = []

    def recording_forward(x: torch.Tensor) -> torch.Tensor:
        norm_calls.append(x)
        return original_forward(x)

    monkeypatch.setattr(norm, "forward", recording_forward)
    op_calls: list[torch.Tensor] = []

    def fake_op(x: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6):
        op_calls.append(x)
        return x, x.float()

    monkeypatch.setattr(v41, "_rms_norm_cast_supported", lambda x, w: True)
    monkeypatch.setattr(v41, "_get_rms_norm_cast_op", lambda: fake_op)
    monkeypatch.setattr(v41, "_rms_norm_cast_active_logged", False)

    moe = _make_w8a8_moe(1, gate_logits=capture_gate_input)
    hidden = torch.randn(3, 8, dtype=torch.bfloat16)
    with pytest.raises(_StopForward):
        v4.DeepseekV4MoE.forward(moe, hidden, post_norm=norm)

    assert len(op_calls) == 1
    assert norm_calls == []
    # The gate routes the widened op result, never a second normalisation.
    assert torch.equal(captured["gate_input"], op_calls[0].float())


def test_rms_norm_cast_logging_states(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing op warns once; an active path logs once."""
    spy = _SpyLogger()
    monkeypatch.setattr(v41, "logger", spy)
    monkeypatch.setattr(v41, "_rms_norm_cast_active_logged", False)
    monkeypatch.setattr(v41, "_rms_norm_cast_fallback_logged", False)
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    weight = torch.ones(8, dtype=torch.bfloat16)

    # missing op: warn once, no active info.
    monkeypatch.setattr(v41, "_get_rms_norm_cast_op", lambda: None)
    v41._log_rms_norm_cast_fallback(x, weight)
    v41._log_rms_norm_cast_fallback(x, weight)
    assert len([message for message in spy.warnings if "RMSNorm+cast" in message]) == 1
    assert spy.infos == []

    # active path: info once.
    monkeypatch.setattr(v41, "_rms_norm_cast_active_logged", False)
    v41._log_rms_norm_cast_active()
    v41._log_rms_norm_cast_active()
    assert len([message for message in spy.infos if "rms_norm_cast" in message]) == 1


def test_pure_torch_moe_applies_post_norm_before_routing() -> None:
    """The pure-torch MoE norms the delegated input exactly once, before routing."""
    captured: dict[str, torch.Tensor] = {}

    def capture_selection(x: torch.Tensor, input_ids: Any) -> Any:
        captured["x"] = x
        raise _StopForward

    norm = v41.DeepseekV41RMSNorm(8, 1e-6, torch.float32)
    hidden = torch.randn(4, 8, dtype=torch.float32)
    moe = SimpleNamespace(cfg=SimpleNamespace(hidden_size=8), _selection_logits=capture_selection)

    with pytest.raises(_StopForward):
        v41.DeepseekV41MoE.forward(moe, hidden, post_norm=norm)
    assert torch.equal(captured["x"], norm(hidden).reshape(-1, 8))

    # No post_norm: routing sees the raw input unchanged.
    captured.clear()
    with pytest.raises(_StopForward):
        v41.DeepseekV41MoE.forward(moe, hidden)
    assert torch.equal(captured["x"], hidden.reshape(-1, 8))


def test_moe_forwards_accept_post_norm_and_drop_hidden_states_fp32() -> None:
    """Both MoE tiers take the ``post_norm`` module; the old fp32 kwarg is gone."""
    for cls in (v41.DeepseekV41MoE, v4.DeepseekV4MoE):
        parameters = inspect.signature(cls.forward).parameters
        assert "post_norm" in parameters, cls.__name__
        assert parameters["post_norm"].default is None, cls.__name__
        assert "hidden_states_fp32" not in parameters, cls.__name__
