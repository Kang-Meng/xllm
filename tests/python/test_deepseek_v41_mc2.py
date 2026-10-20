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

"""Wiring equivalence for the V4.1 decode-only MC2 expert path.

Extracted from the monolithic ``test_deepseek_v41_fused_ops.py``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence, cast

import pytest
import torch
import xllm.python.kernels as kernels

import xllm.python.distributed as distributed
import xllm.python.models.deepseek_v41 as v41
from xllm.python.model_executor.forward_context import (
    AclGraphCaptureContext,
    ForwardContext,
    forward_context,
)

from .conftest import load_source_module

# === [7] MOE_MC2: decode-only MC2 expert path ===


# The shared conftest replaces ``xllm.python.distributed`` with a stub, so load
# the real collectives module directly to exercise its ``_groups`` fallback. The
# shared loader reuses the instance the other suites loaded: executing the file
# again would re-register the global ``xllm_ops::*`` operators, voiding the
# monkeypatches those suites made through their instance.
_COLLECTIVES_PATH = Path(__file__).parents[2] / "xllm" / "python" / "distributed" / "collectives.py"
collectives = load_source_module(_COLLECTIVES_PATH, "_xllm_collectives_under_test")

_MC2_HIDDEN = 4
_TOPK = 2
_TOKENS = 3
_LOCAL_EXPERTS = 2
_TOTAL_EXPERTS = 4
_MC2_INTER = 3
_DISPATCH_ROWS = 5


def _mc2_hidden(dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.randn(_TOKENS, _MC2_HIDDEN, dtype=dtype, device=torch.device("cpu"))


def _context(
    *,
    is_prefill: bool = False,
    is_chunked_prefill: bool = False,
    is_dummy: bool = False,
    token_counts: Sequence[int] | None = None,
    dp_is_decode: Sequence[int] | None = None,
) -> ForwardContext:
    metadata = SimpleNamespace(
        is_prefill=is_prefill,
        is_chunked_prefill=is_chunked_prefill,
        is_dummy=is_dummy,
        dp_execution_token_counts=list(token_counts) if token_counts is not None else [],
        dp_is_decode=list(dp_is_decode) if dp_is_decode is not None else [],
    )
    return ForwardContext(
        attention_backend=cast(Any, None),
        device=torch.device("cpu"),
        metadata=cast(Any, metadata),
        layer_caches=[],
    )


def _capture_context() -> ForwardContext:
    """A decode context whose ACL-graph capture is ACTIVE (``acl_graph`` set).
    Mirrors ``decode_acl_graph._capture``: the real capture wraps the forward in a context carrying an"""
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        is_dummy=False,
        dp_execution_token_counts=[],
        dp_is_decode=[],
    )
    return ForwardContext(
        attention_backend=cast(Any, None),
        device=torch.device("cpu"),
        metadata=cast(Any, metadata),
        layer_caches=[],
        acl_graph=AclGraphCaptureContext(stream=None, tasks=[]),
    )


class _Mc2Recorder:
    """Call log shared by the stubbed MC2 seams."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.base_calls: list[dict[str, Any]] = []
        self.ep_reduces: list[torch.Tensor] = []
        # Rows the dispatch last saw; combine must echo the same count, which is
        # this rank's local token count (no DP gather anymore).
        self.dispatch_rows = _TOKENS


@pytest.fixture
def mc2(monkeypatch: pytest.MonkeyPatch) -> _Mc2Recorder:
    """Stub every MC2 seam plus the base forward, recording all calls."""
    rec = _Mc2Recorder()

    def fake_dispatch(**kwargs: Any) -> tuple:
        rec.calls.append(("dispatch", kwargs))
        rec.dispatch_rows = int(kwargs["x"].shape[0])
        return (
            torch.randn(_DISPATCH_ROWS, _MC2_HIDDEN, dtype=torch.bfloat16),
            None,
            torch.zeros(rec.dispatch_rows * _TOPK, dtype=torch.int32),
            torch.tensor([2, 3], dtype=torch.int64),
            torch.tensor([2], dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
            None,
        )

    def fake_combine(**kwargs: Any) -> torch.Tensor:
        rec.calls.append(("combine", kwargs))
        rows = rec.dispatch_rows
        return torch.full((rows, _MC2_HIDDEN), 0.5, dtype=torch.bfloat16)

    def fake_dynamic_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rec.calls.append(("dynamic_quant", {"x": x}))
        return torch.zeros(x.shape, dtype=torch.int8), torch.ones(x.shape[0], dtype=torch.float32)

    def fake_dequant_swiglu_quant(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        rec.calls.append(("dequant_swiglu_quant", kwargs))
        return (
            torch.zeros(_DISPATCH_ROWS, _MC2_INTER, dtype=torch.int8),
            torch.ones(_DISPATCH_ROWS, dtype=torch.float32),
        )

    def fake_grouped_matmul(**kwargs: Any) -> torch.Tensor:
        rec.calls.append(("grouped_matmul", kwargs))
        rows = kwargs["x"][0].shape[0]
        if kwargs["output_dtype"] == torch.int32:
            return torch.zeros(rows, 2 * _MC2_INTER, dtype=torch.int32)
        return torch.zeros(rows, _MC2_HIDDEN, dtype=torch.bfloat16)

    def fake_gating(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, None]:
        rec.calls.append(("gating", kwargs))
        rows = int(kwargs["x"].shape[0])
        return (
            torch.full((rows, _TOPK), 0.5, dtype=torch.float32),
            torch.zeros(rows, _TOPK, dtype=torch.int64),
            None,
        )

    def fake_all_gather(tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        rec.calls.append(("all_gather", {"tensor": tensor, **kwargs}))
        world_size = int(kwargs["world_size"])
        # Dense rank-major layout: every rank's block in rank order, each block
        # offset by its rank so a mis-slice is visible.
        return torch.cat([tensor + rank for rank in range(world_size)], dim=0)

    def base_forward(
        self: Any,
        hidden: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        post_norm: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        rec.base_calls.append({"hidden": hidden, "input_ids": input_ids, "post_norm": post_norm})
        return hidden + 1.0

    monkeypatch.setattr(v41, "_moe_mc2_dispatch", fake_dispatch)
    monkeypatch.setattr(v41, "_moe_mc2_combine", fake_combine)
    monkeypatch.setattr(v41, "_moe_mc2_grouped_matmul", fake_grouped_matmul)
    monkeypatch.setattr(kernels, "dynamic_quant", fake_dynamic_quant, raising=False)
    monkeypatch.setattr(kernels, "dequant_swiglu_quant", fake_dequant_swiglu_quant, raising=False)
    monkeypatch.setattr(kernels, "moe_gating_top_k_hash", fake_gating, raising=False)
    monkeypatch.setattr(distributed, "moe_ep_rank", lambda device: 1, raising=False)
    monkeypatch.setattr(distributed, "moe_ep_rank_from_native", lambda device: 1, raising=False)
    monkeypatch.setattr(distributed, "moe_ep_comm_name", lambda device: "hccl-ep0", raising=False)
    monkeypatch.setattr(distributed, "moe_ep_all_reduce", lambda x: rec.ep_reduces.append(x), raising=False)
    monkeypatch.setattr(distributed, "moe_tp_all_reduce", lambda x: rec.ep_reduces.append(x), raising=False)
    monkeypatch.setattr(distributed, "all_gather", fake_all_gather, raising=False)
    monkeypatch.setattr(v41, "_moe_mc2_ops_available", lambda: True)
    monkeypatch.setattr(v41, "_moe_mc2_device_supported", lambda x: True)
    # The production env switches are gone; drive the static seams directly.
    monkeypatch.setattr(v41, "_moe_mc2_allow_draft", lambda: False)
    monkeypatch.setattr(v41.DeepseekV4MoE, "forward", base_forward)
    return rec


def _make_moe(**overrides: Any) -> v41.DeepseekV41W8A8MoE:
    moe = object.__new__(v41.DeepseekV41W8A8MoE)
    moe.cfg = SimpleNamespace(ep_size=2, tp_size=1, n_shared_experts=1, swiglu_limit=10.0)
    moe.hash_layer = False
    moe.w4a8_dynamic = False
    moe.moe_tp_size = 1
    moe.dp_size = 1
    moe.dp_rank = 0
    moe.num_total_experts = _TOTAL_EXPERTS
    moe.num_experts_per_rank = _LOCAL_EXPERTS
    moe.topk = _TOPK
    moe.scoring_func = "sqrtsoftplus"
    moe.routed_scaling = 1.5
    moe.shared_expert_gate_is_loaded = False
    moe.experts_w13 = torch.zeros(_LOCAL_EXPERTS, _MC2_HIDDEN, 2 * _MC2_INTER, dtype=torch.int8)
    moe.experts_w2 = torch.zeros(_LOCAL_EXPERTS, _MC2_INTER, _MC2_HIDDEN, dtype=torch.int8)
    moe.experts_w13_scale = torch.ones(_LOCAL_EXPERTS, 2 * _MC2_INTER, dtype=torch.float32)
    moe.experts_w2_scale = torch.ones(_LOCAL_EXPERTS, _MC2_HIDDEN, dtype=torch.float32)
    moe._gate_logits = lambda gate_input, gate_input_ids: torch.zeros(gate_input.shape[0], _TOTAL_EXPERTS)
    moe._run_shared_experts = lambda hidden: torch.full((hidden.shape[0], _MC2_HIDDEN), 0.25, dtype=hidden.dtype)
    moe.enable_fused_mc2 = True
    for key, value in overrides.items():
        setattr(moe, key, value)
    return moe


def _names(rec: _Mc2Recorder) -> list[str]:
    return [name for name, _ in rec.calls]


def test_default_off() -> None:
    """The fused MC2 switch defaults off in the config dataclass."""
    assert "enable_fused_mc2" in v41.DeepseekV41Config.__dataclass_fields__
    assert v41.DeepseekV41Config.__dataclass_fields__["enable_fused_mc2"].default is False


def test_off_delegates_bitwise(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    moe = _make_moe(enable_fused_mc2=False)
    hidden = _mc2_hidden()
    with forward_context(_context()):
        out = moe.forward(hidden)
    assert torch.equal(out, hidden + 1.0)
    assert len(mc2.base_calls) == 1
    assert mc2.calls == []


def test_on_runs_mc2_chain_in_order(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    moe = _make_moe()
    hidden = _mc2_hidden()
    with forward_context(_context()):
        out = moe.forward(hidden)

    assert mc2.base_calls == []
    assert _names(mc2) == [
        "gating",
        "dispatch",
        "dynamic_quant",
        "grouped_matmul",
        "dequant_swiglu_quant",
        "grouped_matmul",
        "combine",
    ]

    dispatch = mc2.calls[1][1]
    assert dispatch["group_ep"] == "hccl-ep0"
    assert dispatch["ep_world_size"] == 2
    assert dispatch["ep_rank_id"] == 1
    assert dispatch["moe_expert_num"] == _TOTAL_EXPERTS
    assert dispatch["quant_mode"] == 0
    assert dispatch["global_bs"] == _TOKENS * 2
    assert dispatch["expert_token_nums_type"] == 1
    assert dispatch["group_tp"] == "" and dispatch["tp_world_size"] == 1
    assert dispatch["x"] is hidden
    assert dispatch["expert_ids"].dtype == torch.int32
    assert dispatch["expert_scales"].dtype == torch.float32

    gemm1 = mc2.calls[3][1]
    assert gemm1["output_dtype"] == torch.int32
    assert gemm1["split_item"] == 2 and gemm1["group_list_type"] == 1
    assert gemm1["group_list"].tolist() == [2, 3]
    assert gemm1["weight"][0] is moe.experts_w13

    dequant = mc2.calls[4][1]
    assert dequant["clamp_limit"] == 10.0
    assert dequant["quant_mode"] == 1 and dequant["swiglu_mode"] == 1
    assert dequant["activate_left"] is True
    assert dequant["group_index"].tolist() == [2, 3]
    assert dequant["weight_scale"] is moe.experts_w13_scale

    gemm2 = mc2.calls[5][1]
    assert gemm2["output_dtype"] == torch.bfloat16
    assert gemm2["scale"][0] is moe.experts_w2_scale
    assert gemm2["per_token_scale"] is not None

    combine = mc2.calls[6][1]
    assert combine["group_ep"] == "hccl-ep0"
    assert combine["ep_send_counts"] is not None
    assert combine["tp_send_counts"] is not None
    assert combine["expand_scales"] is None
    assert combine["expert_ids"].dtype == torch.int32
    assert combine["global_bs"] == _TOKENS * 2

    # combine_v2 already folded the EP sum; a second reduce would double it.
    assert mc2.ep_reduces == []
    expected = torch.full((_TOKENS, _MC2_HIDDEN), 0.5, dtype=torch.bfloat16) + 0.25
    assert torch.equal(out, expected)


@pytest.mark.parametrize(
    "flags",
    [
        {"is_prefill": True},
        {"is_chunked_prefill": True},
        {"is_dummy": True},
    ],
)
def test_mc2_only_on_decode(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder, flags: dict[str, bool]) -> None:
    moe = _make_moe()
    hidden = _mc2_hidden()
    with forward_context(_context(**flags)):
        out = moe.forward(hidden)
    assert torch.equal(out, hidden + 1.0)
    assert len(mc2.base_calls) == 1
    assert mc2.calls == []


def test_mc2_needs_a_forward_context(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    moe = _make_moe()
    hidden = _mc2_hidden()
    out = moe.forward(hidden)
    assert torch.equal(out, hidden + 1.0)
    assert len(mc2.base_calls) == 1
    assert mc2.calls == []


@pytest.mark.parametrize(
    ("override", "expected_reason"),
    [
        ({"hash_layer": True}, "hash"),
        ({"w4a8_dynamic": True}, "W4A8"),
        ({"moe_tp_size": 2}, "moe_tp_size"),
        # dp>1 is no longer rejected by itself: without the DP layout metadata
        # the run cannot pad/gather/slice, so it declines on that instead.
        ({"dp_size": 2}, "DP token counts"),
        (
            {"cfg": SimpleNamespace(ep_size=1, tp_size=1, n_shared_experts=1, swiglu_limit=10.0)},
            "ep_size",
        ),
    ],
)
def test_guard_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
    mc2: _Mc2Recorder,
    override: dict[str, Any],
    expected_reason: str,
) -> None:
    moe = _make_moe(**override)
    hidden = _mc2_hidden()
    with forward_context(_context()):
        out = moe.forward(hidden)
        reason = moe._mc2_guard_reason(hidden)
    assert reason is not None
    assert expected_reason in reason


def test_guard_shape_dtype_device_ops_communicator(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    moe = _make_moe()
    with forward_context(_context()):
        assert "shape" in moe._mc2_guard_reason(torch.zeros(0, _MC2_HIDDEN, dtype=torch.bfloat16))
        assert "dtype" in moe._mc2_guard_reason(_mc2_hidden(dtype=torch.float16))

        monkeypatch.setattr(v41, "_moe_mc2_ops_available", lambda: False)
        assert "unavailable" in moe._mc2_guard_reason(_mc2_hidden())
        monkeypatch.setattr(v41, "_moe_mc2_ops_available", lambda: True)

        monkeypatch.setattr(distributed, "moe_ep_comm_name", lambda device: None)
        assert "communicator" in moe._mc2_guard_reason(_mc2_hidden())

        monkeypatch.setattr(distributed, "moe_ep_comm_name", lambda device: "hccl-ep0")
        monkeypatch.setattr(v41, "_moe_mc2_device_supported", lambda x: False)
        assert "device" in moe._mc2_guard_reason(_mc2_hidden())


def test_guard_falls_back_without_a_real_moe_ep_group(monkeypatch: pytest.MonkeyPatch) -> None:
    """The un-masked path: no Python or native moe_ep group => guard declines.
    Unlike the ``mc2`` fixture, this does NOT patch ``moe_ep_comm_name``: it only forces the native runtime to look"""
    monkeypatch.setattr(distributed, "moe_ep_comm_name", collectives.moe_ep_comm_name, raising=False)
    monkeypatch.setattr(collectives, "_native_runtime_op", lambda name: None)
    collectives._moe_ep_comm_name_cache.clear()
    collectives._groups.pop(("moe_ep", "cpu"), None)

    base_calls: list[torch.Tensor] = []

    def base_forward(
        self: Any,
        hidden: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        post_norm: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        base_calls.append(hidden)
        return hidden + 1.0

    monkeypatch.setattr(v41, "_moe_mc2_ops_available", lambda: True)
    monkeypatch.setattr(v41, "_moe_mc2_device_supported", lambda x: True)
    monkeypatch.setattr(v41.DeepseekV4MoE, "forward", base_forward)

    moe = _make_moe()
    hidden = _mc2_hidden()
    with forward_context(_context()):
        out = moe.forward(hidden)
        reason = moe._mc2_guard_reason(hidden)

    assert torch.equal(out, hidden + 1.0)
    assert len(base_calls) == 1
    assert reason is not None
    assert "communicator" in reason


def test_guard_uses_native_comm_name_without_python_group(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    """The P0 fix: a native-only moe_ep group satisfies the guard.
    The ``mc2`` fixture pins ``moe_ep_comm_name`` to a constant; this test puts the real function back and stubs only the"""
    monkeypatch.setattr(distributed, "moe_ep_comm_name", collectives.moe_ep_comm_name, raising=False)
    monkeypatch.setattr(collectives, "_native_runtime_op", lambda name: (lambda: "hccl-native"))
    collectives._moe_ep_comm_name_cache.clear()
    collectives._groups.pop(("moe_ep", "cpu"), None)

    moe = _make_moe()
    hidden = _mc2_hidden()
    with forward_context(_context()):
        out = moe.forward(hidden)

    assert mc2.base_calls == []
    assert _names(mc2)[0] == "gating"
    assert mc2.calls[1][1]["group_ep"] == "hccl-native"
    assert torch.equal(out, torch.full((_TOKENS, _MC2_HIDDEN), 0.5, dtype=torch.bfloat16) + 0.25)


def test_mc2_uses_native_ep_rank_without_python_group(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    """The P1 fix: dispatch/combine take the native C++ ``ep_rank``, not 0.
    The ``mc2`` fixture pins ``moe_ep_rank`` to 1; this test puts the real native-first function back and stubs only the"""
    monkeypatch.setattr(distributed, "moe_ep_rank", collectives.moe_ep_rank, raising=False)
    collectives._groups.pop(("moe_ep", "cpu"), None)

    def native_op(name: str) -> Any:
        if name == "moe_ep_rank":
            return lambda: 3
        return None

    monkeypatch.setattr(collectives, "_native_runtime_op", native_op)

    moe = _make_moe()
    hidden = _mc2_hidden()
    with forward_context(_context()):
        out = moe.forward(hidden)

    dispatch = next(kwargs for name, kwargs in mc2.calls if name == "dispatch")
    combine = next(kwargs for name, kwargs in mc2.calls if name == "combine")
    assert dispatch["ep_rank_id"] == 3
    assert combine["ep_rank_id"] == 3
    assert torch.equal(out, torch.full((_TOKENS, _MC2_HIDDEN), 0.5, dtype=torch.bfloat16) + 0.25)


def test_mc2_ep_rank_falls_back_to_zero_when_native_absent(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    """A build without the binding (the already-compiled remote .so) still runs:
    the rank silently falls back to the Python ``_groups``/0 chain and must not raise from the guard or the forward."""
    monkeypatch.setattr(distributed, "moe_ep_rank", collectives.moe_ep_rank, raising=False)
    monkeypatch.setattr(collectives, "_native_runtime_op", lambda name: None)
    collectives._groups.pop(("moe_ep", "cpu"), None)

    moe = _make_moe()
    hidden = _mc2_hidden()
    with forward_context(_context()):
        out = moe.forward(hidden)

    dispatch = next(kwargs for name, kwargs in mc2.calls if name == "dispatch")
    assert dispatch["ep_rank_id"] == 0
    assert torch.equal(out, torch.full((_TOKENS, _MC2_HIDDEN), 0.5, dtype=torch.bfloat16) + 0.25)


def test_fallback_logs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v41, "_moe_mc2_fallback_logged", False)
    logged: list[str] = []
    monkeypatch.setattr(v41.logger, "warning", lambda msg: logged.append(msg))
    v41._log_moe_mc2_fallback("not a decode step", enabled=True)
    v41._log_moe_mc2_fallback("a later failure", enabled=True)
    assert len(logged) == 1
    assert "not a decode step" in logged[0]
    # The target prefix stays distinguishable from the DSpark-draft one.
    assert "for the target" in logged[0]


def test_draft_fallback_has_its_own_latch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix-1: the draft exit logs once even after the target latch was consumed.
    The shared process latch used to swallow the draft reason because target warmup fired first; the dedicated latch must"""
    monkeypatch.setattr(v41, "_moe_mc2_fallback_logged", False)
    monkeypatch.setattr(v41, "_moe_mc2_draft_fallback_logged", False)
    logged: list[str] = []
    monkeypatch.setattr(v41.logger, "warning", lambda msg: logged.append(msg))

    v41._log_moe_mc2_fallback("target warmup reason", enabled=True)
    v41._log_moe_mc2_draft_fallback(_DRAFT_REASON, enabled=True)
    assert len(logged) == 2
    assert "for the target" in logged[0] and "target warmup reason" in logged[0]
    assert "DSpark draft (Fix-1, target keeps MC2)" in logged[1]
    assert _DRAFT_REASON in logged[1]

    # Each helper stays one-shot on its OWN latch, in either order.
    v41._log_moe_mc2_fallback("a later target failure", enabled=True)
    v41._log_moe_mc2_draft_fallback("a later draft failure", enabled=True)
    assert len(logged) == 2


def test_draft_fallback_requires_enabled_mc2(monkeypatch: pytest.MonkeyPatch) -> None:
    """No MC2 request, no draft exit log (and the latch stays unset)."""
    monkeypatch.setattr(v41, "_moe_mc2_draft_fallback_logged", False)
    logged: list[str] = []
    monkeypatch.setattr(v41.logger, "warning", lambda msg: logged.append(msg))
    v41._log_moe_mc2_draft_fallback(_DRAFT_REASON, enabled=False)
    assert logged == []
    assert v41._moe_mc2_draft_fallback_logged is False


def test_forward_routes_draft_and_target_to_separate_logs(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    """Both latches are independent: the draft cannot starve the target log."""
    monkeypatch.setattr(v41, "_moe_mc2_fallback_logged", False)
    monkeypatch.setattr(v41, "_moe_mc2_draft_fallback_logged", False)
    logged: list[str] = []
    monkeypatch.setattr(v41.logger, "warning", lambda msg: logged.append(msg))

    draft = _make_moe(dp_size=4, dp_rank=1, _moe_is_dspark_draft=True)
    target = _make_moe(dp_size=4, dp_rank=1)
    layout = dict(token_counts=[1, 5, 1, 1], dp_is_decode=[1, 1, 1, 1])
    hidden = torch.randn(5, _MC2_HIDDEN, dtype=torch.bfloat16)

    with forward_context(_context(**layout)):
        assert torch.equal(draft.forward(hidden), hidden + 1.0)
    assert len(logged) == 1
    assert "DSpark draft (Fix-1, target keeps MC2)" in logged[0]
    assert v41._moe_mc2_fallback_logged is False
    assert v41._moe_mc2_draft_fallback_logged is True

    # A target MoE (no draft flag) reaches the DP token-count clause and logs
    # through the target latch; the draft's latch never masks it.
    with forward_context(_context()):
        assert torch.equal(target.forward(hidden), hidden + 1.0)
    assert len(logged) == 2
    assert "for the target" in logged[1] and "DP token counts unavailable" in logged[1]
    assert v41._moe_mc2_fallback_logged is True


# --- dp>1: local-row MC2 dispatch/combine (no DP gather, no slice) -----------


def test_mc2_dp1_keeps_the_single_replica_chain(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    """dp=1 must stay byte-for-byte the previous slice: local rows, explicit
    global_bs, no mask, no gather, no slice."""
    moe = _make_moe(dp_size=1)
    hidden = _mc2_hidden()
    with forward_context(_context()):
        out = moe.forward(hidden)

    assert "all_gather" not in _names(mc2)
    dispatch = next(kwargs for name, kwargs in mc2.calls if name == "dispatch")
    combine = next(kwargs for name, kwargs in mc2.calls if name == "combine")
    assert dispatch["global_bs"] == _TOKENS * 2
    assert dispatch["x_active_mask"] is None
    assert combine["global_bs"] == _TOKENS * 2
    assert combine["x_active_mask"] is None
    assert out.shape == (_TOKENS, _MC2_HIDDEN)


def test_mc2_dp_local_rows_no_gather_no_slice(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    """dp>1 mirrors the C++ mode-0 EP2 branch: local rows only.
    The post-norm, gate, dispatch and shared expert all run on THIS rank's rows; there is no DP pad/all-gather and no"""
    moe = _make_moe(dp_size=3, dp_rank=1)
    token_counts = [2, 2, 2]
    local_rows = token_counts[moe.dp_rank]
    hidden = torch.randn(local_rows, _MC2_HIDDEN, dtype=torch.bfloat16)

    post_norm_rows: list[int] = []

    def fake_post_norm(hidden: torch.Tensor, post_norm: Any) -> tuple[torch.Tensor, torch.Tensor]:
        post_norm_rows.append(int(hidden.shape[0]))
        return hidden, hidden.to(torch.float32)

    monkeypatch.setattr(moe, "_apply_post_norm", fake_post_norm)

    with forward_context(_context(token_counts=token_counts, dp_is_decode=[1, 1, 1])):
        out = moe.forward(hidden, post_norm=object())

    # No DP collective: the chain is the single-replica one, on local rows.
    assert "all_gather" not in _names(mc2)
    assert _names(mc2) == [
        "gating",
        "dispatch",
        "dynamic_quant",
        "grouped_matmul",
        "dequant_swiglu_quant",
        "grouped_matmul",
        "combine",
    ]
    # post_norm sees exactly this rank's rows, not a gathered block.
    assert post_norm_rows == [local_rows]

    dispatch = next(kwargs for name, kwargs in mc2.calls if name == "dispatch")
    assert dispatch["x"].shape == (local_rows, _MC2_HIDDEN)
    assert dispatch["expert_ids"].shape[0] == local_rows
    assert dispatch["x_active_mask"] is None
    assert dispatch["global_bs"] == 0

    combine = next(kwargs for name, kwargs in mc2.calls if name == "combine")
    assert combine["global_bs"] == 0
    assert combine["x_active_mask"] is None

    # No slice: combine returns exactly this rank's rows.
    assert out.shape == (local_rows, _MC2_HIDDEN)
    expected = torch.full((local_rows, _MC2_HIDDEN), 0.5, dtype=torch.bfloat16) + 0.25
    assert torch.equal(out, expected)
    # combine_v2 scatters+accumulates; no explicit EP reduce may run.
    assert mc2.ep_reduces == []
    assert mc2.base_calls == []


def test_mc2_dp_empty_rank_falls_back_on_all_ranks(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    """The P0 fix: an empty DP rank makes dispatch receive ZERO rows.
    ``npu_moe_distribute_dispatch_v2`` rejects ``bs <= 0`` in infershape"""
    moe = _make_moe(dp_size=4, dp_rank=0)
    token_counts = [0, 3, 1, 2]
    hidden = torch.randn(0, _MC2_HIDDEN, dtype=torch.bfloat16)

    with forward_context(_context(is_dummy=True, token_counts=token_counts, dp_is_decode=[1, 1, 1, 1])):
        out = moe.forward(hidden)
        reason = moe._mc2_guard_reason(hidden)

    assert reason == "an empty DP rank is present"


def test_mc2_dp_empty_rank_decides_like_a_nonempty_peer(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    """The P0 symmetry: a rank-local ``is_dummy`` must not flip the collective.
    An empty rank (``is_dummy=True``, ``token_counts[dp_rank] == 0``) and a non-empty peer see the same host-planned"""
    token_counts = [0, 3, 1, 2]
    layout = dict(token_counts=token_counts, dp_is_decode=[1, 1, 1, 1])

    empty = _make_moe(dp_size=4, dp_rank=0)
    peer = _make_moe(dp_size=4, dp_rank=1)

    with forward_context(_context(is_dummy=True, **layout)):
        empty_reason = empty._mc2_guard_reason(torch.randn(0, _MC2_HIDDEN, dtype=torch.bfloat16))
    with forward_context(_context(is_dummy=False, **layout)):
        peer_reason = peer._mc2_guard_reason(torch.randn(3, _MC2_HIDDEN, dtype=torch.bfloat16))

    # Same reason on both: no rank runs the collective while another skips it.
    assert empty_reason == "an empty DP rank is present"
    assert peer_reason == empty_reason


def test_mc2_dp_rank_above_operator_limit_falls_back(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    """P1-b: a per-rank token count above the A2 ``bs <= 256`` bound declines.
    The operator tiling caps the per-rank batch at 256 (``..._tiling_arch22.cpp:507-510``); ``dp_execution_token_counts``"""
    moe = _make_moe(dp_size=2, dp_rank=0)
    hidden = torch.randn(2, _MC2_HIDDEN, dtype=torch.bfloat16)

    with forward_context(_context(token_counts=[257, 2], dp_is_decode=[1, 1])):
        out = moe.forward(hidden)
        reason = moe._mc2_guard_reason(hidden)

    assert reason is not None and "256" in reason

    # Exactly at the bound stays on the MC2 path (uniform, as the guard requires).
    with forward_context(_context(token_counts=[256, 256], dp_is_decode=[1, 1])):
        assert moe._mc2_guard_reason(hidden) is None


def test_mc2_guard_declines_on_uneven_dp_rows(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    """Defensive: a non-uniform DP layout breaks ``global_bs=0`` -> fall back.
    The ACL-graph runner publishes uniform padded counts (``decode_acl_graph.py:1558-1560``), so the tested"""
    moe = _make_moe(dp_size=3, dp_rank=1)
    hidden = torch.randn(2, _MC2_HIDDEN, dtype=torch.bfloat16)

    def layout(counts: list[int]) -> ForwardContext:
        return _context(token_counts=counts, dp_is_decode=[1, 1, 1])

    # Non-zero but uneven: decline with the exact reason + base fallback.
    with forward_context(layout([3, 2, 4])):
        reason = moe._mc2_guard_reason(hidden)
        out = moe.forward(hidden)
    assert reason == "uneven DP rows (4 vs 2) break global_bs=0"

    # Uniform: the clause does not trip; the chain engages.
    with forward_context(layout([2, 2, 2])):
        assert moe._mc2_guard_reason(hidden) is None

    # Pre-existing bounds are unchanged / not relaxed: the zero and >256 clauses
    # still win first, and uniform-at-bound still engages.
    with forward_context(layout([0, 2, 2])):
        assert moe._mc2_guard_reason(hidden) == "an empty DP rank is present"
    with forward_context(layout([257, 2, 2])):
        over = moe._mc2_guard_reason(hidden)
        assert over is not None and "256" in over
    with forward_context(layout([256, 256, 256])):
        assert moe._mc2_guard_reason(hidden) is None


def test_mc2_guard_declines_when_native_ep_rank_unavailable(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    """P1-a: a communicator name without a native rank must not silently route to 0.
    ``moe_ep_rank`` collapses native/``_groups``/0 sources, so if the comm name resolves but the native rank does not,"""
    monkeypatch.setattr(distributed, "moe_ep_rank_from_native", lambda device: None, raising=False)
    moe = _make_moe()
    hidden = _mc2_hidden()

    with forward_context(_context()):
        out = moe.forward(hidden)
        reason = moe._mc2_guard_reason(hidden)

    assert reason == "moe_ep rank unavailable from native source"


def test_mc2_dp_empty_rank_still_declines_when_a_peer_is_not_decode(
    monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder
) -> None:
    """An empty rank does not force MC2 when a peer is on a dummy/prefill step."""
    empty = _make_moe(dp_size=2, dp_rank=0)
    with forward_context(_context(is_dummy=True, token_counts=[0, 2], dp_is_decode=[1, 0])):
        reason = empty._mc2_guard_reason(torch.randn(0, _MC2_HIDDEN, dtype=torch.bfloat16))
    assert reason is not None and "all DP ranks" in reason


def test_mc2_guard_requires_all_dp_ranks_decode(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    """An uneven DP batch leaves a peer on a dummy step; the collective must
    decline on every rank, so the decision keys on the shared ``dp_is_decode``."""
    moe = _make_moe(dp_size=2, dp_rank=0)
    hidden = torch.randn(2, _MC2_HIDDEN, dtype=torch.bfloat16)

    with forward_context(_context(token_counts=[2, 2], dp_is_decode=[1, 0])):
        reason = moe._mc2_guard_reason(hidden)
        out = moe.forward(hidden)
    assert reason is not None and "all DP ranks" in reason

    with forward_context(_context(token_counts=[2, 2], dp_is_decode=[1, 1])):
        assert moe._mc2_guard_reason(hidden) is None


def test_moe_mc2_all_dp_ranks_decode_mirrors_cpp() -> None:
    """``all_dp_ranks_are_decode`` mirrors ``dp_utils.cpp``."""
    assert v41._moe_mc2_all_dp_ranks_decode(
        SimpleNamespace(dp_is_decode=[1, 1, 1], dp_execution_token_counts=[1, 1, 1])
    )
    assert not v41._moe_mc2_all_dp_ranks_decode(
        SimpleNamespace(dp_is_decode=[1, 1, 0], dp_execution_token_counts=[1, 1, 0])
    )
    # No published flags: fall back to the single-replica test on token counts.
    assert v41._moe_mc2_all_dp_ranks_decode(SimpleNamespace(dp_is_decode=[], dp_execution_token_counts=[]))
    assert v41._moe_mc2_all_dp_ranks_decode(SimpleNamespace(dp_is_decode=[], dp_execution_token_counts=[7]))
    assert not v41._moe_mc2_all_dp_ranks_decode(SimpleNamespace(dp_is_decode=[], dp_execution_token_counts=[7, 7]))


# --- Fix-1: DSpark-draft MC2 veto (structural uneven rows) -------------------

#: Escape hatch that waives the draft veto (see ``v41._moe_mc2_allow_draft``).
_DRAFT_REASON = "DSpark draft MoE (uneven per-rank rows break global_bs=0)"


def test_moe_mc2_allow_draft_defaults_off() -> None:
    assert v41._moe_mc2_allow_draft() is False


def test_mc2_draft_flag_declines_on_every_rank(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    """The static draft flag vetoes the collective identically on all ranks.
    A busy rank (5 rows = one sequence's whole block) and an empty peer (1 dummy row) must reach the SAME branch: the"""
    busy = _make_moe(dp_size=4, dp_rank=1, _moe_is_dspark_draft=True)
    empty = _make_moe(dp_size=4, dp_rank=0, _moe_is_dspark_draft=True)
    layout = dict(token_counts=[1, 5, 1, 1], dp_is_decode=[1, 1, 1, 1])

    with forward_context(_context(**layout)):
        busy_reason = busy._mc2_guard_reason(torch.randn(5, _MC2_HIDDEN, dtype=torch.bfloat16))
    with forward_context(_context(**layout)):
        empty_reason = empty._mc2_guard_reason(torch.randn(1, _MC2_HIDDEN, dtype=torch.bfloat16))

    assert busy_reason == _DRAFT_REASON
    assert empty_reason == _DRAFT_REASON

    # The clause is static: it is observable even without a forward context.
    assert busy._mc2_guard_reason(torch.randn(5, _MC2_HIDDEN, dtype=torch.bfloat16)) == _DRAFT_REASON

    # And the forward falls back to the base path without touching an MC2 seam.
    hidden = torch.randn(5, _MC2_HIDDEN, dtype=torch.bfloat16)
    with forward_context(_context(**layout)):
        out = busy.forward(hidden)


def test_mc2_target_without_flag_keeps_original_guard(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    """A target MoE (no flag) is byte-for-byte the pre-Fix-1 guard behavior."""
    target = _make_moe()
    assert getattr(target, "_moe_is_dspark_draft", False) is False
    hidden = _mc2_hidden()

    with forward_context(_context()):
        assert target._mc2_guard_reason(hidden) is None
        out = target.forward(hidden)

    # The full MC2 chain still engages: the new clause added nothing for targets.
    assert mc2.base_calls == []
    assert _names(mc2)[0] == "gating"
    assert torch.equal(out, torch.full((_TOKENS, _MC2_HIDDEN), 0.5, dtype=torch.bfloat16) + 0.25)

    # An unrelated fallback still reports its own reason verbatim, not the draft one.
    hashed = _make_moe(hash_layer=True)
    with forward_context(_context()):
        reason = hashed._mc2_guard_reason(hidden)
    assert reason is not None and "hash" in reason
    assert _DRAFT_REASON not in reason


def test_mc2_draft_veto_waived_by_escape_hatch(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    """An enabled draft escape hatch skips the draft clause."""
    monkeypatch.setattr(v41, "_moe_mc2_allow_draft", lambda: True)
    moe = _make_moe(_moe_is_dspark_draft=True)
    hidden = _mc2_hidden()

    with forward_context(_context()):
        assert moe._mc2_guard_reason(hidden) is None
        out = moe.forward(hidden)

    assert mc2.base_calls == []
    assert _names(mc2) == [
        "gating",
        "dispatch",
        "dynamic_quant",
        "grouped_matmul",
        "dequant_swiglu_quant",
        "grouped_matmul",
        "combine",
    ]
    assert torch.equal(out, torch.full((_TOKENS, _MC2_HIDDEN), 0.5, dtype=torch.bfloat16) + 0.25)


def test_mc2_forward_asserts_local_expert_count(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    """M4: a shard claiming more experts than ``experts_w13`` holds fails loud."""
    moe = _make_moe(num_experts_per_rank=3)  # experts_w13 still has _LOCAL_EXPERTS rows
    hidden = _mc2_hidden()

    with forward_context(_context()), pytest.raises(RuntimeError) as err:
        moe.forward(hidden)

    message = str(err.value)
    assert "experts_w13" in message
    assert str(moe.experts_w13.shape[0]) in message
    assert "3" in message
    # No silent completion: the grouped GEMM never ran.
    assert "grouped_matmul" not in _names(mc2)


def test_mc2_forward_asserts_grouped_rows_fit_dispatch(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    """M4: expert groups covering more rows than the dispatch produced fail loud."""
    moe = _make_moe()
    hidden = _mc2_hidden()

    def short_dispatch(**kwargs: Any) -> tuple:
        rec = mc2
        rec.calls.append(("dispatch", kwargs))
        rec.dispatch_rows = int(kwargs["x"].shape[0])
        # Only 2 dispatched rows, but the expert groups cover 2 + 3 = 5 rows.
        return (
            torch.randn(2, _MC2_HIDDEN, dtype=torch.bfloat16),
            None,
            torch.zeros(2 * _TOPK, dtype=torch.int32),
            torch.tensor([2, 3], dtype=torch.int64),
            torch.tensor([2], dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
            None,
        )

    monkeypatch.setattr(v41, "_moe_mc2_dispatch", short_dispatch)
    with forward_context(_context()), pytest.raises(RuntimeError) as err:
        moe.forward(hidden)

    message = str(err.value)
    assert "expert groups" in message
    assert "5" in message
    assert "expand_x" in message
    assert "grouped_matmul" not in _names(mc2)


def test_mc2_grouped_rows_check_is_capture_gated(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    """M4 capture-safety: the group-fit check reads tensor content, so it must
    skip ACL-graph capture."""
    moe = _make_moe()
    hidden = _mc2_hidden()

    def overfull_dispatch(**kwargs: Any) -> tuple:
        mc2.calls.append(("dispatch", kwargs))
        mc2.dispatch_rows = int(kwargs["x"].shape[0])
        # The dispatch produced _TOKENS rows, but the local expert groups claim
        # 2 + 3 = 5 rows -- a contract violation the check must catch.
        return (
            torch.randn(_TOKENS, _MC2_HIDDEN, dtype=torch.bfloat16),
            None,
            torch.zeros(_TOKENS * _TOPK, dtype=torch.int32),
            torch.tensor([2, 3], dtype=torch.int64),
            torch.tensor([2], dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
            None,
        )

    monkeypatch.setattr(v41, "_moe_mc2_dispatch", overfull_dispatch)

    reductions: list[int] = []
    real_sum = torch.Tensor.sum

    def spy_sum(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        reductions.append(1)
        return real_sum(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sum", spy_sum)

    # Eager (no acl_graph): the value check runs -> one reduction + fail loud.
    with forward_context(_context()), pytest.raises(RuntimeError) as err:
        moe.forward(hidden)
    assert "expert groups cover" in str(err.value)
    assert len(reductions) == 1
    assert "grouped_matmul" not in _names(mc2)

    # Capture active: the value check is skipped -> no reduction, chain completes.
    reductions.clear()
    mc2.calls.clear()
    with forward_context(_capture_context()):
        out = moe.forward(hidden)
    assert reductions == []
    assert _names(mc2)[-1] == "combine"
    assert out.shape == (_TOKENS, _MC2_HIDDEN)


def test_mc2_expert_count_check_stays_active_in_capture(monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder) -> None:
    """The host-only ``experts_w13`` shape check is NOT gated (capture-safe)."""
    moe = _make_moe(num_experts_per_rank=3)  # experts_w13 still has _LOCAL_EXPERTS rows
    hidden = _mc2_hidden()

    with forward_context(_capture_context()), pytest.raises(RuntimeError) as err:
        moe.forward(hidden)

    assert "experts_w13" in str(err.value)
    # No device work ran: the shape check fires before the grouped GEMM.
    assert "grouped_matmul" not in _names(mc2)


def test_mc2_dispatch_and_combine_share_the_expected_global_bs(
    monkeypatch: pytest.MonkeyPatch, mc2: _Mc2Recorder
) -> None:
    """The matched dispatch/combine pair is fed ONE host ``global_bs``.
    The former M4 "dispatch == combine" guard was a tautology: it compared one local variable with itself (the alias was"""

    # dp == 1: explicit ``local_bs * ep_size``; dp > 1: the auto ``0``. dp > 1
    # uses uniform host-planned counts (the ACL-graph layout; see the guard).
    for dp_size, rows, layout, expected in (
        (1, _TOKENS, None, _TOKENS * 2),
        (3, 2, dict(token_counts=[2, 2, 2], dp_is_decode=[1, 1, 1]), 0),
    ):
        moe = _make_moe(dp_size=dp_size, dp_rank=1 if dp_size > 1 else 0)
        hidden = torch.randn(rows, _MC2_HIDDEN, dtype=torch.bfloat16)
        with forward_context(_context() if layout is None else _context(**layout)):
            moe.forward(hidden)
        dispatch = next(kwargs for name, kwargs in mc2.calls if name == "dispatch")
        combine = next(kwargs for name, kwargs in mc2.calls if name == "combine")
        assert dispatch["global_bs"] == combine["global_bs"] == expected
        mc2.calls.clear()


def test_draft_marking_is_instance_scoped() -> None:
    """The flag is a per-instance attribute, never a class attribute.
    The draft reuses the target's MoE classes, so a class attribute would veto the target too."""
    assert not hasattr(v41.DeepseekV4MoE, "_moe_is_dspark_draft")
