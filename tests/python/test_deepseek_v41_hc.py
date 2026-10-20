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

"""Wiring equivalence for the V4.1 hyper-connection (hc_post + hc_pre).

Extracted from the monolithic ``test_deepseek_v41_fused_ops.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import xllm.python.models.deepseek_v41 as v41
from tests.python._deepseek_v41_fused_testkit import (
    _HC_PRE_MODES,
    _MODE_EAGER,
    _MODE_FUSED,
    _npu_ready,
    _SpyLogger,
)
from tests.python.test_deepseek_v41_model import _DEEPSEEK_V41_TINY_CONFIG

# ---------------------------------------------------------------------------
# R3: hc_post (decoder-layer level, stubbed sub-blocks)
# ---------------------------------------------------------------------------


def test_va_fused_reuses_native_pre_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    """The next sublayer must receive the native gate, without BF16 reprojection."""
    module_path = Path(v41.__file__).parents[1] / "kernels_npu" / "hyper_connection.py"
    spec = importlib.util.spec_from_file_location("_hc_va_fused_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    layer, inputs = _make_stubbed_layer("cpu", seed=83)
    hc = layer.hc
    hidden = inputs["hidden"]
    expected = (
        torch.randn(hidden.size(0), hidden.size(-1)),
        torch.randn(hidden.size(0), hc.hc_mult),
        torch.randn(hidden.size(0), hc.hc_mult, hc.hc_mult),
        torch.randn(hidden.size(0), hc.hc_mult),
    )
    calls: list[tuple[Any, ...]] = []

    def native_op(*args: Any) -> tuple[torch.Tensor, ...]:
        calls.append(args)
        return expected

    monkeypatch.setattr(
        module, "torch", SimpleNamespace(ops=SimpleNamespace(xllm_ops=SimpleNamespace(hc_pre_va=native_op)))
    )
    monkeypatch.setattr(v41, "_get_hc_pre_fused_op", module._resolve_hc_pre_va)
    y, pre, post, comb = hc.hc_pre_fused(hidden, hc.hc_attn_fn, hc.hc_attn_scale, hc.hc_attn_base, inputs["pre_mix"])
    assert y is expected[0] and pre is expected[3]
    assert post is expected[1] and comb is expected[2]
    assert len(calls) == 1 and calls[0][4] is inputs["pre_mix"]


@pytest.mark.skipif(
    not _npu_ready() or getattr(torch.ops.xllm_ops, "hc_pre_va", None) is None,
    reason="needs NPU and the native VA binding",
)
@pytest.mark.parametrize("tokens", [1, 6, 33])
def test_va_gates_match_fp32_reference(tokens: int) -> None:
    """Exercise decode, speculative verification, and prefill shapes on device."""
    torch.manual_seed(83)
    width, hc = 5120, 4
    hidden = torch.randn(tokens, hc, width, dtype=torch.bfloat16)
    weight = torch.randn(24, hc * width) / (hc * width) ** 0.5
    scale = torch.tensor([1.0, 1.0, 1.0])
    base = torch.randn(24) * 0.1
    external_pre = torch.rand(tokens, hc)
    flattened = hidden.float().flatten(-2)
    # Compare against the FP32 formula, without the old BF16 weight/output
    # rounding. The deployed VA package stays within 1e-4 of this reference.
    projection = F.linear(flattened, weight)
    projection *= torch.rsqrt(flattened.square().mean(-1, keepdim=True) + 1e-20)
    expected_pre = torch.sigmoid(projection[:, :hc] * scale[0] + base[:hc]) + 1e-6
    expected_y = (external_pre.unsqueeze(-1) * hidden.float()).sum(-2).to(hidden.dtype)
    y, post, comb, pre = torch.ops.xllm_ops.hc_pre_va(
        hidden.npu(), weight.npu(), scale.npu(), base.npu(), external_pre.npu(), hc, 20, 1e-20, 1e-6
    )
    torch.testing.assert_close(pre.cpu(), expected_pre, rtol=0, atol=1e-4)
    torch.testing.assert_close(y.cpu(), expected_y, rtol=4e-3, atol=4e-3)
    assert post.dtype == comb.dtype == pre.dtype == torch.float32


class _StubAttention(nn.Module):
    def forward(self, x: torch.Tensor, positions: torch.Tensor, cos_sin_cache: Any) -> torch.Tensor:
        return x * 0.9


class _StubNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _StubMlp(nn.Module):
    def forward(self, x: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        return x * 0.5


def _make_stubbed_layer(device: str, seed: int = 77) -> tuple[Any, dict[str, Any]]:
    torch.manual_seed(seed)
    cfg = {**_DEEPSEEK_V41_TINY_CONFIG, "device": device, "dtype": torch.bfloat16}
    lm = v41.DeepseekV41ForCausalLM(cfg)
    with torch.no_grad():
        for name, param in lm.named_parameters():
            # The mHC coefficient tensors live under the nested
            # ``layers.N.hc`` module; leaving the rest at ``torch.empty``
            # would feed NaN/garbage into the layer and make the comparison
            # below depend on allocator luck.
            if not name.startswith("model.layers.3.hc."):
                continue
            std = 0.004 if name.endswith("_fn") else 0.5
            param.normal_(mean=0.0, std=std)
    layer = lm.model.layers[3]
    layer.self_attn = _StubAttention()
    layer.input_layernorm = _StubNorm()
    layer.post_attention_layernorm = _StubNorm()
    layer.mlp = _StubMlp()
    inputs = {
        "hidden": (torch.randn(5, 4, cfg["hidden_size"]) * 1.3).to(torch.bfloat16).to(device),
        "positions": torch.arange(5, device=device),
        "cos_sin_cache": None,
        "input_ids": None,
        "pre_mix": torch.rand(5, 4, device=device) + 0.1,
    }
    return layer, inputs


def _run_layer(layer: Any, inputs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        hidden, ffn_pre = layer(
            inputs["hidden"], inputs["positions"], inputs["cos_sin_cache"], inputs["input_ids"], inputs["pre_mix"]
        )
    return hidden.clone(), ffn_pre.clone()


# ---------------------------------------------------------------------------
# R4: hc_pre (mHC sinkhorn, delayed collapse)
# ---------------------------------------------------------------------------


def _hc_pre_va_op_available() -> bool:
    return getattr(torch.ops.xllm_ops, "hc_pre_va", None) is not None


def _recording_hc_mixes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Record every ``hc_mixes`` call as ``(x, hc_fn, hc_scale, hc_base, pre, post, comb)``."""
    recorded: list[tuple[Any, ...]] = []
    original = v41.DeepseekV41HyperConnection.hc_mixes

    def wrapper(
        self: Any, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre, post, comb = original(self, x, hc_fn, hc_scale, hc_base)
        recorded.append((x, hc_fn, hc_scale, hc_base, pre, post, comb))
        return pre, post, comb

    monkeypatch.setattr(v41.DeepseekV41HyperConnection, "hc_mixes", wrapper)
    return recorded


def _activate_hc_pre_mode(monkeypatch: pytest.MonkeyPatch, layer: Any, mode: str) -> None:
    """Force one of the static hc_pre modes."""
    del monkeypatch
    layer.hc._hc_pre_mode = mode


def _normalized_subblocks(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    layer: Any,
    inputs: dict[str, Any],
) -> tuple[list[tuple[Any, ...]], torch.Tensor, torch.Tensor]:
    """Run one layer forward and return both mHC sub-block calls, normalized.
    Each record is ``(x, hc_fn, hc_scale, hc_base, pre_mix, pre, post, comb, y)`` where ``y`` is the collapsed sub-block"""
    hc = layer.hc
    mixes = _recording_hc_mixes(monkeypatch)
    hidden, ffn_pre = _run_layer(layer, inputs)
    attn_pre = mixes[0][4]
    records = [
        (
            mixes[0][0],
            mixes[0][1],
            mixes[0][2],
            mixes[0][3],
            inputs["pre_mix"],
            attn_pre,
            mixes[0][5],
            mixes[0][6],
            hc.hc_pre(mixes[0][0], inputs["pre_mix"]),
        ),
        (
            mixes[1][0],
            mixes[1][1],
            mixes[1][2],
            mixes[1][3],
            attn_pre,
            mixes[1][4],
            mixes[1][5],
            mixes[1][6],
            hc.hc_pre(mixes[1][0], attn_pre),
        ),
    ]
    return records, hidden, ffn_pre


def _recording_hc_post(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Record every ``hc_post`` call as ``(x, residual, post, comb, out)``."""
    recorded: list[tuple[Any, ...]] = []
    original = v41.DeepseekV41HyperConnection.hc_post

    def wrapper(
        self: Any, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
    ) -> torch.Tensor:
        out = original(self, x, residual, post, comb)
        recorded.append((x, residual, post, comb, out))
        return out

    monkeypatch.setattr(v41.DeepseekV41HyperConnection, "hc_post", wrapper)
    return recorded


def _hc_mixes_of(hc: Any, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        return hc.hc_mixes(x, hc.hc_attn_fn, hc.hc_attn_scale, hc.hc_attn_base)


def test_hc_pre_switch_gates_on_hc_mult_device_and_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fused decision is static (hc_mult + dtype/device + op)."""
    device = torch.device("privateuseone")
    monkeypatch.setattr(v41, "_get_hc_pre_fused_op", lambda: None)  # probe miss
    assert v41._hc_pre_mode_decision(torch.bfloat16, device, 4) == _MODE_EAGER

    # The decision only probes the op's presence; it never calls it.
    monkeypatch.setattr(v41, "_get_hc_pre_fused_op", lambda: object())
    assert v41._hc_pre_mode_decision(torch.bfloat16, device, 4) == _MODE_FUSED
    # hc_mult != 4 is an intentional eager fallback (kernel specialisation),
    # matching the hc_post guard.
    assert v41._hc_pre_mode_decision(torch.bfloat16, device, 2) == _MODE_EAGER
    assert v41._hc_pre_mode_decision(torch.float32, device, 4) == _MODE_EAGER
    assert v41._hc_pre_mode_decision(torch.bfloat16, torch.device("cpu"), 4) == _MODE_EAGER


def test_hc_pre_non_four_mult_fallback_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    """hc_mult != 4 is an intentional eager fallback, not a missing-op alert."""
    spy = _SpyLogger()
    monkeypatch.setattr(v41, "logger", spy)
    monkeypatch.setattr(v41, "_fused_hc_pre_fallback_logged", False)
    monkeypatch.setattr(v41, "_get_hc_pre_fused_op", lambda: None)
    device = torch.device("privateuseone")

    # Unsupported geometry: silent eager fallback.
    v41._log_fused_hc_pre_fallback(torch.bfloat16, device, 2)
    assert spy.warnings == []

    # A missing op at the supported geometry still warns, exactly once.
    v41._log_fused_hc_pre_fallback(torch.bfloat16, device, 4)
    v41._log_fused_hc_pre_fallback(torch.bfloat16, device, 4)
    assert len(spy.warnings) == 1 and "hc_pre" in spy.warnings[0]


def _reset_hc_pre_log_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v41, "_fused_hc_pre_fallback_logged", False)
    monkeypatch.setattr(v41, "_fused_hc_pre_active_logged", False)


def test_hc_pre_fused_selected_logs_info_and_missing_op_warns_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fused selection is an info; a real missing op warns once -- never both."""
    spy = _SpyLogger()
    monkeypatch.setattr(v41, "logger", spy)
    device = torch.device("privateuseone")

    # (a) The op resolves: the decision is fused and only the active-path info
    # may fire.
    _reset_hc_pre_log_state(monkeypatch)
    monkeypatch.setattr(v41, "_get_hc_pre_fused_op", lambda: object())
    assert v41._hc_pre_mode_decision(torch.bfloat16, device, 4) == _MODE_FUSED
    v41._log_hc_pre_mode(torch.bfloat16, device, 4, _MODE_FUSED)
    v41._log_hc_pre_mode(torch.bfloat16, device, 4, _MODE_FUSED)
    assert spy.warnings == []
    assert len(spy.infos) == 1 and "fused" in spy.infos[0]

    # (b) The op is gone: the decision is eager and the warning is about a real
    # absence; no active-path info is emitted.
    _reset_hc_pre_log_state(monkeypatch)
    spy.infos.clear()
    monkeypatch.setattr(v41, "_get_hc_pre_fused_op", lambda: None)
    assert v41._hc_pre_mode_decision(torch.bfloat16, device, 4) == _MODE_EAGER
    v41._log_hc_pre_mode(torch.bfloat16, device, 4, _MODE_EAGER)
    v41._log_hc_pre_mode(torch.bfloat16, device, 4, _MODE_EAGER)
    assert len(spy.warnings) == 1 and "hc_pre" in spy.warnings[0]
    assert spy.infos == []


class _RecordingAttention(nn.Module):
    """Attention stub that captures the mHC-collapsed input it receives."""

    def __init__(self, scale: float = 0.9) -> None:
        super().__init__()
        self.scale = scale
        self.inputs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, positions: torch.Tensor, cos_sin_cache: Any) -> torch.Tensor:
        self.inputs.append(x.clone())
        return x * self.scale


def test_decoder_layer_attention_input_uses_external_pre_mix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Attention input is ``collapse(hidden, external pre_mix)``, not ``attn_pre``.
    The collapse wiring is shared by the eager and fused paths, so no fused-vs-eager comparison can observe a regression that swaps"""
    for mode in _HC_PRE_MODES:
        with monkeypatch.context() as mp:
            layer, inputs = _make_stubbed_layer("cpu", seed=88)
            hc = layer.hc
            attention = _RecordingAttention()
            layer.self_attn = attention
            _activate_hc_pre_mode(mp, layer, mode)

            _normalized_subblocks(mp, mode, layer, inputs)

            hidden = inputs["hidden"]
            # ``input_layernorm`` is the identity stub, so the recorded tensor is
            # exactly the collapsed attention input.
            attn_input = attention.inputs[0]
            expected = hc.hc_pre(hidden, inputs["pre_mix"])
            assert torch.equal(attn_input, expected), f"mode={mode}: attention input does not use the external pre_mix"

            # The wrong wiring (this block's own attn_pre) is far away, so the
            # equality above is a real constraint rather than a vacuous one.
            own_pre = _hc_mixes_of(hc, hidden)[0]
            wrong = hc.hc_pre(hidden, own_pre)
            gap = (expected.float() - wrong.float()).abs().max().item()
            assert gap > 1e-2, f"mode={mode}: pre_mix and attn_pre too close to discriminate (gap={gap:.3e})"


class _RecordingMlp(nn.Module):
    """MLP stub that captures the mHC-collapsed input it receives."""

    def __init__(self, scale: float = 0.5) -> None:
        super().__init__()
        self.scale = scale
        self.inputs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        del input_ids
        self.inputs.append(x.clone())
        return x * self.scale


def test_decoder_layer_mlp_input_uses_this_block_attn_pre(monkeypatch: pytest.MonkeyPatch) -> None:
    """MLP input is ``collapse(post_attn_hidden, attn_pre)`` of THIS block.
    Companion to ``test_decoder_layer_attention_input_uses_external_pre_mix``: the attention input collapses with the"""
    for mode in _HC_PRE_MODES:
        with monkeypatch.context() as mp:
            layer, inputs = _make_stubbed_layer("cpu", seed=89)
            hc = layer.hc
            mlp = _RecordingMlp()
            layer.mlp = mlp
            _activate_hc_pre_mode(mp, layer, mode)
            records, _, ffn_pre_out = _normalized_subblocks(mp, mode, layer, inputs)

            # Record #1 is the attention sub-block (over the block input),
            # record #2 is the FFN sub-block (over the post-attention hidden).
            attn_rec, ffn_rec = records[0], records[1]
            attn_pre = attn_rec[5]
            post_attn_hidden = ffn_rec[0]
            mlp_input = mlp.inputs[0]
            # ``post_attention_layernorm`` is the identity stub, so the recorded
            # tensor is exactly the collapsed MLP input.
            expected = hc.hc_pre(post_attn_hidden, attn_pre)
            assert torch.equal(mlp_input, expected), (
                f"mode={mode}: MLP input does not collapse with this block's attn_pre"
            )
            assert torch.equal(ffn_rec[8], expected), f"mode={mode}: sub-block collapse != MLP input"
            # ...and it is the collapsed 2D stream, not the raw 3D post-attention
            # hidden (the un-collapsed wiring would not even match the shape).
            assert mlp_input.dim() == 2 and mlp_input.shape == expected.shape, (
                f"mode={mode}: shape={tuple(mlp_input.shape)}"
            )

            # The FFN sub-block's external mix is THIS block's attn_pre, and the
            # block hands back the FFN sub-block's own ``pre``.
            assert torch.equal(ffn_rec[4], attn_pre), f"mode={mode}: ffn pre_mix != this block's attn_pre"
            assert torch.equal(ffn_pre_out, ffn_rec[5]), f"mode={mode}: returned ffn_pre mismatch"

            # Non-vacuity: the other plausible coefficient (this block's
            # ffn_pre, i.e. the self-collapse mistake) lands far away.
            ffn_pre = ffn_rec[5]
            wrong_self = hc.hc_pre(post_attn_hidden, ffn_pre)
            gap_self = (expected.float() - wrong_self.float()).abs().max().item()
            assert gap_self > 1e-2, f"mode={mode}: attn_pre and ffn_pre too close (gap={gap_self:.3e})"


def test_decoder_layer_hc_post_uses_the_pre_sub_block_residual(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both ``hc_post`` calls recombine with the residual from BEFORE their sub-block.
    Call #1 (attention) must use the block input ``hidden``; call #2 (FFN) must use the post-attention hidden it was fed."""
    for mode in _HC_PRE_MODES:
        with monkeypatch.context() as mp:
            layer, inputs = _make_stubbed_layer("cpu", seed=90)
            hc = layer.hc
            attention = _RecordingAttention()
            layer.self_attn = attention
            _activate_hc_pre_mode(mp, layer, mode)
            # The re-evaluation below must use the unpatched function (captured
            # before the recorder replaces the class attribute) so it does not
            # append extra records.
            hc_post_handler = v41.DeepseekV41HyperConnection.hc_post
            posts = _recording_hc_post(mp)
            records, layer_hidden, _ = _normalized_subblocks(mp, mode, layer, inputs)
            attn_rec, ffn_rec = records[0], records[1]

            block_hidden = inputs["hidden"]
            attn_post, attn_comb = attn_rec[6], attn_rec[7]
            ffn_post, ffn_comb = ffn_rec[6], ffn_rec[7]
            attn_out, residual_attn, post_attn, comb_attn, post_attn_hidden = posts[0]
            mlp_out, residual_ffn, post_ffn, comb_ffn, final_hidden = posts[1]

            # The attention stub input is the collapsed attention input, so its
            # output identifies the ``x`` operand of the first recombination.
            assert torch.equal(attn_out, attention.inputs[0] * attention.scale), f"mode={mode}: attn_out mismatch"

            # Sub-block coefficients are projected from the same stream their
            # recombination consumes: #1 from the block input, #2 from the
            # post-attention hidden (the #1 output), not the other way round.
            assert torch.equal(attn_rec[0], block_hidden), f"mode={mode}: attn sub-block input != block input"
            assert torch.equal(ffn_rec[0], post_attn_hidden), f"mode={mode}: ffn sub-block input != post-attn hidden"

            # #1: residual is the PRE-attention hidden, not the collapse output
            # and not the post-attention hidden.
            assert torch.equal(residual_attn, block_hidden), f"mode={mode}: attn hc_post residual != block input"
            assert torch.equal(post_attn, attn_post) and torch.equal(comb_attn, attn_comb), (
                f"mode={mode}: attn hc_post post/comb are not the attn sub-block's coefficients"
            )
            expected_post_attn = hc_post_handler(hc, attn_out, block_hidden, attn_post, attn_comb)
            assert torch.equal(post_attn_hidden, expected_post_attn), f"mode={mode}: post-attn hidden mismatch"

            # #2: residual is the post-attention hidden -- the input this
            # sub-block consumed -- not the stale block input.
            assert torch.equal(residual_ffn, post_attn_hidden), f"mode={mode}: ffn hc_post residual != post-attn hidden"
            assert torch.equal(post_ffn, ffn_post) and torch.equal(comb_ffn, ffn_comb), (
                f"mode={mode}: ffn hc_post post/comb are not the ffn sub-block's coefficients"
            )
            expected_final = hc_post_handler(hc, mlp_out, post_attn_hidden, ffn_post, ffn_comb)
            assert torch.equal(final_hidden, expected_final), f"mode={mode}: final hidden mismatch"
            # ...and the block returns exactly that recombined hidden.
            assert torch.equal(layer_hidden, final_hidden), f"mode={mode}: block did not return the recombined hidden"

            # Non-vacuity: every shape-valid wrong residual (and a cross-wired
            # post/comb) lands far away from the correct recombination. The
            # collapse output itself is 2D, so it cannot be a residual operand
            # at all -- that wiring is rejected by hc_post's own einsum.
            wrong_attn = {
                "residual=post-attn hidden": hc_post_handler(hc, attn_out, post_attn_hidden, attn_post, attn_comb),
                "post/comb=ffn sub-block": hc_post_handler(hc, attn_out, block_hidden, ffn_post, ffn_comb),
            }
            for label, wrong in wrong_attn.items():
                gap = (post_attn_hidden.float() - wrong.float()).abs().max().item()
                assert gap > 1e-2, f"mode={mode}: attn hc_post {label} too close (gap={gap:.3e})"
            stale = hc_post_handler(hc, mlp_out, block_hidden, ffn_post, ffn_comb)
            gap_stale = (final_hidden.float() - stale.float()).abs().max().item()
            assert gap_stale > 1e-2, f"mode={mode}: ffn hc_post stale residual too close (gap={gap_stale:.3e})"


def test_hc_pre_falls_back_without_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """No registered op: the module stays eager, bit for bit."""
    if _hc_pre_va_op_available():
        pytest.skip("hc_pre_va op registered; the NPU gates test covers this host")
    device = "npu" if _npu_ready() else "cpu"

    # With the fused op missing, the module must take the eager path.
    monkeypatch.setattr(v41, "_get_hc_pre_fused_op", lambda: None)
    layer_on, inputs_on = _make_stubbed_layer(device, seed=83)
    assert layer_on.hc._hc_pre_mode == _MODE_EAGER
    pre_on, post_on, comb_on = _hc_mixes_of(layer_on.hc, inputs_on["hidden"])

    layer_off, inputs_off = _make_stubbed_layer(device, seed=83)
    assert layer_off.hc._hc_pre_mode == _MODE_EAGER
    pre_off, post_off, comb_off = _hc_mixes_of(layer_off.hc, inputs_off["hidden"])

    # Both runs are the untouched eager formula.
    hc = layer_off.hc
    x = inputs_off["hidden"]
    flattened = x.flatten(-2).float()
    rsqrt = torch.rsqrt(flattened.square().mean(-1, keepdim=True) + hc.norm_eps)
    ref = v41._hc_split_sinkhorn(
        F.linear(flattened, hc.hc_attn_fn) * rsqrt,
        hc.hc_attn_scale,
        hc.hc_attn_base,
        hc.hc_mult,
        hc.sinkhorn_iters,
        hc.hc_eps,
    )
    assert torch.equal(pre_off, ref[0]) and torch.equal(post_off, ref[1]) and torch.equal(comb_off, ref[2])
    assert torch.equal(pre_on, pre_off) and torch.equal(post_on, post_off) and torch.equal(comb_on, comb_off)
