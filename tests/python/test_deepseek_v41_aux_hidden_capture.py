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

"""DeepSeek-V4.1 DSpark aux-hidden capture, across its two layers.

The target that captures aux hidden states for the DSpark drafter is exercised
at two levels, each merged here from its own test module:

``[1] Target aux-hidden capture``
    ``DeepseekV41Model`` buffers the residual stream entering ``layers_to_capture``
    into a ``[tokens, n_captured * hidden]`` tensor; ``dspark_target_layer_ids``
    drives the list and the draft's ``main_proj`` fuses the blocks. Capture must
    be free and invisible when the list is empty, and the ``hc_mult`` copies must
    collapse exactly the way the native V4 target collapses them
    (``hidden.mean(dim=1)``).

``[2] Executor side channel``
    A graph-captured target returns a single Tensor, so its aux-hidden capture is
    written into a persistent buffer and published out of band. ``ModelExecutor``
    appends that buffer to the runner output so the C++ bridge sees the same
    ``(hidden, aux)`` tuple the eager transient capture returns, sliced to the
    real local row count (the padded graph tail is scratch).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import xllm.python.models.deepseek_v41 as deepseek_v41_module
from tests.python.test_deepseek_v41_model import _DEEPSEEK_V41_TINY_CONFIG
from xllm.python.model_executor.executor import ModelExecutor
from xllm.python.models.aux_hidden_capture import AuxHiddenCapture
from xllm.python.models.deepseek_v41 import DeepseekV41Config, DeepseekV41Model

# === [1] Target aux-hidden capture ===


class _Embedding(nn.Module):
    """Deterministic ``[tokens, hidden]`` embedding (no parameters to load)."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        values = input_ids.to(torch.float32).unsqueeze(-1)
        return values.expand(-1, self.hidden_size).contiguous() + 1.0


class _HcCollapse:
    """Stands in for the last block's learned mHC final merge."""

    def hc_pre(self, hidden: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        del pre_mix
        return hidden.mean(dim=1)


class _HcDeltaLayer(nn.Module):
    """Writes ``delta + hc_slot`` into every row, ignoring its input.

    Making the value depend on the ``hc_mult`` slot means the captured tensor
    differs from the raw layer output unless the capture collapses ``hc_mult``
    with the same mean the native target uses. Ignoring the input makes each
    layer's output identify the layer, so a capture that takes the entering
    stream of layer ``i`` is distinguishable from one that takes its output.
    """

    def __init__(self, delta: float, hc_mult: int) -> None:
        super().__init__()
        self.delta = delta
        self.hc = _HcCollapse()
        self.register_buffer("hc_slots", torch.arange(hc_mult, dtype=torch.float32))

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        input_ids: torch.Tensor,
        pre_mix: torch.Tensor,
        *extra: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del positions, cos_sin_cache, input_ids, extra
        values = self.delta + self.hc_slots.view(1, -1, 1)
        return values.expand_as(hidden).contiguous(), pre_mix


def _forward_context() -> SimpleNamespace:
    backend = SimpleNamespace(
        reset_forward=lambda metadata: None,
        prepare_dsa_metadata_for_forward=lambda metadata: None,
    )
    metadata = SimpleNamespace(dsa_graph_mode=False, dsa_metadata=None)
    return SimpleNamespace(attention_backend=backend, metadata=metadata)


def _capture_model(
    monkeypatch: pytest.MonkeyPatch,
    layers_to_capture: tuple[int, ...],
    enable_graph_aux_hidden_states: bool = False,
) -> tuple[DeepseekV41Model, DeepseekV41Config]:
    config_dict = {**_DEEPSEEK_V41_TINY_CONFIG, "layers_to_capture": list(layers_to_capture)}
    if enable_graph_aux_hidden_states:
        config_dict["enable_graph_aux_hidden_states"] = True
    cfg = DeepseekV41Config.from_dict(config_dict)
    model = DeepseekV41Model(cfg, torch.float32, torch.device("cpu"), config=config_dict)
    # Engram needs its checkpoint sidecar; the capture path never reads it.
    model.engram_hash = None
    model.embed_tokens = _Embedding(cfg.hidden_size)
    model.layers = nn.ModuleList([_HcDeltaLayer(float(index), cfg.hc_mult) for index in range(cfg.n_layers)])
    monkeypatch.setattr(deepseek_v41_module, "get_forward_context", _forward_context)
    return model, cfg


def test_config_reads_layers_to_capture() -> None:
    config = DeepseekV41Config.from_dict({**_DEEPSEEK_V41_TINY_CONFIG, "layers_to_capture": [37, 38, 39]})

    assert config.layers_to_capture == [37, 38, 39]
    # Absent list: speculative decoding is off, so capture stays disabled.
    assert DeepseekV41Config.from_dict(_DEEPSEEK_V41_TINY_CONFIG).layers_to_capture == []


def test_config_reads_the_graph_aux_capture_flag() -> None:
    # Default off keeps the target on the eager tuple contract.
    assert DeepseekV41Config.from_dict(_DEEPSEEK_V41_TINY_CONFIG).enable_graph_aux_hidden_states is False
    # The C++ bridge (PyCausalLM::build_config_dict) sets this key.
    flagged = DeepseekV41Config.from_dict({**_DEEPSEEK_V41_TINY_CONFIG, "enable_graph_aux_hidden_states": True})
    assert flagged.enable_graph_aux_hidden_states is True


def test_capture_disabled_returns_a_plain_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    model, cfg = _capture_model(monkeypatch, ())

    output = model(torch.tensor([1, 2]), torch.tensor([0, 1]))

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, cfg.hidden_size)


def test_capture_buffers_the_configured_layers_in_config_order(monkeypatch: pytest.MonkeyPatch) -> None:
    model, cfg = _capture_model(monkeypatch, (2, 0))

    output = model(torch.tensor([1, 2]), torch.tensor([0, 1]))

    assert isinstance(output, tuple)
    _, aux_hidden = output
    # One block per captured layer, ordered by ``layers_to_capture`` (not by
    # layer id): layer 2 lands in slot 0 and layer 0 in slot 1.
    assert aux_hidden.shape == (2, len(model.aux_hidden_capture._layers_to_capture) * cfg.hidden_size)
    hc_mean = (cfg.hc_mult - 1) / 2
    # Each slot holds the stream ENTERING its layer: layer 2 sees layer 1's
    # output, and layer 0 sees the embedding (identical across hc_mult copies,
    # so the collapse leaves the per-token value and adds no hc_mean).
    torch.testing.assert_close(aux_hidden[:, : cfg.hidden_size], torch.full((2, cfg.hidden_size), 1.0 + hc_mean))
    torch.testing.assert_close(
        aux_hidden[:, cfg.hidden_size :],
        torch.tensor([[2.0], [3.0]]).expand(2, cfg.hidden_size),
    )


def test_should_capture_matches_the_configured_layers() -> None:
    capture = AuxHiddenCapture((4, 1))

    assert capture.enabled
    assert capture.should_capture(4)
    assert capture.should_capture(1)
    assert not capture.should_capture(0)
    assert not AuxHiddenCapture(()).enabled


def test_graph_flag_switches_the_model_capture_to_persistent(monkeypatch: pytest.MonkeyPatch) -> None:
    model, cfg = _capture_model(monkeypatch, (2, 0), enable_graph_aux_hidden_states=True)

    assert model.aux_hidden_capture.persistent_buffer is True
    output = model(torch.tensor([1, 2]), torch.tensor([0, 1]))

    # The graph runner only accepts a single Tensor, so the capture is published
    # out of band through ``buffer`` instead of in a tuple.
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, cfg.hidden_size)
    buffer = model.aux_hidden_capture.buffer
    assert buffer is not None
    hc_mean = (cfg.hc_mult - 1) / 2
    assert buffer.shape == (2, 2 * cfg.hidden_size)
    torch.testing.assert_close(buffer[:, : cfg.hidden_size], torch.full((2, cfg.hidden_size), 1.0 + hc_mean))
    torch.testing.assert_close(
        buffer[:, cfg.hidden_size :],
        torch.tensor([[2.0], [3.0]]).expand(2, cfg.hidden_size),
    )


def test_graph_flag_off_keeps_the_eager_tuple_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    # Zero change for every non-graph run: the flag is absent/false and the
    # capture stays transient, so the model still returns ``(hidden, aux)``.
    model, _ = _capture_model(monkeypatch, (2, 0))

    assert model.aux_hidden_capture.persistent_buffer is False
    output = model(torch.tensor([1, 2]), torch.tensor([0, 1]))

    assert isinstance(output, tuple)
    assert model.aux_hidden_capture.buffer is None


def test_persistent_capture_is_reused_across_graph_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    model, _ = _capture_model(monkeypatch, (2, 0), enable_graph_aux_hidden_states=True)

    model(torch.tensor([1, 2]), torch.tensor([0, 1]))
    first = model.aux_hidden_capture.buffer
    model(torch.tensor([3, 4]), torch.tensor([2, 3]))
    second = model.aux_hidden_capture.buffer

    # Same fixed address across steps: graph capture/replay requires it.
    assert first is not None and second is not None
    assert first.data_ptr() == second.data_ptr()


def test_transient_buffers_are_allocated_per_forward() -> None:
    # OFF/by-default behaviour must stay a fresh allocation every forward: the
    # eager path returns the capture in a tuple, so aliasing it would corrupt
    # the previous step's aux hidden states.
    capture = AuxHiddenCapture((4, 1))
    hidden = torch.randn(3, 8)

    first = capture.create_buffer(hidden)
    second = capture.create_buffer(hidden)

    assert first is not None and second is not None
    assert first.data_ptr() != second.data_ptr()
    assert first.shape == (3, 16)
    assert not capture.persistent_buffer
    assert capture.buffer is None


def test_persistent_buffer_is_reused_across_forwards() -> None:
    capture = AuxHiddenCapture((4, 1), persistent_buffer=True)
    hidden = torch.randn(3, 8)

    first = capture.create_buffer(hidden)
    second = capture.create_buffer(hidden)

    assert first is not None and second is not None
    assert first.data_ptr() == second.data_ptr()
    assert capture.persistent_buffer
    assert capture.buffer is first


def test_persistent_buffer_reallocates_only_when_the_shape_changes() -> None:
    capture = AuxHiddenCapture((4, 1), persistent_buffer=True)

    small = capture.create_buffer(torch.randn(3, 8))
    # A new bucket shape gets its own fixed buffer...
    large = capture.create_buffer(torch.randn(5, 8))
    # ...while the original shape keeps the buffer captured for its graph.
    small_again = capture.create_buffer(torch.randn(3, 8))

    assert small is not None and large is not None and small_again is not None
    assert small.shape == (3, 16)
    assert large.shape == (5, 16)
    assert small.data_ptr() != large.data_ptr()
    assert small_again.data_ptr() == small.data_ptr()


def test_persistent_finalize_keeps_the_output_a_plain_tensor() -> None:
    capture = AuxHiddenCapture((4, 1), persistent_buffer=True)
    hidden = torch.zeros(2, 8)
    # Layer 4 lands in slot 0 and layer 1 in slot 1.
    layer4 = torch.full((2, 8), 1.5)
    layer1 = torch.full((2, 8), -2.0)

    buffer = capture.create_buffer(hidden)
    capture.capture_layer(4, layer4, None, buffer)
    capture.capture_layer(1, layer1, None, buffer)
    output = capture.finalize(hidden, buffer)

    # The graph runner only accepts a single Tensor; the capture is published
    # out of band through ``capture.buffer``.
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 8)
    assert capture.buffer is not None
    torch.testing.assert_close(capture.buffer[:, :8], layer4)
    torch.testing.assert_close(capture.buffer[:, 8:], layer1)


def test_persistent_disabled_capture_allocates_nothing() -> None:
    capture = AuxHiddenCapture((), persistent_buffer=True)
    hidden = torch.randn(2, 8)

    assert capture.create_buffer(hidden) is None
    assert capture.finalize(hidden, None) is hidden
    assert capture.buffer is None


# === [2] Executor side channel ===


def _executor(capture: AuxHiddenCapture | None, *, top_level: bool = False) -> ModelExecutor:
    """An executor whose model exposes ``capture`` (on the body by default).

    ``ModelExecutor.__init__`` needs an NPU runtime, so bypass it: the helpers
    under test only touch ``self.model``.
    """
    executor = ModelExecutor.__new__(ModelExecutor)
    body = SimpleNamespace(aux_hidden_capture=capture)
    executor.model = SimpleNamespace(aux_hidden_capture=capture) if top_level else SimpleNamespace(model=body)
    return executor


def _filled_persistent_capture(rows: int, padded_rows: int, hidden: int = 8) -> AuxHiddenCapture:
    capture = AuxHiddenCapture((7,), persistent_buffer=True)
    buffer = capture.create_buffer(torch.zeros(padded_rows, hidden))
    assert buffer is not None
    buffer[:rows] = 1.0
    buffer[rows:] = 99.0
    return capture


# --------------------------------------------------------------------------
# Capture resolution
# --------------------------------------------------------------------------


def test_resolves_the_capture_through_the_model_body() -> None:
    capture = AuxHiddenCapture((7,))
    executor = _executor(capture)

    assert ModelExecutor._resolve_aux_hidden_capture(executor.model) is capture


def test_resolves_the_capture_on_the_top_level_model() -> None:
    capture = AuxHiddenCapture((7,))
    executor = _executor(capture, top_level=True)

    assert ModelExecutor._resolve_aux_hidden_capture(executor.model) is capture


def test_models_without_capture_resolve_to_none() -> None:
    assert ModelExecutor._resolve_aux_hidden_capture(SimpleNamespace(model=SimpleNamespace())) is None
    assert ModelExecutor._resolve_aux_hidden_capture(SimpleNamespace()) is None


# --------------------------------------------------------------------------
# Publishing contract
# --------------------------------------------------------------------------


def test_persistent_capture_is_appended_and_sliced_to_the_real_rows() -> None:
    capture = _filled_persistent_capture(rows=3, padded_rows=5)
    executor = _executor(capture)
    hidden = torch.zeros(3, 8)

    result = executor._attach_persistent_aux_hidden_states(hidden, 3)

    assert isinstance(result, tuple)
    got_hidden, got_aux = result
    assert got_hidden is hidden
    # The padded tail rows are scratch and must not ride along.
    assert got_aux.shape == (3, 8)
    torch.testing.assert_close(got_aux, torch.full((3, 8), 1.0))


def test_persistent_capture_without_padding_is_returned_whole() -> None:
    capture = _filled_persistent_capture(rows=4, padded_rows=4)
    executor = _executor(capture)
    hidden = torch.zeros(4, 8)

    _, got_aux = executor._attach_persistent_aux_hidden_states(hidden, 4)

    assert got_aux.shape == (4, 8)
    torch.testing.assert_close(got_aux, torch.full((4, 8), 1.0))


def test_transient_capture_leaves_the_output_alone() -> None:
    # The eager tuple contract already carries the aux; nothing is appended.
    capture = AuxHiddenCapture((7,))
    capture.create_buffer(torch.zeros(3, 8))
    executor = _executor(capture)
    hidden = torch.zeros(3, 8)

    assert executor._attach_persistent_aux_hidden_states(hidden, 3) is hidden


def test_transient_capture_is_not_wrapped_even_with_a_lingering_buffer() -> None:
    # The persistent-buffer flag, not the mere presence of a buffer, decides:
    # a transient capture's output already carries the aux in its tuple, so
    # wrapping it here would double-publish (and break the tensor contract).
    # A stub pins the flag/buffer combination a real transient capture cannot
    # reach (its buffer stays None by design).
    capture = SimpleNamespace(enabled=True, persistent_buffer=False, buffer=torch.zeros(3, 16))
    executor = _executor(capture)
    hidden = torch.zeros(3, 8)

    assert executor._attach_persistent_aux_hidden_states(hidden, 3) is hidden


def test_disabled_capture_leaves_the_output_alone() -> None:
    executor = _executor(AuxHiddenCapture((), persistent_buffer=True))
    hidden = torch.zeros(3, 8)

    assert executor._attach_persistent_aux_hidden_states(hidden, 3) is hidden


def test_no_capture_leaves_the_output_alone() -> None:
    executor = _executor(None)
    hidden = torch.zeros(3, 8)

    assert executor._attach_persistent_aux_hidden_states(hidden, 3) is hidden


def test_persistent_capture_before_the_first_forward_leaves_the_output_alone() -> None:
    # No forward yet: nothing captured, so the output must stay a Tensor rather
    # than gaining a stale/empty aux.
    executor = _executor(AuxHiddenCapture((7,), persistent_buffer=True))
    hidden = torch.zeros(3, 8)

    assert executor._attach_persistent_aux_hidden_states(hidden, 3) is hidden


def test_an_existing_tuple_is_not_nested() -> None:
    capture = _filled_persistent_capture(rows=3, padded_rows=5)
    executor = _executor(capture)
    output = (torch.zeros(3, 8), torch.zeros(3, 16))

    assert executor._attach_persistent_aux_hidden_states(output, 3) is output
