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

"""KDA state, token ordering and padding contracts using a CPU oracle.

The oracle substitutes device kernels only. Production packing, initial-state
selection, state scatter and output normalization run unmodified. These tests
do not establish Ascend kernel or graph execution correctness.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn.functional as F

from xllm.python.model_loader import ScopedWeightLoader
from xllm.python.models.kimi_k3_kda import (
    KimiK3DeltaAttention,
    KimiK3KDACore,
    KimiK3KDAMetadata,
    _l2norm,
    build_kda_prefill_host_metadata,
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_l2norm_respects_mainline_dtype_contract(monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype) -> None:
    from xllm.python import kernels

    calls: list[float] = []

    def bf16_l2_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
        assert x.dtype == torch.bfloat16
        calls.append(eps)
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.square().sum(-1, keepdim=True) + eps)).to(x.dtype)

    monkeypatch.setattr(kernels, "l2_norm", bf16_l2_norm)
    x = torch.tensor([[0.0, 0.0], [0.25, -0.5]], dtype=dtype)
    expected = (x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6)).to(dtype)
    torch.testing.assert_close(_l2norm(x), expected, rtol=0, atol=0)
    assert calls == ([1e-6] if dtype == torch.bfloat16 else [])


class _StateDict:
    def __init__(self, values: dict[str, torch.Tensor]) -> None:
        self.values = values

    def has(self, name: str) -> bool:
        return name in self.values

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.values[name]


def _checkpoint(quantized: bool) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(41)
    shapes = {
        "q_proj.weight": (16, 12),
        "k_proj.weight": (16, 12),
        "v_proj.weight": (16, 12),
        "g_proj.weight": (16, 12),
        "b_proj.weight": (4, 12),
        "f_a_proj.weight": (4, 12),
        "f_b_proj.weight": (16, 4),
        "o_proj.weight": (12, 16),
        "q_conv1d.weight": (16, 1, 4),
        "k_conv1d.weight": (16, 4),
        "v_conv1d.weight": (16, 1, 4),
        "A_log": (8,),
        "dt_bias": (16,),
        "o_norm.weight": (4,),
    }
    values = {name: torch.randn(shape, generator=generator) * 0.15 for name, shape in shapes.items()}
    if quantized:
        for name in ("q_proj", "k_proj", "v_proj"):
            values[f"{name}.weight"] = torch.randint(-50, 51, (16, 12), generator=generator, dtype=torch.int8)
            values[f"{name}.weight_scale"] = torch.rand(16, generator=generator) * 0.01 + 0.001
            values[f"{name}.weight_offset"] = torch.zeros(16, 1)
    return values


@pytest.fixture
def linear_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    from xllm.python import kernels
    from xllm.python.kernels_npu.linear import prepare_quant_weight, prepare_row_parallel_weight

    def dynamic_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = x.float().abs().amax(-1).clamp_min(1e-12) / 127
        return (x.float() / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8), scale

    def quant_matmul(
        x: torch.Tensor,
        weight: torch.Tensor,
        transpose: bool,
        scale: torch.Tensor,
        offset: object,
        pertoken: torch.Tensor,
        bias: object,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        matrix = weight.t() if transpose else weight
        return ((x.int() @ matrix.int()).float() * scale * pertoken.unsqueeze(-1)).to(dtype)

    monkeypatch.setattr(kernels, "prepare_quant_weight", prepare_quant_weight, raising=False)
    monkeypatch.setattr(kernels, "prepare_row_parallel_weight", prepare_row_parallel_weight, raising=False)
    monkeypatch.setattr(kernels, "dynamic_quant", dynamic_quant, raising=False)
    monkeypatch.setattr(kernels, "quant_matmul", quant_matmul, raising=False)


def _layer(
    values: dict[str, torch.Tensor], quantized: bool, tp_size: int = 1, tp_rank: int = 0
) -> KimiK3DeltaAttention:
    layer = KimiK3DeltaAttention(
        12,
        {"num_heads": 4, "head_dim": 4, "short_conv_kernel_size": 4, "use_full_rank_gate": True},
        quantized=quantized,
        tp_size=tp_size,
        tp_rank=tp_rank,
        dtype=torch.bfloat16 if quantized else torch.float32,
    )
    # Exercise actual scoped/multi-shard resolution, not a custom find callback.
    shards = [
        _StateDict({f"attn.{key}": value for key, value in values.items() if "conv" in key}),
        _StateDict({f"attn.{key}": value for key, value in values.items() if "conv" not in key}),
    ]
    assert layer.load_weights(ScopedWeightLoader(shards, prefix="attn.")) == set(values)
    layer.process_weights_after_loading()
    return layer


@pytest.mark.parametrize("quantized", [False, True])
def test_complete_layer_tp_prefill_and_first_decode(
    cpu_kernels: None,
    linear_kernels: None,
    quantized: bool,
) -> None:
    values = _checkpoint(quantized)
    full = _layer(values, quantized)
    shards = [_layer(values, quantized, 2, rank) for rank in range(2)]
    dtype = full.o_norm_weight.dtype
    layers = [full, *shards]
    conv_states = [torch.zeros(4, *layer.conv_state_shape(), dtype=dtype) for layer in layers]
    recurrent_states = [torch.zeros(4, *layer.recurrent_state_shape()) for layer in layers]
    generator = torch.Generator().manual_seed(7)
    steps = [
        (_metadata([7, 3], [2, 0], 0, [False, False]), 10),
        (_metadata([1, 1, 1, 1], [2, -1, 0, -1], 4, [True] * 4), 4),
    ]
    for metadata, rows in steps:
        hidden = torch.randn(rows, 12, generator=generator).to(dtype)
        outputs = [
            layer(hidden, metadata, conv, state, reduce_o_proj=False)
            for layer, conv, state in zip(layers, conv_states, recurrent_states)
        ]
        tolerance = 5e-4 if quantized else 1e-6
        torch.testing.assert_close(
            outputs[0].float(),
            outputs[1].float() + outputs[2].float(),
            atol=tolerance,
            rtol=0.03 if quantized else 1e-5,
        )
        torch.testing.assert_close(recurrent_states[0], torch.cat(recurrent_states[1:], dim=1))
        restored_conv = torch.cat(
            [torch.cat([state.chunk(3, -1)[part] for state in conv_states[1:]], -1) for part in range(3)], -1
        )
        torch.testing.assert_close(conv_states[0], restored_conv)
        if metadata.num_decode_seqs:
            assert torch.count_nonzero(outputs[0][[1, 3]]) == 0


def test_complete_layer_reduction_and_layout_lifecycle(
    cpu_kernels: None, linear_kernels: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xllm.python import distributed

    layer = _layer(_checkpoint(False), False, 2, 0)
    assert layer.o_proj._use_checkpoint_layout
    assert not layer.o_proj._weight_is_transposed
    pointers = [parameter.data_ptr() for parameter in layer.parameters()]
    layer.process_weights_after_loading()
    assert pointers == [parameter.data_ptr() for parameter in layer.parameters()]
    with pytest.raises(RuntimeError, match="cannot reload"):
        layer.load_weights(ScopedWeightLoader([]))
    calls: list[torch.Tensor] = []
    monkeypatch.setattr(distributed, "all_reduce_", lambda x: calls.append(x))
    metadata = _metadata([1], [0], 1, [True])
    conv = torch.zeros(1, *layer.conv_state_shape())
    state = torch.zeros(1, *layer.recurrent_state_shape())
    hidden = torch.ones(1, 12)
    layer(hidden, metadata, conv, state)
    assert len(calls) == 1
    layer(hidden, metadata, conv, state, reduce_o_proj=False)
    assert len(calls) == 1


def test_complete_layer_rejects_invalid_weights_and_topology(linear_kernels: None) -> None:
    config = {"num_heads": 4, "head_dim": 4, "short_conv_kernel_size": 4, "use_full_rank_gate": True}
    with pytest.raises(ValueError, match="divide TP"):
        KimiK3DeltaAttention(12, config, tp_size=3)
    layer = KimiK3DeltaAttention(12, config)
    with pytest.raises(RuntimeError, match="checkpoint loading"):
        layer.process_weights_after_loading()
    values = _checkpoint(False)
    values["o_proj.weight"] = torch.zeros(12, 15)
    with pytest.raises(ValueError, match="shape"):
        _layer(values, False)
    values = _checkpoint(True)
    values["k_proj.weight_offset"].fill_(1)
    with pytest.raises(ValueError, match="symmetric"):
        _layer(values, True)


def _core() -> KimiK3KDACore:
    generator = torch.Generator().manual_seed(23)
    return KimiK3KDACore(
        torch.randn(4, 24, generator=generator) * 0.1,
        torch.tensor([-1.0, -0.5]),
        torch.randn(8, generator=generator) * 0.1,
        torch.randn(4, generator=generator),
    )


def _delta_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs: list[torch.Tensor] = []
    for row in range(q.shape[0]):
        query = F.normalize(q[row].float(), dim=-1, eps=1e-6) * scale
        key = F.normalize(k[row].float(), dim=-1, eps=1e-6)
        decay = (-a_log.exp().unsqueeze(-1) * F.softplus(gate[row].float() + dt_bias.view_as(key))).exp()
        state = state * decay.unsqueeze(-2)
        prediction = (state * key.unsqueeze(-2)).sum(-1)
        residual = (v[row].float() - prediction) * beta[row].unsqueeze(-1)
        state = state + residual.unsqueeze(-1) * key.unsqueeze(-2)
        outputs.append((state * query.unsqueeze(-2)).sum(-1))
    output = torch.stack(outputs) if outputs else v.new_empty(v.shape)
    return output, state


@pytest.fixture
def cpu_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install isolated numerical oracles with the production FLA signatures."""

    def conv(
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        conv_states: torch.Tensor,
        cache_indices: list[int],
        initial_state_mode: list[int],
        query_start_loc: list[int],
        **kwargs: object,
    ) -> torch.Tensor:
        output = torch.zeros_like(x)
        for index, slot in enumerate(cache_indices):
            if slot < 0:
                continue
            cache = conv_states[slot].clone() if initial_state_mode[index] else torch.zeros_like(conv_states[slot])
            for row in range(query_start_loc[index], query_start_loc[index + 1]):
                window = torch.cat((cache, x[row : row + 1]))
                output[row] = F.silu((window * weight).sum(0))
                cache = window[1:]
            conv_states[slot].copy_(cache)
        return output

    def conv_update(
        x: torch.Tensor,
        state: torch.Tensor,
        weight: torch.Tensor,
        activation: bool,
        bias: object,
        slots: torch.Tensor,
        starts: torch.Tensor,
        max_query_len: int,
        pad_slot: int,
        accepted: object,
    ) -> torch.Tensor:
        output = torch.zeros_like(x)
        for row, slot in enumerate(slots.tolist()):
            if slot == pad_slot:
                continue
            window = torch.cat((state[slot].t(), x[row : row + 1]))
            output[row] = F.silu((window * weight.t()).sum(0))
            state[slot].copy_(window[1:].t())
        return output

    def recurrent(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        *,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        ssm_state_indices: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert kwargs["layout"] == "TND" and kwargs["state_v_first"]
        output = torch.zeros_like(v)
        for sequence, slot in enumerate(ssm_state_indices.tolist()):
            start, end = cu_seqlens[sequence : sequence + 2].tolist()
            if start == end:
                continue
            assert slot >= 0
            result, state = _delta_reference(
                q[start:end],
                k[start:end],
                v[start:end],
                gate[start:end],
                beta[start:end],
                initial_state[slot],
                A_log,
                dt_bias,
                scale,
            )
            output[start:end] = result
            initial_state[slot].copy_(state)
        return output, initial_state

    def chunk(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        *,
        initial_state: torch.Tensor,
        cu_seqlens: list[int],
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert kwargs["layout"] == "BSND" and kwargs["state_v_first"]
        outputs: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        for sequence, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
            result, state = _delta_reference(
                q[0, start:end],
                k[0, start:end],
                v[0, start:end],
                gate[0, start:end],
                beta[0, start:end],
                initial_state[sequence],
                A_log,
                dt_bias,
                scale,
            )
            outputs.append(result)
            states.append(state)
        return torch.cat(outputs).unsqueeze(0), torch.stack(states)

    module = types.ModuleType("fla_npu.ops.ascendc")
    module.npu_causal_conv1d = conv
    module.recurrent_kda = recurrent
    module.chunk_kda_fwd = chunk
    monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", module)
    monkeypatch.setattr(torch.ops.xllm_ops, "causal_conv1d_update_v2", conv_update, raising=False)


def _metadata(lengths: list[int], slots: list[int], decode: int, initial: list[bool]) -> KimiK3KDAMetadata:
    starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    host_cu, host_chunks = build_kda_prefill_host_metadata(starts, decode)
    return KimiK3KDAMetadata(
        starts,
        torch.tensor(slots),
        decode,
        len(lengths) - decode,
        torch.tensor(initial),
        host_cu,
        host_chunks,
        slots if host_cu is not None else None,
        [int(value) for value in initial] if host_cu is not None else None,
        starts.tolist() if host_cu is not None else None,
    )


def test_host_metadata_offsets_decode_and_chunk_boundaries() -> None:
    assert build_kda_prefill_host_metadata(torch.tensor([0, 1, 2, 67, 70]), 2) == ((0, 65, 68), (0, 0, 0, 1, 1, 0))
    assert build_kda_prefill_host_metadata(torch.tensor([0, 1, 2]), 2) == (None, None)
    assert build_kda_prefill_host_metadata(None, 0) == (None, None)


def test_prefill_convolution_reuses_host_lists_without_device_reads(
    cpu_kernels: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    core = _core()
    metadata = _metadata([1, 3, 2], [3, 0, 2], 1, [True, False, True])
    calls: list[dict] = []

    def conv(x: torch.Tensor, weight: torch.Tensor, **kwargs: object) -> torch.Tensor:
        calls.append(kwargs)
        return x

    def forbid_host_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("a KDA layer must reuse per-step host convolution arguments")

    monkeypatch.setattr(sys.modules["fla_npu.ops.ascendc"], "npu_causal_conv1d", conv)
    monkeypatch.setattr(torch.Tensor, "tolist", forbid_host_read)
    monkeypatch.setattr(torch.Tensor, "cpu", forbid_host_read)
    hidden, state = torch.zeros(6, 24), torch.zeros(4, 3, 24)
    for _ in range(3):
        core._causal_conv1d(hidden, state, metadata)
    assert len(calls) == 3
    for arguments in calls:
        assert arguments["cache_indices"] is metadata.conv_state_indices_host
        assert arguments["initial_state_mode"] is metadata.conv_initial_state_mode_host
        assert arguments["query_start_loc"] is metadata.conv_query_start_loc_host


@pytest.mark.parametrize("slots", [[2], [2, -1, 0, -1], [-1, -1, -1], [3, 1, 0, 2]])
def test_decode_preserves_token_order_and_untouched_slots(cpu_kernels: None, slots: list[int]) -> None:
    torch.manual_seed(3)
    core = _core()
    batch = len(slots)
    inputs = torch.randn(batch, 24) * 0.1
    gate = torch.randn(batch, 2, 4) * 0.1
    beta = torch.rand(batch, 2)
    output_gate = torch.randn(batch, 2, 4)
    conv = torch.randn(5, 3, 24) * 0.1
    state = torch.randn(5, 2, 4, 4) * 0.1
    original_conv, original_state = conv.clone(), state.clone()
    metadata = _metadata([1] * batch, slots, batch, [True] * batch)
    actual = core(inputs, gate, beta, output_gate, metadata, conv, state)
    expected = torch.zeros_like(actual)
    for row, slot in enumerate(slots):
        if slot < 0:
            continue
        window = torch.cat((original_conv[slot], inputs[row : row + 1]))
        qkv = F.silu((window * core.conv_weight_t).sum(0)).view(3, 2, 4)
        result, final = _delta_reference(
            qkv[0:1],
            qkv[1:2],
            qkv[2:3],
            gate[row : row + 1],
            beta[row : row + 1],
            original_state[slot],
            core.A_log,
            core.dt_bias,
            core.scale,
        )
        normalized = result * torch.rsqrt(result.square().mean(-1, keepdim=True) + core.o_norm_eps)
        expected[row] = (normalized * core.o_norm_weight * output_gate[row].sigmoid()).flatten()
        torch.testing.assert_close(state[slot], final)
        torch.testing.assert_close(conv[slot], window[1:])
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    unused = list(set(range(5)) - {slot for slot in slots if slot >= 0})
    torch.testing.assert_close(state[unused], original_state[unused], atol=0, rtol=0)
    torch.testing.assert_close(conv[unused], original_conv[unused], atol=0, rtol=0)


def test_chunked_prefill_matches_unsplit_and_first_decode(cpu_kernels: None) -> None:
    torch.manual_seed(8)
    core = _core()
    inputs = torch.randn(8, 24) * 0.1
    gate = torch.randn(8, 2, 4) * 0.1
    beta = torch.rand(8, 2)
    output_gate = torch.randn(8, 2, 4)
    conv = torch.randn(4, 3, 24)
    state = torch.randn(4, 2, 4, 4)
    split_conv, split_state = conv.clone(), state.clone()
    whole = core(inputs[:7], gate[:7], beta[:7], output_gate[:7], _metadata([7], [2], 0, [False]), conv, state)
    first = core(
        inputs[:3], gate[:3], beta[:3], output_gate[:3], _metadata([3], [2], 0, [False]), split_conv, split_state
    )
    second = core(
        inputs[3:7], gate[3:7], beta[3:7], output_gate[3:7], _metadata([4], [2], 0, [True]), split_conv, split_state
    )
    torch.testing.assert_close(torch.cat((first, second)), whole, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(split_state, state)
    torch.testing.assert_close(split_conv, conv)
    metadata = _metadata([1], [2], 1, [True])
    expected = core(inputs[7:], gate[7:], beta[7:], output_gate[7:], metadata, conv, state)
    actual = core(inputs[7:], gate[7:], beta[7:], output_gate[7:], metadata, split_conv, split_state)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(split_state, state)
    torch.testing.assert_close(split_conv, conv)


def test_mixed_batch_matches_independent_sequences(cpu_kernels: None) -> None:
    torch.manual_seed(9)
    core = _core()
    inputs, gate, beta, output_gate = torch.randn(6, 24), torch.randn(6, 2, 4), torch.rand(6, 2), torch.randn(6, 2, 4)
    conv, state = torch.randn(5, 3, 24), torch.randn(5, 2, 4, 4)
    initial_conv, initial_state = conv.clone(), state.clone()
    actual = core(inputs, gate, beta, output_gate, _metadata([1, 3, 2], [3, 0, 2], 1, [True, False, True]), conv, state)
    expected: list[torch.Tensor] = []
    offset = 0
    for length, slot, decode, has_init in [(1, 3, 1, True), (3, 0, 0, False), (2, 2, 0, True)]:
        end = offset + length
        expected.append(
            core(
                inputs[offset:end],
                gate[offset:end],
                beta[offset:end],
                output_gate[offset:end],
                _metadata([length], [slot], decode, [has_init]),
                initial_conv,
                initial_state,
            )
        )
        offset = end
    torch.testing.assert_close(actual, torch.cat(expected))
    torch.testing.assert_close(conv, initial_conv)
    torch.testing.assert_close(state, initial_state)


def test_empty_rank_does_not_touch_state(cpu_kernels: None) -> None:
    core = _core()
    conv, state = torch.randn(3, 3, 24), torch.randn(3, 2, 4, 4)
    original_conv, original_state = conv.clone(), state.clone()
    result = core(
        torch.empty(0, 24),
        torch.empty(0, 2, 4),
        torch.empty(0, 2),
        torch.empty(0, 2, 4),
        _metadata([], [], 0, []),
        conv,
        state,
    )
    assert result.shape == (0, 8)
    torch.testing.assert_close(conv, original_conv, atol=0, rtol=0)
    torch.testing.assert_close(state, original_state, atol=0, rtol=0)


def test_missing_host_metadata_rejected_before_state_update(cpu_kernels: None) -> None:
    core = _core()
    conv, state = torch.randn(3, 3, 24), torch.randn(3, 2, 4, 4)
    original_conv, original_state = conv.clone(), state.clone()
    metadata = _metadata([2], [1], 0, [False])
    metadata.prefill_cu_seqlens_host = None
    with pytest.raises(ValueError, match="shared host"):
        core(torch.randn(2, 24), torch.randn(2, 2, 4), torch.rand(2, 2), torch.randn(2, 2, 4), metadata, conv, state)
    torch.testing.assert_close(conv, original_conv, atol=0, rtol=0)
    torch.testing.assert_close(state, original_state, atol=0, rtol=0)
