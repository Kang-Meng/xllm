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

"""Tests for the NPU paged-attention backend."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from xllm.python.model_executor.forward_context import ForwardContext, forward_context

pytest.importorskip("torch_npu", reason="NPU paged-attention tests require torch_npu")

from xllm.python.attention.backend import LayerCache  # noqa: E402
from xllm.python.attention.npu_paged_attention import (  # noqa: E402
    NpuPagedAttentionBackend,
)

pytestmark = pytest.mark.usefixtures("causal_conv1d_reference")


def test_uses_first_nonempty_key_cache() -> None:
    backend = NpuPagedAttentionBackend(
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        scale=0.125,
        sliding_window=0,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    linear_cache = LayerCache(
        key=None,
        value=None,
        conv=torch.empty(8, 3, 64),
        ssm=torch.empty(8, 2, 4, 4),
    )
    key_cache = torch.empty(17, 128, 2, 64)
    value_cache = torch.empty_like(key_cache)

    backend.bind_kv_caches(
        [
            linear_cache,
            LayerCache(key=key_cache, value=value_cache),
        ]
    )

    assert backend.num_kv_blocks == 17
    assert backend.page_size == 128


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("activation", ["identity", "silu"])
@pytest.mark.parametrize("is_prefill", [False, True])
def test_kda_dense_conv_dispatches_fused_activation(
    monkeypatch: pytest.MonkeyPatch, width: int, activation: str, is_prefill: bool
) -> None:
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    value = torch.arange(2 * 12 * width, dtype=torch.bfloat16).view(2, 12, width).transpose(1, 2)
    weight = torch.ones(12, 1, 4, dtype=torch.float32)
    layer = SimpleNamespace(
        layer_id=0, activation=activation, conv_weight_t=weight.squeeze(1).t().to(value.dtype).contiguous()
    )
    state = torch.zeros(2, 3, 12, dtype=torch.bfloat16)
    expected_conv = torch.linspace(-4, 4, value.numel()).to(torch.bfloat16).view(2, width, 12)

    def native_conv(
        inputs: torch.Tensor,
        weights: torch.Tensor,
        dense_state: torch.Tensor,
        query_start_loc: list[int],
        activation_mode: int,
        run_mode: int,
    ) -> torch.Tensor:
        assert inputs.is_contiguous()
        assert weights is layer.conv_weight_t
        torch.testing.assert_close(weights, weight.squeeze(1).t().to(value.dtype))
        assert dense_state is state
        assert query_start_loc == []
        assert activation_mode == (1 if activation == "silu" else 0)
        assert run_mode == (0 if is_prefill else 1)
        torch.testing.assert_close(inputs, value)
        dense_state.fill_(7)
        return expected_conv.clone()

    monkeypatch.setattr(torch.ops.xllm_ops, "causal_conv1d", native_conv)
    output = backend._causal_conv1d(value, state, layer, is_prefill=is_prefill)
    torch.testing.assert_close(output, expected_conv, rtol=0, atol=0)
    assert output.dtype == torch.bfloat16
    assert torch.all(state == 7)


@pytest.mark.parametrize("in_graph", [False, True])
@pytest.mark.parametrize("value_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kernel_width", [2, 4])
def test_kda_verify_uses_native_conv_in_eager_and_graph(
    monkeypatch: pytest.MonkeyPatch,
    causal_conv1d_reference: list[dict],
    in_graph: bool,
    value_dtype: torch.dtype,
    kernel_width: int,
) -> None:
    module = importlib.import_module("xllm.python.attention.npu_paged_attention")
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._kda_verify_width = 4
    layer = SimpleNamespace(
        head_dim=1,
        num_heads_local=1,
        qkv_dim=1,
        conv_dim=3,
        conv_kernel_size=kernel_width,
        activation="silu",
        layer_id=0,
        conv_weight_t=torch.ones(kernel_width, 3, dtype=value_dtype),
    )
    metadata = SimpleNamespace(
        expanded_decode_metadata=None,
        kv_seq_lens=torch.tensor([1]),
        num_accepted_tokens=torch.ones(1, dtype=torch.int32),
    )
    monkeypatch.setattr(module, "in_acl_graph", lambda: in_graph)

    def recurrent(query: torch.Tensor, *arguments: object, **options: object) -> torch.Tensor:
        return query

    output = backend._spec_verify_v3(
        torch.ones(1, 3, 1, dtype=value_dtype),
        torch.zeros(1, 1, 1),
        torch.ones(1, 1, 1),
        layer,
        torch.tensor([0]),
        metadata,
        torch.zeros(2, kernel_width - 1, 3, dtype=value_dtype),
        torch.zeros(2, 1, 1, 1),
        recurrent,
    )
    assert len(causal_conv1d_reference) == 1
    assert causal_conv1d_reference[0]["activation_mode"] == 1
    assert causal_conv1d_reference[0]["run_mode"] == 1
    assert causal_conv1d_reference[0]["weight"].dtype == value_dtype
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("sequence_count", [1, 4])
@pytest.mark.parametrize("state_length", [1, 3])
def test_kda_conv_tails_preserve_each_proposal_state(
    monkeypatch: pytest.MonkeyPatch, width: int, sequence_count: int, state_length: int
) -> None:
    module = importlib.import_module("xllm.python.attention.npu_paged_attention")
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._kda_verify_width = 4
    capacity = sequence_count * 2 + 1
    conv_dim = 3
    token_count = sequence_count * width
    indices = torch.arange(sequence_count, dtype=torch.int64) * 2 + 1
    mixed = torch.arange(conv_dim * token_count, dtype=torch.float32).reshape(1, conv_dim, token_count)
    conv_cache = torch.arange(capacity * state_length * conv_dim, dtype=torch.float32).reshape(
        capacity, state_length, conv_dim
    )
    initial_cache = conv_cache.clone()
    layer = SimpleNamespace(
        head_dim=1,
        num_heads_local=1,
        qkv_dim=1,
        conv_dim=conv_dim,
        conv_kernel_size=state_length + 1,
        activation="identity",
        layer_id=0,
        conv_weight_t=torch.ones(state_length + 1, conv_dim),
    )
    metadata = SimpleNamespace(
        expanded_decode_metadata=None,
        kv_seq_lens=torch.ones(token_count),
        num_accepted_tokens=torch.ones(sequence_count, dtype=torch.int32),
    )

    def recurrent(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        return value

    monkeypatch.setattr(module, "in_acl_graph", lambda: True)
    backend._spec_verify_v3(
        mixed,
        torch.zeros(1, token_count, 1),
        torch.ones(1, token_count, 1),
        layer,
        indices,
        metadata,
        conv_cache,
        torch.zeros(capacity, 1, 1, 1),
        recurrent,
    )
    expected = torch.zeros_like(backend._kda_v3[0]["combined_conv"])
    values = mixed.reshape(conv_dim, sequence_count, width).permute(1, 2, 0)
    expected_cache = initial_cache.clone()
    for sequence, slot in enumerate(indices.tolist()):
        history = torch.cat([initial_cache[slot], values[sequence]], dim=0)
        for proposal in range(width):
            expected[slot + proposal * capacity] = history[proposal + 1 : proposal + 1 + state_length]
        expected_cache[slot] = expected[slot]
    torch.testing.assert_close(backend._kda_v3[0]["combined_conv"], expected, rtol=0, atol=0)
    torch.testing.assert_close(conv_cache, expected_cache, rtol=0, atol=0)


def _run_kda_verify_step(
    backend: NpuPagedAttentionBackend,
    width: int,
    base_lengths: list[int],
    conv_cache: torch.Tensor,
    ssm_cache: torch.Tensor,
    metadata: SimpleNamespace | None = None,
    layer_id: int = 0,
    accepted_counts: list[int] | None = None,
) -> torch.Tensor:
    sequence_count = len(base_lengths)
    token_count = sequence_count * width
    state_indices = torch.arange(sequence_count, dtype=torch.int64) * 2
    layer = SimpleNamespace(
        head_dim=1,
        num_heads_local=1,
        qkv_dim=1,
        conv_dim=3,
        conv_kernel_size=2,
        activation="identity",
        layer_id=layer_id,
        conv_weight_t=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    )
    if metadata is None:
        metadata = SimpleNamespace(
            expanded_decode_metadata=None,
            kv_seq_lens=(torch.tensor(base_lengths)[:, None] + torch.arange(width)).flatten(),
            num_accepted_tokens=torch.tensor(accepted_counts or [1] * sequence_count, dtype=torch.int32),
        )

    def recurrent_contract(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        offsets = kwargs["cu_seqlens"].tolist()
        assert offsets == list(range(0, token_count + 1, width))
        slots = kwargs["ssm_state_indices"].tolist()
        accepted_counts = kwargs["num_accepted_tokens"].tolist()
        state = kwargs["initial_state"]
        output = torch.empty_like(value)
        for sequence, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            initial_slot = slots[start + accepted_counts[sequence] - 1]
            running_state = state[initial_slot].clone()
            for row in range(start, end):
                running_state = running_state + value[row].unsqueeze(-1)
                state[slots[row]].copy_(running_state)
                output[row].copy_(running_state.squeeze(-1))
        return output

    return backend._spec_verify_v3(
        torch.ones(1, 3, token_count),
        torch.zeros(1, token_count, 1),
        torch.ones(1, token_count, 1),
        layer,
        state_indices,
        metadata,
        conv_cache,
        ssm_cache,
        recurrent_contract,
    )


def test_kda_verify_uses_explicit_acceptance_not_length_delta() -> None:
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._kda_verify_width = 4
    conv_cache = torch.zeros(4, 1, 3)
    ssm_cache = torch.zeros(4, 1, 1, 1)
    metadata = SimpleNamespace(
        expanded_decode_metadata=None,
        kv_seq_lens=torch.tensor([10, 11, 12, 13, 20, 21, 22, 23]),
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
    )
    _run_kda_verify_step(backend, 4, [10, 20], conv_cache, ssm_cache, metadata=metadata)
    metadata.num_accepted_tokens.copy_(torch.tensor([4, 2]))
    output = _run_kda_verify_step(backend, 1, [10, 20], conv_cache, ssm_cache, metadata=metadata)
    torch.testing.assert_close(output.flatten(), torch.tensor([5.0, 3.0], dtype=output.dtype))
    assert "kv_prev" not in backend._kda_v3[0]
    assert "armed_buf" not in backend._kda_v3[0]


@pytest.mark.parametrize("width", [2, 4])
def test_kda_pd_handoff_clears_only_selected_checkpoints(width: int) -> None:
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._kda_verify_width = width
    conv_cache = torch.zeros(4, 1, 3)
    ssm_cache = torch.zeros(4, 1, 1, 1)
    for layer_id in [0, 1]:
        _run_kda_verify_step(backend, width, [10, 20], conv_cache, ssm_cache, layer_id=layer_id)
    snapshots = {
        layer_id: {name: value.clone() for name, value in state.items() if name.startswith("combined_")}
        for layer_id, state in backend._kda_v3.items()
    }
    addresses = {
        (layer_id, name): state[name].data_ptr()
        for layer_id, state in backend._kda_v3.items()
        for name in snapshots[layer_id]
    }

    backend.reset_kda_spec_slots(torch.tensor([0, 0], dtype=torch.int32))

    for layer_id, state in backend._kda_v3.items():
        for name, expected in snapshots[layer_id].items():
            expected[::4].zero_()
            torch.testing.assert_close(state[name], expected)
            assert state[name].data_ptr() == addresses[layer_id, name]
    ssm_cache[0].fill_(100)
    output = _run_kda_verify_step(backend, width, [50, 20], conv_cache, ssm_cache, accepted_counts=[1, width])
    torch.testing.assert_close(output.view(2, width)[0, 0], torch.tensor(101.0, dtype=output.dtype))
    torch.testing.assert_close(
        output.view(2, width)[1, 0],
        snapshots[0]["combined_ssm"][2 + (width - 1) * 4].squeeze().to(output.dtype) + 1,
    )


def test_kda_pd_handoff_before_first_verify_does_not_allocate_state() -> None:
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._kda_verify_width = 4
    backend.reset_kda_spec_slots(torch.empty(0, dtype=torch.int64))
    backend.reset_kda_spec_slots(torch.tensor([0], dtype=torch.int32))
    assert "_kda_v3" not in backend.__dict__


def test_kda_verify_reused_slot_uses_prefill_state_after_acceptance_reset() -> None:
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._kda_verify_width = 4
    conv_cache = torch.zeros(4, 1, 3)
    ssm_cache = torch.zeros(4, 1, 1, 1)
    _run_kda_verify_step(backend, 4, [10, 20], conv_cache, ssm_cache)
    ssm_cache[0].fill_(100)
    ssm_cache[2].fill_(200)
    output = _run_kda_verify_step(backend, 4, [50, 60], conv_cache, ssm_cache, accepted_counts=[1, 1])
    torch.testing.assert_close(output.view(2, 4)[:, 0], torch.tensor([101.0, 201.0], dtype=output.dtype))


def test_kda_verify_reads_live_acceptance_on_every_call() -> None:
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._kda_verify_width = 4
    conv_cache = torch.zeros(4, 1, 3)
    ssm_cache = torch.zeros(4, 1, 1, 1)
    lengths = torch.tensor([10, 11, 12, 13, 20, 21, 22, 23], dtype=torch.int32)
    metadata = SimpleNamespace(
        expanded_decode_metadata=None, kv_seq_lens=lengths, num_accepted_tokens=torch.ones(2, dtype=torch.int32)
    )
    device = torch.device("cpu")
    original_address = lengths.data_ptr()
    with forward_context(ForwardContext(backend, device, metadata, [])):
        _run_kda_verify_step(backend, 4, [10, 20], conv_cache, ssm_cache, metadata=metadata)
        lengths.view(2, 4).add_(torch.tensor([[3], [1]], dtype=torch.int32))
        metadata.num_accepted_tokens.copy_(torch.tensor([3, 1]))
        assert lengths.data_ptr() == original_address
        output = _run_kda_verify_step(backend, 4, [13, 21], conv_cache, ssm_cache, metadata=metadata)
        torch.testing.assert_close(output.view(2, 4)[:, 0], torch.tensor([4.0, 2.0], dtype=output.dtype))
    lengths.add_(1)
    metadata.num_accepted_tokens.fill_(1)
    with forward_context(ForwardContext(backend, device, metadata, [])):
        output = _run_kda_verify_step(backend, 4, [14, 22], conv_cache, ssm_cache, metadata=metadata)
        torch.testing.assert_close(output.view(2, 4)[:, 0], torch.tensor([5.0, 3.0], dtype=output.dtype))


def test_kda_verify_keeps_acceptance_sequence_scoped_with_expanded_metadata() -> None:
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._kda_verify_width = 4
    conv_cache = torch.zeros(8, 1, 3)
    ssm_cache = torch.zeros(8, 1, 1, 1)
    expanded = SimpleNamespace(
        enabled=True,
        kv_seq_lens=torch.tensor([40, 41, 42, 43, 60, 61, 62, 63], dtype=torch.int32),
        block_table=torch.zeros(8, 1, dtype=torch.int32),
        paged_kv_indptr=None,
        paged_kv_indices=None,
        paged_kv_last_page_len=None,
        paged_attention_tiling_data=None,
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=None,
    )
    metadata = SimpleNamespace(
        expanded_decode_metadata=expanded,
        kv_seq_lens=torch.tensor([1, 2]),
        slot_mapping=torch.arange(8),
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
    )
    device = torch.device("cpu")
    with forward_context(ForwardContext(backend, device, metadata, [])):
        _run_kda_verify_step(backend, 4, [40, 60], conv_cache, ssm_cache, metadata=metadata)
        metadata.num_accepted_tokens = torch.ones(4, dtype=torch.int32)
        _run_kda_verify_step(backend, 2, [40, 42, 60, 62], conv_cache, ssm_cache, metadata=metadata)
    expanded.kv_seq_lens.add_(4)
    metadata.num_accepted_tokens = torch.tensor([4, 2], dtype=torch.int32)
    with forward_context(ForwardContext(backend, device, metadata, [])):
        output = _run_kda_verify_step(backend, 4, [44, 64], conv_cache, ssm_cache, metadata=metadata)
        torch.testing.assert_close(output.view(2, 4)[:, 0], torch.tensor([5.0, 4.0], dtype=output.dtype))


def test_kda_verify_selects_boundaries_across_plain_and_verify_steps() -> None:
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._kda_verify_width = 4
    conv_cache = torch.zeros(4, 1, 3)
    ssm_cache = torch.zeros(4, 1, 1, 1)
    _run_kda_verify_step(backend, 1, [10, 20], conv_cache, ssm_cache)
    _run_kda_verify_step(backend, 4, [11, 21], conv_cache, ssm_cache)
    output = _run_kda_verify_step(backend, 1, [14, 25], conv_cache, ssm_cache, accepted_counts=[3, 4])
    torch.testing.assert_close(output.flatten(), torch.tensor([5.0, 6.0], dtype=output.dtype))
    torch.testing.assert_close(ssm_cache[[0, 2]].flatten(), torch.tensor([5.0, 6.0]))


def test_kda_shared_metadata_uses_identical_acceptance_across_layers() -> None:
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._kda_verify_width = 4
    conv_caches = [torch.zeros(4, 1, 3), torch.zeros(4, 1, 3)]
    ssm_caches = [torch.zeros(4, 1, 1, 1), torch.zeros(4, 1, 1, 1)]
    for layer_id, base_length in enumerate((10, 12)):
        _run_kda_verify_step(backend, 4, [base_length], conv_caches[layer_id], ssm_caches[layer_id], layer_id=layer_id)
    metadata = SimpleNamespace(
        expanded_decode_metadata=None,
        kv_seq_lens=torch.arange(14, 18, dtype=torch.int32),
        num_accepted_tokens=torch.tensor([4], dtype=torch.int32),
    )
    with forward_context(ForwardContext(backend, torch.device("cpu"), metadata, [])):
        for layer_id, expected in enumerate((5.0, 5.0)):
            output = _run_kda_verify_step(
                backend, 4, [14], conv_caches[layer_id], ssm_caches[layer_id], metadata=metadata, layer_id=layer_id
            )
            torch.testing.assert_close(output[0, 0], torch.full_like(output[0, 0], expected))
