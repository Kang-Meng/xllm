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

import sys
from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu", reason="NPU paged-attention tests require torch_npu")

from xllm.python.attention.backend import (  # noqa: E402
    LayerCache,
    build_speculative_ssm_state_indices,
)
from xllm.python.attention.npu_paged_attention import (  # noqa: E402
    NpuPagedAttentionBackend,
    _pad_dense_mla_queries,
)
from xllm.python.model_executor.forward_context import (  # noqa: E402
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
)


def test_speculative_ssm_indices_keep_all_bootstrap_checkpoints() -> None:
    state_indices = torch.tensor([3, 7], dtype=torch.int64)

    indices = build_speculative_ssm_state_indices(state_indices, checkpoint_stride=4)

    assert indices.tolist() == [[12, 13, 14, 15], [28, 29, 30, 31]]


pytestmark = pytest.mark.usefixtures("causal_conv1d_reference")


@pytest.mark.parametrize("heads,padded_heads", [(6, 8), (24, 32), (32, 32), (96, 128)])
def test_dense_mla_padding_preserves_real_heads(heads: int, padded_heads: int) -> None:
    query = torch.randn(2, heads, 512)
    position = torch.randn(2, heads, 64)
    padded_query, padded_position = _pad_dense_mla_queries(query, position)
    assert padded_query.shape == (2, padded_heads, 512)
    torch.testing.assert_close(padded_query[:, :heads], query)
    torch.testing.assert_close(padded_position[:, :heads], position)
    assert torch.count_nonzero(padded_query[:, heads:]) == 0
    if heads == padded_heads:
        assert padded_query is query and padded_position is position


def test_dense_mla_padding_leaves_other_head_dimensions_unchanged() -> None:
    query, position = torch.randn(2, 24, 128), torch.randn(2, 24, 64)
    padded_query, padded_position = _pad_dense_mla_queries(query, position)
    assert padded_query is query and padded_position is position


def test_dense_mla_padding_reuses_one_eager_bucket_and_refreshes_inputs() -> None:
    buffers: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]] = {}
    query, position = torch.randn(2, 24, 512), torch.randn(2, 24, 64)
    first = _pad_dense_mla_queries(query, position, eager_buffers=buffers)
    query.add_(1)
    position.sub_(1)
    second = _pad_dense_mla_queries(query, position, eager_buffers=buffers)
    assert all(a.data_ptr() == b.data_ptr() for a, b in zip(first, second))
    torch.testing.assert_close(second[0][:, :24], query)
    torch.testing.assert_close(second[1][:, :24], position)
    assert torch.count_nonzero(second[0][:, 24:]) == 0
    assert torch.count_nonzero(second[1][:, 24:]) == 0
    _pad_dense_mla_queries(query[:1], position[:1], eager_buffers=buffers)
    assert len(buffers) == 1


def test_dense_mla_padding_graph_storage_is_owned_by_entry_and_layer() -> None:
    query, position = torch.randn(2, 24, 512), torch.randn(2, 24, 64)
    states = [AclGraphExecutionState(persistent_buffers={}) for _ in range(2)]
    storage: list[tuple[torch.Tensor, torch.Tensor]] = []
    for state in states:
        context = ForwardContext(None, torch.device("cpu"), None, [], execution_state=state)
        with forward_context(context):
            first = _pad_dense_mla_queries(query, position, layer_id=0)
            again = _pad_dense_mla_queries(query + 1, position + 1, layer_id=0)
            other_layer = _pad_dense_mla_queries(query, position, layer_id=1)
        assert first[0].data_ptr() == again[0].data_ptr()
        assert first[0].data_ptr() != other_layer[0].data_ptr()
        torch.testing.assert_close(first[0][:, :24], query + 1)
        storage.append(first)
    assert storage[0][0].data_ptr() != storage[1][0].data_ptr()


@pytest.mark.parametrize("pad_query_heads", [False, True])
def test_dense_mla_backend_preserves_queries_without_padding_opt_in(
    monkeypatch: pytest.MonkeyPatch, pad_query_heads: bool
) -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend._use_fia_v2 = True
    backend._mla_actual_seq_q_host = [2]
    backend._mla_actual_seq_kv_host = [2]
    backend._mla_eager_query_buffers = {}
    backend._metadata = SimpleNamespace(is_prefill=False, is_chunked_prefill=False)
    backend._causal_mask = None
    backend.scale = 0.125
    query, position = torch.randn(2, 6, 512), torch.randn(2, 6, 64)
    seen: list[torch.Tensor] = []
    kernel_outputs: list[torch.Tensor] = []

    def attention(q: torch.Tensor, *args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        seen.append(q)
        assert kwargs["num_query_heads"] == q.shape[1]
        result = q.clone()
        kernel_outputs.append(result)
        return result, torch.empty(0)

    monkeypatch.setattr(torch.ops.npu, "npu_fused_infer_attention_score_v2", attention, raising=False)
    # Omitting the argument exercises the default used by existing callers.
    options = {"pad_query_heads": True} if pad_query_heads else {}
    with forward_context(ForwardContext(backend, torch.device("cpu"), None, [])):
        output = backend._mla_dense_fia_v2(
            query,
            position,
            torch.empty(1, 128, 1, 512),
            torch.empty(1, 128, 1, 64),
            torch.zeros(1, 1, dtype=torch.int32),
            0,
            **options,
        )
    assert seen[0].shape[1] == (8 if pad_query_heads else 6)
    if not pad_query_heads:
        assert seen[0] is query
        assert output is kernel_outputs[0]
        assert not backend._mla_eager_query_buffers
    torch.testing.assert_close(output, query, rtol=0, atol=0)


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
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    activation: str,
    is_prefill: bool,
) -> None:
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    value = torch.arange(2 * 12 * width, dtype=torch.bfloat16).view(2, width, 12)
    weight = torch.ones(12, 1, 4, dtype=torch.float32)
    layer = SimpleNamespace(
        layer_id=0,
        activation=activation,
        conv_weight_t=weight.squeeze(1).t().to(value.dtype).contiguous(),
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


def test_graph_index_history_uses_full_table_and_masks_unused_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = NpuPagedAttentionBackend(
        num_heads=4,
        num_kv_heads=1,
        head_dim=128,
        scale=128**-0.5,
        sliding_window=0,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    # Cross the former 32K limit while keeping the index width small.
    capacity = 257 * 128
    index = torch.arange(capacity, dtype=torch.float32).reshape(257, 128, 1, 1)
    cache = torch.empty(257, 128, 1, 128)
    layer_cache = LayerCache(key=cache, value=cache, index=index)
    backend.bind_kv_caches([layer_cache])
    metadata = SimpleNamespace(
        block_table=torch.arange(257, dtype=torch.int32).repeat(2, 1),
        kv_seq_lens=torch.tensor([32769, 129], dtype=torch.int32),
    )
    backend._metadata = metadata
    context = ForwardContext(
        attention_backend=backend,
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=[layer_cache],
        execution_state=AclGraphExecutionState(persistent_buffers={}),
    )

    def forbid_host_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("graph history must not read device values on host")

    with forward_context(context), monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbid_host_read)
        patch.setattr(torch.Tensor, "tolist", forbid_host_read)
        patch.setattr(torch.Tensor, "item", forbid_host_read)
        history = backend.gather_index_history(SimpleNamespace(layer_id=0), batch_size=2)
    assert history.shape == (2, capacity, 1)
    torch.testing.assert_close(history[0, :32769, 0], torch.arange(32769, dtype=torch.float32))
    torch.testing.assert_close(history[1, :129, 0], torch.arange(129, dtype=torch.float32))
    assert torch.count_nonzero(history[0, 32769:]) == 0
    assert torch.count_nonzero(history[1, 129:]) == 0


def test_xfia_prepare_preserves_live_device_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = NpuPagedAttentionBackend(
        num_heads=4,
        num_kv_heads=1,
        head_dim=128,
        scale=128**-0.5,
        sliding_window=2048,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        use_xfia_decode=True,
    )
    cache = torch.empty(20, 128, 1, 128, dtype=torch.bfloat16)
    backend.bind_kv_caches([LayerCache(key=cache, value=cache)])
    lengths = torch.tensor([128, 129, 2303, 2304, 2305, 1, 1, 1], dtype=torch.int32)
    table = torch.zeros(8, 20, dtype=torch.int32)
    metadata = SimpleNamespace(
        is_prefill=False, is_chunked_prefill=False, block_table=table, kv_seq_lens=lengths, slot_mapping=torch.arange(8)
    )

    def forbid_host_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("XFIA decode metadata must not be read on host")

    buffers = {}

    def execution_buffer(key: tuple, factory: Callable[[], torch.Tensor]) -> torch.Tensor:
        if key not in buffers:
            buffers[key] = factory()
        return buffers[key]

    with monkeypatch.context() as patch:
        patch.setattr(
            sys.modules["xllm.python.attention.npu_paged_attention"], "get_execution_buffer", execution_buffer
        )
        patch.setattr(torch.Tensor, "cpu", forbid_host_read)
        patch.setattr(torch.Tensor, "tolist", forbid_host_read)
        patch.setattr(torch.Tensor, "item", forbid_host_read)
        backend.prepare(metadata, graph_mode=True)
        row_ends = backend._xfia_query_ends
        graph_lengths = backend._xfia_kv_lengths
        graph_starts = backend._xfia_kv_starts
        lengths.add_(1)
        backend.prepare(metadata, graph_mode=True)
    assert backend._xfia_query_ends is row_ends
    assert backend._xfia_kv_lengths is graph_lengths
    assert backend._xfia_kv_starts is graph_starts
    torch.testing.assert_close(graph_lengths, lengths)
    torch.testing.assert_close(backend._block_table_i32, table)
    assert row_ends.tolist() == list(range(1, 9))
    assert backend._graph_workspace is None
    assert backend._actual_seq_kv == []


@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize("proposal_width", [1, 8])
@pytest.mark.parametrize("sliding_window,expected_starts", [(0, [0, 0, 0]), (2048, [0, 7, 257])])
def test_xfia_single_token_decode_uses_actual_window(
    graph_mode: bool,
    proposal_width: int,
    sliding_window: int,
    expected_starts: list[int],
) -> None:
    backend = NpuPagedAttentionBackend(
        num_heads=4,
        num_kv_heads=1,
        head_dim=128,
        scale=128**-0.5,
        sliding_window=sliding_window,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        use_xfia_decode=True,
        xfia_query_width=proposal_width,
    )
    cache = torch.empty(60, 128, 1, 128, dtype=torch.bfloat16)
    backend.bind_kv_caches([LayerCache(key=cache, value=cache)])
    table = torch.arange(60, dtype=torch.int32).reshape(3, 20)
    lengths = torch.tensor([7, 2055, 2305], dtype=torch.int32)
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        is_spec_verify=False,
        block_table=table,
        kv_seq_lens=lengths,
        slot_mapping=torch.arange(3),
    )
    context = ForwardContext(
        attention_backend=backend,
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=[LayerCache(key=cache, value=cache)],
        execution_state=AclGraphExecutionState(persistent_buffers={}) if graph_mode else None,
    )
    with forward_context(context):
        backend.prepare(metadata, graph_mode=graph_mode)
    assert backend._xfia_query_ends.tolist() == [1, 2, 3]
    assert backend._xfia_kv_starts.tolist() == expected_starts
    torch.testing.assert_close(backend._xfia_kv_lengths, lengths)
    torch.testing.assert_close(backend._block_table_i32, table)


@pytest.mark.parametrize("sliding_window", [0, 2048])
@pytest.mark.parametrize("proposal_width", [8, 15])
def test_xfia_dflash2_chunked_query_keeps_band_window(sliding_window: int, proposal_width: int) -> None:
    backend = NpuPagedAttentionBackend(
        num_heads=4,
        num_kv_heads=1,
        head_dim=128,
        scale=128**-0.5,
        sliding_window=sliding_window,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        use_xfia_decode=True,
        xfia_query_width=proposal_width,
    )
    cache = torch.empty(40, 128, 1, 128, dtype=torch.bfloat16)
    backend.bind_kv_caches([LayerCache(key=cache, value=cache)])
    table = torch.arange(40, dtype=torch.int32).reshape(2, 20)
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=True,
        is_spec_verify=False,
        is_dflash_proposal=True,
        block_table=table,
        kv_seq_lens=torch.tensor([2055, 2305], dtype=torch.int32),
        slot_mapping=torch.arange(2 * proposal_width),
    )
    backend.prepare(metadata)
    assert backend._xfia_query_ends.tolist() == list(range(1, 2 * proposal_width + 1))
    assert backend._xfia_kv_lengths.tolist() == [2055] * proposal_width + [2305] * proposal_width
    expected_starts = [
        max(0, length - proposal_width + offset - sliding_window + 1) if sliding_window else 0
        for length in [2055, 2305]
        for offset in range(proposal_width)
    ]
    assert backend._xfia_kv_starts.tolist() == expected_starts
    torch.testing.assert_close(backend._block_table_i32, table.repeat_interleave(proposal_width, dim=0))


def test_target_spec_verify_chunk_does_not_use_xfia(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = NpuPagedAttentionBackend(
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        scale=0.5,
        sliding_window=2048,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
        use_xfia_decode=True,
        xfia_query_width=8,
    )
    cache = torch.empty(4, 128, 1, 4)
    backend.bind_kv_caches([LayerCache(key=cache, value=torch.empty_like(cache))])
    table = torch.arange(4, dtype=torch.int32).reshape(2, 2)
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=True,
        is_spec_verify=True,
        is_dflash_proposal=False,
        block_table=table,
        kv_seq_lens=torch.full((2,), 136, dtype=torch.int32),
        kv_seq_lens_host_values=[136, 136],
        q_cu_seq_lens=torch.tensor([0, 8, 16], dtype=torch.int32),
        slot_mapping=torch.arange(16),
    )

    def forbid_xfia(*args: object, **kwargs: object) -> None:
        raise AssertionError("target spec-verify must not use XFIA decode")

    monkeypatch.setattr(backend, "_prepare_xfia_decode", forbid_xfia)
    backend.prepare(metadata)
    torch.testing.assert_close(backend._block_table_i32, table)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("is_spec_verify", [False, None])
def test_xfia_matching_prompt_chunk_keeps_causal_prefill(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    is_spec_verify: bool | None,
) -> None:
    width = 8
    backend = NpuPagedAttentionBackend(
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        scale=0.5,
        sliding_window=2048,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
        use_xfia_decode=True,
        xfia_query_width=width,
    )
    cache = torch.empty(batch_size * 2, 128, 1, 4)
    backend.bind_kv_caches([LayerCache(key=cache, value=torch.empty_like(cache))])
    table = torch.arange(batch_size * 2, dtype=torch.int32).reshape(batch_size, 2)
    query_ends = torch.arange(batch_size + 1, dtype=torch.int32) * width
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=True,
        block_table=table,
        kv_seq_lens=torch.full((batch_size,), 128 + width, dtype=torch.int32),
        kv_seq_lens_host_values=[128 + width] * batch_size,
        q_cu_seq_lens=query_ends,
        slot_mapping=torch.arange(batch_size * width),
    )
    if is_spec_verify is not None:
        metadata.is_spec_verify = is_spec_verify

    def forbid_xfia(*args: object, **kwargs: object) -> None:
        raise AssertionError("ordinary prompt chunks must not use XFIA decode")

    monkeypatch.setattr(backend, "_prepare_xfia_decode", forbid_xfia)
    monkeypatch.setattr(backend, "_xfia_decode", forbid_xfia)
    backend.prepare(metadata)
    torch.testing.assert_close(backend._block_table_i32, table)

    module = sys.modules["xllm.python.attention.npu_paged_attention"]
    monkeypatch.setattr(module, "get_forward_context", lambda: SimpleNamespace(cp_context=None))
    monkeypatch.setattr(module.kernels, "reshape_paged_cache", lambda *args: None, raising=False)

    def causal_prefill(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        # Preserve grouped prompt queries and the right-aligned causal mask,
        # including the existing 128-token KV prefix.
        assert kwargs["actual_seq_lengths"] == query_ends[1:].tolist()
        assert kwargs["actual_seq_lengths_kv"] == [128 + width] * batch_size
        assert kwargs["sparse_mode"] == 3
        assert kwargs["atten_mask"] is backend._causal_mask
        torch.testing.assert_close(kwargs["block_table"], table)
        return query.clone(), None

    monkeypatch.setattr(torch.ops.npu, "npu_fused_infer_attention_score", causal_prefill, raising=False)
    query = torch.randn(batch_size * width, 2, 4)
    key = torch.randn(batch_size * width, 1, 4)
    layer = SimpleNamespace(layer_id=0, causal=True, attention_window=None)
    output = backend.execute(query, key, key, layer)
    torch.testing.assert_close(output, query.flatten(1))
