# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""CPU tests for SFA DCP graph-prepare indexer paging."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

pytest.importorskip("torch_npu", reason="SFA DCP backend tests import the NPU attention backend")

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.npu_paged_attention import write_mla_paged_cache
from xllm.python.attention.sfa_dcp_backend import SfaDcpAttentionBackend
from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
)


class _FakeDcpGroup:
    def size(self) -> int:
        return 4

    def rank(self) -> int:
        return 0


def _cpu_context(execution_state: AclGraphExecutionState) -> ForwardContext:
    return ForwardContext(
        attention_backend=MagicMock(),
        device=torch.device("cpu"),
        metadata=MagicMock(),
        layer_caches=[],
        execution_state=execution_state,
    )


def test_graph_prepare_keeps_valid_indexer_pages_for_padded_lanes() -> None:
    backend = SfaDcpAttentionBackend(
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
        scale=0.1,
        sliding_window=0,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        dcp_group=_FakeDcpGroup(),
        index_topk=2048,
        max_num_reqs=8,
    )
    page_size = 128
    backend.bind_kv_caches(
        [
            LayerCache(
                key=torch.empty(16, page_size, 1, 512),
                value=torch.empty(16, page_size, 1, 64),
                index=torch.empty(64, page_size, 1, 128),
            )
        ]
    )

    block_table = torch.zeros((8, 2), dtype=torch.int32)
    block_table[:7] = torch.tensor([[1, 2]] * 7, dtype=torch.int32)
    slot_mapping = torch.tensor([0, 1, 2, 3, 4, 5, 6, -1], dtype=torch.int32)
    kv_seq_lens = torch.tensor([1022, 1022, 1022, 1022, 1022, 1022, 1022, 1], dtype=torch.int32)
    metadata = SimpleNamespace(
        slot_mapping=slot_mapping,
        block_table=block_table,
        kv_seq_lens=kv_seq_lens,
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=None,
        q_cu_seq_lens=None,
        q_seq_lens=None,
        expanded_decode_metadata=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )

    with forward_context(_cpu_context(AclGraphExecutionState({}))):
        backend.prepare(metadata, graph_mode=True)

    expanded = backend._expanded_indexer_block_table
    assert expanded is not None
    assert (expanded[-1] >= 0).all()
    assert torch.equal(expanded[-1], torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32))
    assert torch.equal(expanded[0, :4], torch.tensor([4, 5, 6, 7], dtype=torch.int32))


def test_bind_kv_caches_accepts_missing_nope_rope_cache() -> None:
    backend = SfaDcpAttentionBackend(
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
        scale=0.1,
        sliding_window=0,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        dcp_group=_FakeDcpGroup(),
        index_topk=2048,
        max_num_reqs=8,
    )
    page_size = 128
    nope_cache = torch.empty(16, page_size, 1, 512)
    backend.bind_kv_caches(
        [
            LayerCache(
                key=nope_cache,
                value=None,
                index=torch.empty(64, page_size, 1, 128),
            )
        ]
    )
    bound = backend._kv_caches[0]
    assert bound.key is nope_cache
    assert bound.value is None
    assert backend.is_mla


def test_write_mla_paged_cache_nope_reuses_latent_cache() -> None:
    k_latent = torch.randn(2, 1, 16)
    nope_cache = torch.randn(4, 8, 1, 16)
    slots = torch.zeros(2, dtype=torch.int32)
    with patch.object(torch.ops.xllm_ops, "reshape_paged_cache", create=True) as reshape:
        write_mla_paged_cache(slots, k_latent, None, nope_cache, None)
    assert reshape.call_args.args == (slots, k_latent, k_latent, nope_cache, nope_cache)


def test_write_mla_paged_cache_nope_ignores_zero_width_rope_cache() -> None:
    k_latent = torch.randn(2, 1, 16)
    nope_cache = torch.randn(4, 8, 1, 16)
    rope_cache = torch.empty(4, 8, 1, 0)
    slots = torch.zeros(2, dtype=torch.int32)
    with patch.object(torch.ops.xllm_ops, "reshape_paged_cache", create=True) as reshape:
        write_mla_paged_cache(slots, k_latent, None, nope_cache, rope_cache)
    assert reshape.call_args.args[4] is nope_cache


def test_write_mla_paged_cache_keeps_glm52_rope() -> None:
    k_latent = torch.randn(2, 1, 16)
    k_pe = torch.randn(2, 1, 4)
    nope_cache = torch.randn(4, 8, 1, 16)
    rope_cache = torch.randn(4, 8, 1, 4)
    slots = torch.zeros(2, dtype=torch.int32)
    with patch.object(torch.ops.xllm_ops, "reshape_paged_cache", create=True) as reshape:
        write_mla_paged_cache(slots, k_latent, k_pe, nope_cache, rope_cache)
    assert reshape.call_args.args == (slots, k_latent, k_pe, nope_cache, rope_cache)


def test_execute_mla_nope_accepts_none_q_pe_and_k_pe() -> None:
    backend = SfaDcpAttentionBackend(
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
        scale=0.1,
        sliding_window=0,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        dcp_group=_FakeDcpGroup(),
        index_topk=2048,
        max_num_reqs=8,
    )
    page_size = 128
    nope_cache = torch.empty(16, page_size, 1, 512)
    rope_cache = torch.empty(16, page_size, 1, 0)
    layer_cache = LayerCache(
        key=nope_cache,
        value=rope_cache,
        index=torch.empty(64, page_size, 1, 128),
    )
    backend.bind_kv_caches([layer_cache])

    block_table = torch.zeros((1, 2), dtype=torch.int32)
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0], dtype=torch.int32),
        block_table=block_table,
        kv_seq_lens=torch.tensor([8], dtype=torch.int32),
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=None,
        q_cu_seq_lens=None,
        q_seq_lens=None,
        expanded_decode_metadata=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    q_latent = torch.randn(1, 8, 512)
    k_latent = torch.randn(1, 1, 512)
    topk = torch.zeros(1, 1, 2048, dtype=torch.int32)
    layer = SimpleNamespace(layer_id=0)
    attn_out = torch.randn(1, 8, 512)
    context = ForwardContext(
        attention_backend=backend,
        device=torch.device("cpu"),
        metadata=MagicMock(),
        layer_caches=[layer_cache],
        execution_state=None,
    )
    assert backend._impl is not None
    with (
        forward_context(context),
        patch.object(torch.ops.xllm_ops, "reshape_paged_cache", create=True) as reshape_paged_cache,
        patch.object(backend._impl, "_store_parallel_kv", return_value=(None, k_latent, None)),
        patch.object(backend._impl, "_record_query_gather_context") as record_query,
        patch.object(
            backend._impl,
            "_execute_sparse_flash_attention_process",
            return_value=attn_out,
        ) as execute,
    ):
        backend.prepare(metadata, graph_mode=False)
        out = backend.execute_mla(q_latent, None, k_latent, None, layer, topk=topk)
    assert out is attn_out
    reshape_args = reshape_paged_cache.call_args.args
    assert reshape_args[1] is k_latent
    assert reshape_args[2] is k_latent
    assert reshape_args[3] is nope_cache
    assert reshape_args[4] is nope_cache
    record_query.assert_called_once()
    assert record_query.call_args.args[1] is None
    execute.assert_called_once()
    assert execute.call_args.args[1] is None
    assert execute.call_args.args[2][1] is rope_cache


def test_gather_index_history_dcp_covers_tokens_in_one_logical_page() -> None:
    """Prefill warmup can sit in one logical DCP page.

    With dcp=4 and page_size=128 a logical page covers 512 tokens, so a 256-token
    sequence is one engine block-table column. The indexer cache is still paged
    at page_size=128; gather must walk the expanded table (4 physical pages) or
    it writes 128 rows into a 256-row target.
    """
    backend = SfaDcpAttentionBackend(
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
        scale=0.1,
        sliding_window=0,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        dcp_group=_FakeDcpGroup(),
        index_topk=2048,
        max_num_reqs=8,
    )
    page_size = 128
    width = 257
    n_phys = 16
    index = torch.zeros(n_phys, page_size, 1, width)
    index[:, :, 0, -1] = torch.arange(n_phys, dtype=index.dtype).unsqueeze(1)
    backend.bind_kv_caches(
        [
            LayerCache(
                key=torch.empty(n_phys, page_size, 1, 512),
                value=torch.empty(n_phys, page_size, 1, 0),
                index=index,
            )
        ]
    )

    kv_len = 256
    logical_block_table = torch.tensor([[0]], dtype=torch.int32)
    metadata = SimpleNamespace(
        slot_mapping=torch.arange(kv_len, dtype=torch.int32),
        block_table=logical_block_table,
        kv_seq_lens=torch.tensor([kv_len], dtype=torch.int32),
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=None,
        q_cu_seq_lens=None,
        q_seq_lens=None,
        expanded_decode_metadata=None,
        is_prefill=True,
        is_chunked_prefill=False,
        has_kv_shard=False,
        kv_split_size=4,
    )
    backend.prepare(metadata, graph_mode=False)

    expanded = backend.indexer_block_table()
    assert torch.equal(expanded[0], torch.tensor([0, 1, 2, 3], dtype=torch.int32))

    packed = backend.gather_index_history(SimpleNamespace(layer_id=0), batch_size=kv_len)
    assert packed.shape == (1, kv_len, width)
    assert torch.equal(packed[0, :page_size, -1], torch.zeros(page_size))
    assert torch.equal(packed[0, page_size:kv_len, -1], torch.ones(page_size))
