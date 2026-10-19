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

"""Numerical CP row-contract tests, independent of native NPU kernels.

The collective double concatenates rank-major *padded* shards, not an identity.
This catches both unequal ragged row counts and restore-index misalignment.
NPU/HCCL and paged-cache numerical correctness still require hardware tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from xllm.python.attention.backend import MlaIndexContext
from xllm.python.model_executor import cp_utils
from xllm.python.models import glm5_2

from .test_glm5_2_cp import _make_model


def _contexts(q_lens: list[int], cp_size: int, prefix: int = 0) -> list[SimpleNamespace]:
    """Construct zigzag ownership by chunking, then invert the ownership map."""
    rank_rows: list[list[int]] = [[] for _ in range(cp_size)]
    offset = 0
    for length in q_lens:
        width = (length + 2 * cp_size - 1) // (2 * cp_size)
        chunks = [
            [offset + row if row < length else -1 for row in range(chunk * width, (chunk + 1) * width)]
            for chunk in range(2 * cp_size)
        ]
        for rank in range(cp_size):
            rank_rows[rank].extend(chunks[rank] + chunks[-1 - rank])
        offset += length
    rank_major = torch.tensor([row for rows in rank_rows for row in rows], dtype=torch.int64)
    restore = torch.empty(offset, dtype=torch.int64)
    valid = rank_major >= 0
    restore[rank_major[valid]] = torch.arange(rank_major.numel())[valid]
    contexts = []
    for rank, rows in enumerate(rank_rows):
        shard = torch.tensor(rows, dtype=torch.int64)
        contexts.append(
            SimpleNamespace(
                cp_size=cp_size,
                cp_rank=rank,
                total_local=len(rows),
                shard_index=shard,
                shard_valid_mask=shard >= 0,
                shard_gather_index=shard.clamp_min(0),
                restore_index=restore,
                query_index=torch.nonzero(shard >= 0).flatten(),
                has_prefix=prefix > 0,
            )
        )
    return contexts


@pytest.mark.parametrize("q_lens,cp_size", [([1], 2), ([3], 2), ([5], 2), ([8], 2), ([8, 5], 2), ([17], 4)])
@pytest.mark.parametrize("quantized", [False, True])
def test_indexer_gathers_padded_rows_and_restores_global_queries(
    q_lens: list[int], cp_size: int, quantized: bool
) -> None:
    contexts = _contexts(q_lens, cp_size, prefix=7)
    total = sum(q_lens)
    global_hidden = torch.stack((torch.arange(1, total + 1), torch.arange(101, total + 101)), dim=-1).float()
    local_hidden = [cp_utils.cp_shard_rows(global_hidden, ctx) for ctx in contexts]
    expected_q = global_hidden.unsqueeze(1)
    expected_k = global_hidden
    expected_weights = global_hidden[:, :1]
    global_topk = torch.arange(total, dtype=torch.int32).view(-1, 1, 1)

    for rank, cp_context in enumerate(contexts):
        indexer = glm5_2.Glm52Indexer.__new__(glm5_2.Glm52Indexer)
        nn.Module.__init__(indexer)
        indexer.n_head = 1
        indexer.head_dim = 2
        indexer.rope_dim = 0
        indexer.topk = 1
        indexer.indexer_rope_interleave = False
        indexer.wq_b = nn.Identity()
        indexer.wk = nn.Identity()
        indexer.k_norm = nn.Identity()
        indexer.weights_proj = MagicMock(side_effect=lambda value: value[:, :1])
        indexer.hadamard = torch.eye(2)
        update = MagicMock()
        index_cache = torch.empty(16, 2, 1, 2, dtype=torch.int8 if quantized else torch.float32)
        scale_cache = torch.ones(16, 2, 1, 1, dtype=torch.float16) if quantized else None
        table = torch.tensor([[0]], dtype=torch.int32)
        q_cumulative = torch.tensor(q_lens, dtype=torch.int32).cumsum(0).int()
        kv_cumulative = torch.tensor([length + 7 for length in q_lens], dtype=torch.int32).cumsum(0).int()
        ctx = MlaIndexContext(
            index_cache=index_cache,
            slot_mapping=torch.arange(total, dtype=torch.int32),
            block_table=table,
            actual_seq_q=q_cumulative,
            actual_seq_kv=kv_cumulative,
            index_cache_scale=scale_cache,
            get_quant_indexer_metadata=MagicMock(return_value=torch.empty(0)),
            update_index_cache=update,
            materialize_index_cache=MagicMock(return_value=(index_cache, scale_cache, table)),
            cp_context=cp_context,
        )
        collectives = []

        def all_gather(
            value: torch.Tensor,
            dim: int,
            size: int,
            group: str,
            cp_context=cp_context,
            collectives=collectives,
            local_hidden=local_hidden,
            rank=rank,
        ) -> torch.Tensor:
            assert (dim, size, group) == (0, cp_size, "cp")
            assert value.shape[0] == cp_context.total_local
            position = len(collectives)
            collectives.append(position)
            peers = (
                [hidden.unsqueeze(1) for hidden in local_hidden]
                if position == 0
                else local_hidden
                if position == 1
                else [hidden[:, :1] for hidden in local_hidden]
            )
            torch.testing.assert_close(value, peers[rank])
            return torch.cat(peers, dim=0)

        def dynamic_quant(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # Keep values for row-order assertions; this is not a quantization test.
            return value, torch.ones(value.shape[:-1])

        kernel = MagicMock(return_value=global_topk)
        with (
            patch.object(glm5_2, "_apply_half_rope", side_effect=lambda _cache, value, _positions: value),
            patch.object(cp_utils.distributed, "all_gather", side_effect=all_gather, create=True),
            patch.object(glm5_2.kernels, "dynamic_quant", side_effect=dynamic_quant, create=True),
            patch.object(
                indexer,
                "_pad_q_heads_to_kernel_gsize",
                side_effect=lambda q, scale, weights, _heads: (q, scale, weights),
            ),
            patch.object(glm5_2.kernels, "quant_lightning_indexer", kernel, create=True),
            patch.object(glm5_2.kernels, "lightning_indexer", kernel, create=True),
        ):
            output = indexer.select_qli(
                local_hidden[rank], local_hidden[rank], torch.zeros(cp_context.total_local).long(), ctx, torch.empty(0)
            )
        assert len(collectives) == 3
        args = kernel.call_args.args
        if quantized:
            torch.testing.assert_close(args[0], expected_q * 2**-0.5)
            torch.testing.assert_close(update.call_args.args[0], expected_k * 2**-0.5)
            torch.testing.assert_close(args[2], (expected_weights * 2**-0.5).half())
            assert args[6] is q_cumulative and args[7] is kv_cumulative
        else:
            torch.testing.assert_close(args[0], expected_q)
            torch.testing.assert_close(update.call_args.args[0], expected_k)
            torch.testing.assert_close(args[2], expected_weights)
            assert args[3] is q_cumulative and args[4] is kv_cumulative
        torch.testing.assert_close(output, cp_utils.cp_shard_rows(global_topk, cp_context))


def test_attention_passes_full_cp_layout_to_replicated_indexer() -> None:
    attention = glm5_2.Glm52MLAAttention.__new__(glm5_2.Glm52MLAAttention)
    nn.Module.__init__(attention)
    attention.layer_id = 0
    attention.cfg = SimpleNamespace(tp_size=1, layerwise_split_size=1, layerwise_split_rank=0)
    attention.num_heads_local = 1
    attention.qk_nope_head_dim = 1
    attention.qk_rope_head_dim = 1
    attention.kv_lora_rank = 1
    attention.v_head_dim = 1
    attention.q_a_proj = nn.Identity()
    attention.q_a_layernorm = nn.Identity()
    attention.q_b_proj = nn.Identity()
    attention.kv_a_proj_with_mqa = nn.Identity()
    attention.kv_a_layernorm = nn.Identity()
    attention.o_proj = nn.Identity()
    attention.W_UK = torch.ones(1, 1, 1)
    attention.W_UV = torch.ones(1, 1, 1)
    attention.indexer = MagicMock()
    cp_context = _contexts([5], 2)[0]
    hidden = torch.ones(cp_context.total_local, 2)
    positions = torch.zeros(cp_context.total_local).long()
    assert cp_context.query_index.numel() != cp_context.total_local
    backend = MagicMock()
    backend.execute_mla.return_value = torch.ones(cp_context.total_local, 1, 1)
    with (
        patch.object(
            glm5_2,
            "get_forward_context",
            return_value=SimpleNamespace(cp_context=cp_context, attention_backend=backend),
        ),
        patch.object(glm5_2, "_gather_interleave_cos_sin", return_value=(torch.empty(0), torch.empty(0))),
        patch.object(glm5_2, "_interleave_rope_with", side_effect=lambda value, *_args: value),
    ):
        attention(hidden, positions, torch.empty(0))
    args = attention.indexer.select_qli.call_args.args
    assert args[0] is hidden and args[1] is hidden and args[2] is positions
    assert args[0].shape[0] == cp_context.total_local


@pytest.mark.parametrize("q_lens", [[1], [5], [8, 5]])
def test_cp_aux_hidden_is_merged_with_target_in_capture_order(q_lens: list[int]) -> None:
    contexts = _contexts(q_lens, 2)
    ids = torch.arange(10, 10 + sum(q_lens))
    global_output = (2 * ids + 3).float().unsqueeze(-1)
    # Deliberately non-monotonic capture order: layer 1 then layer 0.
    global_aux = torch.stack((2 * ids + 3, 2 * ids + 1), dim=-1).float()
    global_fused = torch.cat((global_output, global_aux), dim=-1)
    for rank, cp_context in enumerate(contexts):
        model, _ = _make_model([])
        model.aux_hidden_capture = glm5_2.AuxHiddenCapture((1, 0))
        calls = []

        def all_gather(
            value: torch.Tensor,
            dim: int,
            size: int,
            group: str,
            cp_context=cp_context,
            calls=calls,
            rank=rank,
        ) -> torch.Tensor:
            assert (dim, size, group) == (0, 2, "cp")
            assert value.shape == (cp_context.total_local, 3)
            calls.append(value)
            peers = [cp_utils.cp_shard_rows(global_fused, ctx) for ctx in contexts]
            # Padding is discarded by restore_index; only real rows are compared.
            torch.testing.assert_close(value[cp_context.shard_valid_mask], peers[rank][cp_context.shard_valid_mask])
            return torch.cat(peers, dim=0)

        with (
            patch.object(glm5_2, "get_forward_context", return_value=SimpleNamespace(cp_context=cp_context)),
            patch.object(cp_utils.distributed, "all_gather", side_effect=all_gather, create=True),
            patch.object(glm5_2, "record_layer_event"),
        ):
            output, aux = model(ids, torch.arange(ids.numel()).int())
        assert len(calls) == 1
        assert model.restores_aux_hidden_under_cp is True
        torch.testing.assert_close(output, global_output)
        torch.testing.assert_close(aux, global_aux)


def test_indexer_padded_heads_have_neutral_nonzero_scale() -> None:
    indexer = glm5_2.Glm52Indexer.__new__(glm5_2.Glm52Indexer)
    nn.Module.__init__(indexer)
    indexer.n_head = 32
    q = torch.ones(3, 32, 128, dtype=torch.int8)
    scale = torch.full((3, 32), 0.25, dtype=torch.float16)
    weights = torch.ones(3, 32, dtype=torch.float16)
    padded_q, padded_scale, padded_weights = indexer._pad_q_heads_to_kernel_gsize(q, scale, weights, 64)
    torch.testing.assert_close(padded_q[:, :32], q)
    torch.testing.assert_close(padded_scale[:, :32], scale)
    torch.testing.assert_close(padded_weights[:, :32], weights)
    assert torch.count_nonzero(padded_q[:, 32:]) == 0
    assert torch.count_nonzero(padded_weights[:, 32:]) == 0
    assert torch.all(padded_scale[:, 32:] == 1)
