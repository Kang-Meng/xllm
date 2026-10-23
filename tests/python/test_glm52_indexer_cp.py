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

"""glm5_2 indexer CP query-row partition tests (S2 non-quant + S3 quant + S4 gates).

Mirrors test_glm5_next_indexer_cp.py's monkeypatched CPU style: the
lightning_indexer / quant_lightning_indexer kernels are replaced by
reference implementations of the S0-verified varlen contract (TND
cumulative query_seq_lengths, per-segment key_seq_lengths, one
block-table row per segment, sparse_mode=3 right-aligned causal), so
partitioned and full-stream runs are comparable without native NPU
kernels. S0 contract #6 -- row-internal top-k order varies with batch
composition -- means the equality assertions compare each row's sorted
index SET, never a bit-level tensor.

Every test pins the gate explicitly (patching the module-private
``glm5_2._INDEXER_CP_QUERY_PARTITION`` seam), so the file is independent of
the ambient environment; the shipped build has the partition always on and
exposes no environment override.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from xllm.python.attention.backend import MlaIndexContext
from xllm.python.model_executor import cp_utils
from xllm.python.model_executor.cp_utils import CpContext
from xllm.python.models import glm5_2

# Small indexer geometry: hidden == N_HEAD * HEAD_DIM so nn.Identity stands
# in for wq_b, the gathered query rows are [rows, N_HEAD, HEAD_DIM] and the
# gathered weights [rows, N_HEAD]. The key side drops to HEAD_DIM after the
# rope split, so nn.Identity also stands in for wk / k_norm.
N_HEAD = 2
HEAD_DIM = 4
HIDDEN = N_HEAD * HEAD_DIM
BLOCK_SIZE = 4
TOPK = 3
N_DECOY_BLOCKS = 3
INT64_MAX = 9223372036854775807


def _build_cp_context(q_lens: list[int], kv_lens: list[int], cp_size: int, cp_rank: int) -> CpContext:
    """Python mirror of the xllm_ops::build_cp_context host index math.

    The op itself is native (unavailable to CPU tests), so this reproduces
    the zigzag plan exactly as npu_ops_library.cpp builds it: two owned
    chunks per sequence, per-segment cumulative query counts, per-segment
    absolute KV prefix lengths and per-sequence block-table rows.
    """
    num_chunks = cp_size * 2
    first_chunk = cp_rank
    second_chunk = num_chunks - 1 - cp_rank
    shard_index: list[int] = []
    query_index: list[int] = []
    q_cu_seqlens: list[int] = []
    kv_gather_index: list[int] = []
    kv_cu_seqlens: list[int] = []
    segment_seq_indices: list[int] = []
    segment_kv_seq_lens: list[int] = []
    chunk_lens: list[int] = []
    local_seg_offsets: list[int] = []
    global_offset = 0
    local_offset = 0
    q_run = 0
    kv_run = 0
    for seq_id, (length, kv_length) in enumerate(zip(q_lens, kv_lens, strict=True)):
        prefix_cache_length = kv_length - length
        padded = ((length + num_chunks - 1) // num_chunks) * num_chunks
        chunk_len = padded // num_chunks
        chunk_lens.append(chunk_len)
        local_seg_offsets.append(local_offset)
        for half, chunk_id in ((0, first_chunk), (1, second_chunk)):
            seg_local_base = local_offset + half * chunk_len
            seg_start = chunk_id * chunk_len
            real_count = 0
            for j in range(chunk_len):
                pos_in_seq = seg_start + j
                if pos_in_seq < length:
                    shard_index.append(global_offset + pos_in_seq)
                    query_index.append(seg_local_base + j)
                    real_count += 1
                else:
                    shard_index.append(-1)
            if real_count > 0:
                segment_seq_indices.append(seq_id)
                prefix_len = seg_start + real_count
                segment_kv_seq_lens.append(prefix_cache_length + prefix_len)
                q_run += real_count
                q_cu_seqlens.append(q_run)
                kv_gather_index.extend(global_offset + p for p in range(prefix_len))
                kv_run += prefix_len
                kv_cu_seqlens.append(kv_run)
        global_offset += length
        local_offset += 2 * chunk_len
    total_local = local_offset
    restore_index: list[int] = []
    for seq_id, length in enumerate(q_lens):
        chunk_len = chunk_lens[seq_id]
        seg_offset = local_seg_offsets[seq_id]
        for pos_in_seq in range(length):
            chunk_id = pos_in_seq // chunk_len
            row_in_chunk = pos_in_seq % chunk_len
            if chunk_id < cp_size:
                owner_rank = chunk_id
                local_pos = seg_offset + row_in_chunk
            else:
                owner_rank = num_chunks - 1 - chunk_id
                local_pos = seg_offset + chunk_len + row_in_chunk
            restore_index.append(owner_rank * total_local + local_pos)
    shard = torch.tensor(shard_index, dtype=torch.int64)
    valid = shard >= 0
    return CpContext(
        cp_size=cp_size,
        cp_rank=cp_rank,
        total_local=total_local,
        shard_index=shard,
        shard_gather_index=shard.clamp_min(0),
        shard_valid_mask=valid,
        restore_index=torch.tensor(restore_index, dtype=torch.int64),
        query_index=torch.tensor(query_index, dtype=torch.int64),
        q_cu_seqlens=q_cu_seqlens,
        q_cu_seqlens_tensor=torch.tensor(q_cu_seqlens, dtype=torch.int32),
        kv_gather_index=torch.tensor(kv_gather_index, dtype=torch.int64),
        kv_cu_seqlens=kv_cu_seqlens,
        segment_seq_indices=torch.tensor(segment_seq_indices, dtype=torch.int64),
        segment_kv_seq_lens=segment_kv_seq_lens,
        segment_kv_seq_lens_tensor=torch.tensor(segment_kv_seq_lens, dtype=torch.int32),
        has_prefix=any(kv != q for q, kv in zip(q_lens, kv_lens, strict=True)),
    )


def _reference_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_seq_lengths: torch.Tensor,
    key_seq_lengths: torch.Tensor,
    block_table: torch.Tensor,
    layout_query: str,
    layout_key: str,
    selected_count: int,
    sparse_mode: int,
    pre_tokens: int,
    next_tokens: int,
    return_value: bool,
) -> torch.Tensor:
    """CPU reference for the S0-verified lightning_indexer varlen contract.

    ``query`` is TND [rows, heads, dim]; ``key`` is the paged PA_BSND cache
    [blocks, block_size, kv_heads, dim]; ``query_seq_lengths`` holds the
    cumulative per-batch query ends; ``key_seq_lengths`` the per-batch KV
    lengths; ``block_table`` has one row per batch. sparse_mode=3 is
    right-aligned causal, so batch row i attends KV [0, kv_len - q_len + i].
    """
    assert (layout_query, layout_key, sparse_mode, return_value) == ("TND", "PA_BSND", 3, False)
    assert pre_tokens == next_tokens == INT64_MAX
    q_ends = [int(value) for value in query_seq_lengths.tolist()]
    kv_lens = [int(value) for value in key_seq_lengths.tolist()]
    block_size = key.shape[1]
    out = torch.full((query.shape[0], key.shape[2], selected_count), -1, dtype=torch.int32)
    start = 0
    for batch, q_end in enumerate(q_ends):
        q_len = q_end - start
        kv_len = kv_lens[batch]
        table = block_table[batch]
        for i in range(q_len):
            visible = kv_len - q_len + i + 1
            scored = []
            for position in range(visible):
                block = int(table[position // block_size])
                k_vec = key[block, position % block_size, 0].float()
                score = 0.0
                for head in range(query.shape[1]):
                    score += float(weights[start + i, head]) * float(torch.dot(query[start + i, head].float(), k_vec))
                scored.append((score, position))
            scored.sort(key=lambda item: (-item[0], item[1]))
            chosen = [position for _score, position in scored[:selected_count]]
            out[start + i, 0, : len(chosen)] = torch.tensor(chosen, dtype=torch.int32)
        start = q_end
    return out


def _make_indexer() -> glm5_2.Glm52Indexer:
    """A minimal Glm52Indexer whose per-row projections are channel selects.

    ``wq_b`` is the identity (so the gathered query rows are the hidden
    rows re-viewed), while ``wk`` / ``weights_proj`` pick the key and weight
    channels out of the shared hidden input.
    """
    indexer = glm5_2.Glm52Indexer.__new__(glm5_2.Glm52Indexer)
    nn.Module.__init__(indexer)
    indexer.n_head = N_HEAD
    indexer.head_dim = HEAD_DIM
    indexer.rope_dim = 0
    indexer.topk = TOPK
    indexer.indexer_rope_interleave = False
    indexer.wq_b = nn.Identity()
    indexer.wk = MagicMock(side_effect=lambda value: value[:, :HEAD_DIM])
    indexer.k_norm = nn.Identity()
    indexer.weights_proj = MagicMock(side_effect=lambda value: value[:, :N_HEAD])
    indexer.hadamard = torch.eye(HEAD_DIM)
    return indexer


class _Scenario:
    """Synthetic CP prefill inputs shared by every rank of one batch."""

    def __init__(self, q_lens: list[int], prefix: int, cp_size: int, seed: int = 42) -> None:
        torch.manual_seed(seed)
        self.cp_size = cp_size
        self.total = sum(q_lens)
        kv_lens = [length + prefix for length in q_lens]
        self.contexts = [_build_cp_context(q_lens, kv_lens, cp_size, rank) for rank in range(cp_size)]
        # hidden doubles as the query-lora input and the key-side input: the
        # projections above are identities, so the gathered kernel arguments
        # are these global rows verbatim (the key side narrows to HEAD_DIM
        # after the rope split, the weights to the first N_HEAD channels).
        self.global_hidden = torch.randn(self.total, HIDDEN)
        self.global_q = self.global_hidden.view(self.total, N_HEAD, HEAD_DIM)
        self.global_k = self.global_hidden[:, :HEAD_DIM]
        self.global_weights = self.global_hidden[:, :N_HEAD]
        self.local_q = [cp_utils.cp_shard_rows(self.global_q, context) for context in self.contexts]
        self.local_k = [cp_utils.cp_shard_rows(self.global_k, context) for context in self.contexts]
        self.local_weights = [cp_utils.cp_shard_rows(self.global_weights, context) for context in self.contexts]
        # Paged index cache with per-sequence disjoint block pools plus
        # decoy blocks, so a wrong block-table row changes the scores.
        max_blocks = (max(kv_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
        pool = max_blocks + N_DECOY_BLOCKS
        tables = []
        for seq in range(len(q_lens)):
            blocks = torch.randperm(pool) + seq * pool
            tables.append(blocks[:max_blocks].to(torch.int32))
        self.block_table = torch.stack(tables)
        self.index_cache = torch.randn(len(q_lens) * pool, BLOCK_SIZE, 1, HEAD_DIM)
        # Full-stream arguments, mirroring NpuPagedAttentionBackend:
        # cumulative per-sequence query ends, absolute per-sequence KV lengths.
        self.actual_seq_q = torch.tensor(q_lens, dtype=torch.int32).cumsum(0).int()
        self.actual_seq_kv = torch.tensor(kv_lens, dtype=torch.int32)


def _run_select_qli(
    scenario: _Scenario,
    rank: int,
    partition: bool,
    *,
    has_kv_shard: bool = False,
    non_cp: bool = False,
    cp_context_override: object = None,
) -> tuple[torch.Tensor, list[tuple], list[torch.Tensor]]:
    """Run select_qli for one CP rank; returns (output, kernel calls, shard inputs).

    The CP collectives are replaced by a rank-major all_gather over the
    precomputed local shards, cp_shard_rows records the full-layout tensor
    it reslices, and lightning_indexer runs the CPU reference above. The
    keyword arguments drive the S4 alpha_active gate cases: ``non_cp``
    feeds the whole batch the way a non-CP forward would (no gather, no
    shard exit), ``cp_context_override`` swaps the context the model sees
    (duck-typed stand-ins) while this harness keeps sharding with the real
    one, and ``has_kv_shard`` marks a shard-aware index cache.
    """
    context = scenario.contexts[rank]
    indexer = _make_indexer()
    if non_cp:
        ctx_cp_context = None
        local_hidden = scenario.global_hidden
        n_positions = scenario.total
    else:
        ctx_cp_context = cp_context_override if cp_context_override is not None else context
        local_hidden = cp_utils.cp_shard_rows(scenario.global_hidden, context)
        n_positions = context.total_local
    kernel_calls: list[tuple] = []
    shard_inputs: list[torch.Tensor] = []

    def kernel(*args: object, **_kwargs: object) -> torch.Tensor:
        kernel_calls.append(args)
        assert isinstance(args, tuple)
        return _reference_lightning_indexer(*args)

    def all_gather(value: torch.Tensor, dim: int, size: int, group: str) -> torch.Tensor:
        assert (dim, size, group) == (0, scenario.cp_size, "cp")
        if value.dim() == 3:
            peers = scenario.local_q
        elif value.shape[-1] == HEAD_DIM:
            peers = scenario.local_k
        else:
            peers = scenario.local_weights
        torch.testing.assert_close(value, peers[rank])
        return torch.cat(peers, dim=0)

    def shard_rows(value: torch.Tensor, cp_context: CpContext) -> torch.Tensor:
        shard_inputs.append(value)
        return cp_utils.cp_shard_rows(value, cp_context)

    ctx = MlaIndexContext(
        index_cache=scenario.index_cache,
        slot_mapping=torch.arange(scenario.total, dtype=torch.int32),
        block_table=scenario.block_table,
        actual_seq_q=scenario.actual_seq_q,
        actual_seq_kv=scenario.actual_seq_kv,
        index_cache_scale=None,
        get_quant_indexer_metadata=MagicMock(return_value=torch.empty(0)),
        update_index_cache=MagicMock(),
        materialize_index_cache=MagicMock(return_value=(scenario.index_cache, None, scenario.block_table)),
        cp_context=ctx_cp_context,
        has_kv_shard=has_kv_shard,
    )

    with (
        patch.object(glm5_2, "_INDEXER_CP_QUERY_PARTITION", partition),
        patch.object(glm5_2, "_apply_half_rope", side_effect=lambda _cache, value, _positions: value),
        patch.object(cp_utils.distributed, "all_gather", side_effect=all_gather, create=True),
        patch.object(glm5_2, "cp_shard_rows", side_effect=shard_rows),
        patch.object(glm5_2.kernels, "lightning_indexer", side_effect=kernel, create=True),
    ):
        output = indexer.select_qli(
            local_hidden,
            local_hidden,
            torch.zeros(n_positions, dtype=torch.int64),
            ctx,
            torch.empty(0),
        )
    return output, kernel_calls, shard_inputs


@pytest.mark.parametrize(
    "q_lens,cp_size,prefix",
    [
        ([13], 2, 6),
        ([7], 4, 3),
        ([9, 6], 2, 4),
    ],
)
def test_partition_matches_full_stream_for_owned_rows(q_lens: list[int], cp_size: int, prefix: int) -> None:
    """The partitioned top-k equals the full-stream top-k for owned rows.

    S0 contract #6: row-internal order varies with batch composition, so
    each row's indices are sorted before comparing; a bit-level
    torch.equal would false-fail against the real kernel.
    """
    scenario = _Scenario(q_lens, prefix, cp_size)
    for rank in range(cp_size):
        context = scenario.contexts[rank]
        partitioned, partition_calls, _shard_inputs = _run_select_qli(scenario, rank, partition=True)
        full, full_calls, _ = _run_select_qli(scenario, rank, partition=False)
        if context.query_index.numel() == 0:
            # An empty rank owns no real row: the partition skips the
            # scoring call while the full stream still runs it.
            assert not partition_calls and full_calls
            assert torch.equal(partitioned, full)
            continue
        assert len(partition_calls) == 1 and len(full_calls) == 1
        # Only this rank's real rows are scored, packed per segment.
        assert partition_calls[0][0].shape[0] == context.query_index.numel()
        assert full_calls[0][0].shape[0] == scenario.total
        assert torch.equal(
            partitioned[context.query_index].sort(dim=-1).values,
            full[context.query_index].sort(dim=-1).values,
        )
        # Padding rows are zeroed by cp_shard_rows on both paths.
        padding = ~context.shard_valid_mask
        assert torch.equal(partitioned[padding], full[padding])
        assert torch.equal(partitioned[padding], torch.zeros_like(partitioned[padding]))


def test_gate_off_runs_the_byte_identical_full_stream_path() -> None:
    """The seam off must reproduce the legacy (full-stream) output."""
    scenario = _Scenario([13], 6, 2)
    rank = 1
    context = scenario.contexts[rank]
    output, calls, _ = _run_select_qli(scenario, rank, partition=False)
    # The kernel saw the full-stream arguments verbatim: the complete
    # merged stream and the backend's per-sequence seq tensors and table.
    args = calls[0]
    assert args[0].shape[0] == scenario.total
    assert args[3] is scenario.actual_seq_q
    assert args[4] is scenario.actual_seq_kv
    assert args[5] is scenario.block_table
    # Byte-identical reconstruction of the pre-partition path: one
    # full-stream kernel call followed by the cp_shard_rows exit.
    legacy = cp_utils.cp_shard_rows(
        _reference_lightning_indexer(
            scenario.global_q,
            scenario.index_cache,
            scenario.global_weights,
            scenario.actual_seq_q,
            scenario.actual_seq_kv,
            scenario.block_table,
            "TND",
            "PA_BSND",
            TOPK,
            3,
            INT64_MAX,
            INT64_MAX,
            False,
        ),
        context,
    )
    assert torch.equal(output, legacy)


def test_partition_scatters_minus_one_outside_owned_rows() -> None:
    """Non-owned rows stay -1 in the scattered full-layout topk."""
    scenario = _Scenario([13], 6, 2)
    rank = 1
    context = scenario.contexts[rank]
    _output, _calls, shard_inputs = _run_select_qli(scenario, rank, partition=True)
    full_layout = shard_inputs[0]
    assert full_layout.shape[0] == scenario.total
    rows = cp_utils.cp_real_query_rows(context)
    outside = torch.ones(scenario.total, dtype=torch.bool)
    outside[rows] = False
    assert bool(outside.any())
    assert bool((full_layout[outside] == -1).all()), "rows outside query_rows must stay -1 in the full-layout topk"
    # Every owned row still carries a real selection (no all -1 row).
    assert not bool((full_layout[rows] == -1).all(dim=-1).any())


def test_partition_builds_per_segment_metadata_across_sequences() -> None:
    """design.md open question 2: segments = 2 per sequence.

    A multi-sequence batch packs two (sequence, half) segments per
    sequence; the seq tensors and the block table must be built per segment
    across the whole batch, and the partition must still match the full
    stream.
    """
    scenario = _Scenario([9, 6], 4, 2)
    rank = 1
    context = scenario.contexts[rank]
    # Rank 1 owns seq 0's chunks 1 and 2 (rows 3-5 and 6-8, chunk_len 3) plus
    # seq 1's chunks 1 and 2 (rows 2-3 and 4-5, chunk_len 2): four segments,
    # two per sequence, packed in (sequence, half) order.
    assert context.q_cu_seqlens == [3, 6, 8, 10]
    assert context.segment_kv_seq_lens == [10, 13, 8, 10]
    assert context.segment_seq_indices.tolist() == [0, 0, 1, 1]
    partitioned, calls, _ = _run_select_qli(scenario, rank, partition=True)
    full, _, _ = _run_select_qli(scenario, rank, partition=False)
    args = calls[0]
    assert args[3] is context.q_cu_seqlens_tensor
    assert args[4] is context.segment_kv_seq_lens_tensor
    assert args[5].shape[0] == len(context.q_cu_seqlens)
    assert torch.equal(args[5], scenario.block_table.index_select(0, context.segment_seq_indices))
    assert torch.equal(
        partitioned[context.query_index].sort(dim=-1).values,
        full[context.query_index].sort(dim=-1).values,
    )


def test_partition_empty_rank_skips_scoring_and_matches_full_stream() -> None:
    """A rank can own zero real rows (q_len=1 under cp_size=2: both of rank
    1's chunks sit past the single real token).

    The partition then skips the scoring call and hands cp_shard_rows the
    -1-filled full layout; every local row is padding and gets zeroed, so
    the output matches the full-stream path byte for byte.
    """
    scenario = _Scenario([1], 5, 2)
    context = scenario.contexts[1]
    assert context.query_index.numel() == 0
    partitioned, calls, shard_inputs = _run_select_qli(scenario, 1, partition=True)
    assert not calls
    assert bool((shard_inputs[0] == -1).all())
    full, full_calls, _ = _run_select_qli(scenario, 1, partition=False)
    assert full_calls
    assert torch.equal(partitioned, full)
    assert torch.equal(partitioned, torch.zeros_like(partitioned))


# ---------------------------------------------------------------------------
# S3 batch: the quant path (int8 index cache + fp16 dequant scales).
# ---------------------------------------------------------------------------


def _cpu_dynamic_quant(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU mirror of the xllm_ops per-token int8 dynamic_quant.

    3-D ``[rows, heads, dim]`` inputs quantize per token and head (the
    indexer query layout); 2-D ``[rows, dim]`` inputs per token (the key
    layout). The scale is ``amax / 127`` so that ``int8 * scale``
    dequantizes back to the input; the exact rounding only has to be
    shared between the patched code path and the scenario precompute,
    which it is by construction (both call this function).
    """
    dims = values.shape[:-1]
    flat = values.reshape(-1, values.size(-1))
    amax = flat.abs().amax(dim=-1)
    scale = (amax / 127.0).clamp_min(1e-6)
    quantized = torch.round(flat / scale.view(-1, 1)).clamp_(-127, 127).to(torch.int8)
    return quantized.view(values.shape), scale.view(dims)


def _reference_quant_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    metadata: torch.Tensor,
    query_seq_lengths: torch.Tensor,
    key_seq_lengths: torch.Tensor,
    block_table: torch.Tensor,
    selected_count: int,
    cmp_ratio: int,
) -> torch.Tensor:
    """CPU reference for the S0-verified quant_lightning_indexer contract.

    Mirrors what the S0 quant probe fed the real kernel
    (research/s0_probe_quant.py): int8 q/key, fp16 per-token-per-head query
    scales, paged 3-D ``[blocks, block_size, 1]`` key scales (S0 contracts
    #9/#11) and fp16 weights. The dequantized score of query row ``t``
    against KV position ``p`` is
    ``sum_h w[t, h] * qs[t, h] * ks[p] * dot(int8 q[t, h], int8 k[p])``;
    the varlen / causal / top-k contract matches the non-quant reference
    (TND cumulative query ends, per-batch KV lengths, one block-table row
    per batch, sparse_mode=3 right-aligned causal, -1 pads).
    """
    assert query.dtype == torch.int8 and key.dtype == torch.int8
    assert weights.dtype == torch.float16
    assert query_dequant_scale.dtype == torch.float16
    assert key_dequant_scale.dtype == torch.float16 and key_dequant_scale.dim() == 3
    assert cmp_ratio == 1
    q_ends = [int(value) for value in query_seq_lengths.tolist()]
    kv_lens = [int(value) for value in key_seq_lengths.tolist()]
    block_size = key.shape[1]
    out = torch.full((query.shape[0], key.shape[2], selected_count), -1, dtype=torch.int32)
    start = 0
    for batch, q_end in enumerate(q_ends):
        q_len = q_end - start
        kv_len = kv_lens[batch]
        table = block_table[batch]
        for i in range(q_len):
            visible = kv_len - q_len + i + 1
            scored = []
            for position in range(visible):
                block = int(table[position // block_size])
                offset = position % block_size
                k_vec = key[block, offset, 0].float()
                k_scale = float(key_dequant_scale[block, offset, 0])
                score = 0.0
                for head in range(query.shape[1]):
                    q_scale = float(query_dequant_scale[start + i, head])
                    score += (
                        float(weights[start + i, head])
                        * q_scale
                        * k_scale
                        * float(torch.dot(query[start + i, head].float(), k_vec))
                    )
                scored.append((score, position))
            scored.sort(key=lambda item: (-item[0], item[1]))
            chosen = [position for _score, position in scored[:selected_count]]
            out[start + i, 0, : len(chosen)] = torch.tensor(chosen, dtype=torch.int32)
        start = q_end
    return out


class _QuantScenario(_Scenario):
    """Adds the int8 paged index cache and fp16 scales the quant path reads.

    Mirrors the key side of select_qli's quant chain on the global stream
    (the eye hadamard and ``HEAD_DIM**-0.5`` rotation ``_make_indexer``
    installs, then the same ``_cpu_dynamic_quant`` the patched code path
    runs) and pages the quantized current-chunk rows into the
    per-sequence blocks the materialize mock returns -- what
    ``ctx.update_index_cache`` would have written. Prefix positions carry
    older quantized payload (arbitrary int8 with positive scales), as
    after a real chunked-prefill step. The paged scale keeps the 3-D
    ``[blocks, block_size, 1]`` layout of S0 contract #9.
    """

    def __init__(self, q_lens: list[int], prefix: int, cp_size: int, seed: int = 42) -> None:
        super().__init__(q_lens, prefix, cp_size, seed)
        k_rotated = torch.matmul(self.global_k, torch.eye(HEAD_DIM)) * (HEAD_DIM**-0.5)
        k_quantized, k_scales = _cpu_dynamic_quant(k_rotated)
        k_scales = k_scales.unsqueeze(-1).to(torch.float16)
        n_blocks = self.index_cache.size(0)
        self.quant_index_cache = torch.randint(-127, 128, (n_blocks, BLOCK_SIZE, 1, HEAD_DIM), dtype=torch.int8)
        self.quant_index_cache_scale = (torch.rand(n_blocks, BLOCK_SIZE, 1) * 0.1 + 0.05).to(torch.float16)
        kv_lens = [length + prefix for length in q_lens]
        seq_offset = 0
        for seq, kv_len in enumerate(kv_lens):
            table = self.block_table[seq]
            for position in range(prefix, kv_len):
                block = int(table[position // BLOCK_SIZE])
                offset = position % BLOCK_SIZE
                self.quant_index_cache[block, offset, 0] = k_quantized[seq_offset + position - prefix]
                self.quant_index_cache_scale[block, offset, 0] = k_scales[seq_offset + position - prefix, 0]
            seq_offset += q_lens[seq]


class _QuantRunResult(NamedTuple):
    """Recorded evidence of one quant-path select_qli run."""

    output: torch.Tensor
    kernel_calls: list[tuple]
    metadata_calls: list[tuple]
    ctx_metadata: MagicMock
    metadata_sentinel: torch.Tensor
    shard_inputs: list[torch.Tensor]


def _run_quant_select_qli(
    scenario: _QuantScenario,
    rank: int,
    partition: bool,
    *,
    has_kv_shard: bool = False,
    non_cp: bool = False,
    cp_context_override: object = None,
    cp_metadata_getter: object = None,
) -> _QuantRunResult:
    """Run select_qli's quant branch for one CP rank; records every call.

    ``dynamic_quant`` runs the CPU mirror, ``quant_lightning_indexer`` the
    CPU reference, and the direct ``quant_lightning_indexer_metadata``
    call is recorded and answered with a sentinel so the kernel's metadata
    argument stays traceable; the ctx helper mock records any full-stream
    metadata reuse. The CP collectives / cp_shard_rows shims match
    ``_run_select_qli``, and the S4 keyword arguments carry the same
    gate-case meaning as there (``non_cp`` / ``cp_context_override`` /
    ``has_kv_shard``).
    """
    context = scenario.contexts[rank]
    indexer = _make_indexer()
    if non_cp:
        ctx_cp_context = None
        local_hidden = scenario.global_hidden
        n_positions = scenario.total
    else:
        ctx_cp_context = cp_context_override if cp_context_override is not None else context
        local_hidden = cp_utils.cp_shard_rows(scenario.global_hidden, context)
        n_positions = context.total_local
    kernel_calls: list[tuple] = []
    metadata_calls: list[tuple] = []
    shard_inputs: list[torch.Tensor] = []
    metadata_sentinel = torch.empty(0)
    ctx_metadata = MagicMock(return_value=torch.empty(0))

    def kernel(*args: object, **_kwargs: object) -> torch.Tensor:
        kernel_calls.append(args)
        assert isinstance(args, tuple)
        return _reference_quant_lightning_indexer(*args)

    def metadata(*args: object, **_kwargs: object) -> torch.Tensor:
        metadata_calls.append(args)
        return metadata_sentinel

    def all_gather(value: torch.Tensor, dim: int, size: int, group: str) -> torch.Tensor:
        assert (dim, size, group) == (0, scenario.cp_size, "cp")
        if value.dim() == 3:
            peers = scenario.local_q
        elif value.shape[-1] == HEAD_DIM:
            peers = scenario.local_k
        else:
            peers = scenario.local_weights
        torch.testing.assert_close(value, peers[rank])
        return torch.cat(peers, dim=0)

    def shard_rows(value: torch.Tensor, cp_context: CpContext) -> torch.Tensor:
        shard_inputs.append(value)
        return cp_utils.cp_shard_rows(value, cp_context)

    ctx = MlaIndexContext(
        index_cache=scenario.quant_index_cache,
        slot_mapping=torch.arange(scenario.total, dtype=torch.int32),
        block_table=scenario.block_table,
        actual_seq_q=scenario.actual_seq_q,
        actual_seq_kv=scenario.actual_seq_kv,
        index_cache_scale=scenario.quant_index_cache_scale,
        get_quant_indexer_metadata=ctx_metadata,
        update_index_cache=MagicMock(),
        materialize_index_cache=MagicMock(
            return_value=(scenario.quant_index_cache, scenario.quant_index_cache_scale, scenario.block_table)
        ),
        cp_context=ctx_cp_context,
        has_kv_shard=has_kv_shard,
        get_cp_quant_indexer_metadata=cp_metadata_getter,
    )

    with (
        patch.object(glm5_2, "_INDEXER_CP_QUERY_PARTITION", partition),
        patch.object(glm5_2, "_apply_half_rope", side_effect=lambda _cache, value, _positions: value),
        patch.object(cp_utils.distributed, "all_gather", side_effect=all_gather, create=True),
        patch.object(glm5_2, "cp_shard_rows", side_effect=shard_rows),
        patch.object(glm5_2.kernels, "dynamic_quant", side_effect=_cpu_dynamic_quant, create=True),
        patch.object(glm5_2.kernels, "quant_lightning_indexer", side_effect=kernel, create=True),
        patch.object(glm5_2.kernels, "quant_lightning_indexer_metadata", side_effect=metadata, create=True),
    ):
        output = indexer.select_qli(
            local_hidden,
            local_hidden,
            torch.zeros(n_positions, dtype=torch.int64),
            ctx,
            torch.empty(0),
        )
    return _QuantRunResult(
        output=output,
        kernel_calls=kernel_calls,
        metadata_calls=metadata_calls,
        ctx_metadata=ctx_metadata,
        metadata_sentinel=metadata_sentinel,
        shard_inputs=shard_inputs,
    )


@pytest.mark.parametrize(
    "q_lens,cp_size,prefix",
    [
        ([13], 2, 6),
        ([7], 4, 3),
        ([9, 6], 2, 4),
    ],
)
def test_quant_partition_matches_full_stream_for_owned_rows(
    q_lens: list[int],
    cp_size: int,
    prefix: int,
) -> None:
    """The quant partitioned top-k equals the full-stream one for owned rows.

    The sliced q/q_scale/weights must be exactly the owned rows of the
    full-stream kernel arguments (the row slice preserves the padded
    per-token-per-head q_scale layout, S0 contract #10), while the top-k
    sets are compared sorted (S0 contract #6).
    """
    scenario = _QuantScenario(q_lens, prefix, cp_size)
    for rank in range(cp_size):
        context = scenario.contexts[rank]
        partitioned = _run_quant_select_qli(scenario, rank, partition=True)
        full = _run_quant_select_qli(scenario, rank, partition=False)
        if context.query_index.numel() == 0:
            # An empty rank owns no real row: the partition skips the
            # metadata and scoring calls while the full stream runs both.
            assert not partitioned.kernel_calls and not partitioned.metadata_calls
            assert full.kernel_calls
            assert torch.equal(partitioned.output, full.output)
            continue
        assert len(partitioned.kernel_calls) == 1 and len(full.kernel_calls) == 1
        rows = cp_utils.cp_real_query_rows(context)
        partition_args = partitioned.kernel_calls[0]
        full_args = full.kernel_calls[0]
        # Only this rank's real rows are scored, byte-identical to slicing
        # the full-stream q / weights / per-token-per-head q_scale.
        assert partition_args[0].shape[0] == context.query_index.numel()
        assert partition_args[3].shape[0] == context.query_index.numel()
        assert full_args[0].shape[0] == scenario.total
        assert torch.equal(partition_args[0], full_args[0].index_select(0, rows))
        assert torch.equal(partition_args[2], full_args[2].index_select(0, rows))
        assert torch.equal(partition_args[3], full_args[3].index_select(0, rows))
        assert torch.equal(
            partitioned.output[context.query_index].sort(dim=-1).values,
            full.output[context.query_index].sort(dim=-1).values,
        )
        # Padding rows are zeroed by cp_shard_rows on both paths.
        padding = ~context.shard_valid_mask
        assert torch.equal(partitioned.output[padding], full.output[padding])
        assert torch.equal(partitioned.output[padding], torch.zeros_like(partitioned.output[padding]))


def test_quant_partition_recomputes_metadata_from_segment_seq_tensors() -> None:
    """S0 contract #7: the partitioned call must not reuse full-stream metadata.

    The tiling metadata always describes the SLICED call's per-segment seq
    tensors, never the backend's full-stream ones, so the full-stream ctx
    helper stays untouched. This case covers the fallback route (a context
    with no ``get_cp_quant_indexer_metadata``): the metadata is built by the
    direct kernels call with the per-segment maxima, and the scoring kernel
    receives it together with the per-segment seq tensors and block-table
    rows. The primary route -- the backend's per-step CP cache -- is covered
    by the two ``*_cp_getter``/``*_cp_metadata_build`` tests below.
    """
    scenario = _QuantScenario([13], 6, 2)
    rank = 1
    context = scenario.contexts[rank]
    result = _run_quant_select_qli(scenario, rank, partition=True)
    # The full-stream ctx helper stays untouched on the partitioned path.
    assert result.ctx_metadata.call_count == 0
    assert len(result.metadata_calls) == 1
    args = result.metadata_calls[0]
    q_cu_seqlens = context.q_cu_seqlens
    segment_q_lens = [q_cu_seqlens[0]] + [end - begin for begin, end in zip(q_cu_seqlens, q_cu_seqlens[1:])]
    assert args[0] == glm5_2._QLI_KERNEL_GSIZE
    assert args[1] == 1
    assert args[2] == HEAD_DIM
    assert args[3] is context.q_cu_seqlens_tensor
    assert args[4] is context.segment_kv_seq_lens_tensor
    assert args[5] == max(segment_q_lens)
    assert args[6] == max(context.segment_kv_seq_lens)
    assert args[7] == TOPK
    assert args[8] == 1
    kernel_args = result.kernel_calls[0]
    assert kernel_args[5] is result.metadata_sentinel
    assert kernel_args[6] is context.q_cu_seqlens_tensor
    assert kernel_args[7] is context.segment_kv_seq_lens_tensor
    assert torch.equal(kernel_args[8], scenario.block_table.index_select(0, context.segment_seq_indices))


def test_quant_partition_routes_metadata_through_the_cp_getter() -> None:
    """The default-ON quant partition must not rebuild the tiling per layer.

    Regression for the review finding "CP 量化分支每层重算
    quant_lightning_indexer_metadata": ``MlaIndexContext`` is rebuilt per
    layer, so a direct kernels call inside ``_cp_partitioned_quant_topk``
    pays the host-side tiling (and its 1024-int32 allocation) once per
    indexer layer per step. When the backend exposes its per-step CP cache,
    select_qli must route through it -- never through the direct kernel call
    -- and must pass the per-segment maxima plus the head geometry the cache
    keys on.
    """
    scenario = _QuantScenario([13], 6, 2)
    rank = 1
    context = scenario.contexts[rank]
    seen: list[tuple] = []

    def cp_getter(*args: object) -> torch.Tensor:
        seen.append(args)
        return torch.empty(0)

    result = _run_quant_select_qli(scenario, rank, partition=True, cp_metadata_getter=cp_getter)
    # The direct kernel tiling build is skipped entirely on this route.
    assert result.metadata_calls == []
    assert result.ctx_metadata.call_count == 0
    assert len(seen) == 1
    q_cu_seqlens = context.q_cu_seqlens
    segment_q_lens = [q_cu_seqlens[0]] + [end - begin for begin, end in zip(q_cu_seqlens, q_cu_seqlens[1:])]
    args = seen[0]
    assert args[0] == glm5_2._QLI_KERNEL_GSIZE  # padded query heads
    assert args[1] == 1  # num_heads_k of the paged MLA index cache
    assert args[2] == HEAD_DIM
    assert args[3] == max(segment_q_lens)  # per-segment query maximum
    assert args[4] == max(context.segment_kv_seq_lens)  # per-segment KV maximum
    assert args[5] == TOPK
    assert args[6] == 1  # cmp_ratio
    # The getter's tensor is what reaches the scoring kernel.
    assert result.kernel_calls[0][5] is not None


def test_quant_partition_shares_one_cp_metadata_build_across_layers() -> None:
    """Two indexer layers in one step build the CP metadata exactly once.

    Emulates the backend's per-step contract with a dict-backed getter: the
    same (heads, head_dim, maxima, topk, cmp_ratio) geometry must hit the
    cache on the second layer, which is only possible because the alpha path
    now routes through the context getter instead of calling the kernel
    directly once per layer.
    """
    scenario = _QuantScenario([13], 6, 2)
    rank = 1
    builds: list[tuple] = []
    cache: dict[tuple, torch.Tensor] = {}

    def cp_getter(*args: object) -> torch.Tensor:
        key = args
        cached = cache.get(key)
        if cached is None:
            builds.append(key)
            cached = torch.empty(0)
            cache[key] = cached
        return cached

    first = _run_quant_select_qli(scenario, rank, partition=True, cp_metadata_getter=cp_getter)
    second = _run_quant_select_qli(scenario, rank, partition=True, cp_metadata_getter=cp_getter)
    assert len(builds) == 1, f"expected one tiling build per step, got {len(builds)}"
    assert first.metadata_calls == [] and second.metadata_calls == []
    # Both layers' scoring kernels consume the same cached metadata tensor.
    assert torch.equal(first.output, second.output)


def test_quant_partition_empty_rank_skips_scoring_and_matches_full_stream() -> None:
    """A rank owning zero real rows skips the quant metadata and kernel.

    The partition hands cp_shard_rows the -1-filled full layout; every
    local row is padding and gets zeroed, so the output matches the
    full-stream path byte for byte.
    """
    scenario = _QuantScenario([1], 5, 2)
    context = scenario.contexts[1]
    assert context.query_index.numel() == 0
    partitioned = _run_quant_select_qli(scenario, 1, partition=True)
    assert not partitioned.kernel_calls
    assert not partitioned.metadata_calls
    assert partitioned.ctx_metadata.call_count == 0
    assert bool((partitioned.shard_inputs[0] == -1).all())
    full = _run_quant_select_qli(scenario, 1, partition=False)
    assert full.kernel_calls
    assert torch.equal(partitioned.output, full.output)
    assert torch.equal(partitioned.output, torch.zeros_like(partitioned.output))


def test_quant_gate_off_runs_the_byte_identical_full_stream_path() -> None:
    """The seam off must reproduce the legacy (full-stream) quant output."""
    scenario = _QuantScenario([13], 6, 2)
    rank = 1
    context = scenario.contexts[rank]
    result = _run_quant_select_qli(scenario, rank, partition=False)
    # The kernel saw the full-stream arguments verbatim: the complete
    # merged stream (padded to 64 heads), the ctx helper's metadata and
    # the backend's per-sequence seq tensors and table.
    args = result.kernel_calls[0]
    assert args[0].shape[0] == scenario.total
    assert args[3].shape[0] == scenario.total
    assert args[5] is result.ctx_metadata.return_value
    assert args[6] is scenario.actual_seq_q
    assert args[7] is scenario.actual_seq_kv
    assert args[8] is scenario.block_table
    result.ctx_metadata.assert_called_once_with(glm5_2._QLI_KERNEL_GSIZE, HEAD_DIM, TOPK, 1)
    assert not result.metadata_calls
    # Byte-identical reconstruction of the pre-partition path from the
    # scenario data: quantize the rotated global stream, pad to 64 heads,
    # score everything, then shard.
    q_rotated = torch.matmul(scenario.global_q, torch.eye(HEAD_DIM)) * (HEAD_DIM**-0.5)
    q_quantized, q_scales = _cpu_dynamic_quant(q_rotated)
    pad_heads = glm5_2._QLI_KERNEL_GSIZE - N_HEAD
    q_full = torch.cat(
        [q_quantized, torch.zeros(scenario.total, pad_heads, HEAD_DIM, dtype=torch.int8)],
        dim=1,
    )
    q_scale_full = torch.cat(
        [q_scales.to(torch.float16), torch.ones(scenario.total, pad_heads, dtype=torch.float16)],
        dim=1,
    )
    weight_scale = HEAD_DIM**-0.5 * N_HEAD**-0.5
    weights_full = torch.cat([scenario.global_weights, torch.zeros(scenario.total, pad_heads)], dim=1)
    legacy = cp_utils.cp_shard_rows(
        _reference_quant_lightning_indexer(
            q_full,
            scenario.quant_index_cache,
            (weights_full * weight_scale).to(torch.float16),
            q_scale_full,
            scenario.quant_index_cache_scale,
            result.ctx_metadata.return_value,
            scenario.actual_seq_q,
            scenario.actual_seq_kv,
            scenario.block_table,
            TOPK,
            1,
        ),
        context,
    )
    assert torch.equal(result.output, legacy)


# ---------------------------------------------------------------------------
# S4 batch: gate completeness. Every fallback branch of alpha_active must
# run the status-quo full-stream path byte-identically, plus the kv_split
# replicated positive case (the S3-carried item). Each fallback case pins
# _INDEXER_CP_QUERY_PARTITION on, so the term under test is the only one
# that can fail, captures the scoring kernel's call to prove it saw the
# full-stream arguments, and compares the output with the pre-partition
# reference; the positive case proves the partitioned call stays active.
# ---------------------------------------------------------------------------


class _GateCaseResult(NamedTuple):
    """S4-relevant evidence of one gate-case run on either scoring branch."""

    output: torch.Tensor
    kernel_calls: list[tuple]
    shard_inputs: list[torch.Tensor]


def _run_gate_case(
    scenario: _Scenario,
    rank: int,
    partition: bool,
    *,
    quant: bool,
    has_kv_shard: bool = False,
    non_cp: bool = False,
    cp_context_override: object = None,
) -> _GateCaseResult:
    """Run one gate case on the requested scoring branch via the shared runners."""
    if quant:
        result = _run_quant_select_qli(
            scenario,
            rank,
            partition,
            has_kv_shard=has_kv_shard,
            non_cp=non_cp,
            cp_context_override=cp_context_override,
        )
        return _GateCaseResult(result.output, result.kernel_calls, result.shard_inputs)
    output, kernel_calls, shard_inputs = _run_select_qli(
        scenario,
        rank,
        partition,
        has_kv_shard=has_kv_shard,
        non_cp=non_cp,
        cp_context_override=cp_context_override,
    )
    return _GateCaseResult(output, kernel_calls, shard_inputs)


def _assert_full_stream_kernel_call(scenario: _Scenario, result: _GateCaseResult, *, quant: bool) -> None:
    """The scoring kernel must have run on the status-quo full-stream arguments.

    The complete merged query stream (head-padded on the quant branch),
    the backend's per-sequence seq tensors and the full block table --
    passed by identity, the way the pre-partition path passes them.
    """
    assert len(result.kernel_calls) == 1
    args = result.kernel_calls[0]
    assert args[0].shape[0] == scenario.total
    if quant:
        assert args[3].shape[0] == scenario.total
        assert args[6] is scenario.actual_seq_q
        assert args[7] is scenario.actual_seq_kv
        assert args[8] is scenario.block_table
    else:
        assert args[3] is scenario.actual_seq_q
        assert args[4] is scenario.actual_seq_kv
        assert args[5] is scenario.block_table


def _legacy_full_stream_topk(scenario: _Scenario, *, quant: bool) -> torch.Tensor:
    """Pre-partition reference output: score the full merged stream, no partition."""
    if not quant:
        return _reference_lightning_indexer(
            scenario.global_q,
            scenario.index_cache,
            scenario.global_weights,
            scenario.actual_seq_q,
            scenario.actual_seq_kv,
            scenario.block_table,
            "TND",
            "PA_BSND",
            TOPK,
            3,
            INT64_MAX,
            INT64_MAX,
            False,
        )
    q_rotated = torch.matmul(scenario.global_q, torch.eye(HEAD_DIM)) * (HEAD_DIM**-0.5)
    q_quantized, q_scales = _cpu_dynamic_quant(q_rotated)
    pad_heads = glm5_2._QLI_KERNEL_GSIZE - N_HEAD
    q_full = torch.cat(
        [q_quantized, torch.zeros(scenario.total, pad_heads, HEAD_DIM, dtype=torch.int8)],
        dim=1,
    )
    q_scale_full = torch.cat(
        [q_scales.to(torch.float16), torch.ones(scenario.total, pad_heads, dtype=torch.float16)],
        dim=1,
    )
    weight_scale = HEAD_DIM**-0.5 * N_HEAD**-0.5
    weights_full = torch.cat([scenario.global_weights, torch.zeros(scenario.total, pad_heads)], dim=1)
    return _reference_quant_lightning_indexer(
        q_full,
        scenario.quant_index_cache,
        (weights_full * weight_scale).to(torch.float16),
        q_scale_full,
        scenario.quant_index_cache_scale,
        torch.empty(0),
        scenario.actual_seq_q,
        scenario.actual_seq_kv,
        scenario.block_table,
        TOPK,
        1,
    )


def _assert_gate_fallback_full_stream(
    scenario: _Scenario,
    context: CpContext | None,
    result: _GateCaseResult,
    *,
    quant: bool,
) -> None:
    """A gate fallback must reproduce the status-quo path byte for byte.

    ``context`` None marks the non-CP case, whose status-quo path has no
    cp_shard_rows exit at all.
    """
    _assert_full_stream_kernel_call(scenario, result, quant=quant)
    expected = _legacy_full_stream_topk(scenario, quant=quant)
    if context is None:
        assert not result.shard_inputs
    else:
        assert len(result.shard_inputs) == 1
        expected = cp_utils.cp_shard_rows(expected, context)
    assert torch.equal(result.output, expected)


@pytest.mark.parametrize("quant", [False, True], ids=["non-quant", "quant"])
def test_gate_fallback_without_cp_context_runs_full_stream(quant: bool) -> None:
    """Gate term ``ctx.cp_context is not None``.

    A non-CP forward feeds the whole batch to select_qli with no CP gather
    and no shard exit; with the env flag on, the gate must keep that
    forward on the status-quo full-stream path. The scenario supplies only
    the global-stream data here; its per-rank contexts are unused.
    """
    scenario = _QuantScenario([13], 6, 2) if quant else _Scenario([13], 6, 2)
    result = _run_gate_case(scenario, 0, partition=True, quant=quant, non_cp=True)
    _assert_gate_fallback_full_stream(scenario, None, result, quant=quant)


@pytest.mark.parametrize("quant", [False, True], ids=["non-quant", "quant"])
def test_gate_fallback_cp_size_one_runs_full_stream(quant: bool) -> None:
    """Gate term ``ctx.cp_context.cp_size > 1``.

    A cp_size-1 CpContext still flows through the CP collectives (the
    single rank owns both zigzag chunks of every sequence), but the gate
    must treat it as non-partitionable and run the status-quo path.
    """
    scenario = _QuantScenario([13], 6, 1) if quant else _Scenario([13], 6, 1)
    context = scenario.contexts[0]
    result = _run_gate_case(scenario, 0, partition=True, quant=quant)
    _assert_gate_fallback_full_stream(scenario, context, result, quant=quant)


@pytest.mark.parametrize("quant", [False, True], ids=["non-quant", "quant"])
def test_gate_fallback_duck_typed_cp_context_runs_full_stream(quant: bool) -> None:
    """Gate term ``isinstance(ctx.cp_context, CpContext)``.

    A SimpleNamespace stand-in (the shape test_glm5_2_cp_ragged.py's
    contexts take) carrying every field the full-stream path reads must
    stay safe on the full stream: the per-segment metadata the partition
    consumes is part of the CpContext dataclass, not of a duck type.
    """
    scenario = _QuantScenario([13], 6, 2) if quant else _Scenario([13], 6, 2)
    rank = 1
    context = scenario.contexts[rank]
    stand_in = SimpleNamespace(
        cp_size=context.cp_size,
        cp_rank=context.cp_rank,
        total_local=context.total_local,
        shard_index=context.shard_index,
        shard_valid_mask=context.shard_valid_mask,
        shard_gather_index=context.shard_gather_index,
        restore_index=context.restore_index,
        query_index=context.query_index,
        has_prefix=context.has_prefix,
    )
    result = _run_gate_case(scenario, rank, partition=True, quant=quant, cp_context_override=stand_in)
    _assert_gate_fallback_full_stream(scenario, context, result, quant=quant)


@pytest.mark.parametrize("quant", [False, True], ids=["non-quant", "quant"])
def test_gate_fallback_acl_graph_capture_runs_full_stream(quant: bool) -> None:
    """Gate term ``not in_acl_graph()``.

    Graph capture would bake the host-side per-segment lists into the
    captured graph, so a capture-time forward must run the status-quo
    path. in_acl_graph was imported into glm5_2's namespace, so patching
    glm5_2.in_acl_graph is the seam that fakes capture.
    """
    scenario = _QuantScenario([13], 6, 2) if quant else _Scenario([13], 6, 2)
    rank = 1
    context = scenario.contexts[rank]
    with patch.object(glm5_2, "in_acl_graph", return_value=True):
        result = _run_gate_case(scenario, rank, partition=True, quant=quant)
    _assert_gate_fallback_full_stream(scenario, context, result, quant=quant)


@pytest.mark.parametrize("quant", [False, True], ids=["non-quant", "quant"])
def test_gate_fallback_kv_split_sharded_runs_full_stream(quant: bool) -> None:
    """Gate term ``not ctx.has_kv_shard or cp_index_write_replicated()``.

    A shard-aware index cache in the non-default sharded write mode
    reconstructs the readable cache across ranks, which is unverified for
    the per-segment call, so the gate must fall back. The env-derived
    helper defaults to True, so the sharded case patches the
    glm5_2.cp_index_write_replicated seam to False explicitly (keeping
    the file env-independent).
    """
    scenario = _QuantScenario([13], 6, 2) if quant else _Scenario([13], 6, 2)
    rank = 1
    context = scenario.contexts[rank]
    with patch.object(glm5_2, "cp_index_write_replicated", return_value=False):
        result = _run_gate_case(scenario, rank, partition=True, quant=quant, has_kv_shard=True)
    _assert_gate_fallback_full_stream(scenario, context, result, quant=quant)


@pytest.mark.parametrize("quant", [False, True], ids=["non-quant", "quant"])
def test_gate_stays_active_under_kv_split_replicated(quant: bool) -> None:
    """kv_split positive case (S3-carried): the DEFAULT replicated index-write
    mode keeps a full per-rank replica, so the partition stays active.

    The kernel still receives this rank's sliced rows and the per-segment
    seq tensors / block-table rows -- never the full-stream arguments --
    and the output still matches the full-stream reference for the owned
    rows (set level, S0 contract #6) with the padding rows zeroed.
    """
    scenario = _QuantScenario([13], 6, 2) if quant else _Scenario([13], 6, 2)
    # Rank 0 owns chunk 0 (rows 0-3) and chunk 3 (row 12 plus three padding
    # rows), so both the owned-row and the padding comparisons below are
    # non-empty; rank 1 would own real rows only.
    rank = 0
    context = scenario.contexts[rank]
    assert context.query_index.numel() > 0
    with patch.object(glm5_2, "cp_index_write_replicated", return_value=True):
        result = _run_gate_case(scenario, rank, partition=True, quant=quant, has_kv_shard=True)
    assert len(result.kernel_calls) == 1
    args = result.kernel_calls[0]
    assert args[0].shape[0] == context.query_index.numel()
    if quant:
        assert args[3].shape[0] == context.query_index.numel()
        assert args[6] is context.q_cu_seqlens_tensor
        assert args[7] is context.segment_kv_seq_lens_tensor
        segment_table = args[8]
    else:
        assert args[3] is context.q_cu_seqlens_tensor
        assert args[4] is context.segment_kv_seq_lens_tensor
        segment_table = args[5]
    assert segment_table.shape[0] == len(context.q_cu_seqlens)
    assert torch.equal(segment_table, scenario.block_table.index_select(0, context.segment_seq_indices))
    legacy = cp_utils.cp_shard_rows(_legacy_full_stream_topk(scenario, quant=quant), context)
    assert torch.equal(
        result.output[context.query_index].sort(dim=-1).values,
        legacy[context.query_index].sort(dim=-1).values,
    )
    padding = ~context.shard_valid_mask
    assert bool(padding.any())
    assert torch.equal(result.output[padding], legacy[padding])


@pytest.mark.parametrize("quant", [False, True], ids=["non-quant", "quant"])
def test_gate_seam_off_runs_byte_identical_full_stream(quant: bool) -> None:
    """Gate term ``_INDEXER_CP_QUERY_PARTITION`` (the test-only seam).

    With every other gate term eligible -- CP active with cp_size > 1, a
    real CpContext, eager execution and a full-replica index cache
    (has_kv_shard False) -- turning the seam off must reproduce the
    status-quo path byte for byte. The shipped build pins the seam True and
    exposes no environment override; this case is the A/B control that keeps
    the full-stream path covered.
    """
    scenario = _QuantScenario([13], 6, 2) if quant else _Scenario([13], 6, 2)
    rank = 1
    context = scenario.contexts[rank]
    assert context.cp_size > 1
    assert isinstance(context, CpContext)
    assert not glm5_2.in_acl_graph()
    result = _run_gate_case(scenario, rank, partition=False, quant=quant)
    _assert_gate_fallback_full_stream(scenario, context, result, quant=quant)
