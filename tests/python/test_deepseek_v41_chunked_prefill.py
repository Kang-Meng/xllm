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

"""Chunk-boundary regressions, including SWA and ratio-2 ring32 state wrap.

CPU components use the repository kernel harness. Full synthetic-model
execution requires NPU; neither substitutes for full-checkpoint serving tests.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest
import torch

import xllm.python.attention.csa2_attention as csa2
from tests.python.test_deepseek_v41_model import (
    _DEEPSEEK_V41_FORWARD_CONFIG,
    _active_device,
    _allocate_v41_layer_caches,
    _make_csa2_backend,
    _make_forward_lm,
    _V41ForwardHarness,
    npu_only,
)
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.csa2_attention import (
    Csa2AttentionBackend,
    Csa2LayerContext,
    CSA2SharedRuntime,
    _quantize_indexer_rows_int8,
    _window_visible_idxs,
    apply_interleaved_rope,
)
from xllm.python.attention.csa_attention import _scatter_by_slot
from xllm.python.attention.dsa_metadata import last_ring_writes
from xllm.python.model_executor.dsa_cp import DsaCpRuntime, build_dsa_cp_context
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.models.deepseek_v41 import (
    CSA2_MODE_FULL,
    CSA2_MODE_SWA,
    CSA2LayerMode,
    DeepseekV41Config,
    DeepseekV41ForCausalLM,
    DeepseekV41HyperConnection,
)


def _make_swa_backend(window: int, width: int, heads: int) -> Csa2AttentionBackend:
    return Csa2AttentionBackend(
        compress_ratios=[0],
        window_size=window,
        n_layers=1,
        num_heads=heads,
        attn_head_dim=width,
        index_topk=8,
        index_n_heads=2,
        index_head_dim=width,
        rope_head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        kv_source_layer_ids=[],
        index_source_layer_ids=[],
        candidate_source_layer_id=-1,
        candidate_topk_blocks=8,
        candidate_block_size=2,
    )


def _reference_swa(q: torch.Tensor, kv: torch.Tensor, prefix: int, window: int, sink: torch.Tensor) -> torch.Tensor:
    scores = torch.einsum("shd,nd->shn", q, kv) * kv.size(-1) ** -0.5
    positions = prefix + torch.arange(q.size(0))
    columns = torch.arange(kv.size(0))
    visible = (columns <= positions[:, None]) & (columns >= (positions[:, None] - window + 1).clamp_min(0))
    scores.masked_fill_(~visible[:, None, :], float("-inf"))
    maximum = scores.amax(-1, keepdim=True)
    probs = torch.exp(scores - maximum)
    return torch.einsum("shn,nd->shd", probs, kv) / (
        probs.sum(-1, keepdim=True) + torch.exp(sink[None, :, None] - maximum)
    )


@pytest.mark.parametrize(
    "prefixes,chunks",
    [
        ([0], [1]),
        ([0], [33]),
        ([7], [2]),
        ([8], [8]),
        ([9], [31]),
        ([23], [33]),
        ([127], [129]),
        ([0, 8], [5, 17]),
        ([7, 31], [2, 33]),
        ([15, 1, 32], [1, 35, 3]),
    ],
)
@pytest.mark.parametrize("ring_blocks", [1, 2])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("unwritten_value", [0.0, float("nan")], ids=["zero-cache", "uninitialized-cache"])
def test_swa_chunk_reads_history_before_ring_wrap(
    prefixes: list[int],
    chunks: list[int],
    ring_blocks: int,
    cp_size: int,
    unwritten_value: float,
) -> None:
    """Independent causal reference plus complete ring checks and 3 decodes."""
    window, width, heads = 8, 4, 4
    batch = len(prefixes)
    capacity = window * ring_blocks
    torch.manual_seed(817)
    values = [torch.randn(p + c + 3, width) for p, c in zip(prefixes, chunks)]
    queries = [torch.randn(c + 3, heads, width) for c in chunks]
    sink = torch.randn(heads)
    references = []
    for rank in range(cp_size):
        backend = _make_swa_backend(window, width, heads)
        cache = torch.full((1 + batch * ring_blocks, window, width), unwritten_value)
        raw_table = torch.arange(1, 1 + batch * ring_blocks, dtype=torch.int32).view(batch, ring_blocks)
        backend.bind_kv_caches([LayerCache(key=None, value=None, swa=cache)])
        for seq, prefix in enumerate(prefixes):
            pos = torch.arange(max(0, prefix - capacity), prefix)
            slots = raw_table[seq, pos // window % ring_blocks] * window + pos % window
            _scatter_by_slot(cache, slots, values[seq][pos])
        layer = SimpleNamespace(
            mode=CSA2LayerMode(layer_id=0, mode=CSA2_MODE_SWA, compress_ratio=0, main_kv_source_layer=-1),
            layer_id=0,
            attn_sink=sink,
        )
        contexts = list(prefixes)
        for step in range(4):
            lengths = chunks if step == 0 else [1] * batch
            q_parts = [
                queries[s][: chunks[s]] if step == 0 else queries[s][chunks[s] + step - 1 : chunks[s] + step]
                for s in range(batch)
            ]
            q = torch.cat(q_parts)
            kv = torch.cat([values[s][contexts[s] : contexts[s] + lengths[s]] for s in range(batch)])
            positions = torch.cat([torch.arange(contexts[s], contexts[s] + lengths[s]) for s in range(batch)])
            next_contexts = [contexts[s] + lengths[s] for s in range(batch)]
            layer._csa2_ctx = Csa2LayerContext(kv, kv, positions, torch.empty(0))
            dsa = backend._builder.build(
                [raw_table],
                next_contexts,
                lengths,
                positions,
                None,
                is_prefill=step == 0 and not any(prefixes),
                is_chunked_prefill=step == 0,
            )
            metadata = SimpleNamespace(dsa_metadata=dsa)
            backend.prepare(metadata)
            plan = build_dsa_cp_context(sum(lengths), lengths, cp_size, rank) if cp_size > 1 else None
            context = ForwardContext(backend, torch.device("cpu"), metadata, [], cp_context=plan)
            if plan is not None:
                context.layer_shared_cache[DsaCpRuntime] = DsaCpRuntime(plan, plan.local_context_lengths(dsa.seq_lens))
            with torch.inference_mode(), forward_context(context):
                actual = backend.execute_csa2_layer(plan.shard(q) if plan is not None else q, kv, layer)
            expected = torch.cat(
                [
                    _reference_swa(q_parts[s], values[s][: next_contexts[s]], contexts[s], window, sink)
                    for s in range(batch)
                ]
            )
            if plan is not None:
                expected = plan.shard(expected)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            for seq, end in enumerate(next_contexts):
                pos = torch.arange(max(0, end - capacity), end)
                slots = raw_table[seq, pos // window % ring_blocks] * window + pos % window
                torch.testing.assert_close(cache.view(-1, width)[slots.long()], values[seq][pos], atol=0, rtol=0)
            contexts = next_contexts
        references.append(cache)
    for cache in references[1:]:
        torch.testing.assert_close(cache, references[0], atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("device_name", ["cpu", "npu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sparse_attention_query_tiles_bound_kv_memory(
    device_name: str, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bound selected-KV tiles without changing picks, masking or reductions.

    NPU uses the review's prefill geometry: 256 queries, 8 local heads
    (64 heads / TP8), window128, topk512, width512. This is a real attention
    component, not a full checkpoint or collective test. CPU uses a small
    shape and a smaller budget to exercise several tiles including a tail.
    """
    if device_name == "npu" and _active_device().type == "cpu":
        pytest.skip("NPU is unavailable")
    device = torch.device("cpu") if device_name == "cpu" else _active_device()
    window, width, heads, topk, length = (8, 8, 2, 32, 37) if device_name == "cpu" else (128, 512, 8, 512, 256)
    backend = Csa2AttentionBackend(
        compress_ratios=[2],
        window_size=window,
        n_layers=1,
        num_heads=heads,
        attn_head_dim=width,
        index_topk=topk,
        index_n_heads=1,
        index_head_dim=8,
        rope_head_dim=2,
        device=device,
        dtype=dtype,
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0],
        candidate_source_layer_id=-1,
        candidate_topk_blocks=0,
        candidate_block_size=0,
    )
    torch.manual_seed(817)
    cache = torch.randn(2, 128, width, device=device).to(dtype)
    cache[0].fill_(float("nan"))
    swa = torch.full((2, window, width), float("nan"), device=device, dtype=dtype)
    backend.bind_kv_caches([LayerCache(key=cache, value=None, swa=swa)])
    q = torch.randn(length, heads, width, device=device).to(dtype)
    kv = torch.randn(length, width, device=device).to(dtype)
    positions = torch.arange(length, device=device)
    tables = [torch.tensor([[1]], dtype=torch.int32, device=device)] * 2
    dsa = backend._builder.build(tables, [length], [length], positions, None, True, True)
    backend._move_metadata_to_device(dsa)
    layer = SimpleNamespace(
        layer_id=0, mode=CSA2LayerMode(0, CSA2_MODE_FULL, 2, 0), attn_sink=torch.randn(heads, device=device)
    )
    shared = CSA2SharedRuntime(key_cache=cache, key_block_table=tables[1], compress_ratio=2)
    columns = torch.arange(min(topk, length // 2), device=device)
    picks = (
        torch.where(columns[None, :] < (positions[:, None] + 1) // 2, window + length + columns[None, :], -1)
        .unsqueeze(0)
        .to(torch.int32)
    )
    # Also verify that an out-of-range lane is inert, not read out of bounds.
    picks[:, 0, -1] = window + length + 10000
    bytes_per_query = 2 * (window + topk) * width * 4
    budget = bytes_per_query * 3 if device_name == "cpu" else csa2._ATTENTION_KV_TILE_BYTES
    tile_rows = []
    original_einsum = torch.einsum

    def tracked_einsum(equation: str, *operands: torch.Tensor) -> torch.Tensor:
        if equation == "shd,snd->shn":
            tile_rows.append(operands[1].size(0))
            assert operands[1].shape[1:] == (window + topk, width)
        return original_einsum(equation, *operands)

    with torch.inference_mode():
        # The old 256-row layout is the numerical oracle, not a different
        # shared-KV reduction; row support/order must remain unchanged.
        monkeypatch.setattr(csa2, "_ATTENTION_KV_TILE_BYTES", bytes_per_query * length)
        expected = backend._csa2_attention(q, kv, layer, dsa, [length], dsa.seq_lens, picks, shared)
        monkeypatch.setattr(csa2, "_ATTENTION_KV_TILE_BYTES", budget)
        monkeypatch.setattr(torch, "einsum", tracked_einsum)
        if device_name == "npu":
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()
            initial = torch.npu.memory_allocated()
        actual = backend._csa2_attention(q, kv, layer, dsa, [length], dsa.seq_lens, picks, shared)
        if device_name == "npu":
            torch.npu.synchronize()
            # Includes gather/cast/where, scores, output and paged-KV copies;
            # not just the 32MiB selected-KV pair budget.
            assert torch.npu.max_memory_allocated() - initial < 64 * 1024 * 1024
    assert sum(tile_rows) == length and len(tile_rows) > 1
    assert max(tile_rows) * bytes_per_query <= budget
    assert torch.isfinite(actual).all()
    # Changing query batch geometry can change the NPU FP32 GEMM schedule;
    # retain the established model tolerances, not an unsupported bitwise claim.
    _assert_model_logits_close(actual.float(), expected.float(), dtype)
    if dtype == torch.bfloat16:
        assert (actual.float() - expected.float()).abs().max() <= 2**-10


_COMPRESSOR_STATE_ROWS = 32


def _reference_compressor_state_slots(block_table: torch.Tensor, positions: torch.Tensor, window: int) -> torch.Tensor:
    # SWA chooses the physical block; compressor state has its own ring stride.
    columns = positions // window % block_table.numel()
    return block_table[columns].long() * _COMPRESSOR_STATE_ROWS + positions % _COMPRESSOR_STATE_ROWS


def _reference_indexer_quantization(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # The scale is rounded to FP16 BEFORE division/round-to-even of the codes.
    scales = (rows.float().abs().amax(-1) / 127).half()
    scales = scales.clamp_min(2**-24)
    normalized = rows.float() / scales.float().unsqueeze(-1)
    return normalized.round().clamp(-127, 127).to(torch.int8), scales


def _assert_compressor_index_rows(cache: LayerCache, block: int, first: int, rows: torch.Tensor) -> None:
    assert cache.index.dtype == torch.int8
    assert cache.indexer_scale.dtype == torch.float16
    codes, scales = _reference_indexer_quantization(rows)
    end = first + rows.size(0)
    torch.testing.assert_close(cache.index[block, first:end], codes, atol=0, rtol=0)
    torch.testing.assert_close(cache.indexer_scale[block, first:end, 0], scales, atol=0, rtol=0)


def _with_ring32_compressor_states(caches: list[LayerCache]) -> list[LayerCache]:
    """Adapt the shared SWA-sized fixture locally, including device/dtype users."""
    adapted = []
    for cache in caches:
        if cache.compress_kv_state is not None:
            state_shape = (cache.swa.size(0), _COMPRESSOR_STATE_ROWS, cache.swa.size(-1))
            cache = replace(
                cache,
                compress_kv_state=torch.zeros(state_shape, dtype=torch.float32, device=cache.swa.device),
                compress_score_state=torch.zeros(state_shape, dtype=torch.float32, device=cache.swa.device),
            )
        adapted.append(cache)
    return adapted


@pytest.mark.parametrize("device_name", ["cpu", "npu"])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("prefixes,chunks", [([1, 7], [2, 17]), ([15, 31], [33, 17]), ([31, 1], [1, 2])])
def test_compressed_cp_chunk_and_decode_match_unsharded(
    device_name: str, cp_size: int, prefixes: list[int], chunks: list[int]
) -> None:
    """Real ratio-2 compressor -> INT8 indexer -> paged attention on each CP shard.

    Sequential rank execution checks local/global causality and replicated
    producer caches; it intentionally does not claim HCCL collective coverage.
    NaN tail rows detect reads beyond committed capacity. Odd prefixes park a
    compressor member, and long chunks wrap ring32; three following decodes
    check the persisted state, including CP ranks with no owned query rows.
    """
    if device_name == "npu" and _active_device().type == "cpu":
        pytest.skip("NPU is unavailable")
    device = torch.device("cpu") if device_name == "cpu" else _active_device()
    lm = _make_forward_lm(torch.float32, seed=23, device=device)
    cfg, batch = lm.cfg, len(prefixes)
    layer_id = 2
    layer = lm.model.layers[layer_id].self_attn
    assert layer.mode.is_full and layer.mode.compress_ratio == 2
    torch.manual_seed(817)
    ends = [p + c + 3 for p, c in zip(prefixes, chunks)]
    x = [torch.randn(n, cfg.hidden_size, device=device) for n in ends]
    qr = [torch.randn(n, cfg.q_lora_rank, device=device) for n in ends]
    kv = [torch.randn(n, cfg.head_dim, device=device) for n in ends]
    q = [torch.randn(n, cfg.n_heads, cfg.head_dim, device=device) for n in ends]
    tables = [torch.arange(1, 1 + batch * 2, dtype=torch.int32, device=device).view(batch, 2)]
    token_table = torch.arange(1, 1 + batch, dtype=torch.int32, device=device).view(batch, 1)
    tables.extend([token_table, token_table])
    steps = [prefixes, chunks, [1] * batch, [1] * batch, [1] * batch]
    baseline_outputs, baseline_picks, baseline_caches = [], [], []
    for rank in range(-1, cp_size):
        backend = _make_csa2_backend(cfg, device)
        caches = _with_ring32_compressor_states(
            _allocate_v41_layer_caches(cfg, batch, torch.float32, device, reserve_token_sink=True)
        )
        cache = caches[layer_id]
        for name in ("swa", "key", "indexer_scale", "compress_kv_state", "compress_score_state"):
            getattr(cache, name).fill_(float("nan"))
        cache.index.fill_(-128)
        backend.bind_kv_caches(caches)
        contexts = [0] * batch
        for step, lengths in enumerate(steps):
            next_contexts = [p + n for p, n in zip(contexts, lengths)]
            positions = torch.cat([torch.arange(p, end, device=device) for p, end in zip(contexts, next_contexts)])
            current_x, current_qr, current_kv, current_q = [
                torch.cat([rows[p:end] for rows, p, end in zip(values, contexts, next_contexts)])
                for values in (x, qr, kv, q)
            ]
            layer._csa2_ctx = Csa2LayerContext(current_x, current_qr, positions, lm.model.compress_rotary.cos_sin_cache)
            dsa = backend._builder.build(
                tables,
                next_contexts,
                lengths,
                positions,
                None,
                is_prefill=step == 0,
                is_chunked_prefill=step < 2,
            )
            # The shared V4 builder creates host lengths; serving moves them
            # before execution. Preserve that real metadata/device contract.
            backend._move_metadata_to_device(dsa)
            metadata = SimpleNamespace(dsa_metadata=dsa)
            backend.prepare(metadata)
            plan = build_dsa_cp_context(sum(lengths), lengths, cp_size, rank) if rank >= 0 else None
            context = ForwardContext(backend, device, metadata, [], cp_context=plan)
            if plan is not None:
                context.layer_shared_cache[DsaCpRuntime] = DsaCpRuntime(plan, plan.local_context_lengths(dsa.seq_lens))
            with torch.inference_mode(), forward_context(context):
                actual = backend.execute_csa2_layer(
                    plan.shard(current_q) if plan is not None else current_q, current_kv, layer
                )
            assert torch.isfinite(actual).all()
            topk = layer._csa2_used_topk
            picks = [
                topk[seq, :length].clone() for seq, length in enumerate(lengths if plan is None else plan.local_q_lens)
            ]
            state = {
                name: getattr(cache, name).clone()
                for name in ("swa", "key", "index", "indexer_scale", "compress_kv_state", "compress_score_state")
            }
            if plan is None:
                baseline_outputs.append(actual.clone())
                baseline_picks.append(picks)
                baseline_caches.append(state)
            else:
                torch.testing.assert_close(actual, plan.shard(baseline_outputs[step]), atol=2e-6, rtol=2e-5)
                offset = 0
                for seq, (global_len, local_len) in enumerate(zip(lengths, plan.local_q_lens)):
                    start = max(plan.start - offset, 0)
                    expected_picks = baseline_picks[step][seq][start : start + local_len]
                    expected_picks = torch.where(
                        expected_picks >= 0, expected_picks - global_len + local_len, expected_picks
                    )
                    # Capacity is rank-local, so top-k padding widths can differ.
                    # Compare selected compressed rows, not irrelevant -1 columns.
                    for actual_row, expected_row in zip(picks[seq], expected_picks):
                        torch.testing.assert_close(
                            actual_row[actual_row >= 0], expected_row[expected_row >= 0], atol=0, rtol=0
                        )
                    offset += global_len
                for name, tensor in state.items():
                    torch.testing.assert_close(tensor, baseline_caches[step][name], atol=0, rtol=0, equal_nan=True)
            contexts = next_contexts


class _CompressorHarness:
    """Shared cache/metadata wiring; each test keeps its own numerical oracle."""

    def __init__(
        self,
        lm: DeepseekV41ForCausalLM,
        *,
        batch: int = 1,
        layer_id: int = 2,
        ratio: int | None = None,
        ring_blocks: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.cfg = lm.cfg if ratio is None else replace(lm.cfg, compress_ratios=[0, 0, ratio, ratio, 1, 1])
        self.backend = _make_csa2_backend(self.cfg)
        caches = _allocate_v41_layer_caches(self.cfg, batch, dtype)
        self.cache = caches[layer_id]
        if ring_blocks is not None:
            swa = torch.zeros(ring_blocks * batch, self.cfg.window_size, self.cfg.head_dim, dtype=dtype)
            caches[layer_id] = replace(self.cache, swa=swa)
        # Keep the #503 fixture adaptation in this suite's write set.
        caches = _with_ring32_compressor_states(caches)
        self.cache = caches[layer_id]
        self.backend.bind_kv_caches(caches)
        self.layer = lm.model.layers[layer_id].self_attn
        if ratio is not None:
            self.layer.mode = replace(self.layer.mode, compress_ratio=ratio)
        self.mapping = self.backend._resolve_cache_mapping(layer_id, self.layer.mode.compress_ratio)
        self.cos_sin = lm.model.compress_rotary.cos_sin_cache
        self.shared = CSA2SharedRuntime()

    def run(
        self,
        ctx: Csa2LayerContext,
        tables: list[torch.Tensor],
        seq_lens: list[int],
        q_lens: list[int],
        is_prefill: bool = False,
    ) -> None:
        dsa = self.backend._builder.build(tables, seq_lens, q_lens, ctx.positions, None, is_prefill, True)
        self.backend._run_compressor(self.layer, ctx, dsa, self.cache, self.mapping, q_lens, dsa.seq_lens, self.shared)


@pytest.mark.parametrize("ring_blocks", [1, 2])
@pytest.mark.parametrize("prefix,chunk", [(1, 17), (7, 33), (15, 34), (31, 33), (1, 63), (33, 31), (47, 17)])
@pytest.mark.parametrize("reverse_blocks", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ratio2_compressor_reads_parked_prefix_before_wrapping(
    ring_blocks: int,
    prefix: int,
    chunk: int,
    reverse_blocks: bool,
    dtype: torch.dtype,
) -> None:
    lm = _make_forward_lm(dtype, seed=11)
    harness = _CompressorHarness(lm, ring_blocks=ring_blocks, dtype=dtype)
    cfg, cache, layer, cos_sin = harness.cfg, harness.cache, harness.layer, harness.cos_sin
    # Only this chunk and the one parked prefix member are needed. Varying
    # them independently makes a ring-overwrite bug produce a large error.
    torch.manual_seed(53)
    x = torch.randn(chunk + 1, cfg.hidden_size).to(dtype)
    kv, score = layer.compress_kv_score(x)
    table = torch.arange(ring_blocks, dtype=torch.int32).view(1, ring_blocks)
    if reverse_blocks:
        table = table.flip(-1)
    assert cache.compress_kv_state.shape == (ring_blocks, 32, cfg.head_dim)
    assert cache.compress_score_state.shape == cache.compress_kv_state.shape
    assert cache.compress_kv_state.dtype == cache.compress_score_state.dtype == torch.float32
    cache.compress_kv_state.fill_(float("nan"))
    cache.compress_score_state.fill_(float("nan"))
    cache.key.fill_(float("nan"))
    cache.index.fill_(-128)  # Outside the valid symmetric INT8 code range.
    cache.indexer_scale.fill_(float("nan"))
    old_slot = _reference_compressor_state_slots(table[0], torch.tensor([prefix - 1]), cfg.window_size)[0]
    cache.compress_kv_state.view(-1, cfg.head_dim)[old_slot] = kv[0]
    cache.compress_score_state.view(-1, cfg.head_dim)[old_slot] = score[0]
    assert prefix + chunk <= cos_sin.size(0), "test must fit its configured RoPE horizon"
    positions = torch.arange(prefix, prefix + chunk)
    token_table = torch.tensor([[0]], dtype=torch.int32)
    ctx = Csa2LayerContext(x[1:], torch.randn(chunk, cfg.q_lora_rank).to(dtype), positions, cos_sin)
    harness.run(ctx, [table, token_table, token_table], [prefix + chunk], [chunk])
    committed = (prefix + chunk) // 2 - prefix // 2
    member_count = committed * 2
    members = kv[:member_count].float().view(committed, 2, cfg.head_dim)
    gates = score[:member_count].float().view(committed, 2, cfg.head_dim).softmax(1)
    latent = layer.compress_rmsnorm((members * gates).sum(1).to(dtype))
    starts = prefix - 1 + torch.arange(committed) * 2
    # #503 commits RoPE directly in the cache dtype, without FP4 QDQ.
    key = apply_interleaved_rope(latent, starts, cos_sin).to(cache.key.dtype)
    index = apply_interleaved_rope(layer.indexer_key(latent), starts, cos_sin)
    first = prefix // 2
    torch.testing.assert_close(cache.key[0, first : first + committed], key, atol=1e-6, rtol=1e-6)
    _assert_compressor_index_rows(cache, 0, first, index)
    for untouched in (slice(0, first), slice(first + committed, None)):
        assert torch.isnan(cache.key[0, untouched]).all()
        assert (cache.index[0, untouched] == -128).all()
        assert torch.isnan(cache.indexer_scale[0, untouched]).all()
    assert harness.shared.index_cache is cache.index
    assert harness.shared.index_scale is cache.indexer_scale
    pending = torch.arange((prefix + chunk) // 2 * 2, prefix + chunk)
    if pending.numel():
        slots = _reference_compressor_state_slots(table[0], pending, cfg.window_size)
        torch.testing.assert_close(cache.compress_kv_state.view(-1, cfg.head_dim)[slots], kv[-1:].float())
        torch.testing.assert_close(cache.compress_score_state.view(-1, cfg.head_dim)[slots], score[-1:].float())


@npu_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("prompt_len,chunk_size", [(33, 8), (65, 15), (65, 16), (65, 17), (127, 31)])
def test_chunked_model_logits_and_following_decode_match_token_replay(
    prompt_len: int,
    chunk_size: int,
    dtype: torch.dtype,
) -> None:
    device = _active_device()
    config = {**_DEEPSEEK_V41_FORWARD_CONFIG, "max_position_embeddings": 512, "dtype": dtype, "device": str(device)}
    from tests.python.test_deepseek_v41_model import _random_init_v41
    from xllm.python.models.deepseek_v41 import DeepseekV41ForCausalLM

    torch.manual_seed(101)
    lm = DeepseekV41ForCausalLM(config)
    _random_init_v41(lm)
    reference = _V41ForwardHarness(lm, config, 2, dtype, device=device)
    chunked = _V41ForwardHarness(lm, config, 2, dtype, device=device)
    full = _V41ForwardHarness(lm, config, 2, dtype, device=device)
    for harness in (reference, chunked, full):
        harness.layer_caches = _with_ring32_compressor_states(harness.layer_caches)
        # The shared constructor already bound the old fixture. Before any
        # forward/capture, reopen binding so backend and runners use ring32;
        # otherwise the executor silently ignores this second bind.
        harness.executor._kv_bound = False
        harness.executor.bind_kv_caches(harness.layer_caches)
        for bound, cache in zip(harness.backend._kv_caches, harness.layer_caches):
            assert bound.compress_kv_state is cache.compress_kv_state
            assert bound.compress_score_state is cache.compress_score_state
        for cache in harness.layer_caches:
            for name in ("swa", "key", "index", "indexer_scale", "compress_kv_state", "compress_score_state"):
                tensor = getattr(cache, name)
                if tensor is not None:
                    tensor.fill_(-128 if tensor.dtype == torch.int8 else float("nan"))
    ids = (torch.arange(2 * prompt_len) % lm.cfg.vocab_size).reshape(2, prompt_len).to(device)
    with torch.inference_mode():
        expected = torch.stack(
            [
                reference.step(ids[:, p], torch.full((2,), p, device=device), [p + 1] * 2, [1] * 2, is_prefill=False)
                for p in range(prompt_len)
            ],
            dim=1,
        )
        full_hidden = full.step(
            ids.reshape(-1),
            torch.arange(prompt_len, device=device).repeat(2),
            [prompt_len] * 2,
            [prompt_len] * 2,
            is_prefill=True,
        ).view(2, prompt_len, -1)
        chunks = []
        for start in range(0, prompt_len, chunk_size):
            end = min(start + chunk_size, prompt_len)
            hidden = chunked.step(
                ids[:, start:end].reshape(-1),
                torch.arange(start, end, device=device).repeat(2),
                [end] * 2,
                [end - start] * 2,
                is_prefill=start == 0,
                is_chunked_prefill=True,
            )
            chunks.append(hidden.view(2, end - start, -1))
        actual = torch.cat(chunks, dim=1)
        reference_logits = reference.logits(expected.reshape(-1, lm.cfg.hidden_size))
        full_logits = full.logits(full_hidden.reshape(-1, lm.cfg.hidden_size))
        chunked_logits = chunked.logits(actual.reshape(-1, lm.cfg.hidden_size))
        _assert_model_logits_close(full_logits, reference_logits, dtype)
        _assert_model_logits_close(chunked_logits, full_logits, dtype)
        _assert_model_logits_close(chunked_logits, reference_logits, dtype)
        for p in range(prompt_len, prompt_len + 4):
            args = (torch.tensor([7, 11], device=device), torch.full((2,), p, device=device), [p + 1] * 2, [1] * 2)
            reference_logits = reference.logits(reference.step(*args, is_prefill=False))
            full_logits = full.logits(full.step(*args, is_prefill=False))
            chunked_logits = chunked.logits(chunked.step(*args, is_prefill=False))
            _assert_model_logits_close(full_logits, reference_logits, dtype)
            _assert_model_logits_close(chunked_logits, full_logits, dtype)
            _assert_model_logits_close(chunked_logits, reference_logits, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hc_mult", [2, 4, 8])
def test_hc_pre_stream_reduction_is_partition_invariant(dtype: torch.dtype, hc_mult: int) -> None:
    """Identical HC inputs must not change at BF16 midpoints with token geometry."""
    device = _active_device()
    torch.manual_seed(101)
    hidden = torch.randn(129, hc_mult, 128, dtype=dtype, device=device)
    pre_mix = torch.rand(129, hc_mult, dtype=torch.float32, device=device)
    if hc_mult == 4:
        # Real first divergence: layer 2, sequence 1, position 60. The generic
        # NPU sum crossed opposite sides of a BF16 midpoint with these exact
        # inputs even though both the stream and its coefficients matched.
        hidden[60, :, 108] = torch.tensor(
            [-0.193359375, -0.193359375, -0.1923828125, -0.1953125], dtype=dtype, device=device
        )
        pre_mix[60] = torch.tensor(
            [0.5022825598716736, 0.49681442975997925, 0.5081950426101685, 0.49532201886177063],
            dtype=torch.float32,
            device=device,
        )
    hc = object.__new__(DeepseekV41HyperConnection)
    contributions = (pre_mix.unsqueeze(-1) * hidden.float()).unbind(dim=-2)
    expected = contributions[0]
    for contribution in contributions[1:]:
        expected = expected + contribution
    expected = expected.to(dtype)
    actual = hc.hc_pre(hidden, pre_mix)
    partitioned = torch.cat(
        [hc.hc_pre(hidden[start : start + 5], pre_mix[start : start + 5]) for start in range(0, 129, 5)]
    )
    replay = torch.cat([hc.hc_pre(hidden[index : index + 1], pre_mix[index : index + 1]) for index in range(129)])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(partitioned, actual, atol=0, rtol=0)
    torch.testing.assert_close(replay, actual, atol=0, rtol=0)


@pytest.mark.parametrize("required_mode", ["selected", "fused"])
@pytest.mark.parametrize("chunk_size", [1, 8, 31])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_selected_hc_mode_is_partition_invariant(
    required_mode: str, chunk_size: int, dtype: torch.dtype, record_property: Callable[[str, str], None]
) -> None:
    """Exercise the constructor's real dispatch, including native external-pre collapse.

    No fake op or mode override: a fused-only case skips when the installed
    runtime cannot select it. The ordinary case must always run its selected
    path. This covers HC inputs, not full-checkpoint serving acceptance.
    """
    device = _active_device()
    cfg = DeepseekV41Config.from_dict(_DEEPSEEK_V41_FORWARD_CONFIG)
    hc = DeepseekV41HyperConnection(cfg, dtype, device)
    record_property("hc_pre_mode", hc._hc_pre_mode)
    record_property("device", str(device))
    if required_mode == "fused" and hc._hc_pre_mode != "fused":
        pytest.skip(f"native HC not selected: mode={hc._hc_pre_mode}, device={device}, dtype={dtype}")
    torch.manual_seed(101)
    with torch.no_grad():
        for name, parameter in hc.named_parameters():
            parameter.normal_(std=0.004 if name.endswith("_fn") else 0.5)
    hidden = torch.randn(129, cfg.hc_mult, cfg.hidden_size, device=device).to(dtype)
    pre_mix = torch.rand(129, cfg.hc_mult, device=device) + 0.1
    # Pin the previously observed collapse midpoint while traversing dispatch.
    hidden[60, :, 108] = hidden.new_tensor([-0.193359375, -0.193359375, -0.1923828125, -0.1953125])
    pre_mix[60] = pre_mix.new_tensor([0.5022825598716736, 0.49681442975997925, 0.5081950426101685, 0.49532201886177063])

    def run(x: torch.Tensor, previous_pre: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if hc._hc_pre_mode == "fused":
            return hc.hc_pre_fused(x, hc.hc_attn_fn, hc.hc_attn_scale, hc.hc_attn_base, previous_pre)
        pre, post, comb = hc.hc_mixes(x, hc.hc_attn_fn, hc.hc_attn_scale, hc.hc_attn_base)
        return hc.hc_pre(x, previous_pre), pre, post, comb

    with torch.inference_mode():
        full = run(hidden, pre_mix)
        pieces = [
            run(hidden[start : start + chunk_size], pre_mix[start : start + chunk_size])
            for start in range(0, hidden.size(0), chunk_size)
        ]
        for index, expected in enumerate(full):
            actual = torch.cat([piece[index] for piece in pieces])
            assert torch.isfinite(actual).all()
            # A BF16 collapse must not cross a rounding midpoint. FP32 native
            # coefficients retain the existing model's tight FP32 tolerance.
            torch.testing.assert_close(
                actual,
                expected,
                atol=0 if dtype == torch.bfloat16 and index == 0 else 2e-5,
                rtol=0 if dtype == torch.bfloat16 and index == 0 else 2e-5,
            )


def _assert_model_logits_close(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
    """FP32 pins semantics; BF16 additionally checks bounded numerical drift.

    Batch-dependent BF16 GEMMs can differ by a rounding unit, which INT8 quantization
    and sparse top-k amplify. Per-layer FP32/cache controls establish this
    separately from ring corruption. BF16 is not required to be bitwise equal;
    both an absolute ceiling and aggregate relative error constrain the drift.
    """
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        return
    assert torch.isfinite(actual).all()
    error = (actual - expected).abs()
    assert error.max() <= 0.0625
    relative_rms = error.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-6)
    assert relative_rms <= 0.005
    cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=-1)
    assert cosine.min() >= 0.999


@pytest.mark.parametrize("slots", [[], [-1, -1], [3], [4, 1, 4, -1, 1, 4, 2], [0] * 257, list(range(128)) * 17])
def test_last_cache_writes_are_unique_latest_and_ignore_padding(slots: list[int]) -> None:
    values = torch.arange(len(slots) * 3, dtype=torch.float32).view(-1, 3)
    physical = torch.tensor(slots, dtype=torch.int32)
    latest = {slot: i for i, slot in enumerate(slots) if slot >= 0}
    actual_slots, actual_values = last_ring_writes(physical, values)
    assert actual_slots.tolist() == sorted(latest)
    expected_rows = torch.tensor([latest[slot] for slot in sorted(latest)], dtype=torch.long)
    torch.testing.assert_close(actual_values, values[expected_rows], atol=0, rtol=0)


def test_last_cache_writes_rejects_misaligned_values() -> None:
    with pytest.raises(ValueError, match="row counts"):
        last_ring_writes(torch.tensor([1, 2]), torch.ones(1, 4))


@pytest.mark.parametrize("prefix", [15, 31, 255, 511])
def test_decode_drops_released_oldest_window_candidate(prefix: int) -> None:
    """The oldest ring candidate is outside the window and may be released.

    Production SWA tables retain invalid placeholders for released blocks.
    At the last token of a block, the extra candidate comes from a released
    block whereas every actually visible historical token remains allocated.
    """
    window, width, heads = 8, 4, 4
    backend = _make_swa_backend(window, width, heads)
    torch.manual_seed(119)
    values = torch.randn(prefix + 1, width)
    q = torch.randn(1, heads, width)
    cache = torch.full((3, window, width), float("nan"))
    # Only the last two logical blocks are retained; earlier placeholders
    # resolve to the deliberately poisoned reserved block zero.
    logical_blocks = (prefix + 1 + window - 1) // window
    table = torch.full((1, logical_blocks), -1, dtype=torch.int32)
    table[0, -2:] = torch.tensor([1, 2], dtype=torch.int32)
    table[0, -2] = -1
    visible = torch.arange(prefix - window + 1, prefix)
    slots = table[0, visible // window] * window + visible % window
    _scatter_by_slot(cache, slots, values[visible])
    backend.bind_kv_caches([LayerCache(None, None, swa=cache)])
    layer = SimpleNamespace(
        mode=CSA2LayerMode(0, CSA2_MODE_SWA, 0, -1),
        layer_id=0,
        attn_sink=torch.zeros(heads),
        _csa2_ctx=Csa2LayerContext(values[-1:], values[-1:], torch.tensor([prefix]), torch.empty(0)),
    )
    dsa = backend._builder.build([table], [prefix + 1], [1], torch.tensor([prefix]), None, False, False)
    metadata = SimpleNamespace(dsa_metadata=dsa)
    backend.prepare(metadata)
    with torch.inference_mode(), forward_context(ForwardContext(backend, torch.device("cpu"), metadata, [])):
        actual = backend.execute_csa2_layer(q, values[-1:], layer)
    expected = _reference_swa(q, values, prefix, window, layer.attn_sink)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_empty_dp_metadata_has_no_fabricated_csa2_history() -> None:
    """Shared V4 native tiling's dummy context must not become CSA2 KV."""
    backend = _make_swa_backend(8, 4, 4)
    cache = torch.full((2, 8, 4), float("nan"))
    backend.bind_kv_caches([LayerCache(None, None, swa=cache)])
    metadata = SimpleNamespace(
        multi_block_tables=[],
        kv_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        dsa_positions=torch.tensor([0]),
        is_prefill=False,
        is_chunked_prefill=False,
        is_dummy=True,
        dsa_graph_mode=False,
    )
    dsa = backend._build_dsa_metadata_for_forward(metadata)
    assert dsa.seq_lens.tolist() == [0]
    assert dsa.actual_seq_lengths_kv.tolist() == [0]
    assert dsa.seq_lens_q.tolist() == [1]
    assert dsa.kv_cu_seq_lens.tolist() == [0, 0]
    assert dsa.start_pos.tolist() == [-1]
    assert all((slots == -1).all() for layer in dsa.slot_mappings for slots in layer)
    metadata.dsa_metadata = dsa
    backend.prepare(metadata)
    layer = SimpleNamespace(
        mode=CSA2LayerMode(0, CSA2_MODE_SWA, 0, -1),
        layer_id=0,
        attn_sink=torch.zeros(4),
        _csa2_ctx=Csa2LayerContext(torch.ones(1, 4), torch.ones(1, 4), torch.tensor([0]), torch.empty(0)),
    )
    with torch.inference_mode(), forward_context(ForwardContext(backend, torch.device("cpu"), metadata, [])):
        output = backend.execute_csa2_layer(torch.ones(1, 4, 4), torch.ones(1, 4), layer)
    assert torch.equal(output, torch.zeros_like(output))
    assert torch.isnan(cache).all()


@pytest.mark.parametrize("graph_metadata", [False, True])
def test_dummy_model_forward_keeps_positions_safe_and_caches_inert(
    graph_metadata: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A synthetic query retains start=-1 without exposing it as a RoPE index."""
    from xllm.python.attention import csa2_attention

    device = _active_device()
    lm = _make_forward_lm(torch.float32, seed=101, device=device)
    backend = _make_csa2_backend(lm.cfg, device=device, max_model_len=64)
    caches = _with_ring32_compressor_states(
        _allocate_v41_layer_caches(lm.cfg, 1, torch.float32, device=device, reserve_token_sink=True)
    )
    for cache in caches:
        for name in ("swa", "key", "index", "indexer_scale", "compress_kv_state", "compress_score_state"):
            tensor = getattr(cache, name)
            if tensor is not None:
                tensor.fill_(-128 if tensor.dtype == torch.int8 else float("nan"))
    backend.bind_kv_caches(caches)
    metadata = SimpleNamespace(
        multi_block_tables=[],
        kv_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        is_prefill=False,
        is_chunked_prefill=False,
        is_dummy=True,
        dsa_graph_mode=graph_metadata,
        dsa_graph_token_capacity=64 if graph_metadata else 0,
        dsa_graph_block_table_cols=4 if graph_metadata else 0,
        is_spec_verify=False,
    )
    rope_calls: list[torch.Tensor] = []
    original_rope = csa2_attention.apply_interleaved_rope

    def checked_rope(
        value: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor, *, inverse: bool = False
    ) -> torch.Tensor:
        assert (positions >= 0).all()
        assert (positions < cos_sin_cache.size(0)).all()
        rope_calls.append(positions.detach().cpu())
        return original_rope(value, positions, cos_sin_cache, inverse=inverse)

    monkeypatch.setattr(csa2_attention, "apply_interleaved_rope", checked_rope)
    with torch.inference_mode(), forward_context(ForwardContext(backend, device, metadata, caches)):
        output = lm.model(torch.tensor([1], device=device), torch.tensor([0], device=device))
    assert rope_calls
    assert torch.isfinite(output).all()
    dsa = metadata.dsa_metadata
    torch.testing.assert_close(dsa.start_pos, dsa.seq_lens - dsa.seq_lens_q, atol=0, rtol=0)
    assert dsa.start_pos.tolist() == [-1]
    assert all((slots == -1).all() for layer_slots in dsa.slot_mappings for slots in layer_slots)
    for layer in lm.model.layers:
        topk = getattr(layer.self_attn, "_csa2_used_topk", None)
        if topk is not None:
            assert (topk == -1).all()
    for cache in caches:
        for name in ("swa", "key", "index", "indexer_scale", "compress_kv_state", "compress_score_state"):
            tensor = getattr(cache, name)
            if tensor is not None:
                assert (tensor == -128).all() if tensor.dtype == torch.int8 else torch.isnan(tensor).all()


@npu_only
@pytest.mark.parametrize("graph_metadata", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dummy_model_triton_compressor_keeps_ring_states_inert(
    graph_metadata: bool, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real empty-DP builder tables reach Triton as block 0 with start=-1.

    The observer delegates to the real kernel; fallback cannot satisfy the
    successful-launch assertions. Graph metadata is covered, not capture.
    """
    device = _active_device()
    lm = _make_forward_lm(dtype, seed=101, device=device)
    backend = _make_csa2_backend(lm.cfg, device=device, max_model_len=64)
    caches = _with_ring32_compressor_states(
        _allocate_v41_layer_caches(lm.cfg, 1, dtype, device=device, reserve_token_sink=True)
    )
    before = []
    for cache in caches:
        if cache.compress_kv_state is not None:
            # Include the reserved block: neither it nor a live page may change.
            cache.compress_kv_state.copy_(torch.randn_like(cache.compress_kv_state))
            cache.compress_score_state.copy_(torch.randn_like(cache.compress_score_state))
            before.append((cache.compress_kv_state.clone(), cache.compress_score_state.clone()))
    backend.bind_kv_caches(caches)
    assert backend._swa_ring_blocks == 1
    metadata = SimpleNamespace(
        multi_block_tables=[],
        kv_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        is_prefill=False,
        is_chunked_prefill=False,
        is_dummy=True,
        dsa_graph_mode=graph_metadata,
        dsa_graph_token_capacity=64 if graph_metadata else 0,
        dsa_graph_block_table_cols=4 if graph_metadata else 0,
        is_spec_verify=False,
    )
    module = csa2._get_v41_compressor_module()
    assert module is not None, "NPU regression requires the production Triton compressor"
    original = module.compressor_from_projected
    launches: list[tuple[torch.Tensor, torch.Tensor]] = []

    def observe_launch(*args: torch.Tensor, **kwargs: int) -> torch.Tensor:
        result = original(*args, **kwargs)
        # Preserve evidence only after the actual launch succeeds.
        launches.append((args[4].detach().clone(), result.detach().clone()))
        return result

    monkeypatch.setattr(module, "compressor_from_projected", observe_launch)
    with torch.inference_mode(), forward_context(ForwardContext(backend, device, metadata, caches)):
        output = lm.model(torch.tensor([1], device=device), torch.tensor([0], device=device))
    torch.npu.synchronize()
    dsa = metadata.dsa_metadata
    assert dsa.start_pos.tolist() == [-1]
    assert dsa.seq_lens.tolist() == [0]
    assert dsa.seq_lens_q.tolist() == [1]
    assert dsa.block_tables and all(table.numel() > 0 for tables in dsa.block_tables for table in tables)
    # Graph storage may pad zeroed empty-DP tables with -1; both resolve
    # to reserved block 0, never a live state page.
    assert all((table <= 0).all() for tables in dsa.block_tables for table in tables)
    expected_launches = sum(layer.mode.is_full and layer.mode.compress_ratio == 2 for layer in lm.model.layers)
    assert expected_launches > 0
    assert len(launches) == expected_launches
    for controls, pooled in launches:
        assert controls[:, 0].tolist() == [-1, 1, 0, 0, 0]
        torch.testing.assert_close(pooled, torch.zeros_like(pooled), atol=0, rtol=0)
    assert torch.isfinite(output).all()
    after = [cache for cache in caches if cache.compress_kv_state is not None]
    for cache, (kv, scores) in zip(after, before, strict=True):
        torch.testing.assert_close(cache.compress_kv_state, kv, atol=0, rtol=0)
        torch.testing.assert_close(cache.compress_score_state, scores, atol=0, rtol=0)


@pytest.mark.parametrize("previous", [31, 47, 65])
def test_window_gather_is_chronological_with_unordered_physical_blocks(previous: int) -> None:
    """Ring wraps and block-ID order do not reorder the absolute-position gather."""
    device = _active_device()
    window, query_len = 8, 17
    backend = _make_swa_backend(window, 4, 4)
    # Logical block order is deliberately unrelated to physical allocation order.
    table = torch.tensor([[3, 1, 4]], dtype=torch.int32, device=device)
    cache = torch.full((5, window, 1), float("nan"), device=device)
    history = torch.arange(previous - table.size(1) * window, previous, device=device)
    slots = table[0, history // window % table.size(1)] * window + history % window
    cache.view(-1, 1)[slots.long()] = history.float().unsqueeze(-1)
    ring_positions = torch.arange(previous - window, previous, device=device)
    ring = backend._gather_swa_positions(cache.view(-1, 1), table, 0, ring_positions, window)
    torch.testing.assert_close(ring.flatten(), ring_positions.float(), atol=0, rtol=0)
    positions = torch.arange(previous, previous + query_len, device=device)
    window_kv = torch.cat((ring, positions.float().unsqueeze(-1)))
    picks = _window_visible_idxs(positions, torch.tensor(previous, device=device), window)
    expected_positions = positions.unsqueeze(-1) - window + 1 + torch.arange(window, device=device)
    torch.testing.assert_close(window_kv[picks.long()].squeeze(-1), expected_positions.float(), atol=0, rtol=0)


@pytest.mark.parametrize(
    "modules",
    [
        ("models.deepseek_v41_engram", "attention.csa2_attention"),
        ("attention.csa2_attention", "models.deepseek_v41_engram"),
        ("attention.dsa_metadata", "models.deepseek_v41_engram", "attention.csa2_attention"),
    ],
)
def test_ring_write_helper_imports_in_fresh_process_without_model_cycle(modules: tuple[str, ...]) -> None:
    """Import order must not depend on an already-initialized model/backend."""
    root = Path(__file__).resolve().parents[2]
    program = """
import importlib
import sys
import tests.python.conftest
modules = [importlib.import_module("xllm.python." + name) for name in sys.argv[1:]]
metadata = importlib.import_module("xllm.python.attention.dsa_metadata")
backend = importlib.import_module("xllm.python.attention.csa2_attention")
engram = importlib.import_module("xllm.python.models.deepseek_v41_engram")
assert metadata.last_ring_writes.__module__ == metadata.__name__
assert backend.last_ring_writes is metadata.last_ring_writes
assert engram.last_ring_writes is metadata.last_ring_writes
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, *modules], cwd=root, capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("exponents", [range(-24, -14), range(-14, 1), range(1, 15)])
def test_indexer_fp16_scale_binades_preserve_nextafter_boundaries(exponents: range) -> None:
    """Pin FP16 scale transitions, including the subnormal floor, without QDQ."""
    scales = torch.exp2(torch.tensor(list(exponents), dtype=torch.float64)).half()
    next_scales = torch.nextafter(scales, torch.full_like(scales, float("inf")))
    midpoints = (scales.float() + next_scales.float()) * 0.5
    maxima = midpoints * 127
    boundaries = torch.stack(
        [
            torch.nextafter(maxima, torch.zeros_like(maxima)),
            maxima,
            torch.nextafter(maxima, torch.full_like(maxima, float("inf"))),
        ],
        dim=-1,
    ).flatten()
    values = torch.stack([boundaries, -boundaries, boundaries / 3, torch.zeros_like(boundaries)], dim=-1)
    codes, actual_scales = _quantize_indexer_rows_int8(values)
    expected_codes, expected_scales = _reference_indexer_quantization(values)
    torch.testing.assert_close(actual_scales, expected_scales, atol=0, rtol=0)
    torch.testing.assert_close(codes, expected_codes, atol=0, rtol=0)
    assert (actual_scales > 0).all() and torch.isfinite(actual_scales).all()
    assert (actual_scales.reshape(-1, 3)[:, 0] <= actual_scales.reshape(-1, 3)[:, -1]).all()
    partitions = [_quantize_indexer_rows_int8(part) for part in values.tensor_split(3)]
    torch.testing.assert_close(torch.cat([part[0] for part in partitions]), codes, atol=0, rtol=0)
    torch.testing.assert_close(torch.cat([part[1] for part in partitions]), actual_scales, atol=0, rtol=0)


def test_indexer_int8_rounding_all_grid_midpoints() -> None:
    """Pin all half-to-even ties/neighbors, clipping, zero and tiny-row scales."""
    grid = torch.arange(-127, 128, dtype=torch.float32)
    midpoints = (grid[:-1] + grid[1:]) * 0.5
    values = torch.cat(
        [
            grid,
            midpoints,
            torch.nextafter(midpoints, torch.full_like(midpoints, float("-inf"))),
            torch.nextafter(midpoints, torch.full_like(midpoints, float("inf"))),
        ]
    )
    # An exact 127 absmax pins scale=1 independently of the implementation.
    rows = torch.stack([torch.full_like(values, 127), values], dim=-1)
    codes, scales = _quantize_indexer_rows_int8(rows)
    torch.testing.assert_close(scales, torch.ones_like(scales), atol=0, rtol=0)
    torch.testing.assert_close(codes[:, 0], torch.full_like(codes[:, 0], 127), atol=0, rtol=0)
    torch.testing.assert_close(codes[:, 1], values.double().round().to(torch.int8), atol=0, rtol=0)
    assert (codes != -128).all()
    edge_rows = torch.tensor([[0, 0], [2**-25, -(2**-25)], [2**-24, -(2**-24)], [127.04, -127.04]])
    edge_codes, edge_scales = _quantize_indexer_rows_int8(edge_rows)
    expected_codes = torch.tensor([[0, 0], [0, 0], [1, -1], [127, -127]], dtype=torch.int8)
    expected_scales = torch.tensor([2**-24, 2**-24, 2**-24, 1], dtype=torch.float16)
    torch.testing.assert_close(edge_codes, expected_codes, atol=0, rtol=0)
    torch.testing.assert_close(edge_scales, expected_scales, atol=0, rtol=0)


@pytest.mark.parametrize("fail", [False, True])
def test_attention_releases_request_sized_context_after_handoff(fail: bool) -> None:
    lm = _make_forward_lm(torch.float32, seed=19)
    attention = lm.model.layers[0].self_attn

    def execute(q: torch.Tensor, kv: torch.Tensor, layer: object) -> torch.Tensor:
        assert layer._csa2_ctx is not None
        if fail:
            raise RuntimeError("attention failed")
        return torch.zeros_like(q)

    metadata = SimpleNamespace()
    backend = SimpleNamespace(execute_csa2_layer=execute)
    with torch.inference_mode(), forward_context(ForwardContext(backend, torch.device("cpu"), metadata, [])):
        args = (torch.ones(9, lm.cfg.hidden_size), torch.arange(9), lm.model._layer_rope_cache(attention.mode))
        if fail:
            with pytest.raises(RuntimeError, match="attention failed"):
                attention(*args)
        else:
            attention(*args)
    assert attention._csa2_ctx is None


@pytest.mark.parametrize("ratio", [2, 4])
@pytest.mark.parametrize("chunk", [1, 3, 17, 33])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compressor_preserves_every_unfinished_tail_member(ratio: int, chunk: int, dtype: torch.dtype) -> None:
    lm = _make_forward_lm(dtype, seed=37)
    harness = _CompressorHarness(lm, ratio=ratio, dtype=dtype)
    cfg, cache, layer = harness.cfg, harness.cache, harness.layer
    torch.manual_seed(79)
    values = torch.randn(48, cfg.hidden_size).to(dtype)
    kv, scores = layer.compress_kv_score(values)
    count = values.size(0) // ratio
    grouped_kv = kv[: count * ratio].float().view(count, ratio, cfg.head_dim)
    grouped_scores = scores[: count * ratio].float().view(count, ratio, cfg.head_dim)
    latent = layer.compress_rmsnorm((grouped_kv * grouped_scores.softmax(1)).sum(1).to(dtype))
    starts = torch.arange(count) * ratio
    expected = apply_interleaved_rope(latent, starts, harness.cos_sin).to(cache.key.dtype)
    expected_index = apply_interleaved_rope(layer.indexer_key(latent), starts, harness.cos_sin)
    table = torch.tensor([[0, 1]], dtype=torch.int32)
    token_table = torch.tensor([[0]], dtype=torch.int32)
    end = 0
    while end < values.size(0):
        start = end
        end = min(end + chunk, values.size(0))
        positions = torch.arange(start, end)
        ctx = Csa2LayerContext(values[start:end], values[start:end, : cfg.q_lora_rank], positions, harness.cos_sin)
        harness.run(ctx, [table, token_table, token_table], [end], [end - start], is_prefill=start == 0)
        pending = torch.arange(end // ratio * ratio, end)
        if pending.numel():
            slots = _reference_compressor_state_slots(table[0], pending, cfg.window_size)
            torch.testing.assert_close(cache.compress_kv_state.view(-1, cfg.head_dim)[slots.long()], kv[pending])
            torch.testing.assert_close(cache.compress_score_state.view(-1, cfg.head_dim)[slots.long()], scores[pending])
        committed = end // ratio
        torch.testing.assert_close(cache.key[0, :committed], expected[:committed], atol=2e-6, rtol=2e-5)
        _assert_compressor_index_rows(cache, 0, 0, expected_index[:committed])
    torch.testing.assert_close(cache.key[0, :count], expected, atol=2e-6, rtol=2e-5)
    _assert_compressor_index_rows(cache, 0, 0, expected_index)


@pytest.mark.parametrize("layer_id", [2, 4])
@pytest.mark.parametrize("empty_first", [False, True])
def test_compressor_skips_zero_query_sequences(layer_id: int, empty_first: bool) -> None:
    lm = _make_forward_lm(torch.float32, seed=37)
    harness = _CompressorHarness(lm, batch=2, layer_id=layer_id)
    reference = _CompressorHarness(lm, layer_id=layer_id)
    cfg, cache = harness.cfg, harness.cache
    active_seq = 1 if empty_first else 0
    q_lens = [0, 3] if empty_first else [3, 0]
    positions = torch.arange(3)
    values = torch.randn(3, cfg.hidden_size, generator=torch.Generator().manual_seed(119))
    swa_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    token_table = torch.tensor([[0], [1]], dtype=torch.int32)
    ctx = Csa2LayerContext(values, values[:, : cfg.q_lora_rank], positions, harness.cos_sin)
    with torch.no_grad():
        harness.run(ctx, [swa_table, token_table, token_table], q_lens, q_lens, is_prefill=True)
        reference.run(ctx, [swa_table[:1], token_table[:1], token_table[:1]], [3], [3], is_prefill=True)
    for name in ("key", "index", "indexer_scale", "compress_kv_state", "compress_score_state"):
        actual = getattr(cache, name)
        expected = getattr(reference.cache, name)
        if actual is None:
            assert expected is None
            continue
        blocks = expected.size(0)
        torch.testing.assert_close(actual[active_seq * blocks : (active_seq + 1) * blocks], expected, atol=0, rtol=0)
        inactive = 1 - active_seq
        assert torch.count_nonzero(actual[inactive * blocks : (inactive + 1) * blocks]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [0, 1, 127, 128, 129, 256, 257])
def test_hc_projection_keeps_fp32_geometry_across_chunks(
    dtype: torch.dtype, rows: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unused padding, even NaNs, must not affect retained fixed-row GEMMs."""
    torch.manual_seed(917)
    device = _active_device()
    hc = object.__new__(DeepseekV41HyperConnection)
    hidden = torch.randn(rows, 64, dtype=dtype, device=device).float()
    weight = torch.randn(24, 64, device=device)
    linear = torch.nn.functional.linear
    tiles: list[torch.Tensor] = []
    pointers: list[int] = []

    def observed_linear(value: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
        tiles.append(value.clone())
        pointers.append(value.data_ptr())
        return linear(value, projection)

    new_empty = torch.Tensor.new_empty

    def poisoned_empty(value: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        result = new_empty(value, *args, **kwargs)
        # Simulate stale allocator contents in both scratch and output buffers.
        result.fill_(float("nan"))
        return result

    with torch.inference_mode(), monkeypatch.context() as context:
        context.setattr(torch.Tensor, "new_empty", poisoned_empty)
        context.setattr(torch.nn.functional, "linear", observed_linear)
        actual = hc._project_mixes(hidden, weight)
    assert len(tiles) == (rows + 127) // 128
    assert len(set(pointers)) == (1 if rows else 0)
    old_outputs = []
    with torch.inference_mode():
        for index, tile in enumerate(tiles):
            start = index * 128
            count = min(128, rows - start)
            expected_tile = hidden.new_zeros((128, hidden.size(1)))
            expected_tile[:count].copy_(hidden[start : start + count])
            torch.testing.assert_close(tile[:count], expected_tile[:count], atol=0, rtol=0)
            if count < 128:
                padding = tile[count:]
                assert torch.isnan(padding).all() if start == 0 else torch.isfinite(padding).all()
            old_outputs.append(linear(expected_tile, weight)[:count])
        expected = torch.cat(old_outputs) if old_outputs else actual.new_empty((0, weight.size(0)))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        if rows:
            partitioned = torch.cat(
                [hc._project_mixes(hidden[start : start + 5], weight) for start in range(0, rows, 5)]
            )
            torch.testing.assert_close(partitioned, actual, atol=2e-5, rtol=2e-5)
