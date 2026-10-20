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

"""Wiring equivalence for the V4.1 interleaved RoPE + inverse + pre-expand cache.

Extracted from the monolithic ``test_deepseek_v41_fused_ops.py``.
"""

from __future__ import annotations

import pytest
import torch

import xllm.python.attention.csa2_attention as csa2
from tests.python._deepseek_v41_fused_testkit import (
    _load_fused_ops_lib,
    _npu_ready,
)

# ---------------------------------------------------------------------------
# R5: interleaved partial RoPE (function level)
# ---------------------------------------------------------------------------


def _build_cache(max_pos: int, rotary_dim: int, device: str) -> torch.Tensor:
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    pos = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(device).contiguous()


@pytest.mark.skipif(not _npu_ready() or not _load_fused_ops_lib(), reason="needs NPU + fused ops lib")
def test_interleaved_rope_fused_matches_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    device = "npu"
    # Real V4.1 geometry only: the vendor rotary kernel is correct solely for
    # a rope slice of exactly 64 (headDim < 64 is a documented kernel defect),
    # so the fused-vs-eager comparison runs at qk_rope_head_dim=64.
    cache = _build_cache(256, 64, device)
    cases = [
        (8, 4, 512),  # 3D q-like, partial [448:512)
        (8, 1, 512),  # 3D kv-like single head, partial
        (16, 32, 128),  # 3D indexer q_idx, partial [64:128)
        (8, 512),  # 2D latent-like, partial
        (5, 128),  # 2D index-k-like, partial [64:128)
    ]
    for shape in cases:
        torch.manual_seed(sum(shape) + 7)
        x = (torch.randn(*shape) * 0.9).to(torch.bfloat16).to(device)
        positions = torch.randperm(200)[: shape[0]].to(torch.int32).to(device)

        real_get_op = csa2._get_partial_rotary_op
        monkeypatch.setattr(csa2, "_get_partial_rotary_op", lambda: None)
        eager = csa2.apply_interleaved_rope(x.clone(), positions, cache)
        monkeypatch.setattr(csa2, "_get_partial_rotary_op", real_get_op)
        fused = csa2.apply_interleaved_rope(x.clone(), positions, cache)

        assert fused.shape == eager.shape
        diff = (fused.float() - eager.float()).abs()
        assert (diff <= 2**-6 + 2**-6 * eager.float().abs()).all(), f"shape={shape} max_abs={diff.max().item():.3e}"
        # The un-rotated prefix must survive the in-place kernel untouched.
        rope_width = cache.size(-1)
        if shape[-1] > rope_width:
            assert torch.equal(fused[..., : shape[-1] - rope_width], eager[..., : shape[-1] - rope_width])

    # A rope width below the kernel's stable domain (e.g. the tiny-config
    # qk_rope_head_dim=8) must fall back to eager even with the op available.
    tiny_cache = _build_cache(64, 8, device)
    x = torch.randn(4, 2, 32, dtype=torch.bfloat16, device=device)
    positions = torch.arange(4, dtype=torch.int32, device=device)
    tiny = csa2.apply_interleaved_rope(x.clone(), positions, tiny_cache)
    monkeypatch.setattr(csa2, "_get_partial_rotary_op", lambda: None)
    tiny_ref = csa2.apply_interleaved_rope(x.clone(), positions, tiny_cache)
    monkeypatch.setattr(csa2, "_get_partial_rotary_op", real_get_op)
    assert torch.equal(tiny, tiny_ref)

    # The inverse de-rotation fused path must match its eager reference.
    x = (torch.randn(4, 2, 512) * 0.9).to(torch.bfloat16).to(device)
    positions = torch.arange(4, dtype=torch.int32, device=device)
    inv = csa2.apply_interleaved_rope(x.clone(), positions, cache, inverse=True)
    monkeypatch.setattr(csa2, "_get_partial_rotary_op", lambda: None)
    inv_ref = csa2.apply_interleaved_rope(x.clone(), positions, cache, inverse=True)
    monkeypatch.setattr(csa2, "_get_partial_rotary_op", real_get_op)
    diff = (inv.float() - inv_ref.float()).abs()
    assert (diff <= 2**-6 + 2**-6 * inv_ref.float().abs()).all(), (
        f"inverse max_abs={diff.max().item():.3e}"
    )


@pytest.mark.skipif(not _npu_ready() or not _load_fused_ops_lib(), reason="needs NPU + fused ops lib")
def test_interleaved_rope_fused_inverse_matches_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    """P1-a: the R2 inverse de-rotation fused path must match eager on NPU."""
    device = "npu"
    # Same geometry contract as the forward test: the vendor kernel is correct
    # only for a rope slice of exactly 64, and the inverse table is the same
    # per-layer cache the forward call uses.
    cache = _build_cache(256, 64, device)
    cases = [
        (8, 4, 512),  # 3D q-like attn_out, partial [448:512)
        (5, 128),  # 2D latent-like, partial [64:128)
    ]
    for shape in cases:
        torch.manual_seed(sum(shape) + 13)
        x = (torch.randn(*shape) * 0.9).to(torch.bfloat16).to(device)
        positions = torch.randperm(200)[: shape[0]].to(torch.int32).to(device)

        fused = csa2.apply_interleaved_rope(x.clone(), positions, cache, inverse=True)
        real_get_op = csa2._get_partial_rotary_op
        monkeypatch.setattr(csa2, "_get_partial_rotary_op", lambda: None)
        eager = csa2.apply_interleaved_rope(x.clone(), positions, cache, inverse=True)
        monkeypatch.setattr(csa2, "_get_partial_rotary_op", real_get_op)

        assert fused.shape == eager.shape
        diff = (fused.float() - eager.float()).abs()
        assert (diff <= 2**-6 + 2**-6 * eager.float().abs()).all(), (
            f"inverse shape={shape} max_abs={diff.max().item():.3e}"
        )
        # The un-rotated prefix must survive the in-place kernel untouched.
        rope_width = cache.size(-1)
        if shape[-1] > rope_width:
            assert torch.equal(fused[..., : shape[-1] - rope_width], eager[..., : shape[-1] - rope_width])


def test_interleaved_rope_falls_back_without_op(monkeypatch: pytest.MonkeyPatch) -> None:
    device = "npu" if _npu_ready() else "cpu"
    dtype = torch.bfloat16 if device == "npu" else torch.float32
    cache = _build_cache(64, 16, device)
    x = torch.randn(6, 2, 32, dtype=dtype, device=device)
    positions = torch.arange(6, dtype=torch.int32, device=device)

    monkeypatch.setattr(csa2, "_get_partial_rotary_op", lambda: None)  # probe miss: stay eager
    eager = csa2.apply_interleaved_rope(x.clone(), positions, cache)
    fallback = csa2.apply_interleaved_rope(x.clone(), positions, cache)
    assert torch.equal(fallback, eager)


# ---------------------------------------------------------------------------
# R6: pre-expanded RoPE table cache (always on)
# ---------------------------------------------------------------------------


def _per_call_expansion(
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The original per-call table preparation (gather, then expand)."""
    half = cos_sin_cache.size(-1) // 2
    rows = cos_sin_cache.index_select(0, positions.reshape(-1).to(torch.int64))
    cos = rows[..., :half].repeat_interleave(2, dim=-1).to(dtype).contiguous()
    sin = rows[..., half:].repeat_interleave(2, dim=-1).to(dtype).contiguous()
    return cos, sin


def test_rope_preexpand_bitwise_matches_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """expand(table)[pos] must equal expand(table[pos]) bit for bit."""
    device = "npu" if _npu_ready() else "cpu"
    for max_pos, rotary_dim in [(256, 64), (4096, 64)]:
        cache = _build_cache(max_pos, rotary_dim, device)
        for dtype in (torch.bfloat16, torch.float32):
            expanded = csa2._expanded_rope_tables(cache, dtype)
            assert expanded is not None
            cos_t, sin_t = expanded
            assert cos_t.shape == (max_pos, rotary_dim)
            assert cos_t.dtype == dtype and sin_t.dtype == dtype
            for seed in (1, 2, 3):
                gen = torch.Generator().manual_seed(seed)
                # Distinct positions, repeated positions (multi-sequence
                # compressor starts), and the boundary rows 0 / max_pos-1.
                positions = torch.cat(
                    [
                        torch.randperm(max_pos, generator=gen)[:9],
                        torch.tensor([0, 0, max_pos - 1, max_pos - 1]),
                    ]
                ).to(device)
                cos_ref, sin_ref = _per_call_expansion(cache, positions, dtype)
                assert torch.equal(cos_t.index_select(0, positions), cos_ref)
                assert torch.equal(sin_t.index_select(0, positions), sin_ref)


def test_rope_preexpand_cache_memoization(monkeypatch: pytest.MonkeyPatch) -> None:
    """One expansion per (table, dtype); capture-in-progress disables; fresh table rebuilds."""
    device = "npu" if _npu_ready() else "cpu"
    cache = _build_cache(128, 64, device)

    # A build attempted under an active capture is refused so the expansion ops
    # are not recorded into the graph.
    monkeypatch.setattr(csa2, "_is_stream_capturing", lambda: True)
    assert csa2._expanded_rope_tables(cache, torch.bfloat16) is None
    monkeypatch.setattr(csa2, "_is_stream_capturing", lambda: False)

    first = csa2._expanded_rope_tables(cache, torch.bfloat16)
    assert first is not None
    # Same (table, dtype): the identical tensor pair comes back.
    again = csa2._expanded_rope_tables(cache, torch.bfloat16)
    assert again[0] is first[0] and again[1] is first[1]
    # A different dtype gets its own entry.
    other = csa2._expanded_rope_tables(cache, torch.float32)
    assert other is not None and other[0] is not first[0]
    # A fresh table tensor builds a fresh expansion.
    cache2 = _build_cache(128, 64, device)
    fresh = csa2._expanded_rope_tables(cache2, torch.bfloat16)
    assert fresh is not None and fresh[0] is not first[0]


@pytest.mark.skipif(not _npu_ready() or not _load_fused_ops_lib(), reason="needs NPU + fused ops lib")
def test_interleaved_rope_preexpand_fused_bitwise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cached path vs per-call expansion path through the same kernel: bitwise."""
    device = "npu"
    cache = _build_cache(4096, 64, device)
    cases = [
        (1, 8, 512),  # decode q B=1
        (16, 8, 512),  # decode q B=16
        (16, 1, 512),  # decode kv / latent B=16
        (16, 32, 128),  # indexer q_idx B=16
        (16, 1, 128),  # index_k B=16
        (5, 128),  # 2D index-k-like
    ]
    for shape in cases:
        for seed in (1, 2):
            torch.manual_seed(sum(shape) * 10 + seed)
            x = (torch.randn(*shape) * 0.9).to(torch.bfloat16).to(device)
            positions = torch.randperm(cache.size(0) - shape[0] - 1)[: shape[0]].to(torch.int32).to(device)

            real_expanded_tables = csa2._expanded_rope_tables
            monkeypatch.setattr(csa2, "_expanded_rope_tables", lambda *a, **k: None)
            per_call = csa2.apply_interleaved_rope(x.clone(), positions, cache)
            monkeypatch.setattr(csa2, "_expanded_rope_tables", real_expanded_tables)
            cached = csa2.apply_interleaved_rope(x.clone(), positions, cache)
            assert torch.equal(cached, per_call), f"shape={shape} seed={seed}"


@pytest.mark.skipif(not _npu_ready() or not _load_fused_ops_lib(), reason="needs NPU + fused ops lib")
def test_interleaved_rope_preexpand_graph_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """NPUGraph capture with the cache, replay on new values: bitwise vs per-call."""
    device = "npu"
    cache = _build_cache(4096, 64, device)
    shape = (16, 8, 512)
    torch.manual_seed(13)
    x = (torch.randn(*shape) * 0.9).to(torch.bfloat16).to(device)
    positions = (torch.arange(16, dtype=torch.int32, device=device) + 100).contiguous()

    # The eager warmups populate the cache outside capture, mirroring the
    # decode graph runner's _CAPTURE_WARMUP_STEPS flow.
    for _ in range(3):
        csa2.apply_interleaved_rope(x.clone(), positions, cache)
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for _ in range(3):
            csa2.apply_interleaved_rope(x, positions, cache)
    torch.npu.synchronize()
    with torch.npu.graph(graph, stream=stream):
        csa2.apply_interleaved_rope(x, positions, cache)

    torch.manual_seed(14)
    x2 = (torch.randn(*shape) * 0.9).to(torch.bfloat16).to(device)
    pos2 = (torch.arange(16, dtype=torch.int32, device=device) + 700).contiguous()
    x.copy_(x2)
    positions.copy_(pos2)
    graph.replay()
    torch.npu.synchronize()

    real_expanded_tables = csa2._expanded_rope_tables
    monkeypatch.setattr(csa2, "_expanded_rope_tables", lambda *a, **k: None)
    expected = csa2.apply_interleaved_rope(x2.clone(), pos2, cache)
    monkeypatch.setattr(csa2, "_expanded_rope_tables", real_expanded_tables)
    assert torch.equal(x, expected)
