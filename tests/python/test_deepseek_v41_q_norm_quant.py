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

"""qr_scale plumbing contract for the V4.1 indexer query.

The fused q-latent norm+quant candidate behind the field was removed; the
remaining tests pin the ``qr_scale`` carrier contract end to end: the
attention seam's ``indexer_query`` single-arg path, the ``Csa2LayerContext``
field default, and the CSA2 backend's two-arg / single-arg dispatch in
``_run_indexer``.
"""

from __future__ import annotations

import types

import pytest
import torch

import xllm.python.attention.csa2_attention as csa2
import xllm.python.models.deepseek_v41 as v41
from xllm.python.attention.csa2_attention import Csa2LayerContext

_EPS = 1e-6


# --------------------------------------------------------------------------- #
# stubs
# --------------------------------------------------------------------------- #
class _FakeNorm(torch.nn.Module):
    """Deterministic RMSNorm (bf16 weight, fp32 stats) with a call counter."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        xf = x.float()
        var = xf.square().mean(-1, keepdim=True)
        return (xf * torch.rsqrt(var + _EPS) * self.weight.float()).to(x.dtype)


class _FakeW8A8:
    """Records the eager (bf16) and pre-quantized (codes, scale) calls."""

    def __init__(self, out_features: int, in_features: int, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(out_features, in_features, generator=g, dtype=torch.float32)
        self.forward_calls = 0
        self.forward_quantized_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.last_codes: torch.Tensor | None = None
        self.last_scale: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return (x.float() @ self.W.t()).to(torch.bfloat16)

    __call__ = forward  # the real W8A8DynamicLinear is an nn.Module

    def forward_quantized(self, codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        self.last_codes = codes
        self.last_scale = scale
        self.forward_quantized_calls.append((codes, scale))
        return ((codes.float() * scale.view(-1, 1)) @ self.W.t()).to(torch.bfloat16)


def _make_attn(*, quant_mode: str = v41.V41_QUANT_ASCEND, indexer: bool = True):
    attn = object.__new__(v41.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)  # enable nn.Module attribute assignment
    attn.quant_mode = quant_mode
    attn.cfg = types.SimpleNamespace(rms_norm_eps=_EPS)
    attn.q_a_layernorm = _FakeNorm(8)
    attn.q_b_proj = _FakeW8A8(6, 8, seed=1)
    attn.indexer = types.SimpleNamespace(wq_b=_FakeW8A8(4, 8, seed=2), n_heads=2, head_dim=2) if indexer else None
    return attn


# --------------------------------------------------------------------------- #
# indexer_query compatibility / ctx field
# --------------------------------------------------------------------------- #
def test_indexer_query_single_arg_path_still_works() -> None:
    attn = _make_attn()
    qr = torch.randn(5, 8, dtype=torch.bfloat16)

    q_idx = attn.indexer_query(qr)

    assert attn.indexer.wq_b.forward_calls == 1
    assert attn.indexer.wq_b.forward_quantized_calls == []
    assert q_idx.shape == (5, 2, 2)


def test_csa2_layer_context_carries_qr_scale_defaulting_to_none() -> None:
    cache = torch.zeros(4, 4, dtype=torch.bfloat16)
    ctx = Csa2LayerContext(
        x=torch.zeros(2, 3, dtype=torch.bfloat16),
        qr=torch.zeros(2, 3, dtype=torch.bfloat16),
        positions=torch.arange(2),
        cos_sin_cache=cache,
    )
    assert ctx.qr_scale is None

    scale = torch.ones(2, dtype=torch.float32)
    ctx2 = Csa2LayerContext(
        x=torch.zeros(2, 3, dtype=torch.bfloat16),
        qr=torch.zeros(2, 3, dtype=torch.int8),
        positions=torch.arange(2),
        cos_sin_cache=cache,
        qr_scale=scale,
    )
    assert ctx2.qr_scale is scale


# --------------------------------------------------------------------------- #
# _run_indexer q dispatch (P2-1)
# --------------------------------------------------------------------------- #
_QLI_SENTINEL = object()


def _drive_run_indexer(monkeypatch: pytest.MonkeyPatch, *, scale: torch.Tensor | None):
    """Call ``_run_indexer`` up to the QLI branch, recording ``indexer_query``.

    Every stage before the query call is stubbed so the test isolates the
    ``ctx.qr_scale`` dispatch; the QLI branch returns a sentinel immediately,
    skipping the eager scoring pass that needs a full cache runtime.
    """
    backend = object.__new__(csa2.Csa2AttentionBackend)
    monkeypatch.setattr(backend, "_committed_capacity", lambda *a, **k: 4)
    monkeypatch.setattr(backend, "_qli_v3_fallback_reason", lambda *a, **k: None)
    monkeypatch.setattr(backend, "_run_indexer_qli_v3", lambda *a, **k: _QLI_SENTINEL)
    monkeypatch.setattr(csa2, "apply_interleaved_rope", lambda q, *a, **k: q)
    monkeypatch.setattr(csa2, "_log_qli_v3_active", lambda: None)

    calls: list[tuple[torch.Tensor, ...]] = []
    q_idx = torch.zeros(2, 2, 2, dtype=torch.bfloat16)

    def indexer_query(*args: torch.Tensor) -> torch.Tensor:
        calls.append(args)
        return q_idx

    layer = types.SimpleNamespace(layer_id=0, indexer_query=indexer_query)
    qr = torch.zeros(2, 8, dtype=torch.int8 if scale is not None else torch.bfloat16)
    ctx = Csa2LayerContext(
        x=torch.zeros(2, 4, dtype=torch.bfloat16),
        qr=qr,
        positions=torch.arange(2),
        cos_sin_cache=torch.zeros(4, 4, dtype=torch.bfloat16),
        qr_scale=scale,
    )
    shared = types.SimpleNamespace(
        compress_ratio=2,
        index_cache=torch.zeros(1, 8, 8, dtype=torch.int8),
        index_block_table=torch.zeros(1, 1, dtype=torch.int64),
    )
    out = backend._run_indexer(layer, ctx, types.SimpleNamespace(), [2], torch.tensor([2]), shared)
    return calls, out, qr


def test_run_indexer_passes_qr_scale_two_arg_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """qr_scale present => indexer_query(ctx.qr, ctx.qr_scale), not single-arg."""
    scale = torch.rand(2, dtype=torch.float32) + 0.01
    calls, out, qr = _drive_run_indexer(monkeypatch, scale=scale)

    assert len(calls) == 1
    args = calls[0]
    assert len(args) == 2, "single-arg path must not be taken when qr_scale is set"
    assert args[0] is qr
    assert args[1] is scale
    assert out is _QLI_SENTINEL


def test_run_indexer_uses_single_arg_when_scale_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """qr_scale None (the off/default shape) => the legacy single-arg call."""
    calls, out, qr = _drive_run_indexer(monkeypatch, scale=None)

    assert len(calls) == 1
    args = calls[0]
    assert len(args) == 1, "two-arg path must not be taken without qr_scale"
    assert args[0] is qr
    assert out is _QLI_SENTINEL
