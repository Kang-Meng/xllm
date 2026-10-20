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

"""Two-layer coverage for the V4.1 compressor Triton kernels.

Layer 1 (CPU-safe wiring contract): the csa2 pooling path builds the INT32
[5, requests] control metadata and forwards FP32 contiguous projections with a
token-aligned BF16 output, exactly the ``compressor_from_projected`` entry
contract. The kernel module is replaced by a recording fake, mirroring the
vllm-ascend wiring test in ``test_dsa_v41_cp.py``. The static guards that gate
the call site (``_v41_compressor_triton_reason`` / ``_v41_compressor_swa_gate``)
are pure host logic and asserted on CPU as well.

Layer 2 (NPU-only numerical parity): ``compressor_from_projected`` against a
pure-torch reference, adapted from vllm-ascend's ``compressor_ratio2_reference``
to xLLM's split ``kv_state`` / ``score_state`` FP32 planes, the token-aligned
BF16 output form, and the PROTECT_NULL (block-id 0) padding semantics.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import xllm.python.attention.csa2_attention as csa2

RING_ROWS = 32
WIDTH = 64  # compressor width must be a positive power of two


def _npu_ready() -> bool:
    try:
        return torch.npu.is_available()
    except Exception:  # noqa: BLE001 - absent torch_npu exposes no npu attr.
        return False


# ============================================================================
# Layer 1: CPU-safe wiring contract (kernel module faked)
# ============================================================================


def _fake_compressor_module(calls: list) -> SimpleNamespace:
    """Recording stand-in for the triton compressor module."""

    def fake_from_projected(
        kv, scores, kv_state, score_state, meta, out, **launch
    ):
        calls.append(
            {
                "kv": kv,
                "scores": scores,
                "kv_state": kv_state,
                "score_state": score_state,
                "metadata": meta,
                "out": out,
                **launch,
            }
        )
        out.fill_(1.5)
        return out

    module = SimpleNamespace(
        compressor_from_projected=fake_from_projected, _cube_core_num=lambda: 8
    )
    return module


def _pool(
    kv,
    scores,
    kv_state,
    score_state,
    start_pos: int,
    used_len: int,
    block_id: int,
):
    """Thin wrapper so every decline-path call site stays short."""
    return csa2._v41_compressor_pool_triton(
        kv,
        scores,
        kv_state,
        score_state,
        start_pos=torch.tensor(start_pos, dtype=torch.int64),
        used_len=used_len,
        block_id=torch.tensor(block_id, dtype=torch.int64),
    )


def test_pool_wiring_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pool path forwards the documented op contract verbatim."""
    head_dim, q_len = WIDTH, 5
    kv = torch.randn(q_len, head_dim, dtype=torch.float32)
    scores = torch.randn(q_len, head_dim, dtype=torch.float32)
    kv_state = torch.zeros(4, RING_ROWS, head_dim, dtype=torch.float32)
    score_state = torch.zeros_like(kv_state)
    calls: list = []
    fake = _fake_compressor_module(calls)
    monkeypatch.setattr(csa2, "_V41_COMPRESSOR_MODULE", fake)

    out = _pool(kv, scores, kv_state, score_state, 7, q_len, 3)

    assert len(calls) == 1
    got = calls[0]
    # FP32 contiguous [tokens, width] projections, both planes of the ring.
    assert got["kv"].dtype == torch.float32
    assert got["kv"].shape == (q_len, head_dim)
    assert got["scores"].dtype == torch.float32
    assert got["scores"].shape == (q_len, head_dim)
    assert got["kv"].is_contiguous() and got["scores"].is_contiguous()
    assert got["kv_state"] is kv_state and got["score_state"] is score_state
    # INT32 [5, requests] metadata: start / used / out base / in base / block.
    meta = got["metadata"]
    assert meta.dtype == torch.int32 and meta.shape == (5, 1)
    assert meta.is_contiguous()
    assert meta[:, 0].tolist() == [7, q_len, 0, 0, 3]
    # Token-aligned BF16 output, launch bounds from the segment shape.
    assert got["out"].dtype == torch.bfloat16
    assert got["out"].shape == (q_len, head_dim)
    assert got["max_query_len"] == q_len and got["num_cores"] == 8
    # The caller returns the op's own output tensor.
    assert out is not None and out is got["out"]
    assert torch.all(out == torch.tensor(1.5, dtype=torch.bfloat16))


def test_pool_declines_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    """A declining op (exception / no cores / empty rows) yields None."""
    kv = torch.randn(4, WIDTH, dtype=torch.float32)
    scores = torch.randn_like(kv)
    kv_state = torch.zeros(2, RING_ROWS, WIDTH, dtype=torch.float32)
    score_state = torch.zeros_like(kv_state)
    calls: list = []

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    raising = SimpleNamespace(
        compressor_from_projected=_raise, _cube_core_num=lambda: 8
    )
    monkeypatch.setattr(csa2, "_V41_COMPRESSOR_MODULE", raising)
    assert _pool(kv, scores, kv_state, score_state, 0, 4, 1) is None

    zero_cores = SimpleNamespace(
        compressor_from_projected=lambda *a, **k: None, _cube_core_num=lambda: 0
    )
    monkeypatch.setattr(csa2, "_V41_COMPRESSOR_MODULE", zero_cores)
    assert _pool(kv, scores, kv_state, score_state, 0, 4, 1) is None

    # An empty segment (zero projection rows) never reaches the op.
    fake = _fake_compressor_module(calls)
    monkeypatch.setattr(csa2, "_V41_COMPRESSOR_MODULE", fake)
    assert _pool(kv[:0], scores[:0], kv_state, score_state, 0, 0, 1) is None
    assert calls == []


def test_reason_static_guards() -> None:
    """Every declination clause of the static gate is checkable on CPU."""
    ring = torch.zeros(2, RING_ROWS, 8, dtype=torch.float32)
    reason = csa2._v41_compressor_triton_reason
    assert reason(3, ring, ring) == "unsupported ratio 3"
    assert reason(2, None, None) == "no compressor state ring"
    bad_dtype = torch.zeros(2, RING_ROWS, 8, dtype=torch.bfloat16)
    assert reason(2, bad_dtype, bad_dtype) == "ring state not fp32"
    flat = torch.zeros(2, RING_ROWS * 8, dtype=torch.float32)
    assert reason(2, flat, flat) == "ring state not [blocks, rows, dim]"
    short = torch.zeros(2, 16, 8, dtype=torch.float32)
    assert reason(2, short, short) == "ring rows != 32"
    assert reason(2, ring, ring) == "not an NPU device"


def test_swa_gate_static() -> None:
    """The ring-block gate is pure host logic: table presence and count == 1."""
    table = torch.zeros(1, 4, dtype=torch.int64)
    dsa = SimpleNamespace(block_tables=[[table]])
    gate = csa2._v41_compressor_swa_gate
    resolved, why = gate(dsa, 0, 0, ring_blocks=1)
    assert resolved is table and why is None
    why2 = gate(dsa, 0, 0, ring_blocks=2)[1]
    assert why2 == "swa ring blocks per sequence 2 != 1"
    assert gate(dsa, 0, 0)[1] == "swa ring blocks per sequence 0 != 1"
    assert gate(dsa, 9, 0, ring_blocks=1)[1] == "no swa block table"
    empty_dsa = SimpleNamespace(block_tables=[[torch.empty(0)]])
    assert gate(empty_dsa, 0, 0, ring_blocks=1)[1] == "no swa block table"


# ============================================================================
# Layer 2: NPU numerical parity (real triton kernel vs pure-torch reference)
# ============================================================================


def _reference_from_projected(kv, scores, kv_state, score_state, metadata):
    """Pure-torch mirror of the K2 pool and K3 ring-update kernels.

    Adapts vllm-ascend's ``compressor_ratio2_reference`` to the xLLM planes:
    two split FP32 state rings, a token-aligned BF16 output, and PROTECT_NULL
    semantics (block id 0 padding rows neither pool nor write).
    """
    out = torch.zeros_like(kv, dtype=torch.bfloat16)
    starts = metadata[0].tolist()
    used = metadata[1].tolist()
    in_bases = metadata[3].tolist()
    blocks = metadata[4].tolist()
    for b in range(metadata.shape[1]):
        s, u, base, row = starts[b], used[b], in_bases[b], blocks[b]
        if u <= 0 or not 0 < row < kv_state.shape[0]:
            continue
        groups_in_batch = (s + u) // 2 - s // 2
        for lg in range(groups_in_batch):
            gidx = s // 2 + lg
            member_kv, member_score = [], []
            for r in range(2):
                pos = gidx * 2 + r
                off = pos - s
                if off >= 0:
                    member_kv.append(kv[base + off])
                    member_score.append(scores[base + off])
                else:
                    slot = pos % RING_ROWS
                    member_kv.append(kv_state[row, slot])
                    member_score.append(score_state[row, slot])
            kk = torch.stack(member_kv).float()
            ss = torch.stack(member_score).float()
            pooled = (kk * ss.softmax(dim=0)).sum(dim=0)
            out_row = base + (gidx + 1) * 2 - 1 - s
            out[out_row] = pooled.to(torch.bfloat16)
        tail = min(u, RING_ROWS)
        for w in range(tail):
            seg_row = u - tail + w
            slot = (s + seg_row) % RING_ROWS
            kv_state[row, slot] = kv[base + seg_row]
            score_state[row, slot] = scores[base + seg_row]
    return out


def _run_parity_case(starts, used, blocks, *, seed: int, width: int = WIDTH):
    """Drive one kernel-vs-reference parity batch on the NPU."""
    torch.manual_seed(seed)
    requests = len(starts)
    tokens = sum(used)
    dev = torch.npu.current_device()
    kv = torch.randn(max(tokens, 1), width, dtype=torch.float32, device=dev)
    scores = torch.randn(max(tokens, 1), width, dtype=torch.float32, device=dev)
    kv, scores = kv[:tokens], scores[:tokens]
    in_bases = []
    base = 0
    for u in used:
        in_bases.append(base)
        base += u
    metadata = torch.tensor(
        [
            starts,
            used,
            [0] * requests,
            in_bases,
            blocks,
        ],
        dtype=torch.int32,
        device=dev,
    )
    num_state_blocks = max(max(blocks) + 1, 1)
    kv_state = torch.randn(num_state_blocks, RING_ROWS, width, device=dev)
    score_state = torch.randn(num_state_blocks, RING_ROWS, width, device=dev)
    ref_kv_state = kv_state.clone()
    ref_score_state = score_state.clone()

    ref_out = _reference_from_projected(
        kv, scores, ref_kv_state, ref_score_state, metadata
    )

    from xllm.python.kernels_npu.triton import compressor

    out = torch.zeros(tokens, width, dtype=torch.bfloat16, device=dev)
    got = compressor.compressor_from_projected(
        kv.contiguous(),
        scores.contiguous(),
        kv_state,
        score_state,
        metadata.contiguous(),
        out,
        max_query_len=max(used),
        num_cores=8,
    )
    assert got is out
    torch.testing.assert_close(
        out.float(), ref_out.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(kv_state, ref_kv_state)
    torch.testing.assert_close(score_state, ref_score_state)


@pytest.mark.skipif(not _npu_ready(), reason="NPU is not available")
def test_parity_multi_request_random() -> None:
    """Random multi-request batches: odd/even starts, mixed lengths."""
    for seed, starts, used in (
        (0, [0, 3, 29, 64, 97], [32, 5, 8, 17, 1]),
        (1, [11, 40, 7, 100, 61], [9, 48, 3, 2, 33]),
        (2, [5, 16, 71, 33, 90], [40, 6, 12, 31, 4]),
    ):
        _run_parity_case(starts, used, [1, 2, 3, 4, 5], seed=seed)


@pytest.mark.skipif(not _npu_ready(), reason="NPU is not available")
def test_parity_ring_boundaries_and_padding() -> None:
    """Exactly-full ring, wrap-around, residual group, empty and padded rows."""
    # Full ring (start 0, 32 rows), wrap-around (start 29), residual group
    # (odd start reads the ring before writing), empty request, block-id-0
    # padding row, and a plain even-start segment, all in one batch.
    starts = [0, 29, 3, 65, 0, 12]
    used = [32, 8, 4, 7, 0, 5]
    blocks = [1, 2, 3, 4, 5, 0]
    _run_parity_case(starts, used, blocks, seed=3)
