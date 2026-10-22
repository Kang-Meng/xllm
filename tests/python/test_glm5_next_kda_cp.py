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

"""KDA context-parallel prefill tests (enable_kda_cp, task M2 + MA2 + N1 + N3-5).

Three layers of coverage:

* PURE (CPU, always run): the shared gate predicate (segment alignment,
  exclusions, rank-uniformity, the xllm CP ops availability term), the
  zigzag causal order / causally-last owner computation, the causal-order
  fp32 merge composition against an independent math-space reference
  (warm prefix base, empty-span [0|I] identity, bounded >=2-prior
  re-association divergence), the fold implementations (the sequential
  reference, the Hillis-Steele tree scan, the merge-op adapter) against
  each other with measured re-association bounds, the shape-driven fold
  selection, the host-plan tensor cache, and the
  ``kernels_npu.kda_cp_ops`` wrapper's availability gating (the standard
  opp vendor path) plus its ported argument validation.
* ORCHESTRATION (CPU, mocked kernels): the model forward's branch selection
  (merge skipped / kept) and the full backend CP pipeline through the real
  ``execute_linear`` with reference kernels whose affine recurrence the
  mock summary op composes exactly -- CP output, cache writes, the
  final-state broadcast replica invariant and the checkpoint column
  isolation are all checked against a full-stream reference run, at
  head_dim 8 (tree-scan fold) and head_dim 128 (merge fold).
* NPU (gated on ``XLLM_KDA_TEST_NPU_DEVICE``): the same pipeline shape with
  the REAL deployed ``fla_npu`` fused kernels plus OUR standard
  ``custom_xllm_math`` opp vendor (resolved from
  ``${ASCEND_HOME_PATH}/opp/vendors/custom_xllm_math``), comparing the
  two-pass output against a full-stream single call at toy shapes, the
  ma1 factor-equivalence contract of our prepare op vs the fused
  ``disable_recompute`` emission, the M-B2 identity gate (ONE batched
  prepare + ppfm over a multi-segment stream vs the per-segment calls --
  bit-identical or the batching does not ship), and the merge-op fold vs
  the python fold on real summaries (the state_v_first=True device pin).
  Skips gracefully when either op source is unavailable in the test
  environment.
"""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.distributed
import torch.nn.functional as functional
from xllm.python.kernels import l2_norm

from xllm.python.attention import kda_linear_attention as kda_module
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.kda_linear_attention import (
    KDA_CP_CHUNK_SIZE,
    KdaLinearAttentionMixin,
    _kda_cp_pack_state_broadcast,
    _kda_cp_unpack_state_broadcast,
    _kda_cp_zero_prior_segments,
    kda_cp_causal_summary_rows,
    kda_cp_chunk_owner,
    kda_cp_compose_initial_state,
    kda_cp_merge_fold_initial_states,
    kda_cp_prefill_eligible,
    kda_cp_segments_chunk_aligned,
    kda_cp_tree_scan_fold_initial_states,
)
from xllm.python.kernels_npu import kda_cp_ops
from xllm.python.model_executor import cp_utils
from xllm.python.model_executor.cp_utils import CpContext
from xllm.python.models import glm5_next

# Module-level default device: a torch.device(...) call inside a def default
# trips ruff B008.
_CPU_DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Zigzag layout mirror (python twin of xllm_ops::build_cp_context's host math)
# ---------------------------------------------------------------------------


def _zigzag_layout(q_seq_lens: list[int], cp_size: int, cp_rank: int) -> dict[str, Any]:
    """Mirror of the C++ zigzag index math (npu_ops_library.cpp)."""
    num_chunks = 2 * cp_size
    first_chunk = cp_rank
    second_chunk = num_chunks - 1 - cp_rank
    shard_index: list[int] = []
    query_index: list[int] = []
    q_cu_seqlens: list[int] = []
    chunk_lens: list[int] = []
    global_offset = 0
    local_offset = 0
    q_run = 0
    for length in q_seq_lens:
        padded = ((length + num_chunks - 1) // num_chunks) * num_chunks
        chunk_len = padded // num_chunks
        chunk_lens.append(chunk_len)
        for half, chunk_id in ((0, first_chunk), (1, second_chunk)):
            seg_local_base = local_offset + half * chunk_len
            seg_start = chunk_id * chunk_len
            real = 0
            for j in range(chunk_len):
                pos_in_seq = seg_start + j
                if pos_in_seq < length:
                    shard_index.append(global_offset + pos_in_seq)
                    query_index.append(seg_local_base + j)
                    real += 1
                else:
                    shard_index.append(-1)
            if real > 0:
                q_run += real
                q_cu_seqlens.append(q_run)
        global_offset += length
        local_offset += 2 * chunk_len
    return {
        "shard_index": shard_index,
        "query_index": query_index,
        "q_cu_seqlens": q_cu_seqlens,
        "total_local": local_offset,
        "chunk_lens": chunk_lens,
    }


def _toy_cp_context(q_seq_lens: list[int], cp_size: int, cp_rank: int, device: torch.device) -> CpContext:
    """Build a real CpContext from the mirror.

    The fields the KDA CP path reads (cp_size/cp_rank/total_local/
    query_index/q_cu_seqlens) are exact; the FIA-only fields get
    shape-compatible stand-ins (they are never consumed on this path).
    """
    layout = _zigzag_layout(q_seq_lens, cp_size, cp_rank)
    shard_index = torch.tensor(layout["shard_index"], dtype=torch.int64, device=device)
    valid_mask = shard_index >= 0
    gather_index = torch.where(valid_mask, shard_index, torch.zeros_like(shard_index))
    query_index = torch.tensor(layout["query_index"], dtype=torch.int64, device=device)
    segment_count = len(layout["q_cu_seqlens"])
    total_global = sum(q_seq_lens)
    return CpContext(
        cp_size=cp_size,
        cp_rank=cp_rank,
        total_local=layout["total_local"],
        shard_index=shard_index,
        shard_gather_index=gather_index,
        shard_valid_mask=valid_mask,
        restore_index=torch.zeros(max(total_global, 1), dtype=torch.int64, device=device),
        query_index=query_index,
        q_cu_seqlens=layout["q_cu_seqlens"],
        q_cu_seqlens_tensor=torch.tensor(layout["q_cu_seqlens"], dtype=torch.int32, device=device),
        kv_gather_index=torch.zeros(0, dtype=torch.int64, device=device),
        kv_cu_seqlens=[],
        segment_seq_indices=torch.zeros(segment_count, dtype=torch.int64, device=device),
        segment_kv_seq_lens=[],
        segment_kv_seq_lens_tensor=torch.zeros(0, dtype=torch.int32, device=device),
        has_prefix=False,
    )


def _gate_metadata(q_seq_lens: list[int], device: torch.device = _CPU_DEVICE, **overrides: Any) -> SimpleNamespace:
    """Cold prefill metadata with the host forms the gate reads."""
    num_seqs = len(q_seq_lens)
    cumulative = [0]
    for length in q_seq_lens:
        cumulative.append(cumulative[-1] + length)
    metadata = SimpleNamespace(
        linear_state_indices=torch.arange(num_seqs, dtype=torch.int32, device=device),
        linear_state_read_indices=torch.arange(num_seqs, dtype=torch.int32, device=device),
        linear_state_write_indices=torch.arange(num_seqs, dtype=torch.int32, device=device),
        has_initial_state=torch.zeros(num_seqs, dtype=torch.int64, device=device),
        is_prefill=True,
        is_chunked_prefill=False,
        is_spec_verify=False,
        is_mixed=False,
        q_cu_seq_lens=torch.tensor(cumulative, dtype=torch.int32),
        q_seq_lens_host=None,
        q_cu_seq_lens_host_values=None,
        expanded_decode_metadata=None,
    )
    for key, value in overrides.items():
        setattr(metadata, key, value)
    return metadata


def _gate_layer(activation: str = "silu", lower_bound: float | None = -5.0) -> SimpleNamespace:
    return SimpleNamespace(
        activation=activation,
        forget_gate=SimpleNamespace(safe_gate_lower_bound=lower_bound),
    )


def _eligible(q_seq_lens: list[int], cp_size: int, **metadata_overrides: Any) -> bool:
    """Gate shorthand: cp_size>1, silu layer, patched summary-op probe."""
    context = SimpleNamespace(cp_size=cp_size, cp_rank=0)
    return kda_cp_prefill_eligible(context, _gate_metadata(q_seq_lens, **metadata_overrides), _gate_layer())


# ---------------------------------------------------------------------------
# Pure tests: segment alignment gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "length,cp_size,expected",
    [
        # Whole-chunk multiples at every practical cp size (chunked-prefill
        # stride 2048 at cp4 -> 4x512, at cp8 -> 2x1024, at cp2 -> 8x256).
        (512, 4, True),
        (1024, 4, True),
        (2048, 4, True),
        (2048, 8, True),
        (2048, 2, True),
        (256, 2, True),
        (4096, 8, True),
        # Unaligned chunk_len (ceil(L/(2cp)) not a multiple of 64).
        (960, 4, False),
        (1984, 4, False),
        (250, 2, False),
        (2047, 8, False),
        # L itself unaligned even when chunk_len rounds up to a 64 multiple.
        (576, 4, False),
        # Short segments: fewer than one whole chunk.
        (8, 2, False),
        (100, 4, False),
        # Empty trailing chunk (needs cp large enough that 2cp-1 chunks of
        # ceil(L/2cp) swallow L): the naive "L % 64 == 0 and c % 64 == 0"
        # shortcut would wrongly accept this.
        (8128, 64, False),
    ],
)
def test_kda_cp_segments_chunk_aligned_boundary_cases(length: int, cp_size: int, expected: bool) -> None:
    assert kda_cp_segments_chunk_aligned([length], cp_size) is expected


def test_kda_cp_segments_chunk_aligned_multi_sequence_batch() -> None:
    assert kda_cp_segments_chunk_aligned([2048, 512], 4) is True
    # One unaligned sequence fails the WHOLE batch (all-rank legacy fallback).
    assert kda_cp_segments_chunk_aligned([2048, 960], 4) is False
    assert kda_cp_segments_chunk_aligned([0, 2048], 4) is False


def test_kda_cp_segments_chunk_aligned_requires_cp_gt_one() -> None:
    assert kda_cp_segments_chunk_aligned([2048], 1) is False
    assert kda_cp_segments_chunk_aligned([], 4) is True


# ---------------------------------------------------------------------------
# Pure tests: the shared gate's exclusions and metadata forms
# ---------------------------------------------------------------------------


def test_kda_cp_gate_exclusions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kda_module, "_kda_cp_ops", lambda: (object(), object(), object()))
    assert _eligible([2048], 4) is True
    # Decode-shaped steps stay legacy; chunked prefill is prefill-shaped.
    assert _eligible([2048], 4, is_prefill=False) is False
    assert _eligible([2048], 4, is_prefill=False, is_chunked_prefill=True) is True
    # Spec-verify and mixed batches never take the CP branch.
    assert _eligible([2048], 4, is_spec_verify=True) is False
    assert _eligible([2048], 4, is_mixed=True) is False
    # Unaligned segments fall back on every rank together.
    assert _eligible([960], 4) is False
    # cp_size == 1 has no CP at all.
    assert _eligible([2048], 1) is False


def test_kda_cp_gate_layer_and_op_requirements(monkeypatch: pytest.MonkeyPatch) -> None:
    # Non-SiLU conv activation: causal_conv1d_prefill hardcodes SiLU.
    monkeypatch.setattr(kda_module, "_kda_cp_ops", lambda: (object(), object(), object()))
    context = SimpleNamespace(cp_size=4, cp_rank=0)
    metadata = _gate_metadata([2048])
    assert kda_cp_prefill_eligible(context, metadata, _gate_layer(activation="identity")) is False
    # Safe-gate lower bound outside the fused range.
    assert kda_cp_prefill_eligible(context, metadata, _gate_layer(lower_bound=None)) is False
    assert kda_cp_prefill_eligible(context, metadata, _gate_layer(lower_bound=-6.0)) is False
    # Missing CP ops (vendor env unset, library or symbols missing).
    monkeypatch.setattr(kda_module, "_kda_cp_ops", lambda: None)
    assert kda_cp_prefill_eligible(context, metadata, _gate_layer()) is False


def test_kda_cp_gate_falls_back_without_standard_vendor(monkeypatch: pytest.MonkeyPatch) -> None:
    """No CANN root -> kda_cp_ops.available() False -> the REAL probe resolves
    None -> the gate fails on every rank -> legacy merged path (the safe
    fallback; also pins that the probe itself never raises)."""
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.setattr(kda_module, "_kda_cp_ops", kda_module._kda_cp_ops.__wrapped__)
    assert kda_cp_ops.available() is False
    assert kda_module._kda_cp_ops() is None
    assert _eligible([2048], 4) is False


def test_kda_cp_ops_probe_pin_and_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """The op probe pins its answer per process BY DESIGN, and the standard
    ``functools.cache`` reset hook re-probes after an in-process fix.

    Pinning is what keeps the gate rank-uniform (a live flip between two ranks
    would let one enter a collective the other skips); the reset hook is the
    documented escape hatch for tests and diagnostics. Together they answer the
    "cached negative result is a permanent dead end" concern without weakening
    the rank-uniformity guarantee.
    """
    probe = kda_module._kda_cp_ops
    probe.cache_clear()
    try:
        monkeypatch.setattr(kda_cp_ops, "available", lambda: False)
        assert probe() is None
        # A mid-process availability flip must NOT be observed on the next call.
        monkeypatch.setattr(kda_cp_ops, "available", lambda: True)
        assert probe() is None
        # The documented reset hook re-probes and now resolves the op set.
        probe.cache_clear()
        ops = probe()
        assert ops is not None and len(ops) == 3
    finally:
        probe.cache_clear()


def test_kda_cp_ops_standard_vendor_path_derivation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The standard opp vendor path is derived from the CANN runtime root:
    ${ASCEND_HOME_PATH}/opp/vendors/custom_xllm_math/op_api/lib/..."""
    monkeypatch.setenv("ASCEND_HOME_PATH", "/fake/ascend-home")
    assert kda_cp_ops._opapi_library_paths() == [
        os.path.join("/fake/ascend-home", "opp", "vendors", "custom_xllm_math", "op_api", "lib", "libcust_opapi.so")
    ]
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    assert kda_cp_ops._opapi_library_paths() == []


def test_kda_cp_ops_availability_gating(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """available() is a pure capability probe: False for an unset CANN root,
    a missing root or a root without the vendor library layout -- never
    an import-time raise."""
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    assert kda_cp_ops.available() is False
    monkeypatch.setenv("ASCEND_HOME_PATH", "/nonexistent/ascend-home")
    assert kda_cp_ops.available() is False
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))  # no opp/vendors below
    assert kda_cp_ops.available() is False


def _prepare_wrapper_inputs(dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, ...]:
    """Minimal valid CPU inputs for the prepare wrapper's validation
    (BSND, HV=2, K=V=128, T=128; the wrapper validates shapes/dtypes/flags
    before any NPU access, so CPU tensors are fine)."""
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(1, 128, 2, 128, dtype=dtype, generator=generator)
    k = torch.randn(1, 128, 2, 128, dtype=dtype, generator=generator)
    v = torch.randn(1, 128, 2, 128, dtype=dtype, generator=generator)
    g = torch.randn(1, 128, 2, 128, dtype=dtype, generator=generator)
    beta = torch.rand(1, 128, 2, dtype=torch.float32, generator=generator)
    return q, k, v, g, beta


def _prepare_wrapper_kwargs() -> dict[str, Any]:
    a_log = torch.zeros(2, dtype=torch.float32)
    dt_bias = torch.zeros(2 * 128, dtype=torch.float32)
    return dict(
        chunk_size=64,
        layout="BSND",
        cu_seqlens=[0, 128],
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=True,
        A_log=a_log,
        dt_bias=dt_bias,
        epsilon=1e-6,
        use_exp2=True,
        backward_mode="none",
    )


def test_kda_cp_prepare_wrapper_arg_validation() -> None:
    """The ported validation fires with the source wrapper's contract before
    any NPU access (CPU-runnable)."""
    q, k, v, g, beta = _prepare_wrapper_inputs()
    kwargs = _prepare_wrapper_kwargs()
    cases: list[tuple[dict[str, Any], str]] = [
        ({"layout": "SBND"}, "layout"),
        ({"chunk_size": 96}, "chunk_size"),
        ({"backward_mode": "bogus"}, "backward_mode"),
        ({"A_log": torch.zeros(3, dtype=torch.float32)}, "A_log"),
        ({"dt_bias": torch.zeros(5, dtype=torch.float32)}, "dt_bias"),
        ({"epsilon": 0.0}, "epsilon"),
        ({"lower_bound": -6.0}, "lower_bound"),
        ({"cu_seqlens": [1, 128]}, "cu_seqlens"),
        ({"cu_seqlens": [0, 64]}, "cu_seqlens"),
        ({"chunk_indices": [0, 1, 0, 0]}, "chunk_indices"),
    ]
    for override, fragment in cases:
        with pytest.raises(RuntimeError, match=fragment):
            kda_cp_ops.xllm_chunk_kda_fwd_prepare(q, k, v, g, beta, 128**-0.5, **{**kwargs, **override})
    # q/k shape mismatch and the fp32 q/k/v dtype contract.
    with pytest.raises(RuntimeError, match="q/k shapes"):
        kda_cp_ops.xllm_chunk_kda_fwd_prepare(q, k[:, :64], v, g, beta, 128**-0.5, **kwargs)
    with pytest.raises(RuntimeError, match="q/k/v must use"):
        kda_cp_ops.xllm_chunk_kda_fwd_prepare(q.float(), k.float(), v.float(), g, beta, 128**-0.5, **kwargs)
    # Rank-4 BSND varlen input requires B=1.
    no_cu = {key: value for key, value in kwargs.items() if key != "cu_seqlens"}
    with pytest.raises(RuntimeError, match="B=1"):
        kda_cp_ops.xllm_chunk_kda_fwd_prepare(
            torch.cat([q, q], 0),
            torch.cat([k, k], 0),
            torch.cat([v, v], 0),
            torch.cat([g, g], 0),
            torch.cat([beta, beta], 0),
            128**-0.5,
            cu_seqlens=[0, 128],
            **no_cu,
        )
    # K/V dims must both be 64 or both 128.
    small_q = torch.randn(1, 128, 2, 96, dtype=torch.bfloat16)
    small_k = torch.randn(1, 128, 2, 96, dtype=torch.bfloat16)
    small_v = torch.randn(1, 128, 2, 96, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="K/V must both be"):
        kda_cp_ops.xllm_chunk_kda_fwd_prepare(small_q, small_k, small_v, g, beta, 96**-0.5, **kwargs)


def test_kda_cp_ppfm_wrapper_arg_validation() -> None:
    """The ppfm port's validation: cu_seqlens is mandatory varlen host
    input, exactly one of g/gk, B=1, strictly increasing bounds. The
    factors use the op's head-major layout [B, H, T, D]."""
    k_head_major = torch.randn(1, 2, 128, 128)
    w_head_major = torch.randn(1, 2, 128, 128)
    u_head_major = torch.randn(1, 2, 128, 128)
    gk_gate = torch.randn(1, 128, 2, 128)
    with pytest.raises(ValueError, match="cu_seqlens"):
        kda_cp_ops.xllm_pre_process_fwd_kernel_merged(k_head_major, w_head_major, u_head_major, chunk_size=64)
    with pytest.raises(ValueError, match="exactly one of g / gk"):
        kda_cp_ops.xllm_pre_process_fwd_kernel_merged(
            k_head_major, w_head_major, u_head_major, cu_seqlens=[0, 128], chunk_size=64
        )
    with pytest.raises(ValueError, match="exactly one of g / gk"):
        kda_cp_ops.xllm_pre_process_fwd_kernel_merged(
            k_head_major, w_head_major, u_head_major, g=gk_gate, gk=gk_gate, cu_seqlens=[0, 128], chunk_size=64
        )
    with pytest.raises(ValueError, match="B = 1"):
        kda_cp_ops.xllm_pre_process_fwd_kernel_merged(
            torch.randn(2, 2, 128, 128),
            w_head_major,
            u_head_major,
            gk=gk_gate,
            cu_seqlens=[0, 128],
            chunk_size=64,
        )
    with pytest.raises(ValueError, match="strictly"):
        kda_cp_ops.xllm_pre_process_fwd_kernel_merged(
            k_head_major, w_head_major, u_head_major, gk=gk_gate, cu_seqlens=[128, 128], chunk_size=64
        )


def test_kda_cp_ops_raise_clear_error_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid args on a machine without the vendor raise the launcher-shaped
    unavailability error (not a bare dlopen/AttributeError)."""
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.setattr(kda_cp_ops, "_RUNTIME", None)
    q, k, v, g, beta = _prepare_wrapper_inputs()
    with pytest.raises(RuntimeError, match="custom_xllm_math"):
        kda_cp_ops.xllm_chunk_kda_fwd_prepare(q, k, v, g, beta, 128**-0.5, **_prepare_wrapper_kwargs())


def test_kda_cp_gate_reads_every_host_metadata_form(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kda_module, "_kda_cp_ops", lambda: (object(), object(), object()))
    context = SimpleNamespace(cp_size=4, cp_rank=0)
    layer = _gate_layer()
    forms = [
        _gate_metadata([2048]),
        _gate_metadata([], q_seq_lens_host=torch.tensor([2048], dtype=torch.int64)),
        _gate_metadata([], q_cu_seq_lens_host_values=[2048]),
        _gate_metadata([], q_cu_seq_lens_host_values=[0, 2048]),
        _gate_metadata([], q_cu_seq_lens=torch.tensor([0, 2048], dtype=torch.int32)),
    ]
    for metadata in forms:
        assert kda_cp_prefill_eligible(context, metadata, layer) is True, metadata
    # No host form at all (device-only metadata): safe legacy fallback.
    device_only = _gate_metadata([], q_cu_seq_lens=torch.tensor([0, 2048], dtype=torch.int32, device="meta"))
    assert kda_cp_prefill_eligible(context, device_only, layer) is False


def test_kda_cp_gate_is_rank_uniform_across_zigzag_views() -> None:
    """The gate consumes only global batch state, so every rank agrees even
    when the per-rank local segment views differ (padding in the last chunk
    makes rank 0's second segment shorter than its peers')."""
    for q_seq_lens, cp_size, expected in (
        ([2048], 4, True),
        ([1993], 4, False),
        ([1094], 4, False),
        ([513], 2, False),
    ):
        answers = set()
        local_views = []
        for cp_rank in range(cp_size):
            layout = _zigzag_layout(q_seq_lens, cp_size, cp_rank)
            answers.add(kda_cp_segments_chunk_aligned(q_seq_lens, cp_size))
            cu = layout["q_cu_seqlens"]
            local_views.append(tuple([cu[0], *(cu[i] - cu[i - 1] for i in range(1, len(cu)))]))
        assert answers == {expected}
        # The 1993-token case really does produce rank-dependent local views;
        # a gate reading them would desynchronize the group.
        if q_seq_lens == [1993]:
            assert len(set(local_views)) > 1


# ---------------------------------------------------------------------------
# Pure tests: zigzag causal order and the causally-last owner
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_kda_cp_chunk_owner_is_a_permutation_with_balanced_pairs(cp_size: int) -> None:
    owners = [kda_cp_chunk_owner(chunk, cp_size) for chunk in range(2 * cp_size)]
    # Every chunk owned exactly once.
    assert len(set(owners)) == 2 * cp_size
    # Rank r owns chunk r (first half) and chunk 2*cp-1-r (second half).
    for rank in range(cp_size):
        assert owners[rank] == (rank, 0)
        assert owners[2 * cp_size - 1 - rank] == (rank, 1)
    # The causally-LAST chunk belongs to rank 0's SECOND segment at every
    # cp size -- the final-state broadcast root (NOT rank cp-1). This is why
    # the half-0 form of the broadcast pack keeps its symmetry without firing
    # today (see kda_cp_execute_linear's payload pick).
    assert kda_cp_chunk_owner(2 * cp_size - 1, cp_size) == (0, 1)
    # Out-of-range chunk ids are rejected, not silently wrapped.
    with pytest.raises(ValueError):
        kda_cp_chunk_owner(2 * cp_size, cp_size)


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_kda_cp_causal_summary_rows_form_a_permutation(cp_size: int) -> None:
    """The gathered-summary row of each causal chunk (rank*2 + half) must be
    a permutation of [0, 2*cp): a collision would silently compose the wrong
    segment into some rank's initial state."""
    rows = []
    for chunk in range(2 * cp_size):
        owner_rank, owner_half = kda_cp_chunk_owner(chunk, cp_size)
        rows.append(owner_rank * 2 + owner_half)
    assert sorted(rows) == list(range(2 * cp_size))


# ---------------------------------------------------------------------------
# Pure tests: causal-order merge composition (the fp32 fold)
# ---------------------------------------------------------------------------


def _random_summaries(
    n_seqs: int, num_chunks: int, hv: int, k_dim: int, v_dim: int, generator: torch.Generator
) -> torch.Tensor:
    """Random gathered-summaries stack [2*cp, n_seqs, HV, K, V+K] with
    well-conditioned transitions (diagonal-dominant M, mild E)."""
    cp_size = num_chunks // 2
    summaries = torch.zeros(2 * cp_size, n_seqs, hv, k_dim, v_dim + k_dim)
    for chunk in range(num_chunks):
        owner_rank, owner_half = kda_cp_chunk_owner(chunk, cp_size)
        emission = torch.randn(n_seqs, hv, k_dim, v_dim, generator=generator) * 0.05
        transition = (
            torch.eye(k_dim).view(1, 1, k_dim, k_dim).expand(n_seqs, hv, k_dim, k_dim).clone()
            + torch.randn(n_seqs, hv, k_dim, k_dim, generator=generator) * 0.02
        )
        summaries[owner_rank * 2 + owner_half, ..., :v_dim] = emission
        summaries[owner_rank * 2 + owner_half, ..., v_dim:] = transition
    return summaries


def _math_space_reference(base_state: torch.Tensor, summaries: torch.Tensor, prior_chunks: int) -> torch.Tensor:
    """Independent math-space composition: S <- M_math @ S + E_math applied
    per chunk in causal order, with the stored<->math orientations transposed
    explicitly. Any orientation mistake in the fold shows up here."""
    cp_size = summaries.shape[0] // 2
    v_dim = base_state.shape[-2]
    state = base_state.transpose(-1, -2).clone()  # [n, HV, K, V] math space
    for chunk in range(prior_chunks):
        owner_rank, owner_half = kda_cp_chunk_owner(chunk, cp_size)
        summary = summaries[owner_rank * 2 + owner_half]
        emission = summary[..., :v_dim]  # [n, HV, K, V]
        transition = summary[..., v_dim:]  # [n, HV, K, K]
        state = transition @ state + emission
    return state.transpose(-1, -2)  # back to stored [n, HV, V, K]


@pytest.mark.parametrize("cp_size,prior_chunks", [(2, 1), (2, 3), (4, 6), (4, 7), (8, 13)])
def test_kda_cp_compose_matches_math_space_reference(cp_size: int, prior_chunks: int) -> None:
    generator = torch.Generator().manual_seed(42)
    n_seqs, hv, k_dim, v_dim = 2, 3, 8, 6
    base = torch.randn(n_seqs, hv, v_dim, k_dim, generator=generator)  # warm prefix
    base_before = base.clone()
    summaries = _random_summaries(n_seqs, 2 * cp_size, hv, k_dim, v_dim, generator)
    composed = kda_cp_compose_initial_state(base, summaries, prior_chunks)
    expected = _math_space_reference(base_before, summaries, prior_chunks)
    torch.testing.assert_close(composed, expected, rtol=1e-5, atol=1e-6)
    # The fold never mutates the base (both local halves fold from it).
    torch.testing.assert_close(base, base_before, rtol=0, atol=0)


def test_kda_cp_compose_empty_span_identity_is_exact() -> None:
    """An empty segment's summary is [0 | I]: injecting it anywhere in the
    causal chain leaves the composition unchanged (exact fp32 identity)."""
    generator = torch.Generator().manual_seed(7)
    n_seqs, hv, k_dim, v_dim, cp_size, prior_chunks = 1, 2, 8, 8, 4, 5
    base = torch.randn(n_seqs, hv, v_dim, k_dim, generator=generator)
    summaries = _random_summaries(n_seqs, 2 * cp_size, hv, k_dim, v_dim, generator)
    plain = kda_cp_compose_initial_state(base, summaries, prior_chunks)
    identity = torch.zeros_like(summaries)
    identity[..., v_dim:] = torch.eye(k_dim).view(1, 1, k_dim, k_dim)
    # The gathered stack is rank-major (row = owner_rank*2 + owner_half),
    # so the identity chunk is injected through the causal->row mapping:
    # the new causal chain is [c0, ID, c1, c2, ...] -- every chunk after the
    # identity shifts one causal slot later.
    with_identity = summaries.clone()
    for new_chunk in range(2 * cp_size):
        owner_rank, owner_half = kda_cp_chunk_owner(new_chunk, cp_size)
        row = owner_rank * 2 + owner_half
        if new_chunk == 1:
            with_identity[row] = identity[row]
        elif new_chunk >= 2:
            source_rank, source_half = kda_cp_chunk_owner(new_chunk - 1, cp_size)
            with_identity[row] = summaries[source_rank * 2 + source_half]
    shifted = kda_cp_compose_initial_state(base, with_identity, prior_chunks + 1)
    torch.testing.assert_close(shifted, plain, rtol=0, atol=0)


def test_kda_cp_compose_order_matters() -> None:
    """Reversed (newest->oldest) composition must differ: the fold is only
    correct in causal order."""
    generator = torch.Generator().manual_seed(11)
    n_seqs, hv, k_dim, v_dim, cp_size = 1, 2, 8, 8, 2
    base = torch.zeros(n_seqs, hv, v_dim, k_dim)
    summaries = _random_summaries(n_seqs, 2 * cp_size, hv, k_dim, v_dim, generator)
    causal = kda_cp_compose_initial_state(base, summaries, 3)
    wrong = kda_cp_compose_initial_state(base, summaries.flip(0), 3)
    assert not torch.equal(causal, wrong)


def test_kda_cp_compose_divergence_exists_but_is_bounded() -> None:
    """fp32 fold vs an independent fp64 math-space reference: zero priors is
    exact, and >= 2 priors diverge by re-association but stay orders below
    the fla 8e-3 budget (the production divergence -- kernel bf16 factors --
    is the NPU test's job; this pins the fold's own arithmetic)."""
    generator = torch.Generator().manual_seed(13)
    n_seqs, hv, k_dim, v_dim, cp_size = 1, 2, 16, 16, 4
    base = torch.randn(n_seqs, hv, v_dim, k_dim, generator=generator) * 0.1
    summaries = _random_summaries(n_seqs, 2 * cp_size, hv, k_dim, v_dim, generator)
    for prior_chunks, bound in ((1, 1e-5), (3, 1e-4), (7, 1e-3)):
        fp32 = kda_cp_compose_initial_state(base, summaries, prior_chunks)
        fp64 = _math_space_reference(base.double(), summaries.double(), prior_chunks)
        rel = (fp32.double() - fp64).norm() / fp64.norm().clamp_min(1e-30)
        assert rel.item() < bound, (prior_chunks, rel.item())
        if prior_chunks >= 3:
            assert rel.item() > 0.0  # the re-association divergence exists


# ---------------------------------------------------------------------------
# Pure tests: the three fold implementations (n1) -- tree scan + merge op
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cp_size,prior_chunks", [(2, 1), (2, 3), (4, 7), (8, 13)])
@pytest.mark.parametrize("k_dim,v_dim", [(16, 16), (128, 128)])
def test_kda_cp_tree_scan_fold_matches_sequential(cp_size: int, prior_chunks: int, k_dim: int, v_dim: int) -> None:
    """The Hillis-Steele tree scan vs the strictly sequential reference
    fold on random affines: identical causal chain, re-associated summation order.
    The bound is MEASURED and pinned (1-ulp-class at these scales: the
    scan composes the same chain in log2(2P+1) balanced levels)."""
    generator = torch.Generator().manual_seed(20261017)
    n_seqs, hv = 2, 2
    base = torch.randn(n_seqs, hv, v_dim, k_dim, generator=generator) * 0.1
    gathered = _random_summaries(n_seqs, 2 * cp_size, hv, k_dim, v_dim, generator)
    causal_rows = torch.tensor(kda_cp_causal_summary_rows(cp_size), dtype=torch.int64)
    causal = gathered.index_select(0, causal_rows)

    sequential = kda_cp_compose_initial_state(base, gathered, prior_chunks)
    scanned = kda_cp_tree_scan_fold_initial_states(base, causal, (prior_chunks,))[0]

    # Equality at fp32 tolerance with the measured re-association bound:
    # rel_fro stays <= ~2e-6 at every tested (cp, priors, dims) shape.
    difference = (scanned.double() - sequential.double()).norm().item()
    rel = difference / max(sequential.double().norm().item(), 1e-30)
    assert rel < 1e-5, (cp_size, prior_chunks, k_dim, rel)
    # Zero prior chunks must return the base itself.
    assert torch.equal(kda_cp_tree_scan_fold_initial_states(base, causal, (0,))[0], base)


def test_kda_cp_merge_fold_matches_sequential_cpu() -> None:
    """The merge-op adapter (virtual base row + causal reorder + head
    flatten) against the sequential reference fold, through the CPU reference
    merge op at the op's hardcoded K=V=128. Pins the adapter MATH: the
    chain seeded from the virtual base row, the zigzag->causal
    permutation, and the [V,K] state_v_first output orientation."""
    generator = torch.Generator().manual_seed(20261018)
    cp_size, n_seqs, hv = 4, 2, 2
    k_dim = v_dim = 128
    base = torch.randn(n_seqs, hv, v_dim, k_dim, generator=generator) * 0.1
    gathered = _random_summaries(n_seqs, 2 * cp_size, hv, k_dim, v_dim, generator)
    causal_rows = torch.tensor(kda_cp_causal_summary_rows(cp_size), dtype=torch.int64)

    for prior_chunks in (1, 3, 7):
        sequential = kda_cp_compose_initial_state(base, gathered, prior_chunks)
        folded = kda_cp_merge_fold_initial_states(
            _reference_merge_fwd_bwd_kernel, base, gathered, causal_rows, (prior_chunks,)
        )[0]
        assert tuple(folded.shape) == (n_seqs, hv, v_dim, k_dim)
        difference = (folded.double() - sequential.double()).norm().item()
        rel = difference / max(sequential.double().norm().item(), 1e-30)
        # The reference merge runs the same chain in natural orientation
        # (transposed operands): 1-ulp-class vs the stored-orientation fold.
        assert rel < 1e-5, (prior_chunks, rel)
    # Zero prior chunks: the adapter returns the base itself.
    folded = kda_cp_merge_fold_initial_states(_reference_merge_fwd_bwd_kernel, base, gathered, causal_rows, (0,))[0]
    assert torch.equal(folded, base)


def test_kda_cp_merge_fold_requires_128_dims() -> None:
    """The merge fold refuses non-128 dims loudly (the op's hardcoded
    contract); the dispatcher is responsible for selecting the scan."""
    generator = torch.Generator().manual_seed(1)
    base = torch.randn(1, 1, 8, 16, generator=generator)
    gathered = _random_summaries(1, 4, 1, 16, 8, generator)
    causal_rows = torch.tensor(kda_cp_causal_summary_rows(2), dtype=torch.int64)
    with pytest.raises(RuntimeError, match="K = V = 128"):
        kda_cp_merge_fold_initial_states(_reference_merge_fwd_bwd_kernel, base, gathered, causal_rows, (3,))


def test_kda_cp_causal_summary_rows_order() -> None:
    """The causal permutation is the zigzag inverse: causal chunk j maps to
    gathered row 2j below cp, mirrored above (cp4: [0,2,4,6,7,5,3,1])."""
    assert kda_cp_causal_summary_rows(2) == (0, 2, 3, 1)
    assert kda_cp_causal_summary_rows(4) == (0, 2, 4, 6, 7, 5, 3, 1)
    for cp_size in (2, 4, 8):
        rows = kda_cp_causal_summary_rows(cp_size)
        assert sorted(rows) == list(range(2 * cp_size))
        for chunk, row in enumerate(rows):
            owner_rank, owner_half = kda_cp_chunk_owner(chunk, cp_size)
            assert row == owner_rank * 2 + owner_half


def test_kda_cp_fold_mode_selection() -> None:
    """The fold selection is a pure shape function: merge inside the op's
    K=V=128 / n_seqs*HV <= 256 head budget, the tree scan outside it."""
    from xllm.python.attention.kda_linear_attention import _kda_cp_fold_mode

    assert _kda_cp_fold_mode(128, 128, 32) == "merge"
    assert _kda_cp_fold_mode(128, 128, 256) == "merge"  # head budget boundary
    assert _kda_cp_fold_mode(64, 64, 32) == "scan"  # K != 128
    assert _kda_cp_fold_mode(128, 128, 512) == "scan"  # heads > 256
    assert _kda_cp_fold_mode(128, 127, 32) == "scan"  # V != 128


def test_kda_cp_host_plan_reuse_and_contents() -> None:
    """The host-plan cache: same (seg_cu, rank, cp, conv_state_len, device)
    returns the SAME tensor objects (zero per-call reconstruction);
    contents match a fresh construction exactly; distinct shapes get
    distinct plans and the LRU stays bounded."""
    from xllm.python.attention.kda_linear_attention import (
        _KDA_CP_HOST_PLAN_CAPACITY,
        _kda_cp_host_plan,
    )

    device = torch.device("cpu")
    # 2 sequences x 2 halves, cumulative across ALL local segments (the
    # local q_cu_seqlens the CP context builder emits).
    seg_cu = [0, 128, 256, 384, 512]
    plan = _kda_cp_host_plan(seg_cu, 1, 2, 3, device)
    assert _kda_cp_host_plan(seg_cu, 1, 2, 3, device) is plan  # cache hit
    # Rank 1 at cp2 has NO zero-prior segments (both halves own priors), so
    # its plan carries no early/late packs: the single full replay covers it.
    assert plan.early_cu is None and plan.early_rows is None and plan.early_idx is None
    assert plan.late_cu is None and plan.late_rows is None and plan.late_idx is None
    torch.testing.assert_close(plan.conv_cu, torch.tensor(seg_cu, dtype=torch.int64))
    torch.testing.assert_close(plan.replay_cu, torch.tensor(seg_cu, dtype=torch.int32))
    assert plan.causal_rows.tolist() == list(kda_cp_causal_summary_rows(2))
    # halo_pos: each segment's last conv_state_len packed positions, in the
    # halo payload's flat (half, sequence, window) row order.
    assert plan.halo_pos.tolist() == [
        125,
        126,
        127,
        381,
        382,
        383,  # first-half tails: seq 0, seq 1
        253,
        254,
        255,
        509,
        510,
        511,  # second-half tails: seq 0, seq 1
    ]
    # A different conv_state_len is a different plan (the halo positions
    # shift with the window), never a stale cache hit.
    plan_w2 = _kda_cp_host_plan(seg_cu, 1, 2, 2, device)
    assert plan_w2 is not plan
    assert plan_w2.halo_pos.tolist() == [126, 127, 382, 383, 254, 255, 510, 511]

    # Rank 0's first halves are the zero-prior (early) segments.
    plan_zero = _kda_cp_host_plan(seg_cu, 0, 2, 3, device)
    assert plan_zero is not plan
    assert plan_zero.early_cu is not None and plan_zero.early_rows is not None and plan_zero.early_idx is not None
    assert plan_zero.early_idx.tolist() == [0, 2]
    assert plan_zero.late_idx.tolist() == [1, 3]
    assert plan_zero.early_rows.tolist() == list(range(0, 128)) + list(range(256, 384))
    assert plan_zero.late_rows.tolist() == list(range(128, 256)) + list(range(384, 512))
    assert plan_zero.early_cu.tolist() == [0, 128, 256]
    assert plan_zero.late_cu.tolist() == [0, 128, 256]
    assert plan_zero.halo_pos.tolist() == plan.halo_pos.tolist()

    # LRU bound: distinct shapes beyond the capacity evict the oldest.
    for index in range(_KDA_CP_HOST_PLAN_CAPACITY + 4):
        _kda_cp_host_plan([0, 64 * (index + 1), 128 * (index + 1)], 0, 2, 3, device)
    from xllm.python.attention.kda_linear_attention import _KDA_CP_HOST_PLANS

    assert len(_KDA_CP_HOST_PLANS) <= _KDA_CP_HOST_PLAN_CAPACITY
    # The original key was evicted; a re-request rebuilds (new objects).
    assert _kda_cp_host_plan(seg_cu, 1, 2, 3, device) is not plan


def test_kda_cp_rows_are_stream_prefix_guard() -> None:
    """The fast row-staging guard: gate-passing layouts (every segment a
    full chunk, so the packed real rows front-pack into one prefix) pass;
    ragged non-last second halves, short streams, and batch > 1 do not, so
    the gather fallback keeps them correct."""
    from xllm.python.attention.kda_linear_attention import _kda_cp_rows_are_stream_prefix

    def qkv(batch: int, rows: int) -> torch.Tensor:
        return torch.empty(batch, 8, rows)

    # 1 sequence, both halves full (chunk_len 256 == both counts): prefix.
    assert _kda_cp_rows_are_stream_prefix([0, 256, 512], qkv(1, 512))
    # 2 sequences, all halves full: prefix.
    assert _kda_cp_rows_are_stream_prefix([0, 128, 256, 384, 512], qkv(1, 512))
    # 1 sequence, short LAST second half with the padded footprint intact:
    # still a prefix (the real rows front-pack per segment).
    assert _kda_cp_rows_are_stream_prefix([0, 256, 448], qkv(1, 512))
    # 2 sequences, a NON-last second half short: the next sequence's base
    # no longer abuts, so the guard must refuse (gap rows in between).
    assert not _kda_cp_rows_are_stream_prefix([0, 128, 192, 320, 448], qkv(1, 512))
    # Stream shorter than the padded footprint: not a prefix.
    assert not _kda_cp_rows_are_stream_prefix([0, 256, 512], qkv(1, 500))
    # Batch > 1: the flattened stream is not the rank-local layout.
    assert not _kda_cp_rows_are_stream_prefix([0, 256, 512], qkv(2, 512))


def test_kda_cp_warm_host_values_resolution() -> None:
    """Host-side warm-value resolution for the conv call's ism (n6): the
    metadata host twin wins when its length matches; a host-list or CPU
    warm mask is read without any device round trip; a length-mismatched
    twin falls back to the mask; values normalize to 0/1 exactly like the
    bool view the g1 form consumed."""
    from xllm.python.attention.kda_linear_attention import _kda_cp_warm_host_values

    num_seqs = 2
    # 1. The host twin (the engine path): wins over the tensor form.
    metadata = SimpleNamespace(linear_state_validity_host_values=[1, 0])
    warm = torch.tensor([0, 1], dtype=torch.int64)
    assert _kda_cp_warm_host_values(metadata, warm, num_seqs) == [1, 0]
    # 2. A twin with the wrong length is ignored (graph-padded forms).
    metadata = SimpleNamespace(linear_state_validity_host_values=[1, 0, 1])
    assert _kda_cp_warm_host_values(metadata, warm, num_seqs) == [0, 1]
    metadata = SimpleNamespace(linear_state_validity_host_values=[])
    assert _kda_cp_warm_host_values(metadata, warm, num_seqs) == [0, 1]
    # 3. No twin at all: a host-list warm mask reads directly.
    metadata = SimpleNamespace()
    assert _kda_cp_warm_host_values(metadata, [True, False], num_seqs) == [1, 0]
    # 4. No twin: a CPU tensor mask reads via .tolist() (host-only).
    assert _kda_cp_warm_host_values(metadata, torch.tensor([2, 0], dtype=torch.int64), num_seqs) == [1, 0]
    # 5. Values normalize like the bool view did (any nonzero -> 1).
    assert _kda_cp_warm_host_values(metadata, [7, 0], num_seqs) == [1, 0]
    # 6. A device-typed mask with no twin still resolves (the documented
    # sync fallback -- correct everywhere; the engine always supplies the
    # twin). Verified on the CPU device here.
    device_mask = torch.tensor([1, 1], dtype=torch.int64)
    assert _kda_cp_warm_host_values(metadata, device_mask, num_seqs) == [1, 1]


def test_kda_cp_host_plan_conv_metadata_is_cpu() -> None:
    """The conv-call metadata pair in the host plan (conv_cu, empty_i64)
    is CPU int64: the conv op marshals qsl/si/ism to host IntArrayRefs in
    its C++ wrapper, and .to(kCPU, kInt64) is a no-op only on tensors that
    are ALREADY cpu+int64, so the plan tensors must live host-side for the
    marshalling to stay sync-free (n6). The other plan tensors stay on the
    plan's device for the gather/select calls that consume them."""
    from xllm.python.attention.kda_linear_attention import _kda_cp_host_plan

    device = torch.device("cpu")
    seg_cu = [0, 128, 256, 384, 512]
    plan = _kda_cp_host_plan(seg_cu, 0, 2, 3, device)
    assert plan.conv_cu.device.type == "cpu" and plan.conv_cu.dtype == torch.int64
    assert plan.conv_cu.tolist() == seg_cu
    assert plan.empty_i64.device.type == "cpu" and plan.empty_i64.dtype == torch.int64
    assert plan.empty_i64.numel() == 0
    assert plan.replay_cu.device.type == device.type
    assert plan.halo_pos.device.type == device.type


# ---------------------------------------------------------------------------
# Reference kernels (CPU): affine recurrence + exact closed-form summary
# ---------------------------------------------------------------------------


def _mock_decay(gate_raw: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor, lower_bound: float) -> torch.Tensor:
    """Per-channel log2 decay the mock kernel derives from the raw gate.

    Both the recurrence and the summary op use the same values, so their
    affine maps agree by construction. Decay lands in (lower_bound, 0):
    exp2 of it is a mild per-channel multiplier, model-isomorphic.
    """
    _, hv, dim = gate_raw.shape
    a = a_log.float().view(1, hv, 1)
    dt = dt_bias.float().view(1, hv, dim)
    return lower_bound * torch.sigmoid(a * gate_raw.float() + dt)


def _reference_chunk_kda_fwd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor | None,
    beta: torch.Tensor,
    scale: float,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Affine-recurrence stand-in for the NPU fused kernel (stored-space
    state, per-channel diagonal decay): X_t = X_{t-1} @ Diag(exp2(gk_t)) +
    beta_t * outer(v_t, k_t); output row = scale * X_t @ q_t.

    The CP path calls this only for the REPLAY pass (with a merged initial
    state); the summary factors now come from ``_reference_prepare``.
    """
    initial_state = kwargs.get("initial_state")
    cu_seqlens = kwargs.get("cu_seqlens")
    a_log = kwargs["A_log"]
    dt_bias = kwargs["dt_bias"]
    lower_bound = kwargs["lower_bound"]
    cu = cu_seqlens.tolist() if isinstance(cu_seqlens, torch.Tensor) else list(cu_seqlens)
    _, total, hv, dim = query.shape
    q = query.float().reshape(total, hv, dim)
    k = key.float().reshape(total, hv, dim)
    v = value.float().reshape(total, hv, dim)
    b = beta.float().reshape(total, hv)
    decay = _mock_decay(gate.float().reshape(total, hv, dim), a_log, dt_bias, lower_bound)
    lin_decay = torch.exp2(decay)
    n_seqs = len(cu) - 1
    if initial_state is None:
        states = torch.zeros(n_seqs, hv, dim, dim)
    else:
        states = initial_state.float().clone()
    output = torch.zeros(1, total, hv, dim)
    finals = torch.zeros(n_seqs, hv, dim, dim)
    for seq in range(n_seqs):
        start, end = cu[seq], cu[seq + 1]
        state = states[seq].clone()  # [HV, V, K]
        for t in range(start, end):
            state = state * lin_decay[t].view(hv, 1, dim) + b[t].view(hv, 1, 1) * torch.einsum("hv,hk->hvk", v[t], k[t])
            output[0, t] = scale * torch.einsum("hvk,hk->hv", state, q[t])
        finals[seq] = state
    return output.to(query.dtype), finals


def _reference_prepare(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
    **kwargs: Any,
) -> tuple[torch.Tensor | None, ...]:
    """Prepare-op stand-in: the same per-token factor values the reference
    fused mock's recurrence uses (gk/w/u/kg), in the 13-slot prepare layout
    (ma1 fused->prepare correspondence: gk 0, w 3, u 4, kg 6). The engine
    path only consumes slots 0/3/4/6 with backward_mode="none"; the rest
    stay None like the real op's unselected slots."""
    del scale, chunk_size  # the mock derives factors from the raw inputs only
    _, total, hv, dim = query.shape
    q_flat = query.float().reshape(total, hv, dim)
    k_flat = key.float().reshape(total, hv, dim)
    v_flat = value.float().reshape(total, hv, dim)
    b = beta.float().reshape(total, hv)
    decay = _mock_decay(gate.float().reshape(total, hv, dim), kwargs["A_log"], kwargs["dt_bias"], kwargs["lower_bound"])
    # Factor slots (BNSD, mirroring the real kernel's head-major layout):
    # gk = per-channel log2 decay, w = beta*k, u = v, kg = k.
    gk_factor = decay.permute(1, 0, 2).unsqueeze(0).contiguous()
    w_factor = (b.view(total, hv, 1) * k_flat).permute(1, 0, 2).unsqueeze(0).contiguous()
    u_factor = v_flat.permute(1, 0, 2).unsqueeze(0).contiguous()
    kg_factor = k_flat.permute(1, 0, 2).unsqueeze(0).contiguous()
    slots: list[torch.Tensor | None] = [None] * 13
    slots[0], slots[3], slots[4], slots[6] = gk_factor, w_factor, u_factor, kg_factor
    return tuple(slots)


def _reference_ppfm(kg: Any, w: Any, u: Any, *, gk: torch.Tensor, cu_seqlens: Any, chunk_size: int) -> torch.Tensor:
    """Reference summary op: hm = [E | M] closed-form from the mock factors.

    Multi-segment contract of the real op (the ATK design): ONE affine
    chain per cu_seqlens interval, ``hm[Nseq, HV, K, V+K]`` fp32 with row
    i = interval i's summary -- the batched form the engine path uses
    since M-B2 (and the per-segment form is the Nseq=1 case).

    M_stored = Diag(prod_t exp2(gk_t)); E_stored = sum_t Tail_t *
    beta_t * outer(v_t, k_t) with Tail_t = prod_{s>t} exp2(gk_s) per channel
    -- the exact affine map the reference recurrence applies to the initial
    state (final(X) = X @ M_stored + E_stored).
    """
    del kg
    cu = cu_seqlens.tolist() if isinstance(cu_seqlens, torch.Tensor) else list(cu_seqlens)
    hv, k_dim = gk.shape[1], gk.shape[3]
    v_dim = u.shape[3]
    lin = torch.exp2(gk[0].float())  # [HV, T, K]
    w_f = w[0].float()  # [HV, T, K] = beta * k
    u_f = u[0].float()  # [HV, T, V]
    n_seqs = len(cu) - 1
    hm = torch.zeros(n_seqs, hv, k_dim, v_dim + k_dim)
    for seq in range(n_seqs):
        start, end = cu[seq], cu[seq + 1]
        emission = torch.zeros(hv, v_dim, k_dim)
        tail = torch.ones(hv, k_dim)
        for t in range(end - 1, start - 1, -1):
            emission = emission + torch.einsum("hv,hk->hvk", u_f[:, t], w_f[:, t]) * tail.view(hv, 1, k_dim)
            tail = tail * lin[:, t]
        transition = torch.diag_embed(tail)
        hm[seq, :, :, :v_dim] = emission.transpose(-1, -2)
        hm[seq, :, :, v_dim:] = transition.transpose(-1, -2)
    return hm


def _reference_causal_conv1d_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    """CPU reference for xllm_ops::causal_conv1d_prefill: per-sequence
    left-context depthwise SiLU conv, cold mode ignores the state buffer
    (zero context), the final rolling window lands in conv_state in place."""
    del state_indices
    qsl = query_start_loc.tolist() if isinstance(query_start_loc, torch.Tensor) else list(query_start_loc)
    has_state = has_initial_state.tolist() if torch.is_tensor(has_initial_state) else list(has_initial_state)
    channels = x.shape[-1]
    output = torch.empty_like(x)
    for seq in range(len(qsl) - 1):
        start, end = qsl[seq], qsl[seq + 1]
        if start == end:
            continue
        context = conv_state[seq].float() if has_state[seq] else torch.zeros_like(conv_state[seq].float())
        window = torch.cat((context, x[start:end].float()), dim=0)
        convolved = (
            functional.conv1d(window.t().unsqueeze(0), weight.float().t().unsqueeze(1), groups=channels).squeeze(0).t()
        )
        output[start:end] = functional.silu(convolved).to(x.dtype)
        conv_state[seq] = window[-conv_state.shape[1] :].to(conv_state.dtype)
    return output


def _reference_merge_fwd_bwd_kernel(
    ag_hm: torch.Tensor,
    pre_or_post_num_ranks: int,
    rank: int,
    *,
    forward: bool = True,
    state_v_first: bool = True,
) -> torch.Tensor:
    """Reference merge op (CPU stand-in for aclnnMergeFwdBwdKernel).

    Implements the source-pinned contract (task research n1 section 1):
    the chain seeds from the FIRST prior row's E block -- the caller's h
    values are never read -- then applies ``h <- M_i @ h + E_i`` in
    natural [K, V] orientation over the N rows before ``rank`` (forward)
    or after it in reverse order (backward). ``state_v_first=True``
    returns ``h`` stored as [V, K] (a pure transpose, exact in fp32).
    """
    s_count, hv, k_dim, vk_dim = ag_hm.shape
    v_dim = vk_dim - k_dim
    if not 0 <= int(rank) < s_count:
        raise RuntimeError(f"rank out of range: {rank} vs S={s_count}")
    n_ranks = int(pre_or_post_num_ranks)
    if forward:
        rows = list(range(rank - n_ranks, rank))
    else:
        rows = list(range(rank + n_ranks, rank, -1))
    if not rows:
        raise RuntimeError("empty merge window")
    h = ag_hm[rows[0], ..., :v_dim].float().clone()
    for row in rows[1:]:
        h = ag_hm[row, ..., v_dim:] @ h + ag_hm[row, ..., :v_dim]
    if state_v_first:
        h = h.transpose(-1, -2).contiguous()
    return h


def _install_kda_cp_reference_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the reference kernels as fla_npu.ops.ascendc plus the xllm
    CP ops probe and the causal_conv1d_prefill op (all reverted by
    monkeypatch)."""
    ascendc = types.ModuleType("fla_npu.ops.ascendc")
    ascendc.chunk_kda_fwd = _reference_chunk_kda_fwd
    ascendc.recurrent_kda = _reference_chunk_kda_fwd
    fla_npu_ops = types.ModuleType("fla_npu.ops")
    fla_npu_ops.ascendc = ascendc
    fla_npu = types.ModuleType("fla_npu")
    fla_npu.ops = fla_npu_ops
    monkeypatch.setitem(sys.modules, "fla_npu", fla_npu)
    monkeypatch.setitem(sys.modules, "fla_npu.ops", fla_npu_ops)
    monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", ascendc)
    monkeypatch.setattr(
        kda_module,
        "_kda_cp_ops",
        lambda: (_reference_prepare, _reference_ppfm, _reference_merge_fwd_bwd_kernel),
    )
    monkeypatch.setattr(torch.ops.xllm_ops, "causal_conv1d_prefill", _reference_causal_conv1d_prefill, raising=False)


# ---------------------------------------------------------------------------
# Fake CP group (in-process collectives)
# ---------------------------------------------------------------------------


class _FakeAsyncWork:
    """Async-collective handle: wait() records the event, data already landed.

    The fake fills outputs at ISSUE time (single-process ranks run
    sequentially, so issue and wait are within one rank's turn) -- the
    ordering between issue and wait is still observable through the
    recorded event log.
    """

    def __init__(self, group: _FakeCpGroup, kind: str) -> None:
        self._group = group
        self._kind = kind

    def wait(self) -> None:
        self._group.events.append((f"{self._kind}_wait", self._group.rank))


class _FakeCpGroup:
    """In-process CP group for single-process multi-rank simulation.

    Covers the collective seams the CP path uses:

    * ``distributed.all_gather`` (the graph-op wrapper; the CP path must
      NOT use it -- the sync-gather events make a transport regression
      observable);
    * ``torch.distributed.all_gather_into_tensor(async_op=True)`` (the
      async transport): the halo gather is input-derived (the test knows
      every rank's tails), the summary gather records each rank's payload
      and feeds back previously recorded peers (missing peers get zeros --
      the caller re-runs ranks in rounds, mirroring a real multi-process
      exchange);
    * ``distributed.cp_process_group`` (the collective wrapper).

    ``events`` records the transport call order per rank (gather issues,
    gather waits) so ordering assertions (e.g. the zero-prior replay
    running between summary issue and wait) are possible.
    """

    def __init__(self, cp_size: int, halo_by_rank: dict[int, torch.Tensor]) -> None:
        self.cp_size = cp_size
        self.halo_by_rank = halo_by_rank
        self.summaries_by_rank: dict[int, torch.Tensor] = {}
        self.broadcast_payload: list[torch.Tensor] = []
        self.broadcast_sources: list[int] = []
        self.broadcast_calls = 0
        self.events: list[tuple[str, int]] = []
        self._gather_count = 0
        self._broadcast_count = 0
        self.rank = 0

    def _next_gather_kind(self) -> str:
        self._gather_count += 1
        return "halo" if self._gather_count % 2 == 1 else "summary"

    def begin_run(self, rank: int) -> None:
        """Start one virtual rank's forward: resets the per-run collective
        counters (each run performs one halo exchange, one summary exchange
        and one merged final-state broadcast) and marks the per-run slice
        of the event log."""
        self.rank = rank
        self._gather_count = 0
        self._broadcast_count = 0
        self.events_marker = len(self.events)

    def _fill_gather(self, output: torch.Tensor, value: torch.Tensor, kind: str) -> None:
        if kind == "halo":
            # Halo exchange: every rank's payload is input-derived.
            parts = [self.halo_by_rank[rank] for rank in range(self.cp_size)]
        else:
            # Summary exchange: record this rank's payload, feed back the
            # recorded peers (zeros for not-yet-run ranks).
            self.summaries_by_rank[self.rank] = value.clone()
            parts = [
                value if rank == self.rank else self.summaries_by_rank.get(rank, torch.zeros_like(value))
                for rank in range(self.cp_size)
            ]
        output.copy_(torch.cat(parts, dim=0))

    def _all_gather(self, value: torch.Tensor, dim: int, world_size: int, group_name: str) -> torch.Tensor:
        assert dim == 0 and world_size == self.cp_size and group_name == "cp"
        kind = self._next_gather_kind()
        self.events.append((f"{kind}_sync", self.rank))
        output = value.new_empty((value.shape[0] * world_size, *value.shape[1:]))
        self._fill_gather(output, value, kind)
        return output

    def _all_gather_into_tensor(
        self, output: torch.Tensor, value: torch.Tensor, group: Any = None, async_op: bool = False
    ) -> _FakeAsyncWork:
        assert async_op is True, "the async transport must request async_op=True"
        assert value.is_contiguous() and output.is_contiguous(), "HCCL requires contiguous gather buffers"
        assert output.shape[0] == value.shape[0] * self.cp_size
        kind = self._next_gather_kind()
        self.events.append((f"{kind}_issue", self.rank))
        self._fill_gather(output, value, kind)
        return _FakeAsyncWork(self, kind)

    def _broadcast_(self, tensor: torch.Tensor, src: int, group_name: str) -> None:
        assert group_name == "cp"
        assert src == 0, "the causally-last owner is rank 0's second segment at every cp size"
        self._broadcast_count += 1
        self.broadcast_calls += 1
        self.broadcast_sources.append(src)
        if self.rank == src:
            while len(self.broadcast_payload) < self._broadcast_count:
                self.broadcast_payload.append(tensor.clone())
            self.broadcast_payload[self._broadcast_count - 1] = tensor.clone()
        else:
            assert len(self.broadcast_payload) >= self._broadcast_count
            tensor.copy_(self.broadcast_payload[self._broadcast_count - 1])

    def _cp_process_group(self, device: Any) -> Any:
        return self

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cp_utils.distributed, "all_gather", self._all_gather, raising=False)
        monkeypatch.setattr(cp_utils.distributed, "broadcast_", self._broadcast_, raising=False)
        monkeypatch.setattr(cp_utils.distributed, "cp_process_group", self._cp_process_group, raising=False)
        monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", self._all_gather_into_tensor)


# ---------------------------------------------------------------------------
# Orchestration tests: model forward branch selection
# ---------------------------------------------------------------------------


def _forward_test_attention(enable_kda_cp: bool = True) -> glm5_next.Glm5NextKdaAttention:
    attention = glm5_next.Glm5NextKdaAttention.__new__(glm5_next.Glm5NextKdaAttention)
    torch.nn.Module.__init__(attention)
    attention.layer_id = 0
    attention.hidden_size = 1
    attention.head_dim = 1
    attention.qkv_dim = 1
    attention.conv_dim = 3
    attention.conv_kernel_size = 4
    attention.activation = "silu"
    attention.num_heads_local = 1
    attention.tp = 1
    # The config-channel switch (the ctor normally stores it from
    # cfg.enable_kda_cp; the __new__-based stub pins it directly).
    attention.enable_kda_cp = enable_kda_cp
    attention.in_proj_qkvbfg_a = lambda hidden: torch.cat(
        (hidden.expand(-1, -1, 3), hidden, hidden, hidden),
        dim=-1,
    )
    attention.input_projection_sizes = (3, 1, 2)
    attention.forget_gate = SimpleNamespace(
        raw_projection=lambda hidden: hidden.unsqueeze(-1),
        safe_gate_lower_bound=-5.0,
        gate_from_raw=lambda raw: -0.5 * torch.sigmoid(raw),
    )
    attention.g_b_proj = lambda hidden: hidden * 100
    attention._fg_b_weight = None
    attention.o_norm = lambda core, gate: core + gate
    attention.o_proj = torch.nn.Identity()
    return attention


def test_kda_cp_forward_skips_merge_when_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flag on + gate passing: no cp_merge_rows / cp_shard_rows, and the
    backend receives LOCAL rows plus the cp_context."""
    monkeypatch.setattr(kda_module, "_kda_cp_ops", lambda: (object(), object(), object()))
    attention = _forward_test_attention(enable_kda_cp=True)
    device = torch.device("cpu")
    cp_context = _toy_cp_context([256], 2, 0, device)
    assert cp_context.total_local == 128
    local_hidden = torch.arange(128, dtype=torch.float32).view(1, 128, 1) / 128.0

    calls: dict[str, Any] = {}

    def _execute_linear(mixed_qkv, beta_raw, layer, *, raw_gate_proj, cp_context=None):
        calls["mixed_qkv"] = mixed_qkv
        calls["beta_raw"] = beta_raw
        calls["raw_gate_proj"] = raw_gate_proj
        calls["cp_context"] = cp_context
        return torch.zeros(1, cp_context.total_local, 1, 1)

    backend = SimpleNamespace(execute_linear=_execute_linear)
    with (
        patch.object(
            glm5_next,
            "get_forward_context_or_none",
            return_value=SimpleNamespace(
                attention_backend=backend, cp_context=cp_context, metadata=_gate_metadata([256])
            ),
        ),
        patch.object(glm5_next, "cp_merge_rows") as merge_rows,
        patch.object(glm5_next, "cp_shard_rows") as shard_rows,
    ):
        output = attention(
            local_hidden, torch.tensor([[0, 127]], dtype=torch.int32), torch.ones(1, 128, dtype=torch.bool)
        )

    merge_rows.assert_not_called()
    shard_rows.assert_not_called()
    assert calls["cp_context"] is cp_context
    # LOCAL rows: [1, conv_dim, total_local] untouched by any merge.
    assert tuple(calls["mixed_qkv"].shape) == (1, 3, 128)
    expected = local_hidden.expand(1, 128, 3).transpose(1, 2).contiguous()
    torch.testing.assert_close(calls["mixed_qkv"], expected)
    torch.testing.assert_close(calls["beta_raw"], local_hidden)
    assert tuple(output.shape) == (1, 128, 1)


def test_kda_cp_forward_keeps_legacy_path_when_gate_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flag on but segments unaligned (4 tokens at cp2): every rank takes
    the legacy merge/shard path and the backend sees global rows."""
    monkeypatch.setattr(kda_module, "_kda_cp_ops", lambda: (object(), object(), object()))
    attention = _forward_test_attention(enable_kda_cp=True)
    device = torch.device("cpu")
    cp_context = _toy_cp_context([4], 2, 0, device)
    local_hidden = torch.tensor([[[1.0], [2.0]]])

    def _all_gather(value, dim, world_size, group_name):
        assert group_name == "cp"
        return torch.cat((value, value + 10), dim=0)

    calls: dict[str, Any] = {}

    def _execute_linear(mixed_qkv, beta_raw, layer, *, raw_gate_proj, cp_context=None):
        calls["mixed_qkv"] = mixed_qkv
        calls["cp_context"] = cp_context
        return mixed_qkv[:, :1].transpose(1, 2).unsqueeze(2)

    backend = SimpleNamespace(execute_linear=_execute_linear)
    with (
        patch.object(
            glm5_next,
            "get_forward_context_or_none",
            return_value=SimpleNamespace(
                attention_backend=backend, cp_context=cp_context, metadata=_gate_metadata([4])
            ),
        ),
        patch.object(glm5_next.distributed, "all_gather", _all_gather, create=True),
        patch.object(glm5_next, "cp_merge_rows", wraps=glm5_next.cp_merge_rows) as merge_rows,
    ):
        attention(local_hidden, torch.tensor([[0, 3]], dtype=torch.int32), torch.ones(1, 2, dtype=torch.bool))

    assert merge_rows.call_count == 3
    assert calls["cp_context"] is None
    # GLOBAL rows reach the backend (the merged 4-token stream).
    assert tuple(calls["mixed_qkv"].shape) == (1, 3, 4)


def test_kda_cp_forward_legacy_when_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flag off (--enable_kda_cp=false): byte-identical legacy path even for
    a fully eligible batch."""
    monkeypatch.setattr(kda_module, "_kda_cp_ops", lambda: (object(), object(), object()))
    attention = _forward_test_attention(enable_kda_cp=False)
    device = torch.device("cpu")
    cp_context = _toy_cp_context([256], 2, 0, device)
    local_hidden = torch.arange(128, dtype=torch.float32).view(1, 128, 1) / 128.0

    def _all_gather(value, dim, world_size, group_name):
        return torch.cat((value, value + 10), dim=0)

    calls: dict[str, Any] = {}

    def _execute_linear(mixed_qkv, beta_raw, layer, *, raw_gate_proj, cp_context=None):
        calls["cp_context"] = cp_context
        return torch.zeros(1, 256, 1, 1)

    backend = SimpleNamespace(execute_linear=_execute_linear)
    with (
        patch.object(
            glm5_next,
            "get_forward_context_or_none",
            return_value=SimpleNamespace(
                attention_backend=backend, cp_context=cp_context, metadata=_gate_metadata([256])
            ),
        ),
        patch.object(glm5_next.distributed, "all_gather", _all_gather, create=True),
        patch.object(glm5_next, "cp_merge_rows", wraps=glm5_next.cp_merge_rows) as merge_rows,
        patch.object(glm5_next, "cp_shard_rows", wraps=glm5_next.cp_shard_rows) as shard_rows,
    ):
        attention(local_hidden, torch.tensor([[0, 127]], dtype=torch.int32), torch.ones(1, 128, dtype=torch.bool))

    assert merge_rows.call_count == 3
    assert shard_rows.call_count == 1
    assert calls["cp_context"] is None


def test_kda_cp_flag_config_channel() -> None:
    """The config channel (the --enable_kda_cp engine gflag forwarded through
    the model config dict): from_dict picks the boolean up (absent key ->
    default true; explicit false -> off) and the attention ctor stores it
    verbatim."""

    def _attention(enable_kda_cp: bool) -> glm5_next.Glm5NextKdaAttention:
        cfg = glm5_next.Glm5NextConfig(hidden_size=32, kda_num_heads=8, kda_head_dim=8)
        cfg.enable_kda_cp = enable_kda_cp
        return glm5_next.Glm5NextKdaAttention(cfg, 0, torch.float32, torch.device("cpu"))

    assert glm5_next.Glm5NextConfig.from_dict({}).enable_kda_cp is True
    assert glm5_next.Glm5NextConfig.from_dict({"enable_kda_cp": False}).enable_kda_cp is False
    assert _attention(True).enable_kda_cp is True
    assert _attention(False).enable_kda_cp is False


def test_kda_cp_flag_defaults_true() -> None:
    """The DEFAULT is ON: with the flag absent everywhere (no gflag on the
    engine command line, no key in the config dict), KDA CP prefill is
    enabled — the two-pass scheme is the ONLY shipped KDA CP implementation
    (the fused operator stayed out of the standard vendor per the G5
    verdict), so the flag exists to opt OUT with --enable_kda_cp=false.
    Pins every layer of the channel: the dataclass default, the from_dict
    absent-key default, and the attention ctor's stored attribute."""
    # The dataclass default (what py_causal_lm's missing-key fallback and
    # non-NPU builds see).
    assert glm5_next.Glm5NextConfig().enable_kda_cp is True
    # The from_dict absent-key default (the flag-absent engine form).
    assert glm5_next.Glm5NextConfig.from_dict({}).enable_kda_cp is True
    # The ctor store: a config built with no explicit switch leaves the
    # attention layer CP-enabled.
    cfg = glm5_next.Glm5NextConfig(hidden_size=32, kda_num_heads=8, kda_head_dim=8)
    attention = glm5_next.Glm5NextKdaAttention(cfg, 0, torch.float32, torch.device("cpu"))
    assert attention.enable_kda_cp is True


# ---------------------------------------------------------------------------
# Orchestration test: the full backend CP pipeline vs a full-stream reference
# ---------------------------------------------------------------------------


def _model_isomorphic_gate_inputs(
    total: int, num_heads: int, head_dim: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """m0b/m1 model-isomorphic gate/beta regime: small beta and MILD
    per-channel decay (safe-gate targets s ~ U(1e-4, 2e-3) ->
    gate = -5*s in [-0.01, -0.0005] with A_log=dt_bias=0). Full-magnitude
    random gates decay so fast that prior-segment contributions underflow
    to exact zeros, which would make every compose trivially bit-exact and
    hide the re-association divergence this suite must bound."""
    beta_targets = 0.005 + 0.045 * torch.rand(total, num_heads, generator=generator)
    beta_raw = torch.log(beta_targets / (1.0 - beta_targets))
    gate_targets = 1e-4 + 1.9e-3 * torch.rand(total, num_heads, head_dim, generator=generator)
    gate_raw = torch.log(gate_targets / (1.0 - gate_targets))
    return beta_raw, gate_raw


def _toy_layer(
    device: torch.device, dtype: torch.dtype, generator: torch.Generator, head_dim: int = 8
) -> SimpleNamespace:
    num_heads = 2
    qkv_dim = num_heads * head_dim
    conv_dim = 3 * qkv_dim
    return SimpleNamespace(
        layer_id=0,
        conv_kernel_size=4,
        conv_dim=conv_dim,
        head_dim=head_dim,
        num_heads_local=num_heads,
        qkv_dim=qkv_dim,
        activation="silu",
        conv_weight_t=(torch.randn(4, conv_dim, generator=generator) * 0.5).to(device=device, dtype=dtype),
        forget_gate=SimpleNamespace(
            A_log=torch.zeros(num_heads, dtype=torch.float32, device=device),
            dt_bias=torch.zeros(num_heads * head_dim, dtype=torch.float32, device=device),
            safe_gate_lower_bound=-5.0,
        ),
    )


def _run_cp_rank(
    rank: int,
    group: _FakeCpGroup,
    q_seq_lens: list[int],
    layer: SimpleNamespace,
    global_qkv: torch.Tensor,
    global_beta: torch.Tensor,
    global_gate: torch.Tensor,
    cp_size: int,
    device: torch.device,
    metadata: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One virtual rank's real ``execute_linear`` (CP branch) with the fake
    group installed. Returns (output, conv_cache, ssm_cache). ``metadata``
    overrides the default cold gate metadata (warm-mask / host-twin tests)."""
    layout = _zigzag_layout(q_seq_lens, cp_size, rank)
    gather = torch.tensor([row if row >= 0 else 0 for row in layout["shard_index"]], dtype=torch.int64, device=device)
    mixed_qkv = global_qkv.index_select(0, gather).unsqueeze(0).transpose(1, 2).contiguous()
    beta_raw = global_beta.index_select(0, gather).unsqueeze(0).contiguous()
    raw_gate = global_gate.index_select(0, gather).unsqueeze(0).contiguous()

    num_seqs = len(q_seq_lens)
    # Checkpoint stride 2: one speculative column that must stay untouched.
    conv_cache = torch.full((num_seqs, 4, layer.conv_dim), 7.0, dtype=global_qkv.dtype, device=device)
    ssm_cache = torch.full(
        (2 * num_seqs, layer.num_heads_local, layer.head_dim, layer.head_dim), 9.0, dtype=torch.float32, device=device
    )
    backend = SimpleNamespace(
        _metadata=_gate_metadata(q_seq_lens) if metadata is None else metadata,
        _kv_caches=[LayerCache(key=None, value=None, conv=conv_cache, ssm=ssm_cache)],
    )
    cp_context = _toy_cp_context(q_seq_lens, cp_size, rank, device)
    group.begin_run(rank)
    output = KdaLinearAttentionMixin.execute_linear(
        backend, mixed_qkv, beta_raw, layer, raw_gate, cp_context=cp_context
    )
    return output, conv_cache, ssm_cache


@pytest.mark.parametrize("q_seq_lens", [[256], [512], [256, 512], [250]])
def _assert_cp_pipeline_matches_full_stream(
    q_seq_lens: list[int], monkeypatch: pytest.MonkeyPatch, force_gate: bool = False
) -> None:
    """The shared pipeline oracle: run every virtual rank through the real
    ``execute_linear`` CP branch and compare outputs, cache writes and the
    broadcast replica invariant against a full-stream reference run of the
    same reference kernels. The mock summary op composes the mock kernel's
    exact affine map, so the only divergence is fp32 re-association.

    ``force_gate`` bypasses the eligibility gate for layouts the gate
    excludes (the g2 ragged-tail fallbacks: their correctness must not
    rest on the gate's every-chunk-exactly-full invariant)."""
    _install_kda_cp_reference_kernels(monkeypatch)
    if force_gate:
        monkeypatch.setattr(kda_module, "kda_cp_prefill_eligible", lambda *args, **kwargs: True)
    device = torch.device("cpu")
    dtype = torch.float32
    generator = torch.Generator().manual_seed(20261015)
    cp_size = 2
    layer = _toy_layer(device, dtype, generator)
    num_heads, head_dim = layer.num_heads_local, layer.head_dim
    num_seqs = len(q_seq_lens)
    total = sum(q_seq_lens)

    global_qkv = torch.randn(total, layer.conv_dim, generator=generator).to(dtype)
    global_beta, global_gate = _model_isomorphic_gate_inputs(total, num_heads, head_dim, generator)
    global_beta = global_beta.to(dtype)
    global_gate = global_gate.to(dtype)

    # ---- full-stream reference (per-sequence conv + one KDA run) ---------------
    conv_state_ref = torch.zeros(num_seqs, 3, layer.conv_dim, dtype=dtype)
    cumulative = [0]
    for length in q_seq_lens:
        cumulative.append(cumulative[-1] + length)
    conv_ref = _reference_causal_conv1d_prefill(
        global_qkv,
        layer.conv_weight_t,
        conv_state_ref,
        torch.empty(0, dtype=torch.int64),
        torch.ones(num_seqs, dtype=torch.int64),
        torch.tensor(cumulative),
    )
    hidden_shape = (1, total, num_heads, head_dim)
    q_ref = l2_norm(conv_ref[:, : layer.qkv_dim].reshape(hidden_shape))
    k_ref = l2_norm(conv_ref[:, layer.qkv_dim : 2 * layer.qkv_dim].reshape(hidden_shape))
    v_ref = conv_ref[:, 2 * layer.qkv_dim :].reshape(hidden_shape).contiguous()
    beta_ref = global_beta.float().sigmoid().view(1, total, num_heads)
    gate_kwargs = dict(
        A_log=layer.forget_gate.A_log,
        dt_bias=layer.forget_gate.dt_bias,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    )
    out_ref, fin_ref = _reference_chunk_kda_fwd(
        q_ref,
        k_ref,
        v_ref,
        global_gate.view(hidden_shape),
        beta_ref,
        head_dim**-0.5,
        initial_state=torch.zeros(num_seqs, num_heads, head_dim, head_dim),
        cu_seqlens=cumulative,
        **gate_kwargs,
    )

    # ---- the fake group with precomputed halo payloads -------------------------
    halo_by_rank: dict[int, torch.Tensor] = {}
    for rank in range(cp_size):
        layout = _zigzag_layout(q_seq_lens, cp_size, rank)
        gather = [row if row >= 0 else 0 for row in layout["shard_index"]]
        rows = global_qkv.index_select(0, torch.tensor(gather, dtype=torch.int64))
        seg_cu = [0, *layout["q_cu_seqlens"]]
        payload = torch.zeros(2, num_seqs, 3, layer.conv_dim, dtype=dtype)
        for segment in range(2 * num_seqs):
            end = seg_cu[segment + 1]
            payload[segment % 2, segment // 2] = rows[end - 3 : end]
        halo_by_rank[rank] = payload
    group = _FakeCpGroup(cp_size, halo_by_rank)
    group.install(monkeypatch)

    # ---- rank-ordered rounds: halo gathers are exact from the start; summary
    # gathers fill in as ranks record theirs; the second round gives every
    # rank all summaries and the root's recorded broadcast payload ----------
    outputs: dict[int, torch.Tensor] = {}
    caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for _round in range(2):
        for rank in range(cp_size):
            outputs[rank], conv_cache, ssm_cache = _run_cp_rank(
                rank, group, q_seq_lens, layer, global_qkv, global_beta, global_gate, cp_size, device
            )
            caches[rank] = (conv_cache, ssm_cache)

    # ---- assertions -------------------------------------------------------------
    for rank in range(cp_size):
        layout = _zigzag_layout(q_seq_lens, cp_size, rank)
        shard = torch.tensor(layout["shard_index"], dtype=torch.int64)
        # Padding slots (-1) read as zero rows: the CP output contract
        # zeroes them (only the ragged-tail layouts carry padding).
        real_mask = shard >= 0
        expected_rows = out_ref[0].index_select(0, shard.clamp(min=0)) * real_mask.view(-1, 1, 1)
        actual_rows = outputs[rank][0]
        assert tuple(actual_rows.shape) == (layout["total_local"], num_heads, head_dim)
        # The very first segment of the first sequence has no prior chunks:
        # it matches the reference to fp32 rounding (the batched replay's
        # einsum blocking can differ from the sliced reference call by 1 ulp).
        # Every other segment composes prior summaries in fp32: tight
        # tolerance (the mock maps are exact, only re-association differs).
        first_seg_len = layout["q_cu_seqlens"][0]
        torch.testing.assert_close(actual_rows[:first_seg_len], expected_rows[:first_seg_len], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_rows, expected_rows, rtol=1e-4, atol=1e-5)

        conv_cache, ssm_cache = caches[rank]
        # Conv final state == each sequence's full-stream rolling window.
        for seq, length in enumerate(q_seq_lens):
            start = cumulative[seq]
            expected_window = torch.cat(
                (torch.zeros(3, layer.conv_dim, dtype=dtype), global_qkv[start : start + length])
            )[-3:]
            torch.testing.assert_close(conv_cache[seq, :3], expected_window, rtol=0, atol=0)
        # Committed columns only: the speculative checkpoint column (row 3 of
        # the conv cache, rows 1.. of the SSM cache) is never touched.
        assert torch.equal(conv_cache[:, 3:], torch.full_like(conv_cache[:, 3:], 7.0))
        assert torch.equal(ssm_cache[1::2], torch.full_like(ssm_cache[1::2], 9.0))
        # SSM final state == the reference final state (fp32 compose).
        torch.testing.assert_close(ssm_cache[0::2], fin_ref, rtol=1e-4, atol=1e-5)

    # The replica invariant: both ranks wrote the SAME broadcast tensors.
    torch.testing.assert_close(caches[0][0], caches[1][0], rtol=0, atol=0)
    torch.testing.assert_close(caches[0][1], caches[1][1], rtol=0, atol=0)
    # The broadcast root was the causally-last owner (rank 0), every time.
    assert group.broadcast_sources and set(group.broadcast_sources) == {0}


@pytest.mark.parametrize("q_seq_lens", [[256], [512], [256, 512]])
def test_kda_cp_execute_linear_pipeline_matches_full_stream(
    q_seq_lens: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate-passing layouts (every segment exactly full): the packed
    (sequence, half) replay order IS the local row layout, so the outputs
    return directly (g2 identity path) with the cache writes and the
    replica invariant matching the full-stream reference."""
    _assert_cp_pipeline_matches_full_stream(q_seq_lens, monkeypatch)


@pytest.mark.parametrize("q_seq_lens", [[250]])
def test_kda_cp_execute_linear_pipeline_ragged_tail_fallback(
    q_seq_lens: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """g2 guarded fallbacks on a ragged tail the eligibility gate excludes
    (the gate forces every chunk exactly full, so this layout can only
    reach the CP path through the forced gate here -- exactly the
    future-layout change the guards exist for). The causally-last chunk
    carries 61 of its 63 slot rows, so its owner (rank 0, the early rank)
    reassembles through the composed query_index row packs while rank 1
    stays fully tiled; every output row -- including the zeroed padding
    rows -- must land exactly where the old reassemble-then-scatter chain
    put it (verified: the pre-g2 code passes this same oracle).

    Mixed-chunk-length ragged layouts ([250, 514]-class) are deliberately
    NOT here: the two-pass CP scheme itself diverges from the full-stream
    reference on them (pre-existing, identical before and after g2 --
    their per-segment 64-row chunk grids are not 64-multiples), so the
    oracle cannot judge the fallback there. The gate excludes every
    ragged layout; this test pins the fallback's row placement, not the
    scheme's numerics."""
    _assert_cp_pipeline_matches_full_stream(q_seq_lens, monkeypatch, force_gate=True)


def test_kda_cp_execute_linear_pipeline_head_dim_128(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full CP pipeline at the ENGINE head dim (K = V = 128), where the
    fold selection picks the MERGE OP path (the head_dim-8 pipeline tests
    above exercise the tree-scan fallback). The pipeline must reproduce
    the full-stream reference within the existing re-association
    tolerances, and the replica invariant must hold across ranks. The
    merge/scan/sequential folds' pairwise 1-ulp-class agreement is pinned
    separately by the pure fold identity tests."""
    _install_kda_cp_reference_kernels(monkeypatch)
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(20261019)
    cp_size = 2
    layer = _toy_layer(device, torch.float32, generator, head_dim=128)
    num_heads, head_dim = layer.num_heads_local, layer.head_dim
    q_seq_lens = [256]
    num_seqs = len(q_seq_lens)
    total = sum(q_seq_lens)

    global_qkv = torch.randn(total, layer.conv_dim, generator=generator)
    global_beta, global_gate = _model_isomorphic_gate_inputs(total, num_heads, head_dim, generator)

    # Full-stream reference at the same head dim.
    conv_state_ref = torch.zeros(num_seqs, 3, layer.conv_dim)
    cumulative = [0, total]
    conv_ref = _reference_causal_conv1d_prefill(
        global_qkv,
        layer.conv_weight_t,
        conv_state_ref,
        torch.empty(0, dtype=torch.int64),
        torch.ones(num_seqs, dtype=torch.int64),
        torch.tensor(cumulative),
    )
    hidden_shape = (1, total, num_heads, head_dim)
    q_ref = l2_norm(conv_ref[:, : layer.qkv_dim].reshape(hidden_shape))
    k_ref = l2_norm(conv_ref[:, layer.qkv_dim : 2 * layer.qkv_dim].reshape(hidden_shape))
    v_ref = conv_ref[:, 2 * layer.qkv_dim :].reshape(hidden_shape).contiguous()
    out_ref, _fin_ref = _reference_chunk_kda_fwd(
        q_ref,
        k_ref,
        v_ref,
        global_gate.view(hidden_shape),
        global_beta.float().sigmoid().view(1, total, num_heads),
        head_dim**-0.5,
        initial_state=torch.zeros(num_seqs, num_heads, head_dim, head_dim),
        cu_seqlens=cumulative,
        A_log=layer.forget_gate.A_log,
        dt_bias=layer.forget_gate.dt_bias,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    )

    halo_by_rank: dict[int, torch.Tensor] = {}
    for rank in range(cp_size):
        layout = _zigzag_layout(q_seq_lens, cp_size, rank)
        gather = [row if row >= 0 else 0 for row in layout["shard_index"]]
        rows = global_qkv.index_select(0, torch.tensor(gather, dtype=torch.int64))
        seg_cu = [0, *layout["q_cu_seqlens"]]
        payload = torch.zeros(2, num_seqs, 3, layer.conv_dim)
        for segment in range(2 * num_seqs):
            end = seg_cu[segment + 1]
            payload[segment % 2, segment // 2] = rows[end - 3 : end]
        halo_by_rank[rank] = payload
    group = _FakeCpGroup(cp_size, halo_by_rank)
    group.install(monkeypatch)

    outputs: dict[int, torch.Tensor] = {}
    caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for _round in range(2):
        for rank in range(cp_size):
            outputs[rank], conv_cache, ssm_cache = _run_cp_rank(
                rank, group, q_seq_lens, layer, global_qkv, global_beta, global_gate, cp_size, device
            )
            caches[rank] = (conv_cache, ssm_cache)

    # 1. The pipeline reproduces the full-stream reference within the same
    #    re-association tolerances the head_dim-8 pipeline test pins.
    for rank in range(cp_size):
        layout = _zigzag_layout(q_seq_lens, cp_size, rank)
        shard = torch.tensor(layout["shard_index"], dtype=torch.int64)
        expected_rows = out_ref[0].index_select(0, shard)
        torch.testing.assert_close(outputs[rank][0], expected_rows, rtol=1e-4, atol=1e-5)
    # 2. The replica invariant holds across ranks.
    torch.testing.assert_close(caches[0][0], caches[1][0], rtol=0, atol=0)
    torch.testing.assert_close(caches[0][1], caches[1][1], rtol=0, atol=0)


def test_kda_cp_execute_linear_emits_one_time_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The alpha-style probe fires exactly once per process with the active
    segment/row counts and the merge source."""
    _install_kda_cp_reference_kernels(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(
        kda_module,
        "logger",
        SimpleNamespace(
            info=lambda message: calls.append(message),
            warning=lambda message: calls.append(message),
        ),
    )
    monkeypatch.setattr(kda_module, "_KDA_CP_PROBE_EMITTED", False)
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(3)
    layer = _toy_layer(device, torch.float32, generator)
    total = 256
    global_qkv = torch.randn(total, layer.conv_dim, generator=generator)
    global_beta = torch.zeros(total, layer.num_heads_local) - 3.0
    global_gate = torch.randn(total, layer.num_heads_local, layer.head_dim, generator=generator)

    halo = torch.zeros(2, 1, 3, layer.conv_dim)
    group = _FakeCpGroup(2, {0: halo, 1: halo})
    group.install(monkeypatch)
    _run_cp_rank(0, group, [total], layer, global_qkv, global_beta, global_gate, 2, device)
    _run_cp_rank(0, group, [total], layer, global_qkv, global_beta, global_gate, 2, device)
    probes = [message for message in calls if message.startswith("KDA_CP_ACTIVE")]
    assert len(probes) == 1
    # head_dim 8 selects the tree-scan fold (K != 128); the descriptor cache
    # is on by default.
    assert "segments=2" in probes[0] and "rows=128" in probes[0] and "msrc=prepare1+ppfm1+scan" in probes[0]
    assert "desccache=on" in probes[0]


def test_kda_cp_execute_linear_batches_segment_factor_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """M-B2 call structure: each forward issues exactly ONE prepare and ONE
    ppfm call over the rank's whole local stream -- cu_seqlens is the
    combined seg_cu whose canonical chunk_indices enumerate every segment's
    64-row chunks, the factor tensors span all local rows, and the batched
    hm rows keep the (sequence, half) order the all-gather packing and the
    causal merge consume (half-major [2, n_seqs, ...] contribution, row
    2*seq+half == that segment's fold, bit-equal to the per-segment call
    the pre-M-B2 loop made)."""
    _install_kda_cp_reference_kernels(monkeypatch)
    prepare_calls: list[dict[str, Any]] = []
    ppfm_calls: list[dict[str, Any]] = []

    def _recording_prepare(*args: Any, **kwargs: Any) -> Any:
        result = _reference_prepare(*args, **kwargs)
        prepare_calls.append({"args": args, "kwargs": kwargs, "result": result})
        return result

    def _recording_ppfm(*args: Any, **kwargs: Any) -> Any:
        result = _reference_ppfm(*args, **kwargs)
        ppfm_calls.append({"args": args, "kwargs": kwargs, "result": result})
        return result

    def _recording_merge(*args: Any, **kwargs: Any) -> Any:
        # head_dim 8 selects the tree-scan fold, so the merge op must never
        # be reached in this test; the recorder would surface any fold-mode
        # regression loudly instead of silently changing the call structure.
        raise AssertionError("merge fold op reached in a head_dim-8 pipeline (fold mode regression)")

    monkeypatch.setattr(kda_module, "_kda_cp_ops", lambda: (_recording_prepare, _recording_ppfm, _recording_merge))

    q_seq_lens = [256, 512]
    cp_size = 2
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(29)
    layer = _toy_layer(device, torch.float32, generator)
    num_heads, head_dim = layer.num_heads_local, layer.head_dim
    num_seqs = len(q_seq_lens)
    total = sum(q_seq_lens)
    global_qkv = torch.randn(total, layer.conv_dim, generator=generator)
    global_beta, global_gate = _model_isomorphic_gate_inputs(total, num_heads, head_dim, generator)

    halo_by_rank: dict[int, torch.Tensor] = {}
    for rank in range(cp_size):
        layout = _zigzag_layout(q_seq_lens, cp_size, rank)
        gather = [row if row >= 0 else 0 for row in layout["shard_index"]]
        rows = global_qkv.index_select(0, torch.tensor(gather, dtype=torch.int64))
        seg_cu = [0, *layout["q_cu_seqlens"]]
        payload = torch.zeros(2, num_seqs, 3, layer.conv_dim)
        for segment in range(2 * num_seqs):
            end = seg_cu[segment + 1]
            payload[segment % 2, segment // 2] = rows[end - 3 : end]
        halo_by_rank[rank] = payload
    group = _FakeCpGroup(cp_size, halo_by_rank)
    group.install(monkeypatch)

    for rank in range(cp_size):
        _run_cp_rank(rank, group, q_seq_lens, layer, global_qkv, global_beta, global_gate, cp_size, device)

    # One prepare + one ppfm per forward (two virtual ranks -> two runs).
    assert len(prepare_calls) == cp_size
    assert len(ppfm_calls) == cp_size
    for rank, (prepare_call, ppfm_call) in enumerate(zip(prepare_calls, ppfm_calls)):
        layout = _zigzag_layout(q_seq_lens, cp_size, rank)
        seg_cu = [0, *layout["q_cu_seqlens"]]
        total_real = layout["total_local"]
        # ONE prepare over the whole local stream: full-row inputs and the
        # combined cu_seqlens (the per-segment loop passed [0, seg_len]
        # slices).
        query = prepare_call["args"][0]
        assert tuple(query.shape) == (1, total_real, num_heads, head_dim)
        assert prepare_call["kwargs"]["cu_seqlens"] == seg_cu
        # The canonical chunk_indices the wrapper derives from that cu
        # cover BOTH zigzag segments of every sequence (sequence-major,
        # 64-row grid restarting at every boundary; flat (seq, chunk)
        # pairs).
        expected_indices = tuple(
            value
            for segment in range(2 * num_seqs)
            for chunk in range((seg_cu[segment + 1] - seg_cu[segment]) // KDA_CP_CHUNK_SIZE)
            for value in (segment, chunk)
        )
        assert kda_cp_ops._kda_build_chunk_indices(seg_cu, KDA_CP_CHUNK_SIZE) == expected_indices
        # ONE ppfm on the batched factors with the same combined cu.
        assert ppfm_call["kwargs"]["cu_seqlens"] == seg_cu
        kg, w, u = ppfm_call["args"]
        assert tuple(kg.shape) == (1, num_heads, total_real, head_dim)
        gk = ppfm_call["kwargs"]["gk"]
        # hm layout: rows are (sequence, half)-interleaved and row i equals
        # the per-segment fold of rows [seg_cu[i], seg_cu[i+1]) -- the exact
        # values the pre-M-B2 per-segment calls produced.
        hm = ppfm_call["result"]
        assert tuple(hm.shape) == (2 * num_seqs, num_heads, head_dim, 2 * head_dim)
        for segment in range(2 * num_seqs):
            start, end = seg_cu[segment], seg_cu[segment + 1]
            expected = _reference_ppfm(
                kg[:, :, start:end],
                w[:, :, start:end],
                u[:, :, start:end],
                gk=gk[:, :, start:end],
                cu_seqlens=[0, end - start],
                chunk_size=KDA_CP_CHUNK_SIZE,
            )
            torch.testing.assert_close(hm[segment], expected[0], rtol=0, atol=0)
        # The all-gather contribution keeps the half-major [2, n_seqs, ...]
        # layout the gathered-tensor row mapping (rank*2 + half) expects.
        contribution = group.summaries_by_rank[rank]
        assert tuple(contribution.shape) == (2, num_seqs, num_heads, head_dim, 2 * head_dim)
        torch.testing.assert_close(contribution, hm.view(num_seqs, 2, *hm.shape[1:]).transpose(0, 1), rtol=0, atol=0)


def test_kda_cp_conv_call_metadata_is_host_side(monkeypatch: pytest.MonkeyPatch) -> None:
    """n6: the CP conv call passes its qsl/si/ism metadata as CPU int64
    tensors -- the C++ wrapper's ``.to(torch::kCPU, torch::kInt64)`` is a
    no-op on exactly that form, so the marshalling cannot block the host
    behind the compute stream. The VALUES stay identical to the device
    form the g1 path filled: qsl == the rank's seg_cu, si empty, ism ==
    all ones except rank 0's first-half rows carrying the warm values
    (cold mode, the m1 caveat)."""
    _install_kda_cp_reference_kernels(monkeypatch)
    conv_calls: list[dict[str, Any]] = []

    def _recording_conv(
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_state: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> torch.Tensor:
        conv_calls.append(
            {
                "state_indices": state_indices,
                "has_initial_state": has_initial_state,
                "query_start_loc": query_start_loc,
            }
        )
        return _reference_causal_conv1d_prefill(
            x, weight, conv_state, state_indices, has_initial_state, query_start_loc
        )

    monkeypatch.setattr(torch.ops.xllm_ops, "causal_conv1d_prefill", _recording_conv, raising=False)

    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(31)
    layer = _toy_layer(device, torch.float32, generator)
    q_seq_lens = [256, 512]
    cp_size = 2
    num_seqs = len(q_seq_lens)
    total = sum(q_seq_lens)
    global_qkv = torch.randn(total, layer.conv_dim, generator=generator)
    global_beta, global_gate = _model_isomorphic_gate_inputs(total, layer.num_heads_local, layer.head_dim, generator)

    halo_by_rank: dict[int, torch.Tensor] = {}
    for rank in range(cp_size):
        layout = _zigzag_layout(q_seq_lens, cp_size, rank)
        gather = [row if row >= 0 else 0 for row in layout["shard_index"]]
        rows = global_qkv.index_select(0, torch.tensor(gather, dtype=torch.int64))
        seg_cu = [0, *layout["q_cu_seqlens"]]
        payload = torch.zeros(2, num_seqs, 3, layer.conv_dim)
        for segment in range(2 * num_seqs):
            end = seg_cu[segment + 1]
            payload[segment % 2, segment // 2] = rows[end - 3 : end]
        halo_by_rank[rank] = payload
    group = _FakeCpGroup(cp_size, halo_by_rank)
    group.install(monkeypatch)

    # Cold default metadata (the gate metadata's zero warm mask).
    _run_cp_rank(0, group, q_seq_lens, layer, global_qkv, global_beta, global_gate, cp_size, device)
    # Warm override exercising the host twin: sequences [warm, cold].
    warm_metadata = _gate_metadata(q_seq_lens)
    warm_metadata.has_initial_state = torch.tensor([1, 0], dtype=torch.int64)
    warm_metadata.linear_state_validity_host_values = [1, 0]
    _run_cp_rank(0, group, q_seq_lens, layer, global_qkv, global_beta, global_gate, cp_size, device, warm_metadata)

    assert len(conv_calls) == 2
    for call, warm_values in zip(conv_calls, ([0, 0], [1, 0]), strict=True):
        for name in ("state_indices", "has_initial_state", "query_start_loc"):
            tensor = call[name]
            assert torch.is_tensor(tensor), f"{name} must stay a Tensor (the op ABI)"
            assert tensor.device.type == "cpu", f"{name} must be host-side (got {tensor.device})"
            assert tensor.dtype == torch.int64, f"{name} must be int64 (got {tensor.dtype})"
        layout = _zigzag_layout(q_seq_lens, cp_size, 0)
        assert call["query_start_loc"].tolist() == [0, *layout["q_cu_seqlens"]]
        assert call["state_indices"].numel() == 0
        # Rank 0: first-half rows carry the warm values, second halves warm.
        ism = call["has_initial_state"].tolist()
        assert len(ism) == 2 * num_seqs
        assert ism[0::2] == warm_values
        assert ism[1::2] == [1] * num_seqs

    # A non-zero-prior rank passes all-ones ism with the same host forms.
    conv_calls.clear()
    _run_cp_rank(1, group, q_seq_lens, layer, global_qkv, global_beta, global_gate, cp_size, device, warm_metadata)
    assert len(conv_calls) == 1
    call = conv_calls[0]
    assert call["has_initial_state"].device.type == "cpu"
    assert call["has_initial_state"].tolist() == [1] * (2 * num_seqs)
    assert call["query_start_loc"].device.type == "cpu"
    assert call["state_indices"].device.type == "cpu"


def test_kda_cp_execute_linear_rejects_ineligible_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cp_context handed over without the model gate having passed is a
    contract violation and fails loudly (never silently misreads rows)."""
    _install_kda_cp_reference_kernels(monkeypatch)
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(5)
    layer = _toy_layer(device, torch.float32, generator)
    conv_cache = torch.zeros(1, 4, layer.conv_dim)
    ssm_cache = torch.zeros(2, layer.num_heads_local, layer.head_dim, layer.head_dim)
    backend = SimpleNamespace(
        _metadata=_gate_metadata([4]),  # short segments: ineligible
        _kv_caches=[LayerCache(key=None, value=None, conv=conv_cache, ssm=ssm_cache)],
    )
    cp_context = _toy_cp_context([4], 2, 0, device)
    with pytest.raises(RuntimeError, match="ineligible"):
        KdaLinearAttentionMixin.execute_linear(
            backend,
            torch.zeros(1, layer.conv_dim, 4),
            torch.zeros(1, 4, layer.num_heads_local),
            layer,
            torch.zeros(1, 4, layer.num_heads_local, layer.head_dim),
            cp_context=cp_context,
        )


# ---------------------------------------------------------------------------
# M-B transport reductions: pure helpers + transport-selection coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_kda_cp_zero_prior_segments_selection(cp_size: int) -> None:
    """Only rank 0's first halves have zero priors (their initial state is
    the local prefix/zero base); every other local segment composes at
    least one prior summary and must wait for the gather."""
    for rank in range(cp_size):
        segments = _kda_cp_zero_prior_segments(rank, cp_size, 3)
        if rank == 0:
            assert segments == [0, 2, 4]
        else:
            assert segments == []


def test_kda_cp_state_broadcast_pack_round_trip() -> None:
    """The merged broadcast payload round-trips conv+ssm bit-exactly for
    every conv dtype the engine can run (bf16/fp16/fp32 values are exactly
    representable in fp32); fp64 refuses and falls back to two
    broadcasts."""
    generator = torch.Generator().manual_seed(19)
    ssm = torch.randn(3, 2, 4, 4, generator=generator)
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        conv = (torch.randn(3, 2, 5, generator=generator) * 64.0).to(dtype)
        packed = _kda_cp_pack_state_broadcast(conv, ssm)
        assert packed is not None
        flat, conv_elements = packed
        assert flat.dtype == torch.float32 and conv_elements == conv.numel()
        assert flat.numel() == conv.numel() + ssm.numel()
        got_conv, got_ssm = _kda_cp_unpack_state_broadcast(flat, conv_elements, dtype, conv.shape, ssm.shape)
        torch.testing.assert_close(got_conv, conv, rtol=0, atol=0)
        torch.testing.assert_close(got_ssm, ssm, rtol=0, atol=0)
    assert _kda_cp_pack_state_broadcast(ssm.double(), ssm.double()) is None


def _run_cp_pipeline_for_transport(
    q_seq_lens: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    _FakeCpGroup, dict[int, torch.Tensor], dict[int, tuple[torch.Tensor, torch.Tensor]], list[list[tuple[str, int]]]
]:
    """The transport pipeline test's body; returns the group, per-rank
    outputs, per-rank caches and the per-run event slices of the FINAL
    round (for ordering assertions)."""
    _install_kda_cp_reference_kernels(monkeypatch)
    device = torch.device("cpu")
    dtype = torch.float32
    generator = torch.Generator().manual_seed(20261015)
    cp_size = 2
    layer = _toy_layer(device, dtype, generator)
    num_heads, head_dim = layer.num_heads_local, layer.head_dim
    num_seqs = len(q_seq_lens)
    total = sum(q_seq_lens)

    global_qkv = torch.randn(total, layer.conv_dim, generator=generator).to(dtype)
    global_beta, global_gate = _model_isomorphic_gate_inputs(total, num_heads, head_dim, generator)
    global_beta = global_beta.to(dtype)
    global_gate = global_gate.to(dtype)

    halo_by_rank: dict[int, torch.Tensor] = {}
    for rank in range(cp_size):
        layout = _zigzag_layout(q_seq_lens, cp_size, rank)
        gather = [row if row >= 0 else 0 for row in layout["shard_index"]]
        rows = global_qkv.index_select(0, torch.tensor(gather, dtype=torch.int64))
        seg_cu = [0, *layout["q_cu_seqlens"]]
        payload = torch.zeros(2, num_seqs, 3, layer.conv_dim, dtype=dtype)
        for segment in range(2 * num_seqs):
            end = seg_cu[segment + 1]
            payload[segment % 2, segment // 2] = rows[end - 3 : end]
        halo_by_rank[rank] = payload
    group = _FakeCpGroup(cp_size, halo_by_rank)
    group.install(monkeypatch)

    # Record replay calls into the group's event log (ordering evidence).
    ascendc = sys.modules["fla_npu.ops.ascendc"]
    reference_kernel = ascendc.chunk_kda_fwd

    def _recording_kernel(*args: Any, **kwargs: Any) -> Any:
        group.events.append(("replay", group.rank))
        return reference_kernel(*args, **kwargs)

    monkeypatch.setattr(ascendc, "chunk_kda_fwd", _recording_kernel)

    outputs: dict[int, torch.Tensor] = {}
    caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    final_round_events: list[list[tuple[str, int]]] = [[] for _ in range(cp_size)]
    for _round in range(2):
        for rank in range(cp_size):
            outputs[rank], conv_cache, ssm_cache = _run_cp_rank(
                rank, group, q_seq_lens, layer, global_qkv, global_beta, global_gate, cp_size, device
            )
            caches[rank] = (conv_cache, ssm_cache)
            final_round_events[rank] = group.events[group.events_marker :]
    return group, outputs, caches, final_round_events


def _assert_transport_pipeline_matches_reference(
    q_seq_lens: list[int],
    outputs: dict[int, torch.Tensor],
    caches: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    """The same numeric gate the default-transport pipeline test asserts."""
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(20261015)
    cp_size = 2
    layer = _toy_layer(device, torch.float32, generator)
    num_heads, head_dim = layer.num_heads_local, layer.head_dim
    num_seqs = len(q_seq_lens)
    total = sum(q_seq_lens)
    global_qkv = torch.randn(total, layer.conv_dim, generator=generator)
    global_beta, global_gate = _model_isomorphic_gate_inputs(total, num_heads, head_dim, generator)

    conv_state_ref = torch.zeros(num_seqs, 3, layer.conv_dim)
    cumulative = [0]
    for length in q_seq_lens:
        cumulative.append(cumulative[-1] + length)
    conv_ref = _reference_causal_conv1d_prefill(
        global_qkv,
        layer.conv_weight_t,
        conv_state_ref,
        torch.empty(0, dtype=torch.int64),
        torch.ones(num_seqs, dtype=torch.int64),
        torch.tensor(cumulative),
    )
    hidden_shape = (1, total, num_heads, head_dim)
    q_ref = l2_norm(conv_ref[:, : layer.qkv_dim].reshape(hidden_shape))
    k_ref = l2_norm(conv_ref[:, layer.qkv_dim : 2 * layer.qkv_dim].reshape(hidden_shape))
    v_ref = conv_ref[:, 2 * layer.qkv_dim :].reshape(hidden_shape).contiguous()
    out_ref, fin_ref = _reference_chunk_kda_fwd(
        q_ref,
        k_ref,
        v_ref,
        global_gate.view(hidden_shape),
        global_beta.float().sigmoid().view(1, total, num_heads),
        head_dim**-0.5,
        initial_state=torch.zeros(num_seqs, num_heads, head_dim, head_dim),
        cu_seqlens=cumulative,
        A_log=layer.forget_gate.A_log,
        dt_bias=layer.forget_gate.dt_bias,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    )
    for rank in range(cp_size):
        layout = _zigzag_layout(q_seq_lens, cp_size, rank)
        shard = torch.tensor(layout["shard_index"], dtype=torch.int64)
        expected_rows = out_ref[0].index_select(0, shard)
        torch.testing.assert_close(outputs[rank][0], expected_rows, rtol=1e-4, atol=1e-5)
        conv_cache, ssm_cache = caches[rank]
        torch.testing.assert_close(ssm_cache[0::2], fin_ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(caches[0][0], caches[1][0], rtol=0, atol=0)
    torch.testing.assert_close(caches[0][1], caches[1][1], rtol=0, atol=0)


def test_kda_cp_pipeline_transport_is_async_with_merged_broadcast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CP transport: both gathers go through
    torch.distributed.all_gather_into_tensor(async_op=True) with the
    handle waited (never the synchronous graph-op gathers), the final
    states travel in ONE merged broadcast, and the zero-prior replay of
    rank 0 runs BETWEEN the summary issue and its wait (the overlap the
    reduction exists for); rank 1 has no zero-prior segment and waits
    before its single replay."""
    group, outputs, caches, events = _run_cp_pipeline_for_transport([256], monkeypatch)
    kinds = {kind for kind, _ in group.events}
    assert "halo_issue" in kinds and "summary_issue" in kinds
    assert not any(kind.endswith("_sync") for kind in kinds)

    rank0 = [kind for kind, rank in events[0] if rank == 0]
    assert rank0 == ["halo_issue", "halo_wait", "summary_issue", "replay", "summary_wait", "replay"]
    rank1 = [kind for kind, rank in events[1] if rank == 1]
    assert rank1 == ["halo_issue", "halo_wait", "summary_issue", "summary_wait", "replay"]

    # ONE merged broadcast per run (two ranks x two runs).
    assert group.broadcast_calls == 4
    assert set(group.broadcast_sources) == {0}
    _assert_transport_pipeline_matches_reference([256], outputs, caches)


# ---------------------------------------------------------------------------
# NPU-gated tests: the real two-pass pipeline vs a full-stream run
# ---------------------------------------------------------------------------


def _import_npu_kda_ops() -> tuple[Any, Any, Any] | None:
    """Resolve (chunk_kda_fwd, conv_op, kda_cp_ops module) on the NPU stack.

    The fused replay/reference op and the conv op come from the DEPLOYED
    fla_npu wheel (plain import); the summary factor pass and the ppfm fold
    come from OUR standard ``custom_xllm_math`` opp vendor through
    ``kda_cp_ops`` (resolved from ``${ASCEND_HOME_PATH}/opp/vendors``).
    Returns None (skip) when either source is unavailable.
    """
    try:
        from fla_npu.ops.ascendc import chunk_kda_fwd, npu_causal_conv1d  # type: ignore[attr-defined]
    except Exception:
        return None
    if not kda_cp_ops.available():
        return None
    return chunk_kda_fwd, npu_causal_conv1d, kda_cp_ops


@pytest.mark.skipif(not os.getenv("XLLM_KDA_TEST_NPU_DEVICE"), reason="NPU device not configured")
def test_kda_cp_prepare_factors_match_fused_emission_npu() -> None:
    """ma1 factor-equivalence contract, now through OUR wrapper: prepare
    factors vs the deployed fused ``disable_recompute`` emission on the
    same raw inputs differ only at LAST-BIT rounding scale -- gk <= 2e-7
    abs (~1 fp32 ulp), bf16 slots <= 1e-3 abs (~1 bf16 ulp) -- and the hm
    summary built from them shifts <= 3e-4 rel_fro (ma1 measured ~1.7e-4,
    4x inside the accepted cp4 compose divergence)."""
    ops = _import_npu_kda_ops()
    if ops is None:
        pytest.skip("fla_npu or xllm KDA CP ops (standard custom_xllm_math vendor) not available")
    chunk_kda_fwd, _conv_op, cp_ops = ops

    device = torch.device(os.environ["XLLM_KDA_TEST_NPU_DEVICE"])
    torch.npu.set_device(device)
    generator = torch.Generator().manual_seed(20261015)
    num_heads, head_dim = 2, 128
    qkv_dim = num_heads * head_dim
    conv_dim = 3 * qkv_dim
    total = 256
    global_qkv = torch.randn(total, conv_dim, generator=generator).to(device=device, dtype=torch.bfloat16)
    beta_raw_cpu, gate_raw_cpu = _model_isomorphic_gate_inputs(total, num_heads, head_dim, generator)
    global_beta = beta_raw_cpu.to(device=device, dtype=torch.float32)
    global_gate = gate_raw_cpu.to(device=device, dtype=torch.bfloat16)
    A_log = torch.zeros(num_heads, dtype=torch.float32, device=device)
    dt_bias = torch.zeros(num_heads * head_dim, dtype=torch.float32, device=device)
    gate_kwargs = dict(
        A_log=A_log,
        dt_bias=dt_bias,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    )

    # Conv the same way the two-pass test does, then split q/k/v.
    conv_state = torch.zeros(1, 3, conv_dim, dtype=torch.bfloat16, device=device)
    from fla_npu.ops.ascendc import npu_causal_conv1d

    conv_out = npu_causal_conv1d(
        global_qkv,
        torch.randn(4, conv_dim, generator=generator).to(device=device, dtype=torch.bfloat16) * 0.5,
        None,
        conv_state,
        query_start_loc=[0, total],
        initial_state_mode=[0],
        activation_mode=1,
        run_mode=0,
    )
    hidden_shape = (1, total, num_heads, head_dim)
    q_in = l2_norm(conv_out[:, :qkv_dim].reshape(hidden_shape))
    k_in = l2_norm(conv_out[:, qkv_dim : 2 * qkv_dim].reshape(hidden_shape))
    v_in = conv_out[:, 2 * qkv_dim :].reshape(hidden_shape).contiguous()
    g_all = global_gate.view(hidden_shape)
    beta_all = global_beta.view(1, total, num_heads).sigmoid()

    fused = chunk_kda_fwd(
        q_in,
        k_in,
        v_in,
        g_all,
        beta_all,
        head_dim**-0.5,
        chunk_size=KDA_CP_CHUNK_SIZE,
        layout="BSND",
        initial_state=None,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, total], dtype=torch.int32, device=device),
        disable_recompute=True,
        state_v_first=True,
        **gate_kwargs,
    )
    prepare = cp_ops.xllm_chunk_kda_fwd_prepare(
        q_in,
        k_in,
        v_in,
        g_all,
        beta_all,
        head_dim**-0.5,
        chunk_size=KDA_CP_CHUNK_SIZE,
        layout="BSND",
        cu_seqlens=[0, total],
        backward_mode="none",
        **gate_kwargs,
    )
    torch.npu.synchronize()

    # ma1 fused->prepare slot correspondence: gk 2->0, w 5->3, u 6->4, kg 8->6.
    for fused_index, prepare_index, bound in ((2, 0, 2e-7), (5, 3, 1e-3), (6, 4, 1e-3), (8, 6, 1e-3)):
        diff = (prepare[prepare_index].float() - fused[fused_index].float()).abs().max().item()
        assert diff <= bound, (fused_index, prepare_index, diff)

    def _rel_fro(actual: torch.Tensor, expected: torch.Tensor) -> float:
        difference = (actual.float() - expected.float()).norm().item()
        return difference / max(expected.float().norm().item(), 1e-30)

    # Clone before the second ppfm call: both calls share one marshalling
    # signature, so with the descriptor cache on they share ONE static
    # output buffer -- without the clone the comparison below would be the
    # buffer against itself (vacuously 0).
    hm_prepare = (
        cp_ops.xllm_pre_process_fwd_kernel_merged(
            prepare[6],
            prepare[3],
            prepare[4],
            gk=prepare[0],
            cu_seqlens=[0, total],
            chunk_size=KDA_CP_CHUNK_SIZE,
        )
    ).clone()
    hm_fused = cp_ops.xllm_pre_process_fwd_kernel_merged(
        fused[8],
        fused[5],
        fused[6],
        gk=fused[2],
        cu_seqlens=[0, total],
        chunk_size=KDA_CP_CHUNK_SIZE,
    )
    torch.npu.synchronize()
    perturbation = _rel_fro(hm_prepare, hm_fused)
    assert perturbation <= 3e-4, perturbation


@pytest.mark.skipif(not os.getenv("XLLM_KDA_TEST_NPU_DEVICE"), reason="NPU device not configured")
@pytest.mark.parametrize("q_seq_lens", [[256], [512], [256, 512]])
def test_kda_cp_two_pass_matches_full_stream_npu(q_seq_lens: list[int], monkeypatch: pytest.MonkeyPatch) -> None:
    """Real-kernel tolerance test: the two-pass CP pipeline vs one
    full-stream chunk_kda_fwd call at toy shapes. The single-sequence and
    multi-sequence batches both run the M-B2 batched summary path (ONE
    prepare + ONE ppfm over the rank's whole local stream -- the [256,
    512] case folds four segments in one ppfm call).

    Expected magnitudes (task research m1 + ma1): the summary factors now
    come from OUR prepare op, which shifts every hm by ~1.7e-4 rel_fro vs
    the fused emission (1-ulp factor rounding), so NO segment is bit-exact
    anymore -- including the <= 1-prior ones that were exact under the
    fused factor pass. The compose divergence for >= 2 prior chunks
    (~6.6e-4 rel_fro) sits on top. Both are asserted well inside the fla
    8e-3 budget. The broadcast final state must land within ~2e-3 abs of
    the full-stream final (the replica/divergence bound).
    """
    ops = _import_npu_kda_ops()
    if ops is None:
        pytest.skip("fla_npu or xllm KDA CP ops (standard custom_xllm_math vendor) not available")
    chunk_kda_fwd, conv_op, cp_ops = ops

    device = torch.device(os.environ["XLLM_KDA_TEST_NPU_DEVICE"])
    torch.npu.set_device(device)
    generator = torch.Generator().manual_seed(20261015)
    cp_size = 2
    num_heads, head_dim = 2, 128
    qkv_dim = num_heads * head_dim
    conv_dim = 3 * qkv_dim
    # [256] at cp2 -> chunk_len 64: 4 zigzag chunks of exactly 1 KDA chunk
    # (the [512] / [256, 512] variants grow the segments to 2 KDA chunks
    # and add a second sequence -- every segment stays 64-aligned).
    num_seqs = len(q_seq_lens)
    total = sum(q_seq_lens)
    cumulative = [0]
    for length in q_seq_lens:
        cumulative.append(cumulative[-1] + length)
    layer = SimpleNamespace(
        layer_id=0,
        conv_kernel_size=4,
        conv_dim=conv_dim,
        head_dim=head_dim,
        num_heads_local=num_heads,
        qkv_dim=qkv_dim,
        activation="silu",
        conv_weight_t=(torch.randn(4, conv_dim, generator=generator) * 0.5).to(device=device, dtype=torch.bfloat16),
        forget_gate=SimpleNamespace(
            A_log=torch.zeros(num_heads, dtype=torch.float32, device=device),
            dt_bias=torch.zeros(num_heads * head_dim, dtype=torch.float32, device=device),
            safe_gate_lower_bound=-5.0,
        ),
    )
    # Model-isomorphic inputs (m0b/m1 discipline: small beta, mild gates).
    global_qkv = torch.randn(total, conv_dim, generator=generator).to(device=device, dtype=torch.bfloat16)
    beta_raw_cpu, gate_raw_cpu = _model_isomorphic_gate_inputs(total, num_heads, head_dim, generator)
    global_beta = beta_raw_cpu.to(device=device, dtype=torch.float32)
    global_gate = gate_raw_cpu.to(device=device, dtype=torch.bfloat16)

    # Full-stream reference: cold-mode conv + one chunk_kda_fwd call.
    conv_state_ref = torch.zeros(num_seqs, 3, conv_dim, dtype=torch.bfloat16, device=device)
    conv_ref = conv_op(
        global_qkv,
        layer.conv_weight_t,
        None,
        conv_state_ref,
        query_start_loc=cumulative,
        initial_state_mode=[0] * num_seqs,
        activation_mode=1,
        run_mode=0,
    )
    hidden_shape = (1, total, num_heads, head_dim)
    q_ref = l2_norm(conv_ref[:, :qkv_dim].reshape(hidden_shape))
    k_ref = l2_norm(conv_ref[:, qkv_dim : 2 * qkv_dim].reshape(hidden_shape))
    v_ref = conv_ref[:, 2 * qkv_dim :].reshape(hidden_shape).contiguous()
    gate_kwargs = dict(
        A_log=layer.forget_gate.A_log,
        dt_bias=layer.forget_gate.dt_bias,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    )
    reference = chunk_kda_fwd(
        q_ref,
        k_ref,
        v_ref,
        global_gate.view(hidden_shape),
        global_beta.view(1, total, num_heads).sigmoid(),
        head_dim**-0.5,
        chunk_size=KDA_CP_CHUNK_SIZE,
        layout="BSND",
        initial_state=torch.zeros(num_seqs, num_heads, head_dim, head_dim, device=device),
        output_final_state=True,
        cu_seqlens=cumulative,
        state_v_first=True,
        **gate_kwargs,
    )
    out_ref, fin_ref = reference[0], reference[1]

    # Shim causal_conv1d_prefill onto the fla conv op (the same
    # aclnnCausalConv1d the engine's xllm_ops wrapper calls; the reserve
    # container has no built xllm_ops library on the python path).
    def _conv_prefill_shim(x, weight, conv_state, state_indices, has_initial_state, query_start_loc):
        qsl = query_start_loc.tolist() if isinstance(query_start_loc, torch.Tensor) else list(query_start_loc)
        modes = has_initial_state.tolist() if torch.is_tensor(has_initial_state) else list(has_initial_state)
        return conv_op(
            x, weight, None, conv_state, query_start_loc=qsl, initial_state_mode=modes, activation_mode=1, run_mode=0
        )

    monkeypatch.setattr(torch.ops.xllm_ops, "causal_conv1d_prefill", _conv_prefill_shim, raising=False)
    monkeypatch.setattr(
        kda_module,
        "_kda_cp_ops",
        lambda: (
            cp_ops.xllm_chunk_kda_fwd_prepare,
            cp_ops.xllm_pre_process_fwd_kernel_merged,
            cp_ops.xllm_merge_fwd_bwd_kernel,
        ),
    )

    halo_by_rank: dict[int, torch.Tensor] = {}
    for rank in range(cp_size):
        layout = _zigzag_layout(q_seq_lens, cp_size, rank)
        gather = [row if row >= 0 else 0 for row in layout["shard_index"]]
        rows = global_qkv.index_select(0, torch.tensor(gather, dtype=torch.int64, device=device))
        seg_cu = [0, *layout["q_cu_seqlens"]]
        payload = torch.zeros(2, num_seqs, 3, conv_dim, dtype=torch.bfloat16, device=device)
        for segment in range(2 * num_seqs):
            end = seg_cu[segment + 1]
            payload[segment % 2, segment // 2] = rows[end - 3 : end]
        halo_by_rank[rank] = payload
    group = _FakeCpGroup(cp_size, halo_by_rank)
    group.install(monkeypatch)

    def _run_rank(rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layout = _zigzag_layout(q_seq_lens, cp_size, rank)
        gather = torch.tensor(
            [row if row >= 0 else 0 for row in layout["shard_index"]], dtype=torch.int64, device=device
        )
        mixed_qkv = global_qkv.index_select(0, gather).unsqueeze(0).transpose(1, 2).contiguous()
        beta_raw = global_beta.index_select(0, gather).unsqueeze(0).contiguous()
        raw_gate = global_gate.index_select(0, gather).unsqueeze(0).contiguous()
        conv_cache = torch.zeros(num_seqs, 4, conv_dim, dtype=torch.bfloat16, device=device)
        ssm_cache = torch.zeros(2 * num_seqs, num_heads, head_dim, head_dim, dtype=torch.float32, device=device)
        backend = SimpleNamespace(
            _metadata=_gate_metadata(q_seq_lens, device=device),
            _kv_caches=[LayerCache(key=None, value=None, conv=conv_cache, ssm=ssm_cache)],
        )
        group.begin_run(rank)
        output = KdaLinearAttentionMixin.execute_linear(
            backend, mixed_qkv, beta_raw, layer, raw_gate, cp_context=_toy_cp_context(q_seq_lens, cp_size, rank, device)
        )
        return output, conv_cache, ssm_cache

    # Rank-ordered rounds until every rank saw all summaries and the root's
    # valid broadcast payload (see _FakeCpGroup).
    outputs: dict[int, torch.Tensor] = {}
    caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for _round in range(2):
        for rank in range(cp_size):
            outputs[rank], conv_cache, ssm_cache = _run_rank(rank)
            caches[rank] = (conv_cache, ssm_cache)

    def _rel_fro(actual: torch.Tensor, expected: torch.Tensor) -> float:
        difference = (actual.float() - expected.float()).norm().item()
        return difference / max(expected.float().norm().item(), 1e-30)

    for rank in range(cp_size):
        layout = _zigzag_layout(q_seq_lens, cp_size, rank)
        shard = torch.tensor(layout["shard_index"], dtype=torch.int64, device=device)
        expected_rows = out_ref[0].index_select(0, shard)
        actual_rows = outputs[rank][0]
        # Per-segment comparison: prior-chunk counts come from the causal
        # position of each local segment (rank 0: chunks 0, 3; rank 1: 1, 2)
        # -- every sequence contributes a first/second-half segment pair.
        first_chunk = rank
        second_chunk = 2 * cp_size - 1 - rank
        seg_cu = [0, *layout["q_cu_seqlens"]]
        for segment in range(2 * num_seqs):
            chunk_id = first_chunk if segment % 2 == 0 else second_chunk
            start, end = seg_cu[segment], seg_cu[segment + 1]
            actual_seg = actual_rows[start:end]
            expected_seg = expected_rows[start:end]
            # <= 1 prior chunk: bounded by the prepare 1-ulp factor shift
            # (~1.7e-4 on hm); >= 2 prior chunks add the fp32 re-association
            # compose divergence (~6.6e-4). Both well inside the fla budget.
            bound = 1e-3 if chunk_id <= 1 else 8e-3
            rel = _rel_fro(actual_seg, expected_seg)
            assert rel < bound, (rank, segment, chunk_id, rel)
        conv_cache, ssm_cache = caches[rank]
        # Conv final state == each sequence's full-stream rolling window.
        for seq, length in enumerate(q_seq_lens):
            start = cumulative[seq]
            expected_window = torch.cat(
                (torch.zeros(3, conv_dim, dtype=torch.bfloat16, device=device), global_qkv[start : start + length])
            )[-3:]
            torch.testing.assert_close(conv_cache[seq, :3], expected_window, rtol=0, atol=0)
        fin_diff = (ssm_cache[0::2].float() - fin_ref.float()).abs().max().item()
        assert fin_diff < 2e-3, (rank, fin_diff)
    torch.testing.assert_close(caches[0][0], caches[1][0], rtol=0, atol=0)
    torch.testing.assert_close(caches[0][1], caches[1][1], rtol=0, atol=0)


@pytest.mark.skipif(not os.getenv("XLLM_KDA_TEST_NPU_DEVICE"), reason="NPU device not configured")
@pytest.mark.parametrize("seg_lens", [[128, 128], [128, 128, 256, 256]])
def test_kda_cp_batched_prepare_ppfm_match_per_segment_npu(seg_lens: list[int]) -> None:
    """M-B2 identity gate on the real kernels: ONE prepare + ONE ppfm call
    over a multi-segment local stream must produce BIT-IDENTICAL factors
    and hm rows to the per-segment calls the engine ran before M-B2 --
    same kernels, and cu_seqlens restarts the 64-row chunk grid at every
    segment boundary in both forms (the fused op's packed-varlen
    bit-exactness pin covers the prepare stage inside it). Any difference
    here is the M-B2 stop condition: document the diff, do not ship."""
    ops = _import_npu_kda_ops()
    if ops is None:
        pytest.skip("fla_npu or xllm KDA CP ops (standard custom_xllm_math vendor) not available")
    _chunk_kda_fwd, conv_op, cp_ops = ops

    device = torch.device(os.environ["XLLM_KDA_TEST_NPU_DEVICE"])
    torch.npu.set_device(device)
    generator = torch.Generator().manual_seed(20261016)
    num_heads, head_dim = 2, 128
    qkv_dim = num_heads * head_dim
    conv_dim = 3 * qkv_dim
    num_segments = len(seg_lens)
    total = sum(seg_lens)
    seg_cu = [0]
    for length in seg_lens:
        seg_cu.append(seg_cu[-1] + length)

    # Model-isomorphic inputs (the two-pass test's construction): the
    # segments are independent cold sequences, exactly the engine's local
    # (sequence, half) view of a zigzag rank.
    global_qkv = torch.randn(total, conv_dim, generator=generator).to(device=device, dtype=torch.bfloat16)
    beta_raw_cpu, gate_raw_cpu = _model_isomorphic_gate_inputs(total, num_heads, head_dim, generator)
    global_beta = beta_raw_cpu.to(device=device, dtype=torch.float32)
    global_gate = gate_raw_cpu.to(device=device, dtype=torch.bfloat16)
    conv_state = torch.zeros(num_segments, 3, conv_dim, dtype=torch.bfloat16, device=device)
    conv_out = conv_op(
        global_qkv,
        (torch.randn(4, conv_dim, generator=generator) * 0.5).to(device=device, dtype=torch.bfloat16),
        None,
        conv_state,
        query_start_loc=seg_cu,
        initial_state_mode=[0] * num_segments,
        activation_mode=1,
        run_mode=0,
    )
    hidden_shape = (1, total, num_heads, head_dim)
    q_in = l2_norm(conv_out[:, :qkv_dim].reshape(hidden_shape))
    k_in = l2_norm(conv_out[:, qkv_dim : 2 * qkv_dim].reshape(hidden_shape))
    v_in = conv_out[:, 2 * qkv_dim :].reshape(hidden_shape).contiguous()
    g_all = global_gate.view(hidden_shape)
    beta_all = global_beta.view(1, total, num_heads).sigmoid()
    gate_kwargs = dict(
        A_log=torch.zeros(num_heads, dtype=torch.float32, device=device),
        dt_bias=torch.zeros(num_heads * head_dim, dtype=torch.float32, device=device),
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    )

    def _prepare(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, cu: list[int]
    ) -> tuple[Any, ...]:
        return cp_ops.xllm_chunk_kda_fwd_prepare(
            q,
            k,
            v,
            g,
            beta,
            head_dim**-0.5,
            chunk_size=KDA_CP_CHUNK_SIZE,
            layout="BSND",
            cu_seqlens=cu,
            backward_mode="none",
            **gate_kwargs,
        )

    def _ppfm(factors: tuple[Any, ...], cu: list[int]) -> torch.Tensor:
        return cp_ops.xllm_pre_process_fwd_kernel_merged(
            factors[6],
            factors[3],
            factors[4],
            gk=factors[0],
            cu_seqlens=cu,
            chunk_size=KDA_CP_CHUNK_SIZE,
        )

    # ONE call each over the whole multi-segment stream (the M-B2 form).
    batched_factors = _prepare(
        q_in.contiguous(),
        k_in.contiguous(),
        v_in.contiguous(),
        g_all.contiguous(),
        beta_all.contiguous(),
        seg_cu,
    )
    batched_hm = _ppfm(batched_factors, seg_cu)
    torch.npu.synchronize()
    assert tuple(batched_hm.shape) == (num_segments, num_heads, head_dim, 2 * head_dim)

    # Per-segment calls (the pre-M-B2 engine form): row slices + cu=[0, len].
    for segment, length in enumerate(seg_lens):
        start, end = seg_cu[segment], seg_cu[segment + 1]
        seg_factors = _prepare(
            q_in[:, start:end].contiguous(),
            k_in[:, start:end].contiguous(),
            v_in[:, start:end].contiguous(),
            g_all[:, start:end].contiguous(),
            beta_all[:, start:end].contiguous(),
            [0, length],
        )
        seg_hm = _ppfm(seg_factors, [0, length])
        torch.npu.synchronize()
        # Every slot the "none" mode emits (gk 0, Aqk 1, w 3, u 4, kg 6,
        # qg_scaled 7) must be bit-identical sliced per segment, and the hm
        # row must equal the per-segment fold exactly.
        for slot in (0, 1, 3, 4, 6, 7):
            assert torch.equal(batched_factors[slot][:, :, start:end, :], seg_factors[slot]), (segment, slot)
        assert torch.equal(batched_hm[segment], seg_hm[0]), segment


@pytest.mark.skipif(not os.getenv("XLLM_KDA_TEST_NPU_DEVICE"), reason="NPU device not configured")
def test_kda_cp_merge_fold_matches_sequential_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    """N-1 fold gate on the REAL merge op: summaries from the real ppfm,
    folded by aclnnMergeFwdBwdKernel (through the full adapter: virtual
    base row + causal reorder + head flatten + state_v_first=True) vs the
    sequential python fold of the same summaries. This is the device pin
    for the merge op's v-first store path (m0a validated only
    state_v_first=False) and for the adapter math. Expected magnitude:
    fp32 GEMM rounding differences between the AscendC chain and torch
    matmul (m0a measured 5.7e-06..9.5e-06 abs at the model shape) -- 1-ulp
    class, orders inside the fla 8e-3 budget. The tree scan on the same
    data is compared too (its NPU-measured re-association bound)."""
    ops = _import_npu_kda_ops()
    if ops is None:
        pytest.skip("fla_npu or xllm KDA CP ops (standard custom_xllm_math vendor) not available")
    _chunk_kda_fwd, conv_op, cp_ops = ops

    device = torch.device(os.environ["XLLM_KDA_TEST_NPU_DEVICE"])
    torch.npu.set_device(device)
    generator = torch.Generator().manual_seed(20261020)
    num_heads, head_dim = 2, 128
    qkv_dim = num_heads * head_dim
    conv_dim = 3 * qkv_dim
    # 4 causal segments of 128 rows: the cp2-equivalent gathered layout.
    cp_size = 2
    seg_lens = [128, 128, 128, 128]
    total = sum(seg_lens)
    seg_cu = [0]
    for length in seg_lens:
        seg_cu.append(seg_cu[-1] + length)

    global_qkv = torch.randn(total, conv_dim, generator=generator).to(device=device, dtype=torch.bfloat16)
    beta_raw_cpu, gate_raw_cpu = _model_isomorphic_gate_inputs(total, num_heads, head_dim, generator)
    global_beta = beta_raw_cpu.to(device=device, dtype=torch.float32)
    global_gate = gate_raw_cpu.to(device=device, dtype=torch.bfloat16)
    conv_state = torch.zeros(len(seg_lens), 3, conv_dim, dtype=torch.bfloat16, device=device)
    conv_out = conv_op(
        global_qkv,
        (torch.randn(4, conv_dim, generator=generator) * 0.5).to(device=device, dtype=torch.bfloat16),
        None,
        conv_state,
        query_start_loc=seg_cu,
        initial_state_mode=[0] * len(seg_lens),
        activation_mode=1,
        run_mode=0,
    )
    hidden_shape = (1, total, num_heads, head_dim)
    q_in = l2_norm(conv_out[:, :qkv_dim].reshape(hidden_shape))
    k_in = l2_norm(conv_out[:, qkv_dim : 2 * qkv_dim].reshape(hidden_shape))
    v_in = conv_out[:, 2 * qkv_dim :].reshape(hidden_shape).contiguous()
    gate_kwargs = dict(
        A_log=torch.zeros(num_heads, dtype=torch.float32, device=device),
        dt_bias=torch.zeros(num_heads * head_dim, dtype=torch.float32, device=device),
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    )
    factors = cp_ops.xllm_chunk_kda_fwd_prepare(
        q_in.contiguous(),
        k_in.contiguous(),
        v_in.contiguous(),
        global_gate.view(hidden_shape).contiguous(),
        global_beta.view(1, total, num_heads).sigmoid().contiguous(),
        head_dim**-0.5,
        chunk_size=KDA_CP_CHUNK_SIZE,
        layout="BSND",
        cu_seqlens=seg_cu,
        backward_mode="none",
        **gate_kwargs,
    )
    hm_causal = cp_ops.xllm_pre_process_fwd_kernel_merged(
        factors[6],
        factors[3],
        factors[4],
        gk=factors[0],
        cu_seqlens=seg_cu,
        chunk_size=KDA_CP_CHUNK_SIZE,
    ).clone()
    torch.npu.synchronize()
    assert tuple(hm_causal.shape) == (len(seg_lens), num_heads, head_dim, 2 * head_dim)

    # Rearrange the causal rows into the engine's gathered layout
    # (row = rank*2 + half; the fold contract wants [S, n_seqs, HV, K, V+K],
    # so insert the single-sequence axis); kda_cp_causal_summary_rows maps
    # causal position -> gathered row, so the inverse scatter reproduces
    # exactly what the all-gather hands the fold.
    hm_gathered_layout = hm_causal.unsqueeze(1)
    causal_rows = torch.tensor(kda_cp_causal_summary_rows(cp_size), dtype=torch.int64, device=device)
    gathered = torch.empty_like(hm_gathered_layout)
    gathered[causal_rows] = hm_gathered_layout

    base = torch.randn(1, num_heads, head_dim, head_dim, generator=generator).to(device=device) * 0.1
    merge_states = kda_cp_merge_fold_initial_states(
        cp_ops.xllm_merge_fwd_bwd_kernel, base, gathered, causal_rows, (1, 2, 3)
    )
    scan_states = kda_cp_tree_scan_fold_initial_states(base, gathered.index_select(0, causal_rows), (1, 2, 3))
    torch.npu.synchronize()

    def _rel_fro(actual: torch.Tensor, expected: torch.Tensor) -> float:
        difference = (actual.float() - expected.float()).norm().item()
        return difference / max(expected.float().norm().item(), 1e-30)

    for index, prior_chunks in enumerate((1, 2, 3)):
        sequential = kda_cp_compose_initial_state(base, gathered, prior_chunks)
        merge_rel = _rel_fro(merge_states[index], sequential)
        scan_rel = _rel_fro(scan_states[index], sequential)
        # 1-ulp-class GEMM rounding / re-association vs the sequential fold.
        assert merge_rel < 1e-4, (prior_chunks, "merge", merge_rel)
        assert scan_rel < 1e-4, (prior_chunks, "scan", scan_rel)

    # Zero prior chunks return the base itself (no merge call issued).
    zero_state = kda_cp_merge_fold_initial_states(cp_ops.xllm_merge_fwd_bwd_kernel, base, gathered, causal_rows, (0,))[
        0
    ]
    assert zero_state is base


@pytest.mark.skipif(not os.getenv("XLLM_KDA_TEST_NPU_DEVICE"), reason="NPU device not configured")
def test_kda_cp_desc_cache_static_buffer_contract_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    """N-1 descriptor/output cache contract (the N-0a PoC, now in the
    wrapper): static outputs are the SAME objects across cache hits and a
    re-run refreshes them in place; mutating input data in place between
    calls refreshes the outputs (never serves stale buffers); and across
    an LRU eviction cycle (5 shapes through a capacity-4 cache) a rebuild
    reproduces the pre-eviction values bit-exactly."""
    ops = _import_npu_kda_ops()
    if ops is None:
        pytest.skip("fla_npu or xllm KDA CP ops (standard custom_xllm_math vendor) not available")
    _chunk_kda_fwd, _conv_op, cp_ops = ops

    device = torch.device(os.environ["XLLM_KDA_TEST_NPU_DEVICE"])
    torch.npu.set_device(device)
    generator = torch.Generator().manual_seed(20261021)
    num_heads, head_dim = 2, 128
    qkv_dim = num_heads * head_dim
    conv_dim = 3 * qkv_dim
    total = 256
    gate_kwargs = dict(
        A_log=torch.zeros(num_heads, dtype=torch.float32, device=device),
        dt_bias=torch.zeros(num_heads * head_dim, dtype=torch.float32, device=device),
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    )

    global_qkv = torch.randn(total, conv_dim, generator=generator).to(device=device, dtype=torch.bfloat16)
    beta_raw_cpu, gate_raw_cpu = _model_isomorphic_gate_inputs(total, num_heads, head_dim, generator)
    global_beta = beta_raw_cpu.to(device=device, dtype=torch.float32)
    global_gate = gate_raw_cpu.to(device=device, dtype=torch.bfloat16)
    hidden_shape = (1, total, num_heads, head_dim)
    q_in = l2_norm(global_qkv[:, :qkv_dim].reshape(hidden_shape).float()).to(torch.bfloat16).contiguous()
    k_in = l2_norm(global_qkv[:, qkv_dim : 2 * qkv_dim].reshape(hidden_shape).float()).to(torch.bfloat16).contiguous()
    v_in = global_qkv[:, 2 * qkv_dim :].reshape(hidden_shape).contiguous()
    g_all = global_gate.view(hidden_shape).contiguous()
    beta_all = global_beta.view(1, total, num_heads).sigmoid().contiguous()

    def _prepare(cu: list[int]) -> tuple[Any, ...]:
        return cp_ops.xllm_chunk_kda_fwd_prepare(
            q_in,
            k_in,
            v_in,
            g_all,
            beta_all,
            head_dim**-0.5,
            chunk_size=KDA_CP_CHUNK_SIZE,
            layout="BSND",
            cu_seqlens=cu,
            backward_mode="none",
            **gate_kwargs,
        )

    def _ppfm(factors: tuple[Any, ...], cu: list[int]) -> torch.Tensor:
        return cp_ops.xllm_pre_process_fwd_kernel_merged(
            factors[6],
            factors[3],
            factors[4],
            gk=factors[0],
            cu_seqlens=cu,
            chunk_size=KDA_CP_CHUNK_SIZE,
        )

    cu_main = [0, 64, 128, 256]

    # ---- reference values (cloned off the static buffers) -------------------
    factors_first = _prepare(cu_main)
    reference_slots = {
        slot: factors_first[slot].clone() for slot in (0, 1, 3, 4, 6, 7) if factors_first[slot] is not None
    }
    hm_first = _ppfm(factors_first, cu_main).clone()
    torch.npu.synchronize()

    # ---- cached re-run: same objects, identical values ----------------------
    factors_cached = _prepare(cu_main)
    for slot, reference_clone in reference_slots.items():
        assert factors_cached[slot] is factors_first[slot], slot
        assert torch.equal(factors_cached[slot], reference_clone), slot
    hm_cached = _ppfm(factors_cached, cu_main)
    assert torch.equal(hm_cached, hm_first)

    # ---- in-place input mutation: outputs refresh (never stale) -------------
    with torch.no_grad():
        q_in.copy_(torch.randn(q_in.shape, generator=generator).to(device=device, dtype=torch.bfloat16))
        g_all.copy_(torch.randn(g_all.shape, generator=generator).to(device=device, dtype=torch.bfloat16))
    factors_mutated = _prepare(cu_main)
    hm_mutated = _ppfm(factors_mutated, cu_main)
    torch.npu.synchronize()
    # The mutated values actually differ from the originals (the outputs
    # track the input buffers, not stale static copies).
    for slot in reference_slots:
        assert not torch.equal(factors_mutated[slot], reference_slots[slot]), ("mutation", slot)
    assert not torch.equal(hm_mutated, hm_first)

    # ---- merge op: static output identity across cache hits -----------------
    ag_hm = hm_mutated.clone()
    assert ag_hm.shape[0] >= 3
    h_cached = cp_ops.xllm_merge_fwd_bwd_kernel(ag_hm, 2, 2, forward=True, state_v_first=True)
    h_again = cp_ops.xllm_merge_fwd_bwd_kernel(ag_hm, 2, 2, forward=True, state_v_first=True)
    torch.npu.synchronize()
    assert h_again is h_cached
    assert torch.equal(h_again, h_cached)

    # ---- LRU eviction cycle: 5 shapes through the capacity-4 cache ----------
    for start in range(5):
        _prepare([0, 32 * (start + 1), 256])
    prepare_cache = cp_ops._OP_CACHES[cp_ops._PREPARE_SYMBOL]
    assert len(prepare_cache) <= cp_ops._DESC_CACHE_CAPACITY
    # The main shape was evicted; a rebuild reproduces the ORIGINAL
    # (pre-mutation) reference values bit-exactly -- restore the inputs
    # first, then re-run and compare against the reference clones.
    with torch.no_grad():
        q_in.copy_(l2_norm(global_qkv[:, :qkv_dim].reshape(hidden_shape).float()).to(torch.bfloat16))
        g_all.copy_(global_gate.view(hidden_shape))
    factors_rebuilt = _prepare(cu_main)
    for slot in reference_slots:
        assert torch.equal(factors_rebuilt[slot], reference_slots[slot]), ("eviction", slot)
    torch.npu.synchronize()
