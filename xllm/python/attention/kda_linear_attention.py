# Copyright 2025-2026 The xLLM Authors.
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
"""KDA (Kernelized Delta Attention) linear-attention forward + state I/O."""

from __future__ import annotations

import functools
from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import torch

from scripts.logger import logger
from xllm.python.attention.backend import (
    linear_state_checkpoint_stride,
    resolve_linear_state_io_indices,
)
from xllm.python.model_executor.forward_context import (
    get_execution_buffer,
    in_acl_graph,
)

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention
    from xllm.python.model_executor.cp_utils import CpContext


def _get_host_q_cu_seq_lens(metadata, expected_num_seqs: int) -> list[int]:
    """Return cumulative query lengths without synchronizing a device tensor."""
    host_values = getattr(metadata, "q_cu_seq_lens_host_values", None)
    if host_values is not None and len(host_values) == expected_num_seqs + 1:
        return [int(value) for value in host_values]
    if host_values is not None and len(host_values) == expected_num_seqs:
        return [0] + [int(value) for value in host_values]

    q_seq_lens_host = getattr(metadata, "q_seq_lens_host", None)
    if (
        q_seq_lens_host is not None
        and q_seq_lens_host.device.type == "cpu"
        and q_seq_lens_host.numel() == expected_num_seqs
    ):
        q_cu_list = [0]
        cumulative = 0
        for length in q_seq_lens_host.tolist():
            cumulative += int(length)
            q_cu_list.append(cumulative)
        return q_cu_list

    q_cu_seq_lens = metadata.q_cu_seq_lens
    assert q_cu_seq_lens is not None, "multi-sequence linear attention needs q_cu_seq_lens"
    if q_cu_seq_lens.numel() != expected_num_seqs + 1:
        raise RuntimeError(
            f"KDA query/state metadata length mismatch: q_cu={q_cu_seq_lens.numel()}, state={expected_num_seqs}"
        )
    if q_cu_seq_lens.device.type != "cpu":
        raise RuntimeError("NPU KDA multi-sequence prefill requires host q_cu_seq_lens metadata")
    return [int(value) for value in q_cu_seq_lens.to(torch.int64).tolist()]


def _take_leading_token_rows(tensor: torch.Tensor | None, num_rows: int, keep_rows: int) -> torch.Tensor | None:
    """Keep the verify rows and drop a bucket tail that is shorter than one group.

    Callers pass gate ``[1, T, heads, dim]`` and beta ``[1, T, heads]``. The
    token axis is 1 (or 0 for a flat ``[T, ...]`` tensor). Head count can equal
    the bucket, so a later axis must not be treated as the token axis.
    """
    if tensor is None or keep_rows >= num_rows:
        return tensor
    if tensor.dim() >= 2 and tensor.shape[1] == num_rows:
        return tensor[:, :keep_rows]
    if tensor.shape[0] == num_rows:
        return tensor[:keep_rows]
    return tensor


# ---------------------------------------------------------------------------
# KDA context-parallel prefill (two-pass scheme, --enable_kda_cp gflag).
#
# Under prefill CP the legacy path all-gathers mixed_qkv/g_raw/beta_raw to
# global rows (cp_merge_rows), runs the conv + delta-rule replicated on the
# full stream, and re-shards the output (cp_shard_rows). That merge exists
# because it makes the per-sequence conv/ssm state bit-identical on every CP
# rank (BlockType::LINEAR is not a kv-split block type), which PD transfer,
# MTP checkpoints and the linear-state prefix cache all rely on.
#
# The CP path here replaces the replicated compute with the M1-validated
# two-pass scheme per zigzag segment: conv halo exchange -> per-segment WY
# summary -> ONE summary all-gather -> causal-order fp32 merge -> local
# replay with the merged initial state -> final-state broadcast from the
# causally-last segment owner. The broadcast restores the bit-identical
# replica invariant the legacy merge provided.
#
# NUMERICS: the two-pass compose is NOT bit-exact vs the merged path for
# segments with >= 2 prior segments (measured ~6.6e-4 rel_fro at toy shapes,
# task research m1 G2 -- fp32 re-association of the chunk chain plus bf16
# chunking at span boundaries; a single prior segment composes exactly).
# Since ma2-2 the summary factors come from OUR standalone prepare op
# (xllm_chunk_kda_fwd_prepare) instead of the deployed fused emission, and
# those differ from the fused factors at LAST-BIT rounding scale only
# (ma1: gk 1.79e-07 abs ~= 1 fp32 ulp, bf16 slots <= 9.8e-04 ~= 1 bf16
# ulp), perturbing each hm by ~1.7e-4 rel_fro -- 4x inside the accepted
# cp4 compose divergence above, so single-prior segments are no longer
# bit-exact either. Acceptance budget: the fla KDA recipe validates at 8e-3
# (bf16) vs the naive recurrence, and every measured divergence above sits
# >= 5x inside it; the flag-on engine oracle is greedy-text-equality + a
# measured logprobs bound against that budget, NOT a bit-exact gate. The
# --enable_kda_cp flag still defaults true: the two-pass scheme is the only
# shipped KDA CP implementation, so that bounded divergence IS the shipped
# behavior (the flag exists to restore the byte-identical legacy path).
#
# FOLD RE-ASSOCIATION (n1): the causal merge composes the SAME causal
# chain with TWO implementations. "merge" (the default) composes it inside
# ONE aclnnMergeFwdBwdKernel launch per prefix state; on this stack it
# measures BIT-EXACT vs the sequential torch fold at the engine shape
# (n1: 0.0 abs/rel at HV=32, K=V=128, prior depths 3 and 7, both
# model-isomorphic and dense random affines; m0a's 5.7e-06..9.5e-06 was
# vs a CPU-side fp32 compose -- 1-ulp-class remains the conservative
# contract bound). The "scan" fallback is a Hillis-Steele parallel
# prefix: it RE-ASSOCIATES the strictly sequential fold into log2(2P+1)
# balanced levels -- measured 5.4e-08..3.2e-07 rel_fro vs the sequential
# fold (1-ulp-class; the CPU unit test pins the bound). It runs only when
# the merge op's hardcoded K=V=128 / head budget does not fit the shape
# (selection from observable shapes before execution, not a masked
# failure). Both compose identical causal chains; only the fp32 summation
# ORDER differs.
#
# TRANSPORT-ONLY REDUCTIONS (M-B): the collective schedule ships three
# numerics-IDENTICAL reductions -- (1) the final conv + ssm state
# broadcasts merge into ONE broadcast (fp32 promotion of the bf16 conv
# payload round-trips exactly); (2) the conv halo all-gather is issued
# ASYNC before the prefix-state reads; (3) the summary all-gather is
# issued ASYNC right after the summary fold and the zero-prior segments
# replay while it is in flight.
# The exchanged VALUES and every kernel input are unchanged; only WHEN the
# exchanges run moves. HCCL semantics validated on the real stack: task
# research mb (mb-probe-async.py).
#
# BATCHED SEGMENT CALLS (M-B2): the summary pass now issues ONE prepare
# and ONE ppfm call over the rank's whole local stream instead of one pair
# per zigzag segment (the per-segment loop paid the ~1 ms fixed kernel
# launch cost twice at the engine's 256-row spans). This is a
# call-STRUCTURE change only: both ops are varlen-native, cu_seqlens
# restarts the 64-row chunk grid at every segment boundary in both forms,
# and the gate pins every segment length to a multiple of 64, so the
# batched factors and hm rows are IDENTICAL to the per-segment calls
# (pinned on the real kernels by the NPU batched-vs-per-segment identity
# test, task research mb2).
# ---------------------------------------------------------------------------

# chunk_kda_fwd's chunk size; also the exactness boundary for span splits
# (64-aligned chaining is bit-exact, mid-cut is not -- task research m0b B1).
KDA_CP_CHUNK_SIZE = 64

# One-time probe print state (alpha-probe pattern: a single line per
# process recording that the CP path actually engaged).
_KDA_CP_PROBE_EMITTED = False

# One-shot guards for the diagnostics that would otherwise fire on every
# prefill forward (the warm-twin D2H fallback and the row-staging fallback).
_KDA_CP_WARM_TWIN_WARNED = False
_KDA_CP_ROWS_FALLBACK_LOGGED = False


# Pinned once per process ON PURPOSE: availability must not flip between ranks or
# between the gate and the execute step (a mid-step flip would let one rank enter
# a collective the others skip). functools.cache still exposes the standard
# escape hatch -- ``_kda_cp_ops.cache_clear()`` -- for tests and for an operator
# who fixed the environment in-process and wants a re-probe.
@functools.cache
def _kda_cp_ops() -> tuple[object, object, object] | None:
    """Resolve OUR KDA CP op set once, at first gate evaluation.

    The CP path consumes three ops from the standard ``custom_xllm_math``
    opp vendor (task research ma2-1/ma2-2, n1, n3-5):
    ``xllm_chunk_kda_fwd_prepare`` (the prepare-only WY factor pass,
    self-built under the renamed op ``XllmChunkKdaFwdPrepare``),
    ``xllm_pre_process_fwd_kernel_merged`` (the hm=[E|M] summary fold)
    and ``xllm_merge_fwd_bwd_kernel`` (the one-launch CP merge fold,
    wired in n1). The deployed fla_npu wheel has none of them. Before
    ma2-2 the factors came from the deployed fused
    ``chunk_kda_fwd(disable_recompute=True)`` and the fold from a
    source-tree overlay import; both are replaced by this self-contained
    module, so the CP path no longer depends on any source-tree fla-nu
    package or PYTHONPATH overlay.

    The probe stays lazy (kda_linear_attention is imported in CPU-only
    builds too, where the vendor is absent and ``available()`` is a plain
    False) and functools.cache pins the answer for the process --
    availability can never flip mid-step, which a lazy per-call fallback
    would allow. The vendor is discovered from the standard opp install
    (see kda_cp_ops: ``${ASCEND_HOME_PATH}/opp/vendors/custom_xllm_math``);
    the CANN runtime resolves the kernel bins from the same install via
    the vendors config.ini, so no path environment is needed at all.
    Tests monkeypatch this function directly (the cache would otherwise
    pin a stale probe across sys.modules patches).
    """
    try:
        from xllm.python.kernels_npu import kda_cp_ops

        if not kda_cp_ops.available():
            return None
    except Exception as exc:  # broken module state must fail the gate, not the step
        logger.warning("KDA CP op probe failed; falling back to the legacy merged path: %s", exc)
        return None
    return (
        kda_cp_ops.xllm_chunk_kda_fwd_prepare,
        kda_cp_ops.xllm_pre_process_fwd_kernel_merged,
        kda_cp_ops.xllm_merge_fwd_bwd_kernel,
    )


def _kda_cp_batch_q_seq_lens(metadata: object) -> list[int] | None:
    """Per-sequence query lengths of this forward, from host metadata only.

    Uses the same host forms `_get_host_q_cu_seq_lens` accepts, but without
    needing the linear-state slot count: the gate runs in the model layer
    before any slot view exists. Returns None when no host form is available
    (the gate then falls back to the legacy path rather than syncing a
    device tensor on the prefill critical path).
    """
    q_seq_lens_host = getattr(metadata, "q_seq_lens_host", None)
    if q_seq_lens_host is not None and q_seq_lens_host.device.type == "cpu" and q_seq_lens_host.numel() > 0:
        return [int(length) for length in q_seq_lens_host.tolist()]

    host_values = getattr(metadata, "q_cu_seq_lens_host_values", None)
    if host_values:
        values = [int(value) for value in host_values]
        if values[0] != 0:
            values = [0, *values]
        return [end - start for start, end in zip(values[:-1], values[1:])]

    q_cu_seq_lens = getattr(metadata, "q_cu_seq_lens", None)
    if q_cu_seq_lens is not None and q_cu_seq_lens.device.type == "cpu" and q_cu_seq_lens.numel() > 1:
        values = [int(value) for value in q_cu_seq_lens.tolist()]
        if values[0] != 0:
            values = [0, *values]
        return [end - start for start, end in zip(values[:-1], values[1:])]
    return None


def kda_cp_segments_chunk_aligned(q_seq_lens: Sequence[int], cp_size: int) -> bool:
    """Whether every zigzag chunk of every sequence owns whole KDA chunks.

    Zigzag geometry (cp_utils): a sequence of L query rows is padded to a
    multiple of 2*cp_size and cut into 2*cp_size equal chunks of
    ceil(L / (2*cp_size)) rows; chunk j holds clamp(L - j*chunk_len, 0,
    chunk_len) real rows. The kernel restarts its 64-token chunk grid at
    every cu_seqlens boundary, so a CP segment reproduces the merged
    stream's grid exactly only when its real length is a multiple of 64
    (m0b B1: 64-aligned chaining is bit-exact, mid-cut pays ~6e-3 rel). A
    short (< 64) or empty segment also fails: an empty segment owns no
    whole chunk and would make the per-rank summary count (and thus the
    all-gather shape) rank-dependent.
    """
    if cp_size <= 1:
        return False
    chunk_count = 2 * cp_size
    for length in q_seq_lens:
        if length <= 0:
            return False
        chunk_len = -(-length // chunk_count)
        for chunk in range(chunk_count):
            real = min(chunk_len, max(0, length - chunk * chunk_len))
            if real == 0 or real % KDA_CP_CHUNK_SIZE != 0:
                return False
    return True


def kda_cp_chunk_owner(chunk_id: int, cp_size: int) -> tuple[int, int]:
    """(rank, half) of the zigzag chunk ``chunk_id`` in [0, 2*cp_size).

    Rank r owns chunk r (its first-half segment) paired with chunk
    2*cp_size-1-r (its second-half segment). The CAUSALLY-LAST chunk
    2*cp_size-1 therefore belongs to rank 0's second segment -- the
    final-state broadcast root must be computed from this mapping, never
    from a rank id (rank cp-1 is NOT the last owner under zigzag).
    """
    if chunk_id < 0 or chunk_id >= 2 * cp_size:
        raise ValueError(f"zigzag chunk {chunk_id} outside [0, {2 * cp_size})")
    if chunk_id < cp_size:
        return chunk_id, 0
    return 2 * cp_size - 1 - chunk_id, 1


def kda_cp_prefill_eligible(cp_context: object, metadata: object, layer: object) -> bool:
    """The ONE KDA-CP gate, shared by the model layer and the backend.

    Every input is rank-uniform: the batch metadata flags and host sequence
    lengths are identical on every CP rank, the zigzag chunk arithmetic is a
    pure function of those lengths, the layer config is replicated, and the
    CP-ops probe is process-global. A gate reading any rank-local state
    (e.g. the rank-local q_cu_seqlens segment view) could desynchronize the
    group -- one rank would enter a collective the others never join.
    """
    if cp_context is None or getattr(cp_context, "cp_size", 1) <= 1:
        return False
    if not (getattr(metadata, "is_prefill", False) or getattr(metadata, "is_chunked_prefill", False)):
        return False
    # Spec-verify chunked prefill relies on merged global-row inputs
    # (_spec_verify_v3's B6 CP-safety proof); mixed batches are refused at
    # admission but stay excluded here defensively.
    if getattr(metadata, "is_spec_verify", False) or getattr(metadata, "is_mixed", False):
        return False
    if in_acl_graph():
        return False
    # The CP conv entry (causal_conv1d_prefill) hardcodes SiLU, and the
    # summary/replay calls assume the fused safe-gate kwargs.
    if getattr(layer, "activation", "silu") != "silu":
        return False
    forget_gate = getattr(layer, "forget_gate", None)
    gate_lb = getattr(forget_gate, "safe_gate_lower_bound", None) if forget_gate is not None else None
    if gate_lb is None or not -5.0 <= gate_lb < 0.0:
        return False
    if _kda_cp_ops() is None:
        return False
    q_seq_lens = _kda_cp_batch_q_seq_lens(metadata)
    if not q_seq_lens:
        return False
    return kda_cp_segments_chunk_aligned(q_seq_lens, int(cp_context.cp_size))


def kda_cp_compose_initial_state(
    base_state: torch.Tensor,
    gathered_summaries: torch.Tensor,
    prior_chunks: int,
) -> torch.Tensor:
    """Causal-order fp32 fold of prior-segment summaries onto a state.

    The strictly sequential left-to-right composition -- the REFERENCE
    fold the merge/scan identity tests pin against (not used by the CP
    execution path, which always runs the merge op or the tree scan).
    ``base_state`` is ``[n_seqs, HV, V, K]`` fp32 in the kernel's STORED
    (v-first) orientation -- the prefix state read from the SSM read slot,
    or zeros for cold sequences. ``gathered_summaries`` is the CP-group
    all-gather result ``[2*cp, n_seqs, HV, K, V+K]`` (row ``rank*2+half``,
    see kda_cp_execute_linear). Composes causal chunks ``[0,
    prior_chunks)`` oldest -> newest with the m0b-B2-pinned stored-space
    recurrence ``X <- X @ M_stored + E_stored`` where ``E_stored`` /
    ``M_stored`` are the transposed ``[..., :V]`` / ``[..., V:]`` blocks of
    each summary. Out-of-place: ``base_state`` is never mutated (both local
    halves fold from the same base).
    """
    cp_size = gathered_summaries.shape[0] // 2
    v_dim = base_state.shape[-2]
    state = base_state
    for chunk in range(prior_chunks):
        owner_rank, owner_half = kda_cp_chunk_owner(chunk, cp_size)
        summary = gathered_summaries[owner_rank * 2 + owner_half]
        emission = summary[..., :v_dim].transpose(-1, -2).float()
        transition = summary[..., v_dim:].transpose(-1, -2).float()
        state = state @ transition + emission
    return state


def kda_cp_causal_summary_rows(cp_size: int) -> tuple[int, ...]:
    """Gathered-summary row index of each causal chunk, in causal order.

    Row ``rank*2 + half`` of the all-gathered summaries holds rank r's
    half-``half`` segment; causal chunk j < cp lives at row ``2j`` and
    causal chunk j >= cp at row ``2(2cp-1-j)+1`` (zigzag). The result is
    the permutation the merge/scan folds use to read the summaries in
    causal order (a collision would silently compose the wrong segment;
    pinned by test_kda_cp_causal_summary_rows_form_a_permutation).
    """
    rows = []
    for chunk in range(2 * cp_size):
        owner_rank, owner_half = kda_cp_chunk_owner(chunk, cp_size)
        rows.append(owner_rank * 2 + owner_half)
    return tuple(rows)


def kda_cp_tree_scan_fold_initial_states(
    base_state: torch.Tensor,
    causal_summaries: torch.Tensor,
    prior_counts: Sequence[int],
) -> list[torch.Tensor]:
    """Hillis-Steele parallel-prefix fold of the causal summaries.

    Pure-torch fallback fold (the automatic choice when the merge op's
    hardcoded K=V=128 / head budget does not fit). ``causal_summaries`` is
    ``[S, n_seqs, HV, K, V+K]``
    in CAUSAL order (see :func:`kda_cp_causal_summary_rows`); the stored
    recurrence is ``X <- X @ M_stored + E_stored`` per segment. The scan
    seeds entry 0 with the affine ``(M=I, E=base)`` and entry i+1 with
    summary i; level ``step`` composes ``F[i] <- F[i-step] o F[i]`` (=
    ``(M_a M_b, E_a M_b + E_b)``) via two batched matmuls over the valid
    range, so after ceil(log2(S+1)) levels entry j's E block IS the
    initial state of causal segment j. Returns one state per requested
    causal position (0 = the base itself).

    NUMERICS: re-associates the fold (balanced tree instead of strict
    left-to-right) -- 1-ulp-class vs the sequential fold; the CPU unit
    test measures the bound.
    """
    seq_count, head_value, v_dim, k_dim = base_state.shape
    total = int(causal_summaries.shape[0])
    scan_transition = torch.empty(
        total + 1, seq_count, head_value, k_dim, k_dim, dtype=torch.float32, device=base_state.device
    )
    scan_emission = torch.empty(
        total + 1, seq_count, head_value, v_dim, k_dim, dtype=torch.float32, device=base_state.device
    )
    scan_transition[0].copy_(torch.eye(k_dim, dtype=torch.float32, device=base_state.device))
    scan_emission[0].copy_(base_state)
    scan_transition[1:].copy_(causal_summaries[..., v_dim:].transpose(-1, -2))
    scan_emission[1:].copy_(causal_summaries[..., :v_dim].transpose(-1, -2))
    step = 1
    while step <= total:
        # Compose the older affine F[:-step] into F[step:]. Both right-hand
        # sides must be fully evaluated BEFORE either slice assignment:
        # the transition slice is a live view, so assigning it first would
        # feed the already-updated transition into the emission compose.
        new_transition = scan_transition[:-step] @ scan_transition[step:]
        new_emission = scan_emission[:-step] @ scan_transition[step:] + scan_emission[step:]
        scan_transition[step:] = new_transition
        scan_emission[step:] = new_emission
        step *= 2
    return [scan_emission[int(count)].clone() for count in prior_counts]


# Bounded reuse of the merge fold's ag_hm staging buffer (one entry per
# (rows, heads, dims) signature; the engine's chunked prefill is uniform).
# Single-threaded per process by the same assumption as the host plan and
# the op descriptor cache (no locking; the engine runs one forward at a
# time per rank).
_KDA_CP_FOLD_BUFFERS: OrderedDict[tuple, torch.Tensor] = OrderedDict()
_KDA_CP_FOLD_BUFFER_CAPACITY = 8


def kda_cp_merge_fold_initial_states(
    merge_op: object,
    base_state: torch.Tensor,
    gathered_summaries: torch.Tensor,
    causal_rows: torch.Tensor,
    prior_counts: Sequence[int],
) -> list[torch.Tensor]:
    """Fold prior summaries into prefix states via ONE merge-op launch each.

    Default fold. Builds the op's
    ``[1+S, n_seqs*HV, K, V+K]`` argument: row 0 is a VIRTUAL summary whose
    E block is the base state in natural orientation (the op ignores its
    input h and always seeds the chain from the FIRST prior row's E --
    source-pinned, task research n1 section 1; the virtual row turns that
    into our affine-in-the-base semantics, and its M block is never read),
    rows 1..S are the gathered summaries reordered into causal order by
    ``causal_rows``. Then causal position p (p >= 1) composes rows
    ``[0, p]`` with the single call ``merge_op(ag_hm, p+1, p+1)`` -- rank
    and N shift by one to account for the virtual row. ``state_v_first``
    output orientation matches the kernel state cache ([V, K] stored).

    NUMERICS: measured BIT-EXACT vs the sequential torch fold on this
    stack (n1: 0.0 abs/rel at the engine shape, model-isomorphic and
    dense affines); m0a's 1-ulp-class bound (vs a CPU-side compose)
    remains the conservative contract. See the module NUMERICS header.
    """
    seq_count, head_value, v_dim, k_dim = base_state.shape
    total = int(gathered_summaries.shape[0])
    if k_dim != 128 or v_dim != 128:
        raise RuntimeError("kda_cp_merge_fold_initial_states requires K = V = 128 (merge op contract).")
    total_heads = seq_count * head_value
    flat_summaries = gathered_summaries.view(total, total_heads, k_dim, v_dim + k_dim)

    buffer_key = (total + 1, total_heads, k_dim, v_dim + k_dim, base_state.device, torch.float32)
    ag_hm = _KDA_CP_FOLD_BUFFERS.get(buffer_key)
    if ag_hm is None:
        ag_hm = torch.empty(buffer_key[:4], device=base_state.device, dtype=torch.float32)
        _KDA_CP_FOLD_BUFFERS[buffer_key] = ag_hm
        while len(_KDA_CP_FOLD_BUFFERS) > _KDA_CP_FOLD_BUFFER_CAPACITY:
            _KDA_CP_FOLD_BUFFERS.popitem(last=False)
    else:
        _KDA_CP_FOLD_BUFFERS.move_to_end(buffer_key)
    # Register the staging buffer so the merge wrapper's descriptor cache
    # ADOPTS it (descriptors bake its address; the rewrite below refreshes
    # the values every call, so adoption never serves stale data).
    from xllm.python.kernels_npu import kda_cp_ops

    kda_cp_ops.register_reusable_input(ag_hm)

    # Row 0: E block = base in natural [K, V] orientation; M block unused.
    ag_hm[0, ..., :v_dim].copy_(base_state.transpose(-1, -2).reshape(total_heads, k_dim, v_dim))
    ag_hm[0, ..., v_dim:].zero_()
    torch.index_select(flat_summaries, 0, causal_rows, out=ag_hm[1:])

    states = []
    for count in prior_counts:
        if count < 1:
            # Zero prior chunks: the initial state IS the base (the fold
            # with an empty prefix). The op could express it as (rank=1,
            # N=1), but the caller already holds the base tensor.
            states.append(base_state)
            continue
        h = merge_op(ag_hm, count + 1, count + 1, forward=True, state_v_first=True)
        # The merge wrapper returns the DESCRIPTOR CACHE's static output buffer,
        # and that buffer is reused by the next cache-hit call with the same key
        # (the key is (ag_hm signature, n_ranks, rank, forward, state_v_first),
        # so a later layer's fold with the same prior count hits the same entry).
        # Today every consumer has finished before that happens, but aliasing a
        # cached buffer across calls is exactly the kind of implicit dependency
        # that rots -- clone so the returned state owns its storage.
        states.append(h.view(seq_count, head_value, v_dim, k_dim).clone())
    return states


def _kda_cp_fold_mode(k_dim: int, v_dim: int, total_heads: int) -> str:
    """Select the causal-merge fold implementation for this forward.

    ``merge`` (the default) composes the causal chain inside ONE
    aclnnMergeFwdBwdKernel launch per prefix state; it requires the op's
    hardcoded K=V=128 and ``n_seqs*HV <= 256`` head budget. Outside those
    shapes the pure-torch tree scan runs instead (an explicitly supported
    alternate path selected from observable shapes before execution, not a
    masked failure). Rank-uniform by construction: the selection is a pure
    function of the shape tuple, which the eligibility gate already pins
    identical on every CP rank.
    """
    if k_dim == 128 and v_dim == 128 and total_heads <= 256:
        return "merge"
    return "scan"


def _kda_cp_async_all_gather(
    group: object,
    local: torch.Tensor,
    world_size: int,
) -> tuple[torch.Tensor, object]:
    """Async dim-0 all-gather over ``group``; returns (output, work).

    Mirrors ``distributed.all_gather``'s dim-0 form (one flat output;
    HCCL needs a contiguous contribution) but with ``async_op=True`` so
    the caller keeps enqueueing independent work while the exchange is in
    flight; ``work.wait()`` completes it. torch_npu 2.9 HCCL validated on
    the real stack: handle + wait + data OK, partial overlap with compute
    (task research mb, probe mode ``gather``). The CP path is eager-only
    (the gate excludes acl graphs), so bypassing the graph-operator
    wrapper is safe here.
    """
    import torch.distributed as dist

    output = local.new_empty((local.shape[0] * world_size, *local.shape[1:]))
    work = dist.all_gather_into_tensor(output, local.contiguous(), group=group, async_op=True)
    return output, work


# conv payload dtypes whose values round-trip fp32 promotion exactly; any
# other dtype falls back to the two-broadcast form.
_KDA_CP_BROADCASTABLE = (torch.float32, torch.bfloat16, torch.float16)


def _kda_cp_pack_state_broadcast(
    payload_conv: torch.Tensor,
    payload_ssm: torch.Tensor,
) -> tuple[torch.Tensor, int] | None:
    """Flat fp32 [conv | ssm] payload for ONE merged broadcast.

    The final conv and ssm states share one root (the causally-last
    owner) and one group, so one broadcast can carry both. Conv payloads
    are bf16 or fp32 in every engine configuration, and bf16/fp16 values
    are exactly representable in fp32, so the promote/demote round trip
    is bit-exact -- the merged transport is numerics-neutral. The conv
    bytes at most double (a ~74 KB bf16 payload next to a ~2.1 MB fp32
    ssm payload at the 32k engine shape), far cheaper than a second
    collective's fixed cost. Returns None when the conv dtype cannot
    round-trip fp32 (the caller falls back to two broadcasts).
    """
    if payload_conv.dtype not in _KDA_CP_BROADCASTABLE:
        return None
    flat = torch.cat((payload_conv.reshape(-1).float(), payload_ssm.reshape(-1).float()))
    return flat, payload_conv.numel()


def _kda_cp_unpack_state_broadcast(
    flat: torch.Tensor,
    conv_elements: int,
    conv_dtype: torch.dtype,
    conv_shape: torch.Size,
    ssm_shape: torch.Size,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a merged broadcast payload back into (conv, ssm) states."""
    conv = flat[:conv_elements].to(conv_dtype).view(conv_shape)
    ssm = flat[conv_elements:].view(ssm_shape)
    return conv, ssm


def _kda_cp_zero_prior_segments(cp_rank: int, cp_size: int, num_seqs: int) -> list[int]:
    """Local (sequence, half) segments whose causal merge consumes no
    prior summaries -- their initial state IS the prefix/zero base, so
    they can replay while the summary exchange is still in flight.

    General rule: the first half of a sequence sits at causal chunk
    cp_rank (cp_rank priors), the second half at chunk 2*cp-1-cp_rank
    (2*cp-1-cp_rank priors, always >= 1). Only rank 0's first halves
    reach zero priors -- a warm prefix does not change this, since the
    base state (prefix or zeros) is local either way and never waits on
    the gather.
    """
    first_chunk = cp_rank
    second_chunk = 2 * cp_size - 1 - cp_rank
    segments: list[int] = []
    for seq in range(num_seqs):
        if first_chunk == 0:
            segments.append(2 * seq)
        if second_chunk == 0:
            segments.append(2 * seq + 1)
    return segments


class _KdaCpHostPlan(NamedTuple):
    """Per-call constant tensors of one CP forward shape.

    Everything here depends ONLY on (seg_cu, cp_rank, cp_size,
    conv_state_len, device) -- the segment plan is identical across every
    chunk of a request under chunked prefill, so the tensors are built once
    and reused (n1 host prep slimming; bounded LRU below). Data-dependent
    tensors (row selections, halo/conv buffers, prefix states, the warm
    mask) are deliberately NOT here: their values change per call.

    The index tensors (replay/early/late rows and cu, causal_rows,
    halo_pos) live on ``device`` for the gather/select calls that consume
    them. The conv-call metadata pair (conv_cu, empty_i64) is deliberately
    CPU int64: the conv op marshals its qsl/si/ism arguments to host
    IntArrayRefs inside the C++ wrapper (``.to(torch::kCPU,
    torch::kInt64)``), which is a NO-OP on a tensor that is already
    cpu+int64+contiguous -- passing device tensors there made the wrapper
    block the host on the compute stream three times per call (the G1
    conserved-drain finding: the conv stamp carried the whole upstream
    device backlog, 11.9 ms mean at c8192).
    """

    # int64 CPU seg_cu (the conv call's host query_start_loc).
    conv_cu: torch.Tensor
    replay_cu: torch.Tensor  # int32 seg_cu (the full-batch replay call)
    # int64 CPU empty (the conv call's host state_indices).
    empty_i64: torch.Tensor
    early_cu: torch.Tensor | None  # int32 early replay cu_seqlens
    early_rows: torch.Tensor | None  # int64 packed early row indices
    early_idx: torch.Tensor | None  # int64 early segment ids (index_copy)
    late_cu: torch.Tensor | None  # int32 late replay cu_seqlens
    late_rows: torch.Tensor | None  # int64 packed late row indices
    late_idx: torch.Tensor | None  # int64 late segment ids (index_copy)
    causal_rows: torch.Tensor  # int64 gathered-row of each causal chunk
    # int64 packed-stream position of each halo row, in the payload's flat
    # (half, sequence, window) order: seg_cu[2*seq + half + 1] -
    # conv_state_len + w. Indexing query_index with it yields the absolute
    # local-row ids of exactly the tail rows the halo payload packs (g1:
    # the pack reads the RAW stream, before any row staging).
    halo_pos: torch.Tensor


_KDA_CP_HOST_PLANS: OrderedDict[tuple, _KdaCpHostPlan] = OrderedDict()
_KDA_CP_HOST_PLAN_CAPACITY = 8


def _kda_cp_host_plan(
    seg_cu: Sequence[int],
    cp_rank: int,
    cp_size: int,
    conv_state_len: int,
    device: torch.device,
) -> _KdaCpHostPlan:
    """The (bounded, LRU) host plan for one (segment plan, rank, device).

    Not thread-safe (no locking): the KDA CP path is single-threaded per
    process, the same assumption the op descriptor cache documents
    (``kda_cp_ops._launch_op``).
    """
    key = (tuple(seg_cu), cp_rank, cp_size, conv_state_len, device)
    plan = _KDA_CP_HOST_PLANS.get(key)
    if plan is not None:
        _KDA_CP_HOST_PLANS.move_to_end(key)
        return plan

    num_seqs = len(seg_cu) // 2
    seg_cu_list = [int(value) for value in seg_cu]
    early_segments = _kda_cp_zero_prior_segments(cp_rank, cp_size, num_seqs)
    early_set = set(early_segments)
    late_segments = [segment for segment in range(2 * num_seqs) if segment not in early_set]

    def packed_rows(segments: Sequence[int]) -> torch.Tensor:
        return torch.cat([torch.arange(seg_cu[s], seg_cu[s + 1], dtype=torch.int64, device=device) for s in segments])

    def packed_cu(segments: Sequence[int]) -> torch.Tensor:
        cu = [0]
        for segment in segments:
            cu.append(cu[-1] + (seg_cu_list[segment + 1] - seg_cu_list[segment]))
        return torch.tensor(cu, dtype=torch.int32, device=device)

    halo_pos_list: list[int] = []
    for half in (0, 1):
        for seq in range(num_seqs):
            end = seg_cu_list[2 * seq + half + 1]
            halo_pos_list.extend(range(end - conv_state_len, end))

    plan = _KdaCpHostPlan(
        conv_cu=torch.tensor(seg_cu_list, dtype=torch.int64),
        replay_cu=torch.tensor(seg_cu_list, dtype=torch.int32, device=device),
        empty_i64=torch.empty(0, dtype=torch.int64),
        early_cu=packed_cu(early_segments) if early_segments else None,
        early_rows=packed_rows(early_segments) if early_segments else None,
        early_idx=torch.tensor(early_segments, dtype=torch.int64, device=device) if early_segments else None,
        late_cu=packed_cu(late_segments) if early_segments else None,
        late_rows=packed_rows(late_segments) if early_segments else None,
        late_idx=torch.tensor(late_segments, dtype=torch.int64, device=device) if early_segments else None,
        causal_rows=torch.tensor(kda_cp_causal_summary_rows(cp_size), dtype=torch.int64, device=device),
        halo_pos=torch.tensor(halo_pos_list, dtype=torch.int64, device=device),
    )
    _KDA_CP_HOST_PLANS[key] = plan
    while len(_KDA_CP_HOST_PLANS) > _KDA_CP_HOST_PLAN_CAPACITY:
        _KDA_CP_HOST_PLANS.popitem(last=False)
    return plan


def _kda_cp_rows_are_stream_prefix(seg_cu: Sequence[int], mixed_qkv: torch.Tensor) -> bool:
    """Whether the packed real rows are a contiguous prefix of the local stream.

    The zigzag layout places each sequence's segments at [base, base +
    chunk_len) and [base + chunk_len, base + 2*chunk_len) with real rows
    front-packed per segment, so the packed real stream is ONE contiguous
    prefix exactly when every first half is a full chunk (its real count
    then equals chunk_len, so the second half starts where the first
    ended) and every non-last second half is full as well. The
    causally-last chunk is the zigzag's only partial candidate, and the
    64-row eligibility gate forces even it to be full (a ragged tail fails
    the gate and never reaches the CP path), so this holds for every
    forward the gate admits; the guards keep the fast staging path honest
    if that invariant ever changes. Batch > 1 never takes the fast path
    (the flattened stream is then not the rank-local layout).
    """
    if mixed_qkv.dim() != 3 or mixed_qkv.shape[0] != 1:
        return False
    num_seqs = len(seg_cu) // 2
    if num_seqs <= 0:
        return False
    first_half = [int(seg_cu[2 * seq + 1]) - int(seg_cu[2 * seq]) for seq in range(num_seqs)]
    if 2 * sum(first_half) != int(mixed_qkv.shape[2]):
        return False
    return all(int(seg_cu[2 * seq + 2]) - int(seg_cu[2 * seq + 1]) == first_half[seq] for seq in range(num_seqs - 1))


def _kda_cp_warm_host_values(metadata: object, warm: object, num_seqs: int) -> list[int]:
    """Host-side 0/1 warm values for the conv call's initial-state mode.

    The conv op consumes ism as a host IntArrayRef, so the values must be
    host-resident before the call; reading the device mask would block the
    host on the compute stream (the very sync this helper exists to avoid).
    Resolution order (first hit wins):

    1. ``linear_state_validity_host_values`` -- the metadata view's host
       twin, the same host validity vector the NPU builder materialized
       the ``has_initial_state`` device tensor from, so the values are
       identical by construction and reading them never syncs.
    2. A warm mask that is not a tensor (a host list) or already lives on
       the CPU -- ``.tolist()`` is then a pure host read.

    A device warm mask with no usable host twin falls back to reading the
    tensor (a per-call D2H sync): correct everywhere, and outside the
    engine -- which always exposes the twin -- it only re-pays the sync
    the engine path just avoided.
    """
    host_values = getattr(metadata, "linear_state_validity_host_values", None)
    if host_values is not None and len(host_values) == num_seqs:
        return [1 if value else 0 for value in host_values]
    if not isinstance(warm, torch.Tensor):
        return [1 if value else 0 for value in warm]
    # No host twin and a device mask: this .tolist() is a per-call D2H sync.
    # Correct everywhere, but on the CP prefill critical path it is exactly the
    # cost the host twin exists to remove -- surface it so a deployment missing
    # the twin shows up in the logs instead of looking like generic CP slowness.
    # Once per process: this helper runs on every prefill forward, so an
    # unconditional warning would storm the log on such a deployment.
    global _KDA_CP_WARM_TWIN_WARNED
    if not _KDA_CP_WARM_TWIN_WARNED:
        _KDA_CP_WARM_TWIN_WARNED = True
        logger.warning(
            "KDA CP: metadata exposes no linear_state_validity_host_values; reading the device "
            "warm mask re-pays a per-call D2H sync on the prefill path (this warning is one-shot)"
        )
    return [1 if value else 0 for value in warm.view(-1).tolist()]


def kda_cp_execute_linear(
    mixed_qkv: torch.Tensor,
    beta_raw: torch.Tensor,
    layer: Attention,
    raw_gate_proj: torch.Tensor,
    cp_context: CpContext,
    metadata: object,
    conv_cache: torch.Tensor,
    ssm_cache: torch.Tensor,
    idx: torch.Tensor,
    read_idx: torch.Tensor,
    chunk_kda_fwd: object,
) -> torch.Tensor:
    """Two-pass KDA CP prefill over LOCAL zigzag rows (M1 scheme, M2 glue).

    Pipeline per rank (segments = the rank's two zigzag halves per sequence,
    in local (sequence, half) order):
      1. select this rank's real rows (query_index) from the local tensors;
      2. conv halo exchange, issued ASYNC and FIRST: the payload packs
         from the RAW channel-first stream via plan-cached tail positions
         (two tiny selects, before any row staging), so the exchange is in
         flight while the much larger row staging copy runs (all-gather
         over the CP group -- every rank contributes its two conv tails);
         then run ONE conv call over the packed real rows with
         per-segment left-context states (cold rank-0-of-sequence segments
         use the kernel's cold mode, NOT zeros+mode-1 -- the m1
         bit-exactness caveat);
      3. batched summary: ONE xllm_chunk_kda_fwd_prepare call over the
         rank's whole local stream emits the WY factors (the prepare-only
         stage of the fused op, on the same raw inputs and flags -- no
         state/output semantics), and ONE xllm_pre_process_fwd_kernel_
         merged call folds them into hm=[E|M] fp32 with one affine chain
         per cu_seqlens segment (k slot = the PRE-GATED kg); the combined
         cu_seqlens is the local seg_cu, whose boundaries restart the
         64-row chunk grid exactly like the per-segment calls it replaced
         (M-B2: identical values, half the kernel launches);
      4. ONE all-gather of the summaries, issued ASYNC immediately after
         the fold: each rank contributes [2, n_seqs, HV, K, V+K] (row 0 =
         first-half, row 1 = second-half), so the gathered tensor is
         [2*cp, n_seqs, HV, K, V+K] with row ``rank*2 + half``; causal
         chunk j lives at row ``j*2`` for j < cp and ``(2*cp-1-j)*2 + 1``
         otherwise;
      5. zero-prior replay overlap: while the summary exchange is in
         flight, the local segments whose merge consumes no prior
         summaries (rank 0's first halves -- initial state = the local
         prefix/zero base) replay; then the gather is waited on;
      6. causal merge: fold X <- X@M_stored+E_stored starting from the
         prefix state read from the SSM read slot (zeros when cold), over
         the chunks strictly before each remaining segment -- ONE
         aclnnMergeFwdBwdKernel launch per prefix state by default
         (the pure-torch Hillis-Steele tree scan runs only outside the
         merge op's shape budget);
      7. replay: ONE batched chunk_kda_fwd over the remaining segments
         with the merged per-segment initial states (varlen batched ==
         per-segment calls, bit-exact per
         test_chunk_kda_varlen_matches_independent_sequences; splitting
         the batch by the same argument is bit-exact too, which is what
         step 5 relies on);
      8. final-state broadcast: the owner of the causally-LAST chunk
         (kda_cp_chunk_owner(2*cp-1) = rank 0's second segment at any cp)
         broadcasts the final conv + ssm state as ONE merged fp32 payload
         (bit-exact for bf16/fp16/fp32 conv payloads); every rank writes
         the SAME broadcast tensors into its own slots, restoring the
         bit-identical replica invariant the legacy merge provided
         (load-bearing for PD one-writer reshard, MTP and the prefix
         cache);
      9. state writes mirror the legacy prefill path exactly: conv
         committed columns only, SSM base slot only (idx*stride);
         speculative checkpoint columns are never touched.

    All reductions are transport-only: the exchanged values and
    every kernel input are identical to a synchronous schedule.
    The
    per-shape host tensors (cu_seqlens views, early/late row packs,
    causal row indices, the halo tail positions) come from a bounded
    per-shape plan cache (_kda_cp_host_plan): one request's chunks reuse
    one plan. Row staging is ONE strided copy off the raw channel-first
    stream (the packed real rows are a contiguous prefix of the local
    stream under the gate -- _kda_cp_rows_are_stream_prefix guards it and
    falls back to the general gather), and the warm mask never crosses to
    the host: the conv call's ism comes from the metadata's host validity
    twin (or the host forms of the mask), so the call path carries no
    per-call D2H sync (g1 removed the tolist; n6 removed the conv op's
    internal marshalling syncs by passing CPU int64 metadata -- the C++
    wrapper's .to(kCPU, kInt64) is a no-op on tensors already there).

    The replay outputs land directly in the local row layout (g2): under
    the gate every segment is exactly full, so the packed (sequence,
    half) order IS the local layout -- the early/late reassembly writes
    the returned tensor itself (one index_copy_ per replay output, no
    intermediate packed buffer, no second full-tensor scatter), and the
    broadcast payload reads the owning replay's final states verbatim
    instead of a reassembled final_states tensor. A total_real ==
    total_local guard keeps padded fallbacks for ragged tails the gate
    excludes.

    Sched-overlap ordering is preserved by construction: the halo/summary
    work consumes only local activations, the prefix-state read happens
    inside the model forward (after the deferred linear-state restore on the
    compute stream), and the writes land in the write slots before the layer
    returns.

    Returns ``[1, total_local, num_heads_local, head_dim]`` local rows with
    padding rows zero -- the same contract cp_shard_rows gave the model
    layer, so the CP path skips the merge AND the re-shard.
    """
    from xllm.python import distributed
    from xllm.python.kernels import l2_norm

    prepare_op, summary_op, merge_op = _kda_cp_ops()
    if prepare_op is None or summary_op is None or merge_op is None:
        raise RuntimeError("KDA CP executed without the xllm CP ops (gate violated)")

    cp_size = int(cp_context.cp_size)
    cp_rank = int(cp_context.cp_rank)
    device = mixed_qkv.device
    conv_kernel_size = int(layer.conv_kernel_size)
    conv_state_len = conv_kernel_size - 1
    head_dim = int(layer.head_dim)
    num_heads_local = int(layer.num_heads_local)
    qkv_dim = int(layer.qkv_dim)
    conv_dim = int(layer.conv_dim)
    num_seqs = int(idx.shape[0])
    scale = head_dim**-0.5
    if idx.dtype != torch.int64:
        idx = idx.to(torch.int64)
    if read_idx is not None and read_idx.dtype != torch.int64:
        read_idx = read_idx.to(torch.int64)

    # ---- segment plan (host) ----------------------------------------------
    # The gate guarantees every (sequence, half) segment is non-empty, so the
    # rank-local q_cu_seqlens carries exactly 2*num_seqs cumulative counts in
    # local (sequence, half) order (empty segments are omitted by the CP
    # context builder; the gate excluded them).
    seg_cu = [0, *cp_context.q_cu_seqlens]
    if len(seg_cu) != 2 * num_seqs + 1:
        raise RuntimeError(f"KDA CP segment plan mismatch: {len(seg_cu) - 1} segments for {num_seqs} sequences")
    total_real = int(seg_cu[-1])
    if total_real != int(cp_context.query_index.numel()):
        raise RuntimeError("KDA CP real-row count does not match the CP context query index")
    # g2: whether the packed real rows tile the local layout exactly (no
    # padding rows). The zigzag slots are (sequence, half)-ordered with
    # real rows front-packed and every segment's real count at most its
    # slot length (xllm_ops::build_cp_context_npu), so total_real ==
    # total_local forces every segment exactly full and query_index to
    # the identity [0, total_real) -- the invariant the 64-row
    # eligibility gate guarantees for every forward it admits (the proof
    # note in _kda_cp_rows_are_stream_prefix). A ragged tail fails the
    # comparison and takes the guarded fallbacks below, so neither the
    # reassembly nor the return path rests on the gate alone.
    rows_tile_local = total_real == int(cp_context.total_local)

    global _KDA_CP_PROBE_EMITTED
    fold_mode = _kda_cp_fold_mode(head_dim, head_dim, num_seqs * num_heads_local)
    if not _KDA_CP_PROBE_EMITTED:
        _KDA_CP_PROBE_EMITTED = True
        logger.info(
            f"KDA_CP_ACTIVE segments={2 * num_seqs} rows={total_real} "
            f"msrc=prepare1+ppfm1+{fold_mode} desccache=on "
            "halo=allgather gather=async"
        )

    host_plan = _kda_cp_host_plan(seg_cu, cp_rank, cp_size, conv_state_len, device)

    # ---- conv halo exchange, issued FIRST (off the raw stream) --------------
    # raw_rows is the channel-first projection transposed to row-major -- a
    # FREE view at the engine's batch=1 (the split+transpose upstairs is
    # strided, so the reshape only relabels it; batch>1 flattens through a
    # copy exactly like the previous single-statement form). Outgoing
    # payload [2, num_seqs, conv_state_len, conv_dim]: row 0 = every
    # sequence's first-half tail, row 1 = every second-half tail (half-
    # major, matching the summary payload below). All ranks contribute the
    # same shape (the gate pinned every segment >= 64 rows), so the
    # exchange over the CP group (the same group cp_merge_rows uses) is
    # well-formed: the gathered tensor is [2*cp, num_seqs,
    # conv_state_len, conv_dim] with row ``rank*2 + half``.
    #
    # g1 reorder: the payload packs from THIS raw view (two tiny selects:
    # the plan-cached halo_pos picks each segment's last conv_state_len
    # packed positions out of query_index, then one gather stages them) --
    # identical values to the per-segment loop it replaces, but the
    # exchange is in flight BEFORE the much larger row-staging copy is
    # issued, so the wait below returns at the collective instead of at
    # the staging copy's completion. Removing the per-call D2H sync alone
    # would only migrate the drain block to that wait; this reorder is
    # what lets the row-staging device time overlap it.
    raw_rows = mixed_qkv.transpose(1, 2).reshape(-1, conv_dim)
    cp_group = distributed.cp_process_group(device)
    halo_payload = raw_rows.index_select(0, cp_context.query_index.index_select(0, host_plan.halo_pos)).view(
        2, num_seqs, conv_state_len, conv_dim
    )
    gathered_halo, halo_work = _kda_cp_async_all_gather(cp_group, halo_payload, cp_size)

    # ---- local real rows in (sequence, half) packed order ------------------
    # ONE strided copy: under the eligibility gate every segment is whole
    # 64-row chunks, so the packed real rows are a contiguous PREFIX of the
    # local stream and narrow+contiguous moves each element once (~2.5x
    # cheaper than the index_select gather at the c8192 shape; bit-exact --
    # the probe in the task research pins both forms equal). The guard
    # falls back to the general gather whenever the prefix invariant does
    # not hold, so correctness never rests on the gate alone.
    if _kda_cp_rows_are_stream_prefix(seg_cu, mixed_qkv):
        rows = raw_rows[:total_real].contiguous()
    else:
        # The stream-prefix fast path only holds for shape [1, S, C]; anything
        # else (multi-batch packing, an unpredicted layout) falls back to the
        # gathering form. Logged once so an unexpected layout shows up without
        # spamming a debug-enabled run.
        global _KDA_CP_ROWS_FALLBACK_LOGGED
        if not _KDA_CP_ROWS_FALLBACK_LOGGED:
            _KDA_CP_ROWS_FALLBACK_LOGGED = True
            logger.debug(
                "KDA CP: row staging took the index_select fallback (mixed_qkv shape %s, "
                "segments=%d); the stream-prefix fast path requires shape [1, S, C]",
                tuple(mixed_qkv.shape),
                len(seg_cu) - 1,
            )
        rows = raw_rows.index_select(0, cp_context.query_index)
    beta_flat = beta_raw.reshape(-1, num_heads_local).index_select(0, cp_context.query_index)
    gate_flat = raw_gate_proj.reshape(-1, num_heads_local, head_dim).index_select(0, cp_context.query_index)

    # ---- per-segment initial conv states -------------------------------------
    # Prefix conv states from the read slot (the same view the legacy path
    # reads). has_initial_state semantics mirror the legacy path exactly:
    # a per-sequence mask zeroes cold rows; when the metadata carries no
    # mask the read state is used as-is (treated warm). Rank 0's first
    # segments additionally take the kernel's COLD mode (has_state=0)
    # instead of zeros+mode-1: mode-1 with a zeroed buffer is NOT
    # bit-identical to the cold path (m1 caveat), and the cold path is the
    # one the full-stream reference uses. DIVERGENCE NOTE: the ENGINE's
    # legacy merged path runs the xllm_ops causal_conv1d wrapper, which
    # forces initial_state_mode=1 with the zeroed buffer on cold sequences
    # -- so a cold sequence's FIRST chunk conv (and only it) may differ
    # from the flag-off engine output at ulp level. This is a further
    # flag-on divergence source inside the same 8e-3 tolerance budget the
    # engine oracle measures; it is not bit-exact and is not claimed to be.
    ssm_checkpoint_stride = linear_state_checkpoint_stride(conv_cache, ssm_cache)
    conv_prefix = conv_cache[:, :conv_state_len].index_select(0, read_idx).to(rows.dtype)
    ssm_prefix = ssm_cache.index_select(0, read_idx * ssm_checkpoint_stride)
    warm = getattr(metadata, "has_initial_state", None)
    warm_values: list[int] | None = None
    if warm is not None and len(warm) == num_seqs:
        # n6: resolve the conv call's ism values HOST-side (the metadata
        # twin / host mask forms) BEFORE the tensor wrap below -- the conv
        # op consumes ism as a host IntArrayRef, so a device-side mask
        # would have to be read back with a D2H sync.
        warm_values = _kda_cp_warm_host_values(metadata, warm, num_seqs)
        if not isinstance(warm, torch.Tensor):
            warm = torch.tensor(warm, dtype=torch.int64, device=conv_prefix.device)
        warm = warm.to(torch.bool).view(num_seqs, 1, 1)
        conv_prefix = torch.where(warm, conv_prefix, torch.zeros_like(conv_prefix))
        ssm_prefix = torch.where(warm.view(num_seqs, 1, 1, 1), ssm_prefix, torch.zeros_like(ssm_prefix))
        # g1: conv_prefix zeroes cold rows via THIS device mask. The
        # per-call ``warm.tolist()`` it replaced was the compute-stream
        # drain point (the G0 round's dominant glue stamp, 60-66% of the
        # c2048 residual / 88-91% at c8192); the masked conv_prefix already
        # zeroes cold rows, so writing it wholesale into conv_state is
        # bit-identical to the per-sequence warm-only copies it replaces.
    else:
        # No mask: mirror the legacy path and use the read state as-is.
        warm_values = None

    conv_state = rows.new_zeros(2 * num_seqs, conv_state_len, conv_dim)
    # CPU int64: the conv op's qsl/si/ism marshalling (``.to(torch::kCPU,
    # torch::kInt64)`` in npu_ops_library.cpp) is a NO-OP on a tensor that
    # is already cpu+int64+contiguous, so the aclnn launch reads the modes
    # without blocking the host behind the compute stream (n6: with the
    # g1 tolist gone, the conv call had become the FIRST host-blocking
    # point after the halo wait -- its internal D2H metered the whole
    # upstream device backlog into the KDA phase, 11.9 ms mean at c8192).
    # All segments default to warm (mode 1); only rank 0's first segments
    # can be cold (the fill below).
    has_state = torch.ones(2 * num_seqs, dtype=torch.int64)
    first_chunk = cp_rank
    second_chunk = 2 * cp_size - 1 - cp_rank
    # The halo exchange completes here: everything above only consumed
    # local activations (the g1 order packs the halo off the raw stream,
    # so even the row staging copy only overlaps the wait), and the
    # prefix reads and warm-mask handling rode the issue-to-wait window.
    halo_work.wait()
    if first_chunk == 0:
        # The sequences' first chunks: warm prefix state, or cold start
        # (cold rows already carry the zeroed mask above). The ism values
        # come from the host-side warm_values -- identical to the device
        # mask the g1 form filled from (same source vector, same 0/1
        # ints), so the marshalled IntArrayRef is bit-identical.
        conv_state[0::2] = conv_prefix
        if warm_values is not None:
            has_state[0::2] = torch.as_tensor(warm_values, dtype=torch.int64)
    else:
        prior_rank, prior_half = kda_cp_chunk_owner(first_chunk - 1, cp_size)
        conv_state[0::2] = gathered_halo[prior_rank * 2 + prior_half]
    # The second segments always have a causal predecessor chunk.
    prior_rank, prior_half = kda_cp_chunk_owner(second_chunk - 1, cp_size)
    conv_state[1::2] = gathered_halo[prior_rank * 2 + prior_half]

    # One conv call over the packed real rows of BOTH segments; empty
    # state_indices keeps the per-sequence conv_state convention (no cache
    # pool), matching the fla-validated aclnnCausalConv1d calling form the
    # legacy causal_conv1d wrapper also uses. The cu/empty-index tensors
    # come from the host plan (constant across a request's chunks) and
    # has_state is built host-side above -- all three arrive as CPU int64
    # tensors, so the op's marshalling is sync-free (n6).
    conv_out = torch.ops.xllm_ops.causal_conv1d_prefill(
        rows,
        layer.conv_weight_t,
        conv_state,
        host_plan.empty_i64,
        has_state,
        host_plan.conv_cu,
    )
    # conv_state now holds each segment's final rolling window (in-place).

    # ---- per-segment q/k/v + gates -------------------------------------------
    hidden_shape = (1, total_real, num_heads_local, head_dim)
    query = conv_out[:, :qkv_dim].reshape(hidden_shape)
    key = conv_out[:, qkv_dim : 2 * qkv_dim].reshape(hidden_shape)
    value = conv_out[:, 2 * qkv_dim :].reshape(hidden_shape)
    q_in = l2_norm(query)
    k_in = l2_norm(key)
    v_in = value.to(query.dtype).contiguous()
    beta_all = beta_flat.float().sigmoid().view(1, total_real, num_heads_local)

    forget_gate = layer.forget_gate
    gate_lb = forget_gate.safe_gate_lower_bound
    fuse_gate = raw_gate_proj is not None and gate_lb is not None and -5.0 <= gate_lb < 0.0
    if fuse_gate:
        g_all = gate_flat.view(hidden_shape)
        gate_kwargs = dict(
            A_log=forget_gate.A_log.contiguous(),
            dt_bias=forget_gate.dt_bias.contiguous(),
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=gate_lb,
        )
    else:
        g_all = forget_gate.gate_from_raw(gate_flat.view(hidden_shape)).to(torch.float32)
        gate_kwargs = dict(use_gate_in_kernel=False)

    # ---- batched summary: ONE prepare + ONE ppfm over the local stream ----
    # The factor pass runs OUR prepare-only op (ma2-2) on the same raw
    # inputs the fused factor pass consumed: prepare has no state/output
    # semantics, so initial_state/output_final_state/disable_recompute/
    # state_v_first drop out and every shared flag carries over 1:1 (ma1
    # section 7 flag mapping). Since M-B2 both calls run ONCE over the
    # rank's whole local stream instead of once per segment (the ~1 ms
    # fixed launch cost dominated the per-segment calls at the engine's
    # 256-row spans): the rank-local rows already hold the two zigzag
    # segments of every sequence contiguously, so seg_cu doubles as the
    # combined cu_seqlens, and the canonical chunk_indices the prepare
    # wrapper derives from it enumerate each segment's 64-row chunks with
    # the grid restarting at every boundary -- exactly the per-segment
    # grids, so the factor VALUES are identical (varlen-native contract,
    # the same one the deployed fused op uses for the merged stream; the
    # fused op's packed-varlen vs per-sequence bit-exactness pin
    # test_chunk_kda_varlen_matches_independent_sequences covers the
    # prepare stage inside it, and the NPU identity test pins our pair
    # directly -- task research mb2).
    # result = (gk, Aqk, Akk, w, u, qg, kg, qg_scaled, q_hat, k_hat,
    # q_rstd, k_rstd, beta_eff); the ma1 fused->prepare slot mapping is
    # gk 2->0, w 5->3, u 6->4, kg 8->6, and ppfm's k slot consumes the
    # PRE-GATED kg (M1 convention pin: plain k is the wrong convention by
    # 0.198 rel_fro). The factors differ from the fused emission at 1-ulp
    # (see the NUMERICS note above).
    factors = prepare_op(
        q_in.contiguous(),
        k_in.contiguous(),
        v_in.contiguous(),
        g_all.contiguous(),
        beta_all.contiguous(),
        scale,
        chunk_size=KDA_CP_CHUNK_SIZE,
        layout="BSND",
        cu_seqlens=seg_cu,
        backward_mode="none",
        **gate_kwargs,
    )
    # ppfm is multi-segment BY DESIGN: hm[Nseq, HV, K, V+K] fp32, one
    # affine chain per cu_seqlens interval (the ATK contract; validated on
    # the real kernel by the m0a driver's one-call multi-segment run). Row
    # i is segment i's [E | M] summary in the SAME (sequence, half)
    # interleaved order the per-segment loop produced.
    # Name the prepare slots once (see the mb2 slot mapping above) instead of
    # bare indices at the call site.
    kg_gated, w_factor, u_factor, gk_factor = factors[6], factors[3], factors[4], factors[0]
    segment_summaries = summary_op(
        kg_gated,
        w_factor,
        u_factor,
        gk=gk_factor,
        cu_seqlens=seg_cu,
        chunk_size=KDA_CP_CHUNK_SIZE,
    )

    # ---- ONE all-gather of the summaries, issued ASYNC -------------------------
    # The batched hm rows are (sequence, half)-interleaved (s0h0, s0h1,
    # s1h0, s1h1, ...); view(n_seqs, 2, ...).transpose(0, 1) reorders them
    # to [2, n_seqs, ...]: row 0 = every sequence's first-half summary,
    # row 1 = every second-half. The gathered tensor is
    # [2*cp, n_seqs, HV, K, V+K], row rank*2 + half. Issued async right
    # after the fold so the host keeps enqueueing: the zero-prior replay
    # below runs while the exchange is in flight (mb-probe: async handle +
    # partial compute overlap on HCCL).
    local_summaries = segment_summaries.view(num_seqs, 2, *segment_summaries.shape[1:]).transpose(0, 1).contiguous()
    gathered_summaries, summary_work = _kda_cp_async_all_gather(cp_group, local_summaries, cp_size)

    # ---- zero-prior replay (overlaps the summary exchange) ----------------------
    # Segments whose causal merge consumes no prior summaries replay NOW:
    # their initial state is the local prefix/zero base (exactly what the
    # fold with prior_chunks == 0 returns), so they never read the gathered
    # summaries. Splitting the batched replay is bit-exact (batched varlen
    # == per-segment calls, the existing kernel-test pin); rank 0 is the
    # only rank with such segments (its first halves).
    early_segments = _kda_cp_zero_prior_segments(cp_rank, cp_size, num_seqs)
    early_output: torch.Tensor | None = None
    early_final: torch.Tensor | None = None
    early_row_index: torch.Tensor | None = None
    if early_segments:
        early_row_index = host_plan.early_rows
        early_result = chunk_kda_fwd(
            q_in.index_select(1, early_row_index),
            k_in.index_select(1, early_row_index),
            v_in.index_select(1, early_row_index),
            g_all.index_select(1, early_row_index),
            beta_all.index_select(1, early_row_index),
            scale,
            chunk_size=KDA_CP_CHUNK_SIZE,
            layout="BSND",
            initial_state=ssm_prefix,
            output_final_state=True,
            cu_seqlens=host_plan.early_cu,
            state_v_first=True,
            **gate_kwargs,
        )
        early_output = early_result[0].to(query.dtype)
        early_final = early_result[1]

    summary_work.wait()

    # ---- causal merge --------------------------------------------------------
    # Both local halves fold from the same sequence base (prefix state, or
    # zeros for cold sequences) over the chunks strictly before them: chunks
    # [0, cp_rank) for the first half, [0, 2*cp-1-cp_rank) for the second.
    # Rank 0's first half needs no fold (zero priors -- the fold would
    # return the base unchanged), and those segments already replayed.
    # The fold implementation is selected by _kda_cp_fold_mode: the merge
    # op (one aclnn launch per prefix state, default) or the pure-torch
    # Hillis-Steele tree scan (outside the merge op's shape budget).
    prior_counts = (second_chunk,) if first_chunk == 0 else (first_chunk, second_chunk)
    if fold_mode == "merge":
        folded = kda_cp_merge_fold_initial_states(
            merge_op, ssm_prefix, gathered_summaries, host_plan.causal_rows, prior_counts
        )
    else:
        causal_summaries = gathered_summaries.index_select(0, host_plan.causal_rows)
        folded = kda_cp_tree_scan_fold_initial_states(ssm_prefix, causal_summaries, prior_counts)
    state_second = folded[-1]
    state_first = ssm_prefix if first_chunk == 0 else folded[0]
    interleaved_states = (
        torch.stack((state_first, state_second), dim=1).reshape(2 * num_seqs, *state_first.shape[1:]).contiguous()
    )

    # ---- replay: the remaining segments -----------------------------------------
    # With no early segments this is exactly the previous single batched
    # call (same tensors, same cu_seqlens). Otherwise the late segments
    # replay on their packed rows with their folded initial states, and
    # the early/late outputs scatter straight into the local row layout
    # (g2 -- no intermediate packed-order buffer).
    if not early_segments:
        result = chunk_kda_fwd(
            q_in,
            k_in,
            v_in,
            g_all,
            beta_all,
            scale,
            chunk_size=KDA_CP_CHUNK_SIZE,
            layout="BSND",
            initial_state=interleaved_states,
            output_final_state=True,
            cu_seqlens=host_plan.replay_cu,
            state_v_first=True,
            **gate_kwargs,
        )
        core_attn_out = result[0].to(query.dtype)
        # The batched replay emits both halves' finals (sequence, half)
        # interleaved; the broadcast payload below picks the half it
        # owns. Nothing else reads them (g2: the packed final_states
        # tensor this fed disappeared with the reassembly).
        first_half_finals = result[1][0::2]
        second_half_finals = result[1][1::2]
    else:
        late_rows = host_plan.late_rows
        late_initial = interleaved_states.index_select(0, host_plan.late_idx)
        late_result = chunk_kda_fwd(
            q_in.index_select(1, late_rows),
            k_in.index_select(1, late_rows),
            v_in.index_select(1, late_rows),
            g_all.index_select(1, late_rows),
            beta_all.index_select(1, late_rows),
            scale,
            chunk_size=KDA_CP_CHUNK_SIZE,
            layout="BSND",
            initial_state=late_initial,
            output_final_state=True,
            cu_seqlens=host_plan.late_cu,
            state_v_first=True,
            **gate_kwargs,
        )
        # g2: reassemble straight into the LOCAL row layout. The packed
        # (sequence, half) buffer the old reassembly built was only ever
        # an intermediate permutation -- the scatter then copied it a
        # second time into the local layout. Under the eligibility gate
        # every segment is exactly full, so the packed row order IS the
        # local layout and the plan-cached row packs write directly into
        # the returned tensor; a ragged tail (excluded by the gate; the
        # guard keeps it honest) composes the row packs with query_index
        # so each row lands exactly where the two-stage chain put it.
        assert early_output is not None and early_final is not None and early_row_index is not None
        if rows_tile_local:
            out_rows = total_real
            early_dest = early_row_index
            late_dest = late_rows
        else:
            out_rows = int(cp_context.total_local)
            early_dest = cp_context.query_index.index_select(0, early_row_index)
            late_dest = cp_context.query_index.index_select(0, late_rows)
        core_attn_out = early_output.new_zeros(1, out_rows, num_heads_local, head_dim)
        core_attn_out.index_copy_(1, early_dest, early_output)
        core_attn_out.index_copy_(1, late_dest, late_result[0].to(query.dtype))
        # g2: the broadcast payload needs only one half's finals in
        # sequence order (the causally-last chunk's). This rank's early
        # segments are exactly the first halves and its late segments
        # exactly the second halves, so each replay's own final-state
        # output IS that half's payload: the packed final_states
        # reassembly (new_zeros + two index_copy_) and the strided
        # [half::2] pick it fed both disappear. The first-half finals are
        # dead in the taken branch (only chunk 2*cp-1's state reaches the
        # cache) but stay wired for the never-taken opposite-half root.
        first_half_finals = early_final
        second_half_finals = late_result[1]

    # ---- final-state broadcast (restore the replica invariant) ------------------
    # The causally-LAST chunk's owner broadcasts; under zigzag that is rank
    # 0's second segment at every cp size (computed from the segment order,
    # never assumed from the rank id). conv_state keeps its (sequence,
    # half)-interleaved layout (odd rows = second halves), and the finals
    # arrive as per-half picks in sequence order (g2).
    # The conv and ssm payloads share one root and one group, so they merge
    # into ONE broadcast over a flat fp32 buffer (bit-exact: bf16/fp16 conv
    # values round-trip fp32 promotion exactly); non-representable conv
    # dtypes keep the two-broadcast form.
    broadcast_rank, broadcast_half = kda_cp_chunk_owner(2 * cp_size - 1, cp_size)
    # g2: .contiguous() copies the strided batched-replay picks exactly
    # like the clones they replace, and is a FREE pass-through for the
    # early/late finals (fresh contiguous kernel outputs).
    # Zigzag pins the causally-last owner to (rank 0, half 1) for every
    # cp_size >= 2 (kda_cp_chunk_owner(2*cp-1, cp) -> (0, 1), pinned by
    # test_kda_cp_chunk_owner_is_a_permutation_with_balanced_pairs), so the
    # half-0 form below is the symmetry keeper for a future chunk order --
    # not dead weight.
    if broadcast_half == 0:
        payload_conv = conv_state[0::2].clone()
        payload_ssm = first_half_finals.contiguous()
    else:
        payload_conv = conv_state[1::2].clone()
        payload_ssm = second_half_finals.contiguous()
    packed = _kda_cp_pack_state_broadcast(payload_conv, payload_ssm)
    if packed is not None:
        flat, conv_elements = packed
        if cp_rank == broadcast_rank:
            broadcast_flat = flat
        else:
            broadcast_flat = torch.empty_like(flat)
        distributed.broadcast_(broadcast_flat, broadcast_rank, "cp")
        broadcast_conv, broadcast_ssm = _kda_cp_unpack_state_broadcast(
            broadcast_flat,
            conv_elements,
            payload_conv.dtype,
            payload_conv.shape,
            payload_ssm.shape,
        )
    else:
        if cp_rank == broadcast_rank:
            broadcast_conv = payload_conv
            broadcast_ssm = payload_ssm
        else:
            broadcast_conv = torch.empty_like(payload_conv)
            broadcast_ssm = torch.empty_like(payload_ssm)
        distributed.broadcast_(broadcast_conv, broadcast_rank, "cp")
        distributed.broadcast_(broadcast_ssm, broadcast_rank, "cp")

    # ---- state writes (mirror the legacy path's slots exactly) -------------------
    conv_cache[:, :conv_state_len].index_copy_(0, idx, broadcast_conv.to(conv_cache.dtype))
    ssm_cache.index_copy_(
        0,
        idx * ssm_checkpoint_stride,
        broadcast_ssm.float().contiguous(),
    )

    # ---- scatter outputs back into the local row layout --------------------------
    # g2: the output already IS the local row layout whenever the early
    # reassembly ran (it scatters straight into it) or every segment is
    # exactly full (the batched replay's packed order tiles the layout;
    # the 64-row gate forces this for every admitted forward, and only
    # the causally-last chunk's owner -- the early rank -- can ever carry
    # a ragged tail). The padded new_zeros + full-tensor index_copy_
    # remain as the defensive fallback the guard keeps honest.
    if early_segments or rows_tile_local:
        return core_attn_out
    scattered = core_attn_out.new_zeros(cp_context.total_local, num_heads_local, head_dim)
    scattered.index_copy_(0, cp_context.query_index, core_attn_out.reshape(total_real, num_heads_local, head_dim))
    return scattered.unsqueeze(0)


class KdaLinearAttentionMixin:
    """KDA linear-attention + MTP spec-verify state I/O, mixed into NpuPagedAttentionBackend."""

    def _causal_conv1d(
        self,
        value: torch.Tensor,
        state: torch.Tensor,
        layer: Attention,
        *,
        query_start_loc: list[int] | None = None,
        is_prefill: bool = False,
    ) -> torch.Tensor:
        """Convolve [batch, tokens, channels] values and update channel-last states."""
        inputs = value.contiguous()
        if query_start_loc is not None:
            inputs = inputs.reshape(-1, inputs.shape[-1])
        output = torch.ops.xllm_ops.causal_conv1d(
            inputs,
            layer.conv_weight_t,
            state,
            query_start_loc if query_start_loc is not None else [],
            1 if layer.activation == "silu" else 0,
            0 if is_prefill else 1,
        )
        return output.reshape(value.shape)

    def execute_linear(
        self,
        mixed_qkv: torch.Tensor,
        beta_raw: torch.Tensor,
        layer: Attention,
        raw_gate_proj: torch.Tensor,
        *,
        cp_context: CpContext | None = None,
    ) -> torch.Tensor:
        """KDA delta-rule over framework conv/ssm slots.

        Returns ``[B, S, num_heads_local, head_dim]``. The conv1d + delta-rule
        math is identical to the model-layer self-contained path (validated
        against the transformers reference); only the state I/O moved here so
        both attention layer types dispatch through the backend.

        ``raw_gate_proj`` is the pre-gate projection ``f_b(f_a(x))`` and
        ``beta_raw`` is the pre-sigmoid beta projection. Recurrent decode and
        verify kernels consume both raw tensors; chunk-prefill materializes
        only the FP32 beta required by its kernel.

        ``cp_context`` is passed by the model layer ONLY when its KDA-CP gate
        passed (enable_kda_cp on, prefill-shaped, eligible batch): the
        tensors are then LOCAL zigzag rows and this method dispatches to the
        two-pass CP path, returning local rows directly. Any other caller
        gets the legacy merged-row path byte-identically.
        """
        from fla_npu.ops.ascendc import chunk_kda_fwd, recurrent_kda

        from xllm.python.kernels import l2_norm

        metadata = self._metadata
        assert metadata is not None, "execute_linear called before prepare()"
        layer_cache = self._kv_caches[layer.layer_id]
        conv_cache = layer_cache.conv
        ssm_cache = layer_cache.ssm
        assert conv_cache is not None and ssm_cache is not None, (
            "execute_linear requires a linear-attention layer cache (conv/ssm)"
        )

        batch_size, _, seq_len = mixed_qkv.shape
        conv_kernel_size = layer.conv_kernel_size
        conv_state_len = conv_kernel_size - 1
        head_dim = layer.head_dim
        num_heads_local = layer.num_heads_local
        qkv_dim = layer.qkv_dim
        hidden_shape = (batch_size, seq_len, -1, head_dim)

        fg = getattr(layer, "forget_gate", None)
        gate_lb = getattr(fg, "safe_gate_lower_bound", None) if fg is not None else None
        read_idx, idx = resolve_linear_state_io_indices(metadata)
        is_prefill = metadata.is_prefill or metadata.is_chunked_prefill
        is_spec_verify = getattr(metadata, "is_spec_verify", False)
        if is_spec_verify and idx is None:
            raise RuntimeError("KDA spec-verify requires linear-state slots")
        num_seqs = idx.shape[0] if idx is not None else batch_size
        if cp_context is not None:
            # The model layer hands cp_context over only after its own
            # kda_cp_prefill_eligible check passed; re-running the ONE shared
            # gate here turns any disagreement (a future caller bypassing the
            # model gate) into a loud error instead of silent corruption --
            # a legacy fallthrough would treat local rows as global ones.
            if idx is None:
                raise RuntimeError("KDA context-parallel prefill requires linear-state slots")
            if not kda_cp_prefill_eligible(cp_context, metadata, layer):
                raise RuntimeError("KDA CP path received an ineligible forward (model/backend gate disagreement)")
            return kda_cp_execute_linear(
                mixed_qkv,
                beta_raw,
                layer,
                raw_gate_proj,
                cp_context,
                metadata,
                conv_cache,
                ssm_cache,
                idx,
                read_idx,
                chunk_kda_fwd,
            )
        # ACL-graph decode with a flattened batch: the model forward unsqueezes
        # the 1-D ``[num_seqs]`` decode input into ``[1, num_seqs]``, so
        # mixed_qkv arrives as ``[1, conv_dim, num_seqs]`` with idx
        # ``[num_seqs]``. The eager multi-sequence branch (a Python loop over
        # q_cu_seq_lens with host syncs) is not graph-capturable; decode is
        # exactly one token per sequence, so reshape to ``[num_seqs, conv_dim,
        # 1]`` and take the simple per-sequence path with static shapes. gate
        # ``[1, T, nh, hd]`` / beta ``[1, T, nh]`` follow the same transpose.
        in_graph = in_acl_graph()
        is_decode = not metadata.is_prefill and not metadata.is_chunked_prefill
        flatten_graph_decode = (
            in_graph
            and is_decode
            and self._kda_verify_width == 1
            and idx is not None
            and batch_size == 1
            and num_seqs > 1
            and seq_len == num_seqs
            # An expanded spec-verify batch keeps the [1, C, T] packing even
            # though its per-row slot count equals T: the KDA verify grouping
            # reads [B, C, rows_per_seq] from it (see the dispatch below).
            and getattr(metadata, "expanded_decode_metadata", None) is None
        )
        if flatten_graph_decode:
            mixed_qkv = mixed_qkv.transpose(0, 2)  # [1, C, T] -> [T, C, 1]
            batch_size, _, seq_len = mixed_qkv.shape
            hidden_shape = (batch_size, seq_len, -1, head_dim)
            beta_raw = beta_raw.transpose(0, 1)  # [1, T, nh] -> [T, 1, nh]
            # Plain graph decode (spec-verify is excluded above), so the gate is
            # never materialized here; only the raw projection the kernel fuses
            # follows the transpose. [1, T, nh, hd] -> [T, 1, nh, hd].
            raw_gate_proj = raw_gate_proj.transpose(0, 1)
        if idx is None:
            device = mixed_qkv.device
            conv_state = torch.zeros(
                batch_size,
                conv_state_len,
                layer.conv_dim,
                dtype=mixed_qkv.dtype,
                device=device,
            )
            ssm_state = torch.zeros(
                batch_size,
                num_heads_local,
                head_dim,
                head_dim,
                dtype=torch.float32,
                device=device,
            )
        else:
            if idx.dtype != torch.int64:
                idx = idx.to(torch.int64)
            if read_idx is not None and read_idx.dtype != torch.int64:
                read_idx = read_idx.to(torch.int64)
            # MTP spec-verify expands one logical sequence into consecutive
            # batch rows (bonus + drafted tokens, e.g. q_cu=[0,1,2,3,4] for
            # two k=1 sequences, kv=[n,n+1,m,m+1] per row) while
            # linear_state_indices stays per SEQUENCE (one slot id per
            # sequence, e.g. [s1,s2] for the four rows above). Expand each
            # sequence's slot across its contiguous row group, then merge the
            # same-slot rows back into single sequences: the recurrent state
            # must chain bonus -> draft inside one sequence, and the cache
            # read/write needs one row per slot. Mapping rows to sequences by
            # position instead (rows==seqs) silently drops the tail rows of
            # every sequence past the first and corrupts the view/layout
            # downstream.
            q_cu_raw = metadata.q_cu_seq_lens
            if in_graph and getattr(metadata, "expanded_decode_metadata", None) is not None and q_cu_raw is not None:
                # Graph capture/replay of an expanded spec-verify batch. The
                # static metadata keeps the PER-ROW layout the attention
                # backends consume (q_cu/kv/block_table all per token row);
                # the per-sequence group count rides on q_seq_lens (N
                # entries of value spec_width). Pure shape math — no
                # device->host syncs. Row slots arrive per-row and pairwise
                # equal; take each group's row 0. A bucket tail shorter than
                # spec_width is padding and stays out of the recurrent chain.
                num_rows = int(mixed_qkv.shape[2])
                q_seq_lens = getattr(metadata, "q_seq_lens", None)
                n_groups = int(q_seq_lens.numel()) if q_seq_lens is not None else int(q_cu_raw.numel()) - 1
                spec_width = int(getattr(metadata, "spec_group_width", 0) or 0)
                full_rows = n_groups * spec_width if spec_width > 1 and n_groups > 0 else num_rows
                uniform = num_seqs > 0 and n_groups > 0 and full_rows == num_rows and num_rows % n_groups == 0
                padded_tail = num_seqs > 0 and n_groups > 0 and spec_width > 1 and 0 < full_rows < num_rows
                if uniform or padded_tail:
                    row_count = full_rows if padded_tail else num_rows
                    group_width = spec_width if padded_tail else num_rows // n_groups
                    if padded_tail and idx.numel() == num_rows:
                        group_idx = idx[:row_count].view(n_groups, group_width)[:, 0].contiguous()
                    elif idx.numel() == num_rows:
                        group_idx = idx.view(n_groups, group_width)[:, 0].contiguous()
                    else:
                        group_idx = idx
                    verify_qkv = mixed_qkv[:, :, :row_count] if padded_tail else mixed_qkv
                    verify_gate = _take_leading_token_rows(raw_gate_proj, num_rows, row_count)
                    verify_beta = _take_leading_token_rows(beta_raw, num_rows, row_count)
                    verify_out = self._spec_verify_v3(
                        verify_qkv,
                        verify_gate,
                        verify_beta,
                        layer,
                        group_idx,
                        metadata,
                        conv_cache,
                        ssm_cache,
                        recurrent_kda,
                    )
                    if not padded_tail or verify_out.shape[1] == num_rows:
                        return verify_out
                    tail = verify_out.new_zeros(
                        (verify_out.shape[0], num_rows - verify_out.shape[1], *verify_out.shape[2:])
                    )
                    return torch.cat((verify_out, tail), dim=1)
            q_rows = 0 if in_graph else (int(q_cu_raw.numel()) - 1 if q_cu_raw is not None else 0)
            per_row_idx = None
            per_row_cu = None
            if q_rows == idx.numel() and idx.numel() > 1 and bool((idx[1:] == idx[:-1]).any().item()):
                # Defensive: indices already duplicated per row.
                per_row_idx = idx
                per_row_cu = q_cu_raw
            elif q_rows > idx.numel() and q_rows % idx.numel() == 0:
                # Spec-verify expansion (uniform rows per sequence = k+1).
                per_row_idx = idx.repeat_interleave(q_rows // idx.numel())
                per_row_cu = q_cu_raw
            elif (
                0 < q_rows < idx.numel()
                and idx.numel() % q_rows == 0
                and idx.numel() > 1
                and bool((idx[1:] == idx[:-1]).any().item())
            ):
                # Chunked-typed spec verify: q_cu is sequence-scoped while
                # linear_state_indices is per-row (one row per token).
                # Normalize to the per-row view — every row is one token.
                per_row_idx = idx
                per_row_cu = torch.arange(idx.numel() + 1, dtype=torch.int64, device=idx.device)
            elif q_rows > idx.numel():
                raise RuntimeError(
                    f"unaligned linear-state batch: {q_rows} rows vs {idx.numel()} sequences with non-uniform expansion"
                )
            if per_row_idx is not None:
                row_lengths = (per_row_cu[1:] - per_row_cu[:-1]).tolist()
                merged_lengths: list = []
                merged_slots: list = []
                for slot, length in zip(per_row_idx.tolist(), row_lengths):
                    if merged_slots and slot == merged_slots[-1]:
                        merged_lengths[-1] += length
                    else:
                        merged_slots.append(slot)
                        merged_lengths.append(length)
                if len(set(merged_lengths)) != 1:
                    raise RuntimeError("KDA MTP verify requires uniform rows per sequence")
                idx = torch.tensor(merged_slots, dtype=torch.int64, device=idx.device)
                return self._spec_verify_v3(
                    mixed_qkv,
                    raw_gate_proj,
                    beta_raw,
                    layer,
                    idx,
                    metadata,
                    conv_cache,
                    ssm_cache,
                    recurrent_kda,
                )
            if (
                (not is_prefill or is_spec_verify)
                and getattr(metadata, "num_accepted_tokens", None) is not None
                and idx.numel() > 0
                and mixed_qkv.dim() == 3
                and mixed_qkv.shape[2] >= idx.numel()
                and mixed_qkv.shape[2] % idx.numel() == 0
                and (is_spec_verify or mixed_qkv.shape[2] > idx.numel())
            ):
                # A multi-token row group is the MTP/speculative V3 path even
                # when legacy metadata does not mark it as spec verify. Plain
                # width-one decode must remain on the regular recurrent path;
                # ACL graphs still carry an all-ones accepted-token buffer, but
                # that buffer alone does not make a request speculative.
                return self._spec_verify_v3(
                    mixed_qkv,
                    raw_gate_proj,
                    beta_raw,
                    layer,
                    idx,
                    metadata,
                    conv_cache,
                    ssm_cache,
                    recurrent_kda,
                )
            elif is_spec_verify:
                raise RuntimeError("KDA spec-verify requires a non-empty uniform token block within the decoding width")
            ssm_checkpoint_stride = linear_state_checkpoint_stride(
                conv_cache,
                ssm_cache,
            )
            conv_cache_base = conv_cache[:, :conv_state_len]
            state_read_idx = read_idx if is_prefill else idx
            conv_i = conv_cache_base.index_select(0, state_read_idx)
            ssm_read_idx = state_read_idx * ssm_checkpoint_stride
            ssm_i = ssm_cache.index_select(0, ssm_read_idx)
            his = getattr(metadata, "has_initial_state", None)
            if his is not None and len(his) == num_seqs:
                if not isinstance(his, torch.Tensor):
                    his = torch.tensor(his, dtype=torch.int64, device=conv_i.device)
                warm = his.to(torch.bool).view(num_seqs, 1, 1)
                conv_i = torch.where(warm, conv_i, torch.zeros_like(conv_i))
                ssm_i = torch.where(
                    warm.view(num_seqs, 1, 1, 1),
                    ssm_i,
                    torch.zeros_like(ssm_i),
                )
            conv_state, ssm_state = conv_i, ssm_i.contiguous()
        scale = 1.0 / (head_dim**0.5)
        # Route on metadata, not seq_len: MTP/spec decode can carry multiple
        # tokens per sequence (seq_len > 1) but is still a decode step; the
        # seq_len heuristic would wrongly send it to the chunked prefill path.
        device = mixed_qkv.device
        mixed_qkv = mixed_qkv.transpose(1, 2)
        if num_seqs == batch_size:
            mixed_qkv = self._causal_conv1d(mixed_qkv, conv_state, layer, is_prefill=is_prefill)
        else:
            q_cu = metadata.q_cu_seq_lens
            assert q_cu is not None, "multi-sequence linear attention needs q_cu_seq_lens"
            q_cu_list = _get_host_q_cu_seq_lens(metadata, num_seqs)
            mixed_qkv = self._causal_conv1d(
                mixed_qkv, conv_state, layer, query_start_loc=q_cu_list, is_prefill=is_prefill
            )
            # TND packed layout for recurrent_kda: [T, nh, hd] per channel group.
            seq_len = int(q_cu_list[-1])
            hidden_shape = (1, seq_len, -1, head_dim)

        query, key, value = torch.split(mixed_qkv, [qkv_dim] * 3, dim=-1)
        query = query.view(hidden_shape)
        key = key.view(hidden_shape)
        value = value.view(hidden_shape)

        # ``beta_raw`` arrives as [B, S, nh] and is already correct for both
        # layouts: per-sequence [num_seqs, per_seq_len, nh] when the batch rows
        # map 1:1 to sequences, and flattened [1, T, nh] (T = sum of q_cu) for
        # the varlen multi-sequence path. Re-viewing it as
        # (num_seqs, seq_len, nh) assumes a uniform per-seq length == the
        # flattened total and crashes on multi-sequence decode batches
        # (e.g. 2 concurrent requests: view [2, 2, 4] on 8 elements).
        fuse_gate = raw_gate_proj is not None and fg is not None and gate_lb is not None and -5.0 <= gate_lb < 0.0
        if fuse_gate:
            g = None
            g_raw = raw_gate_proj if num_seqs == batch_size else raw_gate_proj.view(hidden_shape)
            _gate_kwargs = dict(
                A_log=fg.A_log.contiguous(),
                dt_bias=fg.dt_bias.contiguous(),
                use_gate_in_kernel=True,
                safe_gate=True,
                lower_bound=gate_lb,
            )
        else:
            gate = fg.gate_from_raw(raw_gate_proj)
            g = gate if num_seqs == batch_size else gate.view(hidden_shape)
            g = g.to(torch.float32)
            g_raw = None
            _gate_kwargs = dict(use_gate_in_kernel=False)
        if not is_prefill:
            # decode (incl. MTP multi-token-per-seq varlen): recurrent_kda on
            # packed TND [T, nh, hd] with cu_seqlens.
            q_tnd = query.reshape(-1, num_heads_local, head_dim).to(torch.bfloat16).contiguous()
            k_tnd = key.reshape(-1, num_heads_local, head_dim).to(torch.bfloat16).contiguous()
            v_tnd = value.reshape(-1, num_heads_local, head_dim).to(torch.bfloat16).contiguous()
            g_tnd = None if fuse_gate else g.reshape(-1, num_heads_local, head_dim).contiguous()
            g_raw_tnd = g_raw.reshape(-1, num_heads_local, head_dim).contiguous() if fuse_gate else None
            beta_raw_tnd = beta_raw.reshape(-1, num_heads_local).contiguous()
            if num_seqs != batch_size:
                cu_seqlens = q_cu.to(torch.int32)
            elif seq_len == 1:
                # B independent single-token sequences.
                if in_graph:
                    # Constant content; allocate once into the persistent
                    # execution buffer so capture records no per-step H2D
                    # arange.
                    cu_seqlens = get_execution_buffer(
                        ("KDA_DECODE_CU_SEQLENS", num_seqs),
                        lambda: torch.arange(num_seqs + 1, dtype=torch.int32, device=device),
                    )
                else:
                    cu_seqlens = torch.arange(num_seqs + 1, dtype=torch.int32, device=device)
            else:
                cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
            # Plain (non-MTP) decode hot path: fuse the safe-gate into the
            # kernel when the model handed a raw projection (``_gate_kwargs``
            # / ``g_raw_tnd``); else keep the python-materialized gate.
            core_attn_out, final_state = recurrent_kda(
                q_tnd,
                k_tnd,
                v_tnd,
                g_raw_tnd if fuse_gate else g_tnd,
                beta_raw_tnd,
                initial_state=ssm_state,
                cu_seqlens=cu_seqlens,
                layout="TND",
                scale=scale,
                output_final_state=True,
                inplace_final_state=False,
                use_qk_l2norm_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                state_v_first=True,
                **_gate_kwargs,
            )
            # recurrent_kda returns the packed TND [T, nh, hd] layout of its
            # inputs; restore the [B, S, nh, hd] grouping the model layer
            # expects (o_norm gates per head). For T == 1 the flat layout
            # happens to broadcast identically, which masked this for
            # single-stream decode; a multi-sequence decode batch (2
            # concurrent requests flattened to [1, 2, ...]) surfaced it.
            core_attn_out = core_attn_out.to(query.dtype).reshape(hidden_shape)
        else:
            # Normalize Q/K with the NPU l2norm_fwd kernel before calling the
            # FLA AscendC operator.
            q_in = l2_norm(query)
            k_in = l2_norm(key)
            v_in = value.to(torch.bfloat16).contiguous()
            beta = beta_raw.float().sigmoid()
            cu_seqlens = (
                q_cu.to(torch.int32)
                if num_seqs != batch_size
                else torch.tensor([0, seq_len], dtype=torch.int32, device=device)
            )
            # The AscendC kernel supports both the single-sequence and rank-4
            # B=1 varlen layouts. Keep all sequences in one call so launch
            # count is independent of the number of concurrent prefills;
            # cu_seqlens selects the independent sequence boundaries.
            result = chunk_kda_fwd(
                q_in,
                k_in,
                v_in,
                g_raw if fuse_gate else g,
                beta,
                scale,
                chunk_size=64,
                layout="BSND",
                initial_state=ssm_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                return_intermediate_states=False,
                state_v_first=True,
                **_gate_kwargs,
            )
            core_attn_out = result[0].to(query.dtype)
            final_state = result[1]

        if idx is not None:
            conv_cache_base.index_copy_(0, idx, conv_state)
            ssm_write_idx = idx * ssm_checkpoint_stride
            ssm_cache.index_copy_(0, ssm_write_idx, final_state.float().contiguous())
        # multi-seq path reshaped mixed_qkv to [num_seqs, ...]; flatten the
        # output back to [1, T, ...] so the KDA forward's hidden_shape [1, T]
        # aligns for o_norm / o_proj.
        if num_seqs != batch_size:
            core_attn_out = core_attn_out.reshape(1, -1, *core_attn_out.shape[2:])
        elif flatten_graph_decode:
            # The flatten-decode graph branch ran the simple path on
            # [num_seqs, 1, nh, hd]; restore the model's flattened [1, T, ...]
            # layout so o_norm's gate ([1, T, nh, hd]) aligns without
            # broadcasting.
            core_attn_out = core_attn_out.transpose(0, 1)
        return core_attn_out
