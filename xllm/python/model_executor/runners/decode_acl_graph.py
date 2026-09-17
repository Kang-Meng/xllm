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

"""NPU (Ascend) ACL graph runner for the Python model executor.

Captures and replays decode-step graphs using ``torch.npu.NPUGraph``.
Mirrors the structure of ``decode_cuda_graph.py`` but adds NPU-specific
logic:

* ``torch.npu.graph_task_group_begin/end`` around FIA ``.out`` calls during
  capture.
* ``torch.npu.graph_task_update_begin/end`` to refresh FIA host params before
  replay.
* Static ``block_table`` and ``slot_mapping`` tensors so the graph records
  fixed addresses whose *contents* are updated via ``_fill_entry`` each step.
* C++ ACLNN ops (RMSNorm, SiLU, reshape_paged_cache) are used in both eager
  and capture modes — no PyTorch fallbacks needed.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.attention.backend import AttentionBackend, AttentionMetadata
from xllm.python.attention.dsa_metadata import DSA_CACHE_TOKEN
from xllm.python.attention.expanded_decode_metadata import (
    ExpandedDecodeMetadata,
    resolve_expanded_decode_metadata,
)
from xllm.python.model_executor.execution_context import (
    allocate_graph_execution_contexts,
    update_graph_execution_contexts,
)
from xllm.python.model_executor.forward_context import (
    AclGraphCaptureContext,
    AclGraphExecutionState,
    AclGraphTask,
    EplbRuntimeState,
    ForwardContext,
    LayerSynchronizer,
    forward_context,
)
from xllm.python.model_executor.runners.base import BaseRunner
from xllm.python.model_executor.runners.decode_cuda_graph import (
    _CAPTURE_WARMUP_STEPS,
    _decode_bucket,
)


def _require_positive_execution_counts(execution_counts: Sequence[int]) -> None:
    if any(count <= 0 for count in execution_counts):
        raise RuntimeError(f"DP execution token counts must be positive, got {execution_counts}")


def _padded_rank_slices(
    token_counts: Sequence[int],
    dp_size: int,
    per_rank_stride: int,
) -> list[tuple[int, int]]:
    """Validate per-rank counts and return offsets into a padded DP buffer."""
    counts = tuple(int(count) for count in token_counts)
    if len(counts) != dp_size:
        raise RuntimeError(f"DP decode step requires {dp_size} token counts, got {len(counts)}")
    if any(count < 0 or count > per_rank_stride for count in counts):
        raise RuntimeError(
            f"DP token counts must fit the ACL graph batch bucket: counts={counts}, bucket={per_rank_stride}"
        )
    return [(rank * per_rank_stride, count) for rank, count in enumerate(counts)]


@dataclass(slots=True)
class _StaticAttentionMetadata:
    slot_mapping: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    qo_indptr: torch.Tensor | None = None
    q_cu_seq_lens: torch.Tensor | None = None
    kv_cu_seq_lens: torch.Tensor | None = None
    kv_seq_lens_host: torch.Tensor | None = None
    kv_seq_lens_host_values: list[int] | None = None
    new_cache_slots_host_values: list[int] | None = None
    paged_kv_indptr_host: torch.Tensor | None = None
    paged_kv_last_page_len_host: torch.Tensor | None = None
    block_table: torch.Tensor | None = None
    kv_seq_lens: torch.Tensor | None = None
    linear_state_indices: torch.Tensor | None = None
    has_initial_state: torch.Tensor | None = None
    dp_execution_token_counts: tuple[int, ...] = ()
    dp_is_decode: tuple[int, ...] = ()
    q_seq_lens: torch.Tensor | None = None
    # Host-side copy of the (per-entry constant) q_cu for prepare()'s
    # sequence-lens read: a .cpu() on the device buffer would block the host
    # until the whole prior step's device queue drains, serializing the
    # scheduler behind the replay.
    q_cu_host_values: list[int] | None = None
    expanded_decode_metadata: ExpandedDecodeMetadata | None = None
    is_prefill: bool = False
    is_chunked_prefill: bool = False
    mega_moe_token_mask: torch.Tensor | None = None
    is_mixed: bool = False
    is_spec_verify: bool = False
    is_dummy: bool = False
    local_slot_mapping: torch.Tensor | None = None
    kv_split_size: int = 1
    kv_split_rank: int = 0
    has_kv_shard: bool = False
    multi_block_tables: Sequence[torch.Tensor | None] = ()
    dsa_metadata: object | None = None
    dsa_positions: torch.Tensor | None = None
    dsa_cos_sin: torch.Tensor | None = None
    dsa_c4_cos_sin: torch.Tensor | None = None
    dsa_c128_cos_sin: torch.Tensor | None = None
    dsa_graph_mode: bool = False
    dsa_graph_block_table_cols: int = 0


class _DecodeGraphEntry:
    __slots__ = (
        "batch_size",
        "graph",
        "static_output",
        "static_input_ids",
        "static_positions",
        "static_input_embedding",
        "static_metadata",
        "kv_seq_lens_delta",
        "graph_tasks",
        "execution_state",
        "eplb",
        "execution_contexts",
    )


_GraphKey = tuple[
    int,
    bool,
    torch.dtype | None,
    torch.device | None,
    tuple[int, ...] | None,
]


class DecodeAclGraphRunner(BaseRunner):
    """Decode graph runner for NPU (Ascend) using ACL graph capture/replay."""

    def __init__(
        self,
        model: nn.Module,
        attention_backend: AttentionBackend,
        device: torch.device,
        max_batch: int,
        max_model_len: int,
        dp_size: int = 1,
        dp_rank: int = 0,
        decode_batch_size_limit: int | None = None,
        num_decoding_tokens: int = 1,
        enable_mega_moe_token_mask: bool = False,
    ) -> None:
        super().__init__(model, attention_backend, device)
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.max_batch = (max_batch + dp_size - 1) // dp_size
        # Token rows per logical sequence: 1 for plain decode, num_spec+1 for
        # MTP spec-verify. Static paging buffers are sized in token rows so the
        # expanded-verify graph (N*width rows) gets matching capacity instead
        # of the sequence-only max_batch (item 二).
        self.num_decoding_tokens = max(1, int(num_decoding_tokens))
        # Optional memory guardrail: buckets above the limit fall back to
        # eager instead of capturing ever-larger graphs (0 disables).
        self.decode_batch_size_limit = int(decode_batch_size_limit or 0)
        self.max_model_len = max_model_len
        self._enable_mega_moe_token_mask = enable_mega_moe_token_mask
        self._graphs: dict[_GraphKey, _DecodeGraphEntry] = {}
        self._paged_kv_indices_buffer: torch.Tensor | None = None
        self._max_blocks_per_sequence: int = 0
        self._stream: torch.npu.Stream | None = None
        self._update_stream: torch.npu.Stream | None = None
        self._replay_done_event: torch.npu.Event | None = None

    def can_execute(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
    ) -> bool:
        if input_ids.dim() != 1:
            return False
        if self.dp_size > 1 and self.num_decoding_tokens > 1:
            # Target validation metadata differs between active and empty DP
            # ranks, so local graph-admission checks cannot guarantee that the
            # whole group selects the same runner. Keep the target executor on
            # eager until the scheduler publishes one group-wide admission
            # decision. Draft executors use num_decoding_tokens=1 and remain
            # graphable.
            return False
        batch_size = input_ids.numel()
        is_expanded_spec_verify = resolve_expanded_decode_metadata(
            metadata
        ) is not None or self._is_untyped_spec_verify(metadata)
        # Debug isolation switch: force spec-verify batches through the eager
        # runner while keeping the chunked-typed (expanded) layout, to A/B the
        # typed-eager semantics against the graph capture/replay path.
        if is_expanded_spec_verify and os.environ.get("XLLM_NO_VERIFY_GRAPH") == "1":
            return False
        # MTP spec-verify packs (num_speculative_tokens+1) token-rows per
        # sequence, so batch_size (input_ids.numel()) is in TOKENS while
        # self.max_batch (derived from max_seqs_per_batch) is in SEQS. Comparing
        # tokens to seqs wrongly rejects an 8-seq × 4-row verify (32 tokens,
        # 8 seqs ≤ 16) as bucket-overflow, forcing eager (~10x slower) under
        # concurrency. For expanded verify, gate on the actual SEQUENCE count
        # (linear_state_indices is per-seq) — the graph still captures/replays
        # at the full token bucket (keyed by padded_batch_size below), this
        # only fixes the graph-vs-eager admission decision.
        if is_expanded_spec_verify:
            lsi = getattr(metadata, "linear_state_indices", None)
            has_kda_layers = lsi is not None and lsi.numel() > 0
            seq_count = lsi.numel() if has_kda_layers else batch_size
            size_check_bs = seq_count
            # The captured bucket is a power of two (1/2/4/8/16k) but the
            # static expanded-verify metadata (per-row q_cu + per-group
            # q_seq_lens) is only laid out when the PADDED bucket divides by
            # the spec width. Widths that are not powers of two (w=3 for
            # num_spec=2, w=5, ...) would capture an entry without the group
            # layout and silently degrade the in-graph verify semantics —
            # refuse graph admission there and fall back to eager instead.
            if batch_size % max(seq_count, 1) == 0 and seq_count > 0:
                width = batch_size // seq_count
                if _decode_bucket(batch_size) % width != 0:
                    return False
        else:
            size_check_bs = batch_size
        if self.dp_size > 1:
            # Prefill-typed batches exit before the DP contract check below
            # (a prefill without DP counts falls back to eager, never raises).
            if (metadata.is_prefill or metadata.is_chunked_prefill) and not is_expanded_spec_verify:
                return False
            # DP ranks share one graph shape, so a missing or malformed
            # Execution counts cannot silently fall back to eager: divergent
            # execution paths across ranks would deadlock HCCL collectives.
            execution_counts = getattr(
                metadata,
                "dp_execution_token_counts",
                None,
            )
            if execution_counts is None or len(execution_counts) != self.dp_size:
                raise RuntimeError(
                    "DP decode step requires valid dp_execution_token_counts "
                    f"(got {execution_counts!r}, "
                    f"expected length {self.dp_size}). All DP ranks must use the same graph shape."
                )
            _require_positive_execution_counts(execution_counts)
            dp_is_decode = getattr(metadata, "dp_is_decode", None)
            if dp_is_decode is not None and not all(dp_is_decode):
                return False
            # MegaMoe shares one fixed graph shape across DP ranks, so the
            # local rows must equal this rank's advertised execution count;
            # a mismatch would dispatch the wrong shape and desync the EP
            # collective rather than silently fall back to eager.
            if execution_counts[self.dp_rank] != input_ids.shape[0]:
                raise RuntimeError(
                    "DP execution token count does not match the local input: "
                    f"rank={self.dp_rank}, rows={input_ids.shape[0]}, "
                    f"counts={execution_counts}"
                )
            # dp_execution_token_counts are TOKEN rows while max_batch is in SEQS (see
            # the seq-vs-token note above); fold expanded verify rows down to
            # sequences so both units agree before the bucket comparison.
            width = 1
            if is_expanded_spec_verify and size_check_bs > 0 and batch_size % size_check_bs == 0:
                width = batch_size // size_check_bs
            max_global_tokens = max(int(c) for c in execution_counts)
            global_batch = max(
                (max_global_tokens + width - 1) // width,
                size_check_bs,
            )
            ok = (
                ((not metadata.is_prefill and not metadata.is_chunked_prefill) or is_expanded_spec_verify)
                and self._has_compatible_decode_metadata(input_ids, metadata)
                and (input_embedding is None or input_embedding.shape[0] == batch_size)
                and _decode_bucket(global_batch) <= self.max_batch
                and (self.decode_batch_size_limit <= 0 or _decode_bucket(global_batch) <= self.decode_batch_size_limit)
            )
            return ok
        bucket_size = _decode_bucket(size_check_bs)
        ok = (
            ((not metadata.is_prefill and not metadata.is_chunked_prefill) or is_expanded_spec_verify)
            and self._has_compatible_decode_metadata(input_ids, metadata)
            and (input_embedding is None or input_embedding.shape[0] == batch_size)
            and bucket_size <= self.max_batch
            and (self.decode_batch_size_limit <= 0 or bucket_size <= self.decode_batch_size_limit)
        )
        return ok

    def _decode_metadata(
        self, metadata: AttentionMetadata
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[int] | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return per-row KV and paging metadata for decode graph replay."""
        expanded = self._expanded_verify_view(metadata)
        block_table = expanded.block_table if expanded is not None else self._effective_block_table(metadata)
        kv_seq_lens = expanded.kv_seq_lens if expanded is not None else metadata.kv_seq_lens
        kv_seq_lens_host_values = (
            expanded.kv_seq_lens_host_values
            if expanded is not None
            else getattr(metadata, "kv_seq_lens_host_values", None)
        )
        if block_table is None or kv_seq_lens is None:
            raise RuntimeError("decode graph requires block and KV metadata")
        kv_seq_lens = kv_seq_lens.to(torch.int32)
        block_table = block_table.to(
            device=kv_seq_lens.device,
            dtype=torch.int32,
        ).contiguous()
        is_mla = getattr(self.attention_backend, "is_mla", False)
        requires_host_kv_lengths = not is_mla or getattr(
            self.attention_backend,
            "requires_host_kv_lengths",
            False,
        )
        if not requires_host_kv_lengths:
            # The C++ engine sizes kv_seq_lens_host_values by the global DP
            # batch, while block_table holds only this rank's local rows. When
            # the backend does not consume host KV lengths (e.g. sparse MLA),
            # drop them so the mismatched global length never reaches shape
            # validation (which would otherwise reject the bucket and silently
            # fall a DP>1 MLA decode back to eager).
            kv_seq_lens_host_values = None
        elif kv_seq_lens_host_values is None:
            raise RuntimeError("decode graph requires scheduler-provided host KV lengths")

        paged_kv_indptr = expanded.paged_kv_indptr if expanded is not None else metadata.paged_kv_indptr
        paged_kv_indices = expanded.paged_kv_indices if expanded is not None else metadata.paged_kv_indices
        paged_kv_last_page_len = (
            expanded.paged_kv_last_page_len if expanded is not None else metadata.paged_kv_last_page_len
        )
        # Rebuild the row-scoped paged metadata when it is missing OR when the
        # C++ builder left it per-sequence while block_table/kv_seq_lens are
        # per-token-row. Under MTP concurrency the batch is frequently MIXED
        # (some sequences in verify with width>1, some in plain decode with
        # width=1), so rows is not a uniform multiple of seqs and the
        # _is_untyped_spec_verify uniform-width gate does not fire; the C++
        # metadata builder expands bt/kv to token rows but leaves
        # paged_kv_last_page_len/indptr per-sequence, yielding a numel mismatch
        # that crashes _validate_decode_metadata_shapes. Rebuild from
        # kv_seq_lens (per-token-row) so paging matches bt in every case.
        _paged_missing = paged_kv_indptr is None or paged_kv_indices is None or paged_kv_last_page_len is None
        _paged_mismatch = (not _paged_missing) and (
            paged_kv_last_page_len.numel() != block_table.shape[0]
            or paged_kv_indptr.numel() != block_table.shape[0] + 1
        )
        if _paged_missing and expanded is None:
            raise RuntimeError("decode graph requires paged KV metadata")
        if _paged_missing or _paged_mismatch:
            (
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
            ) = self._build_row_aligned_paged_kv_metadata(
                block_table,
                kv_seq_lens,
            )
        self._validate_decode_metadata_shapes(
            block_table,
            kv_seq_lens,
            kv_seq_lens_host_values,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        )
        return (
            block_table,
            kv_seq_lens,
            kv_seq_lens_host_values,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        )

    @staticmethod
    def _is_untyped_spec_verify(metadata: AttentionMetadata) -> bool:
        """Shape-only detector for a GENERIC-flow MTP verify batch.

        The untyped (GENERIC) spec-verify batch carries one row per token —
        kv_seq_lens / q_cu_seq_lens / slot_mapping / block_table are all
        per-row — while linear_state_indices stays per logical sequence with
        a fixed verify width > 1 rows per sequence. Pure shape math — no
        device->host sync.
        """
        kv = getattr(metadata, "kv_seq_lens", None)
        bt = getattr(metadata, "block_table", None)
        q_cu = getattr(metadata, "q_cu_seq_lens", None)
        lsi = getattr(metadata, "linear_state_indices", None)
        if kv is None or bt is None or bt.dim() != 2 or kv.dim() != 1:
            return False
        if lsi is None or lsi.dim() != 1:
            return False
        rows = kv.shape[0]
        seqs = lsi.shape[0]
        if bt.shape[0] != rows or seqs <= 0 or rows <= seqs or rows % seqs != 0:
            return False
        # q_cu_seq_lens is per-token-row: graph mode packs it WITHOUT a
        # leading zero (numel == rows), eager/typed paths WITH one
        # (numel == rows + 1). Both are valid per-row cumulative layouts;
        # rejecting numel == rows wrongly classifies a GENERIC-flow MTP
        # verify batch as non-untyped, skipping _build_row_aligned_paged_kv
        # and crashing on the C++ builder's per-sequence paged_kv_last_page_len.
        if q_cu is not None and q_cu.numel() not in (rows, rows + 1):
            return False
        return not (getattr(metadata, "is_prefill", False) or getattr(metadata, "is_chunked_prefill", False))

    def _expanded_verify_view(self, metadata: AttentionMetadata) -> ExpandedDecodeMetadata | None:
        """Resolve the expanded (token-row) view of a spec-verify batch.

        Chunked-typed flows (Qwen3.5) carry the expanded metadata from C++;
        a GENERIC-flow MTP verify batch (e.g. GLM5-next KDA) instead arrives
        decode-typed with per-row kv lens and per-sequence block tables.
        Synthesize the token-row expanded view for the latter — block table
        rows duplicated per verify row, row-scoped paging built on-device —
        so the same graph machinery captures both.
        """
        expanded = resolve_expanded_decode_metadata(metadata, block_size=self.attention_backend.page_size)
        if expanded is not None:
            return expanded
        if not self._is_untyped_spec_verify(metadata):
            return None
        kv_rows = metadata.kv_seq_lens.to(torch.int32)
        block_table_rows = metadata.block_table.to(torch.int32).contiguous()
        host_values = getattr(metadata, "kv_seq_lens_host_values", None)
        (
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        ) = self._build_row_aligned_paged_kv_metadata(
            block_table_rows,
            kv_rows,
        )
        synthesized = ExpandedDecodeMetadata(
            kv_seq_lens=kv_rows,
            block_table=block_table_rows,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            paged_attention_tiling_data=None,
            kv_seq_lens_host=None,
            kv_seq_lens_host_values=host_values,
        )
        return synthesized

    def _build_row_aligned_paged_kv_metadata(
        self,
        block_table: torch.Tensor,
        kv_seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build token-row paging metadata like the C++ graph input builder."""
        page_size = int(self.attention_backend.page_size)
        if page_size <= 0:
            raise RuntimeError("decode graph page size must be positive")

        effective_kv_seq_lens = torch.clamp(kv_seq_lens, min=1)
        page_counts = torch.div(
            effective_kv_seq_lens + page_size - 1,
            page_size,
            rounding_mode="floor",
        ).to(torch.int32)
        paged_kv_indptr = torch.cat(
            (
                torch.zeros(
                    1,
                    dtype=torch.int32,
                    device=block_table.device,
                ),
                torch.cumsum(page_counts, dim=0, dtype=torch.int32),
            )
        )
        page_offsets = torch.arange(
            block_table.shape[1],
            dtype=torch.int32,
            device=block_table.device,
        )
        valid_pages = page_offsets.unsqueeze(0) < page_counts.unsqueeze(1)
        # Callers (_decode_metadata / _expanded_verify_view) already cast
        # block_table to int32; avoid a redundant full-tensor device copy on
        # every paging build.
        paged_kv_indices = block_table.masked_select(valid_pages).contiguous()
        paged_kv_last_page_len = ((effective_kv_seq_lens - 1) % page_size + 1).to(torch.int32)
        return (
            paged_kv_indptr.contiguous(),
            paged_kv_indices,
            paged_kv_last_page_len.contiguous(),
        )

    @staticmethod
    def _validate_decode_metadata_shapes(
        block_table: torch.Tensor,
        kv_seq_lens: torch.Tensor,
        kv_seq_lens_host_values: list[int] | None,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
    ) -> None:
        if block_table.dim() != 2:
            raise RuntimeError("decode block_table must be two-dimensional")
        sequence_count = block_table.shape[0]
        per_sequence_tensors = (
            ("kv_seq_lens", kv_seq_lens),
            ("paged_kv_last_page_len", paged_kv_last_page_len),
        )
        for name, tensor in per_sequence_tensors:
            if tensor.dim() != 1 or tensor.numel() != sequence_count:
                raise RuntimeError(f"decode {name} must contain one value per sequence")
        if kv_seq_lens_host_values is not None and len(kv_seq_lens_host_values) != sequence_count:
            raise RuntimeError("decode kv_seq_lens_host_values must contain one value per sequence")
        if paged_kv_indptr.dim() != 1 or paged_kv_indptr.numel() != sequence_count + 1:
            raise RuntimeError("decode paged_kv_indptr must contain one offset per sequence plus the terminal offset")
        if paged_kv_indices.dim() != 1 or paged_kv_indices.numel() == 0:
            raise RuntimeError("decode paged_kv_indices must be a non-empty flat page list")

    @staticmethod
    def _validate_decode_token_layout(
        input_ids: torch.Tensor,
        positions: torch.Tensor | None,
        slot_mapping: torch.Tensor,
        sequence_count: int,
    ) -> None:
        if input_ids.dim() != 1 or input_ids.numel() != sequence_count:
            raise RuntimeError("ACL graph decode input_ids must contain one token per sequence")
        if slot_mapping.dim() != 1 or slot_mapping.numel() != sequence_count:
            raise RuntimeError("ACL graph decode slot_mapping must contain one slot per token")
        if positions is not None and (positions.dim() != 1 or positions.numel() != sequence_count):
            raise RuntimeError("ACL graph decode positions must contain one value per token")

    def _has_compatible_decode_metadata(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> bool:
        """Check the one-token-per-sequence contract of ACL decode graphs.

        Pure shape/contract admission check — no on-device paging build and no
        exception-based capability detection (§7: expected incompatible shapes
        return False explicitly; genuinely unexpected/malformed states are left
        to raise from _decode_metadata/_validate during fill, propagating rather
        than being swallowed into a silent eager fallback).
        """
        if not self._is_shape_compatible(input_ids, metadata):
            return False
        batch_size = input_ids.numel()
        is_expanded = resolve_expanded_decode_metadata(metadata) is not None or self._is_untyped_spec_verify(metadata)
        if not is_expanded and metadata.kv_cu_seq_lens is not None:
            if metadata.kv_cu_seq_lens.numel() not in (
                batch_size,
                batch_size + 1,
            ):
                return False
        if metadata.q_cu_seq_lens is not None and not is_expanded:
            if metadata.q_cu_seq_lens.numel() not in (
                batch_size,
                batch_size + 1,
            ):
                return False
        has_initial_state = getattr(metadata, "has_initial_state", None)
        if has_initial_state is not None:
            state_count = has_initial_state.numel()
            if state_count != batch_size and not (is_expanded and state_count > 0 and batch_size % state_count == 0):
                return False
        linear_idx = getattr(metadata, "linear_state_indices", None)
        if (
            linear_idx is not None
            and linear_idx.numel() != batch_size
            and not (is_expanded and linear_idx.numel() > 0 and batch_size % linear_idx.numel() == 0)
        ):
            return False
        # Linear-attention (KDA) layers read per-sequence conv/ssm state via
        # linear_state_indices; without it the captured graph would index
        # state slots with a None buffer. A spec-verify batch may carry one
        # slot id per logical sequence (GENERIC flow) — the fill expands it
        # per token row.
        needs_linear_state = any(getattr(cache, "conv", None) is not None for cache in self.layer_caches)
        if needs_linear_state:
            if linear_idx is None:
                return False
        # The kPool indexer's graph gather densifies each sequence to a
        # static max_kv (graph_index_history_max_kv). A sequence whose block
        # table outgrows that cap must take the eager runner's dynamic
        # gather instead of capturing/replaying a too-small graph.
        if any(getattr(cache, "index", None) is not None for cache in self.layer_caches):
            max_kv_cap = getattr(self.attention_backend, "graph_index_history_max_kv", None)
            effective_bt = self._effective_block_table(metadata)
            if (
                max_kv_cap is not None
                and effective_bt is not None
                and effective_bt.shape[1] * self.attention_backend.page_size > max_kv_cap
            ):
                return False
        return True

    @staticmethod
    def _effective_block_table(metadata: AttentionMetadata) -> torch.Tensor | None:
        """Return the primary per-row table used by DSV4 decode."""
        expanded = resolve_expanded_decode_metadata(metadata)
        if expanded is not None:
            return expanded.block_table
        if metadata.block_table is not None:
            return metadata.block_table
        multi_block_tables = tuple(getattr(metadata, "multi_block_tables", ()) or ())
        return multi_block_tables[0] if multi_block_tables else None

    @staticmethod
    def _has_effective_slot_mapping(
        metadata: AttentionMetadata,
        token_count: int,
    ) -> bool:
        slot_mapping = getattr(metadata, "slot_mapping", None)
        if slot_mapping is not None and slot_mapping.dim() == 1 and slot_mapping.numel() == token_count:
            return True
        host_slots = getattr(metadata, "new_cache_slots_host_values", None)
        return host_slots is not None and len(host_slots) == token_count

    @staticmethod
    def _effective_slot_mapping(
        metadata: AttentionMetadata,
        token_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Resolve scheduler cache slots when DSV4 omits the top-level tensor."""
        slot_mapping = getattr(metadata, "slot_mapping", None)
        if slot_mapping is not None and slot_mapping.dim() == 1 and slot_mapping.numel() == token_count:
            return slot_mapping.to(device=device, dtype=torch.int32).contiguous()
        host_slots = getattr(metadata, "new_cache_slots_host_values", None)
        if host_slots is None or len(host_slots) != token_count:
            raise RuntimeError("ACL graph decode requires one scheduler cache slot per token")
        return torch.tensor(host_slots, dtype=torch.int32, device=device)

    def _is_shape_compatible(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> bool:
        """Pure, side-effect-free shape/contract pre-check.

        Mirrors the precondition branches of _decode_metadata and
        _validate_decode_metadata_shapes/_validate_decode_token_layout so that
        the EXPECTED incompatible shapes return False here (admit→eager) while
        any state this does not cover raises later (let-it-crash, §7) instead
        of being silently swallowed. No on-device paging construction.
        """
        block_table = self._effective_block_table(metadata)
        kv_seq_lens = metadata.kv_seq_lens
        expanded = resolve_expanded_decode_metadata(metadata)
        if expanded is not None:
            block_table = expanded.block_table
            kv_seq_lens = expanded.kv_seq_lens
        # For an untyped (GENERIC) spec-verify batch the per-row tensors are
        # the top-level kv_seq_lens / block_table directly (no device build).
        if block_table is None or kv_seq_lens is None:
            return False
        if block_table.dim() != 2 or kv_seq_lens.dim() != 1:
            return False
        sequence_count = block_table.shape[0]
        if kv_seq_lens.numel() != sequence_count:
            return False
        # Host KV lengths are required for non-sparse-MLA backends; a missing
        # or mis-sized host list is an expected "not compatible" (→ eager).
        is_mla = getattr(self.attention_backend, "is_mla", False)
        requires_host = not is_mla or getattr(
            self.attention_backend,
            "requires_host_kv_lengths",
            False,
        )
        if requires_host:
            host_values = (
                expanded.kv_seq_lens_host_values
                if expanded is not None
                else getattr(metadata, "kv_seq_lens_host_values", None)
            )
            if host_values is None or len(host_values) != sequence_count:
                return False
        # Paged KV metadata is only mandatory when the on-device builder will
        # NOT synthesize it for an expanded/untyped spec-verify batch.
        if expanded is None and not self._is_untyped_spec_verify(metadata):
            if (
                metadata.paged_kv_indptr is None
                or metadata.paged_kv_indices is None
                or metadata.paged_kv_last_page_len is None
            ):
                return False
        # One-token-per-sequence (per-row) contract.
        if input_ids.dim() != 1 or input_ids.numel() != sequence_count:
            return False
        if not self._has_effective_slot_mapping(metadata, sequence_count):
            return False
        return True

    @staticmethod
    def _cumulative_lengths(
        sequence_lengths: torch.Tensor,
        cumulative_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """Normalize NPU sequence ends to a cumulative tensor with a zero."""
        batch_size = sequence_lengths.numel()
        if cumulative_lengths is None:
            return torch.cat(
                (
                    torch.zeros(
                        1,
                        dtype=torch.int32,
                        device=sequence_lengths.device,
                    ),
                    torch.cumsum(sequence_lengths, dim=0),
                )
            )
        cumulative_lengths = cumulative_lengths.to(torch.int32)
        if cumulative_lengths.numel() == batch_size + 1:
            return cumulative_lengths
        if cumulative_lengths.numel() == batch_size:
            return torch.cat(
                (
                    torch.zeros(
                        1,
                        dtype=torch.int32,
                        device=cumulative_lengths.device,
                    ),
                    cumulative_lengths,
                )
            )
        raise RuntimeError(
            "cumulative sequence lengths must contain either one value per "
            "sequence or a leading zero plus one value per sequence"
        )

    def warmup(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
    ) -> None:
        batch_size = input_ids.shape[0]
        # Same seq-vs-token admission as execute: MTP verify packs
        # (num_spec+1) token-rows per seq, so the capacity gate (max_batch,
        # in seqs) must compare SEQUENCE count, not token count.
        is_expanded = resolve_expanded_decode_metadata(metadata) is not None or self._is_untyped_spec_verify(metadata)
        if is_expanded:
            lsi = getattr(metadata, "linear_state_indices", None)
            cap_bs = lsi.numel() if lsi is not None and lsi.numel() > 0 else batch_size
        else:
            cap_bs = batch_size
        if _decode_bucket(cap_bs) > self.max_batch:
            raise ValueError("decode batch exceeds ACL graph capacity")

        # Graph capture is performed lazily by ``execute`` on the first decode
        # of each (bucket, expanded, embedding) key -- including the synthetic
        # batches driven by the C++ graph-warmup phase -- so this entry point
        # only validates capacity.  Pre-capturing here would duplicate
        # ``execute``'s first-capture path; deferring keeps a single capture
        # code path and matches the lazy-capture V3 wiring.

    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        layer_synchronizer: LayerSynchronizer | None = None,
        eplb: EplbRuntimeState | None = None,
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        is_expanded = resolve_expanded_decode_metadata(metadata) is not None or self._is_untyped_spec_verify(metadata)
        # Same seq-vs-token admission as can_execute: MTP verify packs
        # (num_spec+1) token-rows per seq, so the capacity gate (max_batch, in
        # seqs) must compare SEQUENCE count, not token count. The graph is still
        # keyed/captured at the full token bucket (padded_batch_size).
        if is_expanded:
            lsi = getattr(metadata, "linear_state_indices", None)
            cap_bs = lsi.numel() if lsi is not None and lsi.numel() > 0 else batch_size
        else:
            cap_bs = batch_size
        if self.dp_size > 1:
            # DP ranks share one captured graph shape: the bucket must be the
            # GLOBAL max token count, not this rank's local batch_size, so
            # every rank captures the same shape (the baked
            # dp_execution_token_counts agree across ranks and the fixed-shape
            # all_gather matches). The bucket is in TOKEN rows — same unit as
            # the non-DP _decode_bucket(batch_size) — because the static
            # buffers / q_cu / q_seq_lens are all per token row; folding tokens
            # back to sequences here would under-size the buffers and crash
            # _fill_entry's per-token-row copy. The sequence capacity gate
            # below (_decode_bucket(cap_bs) <= max_batch) is a separate
            # seq-unit check. Assert this rank's count equals its batch_size.
            execution_counts = getattr(metadata, "dp_execution_token_counts", None)
            if execution_counts is None or len(execution_counts) != self.dp_size:
                raise RuntimeError(
                    "DP decode step requires valid dp_execution_token_counts "
                    f"(got {execution_counts!r}, expected length {self.dp_size})."
                )
            this_rank_count = int(execution_counts[self.dp_rank])
            if this_rank_count != batch_size:
                raise RuntimeError(
                    f"dp_execution_token_counts[{self.dp_rank}]={this_rank_count} "
                    f"!= local batch_size {batch_size}; the DP graph shape is fixed "
                    "by the shared counts so they must agree per rank."
                )
            max_global_tokens = max(int(c) for c in execution_counts)
            padded_batch_size = _decode_bucket(max_global_tokens)
        else:
            padded_batch_size = _decode_bucket(batch_size)
        if _decode_bucket(cap_bs) > self.max_batch:
            raise ValueError("decode batch exceeds ACL graph capacity")

        graph_key = self._graph_key(padded_batch_size, is_expanded, input_embedding)
        entry = self._graphs.get(graph_key)
        first_capture = entry is None
        if first_capture:
            entry = self._allocate_entry(padded_batch_size, input_ids, positions, metadata)
            entry.eplb = self._allocate_graph_eplb_state(eplb, padded_batch_size)
            self._graphs[graph_key] = entry
        entry_eplb = getattr(entry, "eplb", None)
        if not first_capture and (entry_eplb is None) != (eplb is None):
            raise RuntimeError("EPLB state changed after ACL graph capture")
        if not first_capture and eplb is not None and entry_eplb is not None:
            if entry_eplb.expert_load_data.data_ptr() != eplb.expert_load_data.data_ptr():
                raise RuntimeError("EPLB expert-load tensor changed after ACL graph capture")
            if (entry_eplb.decode_token_mask is None) != (eplb.decode_token_mask is None):
                raise RuntimeError("EPLB decode-mask availability changed after ACL graph capture")
            entry_eplb.is_graph_warmup = eplb.is_graph_warmup
        self._fill_graph_eplb_decode_mask(entry, eplb, metadata)

        if self._stream is None:
            self._stream = torch.npu.Stream(device=input_ids.device)
            self._update_stream = torch.npu.Stream(device=input_ids.device, priority=-1)
            self._replay_done_event = torch.npu.Event()

        self._fill_entry(entry, input_ids, positions, metadata, batch_size, input_embedding)

        prepare_context = ForwardContext(
            self.attention_backend,
            self.device,
            entry.static_metadata,
            self.layer_caches,
            execution_state=entry.execution_state,
            eplb=entry_eplb,
            execution_contexts=entry.execution_contexts,
        )
        with forward_context(prepare_context):
            self.attention_backend.prepare(entry.static_metadata, graph_mode=True)
            if not first_capture:
                refresh_dsa = getattr(
                    self.attention_backend,
                    "refresh_dsa_metadata_for_graph_replay",
                    None,
                )
                if refresh_dsa is not None:
                    refresh_dsa(entry.static_metadata)

        if first_capture:
            self._capture(entry)

        # Besides ordering input updates, this wait protects the output view
        # returned by the previous replay.  The graph cannot overwrite its
        # static output until consumers queued on the current stream finish.
        self._stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self._stream):
            entry.graph.replay()
            output = entry.static_output[:batch_size]

        with torch.npu.stream(self._update_stream):
            self._update_stream.wait_event(self._replay_done_event)
            self._update_graph_tasks(self._update_stream, entry.graph_tasks)

        self._replay_done_event.record(self._stream)

        torch.npu.current_stream().wait_stream(self._stream)
        return output

    def _allocate_graph_eplb_state(
        self,
        eplb: EplbRuntimeState | None,
        padded_batch_size: int,
    ) -> EplbRuntimeState | None:
        if eplb is None:
            return None
        decode_token_mask = None
        if eplb.decode_token_mask is not None:
            decode_token_mask = torch.zeros(
                padded_batch_size * self.dp_size,
                dtype=eplb.decode_token_mask.dtype,
                device=eplb.decode_token_mask.device,
            )
        return EplbRuntimeState(
            expert_load_data=eplb.expert_load_data,
            decode_token_mask=decode_token_mask,
            is_graph_warmup=eplb.is_graph_warmup,
        )

    def _fill_graph_eplb_decode_mask(
        self,
        entry: _DecodeGraphEntry,
        eplb: EplbRuntimeState | None,
        metadata: AttentionMetadata,
    ) -> None:
        entry_eplb = getattr(entry, "eplb", None)
        if entry_eplb is None or entry_eplb.decode_token_mask is None:
            return
        if eplb is None or eplb.decode_token_mask is None:
            raise RuntimeError("EPLB decode-mask state is missing for a captured graph")

        destination = entry_eplb.decode_token_mask
        source = eplb.decode_token_mask.reshape(-1)
        destination.zero_()
        if self.dp_size == 1:
            if source.numel() > entry.batch_size:
                raise RuntimeError("EPLB decode mask exceeds graph bucket capacity")
            destination[: source.numel()].copy_(source)
            return

        raw_counts = tuple(metadata.raw_dp_execution_token_counts)
        destination_slices = _padded_rank_slices(raw_counts, self.dp_size, entry.batch_size)
        if source.numel() != sum(raw_counts):
            raise RuntimeError("EPLB decode mask does not match DP token counts")
        source_begin = 0
        for destination_begin, count in destination_slices:
            destination[destination_begin : destination_begin + count].copy_(
                source[source_begin : source_begin + count]
            )
            source_begin += count

    @staticmethod
    def _graph_key(
        padded_batch_size: int,
        is_expanded: bool,
        input_embedding: torch.Tensor | None,
    ) -> _GraphKey:
        """Return the key for a shape- and metadata-specific graph."""
        if input_embedding is None:
            return padded_batch_size, is_expanded, None, None, None
        return (
            padded_batch_size,
            is_expanded,
            input_embedding.dtype,
            input_embedding.device,
            tuple(input_embedding.shape[1:]),
        )

    def _allocate_entry(
        self,
        padded_batch_size: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> _DecodeGraphEntry:
        device = input_ids.device
        (
            _,
            _,
            _,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        ) = self._decode_metadata(metadata)
        if self._paged_kv_indices_buffer is None:
            page_size = self.attention_backend.page_size
            max_blocks_per_sequence = (self.max_model_len + page_size - 1) // page_size
            # Capacity in TOKEN rows: MTP spec-verify captures N*width rows,
            # so the flat paged_kv_indices buffer must hold
            # max_batch * num_decoding_tokens * max_blocks entries (item 二);
            # sizing it by sequences alone underflows on expanded-verify copy.
            self._paged_kv_indices_buffer = torch.zeros(
                self.max_batch * self.num_decoding_tokens * max_blocks_per_sequence,
                dtype=paged_kv_indices.dtype,
                device=device,
            )
            self._max_blocks_per_sequence = max_blocks_per_sequence

        static_block_table = torch.zeros(
            padded_batch_size,
            self._max_blocks_per_sequence,
            dtype=torch.int32,
            device=device,
        )

        entry = _DecodeGraphEntry()
        entry.batch_size = padded_batch_size
        entry.graph = None
        entry.static_output = None
        entry.graph_tasks = []
        entry.execution_state = AclGraphExecutionState({})
        entry.execution_contexts = allocate_graph_execution_contexts(
            self.execution_context_providers,
            padded_batch_size,
            device,
            metadata,
        )
        entry.static_input_ids = torch.zeros(padded_batch_size, dtype=input_ids.dtype, device=device)
        entry.static_positions = torch.zeros(padded_batch_size, dtype=torch.int32, device=device)
        entry.static_input_embedding = None
        entry.static_metadata = _StaticAttentionMetadata(
            slot_mapping=torch.zeros(
                padded_batch_size,
                dtype=torch.int32,
                device=device,
            ),
            paged_kv_indptr=torch.zeros(
                padded_batch_size + 1,
                dtype=paged_kv_indptr.dtype,
                device=device,
            ),
            paged_kv_indices=self._paged_kv_indices_buffer,
            paged_kv_last_page_len=torch.zeros(
                padded_batch_size,
                dtype=paged_kv_last_page_len.dtype,
                device=device,
            ),
            kv_cu_seq_lens=torch.zeros(
                padded_batch_size + 1,
                dtype=torch.int32,
                device=device,
            ),
            # paged_kv_indptr_host / paged_kv_last_page_len_host are consumed
            # only by the CUDA flashinfer backend (see attention/flashinfer.py);
            # the ACL decode path never reads them, so leaving them None keeps
            # capture allocation-only overhead off the CPU.
            kv_seq_lens_host_values=[1] * padded_batch_size,
            # DSA consumes live scheduler slots on the host before capture.
            new_cache_slots_host_values=[],
            block_table=static_block_table,
            # KDA (linear-attention) decode reads per-sequence conv/ssm state
            # slots from these buffers.  Static so the captured graph indexes a
            # fixed address; contents are refreshed by _fill_entry each step.
            linear_state_indices=torch.zeros(padded_batch_size, dtype=torch.int64, device=device),
            has_initial_state=torch.zeros(padded_batch_size, dtype=torch.int32, device=device),
            # DP layers read uniform padded counts off the captured metadata;
            # variable per-rank counts cannot be baked into a graph.
            dp_execution_token_counts=(padded_batch_size,) * self.dp_size if self.dp_size > 1 else (),
            dp_is_decode=tuple([1] * self.dp_size) if self.dp_size > 1 else (),
            # One mask slot per padded row across all DP ranks. aclnnMegaMoe
            # dispatches the fixed graph shape over EP, so padded lanes must be
            # marked inactive or they are routed as real tokens.
            mega_moe_token_mask=(
                torch.zeros(
                    padded_batch_size * self.dp_size,
                    dtype=torch.int8,
                    device=device,
                )
                if self._enable_mega_moe_token_mask
                else None
            ),
        )
        entry.static_metadata.q_cu_host_values = [0] * (padded_batch_size + 1)
        # One resolve for both the boolean and the row-count read below;
        # _expanded_verify_view runs an on-device paging build for the
        # untyped-verify branch, so calling it twice per bucket capture
        # duplicates a masked_select + cumsum for no benefit.
        expanded_view = self._expanded_verify_view(metadata)
        is_expanded = expanded_view is not None
        entry.kv_seq_lens_delta = torch.empty(padded_batch_size, dtype=torch.int32, device=device)
        # The graph metadata update writes per-sequence KV lengths into this
        # buffer.  MLA/SFA consumes the same stable buffer as its key lengths.
        entry.static_metadata.kv_seq_lens = entry.kv_seq_lens_delta
        if is_expanded:
            # Spec-verify rows per logical sequence: the static q_cu holds
            # GROUP boundaries [0, w, 2w, ...] so the in-graph KDA verify
            # grouping derives the sequence count without host syncs.
            # Derive the token-row count from the EXPANDED view (N*width),
            # not the top-level metadata.kv_seq_lens which stays at the logical
            # sequence count (N) for typed expanded verify — otherwise
            # spec_width stays 1 and q_seq_lens is built as N*w single-token
            # groups, breaking the KDA recurrent-state chain.
            expanded_kv = expanded_view.kv_seq_lens
            src_rows = expanded_kv.shape[0] if expanded_kv is not None else 0
            src_lsi = getattr(metadata, "linear_state_indices", None)
            src_seqs = src_lsi.shape[0] if src_lsi is not None else 0
            spec_width = 1
            if src_seqs > 0 and src_rows > src_seqs and src_rows % src_seqs == 0:
                spec_width = src_rows // src_seqs
            w = spec_width
            if padded_batch_size % w == 0:
                # q_cu stays PER-ROW (the eager verify layout the attention
                # backends consume); the per-SEQENCE group count rides on
                # q_seq_lens (N entries of value w).
                entry.static_metadata.q_cu_seq_lens = torch.arange(
                    0, padded_batch_size + 1, 1, dtype=torch.int32, device=device
                )
                entry.static_metadata.q_cu_host_values = list(range(padded_batch_size + 1))
                entry.static_metadata.q_seq_lens = torch.full(
                    (padded_batch_size // w,), w, dtype=torch.int32, device=device
                )
            entry.static_metadata.expanded_decode_metadata = ExpandedDecodeMetadata(
                kv_seq_lens=entry.kv_seq_lens_delta,
                block_table=entry.static_metadata.block_table,
                paged_kv_indptr=entry.static_metadata.paged_kv_indptr,
                paged_kv_indices=entry.static_metadata.paged_kv_indices,
                paged_kv_last_page_len=(entry.static_metadata.paged_kv_last_page_len),
                paged_attention_tiling_data=None,
                kv_seq_lens_host=None,
                kv_seq_lens_host_values=(
                    entry.static_metadata.kv_seq_lens_host_values
                    if getattr(
                        self.attention_backend,
                        "requires_host_kv_lengths",
                        False,
                    )
                    or not getattr(self.attention_backend, "is_mla", False)
                    else None
                ),
            )
        entry.static_metadata.multi_block_tables = self._build_static_multi_block_tables(
            padded_batch_size,
            device,
        )
        entry.static_metadata.dsa_graph_block_table_cols = self._max_blocks_per_sequence
        entry.static_metadata.dsa_graph_mode = True
        return entry

    def _build_static_multi_block_tables(
        self,
        padded_batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        """Build stable SWA/C4/C128 manager block tables for graph capture.

        Keep every manager's dimensions (and therefore its packed DSA metadata
        views) fixed across decodes of the same bucket. The first manager uses
        the runner-wide maximum block column count; compressed managers use the
        ``max_model_len``-derived upper bound. Unused entries point at reserved
        block 0.
        """
        max_swa_cols = self._max_blocks_per_sequence
        group_infos = getattr(self.attention_backend, "group_infos", None)
        if group_infos is None:
            # Test stubs and non-DSA backends only build a sliding-window row.
            manager_infos = [None]
        else:
            manager_infos = list(group_infos)
        tables: list[torch.Tensor] = []
        for group_info in manager_infos:
            if group_info is not None and getattr(group_info, "cache_type", None) == DSA_CACHE_TOKEN:
                ratio = max(int(getattr(group_info, "ratio", 1)), 1)
                block_size = max(int(getattr(group_info, "block_size", 1)), 1)
                compressed_block_size = ratio * block_size
                max_cols = max(
                    1,
                    (self.max_model_len + compressed_block_size - 1) // compressed_block_size,
                )
            else:
                max_cols = max_swa_cols
            tables.append(
                torch.zeros(
                    (padded_batch_size, max_cols),
                    dtype=torch.int32,
                    device=device,
                )
            )
        return tuple(tables)

    def _fill_entry(
        self,
        entry: _DecodeGraphEntry,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        batch_size: int,
        input_embedding: torch.Tensor | None,
    ) -> None:
        padded_batch_size = entry.batch_size
        static_metadata = entry.static_metadata
        static_metadata.is_dummy = bool(getattr(metadata, "is_dummy", False))
        (
            block_table,
            kv_seq_lens,
            kv_seq_lens_host_values,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        ) = self._decode_metadata(metadata)
        self._fill_graph_dsa_positions(entry, positions)
        if batch_size != block_table.shape[0]:
            raise RuntimeError("ACL graph decode batch size must match metadata sequences")
        slot_mapping = self._effective_slot_mapping(
            metadata,
            block_table.shape[0],
            input_ids.device,
        )
        self._validate_decode_token_layout(
            input_ids,
            positions,
            slot_mapping,
            block_table.shape[0],
        )
        # Use the cheap shape-only detectors (no on-device paging build) for
        # the boolean; _decode_metadata above already built paging once via
        # _expanded_verify_view — don't rebuild it just to discard the result.
        is_expanded = resolve_expanded_decode_metadata(metadata) is not None or self._is_untyped_spec_verify(metadata)
        cumulative_kv_seq_lens = self._cumulative_lengths(
            kv_seq_lens,
            None if is_expanded else metadata.kv_cu_seq_lens,
        )
        graph_positions = positions.to(torch.int32).contiguous()
        kernels.update_decode_graph_metadata(
            input_ids,
            graph_positions,
            slot_mapping,
            cumulative_kv_seq_lens,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            entry.static_input_ids,
            entry.static_positions,
            static_metadata.slot_mapping,
            static_metadata.kv_cu_seq_lens,
            entry.kv_seq_lens_delta,
            static_metadata.paged_kv_indptr,
            static_metadata.paged_kv_indices,
            static_metadata.paged_kv_last_page_len,
            padded_batch_size,
        )
        # linear_state_indices is filled once below (the robust
        # repeat_interleave slice path), together with has_initial_state, so
        # there is no first redundant fill here.
        self._fill_host_metadata(entry, kv_seq_lens_host_values, batch_size)
        self._fill_new_cache_slots_host_metadata(entry, metadata, batch_size)

        if input_embedding is not None:
            if input_embedding.shape[0] != batch_size:
                raise ValueError("ACL graph input_embedding token count must match input_ids")
            if entry.static_input_embedding is None:
                entry.static_input_embedding = torch.zeros(
                    entry.batch_size,
                    input_embedding.shape[-1],
                    dtype=input_embedding.dtype,
                    device=input_embedding.device,
                )
            elif entry.static_input_embedding.shape[1:] != input_embedding.shape[1:]:
                raise ValueError("ACL graph input_embedding shape changed for a graph bucket")
            entry.static_input_embedding[:batch_size].copy_(input_embedding)
            if entry.batch_size > batch_size:
                entry.static_input_embedding[batch_size:].zero_()
        elif entry.static_input_embedding is not None:
            entry.static_input_embedding.zero_()

        if static_metadata.block_table is not None:
            src_bt = block_table
            copy_cols = min(
                src_bt.shape[1],
                static_metadata.block_table.shape[1],
            )
            static_metadata.block_table[:batch_size, :copy_cols].copy_(src_bt[:batch_size, :copy_cols])
            if padded_batch_size > batch_size:
                static_metadata.block_table[batch_size:].zero_()

        # Padded lanes must remain valid inputs for sparse MLA tiling.  Their
        # token and slot mapping are dummy values, so one KV token is safe.
        if padded_batch_size > batch_size:
            static_metadata.slot_mapping[batch_size:].fill_(-1)
            entry.kv_seq_lens_delta[batch_size:].fill_(1)

        # KDA (linear-attention) static state slots.  Padded lanes point at
        # slot 0, which the linear-state block manager reserves as its padding
        # slot (block_manager_impl padding_block_), so their conv/ssm writes
        # never touch a live sequence's state.  A GENERIC-flow spec-verify
        # batch carries one slot id per logical sequence: expand per row.
        if static_metadata.linear_state_indices is not None:
            src_idx = getattr(metadata, "linear_state_indices", None)
            if src_idx is not None:
                if src_idx.numel() >= batch_size:
                    static_metadata.linear_state_indices[:batch_size].copy_(src_idx[:batch_size])
                else:
                    width = batch_size // src_idx.numel()
                    static_metadata.linear_state_indices[:batch_size].copy_(
                        src_idx.repeat_interleave(width)[:batch_size]
                    )
            if padded_batch_size > batch_size:
                static_metadata.linear_state_indices[batch_size:].zero_()
        if static_metadata.has_initial_state is not None:
            src_his = getattr(metadata, "has_initial_state", None)
            if src_his is None:
                # The eager runtime omits the validity mask at decode
                # (has_initial_state is None), which execute_linear reads as
                # "leave the gathered slot state untouched". Reproduce that
                # through the static buffer by marking every lane warm —
                # torch.where(warm, state, zeros) then passes the state
                # through unchanged.
                static_metadata.has_initial_state.fill_(1)
            else:
                if not isinstance(src_his, torch.Tensor):
                    src_his = torch.tensor(
                        src_his,
                        dtype=static_metadata.has_initial_state.dtype,
                        device=static_metadata.has_initial_state.device,
                    )
                # has_initial_state is sequence-scoped (N entries) while the
                # static buffer is per token ROW (N*width under MTP expanded
                # verify); expand each sequence's mask across its spec rows
                # before copying, otherwise copy_ hits a shape mismatch.
                if src_his.numel() < batch_size and src_his.numel() > 0 and batch_size % src_his.numel() == 0:
                    width = batch_size // src_his.numel()
                    src_his = src_his.repeat_interleave(width)
                static_metadata.has_initial_state[:batch_size].copy_(src_his[:batch_size])
                if padded_batch_size > batch_size:
                    static_metadata.has_initial_state[batch_size:].zero_()
        self._fill_dsa_block_tables(
            static_metadata,
            metadata,
            block_table,
            batch_size,
        )
        self._fill_mega_moe_token_mask(entry, metadata, batch_size)
        update_graph_execution_contexts(
            self.execution_context_providers,
            entry.execution_contexts,
            metadata,
            batch_size,
        )

    def _fill_dsa_block_tables(
        self,
        static_metadata: _StaticAttentionMetadata,
        metadata: AttentionMetadata,
        block_table: torch.Tensor,
        batch_size: int,
    ) -> None:
        """Refresh every stable DSA manager table for the current decode."""
        static_tables = list(static_metadata.multi_block_tables)
        if not static_tables:
            return
        source_tables = list(getattr(metadata, "multi_block_tables", ()) or ())
        if not source_tables:
            if getattr(metadata, "is_dummy", False):
                for target in static_tables:
                    target.zero_()
                return
            if getattr(self.attention_backend, "group_infos", None) is not None:
                raise RuntimeError("DeepSeek-V4 ACL graph requires all DSA manager block tables")
            source_tables = [block_table]
        if len(source_tables) < len(static_tables):
            raise RuntimeError(f"ACL graph DSA manager count changed: {len(source_tables)} != {len(static_tables)}")
        if len(source_tables) > len(static_tables):
            # MTP draft inputs inherit the target's SWA/C4/C128 tables, while
            # the draft graph registers only its own SWA prefix.
            source_tables = source_tables[: len(static_tables)]

        for manager_id, (target, source) in enumerate(zip(static_tables, source_tables, strict=True)):
            # Every graph padding row points at permanently reserved block 0.
            target.zero_()
            if source is None:
                continue
            if source.dim() != 2:
                raise RuntimeError(f"ACL graph DSA manager {manager_id} block table must be two-dimensional")
            source_rows = int(source.shape[0])
            if source_rows != batch_size:
                if source_rows <= 0 or batch_size % source_rows != 0:
                    raise RuntimeError(
                        f"ACL graph DSA manager {manager_id} row count "
                        f"{source_rows} does not match batch size {batch_size}"
                    )
                source = source.repeat_interleave(batch_size // source_rows, dim=0)
            if source.shape[1] > target.shape[1]:
                raise RuntimeError(
                    f"ACL graph DSA manager {manager_id} requires {source.shape[1]} "
                    f"columns, capacity is {target.shape[1]}"
                )
            target[:batch_size, : source.shape[1]].copy_(source[:batch_size])

    def _fill_graph_dsa_positions(
        self,
        entry: _DecodeGraphEntry,
        positions: torch.Tensor,
    ) -> None:
        """Keep a stable DSA position tensor synced with the static bucket."""
        static_metadata = entry.static_metadata
        padded_batch_size = entry.batch_size
        if static_metadata.dsa_positions is None:
            static_metadata.dsa_positions = torch.zeros(
                padded_batch_size,
                dtype=torch.int64,
                device=entry.static_positions.device,
            )
        graph_positions = positions.to(device=entry.static_positions.device, dtype=torch.int64)
        copy_count = min(int(graph_positions.numel()), padded_batch_size)
        static_metadata.dsa_positions[:copy_count].copy_(graph_positions[:copy_count])
        if padded_batch_size > copy_count:
            static_metadata.dsa_positions[copy_count:].zero_()

    def _fill_mega_moe_token_mask(
        self,
        entry: _DecodeGraphEntry,
        metadata: AttentionMetadata,
        batch_size: int,
    ) -> None:
        mask = entry.static_metadata.mega_moe_token_mask
        if mask is None:
            return
        mask.zero_()
        if self.dp_size == 1:
            mask[:batch_size].fill_(1)
            return

        padded_batch_size = entry.batch_size
        rank_slices = _padded_rank_slices(
            metadata.dp_execution_token_counts,
            self.dp_size,
            padded_batch_size,
        )
        for start, count in rank_slices:
            mask[start : start + count].fill_(1)

    def _fill_host_metadata(
        self,
        entry: _DecodeGraphEntry,
        kv_seq_lens: list[int] | None,
        batch_size: int,
    ) -> None:
        if getattr(self.attention_backend, "is_mla", False) and not getattr(
            self.attention_backend,
            "requires_host_kv_lengths",
            False,
        ):
            return
        if kv_seq_lens is None:
            raise RuntimeError("decode ACL graph requires scheduler-provided host KV lengths")
        if len(kv_seq_lens) != batch_size:
            raise RuntimeError("decode ACL graph requires per-sequence host KV lengths")

        padded_batch_size = entry.batch_size
        static_metadata = entry.static_metadata
        static_kv_seq_lens = static_metadata.kv_seq_lens_host_values
        if static_kv_seq_lens is None:
            raise RuntimeError("decode ACL graph host KV buffer is missing")
        static_kv_seq_lens[:batch_size] = kv_seq_lens
        if padded_batch_size > batch_size:
            static_kv_seq_lens[batch_size:] = [1] * (padded_batch_size - batch_size)

    @staticmethod
    def _fill_new_cache_slots_host_metadata(
        entry: _DecodeGraphEntry,
        metadata: AttentionMetadata,
        batch_size: int,
    ) -> None:
        """Copy scheduler-resolved physical DSA slots into graph metadata."""
        source_slots = getattr(metadata, "new_cache_slots_host_values", None)
        target_slots = entry.static_metadata.new_cache_slots_host_values
        if target_slots is None:
            return
        if source_slots is None:
            entry.static_metadata.new_cache_slots_host_values = []
            return
        if not source_slots and getattr(metadata, "is_dummy", False):
            # Empty-DP dummy rows have no scheduler-owned cache slot. Their
            # DSA tables point entirely at reserved block 0, so let the DSA
            # builder derive a safe slot from that table.
            entry.static_metadata.new_cache_slots_host_values = []
            return
        if len(source_slots) < batch_size:
            raise RuntimeError("decode ACL graph requires one host cache slot per token")
        real_slots = [int(slot) for slot in source_slots[:batch_size]]
        padding_slots = [0] * (entry.batch_size - batch_size)
        entry.static_metadata.new_cache_slots_host_values = real_slots + padding_slots

    def _capture(self, entry: _DecodeGraphEntry) -> None:
        # The pre-capture warmup forwards execute for real. KV/index-cache
        # writes are idempotent (same slot, same value), but linear-attention
        # (KDA) conv/ssm state ADVANCES on every run, so warmup + capture
        # would leave each sequence's recurrent state several steps ahead.
        # Snapshot the touched state slots and restore them after capture.
        linear_snapshot = self._snapshot_linear_state(entry)
        # V3 combined [base|draft0|...|draft{R-1}] pools follow the same
        # lifecycle (warmup + capture advance them); restore entry contents or
        # the first replay resumes from a state several steps stale.
        v3_snapshot = None
        v3_snap_fn = getattr(self.attention_backend, "snapshot_kda_v3_state", None)
        if v3_snap_fn is not None and entry.static_metadata.linear_state_indices is not None:
            v3_snapshot = v3_snap_fn(entry.static_metadata.linear_state_indices)
        context = ForwardContext(
            self.attention_backend,
            self.device,
            entry.static_metadata,
            self.layer_caches,
            execution_state=entry.execution_state,
            eplb=getattr(entry, "eplb", None),
            execution_contexts=entry.execution_contexts,
        )
        # The snapshots above (linear/v3) read conv/ssm state on the
        # current (default) stream, while the warmup forward below advances
        # that state on self._stream. NPU cross-stream accesses to the same
        # memory are not auto-serialized, so make the warmup stream wait for
        # the snapshot reads to complete first — otherwise the snapshot may
        # land after the advance and restore a state that is already stale.
        self._stream.wait_stream(torch.npu.current_stream())
        with forward_context(context), torch.npu.stream(self._stream):
            for _ in range(_CAPTURE_WARMUP_STEPS):
                self._forward_static(entry)
        torch.npu.synchronize()
        entry.graph = torch.npu.NPUGraph()
        capture_context = AclGraphCaptureContext(self._stream, [])
        context = ForwardContext(
            self.attention_backend,
            self.device,
            entry.static_metadata,
            self.layer_caches,
            acl_graph=capture_context,
            execution_state=entry.execution_state,
            eplb=getattr(entry, "eplb", None),
            execution_contexts=entry.execution_contexts,
        )
        with forward_context(context), torch.npu.graph(entry.graph, stream=self._stream):
            entry.static_output = self._forward_static(entry)
        entry.graph_tasks = capture_context.tasks
        self._restore_linear_state(entry, linear_snapshot)
        if v3_snapshot is not None:
            v3_restore_fn = getattr(self.attention_backend, "restore_kda_v3_state", None)
            if v3_restore_fn is not None:
                v3_restore_fn(v3_snapshot)

    def _snapshot_linear_state(
        self, entry: _DecodeGraphEntry
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None:
        """Copy the conv/ssm rows the capture run is about to advance."""
        idx = entry.static_metadata.linear_state_indices
        if idx is None:
            return None
        snapshot = []
        for cache in self.layer_caches:
            conv = getattr(cache, "conv", None)
            ssm = getattr(cache, "ssm", None)
            if conv is None or ssm is None:
                continue
            snapshot.append(
                (
                    conv,
                    ssm,
                    conv.index_select(0, idx).clone(),
                    ssm.index_select(0, idx).clone(),
                )
            )
        return snapshot or None

    @staticmethod
    def _restore_linear_state(
        entry: _DecodeGraphEntry,
        snapshot: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None,
    ) -> None:
        if not snapshot:
            return
        idx = entry.static_metadata.linear_state_indices
        for conv, ssm, conv_rows, ssm_rows in snapshot:
            conv.index_copy_(0, idx, conv_rows)
            ssm.index_copy_(0, idx, ssm_rows)
        torch.npu.synchronize()

    def _forward_static(self, entry: _DecodeGraphEntry) -> torch.Tensor:
        if entry.static_input_embedding is None:
            return self.model(entry.static_input_ids, entry.static_positions)
        return self.model(
            entry.static_input_ids,
            entry.static_positions,
            entry.static_input_embedding,
        )

    @staticmethod
    def _update_graph_tasks(
        stream: torch.npu.Stream,
        graph_tasks: list[AclGraphTask],
    ) -> None:
        for task in graph_tasks:
            torch.npu.graph_task_update_begin(stream, task.handle)
            task.update()
            torch.npu.graph_task_update_end(stream)
            task.event.record(stream)
