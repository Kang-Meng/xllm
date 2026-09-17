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

"""NPU attention backend using Fused-Infer-Attention (FIA).

Registers as the PrivateUse1 (NPU) backend for the Python model executor.
Prefill uses FIA TND with causal mask; decode uses FIA TND with block_table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import torch_npu

from xllm.python import distributed, kernels
from xllm.python.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    LayerCache,
    MlaIndexContext,
    MlaPreprocessContext,
)
from xllm.python.attention.expanded_decode_metadata import (
    resolve_expanded_decode_metadata,
)
from xllm.python.attention.kv_shard_layout import has_rope_dim
from xllm.python.model_executor.cp_utils import cp_gather_kv
from xllm.python.model_executor.forward_context import (
    AclGraphTask,
    get_execution_buffer,
    get_forward_context,
    get_forward_context_or_none,
    in_acl_graph,
)

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention
    from xllm.python.model_executor.cp_utils import CpContext

from xllm.python.attention.kda_linear_attention import (
    KdaLinearAttentionMixin,
)

# Ascend FIA sparse_mode values (see CANN aclnnFusedInferAttentionScore docs).
# 0: no compressed mask; used for single-query decode where no causal mask is
#    needed.
# 3: rightDownCausal; the causal mask is right-aligned to the KV tail, for the
#    prefix-cache / chunked-prefill case where q_len < kv_len so the new queries
#    attend the full cached prefix plus their own tokens (mode 2, leftUpCausal,
#    only aligns when q_len == kv_len and would misalign on a cache hit).
_SPARSE_MODE_NONE = 0
_SPARSE_MODE_RIGHT_DOWN_CAUSAL = 3

_HAS_FIA_V2 = hasattr(torch.ops.npu, "npu_fused_infer_attention_score_v2") and hasattr(
    torch_npu, "_npu_fused_infer_attention_score_v2_get_max_workspace"
)


@dataclass(frozen=True, slots=True)
class _SfaPageLayout:
    source_page_ids: torch.Tensor
    target_page_ids: torch.Tensor
    block_table: torch.Tensor
    page_count: int


def _mla_graph_max_seqlen_k(
    block_table: torch.Tensor,
    page_size: int,
) -> int:
    """Return a replay-stable KV length bound for MLA graph metadata."""
    max_seqlen_k = int(block_table.shape[1]) * int(page_size)
    if max_seqlen_k <= 0:
        raise RuntimeError("MLA graph block-table capacity must be positive")
    return max_seqlen_k


def _build_stable_sfa_page_layout(
    materialized_block_table: torch.Tensor,
) -> _SfaPageLayout:
    """Build a deterministic, legacy-compatible SFA-only page layout."""
    if materialized_block_table.ndim != 2:
        raise RuntimeError("materialized SFA block table must be two-dimensional")

    valid_pages = materialized_block_table >= 0
    stable_block_table = torch.arange(
        materialized_block_table.numel(),
        dtype=torch.int32,
        device=materialized_block_table.device,
    ).view_as(materialized_block_table)
    # The existing KV1 allocator presents the first two pages in [1, 0] order
    # after dense renumbering. Sparse SFA is numerically sensitive to this page
    # order, so preserve it per sequence while deriving every page id and table
    # width from the live materialized metadata.
    if materialized_block_table.shape[1] > 1:
        swap_rows = valid_pages[:, 1]
        first_pages = stable_block_table[:, 0].clone()
        second_pages = stable_block_table[:, 1].clone()
        stable_block_table[:, 0] = torch.where(
            swap_rows,
            second_pages,
            first_pages,
        )
        stable_block_table[:, 1] = torch.where(
            swap_rows,
            first_pages,
            second_pages,
        )
    stable_block_table = torch.where(
        valid_pages,
        stable_block_table,
        torch.full_like(stable_block_table, -1),
    ).contiguous()
    source_page_ids = materialized_block_table.masked_select(valid_pages).to(torch.int64)
    target_page_ids = stable_block_table.masked_select(valid_pages).to(torch.int64)
    return _SfaPageLayout(
        source_page_ids=source_page_ids,
        target_page_ids=target_page_ids,
        block_table=stable_block_table,
        page_count=materialized_block_table.numel(),
    )


def _causal_conv1d_graph_multi(
    cin: torch.Tensor,
    weight: torch.Tensor,
    out_rows: int,
    activation: str = "silu",
) -> torch.Tensor:
    """Graph-capturable multi-row twin of the eager depthwise conv.

    ``cin`` is ``[B, conv_dim, state_len + R]`` (boundary tail + the R
    current rows); the causal outputs for the R rows are the K-wide windows
    STARTING at ``[0, R)``. F.conv1d lowers to an aclop NPUGraph cannot
    capture, so the conv is unrolled into the per-tap multiply-add contract
    of ``_causal_conv1d_update_graph`` — bit-compatible with that plain-decode
    path and the eager F.conv1d path.

    Mirrors _causal_conv1d_update_graph's numeric contract exactly: operands
    cast to fp32, per-tap products exact in fp32, ascending accumulation,
    one final round to the weight dtype. The RNE rounding to 11 mantissa bits
    is a no-op for bf16 sources (7 mantissa bits), so it is skipped to avoid
    RightShift/BitwiseAnd on AI_CPU (see commit 32760093).
    """
    h_r = cin.to(weight.dtype).float()
    w_r = weight.float()
    k_size = weight.shape[-1]
    out = w_r[:, 0:1].unsqueeze(0) * h_r[:, :, 0:out_rows]
    for k in range(1, k_size):
        out = out + w_r[:, k : k + 1].unsqueeze(0) * h_r[:, :, k : k + out_rows]
    out = out.to(weight.dtype)
    if activation == "silu":
        out = F.silu(out)
    return out.to(cin.dtype)


def write_mla_paged_cache(
    slot_mapping: torch.Tensor,
    k_latent_3d: torch.Tensor,
    k_pe_3d: torch.Tensor | None,
    nope_cache: torch.Tensor,
    rope_cache: torch.Tensor | None,
) -> None:
    """Scatter MLA KV into paged caches.

    ATB ``ReshapeAndCache`` rejects a 0-width rope/value tensor. NoPE therefore
    writes the latent into both key and value operands against ``nope_cache``.
    """
    if has_rope_dim(k_pe_3d):
        if not has_rope_dim(rope_cache):
            raise RuntimeError("MLA rope cache is missing for a non-empty k_pe")
        torch.ops.xllm_ops.reshape_paged_cache(
            slot_mapping,
            k_latent_3d,
            k_pe_3d,
            nope_cache,
            rope_cache,
        )
        return
    torch.ops.xllm_ops.reshape_paged_cache(
        slot_mapping,
        k_latent_3d,
        k_latent_3d,
        nope_cache,
        nope_cache,
    )


class NpuPagedAttentionBackend(KdaLinearAttentionMixin, AttentionBackend):
    """NPU attention backend dispatching to npu_fused_infer_attention_score."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        sliding_window: int,
        is_mla: bool,
        device: torch.device,
        dtype: torch.dtype,
        num_decoding_tokens: int = 1,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.sliding_window = sliding_window
        self.dtype = dtype
        self.device = device
        self._use_fia_v2 = _HAS_FIA_V2
        self._is_mla = is_mla
        self._uses_sparse_mla = False
        self._kda_verify_width = num_decoding_tokens

        self._kv_caches: list[LayerCache] = []
        self._num_kv_blocks: int | None = None
        self._page_size: int | None = None
        self._metadata: AttentionMetadata | None = None
        self._graph_workspace: torch.Tensor | None = None
        self._graph_outputs: dict[int, torch.Tensor] = {}
        self._graph_lses: dict[int, torch.Tensor] = {}
        self._current_graph_output: torch.Tensor | None = None
        self._current_graph_lse: torch.Tensor | None = None
        self._block_table_i32: torch.Tensor | None = None
        # Expanded-decode (spec-verify) flag. Default False; the prefill path
        # reads it to decide whether to route expanded tokens through _decode
        # instead of _prefill. Initialized here so non-MLA Python-NPU backends
        # (Qwen3, Qwen3-VL, …) that never set it do not hit AttributeError.
        self._use_expanded_decode: bool = False
        self._actual_seq_lens: list[int] | None = None
        self._actual_seq_q: list[int] = []
        self._actual_seq_kv: list[int] = []
        self._mla_actual_seq_q: torch.Tensor | None = None
        self._mla_actual_seq_kv: torch.Tensor | None = None
        # Dense (FIA v2) MLA state: host cumulative seq-lens consumed by
        # npu_fused_infer_attention_score_v2, plus graph-mode workspace/output
        # buffers. Only populated when the dense path runs (topk is None), so
        # sparse MLA never pays the D2H or allocates these.
        self._mla_actual_seq_q_host: list[int] | None = None
        self._mla_actual_seq_kv_host: list[int] | None = None
        self._mla_graph_workspaces: dict[tuple[int, ...], torch.Tensor] = {}
        self._mla_graph_outputs: dict[tuple[int, ...], torch.Tensor] = {}
        self._mla_graph_lses: dict[tuple[int, ...], torch.Tensor] = {}
        self._mla_quant_indexer_metadata: dict[tuple[int, int, int, int], torch.Tensor] = {}
        self._mla_max_seqlen_q = 0
        self._mla_max_seqlen_k = 0
        self._kv_owner_representatives: torch.Tensor | None = None
        self._materialized_block_table: torch.Tensor | None = None
        self._sfa_page_layout: _SfaPageLayout | None = None
        self._graph_index_history_max_kv: int | None = None

        self._causal_mask = (
            torch.triu(torch.ones(2048, 2048, dtype=torch.float32), 1).to(torch.int8).contiguous().to(device)
        )

    @property
    def num_kv_blocks(self) -> int:
        if self._num_kv_blocks is None:
            raise RuntimeError("full-attention KV caches are not bound")
        return self._num_kv_blocks

    @property
    def page_size(self) -> int:
        if self._page_size is None:
            raise RuntimeError("full-attention KV caches are not bound")
        return self._page_size

    def indexer_block_table_or_none(self) -> torch.Tensor | None:
        """Block table addressing the paged index cache, or ``None`` if unpaged.

        Returns the engine table (one column per ``page_size`` tokens) cast to
        int32. DCP backends override this with the physical-page expansion.
        """
        if self._block_table_i32 is not None:
            return self._block_table_i32
        metadata = self._metadata
        if metadata is None:
            return None
        return metadata.block_table

    def indexer_block_table(self) -> torch.Tensor:
        """Same as :meth:`indexer_block_table_or_none`, but requires a table."""
        block_table = self.indexer_block_table_or_none()
        if block_table is None:
            raise RuntimeError("indexer_block_table needs a paged block_table")
        return block_table

    @property
    def graph_index_history_max_kv(self) -> int:
        """Static KV-length cap for the kPool graph gather.

        The graph branch of ``gather_index_history`` densifies each sequence
        to a fixed ``[num_seqs, max_kv, width]`` buffer; sizing it by the full
        block-table capacity (max_position_embeddings can be 1M) is not
        viable. Decode steps whose block table exceeds this cap fall back to
        the eager runner (see DecodeAclGraphRunner), which keeps the dynamic
        gather. Override with XLLM_GRAPH_INDEX_HISTORY_MAX_KV.
        """
        if self._graph_index_history_max_kv is None:
            self._graph_index_history_max_kv = int(os.environ.get("XLLM_GRAPH_INDEX_HISTORY_MAX_KV", "32768"))
        return self._graph_index_history_max_kv

    @property
    def is_mla(self) -> bool:
        return self._is_mla

    @property
    def requires_host_kv_lengths(self) -> bool:
        """Whether ACL Graph replay must update FIA's host KV-length list."""
        return self._is_mla and not self._uses_sparse_mla

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        full_attention_caches = [(cache.key, cache.value) for cache in kv_caches if cache.key is not None]
        if not full_attention_caches:
            raise RuntimeError("no full-attention KV cache is bound")

        page_sizes = {key.shape[1] for key, _ in full_attention_caches}
        if len(page_sizes) != 1:
            raise RuntimeError("full-attention layers use inconsistent page sizes")

        num_kv_blocks = {key.shape[0] for key, _ in full_attention_caches}
        if len(num_kv_blocks) != 1:
            raise RuntimeError("full-attention layers use inconsistent KV block counts")

        self._kv_caches = kv_caches
        self._page_size = page_sizes.pop()
        self._num_kv_blocks = num_kv_blocks.pop()
        has_sparse_index = any(cache.index is not None for cache in kv_caches)
        # glm5_next DSA layers are NoPE: the latent lives in the key slot and
        # the value/rope slot is a 0-dim tensor normalized to None, while the
        # kPool indexer adds a paged index cache. Either signal marks this
        # backend instance as MLA even though the constructor heuristic
        # (head_dim > 192 and num_kv_heads == 1) does not fire for it.
        has_latent_only_cache = any(
            cache.key is not None and cache.value is None and cache.conv is None for cache in kv_caches
        )
        if has_sparse_index or has_latent_only_cache:
            self._is_mla = True
        self._uses_sparse_mla = self._is_mla and has_sparse_index

    @staticmethod
    def _query_sequence_ends(
        q_cu_seq_lens: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor | None:
        """Accept both NPU q-cumulative layouts used by the runtime."""
        if q_cu_seq_lens is None:
            return None
        if q_cu_seq_lens.numel() == batch_size:
            return q_cu_seq_lens.to(torch.int32)
        if q_cu_seq_lens.numel() == batch_size + 1:
            return q_cu_seq_lens[1:].to(torch.int32)
        raise RuntimeError(
            "q cumulative sequence lengths must contain either one value per "
            "sequence or a leading zero plus one value per sequence"
        )

    def prepare(
        self,
        metadata: AttentionMetadata,
        *,
        graph_mode: bool = False,
    ) -> None:
        self._metadata = metadata
        if getattr(metadata, "q_cu_host_values", None) is not None:
            # Static graph metadata carries the (per-entry constant) host
            # copy: reading the device buffer would block the host until the
            # prior replay drains, serializing the scheduler behind device.
            self._actual_seq_lens = metadata.q_cu_host_values[1:]
        elif metadata.q_cu_seq_lens is not None:
            self._actual_seq_lens = metadata.q_cu_seq_lens[1:].cpu().tolist()
        else:
            self._actual_seq_lens = None

        if metadata.block_table is not None:
            self._block_table_i32 = metadata.block_table.to(torch.int32)

            real_batch = metadata.block_table.shape[0]

            kv_host = getattr(metadata, "kv_seq_lens_host", None)
            if kv_host is not None:
                kv_host = kv_host.cpu()
                if kv_host.numel() == real_batch + 1:
                    per_seq_kv = kv_host[1:] - kv_host[:-1]
                else:
                    per_seq_kv = kv_host
                kv_list = per_seq_kv[:real_batch].tolist()
            else:
                # Graph mode (decode_acl_graph) populates
                # kv_seq_lens_host_values (the host list) and leaves
                # kv_seq_lens_host (device tensor) None; falling back to ones
                # here would make every decode attend to a single KV token and
                # silently corrupt non-MLA graph output (first token right,
                # then collapse). Use the host list the scheduler provided.
                kv_host_values = getattr(metadata, "kv_seq_lens_host_values", None)
                if kv_host_values is not None:
                    kv_list = list(kv_host_values[:real_batch])
                else:
                    kv_list = [1] * real_batch

            self._actual_seq_q = list(range(1, real_batch + 1))
            self._actual_seq_kv = kv_list
        else:
            self._block_table_i32 = None
            self._actual_seq_q = []
            self._actual_seq_kv = []

        if graph_mode and self._block_table_i32 is not None and not self._is_mla:
            graph_batch_size = self._block_table_i32.shape[0]
            if self._graph_workspace is None:
                block_size = self.page_size
                dummy_q = torch.empty(
                    graph_batch_size,
                    self.num_heads,
                    self.head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
                dummy_kv = torch.empty(
                    self.num_kv_blocks,
                    block_size,
                    self.num_kv_heads * self.head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
                if self._use_fia_v2:
                    self._graph_workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                        query=dummy_q,
                        key=dummy_kv,
                        value=dummy_kv,
                        block_table=self._block_table_i32,
                        input_layout="TND",
                        block_size=block_size,
                        actual_seq_qlen=self._actual_seq_q,
                        actual_seq_kvlen=self._actual_seq_kv,
                        num_key_value_heads=self.num_kv_heads,
                        num_query_heads=self.num_heads,
                        sparse_mode=_SPARSE_MODE_NONE,
                        softmax_scale=self.scale,
                        return_softmax_lse=False,
                    )
                else:
                    self._graph_workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                        query=dummy_q,
                        key=dummy_kv,
                        value=dummy_kv,
                        block_table=self._block_table_i32,
                        input_layout="TND",
                        block_size=block_size,
                        actual_seq_lengths=self._actual_seq_q,
                        actual_seq_lengths_kv=self._actual_seq_kv,
                        num_key_value_heads=self.num_kv_heads,
                        num_heads=self.num_heads,
                        sparse_mode=_SPARSE_MODE_NONE,
                        scale=self.scale,
                        softmax_lse_flag=False,
                    )
            if graph_batch_size not in self._graph_outputs:
                self._graph_outputs[graph_batch_size] = torch.empty(
                    graph_batch_size,
                    self.num_heads,
                    self.head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
                self._graph_lses[graph_batch_size] = torch.empty(0, dtype=self.dtype, device=self.device)
            self._current_graph_output = self._graph_outputs[graph_batch_size]
            self._current_graph_lse = self._graph_lses[graph_batch_size]

        # Pre-cache MLA (sparse SFA) seq-lens once per step; shared by
        # execute_mla / mla_index_context instead of re-derived per layer.
        # Gated on kv_seq_lens (not _is_mla) so the eager path is unchanged
        # for every model; the graph_mode sub-branches only swap the tensors
        # into static execution buffers and derive replay-stable bounds.
        self._mla_quant_indexer_metadata.clear()
        if metadata.kv_seq_lens is not None:
            kv_seq_lens = metadata.kv_seq_lens
            mla_device = kv_seq_lens.device
            actual_seq_kv = kv_seq_lens.to(torch.int32).to(mla_device)
            if metadata.q_cu_seq_lens is not None:
                actual_seq_q = metadata.q_cu_seq_lens[1:].to(torch.int32).to(mla_device)
            else:
                batch = kv_seq_lens.size(0)
                actual_seq_q = torch.arange(1, batch + 1, dtype=torch.int32, device=mla_device)
            if graph_mode:
                # ACL graph replay reuses the captured kernel arguments, so
                # the seq-lens must live in static buffers that the runner
                # rewrites before each replay.
                graph_batch = int(actual_seq_kv.numel())
                self._mla_actual_seq_q = get_execution_buffer(
                    ("MLA_ACTUAL_SEQ_Q", graph_batch),
                    lambda: torch.empty_like(actual_seq_q),
                )
                self._mla_actual_seq_kv = get_execution_buffer(
                    ("MLA_ACTUAL_SEQ_KV", graph_batch),
                    lambda: torch.empty_like(actual_seq_kv),
                )
                self._mla_actual_seq_q.copy_(actual_seq_q)
                self._mla_actual_seq_kv.copy_(actual_seq_kv)
            else:
                self._mla_actual_seq_q = actual_seq_q
                self._mla_actual_seq_kv = actual_seq_kv
            # Dense (FIA v2) MLA needs host cumulative seq-lens; sparse SFA
            # does not, so skip the D2H unless the dense path can run.
            if self.requires_host_kv_lengths:
                if metadata.is_prefill or metadata.is_chunked_prefill:
                    self._mla_actual_seq_q_host = actual_seq_q.cpu().tolist()
                else:
                    self._mla_actual_seq_q_host = list(range(1, int(actual_seq_kv.numel()) + 1))
                self._mla_actual_seq_kv_host = actual_seq_kv.cpu().tolist()
            else:
                self._mla_actual_seq_q_host = None
                self._mla_actual_seq_kv_host = None
            if metadata.is_prefill or metadata.is_chunked_prefill:
                q_seq_lens = getattr(metadata, "q_seq_lens", None)
                if q_seq_lens is not None and q_seq_lens.numel() > 0:
                    self._mla_max_seqlen_q = int(q_seq_lens.max().item())
                else:
                    seq_starts = torch.cat([actual_seq_q.new_zeros(1), actual_seq_q[:-1]])
                    self._mla_max_seqlen_q = int((actual_seq_q - seq_starts).max().item())
            else:
                self._mla_max_seqlen_q = 1
            if graph_mode and self._block_table_i32 is not None:
                # QuantLightningIndexer metadata is captured into the ACL
                # graph. Python scalar arguments are fixed at capture time,
                # while the device KV lengths continue to grow on replay.
                # Use the static graph block-table capacity as a safe bound so
                # the captured tiling metadata remains valid for every replay.
                self._mla_max_seqlen_k = _mla_graph_max_seqlen_k(
                    self._block_table_i32,
                    self.page_size,
                )
            else:
                self._mla_max_seqlen_k = int(actual_seq_kv.max().item())
        else:
            self._mla_actual_seq_q = None
            self._mla_actual_seq_kv = None
            self._mla_max_seqlen_q = 0
            self._mla_max_seqlen_k = 0

        self._prepare_kv_shard_materialization(metadata)

    def _prepare_kv_shard_materialization(self, metadata: AttentionMetadata) -> None:
        self._kv_owner_representatives = None
        self._materialized_block_table = None
        self._sfa_page_layout = None
        if not self._is_mla or not (metadata.is_prefill or metadata.is_chunked_prefill):
            return
        if not metadata.has_kv_shard or metadata.kv_split_size <= 1:
            return
        if self._block_table_i32 is None:
            raise RuntimeError("sharded MLA prefill requires a block table")
        cp_size = distributed.cp_world_size(self.device)
        if cp_size <= 1 or cp_size % metadata.kv_split_size:
            raise RuntimeError("KV split must be a positive divisor of the active CP group")

        local_owner = torch.tensor([metadata.kv_split_rank], dtype=torch.int64, device=self.device)
        owner_by_cp_rank = distributed.all_gather(local_owner, 0, cp_size, "cp")
        if torch.any((owner_by_cp_rank < 0) | (owner_by_cp_rank >= metadata.kv_split_size)).item():
            raise RuntimeError("KV split rank must be within the active KV split")
        expected_replicas = cp_size // metadata.kv_split_size
        owner_counts = torch.bincount(owner_by_cp_rank, minlength=metadata.kv_split_size)
        expected_counts = torch.full_like(owner_counts, expected_replicas)
        if owner_counts.numel() != metadata.kv_split_size or not torch.equal(owner_counts, expected_counts):
            raise RuntimeError("KV owner distribution does not match the active CP/KV topology")
        representatives = [
            torch.argmax((owner_by_cp_rank == owner).to(torch.int64)) for owner in range(metadata.kv_split_size)
        ]
        self._kv_owner_representatives = torch.stack(representatives)

        block_table = self._block_table_i32
        entry_ids = torch.arange(
            block_table.numel(),
            dtype=block_table.dtype,
            device=block_table.device,
        ).view_as(block_table)
        owner_offsets = torch.arange(metadata.kv_split_size, dtype=block_table.dtype, device=block_table.device)
        expanded = entry_ids.unsqueeze(-1) * metadata.kv_split_size + owner_offsets
        expanded = torch.where(block_table.unsqueeze(-1) >= 0, expanded, torch.full_like(expanded, -1))
        self._materialized_block_table = expanded.flatten(1).contiguous()
        self._sfa_page_layout = _build_stable_sfa_page_layout(self._materialized_block_table)

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Attention,
    ) -> torch.Tensor:
        metadata = self._metadata
        assert metadata is not None

        layer_id = layer.layer_id
        layer_cache = self._kv_caches[layer_id]
        k_cache, v_cache = layer_cache.key, layer_cache.value
        if k_cache is None or v_cache is None:
            raise RuntimeError(f"KV cache is missing for layer {layer_id}")
        num_tokens = q.shape[0]

        k_3d = k.view(num_tokens, self.num_kv_heads, self.head_dim).contiguous()
        v_3d = v.view(num_tokens, self.num_kv_heads, self.head_dim).contiguous()
        q_3d = q.view(num_tokens, self.num_heads, self.head_dim).contiguous()

        # Context-Parallel prefill: q/k/v are this rank's sequence shard while
        # the slot_mapping/metadata still describe the full global sequence
        # (C++ does not pre-shard the Python path). All-gather K/V to the full
        # sequence, persist this rank's KV shard, and attend over its causal
        # prefix.
        cp_context = get_forward_context().cp_context
        if cp_context is not None:
            if not layer.causal:
                raise NotImplementedError("non-causal draft attention does not support context parallelism")
            if cp_context.has_prefix:
                raise NotImplementedError(
                    "non-MLA Python CP does not support chunked prefill with an existing KV prefix"
                )
            return self._prefill_cp(q_3d, k_3d, v_3d, metadata, cp_context, k_cache, v_cache)

        kernels.reshape_paged_cache(metadata.slot_mapping, k_3d, v_3d, k_cache, v_cache)

        if metadata.is_prefill or metadata.is_chunked_prefill:
            if self._use_expanded_decode:
                return self._decode(q_3d, k_cache, v_cache, metadata, num_tokens)
            return self._prefill(
                q_3d,
                k_3d,
                v_3d,
                k_cache,
                v_cache,
                metadata,
                num_tokens,
                layer.causal,
            )
        return self._decode(q_3d, k_cache, v_cache, metadata, num_tokens)

    def execute_mla(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor | None,
        k_latent_3d: torch.Tensor | None,
        k_pe_3d: torch.Tensor | None,
        layer: Attention,
        topk: torch.Tensor | None = None,
        cache_is_preprocessed: bool = False,
    ) -> torch.Tensor:
        """Absorbed-MLA attention. Returns [T, H, kv_lora]; caller bmm's W_UV."""
        metadata = self._metadata
        assert metadata is not None, "execute_mla called before prepare()"
        layer_id = layer.layer_id
        layer_cache = self._kv_caches[layer_id]
        # MLA reuses the K/V slots for the latent (nope) and rope caches. In the
        # SFA C8 packed layout ``key`` and ``value`` alias the same [num_blocks,
        # block_size, 1, 656] int8 tensor holding [nope | rope | scale] per
        # token, and only ``key`` is used.
        nope_cache, rope_cache = layer_cache.key, layer_cache.value
        if nope_cache is None:
            raise RuntimeError(f"MLA latent cache is missing for layer {layer_id}")
        if self._block_table_i32 is None:
            raise RuntimeError("MLA requires a block table")
        if self._mla_actual_seq_q is None or self._mla_actual_seq_kv is None:
            raise RuntimeError("MLA requires query and KV sequence lengths")

        # SFA C8 routes on both dtype and packed head-dim: a future int8 KV
        # path (for example a plain scale-slab quant) would still have head
        # dim == kv_lora, whereas the C8 packed row is kv_lora + 2 * rope +
        # 4 * (kv_lora // tile) bytes -- always strictly greater than
        # kv_lora. Comparing to q_latent avoids re-deriving kv_lora here.
        c8_enabled = nope_cache.dtype == torch.int8 and nope_cache.size(-1) > q_latent.size(-1)
        if c8_enabled and topk is None:
            raise RuntimeError(
                "SFA C8 packed KV cache only supports sparse (topk) MLA; "
                "dense MLA in this configuration has no C8 kernel."
            )

        cp_context = get_forward_context().cp_context
        if cp_context is None:
            # NoPE (qk_rope_head_dim==0): skip rope cache write + pass None to SFA.
            # The rope/value slot may be empty (a 0-dim tensor) or absent (None) in
            # NoPE models — it is never read, so do not require it.
            rope_dim = getattr(layer, "qk_rope_head_dim", None)
            if rope_dim and rope_dim > 0:
                # RoPE MLA (DeepSeek-V3/V4, GLM-5.2): latent + rotary.
                if rope_cache is None:
                    raise RuntimeError(f"MLA rope cache is missing for layer {layer_id} (qk_rope_head_dim={rope_dim})")
                if not cache_is_preprocessed:
                    if k_latent_3d is None or k_pe_3d is None:
                        raise RuntimeError("MLA cache inputs are required")
                    if c8_enabled:
                        self._write_mla_packed_c8_cache(
                            metadata.slot_mapping,
                            k_latent_3d,
                            k_pe_3d,
                            nope_cache,
                        )
                    else:
                        write_mla_paged_cache(
                            metadata.slot_mapping,
                            k_latent_3d,
                            k_pe_3d,
                            nope_cache,
                            rope_cache,
                        )
                # Dense absorbed MLA (indexer disabled, topk is None): fall back to
                # FIA v2 full attention over the paged latent cache. This is the
                # mainline path used by DeepSeek-V3.2 when index_topk == 0; keep it
                # alongside the sparse/KDA path so the non-sparse MLA config still
                # works. Only models with rope (qk_rope_head_dim > 0) reach here.
                if topk is None:
                    return self._mla_dense_fia_v2(
                        q_latent,
                        q_pe,
                        nope_cache,
                        rope_cache,
                        self._block_table_i32,
                        layer_id,
                    )
                if c8_enabled:
                    return self._mla_sparse_c8(
                        q_latent,
                        q_pe,
                        nope_cache,
                        topk,
                        self._block_table_i32,
                        layer_id,
                        layer.scale,
                    )
                return self._mla_sparse(
                    q_latent,
                    q_pe,
                    nope_cache,
                    rope_cache,
                    topk,
                    self._block_table_i32,
                    self._mla_actual_seq_q,
                    self._mla_actual_seq_kv,
                    layer_id,
                )
            # NoPE path (GLM-5.3-Flash): latent only, no rope.
            if not cache_is_preprocessed:
                if k_latent_3d is None:
                    raise RuntimeError("MLA cache inputs are required")
                write_mla_paged_cache(
                    metadata.slot_mapping,
                    k_latent_3d,
                    k_pe_3d,
                    nope_cache,
                    rope_cache,
                )
            return self._mla_sparse(
                q_latent,
                None,
                nope_cache,
                None,
                topk,
                self._block_table_i32,
                self._mla_actual_seq_q,
                self._mla_actual_seq_kv,
                layer_id,
            )

        # CP prefill path (cp_context is not None) — RoPE MLA only.
        if cache_is_preprocessed:
            raise RuntimeError("CP prefill does not support preprocessed MLA cache inputs")
        if topk is None:
            raise RuntimeError("CP prefill requires sparse MLA index output")
        if k_latent_3d is None or k_pe_3d is None:
            raise RuntimeError("CP prefill requires MLA cache inputs")
        if c8_enabled:
            raise RuntimeError("CP prefill does not support SFA C8 packed KV cache")
        global_latent = cp_gather_kv(k_latent_3d, cp_context).contiguous()
        global_rope = cp_gather_kv(k_pe_3d, cp_context).contiguous()
        cache_slots = metadata.local_slot_mapping if metadata.has_kv_shard else metadata.slot_mapping
        assert cache_slots is not None
        torch.ops.xllm_ops.reshape_paged_cache(
            cache_slots,
            global_latent,
            global_rope,
            nope_cache,
            rope_cache,
        )
        attention_nope, block_table = self._materialize_cp_cache(nope_cache, metadata, cp_context)
        attention_rope, _ = self._materialize_cp_cache(rope_cache, metadata, cp_context)
        if cp_context.query_index.numel() == 0:
            return q_latent.new_zeros(q_latent.shape)
        if metadata.has_kv_shard and metadata.kv_split_size > 1:
            attention_nope, attention_rope, block_table = self._materialize_sfa_layout(
                attention_nope,
                attention_rope,
            )
        query_index = cp_context.query_index
        segment_sequences = cp_context.segment_seq_indices
        q_real = q_latent.index_select(0, query_index).contiguous()
        q_pe_real = q_pe.index_select(0, query_index).contiguous()
        topk_real = topk.index_select(0, query_index).contiguous()
        local_block_table = block_table.index_select(0, segment_sequences).contiguous()
        output = self._mla_sparse(
            q_real,
            q_pe_real,
            attention_nope,
            attention_rope,
            topk_real,
            local_block_table,
            cp_context.q_cu_seqlens_tensor,
            cp_context.segment_kv_seq_lens_tensor,
            layer_id,
        )
        local_output = q_latent.new_zeros(q_latent.shape)
        local_output.index_copy_(0, query_index, output)
        return local_output

    # SFA C8 packed-row tile size. Must stay in sync with the C++
    # `MlaPackedC8Layout` struct in kv_cache_shape.h.
    _MLA_PACKED_C8_TILE_SIZE = 128

    def _write_mla_packed_c8_cache(
        self,
        slot_mapping: torch.Tensor,
        k_latent_3d: torch.Tensor,
        k_pe_3d: torch.Tensor,
        packed_cache: torch.Tensor,
    ) -> None:
        """RMSNorm/RoPE outputs → packed [int8 nope | bf16 rope | fp32 scale].

        Padding-row semantics: the BF16 write path uses
        ``xllm_ops.reshape_paged_cache`` whose kernel skips ``slot_id < 0``
        (see xllm/core/kernels/cuda/reshape_paged_cache.cu). This path uses
        ``kernels.scatter_nd_update``, which has no such skip. That is safe on
        NPU ACL-graph decode because ``GraphPersistentParam::update`` zeros
        the padded tail of both ``persistent_new_cache_slots_`` and
        ``persistent_block_tables_`` via ``zero_tensor_tail`` (see
        xllm/core/runtime/acl_graph_persistent_param.cpp), so padding tokens
        write to slot 0 of block 0, which ``BlockManagerImpl`` reserves as a
        sink and never allocates to a real sequence (see
        xllm/core/framework/block/block_manager_impl.h). Non-NPU / non-
        acl_graph writers pad slot_mapping with -1, but this method is only
        reachable via the ``glm_moe_dsa`` C8 gate which is NPU-only.
        """
        # k_pe elements are bit-reinterpreted (``.view(torch.int8)``) into the
        # int8 packed row and later decoded on the read side as bf16 by the
        # sparse-flash-attention kernel. Any other 16-bit dtype (fp16, etc.)
        # shares the byte width but has different exponent/mantissa
        # partitioning, so the values would decode as garbage without any
        # sizing tripwire firing. Check the exact dtype rather than only the
        # element size, and ``raise`` (not ``assert``) so ``python -O`` cannot
        # strip the check.
        if k_pe_3d.dtype != torch.bfloat16:
            raise RuntimeError(
                f"SFA C8 packed rope path expects bf16 rope, got dtype={k_pe_3d.dtype} ({k_pe_3d.element_size()} B)."
            )
        num_tokens = k_latent_3d.size(0)
        kv_lora = k_latent_3d.size(-1)
        rope_dim = k_pe_3d.size(-1)
        self._verify_packed_c8_layout(packed_cache, kv_lora, rope_dim)
        nope_view = k_latent_3d.contiguous().view(-1, 1, kv_lora)
        k_nope_i8, k_scale_fp32 = kernels.dynamic_block_quant(
            nope_view,
            dst_type=torch.int8,
            row_block_size=1,
            col_block_size=self._MLA_PACKED_C8_TILE_SIZE,
        )
        # Bit-reinterpret (never numeric-cast) rope and scale into the int8
        # byte layout the packed cache row expects. dynamic_block_quant already
        # returns fp32 scales, so view(int8) collapses each element to 4 bytes
        # without any preceding cast; k_pe_3d is bf16 (checked above), so its
        # view(int8) doubles the last dim to 2 * rope_dim bytes.
        k_nope_i8 = k_nope_i8.reshape(num_tokens, kv_lora)
        k_rope_i8 = k_pe_3d.reshape(num_tokens, rope_dim).view(torch.int8)
        k_scale_i8 = k_scale_fp32.reshape(num_tokens, -1).view(torch.int8)
        packed = torch.cat([k_nope_i8, k_rope_i8, k_scale_i8], dim=-1)
        packed_flat = packed_cache.view(-1, packed_cache.size(-1))
        indices = slot_mapping.reshape(-1, 1).to(torch.int32)
        kernels.scatter_nd_update(packed_flat, indices, packed)

    def _verify_packed_c8_layout(
        self,
        packed_cache: torch.Tensor,
        kv_lora: int,
        rope_dim: int,
    ) -> None:
        """Cross-language layout check against the C++ ``MlaPackedC8Layout``
        struct (kv_cache_shape.h). bf16 rope contributes 2 bytes/elem, fp32
        per-tile scale contributes 4 bytes/elem.
        """
        cache_head_dim = packed_cache.size(-1)
        expected = kv_lora + 2 * rope_dim + 4 * (kv_lora // self._MLA_PACKED_C8_TILE_SIZE)
        if cache_head_dim != expected:
            raise RuntimeError(
                f"SFA C8 packed cache trailing dim mismatch: cache has "
                f"{cache_head_dim} bytes/row, expected {expected} "
                f"(kv_lora={kv_lora}, rope_dim={rope_dim}, "
                f"tile={self._MLA_PACKED_C8_TILE_SIZE}). Python and C++ "
                f"MlaPackedC8Layout must stay in lockstep."
            )

    def mla_preprocess_context(
        self,
        layer: Attention,
    ) -> MlaPreprocessContext | None:
        metadata = self._metadata
        if metadata is None or metadata.is_prefill or metadata.is_chunked_prefill:
            return None
        layer_cache = self._kv_caches[layer.layer_id]
        kv_cache, rope_cache = layer_cache.key, layer_cache.value
        if kv_cache is None or rope_cache is None:
            raise RuntimeError(f"MLA latent cache is missing for layer {layer.layer_id}")
        return MlaPreprocessContext(
            kv_cache=kv_cache,
            rope_cache=rope_cache,
            slot_mapping=metadata.slot_mapping,
        )

    def mla_index_context(self, layer: Attention) -> MlaIndexContext:
        metadata = self._metadata
        assert metadata is not None, "mla_index_context called before prepare()"
        assert self._block_table_i32 is not None
        assert self._mla_actual_seq_q is not None
        assert self._mla_actual_seq_kv is not None
        layer_cache = self._kv_caches[layer.layer_id]
        index_cache = layer_cache.index
        if index_cache is None:
            raise RuntimeError(f"MLA index cache is missing for layer {layer.layer_id}")
        index_cache_scale = layer_cache.index_scale
        slot_mapping = metadata.local_slot_mapping if metadata.has_kv_shard else metadata.slot_mapping
        if slot_mapping is None:
            raise RuntimeError("MLA index cache requires a slot mapping")
        return MlaIndexContext(
            index_cache=index_cache,
            slot_mapping=slot_mapping,
            # Optional by contract: the indexer skips its pool path without a table.
            block_table=self.indexer_block_table_or_none(),
            actual_seq_q=self._mla_actual_seq_q,
            actual_seq_kv=self._mla_actual_seq_kv,
            index_cache_scale=index_cache_scale,
            get_quant_indexer_metadata=lambda num_heads_q,
            head_dim,
            sparse_count,
            cmp_ratio: self._get_quant_indexer_metadata(
                num_heads_q,
                index_cache.size(2),
                head_dim,
                sparse_count,
                cmp_ratio,
            ),
            update_index_cache=lambda values, scales: self._update_mla_index_cache(
                index_cache,
                index_cache_scale,
                slot_mapping,
                values,
                scales,
            ),
            materialize_index_cache=lambda: self._materialize_mla_index_cache(
                index_cache,
                index_cache_scale,
                metadata,
                get_forward_context().cp_context,
            ),
            cp_context=get_forward_context().cp_context,
        )

    def _get_quant_indexer_metadata(
        self,
        num_heads_q: int,
        num_heads_k: int,
        head_dim: int,
        sparse_count: int,
        cmp_ratio: int,
    ) -> torch.Tensor:
        assert self._mla_actual_seq_q is not None
        assert self._mla_actual_seq_kv is not None
        cache_key = (num_heads_q, head_dim, sparse_count, cmp_ratio)
        metadata = self._mla_quant_indexer_metadata.get(cache_key)
        if metadata is None:
            metadata = kernels.quant_lightning_indexer_metadata(
                num_heads_q,
                num_heads_k,
                head_dim,
                self._mla_actual_seq_q,
                self._mla_actual_seq_kv,
                self._mla_max_seqlen_q,
                self._mla_max_seqlen_k,
                sparse_count,
                cmp_ratio,
            )
            self._mla_quant_indexer_metadata[cache_key] = metadata
        return metadata

    @staticmethod
    def _update_mla_index_cache(
        index_cache: torch.Tensor,
        index_cache_scale: torch.Tensor | None,
        slot_mapping: torch.Tensor,
        values: torch.Tensor,
        scales: torch.Tensor | None,
    ) -> None:
        cache_view = index_cache.view(-1, index_cache.size(-1))
        scatter_indices = slot_mapping.reshape(-1, 1).clamp_min(0)
        kernels.scatter_nd_update(
            cache_view,
            scatter_indices,
            values,
        )
        if index_cache_scale is not None and scales is not None:
            scale_view = index_cache_scale.view(-1, index_cache_scale.size(-1))
            kernels.scatter_nd_update(scale_view, scatter_indices, scales)

    def _materialize_cp_cache(
        self,
        cache: torch.Tensor,
        metadata: AttentionMetadata,
        cp_context: CpContext | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cp_context is None or not metadata.has_kv_shard or metadata.kv_split_size <= 1:
            assert self._block_table_i32 is not None
            return cache, self._block_table_i32
        if self._kv_owner_representatives is None or self._materialized_block_table is None:
            raise RuntimeError("KV shard materialization was not prepared")
        assert self._block_table_i32 is not None
        flat_blocks = self._block_table_i32.reshape(-1)
        safe_blocks = flat_blocks.clamp_min(0).to(torch.int64)
        local_blocks = cache.index_select(0, safe_blocks)
        gathered = distributed.all_gather(local_blocks, 0, cp_context.cp_size, "cp")
        gathered = gathered.view(cp_context.cp_size, flat_blocks.numel(), *cache.shape[1:])
        owner_blocks = gathered.index_select(0, self._kv_owner_representatives)
        order = [1, 0, *range(2, owner_blocks.dim())]
        materialized = owner_blocks.permute(order).reshape(
            flat_blocks.numel() * metadata.kv_split_size,
            *cache.shape[1:],
        )
        return materialized.contiguous(), self._materialized_block_table

    def _materialize_mla_index_cache(
        self,
        index_cache: torch.Tensor,
        index_cache_scale: torch.Tensor | None,
        metadata: AttentionMetadata,
        cp_context: CpContext | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        materialized_cache, block_table = self._materialize_cp_cache(
            index_cache,
            metadata,
            cp_context,
        )
        materialized_scale = None
        if index_cache_scale is not None:
            materialized_scale, _ = self._materialize_cp_cache(
                index_cache_scale,
                metadata,
                cp_context,
            )
        return materialized_cache, materialized_scale, block_table

    def _materialize_sfa_layout(
        self,
        nope_cache: torch.Tensor,
        rope_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layout = self._sfa_page_layout
        if layout is None:
            raise RuntimeError("SFA cache layout was not prepared")
        target_nope = nope_cache.new_zeros((layout.page_count, *nope_cache.shape[1:]))
        target_rope = rope_cache.new_zeros((layout.page_count, *rope_cache.shape[1:]))
        target_nope.index_copy_(
            0,
            layout.target_page_ids,
            nope_cache.index_select(0, layout.source_page_ids),
        )
        target_rope.index_copy_(
            0,
            layout.target_page_ids,
            rope_cache.index_select(0, layout.source_page_ids),
        )
        return target_nope, target_rope, layout.block_table

    def gather_index_history(
        self,
        layer: Attention,
        batch_size: int,
    ) -> torch.Tensor:
        """Gather the kPool packed history into a dense ``[B, kv_len, W]`` tensor.

        The index cache is paged (``[blocks, block_size, 1, W]``); the kPool
        indexer's ``select_topk`` needs the per-sequence rows contiguous. For
        each sequence we walk its block table, concatenating ``block_size``
        rows per block (the last block yields only ``last_page_len``).
        ``kv_len`` is padded to the max across the batch; out-of-range rows
        are zeroed so ``valid`` channels read as false downstream.
        """
        metadata = self._metadata
        assert metadata is not None, "gather_index_history called before prepare()"
        index_cache = self._kv_caches[layer.layer_id].index
        assert index_cache is not None, "gather_index_history requires a paged index cache"
        block_table = self.indexer_block_table()
        block_size = index_cache.shape[1]
        width = index_cache.shape[3]
        device = index_cache.device

        # batch_size is hidden_states.shape[0] which the engine flattens to 1
        # for multi-sequence batches; the real sequence count is the block
        # table's first dim. Use it to gather every sequence's history.
        num_seqs = block_table.shape[0] if block_table is not None else batch_size

        if in_acl_graph():
            # Graph branch: fixed shapes only (no .item()/host sync). Gather
            # the block table in one vectorized index_select up to a static
            # max_kv (replay-stable; capped by graph_index_history_max_kv —
            # the runner falls back to eager beyond it). Rows past each
            # sequence's live length are zeroed by an explicit kv_seq_lens
            # mask: the valid channel alone cannot be trusted because a
            # recycled block still carries a previous owner's valid=1 rows,
            # and a padded block-table column points at block 0.
            kv_lens_dev = metadata.kv_seq_lens
            if kv_lens_dev is None:
                raise RuntimeError("gather_index_history graph mode needs device kv_seq_lens")
            max_kv = min(
                block_table.shape[1] * block_size,
                self.graph_index_history_max_kv,
            )
            num_blocks = (max_kv + block_size - 1) // block_size
            out = get_execution_buffer(
                ("KPOOL_INDEX_HISTORY", num_seqs, max_kv, width),
                lambda: torch.empty(
                    num_seqs,
                    max_kv,
                    width,
                    dtype=index_cache.dtype,
                    device=device,
                ),
            )
            flat = index_cache.view(-1, width)
            bt = block_table[:num_seqs, :num_blocks].to(torch.int64)
            block_offsets = torch.arange(block_size, device=device)
            slot_ids = (bt[:, :, None] * block_size + block_offsets[None, None, :]).reshape(num_seqs, max_kv)
            gathered = flat.index_select(0, slot_ids.reshape(-1)).view(num_seqs, max_kv, width)
            row_valid = torch.arange(max_kv, device=device)[None, :] < kv_lens_dev[:num_seqs].to(torch.int64)[:, None]
            torch.mul(gathered, row_valid[:, :, None].to(out.dtype), out=out)
            return out

        kv_seq_lens = metadata.kv_seq_lens
        if kv_seq_lens is not None:
            kv_lens = kv_seq_lens[:num_seqs].to(torch.int64)
        else:
            kv_host = getattr(metadata, "kv_seq_lens_host", None)
            if kv_host is not None:
                kl = kv_host.cpu()
                if kl.numel() == num_seqs + 1:
                    kv_lens = (kl[1:] - kl[:-1]).to(torch.int64)
                else:
                    kv_lens = kl[:num_seqs].to(torch.int64)
            else:
                raise RuntimeError("gather_index_history needs kv_seq_lens")

        max_kv = int(kv_lens.max().item()) if num_seqs > 0 else 0
        flat = index_cache.view(-1, width)
        bt = block_table[:num_seqs].to(torch.int64)
        out = torch.zeros(num_seqs, max_kv, width, dtype=index_cache.dtype, device=device)
        for b in range(num_seqs):
            kl = int(kv_lens[b].item())
            if kl == 0:
                continue
            n_full = kl // block_size
            tail = kl - n_full * block_size
            rows = []
            blk = bt[b]
            if n_full > 0:
                slot_ids = (blk[:n_full, None] * block_size + torch.arange(block_size, device=device)[None, :]).reshape(
                    -1
                )
                rows.append(flat.index_select(0, slot_ids))
            if tail > 0:
                last_blk = int(blk[n_full].item())
                slot_ids = last_blk * block_size + torch.arange(tail, device=device)
                rows.append(flat.index_select(0, slot_ids))
            packed = torch.cat(rows, dim=0) if len(rows) > 1 else rows[0]
            out[b, :kl] = packed
        return out

    def _spec_verify_v3(
        self,
        mixed_qkv: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        layer: Attention,
        idx: torch.Tensor,
        metadata,
        conv_cache,
        ssm_cache,
        recurrent_kda,
    ) -> torch.Tensor:
        """Fused multi-slot MTP spec-verify / plain-step path for KDA layers.

        The default KDA verify path uses the fused in-kernel-spec contract:
        a persistent per-layer combined ``[base | draft]`` state pool
        and a single ``recurrent_kda`` call per layer that advances BOTH the
        confirmed (base) and draft (draft) tokens in one multi-token pass,
        writing each token's resulting state to its own slot so both outcomes
        survive to the next step (no stash, no host selection, no per-tap conv
        decomposition of the recurrent state).

        Correctness invariants (see docs/mtp_graph_verify_design.md / B8):
        - The fla_npu ``aclnnRecurrentKda`` kernel writes each token ``seq_i``'s
          state to ``ssm_state_indices[seq_i]`` (per-token-slot writeback,
          recurrent_kda.h CopyOutState). With
          ``ssm_state_indices = [base, base+N]`` (1D packed per seq), processing
          ``[b, d]`` writes after-b -> base slot, after-d -> draft slot; after-b
          is preserved so rejection (next-step num_accepted=1) resumes from it.
        - ``num_accepted_tokens=1`` always resumes from the base slot, which
          holds the *selected* running state (after-b from a rejection, or
          after-d copied base<-draft when the previous draft was accepted).
          The selection uses device-side slot gathers independently of the
          current step's width, not a host branch.
        - Pools use the configured maximum decoding width so changing widths
          never reallocates storage referenced by captured graphs. Cumulative
          lengths are cached by both sequence count and current width.
        - The C++ conv/ssm pools remain the source of truth for plain/prefill
          steps: at verify entry the committed running state is copied into
          the combined base region; at exit after-b (always accepted) is
          committed back, so a plain step sees the correct state.
        - Conv state is handled the same dual-slot way with the combined conv
          pool; the conv itself reuses the proven bit-exact per-tap mul-add
          (``_causal_conv1d_graph_multi``) — graph-capturable, no aclop conv.
        """
        device = mixed_qkv.device
        num_seqs = idx.shape[0]
        rows_per_seq = mixed_qkv.shape[2] // num_seqs  # 2 verify, 1 plain
        if not 1 <= rows_per_seq <= self._kda_verify_width:
            raise ValueError(
                f"KDA verify width {rows_per_seq} exceeds configured decoding width {self._kda_verify_width}"
            )
        head_dim = layer.head_dim
        nh = layer.num_heads_local
        qkv_dim = layer.qkv_dim
        conv_dim = layer.conv_dim
        conv_state_len = layer.conv_kernel_size - 1
        scale = 1.0 / (head_dim**0.5)
        conv_weight = layer.conv1d.weight.squeeze(1)
        in_graph = in_acl_graph()
        nslots = conv_cache.shape[0]  # C++ pool capacity

        st = self.__dict__.setdefault("_kda_v3", {}).setdefault(layer.layer_id, {})
        if "armed_buf" not in st:
            pool_slots = self._kda_verify_width * nslots
            st["combined_conv"] = torch.zeros(
                pool_slots, conv_state_len, conv_dim, dtype=conv_cache.dtype, device=device
            )
            st["combined_ssm"] = torch.zeros(pool_slots, nh, head_dim, head_dim, dtype=ssm_cache.dtype, device=device)
            st["kv_prev"] = torch.zeros(nslots, dtype=torch.int64, device=device)
            st["armed_buf"] = torch.zeros(nslots, dtype=torch.bool, device=device)
        combined_conv = st["combined_conv"]
        combined_ssm = st["combined_ssm"]
        kv_prev = st["kv_prev"]
        armed_buf = st["armed_buf"]

        # ---- per-step m (previous accepted count = kv growth) ----
        # (base_now, m) is recomputed from the live kv_seq_lens on every call
        # and must NOT be cached across steps: the scheduler reuses the same
        # kv_seq_lens host buffer, so a (data_ptr, num_seqs) key repeats every
        # step while the per-seq lengths GROW — a cached (base_now, m) would
        # mis-select the conv/ssm boundary slot and diverge output under
        # temp=0 + HCCL_DETERMINISTIC. The cost is a handful of cheap host->dev
        # + index_select ops per step.
        expanded = resolve_expanded_decode_metadata(metadata)
        kv_src = expanded.kv_seq_lens if expanded is not None else metadata.kv_seq_lens
        kv_rows = kv_src.to(device=device, dtype=torch.int64)
        base_now = kv_rows.view(num_seqs, -1)[:, 0].contiguous()
        armed_h = armed_buf.index_select(0, idx)
        kv_prev_h = kv_prev.index_select(0, idx)
        accepted_counts = torch.where(
            armed_h,
            (base_now - kv_prev_h).clamp(min=1, max=self._kda_verify_width),
            torch.ones_like(base_now),
        )

        idx64 = idx if idx.dtype == torch.int64 else idx.to(torch.int64)
        idx32 = idx64.to(torch.int32)

        # ---- 1) committed running state (C++ pool) -> combined base ----
        combined_conv[:nslots].index_copy_(0, idx64, conv_cache.index_select(0, idx64))
        combined_ssm[:nslots].index_copy_(0, idx64, ssm_cache.index_select(0, idx64))

        slot_offsets = torch.arange(rows_per_seq, dtype=torch.int32, device=device) * nslots
        ssm_state_indices = (idx32.view(-1, 1) + slot_offsets.view(1, -1)).reshape(-1)
        qsl_key = (num_seqs, rows_per_seq)
        qsl_buf = st.setdefault("qsl_buf", {}).get(qsl_key)
        if qsl_buf is None:
            qsl_buf = torch.arange(num_seqs + 1, dtype=torch.int32, device=device) * rows_per_seq
            st["qsl_buf"][qsl_key] = qsl_buf

        # ---- 2) conv-boundary select (slot m-1: prev last-accepted state) ----
        # Boundary = running conv state after the previous step's last accepted
        # token = slot (m-1) per seq (0=base/reject, 1=draft0 accept, ...,
        # R-1=all-accepted). Generalizes the legacy 2-way where(m2, draft, base).
        boundary_slot = idx64 + (accepted_counts - 1) * nslots
        sel_conv = combined_conv.index_select(0, boundary_slot)  # [S, Ks, C]
        combined_ssm.index_copy_(0, idx64, combined_ssm.index_select(0, boundary_slot))

        # ---- 3) conv (per-tap, bit-exact, graph-capturable) ----
        cache_boundary = sel_conv.transpose(1, 2).contiguous()  # [S, C, Ks]
        x = mixed_qkv.reshape(conv_dim, num_seqs, rows_per_seq).permute(1, 0, 2).contiguous()
        cin = torch.cat([cache_boundary.to(x.dtype), x], dim=-1)
        if in_graph:
            conv_out = _causal_conv1d_graph_multi(cin, conv_weight, rows_per_seq, layer.activation)
        else:
            _cin_c = cin.to(conv_weight.dtype).contiguous()
            _cw = conv_weight.unsqueeze(1).contiguous()
            conv_out = torch.nn.functional.conv1d(
                _cin_c,
                _cw,
                bias=None,
                padding=conv_state_len,
                groups=conv_dim,
            )[..., conv_state_len : conv_state_len + rows_per_seq]
            if layer.activation == "silu":
                conv_out = torch.nn.functional.silu(conv_out)
            conv_out = conv_out.to(x.dtype)
        # multi-tail conv_state: slot j (0=base ... R-1=last draft) <- window
        # ending after token j. For R==2 this is base<-tail_b / draft1<-tail_full
        # (== legacy dual-tail); for R==1 only base is written (a plain step's
        # next verify has m=1 -> base, so draft slots are never read).
        for j in range(rows_per_seq):
            tail_j = cin[..., (j + 1) : (j + 1) + conv_state_len].transpose(1, 2).contiguous()
            combined_conv.index_copy_(0, idx64 + j * nslots, tail_j)

        # ---- 4) split conv_out -> q/k/v; gate/beta -> g/b (TND, [T,nh,hd]) ----
        c_split = conv_out.transpose(1, 2).split(qkv_dim, dim=-1)
        q = c_split[0].reshape(-1, nh, head_dim).to(torch.bfloat16)
        k = c_split[1].reshape(-1, nh, head_dim).to(torch.bfloat16)
        v = c_split[2].reshape(-1, nh, head_dim).to(torch.bfloat16)
        if gate.dim() == 3:
            gate = gate.view(num_seqs, rows_per_seq, nh * head_dim)
        cur_g = gate.view(num_seqs, rows_per_seq, nh, head_dim)
        cur_b = beta.view(num_seqs, rows_per_seq, nh)
        g_flat = cur_g.reshape(-1, nh, head_dim).to(torch.float32)
        b_flat = cur_b.reshape(-1, nh).to(torch.float32)

        # ---- 5) fused multi-slot recurrent: advance [b, d0, ..., d{R-1}] ----
        ret = recurrent_kda(
            q,
            k,
            v,
            g_flat,
            b_flat,
            initial_state=combined_ssm,
            cu_seqlens=qsl_buf,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=torch.ones_like(idx32),
            layout="TND",
            scale=scale,
            output_final_state=True,
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=False,
            use_beta_sigmoid_in_kernel=False,
            state_v_first=True,
        )
        core_out = ret[0] if isinstance(ret, tuple) else ret

        # ---- 7) commit after-b (always accepted) -> C++ source of truth ----
        conv_cache.index_copy_(0, idx64, combined_conv[:nslots].index_select(0, idx64))
        ssm_cache.index_copy_(0, idx64, combined_ssm[:nslots].index_select(0, idx64))

        # ---- 8) bookkeeping for next step's m ----
        kv_prev.index_copy_(0, idx64, base_now)
        armed_buf.index_fill_(0, idx64, True)
        return core_out.view(1, num_seqs * rows_per_seq, nh, head_dim)

    def _mla_sparse(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor | None,
        nope_cache: torch.Tensor,
        rope_cache: torch.Tensor | None,
        topk: torch.Tensor,
        block_table: torch.Tensor,
        actual_seq_q: torch.Tensor,
        actual_seq_kv: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        if actual_seq_q is None:
            actual_seq_q = self._mla_actual_seq_q
        if actual_seq_kv is None:
            actual_seq_kv = self._mla_actual_seq_kv
        out = get_execution_buffer(
            ("SFA_OUTPUT", layer_id) + tuple(q_latent.shape),
            lambda: torch.empty_like(q_latent),
        )
        return kernels.sparse_flash_attention_out(
            q_latent,
            nope_cache,
            nope_cache,
            topk,
            block_table,
            actual_seq_q,
            actual_seq_kv,
            q_pe,
            rope_cache,
            self.scale,
            1,
            "TND",
            "PA_BSND",
            3,
            out,
        )  # [T, H, kv_lora]

    def _mla_sparse_c8(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        packed_cache: torch.Tensor,
        topk: torch.Tensor,
        block_table: torch.Tensor,
        layer_id: int,
        scale: float,
    ) -> torch.Tensor:
        """Sparse MLA over a packed C8 KV cache (K==V, no separate rope tensor).

        Query is composed as ``cat([q_latent, q_pe], dim=-1)`` and handed to
        ``npu_kv_quant_sparse_flash_attention`` which reads the fp32 dequant
        scales embedded in each 656-byte packed row on the fly.

        ``scale`` is the caller-provided per-layer softmax scale
        (``layer.scale``), so an MTP draft head or a hypothetical
        heterogeneous head_dim layer cannot silently reuse the target's
        backend-level ``self.scale``.
        """
        del layer_id  # kept for parity with _mla_sparse's signature
        # ``torch.cat`` along the last dim of contiguous inputs already returns
        # a contiguous tensor; no trailing ``.contiguous()`` needed.
        query = torch.cat([q_latent, q_pe], dim=-1)
        return kernels.kv_quant_sparse_flash_attention(
            query=query,
            key=packed_cache,
            value=packed_cache,
            sparse_indices=topk,
            block_table=block_table,
            actual_seq_lengths_query=self._mla_actual_seq_q,
            actual_seq_lengths_kv=self._mla_actual_seq_kv,
            scale_value=scale,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
            quant_scale_repo_mode=1,
            tile_size=self._MLA_PACKED_C8_TILE_SIZE,
            key_quant_mode=2,
            value_quant_mode=2,
            rope_head_dim=q_pe.shape[-1],
        )  # [T, H, kv_lora]

    def _mla_dense_fia_v2_out(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        nope_cache: torch.Tensor,
        rope_cache: torch.Tensor,
        block_table: torch.Tensor,
        workspace: torch.Tensor,
        output: torch.Tensor,
        softmax_lse: torch.Tensor,
    ) -> None:
        if self._mla_actual_seq_q_host is None:
            raise RuntimeError("dense MLA requires query sequence lengths")
        if self._mla_actual_seq_kv_host is None:
            raise RuntimeError("dense MLA requires KV sequence lengths")
        block_size = nope_cache.size(1)
        nope_flat = nope_cache.view(nope_cache.size(0), block_size, -1)
        rope_flat = rope_cache.view(rope_cache.size(0), block_size, -1)
        is_prefill = bool(
            self._metadata is not None and (self._metadata.is_prefill or self._metadata.is_chunked_prefill)
        )
        torch.ops.npu.npu_fused_infer_attention_score_v2.out(
            q_latent,
            nope_flat,
            nope_flat,
            query_rope=q_pe,
            key_rope=rope_flat,
            pse_shift=None,
            atten_mask=self._causal_mask if is_prefill else None,
            actual_seq_qlen=self._mla_actual_seq_q_host,
            actual_seq_kvlen=self._mla_actual_seq_kv_host,
            block_table=block_table,
            num_query_heads=self.num_heads,
            num_key_value_heads=1,
            softmax_scale=self.scale,
            input_layout="TND",
            sparse_mode=(_SPARSE_MODE_RIGHT_DOWN_CAUSAL if is_prefill else _SPARSE_MODE_NONE),
            block_size=block_size,
            return_softmax_lse=False,
            workspace=workspace,
            out=[output, softmax_lse],
        )

    def _mla_dense_fia_v2(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        nope_cache: torch.Tensor,
        rope_cache: torch.Tensor,
        block_table: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Run dense absorbed MLA with FIA v2 and separate RoPE caches."""
        if not self._use_fia_v2:
            raise RuntimeError("dense MLA requires FIA v2 support")
        if self._mla_actual_seq_q_host is None:
            raise RuntimeError("dense MLA requires query sequence lengths")
        if self._mla_actual_seq_kv_host is None:
            raise RuntimeError("dense MLA requires KV sequence lengths")

        block_size = nope_cache.size(1)
        nope_flat = nope_cache.view(nope_cache.size(0), block_size, -1)
        rope_flat = rope_cache.view(rope_cache.size(0), block_size, -1)
        is_prefill = bool(
            self._metadata is not None and (self._metadata.is_prefill or self._metadata.is_chunked_prefill)
        )
        common_kwargs = {
            "query_rope": q_pe,
            "key_rope": rope_flat,
            "pse_shift": None,
            "atten_mask": self._causal_mask if is_prefill else None,
            "actual_seq_qlen": self._mla_actual_seq_q_host,
            "actual_seq_kvlen": self._mla_actual_seq_kv_host,
            "block_table": block_table,
            "num_query_heads": self.num_heads,
            "num_key_value_heads": 1,
            "softmax_scale": self.scale,
            "input_layout": "TND",
            "sparse_mode": (_SPARSE_MODE_RIGHT_DOWN_CAUSAL if is_prefill else _SPARSE_MODE_NONE),
            "block_size": block_size,
            "return_softmax_lse": False,
        }

        graph_context = get_forward_context().acl_graph
        if graph_context is None:
            output, _ = torch.ops.npu.npu_fused_infer_attention_score_v2(
                q_latent,
                nope_flat,
                nope_flat,
                **common_kwargs,
            )
            return output

        output_key = ("MLA_DENSE_OUTPUT", layer_id) + tuple(q_latent.shape)
        output = self._mla_graph_outputs.get(output_key)
        if output is None:
            output = torch.empty_like(q_latent)
            self._mla_graph_outputs[output_key] = output
            self._mla_graph_lses[output_key] = torch.empty(0, dtype=q_latent.dtype, device=q_latent.device)
        softmax_lse = self._mla_graph_lses[output_key]
        workspace = self._mla_graph_workspaces.get(output_key)
        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                q_latent,
                nope_flat,
                nope_flat,
                **common_kwargs,
            )
            self._mla_graph_workspaces[output_key] = workspace

        stream = graph_context.stream
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        torch.npu.graph_task_group_begin(stream)
        try:
            self._mla_dense_fia_v2_out(
                q_latent,
                q_pe,
                nope_cache,
                rope_cache,
                block_table,
                workspace,
                output,
                softmax_lse,
            )
        except Exception:
            torch.npu.graph_task_group_end(stream)
            raise
        handle = torch.npu.graph_task_group_end(stream)

        def _update_mla_fia_v2_args() -> None:
            self._mla_dense_fia_v2_out(
                q_latent,
                q_pe,
                nope_cache,
                rope_cache,
                block_table,
                workspace,
                output,
                softmax_lse,
            )

        graph_context.tasks.append(AclGraphTask(event, handle, _update_mla_fia_v2_args))
        return output

    # ------------------------------------------------------------------
    # Prefill: packed TND with causal mask
    # ------------------------------------------------------------------

    def _prefill(
        self,
        q_3d: torch.Tensor,
        k_3d: torch.Tensor,
        v_3d: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: AttentionMetadata,
        num_tokens: int,
        causal: bool,
    ) -> torch.Tensor:
        actual_seq = self._cumulative_seq_lens(metadata, num_tokens)
        atten_mask = self._causal_mask if causal else None
        sparse_mode = _SPARSE_MODE_RIGHT_DOWN_CAUSAL if causal else _SPARSE_MODE_NONE

        # Prefix-cache hit (or chunked prefill with prior context): part of the
        # KV already lives in the paged cache, so this forward only carries the
        # new tokens (q_len < kv_len). Attend over the full paged KV via
        # block_table, mirroring _decode. Without this, the new query tokens
        # would only see their own KV (actual_seq_lengths_kv == q_len) and never
        # the cached prefix, diverging from a full recompute.
        if metadata.block_table is not None:
            block_size = k_cache.size(1)
            k_flat = k_cache.view(k_cache.size(0), block_size, -1)
            v_flat = v_cache.view(v_cache.size(0), block_size, -1)
            output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_3d,
                k_flat,
                v_flat,
                pse_shift=None,
                atten_mask=atten_mask,
                block_table=self._block_table_i32,
                actual_seq_lengths=actual_seq,
                actual_seq_lengths_kv=self._actual_seq_kv,
                num_heads=self.num_heads,
                scale=self.scale,
                input_layout="TND",
                num_key_value_heads=self.num_kv_heads,
                block_size=block_size,
                sparse_mode=sparse_mode,
                softmax_lse_flag=False,
            )
            return output.reshape(num_tokens, self.num_heads * self.head_dim)

        output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_3d,
            k_3d,
            v_3d,
            pse_shift=None,
            atten_mask=atten_mask,
            actual_seq_lengths=actual_seq,
            actual_seq_lengths_kv=actual_seq,
            num_heads=self.num_heads,
            scale=self.scale,
            input_layout="TND",
            num_key_value_heads=self.num_kv_heads,
            sparse_mode=sparse_mode,
            softmax_lse_flag=False,
        )
        return output.reshape(num_tokens, self.num_heads * self.head_dim)

    # ------------------------------------------------------------------
    # Context-Parallel prefill: all-gather KV, attend over causal prefix
    # ------------------------------------------------------------------

    def _prefill_cp(
        self,
        q_3d: torch.Tensor,
        k_3d: torch.Tensor,
        v_3d: torch.Tensor,
        metadata: AttentionMetadata,
        cp_context: CpContext,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill attention for this rank's zigzag sequence shard.

        q/k/v hold this rank's ``total_local`` rows (two owned chunks per
        sequence, padding rows zeroed). We all-gather K/V back to the full
        global-order sequence, write the complete KV into the paged cache (so a
        later non-CP decode sees every position), then run one FIA over this
        rank's real queries. Each owned (sequence, half) segment is a packed
        sub-sequence: its ``real_count`` queries attend the causal prefix
        ``[0, segment_start + real_count)`` selected by ``kv_gather_index``.
        With ``sparse_mode=3`` (right-aligned causal) query row ``i`` of a
        segment attends KV ``[0, segment_start + i]`` — its exact global causal
        range. Segments are independent sub-sequences delimited by
        ``q_cu_seqlens`` / ``kv_cu_seqlens``, so both owned chunks resolve in a
        single call.
        """
        local_tokens = q_3d.shape[0]

        kv_global_k = cp_gather_kv(k_3d, cp_context)
        kv_global_v = cp_gather_kv(v_3d, cp_context)

        # Persist the full global-order KV into this rank's paged cache.
        kernels.reshape_paged_cache(
            metadata.slot_mapping,
            kv_global_k.contiguous(),
            kv_global_v.contiguous(),
            k_cache,
            v_cache,
        )

        # A CP rank can own only padding chunks when every sequence in the batch
        # is shorter than the zigzag chunk grid (e.g. a 1-token prompt with
        # cp_size > 1). It then has no real queries. The KV all-gather above
        # already ran (collectives must stay in lockstep across ranks) and the
        # full global KV is now in this rank's paged cache, so skip the FIA:
        # calling it with a 0-row query and empty actual_seq_lengths is rejected
        # by npu_fused_infer_attention. Return the all-zero shard directly.
        if cp_context.query_index.numel() == 0:
            return q_3d.new_zeros(local_tokens, self.num_heads * self.head_dim)

        # Real queries this rank owns, packed per (sequence, half) segment.
        q_real = q_3d.index_select(0, cp_context.query_index).contiguous()
        # Each segment's causal KV prefix, packed in the same segment order.
        kv_prefix_k = kv_global_k.index_select(0, cp_context.kv_gather_index).contiguous()
        kv_prefix_v = kv_global_v.index_select(0, cp_context.kv_gather_index).contiguous()

        output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_real,
            kv_prefix_k,
            kv_prefix_v,
            pse_shift=None,
            atten_mask=self._causal_mask,
            actual_seq_lengths=cp_context.q_cu_seqlens,
            actual_seq_lengths_kv=cp_context.kv_cu_seqlens,
            num_heads=self.num_heads,
            scale=self.scale,
            input_layout="TND",
            num_key_value_heads=self.num_kv_heads,
            sparse_mode=3,
            softmax_lse_flag=False,
        )
        output = output.reshape(-1, self.num_heads * self.head_dim)

        # Scatter real-query outputs back into the padded [total_local] layout;
        # padding rows stay zero (they are never selected by restore_index in
        # the subsequent all-gather merge).
        out_local = q_3d.new_zeros(local_tokens, self.num_heads * self.head_dim)
        out_local.index_copy_(0, cp_context.query_index, output)
        return out_local

    # ------------------------------------------------------------------
    # Decode: FIA with block_table (paged KV, no gather)
    # ------------------------------------------------------------------

    def _fia_out(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_size: int,
    ) -> None:
        if self._use_fia_v2:
            torch.ops.npu.npu_fused_infer_attention_score_v2.out(
                q,
                k,
                v,
                query_rope=None,
                key_rope=None,
                pse_shift=None,
                atten_mask=None,
                actual_seq_qlen=self._actual_seq_q,
                actual_seq_kvlen=self._actual_seq_kv,
                block_table=self._block_table_i32,
                num_query_heads=self.num_heads,
                softmax_scale=self.scale,
                input_layout="TND",
                num_key_value_heads=self.num_kv_heads,
                sparse_mode=_SPARSE_MODE_NONE,
                block_size=block_size,
                return_softmax_lse=False,
                workspace=self._graph_workspace,
                out=[self._current_graph_output, self._current_graph_lse],
            )
            return

        torch.ops.npu.npu_fused_infer_attention_score.out(
            q,
            k,
            v,
            pse_shift=None,
            atten_mask=None,
            actual_seq_lengths=self._actual_seq_q,
            actual_seq_lengths_kv=self._actual_seq_kv,
            block_table=self._block_table_i32,
            num_heads=self.num_heads,
            scale=self.scale,
            input_layout="TND",
            num_key_value_heads=self.num_kv_heads,
            sparse_mode=_SPARSE_MODE_NONE,
            block_size=block_size,
            softmax_lse_flag=False,
            workspace=self._graph_workspace,
            out=[self._current_graph_output, self._current_graph_lse],
        )

    def _decode(
        self,
        q_3d: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: AttentionMetadata,
        num_tokens: int,
    ) -> torch.Tensor:
        block_size = k_cache.size(1)
        k_flat = k_cache.view(k_cache.size(0), block_size, -1)
        v_flat = v_cache.view(v_cache.size(0), block_size, -1)

        graph_context = get_forward_context().acl_graph
        if graph_context is not None:
            if self._current_graph_output is None:
                raise RuntimeError("ACL graph output buffer is not prepared")
            stream = graph_context.stream
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            torch.npu.graph_task_group_begin(stream)
            try:
                self._fia_out(q_3d, k_flat, v_flat, block_size)
            except Exception:
                torch.npu.graph_task_group_end(stream)
                raise
            handle = torch.npu.graph_task_group_end(stream)

            def _update_fia_args() -> None:
                self._fia_out(q_3d, k_flat, v_flat, block_size)

            graph_context.tasks.append(AclGraphTask(event, handle, _update_fia_args))
            return self._current_graph_output.reshape(num_tokens, self.num_heads * self.head_dim)

        if self._use_fia_v2:
            output, _ = torch.ops.npu.npu_fused_infer_attention_score_v2(
                q_3d,
                k_flat,
                v_flat,
                query_rope=None,
                key_rope=None,
                pse_shift=None,
                atten_mask=None,
                actual_seq_qlen=self._actual_seq_q[:num_tokens],
                actual_seq_kvlen=self._actual_seq_kv[:num_tokens],
                block_table=self._block_table_i32,
                num_query_heads=self.num_heads,
                softmax_scale=self.scale,
                input_layout="TND",
                num_key_value_heads=self.num_kv_heads,
                sparse_mode=_SPARSE_MODE_NONE,
                block_size=block_size,
                return_softmax_lse=False,
            )
        else:
            output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_3d,
                k_flat,
                v_flat,
                pse_shift=None,
                atten_mask=None,
                actual_seq_lengths=self._actual_seq_q[:num_tokens],
                actual_seq_lengths_kv=self._actual_seq_kv[:num_tokens],
                block_table=self._block_table_i32,
                num_heads=self.num_heads,
                scale=self.scale,
                input_layout="TND",
                num_key_value_heads=self.num_kv_heads,
                sparse_mode=_SPARSE_MODE_NONE,
                block_size=block_size,
                softmax_lse_flag=False,
            )
        return output.reshape(num_tokens, self.num_heads * self.head_dim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cumulative_seq_lens(
        self,
        metadata: AttentionMetadata,
        num_tokens: int,
    ) -> list[int]:
        if self._actual_seq_lens is not None:
            return self._actual_seq_lens
        return [num_tokens]
