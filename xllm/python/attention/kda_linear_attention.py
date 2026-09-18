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

from typing import TYPE_CHECKING

import torch

from xllm.python.attention.backend import resolve_linear_state_io_indices
from xllm.python.model_executor.forward_context import (
    get_execution_buffer,
    in_acl_graph,
)

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention


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
        """Run native convolution and activation against staged channel-last states."""
        inputs = value.transpose(1, 2).contiguous()
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
        return output.reshape(value.shape[0], value.shape[2], value.shape[1]).transpose(1, 2)

    def disarm_kda_v3_slots(self, idx: torch.Tensor) -> None:
        """Mark slots' V3 combined-pool state invalid (prefill restart)."""
        idx64 = idx if idx.dtype == torch.int64 else idx.to(torch.int64)
        for st in self.__dict__.get("_kda_v3", {}).values():
            if "armed_buf" in st:
                st["armed_buf"].index_fill_(0, idx64, False)

    def snapshot_kda_v3_state(self, idx: torch.Tensor):
        """Snapshot V3 combined-pool rows the graph warmup/capture mutates."""
        idx64 = idx if idx.dtype == torch.int64 else idx.to(torch.int64)
        snap = []
        for st in self.__dict__.get("_kda_v3", {}).values():
            if "armed_buf" not in st:
                continue
            # nslots is the C++ pool capacity (armed_buf/kv_prev are sized to
            # it); the combined pool holds rows_per_seq = R slots per seq
            # (base + R-1 drafts), so snapshot ALL R slots — graph capture
            # mutates every slot's conv/ssm state.
            nslots = st["armed_buf"].shape[0]
            rslots = st["combined_conv"].shape[0] // nslots
            slot_idx = [idx64 + j * nslots for j in range(rslots)]
            snap.append(
                (
                    st,
                    idx64,
                    nslots,
                    rslots,
                    [st["combined_conv"].index_select(0, s).clone() for s in slot_idx],
                    [st["combined_ssm"].index_select(0, s).clone() for s in slot_idx],
                    st["kv_prev"].index_select(0, idx64).clone(),
                    st["armed_buf"].index_select(0, idx64).clone(),
                )
            )
        return snap or None

    @staticmethod
    def restore_kda_v3_state(snap) -> None:
        if not snap:
            return
        for st, idx64, nslots, rslots, conv_snaps, ssm_snaps, kv, ar in snap:
            for j in range(rslots):
                st["combined_conv"].index_copy_(0, idx64 + j * nslots, conv_snaps[j])
                st["combined_ssm"].index_copy_(0, idx64 + j * nslots, ssm_snaps[j])
            st["kv_prev"].index_copy_(0, idx64, kv)
            st["armed_buf"].index_copy_(0, idx64, ar)

    def execute_linear(
        self,
        mixed_qkv: torch.Tensor,
        beta: torch.Tensor,
        layer: Attention,
        raw_gate_proj: torch.Tensor,
    ) -> torch.Tensor:
        """KDA delta-rule over framework conv/ssm slots.

        Returns ``[B, S, num_heads_local, head_dim]``. The conv1d + delta-rule
        math is identical to the model-layer self-contained path (validated
        against the transformers reference); only the state I/O moved here so
        both attention layer types dispatch through the backend.

        ``raw_gate_proj`` is the pre-gate projection ``f_b(f_a(x))``. Plain
        decode/prefill kernels fuse the safe-gate calculation in-kernel. MTP
        verify uses the V3 multi-slot path with a materialized safe-gate from
        ``Glm5NextForgetGate.gate_from_raw``.
        """
        from fla_npu.ops.ascendc import chunk_kda_fwd, recurrent_kda

        from xllm.python.models.glm5_next import _l2norm

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
        num_seqs = idx.shape[0] if idx is not None else batch_size
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
            beta = beta.transpose(0, 1)  # [1, T, nh] -> [T, 1, nh]
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
                # equal; take each group's row 0.
                num_rows = int(mixed_qkv.shape[2])
                q_seq_lens = getattr(metadata, "q_seq_lens", None)
                n_groups = int(q_seq_lens.numel()) if q_seq_lens is not None else int(q_cu_raw.numel()) - 1
                if num_seqs > 0 and n_groups > 0 and (num_rows % n_groups == 0):
                    if idx.numel() == num_rows:
                        group_idx = idx.view(n_groups, num_rows // n_groups)[:, 0].contiguous()
                    else:
                        group_idx = idx
                    return self._spec_verify_v3(
                        mixed_qkv,
                        fg.gate_from_raw(raw_gate_proj),
                        beta,
                        layer,
                        group_idx,
                        metadata,
                        conv_cache,
                        ssm_cache,
                        recurrent_kda,
                    )
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
                    fg.gate_from_raw(raw_gate_proj),
                    beta,
                    layer,
                    idx,
                    metadata,
                    conv_cache,
                    ssm_cache,
                    recurrent_kda,
                )
            if is_prefill:
                self.disarm_kda_v3_slots(idx)
            elif (
                self._kda_verify_width > 1
                and idx.numel() > 0
                and mixed_qkv.dim() == 3
                and mixed_qkv.shape[2] >= idx.numel()
                and mixed_qkv.shape[2] % idx.numel() == 0
            ):
                return self._spec_verify_v3(
                    mixed_qkv,
                    fg.gate_from_raw(raw_gate_proj),
                    beta,
                    layer,
                    idx,
                    metadata,
                    conv_cache,
                    ssm_cache,
                    recurrent_kda,
                )
            state_read_idx = read_idx if is_prefill else idx
            conv_i = conv_cache.index_select(0, state_read_idx)
            ssm_i = ssm_cache.index_select(0, state_read_idx)
            his = metadata.has_initial_state
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
        if num_seqs == batch_size:
            mixed_qkv = self._causal_conv1d(mixed_qkv, conv_state, layer, is_prefill=is_prefill)
        else:
            q_cu = metadata.q_cu_seq_lens
            assert q_cu is not None, "multi-sequence linear attention needs q_cu_seq_lens"
            q_cu = q_cu.to(torch.int64)
            q_cu_list = q_cu.tolist()
            mixed_qkv = self._causal_conv1d(
                mixed_qkv, conv_state, layer, query_start_loc=q_cu_list, is_prefill=is_prefill
            )
            # TND packed layout for recurrent_kda: [T, nh, hd] per channel group.
            seq_len = int(q_cu_list[-1])
            hidden_shape = (1, seq_len, -1, head_dim)

        query, key, value = torch.split(mixed_qkv.transpose(1, 2), [qkv_dim] * 3, dim=-1)
        query = query.view(hidden_shape)
        key = key.view(hidden_shape)
        value = value.view(hidden_shape)

        # ``beta`` arrives as [B, S, nh] and is already correct for both
        # layouts: per-sequence [num_seqs, per_seq_len, nh] when the batch rows
        # map 1:1 to sequences, and flattened [1, T, nh] (T = sum of q_cu) for
        # the varlen multi-sequence path. Re-viewing it as
        # (num_seqs, seq_len, nh) assumes a uniform per-seq length == the
        # flattened total and crashes on multi-sequence decode batches
        # (e.g. 2 concurrent requests: view [2, 2, 4] on 8 elements).
        # fla_npu KDA ops require fp32 gate/beta (the pure-torch reference also
        # upcasts them); the model hands them in bf16.
        b = beta.to(torch.float32)
        fuse_gate = raw_gate_proj is not None and fg is not None and gate_lb is not None and -5.0 <= gate_lb < 0.0
        if fuse_gate:
            g = None
            g_raw = raw_gate_proj if num_seqs == batch_size else raw_gate_proj.view(hidden_shape)
            g_raw = g_raw.to(torch.float32)
            kda_A_log = fg.A_log.to(torch.float32).contiguous()
            kda_dt_bias = fg.dt_bias.to(torch.float32).contiguous()
            _gate_kwargs = dict(
                A_log=kda_A_log,
                dt_bias=kda_dt_bias,
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
            b_tnd = b.reshape(-1, num_heads_local).contiguous()
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
                b_tnd,
                initial_state=ssm_state,
                cu_seqlens=cu_seqlens,
                layout="TND",
                scale=scale,
                output_final_state=True,
                inplace_final_state=False,
                use_qk_l2norm_in_kernel=True,
                use_beta_sigmoid_in_kernel=False,
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
            q_in = _l2norm(query.float(), dim=-1, eps=1e-6).to(torch.bfloat16).contiguous()
            k_in = _l2norm(key.float(), dim=-1, eps=1e-6).to(torch.bfloat16).contiguous()
            v_in = value.to(torch.bfloat16).contiguous()
            cu_seqlens = (
                q_cu.to(torch.int32)
                if num_seqs != batch_size
                else torch.tensor([0, seq_len], dtype=torch.int32, device=device)
            )
            if cu_seqlens.numel() > 2:
                # Seq-wise prefill: one single-sequence chunk_kda_fwd per
                # sequence. A merged multi-sequence prefill (engine batches the
                # concurrent requests' prefills) leaves per-seq conv/ssm state
                # that differs from a single-request prefill (state-fingerprint
                # verified: L0 state matches, L1+ diverges on seq1/seq2), and
                # that state drift propagates through every later verify step.
                # state_v_first=True pins the [HV,V,K] state layout so the
                # ssm state handed to decode matches the recurrent path
                # (default False is [HV,K,V] — K/V-transposed; K=V=128 hides
                # the shape mismatch while corrupting decode precision).
                _pout, _pstates = [], []
                for s in range(num_seqs):
                    t0, t1 = q_cu_list[s], q_cu_list[s + 1]
                    sel = slice(int(t0), int(t1))
                    _r = chunk_kda_fwd(
                        q_in[:, sel].contiguous(),
                        k_in[:, sel].contiguous(),
                        v_in[:, sel].contiguous(),
                        (g_raw[:, sel] if fuse_gate else g[:, sel]).contiguous(),
                        b[:, sel].contiguous(),
                        scale,
                        chunk_size=64,
                        layout="BSND",
                        initial_state=ssm_state[s : s + 1],
                        output_final_state=True,
                        cu_seqlens=torch.tensor([0, int(t1 - t0)], dtype=torch.int32, device=device),
                        return_intermediate_states=False,
                        state_v_first=True,
                        **_gate_kwargs,
                    )
                    _pout.append(_r[0])
                    _pstates.append(_r[1])
                # Each _r[0] is [1, seq_len_s, nh, hd] (layout="BSND", one
                # sequence per call). The per-sequence token counts differ
                # across a multi-sequence prefill batch, so they must be
                # concatenated along the token axis (dim=1) to restore the
                # original [1, total_tokens, nh, hd] packing of q_in — cat on
                # dim=0 would require equal seq_len and crashes (aclnnCat 161002
                # "dim 1 of tensor 1 is [X], should be equal to tensor 0 [Y]")
                # at >=2 concurrent prefills of differing length.
                core_attn_out = torch.cat(_pout, dim=1).to(query.dtype)
                final_state = torch.cat(_pstates, dim=0)
            else:
                result = chunk_kda_fwd(
                    q_in,
                    k_in,
                    v_in,
                    g_raw if fuse_gate else g,
                    b,
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
            conv_cache.index_copy_(0, idx, conv_state)
            ssm_cache.index_copy_(0, idx, final_state.float().contiguous())
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
