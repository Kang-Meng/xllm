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

"""Kimi K3 KDA projections, weight loading and local-head computation.

This component reuses the same FLA AscendC kernels as the mainline GLM KDA
backend. Linear layers, quantization and checkpoint access reuse mainline
implementations. State allocation and scheduling remain with their owners.
The layer retains no conv/recurrent cache or
block references between calls. Prefill uses BSND, recurrent decode uses TND,
and recurrent states use VK layout. Speculative decode and CP are unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from xllm.python import kernels

if TYPE_CHECKING:
    from xllm.python.model_loader import ScopedWeightLoader

KDA_CHUNK_SIZE = 64
# The executor uses this sentinel for dummy rows in a static decode graph.
PAD_SLOT_ID = -1
_L2NORM_EPS = 1e-6


@dataclass
class KimiK3KDAMetadata:
    """Per-step scheduling info for KDA layers.

    Tokens of the whole batch are packed along dim 0. The leading
    ``num_decode_seqs`` sequences are decode requests (exactly 1 token each);
    the following ``num_prefill_seqs`` sequences are (chunked-)prefill
    requests.
    """

    # [num_seqs + 1] int32 on device, cumulative token counts.
    query_start_loc: torch.Tensor
    # [num_seqs] int64 on device, conv/recurrent state slot id per sequence.
    state_indices: torch.Tensor
    num_decode_seqs: int
    num_prefill_seqs: int
    # [num_seqs] bool; True when the sequence's conv/recurrent state must be
    # loaded from the caches (always True for decode; True for chunked-prefill
    # continuations). Required whenever ``num_prefill_seqs > 0``.
    has_initial_state: torch.Tensor | None = None
    # Host-side metadata shared by all KDA layers in one prefill step.  The
    # executor builds it once so each layer does not call ``tolist()`` again.
    prefill_cu_seqlens_host: tuple[int, ...] | None = None
    prefill_chunk_indices_host: tuple[int, ...] | None = None
    # Full-batch convolution arguments (including leading decode sequences).
    # Materialized once alongside the chunk metadata, never in a layer kernel.
    conv_state_indices_host: list[int] | None = None
    conv_initial_state_mode_host: list[int] | None = None
    conv_query_start_loc_host: list[int] | None = None


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    # The mainline NPU kernel supports BF16; retain the FP32 oracle and other
    # input dtypes without sending them to an unsupported native kernel.
    if x.dtype == torch.bfloat16:
        return kernels.l2_norm(x, _L2NORM_EPS)
    x32 = x.float()
    return (x32 * torch.rsqrt((x32 * x32).sum(-1, keepdim=True) + _L2NORM_EPS)).to(x.dtype)


def _build_chunk_indices(cu_seqlens: list[int], chunk_size: int) -> list[int]:
    """Flat [seq_idx, chunk_idx] pairs, mirroring build_kda_chunk_indices."""
    indices: list[int] = []
    for seq in range(len(cu_seqlens) - 1):
        seq_len = cu_seqlens[seq + 1] - cu_seqlens[seq]
        for chunk in range(-(-seq_len // chunk_size)):
            indices.extend((seq, chunk))
    return indices


def build_kda_prefill_host_metadata(
    query_start_loc: torch.Tensor | None,
    num_decode_seqs: int,
) -> tuple[tuple[int, ...], tuple[int, ...]] | tuple[None, None]:
    """Build host metadata shared by every KDA layer in one prefill step.

    The AscendC prefill kernels accept host-side cumulative lengths and chunk
    indices.  Building them once per scheduler step avoids a device-to-host
    sync and Python loop in every KDA layer.
    """
    if query_start_loc is None:
        return None, None

    prefill_cu = query_start_loc[num_decode_seqs:]
    if prefill_cu.numel() <= 1:
        return None, None

    cu_values = [int(value) for value in prefill_cu.detach().cpu().reshape(-1).tolist()]
    cu_list = [value - cu_values[0] for value in cu_values]
    chunk_indices = tuple(_build_chunk_indices(cu_list, KDA_CHUNK_SIZE))
    return tuple(cu_list), chunk_indices


class KimiK3KDACore(nn.Module):
    """Short convolution, delta rule and gated RMSNorm for one TP head shard.

    The caller supplies processed weights and already projected inputs. Weights
    are read-only registered buffers; conv/recurrent state is passed per call
    and updated in place. Output is the local head result before output
    projection and TP reduction.
    """

    def __init__(
        self,
        conv_weight_t: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        o_norm_weight: torch.Tensor,
        *,
        rms_norm_eps: float = 1e-6,
        gate_lower_bound: float | None = None,
    ) -> None:
        super().__init__()
        if a_log.ndim != 1 or a_log.numel() == 0:
            raise ValueError("Kimi KDA A_log must contain one value per local head")
        if o_norm_weight.ndim != 1 or o_norm_weight.numel() == 0:
            raise ValueError("Kimi KDA output norm must contain one value per head channel")
        self.local_num_heads = a_log.numel()
        self.head_dim = o_norm_weight.numel()
        self.local_projection_size = self.local_num_heads * self.head_dim
        if conv_weight_t.ndim != 2 or conv_weight_t.shape[1] != 3 * self.local_projection_size:
            raise ValueError("Kimi KDA conv weight must have shape [conv_size, 3 * local_projection_size]")
        if conv_weight_t.shape[0] < 2 or dt_bias.shape != (self.local_projection_size,):
            raise ValueError("Kimi KDA requires conv_size >= 2 and one dt_bias per local channel")
        if rms_norm_eps <= 0:
            raise ValueError("Kimi KDA RMSNorm epsilon must be positive")
        if gate_lower_bound is not None and not -5.0 <= gate_lower_bound < 0:
            raise ValueError("Kimi KDA gate lower bound must be in [-5, 0)")
        self.conv_size = conv_weight_t.shape[0]
        self.scale = self.head_dim**-0.5
        self.o_norm_eps = rms_norm_eps
        self.gate_lower_bound = gate_lower_bound
        self.register_buffer("conv_weight_t", conv_weight_t, persistent=False)
        self.register_buffer("A_log", a_log, persistent=False)
        self.register_buffer("dt_bias", dt_bias, persistent=False)
        self.register_buffer("o_norm_weight", o_norm_weight, persistent=False)

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        output_gate: torch.Tensor,
        metadata: KimiK3KDAMetadata,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> torch.Tensor:
        """Consume packed local projections and return [tokens, local_proj].

        mixed_qkv is [T, 3P], raw/output gates are [T, H, D], and beta is
        [T, H] with sigmoid already applied. Decode sequences precede prefill
        sequences, one token per decode. Dummy decode slots use PAD_SLOT_ID.
        conv_state is [slots, conv_size - 1, 3P]; recurrent_state is
        [slots, H, V, K] in float32. Only addressed valid slots are updated.
        """
        num_tokens = mixed_qkv.shape[0]
        num_decode = metadata.num_decode_seqs
        num_prefill = metadata.num_prefill_seqs
        if mixed_qkv.ndim != 2 or mixed_qkv.shape[1] != 3 * self.local_projection_size:
            raise ValueError("Kimi KDA mixed_qkv must have shape [tokens, 3 * local_projection_size]")
        gate_shape = (num_tokens, self.local_num_heads, self.head_dim)
        if raw_gate.shape != gate_shape or output_gate.shape != gate_shape:
            raise ValueError("Kimi KDA gates must have shape [tokens, local_heads, head_dim]")
        if beta.shape != (num_tokens, self.local_num_heads):
            raise ValueError("Kimi KDA beta must have shape [tokens, local_heads]")
        if num_decode < 0 or num_prefill < 0 or num_decode > num_tokens:
            raise ValueError("Kimi KDA sequence counts must be nonnegative and decode rows must fit the input")
        num_sequences = num_decode + num_prefill
        if metadata.query_start_loc.shape != (num_sequences + 1,) or metadata.state_indices.shape != (num_sequences,):
            raise ValueError("Kimi KDA boundaries and slots must match the sequence count")
        if num_prefill and (metadata.has_initial_state is None or metadata.has_initial_state.shape != (num_sequences,)):
            raise ValueError("Kimi KDA prefill requires one initial-state flag per sequence")
        if num_prefill and (metadata.prefill_cu_seqlens_host is None or metadata.prefill_chunk_indices_host is None):
            raise ValueError("Kimi KDA prefill requires shared host sequence and chunk metadata")
        if num_prefill and (
            metadata.conv_state_indices_host is None
            or len(metadata.conv_state_indices_host) != num_sequences
            or metadata.conv_initial_state_mode_host is None
            or len(metadata.conv_initial_state_mode_host) != num_sequences
            or metadata.conv_query_start_loc_host is None
            or len(metadata.conv_query_start_loc_host) != num_sequences + 1
        ):
            raise ValueError("Kimi KDA prefill requires shared host convolution metadata")
        if conv_state.shape[1:] != (self.conv_size - 1, 3 * self.local_projection_size):
            raise ValueError("Kimi KDA convolution state has an incompatible layout")
        if recurrent_state.shape != (conv_state.shape[0], self.local_num_heads, self.head_dim, self.head_dim):
            raise ValueError("Kimi KDA recurrent state must use [slots, heads, V, K] layout")
        if recurrent_state.dtype != torch.float32:
            raise ValueError("Kimi KDA recurrent state must be float32")

        core_attn_out = mixed_qkv.new_zeros((1, num_tokens, self.local_num_heads, self.head_dim))
        if num_sequences:
            mixed_qkv = self._causal_conv1d(
                mixed_qkv,
                conv_state,
                metadata,
            )
            q, k, v = (
                x.reshape(1, num_tokens, self.local_num_heads, self.head_dim) for x in mixed_qkv.chunk(3, dim=-1)
            )
            raw_gate = raw_gate.unsqueeze(0)
            beta = beta.unsqueeze(0)
            if num_decode:
                core_attn_out[:, :num_decode] = self._recurrent(
                    q[:, :num_decode],
                    k[:, :num_decode],
                    v[:, :num_decode],
                    raw_gate[:, :num_decode],
                    beta[:, :num_decode],
                    recurrent_state,
                    cu_seqlens=metadata.query_start_loc[: num_decode + 1],
                    state_indices=metadata.state_indices[:num_decode],
                )
            if num_prefill:
                core_attn_out[:, num_decode:] = self._prefill(
                    q[:, num_decode:],
                    k[:, num_decode:],
                    v[:, num_decode:],
                    raw_gate[:, num_decode:],
                    beta[:, num_decode:],
                    recurrent_state,
                    metadata,
                )
        out = self._gated_rms_norm(core_attn_out, output_gate.unsqueeze(0))
        return out.reshape(num_tokens, self.local_projection_size)

    def _causal_conv1d(
        self,
        mixed_qkv: torch.Tensor,
        conv_state: torch.Tensor,
        metadata: KimiK3KDAMetadata,
    ) -> torch.Tensor:
        if metadata.num_prefill_seqs:
            from fla_npu.ops.ascendc import npu_causal_conv1d

            return npu_causal_conv1d(
                mixed_qkv,
                self.conv_weight_t,
                conv_states=conv_state,
                cache_indices=metadata.conv_state_indices_host,
                initial_state_mode=metadata.conv_initial_state_mode_host,
                query_start_loc=metadata.conv_query_start_loc_host,
                activation_mode=1,
                pad_slot_id=PAD_SLOT_ID,
                run_mode=0,
                head_num=0,
            )
        # The existing tiled kernel disables its padding check for a negative
        # sentinel. Use a positive out-of-pool sentinel so dummy graph rows
        # never access the cache; valid state slots keep their original IDs.
        pad_slot = conv_state.shape[0]
        cache_indices = metadata.state_indices
        slots = torch.where(cache_indices < 0, pad_slot, cache_indices)
        output = torch.ops.xllm_ops.causal_conv1d_update_v2(
            mixed_qkv,
            conv_state.transpose(1, 2),
            self.conv_weight_t.transpose(0, 1),
            True,
            None,
            slots,
            metadata.query_start_loc,
            1,
            pad_slot,
            None,
        )
        return output.masked_fill(cache_indices.reshape(-1, 1) < 0, 0)

    def _recurrent(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        state_indices: torch.Tensor,
    ) -> torch.Tensor:
        # The shared recurrent kernel aborts a core on an invalid state slot.
        # Represent padding as empty sequences instead, and pack the valid
        # single-token rows without a host synchronization during graph replay.
        valid = state_indices >= 0
        num_sequences = state_indices.numel()
        order = None
        row_counts = None
        if num_sequences > 1:
            positions = torch.arange(num_sequences, device=state_indices.device, dtype=torch.float32)
            order = torch.argsort((~valid).float() * num_sequences + positions)
            # Use one INT32 prefix sum both to construct valid-first boundaries
            # and to restore original row positions. INT64 cumsum can fall back
            # to AICPU; a second scan/gather is unnecessary for padding.
            row_counts = valid.to(torch.int32).cumsum(0, dtype=torch.int32)
        # A one-row graph has no later valid row to lose when the kernel
        # skips a padding slot. Preserve the existing boundaries in this case.
        packed_cu = (
            cu_seqlens
            if num_sequences == 1
            else torch.minimum(
                torch.arange(num_sequences + 1, device=state_indices.device, dtype=torch.int32), row_counts[-1]
            )
        )

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            flat = tensor.reshape(num_sequences, *tensor.shape[-2:])
            return (flat if order is None else flat.index_select(0, order)).contiguous()

        packed_beta = beta.reshape(num_sequences, self.local_num_heads)
        packed_slots = state_indices.to(torch.int64)
        if order is not None:
            packed_beta = packed_beta.index_select(0, order)
            packed_slots = packed_slots.index_select(0, order)
        from fla_npu.ops.ascendc import recurrent_kda

        output, _ = recurrent_kda(
            pack(q),
            pack(k),
            pack(v),
            pack(raw_gate),
            packed_beta.contiguous(),
            initial_state=recurrent_state,
            cu_seqlens=packed_cu,
            ssm_state_indices=packed_slots.contiguous(),
            A_log=self.A_log.reshape(-1).contiguous(),
            dt_bias=self.dt_bias.contiguous(),
            layout="TND",
            scale=self.scale,
            output_final_state=True,
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=False,
            allow_neg_eigval=False,
            safe_gate=self.gate_lower_bound is not None,
            lower_bound=self.gate_lower_bound,
            state_v_first=True,
        )
        if num_sequences > 1:
            # Boundaries and slots both follow the packed order. Restore the
            # original token order using each valid row's compacted index.
            # Padding selects an arbitrary initialized row and is cleared below.
            rows = (row_counts - 1).clamp_min(0)
            output = output.index_select(0, rows)
        output = output.masked_fill(~valid.reshape(-1, 1, 1), 0)
        return output.reshape(v.shape)

    def _prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        metadata: KimiK3KDAMetadata,
    ) -> torch.Tensor:
        num_decode = metadata.num_decode_seqs
        cu_list = metadata.prefill_cu_seqlens_host
        chunk_indices = metadata.prefill_chunk_indices_host
        if cu_list is None or chunk_indices is None:
            raise ValueError("Kimi KDA prefill requires shared host sequence and chunk metadata")
        slots = metadata.state_indices.long()[num_decode:]
        has_init = metadata.has_initial_state[num_decode:]

        initial_state = recurrent_state.index_select(0, slots)
        initial_state = initial_state * has_init.view(-1, 1, 1, 1)
        from fla_npu.ops.ascendc import chunk_kda_fwd

        result = chunk_kda_fwd(
            _l2norm(q.contiguous()),
            _l2norm(k.contiguous()),
            v.contiguous(),
            raw_gate.contiguous(),
            beta.contiguous(),
            self.scale,
            chunk_size=KDA_CHUNK_SIZE,
            layout="BSND",
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_list,
            chunk_indices=chunk_indices,
            use_gate_in_kernel=True,
            A_log=self.A_log.reshape(-1).contiguous(),
            dt_bias=self.dt_bias.contiguous(),
            safe_gate=self.gate_lower_bound is not None,
            lower_bound=self.gate_lower_bound,
            return_intermediate_states=False,
            state_v_first=True,
        )
        recurrent_state.index_copy_(0, slots, result[1].to(recurrent_state.dtype))
        return result[0]

    def _gated_rms_norm(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        y = x32 * torch.rsqrt((x32 * x32).mean(-1, keepdim=True) + self.o_norm_eps) * self.o_norm_weight.float()
        return (y * torch.sigmoid(gate.float())).to(x.dtype)


class KimiK3DeltaAttention(nn.Module):
    """Complete KDA layer with head-sharded weights and caller-owned state.

    Q/K/V may use mainline dynamic W8A8; all gates and output projection stay
    floating point. ``f_a_proj`` and output norm are replicated. All other
    input projections shard head-major rows; ``o_proj`` shards input columns.
    Output TP reduction can be deferred to the caller for sequence parallelism.
    This component does not register a model or allocate runtime caches.
    """

    def __init__(
        self,
        hidden_size: int,
        linear_attn_config: dict[str, Any],
        *,
        tp_size: int = 1,
        tp_rank: int = 0,
        rms_norm_eps: float = 1e-6,
        quantized: bool = False,
        reduce_o_proj: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        from xllm.python.layers import ColumnParallelLinear, RowParallelLinear

        self.num_heads = int(linear_attn_config["num_heads"])
        self.head_dim = int(linear_attn_config["head_dim"])
        self.conv_size = int(linear_attn_config["short_conv_kernel_size"])
        if not linear_attn_config.get("use_full_rank_gate", False):
            raise ValueError("Kimi KDA requires a full-rank gate")
        if tp_size < 1 or not 0 <= tp_rank < tp_size or self.num_heads <= 0 or self.num_heads % tp_size:
            raise ValueError("Kimi KDA heads must divide TP size and TP rank must be in range")
        if hidden_size <= 0 or self.head_dim <= 0 or self.conv_size < 2 or rms_norm_eps <= 0:
            raise ValueError("Kimi KDA requires positive dimensions/epsilon and conv_size >= 2")
        self.gate_lower_bound = linear_attn_config.get("gate_lower_bound")
        if self.gate_lower_bound is not None and not -5.0 <= self.gate_lower_bound < 0:
            raise ValueError("Kimi KDA gate lower bound must be in [-5, 0)")
        if quantized and dtype != torch.bfloat16:
            raise ValueError("Kimi KDA W8A8 uses the mainline bfloat16 output contract")
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.local_num_heads = self.num_heads // tp_size
        self.local_projection_size = self.local_num_heads * self.head_dim
        self.quantized = quantized
        self.reduce_o_proj = reduce_o_proj
        self.o_norm_eps = rms_norm_eps
        self._weights_loaded = False
        self._core: KimiK3KDACore | None = None
        local_proj = self.local_projection_size

        for name in ("q_proj", "k_proj", "v_proj"):
            if quantized:
                from xllm.python.models.deepseek_v32 import W8A8DynamicLinear

                projection = W8A8DynamicLinear(
                    hidden_size, local_proj, torch.device(device), transpose_weight_after_loading=False
                )
            else:
                projection = ColumnParallelLinear(hidden_size, local_proj, tp_size, dtype=dtype, device=device)
            setattr(self, name, projection)
        self.g_proj = ColumnParallelLinear(hidden_size, local_proj, tp_size, dtype=dtype, device=device)
        self.b_proj = ColumnParallelLinear(hidden_size, self.local_num_heads, tp_size, dtype=dtype, device=device)
        # The low-rank decay-gate input is replicated, not head-sharded.
        self.f_a_proj = ColumnParallelLinear(hidden_size, self.head_dim, tp_size, dtype=dtype, device=device)
        self.f_b_proj = ColumnParallelLinear(self.head_dim, local_proj, tp_size, dtype=dtype, device=device)
        self.conv1d_weight = nn.Parameter(
            torch.empty(3 * local_proj, 1, self.conv_size, dtype=torch.float32, device=device), requires_grad=False
        )
        self.A_log = nn.Parameter(
            torch.empty(self.local_num_heads, dtype=torch.float32, device=device), requires_grad=False
        )
        self.dt_bias = nn.Parameter(torch.empty(local_proj, dtype=torch.float32, device=device), requires_grad=False)
        self.o_norm_weight = nn.Parameter(torch.empty(self.head_dim, dtype=dtype, device=device), requires_grad=False)
        # Preserve KDA's checkpoint-layout projection rounding through the
        # mainline row-linear option. Reduction is deferred per forward.
        self.o_proj = RowParallelLinear(
            local_proj,
            hidden_size,
            tp_size,
            dtype=dtype,
            device=device,
            reduce_results=False,
            use_checkpoint_layout=True,
        )

    def conv_state_shape(self) -> tuple[int, int]:
        return self.conv_size - 1, 3 * self.local_projection_size

    def recurrent_state_shape(self) -> tuple[int, int, int]:
        return self.local_num_heads, self.head_dim, self.head_dim

    def load_weights(self, weights: ScopedWeightLoader) -> set[str]:
        """Load a loader scoped to the layer's checkpoint prefix, before packing.

        Packing changes matmul layouts. Reject reloads after packing rather than
        copying checkpoint-layout weights into already transposed parameters.
        """
        from xllm.python.model_loader import shard_tensor

        if self._core is not None:
            raise RuntimeError("Kimi KDA cannot reload weights after layout preparation")
        self._weights_loaded = False
        loaded: set[str] = set()

        def load(parameter: torch.Tensor, name: str, dim: int | None = None) -> None:
            weights.load_tensor(parameter, name, dim=dim, rank=self.tp_rank, world_size=self.tp_size)
            loaded.add(name)

        for name in ("q_proj", "k_proj", "v_proj"):
            projection = getattr(self, name)
            load(projection.weight, f"{name}.weight", 0)
            if self.quantized:
                for suffix in ("weight_scale", "weight_offset"):
                    key = f"{name}.{suffix}"
                    value = weights.get_tensor(key)
                    if value.ndim == 1:
                        value = value.unsqueeze(-1)
                    value = weights.shard_value(value, key, 0, self.tp_rank, self.tp_size)
                    weights.copy(getattr(projection, suffix), value, key)
                    loaded.add(key)
        for name in ("g_proj", "b_proj", "f_b_proj"):
            load(getattr(self, name).weight, f"{name}.weight", 0)
        load(self.f_a_proj.weight, "f_a_proj.weight")
        load(self.o_proj.weight, "o_proj.weight", 1)

        conv_parts: list[torch.Tensor] = []
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            key = f"{name}.weight"
            value = weights.get_tensor(key)
            if value.ndim == 2:
                value = value.unsqueeze(1)
            value = weights.shard_value(value, key, 0, self.tp_rank, self.tp_size)
            if value.shape != (self.local_projection_size, 1, self.conv_size):
                raise ValueError(f"Kimi KDA {key} must have shape [projection_size, 1, conv_size]")
            conv_parts.append(value)
            loaded.add(key)
        weights.copy(self.conv1d_weight, torch.cat(conv_parts, dim=0), "conv1d_weight")
        a_log = weights.get_tensor("A_log")
        if a_log.ndim == 4 and a_log.shape[:2] == (1, 1) and a_log.shape[-1] == 1:
            a_log = a_log.flatten()
        if a_log.ndim != 1 or a_log.numel() < self.num_heads:
            raise ValueError("Kimi KDA A_log must contain at least num_heads entries")
        # Official checkpoints pad this array (e.g. 96 heads in 128 entries).
        a_log = shard_tensor(a_log[: self.num_heads], 0, self.tp_rank, self.tp_size, name="A_log")
        weights.copy(self.A_log, a_log, "A_log")
        loaded.add("A_log")
        load(self.dt_bias, "dt_bias", 0)
        load(self.o_norm_weight, "o_norm.weight")
        self._weights_loaded = True
        return loaded

    def process_weights_after_loading(self) -> None:
        """Prepare device layouts once; no convolution-weight copies in forward."""
        if self._core is not None:
            return
        if not self._weights_loaded:
            raise RuntimeError("Kimi KDA requires checkpoint loading before layout preparation")
        if self.quantized:
            for name in ("q_proj", "k_proj", "v_proj"):
                projection = getattr(self, name)
                # Retain ND packing for Kimi quantized projections; NZ changes
                # QuantMatmul rounding for some token buckets.
                projection.weight.data = projection.weight.detach().t().contiguous()
                projection._weight_is_transposed = True
                projection.process_weights_after_loading()
        self.o_proj.process_weights_after_loading()
        conv_weight_t = self.conv1d_weight.detach().squeeze(1).t().contiguous().to(self.o_norm_weight.dtype)
        self._core = KimiK3KDACore(
            conv_weight_t,
            self.A_log,
            self.dt_bias,
            self.o_norm_weight,
            rms_norm_eps=self.o_norm_eps,
            gate_lower_bound=self.gate_lower_bound,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        metadata: KimiK3KDAMetadata,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        reduce_o_proj: bool | None = None,
    ) -> torch.Tensor:
        if self._core is None:
            raise RuntimeError("Kimi KDA requires layout preparation before forward")
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError("Kimi KDA input must have shape [tokens, hidden_size]")
        gate_shape = (hidden_states.shape[0], self.local_num_heads, self.head_dim)
        mixed_qkv = torch.cat(
            (self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)), dim=-1
        )
        raw_gate = self.f_b_proj(self.f_a_proj(hidden_states)).reshape(gate_shape)
        beta = self.b_proj(hidden_states).float().sigmoid()
        output_gate = self.g_proj(hidden_states).reshape(gate_shape)
        local_out = self._core(mixed_qkv, raw_gate, beta, output_gate, metadata, conv_state, recurrent_state)
        output = self.o_proj(local_out)
        reduce_results = self.reduce_o_proj if reduce_o_proj is None else reduce_o_proj
        if self.tp_size > 1 and reduce_results:
            from xllm.python import distributed

            # Reuse the graph-safe PyTorch HCCL reduction used by Kimi MoE.
            distributed.all_reduce_(output)
        return output


__all__ = ["KimiK3DeltaAttention", "KimiK3KDACore", "KimiK3KDAMetadata", "build_kda_prefill_host_metadata"]
