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

"""NPU-native Qwen3.5 gated delta network composition."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.attention.backend import resolve_linear_state_io_indices
from xllm.python.layers.linear import ColumnParallelLinear, RowParallelLinear
from xllm.python.layers.qwen3_5.common import Qwen3_5GatedDeltaNetConfig
from xllm.python.layers.qwen3_5.gated_delta_net import shard_qkv_rows
from xllm.python.model_executor.forward_context import get_forward_context
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader

_MEGA_GDN_CHUNK_SIZE = 128
_MEGA_GDN_MAX_DECODE_BATCH_SIZE = 32
_MEGA_GDN_FIXED_RMS_NORM_EPS = 1e-6
_SUPPORTED_MEGA_PREFILL_HEAD_COUNTS = frozenset((1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64))


def _validate_mega_gdn_head_geometry(
    num_key_heads: int,
    num_value_heads: int,
) -> None:
    """Reject model geometry unsupported by either MegaGdn phase."""
    if num_value_heads not in _SUPPORTED_MEGA_PREFILL_HEAD_COUNTS:
        raise NotImplementedError(f"Qwen3.5 MegaGdnPrefill does not support {num_value_heads} local value heads")
    if num_key_heads <= 0 or num_key_heads > 16:
        raise NotImplementedError("Qwen3.5 MegaGdnDecode supports 1 to 16 local key heads")
    if num_key_heads & (num_key_heads - 1):
        raise NotImplementedError("Qwen3.5 MegaGdnDecode requires a power-of-two local key-head count")
    if num_value_heads % num_key_heads != 0:
        raise NotImplementedError("Qwen3.5 MegaGdn requires Nv divisible by Nk")
    value_heads_per_key = num_value_heads // num_key_heads
    if value_heads_per_key > 4:
        raise NotImplementedError("Qwen3.5 MegaGdnDecode supports at most four value heads per key head")


@dataclass(frozen=True, slots=True)
class _MegaGdnPrefillPlan:
    conv_read: torch.Tensor
    conv_write: torch.Tensor
    ssm_read: torch.Tensor
    ssm_write: torch.Tensor
    num_matrices: int
    num_sequences: int
    num_tokens: int


def _build_mega_prefill_indices(
    read_state_indices: torch.Tensor,
    write_state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    checkpoint_stride: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the independent Conv and SSM read/write indices for MegaGdn."""
    if checkpoint_stride <= 0:
        raise ValueError("Qwen3.5 SSM checkpoint stride must be positive")
    if (
        read_state_indices.dim() != 1
        or write_state_indices.shape != read_state_indices.shape
        or has_initial_state.shape != read_state_indices.shape
    ):
        raise ValueError("Qwen3.5 state indices and validity must be sequence-scoped")

    conv_read_source = read_state_indices.to(dtype=torch.int32).contiguous()
    conv_write = write_state_indices.to(dtype=torch.int32).contiguous()
    valid_read = has_initial_state.to(dtype=torch.bool) & (conv_read_source > 0)
    invalid = torch.full_like(conv_read_source, -1)
    conv_read = torch.where(valid_read, conv_read_source, invalid).contiguous()
    ssm_write = (conv_write * checkpoint_stride).contiguous()
    ssm_read_source = conv_read_source * checkpoint_stride
    ssm_read = torch.where(valid_read, ssm_read_source, invalid).contiguous()
    return conv_read, conv_write, ssm_read, ssm_write


def _compute_mega_prefill_num_matrices(
    query_lengths: list[int],
    num_value_heads: int,
) -> int:
    """Return the number of 128-token/head matrices consumed by MegaGdn."""
    if not query_lengths or any(length <= 0 for length in query_lengths):
        raise ValueError("Qwen3.5 prefill query lengths must be positive")
    return (
        sum((length + _MEGA_GDN_CHUNK_SIZE - 1) // _MEGA_GDN_CHUNK_SIZE for length in query_lengths) * num_value_heads
    )


def _get_or_build_mega_prefill_plan(
    read_state_indices: torch.Tensor,
    write_state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_lengths_host: torch.Tensor,
    checkpoint_stride: int,
    num_value_heads: int,
) -> _MegaGdnPrefillPlan:
    """Build request-scoped MegaGdnPrefill metadata once for all GDN layers."""
    context = get_forward_context()
    cache_key = (
        __name__,
        "mega_gdn_prefill_plan",
        checkpoint_stride,
        num_value_heads,
    )
    cached = context.layer_shared_cache.get(cache_key)
    if cached is not None:
        if not isinstance(cached, _MegaGdnPrefillPlan):
            raise TypeError("invalid cached Qwen3.5 MegaGdnPrefill plan")
        if cached.num_sequences != write_state_indices.numel():
            raise RuntimeError("Qwen3.5 sequence count changed within one forward")
        return cached

    query_lengths = [int(length) for length in query_lengths_host.tolist()]
    if len(query_lengths) != write_state_indices.numel():
        raise ValueError("Qwen3.5 query lengths must be sequence-scoped")
    conv_read, conv_write, ssm_read, ssm_write = _build_mega_prefill_indices(
        read_state_indices,
        write_state_indices,
        has_initial_state,
        checkpoint_stride,
    )
    plan = _MegaGdnPrefillPlan(
        conv_read=conv_read,
        conv_write=conv_write,
        ssm_read=ssm_read,
        ssm_write=ssm_write,
        num_matrices=_compute_mega_prefill_num_matrices(
            query_lengths,
            num_value_heads,
        ),
        num_sequences=len(query_lengths),
        num_tokens=sum(query_lengths),
    )
    context.layer_shared_cache[cache_key] = plan
    return plan


class NpuQwen3_5GatedDeltaNet(nn.Module):
    """NPU graph whose nodes match the native fused operator boundaries."""

    def __init__(
        self,
        cfg: Qwen3_5GatedDeltaNetConfig,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        if not math.isclose(
            cfg.rms_norm_eps,
            _MEGA_GDN_FIXED_RMS_NORM_EPS,
            rel_tol=1e-6,
            abs_tol=0.0,
        ):
            raise NotImplementedError(
                "Qwen3.5 MegaGdn fuses RMSNorm with a fixed epsilon; "
                f"cfg.rms_norm_eps={cfg.rms_norm_eps} is unsupported"
            )
        self.cfg = cfg
        self.layer_id = layer_id
        self.num_k_heads = cfg.linear_num_key_heads // cfg.tp_size
        self.num_v_heads = cfg.linear_num_value_heads // cfg.tp_size
        _validate_mega_gdn_head_geometry(self.num_k_heads, self.num_v_heads)
        self.key_head_dim = cfg.linear_key_head_dim
        self.value_head_dim = cfg.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.key_head_dim
        self.value_dim = self.num_v_heads * self.value_head_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = cfg.linear_conv_kernel_dim

        self.A_log = nn.Parameter(
            torch.empty(
                self.num_v_heads,
                dtype=torch.float32,
                device=device,
            )
        )
        # MegaGdn consumes dt_bias as FP32. Keep it in the required dtype once
        # instead of allocating a converted tensor on every forward.
        self.dt_bias = nn.Parameter(
            torch.empty(
                self.num_v_heads,
                dtype=torch.float32,
                device=device,
            )
        )
        self.norm_weight = nn.Parameter(
            torch.ones(
                self.value_head_dim,
                dtype=dtype,
                device=device,
            )
        )
        self.out_proj = RowParallelLinear(
            self.value_dim,
            cfg.hidden_size,
            cfg.tp_size,
            dtype=dtype,
            device=device,
        )
        # Keep one rank-local storage for each pair of projections. Prefill
        # consumes contiguous row views, while Decode runs two packed
        # projections and splits their outputs for the existing Mega operator.
        self.in_proj_qkvz = ColumnParallelLinear(
            cfg.hidden_size,
            self.conv_dim + self.value_dim,
            cfg.tp_size,
            dtype=dtype,
            device=device,
        )
        self.in_proj_ba = ColumnParallelLinear(
            cfg.hidden_size,
            2 * self.num_v_heads,
            cfg.tp_size,
            dtype=dtype,
            device=device,
        )
        self.conv1d_weight = nn.Parameter(
            torch.empty(
                self.conv_kernel_size,
                self.conv_dim,
                dtype=dtype,
                device=device,
            )
        )
        self.register_buffer(
            "conv1d_bias",
            torch.zeros(self.conv_dim, dtype=dtype, device=device),
            persistent=False,
        )

    def _input_projection_weight_views(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv_weight = self.in_proj_qkvz.weight[: self.conv_dim]
        z_weight = self.in_proj_qkvz.weight[self.conv_dim :]
        b_weight = self.in_proj_ba.weight[: self.num_v_heads]
        a_weight = self.in_proj_ba.weight[self.num_v_heads :]
        return qkv_weight, z_weight, b_weight, a_weight

    def load_weights(
        self,
        state: ScopedWeightLoader,
        context: ParallelLoadContext,
    ) -> None:
        qkv_weight, z_weight, b_weight, a_weight = self._input_projection_weight_views()
        state.copy(
            qkv_weight,
            shard_qkv_rows(
                state,
                state.get_tensor("in_proj_qkv.weight"),
                "in_proj_qkv.weight",
                self.cfg,
                context,
            ),
            "in_proj_qkv.weight",
        )
        for target, name in (
            (z_weight, "in_proj_z"),
            (b_weight, "in_proj_b"),
            (a_weight, "in_proj_a"),
        ):
            state.load_tensor(
                target,
                f"{name}.weight",
                dim=0,
                rank=context.tp_rank,
                world_size=context.tp_size,
            )

        local_conv = shard_qkv_rows(
            state,
            state.get_tensor("conv1d.weight").squeeze(1),
            "conv1d.weight",
            self.cfg,
            context,
        )
        state.copy(
            self.conv1d_weight,
            local_conv.transpose(0, 1).contiguous(),
            "conv1d.weight",
        )
        for name in ("A_log", "dt_bias"):
            state.load_tensor(
                getattr(self, name),
                name,
                dim=0,
                rank=context.tp_rank,
                world_size=context.tp_size,
            )
        state.load_tensor(self.norm_weight, "norm.weight")
        state.load_tensor(
            self.out_proj.weight,
            "out_proj.weight",
            dim=1,
            rank=context.tp_rank,
            world_size=context.tp_size,
        )

    def _project_prefill_inputs(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv_weight, z_weight, b_weight, a_weight = self._input_projection_weight_views()
        mixed_qkv = torch.nn.functional.linear(hidden, qkv_weight)
        z = torch.nn.functional.linear(hidden, z_weight).view(
            -1,
            self.num_v_heads,
            self.value_head_dim,
        )
        b = torch.nn.functional.linear(hidden, b_weight)
        a = torch.nn.functional.linear(hidden, a_weight)
        return mixed_qkv, a, b, z

    def _decode(
        self,
        hidden: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        read_state_indices: torch.Tensor,
        write_state_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = hidden.shape[0]
        if batch_size <= 0 or read_state_indices.shape != (batch_size,) or write_state_indices.shape != (batch_size,):
            raise ValueError("Qwen3.5 MegaGdnDecode requires one state slot per token")

        mixed_qkvz = self.in_proj_qkvz(hidden)
        mixed_ba = self.in_proj_ba(hidden)
        if mixed_qkvz.dtype != torch.bfloat16 or mixed_ba.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.5 MegaGdnDecode projections must be BF16")
        if mixed_qkvz.shape != (batch_size, self.conv_dim + self.value_dim):
            raise ValueError("Qwen3.5 MegaGdnDecode received an invalid packed QKVZ projection")
        if mixed_ba.shape != (batch_size, 2 * self.num_v_heads):
            raise ValueError("Qwen3.5 MegaGdnDecode received an invalid packed B/A projection")
        mixed_qkv, z = mixed_qkvz.split((self.conv_dim, self.value_dim), dim=-1)
        b, a = mixed_ba.split((self.num_v_heads, self.num_v_heads), dim=-1)
        z = z.view(batch_size, self.num_v_heads, self.value_head_dim)
        if self.conv_kernel_size != 4:
            raise NotImplementedError("Qwen3.5 MegaGdnDecode requires convolution width 4")
        if self.key_head_dim != 128 or self.value_head_dim != 128:
            raise NotImplementedError("Qwen3.5 MegaGdnDecode requires K/V head dimension 128")
        if self.num_k_heads <= 0 or self.num_k_heads > 16:
            raise NotImplementedError("Qwen3.5 MegaGdnDecode supports 1 to 16 local key heads")
        if self.num_k_heads & (self.num_k_heads - 1):
            raise NotImplementedError("Qwen3.5 MegaGdnDecode requires a power-of-two local key-head count")
        if self.num_v_heads % self.num_k_heads != 0:
            raise NotImplementedError("Qwen3.5 MegaGdnDecode requires Nv divisible by Nk")
        value_heads_per_key = self.num_v_heads // self.num_k_heads
        if value_heads_per_key < 1 or value_heads_per_key > 4:
            raise NotImplementedError("Qwen3.5 MegaGdnDecode supports at most four value heads per key head")
        if self.conv1d_weight.shape != (self.conv_kernel_size, self.conv_dim):
            raise ValueError("Qwen3.5 MegaGdnDecode received an invalid Conv weight")
        if self.conv1d_weight.dtype != torch.bfloat16 or conv_state.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.5 MegaGdnDecode Conv tensors must be BF16")
        if conv_state.dim() != 3 or conv_state.shape[1:] != (3, self.conv_dim):
            raise ValueError("Qwen3.5 MegaGdnDecode requires a three-token Conv cache")
        if ssm_state.dim() != 4 or ssm_state.shape != (
            conv_state.shape[0],
            self.num_v_heads,
            self.key_head_dim,
            self.value_head_dim,
        ):
            raise ValueError("Qwen3.5 MegaGdnDecode requires one SSM row per Conv slot")
        if ssm_state.dtype != torch.float32:
            raise NotImplementedError("Qwen3.5 MegaGdnDecode SSM cache must be FP32")
        if self.A_log.dtype != torch.float32 or self.dt_bias.dtype != torch.float32:
            raise ValueError("Qwen3.5 MegaGdnDecode A_log and dt_bias must be FP32")
        if self.norm_weight.dtype != torch.bfloat16:
            raise ValueError("Qwen3.5 MegaGdnDecode norm weight must be BF16")

        read_state_indices = read_state_indices.contiguous()
        write_state_indices = write_state_indices.contiguous()
        chunk_outputs = []
        for start in range(0, batch_size, _MEGA_GDN_MAX_DECODE_BATCH_SIZE):
            end = min(start + _MEGA_GDN_MAX_DECODE_BATCH_SIZE, batch_size)
            chunk_outputs.append(
                kernels.mega_gdn_decode(
                    mixed_qkv[start:end].contiguous(),
                    z[start:end].contiguous(),
                    b[start:end].contiguous(),
                    a[start:end].contiguous(),
                    self.conv1d_weight,
                    conv_state,
                    self.A_log,
                    self.dt_bias,
                    ssm_state,
                    read_state_indices[start:end].contiguous(),
                    write_state_indices[start:end].contiguous(),
                    self.norm_weight,
                    True,
                )
            )
        return chunk_outputs[0] if len(chunk_outputs) == 1 else torch.cat(chunk_outputs, dim=0)

    def _cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        cache = get_forward_context().layer_caches[self.layer_id]
        if cache.conv is None or cache.ssm is None:
            raise RuntimeError(f"linear-attention cache is missing for layer {self.layer_id}")
        if cache.conv.dim() != 3 or cache.conv.size(2) != self.conv_dim:
            raise ValueError("NPU Qwen3.5 conv cache must use [slot, state_len, dim]")
        return cache.conv, cache.ssm

    def _prefill(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        z: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        read_state_indices: torch.Tensor,
        write_state_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        metadata = get_forward_context().metadata
        query_lengths_host = metadata.q_seq_lens_host
        if query_lengths_host is None:
            raise RuntimeError("Qwen3.5 MegaGdnPrefill requires host query lengths")
        if query_lengths_host.device.type != "cpu":
            raise ValueError("Qwen3.5 host query lengths must reside on CPU")

        num_sequences = write_state_indices.numel()
        if cu_seqlens.dtype != torch.int32 or cu_seqlens.dim() != 1:
            raise ValueError("Qwen3.5 cu_seqlens must be a one-dimensional INT32 tensor")
        if cu_seqlens.numel() != num_sequences + 1:
            raise ValueError("Qwen3.5 cu_seqlens must contain B+1 entries")

        if mixed_qkv.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.5 MegaGdnPrefill supports BF16 only")
        if mixed_qkv.dim() != 2 or mixed_qkv.shape[1] != self.conv_dim:
            raise ValueError("Qwen3.5 MegaGdnPrefill received an invalid QKV projection")
        if a.shape != (mixed_qkv.shape[0], self.num_v_heads) or b.shape != a.shape:
            raise ValueError("Qwen3.5 MegaGdnPrefill received invalid A/B projections")
        if z.shape != (
            mixed_qkv.shape[0],
            self.num_v_heads,
            self.value_head_dim,
        ):
            raise ValueError("Qwen3.5 MegaGdnPrefill received an invalid Z projection")
        if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16 or z.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.5 MegaGdnPrefill projections must be BF16")
        if self.conv_kernel_size != 4:
            raise NotImplementedError("Qwen3.5 MegaGdnPrefill requires convolution width 4")
        if self.key_head_dim != 128 or self.value_head_dim != 128:
            raise NotImplementedError("Qwen3.5 MegaGdnPrefill requires K/V head dimension 128")
        if self.num_k_heads <= 0 or self.num_v_heads % self.num_k_heads != 0:
            raise NotImplementedError("Qwen3.5 MegaGdnPrefill requires Nv divisible by Nk")
        if self.num_v_heads not in _SUPPORTED_MEGA_PREFILL_HEAD_COUNTS:
            raise NotImplementedError(f"Qwen3.5 MegaGdnPrefill does not support {self.num_v_heads} local value heads")
        if self.conv1d_weight.shape != (self.conv_kernel_size, self.conv_dim):
            raise ValueError("Qwen3.5 MegaGdnPrefill received an invalid Conv weight")
        if self.conv1d_weight.dtype != torch.bfloat16 or conv_state.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.5 MegaGdnPrefill Conv tensors must be BF16")
        if conv_state.dim() != 3 or conv_state.shape[2] != self.conv_dim:
            raise ValueError("Qwen3.5 MegaGdnPrefill received an invalid Conv cache")
        if ssm_state.dim() != 4 or ssm_state.shape[1:] != (
            self.num_v_heads,
            self.key_head_dim,
            self.value_head_dim,
        ):
            raise ValueError("Qwen3.5 MegaGdnPrefill received an invalid SSM cache")
        if ssm_state.dtype != torch.float32:
            raise NotImplementedError("Qwen3.5 MegaGdnPrefill SSM cache must be FP32")
        if self.A_log.dtype != torch.float32 or self.dt_bias.dtype != torch.float32:
            raise ValueError("Qwen3.5 MegaGdnPrefill A_log and dt_bias must be FP32")
        if self.norm_weight.dtype != torch.bfloat16:
            raise ValueError("Qwen3.5 MegaGdnPrefill norm weight must be BF16")
        if conv_state.shape[0] <= 0 or ssm_state.shape[0] % conv_state.shape[0] != 0:
            raise ValueError("Qwen3.5 SSM cache rows must be divisible by Conv cache slots")
        checkpoint_stride = ssm_state.shape[0] // conv_state.shape[0]
        if conv_state.shape[1] != checkpoint_stride + 2:
            raise ValueError("Qwen3.5 Conv cache history must equal checkpoint stride + 2")

        plan = _get_or_build_mega_prefill_plan(
            read_state_indices,
            write_state_indices,
            has_initial_state,
            query_lengths_host,
            checkpoint_stride,
            self.num_v_heads,
        )
        if plan.num_tokens != mixed_qkv.shape[0]:
            raise ValueError("Qwen3.5 packed token count does not match host query lengths")
        return kernels.mega_gdn_prefill(
            mixed_qkv.contiguous(),
            b.contiguous(),
            a.contiguous(),
            z.contiguous(),
            self.conv1d_weight,
            conv_state,
            self.A_log,
            self.dt_bias,
            plan.conv_read,
            plan.conv_write,
            plan.ssm_read,
            plan.ssm_write,
            ssm_state,
            cu_seqlens.contiguous(),
            self.norm_weight,
            plan.num_matrices,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        metadata = get_forward_context().metadata
        read_state_indices, write_state_indices = resolve_linear_state_io_indices(metadata)
        if read_state_indices is None or write_state_indices is None:
            raise RuntimeError("linear-state read/write indices are required by Qwen3.5")
        read_state_indices = read_state_indices.to(device=hidden.device, dtype=torch.int32)
        write_state_indices = write_state_indices.to(device=hidden.device, dtype=torch.int32)

        conv_state, ssm_state = self._cache()
        if metadata.is_prefill or metadata.is_chunked_prefill:
            has_initial_state = metadata.has_initial_state
            if has_initial_state is None:
                raise RuntimeError("has_initial_state is required by Qwen3.5 prefill")
            cu_seqlens = metadata.q_cu_seq_lens
            if cu_seqlens is None:
                cu_seqlens = torch.arange(
                    write_state_indices.numel() + 1,
                    dtype=torch.int32,
                    device=hidden.device,
                )
            mixed_qkv, a, b, z = self._project_prefill_inputs(hidden)
            output = self._prefill(
                mixed_qkv,
                a,
                b,
                z,
                conv_state,
                ssm_state,
                read_state_indices,
                write_state_indices,
                has_initial_state.to(device=hidden.device, dtype=torch.bool),
                cu_seqlens,
            )
        else:
            output = self._decode(
                hidden,
                conv_state,
                ssm_state,
                read_state_indices,
                write_state_indices,
            )
        return self.out_proj(output.reshape(-1, self.value_dim))
