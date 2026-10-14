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

"""NPU-specific Qwen3.5 full-attention implementation."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.layers.attention import Attention
from xllm.python.layers.linear import ColumnParallelLinear, RowParallelLinear
from xllm.python.layers.npu.layernorm import NpuGemmaRMSNorm
from xllm.python.layers.qwen3_5.common import (
    PartialRotaryEmbedding,
    Qwen3_5AttentionConfig,
)
from xllm.python.layers.sequence_parallel import sp_row_parallel_projection
from xllm.python.layers.sequence_parallel_metadata import SpMetadata
from xllm.python.model_executor.forward_context import get_execution_context, get_forward_context_or_none
from xllm.python.model_loader import (
    ParallelLoadContext,
    ScopedWeightLoader,
    load_gqa_fused_attention,
)


class NpuQwen3_5Attention(nn.Module):
    supports_sequence_parallel = True
    normalization_cls: type[nn.Module] = NpuGemmaRMSNorm

    def __init__(
        self,
        cfg: Qwen3_5AttentionConfig,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        rotary: PartialRotaryEmbedding,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.num_heads, self.num_kv_heads = cfg.head_split()
        self.head_dim = cfg.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        rotary_dim = int(self.head_dim * cfg.partial_rotary_factor)
        is_npu = device.type in ("npu", "privateuseone")
        self._fused_qkv_enabled = (
            is_npu
            and dtype == torch.bfloat16
            and cfg.attn_output_gate
            and len(cfg.rope_scaling_mrope_section) == 3
            and rotary_dim > 0
            and kernels.has_split_qkv_rmsnorm_mrope_specialization(
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )
        )
        self._weights_reordered = False
        if self._fused_qkv_enabled:
            gather_pattern = kernels.build_split_qkv_rmsnorm_mrope_gather_pattern(
                rotary_dim,
                cfg.rope_scaling_mrope_section,
                cfg.rope_scaling_mrope_interleaved,
                device,
            )
        else:
            gather_pattern = None
        self.register_buffer(
            "_mrope_gather_pattern",
            gather_pattern,
            persistent=False,
        )
        q_multiplier = 2 if cfg.attn_output_gate else 1
        self.qkv_proj = ColumnParallelLinear(
            cfg.hidden_size,
            q_multiplier * self.q_size + 2 * self.kv_size,
            cfg.tp_size,
            bias=cfg.attention_bias,
            dtype=dtype,
            device=device,
        )
        self.o_proj = RowParallelLinear(
            self.q_size,
            cfg.hidden_size,
            cfg.tp_size,
            bias=cfg.attention_bias,
            dtype=dtype,
            device=device,
        )
        self.q_norm = self.normalization_cls(
            self.head_dim,
            cfg.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        self.k_norm = self.normalization_cls(
            self.head_dim,
            cfg.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        self.rotary = rotary
        self.attn = Attention(
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.head_dim**-0.5,
            0,
            layer_id,
        )

    def _finish_loading(self) -> None:
        self.o_proj.process_weights_after_loading()
        if not self._fused_qkv_enabled or self._weights_reordered:
            return

        with torch.no_grad():
            qg_rows = self.qkv_proj.weight[: 2 * self.q_size]
            hidden_size = self.qkv_proj.weight.shape[1]
            qg_by_head = qg_rows.view(
                self.num_heads,
                2 * self.head_dim,
                hidden_size,
            )
            query = qg_by_head[:, : self.head_dim]
            gate = qg_by_head[:, self.head_dim :]
            qg_rows.copy_(
                torch.cat(
                    (
                        query.reshape(self.q_size, hidden_size),
                        gate.reshape(self.q_size, hidden_size),
                    ),
                    dim=0,
                )
            )
            if self.qkv_proj.bias is not None:
                qg_bias = self.qkv_proj.bias[: 2 * self.q_size]
                qg_bias_by_head = qg_bias.view(
                    self.num_heads,
                    2 * self.head_dim,
                )
                query_bias = qg_bias_by_head[:, : self.head_dim]
                gate_bias = qg_bias_by_head[:, self.head_dim :]
                qg_bias.copy_(
                    torch.cat(
                        (
                            query_bias.reshape(self.q_size),
                            gate_bias.reshape(self.q_size),
                        ),
                        dim=0,
                    )
                )
            self.q_norm.weight.add_(1.0)
            self.k_norm.weight.add_(1.0)
        self._weights_reordered = True

    def load_weights(
        self,
        state: ScopedWeightLoader,
        context: ParallelLoadContext,
    ) -> None:
        self._weights_reordered = False
        load_gqa_fused_attention(self, state, context, self.cfg.n_kv_heads)
        self._finish_loading()

    def _get_mrope_cos_sin(self, positions: torch.Tensor) -> torch.Tensor:
        context = get_forward_context_or_none()
        if context is None:
            return self.rotary.build_mrope_cos_sin(positions)

        cache_key = (__name__, "mrope_cos_sin", id(self.rotary))
        cached = context.layer_shared_cache.get(cache_key)
        if cached is None:
            cached = self.rotary.build_mrope_cos_sin(positions)
            context.layer_shared_cache[cache_key] = cached
        return cast(torch.Tensor, cached)

    def _project_output(self, output: torch.Tensor) -> torch.Tensor:
        if self.cfg.sequence_parallel_enabled:
            sp_layout = get_execution_context(SpMetadata)
            return sp_row_parallel_projection(self.o_proj, output, sp_layout)
        return self.o_proj(output)

    def _forward_standard(
        self,
        positions: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden)
        if self.cfg.attn_output_gate:
            q_gate, k, v = qkv.split(
                [2 * self.q_size, self.kv_size, self.kv_size],
                dim=-1,
            )
            q_gate = q_gate.view(-1, self.num_heads, 2 * self.head_dim)
            q, gate = q_gate.chunk(2, dim=-1)
        else:
            q, k, v = qkv.split(
                [self.q_size, self.kv_size, self.kv_size],
                dim=-1,
            )
            q = q.view(-1, self.num_heads, self.head_dim)
            gate = None
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rotary(positions, q).reshape(-1, self.q_size)
        k = self.rotary(positions, k).reshape(-1, self.kv_size)
        output = self.attn(q, k, v)
        if gate is not None:
            output = output * torch.sigmoid(gate.reshape(-1, self.q_size))
        return self._project_output(output)

    def forward(
        self,
        positions: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        if not self._fused_qkv_enabled:
            return self._forward_standard(positions, hidden)

        mrope_cos_sin = self._get_mrope_cos_sin(positions)
        qkvg = self.qkv_proj(hidden)
        query, key, value, gate = kernels.split_qkv_rmsnorm_mrope(
            qkvg,
            self.q_norm.weight,
            self.k_norm.weight,
            mrope_cos_sin,
            self._mrope_gather_pattern,
            self.cfg.rms_norm_eps,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )
        num_tokens = qkvg.shape[0]
        output = self.attn(
            query.view(num_tokens, self.q_size),
            key.view(num_tokens, self.kv_size),
            value.view(num_tokens, self.kv_size),
        )
        output = output * torch.sigmoid(gate.view(num_tokens, self.q_size))
        return self._project_output(output)
