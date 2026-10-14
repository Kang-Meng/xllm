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

"""NPU-owned Qwen3.5 decoder layer."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python.layers.gated_mlp import GatedMLP
from xllm.python.layers.npu.layernorm import NpuGemmaRMSNorm
from xllm.python.layers.npu.qwen3_5.attention import NpuQwen3_5Attention
from xllm.python.layers.npu.qwen3_5.gated_delta_net import NpuQwen3_5GatedDeltaNet
from xllm.python.layers.npu.qwen3_5.moe import NpuQwen3_5SparseMoEBlock
from xllm.python.layers.qwen3_5.common import (
    PartialRotaryEmbedding,
    Qwen3_5DecoderConfig,
)
from xllm.python.layers.sequence_parallel import sp_all_gather, sp_reduce_scatter
from xllm.python.layers.sequence_parallel_metadata import SpMetadata
from xllm.python.model_executor.forward_context import get_execution_context, wait_for_layer_load
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader


class NpuQwen3_5DecoderLayer(nn.Module):
    """NPU Qwen3.5 decoder with NPU-owned child implementations."""

    supports_sequence_parallel = True
    attention_cls = NpuQwen3_5Attention
    gated_delta_net_cls = NpuQwen3_5GatedDeltaNet
    sparse_moe_cls = NpuQwen3_5SparseMoEBlock
    normalization_cls: type[nn.Module] = NpuGemmaRMSNorm

    def __init__(
        self,
        cfg: Qwen3_5DecoderConfig,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        rotary: PartialRotaryEmbedding,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.layer_type = cfg.layer_types[layer_id]
        self.input_layernorm = self.normalization_cls(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        if self.layer_type == "full_attention":
            self.self_attn = self.attention_cls(
                cfg,
                layer_id,
                dtype,
                device,
                rotary,
            )
        elif self.layer_type == "linear_attention":
            self.linear_attn = self.gated_delta_net_cls(
                cfg,
                layer_id,
                dtype,
                device,
            )
        else:
            raise ValueError(f"unsupported Qwen3.5 layer type: {self.layer_type}")
        self.post_attention_layernorm = self.normalization_cls(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        if cfg.is_moe_layer(layer_id):
            self.mlp = self.sparse_moe_cls(
                cfg,
                dtype,
                device,
                layer_id=layer_id,
            )
        else:
            self.mlp = GatedMLP(
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.tp_size,
                dtype,
                device,
                reduce_results=not cfg.sequence_parallel_enabled,
            )

    def validate_sp(
        self,
        sp_metadata: SpMetadata,
        *,
        device: torch.device,
    ) -> None:
        """Validate the stateless SP contract before layer collectives."""
        self.mlp.validate_sp(sp_metadata, device=device)

    def load_weights(
        self,
        weights: ScopedWeightLoader,
        context: ParallelLoadContext,
    ) -> None:
        weights.load_tensor(
            self.input_layernorm.weight,
            "input_layernorm.weight",
        )
        weights.load_tensor(
            self.post_attention_layernorm.weight,
            "post_attention_layernorm.weight",
        )
        if self.layer_type == "full_attention":
            self.self_attn.load_weights(
                weights.with_prefix("self_attn."),
                context,
            )
        else:
            self.linear_attn.load_weights(
                weights.with_prefix("linear_attn."),
                context,
            )
        self.mlp.load_weights(weights.with_prefix("mlp."), context)
        self.mlp.process_weights_after_loading()

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        wait_for_layer_load(self.layer_id)
        if self.cfg.sequence_parallel_enabled:
            return self._forward_sequence_parallel(hidden, residual, positions)
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)
        if self.layer_type == "full_attention":
            hidden = self.self_attn(positions, hidden)
        else:
            hidden = self.linear_attn(hidden)
        hidden, residual = self.post_attention_layernorm(hidden, residual)
        return self.mlp(hidden), residual

    def _forward_sequence_parallel(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        layout = get_execution_context(SpMetadata)
        expected_shape = (layout.local_rows, self.cfg.hidden_size)
        if hidden.shape != expected_shape:
            raise ValueError(f"Qwen3.5 hidden states must match the local SP row count: expected {expected_shape}")
        if residual is not None and (residual.shape != hidden.shape or residual.device != hidden.device):
            raise ValueError("Qwen3.5 residual must share the hidden-state layout and device")

        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)

        attention_input = sp_all_gather(hidden, layout)
        if self.layer_type == "full_attention":
            hidden = self.self_attn(positions, attention_input)
        else:
            hidden = self.linear_attn(attention_input)
        hidden, residual = self.post_attention_layernorm(hidden, residual)
        if self.cfg.is_moe_layer(self.layer_id):
            return self.mlp(hidden), residual

        # GatedMLP keeps TP-sharded gate/up and down weights.  Its down
        # projection must see every token row before it can be reduced across
        # TP ranks; reduce-scatter restores the local SP token ownership.
        dense_input = sp_all_gather(hidden, layout)
        dense_partial = self.mlp(dense_input)
        return sp_reduce_scatter(dense_partial, layout), residual
