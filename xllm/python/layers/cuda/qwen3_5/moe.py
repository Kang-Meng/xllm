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

"""CUDA-owned Qwen3.5 sparse MoE composition."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python import distributed
from xllm.python.layers.fused_moe import FusedMoE
from xllm.python.layers.gated_mlp import GatedMLP
from xllm.python.layers.qwen3_5.common import Qwen3_5MoEConfig
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader


class CudaQwen3_5SparseMoEBlock(nn.Module):
    """CUDA expert graph with CUTLASS/Triton-native weight ordering."""

    def __init__(
        self,
        cfg: Qwen3_5MoEConfig,
        dtype: torch.dtype,
        device: torch.device,
        *,
        layer_id: int = 0,
    ) -> None:
        super().__init__()
        del layer_id
        self._fused_reductions_enabled = (
            cfg.dp_size == 1 and cfg.ep_size == 1 and cfg.tp_size == cfg.moe_tp_size and cfg.tp_size > 1
        )
        self.shared_expert = GatedMLP(
            cfg.hidden_size,
            cfg.shared_expert_intermediate_size,
            cfg.tp_size,
            dtype,
            device,
            reduce_results=not self._fused_reductions_enabled,
        )
        self.shared_expert_gate = nn.Linear(
            cfg.hidden_size,
            1,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.experts = FusedMoE(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.moe_intermediate_size,
            num_experts=cfg.num_experts,
            top_k=cfg.num_experts_per_tok,
            renormalize=cfg.norm_topk_prob,
            moe_tp_size=cfg.moe_tp_size,
            moe_tp_rank=cfg.moe_tp_rank,
            ep_size=cfg.ep_size,
            ep_rank=cfg.ep_rank,
            dp_size=cfg.dp_size,
            dp_rank=cfg.dp_rank,
            dtype=dtype,
            device=device,
            reduce_results=not self._fused_reductions_enabled,
        )

    def _pack_gate_up(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return torch.cat((up, gate), dim=1)

    def load_weights(
        self,
        state: ScopedWeightLoader,
        context: ParallelLoadContext,
    ) -> None:
        state.load_tensor(self.experts.gate.weight, "gate.weight")
        state.load_tensor(
            self.shared_expert_gate.weight,
            "shared_expert_gate.weight",
        )

        gate_up = state.shard_value(
            state.get_tensor("experts.gate_up_proj"),
            "experts.gate_up_proj",
            0,
            context.ep_rank,
            context.ep_size,
        )
        gate, up = gate_up.chunk(2, dim=1)
        gate = state.shard_value(
            gate,
            "experts.gate_up_proj[gate]",
            1,
            context.moe_tp_rank,
            context.moe_tp_size,
        )
        up = state.shard_value(
            up,
            "experts.gate_up_proj[up]",
            1,
            context.moe_tp_rank,
            context.moe_tp_size,
        )
        state.copy(self.experts.w13, self._pack_gate_up(gate, up), "experts.gate_up_proj")

        down = state.shard_value(
            state.get_tensor("experts.down_proj"),
            "experts.down_proj",
            0,
            context.ep_rank,
            context.ep_size,
        )
        down = state.shard_value(
            down,
            "experts.down_proj",
            2,
            context.moe_tp_rank,
            context.moe_tp_size,
        )
        state.copy(self.experts.w2, down, "experts.down_proj")
        self.shared_expert.load_weights(state.with_prefix("shared_expert."), context)

    def process_weights_after_loading(self) -> None:
        self.shared_expert.process_weights_after_loading()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        routed = self.experts(hidden)
        shared = self.shared_expert(hidden)
        output = routed + shared * torch.sigmoid(self.shared_expert_gate(hidden))
        if self._fused_reductions_enabled:
            distributed.tp_all_reduce(output)
        return output
