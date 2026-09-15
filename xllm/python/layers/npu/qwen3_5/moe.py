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

"""NPU-owned Qwen3.5 sparse MoE composition."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from xllm.python import distributed, kernels
from xllm.python.layers.moe_dp import dp_gather_tokens, reduce_and_scatter
from xllm.python.layers.npu.mega_moe_context import (
    MegaMoeLayerContext,
    MegaMoeLayerSpec,
    get_mega_moe_layer_context,
)
from xllm.python.layers.qwen3_5.common import Qwen3_5MoEConfig
from xllm.python.layers.qwen3_5.moe import Qwen3_5SparseMoEBlockBase


class _NpuQwen3_5Experts(nn.Module):
    """BF16 routed experts in NPU grouped-matmul layout."""

    def __init__(
        self,
        cfg: Qwen3_5MoEConfig,
        dtype: torch.dtype,
        device: torch.device,
        reduce_results: bool,
        layer_id: int,
    ) -> None:
        super().__init__()
        if dtype != torch.bfloat16:
            raise NotImplementedError("NPU Qwen3.5 routed experts currently support BF16 only")
        local_experts = cfg.num_experts // cfg.ep_size
        local_intermediate = cfg.moe_intermediate_size // cfg.moe_tp_size
        self.top_k = cfg.num_experts_per_tok
        self.renormalize = cfg.norm_topk_prob
        self.num_experts = cfg.num_experts
        self.local_experts = local_experts
        self.start_expert = cfg.ep_rank * local_experts
        self.moe_tp_size = cfg.moe_tp_size
        self.ep_size = cfg.ep_size
        self.dp_size = cfg.dp_size
        self.dp_rank = cfg.dp_rank
        self.tp_rank = cfg.tp_rank
        self.reduce_results = reduce_results
        self._mega_moe_ccl_buffer_size = cfg.mega_moe_ccl_buffer_size
        self._mega_moe_num_max_tokens_per_rank = cfg.mega_moe_num_max_tokens_per_rank
        self.mega_moe_execution_spec = (
            MegaMoeLayerSpec(
                layer_id=layer_id,
                hidden_size=cfg.hidden_size,
                top_k=cfg.num_experts_per_tok,
                token_limit=cfg.mega_moe_num_max_tokens_per_rank,
                dtype=dtype,
                device=device,
                dp_size=cfg.dp_size,
                dp_rank=cfg.dp_rank,
                tp_rank=cfg.tp_rank,
            )
            if cfg.enable_mega_moe
            else None
        )
        self.register_buffer(
            "_mega_moe_context",
            cfg.mega_moe_context,
            persistent=False,
        )
        if self.mega_moe_execution_spec is not None and cfg.mega_moe_context is None:
            raise ValueError("MegaMoe token ownership requires a communication context")

        self.gate = nn.Linear(
            cfg.hidden_size,
            cfg.num_experts,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.w13 = nn.Parameter(
            torch.empty(
                local_experts,
                cfg.hidden_size,
                2 * local_intermediate,
                dtype=dtype,
                device=device,
            )
        )
        self.w2 = nn.Parameter(
            torch.empty(
                local_experts,
                local_intermediate,
                cfg.hidden_size,
                dtype=dtype,
                device=device,
            )
        )

    def _require_mega_moe_context(self) -> torch.Tensor:
        context = self._mega_moe_context
        if context is None:
            raise RuntimeError("MegaMoe communication context is unavailable")
        return context

    def _route(
        self,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return kernels.moe_fused_topk(
            router_logits,
            self.top_k,
            self.renormalize,
            "softmax",
        )

    def _forward_ep_level1(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gathered_states, scatter_state = dp_gather_tokens(hidden_states, self.dp_size, self.dp_rank)
        topk_weights, topk_ids = self._route(self.gate(gathered_states))
        output = kernels.grouped_moe_bf16(
            gathered_states,
            topk_weights,
            topk_ids,
            self.w13,
            self.w2,
            self.num_experts,
            self.start_expert,
            self.local_experts,
        )
        return reduce_and_scatter(
            output,
            scatter_state,
            reduce_results=self.reduce_results,
            moe_tp_size=self.moe_tp_size,
            ep_size=self.ep_size,
        )

    def _forward_mega_moe(
        self,
        hidden_states: torch.Tensor,
        layer_context: MegaMoeLayerContext,
        active_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.tp_rank == 0:
            topk_weights, topk_ids = self._route(self.gate(hidden_states))
            topk_weights = topk_weights.to(torch.float32).contiguous()
            topk_ids = topk_ids.to(torch.int32).contiguous()

            if layer_context.input_buffer is None:
                owner_input = hidden_states.contiguous()
            else:
                owner_input = layer_context.input_buffer
                padded_topk_weights = cast(torch.Tensor, layer_context.topk_weights_buffer)
                padded_topk_ids = cast(torch.Tensor, layer_context.topk_ids_buffer)
                local_tokens = hidden_states.shape[0]
                owner_input[:local_tokens].copy_(hidden_states)
                padded_topk_weights[:local_tokens].copy_(topk_weights)
                padded_topk_ids[:local_tokens].copy_(topk_ids)
                topk_weights = padded_topk_weights
                topk_ids = padded_topk_ids
        else:
            owner_input = cast(torch.Tensor, layer_context.input_buffer)
            topk_weights = cast(torch.Tensor, layer_context.topk_weights_buffer)
            topk_ids = cast(torch.Tensor, layer_context.topk_ids_buffer)

        owner_output = kernels.mega_moe(
            self._require_mega_moe_context(),
            owner_input,
            topk_ids,
            topk_weights,
            self.w13,
            self.w2,
            None,
            None,
            self.num_experts,
            self.ep_size,
            self._mega_moe_ccl_buffer_size,
            self._mega_moe_num_max_tokens_per_rank,
            active_token_mask,
        )

        if self.tp_rank == 0:
            output = owner_output[: hidden_states.shape[0]].contiguous()
        else:
            output = cast(torch.Tensor, layer_context.output_buffer)
        distributed.broadcast_(output, 0, "tp")
        return output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.mega_moe_execution_spec is not None:
            mega_moe_context, layer_context = get_mega_moe_layer_context(self.mega_moe_execution_spec.layer_id)
            if layer_context is not None:
                return self._forward_mega_moe(
                    hidden_states,
                    layer_context,
                    mega_moe_context.active_token_mask,
                )
        return self._forward_ep_level1(hidden_states)


class NpuQwen3_5SparseMoEBlock(Qwen3_5SparseMoEBlockBase):
    """NPU Qwen3.5 routed and shared experts with topology-safe reductions."""

    def __init__(
        self,
        cfg: Qwen3_5MoEConfig,
        dtype: torch.dtype,
        device: torch.device,
        *,
        layer_id: int = 0,
    ) -> None:
        super().__init__(cfg, dtype, device)
        self.experts = _NpuQwen3_5Experts(
            cfg,
            dtype,
            device,
            reduce_results=not self.fuse_reductions,
            layer_id=layer_id,
        )

    def _pack_gate_up(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat((gate, up), dim=1).transpose(1, 2).contiguous()

    def _pack_down(self, down: torch.Tensor) -> torch.Tensor:
        return down.transpose(1, 2).contiguous()
