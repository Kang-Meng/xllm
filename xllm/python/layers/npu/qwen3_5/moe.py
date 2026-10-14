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

from dataclasses import replace
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python import distributed, kernels
from xllm.python.layers.gated_mlp import GatedMLP
from xllm.python.layers.moe_parallel import (
    dp_gather_tokens,
    ep_apply_local,
    ep_gather_tokens,
    ep_reduce_scatter,
    reduce_and_scatter,
)
from xllm.python.layers.npu.mega_moe_metadata import MegaMoeMetadata
from xllm.python.layers.npu.qwen3_5.moe_metadata import MEGA_MOE_MAX_TOKENS, Qwen3_5MoeMetadata
from xllm.python.layers.qwen3_5.common import Qwen3_5MoEConfig
from xllm.python.layers.sequence_parallel_metadata import SpMetadata
from xllm.python.model_executor.forward_context import get_execution_context
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader


def _pad_token_rows(tensor: torch.Tensor, token_capacity: int) -> torch.Tensor:
    token_count = tensor.shape[0]
    if token_count > token_capacity:
        raise RuntimeError(f"MegaMoe input has {token_count} rows, exceeding token capacity {token_capacity}")
    if token_count == token_capacity:
        return tensor.contiguous()
    return F.pad(tensor, (0, 0, 0, token_capacity - token_count))


class _NpuQwen3_5Experts(nn.Module):
    """BF16 routed experts in NPU grouped-matmul layout."""

    def __init__(
        self,
        cfg: Qwen3_5MoEConfig,
        dtype: torch.dtype,
        device: torch.device,
        reduce_results: bool,
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
        self.ep_rank = cfg.ep_rank
        self.tp_size = cfg.tp_size
        self.tp_rank = cfg.tp_rank
        self.reduce_results = reduce_results
        self._mega_moe_ccl_buffer_size = cfg.mega_moe_ccl_buffer_size
        self._mega_moe_num_max_tokens_per_rank = cfg.mega_moe_num_max_tokens_per_rank
        self._mega_moe_enabled = cfg.enable_mega_moe
        self._sequence_parallel_enabled = cfg.sequence_parallel_enabled
        if self._sequence_parallel_enabled:
            if self.moe_tp_size != 1 or self.ep_size != self.tp_size * self.dp_size:
                raise ValueError("Qwen3.5 SP requires direct EP topology: MoE TP must be 1 and EP must equal TP * DP")
            if self.ep_rank != self.dp_rank * self.tp_size + self.tp_rank:
                raise ValueError("Qwen3.5 SP EP rank mapping must match the local DP/TP coordinates")
            if not self.reduce_results:
                raise ValueError("Qwen3.5 SP experts must reduce their partial outputs")
        self.register_buffer(
            "_mega_moe_context",
            cfg.mega_moe_context,
            persistent=False,
        )
        if self._mega_moe_enabled and cfg.mega_moe_context is None:
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

    def _forward_token_owner_mega_moe(
        self,
        hidden_states: torch.Tensor,
        metadata: MegaMoeMetadata,
    ) -> torch.Tensor:
        active_token_mask = metadata.active_token_mask
        token_capacity = active_token_mask.shape[0]
        if self.tp_rank == 0:
            topk_weights, topk_ids = self._route(self.gate(hidden_states))
            owner_input = _pad_token_rows(hidden_states, token_capacity)
            topk_weights = _pad_token_rows(topk_weights.to(torch.float32), token_capacity)
            topk_ids = _pad_token_rows(topk_ids.to(torch.int32), token_capacity)
        else:
            owner_input = metadata.dummy_input
            topk_weights = metadata.dummy_topk_weights
            topk_ids = metadata.dummy_topk_ids
            if owner_input is None or topk_weights is None or topk_ids is None:
                raise RuntimeError("MegaMoe non-owner dummy inputs are unavailable")

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
            output = torch.empty_like(hidden_states)
        distributed.broadcast_(output, 0, "tp")
        return output

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

    def _forward_ep_level1_for_sp(self, hidden_states: torch.Tensor, sp_metadata: SpMetadata) -> torch.Tensor:
        full_hidden = ep_gather_tokens(hidden_states, sp_metadata)
        topk_weights, topk_ids = self._route(self.gate(full_hidden))
        output = kernels.grouped_moe_bf16(
            full_hidden,
            topk_weights,
            topk_ids,
            self.w13,
            self.w2,
            self.num_experts,
            self.start_expert,
            self.local_experts,
        )
        return ep_reduce_scatter(output, sp_metadata)

    def _forward_sp_mega_moe(
        self,
        hidden_states: torch.Tensor,
        moe_metadata: Qwen3_5MoeMetadata,
    ) -> torch.Tensor:
        active_token_mask = moe_metadata.active_token_mask
        token_capacity = active_token_mask.shape[0]
        topk_weights, topk_ids = self._route(self.gate(hidden_states))
        dispatch_input = _pad_token_rows(hidden_states, token_capacity)
        topk_weights = _pad_token_rows(topk_weights.to(torch.float32), token_capacity)
        topk_ids = _pad_token_rows(topk_ids.to(torch.int32), token_capacity)

        dispatch_output = kernels.mega_moe(
            cast(torch.Tensor, self._mega_moe_context),
            dispatch_input,
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
        return dispatch_output[: hidden_states.shape[0]].contiguous()

    def validate_sp(
        self,
        sp_metadata: SpMetadata,
        *,
        device: torch.device,
    ) -> None:
        """Validate SP metadata without reading device values on the host."""
        moe_metadata = get_execution_context(Qwen3_5MoeMetadata)
        if moe_metadata is None:
            raise RuntimeError("Qwen3.5 SP execution metadata is unavailable")
        mask = moe_metadata.active_token_mask
        if mask.ndim != 1 or mask.dtype != torch.int8 or not mask.is_contiguous():
            raise ValueError("Qwen3.5 SP active_token_mask must be contiguous int8[C]")
        if mask.device != device:
            raise ValueError("Qwen3.5 SP active_token_mask and hidden states must share a device")
        if mask.shape[0] < sp_metadata.local_rows:
            raise ValueError("Qwen3.5 SP mask capacity must cover the local SP rows")
        if sp_metadata.ep_layout is None:
            raise RuntimeError("Qwen3.5 SP EP token layout is unavailable")

    def forward(self, hidden_states: torch.Tensor, sp_metadata: SpMetadata | None = None) -> torch.Tensor:
        if self._sequence_parallel_enabled:
            sp_metadata = cast(SpMetadata, sp_metadata)
            if hidden_states.shape != (sp_metadata.local_rows, self.w13.shape[1]):
                raise ValueError("Qwen3.5 SP input must contain the local SP hidden rows")
            metadata = cast(Qwen3_5MoeMetadata, get_execution_context(Qwen3_5MoeMetadata))
            token_capacity = metadata.active_token_mask.shape[0]
            token_limit = min(MEGA_MOE_MAX_TOKENS, self._mega_moe_num_max_tokens_per_rank)
            mega_moe_capacity_available = token_capacity <= token_limit
            if self._mega_moe_enabled and mega_moe_capacity_available:
                return self._forward_sp_mega_moe(hidden_states, metadata)
            return self._forward_ep_level1_for_sp(hidden_states, sp_metadata)
        if self._mega_moe_enabled:
            metadata = get_execution_context(MegaMoeMetadata)
            if metadata is None:
                raise RuntimeError("MegaMoe execution metadata is unavailable")
            token_capacity = metadata.active_token_mask.shape[0]
            token_limit = min(MEGA_MOE_MAX_TOKENS, self._mega_moe_num_max_tokens_per_rank)
            if token_capacity <= token_limit:
                return self._forward_token_owner_mega_moe(hidden_states, metadata)
        return self._forward_ep_level1(hidden_states)


class NpuQwen3_5SparseMoEBlock(nn.Module):
    """NPU Qwen3.5 routed and shared experts with topology-safe reductions."""

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
        self._sequence_parallel_enabled = cfg.sequence_parallel_enabled
        shared_tp_size = 1 if self._sequence_parallel_enabled else cfg.tp_size
        self._fused_reductions_enabled = (
            not self._sequence_parallel_enabled
            and cfg.dp_size == 1
            and cfg.ep_size == 1
            and cfg.tp_size == cfg.moe_tp_size
            and cfg.tp_size > 1
        )
        self.shared_expert = GatedMLP(
            cfg.hidden_size,
            cfg.shared_expert_intermediate_size,
            shared_tp_size,
            dtype,
            device,
            reduce_results=not self._fused_reductions_enabled,
        )
        self.experts = _NpuQwen3_5Experts(
            cfg,
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

    def _pack_gate_up(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat((gate, up), dim=1).transpose(1, 2).contiguous()

    def _pack_down(self, down: torch.Tensor) -> torch.Tensor:
        return down.transpose(1, 2).contiguous()

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
        state.copy(
            self.experts.w13,
            self._pack_gate_up(gate, up),
            "experts.gate_up_proj",
        )

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
        state.copy(self.experts.w2, self._pack_down(down), "experts.down_proj")
        shared_context = replace(context, tp_size=1, tp_rank=0) if self._sequence_parallel_enabled else context
        self.shared_expert.load_weights(state.with_prefix("shared_expert."), shared_context)

    def process_weights_after_loading(self) -> None:
        self.shared_expert.process_weights_after_loading()

    def validate_sp(
        self,
        sp_metadata: SpMetadata,
        *,
        device: torch.device,
    ) -> None:
        """Validate SP metadata before the decoder starts stateful work."""
        self.experts.validate_sp(sp_metadata, device=device)

    def _forward_shared_expert(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.shared_expert(hidden) * torch.sigmoid(self.shared_expert_gate(hidden))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self._sequence_parallel_enabled:
            sp_metadata = cast(SpMetadata, get_execution_context(SpMetadata))
            routed = self.experts(hidden, sp_metadata=sp_metadata)
            shared = ep_apply_local(hidden, sp_metadata, self._forward_shared_expert)
        else:
            routed = self.experts(hidden)
            shared = self._forward_shared_expert(hidden)
        output = routed + shared
        if self._fused_reductions_enabled:
            distributed.tp_all_reduce(output)
        return output
