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

"""GLM5.2 source-owned AllToAllV EP, independent of attention/indexer code.

Attention-TP ranks hold identical CP-local rows. TokenParallelLayout assigns
one TP source owner; AllToAllV sends its selected tokens to expert owners and
returns complete routed results to that source. Shared experts have MoE-TP1
weights and are evaluated once on source rows, without TP/EP sum. Only the
TP token all-gather restores the attention layout. CP zigzag/prefix padding
is excluded by the HCCL token mask, not by changing expert IDs or weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from scripts.logger import logger
from xllm.python import distributed, kernels
from xllm.python.layers.moe import MoECore
from xllm.python.layers.moe_parallel import TokenParallelLayout
from xllm.python.layers.npu.glm5_2_metadata import Glm52EpMetadata
from xllm.python.model_executor.cp_utils import cp_shard_rows
from xllm.python.model_executor.forward_context import get_forward_context, in_acl_graph
from xllm.python.models.deepseek_v32 import DeepseekV3MoE

if TYPE_CHECKING:
    from xllm.python.models.glm5_2 import Glm52Config


def validate_glm52_ep_config(cfg: Glm52Config) -> None:
    """Validate the explicit transport mode; never mask unsupported EPLv2."""
    if cfg.expert_parallel_degree not in (0, 1, 2):
        raise ValueError("GLM5.2 expert_parallel_degree must be 0, 1 or 2")
    if cfg.expert_parallel_degree == 2:
        if cfg.ep_size <= 1 or cfg.ep_size != cfg.world_size or cfg.moe_tp_size != 1:
            raise ValueError("GLM5.2 EPLv2 requires EP == world_size > 1 and MoE-TP1")
        if cfg.enable_mega_moe and cfg.cp_size > 1:
            raise ValueError("GLM5.2 CP AllToAllV EPLv2 does not support enable_mega_moe")
        if cfg.enable_eplb:
            raise ValueError("GLM5.2 AllToAllV EPLv2 does not support EPLB")
    if cfg.ep_size > 1 and cfg.enable_eplb:
        # Prefill always dispatches through AllToAllV when EP is active, but
        # degree 0/1 users configure "plain EP": name their legacy path so
        # the rejection does not point them at a transport they never chose.
        # Either way EPLB's dynamic expert migration is incompatible with the
        # statically sharded expert ranges, so fail fast at construction.
        if cfg.expert_parallel_degree == 2:
            raise ValueError("GLM5.2 AllToAllV EPLv2 does not support EPLB")
        raise ValueError("GLM5.2 legacy EP (degree 0/1) does not support EPLB expert migration")


class Glm52MoE(MoECore, DeepseekV3MoE):
    """AllToAllV for every EP prefill and EPLv2; retain legacy CP1 decode."""

    def __init__(self, cfg: Glm52Config, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        validate_glm52_ep_config(cfg)
        super().__init__(cfg, layer_id, dtype, device)
        if cfg.ep_size > 1 and (self.moe_tp_size != 1 or self.shared_experts.tp != 1):
            raise ValueError("GLM5.2 AllToAllV EP requires complete, MoE-TP1 shared and routed experts")

    @property
    def _ep_w2_scale(self) -> torch.Tensor:
        return self.experts_w2_scale_compute

    @property
    def _swiglu_limit(self) -> float:
        return 0.0

    def _run_shared_source(self, source: torch.Tensor) -> torch.Tensor:
        return self._run_shared_alltoall(source)

    def _uses_alltoall(self) -> bool:
        ctx = get_forward_context()
        if ctx.cp_context is not None:
            return True
        # Retain the established CP1 degree2+Mega decode transport. This is
        # selected from configuration and replicated host step metadata, not
        # as a fallback after an AllToAllV error. Prefill always uses AllToAllV.
        if self.cfg.expert_parallel_degree == 2 and not (self.cfg.cp_size == 1 and self.cfg.enable_mega_moe):
            return True
        metadata = ctx.metadata
        if self.dp_size > 1:
            step_types = tuple(getattr(metadata, "dp_is_decode", ()))
            if len(step_types) != self.dp_size:
                raise ValueError("GLM5.2 EP transport selection requires replicated DP step types")
            # All EP ranks must select the same transport in mixed DP steps.
            return not all(step_types)
        return bool(metadata.is_prefill or metadata.is_chunked_prefill)

    def _source_layout(self, hidden: torch.Tensor) -> tuple[TokenParallelLayout, torch.Tensor]:
        ctx = get_forward_context()
        metadata = ctx.metadata
        if in_acl_graph() or ctx.execution_state is not None:
            raise ValueError("GLM5.2 AllToAllV EP requires eager execution")
        if hidden.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError("GLM5.2 AllToAllV EP requires BF16 or FP16 hidden states")
        if ctx.eplb is not None:
            raise ValueError("GLM5.2 AllToAllV EP does not support runtime EPLB")
        execution_metadata = ctx.execution_contexts.get(Glm52EpMetadata)
        if not isinstance(execution_metadata, Glm52EpMetadata):
            raise ValueError("GLM5.2 AllToAllV EP requires typed active-token metadata")
        mask = execution_metadata.local_token_mask
        cp_context = ctx.cp_context
        if cp_context is not None:
            if (cp_context.cp_size, cp_context.cp_rank) != (self.cfg.cp_size, self.cfg.cp_rank):
                raise ValueError("GLM5.2 EP requires a matching CP context")
            if not (metadata.is_prefill or metadata.is_chunked_prefill):
                raise ValueError("GLM5.2 CP EP supports prefill only")
            if mask.shape != (cp_context.restore_index.numel(),):
                raise ValueError("GLM5.2 EP source activity must describe the pre-CP query rows")
            mask = cp_shard_rows(mask, cp_context)
        elif self.cfg.cp_size > 1 and not bool(getattr(metadata, "is_dummy", False)):
            raise ValueError("GLM5.2 CP EP requires a prefill CP context; CP decode is unsupported")
        if mask.shape != (hidden.shape[0],) or mask.dtype != torch.bool or mask.device != hidden.device:
            raise ValueError("GLM5.2 EP source mask must match the CP-local execution rows")
        layout = TokenParallelLayout(hidden.shape[0], self.cfg.tp_size, self.cfg.tp_rank, routed_backend="alltoall")
        return layout, layout.shard(mask)

    def _forward_alltoall(self, hidden: torch.Tensor) -> torch.Tensor:
        flat = hidden.reshape(-1, self.hidden)
        layout, mask = self._source_layout(flat)
        return self._forward_source_layout(hidden, layout, mask)

    def _run_shared_alltoall(self, source: torch.Tensor) -> torch.Tensor:
        """Use GLM5.2 standard SwiGLU, not the GLM5-next clamped default."""
        if not self._fuse_shared_expert:
            return self.shared_experts(source)
        quantized, input_scale = kernels.dynamic_quant(source)
        gate_up = self.shared_experts.gate_up_proj.forward_accumulated(quantized)
        activated, activation_scale = kernels.dequant_swiglu_quant(
            gate_up,
            self.shared_experts.gate_up_proj.weight_scale,
            input_scale,
            swiglu_mode=0,
            clamp_limit=0.0,
        )
        return self.shared_experts.down_proj.forward_quantized(activated, activation_scale)

    def _combine_expert_outputs(
        self,
        routed: torch.Tensor,
        shared: torch.Tensor,
        use_mega_moe: bool,
    ) -> torch.Tensor:
        # Keep the original GLM TP-only reduction (effective_moe_tp differs
        # from the raw cfg.moe_tp_size for EP1).
        if self.ep_size == 1:
            final = routed + shared
            if self.cfg.tp_size > 1:
                distributed.all_reduce_(final, "tp")
            return final
        return super()._combine_expert_outputs(routed, shared, use_mega_moe)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.ep_size > 1 and self.cfg.cp_size > 1 and self.dp_size > 1:
            # Replicated host admission must reject a real decode DP peer on
            # every EP rank before any CP rank enters AllToAllV. Empty DP
            # ranks remain collective participants, even if marked decode.
            metadata = get_forward_context().metadata
            step_types = tuple(getattr(metadata, "dp_is_decode", ()))
            counts = tuple(getattr(metadata, "raw_dp_execution_token_counts", ()))
            if len(step_types) != self.dp_size or len(counts) != self.dp_size:
                raise ValueError("GLM5.2 CP EP requires replicated DP step types and raw token counts")
            if any(count > 0 and is_decode for count, is_decode in zip(counts, step_types)):
                raise ValueError("GLM5.2 CP EP does not support real decode DP ranks")
        if self.ep_size > 1 and self._uses_alltoall():
            return self._forward_alltoall(hidden)
        return super().forward(hidden)


def validate_glm52_ep_execution_config(config: dict[str, object], cfg: Glm52Config) -> None:
    """Fail at model construction when EPLv2 would be captured by a graph."""
    validate_glm52_ep_config(cfg)
    if cfg.expert_parallel_degree != 2:
        return
    if cfg.cp_size == 1 and cfg.enable_mega_moe:
        logger.info("GLM5.2 prefill selects AllToAllV; CP1 decode retains legacy MegaMoE and graph execution")
        return
    backend = str(config.get("python_graph_backend", "off"))
    if bool(config.get("enable_graph", False)) or backend not in ("off", "none", "disabled"):
        raise ValueError("GLM5.2 AllToAllV EPLv2 is eager-only; disable graph execution")
    logger.info("GLM5.2 EPLv2 selects eager AllToAllV with TP source-token ownership")
