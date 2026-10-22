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

"""GLM5.2 source-owned EP transports, independent of attention/indexer code.

Attention-TP ranks hold identical CP-local rows. TokenParallelLayout assigns
one TP source owner; the routed backend sends its selected tokens to expert
owners and returns complete routed results to that source. Shared experts
have MoE-TP1 weights and are evaluated once on source rows, without TP/EP
sum. Only the TP token all-gather restores the attention layout. CP
zigzag/prefix padding is excluded by the HCCL token mask, not by changing
expert IDs or weights.

Degree-2 decode runs the fixed-capacity MC2 transport inside the capacity
window (graph-capturable, mirroring the GLM5-next EPLv2 admission); steps
beyond the window and every prefill take the variable-length AllToAllV
transport, which stays eager-only.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from scripts.logger import logger
from xllm.python import distributed, kernels
from xllm.python.attention.backend import AttentionMetadata
from xllm.python.layers.moe import MoECore
from xllm.python.layers.moe_parallel import Eplv2CommPolicy, TokenParallelLayout
from xllm.python.layers.npu.glm5_2_metadata import Glm52EpMetadata
from xllm.python.model_executor.cp_utils import cp_shard_rows
from xllm.python.model_executor.forward_context import ForwardContext, get_forward_context, in_acl_graph
from xllm.python.models.deepseek_v32 import DeepseekV3MoE
from xllm.python.models.weight_utils import W8A8WeightLoader

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


def _hccl_buffsize_mb() -> int:
    """Parse HCCL_BUFFSIZE with a diagnostic error for malformed values."""
    raw = os.getenv("HCCL_BUFFSIZE", "200")
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"HCCL_BUFFSIZE must be an integer number of MB, got {raw!r}") from exc


def glm52_eplv2_comm_policy(cfg: Glm52Config) -> Eplv2CommPolicy | None:
    """MC2 admission window for degree-2 decode (fixed capacity, graph-safe).

    Returns ``None`` when the configuration keeps another decode transport
    (degree != 2, CP, MegaMoE); a zero-capacity policy means the HCCL window
    admits no MC2 batch and degree-2 stays AllToAllV eager-only.
    """
    if cfg.expert_parallel_degree != 2 or cfg.cp_size > 1 or cfg.enable_mega_moe:
        return None
    return Eplv2CommPolicy.from_geometry(
        cfg.hidden_size,
        cfg.n_routed_experts,
        cfg.ep_size,
        cfg.num_experts_per_tok,
        _hccl_buffsize_mb(),
        cfg.eplv2_mc2_max_tokens_per_rank,
    )


def _require_dp_step_types(metadata: AttentionMetadata, dp_size: int) -> tuple[bool, ...]:
    """Read the replicated per-rank DP step types or fail loudly.

    The engine copies the whole ``dp_is_decode`` vector into every rank's
    input params, so the mixed-step verdict derived here is identical across
    the EP group; a malformed vector means the transport contract is broken
    and divergent collectives would follow, so raising is the only safe exit.
    """
    step_types = tuple(getattr(metadata, "dp_is_decode", ()))
    if len(step_types) != dp_size:
        raise ValueError("GLM5.2 EP transport selection requires replicated DP step types")
    return step_types


class Glm52MoE(MoECore, DeepseekV3MoE):
    """AllToAllV for EP prefill; MC2 or AllToAllV for EPLv2 decode."""

    def __init__(
        self,
        cfg: Glm52Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        comm_policy: Eplv2CommPolicy | None = None,
    ) -> None:
        validate_glm52_ep_config(cfg)
        super().__init__(cfg, layer_id, dtype, device)
        if cfg.ep_size > 1 and (self.moe_tp_size != 1 or self.shared_experts.tp != 1):
            raise ValueError("GLM5.2 AllToAllV EP requires complete, MoE-TP1 shared and routed experts")
        # Share the model-level policy object when provided (mirrors the
        # GLM5-next layer construction); fall back to a local derivation for
        # directly constructed layers (tests, standalone MoE modules).
        self._comm_policy = comm_policy if comm_policy is not None else glm52_eplv2_comm_policy(cfg)
        self._ep_dispatch_group_info: tuple[str, int, int] | None = None

    def load_from_checkpoint(self, loader: W8A8WeightLoader, mlp_prefix: str) -> None:
        super().load_from_checkpoint(loader, mlp_prefix)
        if self._comm_policy is not None:
            # Resolve the MC2 communication name before graph capture, not
            # on replay (mirrors the GLM5-next EPLv2 group resolution). The
            # base loader already post-processed every weight exactly once.
            self._ep_dispatch_group_info = distributed.moe_ep_hccl_info(self.gate.weight.device)
            if self._ep_dispatch_group_info[1:] != (self.cfg.ep_rank, self.cfg.ep_size):
                raise ValueError("GLM5.2 EPLv2 process group does not match the configured expert rank group")

    @property
    def _ep_w2_scale(self) -> torch.Tensor:
        return self.experts_w2_scale_compute

    @property
    def _swiglu_limit(self) -> float:
        return 0.0

    def _run_shared_source(self, source: torch.Tensor) -> torch.Tensor:
        return self._run_shared_alltoall(source)

    def _uses_source_sp(self) -> bool:
        """Whether this step takes the source-owned SP transport at all.

        Degree-2 decode below the MegaMoE carve-out always enters the SP
        path; ``_source_layout`` then admits MC2 inside the capacity window
        and AllToAllV beyond it. Prefill and CP always use AllToAllV.
        """
        ctx = get_forward_context()
        if ctx.cp_context is not None:
            return True
        metadata = ctx.metadata
        if self.cfg.expert_parallel_degree == 2:
            if metadata.is_prefill or metadata.is_chunked_prefill:
                return True
            if self.dp_size > 1:
                # A prefilling peer keeps every rank on AllToAllV (the step
                # types are replicated, so the verdict is group-wide).
                if not all(_require_dp_step_types(metadata, self.dp_size)):
                    return True
            # Decode: MegaMoE keeps the legacy path; otherwise the SP
            # transport runs (MC2 inside the window, AllToAllV beyond it).
            return not (self.cfg.cp_size == 1 and self.cfg.enable_mega_moe)
        if self.dp_size > 1:
            # All EP ranks must select the same transport in mixed DP steps.
            return not all(_require_dp_step_types(metadata, self.dp_size))
        return bool(metadata.is_prefill or metadata.is_chunked_prefill)

    def _source_layout(self, hidden: torch.Tensor) -> tuple[TokenParallelLayout, torch.Tensor]:
        ctx = get_forward_context()
        metadata = ctx.metadata
        if hidden.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError("GLM5.2 EP requires BF16 or FP16 hidden states")
        if ctx.eplb is not None:
            raise ValueError("GLM5.2 EP does not support runtime EPLB")
        execution_metadata = ctx.execution_contexts.get(Glm52EpMetadata)
        if not isinstance(execution_metadata, Glm52EpMetadata):
            raise ValueError("GLM5.2 EP requires typed active-token metadata")
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
        if self.dp_size > 1:
            # Pad every DP rank's dispatch to the widest peer so the
            # fixed-capacity MC2 window stays shape-stable across DP.
            counts = getattr(metadata, "dp_execution_token_counts", None)
            layout = TokenParallelLayout.from_dp(
                hidden.shape[0], self.cfg.tp_size, self.cfg.tp_rank, self.dp_size, self.cfg.dp_rank, counts
            )
        else:
            layout = TokenParallelLayout(hidden.shape[0], self.cfg.tp_size, self.cfg.tp_rank)
        backend, graph = self._select_routed_backend(layout, metadata, ctx)
        if backend == "alltoall" and graph:
            raise ValueError("GLM5.2 AllToAllV EP requires eager execution")
        return replace(layout, routed_backend=backend), layout.shard(mask)

    def _select_routed_backend(
        self, layout: TokenParallelLayout, metadata: AttentionMetadata, ctx: ForwardContext
    ) -> tuple[str, bool]:
        """Admit the fixed-capacity MC2 transport for decode, AllToAllV otherwise.

        The DP step-type check runs before any local classification: the
        engine replicates ``dp_is_decode`` to every rank, so a mixed step
        keeps the whole EP group on AllToAllV -- one rank selecting MC2
        while a prefilling peer selects AllToAllV would desynchronize the
        collectives. MC2 keeps graph-stable shapes (padded dispatch rows,
        persistent EP group), so decode steps inside the capacity window may
        capture and replay; anything beyond the window takes AllToAllV
        eagerly. The returned graph flag always reflects the real execution
        mode, so an AllToAllV pick under capture still fails closed at the
        caller.
        """
        policy = self._comm_policy
        graph = in_acl_graph() or ctx.execution_state is not None
        if policy is None:
            return "alltoall", graph
        if self.dp_size > 1:
            if not all(_require_dp_step_types(metadata, self.dp_size)):
                return "alltoall", graph
        if metadata.is_prefill or metadata.is_chunked_prefill:
            return "alltoall", graph
        return policy.select(layout.dispatch_tokens, graph=graph), graph

    def _forward_source_sp(self, hidden: torch.Tensor) -> torch.Tensor:
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
            step_types = _require_dp_step_types(metadata, self.dp_size)
            counts = tuple(getattr(metadata, "raw_dp_execution_token_counts", ()))
            if len(counts) != self.dp_size:
                raise ValueError("GLM5.2 CP EP requires replicated raw token counts")
            if any(count > 0 and is_decode for count, is_decode in zip(counts, step_types)):
                raise ValueError("GLM5.2 CP EP does not support real decode DP ranks")
        if self.ep_size > 1 and self._uses_source_sp():
            return self._forward_source_sp(hidden)
        return super().forward(hidden)


def validate_glm52_ep_execution_config(config: dict[str, object], cfg: Glm52Config) -> None:
    """Fail at model construction when EPLv2 would be captured by a graph."""
    validate_glm52_ep_config(cfg)
    if cfg.expert_parallel_degree != 2:
        return
    if cfg.cp_size == 1 and cfg.enable_mega_moe:
        logger.info("GLM5.2 prefill selects AllToAllV; CP1 decode retains legacy MegaMoE and graph execution")
        return
    policy = glm52_eplv2_comm_policy(cfg)
    limit = 0 if policy is None else policy.graph_token_limit
    if limit > 0:
        logger.info(
            "GLM5.2 EPLv2 decode admits MC2 transport for %d dispatch rows per rank; "
            "prefill and over-capacity steps use AllToAllV",
            limit,
        )
        return
    # No MC2 window (CP decode, or geometry the HCCL buffer cannot serve):
    # every step runs the variable-length AllToAllV transport, eager-only.
    # The spelling set mirrors the executor's graph-backend classification
    # (""/"off"/"none"/"0" are soft-off; "disabled" pins eager regardless of
    # --enable_graph), so the raise only fires when graphs would really run.
    backend = str(config.get("python_graph_backend", "off")).lower()
    if bool(config.get("enable_graph", False)) or backend not in ("", "off", "none", "0", "disabled"):
        raise ValueError("GLM5.2 AllToAllV EPLv2 is eager-only; disable graph execution")
    logger.info("GLM5.2 EPLv2 selects eager AllToAllV with TP source-token ownership")
