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

"""Shared gated-MLP and MoE semantics for GLM and DeepSeek models.

Dense and shared-expert FFN blocks of DeepSeek-V3.2/V4, GLM-5.2 and
GLM-next are the same ``gate_up -> SwiGLU -> down`` computation and differ
only in three policy points:

* ``swiglu_limit`` — GLM-next/DeepSeek-V4 clamp gate/up before SiLU
  (limit > 0); DeepSeek-V3.2/GLM-5.2 never clamp (limit == 0).
* ``clamp_precision`` — reference implementations disagree on the clamp
  dtype: HF GLM-next/DSv4 clamp in the native dtype, the DSv4.1 official
  reference upcasts to fp32 first. Both semantics are preserved.
* ``tp_reduce`` — GLM-5.2 reduces the TP partial sum in fp32 for
  stability; the other models reduce in the activation dtype.

Model adapters retain checkpoint storage and transport admission (GLM-5.2
legacy MegaMoE versus GLM-next MC2); this core never selects a fallback
after execution.
"""

from __future__ import annotations

from typing import Literal, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python import distributed, kernels
from xllm.python.layers.moe_parallel import TokenParallelLayout
from xllm.python.model_executor.forward_context import in_acl_graph


def clamped_silu_and_mul(gate_up: torch.Tensor, limit: float = 0.0) -> torch.Tensor:
    """Apply standard SwiGLU, or GLM-next's pre-SiLU asymmetric clamp.

    Native-dtype reference semantics (HF Glm5NextTextMLP/DeepseekV4MLP):
    clamp happens in the activation dtype and SiLU runs on the clamped
    bf16/fp16 tensor.
    """
    gate, up = gate_up.chunk(2, dim=-1)
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


def clamped_silu_and_mul_fp32(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    """fp32-upcast clamped SwiGLU (DSv4.1 official reference semantics).

    ``limit <= 0`` keeps the fused silu_and_mul fast path bit-identical to
    the unclamped models; only the clamped branch pays the upcast.
    """
    if not 0.0 < limit < 1_000_000.0:
        return kernels.silu_and_mul(gate_up)
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.float().clamp_max(limit)
    up = up.float().clamp(min=-limit, max=limit)
    return (F.silu(gate) * up).to(gate_up.dtype)


ClampPrecision = Literal["native", "fp32"]
TpReduce = Literal["native", "fp32"]


def _fused_dequant_swiglu_quant(
    gate_up: torch.Tensor,
    weight_scale: torch.Tensor,
    pertoken: torch.Tensor,
    fused: bool,
    clamp_limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantize a gate_up GEMM accumulator through SwiGLU.

    ``fused`` selects the execution path, not the math: True routes the
    clamp/activation into the fused operator (swiglu_mode=1 + clamp_limit),
    False keeps the split path that returns the raw gate|up int8 for the
    caller's explicit activation (GLM-5.2's historical decomposition).
    """
    if fused:
        return kernels.dequant_swiglu_quant(
            gate_up,
            weight_scale,
            pertoken,
            swiglu_mode=1,
            clamp_limit=clamp_limit if clamp_limit > 0 else 0.0,
        )
    return kernels.dequant_swiglu_quant(gate_up, weight_scale, pertoken, swiglu_mode=0)


class QuantGatedMLP(nn.Module):
    """W8A8 gated-MLP shared by dense and shared-expert blocks.

    Column-parallel fused ``gate_up_proj`` (intermediate dim sharded) plus
    row-parallel ``down_proj``, quantized per W8A8 dynamic-activation
    semantics. Behavior knobs map one-to-one to the three model policy
    points; every default reproduces the DeepSeek-V3.2 dense MLP exactly.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        device: torch.device,
        *,
        tp_size: int,
        skip_tp_reduce: bool = False,
        swiglu_limit: float = 0.0,
        clamp_precision: ClampPrecision = "native",
        tp_reduce: TpReduce = "native",
        fused_swiglu: bool = True,
    ) -> None:
        super().__init__()
        if tp_size <= 0 or intermediate_size % tp_size:
            raise ValueError("MLP intermediate size must divide its tensor-parallel group")
        inter_local = intermediate_size // tp_size
        self.hidden_size = hidden_size
        self.tp = tp_size
        self.skip_tp_reduce = skip_tp_reduce
        self.swiglu_limit = swiglu_limit
        self.clamp_precision = clamp_precision
        self.tp_reduce = tp_reduce
        self.fused_swiglu = fused_swiglu
        # W8A8DynamicLinear must stay import-local: deepseek_v32 and the
        # layer package would otherwise import each other at module load.
        from xllm.python.models.deepseek_v32 import W8A8DynamicLinear

        self.gate_up_proj = W8A8DynamicLinear(hidden_size, 2 * inter_local, device)
        self.down_proj = W8A8DynamicLinear(inter_local, hidden_size, device)

    def process_weights_after_loading(self) -> None:
        self.gate_up_proj.process_weights_after_loading()
        self.down_proj.process_weights_after_loading()

    def load_from_checkpoint(
        self,
        loader,
        mlp_prefix: str,
        *,
        world: int | None = None,
        rank: int | None = None,
    ) -> None:
        """Load a dense/shared W8A8 MLP; ``world``/``rank`` = ``None`` uses the loader TP."""
        loader.load_w8a8_mlp(mlp_prefix, world=world, rank=rank)
        self.process_weights_after_loading()

    def _activate(self, gate_up: torch.Tensor) -> torch.Tensor:
        if self.clamp_precision == "fp32":
            return clamped_silu_and_mul_fp32(gate_up, self.swiglu_limit)
        if self.swiglu_limit > 0:
            return clamped_silu_and_mul(gate_up, self.swiglu_limit)
        return kernels.silu_and_mul(gate_up)

    def _reduce_output(
        self,
        out: torch.Tensor,
        tp_reduce_add: torch.Tensor | None,
    ) -> torch.Tensor:
        if tp_reduce_add is not None:
            out = out + tp_reduce_add
        reduce_result = self.tp > 1 and (not self.skip_tp_reduce or tp_reduce_add is not None)
        if reduce_result and self.tp_reduce == "fp32":
            output_dtype = out.dtype
            out = out.to(torch.float32)
            distributed.tp_all_reduce(out)
            out = out.to(output_dtype)
        elif reduce_result:
            distributed.tp_all_reduce(out)
        return out

    def _forward_gate_up(
        self,
        gate_up: torch.Tensor,
        tp_reduce_add: torch.Tensor | None,
    ) -> torch.Tensor:
        out = self.down_proj(self._activate(gate_up))
        return self._reduce_output(out, tp_reduce_add)

    def forward(self, x: torch.Tensor, tp_reduce_add: torch.Tensor | None = None) -> torch.Tensor:
        return self._forward_gate_up(self.gate_up_proj(x), tp_reduce_add)

    def forward_quantized(
        self,
        x_int8: torch.Tensor,
        pertoken: torch.Tensor,
        tp_reduce_add: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward_quantized(x_int8, pertoken)
        return self._forward_gate_up(gate_up, tp_reduce_add)

    def forward_dequant_swiglu_quant(
        self,
        x: torch.Tensor,
        tp_reduce_add: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_int8, pertoken = kernels.dynamic_quant(x)
        gate_up = self.gate_up_proj.forward_accumulated(x_int8)
        act_int8, act_scale = _fused_dequant_swiglu_quant(
            gate_up,
            self.gate_up_proj.weight_scale,
            pertoken,
            fused=self.fused_swiglu,
            clamp_limit=self.swiglu_limit if self.clamp_precision == "native" else 0.0,
        )
        out = self.down_proj.forward_quantized(act_int8, act_scale)
        return self._reduce_output(out, tp_reduce_add)

    def quantize_and_project_gate_up(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_int8, pertoken = kernels.dynamic_quant(x)
        return self.gate_up_proj.forward_accumulated(x_int8), pertoken

    def activate_and_quantize(self, gate_up: torch.Tensor, pertoken: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _fused_dequant_swiglu_quant(
            gate_up,
            self.gate_up_proj.weight_scale,
            pertoken,
            fused=self.fused_swiglu,
            clamp_limit=self.swiglu_limit if self.clamp_precision == "native" else 0.0,
        )

    def project_down(self, act_int8: torch.Tensor, act_scale: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward_quantized(act_int8, act_scale)


class _GlmMoEConfig(Protocol):
    norm_topk_prob: bool
    ep_size: int
    ep_rank: int
    # Read only by the mc2/megamoe branches of _run_routed_sp; a legacy-only
    # adapter that never selects those backends may leave them defaulted.
    enable_mega_moe: bool
    mega_moe_num_max_tokens_per_rank: int
    mega_moe_ccl_buffer_size: int


class MoECore:
    """Common execution mixin; adapters own parameters and validate admission.

    This is not an nn.Module container: no extra checkpoint prefix or duplicate
    parameter allocation is introduced when either model inherits the core.
    """

    cfg: _GlmMoEConfig
    gate: torch.nn.Linear
    e_score_correction_bias: torch.Tensor
    experts_w13: torch.Tensor
    experts_w2: torch.Tensor
    experts_w13_scale: torch.Tensor
    shared_experts: torch.nn.Module
    hidden: int
    topk: int
    n_group: int
    topk_group: int
    num_experts: int
    routed_scaling: float
    _ep_dispatch_group_info: tuple[str, int, int] | None
    # Optional members consumed by the megamoe branch of _run_routed_sp;
    # provided by adapters that enable the MegaMoE backend (registered
    # buffers on Glm5NextMoE / DeepseekV3MoE).
    mega_moe_context: torch.Tensor
    mega_moe_w13_scale: torch.Tensor
    mega_moe_w2_scale: torch.Tensor

    @property
    def _ep_w2_scale(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def _swiglu_limit(self) -> float:
        raise NotImplementedError

    @property
    def _swiglu_mode(self) -> int:
        return 1 if self._swiglu_limit > 0 else 0

    def _topk(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """HF eager router: correction bias selects IDs, never mixture weights."""
        cfg = self.cfg
        flat = hidden_states.reshape(-1, self.hidden)
        logits = F.linear(flat.float(), self.gate.weight.float())
        scores = logits.sigmoid()
        choice = scores + self.e_score_correction_bias
        group_scores = choice.reshape(-1, self.n_group, self.num_experts // self.n_group).topk(2, dim=-1)[0].sum(-1)
        group_indices = group_scores.topk(self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_indices, 1)
        expert_mask = group_mask.unsqueeze(-1).expand(-1, self.n_group, self.num_experts // self.n_group)
        choice = choice.masked_fill(~expert_mask.reshape(-1, self.num_experts).bool(), float("-inf"))
        indices = choice.topk(self.topk, dim=-1, sorted=False)[1]
        weights = scores.gather(1, indices)
        if cfg.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling, indices

    def _route_ep(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route in FP32; apply routed scaling once when combining outputs."""
        return kernels.moe_gate_routing(
            self.gate(hidden.float()),
            self.e_score_correction_bias,
            self.topk,
            self.topk_group,
            self.n_group,
            self.cfg.norm_topk_prob,
            routed_scaling_factor=1.0,
        )

    def _run_routed_sp(
        self,
        hidden: torch.Tensor,
        layout: TokenParallelLayout,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        backend = layout.routed_backend
        if backend not in ("mc2", "alltoall", "megamoe"):
            raise ValueError(f"Unknown GLM source-owned routed backend: {backend}")
        if backend == "mc2" and self._swiglu_mode != 1:
            raise ValueError("Standard GLM SwiGLU is not admitted by the MC2 provider")
        if layout.valid_tokens:
            weights, indices = self._route_ep(hidden)
        else:
            weights = hidden.new_zeros((layout.shard_tokens, self.topk), dtype=torch.float32)
            indices = hidden.new_zeros((layout.shard_tokens, self.topk), dtype=torch.int32)
        if backend == "alltoall":
            return kernels.ep_moe_w8a8_alltoall(
                hidden,
                weights,
                indices,
                self.experts_w13,
                self.experts_w2,
                self.experts_w13_scale,
                self._ep_w2_scale,
                ep_size=self.cfg.ep_size,
                ep_rank=self.cfg.ep_rank,
                num_experts=self.num_experts,
                x_active_mask=mask,
                swiglu_limit=self._swiglu_limit,
                swiglu_mode=self._swiglu_mode,
            )
        if backend == "megamoe":
            if not self.cfg.enable_mega_moe or layout.dispatch_tokens > self.cfg.mega_moe_num_max_tokens_per_rank:
                raise ValueError("GLM MegaMoE must be enabled and fit the communication capacity")
            # Adapters initialize the scale buffers differently: Glm5NextMoE
            # registers None while DeepseekV3MoE registers an empty int64
            # tensor. Both are "not encoded yet" -- reject either sentinel.
            if (
                self.mega_moe_w13_scale is None
                or self.mega_moe_w13_scale.numel() == 0
                or self.mega_moe_w2_scale is None
                or self.mega_moe_w2_scale.numel() == 0
            ):
                raise RuntimeError("GLM MegaMoE scales must be encoded before execution")
        else:
            group_info = self._ep_dispatch_group_info
            if group_info is None:
                raise RuntimeError("GLM EP communication group must be initialized before execution")
        # Both fixed-capacity backends consume the same EP-wide padded rows.
        dispatch_hidden = layout.pad_dispatch(hidden)
        dispatch_weights = layout.pad_dispatch(weights)
        dispatch_indices = layout.pad_dispatch(indices)
        dispatch_mask = layout.pad_dispatch(mask)
        if backend == "megamoe":
            output = kernels.mega_moe(
                self.mega_moe_context,
                dispatch_hidden,
                dispatch_indices.to(torch.int32),
                dispatch_weights.float(),
                self.experts_w13,
                self.experts_w2,
                self.mega_moe_w13_scale,
                self.mega_moe_w2_scale,
                self.num_experts,
                self.cfg.ep_size,
                self.cfg.mega_moe_ccl_buffer_size,
                self.cfg.mega_moe_num_max_tokens_per_rank,
                x_active_mask=dispatch_mask,
                swiglu_limit=self._swiglu_limit,
            )
        else:
            output = kernels.ep_moe_w8a8(
                dispatch_hidden,
                dispatch_weights,
                dispatch_indices,
                self.experts_w13,
                self.experts_w2,
                self.experts_w13_scale,
                self._ep_w2_scale,
                group_ep=group_info[0],
                ep_size=self.cfg.ep_size,
                ep_rank=self.cfg.ep_rank,
                num_experts=self.num_experts,
                global_bs=0,
                x_active_mask=dispatch_mask,
                swiglu_limit=self._swiglu_limit,
            )
        return output[: layout.shard_tokens]

    def _run_shared_source(self, source: torch.Tensor) -> torch.Tensor:
        return self.shared_experts(source)

    def _forward_source(
        self,
        source: torch.Tensor,
        layout: TokenParallelLayout,
        mask: torch.Tensor,
        *,
        shared_stream: torch.npu.Stream | None = None,
    ) -> torch.Tensor:
        """Compute source rows once; dispatch/combine already completes EP sum."""
        if not layout.valid_tokens:
            shared = torch.zeros_like(source)
            routed = self._run_routed_sp(source, layout, mask)
        else:
            # Shared experts are pure local GEMMs; run them only on the real
            # source rows and scatter the result back so zigzag/DP padding
            # rows never pay two W8A8 GEMMs per layer. The routed side
            # already filters dispatch by the same mask inside the kernel.
            #
            # Eager only: ``mask.all()`` is a device->host sync and
            # ``nonzero`` is dynamically shaped, so an ACL-graph capture
            # (the degree-2 MC2/MegaMoE graph path) must keep the fixed
            # full-bucket shape -- run the shared GEMMs over every padded
            # row and let the final masked_fill drop the padding, exactly
            # as the pre-refactor code did.
            if in_acl_graph():
                # Fixed-shape graph execution (warmup and capture): no
                # auxiliary-stream fork, no row compression (nonzero is
                # dynamically shaped), no mask sync to the host. Run shared
                # and routed serially on the captured stream; masked_fill
                # drops the padding rows, exactly as the pre-refactor code
                # did. Warmup must take this same branch so replay matches
                # what was warmed.
                shared = self._run_shared_source(source)
                routed = self._run_routed_sp(source, layout, mask)
                output = torch.add(shared.float(), routed.float(), alpha=self.routed_scaling).to(source.dtype)
                output.masked_fill_(~mask[:, None], 0)
                return output
            # Pure eager execution (the eager runner sets neither an
            # execution_state nor an acl_graph context): compression is safe.
            # Graph warmup executes the fixed-shape branch above instead.
            rows = None
            if not bool(mask.all()):
                rows = mask.nonzero(as_tuple=False).flatten()
                if rows.numel() == 0:
                    # An empty rank (all rows inactive): no shared GEMM at all.
                    shared = torch.zeros_like(source)
                    routed = self._run_routed_sp(source, layout, mask)
                    output = torch.add(shared.float(), routed.float(), alpha=self.routed_scaling).to(source.dtype)
                    output.masked_fill_(~mask[:, None], 0)
                    return output
                packed = source.index_select(0, rows)
            else:
                packed = source
            if shared_stream is None:
                shared_packed = self._run_shared_source(packed)
            else:
                current = torch.npu.current_stream(source.device)
                shared_stream.wait_stream(current)
                source.record_stream(shared_stream)
                with torch.npu.stream(shared_stream):
                    shared_packed = self._run_shared_source(packed)
                # Keep the routed dispatch on the main stream, concurrent
                # with the auxiliary shared work, then join before merging.
                routed = self._run_routed_sp(source, layout, mask)
                current.wait_stream(shared_stream)
                shared_packed.record_stream(current)
            if rows is not None:
                shared = torch.zeros_like(source)
                shared.index_copy_(0, rows, shared_packed)
            else:
                shared = shared_packed
            if shared_stream is None:
                routed = self._run_routed_sp(source, layout, mask)
        output = torch.add(shared.float(), routed.float(), alpha=self.routed_scaling).to(source.dtype)
        output.masked_fill_(~mask[:, None], 0)
        return output

    def _forward_source_layout(
        self,
        hidden: torch.Tensor,
        layout: TokenParallelLayout,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Bridge an attention-TP replicated layout to unique source ownership."""
        flat = hidden.reshape(-1, self.hidden)
        source = layout.shard(flat).masked_fill(~mask[:, None], 0)
        output = self._forward_source(source, layout, mask)
        return layout.gather(output).reshape(hidden.shape)
