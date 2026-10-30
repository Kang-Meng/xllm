# Copyright 2025-2026 The xLLM Authors.
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

"""Qwen3.5 hybrid-attention causal LM for the Python executor."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from xllm.python.layers import ColumnParallelLinear, HiddenParallelEmbedding
from xllm.python.layers.moe_parallel import Eplv2CommPolicy
from xllm.python.layers.qwen3_5.common import PartialRotaryEmbedding
from xllm.python.layers.qwen3_5.factory import get_qwen3_5_decoder_layer_class
from xllm.python.layers.sequence_parallel import (
    resolve_sequence_parallel,
    sp_all_gather,
    sp_shard,
)
from xllm.python.layers.sequence_parallel_metadata import SpMetadata
from xllm.python.model_executor.forward_context import get_execution_context, record_layer_event
from xllm.python.model_loader import (
    ParallelLoadContext,
    ScopedWeightLoader,
    gqa_head_split,
    load_causal_lm_weights,
)
from xllm.python.models.aux_hidden_capture import AuxHiddenCapture
from xllm.python.models.base import PyModelBase


@dataclass
class Qwen3_5Config:
    hidden_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    rope_scaling_mrope_section: list[int]
    rope_scaling_mrope_interleaved: bool
    max_position_embeddings: int
    vocab_size: int
    layer_types: list[str]
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    attention_bias: bool
    attn_output_gate: bool
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    decoder_sparse_step: int
    mlp_only_layers: list[int]
    norm_topk_prob: bool
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    tp_size: int
    tp_rank: int
    dp_size: int
    dp_rank: int
    world_size: int
    moe_tp_size: int
    moe_tp_rank: int
    ep_size: int
    ep_rank: int
    enable_mega_moe: bool
    sequence_parallel_enabled: bool
    mega_moe_context: torch.Tensor | None
    mega_moe_ccl_buffer_size: int
    mega_moe_num_max_tokens_per_rank: int
    expert_parallel_degree: int = 0
    # Ordinary Lv2 uses cross-layer SP and shared/routed overlap when TP>1.
    eplv2_sequence_parallel: bool = True
    eplv2_comm_mode: str = "auto"
    eplv2_mc2_max_tokens_per_rank: int = 512
    layers_to_capture: tuple[int, ...] = ()
    jev_prompt_attention: str = "causal"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Qwen3_5Config:
        def pick(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in d and d[key] is not None:
                    return d[key]
            return default

        jev_prompt_attention = d.get("jev_prompt_attention", "causal")
        if jev_prompt_attention not in ("causal", "bidirectional"):
            raise ValueError("jev_prompt_attention must be causal or bidirectional")

        n_layers = int(pick("n_layers", "num_hidden_layers", default=0))
        interval = int(pick("full_attention_interval", default=4))
        layer_types = list(pick("layer_types", default=[]))
        if not layer_types:
            layer_types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n_layers)]
        if len(layer_types) != n_layers:
            raise ValueError("layer_types must contain one entry per hidden layer")

        hidden_size = int(pick("hidden_size", default=0))
        n_heads = int(pick("n_heads", "num_attention_heads", default=0))
        tp_size = int(pick("tp_size", default=1))
        dp_size = int(pick("dp_size", default=1))
        world_size = int(pick("world_size", default=tp_size * dp_size))
        ep_size = int(pick("ep_size", default=1))
        num_experts = int(pick("num_experts", "n_routed_experts", default=0))
        degree = int(pick("expert_parallel_degree", default=0))
        enable_mega_moe = bool(pick("enable_mega_moe", default=False))
        eplv2_sp = True
        eplv2_mode = "auto"
        eplv2_token_limit = 512
        if (
            str(pick("device", default="cuda")).split(":", 1)[0] in ("npu", "privateuseone")
            and num_experts > 0
            and degree == 2
            and not enable_mega_moe
        ):
            sp_value = int(pick("eplv2_sequence_parallel", default=os.getenv("XLLM_EPLV2_SEQUENCE_PARALLEL", "1")))
            if sp_value not in (0, 1):
                raise ValueError("Qwen3.5 EPLv2 sequence parallelism must be 0 or 1")
            eplv2_sp = bool(sp_value)
            eplv2_mode = str(pick("eplv2_comm_mode", default=os.getenv("XLLM_EPLV2_COMM_MODE", "auto")))
            if eplv2_mode not in ("auto", "mc2", "alltoall"):
                raise ValueError("Qwen3.5 EPLv2 communication mode must be auto, mc2 or alltoall")
            try:
                eplv2_token_limit = int(
                    pick("eplv2_mc2_max_tokens_per_rank", default=os.getenv("XLLM_EPLV2_MC2_MAX_TOKENS", "512"))
                )
            except ValueError as error:
                raise ValueError(
                    "eplv2_mc2_max_tokens_per_rank (XLLM_EPLV2_MC2_MAX_TOKENS) must be an integer"
                ) from error
        rope_parameters = pick("rope_parameters", default={})
        if not isinstance(rope_parameters, dict):
            rope_parameters = {}
        rope_scaling = pick("rope_scaling", default={})
        if not isinstance(rope_scaling, dict):
            rope_scaling = {}
        mrope_section = pick(
            "rope_scaling_mrope_section",
            default=rope_parameters.get(
                "mrope_section",
                rope_scaling.get("mrope_section", []),
            ),
        )
        mrope_interleaved = pick(
            "rope_scaling_mrope_interleaved",
            default=rope_parameters.get(
                "mrope_interleaved",
                rope_scaling.get("mrope_interleaved", False),
            ),
        )

        return cls(
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=int(pick("n_kv_heads", "num_key_value_heads", default=0)),
            head_dim=int(pick("head_dim", default=hidden_size // n_heads)),
            intermediate_size=int(pick("intermediate_size", default=0)),
            rms_norm_eps=float(pick("rms_norm_eps", default=1e-6)),
            rope_theta=float(pick("rope_theta", default=1e7)),
            partial_rotary_factor=float(pick("partial_rotary_factor", default=0.25)),
            rope_scaling_mrope_section=[int(section) for section in mrope_section],
            rope_scaling_mrope_interleaved=bool(mrope_interleaved),
            max_position_embeddings=int(pick("max_position_embeddings", default=262144)),
            vocab_size=int(pick("vocab_size", default=248320)),
            layer_types=layer_types,
            linear_conv_kernel_dim=int(pick("linear_conv_kernel_dim", default=4)),
            linear_key_head_dim=int(pick("linear_key_head_dim", default=128)),
            linear_value_head_dim=int(pick("linear_value_head_dim", default=128)),
            linear_num_key_heads=int(pick("linear_num_key_heads", default=16)),
            linear_num_value_heads=int(pick("linear_num_value_heads", default=32)),
            attention_bias=bool(pick("attention_bias", default=False)),
            attn_output_gate=bool(pick("attn_output_gate", default=True)),
            tie_word_embeddings=bool(pick("tie_word_embeddings", default=False)),
            num_experts=num_experts,
            num_experts_per_tok=int(pick("num_experts_per_tok", default=0)),
            decoder_sparse_step=int(pick("decoder_sparse_step", default=1)),
            mlp_only_layers=[int(layer_id) for layer_id in pick("mlp_only_layers", default=[])],
            norm_topk_prob=bool(pick("norm_topk_prob", default=True)),
            moe_intermediate_size=int(pick("moe_intermediate_size", default=0)),
            shared_expert_intermediate_size=int(pick("shared_expert_intermediate_size", default=0)),
            tp_size=tp_size,
            tp_rank=int(pick("tp_rank", default=0)),
            dp_size=dp_size,
            dp_rank=int(pick("dp_rank", default=0)),
            world_size=world_size,
            moe_tp_size=int(pick("moe_tp_size", default=world_size // max(ep_size, 1))),
            moe_tp_rank=int(pick("moe_tp_rank", default=0)),
            ep_size=ep_size,
            ep_rank=int(pick("ep_rank", default=0)),
            enable_mega_moe=enable_mega_moe,
            sequence_parallel_enabled=bool(pick("sequence_parallel_enabled", default=False)),
            mega_moe_context=pick("mega_moe_context", default=None),
            mega_moe_ccl_buffer_size=int(pick("mega_moe_ccl_buffer_size", default=0)),
            mega_moe_num_max_tokens_per_rank=int(pick("mega_moe_num_max_tokens_per_rank", default=0)),
            expert_parallel_degree=degree,
            eplv2_sequence_parallel=eplv2_sp,
            eplv2_comm_mode=eplv2_mode,
            eplv2_mc2_max_tokens_per_rank=eplv2_token_limit,
            layers_to_capture=tuple(int(layer_id) for layer_id in pick("layers_to_capture", default=[])),
            jev_prompt_attention=jev_prompt_attention,
        )

    @classmethod
    def prepare_execution_config(
        cls,
        config: dict,
        device: torch.device,
        *,
        allow_speculative_decoding: bool = False,
        allow_draft_engine: bool = False,
    ) -> Qwen3_5Config:
        """Resolve the Qwen3.5 parallel execution policy for one host.

        Every host that embeds the Qwen3.5 stack (the causal LM itself and
        wrappers such as JoyTalker) must derive the EPLv2 checks and the
        sequence-parallel state the same way. The resolved
        ``sequence_parallel_enabled`` is written back into ``config`` because
        the execution metadata builders consume it from that shared dict.
        Hosts that support speculative decoding opt in explicitly; other
        hosts retain the default restriction. MTP drafts also opt in to the
        draft-engine path explicitly.
        """
        cfg = cls.from_dict(config)
        decoder_layer_cls = get_qwen3_5_decoder_layer_class(device)
        ordinary_lv2 = (
            device.type in ("npu", "privateuseone")
            and cfg.num_experts > 0
            and cfg.expert_parallel_degree == 2
            and not cfg.enable_mega_moe
        )
        if ordinary_lv2:
            if int(config.get("cp_size", 1)) != 1:
                raise NotImplementedError("Qwen3.5 EPLv2 requires CP=1")
            if int(config.get("num_speculative_tokens", 0)) != 0 and not allow_speculative_decoding:
                raise NotImplementedError("This Qwen3.5 host does not support speculative decoding with EPLv2")
            if bool(config.get("is_draft_engine", False)) and not allow_draft_engine:
                raise NotImplementedError("Qwen3.5 EPLv2 does not support draft engines")
            # Python BF16 EP does not use the C++ fused-MC2 mode.
            if bool(config.get("enable_eplb", False)):
                raise NotImplementedError("Qwen3.5 EPLv2 requires EPLB to be disabled")
            if cfg.moe_tp_size != 1 or cfg.ep_size <= 1:
                raise ValueError("Qwen3.5 EPLv2 requires MoE-TP=1 and EP>1")
            if cfg.ep_rank != cfg.dp_rank * cfg.tp_size + cfg.tp_rank:
                raise ValueError("Qwen3.5 EPLv2 EP rank must match the DP/TP coordinates")
            # Explicit Lv2 owns the token layout for every instance role.
            cfg.sequence_parallel_enabled = cfg.eplv2_sequence_parallel and cfg.tp_size > 1
        else:
            cfg.sequence_parallel_enabled = resolve_sequence_parallel(config, decoder_layer_cls)
        config["sequence_parallel_enabled"] = cfg.sequence_parallel_enabled
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.n_heads <= 0 or self.n_kv_heads <= 0 or self.n_layers <= 0:
            raise ValueError("invalid Qwen3.5 model dimensions")
        if min(self.tp_size, self.dp_size, self.moe_tp_size, self.ep_size) <= 0:
            raise ValueError("parallel sizes must be positive")
        if self.tp_size * self.dp_size != self.world_size:
            raise ValueError("world_size must equal tp_size * dp_size")
        if self.moe_tp_size * self.ep_size != self.world_size:
            raise ValueError("world_size must equal moe_tp_size * ep_size")
        if not 0 <= self.dp_rank < self.dp_size:
            raise ValueError("dp_rank must be in [0, dp_size)")
        if not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("tp_rank must be in [0, tp_size)")
        if not 0 <= self.moe_tp_rank < self.moe_tp_size:
            raise ValueError("moe_tp_rank must be in [0, moe_tp_size)")
        if not 0 <= self.ep_rank < self.ep_size:
            raise ValueError("ep_rank must be in [0, ep_size)")
        for name, count in (
            ("hidden size", self.hidden_size),
            ("attention heads", self.n_heads),
            ("linear key heads", self.linear_num_key_heads),
            ("linear value heads", self.linear_num_value_heads),
            ("dense intermediate size", self.intermediate_size),
            ("vocabulary size", self.vocab_size),
        ):
            if count % self.tp_size:
                raise ValueError(f"{name} must be divisible by tp_size")
        if self.n_kv_heads >= self.tp_size:
            if self.n_kv_heads % self.tp_size:
                raise ValueError("KV heads must be divisible by tp_size when KV heads are sharded")
        elif self.tp_size % self.n_kv_heads:
            raise ValueError("tp_size must be divisible by KV heads when KV heads are replicated")
        if self.decoder_sparse_step <= 0:
            raise ValueError("decoder_sparse_step must be positive")
        if self.num_experts:
            if self.num_experts_per_tok <= 0:
                raise ValueError("num_experts_per_tok must be positive for MoE")
            if self.num_experts % self.ep_size:
                raise ValueError("num_experts must be divisible by ep_size")
            if self.moe_intermediate_size <= 0:
                raise ValueError("moe_intermediate_size must be positive for MoE")
            if self.moe_intermediate_size % self.moe_tp_size:
                raise ValueError("moe_intermediate_size must be divisible by moe_tp_size")
            if self.shared_expert_intermediate_size <= 0:
                raise ValueError("shared_expert_intermediate_size must be positive for Qwen3.5 MoE")
            if self.shared_expert_intermediate_size % self.tp_size:
                raise ValueError("shared_expert_intermediate_size must be divisible by tp_size")
        if self.enable_mega_moe:
            if not self.sequence_parallel_enabled:
                raise ValueError("MegaMoe requires sequence parallelism; the current configuration disables SP")
            if self.num_experts <= 0:
                raise ValueError("MegaMoe sequence parallelism requires a MoE model")
            if self.dp_size <= 1:
                raise ValueError("MegaMoe sequence parallelism requires DP > 1")
            if self.moe_tp_size != 1 or self.ep_size != self.world_size:
                raise ValueError("MegaMoe sequence parallelism requires MoE-TP = 1 and EP = world_size")
            if self.mega_moe_context is None:
                raise ValueError("MegaMoe sequence parallelism requires a communication context")
            if self.mega_moe_ccl_buffer_size <= 0 or self.mega_moe_num_max_tokens_per_rank <= 0:
                raise ValueError("MegaMoe sequence parallelism requires positive communication capacities")

    def is_moe_layer(self, layer_id: int) -> bool:
        return (
            self.num_experts > 0
            and (layer_id + 1) % self.decoder_sparse_step == 0
            and layer_id not in self.mlp_only_layers
        )

    def head_split(self) -> tuple[int, int]:
        return gqa_head_split(self.n_heads, self.n_kv_heads, self.tp_size)


class Qwen3_5Model(nn.Module):
    def __init__(self, cfg: Qwen3_5Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self._comm_policy = None
        if (
            device.type in ("npu", "privateuseone")
            and cfg.num_experts > 0
            and cfg.expert_parallel_degree == 2
            and not cfg.enable_mega_moe
        ):
            self._comm_policy = Eplv2CommPolicy.from_geometry(
                cfg.hidden_size,
                cfg.num_experts,
                cfg.ep_size,
                cfg.num_experts_per_tok,
                int(os.getenv("HCCL_BUFFSIZE", "200")),
                cfg.eplv2_mc2_max_tokens_per_rank,
            )
            if cfg.eplv2_comm_mode == "alltoall":
                self._comm_policy = Eplv2CommPolicy(mc2_capacity=0)
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size,
            cfg.hidden_size // cfg.tp_size,
            cfg.tp_size,
            dtype=dtype,
            device=device,
        )
        rotary_dim = int(cfg.head_dim * cfg.partial_rotary_factor)
        self.rotary = PartialRotaryEmbedding(
            cfg.head_dim,
            rotary_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            dtype,
            device,
        )
        decoder_layer_cls = get_qwen3_5_decoder_layer_class(device)
        self.layers = nn.ModuleList(
            decoder_layer_cls(cfg, i, dtype, device, self.rotary, comm_policy=self._comm_policy)
            if self._comm_policy is not None
            else decoder_layer_cls(cfg, i, dtype, device, self.rotary)
            for i in range(cfg.n_layers)
        )
        self.norm = decoder_layer_cls.normalization_cls(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        self.aux_hidden_capture = AuxHiddenCapture(cfg.layers_to_capture)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: object | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if kwargs:
            raise TypeError(f"unsupported Qwen3.5 execution inputs: {sorted(kwargs)}")
        if intermediate_tensors is not None:
            raise NotImplementedError("Qwen3.5 does not support pipeline parallelism")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids or inputs_embeds must be provided")

        if self._comm_policy is not None:
            self.layers[0].validate_ep_level2(self._comm_policy, force_mc2=self.cfg.eplv2_comm_mode == "mc2")

        sequence_parallel_enabled = self.cfg.sequence_parallel_enabled
        num_tokens = positions.shape[0] if positions.ndim == 1 else positions.shape[-1]
        sp_layout: SpMetadata | None = None
        if sequence_parallel_enabled:
            sp_layout = get_execution_context(SpMetadata)
            if sp_layout is None:
                raise RuntimeError("Qwen3.5 SP layout metadata is unavailable")
            first_moe_layer = next(
                (layer for layer in self.layers if layer.cfg.is_moe_layer(layer.layer_id)),
                None,
            )
            if first_moe_layer is not None:
                first_moe_layer.validate_sp(sp_layout, device=self.embed_tokens.weight.device)

        hidden = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)

        residual: torch.Tensor | None = None
        if sequence_parallel_enabled:
            # Validate the rows actually sharded on both input paths.
            if hidden.ndim != 2:
                raise ValueError(f"Qwen3.5 SP requires flattened token inputs, got shape {tuple(hidden.shape)}")
            if num_tokens != hidden.shape[0]:
                raise ValueError("Qwen3.5 SP positions must match the input token rows")
            hidden = sp_shard(hidden, sp_layout)
        aux_hidden_buffer = self.aux_hidden_capture.create_buffer(hidden)
        for layer_id, layer in enumerate(self.layers):
            hidden, residual = layer(hidden, residual, positions)
            record_layer_event(layer_id)
            self.aux_hidden_capture.capture_layer(layer_id, hidden, residual, aux_hidden_buffer)
        hidden, _ = self.norm(hidden, residual)
        if sequence_parallel_enabled:
            hidden = sp_all_gather(hidden, sp_layout)
            if aux_hidden_buffer is not None:
                aux_hidden_buffer = sp_all_gather(aux_hidden_buffer, sp_layout)
        return self.aux_hidden_capture.finalize(hidden, aux_hidden_buffer)


class Qwen3_5ForCausalLM(PyModelBase):
    model: Qwen3_5Model
    lm_head: ColumnParallelLinear

    def __init__(self, config: dict) -> None:
        super().__init__()
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "cuda"))
        self.cfg = Qwen3_5Config.prepare_execution_config(config, device, allow_speculative_decoding=True)
        self.dtype = dtype
        self.device = device
        self.model = Qwen3_5Model(self.cfg, dtype, device)  # pyright: ignore[reportIncompatibleVariableOverride]
        self.lm_head = ColumnParallelLinear(  # pyright: ignore[reportIncompatibleVariableOverride]
            self.cfg.hidden_size,
            self.cfg.vocab_size // self.cfg.tp_size,
            self.cfg.tp_size,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        all_weights = ScopedWeightLoader(
            state_dicts,
            src_prefixes=("model.language_model.", "model.", ""),
        )
        context = ParallelLoadContext.from_config(self.cfg, tp_rank, tp_size)
        load_causal_lm_weights(
            self.model,
            self.lm_head.weight,
            all_weights,
            context,
            tie_word_embeddings=self.cfg.tie_word_embeddings,
            embed_fallback=True,
        )
