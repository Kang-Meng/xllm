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

"""Kimi K3 hybrid KDA/Gated-MLA text model using mainline runtime owners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from xllm.python import distributed, kernels
from xllm.python.attention.backend import (
    MlaUnabsorbedPrefill,
    linear_state_checkpoint_stride,
    resolve_linear_state_io_indices,
)
from xllm.python.attention.kda_linear_attention import _get_host_q_cu_seq_lens
from xllm.python.layers import Attention, ColumnParallelLinear, HiddenParallelEmbedding, RMSNorm, RowParallelLinear
from xllm.python.layers.moe_parallel import TokenParallelLayout
from xllm.python.model_executor.forward_context import (
    get_execution_buffer,
    get_forward_context,
    in_acl_graph,
    record_layer_event,
    wait_for_layer_load,
)
from xllm.python.model_loader import ScopedWeightLoader, StateDictLike, shard_tensor
from xllm.python.models.base import PyModelBase
from xllm.python.models.deepseek_v32 import W8A8DynamicLinear
from xllm.python.models.kimi_k3_kda import KDA_CHUNK_SIZE, KimiK3DeltaAttention, KimiK3KDAMetadata, _build_chunk_indices
from xllm.python.models.kimi_k3_moe import KimiK3MoE, SituAndMul, _global_prefill_step, _validate_situ_activation


@dataclass
class KimiK3TextConfig:
    hidden_size: int = 7168
    n_layers: int = 93
    n_heads: int = 96
    n_kv_heads: int = 96
    head_dim: int = 128
    intermediate_size: int = 33792
    vocab_size: int = 163840
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 1048576
    hidden_act: str = "situ"
    tie_word_embeddings: bool = False
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    num_experts: int | None = 896
    num_experts_per_token: int | None = 16
    num_shared_experts: int = 2
    use_grouped_topk: bool = True
    num_expert_group: int = 1
    topk_group: int = 1
    topk_method: str = "noaux_tc"
    moe_intermediate_size: int | None = 3072
    routed_expert_hidden_size: int | None = 3584
    moe_renormalize: bool = True
    moe_router_activation_func: str = "sigmoid"
    routed_scaling_factor: float = 1.0
    latent_moe_use_norm: bool = True
    activation_situ_beta: float | None = 4.0
    activation_situ_linear_beta: float | None = 25.0
    attn_res_block_size: int = 12
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mla_use_nope: bool = True
    mla_use_rope: bool = False
    mla_use_output_gate: bool = True
    num_nextn_predict_layers: int = 0
    logit_scale: float | None = None
    linear_attn_config: dict = field(default_factory=dict)
    quantize_type: str = ""
    quant_method: str = ""
    quant_version: str = ""
    quant_group_size: int = 0
    tp_size: int = 1
    tp_rank: int = 0
    world_size: int = 1
    rank: int = 0
    ep_size: int = 1
    dp_size: int = 1
    dp_rank: int = 0
    mc2_tokens_capacity: int = 512
    enable_flashcomm1: bool = True
    enable_prefix_cache: bool = True

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> KimiK3TextConfig:
        raw = config.get("text_config", config)
        if not isinstance(raw, dict):
            raise TypeError("Kimi K3 text_config must be a dictionary")

        def pick(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in raw:
                    return raw[key]
            for key in keys:
                if key in config:
                    return config[key]
            return default

        linear_attention = raw.get("linear_attn_config", {})
        if not isinstance(linear_attention, dict):
            linear_attention = {}
        n_heads = int(pick("n_heads", "num_attention_heads", default=96))
        hidden_size = int(pick("hidden_size", default=7168))
        configured_head_dim = pick("head_dim")
        head_dim = int(linear_attention.get("head_dim", 128) if configured_head_dim is None else configured_head_dim)
        tp_size = int(config.get("tp_size", 1))
        tp_rank = int(config.get("tp_rank", 0))
        return cls(
            hidden_size=hidden_size,
            n_layers=int(pick("n_layers", "num_hidden_layers", default=93)),
            n_heads=n_heads,
            n_kv_heads=int(pick("n_kv_heads", "num_key_value_heads", default=n_heads)),
            head_dim=head_dim,
            intermediate_size=int(pick("intermediate_size", default=33792)),
            vocab_size=int(pick("vocab_size", default=163840)),
            rms_norm_eps=float(pick("rms_norm_eps", default=1e-5)),
            max_position_embeddings=int(pick("max_position_embeddings", default=1048576)),
            hidden_act=str(pick("hidden_act", default="situ")),
            tie_word_embeddings=bool(pick("tie_word_embeddings", default=False)),
            first_k_dense_replace=int(pick("first_k_dense_replace", default=1)),
            moe_layer_freq=int(pick("moe_layer_freq", default=1)),
            num_experts=(
                None
                if pick("num_experts", "n_routed_experts", default=896) is None
                else int(pick("num_experts", "n_routed_experts", default=896))
            ),
            num_experts_per_token=(
                None
                if pick("num_experts_per_token", "num_experts_per_tok", default=16) is None
                else int(pick("num_experts_per_token", "num_experts_per_tok", default=16))
            ),
            num_shared_experts=int(pick("num_shared_experts", "n_shared_experts", default=2)),
            use_grouped_topk=bool(pick("use_grouped_topk", default=True)),
            num_expert_group=int(pick("num_expert_group", "n_group", default=1)),
            topk_group=int(pick("topk_group", default=1)),
            topk_method=str(pick("topk_method", default="noaux_tc")),
            moe_intermediate_size=(
                None
                if pick("moe_intermediate_size", default=3072) is None
                else int(pick("moe_intermediate_size", default=3072))
            ),
            routed_expert_hidden_size=(
                None
                if pick("routed_expert_hidden_size", default=3584) is None
                else int(pick("routed_expert_hidden_size", default=3584))
            ),
            moe_renormalize=bool(pick("moe_renormalize", "norm_topk_prob", default=True)),
            moe_router_activation_func=str(pick("moe_router_activation_func", default="sigmoid")),
            routed_scaling_factor=float(pick("routed_scaling_factor", default=1.0)),
            latent_moe_use_norm=bool(pick("latent_moe_use_norm", default=True)),
            activation_situ_beta=(
                None
                if pick("activation_situ_beta", default=4.0) is None
                else float(pick("activation_situ_beta", default=4.0))
            ),
            activation_situ_linear_beta=(
                None
                if pick("activation_situ_linear_beta", default=25.0) is None
                else float(pick("activation_situ_linear_beta", default=25.0))
            ),
            attn_res_block_size=int(pick("attn_res_block_size", default=12)),
            q_lora_rank=int(pick("q_lora_rank", default=1536)),
            kv_lora_rank=int(pick("kv_lora_rank", default=512)),
            qk_nope_head_dim=int(pick("qk_nope_head_dim", default=128)),
            qk_rope_head_dim=int(pick("qk_rope_head_dim", default=64)),
            v_head_dim=int(pick("v_head_dim", default=128)),
            mla_use_nope=bool(pick("mla_use_nope", default=True)),
            mla_use_rope=bool(pick("mla_use_rope", default=False)),
            mla_use_output_gate=bool(pick("mla_use_output_gate", default=True)),
            num_nextn_predict_layers=int(pick("num_nextn_predict_layers", default=0)),
            logit_scale=(
                None if pick("logit_scale", default=None) is None else float(pick("logit_scale", default=None))
            ),
            linear_attn_config=dict(linear_attention),
            quantize_type=str(config.get("quantize_type", "")),
            quant_method=str(config.get("quant_method", "")),
            quant_version=str(config.get("quant_version", "")),
            quant_group_size=int(config.get("quant_group_size", 0)),
            tp_size=tp_size,
            tp_rank=tp_rank,
            world_size=int(config.get("world_size", tp_size)),
            rank=int(config.get("rank", tp_rank)),
            ep_size=int(config.get("ep_size", raw.get("ep_size", 1))),
            dp_size=int(config.get("dp_size", raw.get("dp_size", 1))),
            dp_rank=int(config.get("dp_rank", raw.get("dp_rank", 0))),
            mc2_tokens_capacity=int(pick("mc2_tokens_capacity", default=512)),
            enable_flashcomm1=bool(pick("enable_flashcomm1", default=True)),
            enable_prefix_cache=bool(pick("enable_prefix_cache", default=True)),
        )

    def validate(self) -> None:
        if self.tp_size <= 0 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("Kimi K3 TP rank and size are invalid")
        if self.hidden_size <= 0 or self.hidden_size % self.tp_size != 0:
            raise ValueError("Kimi K3 hidden_size must be positive and divisible by tp_size")
        if self.vocab_size <= 0 or self.vocab_size % self.tp_size != 0:
            raise ValueError("Kimi K3 vocab_size must be positive and divisible by tp_size")
        if self.n_layers <= 0 or self.n_heads <= 0 or self.head_dim <= 0:
            raise ValueError("Kimi K3 layer and attention dimensions must be positive")
        if self.rms_norm_eps <= 0:
            raise ValueError("Kimi K3 rms_norm_eps must be positive")
        if self.n_heads % self.tp_size != 0:
            raise ValueError("Kimi K3 attention heads must be divisible by tp_size")
        if self.n_kv_heads % self.tp_size != 0:
            raise ValueError("Kimi K3 key/value heads must be divisible by tp_size")
        if self.intermediate_size <= 0 or self.intermediate_size % self.tp_size != 0:
            raise ValueError("Kimi K3 intermediate_size must be divisible by tp_size")
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("Kimi K3 global rank and size are invalid")
        if self.ep_size <= 0 or self.world_size % self.ep_size != 0:
            raise ValueError("Kimi K3 EP size must divide world size")
        if self.dp_size <= 0 or self.world_size % self.dp_size != 0:
            raise ValueError("Kimi K3 DP size must divide world size")
        if not 0 <= self.dp_rank < self.dp_size:
            raise ValueError("Kimi K3 DP rank and size are invalid")
        if self.ep_size > 1 and self.dp_size != self.ep_size:
            supported = self.dp_size == 1 and self.tp_size == self.ep_size and self.world_size == self.ep_size
            supported = supported or (
                self.ep_size == self.world_size and self.world_size == self.dp_size * self.tp_size
            )
            if not supported:
                raise ValueError("Kimi K3 supports EP with either dp=ep or dp=1, attention_tp=ep")
        if self.moe_layer_freq <= 0 or self.first_k_dense_replace < 0:
            raise ValueError("Kimi K3 MoE layer placement is invalid")
        if self.attn_res_block_size <= 0:
            raise ValueError("Kimi K3 attn_res_block_size must be positive")
        _validate_situ_activation(
            1.0 if self.activation_situ_beta is None else self.activation_situ_beta,
            self.activation_situ_linear_beta,
        )
        if self.hidden_act not in ("situ", "silu"):
            raise ValueError(f"Unsupported Kimi K3 activation: {self.hidden_act}")
        if self.moe_router_activation_func not in ("sigmoid", "softmax"):
            raise ValueError("Kimi K3 router activation must be sigmoid or softmax")
        if self.num_experts is not None:
            if self.num_experts_per_token is None or self.moe_intermediate_size is None:
                raise ValueError("Kimi K3 MoE dimensions are incomplete")
            if self.routed_expert_hidden_size is None:
                raise ValueError("Kimi K3 routed_expert_hidden_size is required")
            if self.moe_intermediate_size % self.tp_size != 0:
                raise ValueError("Kimi K3 MoE intermediate_size must divide tp_size")
            if not 0 < self.num_experts_per_token <= self.num_experts:
                raise ValueError("Kimi K3 num_experts_per_token must be within num_experts")
            if self.use_grouped_topk:
                if self.num_expert_group <= 0:
                    raise ValueError("Kimi K3 num_expert_group must be positive")
                if self.num_experts % self.num_expert_group != 0:
                    raise ValueError("Kimi K3 experts must divide evenly into expert groups")
                if not 0 < self.topk_group <= self.num_expert_group:
                    raise ValueError("Kimi K3 topk_group must be within expert groups")
        if self.num_nextn_predict_layers < 0:
            raise ValueError("Kimi K3 num_nextn_predict_layers must be non-negative")
        if self.logit_scale is not None and self.logit_scale <= 0:
            raise ValueError("Kimi K3 logit_scale must be positive")
        mla_dimensions = {
            "q_lora_rank": self.q_lora_rank,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
        }
        invalid_mla = [name for name, value in mla_dimensions.items() if value <= 0]
        if invalid_mla:
            raise ValueError(f"Kimi K3 MLA dimensions must be positive: {invalid_mla}")
        if not self.mla_use_nope:
            raise ValueError("Kimi K3 MLA requires mla_use_nope")
        if self.mla_use_rope:
            raise ValueError("Kimi K3 MLA does not apply RoPE")
        if not self.mla_use_output_gate:
            raise ValueError("Kimi K3 MLA requires output gating")
        kda_layers = tuple(int(layer) for layer in self.linear_attn_config.get("kda_layers", ()))
        full_attn_layers = tuple(int(layer) for layer in self.linear_attn_config.get("full_attn_layers", ()))
        layer_numbers = kda_layers + full_attn_layers
        if len(set(layer_numbers)) != len(layer_numbers):
            raise ValueError("Kimi K3 attention layer lists must not overlap or repeat")
        if any(layer < 1 or layer > self.n_layers for layer in layer_numbers):
            raise ValueError("Kimi K3 attention layer numbers must be within [1, n_layers]")
        if full_attn_layers and len(layer_numbers) != self.n_layers:
            raise ValueError("Kimi K3 kda_layers and full_attn_layers must cover all layers")
        if self.uses_quantized_weights:
            if self.quant_version != "1.0.0":
                raise ValueError("Kimi K3 W4A8 weights require quant_version 1.0.0")
            if self.quant_group_size != 0:
                raise ValueError("Kimi K3 currently supports per-channel W4A8 weights only")
            if self.num_experts is None or self.routed_expert_hidden_size is None or self.moe_intermediate_size is None:
                raise ValueError("Kimi K3 W4A8 requires routed experts")
            if 16 % self.tp_size != 0:
                raise ValueError("Kimi K3 W4A8 scale_bias requires tp_size <= 16")
            if self.routed_expert_hidden_size % 2 != 0:
                raise ValueError("Kimi K3 W4A8 routed hidden size must be even")
            if self.moe_intermediate_size % (2 * self.tp_size) != 0:
                raise ValueError("Kimi K3 W4A8 expert size must be divisible by 2 * tp_size")

    @property
    def uses_quantized_weights(self) -> bool:
        quantize_type = self.quantize_type.lower()
        quant_method = self.quant_method.lower()
        return quantize_type == "w4a8_dynamic" or quant_method == "ascend_int4"

    def is_moe_layer(self, layer_id: int) -> bool:
        return (
            self.num_experts is not None
            and layer_id >= self.first_k_dense_replace
            and layer_id % self.moe_layer_freq == 0
        )

    @property
    def use_sequence_parallel(self) -> bool:
        """FlashComm1 sequence parallelism gate.

        The residual trunk is sharded over the attention TP group, so it only
        applies when TP is active. It composes with DP and EP: TP is the
        innermost rank dimension, so a TP group stays inside one DP cell and
        shares its batch; and the MoE always returns output replicated across
        the attention TP group (the trunk layout contract), so the EP path is
        treated as a black box — FlashComm1 optimizes only the TP part
        (attention o_proj + residual trunk) and leaves the MoE's own EP
        communication untouched.
        """
        return self.enable_flashcomm1 and self.tp_size > 1

    def is_kda_layer(self, layer_id: int) -> bool:
        """``layer_id`` is 0-based; the config's ``kda_layers`` is 1-based."""
        kda_layers = self.linear_attn_config.get("kda_layers") or []
        return (layer_id + 1) in kda_layers

    def is_mla_layer(self, layer_id: int) -> bool:
        if not 0 <= layer_id < self.n_layers:
            raise ValueError(f"Kimi K3 layer id is out of range: {layer_id}")
        full_attn_layers = self.linear_attn_config.get("full_attn_layers") or []
        if full_attn_layers:
            return (layer_id + 1) in full_attn_layers
        return not self.is_kda_layer(layer_id)


class _KimiW8A8Linear(W8A8DynamicLinear):
    """Mainline W8A8 math with the existing Kimi MoE loader protocol."""

    def __init__(self, in_features: int, out_features: int, device: torch.device) -> None:
        super().__init__(in_features, out_features, device, transpose_weight_after_loading=False)
        self._processed = False

    def load_weight(self, name: str, tensor: torch.Tensor) -> bool:
        if name not in ("weight", "weight_scale", "weight_offset"):
            return False
        if self._processed:
            raise RuntimeError("Kimi W8A8 cannot reload packed weights")
        target = getattr(self, name)
        if name != "weight" and tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        if tensor.shape != target.shape:
            raise ValueError(f"Kimi W8A8 {name} shape {tuple(tensor.shape)} != {tuple(target.shape)}")
        with torch.no_grad():
            target.copy_(tensor)
        return True

    def finish_weight_loading(self) -> None:
        if not self._processed:
            self.weight.data = self.weight.detach().t().contiguous()
            self._weight_is_transposed = True
            self.process_weights_after_loading()
            self._processed = True


def _projection(
    in_features: int,
    out_features: int,
    tp_size: int,
    *,
    quantized: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    if quantized:
        if dtype != torch.bfloat16:
            raise ValueError("Kimi W8A8 requires bfloat16 output")
        return _KimiW8A8Linear(in_features, out_features, device)
    return ColumnParallelLinear(in_features, out_features, tp_size, dtype=dtype, device=device)


def _load_projection(
    module: nn.Module,
    weights: ScopedWeightLoader,
    prefix: str,
    tp_rank: int = 0,
    tp_size: int = 1,
    shard_dim: int | None = None,
) -> set[str]:
    loaded: set[str] = set()
    suffixes = ("weight", "weight_scale", "weight_offset") if isinstance(module, W8A8DynamicLinear) else ("weight",)
    for suffix in suffixes:
        name = prefix + suffix
        tensor = weights.get_tensor(name)
        # Row-parallel scales are replicated; column-parallel scales follow
        # output rows. Never shard per-channel scale on the input dimension.
        dim = shard_dim if suffix == "weight" else (0 if shard_dim == 0 else None)
        if suffix != "weight" and tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        if dim is not None:
            tensor = weights.shard_value(tensor, name, dim, tp_rank, tp_size)
        weights.copy(getattr(module, suffix), tensor, name)
        loaded.add(name)
    if isinstance(module, _KimiW8A8Linear):
        module.finish_weight_loading()
    elif isinstance(module, RowParallelLinear):
        module.process_weights_after_loading()
    return loaded


class _MoEStateDict:
    """Scoped checkpoint view implementing the already merged MoE contract.

    Only MoE requires enumerating expert tensors and nested StateDict views.
    All source lookup, copying and TP slicing still belong to ScopedWeightLoader.
    """

    def __init__(self, weights: ScopedWeightLoader, names: tuple[str, ...]) -> None:
        self.weights = weights
        self._names = names

    def has(self, name: str) -> bool:
        return self.weights.has(name)

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.weights.get_tensor(name)

    def keys(self) -> tuple[str, ...]:
        return self._names

    def size(self) -> int:
        return len(self._names)

    def get_dict_with_prefix(self, prefix: str) -> _MoEStateDict:
        return _MoEStateDict(
            self.weights.with_prefix(prefix),
            tuple(name[len(prefix) :] for name in self._names if name.startswith(prefix)),
        )

    def get_sharded_tensor(self, name: str, dim: int, rank: int, world_size: int) -> torch.Tensor:
        return self.weights.shard_value(self.get_tensor(name), name, dim, rank, world_size)


class KimiK3MLP(nn.Module):
    """Dense/shared SiTU MLP with mainline linear and quantization kernels."""

    def __init__(
        self,
        config: KimiK3TextConfig,
        dtype: torch.dtype,
        device: torch.device,
        intermediate_size: int | None = None,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        size = config.intermediate_size if intermediate_size is None else intermediate_size
        if size <= 0 or size % config.tp_size:
            raise ValueError("Kimi MLP intermediate size must divide TP size")
        self.tp_size, self.tp_rank = config.tp_size, config.tp_rank
        self.quantized = config.uses_quantized_weights
        self.reduce_results = reduce_results
        self.gate_up_proj = _projection(
            config.hidden_size,
            2 * (size // config.tp_size),
            config.tp_size,
            quantized=self.quantized,
            dtype=dtype,
            device=device,
        )
        if self.quantized:
            self.down_proj = _KimiW8A8Linear(size // config.tp_size, config.hidden_size, device)
        else:
            self.down_proj = RowParallelLinear(
                size // config.tp_size,
                config.hidden_size,
                config.tp_size,
                dtype=dtype,
                device=device,
                reduce_results=False,
            )
        beta = 1.0 if config.activation_situ_beta is None else config.activation_situ_beta
        self.activation = SituAndMul(beta, config.activation_situ_linear_beta) if config.hidden_act == "situ" else None
        self._loaded = False

    def forward(self, hidden_states: torch.Tensor, reduce_results: bool | None = None) -> torch.Tensor:
        projected = self.gate_up_proj(hidden_states)
        activated = self.activation(projected) if self.activation is not None else kernels.silu_and_mul(projected)
        output = self.down_proj(activated)
        if self.tp_size > 1 and (self.reduce_results if reduce_results is None else reduce_results):
            distributed.all_reduce_(output)
        return output

    def load_weights(self, state_dict: _MoEStateDict, tp_rank: int, tp_size: int) -> set[str]:
        weights = state_dict.weights
        suffixes = ("weight", "weight_scale", "weight_offset") if self.quantized else ("weight",)
        loaded: set[str] = set()
        for suffix in suffixes:
            names = (f"gate_proj.{suffix}", f"up_proj.{suffix}")
            values = []
            for name in names:
                tensor = weights.get_tensor(name)
                if suffix != "weight" and tensor.ndim == 1:
                    tensor = tensor.unsqueeze(-1)
                values.append(weights.shard_value(tensor, name, 0, tp_rank, tp_size))
            weights.copy(getattr(self.gate_up_proj, suffix), torch.cat(values), f"gate_up_proj.{suffix}")
            loaded.update(names)
        loaded.update(_load_projection(self.down_proj, weights, "down_proj.", tp_rank, tp_size, 1))
        if self.quantized:
            self.gate_up_proj.finish_weight_loading()
        self._loaded = True
        return loaded

    def finish_weight_loading(self) -> None:
        if not self._loaded:
            raise KeyError("Kimi MLP checkpoint weights were not loaded")


class KimiK3KDAAttention(Attention):
    """KDA layer consuming the executor's named linear caches and metadata."""

    attention_kind = "linear"

    def __init__(self, config: KimiK3TextConfig, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        heads = int(config.linear_attn_config["num_heads"])
        dim = int(config.linear_attn_config["head_dim"])
        super().__init__(heads, heads, dim, dim**-0.5, 0, layer_id)
        self.kda = KimiK3DeltaAttention(
            config.hidden_size,
            config.linear_attn_config,
            tp_size=config.tp_size,
            tp_rank=config.tp_rank,
            rms_norm_eps=config.rms_norm_eps,
            quantized=config.uses_quantized_weights,
            dtype=dtype,
            device=device,
        )

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor, reduce_results: bool = True
    ) -> torch.Tensor:
        del positions
        context = get_forward_context()
        if context.cp_context is not None or context.metadata.is_spec_verify:
            raise NotImplementedError("Kimi KDA currently supports non-speculative execution without CP")
        cache = context.layer_caches[self.layer_id]
        if cache.conv is None or cache.ssm is None:
            raise RuntimeError(f"Kimi KDA layer {self.layer_id} requires framework conv/ssm caches")
        read_indices, write_indices = resolve_linear_state_io_indices(context.metadata)
        if write_indices is None:
            raise RuntimeError("Kimi KDA requires framework linear-state slots")
        # C++ exposes channel-last conv history and VK state. The SSM cache
        # reserves checkpoints per slot; checkpoint zero addresses each slot's live state.
        conv = cache.conv[:, : self.kda.conv_size - 1]
        stride = linear_state_checkpoint_stride(cache.conv, cache.ssm)
        state = cache.ssm[::stride]
        if state.shape[1:] != self.kda.recurrent_state_shape():
            raise RuntimeError("Kimi KDA framework state must use local-head VK layout")
        if read_indices is not write_indices:
            # Prefix continuation restores from read slots before advancing the
            # caller-owned live/write slots. No cache ownership escapes forward.
            valid = write_indices > 0
            writes, reads = write_indices[valid].long(), read_indices[valid].long()
            conv.index_copy_(0, writes, conv.index_select(0, reads))
            state.index_copy_(0, writes, state.index_select(0, reads))
        key = KimiK3KDAMetadata
        metadata = context.layer_shared_cache.get(key)
        if metadata is None:
            raw = context.metadata
            prefill = raw.is_prefill or raw.is_chunked_prefill
            num_seqs = write_indices.numel()
            starts = raw.q_cu_seq_lens
            if not prefill and starts is None:
                starts = get_execution_buffer(
                    ("KIMI_KDA_DECODE_CU_SEQLENS", num_seqs),
                    lambda: torch.arange(num_seqs + 1, dtype=torch.int32, device=hidden_states.device),
                )
            host_cu = host_chunks = None
            conv_slots = conv_initial = conv_starts = None
            if prefill:
                host_values = _get_host_q_cu_seq_lens(raw, num_seqs)
                host_cu = tuple(int(value) for value in host_values)
                host_chunks = tuple(_build_chunk_indices(list(host_cu), KDA_CHUNK_SIZE))
                if raw.has_initial_state is None or raw.has_initial_state.shape != (num_seqs,):
                    raise RuntimeError("Kimi KDA prefill requires one initial-state flag per sequence")
                conv_slots = write_indices.detach().cpu().tolist()
                # Convert booleans to the host ABI's integers after the single
                # D2H transfer; no device-side int64 cast is needed.
                conv_initial = [int(value) for value in raw.has_initial_state.detach().cpu().tolist()]
                conv_starts = list(host_cu)
            if starts is None or starts.shape != (num_seqs + 1,):
                raise RuntimeError("Kimi KDA requires one cumulative query boundary per state slot")
            metadata = KimiK3KDAMetadata(
                starts,
                # Prefill warmup uses the framework's reserved scratch slot 0
                # with has_initial_state=False. Chunk KDA gathers its initial
                # state, so it requires a valid index even for this scratch slot.
                # Decode instead uses -1 to keep padding from updating any state.
                write_indices if prefill else torch.where(write_indices > 0, write_indices, -1),
                0 if prefill else num_seqs,
                num_seqs if prefill else 0,
                raw.has_initial_state,
                host_cu,
                host_chunks,
                conv_slots,
                conv_initial,
                conv_starts,
            )
            context.layer_shared_cache[key] = metadata
        return self.kda(hidden_states, metadata, conv, state, reduce_o_proj=reduce_results)

    def load_weights(self, weights: ScopedWeightLoader) -> set[str]:
        loaded = self.kda.load_weights(weights)
        self.kda.process_weights_after_loading()
        return loaded


class KimiK3MLAAttention(Attention):
    """No-RoPE gated MLA with the mainline paged absorbed-MLA backend."""

    attention_kind = "mla"
    requires_dense_mla_head_padding = True

    def __init__(self, config: KimiK3TextConfig, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        heads = config.n_heads // config.tp_size
        super().__init__(
            heads, 1, config.kv_lora_rank, (config.qk_nope_head_dim + config.qk_rope_head_dim) ** -0.5, 0, layer_id
        )
        self.cfg = config
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, config.rms_norm_eps, dtype=dtype, device=device)
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, config.rms_norm_eps, dtype=dtype, device=device)
        qkv_size = config.q_lora_rank + config.kv_lora_rank + config.qk_rope_head_dim
        self.qkv_a_proj = _projection(
            config.hidden_size, qkv_size, 1, quantized=config.uses_quantized_weights, dtype=dtype, device=device
        )
        self.q_b_proj = _projection(
            config.q_lora_rank,
            heads * (config.qk_nope_head_dim + config.qk_rope_head_dim),
            config.tp_size,
            quantized=config.uses_quantized_weights,
            dtype=dtype,
            device=device,
        )
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.g_proj = nn.Linear(config.hidden_size, heads * config.v_head_dim, bias=False, dtype=dtype, device=device)
        self.o_proj = RowParallelLinear(
            heads * config.v_head_dim,
            config.hidden_size,
            config.tp_size,
            dtype=dtype,
            device=device,
            reduce_results=False,
        )
        self.register_buffer(
            "W_UK",
            torch.empty(heads, config.qk_nope_head_dim, config.kv_lora_rank, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "W_UV",
            torch.empty(heads, config.kv_lora_rank, config.v_head_dim, dtype=dtype, device=device),
            persistent=False,
        )

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor, reduce_results: bool = True
    ) -> torch.Tensor:
        del positions  # The named positional slice is retained without rotation.
        cfg = self.cfg
        rows = hidden_states.shape[0]
        q_lora, compressed_kv = self.qkv_a_proj(hidden_states).split(
            [cfg.q_lora_rank, cfg.kv_lora_rank + cfg.qk_rope_head_dim], -1
        )
        query = self.q_b_proj(self.q_a_layernorm(q_lora)).view(rows, self.num_heads, -1)
        q_nope, q_pe = query.split([cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], -1)
        q_latent = kernels.batch_matmul_transpose(q_nope, self.W_UK)
        latent, k_pe = compressed_kv.split([cfg.kv_lora_rank, cfg.qk_rope_head_dim], -1)
        k_latent = self.kv_a_layernorm(latent).unsqueeze(1)
        raw = get_forward_context().metadata
        unabsorbed_prefill = None
        if raw.is_prefill and not raw.is_chunked_prefill:
            kv = self.kv_b_proj(k_latent.squeeze(1)).view(rows, self.num_heads, -1)
            k_nope, value = kv.split([cfg.qk_nope_head_dim, cfg.v_head_dim], dim=-1)
            unabsorbed_prefill = MlaUnabsorbedPrefill(q_nope, k_nope, value)
        output = get_forward_context().attention_backend.execute_mla(
            q_latent,
            q_pe,
            k_latent,
            k_pe.unsqueeze(1),
            self,
            topk=None,
            unabsorbed_prefill=unabsorbed_prefill,
        )
        values = (
            output if unabsorbed_prefill is not None else kernels.batch_matmul_transpose(output, self.W_UV)
        ).reshape(rows, -1)
        gated = values * self.g_proj(hidden_states).sigmoid()
        result = self.o_proj(gated)
        if cfg.tp_size > 1 and reduce_results:
            distributed.all_reduce_(result)
        return result

    def load_weights(self, weights: ScopedWeightLoader) -> set[str]:
        cfg = self.cfg
        loaded: set[str] = set()
        suffixes = ("weight", "weight_scale", "weight_offset") if cfg.uses_quantized_weights else ("weight",)
        for suffix in suffixes:
            names = (f"q_a_proj.{suffix}", f"kv_a_proj_with_mqa.{suffix}")
            values = [weights.get_tensor(name) for name in names]
            if suffix != "weight":
                values = [value.unsqueeze(-1) if value.ndim == 1 else value for value in values]
            weights.copy(getattr(self.qkv_a_proj, suffix), torch.cat(values), f"qkv_a_proj.{suffix}")
            loaded.update(names)
        if isinstance(self.qkv_a_proj, _KimiW8A8Linear):
            self.qkv_a_proj.finish_weight_loading()
        loaded.update(_load_projection(self.q_b_proj, weights, "q_b_proj.", cfg.tp_rank, cfg.tp_size, 0))
        for name, module, dim in (
            ("kv_b_proj", self.kv_b_proj, 0),
            ("g_proj", self.g_proj, 0),
            ("o_proj", self.o_proj, 1),
        ):
            key = f"{name}.weight"
            tensor = weights.get_tensor(key)
            if not tensor.is_floating_point():
                scale_key, offset_key = f"{key}_scale", f"{key}_offset"
                scale, offset = weights.get_tensor(scale_key), weights.get_tensor(offset_key)
                if scale.ndim == 1:
                    scale, offset = scale.unsqueeze(-1), offset.unsqueeze(-1)
                tensor = (tensor.float() - offset.float()) * scale.float()
                loaded.update((scale_key, offset_key))
            tensor = weights.shard_value(tensor, key, dim, cfg.tp_rank, cfg.tp_size)
            weights.copy(module.weight, tensor, key)
            loaded.add(key)
        for name in ("q_a_layernorm", "kv_a_layernorm"):
            key = f"{name}.weight"
            weights.load_tensor(getattr(self, name).weight, key)
            loaded.add(key)
        weight = self.kv_b_proj.weight.detach().view(
            self.num_heads, cfg.qk_nope_head_dim + cfg.v_head_dim, cfg.kv_lora_rank
        )
        uk, uv = weight.split([cfg.qk_nope_head_dim, cfg.v_head_dim], 1)
        self.W_UK.copy_(uk)
        self.W_UV.copy_(uv.transpose(1, 2))
        self.o_proj.process_weights_after_loading()
        return loaded


def _apply_attention_residual(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    projection: nn.Linear,
    norm: RMSNorm,
) -> torch.Tensor:
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    values_float = values.float()
    # Keep the learned residual mixture in FP32 while avoiding separate
    # normalization kernels and a materialized weighted-values tensor.
    normed = kernels.rms_norm(values_float, norm.weight.float(), norm.eps)
    scores = F.linear(normed, projection.weight.float()).squeeze(-1)
    probabilities = torch.softmax(scores, dim=-1).unsqueeze(1)
    return torch.matmul(probabilities, values_float).squeeze(1).to(values.dtype)


def _sp_active(sp_flag: bool) -> bool:
    """FlashComm1 sequence-parallel is prefill-only (matching the C++ gate).
    During ACL graph capture/warmup (decode) it must be disabled so that
    graph-incompatible collective ops (TP all-gather/reduce-scatter) are
    not recorded into the static graph."""
    if not sp_flag:
        return False
    try:
        ctx = get_forward_context()
        if in_acl_graph():
            return False
        metadata = ctx.metadata
        return bool(metadata is not None and (metadata.is_prefill or metadata.is_chunked_prefill))
    except (RuntimeError, AttributeError):
        return False


class KimiK3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: KimiK3TextConfig,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.is_kda = config.is_kda_layer(layer_id)
        self._sp = config.use_sequence_parallel
        self.tp_size = config.tp_size
        self.tp_rank = config.tp_rank
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        if self.is_kda:
            self.self_attn = KimiK3KDAAttention(config, layer_id, dtype, device)
        elif config.is_mla_layer(layer_id):
            self.self_attn = KimiK3MLAAttention(config, layer_id, dtype, device)
        else:
            raise ValueError(f"Kimi K3 layer {layer_id} has no attention implementation")
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        if config.is_moe_layer(layer_id):
            if config.uses_quantized_weights:
                routed_expert_down_proj = _projection(
                    config.hidden_size,
                    config.routed_expert_hidden_size,
                    1,
                    dtype=dtype,
                    device=device,
                    quantized=True,
                )
                routed_expert_up_proj = _projection(
                    config.routed_expert_hidden_size,
                    config.hidden_size,
                    1,
                    dtype=dtype,
                    device=device,
                    quantized=True,
                )
            else:
                routed_expert_down_proj = nn.Linear(
                    config.hidden_size,
                    config.routed_expert_hidden_size,
                    bias=False,
                    dtype=dtype,
                    device=device,
                )
                routed_expert_up_proj = nn.Linear(
                    config.routed_expert_hidden_size,
                    config.hidden_size,
                    bias=False,
                    dtype=dtype,
                    device=device,
                )
            shared_experts = (
                KimiK3MLP(
                    config,
                    dtype,
                    device,
                    intermediate_size=config.moe_intermediate_size * config.num_shared_experts,
                    # KimiK3MoERunner explicitly reduces the shared branch in
                    # both decode and prefill, so keep this projection partial.
                    reduce_results=False,
                )
                if config.num_shared_experts
                else None
            )
            self.block_sparse_moe = KimiK3MoE(
                config,
                dtype,
                device,
                config.tp_size,
                config.tp_rank,
                routed_expert_down_proj=routed_expert_down_proj,
                routed_expert_up_proj=routed_expert_up_proj,
                shared_experts=shared_experts,
                quantized=config.uses_quantized_weights,
            )
            self.block_sparse_moe.layer_id = layer_id
        else:
            self.mlp = KimiK3MLP(
                config,
                dtype,
                device,
                reduce_results=True,
            )
        self.attn_res_block_size = config.attn_res_block_size
        self.self_attention_res_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        self.mlp_res_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        self.self_attention_res_proj = nn.Linear(
            config.hidden_size,
            1,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.mlp_res_proj = nn.Linear(
            config.hidden_size,
            1,
            bias=False,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # With FlashComm1 the residual trunk is sequence-parallel: hidden_states
        # and block_residual carry this rank's token shard. Per-token ops
        # (norms, hyper-connection residual math) run on the shard directly;
        # attention and the FFN need the full token set, so we gather before
        # them and shard their outputs back.
        wait_for_layer_load(self.layer_id)
        num_tokens = positions.shape[0]
        sp_active = _sp_active(self._sp)
        layout = TokenParallelLayout(num_tokens, self.tp_size, self.tp_rank) if sp_active else None

        prefix_sum: torch.Tensor | None = hidden_states
        if block_residual.shape[1] > 0:
            hidden_states = _apply_attention_residual(
                prefix_sum,
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            )
        if self.layer_id % self.attn_res_block_size == 0:
            block_residual = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
            prefix_sum = None
        hidden_states = self.input_layernorm(hidden_states)
        attention_input = layout.gather(hidden_states) if layout is not None else hidden_states
        attention_output = self.self_attn(attention_input, positions, reduce_results=not sp_active)
        if sp_active:
            attention_output = layout.reduce_scatter(attention_output)
        prefix_sum = attention_output if prefix_sum is None else prefix_sum + attention_output
        hidden_states = _apply_attention_residual(
            prefix_sum,
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        if hasattr(self, "block_sparse_moe"):
            # Routed all-to-all can consume the sequence-parallel shard
            # directly.  The legacy all-gather path still gathers full tokens
            # and shards the result back after MoE execution.
            sequence_parallel_routed = getattr(self.block_sparse_moe, "sequence_parallel_routed", False)
            global_prefill_moe = sequence_parallel_routed and _global_prefill_step(self.config.dp_size)
            if global_prefill_moe or (sp_active and sequence_parallel_routed):
                hidden_states = self.block_sparse_moe(
                    hidden_states,
                    sequence_parallel_tokens=num_tokens,
                )
            elif sp_active:
                moe_input = layout.gather(hidden_states)
                hidden_states = layout.shard(self.block_sparse_moe(moe_input))
            else:
                hidden_states = self.block_sparse_moe(hidden_states)
        else:
            if sp_active:
                mlp_input = layout.gather(hidden_states)
                hidden_states = layout.reduce_scatter(self.mlp(mlp_input, reduce_results=False))
            else:
                hidden_states = self.mlp(hidden_states, reduce_results=not sp_active)
        output = prefix_sum + hidden_states
        record_layer_event(self.layer_id)
        return output, block_residual

    def load_weights(self, weights: ScopedWeightLoader, names: tuple[str, ...]) -> set[str]:
        loaded: set[str] = set()
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "self_attention_res_norm",
            "mlp_res_norm",
            "self_attention_res_proj",
            "mlp_res_proj",
        ):
            key = f"{name}.weight"
            weights.load_tensor(getattr(self, name).weight, key)
            loaded.add(key)
        loaded.update(f"self_attn.{name}" for name in self.self_attn.load_weights(weights.with_prefix("self_attn.")))
        prefix = "mlp." if hasattr(self, "mlp") else "block_sparse_moe."
        child_names = tuple(name[len(prefix) :] for name in names if name.startswith(prefix))
        state_dict = _MoEStateDict(weights.with_prefix(prefix), child_names)
        module = self.mlp if hasattr(self, "mlp") else self.block_sparse_moe
        loaded.update(prefix + name for name in module.load_weights(state_dict, self.tp_rank, self.tp_size))
        module.finish_weight_loading()
        return loaded


class KimiK3TextModel(nn.Module):
    def __init__(self, config: KimiK3TextConfig, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = HiddenParallelEmbedding(
            config.vocab_size,
            config.hidden_size // config.tp_size,
            config.tp_size,
            dtype=dtype,
            device=device,
        )
        self.layers = nn.ModuleList([KimiK3DecoderLayer(config, i, dtype, device) for i in range(config.n_layers)])
        self.output_attn_res_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        self.output_attn_res_proj = nn.Linear(
            config.hidden_size,
            1,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)
        self._sp = config.use_sequence_parallel
        self.tp_size = config.tp_size
        self.tp_rank = config.tp_rank

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        num_tokens = hidden_states.shape[0]

        sp_active = _sp_active(self._sp)
        layout = TokenParallelLayout(num_tokens, self.tp_size, self.tp_rank) if sp_active else None
        if sp_active:
            # FlashComm1: shard the token dimension so the residual trunk runs
            # sequence-parallel; the embedding output is replicated, so this is
            # a local slice.
            hidden_states = layout.shard(hidden_states)
        block_residual = hidden_states.new_zeros((hidden_states.shape[0], 0, hidden_states.shape[-1]))
        for layer in self.layers:
            hidden_states, block_residual = layer(hidden_states, positions, block_residual)
        hidden_states = _apply_attention_residual(
            hidden_states,
            block_residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
        )
        if sp_active:
            hidden_states = layout.gather(hidden_states)
        output = self.norm(hidden_states)
        return output

    def load_weights(self, weights: ScopedWeightLoader, names: tuple[str, ...]) -> set[str]:
        loaded: set[str] = set()
        for name, module, dim in (
            ("embed_tokens", self.embed_tokens, 1),
            ("output_attn_res_norm", self.output_attn_res_norm, None),
            ("output_attn_res_proj", self.output_attn_res_proj, None),
            ("norm", self.norm, None),
        ):
            key = f"{name}.weight"
            weights.load_tensor(module.weight, key, dim=dim, rank=self.tp_rank, world_size=self.tp_size)
            loaded.add(key)
        for layer_id, layer in enumerate(self.layers):
            prefix = f"layers.{layer_id}."
            layer_names = tuple(name[len(prefix) :] for name in names if name.startswith(prefix))
            loaded.update(prefix + name for name in layer.load_weights(weights.with_prefix(prefix), layer_names))
        return loaded


class KimiK3ForCausalLM(PyModelBase):
    """Hybrid text model composed by the Kimi VLM registry factory."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = KimiK3TextConfig.from_dict(config)
        self.cfg.validate()
        if int(config.get("cp_size", 1)) != 1 or int(config.get("num_speculative_tokens", 0)):
            raise NotImplementedError("Kimi K3 supports CP=1 and non-speculative execution")
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "npu:0"))
        self.model = KimiK3TextModel(self.cfg, dtype, device)
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // self.cfg.tp_size,
            self.cfg.tp_size,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

    @property
    def supports_prefix_cache(self) -> bool:
        return self.cfg.enable_prefix_cache

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, inputs_embeds: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def compute_logits(self, hidden: torch.Tensor, selected_idxes: torch.Tensor | None) -> torch.Tensor:
        logits = super().compute_logits(hidden, selected_idxes)
        return logits if self.cfg.logit_scale is None else logits * self.cfg.logit_scale

    def load_weights(self, state_dicts: list[StateDictLike], tp_rank: int, tp_size: int) -> set[str]:
        if (tp_rank, tp_size) != (self.cfg.tp_rank, self.cfg.tp_size):
            raise ValueError("Kimi text loader TP coordinates must match the constructed model")
        weights = ScopedWeightLoader(state_dicts, src_prefixes=("language_model.model.", "model.", ""))
        weights = weights.bind_source_root("embed_tokens.weight")
        # MoE's existing expert loader enumerates checkpoint names. Keep only
        # names under the selected backbone root instead of merging roots.
        root = next(
            prefix
            for prefix in ("language_model.model.", "model.", "")
            if any(state.has(prefix + "embed_tokens.weight") for state in state_dicts)
        )
        names = tuple(
            dict.fromkeys(name[len(root) :] for state in state_dicts for name in state.keys() if name.startswith(root))
        )
        loaded = {"model." + name for name in self.model.load_weights(weights, names)}
        if self.cfg.tie_word_embeddings:
            weights.load_tensor(self.lm_head.weight, "embed_tokens.weight", dim=0, rank=tp_rank, world_size=tp_size)
        else:
            head = ScopedWeightLoader(
                state_dicts, src_prefixes=("language_model.lm_head.", "lm_head.", "model.lm_head.", "head.")
            )
            head.load_tensor(self.lm_head.weight, "weight", dim=0, rank=tp_rank, world_size=tp_size)
        loaded.add("lm_head.weight")
        return loaded


__all__ = ["KimiK3ForCausalLM", "KimiK3TextConfig", "KimiK3TextModel"]
