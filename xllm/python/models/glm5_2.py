# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""GLM-5.2 (model_type=glm_moe_dsa) causal LM, adapted from DeepSeek-V3.2.

Shared machinery is imported from ``deepseek_v32``: W8A8 linears, dense MLP,
MoE, YaRN RoPE, and the MLA RoPE helpers. Only the
GLM-5.2 structural deltas live here:

  * cross-layer top-k sharing -- ``indexer_types`` marks full/shared layers;
    shared layers skip the indexer and reuse the previous full layer's top-k.
  * indexer ``wq_b`` is W8A8 (not bf16 ``nn.Linear``).
  * indexer RoPE is configurable (``indexer_rope_interleave``); DSV3.2's
    indexer uses half-rotate only.
  * per-layer MLP type comes from ``mlp_layer_types`` (not a single
    ``first_k_dense_replace`` threshold).
  * the YaRN ``cos_sin_cache`` is built once at the model level and threaded
    through ``forward`` (no per-layer rotary module).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

from xllm.python import distributed, kernels
from xllm.python.attention.backend import MlaIndexContext

# The AICPU tiling of ``aclnnQuantLightningIndexer`` requires
# ``num_heads_q / num_heads_k == 64``. GLM-5.2 uses ``index_n_heads=32`` and
# ``num_heads_k=1``, so we pad Q / q_scale / weights along the head axis to
# 64 heads (padded weights are zero, keeping ``score = sum_h w[h]*q[h]*k``
# mathematically identical). Pure-Python workaround limited to this file.
from xllm.python.layers import (
    Attention,
    ColumnParallelLinear,
    HiddenParallelEmbedding,
    RMSNorm,
)
from xllm.python.layers.moe import QuantGatedMLP
from xllm.python.layers.moe_parallel import Eplv2CommPolicy
from xllm.python.layers.qlinear import QLinear, QLinearWeightLoader
from xllm.python.model_executor.cp_utils import (
    CpContext,
    cp_gather_kv,
    cp_index_write_replicated,
    cp_merge_rows,
    cp_real_query_rows,
    cp_shard_positions,
    cp_shard_rows,
)
from xllm.python.model_executor.forward_context import (
    get_forward_context,
    in_acl_graph,
    record_layer_event,
    wait_for_layer_load,
)
from xllm.python.models.aux_hidden_capture import AuxHiddenCapture
from xllm.python.models.base import PyModelBase
from xllm.python.models.deepseek_v32 import (
    DeepseekYarnRotaryEmbedding as Glm52YarnRotaryEmbedding,
)
from xllm.python.models.deepseek_v32 import (
    _apply_half_rope,
    _create_hadamard_matrix,
    _gather_interleave_cos_sin,
    _interleave_rope_with,
    _tp_rank_from_device,
    _yarn_get_mscale,
)
from xllm.python.models.glm5_2_moe import (
    Glm52MoE,
    glm52_eplv2_comm_policy,
    validate_glm52_ep_config,
    validate_glm52_ep_execution_config,
)
from xllm.python.models.glm5_next import _resolve_module
from xllm.python.models.weight_utils import W8A8WeightLoader, effective_moe_tp, mla_head_split

# aclnnQuantLightningIndexer tiling hard-requires n_heads_q / n_heads_k == 64
# (G_SIZE_LIMIT in third_party/xllm_ops/.../quant_lightning_indexer_tiling.h). The
# xllm_ops kernel fixes matmul M=256 tile with head_dim=128, kv_head=1 for DSV3.2/DSV4
# shapes; other gSize values are neither exposed nor tested. GLM-5.2 is 32/1 and pads Q
# up to 64 heads at the caller (see Glm52Indexer._pad_q_heads_to_kernel_gsize).
_QLI_KERNEL_GSIZE = 64

# lightning_indexer sparse-mask contract, shared by the full-stream and the
# CP-partitioned calls. ``sparse_mode=3`` is right-down-causal: the causal mask
# aligns to the KV tail, which is what a prefix-cache / chunked-prefill batch
# needs (``q_len < kv_len``; the left-up mode only aligns when they are equal).
# The INT64_MAX pair is the CANN "no window bound" sentinel -- it is the op
# schema's own default for ``pre_tokens``/``next_tokens`` (npu_ops_library.cpp
# registers ``int pre_tokens=9223372036854775807``), so the mask stays a pure
# causal mask without extra sliding-window clipping. Named so the two call
# sites cannot drift apart: they must differ only in WHICH query rows are
# scored, never in mask policy.
_QLI_SPARSE_MODE_RIGHT_DOWN_CAUSAL = 3
_QLI_NO_WINDOW_LIMIT = 9223372036854775807

# Indexer query-row partition under prefill CP: Glm52Indexer.select_qli scores
# only this rank's real CP query rows instead of the full merged stream;
# key-side index cache writes stay full-stream. The S0 probe verified that the
# lightning_indexer / quant_lightning_indexer kernels are row-independent and
# that the per-segment varlen suffix layout reproduces the full-stream top-k at
# set level (row-internal order can vary; downstream sparse attention sums by
# index, so its output stays bit-exact; see research/s0-probe-results.md, task
# 10-19-glm52-indexer-cp).
#
# ALWAYS ON in a shipped build, and deliberately NOT an environment variable: a
# process-wide knob that silently changes numerics is a footgun (it would also
# let one rank of a CP group differ from its peers). The remaining terms of the
# gate below are correctness conditions, not toggles. This constant exists only
# as a test seam so the CPU suite can exercise the byte-identical full-stream
# path for comparison -- nothing outside tests may assign it.
_INDEXER_CP_QUERY_PARTITION = True


@dataclass
class Glm52Config:
    """GLM-5.2 (glm_moe_dsa) architecture parameters."""

    model_type: str = "glm_moe_dsa"
    hidden_size: int = 6144
    n_layers: int = 78
    n_heads: int = 64
    head_dim: int = 0
    intermediate_size: int = 12288
    vocab_size: int = 154880
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1.0e6
    max_position_embeddings: int = 202752
    original_max_position_embeddings: int = 202752
    rope_scaling_factor: float = 1.0
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 1.0
    tie_word_embeddings: bool = False
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    qk_head_dim: int = 256
    v_head_dim: int = 256
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    first_k_dense_replace: int = 3
    moe_layer_freq: int = 1
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 2.5
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    moe_intermediate_size: int = 2048
    tp_size: int = 1
    tp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0
    cp_size: int = 1
    cp_rank: int = 0
    layerwise_split_size: int = 1
    layerwise_split_rank: int = 0
    moe_tp_size: int = 1
    moe_tp_rank: int = 0
    world_size: int = 1
    indexer_types: list | None = None
    mlp_layer_types: list | None = None
    index_skip_topk_offset: int = 2
    index_topk_freq: int = 1
    index_topk_pattern: list | None = None
    indexer_rope_interleave: bool = True
    num_nextn_predict_layers: int = 0
    index_share_for_mtp_iteration: bool = False
    layers_to_capture: tuple[int, ...] = ()
    enable_mega_moe: bool = False
    expert_parallel_degree: int = 0
    enable_eplb: bool = False
    mega_moe_context: torch.Tensor | None = None
    mega_moe_ccl_buffer_size: int = 0
    mega_moe_num_max_tokens_per_rank: int = 0
    eplv2_mc2_max_tokens_per_rank: int = 512

    @classmethod
    def from_dict(cls, d: dict) -> Glm52Config:
        def pick(*keys: str, default: Any = None) -> Any:
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return default

        rs_raw = d.get("rope_scaling")
        rs = rs_raw if isinstance(rs_raw, dict) else {}
        if not rs:
            rp = d.get("rope_parameters")
            if isinstance(rp, dict):
                rs = rp

        def rpick(*keys: str, default: Any = None) -> Any:
            for k in keys:
                if isinstance(rs, dict) and k in rs and rs[k] is not None:
                    return rs[k]
                fk = f"rope_scaling_{k}"
                if fk in d and d[fk] is not None:
                    return d[fk]
                if k in d and d[k] is not None:
                    return d[k]
            return default

        def rpick_nz(*keys: str, default: Any) -> Any:
            v = rpick(*keys, default=None)
            if v is None or v == 0 or v == -1 or v == "":
                return default
            return v

        hidden = int(pick("hidden_size", default=6144))
        n_heads = int(pick("n_heads", "num_attention_heads", default=64))
        max_pe = int(pick("max_position_embeddings", default=202752))
        tp_size = int(pick("tp_size", default=1))
        dp_size = int(pick("dp_size", default=1))
        cp_size = int(pick("cp_size", default=1))
        world_size = int(pick("world_size", default=tp_size * dp_size * cp_size))
        expert_parallel_degree = int(pick("expert_parallel_degree", default=0))
        rope_scaling_factor = float(rpick_nz("factor", "rope_scaling_factor", default=1.0))
        original_max = int(rpick_nz("original_max_position_embeddings", default=max_pe))

        cfg = cls(
            model_type=str(pick("model_type", default="glm_moe_dsa")),
            hidden_size=hidden,
            n_layers=int(pick("n_layers", "num_hidden_layers", default=78)),
            n_heads=n_heads,
            head_dim=int(pick("head_dim", default=hidden // n_heads if n_heads else 0)),
            intermediate_size=int(pick("intermediate_size", default=12288)),
            vocab_size=int(pick("vocab_size", default=154880)),
            rms_norm_eps=float(pick("rms_norm_eps", default=1e-5)),
            rope_theta=float(pick("rope_theta", default=1.0e6)),
            max_position_embeddings=max_pe,
            original_max_position_embeddings=original_max,
            rope_scaling_factor=rope_scaling_factor,
            rope_beta_fast=int(rpick_nz("beta_fast", default=32)),
            rope_beta_slow=int(rpick_nz("beta_slow", default=1)),
            rope_mscale=float(rpick_nz("mscale", default=1.0)),
            rope_mscale_all_dim=float(rpick_nz("mscale_all_dim", default=1.0)),
            tie_word_embeddings=bool(pick("tie_word_embeddings", default=False)),
            q_lora_rank=int(pick("q_lora_rank", default=2048)),
            kv_lora_rank=int(pick("kv_lora_rank", default=512)),
            index_n_heads=int(pick("index_n_heads", default=32)),
            index_head_dim=int(pick("index_head_dim", default=128)),
            index_topk=int(pick("index_topk", default=2048)),
            qk_nope_head_dim=int(pick("qk_nope_head_dim", default=192)),
            qk_rope_head_dim=int(pick("qk_rope_head_dim", default=64)),
            qk_head_dim=int(pick("qk_head_dim", default=256)),
            v_head_dim=int(pick("v_head_dim", default=256)),
            first_k_dense_replace=int(pick("first_k_dense_replace", default=3)),
            moe_layer_freq=int(pick("moe_layer_freq", default=1)),
            n_routed_experts=int(pick("n_routed_experts", "num_local_experts", "num_experts", default=256)),
            n_shared_experts=int(pick("n_shared_experts", default=1)),
            num_experts_per_tok=int(pick("num_experts_per_tok", default=8)),
            n_group=int(pick("n_group", default=1)),
            topk_group=int(pick("topk_group", default=1)),
            routed_scaling_factor=float(pick("routed_scaling_factor", default=2.5)),
            topk_method=str(pick("topk_method", default="noaux_tc")),
            norm_topk_prob=bool(pick("norm_topk_prob", default=True)),
            moe_intermediate_size=int(pick("moe_intermediate_size", default=2048)),
            tp_size=tp_size,
            tp_rank=int(pick("tp_rank", default=0)),
            ep_size=int(pick("ep_size", default=1)),
            ep_rank=int(pick("ep_rank", default=0)),
            dp_size=dp_size,
            dp_rank=int(pick("dp_rank", default=0)),
            cp_size=cp_size,
            cp_rank=int(pick("cp_rank", default=0)),
            layerwise_split_size=int(pick("layerwise_split_size", default=1)),
            layerwise_split_rank=int(pick("layerwise_split_rank", default=0)),
            moe_tp_size=int(pick("moe_tp_size", default=1)),
            moe_tp_rank=int(pick("moe_tp_rank", default=0)),
            world_size=world_size,
            indexer_types=pick("indexer_types", default=None) or None,
            mlp_layer_types=pick("mlp_layer_types", default=None) or None,
            index_skip_topk_offset=int(pick("index_skip_topk_offset", default=2)),
            index_topk_freq=int(pick("index_topk_freq", default=1)),
            index_topk_pattern=pick("index_topk_pattern", default=None),
            indexer_rope_interleave=bool(pick("indexer_rope_interleave", default=True)),
            num_nextn_predict_layers=int(pick("num_nextn_predict_layers", default=0)),
            index_share_for_mtp_iteration=bool(pick("index_share_for_mtp_iteration", default=False)),
            layers_to_capture=tuple(int(layer_id) for layer_id in pick("layers_to_capture", default=[])),
            enable_mega_moe=bool(pick("enable_mega_moe", default=False)),
            expert_parallel_degree=expert_parallel_degree,
            enable_eplb=bool(pick("enable_eplb", default=False)),
            mega_moe_context=pick("mega_moe_context", default=None),
            mega_moe_ccl_buffer_size=int(pick("mega_moe_ccl_buffer_size", default=0)),
            mega_moe_num_max_tokens_per_rank=int(pick("mega_moe_num_max_tokens_per_rank", default=0)),
            eplv2_mc2_max_tokens_per_rank=int(
                pick(
                    "eplv2_mc2_max_tokens_per_rank",
                    default=os.getenv("XLLM_EPLV2_MC2_MAX_TOKENS", "512") if expert_parallel_degree == 2 else "512",
                )
            ),
        )
        cfg._resolve_indexer_types()
        cfg._resolve_mlp_layer_types()
        return cfg

    def validate(self) -> None:
        """Validate the orthogonal attention/DP and MoE EP topology."""
        if min(self.tp_size, self.ep_size, self.dp_size, self.cp_size, self.moe_tp_size) <= 0:
            raise ValueError("parallel sizes must be positive")
        if self.tp_size * self.dp_size * self.cp_size != self.world_size:
            raise ValueError("world_size must equal tp_size * dp_size * cp_size")
        if self.ep_size not in (1, self.world_size):
            raise ValueError(f"ep_size must be 1 or world_size ({self.world_size})")
        if self.ep_size > 1:
            if self.n_routed_experts % self.ep_size:
                raise ValueError("n_routed_experts must be divisible by ep_size")
            if self.moe_tp_size * self.ep_size != self.world_size:
                raise ValueError("world_size must equal moe_tp_size * ep_size")
        if self.moe_intermediate_size % effective_moe_tp(self):
            raise ValueError("moe_intermediate_size must be divisible by moe_tp_size")
        if not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("tp_rank must be in [0, tp_size)")
        if not 0 <= self.dp_rank < self.dp_size:
            raise ValueError("dp_rank must be in [0, dp_size)")
        if not 0 <= self.cp_rank < self.cp_size:
            raise ValueError("cp_rank must be in [0, cp_size)")
        if not 0 <= self.ep_rank < self.ep_size:
            raise ValueError("ep_rank must be in [0, ep_size)")
        if not 0 <= self.moe_tp_rank < self.moe_tp_size:
            raise ValueError("moe_tp_rank must be in [0, moe_tp_size)")
        if self.layerwise_split_size <= 0 or self.tp_size % self.layerwise_split_size:
            raise ValueError("layerwise_split_size must be a positive divisor of tp_size")
        if not 0 <= self.layerwise_split_rank < self.layerwise_split_size:
            raise ValueError("layerwise_split_rank must be in [0, layerwise_split_size)")
        if self.layerwise_split_size > 1 and self.cp_size > 1:
            raise ValueError("GLM5.2 Python does not support CP and layerwise split together")
        validate_glm52_ep_config(self)

    def _resolve_indexer_types(self) -> None:
        """Derive per-layer indexer mode (full/shared)."""
        if self.indexer_types is not None:
            return
        pattern = self.index_topk_pattern
        if pattern:
            if isinstance(pattern, str):
                self.indexer_types = [{"F": "full", "S": "shared"}[c] for c in pattern]
            else:
                self.indexer_types = list(pattern)
            return
        freq = max(self.index_topk_freq, 1)
        offset = self.index_skip_topk_offset
        self.indexer_types = [
            "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared" for i in range(self.n_layers)
        ]

    def _resolve_mlp_layer_types(self) -> None:
        """Derive per-layer MLP mode (dense/sparse)."""
        if self.mlp_layer_types is not None:
            return
        n_dense = min(self.first_k_dense_replace, self.n_layers)
        self.mlp_layer_types = ["dense"] * n_dense + ["sparse"] * (self.n_layers - n_dense)

    def head_split(self) -> tuple[int, int]:
        """Per-rank (num_heads_local, num_kv_heads_local=1) — MLA has one latent KV head per rank."""
        return mla_head_split(self.n_heads, self.tp_size)


class Glm52MLP(QuantGatedMLP):
    """GLM-5.2 dense MLP: fp32 TP reduction, split-path fused requant."""

    def __init__(
        self,
        cfg,
        intermediate_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(
            cfg.hidden_size,
            intermediate_size,
            device,
            tp_size=cfg.tp_size,
            # GLM-5.2 reduces the TP partial sum in fp32 (numerical-stability
            # fix validated against the pre-refactor implementation).
            tp_reduce="fp32",
            # Historical split path: requantize returns raw gate|up int8 and
            # activation happens via the explicit silu_and_mul fast path.
            fused_swiglu=False,
        )


class Glm52MLAAttention(Attention):
    """Absorbed-MLA attention with per-layer full/shared DSA indexer."""

    def __init__(
        self,
        cfg: Glm52Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp = cfg.tp_size
        num_heads, _ = cfg.head_split()
        kv_lora = cfg.kv_lora_rank
        qk_nope = cfg.qk_nope_head_dim
        qk_rope = cfg.qk_rope_head_dim
        v_head = cfg.v_head_dim
        scale = (qk_nope + qk_rope) ** -0.5
        attn_mscale = _yarn_get_mscale(cfg.rope_scaling_factor, cfg.rope_mscale_all_dim)
        scale = scale * attn_mscale * attn_mscale
        super().__init__(
            num_heads=num_heads,
            num_kv_heads=1,
            head_dim=kv_lora,
            scale=scale,
            sliding_window=0,
            layer_id=layer_id,
        )
        self.cfg = cfg
        self.qk_nope_head_dim = qk_nope
        self.qk_rope_head_dim = qk_rope
        self.v_head_dim = v_head
        self.kv_lora_rank = kv_lora
        self.num_heads_local = num_heads

        # QLinear defers the static/dynamic W8A8 choice to load time: the
        # loader probes whether the checkpoint carries deq_scale (internal
        # static exports, GLM-5.2) or weight_scale (HF per-channel dynamic
        # exports, GLM-5.3) suffixes.
        self.q_a_proj = QLinear(cfg.hidden_size, cfg.q_lora_rank, device=device, dtype=dtype, kind="static")
        self.kv_a_proj_with_mqa = QLinear(cfg.hidden_size, kv_lora + qk_rope, device=device, dtype=dtype, kind="static")
        self.q_a_layernorm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.kv_a_layernorm = RMSNorm(kv_lora, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.q_b_proj = QLinear(
            cfg.q_lora_rank, num_heads * (qk_nope + qk_rope), device=device, dtype=dtype, kind="static"
        )
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora,
            num_heads * (qk_nope + v_head),
            tp,
            dtype=dtype,
            device=device,
        )
        self.o_proj = QLinear(
            num_heads * v_head, cfg.hidden_size, device=device, dtype=dtype, kind="static", row_parallel=True
        )
        self.register_buffer(
            "W_UK",
            torch.empty(num_heads, qk_nope, kv_lora, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "W_UV",
            torch.empty(num_heads, kv_lora, v_head, dtype=dtype, device=device),
            persistent=False,
        )
        self.is_shared = (
            cfg.indexer_types is not None
            and layer_id < len(cfg.indexer_types)
            and cfg.indexer_types[layer_id] == "shared"
        )
        self.indexer = None
        if not self.is_shared:
            self.indexer = Glm52Indexer(cfg, dtype, device)

    def process_weights_after_loading(self) -> None:
        self.q_a_proj.process_weights_after_loading()
        self.kv_a_proj_with_mqa.process_weights_after_loading()
        self.q_b_proj.process_weights_after_loading()
        self.o_proj.process_weights_after_loading()
        w = self.kv_b_proj.weight.data
        w = w.view(
            self.num_heads_local,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        w_uk, w_uv = w.split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        self.W_UK.copy_(w_uk.contiguous())
        self.W_UV.copy_(w_uv.transpose(1, 2).contiguous())
        if self.indexer is not None:
            self.indexer.process_weights_after_loading()

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        prev_topk_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wait_for_layer_load(self.layer_id)
        num_tokens = hidden.shape[0]
        q_a = self.q_a_proj(hidden)
        q_c = self.q_a_layernorm(q_a)
        forward_ctx = get_forward_context()
        backend = forward_ctx.attention_backend
        layerwise = self.cfg.layerwise_split_size > 1 and not (
            forward_ctx.metadata.is_prefill or forward_ctx.metadata.is_chunked_prefill
        )
        layer_owner = self.layer_id % self.cfg.layerwise_split_size
        owns_layer_cache = self.cfg.layerwise_split_rank == layer_owner
        if self.indexer is not None:
            ctx = backend.mla_index_context(self)
            if layerwise:
                if owns_layer_cache:
                    topk = self.indexer.select_qli(hidden, q_c, positions, ctx, cos_sin_cache)
                else:
                    topk = torch.empty(
                        (num_tokens, ctx.index_cache.size(2), self.cfg.index_topk),
                        dtype=torch.int32,
                        device=hidden.device,
                    )
                distributed.broadcast_(topk, layer_owner, "layerwise")
            else:
                # The replicated indexer gathers Q/K/weights in global order.
                # Keep the complete padded CP-local layout here: real-query
                # packing has unequal row counts on ragged zigzag shards and
                # violates cp_gather_kv's all-gather/restore-index contract.
                topk = self.indexer.select_qli(hidden, q_c, positions, ctx, cos_sin_cache)
        else:
            if prev_topk_indices is None:
                raise ValueError(
                    "Shared DSA layers require top-k indices from a previous "
                    "full indexer layer (prev_topk_indices is None)."
                )
            topk = prev_topk_indices
        q = self.q_b_proj(q_c)
        q = q.view(
            num_tokens,
            self.num_heads_local,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
        )
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_latent = kernels.batch_matmul_transpose(q_nope, self.W_UK)
        cos, sin = _gather_interleave_cos_sin(cos_sin_cache, positions)
        q_pe = _interleave_rope_with(q_rope, cos, sin)
        kv = self.kv_a_proj_with_mqa(hidden)
        k_latent_raw, k_rope_raw = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_latent = self.kv_a_layernorm(k_latent_raw)
        k_pe = _interleave_rope_with(k_rope_raw.unsqueeze(1), cos, sin)
        k_latent_3d = k_latent.view(num_tokens, 1, self.kv_lora_rank)
        k_pe_3d = k_pe.view(num_tokens, 1, self.qk_rope_head_dim)

        if layerwise:
            local_query = torch.cat((q_latent, q_pe), dim=-1)
            gathered_query = distributed.all_gather(
                local_query,
                dim=1,
                world_size=self.cfg.layerwise_split_size,
                group_name="layerwise",
            )
            latent_width = q_latent.shape[-1]
            gathered_q_latent = gathered_query[..., :latent_width]
            gathered_q_pe = gathered_query[..., latent_width:]
            if owns_layer_cache:
                gathered_attn_out = backend.execute_mla(
                    gathered_q_latent,
                    gathered_q_pe,
                    k_latent_3d,
                    k_pe_3d,
                    self,
                    topk=topk,
                )
            else:
                gathered_attn_out = torch.empty_like(gathered_q_latent)
            distributed.broadcast_(gathered_attn_out, layer_owner, "layerwise")
            head_offset = self.cfg.layerwise_split_rank * self.num_heads_local
            attn_out = gathered_attn_out.narrow(1, head_offset, self.num_heads_local)
        else:
            attn_out = backend.execute_mla(q_latent, q_pe, k_latent_3d, k_pe_3d, self, topk=topk)
        v_full = kernels.batch_matmul_transpose(attn_out, self.W_UV)
        v_full = v_full.reshape(num_tokens, self.num_heads_local * self.v_head_dim)
        o = self.o_proj(v_full)
        if self.cfg.tp_size > 1:
            output_dtype = o.dtype
            o = o.to(torch.float32)
            distributed.all_reduce_(o)
            o = o.to(output_dtype)
        return o, topk


class Glm52Indexer(nn.Module):
    """GLM-5.2 DSA lightning indexer (wq_b W8A8, configurable RoPE)."""

    def __init__(self, cfg: Glm52Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.n_head = cfg.index_n_heads
        self.head_dim = cfg.index_head_dim
        self.rope_dim = cfg.qk_rope_head_dim
        self.topk = cfg.index_topk
        self.indexer_rope_interleave = cfg.indexer_rope_interleave
        self.wq_b = QLinear(cfg.q_lora_rank, self.n_head * self.head_dim, device=device, dtype=dtype, kind="static")
        self.wk = nn.Linear(cfg.hidden_size, self.head_dim, bias=False, dtype=dtype, device=device)
        self.weights_proj = nn.Linear(cfg.hidden_size, self.n_head, bias=False, dtype=dtype, device=device)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6, dtype=dtype, device=device)
        self.register_buffer(
            "hadamard",
            _create_hadamard_matrix(self.head_dim, dtype, device),
            persistent=False,
        )

    def process_weights_after_loading(self) -> None:
        self.wq_b.process_weights_after_loading()

    def _pad_q_heads_to_kernel_gsize(
        self,
        q: torch.Tensor,
        q_scale: torch.Tensor,
        weights: torch.Tensor,
        required_q_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # aclnnQuantLightningIndexer tiling hard-requires n_heads_q/n_heads_k == 64
        # (G_SIZE_LIMIT in quant_lightning_indexer_tiling.h; xllm_ops kernel hard-codes
        # matmul M=256 tile, head_dim=128 and kv_head=1 for DSV3.2/DSV4 shapes). GLM-5.2
        # is index_n_heads=32 / kv_head=1, so pad Q from 32 to 64 heads to satisfy the
        # kernel. The score sum_h(w_h * q_h . k) is mathematically unchanged: padded Q
        # rows are zero, so q_h . k = 0 for h >= n_head; padded weights are zero so those
        # zero terms cannot contribute even if the kernel processed them differently.
        pad_heads = required_q_heads - self.n_head
        if pad_heads == 0:
            return q, q_scale, weights
        if pad_heads < 0:
            raise RuntimeError(f"Glm52Indexer expected index_n_heads<={required_q_heads}, got {self.n_head}")
        q = torch.cat(
            [q, torch.zeros((q.size(0), pad_heads, q.size(2)), dtype=q.dtype, device=q.device)],
            dim=1,
        )
        # Zero Q/weight heads contribute nothing; a neutral nonzero scale
        # is safe for either multiplying or reciprocal dequantization.
        q_scale = torch.cat(
            [q_scale, torch.ones((q_scale.size(0), pad_heads), dtype=q_scale.dtype, device=q_scale.device)],
            dim=1,
        )
        weights = torch.cat(
            [weights, torch.zeros((weights.size(0), pad_heads), dtype=weights.dtype, device=weights.device)],
            dim=1,
        )
        return q, q_scale, weights

    def _cp_partitioned_topk(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        index_cache: torch.Tensor,
        block_table: torch.Tensor,
        cp_context: CpContext,
    ) -> torch.Tensor:
        """Score only this rank's real CP query rows (alpha plan, non-quant).

        ``q``/``weights`` are the CP-merged global-stream tensors. The rows
        this rank really owns (``cp_real_query_rows``: ``shard_index``
        gathered at ``query_index``) are packed per (sequence, half) segment,
        which is exactly the order ``q_cu_seqlens`` counts, so one varlen
        ``lightning_indexer`` call over the per-segment lengths reproduces
        the full-stream top-k of those rows at set level (the S0 probe
        contract: row-internal order varies with batch composition while the
        selected index set does not; downstream sparse attention sums by
        index, so its output stays bit-exact -- research/s0-probe-results.md).
        Rows this rank does not own stay -1 in the returned full-layout
        tensor and never reach a consumer: ``cp_shard_rows`` selects only
        this rank's local rows (dropping the peer-owned -1 rows) and
        zeroes the padding ones, so no -1 escapes and the sharded output
        differs from the full-stream path only in the row-internal top-k
        order that S0 contract #6 already covers (byte-identical for an
        empty rank, whose every local row is zeroed padding).
        """
        rows = cp_real_query_rows(cp_context)
        # index_cache here is the materialized paged MLA index cache, which the
        # backend pins to 4-D ``[blocks, block_size, num_heads_k, head_dim]``
        # with ``num_heads_k == 1``: `_is_compact_kpool_cache_compatible`
        # asserts ``shape[2] == 1`` and `materialize_index_cache`'s consumers
        # read the width from ``shape[3]``. The layerwise branch above and the
        # backend's metadata getter use the same ``size(2)`` accessor. Assert
        # the rank so a layout change fails loudly here instead of silently
        # sizing the output as ``[rows, head_dim, topk]``.
        assert index_cache.dim() == 4, (
            f"paged MLA index cache must be 4-D [blocks, block_size, num_heads_k, head_dim], "
            f"got {tuple(index_cache.shape)}"
        )
        topk = torch.full(
            (q.size(0), index_cache.size(2), self.topk),
            -1,
            dtype=torch.int32,
            device=q.device,
        )
        if rows.numel() == 0:
            # An empty rank owns no real query row, so there is nothing to
            # score: every local row is padding and cp_shard_rows zeroes it,
            # matching the full-stream path's output exactly.
            return topk
        # key_seq_lengths takes segment_kv_seq_lens -- the ABSOLUTE
        # per-segment KV length, paged prefix included -- not kv_cu_seqlens
        # (the causal prefix inside the gathered chunk stream): the
        # materialized index-cache block table covers the whole paged
        # sequence, the same absolute lengths ctx.actual_seq_kv pairs with
        # that table on the full-stream path. With sparse_mode=3
        # (right-aligned causal) a segment's query row i then attends
        # exactly [0, segment_start + i] of its own sequence.
        # The ``.contiguous()`` calls below are a layout GUARANTEE for the
        # kernel, not an extra copy: ``index_select`` returns a freshly
        # materialized tensor and ``.contiguous()`` is a no-op on it
        # (``x.contiguous() is x``), so this path allocates only the gathered
        # rows themselves -- no second buffer. They stay because the kernel
        # entry contract requires contiguous inputs, and a source tensor with a
        # non-trivial stride (or a future gather that returns a view) must not
        # silently produce a strided argument.
        topk_rows = kernels.lightning_indexer(
            q.index_select(0, rows).contiguous(),
            index_cache,
            weights.index_select(0, rows).contiguous(),
            cp_context.q_cu_seqlens_tensor,
            cp_context.segment_kv_seq_lens_tensor,
            block_table.index_select(0, cp_context.segment_seq_indices).contiguous(),
            "TND",
            "PA_BSND",
            self.topk,
            _QLI_SPARSE_MODE_RIGHT_DOWN_CAUSAL,
            _QLI_NO_WINDOW_LIMIT,
            _QLI_NO_WINDOW_LIMIT,
            False,
        )
        topk.index_copy_(0, rows, topk_rows)
        return topk

    def _cp_partitioned_quant_topk(
        self,
        q: torch.Tensor,
        q_scale: torch.Tensor,
        weights: torch.Tensor,
        weight_scale: float,
        index_cache: torch.Tensor,
        index_cache_scale: torch.Tensor,
        block_table: torch.Tensor,
        cp_context: CpContext,
        cmp_ratio: int,
        get_cp_metadata: Callable[..., torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Score only this rank's real CP query rows (alpha plan, quant).

        Quant sibling of :meth:`_cp_partitioned_topk` over the head-padded
        int8 ``q``/fp16 ``q_scale``/``weights``: the same global-row
        slicing, per-segment seq tensors, per-segment block-table rows and
        -1-filled full-layout scatter. Slicing by query rows preserves the
        padded 64-head per-token-per-head ``q_scale`` layout (S0 contract
        #10: the row axis is orthogonal to the head axis), while the key
        side (int8 ``index_cache``, paged 3-D ``index_cache_scale``,
        ``block_table``) stays full-stream exactly as on the non-quant
        path. One quant-specific duty: the tiling metadata must describe
        THIS sliced call's seq tensors (S0 contract #7), so it cannot reuse
        the backend's full-stream metadata -- it is built from the
        per-segment seq tensors and their maxima instead. ``get_cp_metadata``
        is the backend's per-STEP cache for exactly that metadata (the
        segment lengths and maxima are identical for every indexer layer in
        one step, so recomputing per layer would burn host-side tiling and a
        1024-int32 allocation per layer); the direct kernel call remains the
        fallback for a context that does not expose it.
        """
        rows = cp_real_query_rows(cp_context)
        # Same 4-D paged index cache contract as the non-quant partition; see
        # the assertion comment there (_is_compact_kpool_cache_compatible pins
        # ``shape[2] == 1``).
        assert index_cache.dim() == 4, (
            f"paged MLA index cache must be 4-D [blocks, block_size, num_heads_k, head_dim], "
            f"got {tuple(index_cache.shape)}"
        )
        topk = torch.full(
            (q.size(0), index_cache.size(2), self.topk),
            -1,
            dtype=torch.int32,
            device=q.device,
        )
        if rows.numel() == 0:
            # Same empty-rank contract as the non-quant partition: no real
            # row to score, so neither the metadata nor the scoring kernel
            # runs and cp_shard_rows zeroes the padding output.
            return topk
        q_cu_seqlens = cp_context.q_cu_seqlens
        # build_cp_context_npu appends one cumulative entry per NON-EMPTY
        # segment and one query_index entry per real row, so a non-empty
        # ``rows`` above implies at least one segment -- and therefore a
        # non-empty q_cu_seqlens. Pin that cross-file invariant instead of
        # letting a future builder change surface as a bare IndexError.
        assert q_cu_seqlens, "non-empty owned query rows require at least one CP segment"
        # q_cu_seqlens holds cumulative ENDS (no leading zero), so the first
        # entry is already segment 0's length.
        segment_q_lens = [q_cu_seqlens[0]] + [end - begin for begin, end in zip(q_cu_seqlens, q_cu_seqlens[1:])]
        max_q = max(segment_q_lens)
        max_kv = max(cp_context.segment_kv_seq_lens)
        if get_cp_metadata is not None:
            qli_metadata = get_cp_metadata(
                q.size(1),
                index_cache.size(2),
                self.head_dim,
                max_q,
                max_kv,
                self.topk,
                cmp_ratio,
            )
        else:
            qli_metadata = kernels.quant_lightning_indexer_metadata(
                q.size(1),
                index_cache.size(2),
                self.head_dim,
                cp_context.q_cu_seqlens_tensor,
                cp_context.segment_kv_seq_lens_tensor,
                max_q,
                max_kv,
                self.topk,
                cmp_ratio,
            )
        # Same layout guarantee as the non-quant call site: ``index_select``
        # already materializes a contiguous result, so ``.contiguous()`` adds no
        # copy; the fused ``* weight_scale -> .to(float16)`` chain is likewise
        # contiguous. Kept for the kernel's contiguous-input contract.
        topk_rows = kernels.quant_lightning_indexer(
            q.index_select(0, rows).contiguous(),
            index_cache,
            (weights.index_select(0, rows) * weight_scale).to(torch.float16).contiguous(),
            q_scale.index_select(0, rows).contiguous(),
            index_cache_scale,
            qli_metadata,
            cp_context.q_cu_seqlens_tensor,
            cp_context.segment_kv_seq_lens_tensor,
            block_table.index_select(0, cp_context.segment_seq_indices).contiguous(),
            self.topk,
            cmp_ratio,
        )
        topk.index_copy_(0, rows, topk_rows)
        return topk

    def select_qli(
        self,
        hidden: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        ctx: MlaIndexContext,
        cos_sin_cache: torch.Tensor,
        cache_hidden: torch.Tensor | None = None,
        cache_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actual_seq_q = ctx.actual_seq_q
        actual_seq_kv = ctx.actual_seq_kv
        q = self.wq_b(qr).view(-1, self.n_head, self.head_dim)
        q_pe, q_nope = torch.split(q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        cache_hidden = hidden if cache_hidden is None else cache_hidden
        cache_positions = positions if cache_positions is None else cache_positions
        k = self.wk(cache_hidden)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        if self.indexer_rope_interleave:
            cos, sin = _gather_interleave_cos_sin(cos_sin_cache, positions)
            q_pe = _interleave_rope_with(q_pe, cos, sin)
            k_cos, k_sin = _gather_interleave_cos_sin(cos_sin_cache, cache_positions)
            k_pe = _interleave_rope_with(k_pe.unsqueeze(1), k_cos, k_sin).squeeze(1)
        else:
            q_pe = _apply_half_rope(cos_sin_cache, q_pe, positions)
            k_pe = _apply_half_rope(cos_sin_cache, k_pe.unsqueeze(1), cache_positions).squeeze(1)
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe, k_nope], dim=-1)
        weights = self.weights_proj(hidden)
        # Alpha-plan gate for the CP query-row partition (design.md): when
        # active, both scoring branches below (quant and non-quant) score
        # only this rank's real CP query rows instead of the full merged
        # stream. Beyond the env flag and CP being active, the gate
        # requires a real CpContext -- the per-segment metadata the varlen
        # call consumes is part of that dataclass, so duck-typed stand-ins
        # without it stay on the full stream -- plus acl_graph capture
        # being off (capture would bake the host-side segment lists) and,
        # under kv_split, the DEFAULT replicated index-write mode (a full
        # per-rank replica; the sharded-mode cross-rank reconstruction is
        # unverified for the per-segment call). design.md's num_seqs == 1
        # and n_real > 0 are deliberately NOT gated: the per-segment
        # metadata packs two segments per sequence in batch order, so the
        # varlen call is batch-composition-agnostic (covered by the
        # multi-sequence test), and an empty rank skips the scoring call
        # with byte-identical output (cp_shard_rows zeroes its padding
        # rows).
        # ``isinstance`` must be tested BEFORE any attribute access on the
        # plan: a duck-typed stand-in (tests, an older host shim) has no real
        # ``cp_size``, and probing it first raised TypeError instead of falling
        # back to the full stream -- the opposite of what this gate is for. The
        # CPU suite caught it (test_sfa_dcp.py passes a MagicMock plan).
        alpha_active = (
            _INDEXER_CP_QUERY_PARTITION
            and isinstance(ctx.cp_context, CpContext)
            and ctx.cp_context.cp_size > 1
            and not in_acl_graph()
            and (not ctx.has_kv_shard or cp_index_write_replicated())
        )
        if ctx.cp_context is not None:
            q = cp_gather_kv(q, ctx.cp_context).contiguous()
            k = cp_gather_kv(k, ctx.cp_context).contiguous()
            weights = cp_gather_kv(weights, ctx.cp_context).contiguous()

        index_cache = ctx.index_cache
        index_cache_scale = ctx.index_cache_scale
        use_quant_indexer = index_cache.dtype == torch.int8 and index_cache_scale is not None
        if use_quant_indexer:
            rotation_scale = self.head_dim**-0.5
            q = torch.matmul(q, self.hadamard) * rotation_scale
            k = torch.matmul(k, self.hadamard) * rotation_scale
            q, q_scale = kernels.dynamic_quant(q)
            k, k_scale = kernels.dynamic_quant(k)
            assert q_scale is not None
            assert k_scale is not None
            q_scale = q_scale.to(torch.float16)
            k_scale = k_scale.unsqueeze(-1).to(torch.float16)
            ctx.update_index_cache(k, k_scale)
            index_cache, index_cache_scale, block_table = ctx.materialize_index_cache()
            assert index_cache_scale is not None
            weight_scale = self.head_dim**-0.5 * self.n_head**-0.5
            # xLLM stores one index key per source token.
            cmp_ratio = 1

            required_q_heads = index_cache.size(2) * _QLI_KERNEL_GSIZE
            q, q_scale, weights_padded = self._pad_q_heads_to_kernel_gsize(q, q_scale, weights, required_q_heads)

            if alpha_active:
                assert ctx.cp_context is not None
                topk = self._cp_partitioned_quant_topk(
                    q,
                    q_scale,
                    weights_padded,
                    weight_scale,
                    index_cache,
                    index_cache_scale,
                    block_table,
                    ctx.cp_context,
                    cmp_ratio,
                    # Backend's per-step cache for the CP per-segment metadata.
                    # Without it every indexer layer rebuilds the same host-side
                    # tiling (and its 1024-int32 allocation) inside one step,
                    # eating the query-partition win.
                    get_cp_metadata=ctx.get_cp_quant_indexer_metadata,
                )
            else:
                qli_metadata = ctx.get_quant_indexer_metadata(required_q_heads, self.head_dim, self.topk, cmp_ratio)
                topk = kernels.quant_lightning_indexer(
                    q,
                    index_cache,
                    (weights_padded * weight_scale).to(torch.float16),
                    q_scale,
                    index_cache_scale,
                    qli_metadata,
                    actual_seq_q,
                    actual_seq_kv,
                    block_table,
                    self.topk,
                    cmp_ratio,
                )
        else:
            ctx.update_index_cache(k, None)
            index_cache, _, block_table = ctx.materialize_index_cache()
            if alpha_active:
                assert ctx.cp_context is not None
                topk = self._cp_partitioned_topk(q, weights, index_cache, block_table, ctx.cp_context)
            else:
                topk = kernels.lightning_indexer(
                    q,
                    index_cache,
                    weights,
                    actual_seq_q,
                    actual_seq_kv,
                    block_table,
                    "TND",
                    "PA_BSND",
                    self.topk,
                    _QLI_SPARSE_MODE_RIGHT_DOWN_CAUSAL,
                    _QLI_NO_WINDOW_LIMIT,
                    _QLI_NO_WINDOW_LIMIT,
                    False,
                )
        if ctx.cp_context is not None:
            topk = cp_shard_rows(topk, ctx.cp_context)
        return topk


class Glm52DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: Glm52Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        comm_policy: Eplv2CommPolicy | None = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.self_attn = Glm52MLAAttention(cfg, layer_id, dtype, device)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        mlp_type = (
            cfg.mlp_layer_types[layer_id]
            if cfg.mlp_layer_types is not None and layer_id < len(cfg.mlp_layer_types)
            else ("dense" if layer_id < cfg.first_k_dense_replace else "sparse")
        )
        if mlp_type == "dense":
            self.mlp = Glm52MLP(cfg, cfg.intermediate_size, dtype, device)
        else:
            self.mlp = Glm52MoE(cfg, layer_id, dtype, device, comm_policy=comm_policy)

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        prev_topk_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)
        hidden, topk_indices = self.self_attn(hidden, positions, cos_sin_cache, prev_topk_indices)
        hidden, residual = self.post_attention_layernorm(hidden, residual)
        hidden = self.mlp(hidden)
        return hidden, residual, topk_indices


class Glm52Model(nn.Module):
    # DFlash consumes captured target states in the original packed row order.
    restores_aux_hidden_under_cp = True

    def __init__(self, cfg: Glm52Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        tp = cfg.tp_size
        assert cfg.hidden_size % tp == 0
        self.cfg = cfg
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size,
            cfg.hidden_size // tp,
            tp,
            dtype=dtype,
            device=device,
        )
        # MC2 admission window consumed by the decode graph runner's capture
        # gating (executor reads _comm_policy from the execution model); every
        # MoE layer shares the same frozen policy object.
        self._comm_policy = glm52_eplv2_comm_policy(cfg)
        self.layers = nn.ModuleList(
            [Glm52DecoderLayer(cfg, i, dtype, device, comm_policy=self._comm_policy) for i in range(cfg.n_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.rotary = Glm52YarnRotaryEmbedding(
            cfg.qk_rope_head_dim,
            cfg.original_max_position_embeddings,
            cfg.rope_scaling_factor,
            cfg.rope_theta,
            cfg.rope_beta_fast,
            cfg.rope_beta_slow,
            cfg.rope_mscale,
            cfg.rope_mscale_all_dim,
            dtype=dtype,
            device=device,
        )
        self.aux_hidden_capture = AuxHiddenCapture(cfg.layers_to_capture)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self.embed_tokens(input_ids)
        positions = positions.to(torch.int64).contiguous()
        cp_context = get_forward_context().cp_context
        if cp_context is not None:
            hidden = cp_shard_rows(hidden, cp_context)
            positions = cp_shard_positions(positions, cp_context).contiguous()
        cos_sin_cache = self.rotary.cos_sin_cache
        residual: torch.Tensor | None = None
        prev_topk: torch.Tensor | None = None
        aux_hidden_buffer = self.aux_hidden_capture.create_buffer(hidden)
        for layer_id, layer in enumerate(self.layers):
            hidden, residual, prev_topk = layer(hidden, residual, positions, cos_sin_cache, prev_topk)
            self.aux_hidden_capture.capture_layer(layer_id, hidden, residual, aux_hidden_buffer)
            record_layer_event(layer_id)
        hidden, _ = self.norm(hidden, residual)
        if cp_context is not None:
            if aux_hidden_buffer is not None:
                # Merge target output and DFlash captures with the same row
                # permutation in one collective, including padded/empty ranks.
                width = hidden.shape[-1]
                merged = cp_merge_rows(torch.cat((hidden, aux_hidden_buffer), dim=-1), cp_context)
                hidden = merged[..., :width]
                aux_hidden_buffer = merged[..., width:]
            else:
                hidden = cp_merge_rows(hidden, cp_context)
        return self.aux_hidden_capture.finalize(hidden, aux_hidden_buffer)


def _resolve_qlinear(model: nn.Module, dotted: str) -> QLinear:
    """Resolve ``dotted`` to the QLinear, reusing GLM-Next's dotted-path walk."""
    mod = _resolve_module(model, dotted)
    assert isinstance(mod, QLinear), f"{dotted} is not a QLinear"
    return mod


class Glm52ForCausalLM(PyModelBase):
    """GLM-5.2 causal LM. Registered under ``model_type='glm_moe_dsa'``."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = Glm52Config.from_dict(config)
        self.cfg.tp_size = int(config.get("tp_size", 1))
        self.cfg.tp_rank = int(config.get("tp_rank", _tp_rank_from_device(config.get("device", "npu:0"))))
        self.cfg.ep_size = int(config.get("ep_size", 1))
        self.cfg.ep_rank = int(config.get("ep_rank", 0))
        self.cfg.dp_size = int(config.get("dp_size", 1))
        self.cfg.dp_rank = int(config.get("dp_rank", 0))
        self.cfg.cp_size = int(config.get("cp_size", 1))
        self.cfg.cp_rank = int(config.get("cp_rank", 0))
        self.cfg.layerwise_split_size = int(config.get("layerwise_split_size", 1))
        self.cfg.layerwise_split_rank = int(config.get("layerwise_split_rank", 0))
        self.cfg.moe_tp_size = int(config.get("moe_tp_size", 1))
        self.cfg.moe_tp_rank = int(config.get("moe_tp_rank", 0))
        self.cfg.world_size = int(config.get("world_size", self.cfg.tp_size * self.cfg.dp_size * self.cfg.cp_size))
        self.cfg.validate()
        validate_glm52_ep_execution_config(config, self.cfg)
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "cuda"))
        self.dtype = dtype
        self.device = device
        tp = self.cfg.tp_size
        assert self.cfg.vocab_size % tp == 0
        self.model = Glm52Model(self.cfg, dtype, device)
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

    def load_weights(
        self,
        state_dicts: list,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        cfg = self.cfg
        loader = W8A8WeightLoader(self, state_dicts, cfg.tp_size, cfg.tp_rank)

        # Quant-format probe, decided once from the first attention projection:
        # internal static exports (GLM-5.2-w8a8) carry q_a_proj.deq_scale, HF
        # per-channel dynamic exports (GLM-5.3-w8a8) carry
        # q_a_proj.weight_scale. Every W8A8 projection in the model uses the
        # same format, so the probe result pins every QLinear's kind. The
        # dense/shared MLP branch below is always dynamic-layout (both export
        # flavors ship MLP weight_scale/weight_offset) and does not read the
        # probe result.
        probe = "model.layers.0.self_attn.q_a_proj."
        has_deq_scale = loader.has(probe + "deq_scale")
        has_weight_scale = loader.has(probe + "weight_scale")
        if has_deq_scale and has_weight_scale:
            raise ValueError(
                "glm_moe_dsa q_a_proj carries both deq_scale (internal static "
                "export) and weight_scale (per-channel dynamic export) suffixes"
            )
        if not has_deq_scale and not has_weight_scale and len(state_dicts) > 0:
            raise ValueError(
                "glm_moe_dsa requires W8A8 projections: neither "
                "q_a_proj.deq_scale (internal static export) nor "
                "q_a_proj.weight_scale (per-channel dynamic export) was found"
            )
        # Empty state dicts (structure-only loads in unit tests) keep the
        # construction-time static default.
        quant_format = "dynamic" if has_weight_scale else "static"
        q_loader = QLinearWeightLoader(self, state_dicts, cfg.tp_size, cfg.tp_rank)

        def load_qlinear(prefix: str, proj: str, shard_dims: dict[str, int] | None) -> None:
            module = _resolve_qlinear(self, prefix + proj)
            module.resolve_quant(True, quant_format=quant_format)
            module.load_w8a8(q_loader, prefix, proj, shard_dims, quant_format=quant_format)

        loader.copy_shard("model.embed_tokens.weight", dim=1)

        for i in range(cfg.n_layers):
            p = f"model.layers.{i}."
            loader.copy_replicated(p + "input_layernorm.weight")
            loader.copy_replicated(p + "post_attention_layernorm.weight")
            attn = p + "self_attn."
            load_qlinear(attn, "q_a_proj", None)
            loader.copy_replicated(attn + "q_a_layernorm.weight")
            load_qlinear(
                attn, "q_b_proj", {"weight": 0, "deq_scale": 0, "quant_bias": 0, "weight_scale": 0, "weight_offset": 0}
            )
            load_qlinear(attn, "kv_a_proj_with_mqa", None)
            loader.copy_replicated(attn + "kv_a_layernorm.weight")
            loader.copy_shard(attn + "kv_b_proj.weight", dim=0)
            # Row-parallel o_proj shards only the weight (dim 1); the
            # per-output-channel scale/bias suffixes stay replicated.
            load_qlinear(attn, "o_proj", {"weight": 1})
            if not self.model.layers[i].self_attn.is_shared:
                idx = attn + "indexer."
                load_qlinear(idx, "wq_b", None)
                loader.copy_replicated(idx + "wk.weight")
                loader.copy_replicated(idx + "k_norm.weight")
                loader.copy_replicated(idx + "k_norm.bias")
                loader.copy_replicated(idx + "weights_proj.weight")
            self.model.layers[i].self_attn.process_weights_after_loading()

            self.model.layers[i].mlp.load_from_checkpoint(loader, p + "mlp.")

        loader.copy_replicated("model.norm.weight")
        loader.copy_shard("lm_head.weight", dim=0)
