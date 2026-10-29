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

"""DeepSeek-V4.1 DSpark draft model for the Python NPU executor.

The DSpark draft is the checkpoint's ``mtp.0/1/2`` block drafter (contract:
``docs/deepseek-v41-flash-xllm-port-plan.md`` section 2.6, design:
``self-improve/designs/dspark_v41_port.md``):

  * ``mtp.0.main_proj`` (ColumnParallel ``[n_capture * H -> H]``) + ``main_norm``
    project the captured target hidden states (layers 37/38/39) once; every
    draft layer then writes its own SWA cache row from that shared projection
    (``write_context_kv``).
  * Three V4.1 decoder layers (``mtp.0/1/2``) with the DSpark expert geometry
    (``dspark_n_routed_experts`` routed experts, top-``dspark_num_experts_per_tok``)
    and **pure SWA** attention (compress ratio 0 -- not the C++ V4 ratio-1
    default), the V4.1 mHC residual stream included.
  * ``mtp.2.norm`` + the last block's learned ``ffn_pre`` collapse, then the
    draft's own vocabulary head (dedicated ``mtp.2.head`` or the shared
    ``head``), the replicated markov head (``markov_w1`` embedding +
    ``markov_w2`` low-rank projection) and the replicated confidence head.

The execution pipeline includes the draft backbone and proposal heads in both
eager and ACL graph execution. The Markov and confidence heads are replicated;
the vocabulary projection retains its checkpoint-specific sharding.
The draft query block is anchor-first and non-causal; the SWA window semantics
live in the CSA2 attention backend, this module only feeds the block and the
pre-injected context rows.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import Optional

import torch
import torch.nn as nn

from scripts.logger import logger
from xllm.python import distributed
from xllm.python.layers.embedding import HiddenParallelEmbedding
from xllm.python.layers.linear import ColumnParallelLinear
from xllm.python.model_executor.forward_context import LayerSynchronizer
from xllm.python.models.deepseek_v4 import DeepseekV4ForCausalLM, DeepseekV4MoE
from xllm.python.models.deepseek_v32 import W8A8DynamicLinear
from xllm.python.models.deepseek_v41 import (
    CSA2_MODE_SWA,
    V41_QUANT_ASCEND,
    V41_QUANT_FP8,
    CSA2LayerMode,
    DeepseekV41Config,
    DeepseekV41DecoderLayer,
    DeepseekV41ForCausalLM,
    DeepseekV41HyperConnection,
    DeepseekV41MoE,
    DeepseekV41RMSNorm,
    _detect_v41_quant_mode,
    _find_checkpoint_prefix,
    _PlainRotaryEmbedding,
    dequant_fp8_block,
)
from xllm.python.models.dspark import DSparkForCausalLMBase
from xllm.python.models.weight_utils import W8A8WeightLoader

#: Draft stage count when the checkpoint/config does not say otherwise.
DSPARK_DEFAULT_LAYERS = 3

_DROPPED_DRAFT_TENSOR_LOGGED = False

#: Ascend (Eco-Tech / ModelSlim) export metadata that names a tensor's storage type.
_QUANT_DESCRIPTION_FILE = "quant_model_description.json"
#: Labels that mean "this tensor is exported unquantized" (bf16/fp16/fp32).
_FLOAT_QUANT_LABELS = frozenset({"FLOAT", "FLOAT16", "FLOAT32", "BF16", "BFLOAT16", "FP16", "FP32"})


def checkpoint_weight_is_quantized(model_path: Optional[str], tensor_name: str) -> Optional[bool]:
    """Whether ``quant_model_description.json`` marks ``tensor_name`` quantized.

    Returns ``None`` when the description file, the entry, or a usable label is
    missing so the caller can fall back to the official export contract instead
    of guessing. The metadata is the only signal available before the model is
    built; the actual tensor dtype is re-checked at load time.
    """
    if not model_path:
        return None
    path = os.path.join(model_path, _QUANT_DESCRIPTION_FILE)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        description = json.load(handle)
    label = description.get(tensor_name)
    if label is None:
        label = description.get("model." + tensor_name)
    if not isinstance(label, str):
        return None
    return label.strip().upper() not in _FLOAT_QUANT_LABELS


def _main_proj_quantized(quant_mode: str, config: dict) -> bool:
    """Whether the checkpoint quantizes ``mtp.0.main_proj`` (W8A8).

    Only the ascend export can quantize the draft. The official DSpark release
    keeps ``main_proj`` in BF16 (the vLLM reference builds it with
    ``quant_config=None``), so an absent or ambiguous description falls back to
    the unquantized path; a surprise int8 export is then caught by the loader's
    dtype guard instead of being silently cast.
    """
    if quant_mode != V41_QUANT_ASCEND:
        return False
    return bool(checkpoint_weight_is_quantized(config.get("model_path"), "mtp.0.main_proj.weight"))


def dspark_swa_mode(layer_id: int) -> CSA2LayerMode:
    """Pure-SWA draft assignment: compress ratio 0, no main-KV source.

    V4.1 draft layers own no compressor and no indexer (the C++ V4 default
    forces ratio 1, which the design corrects to 0).
    """
    return CSA2LayerMode(
        layer_id=layer_id,
        mode=CSA2_MODE_SWA,
        compress_ratio=0,
        main_kv_source_layer=-1,
    )


def anchor_first_query_block(
    anchor_token_ids: torch.Tensor,
    block_size: int,
    noise_token_id: int,
) -> torch.Tensor:
    """Build the anchor-first draft query block ``[..., block_size]``.

    Column 0 is the anchor (the last accepted token, whose hidden state the
    draft step continues from); columns 1.. carry the draft noise token, whose
    embedding row is the trained mask placeholder. The draft block predicts one
    token per column, so flattening the returned block yields exactly the row
    order the draft SWA attention consumes.
    """
    if block_size <= 0:
        raise ValueError("draft block_size must be positive")
    anchor = anchor_token_ids.reshape(-1, 1)
    noise = torch.full(
        (anchor.size(0), block_size - 1),
        int(noise_token_id),
        dtype=anchor.dtype,
        device=anchor.device,
    )
    return torch.cat((anchor, noise), dim=1)


def _swa_cache(cache: object) -> torch.Tensor | None:
    """SWA slot of a ``LayerCache``, a slot tuple, or a bare cache tensor.

    The draft layers are pure SWA, so ``write_context_kv`` writes the single
    pre-RoPE-context SWA row pool; the other cache slots never exist for them.
    """
    swa = getattr(cache, "swa", None)
    if swa is not None:
        return swa
    if isinstance(cache, torch.Tensor):
        return cache
    if isinstance(cache, (tuple, list)):
        from xllm.python.attention.backend import _LAYER_CACHE_SLOTS

        index = _LAYER_CACHE_SLOTS.index("swa")
        return cache[index] if len(cache) > index else None
    return None


def _first_present(loader: W8A8WeightLoader, *names: str) -> Optional[str]:
    for name in names:
        if loader.has(name):
            return name
    return None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DeepseekV41DSparkConfig(DeepseekV41Config):
    """DeepSeek-V4.1 DSpark draft config (V4.1 geometry + drafter head fields)."""

    # Low-rank width of the markov head (checkpoint ``dspark_markov_rank``).
    markov_rank: int = 256
    enable_confidence_head: bool = True
    confidence_head_with_markov: bool = True
    # Full-vocabulary draft: DSpark ships no d2t scatter table.
    draft_vocab_size: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> DeepseekV41DSparkConfig:
        base = DeepseekV41Config.from_dict(d)
        markov_rank = d.get("markov_rank")
        if markov_rank is None:
            markov_rank = base.dspark_markov_rank
        return cls(
            **base.__dict__,
            markov_rank=int(markov_rank if markov_rank is not None else 256),
            enable_confidence_head=bool(d.get("enable_confidence_head", True)),
            confidence_head_with_markov=bool(d.get("confidence_head_with_markov", True)),
            draft_vocab_size=int(d.get("draft_vocab_size") or base.vocab_size),
        )

    def validate(self) -> None:
        if self.dspark_block_size <= 0:
            raise ValueError("DeepSeek-V4.1 DSpark requires a positive block size")
        if self.markov_rank <= 0:
            raise ValueError("DeepSeek-V4.1 DSpark requires markov_rank > 0")
        if self.draft_vocab_size != self.vocab_size:
            raise ValueError("DeepSeek-V4.1 DSpark requires draft_vocab_size == vocab_size")
        if self.num_draft_layers <= 0:
            raise ValueError(
                "DeepSeek-V4.1 DSpark requires at least one draft layer "
                "(num_nextn_predict_layers or dspark_target_layer_ids)"
            )
        if self.hidden_size % self.tp_size:
            raise ValueError("hidden_size must be divisible by tp_size")
        if self.vocab_size % self.tp_size:
            raise ValueError("vocab_size must be divisible by tp_size")

    @property
    def num_draft_layers(self) -> int:
        """Draft block count: ``num_nextn_predict_layers`` (``mtp.0/1/2``).

        A bridged config that drops the count falls back to the captured-layer
        list, which the V4.1 contract pins to the same 3 stages.
        """
        if self.num_nextn_predict_layers > 0:
            return self.num_nextn_predict_layers
        return len(self.dspark_target_layer_ids)

    @property
    def main_proj_in_features(self) -> int:
        """``main_proj`` input width: one ``hidden_size`` block per captured layer."""
        return self.hidden_size * len(self.dspark_target_layer_ids)


# ---------------------------------------------------------------------------
# Context projection
# ---------------------------------------------------------------------------


class DSparkMainProjection(nn.Module):
    """Column-parallel ``[n_capture * H -> H]`` draft context projection.

    ``mtp.0.main_proj`` fuses the captured target hidden states into the single
    vector every draft layer consumes as context. The reference makes it a
    ColumnParallelLinear (output-dim shard + all-gather); the W8A8 dynamic
    kernel does not communicate, so an ascend export that quantizes it keeps the
    same output-dim shard and gathers explicitly. Both layouts compute the
    identical function. The official DSpark release exports ``main_proj`` in
    BF16, so ``quantized`` is resolved from the checkpoint metadata before the
    module is built (see :func:`_main_proj_quantized`).
    """

    def __init__(
        self,
        cfg: DeepseekV41DSparkConfig,
        quantized: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.in_features = cfg.main_proj_in_features
        self.tp_size = cfg.tp_size
        out_local = cfg.hidden_size // cfg.tp_size
        self.quantized = quantized
        self.w8a8: W8A8DynamicLinear | None = None
        self.linear: ColumnParallelLinear | None = None
        if quantized:
            self.w8a8 = W8A8DynamicLinear(
                self.in_features,
                out_local,
                device,
                transpose_weight_after_loading=False,
            )
        else:
            self.linear = ColumnParallelLinear(
                self.in_features,
                out_local,
                cfg.tp_size,
                gather_output=True,
                dtype=dtype,
                device=device,
            )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.size(-1) != self.in_features:
            raise ValueError("captured target hidden size does not match main_proj")
        if self.w8a8 is not None:
            out = self.w8a8(hidden)
            if self.tp_size > 1:
                out = distributed.tp_all_gather(out, dim=-1, world_size=self.tp_size)
            return out
        assert self.linear is not None
        return self.linear(hidden)

    def process_weights_after_loading(self) -> None:
        if self.w8a8 is not None:
            self.w8a8.process_weights_after_loading()


# ---------------------------------------------------------------------------
# Draft body
# ---------------------------------------------------------------------------


class DeepseekV41DSparkModel(nn.Module):
    """DSpark draft body: context projection + 3 pure-SWA layers + output norm."""

    def __init__(
        self,
        cfg: DeepseekV41DSparkConfig,
        dtype: torch.dtype,
        device: torch.device,
        quant_mode: str = "none",
        config: dict | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.quant_mode = quant_mode
        self.dtype = dtype
        self.device = device
        config = config or {}
        self.num_layers = cfg.num_draft_layers
        tp = cfg.tp_size
        # Draft layers are full V4.1 MoE layers with the draft's own expert
        # geometry and pure SWA attention; they carry no engram and no hash
        # routing. ``layer_cfg`` mirrors the checkpoint's mtp.* layout.
        self.layer_cfg = replace(
            cfg,
            n_layers=self.num_layers,
            compress_ratios=[0] * self.num_layers,
            kv_source_layer_ids=[],
            index_source_layer_ids=[],
            engram_layer_ids=[],
            n_hash_layers=0,
            n_routed_experts=cfg.dspark_n_routed_experts,
            n_activated_experts=cfg.dspark_num_experts_per_tok,
            first_k_dense_replace=0,
            moe_layer_freq=1,
        )
        # The draft owns the vocabulary embedding row of its noise/mask token
        # (dedicated ``mtp.0.embed`` on rotated checkpoints, else the shared
        # ``embed``); it is loaded from the checkpoint in load_weights.
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size,
            cfg.hidden_size // tp,
            tp,
            dtype=dtype,
            device=device,
        )
        self.main_proj = DSparkMainProjection(cfg, _main_proj_quantized(quant_mode, config), dtype, device)
        self.main_norm = DeepseekV41RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.layers = nn.ModuleList(
            [
                DeepseekV41DecoderLayer(
                    self.layer_cfg,
                    layer_id,
                    dtype,
                    device,
                    dspark_swa_mode(layer_id),
                    quant_mode,
                )
                for layer_id in range(self.num_layers)
            ]
        )
        # DSpark draft MoE layers have STRUCTURALLY uneven per-rank row counts
        # (a busy rank runs one sequence's whole block; an empty DP shard runs a
        # single dummy row), while the MC2 decode path's ``global_bs=0`` implies
        # equal rows on every rank and derives per-rank window slots/capacity from
        # that. The mismatch corrupts the cross-rank writes, so the MC2 guard must
        # decline the draft. Mark the INSTANCES (the draft reuses the target's
        # MoE classes, so a class attribute would also veto the target); the flag
        # is static and identical on every rank, keeping the collective decision
        # symmetric.
        for layer in self.layers:
            if isinstance(layer.mlp, (DeepseekV41MoE, DeepseekV4MoE)):
                layer.mlp._moe_is_dspark_draft = True
        self.norm = DeepseekV41RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        # Optional learned hc_head (design mapping group 21). V4.1 ships none
        # (the last block's ``ffn_pre`` collapse is the production head); the
        # parameters are registered by load_weights only when the checkpoint
        # provides them.
        self.hc_head_fn: nn.Parameter | None = None
        self.hc_head_base: nn.Parameter | None = None
        self.hc_head_scale: nn.Parameter | None = None
        # The draft is pure SWA: the plain theta-10000 RoPE table, YaRN off.
        self.rotary = _PlainRotaryEmbedding(
            cfg.qk_rope_head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            device,
        )

    @property
    def block_size(self) -> int:
        return self.cfg.dspark_block_size

    @property
    def noise_token_id(self) -> int:
        return self.cfg.dspark_noise_token_id

    def anchor_first_query_block(self, anchor_token_ids: torch.Tensor) -> torch.Tensor:
        """Anchor-first draft block for ``anchor_token_ids`` (see the module helper)."""
        return anchor_first_query_block(anchor_token_ids, self.block_size, self.noise_token_id)

    # -- forward ---------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One non-causal draft block step for either execution mode.

        ``input_ids`` is the anchor-first query block (see
        :func:`anchor_first_query_block`); the trailing context window comes
        from the rows :meth:`write_context_kv` pre-injected into the draft SWA
        caches. The registered attention builder supplies the non-causal
        block layout and its SWA cache inputs for either execution mode.
        """
        hidden = self.embed_tokens(input_ids) if input_embedding is None else input_embedding
        positions = positions.to(torch.int64).contiguous()

        cos_sin_cache = self.rotary.cos_sin_cache
        # Same mHC stream setup as the target: hc_mult identical copies with a
        # one-hot mix on copy 0 for the first block.
        hidden = hidden.unsqueeze(1).expand(-1, self.cfg.hc_mult, -1).contiguous()
        pre_mix = DeepseekV41HyperConnection.make_identity_pre_mix(hidden, self.cfg.hc_mult)
        for layer in self.layers:
            hidden, pre_mix = layer(hidden, positions, cos_sin_cache, input_ids, pre_mix)
        return self.norm(self._collapse_last_hidden(hidden, pre_mix))

    def _collapse_last_hidden(self, hidden: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """Last-block merge: learned hc_head when present, else the V4.1 collapse."""
        if self.hc_head_fn is not None:
            return self._hc_head(hidden)
        return self.layers[-1].hc.hc_pre(hidden, pre_mix)

    def _hc_head(self, hidden: torch.Tensor) -> torch.Tensor:
        """Learned hc_head collapse (only for checkpoints that ship one)."""
        assert self.hc_head_fn is not None
        shape, dtype = hidden.shape, hidden.dtype
        flat = hidden.flatten(1).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.cfg.rms_norm_eps)
        mixes = torch.nn.functional.linear(flat, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.cfg.hc_eps
        return (pre.unsqueeze(-1) * hidden.float()).sum(dim=1).to(dtype)

    # -- context-KV pre-injection ----------------------------------------------

    def write_context_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_slots: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor | None, ...]],
        layer_synchronizer: LayerSynchronizer | None = None,
    ) -> torch.Tensor | None:
        """Project the captured target hidden states and seed every draft SWA cache.

        ``main_proj -> main_norm`` runs once; each draft layer then applies its
        own ``wkv -> kv_norm -> RoPE`` (the exact bf16 chain its attention
        applies to a query-step KV) before scattering the rows into the layer's
        SWA cache slots. Returns the shared projection (``None`` when a layer
        event could not be recorded, matching the native draft contract).
        """
        from xllm.python.attention.csa2_attention import apply_interleaved_rope
        from xllm.python.attention.csa_attention import _scatter_by_slot

        if len(kv_caches) != self.num_layers:
            raise ValueError("draft KV cache layer count mismatch")
        if cache_slots.numel() != target_hidden.size(0):
            raise ValueError("draft context cache slot count mismatch")

        projected = self.main_norm(self.main_proj(target_hidden))
        positions = positions.to(torch.int64).contiguous()
        cos_sin_cache = self.rotary.cos_sin_cache
        for layer_id, (layer, cache) in enumerate(zip(self.layers, kv_caches)):
            attention = layer.self_attn
            kv = attention.kv_a_layernorm(attention.kv_proj(projected))
            kv = apply_interleaved_rope(kv, positions, cos_sin_cache)
            swa = _swa_cache(cache)
            if swa is None:
                raise RuntimeError(f"draft KV cache is missing the SWA pool for layer {layer_id}")
            _scatter_by_slot(swa, cache_slots, kv)
            if layer_synchronizer is not None and layer_synchronizer.record_event(layer_id) is False:
                return None
        return projected


# ---------------------------------------------------------------------------
# Causal LM: weight loading + DSpark head bridge
# ---------------------------------------------------------------------------


class DeepseekV41DSparkForCausalLM(DSparkForCausalLMBase):
    """DeepSeek-V4.1 DSpark draft driven by the C++ PyCausalLM bridge."""

    # The draft borrows the target's MoE loader by calling it unbound with this
    # instance (the draft does not inherit the V4 LM). ``_load_dsv4_moe`` now
    # dispatches its routing half through ``self._load_dsv4_moe_routing``, so
    # re-expose that static helper for the borrowed loader to resolve, exactly
    # as ``DeepseekV41ForCausalLM`` does for the V4.1 target.
    _load_dsv4_moe_routing = staticmethod(DeepseekV4ForCausalLM._load_dsv4_moe_routing)

    def __init__(self, config: dict) -> None:
        cfg = DeepseekV41DSparkConfig.from_dict(config)
        cfg = replace(cfg, dspark_block_size=int(config.get("draft_query_width", cfg.dspark_block_size)))
        cfg.validate()
        if int(config.get("cp_size", 1)) != 1:
            raise ValueError("DeepSeek-V4.1 DSpark block attention requires CP=1")
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "npu:0"))
        quant_mode = _detect_v41_quant_mode(config)
        backbone = DeepseekV41DSparkModel(cfg, dtype, device, quant_mode, config)
        super().__init__(
            backbone=backbone,
            lm_head=None,
            query_width=cfg.dspark_block_size,
            vocab_size=cfg.vocab_size,
            draft_vocab_size=cfg.draft_vocab_size,
            markov_rank=cfg.markov_rank,
            hidden_size=cfg.hidden_size,
            enable_confidence_head=cfg.enable_confidence_head,
            confidence_head_with_markov=cfg.confidence_head_with_markov,
            dtype=dtype,
            device=device,
        )
        self.cfg = cfg
        self.dtype = dtype
        self.device = device
        self.quant_mode = quant_mode
        self._lm_head_source: str | None = None

    def supports_device_only_decode_inputs(self, kv_page_size: int) -> bool:
        """The pipeline derives block attention metadata from device inputs."""
        del kv_page_size
        return True

    # -- names ----------------------------------------------------------------

    def _stage_prefix(self, stage: int) -> str:
        return f"mtp.{stage}."

    @property
    def _last_stage(self) -> int:
        return self.cfg.num_draft_layers - 1

    # -- weight loading --------------------------------------------------------

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        if tp_size != self.cfg.tp_size:
            raise ValueError(f"tp_size {tp_size} does not match the configured {self.cfg.tp_size}")
        # The loader snapshots parameters at construction, so the optional
        # vocabulary head is resolved (and created) first.
        self._prepare_own_lm_head(state_dicts)
        loader = W8A8WeightLoader(
            self,
            state_dicts,
            tp_size,
            tp_rank,
            name_aliases=self._head_aliases(),
        )
        self._load_body(loader)
        self._load_dspark_heads(loader)

    def _prepare_own_lm_head(self, state_dicts: list) -> None:
        """Create the draft's own lm_head when the checkpoint ships one."""
        head_key = None
        for name in (f"{self._stage_prefix(self._last_stage)}head.weight", "head.weight"):
            if W8A8WeightLoader.state_dict_has(state_dicts, name):
                head_key = name
                break
        if head_key is None:
            return
        self.model.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // self.cfg.tp_size,
            self.cfg.tp_size,
            gather_output=True,
            dtype=self.dtype,
            device=self.device,
        )
        self._lm_head_source = head_key

    def _head_aliases(self) -> dict[str, tuple[str, ...]]:
        """Runtime parameter name -> checkpoint names (first present wins).

        The export names the Markov matrices after their operation
        (``markov_head.embed/head``); the runtime uses the explicit
        embedding/projection names (``markov_w1/w2``). The operation names come
        first, the legacy runtime names stay as a fallback for older synthetic
        layouts.
        """
        last = self._last_stage
        return {
            "model.markov_head.markov_w1.weight": (
                f"mtp.{last}.markov_head.embed.weight",
                "markov_head.embed.weight",
                f"mtp.{last}.markov_head.markov_w1.weight",
                "markov_head.markov_w1.weight",
            ),
            "model.markov_head.markov_w2.weight": (
                f"mtp.{last}.markov_head.head.weight",
                "markov_head.head.weight",
                f"mtp.{last}.markov_head.markov_w2.weight",
                "markov_head.markov_w2.weight",
            ),
            "model.confidence_head.proj.weight": (
                f"mtp.{last}.confidence_head.proj.weight",
                "confidence_head.proj.weight",
            ),
            "model.confidence_head.proj.bias": (
                f"mtp.{last}.confidence_head.proj.bias",
                "confidence_head.proj.bias",
            ),
        }

    def _load_body(self, loader: W8A8WeightLoader) -> None:
        if self.quant_mode == V41_QUANT_ASCEND:
            self._load_body_ascend(loader)
        else:
            self._load_body_torch(loader)

    def _load_dspark_heads(self, loader: W8A8WeightLoader) -> None:
        """Load the replicated markov/confidence heads and the optional lm_head."""
        head_source = self._lm_head_source
        if head_source is not None:
            loader.copy_in(
                "model.lm_head.weight",
                loader.shard(loader.load_tensor(head_source), dim=0),
            )
        # Low-rank markov head and the (bias-less) confidence head stay
        # replicated: the serial per-position draft sampling must not collect.
        loader.copy_replicated("model.markov_head.markov_w1.weight")
        loader.copy_replicated("model.markov_head.markov_w2.weight")
        if self.model.confidence_head is not None:
            loader.copy_replicated("model.confidence_head.proj.weight")
            if loader.has("model.confidence_head.proj.bias"):
                loader.copy_replicated("model.confidence_head.proj.bias")
            else:
                # V4.1 ships a bias-less projection; keep the parameter but
                # neutralise the module-init bias the checkpoint does not own.
                with torch.no_grad():
                    self.model.confidence_head.proj.bias.zero_()
                logger.info("DeepSeek-V4.1 DSpark confidence head is bias-less; zeroed the bias")

    # -- ascend (W8A8 export) body loading --------------------------------------

    def _load_body_ascend(self, loader: W8A8WeightLoader) -> None:
        cfg = self.cfg
        loader.copy_in(
            "model.backbone.embed_tokens.weight",
            loader.shard(loader.load_tensor(self._embed_source(loader)), dim=1),
        )
        self._load_main_proj_ascend(loader)
        for stage in range(cfg.num_draft_layers):
            ck = self._stage_prefix(stage)
            pm = f"model.backbone.layers.{stage}."
            self._load_draft_attn_ascend(loader, ck, pm, stage)
            DeepseekV41ForCausalLM._load_common_layer_norms(self, loader, ck, pm)
            mlp = self.model.backbone.layers[stage].mlp
            attn = self.model.backbone.layers[stage].self_attn
            attn.process_weights_after_loading()
            # The W8A8 export stores the draft MoE block under either the native
            # ``ffn.`` or the legacy ``mlp.`` prefix; resolve it against this
            # draft stage's checkpoint prefix and pass it through, mirroring the
            # V4.1 target loader (deepseek_v41._load_ascend_moe).
            source_prefix = _find_checkpoint_prefix(
                loader,
                (ck + "ffn.", ck + "mlp."),
                ("experts.0.w1.weight",),
            )
            if source_prefix is None:
                raise KeyError(f"DeepSeek-V4.1 DSpark stage {stage} MoE expert weights not found")
            DeepseekV4ForCausalLM._load_dsv4_moe(self, loader, ck, pm, stage, source_prefix)
            mlp.process_weights_after_loading()
            self._report_ignored_draft_tensors(loader, ck, mlp)
        loader.copy_in(
            "model.backbone.norm.weight",
            loader.load_tensor(self._stage_prefix(self._last_stage) + "norm.weight"),
        )
        self.model.backbone.main_proj.process_weights_after_loading()
        self._load_hc_head(loader)

    def _load_main_proj_ascend(self, loader: W8A8WeightLoader) -> None:
        """Load ``mtp.0.main_proj`` in the layout the checkpoint declares.

        The ascend export quantizes the draft projections it lists as
        ``W8A8_DYNAMIC``; ``main_proj`` is usually kept in BF16 (official
        contract), so the module was built for that layout. The tensor dtype is
        re-checked here so a metadata/export disagreement fails fast instead of
        casting int8 into bf16 or vice versa.
        """
        prefix = self._stage_prefix(0) + "main_proj"
        if self.model.backbone.main_proj.quantized:
            self._load_w8a8(
                loader,
                prefix,
                "model.backbone.main_proj.w8a8",
                {"weight": 0, "weight_scale": 0, "weight_offset": 0},
            )
        else:
            self._load_dense_weight(loader, prefix, "model.backbone.main_proj.linear", shard_dim=0)
        loader.copy_in(
            "model.backbone.main_norm.weight",
            loader.load_tensor(self._stage_prefix(0) + "main_norm.weight"),
        )

    def _load_dense_weight(
        self,
        loader: W8A8WeightLoader,
        ckpt_prefix: str,
        param_prefix: str,
        shard_dim: int | None = None,
    ) -> None:
        """Load one unquantized projection weight (bf16/fp16/fp32).

        The draft fixes the layout of a projection before any tensor is visible
        (from ``quant_model_description.json``), so an int8 tensor reaching a
        dense parameter means the description and the export disagree; reject it
        rather than let ``copy_`` cast int8 codes into bf16 weights.
        """
        key = f"{ckpt_prefix}.weight"
        tensor = loader.load_tensor(key)
        if tensor.dtype in (torch.int8, torch.uint8):
            raise ValueError(
                f"{key} has dtype {tensor.dtype} but the draft built it as an "
                f"unquantized {self.dtype} projection; quant_model_description.json "
                "must declare it W8A8 (int8 weight + weight_scale)"
            )
        if shard_dim is not None:
            tensor = loader.shard(tensor, dim=shard_dim)
        loader.copy_in(f"{param_prefix}.weight", tensor)

    def _load_draft_attn_ascend(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        pm: str,
        stage: int,
    ) -> None:
        attn = self.model.backbone.layers[stage].self_attn
        self._load_w8a8(loader, ck + "attn.wq_a", pm + "self_attn.q_a_proj")
        self._load_w8a8(
            loader,
            ck + "attn.wq_b",
            pm + "self_attn.q_b_proj",
            {"weight": 0, "weight_scale": 0, "weight_offset": 0},
        )
        self._load_w8a8(loader, ck + "attn.wkv", pm + "self_attn.kv_proj")
        loader.copy_in(
            pm + "self_attn.o_a_proj.weight",
            loader.shard(loader.load_tensor(ck + "attn.wo_a.weight"), dim=0),
        )
        loader.copy_in(
            pm + "self_attn.o_b_proj.weight",
            loader.shard(loader.load_tensor(ck + "attn.wo_b.weight"), dim=1),
        )
        loader.copy_in(
            pm + "self_attn.q_a_layernorm.weight",
            loader.load_tensor(ck + "attn.q_norm.weight"),
        )
        loader.copy_in(
            pm + "self_attn.kv_a_layernorm.weight",
            loader.load_tensor(ck + "attn.kv_norm.weight"),
        )
        self._load_attn_sink(loader, ck, pm, attn)

    def _load_w8a8(
        self,
        loader: W8A8WeightLoader,
        ckpt_prefix: str,
        param_prefix: str,
        shard_dims: dict | None = None,
    ) -> None:
        """Load one W8A8-dynamic projection (weight + scale + offset).

        Fails fast instead of silently casting: a bf16 ``weight`` (i.e. a
        checkpoint that does not quantize this projection) would otherwise be
        written into an int8 parameter as garbage. A missing ``weight_offset``
        is treated as the implied zero of a symmetric-int8 export.
        """
        for suffix in ("weight", "weight_scale", "weight_offset"):
            key = f"{ckpt_prefix}.{suffix}"
            if not loader.has(key):
                if suffix == "weight_offset":
                    self.get_buffer(f"{param_prefix}.weight_offset").zero_()
                    continue
                raise KeyError(f"ascend W8A8 draft tensor not found: {key}")
            tensor = loader.load_tensor(key)
            if suffix == "weight" and tensor.dtype not in (torch.int8, torch.uint8):
                raise ValueError(f"{key} has dtype {tensor.dtype}; the ascend W8A8 draft expects int8 weights")
            dim = (shard_dims or {}).get(suffix)
            if dim is not None:
                tensor = loader.shard(tensor, dim=dim)
            loader.copy_in(f"{param_prefix}.{suffix}", tensor)

    def _load_attn_sink(self, loader: W8A8WeightLoader, ck: str, pm: str, attn: nn.Module) -> None:
        for key in (ck + "attn.attn_sink", ck + "attn.attn_sink.weight"):
            if not loader.has(key):
                continue
            sink = loader.load_tensor(key)
            if sink.dim() == 1 and sink.size(0) == self.cfg.n_heads and loader.tp_size > 1:
                shard = self.cfg.n_heads // loader.tp_size
                sink = sink.narrow(0, loader.tp_rank * shard, shard)
            loader.copy_in(pm + "self_attn.attn_sink", sink)
            attn.attn_sink_loaded = True
            return

    def _report_ignored_draft_tensors(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        mlp: nn.Module | None = None,
    ) -> None:
        """Report the checkpoint tensors the text-only draft deliberately drops.

        The multimodal checkpoint carries a per-layer vision router bias, and
        MoE gates on some exports carry quant companions. DSpark runs text-only
        decode and its gate follows the target's bf16 ``ffn.gate.weight``
        contract, so both are dropped rather than mis-aliased (a
        ``bias_vl`` aliased onto ``e_score_correction_bias`` would change text
        routing). The reused target MoE loader does arm ``bias_vl`` when
        present, so disarm it here too.
        """
        global _DROPPED_DRAFT_TENSOR_LOGGED
        ignored = [
            name
            for name in ("ffn.gate.bias_vl", "ffn.gate.weight_scale", "ffn.gate.weight_offset")
            if loader.has(ck + name)
        ]
        if ignored and not _DROPPED_DRAFT_TENSOR_LOGGED:
            _DROPPED_DRAFT_TENSOR_LOGGED = True
            logger.info(
                "DSpark draft ignores %s under %s (text-only drafter, bf16 gate)",
                ", ".join(ignored),
                ck,
            )
        if mlp is not None and getattr(mlp, "gate_bias_vl_loaded", False):
            with torch.no_grad():
                mlp.gate_bias_vl.zero_()
            mlp.gate_bias_vl_loaded = False

    # -- torch (fp8 / bf16) body loading ---------------------------------------

    def _load_body_torch(self, loader: W8A8WeightLoader) -> None:
        cfg = self.cfg
        is_fp8 = self.quant_mode == V41_QUANT_FP8

        def _linear(ckpt_prefix: str, param_prefix: str, shard_dim: int | None = None) -> None:
            if is_fp8:
                weight = dequant_fp8_block(
                    loader.load_tensor(ckpt_prefix + ".weight"),
                    loader.load_tensor(ckpt_prefix + ".scale"),
                )
            else:
                weight = loader.load_tensor(ckpt_prefix + ".weight")
            if shard_dim is not None:
                weight = loader.shard(weight, dim=shard_dim)
            loader.copy_in(param_prefix + ".weight", weight)

        loader.copy_in(
            "model.backbone.embed_tokens.weight",
            loader.shard(loader.load_tensor(self._embed_source(loader)), dim=1),
        )
        loader.copy_in(
            "model.backbone.main_proj.linear.weight",
            loader.shard(loader.load_tensor("mtp.0.main_proj.weight"), dim=0),
        )
        loader.copy_in("model.backbone.main_norm.weight", loader.load_tensor("mtp.0.main_norm.weight"))
        for stage in range(cfg.num_draft_layers):
            ck = self._stage_prefix(stage)
            pm = f"model.backbone.layers.{stage}."
            for ckpt_name, param_name, shard_dim in (
                ("attn.wq_a", "self_attn.q_a_proj", None),
                ("attn.wq_b", "self_attn.q_b_proj", 0),
                ("attn.wkv", "self_attn.kv_proj", None),
                ("attn.wo_a", "self_attn.o_a_proj", 0),
                ("attn.wo_b", "self_attn.o_b_proj", 1),
            ):
                _linear(ck + ckpt_name, pm + param_name, shard_dim)
            loader.copy_in(
                pm + "self_attn.q_a_layernorm.weight",
                loader.load_tensor(ck + "attn.q_norm.weight"),
            )
            loader.copy_in(
                pm + "self_attn.kv_a_layernorm.weight",
                loader.load_tensor(ck + "attn.kv_norm.weight"),
            )
            self._load_attn_sink(loader, ck, pm, self.model.backbone.layers[stage].self_attn)
            DeepseekV41ForCausalLM._load_common_layer_norms(self, loader, ck, pm)
            mlp = self.model.backbone.layers[stage].mlp
            if not isinstance(mlp, DeepseekV41MoE):
                raise ValueError(f"draft layer {stage} must be a torch MoE layer, got {type(mlp).__name__}")
            DeepseekV41ForCausalLM._load_torch_moe(self, loader, ck, pm, stage, is_fp8)
            self._report_ignored_draft_tensors(loader, ck, mlp)
        loader.copy_in(
            "model.backbone.norm.weight",
            loader.load_tensor(self._stage_prefix(self._last_stage) + "norm.weight"),
        )
        self._load_hc_head(loader)

    def _embed_source(self, loader: W8A8WeightLoader) -> str:
        """Draft vocabulary embedding: dedicated ``mtp.0.embed`` wins over ``embed``."""
        source = _first_present(loader, "mtp.0.embed.weight", "embed.weight")
        if source is None:
            raise KeyError("DSpark draft embedding not found: mtp.0.embed.weight or embed.weight")
        return source

    def _load_hc_head(self, loader: W8A8WeightLoader) -> None:
        """Register the optional learned hc_head (V4.1 checkpoints ship none)."""
        last = self._last_stage
        source = _first_present(loader, f"mtp.{last}.hc_head_fn")
        if source is None:
            return
        for suffix in ("fn", "base", "scale"):
            key = f"mtp.{last}.hc_head_{suffix}"
            tensor = loader.load_tensor(key)
            parameter = nn.Parameter(torch.empty(tensor.shape, dtype=torch.float32, device=self.device))
            parameter.data.copy_(tensor.float())
            setattr(self.model.backbone, f"hc_head_{suffix}", parameter)

    # -- runtime bridge --------------------------------------------------------

    def write_context_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_slots: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor | None, ...]],
        layer_synchronizer: LayerSynchronizer | None = None,
    ) -> torch.Tensor | None:
        return self.model.backbone.write_context_kv(
            target_hidden,
            positions,
            cache_slots,
            kv_caches,
            layer_synchronizer,
        )


# Registration is centralised in xllm.python.registry; import there.
