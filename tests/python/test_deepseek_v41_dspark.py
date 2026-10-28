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

"""DeepSeek-V4.1 DSpark draft model: config, ``mtp.*`` weight mapping, the
context-KV pre-injection math and the eager draft block step.

Geometry notes: ``head_dim`` must be a multiple of the fp8 QDQ group (32) and
``qk_rope_head_dim`` must be even, matching the V4.1 forward contract; the
noise token id is kept inside the tiny vocabulary so the draft embedding row
exists.
"""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from xllm.python import kernels
from xllm.python.attention.csa2_attention import (
    Csa2QueryLayout,
    _block_query_lens,
    apply_interleaved_rope,
)
from xllm.python.attention.csa2_block_metadata import (
    Csa2BlockAttentionMetadata,
    Csa2BlockAttentionMetadataBuilder,
)
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.model_executor.sampling_metadata import ModelSamplingMetadata, ModelSamplingMetadataBuilder
from xllm.python.models import deepseek_v41_dspark as dsp
from xllm.python.models.deepseek_v41_dspark import (
    CSA2_MODE_SWA,
    DeepseekV41DSparkConfig,
    DeepseekV41DSparkForCausalLM,
    anchor_first_query_block,
    dspark_swa_mode,
)
from xllm.python.models.dspark_pipeline import DSparkPipeline
from xllm.python.registry import get_execution_metadata_builder_classes, get_model_class

# Tiny draft geometry: hidden 32 / head_dim 64 (QDQ group 32) / rope 16 /
# 2 draft layers over 2 captured target layers.
_DEEPSEEK_V41_DSPARK_CONFIG = {
    "model_type": "deepseek_v41",
    "architectures": ["DeepseekV41ForCausalLM"],
    "hidden_size": 32,
    "num_hidden_layers": 4,
    "num_attention_heads": 2,
    "head_dim": 64,
    "q_lora_rank": 8,
    "qk_rope_head_dim": 16,
    "o_lora_rank": 8,
    "o_groups": 2,
    "vocab_size": 32,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "max_position_embeddings": 32,
    "window_size": 8,
    "n_routed_experts": 8,
    "n_activated_experts": 2,
    "moe_intermediate_size": 8,
    "hc_mult": 2,
    "hc_sinkhorn_iters": 2,
    "n_hash_layers": 0,
    "first_k_dense_replace": 0,
    "compress_ratios": [0] * 6,
    "num_nextn_predict_layers": 2,
    "dspark_target_layer_ids": [1, 2],
    "dspark_block_size": 3,
    "dspark_noise_token_id": 7,
    "dspark_markov_rank": 4,
    "dspark_n_routed_experts": 4,
    "dspark_num_experts_per_tok": 2,
    "image_token_id": 3,
    "tp_size": 1,
    "tp_rank": 0,
    "dp_size": 1,
    "dp_rank": 0,
    "dtype": "float32",
    "device": "cpu",
}


def _config_dict(**overrides) -> dict:
    values = dict(_DEEPSEEK_V41_DSPARK_CONFIG)
    values.update(overrides)
    return values


class _FakeStateDict:
    """Minimal StateDict stand-in (has/get_tensor) for the loading tests."""

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


def _moe_intermediate(cfg: DeepseekV41DSparkConfig) -> int:
    return cfg.moe_intermediate_size


def _expected_shapes(cfg: DeepseekV41DSparkConfig) -> dict[str, tuple[int, ...]]:
    """Checkpoint-key -> shape for every ``mtp.*`` tensor of one draft stage."""
    hidden = cfg.hidden_size
    q_lora = cfg.q_lora_rank
    head_dim = cfg.head_dim
    inter = _moe_intermediate(cfg)
    n_experts = cfg.dspark_n_routed_experts
    o_lora = cfg.o_lora_rank
    mix_hc = (2 + cfg.hc_mult) * cfg.hc_mult
    hc_dim = cfg.hc_mult * hidden
    shapes: dict[str, tuple[int, ...]] = {
        "attn.wq_a.weight": (q_lora, hidden),
        "attn.wq_b.weight": (cfg.n_heads * head_dim, q_lora),
        "attn.wkv.weight": (head_dim, hidden),
        "attn.wo_a.weight": (cfg.o_groups * o_lora, (cfg.n_heads * head_dim) // cfg.o_groups),
        "attn.wo_b.weight": (hidden, cfg.o_groups * o_lora),
        "attn.q_norm.weight": (q_lora,),
        "attn.kv_norm.weight": (head_dim,),
        "attn.attn_sink": (cfg.n_heads,),
        "attn_norm.weight": (hidden,),
        "ffn_norm.weight": (hidden,),
        "hc_attn_fn": (mix_hc, hc_dim),
        "hc_attn_base": (mix_hc,),
        "hc_attn_scale": (3,),
        "hc_ffn_fn": (mix_hc, hc_dim),
        "hc_ffn_base": (mix_hc,),
        "hc_ffn_scale": (3,),
        "ffn.gate.weight": (n_experts, hidden),
        "ffn.gate.bias": (n_experts,),
        "ffn.gate.bias_vl": (n_experts,),
        "ffn.shared_experts.w1.weight": (inter, hidden),
        "ffn.shared_experts.w2.weight": (hidden, inter),
        "ffn.shared_experts.w3.weight": (inter, hidden),
    }
    for expert in range(n_experts):
        shapes[f"ffn.experts.{expert}.w1.weight"] = (inter, hidden)
        shapes[f"ffn.experts.{expert}.w2.weight"] = (hidden, inter)
        shapes[f"ffn.experts.{expert}.w3.weight"] = (inter, hidden)
    return shapes


def _draft_group_keys(cfg: DeepseekV41DSparkConfig) -> set[str]:
    """Every ``mtp.*`` checkpoint key the draft is expected to consume."""
    shapes = _expected_shapes(cfg)
    keys: set[str] = set()
    for stage in range(cfg.num_draft_layers):
        keys.update(f"mtp.{stage}.{name}" for name in shapes)
    last = cfg.num_draft_layers - 1
    keys.add("mtp.0.main_proj.weight")
    keys.add("mtp.0.main_norm.weight")
    keys.add("mtp.0.embed.weight")
    keys.add(f"mtp.{last}.norm.weight")
    keys.add(f"mtp.{last}.head.weight")
    keys.add(f"mtp.{last}.markov_head.markov_w1.weight")
    keys.add(f"mtp.{last}.markov_head.markov_w2.weight")
    keys.add(f"mtp.{last}.confidence_head.proj.weight")
    return keys


def _build_checkpoint(
    cfg: DeepseekV41DSparkConfig,
    fill,
    *,
    dedicated_vocab: bool = True,
    hc_head: bool = False,
    confidence_bias: bool = False,
) -> dict[str, torch.Tensor]:
    """Synthetic ``mtp.*`` checkpoint in the reference (torch) layout."""
    tensors: dict[str, torch.Tensor] = {}
    hidden = cfg.hidden_size
    for stage in range(cfg.num_draft_layers):
        for name, shape in _expected_shapes(cfg).items():
            tensors[f"mtp.{stage}.{name}"] = fill(f"mtp.{stage}.{name}", shape)
    main_in = cfg.hidden_size * len(cfg.dspark_target_layer_ids)
    tensors["mtp.0.main_proj.weight"] = fill("mtp.0.main_proj.weight", (hidden, main_in))
    tensors["mtp.0.main_norm.weight"] = fill("mtp.0.main_norm.weight", (hidden,))
    if dedicated_vocab:
        tensors["mtp.0.embed.weight"] = fill("mtp.0.embed.weight", (cfg.vocab_size, hidden))
    else:
        tensors["embed.weight"] = fill("embed.weight", (cfg.vocab_size, hidden))
    last = cfg.num_draft_layers - 1
    tensors[f"mtp.{last}.norm.weight"] = fill(f"mtp.{last}.norm.weight", (hidden,))
    if dedicated_vocab:
        tensors[f"mtp.{last}.head.weight"] = fill(f"mtp.{last}.head.weight", (cfg.vocab_size, hidden))
    else:
        tensors["head.weight"] = fill("head.weight", (cfg.vocab_size, hidden))
    tensors[f"mtp.{last}.markov_head.markov_w1.weight"] = fill(
        f"mtp.{last}.markov_head.markov_w1.weight", (cfg.vocab_size, cfg.dspark_markov_rank)
    )
    tensors[f"mtp.{last}.markov_head.markov_w2.weight"] = fill(
        f"mtp.{last}.markov_head.markov_w2.weight", (cfg.vocab_size, cfg.dspark_markov_rank)
    )
    tensors[f"mtp.{last}.confidence_head.proj.weight"] = fill(
        f"mtp.{last}.confidence_head.proj.weight", (1, hidden + cfg.dspark_markov_rank)
    )
    if confidence_bias:
        tensors[f"mtp.{last}.confidence_head.proj.bias"] = fill(f"mtp.{last}.confidence_head.proj.bias", (1,))
    if hc_head:
        mix_hc = cfg.hc_mult
        tensors[f"mtp.{last}.hc_head_fn"] = fill(f"mtp.{last}.hc_head_fn", (mix_hc, cfg.hc_mult * hidden))
        tensors[f"mtp.{last}.hc_head_base"] = fill(f"mtp.{last}.hc_head_base", (mix_hc,))
        tensors[f"mtp.{last}.hc_head_scale"] = fill(f"mtp.{last}.hc_head_scale", (1,))
    return tensors


def _load(lm: DeepseekV41DSparkForCausalLM, tensors: dict[str, torch.Tensor]) -> None:
    lm.load_weights([_FakeStateDict(tensors)], 0, lm.cfg.tp_size)


def _with_patched_forward_context(monkeypatch: pytest.MonkeyPatch, backend) -> None:
    """Point both the draft module and the V4.1 attention at a stub backend."""
    import xllm.python.models.deepseek_v41 as v41

    context = SimpleNamespace(attention_backend=backend, metadata=SimpleNamespace(), execution_contexts={})

    monkeypatch.setattr(v41, "get_forward_context", lambda: context)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_reads_dspark_fields() -> None:
    cfg = DeepseekV41DSparkConfig.from_dict(_config_dict())
    cfg.validate()
    assert cfg.markov_rank == 4
    assert cfg.dspark_markov_rank == 4
    assert cfg.draft_vocab_size == cfg.vocab_size
    assert cfg.num_draft_layers == 2
    assert cfg.main_proj_in_features == 2 * cfg.hidden_size
    assert cfg.enable_confidence_head
    assert cfg.confidence_head_with_markov
    # A plain V4.1 dict still validates with the DSpark defaults.
    defaults = DeepseekV41DSparkConfig.from_dict({"model_type": "deepseek_v41", "n_layers": 6})
    defaults.validate()
    assert defaults.markov_rank == 256
    assert defaults.draft_vocab_size == defaults.vocab_size


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"markov_rank": 0}, "markov_rank > 0"),
        ({"draft_vocab_size": 8}, "draft_vocab_size == vocab_size"),
        ({"num_nextn_predict_layers": 0, "dspark_target_layer_ids": []}, "at least one draft layer"),
        ({"hidden_size": 33, "tp_size": 2}, "divisible by tp_size"),
    ],
)
def test_config_validate_rejects_bad_geometry(overrides: dict, match: str) -> None:
    cfg = DeepseekV41DSparkConfig.from_dict(_config_dict(**overrides))
    with pytest.raises(ValueError, match=match):
        cfg.validate()


def test_config_falls_back_to_the_captured_layer_count() -> None:
    """A bridged config without the draft count still yields the 3 mtp stages."""
    cfg = DeepseekV41DSparkConfig.from_dict(_config_dict(num_nextn_predict_layers=0))
    cfg.validate()
    assert cfg.num_draft_layers == len(cfg.dspark_target_layer_ids) == 2


def test_registry_resolves_deepseek_v41_dspark() -> None:
    assert get_model_class("deepseek_v41_dspark").__name__ == "DeepseekV41DSparkForCausalLM"


def test_config_accepts_the_cpp_routed_draft_model_type() -> None:
    """The C++ draft route hands PyCausalLM this exact model_type.

    ``configure_deepseek_v4_dspark_args`` rewrites the draft's model_type to
    ``util::kDeepseekV41DSparkModelType`` for ``--model_impl=python``, and
    PyCausalLM forwards the same string as ``config["model_type"]``. The draft
    config must therefore accept it (the V4.1 family default is overridden, not
    required), and the model class must still be the registered draft one.
    """
    cfg = DeepseekV41DSparkConfig.from_dict(_config_dict(model_type="deepseek_v41_dspark"))
    cfg.validate()
    assert cfg.model_type == "deepseek_v41_dspark"
    assert cfg.num_draft_layers == len(cfg.dspark_target_layer_ids) == 2
    lm = DeepseekV41DSparkForCausalLM(_config_dict(model_type="deepseek_v41_dspark"))
    assert lm.cfg.model_type == "deepseek_v41_dspark"
    assert len(lm.model.backbone.layers) == 2


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_draft_layers_are_pure_swa_and_moe() -> None:
    assert dspark_swa_mode(2).mode == CSA2_MODE_SWA
    assert dspark_swa_mode(2).compress_ratio == 0
    assert dspark_swa_mode(2).main_kv_source_layer == -1
    assert not dspark_swa_mode(2).is_csa2

    lm = DeepseekV41DSparkForCausalLM(_config_dict())
    assert lm.quant_mode == "none"
    assert len(lm.model.backbone.layers) == 2
    for layer in lm.model.backbone.layers:
        attn = layer.self_attn
        assert attn.mode.mode == CSA2_MODE_SWA
        assert attn.mode.compress_ratio == 0
        assert attn.indexer is None
        assert not hasattr(attn, "cmp_wkv")
        # Draft expert geometry, not the target's: 4 routed experts, top-2.
        assert layer.mlp.n_routed_experts == 4
        assert layer.mlp.topk == 2
        assert layer.engram is None
    cfg = lm.cfg
    assert lm.model.backbone.main_proj.in_features == cfg.main_proj_in_features
    assert lm.model.backbone.main_proj.linear.weight.shape == (cfg.hidden_size, cfg.main_proj_in_features)
    assert lm.model.backbone.main_norm.weight.shape == (cfg.hidden_size,)
    assert lm.model.backbone.norm.weight.shape == (cfg.hidden_size,)
    assert lm.model.backbone.embed_tokens.weight.shape == (cfg.vocab_size, cfg.hidden_size)
    # V4.1 ships no learned hc_head: the last block's pre_mix collapse is the
    # production head, so the parameters stay unregistered.
    assert lm.model.backbone.hc_head_fn is None and lm.model.backbone.hc_head_base is None
    assert lm.model.lm_head is None
    assert tuple(lm.model.markov_head.markov_w1.weight.shape) == (cfg.vocab_size, cfg.markov_rank)
    assert tuple(lm.model.markov_head.markov_w2.weight.shape) == (cfg.vocab_size, cfg.markov_rank)
    assert lm.model.confidence_head is not None
    assert tuple(lm.model.confidence_head.proj.weight.shape) == (1, cfg.hidden_size + cfg.markov_rank)
    assert lm.has_dspark_confidence_head()


# ---------------------------------------------------------------------------
# Weight mapping (torch / none checkpoint mode)
# ---------------------------------------------------------------------------


def _assert_loaded(lm: DeepseekV41DSparkForCausalLM, tensors: dict[str, torch.Tensor]) -> None:
    """Every ``mtp.*`` tensor must land in its documented parameter."""
    cfg = lm.cfg
    last = cfg.num_draft_layers - 1
    assert torch.equal(lm.model.backbone.embed_tokens.weight, tensors["mtp.0.embed.weight"])
    assert torch.equal(lm.model.backbone.main_proj.linear.weight, tensors["mtp.0.main_proj.weight"])
    assert torch.equal(lm.model.backbone.main_norm.weight, tensors["mtp.0.main_norm.weight"])
    assert torch.equal(lm.model.backbone.norm.weight, tensors[f"mtp.{last}.norm.weight"])
    assert lm.model.lm_head is not None
    assert torch.equal(lm.model.lm_head.weight, tensors[f"mtp.{last}.head.weight"])
    assert torch.equal(lm.model.markov_head.markov_w1.weight, tensors[f"mtp.{last}.markov_head.markov_w1.weight"])
    assert torch.equal(lm.model.markov_head.markov_w2.weight, tensors[f"mtp.{last}.markov_head.markov_w2.weight"])
    assert lm.model.confidence_head is not None
    assert torch.equal(lm.model.confidence_head.proj.weight, tensors[f"mtp.{last}.confidence_head.proj.weight"])

    for stage in range(cfg.num_draft_layers):
        ck = f"mtp.{stage}."
        layer = lm.model.backbone.layers[stage]
        attn = layer.self_attn
        assert torch.equal(attn.q_a_proj.weight, tensors[ck + "attn.wq_a.weight"])
        assert torch.equal(attn.q_b_proj.weight, tensors[ck + "attn.wq_b.weight"])
        assert torch.equal(attn.kv_proj.weight, tensors[ck + "attn.wkv.weight"])
        assert torch.equal(attn.o_a_proj.weight, tensors[ck + "attn.wo_a.weight"])
        assert torch.equal(attn.o_b_proj.weight, tensors[ck + "attn.wo_b.weight"])
        assert torch.equal(attn.q_a_layernorm.weight, tensors[ck + "attn.q_norm.weight"])
        assert torch.equal(attn.kv_a_layernorm.weight, tensors[ck + "attn.kv_norm.weight"])
        assert attn.attn_sink_loaded
        assert torch.equal(attn.attn_sink, tensors[ck + "attn.attn_sink"].float())
        assert torch.equal(layer.input_layernorm.weight, tensors[ck + "attn_norm.weight"])
        assert torch.equal(layer.post_attention_layernorm.weight, tensors[ck + "ffn_norm.weight"])
        for part in ("attn", "ffn"):
            for suffix in ("fn", "base", "scale"):
                assert torch.equal(
                    getattr(layer.hc, f"hc_{part}_{suffix}"),
                    tensors[ck + f"hc_{part}_{suffix}"].float(),
                )
        mlp = layer.mlp
        assert torch.equal(mlp.gate.weight, tensors[ck + "ffn.gate.weight"])
        assert torch.equal(mlp.e_score_correction_bias, tensors[ck + "ffn.gate.bias"])
        # The draft is text-only: the vision router bias is never armed, even
        # though the multimodal checkpoint ships it.
        assert not mlp.gate_bias_vl_loaded
        assert torch.count_nonzero(mlp.gate_bias_vl) == 0
        assert torch.equal(mlp.shared_w1.weight, tensors[ck + "ffn.shared_experts.w1.weight"])
        assert torch.equal(mlp.shared_w2.weight, tensors[ck + "ffn.shared_experts.w2.weight"])
        assert torch.equal(mlp.shared_w3.weight, tensors[ck + "ffn.shared_experts.w3.weight"])
        for local_slot in range(mlp.num_experts_per_rank):
            expert = mlp.start_expert_id + local_slot
            assert torch.equal(mlp.experts_w1[local_slot], tensors[ck + f"ffn.experts.{expert}.w1.weight"])
            assert torch.equal(mlp.experts_w2[local_slot], tensors[ck + f"ffn.experts.{expert}.w2.weight"])
            assert torch.equal(mlp.experts_w3[local_slot], tensors[ck + f"ffn.experts.{expert}.w3.weight"])


def test_checkpoint_loads_mtp_layout() -> None:
    torch.manual_seed(3)
    cfg = DeepseekV41DSparkConfig.from_dict(_config_dict())
    lm = DeepseekV41DSparkForCausalLM(_config_dict())
    # Each tensor is filled with a unique constant so a mapping slip cannot
    # silently pass by coincidental equality.
    counter = {"n": 0}

    def fill(key: str, shape: tuple[int, ...]) -> torch.Tensor:
        counter["n"] += 1
        return torch.full(shape, float(counter["n"]))

    tensors = _build_checkpoint(cfg, fill)
    # The synthetic checkpoint is exactly the documented mtp.* key set.
    assert set(tensors) == _draft_group_keys(cfg)
    _load(lm, tensors)
    _assert_loaded(lm, tensors)


def test_checkpoint_loads_shared_vocab_fallback() -> None:
    torch.manual_seed(4)
    cfg = DeepseekV41DSparkConfig.from_dict(_config_dict())
    lm = DeepseekV41DSparkForCausalLM(_config_dict())
    tensors = _build_checkpoint(cfg, lambda key, shape: torch.randn(shape), dedicated_vocab=False)
    _load(lm, tensors)
    assert torch.equal(lm.model.backbone.embed_tokens.weight, tensors["embed.weight"])
    assert lm.model.lm_head is not None
    assert torch.equal(lm.model.lm_head.weight, tensors["head.weight"])


def test_dedicated_vocab_source_wins_over_fallback() -> None:
    torch.manual_seed(5)
    cfg = DeepseekV41DSparkConfig.from_dict(_config_dict())
    lm = DeepseekV41DSparkForCausalLM(_config_dict())
    tensors = _build_checkpoint(cfg, lambda key, shape: torch.randn(shape))
    last = cfg.num_draft_layers - 1
    tensors["embed.weight"] = torch.full((cfg.vocab_size, cfg.hidden_size), 7.0)
    tensors["head.weight"] = torch.full((cfg.vocab_size, cfg.hidden_size), 9.0)
    _load(lm, tensors)
    assert torch.equal(lm.model.backbone.embed_tokens.weight, tensors["mtp.0.embed.weight"])
    assert lm.model.lm_head is not None
    assert torch.equal(lm.model.lm_head.weight, tensors[f"mtp.{last}.head.weight"])


def test_confidence_head_is_bias_less_in_the_checkpoint() -> None:
    torch.manual_seed(6)
    cfg = DeepseekV41DSparkConfig.from_dict(_config_dict())
    lm = DeepseekV41DSparkForCausalLM(_config_dict())
    assert lm.model.confidence_head is not None
    head = lm.model.confidence_head
    with torch.no_grad():
        head.proj.bias.fill_(0.25)  # module-init bias the checkpoint does not own
    tensors = _build_checkpoint(cfg, lambda key, shape: torch.randn(shape))
    _load(lm, tensors)
    assert torch.count_nonzero(head.proj.bias) == 0


def test_confidence_head_bias_loads_when_present() -> None:
    torch.manual_seed(7)
    cfg = DeepseekV41DSparkConfig.from_dict(_config_dict())
    lm = DeepseekV41DSparkForCausalLM(_config_dict())
    tensors = _build_checkpoint(cfg, lambda key, shape: torch.randn(shape), confidence_bias=True)
    _load(lm, tensors)
    assert lm.model.confidence_head is not None
    last = cfg.num_draft_layers - 1
    assert torch.equal(lm.model.confidence_head.proj.bias, tensors[f"mtp.{last}.confidence_head.proj.bias"])


def test_optional_hc_head_registers_and_collapses() -> None:
    torch.manual_seed(8)
    cfg = DeepseekV41DSparkConfig.from_dict(_config_dict())
    lm = DeepseekV41DSparkForCausalLM(_config_dict())
    tensors = _build_checkpoint(cfg, lambda key, shape: torch.randn(shape), hc_head=True)
    _load(lm, tensors)
    last = cfg.num_draft_layers - 1
    assert lm.model.backbone.hc_head_fn is not None
    assert torch.equal(lm.model.backbone.hc_head_fn, tensors[f"mtp.{last}.hc_head_fn"])
    assert torch.equal(lm.model.backbone.hc_head_scale, tensors[f"mtp.{last}.hc_head_scale"])

    hidden = torch.randn(3, cfg.hc_mult, cfg.hidden_size)
    flat = hidden.flatten(1).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
    mixes = flat @ lm.model.backbone.hc_head_fn.T * rsqrt
    pre = torch.sigmoid(mixes * lm.model.backbone.hc_head_scale + lm.model.backbone.hc_head_base) + cfg.hc_eps
    expected = (pre.unsqueeze(-1) * hidden.float()).sum(1)
    torch.testing.assert_close(lm.model.backbone._collapse_last_hidden(hidden, None), expected)


def test_no_hc_head_uses_the_last_block_pre_mix_collapse() -> None:
    torch.manual_seed(9)
    cfg = DeepseekV41DSparkConfig.from_dict(_config_dict())
    lm = DeepseekV41DSparkForCausalLM(_config_dict())
    tensors = _build_checkpoint(cfg, lambda key, shape: torch.randn(shape))
    _load(lm, tensors)
    assert lm.model.backbone.hc_head_fn is None
    hidden = torch.randn(2, cfg.hc_mult, cfg.hidden_size)
    pre_mix = torch.rand(2, cfg.hc_mult)
    expected = lm.model.backbone.layers[-1].hc.hc_pre(hidden, pre_mix)
    torch.testing.assert_close(lm.model.backbone._collapse_last_hidden(hidden, pre_mix), expected)


# ---------------------------------------------------------------------------
# Weight mapping (ascend W8A8 export)
# ---------------------------------------------------------------------------


def _write_quant_description(tmp_path, labels: dict[str, str]) -> None:
    (tmp_path / "quant_model_description.json").write_text(json.dumps(labels), encoding="utf-8")


def _build_w8a8_checkpoint(cfg: DeepseekV41DSparkConfig, *, main_proj_float: bool = False) -> dict[str, torch.Tensor]:
    """W8A8 tensors for the sharded projections (mirrors the Eco-Tech export).

    ``main_proj_float`` reproduces the official DSpark export, which keeps
    ``mtp.0.main_proj`` in BF16 with no scale/offset.
    """
    hidden = cfg.hidden_size
    q_lora = cfg.q_lora_rank
    head_dim = cfg.head_dim
    inter = _moe_intermediate(cfg)
    o_lora = cfg.o_lora_rank
    n_experts = cfg.dspark_n_routed_experts
    tensors: dict[str, torch.Tensor] = {}

    def quant(key: str, shape: tuple[int, int]) -> None:
        tensors[key + ".weight"] = torch.randint(-8, 8, shape, dtype=torch.int8)
        tensors[key + ".weight_scale"] = torch.rand(shape[0], 1)
        tensors[key + ".weight_offset"] = torch.zeros(shape[0], 1)

    for stage in range(cfg.num_draft_layers):
        ck = f"mtp.{stage}."
        for name, shape in _expected_shapes(cfg).items():
            if name.startswith("ffn.experts.") or name.startswith("ffn.shared_experts."):
                continue
            # Attention/norm/mHC tensors stay bf16 in the export (only the
            # projections and the experts are W8A8-quantized).
            tensors[ck + name] = torch.randn(shape)
        for expert in range(n_experts):
            for name, out, inp in (("w1", inter, hidden), ("w2", hidden, inter), ("w3", inter, hidden)):
                quant(ck + f"ffn.experts.{expert}.{name}", (out, inp))
        for name, out, inp in (("w1", inter, hidden), ("w2", hidden, inter), ("w3", inter, hidden)):
            quant(ck + f"ffn.shared_experts.{name}", (out, inp))
        quant(ck + "attn.wq_a", (q_lora, hidden))
        quant(ck + "attn.wq_b", (cfg.n_heads * head_dim, q_lora))
        quant(ck + "attn.wkv", (head_dim, hidden))
        # ``weight_offset`` for the W8A8-dynamic projections must stay zero.
        tensors[ck + "ffn.gate.bias_vl"] = torch.randn(n_experts)
    if main_proj_float:
        # Official DSpark export: BF16 main_proj, no weight_scale/weight_offset.
        tensors["mtp.0.main_proj.weight"] = torch.randn(hidden, 2 * hidden)
    else:
        tensors["mtp.0.main_proj.weight"] = torch.randint(-8, 8, (hidden, 2 * hidden), dtype=torch.int8)
        tensors["mtp.0.main_proj.weight_scale"] = torch.rand(hidden, 1)
        tensors["mtp.0.main_proj.weight_offset"] = torch.zeros(hidden, 1)
    tensors["mtp.0.main_norm.weight"] = torch.randn(hidden)
    # o_a/o_b stay bf16 in the export.
    for stage in range(cfg.num_draft_layers):
        ck = f"mtp.{stage}."
        tensors[ck + "attn.wo_a.weight"] = torch.randn(cfg.o_groups * o_lora, (cfg.n_heads * head_dim) // cfg.o_groups)
        tensors[ck + "attn.wo_b.weight"] = torch.randn(hidden, cfg.o_groups * o_lora)
    last = cfg.num_draft_layers - 1
    tensors["mtp.0.embed.weight"] = torch.randn(cfg.vocab_size, hidden)
    tensors[f"mtp.{last}.norm.weight"] = torch.randn(hidden)
    tensors[f"mtp.{last}.head.weight"] = torch.randn(cfg.vocab_size, hidden)
    tensors[f"mtp.{last}.markov_head.markov_w1.weight"] = torch.randn(cfg.vocab_size, cfg.dspark_markov_rank)
    tensors[f"mtp.{last}.markov_head.markov_w2.weight"] = torch.randn(cfg.vocab_size, cfg.dspark_markov_rank)
    tensors[f"mtp.{last}.confidence_head.proj.weight"] = torch.randn(1, hidden + cfg.dspark_markov_rank)
    return tensors


@pytest.mark.parametrize("tp_size", [1, 2])
def test_ascend_w8a8_mapping_shards_only_the_projections(
    monkeypatch: pytest.MonkeyPatch, tmp_path, tp_size: int
) -> None:
    _write_quant_description(tmp_path, {"mtp.0.main_proj.weight": "W8A8_DYNAMIC"})
    monkeypatch.setattr(kernels, "prepare_quant_weight", lambda weight: weight, raising=False)
    config = _config_dict(model_path=str(tmp_path), tp_size=tp_size)
    cfg = DeepseekV41DSparkConfig.from_dict(config)
    lm = DeepseekV41DSparkForCausalLM(config)
    assert lm.quant_mode == "ascend"
    tensors = _build_w8a8_checkpoint(cfg)
    lm.load_weights([_FakeStateDict(tensors)], 0, tp_size)

    hidden = cfg.hidden_size
    for stage in range(cfg.num_draft_layers):
        ck = f"mtp.{stage}."
        attn = lm.model.backbone.layers[stage].self_attn
        # Output-dim shard: rows are the rank's slice of the checkpoint tensor.
        rows = cfg.n_heads * cfg.head_dim // tp_size
        assert attn.q_b_proj.weight.shape == (rows, cfg.q_lora_rank)
        assert torch.equal(attn.q_b_proj.weight, tensors[ck + "attn.wq_b.weight"][:rows])
        assert attn.q_a_proj.weight.shape == (cfg.q_lora_rank, hidden)
        assert attn.kv_proj.weight.shape == (cfg.head_dim, hidden)
        assert attn.o_a_proj.weight.shape == (
            cfg.o_groups * cfg.o_lora_rank // tp_size,
            (cfg.n_heads * cfg.head_dim) // cfg.o_groups,
        )
        assert attn.o_b_proj.weight.shape == (hidden, cfg.o_groups * cfg.o_lora_rank // tp_size)
        assert attn.attn_sink.shape == (cfg.n_heads // tp_size,)
    assert lm.model.backbone.main_proj.w8a8 is not None
    assert torch.equal(lm.model.backbone.main_proj.w8a8.weight, tensors["mtp.0.main_proj.weight"][: hidden // tp_size])
    # The markov / confidence heads must stay replicated under TP.
    assert tuple(lm.model.markov_head.markov_w1.weight.shape) == (cfg.vocab_size, cfg.dspark_markov_rank)
    assert torch.equal(
        lm.model.markov_head.markov_w1.weight,
        tensors[f"mtp.{cfg.num_draft_layers - 1}.markov_head.markov_w1.weight"],
    )
    assert lm.model.confidence_head is not None
    assert tuple(lm.model.confidence_head.proj.weight.shape) == (
        1,
        cfg.hidden_size + cfg.dspark_markov_rank,
    )


def test_load_weights_rejects_a_mismatched_tp_size() -> None:
    cfg = DeepseekV41DSparkConfig.from_dict(_config_dict())
    lm = DeepseekV41DSparkForCausalLM(_config_dict())
    tensors = _build_checkpoint(cfg, lambda key, shape: torch.randn(shape))
    with pytest.raises(ValueError, match="tp_size"):
        lm.load_weights([_FakeStateDict(tensors)], 0, 2)


def test_ascend_load_fails_fast_on_an_unquantized_projection(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _write_quant_description(tmp_path, {"mtp.0.main_proj.weight": "W8A8_DYNAMIC"})
    config = _config_dict(model_path=str(tmp_path))
    cfg = DeepseekV41DSparkConfig.from_dict(config)
    lm = DeepseekV41DSparkForCausalLM(config)
    tensors = _build_w8a8_checkpoint(cfg)
    # A bf16 wq_a would otherwise be cast into the int8 parameter as garbage.
    tensors["mtp.0.attn.wq_a.weight"] = torch.randn(cfg.q_lora_rank, cfg.hidden_size)
    with pytest.raises(ValueError, match="expects int8 weights"):
        lm.load_weights([_FakeStateDict(tensors)], 0, 1)


def test_ascend_load_fails_fast_on_a_missing_quant_scale(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _write_quant_description(tmp_path, {"mtp.0.main_proj.weight": "W8A8_DYNAMIC"})
    config = _config_dict(model_path=str(tmp_path))
    cfg = DeepseekV41DSparkConfig.from_dict(config)
    lm = DeepseekV41DSparkForCausalLM(config)
    tensors = _build_w8a8_checkpoint(cfg)
    del tensors["mtp.0.main_proj.weight_scale"]
    with pytest.raises(KeyError, match="main_proj.weight_scale"):
        lm.load_weights([_FakeStateDict(tensors)], 0, 1)


def test_ascend_load_implies_a_zero_weight_offset(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _write_quant_description(tmp_path, {"mtp.0.main_proj.weight": "W8A8_DYNAMIC"})
    monkeypatch.setattr(kernels, "prepare_quant_weight", lambda weight: weight, raising=False)
    config = _config_dict(model_path=str(tmp_path))
    cfg = DeepseekV41DSparkConfig.from_dict(config)
    lm = DeepseekV41DSparkForCausalLM(config)
    tensors = _build_w8a8_checkpoint(cfg)
    del tensors["mtp.0.main_proj.weight_offset"]
    lm.load_weights([_FakeStateDict(tensors)], 0, 1)
    assert lm.model.backbone.main_proj.w8a8 is not None
    assert torch.count_nonzero(lm.model.backbone.main_proj.w8a8.weight_offset) == 0


# -- main_proj layout adapts to the checkpoint export ------------------------


@pytest.mark.parametrize("tp_size", [1, 2])
def test_ascend_main_proj_bf16_when_the_export_declares_float(
    monkeypatch: pytest.MonkeyPatch, tmp_path, tp_size: int
) -> None:
    """Official DSpark export: BF16 main_proj, everything else W8A8 (mixed)."""
    _write_quant_description(
        tmp_path,
        {
            "mtp.0.main_proj.weight": "FLOAT",
            "mtp.0.attn.wq_a.weight": "W8A8_DYNAMIC",
        },
    )
    monkeypatch.setattr(kernels, "prepare_quant_weight", lambda weight: weight, raising=False)
    config = _config_dict(model_path=str(tmp_path), tp_size=tp_size)
    cfg = DeepseekV41DSparkConfig.from_dict(config)
    lm = DeepseekV41DSparkForCausalLM(config)
    assert lm.quant_mode == "ascend"
    assert lm.model.backbone.main_proj.quantized is False
    assert lm.model.backbone.main_proj.w8a8 is None
    assert lm.model.backbone.main_proj.linear is not None
    tensors = _build_w8a8_checkpoint(cfg, main_proj_float=True)
    lm.load_weights([_FakeStateDict(tensors)], 0, tp_size)

    hidden = cfg.hidden_size
    # BF16 path shards the output dim like the W8A8 path did.
    assert torch.equal(
        lm.model.backbone.main_proj.linear.weight,
        tensors["mtp.0.main_proj.weight"][: hidden // tp_size],
    )
    assert torch.equal(lm.model.backbone.main_norm.weight, tensors["mtp.0.main_norm.weight"])
    # The projections the export does quantize keep the W8A8 path.
    assert lm.model.backbone.layers[0].self_attn.q_a_proj.weight.dtype == torch.int8
    assert torch.equal(
        lm.model.backbone.layers[0].self_attn.q_a_proj.weight,
        tensors["mtp.0.attn.wq_a.weight"],
    )


@pytest.mark.parametrize("label", ["FLOAT", None])
def test_ascend_main_proj_int8_without_a_quant_declaration_fails_fast(
    monkeypatch: pytest.MonkeyPatch, tmp_path, label: str | None
) -> None:
    """A float/absent declaration plus an int8 export must not silently cast."""
    labels = {} if label is None else {"mtp.0.main_proj.weight": label}
    _write_quant_description(tmp_path, labels)
    config = _config_dict(model_path=str(tmp_path))
    cfg = DeepseekV41DSparkConfig.from_dict(config)
    lm = DeepseekV41DSparkForCausalLM(config)
    tensors = _build_w8a8_checkpoint(cfg)
    with pytest.raises(ValueError, match="mtp.0.main_proj.weight.*int8.*unquantized"):
        lm.load_weights([_FakeStateDict(tensors)], 0, 1)


def test_ascend_main_proj_declared_w8a8_but_exported_bf16_fails_fast(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The opposite disagreement must fail with the tensor name and dtypes."""
    _write_quant_description(tmp_path, {"mtp.0.main_proj.weight": "W8A8_DYNAMIC"})
    config = _config_dict(model_path=str(tmp_path))
    cfg = DeepseekV41DSparkConfig.from_dict(config)
    lm = DeepseekV41DSparkForCausalLM(config)
    assert lm.model.backbone.main_proj.quantized is True
    tensors = _build_w8a8_checkpoint(cfg, main_proj_float=True)
    # Mirror the real export's BF16 storage rather than the tiny test dtype.
    tensors["mtp.0.main_proj.weight"] = tensors["mtp.0.main_proj.weight"].to(torch.bfloat16)
    with pytest.raises(ValueError, match="mtp.0.main_proj.weight.*bfloat16.*int8"):
        lm.load_weights([_FakeStateDict(tensors)], 0, 1)


def test_non_ascend_checkpoint_never_quantizes_main_proj() -> None:
    """The bf16/fp8 modes always build the dense main_proj (no metadata read)."""
    lm = DeepseekV41DSparkForCausalLM(_config_dict())
    assert lm.quant_mode != "ascend"
    assert lm.model.backbone.main_proj.quantized is False
    assert lm.model.backbone.main_proj.linear is not None
    assert lm.model.backbone.main_proj.w8a8 is None


def test_checkpoint_weight_is_quantized_reads_the_description(tmp_path) -> None:
    from xllm.python.models.deepseek_v41_dspark import checkpoint_weight_is_quantized

    _write_quant_description(tmp_path, {"mtp.0.main_proj.weight": "W8A8_DYNAMIC", "mtp.0.main_norm.weight": "FLOAT"})
    assert checkpoint_weight_is_quantized(str(tmp_path), "mtp.0.main_proj.weight") is True
    assert checkpoint_weight_is_quantized(str(tmp_path), "mtp.0.main_norm.weight") is False
    # Unknown / missing metadata is "undecided", never a silent guess.
    assert checkpoint_weight_is_quantized(str(tmp_path), "mtp.0.missing.weight") is None
    assert checkpoint_weight_is_quantized(str(tmp_path / "nope"), "mtp.0.main_proj.weight") is None
    assert checkpoint_weight_is_quantized(None, "mtp.0.main_proj.weight") is None


def test_markov_head_loads_the_production_export_names(tmp_path) -> None:
    """The export names the Markov matrices embed/head, not markov_w1/w2."""
    torch.manual_seed(21)
    cfg = DeepseekV41DSparkConfig.from_dict(_config_dict())
    lm = DeepseekV41DSparkForCausalLM(_config_dict())
    tensors = _build_checkpoint(cfg, lambda key, shape: torch.randn(shape))
    last = cfg.num_draft_layers - 1
    tensors[f"mtp.{last}.markov_head.embed.weight"] = tensors.pop(f"mtp.{last}.markov_head.markov_w1.weight")
    tensors[f"mtp.{last}.markov_head.head.weight"] = tensors.pop(f"mtp.{last}.markov_head.markov_w2.weight")
    _load(lm, tensors)
    assert torch.equal(lm.model.markov_head.markov_w1.weight, tensors[f"mtp.{last}.markov_head.embed.weight"])
    assert torch.equal(lm.model.markov_head.markov_w2.weight, tensors[f"mtp.{last}.markov_head.head.weight"])


# ---------------------------------------------------------------------------
# Context-KV pre-injection
# ---------------------------------------------------------------------------


def _reference_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    xf = x.float()
    var = xf.square().mean(-1, keepdim=True)
    return (weight * (xf * torch.rsqrt(var + eps))).to(dtype)


def test_write_context_kv_matches_eager_reference() -> None:
    torch.manual_seed(11)
    cfg = DeepseekV41DSparkConfig.from_dict(_config_dict())
    lm = DeepseekV41DSparkForCausalLM(_config_dict())
    tensors = _build_checkpoint(cfg, lambda key, shape: torch.randn(shape))
    _load(lm, tensors)
    model = lm.model.backbone

    target_hidden = torch.randn(3, cfg.main_proj_in_features)
    positions = torch.tensor([4, 5, 6])
    slots = torch.tensor([5, 0, 3])
    rows = 16
    caches = [torch.zeros(rows, cfg.head_dim) for _ in range(cfg.num_draft_layers)]
    projected = model.write_context_kv(target_hidden, positions, slots, caches, None)

    # Reference: main_proj -> main_norm, then per layer wkv -> kv_norm -> rope
    # (bf16, no QDQ), scattered at the requested slots.
    ref_projected = _reference_rms_norm(
        torch.nn.functional.linear(target_hidden, model.main_proj.linear.weight),
        model.main_norm.weight,
        cfg.rms_norm_eps,
    )
    torch.testing.assert_close(projected, ref_projected)
    for stage, cache in enumerate(caches):
        attn = model.layers[stage].self_attn
        kv = torch.nn.functional.linear(ref_projected, attn.kv_proj.weight)
        kv = _reference_rms_norm(kv, attn.kv_a_layernorm.weight, cfg.rms_norm_eps)
        kv = apply_interleaved_rope(kv, positions, model.rotary.cos_sin_cache)
        torch.testing.assert_close(cache[slots], kv)
        untouched = [row for row in range(rows) if row not in slots.tolist()]
        assert torch.count_nonzero(cache[untouched]) == 0


def test_write_context_kv_accepts_layer_cache_slots_and_tuple_form() -> None:
    torch.manual_seed(12)
    cfg = DeepseekV41DSparkConfig.from_dict(_config_dict())
    lm = DeepseekV41DSparkForCausalLM(_config_dict())
    _load(lm, _build_checkpoint(cfg, lambda key, shape: torch.randn(shape)))
    from xllm.python.attention.backend import LayerCache

    hidden = torch.randn(2, cfg.main_proj_in_features)
    positions = torch.tensor([1, 2])
    slots = torch.tensor([1, 2])
    bare = [torch.zeros(8, cfg.head_dim) for _ in range(cfg.num_draft_layers)]
    named = [LayerCache(key=None, value=None, swa=cache) for cache in bare]
    lm.model.backbone.write_context_kv(hidden, positions, slots, named, None)
    assert torch.count_nonzero(named[0].swa[slots]) > 0
    # A tuple form that does not reach the SWA slot (index 5 of the LayerCache
    # schema) must fail loudly instead of silently dropping the context rows.
    tuples = [(None, None, None, cache) for cache in bare]
    with pytest.raises(RuntimeError, match="missing the SWA pool"):
        lm.model.backbone.write_context_kv(hidden, positions, slots, tuples, None)


def test_write_context_kv_validates_cache_sizes() -> None:
    torch.manual_seed(13)
    cfg = DeepseekV41DSparkConfig.from_dict(_config_dict())
    lm = DeepseekV41DSparkForCausalLM(_config_dict())
    _load(lm, _build_checkpoint(cfg, lambda key, shape: torch.randn(shape)))
    hidden = torch.randn(2, cfg.main_proj_in_features)
    positions = torch.tensor([1, 2])
    slots = torch.tensor([1, 2])
    caches = [torch.zeros(8, cfg.head_dim) for _ in range(cfg.num_draft_layers)]
    with pytest.raises(ValueError, match="layer count mismatch"):
        lm.model.backbone.write_context_kv(hidden, positions, slots, caches[:-1], None)
    with pytest.raises(ValueError, match="slot count mismatch"):
        lm.model.backbone.write_context_kv(hidden, positions, slots[:1], caches, None)


class _StopAfterFirstLayer:
    def record_event(self, layer_id: int) -> bool:
        return layer_id == 0


def test_write_context_kv_returns_none_when_a_layer_event_fails() -> None:
    torch.manual_seed(14)
    cfg = DeepseekV41DSparkConfig.from_dict(_config_dict())
    lm = DeepseekV41DSparkForCausalLM(_config_dict())
    _load(lm, _build_checkpoint(cfg, lambda key, shape: torch.randn(shape)))
    hidden = torch.randn(2, cfg.main_proj_in_features)
    caches = [torch.zeros(8, cfg.head_dim) for _ in range(cfg.num_draft_layers)]
    assert (
        lm.model.backbone.write_context_kv(
            hidden, torch.tensor([1, 2]), torch.tensor([1, 2]), caches, _StopAfterFirstLayer()
        )
        is None
    )


# ---------------------------------------------------------------------------
# Eager draft block step
# ---------------------------------------------------------------------------


def test_anchor_first_query_block_places_noise_after_the_anchor() -> None:
    block = anchor_first_query_block(torch.tensor([11, 12]), block_size=3, noise_token_id=7)
    assert block.tolist() == [[11, 7, 7], [12, 7, 7]]
    with pytest.raises(ValueError, match="block_size"):
        anchor_first_query_block(torch.tensor([1]), block_size=0, noise_token_id=7)
    cfg = DeepseekV41DSparkConfig.from_dict(_config_dict())
    lm = DeepseekV41DSparkForCausalLM(_config_dict())
    assert lm.model.backbone.block_size == cfg.dspark_block_size
    assert lm.model.backbone.noise_token_id == cfg.dspark_noise_token_id
    assert lm.model.backbone.anchor_first_query_block(torch.tensor([5])).tolist() == [[5, 7, 7]]


def _draft_group_infos() -> list:
    """The draft backend's deduplicated cache groups (pure SWA -> exactly one)."""
    from xllm.python.attention.dsa_metadata import build_cache_specs_v41

    cfg = DeepseekV41DSparkConfig.from_dict(_config_dict())
    _, group_infos = build_cache_specs_v41([0] * cfg.num_draft_layers, [], [], cfg.window_size, cfg.num_draft_layers)
    return group_infos


class _DraftStubBackend:
    """Minimal CSA2 backend stub: one SWA group, records the manager limit."""

    def __init__(self, on_layer=None) -> None:
        self.group_infos = _draft_group_infos()
        self.group_limits: list[int | None] = []
        self._on_layer = on_layer

    def reset_forward(self, metadata) -> None:
        pass

    def prepare_dsa_metadata_for_forward(self, metadata, *, group_limit=None) -> None:
        self.group_limits.append(group_limit)

    def execute_csa2_layer(self, q, kv, layer) -> torch.Tensor:
        if self._on_layer is None:
            return q
        return self._on_layer(q, layer)


def test_draft_step_runs_the_block_in_one_parallel_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(15)
    cfg = DeepseekV41DSparkConfig.from_dict(_config_dict())
    lm = DeepseekV41DSparkForCausalLM(_config_dict())
    _load(lm, _build_checkpoint(cfg, lambda key, shape: torch.randn(shape)))

    calls: list[tuple[int, tuple[int, ...]]] = []

    def on_layer(q, layer):
        calls.append((layer.layer_id, tuple(q.shape)))
        return q * 0.5

    backend = _DraftStubBackend(on_layer)
    _with_patched_forward_context(monkeypatch, backend)
    block = lm.model.backbone.anchor_first_query_block(torch.tensor([5]))
    positions = torch.arange(cfg.dspark_block_size, dtype=torch.int64)
    out = lm.model.backbone(block.reshape(-1), positions)

    block_size = cfg.dspark_block_size
    assert out.shape == (block_size, cfg.hidden_size)
    assert torch.isfinite(out).all()
    assert backend.group_limits == []
    # One non-causal attention pass per draft layer over the whole block (a
    # serial per-token MTP walk would emit block_size calls per layer).
    head_dim = cfg.head_dim
    n_heads_local = cfg.n_heads // cfg.tp_size
    assert calls == [(stage, (block_size, n_heads_local, head_dim)) for stage in range(cfg.num_draft_layers)]


def test_draft_forward_uses_the_rotary_and_norm_head(monkeypatch: pytest.MonkeyPatch) -> None:
    """The head path is ``norm(collapse(last hidden))`` with the draft RoPE table."""
    torch.manual_seed(16)
    cfg = DeepseekV41DSparkConfig.from_dict(_config_dict())
    lm = DeepseekV41DSparkForCausalLM(_config_dict())
    _load(lm, _build_checkpoint(cfg, lambda key, shape: torch.randn(shape)))
    captured = {}

    _with_patched_forward_context(monkeypatch, _DraftStubBackend())
    original = lm.model.backbone._collapse_last_hidden

    def spy(hidden, pre_mix):
        captured["idden"] = hidden.detach().clone()
        captured["pre_mix"] = pre_mix
        return original(hidden, pre_mix)

    monkeypatch.setattr(lm.model.backbone, "_collapse_last_hidden", spy)
    block = lm.model.backbone.anchor_first_query_block(torch.tensor([5]))
    out = lm.model.backbone(block.reshape(-1), torch.arange(cfg.dspark_block_size, dtype=torch.int64))
    expected = lm.model.backbone.norm(original(captured["idden"], captured["pre_mix"]))
    torch.testing.assert_close(out, expected)
    assert captured["idden"].shape == (cfg.dspark_block_size, cfg.hc_mult, cfg.hidden_size)


# ---------------------------------------------------------------------------
# Draft cache-group contract (shared request metadata, own manager tables)
# ---------------------------------------------------------------------------


def test_draft_graph_query_rows_regroup_into_complete_blocks() -> None:
    assert _block_query_lens(20, 5) == [5, 5, 5, 5]
    with pytest.raises(RuntimeError, match="complete blocks"):
        _block_query_lens(16, 5)


@pytest.mark.parametrize("prev_ctx", [0, 1, 8, 11])
@pytest.mark.parametrize("width", [1, 3, 5])
def test_draft_window_includes_all_context_and_all_block_queries(prev_ctx: int, width: int) -> None:
    from xllm.python.attention.csa2_attention import _block_visible_idxs

    win = 8
    positions = torch.arange(-win, prev_ctx + width + 2)
    cols = torch.arange(positions.numel())
    actual = _block_visible_idxs(torch.tensor(prev_ctx), width, positions, cols, win)
    expected = [
        index if max(0, prev_ctx - win) <= position < prev_ctx + width else -1
        for index, position in enumerate(positions.tolist())
    ]
    assert actual.tolist() == [expected] * width


@pytest.mark.parametrize("graph", [False, True])
def test_draft_token_rows_are_grouped_before_attention(graph: bool) -> None:
    from xllm.python.attention.csa2_attention import Csa2AttentionBackend, CSA2SharedRuntime
    from xllm.python.model_executor.forward_context import ForwardContext, forward_context

    backend = object.__new__(Csa2AttentionBackend)
    query_layout = Csa2QueryLayout(3, token_rows=not graph, shared_window=True)
    backend._kv_caches = [SimpleNamespace(swa=None)]
    backend._resolve_cache_mapping = lambda *_: SimpleNamespace(ori_cache_idx=0)
    backend._shared_runtime = CSA2SharedRuntime
    dsa = SimpleNamespace(
        is_acl_graph=graph,
        seq_lens_q=torch.ones(6, dtype=torch.int32),
        seq_lens=torch.tensor([11, 11, 11, 17, 17, 17]),
        slot_mappings=[],
    )
    backend._current_forward_metadata = lambda: SimpleNamespace(dsa_metadata=dsa)
    mode = SimpleNamespace(compress_ratio=0, is_full=False, is_reindex=False, is_reuse=False)
    layer = SimpleNamespace(layer_id=0, mode=mode, _csa2_ctx=SimpleNamespace(x=torch.zeros(6, 4)))
    seen = {}

    def attention(
        q: torch.Tensor,
        kv: torch.Tensor,
        layer: SimpleNamespace,
        dsa: SimpleNamespace,
        q_lens: list[int],
        ctx_lens: torch.Tensor,
        topk: torch.Tensor | None,
        shared: CSA2SharedRuntime,
        cp_context: object | None,
    ) -> torch.Tensor:
        seen["q_lens"] = q_lens
        seen["ctx_lens"] = ctx_lens.tolist()
        return torch.zeros_like(q)

    backend._csa2_attention = attention
    context = ForwardContext(
        backend,
        torch.device("cpu"),
        SimpleNamespace(dsa_metadata=dsa),
        [],
        execution_contexts={Csa2QueryLayout: query_layout},
    )
    with forward_context(context):
        backend.execute_csa2_layer(torch.zeros(6, 1, 4), torch.zeros(6, 4), layer)
    assert seen == {"q_lens": [3, 3], "ctx_lens": [11, 17]}


@pytest.mark.parametrize("graph,token_rows", [(False, False), (True, False), (False, True)])
def test_draft_swa_tail_matches_explicit_context_plus_block(graph: bool, token_rows: bool) -> None:
    from xllm.python.attention.csa2_attention import Csa2AttentionBackend, CSA2SharedRuntime

    torch.manual_seed(37)
    width, win, dim = 3, 4, 2
    backend = object.__new__(Csa2AttentionBackend)
    backend.window_size, backend.scale = win, dim**-0.5
    query_layout = Csa2QueryLayout(width, token_rows=token_rows, shared_window=True)
    backend._smla_fallback_reason = lambda *_: "CPU oracle"
    backend._resolve_cache_mapping = lambda *_: SimpleNamespace(ori_cache_idx=0)
    history = torch.randn(2, win, dim)
    cache = torch.zeros(2, win, 1, dim)
    previous = [6, 9]
    for seq in range(2):
        for row in range(win):
            cache[seq, (previous[seq] - win + row) % win, 0] = history[seq, row]
    backend._kv_caches = [SimpleNamespace(swa=cache)]
    tables = torch.tensor([[0], [1]], dtype=torch.int32)
    if graph or token_rows:
        tables = tables.repeat_interleave(width, dim=0)
    dsa = SimpleNamespace(is_acl_graph=graph, block_tables=[[tables]])
    layer = SimpleNamespace(layer_id=0, mode=SimpleNamespace(compress_ratio=0), attn_sink=torch.zeros(1))
    q = torch.randn(2 * width, 1, dim)
    kv = torch.randn(2 * width, dim)
    actual = backend._csa2_attention(
        q, kv, layer, dsa, [width, width], torch.tensor([9, 12]), None, CSA2SharedRuntime(query_layout=query_layout)
    )
    expected = torch.empty_like(q)
    for seq in range(2):
        keys = torch.cat((history[seq], kv[seq * width : (seq + 1) * width]))
        for row in range(width):
            index = seq * width + row
            scores = keys @ q[index, 0] * backend.scale
            probabilities = torch.softmax(torch.cat((scores, torch.zeros(1))), dim=0)[:-1]
            expected[index, 0] = probabilities @ keys
    torch.testing.assert_close(actual, expected)


@pytest.fixture
def _draft_ring_attention() -> SimpleNamespace:
    from xllm.python.attention.csa2_attention import Csa2AttentionBackend, CSA2SharedRuntime

    width, win, dim, previous = 5, 4, 2, 11
    backend = object.__new__(Csa2AttentionBackend)
    backend.window_size, backend.scale = win, dim**-0.5
    backend._resolve_cache_mapping = lambda *_: SimpleNamespace(ori_cache_idx=0)
    backend._shared_runtime = CSA2SharedRuntime
    cache = torch.zeros(3, win, 1, dim)
    table = torch.tensor([[2, 1]], dtype=torch.int32)
    history = torch.tensor([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0]])
    for row, position in enumerate(range(previous - win, previous)):
        block = int(table[0, (position // win) % table.size(1)])
        cache[block, position % win, 0] = history[row]
    positions = torch.arange(previous, previous + width)
    slots = table[0, (positions // win) % table.size(1)] * win + positions % win
    dsa = SimpleNamespace(
        is_acl_graph=True,
        block_tables=[[table.repeat_interleave(width, dim=0)]],
        slot_mappings=[[slots]],
        seq_lens_q=torch.ones(width, dtype=torch.int32),
        seq_lens=torch.full((width,), previous + width, dtype=torch.int32),
    )
    backend._kv_caches = [SimpleNamespace(swa=cache)]
    backend._current_forward_metadata = lambda: SimpleNamespace(dsa_metadata=dsa)
    layer = SimpleNamespace(
        layer_id=0,
        mode=SimpleNamespace(compress_ratio=0, is_full=False, is_reindex=False, is_reuse=False),
        attn_sink=torch.zeros(1),
        _csa2_ctx=SimpleNamespace(x=torch.zeros(width, dim)),
    )
    kv = torch.zeros(width, dim)
    kv[-1] = torch.tensor([64.0, -32.0])
    return SimpleNamespace(
        backend=backend,
        cache=cache,
        dsa=dsa,
        layer=layer,
        q=torch.zeros(width, 1, dim),
        kv=kv,
        history=history,
        slots=slots,
        query_layout=Csa2QueryLayout(width, shared_window=True),
        width=width,
        win=win,
        previous=previous,
    )


def test_draft_swa_tail_keeps_final_block_row_after_ring_wrap(_draft_ring_attention: SimpleNamespace) -> None:
    from xllm.python.attention.csa2_attention import CSA2SharedRuntime

    case = _draft_ring_attention
    actual = case.backend._csa2_attention(
        case.q,
        case.kv,
        case.layer,
        case.dsa,
        [case.width],
        torch.tensor([case.previous + case.width]),
        None,
        CSA2SharedRuntime(query_layout=case.query_layout),
    )

    # Zero queries and a zero sink give every history/block row equal weight.
    # The final block row must also contribute to the first non-causal query.
    keys = torch.cat((case.history, case.kv))
    expected = keys.sum(dim=0) / (case.win + case.width + 1)
    torch.testing.assert_close(actual, expected.view(1, 1, -1).expand_as(actual))


def test_draft_graph_defers_ring_writes_without_dynamic_filter(
    _draft_ring_attention: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xllm.python.attention import csa2_attention as csa2
    from xllm.python.model_executor.forward_context import ForwardContext, forward_context

    case = _draft_ring_attention
    events: list[str] = []
    attention = case.backend._csa2_attention
    scatter = csa2._scatter_by_slot

    def record_attention(*args: object, **kwargs: object) -> torch.Tensor:
        events.append("attention")
        return attention(*args, **kwargs)

    def record_scatter(cache: torch.Tensor, slots: torch.Tensor, values: torch.Tensor) -> None:
        events.append("scatter")
        assert slots is case.slots
        assert values is case.kv
        scatter(cache, slots, values)

    def reject_dynamic_filter(slots: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pytest.fail("graph block writes must not use data-dependent ring deduplication")

    monkeypatch.setattr(case.backend, "_csa2_attention", record_attention)
    monkeypatch.setattr(csa2, "_scatter_by_slot", record_scatter)
    monkeypatch.setattr(csa2, "last_ring_writes", reject_dynamic_filter)
    context = ForwardContext(
        case.backend,
        torch.device("cpu"),
        SimpleNamespace(dsa_metadata=case.dsa),
        [],
        execution_contexts={Csa2QueryLayout: case.query_layout},
    )
    with forward_context(context):
        actual = case.backend.execute_csa2_layer(case.q, case.kv, case.layer)

    # The last new row overwrites an old history slot: writing before attention
    # would change this oracle even if the final cache contents looked correct.
    keys = torch.cat((case.history, case.kv))
    expected = keys.sum(dim=0) / (case.win + case.width + 1)
    torch.testing.assert_close(actual, expected.view(1, 1, -1).expand_as(actual))
    assert events == ["attention", "scatter"]
    torch.testing.assert_close(case.cache.view(-1, case.kv.size(-1))[case.slots.long()], case.kv)


@pytest.mark.parametrize("persistent", [False, True])
def test_registered_block_metadata_preserves_attention_and_write_order(
    _draft_ring_attention: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    persistent: bool,
) -> None:
    from xllm.python.attention import csa2_attention as csa2
    from xllm.python.model_executor.input_batch import InputBatch

    case = _draft_ring_attention
    positions = torch.arange(case.previous, case.previous + case.width)
    batch = InputBatch.from_runtime(
        torch.arange(case.width),
        positions,
        SimpleNamespace(
            num_reqs=1,
            num_tokens=case.width,
            num_scheduled_tokens=[case.width],
            num_computed_tokens=[],
            query_start_loc=[0, case.width],
            is_prefilling=[False],
        ),
        is_dummy=False,
    )
    builder = Csa2BlockAttentionMetadataBuilder(case.width, case.win, 32)
    source = SimpleNamespace(multi_block_tables=(case.dsa.block_tables[0][0], object()))
    if persistent:
        metadata = builder.allocate_persistent(batch, source)
        builder.update_persistent(metadata, batch, source)
    else:
        metadata = builder.build(batch, source)

    events: list[str] = []
    attention, scatter = case.backend._csa2_attention, csa2._scatter_by_slot

    def record_attention(*args: object, **kwargs: object) -> torch.Tensor:
        events.append("attention")
        return attention(*args, **kwargs)

    def record_scatter(cache: torch.Tensor, slots: torch.Tensor, values: torch.Tensor) -> None:
        events.append("scatter")
        scatter(cache, slots, values)

    def reject_legacy_preparation(*args: object, **kwargs: object) -> None:
        pytest.fail("registered block metadata must not use legacy DSA preparation or dynamic write filtering")

    monkeypatch.setattr(case.backend, "_csa2_attention", record_attention)
    monkeypatch.setattr(case.backend, "_current_forward_metadata", reject_legacy_preparation)
    monkeypatch.setattr(csa2, "_scatter_by_slot", record_scatter)
    monkeypatch.setattr(csa2, "last_ring_writes", reject_legacy_preparation)
    context = ForwardContext(
        case.backend,
        torch.device("cpu"),
        SimpleNamespace(),
        [],
        execution_contexts={Csa2BlockAttentionMetadata: metadata},
    )
    with forward_context(context):
        actual = case.backend.execute_csa2_layer(case.q, case.kv, case.layer)
    keys = torch.cat((case.history, case.kv))
    expected = keys.sum(dim=0) / (case.win + case.width + 1)
    torch.testing.assert_close(actual, expected.view(1, 1, -1).expand_as(actual))
    assert events == ["attention", "scatter"]
    torch.testing.assert_close(case.cache.view(-1, case.kv.size(-1))[metadata.slot_mapping.long()], case.kv)


def test_registered_dummy_block_does_not_read_or_modify_live_cache(_draft_ring_attention: SimpleNamespace) -> None:
    from xllm.python.model_executor.input_batch import InputBatch

    case = _draft_ring_attention
    batch = InputBatch.from_runtime(
        torch.zeros(case.width, dtype=torch.int64),
        torch.zeros(case.width, dtype=torch.int32),
        SimpleNamespace(
            num_reqs=0,
            num_tokens=case.width,
            num_scheduled_tokens=[],
            num_computed_tokens=[],
            query_start_loc=[0],
            is_prefilling=[],
        ),
        is_dummy=True,
    )
    builder = Csa2BlockAttentionMetadataBuilder(case.width, case.win, 32)
    metadata = builder.allocate_persistent(batch, SimpleNamespace())
    builder.update_persistent(metadata, batch, SimpleNamespace())
    case.cache[0].fill_(float("nan"))
    live_cache = case.cache[1:].clone()
    context = ForwardContext(
        case.backend,
        torch.device("cpu"),
        SimpleNamespace(),
        [],
        execution_contexts={Csa2BlockAttentionMetadata: metadata},
    )
    with forward_context(context):
        output = case.backend.execute_csa2_layer(case.q, case.kv, case.layer)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(case.cache[1:], live_cache)


def test_pipeline_checkpoint_without_own_head_can_share_target_head() -> None:
    model = DeepseekV41DSparkForCausalLM(_config_dict())
    tensors = _build_checkpoint(model.cfg, lambda key, shape: torch.randn(shape))
    del tensors[f"mtp.{model.cfg.num_draft_layers - 1}.head.weight"]
    _load(model, tensors)
    assert model.model.lm_head is None
    head = torch.nn.Linear(model.cfg.hidden_size, model.cfg.vocab_size, bias=False)
    model.share_weights_from(SimpleNamespace(get_lm_head=lambda: head))
    assert model.model.lm_head is head


def test_pipeline_matches_sequential_greedy_markov(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(29)
    model = DeepseekV41DSparkForCausalLM(_config_dict())
    _load(model, _build_checkpoint(model.cfg, lambda key, shape: torch.randn(shape)))
    assert isinstance(model.model, DSparkPipeline)
    assert ModelSamplingMetadataBuilder in get_execution_metadata_builder_classes("deepseek_v41_dspark")
    assert Csa2BlockAttentionMetadataBuilder in get_execution_metadata_builder_classes("deepseek_v41_dspark")
    hidden = torch.randn(6, model.cfg.hidden_size)
    monkeypatch.setattr(model.model.backbone, "forward", lambda *args: hidden)
    sampling = SimpleNamespace(
        is_unconstrained_greedy=True,
        return_probs=False,
        return_confidence=False,
        parameters=SimpleNamespace(all_greedy_sample=True),
    )
    runtime = ModuleType("xllm_runtime")
    runtime.ModelSamplingInput = lambda value: value
    runtime.sample_logits = lambda logits, sampling_input: (logits.argmax(-1), None)
    monkeypatch.setitem(sys.modules, "xllm_runtime", runtime)
    input_ids = torch.tensor([2, 7, 7, 4, 7, 7], dtype=torch.int64)
    positions = torch.arange(6, dtype=torch.int64)
    context = ForwardContext(
        _DraftStubBackend(),
        torch.device("cpu"),
        SimpleNamespace(),
        [],
        execution_contexts={ModelSamplingMetadata: ModelSamplingMetadata(sampling, 6)},
    )
    with forward_context(context):
        output = model.model(input_ids, positions)
    base_logits = model.model.lm_head(hidden).view(2, 3, model.cfg.vocab_size)
    previous = input_ids.view(2, 3)[:, 0]
    expected_steps = []
    for token_index in range(3):
        previous = torch.argmax(base_logits[:, token_index] + model.model.markov_head.bias(previous), dim=-1)
        expected_steps.append(previous)
    assert output.draft_proposal is not None
    torch.testing.assert_close(output.draft_proposal.token_ids, torch.stack(expected_steps, dim=1))
    assert output.hidden_states is None


@pytest.mark.parametrize("cp_size", [2, 4])
def test_pipeline_rejects_context_parallelism_at_initialization(cp_size: int) -> None:
    with pytest.raises(ValueError, match="CP=1"):
        DeepseekV41DSparkForCausalLM({**_config_dict(), "cp_size": cp_size})


def test_draft_group_infos_is_a_single_swa_group() -> None:
    """The draft cache spec is pure SWA: exactly one deduplicated group."""
    from xllm.python.attention.dsa_metadata import DSA_CACHE_SLIDING_WINDOW

    group_infos = _draft_group_infos()
    assert len(group_infos) == 1
    assert group_infos[0].cache_type == DSA_CACHE_SLIDING_WINDOW
