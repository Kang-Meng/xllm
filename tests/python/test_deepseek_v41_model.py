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

"""DeepSeek-V4.1 Python model (``deepseek_v41.py``): config parsing, CSA2 plan,
structure, weight loading, forward path, the V4.1 quant kernels inlined in
deepseek_v41 (pure torch; the quant helpers are device-agnostic), and the engram
model integration (decoder routing, graph staging/replay, reconcile, gate).
Pure ``EngramEmbedding`` component tests live in ``test_deepseek_v41_engram.py``.

The forward tests drive :class:`DeepseekV41ForCausalLM` through
:class:`ModelExecutor` with the CSA2 attention backend and synthetic cache /
metadata structures. V4.1 is NPU-only (the backend has no CPU/CUDA path), so
those tests are marked ``npu_only`` and skip elsewhere.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tests.python._engram_reference import _reference_engram_lookup
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.csa2_attention import (
    Csa2AttentionBackend,
    Csa2LayerContext,
    CSA2SharedRuntime,
    _derive_graph_token_capacity_granularity,
    _indexer_score,
    _indexer_topk,
    _select_candidate_blocks,
    _window_visible_idxs,
    apply_interleaved_rope,
)
from xllm.python.attention.dsa_metadata import (
    DSA_CACHE_SLIDING_WINDOW,
    DSA_CACHE_TOKEN,
    build_cache_specs_v41,
)
from xllm.python.model_executor.executor import ModelExecutor
from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
)
from xllm.python.models import deepseek_v41 as dq
from xllm.python.models.deepseek_v41 import (
    CSA2_MODE_FULL,
    CSA2_MODE_REINDEX,
    CSA2_MODE_REUSE,
    CSA2_MODE_SWA,
    V41_QUANT_ASCEND,
    V41_QUANT_FP8,
    V41_QUANT_NONE,
    CSA2LayerMode,
    CSA2LayerPlan,
    DeepseekV41Config,
    DeepseekV41DecoderLayer,
    DeepseekV41ForCausalLM,
    DeepseekV41Model,
    DeepseekV41MoE,
    fp4_act_qdq_e4m3_scale,
    fp4_act_qdq_e8m0,
)
from xllm.python.models.deepseek_v41_engram import (
    ENGRAM_IMAGE_SENTINEL_BASE_ID,
    ENGRAM_STORAGE_INT8,
    Engram,
    EngramEmbedding,
    EngramLayout,
    load_quarot_gate_unrotate,
)
from xllm.python.registry import get_model_class


# DeepSeek-V4.1 is NPU-only: ModelExecutor binds the CSA2 attention backend,
# which has no CPU/CUDA implementation. Tests that drive the executor skip on
# hosts without an NPU instead of monkeypatching a backend in.
def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


npu_only = pytest.mark.skipif(
    not _npu_available(),
    reason="DeepSeek-V4.1 requires an NPU attention backend",
)


def _active_device() -> torch.device:
    """NPU when present (the only supported V4.1 platform); CPU for skip-guarded runs."""
    return torch.device("npu:0") if _npu_available() else torch.device("cpu")


# Reference V4.1 Flash config fields (adaptation report 1.4), with the full
# 43-entry compress_ratios layout: [0,0] + [2]*18 + [1]*20 + draft tail [0,0,0].
_DSV41_CONFIG = {
    "model_type": "deepseek_v41",
    "architectures": ["DeepseekV41ForCausalLM"],
    "hidden_size": 5120,
    "num_hidden_layers": 40,
    "num_attention_heads": 64,
    "head_dim": 512,
    "vocab_size": 129280,
    "rms_norm_eps": 1e-20,
    "rope_theta": 10000.0,
    "max_position_embeddings": 1048576,
    "rope_scaling": {"beta_fast": 32, "beta_slow": 1, "factor": 16, "type": "yarn"},
    "q_lora_rank": 1280,
    "qk_rope_head_dim": 64,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
    "compress_rope_theta": 160000.0,
    "window_size": 128,
    "sliding_window": 128,
    "index_head_dim": 128,
    "index_n_heads": 32,
    "index_topk": 512,
    "kv_source_layer_ids": [2, 8, 14, 20],
    "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
    "candidate_source_layer_id": 20,
    "candidate_topk_blocks": 2048,
    "candidate_block_size": 8,
    "engram_layer_ids": [1, 14],
    "engram_num_embeddings": [384006168, 384016682],
    "engram_max_ngram_size": 4,
    "engram_vocab_size": 16000000,
    "engram_n_heads": 8,
    "engram_head_dim": 256,
    "engram_compressed_vocab_size": 99092,
    "engram_pad_token_id": 2,
    "num_nextn_predict_layers": 3,
    "dspark_block_size": 5,
    "dspark_noise_token_id": 128799,
    "dspark_target_layer_ids": [37, 38, 39],
    "dspark_markov_rank": 256,
    "dspark_n_routed_experts": 128,
    "dspark_num_experts_per_tok": 3,
    "n_activated_experts": 6,
    "n_routed_experts": 384,
    "moe_intermediate_size": 2304,
    "n_shared_experts": 1,
    "scoring_func": "sqrtsoftplus",
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "swiglu_limit": 10.0,
    "image_token_id": 129264,
    "first_k_dense_replace": 0,
    "tie_word_embeddings": False,
}


# Small geometry for construction / weight-loading tests (6 layers: 0-1 SWA,
# 2-3 encoder CSA2 with Full at 2, 4-5 decoder with Full at 4).
#
# Engram is shrunk to a testable bucket layout: with engram_vocab_size 64 the
# layout draws its primes from 63 upward (67, 71, 73, 79 -> 290 rows), the
# injected token_map covers the 64-id vocabulary and compresses onto 16 ids,
# and layer 1 alone carries the module.
def _engram_test_token_map() -> list[int]:
    return (torch.arange(64) % 16).tolist()


_ENGRAM_TINY_OVERRIDE = {
    "engram_layer_ids": [1],
    "engram_num_embeddings": [290],
    "engram_max_ngram_size": 3,
    "engram_vocab_size": 64,
    "engram_n_heads": 2,
    "engram_head_dim": 32,
    "engram_compressed_vocab_size": 16,
    "engram_pad_token_id": 0,
    "engram_token_map": _engram_test_token_map(),
}

_DSV41_TINY_CONFIG = {
    **_DSV41_CONFIG,
    **_ENGRAM_TINY_OVERRIDE,
    "num_hidden_layers": 6,
    "hidden_size": 128,
    "num_attention_heads": 4,
    "head_dim": 32,
    "q_lora_rank": 32,
    "o_lora_rank": 16,
    "o_groups": 2,
    "qk_rope_head_dim": 8,
    "vocab_size": 64,
    "n_routed_experts": 4,
    "n_activated_experts": 2,
    "moe_intermediate_size": 32,
    "index_n_heads": 2,
    "index_head_dim": 16,
    "index_topk": 4,
    "compress_ratios": [0, 0, 2, 2, 1, 1],
    "kv_source_layer_ids": [2, 4],
    "index_source_layer_ids": [2, 4],
}

# Forward-path geometry (contract section 5): head_dim 64 and index_head_dim 32
# satisfy the QDQ group divisibility; index sources [2, 4, 5] make layer 5 a
# Reindex layer consuming layer 4's candidate pool.
_DSV41_FORWARD_CONFIG = {
    **_DSV41_TINY_CONFIG,
    "num_attention_heads": 2,
    "head_dim": 64,
    "qk_rope_head_dim": 16,
    "index_head_dim": 32,
    "max_position_embeddings": 64,
    "window_size": 8,
    "sliding_window": 8,
    "index_source_layer_ids": [2, 4, 5],
    "candidate_source_layer_id": 4,
    "candidate_topk_blocks": 2,
    "candidate_block_size": 4,
}


def test_config_from_dict_and_defaults() -> None:
    cfg = DeepseekV41Config.from_dict(_DSV41_CONFIG)
    assert cfg.model_type == "deepseek_v41"
    assert cfg.n_layers == 40
    assert cfg.q_lora_rank == 1280
    assert cfg.n_routed_experts == 384
    assert cfg.kv_source_layer_ids == [2, 8, 14, 20]
    assert cfg.index_source_layer_ids == [2, 8, 14, 20, 24, 28, 32, 36]
    assert cfg.candidate_source_layer_id == 20
    assert cfg.engram_layer_ids == [1, 14]
    assert cfg.engram_pad_token_id == 2
    assert cfg.image_token_id == 129264
    # Sparse dict: V4.1 defaults, the n_layers alias, no fabricated cache
    # groups and no hash routing.
    sparse = DeepseekV41Config.from_dict({"model_type": "deepseek_v41", "n_layers": 6})
    assert sparse.n_layers == 6
    assert sparse.rms_norm_eps == 1e-20
    assert sparse.compress_ratios == [0] * 6
    assert sparse.n_hash_layers == 0


def test_config_preserves_v41_compress_ratios() -> None:
    """Ratio 0 (SWA), 1 (decoder) and 2 (encoder CSA2) stay distinct.

    V4's ``<= 1 -> 1`` normalization would erase the SWA/decoder distinction;
    the trailing draft entries (3 zeros) are dropped for the 40-layer model.
    """
    cfg = DeepseekV41Config.from_dict(_DSV41_CONFIG)
    assert cfg.compress_ratios == [0, 0] + [2] * 18 + [1] * 20
    assert len(cfg.compress_ratios) == cfg.n_layers


@pytest.mark.parametrize("ratio", [-1, 3, 8, 128])
def test_config_rejects_unsupported_compress_ratios(ratio: int) -> None:
    with pytest.raises(ValueError, match="unsupported DeepSeek-V4.1 compression ratio"):
        DeepseekV41Config.from_dict({**_DSV41_CONFIG, "compress_ratios": [ratio] * 40})


# ---------------------------------------------------------------------------
# CSA2 layer plan (semantics doc section 3)
# ---------------------------------------------------------------------------


def test_csa2_plan_matches_reference_layer_assignment() -> None:
    cfg = DeepseekV41Config.from_dict(_DSV41_CONFIG)
    plan = CSA2LayerPlan.build(cfg.n_layers, cfg.compress_ratios, cfg.kv_source_layer_ids, cfg.index_source_layer_ids)
    # No cross-layer CED hidden capture exists in the reference: every
    # kv-source layer compresses its OWN attention input.
    full_layers = {2, 8, 14, 20}
    reindex_layers = {24, 28, 32, 36}
    encoder_group_source = {
        **{layer: 2 for layer in range(2, 8)},
        **{layer: 8 for layer in range(8, 14)},
        **{layer: 14 for layer in range(14, 20)},
        **{layer: 20 for layer in range(20, 40)},
    }
    for layer_id in range(cfg.n_layers):
        mode = plan.mode(layer_id)
        if layer_id in (0, 1):
            assert mode.mode == CSA2_MODE_SWA
            assert mode.compress_ratio == 0
        elif layer_id in full_layers:
            assert mode.mode == CSA2_MODE_FULL
            assert mode.main_kv_source_layer == layer_id
        elif layer_id in reindex_layers:
            assert mode.mode == CSA2_MODE_REINDEX
            assert mode.main_kv_source_layer == 20
        else:
            assert mode.mode == CSA2_MODE_REUSE
            assert mode.main_kv_source_layer == encoder_group_source[layer_id]
        if layer_id >= 2:
            assert mode.compress_ratio == (2 if layer_id < 20 else 1)


@pytest.mark.parametrize(
    ("n_layers", "ratios", "kv_sources", "index_sources"),
    [
        (40, [1] * 40, [45], [45, 2]),  # kv source out of range
        (40, [1] * 40, [2], [8]),  # kv source not an index source
        (40, [1] * 40, [8], [2, 8]),  # index source before any kv source
        (40, [0] * 40, [2], [2, 8]),  # zero ratio on a kv source
    ],
)
def test_csa2_plan_rejects_invalid_source_layouts(
    n_layers: int,
    ratios: list[int],
    kv_sources: list[int],
    index_sources: list[int],
) -> None:
    with pytest.raises(ValueError):
        CSA2LayerPlan.build(n_layers, ratios, kv_sources, index_sources)


# ---------------------------------------------------------------------------
# Module layout / structure (semantics doc sections 2.6, 5, 6)
# ---------------------------------------------------------------------------


def test_dsa_cp_replicates_query_weights_and_keeps_output_sharded() -> None:
    config = {**_DSV41_TINY_CONFIG, "enable_dsa_cp": True, "tp_size": 2, "tp_rank": 0, "engram_layer_ids": []}
    cfg = DeepseekV41Config.from_dict(config)
    model = DeepseekV41Model(cfg, torch.float32, torch.device("cpu"), config=config)
    attn = model.layers[0].self_attn
    assert attn.enable_dsa_cp
    assert attn.q_b_proj.weight.shape[0] == cfg.n_heads * cfg.head_dim
    assert attn.attn_sink.numel() == cfg.n_heads
    assert attn.n_local_groups == cfg.o_groups // cfg.tp_size
    assert attn.o_b_proj.weight.shape[1] == cfg.o_groups * cfg.o_lora_rank // cfg.tp_size


def test_dsa_cp_rejects_pcp_topology() -> None:
    config = {**_DSV41_TINY_CONFIG, "enable_dsa_cp": True, "cp_size": 2, "engram_layer_ids": []}
    cfg = DeepseekV41Config.from_dict(config)
    with pytest.raises(ValueError, match="enable_dsa_cp cannot be combined with cp_size"):
        DeepseekV41Model(cfg, torch.float32, torch.device("cpu"), config=config)


def test_dsa_cp_rejects_uneven_head_shards_at_construction() -> None:
    config = {
        **_DSV41_TINY_CONFIG,
        "enable_dsa_cp": True,
        "tp_size": 2,
        "num_attention_heads": 3,
        "engram_layer_ids": [],
    }
    cfg = DeepseekV41Config.from_dict(config)
    with pytest.raises(ValueError, match="n_heads divisible by tp_size"):
        DeepseekV41Model(cfg, torch.float32, torch.device("cpu"), config=config)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("batch_kind", ["prefill", "chunked_prefill", "decode", "decode_graph", "mixed", "dummy"])
def test_dsa_cp_all_batch_types_match_original_tp(monkeypatch: pytest.MonkeyPatch, rank: int, batch_kind: str) -> None:
    from xllm.python import distributed
    from xllm.python.model_executor.dsa_cp import get_current_dsa_cp_runtime

    config = {**_DSV41_TINY_CONFIG, "tp_size": 2, "tp_rank": rank, "engram_layer_ids": []}
    enabled_config = {**config, "enable_dsa_cp": True}
    enabled_model = DeepseekV41Model(
        DeepseekV41Config.from_dict(enabled_config), torch.float32, torch.device("cpu"), config=enabled_config
    )
    baseline_model = DeepseekV41Model(
        DeepseekV41Config.from_dict(config), torch.float32, torch.device("cpu"), config=config
    )
    enabled = enabled_model.layers[0].self_attn
    baseline = baseline_model.layers[0].self_attn
    torch.manual_seed(42)
    with torch.no_grad():
        for parameter in enabled.parameters():
            parameter.uniform_(-0.1, 0.1)
        baseline_state = dict(enabled.state_dict())
        head_width = baseline.q_b_proj.weight.size(0)
        baseline_state["q_b_proj.weight"] = enabled.q_b_proj.weight[rank * head_width : (rank + 1) * head_width]
        heads = baseline.attn_sink.numel()
        baseline_state["attn_sink"] = enabled.attn_sink[rank * heads : (rank + 1) * heads]
        baseline.load_state_dict(baseline_state)

    q_lens = [2, 1] if batch_kind in ("mixed", "prefill", "chunked_prefill") else [1]
    tokens = sum(q_lens)
    backend = object.__new__(Csa2AttentionBackend)
    backend.window_size, backend.scale = 2, enabled.head_dim**-0.5
    backend._kv_caches = [SimpleNamespace(swa=None)]
    backend._resolve_cache_mapping = lambda *_: SimpleNamespace(ori_cache_idx=0)
    backend._current_forward_metadata = lambda: context.metadata
    backend._shared_runtime = CSA2SharedRuntime
    backend._smla_fallback_reason = lambda *_: "switch off"
    reference = None
    exchange_calls = 0

    def _execute(q: torch.Tensor, kv: torch.Tensor, layer: dq.DeepseekV41Attention) -> torch.Tensor:
        nonlocal reference
        runtime = get_current_dsa_cp_runtime()
        if layer is enabled:
            assert runtime is not None
            assert q.shape == (runtime.plan.end - runtime.plan.start, enabled.cfg.n_heads, enabled.head_dim)
            ctx = layer._csa2_ctx
            full_q = layer.q_b_proj(ctx.qr).view(tokens, enabled.cfg.n_heads, enabled.head_dim)
            full_q = apply_interleaved_rope(full_q, ctx.positions, ctx.cos_sin_cache)
            reference = backend._csa2_attention(
                full_q,
                kv,
                layer,
                context.metadata.dsa_metadata,
                q_lens,
                context.metadata.dsa_metadata.seq_lens,
                None,
                CSA2SharedRuntime(),
            )
            actual = backend.execute_csa2_layer(q, kv, layer)
            torch.testing.assert_close(actual, runtime.plan.shard(reference))
            return actual
        assert runtime is None
        assert q.shape[1] == heads
        return backend.execute_csa2_layer(q, kv, layer)

    def _exchange(send: torch.Tensor, world_size: int, group_name: str) -> torch.Tensor:
        nonlocal exchange_calls
        exchange_calls += 1
        runtime = get_current_dsa_cp_runtime()
        assert world_size == 2 and group_name == "tp" and reference is not None
        plan = runtime.plan
        padded = reference.new_zeros((plan.tokens_per_rank, enabled.cfg.n_heads, enabled.head_dim))
        padded[: plan.end - plan.start] = plan.shard(reference)
        expected = padded.view(plan.tokens_per_rank, 2, heads, enabled.head_dim).permute(1, 0, 2, 3)
        torch.testing.assert_close(send, expected.reshape(-1, heads, enabled.head_dim))
        received = reference.new_zeros((plan.tokens_per_rank * 2, heads, enabled.head_dim))
        received[:tokens] = reference[:, rank * heads : (rank + 1) * heads]
        return received

    monkeypatch.setattr(dq, "fp8_act_qdq", lambda x, *_: x)
    monkeypatch.setattr(distributed, "all_to_all", _exchange, raising=False)
    monkeypatch.setattr(distributed, "tp_all_reduce", lambda x: x)
    metadata = SimpleNamespace(
        is_prefill=batch_kind in ("prefill", "dummy"),
        is_chunked_prefill=batch_kind in ("mixed", "chunked_prefill"),
        is_mixed=batch_kind == "mixed",
        is_dummy=batch_kind == "dummy",
        dsa_graph_mode=batch_kind == "decode_graph",
        dsa_metadata=SimpleNamespace(
            seq_lens_q=torch.tensor(q_lens),
            seq_lens=torch.tensor(q_lens) + 7,
            block_tables=[],
            slot_mappings=[],
            is_acl_graph=batch_kind == "decode_graph",
        ),
    )
    hidden, positions = torch.randn(tokens, enabled.cfg.hidden_size), torch.arange(tokens)
    cos_sin_cache = torch.cat([torch.ones(tokens, 4), torch.zeros(tokens, 4)], dim=-1)
    context = ForwardContext(SimpleNamespace(execute_csa2_layer=_execute), torch.device("cpu"), metadata, [])
    with torch.no_grad(), forward_context(context):
        expected = baseline(hidden, positions, cos_sin_cache)
    context = ForwardContext(SimpleNamespace(execute_csa2_layer=_execute), torch.device("cpu"), metadata, [])
    with torch.no_grad(), forward_context(context):
        torch.testing.assert_close(enabled(hidden, positions, cos_sin_cache), expected)
    assert exchange_calls == 1


def _tiny_model() -> DeepseekV41Model:
    """Tiny model with the injected engram token map (construction tests)."""
    return DeepseekV41Model(
        DeepseekV41Config.from_dict(_DSV41_TINY_CONFIG),
        torch.float32,
        torch.device("cpu"),
        config=_DSV41_TINY_CONFIG,
    )


def test_module_layout_matches_csa2_modes() -> None:
    cfg = DeepseekV41Config.from_dict(_DSV41_TINY_CONFIG)
    model = _tiny_model()
    swa = model.layers[0].self_attn
    full_encoder = model.layers[2].self_attn
    reuse_encoder = model.layers[3].self_attn
    full_decoder = model.layers[4].self_attn
    reuse_decoder = model.layers[5].self_attn

    # Full layers own the compressor (no ape in V4.1) + indexer and are their
    # own group's main-KV source; Reuse/SWA layers own neither.
    assert full_encoder.indexer is not None
    assert full_encoder.cmp_wkv.out_features == cfg.head_dim
    assert not hasattr(full_encoder, "cmp_ape")
    assert full_encoder.mode.main_kv_source_layer == 2
    assert full_decoder.mode.main_kv_source_layer == 4
    for attn in (reuse_encoder, reuse_decoder, swa):
        assert attn.indexer is None
        assert not hasattr(attn, "cmp_wkv")
    # V4.1 removed the top-level hc_head_* weights (single-pass mHC).
    assert not any(name.startswith("hc_head") for name, _ in model.named_parameters())


def test_v41_indexer_uses_independent_key_path() -> None:
    cfg = DeepseekV41Config.from_dict(_DSV41_TINY_CONFIG)
    model = _tiny_model()
    indexer = model.layers[2].self_attn.indexer
    assert indexer is not None
    assert indexer.wq_b.weight.shape == (cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank)
    # wk acts on the compressor latent (NOT the hidden states) and produces ONE
    # shared key per compressed position: head_dim -> index_head_dim.
    assert indexer.wk.weight.shape == (cfg.index_head_dim, cfg.head_dim)
    assert indexer.k_norm.weight.shape == (cfg.index_head_dim,)
    assert indexer.weights_proj.weight.shape == (cfg.index_n_heads, cfg.hidden_size)


def test_moe_bias_vl_shifts_only_visual_tokens() -> None:
    cfg = DeepseekV41Config.from_dict(_DSV41_TINY_CONFIG)
    model = _tiny_model()
    moe = model.layers[2].mlp
    # gate_bias_vl exists but is unloaded until the checkpoint provides it;
    # V4.1 removed hash routing (no tid2eid).
    assert moe.gate_bias_vl.shape == (cfg.n_routed_experts,)
    assert moe.gate_bias_vl_loaded is False
    assert not hasattr(moe, "tid2eid")
    assert hasattr(moe, "e_score_correction_bias")

    moe.gate_bias_vl_loaded = True
    with torch.no_grad():
        moe.gate_bias_vl.fill_(1.5)
    gate_input = torch.zeros(2, cfg.hidden_size)
    input_ids = torch.tensor([1, cfg.image_token_id])
    logits, scores = moe._selection_logits(gate_input, input_ids)
    # Only the image token's SELECTION logits shift by bias_vl; the unbiased
    # scores (which carry the routing weights) are modality-independent.
    assert torch.allclose(logits[1], logits[0] + 1.5)
    assert torch.allclose(scores[1], scores[0])
    # Text-only tokens route on the unshifted scores.
    text_logits, _ = moe._selection_logits(gate_input, None)
    assert torch.allclose(text_logits, scores.expand(2, -1))


@pytest.mark.parametrize("tokens", [0, 1, 6])
@pytest.mark.parametrize("topk,normalize,limit", [(1, True, 2.0), (2, True, 2.0), (2, False, 0.0)])
@pytest.mark.parametrize("ep_size,ep_rank,selected", [(1, 0, 193), (2, 1, 193), (2, 1, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_moe_active_segments_match_all_expert_reference(
    monkeypatch: pytest.MonkeyPatch,
    tokens: int,
    topk: int,
    normalize: bool,
    limit: float,
    ep_size: int,
    ep_rank: int,
    selected: int,
    dtype: torch.dtype,
) -> None:
    from xllm.python import distributed

    torch.manual_seed(41)
    cfg = DeepseekV41Config.from_dict(
        {
            **_DSV41_TINY_CONFIG,
            "hidden_size": 8,
            "moe_intermediate_size": 6,
            "n_routed_experts": 384,
            "n_activated_experts": topk,
            "norm_topk_prob": normalize,
            "swiglu_limit": limit,
        }
    )
    cfg.ep_size, cfg.ep_rank = ep_size, ep_rank
    moe = DeepseekV41MoE(cfg, 2, dtype, torch.device("cpu"))
    with torch.no_grad():
        for parameter in moe.parameters():
            parameter.normal_(std=0.7)
        moe.e_score_correction_bias.zero_()
        moe.e_score_correction_bias[selected : selected + 2] = 100.0
        moe.gate_bias_vl.zero_()
        moe.gate_bias_vl[selected + 2 : selected + 4] = 100.0
    moe.gate_bias_vl_loaded = True
    hidden = torch.randn(1, tokens, cfg.hidden_size, dtype=dtype)
    input_ids = torch.tensor([cfg.image_token_id if i % 2 else 1 for i in range(tokens)])
    x = hidden.reshape(-1, cfg.hidden_size)
    logits, scores = moe._selection_logits(x, input_ids)
    indices = logits.topk(topk, dim=-1).indices
    weights = scores.gather(1, indices)
    if normalize and topk > 1:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * moe.routed_scaling
    expected = torch.zeros_like(x, dtype=torch.float32)
    active = 0
    for local_slot in range(moe.num_experts_per_rank):
        rows, slots = torch.where(indices == moe.start_expert_id + local_slot)
        if rows.numel() == 0:
            continue
        active += 1
        gate = torch.nn.functional.linear(x[rows], moe.experts_w1[local_slot]).float()
        up = torch.nn.functional.linear(x[rows], moe.experts_w3[local_slot]).float()
        if 0.0 < limit < 1_000_000.0:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        intermediate = weights[rows, slots, None] * (torch.nn.functional.silu(gate) * up)
        output = torch.nn.functional.linear(intermediate.to(dtype), moe.experts_w2[local_slot])
        expected.index_add_(0, rows, output.float())

    reduced: list[torch.Tensor] = []

    def fake_all_reduce(partial: torch.Tensor) -> None:
        reduced.append(partial.clone())
        partial.mul_(2)

    monkeypatch.setattr(distributed, "moe_ep_all_reduce", fake_all_reduce)
    original_where = torch.where
    where_calls = 0

    def counted_where(*args: torch.Tensor) -> tuple[torch.Tensor, ...] | torch.Tensor:
        nonlocal where_calls
        where_calls += 1
        return original_where(*args)

    original_linear = torch.nn.functional.linear
    linear_calls = 0

    def counted_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        nonlocal linear_calls
        linear_calls += 1
        return original_linear(input, weight, bias)

    monkeypatch.setattr(torch, "where", counted_where)
    monkeypatch.setattr(torch.nn.functional, "linear", counted_linear)
    actual = moe(hidden, input_ids)
    # One visual-bias selection and one local-route filter, independent of E.
    assert where_calls == 2
    # Three projections per active expert, plus routing and shared projections.
    assert linear_calls == 3 * active + 4
    assert active <= min(tokens * topk, 4)
    if ep_size > 1:
        assert len(reduced) == 1
        torch.testing.assert_close(reduced[0], expected)
        expected *= 2
    else:
        assert not reduced
    expected += moe._shared_forward(x).float()
    torch.testing.assert_close(actual, expected.to(dtype).view_as(hidden))


def test_registry_resolves_deepseek_v41() -> None:
    """V4.1 is registered, and NPU-gated: non-NPU platforms fail fast."""
    names = ("deepseek_v41", "DeepseekV41ForCausalLM")
    if _npu_available():
        for name in names:
            assert get_model_class(name).__name__ == "DeepseekV41ForCausalLM"
        return
    for name in names:
        with pytest.raises(NotImplementedError, match="supported platforms: \\['npu'\\]"):
            get_model_class(name)


def test_attention_backend_rejects_v41_off_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    """The executor rejects V4.1 before the CUDA branch can return a backend.

    Simulating the CUDA platform is the point: without the entry guard the
    CUDA ``FlashInferBackend`` branch would return a backend for V4.1. Only
    platform detection is patched; no attention backend is faked.
    """
    from xllm.python.model_executor import executor as executor_module

    monkeypatch.setattr(executor_module.current_platform, "is_npu", lambda: False)
    monkeypatch.setattr(executor_module.current_platform, "is_cuda", lambda: True)
    with pytest.raises(NotImplementedError, match="runs on NPU only"):
        executor_module._create_attention_backend(
            first_attention=object(),
            device=torch.device("cuda"),
            dtype=torch.float16,
            config={"model_type": "deepseek_v41"},
        )


# ---------------------------------------------------------------------------
# Weight loading (torch path, "converted" reference layout)
# ---------------------------------------------------------------------------


class _FakeStateDict:
    """Minimal StateDict stand-in for the weight-loading mapping test."""

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


def _build_tiny_checkpoint(cfg: DeepseekV41Config) -> dict[str, torch.Tensor]:
    """Reference (converted) checkpoint layout for the torch load path.

    Plain float tensors with the reference names; no ``.scale`` companions.
    The compressor gate and the indexer key path exist only on the kv-source
    layers, with ``wgate`` absent at ratio 1 and ``wk`` shaped
    ``[index_head_dim, head_dim]``.
    """
    hidden = cfg.hidden_size
    q_lora = cfg.q_lora_rank
    head_dim = cfg.head_dim
    n_heads = cfg.n_heads
    inter = cfg.moe_intermediate_size
    n_experts = cfg.n_routed_experts
    o_lora = cfg.o_lora_rank

    def w(out: int, inp: int) -> torch.Tensor:
        return torch.randn(out, inp)

    tensors: dict[str, torch.Tensor] = {
        "embed.weight": torch.randn(cfg.vocab_size, hidden),
        "norm.weight": torch.ones(hidden),
        "head.weight": torch.randn(cfg.vocab_size, hidden),
    }
    for i in range(cfg.n_layers):
        p = f"layers.{i}."
        tensors[p + "attn_norm.weight"] = torch.ones(hidden)
        tensors[p + "ffn_norm.weight"] = torch.ones(hidden)
        for part in ("attn", "ffn"):
            for suffix, shape in (
                ("fn", ((2 + cfg.hc_mult) * cfg.hc_mult, cfg.hc_mult * hidden)),
                ("scale", (3,)),
                ("base", ((2 + cfg.hc_mult) * cfg.hc_mult,)),
            ):
                tensors[p + f"hc_{part}_{suffix}"] = torch.randn(*shape)
        tensors[p + "attn.wq_a.weight"] = w(q_lora, hidden)
        tensors[p + "attn.wq_b.weight"] = w(n_heads * head_dim, q_lora)
        tensors[p + "attn.wkv.weight"] = w(head_dim, hidden)
        tensors[p + "attn.wo_a.weight"] = w(cfg.o_groups * o_lora, (n_heads * head_dim) // cfg.o_groups)
        tensors[p + "attn.wo_b.weight"] = w(hidden, cfg.o_groups * o_lora)
        tensors[p + "attn.q_norm.weight"] = torch.ones(q_lora)
        tensors[p + "attn.kv_norm.weight"] = torch.ones(head_dim)
        tensors[p + "attn.attn_sink"] = torch.zeros(n_heads)
        if i in (2, 4):  # Full layers: compressor (gate only at ratio 2) + indexer.
            tensors[p + "attn.compressor.wkv.weight"] = w(head_dim, hidden)
            if cfg.compress_ratios[i] > 1:
                tensors[p + "attn.compressor.wgate.weight"] = w(head_dim, hidden)
            tensors[p + "attn.compressor.norm.weight"] = torch.ones(head_dim)
            tensors[p + "attn.indexer.wq_b.weight"] = w(cfg.index_n_heads * cfg.index_head_dim, q_lora)
            tensors[p + "attn.indexer.wk.weight"] = w(cfg.index_head_dim, head_dim)
            tensors[p + "attn.indexer.k_norm.weight"] = torch.ones(cfg.index_head_dim)
            tensors[p + "attn.indexer.weights_proj.weight"] = w(cfg.index_n_heads, hidden)
        # MoE: gate + bias + bias_vl, routed + shared experts.
        tensors[p + "ffn.gate.weight"] = torch.randn(n_experts, hidden)
        tensors[p + "ffn.gate.bias"] = torch.randn(n_experts)
        tensors[p + "ffn.gate.bias_vl"] = torch.randn(n_experts)
        for expert in range(n_experts):
            for name, out, inp in (("w1", inter, hidden), ("w2", hidden, inter), ("w3", inter, hidden)):
                tensors[p + f"ffn.experts.{expert}.{name}.weight"] = w(out, inp)
        for name, out, inp in (("w1", inter, hidden), ("w2", hidden, inter), ("w3", inter, hidden)):
            tensors[p + f"ffn.shared_experts.{name}.weight"] = w(out, inp)
        # Engram layer: FP8-E4M3 payload bytes + ue8m0 scales stay raw (uint8),
        # wkv / q / k arrive as plain floats.
        if i in cfg.engram_layer_ids:
            n_hash_cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads
            head_dim = cfg.engram_head_dim
            rows = cfg.engram_num_embeddings[cfg.engram_layer_ids.index(i)]
            tensors[p + "engram.embed.weight"] = torch.randint(0, 256, (rows, head_dim), dtype=torch.uint8)
            tensors[p + "engram.embed.scale"] = torch.randint(126, 130, (rows, head_dim // 32), dtype=torch.uint8)
            tensors[p + "engram.wkv.weight"] = w(hidden * (cfg.hc_mult + 1), n_hash_cols * head_dim)
            tensors[p + "engram.q_weight"] = w(cfg.hc_mult, hidden)
            tensors[p + "engram.k_weight"] = w(cfg.hc_mult, hidden)
    return tensors


def test_load_weights_maps_v41_checkpoint_layout() -> None:
    cfg = DeepseekV41Config.from_dict(_DSV41_TINY_CONFIG)
    lm = DeepseekV41ForCausalLM({**_DSV41_TINY_CONFIG, "device": "cpu", "dtype": "float32"})
    tensors = _build_tiny_checkpoint(cfg)
    lm.load_weights([_FakeStateDict(tensors)], 0, 1)

    # Attention projections keep the checkpoint layout; the ratio-1 source has
    # no wgate; the indexer key path maps wk + k_norm.
    assert torch.equal(lm.model.layers[0].self_attn.q_a_proj.weight, tensors["layers.0.attn.wq_a.weight"])
    assert lm.model.layers[3].self_attn.attn_sink_loaded
    assert torch.equal(lm.model.layers[2].self_attn.cmp_wkv.weight, tensors["layers.2.attn.compressor.wkv.weight"])
    assert not hasattr(lm.model.layers[4].self_attn, "cmp_wgate")
    assert torch.equal(lm.model.layers[2].self_attn.indexer.wk.weight, tensors["layers.2.attn.indexer.wk.weight"])
    # MoE: gate bias + bias_vl + experts.
    assert torch.equal(lm.model.layers[5].mlp.e_score_correction_bias, tensors["layers.5.ffn.gate.bias"])
    assert lm.model.layers[5].mlp.gate_bias_vl_loaded
    assert torch.equal(lm.model.layers[1].mlp.experts_w3[3], tensors["layers.1.ffn.experts.3.w3.weight"])
    assert torch.equal(lm.model.layers[0].mlp.shared_w1.weight, tensors["layers.0.ffn.shared_experts.w1.weight"])
    # Engram layer 1: the table stays quantized (raw bytes) and is not
    # silently dropped; every other layer carries no module.
    eng = lm.model.layers[1].engram
    assert eng is not None
    assert torch.equal(eng.embed_tokens.weight, tensors["layers.1.engram.embed.weight"])
    assert torch.equal(eng.embed_tokens.weight_scale, tensors["layers.1.engram.embed.scale"])
    assert torch.equal(eng.wkv.weight, tensors["layers.1.engram.wkv.weight"])
    assert torch.equal(eng.q_weight, tensors["layers.1.engram.q_weight"])
    assert torch.equal(eng.k_weight, tensors["layers.1.engram.k_weight"])
    for layer_id in (0, 2, 3, 4, 5):
        assert lm.model.layers[layer_id].engram is None
    # Final norm + head (no hc_head).
    assert torch.equal(lm.model.norm.weight, tensors["norm.weight"])
    assert torch.equal(lm.lm_head.weight, tensors["head.weight"])


def test_load_weights_requires_attention_projections() -> None:
    """A missing mandatory attention projection fails the load.

    :class:`DeepseekV41Attention` constructs q_a/q_b/kv/o_a/o_b
    unconditionally, so an export that omits one must raise instead of leaving
    the parameter at uninitialized values.
    """
    cfg = DeepseekV41Config.from_dict(_DSV41_TINY_CONFIG)
    tensors = _build_tiny_checkpoint(cfg)
    del tensors["layers.0.attn.wq_a.weight"]
    lm = DeepseekV41ForCausalLM({**_DSV41_TINY_CONFIG, "device": "cpu", "dtype": "float32"})
    with pytest.raises(KeyError, match=r"layers\.0\.attn\.wq_a\.weight"):
        lm.load_weights([_FakeStateDict(tensors)], 0, 1)


def test_load_weights_requires_moe_gate() -> None:
    """A MoE layer dispatches on module type, so a missing gate must raise.

    Layer 5 is MoE in the tiny config. Dropping its ``ffn.gate.weight`` must
    surface from ``_load_torch_moe`` instead of silently skipping the MLP
    (the old ``_has(ffn.gate.weight)`` dispatch left the layer uninitialized).
    """
    cfg = DeepseekV41Config.from_dict(_DSV41_TINY_CONFIG)
    tensors = _build_tiny_checkpoint(cfg)
    del tensors["layers.5.ffn.gate.weight"]
    lm = DeepseekV41ForCausalLM({**_DSV41_TINY_CONFIG, "device": "cpu", "dtype": "float32"})
    with pytest.raises(KeyError):
        lm.load_weights([_FakeStateDict(tensors)], 0, 1)


# ---------------------------------------------------------------------------
# W8A8 (ascend export) loading: a missing weight_offset must be zero-filled
# ---------------------------------------------------------------------------


# Minimal ascend geometry: one SWA (no compressor/indexer) dense layer, so the
# loader stays small; the W8A8DynamicLinear buffers are the point of the test.
_DSV41_ASCEND_CONFIG = {
    **_DSV41_CONFIG,
    "num_hidden_layers": 1,
    "hidden_size": 32,
    "num_attention_heads": 2,
    "head_dim": 16,
    "q_lora_rank": 8,
    "o_lora_rank": 4,
    "o_groups": 2,
    "qk_rope_head_dim": 4,
    "vocab_size": 32,
    "moe_intermediate_size": 16,
    "first_k_dense_replace": 1,
    "compress_ratios": [0],
    "kv_source_layer_ids": [],
    "index_source_layer_ids": [],
    "candidate_source_layer_id": -1,
    "candidate_topk_blocks": 0,
    "candidate_block_size": 0,
    "engram_layer_ids": [],
    "max_position_embeddings": 64,
    "window_size": 8,
    "sliding_window": 8,
}


def _make_ascend_lm(tmp_path, monkeypatch: pytest.MonkeyPatch) -> DeepseekV41ForCausalLM:
    """Ascend-mode model; ``quant_model_description.json`` selects the export.

    ``prepare_quant_weight`` is the only kernel hit by
    ``process_weights_after_loading`` on these non-transposed projections, so
    stubbing it keeps the weight-loading test free of NPU kernels.
    """
    from xllm.python import kernels

    monkeypatch.setattr(kernels, "prepare_quant_weight", lambda weight: weight, raising=False)
    (tmp_path / "quant_model_description.json").write_text("{}")
    lm = DeepseekV41ForCausalLM(
        {**_DSV41_ASCEND_CONFIG, "device": "cpu", "dtype": "float32", "model_path": str(tmp_path)}
    )
    assert lm.quant_mode == "ascend"
    return lm


def _ascend_tensors(omit_attn_offsets: bool = False) -> dict[str, torch.Tensor]:
    """W8A8 export layout for the minimal ascend config (dense, one SWA layer)."""
    cfg = _DSV41_ASCEND_CONFIG
    hidden = cfg["hidden_size"]
    q_lora = cfg["q_lora_rank"]
    head_dim = cfg["head_dim"]
    n_heads = cfg["num_attention_heads"]
    o_groups = cfg["o_groups"]
    o_lora = cfg["o_lora_rank"]
    inter = cfg["moe_intermediate_size"]
    p = "layers.0."
    tensors: dict[str, torch.Tensor] = {
        "embed.weight": torch.randn(cfg["vocab_size"], hidden),
        "norm.weight": torch.ones(hidden),
        "head.weight": torch.randn(cfg["vocab_size"], hidden),
    }
    tensors[p + "attn_norm.weight"] = torch.ones(hidden)
    tensors[p + "ffn_norm.weight"] = torch.ones(hidden)
    hc_mult = cfg["hc_mult"]
    for part in ("attn", "ffn"):
        tensors[p + f"hc_{part}_fn"] = torch.randn((2 + hc_mult) * hc_mult, hc_mult * hidden)
        tensors[p + f"hc_{part}_scale"] = torch.randn(3)
        tensors[p + f"hc_{part}_base"] = torch.randn((2 + hc_mult) * hc_mult)

    def w8a8(prefix: str, out: int, inp: int) -> None:
        tensors[prefix + ".weight"] = torch.zeros(out, inp, dtype=torch.int8)
        tensors[prefix + ".weight_scale"] = torch.ones(out, 1)
        if not omit_attn_offsets:
            tensors[prefix + ".weight_offset"] = torch.zeros(out, 1)

    w8a8(p + "attn.wq_a", q_lora, hidden)
    w8a8(p + "attn.wq_b", n_heads * head_dim, q_lora)
    w8a8(p + "attn.wkv", head_dim, hidden)
    tensors[p + "attn.wo_a.weight"] = torch.randn(o_groups * o_lora, (n_heads * head_dim) // o_groups)
    tensors[p + "attn.wo_b.weight"] = torch.randn(hidden, o_groups * o_lora)
    tensors[p + "attn.q_norm.weight"] = torch.ones(q_lora)
    tensors[p + "attn.kv_norm.weight"] = torch.ones(head_dim)
    for name, rows, cols in (
        ("gate_proj", inter, hidden),
        ("up_proj", inter, hidden),
        ("down_proj", hidden, inter),
    ):
        tensors[p + f"ffn.{name}.weight"] = torch.zeros(rows, cols, dtype=torch.int8)
        tensors[p + f"ffn.{name}.weight_scale"] = torch.ones(rows, 1)
        tensors[p + f"ffn.{name}.weight_offset"] = torch.zeros(rows, 1)
    return tensors


def _poison_w8a8_offsets(lm: DeepseekV41ForCausalLM) -> None:
    """Simulate the ``torch.empty`` buffer contents: fill every offset with NaN."""
    for module in lm.modules():
        if hasattr(module, "weight_offset"):
            module.weight_offset.data.fill_(float("nan"))


def _ascend_attn_offsets(lm: DeepseekV41ForCausalLM) -> list[torch.Tensor]:
    attn = lm.model.layers[0].self_attn
    return [getattr(attn, name).weight_offset for name in ("q_a_proj", "q_b_proj", "kv_proj")]


def test_ascend_load_zero_fills_missing_weight_offset(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing ``weight_offset`` is copied in as zeros, not left uninitialized.

    ``W8A8DynamicLinear.weight_offset`` starts as ``torch.empty`` and
    ``process_weights_after_loading`` asserts it is all-zero, so skipping the
    tensor would surface as a spurious ValueError. Pre-filling the buffers with
    NaN makes the uninitialized case deterministic.
    """
    lm = _make_ascend_lm(tmp_path, monkeypatch)
    _poison_w8a8_offsets(lm)
    lm.load_weights([_FakeStateDict(_ascend_tensors(omit_attn_offsets=True))], 0, 1)
    offsets = _ascend_attn_offsets(lm)
    assert all(torch.all(offset == 0) for offset in offsets)
    assert all(offset.dtype == torch.float32 for offset in offsets)
    assert [offset.numel() for offset in offsets] == [8, 32, 16]


def test_ascend_load_rejects_nonzero_weight_offset(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkpoint carrying a non-zero (asymmetric) offset is still rejected."""
    lm = _make_ascend_lm(tmp_path, monkeypatch)
    _poison_w8a8_offsets(lm)
    tensors = _ascend_tensors()
    tensors["layers.0.attn.wq_a.weight_offset"] = torch.ones(8, 1)
    with pytest.raises(ValueError, match="zero weight_offset"):
        lm.load_weights([_FakeStateDict(tensors)], 0, 1)


def test_ascend_load_rejects_missing_weight_and_scale(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The unconditionally-constructed projections require ``weight`` / ``weight_scale``."""
    for missing in ("layers.0.attn.wq_a.weight", "layers.0.attn.wq_a.weight_scale"):
        lm = _make_ascend_lm(tmp_path, monkeypatch)
        tensors = _ascend_tensors()
        del tensors[missing]
        with pytest.raises(KeyError):
            lm.load_weights([_FakeStateDict(tensors)], 0, 1)


# V4.1 quant kernels, inlined in deepseek_v41 (semantics doc sections 2.1-2.5)
# ---------------------------------------------------------------------------

_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _round_e2m1_slow(value: float) -> float:
    """Independent slow E2M1 rounding: nearest value, ties to even mantissa."""
    magnitude = abs(value)
    below = max((g for g in _E2M1_GRID if g <= magnitude), default=0.0)
    above = min((g for g in _E2M1_GRID if g >= magnitude), default=6.0)
    if magnitude - below < above - magnitude:
        nearest = below
    elif magnitude - below > above - magnitude:
        nearest = above
    else:
        nearest = below if _E2M1_GRID.index(below) % 2 == 0 else above
    return -nearest if value < 0 else nearest


def _expected_fp8_act_qdq(x: torch.Tensor, group: int) -> torch.Tensor:
    shape = x.shape
    grouped = x.float().reshape(-1, group)
    amax = grouped.abs().amax(dim=-1).clamp_min(1e-4)
    t = amax * torch.tensor(1.0 / 448.0, dtype=torch.float32)
    scale = torch.exp2(torch.ceil(torch.log2(t.double()))).float()
    normalized = (grouped / scale.unsqueeze(-1)).clamp(-448.0, 448.0)
    return (normalized.to(torch.float8_e4m3fn).float() * scale.unsqueeze(-1)).reshape(shape)


def _expected_fp4_e8m0(x: torch.Tensor, group: int) -> torch.Tensor:
    shape = x.shape
    grouped = x.float().reshape(-1, group)
    amax = grouped.abs().amax(dim=-1).clamp_min(6.0 * 2.0**-126)
    t = amax * torch.tensor(1.0 / 6.0, dtype=torch.float32)
    scale = torch.exp2(torch.ceil(torch.log2(t.double()))).float()
    normalized = (grouped / scale.unsqueeze(-1)).clamp(-6.0, 6.0)
    rounded = torch.tensor(
        [_round_e2m1_slow(v) for v in normalized.flatten().tolist()],
        dtype=torch.float32,
    ).reshape(normalized.shape)
    return (rounded * scale.unsqueeze(-1)).reshape(shape)


def _expected_fp4_e4m3(x: torch.Tensor, group: int) -> torch.Tensor:
    shape = x.shape
    grouped = x.float().reshape(-1, group)
    amax = grouped.abs().amax(dim=-1).clamp_min(6.0 * 2.0**-9)
    scale = (amax / 6.0).to(torch.float8_e4m3fn).float()
    normalized = (grouped / scale.unsqueeze(-1)).clamp(-6.0, 6.0)
    rounded = torch.tensor(
        [_round_e2m1_slow(v) for v in normalized.flatten().tolist()],
        dtype=torch.float32,
    ).reshape(normalized.shape)
    return (rounded * scale.unsqueeze(-1)).reshape(shape)


def _make_block_quantized(out_dim: int, in_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds a synthetic 32x32-block FP8 weight that round-trips exactly."""
    blocks_out = -(-out_dim // 32)
    blocks_in = -(-in_dim // 32)
    exponents = torch.randint(-8, 9, (blocks_out, blocks_in))
    scale_bytes = (exponents + 127).to(torch.uint8)
    scales = (
        torch.exp2(exponents.double())
        .repeat_interleave(32, dim=0)
        .repeat_interleave(32, dim=1)[:out_dim, :in_dim]
        .float()
    )
    base = (torch.randn(out_dim, in_dim) * 100).clamp(-448, 448)
    base = base.to(torch.float8_e4m3fn).float()
    weight = (base * scales).to(torch.bfloat16)
    quantized = (weight.float() / scales).to(torch.float8_e4m3fn)
    return quantized, scale_bytes, weight


def test_ue8m0_to_scale() -> None:
    scale_bytes = torch.tensor([0x01, 0x7F, 0x80, 0x00, 0xFE], dtype=torch.uint8)
    expected = torch.tensor([2.0**-126, 1.0, 2.0, 2.0**-127, 2.0**127])
    assert torch.equal(dq.ue8m0_to_scale(scale_bytes), expected)
    # int8 view of the same bytes; the NaN exponent byte 0xFF is rejected.
    raw = torch.tensor([0x80, 0x7F], dtype=torch.uint8).view(torch.int8)
    assert torch.equal(dq.ue8m0_to_scale(raw), torch.tensor([2.0, 1.0]))
    with pytest.raises(ValueError):
        dq.ue8m0_to_scale(torch.tensor([0x01, 0xFF, 0x02], dtype=torch.uint8))


def test_dequant_fp8_block_roundtrip_exact() -> None:
    torch.manual_seed(0)
    quantized, scale_bytes, weight = _make_block_quantized(96, 160)
    out = dq.dequant_fp8_block(quantized, scale_bytes)
    assert out.dtype == torch.bfloat16
    assert out.shape == weight.shape
    assert torch.equal(out, weight)
    # Non-multiple-of-32 boundaries clip the last block correctly.
    torch.manual_seed(1)
    quantized, scale_bytes, weight = _make_block_quantized(70, 100)
    out = dq.dequant_fp8_block(quantized, scale_bytes)
    assert out.shape == (70, 100)
    assert torch.equal(out, weight)
    with pytest.raises(ValueError):
        dq.dequant_fp8_block(quantized, scale_bytes[:-1])


def test_fp4_nibble_table_and_unpack_order() -> None:
    table = dq.fp4_nibble_table()
    expected = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    assert table.numel() == 16
    assert torch.equal(table, torch.tensor(expected, dtype=torch.float32))
    assert table.view(torch.int32)[8].item() & 0xFFFFFFFF == 0x80000000  # -0.0 bit pattern
    # LOW nibble is element 2j, HIGH nibble is element 2j+1 (convert.py:31-34).
    packed = torch.tensor([[0x12, 0x21, 0x98, 0x9A, 0x00, 0x77, 0xF0]], dtype=torch.uint8)
    out = dq.unpack_fp4(packed)
    expected_row = [1.0, 0.5, 0.5, 1.0, -0.0, -0.5, -1.0, -0.5, 0.0, 0.0, 6.0, 6.0, 0.0, -6.0]
    assert out.dtype == torch.bfloat16 and out.shape == (1, 14)
    assert torch.equal(out.float(), torch.tensor([expected_row]))


def test_dequant_fp4_group_known_scales() -> None:
    torch.manual_seed(3)
    n, k = 8, 96
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8).view(torch.int8)
    # Group scales 2^0, 2^1, 2^-2 for column blocks [0:32), [32:64), [64:96).
    scale_bytes = torch.tensor([[0x7F, 0x80, 0x7D]] * n, dtype=torch.uint8)
    out = dq.dequant_fp4_group(packed, scale_bytes)
    unpacked = dq.unpack_fp4(packed, dtype=torch.float32)
    scales = torch.tensor([1.0, 2.0, 0.25]).repeat_interleave(32)
    assert out.dtype == torch.bfloat16 and out.shape == (n, k)
    assert torch.equal(out, (unpacked * scales).to(torch.bfloat16))
    # Zero scale bytes decode to 2**-127 (16 packed bytes = 32 elements = one group).
    zero_out = dq.dequant_fp4_group(packed[:, :16], torch.zeros(n, 1, dtype=torch.uint8))
    assert torch.equal(zero_out, (unpacked[:, :32] * 2.0**-127).to(torch.bfloat16))


def test_fp8_act_qdq_known_and_reference() -> None:
    # amax = 448 gives scale exactly 1: values round straight through the e4m3
    # grid (100 -> 96, 300 -> 288, round-to-nearest-even; 449 saturates).
    x = torch.tensor([[448.0, 224.0, 1.0, 0.0, -448.0, 100.0, 0.5, 300.0]])
    expected = torch.tensor([[448.0, 224.0, 1.0, 0.0, -448.0, 96.0, 0.5, 288.0]])
    assert torch.equal(dq.fp8_act_qdq(x, group=8), expected)
    assert torch.all(dq.fp8_act_qdq(torch.full((1, 32), 449.0), group=32) == 448.0)
    torch.manual_seed(6)
    xr = (torch.randn(4, 128) * 7).to(torch.bfloat16)
    xr[:, :5] = 0
    out = dq.fp8_act_qdq(xr, group=32)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out.float(), _expected_fp8_act_qdq(xr, 32))
    assert torch.all(out[:, :5] == 0)


def test_fp4_act_qdq_e8m0_known_and_reference() -> None:
    # amax = 12 -> s = 2; includes every tie case, which must round to the
    # even E2M1 neighbor (0.75 -> 1, 1.25 -> 1, 1.75 -> 2, 2.5 -> 2 after /s).
    x = torch.tensor(
        [[6.0, 3.0, 1.5, 0.75, 0.25, 0.0, -6.0, -0.75, 12.0, 5.0, 2.5, 3.5, 0.5, 1.0, 4.0, 0.26]],
        dtype=torch.bfloat16,
    )
    expected = torch.tensor([[6, 3, 2, 1, 0, 0, -6, -1, 12, 4, 2, 4, 0, 1, 4, 0]], dtype=torch.bfloat16)
    assert torch.equal(dq.fp4_act_qdq_e8m0(x, group=16), expected)
    torch.manual_seed(8)
    xr = (torch.randn(4, 128) * 3).to(torch.bfloat16)
    xr[:, :5] = 0
    out = dq.fp4_act_qdq_e8m0(xr, group=32)
    assert torch.equal(out.float(), _expected_fp4_e8m0(xr, 32))
    # All-zero group stays zero.
    assert torch.all(dq.fp4_act_qdq_e8m0(torch.zeros(2, 32, dtype=torch.bfloat16), group=32) == 0)


def test_fp4_act_qdq_e4m3_scale_known_and_reference() -> None:
    # s = e4m3(amax / 6) is NOT a power of two (e.g. 1 -> 0.171875).
    cases = [
        (6.0, 6.0),  # s = e4m3(1) = 1
        (1.0, 1.03125),  # s = e4m3(1/6) = 0.171875
        (0.0234375, 0.0234375),  # 6 * 2**-8: s = 2**-8 exactly
        (5.0, 4.875),  # s = e4m3(5/6) = 0.8125; 5/0.8125 rounds to 6
        (0.0, 0.0),  # all-zero group: amax floor 6 * 2**-9
    ]
    for value, expected in cases:
        out = dq.fp4_act_qdq_e4m3_scale(torch.tensor([value]), group=1)
        assert out.item() == expected, (value, out.item(), expected)
    torch.manual_seed(9)
    xr = (torch.randn(4, 128) * 2).to(torch.bfloat16)
    xr[:, :3] = 0
    out = dq.fp4_act_qdq_e4m3_scale(xr, group=16)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out.float(), _expected_fp4_e4m3(xr, 16))


def test_qdq_rejects_indivisible_group() -> None:
    x = torch.randn(2, 40)
    with pytest.raises(ValueError):
        dq.fp8_act_qdq(x, group=32)
    with pytest.raises(ValueError):
        dq.fp4_act_qdq_e8m0(x, group=32)
    with pytest.raises(ValueError):
        dq.fp4_act_qdq_e4m3_scale(x, group=16)


# ---------------------------------------------------------------------------
# Cache layout (adaptation contract section 2)
# ---------------------------------------------------------------------------


def test_build_cache_specs_v41() -> None:
    caches_info, group_infos = build_cache_specs_v41(
        compress_ratios=[0, 0, 2, 2, 1, 1],
        kv_source_layer_ids=[2, 4],
        index_source_layer_ids=[2, 4, 5],
        window_size=128,
        n_layers=6,
    )
    # Group registration order is exact: [SWA, TOKEN(2), TOKEN(1)] -- also when
    # the kv-source input arrives unsorted (groups register by layer id).
    expected_groups = [
        (DSA_CACHE_SLIDING_WINDOW, 1, 128),
        (DSA_CACHE_TOKEN, 2, 128),
        (DSA_CACHE_TOKEN, 1, 128),
    ]
    assert [(g.cache_type, g.ratio, g.block_size) for g in group_infos] == expected_groups
    _, unsorted = build_cache_specs_v41(
        compress_ratios=[1, 1, 2, 2, 1, 1],
        kv_source_layer_ids=[4, 2],
        index_source_layer_ids=[2, 4, 5],
        window_size=128,
        n_layers=6,
    )
    assert [(g.cache_type, g.ratio, g.block_size) for g in unsorted] == expected_groups

    # A plan whose lowest-id kv-source carries ratio 1 would register
    # [SWA, TOKEN(1), TOKEN(2)], diverging from the worker multi_block_tables
    # export order (kMultiBlockExportOrder -> [SWA, TOKEN(2), TOKEN(1)]). The
    # builder pairs tables and groups by index, so this must fail loud rather
    # than mis-pair them.
    with pytest.raises(ValueError, match="export order"):
        build_cache_specs_v41(
            compress_ratios=[0, 0, 1, 1, 2, 2],
            kv_source_layer_ids=[2, 4],
            index_source_layer_ids=[2, 4, 5],
            window_size=128,
            n_layers=6,
        )

    def entries(layer_id: int) -> list[tuple[int, int, int]]:
        return [(c.cache_type, c.ratio, c.group_id) for c in caches_info[layer_id]]

    swa = (DSA_CACHE_SLIDING_WINDOW, 1, 0)
    # kv-source ratio-2 layer: key + index + [swa, kv_state, score_state];
    # kv-source ratio-1: key + index + swa (no states); others only swa.
    assert entries(0) == [swa]
    assert entries(3) == [swa]
    assert entries(2) == [(DSA_CACHE_TOKEN, 2, 1), (DSA_CACHE_TOKEN, 2, 1), swa, swa, swa]
    assert entries(4) == [(DSA_CACHE_TOKEN, 1, 2), (DSA_CACHE_TOKEN, 1, 2), swa]

    # The backend resolves the LayerCache slot layout from the same specs.
    backend = _make_csa2_backend(DeepseekV41Config.from_dict(_DSV41_FORWARD_CONFIG))
    full_r2 = backend._resolve_cache_mapping(2, 2)
    assert (
        full_r2.cmp_cache_idx,
        full_r2.index_cache_idx,
        full_r2.ori_cache_idx,
        full_r2.kv_state_cache_idx,
        full_r2.score_state_cache_idx,
    ) == (0, 1, 2, 3, 4)
    full_r1 = backend._resolve_cache_mapping(4, 1)
    assert (full_r1.cmp_cache_idx, full_r1.index_cache_idx, full_r1.ori_cache_idx) == (0, 1, 2)
    assert (full_r1.kv_state_cache_idx, full_r1.score_state_cache_idx) == (-1, -1)
    reindex = backend._resolve_cache_mapping(5, 1)
    assert (reindex.cmp_cache_idx, reindex.ori_cache_idx) == (-1, 0)


# ---------------------------------------------------------------------------
# Forward-path harness (ModelExecutor + pure-torch CSA2 backend)
# ---------------------------------------------------------------------------

# TOKEN cache block size of the V4.1 layout (contract section 2).
_TOKEN_BLOCK_SIZE = 128
# SWA ring blocks per sequence (contract: the ring is at least 2 blocks).
_SWA_RING_BLOCKS = 2
# The host-side metadata fields (kv_seq_lens_host / q_seq_lens_host) are CPU
# tensors by production contract: PyAttentionMetadataView::make_host_int32_view
# builds them with torch::kCPU (py_attention_metadata.cpp:481-493). Keep them on
# the host even when the rest of the harness runs on NPU.
_HOST_DEVICE = torch.device("cpu")


def _make_csa2_backend(
    cfg: DeepseekV41Config,
    device: torch.device | None = None,
    *,
    quant_mode: str = V41_QUANT_NONE,
    acl_graph_enabled: bool = False,
    max_model_len: int = 0,
) -> Csa2AttentionBackend:
    return Csa2AttentionBackend(
        compress_ratios=list(cfg.compress_ratios),
        window_size=cfg.window_size,
        n_layers=cfg.n_layers,
        num_heads=cfg.n_heads,
        attn_head_dim=cfg.head_dim,
        index_topk=cfg.index_topk,
        index_n_heads=cfg.index_n_heads,
        index_head_dim=cfg.index_head_dim,
        rope_head_dim=cfg.qk_rope_head_dim,
        device=device or torch.device("cpu"),
        dtype=torch.float32,
        kv_source_layer_ids=list(cfg.kv_source_layer_ids),
        index_source_layer_ids=list(cfg.index_source_layer_ids),
        candidate_source_layer_id=cfg.candidate_source_layer_id,
        candidate_topk_blocks=cfg.candidate_topk_blocks,
        candidate_block_size=cfg.candidate_block_size,
        quant_mode=quant_mode,
        acl_graph_enabled=acl_graph_enabled,
        max_model_len=max_model_len,
    )


# ---------------------------------------------------------------------------
# ACL-graph quant admission + startup capacity check (51361671 A/C)
# ---------------------------------------------------------------------------


def _graph_admission_cfg() -> DeepseekV41Config:
    return DeepseekV41Config.from_dict(_DSV41_FORWARD_CONFIG)


@pytest.mark.parametrize(
    "quant_mode,expected",
    [(V41_QUANT_ASCEND, True), (V41_QUANT_FP8, False), (V41_QUANT_NONE, False)],
)
def test_csa2_marks_moe_graph_capability_from_quant_mode(quant_mode: str, expected: bool) -> None:
    """Only the W8A8 (ascend) export has an ACL-graph-capturable MoE chain."""
    backend = _make_csa2_backend(_graph_admission_cfg(), torch.device("cpu"), quant_mode=quant_mode)
    assert backend.quant_mode == quant_mode
    assert backend.moe_graph_capturable is expected


def test_csa2_acl_graph_capacity_is_checked_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """C: fail at startup, not at the first captured forward.

    A derivable ``max_model_len`` or an explicit env value satisfies the check;
    a non-capturable quant mode never reaches it (graph is disabled anyway).
    """
    monkeypatch.delenv("XLLM_V41_GRAPH_CAPACITY_GRANULARITY", raising=False)
    cfg = _graph_admission_cfg()

    derived = _make_csa2_backend(
        cfg,
        torch.device("cpu"),
        quant_mode=V41_QUANT_ASCEND,
        acl_graph_enabled=True,
        max_model_len=32768,
    )
    assert derived.graph_token_capacity_granularity > 0

    with pytest.raises(RuntimeError, match="static committed-row capacity"):
        _make_csa2_backend(
            cfg,
            torch.device("cpu"),
            quant_mode=V41_QUANT_ASCEND,
            acl_graph_enabled=True,
            max_model_len=0,
        )

    # fp8/none already forces eager: the startup capacity check must not fire.
    _make_csa2_backend(
        cfg,
        torch.device("cpu"),
        quant_mode=V41_QUANT_NONE,
        acl_graph_enabled=True,
        max_model_len=0,
    )

    # An explicit env value overrides the derivation and satisfies the check.
    monkeypatch.setenv("XLLM_V41_GRAPH_CAPACITY_GRANULARITY", "8192")
    explicit = _make_csa2_backend(
        cfg,
        torch.device("cpu"),
        quant_mode=V41_QUANT_ASCEND,
        acl_graph_enabled=True,
        max_model_len=0,
    )
    assert explicit.graph_token_capacity_granularity == 8192


def test_graph_token_capacity_granularity_caps_the_serving_bound() -> None:
    """The derivation caps ``max_model_len`` at 32768 (-> 2048 buckets).

    ``max_position_embeddings`` (1048576 for V4.1) is the RoPE horizon, not a
    served context, and must not drive the per-layer gather width; the cap
    reproduces the launcher's historical 2048 default while still shortening
    buckets for smaller served contexts. ``0`` / unknown keeps the fail-closed
    eager guard.
    """
    assert _derive_graph_token_capacity_granularity(1048576) == 2048
    assert _derive_graph_token_capacity_granularity(32768) == 2048
    assert _derive_graph_token_capacity_granularity(8192) == 1024
    assert _derive_graph_token_capacity_granularity(0) == 0
    assert _derive_graph_token_capacity_granularity(-1) == 0


def _random_init_v41(lm: DeepseekV41ForCausalLM) -> None:
    """Deterministic random weights: RMSNorm ones, biases/sinks zero."""
    with torch.no_grad():
        for name, param in lm.named_parameters():
            if param.dtype == torch.uint8:
                # Engram table bytes: fill E4M3 1.0 payloads and 2^0 scales
                # (uninitialized bytes can decode to NaN).
                fill = 127 if name.endswith("weight_scale") else 0x38
                param.copy_(torch.full_like(param, fill))
                continue
            if name == "model.norm.weight" or name.endswith(("layernorm.weight", "cmp_norm.weight", "k_norm.weight")):
                param.copy_(torch.ones_like(param))
            elif name.endswith(("attn_sink", "e_score_correction_bias", "gate_bias_vl")):
                param.copy_(torch.zeros_like(param))
            else:
                param.normal_(mean=0.0, std=0.05)
            param.requires_grad_(False)


def _allocate_v41_layer_caches(
    cfg: DeepseekV41Config,
    n_seqs: int,
    dtype: torch.dtype,
    device: torch.device | None = None,
    *,
    reserve_token_sink: bool = False,
) -> list[LayerCache]:
    """LayerCache tensors for the contract-section-2 layout.

    With ``reserve_token_sink`` the TOKEN pools carry one extra leading block
    (block 0) that is never written, so real per-sequence blocks are addressed
    1-based. This mirrors the production invariant that block 0 is a reserved
    zero padding sink, which the padded (``-1``) rows clamp to.
    """
    device = device or torch.device("cpu")

    def swa(dtype_: torch.dtype = dtype) -> torch.Tensor:
        return torch.zeros(_SWA_RING_BLOCKS * n_seqs, cfg.window_size, cfg.head_dim, dtype=dtype_, device=device)

    token_blocks = n_seqs + 1 if reserve_token_sink else n_seqs
    caches: list[LayerCache] = []
    for layer_id in range(cfg.n_layers):
        if layer_id not in cfg.kv_source_layer_ids:
            caches.append(LayerCache(key=None, value=None, swa=swa()))
            continue
        ratio = cfg.compress_ratios[layer_id]
        states = ratio > 1
        caches.append(
            LayerCache(
                key=torch.zeros(token_blocks, _TOKEN_BLOCK_SIZE, cfg.head_dim, dtype=dtype, device=device),
                value=None,
                index=torch.zeros(token_blocks, _TOKEN_BLOCK_SIZE, cfg.index_head_dim, dtype=dtype, device=device),
                swa=swa(),
                compress_kv_state=swa(torch.float32) if states else None,
                compress_score_state=swa(torch.float32) if states else None,
            )
        )
    return caches


class _V41ForwardHarness:
    """Drives one model through ModelExecutor with synthetic paging metadata.

    The block tables follow the contract-section-2 groups: the SWA ring (two
    blocks per sequence), the TOKEN(2) pool of layer 2 and the TOKEN(1) pool
    of layer 4, one block per sequence each.
    """

    def __init__(
        self,
        lm: DeepseekV41ForCausalLM,
        config: dict,
        n_seqs: int,
        dtype: torch.dtype,
        device: torch.device | None = None,
    ) -> None:
        self.lm = lm
        self.cfg = lm.cfg
        self.n_seqs = n_seqs
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self.swa_bt = torch.arange(_SWA_RING_BLOCKS * n_seqs, dtype=torch.int32, device=self.device).view(n_seqs, -1)
        # Block 0 of the TOKEN pools is a reserved zero padding sink, so real
        # per-sequence blocks are addressed 1-based (matching production).
        self.tok2_bt = (torch.arange(n_seqs, dtype=torch.int32, device=self.device) + 1).view(n_seqs, 1)
        self.tok1_bt = (torch.arange(n_seqs, dtype=torch.int32, device=self.device) + 1).view(n_seqs, 1)
        self.layer_caches = _allocate_v41_layer_caches(
            self.cfg, n_seqs, dtype, device=self.device, reserve_token_sink=True
        )
        self.executor = ModelExecutor(lm, dict(config), max_seqs_per_batch=n_seqs)
        self.executor.bind_kv_caches(self.layer_caches)
        self.backend = self.executor.attention_backend

    def step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_lens: list[int],
        q_lens: list[int],
        *,
        is_prefill: bool,
        is_chunked_prefill: bool = False,
    ) -> torch.Tensor:
        metadata = SimpleNamespace(
            is_prefill=is_prefill,
            is_chunked_prefill=is_chunked_prefill,
            is_mixed=False,
            is_spec_verify=False,
            multi_block_tables=[self.swa_bt, self.tok2_bt, self.tok1_bt],
            kv_seq_lens_host=torch.tensor(kv_lens, dtype=torch.int32, device=_HOST_DEVICE),
            kv_seq_lens_host_values=list(kv_lens),
            q_seq_lens_host=torch.tensor(q_lens, dtype=torch.int32, device=_HOST_DEVICE),
        )
        return self.executor.execute(input_ids, positions, metadata)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.lm.compute_logits(hidden, None).float()


def _make_forward_lm(
    dtype: torch.dtype,
    seed: int,
    device: torch.device | None = None,
) -> DeepseekV41ForCausalLM:
    torch.manual_seed(seed)
    lm = DeepseekV41ForCausalLM({**_DSV41_FORWARD_CONFIG, "device": str(device or "cpu"), "dtype": dtype})
    _random_init_v41(lm)
    return lm


# ---------------------------------------------------------------------------
# Compressor math (semantics doc section 4)
# ---------------------------------------------------------------------------


def test_compressor_math() -> None:
    lm = _make_forward_lm(torch.float32, seed=11)
    cfg = lm.cfg
    model = lm.model
    backend = _make_csa2_backend(cfg)
    layer_caches = _allocate_v41_layer_caches(cfg, n_seqs=1, dtype=torch.float32)
    backend.bind_kv_caches(layer_caches)
    layer = model.layers[2].self_attn
    mapping = backend._resolve_cache_mapping(2, layer.mode.compress_ratio)
    cos_sin = model.compress_rotary.cos_sin_cache

    def build_dsa(kv_len: int, q_len: int, start: int) -> SimpleNamespace:
        return backend._builder.build(
            multi_block_tables=[
                torch.tensor([[0, 1]], dtype=torch.int32),
                torch.tensor([[0]], dtype=torch.int32),
                torch.tensor([[0]], dtype=torch.int32),
            ],
            kv_seq_lens=[kv_len],
            q_seq_lens=[q_len],
            positions=torch.arange(start, start + q_len, dtype=torch.int64),
            dsa_cos_sin=None,
            is_prefill=False,
            is_chunked_prefill=True,
        )

    def expected_key_row(latent: torch.Tensor, position: int) -> torch.Tensor:
        rotated = apply_interleaved_rope(latent.unsqueeze(0), torch.tensor([position]), cos_sin)
        return fp4_act_qdq_e4m3_scale(rotated, 16)[0]

    def expected_index_rows(latent: torch.Tensor, starts: list[int]) -> torch.Tensor:
        index_k = layer.indexer_key(latent)
        return fp4_act_qdq_e8m0(apply_interleaved_rope(index_k, torch.tensor(starts), cos_sin), 32)

    # Step 1: 3 tokens (positions 0-2). Group (0, 1) commits; position 2 is
    # parked in the state ring.
    x1 = torch.randn(3, cfg.hidden_size)
    ctx1 = Csa2LayerContext(
        x=x1,
        qr=torch.randn(3, cfg.q_lora_rank),
        positions=torch.arange(3),
        cos_sin_cache=cos_sin,
    )
    shared = CSA2SharedRuntime()
    backend._run_compressor(layer, ctx1, build_dsa(3, 3, 0), layer_caches[2], mapping, [3], [3], shared)

    kv1, score1 = layer.compress_kv_score(x1)
    gate0 = score1[:2].softmax(dim=0)
    pooled0 = (kv1[:2] * gate0).sum(dim=0)
    latent0 = layer.compress_rmsnorm(pooled0.unsqueeze(0))[0]
    key_cache = layer_caches[2].key
    assert torch.allclose(key_cache[0, 0], expected_key_row(latent0, 0), atol=1e-6)
    assert torch.allclose(layer_caches[2].index[0, 0], expected_index_rows(latent0.unsqueeze(0), [0])[0], atol=1e-6)
    # One committed row after 3 tokens (ctx // ratio); the partial group's
    # members are parked in the state ring at their SWA slots.
    assert (key_cache[0, : 3 // 2].abs().sum(dim=-1) > 0).all()
    assert (key_cache[0, 3 // 2 :] == 0).all()
    state_kv = layer_caches[2].compress_kv_state.view(-1, cfg.head_dim)
    state_score = layer_caches[2].compress_score_state.view(-1, cfg.head_dim)
    assert torch.allclose(state_kv[2], kv1[2], atol=1e-6)
    assert torch.allclose(state_score[2], score1[2], atol=1e-6)

    # Step 2: 3 more tokens (positions 3-5). Group (2, 3) completes across the
    # chunk boundary: member 2 comes from the parked state, member 3 is fresh.
    x2 = torch.randn(3, cfg.hidden_size)
    ctx2 = Csa2LayerContext(
        x=x2,
        qr=torch.randn(3, cfg.q_lora_rank),
        positions=torch.arange(3, 6),
        cos_sin_cache=cos_sin,
    )
    backend._run_compressor(layer, ctx2, build_dsa(6, 3, 3), layer_caches[2], mapping, [3], [6], shared)

    kv2, score2 = layer.compress_kv_score(x2)
    members = torch.stack((state_kv[2].clone(), kv2[0]))
    gates = torch.stack((state_score[2].clone(), score2[0])).softmax(dim=0)
    pooled1 = (members * gates).sum(dim=0)
    latent1 = layer.compress_rmsnorm(pooled1.unsqueeze(0))[0]
    gate2 = score2[1:3].softmax(dim=0)
    pooled2 = (kv2[1:3] * gate2).sum(dim=0)
    latent2 = layer.compress_rmsnorm(pooled2.unsqueeze(0))[0]
    # RMSNorm sits AFTER pooling; RoPE uses the group-start position.
    assert torch.allclose(key_cache[0, 1], expected_key_row(latent1, 2), atol=1e-6)
    assert torch.allclose(key_cache[0, 2], expected_key_row(latent2, 4), atol=1e-6)
    expected_index = expected_index_rows(torch.stack((latent1, latent2)), [2, 4])
    assert torch.allclose(layer_caches[2].index[0, 1:3], expected_index, atol=1e-6)
    assert (key_cache[0, : 6 // 2].abs().sum(dim=-1) > 0).all()
    assert (key_cache[0, 6 // 2 :] == 0).all()

    # Ratio 1 (layer 4): a plain projection, one row per token, no gate.
    layer4 = model.layers[4].self_attn
    mapping4 = backend._resolve_cache_mapping(4, layer4.mode.compress_ratio)
    x = torch.randn(5, cfg.hidden_size)
    ctx = Csa2LayerContext(
        x=x,
        qr=torch.randn(5, cfg.q_lora_rank),
        positions=torch.arange(5),
        cos_sin_cache=cos_sin,
    )
    shared4 = CSA2SharedRuntime()
    backend._run_compressor(layer4, ctx, build_dsa(5, 5, 0), layer_caches[4], mapping4, [5], [5], shared4)
    kv, _ = layer4.compress_kv_score(x)
    latent = layer4.compress_rmsnorm(kv)
    expected = fp4_act_qdq_e4m3_scale(apply_interleaved_rope(latent, torch.arange(5), cos_sin), 16)
    assert torch.allclose(layer_caches[4].key[0, :5], expected, atol=1e-6)
    assert (layer_caches[4].key[0, 5:] == 0).all()
    # Ratio 1 keeps no pooling states and no gate.
    assert not hasattr(layer4, "cmp_wgate")
    assert layer_caches[4].compress_kv_state is None
    assert layer_caches[4].compress_score_state is None


def test_compressor_padded_row_uses_nonnegative_rope_position() -> None:
    """A fully padded row (q_len=1, ctx_len=0) on a ratio-2 layer must not
    drive the compressor RoPE position negative.

    With ``ctx_len == 0`` the row is inert (``commit_mask`` all-False, every
    slot mask ``-1``), but the ratio-2 branch still feeds
    ``group_starts = (group_idx * ratio)`` to ``apply_interleaved_rope``, whose
    ``cos_sin_cache.index_select(0, positions)`` rejects the negative
    ``group_idx = floor(-1 / 2) = -1``. Only an upper clamp used to guard this,
    so the row crashed before the mask could make it a no-op; the fix clamps
    the group-start element to a valid non-negative position. The placeholder
    latent is invalid, so it must stay out of the caches.
    """
    lm = _make_forward_lm(torch.float32, seed=11)
    cfg = lm.cfg
    model = lm.model
    backend = _make_csa2_backend(cfg)
    layer_caches = _allocate_v41_layer_caches(cfg, n_seqs=1, dtype=torch.float32)
    backend.bind_kv_caches(layer_caches)
    layer = model.layers[2].self_attn
    assert layer.mode.compress_ratio == 2
    mapping = backend._resolve_cache_mapping(2, layer.mode.compress_ratio)
    cos_sin = model.compress_rotary.cos_sin_cache

    dsa = backend._builder.build(
        multi_block_tables=[
            torch.tensor([[0, 1]], dtype=torch.int32),
            torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([[0]], dtype=torch.int32),
        ],
        kv_seq_lens=[0],
        q_seq_lens=[1],
        positions=torch.arange(0, 1, dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=True,
    )
    ctx = Csa2LayerContext(
        x=torch.randn(1, cfg.hidden_size),
        qr=torch.randn(1, cfg.q_lora_rank),
        positions=torch.arange(0, 1),
        cos_sin_cache=cos_sin,
    )
    key_before = layer_caches[2].key.clone()
    index_before = layer_caches[2].index.clone()

    # Pre-fix this raised IndexError inside apply_interleaved_rope (:758).
    backend._run_compressor(layer, ctx, dsa, layer_caches[2], mapping, [1], [0], CSA2SharedRuntime())

    # The padded row is unwritten: the key/index caches are byte-identical.
    assert torch.equal(layer_caches[2].key, key_before)
    assert torch.equal(layer_caches[2].index, index_before)


# ---------------------------------------------------------------------------
# Indexer scoring (semantics doc section 5)
# ---------------------------------------------------------------------------


def test_indexer_scoring() -> None:
    # relu-weighted einsum: score[s, t] = sum_h w[s, h] * relu(q[s, h] . k[t]).
    torch.manual_seed(21)
    q = torch.randn(4, 3, 32)
    index_k = torch.randn(6, 32)
    weights = torch.randn(4, 3)
    score = _indexer_score(q, index_k, weights)
    manual = torch.einsum("shd,td->sht", q, index_k).relu_() * weights.unsqueeze(-1)
    assert torch.allclose(score, manual.sum(dim=1), atol=1e-5)

    # The per-head weights carry the (index_head_dim * index_n_heads)^-0.5 scale.
    lm = _make_forward_lm(torch.float32, seed=22)
    layer = lm.model.layers[2].self_attn
    x = torch.randn(4, lm.cfg.hidden_size)
    scale = layer.indexer.head_dim**-0.5 * layer.indexer.n_heads**-0.5
    assert torch.allclose(layer.indexer_weights(x), layer.indexer.weights_proj(x) * scale)

    # Top-k: picks are re-sorted by position (valid picks ascending, invalid
    # tail last), unreachable picks become -1, valid picks shift by the
    # window-row offset. Positions 3-4 are beyond the visibility count.
    score_row = torch.tensor([[3.0, 1.0, 2.0, float("-inf"), float("-inf")]])
    idxs = _indexer_topk(score_row, torch.tensor([3]), index_topk=4, offset=7)
    assert idxs.tolist() == [[7, 8, 9, -1]]

    # Pool-outside rows hold finite positions inside the visible range (the
    # Reindex pool mask), so a pool smaller than index_topk must drop its
    # -inf-scored picks onto the -1 sentinel instead of letting the
    # visibility check pass them.
    score_pool = torch.tensor([[3.0, float("-inf"), 1.0, 2.0, float("-inf")]])
    idxs_pool = _indexer_topk(score_pool, torch.tensor([5]), index_topk=4, offset=7)
    assert idxs_pool.tolist() == [[7, -1, 9, 10]]

    # Candidate pool: block score = max over its positions, the newest partial
    # block is pinned, unreachable picks are dropped.
    score_pool = torch.tensor(
        [[1.0, 2.0, float("-inf"), float("-inf"), 3.0, 4.0, float("-inf"), float("-inf"), float("-inf"), float("-inf")]]
    )
    pinned = _select_candidate_blocks(score_pool, torch.tensor([6]), topk_blocks=1, block_size=4)
    assert pinned.tolist() == [[False] * 4 + [True] * 4 + [False] * 2]
    both = _select_candidate_blocks(score_pool, torch.tensor([6]), topk_blocks=2, block_size=4)
    assert both.tolist() == [[True] * 8 + [False] * 2]


def test_indexer_causal_visibility_and_candidate_pool() -> None:
    lm = _make_forward_lm(torch.float32, seed=23)
    cfg = lm.cfg
    backend = _make_csa2_backend(cfg)
    cos_sin = lm.model.compress_rotary.cos_sin_cache

    # Full layer 2 (ratio 2): fresh top-k over the shared index cache. Every
    # valid pick must address a compressed group the query has fully passed
    # (t < (p + 1) // ratio) and land inside the window-row offset.
    layer2 = lm.model.layers[2].self_attn
    index_cache = torch.randn(1, _TOKEN_BLOCK_SIZE, cfg.index_head_dim) * 0.5
    shared = CSA2SharedRuntime(
        index_cache=index_cache,
        index_block_table=torch.tensor([[0]], dtype=torch.int32),
        compress_ratio=2,
    )
    q_len, ctx_len = 8, 8
    ctx = Csa2LayerContext(
        x=torch.randn(q_len, cfg.hidden_size),
        qr=torch.randn(q_len, cfg.q_lora_rank),
        positions=torch.arange(q_len),
        cos_sin_cache=cos_sin,
    )
    topk = backend._run_indexer(layer2, ctx, None, [q_len], [ctx_len], shared)
    # The fixed-capacity window is [ring candidates ; chunk], so the
    # compressed region starts after window_size + q_len rows.
    offset = cfg.window_size + q_len
    for query in range(q_len):
        visible = (query + 1) // 2
        picks = topk[0, query]
        valid = picks[picks >= 0] - offset
        assert ((valid >= 0) & (valid < visible)).all()
        assert valid.numel() == min(cfg.index_topk, visible)
    # A Full layer before the candidate source does not build the pool.
    assert shared.candidates is None

    # Candidate-source layer 4 publishes the per-sequence pool.
    layer4 = lm.model.layers[4].self_attn
    shared4 = CSA2SharedRuntime(
        index_cache=torch.randn(1, _TOKEN_BLOCK_SIZE, cfg.index_head_dim) * 0.5,
        index_block_table=torch.tensor([[0]], dtype=torch.int32),
        compress_ratio=1,
    )
    backend._run_indexer(layer4, ctx, None, [q_len], [ctx_len], shared4)
    assert shared4.candidates_layer == cfg.candidate_source_layer_id
    assert len(shared4.candidates) == 1

    # Reindex layer 5 re-scores the shared K restricted to the pool. With the
    # query past enough context that visibility no longer truncates the picks,
    # every pick the visibility filter keeps comes from the pool.
    layer5 = lm.model.layers[5].self_attn
    pool = torch.zeros(1, ctx_len, dtype=torch.bool)
    pool[0, [1, 3, 4, 6]] = True
    shared5 = CSA2SharedRuntime(
        index_cache=shared4.index_cache,
        index_block_table=shared4.index_block_table,
        compress_ratio=1,
        candidates=[pool],
    )
    ctx_late = Csa2LayerContext(
        x=torch.randn(1, cfg.hidden_size),
        qr=torch.randn(1, cfg.q_lora_rank),
        positions=torch.tensor([ctx_len - 1]),
        cos_sin_cache=cos_sin,
    )
    topk5 = backend._run_indexer(layer5, ctx_late, None, [1], [ctx_len], shared5)
    picks = topk5[0, 0]
    assert (picks[picks >= 0] - (cfg.window_size + 1)).tolist() == [1, 3, 4, 6]

    # Regression: a candidate pool smaller than
    # index_topk must not leak pool-outside rows into the picks. The pool
    # holds fewer rows than index_topk while every non-pool row still sits
    # inside the visible range, so the visibility filter alone cannot exclude
    # them -- only the -inf pool mask can, and the picks it cuts must land on
    # the -1 sentinel (which _picked_row_mask drops, so attention reads pool
    # rows + window rows only).
    backend.index_topk = 512
    topk_wide = backend._run_indexer(layer5, ctx_late, None, [1], [ctx_len], shared5)
    picks_wide = topk_wide[0, 0]
    assert picks_wide.numel() == ctx_len
    assert (picks_wide[picks_wide >= 0] - (cfg.window_size + 1)).tolist() == [1, 3, 4, 6]
    assert (picks_wide == -1).sum().item() == ctx_len - 4


def test_swa_window_lower_bound_is_per_query() -> None:
    """The SWA window a query reads is bounded per query, not per chunk.

    Regression guard for :func:`_window_visible_idxs` (the ``_csa2_attention``
    window mask): query ``p`` must see exactly the window columns at absolute
    positions ``[max(p - win + 1, 0), p]``. A chunk-level bound shared by the
    whole chunk (only query index 0 sits at the chunk start) would let every
    later query read positions older than its window.
    """
    cases = [
        # (win, prev_ctx, q_len): the first three over-admit under a chunk-level
        # bound, the last is the no-over-masking boundary at the chunk edge.
        (8, 0, 9),
        (2, 5, 3),
        (128, 200, 2),
        (8, 0, 8),
    ]
    for win, prev_ctx, q_len in cases:
        ring_pos = (prev_ctx - win) + torch.arange(win, dtype=torch.int64)
        positions_seq = prev_ctx + torch.arange(q_len, dtype=torch.int64)
        col_pos = torch.cat((ring_pos, positions_seq), dim=0)
        cols = torch.arange(col_pos.numel(), dtype=torch.int64)
        idxs = _window_visible_idxs(positions_seq, col_pos, cols, win)
        assert idxs.shape == (q_len, win + q_len)
        for i in range(q_len):
            p = prev_ctx + i
            picked = idxs[i][idxs[i] >= 0]
            got = col_pos[picked].tolist()
            expected = list(range(max(p - win + 1, 0), p + 1))
            assert got == expected, f"win={win} prev_ctx={prev_ctx} q_len={q_len} query={i}: {got} != {expected}"


# ---------------------------------------------------------------------------
# mHC final merge (semantics doc section 7)
# ---------------------------------------------------------------------------


@npu_only
def test_mhc_final_merge_is_learned_weighted_sum() -> None:
    device = _active_device()
    assert device.type == "npu"
    lm = _make_forward_lm(torch.float32, seed=31, device=device)
    assert lm.model.norm.weight.device.type == "npu"
    # Boost the hc bases so the per-copy coefficients are clearly non-uniform
    # (a near-uniform sigmoid would make the merge indistinguishable from a mean).
    with torch.no_grad():
        for layer in lm.model.layers:
            layer.hc.hc_attn_base.normal_(mean=0.0, std=1.5)
            layer.hc.hc_ffn_base.normal_(mean=0.0, std=1.5)
            layer.hc.hc_attn_scale.fill_(1.0)
            layer.hc.hc_ffn_scale.fill_(1.0)
    harness = _V41ForwardHarness(lm, _DSV41_FORWARD_CONFIG, n_seqs=1, dtype=torch.float32, device=device)
    assert harness.layer_caches[0].swa.device.type == "npu"

    captured: dict[int, dict[str, torch.Tensor]] = {}
    for layer_id in (4, 5):
        decoder_layer = lm.model.layers[layer_id]
        original = decoder_layer.forward

        def make_wrapper(orig, store):
            def wrapper(
                hidden, positions, cos_sin_cache, input_ids=None, pre_mix=None, engram_hashes=None, engram_mask=None
            ):
                out, ffn_pre = orig(hidden, positions, cos_sin_cache, input_ids, pre_mix, engram_hashes, engram_mask)
                store["hidden"] = out.clone()
                store["ffn_pre"] = ffn_pre.clone()
                return out, ffn_pre

            return wrapper

        decoder_layer.forward = make_wrapper(original, captured.setdefault(layer_id, {}))

    input_ids = torch.randint(0, lm.cfg.vocab_size, (5,)).to(device)
    with torch.inference_mode():
        hidden = harness.step(input_ids, torch.arange(5, device=device), [5], [5], is_prefill=True)

    last = captured[5]
    earlier = captured[4]
    # The learned coefficients are per-token and non-uniform.
    assert last["ffn_pre"].shape == (5, lm.cfg.hc_mult)
    assert (last["ffn_pre"].std(dim=-1) > 1e-3).all()

    with torch.inference_mode():
        hc = lm.model.layers[5].hc
        expected = lm.model.norm(hc.hc_pre(last["hidden"], last["ffn_pre"]))
        mean_merge = lm.model.norm(last["hidden"].float().mean(dim=-2))
        earlier_merge = lm.model.norm(hc.hc_pre(last["hidden"], earlier["ffn_pre"]))
    # The final collapse is the learned weighted sum with the LAST block's
    # ffn_pre -- not a mean, not an earlier block's coefficients.
    assert torch.allclose(hidden, expected, atol=1e-5)
    assert not torch.allclose(hidden, mean_merge, atol=1e-3)
    assert not torch.allclose(hidden, earlier_merge, atol=1e-3)


# ---------------------------------------------------------------------------
# Forward smoke + prefill-vs-decode consistency (milestone gate)
# ---------------------------------------------------------------------------


@npu_only
def test_forward_smoke_and_prefill_decode_consistency() -> None:
    device = _active_device()
    assert device.type == "npu"
    lm = _make_forward_lm(torch.bfloat16, seed=41, device=device)
    assert lm.model.norm.weight.device.type == "npu"
    cfg = lm.cfg
    # A prompt one token past the window keeps every query of the prefill chunk
    # inside the window (prompt_len == window_size is the boundary that a
    # chunk-level window bound would still mask identically).
    n_seqs, prompt_len, decode_steps = 2, cfg.window_size + 1, 4
    harness = _V41ForwardHarness(lm, _DSV41_FORWARD_CONFIG, n_seqs, torch.bfloat16, device=device)
    assert harness.layer_caches[0].swa.device.type == "npu"
    input_ids = torch.randint(0, cfg.vocab_size, (n_seqs, prompt_len)).to(device)

    # -- Prefill: the whole 18-token prompt in one batch (2 seqs x 9 tokens).
    with torch.inference_mode():
        hidden = harness.step(
            input_ids.reshape(-1),
            torch.arange(prompt_len, device=device).repeat(n_seqs),
            [prompt_len] * n_seqs,
            [prompt_len] * n_seqs,
            is_prefill=True,
        )
    assert hidden.shape == (n_seqs * prompt_len, cfg.hidden_size)
    assert torch.isfinite(hidden.float()).all()
    prefill_logits = harness.logits(hidden)
    assert prefill_logits.shape == (n_seqs * prompt_len, cfg.vocab_size)
    assert torch.isfinite(prefill_logits).all()

    # SWA ring writes: positions 0-7 fill the first ring block of each
    # sequence; position 8 opens the second ring block.
    swa = harness.layer_caches[0].swa
    assert (swa[0].abs().sum(dim=-1) > 0).all()
    assert (swa[2].abs().sum(dim=-1) > 0).all()
    assert swa[1, 0].abs().sum(dim=-1) > 0
    assert (swa[1, 1:] == 0).all()
    assert swa[3, 0].abs().sum(dim=-1) > 0
    assert (swa[3, 1:] == 0).all()

    # Committed compressed rows = ctx // ratio for both kv sources. Block 0 is
    # the reserved zero padding sink, so sequence 0's real block is index 1.
    key2 = harness.layer_caches[2].key
    key4 = harness.layer_caches[4].key
    assert (key2[1, : prompt_len // 2].abs().sum(dim=-1) > 0).all()
    assert (key2[1, prompt_len // 2 :] == 0).all()
    assert (key4[1, :prompt_len].abs().sum(dim=-1) > 0).all()
    assert (key4[1, prompt_len:] == 0).all()

    # Reuse layer 3 consumes the group source's top-k object as-is.
    assert lm.model.layers[3].self_attn._csa2_used_topk is lm.model.layers[2].self_attn._csa2_used_topk
    # The candidate pool is published by layer 4 only, and the reindex layer's
    # picks stay inside the pool.
    shared = harness.backend._v41_shared
    assert shared.candidates_layer == cfg.candidate_source_layer_id
    topk5 = lm.model.layers[5].self_attn._csa2_used_topk
    # Layer 5 (ratio 1) ran with q_len = prompt_len, so its compressed picks
    # start after window_size + prompt_len window rows.
    compress_offset = cfg.window_size + prompt_len
    for seq in range(n_seqs):
        pool = shared.candidates[seq]
        for query in range(prompt_len):
            valid = topk5[seq, query] - compress_offset
            for value in valid[valid >= 0].tolist():
                assert pool[query, value]

    # -- Decode: 4 single-token steps per sequence continue the same caches.
    decode_ids = torch.randint(0, cfg.vocab_size, (n_seqs, decode_steps)).to(device)
    for step in range(decode_steps):
        position = prompt_len + step
        with torch.inference_mode():
            step_hidden = harness.step(
                decode_ids[:, step],
                torch.full((n_seqs,), position, device=device),
                [position + 1] * n_seqs,
                [1] * n_seqs,
                is_prefill=False,
            )
        assert step_hidden.shape == (n_seqs, cfg.hidden_size)
        assert torch.isfinite(step_hidden.float()).all()
    # Ring writes for positions 8-12 fill the second ring block rows 0-4
    # (position 8 from the prefill, 9-12 from the decode steps).
    second_block_rows = prompt_len - cfg.window_size + decode_steps
    assert (swa[1, :second_block_rows].abs().sum(dim=-1) > 0).all()
    assert (swa[1, second_block_rows:] == 0).all()
    assert (swa[3, :second_block_rows].abs().sum(dim=-1) > 0).all()
    # Committed rows follow ctx // ratio at ctx = prompt_len + decode_steps.
    # Block 0 is the reserved zero padding sink; sequence 0's block is index 1.
    ctx_len = prompt_len + decode_steps
    assert (key2[1, : ctx_len // 2].abs().sum(dim=-1) > 0).all()
    assert (key2[1, ctx_len // 2 :] == 0).all()
    assert (key4[1, :ctx_len].abs().sum(dim=-1) > 0).all()
    assert (key4[1, ctx_len:] == 0).all()

    # -- Consistency: the same prompt re-run token by token as decode steps
    # with fresh caches reproduces the prefill logits.
    replay = _V41ForwardHarness(lm, _DSV41_FORWARD_CONFIG, n_seqs, torch.bfloat16, device=device)
    replay_logits = []
    for position in range(prompt_len):
        with torch.inference_mode():
            step_hidden = replay.step(
                input_ids[:, position],
                torch.full((n_seqs,), position, device=device),
                [position + 1] * n_seqs,
                [1] * n_seqs,
                is_prefill=False,
            )
        replay_logits.append(replay.logits(step_hidden))
    for position in range(prompt_len):
        for seq in range(n_seqs):
            expected = prefill_logits[seq * prompt_len + position]
            diff = (replay_logits[position][seq] - expected).abs().max().item()
            assert diff < 2e-2, f"decode replay diverged at position {position} of sequence {seq}: {diff}"


# ---------------------------------------------------------------------------
# Engram host offload (engram_config.cpu_offload)
# ---------------------------------------------------------------------------


@pytest.fixture
def _engram_offload_lookup_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the CPU lookup op available to the offload cases when it is absent.

    The offload dequant path (:meth:`EngramEmbedding._local_block_offload`)
    always calls the in-repo ``xllm_ops.engram_int8_lookup_cpu`` op and fails
    loudly when it is missing (no eager fallback). A standalone pytest process
    never links the xLLM binary, so every offload case here would otherwise
    raise. This fixture substitutes the bitwise-equivalent eager reference
    (:func:`_reference_engram_lookup`) for the duration of each case and
    restores the real resolver afterwards via ``monkeypatch``.

    Scope and safety (explicit, not ``autouse``): requesting this fixture by
    name keeps its patch confined to the offload cases that opt in, so it can
    never fire for an unrelated case in this module or any other -- the concern
    about an import-time side effect. The patch is also
    scoped to the offload cases only, and only when the real op is missing:

    * It lives in this file, not ``conftest.py``; the fail-loud contract case
      ``test_deepseek_v41_engram.py::test_engram_lookup_op_missing_is_loud`` --
      a different module -- still sees the unpatched resolver and asserts the
      ``RuntimeError``. No op-missing assertion lives in this module.
    * When the real op *is* registered (a full xLLM build) the probe succeeds
      and the fixture is a **no-op**, so those runs keep exercising the real
      kernel instead of the reference.
    """
    import xllm.python.models.deepseek_v41_engram as engram

    try:
        engram._get_engram_lookup_op()
    except RuntimeError:
        monkeypatch.setattr(engram, "_get_engram_lookup_op", lambda: _reference_engram_lookup)


def _int8_engram_payload(head_sizes: list[int], dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_rows = sum(head_sizes)
    g = torch.Generator().manual_seed(seed)
    payload = torch.randint(-127, 128, (num_rows, dim), dtype=torch.int8, generator=g)
    scales = torch.rand(num_rows, dim // 32, generator=g).float() * 0.05 + 0.01
    return payload, scales


def _make_int8_embedding(
    head_sizes: list[int],
    dim: int,
    payload: torch.Tensor,
    scales: torch.Tensor,
    *,
    tp_size: int = 1,
    tp_rank: int = 0,
    offload: bool = False,
) -> EngramEmbedding:
    emb = EngramEmbedding(
        sum(head_sizes),
        dim,
        head_sizes,
        tp_size,
        tp_rank,
        storage=ENGRAM_STORAGE_INT8,
        device=torch.device("cpu"),
        offload=offload,
    )
    with torch.no_grad():
        emb.weight.copy_(payload[emb.vocab_start : emb.vocab_end])
        emb.weight_scale.copy_(scales[emb.vocab_start : emb.vocab_end])
    return emb


def _engram_ids(head_sizes: list[int], num_tokens: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    ids = torch.zeros(num_tokens, len(head_sizes), dtype=torch.int64)
    for j, size in enumerate(head_sizes):
        ids[:, j] = torch.randint(0, size, (num_tokens,), generator=g) + sum(head_sizes[:j])
    return ids


# ---------------------------------------------------------------------------
# Engram graph-visible staging stability: the graph block address must not move
# ---------------------------------------------------------------------------


def test_engram_graph_staging_address_stable_across_eager_growth(_engram_offload_lookup_op) -> None:
    """The graph-visible block never moves, even when the eager path grows."""
    head_sizes = [7, 11]
    payload, scales = _int8_engram_payload(head_sizes, 32, 11)
    off = _make_int8_embedding(head_sizes, 32, payload, scales, offload=True)

    ids = _engram_ids(head_sizes, 4, 1)
    graph_rows = off.local_block(ids, graph_visible=True).clone()
    graph_ptr = off._graph_staged_device.data_ptr()
    graph_capacity = off._graph_capacity

    # A much larger eager pass allocates/moves its own block...
    eager_rows = off.local_block(_engram_ids(head_sizes, 300, 2), graph_visible=False)
    assert eager_rows.shape[0] == 300
    assert off._eager_capacity >= 300
    # ...and leaves the graph-visible block (address and capacity) untouched.
    assert off._graph_staged_device.data_ptr() == graph_ptr
    assert off._graph_capacity == graph_capacity

    again = off.local_block(ids, graph_visible=True)
    assert again.data_ptr() == graph_ptr
    assert torch.equal(again, graph_rows)
    # Both staging blocks dequantize bitwise identically.
    assert torch.equal(off.local_block(ids, graph_visible=False), graph_rows)


def test_engram_graph_staging_reserves_the_capacity_hint(_engram_offload_lookup_op) -> None:
    """A runner-supplied bucket hint pre-sizes the block for every later bucket.

    Regression: capture is lazy and buckets warm up ascending, so
    without the hint the block would be fixed at the first (smallest) bucket and
    a larger decode bucket would raise. With the hint the first allocation
    already covers the largest bucket.
    """
    head_sizes = [7, 11]
    payload, scales = _int8_engram_payload(head_sizes, 32, 12)
    off = _make_int8_embedding(head_sizes, 32, payload, scales, offload=True)

    off.set_graph_capacity_hint(512)
    small = off.local_block(_engram_ids(head_sizes, 4, 1), graph_visible=True)
    ptr = off._graph_staged_device.data_ptr()
    assert off._graph_capacity == 512  # the hint, not the 256 floor / 4 rows

    # The later, larger bucket stages into the same pre-sized block.
    larger = off.local_block(_engram_ids(head_sizes, 300, 2), graph_visible=True)
    assert larger.shape[0] == 300
    assert off._graph_staged_device.data_ptr() == ptr
    assert off._graph_capacity == 512
    assert not torch.equal(larger, small)  # not a stale prefix read


def test_engram_graph_staging_fails_loud_beyond_captured_capacity(_engram_offload_lookup_op) -> None:
    """Beyond even the hint the block still raises instead of moving the address."""
    head_sizes = [7, 11]
    payload, scales = _int8_engram_payload(head_sizes, 32, 12)
    off = _make_int8_embedding(head_sizes, 32, payload, scales, offload=True)

    off.set_graph_capacity_hint(512)
    _ = off.local_block(_engram_ids(head_sizes, 4, 1), graph_visible=True)
    ptr = off._graph_staged_device.data_ptr()
    capacity = off._graph_capacity
    assert capacity == 512

    with pytest.raises(RuntimeError, match="graph-visible staging capacity exceeded"):
        off.local_block(_engram_ids(head_sizes, 600, 2), graph_visible=True)
    # The refusal must not have moved or freed the captured address.
    assert off._graph_staged_device.data_ptr() == ptr
    assert off._graph_capacity == capacity
    # The eager path is unaffected by the refusal.
    assert off.local_block(_engram_ids(head_sizes, 600, 3), graph_visible=False).shape[0] == 600


def test_engram_graph_staging_matches_device_path(_engram_offload_lookup_op) -> None:
    """Rows staged into the graph-visible block equal the device-path rows."""
    head_sizes = [7, 11, 13, 5, 3]
    dim = 32
    payload, scales = _int8_engram_payload(head_sizes, dim, 13)
    ref = _make_int8_embedding(head_sizes, dim, payload, scales)
    off = _make_int8_embedding(head_sizes, dim, payload, scales, offload=True)
    ids = _engram_ids(head_sizes, 9, 14)
    assert torch.equal(off.local_block(ids, graph_visible=True), ref.local_block(ids))


def test_engram_graph_replay_survives_eager_prefill(_engram_offload_lookup_op) -> None:
    """A large eager prefill must not disturb what a graph replay reads."""
    layout = EngramLayout(
        layer_ids=[1],
        num_embeddings=[290],
        max_ngram_size=3,
        n_heads=2,
        head_dim=32,
        vocab_size=64,
        compressed_vocab_size=16,
        pad_token_id=0,
    )
    dim, hc_mult = 48, 2
    off = _make_int8_engram(layout, dim, hc_mult, offload=True)

    replay_ids = torch.zeros(4, layout.n_hash_cols, dtype=torch.int64)
    off.prepare_embeddings(replay_ids, graph_visible=True)
    graph_ptr = off.embed_tokens._graph_staged_device.data_ptr()
    replay_rows = off.embed(None).clone()

    # A 300-row eager prefill uses the eager block and leaves the graph one alone.
    big = torch.zeros(300, layout.n_hash_cols, dtype=torch.int64)
    off.prepare_embeddings(big, graph_visible=False)
    assert off.embed_tokens._graph_staged_device.data_ptr() == graph_ptr

    # The runner re-stages before every replay; the replay reads the same rows.
    off.prepare_embeddings(replay_ids, graph_visible=True)
    assert torch.equal(off.embed(None), replay_rows)


def _graph_forward_context() -> ForwardContext:
    """A ``ForwardContext`` that makes ``in_acl_graph()`` true (graph warmup)."""
    return ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=None,
        layer_caches=[],
        execution_state=AclGraphExecutionState({}),
    )


def test_engram_graph_visible_forward_keeps_the_graph_block(_engram_offload_lookup_op) -> None:
    """Warmup's ``embed(hash_ids)`` must not re-mount onto the eager block.

    Audit reproduction chain: runner pre-graph staging -> warmup layer call
    ``embed(hash_ids)`` under ``in_acl_graph()`` -> capture ``embed(None)``.
    Before the fix ``embed(hash_ids)`` unconditional re-mounted
    ``_staged_local`` to the eager block, so the capture read -- and the
    captured graph baked -- the eager address, which a long-context eager
    prefill then moves.
    """
    layout = _engram_test_layout(layer_id=1)
    off = _make_int8_engram(layout, 48, 2, offload=True)
    ids = torch.zeros(4, layout.n_hash_cols, dtype=torch.int64)

    with forward_context(_graph_forward_context()):
        off.prepare_embeddings(ids, graph_visible=True)  # runner pre-graph staging
        graph_ptr = off.embed_tokens._graph_staged_device.data_ptr()
        _ = off.embed(ids)  # warmup decoder layer call (hash_ids is not None)
        capture_rows = off.embed(None)  # capture read
        assert capture_rows.data_ptr() == graph_ptr
        assert off._staged_local_graph is True
        assert (
            off._staged_local.untyped_storage().data_ptr()
            == off.embed_tokens._graph_staged_device.untyped_storage().data_ptr()
        )

    # A long-context eager prefill grows its own block, never the graph's.
    off.prepare_embeddings(torch.zeros(300, layout.n_hash_cols, dtype=torch.int64), graph_visible=False)
    assert off.embed_tokens._eager_staged_device.shape[0] >= 300
    assert off.embed_tokens._graph_staged_device.data_ptr() == graph_ptr

    # Every replay re-stages into the graph block and keeps reading it.
    with forward_context(_graph_forward_context()):
        off.prepare_embeddings(ids, graph_visible=True)
        assert off.embed(None).data_ptr() == graph_ptr
        assert off.embed(ids).data_ptr() == graph_ptr


def test_engram_eager_forward_after_graph_staging_uses_the_eager_block(_engram_offload_lookup_op) -> None:
    """An eager forward must not read the graph-visible block."""
    layout = _engram_test_layout(layer_id=1)
    off = _make_int8_engram(layout, 48, 2, offload=True)
    ids = torch.zeros(4, layout.n_hash_cols, dtype=torch.int64)

    with forward_context(_graph_forward_context()):
        off.prepare_embeddings(ids, graph_visible=True)
        _ = off.embed(ids)
        assert off._staged_local_graph is True

    # Outside the graph the same layer call falls back to the eager block.
    assert off.embed(ids).data_ptr() == off.embed_tokens._eager_staged_device.data_ptr()
    assert off._staged_local_graph is False
    assert (
        off._staged_local.untyped_storage().data_ptr()
        == off.embed_tokens._eager_staged_device.untyped_storage().data_ptr()
    )


# ---------------------------------------------------------------------------
# Engram test fixtures
# ---------------------------------------------------------------------------


def _engram_test_layout(layer_id: int) -> EngramLayout:
    return EngramLayout(
        layer_ids=[layer_id],
        num_embeddings=[290],
        max_ngram_size=3,
        n_heads=2,
        head_dim=32,
        vocab_size=64,
        compressed_vocab_size=16,
        pad_token_id=0,
    )


def test_close_host_offload_releases_every_layer_despite_one_failure() -> None:
    """A single layer's close failure must not leak the sibling layers' tables."""
    closed: list[str] = []

    class _Engram:
        def __init__(self, name: str, *, boom: bool = False) -> None:
            self._name = name
            self._boom = boom

        def close_host_offload(self) -> None:
            if self._boom:
                raise RuntimeError(f"{self._name} refuses to close")
            closed.append(self._name)

    class _Layer:
        def __init__(self, engram: object | None) -> None:
            self.engram = engram

    model = type("_Model", (), {})()
    model.layers = [
        _Layer(_Engram("a")),
        _Layer(_Engram("b", boom=True)),
        _Layer(_Engram("c")),
        _Layer(None),  # a non-engram layer is skipped
    ]
    with pytest.raises(RuntimeError, match="layer 1"):
        dq.DeepseekV41Model.close_host_offload(model)
    # The failing layer is reported, but the layers on both sides still closed.
    assert closed == ["a", "c"]


def test_engram_fallback_to_device_refused_after_graph_staging() -> None:
    """``fallback_to_device`` may not run once the graph block is allocated.

    The block's address is baked into a captured ACL graph
    (``_ensure_graph_staging``), so demoting the table to HBM after capture
    would leave every replay reading a freed/reused address. Both production
    call sites (loading-time fallback, post-load reconcile) run before capture,
    so the guard is never expected to fire there -- pinned by the happy path
    below.
    """
    head_sizes = [5, 7]
    dim = 32
    payload, scales = _int8_engram_payload(head_sizes, dim, 6)

    # Happy path: a fresh offload table (no graph block) demotes cleanly.
    fresh = _make_int8_embedding(head_sizes, dim, payload, scales, offload=True)
    assert fresh.fallback_to_device() is True
    assert fresh.offload is False

    # Once a graph-visible block exists the demotion is refused.
    captured = _make_int8_embedding(head_sizes, dim, payload, scales, offload=True)
    captured.set_graph_capacity_hint(16)
    captured._ensure_graph_staging(4)
    assert captured._graph_staged_device is not None
    ptr = captured._graph_staged_device.data_ptr()
    with pytest.raises(RuntimeError, match="before ACL-graph capture"):
        captured.fallback_to_device()
    # The refusal must not have disturbed the captured block.
    assert captured.offload is True
    assert captured._graph_staged_device.data_ptr() == ptr


def test_engram_private_host_copy_failure_logs_traceback_and_falls_back() -> None:
    """A host-copy RuntimeError demotes the layer to HBM and logs a traceback.

    ``_load_engram_ascend_private`` catches ``RuntimeError`` / ``MemoryError``
    -- the two host-copy failure shapes -- and falls back to the HBM table.
    Because ``RuntimeError`` is wider than a host OOM (a weight-path or shape
    defect raises it too), the warning must carry the full traceback via
    ``exc_info=True`` so the failing copy is identifiable; the HBM copy then
    runs and the layer is demoted. If that second copy also fails the loader
    aborts explicitly instead of serving a half-populated table.

    Discriminating: dropping ``exc_info=True`` leaves ``record.exc_info`` None
    and the rendered text traceback-free; removing ``fallback_to_device()``
    leaves ``embed.fallback_calls == 0`` and the retried copy count at 1.
    """

    class _Embed:
        """Minimal stand-in: only ``offload`` / ``fallback_to_device`` are read."""

        def __init__(self) -> None:
            self.offload = True
            self.fallback_calls = 0

        def fallback_to_device(self) -> bool:
            self.fallback_calls += 1
            self.offload = False
            return True

    table = torch.zeros(1, 32, dtype=torch.int8)
    scale = torch.zeros(1, 1, dtype=torch.float32)

    def make_self(copy_rows) -> SimpleNamespace:
        return SimpleNamespace(
            _load_engram_ascend_table=lambda *args, **kwargs: (table, scale, 0, 1),
            _copy_engram_rows=copy_rows,
        )

    xllm_logger = logging.getLogger("xllm")
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture(level=logging.WARNING)
    xllm_logger.addHandler(handler)
    try:
        # First copy (host) fails, the fallback copy (HBM) succeeds.
        embed = _Embed()
        copies = {"count": 0}

        def _fail_first(_embed, _table, _scale, _start, _rows) -> None:
            copies["count"] += 1
            if copies["count"] == 1:
                raise RuntimeError("simulated host copy failure")

        dq.DeepseekV41ForCausalLM._load_engram_ascend_private(make_self(_fail_first), None, "ck.", 7, embed)

        # Both copies fail: abort rather than run on a half-loaded table.
        embed2 = _Embed()

        def _always_fail(_embed, _table, _scale, _start, _rows) -> None:
            raise RuntimeError("simulated HBM copy failure")

        with pytest.raises(RuntimeError, match="partially populated"):
            dq.DeepseekV41ForCausalLM._load_engram_ascend_private(make_self(_always_fail), None, "ck.", 9, embed2)
    finally:
        xllm_logger.removeHandler(handler)

    assert copies["count"] == 2, "the fallback must re-run the copy on the HBM path"
    assert embed.fallback_calls == 1
    assert embed.offload is False
    assert embed2.fallback_calls == 1

    warnings = [
        record
        for record in records
        if "host-table load failed" in record.getMessage() and "layer 7" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert warnings[0].exc_info is not None, "the wide catch must keep the traceback"
    rendered = logging.Formatter().format(warnings[0])
    assert "Traceback" in rendered and "simulated host copy failure" in rendered


def _fake_v41_layers(*engrams: Engram | None) -> list[SimpleNamespace]:
    """Minimal ``DeepseekV41DecoderLayer`` stand-ins carrying only ``engram``."""
    return [SimpleNamespace(engram=engram) for engram in engrams]


def test_engram_offload_reconcile_demotes_all_layers_on_fallback() -> None:
    """One demoted layer must pull the whole model onto the HBM path."""
    layout = _engram_test_layout(layer_id=1)
    off_a = _make_int8_engram(layout, 48, 2, offload=True)
    off_b = _make_int8_engram(layout, 48, 2, offload=True)
    # Simulate ``_load_engram_ascend`` demoting a single layer after a failed
    # host-table copy.
    assert off_b.embed_tokens.fallback_to_device() is True
    assert off_b.offload is False

    fake = SimpleNamespace(
        engram_offload_active=True, dp_shared_memory=False, layers=_fake_v41_layers(off_a, off_b, None)
    )
    assert dq.DeepseekV41Model._reconcile_engram_offload_active(fake) is True
    # All-or-nothing: the model flag is cleared AND the remaining offload layer
    # is demoted, so no layer keeps the (capture-unsafe) host path.
    assert fake.engram_offload_active is False
    assert off_a.offload is False
    assert off_b.offload is False


def test_engram_offload_reconcile_skips_already_demoted_layer() -> None:
    """The layer demoted by the loader must not be demoted a second time.

    ``_load_engram_ascend`` demotes the failed layer before reconcile runs, so
    reconcile must skip it (its ``offload`` is already False) and only demote
    the layers that still offload. Discriminating: without the ``offload``
    guard ``fallback_to_device`` runs on both layers, so the call log is
    ``["a", "b"]`` instead of ``["a"]``.
    """
    layout = _engram_test_layout(layer_id=1)
    off_a = _make_int8_engram(layout, 48, 2, offload=True)
    off_b = _make_int8_engram(layout, 48, 2, offload=True)
    # The failed layer is already demoted by the loader.
    assert off_b.embed_tokens.fallback_to_device() is True
    assert off_b.offload is False

    calls: list[str] = []
    real_a = off_a.embed_tokens.fallback_to_device
    real_b = off_b.embed_tokens.fallback_to_device

    def spy_a() -> bool:
        calls.append("a")
        return real_a()

    def spy_b() -> bool:
        calls.append("b")
        return real_b()

    off_a.embed_tokens.fallback_to_device = spy_a
    off_b.embed_tokens.fallback_to_device = spy_b

    fake = SimpleNamespace(
        engram_offload_active=True, dp_shared_memory=False, layers=_fake_v41_layers(off_a, off_b, None)
    )
    assert dq.DeepseekV41Model._reconcile_engram_offload_active(fake) is True
    assert calls == ["a"], "only the still-offloading layer may be demoted"
    assert off_a.offload is False
    assert fake.engram_offload_active is False


def test_engram_offload_reconcile_keeps_offload_when_all_layers_held() -> None:
    """No per-layer fallback: the model keeps host offload for every layer."""
    layout = _engram_test_layout(layer_id=1)
    off_a = _make_int8_engram(layout, 48, 2, offload=True)
    fake = SimpleNamespace(engram_offload_active=True, dp_shared_memory=False, layers=_fake_v41_layers(off_a, None))
    assert dq.DeepseekV41Model._reconcile_engram_offload_active(fake) is False
    assert fake.engram_offload_active is True
    assert off_a.offload is True
    # Inactive models are never touched.
    inactive = SimpleNamespace(engram_offload_active=False, dp_shared_memory=False, layers=_fake_v41_layers(off_a))
    assert dq.DeepseekV41Model._reconcile_engram_offload_active(inactive) is False


def test_engram_offload_reconcile_skips_demotion_for_dp_shared_memory() -> None:
    """A DP-shared table never enters the per-layer demotion path.

    With ``dp_shared_memory`` set the shared host load is
    fail-fast, so no per-layer fallback exists to reconcile. Reconcile must
    return False and leave every layer offloading -- not even a layer that
    already fell back may drag the shared model onto the HBM path.
    Discriminating: without the ``or self.dp_shared_memory`` guard the mixed
    model would return True and demote ``off_a``, so the spy call log is
    ``["a"]`` instead of ``[]``.
    """
    layout = _engram_test_layout(layer_id=1)
    off_a = _make_int8_engram(layout, 48, 2, offload=True)
    off_b = _make_int8_engram(layout, 48, 2, offload=True)
    # Simulate a shared layer whose host table is already gone (offload False).
    assert off_b.embed_tokens.fallback_to_device() is True
    assert off_b.offload is False

    calls: list[str] = []
    real_a = off_a.embed_tokens.fallback_to_device
    real_b = off_b.embed_tokens.fallback_to_device

    def spy_a() -> bool:
        calls.append("a")
        return real_a()

    def spy_b() -> bool:
        calls.append("b")
        return real_b()

    off_a.embed_tokens.fallback_to_device = spy_a
    off_b.embed_tokens.fallback_to_device = spy_b

    fake = SimpleNamespace(
        engram_offload_active=True,
        dp_shared_memory=True,
        layers=_fake_v41_layers(off_a, off_b, None),
    )
    assert dq.DeepseekV41Model._reconcile_engram_offload_active(fake) is False
    assert calls == [], "a DP-shared table must not trigger any per-layer demotion"
    assert fake.engram_offload_active is True
    assert off_a.offload is True


def test_engram_fallback_layer_serves_embed_none_from_staged_rows() -> None:
    """A demoted layer resolves the graph contract (None ids).

    Reproduces the review chain: ``Engram(offload=True)`` ->
    ``fallback_to_device()`` -> graph-visible ``prepare_embeddings`` (writes
    ``_staged_rows``) -> ``embed(None)``. Pre-fix this raised ``no staged rows``;
    post-fix it returns the device-path rows bit-identically.
    """
    layout = _engram_test_layout(layer_id=1)
    dim, hc_mult = 48, 2
    off = _make_int8_engram(layout, dim, hc_mult, offload=True)
    dev = _make_int8_engram(layout, dim, hc_mult, offload=False)
    with torch.no_grad():
        dev.embed_tokens.weight.copy_(off.embed_tokens.weight)
        dev.embed_tokens.weight_scale.copy_(off.embed_tokens.weight_scale)

    assert off.embed_tokens.fallback_to_device() is True
    assert off.offload is False

    ids = torch.zeros(6, layout.n_hash_cols, dtype=torch.int64)
    off.prepare_embeddings(ids, graph_visible=True)
    assert off._staged_local is None
    assert off._staged_rows is not None

    replay_rows = off.embed(None)
    dev.prepare_embeddings(ids)
    assert torch.equal(replay_rows, dev._staged_rows)


def test_engram_mixed_mode_embed_none_checks_staged_row_count() -> None:
    """The mixed-mode ``embed(None)`` branch refuses a wrong-sized staging block.

    That branch (``_staged_rows`` set while ``hash_ids is None``) is
    production-unreachable today -- the all-or-nothing reconcile clears
    ``engram_offload_active`` so the capture never passes ``None`` to a
    fallback layer -- but if a future change revives it, silently returning a
    block sized for a different batch would feed the graph the wrong rows.
    ``forward`` passes the step's token count, so the mismatch is refused; the
    pre-existing ``embed(None)`` callers (no count) keep working.
    """
    layout = _engram_test_layout(layer_id=1)
    eng = _make_int8_engram(layout, 48, 2, offload=False)
    ids = torch.zeros(6, layout.n_hash_cols, dtype=torch.int64)
    eng.prepare_embeddings(ids)
    assert eng._staged_rows is not None and eng._staged_rows.shape[0] == 6

    # Matching token count resolves; a bare ``embed(None)`` (no count) too.
    assert eng.embed(None, num_tokens=6).shape[0] == 6
    assert eng.embed(None).shape[0] == 6

    # A mismatched step token count is refused instead of returning stale rows.
    with pytest.raises(RuntimeError, match="row mismatch"):
        eng.embed(None, num_tokens=3)


def test_engram_embed_re_gathers_for_a_shorter_batch(_engram_offload_lookup_op) -> None:
    """A smaller explicit batch must re-gather, and
    the case must actually discriminate the shape judge.

    ``prepare_embeddings`` and ``embed`` run in the SAME graph-visibility regime
    (the default eager one: ``in_acl_graph()`` is False under plain pytest). If
    ``prepare`` staged with ``graph_visible=True`` while ``embed`` computed
    ``graph_visible=False``, the ``_staged_local_graph != graph_visible`` term
    would be trivially true and force a re-gather regardless of the shape judge
    -- the buggy ``<`` judge would then still pass, leaving the case with no
    discriminating power. Keeping both undirected leaves the shape judge as the
    only trigger, and the direct ``_staged_local`` assertion pins its outcome.
    """
    layout = _engram_test_layout(layer_id=1)
    off = _make_int8_engram(layout, 48, 2, offload=True)
    g = torch.Generator().manual_seed(31)
    long_ids = torch.randint(0, layout.num_embeddings[0], (8, layout.n_hash_cols), generator=g)
    short_ids = torch.randint(0, layout.num_embeddings[0], (3, layout.n_hash_cols), generator=g)

    off.prepare_embeddings(long_ids)
    assert off._staged_local is not None and off._staged_local.shape[0] == 8

    # A shorter explicit call re-gathers instead of returning the 8-row prefix.
    short_rows = off.embed(short_ids)
    assert short_rows.shape[0] == 3
    # Direct consequence of the judge: the staged block itself is rebuilt at 3
    # rows. With the buggy ``<`` judge (``8 < 3`` is False) no re-gather happens
    # and this still holds the 8-row prefix, so the assertion fails there.
    assert off._staged_local.shape[0] == 3
    # Content-level evidence: the rows come from re-gathering ``short_ids``, not
    # the first 3 rows of the ``long_ids`` gather.
    ref = _make_int8_engram(layout, 48, 2, offload=True)
    ref.prepare_embeddings(short_ids)
    assert torch.equal(short_rows, ref.embed(None))


class _EngramRoutingAttnStub(nn.Module):
    """Shape-preserving attention stub so the post-engram path runs on CPU."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((hidden_size,), 0.5), requires_grad=False)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        return hidden * self.scale


class _EngramRoutingMlpStub(nn.Module):
    """Dense-MLP stub: not ``DeepseekV41MoE``, so forward takes one argument."""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden * 0.25


def _make_engram_routing_decoder_layer() -> DeepseekV41DecoderLayer:
    """A real ``DeepseekV41DecoderLayer`` with a real offloaded ``Engram``.

    The engram routing guard is the code under test, so the layer is real and
    only its attention / MLP -- which need the NPU CSA2 backend -- are stubbed
    with shape-preserving CPU modules. ``hc_*`` parameters start as
    ``torch.empty`` in the constructor, so they are seeded with finite values.
    """
    cfg = DeepseekV41Config.from_dict(_DSV41_TINY_CONFIG)
    layout = EngramLayout(
        layer_ids=[1],
        num_embeddings=[290],
        max_ngram_size=3,
        n_heads=2,
        head_dim=32,
        vocab_size=64,
        compressed_vocab_size=16,
        pad_token_id=0,
    )
    layer = DeepseekV41DecoderLayer(
        cfg,
        1,
        torch.float32,
        torch.device("cpu"),
        CSA2LayerMode(layer_id=1, mode=CSA2_MODE_SWA, compress_ratio=0, main_kv_source_layer=-1),
        engram_layout=layout,
        engram_storage=ENGRAM_STORAGE_INT8,
        engram_offload=True,
    )
    eng = layer.engram
    assert eng is not None
    g = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for name in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base"):
            getattr(layer.hc, name).normal_(0.0, 0.02)
        for name in ("hc_attn_scale", "hc_ffn_scale"):
            getattr(layer.hc, name).fill_(1.0)
        payload = torch.randint(-127, 128, eng.embed_tokens.weight.shape, dtype=torch.int8, generator=g)
        scales = torch.rand(eng.embed_tokens.weight_scale.shape, generator=g).float() * 0.05 + 0.01
        eng.embed_tokens.weight.copy_(payload)
        eng.embed_tokens.weight_scale.copy_(scales)
        eng.wkv.weight.normal_(0.0, 0.05)
        eng.q_weight.normal_(0.0, 0.05)
        eng.k_weight.normal_(0.0, 0.05)
    layer.self_attn = _EngramRoutingAttnStub(cfg.hidden_size)
    layer.mlp = _EngramRoutingMlpStub()
    return layer


def test_decoder_layer_routes_none_hashes_to_staged_rows(_engram_offload_lookup_op) -> None:
    """The offloaded capture/replay call -- ``engram_hashes=None`` plus
    a computed mask -- must still inject engram, reading the pre-graph staged
    rows.

    That guard is the capture production entry: the host gather ran outside the
    graph, so a captured graph passes ``hash_ids=None`` and only the mask. The
    old guard (``engram_hashes is not None``) silently dropped the injection for
    that call, so the capture read nothing and every replay diverged from eager.
    Both halves are pinned here: the mask-only call injects and is bit-identical
    to the explicit-hashes call, while ``(None, None)`` truly skips engram (equal
    to the engram-disabled path).
    """
    layer = _make_engram_routing_decoder_layer()
    eng = layer.engram
    assert eng is not None and eng.offload is True

    hidden_size = layer.input_layernorm.weight.numel()
    hc_mult = layer.hc.hc_mult
    t = 5
    g = torch.Generator().manual_seed(29)
    hashes = torch.randint(
        0,
        eng.embed_tokens.num_embeddings,
        (t, 1, eng.embed_tokens.n_hash_cols),
        generator=g,
    )
    hidden = torch.randn(t, hc_mult, hidden_size, generator=g)
    positions = torch.arange(t, dtype=torch.int64)
    cos_sin_cache = torch.zeros(1)  # the attention stub ignores it
    mask = torch.tensor([True, True, False, True, True])

    # The runner's pre-graph host staging pass fills the rows a replay reads.
    eng.prepare_embeddings(hashes[:, 0], graph_visible=False)

    out_hashes, _ = layer(hidden, positions, cos_sin_cache, None, None, hashes, mask)
    out_none_mask, _ = layer(hidden, positions, cos_sin_cache, None, None, None, mask)
    out_none_none, _ = layer(hidden, positions, cos_sin_cache, None, None, None, None)

    # (None, mask) reads the staged rows: bitwise equal to the explicit ids.
    assert torch.equal(out_none_mask, out_hashes)

    # (None, None) skips engram entirely; the output is the residual path.
    saved = layer.engram
    layer.engram = None
    try:
        ref_none_none, _ = layer(hidden, positions, cos_sin_cache, None, None, None, None)
    finally:
        layer.engram = saved
    assert torch.equal(out_none_none, ref_none_none)

    # Non-vacuity: the mask-only call actually injected (differs from the skip).
    assert not torch.equal(out_none_mask, out_none_none)


def _make_int8_engram(layout: EngramLayout, dim: int, hc_mult: int, offload: bool) -> Engram:
    eng = Engram(
        layout,
        0,
        dim,
        hc_mult,
        rms_norm_eps=1e-6,
        tp_size=1,
        tp_rank=0,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        storage=ENGRAM_STORAGE_INT8,
        offload=offload,
    )
    g = torch.Generator().manual_seed(17)
    with torch.no_grad():
        payload = torch.randint(-127, 128, eng.embed_tokens.weight.shape, dtype=torch.int8, generator=g)
        scales = torch.rand(eng.embed_tokens.weight_scale.shape, generator=g).float() * 0.05 + 0.01
        eng.embed_tokens.weight.copy_(payload)
        eng.embed_tokens.weight_scale.copy_(scales)
        eng.wkv.weight.copy_(torch.randn(eng.wkv.weight.shape, generator=g).bfloat16())
        eng.q_weight.copy_(torch.randn(eng.q_weight.shape, generator=g).bfloat16())
        eng.k_weight.copy_(torch.randn(eng.k_weight.shape, generator=g).bfloat16())
    return eng


def test_engram_offload_forward_without_hash_ids_matches_device(_engram_offload_lookup_op) -> None:
    """A graph-replay forward reads the staged rows (hash_ids=None) bit-identically."""
    layout = EngramLayout(
        layer_ids=[1],
        num_embeddings=[290],
        max_ngram_size=3,
        n_heads=2,
        head_dim=32,
        vocab_size=64,
        compressed_vocab_size=16,
        pad_token_id=0,
    )
    dim, hc_mult = 48, 2
    assert layout.n_hash_cols == 4
    dev = _make_int8_engram(layout, dim, hc_mult, offload=False)
    off = _make_int8_engram(layout, dim, hc_mult, offload=True)
    assert off.offload is True
    # Same table + gate weights on both paths.
    with torch.no_grad():
        off.embed_tokens.weight.copy_(dev.embed_tokens.weight)
        off.embed_tokens.weight_scale.copy_(dev.embed_tokens.weight_scale)
        off.wkv.weight.copy_(dev.wkv.weight)
        off.q_weight.copy_(dev.q_weight)
        off.k_weight.copy_(dev.k_weight)

    t = 4
    hash_ids = torch.zeros(t, layout.n_hash_cols, dtype=torch.int64)
    dev.prepare_embeddings(hash_ids)
    off.prepare_embeddings(hash_ids)
    assert dev._staged_rows is not None
    assert off._staged_local is not None
    # tp_size is 1 here, so the head gather is the identity: the staged local
    # block already equals the fully gathered rows.
    assert torch.equal(off.embed_tokens.gather_heads(off._staged_local), dev._staged_rows)

    hidden = torch.randn(t, hc_mult, dim, generator=torch.Generator().manual_seed(23)).bfloat16()
    out_dev = dev(hidden, hash_ids)
    out_off = off(hidden, None)
    assert torch.equal(out_off, out_dev)

    # Before any staging, the staged-only contract fails fast instead of
    # falling back to a device lookup.
    fresh = _make_int8_engram(layout, dim, hc_mult, offload=True)
    with pytest.raises(RuntimeError, match="no staged rows"):
        fresh.embed(None)


def test_stage_engram_rows_outside_graph_is_noop_when_off() -> None:
    """The runner hook is inert for non-offload models (OFF zero-behavior change)."""
    fake = SimpleNamespace(engram_offload_active=False, engram_hash=None, layers=[])
    assert dq.DeepseekV41Model.stage_engram_rows_outside_graph(fake, None, None, None, None) is False


def test_stage_engram_rows_reports_success_for_the_capture_gate(_engram_offload_lookup_op) -> None:
    """The staging hook records whether engram is usable this step.

    When the SWA cache is unbound (``hashes is None``) the pass stages nothing
    and reports ``False``; the runner's first-capture gate then refuses the
    capture, so the graph never reads a stale staging block.
    """
    layout = _engram_test_layout(layer_id=1)
    token_ids = torch.tensor([ENGRAM_IMAGE_SENTINEL_BASE_ID, 0, 0, 0], dtype=torch.int64)

    # Cache unbound -> the hook reports not staged (the runner gate refuses it).
    unbound = SimpleNamespace(
        engram_offload_active=True,
        engram_hash=object(),
        layers=_fake_v41_layers(_make_int8_engram(layout, 48, 2, offload=True)),
        cfg=SimpleNamespace(tp_rank=0),
    )
    unbound._compute_engram_hashes = lambda *args: (None, None)
    assert dq.DeepseekV41Model.stage_engram_rows_outside_graph(unbound, None, None, token_ids, token_ids) is False

    # Staging succeeds -> the capture mask is the sentinel complement.
    staged = SimpleNamespace(
        engram_offload_active=True,
        engram_hash=object(),
        layers=_fake_v41_layers(_make_int8_engram(layout, 48, 2, offload=True)),
        cfg=SimpleNamespace(tp_rank=0),
    )
    staged._compute_engram_hashes = lambda *args: (torch.zeros(4, 1, layout.n_hash_cols, dtype=torch.int32), None)
    assert dq.DeepseekV41Model.stage_engram_rows_outside_graph(staged, None, None, token_ids, token_ids) is True
    capture_mask = dq.DeepseekV41Model._engram_capture_mask(staged, token_ids)
    assert capture_mask is not None
    assert capture_mask.tolist() == [False, True, True, True]
    assert torch.equal(capture_mask, ~dq.image_sentinel_mask(token_ids))


def test_stage_engram_rows_makes_one_device_sync_for_all_layers(
    _engram_offload_lookup_op, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-graph staging pass drains the device ONCE for every engram layer.

    The per-layer entry (``prepare_embeddings`` -> ``_local_block_offload``)
    synchronizes the whole device after queueing each layer's D2H copies, so the
    old N-layer loop paid N blocking drains where N-1 waited on copies the first
    drain had already retired. The batched two-phase pass must queue every
    layer's D2H, drain once, then run each layer's CPU gather. Discriminating:
    the old per-layer loop makes ``sync_calls == 2`` here, so ``== 1`` fails
    before the fix.

    The batching only pays off if the queued D2H copies are themselves
    non-blocking: a blocking ``copy_`` would stall the host on every layer
    regardless of the drain count. So the test also spies the ``non_blocking``
    argument of every copy whose destination is one of the two pinned host
    staging buffers and asserts it is ``True`` -- dropping ``non_blocking=True``
    from ``stage_ids_offload`` flips these to ``False`` and fails here, while the
    sync count alone would not notice.

    The protective drain inside ``_ensure_host_scratch`` is deliberately NOT
    counted: it only fires when the pinned scratch grows, so the layers are
    pre-warmed (scratch + graph block sized) before the counter is installed.
    """
    import xllm.python.models.deepseek_v41_engram as engram_mod

    layout = _engram_test_layout(layer_id=1)
    first = _make_int8_engram(layout, 48, 2, offload=True)
    second = _make_int8_engram(layout, 48, 2, offload=True)
    second.layer_hash_index = 1  # a distinct hash column for the second layer

    t = 4
    generator = torch.Generator().manual_seed(101)
    hashes = torch.randint(
        0,
        layout.num_embeddings[0],
        (t, 2, layout.n_hash_cols),
        generator=generator,
        dtype=torch.int32,
    )

    # Pre-warm the pinned scratch and the graph staging block so the counted
    # pass hits only the batched drain (``_ensure_host_scratch`` syncs only on
    # growth). These warm-up syncs run before the counter is installed.
    for engram in (first, second):
        engram.prepare_embeddings(hashes[:, engram.layer_hash_index], graph_visible=True)

    sync_calls = 0
    real_sync = engram_mod._sync_accelerator

    def _counting_sync(device: torch.device | None) -> None:
        nonlocal sync_calls
        sync_calls += 1
        real_sync(device)

    monkeypatch.setattr(engram_mod, "_sync_accelerator", _counting_sync)

    # Spy the D2H ``non_blocking`` argument: the sync count alone would still
    # pass if the queued copies stayed blocking. Identify the D2H copies by
    # destination storage -- only the two per-layer pinned host staging buffers
    # count; the H2D upload (NPU destination) and the reference layers below
    # (different storages) are filtered out. The buffers exist after the
    # pre-warm above and are not reallocated by the counted pass.
    host_storages = {engram.embed_tokens._ids_host.untyped_storage().data_ptr() for engram in (first, second)} | {
        engram.embed_tokens._owned_host.untyped_storage().data_ptr() for engram in (first, second)
    }
    d2h_non_blocking: list[bool] = []
    real_copy = torch.Tensor.copy_

    def _spy_copy(self: torch.Tensor, src: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        non_blocking = kwargs.get("non_blocking", args[0] if args else False)
        if self.device.type == "cpu" and self.untyped_storage().data_ptr() in host_storages:
            d2h_non_blocking.append(bool(non_blocking))
        return real_copy(self, src, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", _spy_copy)

    model = SimpleNamespace(
        engram_offload_active=True,
        engram_hash=object(),
        layers=_fake_v41_layers(first, second),
        cfg=SimpleNamespace(tp_rank=0),
    )
    model._compute_engram_hashes = lambda *args: (hashes, None)

    token_ids = torch.zeros(t, dtype=torch.int64)
    assert dq.DeepseekV41Model.stage_engram_rows_outside_graph(model, None, None, token_ids, token_ids) is True
    assert sync_calls == 1
    # Two layers x (ids + owned): every queued D2H copy must be non-blocking.
    assert d2h_non_blocking == [True, True, True, True]

    # Parity + publication: each layer's published block is bit-identical to the
    # per-layer ``prepare_embeddings`` path and points at the persistent
    # graph-visible block (not the growable eager one).
    for column, engram in enumerate((first, second)):
        reference = _make_int8_engram(layout, 48, 2, offload=True)
        with torch.no_grad():
            reference.embed_tokens.weight.copy_(engram.embed_tokens.weight)
            reference.embed_tokens.weight_scale.copy_(engram.embed_tokens.weight_scale)
        reference.prepare_embeddings(hashes[:, column].to(torch.int64), graph_visible=True)

        assert engram._staged_local is not None
        assert engram._staged_local_graph is True
        assert torch.equal(engram.embed(None), reference.embed(None))
        staging = engram.embed_tokens._graph_staged_device
        assert staging is not None
        assert engram._staged_local.untyped_storage().data_ptr() == staging.untyped_storage().data_ptr()


def test_engram_capture_mask_is_unconditional() -> None:
    """The capture mask is unconditional: it never skips engram.

    The runner's first-capture gate (``_engram_first_capture_gate``) refuses the
    capture when ``stage_engram_rows_outside_graph`` staged nothing while engram
    is offloading, so by the time ``_engram_capture_mask`` runs the staged rows
    are present. The mask therefore always returns the sentinel complement; the
    fail-fast lives in the runner, never in a silent in-graph skip.
    """
    token_ids = torch.tensor([ENGRAM_IMAGE_SENTINEL_BASE_ID, 0, 0, 0], dtype=torch.int64)
    expected = ~dq.image_sentinel_mask(token_ids)
    assert expected.tolist() == [False, True, True, True]

    # The mask depends only on the token ids and consults no model state, so a
    # bare object stands in for ``self``: any attribute lookup would raise.
    assert torch.equal(dq.DeepseekV41Model._engram_capture_mask(object(), token_ids), expected)


def test_stage_engram_rows_builds_dsa_metadata_for_the_first_capture(_engram_offload_lookup_op) -> None:
    """The FIRST ACL graph capture must still inject Engram.

    The runner calls this staging hook BEFORE ``_capture``; on that first pass
    ``metadata.dsa_metadata`` has not been built yet, because the capture's
    warmup forward (which runs inside ``_capture``) is what normally builds it.
    The hook must build the DSA metadata itself, or the hashing pass finds no
    SWA group, reports "not staged", and the capture branch bakes an Engram-less
    graph that every replay reuses (ACL replay never re-runs Python). This case
    reproduces the first-capture ordering at the hook boundary.

    Discriminating: removing the DSA build makes ``swa_group_metadata(None)``
    return None, so staging reports False and this case fails.
    """
    layout = _engram_test_layout(layer_id=1)
    token_ids = torch.tensor([ENGRAM_IMAGE_SENTINEL_BASE_ID, 0, 0, 0], dtype=torch.int64)
    positions = torch.arange(4, dtype=torch.int32)

    class _Metadata:
        def __init__(self) -> None:
            # The runner's ``attention_backend.prepare(graph_mode=True)`` set
            # this before handing the metadata over; DSA metadata is still
            # absent on the first capture.
            self.dsa_graph_mode = True
            self.dsa_metadata = None
            self.dsa_positions = None

    class _Dsa:
        seq_lens_q = torch.ones(4, dtype=torch.int32)

    class _SwaGroup:
        cache = torch.zeros(64, dtype=torch.int32)
        block_size = 4
        slot_mapping = torch.zeros(4, dtype=torch.int32)
        block_table = torch.zeros(4, 1, dtype=torch.int32)

    class _Backend:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def reset_forward(self, metadata) -> None:
            self.calls.append("reset_forward")
            # Mirror the real backend reset_forward (csa_attention.py): it also
            # clears dsa_positions/dsa_graph_mode. Without those the metadata
            # restoration mutations (graph-flag restore, positions-after-reset
            # ordering) would go unexercised in these cases; the builder-order
            # case asserts the restored state.
            metadata.dsa_metadata = None
            metadata.dsa_positions = None
            metadata.dsa_graph_mode = False

        def prepare_dsa_metadata_for_forward(self, metadata) -> None:
            self.calls.append("prepare_dsa_metadata_for_forward")
            metadata.dsa_metadata = _Dsa()

        def swa_group_metadata(self, dsa):
            # Real backends return no SWA group while the DSA metadata is
            # absent, which is what makes the un-built path skip engram.
            return _SwaGroup() if dsa is not None else None

    class _Hash:
        def ensure_cache(self, cache, block_size) -> bool:
            return True

        def __call__(self, *args, **kwargs) -> torch.Tensor:
            return torch.zeros(4, 1, layout.n_hash_cols, dtype=torch.int32)

    engram = _make_int8_engram(layout, 48, 2, offload=True)
    backend = _Backend()
    metadata = _Metadata()
    model = SimpleNamespace(
        engram_offload_active=True,
        engram_hash=_Hash(),
        layers=_fake_v41_layers(engram),
        cfg=SimpleNamespace(tp_rank=0),
    )
    # Delegate through the class (call-time lookup) exactly like the real bound
    # methods, so the whole first-capture hook is the production code path and a
    # class-level mutation of the DSA builder is observable.
    model._build_dsa_metadata_for_forward = lambda backend, metadata, positions: (
        dq.DeepseekV41Model._build_dsa_metadata_for_forward(model, backend, metadata, positions)
    )
    model._compute_engram_hashes = lambda backend, metadata, input_ids, positions: (
        dq.DeepseekV41Model._compute_engram_hashes(model, backend, metadata, input_ids, positions)
    )

    assert metadata.dsa_metadata is None
    staged = dq.DeepseekV41Model.stage_engram_rows_outside_graph(model, backend, metadata, token_ids, positions)
    # The hook built the metadata (via the same path ``forward`` uses) before
    # hashing, then staged the rows into the graph-visible block.
    assert backend.calls == ["reset_forward", "prepare_dsa_metadata_for_forward"]
    assert metadata.dsa_metadata is not None
    assert staged is True
    assert engram._staged_local is not None

    # The capture branch therefore enters Engram (mask is the sentinel
    # complement); staging succeeded, so there is no skip path to fall into.
    capture_mask = dq.DeepseekV41Model._engram_capture_mask(model, token_ids)
    assert capture_mask is not None
    assert capture_mask.tolist() == [False, True, True, True]


def test_build_dsa_metadata_for_forward_matches_the_forward_path() -> None:
    """The extracted builder reproduces the forward branch verbatim.

    ``forward`` now delegates its first-build branch to
    ``_build_dsa_metadata_for_forward`` and the offload staging hook calls the
    same method, so every consumer of a freshly built DSA metadata sees the
    identical reset/build/flag sequence.
    """

    class _Metadata:
        def __init__(self) -> None:
            self.dsa_graph_mode = True
            self.dsa_metadata = None
            self.dsa_positions = None

    class _Backend:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def reset_forward(self, metadata) -> None:
            self.calls.append("reset_forward")
            # Mirror the real backend reset_forward (csa_attention.py): it also
            # clears dsa_positions/dsa_graph_mode, so the graph-flag and positions
            # restoration the builder performs is actually exercised here.
            metadata.dsa_metadata = None
            metadata.dsa_positions = None
            metadata.dsa_graph_mode = False

        def prepare_dsa_metadata_for_forward(self, metadata) -> None:
            self.calls.append("prepare_dsa_metadata_for_forward")
            metadata.dsa_metadata = object()

    backend = _Backend()
    metadata = _Metadata()
    model = SimpleNamespace()
    dq.DeepseekV41Model._build_dsa_metadata_for_forward(model, backend, metadata, torch.arange(4, dtype=torch.int32))
    assert backend.calls == ["reset_forward", "prepare_dsa_metadata_for_forward"]
    assert metadata.dsa_metadata is not None
    # Positions are normalized to int64, and the graph flag survives the build.
    assert metadata.dsa_positions.dtype == torch.int64
    assert metadata.dsa_graph_mode is True

    # A missing builder stays a loud error, never a silent skip. Since the
    # builder is called directly (EAFP), a backend without it raises the
    # AttributeError naming the missing method rather than a hand-rolled
    # RuntimeError.
    class _NoPrepare:
        def reset_forward(self, metadata) -> None:
            metadata.dsa_metadata = None

    with pytest.raises(AttributeError, match="prepare_dsa_metadata_for_forward"):
        dq.DeepseekV41Model._build_dsa_metadata_for_forward(
            model, _NoPrepare(), _Metadata(), torch.arange(4, dtype=torch.int32)
        )


def _quarot_gate_reference(
    eng: Engram,
    hidden: torch.Tensor,
    staged_rows: torch.Tensor,
    unrotate: torch.Tensor | None,
) -> torch.Tensor:
    """Manual Engram.forward math (gate dot optionally on the un-rotated stream)."""
    rows = staged_rows.to(eng.wkv.weight.dtype)
    kv = eng.wkv(rows.flatten(-2))
    num_tokens, hc_mult, dim = hidden.shape
    keys = kv[:, : hc_mult * dim].view(num_tokens, hc_mult, dim).float()
    value = kv[:, hc_mult * dim :].float()
    h = hidden.float()
    gate_hidden = h @ unrotate.to(torch.float32) if unrotate is not None else h
    hidden_rms = torch.rsqrt(gate_hidden.square().mean(-1) + eng.eps)
    key_rms = torch.rsqrt(keys.square().mean(-1) + eng.eps)
    dot = (gate_hidden * eng.q_weight.float() * eng.k_weight.float() * keys).sum(-1)
    dot = dot * hidden_rms * key_rms * float(dim) ** -0.5
    gate_input = dot.abs().clamp_min(1e-6).sqrt()
    gate_input = torch.where(dot < 0.0, -gate_input, gate_input)
    gate = torch.sigmoid(gate_input)
    return (h + gate.unsqueeze(-1) * value.unsqueeze(1)).to(hidden.dtype)


def test_engram_gate_quarot_unrotation() -> None:
    """With gate_unrotate the gate dot runs on h_rot @ Q.T; V stays rotated-basis."""
    layout = EngramLayout(
        layer_ids=[1],
        num_embeddings=[290],
        max_ngram_size=3,
        n_heads=2,
        head_dim=32,
        vocab_size=64,
        compressed_vocab_size=16,
        pad_token_id=0,
    )
    dim, hc_mult = 48, 2
    g = torch.Generator().manual_seed(99)
    q_orth = torch.linalg.qr(torch.randn(dim, dim, generator=g))[0]

    def make(gate_unrotate: torch.Tensor | None) -> Engram:
        eng = Engram(
            layout,
            0,
            dim,
            hc_mult,
            rms_norm_eps=1e-6,
            tp_size=1,
            tp_rank=0,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            storage=ENGRAM_STORAGE_INT8,
            gate_unrotate=gate_unrotate,
        )
        with torch.no_grad():
            eng.wkv.weight.copy_(torch.randn(eng.wkv.weight.shape, generator=g).bfloat16())
            eng.q_weight.copy_(torch.randn(hc_mult, dim, generator=g).bfloat16())
            eng.k_weight.copy_(torch.randn(hc_mult, dim, generator=g).bfloat16())
        return eng

    t, n_hash_cols = 7, layout.n_hash_cols
    staged = torch.randn(t, n_hash_cols, layout.head_dim, generator=g).bfloat16()
    hidden = torch.randn(t, hc_mult, dim, generator=g).bfloat16()

    plain = make(None)
    plain._staged_rows = staged
    out_plain = plain(hidden, torch.zeros(t, n_hash_cols, dtype=torch.int64))
    assert torch.equal(out_plain, _quarot_gate_reference(plain, hidden, staged, None))

    rotated = make(q_orth.t().contiguous())
    rotated.wkv.load_state_dict(plain.wkv.state_dict())
    with torch.no_grad():
        rotated.q_weight.copy_(plain.q_weight)
        rotated.k_weight.copy_(plain.k_weight)
    rotated._staged_rows = staged
    out_rot = rotated(hidden, torch.zeros(t, n_hash_cols, dtype=torch.int64))
    assert torch.equal(out_rot, _quarot_gate_reference(rotated, hidden, staged, q_orth.t()))
    # The un-rotation must actually change the gated output (non-identity Q).
    assert not torch.equal(out_plain, out_rot)


def test_load_quarot_gate_unrotate(tmp_path, monkeypatch) -> None:
    """Q.T loads from optional/quarot.safetensors only for rotated checkpoints."""
    import json as _json
    import sys

    import safetensors.torch as _st
    from safetensors import SafetensorError

    model_path = tmp_path / "eco"
    (model_path / "optional").mkdir(parents=True)
    g = torch.Generator().manual_seed(7)
    q = torch.linalg.qr(torch.randn(64, 64, generator=g))[0]

    def write_config(rotated: bool) -> None:
        (model_path / "config.json").write_text(
            _json.dumps({"engram_rotation_config": {"value_projection_rotated": rotated}})
        )

    # Unrotated config -> None (no optional file needed).
    write_config(False)
    assert load_quarot_gate_unrotate(str(model_path), torch.device("cpu")) is None

    # Rotated config without the quarot file stays None only for missing
    # config; with the file present the transpose must load exactly.
    write_config(True)
    _st.save_file({"global_rotation": q.contiguous()}, str(model_path / "optional" / "quarot.safetensors"))
    loaded = load_quarot_gate_unrotate(str(model_path), torch.device("cpu"))
    assert loaded is not None
    assert loaded.dtype == torch.float32
    assert torch.equal(loaded, q.t().contiguous().to(torch.float32))

    # A bf16-saved rotation matrix loads with the header-derived dtype and is
    # still returned as fp32.
    q_bf = q.to(torch.bfloat16)
    _st.save_file({"global_rotation": q_bf.contiguous()}, str(model_path / "optional" / "quarot.safetensors"))
    loaded_bf = load_quarot_gate_unrotate(str(model_path), torch.device("cpu"))
    assert loaded_bf is not None
    assert loaded_bf.dtype == torch.float32
    assert torch.equal(loaded_bf, q_bf.t().contiguous().to(torch.float32))

    # Manual header-parse branch (safetensors import unavailable): both fp32
    # and bf16 payloads decode exactly.
    for q_saved in (q.contiguous(), q_bf.contiguous()):
        _st.save_file({"global_rotation": q_saved}, str(model_path / "optional" / "quarot.safetensors"))
        monkeypatch.setitem(sys.modules, "safetensors.torch", None)
        loaded_manual = load_quarot_gate_unrotate(str(model_path), torch.device("cpu"))
        monkeypatch.undo()
        assert loaded_manual is not None
        assert loaded_manual.dtype == torch.float32
        assert torch.equal(loaded_manual, q_saved.t().contiguous().to(torch.float32))

    # Corrupt quarot payload with safetensors available must raise instead of
    # silently falling back to the manual parser.
    (model_path / "optional" / "quarot.safetensors").write_bytes(b"garbage-bytes")
    with pytest.raises(SafetensorError):
        load_quarot_gate_unrotate(str(model_path), torch.device("cpu"))


def test_engram_gate_math_and_mask_passthrough() -> None:
    lm = _make_forward_lm(torch.float32, seed=43)
    cfg = lm.cfg
    eng = lm.model.layers[1].engram
    assert eng is not None
    with torch.no_grad():
        eng.q_weight.normal_(0.0, 0.05)
        eng.k_weight.normal_(0.0, 0.05)
        eng.wkv.weight.normal_(0.0, 0.05)
    # A deterministic table row: ue8m0 scale 2^0 dequantizes e4m3 payloads.
    rows, dim = eng.embed_tokens.weight.shape
    eng.embed_tokens.weight.zero_()
    payload = torch.tensor([0x38, 0xC0], dtype=torch.uint8)  # 1.0 and -1.0
    eng.embed_tokens.weight[:, 0] = payload[0]
    eng.embed_tokens.weight[:, 1] = payload[1]
    eng.embed_tokens.weight_scale.fill_(127)  # 2^0

    t, hc, hidden = 2, cfg.hc_mult, cfg.hidden_size
    hash_ids = torch.zeros(t, eng.embed_tokens.n_hash_cols, dtype=torch.int64)
    hidden_states = torch.randn(t, hc, hidden)

    def reference() -> torch.Tensor:
        rows_gathered = eng.embed_tokens.lookup(hash_ids).to(eng.wkv.weight.dtype)
        kv = eng.wkv(rows_gathered.flatten(-2))
        keys = kv[:, : hc * hidden].view(t, hc, hidden).float()
        value = kv[:, hc * hidden :].float()
        h = hidden_states.float()
        q = eng.q_weight.float()
        k = eng.k_weight.float()
        eps = cfg.rms_norm_eps
        h_rms = torch.rsqrt(h.square().mean(-1) + eps)
        k_rms = torch.rsqrt(keys.square().mean(-1) + eps)
        dot = (h * q * k * keys).sum(-1) * h_rms * k_rms * hidden**-0.5
        gate_input = dot.abs().clamp_min(1e-6).sqrt()
        gate_input = torch.where(dot < 0.0, -gate_input, gate_input)
        gate = torch.sigmoid(gate_input)
        return h + gate.unsqueeze(-1) * value.unsqueeze(1)

    out = eng(hidden_states, hash_ids)
    assert torch.allclose(out, reference(), atol=1e-4)

    # token_mask False shuts the gate: those rows pass through untouched.
    mask = torch.tensor([True, False])
    out_masked = eng(hidden_states, hash_ids, mask)
    assert torch.allclose(out_masked[0], out[0], atol=1e-6)
    assert torch.allclose(out_masked[1], hidden_states[1], atol=1e-6)

    # Image sentinels break n-grams and close the gate in the model mask.
    assert cfg.image_token_id == ENGRAM_IMAGE_SENTINEL_BASE_ID


@npu_only
def test_engram_injection_shifts_residual_stream() -> None:
    """The model wires hashing, staging and the gated injection end to end."""
    device = _active_device()
    assert device.type == "npu"
    lm = _make_forward_lm(torch.float32, seed=45, device=device)
    assert lm.model.norm.weight.device.type == "npu"
    cfg = lm.cfg
    n_seqs, prompt_len = 1, 6
    harness = _V41ForwardHarness(lm, _DSV41_FORWARD_CONFIG, n_seqs, torch.float32, device=device)
    assert harness.layer_caches[0].swa.device.type == "npu"
    input_ids = torch.randint(0, cfg.vocab_size, (n_seqs, prompt_len)).to(device)
    with torch.inference_mode():
        hidden = harness.step(
            input_ids.reshape(-1),
            torch.arange(prompt_len, device=device),
            [prompt_len],
            [prompt_len],
            is_prefill=True,
        )
    assert torch.isfinite(hidden).all()
    # The engram hash state is wired through the model and its slot cache
    # was sized from the bound SWA cache.
    assert lm.model.engram_hash is not None
    assert lm.model.engram_hash._cache is not None
    assert lm.model.engram_hash._cache.numel() == harness.layer_caches[0].swa.numel() // cfg.head_dim
    # The slot cache holds the chunk's compressed ids (dead tokens excluded).
    state = lm.model.engram_hash
    comp = state.token_map[input_ids.reshape(-1).to(torch.int64)]
    flat_cache = state._cache
    for pos, value in enumerate(comp.tolist()):
        slot = int(harness.swa_bt[0, pos // cfg.window_size]) * cfg.window_size + pos % cfg.window_size
        assert int(flat_cache[slot]) == value
    # Every engram layer staged its rows before the decoder loop, and the
    # staged rows are the dequantized table lookups of the hash ids.
    eng = lm.model.layers[1].engram
    assert eng is not None and eng._staged_rows is not None
    n_hash_cols = cfg.engram_n_heads * (cfg.engram_max_ngram_size - 1)
    assert eng._staged_rows.shape == (prompt_len, n_hash_cols, cfg.engram_head_dim)
    assert torch.isfinite(eng._staged_rows.float()).all()
    # Hash ids stay inside the layer's prime bucket ranges.
    hashes = state(
        input_ids.reshape(-1),
        torch.arange(prompt_len, device=device),
        torch.tensor([0, prompt_len], device=device),
        torch.zeros(prompt_len, dtype=torch.bool, device=device),
        None,
        None,
        None,
        0,
    )
    assert (hashes >= 0).all() and (hashes < cfg.engram_num_embeddings[0]).all()
