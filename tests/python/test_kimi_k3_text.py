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

"""Hybrid text loading and continuation using CPU device-kernel oracles."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from tests.python.test_kimi_k3_kda import _checkpoint, cpu_kernels, linear_kernels  # noqa: F401
from xllm.python import registry
from xllm.python.attention.backend import LayerCache
from xllm.python.model_executor.forward_context import AclGraphExecutionState, ForwardContext, forward_context
from xllm.python.model_executor.runners.decode_acl_graph import _StaticAttentionMetadata
from xllm.python.models.kimi_k3_kda import KimiK3KDAMetadata
from xllm.python.models.kimi_k3_text import KimiK3ForCausalLM, KimiK3TextConfig


def _config() -> dict:
    return {
        "device": "cpu",
        "dtype": "float32",
        "tp_size": 1,
        "text_config": {
            "hidden_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 16,
            "vocab_size": 32,
            "q_lora_rank": 4,
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 4,
            "qk_rope_head_dim": 4,
            "v_head_dim": 4,
            "first_k_dense_replace": 2,
            "num_experts": 4,
            "num_experts_per_token": 2,
            "moe_intermediate_size": 8,
            "routed_expert_hidden_size": 8,
            "linear_attn_config": {
                "num_heads": 4,
                "head_dim": 4,
                "short_conv_kernel_size": 4,
                "use_full_rank_gate": True,
                "kda_layers": [1],
                "full_attn_layers": [2],
            },
        },
        "vision_config": {
            "patch_size": 2,
            "vt_hidden_size": 4,
            "vt_intermediate_size": 8,
            "vt_num_attention_heads": 2,
            "vt_num_hidden_layers": 1,
            "qkv_hidden_size": 8,
            "init_pos_emb_height": 2,
            "init_pos_emb_width": 2,
            "init_pos_emb_time": 2,
            "merge_kernel_size": [2, 2],
            "mm_hidden_size": 4,
            "text_hidden_size": 12,
        },
    }


class _StateDict:
    def __init__(self, values: dict[str, torch.Tensor]) -> None:
        self.values = values

    def has(self, name: str) -> bool:
        return name in self.values

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.values[name]

    def keys(self) -> list[str]:
        return list(self.values)


def _weights(model: KimiK3ForCausalLM) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(421)
    values = {
        name: torch.randn(parameter.shape, generator=generator) * 0.1 for name, parameter in model.named_parameters()
    }
    for layer_id, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{layer_id}."
        attn = prefix + "self_attn."
        if layer.is_kda:
            for name in list(values):
                if name.startswith(attn):
                    del values[name]
            values.update({attn + name: value for name, value in _checkpoint(False).items()})
        else:
            fused = values.pop(attn + "qkv_a_proj.weight")
            values[attn + "q_a_proj.weight"] = fused[:4]
            values[attn + "kv_a_proj_with_mqa.weight"] = fused[4:]
        gate_up = values.pop(prefix + "mlp.gate_up_proj.weight")
        values[prefix + "mlp.gate_proj.weight"], values[prefix + "mlp.up_proj.weight"] = gate_up.chunk(2)
    return values


class _AbsorbedMLAOracle:
    """Paged attention oracle; projection/loading stays in the real model."""

    def __init__(self) -> None:
        self.kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def execute_mla(
        self,
        q: torch.Tensor,
        q_pe: torch.Tensor,
        k: torch.Tensor,
        k_pe: torch.Tensor,
        layer: object,
        topk: object,
        unabsorbed_prefill: object = None,
    ) -> torch.Tensor:
        old = self.kv.get(layer.layer_id)
        history = 0 if old is None else old[0].shape[0]
        if old is not None:
            k, k_pe = torch.cat((old[0], k)), torch.cat((old[1], k_pe))
        self.kv[layer.layer_id] = (k.clone(), k_pe.clone())
        scores = (torch.einsum("thd,shd->hts", q, k) + torch.einsum("thd,shd->hts", q_pe, k_pe)) * layer.scale
        allowed = torch.arange(k.shape[0])[None, :] <= history + torch.arange(q.shape[0])[:, None]
        scores.masked_fill_(~allowed, float("-inf"))
        output = torch.einsum("hts,shd->thd", scores.softmax(-1), k)
        if unabsorbed_prefill is not None:
            return torch.einsum("hts,shd->thd", scores.softmax(-1), unabsorbed_prefill.value)
        return output


def _context(
    backend: _AbsorbedMLAOracle,
    caches: list[LayerCache],
    rows: int,
    prefill: bool,
    initial: bool = False,
    read_slot: int = 1,
    write_slot: int = 1,
) -> ForwardContext:
    metadata = SimpleNamespace(
        is_prefill=prefill,
        is_chunked_prefill=False,
        is_spec_verify=False,
        q_cu_seq_lens=torch.tensor([0, rows], dtype=torch.int32),
        q_cu_seq_lens_host_values=[0, rows],
        linear_state_indices=torch.tensor([write_slot]),
        linear_state_read_indices=torch.tensor([read_slot]),
        linear_state_write_indices=torch.tensor([write_slot]),
        has_initial_state=torch.tensor([initial]),
    )
    return ForwardContext(backend, torch.device("cpu"), metadata, caches)


def _caches(model: KimiK3ForCausalLM) -> list[LayerCache]:
    kda = model.model.layers[0].self_attn.kda
    # Larger conv/checkpoint capacity is the framework allocation contract.
    return [
        LayerCache(
            key=None,
            value=None,
            conv=torch.zeros(3, kda.conv_size + 1, 3 * kda.local_projection_size),
            ssm=torch.zeros(6, *kda.recurrent_state_shape()),
        ),
        LayerCache(None, None),
    ]


@pytest.mark.usefixtures("cpu_kernels", "linear_kernels")
def test_hybrid_checkpoint_and_first_decode_match_full_prefill() -> None:
    model = KimiK3ForCausalLM(_config())
    values = _weights(model)
    # Explicitly distribute names across two checkpoint shards.
    shards = [
        _StateDict(
            {"language_model." + key: value for index, (key, value) in enumerate(values.items()) if index % 2 == part}
        )
        for part in range(2)
    ]
    loaded = model.load_weights(shards, 0, 1)
    assert "model.layers.0.self_attn.q_conv1d.weight" in loaded
    assert "model.layers.1.self_attn.kv_b_proj.weight" in loaded
    tokens = torch.tensor([1, 7, 9, 3])
    caches = _caches(model)
    backend = _AbsorbedMLAOracle()
    with torch.no_grad(), forward_context(_context(backend, caches, 3, True)):
        model(tokens[:3], torch.arange(3))
    with torch.no_grad(), forward_context(_context(backend, caches, 1, False, True)):
        decoded = model(tokens[3:], torch.tensor([3]))
    full_caches = _caches(model)
    with torch.no_grad(), forward_context(_context(_AbsorbedMLAOracle(), full_caches, 4, True)):
        full = model(tokens, torch.arange(4))
    torch.testing.assert_close(decoded, full[-1:], atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(caches[0].conv[1, :3], full_caches[0].conv[1, :3])
    torch.testing.assert_close(caches[0].ssm[2], full_caches[0].ssm[2], atol=2e-5, rtol=2e-4)
    assert torch.count_nonzero(caches[0].ssm[1]) == 0


@pytest.mark.usefixtures("cpu_kernels", "linear_kernels")
def test_prefix_continuation_restores_framework_read_slot() -> None:
    model = KimiK3ForCausalLM(_config())
    model.load_weights([_StateDict(_weights(model))], 0, 1)
    layer = model.model.layers[0].self_attn
    hidden = torch.randn(4, 12)
    caches = _caches(model)
    with torch.no_grad(), forward_context(_context(_AbsorbedMLAOracle(), caches, 2, True)):
        layer(hidden[:2], torch.arange(2))
    saved = caches[0].ssm[2].clone()
    with torch.no_grad(), forward_context(_context(_AbsorbedMLAOracle(), caches, 2, True, True, 1, 2)):
        continuation = layer(hidden[2:], torch.arange(2, 4))
    full_caches = _caches(model)
    with torch.no_grad(), forward_context(_context(_AbsorbedMLAOracle(), full_caches, 4, True)):
        full = layer(hidden, torch.arange(4))
    torch.testing.assert_close(continuation, full[2:])
    torch.testing.assert_close(caches[0].ssm[2], saved)
    torch.testing.assert_close(caches[0].ssm[4], full_caches[0].ssm[2])


def test_registered_vlm_constructs_real_text_with_config_only() -> None:
    model = registry.get_model_class("kimi_k3")(_config())
    assert isinstance(model.language_model, KimiK3ForCausalLM)
    assert model.model.layers[0].self_attn.attention_kind == "linear"
    assert model.model.layers[1].self_attn.attention_kind == "mla"


@pytest.mark.parametrize("name", ["kimi_k3_text", "KimiK3ForCausalLM"])
def test_internal_text_component_is_not_advertised_as_a_service(name: str) -> None:
    with pytest.raises(KeyError, match="not registered"):
        registry.get_model_class(name)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_flashcomm1_respects_explicit_configuration(nested: bool, enabled: bool) -> None:
    config = _config()
    config["tp_size"] = 2
    target = config["text_config"] if nested else config
    target["enable_flashcomm1"] = enabled
    parsed = KimiK3TextConfig.from_dict(config)
    parsed.validate()
    assert parsed.use_sequence_parallel is enabled


def test_flashcomm1_remains_enabled_by_default() -> None:
    config = _config()
    config["tp_size"] = 2
    assert KimiK3TextConfig.from_dict(config).use_sequence_parallel


@pytest.mark.usefixtures("cpu_kernels", "linear_kernels")
def test_kda_convolution_host_metadata_is_materialized_once_per_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = KimiK3ForCausalLM(_config())
    layer = model.model.layers[0].self_attn
    context = _context(_AbsorbedMLAOracle(), _caches(model), 3, True)
    seen: list[KimiK3KDAMetadata] = []

    def record_metadata(
        hidden: torch.Tensor,
        metadata: KimiK3KDAMetadata,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        seen.append(metadata)
        return hidden

    monkeypatch.setattr(layer.kda, "forward", record_metadata)
    tolist = torch.Tensor.tolist
    calls: list[torch.dtype] = []

    def count_tolist(tensor: torch.Tensor) -> list:
        calls.append(tensor.dtype)
        return tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", count_tolist)
    with forward_context(context):
        for _ in range(3):
            layer(torch.zeros(3, 12), torch.arange(3))
    assert calls == [torch.int64, torch.bool]
    assert len(seen) == 3 and all(metadata is seen[0] for metadata in seen)
    assert seen[0].conv_state_indices_host == [1]
    assert seen[0].conv_initial_state_mode_host == [0]
    assert seen[0].conv_query_start_loc_host == [0, 3]
    with forward_context(_context(_AbsorbedMLAOracle(), _caches(model), 2, True, True, 2, 2)):
        layer(torch.zeros(2, 12), torch.arange(2))
    assert calls == [torch.int64, torch.bool] * 2
    assert seen[-1] is not seen[0]
    assert seen[-1].conv_state_indices_host == [2]
    assert seen[-1].conv_initial_state_mode_host == [1]
    assert seen[-1].conv_query_start_loc_host == [0, 2]


@pytest.mark.parametrize("host_format", ["cumulative", "ends", "lengths", "cpu_boundaries"])
@pytest.mark.usefixtures("cpu_kernels", "linear_kernels")
def test_kda_prefill_reuses_mainline_host_query_length_formats(
    monkeypatch: pytest.MonkeyPatch, host_format: str
) -> None:
    model = KimiK3ForCausalLM(_config())
    layer = model.model.layers[0].self_attn
    context = _context(_AbsorbedMLAOracle(), _caches(model), 3, True)
    raw = context.metadata
    if host_format == "ends":
        raw.q_cu_seq_lens_host_values = [3]
    elif host_format == "lengths":
        raw.q_cu_seq_lens_host_values = None
        raw.q_seq_lens_host = torch.tensor([3])
    elif host_format == "cpu_boundaries":
        raw.q_cu_seq_lens_host_values = None

    def record_metadata(
        hidden: torch.Tensor, metadata: KimiK3KDAMetadata, *args: object, **kwargs: object
    ) -> torch.Tensor:
        assert metadata.conv_query_start_loc_host == [0, 3]
        assert metadata.prefill_cu_seqlens_host == (0, 3)
        return hidden

    monkeypatch.setattr(layer.kda, "forward", record_metadata)
    with forward_context(context):
        layer(torch.zeros(3, 12), torch.arange(3))
    assert model.model.layers[1].self_attn.requires_dense_mla_head_padding


def test_explicit_zero_tp_rank_is_preserved() -> None:
    config = _config()
    config.update(tp_size=2, tp_rank=0, rank=3, world_size=4, dp_size=2)
    assert KimiK3TextConfig.from_dict(config).tp_rank == 0


@pytest.mark.usefixtures("linear_kernels")
def test_text_loader_connects_existing_moe_and_shared_experts() -> None:
    config = _config()
    config["text_config"]["first_k_dense_replace"] = 1
    model = KimiK3ForCausalLM(config)
    generator = torch.Generator().manual_seed(88)
    values = {
        name: torch.randn(parameter.shape, generator=generator) * 0.1 for name, parameter in model.named_parameters()
    }
    attn = "model.layers.0.self_attn."
    values = {name: value for name, value in values.items() if not name.startswith(attn)}
    values.update({attn + name: value for name, value in _checkpoint(False).items()})
    attn = "model.layers.1.self_attn."
    fused = values.pop(attn + "qkv_a_proj.weight")
    values[attn + "q_a_proj.weight"], values[attn + "kv_a_proj_with_mqa.weight"] = fused.split([4, 8])
    for prefix in ("model.layers.0.mlp.", "model.layers.1.block_sparse_moe.shared_experts."):
        gate_up = values.pop(prefix + "gate_up_proj.weight")
        values[prefix + "gate_proj.weight"], values[prefix + "up_proj.weight"] = gate_up.chunk(2)
    prefix = "model.layers.1.block_sparse_moe."
    values[prefix + "gate.e_score_correction_bias"] = torch.randn(4, generator=generator)
    routed_before = values[prefix + "experts.w13_weight"].clone()
    loaded = model.load_weights([_StateDict(values)], 0, 1)
    assert prefix + "experts.w13_weight" in loaded
    torch.testing.assert_close(model.model.layers[1].block_sparse_moe.experts.w13_weight, routed_before)
    del values[prefix + "routed_expert_up_proj.weight"]
    with pytest.raises(KeyError, match="routed_expert_up_proj.weight"):
        KimiK3ForCausalLM(config).load_weights([_StateDict(values)], 0, 1)


@pytest.mark.usefixtures("cpu_kernels", "linear_kernels")
def test_graph_decode_uses_framework_buckets_and_preserves_padding_state() -> None:
    model = KimiK3ForCausalLM(_config())
    model.load_weights([_StateDict(_weights(model))], 0, 1)
    layer = model.model.layers[0].self_attn
    caches = _caches(model)
    metadata = _StaticAttentionMetadata(
        slot_mapping=torch.zeros(2, dtype=torch.int64),
        paged_kv_indptr=torch.zeros(3, dtype=torch.int32),
        paged_kv_indices=torch.zeros(2, dtype=torch.int32),
        paged_kv_last_page_len=torch.zeros(2, dtype=torch.int32),
        linear_state_indices=torch.tensor([1, 0]),
        has_initial_state=torch.tensor([True, False]),
    )
    assert metadata.q_cu_seq_lens is None
    context = ForwardContext(
        _AbsorbedMLAOracle(),
        torch.device("cpu"),
        metadata,
        caches,
        execution_state=AclGraphExecutionState(persistent_buffers={}),
    )
    with torch.no_grad(), forward_context(context):
        output = layer(torch.randn(2, 12), torch.zeros(2, dtype=torch.int64))
    assert output.shape == (2, 12)
    assert torch.count_nonzero(caches[0].conv[0]) == 0
    assert torch.count_nonzero(caches[0].ssm[0]) == 0
    assert ("KIMI_KDA_DECODE_CU_SEQLENS", 2) in context.execution_state.persistent_buffers


def test_text_config_preserves_explicit_optional_activation_values() -> None:
    config = _config()
    config["text_config"].update(activation_situ_beta=0.0, activation_situ_linear_beta=None)
    parsed = KimiK3TextConfig.from_dict(config)
    assert parsed.activation_situ_beta == 0.0
    assert parsed.activation_situ_linear_beta is None
    with pytest.raises(ValueError, match="SiTU beta must be positive"):
        KimiK3ForCausalLM(config)


@pytest.mark.usefixtures("cpu_kernels", "linear_kernels")
def test_prefill_warmup_accepts_framework_reserved_scratch_slot() -> None:
    model = KimiK3ForCausalLM(_config())
    model.load_weights([_StateDict(_weights(model))], 0, 1)
    caches = _caches(model)
    context = _context(_AbsorbedMLAOracle(), caches, 3, True, read_slot=0, write_slot=0)
    with torch.no_grad(), forward_context(context):
        output = model.model.layers[0].self_attn(torch.randn(3, 12), torch.arange(3))
    assert output.shape == (3, 12)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(caches[0].ssm[2:]) == 0
