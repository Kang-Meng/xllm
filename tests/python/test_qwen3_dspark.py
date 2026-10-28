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

from __future__ import annotations

import sys
import types
from unittest.mock import Mock

import pytest
import torch

from xllm.python import kernels
from xllm.python.models.dspark_pipeline import DSparkPipeline
from xllm.python.models.qwen3_dspark import (
    Qwen3DSparkConfig,
    Qwen3DSparkForCausalLM,
)


def _config(**overrides) -> Qwen3DSparkConfig:
    values = _config_dict(**overrides)
    config = Qwen3DSparkConfig.from_dict(values)
    config.validate()
    return config


def _config_dict(**overrides) -> dict:
    values = {
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 16,
        "vocab_size": 8,
        "draft_vocab_size": 8,
        "markov_rank": 2,
        "draft_query_width": 3,
        "tp_size": 1,
        "tp_rank": 0,
        "dp_size": 1,
        "dp_rank": 0,
        "dtype": "float32",
        "device": "cpu",
    }
    values.update(overrides)
    return values


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


def test_config_requires_positive_markov_rank() -> None:
    with pytest.raises(ValueError, match="markov_rank > 0"):
        _config(markov_rank=0)

@pytest.mark.parametrize("query_width", [1, 3, 15])
def test_pipeline_uses_runtime_query_width(query_width: int) -> None:
    model = Qwen3DSparkForCausalLM(
        _config_dict(draft_query_width=query_width, num_speculative_tokens=16)
    )

    assert model.model._query_width == query_width

def test_pipeline_requires_runtime_query_width() -> None:
    config = _config_dict(num_speculative_tokens=16)
    del config["draft_query_width"]

    with pytest.raises(KeyError, match="draft_query_width"):
        Qwen3DSparkForCausalLM(config)


@pytest.mark.parametrize("is_ascend_a3", [False, True])
def test_device_only_decode_capability_requires_a3(
    monkeypatch: pytest.MonkeyPatch,
    is_ascend_a3: bool,
) -> None:
    runtime = types.ModuleType("xllm_runtime")
    runtime.is_ascend_a3 = Mock(return_value=is_ascend_a3)
    monkeypatch.setitem(sys.modules, "xllm_runtime", runtime)
    model = Qwen3DSparkForCausalLM(_config_dict(hidden_size=128, head_dim=128, dtype="bfloat16"))

    assert model.supports_device_only_decode_inputs(kv_page_size=128) is is_ascend_a3


@pytest.mark.parametrize(
    "head_dim,kv_page_size,expected",
    [
        (64, 128, False),
        (256, 128, False),
        (128, 64, False),
        (128, 256, False),
        (64, 64, False),
        (128, 128, True),
    ],
)
def test_device_only_decode_capability_requires_supported_geometry(
    monkeypatch: pytest.MonkeyPatch,
    head_dim: int,
    kv_page_size: int,
    expected: bool,
) -> None:
    runtime = types.ModuleType("xllm_runtime")
    runtime.is_ascend_a3 = Mock(return_value=True)
    monkeypatch.setitem(sys.modules, "xllm_runtime", runtime)
    model = Qwen3DSparkForCausalLM(_config_dict(hidden_size=head_dim, head_dim=head_dim, dtype="bfloat16"))

    assert model.supports_device_only_decode_inputs(kv_page_size) is expected


def test_checkpoint_loads_dspark_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    prepare_row_parallel_weight = Mock(side_effect=lambda weight: (weight, False))
    monkeypatch.setattr(
        kernels,
        "prepare_row_parallel_weight",
        prepare_row_parallel_weight,
        raising=False,
    )
    model = Qwen3DSparkForCausalLM(
        _config_dict(
            enable_confidence_head=True,
            confidence_head_with_markov=True,
        )
    )
    tensors = {
        "d2t": torch.tensor([0, 1, 2, 3, 0, 0, 0, 0]),
        "fc.weight": torch.eye(4),
        "hidden_norm.weight": torch.ones(4),
        "layers.0.input_layernorm.weight": torch.ones(4),
        "layers.0.post_attention_layernorm.weight": torch.ones(4),
        "layers.0.self_attn.q_norm.weight": torch.ones(4),
        "layers.0.self_attn.k_norm.weight": torch.ones(4),
        "layers.0.self_attn.q_proj.weight": torch.full((4, 4), 1.0),
        "layers.0.self_attn.k_proj.weight": torch.full((4, 4), 2.0),
        "layers.0.self_attn.v_proj.weight": torch.full((4, 4), 3.0),
        "layers.0.self_attn.o_proj.weight": torch.full((4, 4), 4.0),
        "layers.0.mlp.gate_proj.weight": torch.full((8, 4), 5.0),
        "layers.0.mlp.up_proj.weight": torch.full((8, 4), 6.0),
        "layers.0.mlp.down_proj.weight": torch.full((4, 8), 7.0),
        "norm.weight": torch.ones(4),
        "markov_head.markov_w1.weight": torch.full((8, 2), 8.0),
        "markov_head.markov_w2.weight": torch.full((8, 2), 9.0),
        "confidence_head.proj.weight": torch.full((1, 6), 10.0),
        "confidence_head.proj.bias": torch.full((1,), 11.0),
    }

    model.load_weights([_StateDict(tensors)], tp_rank=0, tp_size=1)

    assert isinstance(model.model, DSparkPipeline)
    assert "model.backbone.fc.weight" in dict(model.named_parameters())
    torch.testing.assert_close(model.model.hot_token_ids, tensors["d2t"] + torch.arange(8))

    assert prepare_row_parallel_weight.call_count == 2
    torch.testing.assert_close(
        model.model.markov_head.markov_w1.weight,
        tensors["markov_head.markov_w1.weight"],
    )
    torch.testing.assert_close(
        model.model.markov_head.markov_w2.weight,
        tensors["markov_head.markov_w2.weight"],
    )
    assert model.model.confidence_head is not None
    torch.testing.assert_close(
        model.model.confidence_head.proj.weight,
        tensors["confidence_head.proj.weight"],
    )
    torch.testing.assert_close(
        model.model.confidence_head.proj.bias,
        tensors["confidence_head.proj.bias"],
    )
