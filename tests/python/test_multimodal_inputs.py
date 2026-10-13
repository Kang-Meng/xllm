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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from xllm.python import kernels
from xllm.python.model_executor.runners.eager import EagerRunner
from xllm.python.models.glm5_next_vl import Glm5NextVLModel
from xllm.python.models.joyai_asr import JoyaiASRForConditionalGeneration
from xllm.python.models.multimodal import MultimodalInputs, SupportsMultimodal, merge_multimodal_embeddings
from xllm.python.models.qwen3 import Qwen3Config
from xllm.python.models.qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLModel


def _text_embedding() -> nn.Embedding:
    return nn.Embedding.from_pretrained(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))


@pytest.mark.parametrize("model_class", [Qwen3VLForConditionalGeneration, Glm5NextVLModel])
def test_model_merges_interleaved_modalities_by_scheduled_positions(model_class: type[nn.Module]) -> None:
    model = model_class.__new__(model_class)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.embed_tokens = _text_embedding()
    model.language_model = model.model
    model.deepstack_num_level = 0
    model._hidden_rot = None
    data = {
        "image|embedding": torch.tensor([[11.0, 12.0]]),
        "image|mask": torch.tensor([False, False, True, False]),
        "video|embedding": torch.tensor([[21.0, 22.0], [23.0, 24.0]]),
        "video|mask": torch.tensor([True, False, False, True]),
    }
    prepared = model.prepare_multimodal_inputs(torch.tensor([0, 1, 0, 0]), data)
    torch.testing.assert_close(
        prepared.inputs_embeds, torch.tensor([[21.0, 22.0], [3.0, 4.0], [11.0, 12.0], [23.0, 24.0]])
    )
    assert isinstance(model, SupportsMultimodal)


@pytest.mark.parametrize("feature_dtype", [torch.float32, torch.bfloat16])
def test_qwen_deepstack_preserves_text_rows_and_each_feature_level(feature_dtype: torch.dtype) -> None:
    model = Qwen3VLForConditionalGeneration.__new__(Qwen3VLForConditionalGeneration)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.embed_tokens = _text_embedding()
    model.deepstack_num_level = 2
    data = {
        "image|embedding": torch.tensor([[10.0, 11.0, 20.0, 21.0, 30.0, 31.0]], dtype=feature_dtype),
        "image|mask": torch.tensor([False, True, False, False]),
        "video|embedding": torch.tensor([[40.0, 41.0, 50.0, 51.0, 60.0, 61.0]], dtype=feature_dtype),
        "video|mask": torch.tensor([False, False, False, True]),
    }
    result = model.prepare_multimodal_inputs(torch.tensor([0, 0, 2, 0]), data)
    torch.testing.assert_close(result.inputs_embeds, torch.tensor([[1.0, 2.0], [10.0, 11.0], [5.0, 6.0], [40.0, 41.0]]))
    torch.testing.assert_close(
        result.deepstack_input_embeds[0], torch.tensor([[0.0, 0.0], [20.0, 21.0], [0.0, 0.0], [50.0, 51.0]])
    )
    torch.testing.assert_close(
        result.deepstack_input_embeds[1], torch.tensor([[0.0, 0.0], [30.0, 31.0], [0.0, 0.0], [60.0, 61.0]])
    )
    assert result.inputs_embeds.is_contiguous()
    assert all(level.is_contiguous() for level in result.deepstack_input_embeds)
    torch.testing.assert_close(
        data["image|embedding"], torch.tensor([[10.0, 11.0, 20.0, 21.0, 30.0, 31.0]], dtype=feature_dtype)
    )
    torch.testing.assert_close(
        data["video|embedding"], torch.tensor([[40.0, 41.0, 50.0, 51.0, 60.0, 61.0]], dtype=feature_dtype)
    )


@pytest.mark.parametrize("deepstack_levels", [0, 2])
def test_qwen_empty_visual_chunk_preserves_text_and_zeroes_deepstack(deepstack_levels: int) -> None:
    model = Qwen3VLForConditionalGeneration.__new__(Qwen3VLForConditionalGeneration)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.embed_tokens = _text_embedding()
    model.deepstack_num_level = deepstack_levels
    data = {
        "video|embedding": torch.empty(0, 2 * (1 + deepstack_levels)),
        "video|mask": torch.tensor([False, False]),
    }
    result = model.prepare_multimodal_inputs(torch.tensor([0, 1]), data)
    torch.testing.assert_close(result.inputs_embeds, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    if deepstack_levels:
        assert len(result.deepstack_input_embeds) == deepstack_levels
        for level in result.deepstack_input_embeds:
            torch.testing.assert_close(level, torch.zeros(2, 2))
    else:
        assert result.deepstack_input_embeds is None


@pytest.mark.parametrize(
    "modality,pixel_key,grid_key",
    [
        ("image", "pixel_values", "image_grid_thw"),
        ("video", "pixel_values_videos", "video_grid_thw"),
    ],
)
@pytest.mark.parametrize("rotated", [False, True])
def test_glm_encoding_aligns_features_before_chunk_fusion(
    modality: str, pixel_key: str, grid_key: str, rotated: bool
) -> None:
    model = Glm5NextVLModel.__new__(Glm5NextVLModel)
    nn.Module.__init__(model)
    model.language_model = nn.Module()
    model.language_model.embed_tokens = _text_embedding()
    model._hidden_rot = torch.tensor([[0.0, 1.0], [1.0, 0.0]]) if rotated else None
    model.encode = lambda pixels, grid: pixels
    features = torch.tensor([[11.0, 12.0], [21.0, 22.0], [31.0, 32.0]])
    expected = features.flip(1) if rotated else features.clone()
    outputs = model.encode_multimodal(
        {pixel_key: features, grid_key: torch.tensor([[1, 1, 1], [1, 1, 2]])}, {modality: [1, 2]}
    )
    encoded_items = outputs[f"{modality}|embedding"]
    torch.testing.assert_close(encoded_items[0], expected[:1])
    torch.testing.assert_close(encoded_items[1], expected[1:])
    torch.testing.assert_close(features, torch.tensor([[11.0, 12.0], [21.0, 22.0], [31.0, 32.0]]))
    for chunk in encoded_items[1].split(1):
        data = {f"{modality}|embedding": chunk, f"{modality}|mask": torch.tensor([True, False])}
        for _ in range(2):
            result = model.prepare_multimodal_inputs(torch.tensor([0, 1]), data)
            torch.testing.assert_close(result.inputs_embeds, torch.cat([chunk, torch.tensor([[3.0, 4.0]])]))
    torch.testing.assert_close(encoded_items[1], expected[1:])


def test_audio_mask_keeps_gap_text_and_unrelated_pad_tokens() -> None:
    model = JoyaiASRForConditionalGeneration.__new__(JoyaiASRForConditionalGeneration)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.embed_tokens = _text_embedding()
    data = {
        "audio|embedding": torch.tensor([[10.0, 11.0], [20.0, 21.0]]),
        "audio|mask": torch.tensor([True, False, True, False]),
    }
    result = model.prepare_multimodal_inputs(torch.tensor([0, 1, 0, 0]), data)
    torch.testing.assert_close(result.inputs_embeds, torch.tensor([[10.0, 11.0], [3.0, 4.0], [20.0, 21.0], [1.0, 2.0]]))
    assert not model.cache_multimodal_embeddings
    assert isinstance(model, SupportsMultimodal)


@pytest.mark.parametrize("deepstack_levels", [0, 2])
def test_fusion_rejects_misaligned_features(deepstack_levels: int) -> None:
    data = {"image|embedding": torch.ones(2, 2 * (1 + deepstack_levels)), "image|mask": torch.tensor([True, False])}
    deepstack = tuple(torch.zeros(2, 2) for _ in range(deepstack_levels))
    with pytest.raises(ValueError, match="rows do not match"):
        merge_multimodal_embeddings(torch.zeros(2, 2), data, deepstack)


def test_empty_visual_chunk_preserves_text_embeddings() -> None:
    text = _text_embedding()(torch.tensor([0, 1]))
    data = {"video|embedding": torch.empty(0, 2), "video|mask": torch.tensor([False, False])}
    result = merge_multimodal_embeddings(text.clone(), data)
    torch.testing.assert_close(result, text)


@pytest.mark.parametrize("model_class", [Qwen3VLForConditionalGeneration, Glm5NextVLModel])
def test_vision_encoder_returns_full_items_in_modality_order(model_class: type[nn.Module]) -> None:
    model = model_class.__new__(model_class)
    nn.Module.__init__(model)
    model._hidden_rot = None
    model.encode = lambda pixels, grid: pixels * grid[0, 0]
    images = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    videos = torch.tensor([[7.0, 8.0]])
    data = {
        "pixel_values": images,
        "image_grid_thw": torch.tensor([[2]]),
        "pixel_values_videos": videos,
        "video_grid_thw": torch.tensor([[3]]),
    }
    output = model.encode_multimodal(data, {"image": [1, 2], "video": [1]})
    torch.testing.assert_close(output["image|embedding"][0], images[:1] * 2)
    torch.testing.assert_close(output["image|embedding"][1], images[1:] * 2)
    torch.testing.assert_close(output["video|embedding"][0], videos * 3)
    assert model.encode_multimodal({}, {}) == {}


def test_asr_encoding_keeps_cached_ctc_hints_in_current_item_order() -> None:
    model = JoyaiASRForConditionalGeneration.__new__(JoyaiASRForConditionalGeneration)
    nn.Module.__init__(model)
    first_audio = torch.tensor([[1.0, 2.0]])
    second_audio = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    model._encode_cache = {11: (first_audio, [101]), 22: (second_audio, [201, 202])}
    model._ctc_results = []
    data = {
        "input_features": torch.empty(0),
        "speech_lengths": torch.tensor([0, 0]),
        "audio_encode_meta": torch.tensor([[22, 0], [11, 0]]),
    }
    output = model.encode_multimodal(data, {"audio": [2, 1]})
    torch.testing.assert_close(output["audio|embedding"][0], second_audio)
    torch.testing.assert_close(output["audio|embedding"][1], first_audio)
    assert model.get_prompt_lookup_hints() == [[201, 202], [101]]
    data["audio_encode_meta"] = torch.tensor([[11, 0], [22, 0]])
    model.encode_multimodal(data, {"audio": [1, 2]})
    assert model.get_prompt_lookup_hints() == [[101], [201, 202]]


def test_asr_rejects_metadata_missing_a_speech_item_before_cache_lookup() -> None:
    model = JoyaiASRForConditionalGeneration.__new__(JoyaiASRForConditionalGeneration)
    nn.Module.__init__(model)
    model._encode_cache = {22: (torch.ones(2, 2), [201, 202])}
    model._ctc_results = [[101]]
    data = {
        "input_features": torch.empty(0),
        "speech_lengths": torch.tensor([0, 0]),
        "audio_encode_meta": torch.tensor([[22, 0]]),
    }
    with pytest.raises(ValueError, match="audio metadata rows"):
        model.encode_multimodal(data, {"audio": [1, 1]})
    assert model.get_prompt_lookup_hints() == [[101]]


class _IdentityDecoderLayer(nn.Module):
    def forward(
        self, hidden: torch.Tensor, residual: torch.Tensor | None, *args: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden, torch.zeros_like(hidden)


class _NoAttentionBackend:
    is_mla = False

    def prepare(self, metadata: object) -> None:
        pass


def test_runner_consumes_explicit_deepstack_without_leaking_to_next_step(monkeypatch: pytest.MonkeyPatch) -> None:
    def residual_norm(
        hidden: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = hidden + residual
        return torch.nn.functional.rms_norm(combined, (combined.shape[-1],), weight, epsilon), combined

    monkeypatch.setattr(kernels, "fused_add_rms_norm", residual_norm, raising=False)
    config = Qwen3Config(hidden_size=2, head_dim=2, n_heads=1, n_kv_heads=1, n_layers=0, vocab_size=3)
    model = Qwen3VLModel(config, dtype=torch.float32, device=torch.device("cpu"))
    model.embed_tokens = _text_embedding()
    model.layers = nn.ModuleList([_IdentityDecoderLayer()])
    model.norm.weight.data.fill_(1.0)
    runner = EagerRunner(model, _NoAttentionBackend(), torch.device("cpu"))
    metadata = SimpleNamespace(is_prefill=True, is_chunked_prefill=False)
    prepared = MultimodalInputs(torch.tensor([[2.0, 1.0]]), (torch.tensor([[1.0, 3.0]]),))
    multimodal_output = runner.execute(torch.tensor([0]), torch.tensor([0]), metadata, multimodal_inputs=prepared)
    expected = torch.tensor([[3.0, 4.0]])
    expected = expected * torch.rsqrt(expected.square().mean(-1, keepdim=True) + config.rms_norm_eps)
    torch.testing.assert_close(multimodal_output, expected)
    text_output = runner.execute(torch.tensor([0]), torch.tensor([0]), metadata)
    text = torch.tensor([[1.0, 2.0]])
    expected_text = text * torch.rsqrt(text.square().mean(-1, keepdim=True) + config.rms_norm_eps)
    torch.testing.assert_close(text_output, expected_text)
