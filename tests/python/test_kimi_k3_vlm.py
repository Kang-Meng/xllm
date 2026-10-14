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

"""Kimi components using the mainline Python VLM executor contract."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from xllm.python import registry
from xllm.python.models.base import PyModelBase
from xllm.python.models.kimi_k3 import KimiK3ForConditionalGeneration
from xllm.python.models.multimodal import SupportsMultimodal


class _TextModel(PyModelBase):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 8)
        self.lm_head = nn.Linear(8, 8, bias=False)
        self.config = config
        self.loaded: tuple | None = None

    @property
    def supports_prefix_cache(self) -> bool:
        return True

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> set[str]:
        self.loaded = (state_dicts, tp_rank, tp_size)
        return {"text.weight"}


@pytest.fixture
def vlm() -> KimiK3ForConditionalGeneration:
    config = {
        "device": "cpu",
        "dtype": "float32",
        "vision_config": {
            "patch_size": 2,
            "in_chans": 3,
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
            "text_hidden_size": 8,
        },
    }
    model = KimiK3ForConditionalGeneration(config, _TextModel(config))
    for parameter in model.parameters():
        nn.init.uniform_(parameter, -0.1, 0.1)
    return model


def test_vlm_exposes_text_model_to_executor(vlm: KimiK3ForConditionalGeneration) -> None:
    assert vlm.model is vlm.language_model.model
    assert vlm.supports_prefix_cache
    hidden = torch.randn(3, 8)
    selected = torch.tensor([2, 0])
    torch.testing.assert_close(
        vlm.compute_logits(hidden, selected),
        vlm.language_model.compute_logits(hidden, selected),
    )


def test_vlm_text_only_keeps_token_embeddings(vlm: KimiK3ForConditionalGeneration) -> None:
    tokens = torch.tensor([1, 2, 3])
    torch.testing.assert_close(
        vlm.get_input_embeddings(tokens),
        vlm.language_model.get_input_embeddings(tokens),
    )


def test_vlm_encode_and_merge_preserves_text_rows(vlm: KimiK3ForConditionalGeneration) -> None:
    pixels = torch.randn(8, 3, 2, 2)
    encoded = vlm.encode(pixels, torch.tensor([[2, 2, 2]]))
    assert isinstance(encoded, torch.Tensor)
    assert encoded.shape == (1, 8)
    tokens = torch.tensor([1, 2, 3, 4])
    text = vlm.language_model.get_input_embeddings(tokens)
    mask = torch.tensor([False, True, False, False])
    merged = vlm.get_input_embeddings(tokens, encoded, image_mask=mask)
    torch.testing.assert_close(merged[mask], encoded)
    torch.testing.assert_close(merged[~mask], text[~mask])


def test_chunk_mask_places_identical_media_pad_tokens_independently(
    vlm: KimiK3ForConditionalGeneration,
) -> None:
    # The C++ gather visitor owns positions: image/video use the same pad ID.
    # Replacing by token ID would also overwrite the unselected placeholder.
    tokens = torch.tensor([3, 3, 3])
    image = torch.randn(1, 8)
    mask = torch.tensor([False, True, False])
    text = vlm.language_model.get_input_embeddings(tokens)
    merged = vlm.get_input_embeddings(tokens, image, image_mask=mask)
    torch.testing.assert_close(merged[1], image[0])
    torch.testing.assert_close(merged[[0, 2]], text[[0, 2]])


@pytest.mark.parametrize(
    ("embeds", "mask", "message"),
    [
        (torch.ones(1, 8), None, "scheduled mask"),
        (None, torch.tensor([True, False]), "requires embeddings"),
        (torch.ones(1, 8), torch.tensor([1, 0]), "one boolean per scheduled token"),
        (torch.ones(1, 8), torch.tensor([True]), "one boolean per scheduled token"),
        (torch.ones(1, 4), torch.tensor([True, False]), "embedding width does not match"),
        (torch.ones(2, 8), torch.tensor([True, False]), "embedding rows do not match the scheduled mask"),
    ],
)
def test_vlm_rejects_inconsistent_embeddings(
    vlm: KimiK3ForConditionalGeneration,
    embeds: torch.Tensor | None,
    mask: torch.Tensor | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        vlm.get_input_embeddings(torch.tensor([1, 2]), embeds, image_mask=mask)


def test_vlm_dispatches_weights_to_vision_and_text(vlm: KimiK3ForConditionalGeneration) -> None:
    state_dicts = [object()]
    vlm.vision_model.load_weights = MagicMock(return_value={"vision.weight"})
    assert vlm.load_weights(state_dicts, 0, 1) == {"vision.weight", "text.weight"}
    vlm.vision_model.load_weights.assert_called_once_with(state_dicts, 0, 1)
    assert vlm.language_model.loaded == (state_dicts, 0, 1)


@pytest.mark.parametrize("name", ["kimi_k3", "KimiK3ForConditionalGeneration"])
def test_incomplete_vlm_is_not_registered_as_a_service(name: str) -> None:
    with pytest.raises(KeyError, match="not registered"):
        registry.get_model_class(name)


@pytest.mark.parametrize("include_image", [False, True])
def test_vlm_merges_video_and_image_by_independent_masks(
    vlm: KimiK3ForConditionalGeneration, include_image: bool
) -> None:
    tokens = torch.tensor([3, 3, 3, 3, 3])
    text = vlm.language_model.get_input_embeddings(tokens)
    image = torch.randn(1, 8) if include_image else None
    image_mask = torch.tensor([False, True, False, False, False]) if include_image else None
    video = torch.randn(2, 8)
    video_mask = torch.tensor([False, False, True, True, False])
    actual = vlm.get_input_embeddings(tokens, image, video, image_mask=image_mask, video_mask=video_mask)
    torch.testing.assert_close(actual[video_mask], video)
    if include_image:
        torch.testing.assert_close(actual[image_mask], image)
    keep_text = ~video_mask if image_mask is None else ~(video_mask | image_mask)
    torch.testing.assert_close(actual[keep_text], text[keep_text])


def test_vlm_requires_video_embedding_and_mask_together(vlm: KimiK3ForConditionalGeneration) -> None:
    with pytest.raises(ValueError, match="video embeddings require a scheduled mask"):
        vlm.get_input_embeddings(torch.tensor([1, 2]), None, torch.ones(1, 8))


def test_vlm_rejects_overlapping_image_and_video_masks_without_writing(vlm: KimiK3ForConditionalGeneration) -> None:
    tokens = torch.tensor([1, 2, 3])
    original = vlm.language_model.get_input_embeddings(tokens)
    vlm.language_model.get_input_embeddings = MagicMock(return_value=original)
    saved = original.clone()
    with pytest.raises(ValueError, match="must not overlap"):
        vlm.get_input_embeddings(
            tokens,
            torch.ones(1, 8),
            torch.zeros(1, 8),
            image_mask=torch.tensor([False, True, False]),
            video_mask=torch.tensor([False, True, False]),
        )
    torch.testing.assert_close(original, saved)


def test_encode_concatenates_temporal_chunks_and_media_in_order(vlm: KimiK3ForConditionalGeneration) -> None:
    grids = torch.tensor([[2, 2, 2], [1, 2, 2]])
    pixels = torch.randn(12, 3, 2, 2)
    expected = torch.cat(vlm.vision_model(pixels, grids), dim=0)
    torch.testing.assert_close(vlm.encode(pixels, grids), expected)


def test_encode_empty_media_preserves_output_width(vlm: KimiK3ForConditionalGeneration) -> None:
    actual = vlm.encode(torch.empty(0, 3, 2, 2), torch.empty(0, 3, dtype=torch.int64))
    assert actual.shape == (0, 8)
    assert actual.dtype == next(vlm.vision_model.parameters()).dtype


def test_item_masks_preserve_video_before_image_order(vlm: KimiK3ForConditionalGeneration) -> None:
    tokens = torch.tensor([3, 3, 3])
    image = torch.full((1, 8), 14.0)
    video = torch.full((1, 8), 27.0)
    text = vlm.language_model.get_input_embeddings(tokens)
    merged = vlm.get_input_embeddings(
        tokens,
        image,
        video,
        image_mask=torch.tensor([False, True, False]),
        video_mask=torch.tensor([True, False, False]),
    )
    torch.testing.assert_close(merged[0], video[0])
    torch.testing.assert_close(merged[1], image[0])
    torch.testing.assert_close(merged[2], text[2])


def test_empty_chunk_mask_needs_no_encoder_rows(vlm: KimiK3ForConditionalGeneration) -> None:
    tokens = torch.tensor([3, 3])
    expected = vlm.language_model.get_input_embeddings(tokens)
    actual = vlm.get_input_embeddings(tokens, image_mask=torch.zeros(2, dtype=torch.bool))
    torch.testing.assert_close(actual, expected)


def test_merge_returns_explicit_embeddings_without_retaining_them(vlm: KimiK3ForConditionalGeneration) -> None:
    tokens = torch.tensor([1, 2])
    image = torch.ones(1, 8)
    merged = vlm.get_input_embeddings(tokens, image, image_mask=torch.tensor([False, True]))
    assert isinstance(vlm, SupportsMultimodal)
    torch.testing.assert_close(merged[1], image[0])
    assert not hasattr(vlm.model, "_inputs_embeds")
    # A following text/decode step cannot reuse a previous media embedding.
    torch.testing.assert_close(vlm.get_input_embeddings(tokens), vlm.language_model.get_input_embeddings(tokens))
