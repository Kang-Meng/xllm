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

"""Multimodal adapter to compose with the Kimi K3 text integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from xllm.python.models.base import PyModelBase
from xllm.python.models.kimi_k3_vit import KimiK3VisionModel
from xllm.python.models.multimodal import (
    EncoderOutputs,
    EncoderTokenCounts,
    MultimodalData,
    MultimodalInputs,
    merge_multimodal_embeddings,
)


class KimiK3ForConditionalGeneration(PyModelBase):
    """Vision/text composition using the mainline Python executor.

    Kimi image and video placeholders share an ID, so embedding merge uses
    the scheduler's item masks rather than token-ID matching. The mainline
    multimodal executor owns encoding, caching and chunk slicing, and passes
    MultimodalInputs to the text runner.
    """

    cache_multimodal_embeddings: bool = True

    def __init__(self, config: dict[str, Any], language_model: PyModelBase | None = None) -> None:
        super().__init__()
        if language_model is None:
            from xllm.python.models.kimi_k3_text import KimiK3ForCausalLM

            # C++ publishes flat runtime/topology fields. Keep checkpoint-only
            # nested configuration within this model's config-only factory.
            model_path = config.get("model_path")
            if model_path is not None:
                with (Path(model_path) / "config.json").open() as source:
                    config = {**json.load(source), **config}
            language_model = KimiK3ForCausalLM(config)
        self.vision_model = KimiK3VisionModel(config)
        self.language_model = language_model

    @property
    def supports_prefix_cache(self) -> bool:
        return self.language_model.supports_prefix_cache

    @property
    def model(self) -> torch.nn.Module:
        return self.language_model.model

    def compute_logits(self, hidden: torch.Tensor, selected_idxes: torch.Tensor | None) -> torch.Tensor:
        return self.language_model.compute_logits(hidden, selected_idxes)

    def encode(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        """Return full encoder rows in media order for executor chunk slicing."""
        outputs = self.vision_model(pixel_values, grid_thws)
        if not outputs:
            reference = next(self.vision_model.parameters())
            return reference.new_empty((0, self.vision_model.config.text_hidden_size))
        return torch.cat(outputs, dim=0)

    def encode_multimodal(self, data: MultimodalData, token_counts: EncoderTokenCounts) -> EncoderOutputs:
        """Split encoder rows by scheduled item counts, including video chunks."""
        if token_counts.get("audio"):
            raise ValueError("Kimi K3 supports image and video inputs only")
        outputs: EncoderOutputs = {}
        for modality, pixel_key, grid_key in (
            ("image", "pixel_values", "image_grid_thw"),
            ("video", "pixel_values_videos", "video_grid_thw"),
        ):
            counts = token_counts.get(modality, [])
            if not counts:
                continue
            pixels, grids = data[pixel_key], data[grid_key]
            if not isinstance(pixels, torch.Tensor) or not isinstance(grids, torch.Tensor):
                raise TypeError("Kimi K3 batched pixels and grids must be tensors")
            if any(count < 0 for count in counts):
                raise ValueError("Kimi K3 media item token counts must not be negative")
            features = self.encode(pixels, grids)
            if features.ndim != 2 or features.shape[0] != sum(counts):
                raise ValueError("Kimi K3 encoder rows must match the uncached media item token counts")
            outputs[f"{modality}|embedding"] = list(features.split(counts, dim=0))
        return outputs

    def prepare_multimodal_inputs(self, input_ids: torch.Tensor, data: MultimodalData) -> MultimodalInputs:
        """Use the mainline executor's already sliced media embeddings/masks."""
        return MultimodalInputs(
            self.get_input_embeddings(
                input_ids,
                data.get("image|embedding"),
                data.get("video|embedding"),
                image_mask=data.get("image|mask"),
                video_mask=data.get("video|mask"),
            )
        )

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        image_embeds: torch.Tensor | None = None,
        video_embeds: torch.Tensor | None = None,
        *,
        image_mask: torch.Tensor | None = None,
        video_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Merge scheduled media rows; an empty chunk may have a false-only mask."""
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        data: dict[str, torch.Tensor] = {}
        for modality, embeds, mask in (
            ("image", image_embeds, image_mask),
            ("video", video_embeds, video_mask),
        ):
            if mask is None:
                if embeds is not None:
                    raise ValueError(f"Kimi K3 {modality} embeddings require a scheduled mask")
                continue
            if embeds is None:
                if bool(mask.any()):
                    raise ValueError(f"Kimi K3 {modality} mask requires embeddings")
                continue
            data[f"{modality}|embedding"] = embeds
            data[f"{modality}|mask"] = mask
        # Shared placeholder IDs require disjoint scheduled masks. Keep this
        # Kimi-specific check before the shared helper scatters either modality.
        if image_mask is not None and video_mask is not None and bool((image_mask & video_mask).any()):
            raise ValueError("Kimi K3 image and video masks must not overlap")
        return merge_multimodal_embeddings(inputs_embeds, data)

    def load_weights(self, state_dicts: list[Any], tp_rank: int, tp_size: int) -> set[str]:
        loaded = self.vision_model.load_weights(state_dicts, tp_rank, tp_size)
        loaded.update(self.language_model.load_weights(state_dicts, tp_rank, tp_size))
        return loaded


__all__ = ["KimiK3ForConditionalGeneration"]
