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

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

MultimodalData = dict[str, torch.Tensor | list[torch.Tensor]]
EncoderTokenCounts = dict[str, list[int]]
EncoderOutputs = dict[str, list[torch.Tensor]]


@dataclass(frozen=True)
class MultimodalInputs:
    inputs_embeds: torch.Tensor
    deepstack_input_embeds: tuple[torch.Tensor, ...] | None = None

    def forward_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {"inputs_embeds": self.inputs_embeds}
        if self.deepstack_input_embeds is not None:
            kwargs["deepstack_input_embeds"] = self.deepstack_input_embeds
        return kwargs


@runtime_checkable
class SupportsMultimodal(Protocol):
    cache_multimodal_embeddings: bool

    def encode_multimodal(self, data: MultimodalData, token_counts: EncoderTokenCounts) -> EncoderOutputs: ...

    def prepare_multimodal_inputs(self, input_ids: torch.Tensor, data: MultimodalData) -> MultimodalInputs: ...


def merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    data: MultimodalData,
    additional_embeds: tuple[torch.Tensor, ...] = (),
) -> torch.Tensor:
    """Scatter features into and return the caller-owned 2-D input tensor.

    The caller supplies writable buffers with the same scheduled token count,
    dtype and device. Feature columns correspond to inputs_embeds followed by
    additional_embeds, which are also updated in place.
    Prepared data supplies one feature row per true mask entry.
    The runtime validates item row counts before gathering on every device;
    this helper checks CPU mask counts without synchronizing device masks.
    Feature tensors and masks in data are read-only inputs.
    """
    token_count = inputs_embeds.shape[0]
    destinations = (inputs_embeds, *additional_embeds)
    widths = [destination.shape[1] for destination in destinations]
    feature_width = sum(widths)
    for key, features in data.items():
        if not key.endswith("|embedding"):
            continue
        modality = key.removesuffix("|embedding")
        mask = data[f"{modality}|mask"]
        if not isinstance(features, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise TypeError("prepared multimodal embeddings and masks must be tensors")
        if features.ndim != 2 or features.shape[1] != feature_width:
            raise ValueError(f"{modality} embedding width does not match the input embeddings")
        if mask.ndim != 1 or mask.numel() != token_count or mask.dtype != torch.bool:
            raise ValueError(f"{modality} mask must contain one boolean per scheduled token")
        if mask.device.type == "cpu" and int(mask.sum().item()) != features.shape[0]:
            raise ValueError(f"{modality} embedding rows do not match the scheduled mask")
        for destination, level_features in zip(destinations, features.to(inputs_embeds.dtype).split(widths, dim=1)):
            destination[mask] = level_features
    return inputs_embeds
