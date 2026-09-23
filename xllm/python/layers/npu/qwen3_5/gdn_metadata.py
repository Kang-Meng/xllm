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

"""Model-visible execution metadata for the Qwen3.5 NPU GDN path."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class GdnStateCache:
    """Long-lived Conv and SSM state owned by one GDN layer."""

    conv_state: torch.Tensor
    ssm_state: torch.Tensor


@dataclass(frozen=True, slots=True)
class GdnMetadata:
    """Common model-visible inputs for a Qwen3.5 GDN execution phase."""

    state_caches: Mapping[int, GdnStateCache]


@dataclass(frozen=True, slots=True)
class GdnPrefillMetadata(GdnMetadata):
    """Operator-ready, request-scoped inputs for MegaGdn prefill."""

    conv_read_indices: torch.Tensor
    conv_write_indices: torch.Tensor
    ssm_read_indices: torch.Tensor
    ssm_write_indices: torch.Tensor
    cu_seqlens: torch.Tensor
    num_matrices: int


@dataclass(frozen=True, slots=True)
class GdnDecodeMetadata(GdnMetadata):
    """Operator-ready, request-scoped inputs for MegaGdn decode."""

    read_state_indices: torch.Tensor
    write_state_indices: torch.Tensor


@dataclass(frozen=True, slots=True)
class GdnSpecVerifyMetadata(GdnDecodeMetadata):
    """Decode-shaped inputs for the MegaGdn MTP speculative-verify path.

    Carries per-sequence accepted-token counts so the kernel commits recurrent
    state only up to each sequence's accepted prefix.
    """

    num_accepted_tokens: torch.Tensor
