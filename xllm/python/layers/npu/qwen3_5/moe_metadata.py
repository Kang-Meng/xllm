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

"""Model-visible token validity for Qwen3.5 NPU sequence parallelism."""

from __future__ import annotations

from dataclasses import dataclass

import torch

MEGA_MOE_MAX_TOKENS = 4096


@dataclass(frozen=True, slots=True)
class Qwen3_5MoeMetadata:
    """Token validity: bool for ordinary EPLv2, int8 for EPLv1/MegaMoE SP."""

    active_token_mask: torch.Tensor
    # Eager sources are valid prefixes; Graph keeps fixed rows and uses the mask.
    active_rows: int | None = None
