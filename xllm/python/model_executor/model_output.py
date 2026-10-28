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

import torch


@dataclass
class DraftProposal:
    token_ids: torch.Tensor
    draft_probs: torch.Tensor | None = None


@dataclass
class ModelOutput:
    draft_proposal: DraftProposal | None = None
    draft_confidence_probs: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    residual: torch.Tensor | None = None
    aux_hidden_states: torch.Tensor | None = None
