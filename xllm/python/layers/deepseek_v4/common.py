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

"""Backend-neutral contracts for DeepSeek-V4 MXFP8 layers."""

from __future__ import annotations

from typing import Protocol


class DeepseekV4LayerConfig(Protocol):
    """Configuration fields read by DeepSeek-V4 MXFP8 layers.

    The concrete configuration is owned by the higher-level ``models``
    package. Layers consume this structural contract so they do not need to
    depend on the model implementation.
    """

    hidden_size: int
    tp_size: int
