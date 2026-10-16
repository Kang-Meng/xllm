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

"""NPU DeepSeek-V4 MXFP8 layer implementation."""

from .linear import (
    NpuDeepseekV4DynamicLinearMXFP8,
    NpuDeepseekV4GroupedOProjectionMXFP8,
    NpuDeepseekV4RowParallelLinearMXFP8,
)
from .mlp import NpuDeepseekV4MLPMXFP8

__all__ = [
    "NpuDeepseekV4DynamicLinearMXFP8",
    "NpuDeepseekV4GroupedOProjectionMXFP8",
    "NpuDeepseekV4MLPMXFP8",
    "NpuDeepseekV4RowParallelLinearMXFP8",
]
