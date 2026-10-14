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

"""Compatibility import for the Qwen3.5 decoder factory.

Decoder implementations live in the CUDA and NPU backend packages.  New code
should import the factory from :mod:`xllm.python.layers.qwen3_5.factory`.
"""

from __future__ import annotations

from xllm.python.layers.qwen3_5.factory import get_qwen3_5_decoder_layer_class

__all__ = ["get_qwen3_5_decoder_layer_class"]
