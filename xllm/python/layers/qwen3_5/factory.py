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

"""Backend selection for Qwen3.5 decoder layers."""

from __future__ import annotations

import torch
import torch.nn as nn


def get_qwen3_5_decoder_layer_class(device: torch.device | str) -> type[nn.Module]:
    """Return the decoder implementation owned by ``device``."""
    device_type = torch.device(device).type
    if device_type == "cuda":
        from xllm.python.layers.cuda.qwen3_5.decoder_layer import (
            CudaQwen3_5DecoderLayer,
        )

        return CudaQwen3_5DecoderLayer
    if device_type in ("npu", "privateuseone"):
        from xllm.python.layers.npu.qwen3_5.decoder_layer import (
            NpuQwen3_5DecoderLayer,
        )

        return NpuQwen3_5DecoderLayer
    raise ValueError(f"Qwen3.5 Python has no decoder implementation for device {device_type!r}")
