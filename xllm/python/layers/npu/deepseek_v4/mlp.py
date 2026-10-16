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

"""NPU MXFP8 gated-SwiGLU MLP for DeepSeek-V4 official FP8 checkpoints."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python import distributed
from xllm.python.layers.deepseek_v4.common import DeepseekV4LayerConfig
from xllm.python.layers.npu.deepseek_v4.linear import NpuDeepseekV4DynamicLinearMXFP8

try:
    from xllm.python import kernels
except Exception:  # pragma: no cover - kernels need the compiled lib
    kernels = None  # type: ignore[assignment]


class NpuDeepseekV4MLPMXFP8(nn.Module):
    """Clamped-SwiGLU FFN with MXFP8 projections (shared experts / dense MLP).

    Counterpart of ``DeepseekV3MLP`` in its W8A8 configuration for official
    FP8 checkpoints: ``gate_up`` is sharded on the output dim, ``down`` on the
    input dim, and the final partial is reduced over the TP group. Weights are
    e4m3 with ue8m0 128x128 block scales, so -- unlike ``DeepseekV3MLP`` --
    there is no ``dtype`` parameter.
    """

    def __init__(
        self,
        cfg: DeepseekV4LayerConfig,
        intermediate_size: int,
        device: torch.device,
        skip_tp_reduce: bool = False,
        tp_override: int | None = None,
        swiglu_limit: float = 0.0,
    ) -> None:
        super().__init__()
        tp = tp_override if tp_override is not None else cfg.tp_size
        assert intermediate_size % tp == 0, f"intermediate_size {intermediate_size} not divisible by tp {tp}"
        self.tp_size = tp
        self.skip_tp_reduce = skip_tp_reduce
        self.swiglu_limit = swiglu_limit
        inter_local = intermediate_size // tp
        self.gate_up_proj = NpuDeepseekV4DynamicLinearMXFP8(cfg.hidden_size, 2 * inter_local, device)
        # down_proj holds its K (intermediate) dim pre-sharded by the loader
        # (checkpoint full width -> narrow(dim=1)); N keeps the full hidden.
        self.down_proj = NpuDeepseekV4DynamicLinearMXFP8(inter_local, cfg.hidden_size, device)

    def process_weights_after_loading(self) -> None:
        self.gate_up_proj.process_weights_after_loading()
        self.down_proj.process_weights_after_loading()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        assert kernels is not None, "MXFP8 MLP requires the NPU kernels module"
        if 0.0 < self.swiglu_limit < 1_000_000.0:
            act = kernels.clipped_swiglu(gate_up, self.swiglu_limit)
        else:
            act = kernels.silu_and_mul(gate_up)
        out = self.down_proj(act)
        if self.tp_size > 1 and not self.skip_tp_reduce:
            distributed.tp_all_reduce(out)
        return out


__all__ = ["NpuDeepseekV4MLPMXFP8"]
