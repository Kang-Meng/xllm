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

"""Package stubs for pure-Python tests that run without platform kernels."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

_PYTHON_ROOT = Path(__file__).parents[2] / "xllm" / "python"


def _rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """CPU reference for weighted RMSNorm over the last dimension."""
    value_fp32 = value.float()
    variance = value_fp32.square().mean(dim=-1, keepdim=True)
    normalized = value_fp32 * torch.rsqrt(variance + eps)
    return (normalized * weight.float()).to(value.dtype)


def _rms_norm_sigmoid_gated(
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """CPU reference for kernels_npu rms_norm_sigmoid_gated.

    Matches the Triton kernel contract: RMSNorm over the last dim, scaled by
    ``weight`` and gated by ``sigmoid(gate)``. Used by pure-Python model
    tests (glm5_next KDA o_norm) that cannot link the NPU kernels.
    """
    input_dtype = value.dtype
    x = value.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x * weight.to(torch.float32) * gate.sigmoid()).to(input_dtype)


def _install_python_package_stub() -> None:
    kernels = types.ModuleType("xllm.python.kernels")
    kernels.rms_norm = _rms_norm
    kernels.rms_norm_sigmoid_gated = _rms_norm_sigmoid_gated
    kernels_npu = types.ModuleType("xllm.python.kernels_npu")
    kernels_npu.__path__ = [str(_PYTHON_ROOT / "kernels_npu")]
    distributed = types.ModuleType("xllm.python.distributed")
    distributed.dcp_group = lambda _device=None: None

    package = types.ModuleType("xllm.python")
    # Keep source submodules importable without executing the real package binding.
    package.__path__ = [str(_PYTHON_ROOT)]
    package.kernels = kernels
    package.kernels_npu = kernels_npu
    package.distributed = distributed

    distributed.tp_rank = lambda device: 0

    sys.modules["xllm.python"] = package
    sys.modules["xllm.python.kernels"] = kernels
    sys.modules["xllm.python.kernels_npu"] = kernels_npu
    sys.modules["xllm.python.distributed"] = distributed


_install_python_package_stub()
