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

"""NPU tests for the PyPTO-Pro mhcpresinkhorn_rmsnorm fused operator.

The host-side contract (argument validation, workspace cache, launch buffers)
is covered by ``test_mhcpresinkhorn_rmsnorm.py`` on CPU.  This module runs the
real kernel on an Ascend device and compares it against a pure-torch
re-implementation of the fused chain:

1. ``mix = x @ hc_fn``            (bf16 matmul, K = 4 * 4096 split over cores)
2. ``r = 1 / sqrt(mean(x^2) + eps)`` per token over all ``4 * 4096`` lanes
3. ``z = mix * r * hc_scale + hc_base``
4. ``pp = sigmoid(z) * pmul + eps`` for the 8 "comb" lanes, ``z`` for the 16
   "post" lanes
5. ``y = sum_m pp[m] * x[m]`` then ``y_norm = y * rstd * gamma``
6. Sinkhorn normalization (20 iterations) over the 4x4 comb matrix
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

torch_npu = pytest.importorskip(
    "torch_npu",
    reason="mhcpresinkhorn_rmsnorm tests require torch_npu",
)
pytest.importorskip(
    "pypto_pro",
    reason="mhcpresinkhorn_rmsnorm tests require the pypto_pro package",
)

_MODULE_NAME = "xllm.python.kernels_npu.pypto_pro_mhcpresinkhorn_rmsnorm_npu"
_SOURCE = (Path(__file__).parents[2] / "xllm" / "python" / "kernels_npu"
           / "pypto-pro" / "mhcpresinkhorn_rmsnorm.py")


def _load_operator():
    """Load the operator module from its file path.

    ``pypto-pro`` is not a valid Python package name, so the module cannot be
    imported by name.
    """
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


op = _load_operator()

HC = op.HC
D = op.D
MIX_HC = op.MIX_HC
HC_D = op.HC_D
EPS = op.EPS
RMS_EPS = op.RMS_EPS
SINKHORN_ITERS = op.SINKHORN_ITERS


def _pmul() -> torch.Tensor:
    """Per-lane multiplier used by the kernel (4x1, 4x2, 16x1)."""
    return torch.tensor([1.0] * 4 + [2.0] * 4 + [1.0] * 16, dtype=torch.float32)


def _reference(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    gamma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 reference of the whole fused chain."""
    t = x.shape[0]
    x_fp32 = x.float()
    x_2d = x_fp32.reshape(t, HC_D)

    # 1. mix = x @ hc_fn, computed in bf16 like the cube unit does.
    mix = (x.reshape(t, HC_D) @ hc_fn).float()

    # 2. r = 1 / sqrt(mean(x^2) + eps) over the whole 4 * 4096 row.
    ssq = x_2d.square().sum(dim=-1, keepdim=True)
    r = torch.rsqrt(ssq * (1.0 / HC_D) + EPS)

    # 3. z = mix * r * scale + base; scale is broadcast per 4/4/16 lanes.
    scale = torch.cat([
        hc_scale[0].expand(4),
        hc_scale[1].expand(4),
        hc_scale[2].expand(16),
    ])
    z = mix * r * scale + hc_base

    # 4. sigmoid on the 8 comb lanes, linear on the 16 post lanes.
    sig = torch.sigmoid(z[:, :8])
    pp = torch.cat([sig * _pmul()[:8] + EPS, z[:, 8:]], dim=-1)

    # 5. y = sum_m pp[m] * x[m], then RmsNorm with the fused gamma.
    y = torch.einsum("tm,tmd->td", pp[:, :HC], x_fp32)
    rstd = torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + RMS_EPS)
    y_norm = (y * rstd * gamma.float()).to(torch.bfloat16)

    # 6. Sinkhorn over the 4x4 comb matrix (row softmax, then 20 iterations).
    comb = pp[:, :HC * HC].reshape(t, HC, HC).clone()
    comb = torch.softmax(comb, dim=-1) + EPS
    comb = comb / (comb.sum(dim=-2, keepdim=True) + EPS)
    for _ in range(SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + EPS)

    post = pp[:, HC * HC:HC * HC + HC].contiguous()
    return y_norm, rstd, post, comb


def _inputs(t: int, seed: int = 2026) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return dict(
        x=torch.randn(t, HC, D, dtype=torch.bfloat16, device="npu"),
        hc_fn=torch.randn(MIX_HC, HC_D, dtype=torch.bfloat16, device="npu") * 0.02,
        hc_scale=torch.randn(3, dtype=torch.float32, device="npu"),
        hc_base=torch.randn(MIX_HC, dtype=torch.float32, device="npu"),
        gamma=torch.randn(D, dtype=torch.bfloat16, device="npu"),
    )


@pytest.mark.parametrize("t", [1, 15, 16, 33])
def test_matches_reference(t: int) -> None:
    torch_npu.npu.set_device(0)
    args = _inputs(t)

    y, rstd, post, comb = op.mhcpresinkhorn_rmsnorm(**args)
    torch_npu.npu.synchronize()

    ref_y, ref_rstd, ref_post, ref_comb = _reference(**args)
    assert (tuple(y.shape), y.dtype) == ((t, D), torch.bfloat16)
    assert (tuple(rstd.shape), rstd.dtype) == ((t, 1), torch.float32)
    assert (tuple(post.shape), post.dtype) == ((t, HC), torch.float32)
    assert (tuple(comb.shape), comb.dtype) == ((t, HC, HC), torch.float32)

    torch.testing.assert_close(y.cpu(), ref_y.cpu(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(rstd.cpu(), ref_rstd.cpu(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(post.cpu(), ref_post.cpu(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(comb.cpu(), ref_comb.cpu(), rtol=2e-3, atol=2e-3)


def test_sinkhorn_output_is_doubly_stochastic() -> None:
    torch_npu.npu.set_device(0)
    args = _inputs(8)

    _, _, _, comb = op.mhcpresinkhorn_rmsnorm(**args)
    torch_npu.npu.synchronize()

    comb = comb.cpu()
    assert torch.isfinite(comb).all()
    assert (comb > 0).all()
    # Every row and every column sums to ~1 + HC * eps.
    expected = 1.0 + HC * EPS
    torch.testing.assert_close(comb.sum(dim=-1), torch.full_like(comb[:, :, 0], expected),
                               rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(comb.sum(dim=-2), torch.full_like(comb[:, :, 0], expected),
                               rtol=1e-3, atol=1e-3)


def test_rmsnorm_output_matches_standalone_rmsnorm() -> None:
    torch_npu.npu.set_device(0)
    args = _inputs(4)

    y, rstd, _, _ = op.mhcpresinkhorn_rmsnorm(**args)
    torch_npu.npu.synchronize()

    # y is the normalized value, so mean(y^2) must be ~1 when gamma is all ones.
    ones = dict(args)
    ones["gamma"] = torch.ones(D, dtype=torch.bfloat16, device="npu")
    y_ones, rstd_ones, _, _ = op.mhcpresinkhorn_rmsnorm(**ones)
    torch_npu.npu.synchronize()

    mean_sq = y_ones.float().square().mean(dim=-1, keepdim=True)
    torch.testing.assert_close(mean_sq.cpu(), torch.ones_like(mean_sq).cpu(),
                               rtol=2e-2, atol=2e-2)
    # rstd must not depend on gamma.
    torch.testing.assert_close(rstd.cpu(), rstd_ones.cpu(), rtol=0, atol=0)
    assert torch.isfinite(y.float()).all()