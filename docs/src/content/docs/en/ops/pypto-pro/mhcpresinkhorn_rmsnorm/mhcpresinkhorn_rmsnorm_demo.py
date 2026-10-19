#!/usr/bin/env python3
# coding: utf-8

# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""Runnable example for mhcpresinkhorn_rmsnorm (v4-K + fused RmsNorm).

Usage (from the xllm repository root, in a configured CANN / PyPTO-Pro
environment):

    TILE_FWK_DEVICE_ID=0 PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa \
        python3 docs/src/content/docs/en/ops/pypto-pro/mhcpresinkhorn_rmsnorm/mhcpresinkhorn_rmsnorm_demo.py 3

The optional trailing argument is the token count `t` (default 3). The script
builds random sample inputs, invokes the operator, and compares all four
outputs against a PyTorch reference chain using per-output tolerances. The
first call triggers JIT compilation.
"""

import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

HC, D, MIX_HC, HC_D = 4, 4096, 24, 16384
EPS, RMS_EPS = 1e-6, 1e-6

# The pypto-pro directory contains a hyphen and is not a Python package, so the
# operator module is loaded by file path instead of a plain import.
_KERNEL_PATH = (Path(__file__).resolve().parents[8]
                / "xllm/python/kernels_npu/pypto-pro/mhcpresinkhorn_rmsnorm.py")


def load_kernel():
    """Load the mhcpresinkhorn_rmsnorm module from its file path."""
    if not _KERNEL_PATH.is_file():
        raise FileNotFoundError(f"kernel not found: {_KERNEL_PATH}")
    spec = importlib.util.spec_from_file_location(
        "_mhcpresinkhorn_rmsnorm", _KERNEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load kernel from {_KERNEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(x, hc_fn, hc_scale, hc_base, gamma):
    """Torch reference chain: fused gating (fp32 compute chain) + RmsNorm."""
    xf = x.float().reshape(x.shape[0], HC_D)
    r = torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + EPS)
    res = (xf @ hc_fn.float().T) * r
    pre = torch.sigmoid(res[:, 0:HC] * hc_scale[0] + hc_base[0:HC]) + EPS
    post = torch.sigmoid(res[:, HC:2 * HC] * hc_scale[1]
                         + hc_base[HC:2 * HC]) * 2.0
    comb_flag = (res[:, 2 * HC:].reshape(-1, HC, HC) * hc_scale[2]
                 + hc_base[2 * HC:].reshape(1, HC, HC))
    y = (x.float() * pre.reshape(-1, HC, 1)).sum(dim=1).to(torch.bfloat16)
    t0 = torch.softmax(comb_flag, dim=-1) + EPS
    comb = t0 / (t0.sum(dim=-2, keepdim=True) + EPS)
    for _ in range(19):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + EPS)
    yf = y.float()
    rstd = torch.rsqrt((yf * yf).mean(dim=-1, keepdim=True) + RMS_EPS)
    y_norm = (yf * rstd * gamma.float()).to(torch.bfloat16)
    return y_norm, rstd, post, comb


def main():
    # Optional token count; the default covers the multi-token path, and
    # t=1 / t=16 / t=17 exercise the single-token and tile-group boundaries.
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    dev = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))}"
    torch.npu.set_device(dev)

    # ---- Build inputs (random data, for demo purposes) ----
    g = torch.Generator(device="cpu").manual_seed(42)
    x = torch.empty((t, HC, D), dtype=torch.bfloat16).uniform_(-1, 1, generator=g)
    hc_fn = torch.empty((MIX_HC, HC_D), dtype=torch.bfloat16).uniform_(
        -1, 1, generator=g)
    hc_scale = torch.empty((3,), dtype=torch.float32).uniform_(-1, 1, generator=g)
    hc_base = torch.empty((MIX_HC,), dtype=torch.float32).uniform_(-1, 1, generator=g)
    gamma = torch.empty((D,), dtype=torch.bfloat16).uniform_(0.5, 1.5, generator=g)
    x, hc_fn = x.to(dev), hc_fn.to(dev)
    hc_scale, hc_base, gamma = (hc_scale.to(dev), hc_base.to(dev), gamma.to(dev))

    # ---- Invoke the operator (first call includes JIT compilation) ----
    kernel = load_kernel()
    y, rstd, post, comb = kernel.mhcpresinkhorn_rmsnorm(
        x, hc_fn, hc_scale, hc_base, gamma)
    torch.npu.synchronize()
    print(f"t={t} | y_norm {tuple(y.shape)}/{y.dtype} | rstd {tuple(rstd.shape)}"
          f" | post {tuple(post.shape)} | comb {tuple(comb.shape)}")

    # ---- Compare against the reference chain using per-output tolerances ----
    yr, rr, pr, cr = reference(x.cpu(), hc_fn.cpu(), hc_scale.cpu(),
                               hc_base.cpu(), gamma.cpu())
    checks = (("y_norm", y.float().cpu(), yr.float(), 1e-2, 1e-2),
              ("rstd", rstd.cpu(), rr, 1e-4, 5e-3),
              ("post", post.cpu(), pr, 2.5e-5, 5e-3),
              ("comb", comb.cpu(), cr, 2.5e-5, 5e-3))
    check_results = []
    for name, got, ref, atol, rtol in checks:
        ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
        check_results.append(ok)
        diff = (got - ref).abs().max().item()
        print(f"  {name:6s}: {'PASS' if ok else 'FAIL'}  max_abs={diff:.3e}")
    print("[DEMO_" + ("PASS" if all(check_results) else "FAIL") + "]")
    return all(check_results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
