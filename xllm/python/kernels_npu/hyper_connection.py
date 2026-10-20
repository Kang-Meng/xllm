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

"""NPU hyper-connection kernels for multi-stream residual (mHC) layers.

``hc_pre`` fuses rsqrt RMS + linear projection + sigmoid gating + Sinkhorn
normalisation + weighted-sum reduction into a single call. ``hc_post``
applies the post-attention/FFN recombination (post * x + comb^T @ residual).
``hc_pre_va`` (aclnnHcPreVa, sourced from the vllm-ascend open AscendC tree)
also returns the block's own ``pre`` vector and collapses with an EXTERNAL
``pre_mix``.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    hc_pre = torch.ops.xllm_ops.hc_pre
except (AttributeError, RuntimeError):
    hc_pre = None

try:
    hc_post = torch.ops.xllm_ops.hc_post
except (AttributeError, RuntimeError):
    hc_post = None


# The DSV4.1 ``hc_pre_va`` op (aclnnHcPreVa, from the vllm-ascend open AscendC
# tree, registered as ``xllm_ops::hc_pre_va``) consumes an optional external
# ``pre_mix`` for the collapse and additionally returns the block's own
# ``pre``:
# ``hc_pre_va(x, hc_fn, hc_scale, hc_base, pre_mix, hc_mult,
#            hc_sinkhorn_iters, norm_eps, hc_eps) -> (y, post, comb_frag, pre)``.
#
# The op's schema declares all nine parameters positional-or-keyword (there is
# no ``*``), so callers may pass them either positionally or by keyword. The thin
# wrapper below forwards them POSITIONALLY regardless, which keeps the call site
# unambiguous and independent of argument ordering; its own defaults mirror the
# op's public signature so external callers can still omit ``pre_mix`` and the
# geometry knobs.
def _make_hc_pre_va_forward(op: Any) -> Any:
    """Bind the registered op into a wrapper that always forwards positionally."""

    def hc_pre_va(
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        pre_mix: torch.Tensor | None = None,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        norm_eps: float = 1e-6,
        hc_eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return op(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            pre_mix,
            hc_mult,
            hc_sinkhorn_iters,
            norm_eps,
            hc_eps,
        )

    return hc_pre_va


# Resolve lazily: the package registers its operators at import time, and the
# kernels_npu module-load order is not a stable contract, so an eager probe
# here could latch ``None`` before the ops library is loaded. Only a successful
# resolution is cached, so a later registration is picked up.
_hc_pre_va_ = None


def _resolve_hc_pre_va():
    """Probe the ``xllm_ops::hc_pre_va`` op, returning the positional wrapper or ``None``.

    A failed probe is NOT cached: the op is registered by the NPU ops library
    at import time, so a later load can make it available and the next probe
    should pick it up (mirrors ``deepseek_v41._get_hc_pre_fused_op``'s
    re-probe).
    """
    global _hc_pre_va_
    if _hc_pre_va_ is not None:
        return _hc_pre_va_
    # The VA kernel already computes the next sublayer's pre vector. Use it
    # directly instead of rebuilding that vector with a BF16 projection.
    va_op = getattr(torch.ops.xllm_ops, "hc_pre_va", None)
    if va_op is not None:
        _hc_pre_va_ = _make_hc_pre_va_forward(va_op)
        return _hc_pre_va_
    return None


def __getattr__(name):
    if name == "hc_pre_va":
        return _resolve_hc_pre_va()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "hc_pre",
    "hc_pre_va",  # noqa: F822
    "hc_post",
]
