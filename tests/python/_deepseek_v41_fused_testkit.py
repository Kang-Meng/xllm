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

"""Shared helpers for the DeepSeek-V4.1 fused-op test modules.

Extracted from the original monolithic ``test_deepseek_v41_fused_ops.py`` so each
fused-op block (HC/NORM/ROPE, SHARED_ACT, Q_NORM_QUANT, MC2, QLI) can live
in its own test file without duplicating infrastructure.
"""

from __future__ import annotations

import torch

import xllm.python.models.deepseek_v41 as v41

# ---------------------------------------------------------------------------
# Shared geometry constants (V4.1 tiny/serving geometry)
# ---------------------------------------------------------------------------
_N_LAYERS = 3
_COMPRESS_RATIOS = [0, 2, 1]
_KV_SOURCE_LAYERS = [1]
_INDEX_SOURCE_LAYERS = [1, 2]
_WINDOW = 128
_HEAD_DIM = 512
_NUM_HEADS = 8
_INDEX_TOPK = 4
_BLOCK_SIZE = 128
_CAPACITY = 256
_ROWS = 4

# hc_pre modes
_MODE_EAGER = v41._HC_PRE_MODE_EAGER
_MODE_FUSED = v41._HC_PRE_MODE_FUSED
_HC_PRE_MODES = (_MODE_EAGER,)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _npu_ready() -> bool:
    try:
        return torch.npu.is_available()
    except Exception:  # noqa: BLE001
        return False


def _load_fused_ops_lib() -> bool:
    """Whether the standalone fused-ops dispatcher ops are registered."""
    return all(
        getattr(torch.ops.xllm_ops, name, None) is not None
        for name in ("hc_post", "hc_pre", "rms_norm", "npu_inplace_partial_rotary_mul")
    )


# ---------------------------------------------------------------------------
# Spy logger (used by HC_PRE, SHARED_ACT, QLI blocks)
# ---------------------------------------------------------------------------
class _SpyLogger:
    """Capture ``logger.warning`` / ``logger.info`` calls for assertion."""

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def info(self, message: str) -> None:
        self.infos.append(message)
