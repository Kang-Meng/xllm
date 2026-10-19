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

"""Golden layout tests for the DeepSeek-V4.1 image block slots.

LOCKED PAIR twin of ``tests/processors/deepseek_v41_prompt_processor_test.cpp``:
both pin the image-block role layout to the same golden masks, so the Python
``_vision_block_slots`` (consumed by ``encode``) and the C++
``DeepseekV41PromptProcessor::image_block_roles`` (consumed by ``process`` /
``adjust_token_layout`` / ``find_mm_spans``) cannot drift apart. Any layout
change must regenerate both golden tables in the same commit.

Pure-Python: does not load compiled operators.
"""

from __future__ import annotations

import pytest

from xllm.python.models.deepseek_v41 import (
    _IMAGE,
    _IMAGE_END,
    _IMAGE_NEWLINE,
    _IMAGE_PAD,
    _IMAGE_START,
    _vision_block_slots,
)

# (n_vit_h, n_vit_w, golden mask) triples. The mask is the per-image block
# role sequence: 'S' for a sentinel slot (any non-pad role, token 129264) and
# 'P' for a compressor-alignment pad (token 129265). n_llm = ceil(n_vit / 3).
LAYOUT_CASES = [
    (6, 6, "SSSSSSSPPS"),
    (9, 9, "SSSSSSSSSSPSPSPSPS"),
    (3, 3, "SSPSPS"),
    (6, 15, "SSSSSSSSSSSSSS"),
    (7, 8, "SSSSSSSSSSPSPSPSPS"),
    (12, 9, "SSSSSSSSSSSSSSSSSS"),
    (9, 3, "SSSSSSPSPS"),
    (5, 5, "SSSSSSSPPS"),
]


def _ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


@pytest.mark.parametrize(("n_vit_h", "n_vit_w", "mask"), LAYOUT_CASES)
def test_block_slots_match_cpp_golden(n_vit_h: int, n_vit_w: int, mask: str) -> None:
    n_llm_h = _ceil_div(n_vit_h, 3)
    n_llm_w = _ceil_div(n_vit_w, 3)
    slots = _vision_block_slots(n_llm_h, n_llm_w)
    got = "".join("S" if role != _IMAGE_PAD else "P" for role, _ in slots)
    assert got == mask


@pytest.mark.parametrize(("n_vit_h", "n_vit_w", "mask"), LAYOUT_CASES)
def test_block_slots_feature_rows_are_dense_and_ordered(n_vit_h: int, n_vit_w: int, mask: str) -> None:
    """Every aligner row appears exactly once, in the interleaved slot order.

    This locks the row order ``encode`` emits; the C++ side scatters those
    rows at the matching sentinel positions, so the permutation must be
    exactly [perm index of each IMAGE slot].
    """
    n_llm_h = _ceil_div(n_vit_h, 3)
    n_llm_w = _ceil_div(n_vit_w, 3)
    slots = _vision_block_slots(n_llm_h, n_llm_w)
    feat_rows = [idx for role, idx in slots if role == _IMAGE]
    assert len(feat_rows) == n_llm_h * n_llm_w
    assert sorted(feat_rows) == list(range(n_llm_h * n_llm_w))
    # Every non-IMAGE slot carries no feature row.
    assert all(idx == -1 for role, idx in slots if role != _IMAGE)
    # Block ends with the IMAGE_END sentinel.
    assert slots[-1][0] == _IMAGE_END
    assert slots[0][0] == _IMAGE_START


@pytest.mark.parametrize("compress_pad", [0, 1, 2, 3])
def test_block_slots_compress_pad_prefixes_pads(compress_pad: int) -> None:
    slots = _vision_block_slots(2, 2, compress_pad)
    leading = [role for role, _ in slots[:compress_pad]]
    assert leading == [_IMAGE_PAD] * compress_pad
    # The block itself is the compress_pad=0 layout.
    assert slots[compress_pad:] == _vision_block_slots(2, 2)


def test_block_slots_roles_are_known_codes() -> None:
    for n_llm_h, n_llm_w, _ in LAYOUT_CASES:
        for role, _ in _vision_block_slots(n_llm_h, n_llm_w):
            assert role in (_IMAGE_START, _IMAGE_PAD, _IMAGE, _IMAGE_NEWLINE, _IMAGE_END)
