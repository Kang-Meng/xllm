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

"""CPU Engram precision contracts for device-side prefix block aliases.

Runs the production model hashing method, including its rolling slot cache.
Does not claim full-model logit accuracy or execution of the NPU operators.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from xllm.python.models.deepseek_v41 import DeepseekV41Model
from xllm.python.models.deepseek_v41_engram import (
    EngramHashState,
    EngramLayout,
    compute_hash_multipliers,
)

_BLOCK_SIZE = 128


def _hash_state() -> EngramHashState:
    layout = EngramLayout(
        layer_ids=[1, 14],
        num_embeddings=[1024, 1024],
        max_ngram_size=3,
        n_heads=2,
        head_dim=32,
        vocab_size=64,
        compressed_vocab_size=16,
        pad_token_id=0,
    )
    return EngramHashState(
        layout,
        torch.arange(64, dtype=torch.int64) % 16,
        compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, 16),
    )


def _compute(
    state: EngramHashState,
    cache: torch.Tensor,
    ids: torch.Tensor,
    positions: torch.Tensor,
    table: torch.Tensor,
    slots: torch.Tensor,
    q_lens: list[int],
) -> torch.Tensor:
    group = SimpleNamespace(cache=cache, block_size=_BLOCK_SIZE, block_table=table, slot_mapping=slots)
    backend = SimpleNamespace(swa_group_metadata=lambda dsa: group)
    metadata = SimpleNamespace(dsa_metadata=SimpleNamespace(seq_lens_q=torch.tensor(q_lens, dtype=torch.int32)))
    hashes, mask = DeepseekV41Model._compute_engram_hashes(
        SimpleNamespace(engram_hash=state), backend, metadata, ids, positions
    )
    assert hashes is not None
    assert mask is not None and mask.all()
    return hashes


def _cold_hashes(ids: torch.Tensor) -> torch.Tensor:
    return _hash_state()(
        ids,
        torch.arange(ids.numel(), dtype=torch.int64),
        torch.tensor([0, ids.numel()], dtype=torch.int64),
        torch.zeros(ids.numel(), dtype=torch.bool),
        None,
        None,
        None,
        _BLOCK_SIZE,
    )


def _tokens(size: int, offset: int = 0) -> torch.Tensor:
    return (torch.arange(size, dtype=torch.int64) * 7 + offset) % 63 + 1


@pytest.mark.parametrize("prefix", [256, 512, 768, 1024])
@pytest.mark.parametrize("suffix", [1, 2, 3, 127, 128, 129])
def test_restored_alias_hashes_match_full_context(prefix: int, suffix: int) -> None:
    """Source and hit share only completed blocks; new writes remain private."""
    state = _hash_state()
    prefix_blocks = prefix // _BLOCK_SIZE
    suffix_blocks = (suffix + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    cache = torch.zeros(prefix_blocks + suffix_blocks + 1, _BLOCK_SIZE, 1)
    ids = _tokens(prefix + suffix)
    source_table = torch.arange(prefix_blocks, dtype=torch.int32).view(1, -1)
    _compute(state, cache, ids[:prefix], torch.arange(prefix), source_table, torch.arange(prefix), [prefix])
    original_slots = state._cache[:prefix].clone()

    # The old logical columns may be sparse, but the last restored block is
    # the same physical block; this preserves the two-token n-gram history.
    hot_table = torch.full((1, prefix_blocks + suffix_blocks), -1, dtype=torch.int32)
    hot_table[0, prefix_blocks - 1] = prefix_blocks - 1
    hot_table[0, prefix_blocks:] = torch.arange(prefix_blocks + 1, prefix_blocks + suffix_blocks + 1)
    positions = torch.arange(prefix, prefix + suffix)
    slots = hot_table[0, positions // _BLOCK_SIZE].to(torch.int64) * _BLOCK_SIZE + positions % _BLOCK_SIZE
    actual = _compute(state, cache, ids[prefix:], positions, hot_table, slots, [suffix])

    assert torch.equal(actual, _cold_hashes(ids)[prefix:])
    assert torch.equal(state._cache[:prefix], original_slots)
    assert state._kv_cache_ref() is cache


@pytest.mark.parametrize("prefix", [256, 512, 768, 1024])
def test_interleaved_hits_keep_shared_history_and_private_continuations(prefix: int) -> None:
    """Two live requests cannot overwrite the restored prefix or each other."""
    state = _hash_state()
    blocks = prefix // _BLOCK_SIZE
    cache = torch.zeros(blocks + 2, _BLOCK_SIZE, 1)
    common = _tokens(prefix)
    table = torch.arange(blocks, dtype=torch.int32).view(1, -1)
    _compute(state, cache, common, torch.arange(prefix), table, torch.arange(prefix), [prefix])
    saved_prefix = state._cache[:prefix].clone()
    tails = [_tokens(3, offset=11), _tokens(3, offset=23)]
    golden = [_cold_hashes(torch.cat((common, tail)))[prefix:] for tail in tails]
    hot_tables = torch.full((2, blocks + 1), -1, dtype=torch.int32)
    hot_tables[:, blocks - 1] = blocks - 1
    hot_tables[:, blocks] = torch.tensor([blocks, blocks + 1], dtype=torch.int32)

    for step in range(3):
        ids = torch.stack([tail[step] for tail in tails])
        positions = torch.full((2,), prefix + step, dtype=torch.int64)
        slots = hot_tables[:, -1].to(torch.int64) * _BLOCK_SIZE + step
        actual = _compute(state, cache, ids, positions, hot_tables, slots, [1, 1])
        assert torch.equal(actual, torch.stack([expected[step] for expected in golden]))
        assert torch.equal(state._cache[:prefix], saved_prefix)
