# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PD history restoration compared with an independent full-token hash pass."""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from xllm.python.models.deepseek_v41_engram import EngramHashState, EngramLayout


def _state() -> EngramHashState:
    layout = EngramLayout(
        layer_ids=[1],
        num_embeddings=[1000],
        max_ngram_size=4,
        n_heads=2,
        head_dim=32,
        vocab_size=64,
        compressed_vocab_size=16,
        pad_token_id=0,
    )
    return EngramHashState(layout, torch.arange(64) % 16, torch.tensor([[3, 5, 7, 11]]))


def test_cpp_shared_memory_history_matches_python_hash(tmp_path: Path) -> None:
    """Builder -> shared memory -> Python restore -> hash; CPU contract test."""
    binary = os.environ.get("XLLM_BATCH_TEST_BINARY")
    if not binary:
        pytest.skip("Set XLLM_BATCH_TEST_BINARY to the compiled batch_test")
    output = tmp_path / "engram_history.jsonl"
    env = dict(os.environ, XLLM_ENGRAM_CONTRACT_OUTPUT=str(output))
    subprocess.run(
        [binary, "--gtest_filter=BatchTest.DecodeEngramHistoryPreservesSparseSwaColumns"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 3
    for record in records:
        state = _state()
        block_size = record["block_size"]
        state.ensure_cache(torch.zeros((record["num_blocks"], block_size, 8)), block_size)
        state._cache.fill_(13)
        state.restore_history(torch.tensor(record["history"]).reshape(-1, 2))
        tokens = record["tokens"] + [record["next_token"]]
        position = len(tokens) - 1
        table = torch.tensor([record["block_table"]], dtype=torch.int32)
        block = record["block_table"][position // block_size]
        assert block >= 0
        actual = state(
            torch.tensor([tokens[-1]]),
            torch.tensor([position]),
            torch.tensor([0, 1]),
            torch.zeros(1, dtype=torch.bool),
            None,
            torch.tensor([block * block_size + position % block_size]),
            table,
            block_size,
        )
        oracle = _state()
        expected = oracle(
            torch.tensor(tokens),
            torch.arange(len(tokens)),
            torch.tensor([0, len(tokens)]),
            torch.zeros(len(tokens), dtype=torch.bool),
            None,
            None,
            None,
            block_size,
        )
        torch.testing.assert_close(actual, expected[-1:], atol=0, rtol=0)


@pytest.mark.parametrize("prefix", [1, 3, 7, 31, 127, 128, 129, 1323])
@pytest.mark.parametrize("width", [1, 6])
@pytest.mark.parametrize("lanes", [1, 4])
def test_pd_hashes_match_full_history(prefix: int, width: int, lanes: int) -> None:
    window = 128
    state = _state()
    kv = torch.zeros((lanes + 2, window, 8))
    state.ensure_cache(kv, window)
    # Receiver blocks differ from sender blocks; old slot contents are arbitrary.
    state._cache.fill_(13)
    table = torch.arange(lanes + 1, 1, -1, dtype=torch.int32).view(-1, 1)
    histories = [[(i * 7 + lane * 11) % 63 + 1 for i in range(prefix + width + 2)] for lane in range(lanes)]
    pairs = [
        [int(table[lane, 0]) * window + i % window, history[i]]
        for lane, history in enumerate(histories)
        for i in range(max(0, prefix - 3), prefix)
    ]
    state.restore_history(torch.tensor(pairs))
    for start, count in [(prefix, width), (prefix + width, 1), (prefix + width + 1, 1)]:
        positions = torch.arange(start, start + count).repeat(lanes)
        slots = torch.cat(
            [int(table[lane, 0]) * window + torch.arange(start, start + count) % window for lane in range(lanes)]
        )
        actual = state(
            torch.tensor([history[start : start + count] for history in histories]).flatten(),
            positions,
            torch.arange(lanes + 1) * count,
            torch.zeros(lanes * count, dtype=torch.bool),
            None,
            slots,
            table,
            window,
        )
        expected = []
        for history in histories:
            oracle = _state()
            full = oracle(
                torch.tensor(history[: start + count]),
                torch.arange(start + count),
                torch.tensor([0, start + count]),
                torch.zeros(start + count, dtype=torch.bool),
                None,
                None,
                None,
                window,
            )
            expected.append(full[start:])
        torch.testing.assert_close(actual, torch.cat(expected), atol=0, rtol=0)


def test_non_pd_ring_cache_matches_full_history() -> None:
    state = _state()
    window = 8
    ids = [(i * 7) % 63 + 1 for i in range(35)]
    state.ensure_cache(torch.zeros((3, window, 8)), window)
    table = torch.tensor([[2]], dtype=torch.int32)
    actual = []
    for position, token in enumerate(ids):
        actual.append(
            state(
                torch.tensor([token]),
                torch.tensor([position]),
                torch.tensor([0, 1]),
                torch.zeros(1, dtype=torch.bool),
                None,
                torch.tensor([2 * window + position % window]),
                table,
                window,
            )
        )
    oracle = _state()
    expected = oracle(
        torch.tensor(ids),
        torch.arange(len(ids)),
        torch.tensor([0, len(ids)]),
        torch.zeros(len(ids), dtype=torch.bool),
        None,
        None,
        None,
        window,
    )
    torch.testing.assert_close(torch.cat(actual), expected, atol=0, rtol=0)


def test_missing_swa_block_does_not_read_block_zero_cache() -> None:
    state = _state()
    window = 4
    state.ensure_cache(torch.zeros((2, window, 8)), window)
    state._cache.fill_(37)
    source = torch.tensor([23], dtype=torch.int64)
    result = state._cache_source(
        source,
        torch.tensor([0]),
        torch.tensor([4]),
        torch.tensor([False]),
        None,
        torch.tensor([[-1]], dtype=torch.int32),
        window,
    )
    assert result.tolist() == [-1]


def test_model_restore_uses_receiver_table() -> None:
    from xllm.python.models.deepseek_v41 import DeepseekV41Model

    state = _state()
    kv = torch.zeros((4, 8, 8))
    backend = SimpleNamespace(swa_history_cache=lambda: kv)
    model = SimpleNamespace(engram_hash=state)
    metadata = SimpleNamespace(engram_history=[26, 9, 27, 12])
    DeepseekV41Model.restore_engram_history(model, backend, metadata)
    assert state._cache[3 * 8 + 2 : 3 * 8 + 4].tolist() == [9, 12]
    DeepseekV41Model.restore_engram_history(model, backend, SimpleNamespace())


def test_expanded_verify_reads_current_candidate_history() -> None:
    state = _state()
    window = 128
    prefix = 1323
    width = 6
    ids = [(i * 7) % 63 + 1 for i in range(prefix + width)]
    kv = torch.zeros((3, window, 8))
    state.ensure_cache(kv, window)
    pairs = [[2 * window + i % window, ids[i]] for i in range(prefix - 3, prefix)]
    state.restore_history(torch.tensor(pairs))
    positions = torch.arange(prefix, prefix + width)
    actual = state(
        torch.tensor(ids[prefix:]),
        positions,
        torch.arange(width + 1),
        torch.zeros(width, dtype=torch.bool),
        None,
        2 * window + positions % window,
        torch.full((width, 1), 2, dtype=torch.int32),
        window,
    )
    oracle = _state()
    expected = oracle(
        torch.tensor(ids),
        torch.arange(len(ids)),
        torch.tensor([0, len(ids)]),
        torch.zeros(len(ids), dtype=torch.bool),
        None,
        None,
        None,
        window,
    )[prefix:]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_npu_restored_hashes_match_cpu_oracle() -> None:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    device = torch.device("npu:0")
    prefix, width, window = 1323, 6, 128
    ids = [(i * 7) % 63 + 1 for i in range(prefix + width)]
    state = _state().to(device)
    kv = torch.zeros((3, window, 8), device=device)
    state.ensure_cache(kv, window)
    state._cache.fill_(13)
    state.restore_history(torch.tensor([[2 * window + i % window, ids[i]] for i in range(prefix - 3, prefix)]))
    positions = torch.arange(prefix, prefix + width, device=device)
    actual = state(
        torch.tensor(ids[prefix:], device=device),
        positions,
        torch.arange(width + 1, device=device),
        torch.zeros(width, dtype=torch.bool, device=device),
        None,
        2 * window + positions % window,
        torch.full((width, 1), 2, dtype=torch.int32, device=device),
        window,
    )
    oracle = _state()
    expected = oracle(
        torch.tensor(ids),
        torch.arange(len(ids)),
        torch.tensor([0, len(ids)]),
        torch.zeros(len(ids), dtype=torch.bool),
        None,
        None,
        None,
        window,
    )[prefix:]
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


def test_model_restore_rejects_odd_engram_history() -> None:
    from xllm.python.models.deepseek_v41 import DeepseekV41Model

    state = _state()
    backend = SimpleNamespace(swa_history_cache=lambda: torch.zeros((4, 8, 8)))
    model = SimpleNamespace(engram_hash=state)
    metadata = SimpleNamespace(engram_history=[26, 9, 27])
    with pytest.raises(ValueError, match="slot, token"):
        DeepseekV41Model.restore_engram_history(model, backend, metadata)


def test_engram_decode_batch_skips_empty_late_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    window = 128
    state.ensure_cache(torch.zeros((4, window, 8)), window)
    num_tokens = 200
    calls: list[torch.Tensor] = []
    original = state._write_hash_cache

    def record(comp, slots, *, deduplicate=False):
        calls.append(slots.clone())
        return original(comp, slots, deduplicate=deduplicate)

    monkeypatch.setattr(state, "_write_hash_cache", record)
    state(
        torch.ones(num_tokens, dtype=torch.int64),
        torch.arange(num_tokens),
        torch.arange(num_tokens + 1),
        torch.zeros(num_tokens, dtype=torch.bool),
        None,
        torch.arange(num_tokens),
        None,
        window,
        max_chunk_len=1,
    )
    assert len(calls) == 1
    assert (calls[0] >= 0).all()


def test_engram_mixed_batch_late_publish_only_long_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    window = 128
    state.ensure_cache(torch.zeros((4, window, 8)), window)
    long_len = 200
    num_tokens = long_len + 1
    calls: list[torch.Tensor] = []
    original = state._write_hash_cache

    def record(comp, slots, *, deduplicate=False):
        calls.append(slots.clone())
        return original(comp, slots, deduplicate=deduplicate)

    monkeypatch.setattr(state, "_write_hash_cache", record)
    state(
        torch.ones(num_tokens, dtype=torch.int64),
        torch.arange(num_tokens),
        torch.tensor([0, long_len, num_tokens]),
        torch.zeros(num_tokens, dtype=torch.bool),
        None,
        torch.arange(num_tokens),
        None,
        window,
        max_chunk_len=long_len,
    )
    assert len(calls) == 2
    early, late = calls
    assert early.shape == (num_tokens,)
    assert (early[:long_len] == -1).all()
    assert (early[long_len:] >= 0).all()
    assert (late[:long_len] >= 0).all()
    assert (late[long_len:] == -1).all()
