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

"""Tests for the NPU ACL decode-graph runner."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from xllm.python.attention.dsa_metadata import build_cache_specs
from xllm.python.model_executor.runners.decode_acl_graph import (
    DecodeAclGraphRunner,
)


def _runner() -> DecodeAclGraphRunner:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    return DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
    )


def _metadata(linear_state_indices: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        slot_mapping=torch.arange(4, dtype=torch.int32),
        paged_kv_indptr=torch.arange(5, dtype=torch.int32),
        paged_kv_indices=torch.tensor([10, 20, 30, 40], dtype=torch.int32),
        paged_kv_last_page_len=torch.arange(1, 5, dtype=torch.int32),
        block_table=torch.tensor(
            [[10, 0], [20, 0], [30, 0], [40, 0]],
            dtype=torch.int32,
        ),
        kv_seq_lens=torch.arange(1, 5, dtype=torch.int32),
        kv_seq_lens_host_values=[1, 2, 3, 4],
        kv_cu_seq_lens=torch.tensor([0, 1, 3, 6, 10], dtype=torch.int32),
        linear_state_indices=linear_state_indices,
        expanded_decode_metadata=None,
    )


def test_linear_state_indices_use_stable_graph_buffer() -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.tensor([3, 7, 11, 15], dtype=torch.int32))
    entry = runner._allocate_entry(
        padded_batch_size=8,
        input_ids=input_ids,
        positions=positions,
        metadata=metadata,
    )
    static_indices = entry.static_metadata.linear_state_indices
    data_ptr = static_indices.data_ptr()

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ):
        runner._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size=4,
            input_embedding=None,
        )
        assert static_indices.tolist() == [3, 7, 11, 15, 0, 0, 0, 0]

        metadata.linear_state_indices = torch.tensor(
            [4, 8, 12, 16],
            dtype=torch.int32,
        )
        runner._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size=4,
            input_embedding=None,
        )

    assert static_indices.data_ptr() == data_ptr
    assert static_indices.tolist() == [4, 8, 12, 16, 0, 0, 0, 0]


@pytest.mark.parametrize("dp_size", [1, 2])
def test_fill_entry_refreshes_mega_moe_mask_in_place(dp_size: int) -> None:
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
        dp_size=dp_size,
        enable_mega_moe_token_mask=True,
    )
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    entry = runner._allocate_entry(
        padded_batch_size=8,
        input_ids=input_ids,
        positions=positions,
        metadata=metadata,
    )
    mask = entry.static_metadata.mega_moe_token_mask
    data_ptr = mask.data_ptr()

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ):
        for remote_count in (3, 1):
            metadata.dp_execution_token_counts = (4, remote_count)[:dp_size]
            runner._fill_entry(entry, input_ids, positions, metadata, batch_size=4, input_embedding=None)
            expected = [True] * 4 + [False] * 4
            if dp_size == 2:
                expected += [True] * remote_count + [False] * (8 - remote_count)
            assert mask.tolist() == expected
            assert entry.static_metadata.mega_moe_token_mask.data_ptr() == data_ptr


def test_dsa_graph_tables_use_compressed_block_counts() -> None:
    _, group_infos = build_cache_specs([1, 4, 128], 128, 3)
    attention_backend = SimpleNamespace(
        page_size=128,
        is_mla=False,
        group_infos=group_infos,
    )
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=8,
        max_model_len=32768,
    )
    runner._max_blocks_per_sequence = 256

    tables = runner._build_static_multi_block_tables(4, torch.device("cpu"))

    assert [tuple(table.shape) for table in tables] == [(4, 256), (4, 64), (4, 2)]
    assert all(torch.all(table == -1) for table in tables)


def test_dsa_graph_refreshes_every_manager_and_clears_tails() -> None:
    _, group_infos = build_cache_specs([1, 4, 128], 128, 3)
    attention_backend = SimpleNamespace(
        page_size=128,
        is_mla=False,
        group_infos=group_infos,
    )
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=4,
        max_model_len=512,
    )
    static_metadata = SimpleNamespace(
        multi_block_tables=(
            torch.full((4, 4), 99, dtype=torch.int32),
            torch.full((4, 2), 99, dtype=torch.int32),
            torch.full((4, 1), 99, dtype=torch.int32),
        )
    )
    metadata = SimpleNamespace(
        multi_block_tables=(
            torch.tensor([[10, 11], [12, 13]], dtype=torch.int32),
            torch.tensor([[20], [21]], dtype=torch.int32),
            torch.tensor([[30], [31]], dtype=torch.int32),
        )
    )

    runner._fill_dsa_block_tables(
        static_metadata,
        metadata,
        torch.empty(0, dtype=torch.int32),
        batch_size=2,
    )

    assert static_metadata.multi_block_tables[0].tolist() == [
        [10, 11, -1, -1],
        [12, 13, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
    ]
    assert static_metadata.multi_block_tables[1].tolist() == [
        [20, -1],
        [21, -1],
        [-1, -1],
        [-1, -1],
    ]
    assert static_metadata.multi_block_tables[2].tolist() == [[30], [31], [-1], [-1]]

    metadata.multi_block_tables = (
        torch.tensor([[40]], dtype=torch.int32),
        torch.tensor([[50]], dtype=torch.int32),
        torch.tensor([[60]], dtype=torch.int32),
    )
    runner._fill_dsa_block_tables(
        static_metadata,
        metadata,
        torch.empty(0, dtype=torch.int32),
        batch_size=1,
    )

    assert static_metadata.multi_block_tables[0].tolist() == [
        [40, -1, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
    ]
    assert static_metadata.multi_block_tables[1].tolist() == [
        [50, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
    ]
    assert static_metadata.multi_block_tables[2].tolist() == [[60], [-1], [-1], [-1]]


def test_dsa_graph_padding_uses_empty_kv_lengths() -> None:
    _, group_infos = build_cache_specs([1, 4, 128], 128, 3)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(
            page_size=128,
            is_mla=False,
            group_infos=group_infos,
        ),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=512,
    )
    entry = SimpleNamespace(
        batch_size=4,
        static_metadata=SimpleNamespace(kv_seq_lens_host_values=[1, 1, 1, 1]),
    )

    runner._fill_host_metadata(entry, [9, 17], batch_size=2)

    assert entry.static_metadata.kv_seq_lens_host_values == [9, 17, 0, 0]


def test_dsa_graph_positions_are_refreshed_from_current_input() -> None:
    runner = _runner()
    entry = SimpleNamespace(
        batch_size=4,
        static_positions=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        static_metadata=SimpleNamespace(dsa_positions=None),
    )

    runner._fill_graph_dsa_positions(
        entry,
        torch.tensor([20, 21], dtype=torch.int32),
    )

    assert entry.static_metadata.dsa_positions.tolist() == [20, 21, 0, 0]


def _dp_metadata(
    token_counts: tuple[int, int],
    dp_is_decode: tuple[int, int] = (1, 1),
) -> SimpleNamespace:
    return SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        dp_execution_token_counts=tuple(1 if count == 0 else count for count in token_counts),
        dp_is_decode=dp_is_decode,
    )


def test_dp_empty_rank_uses_group_wide_acl_graph_bucket() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=1,
    )

    with patch.object(
        runner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runner.can_execute(
            torch.zeros(1, dtype=torch.int32),
            _dp_metadata((5, 0)),
        )


def test_dp_mixed_step_does_not_enter_acl_decode_graph() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
    )

    with patch.object(
        runner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert not runner.can_execute(
            torch.zeros(3, dtype=torch.int32),
            _dp_metadata((3, 2), dp_is_decode=(0, 1)),
        )


def test_dp_acl_graph_requires_group_wide_token_counts() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
    )
    metadata = _dp_metadata((3, 2))
    metadata.dp_execution_token_counts = (3,)

    with (
        patch.object(
            runner,
            "_has_compatible_decode_metadata",
            return_value=True,
        ),
        pytest.raises(RuntimeError, match="valid dp_execution_token_counts"),
    ):
        runner.can_execute(torch.zeros(3, dtype=torch.int32), metadata)
