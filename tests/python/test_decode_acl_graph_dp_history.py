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

"""ACL graph admission with full index history and uneven DP shards."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner


def _runner(
    dp_rank: int,
    *,
    dp_size: int = 2,
    page_size: int = 128,
    is_spec_draft: bool = False,
) -> DecodeAclGraphRunner:
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=page_size, is_mla=False),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=65536,
        dp_size=dp_size,
        dp_rank=dp_rank,
        is_spec_draft=is_spec_draft,
    )
    runner.layer_caches = [SimpleNamespace(index=torch.empty(1), conv=None)]
    return runner


def _metadata(
    local_length: int,
    global_lengths: tuple[int, ...],
    *,
    page_size: int = 128,
    table_columns: int | None = None,
    is_dummy: bool = False,
) -> SimpleNamespace:
    columns = table_columns if table_columns is not None else (local_length + page_size - 1) // page_size
    return SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        is_dummy=is_dummy,
        dp_execution_token_counts=(1, 1),
        dp_is_decode=(1, 1),
        dp_global_kv_max_seq_lens=global_lengths,
        slot_mapping=torch.tensor([0], dtype=torch.int32),
        new_cache_slots_host_values=[0],
        paged_kv_indptr=torch.tensor([0, columns], dtype=torch.int32),
        paged_kv_indices=torch.arange(columns, dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([((local_length - 1) % page_size) + 1], dtype=torch.int32),
        block_table=torch.arange(columns, dtype=torch.int32).reshape(1, columns),
        kv_seq_lens=torch.tensor([local_length], dtype=torch.int32),
        kv_seq_lens_host_values=[local_length],
        max_seq_len=local_length,
        q_cu_seq_lens=torch.tensor([0, 1], dtype=torch.int32),
        kv_cu_seq_lens=torch.tensor([0, local_length], dtype=torch.int32),
        linear_state_indices=torch.tensor([0], dtype=torch.int32),
        has_initial_state=torch.tensor([True]),
        expanded_decode_metadata=None,
        multi_block_tables=(),
    )


@pytest.mark.parametrize("active_dp", [0, 1])
@pytest.mark.parametrize("length", [32767, 32768, 32769, 65536])
@pytest.mark.parametrize("is_spec_draft", [False, True])
def test_long_index_history_keeps_both_dp_ranks_graphable(active_dp: int, length: int, is_spec_draft: bool) -> None:
    input_ids = torch.tensor([1], dtype=torch.int32)
    for rank in range(2):
        metadata = _metadata(length if rank == active_dp else 1, (), is_dummy=rank != active_dp)
        del metadata.dp_global_kv_max_seq_lens
        assert _runner(rank, is_spec_draft=is_spec_draft).can_execute(input_ids, metadata)


@pytest.mark.parametrize("lengths", [(1024, 32769), (32769, 65536)])
def test_nonempty_dp_histories_above_old_limit_remain_graphable(lengths: tuple[int, int]) -> None:
    input_ids = torch.tensor([1], dtype=torch.int32)
    for rank, length in enumerate(lengths):
        metadata = _metadata(length, lengths)
        assert _runner(rank).can_execute(input_ids, metadata)


@pytest.mark.parametrize("is_spec_draft", [False, True])
def test_tp_only_history_above_old_limit_remains_graphable(is_spec_draft: bool) -> None:
    runner = _runner(0, dp_size=1, is_spec_draft=is_spec_draft)
    assert runner.can_execute(torch.tensor([1], dtype=torch.int32), _metadata(65536, ()))


def test_mtp_target_keeps_existing_graph_restriction() -> None:
    for rank in range(2):
        runner = _runner(rank)
        runner.num_decoding_tokens = 4
        assert not runner.can_execute(torch.tensor([1], dtype=torch.int32), _metadata(1, ()))
