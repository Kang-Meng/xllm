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

"""Empty-DP coverage for the NPU ACL decode-graph runner."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.dsa_metadata import build_cache_specs
from xllm.python.model_executor.runners.decode_acl_graph import (
    DecodeAclGraphRunner,
)


def _decode_metadata(is_dummy: bool, accepted: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        slot_mapping=torch.tensor([-1 if is_dummy else 4], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
        paged_kv_indices=torch.tensor([0 if is_dummy else 1], dtype=torch.int32),
        paged_kv_last_page_len=torch.ones(1, dtype=torch.int32),
        block_table=torch.tensor([[0 if is_dummy else 1]], dtype=torch.int32),
        kv_seq_lens=torch.ones(1, dtype=torch.int32),
        kv_seq_lens_host_values=[1],
        kv_cu_seq_lens=torch.tensor([0, 1], dtype=torch.int32),
        q_cu_seq_lens=torch.tensor([0, 1], dtype=torch.int32),
        linear_state_indices=torch.tensor([0 if is_dummy else 3], dtype=torch.int32),
        num_accepted_tokens=None if accepted is None else torch.tensor([accepted], dtype=torch.int32),
        expanded_decode_metadata=None,
        multi_block_tables=(),
        new_cache_slots_host_values=[] if is_dummy else [4],
        is_prefill=False,
        is_chunked_prefill=False,
        is_dummy=is_dummy,
    )


def _linear_runner(num_decoding_tokens: int = 4, linear: bool = True) -> DecodeAclGraphRunner:
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
        dp_size=2,
        num_decoding_tokens=num_decoding_tokens,
    )
    if linear:
        runner.bind_layer_caches([LayerCache(None, None, conv=torch.zeros(8, 1, 3), ssm=torch.zeros(8, 1, 1, 1))])
    return runner


@pytest.mark.parametrize("starts_empty", [True, False])
def test_accepted_buffer_survives_empty_and_busy_dp_transitions(starts_empty: bool) -> None:
    runner = _linear_runner()
    input_ids = torch.ones(1, dtype=torch.int32)
    positions = torch.zeros_like(input_ids)
    initial = _decode_metadata(starts_empty, None if starts_empty else 4)
    entry = runner._allocate_entry(4, input_ids, positions, initial)
    counts = entry.static_metadata.num_accepted_tokens
    assert counts is not None
    address = counts.data_ptr()
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        for metadata in (
            initial,
            _decode_metadata(True),
            _decode_metadata(False, 3),
            _decode_metadata(True, 4),
            _decode_metadata(False, 2),
        ):
            runner._fill_entry(entry, input_ids, positions, metadata, batch_size=1, input_embedding=None)
            expected = 1 if metadata.is_dummy else int(metadata.num_accepted_tokens[0])
            assert counts.tolist() == [expected, 1, 1, 1]
            assert counts.data_ptr() == address
            assert entry.static_metadata.linear_state_indices.tolist() == [0 if metadata.is_dummy else 3, 0, 0, 0]


def test_speculative_linear_graph_rejects_missing_acceptance_on_first_real_batch() -> None:
    runner = _linear_runner()
    input_ids = torch.ones(1, dtype=torch.int32)
    metadata = _decode_metadata(False)
    entry = runner._allocate_entry(4, input_ids, input_ids, metadata)
    with (
        patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True),
        pytest.raises(RuntimeError, match="accepted-token metadata is missing"),
    ):
        runner._fill_entry(entry, input_ids, input_ids, metadata, batch_size=1, input_embedding=None)


@pytest.mark.parametrize("num_decoding_tokens,linear", [(1, True), (4, False)])
def test_empty_non_recurrent_or_non_speculative_graph_does_not_require_acceptance(
    num_decoding_tokens: int, linear: bool
) -> None:
    runner = _linear_runner(num_decoding_tokens, linear)
    input_ids = torch.ones(1, dtype=torch.int32)
    metadata = _decode_metadata(True)
    entry = runner._allocate_entry(4, input_ids, input_ids, metadata)
    assert entry.static_metadata.num_accepted_tokens is None
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        for metadata in (_decode_metadata(True), _decode_metadata(False)):
            runner._fill_entry(entry, input_ids, input_ids, metadata, batch_size=1, input_embedding=None)


def test_dsa_graph_empty_rank_uses_reserved_block_zero() -> None:
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
    static_metadata = SimpleNamespace(
        multi_block_tables=(
            torch.full((4, 4), 99, dtype=torch.int32),
            torch.full((4, 2), 99, dtype=torch.int32),
            torch.full((4, 1), 99, dtype=torch.int32),
        )
    )
    metadata = SimpleNamespace(multi_block_tables=(), is_dummy=True)

    runner._fill_dsa_block_tables(
        static_metadata,
        metadata,
        torch.zeros((1, 1), dtype=torch.int32),
        batch_size=1,
    )

    for table in static_metadata.multi_block_tables:
        assert torch.all(table == 0)
