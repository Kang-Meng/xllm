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

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from xllm.python.attention.dsa_metadata import DsaMetadataBuilder, build_cache_specs
from xllm.python.attention.expanded_decode_metadata import ExpandedDecodeMetadata
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
        q_cu_seq_lens=torch.arange(5, dtype=torch.int32),
        linear_state_indices=linear_state_indices,
        num_accepted_tokens=torch.tensor([1, 2, 1, 2], dtype=torch.int32),
        expanded_decode_metadata=None,
        multi_block_tables=(),
        new_cache_slots_host_values=[0, 1, 2, 3],
        is_prefill=False,
        is_chunked_prefill=False,
    )


def test_eplv2_graph_admission_uses_padded_token_rows_for_dflash2() -> None:
    runner = _runner()
    runner.max_batch = 128
    runner.num_decoding_tokens = 8
    runner._eplv2_graph_token_limit = 16
    metadata = _metadata(torch.arange(3, dtype=torch.int32))
    with (
        patch.object(runner, "_has_compatible_decode_metadata", return_value=True),
        patch(
            "xllm.python.model_executor.runners.decode_acl_graph.resolve_expanded_decode_metadata",
            return_value=object(),
        ),
    ):
        # One and two 8-row sequences fit; three sequences would capture a
        # padded 32-row graph even though the active input has only 24 rows.
        for sequences, expected in ((1, True), (2, True), (3, False)):
            metadata.linear_state_indices = torch.arange(sequences)
            assert runner.can_execute(torch.arange(sequences * 8), metadata) is expected


def test_accepted_tokens_use_live_per_sequence_graph_buffer() -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.tensor([3, 3, 7, 7], dtype=torch.int32))
    metadata.num_accepted_tokens = torch.tensor([4, 2], dtype=torch.int32)
    entry = runner._allocate_entry(8, input_ids, positions, metadata)
    static_counts = entry.static_metadata.num_accepted_tokens
    address = static_counts.data_ptr()
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        runner._fill_entry(entry, input_ids, positions, metadata, batch_size=4, input_embedding=None)
        assert static_counts.tolist() == [4, 2, 1, 1, 1, 1, 1, 1]
        metadata.num_accepted_tokens.copy_(torch.tensor([1, 3]))
        runner._fill_entry(entry, input_ids, positions, metadata, batch_size=4, input_embedding=None)
        assert static_counts.tolist() == [1, 3, 1, 1, 1, 1, 1, 1]
        metadata.num_accepted_tokens = torch.tensor([2], dtype=torch.int32)
        runner._fill_entry(entry, input_ids, positions, metadata, batch_size=4, input_embedding=None)
        assert static_counts.tolist() == [2, 1, 1, 1, 1, 1, 1, 1]
        metadata.num_accepted_tokens = None
        metadata.is_dummy = True
        runner._fill_entry(entry, input_ids, positions, metadata, batch_size=4, input_embedding=None)
    assert static_counts.data_ptr() == address
    assert static_counts.tolist() == [1] * 8

    plain_metadata = _metadata(torch.tensor([3, 7, 11, 15], dtype=torch.int32))
    plain_metadata.num_accepted_tokens = None
    plain_entry = runner._allocate_entry(8, input_ids, positions, plain_metadata)
    assert plain_entry.static_metadata.num_accepted_tokens is None


def test_verify_graph_key_tracks_width_not_active_sequence_count() -> None:
    runner = _runner()
    captured_keys = []

    def capture_key(key: tuple) -> None:
        captured_keys.append(key)
        raise LookupError("graph key recorded")

    runner._graphs = SimpleNamespace(get=capture_key)
    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.resolve_expanded_decode_metadata",
        return_value=object(),
    ):
        for sequence_count, width in ((3, 4), (4, 4), (8, 2)):
            input_ids = torch.arange(sequence_count * width, dtype=torch.int32)
            metadata = SimpleNamespace(linear_state_indices=torch.arange(sequence_count))
            with pytest.raises(LookupError, match="graph key recorded"):
                runner.execute(input_ids, input_ids, metadata)
    assert captured_keys[0] == captured_keys[1]
    assert captured_keys[1] != captured_keys[2]


@pytest.mark.parametrize(
    "accepted_counts,error_type",
    [
        (None, RuntimeError),
        (torch.ones(2, 2, dtype=torch.int32), ValueError),
        (torch.ones(0, dtype=torch.int32), ValueError),
        (torch.ones(5, dtype=torch.int32), ValueError),
    ],
)
def test_accepted_tokens_reject_missing_or_invalid_replay_metadata(
    accepted_counts: torch.Tensor | None, error_type: type[Exception]
) -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.tensor([3, 3, 7, 7], dtype=torch.int32))
    metadata.num_accepted_tokens = torch.ones(2, dtype=torch.int32)
    entry = runner._allocate_entry(8, input_ids, positions, metadata)
    entry.static_metadata.is_spec_verify = True
    metadata.num_accepted_tokens = accepted_counts
    with (
        patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True),
        pytest.raises(error_type, match="accepted-token metadata"),
    ):
        runner._fill_entry(entry, input_ids, positions, metadata, batch_size=4, input_embedding=None)


def test_slice_output_preserves_aux_hidden_tuple() -> None:
    hidden = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    aux_hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4)

    output = DecodeAclGraphRunner._slice_output((hidden, aux_hidden), 3)

    assert isinstance(output, tuple)
    torch.testing.assert_close(output[0], hidden[:3])
    torch.testing.assert_close(output[1], aux_hidden[:3])


def test_linear_state_graph_buffers_are_stable() -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.tensor([3, 7, 11, 15], dtype=torch.int32))
    metadata.kpool_query_lens = [4]
    entry = runner._allocate_entry(
        padded_batch_size=8,
        input_ids=input_ids,
        positions=positions,
        metadata=metadata,
    )
    static_indices = entry.static_metadata.linear_state_indices
    static_num_accepted_tokens = entry.static_metadata.num_accepted_tokens
    data_ptr = static_indices.data_ptr()
    accepted_data_ptr = static_num_accepted_tokens.data_ptr()
    assert entry.static_metadata.kpool_query_lens == (4, 4)
    assert entry.static_metadata.kpool_query_lens_device.tolist() == [4, 4]
    assert DecodeAclGraphRunner._graph_key(
        8,
        True,
        None,
        entry.static_metadata.kpool_query_lens,
    ) != DecodeAclGraphRunner._graph_key(8, True, None, (2, 2, 2, 2))

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
        metadata.num_accepted_tokens = torch.tensor(
            [2, 1, 2, 1],
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
    assert static_num_accepted_tokens.data_ptr() == accepted_data_ptr
    assert static_num_accepted_tokens.tolist() == [2, 1, 2, 1, 1, 1, 1, 1]


def test_linear_state_snapshot_restores_all_ssm_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(synchronize=lambda: None),
        raising=False,
    )
    runner = _runner()
    conv = torch.arange(48, dtype=torch.float32).reshape(4, 2, 6)
    ssm = torch.arange(48, dtype=torch.float32).reshape(12, 1, 2, 2)
    runner.bind_layer_caches(
        [
            SimpleNamespace(
                conv=conv,
                ssm=ssm,
            )
        ]
    )
    entry = SimpleNamespace(
        static_metadata=SimpleNamespace(
            linear_state_indices=torch.tensor([1, 1, 3, 3], dtype=torch.int64),
        )
    )

    snapshot = runner._snapshot_linear_state(entry)
    assert torch.equal(snapshot[0][2], torch.tensor([1, 3], dtype=torch.int64))
    assert torch.equal(snapshot[0][3], torch.tensor([3, 4, 5, 9, 10, 11], dtype=torch.int64))
    expected_conv = conv.clone()
    expected_ssm = ssm.clone()
    conv.zero_()
    ssm.zero_()
    runner._restore_linear_state(entry, snapshot)

    assert torch.equal(conv[[1, 3]], expected_conv[[1, 3]])
    checkpoint_indices = torch.tensor([3, 4, 5, 9, 10, 11])
    assert torch.equal(ssm[checkpoint_indices], expected_ssm[checkpoint_indices])


def test_explicit_verify_reuses_row_aligned_paging_metadata() -> None:
    runner = _runner()
    metadata = _metadata(torch.tensor([1, 2], dtype=torch.int32))
    metadata.expanded_decode_metadata = ExpandedDecodeMetadata(
        kv_seq_lens=metadata.kv_seq_lens,
        block_table=metadata.block_table,
        paged_kv_indptr=metadata.paged_kv_indptr,
        paged_kv_indices=metadata.paged_kv_indices,
        paged_kv_last_page_len=metadata.paged_kv_last_page_len,
        paged_attention_tiling_data=None,
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=metadata.kv_seq_lens_host_values,
    )
    metadata.is_spec_verify = True
    metadata.is_chunked_prefill = True
    metadata.num_accepted_tokens = torch.tensor([4, 2], dtype=torch.int32)
    metadata.q_seq_lens = torch.tensor([2, 2], dtype=torch.int32)
    metadata.q_cu_seq_lens = torch.tensor([0, 2, 4], dtype=torch.int32)
    metadata.kv_seq_lens = torch.tensor([2, 4], dtype=torch.int32)
    metadata.kv_seq_lens_host_values = [2, 4]
    metadata.kv_cu_seq_lens = torch.tensor([0, 2, 6], dtype=torch.int32)
    metadata.block_table = metadata.block_table[1::2]

    with patch.object(
        runner,
        "_build_row_aligned_paged_kv_metadata",
        side_effect=AssertionError("row-aligned paging must come from C++"),
    ):
        resolved = runner._decode_metadata(metadata)

    assert resolved[3] is metadata.paged_kv_indptr
    assert resolved[4] is metadata.paged_kv_indices
    assert resolved[5] is metadata.paged_kv_last_page_len
    input_ids = torch.arange(4, dtype=torch.int32)
    assert runner.can_execute(input_ids, metadata)
    entry = runner._allocate_entry(8, input_ids, input_ids, metadata)
    assert entry.static_metadata.is_spec_verify
    assert entry.static_metadata.q_seq_lens.tolist() == [2, 2, 2, 2]
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        runner._fill_entry(entry, input_ids, input_ids, metadata, batch_size=4, input_embedding=None)
    assert entry.static_metadata.num_accepted_tokens.tolist() == [4, 2, 1, 1, 1, 1, 1, 1]
    assert entry.static_metadata.linear_state_indices.tolist() == [1, 1, 2, 2, 0, 0, 0, 0]


def test_verify_graph_requires_explicit_metadata() -> None:
    runner = _runner()
    metadata = _metadata(torch.tensor([1, 2], dtype=torch.int32))
    assert not runner.can_execute(torch.arange(4, dtype=torch.int32), metadata)


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
    assert all(torch.all(table == 0) for table in tables)

    block_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    indptr, indices, last_page_len = runner._build_row_aligned_paged_kv_metadata(
        block_table,
        [129, 132],
    )
    assert indptr.tolist() == [0, 2, 4]
    assert indices.tolist() == [10, 11, 20, 21]
    assert last_page_len.tolist() == [1, 4]

    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.kv_seq_lens_host_values = None
    metadata.paged_kv_indptr = torch.tensor([0, 1], dtype=torch.int32)
    metadata.paged_kv_last_page_len = torch.tensor([1], dtype=torch.int32)
    assert not runner.can_execute(torch.arange(4, dtype=torch.int32), metadata)


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
        [10, 11, 0, 0],
        [12, 13, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert static_metadata.multi_block_tables[1].tolist() == [
        [20, 0],
        [21, 0],
        [0, 0],
        [0, 0],
    ]
    assert static_metadata.multi_block_tables[2].tolist() == [[30], [31], [0], [0]]

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
        [40, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert static_metadata.multi_block_tables[1].tolist() == [
        [50, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]
    assert static_metadata.multi_block_tables[2].tolist() == [[60], [0], [0], [0]]


def test_dsa_graph_clamps_target_tables_to_draft_groups() -> None:
    _, group_infos = build_cache_specs([1], 128, 1)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=128, is_mla=False, group_infos=group_infos),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=512,
    )
    static_metadata = SimpleNamespace(multi_block_tables=(torch.full((4, 2), 99, dtype=torch.int32),))
    metadata = SimpleNamespace(
        multi_block_tables=(
            torch.tensor([[10], [11]], dtype=torch.int32),
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
        [10, 0],
        [11, 0],
        [0, 0],
        [0, 0],
    ]


def test_dsa_graph_padding_uses_reserved_single_kv_length() -> None:
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

    assert entry.static_metadata.kv_seq_lens_host_values == [9, 17, 1, 1]


def test_dsa_graph_pads_new_cache_slots_to_graph_bucket() -> None:
    entry = SimpleNamespace(
        batch_size=16,
        static_metadata=SimpleNamespace(new_cache_slots_host_values=[]),
    )
    metadata = SimpleNamespace(new_cache_slots_host_values=[301, 302, 401, 402])

    DecodeAclGraphRunner._fill_new_cache_slots_host_metadata(
        entry,
        metadata,
        batch_size=4,
    )

    assert entry.static_metadata.new_cache_slots_host_values == [301, 302, 401, 402] + [0] * 12


def test_dsa_graph_padding_preserves_scheduler_resolved_swa_slots() -> None:
    entry = SimpleNamespace(
        batch_size=4,
        static_metadata=SimpleNamespace(new_cache_slots_host_values=[]),
    )
    metadata = SimpleNamespace(
        new_cache_slots_host_values=[1280, 2560, 3840],
    )
    DecodeAclGraphRunner._fill_new_cache_slots_host_metadata(
        entry,
        metadata,
        batch_size=3,
    )

    caches_info, group_infos = build_cache_specs([0, 4, 128, 4], 128, 4)
    builder = DsaMetadataBuilder(caches_info, group_infos)
    dsa = builder.build(
        multi_block_tables=[
            torch.tensor(
                [
                    [10, 11, 0, 0],
                    [20, 21, 0, 0],
                    [30, 31, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=torch.int32,
            ),
            torch.zeros((4, 4), dtype=torch.int32),
            torch.zeros((4, 4), dtype=torch.int32),
        ],
        kv_seq_lens=[257, 257, 257, 1],
        q_seq_lens=[1, 1, 1, 1],
        positions=torch.tensor([256, 256, 256, 0], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
        new_cache_slots=entry.static_metadata.new_cache_slots_host_values,
        enable_graph=True,
        graph_block_table_capacity_cols=4,
    )

    assert dsa.slot_mappings[0][0].tolist() == [1280, 2560, 3840, 0]


def test_dsa_graph_allows_dummy_without_scheduler_cache_slots() -> None:
    entry = SimpleNamespace(
        batch_size=4,
        static_metadata=SimpleNamespace(new_cache_slots_host_values=[99]),
    )
    metadata = SimpleNamespace(new_cache_slots_host_values=[], is_dummy=True)

    DecodeAclGraphRunner._fill_new_cache_slots_host_metadata(
        entry,
        metadata,
        batch_size=1,
    )

    assert entry.static_metadata.new_cache_slots_host_values == []


def test_dsv4_decode_admits_scheduler_slots_when_top_level_mapping_is_empty() -> None:
    runner = _runner()
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.slot_mapping = torch.empty(0, dtype=torch.int32)
    metadata.block_table = None
    metadata.multi_block_tables = (torch.tensor([[10, 0], [20, 0], [30, 0], [40, 0]], dtype=torch.int32),)
    metadata.new_cache_slots_host_values = [301, 302, 401, 402]

    assert runner.can_execute(torch.arange(4, dtype=torch.int32), metadata)
    assert runner._effective_slot_mapping(metadata, 4, torch.device("cpu")).tolist() == [
        301,
        302,
        401,
        402,
    ]


def test_decode_rejects_mismatched_linear_state_validity_mask() -> None:
    runner = _runner()
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.has_initial_state = torch.ones(3, dtype=torch.int32)

    assert not runner.can_execute(torch.arange(4, dtype=torch.int32), metadata)


def test_decode_rejects_mismatched_linear_state_indices_without_cache_probe() -> None:
    runner = _runner()
    metadata = _metadata(torch.arange(3, dtype=torch.int32))

    assert not runner.can_execute(torch.arange(4, dtype=torch.int32), metadata)


def test_fill_entry_uses_scheduler_slots_for_dsv4_decode() -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.slot_mapping = torch.empty(0, dtype=torch.int32)
    metadata.new_cache_slots_host_values = [301, 302, 401, 402]
    entry = runner._allocate_entry(8, input_ids, positions, metadata)

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ) as update_metadata:
        runner._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size=4,
            input_embedding=None,
        )

    assert update_metadata.call_args.args[2].tolist() == [301, 302, 401, 402]
    assert entry.static_metadata.new_cache_slots_host_values == [301, 302, 401, 402, 0, 0, 0, 0]


def test_dp_real_and_dummy_decode_share_graph_admission() -> None:
    def make_metadata(*, is_dummy: bool) -> SimpleNamespace:
        metadata = _metadata(torch.tensor([0], dtype=torch.int32))
        metadata.slot_mapping = torch.tensor([0], dtype=torch.int32) if is_dummy else torch.empty(0, dtype=torch.int32)
        metadata.block_table = None
        metadata.multi_block_tables = (torch.tensor([[0]], dtype=torch.int32),)
        metadata.kv_seq_lens = torch.tensor([1], dtype=torch.int32)
        metadata.kv_seq_lens_host_values = [1]
        metadata.kv_cu_seq_lens = torch.tensor([0, 1], dtype=torch.int32)
        metadata.q_cu_seq_lens = torch.tensor([0, 1], dtype=torch.int32)
        metadata.paged_kv_indptr = torch.tensor([0, 1], dtype=torch.int32)
        metadata.paged_kv_indices = torch.tensor([0], dtype=torch.int32)
        metadata.paged_kv_last_page_len = torch.tensor([1], dtype=torch.int32)
        metadata.new_cache_slots_host_values = [] if is_dummy else [301]
        metadata.dp_execution_token_counts = (1, 1)
        metadata.dp_is_decode = (1, 1)
        metadata.is_dummy = is_dummy
        return metadata

    real_runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
    )
    dummy_runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=8,
        dp_size=2,
        dp_rank=1,
    )
    input_ids = torch.tensor([1], dtype=torch.int32)

    assert real_runner.can_execute(input_ids, make_metadata(is_dummy=False))
    assert dummy_runner.can_execute(input_ids, make_metadata(is_dummy=True))


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


def test_dp_mtp_target_active_and_empty_ranks_use_eager() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    active_runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
        num_decoding_tokens=4,
    )
    empty_runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=1,
        num_decoding_tokens=4,
    )
    draft_runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
        num_decoding_tokens=1,
    )
    active_metadata = _dp_metadata((4, 0))
    active_metadata.is_dummy = False
    empty_metadata = _dp_metadata((4, 0))
    empty_metadata.is_dummy = True

    with (
        patch.object(
            active_runner,
            "_has_compatible_decode_metadata",
            return_value=True,
        ),
        patch.object(
            empty_runner,
            "_has_compatible_decode_metadata",
            return_value=True,
        ),
        patch.object(
            draft_runner,
            "_has_compatible_decode_metadata",
            return_value=True,
        ),
    ):
        assert not active_runner.can_execute(
            torch.zeros(4, dtype=torch.int32),
            active_metadata,
        )
        assert not empty_runner.can_execute(
            torch.zeros(1, dtype=torch.int32),
            empty_metadata,
        )
        assert draft_runner.can_execute(
            torch.zeros(4, dtype=torch.int32),
            active_metadata,
        )


@pytest.mark.parametrize(
    ("width", "disable_verify_graph"),
    [
        pytest.param(4, False, id="kda-verify-default"),
        pytest.param(4, True, id="verify-graph-disabled"),
        pytest.param(3, False, id="unsupported-verify-width"),
    ],
)
def test_dp_mtp_target_fallback_is_group_wide(
    width: int,
    disable_verify_graph: bool,
) -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    active_runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
        num_decoding_tokens=width,
    )
    empty_runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=1,
        num_decoding_tokens=width,
    )
    active_metadata = _dp_metadata((width, 0))
    active_metadata.is_dummy = False
    active_metadata.kv_seq_lens = torch.ones(width, dtype=torch.int32)
    active_metadata.block_table = torch.zeros((width, 1), dtype=torch.int32)
    active_metadata.linear_state_indices = torch.tensor([0], dtype=torch.int32)
    active_metadata.q_cu_seq_lens = None
    empty_metadata = _dp_metadata((width, 0))
    empty_metadata.is_dummy = True

    with (
        patch.dict(
            os.environ,
            {"XLLM_NO_VERIFY_GRAPH": "1" if disable_verify_graph else "0"},
        ),
        patch.object(
            active_runner,
            "_has_compatible_decode_metadata",
            return_value=True,
        ),
        patch.object(
            empty_runner,
            "_has_compatible_decode_metadata",
            return_value=True,
        ),
    ):
        assert not active_runner.can_execute(
            torch.zeros(width, dtype=torch.int32),
            active_metadata,
        )
        assert not empty_runner.can_execute(
            torch.zeros(1, dtype=torch.int32),
            empty_metadata,
        )


def test_kda_verify_graph_is_enabled_by_default() -> None:
    runner = _runner()
    metadata = _metadata(torch.tensor([1, 3], dtype=torch.int32))
    metadata.q_cu_seq_lens = torch.arange(5, dtype=torch.int32)
    metadata.kv_seq_lens = torch.tensor([5, 6, 9, 10], dtype=torch.int32)
    with (
        patch.dict(os.environ, {"XLLM_NO_VERIFY_GRAPH": "0"}),
        patch.object(runner, "_has_compatible_decode_metadata", return_value=True),
    ):
        assert runner.can_execute(torch.zeros(4, dtype=torch.int32), metadata)


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
