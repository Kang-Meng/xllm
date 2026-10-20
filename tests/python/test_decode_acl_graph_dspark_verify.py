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

"""Non-expanded (V4.1 DSpark) packed spec-verify graph admission.

A DSpark verify step packs ``num_spec+1`` token rows per logical sequence and
arrives decode-typed: V4.1 has no linear-attention layers, so it exports no
``linear_state_indices`` and ``is_spec_verify`` stays False. The C++ validate
builder exports the true packed width on ``AttentionMetadata.spec_verify_width``
so the runner can fold token rows back to sequences and admit the batch under
DP>1 without confusing it with a plain decode of the same row count.

The packed-verify admission is unconditional: whenever the exported
``spec_verify_width`` reports a packed width, the runner folds token rows
back to sequences.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner
from xllm.python.model_executor.runners.decode_cuda_graph import _decode_bucket

def _runner(
    *,
    dp_size: int = 1,
    dp_rank: int = 0,
    max_seqs_per_batch: int = 16,
    num_decoding_tokens: int = 6,
    limit: int | None = None,
) -> DecodeAclGraphRunner:
    return DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=128, is_mla=False),
        torch.device("cpu"),
        max_batch=max_seqs_per_batch,
        max_model_len=65536,
        dp_size=dp_size,
        dp_rank=dp_rank,
        decode_batch_size_limit=limit,
        num_decoding_tokens=num_decoding_tokens,
    )


def _metadata(
    rows: int,
    *,
    spec_verify_width: int | None = 6,
    counts: tuple[int, ...] | None = None,
    dp_size: int = 1,
    per_sequence_tensors: bool = False,
) -> SimpleNamespace:
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        is_dummy=False,
        slot_mapping=torch.arange(rows, dtype=torch.int32),
        new_cache_slots_host_values=[],
        paged_kv_indptr=torch.arange(rows + 1, dtype=torch.int32),
        paged_kv_indices=torch.arange(rows, dtype=torch.int32),
        paged_kv_last_page_len=torch.ones(rows, dtype=torch.int32),
        block_table=torch.zeros((rows, 1), dtype=torch.int32),
        kv_seq_lens=torch.ones(rows, dtype=torch.int32),
        kv_seq_lens_host_values=[1] * rows,
        max_seq_len=1,
        q_cu_seq_lens=torch.arange(rows + 1, dtype=torch.int32),
        kv_cu_seq_lens=None,
        linear_state_indices=None,
        has_initial_state=None,
        expanded_decode_metadata=None,
        multi_block_tables=(),
        kpool_query_lens=(),
        dp_execution_token_counts=counts if counts is not None else (rows,) * dp_size,
        dp_is_decode=(True,) * dp_size,
    )
    if per_sequence_tensors:
        # A packed verify carries per-LOGICAL-SEQUENCE state tensors even though
        # its paging metadata is per token row (the on-hardware V4.1 shape).
        seqs = rows // max(spec_verify_width or 1, 1)
        metadata.linear_state_indices = torch.arange(seqs, dtype=torch.int64)
        metadata.has_initial_state = torch.ones(seqs, dtype=torch.int32)
        metadata.num_accepted_tokens = torch.ones(seqs, dtype=torch.int32)
    if spec_verify_width is not None:
        metadata.spec_verify_width = spec_verify_width
    return metadata


# --------------------------------------------------------------------------
# Exported width folds rows to sequences symmetrically across DP
# --------------------------------------------------------------------------


def test_active_and_idle_ranks_fold_to_the_same_graph_bucket() -> None:
    counts = (36, 1, 1, 1)
    decisions = []
    for rank in range(4):
        rows = counts[rank]
        runner = _runner(dp_size=4, dp_rank=rank, max_seqs_per_batch=64)
        assert runner.max_batch == 64  # merged-tree scheduler publishes the full seq budget (no /dp_size)
        metadata = _metadata(rows, spec_verify_width=6, counts=counts, dp_size=4)
        row_tokens = torch.zeros(rows, dtype=torch.int32)
        decisions.append(runner.can_execute(row_tokens, metadata))
    assert decisions == [True, True, True, True]


def test_per_sequence_state_tensors_admit_a_packed_verify() -> None:
    # The on-hardware V4.1 verify batch carries one linear_state_indices /
    # has_initial_state / num_accepted_tokens entry per LOGICAL sequence while
    # its paging metadata is per token row; admission must fold those too.
    counts = (96, 96, 96, 96)
    runner = _runner(dp_size=4, dp_rank=0, max_seqs_per_batch=64)
    metadata = _metadata(96, spec_verify_width=6, counts=counts, dp_size=4, per_sequence_tensors=True)
    assert metadata.linear_state_indices.numel() == 16
    tokens = torch.zeros(96, dtype=torch.int32)
    assert runner.can_execute(tokens, metadata) is True


def test_per_sequence_state_tensors_do_not_admit_a_plain_decode() -> None:
    # Without an exported packed width the per-sequence state tensors are not a
    # verify signal: the historical width-1 decision (eager under DP>1) holds.
    runner = _runner(dp_size=4, dp_rank=0, max_seqs_per_batch=64)
    metadata = _metadata(96, spec_verify_width=1, counts=(96, 96, 96, 96), dp_size=4, per_sequence_tensors=False)
    tokens = torch.zeros(96, dtype=torch.int32)
    assert runner.can_execute(tokens, metadata) is False


def test_width_folds_to_sequence_count() -> None:
    runner = _runner(num_decoding_tokens=6)
    metadata = _metadata(36, spec_verify_width=6)
    assert runner._spec_verify_width(36, metadata) == 6
    assert runner._spec_verify_sequence_count(metadata, 36, False, 6) == 6
    # Idle shard fake row (< width) still stands for one logical sequence.
    idle = _metadata(1, spec_verify_width=6)
    assert runner._spec_verify_sequence_count(idle, 1, False, 6) == 1


def test_explicit_width_wins_over_shape_guess() -> None:
    runner = _runner(num_decoding_tokens=6)
    # A width-3 pack (num_spec=2) on a width-6 runner: the exported truth
    # source wins.
    width_meta = _metadata(48, spec_verify_width=3)
    assert runner._spec_verify_width(48, width_meta) == 3


def test_missing_field_keeps_the_conservative_width_one() -> None:
    # A binary without the exported field keeps the historical width-1 decode
    # path: a legacy batch is never shape-guessed into a packed verify, so the
    # group-wide plain-decode admission (and the eager fallback for an
    # unexported pack) holds.
    runner = _runner(max_seqs_per_batch=16, num_decoding_tokens=6)
    metadata = _metadata(36, spec_verify_width=None)
    assert not hasattr(metadata, "spec_verify_width")
    assert runner._spec_verify_width(36, metadata) == 1


# --------------------------------------------------------------------------
# Plain decode (exported width 1) must never change its decision
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 12, 16])
def test_plain_decode_inside_capacity_is_unchanged(rows: int) -> None:
    runner = _runner(max_seqs_per_batch=16, num_decoding_tokens=6)
    expected = _decode_bucket(rows) <= runner.max_batch
    metadata = _metadata(rows, spec_verify_width=1)
    tokens = torch.zeros(rows, dtype=torch.int32)
    assert runner.can_execute(tokens, metadata) is expected


def test_width_one_keeps_the_gate_blocked() -> None:
    # A decode-typed batch the C++ marks as width 1 is not a verifiable pack:
    # the historical DP+MTP gate still rejects it.
    runner = _runner(dp_size=4, dp_rank=0, max_seqs_per_batch=16, num_decoding_tokens=6)
    metadata = _metadata(4, spec_verify_width=1, counts=(4, 4, 4, 4), dp_size=4)
    tokens = torch.zeros(4, dtype=torch.int32)
    assert runner.can_execute(tokens, metadata) is False


# --------------------------------------------------------------------------
# The graph key partitions the width
# --------------------------------------------------------------------------


def test_graph_key_separates_plain_decode_from_verify() -> None:
    key_plain = DecodeAclGraphRunner._graph_key(48, False, None, (), verify_width=1, token_capacity_bucket=0)
    key_verify = DecodeAclGraphRunner._graph_key(48, False, None, (), verify_width=6, token_capacity_bucket=0)
    assert key_plain != key_verify
