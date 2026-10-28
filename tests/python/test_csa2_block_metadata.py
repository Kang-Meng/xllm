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

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.csa2_block_metadata import Csa2BlockAttentionMetadataBuilder
from xllm.python.model_executor.input_batch import InputBatch


def _batch(prefixes: tuple[int, ...], width: int = 3, capacity: int | None = None) -> InputBatch:
    num_tokens = len(prefixes) * width if prefixes else width
    positions = torch.tensor([prefix + offset for prefix in prefixes for offset in range(width)], dtype=torch.int32)
    if not prefixes:
        positions = torch.arange(width, dtype=torch.int32)
    result = InputBatch.from_runtime(
        torch.arange(num_tokens),
        positions,
        SimpleNamespace(
            num_reqs=len(prefixes),
            num_tokens=num_tokens,
            num_scheduled_tokens=[width] * len(prefixes),
            num_computed_tokens=[],
            query_start_loc=list(range(0, len(prefixes) * width + 1, width)),
            is_prefilling=[False] * len(prefixes),
        ),
        is_dummy=not prefixes,
    )
    if capacity is None:
        return result
    padded_positions = torch.zeros(capacity, dtype=torch.int32)
    padded_positions[:num_tokens].copy_(positions)
    return result.bind_graph_inputs(
        torch.arange(capacity),
        padded_positions,
        torch.arange(capacity) >= num_tokens,
    )


def _metadata(tables: list[list[int]], width: int = 3, expanded: bool = True) -> SimpleNamespace:
    table = torch.tensor(tables, dtype=torch.int32)
    if expanded:
        table = table.repeat_interleave(width, dim=0)
    return SimpleNamespace(multi_block_tables=(table, object(), object()))


@pytest.mark.parametrize("expanded", [False, True])
def test_builder_uses_request_semantics_and_only_the_swa_group(expanded: bool) -> None:
    builder = Csa2BlockAttentionMetadataBuilder(3, 8, 64)
    batch = _batch((7, 12))
    result = builder.build(batch, _metadata([[4], [7]], expanded=expanded))
    assert result.query_lens == (3, 3)
    assert result.context_lengths.tolist() == [10, 15]
    assert result.block_table.tolist() == [[4], [7]]
    assert result.slot_mapping.tolist() == [39, 32, 33, 60, 61, 62]
    assert batch.num_reqs == 2 and batch.num_computed_tokens == ()


def test_persistent_buffers_keep_addresses_and_clear_previous_requests() -> None:
    builder = Csa2BlockAttentionMetadataBuilder(3, 8, 64)
    first = _batch((7, 12), capacity=9)
    result = builder.allocate_persistent(first, SimpleNamespace())
    tensors = (result.context_lengths, result.block_table, result.slot_mapping)
    addresses = tuple(tensor.data_ptr() for tensor in tensors)
    assert result.context_lengths.tolist() == [0, 0, 0]
    assert result.slot_mapping.tolist() == [-1] * 9
    builder.update_persistent(result, first, _metadata([[4], [7]]))
    assert result.context_lengths.tolist() == [10, 15, 0]
    assert result.block_table[2].tolist() == [0] * 8
    second = _batch((20,), capacity=9)
    builder.update_persistent(result, second, _metadata([[5]]))
    current_tensors = (result.context_lengths, result.block_table, result.slot_mapping)
    assert tuple(tensor.data_ptr() for tensor in current_tensors) == addresses
    assert result.context_lengths.tolist() == [23, 0, 0]
    assert result.block_table[0].tolist() == [5] * 8
    assert result.block_table[1:].count_nonzero() == 0
    assert result.slot_mapping.tolist() == [44, 45, 46] + [-1] * 6


def test_dummy_update_never_reads_live_request_tables() -> None:
    builder = Csa2BlockAttentionMetadataBuilder(3, 8, 64)
    batch = _batch((7,), capacity=6)
    result = builder.allocate_persistent(batch, SimpleNamespace())
    builder.update_persistent(result, batch, _metadata([[4]]))
    builder.update_persistent(result, replace(batch, is_dummy=True), SimpleNamespace())
    assert result.context_lengths.tolist() == [0, 0]
    assert result.block_table.count_nonzero() == 0
    assert result.slot_mapping.tolist() == [-1] * 6
    empty = builder.build(_batch(()), SimpleNamespace())
    assert empty.query_lens == (3,)
    assert empty.context_lengths.tolist() == [0]
    assert empty.slot_mapping.tolist() == [-1] * 3


def test_persistent_table_preserves_the_source_ring_period() -> None:
    builder = Csa2BlockAttentionMetadataBuilder(3, 8, 64)
    batch = _batch((15,), capacity=6)
    result = builder.allocate_persistent(batch, SimpleNamespace())
    builder.update_persistent(result, batch, _metadata([[2, 5]]))
    assert result.block_table[0].tolist() == [2, 5] * 4
    assert result.slot_mapping.tolist() == [47, 16, 17, -1, -1, -1]
    builder.update_persistent(result, batch, _metadata([[3, 4, 5]]))
    assert result.block_table[0].tolist() == [3, 4, 5, 3, 4, 5, 3, 4]
    assert result.slot_mapping.tolist() == [39, 40, 41, -1, -1, -1]


def test_query_wrapping_the_ring_commits_only_the_last_write() -> None:
    builder = Csa2BlockAttentionMetadataBuilder(11, 8, 64)
    result = builder.build(_batch((7,), width=11), _metadata([[4]], width=11))
    assert result.slot_mapping[:3].tolist() == [-1, -1, -1]
    assert result.slot_mapping[3:].tolist() == [34, 35, 36, 37, 38, 39, 32, 33]


def test_builder_uses_the_bound_swa_page_size() -> None:
    builder = Csa2BlockAttentionMetadataBuilder(3, 8, 64)
    builder.bind_layer_caches([LayerCache(key=None, value=None, swa=torch.empty(8, 4, 1, 2))])
    result = builder.build(_batch((5,)), _metadata([[3, 6]]))
    assert result.slot_mapping.tolist() == [25, 26, 27]


def test_builder_does_not_materialize_device_lengths_on_the_host(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = Csa2BlockAttentionMetadataBuilder(3, 8, 64)
    batch, metadata = _batch((7,)), _metadata([[4]])

    def reject_host_read(*args: object, **kwargs: object) -> None:
        pytest.fail("block attention metadata must not copy device values to the host")

    with monkeypatch.context() as scope:
        for name in ("cpu", "item", "tolist"):
            scope.setattr(torch.Tensor, name, reject_host_read)
        result = builder.build(batch, metadata)
    assert result.context_lengths.tolist() == [10]


@pytest.mark.parametrize("tables", [[], [[4], [5]]])
def test_builder_rejects_missing_or_inconsistent_request_tables(tables: list[list[int]]) -> None:
    builder = Csa2BlockAttentionMetadataBuilder(3, 8, 64)
    metadata = _metadata(tables, expanded=False) if tables else SimpleNamespace(multi_block_tables=())
    with pytest.raises(ValueError, match="SWA block table|request count"):
        builder.build(_batch((7,)), metadata)


def test_builder_rejects_partial_query_capacity() -> None:
    builder = Csa2BlockAttentionMetadataBuilder(3, 8, 64)
    with pytest.raises(ValueError, match="complete query blocks"):
        builder.allocate_persistent(_batch((7,), capacity=4), SimpleNamespace())


def test_builder_does_not_infer_request_count_from_table_shape() -> None:
    builder = Csa2BlockAttentionMetadataBuilder(3, 8, 64)
    batch = replace(_batch((7,)), num_reqs=2)
    with pytest.raises(ValueError, match="per request"):
        builder.build(batch, _metadata([[4]]))


def test_block_attention_rejects_compressed_cache_groups_at_initialization() -> None:
    from xllm.python.attention.csa2_attention import Csa2AttentionBackend

    with pytest.raises(ValueError, match="single SWA cache group"):
        Csa2AttentionBackend(
            compress_ratios=[0, 2],
            kv_source_layer_ids=[1],
            index_source_layer_ids=[1],
            window_size=8,
            n_layers=2,
            num_heads=1,
            attn_head_dim=64,
            index_topk=8,
            index_n_heads=1,
            index_head_dim=64,
            rope_head_dim=16,
            candidate_source_layer_id=-1,
            candidate_topk_blocks=0,
            candidate_block_size=0,
            device=torch.device("cpu"),
            dtype=torch.float32,
            block_query_width=3,
        )


@pytest.mark.parametrize("query_width", [0, 3])
def test_block_attention_declares_its_fixed_window_capacity(query_width: int) -> None:
    from xllm.python.attention.csa2_attention import Csa2AttentionBackend

    backend = Csa2AttentionBackend(
        compress_ratios=[0, 0],
        kv_source_layer_ids=[],
        index_source_layer_ids=[],
        window_size=8,
        n_layers=2,
        num_heads=1,
        attn_head_dim=64,
        index_topk=8,
        index_n_heads=1,
        index_head_dim=64,
        rope_head_dim=16,
        candidate_source_layer_id=-1,
        candidate_topk_blocks=0,
        candidate_block_size=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_model_len=64,
        block_query_width=query_width,
        quant_mode="ascend",
        acl_graph_enabled=True,
    )
    assert (backend.graph_token_capacity_granularity == 0) == (query_width > 0)


def test_block_attention_builder_rejects_context_parallelism_at_initialization() -> None:
    with pytest.raises(ValueError, match="context parallelism"):
        Csa2BlockAttentionMetadataBuilder.from_config(
            {
                "draft_query_width": 3,
                "window_size": 8,
                "max_position_embeddings": 64,
                "cp_size": 2,
            }
        )
