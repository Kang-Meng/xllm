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

"""Qwen3.5 DSpark target DP/ACL graph integration test, invoked by the C++ driver."""

from __future__ import annotations

import os
from datetime import timedelta
from types import SimpleNamespace

import torch

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.expanded_decode_metadata import ExpandedDecodeMetadata
from xllm.python.model_executor.executor import ModelExecutor
from xllm.python.models.qwen3_5 import Qwen3_5ForCausalLM

_DEVICE = torch.device(f"npu:{os.environ.get('LOCAL_RANK', '0')}")


def _random(shape: tuple[int, ...], generator: torch.Generator, dtype: torch.dtype) -> torch.Tensor:
    return (torch.randn(shape, generator=generator) * 0.1).to(device=_DEVICE, dtype=dtype)


def _target_metadata(
    slots: list[int],
    accepted: list[int],
    width: int,
    step: int,
    *,
    prefix: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, SimpleNamespace, SimpleNamespace]:
    batch = len(slots)
    tokens = batch * width
    prefix = 2 + step * width if prefix is None else prefix
    logical_slots = torch.tensor(slots, dtype=torch.int32, device=_DEVICE)
    lengths = [prefix + offset + 1 for _ in slots for offset in range(width)]
    block_table = logical_slots[:, None]
    row_blocks = block_table.repeat_interleave(width, dim=0)
    slot_values = [slot * 128 + prefix + offset for slot in slots for offset in range(width)]
    positions = torch.tensor(list(range(prefix, prefix + width)) * batch, dtype=torch.int32, device=_DEVICE)
    input_ids = torch.tensor(
        [(slot + step + offset) % 128 for slot in slots for offset in range(width)], dtype=torch.int32, device=_DEVICE
    )
    expanded = ExpandedDecodeMetadata(
        kv_seq_lens=torch.tensor(lengths, dtype=torch.int32, device=_DEVICE),
        block_table=row_blocks,
        paged_kv_indptr=torch.arange(tokens + 1, dtype=torch.int32, device=_DEVICE),
        paged_kv_indices=row_blocks.flatten(),
        paged_kv_last_page_len=torch.tensor(lengths, dtype=torch.int32, device=_DEVICE),
        paged_attention_tiling_data=None,
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=lengths,
    )
    boundaries = list(range(0, tokens + 1, width))
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor(slot_values, dtype=torch.int32, device=_DEVICE),
        paged_kv_indptr=torch.arange(batch + 1, dtype=torch.int32, device=_DEVICE),
        paged_kv_indices=logical_slots,
        paged_kv_last_page_len=torch.full((batch,), prefix + width, dtype=torch.int32, device=_DEVICE),
        paged_kv_indptr_host=None,
        paged_kv_last_page_len_host=None,
        block_table=block_table,
        kv_seq_lens=torch.full((batch,), prefix + width, dtype=torch.int32, device=_DEVICE),
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=[prefix + width] * batch,
        kv_cu_seq_lens=torch.arange(batch + 1, dtype=torch.int32, device=_DEVICE) * (prefix + width),
        qo_indptr=None,
        q_cu_seq_lens=torch.tensor(boundaries, dtype=torch.int32, device=_DEVICE),
        q_cu_seq_lens_host_values=boundaries,
        q_seq_lens=torch.full((batch,), width, dtype=torch.int32, device=_DEVICE),
        q_seq_lens_host=torch.full((batch,), width, dtype=torch.int32),
        linear_state_indices=logical_slots,
        linear_state_read_indices=logical_slots,
        linear_state_write_indices=logical_slots,
        has_initial_state=torch.ones(batch, dtype=torch.bool, device=_DEVICE),
        num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32, device=_DEVICE),
        kpool_query_lens=(),
        expanded_decode_metadata=expanded,
        multi_block_tables=(),
        new_cache_slots_host_values=slot_values,
        is_prefill=False,
        is_chunked_prefill=True,
        is_spec_verify=True,
        is_mixed=False,
        is_dummy=False,
        dp_execution_token_counts=(),
        raw_dp_execution_token_counts=(),
        dp_global_kv_max_seq_lens=(),
        dp_is_decode=(),
        max_query_len=width,
        max_seq_len=prefix + width,
        dsa_metadata=None,
        dsa_positions=None,
        local_slot_mapping=None,
        kv_split_size=1,
        kv_split_rank=0,
        has_kv_shard=False,
    )
    batch_metadata = SimpleNamespace(
        num_reqs=batch,
        num_tokens=tokens,
        num_scheduled_tokens=[width] * batch,
        num_computed_tokens=[prefix] * batch,
        query_start_loc=boundaries,
        is_prefilling=[False] * batch,
    )
    return input_ids, positions, metadata, batch_metadata


@torch.inference_mode()
def check_dspark_target_aclgraph() -> None:
    """Run DP1 normally; torchrun --no-python selects synchronized DP2 ranks."""
    torch.npu.set_device(_DEVICE)
    if int(os.environ.get("WORLD_SIZE", "1")) == 2:
        torch.distributed.init_process_group("gloo", timeout=timedelta(seconds=120))
        try:
            for width in range(2, 18):
                _check_dspark_target_aclgraph_width(width, torch.distributed.get_rank())
        finally:
            torch.distributed.destroy_process_group()
    else:
        for width in range(2, 18):
            _check_dspark_target_aclgraph_width(width)


def _live_cache_states(caches: list[LayerCache], width: int) -> tuple[torch.Tensor, ...]:
    return caches[0].conv[1:], caches[0].ssm[width:], caches[1].key[1:], caches[1].value[1:]


def _check_dspark_target_aclgraph_width(width: int, dp_rank: int | None = None) -> None:
    generator = torch.Generator().manual_seed(238)
    config = {
        "model_type": "qwen3_5",
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "intermediate_size": 256,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "vocab_size": 128,
        "max_position_embeddings": 128,
        "layers_to_capture": [0, 1],
        "device": str(_DEVICE),
        "dtype": "bfloat16",
        "dp_size": 1 if dp_rank is None else 2,
        "dp_rank": dp_rank or 0,
    }
    model = Qwen3_5ForCausalLM(config)
    for name, parameter in model.named_parameters():
        if name.endswith("norm_weight"):
            parameter.fill_(1)
        else:
            parameter.copy_(_random(tuple(parameter.shape), generator, parameter.dtype))
    conv = _random((10, width + 2, 512), generator, torch.bfloat16)
    ssm = _random((10 * width, 2, 128, 128), generator, torch.float32)
    conv[0].fill_(float("nan"))
    ssm[:width].fill_(float("nan"))
    key = _random((10, 128, 1, 128), generator, torch.bfloat16)
    value = _random((10, 128, 1, 128), generator, torch.bfloat16)
    caches = [LayerCache(None, None, conv=conv, ssm=ssm), LayerCache(key, value)]
    eager_caches = [
        LayerCache(None, None, conv=conv.clone(), ssm=ssm.clone()),
        LayerCache(key.clone(), value.clone()),
    ]
    graph = ModelExecutor(model, {**config, "python_graph_backend": "aclgraph"}, 32, width)
    eager = ModelExecutor(model, {**config, "python_graph_backend": "off"}, 32, width)
    graph.bind_kv_caches(caches)
    eager.bind_kv_caches(eager_caches)
    assert graph.decode_graph_runner is not None
    captured_outputs = {}
    steps = (
        ([1], [width]),
        ([1, 2, 3, 4, 5], [1, 2, width, 1, 2]),
        ([2, 3, 4, 5, 6, 7, 8], [2, width, 1, 2, width, 1, 2]),
        ([5, 4, 3, 2, 1], [width, 1, 2, width, 1]),
        ([1], [width]),
    )
    dp_request_counts = ((0, 5), (3, 2), (5, 0), (1, 0), (0, 7), (5, 0))
    if dp_rank is not None:
        steps = tuple(
            (list(range(1, counts[dp_rank] + 1)), [1 + step % width] * counts[dp_rank])
            for step, counts in enumerate(dp_request_counts)
        )
    for step, (slots, accepted) in enumerate(steps):
        is_prefill = dp_rank is not None and step == 3
        if is_prefill and slots:
            ids, positions, metadata, batch = _target_metadata([9], [1], 2, step, prefix=0)
        elif slots:
            ids, positions, metadata, batch = _target_metadata(slots, accepted, width, step)
        else:
            ids, positions, metadata, batch = _target_metadata([0], [1], 1, 0, prefix=0)
            ids.fill_(1)
            metadata.is_dummy = True
            metadata.is_chunked_prefill = False
            metadata.expanded_decode_metadata = None
            batch.num_reqs = 0
            batch.num_scheduled_tokens = []
            batch.num_computed_tokens = []
            batch.query_start_loc = [0]
            batch.is_prefilling = []
        if is_prefill:
            metadata.is_spec_verify = False
            metadata.is_prefill = bool(slots)
            metadata.is_chunked_prefill = bool(slots)
            metadata.expanded_decode_metadata = None
            metadata.has_initial_state.zero_()
            batch.is_prefilling = [True] * len(slots)
        if dp_rank is not None:
            raw_counts = tuple(count * (2 if is_prefill else width) for count in dp_request_counts[step])
            metadata.raw_dp_execution_token_counts = raw_counts
            metadata.dp_execution_token_counts = tuple(max(1, count) for count in raw_counts)
            metadata.dp_is_decode = (0, 1) if is_prefill else (1, 1)
            kv_length = 2 if is_prefill else 2 + (step + 1) * width
            metadata.dp_global_kv_max_seq_lens = tuple(kv_length if count else 0 for count in dp_request_counts[step])
        assert graph.decode_graph_runner.can_execute(ids, metadata) == (not is_prefill)
        if not slots:
            before = tuple(state.clone() for state in _live_cache_states(eager_caches, width))
        expected = eager.execute(ids, positions, metadata, input_batch_metadata=batch)
        if not slots:
            for state, previous in zip(_live_cache_states(eager_caches, width), before):
                torch.testing.assert_close(state, previous, rtol=0, atol=0)
            before = tuple(state.clone() for state in _live_cache_states(caches, width))
        actual = graph.execute(ids, positions, metadata, input_batch_metadata=batch)
        if not slots:
            for state, previous in zip(_live_cache_states(caches, width), before):
                torch.testing.assert_close(state, previous, rtol=0, atol=0)
        assert isinstance(actual, tuple) and isinstance(expected, tuple)
        assert actual[0].shape == (ids.numel(), 128)
        assert actual[1].shape == (ids.numel(), 256)
        if not is_prefill:
            # Reuse captured storage after B grows/shrinks and eager prefill.
            batch_size = len(slots) if dp_rank is None else max(dp_request_counts[step])
            first_outputs = captured_outputs.setdefault(batch_size, actual)
            assert tuple(t.data_ptr() for t in actual) == tuple(t.data_ptr() for t in first_outputs)
        # Padding owns slot zero; compare all real state rows/checkpoints.
        torch.testing.assert_close(conv[1:].cpu(), eager_caches[0].conv[1:].cpu(), rtol=0, atol=0)
        torch.testing.assert_close(ssm[width:].cpu(), eager_caches[0].ssm[width:].cpu(), rtol=5e-3, atol=2.5e-5)
        # Padded BF16 projections can propagate rounding into near-zero KV values.
        torch.testing.assert_close(key[1:].cpu(), eager_caches[1].key[1:].cpu(), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(value[1:].cpu(), eager_caches[1].value[1:].cpu(), rtol=1e-2, atol=1e-2)
        if slots:
            for output_index, (output, reference) in enumerate(zip(actual, expected)):
                assert torch.isfinite(output).all()
                output_name = "hidden" if output_index == 0 else "aux"
                case = f"rank={dp_rank}, width={width}, step={step}, output={output_name}"
                torch.testing.assert_close(
                    output.cpu(),
                    reference.cpu(),
                    rtol=2e-2,
                    atol=2e-2,
                    msg=lambda message, case=case: f"{case}: {message}",
                )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
