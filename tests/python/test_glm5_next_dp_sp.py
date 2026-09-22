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

"""CPU DP/SP contracts; real communication requires separate service validation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from xllm.python.layers.moe_parallel import Eplv2CommPolicy, TokenParallelLayout
from xllm.python.layers.npu.glm5_next_metadata import (
    Glm5NextEplv2Metadata,
    Glm5NextEplv2MetadataBuilder,
)
from xllm.python.models import glm5_next


def _cfg(dp: int, rank: int) -> glm5_next.Glm5NextConfig:
    tp = 8 // dp
    return glm5_next.Glm5NextConfig(
        hidden_size=8,
        moe_intermediate_size=16,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        tp_size=tp,
        tp_rank=rank % tp,
        dp_size=dp,
        dp_rank=rank // tp,
        ep_size=8,
        ep_rank=rank,
        moe_tp_size=1,
        moe_tp_rank=0,
        expert_parallel_degree=2,
    )


def _context(counts: tuple[int, ...], raw: tuple[int, ...], **kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(
        metadata=SimpleNamespace(
            dp_execution_token_counts=counts,
            raw_dp_execution_token_counts=raw,
            is_spec_verify=False,
            expanded_decode_metadata=None,
            **kwargs,
        )
    )


@pytest.mark.parametrize("dp", [1, 2, 4, 8])
def test_each_global_token_has_one_sp_owner_with_common_ep_capacity(dp: int) -> None:
    counts = tuple(1 + 3 * row for row in range(dp))
    layouts = []
    for rank in range(8):
        cfg = _cfg(dp, rank)
        cfg._validate_moe_parallelism()
        layout = TokenParallelLayout.from_dp(counts[cfg.dp_rank], cfg.tp_size, cfg.tp_rank, dp, cfg.dp_rank, counts)
        layouts.append(layout)
        source = torch.arange(counts[cfg.dp_rank]).view(-1, 1)
        local = layout.shard(source)
        transported = layout.pad_dispatch(local)
        assert transported.shape[0] == layout.dispatch_tokens
        torch.testing.assert_close(transported[: layout.shard_tokens], local)
    assert len({layout.dispatch_tokens for layout in layouts}) == 1
    for dp_rank, count in enumerate(counts):
        owned = [
            value
            for rank in range(dp_rank * (8 // dp), (dp_rank + 1) * (8 // dp))
            for value in layouts[rank].shard(torch.arange(count))[: layouts[rank].valid_tokens].tolist()
        ]
        assert owned == list(range(count))


@pytest.mark.parametrize("counts", [None, (), (1,), (0, 2), (2.5, 2), torch.tensor([3, 2])])
def test_invalid_dp_counts_fail_without_device_queries(counts: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        TokenParallelLayout.from_dp(3, 4, 0, 2, 0, counts)


@pytest.mark.parametrize("dp", [2, 4])
@pytest.mark.parametrize("mode", ["mc2", "alltoall"])
def test_global_counts_select_one_backend_and_keep_empty_dp_inactive(
    dp: int, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = (9,) + (0,) * (dp - 1)
    counts = tuple(max(1, value) for value in raw)
    # With capacity1, locally empty ranks must still follow the larger DP's
    # All-to-All choice, rather than independently choosing MC2.
    policy = Eplv2CommPolicy(32 if mode == "mc2" else 1)
    for rank in range(8):
        cfg = _cfg(dp, rank)
        context = _context(counts, raw)
        rows = counts[cfg.dp_rank]
        mask = torch.arange(rows) < raw[cfg.dp_rank]
        context.execution_contexts = {Glm5NextEplv2Metadata: Glm5NextEplv2Metadata(mask)}
        monkeypatch.setattr(glm5_next, "get_forward_context_or_none", lambda context=context: context)
        monkeypatch.setattr(glm5_next, "in_acl_graph", lambda: False)
        hidden = torch.ones(rows, 8)
        layout, mask = glm5_next._eplv2_token_layout(cfg, hidden, policy)
        assert layout.routed_backend == mode
        assert mask.sum().item() == (layout.valid_tokens if cfg.dp_rank == 0 else 0)


@pytest.mark.parametrize("backend", ["mc2", "alltoall"])
def test_only_routed_transport_is_padded_shared_keeps_local_rows(backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(2, 4)
    layer = glm5_next.Glm5NextMoE(cfg, torch.float32, torch.device("cpu"))
    layer.use_w8a8 = True
    layer._ep_dispatch_group_info = ("ep8", 4, 8)
    for name in ("experts_w13", "experts_w2", "experts_w13_scale", "experts_w2_scale"):
        setattr(layer, name, torch.empty(0))
    layer.shared_experts = nn.Identity()
    shared = Mock(side_effect=lambda x: x * 3)
    monkeypatch.setattr(layer.shared_experts, "forward", shared)
    monkeypatch.setattr(
        layer, "_route_ep", lambda x: (torch.ones(x.shape[0], 2), torch.zeros(x.shape[0], 2, dtype=torch.int32))
    )
    routed = Mock(side_effect=lambda x, *args, **kwargs: x * 5)
    name = "ep_moe_w8a8" if backend == "mc2" else "ep_moe_w8a8_alltoall"
    monkeypatch.setattr(glm5_next.kernels, name, routed, raising=False)
    monkeypatch.setattr(glm5_next, "get_forward_context_or_none", lambda: None)
    layout = TokenParallelLayout(1, 4, 0, backend, dispatch_rows=3)
    hidden = torch.ones(1, 8)
    result = layer(hidden, token_layout=layout, token_mask=torch.ones(1, dtype=torch.bool))
    torch.testing.assert_close(result, hidden * (5 * cfg.routed_scaling_factor + 3))
    assert shared.call_args.args[0].shape == (1, 8)
    assert routed.call_args.args[0].shape == ((3 if backend == "mc2" else 1), 8)
    if backend == "mc2":
        assert routed.call_args.kwargs["x_active_mask"].tolist() == [True, False, False]
        assert routed.call_args.kwargs["ep_rank"] == 4


@pytest.mark.parametrize("rank,expected", [(0, [True, True, True, False]), (1, [True, False, False, False])])
def test_eplv2_metadata_builder_build_allocate_update_lifecycle(rank: int, expected: list[bool]) -> None:
    builder = Glm5NextEplv2MetadataBuilder(dp_size=2, dp_rank=rank)
    metadata = _context((3, 1), (3, 1)).metadata
    eager_batch = SimpleNamespace(
        num_tokens=3 if rank == 0 else 1,
        input_ids=torch.zeros(3 if rank == 0 else 1),
        is_dummy=False,
    )

    eager = builder.build(eager_batch, metadata)
    assert isinstance(eager, Glm5NextEplv2Metadata)
    assert eager.local_token_mask.tolist() == expected[: eager_batch.num_tokens]

    graph_batch = SimpleNamespace(
        num_tokens=eager_batch.num_tokens,
        num_tokens_after_padding=4,
        input_ids=torch.zeros(4),
        is_dummy=False,
    )
    persistent = builder.allocate_persistent(graph_batch, metadata)
    address = persistent.local_token_mask.data_ptr()
    builder.update_persistent(persistent, graph_batch, metadata)
    assert persistent.local_token_mask.tolist() == expected
    assert persistent.local_token_mask.data_ptr() == address

    metadata.raw_dp_execution_token_counts = (0, 2) if rank == 0 else (2, 0)
    graph_batch.num_tokens = 1
    graph_batch.is_dummy = True
    builder.update_persistent(persistent, graph_batch, metadata)
    assert persistent.local_token_mask.tolist() == [False] * 4
    assert persistent.local_token_mask.data_ptr() == address


def test_dp_graph_uses_local_mask_without_baking_runtime_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(2, 4)
    mask = torch.tensor([False, False, False, False])
    context = _context((4, 4), ())
    context.execution_contexts = {Glm5NextEplv2Metadata: Glm5NextEplv2Metadata(mask)}
    context.execution_state = object()
    monkeypatch.setattr(glm5_next, "get_forward_context_or_none", lambda: context)
    monkeypatch.setattr(glm5_next, "in_acl_graph", lambda: True)
    policy = Eplv2CommPolicy(32)
    layout, local = glm5_next._eplv2_token_layout(cfg, torch.ones(4, 8), policy)
    assert layout.dispatch_tokens == 1 and layout.routed_backend == "mc2"
    assert not local.any()
    mask[0] = True
    assert local[0]


def test_dp_graph_does_not_reuse_megamoe_global_dummy_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(2, 4)
    context = _context((4, 4), (), mega_moe_token_mask=torch.ones(8, dtype=torch.int8))
    monkeypatch.setattr(glm5_next, "get_forward_context_or_none", lambda: context)
    monkeypatch.setattr(glm5_next, "in_acl_graph", lambda: True)
    with pytest.raises(ValueError, match="stable active-token mask"):
        glm5_next._eplv2_token_layout(cfg, torch.ones(4, 8), Eplv2CommPolicy(32))


def test_dp1_graph_does_not_fall_back_to_megamoe_global_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """A DP1 graph without a local mask must fail, never borrow MegaMoE's."""
    cfg = _cfg(1, 0)
    context = _context((4,), (), mega_moe_token_mask=torch.ones(4, dtype=torch.int8))
    monkeypatch.setattr(glm5_next, "get_forward_context_or_none", lambda: context)
    monkeypatch.setattr(glm5_next, "in_acl_graph", lambda: True)
    with pytest.raises(ValueError, match="stable active-token mask"):
        glm5_next._eplv2_token_layout(cfg, torch.ones(4, 8), Eplv2CommPolicy(32))
