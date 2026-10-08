# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from xllm.python.layers import moe_parallel
from xllm.python.layers.moe_parallel import TokenParallelLayout
from xllm.python.model_executor import cp_utils
from xllm.python.models import glm5_next
from xllm.python.models.aux_hidden_capture import AuxHiddenCapture


class _Embedding(nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self._events = events

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._events.append("embedding")
        return input_ids.to(torch.float32).unsqueeze(-1)


class _DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, events: list[str]) -> None:
        super().__init__()
        self.layer_id = layer_id
        self._events = events
        self.positions: torch.Tensor | None = None
        self.attention_mask: torch.Tensor | None = None
        self.prev_topk: torch.Tensor | None = None
        self.output_topk: torch.Tensor | None = None

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
        prev_topk: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.positions = positions
        self.attention_mask = attention_mask
        self.prev_topk = prev_topk
        self.output_topk = hidden[:, :1, 0].clone()
        self._events.append(f"layer_{self.layer_id}")
        return hidden + self.layer_id + 1, self.output_topk


class _Norm(nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self._events = events

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self._events.append("norm")
        return hidden


class _HyperHead(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.mean(dim=2)


def _make_model(events: list[str]) -> tuple[glm5_next.Glm5NextModel, list[_DecoderLayer]]:
    model = glm5_next.Glm5NextModel.__new__(glm5_next.Glm5NextModel)
    nn.Module.__init__(model)
    model.cfg = SimpleNamespace(hidden_size=1, hc_mult=4)
    model.embed_tokens = _Embedding(events)
    layers = [_DecoderLayer(layer_id, events) for layer_id in range(2)]
    model.layers = nn.ModuleList(layers)
    model.norm = _Norm(events)
    model.hc_head = _HyperHead()
    model.aux_hidden_capture = AuxHiddenCapture(())
    model._inputs_embeds = None
    return model, layers


def test_cp_model_loop_shards_rows_and_merges_final_hidden() -> None:
    events: list[str] = []
    model, layers = _make_model(events)
    cp_context = SimpleNamespace(shard_valid_mask=torch.tensor([True, False]))
    merged_output = torch.tensor([[13.0], [23.0], [33.0], [43.0]])

    def shard_rows(hidden: torch.Tensor, context: object) -> torch.Tensor:
        assert context is cp_context
        events.append("shard_rows")
        return hidden.index_select(0, torch.tensor([3, 0]))

    def shard_positions(positions: torch.Tensor, context: object) -> torch.Tensor:
        assert context is cp_context
        events.append("shard_positions")
        return positions.index_select(0, torch.tensor([3, 0]))

    def merge_rows(hidden: torch.Tensor, context: object) -> torch.Tensor:
        assert context is cp_context
        torch.testing.assert_close(hidden, torch.tensor([[43.0], [13.0]]))
        events.append("merge_rows")
        return merged_output

    with (
        patch.object(glm5_next, "get_forward_context", return_value=SimpleNamespace(cp_context=cp_context)),
        patch.object(glm5_next, "cp_shard_rows", side_effect=shard_rows),
        patch.object(glm5_next, "cp_shard_positions", side_effect=shard_positions),
        patch.object(glm5_next, "cp_merge_rows", side_effect=merge_rows),
    ):
        output = model(
            torch.tensor([10, 20, 30, 40]),
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

    assert events == [
        "embedding",
        "shard_rows",
        "shard_positions",
        "layer_0",
        "layer_1",
        "norm",
        "merge_rows",
    ]
    torch.testing.assert_close(layers[0].positions, torch.tensor([[3, 0]], dtype=torch.int32))
    torch.testing.assert_close(layers[0].attention_mask, torch.tensor([[True, False]]))
    assert layers[1].prev_topk is layers[0].output_topk
    torch.testing.assert_close(output, merged_output)


class _SpMoe(glm5_next.Glm5NextMoE):
    def __init__(self) -> None:
        nn.Module.__init__(self)


class _SpDecoder(nn.Module):
    """Last sparse layer: keep only this TP rank's shard row."""

    def __init__(self) -> None:
        super().__init__()
        self.layer_id = 0
        self.mlp = _SpMoe()

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
        prev_topk: torch.Tensor | None,
        *,
        input_layout: TokenParallelLayout | None = None,
        output_layout: TokenParallelLayout | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        assert output_layout is not None and output_layout.shard_tokens == 1
        local = hidden[:, :1] + 1
        return local, None


def test_pcp_sp_gathers_tp_rows_before_cp_merge() -> None:
    """CP=2, TP=4, 8 tokens: SP owns 1 row, CP merge needs all 4 local rows."""
    events: list[str] = []
    model, _layers = _make_model(events)
    model.cfg = SimpleNamespace(
        hidden_size=1,
        hc_mult=4,
        expert_parallel_degree=2,
        eplv2_sequence_parallel=True,
        tp_size=4,
        tp_rank=0,
        layers_to_capture=(),
    )
    model.layers = nn.ModuleList([_SpDecoder()])
    layout = TokenParallelLayout(4, 4, 0)
    model._sp_layout_and_mask = lambda hidden, attention_mask: (layout, torch.ones(1, dtype=torch.bool))
    cp_context = SimpleNamespace(
        cp_size=2,
        total_local=4,
        shard_gather_index=torch.arange(4),
        shard_valid_mask=torch.ones(4, dtype=torch.bool),
        restore_index=torch.arange(8),
    )

    def tp_all_gather(value: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
        assert dim == 0 and world_size == 4 and value.shape[0] == 1
        events.append("tp_gather")
        return value.repeat(4, *([1] * (value.dim() - 1)))

    def all_gather(value: torch.Tensor, dim: int, world_size: int, group_name: str) -> torch.Tensor:
        assert dim == 0 and world_size == 2 and group_name == "cp"
        assert value.shape[0] == cp_context.total_local
        events.append(f"cp_merge:{tuple(value.shape)}")
        return torch.cat((value, value + 10), dim=0)

    with (
        patch.object(glm5_next, "get_forward_context_or_none", return_value=SimpleNamespace(cp_context=cp_context)),
        patch.object(moe_parallel.distributed, "tp_all_gather", side_effect=tp_all_gather, create=True),
        patch.object(cp_utils.distributed, "all_gather", side_effect=all_gather, create=True),
    ):
        output = model(torch.arange(8), torch.arange(8))

    assert events[-2:] == ["tp_gather", "cp_merge:(4, 1)"]
    assert output.shape == (8, 1)


def test_pcp_sp_aux_capture_merges_full_cp_local_rows() -> None:
    events: list[str] = []
    model, _layers = _make_model(events)
    model.cfg = SimpleNamespace(
        hidden_size=1,
        hc_mult=4,
        expert_parallel_degree=2,
        eplv2_sequence_parallel=True,
        tp_size=4,
        tp_rank=0,
        layers_to_capture=(0,),
    )
    model.layers = nn.ModuleList([_SpDecoder()])
    model.aux_hidden_capture = AuxHiddenCapture(
        (0,),
        transform=lambda streams: streams.mean(dim=2).reshape(-1, 1),
    )
    layout = TokenParallelLayout(4, 4, 0)
    model._sp_layout_and_mask = lambda hidden, attention_mask: (layout, torch.ones(1, dtype=torch.bool))
    cp_context = SimpleNamespace(
        cp_size=2,
        total_local=4,
        shard_gather_index=torch.arange(4),
        shard_valid_mask=torch.ones(4, dtype=torch.bool),
        restore_index=torch.arange(8),
    )
    merged_shapes: list[tuple[int, ...]] = []

    def tp_all_gather(value: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
        assert value.shape[0] == 1
        return value.repeat(4, *([1] * (value.dim() - 1)))

    def all_gather(value: torch.Tensor, dim: int, world_size: int, group_name: str) -> torch.Tensor:
        assert group_name == "cp" and value.shape[0] == 4
        merged_shapes.append(tuple(value.shape))
        return torch.cat((value, value + 10), dim=0)

    with (
        patch.object(glm5_next, "get_forward_context_or_none", return_value=SimpleNamespace(cp_context=cp_context)),
        patch.object(moe_parallel.distributed, "tp_all_gather", side_effect=tp_all_gather, create=True),
        patch.object(cp_utils.distributed, "all_gather", side_effect=all_gather, create=True),
    ):
        hidden, aux = model(torch.arange(8), torch.arange(8))

    assert hidden.shape == (8, 1)
    assert aux.shape[0] == 8
    # The model exit packs h [4, 1] and the capture [4, 1] along the feature
    # axis, so the full CP-local rows still reach the CP group -- now as one
    # collective of packed width 2 instead of two of width 1.
    assert merged_shapes == [(4, 2)]


def _cp2_context(rank: int) -> SimpleNamespace:
    shard_index = torch.tensor([0, 3], dtype=torch.int64) if rank == 0 else torch.tensor([1, 2], dtype=torch.int64)
    # Global 4 tokens, zigzag cp_size=2, chunk_len=1: every rank owns two real
    # rows, one per (sequence, half) segment (rank 0: global 0 then 3; rank 1:
    # global 1 then 2). query_index is therefore the identity here, while the
    # per-segment causal prefixes differ per rank.
    segment_kv_seq_lens = [1, 4] if rank == 0 else [2, 3]
    return SimpleNamespace(
        cp_size=2,
        cp_rank=rank,
        total_local=2,
        shard_index=shard_index,
        shard_gather_index=shard_index,
        shard_valid_mask=torch.ones(2, dtype=torch.bool),
        restore_index=torch.tensor([0, 2, 3, 1], dtype=torch.int64),
        query_index=torch.tensor([0, 1], dtype=torch.int64),
        q_cu_seqlens=[1, 2],
        q_cu_seqlens_tensor=torch.tensor([1, 2], dtype=torch.int32),
        segment_seq_indices=torch.tensor([0, 0], dtype=torch.int64),
        segment_kv_seq_lens=segment_kv_seq_lens,
        segment_kv_seq_lens_tensor=torch.tensor(segment_kv_seq_lens, dtype=torch.int32),
    )


def _patch_cp_gather(monkeypatch, remote_by_local: dict[tuple[object, ...], torch.Tensor]) -> MagicMock:
    def all_gather(value: torch.Tensor, dim: int, world_size: int, group_name: str) -> torch.Tensor:
        assert dim == 0
        assert world_size == 2
        assert group_name == "cp"
        key = tuple(value.reshape(-1).tolist())
        # The peer's rows are supplied flat; match the gathered value's trailing
        # dims so one map serves 2-D and 3-D call sites.
        remote = remote_by_local[key].reshape(-1, *value.shape[1:])
        return torch.cat([value, remote], dim=0)

    gather = MagicMock(side_effect=all_gather)
    monkeypatch.setattr(glm5_next.distributed, "all_gather", gather, raising=False)
    return gather


def test_kda_cp_materializes_global_rows_and_reshards_output(monkeypatch) -> None:
    attention = glm5_next.Glm5NextKdaAttention.__new__(glm5_next.Glm5NextKdaAttention)
    nn.Module.__init__(attention)
    attention.hidden_size = 1
    attention.head_dim = 1
    attention.qkv_dim = 1
    attention.conv_dim = 3
    attention.num_heads_local = 1
    attention.tp = 1
    attention.in_proj_qkvbfg_a = lambda hidden: torch.cat(
        (hidden.expand(-1, -1, 3), hidden, hidden, hidden),
        dim=-1,
    )
    attention.input_projection_sizes = (3, 1, 2)
    attention.forget_gate = SimpleNamespace(raw_projection=lambda hidden: hidden.unsqueeze(-1) * 10)
    attention.g_b_proj = lambda hidden: hidden * 100
    attention._fg_b_weight = None
    attention.o_norm = MagicMock(side_effect=lambda core, gate: core + gate)
    attention.o_proj = nn.Identity()

    cp_context = _cp2_context(0)
    local_hidden = torch.tensor([[[1.0], [4.0]]])
    remote_hidden = torch.tensor([[2.0], [3.0]])
    remote_by_local = {
        (1.0, 1.0, 1.0, 4.0, 4.0, 4.0): remote_hidden.expand(-1, 3),
        (10.0, 40.0): remote_hidden.mul(10).unsqueeze(-1),
        (1.0, 4.0): remote_hidden,
    }
    gather = _patch_cp_gather(monkeypatch, remote_by_local)

    backend = MagicMock()
    backend.execute_linear.side_effect = lambda mixed_qkv, _beta, _layer, *, raw_gate_proj: (
        mixed_qkv[:, :1].transpose(1, 2).unsqueeze(2)
    )
    with patch.object(
        glm5_next,
        "get_forward_context_or_none",
        return_value=SimpleNamespace(attention_backend=backend, cp_context=cp_context),
    ):
        output = attention(local_hidden, torch.tensor([[0, 3]], dtype=torch.int32), torch.tensor([[True, True]]))

    mixed_qkv, beta, layer = backend.execute_linear.call_args.args
    raw_gate_proj = backend.execute_linear.call_args.kwargs["raw_gate_proj"]
    assert layer is attention
    expected_hidden = torch.tensor([1.0, 2.0, 3.0, 4.0])
    torch.testing.assert_close(mixed_qkv, expected_hidden.view(1, 1, 4).expand(1, 3, 4))
    torch.testing.assert_close(raw_gate_proj, expected_hidden.view(1, 4, 1, 1) * 10)
    torch.testing.assert_close(beta, expected_hidden.view(1, 4, 1))
    assert gather.call_count == 3
    torch.testing.assert_close(attention.o_norm.call_args.args[1], local_hidden.unsqueeze(-1) * 100)
    torch.testing.assert_close(output, torch.tensor([[[101.0], [404.0]]]))


def test_kda_cp_one_keeps_full_postprocessing() -> None:
    attention = glm5_next.Glm5NextKdaAttention.__new__(glm5_next.Glm5NextKdaAttention)
    nn.Module.__init__(attention)
    attention.hidden_size = 1
    attention.head_dim = 1
    attention.qkv_dim = 1
    attention.conv_dim = 3
    attention.num_heads_local = 1
    attention.tp = 1
    attention.in_proj_qkvbfg_a = lambda hidden: torch.cat(
        (hidden.expand(-1, -1, 3), hidden, hidden, hidden),
        dim=-1,
    )
    attention.input_projection_sizes = (3, 1, 2)
    attention.forget_gate = SimpleNamespace(raw_projection=lambda hidden: hidden.unsqueeze(-1) * 10)
    attention.g_b_proj = lambda hidden: hidden * 100
    attention._fg_b_weight = None
    attention.o_norm = MagicMock(side_effect=lambda core, gate: core + gate)
    attention.o_proj = nn.Identity()
    hidden = torch.tensor([[[1.0], [2.0], [3.0]]])
    backend = MagicMock()
    backend.execute_linear.return_value = hidden.unsqueeze(2)

    with (
        patch.object(
            glm5_next,
            "get_forward_context_or_none",
            return_value=SimpleNamespace(attention_backend=backend, cp_context=None),
        ),
        patch.object(glm5_next, "cp_merge_rows") as merge_rows,
        patch.object(glm5_next, "cp_shard_rows") as shard_rows,
    ):
        output = attention(hidden, torch.tensor([[0, 1, 2]], dtype=torch.int32), torch.ones(1, 3, dtype=torch.bool))

    merge_rows.assert_not_called()
    shard_rows.assert_not_called()
    torch.testing.assert_close(output, hidden * 101)


def _cp2_padded_context() -> SimpleNamespace:
    """Global 3 tokens under zigzag cp_size=2, rank 0: one real row (global 0)
    plus one padding row. ``query_index`` is ``[0]`` -- not the identity -- so a
    bug that drops the query selection and uses every local row is observable.
    """
    return SimpleNamespace(
        cp_size=2,
        cp_rank=0,
        total_local=2,
        shard_index=torch.tensor([0, -1], dtype=torch.int64),
        shard_gather_index=torch.tensor([0, 0], dtype=torch.int64),
        shard_valid_mask=torch.tensor([True, False]),
        restore_index=torch.tensor([0, 2, 3], dtype=torch.int64),
        query_index=torch.tensor([0], dtype=torch.int64),
        q_cu_seqlens=[1],
        q_cu_seqlens_tensor=torch.tensor([1], dtype=torch.int32),
        segment_seq_indices=torch.tensor([0], dtype=torch.int64),
        segment_kv_seq_lens=[1],
        segment_kv_seq_lens_tensor=torch.tensor([1], dtype=torch.int32),
    )


class _KdaNorm(nn.Module):
    def forward(self, core: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        assert core.shape == (1, 2, 2, 2)
        assert gate.shape == (1, 2, 2, 2)
        return core


class _KdaOutputProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_shape: tuple[int, ...] | None = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.input_shape = tuple(hidden.shape)
        assert hidden.shape == (1, 2, 4)
        return torch.tensor(
            [[[1.0, 2.0, 3.0], [float("nan"), float("inf"), -float("inf")]]],
            device=hidden.device,
        )


class _MlaOutputProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_shape: tuple[int, ...] | None = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.input_shape = tuple(hidden.shape)
        assert hidden.shape == (2, 6)
        return torch.tensor(
            [[1.0, 2.0, 3.0], [float("nan"), float("inf"), -float("inf")]],
            device=hidden.device,
        )


def test_kda_cp_projects_only_local_rows_and_overwrites_padding(monkeypatch) -> None:
    attention = glm5_next.Glm5NextKdaAttention.__new__(glm5_next.Glm5NextKdaAttention)
    nn.Module.__init__(attention)
    attention.hidden_size = 3
    attention.head_dim = 2
    attention.qkv_dim = 4
    attention.conv_dim = 12
    attention.num_heads_local = 2
    attention.tp = 2
    attention.input_projection_sizes = (12, 2, 4)
    attention.in_proj_qkvbfg_a = lambda hidden: torch.cat(
        (
            hidden[..., :1].expand(-1, -1, 12),
            hidden[..., :1].expand(-1, -1, 2),
            hidden[..., :1].expand(-1, -1, 4),
        ),
        dim=-1,
    )
    attention._project_fg = lambda latents: (latents.view(1, 2, 2, 2), latents.view(1, 2, 2, 2))
    attention.o_norm = _KdaNorm()
    attention.o_proj = _KdaOutputProjection()

    cp_context = _cp2_padded_context()
    local_hidden = torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]])
    global_core = torch.arange(12, dtype=torch.float32).view(1, 3, 2, 2)
    backend = MagicMock()
    backend.execute_linear.return_value = global_core
    merge = MagicMock(
        side_effect=[
            torch.ones(3, 12),
            torch.ones(3, 2, 2),
            torch.ones(3, 2),
        ]
    )

    def compensate(output: torch.Tensor) -> None:
        output.add_(7)

    with (
        patch.object(
            glm5_next,
            "get_forward_context_or_none",
            return_value=SimpleNamespace(attention_backend=backend, cp_context=cp_context),
        ),
        patch.object(glm5_next, "cp_merge_rows", merge),
        patch.object(glm5_next.distributed, "all_reduce_", side_effect=compensate, create=True) as all_reduce,
    ):
        output = attention(local_hidden, torch.tensor([[0, 0]], dtype=torch.int32), torch.tensor([[True, False]]))

    assert merge.call_count == 3
    assert backend.execute_linear.call_args.args[0].shape == (1, 12, 3)
    assert backend.execute_linear.call_args.args[1].shape == (1, 3, 2)
    assert backend.execute_linear.call_args.kwargs["raw_gate_proj"].shape == (1, 3, 2, 2)
    assert attention.o_proj.input_shape == (1, 2, 4)
    all_reduce.assert_called_once()
    torch.testing.assert_close(output[0, 0], torch.tensor([8.0, 9.0, 10.0]))
    assert torch.equal(output[0, 1], torch.zeros(3))


def test_kda_pcp_eplv2_sp_masks_padding_before_reducing() -> None:
    attention = glm5_next.Glm5NextKdaAttention.__new__(glm5_next.Glm5NextKdaAttention)
    nn.Module.__init__(attention)
    attention.hidden_size = 3
    attention.head_dim = 2
    attention.conv_dim = 12
    attention.num_heads_local = 2
    attention.input_projection_sizes = (12, 2, 4)
    attention.cfg = SimpleNamespace(hidden_size=3)
    attention.in_proj_qkvbfg_a = lambda hidden: torch.cat(
        (
            hidden[..., :1].expand(-1, -1, 12),
            hidden[..., :1].expand(-1, -1, 2),
            hidden[..., :1].expand(-1, -1, 4),
        ),
        dim=-1,
    )
    attention._project_fg = lambda latent: (latent.view(1, 2, 2, 2), latent.view(1, 2, 2, 2))
    attention.o_norm = _KdaNorm()
    attention.o_proj = _KdaOutputProjection()
    backend = MagicMock()
    backend.execute_linear.return_value = torch.arange(12, dtype=torch.float32).view(1, 3, 2, 2)
    with (
        patch.object(
            glm5_next,
            "get_forward_context_or_none",
            return_value=SimpleNamespace(attention_backend=backend, cp_context=_cp2_padded_context()),
        ),
        patch.object(
            glm5_next, "cp_merge_rows", side_effect=[torch.ones(3, 12), torch.ones(3, 2, 2), torch.ones(3, 2)]
        ),
        patch.object(glm5_next.distributed, "all_reduce_", side_effect=AssertionError("SP requires RS"), create=True),
    ):
        output = attention(
            torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]]),
            torch.tensor([[0, 0]], dtype=torch.int32),
            torch.tensor([[True, False]]),
            output_layout=TokenParallelLayout(2, 1, 0),
        )
    torch.testing.assert_close(output[0, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(output[0, 1], torch.zeros(3))


def _make_mla_attention(indexer: object | None) -> glm5_next.Glm5NextMlaAttention:
    attention = glm5_next.Glm5NextMlaAttention.__new__(glm5_next.Glm5NextMlaAttention)
    nn.Module.__init__(attention)
    attention.hidden_size = 1
    attention.qk_nope_head_dim = 1
    attention.qk_rope_head_dim = 0
    attention.qk_head_dim = 1
    attention.kv_lora_rank = 1
    attention.num_heads_local = 1
    attention.v_head_dim = 1
    attention.cfg = SimpleNamespace(tp_size=1)
    attention.q_a_proj = nn.Identity()
    attention.q_a_layernorm = nn.Identity()
    attention.q_b_proj = nn.Identity()
    attention.kv_a_proj_with_mqa = nn.Identity()
    attention.kv_a_layernorm = nn.Identity()
    attention._qkv_a_weight = None
    attention.o_proj = nn.Identity()
    attention.register_buffer("W_UK", torch.ones(1, 1, 1), persistent=False)
    attention.register_buffer("W_UV", torch.ones(1, 1, 1), persistent=False)
    attention.indexer = indexer
    return attention


def test_mla_cp_one_keeps_full_postprocessing() -> None:
    attention = _make_mla_attention(MagicMock())
    attention.indexer.select_qli.return_value = torch.tensor([[[10]], [[20]], [[30]]], dtype=torch.int32)
    hidden = torch.tensor([[[1.0], [2.0], [3.0]]])
    backend = MagicMock()
    backend.mla_index_context.return_value = object()
    backend.execute_mla.return_value = hidden.reshape(3, 1, 1)

    with (
        patch.object(
            glm5_next,
            "get_forward_context",
            return_value=SimpleNamespace(cp_context=None, attention_backend=backend),
        ),
        patch.object(glm5_next, "cp_merge_rows") as merge_rows,
        patch.object(glm5_next, "cp_shard_rows") as shard_rows,
    ):
        output, topk = attention(
            hidden,
            torch.tensor([[0, 1, 2]], dtype=torch.int32),
            torch.ones(1, 3, dtype=torch.bool),
        )

    merge_rows.assert_not_called()
    shard_rows.assert_not_called()
    torch.testing.assert_close(output, hidden.reshape(3, 1))
    torch.testing.assert_close(topk, torch.tensor([[[10]], [[20]], [[30]]], dtype=torch.int32))


def test_mla_cp_projects_global_rows_and_reshards_output() -> None:
    """The model-side CP row layout is partition-independent.

    ``Glm5NextMlaAttention`` merges to the complete logical stream, runs the
    indexer and every projection on those global rows, and reshards the
    backend output. Only the attention backend partitions the query rows, so
    the projection GEMMs keep ``M = the global token count`` exactly as in
    the ``cp_size == 1`` path.
    """
    indexer = MagicMock()
    global_topk = torch.arange(6, dtype=torch.int32).view(3, 1, 2)
    indexer.select_qli.return_value = global_topk
    attention = _make_mla_attention(indexer)
    attention.hidden_size = 3
    attention.qk_nope_head_dim = 2
    attention.qk_head_dim = 2
    attention.kv_lora_rank = 3
    attention.num_heads_local = 2
    attention.v_head_dim = 3
    attention.cfg = SimpleNamespace(tp_size=2)
    attention.q_b_proj = nn.Linear(3, 4, bias=False)
    attention.kv_a_proj_with_mqa = nn.Identity()
    attention.o_proj = _MlaOutputProjection()
    attention.W_UK = torch.ones(2, 2, 3)
    attention.W_UV = torch.ones(2, 3, 3)

    cp_context = _cp2_padded_context()
    global_hidden = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    global_positions = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    global_attn_out = torch.arange(18, dtype=torch.float32).view(3, 2, 3)
    backend = MagicMock()
    backend.mla_index_context.return_value = object()
    backend.execute_mla.return_value = global_attn_out
    merge = MagicMock(side_effect=[global_hidden.squeeze(0), global_positions.reshape(-1, 1)])

    def fake_shard(value: torch.Tensor, _ctx: object) -> torch.Tensor:
        # CP-local layout: this rank's one real row plus one zeroed padding row.
        rows = value.index_select(0, torch.tensor([0]))
        padding = torch.zeros(1, *value.shape[1:], dtype=value.dtype)
        return torch.cat([rows, padding], dim=0)

    shard = MagicMock(side_effect=fake_shard)

    with (
        patch.object(
            glm5_next,
            "get_forward_context",
            return_value=SimpleNamespace(cp_context=cp_context, attention_backend=backend),
        ),
        patch.object(glm5_next, "cp_merge_rows", merge),
        patch.object(glm5_next, "cp_shard_rows", shard),
        patch.object(glm5_next.distributed, "all_reduce_", create=True),
    ):
        output, topk = attention(
            torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]]),
            torch.tensor([[0, 0]], dtype=torch.int32),
            torch.tensor([[True, False]]),
        )

    # Merge to global rows, then reshard: identical for either switch value.
    assert merge.call_count == 2
    assert indexer.select_qli.call_args.args[0].shape == (1, 3, 3)
    torch.testing.assert_close(indexer.select_qli.call_args.args[0], global_hidden)
    assert backend.execute_mla.call_args.args[0].shape == (3, 2, 3)
    assert backend.execute_mla.call_args.args[2].shape == (3, 1, 3)
    torch.testing.assert_close(backend.execute_mla.call_args.kwargs["topk"], global_topk)
    assert shard.call_count == 2
    assert attention.o_proj.input_shape == (2, 6)


def test_mla_pcp_eplv2_sp_masks_padding_before_reducing_and_returns_local_topk(monkeypatch) -> None:
    indexer = MagicMock()
    global_topk = torch.tensor([[[10]], [[20]], [[30]]], dtype=torch.int32)
    indexer.select_qli.return_value = global_topk
    attention = _make_mla_attention(indexer)
    cp_context = _cp2_padded_context()
    backend = MagicMock()
    backend.mla_index_context.return_value = object()
    backend.execute_mla.return_value = torch.tensor([[[11.0]], [[22.0]], [[33.0]]])
    merge = MagicMock(
        side_effect=[
            torch.tensor([[1.0], [2.0], [3.0]]),
            torch.tensor([[0], [1], [2]], dtype=torch.int32),
        ]
    )
    with (
        patch.object(
            glm5_next,
            "get_forward_context",
            return_value=SimpleNamespace(cp_context=cp_context, attention_backend=backend),
        ),
        patch.object(glm5_next, "cp_merge_rows", merge),
        patch.object(
            glm5_next.distributed, "all_reduce_", side_effect=AssertionError("SP must reduce-scatter"), create=True
        ),
    ):
        output, topk = attention(
            torch.tensor([[[1.0], [0.0]]]),
            torch.tensor([[0, 0]], dtype=torch.int32),
            torch.tensor([[True, False]]),
            output_layout=TokenParallelLayout(2, 1, 0),
        )
    torch.testing.assert_close(output, torch.tensor([[11.0], [0.0]]))
    torch.testing.assert_close(topk, torch.tensor([[[10]], [[0]]], dtype=torch.int32))


def test_mla_cp_indexer_and_backend_receive_global_rows_for_each_rank(monkeypatch) -> None:
    """MLA/DSA counterpart of the KDA rank-invariance test.

    Under CP the model merges every rank up to the complete logical stream
    before touching the indexer or the backend, so both ranks hand them the
    *same* global rows. Every projection GEMM therefore runs with the global
    batch shape on every rank, exactly as in the ``cp_size == 1`` path.
    """
    indexer = MagicMock()
    indexer.select_qli.return_value = torch.zeros(4, 1, 1, dtype=torch.int32)
    backend = MagicMock()
    backend.mla_index_context.return_value = object()
    backend.execute_mla.return_value = torch.zeros(4, 1, 1)
    global_hidden = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    global_positions = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)

    captured_by_rank: dict[int, dict[str, torch.Tensor]] = {}
    for rank, local_hidden, local_positions in (
        (0, torch.tensor([[[1.0], [4.0]]]), torch.tensor([[0, 3]], dtype=torch.int32)),
        (1, torch.tensor([[[2.0], [3.0]]]), torch.tensor([[1, 2]], dtype=torch.int32)),
    ):
        attention = _make_mla_attention(indexer)
        cp_context = _cp2_context(rank)
        merge = MagicMock(side_effect=[global_hidden.squeeze(0), global_positions.reshape(-1, 1)])
        indexer.select_qli.reset_mock()
        backend.execute_mla.reset_mock()
        with (
            patch.object(
                glm5_next,
                "get_forward_context",
                return_value=SimpleNamespace(cp_context=cp_context, attention_backend=backend),
            ),
            patch.object(glm5_next, "cp_merge_rows", merge),
            patch.object(glm5_next, "cp_shard_rows", MagicMock(return_value=torch.zeros(2, 1, 1))),
        ):
            attention(local_hidden, local_positions, torch.tensor([[True, True]]))
        assert merge.call_count == 2
        index_args = indexer.select_qli.call_args.args
        mla_args = backend.execute_mla.call_args.args
        captured_by_rank[rank] = {
            "index_hidden_states": index_args[0],
            "index_position_ids": index_args[2],
            "q_latent": mla_args[0],
            "k_latent_3d": mla_args[2],
        }

    for rank in (0, 1):
        torch.testing.assert_close(captured_by_rank[rank]["index_hidden_states"], global_hidden)
        torch.testing.assert_close(captured_by_rank[rank]["index_position_ids"], global_positions)
        torch.testing.assert_close(captured_by_rank[rank]["q_latent"], global_hidden.reshape(4, 1, 1))
        torch.testing.assert_close(captured_by_rank[rank]["k_latent_3d"], global_hidden.reshape(4, 1, 1))

    # The two ranks are no longer fed different token rows: global means global.
    assert torch.equal(
        captured_by_rank[0]["index_hidden_states"],
        captured_by_rank[1]["index_hidden_states"],
    )


def test_dsa_cp_full_indexer_uses_global_rows_and_reshards_topk(monkeypatch) -> None:
    """A full-indexer DSA layer under CP merges to global rows first: the
    indexer sees the complete logical stream, the MLA backend sees global
    q/kv/topk (so its projections keep ``M = T_global``), and the returned
    top-k is resharded to CP-local for the next shared-indexer layer.
    ``previous_topk_indices`` is ignored because a full indexer recomputes.
    """
    indexer = MagicMock()
    global_topk = torch.tensor([[[10]], [[20]], [[30]], [[40]]], dtype=torch.int32)
    indexer.select_qli.return_value = global_topk
    attention = _make_mla_attention(indexer)
    cp_context = _cp2_context(0)
    previous_topk = torch.tensor([[[50]], [[80]]], dtype=torch.int32)
    global_hidden = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    global_positions = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    backend = MagicMock()
    backend.mla_index_context.return_value = object()
    backend.execute_mla.return_value = torch.tensor([[[11.0]], [[22.0]], [[33.0]], [[44.0]]])

    with (
        patch.object(
            glm5_next,
            "get_forward_context",
            return_value=SimpleNamespace(cp_context=cp_context, attention_backend=backend),
        ),
        patch.object(
            glm5_next,
            "cp_merge_rows",
            MagicMock(side_effect=[global_hidden.squeeze(0), global_positions.reshape(-1, 1)]),
        ) as merge,
        patch.object(
            glm5_next,
            "cp_shard_rows",
            side_effect=lambda value, _ctx: value.index_select(0, torch.tensor([0, 3])),
        ) as shard,
    ):
        output, topk = attention(
            torch.tensor([[[1.0], [4.0]]]),
            torch.tensor([[0, 3]], dtype=torch.int32),
            torch.tensor([[True, True]]),
            previous_topk,
        )

    # Hidden + positions are merged; a full indexer never merges prev top-k.
    assert merge.call_count == 2
    assert shard.call_count == 2
    index_args = indexer.select_qli.call_args.args
    torch.testing.assert_close(index_args[0], global_hidden)
    torch.testing.assert_close(index_args[1], global_hidden.reshape(4, 1))
    torch.testing.assert_close(index_args[2], global_positions)
    torch.testing.assert_close(index_args[3], torch.ones(1, 4, dtype=torch.bool))
    torch.testing.assert_close(backend.execute_mla.call_args.args[0], global_hidden.reshape(4, 1, 1))
    torch.testing.assert_close(backend.execute_mla.call_args.kwargs["topk"], global_topk)
    torch.testing.assert_close(output, torch.tensor([[11.0], [44.0]]))
    torch.testing.assert_close(topk, global_topk.index_select(0, torch.tensor([0, 3])))


def test_dsa_cp_shared_indexer_merges_local_topk_back_to_global(monkeypatch) -> None:
    """A shared-indexer DSA layer receives the preceding layer's CP-local top-k,
    merges it (with hidden and positions) back to global rows, and hands the
    global tensor to the backend. The returned top-k is resharded again for the
    next shared layer.
    """
    attention = _make_mla_attention(None)
    cp_context = _cp2_context(0)
    local_topk = torch.tensor([[[10]], [[40]]], dtype=torch.int32)
    global_topk = torch.tensor([[[10]], [[20]], [[30]], [[40]]], dtype=torch.int32)
    global_hidden = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    global_positions = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    backend = MagicMock()
    backend.execute_mla.return_value = torch.tensor([[[11.0]], [[22.0]], [[33.0]], [[44.0]]])

    with (
        patch.object(
            glm5_next,
            "get_forward_context",
            return_value=SimpleNamespace(cp_context=cp_context, attention_backend=backend),
        ),
        patch.object(
            glm5_next,
            "cp_merge_rows",
            MagicMock(
                side_effect=[
                    global_hidden.squeeze(0),
                    global_positions.reshape(-1, 1),
                    global_topk.reshape(4, -1),
                ]
            ),
        ) as merge,
        patch.object(
            glm5_next,
            "cp_shard_rows",
            side_effect=lambda value, _ctx: value.index_select(0, torch.tensor([0, 3])),
        ),
    ):
        output, topk = attention(
            torch.tensor([[[1.0], [4.0]]]),
            torch.tensor([[0, 3]], dtype=torch.int32),
            torch.tensor([[True, True]]),
            local_topk,
        )

    assert merge.call_count == 3  # hidden + positions + the shared top-k
    torch.testing.assert_close(backend.execute_mla.call_args.kwargs["topk"], global_topk)
    torch.testing.assert_close(output, torch.tensor([[11.0], [44.0]]))
    torch.testing.assert_close(topk, local_topk)


# ---------------------------------------------------------------------------
# C1: the indexer under CP -- it receives global rows, so the key-side cache
# write and the query-side top-k both run on the complete logical stream.
# ---------------------------------------------------------------------------


def _make_simple_indexer() -> glm5_next.Glm5NextIndexer:
    """A real ``Glm5NextIndexer`` with the non-compressed (index_kpool=1) path,
    which is the smallest body that still exercises the key-side write and the
    query-side reshaping without NPU kPool kernels."""
    indexer = glm5_next.Glm5NextIndexer.__new__(glm5_next.Glm5NextIndexer)
    nn.Module.__init__(indexer)
    indexer.layer_id = 0
    indexer.n_heads = 1
    indexer.head_dim = 1
    indexer.topk = 1
    indexer.index_kpool = 1
    indexer.index_kpool_compress = False
    indexer.index_kpool_always_select_tail = False
    indexer._uses_npu_compressed_tail = False
    indexer._update_compact_kpool = None
    indexer._compact_kpool_model_compatible = False
    indexer.softmax_scale = 1.0
    indexer.wq_b = nn.Linear(1, 1, bias=False)
    indexer.wk = nn.Linear(1, 1, bias=False)
    indexer.weights_proj = nn.Linear(1, 1, bias=False)
    for module in (indexer.wq_b, indexer.wk, indexer.weights_proj):
        with torch.no_grad():
            module.weight.fill_(1.0)
    indexer.k_norm = nn.Identity()
    indexer.register_buffer("_wk_weights_weight", None, persistent=False)
    indexer.index_kpool_compress_ape = torch.zeros(1, 1)
    indexer.index_kpool_compress_gate = torch.zeros(1, 1)
    return indexer


def _index_context_recording(cp_context, recorded: dict[str, torch.Tensor]) -> SimpleNamespace:
    def update_index_cache(values: torch.Tensor, scales: torch.Tensor | None) -> None:
        del scales
        recorded["packed"] = values.detach().clone()

    return SimpleNamespace(
        block_table=torch.tensor([[0]], dtype=torch.int32),
        actual_seq_kv=torch.tensor([4], dtype=torch.int64),
        kpool_tail=None,
        kpool_tail_read_indices=None,
        kpool_tail_write_indices=None,
        slot_mapping=torch.arange(4, dtype=torch.int64),
        # Dummy index-cache tensor: only its presence matters to select_qli
        # (the recording update_index_cache above ignores its content).
        index_cache=torch.zeros(1, 8, 1, 1),
        index_cache_scale=None,
        cp_context=cp_context,
        kpool_query_lens=(),
        kpool_query_lens_device=None,
        kpool_cache_triton_compatible=False,
        update_index_cache=update_index_cache,
    )


def test_dsa_cp_indexer_writes_global_key_and_selects_global_query_rows(monkeypatch) -> None:
    """The real ``Glm5NextIndexer.select_qli`` receives global rows under CP
    (the owning MLA layer merged them), so it must:
    (a) write the index cache in **global** token order with no extra gather,
    (b) run the top-k selection on those same global rows, and
    (c) return a global-layout top-k for the backend to partition.
    """
    indexer = _make_simple_indexer()
    cp_context = _cp2_context(0)
    recorded: dict[str, torch.Tensor] = {}
    ctx = _index_context_recording(cp_context, recorded)
    backend = MagicMock()
    backend.gather_index_history.return_value = torch.zeros(1, 4, 1)
    select_topk = MagicMock(return_value=torch.zeros(1, 4, 1, dtype=torch.long))
    indexer.select_topk = select_topk

    topk = indexer.select_qli(
        hidden_states=torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]),
        qr=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        positions=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        attention_mask=torch.tensor([[True, True, True, True]]),
        ctx=ctx,
        layer=None,
        backend=backend,
    )

    # (a) the global key side reaches the cache verbatim.
    torch.testing.assert_close(recorded["packed"].reshape(-1), torch.tensor([1.0, 2.0, 3.0, 4.0]))
    # (b) the query side is the same global rows.
    select_args = select_topk.call_args.args
    torch.testing.assert_close(select_args[1].reshape(-1), torch.tensor([1.0, 2.0, 3.0, 4.0]))
    torch.testing.assert_close(
        select_topk.call_args.kwargs["query_positions"].reshape(-1),
        torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    )
    # (c) the returned top-k is in the global [T_global] layout.
    assert topk.shape == (4, 1, 1)
    assert topk.dtype == torch.int32


def test_dsa_indexer_packs_global_rows_per_request_for_varlen_batch(monkeypatch) -> None:
    """The per-request query packing (``is_varlen`` branch of ``select_qli``)
    maps each request's **global** query rows to the right padded window slot,
    masks the pad rows, and gathers the surviving top-k rows back to the flat
    global order.

    Under CP the model has merged to global rows, so these are the original
    request boundaries (7 global tokens split 5/2). A request-blind layout --
    for example assuming every row belongs to the first request -- misplaces
    both the inputs and the returned top-k.
    """
    indexer = _make_simple_indexer()
    recorded: dict[str, torch.Tensor] = {}
    ctx = _index_context_recording(None, recorded)
    ctx.slot_mapping = torch.arange(7, dtype=torch.int64)
    ctx.actual_seq_kv = torch.tensor([5, 2], dtype=torch.int64)
    backend = MagicMock()
    backend.gather_index_history.return_value = torch.zeros(2, 5, 1)
    select_topk = MagicMock(return_value=torch.arange(10, dtype=torch.long).view(2, 5, 1))
    indexer.select_topk = select_topk

    hidden = torch.arange(10.0, 17.0).view(1, 7, 1)
    with patch.object(
        glm5_next,
        "get_forward_context",
        return_value=SimpleNamespace(metadata=SimpleNamespace(q_cu_seq_lens=torch.tensor([0, 5, 7]))),
    ):
        topk = indexer.select_qli(
            hidden_states=hidden,
            qr=hidden.reshape(7, 1),
            positions=torch.tensor([[0, 1, 2, 3, 4, 0, 1]], dtype=torch.int32),
            attention_mask=torch.ones(1, 7, dtype=torch.bool),
            ctx=ctx,
            layer=None,
            backend=backend,
        )

    # Padded window [num_seqs=2, max_q=5]: request 0 -> rows 10..14, request 1 ->
    # rows 15/16 with the last three slots masked (its real run is 2 rows).
    torch.testing.assert_close(
        select_topk.call_args.args[0],
        torch.tensor(
            [
                [[10.0], [11.0], [12.0], [13.0], [14.0]],
                [[15.0], [16.0], [16.0], [16.0], [16.0]],
            ]
        ),
    )
    torch.testing.assert_close(select_topk.call_args.args[1], select_topk.call_args.args[0])
    assert torch.equal(
        select_topk.call_args.args[2],
        torch.tensor([[True, True, True, True, True], [True, True, False, False, False]]),
    )
    torch.testing.assert_close(
        select_topk.call_args.kwargs["query_positions"],
        torch.tensor([[0, 1, 2, 3, 4], [0, 1, 1, 1, 1]], dtype=torch.int32),
    )
    # The seven real top-k rows survive the packed window and land back in the
    # flat global order.
    assert topk.shape == (7, 1, 1)
    assert topk.dtype == torch.int32
    assert torch.equal(topk, torch.arange(7, dtype=torch.int32).view(7, 1, 1))


# ---------------------------------------------------------------------------
# B1 (implement.md): GLM-5-Next model-level CP + aux-hidden-capture restore.
#
# Mirrors DSV4's
# ``test_layer_capture_restores_global_rows_with_context_parallelism``
# (test_deepseek_v4_model.py). ``Glm5NextModel.forward`` packs the exit hidden
# and the auxiliary capture into a single ``cp_merge_rows`` (one CP
# all-gather), so this section exercises the fused wiring end-to-end (real
# ``AuxHiddenCapture``/``cp_merge_rows``, mocked only at the
# ``distributed.all_gather`` collective): the fused result must stay
# element-wise equal to the former two-merge path, including the empty-local
# and capture-disabled boundaries.
# ---------------------------------------------------------------------------


def _make_model_with_capture(
    events: list[str], layers_to_capture: tuple[int, ...], hidden_size: int = 1
) -> tuple[glm5_next.Glm5NextModel, list[_DecoderLayer]]:
    """Like ``_make_model``, but wires ``aux_hidden_capture`` the way the real
    ``Glm5NextModel.__init__`` does (mean-collapse over the mHC streams via
    ``hc_head``) instead of the empty, capture-disabled default."""
    model, layers = _make_model(events)
    model.cfg.hidden_size = hidden_size
    model.cfg.layers_to_capture = layers_to_capture
    hc_head = model.hc_head
    model.aux_hidden_capture = AuxHiddenCapture(
        layers_to_capture,
        transform=lambda streams: hc_head(streams).reshape(-1, hidden_size),
    )
    return model, layers


def test_layer_capture_restores_global_rows_with_context_parallelism(monkeypatch) -> None:
    """No-padding case: 4 global tokens, zigzag cp_size=2, rank 0's local
    shard (global tokens 0 and 3; rank 1 owns 1 and 2 -- see ``_cp2_context``).

    Each mock ``_DecoderLayer`` adds ``layer_id + 1`` to its input, so after
    both layers a global token's final hidden value is ``value + 3`` and its
    captured (layer-0-only) auxiliary value is ``value + 1``. The model exit
    packs ``h`` and ``aux_hidden_buffer`` into a single all-gather, so
    ``h_gathered``/``aux_gathered`` below are supplied as one fused rank-major
    concatenation (rank 0's two local rows, then rank 1's two local rows, with
    the two tensors side by side on the feature axis); the test asserts the
    local shard fed to that collective is ``[h_local | aux_local]`` and that
    ``cp_merge_rows``' ``restore_index`` reassembles both parts back into
    original global token order [10, 20, 30, 40].
    """
    events: list[str] = []
    model, layers = _make_model_with_capture(events, layers_to_capture=(0,))
    del layers
    cp_context = _cp2_context(0)

    h_gathered = torch.tensor([[13.0], [43.0], [23.0], [33.0]])
    aux_gathered = torch.tensor([[11.0], [41.0], [21.0], [31.0]])
    fused_gathered = torch.cat([h_gathered, aux_gathered], dim=-1)
    gather = MagicMock(return_value=fused_gathered)
    monkeypatch.setattr(glm5_next.distributed, "all_gather", gather, raising=False)

    with patch.object(glm5_next, "get_forward_context", return_value=SimpleNamespace(cp_context=cp_context)):
        output = model(
            torch.tensor([10, 20, 30, 40]),
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

    assert isinstance(output, tuple)
    hidden, aux_hidden = output
    torch.testing.assert_close(hidden, torch.tensor([[13.0], [23.0], [33.0], [43.0]]))
    torch.testing.assert_close(aux_hidden, torch.tensor([[11.0], [21.0], [31.0], [41.0]]))
    # One collective instead of two, carrying rank 0's local rows (globals 0
    # and 3) with h and aux packed on the feature axis.
    assert gather.call_count == 1
    torch.testing.assert_close(gather.call_args_list[0].args[0], torch.tensor([[13.0, 11.0], [43.0, 41.0]]))


def test_layer_capture_restores_global_rows_with_context_parallelism_padding_row(monkeypatch) -> None:
    """Padding-row case (mirrors DSV4's own parametrization, which covers a
    token count that does not divide evenly across CP ranks): 3 global
    tokens under zigzag cp_size=2 leave rank 0 with one real row and one
    padding row (``_cp2_padded_context``). The padding row's (garbage) value
    must never leak into ``restore_index``'s output -- ``restore_index``
    (``[0, 2, 3]``) never references gathered row 1, the padding slot.
    """
    events: list[str] = []
    model, layers = _make_model_with_capture(events, layers_to_capture=(0,))
    del layers
    cp_context = _cp2_padded_context()

    # Rank 0's local padding row (gathered index 1) carries a deterministic
    # but otherwise-irrelevant value; it is never selected by restore_index.
    h_gathered = torch.tensor([[13.0], [3.0], [23.0], [33.0]])
    aux_gathered = torch.tensor([[11.0], [1.0], [21.0], [31.0]])
    fused_gathered = torch.cat([h_gathered, aux_gathered], dim=-1)
    gather = MagicMock(return_value=fused_gathered)
    monkeypatch.setattr(glm5_next.distributed, "all_gather", gather, raising=False)

    with patch.object(glm5_next, "get_forward_context", return_value=SimpleNamespace(cp_context=cp_context)):
        output = model(
            torch.tensor([10, 20, 30]),
            torch.tensor([0, 1, 2], dtype=torch.int32),
        )

    assert isinstance(output, tuple)
    hidden, aux_hidden = output
    torch.testing.assert_close(hidden, torch.tensor([[13.0], [23.0], [33.0]]))
    torch.testing.assert_close(aux_hidden, torch.tensor([[11.0], [21.0], [31.0]]))
    assert gather.call_count == 1
    torch.testing.assert_close(gather.call_args_list[0].args[0], torch.tensor([[13.0, 11.0], [3.0, 1.0]]))


def _cp2_empty_local_context() -> SimpleNamespace:
    """Zigzag cp_size=2 plan with no local rows at all (``T_local == 0``).

    ``total_local`` is identical on every CP rank (the sharding plan's
    contract), so an empty local shard is a group-wide state: every rank has
    to enter the exit all-gather even then, or the whole group deadlocks.
    """
    empty_indices = torch.empty(0, dtype=torch.int64)
    return SimpleNamespace(
        cp_size=2,
        cp_rank=0,
        total_local=0,
        shard_index=empty_indices,
        shard_gather_index=empty_indices,
        shard_valid_mask=torch.empty(0, dtype=torch.bool),
        restore_index=empty_indices,
        query_index=empty_indices,
        q_cu_seqlens=[],
        q_cu_seqlens_tensor=torch.empty(0, dtype=torch.int32),
        segment_seq_indices=empty_indices,
        segment_kv_seq_lens=[],
        segment_kv_seq_lens_tensor=torch.empty(0, dtype=torch.int32),
    )


def _patch_recording_cp_all_gather(monkeypatch: pytest.MonkeyPatch, peer_scale: float = 10.0) -> list[torch.Tensor]:
    """Patch ``distributed.all_gather`` with a recording, deterministic 2-rank
    collective and return the gathered inputs in call order.

    The peer rank's shard is ``shard * peer_scale``: arbitrary but fully
    deterministic and shape-compatible, so every merged element is specified
    by this rank's local shard alone. Only the axis semantics of the
    collective are under test; the peer payload just has to be distinct.
    """
    gathered_inputs: list[torch.Tensor] = []

    def all_gather(value: torch.Tensor, dim: int, world_size: int, group_name: str) -> torch.Tensor:
        assert dim == 0
        assert world_size == 2
        assert group_name == "cp"
        gathered_inputs.append(value)
        return torch.cat([value, value * peer_scale], dim=dim)

    monkeypatch.setattr(glm5_next.distributed, "all_gather", all_gather, raising=False)
    return gathered_inputs


def test_model_exit_fuses_cp_merges_into_one_all_gather(monkeypatch) -> None:
    """The model exit issues one all-gather, element-wise equal to merging
    ``h`` and the auxiliary capture separately.

    ``h`` is ``[T_local, D]`` and the capture is ``[T_local, D * k]`` over the
    same local rows (``k = 2`` here). The reference is the pre-fusion code
    path replayed through the real ``cp_merge_rows``; because the packing
    happens on the feature axis and the merge on the row axis, both the
    intermediate packed shard and the split result have to match exactly.
    """
    hidden_size = 2
    captured_layers = (0, 1)  # k = 2 -> auxiliary width 4
    events: list[str] = []
    model, _ = _make_model_with_capture(events, layers_to_capture=captured_layers, hidden_size=hidden_size)
    cp_context = _cp2_context(0)
    gathered_inputs = _patch_recording_cp_all_gather(monkeypatch)
    # Global tokens 0..3 carry [value, value + 0.5]. The capture is a mean over
    # the identical mHC streams; the two mock layers add +1 and +2, so rank 0's
    # local rows (globals 0 and 3) hold ``embed + 3`` for the exit hidden and
    # ``[embed + 1 | embed + 3]`` for the two captured slots.
    embeds = torch.tensor([[[0.0, 0.5], [1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]])
    local_h = torch.tensor([[3.0, 3.5], [6.0, 6.5]])
    local_aux = torch.tensor([[1.0, 1.5, 3.0, 3.5], [4.0, 4.5, 6.0, 6.5]])

    # "Two merges separately": the previous exit path, through the real merge.
    expected_h = glm5_next.cp_merge_rows(local_h, cp_context)
    expected_aux = glm5_next.cp_merge_rows(local_aux, cp_context)

    model._inputs_embeds = embeds
    with patch.object(glm5_next, "get_forward_context", return_value=SimpleNamespace(cp_context=cp_context)):
        output = model(
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

    assert isinstance(output, tuple)
    hidden, aux_hidden = output
    assert torch.equal(hidden, expected_h)
    assert torch.equal(aux_hidden, expected_aux)
    assert hidden.shape == (4, hidden_size)
    assert aux_hidden.shape == (4, hidden_size * len(captured_layers))
    # Two reference merges, then the single fused exit merge.
    assert len(gathered_inputs) == 3
    reference_h, reference_aux, fused_input = gathered_inputs
    assert torch.equal(reference_h, local_h)
    assert torch.equal(reference_aux, local_aux)
    assert fused_input.shape == (2, hidden_size * (1 + len(captured_layers)))
    assert torch.equal(fused_input, torch.cat([reference_h, reference_aux], dim=-1))


def test_model_exit_empty_local_rank_still_joins_the_all_gather(monkeypatch) -> None:
    """``T_local == 0`` must not turn into a skipped collective.

    The fused concat is ``[0, D]`` with ``[0, D * k]`` -> ``[0, D * (1 + k)]``;
    both ``all_gather`` and ``index_select`` handle the empty row axis, and the
    rank still enters the collective exactly once. Every rank sees the same
    empty shard, so an early return here would hang the CP group.
    """
    events: list[str] = []
    model, _ = _make_model_with_capture(events, layers_to_capture=(0,))
    cp_context = _cp2_empty_local_context()
    gathered_inputs = _patch_recording_cp_all_gather(monkeypatch)
    expected_h = glm5_next.cp_merge_rows(torch.empty(0, 1), cp_context)
    expected_aux = glm5_next.cp_merge_rows(torch.empty(0, 1), cp_context)

    with patch.object(glm5_next, "get_forward_context", return_value=SimpleNamespace(cp_context=cp_context)):
        output = model(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int32))

    assert isinstance(output, tuple)
    hidden, aux_hidden = output
    assert hidden.shape == (0, 1)
    assert aux_hidden.shape == (0, 1)
    assert torch.equal(hidden, expected_h)
    assert torch.equal(aux_hidden, expected_aux)
    # Two reference merges, then the exit merge the empty rank still joins.
    assert len(gathered_inputs) == 3
    assert gathered_inputs[2].shape == (0, 2)
    assert torch.equal(gathered_inputs[2], torch.cat([gathered_inputs[0], gathered_inputs[1]], dim=-1))


def test_model_exit_without_aux_capture_keeps_a_single_merge(monkeypatch) -> None:
    """No capture buffer -> the exit falls back to merging ``h`` alone."""
    events: list[str] = []
    model, _ = _make_model(events)  # AuxHiddenCapture(()) -> create_buffer returns None
    cp_context = _cp2_context(0)
    gathered_inputs = _patch_recording_cp_all_gather(monkeypatch)
    # Rank 0's local rows are globals 0 and 3: values [10, 40] plus 2 layers.
    expected = glm5_next.cp_merge_rows(torch.tensor([[13.0], [43.0]]), cp_context)

    with patch.object(glm5_next, "get_forward_context", return_value=SimpleNamespace(cp_context=cp_context)):
        output = model(
            torch.tensor([10, 20, 30, 40]),
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

    assert not isinstance(output, tuple)
    assert torch.equal(output, expected)
    assert len(gathered_inputs) == 2
    assert gathered_inputs[1].shape == (2, 1)  # h only: no fused width


def test_cp_one_preserves_full_rows_without_shard_or_merge(monkeypatch) -> None:
    events: list[str] = []
    model, layers = _make_model(events)
    # The CP package is stubbed on this host, so the collective only exists
    # once a test installs it; it must stay untouched for cp_size == 1.
    gather = MagicMock()
    monkeypatch.setattr(glm5_next.distributed, "all_gather", gather, raising=False)
    with (
        patch.object(glm5_next, "get_forward_context", return_value=SimpleNamespace(cp_context=None)),
        patch.object(glm5_next, "cp_shard_rows") as shard_rows,
        patch.object(glm5_next, "cp_shard_positions") as shard_positions,
        patch.object(glm5_next, "cp_merge_rows") as merge_rows,
    ):
        output = model(
            torch.tensor([10, 20, 30, 40]),
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

    shard_rows.assert_not_called()
    shard_positions.assert_not_called()
    merge_rows.assert_not_called()
    gather.assert_not_called()
    assert events == ["embedding", "layer_0", "layer_1", "norm"]
    torch.testing.assert_close(layers[0].positions, torch.tensor([[0, 1, 2, 3]], dtype=torch.int32))
    torch.testing.assert_close(output, torch.tensor([[13.0], [23.0], [33.0], [43.0]]))
