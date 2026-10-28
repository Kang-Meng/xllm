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

"""Opt-in, collective layout and cache contracts of approximate SWA replay."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from xllm.python.model_executor.forward_context import ForwardContext, get_forward_context
from xllm.python.models.deepseek_v41 import CSA2LayerPlan, DeepseekV41Config
from xllm.python.models.swa_bounded_replay import (
    run_swa_bounded_consumers,
    should_run_swa_bounded_replay,
    validate_swa_bounded_replay,
)


def _config(**overrides: object) -> DeepseekV41Config:
    return DeepseekV41Config.from_dict(
        dict(
            n_layers=40,
            compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
            kv_source_layer_ids=[2, 8, 14, 20],
            index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
            candidate_source_layer_id=20,
            **overrides,
        )
    )


def _validate(cfg: DeepseekV41Config) -> None:
    plan = CSA2LayerPlan.build(cfg.n_layers, cfg.compress_ratios, cfg.kv_source_layer_ids, cfg.index_source_layer_ids)
    validate_swa_bounded_replay(cfg, plan)


def test_opt_in_and_unsupported_configurations() -> None:
    cfg = _config()
    assert not cfg.enable_swa_bounded_replay
    _validate(replace(cfg, cp_size=2))
    cfg = replace(cfg, enable_swa_bounded_replay=True)
    _validate(cfg)
    _validate(replace(cfg, cp_size=2, layers_to_capture=[37, 38, 39]))
    for changes in (
        dict(window_size=256),
        dict(candidate_source_layer_id=14),
        dict(engram_layer_ids=[21]),
    ):
        with pytest.raises(ValueError, match="SWA bounded replay"):
            _validate(replace(cfg, **changes))


def test_selection_is_collective_symmetric_and_excludes_graph_decode() -> None:
    metadata = SimpleNamespace(dp_execution_token_counts=(1024, 1, 1, 1))
    context = ForwardContext(None, torch.device("cpu"), metadata, [])
    assert not should_run_swa_bounded_replay(False, context)
    assert should_run_swa_bounded_replay(True, context)
    assert not should_run_swa_bounded_replay(True, replace(context, execution_state=object()))
    metadata.dp_execution_token_counts = (1, 1, 1, 1)
    assert not should_run_swa_bounded_replay(True, context)


@dataclass
class _Dsa:
    seq_lens_q: torch.Tensor
    max_query_len: int
    input_positions: torch.Tensor
    slot_mappings: list[list[torch.Tensor | None]]
    seq_lens: torch.Tensor


@pytest.mark.parametrize("rows,logical_rows", [(1024, 1024), (1, 0)])
def test_compaction_preserves_source_cache_and_final_output_slot(
    rows: int, logical_rows: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xllm.python.models import swa_bounded_replay

    monkeypatch.setattr(
        swa_bounded_replay.distributed,
        "dp_all_gather",
        lambda value, counts, rank: torch.tensor([140, 1]),
        raising=False,
    )
    positions = torch.arange(rows)
    compressed_slots = torch.arange(max(1, rows // 2))
    dsa = _Dsa(
        torch.tensor([logical_rows]), rows, positions, [[positions, compressed_slots, None]], torch.tensor([4096])
    )
    metadata = SimpleNamespace(
        dsa_metadata=dsa,
        dp_execution_token_counts=(1024, 1),
        raw_dp_execution_token_counts=(1024, 0),
    )
    source_cache = torch.randn(9, 16, 8)
    # The compressed cache starts after [window=128 ; current chunk].
    picks = torch.full((1, rows, 2), 128 + rows + 7, dtype=torch.int32)
    picks[:, :, 1] = -1
    shared = SimpleNamespace(
        topk_idxs=picks,
        candidate_blocks=torch.arange(rows),
        candidates=[torch.arange(rows), None],
        key_cache=source_cache,
    )
    backend = SimpleNamespace(_shared_runtime=lambda: shared)
    context = ForwardContext(backend, torch.device("cpu"), metadata, [])
    seen: list[int] = []

    def layer(
        hidden: torch.Tensor,
        pos: torch.Tensor,
        rope: object,
        ids: torch.Tensor,
        pre_mix: torch.Tensor,
        hashes: object,
        mask: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active = get_forward_context()
        assert active.metadata.swa_bounded_replay_active
        assert active.metadata.dp_execution_token_counts == (140, 1)
        assert active.metadata.raw_dp_execution_token_counts == (140, 0)
        assert active.metadata.dsa_metadata.seq_lens is dsa.seq_lens
        compact_slots = active.metadata.dsa_metadata.slot_mappings[0]
        torch.testing.assert_close(compact_slots[0], positions[-min(rows, 140) :])
        if rows > 1:
            assert compact_slots[1] is compressed_slots
        assert compact_slots[2] is None
        assert active.metadata.dsa_metadata.seq_lens_q.item() == min(logical_rows, 140)
        assert hidden.size(0) == min(rows, 140)
        assert pos[-1].item() == rows - 1
        assert shared.key_cache is source_cache
        assert torch.all(shared.topk_idxs[:, :, 0] == 128 + min(rows, 140) + 7)
        assert torch.all(shared.topk_idxs[:, :, 1] == -1)
        seen.append(hidden.size(0))
        return hidden + 1, pre_mix

    last = SimpleNamespace(hc=SimpleNamespace(hc_pre=lambda h, mix: h[:, 0]))

    class _LastLayer:
        hc = last.hc

        def __call__(self, *args: object) -> tuple[torch.Tensor, torch.Tensor]:
            return layer(*args)

    model = SimpleNamespace(
        cfg=SimpleNamespace(dp_rank=0),
        aux_hidden_capture=SimpleNamespace(finalize=lambda hidden, buffer: hidden),
        layers=[None] * 21 + [layer] * 18 + [_LastLayer()],
        csa2_plan=SimpleNamespace(mode=lambda index: index),
        _layer_rope_cache=lambda mode: None,
        norm=lambda hidden: hidden,
    )
    hidden = torch.arange(rows, dtype=torch.float32).view(rows, 1, 1)
    output = run_swa_bounded_consumers(model, context, hidden, torch.ones(rows, 1), positions, positions)
    assert len(seen) == 19
    assert output.shape == (rows, 1)
    assert output[-1].item() == rows - 1 + 19
    assert torch.count_nonzero(output[: max(0, rows - 140)]) == 0
    assert dsa.seq_lens_q.item() == logical_rows
    assert metadata.dp_execution_token_counts == (1024, 1)


@pytest.mark.parametrize("persistent", [False, True])
def test_multiple_requests_keep_each_final_row_and_aux_capture(persistent: bool) -> None:
    from xllm.python.models.aux_hidden_capture import AuxHiddenCapture

    lengths = [200, 3, 180]
    total = sum(lengths)
    positions = torch.arange(total)
    dsa = _Dsa(torch.tensor(lengths), 200, positions, [[positions]], torch.tensor([200, 3, 180]))
    metadata = SimpleNamespace(
        dsa_metadata=dsa, dp_execution_token_counts=(total,), raw_dp_execution_token_counts=(total,)
    )
    topk = torch.full((3, 200, 1), -1, dtype=torch.int32)
    for seq, length in enumerate(lengths):
        topk[seq, :length] = 128 + length + 9
    shared = SimpleNamespace(topk_idxs=topk, candidate_blocks=positions, candidates=[torch.arange(n) for n in lengths])
    context = ForwardContext(SimpleNamespace(_shared_runtime=lambda: shared), torch.device("cpu"), metadata, [])
    capture = AuxHiddenCapture((37, 38, 39), persistent_buffer=persistent)
    hidden = positions.float().view(-1, 1, 1)
    aux_buffer = capture.create_buffer(hidden.mean(1))

    class _Layer:
        hc = SimpleNamespace(hc_pre=lambda value, mix: value[:, 0])

        def __call__(self, *args: object) -> tuple[torch.Tensor, torch.Tensor]:
            active = get_forward_context().metadata
            assert active.dp_execution_token_counts == (283,)
            assert active.dsa_metadata.seq_lens_q.tolist() == [140, 3, 140]
            for seq, keep in enumerate([140, 3, 140]):
                assert torch.all(shared.topk_idxs[seq, :keep] == 128 + keep + 9)
            return args[0] + 1, args[4]

    model = SimpleNamespace(
        layers=[None] * 21 + [_Layer() for _ in range(19)],
        csa2_plan=SimpleNamespace(mode=lambda layer_id: layer_id),
        _layer_rope_cache=lambda mode: None,
        norm=lambda value: value,
        aux_hidden_capture=capture,
    )
    result = run_swa_bounded_consumers(model, context, hidden, torch.ones(total, 1), positions, positions, aux_buffer)
    output, aux = (result, capture.buffer) if persistent else result
    kept = torch.cat((torch.arange(60, 200), torch.arange(200, 203), torch.arange(243, 383)))
    torch.testing.assert_close(output[kept, 0], kept.float() + 19, rtol=0, atol=0)
    # Capture takes the stream entering each layer, so layers 37/38/39 see 16/17/18
    # increments while the output has all 19.
    expected_aux = kept.float().unsqueeze(-1) + torch.tensor([16, 17, 18])
    torch.testing.assert_close(aux[kept], expected_aux, rtol=0, atol=0)
    for dropped in (slice(0, 60), slice(203, 243)):
        assert torch.count_nonzero(output[dropped]) == 0
        assert torch.count_nonzero(aux[dropped]) == 0


@pytest.mark.parametrize("compact,drop", [(False, 0), (True, 884), (True, 0)])
def test_smla_visibility_masks_only_the_missing_consumer_prefix(
    compact: bool, drop: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xllm.python.attention.csa2_attention import Csa2AttentionBackend
    from xllm.python.model_executor.forward_context import forward_context

    backend = object.__new__(Csa2AttentionBackend)
    backend.window_size = 128
    backend.scale = 0.5
    monkeypatch.setattr(torch.ops.xllm_ops, "smla_va_metadata", lambda *args, **kwargs: None, raising=False)
    seen: list[torch.Tensor] = []

    def attention(query: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, None]:
        seen.append(kwargs["ori_topk_length"])
        return query, None

    monkeypatch.setattr(torch.ops.xllm_ops, "smla_va", attention, raising=False)
    context = ForwardContext(
        backend,
        torch.device("cpu"),
        SimpleNamespace(swa_bounded_replay_active=compact, swa_bounded_replay_dropped_rows=(drop,)),
        [],
    )
    layer = SimpleNamespace(mode=SimpleNamespace(compress_ratio=0), attn_sink=None)
    with forward_context(context):
        backend._smla_attention(
            torch.zeros(140, 2, 8),
            torch.zeros(268, 8),
            layer,
            torch.tensor([1024]),
            torch.tensor([1024]),
            140,
            140,
            0,
            None,
            SimpleNamespace(),
            torch.zeros(2),
        )
    expected = torch.arange(1, 141, dtype=torch.int32).clamp(max=128) if drop else torch.full((140,), 128)
    torch.testing.assert_close(seen[0].flatten(), expected, check_dtype=False)
