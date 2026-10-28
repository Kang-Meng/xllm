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

"""Explicitly lossy V4.1 prefill consumer compaction.

Keep the tested 140-row suffix in layers 21..39 after source layer 20 has
written the complete main/index caches. Missing consumer SWA dependencies
make this approximate. Only suffix outputs are valid; generation consumes
the final row. Decode, including ACL graph capture/replay, is unchanged.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch

from scripts.logger import logger
from xllm.python import distributed
from xllm.python.model_executor.forward_context import ForwardContext, forward_context, record_layer_event

if TYPE_CHECKING:
    from xllm.python.models.deepseek_v41 import CSA2LayerPlan, DeepseekV41Config, DeepseekV41Model

SWA_BOUNDED_FIRST_CONSUMER = 21
SWA_BOUNDED_QUERY_ROWS = 140


class _MetadataView:
    def __init__(self, original: Any, **overrides: Any) -> None:
        self._original = original
        self.__dict__.update(overrides)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


def validate_swa_bounded_replay(cfg: DeepseekV41Config, plan: CSA2LayerPlan) -> None:
    """Reject unsupported configurations before allocating model weights."""
    if not cfg.enable_swa_bounded_replay:
        return
    if (
        cfg.n_layers != 40
        or cfg.window_size != 128
        or cfg.candidate_source_layer_id != 20
        or not plan.mode(20).is_full
        or any(plan.mode(i).main_kv_source_layer != 20 for i in range(SWA_BOUNDED_FIRST_CONSUMER, cfg.n_layers))
        or any(i >= SWA_BOUNDED_FIRST_CONSUMER for i in cfg.engram_layer_ids)
    ):
        raise ValueError("SWA bounded replay requires the V4.1 Flash 40-layer/source-20/window-128 layout")
    logger.warning(
        "Lossy SWA bounded replay enabled: retain 140 query rows in layers 21..39; "
        "prefix hidden states are approximate; decode uses full computation"
    )


def should_run_swa_bounded_replay(enabled: bool, context: ForwardContext) -> bool:
    """Choose the same collective path on active and dummy DP replicas."""
    if not enabled or context.acl_graph is not None or context.execution_state is not None:
        return False
    metadata = context.metadata
    if max(metadata.dp_execution_token_counts, default=0) <= SWA_BOUNDED_QUERY_ROWS:
        return False
    # Empty DP replicas must enter too: shrinking only the active rank gives
    # inconsistent MoE gather sizes.
    return True


def run_swa_bounded_consumers(
    model: DeepseekV41Model,
    context: ForwardContext,
    hidden: torch.Tensor,
    pre_mix: torch.Tensor,
    positions: torch.Tensor,
    input_ids: torch.Tensor,
    aux_hidden_buffer: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Execute consumer layers on a suffix, preserving full source caches."""
    metadata = context.metadata
    dsa = metadata.dsa_metadata
    q_lens = dsa.seq_lens_q.tolist()  # Host metadata: no device synchronization.
    original_rows = hidden.shape[0]
    retained_lens = [min(int(n), SWA_BOUNDED_QUERY_ROWS) for n in q_lens]
    drops = [int(n) - keep for n, keep in zip(q_lens, retained_lens)]
    indices = []
    offset = 0
    for length, drop in zip(q_lens, drops):
        indices.extend(range(offset + drop, offset + int(length)))
        offset += int(length)
    # Preserve the framework's dummy row on an empty DP replica.
    if not indices:
        indices = [0]
    rows = torch.tensor(indices, dtype=torch.int64, device=hidden.device)
    hidden = hidden.index_select(0, rows)
    pre_mix = pre_mix.index_select(0, rows)
    positions = positions.index_select(0, rows)
    input_ids = input_ids.index_select(0, rows)
    compact_dsa = replace(
        dsa,
        seq_lens_q=torch.clamp(dsa.seq_lens_q, max=SWA_BOUNDED_QUERY_ROWS),
        max_query_len=min(dsa.max_query_len, SWA_BOUNDED_QUERY_ROWS),
        input_positions=positions,
        # SWA maps have one slot per input row. Compressed source planes
        # can have fewer rows (e.g. ratio 2); consumers never rewrite them.
        slot_mappings=[
            [
                slots.index_select(0, rows) if slots is not None and slots.size(0) == original_rows else slots
                for slots in group
            ]
            for group in dsa.slot_mappings
        ],
    )
    shared = context.attention_backend._shared_runtime()
    if shared.topk_idxs is not None:
        original = shared.topk_idxs
        compact = original.new_full((len(q_lens), max(retained_lens, default=0), original.size(-1)), -1)
        for seq, (drop, keep) in enumerate(zip(drops, retained_lens)):
            picks = original[seq, drop : drop + keep]
            compact[seq, :keep] = torch.where(picks >= 0, picks - drop, picks)
        shared.topk_idxs = compact
    if shared.candidate_blocks is not None:
        shared.candidate_blocks = shared.candidate_blocks.index_select(0, rows)
    if shared.candidates is not None:
        shared.candidates = [
            None if picks is None else picks[drop:].contiguous() for picks, drop in zip(shared.candidates, drops)
        ]
    counts = (hidden.size(0),)
    if len(metadata.dp_execution_token_counts) > 1:
        # Each replica can own several requests. Exchange the resulting row
        # counts instead of assuming every replica retains exactly 140 rows.
        # The .cpu() below synchronizes the collective so subsequent MoE
        # gathers can use the actual per-replica counts on the host.
        local_count = torch.tensor([hidden.size(0)], dtype=torch.int64, device=hidden.device)
        counts = tuple(
            int(n)
            for n in distributed.dp_all_gather(
                local_count, [1] * len(metadata.dp_execution_token_counts), model.cfg.dp_rank
            )
            .cpu()
            .tolist()
        )
    view = _MetadataView(
        metadata,
        dsa_metadata=compact_dsa,
        swa_bounded_replay_active=True,
        swa_bounded_replay_dropped_rows=tuple(drops),
        dp_execution_token_counts=counts,
        raw_dp_execution_token_counts=tuple(
            count if int(raw) else 0 for count, raw in zip(counts, metadata.raw_dp_execution_token_counts)
        ),
    )
    compact_context = replace(context, metadata=view, layer_shared_cache={})
    captured = None
    with forward_context(compact_context):
        for layer_id in range(SWA_BOUNDED_FIRST_CONSUMER, len(model.layers)):
            # Same capture point as the eager loop: a V4.1 target layer id names
            # the residual stream entering that layer, so capture before it runs.
            if aux_hidden_buffer is not None and model.aux_hidden_capture.should_capture(layer_id):
                if captured is None:
                    captured = hidden.new_zeros((original_rows, hidden.size(-1)))
                captured.index_copy_(0, rows, hidden.mean(dim=1))
                model.aux_hidden_capture.capture_layer(layer_id, captured, None, aux_hidden_buffer)
            hidden, pre_mix = model.layers[layer_id](
                hidden,
                positions,
                model._layer_rope_cache(model.csa2_plan.mode(layer_id)),
                input_ids,
                pre_mix,
                None,
                None,
            )
            record_layer_event(layer_id)
        result = model.norm(model.layers[-1].hc.hc_pre(hidden, pre_mix))
    # Preserve the executor's output indexing. Prefix logits are unsupported
    # for this opt-in mode; the final generation row remains in its old slot.
    full_result = result.new_zeros((original_rows, result.size(-1)))
    full_result.index_copy_(0, rows, result)
    return model.aux_hidden_capture.finalize(full_result, aux_hidden_buffer)
