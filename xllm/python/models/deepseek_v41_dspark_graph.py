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

"""Model-owned adaptation of the V4.1 block drafter to ACL graph execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from scripts.logger import logger
from xllm.python.attention.backend import AttentionMetadata
from xllm.python.attention.dsa_metadata import DSA_CACHE_SLIDING_WINDOW
from xllm.python.model_executor.executor import DecodeGraphAdapter
from xllm.python.model_executor.runners.base import DraftProposalOutput
from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner

if TYPE_CHECKING:
    from xllm.python.models.dspark import DSparkForCausalLMBase


class DSparkGreedyGraphModel(nn.Module):
    """One graph-capturable draft body + LM head + sequential Markov chain."""

    def __init__(
        self,
        owner: DSparkForCausalLMBase,
        query_width: int,
    ) -> None:
        super().__init__()
        # The executor is constructed before weight loading may create or
        # share ``lm_head``. Keep the top-level owner so forward resolves the
        # final modules after that lifecycle completes.
        self.owner = owner
        self.query_width = int(query_width)
        if self.query_width <= 0:
            raise ValueError("DSpark graph query width must be positive")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # The eager path lets the C++ bridge share the target's lm_head when
        # the checkpoint ships no draft head. A captured proposal graph cannot
        # borrow it (the whole body -> LM head -> Markov chain is one graph), so
        # an lm_head-less checkpoint must fail here with a diagnosable message
        # instead of raising "NoneType is not callable" at the first draft step.
        if self.owner.lm_head is None:
            raise RuntimeError(
                "DSpark full graph requires the draft's own lm_head: the "
                "checkpoint must ship mtp.<last>.head.weight or head.weight. "
                "Disable --enable_dspark_draft_graph to use the shared target "
                "head on the eager path."
            )
        hidden = self.owner.model(input_ids, positions)
        rows = hidden.size(0)
        if rows % self.query_width:
            raise RuntimeError(
                f"DSpark graph rows must contain complete query blocks: rows={rows}, width={self.query_width}"
            )
        logits = self.owner.lm_head(hidden)
        base_logits = logits.view(-1, self.query_width, logits.size(-1))
        previous = input_ids.view(-1, self.query_width)[:, 0].to(torch.long)
        proposal_steps = []
        for token_idx in range(self.query_width):
            step_logits = base_logits[:, token_idx] + self.owner.markov_head.bias(previous)
            previous = torch.argmax(step_logits, dim=-1)
            proposal_steps.append(previous)
        return torch.stack(proposal_steps, dim=1).reshape(-1)


class _DraftCacheMetadata:
    """A draft-local view of the target's exported cache manager tables."""

    def __init__(self, base: AttentionMetadata, tables: tuple[torch.Tensor, ...]) -> None:
        self._base = base
        self.multi_block_tables = tables

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


class DeepseekV41DSparkGraphRunner(DecodeAclGraphRunner):
    """Reuse ACL graph storage and replay with complete draft query blocks."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The shared executor resolves the final width from both its explicit
        # argument and the adapter. Proposal rows already contain whole blocks.
        if self.num_decoding_tokens != 1:
            raise ValueError("V4.1 DSpark proposal graph requires num_decoding_tokens=1")

    def _normalize_dsa_metadata(self, metadata: AttentionMetadata) -> AttentionMetadata:
        group_infos = getattr(self.attention_backend, "group_infos", None)
        if group_infos is not None:
            if len(group_infos) != 1 or group_infos[0].cache_type != DSA_CACHE_SLIDING_WINDOW:
                raise ValueError("V4.1 DSpark draft requires exactly one SWA cache group")
            tables = tuple(getattr(metadata, "multi_block_tables", ()) or ())
            if tables:
                # C++ kMultiBlockExportOrder exports SWA first, matching group
                # zero in build_cache_specs_v41. The pure-SWA draft consumes
                # that leading table through a view without changing the
                # target's complete manager list.
                metadata = _DraftCacheMetadata(metadata, tables[:1])
        return super()._normalize_dsa_metadata(metadata)

    def _padded_token_rows(self, token_rows: int, admission_batch: int | None) -> int:
        if admission_batch is None:
            raise ValueError("draft decode batch does not contain complete query blocks")
        sequence_bucket = super()._padded_token_rows(admission_batch, admission_batch)
        return sequence_bucket * self._draft_query_width

    def _capture(self, entry: Any) -> None:
        super()._capture(entry)
        token_rows = entry.static_input_ids.numel()
        logger.info(
            f"Captured DSpark full ACL graph: token_rows={token_rows}, "
            f"sequence_bucket={token_rows // self._draft_query_width}, "
            f"query_width={self._draft_query_width}"
        )

    def execute(self, *args: Any, **kwargs: Any) -> DraftProposalOutput:
        token_ids = super().execute(*args, **kwargs)
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError("DSpark proposal graph must return a token ID tensor")
        return DraftProposalOutput(token_ids)


def create_dspark_decode_graph_adapter(
    model: DSparkForCausalLMBase,
    config: dict,
    graph_backend: str,
    *,
    is_spec_draft: bool = False,
) -> DecodeGraphAdapter | None:
    """Supply proposal graph execution without replacing the shared executor."""
    if not bool(config.get("draft_graph_enabled", False)):
        return None
    if not is_spec_draft:
        raise ValueError("DSpark proposal graph requires a draft executor")
    if graph_backend != "aclgraph":
        raise ValueError("DSpark proposal graph requires python_graph_backend=aclgraph")
    if int(config.get("cp_size", 1)) != 1:
        raise ValueError("DSpark block attention does not support context parallelism")
    width = int(config.get("draft_query_width", 1))
    if width <= 0:
        raise ValueError("DSpark graph query width must be positive")
    # The proposal input already contains the whole query block. Its row
    # capacity must not be multiplied by the target's verify width again.
    return DecodeGraphAdapter(
        model=DSparkGreedyGraphModel(model, width),
        runner_type=DeepseekV41DSparkGraphRunner,
        num_decoding_tokens=1,
    )
