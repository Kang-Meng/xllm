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

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from xllm.python.model_executor.forward_context import get_forward_context_or_none
from xllm.python.model_executor.model_output import DraftProposal, ModelOutput
from xllm.python.model_executor.sampling_metadata import (
    ModelSamplingMetadata,
    prepare_sampling_input,
    sample_logits,
)
from xllm.python.models.base import PyModelBase

if TYPE_CHECKING:
    from xllm.python.models.dspark import DSparkConfidenceHead, DSparkMarkovHead


class DSparkPipeline(PyModelBase):
    """Returns complete draft proposals from the backbone and output heads."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        lm_head: nn.Module | None,
        markov_head: DSparkMarkovHead,
        confidence_head: DSparkConfidenceHead | None,
        query_width: int,
        target_vocab_size: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.lm_head = lm_head
        self.markov_head = markov_head
        self.confidence_head = confidence_head
        self._query_width = query_width
        self.target_vocab_size = target_vocab_size
        self.register_buffer("hot_token_ids", None, persistent=False)

    @property
    def cfg(self) -> Any:
        return self.backbone.cfg

    @property
    def dtype(self) -> torch.dtype:
        return self.markov_head.markov_w1.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.markov_head.markov_w1.weight.device

    def get_token_embedding(self) -> nn.Module | None:
        return self.backbone.embed_tokens

    def share_weights_from(self, source: PyModelBase) -> None:
        if self.lm_head is None:
            self.lm_head = source.get_lm_head()
        if self.backbone.embed_tokens is None:
            self.backbone.embed_tokens = source.get_token_embedding()

    def confidence_probs(
        self,
        hidden: torch.Tensor,
        previous_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.confidence_head is None:
            raise RuntimeError("DSpark confidence head is not enabled")
        markov_embedding = self.markov_head.embed(previous_tokens) if previous_tokens is not None else None
        return self.confidence_head(hidden, markov_embedding)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embedding: torch.Tensor | None = None,
    ) -> ModelOutput:
        context = get_forward_context_or_none()
        sampling_metadata = context.execution_contexts.get(ModelSamplingMetadata) if context is not None else None
        if sampling_metadata is None or sampling_metadata.sampling is None:
            raise RuntimeError("DSpark pipeline requires sampling metadata to produce a draft proposal")

        if self._query_width <= 0 or sampling_metadata.num_tokens % self._query_width:
            raise ValueError("DSpark input must contain complete query blocks")

        sampling_input = sampling_metadata.sampling
        model_output = self.backbone(input_ids, positions, input_embedding)
        hidden = model_output[0] if isinstance(model_output, tuple) else model_output
        # The backbone returns this rank's query rows in input order, so no DP
        # remapping is needed. Compute every complete query block, including
        # graph padding; the runner trims padded proposals after graph replay.
        num_sequences = input_ids.numel() // self._query_width
        num_queries = num_sequences * self._query_width
        hidden = hidden[:num_queries]
        logits = self.compute_logits(hidden, None)
        base_logits = logits.reshape(num_sequences, self._query_width, -1)
        anchor_tokens = input_ids[:num_queries].reshape(num_sequences, self._query_width)[:, 0].long()
        prepared_sampling = prepare_sampling_input(sampling_input, logits)
        previous_tokens = anchor_tokens
        sampled_steps: list[torch.Tensor] = []
        probability_steps: list[torch.Tensor] = []

        for token_index in range(self._query_width):
            markov_bias = self.markov_head.bias(previous_tokens)
            step_logits = base_logits[:, token_index, :] + markov_bias
            sampled_tokens, probabilities = sample_logits(
                step_logits,
                prepared_sampling,
                tp_size=sampling_metadata.tp_size,
                # TODO: Move TP backend selection into the communication layer
                # instead of exposing native/Python TP compatibility in this pipeline.
                use_native_tp=sampling_metadata.use_native_tp,
            )
            if self.hot_token_ids is not None:
                sampled_tokens = self.hot_token_ids.index_select(0, sampled_tokens)
                if probabilities is not None:
                    full_probs = probabilities.new_zeros((num_sequences, self.target_vocab_size))
                    full_probs[:, self.hot_token_ids] = probabilities
                    probabilities = full_probs
            sampled_steps.append(sampled_tokens)
            if prepared_sampling.return_probs:
                if probabilities is None:
                    raise RuntimeError("Draft sampling did not return the required probabilities")
                probability_steps.append(probabilities)
            previous_tokens = sampled_tokens

        token_ids = torch.stack(sampled_steps, dim=1)
        confidence = None
        if prepared_sampling.return_confidence:
            previous_matrix = torch.cat((anchor_tokens[:, None], token_ids[:, :-1]), dim=1)
            confidence = self.confidence_probs(hidden.reshape(num_sequences, self._query_width, -1), previous_matrix)
        return ModelOutput(
            draft_proposal=DraftProposal(
                token_ids=token_ids,
                draft_probs=torch.stack(probability_steps, dim=1) if prepared_sampling.return_probs else None,
            ),
            draft_confidence_probs=confidence,
        )
