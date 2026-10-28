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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from xllm.python import distributed

if TYPE_CHECKING:
    from xllm.python.attention.backend import AttentionMetadata
    from xllm.python.model_executor.input_batch import InputBatch


class SamplingParameters(Protocol):
    all_greedy_sample: bool
    selected_token_idxes: torch.Tensor | None
    sample_idxes: torch.Tensor | None
    logprobs: bool
    max_top_logprobs: int
    use_beam_search: bool


class ModelSamplingInput(Protocol):
    parameters: SamplingParameters
    return_probs: bool
    return_confidence: bool
    is_unconstrained_greedy: bool


def _supports_graph_sampling(sampling: ModelSamplingInput | None) -> bool:
    return (
        sampling is not None
        and sampling.is_unconstrained_greedy
        and not sampling.return_probs
        and not sampling.return_confidence
    )


def prepare_sampling_input(
    sampling_input: ModelSamplingInput,
    logits: torch.Tensor,
) -> ModelSamplingInput:
    import xllm_runtime

    prepared_sampling = xllm_runtime.ModelSamplingInput(sampling_input)
    if prepared_sampling.is_unconstrained_greedy:
        return prepared_sampling

    parameters = xllm_runtime.sampling_parameters_to(sampling_input.parameters, logits)
    empty_indices = logits.new_empty((0,), dtype=torch.int32)
    parameters.selected_token_idxes = empty_indices
    parameters.sample_idxes = empty_indices
    parameters.logprobs = False
    parameters.max_top_logprobs = 0
    parameters.use_beam_search = False
    prepared_sampling.parameters = parameters
    return prepared_sampling


def sample_logits(
    logits: torch.Tensor,
    sampling_input: ModelSamplingInput,
    *,
    tp_size: int,
    use_native_tp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    import xllm_runtime

    tokens, probabilities = xllm_runtime.sample_logits(logits, sampling_input)
    if not sampling_input.parameters.all_greedy_sample and tp_size > 1:
        tokens = tokens.contiguous()
        if use_native_tp:
            # Reuse the native TP group and select TP rank 0's slice, matching
            # broadcast(src=0) without creating a second Python process group.
            gathered_tokens = xllm_runtime.tp_all_gather(tokens, 0)
            expected_shape = (tokens.shape[0] * tp_size, *tokens.shape[1:])
            if gathered_tokens.shape != expected_shape:
                raise RuntimeError(
                    f"Native sampling TP gather returned {tuple(gathered_tokens.shape)}, expected {expected_shape}"
                )
            tokens = gathered_tokens.narrow(0, 0, tokens.shape[0])
        else:
            distributed.broadcast_(tokens, src=0, group_name="tp")
    return tokens, probabilities


@dataclass
class ModelSamplingMetadata:
    sampling: ModelSamplingInput | None
    num_tokens: int
    tp_size: int = 1
    use_native_tp: bool = False


class ModelSamplingMetadataBuilder:
    metadata_type = ModelSamplingMetadata

    def __init__(self, tp_size: int = 1, *, use_native_tp: bool = False) -> None:
        if tp_size < 1:
            raise ValueError("Sampling TP size must be positive")
        self._tp_size = tp_size
        self._use_native_tp = use_native_tp

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> ModelSamplingMetadataBuilder:
        # V4/V4.1 drafts currently reuse C++ TP communicators rather than
        # registering a Python TP group, so sampling must use the native bridge.
        # TODO: Resolve this through a unified communication interface instead
        # of model-type checks and backend selection in sampling metadata.
        builder = cls(
            tp_size=int(config.get("tp_size", 1)),
            use_native_tp=config.get("model_type") in ("deepseek_v4_dspark", "deepseek_v41_dspark"),
        )
        tp_rank = int(config.get("tp_rank", 0))
        if not 0 <= tp_rank < builder._tp_size:
            raise ValueError(f"Invalid sampling TP rank {tp_rank}/{builder._tp_size}")
        if builder._tp_size > 1 and not builder._use_native_tp:
            group = distributed.tp_group(str(config["device"]))
            if group.size() != builder._tp_size or group.rank() != tp_rank:
                raise ValueError(
                    f"Sampling TP topology {tp_rank}/{builder._tp_size} does not match "
                    f"the initialized group {group.rank()}/{group.size()}"
                )
        return builder

    def build(
        self,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> ModelSamplingMetadata:
        return ModelSamplingMetadata(
            input_batch.sampling,
            input_batch.num_tokens,
            self._tp_size,
            self._use_native_tp,
        )

    def can_execute_graph(
        self,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> bool:
        return _supports_graph_sampling(input_batch.sampling)

    def allocate_persistent(
        self,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> ModelSamplingMetadata:
        # The graph path currently supports only unconstrained greedy sampling,
        # which has no external sampling tensors requiring persistent buffers.
        self._validate_graph_sampling(input_batch.sampling)
        return self.build(input_batch, metadata)

    def update_persistent(
        self,
        persistent_metadata: object,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> None:
        if not isinstance(persistent_metadata, ModelSamplingMetadata):
            raise TypeError("Sampling received invalid persistent metadata")
        self._validate_graph_sampling(input_batch.sampling)
        # Unconstrained greedy sampling only needs host metadata updates here.
        persistent_metadata.sampling = input_batch.sampling
        persistent_metadata.num_tokens = input_batch.num_tokens

    @staticmethod
    def _validate_graph_sampling(sampling: ModelSamplingInput | None) -> None:
        if sampling is None:
            raise ValueError("Pipeline graph execution requires sampling input")
        if not _supports_graph_sampling(sampling):
            raise ValueError(
                "Pipeline graph execution requires unconstrained greedy sampling without probabilities or confidence"
            )
