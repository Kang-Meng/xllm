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

import sys
from contextlib import nullcontext
from dataclasses import dataclass, field, fields, replace
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from xllm.python import distributed
from xllm.python.model_executor.executor import ModelExecutor
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.model_executor.input_batch import InputBatch
from xllm.python.model_executor.model_output import DraftProposal, ModelOutput
from xllm.python.model_executor.runners.base import BaseRunner
from xllm.python.model_executor.runners.decode_cuda_graph import DecodeCudaGraphRunner
from xllm.python.model_executor.runners.eager import EagerRunner
from xllm.python.model_executor.runners.inductor import InductorRunner
from xllm.python.model_executor.sampling_metadata import (
    ModelSamplingMetadata,
    ModelSamplingMetadataBuilder,
    prepare_sampling_input,
    sample_logits,
)
from xllm.python.models.base import PyModelBase
from xllm.python.models.dspark_pipeline import DSparkPipeline


class _Backbone(nn.Module):
    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self.calls = calls

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.calls.append("draft")
        return torch.stack((input_ids.float(), positions.float()), dim=-1) * 0.01


class _LMHead(nn.Linear):
    def __init__(self, calls: list[str], vocab_size: int) -> None:
        super().__init__(2, vocab_size, bias=False)
        self.calls = calls
        with torch.no_grad():
            self.weight.copy_(torch.linspace(-0.2, 0.2, vocab_size * 2).reshape(vocab_size, 2))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.calls.append("lm_head")
        return super().forward(hidden)


class _MarkovHead(nn.Module):
    def __init__(self, calls: list[str], previous_tokens: list[torch.Tensor], vocab_size: int) -> None:
        super().__init__()
        self.calls = calls
        self.previous_tokens = previous_tokens
        self.vocab_size = vocab_size

    def embed(self, previous_tokens: torch.Tensor) -> torch.Tensor:
        return previous_tokens.float()

    def bias(self, previous_tokens: torch.Tensor) -> torch.Tensor:
        self.calls.append("markov")
        self.previous_tokens.append(previous_tokens.clone())
        preferred = (previous_tokens + 1) % self.vocab_size
        return torch.nn.functional.one_hot(preferred, self.vocab_size).float() * 4


class _ConfidenceHead(nn.Module):
    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self.calls = calls

    def forward(self, hidden: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        self.calls.append("confidence")
        return torch.sigmoid(hidden[..., 0] + previous)


class _DraftModel(PyModelBase):
    def __init__(self, vocab_size: int = 7, width: int = 3) -> None:
        super().__init__()
        self.calls: list[str] = []
        self.previous_tokens: list[torch.Tensor] = []
        self.model = DSparkPipeline(
            backbone=_Backbone(self.calls),
            lm_head=_LMHead(self.calls, vocab_size),
            markov_head=_MarkovHead(self.calls, self.previous_tokens, vocab_size),
            confidence_head=_ConfidenceHead(self.calls),
            query_width=width,
            target_vocab_size=7,
        )


@dataclass
class _SamplingParameters:
    all_greedy_sample: bool = True
    selected_token_idxes: torch.Tensor | None = None
    sample_idxes: torch.Tensor | None = None
    logprobs: bool = False
    max_top_logprobs: int = 0
    use_beam_search: bool = False


@dataclass
class _SamplingInput:
    return_probs: bool = False
    return_confidence: bool = False
    is_unconstrained_greedy: bool = True
    parameters: _SamplingParameters = field(default_factory=_SamplingParameters)


def _copy_sampling_input(sampling_input: _SamplingInput) -> _SamplingInput:
    return replace(sampling_input, parameters=replace(sampling_input.parameters))


def _sampling_parameters_to(
    parameters: _SamplingParameters,
    logits: torch.Tensor,
) -> _SamplingParameters:
    return replace(parameters)


def _sample_logits(
    logits: torch.Tensor,
    sampling_input: _SamplingInput,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    probabilities = logits.softmax(-1) if sampling_input.return_probs else None
    return logits.argmax(-1), probabilities


@pytest.fixture(autouse=True)
def sampling_runtime(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    runtime = ModuleType("xllm_runtime")
    runtime.ModelSamplingInput = Mock(side_effect=_copy_sampling_input)
    runtime.sampling_parameters_to = Mock(side_effect=_sampling_parameters_to)
    runtime.sample_logits = Mock(side_effect=_sample_logits)
    monkeypatch.setitem(sys.modules, "xllm_runtime", runtime)
    return runtime


def _batch_metadata(
    num_tokens: int,
    width: int,
) -> SimpleNamespace:
    num_requests = num_tokens // width
    return SimpleNamespace(
        num_tokens=num_tokens,
        num_reqs=num_requests,
        num_scheduled_tokens=[width] * num_requests,
        num_computed_tokens=[0] * num_requests,
        query_start_loc=list(range(0, num_tokens + 1, width)),
        is_prefilling=[False] * num_requests,
    )


def _sampling_batch(sampling: _SamplingInput | None, num_tokens: int = 3) -> InputBatch:
    metadata = _batch_metadata(num_tokens, 3)
    metadata.num_computed_tokens = []
    return InputBatch.from_runtime(
        torch.arange(num_tokens),
        torch.arange(num_tokens),
        metadata,
        is_dummy=False,
        sampling=sampling,
    )


def _context(
    sampling: _SamplingInput | None,
    num_tokens: int,
    *,
    tp_size: int = 1,
) -> ForwardContext:
    return ForwardContext(
        Mock(),
        torch.device("cpu"),
        Mock(),
        [],
        execution_contexts={
            ModelSamplingMetadata: ModelSamplingMetadata(sampling, num_tokens, tp_size),
        },
    )


@pytest.mark.parametrize(
    "all_greedy,unconstrained,tp_size,expected_sync",
    [
        (True, True, 4, False),
        (True, False, 4, False),
        (False, False, 1, False),
        (False, False, 4, True),
    ],
)
def test_sampling_sync_uses_python_group_and_preserves_probabilities(
    all_greedy: bool,
    unconstrained: bool,
    tp_size: int,
    expected_sync: bool,
    sampling_runtime: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_tokens = torch.tensor([1, 0, 2, 0])[::2]
    probabilities = torch.ones(2, 7) / 7
    sampling_runtime.sample_logits.side_effect = None
    sampling_runtime.sample_logits.return_value = (local_tokens, probabilities)
    sampling = _SamplingInput(
        return_probs=not unconstrained,
        is_unconstrained_greedy=unconstrained,
        parameters=_SamplingParameters(all_greedy_sample=all_greedy),
    )
    logits = torch.zeros(2, 7)
    broadcast = Mock(side_effect=lambda tokens, **kwargs: tokens.fill_(5))
    monkeypatch.setattr(distributed, "broadcast_", broadcast)

    tokens, actual_probabilities = sample_logits(
        logits,
        sampling,
        tp_size=tp_size,
    )

    sampling_runtime.sample_logits.assert_called_once_with(logits, sampling)
    assert actual_probabilities is probabilities
    if expected_sync:
        broadcast.assert_called_once_with(tokens, src=0, group_name="tp")
        assert tokens.is_contiguous()
        torch.testing.assert_close(tokens, torch.tensor([5, 5]))
        torch.testing.assert_close(local_tokens, torch.tensor([1, 2]))
    else:
        broadcast.assert_not_called()
        assert tokens is local_tokens


def test_pipeline_syncs_each_sample_before_remapping_and_markov_feedback(
    sampling_runtime: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _DraftModel(vocab_size=3, width=3)
    model.model.hot_token_ids = torch.tensor([1, 4, 6])
    sampling = _SamplingInput(
        return_probs=True,
        is_unconstrained_greedy=False,
        parameters=_SamplingParameters(all_greedy_sample=False),
    )
    input_ids = torch.tensor([2, 0, 0, 5, 0, 0])
    positions = torch.arange(6)
    synchronized_tokens = iter([1, 2, 0])

    def broadcast(tokens: torch.Tensor, *, src: int, group_name: str) -> None:
        assert src == 0
        assert group_name == "tp"
        tokens.fill_(next(synchronized_tokens))

    sync = Mock(side_effect=broadcast)
    monkeypatch.setattr(distributed, "broadcast_", sync)
    with forward_context(_context(sampling, input_ids.numel(), tp_size=2)):
        output = model.model(input_ids, positions)

    assert sync.call_count == 3
    assert sampling_runtime.sample_logits.call_count == 3
    torch.testing.assert_close(output.draft_proposal.token_ids, torch.tensor([[4, 6, 1], [4, 6, 1]]))
    expected_feedback = [torch.tensor([2, 5]), torch.tensor([4, 4]), torch.tensor([6, 6])]
    assert len(model.previous_tokens) == len(expected_feedback)
    for actual, expected in zip(model.previous_tokens, expected_feedback):
        torch.testing.assert_close(actual, expected)


def _reference(
    model: _DraftModel,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    width: int,
    sampling: _SamplingInput,
) -> ModelOutput:
    batch_size = input_ids.numel() // width
    pipeline = model.model
    hidden = pipeline.backbone(input_ids, positions)
    logits = pipeline.compute_logits(hidden, None).reshape(batch_size, width, -1)
    anchors = input_ids.reshape(batch_size, width)[:, 0]
    previous = anchors
    tokens = torch.empty((batch_size, width), dtype=torch.long)
    distributions: list[torch.Tensor] = []
    for step in range(width):
        step_logits = logits[:, step] + pipeline.markov_head.bias(previous)
        next_tokens = step_logits.argmax(-1)
        probabilities = step_logits.softmax(-1) if sampling.return_probs else None
        if model.model.hot_token_ids is not None:
            next_tokens = model.model.hot_token_ids[next_tokens]
            if probabilities is not None:
                full_probs = probabilities.new_zeros((batch_size, model.model.target_vocab_size))
                full_probs[:, model.model.hot_token_ids] = probabilities
                probabilities = full_probs
        tokens[:, step] = next_tokens
        if probabilities is not None:
            distributions.append(probabilities)
        previous = next_tokens
    previous_matrix = torch.cat((anchors[:, None], tokens[:, :-1]), dim=1)
    return ModelOutput(
        draft_proposal=DraftProposal(
            token_ids=tokens,
            draft_probs=torch.stack(distributions, dim=1) if distributions else None,
        ),
        draft_confidence_probs=pipeline.confidence_probs(hidden.reshape(batch_size, width, -1), previous_matrix),
    )


@pytest.mark.parametrize("width", [1, 3, 5, 15])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("reduced_vocab", [False, True])
@pytest.mark.parametrize("return_probs", [False, True])
def test_pipeline_matches_separate_draft_head_and_markov(
    width: int,
    batch_size: int,
    reduced_vocab: bool,
    return_probs: bool,
    sampling_runtime: ModuleType,
) -> None:
    model = _DraftModel(3 if reduced_vocab else 7, width)
    model.model.hot_token_ids = torch.tensor([1, 4, 6]) if reduced_vocab else None
    sampling = _SamplingInput(
        return_probs=return_probs,
        return_confidence=True,
        is_unconstrained_greedy=not return_probs,
    )
    input_ids = torch.zeros(batch_size * width, dtype=torch.long)
    input_ids[::width] = torch.arange(batch_size) + 1
    positions = torch.arange(input_ids.numel())
    expected = _reference(model, input_ids, positions, width, sampling)
    expected_feedback = list(model.previous_tokens)
    model.calls.clear()
    model.previous_tokens.clear()

    with forward_context(_context(sampling, input_ids.numel())):
        output = model.model(input_ids, positions)

    assert isinstance(model.model, PyModelBase)
    assert isinstance(output, ModelOutput)
    assert model.calls == ["draft", "lm_head"] + ["markov"] * width + ["confidence"]
    sampling_runtime.ModelSamplingInput.assert_called_once_with(sampling)
    assert sampling_runtime.sampling_parameters_to.call_count == int(not sampling.is_unconstrained_greedy)
    assert sampling_runtime.sample_logits.call_count == width
    prepared_sampling = sampling_runtime.sample_logits.call_args_list[0].args[1]
    assert prepared_sampling is not sampling
    assert prepared_sampling.parameters is not sampling.parameters
    assert prepared_sampling.return_probs == sampling.return_probs
    assert prepared_sampling.return_confidence == sampling.return_confidence
    assert all(call.args[1] is prepared_sampling for call in sampling_runtime.sample_logits.call_args_list)
    assert torch.equal(output.draft_proposal.token_ids, expected.draft_proposal.token_ids)
    assert torch.equal(output.draft_confidence_probs, expected.draft_confidence_probs)
    assert all(torch.equal(actual, expected) for actual, expected in zip(model.previous_tokens, expected_feedback))
    if return_probs:
        torch.testing.assert_close(output.draft_proposal.draft_probs, expected.draft_proposal.draft_probs)
    else:
        assert output.draft_proposal.draft_probs is None


@pytest.mark.parametrize("padding_requests", [0, 1, 2])
def test_pipeline_computes_local_query_blocks_including_graph_padding(
    padding_requests: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _DraftModel()
    width = 3
    input_ids = torch.tensor([1, 0, 0, 4, 0, 0])
    positions = torch.arange(input_ids.numel())
    sampling = _SamplingInput(return_probs=True, return_confidence=True)
    expected = _reference(model, input_ids, positions, width, sampling)
    padded_input_ids = torch.cat((input_ids, input_ids.new_full((padding_requests * width,), 2)))
    padded_positions = torch.arange(padded_input_ids.numel())
    expected_hidden = model.model.backbone(padded_input_ids, padded_positions)
    lm_head_forward = Mock(wraps=model.model.lm_head.forward)
    monkeypatch.setattr(model.model.lm_head, "forward", lm_head_forward)

    with forward_context(_context(sampling, padded_input_ids.numel())):
        output = model.model(padded_input_ids, padded_positions)

    lm_head_forward.assert_called_once()
    assert torch.equal(lm_head_forward.call_args.args[0], expected_hidden)
    assert output.draft_proposal.token_ids.shape == (input_ids.numel() // width + padding_requests, width)
    trimmed = BaseRunner._slice_output(output, input_ids.numel(), num_reqs=2)
    assert torch.equal(trimmed.draft_proposal.token_ids, expected.draft_proposal.token_ids)
    assert torch.equal(trimmed.draft_confidence_probs, expected.draft_confidence_probs)
    torch.testing.assert_close(trimmed.draft_proposal.draft_probs, expected.draft_proposal.draft_probs)


def test_prepare_sampling_input_adapts_only_the_owned_copy(sampling_runtime: ModuleType) -> None:
    selected_indices = torch.tensor([0, 2], dtype=torch.int32)
    sample_indices = torch.tensor([1], dtype=torch.int32)
    source_parameters = _SamplingParameters(
        selected_token_idxes=selected_indices,
        sample_idxes=sample_indices,
        logprobs=True,
        max_top_logprobs=5,
        use_beam_search=True,
    )
    sampling = _SamplingInput(
        return_probs=True,
        return_confidence=True,
        is_unconstrained_greedy=False,
        parameters=source_parameters,
    )
    logits = torch.zeros((2, 7), dtype=torch.float64)

    prepared = prepare_sampling_input(sampling, logits)

    sampling_runtime.ModelSamplingInput.assert_called_once_with(sampling)
    sampling_runtime.sampling_parameters_to.assert_called_once_with(source_parameters, logits)
    assert prepared is not sampling
    assert prepared.parameters is not source_parameters
    assert prepared.return_probs
    assert prepared.return_confidence
    parameters = prepared.parameters
    assert parameters.selected_token_idxes.shape == (0,)
    assert parameters.selected_token_idxes.dtype == torch.int32
    assert parameters.selected_token_idxes.device == logits.device
    assert parameters.sample_idxes is parameters.selected_token_idxes
    assert not parameters.logprobs
    assert parameters.max_top_logprobs == 0
    assert not parameters.use_beam_search
    assert sampling.parameters is source_parameters
    assert source_parameters.selected_token_idxes is selected_indices
    assert source_parameters.sample_idxes is sample_indices
    assert source_parameters.logprobs
    assert source_parameters.max_top_logprobs == 5
    assert source_parameters.use_beam_search


def test_prepare_sampling_input_greedy_skips_parameter_transfer(sampling_runtime: ModuleType) -> None:
    selected_indices = torch.tensor([0, 2], dtype=torch.int32)
    sampling = _SamplingInput(parameters=_SamplingParameters(selected_token_idxes=selected_indices))

    prepared = prepare_sampling_input(sampling, torch.zeros((2, 7)))

    sampling_runtime.ModelSamplingInput.assert_called_once_with(sampling)
    sampling_runtime.sampling_parameters_to.assert_not_called()
    assert prepared is not sampling
    assert prepared.parameters is not sampling.parameters
    assert prepared.parameters.selected_token_idxes is selected_indices


def test_pipeline_uses_current_anchor_tokens_without_storing_request_state() -> None:
    model = _DraftModel()
    positions = torch.arange(3)
    with forward_context(_context(_SamplingInput(), 3)):
        first = model.model(torch.tensor([1, 0, 0]), positions)
    with forward_context(_context(_SamplingInput(), 3)):
        second = model.model(torch.tensor([4, 0, 0]), positions)
    raw = model.model.backbone(torch.tensor([1]), torch.tensor([0]))
    assert first.draft_proposal.token_ids.tolist() == [[2, 3, 4]]
    assert second.draft_proposal.token_ids.tolist() == [[5, 6, 0]]
    assert isinstance(raw, torch.Tensor)
    assert type(model.model) is DSparkPipeline
    assert not hasattr(model.model, "model")
    assert type(model.model.backbone) is _Backbone
    assert not isinstance(model.model.backbone, DSparkPipeline)
    assert list(model.children()) == [model.model]


@pytest.mark.parametrize("missing", ["context", "metadata", "sampling"])
def test_pipeline_requires_sampling_before_backbone(missing: str) -> None:
    model = _DraftModel()
    context = _context(None, 3)
    if missing == "metadata":
        context.execution_contexts.clear()
    context_scope = nullcontext() if missing == "context" else forward_context(context)
    with context_scope, pytest.raises(RuntimeError, match="requires sampling metadata"):
        model.model(torch.tensor([1, 0, 0]), torch.arange(3))
    assert model.calls == []


@pytest.mark.parametrize("width,num_tokens", [(0, 3), (-1, 3), (3, 2)])
def test_pipeline_rejects_invalid_query_blocks_before_backbone(width: int, num_tokens: int) -> None:
    model = _DraftModel(width=width)
    with (
        forward_context(_context(_SamplingInput(), num_tokens)),
        pytest.raises(ValueError, match="complete query blocks"),
    ):
        model.model(torch.zeros(num_tokens, dtype=torch.long), torch.arange(num_tokens))
    assert model.calls == []


def test_runner_trims_pipeline_graph_padding() -> None:
    model = _DraftModel()
    with forward_context(_context(_SamplingInput(return_probs=True, return_confidence=True), 6)):
        output = model.model(torch.arange(8), torch.arange(8))
    assert output.draft_proposal.token_ids.shape == (2, 3)
    trimmed = BaseRunner._slice_output(output, 3, num_reqs=1, clone=True)
    tensors = (
        (output.draft_proposal.token_ids, trimmed.draft_proposal.token_ids),
        (output.draft_proposal.draft_probs, trimmed.draft_proposal.draft_probs),
        (output.draft_confidence_probs, trimmed.draft_confidence_probs),
    )
    for original, actual in tensors:
        assert torch.equal(actual, original[:1])
        assert actual.data_ptr() != original.data_ptr()


@pytest.mark.parametrize("output_kind", ["tensor", "tuple", "pipeline"])
@pytest.mark.parametrize("clone", [False, True])
def test_runner_preserves_output_structure(output_kind: str, clone: bool) -> None:
    hidden = torch.arange(12).reshape(6, 2)
    if output_kind == "pipeline":
        output = ModelOutput(hidden_states=hidden, residual=hidden + 2, aux_hidden_states=hidden + 1)
        sliced = BaseRunner._slice_output(output, 2, clone=clone)
        assert torch.equal(sliced.hidden_states, hidden[:2])
        assert torch.equal(sliced.residual, hidden[:2] + 2)
        assert torch.equal(sliced.aux_hidden_states, hidden[:2] + 1)
        assert sliced.draft_proposal is None
        assert sliced.draft_confidence_probs is None
        sliced_hidden = sliced.hidden_states
    elif output_kind == "tuple":
        sliced = BaseRunner._slice_output((hidden, hidden + 1), 2, clone=clone)
        assert torch.equal(sliced[0], hidden[:2])
        assert torch.equal(sliced[1], hidden[:2] + 1)
        sliced_hidden = sliced[0]
    else:
        sliced_hidden = BaseRunner._slice_output(hidden, 2, clone=clone)
        assert torch.equal(sliced_hidden, hidden[:2])
    assert (sliced_hidden.data_ptr() != hidden.data_ptr()) == clone


def test_pipeline_output_fields_match_native_contract() -> None:
    assert [member.name for member in fields(ModelOutput)] == [
        "draft_proposal",
        "draft_confidence_probs",
        "hidden_states",
        "residual",
        "aux_hidden_states",
    ]
    assert [member.name for member in fields(DraftProposal)] == ["token_ids", "draft_probs"]


@pytest.mark.parametrize("num_tokens,num_reqs", [(1, 1), (6, 2), (5, 2), (12, 4)])
@pytest.mark.parametrize("return_probs", [False, True])
@pytest.mark.parametrize("return_confidence", [False, True])
@pytest.mark.parametrize("clone", [False, True])
def test_runner_trims_pipeline_output_with_independent_token_and_request_counts(
    num_tokens: int, num_reqs: int, return_probs: bool, return_confidence: bool, clone: bool
) -> None:
    hidden = torch.arange(60).reshape(12, 5)
    token_ids = torch.arange(12).reshape(4, 3)
    probabilities = torch.arange(36).reshape(4, 3, 3) if return_probs else None
    confidence = torch.ones(4, 3) if return_confidence else None
    output = ModelOutput(
        draft_proposal=DraftProposal(token_ids, probabilities),
        draft_confidence_probs=confidence,
        hidden_states=hidden,
        residual=hidden + 1,
        aux_hidden_states=hidden + 2,
    )
    sliced = BaseRunner._slice_output(output, num_tokens, num_reqs=num_reqs, clone=clone)
    for name in ("hidden_states", "residual", "aux_hidden_states"):
        original = getattr(output, name)
        actual = getattr(sliced, name)
        assert torch.equal(actual, original[:num_tokens])
        assert (actual.data_ptr() != original.data_ptr()) == clone
    assert torch.equal(sliced.draft_proposal.token_ids, token_ids[:num_reqs])
    assert (sliced.draft_proposal.token_ids.data_ptr() != token_ids.data_ptr()) == clone
    assert output.draft_proposal.token_ids.shape == (4, 3)
    if return_probs:
        assert torch.equal(sliced.draft_proposal.draft_probs, probabilities[:num_reqs])
        assert (sliced.draft_proposal.draft_probs.data_ptr() != probabilities.data_ptr()) == clone
    else:
        assert sliced.draft_proposal.draft_probs is None
    if return_confidence:
        assert torch.equal(sliced.draft_confidence_probs, confidence[:num_reqs])
        assert (sliced.draft_confidence_probs.data_ptr() != confidence.data_ptr()) == clone
    else:
        assert sliced.draft_confidence_probs is None


@pytest.mark.parametrize("with_proposal", [False, True])
def test_runner_requires_request_count_for_request_scoped_output(with_proposal: bool) -> None:
    output = ModelOutput(
        draft_proposal=DraftProposal(torch.ones(2, 3, dtype=torch.long)) if with_proposal else None,
        draft_confidence_probs=None if with_proposal else torch.ones(2, 3),
    )
    with pytest.raises(ValueError, match="requires num_reqs"):
        BaseRunner._slice_output(output, 3)


def test_runner_trims_confidence_without_proposal() -> None:
    confidence = torch.arange(9).reshape(3, 3)
    output = ModelOutput(draft_confidence_probs=confidence)
    sliced = BaseRunner._slice_output(output, 5, num_reqs=2)
    assert sliced.draft_proposal is None
    assert sliced.hidden_states is None
    assert torch.equal(sliced.draft_confidence_probs, confidence[:2])


@pytest.mark.parametrize("clone", [False, True])
def test_runner_trims_pipeline_output_to_empty_batch(clone: bool) -> None:
    output = ModelOutput(
        draft_proposal=DraftProposal(torch.ones(2, 3, dtype=torch.long), torch.ones(2, 3, 7)),
        draft_confidence_probs=torch.ones(2, 3),
        hidden_states=torch.ones(6, 5),
    )
    sliced = BaseRunner._slice_output(output, 0, num_reqs=0, clone=clone)
    assert sliced.hidden_states.shape == (0, 5)
    assert sliced.draft_proposal.token_ids.shape == (0, 3)
    assert sliced.draft_proposal.draft_probs.shape == (0, 3, 7)
    assert sliced.draft_confidence_probs.shape == (0, 3)


def test_runner_rejects_outputs_outside_the_explicit_contract() -> None:
    with pytest.raises(TypeError, match="Graph output must be"):
        BaseRunner._slice_output(SimpleNamespace(slice_tokens=Mock()), 3)


@pytest.mark.parametrize("output_kind", ["tensor", "tuple", "pipeline"])
def test_cuda_runner_slices_replay_output_with_input_batch_counts(
    output_kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _sampling_batch(_SamplingInput(), num_tokens=9)
    hidden = torch.arange(32).reshape(16, 2)
    source = hidden
    if output_kind == "tuple":
        source = (hidden, hidden + 1)
    elif output_kind == "pipeline":
        source = ModelOutput(
            draft_proposal=DraftProposal(torch.arange(15).reshape(5, 3)),
            draft_confidence_probs=torch.ones(5, 3),
        )
    backend = SimpleNamespace(prepare=Mock())
    runner = DecodeCudaGraphRunner(nn.Identity(), backend, torch.device("cpu"), max_batch=16, max_model_len=16)
    metadata = SimpleNamespace()
    entry = SimpleNamespace(graph=Mock(), static_output=source, static_metadata=metadata)
    runner._graphs[runner._graph_key(batch.input_ids, metadata)] = entry
    runner._stream = Mock()
    monkeypatch.setattr(runner, "_fill_entry", Mock())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: runner._stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    output = runner.execute(
        batch.input_ids,
        batch.positions,
        metadata,
        input_batch=batch if output_kind == "pipeline" else None,
    )

    entry.graph.replay.assert_called_once()
    if output_kind == "pipeline":
        assert torch.equal(output.draft_proposal.token_ids, source.draft_proposal.token_ids[: batch.num_reqs])
        assert output.draft_proposal.token_ids.data_ptr() != source.draft_proposal.token_ids.data_ptr()
        assert torch.equal(output.draft_confidence_probs, source.draft_confidence_probs[: batch.num_reqs])
        assert output.hidden_states is None
    elif output_kind == "tuple":
        for actual, original in zip(output, source):
            assert torch.equal(actual, original[: batch.num_tokens])
            assert actual.data_ptr() != original.data_ptr()
    else:
        assert torch.equal(output, source[: batch.num_tokens])
        assert output.data_ptr() != source.data_ptr()


@pytest.mark.parametrize(
    "sampling",
    [
        _SamplingInput(),
        _SamplingInput(is_unconstrained_greedy=False),
        _SamplingInput(return_probs=True),
        _SamplingInput(return_confidence=True),
        None,
    ],
)
@pytest.mark.parametrize("runner", ["graph", "inductor", "eager"])
@pytest.mark.parametrize("with_batch_metadata", [False, True])
def test_executor_uses_original_execute_interface(
    sampling: _SamplingInput | None,
    runner: str,
    with_batch_metadata: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialize_batch = Mock(wraps=InputBatch.from_runtime)
    monkeypatch.setattr(InputBatch, "from_runtime", materialize_batch)
    model = _DraftModel()
    builder = ModelSamplingMetadataBuilder.from_config({})
    executor = ModelExecutor.__new__(ModelExecutor)
    executor._restore_engram_history = ModelExecutor._resolve_engram_restore(model)
    executor._kv_bound = True
    executor._is_spec_draft = True
    executor._execution_metadata_builders = (builder,)
    executor.layerwise_split_size = 1

    synchronizer = Mock()
    synchronizer.size.return_value = 4
    synchronizer.synchronize_layer.return_value = True

    def execute_model(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: object,
        input_embedding: torch.Tensor | None = None,
        **kwargs: object,
    ) -> ModelOutput:
        synchronizer.synchronize_layer.assert_called_once_with(3)
        context = _context(None, input_ids.numel())
        input_batch = kwargs["input_batch"]
        assert input_batch.sampling is sampling
        assert input_batch.num_computed_tokens == ()
        assert "sampling_input" not in kwargs
        context.execution_contexts[ModelSamplingMetadata] = builder.build(input_batch, metadata)
        with forward_context(context):
            return model.model(input_ids, positions)

    executor.eager_runner = Mock()
    executor.eager_runner.execute.side_effect = execute_model
    inductor_runner = Mock()
    inductor_runner.execute.side_effect = execute_model
    executor.inductor_runner = inductor_runner if runner != "eager" else None
    executor.decode_graph_runner = Mock()
    executor.decode_graph_runner.can_execute.return_value = runner == "graph"
    executor.decode_graph_runner.execute.side_effect = execute_model
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        slot_mapping=torch.arange(3),
    )
    batch_metadata = _batch_metadata(3, 3) if with_batch_metadata else None
    if batch_metadata is not None:
        batch_metadata.num_computed_tokens = []
    if batch_metadata is None:
        expectation = pytest.raises(RuntimeError, match="require upstream InputBatch")
    elif sampling is None:
        expectation = pytest.raises(RuntimeError, match="requires sampling metadata")
    else:
        expectation = nullcontext()
    with expectation:
        output = executor.execute(
            torch.tensor([1, 0, 0]),
            torch.arange(3),
            metadata,
            input_batch_metadata=batch_metadata,
            sampling_input=sampling,
            layer_load_synchronizer=synchronizer,
            draft_load_event_index=3,
        )
        assert output.draft_proposal.token_ids.tolist() == [[2, 3, 4]]
    if batch_metadata is None:
        materialize_batch.assert_not_called()
        synchronizer.synchronize_layer.assert_not_called()
        executor.decode_graph_runner.can_execute.assert_not_called()
        executor.decode_graph_runner.execute.assert_not_called()
        inductor_runner.execute.assert_not_called()
        executor.eager_runner.execute.assert_not_called()
        assert model.calls == []
        return
    if sampling is None:
        assert model.calls == []
    executor.decode_graph_runner.can_execute.assert_called_once()
    assert executor.decode_graph_runner.execute.call_count == int(runner == "graph")
    assert inductor_runner.execute.call_count == int(runner == "inductor")
    assert executor.eager_runner.execute.call_count == int(runner == "eager")
    assert not hasattr(executor, "draft_pipeline")
    assert not hasattr(executor, "execute_pipeline")
    materialize_batch.assert_called_once()
    assert materialize_batch.call_args.kwargs["sampling"] is sampling


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("with_batch_builder", [False, True])
def test_pipeline_runners_share_batch_carrier_without_host_cached_counts(
    compiled: bool,
    with_batch_builder: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _DraftModel()
    backend = SimpleNamespace(is_mla=False, prepare=Mock())
    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: model)
    runner = (
        InductorRunner(model.model, backend, torch.device("cpu"), backend="inductor")
        if compiled
        else EagerRunner(model.model, backend, torch.device("cpu"))
    )
    sampling_builder = ModelSamplingMetadataBuilder()
    batch_builder = Mock()
    batch_builder.metadata_type = InputBatch
    builders = (batch_builder, sampling_builder) if with_batch_builder else (sampling_builder,)
    runner.bind_execution_metadata_builders(builders)
    assert runner.execution_metadata_builders == builders

    input_ids = torch.tensor([1, 0, 0])
    positions = torch.arange(3)
    sampling = _SamplingInput()
    batch_metadata = _batch_metadata(3, 3)
    batch_metadata.num_computed_tokens = []
    input_batch = InputBatch.from_runtime(
        input_ids,
        positions,
        batch_metadata,
        is_dummy=False,
        sampling=sampling,
    )
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        slot_mapping=torch.arange(3),
    )

    output = runner.execute(
        input_ids,
        positions,
        metadata,
        input_batch=input_batch,
    )

    assert output.draft_proposal.token_ids.tolist() == [[2, 3, 4]]
    backend.prepare.assert_called_once_with(metadata)
    if with_batch_builder:
        batch_builder.build.assert_called_once_with(input_batch, metadata)
    else:
        batch_builder.build.assert_not_called()


def test_persistent_sampling_metadata_refreshes_greedy_inputs() -> None:
    builder = ModelSamplingMetadataBuilder()
    metadata = SimpleNamespace()
    first_sampling = _SamplingInput()
    next_sampling = _SamplingInput()
    assert builder.can_execute_graph(_sampling_batch(first_sampling), metadata)
    persistent = builder.allocate_persistent(_sampling_batch(first_sampling), metadata)
    builder.update_persistent(persistent, _sampling_batch(next_sampling, 6), metadata)
    assert persistent.sampling is next_sampling
    assert persistent.num_tokens == 6


@pytest.mark.parametrize(
    "sampling",
    [
        None,
        _SamplingInput(return_probs=True),
        _SamplingInput(return_confidence=True),
        _SamplingInput(is_unconstrained_greedy=False),
    ],
)
def test_graph_sampling_rejects_unsupported_inputs_before_updating(sampling: _SamplingInput | None) -> None:
    builder = ModelSamplingMetadataBuilder()
    metadata = SimpleNamespace()
    first_sampling = _SamplingInput()
    persistent = builder.allocate_persistent(_sampling_batch(first_sampling), metadata)
    assert not builder.can_execute_graph(_sampling_batch(sampling), metadata)
    with pytest.raises(ValueError, match="Pipeline graph execution requires"):
        builder.allocate_persistent(_sampling_batch(sampling), metadata)
    with pytest.raises(ValueError, match="Pipeline graph execution requires"):
        builder.update_persistent(persistent, _sampling_batch(sampling, 6), metadata)
    assert persistent.sampling is first_sampling
    assert persistent.num_tokens == 3
    assert builder.build(_sampling_batch(sampling), metadata).sampling is sampling


@pytest.mark.parametrize(
    "config,tp_size",
    [
        ({}, 1),
        ({"tp_size": 1}, 1),
        ({"tp_size": 2, "tp_rank": 1, "device": "cpu"}, 2),
        ({"tp_size": 4, "tp_rank": 0, "device": "cpu"}, 4),
    ],
)
def test_sampling_metadata_uses_initialized_tp_topology(
    config: dict[str, object],
    tp_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = SimpleNamespace(size=lambda: tp_size, rank=lambda: config.get("tp_rank", 0))
    get_group = Mock(return_value=group)
    monkeypatch.setattr(distributed, "tp_group", get_group)
    builder = ModelSamplingMetadataBuilder.from_config(config)
    sampling = _SamplingInput()
    metadata = builder.build(_sampling_batch(sampling), SimpleNamespace())
    assert metadata.tp_size == tp_size
    if tp_size > 1:
        get_group.assert_called_once_with("cpu")
    else:
        get_group.assert_not_called()


@pytest.mark.parametrize("tp_size", [0, -1])
def test_sampling_metadata_rejects_invalid_tp_size(tp_size: int) -> None:
    with pytest.raises(ValueError, match="TP size must be positive"):
        ModelSamplingMetadataBuilder.from_config({"tp_size": tp_size})


@pytest.mark.parametrize("tp_rank", [-1, 2])
def test_sampling_metadata_rejects_invalid_tp_rank(tp_rank: int) -> None:
    with pytest.raises(ValueError, match="Invalid sampling TP rank"):
        ModelSamplingMetadataBuilder.from_config({"tp_size": 2, "tp_rank": tp_rank, "device": "cpu"})


def test_sampling_metadata_rejects_missing_tp_at_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    get_group = Mock(side_effect=RuntimeError("TP group is not initialized for cpu"))
    monkeypatch.setattr(distributed, "tp_group", get_group)
    with pytest.raises(RuntimeError, match="TP group is not initialized"):
        ModelSamplingMetadataBuilder.from_config({"tp_size": 2, "device": "cpu"})
    get_group.assert_called_once_with("cpu")


@pytest.mark.parametrize("group_rank,group_size", [(0, 4), (1, 2)])
def test_sampling_metadata_rejects_mismatched_tp_at_initialization(
    group_rank: int,
    group_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = SimpleNamespace(size=lambda: group_size, rank=lambda: group_rank)
    monkeypatch.setattr(distributed, "tp_group", Mock(return_value=group))
    with pytest.raises(ValueError, match="does not match the initialized group"):
        ModelSamplingMetadataBuilder.from_config({"tp_size": 2, "tp_rank": 0, "device": "cpu"})


@pytest.mark.parametrize("model_type", ["deepseek_v4_dspark", "deepseek_v41_dspark"])
def test_sampling_metadata_uses_native_tp_for_v4_drafts(
    model_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_group = Mock(side_effect=RuntimeError("TP group is not initialized for cpu"))
    monkeypatch.setattr(distributed, "tp_group", get_group)
    builder = ModelSamplingMetadataBuilder.from_config(
        {"model_type": model_type, "tp_size": 4, "tp_rank": 2, "device": "cpu"}
    )
    metadata = builder.build(_sampling_batch(_SamplingInput()), Mock())
    assert metadata.tp_size == 4
    assert metadata.use_native_tp
    get_group.assert_not_called()


@pytest.mark.parametrize("all_greedy,tp_size", [(True, 4), (False, 1), (False, 4)])
def test_sampling_native_tp_sync_uses_root_tokens(
    all_greedy: bool,
    tp_size: int,
    sampling_runtime: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens = torch.tensor([1, 2])
    root_tokens = torch.tensor([5, 6])
    sampling_runtime.sample_logits = Mock(return_value=(tokens, None))
    sampling_runtime.tp_all_gather = Mock(return_value=torch.cat([root_tokens] + [tokens] * (tp_size - 1)))
    broadcast = Mock(side_effect=AssertionError("Native TP must not use Python TP"))
    monkeypatch.setattr(distributed, "broadcast_", broadcast)
    sampling = _SamplingInput(
        is_unconstrained_greedy=all_greedy,
        parameters=_SamplingParameters(all_greedy_sample=all_greedy),
    )
    actual, probabilities = sample_logits(torch.zeros(2, 7), sampling, tp_size=tp_size, use_native_tp=True)
    expected_sync = not all_greedy and tp_size > 1
    assert torch.equal(actual, root_tokens if expected_sync else tokens)
    assert probabilities is None
    assert sampling_runtime.tp_all_gather.call_count == int(expected_sync)
    broadcast.assert_not_called()


def test_sampling_native_tp_rejects_missing_or_wrong_native_group(sampling_runtime: ModuleType) -> None:
    tokens = torch.tensor([1, 2])
    sampling_runtime.sample_logits = Mock(return_value=(tokens, None))
    sampling_runtime.tp_all_gather = Mock(return_value=tokens)
    sampling = _SamplingInput(
        is_unconstrained_greedy=False,
        parameters=_SamplingParameters(all_greedy_sample=False),
    )
    with pytest.raises(RuntimeError, match="Native sampling TP gather"):
        sample_logits(torch.zeros(2, 7), sampling, tp_size=4, use_native_tp=True)


def test_pipeline_carries_native_tp_sampling_backend(
    sampling_runtime: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _DraftModel()
    sampling_runtime.tp_all_gather = Mock(side_effect=lambda tokens, dim: torch.cat([tokens] * 4, dim=dim))
    broadcast = Mock(side_effect=AssertionError("Native TP must not use Python TP"))
    monkeypatch.setattr(distributed, "broadcast_", broadcast)
    sampling = _SamplingInput(
        is_unconstrained_greedy=False,
        parameters=_SamplingParameters(all_greedy_sample=False),
    )
    context = _context(sampling, 3, tp_size=4)
    context.execution_contexts[ModelSamplingMetadata].use_native_tp = True
    with forward_context(context):
        output = model.model(torch.tensor([1, 0, 0]), torch.arange(3))
    assert output.draft_proposal.token_ids.tolist() == [[2, 3, 4]]
    assert sampling_runtime.tp_all_gather.call_count == 3
    broadcast.assert_not_called()
