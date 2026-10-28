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

import pytest
import torch
from torch import nn

from xllm.python.models.base import PyModelBase

from xllm.python.models.dspark import (
    DSparkConfidenceHead,
    DSparkForCausalLMBase,
    DSparkMarkovHead,
)


def _model(*, enable_confidence_head: bool = True) -> DSparkForCausalLMBase:
    return DSparkForCausalLMBase(
        backbone=nn.Identity(),
        lm_head=None,
        query_width=2,
        vocab_size=4,
        draft_vocab_size=4,
        markov_rank=2,
        hidden_size=3,
        enable_confidence_head=enable_confidence_head,
        confidence_head_with_markov=True,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def test_markov_bias_matches_embedding_projection() -> None:
    head = DSparkMarkovHead(4, 4, 2, torch.float32, torch.device("cpu"))
    with torch.no_grad():
        head.markov_w1.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 2.0],
                    [-1.0, 1.0],
                ]
            )
        )
        head.markov_w2.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, -1.0],
                ]
            )
        )

    output = head.bias(torch.tensor([0, 2]))

    expected = head.markov_w1(torch.tensor([0, 2])) @ head.markov_w2.weight.T
    torch.testing.assert_close(output, expected)


def test_confidence_head_supports_batched_markov_features() -> None:
    head = DSparkConfidenceHead(3, 2, True, torch.device("cpu"))
    with torch.no_grad():
        head.proj.weight.copy_(torch.tensor([[1.0, -1.0, 0.5, 2.0, -2.0]]))
        head.proj.bias.copy_(torch.tensor([0.25]))
    hidden = torch.tensor([[[1.0, 2.0, 3.0], [0.5, 0.0, -1.0]]])
    markov = torch.tensor([[[0.25, 0.5], [1.0, -1.0]]])

    output = head(hidden, markov)

    expected = torch.sigmoid(head.proj(torch.cat((hidden, markov), dim=-1))).squeeze(-1)
    torch.testing.assert_close(output, expected)


def test_pipeline_owns_dspark_head_parameters() -> None:
    assert set(_model().state_dict()) == {
        "model.markov_head.markov_w1.weight",
        "model.markov_head.markov_w2.weight",
        "model.confidence_head.proj.weight",
        "model.confidence_head.proj.bias",
    }


def test_base_forwards_confidence_for_batched_hidden() -> None:
    model = _model()
    hidden = torch.ones(1, 2, 3)
    prev_matrix = torch.tensor([[0, 1]])

    output = model.dspark_confidence_probs(hidden, prev_matrix)

    confidence_head = model.model.confidence_head
    assert confidence_head is not None
    expected = confidence_head(hidden, model.model.markov_head.embed(prev_matrix))
    torch.testing.assert_close(output, expected)
    assert model.has_dspark_confidence_head()


def test_base_rejects_confidence_without_head() -> None:
    model = _model(enable_confidence_head=False)

    with pytest.raises(RuntimeError, match="not enabled"):
        model.dspark_confidence_probs(torch.ones(1, 1, 3), torch.tensor([[0]]))
    assert not model.has_dspark_confidence_head()


@pytest.mark.parametrize("own_head", [False, True])
@pytest.mark.parametrize("own_embedding", [False, True])
def test_shared_weights_bind_to_pipeline_components(own_head: bool, own_embedding: bool) -> None:
    draft = _model()
    execution_model = draft.model
    own_lm_head = nn.Linear(3, 4, bias=False) if own_head else None
    own_token_embedding = nn.Embedding(4, 3) if own_embedding else None
    execution_model.lm_head = own_lm_head
    execution_model.backbone.embed_tokens = own_token_embedding

    source = PyModelBase()
    source.model = nn.Module()
    source.model.embed_tokens = nn.Embedding(4, 3)
    source.lm_head = nn.Linear(3, 4, bias=False)
    draft.share_weights_from(source)

    assert draft.model is execution_model
    assert draft.get_lm_head() is (own_lm_head if own_head else source.lm_head)
    assert draft.get_token_embedding() is (own_token_embedding if own_embedding else source.model.embed_tokens)
    hidden = torch.ones(2, 3)
    torch.testing.assert_close(draft.compute_logits(hidden, None), execution_model.lm_head(hidden))


def test_plain_model_weight_sharing_keeps_existing_layout() -> None:
    source = PyModelBase()
    source.model = nn.Module()
    source.model.embed_tokens = nn.Embedding(4, 3)
    source.lm_head = nn.Linear(3, 4, bias=False)
    draft = PyModelBase()
    draft.model = nn.Module()
    draft.model.embed_tokens = None
    draft.lm_head = None

    draft.share_weights_from(source)

    assert draft.lm_head is source.lm_head
    assert draft.model.embed_tokens is source.model.embed_tokens
