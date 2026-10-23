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

"""Contracts between the V4.1 proposal adapter and the shared executor."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from xllm.python.layers.attention import Attention
from xllm.python.model_executor.executor import DecodeGraphAdapter, ModelExecutor
from xllm.python.model_executor.runners.base import BaseRunner, DraftProposalOutput
from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner
from xllm.python.models.deepseek_v41_dspark_graph import (
    DeepseekV41DSparkGraphRunner,
    DSparkGreedyGraphModel,
    create_dspark_decode_graph_adapter,
)
from xllm.python.models.dspark import DSparkMarkovHead


class _TargetModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Linear(1, 1)
        self.attention = Attention(8, 2, 64, 0.125, 0, 0)


class _DraftModel(_TargetModel):
    def create_decode_graph_adapter(
        self, config: dict, graph_backend: str, *, is_spec_draft: bool = False
    ) -> DecodeGraphAdapter | None:
        return create_dspark_decode_graph_adapter(self, config, graph_backend, is_spec_draft=is_spec_draft)


def _config(enabled: bool = True) -> dict:
    return {
        "model_type": "deepseek_v41_dspark",
        "max_position_embeddings": 128,
        "python_graph_backend": "aclgraph" if enabled else "off",
        "num_speculative_tokens": 5,
        "draft_query_width": 5,
        "draft_graph_enabled": enabled,
    }


@pytest.mark.parametrize("enabled,num_decoding_tokens", [(False, 1), (False, 4), (True, 1)])
@patch("xllm.python.model_executor.executor._create_attention_backend")
def test_draft_adapter_keeps_eager_backbone_and_resolves_graph_width(
    create_backend: MagicMock, enabled: bool, num_decoding_tokens: int
) -> None:
    create_backend.return_value = SimpleNamespace(page_size=4, is_mla=False)
    model = _DraftModel()
    config = _config(enabled)
    executor = ModelExecutor(model, config, 32, num_decoding_tokens=num_decoding_tokens, is_spec_draft=True)
    assert config["num_speculative_tokens"] == 5
    assert executor.eager_runner.model is model.model
    expected_width = max(num_decoding_tokens, 1 if enabled else 6)
    assert create_backend.call_args.kwargs["num_decoding_tokens"] == expected_width
    if enabled:
        assert isinstance(executor.decode_graph_runner, DeepseekV41DSparkGraphRunner)
        assert isinstance(executor.decode_graph_runner.model, DSparkGreedyGraphModel)
        assert executor.decode_graph_runner.model.owner is model
        assert executor.decode_graph_runner.model.query_width == 5
        assert executor.decode_graph_runner.num_decoding_tokens == expected_width
    else:
        assert executor.decode_graph_runner is None


@pytest.mark.parametrize("num_decoding_tokens", [2, 4, 6])
@patch("xllm.python.model_executor.executor._create_attention_backend")
def test_draft_graph_rejects_a_target_verify_width(create_backend: MagicMock, num_decoding_tokens: int) -> None:
    create_backend.return_value = SimpleNamespace(page_size=4, is_mla=False)
    config = _config()
    with pytest.raises(ValueError, match="requires num_decoding_tokens=1"):
        ModelExecutor(_DraftModel(), config, 32, num_decoding_tokens=num_decoding_tokens, is_spec_draft=True)
    assert config == _config()


@patch("xllm.python.model_executor.executor._create_attention_backend")
def test_target_and_draft_keep_independent_widths_with_the_same_spec_config(create_backend: MagicMock) -> None:
    create_backend.return_value = SimpleNamespace(page_size=4, is_mla=False)
    config = _config()
    target_before = ModelExecutor(_TargetModel(), config, 32, num_decoding_tokens=6)
    draft = ModelExecutor(_DraftModel(), config, 32, num_decoding_tokens=1, is_spec_draft=True)
    target_after = ModelExecutor(_TargetModel(), config, 32, num_decoding_tokens=6)

    assert type(target_before.decode_graph_runner) is DecodeAclGraphRunner
    assert type(target_after.decode_graph_runner) is DecodeAclGraphRunner
    assert isinstance(draft.decode_graph_runner, DeepseekV41DSparkGraphRunner)
    assert [
        target_before.decode_graph_runner.num_decoding_tokens,
        draft.decode_graph_runner.num_decoding_tokens,
        target_after.decode_graph_runner.num_decoding_tokens,
    ] == [6, 1, 6]
    assert [call.kwargs["num_decoding_tokens"] for call in create_backend.call_args_list] == [6, 1, 6]
    assert config == _config()


@patch("xllm.python.model_executor.executor._create_attention_backend")
def test_adapter_resolves_the_head_installed_after_executor_construction(create_backend: MagicMock) -> None:
    class _Backbone(nn.Embedding):
        def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            return super().forward(input_ids)

    create_backend.return_value = SimpleNamespace(page_size=4, is_mla=False)
    model = _DraftModel()
    model.model = _Backbone(6, 1)
    model.lm_head = None
    model.markov_head = DSparkMarkovHead(6, 6, 6, torch.float32, torch.device("cpu"))
    with torch.no_grad():
        model.model.weight.fill_(1)
        model.markov_head.markov_w1.weight.copy_(torch.eye(6))
        # The transition table maps token i to (i + 1) modulo vocabulary size.
        model.markov_head.markov_w2.weight.copy_(torch.eye(6).roll(1, dims=0))
    executor = ModelExecutor(model, _config(), 32, is_spec_draft=True)
    wrapper = executor.decode_graph_runner.model
    assert wrapper.owner.lm_head is None

    # Weight loading installs the head after the executor and its adapter exist.
    model.lm_head = nn.Linear(1, 6, bias=False)
    with torch.no_grad():
        model.lm_head.weight.zero_()
    input_ids = torch.tensor([2, 0, 0, 0, 0], dtype=torch.int64)
    torch.testing.assert_close(wrapper(input_ids, torch.arange(5)), torch.tensor([3, 4, 5, 0, 1]))

    # Resolve the owner's current head rather than retaining its original value.
    replacement_head = nn.Linear(1, 6, bias=False)
    with torch.no_grad():
        replacement_head.weight.zero_()
        replacement_head.weight[0] = 2
    model.lm_head = replacement_head
    torch.testing.assert_close(wrapper(input_ids, torch.arange(5)), torch.zeros(5, dtype=torch.int64))


@patch("xllm.python.model_executor.executor._create_attention_backend")
def test_proposal_result_is_explicit_and_eager_result_remains_hidden(create_backend: MagicMock) -> None:
    create_backend.return_value = SimpleNamespace(page_size=4, is_mla=False)
    model = _DraftModel()
    executor = ModelExecutor(model, _config(), 32, is_spec_draft=True)
    executor._kv_bound = True
    proposals = torch.tensor([3, 1, 4, 1, 5], dtype=torch.int64)
    metadata = SimpleNamespace(is_prefill=False, is_chunked_prefill=False)
    with (
        patch.object(DeepseekV41DSparkGraphRunner, "can_execute", return_value=True),
        patch.object(DeepseekV41DSparkGraphRunner, "warmup"),
        patch.object(DecodeAclGraphRunner, "execute", return_value=proposals),
    ):
        result = executor.execute(torch.zeros(5, dtype=torch.int64), torch.arange(5), metadata)
    assert isinstance(result, DraftProposalOutput)
    assert result.token_ids is proposals
    hidden = torch.zeros(5, 4)
    with (
        patch.object(DeepseekV41DSparkGraphRunner, "can_execute", return_value=False),
        patch.object(executor.eager_runner, "execute", return_value=hidden),
    ):
        result = executor.execute(torch.zeros(5, dtype=torch.int64), torch.arange(5), metadata)
    assert result is hidden


def test_explicit_proposal_survives_aux_attachment_and_padding_trim() -> None:
    result = DraftProposalOutput(torch.arange(10))
    executor = object.__new__(ModelExecutor)
    executor.model = SimpleNamespace(
        aux_hidden_capture=SimpleNamespace(enabled=True, persistent_buffer=True, buffer=torch.ones(10, 2))
    )
    assert executor._attach_persistent_aux_hidden_states(result, 5) is result
    trimmed = BaseRunner._slice_output(result, 5, clone=True)
    assert isinstance(trimmed, DraftProposalOutput)
    torch.testing.assert_close(trimmed.token_ids, torch.arange(5))
    assert trimmed.token_ids.data_ptr() != result.token_ids.data_ptr()


@pytest.mark.parametrize(
    "cp_size,backend,is_draft", [(2, "aclgraph", True), (1, "cudagraphs", True), (1, "aclgraph", False)]
)
def test_unsupported_execution_contract_is_rejected_before_runner_creation(
    cp_size: int, backend: str, is_draft: bool
) -> None:
    config = {**_config(), "cp_size": cp_size, "python_graph_backend": backend}
    with (
        patch("xllm.python.model_executor.executor._create_attention_backend") as create_backend,
        pytest.raises(ValueError),
    ):
        ModelExecutor(_DraftModel(), config, 32, is_spec_draft=is_draft)
    create_backend.assert_not_called()


def test_dp_empty_shard_uses_the_same_complete_block_bucket() -> None:
    runner = DeepseekV41DSparkGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=8,
        max_model_len=128,
        dp_size=2,
        is_spec_draft=True,
        draft_query_width=5,
    )
    active = SimpleNamespace(
        dp_execution_token_counts=(15, 1),
        raw_dp_execution_token_counts=(15, 0),
    )
    empty = SimpleNamespace(
        dp_execution_token_counts=(15, 1),
        raw_dp_execution_token_counts=(15, 0),
    )
    active_plan = runner._shared_graph_plan(active, 15)
    empty_plan = runner._shared_graph_plan(empty, 1)
    assert active_plan == empty_plan == (False, 1, 3)
    assert runner._padded_token_rows(15, active_plan[2]) == runner._padded_token_rows(1, empty_plan[2]) == 20
