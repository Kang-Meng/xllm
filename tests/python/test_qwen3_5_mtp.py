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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from xllm.python.layers.npu.qwen3_5.moe_metadata import Qwen3_5MoeMetadata
from xllm.python.layers.sequence_parallel_metadata import SpMetadata
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader
from xllm.python.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM
from xllm.python.models.qwen3_5_mtp import Qwen3_5MtpForCausalLM, Qwen3_5MtpModel, _MtpStateDict
from xllm.python.registry import get_execution_metadata_builder_classes

from .qwen3_5_test_utils import StateDict, gemma_rms_norm, make_config


class _Norm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(width))

    def forward(self, value: torch.Tensor, residual: torch.Tensor | None = None) -> object:
        if residual is None:
            return gemma_rms_norm(value, self.weight, 1e-6)
        value = value + residual
        return gemma_rms_norm(value, self.weight, 1e-6), value


@pytest.mark.parametrize("with_carried_hidden", [False, True])
def test_mtp_norm_projection_uses_carried_hidden(with_carried_hidden: bool) -> None:
    torch.manual_seed(101)
    model = SimpleNamespace(
        cfg=SimpleNamespace(sequence_parallel_enabled=False),
        _comm_policy=None,
        embed_tokens=nn.Embedding(16, 8),
        pre_fc_norm_embedding=_Norm(8),
        pre_fc_norm_hidden=_Norm(8),
        fc=nn.Linear(16, 8, bias=False),
        layers=(),
        norm=lambda hidden, residual: (hidden, residual),
    )
    ids = torch.tensor([2, 7])
    embedding = model.embed_tokens(ids)
    carried = torch.randn_like(embedding) if with_carried_hidden else None
    expected = nn.functional.linear(
        torch.cat(
            (
                gemma_rms_norm(embedding, model.pre_fc_norm_embedding.weight, 1e-6),
                gemma_rms_norm(embedding if carried is None else carried, model.pre_fc_norm_hidden.weight, 1e-6),
            ),
            dim=-1,
        ),
        model.fc.weight,
    )
    result = Qwen3_5MtpModel.forward(model, ids, torch.tensor([0, 1]), carried)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    if carried is not None:
        unconditioned = Qwen3_5MtpModel.forward(model, ids, torch.tensor([0, 1]))
        assert not torch.allclose(result, unconditioned)


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = _Norm(8)

    def load_weights(self, weights: ScopedWeightLoader, context: ParallelLoadContext) -> None:
        weights.load_tensor(self.input_layernorm.weight, "input_layernorm.weight")


@pytest.mark.parametrize("prefix", ["mtp.", "model.mtp."])
@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("head_source", ["untied", "tied", "missing"])
def test_mtp_loader_keeps_shared_and_draft_weights_separate(prefix: str, tp_rank: int, head_source: str) -> None:
    model = SimpleNamespace(
        cfg=make_config(tp_size=2, world_size=2, moe_tp_size=1, ep_size=2, tie_word_embeddings=head_source == "tied"),
        model=SimpleNamespace(
            embed_tokens=nn.Embedding(16, 4),
            pre_fc_norm_embedding=_Norm(8),
            pre_fc_norm_hidden=_Norm(8),
            fc=nn.Linear(16, 8, bias=False),
            norm=_Norm(8),
            layers=[_Layer()],
        ),
        lm_head=nn.Linear(8, 8, bias=False),
    )
    embedding_weight = torch.arange(128, dtype=torch.float32).reshape(16, 8)
    head_weight = embedding_weight + 1000
    values = {
        "model.embed_tokens.weight": embedding_weight,
        "model.layers.0.input_layernorm.weight": torch.full((8,), -10.0),
        prefix + "fc.weight": torch.full((8, 16), 3.0),
        prefix + "pre_fc_norm_embedding.weight": torch.full((8,), 4.0),
        prefix + "pre_fc_norm_hidden.weight": torch.full((8,), 5.0),
        prefix + "norm.weight": torch.full((8,), 6.0),
        prefix + "layers.0.input_layernorm.weight": torch.full((8,), 7.0),
    }
    if head_source != "missing":
        values["lm_head.weight"] = head_weight
    with torch.no_grad():
        Qwen3_5MtpForCausalLM.load_weights(model, [StateDict(values)], tp_rank, 2)
    torch.testing.assert_close(model.model.embed_tokens.weight, embedding_weight[:, tp_rank * 4 : (tp_rank + 1) * 4])
    expected_head = head_weight if head_source == "untied" else embedding_weight
    torch.testing.assert_close(model.lm_head.weight, expected_head[tp_rank * 8 : (tp_rank + 1) * 8])
    assert torch.all(model.model.fc.weight == 3)
    assert torch.all(model.model.pre_fc_norm_embedding.weight == 4)
    assert torch.all(model.model.pre_fc_norm_hidden.weight == 5)
    assert torch.all(model.model.norm.weight == 6)
    assert torch.all(model.model.layers[0].input_layernorm.weight == 7)


def test_mtp_draft_configuration_uses_full_attention_and_lv1() -> None:
    cfg = make_config(tp_size=1, world_size=1, moe_tp_size=1, expert_parallel_degree=1)
    config = {
        "device": "privateuseone",
        "dtype": "bfloat16",
        "num_nextn_predict_layers": 1,
        "expert_parallel_degree": 1,
    }
    with (
        patch("xllm.python.models.qwen3_5_mtp.Qwen3_5Config.from_dict", return_value=cfg),
        patch("xllm.python.models.qwen3_5_mtp.Qwen3_5MtpModel", return_value=nn.Identity()),
        patch("xllm.python.models.qwen3_5_mtp.ColumnParallelLinear", return_value=nn.Identity()),
    ):
        model = Qwen3_5MtpForCausalLM(config)
    assert model.cfg.n_layers == 1
    assert model.cfg.layer_types == ["full_attention"]
    assert model.cfg.expert_parallel_degree == 1
    assert not model.cfg.sequence_parallel_enabled
    assert config["sequence_parallel_enabled"] is False


@pytest.mark.parametrize("prefix", ["mtp.", "model.mtp."])
def test_mtp_individual_experts_are_packed_in_expert_order(prefix: str) -> None:
    root = prefix + "layers.0.mlp.experts."
    shards = []
    for expert in range(2):
        shards.append(
            StateDict(
                {
                    f"{root}{expert}.gate_proj.weight": torch.full((3, 4), expert + 1.0),
                    f"{root}{expert}.up_proj.weight": torch.full((3, 4), expert + 11.0),
                    f"{root}{expert}.down_proj.weight": torch.full((4, 3), expert + 21.0),
                }
            )
        )
    source = _MtpStateDict(shards, 2)
    assert source.has(root + "gate_up_proj")
    assert source.has(root + "down_proj")
    gate_up = source.get_tensor(root + "gate_up_proj")
    down = source.get_tensor(root + "down_proj")
    assert gate_up.shape == (2, 6, 4)
    assert down.shape == (2, 4, 3)
    for expert in range(2):
        assert torch.all(gate_up[expert, :3] == expert + 1)
        assert torch.all(gate_up[expert, 3:] == expert + 11)
        assert torch.all(down[expert] == expert + 21)


@pytest.mark.parametrize("sp", [False, True])
def test_mtp_lv2_configuration_builds_matching_execution_metadata(sp: bool) -> None:
    cfg = make_config(
        device="privateuseone", moe_tp_size=1, ep_size=2, expert_parallel_degree=2, eplv2_sequence_parallel=sp
    )
    config = {
        **vars(cfg),
        "device": "privateuseone",
        "dtype": "bfloat16",
        "num_nextn_predict_layers": 1,
        "is_draft_engine": True,
    }
    with (
        patch("xllm.python.models.qwen3_5_mtp.Qwen3_5Config.from_dict", return_value=cfg),
        patch("xllm.python.models.qwen3_5_mtp.Qwen3_5MtpModel", return_value=nn.Identity()),
        patch("xllm.python.models.qwen3_5_mtp.ColumnParallelLinear", return_value=nn.Identity()),
    ):
        model = Qwen3_5MtpForCausalLM(config)
    assert model.cfg.expert_parallel_degree == config["expert_parallel_degree"] == 2
    assert model.cfg.sequence_parallel_enabled == config["sequence_parallel_enabled"] == sp
    assert model.cfg.layer_types == ["full_attention"]
    builders = [
        builder
        for cls in get_execution_metadata_builder_classes("qwen3_5_moe_mtp")
        if (builder := cls.from_config(config)) is not None
    ]
    assert {builder.metadata_type for builder in builders} == (
        {SpMetadata, Qwen3_5MoeMetadata} if sp else {Qwen3_5MoeMetadata}
    )


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("rows", [1, 5])
def test_mtp_sp_preserves_carried_hidden_and_output_token_order(rank: int, rows: int) -> None:
    torch.manual_seed(102)
    ids = torch.arange(rows)
    positions = torch.arange(rows)
    layout = SpMetadata.from_execution_rows(rows, 2, rank)
    model = SimpleNamespace(
        cfg=SimpleNamespace(sequence_parallel_enabled=True),
        _comm_policy=None,
        embed_tokens=nn.Embedding(16, 8),
        pre_fc_norm_embedding=_Norm(8),
        pre_fc_norm_hidden=_Norm(8),
        fc=nn.Linear(16, 8, bias=False),
        norm=lambda hidden, residual: (hidden, residual),
    )
    carried = torch.randn(rows, 8)
    original_carried = carried.clone()
    fused = model.fc(
        torch.cat((model.pre_fc_norm_embedding(model.embed_tokens(ids)), model.pre_fc_norm_hidden(carried)), dim=-1)
    )
    expected = fused * 2
    padded = torch.nn.functional.pad(fused, (0, 0, 0, layout.local_rows * 2 - rows))
    local = padded.narrow(0, layout.local_offset, layout.local_rows)
    layer = MagicMock(side_effect=lambda hidden, residual, pos: (hidden * 2, residual))
    model.layers = [layer]

    def gather(hidden: torch.Tensor, actual_layout: SpMetadata) -> torch.Tensor:
        assert actual_layout is layout
        torch.testing.assert_close(hidden, local * 2, rtol=0, atol=0)
        return expected

    with (
        patch("xllm.python.models.qwen3_5_mtp.get_execution_context", return_value=layout),
        patch("xllm.python.models.qwen3_5_mtp.sp_all_gather", side_effect=gather),
    ):
        output = Qwen3_5MtpModel.forward(model, ids, positions, carried)
    layer.validate_sp.assert_called_once_with(layout, device=torch.device("cpu"))
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(carried, original_carried, rtol=0, atol=0)
    assert output.shape == carried.shape


def test_mtp_target_preserves_other_backbone_speculative_restriction() -> None:
    cfg = make_config(device="privateuseone", moe_tp_size=1, ep_size=2, expert_parallel_degree=2)
    config = {**vars(cfg), "device": "privateuseone", "dtype": "bfloat16", "num_speculative_tokens": 3}
    with patch("xllm.python.models.qwen3_5.Qwen3_5Config.from_dict", return_value=cfg):
        with pytest.raises(NotImplementedError, match="does not support speculative decoding"):
            Qwen3_5Config.prepare_execution_config(config, torch.device("privateuseone"))
        permitted = Qwen3_5Config.prepare_execution_config(
            config, torch.device("privateuseone"), allow_speculative_decoding=True
        )
        assert permitted.sequence_parallel_enabled is True
        with (
            patch("xllm.python.models.qwen3_5.Qwen3_5Model", return_value=nn.Identity()),
            patch("xllm.python.models.qwen3_5.ColumnParallelLinear", return_value=nn.Identity()),
        ):
            model = Qwen3_5ForCausalLM(config)
    assert model.cfg.expert_parallel_degree == 2
    assert model.cfg.sequence_parallel_enabled == config["sequence_parallel_enabled"] is True


@pytest.mark.parametrize("role", ["PREFILL", "DECODE"])
@pytest.mark.parametrize("speculative_tokens", [0, 3])
def test_mtp_lv1_configuration_uses_shared_sp_policy(role: str, speculative_tokens: int) -> None:
    cfg = make_config(device="privateuseone", expert_parallel_degree=1)
    config = {
        **vars(cfg),
        "device": "privateuseone",
        "dtype": "bfloat16",
        "instance_role": role,
        "num_speculative_tokens": speculative_tokens,
        "is_draft_engine": True,
    }
    with (
        patch("xllm.python.models.qwen3_5_mtp.Qwen3_5Config.from_dict", return_value=cfg),
        patch("xllm.python.models.qwen3_5_mtp.Qwen3_5MtpModel", return_value=nn.Identity()),
        patch("xllm.python.models.qwen3_5_mtp.ColumnParallelLinear", return_value=nn.Identity()),
    ):
        model = Qwen3_5MtpForCausalLM(config)
    assert model.cfg.sequence_parallel_enabled == config["sequence_parallel_enabled"] == (speculative_tokens == 0)
    assert model.cfg.expert_parallel_degree == config["expert_parallel_degree"] == 1


def test_mtp_draft_opt_in_preserves_other_host_restriction() -> None:
    cfg = make_config(device="privateuseone", moe_tp_size=1, ep_size=2, expert_parallel_degree=2)
    config = {**vars(cfg), "is_draft_engine": True}
    with (
        patch("xllm.python.models.qwen3_5.Qwen3_5Config.from_dict", return_value=cfg),
        pytest.raises(NotImplementedError, match="does not support draft engines"),
    ):
        Qwen3_5Config.prepare_execution_config(config, torch.device("privateuseone"), allow_speculative_decoding=True)


def test_mtp_rejects_unsupported_mega_moe() -> None:
    with pytest.raises(NotImplementedError, match="does not support MegaMoe"):
        Qwen3_5MtpForCausalLM({"enable_mega_moe": True})
