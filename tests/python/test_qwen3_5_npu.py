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

"""NPU-specific parallel and composition tests for Python Qwen3.5."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.python.qwen3_5_test_utils import (
    ConstantModule as _ConstantModule,
)
from tests.python.qwen3_5_test_utils import (
    StateDict as _StateDict,
)
from tests.python.qwen3_5_test_utils import (
    gemma_rms_norm as _gemma_rms_norm,
)
from tests.python.qwen3_5_test_utils import (
    make_config as _config,
)
from xllm.python import distributed, kernels
from xllm.python.kernels_npu.causal_conv1d import (
    causal_conv1d_decode as npu_causal_conv1d_decode,
)
from xllm.python.layers.moe_parallel import Eplv2CommPolicy
from xllm.python.layers.npu.layernorm import NpuGemmaRMSNorm
from xllm.python.layers.npu.qwen3_5.attention import NpuQwen3_5Attention
from xllm.python.layers.npu.qwen3_5.decoder_layer import (
    NpuQwen3_5DecoderLayer,
)
from xllm.python.layers.npu.qwen3_5.gated_delta_net import (
    NpuQwen3_5GatedDeltaNet,
)
from xllm.python.layers.npu.qwen3_5.gdn_metadata import (
    GdnDecodeMetadata,
    GdnMetadata,
    GdnPrefillMetadata,
    GdnStateCache,
)
from xllm.python.layers.npu.qwen3_5.moe import (
    NpuQwen3_5SparseMoEBlock,
    _NpuQwen3_5Experts,
)
from xllm.python.layers.npu.qwen3_5.moe_metadata import Qwen3_5MoeMetadata
from xllm.python.layers.npu.qwen3_5.moe_metadata_builder import Qwen3_5MoeMetadataBuilder
from xllm.python.layers.qwen3_5.decoder_layer import (
    get_qwen3_5_decoder_layer_class,
)
from xllm.python.layers.sequence_parallel_metadata import EpSpTokenLayout, SpMetadata
from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
)
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader
from xllm.python.models.qwen3_5 import Qwen3_5Config

kernels.gemma_rms_norm = _gemma_rms_norm
kernels.moe_fused_topk = MagicMock()
kernels.grouped_moe_bf16 = MagicMock()
kernels.mega_moe = MagicMock()
kernels.prepare_row_parallel_weight = MagicMock(side_effect=lambda weight: (weight, False))
distributed.all_gather_variable = MagicMock()
distributed.all_gather = MagicMock()
distributed.gather_dp_execution_tokens = MagicMock()
distributed.all_reduce_ = MagicMock()
distributed.tp_all_reduce = MagicMock()
distributed.moe_tp_all_reduce = MagicMock()
distributed.moe_ep_all_reduce = MagicMock()
distributed.broadcast_ = MagicMock()


def _gdn_config(**overrides: object) -> Qwen3_5Config:
    values: dict[str, object] = {
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "layer_types": ["linear_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "num_experts": 0,
        "num_experts_per_tok": 0,
        "moe_intermediate_size": 0,
        "shared_expert_intermediate_size": 0,
        "tp_size": 1,
        "world_size": 1,
        "moe_tp_size": 1,
    }
    values.update(overrides)
    return _config(**values)


def test_npu_decoder_factory_selects_privateuseone_backend() -> None:
    assert get_qwen3_5_decoder_layer_class("privateuseone") is NpuQwen3_5DecoderLayer
    assert NpuQwen3_5DecoderLayer.normalization_cls is NpuGemmaRMSNorm


def test_npu_gemma_rms_norm_uses_fused_residual_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_output = torch.zeros(2, 4, dtype=torch.bfloat16)
    expected_residual = torch.ones(2, 4, dtype=torch.bfloat16)
    fused_norm = MagicMock(return_value=(expected_output, expected_residual))
    monkeypatch.setattr(
        kernels,
        "fused_add_gemma_rms_norm",
        fused_norm,
        raising=False,
    )
    layer = NpuGemmaRMSNorm(
        4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    hidden = torch.randn(2, 4, dtype=torch.bfloat16)
    residual = torch.randn(2, 4, dtype=torch.bfloat16)

    actual_output, actual_residual = layer(hidden, residual)

    fused_norm.assert_called_once_with(
        hidden,
        residual,
        layer.weight,
        layer.eps,
    )
    assert layer.weight.dtype == torch.bfloat16
    assert actual_output is expected_output
    assert actual_residual is expected_residual


def test_npu_fused_attention_preparation_matches_unfused_after_reload() -> None:
    cfg = _config(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        tp_size=1,
        world_size=1,
        moe_tp_size=1,
        attention_bias=True,
    )

    def new_layer() -> NpuQwen3_5Attention:
        layer = NpuQwen3_5Attention(
            cfg,
            0,
            torch.float32,
            torch.device("cpu"),
            MagicMock(),
        )
        layer._fused_qkv_enabled = True
        return layer

    def checkpoint(offset: float) -> dict[str, torch.Tensor]:
        query = torch.arange(
            2 * cfg.n_heads * cfg.head_dim * cfg.hidden_size,
            dtype=torch.float32,
        ).view(2 * cfg.n_heads * cfg.head_dim, cfg.hidden_size)
        key = torch.arange(
            cfg.n_kv_heads * cfg.head_dim * cfg.hidden_size,
            dtype=torch.float32,
        ).view(cfg.n_kv_heads * cfg.head_dim, cfg.hidden_size)
        query_bias = torch.arange(
            2 * cfg.n_heads * cfg.head_dim,
            dtype=torch.float32,
        )
        key_bias = torch.arange(
            cfg.n_kv_heads * cfg.head_dim,
            dtype=torch.float32,
        )
        q_norm = torch.arange(cfg.head_dim, dtype=torch.float32)
        return {
            "q_proj.weight": query + offset,
            "k_proj.weight": key + offset,
            "v_proj.weight": key + offset + 100000,
            "q_proj.bias": query_bias + offset,
            "k_proj.bias": key_bias + offset,
            "v_proj.bias": key_bias + offset + 1000,
            "o_proj.weight": torch.full(
                (cfg.hidden_size, cfg.n_heads * cfg.head_dim),
                offset,
            ),
            "o_proj.bias": torch.full((cfg.hidden_size,), offset),
            "q_norm.weight": q_norm + offset,
            "k_norm.weight": q_norm + offset + 100,
        }

    def load(
        layer: NpuQwen3_5Attention,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        layer.load_weights(
            ScopedWeightLoader([_StateDict(tensors)]),
            context,
        )

    context = ParallelLoadContext(tp_rank=0, tp_size=1)
    layer = new_layer()
    initial_checkpoint = checkpoint(0.0)

    load(layer, initial_checkpoint)
    prepared_qkv = layer.qkv_proj.weight.detach().clone()
    assert layer.qkv_proj.bias is not None
    prepared_qkv_bias = layer.qkv_proj.bias.detach().clone()
    prepared_q_norm = layer.q_norm.weight.detach().clone()
    prepared_k_norm = layer.k_norm.weight.detach().clone()

    load(layer, initial_checkpoint)

    torch.testing.assert_close(layer.qkv_proj.weight, prepared_qkv)
    assert layer.qkv_proj.bias is not None
    torch.testing.assert_close(layer.qkv_proj.bias, prepared_qkv_bias)
    torch.testing.assert_close(layer.q_norm.weight, prepared_q_norm)
    torch.testing.assert_close(layer.k_norm.weight, prepared_k_norm)

    updated_checkpoint = checkpoint(10.0)
    incomplete_checkpoint = dict(updated_checkpoint)
    del incomplete_checkpoint["k_norm.weight"]
    with pytest.raises(KeyError, match="k_norm.weight"):
        load(layer, incomplete_checkpoint)

    load(layer, updated_checkpoint)
    fresh_layer = new_layer()
    kernels.prepare_row_parallel_weight.reset_mock()
    load(fresh_layer, updated_checkpoint)
    kernels.prepare_row_parallel_weight.assert_called_once()

    torch.testing.assert_close(layer.qkv_proj.weight, fresh_layer.qkv_proj.weight)
    assert layer.qkv_proj.bias is not None
    assert fresh_layer.qkv_proj.bias is not None
    torch.testing.assert_close(layer.qkv_proj.bias, fresh_layer.qkv_proj.bias)
    torch.testing.assert_close(layer.q_norm.weight, fresh_layer.q_norm.weight)
    torch.testing.assert_close(layer.k_norm.weight, fresh_layer.k_norm.weight)
    torch.testing.assert_close(
        layer.q_norm.weight,
        updated_checkpoint["q_norm.weight"] + 1.0,
    )
    torch.testing.assert_close(
        layer.k_norm.weight,
        updated_checkpoint["k_norm.weight"] + 1.0,
    )

    hidden = torch.arange(
        2 * cfg.hidden_size,
        dtype=torch.float32,
    ).view(2, cfg.hidden_size)
    torch.testing.assert_close(
        layer.qkv_proj(hidden),
        fresh_layer.qkv_proj(hidden),
    )
    unfused_qg = torch.nn.functional.linear(
        hidden,
        updated_checkpoint["q_proj.weight"],
        updated_checkpoint["q_proj.bias"],
    )
    unfused_query, unfused_gate = unfused_qg.view(
        2,
        cfg.n_heads,
        2 * cfg.head_dim,
    ).chunk(2, dim=-1)
    fused_qg = torch.nn.functional.linear(
        hidden,
        layer.qkv_proj.weight[: 2 * layer.q_size],
        layer.qkv_proj.bias[: 2 * layer.q_size],
    )
    fused_query, fused_gate = fused_qg.split(layer.q_size, dim=-1)

    torch.testing.assert_close(
        fused_query,
        unfused_query.reshape(2, layer.q_size),
    )
    torch.testing.assert_close(
        fused_gate,
        unfused_gate.reshape(2, layer.q_size),
    )


def test_npu_gdn_rejects_unsupported_rms_norm_epsilon() -> None:
    cfg = _gdn_config(rms_norm_eps=1e-5)

    with pytest.raises(NotImplementedError, match="fixed epsilon"):
        NpuQwen3_5GatedDeltaNet(
            cfg,
            0,
            torch.bfloat16,
            torch.device("cpu"),
        )


def test_npu_gdn_accepts_float32_supported_rms_norm_epsilon() -> None:
    float32_epsilon = float(torch.tensor(1e-6, dtype=torch.float32).item())
    cfg = _gdn_config(rms_norm_eps=float32_epsilon)

    layer = NpuQwen3_5GatedDeltaNet(
        cfg,
        0,
        torch.bfloat16,
        torch.device("cpu"),
    )

    assert layer.cfg.rms_norm_eps == float32_epsilon


@pytest.mark.parametrize(
    ("num_key_heads", "num_value_heads", "error"),
    (
        (3, 3, "power-of-two"),
        (1, 6, "at most four"),
    ),
)
def test_npu_gdn_rejects_geometry_unsupported_by_decode(
    num_key_heads: int,
    num_value_heads: int,
    error: str,
) -> None:
    cfg = _gdn_config(
        linear_num_key_heads=num_key_heads,
        linear_num_value_heads=num_value_heads,
        tp_size=1,
        world_size=1,
        moe_tp_size=1,
    )

    with pytest.raises(NotImplementedError, match=error):
        NpuQwen3_5GatedDeltaNet(
            cfg,
            0,
            torch.bfloat16,
            torch.device("cpu"),
        )


def test_npu_gdn_loads_native_conv_weight_layout() -> None:
    kernels.prepare_row_parallel_weight.reset_mock()
    cfg = _gdn_config()
    layer = NpuQwen3_5GatedDeltaNet(
        cfg,
        0,
        torch.bfloat16,
        torch.device("cpu"),
    )
    conv_dim = 768
    checkpoint_conv = torch.arange(
        conv_dim * 4,
        dtype=torch.float32,
    ).view(conv_dim, 1, 4)
    tensors = {
        "linear_attn.in_proj_qkv.weight": torch.zeros(conv_dim, 8),
        "linear_attn.in_proj_z.weight": torch.zeros(256, 8),
        "linear_attn.in_proj_b.weight": torch.zeros(2, 8),
        "linear_attn.in_proj_a.weight": torch.zeros(2, 8),
        "linear_attn.conv1d.weight": checkpoint_conv,
        "linear_attn.A_log": torch.zeros(2),
        "linear_attn.dt_bias": torch.zeros(2),
        "linear_attn.norm.weight": torch.zeros(128),
        "linear_attn.out_proj.weight": torch.zeros(8, 256),
    }

    layer.load_weights(
        ScopedWeightLoader([_StateDict(tensors)], "linear_attn."),
        ParallelLoadContext(tp_rank=0, tp_size=1),
    )

    assert layer.conv1d_weight.shape == (4, conv_dim)
    torch.testing.assert_close(
        layer.conv1d_weight,
        checkpoint_conv.squeeze(1).transpose(0, 1).to(torch.bfloat16),
    )
    kernels.prepare_row_parallel_weight.assert_called_once()


def test_npu_gdn_packs_rank_local_projection_weights() -> None:
    cfg = _gdn_config(
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        tp_size=2,
        tp_rank=1,
        world_size=2,
        moe_tp_size=2,
    )
    layer = NpuQwen3_5GatedDeltaNet(
        cfg,
        0,
        torch.bfloat16,
        torch.device("cpu"),
    )
    global_key_dim = 512
    global_value_dim = 512
    global_conv_dim = 2 * global_key_dim + global_value_dim
    qkv = torch.arange(global_conv_dim * 8, dtype=torch.float32).view(
        global_conv_dim,
        8,
    )
    z = torch.arange(global_value_dim * 8, dtype=torch.float32).view(
        global_value_dim,
        8,
    )
    b = torch.arange(4 * 8, dtype=torch.float32).view(4, 8)
    a = b + 1000
    tensors = {
        "linear_attn.in_proj_qkv.weight": qkv,
        "linear_attn.in_proj_z.weight": z,
        "linear_attn.in_proj_b.weight": b,
        "linear_attn.in_proj_a.weight": a,
        "linear_attn.conv1d.weight": torch.zeros(global_conv_dim, 1, 4),
        "linear_attn.A_log": torch.zeros(4),
        "linear_attn.dt_bias": torch.zeros(4),
        "linear_attn.norm.weight": torch.zeros(128),
        "linear_attn.out_proj.weight": torch.zeros(8, global_value_dim),
    }

    layer.load_weights(
        ScopedWeightLoader([_StateDict(tensors)], "linear_attn."),
        ParallelLoadContext(tp_rank=1, tp_size=2),
    )

    q, k, v = qkv.split((global_key_dim, global_key_dim, global_value_dim))
    expected_qkv = torch.cat((q.chunk(2)[1], k.chunk(2)[1], v.chunk(2)[1]))
    qkv_weight, z_weight, b_weight, a_weight = layer._input_projection_weight_views()
    torch.testing.assert_close(qkv_weight, expected_qkv.to(torch.bfloat16))
    torch.testing.assert_close(z_weight, z.chunk(2)[1].to(torch.bfloat16))
    torch.testing.assert_close(b_weight, b.chunk(2)[1].to(torch.bfloat16))
    torch.testing.assert_close(a_weight, a.chunk(2)[1].to(torch.bfloat16))
    assert qkv_weight.is_contiguous()
    assert z_weight.is_contiguous()
    assert qkv_weight.untyped_storage().data_ptr() == z_weight.untyped_storage().data_ptr()
    assert b_weight.untyped_storage().data_ptr() == a_weight.untyped_storage().data_ptr()


def test_npu_gdn_prefill_uses_packed_weight_views(monkeypatch) -> None:
    layer = NpuQwen3_5GatedDeltaNet(
        _gdn_config(),
        0,
        torch.bfloat16,
        torch.device("cpu"),
    )
    with torch.no_grad():
        layer.in_proj_qkvz.weight.copy_(
            (torch.arange(layer.in_proj_qkvz.weight.numel(), dtype=torch.float32) / 1024).view_as(
                layer.in_proj_qkvz.weight
            )
        )
        layer.in_proj_ba.weight.copy_(
            (torch.arange(layer.in_proj_ba.weight.numel(), dtype=torch.float32) / 1024).view_as(layer.in_proj_ba.weight)
        )
    qkvz_forward = MagicMock(wraps=layer.in_proj_qkvz.forward)
    ba_forward = MagicMock(wraps=layer.in_proj_ba.forward)
    monkeypatch.setattr(layer.in_proj_qkvz, "forward", qkvz_forward)
    monkeypatch.setattr(layer.in_proj_ba, "forward", ba_forward)

    prefill_outputs = layer._project_prefill_inputs(torch.ones(2, 8, dtype=torch.bfloat16))
    qkvz_forward.assert_not_called()
    ba_forward.assert_not_called()
    assert [tuple(output.shape) for output in prefill_outputs] == [
        (2, 768),
        (2, 2),
        (2, 2),
        (2, 2, 128),
    ]


def test_npu_decode_uses_mega_fusion_boundary(monkeypatch) -> None:
    batch_size = 33
    calls = MagicMock()
    mega = MagicMock(side_effect=lambda _qkv, z, *_args: torch.zeros_like(z))
    rms = MagicMock(side_effect=lambda output, *_args: output)
    calls.attach_mock(mega, "mega")
    calls.attach_mock(rms, "rms")
    monkeypatch.setattr(kernels, "mega_gdn_decode", mega, raising=False)
    monkeypatch.setattr(kernels, "causal_conv1d_decode", MagicMock(), raising=False)
    monkeypatch.setattr(kernels, "fused_sigmoid_gating_delta_rule_decode", MagicMock(), raising=False)
    monkeypatch.setattr(kernels, "rms_norm_gated", rms, raising=False)

    cfg = _config(
        hidden_size=8,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        shared_expert_intermediate_size=0,
        tp_size=1,
        world_size=1,
        moe_tp_size=1,
    )
    layer = NpuQwen3_5GatedDeltaNet(
        cfg,
        0,
        torch.bfloat16,
        torch.device("cpu"),
    )
    with torch.no_grad():
        layer.in_proj_qkvz.weight.zero_()
        layer.in_proj_ba.weight.zero_()
    qkvz_forward = MagicMock(wraps=layer.in_proj_qkvz.forward)
    ba_forward = MagicMock(wraps=layer.in_proj_ba.forward)
    monkeypatch.setattr(layer.in_proj_qkvz, "forward", qkvz_forward)
    monkeypatch.setattr(layer.in_proj_ba, "forward", ba_forward)
    layer.out_proj = torch.nn.Identity()
    state_indices = torch.arange(1, batch_size + 1, dtype=torch.int32)
    conv_state = torch.zeros(
        batch_size + 1,
        3,
        384,
        dtype=torch.bfloat16,
    )
    ssm_state = torch.zeros(
        batch_size + 1,
        1,
        128,
        128,
        dtype=torch.float32,
    )
    context = ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=None,
        layer_caches=[],
        execution_contexts={
            GdnMetadata: GdnDecodeMetadata(
                state_caches={0: GdnStateCache(conv_state, ssm_state)},
                read_state_indices=state_indices,
                write_state_indices=state_indices,
            )
        },
    )

    with forward_context(context):
        assert layer(torch.zeros(batch_size, 8, dtype=torch.bfloat16)).shape == (
            batch_size,
            128,
        )

    assert [call[0] for call in calls.mock_calls] == ["mega", "mega"]
    assert [call.args[0].shape[0] for call in mega.call_args_list] == [32, 1]
    torch.testing.assert_close(mega.call_args_list[0].args[9], state_indices[:32])
    torch.testing.assert_close(mega.call_args_list[0].args[10], state_indices[:32])
    torch.testing.assert_close(mega.call_args_list[1].args[9], state_indices[32:])
    torch.testing.assert_close(mega.call_args_list[1].args[10], state_indices[32:])
    qkvz_forward.assert_called_once()
    ba_forward.assert_called_once()
    kernels.causal_conv1d_decode.assert_not_called()
    kernels.fused_sigmoid_gating_delta_rule_decode.assert_not_called()
    rms.assert_not_called()


def test_npu_moe_loads_native_weight_order() -> None:
    cfg = _config(tp_rank=1, moe_tp_rank=1)
    layer = NpuQwen3_5DecoderLayer(
        cfg,
        0,
        torch.bfloat16,
        torch.device("cpu"),
        MagicMock(),
    )
    gate_up = torch.arange(
        cfg.num_experts * 2 * cfg.moe_intermediate_size * cfg.hidden_size,
        dtype=torch.float32,
    ).view(cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size)
    down = torch.arange(
        cfg.num_experts * cfg.hidden_size * cfg.moe_intermediate_size,
        dtype=torch.float32,
    ).view(cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size)
    tensors = {
        "mlp.gate.weight": torch.zeros(cfg.num_experts, cfg.hidden_size),
        "mlp.shared_expert_gate.weight": torch.zeros(1, cfg.hidden_size),
        "mlp.experts.gate_up_proj": gate_up,
        "mlp.experts.down_proj": down,
        "mlp.shared_expert.gate_proj.weight": torch.zeros(
            cfg.shared_expert_intermediate_size,
            cfg.hidden_size,
        ),
        "mlp.shared_expert.up_proj.weight": torch.zeros(
            cfg.shared_expert_intermediate_size,
            cfg.hidden_size,
        ),
        "mlp.shared_expert.down_proj.weight": torch.zeros(
            cfg.hidden_size,
            cfg.shared_expert_intermediate_size,
        ),
    }
    context = ParallelLoadContext(
        tp_rank=1,
        tp_size=2,
        moe_tp_rank=1,
        moe_tp_size=2,
    )

    layer.mlp.load_weights(
        ScopedWeightLoader([_StateDict(tensors)], "mlp."),
        context,
    )
    kernels.prepare_row_parallel_weight.reset_mock()
    layer.mlp.process_weights_after_loading()

    gate, up = gate_up.chunk(2, dim=1)
    local_gate = gate.chunk(2, dim=1)[1]
    local_up = up.chunk(2, dim=1)[1]
    expected_w13 = (
        torch.cat(
            (local_gate, local_up),
            dim=1,
        )
        .transpose(1, 2)
        .contiguous()
    )
    expected_w2 = down.chunk(2, dim=2)[1].transpose(1, 2).contiguous()
    assert isinstance(layer.mlp, NpuQwen3_5SparseMoEBlock)
    torch.testing.assert_close(
        layer.mlp.experts.w13,
        expected_w13.to(torch.bfloat16),
    )
    torch.testing.assert_close(
        layer.mlp.experts.w2,
        expected_w2.to(torch.bfloat16),
    )
    kernels.prepare_row_parallel_weight.assert_called_once()


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("lv2", [False, True])
def test_npu_shared_expert_loads_full_checkpoint_on_each_rank(tp_rank: int, lv2: bool) -> None:
    cfg = _config(
        tp_rank=tp_rank,
        dp_size=2,
        world_size=4,
        moe_tp_size=1,
        ep_size=4,
        ep_rank=tp_rank,
        sequence_parallel_enabled=not lv2,
    )
    policy = Eplv2CommPolicy(32) if lv2 else None
    block = NpuQwen3_5SparseMoEBlock(cfg, torch.bfloat16, torch.device("cpu"), comm_policy=policy)
    gate = (
        torch.arange(cfg.shared_expert_intermediate_size * cfg.hidden_size)
        .reshape(cfg.shared_expert_intermediate_size, cfg.hidden_size)
        .to(torch.bfloat16)
    )
    up = gate + 10
    down = gate.t().contiguous()
    tensors = {
        "gate.weight": torch.zeros(cfg.num_experts, cfg.hidden_size),
        "shared_expert_gate.weight": torch.zeros(1, cfg.hidden_size),
        "experts.gate_up_proj": torch.zeros(cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size),
        "experts.down_proj": torch.zeros(cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size),
        "shared_expert.gate_proj.weight": gate,
        "shared_expert.up_proj.weight": up,
        "shared_expert.down_proj.weight": down,
    }
    context = ParallelLoadContext(tp_rank=tp_rank, tp_size=2, ep_rank=tp_rank, ep_size=4)
    block.load_weights(ScopedWeightLoader([_StateDict(tensors)]), context)
    assert block.shared_expert.gate_up_proj.tp_size == 1
    assert block.shared_expert.down_proj.tp_size == 1
    assert context.tp_size == 2 and context.tp_rank == tp_rank
    torch.testing.assert_close(block.shared_expert.gate_up_proj.weight, torch.cat((gate, up)))
    torch.testing.assert_close(block.shared_expert.down_proj.weight, down)


def test_npu_gdn_uses_npu_prefill_fusion_boundary(monkeypatch) -> None:
    calls = MagicMock()
    mega = MagicMock(return_value=torch.zeros(1, 1, 128, dtype=torch.bfloat16))
    rms = MagicMock(side_effect=lambda output, *_args: output)
    for name, mock in (("mega", mega), ("rms", rms)):
        calls.attach_mock(mock, name)
    monkeypatch.setattr(kernels, "mega_gdn_prefill", mega, raising=False)
    monkeypatch.setattr(kernels, "rms_norm_gated", rms, raising=False)
    for old_kernel in (
        "causal_conv1d_qkv_prefill",
        "fused_gdn_gating",
        "chunk_gated_delta_rule",
    ):
        monkeypatch.setattr(kernels, old_kernel, MagicMock(), raising=False)

    cfg = _config(
        hidden_size=8,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        shared_expert_intermediate_size=0,
        tp_size=1,
        world_size=1,
        moe_tp_size=1,
    )
    layer = NpuQwen3_5GatedDeltaNet(
        cfg,
        0,
        torch.bfloat16,
        torch.device("cpu"),
    )
    with torch.no_grad():
        layer.in_proj_qkvz.weight.zero_()
        layer.in_proj_ba.weight.zero_()
    layer.out_proj = torch.nn.Identity()
    conv_state = torch.zeros(3, 3, 384, dtype=torch.bfloat16)
    ssm_state = torch.zeros(3, 1, 128, 128, dtype=torch.float32)
    context = ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=None,
        layer_caches=[],
        execution_contexts={
            GdnMetadata: GdnPrefillMetadata(
                state_caches={0: GdnStateCache(conv_state, ssm_state)},
                conv_read_indices=torch.tensor([1], dtype=torch.int32),
                conv_write_indices=torch.tensor([2], dtype=torch.int32),
                ssm_read_indices=torch.tensor([1], dtype=torch.int32),
                ssm_write_indices=torch.tensor([2], dtype=torch.int32),
                cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
                num_matrices=1,
            )
        },
    )
    with forward_context(context):
        output = layer(torch.zeros(1, 8, dtype=torch.bfloat16))

    assert output.shape == (1, 128)
    assert [call[0] for call in calls.mock_calls] == ["mega"]
    kernels.causal_conv1d_qkv_prefill.assert_not_called()
    kernels.fused_gdn_gating.assert_not_called()
    kernels.chunk_gated_delta_rule.assert_not_called()
    args = mega.call_args.args
    torch.testing.assert_close(args[8], torch.tensor([1], dtype=torch.int32))
    torch.testing.assert_close(args[9], torch.tensor([2], dtype=torch.int32))
    torch.testing.assert_close(args[10], torch.tensor([1], dtype=torch.int32))
    torch.testing.assert_close(args[11], torch.tensor([2], dtype=torch.int32))
    assert args[15] == 1


def test_npu_decode_passes_native_weight_and_cache_to_tilelang(
    monkeypatch,
) -> None:
    output = torch.zeros(1, 24)
    kernel = MagicMock(return_value=output)
    tilelang_wrapper = types.ModuleType("xllm.python.kernels_npu.tilelang.causal_conv1d_decode")
    tilelang_wrapper.DIM_PER_CORE = 2048
    tilelang_wrapper._build_decode_kernel_jit = MagicMock(return_value=kernel)
    monkeypatch.setitem(
        sys.modules,
        "xllm.python.kernels_npu.tilelang.causal_conv1d_decode",
        tilelang_wrapper,
    )
    value = torch.zeros(1, 24)
    weight = torch.zeros(4, 24)
    conv_state = torch.zeros(2, 3, 24)
    state_indices = torch.tensor([1], dtype=torch.int32)

    actual = npu_causal_conv1d_decode(
        value,
        weight,
        conv_state,
        state_indices,
    )

    assert actual is output
    args = kernel.call_args.args
    assert args[1] is weight
    assert args[2] is conv_state


def test_npu_moe_rejects_non_bf16_weights() -> None:
    with pytest.raises(NotImplementedError, match="support BF16 only"):
        NpuQwen3_5SparseMoEBlock(
            _config(),
            torch.float16,
            torch.device("cpu"),
        )


def test_npu_moe_partitions_expert_axis_for_ep() -> None:
    cfg = _config(
        tp_size=4,
        tp_rank=2,
        world_size=4,
        moe_tp_size=1,
        moe_tp_rank=0,
        ep_size=4,
        ep_rank=2,
        num_attention_heads=4,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
    )
    layer = NpuQwen3_5DecoderLayer(
        cfg,
        0,
        torch.bfloat16,
        torch.device("cpu"),
        MagicMock(),
    )
    gate_up = torch.arange(
        cfg.num_experts * 2 * cfg.moe_intermediate_size * cfg.hidden_size,
        dtype=torch.float32,
    ).view(cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size)
    down = torch.arange(
        cfg.num_experts * cfg.hidden_size * cfg.moe_intermediate_size,
        dtype=torch.float32,
    ).view(cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size)
    tensors = {
        "mlp.gate.weight": torch.zeros(cfg.num_experts, cfg.hidden_size),
        "mlp.shared_expert_gate.weight": torch.zeros(1, cfg.hidden_size),
        "mlp.experts.gate_up_proj": gate_up,
        "mlp.experts.down_proj": down,
        "mlp.shared_expert.gate_proj.weight": torch.zeros(
            cfg.shared_expert_intermediate_size,
            cfg.hidden_size,
        ),
        "mlp.shared_expert.up_proj.weight": torch.zeros(
            cfg.shared_expert_intermediate_size,
            cfg.hidden_size,
        ),
        "mlp.shared_expert.down_proj.weight": torch.zeros(
            cfg.hidden_size,
            cfg.shared_expert_intermediate_size,
        ),
    }
    context = ParallelLoadContext(
        tp_rank=2,
        tp_size=4,
        moe_tp_rank=0,
        moe_tp_size=1,
        ep_rank=2,
        ep_size=4,
    )

    layer.mlp.load_weights(
        ScopedWeightLoader([_StateDict(tensors)], "mlp."),
        context,
    )

    expected_gate_up = gate_up.narrow(0, 4, 2).transpose(1, 2).contiguous()
    expected_down = down.narrow(0, 4, 2).transpose(1, 2).contiguous()
    torch.testing.assert_close(
        layer.mlp.experts.w13,
        expected_gate_up.to(torch.bfloat16),
    )
    torch.testing.assert_close(
        layer.mlp.experts.w2,
        expected_down.to(torch.bfloat16),
    )


def test_npu_moe_uses_moe_tp_collective(monkeypatch) -> None:
    cfg = _config()
    block = NpuQwen3_5SparseMoEBlock(
        cfg,
        torch.bfloat16,
        torch.device("cpu"),
    )
    block.experts.reduce_results = True
    block.experts.gate = _ConstantModule(torch.zeros(3, cfg.num_experts, dtype=torch.float32))
    monkeypatch.setattr(
        kernels,
        "moe_fused_topk",
        MagicMock(
            return_value=(
                torch.ones(3, cfg.num_experts_per_tok),
                torch.zeros(
                    3,
                    cfg.num_experts_per_tok,
                    dtype=torch.int32,
                ),
            )
        ),
    )
    grouped_output = torch.ones(3, cfg.hidden_size)
    grouped_moe = MagicMock(return_value=grouped_output)
    monkeypatch.setattr(kernels, "grouped_moe_bf16", grouped_moe, raising=False)
    distributed.moe_tp_all_reduce.reset_mock()
    distributed.moe_ep_all_reduce.reset_mock()

    output = block.experts(torch.zeros(3, cfg.hidden_size))

    assert output is grouped_output
    distributed.moe_tp_all_reduce.assert_called_once_with(grouped_output)
    distributed.moe_ep_all_reduce.assert_not_called()
    grouped_moe.assert_called_once()


def test_npu_moe_uses_moe_ep_collective(monkeypatch) -> None:
    cfg = _config(
        tp_size=4,
        world_size=4,
        moe_tp_size=1,
        ep_size=4,
        ep_rank=1,
        num_attention_heads=4,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
    )
    block = NpuQwen3_5SparseMoEBlock(
        cfg,
        torch.bfloat16,
        torch.device("cpu"),
    )
    block.experts.gate = _ConstantModule(torch.zeros(3, cfg.num_experts, dtype=torch.float32))
    monkeypatch.setattr(
        kernels,
        "moe_fused_topk",
        MagicMock(
            return_value=(
                torch.ones(3, cfg.num_experts_per_tok),
                torch.zeros(
                    3,
                    cfg.num_experts_per_tok,
                    dtype=torch.int32,
                ),
            )
        ),
    )
    grouped_output = torch.ones(3, cfg.hidden_size)
    monkeypatch.setattr(
        kernels,
        "grouped_moe_bf16",
        MagicMock(return_value=grouped_output),
        raising=False,
    )
    distributed.moe_tp_all_reduce.reset_mock()
    distributed.moe_ep_all_reduce.reset_mock()

    output = block.experts(torch.zeros(3, cfg.hidden_size))

    assert output is grouped_output
    distributed.moe_tp_all_reduce.assert_not_called()
    distributed.moe_ep_all_reduce.assert_called_once_with(grouped_output)


def test_npu_moe_uses_both_collectives_for_partial_ep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(
        tp_size=4,
        world_size=4,
        moe_tp_size=2,
        ep_size=2,
        ep_rank=1,
        num_attention_heads=4,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
    )
    block = NpuQwen3_5SparseMoEBlock(
        cfg,
        torch.bfloat16,
        torch.device("cpu"),
    )
    block.experts.gate = _ConstantModule(torch.zeros(3, cfg.num_experts, dtype=torch.bfloat16))
    monkeypatch.setattr(
        kernels,
        "moe_fused_topk",
        MagicMock(
            return_value=(
                torch.ones(
                    3,
                    cfg.num_experts_per_tok,
                    dtype=torch.bfloat16,
                ),
                torch.zeros(
                    3,
                    cfg.num_experts_per_tok,
                    dtype=torch.int32,
                ),
            )
        ),
    )
    grouped_output = torch.ones(
        3,
        cfg.hidden_size,
        dtype=torch.bfloat16,
    )
    monkeypatch.setattr(
        kernels,
        "grouped_moe_bf16",
        MagicMock(return_value=grouped_output),
        raising=False,
    )
    distributed.moe_tp_all_reduce.reset_mock()
    distributed.moe_ep_all_reduce.reset_mock()

    output = block.experts(torch.zeros(3, cfg.hidden_size, dtype=torch.bfloat16))

    assert output is grouped_output
    distributed.moe_tp_all_reduce.assert_called_once_with(grouped_output)
    distributed.moe_ep_all_reduce.assert_called_once_with(grouped_output)


def test_npu_moe_requires_resolved_sequence_parallel_state() -> None:
    cfg = _config()
    del cfg.sequence_parallel_enabled
    with pytest.raises(AttributeError, match="sequence_parallel_enabled"):
        NpuQwen3_5SparseMoEBlock(cfg, torch.bfloat16, torch.device("cpu"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"moe_tp_size": 2, "ep_size": 2}, "requires direct EP topology"),
        ({"ep_size": 2}, "requires direct EP topology"),
        ({"ep_rank": 1}, "EP rank mapping"),
        ({"dp_rank": 1, "ep_rank": 1}, "EP rank mapping"),
    ],
)
def test_npu_sp_experts_validate_topology_at_construction(overrides: dict[str, object], message: str) -> None:
    cfg = _config(
        **{
            "tp_size": 2,
            "dp_size": 2,
            "world_size": 4,
            "moe_tp_size": 1,
            "ep_size": 4,
            "sequence_parallel_enabled": True,
            **overrides,
        }
    )
    with pytest.raises(ValueError, match=message):
        _NpuQwen3_5Experts(cfg, torch.bfloat16, torch.device("cpu"), reduce_results=True)


@pytest.mark.parametrize("dp_rank", range(2))
@pytest.mark.parametrize("tp_rank", range(2))
def test_npu_sp_experts_accept_direct_ep_rank_mapping(dp_rank: int, tp_rank: int) -> None:
    cfg = _config(
        tp_size=2,
        tp_rank=tp_rank,
        dp_size=2,
        dp_rank=dp_rank,
        world_size=4,
        moe_tp_size=1,
        ep_size=4,
        ep_rank=dp_rank * 2 + tp_rank,
        sequence_parallel_enabled=True,
    )
    experts = _NpuQwen3_5Experts(cfg, torch.bfloat16, torch.device("cpu"), reduce_results=True)
    assert experts.ep_rank == dp_rank * 2 + tp_rank


def test_npu_sp_experts_require_reduction_at_construction() -> None:
    cfg = _config(dp_size=2, world_size=4, moe_tp_size=1, ep_size=4, sequence_parallel_enabled=True)
    with pytest.raises(ValueError, match="must reduce their partial outputs"):
        _NpuQwen3_5Experts(cfg, torch.bfloat16, torch.device("cpu"), reduce_results=False)


def test_npu_moe_uses_direct_ep_when_capacity_exceeds_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(
        tp_size=2,
        tp_rank=0,
        dp_size=2,
        dp_rank=0,
        world_size=4,
        moe_tp_size=1,
        moe_tp_rank=0,
        ep_size=4,
        ep_rank=0,
        sequence_parallel_enabled=True,
        enable_mega_moe=True,
        mega_moe_context=torch.zeros(1, dtype=torch.int32),
        mega_moe_ccl_buffer_size=1024,
        mega_moe_num_max_tokens_per_rank=1,
    )
    block = NpuQwen3_5SparseMoEBlock(
        cfg,
        torch.bfloat16,
        torch.device("cpu"),
    )
    hidden = torch.ones(2, cfg.hidden_size, dtype=torch.bfloat16)
    expected = torch.full_like(hidden, 7)
    direct = MagicMock(return_value=expected)
    monkeypatch.setattr(block.experts, "_forward_ep_level1_for_sp", direct)
    mega_moe = MagicMock()
    monkeypatch.setattr(kernels, "mega_moe", mega_moe, raising=False)
    context = ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=SimpleNamespace(dp_execution_token_counts=(3, 3)),
        layer_caches=[],
        execution_contexts={
            SpMetadata: SpMetadata.from_execution_rows(
                3,
                2,
                0,
                ep_layout=EpSpTokenLayout.from_token_counts(
                    (3, 3), (3, 3), tp_size=2, tp_rank=0, dp_size=2, dp_rank=0, compacted=True
                ),
            ),
            Qwen3_5MoeMetadata: Qwen3_5MoeMetadata(
                active_token_mask=torch.ones(2, dtype=torch.int8),
            ),
        },
    )

    with forward_context(context):
        output = block.experts(hidden, sp_metadata=context.execution_contexts[SpMetadata])

    assert output is expected
    direct.assert_called_once_with(hidden, context.execution_contexts[SpMetadata])
    mega_moe.assert_not_called()


@pytest.mark.parametrize(("sp", "lv2"), [(False, False), (False, True), (True, False), (True, True)])
def test_npu_shared_stream_is_initialized_only_for_lv2_sp(monkeypatch: pytest.MonkeyPatch, sp: bool, lv2: bool) -> None:
    from xllm.python.layers.npu.qwen3_5 import moe as qwen_moe

    cfg = _config(tp_size=2, ep_size=2, moe_tp_size=1, sequence_parallel_enabled=sp)
    policy = Eplv2CommPolicy(mc2_capacity=86) if lv2 else None
    stream = object()
    stream_factory = MagicMock(return_value=stream)
    monkeypatch.setattr(qwen_moe, "shared_expert_stream", stream_factory)
    monkeypatch.setattr(distributed, "moe_ep_hccl_info", MagicMock(return_value=("test-ep", 0, 2)), raising=False)
    block = NpuQwen3_5SparseMoEBlock(cfg, torch.bfloat16, torch.device("cpu"), comm_policy=policy)
    stream_factory.assert_not_called()
    assert block._shared_stream is None

    block.process_weights_after_loading()

    if sp and lv2:
        stream_factory.assert_called_once_with(block.experts.w13.device)
        assert block._shared_stream is stream
    else:
        stream_factory.assert_not_called()
        assert block._shared_stream is None


def test_npu_lv2_persistent_mask_reaches_mc2_without_active_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config(tp_size=2, tp_rank=1, ep_size=2, ep_rank=1, moe_tp_size=1)
    policy = Eplv2CommPolicy(mc2_capacity=2)
    monkeypatch.setattr(distributed, "moe_ep_hccl_info", MagicMock(return_value=("test-ep", 1, 2)), raising=False)
    block = NpuQwen3_5SparseMoEBlock(cfg, torch.bfloat16, torch.device("cpu"), comm_policy=policy)
    block.process_weights_after_loading()
    hidden = torch.ones(2, cfg.hidden_size, dtype=torch.bfloat16)
    route = MagicMock(
        return_value=(
            torch.ones(2, cfg.num_experts_per_tok),
            torch.zeros(2, cfg.num_experts_per_tok, dtype=torch.int32),
        )
    )
    monkeypatch.setattr(block.experts, "_route", route)
    mc2 = MagicMock(return_value=hidden)
    alltoall = MagicMock()
    monkeypatch.setattr(kernels, "ep_moe_bf16", mc2, raising=False)
    monkeypatch.setattr(kernels, "ep_moe_bf16_alltoall", alltoall, raising=False)
    builder = Qwen3_5MoeMetadataBuilder(2, 1, 1, 0, mask_dtype=torch.bool)
    batch = SimpleNamespace(
        num_tokens=3,
        num_tokens_after_padding=4,
        input_ids=torch.zeros(4, dtype=torch.int32),
        is_padding=torch.arange(4) >= 3,
        is_dummy=False,
    )
    metadata = SimpleNamespace()
    persistent = builder.allocate_persistent(batch, metadata)
    mask = persistent.active_token_mask
    context = ForwardContext(
        attention_backend=SimpleNamespace(),
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=[],
        execution_state=AclGraphExecutionState({}),
        execution_contexts={Qwen3_5MoeMetadata: persistent},
    )
    for rows in (3, 1, 3):
        batch.num_tokens = rows
        batch.is_padding = torch.arange(4) >= rows
        builder.update_persistent(persistent, batch, metadata)
        assert persistent.active_rows is None
        assert persistent.active_token_mask is mask
        with forward_context(context):
            NpuQwen3_5DecoderLayer.validate_ep_level2(policy, force_mc2=False)
            torch.testing.assert_close(block.experts(hidden), hidden)
        assert mc2.call_args.args[-1] is mask
        assert mc2.call_args.args[5] == "test-ep"
        assert mask.tolist() == ([True, False] if rows == 3 else [False, False])
    assert mc2.call_count == 3
    alltoall.assert_not_called()


@pytest.mark.parametrize("physical_rows", [0, 1])
def test_npu_lv2_alltoall_forwards_empty_source(monkeypatch: pytest.MonkeyPatch, physical_rows: int) -> None:
    cfg = _config(moe_tp_size=1, ep_size=2)
    experts = _NpuQwen3_5Experts(
        cfg, torch.bfloat16, torch.device("cpu"), reduce_results=True, comm_policy=Eplv2CommPolicy(mc2_capacity=0)
    )
    hidden = torch.ones(physical_rows, cfg.hidden_size, dtype=torch.bfloat16)
    route = MagicMock(
        return_value=(
            torch.ones(physical_rows, cfg.num_experts_per_tok),
            torch.zeros(physical_rows, cfg.num_experts_per_tok, dtype=torch.int32),
        )
    )
    monkeypatch.setattr(experts, "_route", route)
    alltoall = MagicMock(return_value=hidden[:0])
    monkeypatch.setattr(kernels, "ep_moe_bf16_alltoall", alltoall, raising=False)
    context = ForwardContext(
        SimpleNamespace(),
        torch.device("cpu"),
        SimpleNamespace(),
        [],
        execution_contexts={Qwen3_5MoeMetadata: Qwen3_5MoeMetadata(torch.zeros(1, dtype=torch.bool), active_rows=0)},
    )
    with forward_context(context):
        output = experts(hidden)
    alltoall.assert_called_once()
    assert [tensor.shape for tensor in alltoall.call_args.args[:3]] == [
        (0, cfg.hidden_size),
        (0, cfg.num_experts_per_tok),
        (0, cfg.num_experts_per_tok),
    ]
    torch.testing.assert_close(output, torch.zeros_like(hidden))


@pytest.mark.parametrize(
    ("rows", "force_mc2", "error"),
    [
        (None, False, "metadata is unavailable"),
        (4, True, "source rows 4 exceed capacity 3"),
        (4, False, None),
        (3, True, None),
    ],
)
def test_npu_lv2_transport_admission(rows: int | None, force_mc2: bool, error: str | None) -> None:
    metadata = (
        {} if rows is None else {Qwen3_5MoeMetadata: Qwen3_5MoeMetadata(torch.ones(rows, dtype=torch.bool), rows)}
    )
    context = ForwardContext(SimpleNamespace(), torch.device("cpu"), SimpleNamespace(), [], execution_contexts=metadata)
    with forward_context(context):
        if error is not None:
            with pytest.raises(RuntimeError if rows is None else ValueError, match=error):
                NpuQwen3_5DecoderLayer.validate_ep_level2(Eplv2CommPolicy(mc2_capacity=3), force_mc2=force_mc2)
        else:
            NpuQwen3_5DecoderLayer.validate_ep_level2(Eplv2CommPolicy(mc2_capacity=3), force_mc2=force_mc2)


@pytest.mark.parametrize(("raw_rows", "graph"), [(3, False), (0, False), (3, True), (0, True)])
def test_npu_shared_expert_uses_valid_sp_prefix(monkeypatch: pytest.MonkeyPatch, raw_rows: int, graph: bool) -> None:
    def unexpected_derivation(*args: object, **kwargs: object) -> None:
        pytest.fail("MoE layers must consume the SP builder's prepared layout instead of deriving it")

    cfg = _config(
        tp_size=2,
        tp_rank=1,
        dp_size=2,
        dp_rank=0,
        world_size=4,
        moe_tp_size=1,
        ep_size=4,
        ep_rank=1,
        sequence_parallel_enabled=True,
    )
    block = NpuQwen3_5SparseMoEBlock(cfg, torch.bfloat16, torch.device("cpu"))
    hidden = torch.arange(2 * cfg.hidden_size, dtype=torch.float32).view(2, cfg.hidden_size) / 100
    routed = torch.full_like(hidden, 7)
    direct = MagicMock(return_value=routed)
    shared = MagicMock(side_effect=lambda value: value * 2)
    gate = MagicMock(side_effect=lambda value: value.new_zeros((value.shape[0], 1)))
    monkeypatch.setattr(block.experts, "_forward_ep_level1_for_sp", direct)
    monkeypatch.setattr(block.shared_expert, "forward", shared)
    monkeypatch.setattr(block.shared_expert_gate, "forward", gate)
    context = ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=SimpleNamespace(
            dp_execution_token_counts=(4, 4),
            raw_dp_execution_token_counts=(raw_rows, 4),
        ),
        layer_caches=[],
        execution_state=AclGraphExecutionState({}) if graph else None,
        execution_contexts={
            SpMetadata: SpMetadata.from_execution_rows(
                4,
                2,
                1,
                ep_layout=EpSpTokenLayout.from_token_counts(
                    (4, 4), (raw_rows, 4), tp_size=2, tp_rank=1, dp_size=2, dp_rank=0, compacted=not graph
                ),
            ),
            Qwen3_5MoeMetadata: Qwen3_5MoeMetadata(
                active_token_mask=torch.ones(2, dtype=torch.int8),
            ),
        },
    )
    monkeypatch.setattr(EpSpTokenLayout, "from_token_counts", unexpected_derivation)

    with forward_context(context):
        with pytest.raises(ValueError, match="local SP hidden rows"):
            block(torch.zeros(3, cfg.hidden_size))
        output = block(hidden)

    direct.assert_called_once()
    assert direct.call_args.args[0] is hidden
    assert direct.call_args.args[1] is context.execution_contexts[SpMetadata]
    assert context.layer_shared_cache == {}
    valid_rows = max(0, raw_rows - 2)
    computed_rows = 2 if graph else valid_rows
    expected = routed.clone()
    expected[:computed_rows] += hidden[:computed_rows]
    torch.testing.assert_close(output, expected)
    if computed_rows:
        shared.assert_called_once()
        gate.assert_called_once()
        torch.testing.assert_close(shared.call_args.args[0], hidden[:computed_rows])
        torch.testing.assert_close(gate.call_args.args[0], hidden[:computed_rows])
    else:
        shared.assert_not_called()
        gate.assert_not_called()


def test_npu_routed_expert_computes_only_valid_sp_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config(
        tp_size=2,
        tp_rank=1,
        dp_size=2,
        dp_rank=0,
        world_size=4,
        moe_tp_size=1,
        ep_size=4,
        ep_rank=1,
        sequence_parallel_enabled=True,
    )
    block = NpuQwen3_5SparseMoEBlock(cfg, torch.bfloat16, torch.device("cpu"))
    hidden = torch.ones(2, cfg.hidden_size, dtype=torch.bfloat16)
    hidden[1].fill_(-99)
    gathered = torch.arange(8 * cfg.hidden_size).view(8, cfg.hidden_size).to(torch.bfloat16)

    def gather(value: torch.Tensor, counts: list[int], rank: int, group_name: str) -> torch.Tensor:
        assert counts == [2, 1, 3, 2] and rank == 1 and group_name == "moe_ep"
        torch.testing.assert_close(value, hidden[:1])
        return gathered

    gate = MagicMock(return_value=torch.zeros(8, cfg.num_experts))
    topk = MagicMock(
        return_value=(
            torch.ones(8, cfg.num_experts_per_tok),
            torch.zeros(8, cfg.num_experts_per_tok, dtype=torch.int32),
        )
    )
    grouped = MagicMock(side_effect=lambda value, *args: value * 2)
    monkeypatch.setattr(distributed, "all_gather_variable", gather)
    monkeypatch.setattr(
        distributed, "reduce_scatter", lambda value, world_size, group_name: value.chunk(world_size)[1], raising=False
    )
    monkeypatch.setattr(block.experts.gate, "forward", gate)
    monkeypatch.setattr(kernels, "moe_fused_topk", topk)
    monkeypatch.setattr(kernels, "grouped_moe_bf16", grouped)
    context = ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=SimpleNamespace(dp_execution_token_counts=(3, 5), raw_dp_execution_token_counts=(3, 5)),
        layer_caches=[],
        execution_contexts={
            SpMetadata: SpMetadata.from_execution_rows(
                3,
                2,
                1,
                ep_layout=EpSpTokenLayout.from_token_counts(
                    (3, 5), (3, 5), tp_size=2, tp_rank=1, dp_size=2, dp_rank=0, compacted=True
                ),
            ),
            Qwen3_5MoeMetadata: Qwen3_5MoeMetadata(
                active_token_mask=torch.tensor([1, 0], dtype=torch.int8),
            ),
        },
    )

    with forward_context(context):
        output = block.experts(hidden, sp_metadata=context.execution_contexts[SpMetadata])

    gate.assert_called_once_with(gathered)
    grouped.assert_called_once()
    assert grouped.call_args.args[0] is gathered
    torch.testing.assert_close(output[:1], gathered[2:3] * 2)
    torch.testing.assert_close(output[1:], torch.zeros_like(hidden[1:]))


def test_npu_moe_graph_dp_gather_uses_fixed_shape(monkeypatch) -> None:
    cfg = _config(
        tp_size=1,
        dp_size=2,
        world_size=2,
        moe_tp_size=2,
        ep_size=1,
    )
    block = NpuQwen3_5SparseMoEBlock(
        cfg,
        torch.bfloat16,
        torch.device("cpu"),
    )
    gathered = torch.zeros(8, cfg.hidden_size)
    grouped_output = torch.arange(
        8 * cfg.hidden_size,
        dtype=torch.float32,
    ).view(8, cfg.hidden_size)
    block.experts.gate = _ConstantModule(torch.zeros(8, cfg.num_experts, dtype=torch.float32))
    distributed.gather_dp_execution_tokens.reset_mock()
    distributed.gather_dp_execution_tokens.return_value = gathered, 0
    monkeypatch.setattr(
        kernels,
        "moe_fused_topk",
        MagicMock(
            return_value=(
                torch.ones(8, cfg.num_experts_per_tok),
                torch.zeros(
                    8,
                    cfg.num_experts_per_tok,
                    dtype=torch.int32,
                ),
            )
        ),
    )
    monkeypatch.setattr(
        kernels,
        "grouped_moe_bf16",
        MagicMock(return_value=grouped_output),
        raising=False,
    )
    metadata = SimpleNamespace(
        dp_execution_token_counts=(4, 4),
        dp_is_decode=(1, 1),
        is_prefill=False,
        is_chunked_prefill=False,
    )
    context = ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=[],
        execution_state=AclGraphExecutionState({}),
    )

    with forward_context(context):
        local_input = torch.zeros(4, cfg.hidden_size)
        output = block.experts(local_input)

    assert output.shape == (4, cfg.hidden_size)
    torch.testing.assert_close(output, grouped_output[:4])
    distributed.gather_dp_execution_tokens.assert_called_once_with(
        local_input,
        (4, 4),
        0,
    )


def test_npu_moe_uses_uneven_execution_counts(monkeypatch) -> None:
    cfg = _config(
        tp_size=1,
        dp_size=2,
        dp_rank=1,
        world_size=2,
        moe_tp_size=2,
        ep_size=1,
    )
    block = NpuQwen3_5SparseMoEBlock(
        cfg,
        torch.bfloat16,
        torch.device("cpu"),
    )
    gathered = torch.zeros(7, cfg.hidden_size)
    grouped_output = torch.arange(
        7 * cfg.hidden_size,
        dtype=torch.float32,
    ).view(7, cfg.hidden_size)
    block.experts.gate = _ConstantModule(torch.zeros(7, cfg.num_experts, dtype=torch.float32))
    distributed.gather_dp_execution_tokens.reset_mock()
    distributed.gather_dp_execution_tokens.return_value = gathered, 3
    monkeypatch.setattr(
        kernels,
        "moe_fused_topk",
        MagicMock(
            return_value=(
                torch.ones(7, cfg.num_experts_per_tok),
                torch.zeros(
                    7,
                    cfg.num_experts_per_tok,
                    dtype=torch.int32,
                ),
            )
        ),
    )
    monkeypatch.setattr(
        kernels,
        "grouped_moe_bf16",
        MagicMock(return_value=grouped_output),
        raising=False,
    )
    metadata = SimpleNamespace(
        dp_execution_token_counts=(3, 4),
        dp_is_decode=(1, 1),
        is_prefill=False,
        is_chunked_prefill=False,
    )
    context = ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=[],
    )

    with forward_context(context):
        local_input = torch.zeros(4, cfg.hidden_size)
        output = block.experts(local_input)

    torch.testing.assert_close(output, grouped_output[3:7])
    distributed.gather_dp_execution_tokens.assert_called_once_with(
        local_input,
        (3, 4),
        1,
    )


@pytest.mark.parametrize(
    ("dp_rank", "token_counts", "expected_start"),
    (
        (1, (3, 0), 3),
        (0, (0, 5), 0),
    ),
)
def test_npu_moe_empty_dp_rank_uses_execution_counts(
    monkeypatch,
    dp_rank: int,
    token_counts: tuple[int, int],
    expected_start: int,
) -> None:
    cfg = _config(
        tp_size=1,
        dp_size=2,
        dp_rank=dp_rank,
        world_size=2,
        moe_tp_size=2,
        ep_size=1,
    )
    block = NpuQwen3_5SparseMoEBlock(
        cfg,
        torch.bfloat16,
        torch.device("cpu"),
    )
    execution_counts = tuple(max(count, 1) for count in token_counts)
    gathered_rows = sum(execution_counts)
    gathered = torch.zeros(gathered_rows, cfg.hidden_size)
    grouped_output = torch.arange(
        gathered_rows * cfg.hidden_size,
        dtype=torch.float32,
    ).view(gathered_rows, cfg.hidden_size)
    block.experts.gate = _ConstantModule(
        torch.zeros(
            gathered_rows,
            cfg.num_experts,
            dtype=torch.float32,
        )
    )
    distributed.gather_dp_execution_tokens.reset_mock()
    distributed.gather_dp_execution_tokens.return_value = gathered, expected_start
    monkeypatch.setattr(
        kernels,
        "moe_fused_topk",
        MagicMock(
            return_value=(
                torch.ones(
                    gathered_rows,
                    cfg.num_experts_per_tok,
                ),
                torch.zeros(
                    gathered_rows,
                    cfg.num_experts_per_tok,
                    dtype=torch.int32,
                ),
            )
        ),
    )
    monkeypatch.setattr(
        kernels,
        "grouped_moe_bf16",
        MagicMock(return_value=grouped_output),
        raising=False,
    )
    metadata = SimpleNamespace(
        dp_execution_token_counts=execution_counts,
        dp_is_decode=(1, 1),
        is_prefill=False,
        is_chunked_prefill=False,
    )
    context = ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=[],
    )

    local_input = torch.zeros(execution_counts[dp_rank], cfg.hidden_size)
    with forward_context(context):
        output = block.experts(local_input)

    assert output.shape == (execution_counts[dp_rank], cfg.hidden_size)
    torch.testing.assert_close(
        output,
        grouped_output[expected_start : expected_start + execution_counts[dp_rank]],
    )
    distributed.gather_dp_execution_tokens.assert_called_once_with(
        local_input,
        execution_counts,
        dp_rank,
    )


def test_npu_moe_does_not_branch_on_dp_execution_phase(monkeypatch) -> None:
    cfg = _config(
        tp_size=1,
        dp_size=2,
        world_size=2,
        moe_tp_size=2,
        ep_size=1,
    )
    block = NpuQwen3_5SparseMoEBlock(
        cfg,
        torch.bfloat16,
        torch.device("cpu"),
    )
    gathered = torch.zeros(5, cfg.hidden_size)
    grouped_output = torch.zeros(5, cfg.hidden_size)
    block.experts.gate = _ConstantModule(torch.zeros(5, cfg.num_experts, dtype=torch.float32))
    distributed.gather_dp_execution_tokens.reset_mock()
    distributed.gather_dp_execution_tokens.return_value = gathered, 0
    monkeypatch.setattr(
        kernels,
        "moe_fused_topk",
        MagicMock(
            return_value=(
                torch.ones(5, cfg.num_experts_per_tok),
                torch.zeros(
                    5,
                    cfg.num_experts_per_tok,
                    dtype=torch.int32,
                ),
            )
        ),
    )
    monkeypatch.setattr(
        kernels,
        "grouped_moe_bf16",
        MagicMock(return_value=grouped_output),
        raising=False,
    )
    metadata = SimpleNamespace(
        dp_execution_token_counts=(2, 3),
        dp_is_decode=(0, 1),
    )
    context = ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=[],
    )

    with forward_context(context):
        local_input = torch.zeros(2, cfg.hidden_size)
        output = block.experts(local_input)

    assert output.shape == (2, cfg.hidden_size)
    distributed.gather_dp_execution_tokens.assert_called_once_with(
        local_input,
        (2, 3),
        0,
    )
