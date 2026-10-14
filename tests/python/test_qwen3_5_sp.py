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

"""CPU numerical tests for Qwen3.5 TP/SP boundaries and residual ownership."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock, local
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from tests.python.qwen3_5_test_utils import gemma_rms_norm, make_config
from xllm.python import distributed, kernels
from xllm.python.layers import sequence_parallel as sequence_parallel_ops
from xllm.python.layers.layernorm import GemmaRMSNorm
from xllm.python.layers.linear import RowParallelLinear
from xllm.python.layers.npu.qwen3_5.attention import NpuQwen3_5Attention
from xllm.python.layers.npu.qwen3_5.decoder_layer import NpuQwen3_5DecoderLayer
from xllm.python.layers.npu.qwen3_5.gated_delta_net import NpuQwen3_5GatedDeltaNet
from xllm.python.layers.npu.qwen3_5.gdn_metadata import (
    GdnDecodeMetadata,
    GdnMetadata,
    GdnPrefillMetadata,
    GdnStateCache,
)
from xllm.python.layers.npu.qwen3_5.moe_metadata import Qwen3_5MoeMetadata
from xllm.python.layers.sequence_parallel import (
    resolve_sequence_parallel,
    sp_local_rows,
    sp_reduce_scatter,
    sp_shard,
)
from xllm.python.layers.sequence_parallel_metadata import EpSpTokenLayout, SpMetadata
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.models import qwen3_5 as qwen3_5_model
from xllm.python.models.qwen3_5 import Qwen3_5Model


@pytest.mark.parametrize(
    ("instance_role", "expected"),
    [("PREFILL", True), ("DECODE", True), ("DEFAULT", False), ("MIX", False)],
)
def test_sequence_parallel_default_follows_instance_role(
    instance_role: str,
    expected: bool,
) -> None:
    config = {
        "instance_role": instance_role,
        "tp_size": 2,
        "cp_size": 1,
        "num_speculative_tokens": 0,
    }
    assert resolve_sequence_parallel(config, NpuQwen3_5DecoderLayer) is expected


@pytest.mark.parametrize(
    "decoder_layer_cls",
    [torch.nn.Module, type("NonSpDecoder", (torch.nn.Module,), {"supports_sequence_parallel": False})],
)
def test_sequence_parallel_still_requires_model_support(decoder_layer_cls: type[torch.nn.Module]) -> None:
    config = {"instance_role": "DECODE", "tp_size": 2}
    assert resolve_sequence_parallel(config, decoder_layer_cls) is False


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"tp_size": 1}, False),
        ({"cp_size": 2}, False),
        ({"num_speculative_tokens": 1}, False),
        ({"sequence_parallel_enabled": False}, True),
        ({"sequence_parallel_enabled": True, "tp_size": 1}, False),
    ],
)
def test_sequence_parallel_resolver_owns_parallel_policy(overrides: dict[str, object], expected: bool) -> None:
    config = {"instance_role": "DECODE", "tp_size": 2, "cp_size": 1, "num_speculative_tokens": 0, **overrides}
    assert resolve_sequence_parallel(config, NpuQwen3_5DecoderLayer) is expected


@pytest.mark.parametrize(
    ("instance_role", "tp_size", "expected"),
    [("PREFILL", 2, True), ("DECODE", 2, True), ("DEFAULT", 2, False), ("DECODE", 1, False)],
)
def test_model_propagates_resolved_sequence_parallel_state(
    instance_role: str,
    tp_size: int,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = make_config(num_experts=0, tp_size=tp_size, world_size=tp_size, moe_tp_size=tp_size)
    config = {
        **vars(cfg),
        "device": "cpu",
        "dtype": "bfloat16",
        "instance_role": instance_role,
        "sequence_parallel_enabled": not expected,
    }
    monkeypatch.setattr(qwen3_5_model, "get_qwen3_5_decoder_layer_class", lambda device: NpuQwen3_5DecoderLayer)
    model = qwen3_5_model.Qwen3_5ForCausalLM(config)
    assert config["sequence_parallel_enabled"] is expected
    assert model.cfg.sequence_parallel_enabled is expected
    assert model.model.cfg.sequence_parallel_enabled is expected
    assert model.model.layers[0].cfg is model.cfg
    assert model.model.layers[0].self_attn.cfg is model.cfg


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"tp_size": 2}, False),
        ({"tp_size": 2, "sequence_parallel_enabled": False}, False),
        ({"tp_size": 1, "sequence_parallel_enabled": True}, True),
        ({"tp_size": 2, "sequence_parallel_enabled": True}, True),
        ({"sequence_parallel_enabled": True}, True),
        ({"sequence_parallel_enabled": True, "cp_size": 2}, True),
        ({"sequence_parallel_enabled": True, "num_speculative_tokens": 1}, True),
    ],
)
def test_sequence_parallel_layers_and_builder_read_resolved_state(config: dict[str, object], expected: bool) -> None:
    assert sequence_parallel_ops.sequence_parallel_enabled(config) is expected
    assert sequence_parallel_ops.sequence_parallel_enabled(SimpleNamespace(**config)) is expected
    builder = sequence_parallel_ops.SpMetadataBuilder.from_config({"device": "privateuseone", **config})
    assert (builder is not None) is expected


class _Collectives:
    """Execute real SUM/gather tensor math for concurrent CPU rank forwards."""

    def __init__(self, size: int, monkeypatch: pytest.MonkeyPatch, dp_size: int = 1) -> None:
        self.tp_size = size
        self.size = size * dp_size
        self._rank = local()
        self._barrier = Barrier(self.size, timeout=10)
        self._lock = Lock()
        self._inputs: dict[int, dict[int, torch.Tensor]] = {}
        self.operations: list[str] = []
        monkeypatch.setattr(distributed, "tp_all_gather", self._gather, raising=False)
        monkeypatch.setattr(distributed, "tp_all_reduce", self._reduce, raising=False)
        monkeypatch.setattr(distributed, "reduce_scatter", self._scatter, raising=False)
        monkeypatch.setattr(distributed, "gather_dp_execution_tokens", self._dp_gather)
        monkeypatch.setattr(distributed, "all_reduce_", self._group_reduce)

    def _exchange(self, tensor: torch.Tensor, operation: str, group: str = "tp") -> list[torch.Tensor]:
        step = self._rank.step
        self._rank.step += 1
        with self._lock:
            self._inputs.setdefault(step, {})[self._rank.index] = tensor.clone()
            if self._rank.index == 0:
                self.operations.append(operation)
        self._barrier.wait()
        if group == "tp":
            start = self._rank.index // self.tp_size * self.tp_size
            ranks = range(start, start + self.tp_size)
        else:
            assert group == "dp"
            ranks = range(self._rank.index % self.tp_size, self.size, self.tp_size)
        return [self._inputs[step][rank] for rank in ranks]

    def _gather(self, tensor: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
        assert world_size == self.tp_size
        return torch.cat(self._exchange(tensor, "gather"), dim=dim)

    def _reduce(self, tensor: torch.Tensor) -> None:
        tensor.copy_(torch.stack(self._exchange(tensor, "reduce")).sum(0))

    def _scatter(self, tensor: torch.Tensor, world_size: int, group_name: str) -> torch.Tensor:
        assert world_size == self.tp_size and group_name == "tp"
        reduced = torch.stack(self._exchange(tensor, "scatter")).sum(0)
        return reduced.chunk(self.tp_size, dim=0)[self._rank.index % self.tp_size].contiguous()

    def _dp_gather(self, tensor: torch.Tensor, counts: tuple[int, ...], rank: int) -> tuple[torch.Tensor, int]:
        parts = self._exchange(tensor, "dp_gather", "dp")
        assert tuple(part.shape[0] for part in parts) == counts
        return torch.cat(parts), sum(counts[:rank])

    def _group_reduce(self, tensor: torch.Tensor, group: str) -> None:
        tensor.copy_(torch.stack(self._exchange(tensor, "dp_reduce", group)).sum(0))

    def run(self, function: Callable[[int], torch.Tensor]) -> list[torch.Tensor]:
        def worker(rank: int) -> torch.Tensor:
            self._rank.index = rank
            self._rank.step = 0
            with torch.no_grad():
                return function(rank)

        with ThreadPoolExecutor(max_workers=self.size) as executor:
            futures = [executor.submit(worker, rank) for rank in range(self.size)]
            return [future.result(timeout=20) for future in futures]


def _weights(layer_id: int, tp_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = 2 * tp_size
    input_weight = torch.arange(width * 4, dtype=torch.float32).view(width, 4) / 31 - 0.4
    output_weight = torch.arange(4 * width, dtype=torch.float32).view(4, width) / 47 - 0.2
    bias = torch.tensor([0.1, -0.2, 0.3, 0.4]) * (layer_id + 1)
    return input_weight / (layer_id + 1), output_weight / (layer_id + 1), bias


class _Mixer(torch.nn.Module):
    def __init__(self, layer_id: int, rank: int, tp_size: int, rows: int) -> None:
        super().__init__()
        input_weight, output_weight, bias = _weights(layer_id, tp_size)
        self.input_weight = input_weight.chunk(tp_size, dim=0)[rank]
        self.projection = RowParallelLinear(2, 4, tp_size, bias=True, dtype=torch.float32)
        with torch.no_grad():
            self.projection.weight.copy_(output_weight.chunk(tp_size, dim=1)[rank])
            self.projection.bias.copy_(bias)
        self.rows = rows

    def forward(
        self,
        hidden_or_positions: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden is None:
            hidden = hidden_or_positions
        else:
            torch.testing.assert_close(hidden_or_positions, torch.arange(self.rows))
        assert hidden.shape == (self.rows, 4)
        projected = F.linear(hidden, self.input_weight).tanh()
        if self._sequence_parallel_enabled:
            if self.projection._weight_is_transposed:
                partial = torch.matmul(projected, self.projection.weight)
            else:
                partial = F.linear(projected, self.projection.weight)
            output = sp_reduce_scatter(partial, self.projection.tp_size)
            return output + self.projection.bias
        return self.projection(projected)


class _Mlp(torch.nn.Module):
    def __init__(self, layer_id: int, partial_scale: float = 1.0) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.partial_scale = partial_scale

    def validate_sp(
        self,
        sp_metadata: SpMetadata,
        *,
        device: torch.device,
    ) -> None:
        assert device.type == "cpu"
        assert sp_metadata.tp_size > 1

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.sigmoid() * (self.layer_id + 1) / 5 * self.partial_scale


def _norm() -> GemmaRMSNorm:
    norm = GemmaRMSNorm(4, eps=1e-6, dtype=torch.float32)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.0, 0.2, -0.1, 0.3]))
    return norm


def _layer(
    layer_id: int,
    rank: int,
    tp_size: int,
    rows: int,
    moe_layers: tuple[bool, ...],
    mega_moe: bool = False,
    sequence_parallel: bool = False,
) -> NpuQwen3_5DecoderLayer:
    layer = NpuQwen3_5DecoderLayer.__new__(NpuQwen3_5DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.cfg = SimpleNamespace(
        enable_mega_moe=mega_moe,
        sequence_parallel_enabled=sequence_parallel,
        tp_size=tp_size,
        tp_rank=rank,
        hidden_size=4,
        is_moe_layer=lambda index: moe_layers[index],
    )
    layer.layer_id = layer_id
    layer.layer_type = "full_attention" if layer_id % 2 else "linear_attention"
    layer.input_layernorm = _norm()
    layer.post_attention_layernorm = _norm()
    mixer = _Mixer(layer_id, rank, tp_size, rows)
    mixer._sequence_parallel_enabled = sequence_parallel
    if layer.layer_type == "full_attention":
        layer.self_attn = mixer
    else:
        layer.linear_attn = mixer
    partial_scale = 1.0 / tp_size if sequence_parallel and not moe_layers[layer_id] else 1.0
    layer.mlp = _Mlp(layer_id, partial_scale)
    return layer


def _reference(
    hidden: torch.Tensor,
    tp_size: int,
    layer_count: int,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    norm = _norm()
    residual = None
    captures: dict[int, torch.Tensor] = {}
    for layer_id in range(layer_count):
        if residual is None:
            residual = hidden
            hidden = norm(hidden)
        else:
            hidden, residual = norm(hidden, residual)
        input_weight, output_weight, bias = _weights(layer_id, tp_size)
        hidden = F.linear(F.linear(hidden, input_weight).tanh(), output_weight, bias)
        hidden, residual = norm(hidden, residual)
        hidden = hidden.sigmoid() * (layer_id + 1) / 5
        captures[layer_id] = hidden + residual
    hidden, _ = norm(hidden, residual)
    return hidden, captures


@pytest.mark.parametrize(
    ("rows", "tp_size", "moe_layers"),
    [
        (5, 2, (True, True, True)),
        (1, 8, (True, True)),
        (3, 4, (True, False, True)),
        (5, 2, (True, True, False)),
        (3, 2, (False, True, True)),
    ],
)
@pytest.mark.parametrize("mega_moe", [False, True])
@pytest.mark.parametrize("layers_to_capture", [(), (1, 0)])
def test_sp_model_matches_full_residual_reference(
    rows: int,
    tp_size: int,
    moe_layers: tuple[bool, ...],
    mega_moe: bool,
    layers_to_capture: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kernels, "gemma_rms_norm", gemma_rms_norm, raising=False)
    collectives = _Collectives(tp_size, monkeypatch)
    hidden = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4) / 17 - 0.5
    models = []
    for rank in range(tp_size):
        model = Qwen3_5Model.__new__(Qwen3_5Model)
        torch.nn.Module.__init__(model)
        model.cfg = SimpleNamespace(sequence_parallel_enabled=True)
        model.embed_tokens = torch.nn.Embedding.from_pretrained(hidden.clone())
        model.layers = torch.nn.ModuleList(
            _layer(index, rank, tp_size, rows, moe_layers, mega_moe, sequence_parallel=True)
            for index in range(len(moe_layers))
        )
        model.norm = _norm()
        model.aux_hidden_capture = qwen3_5_model.AuxHiddenCapture(layers_to_capture)
        models.append(model)

    expected, reference_captures = _reference(hidden, tp_size, len(moe_layers))

    def forward(rank: int) -> torch.Tensor:
        context = ForwardContext(
            None,
            torch.device("cpu"),
            None,
            [],
            execution_contexts={
                SpMetadata: SpMetadata.from_execution_rows(
                    rows,
                    tp_size,
                    rank,
                    ep_layout=EpSpTokenLayout.from_token_counts(
                        (rows,), (rows,), tp_size=tp_size, tp_rank=rank, dp_size=1, dp_rank=0, compacted=True
                    ),
                ),
                Qwen3_5MoeMetadata: Qwen3_5MoeMetadata(
                    torch.ones(sp_local_rows(rows, tp_size), dtype=torch.int8),
                ),
            },
        )
        with forward_context(context):
            result = models[rank](torch.arange(rows), torch.arange(rows))
        if layers_to_capture:
            output, captured = result
            expected_captures = torch.cat([reference_captures[layer_id] for layer_id in layers_to_capture], dim=-1)
            assert captured.shape == (rows, 4 * len(layers_to_capture))
            torch.testing.assert_close(captured, expected_captures, rtol=1e-5, atol=1e-6)
            return output
        return result

    outputs = collectives.run(forward)
    for output in outputs:
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
    dense_layers = sum(not is_moe for is_moe in moe_layers)
    assert collectives.operations.count("scatter") == len(moe_layers) + dense_layers
    assert collectives.operations.count("gather") == len(moe_layers) + dense_layers + 1 + bool(layers_to_capture)
    assert collectives.operations.count("reduce") == 0


@pytest.mark.parametrize("transposed", [False, True])
def test_row_projection_reduces_before_adding_replicated_bias(
    transposed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collectives = _Collectives(2, monkeypatch)
    inputs = torch.arange(20, dtype=torch.float32).view(5, 4) / 7
    weight = torch.arange(12, dtype=torch.float32).view(3, 4) / 11
    bias = torch.tensor([0.25, -0.5, 0.75])
    projections = []
    for rank in range(2):
        projection = RowParallelLinear(2, 3, 2, bias=True)
        with torch.no_grad():
            projection.weight.copy_(weight.chunk(2, dim=1)[rank])
            projection.bias.copy_(bias)
            if transposed:
                projection.weight.data = projection.weight.t().contiguous()
                projection._weight_is_transposed = True
        projections.append(projection)
    expected = F.linear(inputs, weight, bias)

    def project_and_reduce(rank: int) -> torch.Tensor:
        layout = SpMetadata.from_execution_rows(inputs.shape[0], 2, rank)
        return sequence_parallel_ops.sp_row_parallel_projection(projections[rank], inputs.chunk(2, dim=1)[rank], layout)

    outputs = collectives.run(project_and_reduce)
    torch.testing.assert_close(torch.cat(outputs)[:5], expected)
    assert collectives.operations == ["scatter"]
    assert all(projection.reduce_results for projection in projections)


def test_local_row_projection_reduces_features_without_token_scatter(monkeypatch: pytest.MonkeyPatch) -> None:
    collectives = _Collectives(2, monkeypatch)
    inputs = torch.arange(12, dtype=torch.float32).view(3, 4) / 7
    weight = torch.arange(16, dtype=torch.float32).view(4, 4) / 11
    bias = torch.tensor([0.1, -0.2, 0.3, 0.4])
    projections = []
    for rank in range(2):
        projection = RowParallelLinear(2, 4, 2, bias=True, dtype=torch.float32)
        with torch.no_grad():
            projection.weight.copy_(weight[:, rank::2])
            projection.bias.copy_(bias)
        projections.append(projection)

    outputs = collectives.run(lambda rank: projections[rank](inputs[:, rank::2]))
    expected = F.linear(inputs, weight, bias)
    torch.testing.assert_close(outputs[0], expected)
    torch.testing.assert_close(outputs[1], expected)
    assert collectives.operations == ["reduce"]


def test_dense_ffn_sp_keeps_tp_weights_and_defers_down_reduction() -> None:
    cfg = make_config(
        mlp_only_layers=[0],
        sequence_parallel_enabled=True,
        tp_size=2,
        tp_rank=0,
        world_size=2,
        moe_tp_size=2,
        ep_size=1,
    )
    layer = NpuQwen3_5DecoderLayer(
        cfg,
        0,
        torch.float32,
        torch.device("cpu"),
        _Rotary(),
    )

    assert layer.mlp.down_proj.tp_size == cfg.tp_size
    assert layer.mlp.down_proj.reduce_results is False
    assert layer.mlp.gate_up_proj.weight.shape[0] == 2 * (cfg.intermediate_size // cfg.tp_size)
    assert layer.mlp.down_proj.weight.shape[1] == cfg.intermediate_size // cfg.tp_size


@pytest.mark.parametrize(
    ("mask", "error", "message"),
    [
        (None, RuntimeError, "execution metadata is unavailable"),
        (torch.ones(1, dtype=torch.int8), ValueError, "capacity must cover"),
        (torch.ones(2), ValueError, "contiguous int8"),
        (torch.ones(2, 1, dtype=torch.int8), ValueError, "contiguous int8"),
        (torch.ones(4, dtype=torch.int8)[::2], ValueError, "contiguous int8"),
    ],
)
@pytest.mark.parametrize("mega_moe", [False, True])
def test_dense_first_sp_rejects_metadata_before_embedding_or_attention(
    mask: torch.Tensor | None,
    error: type[Exception],
    message: str,
    mega_moe: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = make_config(
        num_hidden_layers=2,
        layer_types=["full_attention", "full_attention"],
        mlp_only_layers=[0],
        dp_size=2,
        world_size=4,
        moe_tp_size=1,
        ep_size=4,
        sequence_parallel_enabled=True,
        enable_mega_moe=mega_moe,
        mega_moe_context=torch.zeros(1, dtype=torch.int32) if mega_moe else None,
        mega_moe_ccl_buffer_size=1024,
        mega_moe_num_max_tokens_per_rank=4096,
    )
    monkeypatch.setattr(
        qwen3_5_model,
        "get_qwen3_5_decoder_layer_class",
        lambda device: NpuQwen3_5DecoderLayer,
    )
    model = Qwen3_5Model(cfg, torch.bfloat16, torch.device("cpu"))
    assert model.cfg.sequence_parallel_enabled is True
    embedding = MagicMock()
    attention = MagicMock()
    gather = MagicMock()
    monkeypatch.setattr(model.embed_tokens, "forward", embedding)
    monkeypatch.setattr(model.layers[0].self_attn, "forward", attention)
    monkeypatch.setattr(distributed, "tp_all_gather", gather, raising=False)
    context = ForwardContext(None, torch.device("cpu"), None, [])
    context.execution_contexts[SpMetadata] = SpMetadata.from_execution_rows(
        3,
        cfg.tp_size,
        cfg.tp_rank,
        ep_layout=EpSpTokenLayout.from_token_counts(
            (3, 3), (3, 3), tp_size=cfg.tp_size, tp_rank=cfg.tp_rank, dp_size=2, dp_rank=0, compacted=True
        ),
    )
    if mask is not None:
        context.execution_contexts[Qwen3_5MoeMetadata] = Qwen3_5MoeMetadata(
            active_token_mask=mask,
        )
    with forward_context(context), pytest.raises(error, match=message):
        model(torch.arange(3), torch.arange(3))
    embedding.assert_not_called()
    attention.assert_not_called()
    gather.assert_not_called()


class _Rotary(torch.nn.Module):
    def forward(self, positions: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        assert value.shape[0] == positions.numel()
        return value

    def build_mrope_cos_sin(self, positions: torch.Tensor) -> torch.Tensor:
        return positions[:, None]


class _QueryAttention(torch.nn.Module):
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return query + 0.2 * value


@pytest.mark.parametrize("fused", [False, True])
def test_npu_attention_sp_projection_matches_tp_output(fused: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kernels, "gemma_rms_norm", gemma_rms_norm, raising=False)

    def split_qkvg(qkvg: torch.Tensor, *args: object) -> tuple[torch.Tensor, ...]:
        query, gate, key, value = qkvg.split(2, dim=-1)
        return query, key, value, gate

    monkeypatch.setattr(kernels, "split_qkv_rmsnorm_mrope", split_qkvg, raising=False)
    hidden = torch.arange(12, dtype=torch.float32).view(3, 4) / 13
    layers = []
    for rank in range(2):
        cfg = make_config(
            hidden_size=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=2,
            partial_rotary_factor=1.0,
            tp_rank=rank,
            attention_bias=True,
        )
        layer = NpuQwen3_5Attention(cfg, 0, torch.float32, torch.device("cpu"), _Rotary())
        layer._fused_qkv_enabled = fused
        layer._mrope_gather_pattern = torch.zeros(1, dtype=torch.int64)
        layer.attn = _QueryAttention()
        with torch.no_grad():
            layer.qkv_proj.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(8, 4) / 37 + rank / 10)
            layer.qkv_proj.bias.fill_(0.1)
            layer.o_proj.weight.copy_(torch.arange(8, dtype=torch.float32).reshape(4, 2) / 19 + rank / 10)
            layer.o_proj.bias.fill_(0.7)
        layers.append(layer)
    tp_collectives = _Collectives(2, monkeypatch)
    for layer in layers:
        layer.cfg.sequence_parallel_enabled = False
    expected = tp_collectives.run(lambda rank: layers[rank](torch.arange(3), hidden))
    sp_collectives = _Collectives(2, monkeypatch)
    for layer in layers:
        layer.cfg.sequence_parallel_enabled = True

    def sp_forward(rank: int) -> torch.Tensor:
        context = ForwardContext(
            None,
            torch.device("cpu"),
            None,
            [],
            execution_contexts={SpMetadata: SpMetadata.from_execution_rows(3, 2, rank)},
        )
        with forward_context(context):
            return layers[rank](torch.arange(3), hidden)

    actual = sp_collectives.run(sp_forward)
    torch.testing.assert_close(torch.cat(actual)[:3], expected[0])
    assert tp_collectives.operations == ["reduce"]
    assert sp_collectives.operations == ["scatter"]


@pytest.mark.parametrize("prefill", [False, True])
def test_npu_gdn_sp_preserves_full_token_state_inputs(prefill: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = 3
    hidden = torch.arange(12, dtype=torch.float32).reshape(rows, 4) / 13
    state = GdnStateCache(torch.zeros(4, 3, 8), torch.zeros(4, 1, 2, 2))
    indices = torch.tensor([3, 1, 2], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, rows], dtype=torch.int32)
    metadata = (
        GdnPrefillMetadata({0: state}, indices[:1], indices[:1], indices[:1], indices[:1], cu_seqlens, 1)
        if prefill
        else GdnDecodeMetadata({0: state}, indices, indices)
    )

    def decode(
        self: NpuQwen3_5GatedDeltaNet,
        actual_hidden: torch.Tensor,
        conv: torch.Tensor,
        ssm: torch.Tensor,
        read: torch.Tensor,
        write: torch.Tensor,
    ) -> torch.Tensor:
        assert conv is state.conv_state and ssm is state.ssm_state
        assert read is indices and write is indices
        torch.testing.assert_close(actual_hidden, hidden)
        return actual_hidden[:, :2]

    def project_inputs(
        self: NpuQwen3_5GatedDeltaNet,
        actual_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.testing.assert_close(actual_hidden, hidden)
        return (actual_hidden,) * 4

    def prefill_kernel(mixed_qkv: torch.Tensor, *args: object) -> torch.Tensor:
        assert args[4] is state.conv_state
        assert args[11] is state.ssm_state
        assert args[12] is cu_seqlens
        return mixed_qkv[:, :2]

    monkeypatch.setattr(NpuQwen3_5GatedDeltaNet, "_decode", decode)
    monkeypatch.setattr(NpuQwen3_5GatedDeltaNet, "_project_prefill_inputs", project_inputs)
    monkeypatch.setattr(kernels, "mega_gdn_prefill", prefill_kernel, raising=False)
    layers = []
    for rank in range(2):
        layer = NpuQwen3_5GatedDeltaNet.__new__(NpuQwen3_5GatedDeltaNet)
        torch.nn.Module.__init__(layer)
        layer.layer_id = 0
        layer.cfg = SimpleNamespace(tp_size=2, sequence_parallel_enabled=False)
        layer.value_dim = 2
        layer.out_proj = RowParallelLinear(2, 4, 2)
        with torch.no_grad():
            layer.out_proj.weight.fill_(rank + 0.25)
        for name in ("conv1d_weight", "A_log", "dt_bias", "norm_weight"):
            setattr(layer, name, torch.ones(1))
        layers.append(layer)

    def forward(rank: int, sp: bool) -> torch.Tensor:
        layers[rank].cfg.sequence_parallel_enabled = sp
        execution_contexts = {GdnMetadata: metadata}
        if sp:
            execution_contexts[SpMetadata] = SpMetadata.from_execution_rows(rows, 2, rank)
        context = ForwardContext(None, torch.device("cpu"), None, [], execution_contexts=execution_contexts)
        with forward_context(context):
            return layers[rank](hidden)

    tp_collectives = _Collectives(2, monkeypatch)
    expected = tp_collectives.run(lambda rank: forward(rank, False))
    sp_collectives = _Collectives(2, monkeypatch)
    actual = sp_collectives.run(lambda rank: forward(rank, True))
    torch.testing.assert_close(torch.cat(actual)[:rows], expected[0])
    assert tp_collectives.operations == ["reduce"]
    assert sp_collectives.operations == ["scatter"]
