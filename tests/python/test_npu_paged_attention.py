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

"""Tests for the NPU paged-attention backend."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu", reason="NPU paged-attention tests require torch_npu")

from xllm.python.attention.backend import (  # noqa: E402
    LayerCache,
    build_speculative_ssm_state_indices,
)
from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend  # noqa: E402


def test_speculative_ssm_indices_keep_all_bootstrap_checkpoints() -> None:
    state_indices = torch.tensor([3, 7], dtype=torch.int64)

    indices = build_speculative_ssm_state_indices(state_indices, checkpoint_stride=4)

    assert indices.tolist() == [[12, 13, 14, 15], [28, 29, 30, 31]]


pytestmark = pytest.mark.usefixtures("causal_conv1d_reference")


def test_uses_first_nonempty_key_cache() -> None:
    backend = NpuPagedAttentionBackend(
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        scale=0.125,
        sliding_window=0,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    linear_cache = LayerCache(
        key=None,
        value=None,
        conv=torch.empty(8, 3, 64),
        ssm=torch.empty(8, 2, 4, 4),
    )
    key_cache = torch.empty(17, 128, 2, 64)
    value_cache = torch.empty_like(key_cache)

    backend.bind_kv_caches(
        [
            linear_cache,
            LayerCache(key=key_cache, value=value_cache),
        ]
    )

    assert backend.num_kv_blocks == 17
    assert backend.page_size == 128


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("activation", ["identity", "silu"])
@pytest.mark.parametrize("is_prefill", [False, True])
def test_kda_dense_conv_dispatches_fused_activation(
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    activation: str,
    is_prefill: bool,
) -> None:
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    value = torch.arange(2 * 12 * width, dtype=torch.bfloat16).view(2, width, 12)
    weight = torch.ones(12, 1, 4, dtype=torch.float32)
    layer = SimpleNamespace(
        layer_id=0,
        activation=activation,
        conv_weight_t=weight.squeeze(1).t().to(value.dtype).contiguous(),
    )
    state = torch.zeros(2, 3, 12, dtype=torch.bfloat16)
    expected_conv = torch.linspace(-4, 4, value.numel()).to(torch.bfloat16).view(2, width, 12)

    def native_conv(
        inputs: torch.Tensor,
        weights: torch.Tensor,
        dense_state: torch.Tensor,
        query_start_loc: list[int],
        activation_mode: int,
        run_mode: int,
    ) -> torch.Tensor:
        assert inputs.is_contiguous()
        assert weights is layer.conv_weight_t
        torch.testing.assert_close(weights, weight.squeeze(1).t().to(value.dtype))
        assert dense_state is state
        assert query_start_loc == []
        assert activation_mode == (1 if activation == "silu" else 0)
        assert run_mode == (0 if is_prefill else 1)
        torch.testing.assert_close(inputs, value)
        dense_state.fill_(7)
        return expected_conv.clone()

    monkeypatch.setattr(torch.ops.xllm_ops, "causal_conv1d", native_conv)
    output = backend._causal_conv1d(value, state, layer, is_prefill=is_prefill)
    torch.testing.assert_close(output, expected_conv, rtol=0, atol=0)
    assert output.dtype == torch.bfloat16
    assert torch.all(state == 7)
