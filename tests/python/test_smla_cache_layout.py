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

"""Check real allocator cache layouts at the Python/native SMLA boundary."""

from types import SimpleNamespace

import pytest
import torch

from xllm.python.attention.csa2_attention import Csa2AttentionBackend
from xllm.python.model_executor.forward_context import ForwardContext, forward_context


@pytest.mark.parametrize("shape", [(2, 128, 512), (2, 128, 1, 512)])
def test_compressed_cache_accepts_both_single_head_layouts(
    shape: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = object.__new__(Csa2AttentionBackend)
    backend.window_size = 128
    backend.scale = 0.5
    cache = torch.randn(shape)
    table = torch.tensor([[0, 1]], dtype=torch.int32)
    shared = SimpleNamespace(key_cache=cache, key_block_table=table)
    monkeypatch.setattr(torch.ops.xllm_ops, "smla_va_metadata", lambda *args, **kwargs: None, raising=False)
    calls: list[torch.Tensor] = []

    def attention(query: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, None]:
        cmp_kv = kwargs["cmp_kv"]
        assert cmp_kv.shape == (2, 128, 1, 512)
        assert cmp_kv.data_ptr() == cache.data_ptr()
        torch.testing.assert_close(cmp_kv.flatten(), cache.flatten(), rtol=0, atol=0)
        calls.append(cmp_kv)
        return query, None

    monkeypatch.setattr(torch.ops.xllm_ops, "smla_va", attention, raising=False)
    context = ForwardContext(backend, torch.device("cpu"), SimpleNamespace(), [])
    layer = SimpleNamespace(mode=SimpleNamespace(compress_ratio=2))
    q = torch.zeros(1, 2, 512)
    with forward_context(context):
        result = backend._smla_attention(
            q,
            torch.zeros(129, 512),
            layer,
            torch.tensor([15]),
            torch.tensor([15]),
            1,
            1,
            0,
            torch.tensor([[[129, -1]]]),
            shared,
            torch.zeros(2),
        )
    assert result is q and len(calls) == 1


@pytest.mark.parametrize("shape", [(2, 128, 2, 512), (2, 128, 256), (128, 512)])
def test_compressed_cache_rejects_incompatible_layout(shape: tuple[int, ...]) -> None:
    backend = object.__new__(Csa2AttentionBackend)
    cache = torch.empty(shape)
    with pytest.raises(ValueError, match="compressed KV cache must have shape"):
        backend._smla_attention(
            torch.zeros(1, 2, 512),
            torch.zeros(129, 512),
            SimpleNamespace(mode=SimpleNamespace(compress_ratio=2)),
            torch.tensor([15]),
            torch.tensor([15]),
            1,
            1,
            0,
            torch.tensor([[[129]]]),
            SimpleNamespace(key_cache=cache),
            torch.zeros(2),
        )
