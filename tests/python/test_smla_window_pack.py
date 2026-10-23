# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""NPU packing checks against SMLA's padded ring-gather semantics."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable

import pytest
import torch


@pytest.fixture(scope="module")
def pack() -> Callable:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU is required for Triton window packing")
    source = Path(__file__).resolve().parents[2] / "xllm/python/kernels_npu/triton/smla_window_pack.py"
    spec = importlib.util.spec_from_file_location("smla_window_pack_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.npu.set_device(0)
    torch.npu.set_deterministic_level(1)
    return module.pack_smla_window


def _reference(cache: torch.Tensor, table: torch.Tensor, chunk: torch.Tensor, length: int, seq: int) -> torch.Tensor:
    absolute = length - chunk.size(0) - 128 + torch.arange(128)
    positions = absolute.clamp_min(0)
    columns = positions // 128 % table.size(1)
    slots = table[seq, columns].long().clamp_min(0) * 128 + positions % 128
    history = torch.where((absolute >= 0).unsqueeze(-1), cache[slots], 0)
    current = chunk if length > 0 else torch.zeros_like(chunk)
    return torch.cat((history, current))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("query_len", [1, 7, 1024, 2048])
def test_window_pack_boundaries_and_strided_tables(pack: Callable, dtype: torch.dtype, query_len: int) -> None:
    torch.manual_seed(42)
    cache_cpu = torch.randn(8 * 128, 512).bfloat16()
    chunk_cpu = torch.randn(query_len, 512).bfloat16()
    table_cpu = torch.tensor([[2, 0, -1, 0, 1, 0], [7, 0, 6, 0, 4, 0]], dtype=dtype)[:, ::2]
    backing = torch.zeros(2, 6, dtype=dtype, device="npu")
    backing[:, ::2].copy_(table_cpu)
    table = backing[:, ::2]
    cache, chunk = cache_cpu.npu(), chunk_cpu.npu()
    for previous in [0, 1, 127, 128, 2048, 4097]:
        lengths = torch.tensor([previous + query_len, previous + query_len + 1], device="npu")
        for seq in [0, 1]:
            actual = pack(cache, table, chunk, lengths, seq, 128, 128)
            expected = _reference(cache_cpu, table_cpu, chunk_cpu, previous + query_len + seq, seq)
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_window_pack_graph_zeroes_nan_history_and_dummy_rows(pack: Callable) -> None:
    cache = torch.full((8 * 128, 512), float("nan"), dtype=torch.bfloat16, device="npu")
    table = torch.zeros(1, 3, dtype=torch.int32, device="npu")
    chunk = torch.ones(1, 512, dtype=torch.bfloat16, device="npu")
    lengths = torch.ones(1, dtype=torch.int64, device="npu")
    for _ in range(3):
        pack(cache, table, chunk, lengths, 0, 128, 128)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = pack(cache, table, chunk, lengths, 0, 128, 128)
    for length in [1, 0, 127, 128, 2048]:
        cache_cpu = torch.randn(8 * 128, 512).bfloat16()
        chunk_cpu = torch.randn(1, 512).bfloat16()
        if length <= 1:
            cache_cpu.fill_(float("nan"))
        if length == 0:
            chunk_cpu.fill_(float("nan"))
        cache.copy_(cache_cpu)
        chunk.copy_(chunk_cpu)
        lengths.fill_(length)
        table.fill_(length % 8)
        graph.replay()
        expected = _reference(cache_cpu, table.cpu(), chunk_cpu, length, 0)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        assert torch.isfinite(actual).all()
