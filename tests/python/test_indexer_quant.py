# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Bitwise codes/scales and live ACL graph replay for fused indexer INT8."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable

import pytest
import torch


@pytest.fixture(scope="module")
def quantize() -> Callable:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU is required for Triton indexer quantization")
    source = Path(__file__).resolve().parents[2] / "xllm/python/kernels_npu/triton/indexer_quant.py"
    spec = importlib.util.spec_from_file_location("indexer_quant_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.npu.set_device(0)
    torch.npu.set_deterministic_level(1)
    return module.quantize_indexer_rows_int8


def _reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = x.float().abs().amax(dim=-1)
    scale = (maximum / 127.0).half().float().clamp_min(2.0**-24)
    codes = torch.round(x.float() / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return codes, scale.half()


@pytest.mark.parametrize(
    "shape",
    [
        (0, 128),
        (1, 128),
        (7, 128),
        (1, 32, 128),
        (6, 32, 128),
        (17, 32, 128),
        (3, 7, 128),
        (2, 1, 128),
        (1024, 32, 128),
    ],
)
def test_indexer_quant_live_graph_is_bitwise_exact(quantize: Callable, shape: tuple[int, ...]) -> None:
    torch.manual_seed(42)
    x = torch.randn(shape, device="npu", dtype=torch.bfloat16)
    for _ in range(3):
        quantize(x)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = quantize(x)
    for kind in ["random", "zero", "tiny", "halfway", "new_input"]:
        x.normal_()
        if kind == "zero":
            x.zero_()
        elif kind == "tiny":
            x.mul_(1e-9)
        elif kind == "halfway":
            pattern = torch.tensor(
                [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 127], device="npu", dtype=torch.bfloat16
            ).repeat(16)
            x.copy_(pattern)
        graph.replay()
        expected = _reference(x)
        for lhs, rhs in zip(actual, expected):
            torch.testing.assert_close(lhs.cpu(), rhs.cpu(), rtol=0, atol=0)
