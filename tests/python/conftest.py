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

"""Package stubs for pure-Python tests that run without platform kernels."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

_PYTHON_ROOT = Path(__file__).parents[2] / "xllm" / "python"
_SHARED_SOURCE_MODULES: dict[Path, types.ModuleType] = {}


def load_source_module(path: str | Path, name: str) -> types.ModuleType:
    """Load a source file once per resolved path and share that one instance.

    Suites that need the real implementation behind the stubs below load it
    straight from its file rather than importing it. ``distributed/collectives.py``
    registers the global ``xllm_ops::*`` operators while it executes, so loading
    it twice in one process silently re-registers them and hands dispatch to the
    latest module object: monkeypatches made through the first instance stop
    applying and the suites become order dependent. Keying the cache on the
    resolved path, and reusing the module already registered under ``name``,
    keeps every caller on the same instance.
    """
    resolved = Path(path).resolve()
    module = _SHARED_SOURCE_MODULES.get(resolved)
    if module is None:
        registered = sys.modules.get(name)
        if registered is not None and Path(getattr(registered, "__file__", "") or "").resolve() == resolved:
            module = registered
    if module is None:
        spec = importlib.util.spec_from_file_location(name, resolved)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    _SHARED_SOURCE_MODULES[resolved] = module
    return module


def _rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """CPU reference for weighted RMSNorm over the last dimension."""
    value_fp32 = value.float()
    variance = value_fp32.square().mean(dim=-1, keepdim=True)
    normalized = value_fp32 * torch.rsqrt(variance + eps)
    return (normalized * weight.float()).to(value.dtype)


def _rms_norm_sigmoid_gated(
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """CPU reference for kernels_npu rms_norm_sigmoid_gated.

    Matches the Triton kernel contract: RMSNorm over the last dim, scaled by
    ``weight`` and gated by ``sigmoid(gate)``. Used by pure-Python model
    tests (glm5_next KDA o_norm) that cannot link the NPU kernels.
    """
    input_dtype = value.dtype
    x = value.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x * weight.to(torch.float32) * gate.sigmoid()).to(input_dtype)


def _l2_norm(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """CPU reference for L2 normalization over the last dimension."""
    input_dtype = value.dtype
    value_fp32 = value.to(torch.float32)
    normalized = value_fp32 * torch.rsqrt(value_fp32.square().sum(dim=-1, keepdim=True) + eps)
    return normalized.to(input_dtype)


def _batch_matmul_transpose(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """CPU reference for the MLA [T,H,D] projection contract."""
    return torch.bmm(x.transpose(0, 1), weight).transpose(0, 1)


def _scatter_nd_update(
    value: torch.Tensor,
    indices: torch.Tensor,
    updates: torch.Tensor,
) -> None:
    """CPU reference for the kernels_npu scatter_nd_update in-place scatter.

    Aligned with the ``xllm.python.kernels_npu.scatter_nd_update`` contract:
    write every row of ``updates`` into ``value`` in place at the row named by
    ``indices``, where ``indices`` has shape ``[num_updates, 1]`` (one row index
    per update, as every caller passes it). Provided here so pure-Python tests
    can exercise the sparse-attention paths without loading the native
    ``xllm_ops`` extension.
    """
    flat_indices = indices.reshape(-1).to(torch.long)
    value.index_copy_(0, flat_indices, updates.to(value.dtype))


def _unmocked_platform_operation(*args: object, **kwargs: object) -> None:
    raise RuntimeError("Platform operation requires an explicit test mock")


def _install_python_package_stub() -> None:
    kernels = types.ModuleType("xllm.python.kernels")
    kernels.update_decode_graph_metadata = _unmocked_platform_operation
    kernels.rms_norm = _rms_norm
    kernels.rms_norm_sigmoid_gated = _rms_norm_sigmoid_gated
    kernels.l2_norm = _l2_norm
    kernels.batch_matmul_transpose = _batch_matmul_transpose
    kernels.scatter_nd_update = _scatter_nd_update
    kernels_npu = types.ModuleType("xllm.python.kernels_npu")
    kernels_npu.__path__ = [str(_PYTHON_ROOT / "kernels_npu")]
    distributed = types.ModuleType("xllm.python.distributed")
    distributed.dcp_group = lambda _device=None: None

    package = types.ModuleType("xllm.python")
    # Keep source submodules importable without executing the real package binding.
    package.__path__ = [str(_PYTHON_ROOT)]
    package.kernels = kernels
    package.kernels_npu = kernels_npu
    package.distributed = distributed

    distributed.tp_rank = lambda device: 0
    distributed.tp_group = _unmocked_platform_operation
    distributed.moe_ep_all_reduce = lambda x: None
    distributed.moe_tp_all_reduce = lambda x: None
    # V4.1 runs tp_size == 1 in the pure-Python tests, so the TP ops are
    # single-rank identities (mirroring moe_tp_all_reduce above).
    distributed.tp_all_reduce = lambda x: None
    distributed.tp_all_gather = lambda x, dim, world_size: x
    # Engram host-offload surface used by deepseek_v41 / engram_shared_memory.
    # The object collectives are no-ops (the pure-Python tests never join a
    # real gloo group); ``engram_dp_shard_size`` returns 1 (no sharing) and
    # ``init_engram_dp_group`` raises, matching the real "no world" behaviour so
    # a dp_shared_memory request fails loudly instead of silently sharing.
    distributed.engram_broadcast_object_list = lambda object_list, src=0, group_name=None: None
    distributed.engram_all_gather_object = lambda object_list, obj, group_name=None: None
    distributed.engram_barrier = lambda group_name=None, timeout=None: None
    distributed.world_topology = lambda: None
    distributed.engram_dp_shard_size = lambda dp_size, replica_size: 1

    def _init_engram_dp_group(_device: str, _dp_size: int, _shard_size: int):
        raise RuntimeError("engram DP-shared host table requires an initialized process-group world")

    distributed.init_engram_dp_group = _init_engram_dp_group
    distributed.gather_dp_execution_tokens = lambda *args, **kwargs: None
    distributed.all_reduce_ = lambda *args, **kwargs: None
    distributed.broadcast_ = _unmocked_platform_operation

    sys.modules["xllm.python"] = package
    sys.modules["xllm.python.kernels"] = kernels
    sys.modules["xllm.python.kernels_npu"] = kernels_npu
    sys.modules["xllm.python.distributed"] = distributed


_install_python_package_stub()


@pytest.fixture
def causal_conv1d_reference(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def native_conv(
        inputs: torch.Tensor,
        weight: torch.Tensor,
        state: torch.Tensor,
        query_start_loc: list[int],
        activation_mode: int,
        run_mode: int,
    ) -> torch.Tensor:
        calls.append(
            dict(
                inputs=inputs.clone(),
                weight=weight,
                state=state.clone(),
                query_start_loc=query_start_loc,
                activation_mode=activation_mode,
                run_mode=run_mode,
            )
        )
        assert inputs.is_contiguous()
        assert weight.is_contiguous()
        assert state.is_contiguous()
        assert inputs.dtype == weight.dtype == state.dtype
        channels = inputs.shape[-1]
        flat_input = inputs.reshape(-1, channels)
        boundaries = query_start_loc or [sequence * inputs.shape[1] for sequence in range(inputs.shape[0] + 1)]
        output = torch.empty_like(flat_input)
        for sequence, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if start == end:
                continue
            window = torch.cat((state[sequence].float(), flat_input[start:end].float()), dim=0)
            convolved = (
                torch.nn.functional.conv1d(window.t().unsqueeze(0), weight.float().t().unsqueeze(1), groups=channels)
                .squeeze(0)
                .t()
            )
            if activation_mode == 1:
                convolved = torch.nn.functional.silu(convolved)
            output[start:end].copy_(convolved)
            state[sequence].copy_(window[-state.shape[1] :])
        output = output.view_as(inputs)
        calls[-1]["output"] = output.clone()
        return output

    monkeypatch.setattr(torch.ops.xllm_ops, "causal_conv1d", native_conv, raising=False)
    return calls
