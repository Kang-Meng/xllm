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

"""Extension/runtime registration contracts for the embedded interpreter.

Both properties are checked against a real ``xllm_export`` build:

* importing the extension into a live interpreter must not abort it, and
* the import must publish the ``xllm_runtime`` module the Python model
  executor uses to reach the native collectives.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_RUNTIME_SYMBOLS = (
    "AttentionMetadataView",
    "ExpandedDecodeMetadataView",
    "InputBatchMetadataView",
    "LayerLoadSynchronizer",
    "tp_all_reduce",
    "tp_all_gather",
    "tp_all_to_all",
    "dp_all_gather",
    "moe_tp_all_reduce",
    "moe_ep_all_reduce",
    "moe_ep_comm_name",
    "moe_ep_rank",
    "eplb_batch_isend_irecv",
    "eplb_wait_batch_isend_irecv",
)


def _export_path() -> Path:
    value = os.environ.get("XLLM_EXPORT_PATH")
    if not value:
        pytest.skip("set XLLM_EXPORT_PATH to the built xllm_export extension")
    artifact = Path(value).resolve()
    assert artifact.is_file(), f"built extension does not exist: {artifact}"
    return artifact


def test_import_into_running_python_does_not_abort(tmp_path: Path) -> None:
    artifact = _export_path()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util, sys; import torch; "
            "spec = importlib.util.spec_from_file_location('xllm_export', sys.argv[1]); "
            "module = importlib.util.module_from_spec(spec); "
            "sys.modules['xllm_export'] = module; "
            "spec.loader.exec_module(module)",
            str(artifact),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_import_registers_xllm_runtime(tmp_path: Path) -> None:
    artifact = _export_path()
    symbols = ", ".join(repr(name) for name in _RUNTIME_SYMBOLS)
    script = f"""
import importlib.util
import sys

import torch

spec = importlib.util.spec_from_file_location("xllm_export", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules["xllm_export"] = module
spec.loader.exec_module(module)

import xllm_runtime

for name in ({symbols},):
    assert hasattr(xllm_runtime, name), name

# The native collective must operate in place on the tensor it is handed; a
# copy would silently drop the reduction result.
value = torch.arange(4, dtype=torch.float32)
result = xllm_runtime.tp_all_reduce(value)
assert result.data_ptr() == value.data_ptr()
assert xllm_runtime.tp_all_to_all(value, 1) is None

options = module.Options()
assert options.model_impl == "native"
options.model_impl = "python"
assert options.model_impl == "python"
assert "xllm_weight_loader" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(artifact)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
