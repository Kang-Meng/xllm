# Copyright 2025-2026 The xLLM Authors.
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

import json
import os
import socket

import psutil
import xllm_export


def terminate_process(pid: int, timeout: int | float = 30) -> None:
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    children = parent.children(recursive=True)
    procs = children + [parent]

    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass

    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        _, port = s.getsockname()
    return port


def _normalize_model_impl(model_impl: str) -> str:
    # Keep in lockstep with ModelConfig::normalize_model_impl (the C++ owner of
    # these aliases): ""/"native" -> "native", "python"/"py" -> "python".
    if model_impl in ("", "native"):
        return "native"
    if model_impl in ("python", "py"):
        return "python"
    raise ValueError(f"unsupported model_impl: {model_impl}; expected native, python, or py")


def _normalize_offline_nnodes(nnodes: int) -> int:
    # Offline inference drives every visible device of a single node through
    # spawned workers, so multi-node offline requests are rejected here as a
    # Python exception instead of aborting inside the Master constructor.
    if nnodes <= 0:
        raise ValueError(f"nnodes must be positive, got {nnodes}")
    if nnodes > 1:
        raise ValueError(
            "offline inference does not support nnodes > 1; it drives every "
            "visible device of a single node. Use online serving for "
            "multi-node deployment."
        )
    return nnodes


def _configure_python_model_path() -> None:
    # The Python model executor imports the 'xllm.python' subpackage in an
    # embedded interpreter: rank 0 reuses the one xllm_export lives in, while
    # spawned workers create their own and therefore do not inherit this
    # process's sys.path. Publish the directory that holds the 'xllm' package
    # in the environment the workers inherit.
    #
    # __file__ resolves both shapes the package ships in, because utils.py
    # always sits at '<prefix>/xllm/pybind/utils.py': an installed wheel keeps
    # the package in site-packages, an in-tree build in the build directory,
    # and a source checkout in the repository. A pre-set value still wins, but
    # every published value is verified to hold the package, so a wrong path
    # fails here instead of in a spawned worker's import.
    path = os.environ.get("XLLM_PYTHON_MODEL_PATH")
    if not path:
        # utils.py lives three levels below the package root; verify the
        # result actually holds the package instead of trusting the layout,
        # so a relocated file fails loudly rather than handing workers a
        # wrong directory.
        path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    if not os.path.isdir(os.path.join(path, "xllm")):
        raise RuntimeError(
            f"XLLM_PYTHON_MODEL_PATH {path!r} does not contain the 'xllm' "
            "package; set it to the directory holding the package."
        )
    os.environ.setdefault("XLLM_PYTHON_MODEL_PATH", path)


def _configure_npu_runtime(device_index: int = 0) -> None:
    # Offline inference runs inside the caller's interpreter, so the
    # accelerator belongs to the host: bring torch_npu up on the device this
    # process uses. Native processes xLLM starts do the same from C++
    # (xllm/core/platform/npu/npu_python_runtime.cpp) because they have no
    # interpreter until xLLM creates one. No-op without torch_npu.
    try:
        import torch_npu
    except ImportError:
        return
    try:
        torch_npu._C._npu_init()
    except RuntimeError as error:
        if "already initialized" not in str(error).lower():
            raise
    # Bypassing torch_npu's lazy init leaves its bookkeeping stale: without
    # these flags a later torch.npu.* entry re-runs _npu_init() (harmless but
    # noisy) and, worse, a fork with a stale pid looks like a fresh process to
    # _after_fork. Keep this in sync with the C++ bootstrap script.
    import torch_npu.npu as npu_module

    npu_module._initialized = True
    npu_module._original_pid = os.getpid()
    torch_npu._C._npu_setDevice(device_index)


def _configure_python_runtime() -> None:
    # Everything the Python model executor needs from the caller's process: the
    # package root spawned workers import from, plus the accelerator state rank
    # 0 runs with (offline is single node, so rank 0 owns device 0).
    _configure_python_model_path()
    _configure_npu_runtime(0)


def _read_json(path: str) -> dict[str, object]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _infer_model_type(model_path: str) -> str:
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise ValueError("config.json or model_index.json is required for backend detection")
    data = _read_json(config_path)
    model_type = data.get("model_type") or data.get("model_name")
    if not model_type:
        raise ValueError("config.json must contain model_type or model_name")
    return str(model_type)


def _get_model_backend(model_type: str) -> str:
    get_backend = getattr(xllm_export, "get_model_backend", None)
    if not callable(get_backend):
        raise ValueError(
            "xllm_export.get_model_backend is not available. Please rebuild xllm_export or explicitly specify backend."
        )
    try:
        backend = get_backend(model_type)
    except Exception as exc:
        raise ValueError(f"Failed to resolve backend for model_type: {model_type}") from exc
    if not backend:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return backend


def _infer_model_type_and_backend(model_path: str) -> tuple[str | None, str]:
    model_index_path = os.path.join(model_path, "model_index.json")
    if os.path.exists(model_index_path):
        data = _read_json(model_index_path)
        if "_diffusers_version" in data:
            return None, "dit"

    model_type = _infer_model_type(model_path)
    return model_type, _get_model_backend(model_type)


def _configure_cpp_chat_template(
    use_cpp_chat_template: bool,
    model_type: str,
) -> None:
    configure = getattr(xllm_export, "configure_cpp_chat_template", None)
    if not callable(configure):
        raise ValueError("xllm_export.configure_cpp_chat_template is not available. Please rebuild xllm_export.")
    configure(use_cpp_chat_template, model_type)
