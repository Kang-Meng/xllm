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

"""Self-contained ctypes wrappers for the xLLM KDA context-parallel op set.

The ops live in the standard ``custom_xllm_math`` opp vendor built from
``third_party/xllm_ops`` (the P1 port, task 10-15-kda-cp-feasibility,
research/n3-5-p1-port.md):

* ``aclnnXllmChunkKdaFwdPrepare`` -- the prepare-only WY factor pass of the
  fused chunk KDA forward: same factor emission, no state/output semantics,
  ~25-50% cheaper than a fused ``disable_recompute`` run (ma1). This is the
  summary factor source of the KDA CP prefill path.
* ``aclnnPreProcessFwdKernelMerged`` -- folds those factors into the
  per-span affine summary ``hm = [E | M]`` consumed by the CP merge fold.
* ``aclnnMergeFwdBwdKernel`` -- composes prior-segment summaries into a
  prefix state in ONE launch (the CP merge fold; n1 section 1).

The vendor is resolved from the STANDARD opp install
(``${ASCEND_HOME_PATH}/opp/vendors/custom_xllm_math``, the target of the
xllm_ops ``build.sh`` install step; ``ASCEND_HOME_PATH`` is the CANN
runtime root every engine process inherits). The CANN runtime resolves the
kernel binaries from the same install through the vendors ``config.ini``
``load_priority`` chain, so no path environment is needed at all. Importing
this module never raises and never touches the NPU runtime;
:func:`available` probes lazily and returns False when the vendor library
or symbols are missing, so the KDA CP gate falls back to the legacy merged
path safely.

The marshalling is a thin, faithful port of the fla-nu source-tree wrappers
``_aclnn_ctypes.py::npu_chunk_kda_fwd_prepare`` and
``npu_pre_process_fwd_kernel_merged`` (fla-nu @ f623f2b0): identical
argument order to the aclnn entry points, identical
workspace-size-then-run two-step launch, identical output-slot allocation.
Only the entry symbol names are our renamed ones (the ma2-1 skeleton
``ma21_xllm_prepare.py`` validated the exact sequence on the real device).
The merge wrapper ports ``npu_merge_fwd_bwd_kernel`` with ONE deviation:
``h`` is allocated and returned by this wrapper instead of being passed in,
because the kernel never reads the caller's ``h`` values (source-pinned,
n1 section 1: the vector stage seeds the chain from the first prior row's
E block and only ever stores to the user ``h``).

DESCRIPTOR/OUTPUT CACHE (n1, the N-0a-validated design): every launch goes
through a per-op bounded LRU keyed on the full marshalling signature
(shapes, dtypes, cu, flags, scalars, device -- strides excluded because
caller values are always copied into our contiguous static buffers). A
cache entry holds pre-built acl tensor descriptors, pre-built int arrays,
the frozen ctypes argument list, static output tensors and the workspace
tensor. Per call: inputs are copied into the static input buffers ONLY
when the passed tensor is not already the cached one (identity ``is`` --
the engine's produce-consume pattern chains prepare's static outputs
straight into ppfm with zero copies), then a FRESH GetWorkspaceSize (the
aclnn executor is single-use; re-launching one segfaults, N-0a section 5)
and the launch. Aliasing contract: static outputs are OVERWRITTEN by the
next cache-hit call on the same key, and an input slot that adopted a
reusable buffer is written in place when a different tensor is passed --
consumers must clone anything they keep past the next call. The cache is
always on (historically measured bit-exact against the since-removed fresh
per-call path; the NPU test now pins the static-buffer contract: object
identity across hits, in-place input mutation refreshing the outputs, and
bit-exact rebuild across LRU eviction).

NUMERICS: the prepare factors correspond to the deployed fused emission but
are NOT bit-exact -- last-bit rounding differences only (ma1: gk 1.79e-07
abs ~= 1 fp32 ulp, bf16 slots <= 9.8e-04 ~= 1 bf16 ulp; the hm summary
shifts by ~1.7e-4 rel_fro, 4x inside the accepted cp4 compose divergence).
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

# The standard opp vendor install: third_party/xllm_ops' build.sh installs
# the custom_xllm_math vendor into ${ASCEND_HOME_PATH}/opp/vendors and
# registers it in the vendors config.ini, so both the kernel binaries (CANN
# runtime) and the host aclnn entry points (this module) resolve from the
# same root with zero path configuration. ASCEND_HOME_PATH is the CANN
# runtime root every engine process inherits -- a standard CANN variable,
# not an xllm configuration knob.
_VENDOR_NAME = "custom_xllm_math"
_VENDOR_OPAPI_RELATIVE_PATH = os.path.join("opp", "vendors", _VENDOR_NAME, "op_api", "lib", "libcust_opapi.so")

_PREPARE_SYMBOL = "aclnnXllmChunkKdaFwdPrepare"
_PPFM_SYMBOL = "aclnnPreProcessFwdKernelMerged"
_MERGE_SYMBOL = "aclnnMergeFwdBwdKernel"

# Base aclnn plumbing every launch needs; these resolve through our
# libcust_opapi.so dependency chain into the CANN runtime (verified against
# the standalone vendor on the real container -- ma2-2).
_BASE_SYMBOLS = (
    "aclCreateTensor",
    "aclDestroyTensor",
    "aclCreateIntArray",
    "aclDestroyIntArray",
)

_KDA_FWD_SUPPORTED_KV_DIMS = frozenset({64, 128})

_ACL_SUCCESS = 0
_ACL_FORMAT_NCHW = 0
_ACL_FORMAT_ND = 2
_ACL_FORMAT_NCDHW = 30
_ACL_FORMAT_NCL = 47

_ACL_FORMAT_BY_NAME = {
    "NCHW": _ACL_FORMAT_NCHW,
    "ND": _ACL_FORMAT_ND,
    "NCDHW": _ACL_FORMAT_NCDHW,
    "NCL": _ACL_FORMAT_NCL,
}

_DTYPE_TO_ACL = {
    torch.float32: 0,  # ACL_FLOAT
    torch.float16: 1,  # ACL_FLOAT16
    torch.int8: 2,  # ACL_INT8
    torch.int32: 3,  # ACL_INT32
    torch.uint8: 4,  # ACL_UINT8
    torch.int16: 6,  # ACL_INT16
    torch.int64: 9,  # ACL_INT64
    torch.float64: 11,  # ACL_DOUBLE
    torch.bool: 12,  # ACL_BOOL
    torch.bfloat16: 27,  # ACL_BF16
}


# ---------------------------------------------------------------------------
# Vendor library loading (graceful unavailability)
# ---------------------------------------------------------------------------


_libraries: dict[str, ctypes.CDLL] = {}


def _opapi_library_paths() -> list[str]:
    """Standard opp vendor library paths, in CANN discovery order.

    ``ASCEND_HOME_PATH`` is the CANN runtime root present in every NPU
    engine process (the same root the CANN runtime resolves ``opp/vendors``
    from). When it is unset, no candidate is produced and the KDA CP gate
    safely falls back to the legacy merged path.
    """
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if not ascend_home:
        return []
    return [os.path.join(ascend_home, _VENDOR_OPAPI_RELATIVE_PATH)]


def _opapi_library() -> Optional[ctypes.CDLL]:
    """Load the standard vendor's libcust_opapi.so once per path; None = unavailable.

    Mirrors the deployed fla-nu loader's mode (RTLD_GLOBAL | RTLD_NOW |
    RTLD_NODELETE). Failed probes are not cached: the probe is cheap (one
    standard-path lookup plus one dlopen attempt) and the KDA CP gate asks
    once per process, so a launcher that fixes the environment between two
    processes is never served a stale failure.
    """
    for library_path in _opapi_library_paths():
        library = _libraries.get(library_path)
        if library is not None:
            return library
        mode = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0) | getattr(os, "RTLD_NODELETE", 0)
        try:
            candidate = ctypes.CDLL(library_path, mode=mode)
            for symbol in (
                *_BASE_SYMBOLS,
                _PREPARE_SYMBOL,
                f"{_PREPARE_SYMBOL}GetWorkspaceSize",
                _PPFM_SYMBOL,
                f"{_PPFM_SYMBOL}GetWorkspaceSize",
                _MERGE_SYMBOL,
                f"{_MERGE_SYMBOL}GetWorkspaceSize",
            ):
                getattr(candidate, symbol)
        except (OSError, AttributeError) as exc:
            # Distinguish "vendor absent" (a CPU-only build: expected, debug)
            # from "vendor present but unusable" (a deployment error worth a
            # warning); the gate still degrades gracefully either way.
            log = logger.debug if not any(os.path.exists(p) for p in _opapi_library_paths()) else logger.warning
            log("KDA CP op library unusable at %s: %s", library_path, exc)
            continue
        _libraries[library_path] = candidate
        return candidate
    return None


def available() -> bool:
    """Whether the xLLM KDA CP op set can run in this process.

    False when the standard ``custom_xllm_math`` opp vendor (or its entry
    symbols) is missing. Never raises and never touches the device:
    the KDA CP gate calls this before deciding whether the two-pass CP
    prefill path is eligible at all.
    """
    return _opapi_library() is not None


# ---------------------------------------------------------------------------
# Minimal aclnn ctypes runtime (ported from the fla-nu _runtime.py helpers)
# ---------------------------------------------------------------------------


def _shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _stride(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.stride())


def _storage_numel(tensor: torch.Tensor) -> int:
    try:
        nbytes = tensor.untyped_storage().nbytes()
    except AttributeError:
        nbytes = tensor.storage().nbytes()
    return int(nbytes // tensor.element_size())


def _storage_data_ptr(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.untyped_storage().data_ptr())
    except AttributeError:
        return int(tensor.storage().data_ptr())


def _acl_format(tensor: torch.Tensor) -> int:
    # torch_npu can report the real NPU layout format of a tensor. This
    # module must not import torch_npu itself, so only reuse it when some
    # other code already loaded it; otherwise infer conservatively from the
    # logical rank.
    try:
        torch_npu = sys.modules.get("torch_npu")
        if torch_npu is None:
            raise RuntimeError("torch_npu is not loaded")
        npu_format = torch_npu.get_npu_format(tensor)
    except Exception:
        npu_format = None

    if isinstance(npu_format, str):
        acl_format_value = _ACL_FORMAT_BY_NAME.get(npu_format)
        if acl_format_value is not None:
            return acl_format_value
    elif npu_format is not None:
        try:
            return int(npu_format)
        except (TypeError, ValueError):
            pass

    dim = tensor.dim()
    if dim == 3:
        return _ACL_FORMAT_NCL
    if dim == 4:
        return _ACL_FORMAT_NCHW
    if dim == 5:
        return _ACL_FORMAT_NCDHW
    return _ACL_FORMAT_ND


def _ensure_npu_tensor(tensor: Any, name: str) -> torch.Tensor:
    if tensor is None:
        return tensor
    if not hasattr(tensor, "device") or tensor.device.type != "npu":
        raise TypeError(f"{name} must be a torch NPU tensor, got {type(tensor)!r}.")
    return tensor


def _optional_bool(value: Any, default: bool) -> bool:
    return default if value is None else bool(value)


def _optional_int(value: Any, default: int) -> int:
    return default if value is None else int(value)


def _optional_float(value: Any, default: float) -> float:
    return default if value is None else float(value)


def _empty(shape_: Iterable[int], like: torch.Tensor, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    return torch.empty(tuple(int(dim) for dim in shape_), device=like.device, dtype=dtype or like.dtype)


def _zeros(shape_: Iterable[int], like: torch.Tensor, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    return torch.zeros(tuple(int(dim) for dim in shape_), device=like.device, dtype=dtype or like.dtype)


def _npu_device_index(device: Any) -> int:
    if getattr(device, "type", None) != "npu":
        raise TypeError(f"Expected a torch NPU device, got {device!r}.")
    index = getattr(device, "index", None)
    if index is not None:
        return int(index)
    return int(torch.npu.current_device())


@contextmanager
def _npu_device_guard(device: Any):
    """Switch to the tensor's NPU for the duration of one aclnn call."""
    device_index = _npu_device_index(device)
    device_guard = getattr(torch.npu, "device", None)
    if device_guard is not None:
        with device_guard(device_index):
            yield device_index
        return

    # Early NPU python runtimes only expose current_device/set_device.
    previous_index = int(torch.npu.current_device())
    if previous_index != device_index:
        torch.npu.set_device(device_index)
    try:
        yield device_index
    finally:
        if previous_index != device_index:
            torch.npu.set_device(previous_index)


def _current_stream_ptr(device: Any) -> int:
    device_index = _npu_device_index(device)
    try:
        stream = torch.npu.current_stream(device_index)
    except TypeError:
        # current_stream() without arguments only; the caller already holds
        # the device guard.
        with _npu_device_guard(device):
            stream = torch.npu.current_stream()
    return int(stream.npu_stream)


class _AclTensor:
    """An aclTensor descriptor kept alive for one aclnn call."""

    def __init__(
        self,
        runtime: _AclnnRuntime,
        tensor: torch.Tensor,
        *,
        storage_shape_override: Optional[Iterable[int]] = None,
        acl_format_override: Optional[int] = None,
    ) -> None:
        tensor = _ensure_npu_tensor(tensor, "tensor")
        acl_dtype = _DTYPE_TO_ACL.get(tensor.dtype)
        if acl_dtype is None:
            raise TypeError(f"Unsupported dtype for aclnn tensor descriptor: {tensor.dtype}.")
        self._runtime = runtime
        self._tensor = tensor
        self._shape = (ctypes.c_int64 * tensor.dim())(*_shape(tensor))
        self._stride = (ctypes.c_int64 * tensor.dim())(*_stride(tensor))
        descriptor_storage_shape = (
            (_storage_numel(tensor),)
            if storage_shape_override is None
            else tuple(int(dim) for dim in storage_shape_override)
        )
        self._storage_shape = (ctypes.c_int64 * len(descriptor_storage_shape))(*descriptor_storage_shape)
        self.ptr = runtime.acl_create_tensor(
            self._shape,
            ctypes.c_uint64(tensor.dim()),
            ctypes.c_int(acl_dtype),
            self._stride,
            ctypes.c_int64(int(tensor.storage_offset())),
            ctypes.c_int(_acl_format(tensor) if acl_format_override is None else acl_format_override),
            self._storage_shape,
            ctypes.c_uint64(len(descriptor_storage_shape)),
            ctypes.c_void_p(_storage_data_ptr(tensor)),
        )
        if not self.ptr:
            raise RuntimeError("aclCreateTensor returned nullptr.")

    def destroy(self) -> None:
        if self.ptr:
            self._runtime.acl_destroy_tensor(self.ptr)
        self.ptr = None


class _AclIntArray:
    """An aclIntArray descriptor for optional host sequence inputs."""

    def __init__(self, runtime: _AclnnRuntime, values: Optional[Sequence[int]]) -> None:
        self._runtime = runtime
        self.ptr = None
        if values is None:
            return
        values = tuple(int(value) for value in values)
        if not values:
            return
        self._values = (ctypes.c_int64 * len(values))(*values)
        self.ptr = runtime.acl_create_int_array(self._values, ctypes.c_uint64(len(values)))
        if not self.ptr:
            raise RuntimeError("aclCreateIntArray returned nullptr.")

    def destroy(self) -> None:
        if self.ptr:
            self._runtime.acl_destroy_int_array(self.ptr)
        self.ptr = None


class _CallContext:
    """Tracks descriptors created while building one aclnn call."""

    def __init__(self, runtime: _AclnnRuntime, device: Any) -> None:
        self.runtime = runtime
        self.device = device
        self.device_index = _npu_device_index(device)
        self.resources: list[Any] = []

    def tensor(
        self,
        tensor: Any,
        name: str = "tensor",
        *,
        storage_shape_override: Optional[Iterable[int]] = None,
        acl_format_override: Optional[int] = None,
    ) -> ctypes.c_void_p:
        if tensor is None:
            return ctypes.c_void_p()
        tensor = _ensure_npu_tensor(tensor, name)
        tensor_device_index = _npu_device_index(tensor.device)
        if tensor_device_index != self.device_index:
            raise ValueError(
                f"{name} must be on npu:{self.device_index}, got npu:{tensor_device_index}; "
                "one aclnn call cannot mix tensors from different NPU devices."
            )
        descriptor = _AclTensor(
            self.runtime,
            tensor,
            storage_shape_override=storage_shape_override,
            acl_format_override=acl_format_override,
        )
        self.resources.append(descriptor)
        return ctypes.c_void_p(descriptor.ptr)

    def int_array(self, values: Optional[Sequence[int]]) -> ctypes.c_void_p:
        descriptor = _AclIntArray(self.runtime, values)
        self.resources.append(descriptor)
        return ctypes.c_void_p(descriptor.ptr or 0)

    def destroy(self) -> None:
        for resource in reversed(self.resources):
            resource.destroy()
        self.resources.clear()


class _AclnnRuntime:
    """Two-phase aclnn launcher bound to our vendor library only."""

    def __init__(self, library: ctypes.CDLL) -> None:
        self._library = library
        self._symbols: dict[str, Any] = {}
        # aclGetRecentErrMsg surfaces the CANN ErrorManager text when an
        # aclnn call fails; bound lazily, degrading to the bare status code.
        self._get_recent_err_msg: Optional[Any] = None
        self.acl_create_tensor = self.symbol("aclCreateTensor")
        self.acl_create_tensor.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.acl_create_tensor.restype = ctypes.c_void_p

        self.acl_destroy_tensor = self.symbol("aclDestroyTensor")
        self.acl_destroy_tensor.argtypes = [ctypes.c_void_p]
        self.acl_destroy_tensor.restype = ctypes.c_int

        self.acl_create_int_array = self.symbol("aclCreateIntArray")
        self.acl_create_int_array.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64]
        self.acl_create_int_array.restype = ctypes.c_void_p

        self.acl_destroy_int_array = self.symbol("aclDestroyIntArray")
        self.acl_destroy_int_array.argtypes = [ctypes.c_void_p]
        self.acl_destroy_int_array.restype = ctypes.c_int

    def symbol(self, name: str) -> Any:
        if name in self._symbols:
            return self._symbols[name]
        symbol = getattr(self._library, name)
        self._symbols[name] = symbol
        return symbol

    def _recent_error_message(self) -> str:
        if self._get_recent_err_msg is None:
            try:
                get_recent_err_msg = self.symbol("aclGetRecentErrMsg")
            except AttributeError:
                # Not exported by our libcust_opapi.so; fall back to the
                # CANN runtime library.
                try:
                    get_recent_err_msg = ctypes.CDLL("libascendcl.so").aclGetRecentErrMsg
                except (OSError, AttributeError):
                    return ""
            get_recent_err_msg.restype = ctypes.c_char_p
            self._get_recent_err_msg = get_recent_err_msg
        try:
            raw = self._get_recent_err_msg()
        except Exception:
            return ""
        if not raw:
            return ""
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace").strip()
        return str(raw).strip()

    def _aclnn_error(self, name: str, ret: int) -> str:
        message = f"{name} failed with aclnnStatus={ret}."
        err_msg = self._recent_error_message()
        if err_msg:
            message += f" Error message: {err_msg}"
        return message

    def call(
        self, name: str, args: Sequence[object], device: Any, workspace: Optional[torch.Tensor] = None
    ) -> torch.Tensor | None:
        # aclnn calling convention: the first phase creates an executor and
        # returns the workspace size; the second phase passes the workspace
        # and the stream pointer and enqueues the actual launch. The
        # workspace is a plain torch NPU tensor so it follows the target
        # device's PyTorch allocator. A caller-provided workspace (the
        # descriptor cache's static buffer) is reused as long as it is at
        # least as large as the fresh size; the executor is ALWAYS fresh
        # (single-use; N-0a section 5).
        get_workspace = self.symbol(f"{name}GetWorkspaceSize")
        launch = self.symbol(name)
        get_workspace.restype = ctypes.c_int
        launch.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p]
        launch.restype = ctypes.c_int
        workspace_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()

        ret = get_workspace(*args, ctypes.byref(workspace_size), ctypes.byref(executor))
        if ret != _ACL_SUCCESS:
            raise RuntimeError(self._aclnn_error(f"{name}GetWorkspaceSize", ret))

        if workspace_size.value and (workspace is None or workspace.nbytes < int(workspace_size.value)):
            workspace = torch.empty((int(workspace_size.value),), dtype=torch.uint8, device=device)
        workspace_ptr = (
            ctypes.c_void_p(int(workspace.data_ptr()))
            if workspace is not None and workspace_size.value
            else ctypes.c_void_p()
        )

        ret = launch(
            workspace_ptr,
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(_current_stream_ptr(device)),
        )
        if ret != _ACL_SUCCESS:
            # The executor is handed to the stream only by a SUCCESSFUL launch;
            # on failure it is still owned here, so release it before raising
            # (CANN exposes aclDestroyAclOpExecutor; resolved opportunistically
            # so a CANN build without it keeps working).
            self._destroy_executor(executor)
            raise RuntimeError(self._aclnn_error(name, ret))
        return workspace

    def _destroy_executor(self, executor: ctypes.c_void_p) -> None:
        try:
            destroy = self.symbol("aclDestroyAclOpExecutor")
        except AttributeError:
            return
        destroy.argtypes = [ctypes.c_void_p]
        destroy.restype = ctypes.c_int
        destroy(executor)


_RUNTIME: Optional[_AclnnRuntime] = None


def _runtime() -> _AclnnRuntime:
    """The launcher singleton; raises a clear error when unavailable."""
    global _RUNTIME
    if _RUNTIME is None:
        library = _opapi_library()
        if library is None:
            raise RuntimeError(
                "xllm KDA CP ops are unavailable: the standard "
                "custom_xllm_math opp vendor is not installed under "
                "${ASCEND_HOME_PATH}/opp/vendors (build and install it from "
                "third_party/xllm_ops via its build.sh) or does not export "
                "the KDA CP entry symbols."
            )
        _RUNTIME = _AclnnRuntime(library)
    return _RUNTIME


def _call_device(outputs: Sequence[Any]) -> Any:
    device = None
    device_index = None
    for index, output in enumerate(outputs):
        if output is None:
            continue
        output = _ensure_npu_tensor(output, f"output[{index}]")
        output_device_index = _npu_device_index(output.device)
        if device_index is None:
            device = output.device
            device_index = output_device_index
        elif output_device_index != device_index:
            raise ValueError(
                f"output[{index}] must be on npu:{device_index}, got npu:{output_device_index}; "
                "one aclnn call cannot mix tensors from different NPU devices."
            )
    if device is None:
        raise ValueError("An aclnn call must have at least one NPU output tensor.")
    return device


# ---------------------------------------------------------------------------
# Descriptor/output cache (n1; the N-0a-validated maximal cache shape)
# ---------------------------------------------------------------------------

# Live static buffers by object identity. Values are weak so an evicted
# entry's tensors die (and deregister) as soon as no other entry or caller
# holds them; CPython removes the dict entry synchronously when the object
# is freed, so a recycled id can never alias a stale registration.
_REUSABLE_INPUTS: weakref.WeakValueDictionary[int, torch.Tensor] = weakref.WeakValueDictionary()

# Engine realistic working set is 1-2 shapes per op (chunked prefill is
# uniform); 4 entries absorb a mixed batch without unbounded growth.
_DESC_CACHE_CAPACITY = 4

_OP_CACHES: dict[str, OrderedDict[Any, _CachedOpCall]] = {}


def register_reusable_input(tensor: torch.Tensor) -> None:
    """Promise that ``tensor`` is a stable input buffer reused across calls.

    The caller keeps the tensor alive and keeps passing it (or deliberately
    passes something else, which then copies in place over it). A cached op
    entry built while ``tensor`` is an input ADOPTS it as its static input
    slot instead of copying: the descriptors bake the buffer's address, so
    subsequent calls that pass the same object launch with zero copies.
    """
    _REUSABLE_INPUTS[id(tensor)] = tensor


class _CachedOpCall:
    """One cached aclnn launch site.

    Holds the frozen ctypes argument list (with pre-built descriptors and
    int arrays), the static input slots and output tensors, and the
    workspace buffer. The aclnn executor itself is NEVER cached (it is
    single-use; re-launching one segfaults -- N-0a section 5): every
    launch goes through a fresh GetWorkspaceSize.
    """

    __slots__ = ("name", "device", "args", "outputs", "output_is_tuple", "input_slots", "resources", "workspace")

    def __init__(
        self,
        name: str,
        device: Any,
        args: list[Any],
        outputs: tuple[Any, ...],
        input_slots: tuple[Any, ...],
        output_is_tuple: bool,
    ) -> None:
        self.name = name
        self.device = device
        self.args = args
        self.outputs = outputs
        self.output_is_tuple = output_is_tuple
        self.input_slots = input_slots
        self.resources: list[Any] = []
        self.workspace: Optional[torch.Tensor] = None

    def adopt_resources(self, resources: list[Any]) -> None:
        self.resources = list(resources)

    def launch(self, runtime: _AclnnRuntime) -> None:
        # Fresh executor per call; reuse (and grow if ever needed) the
        # static workspace buffer.
        self.workspace = runtime.call(self.name, self.args, self.device, workspace=self.workspace)

    def result(self) -> Any:
        return self.outputs if self.output_is_tuple else self.outputs[0]

    def destroy(self) -> None:
        for resource in reversed(self.resources):
            resource.destroy()
        self.resources = []
        self.workspace = None
        self.outputs = ()
        self.input_slots = ()


def _tensor_signature(tensor: Optional[torch.Tensor]) -> Optional[tuple]:
    """Hashable marshalling signature of one tensor for the cache key.

    Strides and storage offsets are deliberately excluded: caller values
    are always copied into our contiguous static buffers, so only the
    logical shape, dtype and device identity matter. The device is encoded
    as a string so key derivation never initializes the NPU runtime (the
    unavailable-vendor error must stay the first failure a CPU-tensor
    caller sees) and never resolves an index-less device.
    """
    if tensor is None:
        return None
    return (tuple(int(dim) for dim in tensor.shape), tensor.dtype, str(tensor.device))


def _static_input_buffer(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """The static input slot for one caller tensor.

    A registered reusable buffer (another cache entry's output, or a caller
    buffer declared via :func:`register_reusable_input`) is adopted as-is:
    its address is stable and its owner keeps it alive. Anything else is
    copied once into a fresh contiguous buffer; later calls with different
    tensors copy in place over the same buffer, so caller strides and
    storage offsets never leak into the frozen descriptors.
    """
    if tensor is None:
        return None
    if id(tensor) in _REUSABLE_INPUTS:
        return tensor
    static = torch.empty(tuple(int(dim) for dim in tensor.shape), dtype=tensor.dtype, device=tensor.device)
    static.copy_(tensor)
    return static


def _build_cached_op_call(
    runtime: _AclnnRuntime,
    name: str,
    inputs: tuple[Any, ...],
    make_outputs: Callable[[], Any],
    build_args: Callable[[Any, tuple[Any, ...], tuple[Any, ...]], list[Any]],
) -> _CachedOpCall:
    outputs = make_outputs()
    outputs_tuple = outputs if isinstance(outputs, tuple) else (outputs,)
    device = _call_device(outputs_tuple)
    static_inputs = tuple(_static_input_buffer(tensor) for tensor in inputs)
    context = _CallContext(runtime, device)
    entry: Optional[_CachedOpCall] = None
    with _npu_device_guard(device):
        try:
            args = build_args(context, static_inputs, outputs_tuple)
            entry = _CachedOpCall(name, device, args, outputs_tuple, static_inputs, isinstance(outputs, tuple))
            entry.adopt_resources(context.resources)
            context.resources = []
            entry.launch(runtime)
        except BaseException:
            if entry is not None:
                entry.destroy()
            raise
        finally:
            context.destroy()
    for tensor in (*static_inputs, *outputs_tuple):
        if tensor is not None:
            _REUSABLE_INPUTS[id(tensor)] = tensor
    return entry


def _launch_op(
    name: str,
    inputs: tuple[Any, ...],
    make_outputs: Callable[[], Any],
    build_args: Callable[[Any, tuple[Any, ...], tuple[Any, ...]], list[Any]],
    key: Any,
) -> Any:
    """Run one aclnn op through the descriptor/output cache.

    ``build_args(ctx, static_inputs, static_outputs)`` must build the ctypes
    argument list referencing ONLY the given static tensors and values
    captured from the (already validated) call. Not thread-safe: the KDA
    CP path is single-threaded per process.
    """
    runtime = _runtime()
    cache = _OP_CACHES.setdefault(name, OrderedDict())
    entry = cache.get(key)
    if entry is None:
        entry = _build_cached_op_call(runtime, name, tuple(inputs), make_outputs, build_args)
        cache[key] = entry
        while len(cache) > _DESC_CACHE_CAPACITY:
            _, evicted = cache.popitem(last=False)
            evicted.destroy()
    else:
        cache.move_to_end(key)
        for caller, slot in zip(inputs, entry.input_slots):
            if caller is not None and slot is not None and caller is not slot:
                slot.copy_(caller)
        with _npu_device_guard(entry.device):
            entry.launch(runtime)
    return entry.result()


# ---------------------------------------------------------------------------
# Op: aclnnXllmChunkKdaFwdPrepare (the renamed prepare-only factor pass)
# ---------------------------------------------------------------------------


def _kda_ceil_div(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _kda_build_chunk_indices(cu_seqlens: Any, chunk_size: int) -> Optional[tuple[int, ...]]:
    if cu_seqlens is None:
        return None
    cu = tuple(int(value) for value in cu_seqlens)
    indices = []
    for seq in range(len(cu) - 1):
        seq_len = cu[seq + 1] - cu[seq]
        for chunk in range(_kda_ceil_div(seq_len, chunk_size)):
            indices.extend((seq, chunk))
    return tuple(indices)


def xllm_chunk_kda_fwd_prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
    *,
    layout: str = "BSND",
    cu_seqlens: Optional[Sequence[int]] = None,
    chunk_indices: Optional[Sequence[int]] = None,
    safe_gate: bool = False,
    lower_bound: Optional[float] = None,
    use_gate_in_kernel: bool = False,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    use_exp2: bool = True,
    backward_mode: str = "none",
) -> tuple[Optional[torch.Tensor], ...]:
    """Run aclnnXllmChunkKdaFwdPrepare: the prepare-stage WY factors.

    Thin port of the fla-nu ``npu_chunk_kda_fwd_prepare`` wrapper under the
    renamed symbol (ma2-1). Consumes the same raw inputs as the fused
    ``chunk_kda_fwd`` (no ``initial_state`` / ``output_final_state`` /
    ``disable_recompute`` / ``state_v_first``: prepare has no state or
    output semantics -- ma1 section 7 flag mapping). Returns the 13-slot
    prepare tuple ``(gk, Aqk, Akk, w, u, qg, kg, qg_scaled, q_hat, k_hat,
    q_rstd, k_rstd, beta_eff)``; slots outside ``backward_mode`` are None.
    The ma1 fused->prepare slot correspondence is gk 2->0, aqk 3->1,
    akk 4->2, w 5->3, u 6->4, qg 7->5, kg 8->6.
    """
    layout = str(layout)
    if layout not in {"BSND", "BNSD", "TND", "NTD"}:
        raise RuntimeError("xllm_chunk_kda_fwd_prepare: layout must be BSND, BNSD, TND or NTD.")
    chunk_size = int(chunk_size)
    if chunk_size not in {64, 128}:
        raise RuntimeError("xllm_chunk_kda_fwd_prepare: chunk_size must be 64 or 128.")

    is_rank3 = layout in {"TND", "NTD"}
    is_sequence_major = layout in {"BSND", "TND"}
    q_shape, k_shape, v_shape, g_shape, beta_shape = map(_shape, (q, k, v, g, beta))
    expected_rank = 3 if is_rank3 else 4
    if any(len(shape) != expected_rank for shape in (q_shape, k_shape, v_shape, g_shape)):
        raise RuntimeError("xllm_chunk_kda_fwd_prepare: q/k/v/g rank does not match layout.")
    if q_shape != k_shape or len(beta_shape) != expected_rank - 1:
        raise RuntimeError("xllm_chunk_kda_fwd_prepare: q/k shapes must match and beta rank must match layout.")

    if is_rank3:
        seqlen, h_num, k_dim = q_shape
        hv_num, v_dim = v_shape[1], v_shape[2]
        batch = 1
    else:
        if is_sequence_major:
            batch, seqlen, h_num, k_dim = q_shape
            hv_num, v_dim = v_shape[2], v_shape[3]
        else:
            batch, h_num, seqlen, k_dim = q_shape
            hv_num, v_dim = v_shape[1], v_shape[3]
    if h_num <= 0 or hv_num < h_num or hv_num % h_num != 0 or h_num > 128 or hv_num > 128:
        raise RuntimeError("xllm_chunk_kda_fwd_prepare: H/HV must satisfy 0 < H <= HV <= 128 and HV % H == 0.")
    if q.dtype not in {torch.float16, torch.bfloat16} or k.dtype != q.dtype or v.dtype != q.dtype:
        raise RuntimeError("xllm_chunk_kda_fwd_prepare: q/k/v must use the same float16 or bfloat16 dtype.")
    if g.dtype not in {torch.float32, torch.bfloat16} or beta.dtype not in {torch.float32, torch.bfloat16}:
        raise RuntimeError("xllm_chunk_kda_fwd_prepare: g and beta must be float32 or bfloat16.")
    if k_dim != v_dim or k_dim not in _KDA_FWD_SUPPORTED_KV_DIMS:
        raise RuntimeError(
            f"xllm_chunk_kda_fwd_prepare: K/V must both be 64 or both be 128, but got K={k_dim}, V={v_dim}."
        )
    epsilon = _optional_float(epsilon, 1e-6)
    if not (float(epsilon) > 0.0):
        raise RuntimeError("xllm_chunk_kda_fwd_prepare: epsilon must be a positive finite number.")
    use_gate_in_kernel = _optional_bool(use_gate_in_kernel, False)
    safe_gate = _optional_bool(safe_gate, False)
    use_qk_l2norm_in_kernel = _optional_bool(use_qk_l2norm_in_kernel, False)
    use_beta_sigmoid_in_kernel = _optional_bool(use_beta_sigmoid_in_kernel, False)
    allow_neg_eigval = _optional_bool(allow_neg_eigval, False)
    use_exp2 = _optional_bool(use_exp2, True)
    if use_gate_in_kernel:
        if A_log is None or _shape(A_log) != (hv_num,) or A_log.dtype != torch.float32:
            raise RuntimeError("xllm_chunk_kda_fwd_prepare: A_log must be float32 [HV] when use_gate_in_kernel=True.")
        if dt_bias is not None and (_shape(dt_bias) != (hv_num * k_dim,) or dt_bias.dtype != torch.float32):
            raise RuntimeError("xllm_chunk_kda_fwd_prepare: dt_bias must be float32 [HV*K].")
    lower_bound = _optional_float(lower_bound, -5.0)
    if use_gate_in_kernel and safe_gate and not (-5.0 <= lower_bound < 0.0):
        raise RuntimeError("xllm_chunk_kda_fwd_prepare: lower_bound must be in [-5, 0) for safe gate.")
    cu = None if cu_seqlens is None else tuple(int(value) for value in cu_seqlens)
    if cu is not None:
        if len(cu) < 2 or cu[0] != 0 or cu[-1] != seqlen or any(a > b for a, b in zip(cu, cu[1:])):
            raise RuntimeError("xllm_chunk_kda_fwd_prepare: cu_seqlens must be nondecreasing, start at 0 and end at T.")
        if not is_rank3 and batch != 1:
            raise RuntimeError("xllm_chunk_kda_fwd_prepare: rank4 varlen input requires B=1.")
    canonical_indices = _kda_build_chunk_indices(cu, chunk_size)
    indices = canonical_indices if chunk_indices is None else tuple(int(value) for value in chunk_indices)
    if indices is not None and indices != canonical_indices:
        raise RuntimeError("xllm_chunk_kda_fwd_prepare: chunk_indices must use canonical sequence-major order.")

    k_shape_head = (hv_num, seqlen, k_dim) if is_rank3 else (batch, hv_num, seqlen, k_dim)
    qk_shape_head = (h_num, seqlen, k_dim) if is_rank3 else (batch, h_num, seqlen, k_dim)
    v_shape_head = (hv_num, seqlen, v_dim) if is_rank3 else (batch, hv_num, seqlen, v_dim)
    matrix_shape = (hv_num, seqlen, chunk_size) if is_rank3 else (batch, hv_num, seqlen, chunk_size)
    qk_scalar_shape = (h_num, seqlen) if is_rank3 else (batch, h_num, seqlen)
    value_scalar_shape = (hv_num, seqlen) if is_rank3 else (batch, hv_num, seqlen)

    # Slot shapes/dtypes and the backward_mode tier contract (mirrors the
    # op document: gk/Aqk/w/u/kg/qg_scaled are mandatory in every mode;
    # Akk appears from the "forward" tier; the aux slots q_hat/k_hat/q_rstd/
    # k_rstd/beta_eff only in "recompute"/"save").
    expected_outputs = {
        "gk": (k_shape_head, torch.float32),
        "Aqk": (matrix_shape, q.dtype),
        "Akk": (matrix_shape, q.dtype),
        "w": (k_shape_head, q.dtype),
        "u": (v_shape_head, q.dtype),
        "qg": (k_shape_head, q.dtype),
        "kg": (k_shape_head, q.dtype),
        "qg_scaled": (k_shape_head, q.dtype),
        "q_hat": (qk_shape_head, q.dtype),
        "k_hat": (qk_shape_head, q.dtype),
        "q_rstd": (qk_scalar_shape, torch.float32),
        "k_rstd": (qk_scalar_shape, torch.float32),
        "beta_eff": (value_scalar_shape, torch.float32),
    }
    mode_slots = {
        "none": ("gk", "Aqk", "w", "u", "kg", "qg_scaled"),
        "forward": ("gk", "Aqk", "Akk", "w", "u", "kg", "qg_scaled"),
        "recompute": (
            "gk",
            "Aqk",
            "Akk",
            "w",
            "u",
            "kg",
            "qg_scaled",
            "q_hat",
            "k_hat",
            "q_rstd",
            "k_rstd",
            "beta_eff",
        ),
        "save": tuple(expected_outputs),
    }
    backward_mode = str(backward_mode).lower()
    if backward_mode not in mode_slots:
        raise RuntimeError("xllm_chunk_kda_fwd_prepare: backward_mode must be none/forward/recompute/save.")
    selected = set(mode_slots[backward_mode])

    def make_outputs() -> tuple[Optional[torch.Tensor], ...]:
        produced = {
            name: (_empty(shape, q, dtype=dtype) if name in selected else None)
            for name, (shape, dtype) in expected_outputs.items()
        }
        return (
            produced["gk"],
            produced["Aqk"],
            produced["Akk"],
            produced["w"],
            produced["u"],
            produced["qg"],
            produced["kg"],
            produced["qg_scaled"],
            produced["q_hat"],
            produced["k_hat"],
            produced["q_rstd"],
            produced["k_rstd"],
            produced["beta_eff"],
        )

    layout_buffer = ctypes.create_string_buffer(layout.encode("utf-8"))

    def build_args(ctx: Any, static_inputs: tuple[Any, ...], outputs: tuple[Any, ...]) -> list[Any]:
        q_s, k_s, v_s, g_s, beta_s, a_log_s, dt_bias_s = static_inputs

        def slot(tensor: Optional[torch.Tensor], name: str) -> ctypes.c_void_p:
            # The storage shape must carry the LOGICAL dims (the prepare
            # tiling reads them); a plain descriptor would flatten to numel.
            # None slots pass a null descriptor.
            if tensor is None:
                return ctx.tensor(None, name)
            return ctx.tensor(tensor, name, storage_shape_override=_shape(tensor))

        return [
            slot(q_s, "q"),
            slot(k_s, "k"),
            slot(v_s, "v"),
            slot(g_s, "g"),
            slot(beta_s, "beta"),
            slot(a_log_s, "A_log"),
            slot(dt_bias_s, "dt_bias"),
            ctx.int_array(cu),
            ctx.int_array(indices),
            ctypes.cast(layout_buffer, ctypes.c_char_p),
            ctypes.c_double(float(scale)),
            ctypes.c_int64(chunk_size),
            ctypes.c_double(float(epsilon)),
            ctypes.c_bool(use_qk_l2norm_in_kernel),
            ctypes.c_bool(use_gate_in_kernel),
            ctypes.c_bool(use_beta_sigmoid_in_kernel),
            ctypes.c_bool(allow_neg_eigval),
            ctypes.c_bool(safe_gate),
            ctypes.c_double(lower_bound),
            ctypes.c_bool(use_exp2),
            slot(outputs[0], "gk"),
            slot(outputs[1], "Aqk"),
            slot(outputs[2], "Akk"),
            slot(outputs[3], "w"),
            slot(outputs[4], "u"),
            slot(outputs[5], "qg"),
            slot(outputs[6], "kg"),
            slot(outputs[7], "qg_scaled"),
            slot(outputs[8], "q_hat"),
            slot(outputs[9], "k_hat"),
            slot(outputs[10], "q_rstd"),
            slot(outputs[11], "k_rstd"),
            slot(outputs[12], "beta_eff"),
        ]

    key = (
        _PREPARE_SYMBOL,
        _tensor_signature(q),
        _tensor_signature(k),
        _tensor_signature(v),
        _tensor_signature(g),
        _tensor_signature(beta),
        _tensor_signature(A_log),
        _tensor_signature(dt_bias),
        cu,
        layout,
        int(chunk_size),
        str(backward_mode),
        float(scale),
        float(epsilon),
        float(lower_bound),
        bool(use_qk_l2norm_in_kernel),
        bool(use_gate_in_kernel),
        bool(use_beta_sigmoid_in_kernel),
        bool(allow_neg_eigval),
        bool(safe_gate),
        bool(use_exp2),
    )
    return _launch_op(_PREPARE_SYMBOL, (q, k, v, g, beta, A_log, dt_bias), make_outputs, build_args, key)


# ---------------------------------------------------------------------------
# Op: aclnnPreProcessFwdKernelMerged (the per-span affine summary fold)
# ---------------------------------------------------------------------------


def xllm_pre_process_fwd_kernel_merged(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    *,
    gk: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[Sequence[int]] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Run aclnnPreProcessFwdKernelMerged: fold factors into hm = [E | M].

    Thin port of the fla-nu ``npu_pre_process_fwd_kernel_merged`` wrapper.
    Takes whole tensors plus a sub-interval ``cu_seqlens`` (``cu[0] > 0``
    and ``cu[-1] < T`` are legal; then Nseq = 1) or several segments at
    once (one affine chain per segment). Returns ``hm[Nseq, HV, K, V+K]``
    fp32: the left ``[0, V)`` block is E and the right ``[V, V+K)`` block
    is M. ``k`` is on the HK heads and the other factors on the HV heads
    (GVA: ``HV % HK == 0``); the DPLR-only ``bg``/``v`` ABI slots are not
    exposed by this port and always pass null descriptors.
    """
    cu = None if cu_seqlens is None else [int(value) for value in cu_seqlens]
    if cu is None or len(cu) < 2:
        raise ValueError(
            "xllm_pre_process_fwd_kernel_merged requires cu_seqlens (varlen only): "
            "pass a host int list such as [bos, eos] or [0, s1, s2, ..., T]."
        )
    if (g is None) == (gk is None):
        raise ValueError("exactly one of g / gk must be provided.")
    chunk_size = _optional_int(chunk_size, 64)

    batch, head_key, seqlen, key_dim = _shape(k)
    if batch != 1:
        raise ValueError(
            "xllm_pre_process_fwd_kernel_merged requires B = 1 (varlen packed), "
            f"got B = {batch}. Pack a dense batch into the token axis first."
        )
    head_value, value_dim = _shape(u)[1], _shape(u)[-1]
    # KDA's gk is given per value head as [B, T, HV, K] (GVA); k stays on the
    # HK heads mapped through the HV // HK grouping, so HK < HV is legal.
    if cu[0] < 0 or cu[-1] > seqlen or any(b <= a for a, b in zip(cu, cu[1:])):
        raise ValueError(
            "xllm_pre_process_fwd_kernel_merged: cu_seqlens must be strictly "
            f"increasing with 0 <= cu[0] < cu[-1] <= T = {seqlen}, got {cu}. "
            "(sub-interval windows are allowed: cu[0] > 0 or cu[-1] < T)"
        )
    nseq = len(cu) - 1

    # hm is pre-allocated on the python side (both the E and M blocks are
    # fully written by the kernel).
    def make_outputs() -> torch.Tensor:
        return _zeros((nseq, head_value, key_dim, value_dim + key_dim), k, dtype=torch.float32)

    def build_args(ctx: Any, static_inputs: tuple[Any, ...], outputs: tuple[Any, ...]) -> list[Any]:
        k_s, w_s, u_s, g_s, gk_s = static_inputs
        (hm_s,) = outputs
        return [
            ctx.tensor(k_s, "k"),
            ctx.tensor(w_s, "w"),
            ctx.tensor(u_s, "u"),
            ctx.tensor(g_s, "g"),
            ctx.tensor(gk_s, "gk"),
            ctx.tensor(None, "bg"),  # DPLR-only ABI slot: always null here.
            ctx.tensor(None, "v"),  # DPLR-only ABI slot: always null here.
            ctx.int_array(cu),
            ctypes.c_int64(chunk_size),
            ctx.tensor(hm_s, "hm"),
        ]

    key = (
        _PPFM_SYMBOL,
        _tensor_signature(k),
        _tensor_signature(w),
        _tensor_signature(u),
        _tensor_signature(g),
        _tensor_signature(gk),
        tuple(cu),
        int(chunk_size),
    )
    return _launch_op(_PPFM_SYMBOL, (k, w, u, g, gk), make_outputs, build_args, key)


# ---------------------------------------------------------------------------
# Op: aclnnMergeFwdBwdKernel (the CP merge fold, one launch per prefix state)
# ---------------------------------------------------------------------------


def xllm_merge_fwd_bwd_kernel(
    ag_hm: torch.Tensor,
    pre_or_post_num_ranks: int,
    rank: int,
    *,
    forward: bool = True,
    state_v_first: bool = True,
) -> torch.Tensor:
    """Run aclnnMergeFwdBwdKernel: compose prior-segment summaries into h.

    Thin port of the fla-nu ``npu_merge_fwd_bwd_kernel`` wrapper with ONE
    deviation: ``h`` is allocated and returned here (the kernel never reads
    the caller's h values -- source-pinned in the module docstring; the
    fla wrapper's in-place ``h`` argument is a pure output buffer).
    ``ag_hm`` is ``[S, HV, K, V+K]`` with K == V == 128 and the last dim
    == 256 (op-hardcoded), dtype fp32/bf16, laid out exactly like the
    ppfm ``hm`` output: E block at ``[..., :V]`` and M block at
    ``[..., V:]``, both in natural orientation. ``forward=True`` composes
    the ``pre_or_post_num_ranks`` rows immediately before ``rank``, oldest
    first, with the chain ``h <- E(rank-N); h <- M_i @ h + E_i`` (natural
    [K, V] orientation internally); ``forward=False`` composes the rows
    after ``rank`` in reverse order (the bwd direction; unused by the CP
    prefill path). ``state_v_first=True`` returns ``h`` as ``[HV, V, K]``
    (the stored orientation the KDA state cache uses); ``False`` returns
    the natural ``[HV, K, V]``.
    """
    if ag_hm is None:
        raise RuntimeError("xllm_merge_fwd_bwd_kernel: ag_hm is required.")
    ag_hm = ag_hm.contiguous()
    if ag_hm.dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError("xllm_merge_fwd_bwd_kernel: ag_hm must be float32 or bfloat16.")
    if len(ag_hm.shape) != 4:
        raise RuntimeError("xllm_merge_fwd_bwd_kernel: ag_hm must be [S, HV, K, V+K].")
    s_count, head_value, k_dim, vk_dim = (int(dim) for dim in ag_hm.shape)
    if k_dim != 128 or vk_dim != 256:
        raise RuntimeError("xllm_merge_fwd_bwd_kernel: K must be 128 and last dim must be 256.")
    if not 0 <= int(rank) < s_count:
        raise RuntimeError(f"xllm_merge_fwd_bwd_kernel: rank must be in [0, S={s_count}).")
    if not (1 <= s_count <= 1024 and 1 <= head_value <= 256):
        raise RuntimeError("xllm_merge_fwd_bwd_kernel: S must be in [1, 1024] and HV in [1, 256].")
    n_ranks = int(pre_or_post_num_ranks)
    if not 1 <= n_ranks <= s_count:
        raise RuntimeError(f"xllm_merge_fwd_bwd_kernel: pre_or_post_num_ranks must be in [1, S={s_count}].")
    if not (s_count == 1 and n_ranks == 1 and int(rank) == 0):
        if forward and int(rank) < n_ranks:
            raise RuntimeError("xllm_merge_fwd_bwd_kernel: forward merge requires rank >= N.")
        if not forward and int(rank) + n_ranks >= s_count:
            raise RuntimeError("xllm_merge_fwd_bwd_kernel: backward merge requires rank + N < S.")

    def make_outputs() -> torch.Tensor:
        # [HV, K, V] / [HV, V, K] are the same shape at K == V == 128; the
        # state_v_first flag only selects the kernel-side store orientation.
        return _empty((head_value, k_dim, vk_dim - k_dim), ag_hm)

    def build_args(ctx: Any, static_inputs: tuple[Any, ...], outputs: tuple[Any, ...]) -> list[Any]:
        (ag_s,) = static_inputs
        (h_s,) = outputs
        return [
            ctx.tensor(h_s, "h", storage_shape_override=_shape(h_s), acl_format_override=_ACL_FORMAT_ND),
            ctx.tensor(ag_s, "ag_hm", storage_shape_override=_shape(ag_s), acl_format_override=_ACL_FORMAT_ND),
            ctypes.c_int64(n_ranks),
            ctypes.c_int64(int(rank)),
            ctypes.c_bool(bool(forward)),
            ctypes.c_bool(bool(state_v_first)),
        ]

    key = (
        _MERGE_SYMBOL,
        _tensor_signature(ag_hm),
        n_ranks,
        int(rank),
        bool(forward),
        bool(state_v_first),
    )
    return _launch_op(_MERGE_SYMBOL, (ag_hm,), make_outputs, build_args, key)
