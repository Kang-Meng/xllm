/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/platform/npu/npu_python_runtime.h"

#include <acl/acl.h>
#include <glog/logging.h>
#include <pybind11/embed.h>

#include <mutex>

namespace py = pybind11;

namespace xllm::npu {
namespace {

// Registers fla_npu's embedded AscendC opapi and its OPP vendor dir before
// aclInit, which caches the OPP scan: the KDA forward op and its kernels live
// in that OPP, and without this registration the opapi cannot find the kernel
// binary and the aclnn call fails with 561103. Builds without fla_npu stay
// unaffected.
constexpr char kRegisterFlaNpuOpapiScript[] = R"(
try:
    import fla_npu
    fla_npu.load_ascendc_opapi_libraries()
except (ImportError, RuntimeError) as error:
    # Absent or broken fla_npu is a supported configuration (the KDA forward
    # op degrades with it), but say so: a later 561103 on an aclnn call would
    # otherwise point nowhere near the root cause. Same wording as
    # xllm/python/_npu_bootstrap.py's secondary guard.
    try:
        from scripts.logger import logger

        logger.warning(f"fla_npu opapi registration skipped: {error}")
    except ImportError:
        print(f"fla_npu opapi registration skipped: {error}")
)";

// xLLM owns this process's interpreter, so the runtime is established from
// scratch. TORCH_DEVICE_BACKEND_AUTOLOAD must be off before torch is imported:
// torch decides at import time whether to autoload device backends, and the
// accelerator accessor is pinned to CPU while torch_npu is imported, so that
// torch does not resolve the device backend during that import either.
constexpr char kInitializeTorchNpuScript[] = R"(
import os

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import torch

original_accelerator = torch._C._get_accelerator
try:
    torch._C._get_accelerator = lambda: torch.device('cpu')
    import torch_npu
finally:
    torch._C._get_accelerator = original_accelerator

import torch_npu.npu as npu_module
try:
    torch_npu._C._npu_init()
except RuntimeError as error:
    if 'already initialized' not in str(error).lower():
        raise
npu_module._initialized = True
npu_module._original_pid = os.getpid()
torch_npu._C._npu_setDevice(device_index)
)";

}  // namespace

void initialize_python_runtime(int32_t device_index) {
  CHECK_GE(device_index, 0) << "device index must not be negative";
  // The runtime is process-wide: aclInit and the torch_npu bootstrap run once
  // and pin one accelerator, so a second call with a different index would
  // silently no-op. The serving binary and the spawn worker each call this
  // exactly once with their own topology-derived index; treat a mismatched
  // repeat as the caller bug it is.
  static std::once_flag flag;
  static int32_t initialized_device_index = -1;
  std::call_once(flag, [device_index]() {
    initialized_device_index = device_index;
    CHECK(Py_IsInitialized() == 0)
        << "initialize_python_runtime() creates the embedded interpreter; a "
           "process that already owns one configures the accelerator from "
           "Python (xllm/pybind/utils.py) instead";
    py::initialize_interpreter(/*init_signal_handlers=*/false);
    // The interpreter lives for the whole process; release the GIL so worker
    // threads can acquire it on demand.
    PyEval_SaveThread();

    {
      py::gil_scoped_acquire gil;
      py::exec(kRegisterFlaNpuOpapiScript);
    }

    const auto acl_ret = aclInit(nullptr);
    CHECK(acl_ret == ACL_SUCCESS || acl_ret == ACL_ERROR_INTERNAL_ERROR)
        << "aclInit failed with error " << acl_ret;

    py::gil_scoped_acquire gil;
    py::dict scope;
    scope["device_index"] = device_index;
    py::exec(kInitializeTorchNpuScript, scope);
  });
  CHECK_EQ(initialized_device_index, device_index)
      << "initialize_python_runtime() was already called with device index "
      << initialized_device_index << "; it configures exactly one accelerator "
      << "per process and cannot be re-targeted to " << device_index;
}

}  // namespace xllm::npu
