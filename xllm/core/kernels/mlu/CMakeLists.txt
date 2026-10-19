include(cc_library)

add_subdirectory(bangc)

file(GLOB_RECURSE MLU_HEADER_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.h"
)

file(GLOB_RECURSE MLU_SOURCE_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
)

# BangC Tensor wrappers and registrations belong exclusively to xllm_mlu_ops.
list(FILTER MLU_HEADER_FILES EXCLUDE REGEX "/bangc/")
list(FILTER MLU_SOURCE_FILES EXCLUDE REGEX "/bangc/")

set(MLU_KERNEL_DEPS
  torch
  :platform
  :triton_jit
  cnclep
  torch_mlu
  -Wl,--no-as-needed
  torch_mlu_ops
  xllm_mlu_ops
  ${MLU_CNRT_LIBRARY}
  -Wl,--as-needed
  Python::Python
)

cc_library(
  NAME
    mlu_kernels
  HDRS
    ${MLU_HEADER_FILES}
  SRCS
    ${MLU_SOURCE_FILES}
  DEPS
    ${MLU_KERNEL_DEPS}
)
