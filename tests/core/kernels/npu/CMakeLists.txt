include(cc_test)

cc_test(
  NAME
    npu_xllm_ops_test
  SRCS
    npu_xllm_ops_test.cpp
  DEPS
    ascendcl
    torch
    torch_npu
    kernels
    npu_kernels
    GTest::gtest_main
    glog::glog
    unified_dlog
    pybind11::embed
)

cc_test(
  NAME
    glm5_next_moe_test
  SRCS
    glm5_next_moe_test.cpp
  DEPS
    ascendcl
    torch
    torch_npu
    kernels
    npu_kernels
    GTest::gtest
    glog::glog
    unified_dlog
    pybind11::embed
)

# Keep real runtime graph tests isolated from tests/python's CPU package stubs.
cc_test(
  NAME
    qwen3_5_target_graph_test
  SRCS
    qwen3_5_target_graph_test.cpp
  DEPS
    ascendcl
    torch
    torch_npu
    kernels
    npu_kernels
    GTest::gtest
    glog::glog
    unified_dlog
    pybind11::embed
)

# Temporarily disabled: Dsv4ScatterCache tests are unstable in the current
# NPU test environment.
# cc_test(
#   NAME
#     dsv4_scatter_cache_write_test
#   SRCS
#     dsv4_scatter_cache_write_test.cpp
#   DEPS
#     ascendcl
#     torch
#     torch_npu
#     xllm_ops
#     GTest::gtest_main
#     glog::glog
# )

cc_test(
  NAME
    atb_sparse_flash_attention_test
  SRCS
    atb_sparse_flash_attention_test.cpp
  DEPS
    ascendcl
    xllm_atb_layers
    nnopbase
    c_sec
    opapi
    torch
    torch_npu
    GTest::gtest_main
    glog::glog
)
# The atb adapter headers and libatb runtime are private deps of
# xllm_atb_layers, so the test wires them explicitly.
target_include_directories(atb_sparse_flash_attention_test PRIVATE
  $ENV{ATB_HOME_PATH}/include)
target_link_libraries(atb_sparse_flash_attention_test PRIVATE
  $ENV{ATB_HOME_PATH}/lib/libatb.so)
if(EXISTS "$ENV{PYTORCH_INSTALL_PATH}/../torch.libs")
  target_link_options(atb_sparse_flash_attention_test PRIVATE
    "-Wl,-rpath-link,$ENV{PYTORCH_INSTALL_PATH}/../torch.libs")
endif()

cc_test(
  NAME
    mtp_prepare_next_draft_test
  SRCS
    mtp_prepare_next_draft_test.cpp
  DEPS
    :xllm_ops
    ascendcl
    nnopbase
    torch
    torch_npu
    GTest::gtest_main
    glog::glog
)
if(EXISTS "$ENV{PYTORCH_INSTALL_PATH}/../torch.libs")
  target_link_options(mtp_prepare_next_draft_test PRIVATE
    "-Wl,-rpath-link,$ENV{PYTORCH_INSTALL_PATH}/../torch.libs")
endif()

# build_cp_context is CompositeExplicitAutograd pure host index math, so this
# runs on CPU with no NPU; it only needs the xllm_ops registrations linked in.
cc_test(
  NAME
    cp_context_builder_test
  CPU_ONLY
  SRCS
    cp_context_builder_test.cpp
  DEPS
    :xllm_ops
    ascendcl
    torch
    torch_npu
    kernels
    npu_kernels
    GTest::gtest_main
    glog::glog
)
if(EXISTS "$ENV{PYTORCH_INSTALL_PATH}/../torch.libs")
  target_link_options(cp_context_builder_test PRIVATE
    "-Wl,-rpath-link,$ENV{PYTORCH_INSTALL_PATH}/../torch.libs")
endif()

add_subdirectory(tilelang)
