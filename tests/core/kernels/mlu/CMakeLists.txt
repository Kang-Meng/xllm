include(cc_test)

add_subdirectory(dcp)

get_filename_component(_xllm_build_import_root
  "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" DIRECTORY)

cc_test(
  NAME
    causal_conv1d_prefill_state_test
  SRCS
    causal_conv1d_prefill_state_test.cpp
  DEPS
    :mlu_kernels
    :platform
    glog::glog
    torch
    GTest::gtest_main
  ENVIRONMENT
    "PYTHONPATH=${_xllm_build_import_root}:$ENV{PYTHONPATH}"
)

cc_test(
  NAME
    pack_cache_blocks_test
  SRCS
    pack_cache_blocks_test.cpp
  DEPS
    :mlu_kernels
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    fused_moe_clamp_mlu_test
  SRCS
    fused_moe_clamp_test.cpp
  DEPS
    :mlu_kernels
    torch
    glog::glog
    GTest::gtest_main
)

cc_test(
  NAME
    scaled_quantize_mlu_test
  SRCS
    scaled_quantize_test.cpp
  DEPS
    :mlu_kernels
    torch
    glog::glog
    GTest::gtest_main
)

cc_test(
  NAME
    triton_jit_ops_test
  RUN_SERIAL
  SRCS
    fused_gdn_gating_test.cpp
    causal_conv1d_update_decode_test.cpp
    fused_recurrent_gated_delta_rule_packed_decode_test.cpp
    causal_conv1d_fn_test.cpp
    fused_recurrent_gated_delta_rule_test.cpp
    chunk_gated_delta_rule_regression_test.cpp
    chunk_kda_test.cpp
    chunk_kda_pcp_test.cpp
    fused_recurrent_gated_delta_rule_beta_regression_test.cpp
    fused_sigmoid_update_regression_test.cpp
  DEPS
    :triton_jit
    :mlu_kernels
    torch
    glog::glog
    GTest::gtest_main
  ENVIRONMENT
    "PYTHONPATH=${_xllm_build_import_root}:$ENV{PYTHONPATH};OMP_NUM_THREADS=1;MKL_NUM_THREADS=1"
)

add_subdirectory(kpool)

# The BangC library is built from the imported MLU sources, with no gen/ ELF.
cc_test(
  NAME
    qwen4_ple_bangc_test
  SRCS
    qwen4_ple_bangc_test.cpp
  DEPS
    xllm_mlu_ops
    torch
    glog::glog
    GTest::gtest_main
)
