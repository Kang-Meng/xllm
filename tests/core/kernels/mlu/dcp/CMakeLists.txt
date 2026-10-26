# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
# See https://github.com/xLLM-AI/xllm/blob/main/LICENSE

include(cc_test)

get_filename_component(_xllm_build_import_root
  "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" DIRECTORY)

cc_test(
  NAME
    dcp_kernels_test
  SRCS
    dcp_indexer_test.cpp
    dcp_correct_attn_out_test.cpp
    dcp_merge_topk_test.cpp
    dcp_localize_topk_test.cpp
  DEPS
    :triton_jit
    :mlu_kernels
    torch
    glog::glog
    GTest::gtest_main
  ENVIRONMENT
    "PYTHONPATH=${_xllm_build_import_root}:$ENV{PYTHONPATH}"
)
