# Copyright 2026 The xLLM Authors. All Rights Reserved.

include(cc_test)

cc_test(
  NAME
    mm_data_test
  CPU_ONLY
  SRCS
    mm_data_test.cpp
  DEPS
    :multimodal
    GTest::gtest_main
)
