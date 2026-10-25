# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(cc_test)

cc_test(
  NAME
    deepseek_v2_attention_multi_device_test
  SRCS
    deepseek_v32_attention_multi_device_test.cpp
    v32_prefill_parallel_test.cpp
    dsa_deep_ep_decoder_test.cpp
  DEPS
    :tests_utils
    :config
    :block
    :mlu_layers
    :model_context
    :parallel_state
    :state_dict
    GTest::gtest_main
    torch
    glog::glog
  ENVIRONMENT
    "PYTHONPATH=${_xllm_build_import_root}:$ENV{PYTHONPATH}"
)

# This target forks MLU ranks and must remain isolated from layer_test, which
# may initialize an MLU device before the candidate collective test runs.
cc_test(
  NAME
    dcp_indexer_candidate_multi_device_test
  SRCS
    dcp_indexer_candidate_multi_device_test.cpp
  DEPS
    :tests_utils
    :mlu_layers
    :parallel_state
    :util
    GTest::gtest_main
    torch
    glog::glog
)

# Cross-layer DSA top-k relay state is pure CPU logic, but the protocol belongs
# to the MLU decoder/attention boundary.
cc_test(
  NAME
    dsa_topk_relay_test
  SRCS
    dsa_topk_relay_test.cpp
  DEPS
    :mlu_layers
    GTest::gtest_main
    torch
    glog::glog
)

# Forks four ranks; keep separate from binaries that initialize MLU in the parent.
cc_test(
  NAME
    glm5_attention_dcp_test
  SRCS
    glm5_attention_dcp_test.cpp
  DEPS
    :tests_utils
    :config
    :block
    :mlu_layers
    :model_context
    :parallel_state
    :state_dict
    :util
    GTest::gtest_main
    torch
    glog::glog
  ENVIRONMENT
    "PYTHONPATH=${_xllm_build_import_root}:$ENV{PYTHONPATH}"
)

# Forks eight ranks before MLU initialization; keep independent of layer_test.
cc_test(
  NAME
    glm53_indexer_dcp_multi_device_test
  SRCS
    glm53_indexer_dcp_multi_device_test.cpp
  DEPS
    :config
    :mlu_layers
    :parallel_state
    :state_dict
    :util
    GTest::gtest_main
    torch
    glog::glog
  ENVIRONMENT
    "PYTHONPATH=${_xllm_build_import_root}:$ENV{PYTHONPATH}"
)

# Each distributed scenario consumes all eight cards, including under ctest -j.
if(TARGET glm53_indexer_dcp_multi_device_test)
  set_tests_properties(
    Glm53IndexerDcpMultiDeviceTest.ChunkedBatch1
    Glm53IndexerDcpMultiDeviceTest.ChunkedBatch16
    Glm53IndexerDcpMultiDeviceTest.PrefillBatch1
    Glm53IndexerDcpMultiDeviceTest.PrefillBatch16
    PROPERTIES RUN_SERIAL TRUE
  )
endif()

# Forks ranks before device initialization; kernel maintenance remains single-card.
cc_test(
  NAME glm53_dcp_merge_topk_multi_device_test
  SRCS glm53_dcp_merge_topk_multi_device_test.cpp
  DEPS :config :mlu_kernels :parallel_state :util torch glog::glog GTest::gtest_main
  ENVIRONMENT "PYTHONPATH=${_xllm_build_import_root}:$ENV{PYTHONPATH}"
)
set_tests_properties(
  Glm53DcpMergeTopkMultiDeviceTest.Bf16CandidatesQ8192
  Glm53DcpMergeTopkMultiDeviceTest.Bf16EdgesAndStrides
  Glm53DcpMergeTopkMultiDeviceTest.PrefillLongHistoryBatch1
  Glm53DcpMergeTopkMultiDeviceTest.PrefillLongHistoryBatch8
  Glm53DcpMergeTopkMultiDeviceTest.DecodeDp2Batch16
  Glm53DcpMergeTopkMultiDeviceTest.DecodeDp2Batch32
  PROPERTIES RUN_SERIAL TRUE
)
