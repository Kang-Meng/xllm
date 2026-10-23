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
