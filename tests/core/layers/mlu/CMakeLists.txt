include(cc_test)
include(cc_library)

cc_library(
  NAME tests_utils
  SRCS tests_utils.cpp
  INCLUDES "${PROJECT_SOURCE_DIR}/tests/core" "${PROJECT_SOURCE_DIR}/xllm"
  DEPS :model :model_context :state_dict :parallel_state :platform :util torch glog::glog GTest::gtest
)

get_filename_component(_xllm_build_import_root
  "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" DIRECTORY)

# Add test for Qwen3.5 MLU layers (gated delta net operators, GDN layer, and
# hybrid attention layer).
cc_test(
  NAME
    qwen3_5_layer_test
  SRCS
    qwen3_5_gated_delta_net_ops_test.cpp
    qwen3_5_gated_delta_net_test.cpp
    qwen3_5_attention_test.cpp
    qwen3_5_fused_moe_test.cpp
    qwen3_next_rms_norm_test.cpp
  DEPS
    :tests_utils
    :config
    :mlu_layers
    :common_layers
    :block
    :parallel_state
    :model
    :model_context
    :state_dict
    glog::glog
    torch
    GTest::gtest_main
  ENVIRONMENT
    "PYTHONPATH=${_xllm_build_import_root}:$ENV{PYTHONPATH}"
)

cc_test(
  NAME
    glm5_next_kda_test
  RUN_SERIAL
  SRCS
    glm5_next_kda_test.cpp
  DEPS
    :config
    :mlu_layers
    :common_layers
    :block
    :parallel_state
    :model
    :model_context
    :state_dict
    glog::glog
    torch
    GTest::gtest_main
  ENVIRONMENT
    "PYTHONPATH=${_xllm_build_import_root}:$ENV{PYTHONPATH};OMP_NUM_THREADS=1;MKL_NUM_THREADS=1"
)

cc_test(
  NAME
    glm5_next_decoder_layer_test
  RUN_SERIAL
  SRCS
    glm5_next_decoder_layer_test.cpp
  DEPS
    :tests_utils
    :config
    :mlu_layers
    :common_layers
    :block
    :parallel_state
    :model
    :model_context
    :state_dict
    glog::glog
    torch
    GTest::gtest_main
  ENVIRONMENT
    "PYTHONPATH=${_xllm_build_import_root}:$ENV{PYTHONPATH};OMP_NUM_THREADS=1;MKL_NUM_THREADS=1"
)

cc_test(
  NAME
    glm5_next_pcp_context_test
  SRCS
    glm5_next_pcp_context_test.cpp
  DEPS
    :mlu_layers
    glog::glog
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    glm5_next_kpool_indexer_test
  RUN_SERIAL
  SRCS
    glm5_next_kpool_indexer_test.cpp
  DEPS
    :tests_utils
    :speculative
    :config
    :mlu_layers
    :common_layers
    :block
    :parallel_state
    :model
    :model_context
    :state_dict
    glog::glog
    torch
    GTest::gtest_main
  ENVIRONMENT
    "PYTHONPATH=${_xllm_build_import_root}:$ENV{PYTHONPATH};OMP_NUM_THREADS=1;MKL_NUM_THREADS=1"
)

# Add test for MLU layers
cc_test(
  NAME
    layer_test
  SRCS
    dense_mlp_test.cpp
    mla_decode_lse_test.cpp
    deepseek_v32/dcp_indexer_candidate_test.cpp
    deepseek_v32/dcp_decode_context_test.cpp
    deepseek_v32/deepseek_v32_cp_utils_test.cpp
    dp_utils_test.cpp
    deepseek_v32/indexer_test.cpp
    deepseek_v32/mla_test.cpp
    deepseek_v32/dsa_characterization_test.cpp
    deepseek_v32/deepseek_v32_decoder_layer_test.cpp
    deepseek_v32/dsa_decoder_math_test.cpp
    qwen2_attention_test.cpp
    qwen2_vision_attention_test.cpp
  DEPS
    :tests_utils
    :config
    :mlu_layers
    :common_layers
    :block
    :parallel_state
    :model
    :model_context
    :state_dict
    glog::glog
    torch
    GTest::gtest_main
  ENVIRONMENT
    "PYTHONPATH=${_xllm_build_import_root}:$ENV{PYTHONPATH}"
)

cc_test(
  NAME
    deepseek_v4_attention_layer_test
  SRCS
    deepseek_v4_attention_test.cpp
    deepseek_v4_compressor_test.cpp
    deepseek_v4_indexer_test.cpp
    deepseek_v4_ref_utils.cpp
  DEPS
    :tests_utils
    :mlu_layers
    :common_layers
    :block
    :util
    :parallel_state
    :model
    :model_context
    :state_dict
    glog::glog
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    hyper_connection_test
  SRCS
    hyper_connection_test.cpp
  DEPS
    :tests_utils
    :mlu_layers
    :common_layers
    :block
    :parallel_state
    :model
    :model_context
    :state_dict
    glog::glog
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    deepseek_v4_decoder_layer_test
  SRCS
    deepseek_v4_decoder_layer_test.cpp
    deepseek_v4_ref_utils.cpp
  DEPS
    :tests_utils
    :mlu_layers
    :common_layers
    :block
    :parallel_state
    :model
    :model_context
    :state_dict
    glog::glog
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    deepseek_v4_dsa_metadata_builder_test
  SRCS
    dsa_metadata_builder_mlu_test.cpp
  DEPS
    :mlu_layers
    :common_layers
    glog::glog
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    deepseek_v4_cp_context_test
  SRCS
    deepseek_v4_cp_context_test.cpp
  DEPS
    :mlu_layers
    GTest::gtest_main
    glog::glog
)

# MoE Gate unit test for MLU (uses layers/mlu/moe_gate)
cc_test(
  NAME
    moe_layer_test
  SRCS
    moe_gate_test.cpp
    fused_moe_test.cpp
    moe_weight_loader_helper_test.cpp
    deepseek_v32/deepseek_v32_sparse_moe_block_test.cpp
    sparse_moe_block_test.cpp
  DEPS
    :tests_utils
    :config
    :mlu_layers
    :common_layers
    :block
    :parallel_state
    :model
    :model_context
    :state_dict
    glog::glog
    torch
    GTest::gtest_main
)

# Isolate forked CNCL ranks from tests that initialize MLU in the parent.
cc_test(
  NAME
    sparse_moe_collective_test
  SRCS
    sparse_moe_collective_test.cpp
  DEPS
    :config
    :mlu_layers
    :common_layers
    :parallel_state
    :state_dict
    :util
    GTest::gtest_main
    torch
    glog::glog
)

# Add test for DeepEP
# This test must exist individually, because it contains forked processes
#  which does not allow any device init on main process
cc_test(
  NAME
    deep_ep_test
  SRCS
    deep_ep_test.cpp
  DEPS
    :tests_utils
    :mlu_layers
    :common_layers
    :parallel_state
    :model
    :model_context
    :state_dict
    GTest::gtest_main
    torch
    glog::glog
)

# Add test for FusedMoE All2All path
# This test must exist individually, because it contains forked processes
#  which does not allow any device init on main process
cc_test(
  NAME
    fused_moe_all2all_test
  SRCS
    fused_moe_all2all_test.cpp
  DEPS
    :tests_utils
    :config
    :mlu_layers
    :common_layers
    :block
    :parallel_state
    :model
    :model_context
    :state_dict
    GTest::gtest_main
    torch
    glog::glog
)

cc_test(
  NAME
    compressed_tensors_mlu_test
  SRCS
    compressed_tensors_test.cpp
  DEPS
    :tests_utils
    :config
    :mlu_layers
    :common_layers
    :model_context
    :parallel_state
    :state_dict
    glog::glog
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    dflash2_test
  SRCS
    dflash2_test.cpp
  DEPS
    :mlu_layers
    :common_layers
    :model_context
    :state_dict
    :kv_cache
    glog::glog
    torch
    GTest::gtest_main
)

# Forks real CP ranks before MLU initialization in the parent process.
cc_test(
  NAME
    glm5_next_state_pcp_test
  SRCS
    glm5_next_state_pcp_test.cpp
  DEPS
    :mlu_layers
    :parallel_state
    :kv_cache
    :util
    glog::glog
    torch
    GTest::gtest_main
  ENVIRONMENT
    "PYTHONPATH=${_xllm_build_import_root}:$ENV{PYTHONPATH}"
)

add_subdirectory(deepseek_v32)
