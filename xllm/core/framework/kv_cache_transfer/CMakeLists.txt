include(cc_library)
include(cc_test)

cc_library(
  NAME
    push_route
  HDRS
    push_route.h
  SRCS
    push_route.cpp
)

cc_library(
  NAME
    pd_topology_guard
  HDRS
    pd_topology_guard.h
  SRCS
    pd_topology_guard.cpp
  DEPS
    :common
    glog::glog
)

cc_library(
  NAME
    kv_cache_transfer
  HDRS
    kv_cache_transfer.h
    $<$<BOOL:${USE_NPU}>:llm_data_dist_transfer.h>
    $<$<BOOL:${USE_NPU}>:spec_kv_cache_transfer.h>
    $<$<BOOL:${ENABLE_MOONCAKE}>:kv_cache_store.h>
    $<$<BOOL:${ENABLE_MOONCAKE}>:hierarchy_kv_cache_transfer.h>
    $<$<AND:$<BOOL:${ENABLE_MOONCAKE}>,$<OR:$<BOOL:${USE_NPU}>,$<BOOL:${USE_MLU}>>>:mooncake_transfer_engine.h>
    $<$<AND:$<BOOL:${ENABLE_MOONCAKE}>,$<OR:$<BOOL:${USE_NPU}>,$<BOOL:${USE_MLU}>>>:mooncake_kv_cache_transfer.h>
    $<$<AND:$<BOOL:${ENABLE_MOONCAKE}>,$<BOOL:${USE_NPU}>>:mooncake_weight_transfer.h>
  SRCS
    kv_cache_transfer.cpp
    $<$<BOOL:${USE_NPU}>:llm_data_dist_transfer.cpp>
    $<$<BOOL:${USE_NPU}>:spec_kv_cache_transfer.cpp>
    $<$<BOOL:${ENABLE_MOONCAKE}>:kv_cache_store.cpp>
    $<$<BOOL:${ENABLE_MOONCAKE}>:hierarchy_kv_cache_transfer.cpp>
    $<$<AND:$<BOOL:${ENABLE_MOONCAKE}>,$<OR:$<BOOL:${USE_NPU}>,$<BOOL:${USE_MLU}>>>:mooncake_transfer_engine.cpp>
    $<$<AND:$<BOOL:${ENABLE_MOONCAKE}>,$<OR:$<BOOL:${USE_NPU}>,$<BOOL:${USE_MLU}>>>:mooncake_kv_cache_transfer.cpp>
    $<$<AND:$<BOOL:${ENABLE_MOONCAKE}>,$<BOOL:${USE_NPU}>>:mooncake_weight_transfer.cpp>
  DEPS
    :common
    :kv_cache
    :push_route
    :xtensor
    $<$<BOOL:${USE_NPU}>:graph>
    glog::glog
    $<$<BOOL:${USE_NPU}>:llm_datadist>
    $<$<BOOL:${USE_NPU}>:c_sec>
    torch
    $<$<BOOL:${USE_NPU}>:torch_npu>
    $<$<BOOL:${ENABLE_MOONCAKE}>:mooncake_store>
    proto::xllm_proto
    $<$<BOOL:${USE_NPU}>:platform_npu>
)

cc_test(
  NAME
    push_route_test
  SRCS
    push_route_test.cpp
  DEPS
    :push_route
    GTest::gtest_main
)

cc_test(
  NAME
    pd_topology_guard_test
  SRCS
    pd_topology_guard_test.cpp
  DEPS
    :pd_topology_guard
    GTest::gtest_main
)

if(ENABLE_MOONCAKE AND (USE_NPU OR USE_MLU))
cc_test(
  NAME
    mooncake_transfer_engine_test
  SRCS
    mooncake_transfer_engine_test.cpp
  DEPS
    :kv_cache_transfer
    :xllm_server
    GTest::gtest_main
)

# Resolve static link order between xtensor and xllm_server for this test target.
target_link_libraries(mooncake_transfer_engine_test PRIVATE
  "$<LINK_GROUP:RESCAN,xtensor,xllm_server>")
endif()
