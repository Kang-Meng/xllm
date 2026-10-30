include(cc_test)

cc_test(
  NAME
    provider_protocol_test
  SRCS
    provider_protocol_test.cpp
  DEPS
    proto::xllm_proto
    GTest::gtest_main
)

cc_test(
  NAME
    request_event_protocol_test
  SRCS
    request_event_protocol_test.cpp
  DEPS
    :request
    proto::xllm_proto
    GTest::gtest_main
)
