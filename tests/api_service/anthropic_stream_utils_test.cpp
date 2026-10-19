/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "api_service/anthropic_stream_utils.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace xllm {
namespace {

TEST(AnthropicUsageTest, SplitsPromptTokensIntoUncachedAndCachedUsage) {
  for (const int32_t cached_tokens : {int32_t{0}, int32_t{6}, int32_t{10}}) {
    const proto::AnthropicUsage usage = api_service::make_anthropic_usage(
        /*prompt_tokens=*/10, /*generated_tokens=*/3, cached_tokens);
    EXPECT_EQ(usage.input_tokens(), 10 - cached_tokens);
    EXPECT_EQ(usage.output_tokens(), 3);
    ASSERT_TRUE(usage.has_cache_read_input_tokens());
    EXPECT_EQ(usage.cache_read_input_tokens(), cached_tokens);
  }
}

TEST(AnthropicUsageTest, ExcessiveCacheHitsDoNotProduceNegativeInputTokens) {
  const proto::AnthropicUsage usage = api_service::make_anthropic_usage(
      /*prompt_tokens=*/10, /*generated_tokens=*/3, /*cached_tokens=*/11);
  EXPECT_EQ(usage.input_tokens(), 0);
  EXPECT_EQ(usage.output_tokens(), 3);
  EXPECT_EQ(usage.cache_read_input_tokens(), 10);
}

TEST(AnthropicUsageTest, InvalidCountsAreClampedBeforeArithmetic) {
  const proto::AnthropicUsage negative = api_service::make_anthropic_usage(
      /*prompt_tokens=*/-1, /*generated_tokens=*/-2, /*cached_tokens=*/-3);
  EXPECT_EQ(negative.input_tokens(), 0);
  EXPECT_EQ(negative.output_tokens(), 0);
  EXPECT_EQ(negative.cache_read_input_tokens(), 0);

  const int32_t max_tokens = std::numeric_limits<int32_t>::max();
  const proto::AnthropicUsage boundary = api_service::make_anthropic_usage(
      max_tokens, /*generated_tokens=*/0, std::numeric_limits<int32_t>::min());
  EXPECT_EQ(boundary.input_tokens(), max_tokens);
  EXPECT_EQ(boundary.cache_read_input_tokens(), 0);
}

TEST(AnthropicStreamUtilsTest, MapsToolFinishReasonsToToolUse) {
  EXPECT_EQ(api_service::convert_finish_reason_to_anthropic("tool_calls"),
            "tool_use");
  EXPECT_EQ(api_service::convert_finish_reason_to_anthropic("function_call"),
            "tool_use");
}

TEST(AnthropicStreamUtilsTest, MapsTextFinishReasons) {
  EXPECT_EQ(api_service::convert_finish_reason_to_anthropic("stop"),
            "end_turn");
  EXPECT_EQ(api_service::convert_finish_reason_to_anthropic("length"),
            "max_tokens");
  EXPECT_EQ(api_service::convert_finish_reason_to_anthropic("unknown"),
            "end_turn");
}

TEST(AnthropicStreamUtilsTest, ToolCallOverridesStreamStopReason) {
  EXPECT_EQ(api_service::get_stream_stop_reason(true, true, "stop"),
            "tool_use");
  EXPECT_EQ(api_service::get_stream_stop_reason(true, true, "length"),
            "tool_use");
  EXPECT_EQ(api_service::get_stream_stop_reason(true, true, ""), "tool_use");
}

TEST(AnthropicStreamUtilsTest, TextStreamKeepsMappedStopReason) {
  EXPECT_EQ(api_service::get_stream_stop_reason(true, false, "stop"),
            "end_turn");
  EXPECT_EQ(api_service::get_stream_stop_reason(true, false, "length"),
            "max_tokens");
}

TEST(AnthropicStreamUtilsTest, CancelledStreamUsesStop) {
  EXPECT_EQ(api_service::get_stream_stop_reason(false, true, "length"), "stop");
  EXPECT_EQ(api_service::get_stream_stop_reason(false, false, "stop"), "stop");
}

TEST(AnthropicStreamUtilsTest, EmptyToolArgsDoNotCreateInputDelta) {
  std::optional<proto::AnthropicStreamEvent> event =
      api_service::make_input_json_delta_event(2, "");

  EXPECT_FALSE(event.has_value());
}

TEST(AnthropicStreamUtilsTest, NonEmptyToolArgsCreateInputDelta) {
  std::optional<proto::AnthropicStreamEvent> event =
      api_service::make_input_json_delta_event(2, "{\"city\":\"Paris\"}");

  ASSERT_TRUE(event.has_value());
  EXPECT_EQ(event->type(), "content_block_delta");
  ASSERT_TRUE(event->has_index());
  EXPECT_EQ(event->index(), 2);
  ASSERT_TRUE(event->has_delta());
  EXPECT_EQ(event->delta().type(), "input_json_delta");
  EXPECT_EQ(event->delta().partial_json(), "{\"city\":\"Paris\"}");
}

}  // namespace
}  // namespace xllm
