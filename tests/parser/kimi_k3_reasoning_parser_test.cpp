/* Copyright 2026 The xLLM Authors.

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

#include <gtest/gtest.h>

#include <string>

#include "parser/reasoning_parser.h"

namespace xllm {
namespace {

TEST(KimiK3ReasoningParserTest, AutoRegistryRecognizesKimiK3) {
  EXPECT_EQ(ReasoningParser::get_parser_auto("auto", "kimi_k3"), "kimi_k3");
  const auto& registry = DetectorRegistry::get_instance();
  EXPECT_TRUE(registry.prompt_opens_think("kimi_k3"));
  EXPECT_FALSE(registry.prompt_opens_think(""));
  EXPECT_FALSE(registry.prompt_opens_think("unknown"));
  EXPECT_FALSE(registry.prompt_opens_think("qwen3"));
}

TEST(KimiK3ReasoningParserTest, ParsesConfiguredThinkDelimiters) {
  for (bool force_reasoning : {false, true}) {
    ReasoningParser parser("kimi_k3", false, force_reasoning);
    const std::string text =
        (force_reasoning ? "" : "<|open|>think<|sep|>") +
        std::string("  plan<|close|>think<|sep|> answer\n");
    const auto result = parser.parse_non_stream(text);
    EXPECT_EQ(result.reasoning_text, "plan");
    EXPECT_EQ(result.normal_text, "answer");
  }
}

TEST(KimiK3ReasoningParserTest, StreamsReasoningUsingMainlineBehavior) {
  for (bool stream_reasoning : {false, true}) {
    ReasoningParser parser("kimi_k3", stream_reasoning, true);
    const auto first = parser.parse_stream_chunk("plan");
    EXPECT_EQ(first.reasoning_text.value_or(""),
              stream_reasoning ? "plan" : "");
    const auto closed =
        parser.parse_stream_chunk("<|close|>think<|sep|> answer\n");
    EXPECT_EQ(closed.reasoning_text.value_or(""),
              stream_reasoning ? "" : "plan");
    EXPECT_EQ(closed.normal_text, "answer");
  }
}

TEST(KimiK3ReasoningParserTest, LeavesResponseEnvelopeForToolParser) {
  const std::string tail =
      "<|open|>response<|sep|>answer<|close|>response<|sep|>"
      "<|open|>tools<|sep|>tool payload<|close|>tools<|sep|>"
      "<|close|>message<|sep|>";
  const std::string text = "plan<|close|>think<|sep|>" + tail;
  ReasoningParser non_stream("kimi_k3", false, true);
  const auto parsed = non_stream.parse_non_stream(text);
  EXPECT_EQ(parsed.normal_text, tail);
  EXPECT_EQ(parsed.reasoning_text, "plan");
  ReasoningParser streaming("kimi_k3", true, true);
  const auto delta = streaming.parse_stream_chunk(text);
  EXPECT_EQ(delta.normal_text, tail);
  EXPECT_EQ(delta.reasoning_text, "plan");
}

}  // namespace
}  // namespace xllm
