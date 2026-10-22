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

#include "function_call/kimi_k3_detector.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <string_view>

#include "function_call/function_call_parser.h"

namespace xllm {
namespace function_call {
namespace {

JsonTool make_tool(const std::string& name) {
  return JsonTool(
      "function",
      JsonFunction(name, "test tool", nlohmann::json{{"type", "object"}}));
}

constexpr const char* kResponseEnd = "<|close|>response<|sep|>";
constexpr const char* kToolsStart = "<|open|>tools<|sep|>";
constexpr const char* kToolsEnd = "<|close|>tools<|sep|>";
constexpr const char* kMessageEnd = "<|close|>message<|sep|>";
constexpr const char* kEndOfMessage = "<|end_of_msg|>";

TEST(KimiK3DetectorTest, ParsesJsonAndTypedArguments) {
  KimiK3Detector detector;
  const std::vector<JsonTool> tools = {make_tool("get_weather"),
                                       make_tool("calculate")};
  const std::string text =
      "The answer" + std::string(kResponseEnd) + kToolsStart +
      "<|open|>call tool=\"get_weather\" index=\"1\"<|sep|>"
      "<|open|>json type=\"object\"<|sep|>{\"city\": \"Beijing\"}"
      "<|close|>json<|sep|><|close|>call<|sep|>"
      "<|open|>call tool=\"calculate\" index=\"2\"<|sep|>"
      "<|open|>argument key=\"value\" type=\"number\"<|sep|>3"
      "<|close|>argument<|sep|><|close|>call<|sep|>" +
      std::string(kToolsEnd) + kMessageEnd + kEndOfMessage;

  const auto result = detector.detect_and_parse(text, tools);
  ASSERT_EQ(result.normal_text, "The answer");
  ASSERT_EQ(result.calls.size(), 2);
  EXPECT_EQ(result.calls[0].tool_index, 0);
  EXPECT_EQ(result.calls[0].name, "get_weather");
  EXPECT_EQ(nlohmann::json::parse(result.calls[0].parameters)["city"],
            "Beijing");
  EXPECT_EQ(result.calls[1].tool_index, 1);
  EXPECT_EQ(result.calls[1].name, "calculate");
  EXPECT_EQ(nlohmann::json::parse(result.calls[1].parameters)["value"], 3);
}

TEST(KimiK3DetectorTest, FunctionParserCleansContentWithoutToolCalls) {
  FunctionCallParser parser({make_tool("get_weather")}, "kimi_k3");
  const auto [content, calls] = parser.parse_non_stream(
      "plain response" + std::string(kResponseEnd) + kMessageEnd);

  EXPECT_EQ(content, "plain response");
  EXPECT_TRUE(calls.empty());
}

TEST(KimiK3DetectorTest, EmitsStreamingCallsOnlyAfterMessageEnd) {
  KimiK3Detector detector;
  const std::vector<JsonTool> tools = {make_tool("get_weather")};
  const std::string prefix =
      "hello" + std::string(kResponseEnd) + kToolsStart +
      "<|open|>call tool=\"get_weather\" index=\"1\"<|sep|>"
      "<|open|>json type=\"object\"<|sep|>{}"
      "<|close|>json<|sep|><|close|>call<|sep|>" +
      std::string(kToolsEnd);
  EXPECT_EQ(detector.parse_streaming_increment(prefix, tools).calls.size(), 0);
  const auto result = detector.parse_streaming_increment(
      std::string(kMessageEnd) + kEndOfMessage, tools);
  ASSERT_EQ(result.calls.size(), 1);
  EXPECT_EQ(result.calls[0].name, "get_weather");
  EXPECT_EQ(result.calls[0].parameters, "{}");
}

TEST(KimiK3DetectorTest, RejectsUnknownToolAndNonSequentialIndex) {
  KimiK3Detector detector;
  const std::vector<JsonTool> tools = {make_tool("get_weather")};
  const std::string unknown =
      std::string(kResponseEnd) + kToolsStart +
      "<|open|>call tool=\"unknown\" index=\"1\"<|sep|><|close|>call<|sep|>" +
      std::string(kToolsEnd) + kMessageEnd;
  EXPECT_TRUE(detector.detect_and_parse(unknown, tools).calls.empty());

  const std::string bad_index = std::string(kResponseEnd) + kToolsStart +
                                "<|open|>call tool=\"get_weather\" "
                                "index=\"2\"<|sep|><|close|>call<|sep|>" +
                                std::string(kToolsEnd) + kMessageEnd;
  EXPECT_TRUE(detector.detect_and_parse(bad_index, tools).calls.empty());
}

std::string tool_message(const std::string& arguments) {
  return std::string(kResponseEnd) + kToolsStart +
         "<|open|>call tool=\"get_weather\" index=\"1\"<|sep|>"
         "<|open|>json type=\"object\"<|sep|>" +
         arguments + "<|close|>json<|sep|><|close|>call<|sep|>" + kToolsEnd +
         kMessageEnd;
}

TEST(KimiK3DetectorTest, MessageCloseCompletesWithoutOptionalStopToken) {
  const std::vector<JsonTool> tools = {make_tool("get_weather")};
  const std::string text = "answer" + tool_message("{}");
  // Accept every prefix of the optional stop token, including an empty one.
  for (size_t length = 0; length <= std::string(kEndOfMessage).size();
       ++length) {
    const std::string raw = text + std::string(kEndOfMessage).substr(0, length);
    KimiK3Detector detector;
    const auto result = detector.detect_and_parse(raw, tools);
    EXPECT_EQ(result.normal_text, "answer");
    ASSERT_EQ(result.calls.size(), 1);
    for (size_t split = 0; split <= raw.size(); ++split) {
      KimiK3Detector streaming;
      const auto first =
          streaming.parse_streaming_increment(raw.substr(0, split), tools);
      const auto second =
          streaming.parse_streaming_increment(raw.substr(split), tools);
      EXPECT_EQ(first.normal_text + second.normal_text, "answer");
      EXPECT_EQ(first.calls.size() + second.calls.size(), 1);
      EXPECT_TRUE(streaming.parse_streaming_increment("", tools).calls.empty());
    }
  }
  const std::string before_close = text.substr(0, text.find(kMessageEnd));
  for (size_t length = 0; length < std::string(kMessageEnd).size(); ++length) {
    const std::string raw =
        before_close + std::string(kMessageEnd).substr(0, length);
    KimiK3Detector detector;
    EXPECT_TRUE(detector.detect_and_parse(raw, tools).calls.empty());
    KimiK3Detector streaming;
    EXPECT_TRUE(streaming.parse_streaming_increment(raw, tools).calls.empty());
  }
}

TEST(KimiK3DetectorTest, RechecksFoundCloseWhenToolsWereIncomplete) {
  class ErrorSink final : public google::LogSink {
   public:
    ErrorSink() { google::AddLogSink(this); }
    ~ErrorSink() override { google::RemoveLogSink(this); }
    void send(google::LogSeverity severity,
              const char*,
              const char*,
              int,
              const struct tm*,
              const char* message,
              size_t message_len) override {
      if (severity == google::GLOG_ERROR &&
          std::string_view(message, message_len)
              .starts_with("Invalid streaming Kimi K3 XTML output:")) {
        ++errors;
      }
    }
    size_t errors = 0;
  } errors;

  const std::vector<JsonTool> tools = {make_tool("get_weather")};
  KimiK3Detector detector;
  // A message close inside an unclosed tools block is not a complete envelope.
  // Padding puts the candidate before the next unsuccessful-search offset.
  const std::string first_chunk =
      "answer" + std::string(kResponseEnd) + kToolsStart +
      "<|open|>call tool=\"get_weather\" index=\"1\"<|sep|>"
      "<|close|>call<|sep|>" +
      std::string(kMessageEnd) + std::string(64, ' ');
  const auto first = detector.parse_streaming_increment(first_chunk, tools);
  EXPECT_EQ(first.normal_text, "answer");
  EXPECT_TRUE(first.calls.empty());
  EXPECT_EQ(errors.errors, 0);

  // Completing tools exposes the invalid embedded close. Revalidate the found
  // candidate without requiring another message marker; never publish calls.
  const auto second = detector.parse_streaming_increment(kToolsEnd, tools);
  EXPECT_TRUE(second.normal_text.empty());
  EXPECT_TRUE(second.calls.empty());
  EXPECT_EQ(errors.errors, 1);
  EXPECT_TRUE(
      detector.parse_streaming_increment(kMessageEnd, tools).calls.empty());
}

TEST(KimiK3DetectorTest, PreservesResponseBytesAtEverySplit) {
  const std::string start = "<|open|>response<|sep|>";
  const std::string content = "A<x 北京";
  const std::string raw = start + content + kResponseEnd + kMessageEnd;
  for (size_t split = 0; split <= raw.size(); ++split) {
    KimiK3Detector detector;
    std::string emitted;
    for (const auto& chunk : {raw.substr(0, split), raw.substr(split)}) {
      const auto result = detector.parse_streaming_increment(chunk, {});
      emitted += result.normal_text;
    }
    EXPECT_EQ(emitted, content);
  }
}

TEST(KimiK3DetectorTest, StreamingMatchesAtEveryByteBoundary) {
  const std::vector<JsonTool> tools = {make_tool("get_weather")};
  const std::string text =
      "<|open|>response<|sep|>  answer\n" +
      tool_message(R"({"city":"北京","nested":{"city":1}})") + kEndOfMessage;
  KimiK3Detector reference;
  const auto expected = reference.detect_and_parse(text, tools);
  ASSERT_EQ(expected.calls.size(), 1);
  for (size_t split = 0; split <= text.size(); ++split) {
    KimiK3Detector detector;
    auto first =
        detector.parse_streaming_increment(text.substr(0, split), tools);
    auto second = detector.parse_streaming_increment(text.substr(split), tools);
    EXPECT_EQ(first.normal_text + second.normal_text, expected.normal_text)
        << split;
    first.calls.insert(
        first.calls.end(), second.calls.begin(), second.calls.end());
    ASSERT_EQ(first.calls.size(), 1) << split;
    EXPECT_EQ(first.calls[0].parameters, expected.calls[0].parameters) << split;
  }
  KimiK3Detector detector;
  std::string content;
  size_t calls = 0;
  for (char ch : text) {
    auto delta = detector.parse_streaming_increment(std::string(1, ch), tools);
    content += delta.normal_text;
    calls += delta.calls.size();
  }
  EXPECT_EQ(content, expected.normal_text);
  EXPECT_EQ(calls, 1);
}

TEST(KimiK3DetectorTest, ReturnsTruncatedResponseButNotIncompleteCalls) {
  KimiK3Detector detector;
  EXPECT_EQ(detector.detect_and_parse("  partial answer", {}).normal_text,
            "  partial answer");
  EXPECT_EQ(detector.detect_and_parse("answer<|close|>res", {}).normal_text,
            "answer");
  auto text = tool_message("{}");
  text.resize(text.find(kToolsEnd));
  EXPECT_TRUE(detector.detect_and_parse(text, {make_tool("get_weather")})
                  .calls.empty());
  text = tool_message("{}");
  text.resize(text.find(kMessageEnd));
  EXPECT_TRUE(detector.detect_and_parse(text, {make_tool("get_weather")})
                  .calls.empty());
}

TEST(KimiK3DetectorTest, LiteralResponseCloseIsAStructuralBoundary) {
  // Bodies have no entity codec for XTML markers. Repeating a close marker
  // as text ends the response at its first occurrence, rather than escaping it.
  const std::string text = "before" + std::string(kResponseEnd) +
                           "literal marker after" + kResponseEnd + kMessageEnd;
  KimiK3Detector detector;
  EXPECT_EQ(detector.detect_and_parse(text, {}).normal_text, "before");
  KimiK3Detector streaming;
  std::string content;
  for (const char ch : text) {
    content +=
        streaming.parse_streaming_increment(std::string(1, ch), {}).normal_text;
  }
  EXPECT_EQ(content, "before");
}

TEST(KimiK3DetectorTest, LiteralJsonCloseDoesNotBecomeAnEscapedArgument) {
  KimiK3Detector detector;
  const auto result = detector.detect_and_parse(
      "answer" + tool_message(R"({"city":"<|close|>json<|sep|>"})"),
      {make_tool("get_weather")});
  EXPECT_EQ(result.normal_text, "answer");
  EXPECT_TRUE(result.calls.empty());
}

TEST(KimiK3DetectorTest, MalformedToolsPreserveResponseContent) {
  const std::vector<JsonTool> tools = {make_tool("get_weather")};
  const std::string text = "  useful answer\n" + tool_message("invalid JSON");
  KimiK3Detector detector;
  const auto result = detector.detect_and_parse(text, tools);
  EXPECT_EQ(result.normal_text, "  useful answer\n");
  EXPECT_TRUE(result.calls.empty());
  for (size_t split = 0; split <= text.size(); ++split) {
    KimiK3Detector streaming;
    const auto first =
        streaming.parse_streaming_increment(text.substr(0, split), tools);
    const auto second =
        streaming.parse_streaming_increment(text.substr(split), tools);
    EXPECT_EQ(first.normal_text + second.normal_text, result.normal_text)
        << split;
    EXPECT_TRUE(first.calls.empty());
    EXPECT_TRUE(second.calls.empty());
  }
}

TEST(KimiK3DetectorTest, JsonErrorsStayInsideDetector) {
  const std::vector<JsonTool> tools = {make_tool("get_weather")};
  for (const std::string arguments :
       {R"({"city":)", R"({"city":"\uD800"})", R"({"city":1e1000})"}) {
    SCOPED_TRACE(arguments);
    // Invalid syntax/unicode may return discarded JSON; an overflowing
    // number can throw a library exception even with allow_exceptions=false.
    const std::string text = "answer" + tool_message(arguments);
    KimiK3Detector detector;
    const auto result = detector.detect_and_parse(text, tools);
    EXPECT_EQ(result.normal_text, "answer");
    EXPECT_TRUE(result.calls.empty());
    KimiK3Detector streaming;
    const auto delta = streaming.parse_streaming_increment(text, tools);
    EXPECT_EQ(delta.normal_text, "answer");
    EXPECT_TRUE(delta.calls.empty());
  }
}

TEST(KimiK3DetectorTest, StreamsLongResponsesAndArgumentsWithoutDuplicates) {
  const std::vector<JsonTool> tools = {make_tool("get_weather")};
  const std::string content(65536, 'a');
  const std::string city(65536, 'b');
  const std::string text =
      "<|open|>response<|sep|>" + content +
      tool_message(nlohmann::json({{"city", city}}).dump());
  KimiK3Detector detector;
  std::string output;
  std::vector<ToolCallItem> calls;
  for (size_t offset = 0; offset < text.size(); offset += 17) {
    const auto delta =
        detector.parse_streaming_increment(text.substr(offset, 17), tools);
    output += delta.normal_text;
    calls.insert(calls.end(), delta.calls.begin(), delta.calls.end());
  }
  EXPECT_EQ(output, content);
  ASSERT_EQ(calls.size(), 1);
  EXPECT_EQ(nlohmann::json::parse(calls[0].parameters)["city"], city);
  const auto extra = detector.parse_streaming_increment(kEndOfMessage, tools);
  EXPECT_TRUE(extra.normal_text.empty());
  EXPECT_TRUE(extra.calls.empty());
}

TEST(KimiK3DetectorTest,
     ValidatesNestedDuplicateKeysWithoutRejectingSiblingKeys) {
  KimiK3Detector detector;
  const std::vector<JsonTool> tools = {make_tool("get_weather")};
  EXPECT_EQ(
      detector
          .detect_and_parse(tool_message(R"({"a":[{"k":1},{"k":2}]})"), tools)
          .calls.size(),
      1);
  for (const std::string arguments : {R"({"a":{"k":1,"k":2}})",
                                      R"({"a":[{"k":1,"k":2}]})",
                                      R"({"a":{"k":1},"a":{"k":2}})"}) {
    const std::string text = "answer" + tool_message(arguments);
    const auto result = detector.detect_and_parse(text, tools);
    EXPECT_EQ(result.normal_text, "answer");
    EXPECT_TRUE(result.calls.empty());
    KimiK3Detector streaming;
    const auto streamed = streaming.parse_streaming_increment(text, tools);
    EXPECT_EQ(streamed.normal_text, "answer");
    EXPECT_TRUE(streamed.calls.empty());
  }
}

TEST(KimiK3DetectorTest, DecodesTheInputTemplatesAttributeEntities) {
  KimiK3Detector detector;
  std::string text = tool_message("{}");
  text.replace(text.find("get_weather"), 11, "a&amp;&quot;");
  auto result = detector.detect_and_parse(text, {make_tool("a&\"")});
  ASSERT_EQ(result.calls.size(), 1);
  EXPECT_EQ(result.calls[0].name, "a&\"");
}

TEST(KimiK3DetectorTest, PeerParserLeavesPlainContentUnchanged) {
  for (const auto& format : {"qwen25",
                             "qwen3_coder",
                             "kimi_k2",
                             "deepseekv3",
                             "deepseekv32",
                             "deepseekv4",
                             "deepseekv41",
                             "glm45",
                             "glm47",
                             "glm5"}) {
    FunctionCallParser parser({make_tool("get_weather")}, format);
    const auto [content, calls] = parser.parse_non_stream("  plain content\n");
    EXPECT_EQ(content, "  plain content\n") << format;
    EXPECT_TRUE(calls.empty()) << format;
    EXPECT_FALSE(parser.parses_response_envelope()) << format;
  }
}

TEST(KimiK3DetectorTest, AllowsWhitespaceBetweenBlocksAtEverySplit) {
  const std::vector<JsonTool> tools = {make_tool("get_weather")};
  for (bool with_tools : {false, true}) {
    std::string text = with_tools
                           ? "answer" + tool_message("{}")
                           : "answer" + std::string(kResponseEnd) + kMessageEnd;
    text.insert(text.find(kMessageEnd), " \n\t");
    if (with_tools) {
      text.insert(text.find(kToolsStart), "\n ");
      text.insert(text.find("<|open|>json"), " \n\t");
      text.insert(text.find("<|close|>call"), "\t \n");
    }
    KimiK3Detector detector;
    const auto expected = detector.detect_and_parse(text, tools);
    EXPECT_EQ(expected.normal_text, "answer");
    ASSERT_EQ(expected.calls.size(), with_tools ? 1 : 0);
    for (size_t split = 0; split <= text.size(); ++split) {
      KimiK3Detector streaming;
      const auto first =
          streaming.parse_streaming_increment(text.substr(0, split), tools);
      const auto second =
          streaming.parse_streaming_increment(text.substr(split), tools);
      EXPECT_EQ(first.normal_text + second.normal_text, "answer") << split;
      EXPECT_EQ(first.calls.size() + second.calls.size(), expected.calls.size())
          << split;
    }
  }
}

TEST(KimiK3DetectorTest, RejectsOversizedIndexAndNonCanonicalQuotes) {
  const std::vector<JsonTool> tools = {make_tool("get_weather")};
  for (const std::string& index :
       std::vector<std::string>{std::string(4096, '9'), "01", "+1"}) {
    std::string text = "answer" + tool_message("{}");
    text.replace(text.find("index=\"1\""), 9, "index=\"" + index + "\"");
    KimiK3Detector detector;
    EXPECT_NO_THROW({
      const auto result = detector.detect_and_parse(text, tools);
      EXPECT_EQ(result.normal_text, "answer");
      EXPECT_TRUE(result.calls.empty());
    });
  }
  for (const std::string attr : {"tool='get_weather'", "tool=get_weather"}) {
    std::string text = "answer" + tool_message("{}");
    text.replace(text.find("tool=\"get_weather\""), 18, attr);
    KimiK3Detector detector;
    EXPECT_TRUE(detector.detect_and_parse(text, tools).calls.empty());
  }
}

TEST(KimiK3DetectorTest, MalformedLaterCallDoesNotExecuteValidPrefix) {
  const std::vector<JsonTool> tools = {make_tool("get_weather")};
  std::string text = "answer" + tool_message("{}");
  const std::string invalid_call =
      "<|open|>call tool=\"get_weather\" index=\"2\"<|sep|>"
      "<|open|>json type=\"object\"<|sep|>bad JSON"
      "<|close|>json<|sep|><|close|>call<|sep|>";
  text.insert(text.find(kToolsEnd), invalid_call);
  KimiK3Detector detector;
  const auto result = detector.detect_and_parse(text, tools);
  EXPECT_EQ(result.normal_text, "answer");
  EXPECT_TRUE(result.calls.empty());
  for (size_t split = 0; split <= text.size(); ++split) {
    KimiK3Detector streaming;
    const auto first =
        streaming.parse_streaming_increment(text.substr(0, split), tools);
    const auto second =
        streaming.parse_streaming_increment(text.substr(split), tools);
    EXPECT_EQ(first.normal_text + second.normal_text, "answer");
    EXPECT_TRUE(first.calls.empty());
    EXPECT_TRUE(second.calls.empty());
  }
}

}  // namespace
}  // namespace function_call
}  // namespace xllm
