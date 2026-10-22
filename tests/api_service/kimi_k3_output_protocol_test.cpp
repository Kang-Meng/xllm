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

#include <google/protobuf/util/json_util.h>
#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "anthropic.pb.h"
#include "api_service/chat_json_parser.h"
#include "api_service/stream_output_parser.h"
#include "api_service/utils.h"
#include "core/framework/chat_template/kimi_k3_cpp_template.h"
#include "core/framework/request/request_params.h"
#include "function_call/function_call_parser.h"
#include "multimodal.pb.h"

namespace xllm {
namespace {

// The shared reasoning parser expects a complete think close in a chunk.
// Keep splitting response/tool envelopes to exercise their independent parser.
std::vector<std::string> stream_chunks(const std::string& text, bool thinking) {
  std::vector<std::string> chunks;
  size_t offset = 0;
  if (thinking) {
    const std::string end = "<|close|>think<|sep|>";
    offset = text.find(end) + end.size();
    chunks.push_back(text.substr(0, offset));
  }
  for (; offset < text.size(); ++offset) {
    chunks.push_back(text.substr(offset, 1));
  }
  return chunks;
}

TEST(KimiK3OutputProtocolTest, OptInPreservesOtherParserSelections) {
  StreamOutputParser kimi({}, "kimi_k3", "kimi_k3", true);
  EXPECT_TRUE(kimi.is_tool_call());
  EXPECT_NE(kimi.get_tool_call_parser(0), nullptr);
  EXPECT_TRUE(api_service::should_run_tool_call_parser({}, "kimi_k3"));
  EXPECT_TRUE(api_service::should_run_tool_call_parser(
      {}, *kimi.get_tool_call_parser(0)));
  const std::vector<JsonTool> tools = {JsonTool(
      "function", JsonFunction("weather", "", nlohmann::json::object()))};
  StreamOutputParser kimi_with_tools(tools, "kimi_k3", "kimi_k3", true);
  EXPECT_TRUE(kimi_with_tools.is_tool_call());
  EXPECT_NE(kimi_with_tools.get_tool_call_parser(0), nullptr);
  EXPECT_TRUE(api_service::should_run_tool_call_parser(tools, "qwen25"));
  for (const auto& format : {"qwen25",
                             "qwen3_coder",
                             "kimi_k2",
                             "deepseekv3",
                             "deepseekv32",
                             "deepseekv4",
                             "deepseekv41",
                             "glm45",
                             "glm47",
                             "glm5",
                             ""}) {
    StreamOutputParser peer({}, format, "");
    EXPECT_FALSE(peer.is_tool_call());
    EXPECT_EQ(peer.get_tool_call_parser(0), nullptr);
    EXPECT_FALSE(api_service::should_run_tool_call_parser({}, format));
    if (std::string(format).empty()) {
      continue;
    }
    function_call::FunctionCallParser selected({}, format);
    EXPECT_FALSE(api_service::should_run_tool_call_parser({}, selected));
    EXPECT_TRUE(api_service::should_run_tool_call_parser(tools, selected));
  }
}

TEST(KimiK3OutputProtocolTest, NoToolsSequencesKeepIndependentEnvelopeState) {
  StreamOutputParser parser({}, "kimi_k3", "kimi_k3", false);
  EXPECT_TRUE(parser.is_tool_call());
  const auto feed = [&parser](size_t index, const std::string& chunk) {
    const auto reasoning =
        parser.get_reasoning_parser(index)->parse_stream_chunk(chunk);
    if (!reasoning.normal_text.has_value()) {
      return std::string();
    }
    return parser.get_tool_call_parser(index)
        ->parse_streaming_increment(*reasoning.normal_text)
        .normal_text;
  };

  EXPECT_EQ(feed(0, "<|open|>response<|sep|>first<|cl"), "first");
  EXPECT_EQ(feed(1,
                 "<|open|>response<|sep|>second<|close|>response<|sep|>"
                 "<|close|>message<|sep|>"),
            "second");
  EXPECT_TRUE(feed(0, "ose|>response<|sep|><|close|>message<|sep|>").empty());
  EXPECT_NE(parser.get_tool_call_parser(1), nullptr);
}

TEST(KimiK3OutputProtocolTest, EnvelopeParserPreservesOrdinaryText) {
  for (const std::string text :
       {"  ordinary answer\n",
        "Literal <|open|>tools-like text and <|close|>response prose.",
        "The JSON value is {\"response\": \"plain text\"}."}) {
    const auto result =
        api_service::process_tool_calls(text, {}, "kimi_k3", "stop");
    EXPECT_EQ(result.text, text);
    EXPECT_EQ(result.finish_reason, "stop");
    EXPECT_FALSE(result.tool_calls.has_value());
  }
}

TEST(KimiK3OutputProtocolTest, MalformedEnvelopePreservesTextAndFinishReason) {
  const std::vector<JsonTool> tools = {JsonTool(
      "function", JsonFunction("weather", "", nlohmann::json::object()))};
  const std::string text =
      "useful answer<|close|>response<|sep|><|open|>tools<|sep|>"
      "<|open|>call tool=\"weather\" index=\"1\"<|sep|>"
      "<|open|>json type=\"object\"<|sep|>invalid JSON"
      "<|close|>json<|sep|><|close|>call<|sep|>"
      "<|close|>tools<|sep|><|close|>message<|sep|>";
  for (const std::string finish : {"stop", "length"}) {
    const auto result =
        api_service::process_tool_calls(text, tools, "kimi_k3", finish);
    EXPECT_EQ(result.text, "useful answer");
    EXPECT_EQ(result.finish_reason, finish);
    EXPECT_FALSE(result.tool_calls.has_value());
  }
}

TEST(KimiK3OutputProtocolTest,
     CompleteToolsSetFinishReasonThroughTheApiHelper) {
  const std::vector<JsonTool> tools = {JsonTool(
      "function", JsonFunction("weather", "", nlohmann::json::object()))};
  const std::string text =
      "<|close|>response<|sep|><|open|>tools<|sep|>"
      "<|open|>call tool=\"weather\" index=\"1\"<|sep|>"
      "<|open|>json type=\"object\"<|sep|>{\"city\":\"Beijing\"}"
      "<|close|>json<|sep|><|close|>call<|sep|>"
      "<|close|>tools<|sep|><|close|>message<|sep|>";
  auto result = api_service::process_tool_calls(text, tools, "kimi_k3", "stop");
  EXPECT_EQ(result.text, "");
  EXPECT_EQ(result.finish_reason, "tool_calls");
  ASSERT_TRUE(result.tool_calls.has_value());
  ASSERT_EQ(result.tool_calls->size(), 1);
  EXPECT_EQ(result.tool_calls->Get(0).function().name(), "weather");
  EXPECT_EQ(
      nlohmann::json::parse(result.tool_calls->Get(0).function().arguments()),
      nlohmann::json({{"city", "Beijing"}}));
  result = api_service::process_tool_calls(text, tools, "kimi_k3", "length");
  EXPECT_EQ(result.finish_reason, "length");
}

TEST(KimiK3OutputProtocolTest,
     StreamsToolCallsThroughTheProductionParserChain) {
  proto::AnthropicMessagesRequest request;
  request.set_model("kimi-k3");
  request.set_max_tokens(128);
  request.add_tools()->set_name("weather");
  RequestParams params(request, "", "");
  ASSERT_FALSE(request.has_thinking());
  ASSERT_TRUE(params.chat_template_kwargs.empty());
  const bool force_reasoning =
      DetectorRegistry::get_instance().prompt_opens_think("kimi_k3") &&
      api_service::get_enable_thinking_from_request(params.chat_template_kwargs,
                                                    "kimi_k3");
  KimiK3CppTemplate chat_template(TokenizerArgs{});
  const auto prompt = chat_template.apply(
      {Message("user", "hello")}, params.tools, params.chat_template_kwargs);
  ASSERT_TRUE(prompt.has_value());
  ASSERT_TRUE(prompt->ends_with("<|open|>think<|sep|>"));
  ASSERT_TRUE(force_reasoning);
  StreamOutputParser parser(
      params.tools, "kimi_k3", "kimi_k3", force_reasoning);
  const std::string text =
      "plan<|close|>think<|sep|><|open|>response<|sep|>"
      "<|close|>response<|sep|><|open|>tools<|sep|>"
      "<|open|>call tool=\"weather\" index=\"1\"<|sep|>"
      "<|open|>json type=\"object\"<|sep|>{\"city\":\"Beijing\"}"
      "<|close|>json<|sep|><|close|>call<|sep|>"
      "<|close|>tools<|sep|><|close|>message<|sep|>";
  ReasoningParser non_stream("kimi_k3", false, force_reasoning);
  const auto parsed = non_stream.parse_non_stream(text);
  EXPECT_EQ(parsed.reasoning_text, "plan");
  ASSERT_TRUE(parsed.normal_text.has_value());
  const auto result = api_service::process_tool_calls(
      *parsed.normal_text, params.tools, "kimi_k3", "stop");
  EXPECT_EQ(result.text, "");
  EXPECT_EQ(result.finish_reason, "tool_calls");
  ASSERT_TRUE(result.tool_calls.has_value());
  ASSERT_EQ(result.tool_calls->size(), 1);
  EXPECT_EQ(result.tool_calls->Get(0).function().name(), "weather");
  std::string reasoning, content;
  std::vector<function_call::ToolCallItem> calls;
  for (const auto& chunk : stream_chunks(text, force_reasoning)) {
    auto result = parser.get_reasoning_parser(0)->parse_stream_chunk(chunk);
    reasoning += result.reasoning_text.value_or("");
    if (!result.normal_text.has_value()) {
      continue;
    }
    auto delta = parser.get_tool_call_parser(0)->parse_streaming_increment(
        *result.normal_text);
    content += delta.normal_text;
    calls.insert(calls.end(), delta.calls.begin(), delta.calls.end());
  }
  EXPECT_EQ(reasoning, "plan");
  EXPECT_EQ(content, "");
  ASSERT_EQ(calls.size(), 1);
  EXPECT_EQ(calls[0].name, "weather");
  EXPECT_EQ(nlohmann::json::parse(calls[0].parameters),
            nlohmann::json({{"city", "Beijing"}}));
}

TEST(KimiK3OutputProtocolTest, ThinkingMatchesTemplateAndParserChain) {
  const std::vector<std::pair<nlohmann::ordered_json, bool>> cases = {
      {nlohmann::ordered_json::object(), true},
      {{{"thinking", false}}, false},
      {{{"thinking", true}}, true},
      // The shared resolver accepts only bool flags, without coercing a
      // string/number/null. Template and response parsing must agree on it.
      {{{"thinking", "false"}}, false},
      {{{"thinking", 0}}, false},
      {{{"thinking", nullptr}}, false},
      {{{"thinking", "false"}, {"enable_thinking", true}}, true},
      {{{"enable_thinking", false}}, false},
      {{{"enable_thinking", true}}, true},
      {{{"thinking", false}, {"enable_thinking", true}}, true},
      {{{"thinking", true}, {"enable_thinking", false}}, true}};
  for (const auto& [kwargs, thinking] : cases) {
    proto::AnthropicMessagesRequest request;
    request.set_model("kimi-k3");
    request.set_max_tokens(128);
    RequestParams params(request, "", "");
    params.chat_template_kwargs = kwargs;
    const bool force_reasoning =
        DetectorRegistry::get_instance().prompt_opens_think("kimi_k3") &&
        api_service::get_enable_thinking_from_request(
            params.chat_template_kwargs, "kimi_k3");
    EXPECT_EQ(force_reasoning, thinking);
    KimiK3CppTemplate chat_template(TokenizerArgs{});
    const auto prompt = chat_template.apply(
        {Message("user", "hello")}, {}, params.chat_template_kwargs);
    ASSERT_TRUE(prompt.has_value());
    EXPECT_TRUE(prompt->ends_with(thinking ? "<|open|>think<|sep|>"
                                           : "<|open|>response<|sep|>"));
    const std::string text =
        (thinking ? "plan<|close|>think<|sep|><|open|>response<|sep|>" : "") +
        std::string("answer<|close|>response<|sep|> \n<|close|>message<|sep|>");
    ReasoningParser non_stream("kimi_k3", false, force_reasoning);
    const auto expected = non_stream.parse_non_stream(text);
    EXPECT_EQ(expected.normal_text,
              (thinking ? "<|open|>response<|sep|>" : "") +
                  std::string("answer<|close|>response<|sep|> \n"
                              "<|close|>message<|sep|>"));
    EXPECT_EQ(expected.reasoning_text.value_or(""), thinking ? "plan" : "");
    const auto visible = api_service::process_tool_calls(
        *expected.normal_text, {}, "kimi_k3", "stop");
    EXPECT_EQ(visible.text, "answer");
    StreamOutputParser streaming({}, "kimi_k3", "kimi_k3", force_reasoning);
    EXPECT_TRUE(streaming.is_tool_call());
    std::string content;
    std::string reasoning;
    for (const auto& chunk : stream_chunks(text, force_reasoning)) {
      const auto result =
          streaming.get_reasoning_parser(0)->parse_stream_chunk(chunk);
      reasoning += result.reasoning_text.value_or("");
      if (result.normal_text.has_value()) {
        content += streaming.get_tool_call_parser(0)
                       ->parse_streaming_increment(*result.normal_text)
                       .normal_text;
      }
    }
    EXPECT_EQ(content, visible.text);
    EXPECT_EQ(reasoning, expected.reasoning_text.value_or(""));
  }
}

TEST(KimiK3OutputProtocolTest, NoToolsEnvelopeUsesSelectedToolParser) {
  for (bool thinking : {false, true}) {
    const nlohmann::json kwargs = {{"thinking", thinking}};
    const bool force_reasoning =
        api_service::get_enable_thinking_from_request(kwargs, "kimi_k3");
    const std::string text =
        (thinking ? "plan<|close|>think<|sep|><|open|>response<|sep|>" : "") +
        std::string("answer<|close|>response<|sep|><|close|>message<|sep|>");
    StreamOutputParser streaming({}, "kimi_k3", "kimi_k3", force_reasoning);
    ASSERT_TRUE(streaming.is_tool_call());
    std::string reasoning;
    std::string content;
    for (const auto& chunk : stream_chunks(text, force_reasoning)) {
      const auto result =
          streaming.get_reasoning_parser(0)->parse_stream_chunk(chunk);
      reasoning += result.reasoning_text.value_or("");
      if (result.normal_text.has_value()) {
        content += streaming.get_tool_call_parser(0)
                       ->parse_streaming_increment(*result.normal_text)
                       .normal_text;
      }
    }
    EXPECT_EQ(content, "answer");
    EXPECT_EQ(reasoning, thinking ? "plan" : "");

    ReasoningParser non_stream("kimi_k3", false, force_reasoning);
    const auto parsed = non_stream.parse_non_stream(text);
    EXPECT_EQ(api_service::process_tool_calls(
                  *parsed.normal_text, {}, "kimi_k3", "stop")
                  .text,
              "answer");
  }
}

TEST(KimiK3OutputProtocolTest, PeerThinkingDoesNotSelectKimiEnvelopeRules) {
  const std::vector<JsonTool> tools = {JsonTool(
      "function", JsonFunction("weather", "", nlohmann::json::object()))};
  const std::string content = "answer<|close|>response<|sep|>";
  for (bool thinking : {false, true}) {
    const nlohmann::json kwargs = {{"thinking", thinking}};
    const bool force_reasoning =
        api_service::get_enable_thinking_from_request(kwargs, "qwen3");
    EXPECT_EQ(force_reasoning, thinking);
    const std::string output = (thinking ? "plan</think>" : "") + content;
    ReasoningParser non_stream("qwen3", false, force_reasoning);
    const auto expected = non_stream.parse_non_stream(output);
    EXPECT_EQ(expected.normal_text, content);
    EXPECT_EQ(expected.reasoning_text.value_or(""), thinking ? "plan" : "");

    StreamOutputParser streaming(tools, "qwen25", "qwen3", force_reasoning);
    if (thinking) {
      const auto reasoning =
          streaming.get_reasoning_parser(0)->parse_stream_chunk("plan</think>");
      EXPECT_EQ(reasoning.reasoning_text, "plan");
      EXPECT_TRUE(reasoning.normal_text.value_or("").empty());
    }
    const auto body =
        streaming.get_reasoning_parser(0)->parse_stream_chunk(content);
    EXPECT_EQ(body.normal_text, content);
    const auto result = api_service::process_tool_calls(
        body.normal_text.value_or(""), tools, "qwen25", "stop");
    EXPECT_EQ(result.text, content);
    EXPECT_EQ(result.finish_reason, "stop");
    EXPECT_FALSE(result.tool_calls.has_value());
  }
}

TEST(KimiK3OutputProtocolTest, EffortAloneDoesNotSelectThinkingMode) {
  KimiK3CppTemplate chat_template(TokenizerArgs{});
  const ChatMessages messages = {Message("user", "hello")};
  const auto expected = chat_template.apply(messages);
  ASSERT_TRUE(expected.has_value());
  for (const std::string effort : {"none", "high"}) {
    RequestParams params;
    params.chat_template_kwargs = {{"reasoning_effort", effort}};
    EXPECT_FALSE(params.chat_template_kwargs.contains("thinking"));
    EXPECT_FALSE(params.chat_template_kwargs.contains("enable_thinking"));
    EXPECT_FALSE(params.chat_template_kwargs.contains("thinking_effort"));
    EXPECT_TRUE(api_service::get_enable_thinking_from_request(
        params.chat_template_kwargs, "kimi_k3"));
    EXPECT_EQ(chat_template.apply(messages, {}, params.chat_template_kwargs),
              expected);

    params.chat_template_kwargs["thinking"] = false;
    EXPECT_FALSE(api_service::get_enable_thinking_from_request(
        params.chat_template_kwargs, "kimi_k3"));
    const auto prompt =
        chat_template.apply(messages, {}, params.chat_template_kwargs);
    ASSERT_TRUE(prompt.has_value());
    EXPECT_TRUE(prompt->ends_with("<|open|>response<|sep|>"));
  }
}

TEST(KimiK3OutputProtocolTest, MultimodalRequestUsesTheSameOutputProtocol) {
  VlmChatJsonParser json_parser;
  const auto [status, json] = json_parser.preprocess(
      R"({"model":"kimi-k3","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}},{"type":"text","text":"describe"}]}],"chat_template_kwargs":{"thinking":false}})");
  ASSERT_TRUE(status.ok()) << status.message();
  proto::MMChatRequest request;
  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  const auto parsed =
      google::protobuf::util::JsonStringToMessage(json, &request, options);
  ASSERT_TRUE(parsed.ok()) << parsed.ToString();
  RequestParams params(request, "", "");
  const bool force_reasoning = api_service::get_enable_thinking_from_request(
      params.chat_template_kwargs, "kimi_k3");
  EXPECT_FALSE(force_reasoning);
  const std::string output =
      "description<|close|>response<|sep|><|close|>message<|sep|>";
  StreamOutputParser streaming(
      params.tools, "kimi_k3", "kimi_k3", force_reasoning);
  const auto reasoning =
      streaming.get_reasoning_parser(0)->parse_stream_chunk(output);
  ASSERT_TRUE(reasoning.normal_text.has_value());
  EXPECT_TRUE(streaming.is_tool_call());
  const auto stream_result =
      streaming.get_tool_call_parser(0)->parse_streaming_increment(
          *reasoning.normal_text);
  EXPECT_EQ(stream_result.normal_text, "description");
  ReasoningParser non_stream("kimi_k3", false, force_reasoning);
  const auto reasoning_result = non_stream.parse_non_stream(output);
  ASSERT_TRUE(reasoning_result.normal_text.has_value());
  EXPECT_EQ(reasoning_result.normal_text, reasoning.normal_text);
  EXPECT_EQ(api_service::process_tool_calls(
                *reasoning_result.normal_text, {}, "kimi_k3", "stop")
                .text,
            "description");
}

}  // namespace
}  // namespace xllm
