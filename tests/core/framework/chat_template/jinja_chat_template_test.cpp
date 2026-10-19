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

#include "core/framework/chat_template/jinja_chat_template.h"

#include <gtest/gtest.h>

#include "core/framework/chat_template/tool_call_arguments.h"

namespace xllm {

class TestableJinjaChatTemplate : public JinjaChatTemplate {
 public:
  TestableJinjaChatTemplate(const TokenizerArgs& args)
      : JinjaChatTemplate(args) {}

  using JinjaChatTemplate::apply;
};

TEST(JinjaChatTemplate, OpenChatModel) {
  // clang-format off
  const std::string template_str =
      "<s>"
      "{% for message in messages %}"
        "{{ 'GPT4 Correct ' + message['role'] + ': ' + message['content'] + '<|end_of_turn|>'}}"
      "{% endfor %}"
      "{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}";

  nlohmann::ordered_json messages = {
      {{"role", "system"}, {"content", "you are a helpful assistant."}},
      {{"role", "user"}, {"content", "hi"}},
      {{"role", "assistant"}, {"content", "what i can do for you?"}},
      {{"role", "user"}, {"content", "how are you?"}}};
  const std::string expected =
    "<s>"
    "GPT4 Correct system: you are a helpful assistant.<|end_of_turn|>"
    "GPT4 Correct user: hi<|end_of_turn|>"
    "GPT4 Correct assistant: what i can do for you?<|end_of_turn|>"
    "GPT4 Correct user: how are you?<|end_of_turn|>"
    "GPT4 Correct Assistant:";
  // clang-format on

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("<|end_of_turn|>");
  TestableJinjaChatTemplate template_(args);
  auto result = template_.apply(messages);
  ASSERT_TRUE(result.has_value());

  EXPECT_EQ(result.value(), expected);
}

TEST(JinjaChatTemplate, AppliesChatTemplateKwargs) {
  const std::string template_str =
      "{% if enable_thinking %}<think>{% endif %}"
      "{% for message in messages %}"
      "{{ message['role'] + ': ' + message['content'] }}"
      "{% endfor %}"
      "{% if not enable_thinking %}<no_think>{% endif %}";

  nlohmann::ordered_json messages = {
      {{"role", "user"}, {"content", "describe this image"}}};
  nlohmann::ordered_json chat_template_kwargs = {{"enable_thinking", false}};

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  TestableJinjaChatTemplate template_(args);
  const nlohmann::ordered_json tools = nlohmann::json::array();
  auto result = template_.apply(messages, tools, chat_template_kwargs);
  ASSERT_TRUE(result.has_value());

  EXPECT_EQ(result.value(), "user: describe this image<no_think>");
}

TEST(JinjaChatTemplate, ReportsRenderedGenerationMode) {
  const std::string template_str =
      "{% if enable_thinking %}<think>{% else %}</think>{% endif %}";
  ChatMessages messages;
  messages.emplace_back("user", "hello");
  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  TestableJinjaChatTemplate template_(args);

  const std::vector<JsonTool> tools;
  auto reasoning = template_.apply_with_generation_mode(
      messages, tools, nlohmann::ordered_json{{"enable_thinking", true}});
  ASSERT_TRUE(reasoning.has_value());
  EXPECT_EQ(reasoning->generation_mode, ChatTemplateGenerationMode::REASONING);

  auto chat = template_.apply_with_generation_mode(
      messages, tools, nlohmann::ordered_json{{"enable_thinking", false}});
  ASSERT_TRUE(chat.has_value());
  EXPECT_EQ(chat->generation_mode, ChatTemplateGenerationMode::CHAT);

  args.chat_template("assistant:");
  TestableJinjaChatTemplate unknown_template(args);
  auto unknown = unknown_template.apply_with_generation_mode(
      messages, tools, nlohmann::ordered_json::object());
  ASSERT_TRUE(unknown.has_value());
  EXPECT_EQ(unknown->generation_mode, ChatTemplateGenerationMode::UNKNOWN);
}

TEST(ChatTemplate, ClassifiesTerminalGenerationMarkers) {
  EXPECT_EQ(ChatTemplate::generation_mode_from_prompt("prefix <think>\n"),
            ChatTemplateGenerationMode::REASONING);
  EXPECT_EQ(ChatTemplate::generation_mode_from_prompt("prefix </think>\t"),
            ChatTemplateGenerationMode::CHAT);
  EXPECT_EQ(ChatTemplate::generation_mode_from_prompt("prefix"),
            ChatTemplateGenerationMode::UNKNOWN);
}

TEST(JinjaChatTemplate, SupportsUndefinedTests) {
  // Qwen3.8 uses both forms for optional chat-template arguments. Quoted text
  // must remain unchanged while executable Jinja expressions are normalized.
  const std::string template_str =
      "plain text: value is undefined|"
      "{{ 'quoted value is undefined' }}|"
      "{% if enable_thinking is undefined %}default"
      "{% else %}configured{% endif %}|"
      "{% if enable_thinking is not undefined %}present"
      "{% else %}missing{% endif %}";

  nlohmann::ordered_json messages = {{{"role", "user"}, {"content", "hello"}}};
  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  TestableJinjaChatTemplate template_(args);

  auto default_result = template_.apply(messages);
  ASSERT_TRUE(default_result.has_value());
  EXPECT_EQ(default_result.value(),
            "plain text: value is undefined|quoted value is undefined|default|"
            "missing");

  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json kwargs = {{"enable_thinking", false}};
  auto configured_result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(configured_result.has_value());
  EXPECT_EQ(configured_result.value(),
            "plain text: value is undefined|quoted value is undefined|"
            "configured|present");
}

TEST(JinjaChatTemplate, ProjectsTextBlocksWithoutLosingReasoning) {
  TokenizerArgs args;
  args.chat_template(
      "{{ messages[0].content }}|{{ messages[0].reasoning_content }}");
  JinjaChatTemplate tmpl(args);
  Message message(
      "assistant",
      MMContentVec{MMContent("text", "first"), MMContent("text", "second")});
  message.reasoning_content = "plan";
  const auto prompt = tmpl.apply(ChatMessages{message});
  ASSERT_TRUE(prompt.has_value());
  EXPECT_EQ(*prompt, "first\nsecond|plan");
  EXPECT_TRUE(std::holds_alternative<MMContentVec>(message.content));
}

TEST(JinjaChatTemplate, PreservesReasoningAcrossMultipleHistoryMessages) {
  TokenizerArgs args;
  args.chat_template(
      "{% for message in messages %}{{ message.content }}|{% endfor %}");
  JinjaChatTemplate tmpl(args);
  Message first("assistant", "answer");
  first.reasoning_content = "plan";
  Message second("assistant", "");
  second.reasoning_content = "next";
  const auto prompt = tmpl.apply(ChatMessages{first, second});
  ASSERT_TRUE(prompt.has_value());
  EXPECT_EQ(*prompt, "<think>plan</think>\nanswer|<think>next</think>|");
  EXPECT_EQ(std::get<std::string>(first.content), "answer");
}

namespace {

TokenizerArgs tool_history_args(const std::string& template_text) {
  TokenizerArgs args;
  args.chat_template(template_text);
  args.bos_token("");
  args.eos_token("");
  return args;
}

ChatMessages tool_history_messages(const std::string& arguments) {
  Message assistant("assistant", "");
  Message::ToolCall call;
  call.id = "call_history";
  call.type = "function";
  call.function.name = "get_weather";
  call.function.arguments = arguments;
  assistant.tool_calls = Message::ToolCallVec{call};
  Message tool("tool", "Sunny, 20 C.");
  tool.tool_call_id = "call_history";
  return {Message("user", "Weather in Beijing?"),
          assistant,
          tool,
          Message("user", "Summarize.")};
}

}  // namespace

TEST(JinjaChatTemplate, ToolArgumentsAreCanonicalObjectsWithoutMutatingApi) {
  const std::string template_text =
      "{% for call in messages[1].tool_calls %}"
      "{% if call.function.arguments is string %}string"
      "{% else %}{{ call.function.arguments | tojson }}{% endif %}"
      "{% endfor %}";
  TestableJinjaChatTemplate tmpl(tool_history_args(template_text));
  const std::string arguments = R"({"z":0,"z":{"z":2,"a":1},"a":[3,2,1]})";
  const auto messages = tool_history_messages(arguments);
  const auto result = tmpl.apply(messages);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, R"({"a": [3, 2, 1], "z": {"a": 1, "z": 2}})");
  EXPECT_EQ(messages[1].tool_calls->front().function.arguments, arguments);

  auto reordered = tool_history_messages(R"({"a":[3,2,1],"z":{"a":1,"z":2}})");
  EXPECT_EQ(tmpl.apply(reordered), result);
}

TEST(JinjaChatTemplate, GlmArgumentsItemsUsesSortedKeys) {
  const std::string template_text =
      "{% for k, v in messages[1].tool_calls[0].function.arguments.items() %}"
      "<arg_key>{{ k }}</arg_key><arg_value>{{ v | tojson }}</arg_value>"
      "{% endfor %}";
  TestableJinjaChatTemplate tmpl(tool_history_args(template_text));
  const auto result =
      tmpl.apply(tool_history_messages(R"({"z":2,"a":"北京"})"));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(
      *result,
      R"(<arg_key>a</arg_key><arg_value>"北京"</arg_value><arg_key>z</arg_key><arg_value>2</arg_value>)");
}

TEST(JinjaChatTemplate, InvalidToolArgumentsNeverRenderEmptyFallback) {
  TestableJinjaChatTemplate tmpl(tool_history_args("otherwise valid"));
  for (const std::string arguments : {"",
                                      " ",
                                      "{",
                                      "{\"a\":1} trailing",
                                      "[]",
                                      "[{}]",
                                      "null",
                                      "true",
                                      "42",
                                      "\"text\""}) {
    SCOPED_TRACE(arguments);
    const auto messages = tool_history_messages(arguments);
    EXPECT_FALSE(tmpl.apply(messages).has_value());
    const Status status = validate_tool_call_arguments(messages);
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.reason(), FailureReason::INVALID_PARAMETER);
    EXPECT_EQ(status.param(), "messages[1].tool_calls[0].function.arguments");
  }
  EXPECT_TRUE(tmpl.apply(tool_history_messages("{}")).has_value());
}

TEST(JinjaChatTemplate, RawInputsFollowSameContractAndPreserveCaller) {
  TestableJinjaChatTemplate tmpl(
      tool_history_args("{% for message in messages %}"
                        "{% for call in message.tool_calls %}"
                        "{{ call.function.arguments | tojson }}"
                        "{% endfor %}{% endfor %}"));
  nlohmann::ordered_json messages = {
      {{"role", "assistant"},
       {"content", ""},
       {"tool_calls",
        {{{"function",
           {{"name", "probe"}, {"arguments", R"({"z":2,"a":1})"}}}}}}}};
  const auto original = messages;
  auto result = tmpl.apply(messages);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, R"({"a": 1, "z": 2})");
  EXPECT_EQ(messages, original);
  messages[0]["tool_calls"][0]["function"]["arguments"] =
      nlohmann::ordered_json{{"z", 2}, {"a", 1}};
  EXPECT_EQ(tmpl.apply(messages), result);
  messages[0]["tool_calls"][0]["function"]["arguments"] = 1;
  EXPECT_FALSE(tmpl.apply(messages).has_value());
}

}  // namespace xllm
