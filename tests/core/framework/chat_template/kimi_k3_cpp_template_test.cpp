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

#include "framework/chat_template/kimi_k3_cpp_template.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(KimiK3CppTemplateTest, TextArrayMatchesPlainText) {
  KimiK3CppTemplate chat_template(TokenizerArgs{});
  const ChatMessages plain = {Message("user", "hello")};
  const ChatMessages multimodal = {
      Message("user", MMContentVec{MMContent("text", std::string("hello"))})};
  EXPECT_EQ(chat_template.apply(plain), chat_template.apply(multimodal));
}

TEST(KimiK3CppTemplateTest, KeepsImagePlaceholderBeforeText) {
  KimiK3CppTemplate chat_template(TokenizerArgs{});
  const ChatMessages messages = {
      Message("user",
              MMContentVec{MMContent("image_url", ImageURL{}),
                           MMContent("text", std::string("describe"))})};
  const auto prompt = chat_template.apply(messages);
  ASSERT_TRUE(prompt.has_value());
  EXPECT_NE(prompt->find("<|media_begin|>image<|media_content|>"
                         "<|media_pad|><|media_end|>describe"),
            std::string::npos);
}

TEST(KimiK3CppTemplateTest, KeepsVideoPlaceholdersInMediaOrder) {
  KimiK3CppTemplate chat_template(TokenizerArgs{});
  for (const std::string type : {"video", "video_url", "video_embedding"}) {
    const ChatMessages messages = {
        Message("user",
                MMContentVec{MMContent("image_url", ImageURL{}),
                             MMContent(type, VideoURL{}),
                             MMContent("text", std::string("describe"))})};
    const auto prompt = chat_template.apply(messages);
    ASSERT_TRUE(prompt.has_value());
    EXPECT_NE(
        prompt->find(
            "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
            "<|media_begin|>video<|media_content|><|media_pad|><|media_end|>"
            "describe"),
        std::string::npos);
  }
}

TEST(KimiK3CppTemplateTest, RejectsUnsupportedMediaInsteadOfDroppingIt) {
  KimiK3CppTemplate chat_template(TokenizerArgs{});
  for (const std::string type :
       {"audio", "audio_url", "audio_embedding", "unknown"}) {
    for (const std::string role : {"user", "assistant"}) {
      const ChatMessages messages = {
          Message(role, MMContentVec{MMContent(type, AudioURL{})})};
      EXPECT_FALSE(chat_template.apply(messages).has_value()) << type << role;
    }
  }
}

TEST(KimiK3CppTemplateTest, AddsRequiredToolChoiceInstruction) {
  KimiK3CppTemplate chat_template(TokenizerArgs{});
  ChatMessages messages = {Message("user", "weather?")};
  nlohmann::ordered_json kwargs = {{"thinking", false},
                                   {"tool_choice", "required"}};

  const auto prompt = chat_template.apply(messages, {}, kwargs);

  ASSERT_TRUE(prompt.has_value());
  EXPECT_NE(prompt->find("<|open|>message role=\"system\" "
                         "type=\"tool-choice\"<|sep|>"),
            std::string::npos);
  EXPECT_NE(prompt->find("You MUST call tools in the next message."),
            std::string::npos);
  EXPECT_TRUE(
      prompt->ends_with("<|open|>message role=\"assistant\"<|sep|>"
                        "<|open|>response<|sep|>"));
}

TEST(KimiK3CppTemplateTest, UsesMappedThinkingEffort) {
  KimiK3CppTemplate chat_template(TokenizerArgs{});
  ChatMessages messages = {Message("user", "hello")};
  nlohmann::ordered_json kwargs = {{"thinking", true},
                                   {"thinking_effort", "high"}};

  const auto prompt = chat_template.apply(messages, {}, kwargs);

  ASSERT_TRUE(prompt.has_value());
  EXPECT_NE(prompt->find("thinking_effort=high"), std::string::npos);
  EXPECT_TRUE(
      prompt->ends_with("<|open|>message role=\"assistant\"<|sep|>"
                        "<|open|>think<|sep|>"));
}

TEST(KimiK3CppTemplateTest, AcceptsMediumThinkingEffort) {
  KimiK3CppTemplate chat_template(TokenizerArgs{});
  ChatMessages messages = {Message("user", "hello")};
  nlohmann::ordered_json kwargs = {{"thinking", true},
                                   {"thinking_effort", "medium"}};

  const auto prompt = chat_template.apply(messages, {}, kwargs);

  ASSERT_TRUE(prompt.has_value());
  EXPECT_NE(prompt->find("thinking_effort=medium"), std::string::npos);
  EXPECT_TRUE(
      prompt->ends_with("<|open|>message role=\"assistant\"<|sep|>"
                        "<|open|>think<|sep|>"));
}

TEST(KimiK3CppTemplateTest, RejectsUnsupportedThinkingEffort) {
  KimiK3CppTemplate chat_template(TokenizerArgs{});
  ChatMessages messages = {Message("user", "hello")};
  nlohmann::ordered_json kwargs = {{"thinking", true},
                                   {"thinking_effort", "invalid"}};

  EXPECT_FALSE(chat_template.apply(messages, {}, kwargs).has_value());
}

TEST(KimiK3CppTemplateTest, ResolvesToolResultNameFromToolCallId) {
  KimiK3CppTemplate chat_template(TokenizerArgs{});
  Message assistant("assistant", "");
  Message::ToolCall tool_call;
  tool_call.id = "call_123";
  tool_call.type = "function";
  tool_call.function.name = "get_weather";
  tool_call.function.arguments = R"({"city":"Beijing"})";
  assistant.tool_calls = Message::ToolCallVec{tool_call};
  Message tool_result("tool", R"({"temperature":20})");
  tool_result.tool_call_id = "call_123";
  ChatMessages messages = {Message("user", "weather?"), assistant, tool_result};
  nlohmann::ordered_json kwargs = {{"thinking", false}};

  const auto prompt = chat_template.apply(messages, {}, kwargs);

  ASSERT_TRUE(prompt.has_value());
  EXPECT_NE(prompt->find("<|open|>message role=\"tool\" "
                         "tool=\"get_weather\" index=\"1\"<|sep|>"),
            std::string::npos);
  EXPECT_EQ(prompt->find("tool=\"call_123\""), std::string::npos);
}

TEST(KimiK3CppTemplateTest, RejectsToolAttributeDelimiters) {
  KimiK3CppTemplate chat_template(TokenizerArgs{});
  for (const std::string name : {"x<sep", "x>sep", "x|sep", "x<|sep|>"}) {
    Message assistant("assistant", "");
    Message::ToolCall tool_call;
    tool_call.function.name = name;
    assistant.tool_calls = Message::ToolCallVec{tool_call};
    EXPECT_FALSE(chat_template.apply({assistant}).has_value()) << name;

    JsonTool declaration;
    declaration.type = "function";
    declaration.function.name = name;
    EXPECT_FALSE(chat_template
                     .apply({Message("user", "call")},
                            {declaration},
                            nlohmann::ordered_json::object())
                     .has_value())
        << name;
  }
}

TEST(KimiK3CppTemplateTest, PreservesCheckpointAttributeEscaping) {
  KimiK3CppTemplate chat_template(TokenizerArgs{});
  Message assistant("assistant", "");
  Message::ToolCall tool_call;
  tool_call.id = "call_123";
  tool_call.function.name = "x&\"";
  assistant.tool_calls = Message::ToolCallVec{tool_call};
  Message result("tool", "done");
  result.tool_call_id = "call_123";
  const auto prompt = chat_template.apply({assistant, result});
  ASSERT_TRUE(prompt.has_value());
  EXPECT_NE(prompt->find("<|open|>call tool=\"x&amp;&quot;\" index=\"1\""),
            std::string::npos);
  EXPECT_NE(prompt->find("<|open|>message role=\"tool\" tool=\"x&amp;&quot;\""),
            std::string::npos);
}

}  // namespace
}  // namespace xllm
