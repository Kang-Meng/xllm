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

#include "framework/chat_template/deepseek_v41_cpp_template.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include "core/common/constants.h"
#include "core/common/message.h"

namespace xllm {
namespace {

DeepseekV41CppTemplate make_encoder() {
  TokenizerArgs args;
  args.bos_token("<｜begin▁of▁sentence｜>");
  return DeepseekV41CppTemplate(args);
}

size_t count_image_placeholders(const std::string& text) {
  size_t count = 0;
  for (size_t pos = text.find(kDeepseekV41ImageAlias); pos != std::string::npos;
       pos = text.find(kDeepseekV41ImageAlias,
                       pos + sizeof(kDeepseekV41ImageAlias) - 1)) {
    ++count;
  }
  for (size_t pos = text.find(kDeepseekV41ImagePlaceholder);
       pos != std::string::npos;
       pos = text.find(kDeepseekV41ImagePlaceholder,
                       pos + sizeof(kDeepseekV41ImagePlaceholder) - 1)) {
    ++count;
  }
  return count;
}

MMContent text_item(const std::string& text) { return MMContent("text", text); }

MMContent image_item(const std::string& url) {
  return MMContent("image_url", ImageURL{url, {}});
}

std::string render(const ChatMessages& messages) {
  auto encoder = make_encoder();
  nlohmann::ordered_json kwargs = nlohmann::json::object();
  kwargs["thinking"] = false;
  auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
  EXPECT_TRUE(prompt.has_value());
  return prompt.value_or("");
}

// One text marker plus two images used to emit a single placeholder (the old
// have-any-marker boolean suppressed all injection), so the processor rejected
// the request for having two images but one slot.
TEST(DeepseekV41CppTemplate, MixedMarkerAndImagesKeepPlaceholderCount) {
  ChatMessages messages;
  messages.emplace_back("user",
                        MMContentVec{text_item("look at this <image>"),
                                     image_item("a.png"),
                                     image_item("b.png")});

  EXPECT_EQ(count_image_placeholders(render(messages)), 2u);
}

TEST(DeepseekV41CppTemplate, EscapedMarkerIsLiteralText) {
  ChatMessages messages;
  messages.emplace_back(
      "user",
      MMContentVec{text_item("literal \\<image>"), image_item("a.png")});

  const std::string prompt = render(messages);
  // The escaped alias is not counted as a marker, so the image still gets one
  // injected native placeholder; the escaped text is parked in the sentinel
  // until the prompt processor restores it.
  EXPECT_EQ(count_image_placeholders(prompt), 1u);
  EXPECT_EQ(prompt.find(kDeepseekV41ImageAliasEscape), std::string::npos);
  EXPECT_NE(prompt.find(kDeepseekV41ImageLiteralSentinel), std::string::npos);
}

TEST(DeepseekV41CppTemplate, EscapedMarkerInStringMessageStaysLiteral) {
  ChatMessages messages;
  messages.emplace_back("system", "literal \\<image>");
  messages.emplace_back("user", MMContentVec{image_item("a.png")});

  const std::string prompt = render(messages);
  // The image lives in another message, so the escaped marker must remain
  // protected until that image has been expanded by the prompt processor.
  EXPECT_EQ(count_image_placeholders(prompt), 1u);
  EXPECT_EQ(prompt.find(kDeepseekV41ImageAliasEscape), std::string::npos);
  EXPECT_NE(prompt.find(kDeepseekV41ImageLiteralSentinel), std::string::npos);
}

TEST(DeepseekV41CppTemplate, EscapedMarkerInStringMessageRestoresWithoutImage) {
  ChatMessages messages;
  messages.emplace_back("user", "literal \\<image>");

  const std::string prompt = render(messages);
  // With no image in the request there is no prompt-processor expansion step,
  // so the template must restore the literal itself.
  EXPECT_EQ(prompt.find(kDeepseekV41ImageLiteralSentinel), std::string::npos);
  EXPECT_NE(prompt.find("literal <image>"), std::string::npos);
}

TEST(DeepseekV41CppTemplate, ContentOnlyImagesGetOnePlaceholderEach) {
  ChatMessages messages;
  messages.emplace_back("user",
                        MMContentVec{image_item("a.png"), image_item("b.png")});

  EXPECT_EQ(count_image_placeholders(render(messages)), 2u);
}

// When the markers already cover every image, no extra placeholder is added.
TEST(DeepseekV41CppTemplate, MatchingMarkersAreNotDoubled) {
  ChatMessages messages;
  messages.emplace_back("user",
                        MMContentVec{text_item("<image> and <image>"),
                                     image_item("a.png"),
                                     image_item("b.png")});

  EXPECT_EQ(count_image_placeholders(render(messages)), 2u);
}

// A text marker without any image is left untouched for the processor to strip.
TEST(DeepseekV41CppTemplate, TextMarkerWithoutImagesSurvives) {
  ChatMessages messages;
  messages.emplace_back("user", MMContentVec{text_item("only <image> text")});

  EXPECT_EQ(count_image_placeholders(render(messages)), 1u);
}

}  // namespace
}  // namespace xllm
