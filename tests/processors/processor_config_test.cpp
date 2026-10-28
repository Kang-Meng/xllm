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

#include "processors/processor_config.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace xllm {
namespace {

struct ExampleArgs {
  int32_t stride = 4;
};

class ExampleProcessor final {
 public:
  ExampleProcessor(const ExampleArgs& args, const ProcessorConfig& config)
      : stride_(config.value("frame_stride", args.stride)) {}

  int32_t stride() const { return stride_; }

 private:
  int32_t stride_;
};

TEST(ProcessorConfigTest,
     NewProcessorUsesParametersWithoutCentralRegistration) {
  const ProcessorConfig config(nlohmann::json{{"frame_stride", 8}},
                               "mm_args.video");
  const ExampleProcessor processor(ExampleArgs(), config);
  config.validate();
  EXPECT_EQ(processor.stride(), 8);
  const ProcessorConfig defaults;
  EXPECT_EQ(ExampleProcessor(ExampleArgs(), defaults).stride(), 4);
}

TEST(ProcessorConfigTest, EmptyAndMissingModalitiesUseDefaults) {
  const ProcessorConfig config(
      nlohmann::json{{"image", nlohmann::json::object()}});
  EXPECT_TRUE(config.empty("image"));
  EXPECT_TRUE(config.empty("video"));
  EXPECT_EQ(config.value<int32_t>("image.max_pixels", 4096), 4096);
  config.validate("image");
  config.validate("video");
  config.validate("audio");
}

TEST(ProcessorConfigTest, ConsumersReadIndependentModalities) {
  const ProcessorConfig config(nlohmann::json{
      {"image", {{"max_pixels", 4096}}}, {"video", {{"frame_stride", 8}}}});
  EXPECT_FALSE(config.empty("image"));
  EXPECT_FALSE(config.empty("video"));
  EXPECT_EQ(config.value<int32_t>("image.max_pixels", 0), 4096);
  config.validate("image");
  EXPECT_DEATH(config.validate("video"), "video.frame_stride");
  EXPECT_EQ(config.value<int32_t>("video.frame_stride", 4), 8);
  config.validate("video");
  config.validate();
}

TEST(ProcessorConfigTest, ReadsNestedFieldsAndValidatesEveryLeaf) {
  const ProcessorConfig config(
      nlohmann::json{{"size", {{"height", 64}, {"width", 128}}}});
  EXPECT_EQ(config.value<int32_t>("size.height", 32), 64);
  EXPECT_DEATH(config.validate(), "size.width");
  EXPECT_EQ(config.value<int32_t>("size.width", 32), 128);
  config.validate();
}

TEST(ProcessorConfigTest, LiteralKeysCannotImpersonateNestedFields) {
  for (const std::string key :
       {"size.longest_edge", "size/longest_edge", "size~longest_edge"}) {
    const ProcessorConfig config(nlohmann::json{
        {"image", {{"size", {{"longest_edge", 4096}}}, {key, 8192}}}});
    EXPECT_EQ(config.value<int32_t>("image.size.longest_edge", 0), 4096);
    EXPECT_DEATH(config.validate("image"), "Unsupported processor parameter");
    EXPECT_DEATH(config.validate(), "Unsupported processor parameter");
  }
}

TEST(ProcessorConfigTest, ReadsBooleansNumbersStringsAndArrays) {
  const ProcessorConfig config(nlohmann::json{{"enabled", true},
                                              {"fps", 2.5},
                                              {"mode", "bilinear"},
                                              {"mean", {0.25, 0.5, 0.75}}});
  EXPECT_TRUE(config.value<bool>("enabled", false));
  EXPECT_DOUBLE_EQ(config.value<double>("fps", 1.0), 2.5);
  EXPECT_EQ(config.value<std::string>("mode", "bicubic"), "bilinear");
  EXPECT_EQ(config.value<std::vector<double>>("mean", {}),
            (std::vector<double>{0.25, 0.5, 0.75}));
  config.validate();
}

TEST(ProcessorConfigTest, RejectsImplicitScalarConversionsAndIntegerOverflow) {
  const ProcessorConfig config(
      nlohmann::json{{"fraction", 2.5},
                     {"text", "8"},
                     {"number", 1},
                     {"large", 2147483648ULL},
                     {"unsigned_max", 18446744073709551615ULL},
                     {"negative", -1}});
  EXPECT_DEATH(config.value<int32_t>("fraction", 0), "must be an integer");
  EXPECT_DEATH(config.value<int32_t>("text", 0), "must be an integer");
  EXPECT_DEATH(config.value<bool>("number", false), "must be a boolean");
  EXPECT_DEATH(config.value<int32_t>("large", 0), "out of range");
  EXPECT_DEATH(config.value<int64_t>("unsigned_max", 0), "out of range");
  EXPECT_DEATH(config.value<uint32_t>("negative", 0), "out of range");
}

TEST(ProcessorConfigTest, RejectsWrongArrayElementTypesAndUnknownObjects) {
  const ProcessorConfig config(
      nlohmann::json{{"mean", {0.5, "invalid"}}, {"extra", {}}});
  EXPECT_DEATH(config.value<std::vector<double>>("mean", {}),
               "must be a number");
  EXPECT_DEATH(config.validate(), "Unsupported processor parameter");
}

TEST(ProcessorConfigTest, MultipleConsumersReadTheSameEffectiveConfiguration) {
  const ProcessorConfig config(nlohmann::json{{"frame_stride", 8}});
  const ExampleProcessor first(ExampleArgs(), config);
  const ExampleProcessor second(ExampleArgs(), config);
  config.validate();
  EXPECT_EQ(first.stride(), second.stride());
}

}  // namespace
}  // namespace xllm
