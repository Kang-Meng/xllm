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
#include <vector>

#include "processors/glm4v_image_processor.h"
#include "processors/kimi25_image_processor.h"
#include "processors/minicpmv_image_processor.h"
#include "processors/minicpmv_prompt_processor.h"
#include "processors/mistral3_image_processor.h"
#include "processors/multimodal_processor.h"
#include "processors/qwen2_vl_image_processor.h"

namespace xllm {
namespace {

class VideoConfigPromptProcessor final {
 public:
  VideoConfigPromptProcessor(const ModelArgs&, const ProcessorConfig& config)
      : stride_(config.value<int32_t>("video.frame_stride", 4)) {}

  int32_t stride() const { return stride_; }

 private:
  int32_t stride_;
};

TEST(ImageProcessorConfigTest, PromptCanConsumeVideoConfiguration) {
  ModelArgs args;
  args.mm_patch_size(14).mm_spatial_merge_size(2).mm_image_size(112);
  const ProcessorConfig config(nlohmann::json{
      {"image", {{"longest_edge", 56}}}, {"video", {{"frame_stride", 8}}}});
  const auto image = create_processor<Mistral3ImageProcessor>(args, config);
  config.validate("image");
  EXPECT_DEATH(config.validate("video"), "video.frame_stride");
  const auto prompt =
      create_processor<VideoConfigPromptProcessor>(args, config);
  config.validate("video");
  EXPECT_EQ(prompt->stride(), 8);
  std::vector<MMDataItem> items;
  ASSERT_TRUE(
      image->process({torch::zeros({3, 112, 112}, torch::kUInt8)}, items));
  const auto grid = items[0].get<torch::Tensor>("image_grid_thw");
  ASSERT_TRUE(grid.has_value());
  EXPECT_TRUE(torch::equal(*grid, torch::tensor({{1, 4, 4}}, torch::kInt64)));
}

TEST(ImageProcessorConfigTest, LegacyProcessorDoesNotConsumeOverrides) {
  const ModelArgs args{};
  const ProcessorConfig defaults;
  const auto processor = create_processor<ImageNoneProcessor>(args, defaults);
  defaults.validate("image");
  const ProcessorConfig config(
      nlohmann::json{{"image", {{"max_pixels", 4096}}}});
  const auto configured = create_processor<ImageNoneProcessor>(args, config);
  EXPECT_DEATH(config.validate("image"), "image.max_pixels");
}

TEST(ImageProcessorConfigTest, MistralUsesItsOwnEdgeParameter) {
  ModelArgs args;
  args.mm_patch_size(14).mm_spatial_merge_size(2).mm_image_size(112);
  const ProcessorConfig config(
      nlohmann::json{{"image", {{"longest_edge", 56}}}});
  const auto processor = create_processor<Mistral3ImageProcessor>(args, config);
  config.validate();
  std::vector<MMDataItem> items;
  ASSERT_TRUE(
      processor->process({torch::zeros({3, 112, 112}, torch::kUInt8)}, items));
  ASSERT_EQ(items.size(), 1);
  const auto grid = items[0].get<torch::Tensor>("image_grid_thw");
  ASSERT_TRUE(grid.has_value());
  EXPECT_TRUE(torch::equal(*grid, torch::tensor({{1, 4, 4}}, torch::kInt64)));
  EXPECT_EQ(args.mm_image_size(), 112);
}

TEST(ImageProcessorConfigTest, KimiUsesItsOwnPatchBudgetParameter) {
  ModelArgs args;
  const ProcessorConfig config(
      nlohmann::json{{"image", {{"in_patch_limit", 16}}}});
  const auto processor = create_processor<KimiK25ImageProcessor>(args, config);
  config.validate();
  std::vector<MMDataItem> items;
  ASSERT_TRUE(
      processor->process({torch::zeros({3, 112, 112}, torch::kUInt8)}, items));
  ASSERT_EQ(items.size(), 1);
  const auto grid = items[0].get<torch::Tensor>("image_grid_thw");
  ASSERT_TRUE(grid.has_value());
  EXPECT_TRUE(torch::equal(*grid, torch::tensor({{1, 4, 4}}, torch::kInt64)));
}

TEST(ImageProcessorConfigTest, UnknownParametersAreRejectedByActualConsumer) {
  ModelArgs args;
  args.mm_patch_size(14).mm_spatial_merge_size(2).mm_image_size(112);
  const ProcessorConfig config(
      nlohmann::json{{"image", {{"max_pixels", 4096}}}}, "mm_args");
  const auto processor = create_processor<Mistral3ImageProcessor>(args, config);
  EXPECT_DEATH(config.validate(),
               "Unsupported processor parameter.*max_pixels");
}

TEST(ImageProcessorConfigTest, VideoOverridesDoNotEnableImagePixelValidation) {
  ModelArgs args;
  args.mm_image_min_tokens(4).mm_image_max_tokens(16);
  const ProcessorConfig config(
      nlohmann::json{{"video", {{"frame_stride", 8}}}});
  const auto processor = create_processor<Qwen2VLImageProcessor>(args, config);
  config.validate("image");
  EXPECT_DEATH(config.validate("video"), "video.frame_stride");
}

TEST(ImageProcessorConfigTest, GlmPreservesItsTemporalAreaSemantics) {
  ModelArgs args;
  args.mm_image_patch_size(14)
      .mm_image_merge_size(2)
      .mm_image_temporal_patch_size(2)
      .mm_image_min_pixels(3136)
      .mm_image_max_pixels(12845056)
      .mm_image_normalize_mean({0.5, 0.5, 0.5})
      .mm_image_normalize_std({0.5, 0.5, 0.5});
  const ProcessorConfig config(
      nlohmann::json{{"image", {{"max_pixels", 12544}}}});
  const auto processor = create_processor<Glm4VImageProcessor>(args, config);
  const auto qwen_processor =
      create_processor<Qwen2VLImageProcessor>(args, config);
  config.validate();
  std::vector<MMDataItem> items;
  ASSERT_TRUE(
      processor->process({torch::zeros({3, 112, 112}, torch::kUInt8)}, items));
  const auto grid = items[0].get<torch::Tensor>("image_grid_thw");
  ASSERT_TRUE(grid.has_value());
  EXPECT_TRUE(torch::equal(*grid, torch::tensor({{1, 4, 4}}, torch::kInt64)));
  std::vector<MMDataItem> qwen_items;
  ASSERT_TRUE(qwen_processor->process(
      {torch::zeros({3, 112, 112}, torch::kUInt8)}, qwen_items));
  const auto qwen_grid = qwen_items[0].get<torch::Tensor>("image_grid_thw");
  ASSERT_TRUE(qwen_grid.has_value());
  EXPECT_TRUE(
      torch::equal(*qwen_grid, torch::tensor({{1, 8, 8}}, torch::kInt64)));
}

TEST(ImageProcessorConfigTest, PixelBudgetsPreserveProcessorMinimumGridSizes) {
  ModelArgs args;
  args.mm_image_patch_size(14)
      .mm_image_merge_size(2)
      .mm_image_temporal_patch_size(2)
      .mm_image_normalize_mean({0.5, 0.5, 0.5})
      .mm_image_normalize_std({0.5, 0.5, 0.5});
  const ProcessorConfig config(
      nlohmann::json{{"image", {{"min_pixels", 784}, {"max_pixels", 784}}}});
  const auto processor = create_processor<Qwen2VLImageProcessor>(args, config);
  config.validate();
  std::vector<MMDataItem> items;
  ASSERT_TRUE(
      processor->process({torch::zeros({3, 28, 28}, torch::kUInt8)}, items));
  const auto grid = items[0].get<torch::Tensor>("image_grid_thw");
  ASSERT_TRUE(grid.has_value());
  EXPECT_TRUE(torch::equal(*grid, torch::tensor({{1, 2, 2}}, torch::kInt64)));
  EXPECT_DEATH(create_processor<Glm4VImageProcessor>(args, config),
               "one aligned image patch group");
}

TEST(ImageProcessorConfigTest, MiniCpmSlicesAndPromptShareConfiguration) {
  ModelArgs args;
  args.vision_max_slice_nums(9)
      .mm_scale_resolution(112)
      .mm_patch_size(14)
      .mm_slice_mode(true)
      .mm_image_feature_size(4)
      .mm_image_normalize_mean({0.5, 0.5, 0.5})
      .mm_image_normalize_std({0.5, 0.5, 0.5});
  const auto image = torch::zeros({3, 112, 112}, torch::kUInt8);
  for (int32_t max_slices : {1, 4}) {
    const ProcessorConfig config(nlohmann::json{
        {"image", {{"max_slice_nums", max_slices}, {"scale_resolution", 56}}}});
    const auto processor =
        create_processor<MiniCPMVImageProcessor>(args, config);
    const auto prompt_processor =
        create_processor<npu::model::MiniCPMPromptProcessor>(args, config);
    config.validate();
    std::vector<MMDataItem> items;
    ASSERT_TRUE(processor->process({image}, items));
    ASSERT_EQ(items.size(), 1);
    const auto pixels =
        items[0].get<std::vector<torch::Tensor>>("pixel_values");
    ASSERT_TRUE(pixels.has_value());
    EXPECT_EQ(pixels->size(), max_slices == 1 ? 1 : 5);
    MMData data(MMType::IMAGE, items);
    std::string prompt = "(<image>./</image>)";
    ASSERT_TRUE(prompt_processor->process(prompt, data));
    size_t unknown_count = 0;
    size_t offset = 0;
    while ((offset = prompt.find("<unk>", offset)) != std::string::npos) {
      ++unknown_count;
      offset += 5;
    }
    EXPECT_EQ(unknown_count, pixels->size() * 4);
  }
}

}  // namespace
}  // namespace xllm
