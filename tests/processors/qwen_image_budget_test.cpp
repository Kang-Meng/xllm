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

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "core/framework/config/model_config.h"
#include "core/framework/config/multimodal_config.h"
#include "processors/qwen2_vl_image_processor.h"

namespace xllm {
namespace {

ModelArgs image_args(bool pixel_layout = false) {
  ModelArgs args;
  args.mm_image_patch_size(16)
      .mm_image_merge_size(2)
      .mm_image_temporal_patch_size(2)
      .mm_image_normalize_mean({0.5, 0.5, 0.5})
      .mm_image_normalize_std({0.5, 0.5, 0.5});
  if (pixel_layout) {
    args.mm_image_min_pixels(65536).mm_image_max_pixels(16777216);
  } else {
    args.mm_image_shortest_edge(65536).mm_image_longest_edge(16777216);
  }
  return args;
}

int64_t image_tokens(const Qwen2VLImageProcessor& processor,
                     int32_t height,
                     int32_t width) {
  std::vector<MMDataItem> items;
  const auto image = torch::zeros({3, height, width}, torch::kUInt8);
  EXPECT_TRUE(processor.process({image}, items));
  if (items.size() != 1) {
    ADD_FAILURE() << "Expected one image output";
    return -1;
  }
  const auto pixels = items[0].get<torch::Tensor>("pixel_values");
  const auto grids = items[0].get<torch::Tensor>("image_grid_thw");
  if (!pixels.has_value() || !grids.has_value()) {
    ADD_FAILURE() << "Expected pixel_values and image_grid_thw";
    return -1;
  }
  const auto grid = grids->reshape({3});
  const int64_t grid_t = grid[0].item<int64_t>();
  const int64_t grid_h = grid[1].item<int64_t>();
  const int64_t grid_w = grid[2].item<int64_t>();
  EXPECT_EQ(grid_t, 1);
  EXPECT_EQ(grid_h % 2, 0);
  EXPECT_EQ(grid_w % 2, 0);
  EXPECT_EQ(pixels->size(0), grid_t * grid_h * grid_w);
  EXPECT_EQ(pixels->size(1), 3 * 2 * 16 * 16);
  return grid_t * grid_h * grid_w / 4;
}

class QwenImageBudgetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ModelConfig& config = ModelConfig::get_instance();
    saved_model_impl_ = config.model_impl();
    saved_model_path_ = config.model();
    config.model_impl("python").model("/nonexistent/xllm-processor-test");
  }

  void TearDown() override {
    ModelConfig::get_instance()
        .model_impl(saved_model_impl_)
        .model(saved_model_path_);
  }

 private:
  std::string saved_model_impl_;
  std::string saved_model_path_;
};

TEST_F(QwenImageBudgetTest, UsesBothModelPixelBudgetLayouts) {
  for (bool pixel_layout : {false, true}) {
    const Qwen2VLImageProcessor processor(image_args(pixel_layout));
    EXPECT_EQ(image_tokens(processor, 512, 512), 256);
    EXPECT_EQ(image_tokens(processor, 32, 32), 64);
  }
}

TEST_F(QwenImageBudgetTest, ModelMaximumControlsActualGrid) {
  auto args = image_args(true);
  args.mm_image_max_pixels(262144);
  const Qwen2VLImageProcessor processor(args);
  EXPECT_EQ(image_tokens(processor, 1024, 1024), 256);
  EXPECT_EQ(image_tokens(processor, 32, 32), 64);
}

TEST_F(QwenImageBudgetTest, PixelLayoutTakesPrecedence) {
  auto args = image_args();
  args.mm_image_min_pixels(4096).mm_image_max_pixels(65536);
  const Qwen2VLImageProcessor processor(args);
  EXPECT_EQ(image_tokens(processor, 512, 512), 64);
  EXPECT_EQ(image_tokens(processor, 32, 32), 4);
}

TEST_F(QwenImageBudgetTest, RetainsModelTokenBudgetResizing) {
  auto args = image_args();
  args.mm_image_min_tokens(4).mm_image_max_tokens(16);
  const Qwen2VLImageProcessor processor(args);
  EXPECT_EQ(image_tokens(processor, 512, 512), 16);
  EXPECT_EQ(image_tokens(processor, 32, 32), 4);
}

TEST_F(QwenImageBudgetTest, RejectsUnsupportedAspectRatio) {
  const Qwen2VLImageProcessor processor(image_args());
  std::vector<MMDataItem> items;
  EXPECT_FALSE(
      processor.process({torch::zeros({3, 32, 6432}, torch::kUInt8)}, items));
}

TEST_F(QwenImageBudgetTest, ServiceOverridesChangeActualImageGrid) {
  google::FlagSaver flag_saver;
  google::SetCommandLineOption("config_json_file", "");
  for (bool pixel_layout : {false, true}) {
    google::SetCommandLineOption("mm_args",
                                 R"({"image":{"max_pixels":262144}})");
    MultimodalConfig config;
    config.initialize();
    const auto original = image_args(pixel_layout);
    const ProcessorConfig image_config(config.processor_args());
    const Qwen2VLImageProcessor processor(original, image_config);
    image_config.validate();
    EXPECT_EQ(image_tokens(processor, 1024, 1024), 256);
    EXPECT_EQ(image_tokens(processor, 32, 32), 64);
    EXPECT_EQ(original.mm_image_max_pixels(), pixel_layout ? 16777216 : 0);
  }
}

TEST_F(QwenImageBudgetTest, TightPixelBudgetKeepsNarrowDimensionsNonzero) {
  auto args = image_args(true);
  args.mm_image_min_pixels(1024).mm_image_max_pixels(1024);
  const Qwen2VLImageProcessor processor(args);
  EXPECT_EQ(image_tokens(processor, 32, 6400), 14);
}

TEST_F(QwenImageBudgetTest, ServicePixelBudgetPreservesNarrowImageAlignment) {
  google::FlagSaver flag_saver;
  google::SetCommandLineOption("config_json_file", "");
  google::SetCommandLineOption(
      "mm_args", R"({"image":{"min_pixels":1024,"max_pixels":1024}})");
  MultimodalConfig config;
  config.initialize();
  const ProcessorConfig image_config(config.processor_args());
  const Qwen2VLImageProcessor processor(image_args(), image_config);
  image_config.validate();
  EXPECT_EQ(image_tokens(processor, 32, 6400), 14);
}

TEST_F(QwenImageBudgetTest, SupportsSingleMinimumAndNestedSizeOverrides) {
  const ProcessorConfig min_config(
      nlohmann::json{{"image", {{"min_pixels", 4096}}}});
  const Qwen2VLImageProcessor small_min(image_args(), min_config);
  min_config.validate();
  EXPECT_EQ(image_tokens(small_min, 32, 32), 4);
  const ProcessorConfig size_config(
      nlohmann::json{{"image", {{"size", {{"longest_edge", 262144}}}}}});
  const Qwen2VLImageProcessor small_max(image_args(), size_config);
  size_config.validate();
  EXPECT_EQ(image_tokens(small_max, 1024, 1024), 256);
}

TEST_F(QwenImageBudgetTest, RejectsInvalidOverridesBeforeProcessing) {
  const auto args = image_args();
  EXPECT_DEATH(
      Qwen2VLImageProcessor(
          args,
          ProcessorConfig(nlohmann::json{{"image", {{"max_pixels", 1.5}}}})),
      "must be an integer");
  EXPECT_DEATH(
      Qwen2VLImageProcessor(
          args,
          ProcessorConfig(nlohmann::json{{"image", {{"min_pixels", 0}}}})),
      "must be positive");
  EXPECT_DEATH(
      Qwen2VLImageProcessor(
          args,
          ProcessorConfig(nlohmann::json{{"image", {{"max_pixels", 4096}}}})),
      "must not exceed max_pixels");
  EXPECT_DEATH(Qwen2VLImageProcessor(
                   args,
                   ProcessorConfig(nlohmann::json{
                       {"image", {{"min_pixels", 1}, {"max_pixels", 100}}}})),
               "one aligned image patch group");
  auto token_args = args;
  token_args.mm_image_min_tokens(4).mm_image_max_tokens(16);
  EXPECT_DEATH(
      Qwen2VLImageProcessor(
          token_args,
          ProcessorConfig(nlohmann::json{{"image", {{"max_pixels", 65536}}}})),
      "token-budget resizing");
}

TEST_F(QwenImageBudgetTest, NativePixelsPreservePatchOrderAcrossModelImpls) {
  constexpr int32_t kHeight = 56;
  constexpr int32_t kWidth = 84;
  constexpr int32_t kPatchSize = 14;
  constexpr int32_t kMergeSize = 2;
  constexpr int32_t kTemporalSize = 2;
  constexpr int32_t kGridHeight = kHeight / kPatchSize;
  constexpr int32_t kGridWidth = kWidth / kPatchSize;
  auto args = image_args(true);
  args.mm_image_patch_size(kPatchSize)
      .mm_image_min_pixels(784)
      .mm_image_max_pixels(65536);
  const auto image = torch::arange(3 * kHeight * kWidth, torch::kInt64)
                         .remainder(256)
                         .to(torch::kUInt8)
                         .reshape({3, kHeight, kWidth});

  for (const std::string model_impl : {"cpp", "python", "py"}) {
    ModelConfig::get_instance().model_impl(model_impl);
    const Qwen2VLImageProcessor processor(args);
    std::vector<MMDataItem> items;
    ASSERT_TRUE(processor.process({image, 255 - image}, items));
    ASSERT_EQ(items.size(), 2);
    for (size_t image_index = 0; image_index < items.size(); ++image_index) {
      const auto pixels = items[image_index].get<torch::Tensor>("pixel_values");
      const auto grid = items[image_index].get<torch::Tensor>("image_grid_thw");
      ASSERT_TRUE(pixels.has_value());
      ASSERT_TRUE(grid.has_value());
      EXPECT_TRUE(torch::equal(
          *grid, torch::tensor({{1, kGridHeight, kGridWidth}}, torch::kInt64)));
      ASSERT_EQ(
          pixels->sizes(),
          torch::IntArrayRef({kGridHeight * kGridWidth,
                              3 * kTemporalSize * kPatchSize * kPatchSize}));
      ASSERT_EQ(pixels->scalar_type(), torch::kFloat32);
      const auto actual = pixels->accessor<float, 2>();
      for (int32_t grid_row = 0; grid_row < kGridHeight; ++grid_row) {
        for (int32_t grid_col = 0; grid_col < kGridWidth; ++grid_col) {
          const int32_t patch_index =
              ((grid_row / kMergeSize) * (kGridWidth / kMergeSize) +
               grid_col / kMergeSize) *
                  kMergeSize * kMergeSize +
              (grid_row % kMergeSize) * kMergeSize + grid_col % kMergeSize;
          for (int32_t channel = 0; channel < 3; ++channel) {
            for (int32_t temporal = 0; temporal < kTemporalSize; ++temporal) {
              for (int32_t patch_row = 0; patch_row < kPatchSize; ++patch_row) {
                for (int32_t patch_col = 0; patch_col < kPatchSize;
                     ++patch_col) {
                  const int32_t source_index =
                      channel * kHeight * kWidth +
                      (grid_row * kPatchSize + patch_row) * kWidth +
                      grid_col * kPatchSize + patch_col;
                  const int32_t pixel = image_index == 0
                                            ? source_index % 256
                                            : 255 - source_index % 256;
                  const int32_t feature_index =
                      ((channel * kTemporalSize + temporal) * kPatchSize +
                       patch_row) *
                          kPatchSize +
                      patch_col;
                  const float expected = (pixel - 127.5f) / 127.5f;
                  EXPECT_FLOAT_EQ(actual[patch_index][feature_index], expected);
                }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace
}  // namespace xllm
