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

#include <cstdint>
#include <limits>
#include <vector>

#include "core/framework/config/multimodal_config.h"
#include "processors/qwen2_vl_image_processor.h"
#include "processors/qwen3_omni_image_processor.h"
#include "processors/qwen3_vl_video_processor.h"

namespace xllm {
namespace {

ModelArgs image_args(bool pixel_layout = false) {
  ModelArgs args;
  args.mm_image_patch_size(/*value=*/16)
      .mm_image_merge_size(/*value=*/2)
      .mm_image_temporal_patch_size(/*value=*/2)
      .mm_image_normalize_mean(/*value=*/{0.5, 0.5, 0.5})
      .mm_image_normalize_std(/*value=*/{0.5, 0.5, 0.5});
  if (pixel_layout) {
    args.mm_image_min_pixels(/*value=*/65536)
        .mm_image_max_pixels(/*value=*/16777216);
  } else {
    args.mm_image_shortest_edge(/*value=*/65536)
        .mm_image_longest_edge(/*value=*/16777216);
  }
  return args;
}

int64_t image_tokens(const Qwen2VLImageProcessor& processor,
                     int32_t height,
                     int32_t width,
                     int32_t patch_size = 16,
                     int32_t merge_size = 2) {
  std::vector<torch::Tensor> pixels;
  std::vector<torch::Tensor> grids;
  const auto image =
      torch::zeros(/*size=*/{3, height, width}, /*options=*/torch::kUInt8);
  EXPECT_TRUE(processor.process_image({image}, pixels, grids));
  if (grids.size() != 1 || pixels.size() != 1) {
    ADD_FAILURE() << "Expected one image output";
    return -1;
  }
  const auto grid = grids[0].reshape(/*shape=*/{3});
  const int64_t grid_t = grid[0].item<int64_t>();
  const int64_t grid_h = grid[1].item<int64_t>();
  const int64_t grid_w = grid[2].item<int64_t>();
  EXPECT_EQ(grid_t, 1);
  EXPECT_EQ(grid_h % merge_size, 0);
  EXPECT_EQ(grid_w % merge_size, 0);
  EXPECT_EQ(pixels[0].size(/*dim=*/0), grid_t * grid_h * grid_w);
  EXPECT_EQ(pixels[0].size(/*dim=*/1), 3 * 2 * patch_size * patch_size);
  const int64_t tokens = grid_t * grid_h * grid_w / (merge_size * merge_size);
  EXPECT_EQ(tokens * patch_size * patch_size * merge_size * merge_size,
            grid_h * patch_size * grid_w * patch_size);
  return tokens;
}

class QwenImageBudgetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    saved_tokens_ = MultimodalConfig::get_instance().image_max_tokens_num();
    MultimodalConfig::get_instance().image_max_tokens_num(/*value=*/0);
  }
  void TearDown() override {
    MultimodalConfig::get_instance().image_max_tokens_num(saved_tokens_);
  }

 private:
  int32_t saved_tokens_ = 0;
};

TEST_F(QwenImageBudgetTest, ZeroPreservesBothModelLayouts) {
  for (bool pixel_layout : {false, true}) {
    const auto args = image_args(pixel_layout);
    const Qwen35ImageProcessor implicit(args);
    const Qwen35ImageProcessor zero(args, /*max_image_tokens=*/0);
    const Qwen2VLImageProcessor original(args);
    EXPECT_EQ(image_tokens(implicit, /*height=*/512, /*width=*/512), 256);
    EXPECT_EQ(image_tokens(zero, /*height=*/512, /*width=*/512),
              image_tokens(original, /*height=*/512, /*width=*/512));
    EXPECT_EQ(image_tokens(zero, /*height=*/32, /*width=*/32), 64);
  }
}

TEST_F(QwenImageBudgetTest, OverrideChangesActualGridInBothLayouts) {
  for (bool pixel_layout : {false, true}) {
    const auto args = image_args(pixel_layout);
    const Qwen35ImageProcessor small(args, /*max_image_tokens=*/1024);
    EXPECT_EQ(image_tokens(small, /*height=*/1536, /*width=*/1536), 1024);
    EXPECT_EQ(image_tokens(small, /*height=*/32, /*width=*/32), 64);
    const Qwen35ImageProcessor boundary(args, /*max_image_tokens=*/64);
    EXPECT_EQ(image_tokens(boundary, /*height=*/1536, /*width=*/1536), 64);
    EXPECT_EQ(image_tokens(boundary, /*height=*/32, /*width=*/64), 64);
    EXPECT_EQ(image_tokens(boundary, /*height=*/32, /*width=*/6400), 64);
    const int64_t rectangular_tokens =
        image_tokens(small, /*height=*/1024, /*width=*/2048);
    EXPECT_GT(rectangular_tokens, 64);
    EXPECT_LE(rectangular_tokens, 1024);
    auto lower_max_args = args;
    if (pixel_layout) {
      lower_max_args.mm_image_max_pixels(/*value=*/262144);
    } else {
      lower_max_args.mm_image_longest_edge(/*value=*/262144);
    }
    const Qwen35ImageProcessor lower_max(lower_max_args,
                                         /*max_image_tokens=*/0);
    const Qwen35ImageProcessor raised(lower_max_args,
                                      /*max_image_tokens=*/1024);
    EXPECT_EQ(image_tokens(lower_max, /*height=*/1536, /*width=*/1536), 256);
    EXPECT_EQ(image_tokens(raised, /*height=*/1536, /*width=*/1536), 1024);
  }
}

TEST_F(QwenImageBudgetTest, OverrideCanExceedOriginalMaximum) {
  const auto args = image_args();
  const Qwen35ImageProcessor original(args, /*max_image_tokens=*/0);
  const Qwen35ImageProcessor larger(args, /*max_image_tokens=*/32768);
  EXPECT_EQ(image_tokens(original, /*height=*/4608, /*width=*/4608), 16384);
  EXPECT_EQ(image_tokens(larger, /*height=*/4608, /*width=*/4608), 20736);
}

TEST_F(QwenImageBudgetTest, TightAlignmentIsScopedToExplicitOverride) {
  auto args = image_args(/*pixel_layout=*/true);
  args.mm_image_max_pixels(/*value=*/65536);
  const Qwen2VLImageProcessor shared(args);
  const Qwen35ImageProcessor zero(args, /*max_image_tokens=*/0);
  const Qwen35ImageProcessor overridden(args, /*max_image_tokens=*/64);
  EXPECT_EQ(image_tokens(shared, /*height=*/32, /*width=*/64), 72);
  EXPECT_EQ(image_tokens(zero, /*height=*/32, /*width=*/64), 72);
  EXPECT_EQ(image_tokens(overridden, /*height=*/32, /*width=*/64), 64);
}

TEST_F(QwenImageBudgetTest, UsesConfiguredGeometryAndMinimum) {
  auto args = image_args(/*pixel_layout=*/true);
  args.mm_image_patch_size(/*value=*/8)
      .mm_image_merge_size(/*value=*/2)
      .mm_image_min_pixels(/*value=*/4096);
  const Qwen35ImageProcessor processor(args, /*max_image_tokens=*/16);
  EXPECT_EQ(
      image_tokens(processor, /*height=*/256, /*width=*/256, /*patch_size=*/8),
      16);
}

TEST_F(QwenImageBudgetTest, RejectsInvalidBudgetsBeforeProcessing) {
  const auto args = image_args();
  EXPECT_DEATH(Qwen35ImageProcessor(args, /*max_image_tokens=*/-1),
               "must be nonnegative");
  EXPECT_DEATH(Qwen35ImageProcessor(args, /*max_image_tokens=*/63),
               "below the model minimum");
  EXPECT_DEATH(
      Qwen35ImageProcessor(
          args, /*max_image_tokens=*/std::numeric_limits<int32_t>::max()),
      "exceeds int32 range");
  EXPECT_DEATH(Qwen35ImageProcessor(args, /*max_image_tokens=*/2097152),
               "exceeds int32 range");
  EXPECT_NO_FATAL_FAILURE(
      Qwen35ImageProcessor(args, /*max_image_tokens=*/2097151));
  auto huge_geometry = args;
  huge_geometry
      .mm_image_patch_size(/*value=*/std::numeric_limits<int32_t>::max())
      .mm_image_merge_size(/*value=*/std::numeric_limits<int32_t>::max());
  EXPECT_DEATH(Qwen35ImageProcessor(huge_geometry, /*max_image_tokens=*/64),
               "exceeds int32 range");
}

TEST_F(QwenImageBudgetTest, PixelLayoutTakesPrecedence) {
  auto args = image_args();
  args.mm_image_min_pixels(/*value=*/4096)
      .mm_image_max_pixels(/*value=*/1048576);
  const Qwen35ImageProcessor processor(args, /*max_image_tokens=*/4);
  EXPECT_EQ(image_tokens(processor, /*height=*/512, /*width=*/512), 4);
}

TEST_F(QwenImageBudgetTest, RetainsProcessorDefaultsWithoutModelBounds) {
  auto args = image_args();
  args.mm_image_shortest_edge(/*value=*/0).mm_image_longest_edge(/*value=*/0);
  const Qwen35ImageProcessor processor(args, /*max_image_tokens=*/4);
  EXPECT_EQ(image_tokens(processor, /*height=*/512, /*width=*/512), 4);
  EXPECT_DEATH(Qwen35ImageProcessor(args, /*max_image_tokens=*/3),
               "below the model minimum");
}

TEST_F(QwenImageBudgetTest, OnlyQwen35ConsumesNewOverride) {
  const auto args = image_args();
  MultimodalConfig::get_instance().image_max_tokens_num(/*value=*/64);
  const Qwen35ImageProcessor qwen35(args);
  const Qwen2VLImageProcessor shared(args);
  EXPECT_EQ(image_tokens(qwen35, /*height=*/512, /*width=*/512), 64);
  EXPECT_EQ(image_tokens(shared, /*height=*/512, /*width=*/512), 256);
  MultimodalConfig::get_instance().image_max_tokens_num(/*value=*/-1);
  const Qwen2VLImageProcessor shared_negative(args);
  EXPECT_EQ(image_tokens(shared_negative, /*height=*/512, /*width=*/512), 256);
}

TEST_F(QwenImageBudgetTest, OmniRetainsItsMinimumPolicy) {
  const auto args = image_args();
  const Qwen3OmniImageProcessor original(args);
  EXPECT_EQ(image_tokens(original, /*height=*/32, /*width=*/32), 64);
  MultimodalConfig::get_instance().image_max_tokens_num(/*value=*/16);
  const Qwen3OmniImageProcessor overridden(args);
  EXPECT_EQ(image_tokens(overridden, /*height=*/32, /*width=*/32), 4);
  EXPECT_EQ(image_tokens(overridden, /*height=*/512, /*width=*/512), 16);
  MultimodalConfig::get_instance().image_max_tokens_num(/*value=*/3);
  EXPECT_DEATH(Qwen3OmniImageProcessor invalid(args), "must be at least 4");
}

TEST_F(QwenImageBudgetTest, TokenResizeRemainsIndependent) {
  auto args = image_args();
  args.mm_image_min_tokens(/*value=*/4).mm_image_max_tokens(/*value=*/16);
  MultimodalConfig::get_instance().image_max_tokens_num(/*value=*/64);
  const Qwen2VLImageProcessor shared(args);
  const Qwen35ImageProcessor zero(args, /*max_image_tokens=*/0);
  EXPECT_EQ(image_tokens(shared, /*height=*/512, /*width=*/512), 16);
  EXPECT_EQ(image_tokens(zero, /*height=*/512, /*width=*/512), 16);
  EXPECT_DEATH(Qwen35ImageProcessor(args, /*max_image_tokens=*/64),
               "requires pixel-budget resizing");
}

TEST_F(QwenImageBudgetTest, VideoIgnoresImageOverride) {
  const auto args = image_args();
  const auto video =
      torch::zeros(/*size=*/{4, 3, 64, 64}, /*options=*/torch::kUInt8);
  VideoMetadata metadata;
  metadata.total_num_frames = 4;
  metadata.fps = 2.0;
  MMDataItem original(/*type=*/MMType::VIDEO);
  MMDataItem overridden(/*type=*/MMType::VIDEO);
  ASSERT_TRUE(Qwen3VLVideoProcessor(args).process(video, metadata, original));
  MultimodalConfig::get_instance().image_max_tokens_num(/*value=*/1);
  ASSERT_TRUE(Qwen3VLVideoProcessor(args).process(video, metadata, overridden));
  for (const auto& [key, value] : original.data()) {
    EXPECT_TRUE(
        torch::equal(std::get<torch::Tensor>(value),
                     std::get<torch::Tensor>(overridden.data().at(key))));
  }
}

}  // namespace
}  // namespace xllm
