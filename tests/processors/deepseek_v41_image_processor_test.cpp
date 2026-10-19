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

#include "processors/deepseek_v41_image_processor.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "core/common/constants.h"
#include "core/framework/model/model_args.h"
#include "core/framework/multimodal/mm_data.h"

namespace xllm {
namespace {

// Default patch size when ModelArgs leaves it unset (see the .h default).
constexpr int64_t kPatch = 14;

// Reference normalization used by the processor: uint8 [0, 255] -> [-1, 1].
double normalize(double v) { return (v / 255.0 - 0.5) / 0.5; }

struct ImageOut {
  torch::Tensor pixel_values;
  torch::Tensor grid;
};

torch::Tensor filled(int64_t channels,
                     int64_t height,
                     int64_t width,
                     double value) {
  return torch::full(
      {channels, height, width}, value, torch::dtype(torch::kFloat32));
}

// Spatially varying pattern (value grows with y and x) so patch ordering is
// observable. Kept below 256 so the uint8 resample is lossless.
torch::Tensor ramp(int64_t height, int64_t width) {
  auto img = torch::empty({3, height, width}, torch::dtype(torch::kFloat32));
  for (int64_t c = 0; c < 3; ++c) {
    for (int64_t y = 0; y < height; ++y) {
      for (int64_t x = 0; x < width; ++x) {
        img[c][y][x] = static_cast<double>((y * 2 + x) % 256);
      }
    }
  }
  return img;
}

ImageOut process_one(const DeepseekV41ImageProcessor& processor,
                     const torch::Tensor& image) {
  const std::vector<torch::Tensor> images{image};
  std::vector<MMDataItem> items;
  if (!processor.process(images, items) || items.size() != 1u) {
    return ImageOut{};
  }
  return ImageOut{
      items[0].get<torch::Tensor>("pixel_values").value(),
      items[0].get<torch::Tensor>("image_grid_thw").value(),
  };
}

int64_t n_vit_h(const ImageOut& out) { return out.grid[0][0].item<int64_t>(); }
int64_t n_vit_w(const ImageOut& out) { return out.grid[0][1].item<int64_t>(); }

// Same LLM-grid token count as the processor's internal grid_tokens().
int64_t image_block_tokens(const ImageOut& out) {
  constexpr int64_t kDownsample = 3;
  const int64_t n_llm_h = (n_vit_h(out) + kDownsample - 1) / kDownsample;
  const int64_t n_llm_w = (n_vit_w(out) + kDownsample - 1) / kDownsample;
  int64_t tokens = n_llm_h * (n_llm_w + 1) + 2;
  if (n_llm_h % 2 == 1) {
    tokens += n_llm_w + 1;
  }
  tokens += ((((n_llm_h + 1) / 2) * (n_llm_w + 1)) % 2) * 2;
  return tokens;
}

TEST(DeepseekV41ImageProcessorTest, ProcessRejectsNonThreeDimensionalInput) {
  ModelArgs args;
  const DeepseekV41ImageProcessor processor(args);
  std::vector<MMDataItem> items;
  EXPECT_FALSE(processor.process({torch::zeros({8, 8})}, items));
  EXPECT_FALSE(processor.process({torch::zeros({1, 3, 8, 8})}, items));
}

TEST(DeepseekV41ImageProcessorTest, ProcessRejectsWrongChannelCount) {
  ModelArgs args;
  const DeepseekV41ImageProcessor processor(args);
  std::vector<MMDataItem> items;
  // The patch-embed weights are [hidden, patch, patch, 3]; a 1- or 4-channel
  // image must be rejected here instead of failing at the matmul downstream.
  EXPECT_FALSE(processor.process({torch::zeros({1, 8, 8})}, items));
  EXPECT_FALSE(processor.process({torch::zeros({4, 8, 8})}, items));
  // The configured channel count is honored when the checkpoint sets it.
  ModelArgs grey_args;
  grey_args.mm_num_channels(1);
  const DeepseekV41ImageProcessor grey(grey_args);
  EXPECT_TRUE(grey.process({torch::zeros({1, 8, 8})}, items));
}

TEST(DeepseekV41ImageProcessorTest, ProcessRejectsEmptySpatialDimensions) {
  ModelArgs args;
  const DeepseekV41ImageProcessor processor(args);
  std::vector<MMDataItem> items;
  EXPECT_FALSE(processor.process({torch::zeros({3, 0, 0})}, items));
  EXPECT_FALSE(processor.process({torch::zeros({3, 0, 4})}, items));
  EXPECT_FALSE(processor.process({torch::zeros({3, 4, 0})}, items));
  EXPECT_TRUE(items.empty());
}

TEST(DeepseekV41ImageProcessorTest, EmptyInputProducesNoItems) {
  ModelArgs args;
  const DeepseekV41ImageProcessor processor(args);
  std::vector<MMDataItem> items;
  EXPECT_TRUE(processor.process({}, items));
  EXPECT_TRUE(items.empty());
}

TEST(DeepseekV41ImageProcessorTest, PatchAlignedSquareKeepsGridAndShape) {
  ModelArgs args;
  const DeepseekV41ImageProcessor processor(args);
  const ImageOut out = process_one(processor, filled(3, 28, 28, 200.0));
  ASSERT_TRUE(out.grid.defined());
  ASSERT_EQ(out.grid.size(0), 1);
  ASSERT_EQ(out.grid.size(1), 2);
  EXPECT_EQ(n_vit_h(out), 2);
  EXPECT_EQ(n_vit_w(out), 2);
  ASSERT_EQ(out.pixel_values.dim(), 4);
  EXPECT_EQ(out.pixel_values.size(0), n_vit_h(out) * n_vit_w(out));
  EXPECT_EQ(out.pixel_values.size(1), 3);
  EXPECT_EQ(out.pixel_values.size(2), kPatch);
  EXPECT_EQ(out.pixel_values.size(3), kPatch);
  EXPECT_EQ(out.pixel_values.scalar_type(), torch::kBFloat16);
}

TEST(DeepseekV41ImageProcessorTest, UpscalesToNextPatchMultiple) {
  ModelArgs args;
  const DeepseekV41ImageProcessor processor(args);
  // 20 -> ceil(20 / 14) * 14 = 28 -> 2x2 patches.
  const ImageOut out = process_one(processor, filled(3, 20, 20, 100.0));
  ASSERT_TRUE(out.grid.defined());
  EXPECT_EQ(n_vit_h(out), 2);
  EXPECT_EQ(n_vit_w(out), 2);
  EXPECT_EQ(out.pixel_values.size(0), 4);
}

TEST(DeepseekV41ImageProcessorTest, PadsWithGreyAndCentersContent) {
  ModelArgs args;
  const DeepseekV41ImageProcessor processor(args);
  // 40x14 -> padded to 42x14 with scale 1, so one grey row top and bottom.
  const ImageOut out = process_one(processor, filled(3, 40, 14, 255.0));
  ASSERT_TRUE(out.grid.defined());
  EXPECT_EQ(n_vit_h(out), 3);
  EXPECT_EQ(n_vit_w(out), 1);
  EXPECT_EQ(out.pixel_values.size(0), 3);

  const double grey = normalize(127.0);
  // First patch: top row is padding, the row below is the (white) content.
  for (int64_t c = 0; c < 3; ++c) {
    EXPECT_NEAR(out.pixel_values[0][c][0][0].item<float>(), grey, 2e-3);
    EXPECT_NEAR(out.pixel_values[0][c][1][0].item<float>(), 1.0, 2e-3);
  }
  // Last patch: final row is padding again.
  for (int64_t c = 0; c < 3; ++c) {
    EXPECT_NEAR(
        out.pixel_values[2][c][kPatch - 1][0].item<float>(), grey, 2e-3);
  }
}

TEST(DeepseekV41ImageProcessorTest, NormalizesGreyToMinusOne) {
  ModelArgs args;
  const DeepseekV41ImageProcessor processor(args);
  const ImageOut out = process_one(processor, filled(3, 28, 28, 127.0));
  ASSERT_TRUE(out.pixel_values.defined());
  const float min_v = out.pixel_values.min().item<float>();
  const float max_v = out.pixel_values.max().item<float>();
  EXPECT_NEAR(static_cast<double>(min_v), normalize(127.0), 2e-3);
  EXPECT_NEAR(static_cast<double>(max_v), normalize(127.0), 2e-3);
}

TEST(DeepseekV41ImageProcessorTest, AcceptsUint8InputDirectly) {
  // Production decode hands the processor uint8 [C, H, W]; the pipeline must
  // consume it without an intermediate float round-trip.
  ModelArgs args;
  const DeepseekV41ImageProcessor processor(args);
  auto grey = torch::full({3, 28, 28}, 127, torch::dtype(torch::kUInt8));
  const ImageOut out = process_one(processor, grey);
  ASSERT_TRUE(out.pixel_values.defined());
  EXPECT_EQ(out.pixel_values.dtype(), torch::kBFloat16);
  EXPECT_NEAR(out.pixel_values.min().item<float>(), normalize(127.0), 2e-3);
  EXPECT_NEAR(out.pixel_values.max().item<float>(), normalize(127.0), 2e-3);
}

TEST(DeepseekV41ImageProcessorTest, PatchOrderIsRowMajorOverGrid) {
  ModelArgs args;
  const DeepseekV41ImageProcessor processor(args);
  const ImageOut out = process_one(processor, ramp(28, 28));
  ASSERT_TRUE(out.pixel_values.defined());
  ASSERT_EQ(out.pixel_values.size(0), 4);  // 2x2 grid
  // Patch index = ih * n_vit_w + iw; each patch's [0][0] is its top-left
  // canvas pixel.
  EXPECT_NEAR(out.pixel_values[0][0][0][0].item<float>(), normalize(0.0), 2e-3);
  EXPECT_NEAR(
      out.pixel_values[1][0][0][0].item<float>(), normalize(14.0), 2e-3);
  EXPECT_NEAR(
      out.pixel_values[2][0][0][0].item<float>(), normalize(28.0), 2e-3);
  // Bottom-right pixel of the last patch.
  EXPECT_NEAR(out.pixel_values[3][0][kPatch - 1][kPatch - 1].item<float>(),
              normalize(static_cast<double>((27 * 2 + 27) % 256)),
              2e-3);
}

TEST(DeepseekV41ImageProcessorTest, MinPixelsUpscalesSmallImage) {
  ModelArgs limited;
  limited.mm_image_min_pixels(28 * 28);
  const DeepseekV41ImageProcessor upscaling(limited);
  const ImageOut out = process_one(upscaling, filled(3, 14, 14, 128.0));
  ASSERT_TRUE(out.grid.defined());
  EXPECT_EQ(n_vit_h(out), 2);
  EXPECT_EQ(n_vit_w(out), 2);

  ModelArgs plain;
  const DeepseekV41ImageProcessor baseline(plain);
  const ImageOut base_out = process_one(baseline, filled(3, 14, 14, 128.0));
  ASSERT_TRUE(base_out.grid.defined());
  EXPECT_EQ(n_vit_h(base_out), 1);
  EXPECT_EQ(n_vit_w(base_out), 1);
}

TEST(DeepseekV41ImageProcessorTest, MaxTokensShrinksGrid) {
  ModelArgs limited;
  limited.mm_image_max_tokens(32);
  const DeepseekV41ImageProcessor capped(limited);
  const ImageOut out = process_one(capped, filled(3, 224, 224, 200.0));
  ASSERT_TRUE(out.grid.defined());
  const int64_t capped_patches = n_vit_h(out) * n_vit_w(out);

  ModelArgs plain;
  const DeepseekV41ImageProcessor baseline(plain);
  const ImageOut base_out = process_one(baseline, filled(3, 224, 224, 200.0));
  ASSERT_TRUE(base_out.grid.defined());
  EXPECT_LT(capped_patches, n_vit_h(base_out) * n_vit_w(base_out));
  // Patch rows always match the reported grid.
  EXPECT_EQ(out.pixel_values.size(0), capped_patches);
  EXPECT_EQ(out.pixel_values.size(2), kPatch);
}

TEST(DeepseekV41ImageProcessorTest, RespectsConfiguredPatchAndDownsample) {
  ModelArgs args;
  args.mm_image_patch_size(16);
  args.mm_downsample_ratio(2);
  const DeepseekV41ImageProcessor processor(args);
  const ImageOut out = process_one(processor, filled(3, 32, 32, 150.0));
  ASSERT_TRUE(out.grid.defined());
  EXPECT_EQ(n_vit_h(out), 2);
  EXPECT_EQ(n_vit_w(out), 2);
  EXPECT_EQ(out.pixel_values.size(2), 16);
  EXPECT_EQ(out.pixel_values.size(3), 16);
}

TEST(DeepseekV41ImageProcessorTest, FallsBackToGeneralMmPatchSize) {
  ModelArgs args;
  args.mm_patch_size(8);
  const DeepseekV41ImageProcessor processor(args);
  const ImageOut out = process_one(processor, filled(3, 16, 16, 150.0));
  ASSERT_TRUE(out.grid.defined());
  EXPECT_EQ(out.pixel_values.size(2), 8);
  EXPECT_EQ(out.pixel_values.size(3), 8);
  EXPECT_EQ(n_vit_h(out), 2);
  EXPECT_EQ(n_vit_w(out), 2);
}

TEST(DeepseekV41ImageProcessorTest, ProcessHandlesMultipleImagesAndClears) {
  ModelArgs args;
  const DeepseekV41ImageProcessor processor(args);
  const std::vector<torch::Tensor> images{filled(3, 28, 28, 200.0),
                                          filled(3, 14, 14, 100.0)};
  std::vector<MMDataItem> items;
  items.emplace_back(MMType::IMAGE, MMDict{});  // must be cleared
  ASSERT_TRUE(processor.process(images, items));
  ASSERT_EQ(items.size(), 2u);
  for (const auto& item : items) {
    EXPECT_EQ(item.type(), MMType::IMAGE);
    EXPECT_TRUE(item.get<torch::Tensor>("pixel_values").has_value());
    EXPECT_TRUE(item.get<torch::Tensor>("image_grid_thw").has_value());
  }
  EXPECT_EQ(items[0]
                .get<torch::Tensor>("image_grid_thw")
                .value()[0][0]
                .item<int64_t>(),
            2);
  EXPECT_EQ(items[1]
                .get<torch::Tensor>("image_grid_thw")
                .value()[0][0]
                .item<int64_t>(),
            1);
}

TEST(DeepseekV41ImageProcessorTest, RejectsMaxTokensBelowMinimum) {
  // A 1x1 image block is already 6 LLM tokens, and the prompt processor may
  // reserve kDeepseekV41CompressPadTo - 1 pads before it. Budgets below 9
  // cannot be satisfied by any grid, so the constructor must reject them.
  for (const int bad : {1, 4, 5, 8}) {
    ModelArgs args;
    args.mm_image_max_tokens(bad);
    EXPECT_DEATH(
        { DeepseekV41ImageProcessor processor(args); }, "max_image_tokens");
  }
}

TEST(DeepseekV41ImageProcessorTest, ExtremeAspectRatiosKeepValidGrid) {
  // Wide/tall inputs are the cases where a fractional min-pixel scale or a
  // sub-patch canvas can produce an empty or mis-sized grid. Pin the grid to
  // the patch-aligned ceil dimensions and verify every emitted patch is
  // present.
  struct Case {
    int64_t h;
    int64_t w;
    int64_t expect_h;
    int64_t expect_w;
  };
  const Case cases[] = {
      {20, 700, 2, 50}, {700, 20, 50, 2}, {4, 2000, 1, 143}, {2000, 4, 143, 1}};
  ModelArgs args;
  const DeepseekV41ImageProcessor processor(args);
  for (const Case& c : cases) {
    const ImageOut out = process_one(processor, filled(3, c.h, c.w, 128.0));
    ASSERT_TRUE(out.grid.defined()) << c.h << "x" << c.w;
    EXPECT_EQ(n_vit_h(out), c.expect_h) << c.h << "x" << c.w;
    EXPECT_EQ(n_vit_w(out), c.expect_w) << c.h << "x" << c.w;
    EXPECT_EQ(out.pixel_values.size(0), c.expect_h * c.expect_w);
    EXPECT_EQ(out.pixel_values.size(2), kPatch);
    EXPECT_EQ(out.pixel_values.size(3), kPatch);
    EXPECT_TRUE(torch::isfinite(out.pixel_values).all().item<bool>());
  }
}

TEST(DeepseekV41ImageProcessorTest, MinimumTokenBudgetStaysWithinLimit) {
  // 9 is the smallest accepted budget: the 6-token minimum block plus the 3
  // splice pads. Near-square and extreme aspect ratios must all stay within it
  // even when solve_resize_ratio degrades to the minimum canvas.
  ModelArgs args;
  args.mm_image_max_tokens(9);
  const DeepseekV41ImageProcessor processor(args);
  const int64_t pad_slots = kDeepseekV41CompressPadTo - 1;
  for (const auto& [h, w] : std::vector<std::pair<int64_t, int64_t>>{
           {224, 224}, {20, 700}, {700, 20}, {4, 2000}, {2000, 4}}) {
    const ImageOut out = process_one(processor, filled(3, h, w, 128.0));
    ASSERT_TRUE(out.grid.defined()) << h << "x" << w;
    EXPECT_GE(n_vit_h(out), 1) << h << "x" << w;
    EXPECT_GE(n_vit_w(out), 1) << h << "x" << w;
    EXPECT_LE(image_block_tokens(out) + pad_slots, 9) << h << "x" << w;
    EXPECT_TRUE(torch::isfinite(out.pixel_values).all().item<bool>())
        << h << "x" << w;
  }
}

}  // namespace
}  // namespace xllm
