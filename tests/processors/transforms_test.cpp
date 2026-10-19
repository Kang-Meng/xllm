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

#include "processors/transforms.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace xllm::transforms {
namespace {

// Deterministic uint8 pattern shared with the golden generator.
torch::Tensor make_pattern(int64_t channels, int64_t height, int64_t width) {
  auto img =
      torch::empty({channels, height, width}, torch::dtype(torch::kUInt8));
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t y = 0; y < height; ++y) {
      for (int64_t x = 0; x < width; ++x) {
        img[c][y][x] = static_cast<uint8_t>((y * 31 + x * 17 + c * 83) % 256);
      }
    }
  }
  return img;
}

struct BicubicCase {
  int64_t in_h;
  int64_t in_w;
  int64_t out_h;
  int64_t out_w;
  // Golden uint8 output per channel (row-major), generated with
  // Pillow: Image.fromarray(channel).resize((out_w, out_h), BICUBIC).
  std::vector<std::vector<uint8_t>> golden;
};

// Golden outputs produced by Pillow 11.2.1 (libImaging/Resample.c). The C++
// port must stay bit-exact with Pillow's 8bpc fixed-point BICUBIC resampler:
// regenerate this table with PIL if the resampler contract ever changes.
const BicubicCase kBicubicCases[] = {
    {// in 5x7 -> out 3x4
     5,
     7,
     3,
     4,
     {
         {18, 46, 78, 106, 69, 97, 129, 157, 120, 148, 180, 208},
         {101, 129, 162, 197, 152, 184, 211, 186, 209, 193, 88, 26},
         {192, 225, 174, 79, 183, 88, 43, 60, 22, 50, 88, 118},
     }},
    {// in 8x6 -> out 5x10
     8,
     6,
     5,
     10,
     {
         {10,  15,  27,  37,  47,  58,  68,  78,  90,  96,  57,  63,  75,
          85,  95,  106, 116, 126, 138, 144, 108, 114, 126, 136, 146, 157,
          166, 177, 194, 204, 158, 164, 176, 185, 197, 215, 230, 236, 200,
          175, 206, 212, 224, 241, 220, 122, 101, 118, 59,  20},
         {92,  98,  110, 120, 130, 141, 151, 161, 177, 185, 140, 146, 158,
          168, 180, 199, 214, 221, 202, 190, 199, 206, 217, 233, 211, 131,
          123, 132, 59,  16,  176, 189, 194, 111, 42,  27,  30,  42,  61,
          72,  23,  28,  41,  53,  66,  81,  92,  102, 114, 120},
         {181, 187, 199, 212, 225, 239, 203, 145, 151, 163, 187, 200, 205,
          118, 66,  82,  63,  27,  35,  45,  15,  21,  33,  39,  46,  57,
          71,  87,  99,  105, 67,  73,  85,  96,  106, 117, 127, 137, 149,
          155, 116, 122, 134, 144, 154, 165, 175, 185, 197, 203},
     }},
    {// in 4x4 -> out 4x4 (identity)
     4,
     4,
     4,
     4,
     {
         {0, 17, 34, 51, 31, 48, 65, 82, 62, 79, 96, 113, 93, 110, 127, 144},
         {83,
          100,
          117,
          134,
          114,
          131,
          148,
          165,
          145,
          162,
          179,
          196,
          176,
          193,
          210,
          227},
         {166,
          183,
          200,
          217,
          197,
          214,
          231,
          248,
          228,
          245,
          6,
          23,
          3,
          20,
          37,
          54},
     }},
    {// in 3x3 -> out 1x1
     3,
     3,
     1,
     1,
     {
         {48},
         {131},
         {190},
     }},
    {// in 7x7 -> out 7x5 (width-only)
     7,
     7,
     7,
     5,
     {
         {4,   27,  51,  75,  98,  35,  58,  82,  106, 129, 66,  89,
          113, 137, 160, 97,  120, 144, 168, 191, 128, 151, 175, 199,
          222, 159, 182, 206, 240, 71,  190, 213, 250, 151, 14},
         {87,  110, 134, 158, 181, 118, 141, 165, 189, 212, 149, 172,
          196, 220, 243, 180, 203, 240, 141, 4,   211, 239, 39,  21,
          49,  255, 119, 20,  57,  80,  17,  40,  64,  88,  111},
         {170, 193, 217, 251, 82,  201, 229, 211, 11,  39,  246, 109,
          10,  47,  70,  7,   30,  54,  78,  101, 38,  61,  85,  109,
          132, 69,  92,  116, 140, 163, 100, 123, 147, 171, 194},
     }},
    {// in 4x3 -> out 8x6 (upscale)
     4,
     3,
     8,
     6,
     {
         {0,  0,  9,   19,  28,  33,  5,  8,  17,  27,  36,  41,
          20, 25, 34,  44,  53,  58,  37, 42, 51,  61,  70,  75,
          52, 57, 66,  76,  85,  90,  69, 74, 83,  93,  102, 107,
          86, 91, 100, 110, 119, 124, 94, 99, 108, 118, 127, 132},
         {78,  83,  92,  102, 111, 116, 86,  91,  100, 110, 119, 124,
          103, 108, 117, 127, 136, 141, 120, 125, 134, 144, 153, 158,
          135, 140, 149, 159, 168, 173, 152, 157, 166, 176, 185, 190,
          169, 174, 183, 193, 202, 207, 177, 182, 191, 201, 210, 215},
         {161, 166, 175, 185, 194, 199, 169, 174, 183, 194, 207, 213,
          186, 191, 199, 214, 233, 242, 209, 215, 226, 220, 195, 187,
          236, 246, 255, 211, 90,  45,  178, 188, 205, 154, 36,  0,
          44,  50,  62,  56,  34,  26,  0,   0,   0,   10,  32,  42},
     }},
};

class ResizePilBicubicTest : public ::testing::TestWithParam<BicubicCase> {};

TEST_P(ResizePilBicubicTest, MatchesPillowGoldenOutput) {
  const BicubicCase& test_case = GetParam();
  const torch::Tensor input =
      make_pattern(/*channels=*/3, test_case.in_h, test_case.in_w);
  const torch::Tensor out =
      resize_pil_bicubic(input, test_case.out_h, test_case.out_w);
  ASSERT_EQ(out.scalar_type(), torch::kUInt8);
  ASSERT_EQ(out.size(0), 3);
  ASSERT_EQ(out.size(1), test_case.out_h);
  ASSERT_EQ(out.size(2), test_case.out_w);
  for (int64_t c = 0; c < 3; ++c) {
    const torch::Tensor flat =
        out[c].reshape({test_case.out_h * test_case.out_w}).to(torch::kInt32);
    const std::vector<int32_t> got(flat.data_ptr<int32_t>(),
                                   flat.data_ptr<int32_t>() + flat.numel());
    const std::vector<int32_t> expected(
        test_case.golden[static_cast<size_t>(c)].begin(),
        test_case.golden[static_cast<size_t>(c)].end());
    EXPECT_EQ(got, expected) << "channel " << c << " mismatch";
  }
}

INSTANTIATE_TEST_SUITE_P(BicubicGolden,
                         ResizePilBicubicTest,
                         ::testing::ValuesIn(kBicubicCases));

TEST(ResizePilBicubicTest, ClampsFloatInputToUint8Range) {
  // Float input outside [0, 255] must clamp before the uint8 quantize.
  auto input =
      torch::arange(16, torch::dtype(torch::kFloat32)).reshape({1, 4, 4}) *
          30.0 -
      100.0;
  const torch::Tensor out = resize_pil_bicubic(input, 4, 4);
  ASSERT_EQ(out.scalar_type(), torch::kUInt8);
  EXPECT_EQ(out[0][0][0].item<int32_t>(), 0);    // -100 clamps to 0
  EXPECT_EQ(out[0][0][1].item<int32_t>(), 0);    // -70 clamps to 0
  EXPECT_EQ(out[0][3][3].item<int32_t>(), 255);  // 350 clamps to 255
}

TEST(ResizePilBicubicTest, RejectsNonPositiveOutputDims) {
  // A zero output dimension would divide by zero in the coefficient
  // precompute, so guard it at the API boundary instead.
  const torch::Tensor input = make_pattern(/*channels=*/3, 4, 4);
  EXPECT_DEATH(resize_pil_bicubic(input, 0, 4), "positive output height");
  EXPECT_DEATH(resize_pil_bicubic(input, 4, 0), "positive output width");
}

}  // namespace
}  // namespace xllm::transforms
