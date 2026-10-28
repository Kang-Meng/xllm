/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <torch/torch.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "kernels/mlu/mlu_ops_api.h"

namespace xllm {
namespace {

torch::Tensor reference_clamp(const torch::Tensor& input, float limit) {
  const auto parts = input.chunk(/*chunks=*/2, /*dim=*/-1);
  return torch::cat({parts[0].clamp_max(limit), parts[1].clamp(-limit, limit)},
                    /*dim=*/-1);
}

torch::Tensor with_specials(torch::Tensor input) {
  const int64_t width = input.size(-1);
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<float> specials = {
      -inf, inf, std::numeric_limits<float>::quiet_NaN(), -0.0f, 10.0f, -10.0f};
  torch::Tensor rows = input.view({-1, width});
  for (int64_t i = 0; i < static_cast<int64_t>(specials.size()); ++i) {
    rows[0][i % width] = specials[i];
    rows[-1][(width / 2 + i) % width] = specials[i];
  }
  return input;
}

// Per-expert counts summing to `total`, spread over `experts` entries.
torch::Tensor make_counts(int64_t total,
                          int64_t experts,
                          torch::ScalarType dtype) {
  torch::Tensor counts = torch::zeros({experts}, torch::kInt64);
  for (int64_t i = 0; i < total; ++i) {
    counts[(i * 7) % experts] += 1;
  }
  return counts.to(dtype);
}

TEST(FusedMoeClampTest, MatchesReferenceForDtypesWidthsAndSpecials) {
  const torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  for (const torch::ScalarType dtype :
       {torch::kBFloat16, torch::kFloat16, torch::kFloat32}) {
    for (const int64_t width : {2, 14, 258, 4096, 6144}) {
      for (const int64_t rows : {1, 9, 257, 4096}) {
        const torch::Tensor input = with_specials(
            torch::randn({rows, width}).mul(/*other=*/12).to(dtype));
        const torch::Tensor expected = reference_clamp(input, /*limit=*/10);
        torch::Tensor output = input.to(device);
        kernel::mlu::fused_moe_clamp(output, /*limit=*/10, std::nullopt);
        EXPECT_TRUE(torch::allclose(
            output.cpu(), expected, /*rtol=*/0, /*atol=*/0, /*equal_nan=*/true))
            << "dtype " << dtype << " width " << width << " rows " << rows;
      }
    }
  }
}

TEST(FusedMoeClampTest, ClampsCountedRowPrefix) {
  const torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  // 32 x 512 takes the small-input path; the others the counted path.
  for (const auto& [rows, width] : std::vector<std::pair<int64_t, int64_t>>{
           {32, 512}, {4096, 512}, {4096, 4096}, {300, 6144}}) {
    for (const int64_t total : {int64_t{0}, int64_t{31}, rows - 1, rows * 3}) {
      for (const torch::ScalarType count_dtype :
           {torch::kInt32, torch::kInt64}) {
        const torch::Tensor input = with_specials(
            torch::randn({rows, width}).mul(/*other=*/12).to(torch::kBFloat16));
        const torch::Tensor expected = reference_clamp(input, /*limit=*/10);
        torch::Tensor output = input.to(device);
        kernel::mlu::fused_moe_clamp(
            output,
            /*limit=*/10,
            make_counts(total, /*experts=*/36, count_dtype).to(device));
        const torch::Tensor actual = output.cpu();
        const int64_t valid = std::min(total, rows);
        EXPECT_TRUE(
            torch::allclose(actual.slice(/*dim=*/0, /*start=*/0, valid),
                            expected.slice(/*dim=*/0, /*start=*/0, valid),
                            /*rtol=*/0,
                            /*atol=*/0,
                            /*equal_nan=*/true))
            << rows << "x" << width << " valid " << valid;
        // Rows past the counted prefix are either untouched or clamped.
        for (int64_t row = valid; row < rows; ++row) {
          EXPECT_TRUE(torch::allclose(actual[row],
                                      input[row],
                                      /*rtol=*/0,
                                      /*atol=*/0,
                                      /*equal_nan=*/true) ||
                      torch::allclose(actual[row],
                                      expected[row],
                                      /*rtol=*/0,
                                      /*atol=*/0,
                                      /*equal_nan=*/true))
              << rows << "x" << width << " row " << row;
        }
      }
    }
  }
}

TEST(FusedMoeClampTest, ThreeDimensionalInputAndEmptyBatch) {
  const torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  const torch::Tensor input =
      torch::randn({2, 8, 4096}).mul(/*other=*/12).to(torch::kBFloat16);
  torch::Tensor output = input.to(device);
  kernel::mlu::fused_moe_clamp(output, /*limit=*/10, std::nullopt);
  EXPECT_TRUE(torch::equal(output.cpu(), reference_clamp(input, /*limit=*/10)));
  torch::Tensor empty = torch::empty({0, 4096}, input.options().device(device));
  kernel::mlu::fused_moe_clamp(
      empty,
      /*limit=*/10,
      make_counts(/*total=*/0, /*experts=*/36, torch::kInt32).to(device));
  EXPECT_EQ(empty.numel(), 0);
}

TEST(FusedMoeClampTest, MatchesOriginalActivationQuantizationChain) {
  const torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  const torch::Tensor input = torch::randn({16, 4096})
                                  .mul(/*other=*/12)
                                  .to(torch::kBFloat16)
                                  .to(device);
  const torch::Tensor original = reference_clamp(input, /*limit=*/10);
  torch::Tensor fused = input.clone();
  kernel::mlu::fused_moe_clamp(fused, /*limit=*/10, std::nullopt);
  const torch::Tensor smooth =
      torch::ones({2048}, input.options().dtype(torch::kFloat32));
  auto quantize = [&smooth](const torch::Tensor& value) {
    return kernel::mlu::scaled_quantize(value,
                                        smooth,
                                        std::nullopt,
                                        std::nullopt,
                                        std::nullopt,
                                        std::nullopt,
                                        std::nullopt,
                                        std::nullopt,
                                        /*act_mode=*/"silu",
                                        /*active_coef=*/1.0,
                                        /*is_gated=*/true);
  };
  const auto [expected, expected_scale] = quantize(original);
  const auto [actual, actual_scale] = quantize(fused);
  EXPECT_TRUE(torch::equal(actual.cpu(), expected.cpu()));
  EXPECT_TRUE(torch::equal(actual_scale.cpu(), expected_scale.cpu()));
}

}  // namespace
}  // namespace xllm
