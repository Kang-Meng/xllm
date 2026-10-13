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
#include <vector>

#include "core/layers/npu/loader/deepseek_loader_family_utils.h"

namespace xllm::layer {
namespace {

using deepseek_loader_family_utils::trans_front_rope_weight;

constexpr int64_t kHeadDim = 128;
constexpr int64_t kHiddenDim = 3;

torch::Tensor expected_channel_order(int64_t rope_dim) {
  std::vector<int64_t> channels;
  channels.reserve(kHeadDim);
  for (int64_t channel = 0; channel < kHeadDim; ++channel) {
    if (channel >= rope_dim) {
      channels.emplace_back(channel);
      continue;
    }
    // The first half selects even channels; the second half selects odd ones.
    channels.emplace_back(channel < rope_dim / 2
                              ? 2 * channel
                              : 2 * (channel - rope_dim / 2) + 1);
  }
  return torch::tensor(channels, torch::kInt64);
}

TEST(DeepseekLoaderRopeTest, AffineAndProjectionUseSameChannelOrder) {
  const torch::Tensor affine = torch::arange(kHeadDim, torch::kFloat32);
  const torch::Tensor wk = torch::arange(kHeadDim * kHiddenDim, torch::kFloat32)
                               .view({kHeadDim, kHiddenDim});
  const torch::Tensor wq =
      torch::arange(2 * kHeadDim * kHiddenDim, torch::kFloat32)
          .view({2, kHeadDim, kHiddenDim});
  for (int64_t rope_dim : {64, 128}) {
    SCOPED_TRACE(rope_dim);
    const torch::Tensor order = expected_channel_order(rope_dim);
    const torch::Tensor reordered_affine = trans_front_rope_weight(
        affine, /*dim=*/0, rope_dim, /*clone_first=*/true);
    const torch::Tensor reordered_wk =
        trans_front_rope_weight(wk, /*dim=*/-2, rope_dim, /*clone_first=*/true);
    const torch::Tensor reordered_wq =
        trans_front_rope_weight(wq, /*dim=*/-2, rope_dim, /*clone_first=*/true);
    EXPECT_TRUE(torch::equal(reordered_affine, affine.index_select(0, order)));
    EXPECT_TRUE(torch::equal(reordered_wk, wk.index_select(-2, order)));
    EXPECT_TRUE(torch::equal(reordered_wq, wq.index_select(-2, order)));
    EXPECT_TRUE(
        torch::equal(reordered_affine, reordered_wk.select(1, 0) / kHiddenDim));
    EXPECT_TRUE(reordered_affine.is_contiguous());
    EXPECT_TRUE(reordered_wk.is_contiguous());
    EXPECT_TRUE(torch::equal(affine, torch::arange(kHeadDim, torch::kFloat32)));
    EXPECT_TRUE(
        torch::equal(wk,
                     torch::arange(kHeadDim * kHiddenDim, torch::kFloat32)
                         .view({kHeadDim, kHiddenDim})));
  }
}

TEST(DeepseekLoaderRopeTest, ReorderedAffinePreservesLayerNormChannelPairing) {
  const torch::Tensor wk = torch::arange(kHeadDim * kHiddenDim, torch::kFloat64)
                               .view({kHeadDim, kHiddenDim});
  const torch::Tensor input = torch::tensor({0.5, -1.0, 2.0}, torch::kFloat64);
  const torch::Tensor gamma =
      (torch::arange(kHeadDim, torch::kFloat64) + 0.5) / kHeadDim;
  const torch::Tensor beta = -gamma / 4;
  const auto normalize = [](const torch::Tensor& k) {
    const torch::Tensor centered = k - k.mean();
    return centered / (centered.square().mean() + 1e-5).sqrt();
  };
  const torch::Tensor expected =
      (normalize(torch::matmul(wk, input)) * gamma + beta)
          .index_select(/*dim=*/0, expected_channel_order(/*rope_dim=*/64));
  const torch::Tensor reordered_wk = trans_front_rope_weight(
      wk, /*dim=*/-2, /*rope_dim=*/64, /*clone_first=*/true);
  const torch::Tensor reordered_gamma = trans_front_rope_weight(
      gamma, /*dim=*/0, /*rope_dim=*/64, /*clone_first=*/true);
  const torch::Tensor reordered_beta = trans_front_rope_weight(
      beta, /*dim=*/0, /*rope_dim=*/64, /*clone_first=*/true);
  const torch::Tensor normalized =
      normalize(torch::matmul(reordered_wk, input));
  EXPECT_TRUE(torch::allclose(normalized * reordered_gamma + reordered_beta,
                              expected,
                              /*rtol=*/1e-12,
                              /*atol=*/1e-12));
  // The pre-fix pairing must fail on this non-uniform gamma/beta example.
  EXPECT_FALSE(torch::allclose(normalized * gamma + beta, expected));
}

TEST(DeepseekLoaderRopeTest, EagerModeUpdatesExistingStorage) {
  torch::Tensor wk = torch::arange(kHeadDim * kHiddenDim, torch::kFloat32)
                         .view({kHeadDim, kHiddenDim});
  const torch::Tensor expected =
      wk.index_select(/*dim=*/-2, expected_channel_order(/*rope_dim=*/64));
  const torch::Tensor reordered_wk = trans_front_rope_weight(
      wk, /*dim=*/-2, /*rope_dim=*/64, /*clone_first=*/false);
  EXPECT_EQ(reordered_wk.data_ptr(), wk.data_ptr());
  EXPECT_TRUE(torch::equal(wk, expected));
}

TEST(DeepseekLoaderRopeDeathTest, RejectsOddRopeDimension) {
  const torch::Tensor affine = torch::arange(kHeadDim);
  const torch::Tensor wk =
      torch::arange(kHeadDim * kHiddenDim).view({kHeadDim, kHiddenDim});
  EXPECT_DEATH(trans_front_rope_weight(affine,
                                       /*dim=*/0,
                                       /*rope_dim=*/65,
                                       /*clone_first=*/true),
               "Front RoPE dim must be even, got 65");
  EXPECT_DEATH(trans_front_rope_weight(wk,
                                       /*dim=*/-2,
                                       /*rope_dim=*/65,
                                       /*clone_first=*/true),
               "Front RoPE dim must be even, got 65");
}

}  // namespace
}  // namespace xllm::layer
