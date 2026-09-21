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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <limits>

#include "kernels/mlu/kpool.h"
#include "platform/platform.h"

namespace xllm::kernel::mlu {
namespace {

TEST(KPoolKernelTest, ScoresRespectHeadsCausalityAndPages) {
  torch::manual_seed(203);
  const torch::Device device(Platform::type_torch(), 0);
  const auto opts = torch::TensorOptions().dtype(torch::kBFloat16);
  const torch::Tensor q = torch::randn({7, 32, 128}, opts);
  const torch::Tensor w = torch::randn({7, 32});
  const torch::Tensor cache = torch::randn({4, 1, 4, 128}, opts);
  const torch::Tensor table = torch::tensor({{3, 0}, {2, -1}}, torch::kInt32);
  const torch::Tensor positions =
      torch::tensor({-1, 2, 3, 6, 7, 17, 22}, torch::kInt32);
  const torch::Tensor rows =
      torch::tensor({0, 0, 0, 0, 0, 1, 1}, torch::kInt64);
  torch::Tensor expected =
      torch::full({7, 8}, -std::numeric_limits<float>::infinity());
  for (int64_t i = 0; i < 7; ++i) {
    for (int64_t j = 0; j < 8; ++j) {
      const int64_t page =
          table[rows[i].item<int64_t>()][j / 4].item<int32_t>();
      if (page < 0 || (j + 1) * 4 > positions[i].item<int32_t>() + 1) {
        continue;
      }
      const torch::Tensor dots =
          (q[i].to(torch::kFloat32) * cache[page][0][j % 4].to(torch::kFloat32))
              .sum(1);
      expected[i][j] = (torch::relu(dots * 0.125) * w[i]).sum();
    }
  }
  {
    torch::Tensor storage =
        torch::full({7, 10}, 99.0, expected.options().device(device));
    torch::Tensor scores = storage.narrow(1, 1, 8);
    score_kpool(q.to(device),
                w.to(device),
                cache.to(device),
                table.to(device),
                positions.to(device),
                rows.to(device),
                scores,
                16,
                4,
                0.125);
    const torch::Tensor actual = scores.cpu();
    EXPECT_TRUE(
        torch::equal(torch::isneginf(actual), torch::isneginf(expected)));
    const torch::Tensor finite = torch::isfinite(expected);
    EXPECT_TRUE(torch::allclose(actual.masked_select(finite),
                                expected.masked_select(finite),
                                1e-4,
                                1e-4));
    EXPECT_TRUE(storage.select(1, 0).eq(99).all().item<bool>());
    EXPECT_TRUE(storage.select(1, 9).eq(99).all().item<bool>());
  }
}

TEST(KPoolKernelTest, NativeTopkHandlesTiesNonfiniteAndEmpty) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  for (const int64_t columns : {0, 17, 513}) {
    torch::Tensor scores = torch::zeros({3, columns}, opts);
    if (columns > 0) {
      scores[1].fill_(-std::numeric_limits<float>::infinity());
      scores[2][0] = std::numeric_limits<float>::quiet_NaN();
      scores[2][1] = std::numeric_limits<float>::infinity();
    }
    const torch::Tensor actual = select_kpool(scores, 512);
    EXPECT_EQ(actual.scalar_type(), torch::kInt64);
    EXPECT_EQ(actual.size(1), 512);
    const int64_t k = std::min<int64_t>(512, columns);
    if (k > 0) {
      const auto [v, ids] = scores.topk(k, -1);
      EXPECT_TRUE(torch::equal(actual.narrow(1, 0, k),
                               torch::where(torch::isfinite(v), ids, -1)));
    }
    EXPECT_TRUE(actual.narrow(1, k, 512 - k).eq(-1).all().item<bool>());
  }
  EXPECT_EQ(select_kpool(torch::empty({0, 5}, opts), 3).numel(), 0);
  EXPECT_EQ(select_kpool(torch::empty({2, 5}, opts), 0).numel(), 0);
}
}  // namespace
}  // namespace xllm::kernel::mlu
