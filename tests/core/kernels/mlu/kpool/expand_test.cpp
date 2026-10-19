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

#include "kernels/mlu/kpool.h"
#include "platform/platform.h"

namespace xllm::kernel::mlu {
namespace {

TEST(KPoolKernelTest, ExpansionMapsLogicalTokensAndAllTailLengths) {
  const torch::Device device(Platform::type_torch(), 0);
  for (const int64_t block_size : {16, 32}) {
    const torch::Tensor table = torch::tensor({{3, 1, 4, 0, 2}}, torch::kInt64);
    torch::Tensor ids = torch::full({4, 512}, -1, torch::kInt64);
    ids.narrow(1, 0, 3).copy_(torch::tensor(
        {{0, -1, 1}, {0, -1, 1}, {0, -1, 1}, {0, -1, 1}}, torch::kInt64));
    const torch::Tensor pos = torch::tensor({7, 8, 9, 10}, torch::kInt32);
    const torch::Tensor rows = torch::zeros({4}, torch::kInt64);
    for (const bool always_tail : {false, true}) {
      torch::Tensor expected = torch::full({4, 2051}, -1, torch::kInt32);
      torch::Tensor lengths = torch::zeros({4}, torch::kInt32);
      for (int64_t r = 0; r < 4; ++r) {
        const int64_t count = 8 + (always_tail ? r : 0);
        lengths[r] = count;
        for (int64_t token = 0; token < count; ++token) {
          expected[r][token] =
              table[0][token / block_size].item<int64_t>() * block_size +
              token % block_size;
        }
      }
      const auto actual = expand_kpool(ids.to(device),
                                       pos.to(device),
                                       rows.to(device),
                                       table.to(device),
                                       block_size,
                                       2048,
                                       4,
                                       always_tail);
      EXPECT_EQ(actual.physical_slots.scalar_type(), torch::kInt32);
      EXPECT_TRUE(torch::equal(actual.physical_slots.cpu(), expected));
      EXPECT_TRUE(torch::equal(actual.context_lens.cpu(), lengths));
    }
  }
}

TEST(KPoolKernelTest, ExpansionMasksFuturePoolsAndMissingPages) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto opts = torch::TensorOptions().dtype(torch::kInt64).device(device);
  torch::Tensor ids = torch::full({3, 512}, -1, opts);
  ids.narrow(1, 0, 3).copy_(
      torch::tensor({{0, 4, -1}, {0, 4, -1}, {0, 4, -1}}, opts));
  const auto actual = expand_kpool(ids,
                                   torch::tensor({1, -1, 19}, opts),
                                   torch::zeros({3}, opts),
                                   torch::tensor({{3, -1}}, opts),
                                   16,
                                   2048,
                                   4,
                                   false);
  const torch::Tensor slots = actual.physical_slots.cpu();
  EXPECT_EQ(slots[0][0].item<int32_t>(), 48);
  EXPECT_EQ(slots[0][1].item<int32_t>(), 49);
  EXPECT_TRUE(slots[0].slice(0, 2).eq(-1).all().item<bool>());
  EXPECT_TRUE(slots[1].eq(-1).all().item<bool>());
  EXPECT_TRUE(torch::equal(actual.context_lens.cpu(),
                           torch::tensor({2, 0, 4}, torch::kInt32)));
}
TEST(KPoolKernelTest, ExpansionSupportsWideTablesAndPartialPools) {
  const torch::Device device(Platform::type_torch(), 0);
  for (const int64_t width : {8192, 8193, 65536}) {
    const torch::Tensor table =
        torch::arange(width * 2, torch::kInt64).flip({0}).view({2, width});
    table[0][1] = -1;
    table[1][2] = -1;
    for (const int64_t position : {-1, 1, 32767, 32768, 65537, 1048574}) {
      torch::Tensor ids = torch::full({2, 512}, -1, torch::kInt64);
      for (int64_t r = 0; r < 2; ++r) {
        for (int64_t c = 0; c < 512; ++c) {
          ids[r][c] = c % 7 == 0 ? -1 : (c * 131 + r) % (width * 4);
        }
        ids[r][31] = position >= 0 ? position / 4 : 0;
        ids[r][32] = width * 4;
        ids[r][33] = 4;
        ids[r][34] = 8;
      }
      const torch::Tensor rows = torch::tensor({1, 0}, torch::kInt64);
      for (const bool always_tail : {false, true}) {
        torch::Tensor expected = torch::full({2, 2051}, -1, torch::kInt32);
        torch::Tensor lengths = torch::zeros({2}, torch::kInt32);
        for (int64_t r = 0; r < 2; ++r) {
          int64_t written = 0;
          const auto append_token = [&](int64_t token) {
            if (token < 0 || token > position || token / 16 >= width) {
              return;
            }
            const int64_t page = table[1 - r][token / 16].item<int64_t>();
            if (page >= 0) {
              expected[r][written++] = page * 16 + token % 16;
            }
          };
          for (int64_t c = 0; c < 512; ++c) {
            const int64_t pool = ids[r][c].item<int64_t>();
            for (int64_t member = 0; member < 4; ++member) {
              append_token(pool * 4 + member);
            }
          }
          if (always_tail) {
            for (int64_t token = (position + 1) / 4 * 4; token <= position;
                 ++token) {
              append_token(token);
            }
          }
          lengths[r] = written;
        }
        const auto actual =
            expand_kpool(ids.to(device),
                         torch::full({2}, position, torch::kInt64).to(device),
                         rows.to(device),
                         table.to(device),
                         /*block_size=*/16,
                         /*token_budget=*/2048,
                         /*pool_size=*/4,
                         always_tail);
        SCOPED_TRACE(::testing::Message()
                     << "width=" << width << " position=" << position
                     << " tail=" << always_tail);
        EXPECT_TRUE(torch::equal(actual.physical_slots.cpu(), expected));
        EXPECT_TRUE(torch::equal(actual.context_lens.cpu(), lengths));
      }
    }
  }
}

}  // namespace
}  // namespace xllm::kernel::mlu

namespace xllm::kernel::mlu {
TEST(KPoolKernelTest, PrefillExpansionCompactsAcrossScanGroups) {
  const torch::Device device(Platform::type_torch(), 0);
  const torch::Tensor table =
      torch::arange(100, torch::kInt32).flip({0}).view({1, 100});
  torch::Tensor ids = torch::full({1, 512}, -1, torch::kInt64);
  ids.narrow(1, 0, 300).copy_(torch::arange(300, torch::kInt64));
  ids.slice(1, 0, 300, 3).fill_(-1);
  torch::Tensor expected = torch::full({1, 2051}, -1, torch::kInt32);
  int64_t written = 0;
  for (int64_t token = 0; token < 1200; ++token) {
    if ((token / 4) % 3 == 0) {
      continue;
    }
    expected[0][written++] =
        table[0][token / 16].item<int32_t>() * 16 + token % 16;
  }
  for (int64_t token = 1200; token < 1203; ++token) {
    expected[0][written++] =
        table[0][token / 16].item<int32_t>() * 16 + token % 16;
  }
  const auto actual = expand_kpool(
      ids.repeat({512, 1}).to(device),
      torch::full({512},
                  1202,
                  torch::TensorOptions().dtype(torch::kInt64).device(device)),
      torch::zeros({512},
                   torch::TensorOptions().dtype(torch::kInt64).device(device)),
      table.to(device),
      16,
      2048,
      4,
      true);
  EXPECT_TRUE(
      torch::equal(actual.physical_slots.cpu(), expected.repeat({512, 1})));
  EXPECT_TRUE(actual.context_lens.eq(written).all().item<bool>());
}
}  // namespace xllm::kernel::mlu
