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

#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "core/kernels/mlu/dcp/dcp_merge_test_utils.h"
#include "core/kernels/mlu/dcp/dcp_topk_test_utils.h"
#include "core/platform/platform.h"
#include "kernels/mlu/mlu_ops_api.h"

namespace xllm {
namespace {
using dcp_test::Candidates;
using kernel::mlu::dcp_merge_topk;
constexpr int64_t kDcp = 4;
constexpr int64_t kTopk = 2048;

Candidates make_input(int32_t rows,
                      bool edges = false,
                      int32_t dcp = static_cast<int32_t>(kDcp)) {
  std::vector<torch::Tensor> scores;
  std::vector<torch::Tensor> slots;
  scores.reserve(dcp);
  slots.reserve(dcp);
  torch::Tensor mapping;
  for (int32_t shard = 0; shard < dcp; ++shard) {
    auto input = dcp_test::make_candidates(rows,
                                           shard,
                                           /*tp_rank=*/0,
                                           /*seed=*/42,
                                           edges,
                                           dcp);
    scores.emplace_back(std::move(input.scores));
    slots.emplace_back(std::move(input.slots));
    mapping = std::move(input.mapping);
  }
  const torch::Device device{torch::kPrivateUse1, 0};
  return {torch::stack(scores).to(device),
          torch::stack(slots).to(device),
          mapping.to(device)};
}

torch::Tensor storage_bits(const torch::Tensor& tensor) {
  const int64_t count =
      static_cast<int64_t>(tensor.storage().nbytes() / tensor.element_size());
  auto storage = tensor.as_strided({count}, {1}, /*storage_offset=*/0);
  return (tensor.scalar_type() == torch::kBFloat16 ? storage.view(torch::kInt16)
                                                   : storage)
      .cpu()
      .clone();
}

void check_input(Candidates input, bool compose = false) {
  const int64_t rows = input.scores.size(1);
  const auto scores_before = storage_bits(input.scores);
  const auto slots_before = storage_bits(input.slots);
  const auto mapping_before = storage_bits(input.mapping);
  const auto reference = dcp_test::merge_reference(
      input.scores, input.slots, input.mapping, kTopk);
  GuardedTopk out(rows, kTopk, input.scores.device());
  for (int32_t repeat = 0; repeat < 2; ++repeat) {
    out.output.fill_(kSentinel);
    out.lengths.fill_(kSentinel);
    dcp_merge_topk(input.scores,
                   input.slots,
                   input.mapping,
                   kTopk,
                   out.output,
                   out.lengths);
    out.check({reference.first.cpu(), reference.second.cpu()});
  }
  EXPECT_TRUE(torch::equal(storage_bits(input.scores), scores_before));
  EXPECT_TRUE(torch::equal(storage_bits(input.slots), slots_before));
  EXPECT_TRUE(torch::equal(storage_bits(input.mapping), mapping_before));
  if (!compose) {
    return;
  }
  const int32_t dcp = static_cast<int32_t>(input.scores.size(0));
  GuardedTopk local(rows, kTopk, input.scores.device());
  for (int32_t rank = 0; rank < dcp; ++rank) {
    kernel::mlu::dcp_localize_topk(out.output,
                                   out.lengths,
                                   rank,
                                   /*size=*/dcp,
                                   /*interleave=*/16,
                                   local.output,
                                   local.lengths);
    local.check(local_reference(reference.first.cpu(),
                                reference.second.cpu(),
                                rank,
                                /*size=*/dcp,
                                /*interleave=*/16));
  }
}

class DcpMergeTopkTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (Platform::device_count() == 0) {
      GTEST_SKIP() << "Requires an MLU device";
    }
  }
};

TEST_F(DcpMergeTopkTest, EmptyAndQueryBoundaries) {
  for (int32_t rows : {0,  1,  7,  8,   9,   16,  24,  31,  32,  33,
                       63, 64, 65, 127, 128, 129, 255, 256, 257, 8192}) {
    SCOPED_TRACE(rows);
    check_input(make_input(rows));
  }
}

TEST_F(DcpMergeTopkTest, ExactTiesSpecialValuesAndInvalidCandidates) {
  check_input(make_input(/*rows=*/16, /*edges=*/true), /*compose=*/true);
}

TEST_F(DcpMergeTopkTest, StridesOffsetsAndTransposedInputs) {
  for (int32_t rows : {1, 16, 33, 65, 129, 257}) {
    auto input = make_input(rows, /*edges=*/rows == 16);
    input.scores = dcp_test::strided_copy(input.scores);
    input.slots = dcp_test::strided_copy(input.slots);
    input.mapping = dcp_test::strided_copy(input.mapping);
    check_input(std::move(input));
  }
  auto input = make_input(/*rows=*/33);
  input.scores = input.scores.transpose(0, 1).contiguous().transpose(0, 1);
  input.slots = input.slots.transpose(1, 2).contiguous().transpose(1, 2);
  check_input(std::move(input));
}

TEST_F(DcpMergeTopkTest, RawBf16BitsAndDuplicateSlots) {
  auto input = make_input(/*rows=*/65);
  torch::manual_seed(/*seed=*/2026);
  auto bits =
      torch::randint(-32768, 32768, input.scores.sizes(), torch::kInt16);
  input.scores.copy_(bits.view(torch::kBFloat16));
  input.slots.copy_(input.slots.abs().remainder(17));
  input.mapping.fill_(0);
  check_input(std::move(input));
}

TEST_F(DcpMergeTopkTest, FullCandidatesAtDecodeAndVerificationSizes) {
  for (int32_t rows : {8, 16, 32, 64}) {
    auto input = make_input(rows);
    input.slots.copy_(torch::arange(input.slots.numel(), input.slots.options())
                          .view(input.slots.sizes()));
    input.mapping.fill_(0);
    check_input(std::move(input));
  }
}

TEST_F(DcpMergeTopkTest, NonDefaultStreamAndGraphReplay) {
  auto input = make_input(/*rows=*/16, /*edges=*/true);
  GuardedTopk out(/*q=*/16, kTopk, input.scores.device());
  auto run = [&] {
    dcp_merge_topk(input.scores,
                   input.slots,
                   input.mapping,
                   kTopk,
                   out.output,
                   out.lengths);
  };
  run();
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  {
    torch_mlu::mlu::MLUStreamGuard guard(
        torch_mlu::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0));
    run();
    graph.capture_begin();
    run();
    graph.capture_end();
  }
  torch_mlu::synchronize();
  for (int32_t round = 0; round < 3; ++round) {
    input.mapping.copy_(input.mapping.flip({0}));
    input.scores.copy_(input.scores.flip({2}));
    const auto expected = dcp_test::merge_reference(
        input.scores, input.slots, input.mapping, kTopk);
    graph.replay();
    torch_mlu::synchronize();
    out.check({expected.first.cpu(), expected.second.cpu()});
  }
}

TEST(DcpMergeTopkDeathTest, RejectsUnsupportedContractsAndAliasing) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "Requires an MLU device";
  }
  auto input = make_input(/*rows=*/2);
  GuardedTopk out(/*q=*/2, kTopk, input.scores.device());
  EXPECT_DEATH(dcp_merge_topk(input.scores.to(torch::kFloat32),
                              input.slots,
                              input.mapping,
                              kTopk,
                              out.output,
                              out.lengths),
               "scalar_type");
  for (const int64_t shards : {1, 3}) {
    EXPECT_DEATH(dcp_merge_topk(input.scores.slice(0, 0, shards),
                                input.slots.slice(0, 0, shards),
                                input.mapping,
                                kTopk,
                                out.output,
                                out.lengths),
                 "power-of-two shard counts");
  }
  auto twelve = make_input(/*rows=*/2, /*edges=*/false, /*dcp=*/12);
  EXPECT_DEATH(dcp_merge_topk(twelve.scores,
                              twelve.slots,
                              twelve.mapping,
                              kTopk,
                              out.output,
                              out.lengths),
               "power-of-two shard counts");
  EXPECT_DEATH(dcp_merge_topk(input.scores.slice(2, 0, 1024),
                              input.slots.slice(2, 0, 1024),
                              input.mapping,
                              kTopk,
                              out.output,
                              out.lengths),
               "scores.size");
  EXPECT_DEATH(dcp_merge_topk(input.scores,
                              input.slots,
                              input.mapping,
                              1024,
                              out.output,
                              out.lengths),
               "topk");
  EXPECT_DEATH(dcp_merge_topk(input.scores,
                              input.slots,
                              input.mapping.slice(0, 0, 1),
                              kTopk,
                              out.output,
                              out.lengths),
               "mapping.size");
  auto alias = input.slots[0];
  EXPECT_DEATH(
      dcp_merge_topk(
          input.scores, input.slots, input.mapping, kTopk, alias, out.lengths),
      "alias");
  auto strided = dcp_test::strided_copy(out.output);
  EXPECT_DEATH(dcp_merge_topk(input.scores,
                              input.slots,
                              input.mapping,
                              kTopk,
                              strided,
                              out.lengths),
               "contiguous");
  EXPECT_DEATH(dcp_merge_topk(input.scores,
                              input.slots,
                              input.mapping,
                              kTopk,
                              out.output,
                              input.mapping),
               "alias");
  auto expanded = input.scores.slice(0, 0, 1).expand({4, 2, kTopk});
  EXPECT_DEATH(
      dcp_merge_topk(
          expanded, input.slots, input.mapping, kTopk, out.output, out.lengths),
      "stride");
}
TEST_F(DcpMergeTopkTest, TiesUsePhysicalSlotsAcrossCandidatePermutations) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto options = torch::TensorOptions().device(device);
  auto scores = torch::ones({4, 1, 2048}, options.dtype(torch::kBFloat16));
  torch::manual_seed(42);
  auto slots = (torch::randperm(8192, torch::kInt32) + (1 << 30))
                   .reshape({4, 1, 2048})
                   .to(device);
  auto mapping = torch::zeros({1}, options.dtype(torch::kInt32));
  auto output = torch::empty({1, 2048}, slots.options());
  auto lengths = torch::empty({1}, slots.options());
  auto expected = (torch::arange(2048, torch::kInt32) + (1 << 30)).unsqueeze(0);
  for (int32_t repeat = 0; repeat < 2; ++repeat) {
    dcp_merge_topk(scores, slots, mapping, /*topk=*/2048, output, lengths);
    EXPECT_TRUE(torch::equal(output.cpu(), expected));
    EXPECT_EQ(lengths[0].item<int32_t>(), 2048);
    slots = slots.flip({0, 2});
  }
}

TEST_F(DcpMergeTopkTest, OrdersEveryPhysicalSlotBit) {
  const torch::Device device(torch::kPrivateUse1, 0);
  auto slots = torch::empty({kDcp, 32, kTopk}, torch::kInt32);
  const auto pattern = torch::arange(kTopk, torch::kInt32).remainder(2);
  for (int32_t bit = 0; bit < 32; ++bit) {
    const int32_t value =
        bit < 31 ? static_cast<int32_t>(uint32_t{1} << bit) : 0x7fffffff;
    slots.select(1, bit).copy_(pattern.mul(value).expand({kDcp, kTopk}));
  }
  check_input({torch::ones({kDcp, 32, kTopk}, torch::kBFloat16).to(device),
               slots.to(device),
               torch::zeros({32}, torch::kInt32).to(device)});
}

TEST_F(DcpMergeTopkTest, ConstantScoresAndSparseValidSlots) {
  const torch::Device device(torch::kPrivateUse1, 0);
  auto scores = torch::full({kDcp, 5, kTopk},
                            -std::numeric_limits<float>::infinity(),
                            torch::kBFloat16);
  auto slots = torch::arange(kDcp * kTopk, torch::kInt32)
                   .reshape({kDcp, 1, kTopk})
                   .repeat({1, 5, 1});
  slots.select(1, 1).fill_(-1);
  slots.select(1, 2).fill_(-1);
  slots[0][2][0] = std::numeric_limits<int32_t>::max();
  scores.select(1, 3).fill_(std::numeric_limits<float>::quiet_NaN());
  slots.select(1, 4).fill_(17);
  check_input({scores.to(device),
               slots.to(device),
               torch::zeros({5}, torch::kInt32).to(device)});
}

TEST_F(DcpMergeTopkTest, PreservesNarrowAndWidePriorityRanges) {
  const torch::Device device(torch::kPrivateUse1, 0);
  torch::manual_seed(2026);
  const auto order = torch::randperm(kDcp * kTopk, torch::kInt32);
  const auto signed_scores = torch::where(order.remainder(2) == 0, 1.0f, -1.0f)
                                 .to(torch::kBFloat16)
                                 .reshape({kDcp, 1, kTopk});
  for (const int32_t stride : {4, 8}) {
    const auto slots = (order * stride + (1 << 30)).reshape({kDcp, 1, kTopk});
    check_input({signed_scores.to(device),
                 slots.to(device),
                 torch::zeros({1}, torch::kInt32).to(device)});
  }
}

// Two shards run natively; eight and sixteen reduce groups of four shards
// before the final native merge.
class DcpMergeTopkShardTest : public DcpMergeTopkTest,
                              public ::testing::WithParamInterface<int32_t> {};

TEST_P(DcpMergeTopkShardTest, QueryBoundariesMatchReference) {
  for (int32_t rows : {0, 1, 7, 9, 33, 129, 257}) {
    SCOPED_TRACE(rows);
    check_input(make_input(rows, /*edges=*/false, GetParam()));
  }
}

TEST_P(DcpMergeTopkShardTest, ExactTiesSpecialValuesAndInvalidCandidates) {
  check_input(make_input(/*rows=*/16, /*edges=*/true, GetParam()),
              /*compose=*/true);
}

TEST_P(DcpMergeTopkShardTest, StridesOffsetsAndTransposedInputs) {
  for (int32_t rows : {1, 16, 65}) {
    auto input = make_input(rows, /*edges=*/rows == 16, GetParam());
    input.scores = dcp_test::strided_copy(input.scores);
    input.slots = dcp_test::strided_copy(input.slots);
    input.mapping = dcp_test::strided_copy(input.mapping);
    check_input(std::move(input));
  }
  auto input = make_input(/*rows=*/33, /*edges=*/false, GetParam());
  input.scores = input.scores.transpose(0, 1).contiguous().transpose(0, 1);
  input.slots = input.slots.transpose(1, 2).contiguous().transpose(1, 2);
  check_input(std::move(input));
}

TEST_P(DcpMergeTopkShardTest, RawBf16BitsAndDuplicateSlots) {
  auto input = make_input(/*rows=*/65, /*edges=*/false, GetParam());
  torch::manual_seed(/*seed=*/2026);
  auto bits =
      torch::randint(-32768, 32768, input.scores.sizes(), torch::kInt16);
  input.scores.copy_(bits.view(torch::kBFloat16));
  input.slots.copy_(input.slots.abs().remainder(17));
  input.mapping.fill_(0);
  check_input(std::move(input));
}

TEST_P(DcpMergeTopkShardTest, TiesUsePhysicalSlotsAcrossShards) {
  const int64_t dcp = GetParam();
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto options = torch::TensorOptions().device(device);
  auto scores = torch::ones({dcp, 1, kTopk}, options.dtype(torch::kBFloat16));
  torch::manual_seed(42);
  auto slots = (torch::randperm(dcp * kTopk, torch::kInt32) + (1 << 30))
                   .reshape({dcp, 1, kTopk})
                   .to(device);
  auto mapping = torch::zeros({1}, options.dtype(torch::kInt32));
  auto output = torch::empty({1, kTopk}, slots.options());
  auto lengths = torch::empty({1}, slots.options());
  auto expected =
      (torch::arange(kTopk, torch::kInt32) + (1 << 30)).unsqueeze(0);
  for (int32_t repeat = 0; repeat < 2; ++repeat) {
    dcp_merge_topk(scores, slots, mapping, kTopk, output, lengths);
    EXPECT_TRUE(torch::equal(output.cpu(), expected));
    EXPECT_EQ(lengths[0].item<int32_t>(), kTopk);
    slots = slots.flip({0, 2});
  }
}

TEST_P(DcpMergeTopkShardTest, ConstantScoresAndSparseValidSlots) {
  const int64_t dcp = GetParam();
  const torch::Device device(torch::kPrivateUse1, 0);
  auto scores = torch::full({dcp, 5, kTopk},
                            -std::numeric_limits<float>::infinity(),
                            torch::kBFloat16);
  auto slots = torch::arange(dcp * kTopk, torch::kInt32)
                   .reshape({dcp, 1, kTopk})
                   .repeat({1, 5, 1});
  slots.select(1, 1).fill_(-1);
  slots.select(1, 2).fill_(-1);
  slots[dcp - 1][2][0] = std::numeric_limits<int32_t>::max();
  scores.select(1, 3).fill_(std::numeric_limits<float>::quiet_NaN());
  slots.select(1, 4).fill_(17);
  check_input({scores.to(device),
               slots.to(device),
               torch::zeros({5}, torch::kInt32).to(device)});
}

TEST_P(DcpMergeTopkShardTest, NonDefaultStreamAndGraphReplay) {
  auto input = make_input(/*rows=*/16, /*edges=*/true, GetParam());
  GuardedTopk out(/*q=*/16, kTopk, input.scores.device());
  auto run = [&] {
    dcp_merge_topk(input.scores,
                   input.slots,
                   input.mapping,
                   kTopk,
                   out.output,
                   out.lengths);
  };
  run();
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  {
    torch_mlu::mlu::MLUStreamGuard guard(
        torch_mlu::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0));
    run();
    graph.capture_begin();
    run();
    graph.capture_end();
  }
  torch_mlu::synchronize();
  for (int32_t round = 0; round < 3; ++round) {
    input.mapping.copy_(input.mapping.flip({0}));
    input.scores.copy_(input.scores.flip({2}));
    const auto expected = dcp_test::merge_reference(
        input.scores, input.slots, input.mapping, kTopk);
    graph.replay();
    torch_mlu::synchronize();
    out.check({expected.first.cpu(), expected.second.cpu()});
  }
}

INSTANTIATE_TEST_SUITE_P(SupportedShardCounts,
                         DcpMergeTopkShardTest,
                         ::testing::Values(2, 8, 16));

}  // namespace
}  // namespace xllm
