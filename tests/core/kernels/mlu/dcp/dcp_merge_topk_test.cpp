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

Candidates make_input(int32_t rows, bool edges = false) {
  std::vector<torch::Tensor> scores;
  std::vector<torch::Tensor> slots;
  scores.reserve(kDcp);
  slots.reserve(kDcp);
  torch::Tensor mapping;
  for (int32_t shard = 0; shard < kDcp; ++shard) {
    auto input = dcp_test::make_candidates(rows,
                                           shard,
                                           /*tp_rank=*/0,
                                           /*seed=*/42,
                                           edges);
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
  GuardedTopk local(rows, kTopk, input.scores.device());
  for (int32_t rank = 0; rank < kDcp; ++rank) {
    kernel::mlu::dcp_localize_topk(out.output,
                                   out.lengths,
                                   rank,
                                   /*size=*/kDcp,
                                   /*interleave=*/16,
                                   local.output,
                                   local.lengths);
    local.check(local_reference(reference.first.cpu(),
                                reference.second.cpu(),
                                rank,
                                /*size=*/kDcp,
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
  EXPECT_DEATH(dcp_merge_topk(input.scores.slice(0, 0, 2),
                              input.slots.slice(0, 0, 2),
                              input.mapping,
                              kTopk,
                              out.output,
                              out.lengths),
               "scores.size");
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
}  // namespace
}  // namespace xllm
