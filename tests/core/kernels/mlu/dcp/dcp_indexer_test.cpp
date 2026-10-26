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

#include "kernels/mlu/bangc/include/dcp_indexer.h"

#include <framework/core/MLUStream.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/kernels/mlu/dcp/dcp_merge_test_utils.h"
#include "core/kernels/mlu/dcp/dcp_precision_test_utils.h"
#include "kernels/mlu/mlu_ops_api.h"

namespace xllm {
namespace {
constexpr int32_t kTopk = 2048;
constexpr int32_t kPage = 16;
const torch::Device kDevice(torch::kPrivateUse1, 0);
using Result = std::tuple<torch::Tensor, torch::Tensor>;

struct Inputs {
  torch::Tensor query;
  torch::Tensor cache;
  torch::Tensor weights;
  torch::Tensor table;
  torch::Tensor cu;
  torch::Tensor positions;
  torch::Tensor lens;
  int32_t size = 4;
};

Inputs make_inputs(const std::vector<int32_t>& queries,
                   const std::vector<int32_t>& lengths,
                   int32_t size = 4) {
  CHECK_EQ(queries.size(), lengths.size());
  torch::manual_seed(42);
  std::vector<int32_t> cu{0};
  std::vector<int32_t> positions;
  std::vector<int32_t> pages;
  cu.reserve(queries.size() + 1);
  pages.reserve(queries.size());
  int32_t total_pages = 0;
  int32_t width = 0;
  for (size_t req = 0; req < queries.size(); ++req) {
    cu.emplace_back(cu.back() + queries[req]);
    const int32_t count = (lengths[req] + kPage * size - 1) / (kPage * size);
    pages.emplace_back(count);
    total_pages += count;
    width = std::max(width, count);
  }
  positions.reserve(cu.back());
  for (size_t req = 0; req < queries.size(); ++req) {
    for (int32_t row = 0; row < queries[req]; ++row) {
      positions.emplace_back(lengths[req] >= queries[req]
                                 ? lengths[req] - queries[req] + row
                                 : -1);
    }
  }
  auto table = torch::full(
      {static_cast<int64_t>(queries.size()), width}, -1, torch::kInt32);
  auto order = torch::randperm(total_pages, torch::kInt32) + 3;
  int32_t offset = 0;
  for (size_t req = 0; req < queries.size(); ++req) {
    table[req]
        .narrow(0, 0, pages[req])
        .copy_(order.narrow(0, offset, pages[req]));
    offset += pages[req];
  }
  return Inputs{
      torch::randn({cu.back(), 32, 128}).to(torch::kBFloat16).to(kDevice),
      torch::randn({total_pages + 3, 1, kPage, 128})
          .to(torch::kBFloat16)
          .to(kDevice),
      torch::randn({cu.back(), 32}).to(torch::kBFloat16).to(kDevice),
      table.to(kDevice),
      torch::tensor(cu, torch::kInt32).to(kDevice),
      torch::tensor(positions, torch::kInt32).to(kDevice),
      torch::tensor(lengths, torch::kInt32).to(kDevice),
      size};
}

Result run(const Inputs& x, int32_t rank, bool prefill) {
  return kernel::mlu::bangc::dcp_indexer(x.query,
                                         x.cache,
                                         x.weights,
                                         x.table,
                                         x.cu,
                                         x.positions,
                                         x.lens,
                                         rank,
                                         x.size,
                                         prefill);
}

// Enumerates logical tokens and uses Torch scoring, independently of launch
// policy.
Result check_candidates(const Inputs& x, int32_t rank, const Result& actual) {
  const auto& [score, slot] = actual;
  CHECK_EQ(score.scalar_type(), torch::kBFloat16);
  CHECK_EQ(slot.scalar_type(), torch::kInt32);
  CHECK_EQ(score.device(), kDevice);
  CHECK_EQ(slot.device(), kDevice);
  CHECK(score.is_contiguous() && slot.is_contiguous());
  const int64_t rows = x.query.size(0);
  CHECK(score.sizes() == torch::IntArrayRef({rows, kTopk}));
  CHECK(score.sizes() == slot.sizes());
  auto actual_scores = score.to(torch::kFloat32).cpu().contiguous();
  auto actual_slots = slot.cpu().contiguous();
  const auto values = actual_scores.accessor<float, 2>();
  const auto ids = actual_slots.accessor<int32_t, 2>();
  auto table_cpu = x.table.cpu();
  auto cu_cpu = x.cu.cpu();
  auto pos_cpu = x.positions.cpu();
  auto lens_cpu = x.lens.cpu();
  const auto table = table_cpu.accessor<int32_t, 2>();
  const auto cu = cu_cpu.accessor<int32_t, 1>();
  const auto positions = pos_cpu.accessor<int32_t, 1>();
  const auto lengths = lens_cpu.accessor<int32_t, 1>();
  auto ref_scores = torch::full({rows, kTopk}, -INFINITY, torch::kFloat32);
  auto ref_slots = torch::full({rows, kTopk}, -1, torch::kInt32);
  auto aligned = torch::full({rows, kTopk}, -INFINITY, torch::kFloat32);
  auto ref_values = ref_scores.accessor<float, 2>();
  auto ref_ids = ref_slots.accessor<int32_t, 2>();
  auto baseline = aligned.accessor<float, 2>();
  for (int64_t req = 0; req < x.lens.numel(); ++req) {
    for (int64_t row = cu[req]; row < cu[req + 1]; ++row) {
      const int32_t visible =
          std::max(0, std::min(positions[row] + 1, lengths[req]));
      std::vector<int64_t> local;
      std::vector<int32_t> global;
      local.reserve(visible / x.size + kPage);
      global.reserve(visible / x.size + kPage);
      for (int32_t token = 0; token < visible; ++token) {
        if (token / kPage % x.size != rank) {
          continue;
        }
        const int32_t page = table[req][token / (kPage * x.size)];
        local.emplace_back(page * kPage + token % kPage);
        global.emplace_back(page * kPage * x.size + rank * kPage +
                            token % kPage);
      }
      std::vector<std::pair<float, int32_t>> candidates;
      candidates.reserve(global.size());
      std::unordered_map<int32_t, float> by_slot;
      by_slot.reserve(global.size());
      if (!local.empty()) {
        auto indices = torch::tensor(local, torch::kInt64).to(kDevice);
        auto keys = x.cache.reshape({-1, 128}).index_select(0, indices);
        auto qk = torch::matmul(x.query[row], keys.t()).relu();
        auto weights =
            (x.weights[row].to(torch::kFloat32) / 64).to(torch::kBFloat16);
        auto scores =
            (qk.to(torch::kFloat32) * weights.to(torch::kFloat32).unsqueeze(1))
                .sum(0)
                .to(torch::kBFloat16)
                .to(torch::kFloat32)
                .cpu();
        const auto data = scores.accessor<float, 1>();
        for (size_t col = 0; col < global.size(); ++col) {
          candidates.emplace_back(data[col], global[col]);
          CHECK(by_slot.emplace(global[col], data[col]).second);
        }
      }
      if (candidates.size() > kTopk) {
        std::sort(candidates.begin(),
                  candidates.end(),
                  [](const auto& left, const auto& right) {
                    return left.first != right.first
                               ? left.first > right.first
                               : left.second < right.second;
                  });
      }
      const int64_t count = std::min<int64_t>(kTopk, candidates.size());
      std::unordered_set<int32_t> selected;
      selected.reserve(count);
      for (int64_t col = 0; col < count; ++col) {
        ref_values[row][col] = candidates[col].first;
        ref_ids[row][col] = candidates[col].second;
        auto found = by_slot.find(ids[row][col]);
        CHECK(found != by_slot.end())
            << "rank=" << rank << " row=" << row << " slot=" << ids[row][col];
        CHECK(selected.emplace(ids[row][col]).second);
        CHECK(std::isfinite(values[row][col]));
        baseline[row][col] = found->second;
        if (global.size() <= kTopk) {
          CHECK_EQ(ids[row][col], global[col]);
        } else if (col > 0) {
          CHECK_LE(values[row][col], values[row][col - 1]);
          if (values[row][col] == values[row][col - 1]) {
            CHECK_GT(ids[row][col], ids[row][col - 1]);
          }
        }
      }
      for (int64_t col = count; col < kTopk; ++col) {
        CHECK_EQ(ids[row][col], -1);
        CHECK_EQ(values[row][col], -INFINITY);
      }
    }
  }
  const auto golden =
      dcp_test::aligned_golden(x.query, x.weights, x.cache, slot, x.size);
  CHECK(dcp_test::score_precision(score, aligned.to(torch::kBFloat16), golden));
  return {ref_scores.to(torch::kBFloat16), ref_slots};
}

void check_case(Inputs x, bool prefill, bool merge = true) {
  std::vector<torch::Tensor> input_before;
  const std::vector<torch::Tensor> tensors{
      x.query, x.weights, x.cache, x.table, x.cu, x.positions, x.lens};
  input_before.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    input_before.emplace_back(tensor.cpu().clone());
  }
  std::vector<torch::Tensor> actual_scores, actual_slots, ref_scores, ref_slots;
  actual_scores.reserve(x.size);
  actual_slots.reserve(x.size);
  ref_scores.reserve(x.size);
  ref_slots.reserve(x.size);
  for (int32_t rank = 0; rank < x.size; ++rank) {
    SCOPED_TRACE(rank);
    auto actual = run(x, rank, prefill);
    auto reference = check_candidates(x, rank, actual);
    actual_scores.emplace_back(std::get<0>(actual));
    actual_slots.emplace_back(std::get<1>(actual));
    ref_scores.emplace_back(std::get<0>(reference));
    ref_slots.emplace_back(std::get<1>(reference));
  }
  for (size_t i = 0; i < tensors.size(); ++i) {
    EXPECT_TRUE(torch::equal(tensors[i].cpu(), input_before[i]));
  }
  if (!merge || x.size != 4 || x.query.size(0) == 0) {
    return;
  }
  auto mapping = torch::where(x.positions >= 0, 0, -1).to(torch::kInt32);
  auto expected =
      dcp_test::merge_reference(torch::stack(ref_scores).to(kDevice),
                                torch::stack(ref_slots).to(kDevice),
                                mapping,
                                kTopk);
  expected.first = expected.first.cpu();
  expected.second = expected.second.cpu();
  auto output = torch::empty({x.query.size(0), kTopk}, x.table.options());
  auto lens = torch::empty({x.query.size(0)}, x.table.options());
  kernel::mlu::dcp_merge_topk(torch::stack(actual_scores),
                              torch::stack(actual_slots),
                              mapping,
                              kTopk,
                              output,
                              lens);
  EXPECT_TRUE(torch::equal(lens.cpu(), expected.second));
  auto out_cpu = output.cpu();
  const auto ids = out_cpu.accessor<int32_t, 2>();
  const auto expected_ids = expected.first.accessor<int32_t, 2>();
  const auto lengths = expected.second.accessor<int32_t, 1>();
  for (int64_t row = 0; row < output.size(0); ++row) {
    std::unordered_set<int32_t> ref;
    std::unordered_set<int32_t> selected;
    ref.reserve(lengths[row]);
    selected.reserve(lengths[row]);
    for (int32_t col = 0; col < lengths[row]; ++col) {
      ref.emplace(expected_ids[row][col]);
    }
    int32_t intersection = 0;
    for (int32_t col = 0; col < lengths[row]; ++col) {
      EXPECT_TRUE(selected.emplace(ids[row][col]).second);
      intersection += ref.count(ids[row][col]);
    }
    EXPECT_GE(100 * intersection, 99 * lengths[row]) << "row=" << row;
    for (int32_t col = lengths[row]; col < kTopk; ++col) {
      EXPECT_EQ(ids[row][col], -1);
    }
  }
}

TEST(DcpIndexerTest, PrefillChunkedAndEmptyRequests) {
  torch::NoGradGuard guard;
  check_case(make_inputs({17, 0, 3}, {17, 0, 65539}), /*prefill=*/true);
  check_case(make_inputs({0, 0}, {0, 0}), /*prefill=*/true);
  check_case(make_inputs({1, 1}, {0, 0}), /*prefill=*/false);
}

TEST(DcpIndexerTest, FusedPrefillSortScratch) {
  torch::NoGradGuard guard;
  auto x = make_inputs({8192}, {65536 + 8192});
  const auto [scores, slots] = run(x, /*rank=*/0, /*prefill=*/true);
  const auto rows = torch::tensor({0, 4096, 8191}, torch::kInt64).to(kDevice);
  x.query = x.query.index_select(0, rows);
  x.weights = x.weights.index_select(0, rows);
  x.positions = x.positions.index_select(0, rows);
  x.cu = torch::tensor({0, 3}, torch::kInt32).to(kDevice);
  check_candidates(x,
                   /*rank=*/0,
                   {scores.index_select(0, rows), slots.index_select(0, rows)});
}

TEST(DcpIndexerTest, LocalVisibilityBoundaries) {
  torch::NoGradGuard guard;
  for (const int32_t count : {15,
                              16,
                              17,
                              2047,
                              2048,
                              2049,
                              8191,
                              8192,
                              8193,
                              24575,
                              24576,
                              24577,
                              32767,
                              32768,
                              32769,
                              33791,
                              33792,
                              33793}) {
    SCOPED_TRACE(count);
    // The last owner has exactly count visible local keys.
    const int32_t length = (count - 1) / 16 * 64 + 48 + (count - 1) % 16 + 1;
    check_case(make_inputs({1}, {length}), /*prefill=*/false);
  }
}

TEST(DcpIndexerTest, MtpPackedExpandedAndDraft) {
  torch::NoGradGuard guard;
  for (const int32_t query : {1, 4}) {
    auto packed = make_inputs({query, query}, {65536 + query, 65536 + query});
    check_case(packed, /*prefill=*/false);
    auto expanded = packed;
    expanded.table =
        packed.table.repeat_interleave(query, /*dim=*/0).contiguous();
    expanded.cu = torch::arange(query * 2 + 1, packed.cu.options());
    expanded.lens = packed.positions + 1;
    check_case(expanded, /*prefill=*/false);
  }
  // A small-visibility row between large rows must preserve the cached map.
  auto reused = make_inputs({65}, {131072 + 65});
  reused.positions[1] = 0;
  check_case(reused, /*prefill=*/false);
}

TEST(DcpIndexerTest, TopologiesAndSharedPages) {
  torch::NoGradGuard guard;
  for (const int32_t size : {1, 2, 4, 8}) {
    SCOPED_TRACE(size);
    auto x = make_inputs({1, 0, 3}, {9001, 0, 9003}, size);
    x.table[2].copy_(x.table[0]);
    x.positions[0] = -1;
    check_case(x, /*prefill=*/false);
  }
}

TEST(DcpIndexerTest, Fp32HeadProductsAndDeterministicTies) {
  torch::NoGradGuard guard;
  for (const bool prefill : {true, false}) {
    for (const int32_t rows : {1, 16, 2048}) {
      auto x = make_inputs({rows}, {10000});
      x.query.zero_();
      x.weights.zero_();
      x.cache.zero_();
      x.cache.select(3, 0).fill_(1);
      x.query.select(1, 0).select(1, 0).fill_(1 + 1.0 / 128);
      x.query.select(1, 1).select(1, 0).fill_(1 + 2.0 / 128);
      x.weights.select(1, 0).fill_(1 + 1.0 / 128);
      x.weights.select(1, 1).fill_(-1);
      // Only boundary rows need full oracle evaluation in the large-row
      // schedule.
      auto actual = run(x, /*rank=*/0, prefill);
      const auto& scores = std::get<0>(actual);
      const auto& slots = std::get<1>(actual);
      EXPECT_TRUE(
          scores.masked_select(slots >= 0).eq(0x1p-20f).all().item<bool>());
      if (rows < 2048) {
        check_case(x, prefill);
      } else {
        const auto indices =
            torch::tensor({0, rows - 1}, x.cu.options().dtype(torch::kInt64));
        auto sample = x;
        sample.query = x.query.index_select(0, indices);
        sample.weights = x.weights.index_select(0, indices);
        sample.positions = x.positions.index_select(0, indices);
        sample.cu = torch::tensor({0, 2}, x.cu.options());
        check_candidates(
            sample,
            /*rank=*/0,
            {scores.index_select(0, indices), slots.index_select(0, indices)});
      }
    }
  }
}

TEST(DcpIndexerTest, RoundsOutputAtBf16Midpoints) {
  torch::NoGradGuard guard;
  auto x = make_inputs({2}, {2});
  x.query.zero_();
  x.weights.zero_();
  x.cache.zero_();
  x.query.select(1, 0).select(1, 0).fill_(1);
  x.query.select(1, 1).select(1, 0).fill_(1);
  x.cache.select(3, 0).fill_(1);
  x.weights[0][0] = 1;
  x.weights[1][0] = 1 + 1.0 / 128;
  x.weights.select(1, 1).fill_(1.0 / 256);
  for (const bool prefill : {true, false}) {
    auto [scores, slots] = run(x, /*rank=*/0, prefill);
    EXPECT_EQ(scores[0][0].item<float>(), 1.0f / 64);
    EXPECT_EQ(scores[1][0].item<float>(), (1.0f + 2.0f / 128) / 64);
    check_candidates(x, /*rank=*/0, {scores, slots});
  }
}

TEST(DcpIndexerDeathTest, RejectsUnsupportedTensorContract) {
  auto x = make_inputs({1}, {65});
  auto wrong = x;
  wrong.query = x.query.to(torch::kFloat32);
  EXPECT_DEATH(run(wrong, /*rank=*/0, /*prefill=*/false), "scalar_type");
  wrong = x;
  wrong.cache = x.cache.reshape({-1, 1, 8, 128});
  EXPECT_DEATH(run(wrong, /*rank=*/0, /*prefill=*/false),
               "k_cache.size\\(2\\)");
  wrong = x;
  wrong.weights = torch::zeros({1, 64}, x.weights.options()).slice(1, 0, 64, 2);
  EXPECT_DEATH(run(wrong, /*rank=*/0, /*prefill=*/false), "contiguous");
  EXPECT_DEATH(run(x, /*rank=*/4, /*prefill=*/false), "dcp_rank");
}

TEST(DcpIndexerTest, NondefaultStreamAndGraphReplay) {
  torch::NoGradGuard guard;
  auto x = make_inputs({4}, {9004});
  auto expected = run(x, /*rank=*/0, /*prefill=*/false);
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  Result captured;
  {
    torch_mlu::mlu::MLUStreamGuard stream(torch_mlu::getStreamFromPool(
        /*isHighPriority=*/false, kDevice.index()));
    auto other = run(x, /*rank=*/0, /*prefill=*/false);
    EXPECT_TRUE(torch::equal(std::get<1>(other), std::get<1>(expected)));
    graph.capture_begin();
    captured = run(x, /*rank=*/0, /*prefill=*/false);
    graph.capture_end();
  }
  x.positions.fill_(-1);
  graph.replay();
  torch_mlu::synchronize();
  EXPECT_TRUE(std::get<1>(captured).eq(-1).all().item<bool>());
  EXPECT_TRUE(std::get<0>(captured).isneginf().all().item<bool>());
  x.lens.fill_(9003);
  x.positions.copy_(torch::arange(8999, 9003, x.positions.options()));
  x.table.copy_(x.table.flip({1}));
  x.query.mul_(-1);
  expected = run(x, /*rank=*/0, /*prefill=*/false);
  graph.replay();
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::equal(std::get<1>(captured), std::get<1>(expected)));
  EXPECT_TRUE(torch::equal(std::get<0>(captured), std::get<0>(expected)));
}

TEST(DcpIndexerPrecisionTest, RejectsWrongRoundingAndNonfiniteValues) {
  auto baseline = torch::full({128}, 0x1p-20f, torch::kBFloat16);
  auto golden = baseline.to(torch::kFloat64);
  EXPECT_TRUE(dcp_test::score_precision(baseline, baseline, golden));
  EXPECT_FALSE(
      dcp_test::score_precision(torch::zeros_like(baseline), baseline, golden));
  auto wrong = baseline.clone();
  wrong[0] = NAN;
  EXPECT_FALSE(dcp_test::score_precision(wrong, baseline, golden));
}

}  // namespace
}  // namespace xllm
