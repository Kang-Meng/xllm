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

#include <framework/core/MLUStream.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "kernels/mlu/bangc/include/kpool_bangc.h"
#include "kernels/mlu/kpool.h"
#include "platform/platform.h"

namespace xllm::kernel::mlu {
namespace {

TEST(KPoolConfigDeathTest, RejectsUnsupportedScoringBeforeLaunch) {
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const torch::Tensor query = torch::zeros({1, 32, 128}, bf16);
  const torch::Tensor weights = torch::ones({1, 32});
  const torch::Tensor cache = torch::zeros({1, 1, 4, 128}, bf16);
  const torch::Tensor table = torch::zeros({1, 1}, torch::kInt32);
  const torch::Tensor positions = torch::zeros({1}, torch::kInt32);
  const torch::Tensor rows = torch::zeros({1}, torch::kInt64);
  torch::Tensor scores = torch::zeros({1, 4});
  const auto score = [&](const torch::Tensor& q, int64_t block, int64_t pool) {
    score_kpool(
        q, weights, cache, table, positions, rows, scores, block, pool, 1.0);
  };
  EXPECT_DEATH(score(query.narrow(1, 0, 16), 16, 4), "H=32");
  EXPECT_DEATH(score(query.narrow(2, 0, 64), 16, 4), "D=128");
  EXPECT_DEATH(score(query, 64, 4), "block_size=16");
  EXPECT_DEATH(score(query, 16, 8), "pool_size=4");
}

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

// select_kpool returns the ID set torch.topk selects (ties prefer lower IDs,
// NaN and +inf rank first and become -1) in unspecified order. Few long rows
// are selected by whole clusters.
TEST(KPoolKernelTest, TopkSetHandlesTiesNonfiniteAndEmpty) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  for (const int64_t columns : {0, 17, 513, 8195, 65536, 131072, 262144}) {
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
      const torch::Tensor expected = torch::where(torch::isfinite(v), ids, -1);
      EXPECT_TRUE(torch::equal(std::get<0>(actual.narrow(1, 0, k).sort(-1)),
                               std::get<0>(expected.sort(-1))))
          << columns;
    }
    EXPECT_TRUE(actual.narrow(1, k, 512 - k).eq(-1).all().item<bool>());
  }
  EXPECT_EQ(select_kpool(torch::empty({0, 5}, opts), 3).numel(), 0);
  EXPECT_EQ(select_kpool(torch::empty({2, 5}, opts), 0).numel(), 0);
}

// A selection boundary at exactly 0 with negatives below: the threshold
// search must reach 0 itself (bisecting values toward 0 stalls), otherwise
// zeros displace positive scores.
TEST(KPoolKernelTest, TopkBoundaryAtZeroKeepsPositiveScores) {
  torch::manual_seed(7);
  const torch::Device device(Platform::type_torch(), 0);
  const auto opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  for (const int64_t pools : {1000, 2048, 65536}) {
    const int64_t rows = 4;
    // Fewer than 512 positives, then exact zeros, then negatives.
    const torch::Tensor x = torch::randn({rows, pools}, opts);
    torch::Tensor scores = torch::where(
        x.gt(0.3),
        x.abs() + 1,
        torch::where(x.gt(-0.3), torch::zeros_like(x), -x.abs() - 1));
    const int64_t live = std::min<int64_t>(pools, 1000);
    scores.narrow(1, live, pools - live)
        .fill_(-std::numeric_limits<float>::infinity());
    const torch::Tensor positions =
        torch::full({rows}, live * 4 - 1, opts.dtype(torch::kInt32));
    const torch::Tensor expected =
        std::get<0>(std::get<0>(scores.topk(512, -1)).sort(-1));
    for (const torch::Tensor& ids :
         {select_kpool(scores, 512),
          select_kpool_live(scores, 512, positions, /*pool_size=*/4)}) {
      EXPECT_TRUE(
          torch::equal(std::get<0>(scores.gather(1, ids).sort(-1)), expected))
          << pools;
    }
  }
}

TEST(KPoolKernelTest, LiveTopkMatchesFullRowSelection) {
  torch::manual_seed(40717);
  const torch::Device device(Platform::type_torch(), 0);
  const auto opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const float inf = std::numeric_limits<float>::infinity();
  for (const int64_t pools : {17, 513, 8195, 40000, 65536, 70001, 131072}) {
    const int64_t rows = 37;
    // Coarse values leave many ties at the selection boundary.
    torch::Tensor scores = torch::round(torch::randn({rows, pools}, opts) * 8);
    torch::Tensor positions =
        torch::randint(-1, pools * 4 + 9, {rows}, opts.dtype(torch::kInt32));
    positions[0] = -1;
    positions[1] = static_cast<int32_t>(pools * 4 - 1);
    positions[2] = 2;
    const torch::Tensor live =
        (positions.to(torch::kInt64) + 1).div(4, "floor").clamp(0, pools);
    // score_kpool writes -inf past each row's live prefix.
    scores.masked_fill_(torch::arange(pools, opts.dtype(torch::kInt64))
                            .unsqueeze(0)
                            .ge(live.unsqueeze(1)),
                        -inf);
    scores[1][0] = std::numeric_limits<float>::quiet_NaN();
    scores[1][pools - 1] = inf;
    scores[3].narrow(0, 0, 1).fill_(inf);
    const torch::Tensor expected = select_kpool(scores, 512);
    const torch::Tensor actual =
        select_kpool_live(scores, 512, positions, /*pool_size=*/4);
    EXPECT_TRUE(torch::equal(std::get<0>(actual.sort(-1)),
                             std::get<0>(expected.sort(-1))))
        << pools;
  }
}

// Long and mixed rows exercise every selection path: short rows (one row per
// core), a few long rows per cluster, many long rows streamed per core, mixed
// batches, NaN/+inf in long rows, and rows without positions (every pool).
TEST(KPoolKernelTest, LongAndMixedRowsMatchTorch) {
  torch::manual_seed(90211);
  const torch::Device device(Platform::type_torch(), 0);
  const auto opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const float inf = std::numeric_limits<float>::infinity();
  struct Shape {
    int64_t rows;
    int64_t pools;
    bool mixed;
    bool nonfinite;
  };
  const std::vector<Shape> shapes = {{1, 2048, false, false},
                                     {40, 8195, false, false},
                                     {4, 262144, false, false},
                                     {4, 262144, false, true},
                                     {40, 65536, false, false},
                                     {40, 65536, false, true},
                                     {64, 131072, true, false},
                                     {9, 40000, true, true}};
  for (const Shape& shape : shapes) {
    torch::Tensor scores = torch::randn({shape.rows, shape.pools}, opts);
    torch::Tensor positions = torch::full(
        {shape.rows}, shape.pools * 4 - 1, opts.dtype(torch::kInt32));
    if (shape.mixed) {
      // One full row, the others short; one row inactive.
      positions.narrow(0, 1, shape.rows - 1)
          .copy_(torch::randint(-1,
                                std::min<int64_t>(shape.pools, 16384) * 4,
                                {shape.rows - 1},
                                opts.dtype(torch::kInt32)));
      positions[1] = -1;
    }
    const torch::Tensor live =
        (positions.to(torch::kInt64) + 1).div(4, "floor").clamp(0, shape.pools);
    scores.masked_fill_(torch::arange(shape.pools, opts.dtype(torch::kInt64))
                            .unsqueeze(0)
                            .ge(live.unsqueeze(1)),
                        -inf);
    if (shape.nonfinite) {
      scores[0][3] = std::numeric_limits<float>::quiet_NaN();
      scores[0][shape.pools / 2] = inf;
    }
    const int64_t k = std::min<int64_t>(512, shape.pools);
    const auto [values, ids] = scores.topk(k, -1);
    torch::Tensor expected = torch::full({shape.rows, 512}, -1, ids.options());
    expected.narrow(1, 0, k).copy_(
        torch::where(torch::isfinite(values), ids, -1));
    expected = std::get<0>(expected.sort(-1));
    const torch::Tensor live_ids =
        select_kpool_live(scores, 512, positions, /*pool_size=*/4);
    EXPECT_TRUE(torch::equal(std::get<0>(live_ids.sort(-1)), expected))
        << shape.rows << "x" << shape.pools;
    if (!shape.mixed) {
      EXPECT_TRUE(torch::equal(std::get<0>(select_kpool(scores, 512).sort(-1)),
                               expected))
          << shape.rows << "x" << shape.pools;
    }
  }
}

TEST(KPoolKernelTest, LiveTopkRespectsPoolSize) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  // With pool size 2 and position 2047, all 1024 pools are live. Assuming
  // pool size 4 would silently select IDs 0..511 instead of 512..1023.
  const torch::Tensor scores = torch::arange(1024, opts).repeat({9, 1});
  const torch::Tensor expected =
      torch::arange(512, 1024, opts.dtype(torch::kInt64)).repeat({9, 1});
  for (const int64_t pool_size : {1, 2, 4, 8}) {
    const torch::Tensor positions =
        torch::full({9}, 1024 * pool_size - 1, opts.dtype(torch::kInt32));
    const torch::Tensor actual =
        select_kpool_live(scores, /*count=*/512, positions, pool_size);
    EXPECT_TRUE(torch::equal(std::get<0>(actual.sort(-1)), expected))
        << "pool_size=" << pool_size;
  }
}

TEST(KPoolKernelTest, PrefillScoresMatchPerQueryScoring) {
  torch::manual_seed(52811);
  const torch::Device device(Platform::type_torch(), 0);
  const auto bf16 =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const auto fp32 = bf16.dtype(torch::kFloat32);
  const auto ints = bf16.dtype(torch::kInt32);
  const std::vector<std::vector<int64_t>> batches = {
      {300}, {257, 37, 1, 129}, {64, 64}};
  const std::vector<std::vector<int64_t>> histories = {
      {1000}, {0, 5000, 3, 64}, {16, 2049}};
  for (size_t b = 0; b < batches.size(); ++b) {
    const std::vector<int64_t>& lengths = batches[b];
    const int64_t requests = static_cast<int64_t>(lengths.size());
    std::vector<torch::Tensor> position_parts;
    std::vector<torch::Tensor> row_parts;
    int64_t capacity = 0;
    for (int64_t r = 0; r < requests; ++r) {
      const int64_t kv = histories[b][r] + lengths[r];
      capacity = std::max(capacity, kv);
      position_parts.emplace_back(torch::arange(kv - lengths[r], kv, ints));
      row_parts.emplace_back(torch::full({lengths[r]}, r, ints));
    }
    torch::Tensor positions = torch::cat(position_parts);
    positions[1] = -1;
    const torch::Tensor rows = torch::cat(row_parts);
    const int64_t tokens = positions.numel();
    const int64_t width = (capacity + 15) / 16;
    torch::Tensor table =
        torch::randperm(requests * width, ints).view({requests, width});
    table.index_put_({0, std::min<int64_t>(1, width - 1)}, -1);
    const torch::Tensor cache =
        torch::randn({requests * width, 1, 4, 128}, bf16);
    const torch::Tensor query = torch::randn({tokens, 32, 128}, bf16);
    const torch::Tensor weights = torch::randn({tokens, 32}, fp32);
    const int64_t pools = (capacity + 3) / 4;
    // More than eight queries per request take the prefill kernel; one
    // query at a time takes the decode kernel.
    torch::Tensor batched = torch::empty({tokens, pools}, fp32);
    score_kpool(query,
                weights,
                cache,
                table,
                positions,
                rows,
                batched,
                /*block_size=*/16,
                /*pool_size=*/4,
                /*scale=*/0.015625);
    torch::Tensor single = torch::empty({tokens, pools}, fp32);
    for (int64_t t = 0; t < tokens; ++t) {
      torch::Tensor out = single.narrow(0, t, 1);
      score_kpool(query.narrow(0, t, 1),
                  weights.narrow(0, t, 1),
                  cache,
                  table,
                  positions.narrow(0, t, 1),
                  rows.narrow(0, t, 1),
                  out,
                  /*block_size=*/16,
                  /*pool_size=*/4,
                  /*scale=*/0.015625);
    }
    EXPECT_TRUE(torch::equal(batched, single)) << b;
  }
}

void check_verify_scores(int64_t requests, int64_t capacity) {
  torch::manual_seed(19023);
  const torch::Device device(Platform::type_torch(), 0);
  const auto bf16 =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const auto fp32 = bf16.dtype(torch::kFloat32);
  const auto ints = bf16.dtype(torch::kInt32);
  const int64_t tokens = requests * 4;
  const int64_t pages_per_request = (capacity + 3) / 4;
  const int64_t pages = requests * pages_per_request;
  const torch::Tensor query = torch::randn({tokens, 32, 128}, bf16);
  const torch::Tensor weights = torch::randn({tokens, 32}, fp32);
  const torch::Tensor cache = torch::randn({pages, 1, 4, 128}, bf16);
  const torch::Tensor query_before = query.clone();
  const torch::Tensor weights_before = weights.clone();
  const torch::Tensor cache_before = cache.clone();
  torch::Tensor table =
      torch::arange(pages, ints).flip({0}).view({requests, pages_per_request});
  table.index_put_({0, 0}, -1);
  table.index_put_({1, 1}, pages + 3);
  torch::Tensor positions =
      (torch::arange(4, ints) + capacity * 4 - 4).repeat({requests});
  positions.index_put_({0}, -1);
  const torch::Tensor rows = torch::arange(requests, ints).repeat_interleave(4);
  const torch::Tensor starts = torch::arange(requests + 1, ints) * 4;
  torch::Tensor expected = torch::empty({tokens, capacity}, fp32);
  torch::Tensor storage = torch::full({tokens, capacity + 2}, 99.0, fp32);
  torch::Tensor actual = storage.narrow(/*dim=*/1, /*start=*/1, capacity);
  score_kpool(query,
              weights,
              cache,
              table,
              positions,
              rows,
              expected,
              /*block_size=*/16,
              /*pool_size=*/4,
              /*scale=*/0.015625);
  score_kpool(query,
              weights,
              cache,
              table,
              positions,
              rows,
              actual,
              /*block_size=*/16,
              /*pool_size=*/4,
              /*scale=*/0.015625,
              starts);
  EXPECT_TRUE(torch::equal(torch::isneginf(actual), torch::isneginf(expected)));
  const torch::Tensor finite = torch::isfinite(expected);
  EXPECT_TRUE(torch::allclose(actual.masked_select(finite),
                              expected.masked_select(finite),
                              /*rtol=*/1e-4,
                              /*atol=*/1e-4));
  EXPECT_TRUE(torch::equal(select_kpool(actual, /*count=*/512),
                           select_kpool(expected, /*count=*/512)));
  EXPECT_TRUE(storage.select(1, 0).eq(99.0).all().item<bool>());
  EXPECT_TRUE(storage.select(1, capacity + 1).eq(99.0).all().item<bool>());
  EXPECT_TRUE(torch::equal(query, query_before));
  EXPECT_TRUE(torch::equal(weights, weights_before));
  EXPECT_TRUE(torch::equal(cache, cache_before));
}

TEST(KPoolKernelTest, VerifyScoresPreserveFp32WeightsPagesAndCausality) {
  check_verify_scores(/*requests=*/3, /*capacity=*/137);
}

TEST(KPoolKernelTest, VerifyScoresMatchAtBatch16Context64K) {
  check_verify_scores(/*requests=*/16, /*capacity=*/16384);
}

TEST(KPoolKernelTest, VerifyScoresMatchAtCapacityBoundaries) {
  for (const int64_t capacity : {4095, 4096, 4097, 32767, 32768, 32769}) {
    SCOPED_TRACE(capacity);
    check_verify_scores(/*requests=*/16, capacity);
  }
}

TEST(KPoolKernelTest, VerifyScoresMatchAtAdjacentBatchSizes) {
  for (const int64_t requests : {15, 17}) {
    SCOPED_TRACE(requests);
    check_verify_scores(requests, /*capacity=*/4097);
  }
}

TEST(KPoolKernelTest, VerifyGraphReadsLiveOffsetsAndClearsUnassignedPadding) {
  torch::NoGradGuard no_grad;
  torch::manual_seed(8221);
  const torch::Device device(Platform::type_torch(), 0);
  const auto bf16 =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const auto fp32 = bf16.dtype(torch::kFloat32);
  const auto ints = bf16.dtype(torch::kInt64);
  const torch::Tensor query = torch::randn({12, 32, 128}, bf16);
  const torch::Tensor weights = torch::randn({12, 32}, fp32);
  const torch::Tensor cache = torch::randn({12, 1, 4, 128}, bf16);
  torch::Tensor table = torch::arange(12, ints).view({3, 4});
  torch::Tensor positions = torch::full({12}, 63, ints);
  torch::Tensor rows = torch::arange(3, ints).repeat_interleave(4);
  torch::Tensor starts = torch::tensor({0, 4, 8, 12}, ints);
  torch::Tensor storage = torch::full({12, 18}, 99.0, fp32);
  torch::Tensor actual = storage.narrow(/*dim=*/1, /*start=*/1, /*length=*/16);
  torch::Tensor expected = torch::empty({12, 16}, fp32);
  const auto score_verify = [&]() {
    score_kpool(query,
                weights,
                cache,
                table,
                positions,
                rows,
                actual,
                /*block_size=*/16,
                /*pool_size=*/4,
                /*scale=*/0.015625,
                starts);
  };
  score_verify();
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  {
    torch_mlu::mlu::MLUStreamGuard guard(
        torch_mlu::getStreamFromPool(/*isHighPriority=*/false,
                                     /*device_index=*/0));
    graph.capture_begin();
    score_verify();
    graph.capture_end();
  }
  // The middle request becomes empty, the last request moves to row two,
  // and the fixed graph bucket retains six completely unassigned rows.
  starts.copy_(torch::tensor({0, 2, 2, 6}, ints));
  rows.copy_(torch::tensor({0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0}, ints));
  positions.copy_(
      torch::tensor({3, 4, 7, 8, 11, 12, -1, -1, -1, -1, -1, -1}, ints));
  table.copy_(table.flip({1}));
  graph.replay();
  torch_mlu::synchronize();
  score_kpool(query,
              weights,
              cache,
              table,
              positions,
              rows,
              expected,
              /*block_size=*/16,
              /*pool_size=*/4,
              /*scale=*/0.015625);
  EXPECT_TRUE(torch::equal(torch::isneginf(actual), torch::isneginf(expected)));
  const torch::Tensor finite = torch::isfinite(expected);
  EXPECT_TRUE(torch::allclose(actual.masked_select(finite),
                              expected.masked_select(finite),
                              /*rtol=*/1e-4,
                              /*atol=*/1e-4));
  EXPECT_TRUE(storage.select(1, 0).eq(99.0).all().item<bool>());
  EXPECT_TRUE(storage.select(1, 17).eq(99.0).all().item<bool>());
}
TEST(KPoolKernelTest, BangcRowsUseTensorDeviceAndRestoreCurrentDevice) {
  if (Platform::device_count() < 2) {
    GTEST_SKIP() << "Two visible MLU devices are required";
  }
  const torch::Device caller_device(Platform::type_torch(), 0);
  const torch::Device tensor_device(Platform::type_torch(), 1);
  const torch::DeviceGuard caller_guard(caller_device);
  const auto options =
      torch::TensorOptions().dtype(torch::kInt64).device(tensor_device);
  const torch::Tensor starts = torch::arange(17, options) * 64;
  torch::Tensor rows = torch::full({1024}, -7, options);
  EXPECT_EQ(torch_mlu::current_device(), 0);
  ASSERT_TRUE(bangc_kpool_rows(starts, rows));
  EXPECT_EQ(torch_mlu::current_device(), 0);
  const torch::Tensor expected =
      torch::arange(16, torch::kInt64).repeat_interleave(64);
  EXPECT_TRUE(torch::equal(rows.cpu(), expected));
}

TEST(KPoolKernelTest, BangcRowsRejectUnsupportedMetadataBeforeMutation) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kInt64).device(device);
  const torch::Tensor backing = torch::arange(34, options) * 32;
  const torch::Tensor starts = backing.slice(/*dim=*/0,
                                             /*start=*/0,
                                             /*end=*/34,
                                             /*step=*/2);
  torch::Tensor rows = torch::full({1024}, -7, options);
  EXPECT_FALSE(bangc_kpool_rows(starts, rows));
  EXPECT_TRUE(rows.eq(-7).all().item<bool>());
}

TEST(KPoolKernelTest, BangcRowsRejectDtypeLayoutAndEmptyMetadata) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kInt64).device(device);
  const torch::Tensor starts = torch::arange(17, options) * 64;
  const auto expect_rejected = [&](const torch::Tensor& metadata,
                                   torch::Tensor output) {
    const torch::Tensor before = output.clone();
    EXPECT_FALSE(bangc_kpool_rows(metadata, output));
    EXPECT_TRUE(torch::equal(output, before));
  };
  expect_rejected(starts.to(torch::kFloat32), torch::full({1024}, -7, options));
  expect_rejected(starts,
                  torch::full({1024}, -7, options.dtype(torch::kInt32)));
  expect_rejected(starts.unsqueeze(0), torch::full({1024}, -7, options));
  expect_rejected(torch::zeros({1}, options), torch::full({1024}, -7, options));
  expect_rejected(starts, torch::empty({0}, options));
  expect_rejected(starts,
                  torch::full({2048}, -7, options).slice(0, 0, 2048, 2));
  expect_rejected(starts.cpu(), torch::full({1024}, -7, options));
}

TEST(KPoolKernelTest, BangcScoresRejectUnsupportedInputsBeforeMutation) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options = torch::TensorOptions().device(device);
  const std::vector<torch::Tensor> inputs = {
      torch::zeros({2, 32, 128}, options.dtype(torch::kBFloat16)),
      torch::ones({2, 32}, options.dtype(torch::kFloat32)),
      torch::zeros({2, 1, 4, 128}, options.dtype(torch::kBFloat16)),
      torch::zeros({2, 2}, options.dtype(torch::kInt32)),
      torch::full({2}, 7, options.dtype(torch::kInt32)),
      torch::arange(2, options.dtype(torch::kInt64)),
      torch::full({2, 8}, 97, options.dtype(torch::kFloat32))};
  const auto expect_rejected = [&](int64_t index, const torch::Tensor& bad) {
    std::vector<torch::Tensor> arguments = inputs;
    arguments[index] = bad;
    const torch::Tensor before = arguments[6].clone();
    EXPECT_FALSE(bangc_kpool_score(arguments[0],
                                   arguments[1],
                                   arguments[2],
                                   arguments[3],
                                   arguments[4],
                                   arguments[5],
                                   arguments[6],
                                   16,
                                   4,
                                   0.125,
                                   torch::Tensor()));
    EXPECT_TRUE(torch::equal(arguments[6], before));
  };
  expect_rejected(0, inputs[0].to(torch::kFloat32));
  expect_rejected(1, inputs[1].to(torch::kBFloat16));
  expect_rejected(2, inputs[2].to(torch::kFloat32));
  // Integer metadata widths are supported by the typed preparation path.
  const auto expect_supported_metadata = [&](int64_t index,
                                             const torch::Tensor& metadata) {
    std::vector<torch::Tensor> arguments = inputs;
    arguments[index] = metadata;
    arguments[6] = torch::full_like(inputs[6], 97);
    ASSERT_TRUE(bangc_kpool_score(arguments[0],
                                  arguments[1],
                                  arguments[2],
                                  arguments[3],
                                  arguments[4],
                                  arguments[5],
                                  arguments[6],
                                  16,
                                  4,
                                  0.125,
                                  torch::Tensor()));
    torch::Tensor expected =
        torch::full({2, 8}, -std::numeric_limits<float>::infinity());
    expected.narrow(1, 0, 2).zero_();
    EXPECT_TRUE(torch::equal(arguments[6].cpu(), expected));
  };
  expect_supported_metadata(3, inputs[3].to(torch::kInt64));
  expect_supported_metadata(4, inputs[4].to(torch::kInt64));
  expect_supported_metadata(5, inputs[5].to(torch::kInt32));
  expect_rejected(6, inputs[6].to(torch::kBFloat16));
  expect_rejected(0,
                  torch::zeros({2, 32, 256}, options.dtype(torch::kBFloat16))
                      .slice(2, 0, 256, 2));
  expect_rejected(
      6,
      torch::full({2, 10}, 97, options.dtype(torch::kFloat32)).narrow(1, 1, 8));
  expect_rejected(1, inputs[1].cpu());
}

TEST(KPoolKernelTest, NonVerifyRaggedQueriesKeepRequestOwnership) {
  const torch::Device device(Platform::type_torch(), 0);
  torch::manual_seed(20261019);
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  for (const std::vector<int64_t>& pair : {std::vector<int64_t>{1, 7},
                                           std::vector<int64_t>{4, 4},
                                           std::vector<int64_t>{1, 5},
                                           std::vector<int64_t>{3, 3}}) {
    SCOPED_TRACE(::testing::Message()
                 << "lengths=" << pair[0] << "," << pair[1]);
    std::vector<int64_t> request_rows;
    std::vector<int32_t> query_positions;
    for (int64_t request = 0; request < 8; ++request) {
      const int64_t length = pair[request % 2];
      for (int64_t offset = 0; offset < length; ++offset) {
        request_rows.push_back(request);
        query_positions.push_back(static_cast<int32_t>(32 - length + offset));
      }
    }
    const int64_t tokens = static_cast<int64_t>(request_rows.size());
    const torch::Tensor query = torch::randn({tokens, 32, 128}, bf16);
    const torch::Tensor weights = torch::randn({tokens, 32});
    const torch::Tensor cache = torch::randn({16, 1, 4, 128}, bf16);
    const torch::Tensor table =
        torch::arange(16, torch::kInt32).reshape({8, 2});
    const torch::Tensor positions =
        torch::tensor(query_positions, torch::kInt32);
    const torch::Tensor rows = torch::tensor(request_rows, torch::kInt64);
    torch::Tensor expected =
        torch::full({tokens, 8}, -std::numeric_limits<float>::infinity());
    for (int64_t token = 0; token < tokens; ++token) {
      for (int64_t pool = 0; pool < 8; ++pool) {
        if ((pool + 1) * 4 > query_positions[token] + 1) {
          continue;
        }
        const int64_t page = request_rows[token] * 2 + pool / 4;
        const torch::Tensor dot = (query[token].to(torch::kFloat32) *
                                   cache[page][0][pool % 4].to(torch::kFloat32))
                                      .sum(1);
        expected[token][pool] =
            (torch::relu(dot * 0.125) * weights[token]).sum();
      }
    }
    torch::Tensor storage =
        torch::full({tokens + 2, 8}, 97.0, expected.options().device(device));
    torch::Tensor output = storage.narrow(0, 1, tokens);
    score_kpool(query.to(device),
                weights.to(device),
                cache.to(device),
                table.to(device),
                positions.to(device),
                rows.to(device),
                output,
                16,
                4,
                0.125);
    const torch::Tensor actual = output.cpu();
    EXPECT_TRUE(
        torch::equal(torch::isneginf(actual), torch::isneginf(expected)));
    const torch::Tensor finite = torch::isfinite(expected);
    EXPECT_TRUE(torch::allclose(actual.masked_select(finite),
                                expected.masked_select(finite),
                                1e-4,
                                1e-4));
    EXPECT_TRUE(storage[0].eq(97.0).all().item<bool>());
    EXPECT_TRUE(storage[tokens + 1].eq(97.0).all().item<bool>());
  }
}

TEST(KPoolKernelTest, OptionalOffsetsDoNotOverrideEachQueriesTableRow) {
  const torch::Device device(Platform::type_torch(), 0);
  torch::manual_seed(20261025);
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  constexpr int64_t kBatch = 3;
  constexpr int64_t kCapacity = 8;
  const torch::Tensor table_cpu =
      torch::tensor({{0, 1}, {2, 3}, {4, 5}}, torch::kInt32);
  const torch::Tensor cache_cpu = torch::randn({6, 1, 4, 128}, bf16);
  for (const int64_t tokens : {4, 12}) {
    const torch::Tensor query_cpu = torch::randn({tokens, 32, 128}, bf16);
    const torch::Tensor weights_cpu = torch::randn({tokens, 32}) * 0.1;
    const torch::Tensor rows_cpu =
        torch::arange(tokens, torch::kInt64).remainder(kBatch);
    torch::Tensor positions_cpu = torch::full({tokens}, 31, torch::kInt32);
    positions_cpu[0] = -1;
    const torch::Tensor starts_cpu =
        tokens == 4 ? torch::tensor({0, 2, 3, 4}, torch::kInt64)
                    : torch::tensor({0, 2, 7, 12}, torch::kInt64);
    torch::Tensor expected = torch::full(
        {tokens, kCapacity}, -std::numeric_limits<float>::infinity());
    for (int64_t token = 1; token < tokens; ++token) {
      const int64_t row = rows_cpu[token].item<int64_t>();
      for (int64_t pool = 0; pool < kCapacity; ++pool) {
        const int64_t page = table_cpu[row][pool / 4].item<int32_t>();
        const torch::Tensor dots =
            (query_cpu[token].to(torch::kFloat64) *
             cache_cpu[page][0][pool % 4].to(torch::kFloat64))
                .sum(1);
        expected[token][pool] =
            (torch::relu(dots * 0.125) * weights_cpu[token].to(torch::kFloat64))
                .sum();
      }
    }
    for (const bool wide_metadata : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "tokens=" << tokens
                                        << " wide_metadata=" << wide_metadata);
      const torch::ScalarType integer_type =
          wide_metadata ? torch::kInt64 : torch::kInt32;
      const torch::Tensor query = query_cpu.to(device);
      const torch::Tensor weights = weights_cpu.to(device);
      const torch::Tensor cache = cache_cpu.to(device);
      const torch::Tensor rows = rows_cpu.to(device);
      const torch::Tensor table = table_cpu.to(integer_type).to(device);
      const torch::Tensor positions = positions_cpu.to(integer_type).to(device);
      const torch::Tensor starts = starts_cpu.to(device);
      torch::Tensor storage =
          torch::full({tokens + 2, kCapacity}, 97, weights.options());
      torch::Tensor output = storage.narrow(0, 1, tokens);
      score_kpool(query,
                  weights,
                  cache,
                  table,
                  positions,
                  rows,
                  output,
                  16,
                  4,
                  0.125,
                  starts);
      const torch::Tensor actual = output.cpu();
      EXPECT_TRUE(
          torch::equal(torch::isneginf(actual), torch::isneginf(expected)));
      const torch::Tensor finite = torch::isfinite(expected);
      EXPECT_TRUE(torch::allclose(actual.masked_select(finite),
                                  expected.masked_select(finite),
                                  1e-4,
                                  1e-4));
      EXPECT_TRUE(torch::equal(select_kpool(output, 512),
                               select_kpool(expected.to(device), 512)));
      EXPECT_TRUE(storage[0].eq(97).all().item<bool>());
      EXPECT_TRUE(storage[-1].eq(97).all().item<bool>());
      EXPECT_TRUE(torch::equal(query.cpu(), query_cpu));
      EXPECT_TRUE(torch::equal(weights.cpu(), weights_cpu));
      EXPECT_TRUE(torch::equal(cache.cpu(), cache_cpu));
      EXPECT_TRUE(torch::equal(rows.cpu(), rows_cpu));
      EXPECT_TRUE(torch::equal(table.cpu(), table_cpu.to(integer_type)));
      EXPECT_TRUE(
          torch::equal(positions.cpu(), positions_cpu.to(integer_type)));
      EXPECT_TRUE(torch::equal(starts.cpu(), starts_cpu));
    }
  }
}

TEST(KPoolKernelTest, FullWidthInvalidRowsRemainMaskedBeforeTableAccess) {
  const torch::Device device(Platform::type_torch(), 0);
  torch::manual_seed(20261026);
  constexpr int64_t kTokens = 12;
  constexpr int64_t kCapacity = 8;
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const torch::Tensor table_cpu =
      torch::tensor({{0, 1}, {2, 3}, {4, 5}}, torch::kInt32);
  const torch::Tensor rows_cpu = torch::tensor(
      std::vector<int64_t>{
          -1LL,
          3LL,
          static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
          1LL,
          std::numeric_limits<int64_t>::min(),
          -(int64_t{1} << 32) + 1,
          int64_t{1} << 32,
          std::numeric_limits<int64_t>::max(),
          0LL,
          -1LL,
          3LL,
          1LL},
      torch::kInt64);
  const torch::Tensor cache_cpu = torch::randn({6, 1, 4, 128}, bf16);
  for (const int64_t heads : {32}) {
    const torch::Tensor query_cpu = torch::randn({kTokens, heads, 128}, bf16);
    const torch::Tensor weights_cpu = torch::randn({kTokens, heads}) * 0.1;
    torch::Tensor expected = torch::full(
        {kTokens, kCapacity}, -std::numeric_limits<float>::infinity());
    for (const int64_t token : {3, 8, 11}) {
      const int64_t row = rows_cpu[token].item<int64_t>();
      for (int64_t pool = 0; pool < kCapacity; ++pool) {
        const int64_t page = table_cpu[row][pool / 4].item<int32_t>();
        const torch::Tensor dots =
            (query_cpu[token].to(torch::kFloat64) *
             cache_cpu[page][0][pool % 4].to(torch::kFloat64))
                .sum(1);
        expected[token][pool] =
            (torch::relu(dots * 0.125) * weights_cpu[token].to(torch::kFloat64))
                .sum();
      }
    }
    for (const bool wide_metadata : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "heads=" << heads << " wide=" << wide_metadata);
      const torch::ScalarType dtype =
          wide_metadata ? torch::kInt64 : torch::kInt32;
      const torch::Tensor query = query_cpu.to(device);
      const torch::Tensor weights = weights_cpu.to(device);
      const torch::Tensor cache = cache_cpu.to(device);
      const torch::Tensor table = table_cpu.to(dtype).to(device);
      const torch::Tensor rows = rows_cpu.to(device);
      const torch::Tensor positions =
          torch::full({kTokens}, 31, table.options());
      const torch::Tensor starts =
          torch::tensor({0, 2, 7, 12}, table.options().dtype(torch::kInt64));
      torch::Tensor storage =
          torch::full({kTokens + 2, kCapacity}, 97, weights.options());
      torch::Tensor output = storage.narrow(0, 1, kTokens);
      score_kpool(query,
                  weights,
                  cache,
                  table,
                  positions,
                  rows,
                  output,
                  16,
                  4,
                  0.125,
                  starts);
      const torch::Tensor actual = output.cpu();
      EXPECT_TRUE(
          torch::equal(torch::isneginf(actual), torch::isneginf(expected)));
      const torch::Tensor finite = torch::isfinite(expected);
      EXPECT_TRUE(torch::allclose(actual.masked_select(finite),
                                  expected.masked_select(finite),
                                  1e-4,
                                  1e-4));
      EXPECT_TRUE(torch::equal(select_kpool(output, 512),
                               select_kpool(expected.to(device), 512)));
      EXPECT_TRUE(storage[0].eq(97).all().item<bool>());
      EXPECT_TRUE(storage[-1].eq(97).all().item<bool>());
      EXPECT_TRUE(torch::equal(query.cpu(), query_cpu));
      EXPECT_TRUE(torch::equal(weights.cpu(), weights_cpu));
      EXPECT_TRUE(torch::equal(cache.cpu(), cache_cpu));
      EXPECT_TRUE(torch::equal(table.cpu(), table_cpu.to(dtype)));
      EXPECT_TRUE(torch::equal(rows.cpu(), rows_cpu));
      EXPECT_TRUE(positions.eq(31).all().item<bool>());
    }
  }
}

// Full production launches, with an independent CPU formula. Long prefill
// checks sampled rows (including request/tile ends) to bound reference cost;
// short decode/verify checks every row. All requests own disjoint physical
// pages.
void check_budgeted_scores(const std::vector<int64_t>& lengths,
                           const std::vector<int64_t>& histories) {
  ASSERT_EQ(lengths.size(), histories.size());
  torch::manual_seed(20261026);
  const torch::Device device(Platform::type_torch(), 0);
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const auto ints = torch::TensorOptions().dtype(torch::kInt64);
  const int64_t batch = static_cast<int64_t>(lengths.size());
  std::vector<int64_t> offsets{0};
  offsets.reserve(batch + 1);
  int64_t capacity = 0;
  std::vector<torch::Tensor> position_parts;
  std::vector<torch::Tensor> row_parts;
  position_parts.reserve(batch);
  row_parts.reserve(batch);
  for (int64_t request = 0; request < batch; ++request) {
    offsets.emplace_back(offsets.back() + lengths[request]);
    capacity = std::max(capacity, histories[request] + lengths[request]);
    position_parts.emplace_back(torch::arange(
        histories[request], histories[request] + lengths[request], ints));
    row_parts.emplace_back(torch::full({lengths[request]}, request, ints));
  }
  const int64_t tokens = offsets.back();
  const int64_t pages_per_request = (capacity + 15) / 16;
  const int64_t pools = pages_per_request * 4;
  const torch::Tensor table = torch::randperm(batch * pages_per_request, ints)
                                  .view({batch, pages_per_request});
  const torch::Tensor cache =
      torch::randn({batch * pages_per_request, 1, 4, 128}, bf16);
  const torch::Tensor query = torch::randn({tokens, 32, 128}, bf16);
  const torch::Tensor weights = torch::randn({tokens, 32});
  const torch::Tensor positions = torch::cat(position_parts);
  torch::Tensor storage =
      torch::full({tokens * pools + 128},
                  97.0,
                  torch::TensorOptions().dtype(torch::kFloat32).device(device));
  torch::Tensor scores =
      storage.narrow(0, 64, tokens * pools).view({tokens, pools});
  ASSERT_TRUE(scores.is_contiguous());
  score_kpool(query.to(device),
              weights.to(device),
              cache.to(device),
              table.to(device, torch::kInt32),
              positions.to(device),
              torch::cat(row_parts).to(device),
              scores,
              /*block_size=*/16,
              /*pool_size=*/4,
              /*scale=*/1.0 / 64,
              torch::tensor(offsets, ints).to(device));
  EXPECT_TRUE(storage.narrow(0, 0, 64).eq(97).all().item<bool>());
  EXPECT_TRUE(
      storage.narrow(0, 64 + tokens * pools, 64).eq(97).all().item<bool>());
  const torch::Tensor columns = torch::arange(pools, ints);
  for (int64_t request = 0; request < batch; ++request) {
    SCOPED_TRACE(request);
    const torch::Tensor keys = cache.index_select(0, table[request])
                                   .view({pools, 128})
                                   .to(torch::kFloat32);
    std::vector<int64_t> samples;
    samples.reserve(32);
    const int64_t stride = std::max<int64_t>(1, lengths[request] / 16);
    for (int64_t local = 0; local < lengths[request]; local += stride) {
      samples.emplace_back(local);
    }
    samples.emplace_back(lengths[request] - 1);
    for (const int64_t local : samples) {
      const int64_t token = offsets[request] + local;
      SCOPED_TRACE(token);
      const torch::Tensor expected =
          (torch::relu(torch::matmul(query[token].to(torch::kFloat32),
                                     keys.transpose(0, 1)) /
                       64) *
           weights[token].unsqueeze(1))
              .sum(0);
      const torch::Tensor valid =
          (columns + 1) * 4 <= histories[request] + local + 1;
      const torch::Tensor actual = scores[token].cpu();
      EXPECT_TRUE(torch::equal(torch::isneginf(actual), ~valid));
      EXPECT_TRUE(torch::allclose(actual.masked_select(valid),
                                  expected.masked_select(valid),
                                  /*rtol=*/1e-4,
                                  /*atol=*/1e-4));
    }
  }
}

TEST(KPoolKernelTest, BudgetedPrefillScoresMatchIndependentTorch) {
  // Explicit M=32K kernel capability budget, independent of serving defaults.
  const std::vector<std::vector<int64_t>> batches = {{8192},
                                                     {16384},
                                                     {8192, 8192},
                                                     {8192, 8192, 8192, 8192},
                                                     {16384, 16384},
                                                     {8192, 8192, 3001},
                                                     {16384, 8192, 511, 17}};
  for (const auto& lengths : batches) {
    SCOPED_TRACE(::testing::PrintToString(lengths));
    std::vector<int64_t> histories(lengths.size(), 0);
    for (size_t request = 1; request < histories.size(); ++request) {
      histories[request] = 1024 + static_cast<int64_t>(request) * 17;
    }
    check_budgeted_scores(lengths, histories);
  }
}

TEST(KPoolKernelTest, DecodeVerifyScoresCoverEveryBatchThrough128) {
  for (int64_t batch = 1; batch <= 128; ++batch) {
    for (const int64_t query : {1, 4}) {
      SCOPED_TRACE(::testing::Message() << batch << ":" << query);
      std::vector<int64_t> histories;
      histories.reserve(batch);
      for (int64_t row = 0; row < batch; ++row) {
        histories.emplace_back(32 + row % 17);
      }
      check_budgeted_scores(std::vector<int64_t>(batch, query), histories);
    }
  }
}

TEST(KPoolKernelTest, LongHistoryAndLargerVerifyBatchesMatchIndependentTorch) {
  for (const int64_t history : {1024, 65536, 262144, 270336}) {
    for (const int64_t query : {1, 4, 6, 8}) {
      SCOPED_TRACE(::testing::Message() << history << ":" << query);
      check_budgeted_scores({query, query - (query > 1 ? 1 : 0)},
                            {history, history - 17});
    }
  }
  for (const int64_t batch : {129, 160}) {
    for (const int64_t query : {1, 4, 6, 8}) {
      SCOPED_TRACE(::testing::Message() << batch << ":" << query);
      check_budgeted_scores(std::vector<int64_t>(batch, query),
                            std::vector<int64_t>(batch, 1024));
    }
  }
}

}  // namespace
}  // namespace xllm::kernel::mlu
