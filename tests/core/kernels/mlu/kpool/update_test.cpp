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
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "kernels/mlu/kpool.h"
#include "platform/platform.h"

namespace xllm::kernel::mlu {
namespace {

TEST(KPoolConfigDeathTest, RejectsUnsupportedUpdateBeforeCacheMutation) {
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const torch::Tensor ids = torch::ones({1}, torch::kInt64);
  const torch::Tensor starts = torch::tensor({0, 1}, torch::kInt32);
  const torch::Tensor rows = torch::zeros({1}, torch::kInt32);
  const torch::Tensor table = torch::zeros({1, 1}, torch::kInt32);
  const auto update = [&](int64_t dimension,
                          int64_t block,
                          int64_t pool,
                          bool oversized_cache) {
    const torch::Tensor keys = torch::zeros({1, dimension}, bf16);
    const torch::Tensor ape = torch::zeros({4, dimension});
    const torch::Tensor matrix = torch::eye(dimension);
    torch::Tensor tail = torch::zeros({2, 2, 4, dimension}, bf16);
    torch::Tensor cache = torch::zeros({1, 1, 4, dimension}, bf16);
    if (oversized_cache) {
      cache = cache.expand({4194304, 1, 4, dimension});
    }
    update_kpool(keys,
                 keys,
                 ape,
                 matrix,
                 cache,
                 tail,
                 ids,
                 table,
                 rows,
                 rows,
                 starts,
                 block,
                 pool);
  };
  EXPECT_DEATH(update(65, 16, 4, false), "D=128");
  EXPECT_DEATH(update(128, 64, 4, false), "block_size=16");
  EXPECT_DEATH(update(128, 16, 8, false), "pool_size=4");
  EXPECT_DEATH(update(128, 16, 4, true), "cache.numel.*kMaxInt32");
}

TEST(KPoolStateIoContractTest, RejectsInvalidSourceMetadataBeforeLaunch) {
  const auto options = torch::TensorOptions().dtype(torch::kBFloat16);
  const torch::Tensor keys = torch::zeros({2, 128}, options);
  torch::Tensor tail = torch::zeros({4, 2, 4, 128}, options);
  const torch::Tensor targets = torch::tensor({2, 3}, torch::kInt32);
  const torch::Tensor sources = torch::tensor({1, 1}, torch::kInt32);
  const torch::Tensor positions = torch::tensor({3, 3}, torch::kInt32);
  const torch::Tensor starts = torch::tensor({0, 1, 2}, torch::kInt32);
  EXPECT_DEATH(
      stash_kpool_from_source(
          keys, keys, tail, targets, sources.view({1, 2}), positions, starts),
      "one-dimensional");
  EXPECT_DEATH(stash_kpool_from_source(keys,
                                       keys,
                                       tail,
                                       targets,
                                       sources.to(torch::kFloat32),
                                       positions,
                                       starts),
               "integer dtype");
  EXPECT_DEATH(
      stash_kpool_from_source(
          keys, keys, tail, targets, sources.slice(0, 0, 1), positions, starts),
      "requests");
  const torch::Tensor strided_sources =
      torch::tensor({1, 0, 1, 0}, torch::kInt32).slice(0, 0, 4, 2);
  EXPECT_DEATH(
      stash_kpool_from_source(
          keys, keys, tail, targets, strided_sources, positions, starts),
      "is_contiguous");
}

TEST(KPoolStateIoContractTest, RejectsMalformedUpdateMetadataBeforeLaunch) {
  const auto options = torch::TensorOptions().dtype(torch::kBFloat16);
  const torch::Tensor keys = torch::zeros({2, 128}, options);
  torch::Tensor cache = torch::zeros({1, 1, 4, 128}, options);
  torch::Tensor tail = torch::zeros({4, 2, 4, 128}, options);
  const torch::Tensor ape = torch::zeros({4, 128});
  const torch::Tensor matrix = torch::eye(128);
  const torch::Tensor ids = torch::tensor({2, 3}, torch::kInt32);
  const torch::Tensor table = torch::zeros({2, 1}, torch::kInt32);
  const torch::Tensor positions = torch::tensor({3, 3}, torch::kInt32);
  const torch::Tensor rows = torch::tensor({0, 1}, torch::kInt32);
  const torch::Tensor starts = torch::tensor({0, 1, 2}, torch::kInt32);
  const auto update = [&](const torch::Tensor& target_ids,
                          const torch::Tensor& pages,
                          const torch::Tensor& token_positions) {
    update_kpool(keys,
                 keys,
                 ape,
                 matrix,
                 cache,
                 tail,
                 target_ids,
                 pages,
                 token_positions,
                 rows,
                 starts,
                 /*block_size=*/16,
                 /*pool_size=*/4);
  };
  EXPECT_DEATH(update(ids.view({1, 2}), table, positions), "one-dimensional");
  EXPECT_DEATH(update(ids.to(torch::kFloat32), table, positions),
               "integer dtype");
  EXPECT_DEATH(update(ids, table.to(torch::kFloat32), positions),
               "table must have integer dtype");
  EXPECT_DEATH(update(ids, table, positions.to(torch::kFloat32)),
               "integer dtype");
}

TEST(KPoolStateIoContractTest, RejectsUnrepresentableSizesBeforeLaunch) {
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const int64_t too_wide =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
  const torch::Tensor keys = torch::empty({0, too_wide}, bf16);
  torch::Tensor tail = torch::empty({0, 2, 1, too_wide}, bf16);
  const torch::Tensor ids = torch::empty({0}, torch::kInt64);
  const torch::Tensor starts = torch::zeros({1}, torch::kInt64);
  EXPECT_DEATH(stash_kpool_from_source(keys, keys, tail, ids, ids, ids, starts),
               "tail.size\\(3\\).*kMaxInt32");
  const torch::Tensor empty_keys = torch::empty({0, 1}, bf16);
  torch::Tensor empty_cache = torch::empty({0, 1, 1, 1}, bf16);
  torch::Tensor empty_tail = torch::empty({0, 2, 1, 1}, bf16);
  const torch::Tensor matrix = torch::eye(1);
  const torch::Tensor table = torch::empty({0, 0}, torch::kInt64);
  EXPECT_DEATH(update_kpool(empty_keys,
                            empty_keys,
                            matrix,
                            matrix,
                            empty_cache,
                            empty_tail,
                            ids,
                            table,
                            ids,
                            ids,
                            starts,
                            /*block_size=*/too_wide,
                            /*pool_size=*/4),
               "MLU KPool requires block_size=16");
}

// Independent sequential CPU reference: write ring, pool across members,
// round through BF16, rotate, then scatter to the physical pool page.
void update_reference(const torch::Tensor& raw,
                      const torch::Tensor& gate,
                      const torch::Tensor& ape,
                      const torch::Tensor& had,
                      const torch::Tensor& pos,
                      const torch::Tensor& starts,
                      const torch::Tensor& ids,
                      const torch::Tensor& table,
                      torch::Tensor& cache,
                      torch::Tensor& tail) {
  for (int64_t req = 0; req < ids.numel(); ++req) {
    const int64_t state = ids[req].item<int64_t>();
    for (int64_t i = starts[req].item<int64_t>();
         i < starts[req + 1].item<int64_t>();
         ++i) {
      const int64_t p = pos[i].item<int64_t>();
      if (p < 0) {
        continue;
      }
      tail[state][0][p % tail.size(2)].copy_(raw[i]);
      tail[state][1][p % tail.size(2)].copy_(gate[i]);
      if (p % 4 != 3) {
        continue;
      }
      const torch::Tensor members =
          torch::arange(p - 3, p + 1, torch::kInt64).remainder(tail.size(2));
      const torch::Tensor k =
          tail[state][0].index_select(0, members).to(torch::kFloat32);
      const torch::Tensor g =
          tail[state][1].index_select(0, members).to(torch::kFloat32);
      const torch::Tensor pooled = (k * torch::softmax(g + ape, 0))
                                       .sum(0)
                                       .to(torch::kBFloat16)
                                       .to(torch::kFloat32);
      const int64_t pool = p / 4;
      const int64_t page = table[req][pool / 4].item<int64_t>();
      cache[page][0][pool % 4].copy_(
          torch::matmul(pooled, had.transpose(0, 1)));
    }
  }
}

TEST(KPoolKernelTest, UpdatesPreservePoolsAndAliasedSourceRings) {
  torch::manual_seed(20261027);
  const torch::Device device(Platform::type_torch(), 0);
  constexpr int64_t kWidth = 128;
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const torch::Tensor keys = torch::randn({8, kWidth}, bf16);
  const torch::Tensor gates = torch::randn_like(keys);
  const torch::Tensor ape = torch::randn({4, kWidth}) * 0.1;
  const torch::Tensor matrix = torch::eye(kWidth);
  const torch::Tensor ids = torch::tensor({1, 2}, torch::kInt64);
  const torch::Tensor starts = torch::tensor({0, 4, 8}, torch::kInt32);
  const torch::Tensor positions =
      torch::tensor({1, 2, 3, 4, 3, 4, 5, 6}, torch::kInt64);
  const torch::Tensor rows =
      torch::tensor({0, 0, 0, 0, 1, 1, 1, 1}, torch::kInt64);
  const torch::Tensor table = torch::tensor({{0}, {1}}, torch::kInt64);
  const torch::Tensor original_tail = torch::randn({4, 2, 5, kWidth}, bf16);
  const torch::Tensor original_cache = torch::randn({2, 1, 4, kWidth}, bf16);
  for (const bool decode : {false, true}) {
    for (const bool swap_sources : {false, true}) {
      SCOPED_TRACE(decode);
      SCOPED_TRACE(swap_sources);
      const torch::Tensor sources =
          (swap_sources ? ids.flip({0}) : ids).to(torch::kInt32);
      torch::Tensor expected_tail = original_tail.clone();
      for (int64_t request = 0; request < 2; ++request) {
        expected_tail[ids[request].item<int64_t>()].copy_(
            original_tail[sources[request].item<int64_t>()]);
      }
      torch::Tensor expected_cache = original_cache.clone();
      update_reference(keys,
                       gates,
                       ape,
                       matrix,
                       positions,
                       starts,
                       ids,
                       table,
                       expected_cache,
                       expected_tail);
      torch::Tensor cache = original_cache.to(device);
      torch::Tensor tail = original_tail.to(device);
      update_kpool(keys.to(device),
                   gates.to(device),
                   ape.to(device),
                   matrix.to(device),
                   cache,
                   tail,
                   ids.to(device),
                   table.to(device),
                   positions.to(device),
                   rows.to(device),
                   starts.to(device),
                   /*block_size=*/16,
                   /*pool_size=*/4,
                   decode,
                   sources.to(device));
      EXPECT_TRUE(torch::equal(tail.cpu(), expected_tail));
      EXPECT_TRUE(torch::allclose(cache.cpu().to(torch::kFloat32),
                                  expected_cache.to(torch::kFloat32),
                                  /*rtol=*/0.02,
                                  /*atol=*/0.02));
    }
  }
}

void check_batched_pool_updates(int64_t batch, int64_t query) {
  torch::manual_seed(20261026);
  const torch::Device device(Platform::type_torch(), 0);
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const auto ints = torch::TensorOptions().dtype(torch::kInt64);
  const torch::Tensor ape = torch::randn({4, 128}) * 0.1;
  const torch::Tensor had = torch::eye(128) + torch::randn({128, 128}) * 0.02;
  const torch::Tensor ids = torch::arange(1, batch + 1, ints).flip({0});
  const torch::Tensor table = torch::randperm(batch * 8, ints).view({batch, 8});
  const torch::Tensor starts = torch::arange(batch + 1, ints) * query;
  const torch::Tensor rows =
      torch::arange(batch, ints).repeat_interleave(query);
  torch::Tensor expected_cache = torch::randn({batch * 8 + 1, 1, 4, 128}, bf16);
  // Retain the previous pool remainder plus the speculative window for
  // rollback. Q6/Q8 must not inherit the shorter MTP3 ring capacity.
  const int64_t ring = std::max<int64_t>(12, 2 * query + 4);
  torch::Tensor expected_tail = torch::randn({batch + 2, 2, ring, 128}, bf16);
  const torch::Tensor untouched_page = expected_cache[batch * 8].clone();
  const torch::Tensor untouched_tail = expected_tail[batch + 1].clone();
  torch::Tensor cache = expected_cache.to(device);
  torch::Tensor tail = expected_tail.to(device);
  // Start at different pool remainders. Round 3 overwrites a speculative
  // suffix; round 4 continues from that rewritten history.
  for (const int64_t begin : std::vector<int64_t>{
           8, 8 + query, 8 + 2 * query, 8 + query + 1, 8 + 2 * query + 1}) {
    SCOPED_TRACE(begin);
    const torch::Tensor positions =
        (torch::arange(batch, ints).remainder(4).unsqueeze(1) +
         torch::arange(begin, begin + query, ints).unsqueeze(0))
            .flatten();
    const torch::Tensor raw = torch::randn({batch * query, 128}, bf16);
    const torch::Tensor gate = torch::randn_like(raw);
    update_reference(raw,
                     gate,
                     ape,
                     had,
                     positions,
                     starts,
                     ids,
                     table,
                     expected_cache,
                     expected_tail);
    update_kpool(raw.to(device),
                 gate.to(device),
                 ape.to(device),
                 had.to(device),
                 cache,
                 tail,
                 ids.to(device),
                 table.to(device, torch::kInt32),
                 positions.to(device),
                 rows.to(device),
                 starts.to(device),
                 /*block_size=*/16,
                 /*pool_size=*/4,
                 /*decode=*/true);
    EXPECT_TRUE(torch::equal(tail.cpu(), expected_tail));
    EXPECT_TRUE(torch::allclose(cache.cpu(),
                                expected_cache,
                                /*rtol=*/0.015625,
                                /*atol=*/0.015625));
    EXPECT_TRUE(torch::equal(cache[batch * 8].cpu(), untouched_page));
    EXPECT_TRUE(torch::equal(tail[batch + 1].cpu(), untouched_tail));
  }
}

TEST(KPoolKernelTest, DecodeVerifyUpdatesCoverEveryBatchThrough128) {
  for (int64_t batch = 1; batch <= 128; ++batch) {
    for (const int64_t query : {1, 4}) {
      SCOPED_TRACE(::testing::Message() << batch << ":" << query);
      check_batched_pool_updates(batch, query);
    }
  }
}

TEST(KPoolKernelTest, LargerVerifyUpdatesPreserveSixEightTokenWindows) {
  for (const int64_t batch : {17, 128, 129, 160}) {
    for (const int64_t query : {1, 4, 6, 8}) {
      SCOPED_TRACE(::testing::Message() << batch << ":" << query);
      check_batched_pool_updates(batch, query);
    }
  }
}

TEST(KPoolKernelTest, PagedUpdatesPreserveRaggedRingAndUntouchedState) {
  torch::manual_seed(902);
  const torch::Device device(Platform::type_torch(), 0);
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const torch::Tensor ape = torch::randn({4, 128});
  const torch::Tensor had = torch::eye(128) + torch::randn({128, 128}) * 0.02;
  const torch::Tensor ids = torch::tensor({3, 1, 0}, torch::kInt64);
  const torch::Tensor table =
      torch::tensor({{4, 0, 2}, {3, 1, 5}, {6, 7, 8}}, torch::kInt32);
  for (const bool decode : {false, true}) {
    torch::Tensor expected_cache = torch::randn({9, 1, 4, 128}, bf16);
    torch::Tensor expected_tail = torch::randn({5, 2, 12, 128}, bf16);
    torch::Tensor cache = expected_cache.to(device);
    torch::Tensor tail = expected_tail.to(device);
    const void* cache_ptr = cache.data_ptr();
    const void* tail_ptr = tail.data_ptr();
    // Non-aligned first chunk, ring wrap, then rejected speculative positions
    // overwritten by a shorter continuation. Request 2 is entirely padding.
    for (const int64_t start : {2, 8, 14, 15, 19}) {
      const torch::Tensor positions =
          torch::cat({torch::arange(start, start + 6, torch::kInt64),
                      torch::arange(start + 1, start + 4, torch::kInt64),
                      torch::full({2}, -1, torch::kInt64)});
      const torch::Tensor starts = torch::tensor({0, 6, 9, 11}, torch::kInt64);
      const torch::Tensor rows =
          torch::tensor({0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2}, torch::kInt64);
      const torch::Tensor raw = torch::randn({11, 128}, bf16);
      const torch::Tensor gate = torch::randn({11, 128}, bf16);
      update_reference(raw,
                       gate,
                       ape,
                       had,
                       positions,
                       starts,
                       ids,
                       table,
                       expected_cache,
                       expected_tail);
      update_kpool(raw.to(device),
                   gate.to(device),
                   ape.to(device),
                   had.to(device),
                   cache,
                   tail,
                   ids.to(device),
                   table.to(device),
                   positions.to(device),
                   rows.to(device),
                   starts.to(device),
                   16,
                   4,
                   decode);
      EXPECT_EQ(cache_ptr, cache.data_ptr());
      EXPECT_EQ(tail_ptr, tail.data_ptr());
      EXPECT_TRUE(torch::equal(tail.cpu(), expected_tail));
      EXPECT_TRUE(
          torch::equal(cache.slice(0, 6).cpu(), expected_cache.slice(0, 6)));
      EXPECT_TRUE(
          torch::allclose(cache.cpu(), expected_cache, 0.015625, 0.015625));
    }
  }
}
}  // namespace
TEST(KPoolKernelTest, MissingPagesPreserveEveryChronologicalTailToken) {
  torch::manual_seed(20261019);
  const torch::Device device(Platform::type_torch(), /*index=*/0);
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const torch::Tensor ape = torch::randn({4, 128}).to(device);
  const torch::Tensor hadamard =
      (torch::eye(128) + torch::randn({128, 128}) * 0.02).to(device);
  for (const int64_t batch : {3, 7, 16}) {
    for (const int64_t ring : {12, 64, 65}) {
      SCOPED_TRACE(batch);
      SCOPED_TRACE(ring);
      const torch::Tensor ids_cpu = torch::arange(1, batch + 1, torch::kInt64);
      const torch::Tensor starts =
          torch::arange(0, 4 * batch + 1, 4, torch::kInt64).to(device);
      const torch::Tensor positions =
          torch::arange(65533, 65537, torch::kInt32).repeat({batch}).to(device);
      const torch::Tensor rows =
          torch::arange(batch, torch::kInt64).repeat_interleave(4).to(device);
      const torch::Tensor table =
          torch::full({batch, 65536}, -1, torch::kInt32).to(device);
      const torch::Tensor ids = ids_cpu.to(device);
      for (const int32_t source_mode : {0, 1, 2}) {
        SCOPED_TRACE(source_mode);
        const torch::Tensor sources_cpu =
            source_mode == 0 ? torch::zeros_like(ids_cpu)
                             : ids_cpu + (source_mode == 1 ? 0 : batch);
        const torch::Tensor sources = sources_cpu.to(device);
        const torch::Tensor original_tail =
            torch::randn({2 * batch + 1, 2, ring, 128}, bf16);
        const torch::Tensor original_cache = torch::randn({1, 1, 4, 128}, bf16);
        torch::Tensor tail = original_tail.to(device);
        torch::Tensor cache = original_cache.to(device);
        for (int32_t replay = 0; replay < 20; ++replay) {
          SCOPED_TRACE(replay);
          const torch::Tensor keys_cpu = torch::randn({4 * batch, 128}, bf16);
          const torch::Tensor gates_cpu = torch::randn_like(keys_cpu);
          torch::Tensor expected_tail = original_tail.clone();
          for (int64_t sequence = 0; sequence < batch; ++sequence) {
            const int64_t destination = sequence + 1;
            const int64_t source = sources_cpu[sequence].item<int64_t>();
            if (source == 0) {
              expected_tail[destination].zero_();
            } else {
              expected_tail[destination].copy_(original_tail[source]);
            }
            for (int64_t token = 0; token < 4; ++token) {
              const int64_t row = sequence * 4 + token;
              const int64_t slot = (65533 + token) % ring;
              expected_tail[destination][0][slot].copy_(keys_cpu[row]);
              expected_tail[destination][1][slot].copy_(gates_cpu[row]);
            }
          }
          tail.copy_(original_tail.to(device));
          update_kpool(keys_cpu.to(device),
                       gates_cpu.to(device),
                       ape,
                       hadamard,
                       cache,
                       tail,
                       ids,
                       table,
                       positions,
                       rows,
                       starts,
                       /*block_size=*/16,
                       /*pool_size=*/4,
                       /*decode=*/true,
                       sources);
          EXPECT_TRUE(torch::equal(tail.cpu(), expected_tail));
          EXPECT_TRUE(torch::equal(cache.cpu(), original_cache));
        }
      }
    }
  }
}

TEST(KPoolKernelTest, RaggedRequestWavesPreserveUpdateAndStashTailSemantics) {
  torch::manual_seed(20261019);
  const torch::Device device(Platform::type_torch(), /*index=*/0);
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  std::vector<std::vector<int64_t>> cases;
  cases.reserve(12);
  for (const int64_t batch : {31, 32, 33, 63, 64, 65}) {
    cases.emplace_back(batch, 4);
  }
  for (int64_t long_request = 0; long_request < 4; ++long_request) {
    std::vector<int64_t> lengths(4, 32);
    lengths[long_request] = 1024;
    cases.emplace_back(std::move(lengths));
  }
  cases.emplace_back(std::vector<int64_t>{280, 280, 280, 280});
  cases.emplace_back(std::vector<int64_t>{31, 0, 33, 1024});
  constexpr int64_t kRing = 65;
  for (const auto& lengths : cases) {
    const int64_t batch = static_cast<int64_t>(lengths.size());
    const torch::Tensor starts_cpu =
        torch::cat({torch::zeros({1}, torch::kInt64), torch::tensor(lengths)})
            .cumsum(0);
    const int64_t tokens = starts_cpu[-1].item<int64_t>();
    std::vector<torch::Tensor> position_parts;
    std::vector<torch::Tensor> row_parts;
    position_parts.reserve(batch);
    row_parts.reserve(batch);
    for (int64_t request = 0; request < batch; ++request) {
      position_parts.emplace_back(
          torch::arange(8192 - lengths[request], 8192, torch::kInt32));
      row_parts.emplace_back(
          torch::full({lengths[request]}, request, torch::kInt64));
    }
    torch::Tensor positions_cpu = torch::cat(position_parts);
    positions_cpu.slice(0, 1, tokens, 17).fill_(-1);
    const torch::Tensor keys_cpu = torch::randn({tokens, 128}, bf16);
    const torch::Tensor gates_cpu = torch::randn_like(keys_cpu);
    const torch::Tensor original_tail =
        torch::randn({2 * batch + 1, 2, kRing, 128}, bf16);
    const torch::Tensor original_cache = torch::randn({1, 1, 4, 128}, bf16);
    const torch::Tensor ids_cpu = torch::arange(1, batch + 1, torch::kInt64);
    const torch::Tensor sources_cpu = ids_cpu + batch;
    torch::Tensor expected_update = original_tail.clone();
    torch::Tensor expected_stash = original_tail.clone();
    for (int64_t request = 0; request < batch; ++request) {
      const int64_t target = request + 1;
      const int64_t source = target + batch;
      expected_update[target].copy_(original_tail[source]);
      expected_stash[target].copy_(original_tail[source]);
      const int64_t begin = starts_cpu[request].item<int64_t>();
      const int64_t end = starts_cpu[request + 1].item<int64_t>();
      for (int64_t token = begin; token < end; ++token) {
        const int64_t position = positions_cpu[token].item<int32_t>();
        if (position < 0) {
          continue;
        }
        expected_update[target][0][position % kRing].copy_(keys_cpu[token]);
        expected_update[target][1][position % kRing].copy_(gates_cpu[token]);
      }
      int64_t valid_end = end;
      while (valid_end > begin &&
             positions_cpu[valid_end - 1].item<int32_t>() < 0) {
        --valid_end;
      }
      if (valid_end == begin) {
        continue;
      }
      const int64_t last = positions_cpu[valid_end - 1].item<int32_t>();
      for (int64_t slot = 0; slot < kRing; ++slot) {
        const int64_t token = valid_end - 1 - (last - slot + kRing) % kRing;
        if (token < begin || positions_cpu[token].item<int32_t>() < 0) {
          continue;
        }
        expected_stash[target][0][slot].copy_(keys_cpu[token]);
        expected_stash[target][1][slot].copy_(gates_cpu[token]);
      }
    }
    const torch::Tensor keys = keys_cpu.to(device);
    const torch::Tensor gates = gates_cpu.to(device);
    const torch::Tensor positions = positions_cpu.to(device);
    const torch::Tensor starts = starts_cpu.to(device);
    const torch::Tensor rows = torch::cat(row_parts).to(device);
    const torch::Tensor table =
        torch::full({batch, 65536}, -1, torch::kInt32).to(device);
    const torch::Tensor ape = torch::zeros({4, 128}).to(device);
    const torch::Tensor hadamard = torch::eye(128).to(device);
    for (const torch::ScalarType metadata_type :
         {torch::kInt32, torch::kInt64}) {
      SCOPED_TRACE(::testing::Message()
                   << "batch=" << batch << " tokens=" << tokens
                   << " metadata_type=" << metadata_type);
      const torch::Tensor ids = ids_cpu.to(metadata_type).to(device);
      const torch::Tensor sources = sources_cpu.to(metadata_type).to(device);
      torch::Tensor tail = original_tail.to(device);
      torch::Tensor cache = original_cache.to(device);
      update_kpool(keys,
                   gates,
                   ape,
                   hadamard,
                   cache,
                   tail,
                   ids,
                   table,
                   positions,
                   rows,
                   starts,
                   /*block_size=*/16,
                   /*pool_size=*/4,
                   /*decode=*/true,
                   sources);
      EXPECT_TRUE(torch::equal(cache.cpu(), original_cache));
      EXPECT_TRUE(torch::equal(tail.cpu(), expected_update));
      tail.copy_(original_tail.to(device));
      stash_kpool_from_source(
          keys, gates, tail, ids, sources, positions, starts);
      EXPECT_TRUE(torch::equal(tail.cpu(), expected_stash));
    }
  }
}

TEST(KPoolKernelTest, UpdatesReadImmutableSourcesAcrossRequestAliasCycles) {
  torch::manual_seed(20261019);
  const torch::Device device(Platform::type_torch(), /*index=*/0);
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  constexpr int64_t kBatch = 4;
  constexpr int64_t kRing = 65;
  const std::vector<std::vector<int64_t>> length_cases = {{4, 4, 4, 4},
                                                          {31, 0, 33, 1024}};
  const std::vector<std::vector<int64_t>> source_cases = {
      {2, 1, 4, 3}, {2, 3, 4, 1}, {1, 1, 1, 1}};
  for (const auto& lengths : length_cases) {
    const torch::Tensor starts_cpu =
        torch::cat({torch::zeros({1}, torch::kInt64), torch::tensor(lengths)})
            .cumsum(0);
    const int64_t tokens = starts_cpu[-1].item<int64_t>();
    std::vector<torch::Tensor> position_parts;
    std::vector<torch::Tensor> row_parts;
    position_parts.reserve(kBatch);
    row_parts.reserve(kBatch);
    for (int64_t request = 0; request < kBatch; ++request) {
      position_parts.emplace_back(
          torch::arange(8192 - lengths[request], 8192, torch::kInt32));
      row_parts.emplace_back(
          torch::full({lengths[request]}, request, torch::kInt64));
    }
    torch::Tensor positions_cpu = torch::cat(position_parts);
    positions_cpu.slice(0, 1, tokens, 17).fill_(-1);
    const torch::Tensor keys_cpu = torch::randn({tokens, 128}, bf16);
    const torch::Tensor gates_cpu = torch::randn_like(keys_cpu);
    const torch::Tensor original_tail =
        torch::randn({kBatch + 1, 2, kRing, 128}, bf16);
    const torch::Tensor original_cache = torch::randn({1, 1, 4, 128}, bf16);
    const torch::Tensor keys = keys_cpu.to(device);
    const torch::Tensor gates = gates_cpu.to(device);
    const torch::Tensor positions = positions_cpu.to(device);
    const torch::Tensor rows = torch::cat(row_parts).to(device);
    const torch::Tensor starts = starts_cpu.to(device);
    const torch::Tensor table =
        torch::full({kBatch, 65536}, -1, torch::kInt32).to(device);
    const torch::Tensor ape = torch::zeros({4, 128}).to(device);
    const torch::Tensor hadamard = torch::eye(128).to(device);
    for (const auto& sources : source_cases) {
      torch::Tensor expected_tail = original_tail.clone();
      for (int64_t request = 0; request < kBatch; ++request) {
        const int64_t target = request + 1;
        expected_tail[target].copy_(original_tail[sources[request]]);
        const int64_t begin = starts_cpu[request].item<int64_t>();
        const int64_t end = starts_cpu[request + 1].item<int64_t>();
        for (int64_t token = begin; token < end; ++token) {
          const int64_t position = positions_cpu[token].item<int32_t>();
          if (position < 0) {
            continue;
          }
          expected_tail[target][0][position % kRing].copy_(keys_cpu[token]);
          expected_tail[target][1][position % kRing].copy_(gates_cpu[token]);
        }
      }
      for (const torch::ScalarType metadata_type :
           {torch::kInt32, torch::kInt64}) {
        SCOPED_TRACE(::testing::Message()
                     << "tokens=" << tokens
                     << " metadata_type=" << metadata_type);
        const torch::Tensor ids = torch::arange(1, kBatch + 1, torch::kInt64)
                                      .to(metadata_type)
                                      .to(device);
        const torch::Tensor read_ids =
            torch::tensor(sources).to(metadata_type).to(device);
        torch::Tensor cache = original_cache.to(device);
        torch::Tensor tail = original_tail.to(device);
        for (int32_t repeat = 0; repeat < 10; ++repeat) {
          tail.copy_(original_tail.to(device));
          update_kpool(keys,
                       gates,
                       ape,
                       hadamard,
                       cache,
                       tail,
                       ids,
                       table,
                       positions,
                       rows,
                       starts,
                       /*block_size=*/16,
                       /*pool_size=*/4,
                       /*decode=*/true,
                       read_ids);
          EXPECT_TRUE(torch::equal(cache.cpu(), original_cache));
          EXPECT_TRUE(torch::equal(tail.cpu(), expected_tail));
        }
      }
    }
  }
}

TEST(KPoolKernelTest, CapturedUpdatesAndStashPreserveCrossRequestSources) {
  const torch::Device device(Platform::type_torch(), 0);
  torch::NoGradGuard no_grad;
  torch::manual_seed(20261025);
  const torch_mlu::mlu::MLUStreamGuard stream_guard(
      torch_mlu::getStreamFromPool(false, 0));
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  // Small and multi-request cooperative slices, followed by the general
  // workspace path for both update and stash. The final request is empty.
  const std::vector<std::vector<int64_t>> shapes = {
      {4, 12, 1}, {9, 12, 33}, {33, 64, 4}, {129, 129, 4}};
  for (const auto& shape : shapes) {
    const int64_t batch = shape[0];
    const int64_t ring = shape[1];
    const int64_t length = shape[2];
    SCOPED_TRACE(::testing::Message() << "batch=" << batch << " ring=" << ring
                                      << " length=" << length);
    const int64_t tokens = (batch - 1) * length;
    const torch::Tensor starts_cpu =
        torch::cat({torch::arange(batch, torch::kInt64) * length,
                    torch::full({1}, tokens, torch::kInt64)});
    const torch::Tensor positions_cpu =
        torch::arange(8192 - length, 8192, torch::kInt32).repeat({batch - 1});
    const torch::Tensor rows_cpu =
        torch::arange(batch - 1, torch::kInt64).repeat_interleave(length);
    const torch::Tensor keys_cpu = torch::randn({tokens, 128}, bf16);
    const torch::Tensor gates_cpu = torch::randn_like(keys_cpu);
    const torch::Tensor original_tail =
        torch::randn({batch + 1, 2, ring, 128}, bf16);
    const torch::Tensor original_cache = torch::randn({1, 1, 4, 128}, bf16);
    const torch::Tensor keys = keys_cpu.to(device);
    const torch::Tensor gates = gates_cpu.to(device);
    const torch::Tensor positions = positions_cpu.to(device);
    const torch::Tensor rows = rows_cpu.to(device);
    const torch::Tensor starts = starts_cpu.to(device);
    const torch::Tensor table =
        torch::full({batch, 512}, -1, torch::kInt32).to(device);
    const torch::Tensor ape = torch::zeros({4, 128}).to(device);
    const torch::Tensor hadamard = torch::eye(128).to(device);
    for (const torch::ScalarType metadata_type :
         {torch::kInt32, torch::kInt64}) {
      const torch::Tensor ids_cpu = torch::arange(1, batch + 1, torch::kInt64);
      const torch::Tensor ids = ids_cpu.to(metadata_type).to(device);
      for (const bool shared_source : {false, true}) {
        const torch::Tensor sources_cpu =
            shared_source
                ? torch::full({batch}, 2, torch::kInt64)
                : torch::cat({ids_cpu.slice(0, 1), ids_cpu.slice(0, 0, 1)});
        const torch::Tensor sources = sources_cpu.to(metadata_type).to(device);
        torch::Tensor expected = original_tail.clone();
        for (int64_t request = 0; request < batch; ++request) {
          const int64_t destination = request + 1;
          expected[destination].copy_(
              original_tail[sources_cpu[request].item<int64_t>()]);
          for (int64_t token = starts_cpu[request].item<int64_t>();
               token < starts_cpu[request + 1].item<int64_t>();
               ++token) {
            const int64_t slot = positions_cpu[token].item<int32_t>() % ring;
            expected[destination][0][slot].copy_(keys_cpu[token]);
            expected[destination][1][slot].copy_(gates_cpu[token]);
          }
        }
        for (const int32_t metadata_mode : {0, 1, 2, 3}) {
          const torch::Tensor active_positions =
              metadata_mode != 0 ? positions.to(torch::kInt64) : positions;
          const torch::Tensor active_starts =
              metadata_mode == 2 ? starts.to(torch::kInt32) : starts;
          const torch::Tensor active_table =
              metadata_mode != 0 ? table.to(torch::kInt64) : table;
          const torch::Tensor active_sources =
              metadata_mode == 3
                  ? sources.to(metadata_type == torch::kInt32 ? torch::kInt64
                                                              : torch::kInt32)
                  : sources;
          for (const bool stash : {false, true}) {
            SCOPED_TRACE(::testing::Message()
                         << "metadata=" << metadata_type << " shared_source="
                         << shared_source << " stash=" << stash
                         << " metadata_mode=" << metadata_mode);
            torch::Tensor backing =
                torch::full({batch + 3, 2, ring, 128}, 97, bf16.device(device));
            torch::Tensor tail = backing.narrow(0, 1, batch + 1);
            torch::Tensor cache = original_cache.to(device);
            tail.copy_(original_tail.to(device));
            const auto invoke = [&]() {
              if (stash) {
                stash_kpool_from_source(keys,
                                        gates,
                                        tail,
                                        ids,
                                        active_sources,
                                        active_positions,
                                        active_starts);
              } else {
                update_kpool(keys,
                             gates,
                             ape,
                             hadamard,
                             cache,
                             tail,
                             ids,
                             active_table,
                             active_positions,
                             rows,
                             active_starts,
                             16,
                             4,
                             true,
                             active_sources);
              }
            };
            invoke();
            torch_mlu::synchronize();
            torch_mlu::MLUGraph graph;
            graph.capture_begin();
            invoke();
            graph.capture_end();
            for (int32_t replay = 0; replay < 8; ++replay) {
              tail.copy_(original_tail.to(device));
              graph.replay();
              EXPECT_TRUE(torch::equal(tail.cpu(), expected));
              EXPECT_TRUE(torch::equal(cache.cpu(), original_cache));
              EXPECT_TRUE(backing[0].eq(97).all().item<bool>());
              EXPECT_TRUE(backing[-1].eq(97).all().item<bool>());
            }
            EXPECT_TRUE(torch::equal(keys.cpu(), keys_cpu));
            EXPECT_TRUE(torch::equal(gates.cpu(), gates_cpu));
            EXPECT_TRUE(
                torch::equal(active_sources.cpu(),
                             sources_cpu.to(active_sources.scalar_type())));
            EXPECT_TRUE(
                torch::equal(active_starts.cpu(),
                             starts_cpu.to(active_starts.scalar_type())));
            EXPECT_TRUE(
                torch::equal(active_positions.cpu(),
                             positions_cpu.to(active_positions.scalar_type())));
            EXPECT_TRUE(active_table.eq(-1).all().item<bool>());
          }
        }
      }
    }
  }
}

// Changing only an integer metadata representation must preserve the original
// pool arithmetic, including its BF16 rounding before and after projection.
TEST(KPoolKernelTest, MixedStateIdWidthsPreserveRandomPoolBytes) {
  const torch::Device device(Platform::type_torch(), 0);
  torch::NoGradGuard no_grad;
  const torch_mlu::mlu::MLUStreamGuard stream_guard(
      torch_mlu::getStreamFromPool(false, 0));
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const torch::Tensor starts =
      torch::tensor({0, 5, 9, 14, 14}, torch::kInt64).to(device);
  const torch::Tensor rows =
      torch::tensor({0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2}, torch::kInt64)
          .to(device);
  const torch::Tensor positions =
      torch::cat({torch::arange(27, 32, torch::kInt64),
                  torch::arange(28, 32, torch::kInt64),
                  torch::arange(27, 32, torch::kInt64)})
          .to(device);
  const torch::Tensor table =
      torch::arange(1, 9, torch::kInt32).view({4, 2}).to(device);
  const torch::Tensor ids =
      torch::tensor({1, 2, 3, 4}, torch::kInt64).to(device);
  const torch::Tensor sources =
      torch::tensor({2, 1, 0, 1}, torch::kInt64).to(device);
  const torch::Tensor ids_i32 = ids.to(torch::kInt32);
  const torch::Tensor sources_i32 = sources.to(torch::kInt32);
  for (const int64_t seed : {20262020, 20262023, 20262027}) {
    torch::manual_seed(seed);
    const torch::Tensor keys = torch::randn({14, 128}, bf16).to(device);
    const torch::Tensor gates = torch::randn({14, 128}, bf16).to(device);
    const torch::Tensor ape = torch::randn({4, 128}).to(device);
    const torch::Tensor matrix =
        (torch::randn({128, 128}) / std::sqrt(128.0)).to(device);
    const torch::Tensor original_cache =
        torch::randn({9, 1, 4, 128}, bf16).to(device);
    const torch::Tensor original_tail =
        torch::randn({5, 2, 12, 128}, bf16).to(device);
    for (const bool decode : {false, true}) {
      torch::Tensor expected_cache = original_cache.clone();
      torch::Tensor expected_tail = original_tail.clone();
      update_kpool(keys,
                   gates,
                   ape,
                   matrix,
                   expected_cache,
                   expected_tail,
                   ids,
                   table,
                   positions,
                   rows,
                   starts,
                   16,
                   4,
                   decode,
                   sources);
      const torch::Tensor cache_bytes =
          expected_cache.cpu().view(torch::kUInt8);
      const torch::Tensor tail_bytes = expected_tail.cpu().view(torch::kUInt8);
      for (const bool narrow_destination : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << "seed=" << seed << " decode=" << decode
                     << " narrow_destination=" << narrow_destination);
        const torch::Tensor active_ids = narrow_destination ? ids_i32 : ids;
        const torch::Tensor active_sources =
            narrow_destination ? sources : sources_i32;
        torch::Tensor cache = original_cache.clone();
        torch::Tensor tail = original_tail.clone();
        const auto invoke = [&]() {
          update_kpool(keys,
                       gates,
                       ape,
                       matrix,
                       cache,
                       tail,
                       active_ids,
                       table,
                       positions,
                       rows,
                       starts,
                       16,
                       4,
                       decode,
                       active_sources);
        };
        invoke();
        EXPECT_TRUE(torch::equal(cache.cpu().view(torch::kUInt8), cache_bytes));
        EXPECT_TRUE(torch::equal(tail.cpu().view(torch::kUInt8), tail_bytes));
        torch_mlu::MLUGraph graph;
        graph.capture_begin();
        invoke();
        graph.capture_end();
        for (int32_t replay = 0; replay < 16; ++replay) {
          cache.copy_(original_cache);
          tail.copy_(original_tail);
          graph.replay();
          EXPECT_TRUE(
              torch::equal(cache.cpu().view(torch::kUInt8), cache_bytes));
          EXPECT_TRUE(torch::equal(tail.cpu().view(torch::kUInt8), tail_bytes));
        }
        EXPECT_TRUE(torch::equal(active_ids.cpu(),
                                 ids.cpu().to(active_ids.scalar_type())));
        EXPECT_TRUE(
            torch::equal(active_sources.cpu(),
                         sources.cpu().to(active_sources.scalar_type())));
      }
    }
  }
}

TEST(KPoolKernelTest, StridedStashPreservesWidePositionsAndInitialSources) {
  torch::NoGradGuard no_grad;
  torch::manual_seed(20261026);
  const torch::Device device(Platform::type_torch(), 0);
  const torch_mlu::mlu::MLUStreamGuard stream_guard(
      torch_mlu::getStreamFromPool(false, 0));
  constexpr int64_t kWidth = 128;
  constexpr int64_t kRing = 5;
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const torch::Tensor keys_cpu = torch::randn({5, kWidth}, bf16);
  const torch::Tensor gates_cpu = torch::randn_like(keys_cpu);
  const torch::Tensor positions_cpu =
      torch::tensor({std::numeric_limits<int64_t>::max(),
                     int64_t{1} << 32,
                     std::numeric_limits<int64_t>::min(),
                     (int64_t{1} << 32) + 3,
                     int64_t{-1}},
                    torch::kInt64);
  const torch::Tensor starts_cpu = torch::tensor({0, 1, 4, 5}, torch::kInt64);
  const torch::Tensor sources_cpu = torch::tensor({2, 3, 1}, torch::kInt64);
  const torch::Tensor original_tail = torch::randn({4, 2, kRing, kWidth}, bf16);
  torch::Tensor expected = original_tail.clone();
  for (int64_t request = 0; request < 3; ++request) {
    expected[request + 1].copy_(
        original_tail[sources_cpu[request].item<int64_t>()]);
    const int64_t begin = starts_cpu[request].item<int64_t>();
    int64_t end = starts_cpu[request + 1].item<int64_t>();
    while (end > begin && positions_cpu[end - 1].item<int64_t>() < 0) {
      --end;
    }
    if (end == begin) {
      continue;
    }
    const int64_t last_slot = positions_cpu[end - 1].item<int64_t>() % kRing;
    for (int64_t slot = 0; slot < kRing; ++slot) {
      const int64_t token = end - 1 - (last_slot - slot + kRing) % kRing;
      if (token < begin || positions_cpu[token].item<int64_t>() < 0) {
        continue;
      }
      expected[request + 1][0][slot].copy_(keys_cpu[token]);
      expected[request + 1][1][slot].copy_(gates_cpu[token]);
    }
  }
  torch::Tensor key_backing =
      torch::full({10, kWidth}, 97, bf16.device(device));
  torch::Tensor gate_backing = torch::full_like(key_backing, 97);
  const torch::Tensor keys = key_backing.slice(0, 0, 10, 2);
  const torch::Tensor gates = gate_backing.slice(0, 0, 10, 2);
  keys.copy_(keys_cpu.to(device));
  gates.copy_(gates_cpu.to(device));
  ASSERT_FALSE(keys.is_contiguous());
  ASSERT_EQ(keys.stride(1), 1);
  const torch::Tensor original_keys = key_backing.cpu();
  const torch::Tensor original_gates = gate_backing.cpu();
  const torch::Tensor ids = torch::tensor({1, 2, 3}, torch::kInt64).to(device);
  const torch::Tensor sources = sources_cpu.to(device);
  const torch::Tensor positions = positions_cpu.to(device);
  const torch::Tensor starts = starts_cpu.to(device);
  torch::Tensor backing =
      torch::full({6, 2, kRing, kWidth}, 97, bf16.device(device));
  torch::Tensor tail = backing.narrow(0, 1, 4);
  const auto invoke = [&]() {
    stash_kpool_from_source(keys, gates, tail, ids, sources, positions, starts);
  };
  tail.copy_(original_tail.to(device));
  invoke();
  EXPECT_TRUE(torch::equal(tail.cpu(), expected));
  torch_mlu::MLUGraph graph;
  graph.capture_begin();
  invoke();
  graph.capture_end();
  for (int32_t replay = 0; replay < 8; ++replay) {
    tail.copy_(original_tail.to(device));
    graph.replay();
    EXPECT_TRUE(torch::equal(tail.cpu(), expected));
    EXPECT_TRUE(backing[0].eq(97).all().item<bool>());
    EXPECT_TRUE(backing[-1].eq(97).all().item<bool>());
    EXPECT_TRUE(torch::equal(key_backing.cpu(), original_keys));
    EXPECT_TRUE(torch::equal(gate_backing.cpu(), original_gates));
    EXPECT_TRUE(torch::equal(positions.cpu(), positions_cpu));
    EXPECT_TRUE(torch::equal(starts.cpu(), starts_cpu));
    EXPECT_TRUE(torch::equal(sources.cpu(), sources_cpu));
    EXPECT_TRUE(
        torch::equal(ids.cpu(), torch::tensor({1, 2, 3}, torch::kInt64)));
  }
}

TEST(KPoolKernelTest, WideMetadataFallbackPoolsMissingSourcesByDestination) {
  torch::NoGradGuard no_grad;
  const torch::Device device(Platform::type_torch(), 0);
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const torch::Tensor keys_cpu = torch::tensor({4, 8, 12}, torch::kFloat32)
                                     .view({3, 1})
                                     .expand({3, 128})
                                     .to(torch::kBFloat16)
                                     .contiguous();
  const torch::Tensor keys = keys_cpu.to(device);
  const torch::Tensor gates = torch::zeros_like(keys);
  const torch::Tensor ape = torch::zeros({4, 128}).to(device);
  const torch::Tensor hadamard = torch::eye(128).to(device);
  // One token completes a pool for each nonempty span. The last span is empty.
  const torch::Tensor starts =
      torch::tensor({0, 1, 2, 3, 3}, torch::kInt32).to(device);
  const torch::Tensor positions =
      torch::full({3}, 3, torch::TensorOptions().dtype(torch::kInt64))
          .to(device);
  const torch::Tensor rows = torch::tensor({0, 1, 2}, torch::kInt64).to(device);
  const torch::Tensor table =
      torch::arange(4, torch::kInt64).view({4, 1}).to(device);
  torch::Tensor original_tail = torch::zeros({5, 2, 12, 128}, bf16);
  original_tail[1][0].fill_(2);
  original_tail[2][0].fill_(4);
  original_tail[3][0].fill_(6);
  original_tail[4][0].fill_(8);
  const torch::Tensor original_cache = torch::full({4, 1, 4, 128}, 97, bf16);
  for (const int64_t padding : {int64_t{0},
                                std::numeric_limits<int64_t>::min(),
                                -(int64_t{1} << 32) + 1}) {
    const torch::Tensor ids =
        torch::tensor(std::vector<int64_t>{1, padding, 3, 4}, torch::kInt64)
            .to(device);
    const torch::Tensor sources =
        torch::tensor(std::vector<int64_t>{padding, 2, 1, 3}, torch::kInt64)
            .to(device);
    torch::Tensor expected_tail = original_tail.clone();
    expected_tail[1].zero_();
    expected_tail[1][0][3].fill_(4);
    expected_tail[3].copy_(original_tail[1]);
    expected_tail[3][0][3].fill_(12);
    expected_tail[4].copy_(original_tail[3]);
    torch::Tensor expected_cache = original_cache.clone();
    // Zero source history contributes three zeros; the other live request
    // reads three original source values of two, before destination one writes.
    expected_cache[0][0][0].fill_(1);
    expected_cache[2][0][0].fill_(4.5);
    for (const bool decode : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "padding=" << padding << " decode=" << decode);
      torch::Tensor tail = original_tail.to(device);
      torch::Tensor cache = original_cache.to(device);
      update_kpool(keys,
                   gates,
                   ape,
                   hadamard,
                   cache,
                   tail,
                   ids,
                   table,
                   positions,
                   rows,
                   starts,
                   16,
                   4,
                   decode,
                   sources);
      EXPECT_TRUE(torch::equal(cache.cpu(), expected_cache));
      EXPECT_TRUE(torch::equal(tail.cpu(), expected_tail));
      EXPECT_TRUE(torch::equal(keys.cpu(), keys_cpu));
    }
  }
}

}  // namespace xllm::kernel::mlu

namespace xllm::kernel::mlu {
namespace {
void check_chunked_pool_chain(int64_t total_tokens, int64_t chunk_size) {
  torch::NoGradGuard no_grad;
  torch::manual_seed(72317);
  const torch::Device device(Platform::type_torch(), 0);
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  // GLM-5.3-Flash-W4A8: H=32, D=128, P=4, top-k=2048; MTP3 ring=12.
  const torch::Tensor raw = torch::randn({total_tokens, 128}, bf16);
  const torch::Tensor gate = torch::randn_like(raw);
  const torch::Tensor ape = torch::randn({4, 128}) * 0.1;
  torch::Tensor had = torch::ones({1, 1});
  for (int64_t size = 1; size < 128; size *= 2) {
    had =
        torch::cat({torch::cat({had, had}, 1), torch::cat({had, -had}, 1)}, 0);
  }
  had /= std::sqrt(128.0);
  const torch::Tensor probabilities =
      torch::softmax(gate.to(torch::kFloat32).view({total_tokens / 4, 4, 128}) +
                         ape.unsqueeze(0),
                     1);
  const torch::Tensor pooled =
      (raw.to(torch::kFloat32).view({total_tokens / 4, 4, 128}) * probabilities)
          .sum(1)
          .to(torch::kBFloat16)
          .to(torch::kFloat32);
  const torch::Tensor reference = torch::matmul(pooled, had.transpose(0, 1))
                                      .to(torch::kBFloat16)
                                      .to(torch::kFloat32);
  const torch::Tensor pages = torch::randperm(total_tokens / 16, torch::kInt64)
                                  .view({1, total_tokens / 16});
  const torch::Tensor table = pages.to(device, torch::kInt32);
  const torch::Tensor state_ids = torch::tensor({1}, torch::kInt64).to(device);

  torch::Tensor cache =
      torch::zeros({total_tokens / 16, 1, 4, 128}, bf16.device(device));
  torch::Tensor tail = torch::zeros({2, 2, 12, 128}, bf16.device(device));
  int64_t strict_bad = 0;
  for (int64_t begin = 0; begin < total_tokens; begin += chunk_size) {
    SCOPED_TRACE(begin);
    const int64_t count = std::min(chunk_size, total_tokens - begin);
    const int64_t end = begin + count;
    const torch::Tensor starts =
        torch::tensor(std::vector<int64_t>{0, count}, torch::kInt64).to(device);
    const torch::Tensor rows = torch::zeros({count}, torch::kInt64).to(device);
    const int64_t pools = end / 4;
    const torch::Tensor positions =
        torch::arange(begin, end, torch::kInt64).to(device);
    update_kpool(raw.narrow(0, begin, count).to(device),
                 gate.narrow(0, begin, count).to(device),
                 ape.to(device),
                 had.to(device),
                 cache,
                 tail,
                 state_ids,
                 table,
                 positions,
                 rows,
                 starts,
                 16,
                 4,
                 false);
    const torch::Tensor columns = torch::arange(pools, torch::kInt64);
    const torch::Tensor slots =
        pages[0].index_select(0, torch::floor_divide(columns, 4)) * 4 +
        columns.remainder(4);
    const torch::Tensor actual_cache =
        cache.cpu().view({-1, 128}).index_select(0, slots).to(torch::kFloat32);
    const torch::Tensor ref = reference.narrow(0, 0, pools);
    ASSERT_TRUE(torch::allclose(actual_cache, ref, 0.015625, 0.015625));
    for (int64_t token = end - 12; token < end; ++token) {
      EXPECT_TRUE(torch::equal(tail[1][0][token % 12].cpu(), raw[token]));
      EXPECT_TRUE(torch::equal(tail[1][1][token % 12].cpu(), gate[token]));
    }
    const torch::Tensor q =
        torch::randn({8, 32, 128}, bf16).to(torch::kFloat32);
    const torch::Tensor weights = torch::randn({8, 32});
    const auto score = [&](const torch::Tensor& keys) {
      return (torch::relu(torch::matmul(q, keys.transpose(0, 1)) / 64) *
              weights.unsqueeze(-1))
          .sum(1);
    };
    const torch::Tensor expected = score(ref);
    const torch::Tensor same_cache = score(actual_cache);
    // ReLU is 1-Lipschitz: propagate independently measured cache rounding
    // through |scale| sum_h |w_h| sum_d |Q_hd| |delta K_d|. The score-stage
    // tolerance is unchanged. Approved migration contract, 2026-09-17.
    const torch::Tensor bound =
        (torch::matmul(q.abs(), (actual_cache - ref).abs().transpose(0, 1)) *
         weights.abs().unsqueeze(-1))
            .sum(1) /
        64;
    const torch::Tensor query_positions = positions.narrow(0, count - 8, 8);
    const torch::Tensor valid = (columns + 1).unsqueeze(0) * 4 <=
                                query_positions.cpu().unsqueeze(1) + 1;
    torch::Tensor output = torch::empty(
        {8, pools},
        torch::TensorOptions().device(device).dtype(torch::kFloat32));
    score_kpool(q.to(device, torch::kBFloat16),
                weights.to(device),
                cache,
                table,
                query_positions,
                rows.narrow(0, 0, 8),
                output,
                16,
                4,
                1.0 / 64);
    const torch::Tensor actual = output.cpu();
    ASSERT_TRUE(torch::allclose(actual.masked_select(valid),
                                same_cache.masked_select(valid),
                                1e-4,
                                1e-4));
    const torch::Tensor error = (actual - expected).abs();
    strict_bad += (valid & (error > 0.001 + expected.abs() * 0.001))
                      .sum()
                      .item<int64_t>();
    EXPECT_FALSE((valid & (error > bound + 1e-4 + same_cache.abs() * 1e-4))
                     .any()
                     .item<bool>());
    const torch::Tensor expected_masked =
        expected.masked_fill(~valid, -std::numeric_limits<float>::infinity());
    const torch::Tensor ids = select_kpool(output, 512).cpu();
    const torch::Tensor selected = expected_masked.gather(1, ids);
    const torch::Tensor cutoff =
        std::get<0>(expected_masked.topk(512, -1)).select(1, 511).unsqueeze(1);
    const torch::Tensor row_bound =
        std::get<0>((bound + 1e-4 + same_cache.abs() * 1e-4).max(1, true));
    EXPECT_TRUE((selected >= cutoff - 2 * row_bound).all().item<bool>());
  }
  // Retain the historical check as a diagnostic, not an unexplained pass.
  ::testing::Test::RecordProperty("strict_chain_score_violations",
                                  std::to_string(strict_bad));
}
}  // namespace

TEST(KPoolKernelTest, Chunked64KMatchesTorchWithinRoundingPropagation) {
  check_chunked_pool_chain(/*total_tokens=*/65536, /*chunk_size=*/8192);
}

TEST(KPoolKernelTest, Chunked264KMatchesTorchWithEightKChunks) {
  check_chunked_pool_chain(/*total_tokens=*/270336, /*chunk_size=*/8192);
}

TEST(KPoolKernelTest, Chunked264KMatchesTorchWithSixteenKChunksAndTail) {
  check_chunked_pool_chain(/*total_tokens=*/270336, /*chunk_size=*/16384);
}
}  // namespace xllm::kernel::mlu
