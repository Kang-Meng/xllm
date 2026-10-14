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

#pragma once

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <tuple>
#include <vector>

#include "framework/config/kv_cache_config.h"
#include "layers/mlu/glm5_next/glm5_next_kpool_indexer.h"

namespace xllm::layer::test {

class BlockSizeGuard final {
 public:
  explicit BlockSizeGuard(int64_t block_size)
      : saved_(KVCacheConfig::get_instance().block_size()) {
    KVCacheConfig::get_instance().block_size(block_size);
  }
  ~BlockSizeGuard() { KVCacheConfig::get_instance().block_size(saved_); }
  BlockSizeGuard(const BlockSizeGuard&) = delete;
  BlockSizeGuard& operator=(const BlockSizeGuard&) = delete;

 private:
  const int64_t saved_;
};

// The same indexer scenario runs locally and through real PCP collectives.
inline ::testing::AssertionResult check_kpool_state(
    const torch::Device& device,
    ProcessGroup* cp_group = nullptr,
    int32_t cp_rank = 0) {
  torch::InferenceMode inference_mode;
  torch::manual_seed(/*seed=*/20261013);
  const BlockSizeGuard block_size_guard(/*block_size=*/16);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  const auto ints = options.dtype(torch::kInt32);
  const auto random_tensor = [&](const std::vector<int64_t>& shape) {
    return torch::randn(shape, torch::TensorOptions().dtype(torch::kBFloat16))
        .to(device);
  };
  ModelArgs args;
  args.model_type("glm5_next")
      .hidden_size(128)
      .q_lora_rank(128)
      .index_n_heads(4)
      .index_head_dim(128)
      .qk_rope_head_dim(0)
      .index_topk(16)
      .index_kpool(4)
      .index_kpool_compress(true)
      .index_kpool_always_select_tail(true)
      .max_position_embeddings(1024);
  ParallelArgs parallel(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  Glm5NextKPoolIndexer indexer(args, QuantArgs(), parallel, nullptr, options);
  for (auto& parameter : indexer->named_parameters()) {
    parameter.value().copy_(random_tensor(parameter.value().sizes().vec())
                                .to(parameter.value().scalar_type()) *
                            0.1f);
  }
  const torch::Tensor targets = torch::tensor({2, 3, 4}, ints);
  const torch::Tensor sources = torch::tensor({1, 1, 4}, ints);
  const torch::Tensor table = torch::arange(48, ints).view({3, 16});
  struct StateCase {
    const char* name;
    int64_t tail_length;
    int32_t prefill_length;
    int32_t prefix_length;
    std::vector<int64_t> masked_tokens;
    bool verify;
  };
  const std::array<StateCase, 24> cases{{
      {"ring4_short1", 4, 1, 3, {}, false},
      {"ring4_short2", 4, 2, 3, {}, false},
      {"ring4_masked6", 4, 6, 3, {4}, false},
      {"ring4_masked8", 4, 8, 3, {4}, false},
      {"ring4_masked12", 4, 12, 3, {4, 8}, false},
      {"ring4_wrap33", 4, 33, 3, {}, false},
      {"ring7_short1", 7, 1, 65, {}, false},
      {"ring7_masked2", 7, 2, 65, {1}, false},
      {"ring7_masked6", 7, 6, 65, {4}, false},
      {"ring7_masked8", 7, 8, 65, {4}, false},
      {"ring7_masked12", 7, 12, 65, {4, 8}, false},
      {"ring7_wrap33", 7, 33, 65, {}, false},
      {"ring12_short1", 12, 1, 11, {}, true},
      {"ring12_short2", 12, 2, 11, {}, true},
      {"ring12_masked6", 12, 6, 11, {4}, true},
      {"ring12_masked8", 12, 8, 11, {4}, true},
      {"ring12_masked12", 12, 12, 11, {4, 8}, true},
      {"ring12_wrap33", 12, 33, 11, {}, true},
      {"ring16_short1", 16, 1, 15, {}, true},
      {"ring16_short2", 16, 2, 15, {}, true},
      {"ring16_masked6", 16, 6, 15, {4}, true},
      {"ring16_masked8", 16, 8, 15, {4}, true},
      {"ring16_masked12", 16, 12, 15, {4, 8}, true},
      {"ring16_wrap33", 16, 33, 15, {}, true},
  }};
  for (const auto& scenario : cases) {
    const int64_t tail_length = scenario.tail_length;
    const int32_t short_length = scenario.prefill_length;
    const std::vector<int32_t> prefill_lengths{
        short_length, cp_group != nullptr ? 65 : short_length + 1, 3};
    torch::Tensor initial_tail = random_tensor({5, 2, tail_length, 128});
    torch::Tensor direct_tail = initial_tail.clone();
    torch::Tensor copied_tail = initial_tail.clone();
    copied_tail[2].copy_(initial_tail[1]);
    copied_tail[3].copy_(initial_tail[1]);
    torch::Tensor copied_cache = random_tensor({48, 1, 4, 128});
    torch::Tensor direct_cache = copied_cache.clone();
    std::vector<int32_t> starts{
        scenario.prefix_length, scenario.prefix_length, 0};
    for (int32_t step = 0; step < 4; ++step) {
      const bool verify = step == 2 && scenario.verify;
      const std::vector<int32_t> lengths =
          step == 0 ? prefill_lengths : std::vector<int32_t>(3, verify ? 4 : 1);
      std::vector<int32_t> positions;
      positions.reserve(lengths[0] + lengths[1] + lengths[2]);
      std::vector<int32_t> ends;
      ends.reserve(3);
      for (size_t row = 0; row < lengths.size(); ++row) {
        for (int32_t token = 0; token < lengths[row]; ++token) {
          positions.emplace_back(starts[row] + token);
        }
        ends.emplace_back(starts[row] + lengths[row]);
      }
      const torch::Tensor position_tensor = torch::tensor(positions, ints);
      const torch::Tensor hidden =
          random_tensor({static_cast<int64_t>(positions.size()), 128});
      const torch::Tensor query =
          random_tensor({static_cast<int64_t>(positions.size()), 128});
      AttentionMetadata metadata{};
      metadata.block_table = table;
      metadata.linear_state_indices = targets;
      metadata.q_seq_lens = torch::tensor(lengths, ints);
      metadata.kv_seq_lens = torch::tensor(ends, ints);
      metadata.q_seq_lens_vec = lengths;
      metadata.kv_seq_lens_vec = ends;
      metadata.is_chunked_prefill = step == 0 || verify;
      metadata.is_spec_verify = verify;
      if (step == 0 && !scenario.masked_tokens.empty()) {
        metadata.slot_mapping = torch::ones_like(position_tensor);
        // Masked tokens must preserve valid ring contents, including earlier
        // writes to the same cell and inherited source state.
        for (const int64_t token : scenario.masked_tokens) {
          metadata.slot_mapping[token] = 0;
        }
      }
      AttentionMetadata direct_metadata = metadata;
      if (step == 0) {
        direct_metadata.linear_state_read_indices = sources;
      }
      std::tuple<torch::Tensor, torch::Tensor> expected;
      std::tuple<torch::Tensor, torch::Tensor> actual;
      if (cp_group != nullptr && step == 0) {
        const auto context = glm5_next_pcp::build_context(
            lengths, cp_rank, cp_group, /*chunk_size=*/64, device);
        AttentionMetadata local_metadata = metadata;
        local_metadata.q_seq_lens_vec = context.local_query_lengths;
        local_metadata.q_seq_lens =
            torch::tensor(context.local_query_lengths, ints);
        if (metadata.slot_mapping.defined()) {
          local_metadata.slot_mapping =
              glm5_next_pcp::shard_rows(metadata.slot_mapping, context);
        }
        const torch::Tensor local_hidden =
            glm5_next_pcp::shard_rows(hidden, context);
        const torch::Tensor local_query =
            glm5_next_pcp::shard_rows(query, context);
        const torch::Tensor local_positions =
            glm5_next_pcp::shard_rows(position_tensor, context);
        expected = indexer->forward_pcp(local_hidden,
                                        local_query,
                                        local_positions,
                                        position_tensor,
                                        copied_cache,
                                        copied_tail,
                                        metadata,
                                        local_metadata,
                                        context);
        actual = indexer->forward_pcp(local_hidden,
                                      local_query,
                                      local_positions,
                                      position_tensor,
                                      direct_cache,
                                      direct_tail,
                                      direct_metadata,
                                      local_metadata,
                                      context);
      } else {
        expected = indexer->forward(hidden,
                                    query,
                                    position_tensor,
                                    copied_cache,
                                    copied_tail,
                                    metadata);
        actual = indexer->forward(hidden,
                                  query,
                                  position_tensor,
                                  direct_cache,
                                  direct_tail,
                                  direct_metadata);
      }
      const auto failure = [&](const char* item) {
        return ::testing::AssertionFailure()
               << "KPOOL " << item
               << " mode=" << (cp_group != nullptr ? "pcp" : "local")
               << " rank=" << cp_rank << " case=" << scenario.name
               << " ring=" << tail_length << " length=" << short_length
               << " step=" << step;
      };
      if (!torch::equal(std::get<0>(actual), std::get<0>(expected))) {
        return failure("selected slots");
      }
      if (!torch::equal(std::get<1>(actual), std::get<1>(expected))) {
        return failure("context lengths");
      }
      if (!torch::equal(direct_cache, copied_cache)) {
        return failure("compressed cache");
      }
      if (!torch::equal(direct_tail.index_select(/*dim=*/0, targets),
                        copied_tail.index_select(/*dim=*/0, targets))) {
        return failure("destination tail");
      }
      if (!torch::equal(direct_tail[1], initial_tail[1])) {
        return failure("shared source tail");
      }
      if (!torch::equal(direct_tail[0], initial_tail[0])) {
        return failure("null slot tail");
      }
      starts = ends;
      if (verify) {
        // Reject a different suffix of each verify span, then resume decode
        // from the accepted positions while rejected entries remain in ring.
        for (size_t row = 0; row < starts.size(); ++row) {
          starts[row] -= static_cast<int32_t>(3 - row);
        }
      }
    }
  }
  return ::testing::AssertionSuccess();
}

}  // namespace xllm::layer::test
