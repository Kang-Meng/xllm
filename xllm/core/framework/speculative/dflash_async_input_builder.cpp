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

#include "core/framework/speculative/dflash_async_input_builder.h"

#include <glog/logging.h>

#include "core/framework/speculative/speculative_cache_utils.h"

namespace xllm::dflash_async {
namespace {

torch::Tensor map_cache_slots(const DraftCacheBlockTable& block_table,
                              const torch::Tensor& positions,
                              int64_t block_size) {
  if (block_table.is_circular) {
    CHECK_GT(block_table.blocks.size(1), 0);
    return speculative::map_positions_to_cache_slots(
        block_table.blocks,
        positions.remainder(block_table.blocks.size(1) * block_size),
        block_size);
  }
  return speculative::map_positions_to_cache_slots(
      block_table.blocks, positions, block_size);
}

}  // namespace

DraftCacheBlockTable prepare_draft_cache_block_table(
    const torch::Tensor& block_table,
    const std::vector<torch::Tensor>& multi_block_tables,
    int64_t num_sequences,
    const torch::Device& device) {
  if (multi_block_tables.empty()) {
    return {block_table, false};
  }
  const torch::Tensor& table = multi_block_tables.front();
  CHECK(table.defined());
  CHECK_EQ(table.dim(), 2);
  CHECK_GT(table.size(1), 0);
  CHECK_GE(table.size(0), num_sequences);
  torch::Tensor blocks = table.slice(0, 0, num_sequences);
  if (blocks.device() == device) {
    return {blocks, true};
  }
  CHECK(blocks.device().is_cpu());
  torch::Tensor pinned_blocks = torch::empty(
      blocks.sizes(), blocks.options().pinned_memory(!device.is_cpu()));
  pinned_blocks.copy_(blocks);
  return {pinned_blocks.to(device, true), true};
}

ContextKvInputs prepare_context_kv(const torch::Tensor& accepted_tokens,
                                   const torch::Tensor& base_positions,
                                   const torch::Tensor& block_table,
                                   int64_t block_size) {
  return prepare_context_kv(accepted_tokens,
                            base_positions,
                            DraftCacheBlockTable{block_table, false},
                            block_size);
}

ContextKvInputs prepare_context_kv(const torch::Tensor& accepted_tokens,
                                   const torch::Tensor& base_positions,
                                   const DraftCacheBlockTable& block_table,
                                   int64_t block_size) {
  CHECK_EQ(accepted_tokens.dim(), 2);
  CHECK_EQ(base_positions.dim(), 1);
  CHECK_EQ(block_table.blocks.dim(), 2);
  CHECK_EQ(base_positions.numel(), accepted_tokens.size(0));
  CHECK_EQ(block_table.blocks.size(0), accepted_tokens.size(0));
  CHECK_GT(block_size, 0);
  const int64_t accepted_width = accepted_tokens.size(1);
  torch::Tensor offsets =
      torch::arange(accepted_width,
                    base_positions.options().dtype(torch::kLong))
          .view({1, accepted_width});
  torch::Tensor positions =
      base_positions.to(torch::kLong).unsqueeze(1) + offsets;
  torch::Tensor valid = accepted_tokens.ge(0).to(torch::kLong).cumprod(1).ne(0);
  torch::Tensor slots = map_cache_slots(
      block_table,
      torch::where(valid, positions, torch::zeros_like(positions)),
      block_size);
  return {positions.reshape({-1}),
          slots.masked_fill(valid.reshape({-1}).logical_not(), -1)};
}

NextDraftInputs prepare_next_draft(const torch::Tensor& accepted_tokens,
                                   const torch::Tensor& base_positions,
                                   const torch::Tensor& block_table,
                                   int64_t mask_token_id,
                                   int64_t block_size) {
  CHECK_EQ(accepted_tokens.dim(), 2);
  return prepare_next_draft(accepted_tokens,
                            base_positions,
                            block_table,
                            mask_token_id,
                            block_size,
                            accepted_tokens.size(1));
}

NextDraftInputs prepare_next_draft(const torch::Tensor& accepted_tokens,
                                   const torch::Tensor& base_positions,
                                   const torch::Tensor& block_table,
                                   int64_t mask_token_id,
                                   int64_t block_size,
                                   int64_t query_width) {
  return prepare_next_draft(accepted_tokens,
                            base_positions,
                            DraftCacheBlockTable{block_table, false},
                            mask_token_id,
                            block_size,
                            query_width);
}

NextDraftInputs prepare_next_draft(const torch::Tensor& accepted_tokens,
                                   const torch::Tensor& base_positions,
                                   const DraftCacheBlockTable& block_table,
                                   int64_t mask_token_id,
                                   int64_t block_size,
                                   int64_t query_width) {
  CHECK_EQ(accepted_tokens.dim(), 2);
  CHECK_EQ(base_positions.dim(), 1);
  CHECK_EQ(block_table.blocks.dim(), 2);
  CHECK_EQ(base_positions.numel(), accepted_tokens.size(0));
  CHECK_EQ(block_table.blocks.size(0), accepted_tokens.size(0));
  CHECK_GT(block_size, 0);
  CHECK_GT(query_width, 0);
  const int64_t accepted_width = accepted_tokens.size(1);
  torch::Tensor context_offsets =
      torch::arange(accepted_width,
                    base_positions.options().dtype(torch::kLong))
          .view({1, accepted_width});
  torch::Tensor valid = accepted_tokens.ge(0).to(torch::kLong).cumprod(1);
  NextDraftInputs result;
  result.accepted_counts = valid.sum(1);
  // The rejection sampler guarantees at least one token per active row.
  torch::Tensor last = (result.accepted_counts - 1).clamp_min(0).unsqueeze(1);
  result.anchor_tokens = accepted_tokens.gather(1, last).reshape({-1});
  torch::Tensor query_offsets =
      query_width == accepted_width
          ? context_offsets
          : torch::arange(query_width,
                          base_positions.options().dtype(torch::kLong))
                .view({1, query_width});
  result.query_tokens =
      torch::where(query_offsets.eq(0),
                   result.anchor_tokens.unsqueeze(1),
                   query_width == accepted_width
                       ? torch::full_like(accepted_tokens, mask_token_id)
                       : torch::full({accepted_tokens.size(0), query_width},
                                     mask_token_id,
                                     accepted_tokens.options()))
          .reshape({-1});
  torch::Tensor base = base_positions.to(torch::kLong).unsqueeze(1);
  torch::Tensor next_positions =
      base + result.accepted_counts.unsqueeze(1) + query_offsets;
  result.query_positions = next_positions.reshape({-1});
  result.query_slots = map_cache_slots(block_table, next_positions, block_size);
  result.kv_lengths =
      (base_positions.to(torch::kLong) + result.accepted_counts + query_width)
          .to(torch::kInt32);
  torch::Tensor context_positions = base + context_offsets;
  result.context_positions = context_positions.reshape({-1});
  torch::Tensor context_slots =
      map_cache_slots(block_table, context_positions, block_size);
  result.context_slots = torch::where(valid.reshape({-1}).ne(0),
                                      context_slots,
                                      torch::full_like(context_slots, -1));
  return result;
}
}  // namespace xllm::dflash_async
