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

#include <torch/torch.h>

#include <vector>

namespace xllm::dflash_async {

struct DraftCacheBlockTable {
  torch::Tensor blocks;
  bool is_circular = false;
};

DraftCacheBlockTable prepare_draft_cache_block_table(
    const torch::Tensor& block_table,
    const std::vector<torch::Tensor>& multi_block_tables,
    int64_t num_sequences,
    const torch::Device& device);

struct ContextKvInputs {
  torch::Tensor positions;
  torch::Tensor slots;
};

ContextKvInputs prepare_context_kv(const torch::Tensor& accepted_tokens,
                                   const torch::Tensor& base_positions,
                                   const torch::Tensor& block_table,
                                   int64_t block_size);

ContextKvInputs prepare_context_kv(const torch::Tensor& accepted_tokens,
                                   const torch::Tensor& base_positions,
                                   const DraftCacheBlockTable& block_table,
                                   int64_t block_size);

struct NextDraftInputs {
  torch::Tensor accepted_counts;
  torch::Tensor anchor_tokens;
  torch::Tensor query_tokens;
  torch::Tensor query_positions;
  torch::Tensor query_slots;
  torch::Tensor kv_lengths;
  torch::Tensor context_positions;
  torch::Tensor context_slots;
};

// Fixed-width device preparation. The valid prefix is written to context KV;
// unused rows use slot -1 and the proposal starts after the accepted inputs.
NextDraftInputs prepare_next_draft(const torch::Tensor& accepted_tokens,
                                   const torch::Tensor& base_positions,
                                   const torch::Tensor& block_table,
                                   int64_t mask_token_id,
                                   int64_t block_size);

NextDraftInputs prepare_next_draft(const torch::Tensor& accepted_tokens,
                                   const torch::Tensor& base_positions,
                                   const torch::Tensor& block_table,
                                   int64_t mask_token_id,
                                   int64_t block_size,
                                   int64_t query_width);

NextDraftInputs prepare_next_draft(const torch::Tensor& accepted_tokens,
                                   const torch::Tensor& base_positions,
                                   const DraftCacheBlockTable& block_table,
                                   int64_t mask_token_id,
                                   int64_t block_size,
                                   int64_t query_width);

}  // namespace xllm::dflash_async
