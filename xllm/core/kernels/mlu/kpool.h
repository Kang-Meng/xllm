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

#pragma once
#include <torch/torch.h>

#include <cstdint>

namespace xllm::kernel::mlu {

struct KPoolSelection final {
  torch::Tensor physical_slots;
  torch::Tensor context_lens;
};

// MLU KPool supports BF16 keys with D=128, pool_size=4 and block_size=16.
// Inputs are request-major, with ascending consecutive active positions.
// Cache storage is framework-owned; padding positions never mutate state.
// Positive state IDs and physical page IDs name allocated cache/tail storage.
void update_kpool(const torch::Tensor& k,
                  const torch::Tensor& gate,
                  const torch::Tensor& ape,
                  const torch::Tensor& hadamard,
                  torch::Tensor& cache,
                  torch::Tensor& tail,
                  const torch::Tensor& tail_ids,
                  const torch::Tensor& table,
                  const torch::Tensor& positions,
                  const torch::Tensor& rows,
                  const torch::Tensor& starts,
                  int64_t block_size,
                  int64_t pool_size,
                  bool decode = false,
                  const torch::Tensor& read_tail_ids = torch::Tensor());

// Standalone test support for tail layouts and source aliasing. Production
// prefill updates must use update_kpool, which includes tail stashing after
// pooling reads; this entry point does not validate that integrated path.
// Complete unique positive destination states from the immutable initial source
// ring and new tokens. Sources may equal another request's destination or be
// shared across requests. Positive IDs name allocated tail slots.
// Nonpositive state IDs represent missing/padding
// states. All pooling reads must complete before this operation updates the
// tail.
void stash_kpool_from_source(const torch::Tensor& k,
                             const torch::Tensor& gate,
                             torch::Tensor& tail,
                             const torch::Tensor& tail_ids,
                             const torch::Tensor& read_tail_ids,
                             const torch::Tensor& positions,
                             const torch::Tensor& starts);

// Writes caller-owned FP32 workspace. Positions are causal, zero-based token
// positions; -1 masks the whole row. Rows map queries to request page tables.
// Score each query using its actual row. query_starts is optional scheduling
// metadata; spans may be ragged, contain padding, or name different requests.
// Negative rows/positions produce no live pools and never read the page table.
void score_kpool(const torch::Tensor& query,
                 const torch::Tensor& weights,
                 const torch::Tensor& cache,
                 const torch::Tensor& table,
                 const torch::Tensor& positions,
                 const torch::Tensor& rows,
                 torch::Tensor& scores,
                 int64_t block_size,
                 int64_t pool_size,
                 double scale,
                 const torch::Tensor& query_starts = torch::Tensor());

// The ID set torch.topk(count) selects per row, in unspecified order;
// nonfinite selections and missing columns -1.
torch::Tensor select_kpool(const torch::Tensor& scores, int64_t count);

// Same result as select_kpool for scores written by score_kpool with the
// given pool_size. Uses live-prefix selection when supported.
torch::Tensor select_kpool_live(const torch::Tensor& scores,
                                int64_t count,
                                const torch::Tensor& positions,
                                int64_t pool_size);

// active = tail_ids[rows] > padding_id && (slot_mapping undefined ||
// slot_mapping > 0); returns active ? positions : -1 with positions' dtype.
torch::Tensor kpool_cache_positions(const torch::Tensor& tail_ids,
                                    const torch::Tensor& rows,
                                    const torch::Tensor& positions,
                                    const torch::Tensor& slot_mapping,
                                    int64_t padding_id);

KPoolSelection expand_kpool(const torch::Tensor& ids,
                            const torch::Tensor& positions,
                            const torch::Tensor& rows,
                            const torch::Tensor& table,
                            int64_t block_size,
                            int64_t token_budget,
                            int64_t pool_size,
                            bool always_tail,
                            bool prefill = false);
}  // namespace xllm::kernel::mlu
