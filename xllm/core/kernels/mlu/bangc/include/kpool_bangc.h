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

#include <cstdint>

namespace xllm::kernel::mlu {

// Each function inspects metadata before launch and returns true when the
// operation has been dispatched. False means unsupported inputs and no device
// work or output mutation. Callers decide whether to reject unsupported inputs
// (update/score/expand use CHECK) or select a supported fallback before launch.
bool bangc_kpool_update(const torch::Tensor& key,
                        const torch::Tensor& gate,
                        const torch::Tensor& ape,
                        const torch::Tensor& matrix,
                        torch::Tensor& cache,
                        torch::Tensor& tail,
                        const torch::Tensor& ids,
                        const torch::Tensor& read_ids,
                        const torch::Tensor& table,
                        const torch::Tensor& positions,
                        const torch::Tensor& rows,
                        const torch::Tensor& starts,
                        int64_t block_size,
                        int64_t pool_size,
                        bool decode,
                        bool inherit);

bool bangc_kpool_stash(const torch::Tensor& key,
                       const torch::Tensor& gate,
                       torch::Tensor& tail,
                       const torch::Tensor& ids,
                       const torch::Tensor& read_ids,
                       const torch::Tensor& positions,
                       const torch::Tensor& starts);

bool bangc_kpool_score(const torch::Tensor& query,
                       const torch::Tensor& weights,
                       const torch::Tensor& cache,
                       const torch::Tensor& table,
                       const torch::Tensor& positions,
                       const torch::Tensor& rows,
                       torch::Tensor& scores,
                       int64_t block_size,
                       int64_t pool_size,
                       double scale,
                       const torch::Tensor& starts);

bool bangc_kpool_expand(const torch::Tensor& ids,
                        const torch::Tensor& positions,
                        const torch::Tensor& rows,
                        const torch::Tensor& table,
                        torch::Tensor& slots,
                        torch::Tensor& lengths,
                        int64_t block_size,
                        int64_t token_budget,
                        int64_t pool_size,
                        bool always_tail,
                        bool prefill);

bool bangc_kpool_rows(const torch::Tensor& starts, torch::Tensor& rows);

// Writes every column of selected: per row, the IDs torch.topk(count) selects
// (NaN and +inf rank above finite scores, ties prefer lower IDs) with
// nonfinite selections as -1, in unspecified order; missing columns -1.
bool bangc_kpool_select(const torch::Tensor& scores, torch::Tensor& selected);

// bangc_kpool_select reading only each row's live prefix, given the
// int32 cache positions the scores were computed from.
bool bangc_kpool_select_live(const torch::Tensor& scores,
                             const torch::Tensor& positions,
                             torch::Tensor& selected);

// Writes output = tail_ids[rows] > 0 && slot_mapping > 0 ? positions : -1.
bool bangc_kpool_cache_positions(const torch::Tensor& tail_ids,
                                 const torch::Tensor& rows,
                                 const torch::Tensor& positions,
                                 const torch::Tensor& slot_mapping,
                                 torch::Tensor& output);

bool bangc_kpool_tail_generic(const torch::Tensor& key,
                              const torch::Tensor& gate,
                              torch::Tensor& tail,
                              const torch::Tensor& ids,
                              const torch::Tensor& read_ids,
                              const torch::Tensor& positions,
                              const torch::Tensor& starts,
                              bool decode);

}  // namespace xllm::kernel::mlu
