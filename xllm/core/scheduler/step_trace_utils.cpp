/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "scheduler/step_trace_utils.h"

#include <algorithm>

#include "framework/request/sequence.h"

namespace xllm {

int64_t compute_step_qlen(const std::vector<Sequence*>& running_sequences,
                          const std::vector<size_t>& running_sequences_budgets,
                          const BatchForwardType& batch_forward_type) {
  if (running_sequences.size() != running_sequences_budgets.size()) {
    return 0;
  }

  if (batch_forward_type.has_decode()) {
    return static_cast<int64_t>(running_sequences.size());
  }

  int64_t qlen = 0;
  for (size_t i = 0; i < running_sequences.size(); ++i) {
    const Sequence* sequence = running_sequences[i];
    if (sequence == nullptr) {
      continue;
    }
    const size_t remaining_prompt_tokens =
        sequence->num_prompt_tokens() > sequence->kv_cache_tokens_num()
            ? sequence->num_prompt_tokens() - sequence->kv_cache_tokens_num()
            : 0;
    const size_t prompt_tokens =
        std::min(remaining_prompt_tokens, running_sequences_budgets[i]);
    qlen += static_cast<int64_t>(prompt_tokens);
  }

  return qlen;
}

int64_t compute_step_qlen(const std::vector<Batch>& batches,
                          const BatchForwardType& batch_forward_type) {
  std::vector<Sequence*> sequences;
  std::vector<size_t> budgets;

  for (const Batch& batch : batches) {
    if (batch.empty()) {
      continue;
    }
    std::vector<Sequence*> batch_sequences = batch.get_sequences();
    const std::vector<uint32_t>& allowed_max_tokens =
        batch.get_allowed_max_tokens();
    if (batch_sequences.size() != allowed_max_tokens.size()) {
      continue;
    }
    sequences.insert(
        sequences.end(), batch_sequences.begin(), batch_sequences.end());
    budgets.insert(
        budgets.end(), allowed_max_tokens.begin(), allowed_max_tokens.end());
  }

  return compute_step_qlen(sequences, budgets, batch_forward_type);
}

}  // namespace xllm
