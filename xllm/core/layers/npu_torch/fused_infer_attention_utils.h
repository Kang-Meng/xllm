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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "layers/common/attention_metadata.h"

namespace xllm::layer::detail {

inline std::vector<int64_t> make_decode_actual_seq_lengths(int64_t num_tokens) {
  std::vector<int64_t> actual_seq_lengths;
  actual_seq_lengths.reserve(static_cast<size_t>(num_tokens));
  for (int64_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
    actual_seq_lengths.emplace_back(token_idx + 1);
  }
  return actual_seq_lengths;
}

// FIA protocol values for full attention and right-down causal masking.
constexpr int64_t kFiaSparseModeNone = 0;
constexpr int64_t kFiaSparseModeRightDown = 3;

// FIA requires sparse mode 3 when an explicit attention mask is used or the
// query is causal; DFlash2 supplies a band-mode override via fia_sparse_mode,
// so the mask and causality controls stay independent of it.
inline int64_t resolve_fia_sparse_mode(const AttentionMetadata& metadata) {
  if (metadata.fia_sparse_mode >= 0) {
    return metadata.fia_sparse_mode;
  }
  return (metadata.fia_attn_mask.defined() || metadata.is_causal)
             ? kFiaSparseModeRightDown
             : kFiaSparseModeNone;
}

inline std::optional<std::string> validate_jev_full_prefill(
    const AttentionMetadata& metadata,
    int64_t num_tokens) {
  if (!metadata.is_prefill || metadata.is_chunked_prefill ||
      metadata.is_mixed || metadata.is_spec_verify || metadata.is_dummy ||
      metadata.expanded_decode.enabled || metadata.prefill_without_cache) {
    return "JEV native attention requires a pure full prefill with KV caching";
  }
  const std::vector<int64_t>& query_lengths = metadata.q_cu_seq_lens_host_vec;
  const std::vector<int64_t>& kv_lengths = metadata.kv_cu_seq_lens_host_vec;
  const size_t offset = query_lengths.size() == kv_lengths.size() + 1 ? 1 : 0;
  if (kv_lengths.empty() ||
      query_lengths.size() != kv_lengths.size() + offset ||
      (offset == 1 && query_lengths.front() != 0)) {
    return "JEV native attention requires host cumulative query and KV lengths";
  }
  int64_t previous = 0;
  for (size_t index = 0; index < kv_lengths.size(); ++index) {
    const int64_t query_end = query_lengths[index + offset];
    if (query_end <= previous || query_end != kv_lengths[index]) {
      return "JEV native attention requires q_len == kv_len from position zero";
    }
    previous = query_end;
  }
  if (previous != num_tokens) {
    return "JEV native attention requires the complete prompt tokens";
  }
  return std::nullopt;
}

}  // namespace xllm::layer::detail
