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

#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

#include "core/framework/model/causal_lm.h"
#include "core/runtime/options.h"
#include "models/model_registry.h"

namespace xllm::mlu {

int64_t get_block_capacity(int64_t max_kv_seq_len, int32_t block_size);

struct GraphLayout {
  int64_t num_reqs = 0;
  int64_t padded_num_reqs = 0;
  int64_t tokens_per_request = 1;
  int64_t attention_row_query_len = 1;
  int64_t padded_num_tokens = 0;
  int64_t cache_pad_slot = 0;
};

struct GraphKey {
  int64_t padded_num_tokens = 0;
  int64_t padded_num_reqs = 0;
  int64_t attention_row_query_len = 1;
  bool has_input_embedding = false;
  std::vector<int64_t> multi_block_table_column_counts;
  std::vector<int64_t> mtp_topk_schema;
};

inline auto get_key_fields(const GraphKey& key) {
  return std::tie(key.padded_num_tokens,
                  key.padded_num_reqs,
                  key.attention_row_query_len,
                  key.has_input_embedding,
                  key.multi_block_table_column_counts,
                  key.mtp_topk_schema);
}

// Lengths preserve vector boundaries in the scalar stream used by hashing and
// DP agreement. Visiting fields never allocates intermediate key storage.
template <typename Visitor>
void visit_graph_key(const GraphKey& key, Visitor&& visitor) {
  const auto visit_field = [&visitor](const auto& field) {
    using Field = std::decay_t<decltype(field)>;
    if constexpr (std::is_same_v<Field, std::vector<int64_t>>) {
      visitor(static_cast<int64_t>(field.size()));
      for (int64_t value : field) {
        visitor(value);
      }
    } else {
      visitor(static_cast<int64_t>(field));
    }
  };
  std::apply(
      [&visit_field](const auto&... fields) { (visit_field(fields), ...); },
      get_key_fields(key));
}

// Data-parallel ranks agree on a plan through a fixed-size packet, so the
// variable-length key fields share this budget.
constexpr std::size_t kMaxGraphKeyLayoutFields = 20;
constexpr std::size_t kMaxGraphKeyFields =
    std::tuple_size_v<decltype(get_key_fields(GraphKey{}))> +
    kMaxGraphKeyLayoutFields;

std::size_t get_key_size(const GraphKey& key);

bool operator==(const GraphKey& lhs, const GraphKey& rhs);

// Hashes are process-local lookup values, not stable identifiers across
// releases. Do not persist them or compare them across versions.
class GraphKeyHash final {
 public:
  std::size_t operator()(const GraphKey& key) const;
};

enum class EagerReason : int8_t {
  NON_DECODE,
  UNSUPPORTED_INPUT,
  INVALID_LAYOUT,
  TOKEN_LIMIT,
  HISTORY_LIMIT,
};

struct GraphPlan {
  GraphLayout layout;
  GraphKey key;
  int64_t history_capacity = 0;
  int64_t main_block_table_columns = 0;
};

using GraphDecision = std::variant<GraphPlan, EagerReason>;

struct GraphPlannerConfig {
  runtime::Options options;
  MtpModelCapabilities capabilities;
  bool state_required = false;
  int64_t history_capacity = 0;
  int64_t max_tokens = 0;
};

struct GraphStepMetadata {
  int64_t tokens = 0;
  const ModelInputParams& params;
};

class MluGraphPlanner final {
 public:
  explicit MluGraphPlanner(GraphPlannerConfig config);

  GraphDecision plan(const GraphStepMetadata& step) const;

 private:
  GraphPlannerConfig config_;
};

}  // namespace xllm::mlu
