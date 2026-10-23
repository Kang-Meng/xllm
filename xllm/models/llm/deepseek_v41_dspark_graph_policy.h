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

// Graph-path policy for the DeepSeek-V4.1 DSpark draft implemented in Python
// (xllm/python/models/deepseek_v41_dspark.py). This header owns graph admission
// and its request/warmup requirements; model computation stays in Python.

#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/framework/config/model_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/request/request_params.h"
#include "core/framework/sampling/draft_sampling_mode.h"
#include "core/runtime/decode_graph_bucket.h"
#include "core/runtime/options.h"
#include "core/util/utils.h"

namespace xllm::deepseek_v41 {

// Resolve the same proposal capability for the target's warmup plan and the
// actual draft model. Callers provide configuration; no global state is read.
template <typename Options>
inline bool resolve_draft_graph(const Options& options,
                                std::string_view model_type,
                                std::string_view model_impl,
                                bool is_npu) {
  return is_npu &&
         (util::is_deepseek_v41_model_type(model_type) ||
          util::is_deepseek_v41_dspark_model_type(model_type)) &&
         ModelConfig::is_python_model_impl(model_impl) &&
         SpeculativeConfig::is_dspark_algorithm(
             options.speculative_algorithm()) &&
         options.enable_dspark_draft_graph() && options.enable_graph() &&
         options.num_speculative_tokens() > 0 &&
         parse_draft_sampling_mode(options.draft_sampling_mode()) ==
             DraftSamplingMode::GREEDY &&
         !options.enable_adaptive_speculative_decode();
}

inline bool resolve_model_draft_graph(const runtime::Options& options,
                                      std::string_view model_type,
                                      std::string_view model_impl,
                                      bool is_npu,
                                      int32_t cp_size) {
  if (!options.is_draft_engine() ||
      !util::is_deepseek_v41_dspark_model_type(model_type)) {
    return false;
  }
  CHECK_LE(cp_size, 1)
      << "DeepSeek-V4.1 DSpark draft does not support context parallelism";
  return resolve_draft_graph(options, model_type, model_impl, is_npu);
}

// Build after model initialization so admission uses the same effective
// configuration as the workers. Reuse the loaded draft metadata rather than
// interpreting the checkpoint a second time. The callback retains no state.
template <typename Options>
inline std::function<std::optional<std::string>(const RequestParams&)>
make_draft_graph_request_validator(const Options& options,
                                   const ModelArgs* draft_model_args,
                                   std::string_view model_impl,
                                   bool is_npu) {
  if (draft_model_args == nullptr ||
      !resolve_draft_graph(
          options, draft_model_args->model_type(), model_impl, is_npu)) {
    return {};
  }
  return [](const RequestParams& params) -> std::optional<std::string> {
    if (params.frequency_penalty != 0.0f || params.presence_penalty != 0.0f ||
        params.repetition_penalty != 1.0f ||
        params.response_format == ResponseFormatType::JSON_OBJECT) {
      return "DSpark draft graph does not support sampling penalties or "
             "JSON response_format; disable --enable_dspark_draft_graph "
             "to use these request options";
    }
    return std::nullopt;
  };
}

inline std::vector<int32_t> draft_graph_warmup_batch_sizes(
    int32_t max_global_batch_size,
    int32_t dp_size,
    int32_t max_graph_batch_size) {
  if (dp_size <= 0 || max_global_batch_size < dp_size) {
    return {};
  }
  const int32_t max_local_batch_size = max_global_batch_size / dp_size;
  std::vector<int32_t> batch_sizes;
  batch_sizes.reserve(static_cast<size_t>(max_local_batch_size));
  int64_t previous_bucket = 0;
  for (int32_t local_batch_size = 1; local_batch_size <= max_local_batch_size;
       ++local_batch_size) {
    const int64_t bucket = runtime::get_decode_graph_token_bucket(
        local_batch_size, /*enable_no_padding=*/false);
    if (max_graph_batch_size > 0 && bucket > max_graph_batch_size) {
      break;
    }
    if (bucket == previous_bucket) {
      continue;
    }
    // Choose the first representative so filtering to a smaller allocatable
    // capacity still preserves every reachable sequence bucket.
    batch_sizes.emplace_back(local_batch_size * dp_size);
    previous_bucket = bucket;
  }
  return batch_sizes;
}

inline std::vector<int32_t> additional_draft_graph_warmup_batches(
    const runtime::Options& options,
    const ModelArgs* draft_model_args,
    std::string_view model_impl,
    bool is_npu,
    int32_t max_graph_batch_size) {
  if (draft_model_args == nullptr ||
      !resolve_draft_graph(
          options, draft_model_args->model_type(), model_impl, is_npu)) {
    return {};
  }
  return draft_graph_warmup_batch_sizes(
      options.max_seqs_per_batch(), options.dp_size(), max_graph_batch_size);
}

}  // namespace xllm::deepseek_v41
