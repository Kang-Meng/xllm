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

#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string>
#include <string_view>

#include "core/common/macros.h"
#include "core/framework/config/option_category.h"

namespace xllm {

class JsonReader;

class ModelConfig final {
 public:
  ModelConfig() = default;
  ~ModelConfig() = default;

  static ModelConfig& get_instance();

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();
  void normalize_cpp_chat_template(const std::string& model_type);

  [[nodiscard]] static bool is_python_model_impl(std::string_view model_impl);
  [[nodiscard]] static std::optional<std::string>
  validate_python_speculative_decode(std::string_view model_impl,
                                     std::string_view model_type,
                                     std::string_view speculative_algorithm,
                                     int32_t num_speculative_tokens);

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "MODEL OPTIONS",
        {"model_id",
         "model",
         "model_impl",
         "backend",
         "task",
         "capture_hidden_state_layer",
         "python_model_path",
         "reasoning_parser",
         "tool_call_parser",
         "enable_qwen3_reranker",
         "engram_config",
         "flashinfer_workspace_buffer_size",
         "use_cpp_chat_template"}};
    return kOptionCategory;
  }

  PROPERTY(std::string, model_id);

  PROPERTY(std::string, model);

  PROPERTY(std::string, model_impl);

  PROPERTY(std::string, python_model_path);

  PROPERTY(std::string, backend);

  PROPERTY(std::string, task) = "generate";

  PROPERTY(int32_t, capture_hidden_state_layer) = -1;

  PROPERTY(std::string, reasoning_parser);

  PROPERTY(std::string, tool_call_parser);

  PROPERTY(bool, enable_qwen3_reranker) = false;

  // Engram host-offload switches as a raw JSON object string, e.g.
  // '{"cpu_offload":true,"dp_shared_memory":true}'. Passed through to the
  // Python executor verbatim; parsing/validation lives there
  // (xllm/python/models/engram_config.py). "{}" disables both switches.
  PROPERTY(std::string, engram_config) = "{}";

  PROPERTY(int32_t, flashinfer_workspace_buffer_size) = 128 * 1024 * 1024;

  PROPERTY(bool, use_cpp_chat_template) = true;
};

}  // namespace xllm
