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
#include <nlohmann/json.hpp>
#include <string>

#include "core/common/macros.h"
#include "core/framework/config/option_category.h"

namespace xllm {

class JsonReader;

class MultimodalConfig final {
 public:
  MultimodalConfig() = default;
  ~MultimodalConfig() = default;

  static MultimodalConfig& get_instance();

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();
  const nlohmann::json& processor_args() const { return processor_args_; }
  const nlohmann::json& processor_args(const std::string& modality) const;

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "MULTIMODAL OPTIONS",
        {"use_ctc",
         "enable_return_embedding_modality_tags",
         "limit_image_per_prompt",
         "mm_args",
         "fps_min_frames",
         "fps_max_frames",
         "video_max_token_num",
         "max_encoder_cache_size",
         "max_processor_cache_items",
         "enable_return_mm_full_embeddings",
         "mm_download_headers",
         "use_audio_in_video",
         "audio_max_upload_file_mb",
         "audio_max_decode_duration_s",
         "enable_mm_encoder_dp"}};
    return kOptionCategory;
  }

  PROPERTY(bool, use_ctc) = false;

  PROPERTY(bool, enable_return_embedding_modality_tags) = false;

  PROPERTY(int32_t, limit_image_per_prompt) = 8;

  PROPERTY(std::string, mm_args) = "{}";

  PROPERTY(int32_t, fps_min_frames) = 4;

  PROPERTY(int32_t, fps_max_frames) = 768;

  PROPERTY(int32_t, video_max_token_num) = -1;

  PROPERTY(int64_t, max_encoder_cache_size) = 0;

  PROPERTY(int64_t, max_processor_cache_items) = 256;

  PROPERTY(bool, enable_return_mm_full_embeddings) = false;

  PROPERTY(std::string, mm_download_headers) = "";

  PROPERTY(bool, use_audio_in_video) = false;

  PROPERTY(int32_t, audio_max_upload_file_mb) = 25;

  PROPERTY(int32_t, audio_max_decode_duration_s) = 600;

  PROPERTY(bool, enable_mm_encoder_dp) = false;

 private:
  nlohmann::json processor_args_ = nlohmann::json::object();
};

}  // namespace xllm
