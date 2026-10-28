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

#include "core/framework/config/multimodal_config.h"

#include <glog/logging.h>

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"

DEFINE_bool(use_ctc,
            false,
            "Enable the CTC head of audio models whose checkpoint ships "
            "one. The server renders the CTC chat-template region "
            "(context variable use_ctc, default off) only while this is "
            "on, so a client must not emit its own CTC segment.");

DEFINE_bool(enable_return_embedding_modality_tags,
            false,
            "For embedding (VLM) workers: append per-token modality tags to "
            "full multimodal embedding outputs.");

DEFINE_int32(limit_image_per_prompt,
             8,
             "Maximum number of image per prompt. Only applicable for "
             "multimodal models.");

DEFINE_string(mm_args,
              "{}",
              "Service-level processor overrides grouped by modality as JSON. "
              "Fields are interpreted by the selected native processor. "
              "Example: '{\"image\":{\"max_pixels\":1048576}}'");

DEFINE_int32(fps_min_frames,
             4,
             "Minimum number of frames sampled from a video when sampling "
             "with fps (qwen_omni_utils FPS_MIN_FRAMES). Only applicable for "
             "Qwen omni-series multimodal models.");

DEFINE_int32(fps_max_frames,
             768,
             "Maximum number of frames sampled from a video when sampling "
             "with fps (qwen_omni_utils FPS_MAX_FRAMES). Only applicable for "
             "Qwen omni-series multimodal models.");

DEFINE_int32(video_max_token_num,
             -1,
             "Per-frame token budget for video frames; the pixel budget is "
             "video_max_token_num * (patch_size * merge_size)^2. -1 uses "
             "video_max_pixels from the model configuration.");

DEFINE_int64(max_encoder_cache_size,
             0,
             "Max gpu/npu memory size in MB for encoder cache per worker. "
             "Default is 0, which disables encoder cache.");

DEFINE_int64(max_processor_cache_items,
             256,
             "Maximum number of multimodal processor results cached on the "
             "master. Capacity is counted by item. "
             "Default is 256; set to 0 to disable processor cache.");

DEFINE_bool(enable_return_mm_full_embeddings,
            false,
            "return vit and sequence embeddings for vlm models");

DEFINE_string(mm_download_headers,
              "",
              "Service-level default HTTP headers for multimodal downloads, "
              "as a JSON object. Per-request headers take precedence. "
              "Example: '{\"Authorization\":\"Bearer xxx\"}'");

DEFINE_bool(
    use_audio_in_video,
    false,
    "Whether to decode both audio and video when the input is a video.");

DEFINE_int32(audio_max_upload_file_mb,
             25,
             "Maximum accepted speech-to-text audio file size in MiB.");

DEFINE_int32(audio_max_decode_duration_s,
             600,
             "Maximum decoded speech-to-text audio duration in seconds. "
             "0 disables the limit.");

DEFINE_bool(enable_mm_encoder_dp,
            false,
            "Enable encoder data parallelism for multi-modal models.");

namespace xllm {

void MultimodalConfig::from_flags() {
  XLLM_CONFIG_ASSIGN_FROM_FLAG(use_ctc);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_return_embedding_modality_tags);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(limit_image_per_prompt);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(mm_args);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(fps_min_frames);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(fps_max_frames);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(video_max_token_num);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(max_encoder_cache_size);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(max_processor_cache_items);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_return_mm_full_embeddings);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(mm_download_headers);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(use_audio_in_video);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(audio_max_upload_file_mb);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(audio_max_decode_duration_s);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_mm_encoder_dp);
}

void MultimodalConfig::from_json(const JsonReader& json) {
  XLLM_CONFIG_ASSIGN_FROM_JSON(use_ctc);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_return_embedding_modality_tags);
  XLLM_CONFIG_ASSIGN_FROM_JSON(limit_image_per_prompt);
  XLLM_CONFIG_ASSIGN_FROM_JSON(mm_args);
  XLLM_CONFIG_ASSIGN_FROM_JSON(fps_min_frames);
  XLLM_CONFIG_ASSIGN_FROM_JSON(fps_max_frames);
  XLLM_CONFIG_ASSIGN_FROM_JSON(video_max_token_num);
  XLLM_CONFIG_ASSIGN_FROM_JSON(max_encoder_cache_size);
  XLLM_CONFIG_ASSIGN_FROM_JSON(max_processor_cache_items);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_return_mm_full_embeddings);
  XLLM_CONFIG_ASSIGN_FROM_JSON(mm_download_headers);
  XLLM_CONFIG_ASSIGN_FROM_JSON(use_audio_in_video);
  XLLM_CONFIG_ASSIGN_FROM_JSON(audio_max_upload_file_mb);
  XLLM_CONFIG_ASSIGN_FROM_JSON(audio_max_decode_duration_s);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_mm_encoder_dp);
}

void MultimodalConfig::append_config_json(
    nlohmann::ordered_json& config_json) const {
  const MultimodalConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(config_json, default_config, use_ctc);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_return_embedding_modality_tags);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, limit_image_per_prompt);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(config_json, default_config, mm_args);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, fps_min_frames);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, fps_max_frames);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, video_max_token_num);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, max_encoder_cache_size);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, max_processor_cache_items);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_return_mm_full_embeddings);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, mm_download_headers);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, use_audio_in_video);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, audio_max_upload_file_mb);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, audio_max_decode_duration_s);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_mm_encoder_dp);
}

MultimodalConfig& MultimodalConfig::get_instance() {
  static MultimodalConfig config;
  return config;
}

void MultimodalConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
  processor_args_ = nlohmann::json::object();
  if (mm_args().empty()) {
    return;
  }
  processor_args_ = nlohmann::json::parse(mm_args(), nullptr, false);
  CHECK(!processor_args_.is_discarded()) << "mm_args must be valid JSON";
  CHECK(processor_args_.is_object())
      << "mm_args must be a JSON object, got " << processor_args_.type_name();
  for (const auto& [modality, values] : processor_args_.items()) {
    CHECK(modality == "image" || modality == "video" || modality == "audio")
        << "Unsupported mm_args modality: " << modality;
    CHECK(values.is_object()) << "mm_args." << modality << " must be an object";
  }
}

const nlohmann::json& MultimodalConfig::processor_args(
    const std::string& modality) const {
  static const nlohmann::json kEmptyArgs = nlohmann::json::object();
  const auto found = processor_args_.find(modality);
  return found == processor_args_.end() ? kEmptyArgs : *found;
}

}  // namespace xllm
