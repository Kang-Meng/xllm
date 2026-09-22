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

#include "core/framework/request/dit_request_parser.h"

#include <butil/strings/string_number_conversions.h>
#include <butil/strings/string_piece.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "audio_generation.pb.h"
#include "core/common/types.h"
#include "core/framework/config/dit_config.h"
#include "core/framework/request/dit_request_params.h"
#include "core/framework/request/dit_request_state.h"
#include "core/framework/request/dit_source_decoder.h"
#include "core/runtime/params_utils.h"
#include "core/util/binary_payload.h"
#include "core/util/utils.h"
#include "image_generation.pb.h"
#include "text_generation.pb.h"
#include "video_generation.pb.h"

namespace xllm {

namespace {

template <typename Proto, typename Variant>
const Proto* get_proto(const Variant& value,
                       std::string_view name,
                       Status& status) {
  const auto* proto_ptr = std::get_if<const Proto*>(&value);
  if (proto_ptr == nullptr) {
    status = Status(StatusCode::INVALID_ARGUMENT,
                    std::string(name) + " type does not match parser");
    return nullptr;
  }
  if (*proto_ptr == nullptr) {
    status =
        Status(StatusCode::INVALID_ARGUMENT, std::string(name) + " is missing");
    return nullptr;
  }
  return *proto_ptr;
}

bool parse_resolution(std::string_view resolution,
                      int32_t& width,
                      int32_t& height) {
  const size_t separator = resolution.find('*');
  if (separator == std::string_view::npos || separator == 0 ||
      separator + 1 == resolution.size() ||
      resolution.find('*', separator + 1) != std::string_view::npos) {
    return false;
  }

  const butil::StringPiece width_text(resolution.data(), separator);
  const butil::StringPiece height_text(resolution.data() + separator + 1,
                                       resolution.size() - separator - 1);
  size_t parsed_width = 0;
  size_t parsed_height = 0;
  if (!butil::StringToSizeT(width_text, &parsed_width) ||
      !butil::StringToSizeT(height_text, &parsed_height) ||
      parsed_width > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
      parsed_height >
          static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return false;
  }
  width = static_cast<int32_t>(parsed_width);
  height = static_cast<int32_t>(parsed_height);
  return true;
}

bool decode_tensor_source(std::string_view name,
                          const proto::Tensor& proto_tensor,
                          const BinaryPayload& binary_payload,
                          DiTTensorSources& tensor_sources,
                          Status& status) {
  torch::Tensor tensor = util::proto_to_torch(proto_tensor, binary_payload);
  if (!tensor.defined()) {
    status = Status(StatusCode::INVALID_ARGUMENT,
                    "invalid " + std::string(name) + " tensor");
    return false;
  }
  std::optional<TensorParameters> parameters =
      proto_to_tensor_parameters(proto_tensor);
  if (!parameters.has_value()) {
    status = Status(StatusCode::INVALID_ARGUMENT,
                    "invalid " + std::string(name) + " tensor parameters");
    return false;
  }
  tensor_sources.add(
      std::string(name), std::move(tensor), std::move(*parameters));
  return true;
}

}  // namespace

bool ImageRequestParser::parse_parameters(
    const DiTParametersProto& parameters,
    DiTGenerationParams& generation_params,
    Status& status) const {
  const auto* proto_ptr =
      get_proto<proto::Parameters>(parameters, "parameters", status);
  if (proto_ptr == nullptr) {
    return false;
  }
  const auto& params = *proto_ptr;
  if (params.has_size() &&
      !parse_resolution(
          params.size(), generation_params.width, generation_params.height)) {
    status = Status(StatusCode::INVALID_ARGUMENT,
                    "size must use the width*height format");
    return false;
  }
  if (params.has_num_inference_steps()) {
    generation_params.num_inference_steps = params.num_inference_steps();
  }
  if (params.has_true_cfg_scale()) {
    generation_params.true_cfg_scale = params.true_cfg_scale();
  }
  if (params.has_guidance_scale()) {
    generation_params.guidance_scale = params.guidance_scale();
  }
  if (params.has_num_images_per_prompt()) {
    generation_params.num_images_per_prompt = params.num_images_per_prompt();
  }
  if (params.has_seed()) {
    generation_params.seed = params.seed();
    generation_params.seed_is_set = true;
  }
  if (params.has_max_sequence_length()) {
    generation_params.max_sequence_length = params.max_sequence_length();
  }
  if (params.has_enable_cfg_renorm()) {
    generation_params.enable_cfg_renorm = params.enable_cfg_renorm();
  }
  if (params.has_cfg_renorm_min()) {
    generation_params.cfg_renorm_min = params.cfg_renorm_min();
  }
  if (params.has_output_type()) {
    generation_params.output_type = params.output_type();
  }
  return true;
}

bool ImageRequestParser::parse_input(
    const DiTRawInputParams& raw_input_params,
    const DiTGenerationParams& /*generation_params*/,
    DiTInputParams& input_params,
    Status& status) const {
  const auto* proto_ptr =
      get_proto<proto::Input>(raw_input_params.input, "input", status);
  if (proto_ptr == nullptr) {
    return false;
  }
  const auto& input = *proto_ptr;
  const BinaryPayload& binary_payload = raw_input_params.binary_payload;
  input_params.prompt = input.prompt();
  input_params.prompt_2 = input.prompt_2();
  input_params.negative_prompt = input.negative_prompt();
  input_params.negative_prompt_2 = input.negative_prompt_2();
  if (input.has_prompt_embed() &&
      !decode_tensor_source("prompt_embed",
                            input.prompt_embed(),
                            binary_payload,
                            input_params.tensor_sources,
                            status)) {
    return false;
  }
  if (input.has_pooled_prompt_embed() &&
      !decode_tensor_source("pooled_prompt_embed",
                            input.pooled_prompt_embed(),
                            binary_payload,
                            input_params.tensor_sources,
                            status)) {
    return false;
  }
  if (input.has_negative_prompt_embed() &&
      !decode_tensor_source("negative_prompt_embed",
                            input.negative_prompt_embed(),
                            binary_payload,
                            input_params.tensor_sources,
                            status)) {
    return false;
  }
  if (input.has_negative_pooled_prompt_embed() &&
      !decode_tensor_source("negative_pooled_prompt_embed",
                            input.negative_pooled_prompt_embed(),
                            binary_payload,
                            input_params.tensor_sources,
                            status)) {
    return false;
  }
  if (input.has_latent() && !decode_tensor_source("latent",
                                                  input.latent(),
                                                  binary_payload,
                                                  input_params.tensor_sources,
                                                  status)) {
    return false;
  }
  if (input.has_masked_image_latent() &&
      !decode_tensor_source("masked_image_latent",
                            input.masked_image_latent(),
                            binary_payload,
                            input_params.tensor_sources,
                            status)) {
    return false;
  }

  DiTSourceDecoder decoder(binary_payload);
  // Preserve reference-image order: media_sources before legacy images.
  if (!decoder.add_sources(input.media_sources(), status)) {
    return false;
  }
  decoder.add_sources(input.images(), /*default_name=*/"unknown");

  std::vector<MediaNamedTensor> decoded_sources;
  if (!decoder.decode(decoded_sources, status)) {
    return false;
  }
  for (MediaNamedTensor& source : decoded_sources) {
    input_params.media_sources.add(std::move(source.name),
                                   std::move(source.modality),
                                   std::move(source.tensor),
                                   std::move(source.parameters));
  }
  return true;
}

bool ImageRequestParser::verify_params(
    const DiTGenerationParams& generation_params,
    Status& status) const {
  if (generation_params.width <= 0 || generation_params.height <= 0) {
    status =
        Status(StatusCode::INVALID_ARGUMENT,
               "Invalid image dimensions: width and height must be positive.");
    return false;
  }

  if (generation_params.num_inference_steps <= 0) {
    status = Status(StatusCode::INVALID_ARGUMENT,
                    "num_inference_steps must be positive.");
    return false;
  }

  if (generation_params.num_images_per_prompt <= 0) {
    status = Status(StatusCode::INVALID_ARGUMENT,
                    "num_images_per_prompt must be greater than 0.");
    return false;
  }

  if (generation_params.output_type != "base64" &&
      generation_params.output_type != "binary") {
    status = Status(StatusCode::INVALID_ARGUMENT,
                    "output_type must be either base64 or binary");
    return false;
  }

  // Check if the image area exceeds the maximum allowed area.
  if (::xllm::DiTConfig::get_instance().dit_generation_image_area_max() > 0) {
    int64_t area = static_cast<int64_t>(generation_params.width) *
                   static_cast<int64_t>(generation_params.height);
    if (area >
        ::xllm::DiTConfig::get_instance().dit_generation_image_area_max()) {
      status = Status(StatusCode::INVALID_ARGUMENT,
                      "Requested image area (" + std::to_string(area) +
                          ") exceeds the maximum allowed area (" +
                          std::to_string(::xllm::DiTConfig::get_instance()
                                             .dit_generation_image_area_max()) +
                          ").");
      return false;
    }
  }

  return true;
}

bool AudioRequestParser::parse_parameters(
    const DiTParametersProto& parameters,
    DiTGenerationParams& generation_params,
    Status& status) const {
  const auto* proto_ptr =
      get_proto<proto::AudioParameters>(parameters, "parameters", status);
  if (proto_ptr == nullptr) {
    return false;
  }
  const auto& params = *proto_ptr;
  if (params.has_seed()) {
    generation_params.seed = params.seed();
    generation_params.seed_is_set = true;
  }
  if (params.has_max_sequence_length()) {
    generation_params.max_sequence_length = params.max_sequence_length();
  }
  if (params.has_guidance_scale()) {
    generation_params.guidance_scale = params.guidance_scale();
  }
  if (params.has_audio_duration_frames()) {
    generation_params.audio_duration_frames = params.audio_duration_frames();
  }
  if (params.has_audio_steps()) {
    generation_params.audio_steps = params.audio_steps();
  }
  if (params.has_audio_guidance_method()) {
    generation_params.audio_guidance_method = params.audio_guidance_method();
  }
  if (params.has_sampling_rate()) {
    generation_params.audio_sampling_rate = params.sampling_rate();
  }
  if (params.has_output_type()) {
    generation_params.output_type = params.output_type();
  }
  return true;
}

bool AudioRequestParser::parse_input(
    const DiTRawInputParams& raw_input_params,
    const DiTGenerationParams& generation_params,
    DiTInputParams& input_params,
    Status& status) const {
  const auto* proto_ptr =
      get_proto<proto::AudioInput>(raw_input_params.input, "input", status);
  if (proto_ptr == nullptr) {
    return false;
  }
  const auto& input = *proto_ptr;
  const BinaryPayload& binary_payload = raw_input_params.binary_payload;
  const int32_t audio_sampling_rate = generation_params.audio_sampling_rate;
  input_params.prompt = input.prompt();
  if (input.has_prompt_audio()) {
    DiTSourceDecoder decoder(binary_payload);
    if (!decoder.add_source(input.prompt_audio(),
                            /*name=*/"prompt_audio",
                            /*modality=*/"audio",
                            status)) {
      return false;
    }
    std::vector<MediaNamedTensor> decoded_sources;
    if (!decoder.decode(decoded_sources,
                        status,
                        /*audio_channels=*/1,
                        audio_sampling_rate)) {
      return false;
    }
    MediaNamedTensor& decoded_source = decoded_sources.front();
    input_params.media_sources.add("prompt_audio",
                                   std::move(decoded_source.modality),
                                   std::move(decoded_source.tensor),
                                   std::move(decoded_source.parameters));
  }
  input_params.audio_prompt_text = input.prompt_text();
  return true;
}

bool AudioRequestParser::verify_params(
    const DiTGenerationParams& generation_params,
    Status& status) const {
  if (generation_params.output_type != "base64" &&
      generation_params.output_type != "binary") {
    status = Status(StatusCode::INVALID_ARGUMENT,
                    "output_type must be either base64 or binary");
    return false;
  }
  return true;
}

bool VideoRequestParser::parse_parameters(
    const DiTParametersProto& parameters,
    DiTGenerationParams& generation_params,
    Status& status) const {
  const auto* proto_ptr =
      get_proto<proto::VideoParameters>(parameters, "parameters", status);
  if (proto_ptr == nullptr) {
    return false;
  }
  const auto& params = *proto_ptr;
  if (params.has_size() &&
      !parse_resolution(
          params.size(), generation_params.width, generation_params.height)) {
    status = Status(StatusCode::INVALID_ARGUMENT,
                    "size must use the width*height format");
    return false;
  }
  if (params.has_num_inference_steps()) {
    generation_params.num_inference_steps = params.num_inference_steps();
  }
  if (params.has_guidance_scale()) {
    generation_params.guidance_scale = params.guidance_scale();
  }
  if (params.has_guidance_scale_2()) {
    generation_params.guidance_scale_2 = params.guidance_scale_2();
  }
  if (params.has_num_videos_per_prompt()) {
    generation_params.num_videos_per_prompt = params.num_videos_per_prompt();
  }
  if (params.has_seed()) {
    generation_params.seed = params.seed();
    generation_params.seed_is_set = true;
  }
  if (params.has_max_sequence_length()) {
    generation_params.max_sequence_length = params.max_sequence_length();
  }
  if (params.has_num_frames()) {
    generation_params.num_frames = params.num_frames();
  }
  if (params.has_fps()) {
    generation_params.video_fps = params.fps();
  }
  if (params.has_seconds()) {
    generation_params.seconds = params.seconds();
  }
  if (params.has_boundary_ratio()) {
    generation_params.boundary_ratio = params.boundary_ratio();
  }
  if (params.has_flow_shift()) {
    generation_params.flow_shift = params.flow_shift();
  }
  if (params.has_true_cfg_scale()) {
    generation_params.true_cfg_scale = params.true_cfg_scale();
  }
  if (params.has_output_type()) {
    generation_params.output_type = params.output_type();
  }
  if (params.has_sampling_rate()) {
    generation_params.audio_sampling_rate = params.sampling_rate();
  }
  return true;
}

bool VideoRequestParser::parse_input(
    const DiTRawInputParams& raw_input_params,
    const DiTGenerationParams& generation_params,
    DiTInputParams& input_params,
    Status& status) const {
  const auto* proto_ptr =
      get_proto<proto::VideoInput>(raw_input_params.input, "input", status);
  if (proto_ptr == nullptr) {
    return false;
  }
  const auto& input = *proto_ptr;
  const BinaryPayload& binary_payload = raw_input_params.binary_payload;
  const int32_t audio_sampling_rate = generation_params.audio_sampling_rate;
  input_params.prompt = input.prompt();
  input_params.negative_prompt = input.negative_prompt();
  if (input.has_prompt_embed() &&
      !decode_tensor_source("prompt_embed",
                            input.prompt_embed(),
                            binary_payload,
                            input_params.tensor_sources,
                            status)) {
    return false;
  }
  if (input.has_negative_prompt_embed() &&
      !decode_tensor_source("negative_prompt_embed",
                            input.negative_prompt_embed(),
                            binary_payload,
                            input_params.tensor_sources,
                            status)) {
    return false;
  }

  DiTSourceDecoder decoder(binary_payload);
  if (!decoder.add_sources(input.media_sources(), status)) {
    return false;
  }
  std::vector<MediaNamedTensor> decoded_sources;
  if (!decoder.decode(decoded_sources,
                      status,
                      /*audio_channels=*/2,
                      audio_sampling_rate)) {
    return false;
  }
  for (MediaNamedTensor& source : decoded_sources) {
    input_params.media_sources.add(std::move(source.name),
                                   std::move(source.modality),
                                   std::move(source.tensor),
                                   std::move(source.parameters));
  }
  return true;
}

bool VideoRequestParser::verify_params(
    const DiTGenerationParams& generation_params,
    Status& status) const {
  if (generation_params.width <= 0 || generation_params.height <= 0) {
    status =
        Status(StatusCode::INVALID_ARGUMENT,
               "Invalid image dimensions: width and height must be positive.");
    return false;
  }

  if (generation_params.num_inference_steps <= 0) {
    status = Status(StatusCode::INVALID_ARGUMENT,
                    "num_inference_steps must be positive.");
    return false;
  }

  if (generation_params.num_videos_per_prompt <= 0) {
    status = Status(StatusCode::INVALID_ARGUMENT,
                    "num_videos_per_prompt must be greater than 0.");
    return false;
  }

  if (generation_params.output_type != "base64" &&
      generation_params.output_type != "binary") {
    status = Status(StatusCode::INVALID_ARGUMENT,
                    "output_type must be either base64 or binary");
    return false;
  }

  // Check if the image area exceeds the maximum allowed area.
  if (::xllm::DiTConfig::get_instance().dit_generation_image_area_max() > 0) {
    int64_t area = static_cast<int64_t>(generation_params.width) *
                   static_cast<int64_t>(generation_params.height);
    if (area >
        ::xllm::DiTConfig::get_instance().dit_generation_image_area_max()) {
      status = Status(StatusCode::INVALID_ARGUMENT,
                      "Requested image area (" + std::to_string(area) +
                          ") exceeds the maximum allowed area (" +
                          std::to_string(::xllm::DiTConfig::get_instance()
                                             .dit_generation_image_area_max()) +
                          ").");
      return false;
    }
  }

  return true;
}

bool TextRequestParser::parse_parameters(const DiTParametersProto& parameters,
                                         DiTGenerationParams& generation_params,
                                         Status& status) const {
  const auto* proto_ptr =
      get_proto<proto::TextParameters>(parameters, "parameters", status);
  if (proto_ptr == nullptr) {
    return false;
  }
  const auto& params = *proto_ptr;
  if (params.has_seed()) {
    generation_params.seed = params.seed();
    generation_params.seed_is_set = true;
  }
  if (params.has_max_new_tokens()) {
    generation_params.max_new_tokens = params.max_new_tokens();
  }
  if (params.has_diffusion_steps()) {
    generation_params.diffusion_steps = params.diffusion_steps();
  }
  if (params.has_guidance_scale()) {
    generation_params.guidance_scale = params.guidance_scale();
  }
  if (params.has_temperature()) {
    generation_params.temperature = params.temperature();
  }
  if (params.has_top_k()) {
    generation_params.top_k = params.top_k();
  }
  if (params.has_top_p()) {
    generation_params.top_p = params.top_p();
  }
  if (params.has_repetition_penalty()) {
    generation_params.repetition_penalty = params.repetition_penalty();
  }
  return true;
}

bool TextRequestParser::parse_input(
    const DiTRawInputParams& raw_input_params,
    const DiTGenerationParams& /*generation_params*/,
    DiTInputParams& input_params,
    Status& status) const {
  const auto* proto_ptr =
      get_proto<proto::TextInput>(raw_input_params.input, "input", status);
  if (proto_ptr == nullptr) {
    return false;
  }
  const auto& input = *proto_ptr;
  input_params.prompt = input.prompt();
  return true;
}

bool TextRequestParser::verify_params(
    const DiTGenerationParams& generation_params,
    Status& status) const {
  if (generation_params.max_new_tokens <= 0) {
    status = Status(StatusCode::INVALID_ARGUMENT,
                    "max_new_tokens must be positive.");
    return false;
  }
  if (generation_params.diffusion_steps <= 0) {
    status = Status(StatusCode::INVALID_ARGUMENT,
                    "diffusion_steps must be positive.");
    return false;
  }
  return true;
}

}  // namespace xllm
