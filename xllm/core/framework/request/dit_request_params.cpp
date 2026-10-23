/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "core/framework/request/dit_request_params.h"

#include <string>
#include <utility>
#include <variant>

#include "core/common/instance_name.h"
#include "core/framework/request/dit_request_parser.h"
#include "core/util/uuid.h"

namespace xllm {
namespace {
thread_local ShortUUID short_uuid;

const ImageRequestParser kImageParser;
const AudioRequestParser kAudioParser;
const VideoRequestParser kVideoParser;
const TextRequestParser kTextParser;

std::string generate_request_id(const std::string& prefix) {
  return prefix + InstanceName::name()->get_name_hash() + "-" +
         short_uuid.random();
}

const DiTRequestParser* get_parser(DiTRequestKind request_kind) {
  switch (request_kind) {
    case DiTRequestKind::kImage:
      return &kImageParser;
    case DiTRequestKind::kAudio:
      return &kAudioParser;
    case DiTRequestKind::kVideo:
      return &kVideoParser;
    case DiTRequestKind::kText:
      return &kTextParser;
    default:
      return nullptr;
  }
}

}  // namespace

DiTRequestParams::DiTRequestParams(DiTRequestKind request_kind)
    : request_kind_(request_kind), parser_(get_parser(request_kind)) {}

// ---------------------------------------------------------------------------
// Image generation constructor
// ---------------------------------------------------------------------------
DiTRequestParams::DiTRequestParams(const proto::ImageGenerationRequest& request,
                                   const std::string& x_rid,
                                   const std::string& x_rtime,
                                   BinaryPayload request_payload)
    : DiTRequestParams(DiTRequestKind::kImage) {
  request_id = request.has_request_id() ? request.request_id()
                                        : generate_request_id("imggen-");
  x_request_id = x_rid;
  x_request_time = x_rtime;
  raw_input_params.binary_payload = std::move(request_payload);
  if (request.has_input()) {
    raw_input_params.input = &request.input();
  }

  if (request.has_parameters()) {
    parser_->parse_parameters(
        &request.parameters(), generation_params, request_parse_status);
  }
}

// ---------------------------------------------------------------------------
// Text generation constructor (Cola-DLM)
// ---------------------------------------------------------------------------
DiTRequestParams::DiTRequestParams(const proto::TextGenerationRequest& request,
                                   const std::string& x_rid,
                                   const std::string& x_rtime)
    : DiTRequestParams(DiTRequestKind::kText) {
  // Cola-DLM official defaults (inference.py / run_cola.py), not image DiT.
  generation_params.guidance_scale = 7.0f;
  generation_params.max_new_tokens = 32;
  request_id = request.has_request_id() ? request.request_id()
                                        : generate_request_id("textgen-");
  x_request_id = x_rid;
  x_request_time = x_rtime;

  if (request.has_input()) {
    raw_input_params.input = &request.input();
  }

  if (request.has_parameters()) {
    parser_->parse_parameters(
        &request.parameters(), generation_params, request_parse_status);
  }
}

// ---------------------------------------------------------------------------
// Audio generation constructor
// ---------------------------------------------------------------------------
DiTRequestParams::DiTRequestParams(const proto::AudioGenerationRequest& request,
                                   const std::string& x_rid,
                                   const std::string& x_rtime,
                                   BinaryPayload request_payload)
    : DiTRequestParams(DiTRequestKind::kAudio) {
  if (request.has_request_id()) {
    request_id = request.request_id();
  } else {
    request_id = generate_request_id("audiogen-");
  }
  x_request_id = x_rid;
  x_request_time = x_rtime;
  raw_input_params.binary_payload = std::move(request_payload);

  if (request.has_input()) {
    raw_input_params.input = &request.input();
  }

  if (request.has_parameters()) {
    parser_->parse_parameters(
        &request.parameters(), generation_params, request_parse_status);
  }
}

// ---------------------------------------------------------------------------
// Video generation constructor
// ---------------------------------------------------------------------------
DiTRequestParams::DiTRequestParams(const proto::VideoGenerationRequest& request,
                                   const std::string& x_rid,
                                   const std::string& x_rtime,
                                   BinaryPayload request_payload)
    : DiTRequestParams(DiTRequestKind::kVideo) {
  request_id = request.has_request_id() ? request.request_id()
                                        : generate_request_id("vidgen-");
  x_request_id = x_rid;
  x_request_time = x_rtime;
  raw_input_params.binary_payload = std::move(request_payload);
  if (request.has_input()) {
    raw_input_params.input = &request.input();
  }

  generation_params.audio_sampling_rate = 32000;
  if (request.has_parameters()) {
    parser_->parse_parameters(
        &request.parameters(), generation_params, request_parse_status);
  }
}

bool DiTRequestParams::parse_input_params(Status& status,
                                          bool preserve_image_alpha) {
  if (std::holds_alternative<std::monostate>(raw_input_params.input)) {
    return true;
  }

  if (parser_ == nullptr) {
    status = Status(StatusCode::INVALID_ARGUMENT, "request kind is invalid");
    return false;
  }

  DiTInputParams parsed_input_params;
  if (!parser_->parse_input(raw_input_params,
                            generation_params,
                            parsed_input_params,
                            status,
                            preserve_image_alpha)) {
    return false;
  }

  input_params = std::move(parsed_input_params);
  return true;
}

bool DiTRequestParams::verify_params(Status& status) const {
  status = request_parse_status;
  if (!status.ok()) {
    return false;
  }

  if (parser_ == nullptr) {
    status = Status(StatusCode::INVALID_ARGUMENT, "request kind is invalid");
    return false;
  }

  if (input_params.prompt.empty() &&
      !input_params.tensor_sources.contains("prompt_embed")) {
    status = Status(StatusCode::INVALID_ARGUMENT, "prompt is empty");
    return false;
  }

  return parser_->verify_params(generation_params, status);
}

}  // namespace xllm
