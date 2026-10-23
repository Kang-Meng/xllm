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

#include <variant>

namespace xllm {

class Status;
struct DiTGenerationParams;
struct DiTInputParams;
struct DiTRawInputParams;
namespace proto {
class Parameters;
class AudioParameters;
class VideoParameters;
class TextParameters;
}  // namespace proto

// Non-owning pointers used only for the duration of parameter parsing.
using DiTParametersProto = std::variant<const proto::Parameters*,
                                        const proto::AudioParameters*,
                                        const proto::VideoParameters*,
                                        const proto::TextParameters*>;

class DiTRequestParser {
 public:
  virtual ~DiTRequestParser() = default;

  virtual bool parse_parameters(const DiTParametersProto& parameters,
                                DiTGenerationParams& generation_params,
                                Status& status) const = 0;
  // Only image parsing uses the model-selected alpha policy.
  virtual bool parse_input(const DiTRawInputParams& raw_input_params,
                           const DiTGenerationParams& generation_params,
                           DiTInputParams& input_params,
                           Status& status,
                           bool preserve_image_alpha = false) const = 0;

  virtual bool verify_params(const DiTGenerationParams& generation_params,
                             Status& status) const = 0;
};

class ImageRequestParser final : public DiTRequestParser {
 public:
  bool parse_parameters(const DiTParametersProto& parameters,
                        DiTGenerationParams& generation_params,
                        Status& status) const override;
  bool parse_input(const DiTRawInputParams& raw_input_params,
                   const DiTGenerationParams& generation_params,
                   DiTInputParams& input_params,
                   Status& status,
                   bool preserve_image_alpha = false) const override;
  bool verify_params(const DiTGenerationParams& generation_params,
                     Status& status) const override;
};

class AudioRequestParser final : public DiTRequestParser {
 public:
  bool parse_parameters(const DiTParametersProto& parameters,
                        DiTGenerationParams& generation_params,
                        Status& status) const override;
  bool parse_input(const DiTRawInputParams& raw_input_params,
                   const DiTGenerationParams& generation_params,
                   DiTInputParams& input_params,
                   Status& status,
                   bool preserve_image_alpha = false) const override;
  bool verify_params(const DiTGenerationParams& generation_params,
                     Status& status) const override;
};

class VideoRequestParser final : public DiTRequestParser {
 public:
  bool parse_parameters(const DiTParametersProto& parameters,
                        DiTGenerationParams& generation_params,
                        Status& status) const override;
  bool parse_input(const DiTRawInputParams& raw_input_params,
                   const DiTGenerationParams& generation_params,
                   DiTInputParams& input_params,
                   Status& status,
                   bool preserve_image_alpha = false) const override;
  bool verify_params(const DiTGenerationParams& generation_params,
                     Status& status) const override;
};

class TextRequestParser final : public DiTRequestParser {
 public:
  bool parse_parameters(const DiTParametersProto& parameters,
                        DiTGenerationParams& generation_params,
                        Status& status) const override;
  bool parse_input(const DiTRawInputParams& raw_input_params,
                   const DiTGenerationParams& generation_params,
                   DiTInputParams& input_params,
                   Status& status,
                   bool preserve_image_alpha = false) const override;
  bool verify_params(const DiTGenerationParams& generation_params,
                     Status& status) const override;
};

}  // namespace xllm
