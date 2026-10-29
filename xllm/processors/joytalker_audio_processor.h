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

#include <torch/torch.h>

#include <cstdint>

#include "processors/qwen3_omni_audio_processor.h"

namespace xllm {

// JoyTalker front end: peak-normalize the waveform and keep the partial tail
// hop, then reuse Qwen3-Omni's Whisper-style padding/mel/mask.
class JoyTalkerAudioProcessor final : public Qwen3OmniAudioProcessor {
 public:
  explicit JoyTalkerAudioProcessor(const ModelArgs& args);

  bool process(const torch::Tensor& origin_audio,
               const AudioMetadata& metadata,
               MMDataItem& output_item) const override;

 private:
  int64_t hop_length_;
};

}  // namespace xllm
