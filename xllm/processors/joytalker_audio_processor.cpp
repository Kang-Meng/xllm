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

#include "processors/joytalker_audio_processor.h"

#include <glog/logging.h>

namespace xllm {

JoyTalkerAudioProcessor::JoyTalkerAudioProcessor(const ModelArgs& args)
    : Qwen3OmniAudioProcessor(args), hop_length_(args.mm_audio_hop_length()) {}

bool JoyTalkerAudioProcessor::process(const torch::Tensor& origin_audio,
                                      const AudioMetadata& metadata,
                                      MMDataItem& output_item) const {
  // The codec decodes audio to host float32.
  torch::Tensor wave = origin_audio.to(torch::kCPU, torch::kFloat32);
  if (wave.dim() == 2 && wave.size(0) == 1) {
    wave = wave.squeeze(0);
  }
  if (wave.dim() != 1) {
    LOG(ERROR) << "Only mono-channel audio is supported";
    return false;
  }
  const int64_t valid_length = wave.size(0);
  if (valid_length == 0) {
    LOG(ERROR) << "Audio is empty";
    return false;
  }

  // JoyTalker preprocess_audio: wave / (max|wave| + 1e-8) before padding.
  const double max_abs = wave.abs().max().item<double>();
  wave = wave / (max_abs + 1e-8);

  // Keep the partial tail hop: zero-pad it to a full hop so the inherited
  // trim is a no-op. The extra zeros stay inside the last valid frame, so the
  // extracted features are unchanged.
  const int64_t padded_length =
      (valid_length + hop_length_ - 1) / hop_length_ * hop_length_;
  if (padded_length > valid_length) {
    wave = torch::cat(
        {wave, torch::zeros({padded_length - valid_length}, wave.options())},
        /*dim=*/0);
  }
  return Qwen3OmniAudioProcessor::process(wave, metadata, output_item);
}

}  // namespace xllm
