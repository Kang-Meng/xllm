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

#include <cstdint>
#include <string>
#include <vector>

#include "core/common/constants.h"
#include "core/framework/model/model_args.h"
#include "processors/prompt_processor.h"

namespace xllm {

class Tokenizer;

// Expands the DeepSeek-V4.1 image placeholder into the sentinel block the model
// expects: every non-pad slot uses the native `<｜deepseek_image｜>` token
// (129264) and the compressor-alignment pads use
// `<|place_holder_mm_span_0436|>` (129265), laid out in the N order of
// `vllm/models/deepseek_v4/common/mm_preprocess.py:build_image_block`. The
// Python model then scatters the ViT+aligner rows at the 129264 positions.
class DeepseekV41PromptProcessor final : public PromptProcessor {
 public:
  explicit DeepseekV41PromptProcessor(const ModelArgs& args);

  // Expands one placeholder per `image_grid_thw` entry, in order. A shortfall
  // (fewer placeholders than images) is rejected: the leftover images would
  // have no span and their embeddings would be dropped. A surplus (more
  // placeholders than images, e.g. a literal `<image>` in user text) is NOT
  // rejected -- a benign marker in prose must not fail the whole request -- but
  // stripped from the tail with a WARNING, so the silent correction stays
  // observable.
  bool process(std::string& prompt, const MMData& mm_data) override;
  // Prepends `COMPRESS_PAD_TO - 1 - start_idx % COMPRESS_PAD_TO` IMAGE_PAD
  // sentinels before every image block so its first token lands at
  // `COMPRESS_PAD_TO - 1 (mod COMPRESS_PAD_TO)` -- the last slot of the
  // compressor's 4-token run boundary (mirrors the online splice in
  // `mm_preprocess.py:_apply_token_matches_with_placeholders`).
  bool adjust_token_layout(std::vector<int32_t>& token_ids,
                           const MMData& mm_data) override;
  bool find_mm_spans(const std::vector<int32_t>& token_ids,
                     MMData& mm_data) override;

 private:
  // Block role sequence for one image; see `deepseek_v41_image_processor.cpp`.
  std::vector<char> image_block_roles(int64_t n_vit_h, int64_t n_vit_w) const;

  // Number of sentinel (non-pad) tokens in one image block, derived from a
  // per-image `image_grid_thw` tensor. -1 when the grid is unusable.
  int32_t block_sentinel_count(const torch::Tensor& grid) const;

  int32_t image_token_id_ = -1;
  int32_t image_pad_id_ = -1;
  int64_t downsample_ratio_ = 3;

  // Native placeholder (U+FF5C fullwidth bars) and its generic alias. The
  // native spelling is the shared constant the C++ chat template injects, so
  // template and processor cannot drift.
  const std::string image_token_ = kDeepseekV41ImagePlaceholder;
  // Splice-time pad token inserted between image blocks so each block's first
  // token lands at `kDeepseekV41CompressPadTo - 1 (mod
  // kDeepseekV41CompressPadTo)`. It is fixed by the
  // DeepSeek-V4.1-Flash tokenizer/checkpoint (the compressor reserves these
  // span ids), not a per-request or per-model knob, so it lives here as a
  // constant rather than being threaded through ModelArgs.
  const std::string image_pad_token_ = kDeepseekV41ImagePadPlaceholder;
  const std::string image_alias_ = kDeepseekV41ImageAlias;
};

// Checks that the tokenizer spells the two protocol tokens as the configured
// image id and image id + 1. Returns true when image_token_id is negative
// (text-only checkpoint) because no image expansion will occur.
bool validate_deepseek_v41_tokenizer_binding(const Tokenizer& tokenizer,
                                             int32_t image_token_id);

}  // namespace xllm
