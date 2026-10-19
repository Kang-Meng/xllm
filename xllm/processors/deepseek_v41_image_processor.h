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
#include <vector>

#include "core/framework/model/model_args.h"
#include "processors/image_processor.h"

namespace xllm {

// DeepSeek-V4.1-Flash vision image processor. Mirrors the reference
// preprocess (`vllm/models/deepseek_v4/common/mm_preprocess.py:load_image`):
// min-pixel upscale / token-budget resize (`safe_resize`+`solve_resize_ratio`),
// grey (127) pad to the patch/downsample-aligned size, normalize to [-1, 1],
// then patchify to `[n_vit, C, patch, patch]` bf16. Emits one `MMDataItem` per
// image with `pixel_values` + a spatial `image_grid_thw` of `[n_vit_h,
// n_vit_w]` (two columns; the C++ bridge / Python `encode` read the last two).
class DeepseekV41ImageProcessor final : public ImageProcessor {
 public:
  explicit DeepseekV41ImageProcessor(const ModelArgs& args);

  bool process(const std::vector<torch::Tensor>& images,
               std::vector<MMDataItem>& output_items) const override;

 private:
  // Single-image pipeline: resize + pad + normalize + patchify. Writes the
  // `[n_vit, C, patch, patch]` bf16 patches into `pixel_values` and the
  // `[n_vit_h, n_vit_w]` grid into `grid`.
  void process_image(const torch::Tensor& image,
                     torch::Tensor& pixel_values,
                     torch::Tensor& grid) const;

  int64_t patch_size_ = 14;
  int64_t downsample_ratio_ = 3;
  // Input channel count the patch-embed weights expect (`mm_num_channels`, or 3
  // when the checkpoint leaves it unset). `process` rejects any other count so
  // a mismatched tensor fails here instead of at the patch-embed matmul.
  int64_t channels_ = 3;
  // LLM-token budget and pixel floor from vision_config (max_image_tokens /
  // min_pixels). 0 disables the corresponding clamp; a non-zero
  // max_image_tokens must leave room for the minimum image block plus reserved
  // splice pads (enforced in the constructor).
  int64_t max_n_token_ = 0;
  int64_t min_pixels_ = 0;
};

}  // namespace xllm
