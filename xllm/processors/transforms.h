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

#include <torch/torch.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace xllm::transforms {

torch::Tensor resize(const torch::Tensor& image,
                     const std::vector<int64_t>& size,
                     int32_t resample,
                     bool antialias = true);

// Bit-exact re-implementation of Pillow's BICUBIC resize for 8-bit (C, H, W)
// images (see libImaging/Resample.c). Pillow resamples 8-bit samples, so this
// port does too: uint8 input is used as-is, any other dtype is clamped to
// [0, 255] and quantized to uint8 first, and the result is always uint8. That
// quantization is lossy for genuinely sub-8-bit float images, which are out of
// scope here; the vision pipeline feeds 8-bit image data, where it is exact.
torch::Tensor resize_pil_bicubic(const torch::Tensor& image,
                                 int64_t out_h,
                                 int64_t out_w);

torch::Tensor center_crop(const torch::Tensor& image,
                          const std::pair<int32_t, int32_t>& crop_size);

torch::Tensor rescale(const torch::Tensor& image, double scale);

torch::Tensor normalize(const torch::Tensor& image,
                        const torch::Tensor& mean,
                        const torch::Tensor& std);

}  // namespace xllm::transforms
