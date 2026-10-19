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

#include "processors/deepseek_v41_image_processor.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "core/common/constants.h"
#include "core/util/utils.h"
#include "processors/transforms.h"

namespace xllm {

namespace {

// An image block never shrinks below a 1x1 ViT grid, which is 6 LLM tokens.
constexpr int64_t kMinImageTokens = 6;
// Splice-time pads reserved by the prompt processor before the image block.
constexpr int64_t kPadSlots = kDeepseekV41CompressPadTo - 1;

// vllm mm_preprocess.py:grid_tokens (LLM grid + block length).
int64_t grid_tokens(int64_t n_vit_h,
                    int64_t n_vit_w,
                    int64_t downsample_ratio) {
  const int64_t n_llm_h = util::ceil_div(n_vit_h, downsample_ratio);
  const int64_t n_llm_w = util::ceil_div(n_vit_w, downsample_ratio);
  int64_t num_tokens = n_llm_h * (n_llm_w + 1) + 2;
  if (n_llm_h % 2 == 1) {
    num_tokens += n_llm_w + 1;
  }
  num_tokens += ((((n_llm_h + 1) / 2) * (n_llm_w + 1)) % 2) * 2;
  return num_tokens;
}

// vllm mm_preprocess.py:solve_resize_ratio.
void solve_resize_ratio(double height,
                        double width,
                        int64_t patch_size,
                        int64_t downsample_ratio,
                        int64_t max_n_token,
                        int64_t& best_height,
                        int64_t& best_width,
                        int64_t& num_tokens) {
  CHECK_GT(height, 0.0);
  CHECK_GT(width, 0.0);
  double ratio = height / width;
  // A shrunken token budget can make the radicand negative (`max_n_token <= 2`
  // against a near-square image); std::sqrt would yield NaN and the floor/cast
  // below would turn it into garbage canvas sizes. Clamp the radicand so the
  // budget degrades to the smallest valid canvas instead.
  const double radicand = (max_n_token - 2) / ratio + 0.25;
  double max_w_float = radicand > 0.0 ? std::sqrt(radicand) - 0.5 : 0.0;
  double max_h_float = max_w_float * ratio;
  if (max_w_float < 1.0) {
    int64_t max_w = 1;
    int64_t max_h = (max_n_token - 2) / (max_w + 1);
    if (max_h % 2 == 1) {
      max_h -= 1;
    }
    best_width = max_w * patch_size * downsample_ratio;
    best_height = max_h * patch_size * downsample_ratio;
  } else if (max_h_float < 2.0) {
    int64_t max_h = 2;
    int64_t max_w = ((max_n_token - 2) / max_h) - 1;
    best_width = max_w * patch_size * downsample_ratio;
    best_height = max_h * patch_size * downsample_ratio;
  } else {
    int64_t max_w = static_cast<int64_t>(std::floor(max_w_float));
    int64_t max_h = static_cast<int64_t>(std::floor(max_h_float));
    if (max_h % 2 == 1) {
      max_h -= 1;
    }
    double beta = std::min(max_w * patch_size * downsample_ratio / width,
                           max_h * patch_size * downsample_ratio / height);
    best_width = static_cast<int64_t>(std::floor(width * beta / patch_size)) *
                 patch_size;
    best_height = static_cast<int64_t>(std::floor(height * beta / patch_size)) *
                  patch_size;
  }
  // Never emit a sub-patch canvas: keep at least one patch per side so the
  // downstream pad/resize always sees positive dimensions.
  best_width = std::max<int64_t>(best_width, patch_size);
  best_height = std::max<int64_t>(best_height, patch_size);
  num_tokens = grid_tokens(
      best_height / patch_size, best_width / patch_size, downsample_ratio);
}

}  // namespace

DeepseekV41ImageProcessor::DeepseekV41ImageProcessor(const ModelArgs& args) {
  if (args.mm_image_patch_size() > 0) {
    patch_size_ = args.mm_image_patch_size();
  } else if (args.mm_patch_size() > 0) {
    patch_size_ = args.mm_patch_size();
  }
  if (args.mm_downsample_ratio() > 0) {
    downsample_ratio_ = args.mm_downsample_ratio();
  }
  if (args.mm_num_channels() > 0) {
    channels_ = args.mm_num_channels();
  }
  max_n_token_ = args.mm_image_max_tokens();
  // The smallest possible image block is 6 LLM tokens, and the prompt
  // processor may splice in up to kPadSlots pads before it. A smaller budget
  // cannot be satisfied by any grid, so reject it at construction instead of
  // letting the clamp emit an over-budget block.
  constexpr int64_t kMinMaxTokens = kMinImageTokens + kPadSlots;
  CHECK(max_n_token_ == 0 || max_n_token_ >= kMinMaxTokens)
      << "max_image_tokens must be 0 (disabled) or at least " << kMinMaxTokens
      << " (minimum image block plus pads), got " << max_n_token_;
  min_pixels_ = args.mm_image_min_pixels();
}

void DeepseekV41ImageProcessor::process_image(const torch::Tensor& image,
                                              torch::Tensor& pixel_values,
                                              torch::Tensor& grid) const {
  const int64_t p = patch_size_;
  const int64_t r = downsample_ratio_;
  const auto& img = image;  // [C, H, W], 8-bit in [0, 255]
  const int64_t channels = channels_;
  const int64_t src_h = img.size(1);
  const int64_t src_w = img.size(2);
  double width = static_cast<double>(src_w);
  double height = static_cast<double>(src_h);

  if (min_pixels_ > 0 && width * height > 0 &&
      width * height < static_cast<double>(min_pixels_)) {
    double scale =
        std::sqrt(static_cast<double>(min_pixels_) / (width * height));
    width *= scale;
    height *= scale;
  }

  int64_t best_width = static_cast<int64_t>(std::ceil(width / p)) * p;
  int64_t best_height = static_cast<int64_t>(std::ceil(height / p)) * p;

  // safe_resize: shrink until the block plus its reserved pads fits. Budgets
  // below kMinImageTokens cannot produce a smaller block, so the constructor
  // rejects them; the loop stops at the minimum to avoid pointless shrink calls
  // and the final check keeps the invariant explicit.
  const int64_t token_limit = max_n_token_ > 0 ? max_n_token_ - kPadSlots : 0;
  int64_t budget = token_limit;
  int64_t num_tokens = grid_tokens(best_height / p, best_width / p, r);
  while (budget >= kMinImageTokens && num_tokens > budget) {
    solve_resize_ratio(static_cast<double>(src_h),
                       static_cast<double>(src_w),
                       p,
                       r,
                       budget,
                       best_height,
                       best_width,
                       num_tokens);
    budget -= 1;
  }
  if (max_n_token_ > 0) {
    CHECK_LE(num_tokens, token_limit)
        << "DeepSeek-V4.1 image needs " << num_tokens
        << " tokens, but max_image_tokens leaves " << token_limit;
  }
  const int64_t n_vit_h = best_height / p;
  const int64_t n_vit_w = best_width / p;

  // Resize preserving aspect, then grey-pad to (best_height, best_width).
  double scale = std::min(best_width / width, best_height / height);
  int64_t fit_w =
      std::max<int64_t>(1, static_cast<int64_t>(std::llround(width * scale)));
  int64_t fit_h =
      std::max<int64_t>(1, static_cast<int64_t>(std::llround(height * scale)));
  // The image already carries 8-bit data in [0, 255]; resize_pil_bicubic
  // mirrors Pillow's 8bpc resampler and passes a uint8 input through without
  // quantization, so there is no float round-trip here.
  auto resized = transforms::resize_pil_bicubic(img, fit_h, fit_w);
  // Grey-pad canvas, explicitly 8-bit in [0, 255] like `resized` (which
  // resize_pil_bicubic always returns as uint8); pinning the dtype keeps the
  // uint8 block from being implicitly promoted if a caller ever passes a
  // non-uint8 `img`.
  auto canvas = torch::full(
      {channels, best_height, best_width},
      127.0,
      torch::TensorOptions().dtype(torch::kUInt8).device(img.device()));
  const int64_t top = (best_height - fit_h) / 2;
  const int64_t left = (best_width - fit_w) / 2;
  canvas.index_put_({torch::indexing::Slice(),
                     torch::indexing::Slice(top, top + fit_h),
                     torch::indexing::Slice(left, left + fit_w)},
                    resized);

  auto x = canvas / 255.0;
  x = (x - 0.5) / 0.5;
  x = x.to(torch::kBFloat16);
  pixel_values = x.reshape({channels, n_vit_h, p, n_vit_w, p})
                     .permute({1, 3, 0, 2, 4})
                     .reshape({n_vit_h * n_vit_w, channels, p, p})
                     .contiguous();
  grid = torch::tensor({n_vit_h, n_vit_w}, torch::dtype(torch::kInt64))
             .reshape({1, 2});
}

bool DeepseekV41ImageProcessor::process(
    const std::vector<torch::Tensor>& images,
    std::vector<MMDataItem>& output_items) const {
  output_items.clear();
  output_items.reserve(images.size());
  for (const auto& image : images) {
    if (image.dim() != 3 || image.size(0) != channels_) {
      const int64_t got_channels = image.dim() == 3 ? image.size(0) : -1;
      LOG(ERROR) << "DeepSeek-V4.1 image must be [C, H, W] with C=" << channels_
                 << ", got dim " << image.dim() << " and C=" << got_channels;
      return false;
    }
    if (image.size(1) <= 0 || image.size(2) <= 0) {
      LOG(ERROR) << "DeepSeek-V4.1 image must have positive height and width, "
                 << "got " << image.size(1) << "x" << image.size(2);
      return false;
    }
    torch::Tensor pixel_values;
    torch::Tensor grid;
    process_image(image, pixel_values, grid);
    output_items.emplace_back(
        MMType::IMAGE,
        MMDict{{"pixel_values", pixel_values}, {"image_grid_thw", grid}});
  }
  return true;
}

}  // namespace xllm
