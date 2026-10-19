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

#include "processors/transforms.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace xllm::transforms {

namespace {

// Fixed-point precision of Pillow's 8bpc resampler (Resample.c).
constexpr int kPrecBits = 32 - 8 - 2;  // 22

inline double bicubic_filter(double x) {
  constexpr double a = -0.5;
  if (x < 0.0) {
    x = -x;
  }
  if (x < 1.0) {
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
  }
  if (x < 2.0) {
    return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
  }
  return 0.0;
}

// Mirrors Pillow precompute_coeffs + normalize_coeffs_8bpc for BICUBIC.
struct PilCoeffs {
  int32_t ksize = 0;
  std::vector<int32_t> bounds;  // 2 * out_size
  std::vector<int32_t> k;       // out_size * ksize
};

PilCoeffs precompute_pil_coeffs(int32_t in_size, int32_t out_size) {
  PilCoeffs c;
  const double scale = static_cast<double>(in_size) / out_size;
  const double filterscale = std::max(1.0, scale);
  const double support = 2.0 * filterscale;
  const double ss = 1.0 / filterscale;
  c.ksize = static_cast<int32_t>(std::ceil(support)) * 2 + 1;
  c.bounds.assign(static_cast<size_t>(out_size) * 2, 0);
  c.k.assign(static_cast<size_t>(out_size) * c.ksize, 0);
  // Reused across output pixels; `xmax` varies, so `assign` keeps the
  // allocation instead of reallocating a fresh vector every iteration.
  std::vector<double> w;
  for (int32_t xx = 0; xx < out_size; ++xx) {
    const double center = (xx + 0.5) * scale;
    int32_t xmin = static_cast<int32_t>(center - support + 0.5);
    if (xmin < 0) {
      xmin = 0;
    }
    int32_t xmax = static_cast<int32_t>(center + support + 0.5);
    if (xmax > in_size) {
      xmax = in_size;
    }
    xmax -= xmin;
    c.bounds[xx * 2] = xmin;
    c.bounds[xx * 2 + 1] = xmax;
    w.assign(static_cast<size_t>(xmax), 0.0);
    double ww = 0.0;
    for (int32_t x = 0; x < xmax; ++x) {
      w[x] = bicubic_filter((x + xmin - center + 0.5) * ss);
      ww += w[x];
    }
    for (int32_t x = 0; x < xmax; ++x) {
      const double v = ww != 0.0 ? w[x] / ww : w[x];
      c.k[xx * c.ksize + x] =
          v < 0 ? static_cast<int32_t>(-0.5 + v * (1 << kPrecBits))
                : static_cast<int32_t>(0.5 + v * (1 << kPrecBits));
    }
  }
  return c;
}

inline uint8_t clip8(int32_t acc) {
  const int32_t idx = acc >> kPrecBits;
  if (idx < 0) {
    return 0;
  }
  if (idx > 255) {
    return 255;
  }
  return static_cast<uint8_t>(idx);
}

void pil_horizontal(const uint8_t* src,
                    uint8_t* dst,
                    int64_t rows,
                    int64_t in_w,
                    int64_t out_w,
                    const PilCoeffs& cf) {
  for (int64_t y = 0; y < rows; ++y) {
    const uint8_t* row = src + y * in_w;
    uint8_t* drow = dst + y * out_w;
    for (int64_t xx = 0; xx < out_w; ++xx) {
      const int32_t xmin = cf.bounds[xx * 2];
      const int32_t xmax = cf.bounds[xx * 2 + 1];
      const int32_t* k = &cf.k[xx * cf.ksize];
      int32_t acc = 1 << (kPrecBits - 1);
      for (int32_t x = 0; x < xmax; ++x) {
        acc += static_cast<int32_t>(row[x + xmin]) * k[x];
      }
      drow[xx] = clip8(acc);
    }
  }
}

void pil_vertical(const uint8_t* src,
                  uint8_t* dst,
                  int64_t out_h,
                  int64_t width,
                  int64_t in_stride,
                  const PilCoeffs& cf) {
  for (int64_t yy = 0; yy < out_h; ++yy) {
    const int32_t ymin = cf.bounds[yy * 2];
    const int32_t ymax = cf.bounds[yy * 2 + 1];
    const int32_t* k = &cf.k[yy * cf.ksize];
    uint8_t* drow = dst + yy * width;
    for (int64_t x = 0; x < width; ++x) {
      int32_t acc = 1 << (kPrecBits - 1);
      for (int32_t y = 0; y < ymax; ++y) {
        acc += static_cast<int32_t>(src[(ymin + y) * in_stride + x]) * k[y];
      }
      drow[x] = clip8(acc);
    }
  }
}

}  // namespace

torch::Tensor resize_pil_bicubic(const torch::Tensor& image,
                                 int64_t out_h,
                                 int64_t out_w) {
  CHECK_EQ(image.dim(), 3)
      << "resize_pil_bicubic expects a (C, H, W) tensor, got " << image.dim()
      << " dims";
  CHECK_GT(out_h, 0)
      << "resize_pil_bicubic requires a positive output height, got " << out_h;
  CHECK_GT(out_w, 0)
      << "resize_pil_bicubic requires a positive output width, got " << out_w;
  auto input = image.to(torch::kCPU);
  // Pillow's BICUBIC resampler operates on 8-bit samples, so this port does
  // too: quantize any non-uint8 input to [0, 255] to stay bit-exact with the
  // reference. The vision pipeline feeds 8-bit image data, where the cast is
  // exact; sub-8-bit float images are intentionally out of scope here.
  if (input.scalar_type() != torch::kUInt8) {
    input = input.clamp(0, 255).to(torch::kUInt8);
  }
  input = input.contiguous();
  const int64_t channels = input.size(0);
  const int64_t in_h = input.size(1);
  const int64_t in_w = input.size(2);
  auto out = torch::empty({channels, out_h, out_w}, input.options());
  if (out_h == in_h && out_w == in_w) {
    out.copy_(input);
    return out;
  }
  const uint8_t* src = input.data_ptr<uint8_t>();
  uint8_t* dst = out.data_ptr<uint8_t>();
  const bool need_horizontal = out_w != in_w;
  const bool need_vertical = out_h != in_h;
  if (need_horizontal) {
    const PilCoeffs ch = precompute_pil_coeffs(static_cast<int32_t>(in_w),
                                               static_cast<int32_t>(out_w));
    // Reuse one channel-sized scratch buffer. Allocating all channels up front
    // multiplies peak memory by `channels` for large source images, while the
    // horizontal pass can be completed one channel at a time.
    std::vector<uint8_t> temp(static_cast<size_t>(in_h) * out_w);
    if (need_vertical) {
      const PilCoeffs cv = precompute_pil_coeffs(static_cast<int32_t>(in_h),
                                                 static_cast<int32_t>(out_h));
      for (int64_t c = 0; c < channels; ++c) {
        pil_horizontal(
            src + c * in_h * in_w, temp.data(), in_h, in_w, out_w, ch);
        pil_vertical(
            temp.data(), dst + c * out_h * out_w, out_h, out_w, out_w, cv);
      }
    } else {
      for (int64_t c = 0; c < channels; ++c) {
        pil_horizontal(
            src + c * in_h * in_w, temp.data(), in_h, in_w, out_w, ch);
        std::memcpy(dst + c * in_h * out_w, temp.data(), temp.size());
      }
    }
  } else {
    const PilCoeffs cv = precompute_pil_coeffs(static_cast<int32_t>(in_h),
                                               static_cast<int32_t>(out_h));
    for (int64_t c = 0; c < channels; ++c) {
      pil_vertical(src + c * in_h * in_w,
                   dst + c * out_h * out_w,
                   out_h,
                   in_w,
                   in_w,
                   cv);
    }
  }
  return out;
}

torch::Tensor resize(const torch::Tensor& image,
                     const std::vector<int64_t>& size,
                     int32_t resample,
                     bool antialias) {
  if (image.dim() != 3 && image.dim() != 4) {
    LOG(FATAL) << "Input image must be a 3D or 4D tensor "
               << "((C x H x W) or (N x C x H x W)).";
  }
  auto options = torch::nn::functional::InterpolateFuncOptions()
                     .size(size)
                     .align_corners(false)
                     .antialias(antialias);
  switch (resample) {
    case 1:
      options.mode(torch::kNearest);
      break;
    case 2:
      options.mode(torch::kBilinear);
      break;
    case 3:
      options.mode(torch::kBicubic);
      break;
    default:
      LOG(FATAL) << "Invalid resample value. Must be one of 1, 2, or 3.";
  }
  auto input = image.dim() == 3 ? image.unsqueeze(0) : image;
  auto resized =
      torch::nn::functional::interpolate(input, options).clamp(0, 255);
  return image.dim() == 3 ? resized.squeeze(0) : resized;
}

torch::Tensor center_crop(const torch::Tensor& image,
                          const std::pair<int32_t, int32_t>& crop_size) {
  if (image.dim() != 3) {
    LOG(FATAL)
        << "Input image must be a 3-dimensional tensor in (C, H, W) format.";
  }

  int32_t crop_height = crop_size.first;
  int32_t crop_width = crop_size.second;
  int32_t orig_height = image.size(1);
  int32_t orig_width = image.size(2);

  int32_t top = (orig_height - crop_height) / 2;
  int32_t bottom = top + crop_height;
  int32_t left = (orig_width - crop_width) / 2;
  int32_t right = left + crop_width;

  if (top >= 0 && bottom <= orig_height && left >= 0 && right <= orig_width) {
    return image.index({torch::indexing::Slice(),
                        torch::indexing::Slice(top, bottom),
                        torch::indexing::Slice(left, right)});
  }

  int32_t new_height = std::max(crop_height, orig_height);
  int32_t new_width = std::max(crop_width, orig_width);
  auto padded_image =
      torch::zeros({image.size(0), new_height, new_width}, image.options());

  int32_t top_pad = (new_height - orig_height + 1) / 2;
  int32_t left_pad = (new_width - orig_width + 1) / 2;

  padded_image.index_put_(
      {torch::indexing::Slice(),
       torch::indexing::Slice(top_pad, top_pad + orig_height),
       torch::indexing::Slice(left_pad, left_pad + orig_width)},
      image);

  top = (new_height - crop_height) / 2;
  bottom = top + crop_height;
  left = (new_width - crop_width) / 2;
  right = left + crop_width;

  return padded_image.index({torch::indexing::Slice(),
                             torch::indexing::Slice(top, bottom),
                             torch::indexing::Slice(left, right)});
}

torch::Tensor rescale(const torch::Tensor& image, double scale) {
  return image * scale;
}

torch::Tensor normalize(const torch::Tensor& image,
                        const torch::Tensor& mean,
                        const torch::Tensor& std) {
  if (image.dim() != 3 && image.dim() != 4) {
    LOG(FATAL) << "Input image must be a 3D or 4D tensor "
               << "((C, H, W) or (N, C, H, W)).";
  }

  int32_t num_channels = image.dim() == 3 ? image.size(0) : image.size(1);
  if (mean.dim() != 1 || std.dim() != 1 || mean.size(0) != num_channels ||
      std.size(0) != num_channels) {
    LOG(FATAL) << "Mean and std tensors must have the same number "
               << "of elements as the number of channels in the "
               << "image.";
  }

  auto result = image;
  if (!image.is_floating_point()) {
    result = image.to(torch::kFloat32);
  }

  auto device = image.device();
  auto options = torch::dtype(torch::kFloat32).device(device);

  auto mean_shape = image.dim() == 3 ? std::vector<int64_t>{-1, 1, 1}
                                     : std::vector<int64_t>{1, -1, 1, 1};
  auto mean_tensor = mean.to(options).reshape(mean_shape);
  auto std_tensor = std.to(options).reshape(mean_shape);

  result = result.sub(mean_tensor);
  return result.div_(std_tensor);
}

}  // namespace xllm::transforms
