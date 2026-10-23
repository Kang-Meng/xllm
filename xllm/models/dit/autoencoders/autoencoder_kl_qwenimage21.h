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

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/state_dict/utils.h"
#include "models/dit/utils/dit_parallel_mixin.h"

namespace xllm::qwen_image_21 {

// Optional VAE tiling and overlap blending.

// Dimensions are in sample/pixel space, not latent space. 512/384 matches the
// QwenImage21 reference's 16x spatial VAE rather than the old 8x VAE defaults.
struct VaeTilingConfig {
  int64_t tile_height = 512;
  int64_t tile_width = 512;
  int64_t stride_height = 384;
  int64_t stride_width = 384;
};

inline void validate_vae_tiling(const VaeTilingConfig& config,
                                int64_t spatial_scale) {
  CHECK_GT(spatial_scale, 0);
  CHECK_GT(config.tile_height, 0);
  CHECK_GT(config.tile_width, 0);
  CHECK_GT(config.stride_height, 0);
  CHECK_GT(config.stride_width, 0);
  CHECK_LT(config.stride_height, config.tile_height)
      << "VAE tiles require positive vertical overlap";
  CHECK_LT(config.stride_width, config.tile_width)
      << "VAE tiles require positive horizontal overlap";
  CHECK_EQ(config.tile_height % spatial_scale, 0);
  CHECK_EQ(config.tile_width % spatial_scale, 0);
  CHECK_EQ(config.stride_height % spatial_scale, 0);
  CHECK_EQ(config.stride_width % spatial_scale, 0);
}

// Same ramp and direction as the reference's blend_v/blend_h, vectorized.
// Mutates only the current tile; previous tiles may also contain earlier
// blends and must remain unchanged. Last partial tiles clamp the overlap.
inline void blend_vae_tile(const torch::Tensor& previous,
                           torch::Tensor& current,
                           int64_t axis,
                           int64_t extent) {
  CHECK(axis == 3 || axis == 4);
  extent = std::min({extent, previous.size(axis), current.size(axis)});
  if (extent == 0) {
    return;
  }
  CHECK_GT(extent, 0);
  std::vector<int64_t> shape(5, 1);
  shape[axis] = extent;
  auto ramp = (torch::arange(extent, current.options().dtype(torch::kFloat32)) /
               static_cast<double>(extent))
                  .reshape(shape);
  // Match reference scalar weights after their conversion to the tile dtype;
  // form both ramps in FP32, not 1 - already-rounded BF16 ramp.
  auto from_previous = (1.0 - ramp).to(current.scalar_type());
  auto from_current = ramp.to(current.scalar_type());
  auto tail = previous.slice(axis, previous.size(axis) - extent);
  auto head = current.slice(axis, 0, extent);
  head.copy_(tail * from_previous + head * from_current);
}

// Generic fixed-ratio tiled transform. input/output scales are coordinate
// units per shared grid cell: encode uses (spatial_scale,1), decode uses
// (1,spatial_scale). The callback runs an UNTILED component on each tile.
// Stitch directly into one output allocation and retain only a single tile
// row. No quadratic score buffer or all-tiles output list is retained here.
// Inference-only: the caller uses NoGradGuard, as for the enclosing pipeline.
template <typename Transform>
torch::Tensor tiled_vae_transform(const torch::Tensor& input,
                                  const VaeTilingConfig& config,
                                  int64_t input_scale,
                                  int64_t output_scale,
                                  Transform&& transform) {
  CHECK_EQ(input.dim(), 5);
  CHECK_EQ(input.size(0), 1);
  CHECK_EQ(input.size(2), 1);
  CHECK_GT(input.size(3), 0);
  CHECK_GT(input.size(4), 0);
  CHECK_GT(input_scale, 0);
  CHECK_GT(output_scale, 0);
  CHECK(input_scale == 1 || output_scale == 1);
  const int64_t spatial_scale = std::max(input_scale, output_scale);
  validate_vae_tiling(config, spatial_scale);
  CHECK_EQ(input.size(3) % input_scale, 0);
  CHECK_EQ(input.size(4) % input_scale, 0);
  CHECK_LE(input.size(3) / input_scale,
           std::numeric_limits<int64_t>::max() / output_scale);
  CHECK_LE(input.size(4) / input_scale,
           std::numeric_limits<int64_t>::max() / output_scale);
  const int64_t tile_height = config.tile_height / spatial_scale * input_scale;
  const int64_t tile_width = config.tile_width / spatial_scale * input_scale;
  const int64_t stride_height =
      config.stride_height / spatial_scale * input_scale;
  const int64_t stride_width =
      config.stride_width / spatial_scale * input_scale;
  const int64_t output_height = input.size(3) / input_scale * output_scale;
  const int64_t output_width = input.size(4) / input_scale * output_scale;
  const int64_t blend_height = (config.tile_height - config.stride_height) /
                               spatial_scale * output_scale;
  const int64_t blend_width =
      (config.tile_width - config.stride_width) / spatial_scale * output_scale;
  const int64_t output_stride_height =
      stride_height / input_scale * output_scale;
  const int64_t output_stride_width = stride_width / input_scale * output_scale;
  const int64_t columns = (input.size(4) - 1) / stride_width + 1;
  std::vector<torch::Tensor> previous_row(static_cast<size_t>(columns));
  torch::Tensor output;
  for (int64_t y = 0; y < input.size(3); y += stride_height) {
    const int64_t bottom = y + std::min(tile_height, input.size(3) - y);
    for (int64_t x = 0, column = 0; x < input.size(4);
         x += stride_width, ++column) {
      const int64_t right = x + std::min(tile_width, input.size(4) - x);
      auto tile =
          transform(input.slice(3, y, bottom).slice(4, x, right).contiguous());
      CHECK_EQ(tile.dim(), 5);
      CHECK_EQ(tile.size(0), 1);
      CHECK_EQ(tile.size(2), 1);
      CHECK_EQ(tile.size(3), (bottom - y) / input_scale * output_scale);
      CHECK_EQ(tile.size(4), (right - x) / input_scale * output_scale);
      CHECK(tile.device() == input.device());
      if (!output.defined()) {
        output = torch::empty({1, tile.size(1), 1, output_height, output_width},
                              tile.options());
      }
      CHECK_EQ(tile.size(1), output.size(1));
      CHECK(tile.scalar_type() == output.scalar_type());
      if (y != 0) {
        blend_vae_tile(previous_row[column], tile, 3, blend_height);
      }
      // previous_row[column-1] has already been replaced by this row's tile.
      if (column != 0) {
        blend_vae_tile(previous_row[column - 1], tile, 4, blend_width);
      }
      const int64_t output_y = y / input_scale * output_scale;
      const int64_t output_x = x / input_scale * output_scale;
      const int64_t copy_height =
          std::min(output_stride_height, output_height - output_y);
      const int64_t copy_width =
          std::min(output_stride_width, output_width - output_x);
      output.slice(3, output_y, output_y + copy_height)
          .slice(4, output_x, output_x + copy_width)
          .copy_(tile.slice(3, 0, copy_height).slice(4, 0, copy_width));
      previous_row[column] = std::move(tile);
    }
  }
  return output;
}

// Image-only convolution with spatial halos.

class Conv2dImpl final : public torch::nn::Module,
                         public dit::VaeParallelMixin {
 public:
  Conv2dImpl(const ModelContext& context,
             int64_t input,
             int64_t output,
             int64_t kernel,
             int64_t stride,
             int64_t padding,
             const torch::TensorOptions& options)
      : dit::VaeParallelMixin(context), stride_(stride), padding_(padding) {
    weight_ = register_parameter(
        "weight", torch::empty({output, input, kernel, kernel}, options));
    bias_ = register_parameter("bias", torch::empty({output}, options));
  }
  torch::Tensor forward(const torch::Tensor& input) {
    CHECK(input.dim() == 4 || (input.dim() == 5 && input.size(2) == 1))
        << "QwenImage21 convolutions support static images only";
    auto x = input.dim() == 5 ? input.squeeze(2) : input;
    if (vae_parallel_enabled() && stride_ == 1 && padding_ == 1) {
      CHECK_EQ(weight_.size(2), 3);
      CHECK_EQ(weight_.size(3), 3);
      x = vae_parallel_exchange(x, /*pad=*/true);
      x = torch::nn::functional::pad(
          x, torch::nn::functional::PadFuncOptions({0, 0, 1, 1}));
      auto result = torch::nn::functional::conv2d(
          x,
          weight_,
          torch::nn::functional::Conv2dFuncOptions()
              .bias(bias_)
              .stride(1)
              .padding(0));
      return input.dim() == 5 ? result.unsqueeze(2) : result;
    }
    auto result =
        torch::nn::functional::conv2d(x,
                                      weight_,
                                      torch::nn::functional::Conv2dFuncOptions()
                                          .bias(bias_)
                                          .stride(stride_)
                                          .padding(padding_));
    return input.dim() == 5 ? result.unsqueeze(2) : result;
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "weight", weight_, weight_is_loaded_);
    weight::load_weight(state_dict, "bias", bias_, bias_is_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(weight_is_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(bias_is_loaded_) << "bias is not loaded for " << prefix + "bias";
  }

 private:
  int64_t stride_;
  int64_t padding_;
  torch::Tensor weight_;
  bool weight_is_loaded_ = false;
  torch::Tensor bias_;
  bool bias_is_loaded_ = false;
};
TORCH_MODULE(Conv2d);

// An explicit FP32, chunked math path for CPU reference tests and the VAE's
// large single attention head (the real decoder has head_dim=1152). It does not
// depend on fused-attention head-size limits or materialize all L*L scores.
inline torch::Tensor spatial_attention(const torch::Tensor& q,
                                       const torch::Tensor& k,
                                       const torch::Tensor& v,
                                       int64_t chunk_size = 128) {
  CHECK_GT(chunk_size, 0);
  const double scale = 1.0 / std::sqrt(static_cast<double>(q.size(-1)));
  auto key = k.to(torch::kFloat32).transpose(-1, -2);
  auto value = v.to(torch::kFloat32);
  std::vector<torch::Tensor> chunks;
  chunks.reserve((q.size(-2) + chunk_size - 1) / chunk_size);
  for (int64_t start = 0; start < q.size(-2); start += chunk_size) {
    auto query = q.slice(-2, start, std::min(start + chunk_size, q.size(-2)))
                     .to(torch::kFloat32);
    auto scores = torch::matmul(query, key) * scale;
    chunks.emplace_back(
        torch::matmul(torch::softmax(scores, -1), value).to(q.scalar_type()));
  }
  return torch::cat(chunks, -2);
}

}  // namespace xllm::qwen_image_21

namespace xllm {

struct QwenImage21VaeConfig {
  int64_t in_channels = 4;
  int64_t out_channels = 4;
  int64_t base_dim = 96;
  int64_t decoder_base_dim = 144;
  int64_t z_dim = 64;
  int64_t num_res_blocks = 2;
  bool is_residual = true;
  // null/1 both mean no spatial patchification, the only supported variant.
  int64_t patch_size = 1;
  std::vector<int64_t> dim_mult = {1, 2, 4, 8, 8};
  std::vector<bool> temperal_downsample = {false, true, true, true};
  std::vector<double> latents_mean;
  std::vector<double> latents_std;
};

namespace qwen_image_21 {

class ChannelNormImpl final : public torch::nn::Module {
 public:
  ChannelNormImpl(int64_t dim, bool images, const torch::TensorOptions& options)
      : scale_(std::sqrt(static_cast<double>(dim))) {
    const std::vector<int64_t> shape = images
                                           ? std::vector<int64_t>{dim, 1, 1}
                                           : std::vector<int64_t>{dim, 1, 1, 1};
    gamma_ = register_parameter("gamma", torch::empty(shape, options));
  }
  torch::Tensor forward(const torch::Tensor& input) {
    auto x = input.to(torch::kFloat32);
    // F.normalize clamps the L2 norm, not mean-square + epsilon.
    auto denominator =
        x.square().sum(1, /*keepdim=*/true).sqrt().clamp_min(1e-12);
    return (x / denominator).to(input.scalar_type()) * scale_ * gamma_;
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "gamma", gamma_, gamma_is_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(gamma_is_loaded_) << "gamma is not loaded for " << prefix + "gamma";
  }

 private:
  double scale_;
  torch::Tensor gamma_;
  bool gamma_is_loaded_ = false;
};
TORCH_MODULE(ChannelNorm);

class VaeResidualImpl final : public torch::nn::Module {
 public:
  VaeResidualImpl(const ModelContext& context,
                  int64_t input,
                  int64_t output,
                  const torch::TensorOptions& options) {
    norm1_ = register_module("norm1", ChannelNorm(input, false, options));
    conv1_ = register_module("conv1",
                             Conv2d(context, input, output, 3, 1, 1, options));
    norm2_ = register_module("norm2", ChannelNorm(output, false, options));
    conv2_ = register_module("conv2",
                             Conv2d(context, output, output, 3, 1, 1, options));
    if (input != output) {
      shortcut_ = register_module(
          "conv_shortcut", Conv2d(context, input, output, 1, 1, 0, options));
    }
  }
  torch::Tensor forward(const torch::Tensor& input) {
    auto residual = shortcut_ ? shortcut_->forward(input) : input;
    auto hidden = conv1_->forward(torch::silu(norm1_->forward(input)));
    return conv2_->forward(torch::silu(norm2_->forward(hidden))) + residual;
  }

  void load_state_dict(const StateDict& state_dict) {
    norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
    conv1_->load_state_dict(state_dict.get_dict_with_prefix("conv1."));
    norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));
    conv2_->load_state_dict(state_dict.get_dict_with_prefix("conv2."));
    if (shortcut_) {
      shortcut_->load_state_dict(
          state_dict.get_dict_with_prefix("conv_shortcut."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    norm1_->verify_loaded_weights(prefix + "norm1.");
    conv1_->verify_loaded_weights(prefix + "conv1.");
    norm2_->verify_loaded_weights(prefix + "norm2.");
    conv2_->verify_loaded_weights(prefix + "conv2.");
    if (shortcut_) {
      shortcut_->verify_loaded_weights(prefix + "conv_shortcut.");
    }
  }

 private:
  ChannelNorm norm1_{nullptr}, norm2_{nullptr};
  Conv2d conv1_{nullptr}, conv2_{nullptr}, shortcut_{nullptr};
};
TORCH_MODULE(VaeResidual);

class VaeAttentionImpl final : public torch::nn::Module,
                               public dit::VaeParallelMixin {
 public:
  VaeAttentionImpl(const ModelContext& context,
                   int64_t dim,
                   const torch::TensorOptions& options)
      : dit::VaeParallelMixin(context) {
    norm_ = register_module("norm", ChannelNorm(dim, true, options));
    qkv_ = register_module("to_qkv",
                           Conv2d(context, dim, dim * 3, 1, 1, 0, options));
    output_ =
        register_module("proj", Conv2d(context, dim, dim, 1, 1, 0, options));
  }
  torch::Tensor forward(const torch::Tensor& input) {
    CHECK_EQ(input.size(2), 1);
    auto x = input.squeeze(2);
    auto spatial_qkv = qkv_->forward(norm_->forward(x)).chunk(3, 1);
    auto key = vae_parallel_merge(spatial_qkv[1]);
    auto value = vae_parallel_merge(spatial_qkv[2]);
    // Gather along spatial width BEFORE flattening, preserving row-major H/W
    // order. Queries remain local but attend to all global pixels.
    auto query = spatial_qkv[0].flatten(2).transpose(1, 2).unsqueeze(1);
    key = key.flatten(2).transpose(1, 2).unsqueeze(1);
    value = value.flatten(2).transpose(1, 2).unsqueeze(1);
    auto hidden = spatial_attention(query, key, value);
    hidden = hidden.squeeze(1).transpose(1, 2).reshape(x.sizes());
    return output_->forward(hidden).unsqueeze(2) + input;
  }

  void load_state_dict(const StateDict& state_dict) {
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
    qkv_->load_state_dict(state_dict.get_dict_with_prefix("to_qkv."));
    output_->load_state_dict(state_dict.get_dict_with_prefix("proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    norm_->verify_loaded_weights(prefix + "norm.");
    qkv_->verify_loaded_weights(prefix + "to_qkv.");
    output_->verify_loaded_weights(prefix + "proj.");
  }

 private:
  ChannelNorm norm_{nullptr};
  Conv2d qkv_{nullptr}, output_{nullptr};
};
TORCH_MODULE(VaeAttention);

class VaeMidBlockImpl final : public torch::nn::Module {
 public:
  VaeMidBlockImpl(const ModelContext& context,
                  int64_t dim,
                  const torch::TensorOptions& options) {
    resnets_ = register_module(
        "resnets",
        torch::nn::ModuleList(VaeResidual(context, dim, dim, options),
                              VaeResidual(context, dim, dim, options)));
    attentions_ = register_module(
        "attentions",
        torch::nn::ModuleList(VaeAttention(context, dim, options)));
  }
  torch::Tensor forward(const torch::Tensor& input) {
    auto x = resnets_->at<VaeResidualImpl>(0).forward(input);
    x = attentions_->at<VaeAttentionImpl>(0).forward(x);
    return resnets_->at<VaeResidualImpl>(1).forward(x);
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnets_->size(); ++i) {
      resnets_->at<VaeResidualImpl>(i).load_state_dict(
          state_dict.get_dict_with_prefix("resnets." + std::to_string(i) +
                                          "."));
    }
    for (size_t i = 0; i < attentions_->size(); ++i) {
      attentions_->at<VaeAttentionImpl>(i).load_state_dict(
          state_dict.get_dict_with_prefix("attentions." + std::to_string(i) +
                                          "."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    for (size_t i = 0; i < resnets_->size(); ++i) {
      resnets_->at<VaeResidualImpl>(i).verify_loaded_weights(
          prefix + "resnets." + std::to_string(i) + ".");
    }
    for (size_t i = 0; i < attentions_->size(); ++i) {
      attentions_->at<VaeAttentionImpl>(i).verify_loaded_weights(
          prefix + "attentions." + std::to_string(i) + ".");
    }
  }

 private:
  torch::nn::ModuleList resnets_{nullptr}, attentions_{nullptr};
};
TORCH_MODULE(VaeMidBlock);

inline torch::Tensor average_downsample(const torch::Tensor& input,
                                        int64_t output_channels,
                                        int64_t factor_t,
                                        int64_t factor_s) {
  CHECK_EQ(input.dim(), 5);
  CHECK_GT(factor_t, 0);
  CHECK_GT(factor_s, 0);
  CHECK_GT(output_channels, 0);
  CHECK_EQ(input.size(3) % factor_s, 0);
  CHECK_EQ(input.size(4) % factor_s, 0);
  const int64_t factor = factor_t * factor_s * factor_s;
  CHECK_EQ(input.size(1) * factor % output_channels, 0);
  const int64_t group = input.size(1) * factor / output_channels;
  const int64_t pad_t = (factor_t - input.size(2) % factor_t) % factor_t;
  auto x = torch::nn::functional::pad(
      input, torch::nn::functional::PadFuncOptions({0, 0, 0, 0, pad_t, 0}));
  const int64_t batch = x.size(0), channels = x.size(1);
  const int64_t frames = x.size(2) / factor_t;
  const int64_t height = x.size(3) / factor_s, width = x.size(4) / factor_s;
  x = x.reshape({batch,
                 channels,
                 frames,
                 factor_t,
                 height,
                 factor_s,
                 width,
                 factor_s})
          .permute({0, 1, 3, 5, 7, 2, 4, 6})
          .contiguous();
  return x.reshape({batch, output_channels, group, frames, height, width})
      .mean(2);
}

inline torch::Tensor duplicate_upsample(const torch::Tensor& input,
                                        int64_t output_channels,
                                        int64_t factor_t,
                                        int64_t factor_s) {
  CHECK_EQ(input.dim(), 5);
  CHECK_EQ(input.size(2), 1)
      << "Only the first static-image chunk is supported";
  const int64_t factor = factor_t * factor_s * factor_s;
  CHECK_EQ(output_channels * factor % input.size(1), 0);
  const int64_t repeats = output_channels * factor / input.size(1);
  auto x = input.repeat_interleave(repeats, 1)
               .reshape({input.size(0),
                         output_channels,
                         factor_t,
                         factor_s,
                         factor_s,
                         input.size(2),
                         input.size(3),
                         input.size(4)})
               .permute({0, 1, 5, 2, 6, 3, 7, 4})
               .contiguous()
               .reshape({input.size(0),
                         output_channels,
                         input.size(2) * factor_t,
                         input.size(3) * factor_s,
                         input.size(4) * factor_s});
  return x.slice(2, factor_t - 1);  // reference first_chunk=True
}

class VaeResampleImpl final : public torch::nn::Module,
                              public dit::VaeParallelMixin {
 public:
  VaeResampleImpl(const ModelContext& context,
                  int64_t dim,
                  bool up,
                  bool temporal,
                  const torch::TensorOptions& options)
      : dit::VaeParallelMixin(context), up_(up) {
    // Keep checkpoint's resample.1 naming. Spatial operations at index 0 have
    // no weights and are performed explicitly in forward.
    resample_ = register_module(
        "resample",
        torch::nn::Sequential(
            torch::nn::Identity(),
            Conv2d(context, dim, dim, 3, up ? 1 : 2, up ? 1 : 0, options)));
    if (temporal) {
      time_conv_ = register_module(
          "time_conv",
          Conv2d(context, dim, up ? 2 * dim : dim, 1, 1, 0, options));
    }
  }
  torch::Tensor forward(const torch::Tensor& input) {
    CHECK_EQ(input.size(2), 1);
    auto x = input.squeeze(2);
    if (up_) {
      // Integer 2x repetition is exactly nearest-exact interpolation.
      x = x.repeat_interleave(2, 2).repeat_interleave(2, 3);
    } else {
      if (vae_parallel_enabled()) {
        CHECK_EQ(x.size(-1) % 2, 0)
            << "VAE width shards must preserve downsample stride alignment";
        // Stride-2 convolution starts on the local grid and needs only the
        // right neighbour. Exchange pads both global edges; drop the left halo.
        x = vae_parallel_exchange(x, /*pad=*/true).slice(/*dim=*/-1, 1);
        x = torch::nn::functional::pad(
            x, torch::nn::functional::PadFuncOptions({0, 0, 0, 1}));
      } else {
        x = torch::nn::functional::pad(
            x, torch::nn::functional::PadFuncOptions({0, 1, 0, 1}));
      }
    }
    // In the reference, a fresh feature cache bypasses time_conv for the
    // first chunk. Load those weights for structural parity, but do not run
    // them for an F=1 image. No temporal state persists across requests.
    return resample_->at<Conv2dImpl>(1).forward(x).unsqueeze(2);
  }

  void load_state_dict(const StateDict& state_dict) {
    resample_->at<Conv2dImpl>(1).load_state_dict(
        state_dict.get_dict_with_prefix("resample.1."));
    if (time_conv_) {
      time_conv_->load_state_dict(
          state_dict.get_dict_with_prefix("time_conv."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    resample_->at<Conv2dImpl>(1).verify_loaded_weights(prefix + "resample.1.");
    if (time_conv_) {
      time_conv_->verify_loaded_weights(prefix + "time_conv.");
    }
  }

 private:
  bool up_;
  torch::nn::Sequential resample_{nullptr};
  Conv2d time_conv_{nullptr};
};
TORCH_MODULE(VaeResample);

class VaeDownBlockImpl final : public torch::nn::Module {
 public:
  VaeDownBlockImpl(const ModelContext& context,
                   int64_t input,
                   int64_t output,
                   int64_t blocks,
                   bool down,
                   bool temporal,
                   const torch::TensorOptions& options)
      : output_(output), factor_t_(temporal ? 2 : 1), factor_s_(down ? 2 : 1) {
    resnets_ = register_module("resnets", torch::nn::ModuleList());
    for (int64_t i = 0; i < blocks; ++i) {
      resnets_->push_back(
          VaeResidual(context, i == 0 ? input : output, output, options));
    }
    if (down) {
      downsample_ = register_module(
          "downsampler",
          VaeResample(context, output, false, temporal, options));
    }
  }
  torch::Tensor forward(const torch::Tensor& input) {
    auto x = input;
    for (const auto& block : *resnets_) {
      x = block->as<VaeResidualImpl>()->forward(x);
    }
    if (downsample_) {
      x = downsample_->forward(x);
    }
    return x + average_downsample(input, output_, factor_t_, factor_s_);
  }

  void load_state_dict(const StateDict& state_dict) {
    if (downsample_) {
      downsample_->load_state_dict(
          state_dict.get_dict_with_prefix("downsampler."));
    }
    for (size_t i = 0; i < resnets_->size(); ++i) {
      resnets_->at<VaeResidualImpl>(i).load_state_dict(
          state_dict.get_dict_with_prefix("resnets." + std::to_string(i) +
                                          "."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    if (downsample_) {
      downsample_->verify_loaded_weights(prefix + "downsampler.");
    }
    for (size_t i = 0; i < resnets_->size(); ++i) {
      resnets_->at<VaeResidualImpl>(i).verify_loaded_weights(
          prefix + "resnets." + std::to_string(i) + ".");
    }
  }

 private:
  int64_t output_, factor_t_, factor_s_;
  torch::nn::ModuleList resnets_{nullptr};
  VaeResample downsample_{nullptr};
};
TORCH_MODULE(VaeDownBlock);

class VaeUpBlockImpl final : public torch::nn::Module {
 public:
  VaeUpBlockImpl(const ModelContext& context,
                 int64_t input,
                 int64_t output,
                 int64_t blocks,
                 bool up,
                 bool temporal,
                 const torch::TensorOptions& options)
      : output_(output), factor_t_(temporal ? 2 : 1) {
    resnets_ = register_module("resnets", torch::nn::ModuleList());
    for (int64_t i = 0; i < blocks + 1; ++i) {
      resnets_->push_back(
          VaeResidual(context, i == 0 ? input : output, output, options));
    }
    if (up) {
      upsample_ = register_module(
          "upsampler", VaeResample(context, output, true, temporal, options));
    }
  }
  torch::Tensor forward(const torch::Tensor& input) {
    auto x = input;
    for (const auto& block : *resnets_) {
      x = block->as<VaeResidualImpl>()->forward(x);
    }
    if (upsample_) {
      x = upsample_->forward(x) +
          duplicate_upsample(input, output_, factor_t_, 2);
    }
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    if (upsample_) {
      upsample_->load_state_dict(state_dict.get_dict_with_prefix("upsampler."));
    }
    for (size_t i = 0; i < resnets_->size(); ++i) {
      resnets_->at<VaeResidualImpl>(i).load_state_dict(
          state_dict.get_dict_with_prefix("resnets." + std::to_string(i) +
                                          "."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    if (upsample_) {
      upsample_->verify_loaded_weights(prefix + "upsampler.");
    }
    for (size_t i = 0; i < resnets_->size(); ++i) {
      resnets_->at<VaeResidualImpl>(i).verify_loaded_weights(
          prefix + "resnets." + std::to_string(i) + ".");
    }
  }

 private:
  int64_t output_, factor_t_;
  torch::nn::ModuleList resnets_{nullptr};
  VaeResample upsample_{nullptr};
};
TORCH_MODULE(VaeUpBlock);

class VaeEncoderImpl final : public torch::nn::Module {
 public:
  VaeEncoderImpl(const ModelContext& context,
                 const QwenImage21VaeConfig& config,
                 const torch::TensorOptions& options) {
    input_ = register_module(
        "conv_in",
        Conv2d(context, config.in_channels, config.base_dim, 3, 1, 1, options));
    blocks_ = register_module("down_blocks", torch::nn::ModuleList());
    int64_t input = config.base_dim;
    for (int64_t i = 0; i < static_cast<int64_t>(config.dim_mult.size()); ++i) {
      const int64_t output = config.base_dim * config.dim_mult[i];
      const bool down = i + 1 < static_cast<int64_t>(config.dim_mult.size());
      blocks_->push_back(VaeDownBlock(context,
                                      input,
                                      output,
                                      config.num_res_blocks,
                                      down,
                                      down && config.temperal_downsample[i],
                                      options));
      input = output;
    }
    middle_ =
        register_module("mid_block", VaeMidBlock(context, input, options));
    norm_ = register_module("norm_out", ChannelNorm(input, false, options));
    output_ = register_module(
        "conv_out", Conv2d(context, input, 2 * config.z_dim, 3, 1, 1, options));
  }
  torch::Tensor forward(const torch::Tensor& input) {
    auto x = input_->forward(input);
    for (const auto& block : *blocks_) {
      x = block->as<VaeDownBlockImpl>()->forward(x);
    }
    return output_->forward(torch::silu(norm_->forward(middle_->forward(x))));
  }

  void load_state_dict(const StateDict& state_dict) {
    input_->load_state_dict(state_dict.get_dict_with_prefix("conv_in."));
    middle_->load_state_dict(state_dict.get_dict_with_prefix("mid_block."));
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm_out."));
    output_->load_state_dict(state_dict.get_dict_with_prefix("conv_out."));
    for (size_t i = 0; i < blocks_->size(); ++i) {
      blocks_->at<VaeDownBlockImpl>(i).load_state_dict(
          state_dict.get_dict_with_prefix("down_blocks." + std::to_string(i) +
                                          "."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    input_->verify_loaded_weights(prefix + "conv_in.");
    middle_->verify_loaded_weights(prefix + "mid_block.");
    norm_->verify_loaded_weights(prefix + "norm_out.");
    output_->verify_loaded_weights(prefix + "conv_out.");
    for (size_t i = 0; i < blocks_->size(); ++i) {
      blocks_->at<VaeDownBlockImpl>(i).verify_loaded_weights(
          prefix + "down_blocks." + std::to_string(i) + ".");
    }
  }

 private:
  Conv2d input_{nullptr}, output_{nullptr};
  ChannelNorm norm_{nullptr};
  VaeMidBlock middle_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
};
TORCH_MODULE(VaeEncoder);

class VaeDecoderImpl final : public torch::nn::Module {
 public:
  VaeDecoderImpl(const ModelContext& context,
                 const QwenImage21VaeConfig& config,
                 const torch::TensorOptions& options) {
    const std::vector<int64_t> mult(config.dim_mult.rbegin(),
                                    config.dim_mult.rend());
    const std::vector<bool> temporal(config.temperal_downsample.rbegin(),
                                     config.temperal_downsample.rend());
    int64_t input = config.decoder_base_dim * mult[0];
    input_ = register_module(
        "conv_in", Conv2d(context, config.z_dim, input, 3, 1, 1, options));
    middle_ =
        register_module("mid_block", VaeMidBlock(context, input, options));
    blocks_ = register_module("up_blocks", torch::nn::ModuleList());
    for (int64_t i = 0; i < static_cast<int64_t>(mult.size()); ++i) {
      const int64_t output = config.decoder_base_dim * mult[i];
      const bool up = i + 1 < static_cast<int64_t>(mult.size());
      blocks_->push_back(VaeUpBlock(context,
                                    input,
                                    output,
                                    config.num_res_blocks,
                                    up,
                                    up && temporal[i],
                                    options));
      input = output;
    }
    norm_ = register_module("norm_out", ChannelNorm(input, false, options));
    output_ = register_module(
        "conv_out",
        Conv2d(context, input, config.out_channels, 3, 1, 1, options));
  }
  torch::Tensor forward(const torch::Tensor& input) {
    auto x = middle_->forward(input_->forward(input));
    for (const auto& block : *blocks_) {
      x = block->as<VaeUpBlockImpl>()->forward(x);
    }
    return output_->forward(torch::silu(norm_->forward(x)));
  }

  void load_state_dict(const StateDict& state_dict) {
    input_->load_state_dict(state_dict.get_dict_with_prefix("conv_in."));
    middle_->load_state_dict(state_dict.get_dict_with_prefix("mid_block."));
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm_out."));
    output_->load_state_dict(state_dict.get_dict_with_prefix("conv_out."));
    for (size_t i = 0; i < blocks_->size(); ++i) {
      blocks_->at<VaeUpBlockImpl>(i).load_state_dict(
          state_dict.get_dict_with_prefix("up_blocks." + std::to_string(i) +
                                          "."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    input_->verify_loaded_weights(prefix + "conv_in.");
    middle_->verify_loaded_weights(prefix + "mid_block.");
    norm_->verify_loaded_weights(prefix + "norm_out.");
    output_->verify_loaded_weights(prefix + "conv_out.");
    for (size_t i = 0; i < blocks_->size(); ++i) {
      blocks_->at<VaeUpBlockImpl>(i).verify_loaded_weights(
          prefix + "up_blocks." + std::to_string(i) + ".");
    }
  }

 private:
  Conv2d input_{nullptr}, output_{nullptr};
  ChannelNorm norm_{nullptr};
  VaeMidBlock middle_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
};
TORCH_MODULE(VaeDecoder);

}  // namespace qwen_image_21

class AutoencoderKLQwenImage21Impl final : public torch::nn::Module,
                                           public dit::VaeParallelMixin {
 public:
  explicit AutoencoderKLQwenImage21Impl(
      const ModelContext& context,
      QwenImage21VaeConfig config,
      const torch::TensorOptions& options =
          torch::TensorOptions().dtype(torch::kFloat32))
      : dit::VaeParallelMixin(context), config_(std::move(config)) {
    CHECK(config_.is_residual);
    CHECK_EQ(config_.patch_size, 1);
    CHECK_EQ(config_.in_channels, 4);
    CHECK_EQ(config_.out_channels, 4);
    CHECK_GT(config_.base_dim, 0);
    CHECK_GT(config_.decoder_base_dim, 0);
    CHECK_GT(config_.z_dim, 0);
    CHECK_GT(config_.num_res_blocks, 0);
    CHECK_EQ(config_.dim_mult.size(), config_.temperal_downsample.size() + 1);
    CHECK_GE(config_.dim_mult.size(), 2);
    CHECK_LE(config_.temperal_downsample.size(), 8);
    for (int64_t multiplier : config_.dim_mult) {
      CHECK_GT(multiplier, 0);
    }
    spatial_scale_ = int64_t{1} << config_.temperal_downsample.size();
    encoder_ = register_module(
        "encoder", qwen_image_21::VaeEncoder(context, config_, options));
    quant_ = register_module(
        "quant_conv",
        qwen_image_21::Conv2d(
            context, 2 * config_.z_dim, 2 * config_.z_dim, 1, 1, 0, options));
    post_quant_ = register_module(
        "post_quant_conv",
        qwen_image_21::Conv2d(
            context, config_.z_dim, config_.z_dim, 1, 1, 0, options));
    decoder_ = register_module(
        "decoder", qwen_image_21::VaeDecoder(context, config_, options));
  }
  // Configure only at model setup, not concurrently with forward calls.
  void enable_tiling(qwen_image_21::VaeTilingConfig config = {}) {
    CHECK(!vae_parallel_enabled())
        << "VAE width parallelism and approximate tiling cannot be combined";
    qwen_image_21::validate_vae_tiling(config, spatial_scale_);
    tiling_config_ = config;
    tiling_enabled_ = true;
  }
  void disable_tiling() { tiling_enabled_ = false; }
  bool tiling_enabled() const { return tiling_enabled_; }
  const qwen_image_21::VaeTilingConfig& tiling_config() const {
    return tiling_config_;
  }

  // Returns raw posterior moments. Normalization is intentionally a separate
  // operation so it cannot be applied twice by the embedding-only pipeline.
  torch::Tensor encode_moments(const torch::Tensor& image) {
    check_input(image, config_.in_channels);
    CHECK_EQ(image.size(3) % spatial_scale_, 0);
    CHECK_EQ(image.size(4) % spatial_scale_, 0);
    if (tiling_enabled_ && (image.size(3) > tiling_config_.tile_height ||
                            image.size(4) > tiling_config_.tile_width)) {
      return qwen_image_21::tiled_vae_transform(
          image,
          tiling_config_,
          spatial_scale_,
          1,
          [this](const torch::Tensor& tile) {
            return encode_moments_untiled(tile);
          });
    }
    auto local = vae_parallel_split(image, spatial_scale_);
    return vae_parallel_merge(encode_moments_untiled(local));
  }
  torch::Tensor encode_mode(const torch::Tensor& image) {
    return encode_moments(image).chunk(2, 1)[0];
  }
  torch::Tensor decode(const torch::Tensor& latent) {
    check_input(latent, config_.z_dim);
    if (tiling_enabled_ &&
        (latent.size(3) > tiling_config_.tile_height / spatial_scale_ ||
         latent.size(4) > tiling_config_.tile_width / spatial_scale_)) {
      // Blend raw decoder outputs, then clamp once, matching reference tiling.
      return qwen_image_21::tiled_vae_transform(
                 latent,
                 tiling_config_,
                 1,
                 spatial_scale_,
                 [this](const torch::Tensor& tile) {
                   return decode_untiled(tile);
                 })
          .clamp(-1.0, 1.0);
    }
    auto local = vae_parallel_split(latent, /*alignment=*/1);
    return vae_parallel_merge(decode_untiled(local)).clamp(-1.0, 1.0);
  }
  torch::Tensor normalize_latents(const torch::Tensor& latent) const {
    auto [mean, stddev] = latent_statistics(latent);
    return (latent - mean) / stddev;
  }
  torch::Tensor denormalize_latents(const torch::Tensor& latent) const {
    auto [mean, stddev] = latent_statistics(latent);
    return latent * stddev + mean;
  }
  int64_t spatial_scale() const { return spatial_scale_; }
  const QwenImage21VaeConfig& config() const { return config_; }

  void load_state_dict(const StateDict& state_dict) {
    encoder_->load_state_dict(state_dict.get_dict_with_prefix("encoder."));
    decoder_->load_state_dict(state_dict.get_dict_with_prefix("decoder."));
    quant_->load_state_dict(state_dict.get_dict_with_prefix("quant_conv."));
    post_quant_->load_state_dict(
        state_dict.get_dict_with_prefix("post_quant_conv."));
  }

  void verify_loaded_weights(const std::string& prefix = "") const {
    encoder_->verify_loaded_weights(prefix + "encoder.");
    decoder_->verify_loaded_weights(prefix + "decoder.");
    quant_->verify_loaded_weights(prefix + "quant_conv.");
    post_quant_->verify_loaded_weights(prefix + "post_quant_conv.");
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    CHECK(loader != nullptr);
    for (const auto& state_dict : loader->get_state_dicts()) {
      load_state_dict(*state_dict);
    }
    verify_loaded_weights("");
  }

 private:
  torch::Tensor encode_moments_untiled(const torch::Tensor& image) {
    return quant_->forward(encoder_->forward(image));
  }
  torch::Tensor decode_untiled(const torch::Tensor& latent) {
    return decoder_->forward(post_quant_->forward(latent));
  }
  static void check_input(const torch::Tensor& input, int64_t channels) {
    CHECK_EQ(input.dim(), 5);
    CHECK_EQ(input.size(0), 1);
    CHECK_EQ(input.size(1), channels);
    CHECK_EQ(input.size(2), 1);
    CHECK_GT(input.size(3), 0);
    CHECK_GT(input.size(4), 0);
  }
  std::pair<torch::Tensor, torch::Tensor> latent_statistics(
      const torch::Tensor& latent) const {
    check_input(latent, config_.z_dim);
    CHECK_EQ(static_cast<int64_t>(config_.latents_mean.size()), config_.z_dim);
    CHECK_EQ(static_cast<int64_t>(config_.latents_std.size()), config_.z_dim);
    for (double stddev : config_.latents_std) {
      CHECK_GT(stddev, 0.0);
    }
    auto mean = torch::tensor(config_.latents_mean, torch::kFloat64)
                    .to(latent.options())
                    .reshape({1, config_.z_dim, 1, 1, 1});
    auto stddev = torch::tensor(config_.latents_std, torch::kFloat64)
                      .to(latent.options())
                      .reshape({1, config_.z_dim, 1, 1, 1});
    return {mean, stddev};
  }
  QwenImage21VaeConfig config_;
  int64_t spatial_scale_;
  bool tiling_enabled_ = false;
  qwen_image_21::VaeTilingConfig tiling_config_;
  qwen_image_21::VaeEncoder encoder_{nullptr};
  qwen_image_21::VaeDecoder decoder_{nullptr};
  qwen_image_21::Conv2d quant_{nullptr}, post_quant_{nullptr};
};
TORCH_MODULE(AutoencoderKLQwenImage21);

}  // namespace xllm
