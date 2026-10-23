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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "core/framework/model_context.h"
#include "core/framework/parallel_state/process_group.h"
#include "models/dit/autoencoders/autoencoder_kl_qwenimage21.h"

namespace xllm {
namespace {

using qwen_image_21::VaeTilingConfig;

// Deliberately non-streaming, scalar-loop translation of the reference
// tiled_encode/tiled_decode: materialize all rows, blend above then left, crop
// to stride, concatenate, crop final size. Do not call the production blender.
template <typename Transform>
torch::Tensor reference_tiling(const torch::Tensor& input,
                               const VaeTilingConfig& config,
                               int64_t input_scale,
                               int64_t output_scale,
                               Transform&& transform) {
  const int64_t scale = std::max(input_scale, output_scale);
  const int64_t th = config.tile_height / scale * input_scale;
  const int64_t tw = config.tile_width / scale * input_scale;
  const int64_t sh = config.stride_height / scale * input_scale;
  const int64_t sw = config.stride_width / scale * input_scale;
  const int64_t bh =
      (config.tile_height - config.stride_height) / scale * output_scale;
  const int64_t bw =
      (config.tile_width - config.stride_width) / scale * output_scale;
  const int64_t osh = sh / input_scale * output_scale;
  const int64_t osw = sw / input_scale * output_scale;
  std::vector<std::vector<torch::Tensor>> rows;
  rows.reserve((input.size(3) + sh - 1) / sh);
  for (int64_t y = 0; y < input.size(3); y += sh) {
    std::vector<torch::Tensor> row;
    row.reserve((input.size(4) + sw - 1) / sw);
    for (int64_t x = 0; x < input.size(4); x += sw) {
      row.emplace_back(transform(
          input.slice(3, y, y + th).slice(4, x, x + tw).contiguous()));
    }
    rows.emplace_back(std::move(row));
  }
  std::vector<torch::Tensor> output_rows;
  output_rows.reserve(rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    std::vector<torch::Tensor> output_row;
    output_row.reserve(rows[i].size());
    for (size_t j = 0; j < rows[i].size(); ++j) {
      auto& tile = rows[i][j];
      if (i > 0) {
        const auto& above = rows[i - 1][j];
        const int64_t extent = std::min({bh, above.size(3), tile.size(3)});
        for (int64_t y = 0; y < extent; ++y) {
          const double ratio = static_cast<double>(y) / extent;
          tile.select(3, y).copy_(above.select(3, above.size(3) - extent + y) *
                                      (1 - ratio) +
                                  tile.select(3, y) * ratio);
        }
      }
      if (j > 0) {
        const auto& left = rows[i][j - 1];
        const int64_t extent = std::min({bw, left.size(4), tile.size(4)});
        for (int64_t x = 0; x < extent; ++x) {
          const double ratio = static_cast<double>(x) / extent;
          tile.select(4, x).copy_(left.select(4, left.size(4) - extent + x) *
                                      (1 - ratio) +
                                  tile.select(4, x) * ratio);
        }
      }
      output_row.emplace_back(tile.slice(3, 0, osh).slice(4, 0, osw));
    }
    output_rows.emplace_back(torch::cat(output_row, 4));
  }
  return torch::cat(output_rows, 3)
      .slice(3, 0, input.size(3) / input_scale * output_scale)
      .slice(4, 0, input.size(4) / input_scale * output_scale);
}

QwenImage21VaeConfig small_vae_config() {
  QwenImage21VaeConfig config;
  config.base_dim = 2;
  config.decoder_base_dim = 4;
  config.z_dim = 2;
  config.num_res_blocks = 1;
  config.dim_mult = {1, 2, 2};
  config.temperal_downsample = {false, true};
  config.latents_mean = {0.3, -0.4};
  config.latents_std = {1.5, 2.5};
  return config;
}

void fill_test_weights(torch::nn::Module& module) {
  torch::manual_seed(21);
  for (auto& parameter : module.parameters()) {
    parameter.normal_(0, 0.15);
  }
}

TEST(QwenImage21VaeTiling, DefaultsMatchSixteenFoldReference) {
  VaeTilingConfig config;
  EXPECT_EQ(config.tile_height, 512);
  EXPECT_EQ(config.tile_width, 512);
  EXPECT_EQ(config.stride_height, 384);
  EXPECT_EQ(config.stride_width, 384);
  qwen_image_21::validate_vae_tiling(config, 16);
}

TEST(QwenImage21VaeTiling, BlendUsesReferenceRampAndDoesNotMutateNeighbors) {
  torch::NoGradGuard guard;
  auto previous = torch::full({1, 4, 1, 4, 5}, 10.0);
  auto current = torch::full({1, 4, 1, 3, 5}, 2.0);
  auto saved = previous.clone();
  qwen_image_21::blend_vae_tile(previous, current, 3, 4);
  EXPECT_TRUE(torch::equal(previous, saved));
  EXPECT_FLOAT_EQ(current[0][3][0][0][0].item<float>(), 10.0f);
  EXPECT_NEAR(current[0][3][0][1][0].item<float>(), 22.0f / 3, 1e-6f);
  EXPECT_NEAR(current[0][3][0][2][0].item<float>(), 14.0f / 3, 1e-6f);
  // Alpha is blended with the same ramp, not discarded or forced opaque.
  EXPECT_TRUE(torch::equal(current.select(1, 0), current.select(1, 3)));
  auto no_overlap = current.clone();
  qwen_image_21::blend_vae_tile(previous, current, 3, 0);
  EXPECT_TRUE(torch::equal(current, no_overlap));
}

TEST(QwenImage21VaeTiling, StreamingMatchesReferenceOnPartialRectangularTiles) {
  torch::NoGradGuard guard;
  torch::manual_seed(23);
  for (const auto& shape : std::vector<std::array<int64_t, 2>>{
           {4, 20}, {20, 4}, {20, 28}, {24, 32}, {32, 24}}) {
    const VaeTilingConfig config{16, 24, 12, 16};
    auto input = torch::randn({1, 4, 1, shape[0], shape[1]});
    auto saved = input.clone();
    auto transform = [](const torch::Tensor& tile) {
      // Tile-dependent response makes the blending order observable.
      return (tile.slice(3, 0, tile.size(3), 4).slice(4, 0, tile.size(4), 4) +
              tile.mean())
          .clone();
    };
    auto expected = reference_tiling(input, config, 4, 1, transform);
    auto actual =
        qwen_image_21::tiled_vae_transform(input, config, 4, 1, transform);
    EXPECT_TRUE(torch::allclose(actual, expected, 1e-6, 1e-6));
    EXPECT_TRUE(torch::equal(input, saved));
    EXPECT_EQ(actual.sizes(),
              (torch::IntArrayRef{1, 4, 1, shape[0] / 4, shape[1] / 4}));
  }
}

TEST(QwenImage21VaeTiling, DecodeGeometryHasNoHolesAtSixteenFoldScale) {
  torch::NoGradGuard guard;
  // Covers tall, wide, odd latent dimensions, stride-exact and tiny edge tiles.
  for (const auto& shape : std::vector<std::array<int64_t, 2>>{
           {17, 65}, {65, 17}, {64, 64}, {49, 73}, {48, 72}}) {
    auto input = torch::arange(4 * shape[0] * shape[1], torch::kFloat32)
                     .reshape({1, 4, 1, shape[0], shape[1]});
    int64_t calls = 0;
    auto transform = [&](const torch::Tensor& tile) {
      ++calls;
      EXPECT_LE(tile.size(3), 32);
      EXPECT_LE(tile.size(4), 32);
      return tile.repeat_interleave(16, 3).repeat_interleave(16, 4);
    };
    auto actual =
        qwen_image_21::tiled_vae_transform(input, {}, 1, 16, transform);
    auto expected = input.repeat_interleave(16, 3).repeat_interleave(16, 4);
    EXPECT_TRUE(torch::allclose(actual, expected, 1e-6, 1e-5));
    EXPECT_EQ(calls, ((shape[0] + 23) / 24) * ((shape[1] + 23) / 24));
    EXPECT_TRUE(torch::isfinite(actual).all().item<bool>());
  }
}

TEST(QwenImage21VaeTiling, LargeOverlapAndMinimalStrideMatchReference) {
  torch::NoGradGuard guard;
  auto input = torch::randn({1, 4, 1, 9, 11});
  const VaeTilingConfig config{16, 20, 4, 4};
  auto transform = [](const torch::Tensor& tile) {
    auto value = tile.repeat_interleave(4, 3).repeat_interleave(4, 4);
    return value + tile.mean();
  };
  auto expected = reference_tiling(input, config, 1, 4, transform);
  auto actual =
      qwen_image_21::tiled_vae_transform(input, config, 1, 4, transform);
  EXPECT_TRUE(torch::allclose(actual, expected, 1e-6, 1e-6));
}

TEST(QwenImage21VaeTiling, DisabledAndBelowThresholdRemainBitwiseBaseline) {
  torch::NoGradGuard guard;
  AutoencoderKLQwenImage21 vae(ModelContext{}, small_vae_config());
  fill_test_weights(*vae);
  EXPECT_FALSE(vae->tiling_enabled());
  auto image = torch::randn({1, 4, 1, 16, 24});
  auto latent = torch::randn({1, 2, 1, 4, 6});
  auto encoded = vae->encode_moments(image);
  auto decoded = vae->decode(latent);
  vae->enable_tiling({16, 24, 12, 16});
  EXPECT_TRUE(vae->tiling_enabled());
  EXPECT_TRUE(torch::equal(vae->encode_moments(image), encoded));
  EXPECT_TRUE(torch::equal(vae->decode(latent), decoded));
  auto large_image = torch::randn({1, 4, 1, 28, 36});
  vae->disable_tiling();
  auto before = vae->encode_moments(large_image);
  vae->enable_tiling({16, 24, 12, 16});
  vae->encode_moments(large_image);
  vae->disable_tiling();
  EXPECT_TRUE(torch::equal(before, vae->encode_moments(large_image)));
}

TEST(QwenImage21VaeTiling, EncoderMatchesScalarReferenceAndNormalizesOnce) {
  torch::NoGradGuard guard;
  AutoencoderKLQwenImage21 vae(ModelContext{}, small_vae_config());
  fill_test_weights(*vae);
  const VaeTilingConfig config{16, 24, 12, 16};
  auto image = torch::randn({1, 4, 1, 28, 36});
  auto expected =
      reference_tiling(image, config, 4, 1, [&](const torch::Tensor& tile) {
        return vae->encode_moments(tile);
      });
  vae->enable_tiling(config);
  auto actual = vae->encode_moments(image);
  EXPECT_TRUE(torch::allclose(actual, expected, 1e-6, 1e-6));
  EXPECT_EQ(actual.sizes(), (torch::IntArrayRef{1, 4, 1, 7, 9}));
  auto mode = vae->encode_mode(image);
  EXPECT_TRUE(torch::equal(mode, actual.chunk(2, 1)[0]));
  EXPECT_TRUE(
      torch::allclose(vae->denormalize_latents(vae->normalize_latents(mode)),
                      mode,
                      1e-5,
                      1e-6));
}

TEST(QwenImage21VaeTiling, DecoderMatchesRawBlendThenClampReference) {
  torch::NoGradGuard guard;
  AutoencoderKLQwenImage21 vae(ModelContext{}, small_vae_config());
  fill_test_weights(*vae);
  const VaeTilingConfig config{16, 24, 12, 16};
  auto latent = torch::randn({1, 2, 1, 7, 9});
  auto modules = vae->named_modules();
  auto* post_quant =
      modules["post_quant_conv"]->as<qwen_image_21::Conv2dImpl>();
  auto* decoder = modules["decoder"]->as<qwen_image_21::VaeDecoderImpl>();
  auto expected =
      reference_tiling(latent, config, 1, 4, [&](const torch::Tensor& tile) {
        return decoder->forward(post_quant->forward(tile));
      }).clamp(-1.0, 1.0);
  vae->enable_tiling(config);
  auto actual = vae->decode(latent);
  EXPECT_TRUE(torch::allclose(actual, expected, 1e-6, 1e-6));
  EXPECT_EQ(actual.sizes(), (torch::IntArrayRef{1, 4, 1, 28, 36}));
  EXPECT_TRUE(torch::isfinite(actual).all().item<bool>());
  EXPECT_LE(actual.abs().max().item<float>(), 1.0f);
  auto other = torch::randn({1, 2, 1, 9, 5});
  vae->decode(other);
  EXPECT_TRUE(torch::equal(actual, vae->decode(latent)));
}

TEST(QwenImage21VaeTiling, InvalidGeometryRejectedBeforeForward) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(qwen_image_21::validate_vae_tiling({0, 512, 384, 384}, 16),
               "tile_height");
  EXPECT_DEATH(qwen_image_21::validate_vae_tiling({512, 512, 0, 384}, 16),
               "stride_height");
  EXPECT_DEATH(qwen_image_21::validate_vae_tiling({512, 512, -16, 384}, 16),
               "stride_height");
  EXPECT_DEATH(qwen_image_21::validate_vae_tiling({512, 512, 512, 384}, 16),
               "vertical overlap");
  EXPECT_DEATH(qwen_image_21::validate_vae_tiling({512, 512, 384, 528}, 16),
               "horizontal overlap");
  EXPECT_DEATH(qwen_image_21::validate_vae_tiling({513, 512, 384, 384}, 16),
               "tile_height");
  EXPECT_DEATH(qwen_image_21::validate_vae_tiling({512, 512, 384, 385}, 16),
               "stride_width");
}

// CPU transport for the production mixin. Each rank runs on one thread;
// sends block until the peer has copied the tensor, just like ProcessGroup.
class CpuCollective final {
 public:
  explicit CpuCollective(int32_t size)
      : size_(size), slots_(size), messages_(size * size), barrier_(size) {}
  std::vector<torch::Tensor> gather(int32_t rank, const torch::Tensor& value) {
    slots_[rank] = value.clone();
    barrier_.arrive_and_wait();
    auto result = slots_;
    barrier_.arrive_and_wait();
    return result;
  }

  void send(int32_t source, int32_t destination, const torch::Tensor& value) {
    const int32_t slot = source * size_ + destination;
    std::unique_lock<std::mutex> lock(mutex_);
    CHECK(!messages_[slot].defined());
    messages_[slot] = value.clone();
    ready_.notify_all();
    CHECK(ready_.wait_for(lock, std::chrono::seconds(30), [&] {
      return !messages_[slot].defined();
    })) << "Timed out waiting for VAE halo receiver";
  }

  void recv(int32_t source, int32_t destination, torch::Tensor& value) {
    const int32_t slot = source * size_ + destination;
    std::unique_lock<std::mutex> lock(mutex_);
    CHECK(ready_.wait_for(lock, std::chrono::seconds(30), [&] {
      return messages_[slot].defined();
    })) << "Timed out waiting for VAE halo sender";
    CHECK(value.sizes() == messages_[slot].sizes());
    value.copy_(messages_[slot]);
    messages_[slot] = torch::Tensor();
    ready_.notify_all();
  }

 private:
  int32_t size_;
  std::vector<torch::Tensor> slots_;
  std::vector<torch::Tensor> messages_;
  std::barrier<> barrier_;
  std::mutex mutex_;
  std::condition_variable ready_;
};

class CpuVaeProcessGroup final : public ProcessGroup {
 public:
  CpuVaeProcessGroup(int32_t rank,
                     int32_t size,
                     std::shared_ptr<CpuCollective> group)
      : ProcessGroup(rank, size, torch::Device(torch::kCPU)),
        group_(std::move(group)) {}

  void allgather(const torch::Tensor& value,
                 std::vector<torch::Tensor>& outputs) override {
    auto gathered = group_->gather(rank(), value);
    CHECK_EQ(outputs.size(), gathered.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      CHECK(outputs[i].sizes() == gathered[i].sizes());
      outputs[i].copy_(gathered[i]);
    }
  }

  void send(const torch::Tensor& value, int destination, int tag) override {
    CHECK_EQ(tag, 0);
    group_->send(rank(), destination, value);
  }

  void recv(torch::Tensor& value, int source, int tag) override {
    CHECK_EQ(tag, 0);
    group_->recv(source, rank(), value);
  }

 private:
  std::shared_ptr<CpuCollective> group_;
};

ModelContext parallel_context(ProcessGroup* group) {
  ParallelArgs args(
      group->rank(), group->world_size(), /*process_group=*/nullptr);
  args.vae_size() = group->world_size();
  args.dit_vae_group_ = group;
  const auto options = torch::TensorOptions().device(torch::kCPU);
#if defined(USE_NPU)
  // The four-argument constructor initializes an NPU even for CPU options.
  return ModelContext(
      args, ModelArgs{}, QuantArgs{}, options, /*context=*/nullptr);
#else
  return ModelContext(args, ModelArgs{}, QuantArgs{}, options);
#endif
}

template <typename Operation>
std::vector<torch::Tensor> run_ranks(int32_t size, Operation&& operation) {
  auto group = std::make_shared<CpuCollective>(size);
  std::vector<std::future<torch::Tensor>> futures;
  futures.reserve(size);
  for (int32_t rank = 0; rank < size; ++rank) {
    futures.emplace_back(std::async(std::launch::async, [&, rank, group] {
      torch::NoGradGuard guard;
      CpuVaeProcessGroup process_group(rank, size, group);
      const auto context = parallel_context(&process_group);
      const dit::VaeParallelMixin parallel(context);
      return operation(context, parallel);
    }));
  }
  std::vector<torch::Tensor> results;
  results.reserve(size);
  for (auto& future : futures) {
    results.emplace_back(future.get());
  }
  return results;
}

QwenImage21VaeConfig parallel_test_config() {
  QwenImage21VaeConfig config;
  config.base_dim = 2;
  config.decoder_base_dim = 4;
  config.z_dim = 2;
  config.num_res_blocks = 1;
  config.dim_mult = {1, 2, 2};
  config.temperal_downsample = {false, true};
  return config;
}

void deterministic_weights(torch::nn::Module& model) {
  torch::NoGradGuard guard;
  for (auto& value : model.parameters()) {
    value.copy_(
        ((torch::arange(value.numel(), torch::kFloat32).remainder(19) - 9) *
         0.02)
            .reshape(value.sizes()));
  }
}

TEST(QwenImage21VaeParallel, LatentGridPartitionPreservesStrideAlignment) {
  auto latent = torch::arange(11, torch::kFloat32);
  auto pixels = latent.repeat_interleave(16);
  for (int32_t size : {2, 3, 4, 8}) {
    auto shards = run_ranks(size, [&](const auto&, const auto& parallel) {
      auto local_pixels = parallel.vae_parallel_split(pixels, /*alignment=*/16);
      auto local_latent = parallel.vae_parallel_split(latent, /*alignment=*/1);
      EXPECT_TRUE(
          torch::equal(local_pixels, local_latent.repeat_interleave(16)));
      EXPECT_EQ(local_pixels.size(-1) % 16, 0);
      EXPECT_GT(local_latent.size(-1), 0);
      return local_pixels;
    });
    EXPECT_TRUE(torch::equal(torch::cat(shards, -1), pixels));
  }
}

TEST(QwenImage21VaeParallel, HaloAndVariableWidthMergeAreExact) {
  auto full =
      torch::arange(2 * 3 * 11, torch::kFloat32).reshape({1, 2, 1, 3, 11});
  for (int32_t size : {2, 3, 4}) {
    auto results = run_ranks(size, [&](const auto&, const auto& parallel) {
      auto local = parallel.vae_parallel_split(full, /*alignment=*/1);
      auto halo = parallel.vae_parallel_exchange(local, /*pad=*/true);
      const int64_t start = local.flatten()[0].template item<int64_t>();
      const int64_t length = local.size(-1);
      auto padded = torch::nn::functional::pad(
          full, torch::nn::functional::PadFuncOptions({1, 1}));
      EXPECT_TRUE(
          torch::equal(halo, padded.slice(-1, start, start + length + 2)));
      auto right =
          parallel.vae_parallel_exchange(local, /*pad=*/true).slice(-1, 1);
      EXPECT_TRUE(
          torch::equal(right, padded.slice(-1, start + 1, start + length + 2)));
      return parallel.vae_parallel_merge(local);
    });
    for (const auto& result : results) {
      EXPECT_TRUE(torch::equal(result, full));
    }
  }
}

TEST(QwenImage21VaeParallel, ConvolutionAcrossSeamsMatchesSerial) {
  torch::NoGradGuard guard;
  torch::set_num_threads(1);
  qwen_image_21::Conv2d serial(ModelContext{},
                               4,
                               6,
                               3,
                               1,
                               1,
                               torch::TensorOptions().dtype(torch::kFloat32));
  deterministic_weights(*serial);
  auto input = torch::randn({1, 4, 1, 8, 19});
  auto expected = serial->forward(input);
  for (int32_t size : {2, 3, 4}) {
    auto outputs =
        run_ranks(size, [&](const auto& context, const auto& parallel) {
          qwen_image_21::Conv2d local(
              context,
              4,
              6,
              3,
              1,
              1,
              torch::TensorOptions().dtype(torch::kFloat32));
          deterministic_weights(*local);
          return parallel.vae_parallel_merge(
              local->forward(parallel.vae_parallel_split(input, 1)));
        });
    for (const auto& output : outputs) {
      EXPECT_TRUE(torch::allclose(output, expected, 1e-5, 1e-6));
    }
  }
}

TEST(QwenImage21VaeParallel, ResampleAndResidualShortcutsMatchGlobalGrid) {
  torch::NoGradGuard guard;
  torch::set_num_threads(1);
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  for (bool up : {false, true}) {
    qwen_image_21::VaeResample serial(ModelContext{}, 4, up, true, options);
    deterministic_weights(*serial);
    auto input = torch::randn({1, 4, 1, 8, 44});
    auto expected = serial->forward(input);
    for (int32_t size : {2, 3, 4}) {
      auto outputs =
          run_ranks(size, [&](const auto& context, const auto& parallel) {
            qwen_image_21::VaeResample local(context, 4, up, true, options);
            deterministic_weights(*local);
            auto shard = parallel.vae_parallel_split(input, 4);
            auto result = local->forward(shard);
            auto shortcut =
                up ? qwen_image_21::duplicate_upsample(shard, 4, 2, 2)
                   : qwen_image_21::average_downsample(shard, 4, 2, 2);
            auto full_shortcut =
                up ? qwen_image_21::duplicate_upsample(input, 4, 2, 2)
                   : qwen_image_21::average_downsample(input, 4, 2, 2);
            EXPECT_TRUE(torch::allclose(parallel.vae_parallel_merge(shortcut),
                                        full_shortcut,
                                        1e-6,
                                        1e-6));
            return parallel.vae_parallel_merge(result);
          });
      for (const auto& output : outputs) {
        EXPECT_TRUE(torch::allclose(output, expected, 1e-5, 1e-6));
      }
    }
  }
}

TEST(QwenImage21VaeParallel, AttentionUsesGlobalRowMajorKeys) {
  torch::NoGradGuard guard;
  torch::set_num_threads(1);
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  qwen_image_21::VaeAttention serial(ModelContext{}, 8, options);
  deterministic_weights(*serial);
  auto input = torch::randn({1, 8, 1, 3, 11});
  auto expected = serial->forward(input);
  auto outputs = run_ranks(3, [&](const auto& context, const auto& parallel) {
    qwen_image_21::VaeAttention local(context, 8, options);
    deterministic_weights(*local);
    return parallel.vae_parallel_merge(
        local->forward(parallel.vae_parallel_split(input, 1)));
  });
  for (const auto& output : outputs) {
    EXPECT_TRUE(torch::allclose(output, expected, 1e-5, 1e-6));
  }
}

TEST(QwenImage21VaeParallel, CompleteSmallVaeEncodeDecodeMatchesUntiledSerial) {
  torch::NoGradGuard guard;
  torch::set_num_threads(1);
  auto config = parallel_test_config();
  AutoencoderKLQwenImage21 serial(ModelContext{}, config);
  deterministic_weights(*serial);
  auto image = torch::randn({1, 4, 1, 12, 44});
  // Nonopaque alpha and uneven latent columns are part of the same computation.
  image.select(1, 3).copy_(
      torch::linspace(-1, 1, 44).reshape({1, 1, 1, 44}).expand({1, 1, 12, 44}));
  auto expected_encode = serial->encode_moments(image);
  auto latent = expected_encode.chunk(2, 1)[0].contiguous();
  auto expected_decode = serial->decode(latent);
  auto second_latent = torch::randn({1, 2, 1, 2, 13});
  auto expected_second = serial->decode(second_latent);
  for (int32_t size : {2, 3, 4}) {
    auto outputs = run_ranks(size, [&](const auto& context, const auto&) {
      AutoencoderKLQwenImage21 local(context, config);
      deterministic_weights(*local);
      EXPECT_TRUE(local->vae_parallel_enabled());
      auto encoded = local->encode_moments(image);
      EXPECT_TRUE(torch::allclose(encoded, expected_encode, 1e-5, 1e-6))
          << "encode max="
          << (encoded - expected_encode).abs().max().template item<float>();
      auto first = local->decode(latent);
      EXPECT_TRUE(torch::allclose(first, expected_decode, 1e-5, 1e-6));
      EXPECT_TRUE(torch::allclose(
          local->decode(second_latent), expected_second, 1e-5, 1e-6));
      return local->decode(latent);
    });
    for (const auto& output : outputs) {
      EXPECT_TRUE(torch::allclose(output, expected_decode, 1e-5, 1e-6));
    }
  }
}

TEST(QwenImage21VaeParallel, FullSpatialScaleAndSingleColumnShardsMatchSerial) {
  torch::NoGradGuard guard;
  torch::set_num_threads(1);
  auto config = parallel_test_config();
  // Production stage count and temporal flags, reduced channels for CPU CI.
  config.dim_mult = {1, 2, 4, 8, 8};
  config.temperal_downsample = {false, true, true, true};
  config.num_res_blocks = 2;
  AutoencoderKLQwenImage21 serial(ModelContext{}, config);
  deterministic_weights(*serial);
  ASSERT_EQ(serial->spatial_scale(), 16);
  auto image = torch::randn({1, 4, 1, 32, 80});
  auto expected_encode = serial->encode_moments(image);
  auto latent = expected_encode.chunk(2, 1)[0].contiguous();
  auto expected_decode = serial->decode(latent);
  // 5 latent columns over 4 ranks -> widths 2,1,1,1. Even the smallest
  // shards must preserve all four downsample/upsample boundaries.
  auto outputs = run_ranks(4, [&](const auto& context, const auto&) {
    AutoencoderKLQwenImage21 local(context, config);
    deterministic_weights(*local);
    EXPECT_TRUE(torch::allclose(
        local->encode_moments(image), expected_encode, 1e-5, 1e-6));
    return local->decode(latent);
  });
  for (const auto& output : outputs) {
    EXPECT_TRUE(torch::allclose(output, expected_decode, 1e-5, 1e-6));
  }
}

TEST(QwenImage21VaeParallel, SingleRankOperationsAreIdentity) {
  const dit::VaeParallelMixin parallel(ModelContext{});
  auto value = torch::randn({1, 2, 1, 3, 11});
  EXPECT_FALSE(parallel.vae_parallel_enabled());
  EXPECT_TRUE(parallel.vae_parallel_split(value).is_same(value));
  EXPECT_TRUE(parallel.vae_parallel_split(value, 1).is_same(value));
  EXPECT_TRUE(parallel.vae_parallel_merge(value).is_same(value));
  EXPECT_TRUE(parallel.vae_parallel_exchange(value, true).is_same(value));
  EXPECT_TRUE(parallel.vae_parallel_exchange(value, false).is_same(value));
  EXPECT_TRUE(parallel.vae_parallel_trim_halo(value, 2).is_same(value));
}

TEST(QwenImage21VaeParallel, LegacySplitStillAllowsEmptyShards) {
  auto value = torch::arange(2, torch::kFloat32);
  auto shards = run_ranks(4, [&](const auto&, const auto& parallel) {
    return parallel.vae_parallel_split(value);
  });
  ASSERT_EQ(shards.size(), 4);
  EXPECT_EQ(shards[0].numel(), 1);
  EXPECT_EQ(shards[1].numel(), 1);
  EXPECT_EQ(shards[2].numel(), 0);
  EXPECT_EQ(shards[3].numel(), 0);
  EXPECT_TRUE(torch::equal(torch::cat(shards, -1), value));
}

TEST(QwenImage21VaeParallel, InvalidWidthAndTilingCombinationRejected) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  auto group = std::make_shared<CpuCollective>(2);
  CpuVaeProcessGroup process_group(/*rank=*/0, /*size=*/2, group);
  const auto context = parallel_context(&process_group);
  const dit::VaeParallelMixin parallel(context);
  EXPECT_DEATH(parallel.vae_parallel_split(torch::empty({16}), 16),
               "aligned column");
  EXPECT_DEATH(parallel.vae_parallel_split(torch::empty({33}), 16),
               "alignment");
  EXPECT_DEATH(parallel.vae_parallel_split(torch::empty({32}), 0), "alignment");
  AutoencoderKLQwenImage21 serial(ModelContext{}, parallel_test_config());
  serial->enable_tiling();
  serial->disable_tiling();
  AutoencoderKLQwenImage21 vae(context, parallel_test_config());
  EXPECT_DEATH(vae->enable_tiling(), "cannot be combined");
  EXPECT_DEATH(vae->decode(torch::zeros({1, 2, 1, 2, 1})), "aligned column");
}

}  // namespace
}  // namespace xllm
