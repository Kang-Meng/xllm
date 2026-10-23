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
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "core/framework/parallel_state/process_group.h"
#include "core/framework/state_dict/state_dict.h"
#include "models/dit/transformers/transformer_qwen_image_21.h"

namespace xllm {
namespace {

void expect_noise_close(const torch::Tensor& actual,
                        const torch::Tensor& expected) {
  auto a = actual.to(torch::kCPU, torch::kFloat64).flatten();
  auto e = expected.to(torch::kCPU, torch::kFloat64).flatten();
  ASSERT_TRUE(torch::isfinite(a).all().item<bool>());
  const double relative_l2 =
      ((a - e).norm() / e.norm().clamp_min(1e-12)).item<double>();
  const double cosine =
      (a.dot(e) / (a.norm() * e.norm()).clamp_min(1e-12)).item<double>();
  LOG(INFO) << "QwenImage21 cached noise: relative_l2=" << relative_l2
            << " cosine=" << cosine;
  EXPECT_LT(relative_l2, 0.01);
  EXPECT_GT(cosine, 0.999);
}

TEST(QwenImage21CacheNpu, RectangularAttentionPreservesPaddingAndTargetRows) {
  torch::NoGradGuard guard;
  torch::manual_seed(21);
  const auto device = torch::Device(torch::kPrivateUse1, 0);
  for (int64_t prefix : {6, 15, 56, 128, 143}) {
    const int64_t total = prefix + 256;
    auto q = torch::randn({1, 2, total, 128}).to(device, torch::kBFloat16);
    auto k = torch::randn_like(q);
    auto v = torch::randn_like(q);
    for (bool padding : {false, true}) {
      auto allowed = torch::ones({1, 1, total, total}, torch::kBool).tril();
      allowed.slice(2, prefix).fill_(true);
      if (padding) {
        allowed.select(3, 2).fill_(false);
      }
      allowed = allowed.to(device);
      auto full = qwen_image_21::transformer_attention(q, k, v, allowed);
      auto decode_mask =
          padding ? allowed.slice(2, prefix).clone() : torch::Tensor();
      auto target = qwen_image_21::transformer_attention(
          q.slice(2, prefix), k, v, decode_mask, prefix);
      expect_noise_close(target, full.slice(2, prefix));
    }
  }
}

TEST(QwenImage21CacheNpu, Bf16TransformerCacheMatchesChangingTimesteps) {
  torch::NoGradGuard guard;
  torch::manual_seed(21);
  const auto device = torch::Device(torch::kPrivateUse1, 0);
  QwenImage21TransformerConfig config;
  config.num_heads = 2;
  config.num_layers = 4;
  config.context_in_dim = 32;
  auto options = torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  QwenImage21Transformer2DModel model(config, options);
  for (auto& weight : model->parameters()) {
    weight.copy_(torch::randn(weight.sizes(), torch::kFloat32) * 0.02);
  }
  for (bool padding : {false, true}) {
    auto mask =
        torch::tensor({{false, true, true, false, false}}, torch::kBool);
    auto valid = torch::ones({1, 5}, torch::kBool);
    if (padding) {
      valid[0][4] = false;
    }
    auto layout = model->prepare_layout(
        mask, valid, {{1, 2, 2}, {1, 2, 2}, {1, 4, 4}}, device);
    auto prompt = torch::randn({1, 5, 32}).to(options);
    auto hidden = torch::randn({1, 24, 64}).to(options);
    qwen_image_21::PrefixKVCache cache(config.num_layers);
    auto first = model->forward(
        hidden, prompt, torch::tensor({1.0f}).to(options), layout, &cache);
    ASSERT_TRUE(cache.ready());
    auto saved_key = cache.layer(0).key.cpu().clone();
    for (float timestep : {0.7f, 0.25f}) {
      auto target = torch::randn({1, 16, 64}).to(options);
      hidden.slice(1, 8).copy_(target);
      auto t = torch::tensor({timestep}).to(options);
      auto expected = model->forward(hidden, prompt, t, layout);
      auto actual = model->forward(target, prompt, t, layout, &cache);
      expect_noise_close(actual, expected);
      EXPECT_TRUE(torch::equal(saved_key, cache.layer(0).key.cpu()));
    }
  }
}

// Opt-in real 32-layer checkpoint comparison; ordinary CI does not allocate
// these weights or depend on a developer's local model/fixture directory.
TEST(QwenImage21CacheNpu, RealCheckpointNoiseParity) {
  const char* model_dir = std::getenv("XLLM_QWEN21_TEST_MODEL");
  const char* fixture_file = std::getenv("XLLM_QWEN21_TEST_EMBEDDINGS");
  if (model_dir == nullptr || fixture_file == nullptr) {
    GTEST_SKIP() << "Set XLLM_QWEN21_TEST_MODEL and "
                    "XLLM_QWEN21_TEST_EMBEDDINGS to opt in";
  }
  torch::NoGradGuard guard;
  torch::manual_seed(21);
  const auto device = torch::Device(torch::kPrivateUse1, 0);
  auto options = torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  QwenImage21Transformer2DModel model(QwenImage21TransformerConfig{}, options);
  std::vector<std::filesystem::path> files;
  for (const auto& file : std::filesystem::directory_iterator(
           std::filesystem::path(model_dir) / "transformer")) {
    if (file.path().extension() == ".safetensors") {
      files.emplace_back(file.path());
    }
  }
  std::sort(files.begin(), files.end());
  for (const auto& file : files) {
    auto weights = StateDictFromSafeTensor::load(file.string());
    model->load_state_dict(*weights);
  }
  model->verify_loaded_weights();
  auto fixture = StateDictFromSafeTensor::load(fixture_file);
  for (const std::string& prefix : {std::string(), std::string("negative_")}) {
    auto prompt = fixture->get_tensor(prefix + "prompt_embeds").to(options);
    auto mask = fixture->get_tensor(prefix + "image_pad_mask");
    auto valid = fixture->get_tensor(prefix + "prompt_embeds_mask");
    if (prompt.dim() == 2) {
      ASSERT_EQ(mask.dim(), 1);
      ASSERT_EQ(valid.dim(), 1);
      prompt = prompt.unsqueeze(0);
      mask = mask.unsqueeze(0);
      valid = valid.unsqueeze(0);
    }
    ASSERT_EQ(mask.sum().item<int64_t>(), 1024)
        << "This opt-in case expects one 1024x1024 condition image";
    auto layout =
        model->prepare_layout(mask, valid, {{1, 64, 64}, {1, 64, 64}}, device);
    auto full = torch::randn({1, 8192, 64}).to(options);
    qwen_image_21::PrefixKVCache cache(32);
    auto t = torch::tensor({1.0f}).to(options);
    auto expected = model->forward(full, prompt, t, layout);
    auto prefill = model->forward(full, prompt, t, layout, &cache);
    expect_noise_close(prefill, expected);
    LOG(INFO) << "Real checkpoint branch=" << prefix
              << " kv_bytes=" << cache.bytes();
    for (float timestep : {0.5f, 0.1f}) {
      auto target = torch::randn({1, 4096, 64}).to(options);
      full.slice(1, 4096).copy_(target);
      t = torch::tensor({timestep}).to(options);
      qwen_image_21::PrefixKVCache recomputed(32);
      expected = model->forward(full, prompt, t, layout, &recomputed);
      for (int64_t index = 0; index < 32; ++index) {
        const float key_difference =
            (cache.layer(index).key - recomputed.layer(index).key)
                .abs()
                .max()
                .item<float>();
        const float value_difference =
            (cache.layer(index).value - recomputed.layer(index).value)
                .abs()
                .max()
                .item<float>();
        EXPECT_EQ(key_difference, 0) << "Prefix key changed at layer " << index;
        EXPECT_EQ(value_difference, 0)
            << "Prefix value changed at layer " << index;
      }
      auto actual = model->forward(target, prompt, t, layout, &cache);
      expect_noise_close(actual, expected);
    }
  }
}

void compare_sp(const torch::Tensor& expected, const torch::Tensor& actual) {
  auto e = expected.to(torch::kCPU, torch::kFloat64),
       a = actual.to(torch::kCPU, torch::kFloat64);
  ASSERT_EQ(e.sizes(), a.sizes());
  ASSERT_TRUE(torch::isfinite(a).all().item<bool>());
  const double relative =
      ((a - e).norm() / e.norm().clamp_min(1e-12)).item<double>();
  LOG(INFO) << "SP noise relative_l2=" << relative
            << " max_abs=" << (a - e).abs().max().item<double>();
  EXPECT_LT(relative, 0.01);
}

TEST(QwenImage21SpNpu, TwoRankUlyssesAndPrefixCacheParity) {
  const char* rank_env = std::getenv("XLLM_QWEN21_SP_TEST_RANK");
  const char* port_env = std::getenv("XLLM_QWEN21_SP_TEST_PORT");
  if (!rank_env || !port_env) {
    GTEST_SKIP() << "Opt in by launching rank0/1 with a shared test port";
  }
  const int32_t rank = std::stoi(rank_env);
  const auto device = torch::Device(torch::kPrivateUse1, 0);
  auto group = create_process_group(rank,
                                    2,
                                    2,
                                    std::stoi(port_env),
                                    false,
                                    "127.0.0.1",
                                    "qwen21_sp_test",
                                    device);
  torch::NoGradGuard guard;
  torch::manual_seed(21);
  auto options = torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  QwenImage21TransformerConfig config;
  const char* model_root = std::getenv("XLLM_QWEN21_TEST_MODEL");
  const char* fixture = std::getenv("XLLM_QWEN21_TEST_EMBEDDINGS");
  const bool real = model_root && fixture;
  if (!real) {
    config.num_heads = 4;
    config.num_layers = 4;
    config.context_in_dim = 32;
  }
  QwenImage21Transformer2DModel model(config, options);
  if (real) {
    std::vector<std::filesystem::path> files;
    for (const auto& f : std::filesystem::directory_iterator(
             std::filesystem::path(model_root) / "transformer")) {
      if (f.path().extension() == ".safetensors") {
        files.emplace_back(f.path());
      }
    }
    std::sort(files.begin(), files.end());
    for (const auto& f : files) {
      auto weights = StateDictFromSafeTensor::load(f.string());
      model->load_state_dict(*weights);
    }
    model->verify_loaded_weights();
  } else {
    for (auto& weight : model->parameters()) {
      weight.copy_(torch::randn(weight.sizes(), torch::kFloat32) * 0.02);
    }
  }
  for (int32_t branch = 0; branch < 2; ++branch) {
    torch::Tensor prompt, mask, valid;
    std::vector<qwen_image_21::ImageShape> shapes;
    int64_t condition_tokens = 4, target_tokens = 24;
    if (real) {
      auto data = StateDictFromSafeTensor::load(fixture);
      const std::string prefix = branch == 0 ? "" : "negative_";
      prompt = data->get_tensor(prefix + "prompt_embeds").to(options);
      mask = data->get_tensor(prefix + "image_pad_mask").clone();
      valid = data->get_tensor(prefix + "prompt_embeds_mask").clone();
      if (prompt.dim() == 2) {
        ASSERT_EQ(mask.dim(), 1);
        ASSERT_EQ(valid.dim(), 1);
        prompt = prompt.unsqueeze(0);
        mask = mask.unsqueeze(0);
        valid = valid.unsqueeze(0);
      }
      condition_tokens = 4096;
      target_tokens = 4096;
      shapes = {{1, 64, 64}, {1, 64, 64}};
    } else {
      prompt = torch::randn({1, 5, 32}).to(options);
      mask = torch::tensor({{false, true, false, false, false}}, torch::kBool);
      valid = torch::ones({1, 5}, torch::kBool);
      if (branch == 1) {
        valid[0][4] = false;
      }
      shapes = {{1, 2, 2}, {1, 4, 6}};
    }
    auto layout = model->prepare_layout(mask, valid, shapes, device);
    qwen_image_21::UlyssesContext sp(layout, group.get());
    auto full =
        torch::randn({1, condition_tokens + target_tokens, 64}).to(options);
    qwen_image_21::PrefixKVCache cache(config.num_layers);
    for (float time : {1.0f, 0.5f, 0.1f}) {
      auto target = torch::randn({1, target_tokens, 64}).to(options);
      full.slice(1, condition_tokens).copy_(target);
      auto t = torch::tensor({time}).to(options);
      auto expected = model->forward(full, prompt, t, layout);
      auto uncached = model->forward(full, prompt, t, layout, nullptr, &sp);
      compare_sp(expected, uncached);
      auto cached = model->forward(
          cache.ready() ? target : full, prompt, t, layout, &cache, &sp);
      compare_sp(expected, cached);
    }
    EXPECT_EQ(cache.layer(0).key.size(1), config.num_heads);
    EXPECT_EQ(cache.layer(0).key.size(2), layout.prefix_tokens);
  }
}

}  // namespace
}  // namespace xllm
