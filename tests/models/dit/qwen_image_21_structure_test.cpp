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

#include <cstdlib>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "models/dit/autoencoders/autoencoder_kl_qwenimage21.h"
#include "models/dit/transformers/transformer_qwen_image_21.h"

namespace xllm {
namespace {

QwenImage21TransformerConfig small_transformer_config() {
  QwenImage21TransformerConfig config;
  config.in_channels = 4;
  config.out_channels = 4;
  config.context_in_dim = 8;
  config.num_heads = 2;
  config.head_dim = 8;
  config.num_layers = 2;
  config.axes_dims_rope = {2, 2, 4};
  return config;
}

void initialize_weights(torch::nn::Module& module) {
  torch::NoGradGuard guard;
  torch::manual_seed(21);
  for (auto& tensor : module.parameters()) {
    tensor.normal_(0.0, 0.02);
  }
}

TEST(QwenImage21Structure, CheckpointParameterCountsAndShapes) {
  const auto options = torch::TensorOptions().device(torch::kMeta);
  QwenImage21Transformer2DModel transformer(QwenImage21TransformerConfig{},
                                            options.dtype(torch::kBFloat16));
  AutoencoderKLQwenImage21 vae(
      ModelContext{}, QwenImage21VaeConfig{}, options.dtype(torch::kFloat32));
  EXPECT_EQ(transformer->named_parameters().size(), 297);
  EXPECT_EQ(vae->named_parameters().size(), 238);
  EXPECT_EQ(transformer->named_parameters()["modulation.1.weight"].sizes(),
            (torch::IntArrayRef{16384, 4096}));
  EXPECT_EQ(vae->named_parameters()["decoder.conv_in.weight"].sizes(),
            (torch::IntArrayRef{1152, 64, 3, 3}));
  const char* manifest = std::getenv("XLLM_QWEN_IMAGE21_WEIGHT_MANIFEST");
  if (manifest == nullptr) {
    return;
  }
  std::ifstream stream(manifest);
  ASSERT_TRUE(stream.good());
  nlohmann::json expected;
  stream >> expected;
  for (const auto& [component, module] :
       std::vector<std::pair<std::string, torch::nn::Module*>>{
           {"transformer", transformer.get()}, {"vae", vae.get()}}) {
    auto parameters = module->named_parameters();
    ASSERT_EQ(parameters.size(), expected.at(component).size());
    for (const auto& [key, value] : expected.at(component).items()) {
      ASSERT_TRUE(parameters.contains(key)) << component << "." << key;
      EXPECT_EQ(parameters[key].sizes().vec(),
                value.at("shape").get<std::vector<int64_t>>())
          << key;
      EXPECT_EQ(
          parameters[key].scalar_type(),
          component == "transformer" ? torch::kBFloat16 : torch::kFloat32);
    }
  }
}

TEST(QwenImage21Structure, AdjacentImagesStaySeparateCausalBlocks) {
  auto mask = torch::tensor({{false, true, true, false}}, torch::kBool);
  auto layout = qwen_image_21::prepare_sequence_layout(
      mask, {}, {{1, 2, 2}, {1, 2, 2}, {1, 2, 2}}, {2, 2, 4}, torch::kCPU);
  EXPECT_EQ(layout.target_tokens, 4);
  EXPECT_EQ(layout.prefix_tokens, 10);
  auto allowed = layout.allowed_attention.squeeze();
  EXPECT_TRUE(allowed[1][4].item<bool>());
  EXPECT_FALSE(allowed[1][5].item<bool>());
  EXPECT_TRUE(allowed[5][1].item<bool>());
  EXPECT_FALSE(allowed[9][10].item<bool>());
  EXPECT_TRUE(allowed[10][13].item<bool>());
  EXPECT_TRUE(torch::equal(
      layout.image_ids,
      torch::tensor({-1, 0, 0, 0, 0, 1, 1, 1, 1, -1, 2, 2, 2, 2})));
}

TEST(QwenImage21Structure, PaddingKeysAreMaskedIncludingTargetQueries) {
  auto image_mask = torch::zeros({1, 3}, torch::kBool);
  auto valid = torch::tensor({{1, 1, 0}}, torch::kLong);
  auto layout = qwen_image_21::prepare_sequence_layout(
      image_mask, valid, {{1, 2, 2}}, {2, 2, 4}, torch::kCPU);
  EXPECT_FALSE(layout.allowed_attention.select(-1, 2).any().item<bool>());
  EXPECT_TRUE(layout.allowed_attention[0][0][3][6].item<bool>());
  EXPECT_FALSE(layout.allowed_attention[0][0][0][3].item<bool>());
}

TEST(QwenImage21Structure, ExportedSingleImageLayout) {
  // Layout of edit_20261003_203530_606670; no inference or fixture payload is
  // needed.
  auto mask = torch::zeros({1, 1464}, torch::kBool);
  mask.slice(1, 8, 1032).fill_(true);
  auto layout = qwen_image_21::prepare_sequence_layout(
      mask, {}, {{1, 64, 64}, {1, 64, 64}}, {16, 56, 56}, torch::kCPU);
  EXPECT_EQ(layout.prefix_tokens, 4536);
  EXPECT_EQ(layout.target_tokens, 4096);
  EXPECT_EQ(layout.source_indices.numel(), 8632);
  EXPECT_EQ(layout.image_positions.numel(), 8192);
  EXPECT_FALSE(layout.allowed_attention[0][0][8][4536].item<bool>());
  EXPECT_TRUE(layout.allowed_attention[0][0][4536][8631].item<bool>());
}

TEST(QwenImage21Structure, RotaryPreservesPairNorm) {
  auto layout =
      qwen_image_21::prepare_sequence_layout(torch::zeros({1, 2}, torch::kBool),
                                             {},
                                             {{1, 2, 2}},
                                             {2, 2, 4},
                                             torch::kCPU);
  auto input = torch::randn({1, 6, 2, 8});
  auto result = qwen_image_21::apply_rotary(input, layout);
  EXPECT_TRUE(torch::allclose(
      input.square().sum(-1), result.square().sum(-1), 1e-5, 1e-5));
}

TEST(QwenImage21Structure, TransformerSyntheticForward) {
  torch::NoGradGuard guard;
  auto config = small_transformer_config();
  QwenImage21Transformer2DModel model(config);
  initialize_weights(*model);
  auto layout =
      model->prepare_layout(torch::tensor({{false, true, false}}, torch::kBool),
                            {},
                            {{1, 2, 2}, {1, 2, 2}},
                            torch::kCPU);
  auto result = model->forward(torch::randn({1, 8, 4}),
                               torch::randn({1, 3, 8}),
                               torch::tensor({0.5f}),
                               layout);
  EXPECT_EQ(result.sizes(), (torch::IntArrayRef{1, 4, 4}));
  EXPECT_TRUE(torch::isfinite(result).all().item<bool>());
}

TEST(QwenImage21Structure, PrefixCannotSeeTargetNoise) {
  torch::NoGradGuard guard;
  auto config = small_transformer_config();
  qwen_image_21::TransformerBlock block(
      config, torch::TensorOptions().dtype(torch::kFloat32));
  initialize_weights(*block);
  auto layout = qwen_image_21::prepare_sequence_layout(
      torch::tensor({{false, true, false}}, torch::kBool),
      {},
      {{1, 2, 2}, {1, 2, 2}},
      {2, 2, 4},
      torch::kCPU);
  auto input = torch::randn({1, 10, 16});
  auto altered = input.clone();
  altered.slice(1, layout.prefix_tokens).normal_(2, 3);
  auto modulation = torch::randn({2, 64});
  auto a = block->forward(input, modulation, layout);
  auto b = block->forward(altered, modulation, layout);
  EXPECT_TRUE(torch::equal(a.slice(1, 0, layout.prefix_tokens),
                           b.slice(1, 0, layout.prefix_tokens)));
}

TEST(QwenImage21Structure, StaticTemporalShortcutSemantics) {
  auto input = torch::ones({1, 2, 1, 2, 2});
  auto down = qwen_image_21::average_downsample(input, 2, 2, 2);
  EXPECT_EQ(down.sizes(), (torch::IntArrayRef{1, 2, 1, 1, 1}));
  EXPECT_TRUE(torch::equal(down, torch::full_like(down, 0.5)));
  auto up = qwen_image_21::duplicate_upsample(down, 2, 2, 2);
  EXPECT_EQ(up.sizes(), input.sizes());
  EXPECT_TRUE(torch::equal(up, torch::full_like(input, 0.5)));
}

TEST(QwenImage21Structure, VaeSyntheticEncodeDecodeRgba) {
  torch::NoGradGuard guard;
  QwenImage21VaeConfig config;
  config.base_dim = 2;
  config.decoder_base_dim = 4;
  config.z_dim = 2;
  config.num_res_blocks = 1;
  config.dim_mult = {1, 2, 2};
  config.temperal_downsample = {false, true};
  config.latents_mean = {0.5, -0.5};
  config.latents_std = {2.0, 3.0};
  AutoencoderKLQwenImage21 model(ModelContext{}, config);
  initialize_weights(*model);
  auto encoded = model->encode_mode(torch::randn({1, 4, 1, 8, 12}));
  EXPECT_EQ(encoded.sizes(), (torch::IntArrayRef{1, 2, 1, 2, 3}));
  auto decoded = model->decode(encoded);
  EXPECT_EQ(decoded.sizes(), (torch::IntArrayRef{1, 4, 1, 8, 12}));
  EXPECT_TRUE(torch::isfinite(decoded).all().item<bool>());
  EXPECT_LE(decoded.abs().max().item<float>(), 1.0);
  EXPECT_TRUE(torch::allclose(
      encoded,
      model->denormalize_latents(model->normalize_latents(encoded)),
      1e-5,
      1e-6));
}

TEST(QwenImage21Structure, LayerWeightLoadingChecksMissingDuplicateAndShape) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  qwen_image_21::RmsNorm norm(3,
                              1e-6,
                              /*zero_centered=*/false,
                              torch::TensorOptions().dtype(torch::kFloat32));
  // An unrelated shard must not mark the layer loaded.
  norm->load_state_dict(StateDict({}));
  EXPECT_DEATH(norm->verify_loaded_weights("norm."), "norm.weight");
  EXPECT_DEATH(
      norm->load_state_dict(StateDict({{"weight", torch::zeros({2})}})),
      "weight size mismatch");
  norm->load_state_dict(StateDict({{"weight", torch::ones({3})}}));
  norm->verify_loaded_weights("norm.");
  EXPECT_DEATH(
      norm->load_state_dict(StateDict({{"weight", torch::zeros({3})}})),
      "weight already loaded");
}

template <typename Model>
void check_sharded_layer_loading(Model& model) {
  auto parameters = model->named_parameters();
  std::unordered_map<std::string, torch::Tensor> first, second;
  first.reserve(parameters.size());
  second.reserve(parameters.size());
  int64_t index = 0;
  for (const auto& parameter : parameters) {
    auto& shard = index++ % 2 == 0 ? first : second;
    shard.emplace("component." + parameter.key(),
                  torch::randn(parameter.value().sizes(), torch::kFloat64));
  }
  // Match EditPlus: unused checkpoint keys are ignored by layer loaders.
  first.emplace("component.unused", torch::zeros({1}));
  StateDict first_shard(std::move(first));
  StateDict second_shard(std::move(second));
  model->load_state_dict(first_shard.get_dict_with_prefix("component."));
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(model->verify_loaded_weights("component."), "is not loaded for");
  model->load_state_dict(second_shard.get_dict_with_prefix("component."));
  model->verify_loaded_weights("component.");
  EXPECT_DEATH(
      model->load_state_dict(first_shard.get_dict_with_prefix("component.")),
      "weight already loaded");
  auto loaded = model->named_parameters();
  for (const auto& parameter : loaded) {
    auto value = first_shard.get_tensor("component." + parameter.key());
    if (!value.defined()) {
      value = second_shard.get_tensor("component." + parameter.key());
    }
    ASSERT_TRUE(value.defined()) << parameter.key();
    EXPECT_EQ(parameter.value().data_ptr(),
              parameters[parameter.key()].data_ptr());
    EXPECT_TRUE(
        torch::equal(parameter.value(), value.to(parameter.value().options())))
        << parameter.key();
  }
}

TEST(QwenImage21Structure, TransformerLoadsWeightsWithNestedPrefixes) {
  QwenImage21Transformer2DModel model(small_transformer_config());
  check_sharded_layer_loading(model);
}

TEST(QwenImage21Structure, VaeLoadsWeightsWithNestedPrefixes) {
  QwenImage21VaeConfig config;
  config.base_dim = 4;
  config.decoder_base_dim = 4;
  config.z_dim = 2;
  config.num_res_blocks = 1;
  config.dim_mult = {1, 2, 2};
  config.temperal_downsample = {false, true};
  AutoencoderKLQwenImage21 model(ModelContext{}, config);
  check_sharded_layer_loading(model);
}

TEST(QwenImage21Structure, SharedLinearMatchesBiasFreeFunctionalLinear) {
  for (const auto dtype : {torch::kFloat32, torch::kBFloat16}) {
    layer::AddMatmul linear(
        4, 3, /*with_bias=*/false, torch::TensorOptions().dtype(dtype));
    auto weight = torch::randn({3, 4}).to(dtype);
    linear->load_state_dict(StateDict({{"weight", weight}}));
    linear->verify_loaded_weights("linear.");
    EXPECT_EQ(linear->named_parameters().size(), 1);
    EXPECT_FALSE(linear->named_parameters().contains("bias"));
    auto input = torch::randn({2, 5, 4}).to(dtype);
    for (const auto& value : {input, input.transpose(0, 1), input[0]}) {
      EXPECT_TRUE(torch::equal(linear->forward(value),
                               torch::nn::functional::linear(value, weight)));
    }
  }
}

TEST(QwenImage21Structure, RejectInconsistentImageSlotCount) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      qwen_image_21::prepare_sequence_layout(torch::zeros({1, 2}, torch::kBool),
                                             {},
                                             {{1, 2, 2}, {1, 2, 2}},
                                             {2, 2, 4},
                                             torch::kCPU),
      "VLM image slots");
}

}  // namespace
}  // namespace xllm

namespace xllm {
namespace {

void check_prefix_cache_case(
    const torch::Tensor& image_mask,
    const torch::Tensor& valid_mask,
    const std::vector<qwen_image_21::ImageShape>& shapes) {
  torch::NoGradGuard guard;
  auto config = small_transformer_config();
  QwenImage21Transformer2DModel model(config);
  initialize_weights(*model);
  auto layout =
      model->prepare_layout(image_mask, valid_mask, shapes, torch::kCPU);
  auto prompt = torch::randn({1, image_mask.size(1), config.context_in_dim});
  auto full =
      torch::randn({1, layout.image_positions.numel(), config.in_channels});
  qwen_image_21::PrefixKVCache cache(config.num_layers);
  auto timestep = torch::tensor({1.0f});
  auto uncached = model->forward(full, prompt, timestep, layout);
  auto prefill = model->forward(full, prompt, timestep, layout, &cache);
  EXPECT_TRUE(torch::equal(uncached, prefill));
  ASSERT_TRUE(cache.ready());
  const int64_t expected_bytes = config.num_layers * 2 * layout.prefix_tokens *
                                 config.num_heads * config.head_dim *
                                 sizeof(float);
  EXPECT_EQ(cache.bytes(), expected_bytes);
  EXPECT_EQ(cache.decode_layout().allowed_attention.defined(),
            layout.has_padding);
  EXPECT_EQ(cache.decode_layout().rope_cos.size(0), layout.target_tokens);
  auto saved_key = cache.layer(0).key.clone();
  for (int64_t index = 0; index < config.num_layers; ++index) {
    const auto& kv = cache.layer(index);
    EXPECT_TRUE(kv.key.is_contiguous());
    EXPECT_TRUE(kv.value.is_contiguous());
    EXPECT_EQ(kv.key.storage_offset(), 0);
    EXPECT_EQ(kv.key.storage().nbytes(),
              kv.key.numel() * kv.key.element_size());
  }
  for (float time : {0.7f, 0.4f, 0.05f}) {
    auto target = torch::randn({1, layout.target_tokens, config.in_channels});
    full.slice(1, full.size(1) - layout.target_tokens).copy_(target);
    auto t = torch::tensor({time});
    auto expected = model->forward(full, prompt, t, layout);
    auto actual = model->forward(target, prompt, t, layout, &cache);
    EXPECT_TRUE(torch::allclose(actual, expected, 1e-4, 1e-6))
        << "max abs=" << (actual - expected).abs().max().item<float>();
    EXPECT_TRUE(torch::equal(saved_key, cache.layer(0).key));
  }
  qwen_image_21::PrefixKVCache fresh(config.num_layers);
  EXPECT_FALSE(fresh.ready());
  EXPECT_EQ(fresh.bytes(), 0);
}

TEST(QwenImage21PrefixCache, TextOnlyMatchesUncachedAcrossSteps) {
  check_prefix_cache_case(torch::zeros({1, 3}, torch::kBool), {}, {{1, 2, 2}});
}

TEST(QwenImage21PrefixCache, AdjacentConditionImagesMatchAcrossSteps) {
  check_prefix_cache_case(
      torch::tensor({{false, true, true, false}}, torch::kBool),
      {},
      {{1, 2, 2}, {1, 2, 2}, {1, 2, 2}});
}

TEST(QwenImage21PrefixCache, PaddingKeysRemainMaskedDuringDecode) {
  check_prefix_cache_case(
      torch::tensor({{false, true, false, false}}, torch::kBool),
      torch::tensor({{1, 1, 1, 0}}, torch::kLong),
      {{1, 2, 2}, {1, 2, 2}});
}

TEST(QwenImage21PrefixCache, CfgBranchesHaveIndependentStorage) {
  torch::NoGradGuard guard;
  auto config = small_transformer_config();
  QwenImage21Transformer2DModel model(config);
  initialize_weights(*model);
  auto positive_layout = model->prepare_layout(
      torch::zeros({1, 3}, torch::kBool), {}, {{1, 2, 2}}, torch::kCPU);
  auto negative_layout = model->prepare_layout(
      torch::zeros({1, 5}, torch::kBool), {}, {{1, 2, 2}}, torch::kCPU);
  auto positive = torch::randn({1, 3, config.context_in_dim});
  auto negative = torch::randn({1, 5, config.context_in_dim});
  auto target = torch::randn({1, 4, config.in_channels});
  qwen_image_21::PrefixKVCache cond(config.num_layers),
      uncond(config.num_layers);
  model->forward(
      target, positive, torch::tensor({1.0f}), positive_layout, &cond);
  auto original = cond.layer(0).key.clone();
  model->forward(
      target, negative, torch::tensor({1.0f}), negative_layout, &uncond);
  EXPECT_TRUE(torch::equal(original, cond.layer(0).key));
  EXPECT_NE(cond.layer(0).key.data_ptr(), uncond.layer(0).key.data_ptr());
  EXPECT_EQ(cond.layer(0).key.size(2), 3);
  EXPECT_EQ(uncond.layer(0).key.size(2), 5);
  auto time = torch::tensor({0.25f});
  auto cached =
      model->forward(target, negative, time, negative_layout, &uncond);
  auto expected = model->forward(target, negative, time, negative_layout);
  EXPECT_TRUE(torch::allclose(cached, expected, 1e-4, 1e-6));
}

TEST(QwenImage21PrefixCache, RejectsUnsafeConfigurationAndBranchReuse) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  torch::NoGradGuard guard;
  auto config = small_transformer_config();
  QwenImage21Transformer2DModel model(config);
  initialize_weights(*model);
  auto layout = model->prepare_layout(
      torch::zeros({1, 3}, torch::kBool), {}, {{1, 2, 2}}, torch::kCPU);
  auto prompt = torch::randn({1, 3, config.context_in_dim});
  auto target = torch::randn({1, 4, config.in_channels});
  qwen_image_21::PrefixKVCache cache(config.num_layers);
  model->forward(target, prompt, torch::tensor({1.0f}), layout, &cache);
  EXPECT_DEATH(
      model->forward(
          target, prompt.clone(), torch::tensor({0.5f}), layout, &cache),
      "prompt/CFG branch changed");
  for (bool disable_condition : {true, false}) {
    auto unsafe = config;
    unsafe.causal_condition = !disable_condition;
    unsafe.causal_block = disable_condition;
    QwenImage21Transformer2DModel invalid(unsafe);
    EXPECT_DEATH(
        invalid->forward(target, prompt, torch::tensor({0.5f}), layout, &cache),
        "requires causal_condition and causal_block");
  }
}

}  // namespace
}  // namespace xllm

namespace xllm {
namespace {
TEST(QwenImage21Sp, TargetShardsReassembleAndPrefixIsReplicated) {
  auto layout = qwen_image_21::prepare_sequence_layout(
      torch::tensor({{false, true, false}}, torch::kBool),
      {},
      {{1, 2, 2}, {1, 2, 6}},
      {2, 2, 4},
      torch::kCPU);
  auto joint = torch::arange((layout.prefix_tokens + 12) * 8, torch::kFloat32)
                   .reshape({1, -1, 8});
  std::vector<torch::Tensor> targets;
  for (int32_t rank = 0; rank < 2; ++rank) {
    qwen_image_21::SpSequenceLayout sp(layout, rank, 2);
    auto local = sp.split_joint(joint);
    EXPECT_EQ(local.size(1), layout.prefix_tokens + 6);
    EXPECT_TRUE(torch::equal(local.slice(1, 0, layout.prefix_tokens),
                             joint.slice(1, 0, layout.prefix_tokens)));
    targets.emplace_back(local.slice(1, layout.prefix_tokens));
    EXPECT_EQ(sp.local(false).target_mask.sum().item<int64_t>(), 6);
    EXPECT_TRUE(torch::equal(
        sp.local(true).rope_cos,
        layout.rope_cos.slice(0,
                              layout.prefix_tokens + rank * 6,
                              layout.prefix_tokens + (rank + 1) * 6)));
    EXPECT_FALSE(sp.attention_mask(true).defined());
  }
  EXPECT_TRUE(torch::equal(torch::cat(targets, 1),
                           joint.slice(1, layout.prefix_tokens)));
}

TEST(QwenImage21Sp, PaddingKeysMaskedAndPaddedQueryRowsValid) {
  auto full = qwen_image_21::prepare_sequence_layout(
      torch::zeros({1, 3}, torch::kBool),
      torch::tensor({{1, 1, 0}}, torch::kLong),
      {{1, 2, 6}},
      {2, 2, 4},
      torch::kCPU);
  std::vector<torch::Tensor> outputs;
  auto target = torch::arange(12, torch::kFloat32).reshape({1, 12, 1});
  for (int32_t rank = 0; rank < 5; ++rank) {
    qwen_image_21::SpSequenceLayout sp(full, rank, 5);
    EXPECT_EQ(sp.padding(), 3);
    EXPECT_EQ(sp.shard_tokens(), 3);
    EXPECT_EQ(sp.attention_mask(false).sizes(),
              (torch::IntArrayRef{1, 1, 18, 18}));
    EXPECT_FALSE(sp.attention_mask(true).select(3, 2).any().item<bool>());
    EXPECT_FALSE(sp.attention_mask(true).slice(3, 15).any().item<bool>());
    EXPECT_TRUE(sp.attention_mask(true)
                    .slice(2, 12)
                    .slice(3, 3, 15)
                    .all()
                    .item<bool>());
    EXPECT_EQ(sp.local(true).attention_query_offset, 3);
    outputs.emplace_back(sp.shard(target, 1));
    if (rank == 4) {
      EXPECT_TRUE(sp.local(true).rope_cos.eq(1).all().item<bool>());
      EXPECT_TRUE(sp.local(true).rope_sin.eq(0).all().item<bool>());
    }
  }
  EXPECT_TRUE(torch::equal(torch::cat(outputs, 1).slice(1, 0, 12), target));
}

TEST(QwenImage21Sp, HeadPartitionsRecombineFullAttention) {
  auto layout = qwen_image_21::prepare_sequence_layout(
      torch::tensor({{false, true, false}}, torch::kBool),
      {},
      {{1, 2, 2}, {1, 2, 2}},
      {2, 2, 4},
      torch::kCPU);
  const int64_t total = layout.prefix_tokens + layout.target_tokens;
  auto q = torch::randn({1, 4, total, 8}), k = torch::randn_like(q),
       v = torch::randn_like(q);
  auto full =
      qwen_image_21::transformer_attention(q, k, v, layout.allowed_attention);
  std::vector<torch::Tensor> per_head;
  for (int64_t rank = 0; rank < 2; ++rank) {
    per_head.emplace_back(qwen_image_21::transformer_attention(
        q.slice(1, 2 * rank, 2 * (rank + 1)),
        k.slice(1, 2 * rank, 2 * (rank + 1)),
        v.slice(1, 2 * rank, 2 * (rank + 1)),
        layout.allowed_attention));
  }
  EXPECT_TRUE(torch::allclose(full, torch::cat(per_head, 1), 1e-6, 1e-6));
}
}  // namespace
}  // namespace xllm

namespace xllm {
namespace {
TEST(QwenImage21SpDegree, AcceptsPositiveDivisorsWithoutAWhitelist) {
  for (int32_t degree : {1, 2, 4, 8, 16, 32}) {
    qwen_image_21::validate_ulysses_head_partition(32, degree);
  }
  // General rule, not a hard-coded power-of-two rule or a 32-head check.
  for (int32_t degree : {1, 2, 3, 4, 6, 8, 12, 24}) {
    qwen_image_21::validate_ulysses_head_partition(24, degree);
  }
}

TEST(QwenImage21SpDegree, RejectsInvalidDegreesBeforeModuloOrCommunication) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(qwen_image_21::validate_ulysses_head_partition(32, 0),
               "sp_size must be positive");
  EXPECT_DEATH(qwen_image_21::validate_ulysses_head_partition(32, -2),
               "sp_size must be positive");
  EXPECT_DEATH(qwen_image_21::validate_ulysses_head_partition(0, 2),
               "num_heads must be positive");
  for (int32_t degree : {3, 5, 6, 7, 31, 33, 64}) {
    EXPECT_DEATH(qwen_image_21::validate_ulysses_head_partition(32, degree),
                 "num_heads divisible by sp_size");
  }
}

TEST(QwenImage21SpDegree, HigherDegreesPreservePrefixRopeAndPadding) {
  // Twelve target tokens forces padding for SP8/16/32, including ranks whose
  // shards consist entirely of padding. No rank can expose dummy keys.
  auto full = qwen_image_21::prepare_sequence_layout(
      torch::tensor({{false, true, false, false}}, torch::kBool),
      torch::tensor({{1, 1, 1, 0}}, torch::kLong),
      {{1, 2, 2}, {1, 2, 6}},
      {2, 2, 4},
      torch::kCPU);
  auto joint = torch::arange((full.prefix_tokens + 12) * 8, torch::kFloat32)
                   .reshape({1, -1, 8});
  for (int32_t degree : {2, 4, 8, 16, 32}) {
    qwen_image_21::validate_ulysses_head_partition(32, degree);
    std::vector<torch::Tensor> targets, cosines, sines;
    targets.reserve(degree);
    cosines.reserve(degree);
    sines.reserve(degree);
    for (int32_t rank = 0; rank < degree; ++rank) {
      qwen_image_21::SpSequenceLayout sp(full, rank, degree);
      auto local = sp.split_joint(joint);
      EXPECT_EQ(local.size(1), full.prefix_tokens + sp.shard_tokens());
      EXPECT_TRUE(torch::equal(local.slice(1, 0, full.prefix_tokens),
                               joint.slice(1, 0, full.prefix_tokens)));
      targets.emplace_back(local.slice(1, full.prefix_tokens));
      cosines.emplace_back(sp.local(true).rope_cos);
      sines.emplace_back(sp.local(true).rope_sin);
      EXPECT_EQ(sp.local(true).attention_query_offset, full.prefix_tokens);
      EXPECT_FALSE(sp.attention_mask(true)
                       .select(3, full.prefix_tokens - 1)
                       .any()
                       .item<bool>());
      EXPECT_TRUE(sp.attention_mask(true).any(-1).all().item<bool>());
      if (sp.padding() > 0) {
        EXPECT_FALSE(sp.attention_mask(true)
                         .slice(3, full.prefix_tokens + 12)
                         .any()
                         .item<bool>());
      }
    }
    auto target = torch::cat(targets, 1);
    EXPECT_TRUE(torch::equal(target.slice(1, 0, 12),
                             joint.slice(1, full.prefix_tokens)));
    EXPECT_TRUE(target.slice(1, 12).eq(0).all().item<bool>());
    auto cosine = torch::cat(cosines, 0);
    auto sine = torch::cat(sines, 0);
    EXPECT_TRUE(torch::equal(cosine.slice(0, 0, 12),
                             full.rope_cos.slice(0, full.prefix_tokens)));
    EXPECT_TRUE(torch::equal(sine.slice(0, 0, 12),
                             full.rope_sin.slice(0, full.prefix_tokens)));
    EXPECT_TRUE(cosine.slice(0, 12).eq(1).all().item<bool>());
    EXPECT_TRUE(sine.slice(0, 12).eq(0).all().item<bool>());
  }
}

TEST(QwenImage21SpDegree, AllHeadPartitionsRecombinePaddedAttention) {
  torch::manual_seed(21);
  auto full = qwen_image_21::prepare_sequence_layout(
      torch::zeros({1, 3}, torch::kBool),
      torch::tensor({{1, 1, 0}}, torch::kLong),
      {{1, 2, 6}},
      {2, 2, 4},
      torch::kCPU);
  const int64_t total = full.prefix_tokens + full.target_tokens;
  auto query = torch::randn({1, 32, total, 8});
  auto key = torch::randn_like(query);
  auto value = torch::randn_like(query);
  auto expected = qwen_image_21::transformer_attention(
      query, key, value, full.allowed_attention);
  for (int32_t degree : {2, 4, 8, 16, 32}) {
    qwen_image_21::SpSequenceLayout geometry(full, 0, degree);
    const int64_t padding = geometry.padding();
    auto pad = [&](const torch::Tensor& input) {
      return torch::cat(
          {input, torch::zeros({1, 32, padding, 8}, input.options())}, 2);
    };
    auto q = pad(query), k = pad(key), v = pad(value);
    std::vector<torch::Tensor> parts, decode_parts;
    parts.reserve(degree);
    decode_parts.reserve(degree);
    const int64_t heads = 32 / degree;
    for (int32_t rank = 0; rank < degree; ++rank) {
      auto hq = q.slice(1, rank * heads, (rank + 1) * heads);
      auto hk = k.slice(1, rank * heads, (rank + 1) * heads);
      auto hv = v.slice(1, rank * heads, (rank + 1) * heads);
      parts.emplace_back(qwen_image_21::transformer_attention(
          hq, hk, hv, geometry.attention_mask(false)));
      decode_parts.emplace_back(
          qwen_image_21::transformer_attention(hq.slice(2, full.prefix_tokens),
                                               hk,
                                               hv,
                                               geometry.attention_mask(true)));
    }
    EXPECT_TRUE(torch::allclose(
        torch::cat(parts, 1).slice(2, 0, total), expected, 1e-6, 1e-6));
    EXPECT_TRUE(torch::allclose(torch::cat(decode_parts, 1).slice(2, 0, 12),
                                expected.slice(2, full.prefix_tokens),
                                1e-6,
                                1e-6));
  }
}
}  // namespace
}  // namespace xllm

namespace xllm {
namespace {
TaylorSeerOptions taylor_options(int32_t warmup,
                                 int32_t interval,
                                 int32_t order) {
  TaylorSeerOptions result;
  result.warmup_steps = warmup;
  result.skip_interval_steps = interval;
  result.n_derivatives = order;
  return result;
}

TEST(QwenImage21TaylorSeer, PrefixPrefillIsExcludedAndFirstDecodeForcedFull) {
  qwen_image_21::RequestTaylorSeer state(taylor_options(0, 2, 1), 6);
  EXPECT_FALSE(state.begin_step(1, true));
  state.record(torch::ones({1, 12, 8}));
  EXPECT_EQ(state.bytes(), 0);
  EXPECT_FALSE(state.begin_step(2, false));
  state.record(torch::full({1, 4, 8}, 2.0));
  EXPECT_TRUE(state.begin_step(3, false));
  EXPECT_TRUE(torch::equal(state.predict(torch::zeros({1, 4, 8})),
                           torch::full({1, 4, 8}, 2.0)));
  EXPECT_FALSE(state.begin_step(4, false));
  state.record(torch::full({1, 4, 8}, 4.0));
  EXPECT_TRUE(state.begin_step(5, false));
  EXPECT_TRUE(torch::equal(state.predict(torch::zeros({1, 4, 8})),
                           torch::full({1, 4, 8}, 5.0)));
  EXPECT_FALSE(state.begin_step(6, false));
  state.record(torch::full({1, 4, 8}, 6.0));
  EXPECT_EQ(state.full_steps(), 4);
  EXPECT_EQ(state.hit_steps(), 2);
}

TEST(QwenImage21TaylorSeer, FiniteDifferencesAndFactorialScaling) {
  qwen_image_21::RequestTaylorSeer state(taylor_options(3, 2, 2), 5);
  for (int64_t step = 1; step <= 3; ++step) {
    EXPECT_FALSE(state.begin_step(step, false));
    state.record(torch::full({1}, static_cast<float>(step * step)));
  }
  EXPECT_EQ(state.available_order(), 2);
  EXPECT_TRUE(state.begin_step(4, false));
  // Backward finite differences: Y=9, D1=5, D2=2 -> 9+5+2/2=15.
  EXPECT_FLOAT_EQ(state.predict(torch::zeros({1})).item<float>(), 15);
  EXPECT_FALSE(state.begin_step(5, false));
  state.record(torch::full({1}, 25.0f));
}

TEST(QwenImage21TaylorSeer, ScheduleUsesActualStepsAndIsRankIndependent) {
  for (bool prefix : {false, true}) {
    qwen_image_21::RequestTaylorSeer a(taylor_options(8, 2, 1), 50);
    qwen_image_21::RequestTaylorSeer b(taylor_options(8, 2, 1), 50);
    for (int64_t step = 1; step <= 50; ++step) {
      const bool hit = a.begin_step(step, prefix && step == 1);
      EXPECT_EQ(hit, b.begin_step(step, prefix && step == 1));
      if (hit) {
        a.predict(torch::zeros({1, 5, 8}));
        b.predict(torch::ones({1, 7, 8}));
      } else {
        a.record(torch::full({1, 5, 8}, static_cast<float>(step)));
        b.record(torch::full({1, 7, 8}, static_cast<float>(step * 2)));
      }
    }
    EXPECT_EQ(a.full_steps(), 29);
    EXPECT_EQ(a.hit_steps(), 21);
  }
}

TEST(QwenImage21TaylorSeer, ShortRequestsAndDisabledSkippingAlwaysCompute) {
  for (int64_t total : {2, 3, 8}) {
    qwen_image_21::RequestTaylorSeer state(taylor_options(8, 2, 1), total);
    for (int64_t step = 1; step <= total; ++step) {
      EXPECT_FALSE(state.begin_step(step, step == 1));
      state.record(torch::zeros({1, 4, 8}));
    }
    EXPECT_EQ(state.hit_steps(), 0);
  }
  qwen_image_21::RequestTaylorSeer state(taylor_options(0, 1, 4), 10);
  for (int64_t step = 1; step <= 10; ++step) {
    EXPECT_FALSE(state.begin_step(step, false));
    state.record(torch::ones({1}));
  }
  EXPECT_EQ(state.hit_steps(), 0);
}

TEST(QwenImage21TaylorSeer, OwnsHistoryAndBranchesDoNotLeak) {
  qwen_image_21::RequestTaylorSeer positive(taylor_options(0, 3, 0), 4);
  qwen_image_21::RequestTaylorSeer negative(taylor_options(0, 3, 0), 4);
  auto output = torch::ones({1, 4, 8}, torch::kBFloat16);
  EXPECT_FALSE(positive.begin_step(1, false));
  positive.record(output);
  output.fill_(7);
  EXPECT_FALSE(negative.begin_step(1, false));
  negative.record(output);
  EXPECT_TRUE(positive.begin_step(2, false));
  EXPECT_TRUE(negative.begin_step(2, false));
  EXPECT_TRUE(positive.predict(output).eq(1).all().item<bool>());
  EXPECT_TRUE(negative.predict(output).eq(7).all().item<bool>());
  EXPECT_EQ(positive.bytes(), 1 * 4 * 8 * sizeof(float));
  qwen_image_21::RequestTaylorSeer fresh(taylor_options(0, 3, 0), 4);
  EXPECT_EQ(fresh.bytes(), 0);
  EXPECT_FALSE(fresh.begin_step(1, false));
}

TEST(QwenImage21TaylorSeer, InvalidOptionsAndSequencingAreRejected) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(qwen_image_21::RequestTaylorSeer(taylor_options(0, 0, 1), 4),
               "interval must be positive");
  EXPECT_DEATH(qwen_image_21::RequestTaylorSeer(taylor_options(-1, 2, 1), 4),
               "warmup_steps");
  EXPECT_DEATH(qwen_image_21::RequestTaylorSeer(taylor_options(0, 2, -1), 4),
               "n_derivatives");
  EXPECT_DEATH(qwen_image_21::RequestTaylorSeer(taylor_options(0, 2, 5), 4),
               "order 0..4");
  qwen_image_21::RequestTaylorSeer state(taylor_options(0, 2, 1), 5);
  EXPECT_DEATH(state.begin_step(2, false), "consecutive");
  state.begin_step(1, false);
  EXPECT_DEATH(state.begin_step(2, false), "not recorded");
  state.record(torch::zeros({1, 4, 8}));
  state.begin_step(2, false);
  EXPECT_DEATH(state.predict(torch::zeros({1, 8, 8})), "shape changed");
}

TEST(QwenImage21TaylorSeer,
     NoHitTransformerPathMatchesBaselineWithAndWithoutKv) {
  torch::NoGradGuard guard;
  auto config = small_transformer_config();
  QwenImage21Transformer2DModel model(config);
  initialize_weights(*model);
  auto layout =
      model->prepare_layout(torch::tensor({{false, true, false}}, torch::kBool),
                            {},
                            {{1, 2, 2}, {1, 2, 2}},
                            torch::kCPU);
  auto prompt = torch::randn({1, 3, config.context_in_dim});
  auto full = torch::randn({1, 8, config.in_channels});
  for (bool use_kv : {false, true}) {
    qwen_image_21::PrefixKVCache kv(config.num_layers);
    qwen_image_21::RequestTaylorSeer ts(taylor_options(0, 1, 1), 4);
    for (int64_t step = 1; step <= 4; ++step) {
      auto t = torch::tensor({1.0f / step});
      auto expected = model->forward(full, prompt, t, layout);
      auto input = use_kv && kv.ready() ? full.slice(1, 4) : full;
      auto actual = model->forward(
          input, prompt, t, layout, use_kv ? &kv : nullptr, nullptr, &ts, step);
      EXPECT_TRUE(torch::allclose(actual, expected, 1e-5, 1e-6));
    }
    EXPECT_EQ(ts.hit_steps(), 0);
  }
}

TEST(QwenImage21TaylorSeer, TransformerHitsRemainFiniteAndPreservePrefixKv) {
  torch::NoGradGuard guard;
  auto config = small_transformer_config();
  QwenImage21Transformer2DModel model(config);
  initialize_weights(*model);
  auto layout = model->prepare_layout(
      torch::zeros({1, 3}, torch::kBool), {}, {{1, 2, 2}}, torch::kCPU);
  auto prompt = torch::randn({1, 3, config.context_in_dim});
  auto target = torch::randn({1, 4, config.in_channels});
  qwen_image_21::PrefixKVCache kv(config.num_layers);
  qwen_image_21::RequestTaylorSeer ts(taylor_options(2, 2, 1), 6);
  torch::Tensor saved;
  for (int64_t step = 1; step <= 6; ++step) {
    auto out = model->forward(target,
                              prompt,
                              torch::tensor({1.0f / step}),
                              layout,
                              &kv,
                              nullptr,
                              &ts,
                              step);
    EXPECT_TRUE(torch::isfinite(out).all().item<bool>());
    EXPECT_EQ(out.sizes(), target.sizes());
    if (step == 1) {
      saved = kv.layer(0).key.clone();
    }
    EXPECT_TRUE(torch::equal(saved, kv.layer(0).key));
  }
  EXPECT_EQ(ts.full_steps(), 4);
  EXPECT_EQ(ts.hit_steps(), 2);
}
}  // namespace
}  // namespace xllm
