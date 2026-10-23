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
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "models/dit/pipelines/pipeline_qwenimage_21.h"

namespace xllm {
namespace {

TEST(QwenImage21Structure, PackUnpackIsUnpatchedSpatialFlatten) {
  auto input = torch::arange(48, torch::kFloat32).reshape({1, 4, 1, 2, 6});
  auto packed = qwen_image_21::pack_latents(input);
  EXPECT_EQ(packed.sizes(), (torch::IntArrayRef{1, 12, 4}));
  EXPECT_TRUE(torch::equal(qwen_image_21::unpack_latents(packed, 2, 6), input));
  EXPECT_EQ(packed[0][1][2].item<float>(), input[0][2][0][0][1].item<float>());
}

NamedTensor qwen_image_21_test_prompt() {
  return {"prompt_embed",
          torch::zeros({1, 3, 4096}),
          {{"prompt_embeds_mask", std::vector<int64_t>{1, 1, 1}},
           {"image_pad_mask", std::vector<int64_t>{0, 1, 0}},
           {"input_image_sizes", std::vector<int64_t>{32, 32}}}};
}

TEST(QwenImage21Input, SharedReaderHandlesRequestAndPipelineViews) {
  auto source = qwen_image_21_test_prompt();
  QwenImage21PromptInput result;
  EXPECT_TRUE(read_qwen_image_21_prompt(source, {{32, 32}}, result).empty());
  EXPECT_EQ(result.embeddings.sizes(), (torch::IntArrayRef{1, 3, 4096}));
  EXPECT_EQ(result.image_mask.sum().item<int64_t>(), 1);
  EXPECT_EQ(result.valid_mask.scalar_type(), torch::kBool);
  source.tensor = source.tensor.squeeze(0);
  EXPECT_TRUE(read_qwen_image_21_prompt(source, {{32, 32}}, result).empty());
  EXPECT_EQ(result.embeddings.dim(), 3);
}

TEST(QwenImage21Input, RejectsWrongMaskAndImageDimensions) {
  auto source = qwen_image_21_test_prompt();
  QwenImage21PromptInput result;
  EXPECT_FALSE(read_qwen_image_21_prompt(source, {{32, 64}}, result).empty());
  source.parameters["image_pad_mask"] = std::vector<int64_t>{0, 2, 0};
  EXPECT_FALSE(read_qwen_image_21_prompt(source, {{32, 32}}, result).empty());
  source.parameters.erase("image_pad_mask");
  EXPECT_FALSE(read_qwen_image_21_prompt(source, {{32, 32}}, result).empty());
}

TEST(QwenImage21Input, TextOnlyCanOmitMasks) {
  NamedTensor source{"prompt_embed", torch::zeros({1, 2, 4096}), {}};
  QwenImage21PromptInput result;
  EXPECT_TRUE(read_qwen_image_21_prompt(source, {}, result).empty());
  EXPECT_FALSE(result.image_mask.any().item<bool>());
  EXPECT_TRUE(result.valid_mask.all().item<bool>());
  source.tensor[0][0][0] = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(read_qwen_image_21_prompt(source, {}, result).empty());
}

TEST(QwenImage21Input, SharedImageSizeValidationPreservesGeometryLimits) {
  for (const int64_t side : {32, 64, 32768}) {
    EXPECT_TRUE(is_valid_qwen_image_21_image_size(side, /*height=*/32));
    EXPECT_TRUE(is_valid_qwen_image_21_image_size(/*width=*/32, side));
  }
  const std::vector<int64_t> invalid_sides{
      -32, 0, 31, 33, 32769, 32800, std::numeric_limits<int64_t>::max()};
  for (const int64_t side : invalid_sides) {
    EXPECT_FALSE(is_valid_qwen_image_21_image_size(side, /*height=*/32));
    EXPECT_FALSE(is_valid_qwen_image_21_image_size(/*width=*/32, side));
  }
  // The shared rule must not impose the internal VLM processor's aspect ratio
  // on externally encoded images or the output image.
  EXPECT_TRUE(is_valid_qwen_image_21_image_size(/*width=*/32, /*height=*/6432));
}

TEST(QwenImage21Input, GenerationIgnoresCfgRenorm) {
  DiTGenerationParams params;
  for (bool enabled : {false, true}) {
    params.enable_cfg_renorm = enabled;
    params.cfg_renorm_min = 0.5f;
    EXPECT_TRUE(validate_qwen_image_21_generation(params).empty());
  }
}

TEST(QwenImage21Input, ExternalPromptsMayExceedFormerTokenBudget) {
  for (const auto& size :
       {QwenImage21ImageSize{4096, 2048}, QwenImage21ImageSize{8192, 4096}}) {
    const int64_t slots = (size[0] / 32) * (size[1] / 32);
    const int64_t length = slots + 2;
    std::vector<int64_t> image_mask(length, 1);
    image_mask.front() = image_mask.back() = 0;
    // Check both 4x-expanded length > 32768 and raw VLM length > 32768.
    // Use a broadcast view; no large embedding storage or attention mask.
    NamedTensor source{
        "prompt_embed",
        torch::zeros({1, 1, 4096}).expand({1, length, 4096}),
        {{"image_pad_mask", std::move(image_mask)},
         {"input_image_sizes", std::vector<int64_t>{size[0], size[1]}}}};
    QwenImage21PromptInput result;
    const std::string error = read_qwen_image_21_prompt(source, {size}, result);
    ASSERT_TRUE(error.empty()) << error;
    EXPECT_EQ(result.embeddings.size(1), length);
    EXPECT_EQ(result.image_mask.sum().item<int64_t>(), slots);
    EXPECT_GT(length + 3 * slots, 32768);
  }
}

// Fixed encoding of a system prefix followed by text/image/text. Only image
// expansion and validation belong to the helper under test, not tokenization.
class QwenImage21TestTokenizer final : public Tokenizer {
 public:
  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool /*add_special_tokens*/) const override {
    *ids = text == qwen_image_21_system_prefix()
               ? std::vector<int32_t>{1}
               : std::vector<int32_t>{1, 2, 3, 4};
    return true;
  }

  std::optional<int32_t> token_to_id(
      const std::string_view& token) const override {
    return token == "<|image_pad|>" ? std::optional<int32_t>(3) : std::nullopt;
  }
};

TEST(QwenImage21Input, InternalPromptsMayExceedFormerTokenBudget) {
  QwenImage21TestTokenizer tokenizer;
  QwenImage21PromptTokens result;
  const std::string error = prepare_qwen_image_21_prompt_tokens(
      tokenizer, "edit image", {{8192, 4096}}, result);
  ASSERT_TRUE(error.empty()) << error;
  EXPECT_EQ(result.drop_idx, 1);
  EXPECT_EQ(result.ids.size(), 32768 + 3);
  EXPECT_EQ(result.ids.size(), result.image_mask.size());
  EXPECT_EQ(std::count(result.image_mask.begin(), result.image_mask.end(), 1),
            32768);
}

TEST(QwenImage21Input, RemovingTokenBudgetPreservesRopeRangeValidation) {
  NamedTensor source{
      "prompt_embed", torch::zeros({1, 1, 4096}).expand({1, 8192, 4096}), {}};
  QwenImage21PromptInput result;
  EXPECT_NE(read_qwen_image_21_prompt(source, {}, result).find("RoPE"),
            std::string::npos);
}

TEST(QwenImage21Input, GenerationValidatesParametersWithoutTokenBudget) {
  DiTGenerationParams params;
  params.width = 32;
  params.height = 64;
  params.num_inference_steps = 2;
  params.enable_cfg_renorm = false;
  EXPECT_TRUE(validate_qwen_image_21_generation(params).empty());
  params.num_inference_steps = 1;
  EXPECT_FALSE(validate_qwen_image_21_generation(params).empty());
  params.num_inference_steps = 2;
  params.width = 33;
  EXPECT_FALSE(validate_qwen_image_21_generation(params).empty());
  params.width = 4096;
  params.height = 4096;  // 65536 target tokens, above the former 32768 cap.
  EXPECT_TRUE(validate_qwen_image_21_generation(params).empty());
  params.width = 32768 + 32;
  EXPECT_FALSE(validate_qwen_image_21_generation(params).empty());
}

TEST(QwenImage21CfgParallel, BranchRoutingForSingleAndDualRank) {
  for (bool cfg : {false, true}) {
    qwen_image_21::CfgExecutionPlan single(1, 0, cfg);
    EXPECT_TRUE(single.leader());
    EXPECT_TRUE(single.positive());
    EXPECT_EQ(single.negative(), cfg);
    EXPECT_FALSE(single.distributed());
    EXPECT_FALSE(single.idle());
    qwen_image_21::CfgExecutionPlan leader(2, 0, cfg);
    qwen_image_21::CfgExecutionPlan follower(2, 1, cfg);
    EXPECT_TRUE(leader.leader());
    EXPECT_FALSE(follower.leader());
    EXPECT_TRUE(leader.positive());
    EXPECT_FALSE(leader.negative());
    EXPECT_FALSE(follower.positive());
    EXPECT_EQ(follower.negative(), cfg);
    EXPECT_EQ(leader.distributed(), cfg);
    EXPECT_EQ(follower.distributed(), cfg);
    EXPECT_EQ(follower.idle(), !cfg);
  }
}

TEST(QwenImage21CfgParallel, RankOrderedGatherPreservesSerialFormula) {
  torch::manual_seed(21);
  for (auto dtype : {torch::kFloat32, torch::kBFloat16}) {
    auto positive = torch::randn({1, 12, 64}).to(dtype);
    auto negative = torch::randn({1, 12, 64}).to(dtype);
    auto expected = negative + 4.0f * (positive - negative);
    auto result = qwen_image_21::combine_gathered_cfg_noise(
        torch::cat({positive, negative}, 0), 4.0f);
    EXPECT_EQ(result.sizes(), positive.sizes());
    EXPECT_TRUE(torch::equal(result, expected));
  }
}

TEST(QwenImage21CfgParallel, InvalidRoutingRejected) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(qwen_image_21::CfgExecutionPlan(3, 0, true), "cfg_size");
  EXPECT_DEATH(qwen_image_21::CfgExecutionPlan(2, 2, true), "cfg_rank");
  EXPECT_DEATH(
      qwen_image_21::combine_gathered_cfg_noise(torch::zeros({1, 4, 64}), 4.0f),
      "gathered.size");
}

}  // namespace
}  // namespace xllm
