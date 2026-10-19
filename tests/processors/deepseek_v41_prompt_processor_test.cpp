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

#include "processors/deepseek_v41_prompt_processor.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/tokenizer/tokenizer.h"

namespace xllm {
namespace {

constexpr int32_t kImageTokenId = 129264;
constexpr int32_t kImagePadId = 129265;

// The processor injects the shared protocol spellings (see
// core/common/constants.h); reference the same constants so the test cannot
// drift from production.
const std::string kImageToken = kDeepseekV41ImagePlaceholder;
const std::string kImagePadToken = kDeepseekV41ImagePadPlaceholder;

DeepseekV41PromptProcessor make_processor() {
  ModelArgs args;
  args.image_token_id(kImageTokenId);
  return DeepseekV41PromptProcessor(args);
}

MMData make_grid_mm_data(std::initializer_list<std::vector<int64_t>> grids) {
  MMData data;
  for (const auto& grid : grids) {
    const torch::Tensor grid_tensor =
        torch::tensor(grid, torch::dtype(torch::kInt64)).reshape({1, 2});
    data.add(
        MMType::IMAGE,
        MMDataItem(MMType::IMAGE, MMDict{{"image_grid_thw", grid_tensor}}));
  }
  return data;
}

// Extracts the per-image block role mask from a processed prompt: one entry
// per expanded placeholder token, 'S' for the sentinel (129264) and 'P' for
// the compressor-alignment pad (129265).
std::string extract_role_mask(const std::string& prompt) {
  const std::string image_token(kImageToken);
  const std::string image_pad(kImagePadToken);
  std::string mask;
  size_t i = 0;
  while (i < prompt.size()) {
    if (prompt.compare(i, image_token.size(), image_token) == 0) {
      mask.push_back('S');
      i += image_token.size();
    } else if (prompt.compare(i, image_pad.size(), image_pad) == 0) {
      mask.push_back('P');
      i += image_pad.size();
    } else {
      ++i;
    }
  }
  return mask;
}

std::vector<int32_t> block_tokens(const std::string& mask) {
  std::vector<int32_t> tokens;
  tokens.reserve(mask.size());
  for (const char role : mask) {
    tokens.push_back(role == 'P' ? kImagePadId : kImageTokenId);
  }
  return tokens;
}

// Golden pad/sentinel masks for `image_block_roles`, generated from the
// Python reference `_vision_block_slots` (xllm/python/models/deepseek_v41.py)
// with n_llm = ceil(n_vit / 3). This pins the LOCKED PAIR: any change to the
// layout on either side must regenerate this table (and the Python twin in
// tests/python/test_deepseek_v41_vision_layout.py) in the same commit.
struct LayoutCase {
  int64_t n_vit_h;
  int64_t n_vit_w;
  const char* mask;  // 'S' sentinel, 'P' compressor-alignment pad
};

const LayoutCase kLayoutCases[] = {
    {/* n_vit=(6, 6) n_llm=(2, 2) */ 6, 6, "SSSSSSSPPS"},
    {/* n_vit=(9, 9) n_llm=(3, 3) */ 9, 9, "SSSSSSSSSSPSPSPSPS"},
    {/* n_vit=(3, 3) n_llm=(1, 1) */ 3, 3, "SSPSPS"},
    {/* n_vit=(6, 15) n_llm=(2, 5) */ 6, 15, "SSSSSSSSSSSSSS"},
    {/* n_vit=(7, 8) n_llm=(3, 3) */ 7, 8, "SSSSSSSSSSPSPSPSPS"},
    {/* n_vit=(12, 9) n_llm=(4, 3) */ 12, 9, "SSSSSSSSSSSSSSSSSS"},
    {/* n_vit=(9, 3) n_llm=(3, 1) */ 9, 3, "SSSSSSPSPS"},
    {/* n_vit=(5, 5) n_llm=(2, 2) */ 5, 5, "SSSSSSSPPS"},
};

class DeepseekV41LayoutTest : public ::testing::TestWithParam<LayoutCase> {};

TEST_P(DeepseekV41LayoutTest, ProcessMatchesPythonGoldenMask) {
  const LayoutCase& test_case = GetParam();
  auto processor = make_processor();
  MMData mm_data = make_grid_mm_data({{test_case.n_vit_h, test_case.n_vit_w}});
  std::string prompt = "look at <image> please";
  ASSERT_TRUE(processor.process(prompt, mm_data));
  const std::string mask = extract_role_mask(prompt);
  EXPECT_EQ(mask, test_case.mask);
  // The surrounding text survives the expansion.
  EXPECT_NE(prompt.find("look at "), std::string::npos);
  EXPECT_NE(prompt.find(" please"), std::string::npos);
  // Sentinel count equals the Python-side encode row count and pads are only
  // the compressor-alignment slots.
  const size_t sentinels =
      static_cast<size_t>(std::count(mask.begin(), mask.end(), 'S'));
  const size_t pads =
      static_cast<size_t>(std::count(mask.begin(), mask.end(), 'P'));
  EXPECT_EQ(sentinels + pads, mask.size());
  EXPECT_GT(sentinels, 0u);
}

INSTANTIATE_TEST_SUITE_P(DeepseekV41LayoutCases,
                         DeepseekV41LayoutTest,
                         ::testing::ValuesIn(kLayoutCases));

TEST(DeepseekV41PromptProcessorTest, ProcessRejectsImageWithoutPlaceholder) {
  auto processor = make_processor();
  MMData mm_data = make_grid_mm_data({{6, 6}});
  std::string prompt = "no placeholder here";
  // An image without a matching placeholder would be silently dropped, so the
  // request is rejected and the prompt is left untouched.
  EXPECT_FALSE(processor.process(prompt, mm_data));
  EXPECT_EQ(prompt, "no placeholder here");
}

TEST(DeepseekV41PromptProcessorTest,
     ProcessRejectsFewerPlaceholdersThanImages) {
  auto processor = make_processor();
  MMData mm_data = make_grid_mm_data({{6, 6}, {3, 3}});
  std::string prompt = std::string("only one ") + kImageToken + " here";
  EXPECT_FALSE(processor.process(prompt, mm_data));
}

TEST(DeepseekV41PromptProcessorTest, RestoresEscapedImageLiteral) {
  auto processor = make_processor();
  MMData mm_data = make_grid_mm_data({{6, 6}});
  std::string prompt = std::string("literal ") +
                       kDeepseekV41ImageLiteralSentinel + " and <image>";
  ASSERT_TRUE(processor.process(prompt, mm_data));
  EXPECT_EQ(extract_role_mask(prompt), std::string(kLayoutCases[0].mask));
  EXPECT_NE(prompt.find("literal <image> and "), std::string::npos);
  EXPECT_EQ(prompt.find(kDeepseekV41ImageLiteralSentinel), std::string::npos);
}

TEST(DeepseekV41PromptProcessorTest, RestoresEscapedImageLiteralWithoutImages) {
  auto processor = make_processor();
  MMData mm_data;
  std::string prompt =
      std::string("literal ") + kDeepseekV41ImageLiteralSentinel;
  ASSERT_TRUE(processor.process(prompt, mm_data));
  EXPECT_EQ(prompt, "literal <image>");
}

// A prompt with more placeholders than images (e.g. a caller marker plus a
// template-injected one) must not leave a stray native placeholder: the surplus
// tokenizes to the image sentinel id with no embed row and would crash
// `get_input_embeddings`. The first placeholder is expanded and the rest are
// stripped.
TEST(DeepseekV41PromptProcessorTest, ProcessStripsSurplusPlaceholders) {
  auto processor = make_processor();
  MMData mm_data = make_grid_mm_data({{6, 6}});
  std::string prompt = std::string("x <image> y ") + kImageToken + " z";
  ASSERT_TRUE(processor.process(prompt, mm_data));
  // Exactly one image block; the leftover native placeholder is gone.
  EXPECT_EQ(extract_role_mask(prompt), std::string(kLayoutCases[0].mask));
  EXPECT_NE(prompt.find("x "), std::string::npos);
  EXPECT_NE(prompt.find(" z"), std::string::npos);
}

TEST(DeepseekV41PromptProcessorTest, ProcessExpandsNativeTokenAndMultiple) {
  auto processor = make_processor();
  MMData mm_data = make_grid_mm_data({{6, 6}, {3, 3}});
  std::string prompt = std::string("a ") + kImageToken + " b <image> c";
  ASSERT_TRUE(processor.process(prompt, mm_data));
  const std::string mask = extract_role_mask(prompt);
  // First image (6, 6) then second image (3, 3); the fixed text is intact.
  const std::string mask_6x6 = kLayoutCases[0].mask;
  const std::string mask_3x3 = kLayoutCases[2].mask;
  EXPECT_EQ(mask, mask_6x6 + mask_3x3);
  EXPECT_NE(prompt.find("a "), std::string::npos);
  EXPECT_NE(prompt.find(" b "), std::string::npos);
  EXPECT_NE(prompt.find(" c"), std::string::npos);
}

TEST(DeepseekV41PromptProcessorTest, AdjustTokenLayoutInsertsCompressPads) {
  auto processor = make_processor();
  // grid (6, 6): block mask "SSSSSSSPPS" (10 tokens, 8 sentinels).
  MMData mm_data = make_grid_mm_data({{6, 6}});
  const std::vector<int32_t> block = block_tokens("SSSSSSSPPS");
  std::vector<int32_t> token_ids = {1, 2, 3, 4, 5};
  token_ids.insert(token_ids.end(), block.begin(), block.end());
  token_ids.push_back(6);

  const size_t original_start = 5;  // block begins after 5 text tokens
  const int32_t expected_pads =
      4 - 1 - static_cast<int32_t>(original_start % 4);  // = 2
  ASSERT_TRUE(processor.adjust_token_layout(token_ids, mm_data));

  ASSERT_EQ(token_ids.size(), block.size() + 6 + expected_pads);
  // Text tokens are untouched.
  for (size_t k = 0; k < original_start; ++k) {
    EXPECT_EQ(token_ids[k], static_cast<int32_t>(k + 1));
  }
  EXPECT_EQ(token_ids.back(), 6);
  // Inserted compressor pads sit right before the block.
  for (int32_t k = 0; k < expected_pads; ++k) {
    EXPECT_EQ(token_ids[original_start + k], kImagePadId);
  }
  // The block itself is preserved verbatim and its first sentinel lands on
  // the reference alignment ((start % 4) == COMPRESS_PAD_TO - 1).
  const size_t block_start = original_start + expected_pads;
  EXPECT_EQ(block_start % 4, 3u);
  for (size_t k = 0; k < block.size(); ++k) {
    EXPECT_EQ(token_ids[block_start + k], block[k]);
  }
}

TEST(DeepseekV41PromptProcessorTest, AdjustTokenLayoutAlignsAdjacentBlocks) {
  auto processor = make_processor();
  // Two images with no text between them: the sentinel run is contiguous and
  // must be segmented per image so each block is aligned independently.
  MMData mm_data = make_grid_mm_data({{6, 6}, {3, 3}});
  const std::vector<int32_t> block1 = block_tokens("SSSSSSSPPS");
  const std::vector<int32_t> block2 = block_tokens("SSPSPS");
  std::vector<int32_t> token_ids = {9, 8};
  token_ids.insert(token_ids.end(), block1.begin(), block1.end());
  token_ids.insert(token_ids.end(), block2.begin(), block2.end());
  token_ids.insert(token_ids.end(), {7, 6, 5});

  ASSERT_TRUE(processor.adjust_token_layout(token_ids, mm_data));

  // block1 starts at 2 -> 1 pad; block2 then starts at 2 + 1 + 10 = 13 -> 2
  // pads. Both first sentinels end up at % 4 == 3.
  const size_t text_prefix = 2;
  const int32_t pads1 = 4 - 1 - static_cast<int32_t>(text_prefix % 4);
  const size_t block1_start = text_prefix + pads1;
  const size_t block2_orig_start = block1_start + block1.size();
  const int32_t pads2 = 4 - 1 - static_cast<int32_t>(block2_orig_start % 4);
  const size_t block2_start = block2_orig_start + pads2;

  ASSERT_EQ(token_ids.size(),
            text_prefix + pads1 + block1.size() + pads2 + block2.size() + 3);
  EXPECT_EQ(block1_start % 4, 3u);
  EXPECT_EQ(block2_start % 4, 3u);
  for (size_t k = 0; k < block1.size(); ++k) {
    EXPECT_EQ(token_ids[block1_start + k], block1[k]);
  }
  for (size_t k = 0; k < block2.size(); ++k) {
    EXPECT_EQ(token_ids[block2_start + k], block2[k]);
  }
  EXPECT_EQ(token_ids[0], 9);
  EXPECT_EQ(token_ids[1], 8);
  EXPECT_EQ(token_ids[token_ids.size() - 3], 7);
  EXPECT_EQ(token_ids[token_ids.size() - 2], 6);
  EXPECT_EQ(token_ids[token_ids.size() - 1], 5);
}

TEST(DeepseekV41PromptProcessorTest, AdjustTokenLayoutSkipsPureText) {
  auto processor = make_processor();
  MMData mm_data = make_grid_mm_data({{6, 6}});
  const std::vector<int32_t> token_ids = {1, 2, 3, 4, 5, 6, 7};
  std::vector<int32_t> adjusted = token_ids;
  ASSERT_TRUE(processor.adjust_token_layout(adjusted, mm_data));
  EXPECT_EQ(adjusted, token_ids);
}

TEST(DeepseekV41PromptProcessorTest, FindMMSpansSegmentsAdjacentBlocks) {
  auto processor = make_processor();
  MMData mm_data = make_grid_mm_data({{6, 6}, {3, 3}});
  const std::vector<int32_t> block1 = block_tokens("SSSSSSSPPS");
  const std::vector<int32_t> block2 = block_tokens("SSPSPS");
  std::vector<int32_t> token_ids = {9, 8};
  token_ids.insert(token_ids.end(), block1.begin(), block1.end());
  token_ids.insert(token_ids.end(), block2.begin(), block2.end());
  token_ids.insert(token_ids.end(), {7, 6, 5});
  ASSERT_TRUE(processor.adjust_token_layout(token_ids, mm_data));
  ASSERT_TRUE(processor.find_mm_spans(token_ids, mm_data));

  const auto& items = mm_data.items<MMItemVec>();
  ASSERT_EQ(items.size(), 2u);

  // Item 0: span covers the compressor pads + block 1; only the sentinel
  // positions are masked in.
  const auto& state0 = items[0].state();
  const int32_t start0 = state0.token_pos().offset;
  const int32_t length0 = state0.token_pos().length;
  EXPECT_EQ(start0, 2);  // first compressor pad inserted before block 1
  EXPECT_EQ(length0, 1 + 10);
  EXPECT_EQ(state0.mm_token_num(), 8);
  ASSERT_TRUE(state0.mm_token_mask().defined());
  ASSERT_EQ(state0.mm_token_mask().numel(), length0);
  for (int32_t k = 0; k < length0; ++k) {
    EXPECT_EQ(state0.mm_token_mask()[k].item<bool>(),
              token_ids[start0 + k] == kImageTokenId)
        << "mask mismatch at " << k;
  }

  // Item 1: independently segmented second block.
  const auto& state1 = items[1].state();
  const int32_t start1 = state1.token_pos().offset;
  const int32_t length1 = state1.token_pos().length;
  EXPECT_EQ(start1, 2 + 1 + 10);  // first compressor pad of block 2
  EXPECT_EQ(length1, 2 + 6);
  EXPECT_EQ(state1.mm_token_num(), 4);
  ASSERT_TRUE(state1.mm_token_mask().defined());
  ASSERT_EQ(state1.mm_token_mask().numel(), length1);
  for (int32_t k = 0; k < length1; ++k) {
    EXPECT_EQ(state1.mm_token_mask()[k].item<bool>(),
              token_ids[start1 + k] == kImageTokenId)
        << "mask mismatch at " << k;
  }
}

TEST(DeepseekV41PromptProcessorTest, FindMMSpansSingleImage) {
  auto processor = make_processor();
  MMData mm_data = make_grid_mm_data({{9, 3}});
  const std::vector<int32_t> block = block_tokens("SSSSSSPSPS");
  std::vector<int32_t> token_ids = {1, 2};
  token_ids.insert(token_ids.end(), block.begin(), block.end());
  token_ids.push_back(3);
  ASSERT_TRUE(processor.find_mm_spans(token_ids, mm_data));

  const auto& items = mm_data.items<MMItemVec>();
  ASSERT_EQ(items.size(), 1u);
  const auto& state = items[0].state();
  EXPECT_EQ(state.token_pos().offset, 2);
  EXPECT_EQ(state.token_pos().length, 10);
  EXPECT_EQ(state.mm_token_num(), 8);
}

TEST(DeepseekV41PromptProcessorTest, FindMMSpansRejectsUnassignedItems) {
  auto processor = make_processor();
  // Two items declared, but the token sequence only carries one block: the
  // second image would have no span, so the call must fail.
  MMData mm_data = make_grid_mm_data({{6, 6}, {3, 3}});
  const std::vector<int32_t> block = block_tokens("SSSSSSSPPS");
  std::vector<int32_t> token_ids = {1, 2};
  token_ids.insert(token_ids.end(), block.begin(), block.end());
  token_ids.push_back(3);
  EXPECT_FALSE(processor.find_mm_spans(token_ids, mm_data));
}

TEST(DeepseekV41PromptProcessorTest, FindMMSpansSkipsNonImageItems) {
  auto processor = make_processor();
  // A non-image item before the image must not consume the image's sentinel
  // block: the image still gets its own span and the call succeeds.
  MMData mm_data;
  mm_data.add(MMType::VIDEO, MMDataItem(MMType::VIDEO));
  const torch::Tensor grid_tensor =
      torch::tensor({6, 6}, torch::dtype(torch::kInt64)).reshape({1, 2});
  mm_data.add(
      MMType::IMAGE,
      MMDataItem(MMType::IMAGE, MMDict{{"image_grid_thw", grid_tensor}}));
  const std::vector<int32_t> block = block_tokens("SSSSSSSPPS");
  std::vector<int32_t> token_ids = {1, 2};
  token_ids.insert(token_ids.end(), block.begin(), block.end());
  token_ids.push_back(3);

  ASSERT_TRUE(processor.find_mm_spans(token_ids, mm_data));

  const auto& items = mm_data.items<MMItemVec>();
  ASSERT_EQ(items.size(), 2u);
  EXPECT_EQ(items[0].type(), MMType::VIDEO);
  // The image (item 1) owns the block, not the skipped video item.
  const auto& state = items[1].state();
  EXPECT_EQ(state.token_pos().offset, 2);
  EXPECT_EQ(state.token_pos().length, 10);
  EXPECT_EQ(state.mm_token_num(), 8);
}

TEST(DeepseekV41PromptProcessorTest, ProcessReadsTrailingGridColumns) {
  auto processor = make_processor();
  // A THW-style 3-column grid [t, n_vit_h, n_vit_w]: only the trailing two
  // columns drive the block layout, matching block_sentinel_count/Python.
  MMData mm_data;
  const torch::Tensor thw =
      torch::tensor({1, 6, 6}, torch::dtype(torch::kInt64)).reshape({1, 3});
  mm_data.add(MMType::IMAGE,
              MMDataItem(MMType::IMAGE, MMDict{{"image_grid_thw", thw}}));
  std::string prompt = "look <image>";
  ASSERT_TRUE(processor.process(prompt, mm_data));
  EXPECT_EQ(extract_role_mask(prompt), std::string(kLayoutCases[0].mask));
}

class StubTokenizer final : public Tokenizer {
 public:
  StubTokenizer(int32_t image_id, int32_t pad_id)
      : image_id_(image_id), pad_id_(pad_id) {}
  std::optional<int32_t> token_to_id(
      const std::string_view& token) const override {
    if (token == kImageToken) return image_id_;
    if (token == kImagePadToken) return pad_id_;
    return std::nullopt;
  }

 private:
  int32_t image_id_;
  int32_t pad_id_;
};

TEST(DeepseekV41PromptProcessorTest, TokenizerBindingMatchesConfiguredIds) {
  EXPECT_TRUE(validate_deepseek_v41_tokenizer_binding(
      StubTokenizer(kImageTokenId, kImagePadId), kImageTokenId));
  EXPECT_FALSE(validate_deepseek_v41_tokenizer_binding(
      StubTokenizer(kImageTokenId + 1, kImagePadId), kImageTokenId));
  EXPECT_FALSE(validate_deepseek_v41_tokenizer_binding(
      StubTokenizer(kImageTokenId, kImagePadId + 1), kImageTokenId));
  EXPECT_TRUE(validate_deepseek_v41_tokenizer_binding(StubTokenizer(0, 1), -1));
}

}  // namespace
}  // namespace xllm
