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

#include "core/framework/multimodal/mm_data.h"

#include <gtest/gtest.h>

#include <cstdint>

#include "core/framework/multimodal/mm_batch_data.h"
#include "core/framework/multimodal/mm_visitor.h"

namespace xllm {
namespace {

struct TypeMaskCase {
  uint32_t stored_type;
  uint32_t query_type;
  bool expected;
};

class MMDataTypeMaskTest : public ::testing::TestWithParam<TypeMaskCase> {};

TEST_P(MMDataTypeMaskTest, ChecksDataTypes) {
  const TypeMaskCase& test_case = GetParam();
  MMData data;
  data.set(test_case.stored_type, MMDict{});

  EXPECT_EQ(data.has(test_case.query_type), test_case.expected);
  EXPECT_EQ(data.has(MMType{static_cast<MMType::Value>(test_case.query_type)}),
            test_case.expected);
}

TEST_P(MMDataTypeMaskTest, ChecksBatchTypes) {
  const TypeMaskCase& test_case = GetParam();
  const MMBatchData data(test_case.stored_type, MMDict{});

  EXPECT_EQ(data.has(test_case.query_type), test_case.expected);
}

TEST(MMDataEncoderHandoffTest, RestoresIndependentModalityMasksInChunk) {
  MMDataItem video(MMType::VIDEO);
  auto& video_state = video.mutable_state();
  video_state.mutable_token_pos() = {0, 4};
  video_state.mutable_mm_token_num() = 2;
  video_state.mutable_mm_token_mask() =
      torch::tensor({false, true, false, true}, torch::kBool);
  video_state.mutable_seq_index() = 0;
  video_state.mutable_schedule_data().start_pos = 2;
  video_state.mutable_schedule_data().end_pos = 4;

  MMDataItem image(MMType::IMAGE);
  auto& image_state = image.mutable_state();
  image_state.mutable_token_pos() = {4, 1};
  image_state.mutable_mm_token_num() = 1;
  image_state.mutable_mm_token_mask() = torch::ones({1}, torch::kBool);
  image_state.mutable_seq_index() = 0;
  image_state.mutable_schedule_data().end_pos = 1;
  // An already-embedded item must bypass encoder output scattering.
  image.add("image|embedding", torch::full({1, 2}, 14.0));

  MMData data(MMType::VIDEO | MMType::IMAGE, MMItemVec{video, image});
  MMBatchData batch(std::vector<MMData>{data});
  MMTokenNumVisitor video_counts(MMType::VIDEO);
  MMTokenNumVisitor image_counts(MMType::IMAGE);
  ASSERT_TRUE(batch.foreach (video_counts));
  ASSERT_TRUE(batch.foreach (image_counts));
  EXPECT_EQ(video_counts.token_nums(), (std::vector<int32_t>{2}));
  EXPECT_TRUE(image_counts.token_nums().empty());

  MMDict encoded;
  encoded["video|embedding"] =
      std::vector<torch::Tensor>{torch::tensor({{20.0, 20.0}, {27.0, 27.0}})};
  EncoderOutputScatterVisitor scatter(encoded);
  ASSERT_TRUE(batch.foreach (scatter));
  ASSERT_TRUE(scatter.finish());
#if defined(USE_NPU) || defined(USE_MUSA)
  const std::vector<int32_t> kv_lens{5};
  const std::vector<int32_t> q_lens{3};
#else
  const std::vector<int32_t> kv_lens{0, 5};
  const std::vector<int32_t> q_lens{0, 3};
#endif
  EncoderEmbeddingGatherVisitor gather(kv_lens, q_lens);
  ASSERT_TRUE(batch.foreach (gather));
  ASSERT_TRUE(gather.finish(batch));
  EXPECT_TRUE(torch::equal(batch.get<torch::Tensor>("video|embedding").value(),
                           torch::full({1, 2}, 27.0)));
  EXPECT_TRUE(torch::equal(batch.get<torch::Tensor>("image|embedding").value(),
                           torch::full({1, 2}, 14.0)));
  EXPECT_TRUE(torch::equal(batch.get<torch::Tensor>("video|mask").value(),
                           torch::tensor({false, true, false}, torch::kBool)));
  EXPECT_TRUE(torch::equal(batch.get<torch::Tensor>("image|mask").value(),
                           torch::tensor({false, false, true}, torch::kBool)));
}

INSTANTIATE_TEST_SUITE_P(
    TypeMasks,
    MMDataTypeMaskTest,
    ::testing::Values(
        TypeMaskCase{MMType::NONE, MMType::NONE, false},
        TypeMaskCase{MMType::NONE, MMType::IMAGE, false},
        TypeMaskCase{MMType::NONE, MMType::VIDEO, false},
        TypeMaskCase{MMType::NONE, MMType::AUDIO, false},
        TypeMaskCase{MMType::NONE, MMType::EMBEDDING, false},
        TypeMaskCase{MMType::IMAGE, MMType::NONE, false},
        TypeMaskCase{MMType::IMAGE, MMType::IMAGE, true},
        TypeMaskCase{MMType::IMAGE, MMType::VIDEO, false},
        TypeMaskCase{MMType::VIDEO, MMType::VIDEO, true},
        TypeMaskCase{MMType::VIDEO, MMType::IMAGE, false},
        TypeMaskCase{MMType::AUDIO, MMType::AUDIO, true},
        TypeMaskCase{MMType::AUDIO, MMType::IMAGE, false},
        TypeMaskCase{MMType::EMBEDDING, MMType::EMBEDDING, true},
        TypeMaskCase{MMType::EMBEDDING, MMType::IMAGE, false},
        TypeMaskCase{MMType::VIDEO | MMType::EMBEDDING, MMType::NONE, false},
        TypeMaskCase{MMType::VIDEO | MMType::EMBEDDING, MMType::VIDEO, true},
        TypeMaskCase{MMType::VIDEO | MMType::EMBEDDING,
                     MMType::EMBEDDING,
                     true},
        TypeMaskCase{MMType::VIDEO | MMType::EMBEDDING, MMType::IMAGE, false},
        TypeMaskCase{MMType::VIDEO | MMType::EMBEDDING, MMType::AUDIO, false},
        TypeMaskCase{MMType::VIDEO, MMType::VIDEO | MMType::AUDIO, true},
        TypeMaskCase{MMType::VIDEO, MMType::IMAGE | MMType::AUDIO, false},
        TypeMaskCase{MMType::IMAGE | MMType::AUDIO,
                     MMType::VIDEO | MMType::AUDIO,
                     true}));

}  // namespace
}  // namespace xllm
