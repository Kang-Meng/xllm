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

#include "core/runtime/encoder_executor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "core/framework/multimodal/mm_visitor.h"

namespace xllm {
namespace {

std::vector<int32_t> sequence_lengths(int32_t length) {
#if defined(USE_NPU) || defined(USE_MUSA)
  return {length};
#else
  return {0, length};
#endif
}

MMDataItem make_item(MMType modality,
                     int32_t offset,
                     const torch::Tensor& mask,
                     uint8_t key) {
  MMDataItem item(modality);
  auto& state = item.mutable_state();
  state.mutable_token_pos() = {offset, static_cast<int32_t>(mask.numel())};
  state.mutable_mm_token_mask() = mask;
  state.mutable_mm_token_num() = mask.sum().item<int32_t>();
  state.mutable_schedule_data().key.data[0] = key;
  return item;
}

ModelInputParams schedule(MMData data, int32_t computed, int32_t query_length) {
  UpdateMMItemScheduleStateVisitor visitor(computed, query_length, 0);
  EXPECT_TRUE(data.foreach (visitor));
  ModelInputParams params;
  params.multimodal.mm_data =
      MMBatchData({MMData(visitor.scheduled_type_, visitor.mm_data_items_)});
  params.attention.host.kv_seq_lens = sequence_lengths(computed + query_length);
  params.attention.host.q_seq_lens = sequence_lengths(query_length);
  return params;
}

TEST(EncoderExecutorTest, EncodesGatheredItemsWithCurrentExecutionContext) {
  MMDataItem first = make_item(MMType::IMAGE, 0, torch::tensor({true}), 11);
  MMDataItem second = make_item(MMType::IMAGE, 1, torch::tensor({true}), 12);
  first.add("pixel_values", torch::tensor({{1.0f, 2.0f}}));
  second.add("pixel_values", torch::tensor({{3.0f, 4.0f}}));
  auto params = schedule(MMData(MMType::IMAGE, MMItemVec{first, second}), 0, 2);
  params.meta.num_sequences = 1;
  EncoderExecutor executor(0);
  int32_t encoder_calls = 0;
  executor.prepare(
      params, [&encoder_calls](const ModelInputParams& inputs) -> MMDict {
        ++encoder_calls;
        EXPECT_EQ(inputs.meta.num_sequences, 1);
        EXPECT_EQ(inputs.attention.host.kv_seq_lens, sequence_lengths(2));
        EXPECT_EQ(inputs.attention.host.q_seq_lens, sequence_lengths(2));
        const torch::Tensor pixels =
            inputs.multimodal.mm_data.get<torch::Tensor>("pixel_values")
                .value();
        EXPECT_TRUE(
            torch::equal(pixels, torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}})));
        return {{"image|embedding", pixels.mul(10).split(1)}};
      });
  EXPECT_EQ(encoder_calls, 1);
  EXPECT_TRUE(torch::equal(
      params.multimodal.mm_data.get<torch::Tensor>("image|embedding").value(),
      torch::tensor({{10.0f, 20.0f}, {30.0f, 40.0f}})));
}

class EncoderExecutorDeathTest : public ::testing::Test {
 protected:
  void SetUp() override { GTEST_FLAG_SET(death_test_style, "threadsafe"); }
};

TEST_F(EncoderExecutorDeathTest, RejectsMismatchedSequenceLengthCounts) {
  EXPECT_DEATH(EncoderEmbeddingGatherVisitor(sequence_lengths(2),
                                             std::vector<int32_t>{}),
               "sequence length counts do not match");
}

TEST_F(EncoderExecutorDeathTest, RejectsContextShorterThanScheduledQuery) {
  EXPECT_DEATH(
      EncoderEmbeddingGatherVisitor(sequence_lengths(1), sequence_lengths(2)),
      "kv length below q length");
}

TEST_F(EncoderExecutorDeathTest, RejectsItemOutsideScheduledSequences) {
  MMDataItem item = make_item(MMType::IMAGE, 0, torch::tensor({true}), 15);
  item.add("image|embedding", torch::ones({1, 2}));
  item.mutable_state().mutable_seq_index() = 1;
  EncoderEmbeddingGatherVisitor gather(sequence_lengths(1),
                                       sequence_lengths(1));
  EXPECT_DEATH(gather.visit(item),
               "item sequence index outside scheduled batch");
}

TEST_F(EncoderExecutorDeathTest, RejectsItemBeforeItsSequenceWindow) {
  MMDataItem item = make_item(MMType::AUDIO, 0, torch::tensor({true}), 16);
  item.add("audio|embedding", torch::ones({1, 2}));
  item.mutable_state().mutable_seq_index() = 0;
  item.mutable_state().mutable_schedule_data().end_pos = 1;
  EncoderEmbeddingGatherVisitor gather(sequence_lengths(2),
                                       sequence_lengths(1));
  EXPECT_DEATH(gather.visit(item),
               "scatter range starts before its sequence window");
}

TEST_F(EncoderExecutorDeathTest, RejectsItemPastItsSequenceWindow) {
  MMDataItem item = make_item(MMType::AUDIO, 1, torch::tensor({true}), 17);
  item.add("audio|embedding", torch::ones({1, 2}));
  item.mutable_state().mutable_seq_index() = 0;
  item.mutable_state().mutable_schedule_data().end_pos = 1;
  EncoderEmbeddingGatherVisitor gather(sequence_lengths(1),
                                       sequence_lengths(1));
  EXPECT_DEATH(gather.visit(item),
               "scatter range ends past its sequence window");
}

TEST_F(EncoderExecutorDeathTest, RejectsPrecomputedEmbeddingWithMissingRows) {
  MMDataItem item =
      make_item(MMType::IMAGE, 0, torch::tensor({true, true}), 13);
  item.add("image|embedding", torch::ones({1, 2}));
  auto params = schedule(MMData(MMType::IMAGE, MMItemVec{item}), 0, 2);
  EncoderExecutor executor(0);
  EXPECT_DEATH(
      executor.prepare(params,
                       [](const ModelInputParams& inputs) -> MMDict {
                         EXPECT_TRUE(inputs.multimodal.mm_data.data().empty());
                         return {};
                       }),
      "embedding rows do not match");
}

TEST_F(EncoderExecutorDeathTest, RejectsEncoderOutputWithMissingRows) {
  MMDataItem item =
      make_item(MMType::IMAGE, 0, torch::tensor({true, true}), 14);
  item.add("pixel_values", torch::ones({2, 2}));
  auto params = schedule(MMData(MMType::IMAGE, MMItemVec{item}), 0, 2);
  EncoderExecutor executor(0);
  EXPECT_DEATH(
      executor.prepare(
          params,
          [](const ModelInputParams& inputs) -> MMDict {
            const torch::Tensor pixels =
                inputs.multimodal.mm_data.get<torch::Tensor>("pixel_values")
                    .value();
            return {{"image|embedding",
                     std::vector<torch::Tensor>{pixels.narrow(0, 0, 1)}}};
          }),
      "embedding rows do not match");
}

TEST(EncoderExecutorTest, ReusesFullItemEncodingAcrossChunks) {
  MMDataItem item =
      make_item(MMType::IMAGE, 1, torch::ones({4}, torch::kBool), 1);
  item.add("pixel_values", torch::ones({4, 2}));
  const MMData request(MMType::IMAGE, MMItemVec{item});
  EncoderExecutor executor(1024);
  int32_t encoder_calls = 0;
  const torch::Tensor full = torch::tensor(
      {{10.0f, 11.0f}, {20.0f, 21.0f}, {30.0f, 31.0f}, {40.0f, 41.0f}});
  const auto encode = [&encoder_calls,
                       &full](const ModelInputParams& params) -> MMDict {
    const auto& inputs = params.multimodal.mm_data;
    if (!inputs.has("pixel_values")) {
      return {};
    }
    ++encoder_calls;
    return {{"image|embedding", std::vector<torch::Tensor>{full}}};
  };
  auto first = schedule(request, 0, 3);
  executor.prepare(first, encode);
  EXPECT_EQ(encoder_calls, 1);
  EXPECT_TRUE(torch::equal(
      first.multimodal.mm_data.get<torch::Tensor>("image|embedding").value(),
      full.slice(0, 0, 2)));
  EXPECT_TRUE(torch::equal(
      first.multimodal.mm_data.get<torch::Tensor>("image|mask").value(),
      torch::tensor({false, true, true})));

  auto second = schedule(request, 3, 3);
  executor.prepare(second, encode);
  EXPECT_EQ(encoder_calls, 1);
  EXPECT_TRUE(torch::equal(
      second.multimodal.mm_data.get<torch::Tensor>("image|embedding").value(),
      full.slice(0, 2, 4)));
  EXPECT_TRUE(torch::equal(
      second.multimodal.mm_data.get<torch::Tensor>("image|mask").value(),
      torch::tensor({true, true, false})));
}

TEST(EncoderExecutorTest, EncodesOnlyCacheMissesInMixedModalityBatch) {
  MMDataItem image = make_item(MMType::IMAGE, 0, torch::tensor({true}), 7);
  image.add("pixel_values", torch::tensor({{1.0f, 2.0f}}));
  MMDataItem video =
      make_item(MMType::VIDEO, 1, torch::tensor({true, false, true}), 8);
  video.add("pixel_values_videos", torch::tensor({{3.0f, 4.0f}, {5.0f, 6.0f}}));
  EncoderExecutor executor(1024);
  int32_t image_encodes = 0;
  int32_t video_encodes = 0;
  const auto encode = [&image_encodes, &video_encodes](
                          const ModelInputParams& params) -> MMDict {
    const auto& inputs = params.multimodal.mm_data;
    MMDict outputs;
    if (inputs.has("pixel_values")) {
      ++image_encodes;
      outputs["image|embedding"] = std::vector<torch::Tensor>{
          inputs.get<torch::Tensor>("pixel_values").value() * 10};
    }
    if (inputs.has("pixel_values_videos")) {
      ++video_encodes;
      outputs["video|embedding"] = std::vector<torch::Tensor>{
          inputs.get<torch::Tensor>("pixel_values_videos").value() * 10};
    }
    return outputs;
  };
  auto warmup = schedule(MMData(MMType::IMAGE, MMItemVec{image}), 0, 1);
  executor.prepare(warmup, encode);
  EXPECT_EQ(image_encodes, 1);
  EXPECT_EQ(video_encodes, 0);

  const MMData request(MMType::IMAGE | MMType::VIDEO, MMItemVec{image, video});
  auto mixed = schedule(request, 0, 4);
  executor.prepare(mixed, encode);
  EXPECT_EQ(image_encodes, 1);
  EXPECT_EQ(video_encodes, 1);
  EXPECT_TRUE(torch::equal(
      mixed.multimodal.mm_data.get<torch::Tensor>("image|embedding").value(),
      torch::tensor({{10.0f, 20.0f}})));
  EXPECT_TRUE(torch::equal(
      mixed.multimodal.mm_data.get<torch::Tensor>("video|embedding").value(),
      torch::tensor({{30.0f, 40.0f}, {50.0f, 60.0f}})));
  EXPECT_TRUE(torch::equal(
      mixed.multimodal.mm_data.get<torch::Tensor>("image|mask").value(),
      torch::tensor({true, false, false, false})));
  EXPECT_TRUE(torch::equal(
      mixed.multimodal.mm_data.get<torch::Tensor>("video|mask").value(),
      torch::tensor({false, true, false, true})));

  auto cached = schedule(request, 0, 4);
  executor.prepare(cached, encode);
  EXPECT_EQ(image_encodes, 1);
  EXPECT_EQ(video_encodes, 1);
  EXPECT_TRUE(torch::equal(
      cached.multimodal.mm_data.get<torch::Tensor>("video|embedding").value(),
      mixed.multimodal.mm_data.get<torch::Tensor>("video|embedding").value()));
}

TEST(EncoderExecutorTest, ReencodesEvictedItem) {
  const torch::Tensor first_features = torch::tensor({{1.0f, 2.0f}});
  const torch::Tensor second_features = torch::tensor({{3.0f, 4.0f}});
  MMDataItem first = make_item(MMType::IMAGE, 0, torch::tensor({true}), 9);
  first.add("pixel_values", first_features);
  MMDataItem second = make_item(MMType::IMAGE, 0, torch::tensor({true}), 10);
  second.add("pixel_values", second_features);
  EncoderExecutor executor(static_cast<int64_t>(first_features.nbytes()));
  int32_t encoder_calls = 0;
  const auto encode =
      [&encoder_calls](const ModelInputParams& params) -> MMDict {
    const auto& inputs = params.multimodal.mm_data;
    if (!inputs.has("pixel_values")) {
      return {};
    }
    ++encoder_calls;
    return {{"image|embedding",
             std::vector<torch::Tensor>{
                 inputs.get<torch::Tensor>("pixel_values").value() * 10}}};
  };
  const auto prepare = [&executor, &encode](const MMDataItem& item) {
    auto batch = schedule(MMData(MMType::IMAGE, MMItemVec{item}), 0, 1);
    executor.prepare(batch, encode);
    return batch.multimodal.mm_data.get<torch::Tensor>("image|embedding")
        .value();
  };
  EXPECT_TRUE(torch::equal(prepare(first), first_features * 10));
  EXPECT_EQ(encoder_calls, 1);
  EXPECT_TRUE(torch::equal(prepare(first), first_features * 10));
  EXPECT_EQ(encoder_calls, 1);
  EXPECT_TRUE(torch::equal(prepare(second), second_features * 10));
  EXPECT_EQ(encoder_calls, 2);
  EXPECT_TRUE(torch::equal(prepare(first), first_features * 10));
  EXPECT_EQ(encoder_calls, 3);
}

TEST(EncoderExecutorTest, CombinesPrecomputedAndEncodedItemsInPromptOrder) {
  MMDataItem video =
      make_item(MMType::VIDEO, 0, torch::tensor({true, false, true}), 2);
  video.add("video|embedding", torch::tensor({{21.0f, 22.0f}, {23.0f, 24.0f}}));
  MMDataItem image = make_item(MMType::IMAGE, 3, torch::tensor({true}), 3);
  image.add("pixel_values", torch::ones({1, 2}));
  auto batch = schedule(
      MMData(MMType::IMAGE | MMType::VIDEO, MMItemVec{video, image}), 0, 4);
  EncoderExecutor executor(0);
  executor.prepare(batch, [](const ModelInputParams& params) -> MMDict {
    const auto& inputs = params.multimodal.mm_data;
    EXPECT_FALSE(inputs.has("video|embedding"));
    EXPECT_TRUE(inputs.has("pixel_values"));
    return {{"image|embedding",
             std::vector<torch::Tensor>{torch::tensor({{11.0f, 12.0f}})}}};
  });
  EXPECT_TRUE(torch::equal(
      batch.multimodal.mm_data.get<torch::Tensor>("video|mask").value(),
      torch::tensor({true, false, true, false})));
  EXPECT_TRUE(torch::equal(
      batch.multimodal.mm_data.get<torch::Tensor>("image|mask").value(),
      torch::tensor({false, false, false, true})));
  EXPECT_TRUE(torch::equal(
      batch.multimodal.mm_data.get<torch::Tensor>("video|embedding").value(),
      torch::tensor({{21.0f, 22.0f}, {23.0f, 24.0f}})));
}

TEST(EncoderExecutorTest, AudioGapChunkHasNoReplacementRows) {
  MMDataItem audio =
      make_item(MMType::AUDIO, 0, torch::tensor({true, false, false, true}), 4);
  audio.add("audio|embedding", torch::tensor({{10.0f, 11.0f}, {20.0f, 21.0f}}));
  auto batch = schedule(MMData(MMType::AUDIO, MMItemVec{audio}), 1, 2);
  EncoderExecutor executor(0);
  executor.prepare(batch, [](const ModelInputParams& params) -> MMDict {
    const auto& inputs = params.multimodal.mm_data;
    EXPECT_TRUE(inputs.data().empty());
    return {};
  });
  EXPECT_EQ(batch.multimodal.mm_data.get<torch::Tensor>("audio|embedding")
                .value()
                .size(0),
            0);
  EXPECT_TRUE(torch::equal(
      batch.multimodal.mm_data.get<torch::Tensor>("audio|mask").value(),
      torch::tensor({false, false})));
}

TEST(EncoderExecutorTest, PreservesMaskRowsAcrossItemsOfTheSameModality) {
  MMDataItem first =
      make_item(MMType::IMAGE, 0, torch::tensor({true, false}), 5);
  MMDataItem second = make_item(MMType::IMAGE, 2, torch::tensor({true}), 6);
  first.add("image|embedding", torch::tensor({{1.0f, 2.0f}}));
  second.add("image|embedding", torch::tensor({{3.0f, 4.0f}}));
  auto batch = schedule(MMData(MMType::IMAGE, MMItemVec{first, second}), 0, 3);
  EncoderExecutor executor(0);
  executor.prepare(batch, [](const ModelInputParams& params) -> MMDict {
    const auto& inputs = params.multimodal.mm_data;
    EXPECT_TRUE(inputs.data().empty());
    return {};
  });
  const auto mask =
      batch.multimodal.mm_data.get<torch::Tensor>("image|mask").value();
  const auto embeddings =
      batch.multimodal.mm_data.get<torch::Tensor>("image|embedding").value();
  EXPECT_EQ(mask.device(), embeddings.device());
  EXPECT_EQ(mask.scalar_type(), torch::kBool);
  EXPECT_TRUE(torch::equal(mask, torch::tensor({true, false, true})));
  EXPECT_TRUE(
      torch::equal(embeddings, torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}})));
}

}  // namespace
}  // namespace xllm
