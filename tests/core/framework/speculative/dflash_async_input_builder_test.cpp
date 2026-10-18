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

#include "core/framework/speculative/dflash_async_input_builder.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <array>

namespace xllm::dflash_async {
namespace {

testing::AssertionResult compare_next_draft_inputs(
    const NextDraftInputs& actual,
    const NextDraftInputs& expected) {
  constexpr std::array<const char*, 8> kFieldNames = {"accepted_counts",
                                                      "anchor_tokens",
                                                      "query_tokens",
                                                      "query_positions",
                                                      "query_slots",
                                                      "kv_lengths",
                                                      "context_positions",
                                                      "context_slots"};
  const std::array<torch::Tensor, 8> actual_fields = {actual.accepted_counts,
                                                      actual.anchor_tokens,
                                                      actual.query_tokens,
                                                      actual.query_positions,
                                                      actual.query_slots,
                                                      actual.kv_lengths,
                                                      actual.context_positions,
                                                      actual.context_slots};
  const std::array<torch::Tensor, 8> expected_fields = {
      expected.accepted_counts,
      expected.anchor_tokens,
      expected.query_tokens,
      expected.query_positions,
      expected.query_slots,
      expected.kv_lengths,
      expected.context_positions,
      expected.context_slots};
  for (size_t field_index = 0; field_index < kFieldNames.size();
       ++field_index) {
    if (actual_fields[field_index].scalar_type() !=
            expected_fields[field_index].scalar_type() ||
        actual_fields[field_index].device() !=
            expected_fields[field_index].device() ||
        actual_fields[field_index].sizes() !=
            expected_fields[field_index].sizes() ||
        !torch::equal(actual_fields[field_index],
                      expected_fields[field_index])) {
      return testing::AssertionFailure() << kFieldNames[field_index];
    }
  }
  return testing::AssertionSuccess();
}

void check_all_acceptance_lengths(int64_t accepted_width, int64_t query_width) {
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kMaskTokenId = 999;
  torch::Tensor base_offsets =
      torch::tensor({0, 112, 113, 126, 127, 128, 2303, 2304}, torch::kLong);
  const int64_t base_offset_count = base_offsets.numel();
  const int64_t batch_size = accepted_width * base_offset_count;
  torch::Tensor base_positions = base_offsets.repeat({accepted_width});
  torch::Tensor accepted_counts =
      torch::arange(1, accepted_width + 1, torch::kLong)
          .repeat_interleave(base_offset_count);
  torch::Tensor token_offsets =
      torch::arange(accepted_width, torch::kLong).view({1, accepted_width});
  torch::Tensor tokens =
      torch::arange(batch_size * accepted_width, torch::kLong)
          .reshape({batch_size, accepted_width}) +
      100;
  torch::Tensor accepted_tokens =
      torch::where(token_offsets.lt(accepted_counts.unsqueeze(1)),
                   tokens,
                   torch::full_like(tokens, -1));
  torch::Tensor block_table = torch::arange(batch_size * 32, torch::kInt32)
                                  .reshape({batch_size, 32})
                                  .flip({1});
  NextDraftInputs result = prepare_next_draft(accepted_tokens,
                                              base_positions,
                                              block_table,
                                              kMaskTokenId,
                                              kBlockSize,
                                              query_width);
  ContextKvInputs context = prepare_context_kv(
      accepted_tokens, base_positions, block_table, kBlockSize);
  EXPECT_TRUE(torch::equal(context.positions, result.context_positions));
  EXPECT_TRUE(torch::equal(context.slots, result.context_slots));
  EXPECT_EQ(result.query_tokens.numel(), batch_size * query_width);
  EXPECT_EQ(result.query_positions.numel(), batch_size * query_width);
  EXPECT_EQ(result.query_slots.numel(), batch_size * query_width);
  EXPECT_EQ(result.context_positions.numel(), batch_size * accepted_width);
  EXPECT_EQ(result.context_slots.numel(), batch_size * accepted_width);
  EXPECT_EQ(result.kv_lengths.scalar_type(), torch::kInt32);
  for (int64_t sequence_index = 0; sequence_index < batch_size;
       ++sequence_index) {
    const int64_t prefix_length = sequence_index / base_offset_count + 1;
    const int64_t base_position =
        base_positions[sequence_index].item<int64_t>();
    const int64_t anchor_token =
        accepted_tokens[sequence_index][prefix_length - 1].item<int64_t>();
    EXPECT_EQ(result.accepted_counts[sequence_index].item<int64_t>(),
              prefix_length);
    EXPECT_EQ(result.anchor_tokens[sequence_index].item<int64_t>(),
              anchor_token);
    EXPECT_EQ(result.kv_lengths[sequence_index].item<int64_t>(),
              base_position + prefix_length + query_width);
    for (int64_t token_index = 0; token_index < query_width; ++token_index) {
      const int64_t result_index = sequence_index * query_width + token_index;
      const int64_t query_position =
          base_position + prefix_length + token_index;
      const int64_t block_id =
          block_table[sequence_index][query_position / kBlockSize]
              .item<int64_t>();
      EXPECT_EQ(result.query_positions[result_index].item<int64_t>(),
                query_position);
      EXPECT_EQ(result.query_slots[result_index].item<int64_t>(),
                block_id * kBlockSize + query_position % kBlockSize);
      EXPECT_EQ(result.query_tokens[result_index].item<int64_t>(),
                token_index == 0 ? anchor_token : kMaskTokenId);
    }
    for (int64_t token_index = 0; token_index < accepted_width; ++token_index) {
      const int64_t result_index =
          sequence_index * accepted_width + token_index;
      const int64_t context_position = base_position + token_index;
      const int64_t context_slot =
          token_index < prefix_length
              ? block_table[sequence_index][context_position / kBlockSize]
                            .item<int64_t>() *
                        kBlockSize +
                    context_position % kBlockSize
              : -1;
      EXPECT_EQ(result.context_positions[result_index].item<int64_t>(),
                context_position);
      EXPECT_EQ(result.context_slots[result_index].item<int64_t>(),
                context_slot);
    }
  }
}

TEST(DFlashAsyncInputBuilderTest, ReorderedSubsetsPreserveRowIdentity) {
  constexpr int64_t kAcceptedWidth = 16;
  constexpr int64_t kQueryWidth = 15;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kMaskTokenId = 42;
  constexpr int64_t kBatchSize = 8;
  torch::Tensor base_positions =
      torch::tensor({95, 112, 113, 126, 127, 128, 255, 256}, torch::kLong);
  torch::Tensor accepted_counts =
      torch::tensor({1, 16, 2, 15, 4, 3, 8, 1}, torch::kLong);
  torch::Tensor token_offsets =
      torch::arange(kAcceptedWidth, torch::kLong).unsqueeze(0);
  torch::Tensor tokens =
      torch::arange(kBatchSize * kAcceptedWidth, torch::kLong)
          .view({kBatchSize, kAcceptedWidth});
  torch::Tensor accepted =
      torch::where(token_offsets.lt(accepted_counts.unsqueeze(1)),
                   tokens,
                   torch::full_like(tokens, -1));
  torch::Tensor block_tables =
      torch::arange(kBatchSize * 4, torch::kInt32).view({kBatchSize, 4});
  NextDraftInputs whole = prepare_next_draft(accepted,
                                             base_positions,
                                             block_tables,
                                             kMaskTokenId,
                                             kBlockSize,
                                             kQueryWidth);
  for (const torch::Tensor& rows :
       {torch::tensor({7, 1, 4}, torch::kLong),
        torch::tensor({0, 6, 3, 2, 5}, torch::kLong)}) {
    torch::Tensor local_accepted = accepted.index_select(0, rows);
    torch::Tensor local_positions = base_positions.index_select(0, rows);
    torch::Tensor local_tables = block_tables.index_select(0, rows);
    NextDraftInputs local = prepare_next_draft(local_accepted,
                                               local_positions,
                                               local_tables,
                                               kMaskTokenId,
                                               kBlockSize,
                                               kQueryWidth);
    ContextKvInputs context = prepare_context_kv(
        local_accepted, local_positions, local_tables, kBlockSize);
    EXPECT_TRUE(torch::equal(local.accepted_counts,
                             whole.accepted_counts.index_select(0, rows)));
    EXPECT_TRUE(torch::equal(local.anchor_tokens,
                             whole.anchor_tokens.index_select(0, rows)));
    EXPECT_TRUE(
        torch::equal(local.kv_lengths, whole.kv_lengths.index_select(0, rows)));
    EXPECT_TRUE(
        torch::equal(local.query_slots.view({rows.numel(), kQueryWidth}),
                     whole.query_slots.view({kBatchSize, kQueryWidth})
                         .index_select(0, rows)));
    EXPECT_TRUE(
        torch::equal(local.query_tokens.view({rows.numel(), kQueryWidth}),
                     whole.query_tokens.view({kBatchSize, kQueryWidth})
                         .index_select(0, rows)));
    EXPECT_TRUE(torch::equal(context.slots, local.context_slots));
    EXPECT_TRUE(
        torch::equal(context.slots.view({rows.numel(), kAcceptedWidth}),
                     whole.context_slots.view({kBatchSize, kAcceptedWidth})
                         .index_select(0, rows)));
  }
}

TEST(DFlashAsyncInputBuilderTest,
     TenThousandBatchesMatchIndependentSlotOracle) {
  constexpr int64_t kAcceptedWidth = 16;
  constexpr int64_t kQueryWidth = 15;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kBlocksPerRow = 32;
  constexpr int64_t kMaskTokenId = 42;
  constexpr std::array<int64_t, 10> kBasePositions = {
      0, 112, 113, 126, 127, 128, 255, 256, 2303, 2304};
  for (int64_t case_index = 0; case_index < 10000; ++case_index) {
    SCOPED_TRACE(case_index);
    const int64_t batch_size = case_index % 8 + 1;
    torch::Tensor accepted =
        torch::full({batch_size, kAcceptedWidth}, -1, torch::kLong);
    torch::Tensor bases = torch::empty({batch_size}, torch::kLong);
    torch::Tensor tables =
        torch::empty({batch_size, kBlocksPerRow}, torch::kInt32);
    torch::Tensor expected_counts = torch::empty_like(bases);
    torch::Tensor expected_anchors = torch::empty_like(bases);
    torch::Tensor expected_lengths = torch::empty({batch_size}, torch::kInt32);
    torch::Tensor expected_tokens =
        torch::full({batch_size * kQueryWidth}, kMaskTokenId, torch::kLong);
    torch::Tensor expected_positions = torch::empty_like(expected_tokens);
    torch::Tensor expected_slots = torch::empty_like(expected_tokens);
    torch::Tensor expected_context_positions =
        torch::empty({batch_size * kAcceptedWidth}, torch::kLong);
    torch::Tensor expected_context_slots =
        torch::full_like(expected_context_positions, -1);
    for (int64_t row = 0; row < batch_size; ++row) {
      const int64_t base = kBasePositions[(case_index / 8 + row) % 10];
      const int64_t count = (case_index / 80 + row) % kAcceptedWidth + 1;
      bases.data_ptr<int64_t>()[row] = base;
      expected_counts.data_ptr<int64_t>()[row] = count;
      const int64_t first_token = case_index * 128 + row * kAcceptedWidth;
      expected_anchors.data_ptr<int64_t>()[row] = first_token + count - 1;
      expected_lengths.data_ptr<int32_t>()[row] = base + count + kQueryWidth;
      for (int64_t block = 0; block < kBlocksPerRow; ++block) {
        tables.data_ptr<int32_t>()[row * kBlocksPerRow + block] =
            row * kBlocksPerRow + (block * 17 + case_index) % kBlocksPerRow;
      }
      const auto slot_for_position = [&](int64_t position) {
        const int64_t block =
            row * kBlocksPerRow +
            ((position / kBlockSize) * 17 + case_index) % kBlocksPerRow;
        return block * kBlockSize + position % kBlockSize;
      };
      for (int64_t offset = 0; offset < kAcceptedWidth; ++offset) {
        const int64_t index = row * kAcceptedWidth + offset;
        expected_context_positions.data_ptr<int64_t>()[index] = base + offset;
        if (offset < count) {
          accepted.data_ptr<int64_t>()[index] = first_token + offset;
          expected_context_slots.data_ptr<int64_t>()[index] =
              slot_for_position(base + offset);
        }
      }
      for (int64_t offset = 0; offset < kQueryWidth; ++offset) {
        const int64_t index = row * kQueryWidth + offset;
        expected_positions.data_ptr<int64_t>()[index] = base + count + offset;
        expected_slots.data_ptr<int64_t>()[index] =
            slot_for_position(base + count + offset);
      }
      expected_tokens.data_ptr<int64_t>()[row * kQueryWidth] =
          first_token + count - 1;
    }
    NextDraftInputs result = prepare_next_draft(
        accepted, bases, tables, kMaskTokenId, kBlockSize, kQueryWidth);
    ContextKvInputs context =
        prepare_context_kv(accepted, bases, tables, kBlockSize);
    ASSERT_TRUE(torch::equal(result.accepted_counts, expected_counts));
    ASSERT_TRUE(torch::equal(result.anchor_tokens, expected_anchors));
    ASSERT_TRUE(torch::equal(result.kv_lengths, expected_lengths));
    ASSERT_TRUE(torch::equal(result.query_tokens, expected_tokens));
    ASSERT_TRUE(torch::equal(result.query_positions, expected_positions));
    ASSERT_TRUE(torch::equal(result.query_slots, expected_slots));
    ASSERT_TRUE(
        torch::equal(result.context_positions, expected_context_positions));
    ASSERT_TRUE(torch::equal(result.context_slots, expected_context_slots));
    ASSERT_TRUE(torch::equal(context.positions, expected_context_positions));
    ASSERT_TRUE(torch::equal(context.slots, expected_context_slots));
  }
}

TEST(DFlashAsyncInputBuilderTest, DSparkAcceptanceLengthsAcrossPageBoundaries) {
  constexpr int64_t kAcceptedWidth = 16;
  constexpr int64_t kQueryWidth = 15;
  check_all_acceptance_lengths(kAcceptedWidth, kQueryWidth);
}

TEST(DFlashAsyncInputBuilderTest, SingleTokenQueryAcrossPageBoundaries) {
  constexpr int64_t kAcceptedWidth = 16;
  constexpr int64_t kQueryWidth = 1;
  check_all_acceptance_lengths(kAcceptedWidth, kQueryWidth);
}

TEST(DFlashAsyncInputBuilderTest, AllAcceptanceLengthsAcrossPageBoundaries) {
  constexpr int64_t kWidth = 8;
  constexpr int64_t kBlockSize = 128;
  torch::Tensor accepted = torch::full({kWidth, kWidth}, -1, torch::kLong);
  torch::Tensor base =
      torch::tensor({120, 121, 126, 127, 128, 2297, 2303, 2304}, torch::kLong);
  torch::Tensor table =
      torch::arange(8 * 32, torch::kInt32).reshape({8, 32}).flip({1});
  for (int64_t seq = 0; seq < kWidth; ++seq) {
    for (int64_t token = 0; token <= seq; ++token) {
      accepted[seq][token] = 100 + seq * kWidth + token;
    }
  }
  NextDraftInputs result = prepare_next_draft(
      accepted, base, table, /*mask_token_id=*/999, kBlockSize);
  constexpr int64_t kMaskTokenId = 999;
  NextDraftInputs explicit_result = prepare_next_draft(
      accepted, base, table, kMaskTokenId, kBlockSize, kWidth);
  EXPECT_TRUE(
      torch::equal(result.accepted_counts, explicit_result.accepted_counts));
  EXPECT_TRUE(
      torch::equal(result.anchor_tokens, explicit_result.anchor_tokens));
  EXPECT_TRUE(torch::equal(result.query_tokens, explicit_result.query_tokens));
  EXPECT_TRUE(
      torch::equal(result.query_positions, explicit_result.query_positions));
  EXPECT_TRUE(torch::equal(result.query_slots, explicit_result.query_slots));
  EXPECT_TRUE(torch::equal(result.kv_lengths, explicit_result.kv_lengths));
  EXPECT_TRUE(torch::equal(result.context_positions,
                           explicit_result.context_positions));
  EXPECT_TRUE(
      torch::equal(result.context_slots, explicit_result.context_slots));
  for (int64_t seq = 0; seq < kWidth; ++seq) {
    const int64_t prefix = seq + 1;
    const int64_t old_position = base[seq].item<int64_t>();
    EXPECT_EQ(result.accepted_counts[seq].item<int64_t>(), prefix);
    EXPECT_EQ(result.kv_lengths[seq].item<int64_t>(),
              old_position + prefix + kWidth);
    for (int64_t token = 0; token < kWidth; ++token) {
      const int64_t index = seq * kWidth + token;
      const int64_t query_position = old_position + prefix + token;
      const int64_t page =
          table[seq][query_position / kBlockSize].item<int64_t>();
      EXPECT_EQ(result.query_positions[index].item<int64_t>(), query_position);
      EXPECT_EQ(result.query_slots[index].item<int64_t>(),
                page * kBlockSize + query_position % kBlockSize);
      EXPECT_EQ(result.query_tokens[index].item<int64_t>(),
                token == 0 ? accepted[seq][prefix - 1].item<int64_t>() : 999);
      const int64_t context_position = old_position + token;
      const int64_t context_slot =
          token < prefix
              ? table[seq][context_position / kBlockSize].item<int64_t>() *
                        kBlockSize +
                    context_position % kBlockSize
              : -1;
      EXPECT_EQ(result.context_positions[index].item<int64_t>(),
                context_position);
      EXPECT_EQ(result.context_slots[index].item<int64_t>(), context_slot);
    }
  }
}

TEST(DFlashAsyncInputBuilderTest,
     OnlyTheContiguousAcceptedPrefixWritesContext) {
  torch::Tensor accepted = torch::tensor(
      {{31, 32, -1, 77, -1, -1, 88, -1, -1, -1, -1, -1, -1, -1, -1, 99}},
      torch::kLong);
  constexpr int64_t kMaskTokenId = 999;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kQueryWidth = 15;
  NextDraftInputs result =
      prepare_next_draft(accepted,
                         torch::tensor({127}, torch::kLong),
                         torch::tensor({{8, 3, 9}}, torch::kInt32),
                         kMaskTokenId,
                         kBlockSize,
                         kQueryWidth);
  EXPECT_EQ(result.accepted_counts.item<int64_t>(), 2);
  EXPECT_EQ(result.anchor_tokens.item<int64_t>(), 32);
  torch::Tensor context_slots = torch::full({16}, -1, torch::kInt32);
  context_slots[0] = 1151;
  context_slots[1] = 384;
  EXPECT_TRUE(torch::equal(result.context_slots, context_slots));
  EXPECT_TRUE(torch::equal(result.context_positions,
                           torch::arange(127, 143, torch::kLong)));
  torch::Tensor query_tokens =
      torch::full({kQueryWidth}, kMaskTokenId, torch::kLong);
  query_tokens[0] = 32;
  EXPECT_TRUE(torch::equal(result.query_tokens, query_tokens));
  EXPECT_TRUE(torch::equal(result.query_positions,
                           torch::arange(129, 144, torch::kLong)));
  EXPECT_TRUE(
      torch::equal(result.query_slots, torch::arange(385, 400, torch::kInt32)));
  EXPECT_EQ(result.kv_lengths.item<int64_t>(), 144);
}

TEST(DFlashAsyncInputBuilderTest, RejectsNonPositiveQueryWidth) {
  torch::Tensor accepted_tokens = torch::tensor({{31, -1}}, torch::kLong);
  torch::Tensor base_positions = torch::tensor({0}, torch::kLong);
  torch::Tensor block_table = torch::tensor({{0}}, torch::kInt32);
  constexpr int64_t kMaskTokenId = 999;
  constexpr int64_t kBlockSize = 128;
  for (const int64_t query_width : {0, -1}) {
    EXPECT_DEATH(prepare_next_draft(accepted_tokens,
                                    base_positions,
                                    block_table,
                                    kMaskTokenId,
                                    kBlockSize,
                                    query_width),
                 "query_width");
  }
}

TEST(DFlashAsyncInputBuilderTest,
     FixedReplayAndCheckerDetectEveryChangedField) {
  const torch::Tensor accepted =
      torch::tensor({{0, 17, -1, 99}, {31, 32, 33, -1}}, torch::kLong);
  const torch::Tensor bases = torch::tensor({126, 127}, torch::kLong);
  const torch::Tensor tables =
      torch::tensor({{7, 2, 9}, {3, 8, 5}}, torch::kInt32);
  const NextDraftInputs expected{
      torch::tensor({2, 3}, torch::kLong),
      torch::tensor({17, 33}, torch::kLong),
      torch::tensor({17, 999, 33, 999}, torch::kLong),
      torch::tensor({128, 129, 130, 131}, torch::kLong),
      torch::tensor({256, 257, 1026, 1027}, torch::kInt32),
      torch::tensor({130, 132}, torch::kInt32),
      torch::tensor({126, 127, 128, 129, 127, 128, 129, 130}, torch::kLong),
      torch::tensor({1022, 1023, -1, -1, 511, 1024, 1025, -1}, torch::kInt32)};
  for (int32_t repeat_index = 0; repeat_index < 3; ++repeat_index) {
    SCOPED_TRACE(repeat_index);
    const NextDraftInputs actual =
        prepare_next_draft(accepted, bases, tables, 999, 128, 2);
    ASSERT_TRUE(compare_next_draft_inputs(actual, expected));
  }
  constexpr std::array<const char*, 8> kFieldNames = {"accepted_counts",
                                                      "anchor_tokens",
                                                      "query_tokens",
                                                      "query_positions",
                                                      "query_slots",
                                                      "kv_lengths",
                                                      "context_positions",
                                                      "context_slots"};
  for (size_t field_index = 0; field_index < kFieldNames.size();
       ++field_index) {
    SCOPED_TRACE(kFieldNames[field_index]);
    NextDraftInputs corrupted =
        prepare_next_draft(accepted, bases, tables, 999, 128, 2);
    const std::array<torch::Tensor*, 8> fields = {&corrupted.accepted_counts,
                                                  &corrupted.anchor_tokens,
                                                  &corrupted.query_tokens,
                                                  &corrupted.query_positions,
                                                  &corrupted.query_slots,
                                                  &corrupted.kv_lengths,
                                                  &corrupted.context_positions,
                                                  &corrupted.context_slots};
    *fields[field_index] = fields[field_index]->clone();
    fields[field_index]->reshape({-1})[0].add_(1);
    const testing::AssertionResult mismatch =
        compare_next_draft_inputs(corrupted, expected);
    EXPECT_FALSE(mismatch);
    EXPECT_STREQ(mismatch.message(), kFieldNames[field_index]);
  }
  NextDraftInputs wrong_dtype =
      prepare_next_draft(accepted, bases, tables, 999, 128, 2);
  wrong_dtype.query_slots = wrong_dtype.query_slots.to(torch::kLong);
  const testing::AssertionResult dtype_mismatch =
      compare_next_draft_inputs(wrong_dtype, expected);
  EXPECT_FALSE(dtype_mismatch);
  EXPECT_STREQ(dtype_mismatch.message(), "query_slots");
}

TEST(DFlashAsyncInputBuilderTest, RejectsInvalidBatchGeometry) {
  const torch::Tensor accepted = torch::tensor({{0, -1}}, torch::kLong);
  const torch::Tensor bases = torch::tensor({0}, torch::kLong);
  const torch::Tensor tables = torch::tensor({{7, 2}}, torch::kInt32);
  EXPECT_DEATH(prepare_context_kv(accepted.reshape({2}), bases, tables, 128),
               "Check failed");
  EXPECT_DEATH(prepare_context_kv(accepted, bases.reshape({1, 1}), tables, 128),
               "Check failed");
  EXPECT_DEATH(prepare_context_kv(accepted, bases, tables.reshape({2}), 128),
               "Check failed");
  EXPECT_DEATH(prepare_context_kv(accepted.repeat({2, 1}), bases, tables, 128),
               "Check failed");
  EXPECT_DEATH(
      prepare_next_draft(accepted, bases, tables.repeat({2, 1}), 999, 128, 15),
      "Check failed");
}

TEST(DFlashAsyncInputBuilderTest, RejectsOutOfRangeValidContextAndQuerySlots) {
  const torch::Tensor accepted = torch::tensor({{0, -1}}, torch::kLong);
  const torch::Tensor tables = torch::tensor({{7}}, torch::kInt32);
  EXPECT_ANY_THROW(prepare_context_kv(
      accepted, torch::tensor({-1}, torch::kLong), tables, 128));
  EXPECT_ANY_THROW(prepare_context_kv(
      accepted, torch::tensor({128}, torch::kLong), tables, 128));
  EXPECT_ANY_THROW(prepare_next_draft(
      accepted, torch::tensor({127}, torch::kLong), tables, 999, 128, 15));
}

TEST(DFlashAsyncInputBuilderTest, ContextKvUsesOnlyContiguousAcceptedPrefixes) {
  torch::Tensor accepted_tokens = torch::tensor({{0, -1, 77, 88},
                                                 {31, 0, -1, 99},
                                                 {0, 42, 43, -1},
                                                 {51, 52, 0, 54},
                                                 {-1, 0, 63, 64}},
                                                torch::kLong);
  torch::Tensor base_positions = torch::tensor({0, 3, 6, 7, 8}, torch::kInt32);
  torch::Tensor block_table = torch::tensor(
      {{3, 7, 2}, {5, 2, 8}, {9, 4, 6}, {8, 3, 1}, {4, 5, 7}}, torch::kInt32);
  constexpr int64_t kBlockSize = 4;
  ContextKvInputs result = prepare_context_kv(
      accepted_tokens, base_positions, block_table, kBlockSize);
  torch::Tensor expected_positions = torch::tensor(
      {{0, 1, 2, 3}, {3, 4, 5, 6}, {6, 7, 8, 9}, {7, 8, 9, 10}, {8, 9, 10, 11}},
      torch::kLong);
  torch::Tensor expected_slots = torch::tensor({{12, -1, -1, -1},
                                                {23, 8, -1, -1},
                                                {18, 19, 24, -1},
                                                {15, 4, 5, 6},
                                                {-1, -1, -1, -1}},
                                               torch::kInt32);
  EXPECT_TRUE(torch::equal(result.positions, expected_positions.reshape({-1})));
  EXPECT_TRUE(torch::equal(result.slots, expected_slots.reshape({-1})));
  EXPECT_EQ(result.positions.scalar_type(), torch::kLong);
  EXPECT_EQ(result.slots.scalar_type(), torch::kInt32);
  EXPECT_EQ(result.positions.device(), base_positions.device());
  EXPECT_EQ(result.slots.device(), block_table.device());
}

TEST(DFlashAsyncInputBuilderTest, ContextKvDoesNotRequireNextQueryBlocks) {
  constexpr int64_t kAcceptedWidth = 16;
  constexpr int64_t kBlockSize = 128;
  torch::Tensor accepted_tokens =
      torch::arange(kAcceptedWidth, torch::kLong).reshape({1, kAcceptedWidth});
  torch::Tensor base_positions = torch::tensor({112}, torch::kLong);
  torch::Tensor block_table = torch::tensor({{8}}, torch::kInt32);
  ContextKvInputs result = prepare_context_kv(
      accepted_tokens, base_positions, block_table, kBlockSize);
  EXPECT_TRUE(
      torch::equal(result.positions, torch::arange(112, 128, torch::kLong)));
  EXPECT_TRUE(
      torch::equal(result.slots, torch::arange(1136, 1152, torch::kInt32)));
}

TEST(DFlashAsyncInputBuilderTest, ContextKvDoesNotMapRejectedTailPositions) {
  torch::Tensor accepted_tokens = torch::tensor(
      {{0, -1, 42, -1}, {0, 33, -1, 77}, {-1, 0, 99, -1}}, torch::kLong);
  torch::Tensor base_positions = torch::tensor({127, 126, 128}, torch::kLong);
  torch::Tensor block_table = torch::tensor({{8}, {3}, {9}}, torch::kInt32);
  constexpr int64_t kBlockSize = 128;
  ContextKvInputs result = prepare_context_kv(
      accepted_tokens, base_positions, block_table, kBlockSize);
  torch::Tensor expected_positions = torch::tensor(
      {{127, 128, 129, 130}, {126, 127, 128, 129}, {128, 129, 130, 131}},
      torch::kLong);
  torch::Tensor expected_slots =
      torch::tensor({{1151, -1, -1, -1}, {510, 511, -1, -1}, {-1, -1, -1, -1}},
                    torch::kInt32);
  EXPECT_TRUE(torch::equal(result.positions, expected_positions.reshape({-1})));
  EXPECT_TRUE(torch::equal(result.slots, expected_slots.reshape({-1})));
}

TEST(DFlashAsyncInputBuilderTest, ContextKvDoesNotClampAcceptedPositions) {
  torch::Tensor accepted_tokens = torch::tensor({{0, -1}}, torch::kLong);
  torch::Tensor base_positions = torch::tensor({128}, torch::kLong);
  torch::Tensor block_table = torch::tensor({{8}}, torch::kInt32);
  constexpr int64_t kBlockSize = 128;
  EXPECT_ANY_THROW(prepare_context_kv(
      accepted_tokens, base_positions, block_table, kBlockSize));
}

TEST(DFlashAsyncInputBuilderTest, ContextKvRejectsNonPositiveBlockSize) {
  torch::Tensor accepted_tokens = torch::tensor({{0, -1}}, torch::kLong);
  torch::Tensor base_positions = torch::tensor({0}, torch::kLong);
  torch::Tensor block_table = torch::tensor({{0}}, torch::kInt32);
  for (const int64_t block_size : {0, -1}) {
    EXPECT_DEATH(prepare_context_kv(
                     accepted_tokens, base_positions, block_table, block_size),
                 "block_size");
  }
}

TEST(DFlashAsyncInputBuilderTest,
     MultiTableContextAndDraftUseCircularSwaSlots) {
  constexpr int64_t kBlockSize = 4;
  constexpr int64_t kAcceptedWidth = 16;
  constexpr int64_t kQueryWidth = 15;
  std::vector<torch::Tensor> multi_block_tables = {
      torch::tensor({{7, 3}}, torch::kInt32).repeat({kAcceptedWidth, 1}),
      torch::tensor({{11, 17, 23}}, torch::kInt32).repeat({kAcceptedWidth, 1})};
  DraftCacheBlockTable table =
      prepare_draft_cache_block_table(torch::Tensor(),
                                      multi_block_tables,
                                      kAcceptedWidth,
                                      torch::Device(torch::kCPU));
  ASSERT_TRUE(table.is_circular);
  EXPECT_TRUE(torch::equal(table.blocks, multi_block_tables.front()));
  torch::Tensor bases = torch::arange(kAcceptedWidth, torch::kLong) * 7;
  torch::Tensor tokens =
      torch::arange(kAcceptedWidth * kAcceptedWidth, torch::kLong)
          .view({kAcceptedWidth, kAcceptedWidth});
  for (int64_t row = 0; row < kAcceptedWidth; ++row) {
    tokens[row].slice(0, row + 1).fill_(-1);
  }
  ContextKvInputs context =
      prepare_context_kv(tokens, bases, table, kBlockSize);
  NextDraftInputs next =
      prepare_next_draft(tokens, bases, table, 999, kBlockSize, kQueryWidth);
  EXPECT_TRUE(torch::equal(context.positions, next.context_positions));
  EXPECT_TRUE(torch::equal(context.slots, next.context_slots));
  for (int64_t row = 0; row < kAcceptedWidth; ++row) {
    for (int64_t offset = 0; offset < kAcceptedWidth; ++offset) {
      const int64_t position = row * 7 + offset;
      const int64_t page = position / kBlockSize % 2 == 0 ? 7 : 3;
      const int64_t expected_slot =
          offset <= row ? page * kBlockSize + position % kBlockSize : -1;
      EXPECT_EQ(context.slots[row * kAcceptedWidth + offset].item<int32_t>(),
                expected_slot);
      EXPECT_EQ(
          context.positions[row * kAcceptedWidth + offset].item<int64_t>(),
          position);
    }
    for (int64_t offset = 0; offset < kQueryWidth; ++offset) {
      const int64_t position = row * 7 + row + 1 + offset;
      const int64_t page = position / kBlockSize % 2 == 0 ? 7 : 3;
      EXPECT_EQ(next.query_slots[row * kQueryWidth + offset].item<int32_t>(),
                page * kBlockSize + position % kBlockSize);
    }
  }
  EXPECT_EQ(multi_block_tables.size(), 2);
  EXPECT_EQ(multi_block_tables[1][0][0].item<int32_t>(), 11);
}

TEST(DFlashAsyncInputBuilderTest, MultiTableIgnoresPaddedBatchRows) {
  std::vector<torch::Tensor> multi_block_tables = {
      torch::tensor({{7, 3}, {-1, -1}}, torch::kInt32),
      torch::tensor({{11}, {-1}}, torch::kInt32)};
  DraftCacheBlockTable table = prepare_draft_cache_block_table(
      torch::Tensor(), multi_block_tables, 1, torch::Device(torch::kCPU));
  EXPECT_EQ(table.blocks.size(0), 1);
  torch::Tensor tokens = torch::tensor({{0, -1, 42}}, torch::kLong);
  torch::Tensor bases = torch::tensor({7}, torch::kLong);
  ContextKvInputs context = prepare_context_kv(tokens, bases, table, 4);
  EXPECT_TRUE(torch::equal(context.slots, torch::tensor({15, -1, -1})));
  EXPECT_EQ(multi_block_tables.front().size(0), 2);
}

TEST(DFlashAsyncInputBuilderTest, SingleTableKeepsLinearAddressing) {
  torch::Tensor blocks = torch::tensor({{7, 3, 11}}, torch::kInt32);
  DraftCacheBlockTable table = prepare_draft_cache_block_table(
      blocks, {}, 1, torch::Device(torch::kCPU));
  EXPECT_FALSE(table.is_circular);
  EXPECT_EQ(table.blocks.data_ptr(), blocks.data_ptr());
  ContextKvInputs context =
      prepare_context_kv(torch::tensor({{42}}, torch::kLong),
                         torch::tensor({8}, torch::kLong),
                         table,
                         4);
  EXPECT_EQ(context.slots[0].item<int32_t>(), 44);
}

}  // namespace
}  // namespace xllm::dflash_async
