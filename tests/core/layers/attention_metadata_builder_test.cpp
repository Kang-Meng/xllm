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

#include "core/layers/common/attention_metadata_builder.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>

#include "core/framework/batch/batch_forward_type.h"
#include "core/framework/model/model_input_params.h"
#include "core/layers/common/attention_metadata.h"
#include "core/platform/device.h"

namespace xllm::layer {
namespace {

ModelInputParams make_params() {
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::PREFILL;
  params.meta.num_sequences = 3;
  params.meta.q_max_seq_len = 1;
  params.meta.kv_max_seq_len = 8;
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  params.attention.device.q_seq_lens = torch::tensor({1, 1, 1}, options);
  params.attention.device.kv_seq_lens = torch::tensor({1, 4, 8}, options);
  params.attention.device.q_cu_seq_lens = torch::tensor({1, 2, 3}, options);
  params.attention.device.kv_cache_tokens_nums =
      torch::tensor({7, 0, 5}, options);
  params.embedding.linear_state_ids = {4, 2, 9};
  params.embedding.linear_state_indices = torch::tensor({4, 2, 9}, options);
  params.linear_state_validity_mask = {0, 1, 0};
  return params;
}

class DummyAttentionMetadataTest : public ::testing::TestWithParam<bool> {};

TEST_P(DummyAttentionMetadataTest, ConstantsMatchLegacyValues) {
  torch::Device destination(torch::kCPU);
#if defined(USE_NPU)
  if (GetParam()) {
    Device device(0);
    device.set_device();
    device.init_device_context();
    destination = device.unwrap();
  }
#endif
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  auto metadata =
      AttentionMetadataBuilder::build(params, false, {}, destination);
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kInt32).device(destination);
  AttentionMetadata expected;
  expected.slot_mapping = torch::tensor({0}, options);
  expected.q_cu_seq_lens = torch::tensor({0, 1}, options);
  expected.kv_cu_seq_lens = torch::tensor({0, 1}, options);
  expected.q_seq_lens = torch::tensor({1}, options);
  expected.kv_seq_lens = torch::tensor({1}, options);
  expected.paged_kv_indptr = torch::tensor({0, 1}, options);
  expected.paged_kv_indices = torch::tensor({0}, options);
  expected.paged_kv_last_page_len = torch::tensor({1}, options);
  for (const auto member : {&AttentionMetadata::slot_mapping,
                            &AttentionMetadata::q_cu_seq_lens,
                            &AttentionMetadata::kv_cu_seq_lens,
                            &AttentionMetadata::q_seq_lens,
                            &AttentionMetadata::kv_seq_lens,
                            &AttentionMetadata::paged_kv_indptr,
                            &AttentionMetadata::paged_kv_indices,
                            &AttentionMetadata::paged_kv_last_page_len}) {
    const torch::Tensor& actual_tensor = metadata.*member;
    const torch::Tensor& expected_tensor = expected.*member;
    EXPECT_EQ(actual_tensor.scalar_type(), expected_tensor.scalar_type());
    EXPECT_EQ(actual_tensor.device(), expected_tensor.device());
    EXPECT_EQ(actual_tensor.sizes(), expected_tensor.sizes());
    EXPECT_TRUE(torch::equal(actual_tensor.cpu(), expected_tensor.cpu()));
  }
}

INSTANTIATE_TEST_SUITE_P(SupportedDevices,
                         DummyAttentionMetadataTest,
#if defined(USE_NPU)
                         ::testing::Values(false, true)
#else
                         ::testing::Values(false)
#endif
);

TEST(AttentionMetadataBuilderTest, LinearPrefillSchedulesRespectChunkSize) {
  for (const int32_t block_size : {16, 64}) {
    for (const bool chunked : {false, true}) {
      AttentionMetadata metadata;
      metadata.is_prefill = !chunked;
      metadata.is_chunked_prefill = chunked;
      metadata.q_cu_seq_lens =
          torch::tensor({0, block_size, 2 * block_size + 1}, torch::kInt32);
      AttentionMetadataBuilder::build_linear_prefill(metadata, block_size);
      EXPECT_EQ(metadata.tot, 3);
      EXPECT_TRUE(torch::equal(metadata.batch.narrow(0, 0, 3),
                               torch::tensor({0, 1, 1}, torch::kInt32)));
      EXPECT_TRUE(torch::equal(metadata.token_block_offset.narrow(0, 0, 3),
                               torch::tensor({0, 0, 1}, torch::kInt32)));
      EXPECT_TRUE(
          torch::equal(metadata.chunk_indices,
                       torch::tensor({{0, 0}, {1, 0}, {1, 1}}, torch::kInt32)));
      EXPECT_TRUE(metadata.batch.slice(0, 3).eq(-1).all().item<bool>());
      EXPECT_TRUE(
          metadata.token_block_offset.slice(0, 3).eq(-1).all().item<bool>());
    }
  }
}

TEST(AttentionMetadataBuilderTest, LinearDecodeDoesNotRequirePrefillLengths) {
  AttentionMetadata metadata;
  metadata.is_prefill = false;
  metadata.is_chunked_prefill = false;
  AttentionMetadataBuilder::build_linear_prefill(metadata, /*block_size=*/64);
  EXPECT_FALSE(metadata.batch.defined());
  EXPECT_FALSE(metadata.chunk_indices.defined());
}

TEST(AttentionMetadataBuilderTest, GraphCapacityPreservesBatchHistory) {
  ModelInputParams params = make_params();
  EXPECT_EQ(
      AttentionMetadataBuilder::build(params, /*enable_mla=*/false).max_seq_len,
      8);

  params.graph.kv_seq_len_capacity = 8192;
  params = params.to(torch::kCPU);
  for (const int32_t history : {16, 4097, 8192, 16}) {
    params.meta.kv_max_seq_len = history;
    const auto metadata =
        AttentionMetadataBuilder::build(params, /*enable_mla=*/false);
    EXPECT_EQ(metadata.max_seq_len, 8192);
    EXPECT_EQ(params.meta.kv_max_seq_len, history);
  }
}

TEST(AttentionMetadataBuilderTest, MaterializesCanonicalInitialStateMask) {
  ModelInputParams params = make_params();

  AttentionMetadata metadata =
      AttentionMetadataBuilder::build(params, /*enable_mla=*/false);

  ASSERT_TRUE(metadata.has_initial_states.defined());
  EXPECT_EQ(metadata.has_initial_states.scalar_type(), torch::kBool);
  EXPECT_EQ(metadata.has_initial_states.device(), torch::Device(torch::kCPU));
  EXPECT_TRUE(torch::equal(metadata.has_initial_states,
                           torch::tensor({false, true, false}, torch::kBool)));
}

TEST(AttentionMetadataBuilderTest, LeavesSourceStatePolicyToModel) {
  ModelInputParams params = make_params();
  params.embedding.linear_state_read_ids = {1, 1, 9};
  params.embedding.linear_state_read_indices =
      torch::tensor({1, 1, 9}, torch::kInt32);
  const AttentionMetadata metadata =
      AttentionMetadataBuilder::build(params, /*enable_mla=*/false);
  EXPECT_FALSE(metadata.linear_state_read_indices.defined());
  EXPECT_TRUE(torch::equal(metadata.linear_state_indices,
                           params.embedding.linear_state_indices));
}

TEST(AttentionMetadataBuilderTest, DoesNotDeriveValidityFromContextOrSlot) {
  ModelInputParams params = make_params();
  params.embedding.linear_state_indices = torch::tensor({0, 0, 0}, torch::kInt);
  params.attention.device.kv_cache_tokens_nums =
      torch::tensor({9, 0, 11}, torch::kInt);
  params.linear_state_validity_mask = {0, 1, 0};

  AttentionMetadata metadata =
      AttentionMetadataBuilder::build(params, /*enable_mla=*/false);

  EXPECT_TRUE(torch::equal(metadata.has_initial_states,
                           torch::tensor({false, true, false}, torch::kBool)));
}

TEST(AttentionMetadataBuilderTest, MaterializesChunkedPrefillMask) {
  ModelInputParams params = make_params();
  params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;

  AttentionMetadata metadata =
      AttentionMetadataBuilder::build(params, /*enable_mla=*/false);

  ASSERT_TRUE(metadata.has_initial_states.defined());
  EXPECT_TRUE(torch::equal(metadata.has_initial_states,
                           torch::tensor({false, true, false}, torch::kBool)));
}

TEST(AttentionMetadataBuilderTest, SkipsInitialStateMaterializationOnRequest) {
  ModelInputParams params = make_params();
  AttentionMetadataBuildOptions build_options;
  build_options.materialize_linear_state_validity = false;

  AttentionMetadata metadata = AttentionMetadataBuilder::build(
      params, /*enable_mla=*/false, {}, std::nullopt, build_options);

  EXPECT_FALSE(metadata.has_initial_states.defined());
  EXPECT_EQ(params.linear_state_validity_mask,
            LinearStateValidityMask({0, 1, 0}));
  EXPECT_EQ(params.embedding.linear_state_ids, std::vector<int32_t>({4, 2, 9}));
}

TEST(AttentionMetadataBuilderTest,
     SkippingInitialStateMaterializationKeepsValidation) {
  ModelInputParams params = make_params();
  params.linear_state_validity_mask.pop_back();
  AttentionMetadataBuildOptions build_options;
  build_options.materialize_linear_state_validity = false;

  EXPECT_DEATH(
      AttentionMetadataBuilder::build(
          params, /*enable_mla=*/false, {}, std::nullopt, build_options),
      "linear state mask row count mismatch");
}

TEST(AttentionMetadataBuilderTest, IgnoresTransportIdsWithoutLinearStateOps) {
  ModelInputParams params = make_params();
  params.embedding.linear_state_ids = {-1, -1, -1};
  params.embedding.linear_state_indices =
      torch::tensor({-1, -1, -1}, torch::kInt);
  params.linear_state_validity_mask.clear();

  AttentionMetadata metadata =
      AttentionMetadataBuilder::build(params, /*enable_mla=*/false);

  EXPECT_FALSE(metadata.has_initial_states.defined());
}

TEST(AttentionMetadataBuilderTest, MaterializesColdMaskForDummyShard) {
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  params.meta.num_sequences = 0;
  params.meta.q_max_seq_len = 0;
  params.meta.kv_max_seq_len = 0;

  AttentionMetadata metadata =
      AttentionMetadataBuilder::build(params,
                                      /*enable_mla=*/false,
                                      /*attn_mask=*/{},
                                      torch::Device(torch::kCPU));

  ASSERT_TRUE(metadata.is_dummy);
#if defined(USE_NPU)
  EXPECT_EQ(metadata.q_seq_lens_vec, std::vector<int32_t>({1}));
  EXPECT_EQ(metadata.kv_seq_lens_vec, std::vector<int32_t>({1}));
#else
  EXPECT_EQ(metadata.q_seq_lens_vec, std::vector<int32_t>({0, 1}));
  EXPECT_EQ(metadata.kv_seq_lens_vec, std::vector<int32_t>({0, 1}));
#endif
  EXPECT_TRUE(
      torch::equal(metadata.q_seq_lens, torch::tensor({1}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(metadata.kv_seq_lens, torch::tensor({1}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(metadata.q_cu_seq_lens,
                           torch::tensor({0, 1}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(metadata.kv_cu_seq_lens,
                           torch::tensor({0, 1}, torch::kInt32)));
  ASSERT_TRUE(metadata.has_initial_states.defined());
  EXPECT_TRUE(
      torch::equal(metadata.has_initial_states, torch::tensor({false})));
}

#if defined(USE_NPU)
TEST(AttentionMetadataBuilderTest, PreservesMaterializedDummyQueryGeometry) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  for (const bool decode_rows : {false, true}) {
    constexpr int32_t kQueryWidth = 4;
    const int32_t row_count = decode_rows ? kQueryWidth : 1;
    const int32_t query_len = decode_rows ? 1 : kQueryWidth;
    ModelInputParams params;
    params.meta.batch_forward_type = decode_rows
                                         ? BatchForwardType::DECODE
                                         : BatchForwardType::CHUNKED_PREFILL;
    params.meta.num_sequences = 0;
    params.meta.q_max_seq_len = query_len;
    params.meta.kv_max_seq_len = query_len;
    params.execution_batch.num_tokens = kQueryWidth;
    params.attention.host.q_seq_lens.assign(row_count, query_len);
    for (int32_t row = 0; row <= row_count; ++row) {
      params.attention.host.q_cu_seq_lens.emplace_back(row * query_len);
    }
    params.attention.host.kv_seq_lens.assign(row_count, query_len);
    params.attention.host.new_cache_slots.assign(kQueryWidth, -1);
    params.attention.host.block_tables =
        torch::zeros({row_count, query_len}, torch::kInt32);
    params.attention.rebuild_device_buffer(device.unwrap());

    const AttentionMetadata metadata =
        AttentionMetadataBuilder::build(params, false);
    EXPECT_TRUE(metadata.is_dummy);
    EXPECT_EQ(metadata.is_chunked_prefill, !decode_rows);
    EXPECT_EQ(metadata.max_query_len, query_len);
    EXPECT_EQ(metadata.q_seq_lens.numel(), row_count);
    EXPECT_EQ(metadata.kv_seq_lens.numel(), row_count);
    EXPECT_EQ(metadata.q_cu_seq_lens.device(), device.unwrap());
    EXPECT_EQ(metadata.slot_mapping.device(), device.unwrap());
    EXPECT_TRUE(torch::equal(
        metadata.q_cu_seq_lens.cpu(),
        torch::tensor(params.attention.host.q_cu_seq_lens, torch::kInt32)));
    EXPECT_TRUE(torch::equal(metadata.slot_mapping.cpu(),
                             torch::full({kQueryWidth}, -1, torch::kInt32)));
    EXPECT_EQ(metadata.block_table.size(0), row_count);
    EXPECT_EQ(metadata.block_table.size(1), query_len);
    EXPECT_EQ(params.execution_batch.num_reqs, 0);
    EXPECT_TRUE(params.execution_batch.num_scheduled_tokens.empty());

    params.meta.num_sequences = 1;
    EXPECT_FALSE(AttentionMetadataBuilder::build(params, false).is_dummy);
  }
}
#endif

#if defined(USE_MUSA)
TEST(AttentionMetadataBuilderTest, BuildsMusaMetadataWithCommonBuilder) {
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.meta.num_sequences = 2;
  params.meta.q_max_seq_len = 1;
  params.meta.kv_max_seq_len = 7;
  params.attention.host.q_seq_lens = {1, 1};
  params.attention.host.kv_seq_lens = {3, 7};

  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  params.attention.device.q_seq_lens = torch::tensor({1, 1}, options);
  params.attention.device.kv_seq_lens = torch::tensor({3, 7}, options);
  params.attention.device.q_cu_seq_lens = torch::tensor({1, 2}, options);
  params.attention.device.block_tables =
      torch::tensor({{0, -1}, {1, 2}}, options);
  params.attention.device.paged_kv_indptr = torch::tensor({0, 1, 3}, options);
  params.attention.device.paged_kv_indices = torch::tensor({0, 1, 2}, options);
  params.attention.device.paged_kv_last_page_len =
      torch::tensor({3, 7}, options);

  params.attn_metadata = std::make_shared<AttentionMetadata>();
  params.attn_metadata->fa3_metadata.share_fa3_scheduler_metadata = true;
  params.attn_metadata->fa3_metadata.fa3_scheduler_metadata =
      torch::tensor({4, 3, 2, 1}, options);

  AttentionMetadata metadata =
      AttentionMetadataBuilder::build(params, /*enable_mla=*/false);

  EXPECT_TRUE(torch::equal(metadata.q_cu_seq_lens,
                           torch::tensor({0, 1, 2}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(metadata.kv_cu_seq_lens,
                           torch::tensor({0, 3, 10}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(metadata.q_seq_lens, torch::tensor({1, 1}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(metadata.kv_seq_lens, torch::tensor({3, 7}, torch::kInt32)));
  EXPECT_EQ(metadata.block_table.scalar_type(), torch::kInt32);
  EXPECT_EQ(metadata.fa3_metadata.paged_kv_indptr_host.device(),
            torch::Device(torch::kCPU));
  EXPECT_EQ(metadata.fa3_metadata.paged_kv_indptr_host.scalar_type(),
            torch::kInt32);
  EXPECT_EQ(metadata.fa3_metadata.paged_kv_indices_host.scalar_type(),
            torch::kInt32);
  EXPECT_EQ(metadata.fa3_metadata.paged_kv_last_page_len_host.scalar_type(),
            torch::kInt32);
  EXPECT_TRUE(metadata.fa3_metadata.share_fa3_scheduler_metadata);
  EXPECT_TRUE(
      torch::equal(metadata.fa3_metadata.fa3_scheduler_metadata,
                   params.attn_metadata->fa3_metadata.fa3_scheduler_metadata));
}
#endif

}  // namespace
}  // namespace xllm::layer
