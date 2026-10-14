/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <memory>
#include <vector>

#include "models/llm/mlu/glm5_next.h"

namespace xllm::mlu::model {
class Glm5NextModelTestPeer final {
 public:
  static void prepare_metadata(const Glm5NextModelImpl& model,
                               ModelInputParams& params) {
    model.prepare_attention_metadata(params);
  }
};

namespace {

class Glm5NextMetadataTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const torch::Device device(torch::kCPU);
    group_ =
        std::make_unique<ProcessGroup>(/*rank=*/0, /*world_size=*/1, device);
    ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, group_.get());
    parallel.tp_group_ = group_.get();
    ModelArgs args;
    args.model_type("glm5_next")
        .n_layers(0)
        .hidden_size(4)
        .vocab_size(8)
        .hc_mult(1)
        .linear_num_key_heads(1);
    model_ = std::make_unique<Glm5NextModelImpl>(ModelContext(
        parallel,
        args,
        QuantArgs(),
        torch::TensorOptions().dtype(torch::kFloat32).device(device)));
  }

  ModelInputParams make_params() const {
    ModelInputParams params;
    params.meta.batch_forward_type = BatchForwardType::PREFILL;
    params.meta.num_sequences = 2;
    params.meta.q_max_seq_len = 1;
    params.meta.kv_max_seq_len = 5;
    params.attention.device.q_seq_lens =
        torch::tensor({0, 1, 2}, torch::kInt32);
    params.attention.device.kv_seq_lens =
        torch::tensor({0, 5, 10}, torch::kInt32);
    params.attention.device.new_cache_slots =
        torch::tensor({0, 1}, torch::kInt32);
    params.embedding.linear_state_ids = {2, 3};
    params.embedding.linear_state_indices =
        torch::tensor({2, 3}, torch::kInt32);
    params.embedding.linear_state_read_ids = {1, 1};
    params.linear_state_validity_mask = {1, 1};
    return params;
  }

  void prepare(ModelInputParams& params) const {
    Glm5NextModelTestPeer::prepare_metadata(*model_, params);
  }

  std::unique_ptr<ProcessGroup> group_;
  std::unique_ptr<Glm5NextModelImpl> model_;
};

TEST_F(Glm5NextMetadataTest,
       PreparesNewAndExistingMetadataWithoutMutatingCaller) {
  for (const bool existing : {false, true}) {
    for (const bool device_indices : {false, true}) {
      auto params = make_params();
      if (device_indices) {
        params.embedding.linear_state_read_indices =
            torch::tensor({1, 1}, torch::kInt32);
      }
      if (existing) {
        params.attn_metadata = std::make_shared<layer::AttentionMetadata>(
            layer::AttentionMetadataBuilder::build(params,
                                                   /*enable_mla=*/false));
      }
      const auto caller_metadata = params.attn_metadata;
      prepare(params);
      EXPECT_TRUE(torch::equal(params.attn_metadata->linear_state_read_indices,
                               torch::tensor({1, 1}, torch::kInt32)));
      EXPECT_TRUE(params.attn_metadata->chunk_indices.defined());
      if (device_indices) {
        EXPECT_EQ(params.attn_metadata->linear_state_read_indices.data_ptr(),
                  params.embedding.linear_state_read_indices.data_ptr());
      }
      if (existing) {
        EXPECT_NE(params.attn_metadata, caller_metadata);
        EXPECT_FALSE(caller_metadata->linear_state_read_indices.defined());
        EXPECT_EQ(params.attn_metadata->linear_state_indices.data_ptr(),
                  caller_metadata->linear_state_indices.data_ptr());
      }
    }
  }
}

TEST_F(Glm5NextMetadataTest, ReusedMetadataUsesCurrentSourceSlots) {
  auto params = make_params();
  prepare(params);
  const auto previous = params.attn_metadata;
  params.embedding.linear_state_read_ids = {4, 4};
  prepare(params);
  EXPECT_TRUE(torch::equal(params.attn_metadata->linear_state_read_indices,
                           torch::tensor({4, 4}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(previous->linear_state_read_indices,
                           torch::tensor({1, 1}, torch::kInt32)));
  const auto distinct = params.attn_metadata;
  params.embedding.linear_state_read_ids = params.embedding.linear_state_ids;
  prepare(params);
  EXPECT_FALSE(params.attn_metadata->linear_state_read_indices.defined());
  EXPECT_TRUE(distinct->linear_state_read_indices.defined());
}

TEST_F(Glm5NextMetadataTest,
       DecodeAndVerifyClearStaleSourcesWithoutChangingCaller) {
  for (const BatchForwardType type : {BatchForwardType::DECODE,
                                      BatchForwardType::MIXED,
                                      BatchForwardType::CHUNKED_PREFILL}) {
    auto params = make_params();
    prepare(params);
    params.meta.batch_forward_type = type;
    params.is_spec_verify = type.is_chunked_prefill();
    params.attn_metadata->is_prefill = false;
    params.attn_metadata->is_chunked_prefill = params.is_spec_verify;
    params.attn_metadata->is_spec_verify = params.is_spec_verify;
    const auto previous = params.attn_metadata;
    prepare(params);
    EXPECT_FALSE(params.attn_metadata->linear_state_read_indices.defined());
    EXPECT_TRUE(previous->linear_state_read_indices.defined());
    const auto canonical = params.attn_metadata;
    prepare(params);
    EXPECT_EQ(params.attn_metadata, canonical);
  }
}

TEST_F(Glm5NextMetadataTest, RejectsMalformedSourceIndices) {
  auto params = make_params();
  params.embedding.linear_state_read_indices =
      torch::tensor({1}, torch::kInt32);
  EXPECT_DEATH(prepare(params), "source index count");
  params.embedding.linear_state_read_indices =
      torch::tensor({1, 1}, torch::kInt32).view({1, 2});
  EXPECT_DEATH(prepare(params), "source indices must be 1D");
  params.embedding.linear_state_read_indices =
      torch::tensor({1, 1}, torch::kFloat32);
  EXPECT_DEATH(prepare(params), "source indices must be integers");
}

TEST(Glm5NextMluDeathTest, RejectsNonSpanVerificationBeforeUsingInputs) {
  const torch::Device device(torch::kCPU);
  ProcessGroup group(/*rank=*/0, /*world_size=*/1, device);
  ParallelArgs parallel_args(/*rank=*/0, /*world_size=*/1, &group);
  parallel_args.tp_group_ = &group;
  ModelArgs args;
  args.model_type("glm5_next");
  args.n_layers(0);
  args.hidden_size(4);
  args.vocab_size(8);
  args.hc_mult(1);
  args.linear_num_key_heads(1);
  const ModelContext context(
      parallel_args,
      args,
      QuantArgs(),
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  Glm5NextModelImpl model(context);
  std::vector<KVCache> caches;

  // Undefined inputs must never reach embedding, metadata building, or layers.
  const torch::Tensor tokens;
  const torch::Tensor positions;
  ModelInputParams params;
  params.is_spec_verify = true;
  EXPECT_DEATH(model.forward(tokens, positions, caches, params),
               "requires chunked-prefill");

  params.is_spec_verify = false;
  params.attn_metadata = std::make_shared<layer::AttentionMetadata>();
  params.attn_metadata->is_spec_verify = true;
  EXPECT_DEATH(model.forward(tokens, positions, caches, params),
               "requires chunked-prefill");
}

}  // namespace
}  // namespace xllm::mlu::model
