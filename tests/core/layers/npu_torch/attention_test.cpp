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

#include "core/layers/npu_torch/attention.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/layers/common/attention_metadata.h"

namespace xllm::layer {
namespace {

TEST(AttentionTest, JevDummyPrefillSkipsValidationAndCacheAccess) {
  AttentionImpl attention(/*num_heads=*/1,
                          /*head_size=*/8,
                          /*scale=*/1.F,
                          /*num_kv_heads=*/1,
                          /*sliding_window=*/-1,
                          /*enable_fia_decode=*/false,
                          /*dcp_group=*/nullptr,
                          /*bidirectional_prompt=*/true);
  AttentionMetadata metadata;
  metadata.is_prefill = true;
  metadata.is_dummy = true;
  torch::Tensor query = torch::zeros({1, 1, 8});
  torch::Tensor key;
  torch::Tensor value;
  KVCache kv_cache;

  const auto [output, output_lse] =
      attention.forward(metadata, query, key, value, kv_cache);

  EXPECT_EQ(output.sizes(), query.sizes());
  EXPECT_EQ(output.scalar_type(), query.scalar_type());
  EXPECT_EQ(output.device(), query.device());
  EXPECT_FALSE(output_lse.has_value());
}

TEST(AttentionTest, JevRealPrefillStillRejectsMissingLengths) {
  AttentionImpl attention(/*num_heads=*/1,
                          /*head_size=*/8,
                          /*scale=*/1.F,
                          /*num_kv_heads=*/1,
                          /*sliding_window=*/-1,
                          /*enable_fia_decode=*/false,
                          /*dcp_group=*/nullptr,
                          /*bidirectional_prompt=*/true);
  AttentionMetadata metadata;
  metadata.is_prefill = true;
  metadata.is_dummy = false;
  metadata.is_chunked_prefill = false;
  metadata.is_mixed = false;
  metadata.is_spec_verify = false;
  torch::Tensor query = torch::zeros({1, 1, 8});
  torch::Tensor key;
  torch::Tensor value;
  KVCache kv_cache;

  EXPECT_DEATH(
      attention.forward(metadata, query, key, value, kv_cache),
      "JEV native attention requires host cumulative query and KV lengths");
}

}  // namespace
}  // namespace xllm::layer
