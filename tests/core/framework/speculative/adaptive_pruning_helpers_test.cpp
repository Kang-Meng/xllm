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

#include "framework/speculative/adaptive_pruning_helpers.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <vector>

namespace xllm::adaptive_pruning {
namespace {

// Builds a ForwardOutput whose logits / next_tokens are the cu-packed varlen
// target output for `per_seq_val_tokens`, with deterministic, position-encoded
// values so the scatter destination of every source row is checkable:
//   logits[r, :] == r  and  next_tokens[r] == 100 + r.
// next_tokens use ids >= 100 that are never -1, so any pad leakage is visible.
ForwardOutput make_varlen_output(const std::vector<int32_t>& per_seq_val_tokens,
                                 int32_t vocab_size) {
  int64_t total = 0;
  for (int32_t n : per_seq_val_tokens) {
    total += n;
  }
  ForwardOutput out;
  out.logits = torch::arange(total, torch::kFloat)
                   .unsqueeze(/*dim=*/1)
                   .expand({total, vocab_size})
                   .contiguous();
  out.sample_output.next_tokens = torch::arange(total, torch::kLong) + 100;
  return out;
}

}  // namespace

// Regression for the MTP variable-length validate padding token. A padded slot
// must carry the reject marker -1, never a real token id: validate() reads the
// padded next_tokens (bonus-token gather, greedy-accept compare) BEFORE
// apply_pruned_prefix_lengths masks trailing positions, so a legal id such as 0
// leaks a spurious token into non-uniform per-seq validation.
TEST(ScatterVarlenTargetOutput, PadsNextTokensWithRejectMarker) {
  const std::vector<int32_t> per_seq_val_tokens = {2, 3};
  ForwardOutput out = make_varlen_output(per_seq_val_tokens, /*vocab_size=*/4);

  scatter_varlen_target_output_to_dense(out,
                                        per_seq_val_tokens,
                                        /*batch_size=*/2,
                                        /*max_val_tokens=*/3,
                                        /*next_token_pad_value=*/-1);

  ASSERT_EQ(out.sample_output.next_tokens.numel(), 2 * 3);
  const torch::Tensor nt = out.sample_output.next_tokens.to(torch::kCPU);
  const int64_t* p = nt.data_ptr<int64_t>();
  // seq0 (2 real tokens): dense cols [0,1] real, col 2 padded.
  EXPECT_EQ(p[0], 100);
  EXPECT_EQ(p[1], 101);
  EXPECT_EQ(p[2], -1);  // <-- the regression: pad slot, not a real id.
  // seq1 (3 real tokens): dense cols [0,1,2] all real.
  EXPECT_EQ(p[3], 102);
  EXPECT_EQ(p[4], 103);
  EXPECT_EQ(p[5], 104);
}

// The pad value is a pure parameter of the scatter. Every padded slot of a
// short sequence must take it; guards against a regression back to id 0.
TEST(ScatterVarlenTargetOutput, PadValueFillsEveryPaddedSlot) {
  const std::vector<int32_t> per_seq_val_tokens = {1, 3};
  ForwardOutput out = make_varlen_output(per_seq_val_tokens, /*vocab_size=*/2);

  scatter_varlen_target_output_to_dense(out,
                                        per_seq_val_tokens,
                                        /*batch_size=*/2,
                                        /*max_val_tokens=*/3,
                                        /*next_token_pad_value=*/-1);

  const torch::Tensor nt = out.sample_output.next_tokens.to(torch::kCPU);
  const int64_t* p = nt.data_ptr<int64_t>();
  EXPECT_EQ(p[0], 100);  // seq0 only real token.
  EXPECT_EQ(p[1], -1);   // seq0 pad.
  EXPECT_EQ(p[2], -1);   // seq0 pad.
}

// Padded logits rows must be the strictly-rejecting -inf sentinel so a padded
// position can never be sampled or accepted, while real rows keep their source
// values scattered to the per-seq column layout.
TEST(ScatterVarlenTargetOutput, PadsLogitsWithRejectingSentinel) {
  const std::vector<int32_t> per_seq_val_tokens = {2, 3};
  ForwardOutput out = make_varlen_output(per_seq_val_tokens, /*vocab_size=*/4);

  scatter_varlen_target_output_to_dense(out,
                                        per_seq_val_tokens,
                                        /*batch_size=*/2,
                                        /*max_val_tokens=*/3,
                                        /*next_token_pad_value=*/-1);

  const torch::Tensor logits = out.logits.to(torch::kCPU);
  ASSERT_EQ(logits.size(0), 2 * 3);
  // Dense row 2 is seq0's pad row.
  EXPECT_LT(logits[2].max().item<float>(), -1e8f);
  // Real rows preserve source values: seq0 col0 <- src row 0 (== 0.0),
  // seq1 col0 <- src row 2 (== 2.0).
  EXPECT_FLOAT_EQ(logits[0][0].item<float>(), 0.0f);
  EXPECT_FLOAT_EQ(logits[3][0].item<float>(), 2.0f);
}

// Uniform-width input (rows already == batch_size * max_val_tokens) is a no-op.
TEST(ScatterVarlenTargetOutput, UniformWidthIsNoOp) {
  const std::vector<int32_t> per_seq_val_tokens = {3, 3};
  ForwardOutput out = make_varlen_output(per_seq_val_tokens, /*vocab_size=*/2);
  const torch::Tensor before = out.sample_output.next_tokens.clone();

  scatter_varlen_target_output_to_dense(out,
                                        per_seq_val_tokens,
                                        /*batch_size=*/2,
                                        /*max_val_tokens=*/3,
                                        /*next_token_pad_value=*/-1);

  EXPECT_TRUE(out.sample_output.next_tokens.equal(before));
}

}  // namespace xllm::adaptive_pruning