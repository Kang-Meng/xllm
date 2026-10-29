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

#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "kernels/mlu/mlu_ops_api.h"
#include "triton_jit/include/spec.h"

namespace xllm {
namespace {

using xllm::kernel::mlu::fused_sigmoid_gating_delta_rule_update;

torch::Tensor kda_reference_step(torch::Tensor query,
                                 torch::Tensor key,
                                 const torch::Tensor& value,
                                 const torch::Tensor& gate_logits,
                                 const torch::Tensor& beta_logits,
                                 const torch::Tensor& a_log,
                                 const torch::Tensor& dt_bias,
                                 torch::Tensor& state,
                                 double scale,
                                 float gate_lower_bound) {
  query =
      query *
      torch::rsqrt((query * query).sum(/*dim=*/-1, /*keepdim=*/true) + 1e-6) *
      scale;
  key =
      key * torch::rsqrt((key * key).sum(/*dim=*/-1, /*keepdim=*/true) + 1e-6);
  torch::Tensor gate = torch::exp(
      gate_lower_bound * torch::sigmoid(torch::exp(a_log).unsqueeze(/*dim=*/1) *
                                        (gate_logits + dt_bias)));
  state = state * gate.unsqueeze(/*dim=*/1);
  torch::Tensor residual =
      value - (state * key.unsqueeze(/*dim=*/1)).sum(/*dim=*/-1);
  residual = residual * torch::sigmoid(beta_logits).unsqueeze(/*dim=*/1);
  state = state + residual.unsqueeze(/*dim=*/-1) * key.unsqueeze(/*dim=*/1);
  return (state * query.unsqueeze(/*dim=*/1)).sum(/*dim=*/-1);
}

void expect_kda_update_matches_reference(int64_t seq_len) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);

  constexpr int64_t kNumHeads = 8;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kStateIndex = 1;
  constexpr float kGateLowerBound = -5.0f;
  const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
  torch::manual_seed(20260902);

  torch::TensorOptions cpu_bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU);
  torch::TensorOptions cpu_fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  torch::TensorOptions cpu_int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

  torch::Tensor a_log_cpu = torch::randn({kNumHeads}, cpu_fp32_options) * 0.25;
  torch::Tensor a_cpu =
      (torch::randn({seq_len, kNumHeads * kHeadDim}, cpu_fp32_options) * 0.25)
          .to(cpu_bf16_options);
  torch::Tensor b_cpu =
      (torch::randn({seq_len, kNumHeads}, cpu_fp32_options) * 0.25)
          .to(cpu_bf16_options);
  torch::Tensor dt_bias_cpu =
      torch::randn({kNumHeads * kHeadDim}, cpu_fp32_options) * 0.1;
  torch::Tensor q_cpu =
      torch::randn({1, seq_len, kNumHeads, kHeadDim}, cpu_bf16_options);
  torch::Tensor k_cpu =
      torch::randn({1, seq_len, kNumHeads, kHeadDim}, cpu_bf16_options);
  torch::Tensor v_cpu =
      torch::randn({1, seq_len, kNumHeads, kHeadDim}, cpu_bf16_options);
  torch::Tensor initial_state_cpu =
      torch::randn({2, kNumHeads, kHeadDim, kHeadDim}, cpu_fp32_options) * 0.05;
  torch::Tensor state_indices_cpu =
      torch::full({1, seq_len}, kStateIndex, cpu_int_options);
  torch::Tensor cu_seqlens_cpu =
      torch::tensor({0, static_cast<int32_t>(seq_len)}, cpu_int_options);

  torch::Tensor expected_out_cpu = torch::empty_like(v_cpu);
  torch::Tensor expected_state_cpu = initial_state_cpu.clone();
  torch::Tensor state =
      expected_state_cpu.select(/*dim=*/0, /*index=*/kStateIndex).clone();
  torch::Tensor a_fp32 =
      a_cpu.to(torch::kFloat32).view({seq_len, kNumHeads, kHeadDim});
  torch::Tensor b_fp32 = b_cpu.to(torch::kFloat32);
  torch::Tensor dt_bias_fp32 = dt_bias_cpu.view({kNumHeads, kHeadDim});
  torch::Tensor q_fp32 = q_cpu.to(torch::kFloat32)
                             .select(/*dim=*/0,
                                     /*index=*/0);
  torch::Tensor k_fp32 = k_cpu.to(torch::kFloat32)
                             .select(/*dim=*/0,
                                     /*index=*/0);
  torch::Tensor v_fp32 = v_cpu.to(torch::kFloat32)
                             .select(/*dim=*/0,
                                     /*index=*/0);
  for (int64_t token_index = 0; token_index < seq_len; ++token_index) {
    torch::Tensor output =
        kda_reference_step(q_fp32.select(/*dim=*/0, token_index),
                           k_fp32.select(/*dim=*/0, token_index),
                           v_fp32.select(/*dim=*/0, token_index),
                           a_fp32.select(/*dim=*/0, token_index),
                           b_fp32.select(/*dim=*/0, token_index),
                           a_log_cpu,
                           dt_bias_fp32,
                           state,
                           scale,
                           kGateLowerBound);
    expected_out_cpu.select(/*dim=*/0, /*index=*/0)
        .select(/*dim=*/0, token_index)
        .copy_(output.to(torch::kBFloat16));
  }
  expected_state_cpu.select(/*dim=*/0, /*index=*/kStateIndex).copy_(state);

  torch::Tensor a_log = a_log_cpu.to(device);
  torch::Tensor a = a_cpu.to(device);
  torch::Tensor b = b_cpu.to(device);
  torch::Tensor dt_bias = dt_bias_cpu.to(device);
  torch::Tensor q = q_cpu.to(device);
  torch::Tensor k = k_cpu.to(device);
  torch::Tensor v = v_cpu.to(device);
  torch::Tensor initial_state = initial_state_cpu.to(device);
  torch::Tensor state_indices = state_indices_cpu.to(device);
  torch::Tensor cu_seqlens = cu_seqlens_cpu.to(device);

  auto [out, final_state] = fused_sigmoid_gating_delta_rule_update(
      a_log,
      a,
      b,
      dt_bias,
      q,
      k,
      v,
      initial_state,
      state_indices,
      cu_seqlens,
      scale,
      /*use_qk_l2norm_in_kernel=*/true,
      /*softplus_beta=*/1.0f,
      /*softplus_threshold=*/20.0f,
      /*num_accepted_tokens_opt=*/std::nullopt,
      /*inplace_final_state=*/true,
      /*is_kda=*/true,
      /*kda_use_safe_gate=*/true,
      kGateLowerBound);
  torch_mlu::synchronize();

  EXPECT_TRUE(torch::allclose(out.cpu().to(torch::kFloat32),
                              expected_out_cpu.to(torch::kFloat32),
                              /*rtol=*/2e-2,
                              /*atol=*/2e-2));
  EXPECT_TRUE(torch::allclose(final_state.cpu(),
                              expected_state_cpu,
                              /*rtol=*/2e-3,
                              /*atol=*/2e-3));
  EXPECT_TRUE(torch::equal(final_state.cpu().select(/*dim=*/0, /*index=*/0),
                           initial_state_cpu.select(/*dim=*/0, /*index=*/0)));
}

void expect_sparse_kda_matches_reference(
    const std::vector<int32_t>& sequence_lengths,
    const std::vector<int32_t>& accepted_counts,
    bool inplace_final_state,
    bool include_invalid_slots,
    int64_t checkpoint_capacity = 4,
    int64_t num_heads = 8,
    int64_t padded_token_count = 0) {
  ASSERT_EQ(sequence_lengths.size(), accepted_counts.size());
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  torch::manual_seed(20260908);

  const int64_t kNumHeads = num_heads;
  constexpr int64_t kHeadDim = 128;
  constexpr float kGateLowerBound = -5.0f;
  const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
  const int64_t batch_size = static_cast<int64_t>(sequence_lengths.size());
  const int64_t state_slots = batch_size * checkpoint_capacity + 1;
  std::vector<int32_t> offsets;
  offsets.reserve(sequence_lengths.size() + 1);
  offsets.emplace_back(0);
  for (int32_t length : sequence_lengths) {
    offsets.emplace_back(offsets.back() + length);
  }
  const int64_t total_tokens = offsets.back();
  const int64_t allocated_tokens =
      padded_token_count > 0 ? padded_token_count : total_tokens;
  ASSERT_GE(allocated_tokens, total_tokens);
  auto fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto bf16_options = fp32_options.dtype(torch::kBFloat16);
  auto int_options = fp32_options.dtype(torch::kInt32);
  auto a_log_cpu = torch::randn({kNumHeads}, fp32_options) * 0.25;
  auto dt_bias_cpu = torch::randn({kNumHeads * kHeadDim}, fp32_options) * 0.1;
  auto a_cpu =
      torch::randn({allocated_tokens, kNumHeads * kHeadDim}, bf16_options);
  auto b_cpu = torch::randn({allocated_tokens, kNumHeads}, bf16_options);
  auto q_cpu =
      torch::randn({1, allocated_tokens, kNumHeads, kHeadDim}, bf16_options);
  auto k_cpu = torch::randn_like(q_cpu);
  auto v_cpu = torch::randn_like(q_cpu);
  auto initial_state_cpu =
      torch::randn({state_slots, kNumHeads, kHeadDim, kHeadDim}, fp32_options) *
      0.05;
  auto state_indices_cpu = torch::arange(1, state_slots, int_options)
                               .view({batch_size, checkpoint_capacity});
  if (include_invalid_slots) {
    ASSERT_TRUE(inplace_final_state);
    ASSERT_GE(batch_size, 3);
    // Skip one nonempty sequence entirely and skip only the first checkpoint
    // write of another sequence whose accepted state remains valid.
    state_indices_cpu.select(/*dim=*/0, /*index=*/batch_size - 2).zero_();
    ASSERT_GT(accepted_counts.back(), 1);
    state_indices_cpu.select(/*dim=*/0, /*index=*/batch_size - 1)
        .select(/*dim=*/0, /*index=*/0)
        .zero_();
    if (checkpoint_capacity > 4) {
      // A missing checkpoint at the end of the first token window must not
      // reset the recurrence feeding the next window.
      ASSERT_NE(accepted_counts.back(), 4);
      state_indices_cpu.select(/*dim=*/0, /*index=*/batch_size - 1)
          .select(/*dim=*/0, /*index=*/3)
          .zero_();
    }
  }

  auto expected_out_cpu = torch::zeros_like(v_cpu);
  auto expected_state_cpu =
      inplace_final_state
          ? initial_state_cpu.clone()
          : torch::empty({allocated_tokens, kNumHeads, kHeadDim, kHeadDim},
                         fp32_options);
  auto query_cpu = q_cpu.to(torch::kFloat32).select(/*dim=*/0, /*index=*/0);
  auto key_cpu = k_cpu.to(torch::kFloat32).select(/*dim=*/0, /*index=*/0);
  auto value_cpu = v_cpu.to(torch::kFloat32).select(/*dim=*/0, /*index=*/0);
  auto gate_cpu =
      a_cpu.to(torch::kFloat32).view({allocated_tokens, kNumHeads, kHeadDim});
  auto beta_cpu = b_cpu.to(torch::kFloat32);
  auto bias_cpu = dt_bias_cpu.view({kNumHeads, kHeadDim});
  for (int64_t sequence = 0; sequence < batch_size; ++sequence) {
    const int64_t source_slot =
        state_indices_cpu.select(/*dim=*/0, sequence)
            .select(/*dim=*/0, accepted_counts[sequence] - 1)
            .item<int32_t>();
    if (source_slot <= 0 || sequence_lengths[sequence] == 0) {
      continue;
    }
    auto state = initial_state_cpu.select(/*dim=*/0, source_slot).clone();
    for (int64_t token = offsets[sequence]; token < offsets[sequence + 1];
         ++token) {
      auto output = kda_reference_step(query_cpu.select(/*dim=*/0, token),
                                       key_cpu.select(/*dim=*/0, token),
                                       value_cpu.select(/*dim=*/0, token),
                                       gate_cpu.select(/*dim=*/0, token),
                                       beta_cpu.select(/*dim=*/0, token),
                                       a_log_cpu,
                                       bias_cpu,
                                       state,
                                       scale,
                                       kGateLowerBound);
      expected_out_cpu.select(/*dim=*/0, /*index=*/0)
          .select(/*dim=*/0, token)
          .copy_(output.to(torch::kBFloat16));
      const int64_t destination_slot =
          inplace_final_state
              ? state_indices_cpu.select(/*dim=*/0, sequence)
                    .select(/*dim=*/0, token - offsets[sequence])
                    .item<int32_t>()
              : token;
      if (inplace_final_state && destination_slot <= 0) {
        continue;
      }
      expected_state_cpu.select(/*dim=*/0, destination_slot).copy_(state);
    }
  }

  auto a_log = a_log_cpu.to(device);
  auto dt_bias = dt_bias_cpu.to(device);
  auto a = a_cpu.to(device);
  auto b = b_cpu.to(device);
  auto q = q_cpu.to(device);
  auto k = k_cpu.to(device);
  auto v = v_cpu.to(device);
  auto initial_state = initial_state_cpu.to(device);
  auto state_indices = state_indices_cpu.to(device);
  auto cu_seqlens = torch::tensor(offsets, int_options).to(device);
  auto accepted_tokens = torch::tensor(accepted_counts, int_options).to(device);
  auto [out, final_state] = fused_sigmoid_gating_delta_rule_update(
      a_log,
      a,
      b,
      dt_bias,
      q,
      k,
      v,
      initial_state,
      state_indices,
      cu_seqlens,
      scale,
      /*use_qk_l2norm_in_kernel=*/true,
      /*softplus_beta=*/1.0f,
      /*softplus_threshold=*/20.0f,
      /*num_accepted_tokens_opt=*/accepted_tokens,
      inplace_final_state,
      /*is_kda=*/true,
      /*kda_use_safe_gate=*/true,
      kGateLowerBound);
  torch_mlu::synchronize();

  EXPECT_EQ(out.sizes(), expected_out_cpu.sizes());
  EXPECT_EQ(final_state.sizes(), expected_state_cpu.sizes());
  EXPECT_TRUE(torch::allclose(out.cpu().to(torch::kFloat32),
                              expected_out_cpu.to(torch::kFloat32),
                              /*rtol=*/2e-2,
                              /*atol=*/2e-2));
  EXPECT_TRUE(torch::allclose(final_state.cpu(),
                              expected_state_cpu,
                              /*rtol=*/2e-3,
                              /*atol=*/2e-3));
  if (!inplace_final_state || total_tokens == 0) {
    EXPECT_TRUE(torch::equal(initial_state.cpu(), initial_state_cpu));
  } else {
    EXPECT_TRUE(torch::equal(final_state.cpu().select(/*dim=*/0, /*index=*/0),
                             initial_state_cpu.select(/*dim=*/0, /*index=*/0)));
  }
}

TEST(FusedSigmoidUpdateTest, SparseSingleSequenceUsesAcceptedCheckpoint) {
  for (int32_t accepted_count = 1; accepted_count <= 4; ++accepted_count) {
    SCOPED_TRACE(accepted_count);
    expect_sparse_kda_matches_reference({1},
                                        {accepted_count},
                                        /*inplace_final_state=*/true,
                                        /*include_invalid_slots=*/false);
    expect_sparse_kda_matches_reference({2},
                                        {accepted_count},
                                        /*inplace_final_state=*/true,
                                        /*include_invalid_slots=*/false);
  }
}

TEST(FusedSigmoidUpdateTest, SparseKdaBatchSkipsEmptyAndInvalidSlots) {
  expect_sparse_kda_matches_reference({0, 1, 2, 1, 2},
                                      {1, 4, 2, 3, 4},
                                      /*inplace_final_state=*/true,
                                      /*include_invalid_slots=*/true);
}

TEST(FusedSigmoidUpdateTest,
     SparseNonInplaceKdaReturnsEveryStateWithoutMutatingInput) {
  expect_sparse_kda_matches_reference({2},
                                      {4},
                                      /*inplace_final_state=*/false,
                                      /*include_invalid_slots=*/false);
  expect_sparse_kda_matches_reference({0, 1, 2, 0, 2},
                                      {1, 4, 2, 3, 4},
                                      /*inplace_final_state=*/false,
                                      /*include_invalid_slots=*/false);
}

TEST(FusedSigmoidUpdateTest,
     GroupedKdaBatchPreservesPaddingAndAcceptedCheckpoints) {
  std::vector<int32_t> sequence_lengths(32, 4);
  std::vector<int32_t> accepted_counts(32, 2);
  sequence_lengths[0] = 0;
  sequence_lengths[1] = 1;
  sequence_lengths[2] = 3;
  accepted_counts[1] = 4;
  // The last two rows cover a fully padded state and a missing destination
  // checkpoint while retaining a valid accepted state, respectively.
  expect_sparse_kda_matches_reference(sequence_lengths,
                                      accepted_counts,
                                      /*inplace_final_state=*/true,
                                      /*include_invalid_slots=*/true);
}

TEST(FusedSigmoidUpdateTest,
     SixteenHeadGraphPaddedMtpFitsNramAndMatchesReference) {
  // Physical [1, 32, 16, 128] tensors retain graph padding while the eight
  // CU rows include an empty row and a nonempty row with an invalid state.
  expect_sparse_kda_matches_reference({4, 0, 4, 1, 2, 4, 2, 3},
                                      {1, 4, 2, 3, 4, 1, 2, 3},
                                      /*inplace_final_state=*/true,
                                      /*include_invalid_slots=*/true,
                                      /*checkpoint_capacity=*/4,
                                      /*num_heads=*/16,
                                      /*padded_token_count=*/32);
}

TEST(FusedSigmoidUpdateTest,
     ThirtyTwoHeadGraphPaddedMtpFitsNramAndMatchesReference) {
  // TP2 doubles local heads; the physical graph shape and eight CU rows stay
  // the same, while the generic KDA kernel must split the head dimension.
  expect_sparse_kda_matches_reference({4, 0, 4, 1, 2, 4, 2, 3},
                                      {1, 4, 2, 3, 4, 1, 2, 3},
                                      /*inplace_final_state=*/true,
                                      /*include_invalid_slots=*/true,
                                      /*checkpoint_capacity=*/4,
                                      /*num_heads=*/32,
                                      /*padded_token_count=*/32);
}

TEST(FusedSigmoidUpdateTest, LongCheckpointsSurviveNullWindowBoundaries) {
  expect_sparse_kda_matches_reference({5, 8, 6},
                                      {5, 8, 2},
                                      /*inplace_final_state=*/true,
                                      /*include_invalid_slots=*/true,
                                      /*checkpoint_capacity=*/8);
}

TEST(FusedSigmoidUpdateTest, NonInplaceLongQueriesIgnoreCheckpointCapacity) {
  expect_sparse_kda_matches_reference({5, 9, 6},
                                      {1, 1, 1},
                                      /*inplace_final_state=*/false,
                                      /*include_invalid_slots=*/false,
                                      /*checkpoint_capacity=*/1);
}

TEST(FusedSigmoidUpdateTest, EmptyKdaBatchPreservesInputState) {
  expect_sparse_kda_matches_reference({0, 0, 0},
                                      {1, 2, 4},
                                      /*inplace_final_state=*/true,
                                      /*include_invalid_slots=*/false);
  expect_sparse_kda_matches_reference({0, 0, 0},
                                      {1, 2, 4},
                                      /*inplace_final_state=*/false,
                                      /*include_invalid_slots=*/false);
}

TEST(FusedSigmoidUpdateTest, MultipleKTilesUseFullKeyDimension) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);

  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kSeqLen = 1;
  constexpr int64_t kNumHeads = 1;
  constexpr int64_t kHeadKDim = 256;
  constexpr int64_t kHeadVDim = 1;

  torch::TensorOptions bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  torch::TensorOptions fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  torch::Tensor a_log = torch::zeros({kNumHeads}, bf16_options);
  torch::Tensor a =
      torch::zeros({kBatchSize * kSeqLen, kNumHeads}, bf16_options);
  torch::Tensor b =
      torch::zeros({kBatchSize * kSeqLen, kNumHeads}, bf16_options);
  torch::Tensor dt_bias = torch::zeros({kNumHeads}, bf16_options);
  torch::Tensor q =
      torch::ones({kBatchSize, kSeqLen, kNumHeads, kHeadKDim}, bf16_options);
  torch::Tensor k =
      torch::ones({kBatchSize, kSeqLen, kNumHeads, kHeadKDim}, bf16_options);
  torch::Tensor v =
      torch::ones({kBatchSize, kSeqLen, kNumHeads, kHeadVDim}, bf16_options);
  torch::Tensor initial_state =
      torch::zeros({kBatchSize, kNumHeads, kHeadVDim, kHeadKDim}, fp32_options);
  torch::Tensor state_indices;
  torch::Tensor cu_seqlens;

  auto [out, final_state] = fused_sigmoid_gating_delta_rule_update(
      a_log,
      a,
      b,
      dt_bias,
      q,
      k,
      v,
      initial_state,
      state_indices,
      cu_seqlens,
      /*scale=*/1.0,
      /*use_qk_l2norm_in_kernel=*/false,
      /*softplus_beta=*/1.0f,
      /*softplus_threshold=*/20.0f,
      /*num_accepted_tokens_opt=*/std::nullopt,
      /*inplace_final_state=*/false,
      /*is_kda=*/false);
  torch_mlu::synchronize();

  ASSERT_EQ(out.numel(), 1);
  EXPECT_FLOAT_EQ(out.item<float>(), 128.0f)
      << "the output must reduce across all 256 key elements";
  EXPECT_TRUE(torch::allclose(final_state,
                              torch::full_like(final_state, /*fill_value=*/0.5),
                              /*rtol=*/0.0,
                              /*atol=*/0.0));
}

TEST(FusedSigmoidUpdateTest, InvalidContinuousBatchingRowsReturnZero) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);

  constexpr int64_t kNumHeads = 1;
  constexpr int64_t kHeadDim = 128;
  torch::TensorOptions bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  torch::TensorOptions fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  torch::TensorOptions int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);

  torch::Tensor a_log = torch::zeros({kNumHeads}, fp32_options);
  torch::Tensor a = torch::zeros({1, kNumHeads * kHeadDim}, bf16_options);
  torch::Tensor b = torch::zeros({1, kNumHeads}, bf16_options);
  torch::Tensor dt_bias = torch::zeros({kNumHeads * kHeadDim}, fp32_options);
  torch::Tensor q = torch::ones({1, 1, kNumHeads, kHeadDim}, bf16_options);
  torch::Tensor k = torch::ones({1, 1, kNumHeads, kHeadDim}, bf16_options);
  torch::Tensor v = torch::ones({1, 1, kNumHeads, kHeadDim}, bf16_options);
  torch::Tensor initial_state =
      torch::ones({2, kNumHeads, kHeadDim, kHeadDim}, fp32_options);
  torch::Tensor state_indices = torch::zeros({1, 1}, int_options);
  torch::Tensor cu_seqlens = torch::tensor({0, 1}, int_options);

  auto [out, final_state] = fused_sigmoid_gating_delta_rule_update(
      a_log,
      a,
      b,
      dt_bias,
      q,
      k,
      v,
      initial_state,
      state_indices,
      cu_seqlens,
      /*scale=*/1.0 / std::sqrt(static_cast<double>(kHeadDim)),
      /*use_qk_l2norm_in_kernel=*/true,
      /*softplus_beta=*/1.0f,
      /*softplus_threshold=*/20.0f,
      /*num_accepted_tokens_opt=*/std::nullopt,
      /*inplace_final_state=*/true,
      /*is_kda=*/true,
      /*kda_use_safe_gate=*/true,
      /*kda_gate_lower_bound=*/-5.0f);
  torch_mlu::synchronize();

  EXPECT_TRUE(torch::equal(out, torch::zeros_like(out)));
  EXPECT_TRUE(torch::equal(final_state, initial_state));
}

TEST(FusedSigmoidUpdateTest, BatchedKdaShapeFitsMluNram) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);

  constexpr int64_t kNumSequences = 64;
  constexpr int64_t kNumHeads = 8;
  constexpr int64_t kHeadDim = 128;
  torch::TensorOptions bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  torch::TensorOptions fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  torch::TensorOptions int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);

  torch::Tensor a_log = torch::zeros({kNumHeads}, fp32_options);
  torch::Tensor a =
      torch::zeros({kNumSequences, kNumHeads * kHeadDim}, bf16_options);
  torch::Tensor b = torch::zeros({kNumSequences, kNumHeads}, bf16_options);
  torch::Tensor dt_bias = torch::zeros({kNumHeads * kHeadDim}, fp32_options);
  torch::Tensor q =
      torch::zeros({1, kNumSequences, kNumHeads, kHeadDim}, bf16_options);
  torch::Tensor k = torch::zeros_like(q);
  torch::Tensor v = torch::zeros_like(q);
  torch::Tensor initial_state = torch::zeros(
      {kNumSequences + 1, kNumHeads, kHeadDim, kHeadDim}, fp32_options);
  torch::Tensor state_indices =
      torch::arange(1, kNumSequences + 1, int_options).view({kNumSequences, 1});
  torch::Tensor cu_seqlens = torch::arange(0, kNumSequences + 1, int_options);

  auto [out, final_state] = fused_sigmoid_gating_delta_rule_update(
      a_log,
      a,
      b,
      dt_bias,
      q,
      k,
      v,
      initial_state,
      state_indices,
      cu_seqlens,
      /*scale=*/1.0 / std::sqrt(static_cast<double>(kHeadDim)),
      /*use_qk_l2norm_in_kernel=*/true,
      /*softplus_beta=*/1.0f,
      /*softplus_threshold=*/20.0f,
      /*num_accepted_tokens_opt=*/std::nullopt,
      /*inplace_final_state=*/true,
      /*is_kda=*/true,
      /*kda_use_safe_gate=*/true,
      /*kda_gate_lower_bound=*/-5.0f);
  torch_mlu::synchronize();

  EXPECT_EQ(out.sizes(),
            torch::IntArrayRef({1, kNumSequences, kNumHeads, kHeadDim}));
  EXPECT_TRUE(torch::equal(out, torch::zeros_like(out)));
  EXPECT_TRUE(torch::equal(final_state, initial_state));
}

TEST(FusedSigmoidUpdateTest, BatchedSixteenHeadKdaMatchesSingleSteps) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  torch::manual_seed(20260923);
  constexpr int64_t kSequences = 16;
  constexpr int64_t kHeads = 16;
  constexpr int64_t kDim = 128;
  const torch::TensorOptions fp32 =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const torch::TensorOptions bf16 = fp32.dtype(torch::kBFloat16);
  const torch::TensorOptions int32 = fp32.dtype(torch::kInt32);
  const torch::Tensor a_log = torch::zeros({kHeads}, fp32);
  const torch::Tensor dt_bias = torch::zeros({kHeads * kDim}, fp32);
  torch::Tensor gate =
      (torch::randn({kSequences, kHeads * kDim}, fp32) - 2.0f).to(bf16);
  torch::Tensor beta = torch::randn({kSequences, kHeads}, bf16);
  torch::Tensor q = torch::randn({1, kSequences, kHeads, kDim}, bf16);
  torch::Tensor k = torch::randn_like(q);
  torch::Tensor v = torch::randn_like(q);
  torch::Tensor initial =
      torch::randn({kSequences + 1, kHeads, kDim, kDim}, fp32) * 0.01f;
  torch::Tensor batched_state = initial.clone();
  torch::Tensor sequential_state = initial.clone();
  torch::Tensor indices =
      torch::arange(1, kSequences + 1, int32).view({kSequences, 1});
  torch::Tensor cu = torch::arange(0, kSequences + 1, int32);
  auto run = [&](torch::Tensor query,
                 torch::Tensor key,
                 torch::Tensor value,
                 torch::Tensor raw_gate,
                 torch::Tensor raw_beta,
                 torch::Tensor state_indices,
                 torch::Tensor starts,
                 torch::Tensor& state) {
    return fused_sigmoid_gating_delta_rule_update(
        a_log,
        raw_gate,
        raw_beta,
        dt_bias,
        query,
        key,
        value,
        state,
        state_indices,
        starts,
        /*scale=*/1.0 / std::sqrt(static_cast<double>(kDim)),
        /*use_qk_l2norm_in_kernel=*/true,
        /*softplus_beta=*/1.0f,
        /*softplus_threshold=*/20.0f,
        /*num_accepted_tokens_opt=*/std::nullopt,
        /*inplace_final_state=*/true,
        /*is_kda=*/true,
        /*kda_use_safe_gate=*/true,
        /*kda_gate_lower_bound=*/-5.0f);
  };
  const torch::Tensor batched_output =
      run(q, k, v, gate, beta, indices, cu, batched_state).first;
  std::vector<torch::Tensor> outputs;
  outputs.reserve(kSequences);
  for (int64_t sequence = 0; sequence < kSequences; ++sequence) {
    outputs.emplace_back(run(q.narrow(1, sequence, 1),
                             k.narrow(1, sequence, 1),
                             v.narrow(1, sequence, 1),
                             gate.narrow(0, sequence, 1),
                             beta.narrow(0, sequence, 1),
                             indices.narrow(0, sequence, 1),
                             torch::tensor({0, 1}, int32),
                             sequential_state)
                             .first);
  }
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(batched_output,
                              torch::cat(outputs, /*dim=*/1),
                              /*rtol=*/5e-3,
                              /*atol=*/2e-5));
  EXPECT_TRUE(torch::allclose(batched_state,
                              sequential_state,
                              /*rtol=*/1e-3,
                              /*atol=*/2e-5));
}

TEST(FusedSigmoidUpdateTest, CompileHintsSeparateKernelCacheEntries) {
  const triton_jit::SpecList specs;
  const triton_jit::LaunchCfg default_cfg{1, 4};
  const triton_jit::LaunchCfg mv_cfg{1, 4, "mv"};
  EXPECT_NE(triton_jit::serialize_key(specs, default_cfg, /*device=*/0),
            triton_jit::serialize_key(specs, mv_cfg, /*device=*/0));
  triton_jit::LaunchCfg collapse_enabled_cfg = default_cfg;
  collapse_enabled_cfg.disable_trans_collapse_pass = false;
  triton_jit::LaunchCfg collapse_disabled_cfg = default_cfg;
  collapse_disabled_cfg.disable_trans_collapse_pass = true;
  const std::string default_key =
      triton_jit::serialize_key(specs, default_cfg, /*device=*/0);
  const std::string enabled_key =
      triton_jit::serialize_key(specs, collapse_enabled_cfg, /*device=*/0);
  const std::string disabled_key =
      triton_jit::serialize_key(specs, collapse_disabled_cfg, /*device=*/0);
  EXPECT_NE(default_key, enabled_key);
  EXPECT_NE(default_key, disabled_key);
  EXPECT_NE(enabled_key, disabled_key);
  collapse_disabled_cfg.disable_trans_collapse_pass = std::nullopt;
  EXPECT_EQ(
      default_key,
      triton_jit::serialize_key(specs, collapse_disabled_cfg, /*device=*/0));
}

TEST(FusedSigmoidUpdateTest, PersistentKdaBatchesMatchFp32Checkpoints) {
  for (int64_t batch_size : {1, 2, 16, 32}) {
    SCOPED_TRACE(batch_size);
    std::vector<int32_t> lengths(batch_size, 4);
    std::vector<int32_t> accepted;
    accepted.reserve(batch_size);
    for (int64_t sequence = 0; sequence < batch_size; ++sequence) {
      accepted.emplace_back(static_cast<int32_t>(sequence % 4 + 1));
    }
    expect_sparse_kda_matches_reference(lengths,
                                        accepted,
                                        /*inplace_final_state=*/true,
                                        /*include_invalid_slots=*/false);
    expect_sparse_kda_matches_reference(lengths,
                                        accepted,
                                        /*inplace_final_state=*/false,
                                        /*include_invalid_slots=*/false);
  }
}

TEST(FusedSigmoidUpdateTest, SingleTokenKdaEightHeadsMatchesReference) {
  expect_kda_update_matches_reference(/*seq_len=*/1);
}

TEST(FusedSigmoidUpdateTest, MultiTokenKdaEightHeadsMatchesReference) {
  expect_kda_update_matches_reference(/*seq_len=*/2);
}

TEST(FusedSigmoidUpdateTest, FourTokenKdaEightHeadsMatchesReference) {
  expect_kda_update_matches_reference(/*seq_len=*/4);
}

void expect_kda_batch_matches_sequential(
    const std::vector<int32_t>& sequence_lengths,
    bool use_int64_metadata = false) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  torch::manual_seed(20260907);

  constexpr int64_t kSequenceLength = 4;
  constexpr int64_t kNumHeads = 8;
  constexpr int64_t kHeadDim = 128;
  const int64_t batch_size = static_cast<int64_t>(sequence_lengths.size());
  std::vector<int32_t> offsets;
  offsets.reserve(sequence_lengths.size() + 1);
  offsets.emplace_back(0);
  for (int32_t length : sequence_lengths) {
    offsets.emplace_back(offsets.back() + length);
  }
  const int64_t total_tokens = offsets.back();
  const int64_t state_slots = batch_size * kSequenceLength;
  auto bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  auto fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto int_options =
      torch::TensorOptions()
          .dtype(use_int64_metadata ? torch::kInt64 : torch::kInt32)
          .device(device);

  auto a_log = torch::randn({kNumHeads}, fp32_options) * 0.1;
  auto dt_bias = torch::randn({kNumHeads * kHeadDim}, fp32_options) * 0.1;
  auto a = torch::randn({total_tokens, kNumHeads * kHeadDim}, bf16_options);
  auto b = torch::randn({total_tokens, kNumHeads}, bf16_options);
  auto q = torch::randn({1, total_tokens, kNumHeads, kHeadDim}, bf16_options);
  auto k = torch::randn_like(q);
  auto v = torch::randn_like(q);
  auto batched_state =
      torch::randn({state_slots + 1, kNumHeads, kHeadDim, kHeadDim},
                   fp32_options) *
      0.05;
  auto sequential_state = batched_state.clone();
  auto state_indices = torch::arange(1, state_slots + 1, int_options)
                           .view({batch_size, kSequenceLength});
  state_indices.select(0, batch_size - 2).zero_();
  state_indices.select(0, batch_size - 1)
      .select(0, kSequenceLength - 1)
      .zero_();
  auto accepted_tokens =
      torch::arange(0, batch_size, int_options).remainder(kSequenceLength) + 1;
  accepted_tokens.select(0, batch_size - 1).fill_(1);
  auto cu_seqlens = torch::tensor(offsets, int_options);

  auto run = [&](torch::Tensor query,
                 torch::Tensor key,
                 torch::Tensor value,
                 torch::Tensor gate,
                 torch::Tensor beta_logits,
                 torch::Tensor indices,
                 torch::Tensor offsets,
                 const torch::Tensor& accepted,
                 torch::Tensor& state) {
    return fused_sigmoid_gating_delta_rule_update(
        a_log,
        gate,
        beta_logits,
        dt_bias,
        query,
        key,
        value,
        state,
        indices,
        offsets,
        /*scale=*/1.0 / std::sqrt(static_cast<double>(kHeadDim)),
        /*use_qk_l2norm_in_kernel=*/true,
        /*softplus_beta=*/1.0f,
        /*softplus_threshold=*/20.0f,
        /*num_accepted_tokens_opt=*/accepted,
        /*inplace_final_state=*/true,
        /*is_kda=*/true,
        /*kda_use_safe_gate=*/true,
        /*kda_gate_lower_bound=*/-5.0f);
  };

  auto batched_result = run(
      q, k, v, a, b, state_indices, cu_seqlens, accepted_tokens, batched_state);
  auto expected_output = torch::zeros_like(batched_result.first);
  for (int64_t sequence = 0; sequence < batch_size; ++sequence) {
    const int64_t offset = offsets[sequence];
    const int64_t length = sequence_lengths[sequence];
    auto single_cu_seqlens =
        torch::tensor({int32_t{0}, static_cast<int32_t>(length)}, int_options);
    auto sequence_result = run(q.narrow(1, offset, length),
                               k.narrow(1, offset, length),
                               v.narrow(1, offset, length),
                               a.narrow(0, offset, length),
                               b.narrow(0, offset, length),
                               state_indices.narrow(0, sequence, 1),
                               single_cu_seqlens,
                               accepted_tokens.narrow(0, sequence, 1),
                               sequential_state);
    expected_output.narrow(1, offset, length).copy_(sequence_result.first);
  }
  torch_mlu::synchronize();

  EXPECT_TRUE(
      torch::allclose(batched_result.first.to(torch::kCPU, torch::kFloat32),
                      expected_output.to(torch::kCPU, torch::kFloat32),
                      /*rtol=*/2e-2,
                      /*atol=*/2e-2));
  EXPECT_TRUE(torch::allclose(batched_state.cpu(),
                              sequential_state.cpu(),
                              /*rtol=*/2e-3,
                              /*atol=*/2e-3));
}

TEST(FusedSigmoidUpdateTest,
     UnevenKdaBatchPreservesEverySpeculativeCheckpoint) {
  expect_kda_batch_matches_sequential(
      std::vector<int32_t>(/*count=*/48, /*value=*/4));
}

TEST(FusedSigmoidUpdateTest,
     LargeKdaBatchesPreserveEverySpeculativeCheckpoint) {
  expect_kda_batch_matches_sequential(
      std::vector<int32_t>(/*count=*/32, /*value=*/4));
  expect_kda_batch_matches_sequential(
      std::vector<int32_t>(/*count=*/40, /*value=*/4));
  expect_kda_batch_matches_sequential(
      std::vector<int32_t>(/*count=*/64, /*value=*/4));
}

TEST(FusedSigmoidUpdateTest,
     ChangingKdaBatchSizePreservesEverySpeculativeCheckpoint) {
  expect_kda_batch_matches_sequential(
      std::vector<int32_t>(/*count=*/3, /*value=*/4));
  expect_kda_batch_matches_sequential(
      std::vector<int32_t>(/*count=*/5, /*value=*/4));
}

TEST(FusedSigmoidUpdateTest,
     RaggedKdaBatchPreservesEverySpeculativeCheckpoint) {
  expect_kda_batch_matches_sequential({0, 1, 2, 3, 0, 3, 2, 1});
}

TEST(FusedSigmoidUpdateTest, Int64KdaMetadataPreservesEveryCheckpoint) {
  expect_kda_batch_matches_sequential(
      std::vector<int32_t>(/*count=*/16, /*value=*/4),
      /*use_int64_metadata=*/true);
}

void expect_tp4_decode_graph_matches_reference(int64_t bucket,
                                               int64_t live_sequences,
                                               bool include_empty_query,
                                               bool remap_requests = false,
                                               int32_t rounds = 3) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  torch::manual_seed(20261008);
  constexpr int64_t kHeads = 16;
  constexpr int64_t kDim = 128;
  constexpr float kLowerBound = -5.0f;
  const double scale = 1.0 / std::sqrt(static_cast<double>(kDim));
  const auto fp32 = torch::TensorOptions().dtype(torch::kFloat32);
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const auto ints = torch::TensorOptions().dtype(torch::kInt32);
  std::vector<int32_t> offsets;
  std::vector<int32_t> slots;
  offsets.reserve(bucket + 1);
  slots.reserve(bucket);
  offsets.emplace_back(0);
  for (int64_t row = 0; row < bucket; ++row) {
    const bool empty = include_empty_query && row == 1;
    offsets.emplace_back(offsets.back() + (empty ? 0 : 1));
    const int32_t slot = row < live_sequences
                             ? static_cast<int32_t>(live_sequences - row)
                             : (row % 2 == 0 ? 0 : -1);
    slots.emplace_back(slot);
  }
  torch::Tensor a_log_cpu = torch::randn({kHeads}, fp32) * 0.1;
  torch::Tensor bias_cpu = torch::randn({kHeads * kDim}, fp32) * 0.1;
  torch::Tensor gate_cpu = torch::randn({bucket, kHeads * kDim}, bf16);
  torch::Tensor beta_cpu = torch::randn({bucket, kHeads}, bf16);
  torch::Tensor q_cpu = torch::randn({1, bucket, kHeads, kDim}, bf16);
  torch::Tensor k_cpu = torch::randn_like(q_cpu);
  torch::Tensor v_cpu = torch::randn_like(q_cpu);
  torch::Tensor original_state =
      torch::randn({bucket + 2, kHeads, kDim, kDim}, fp32) * 0.05;
  torch::Tensor expected_state = original_state.clone();
  torch::Tensor state = original_state.to(device);
  torch::Tensor a_log = a_log_cpu.to(device);
  torch::Tensor bias = bias_cpu.to(device);
  torch::Tensor gate = gate_cpu.to(device);
  torch::Tensor beta = beta_cpu.to(device);
  torch::Tensor q = q_cpu.to(device);
  torch::Tensor k = k_cpu.to(device);
  torch::Tensor v = v_cpu.to(device);
  torch::Tensor indices = torch::tensor(slots, ints).to(device);
  torch::Tensor cu = torch::tensor(offsets, ints).to(device);
  auto run = [&]() {
    return fused_sigmoid_gating_delta_rule_update(
        a_log,
        gate,
        beta,
        bias,
        q,
        k,
        v,
        state,
        indices,
        cu,
        scale,
        /*use_qk_l2norm_in_kernel=*/true,
        /*softplus_beta=*/1.0f,
        /*softplus_threshold=*/20.0f,
        /*num_accepted_tokens_opt=*/std::nullopt,
        /*inplace_final_state=*/true,
        /*is_kda=*/true,
        /*kda_use_safe_gate=*/true,
        /*kda_gate_lower_bound=*/kLowerBound);
  };
  // Warm the actual C++ dispatcher before capture; never reproduce its launch
  // parameters in the test. Capture also exercises wrapper output zeroing.
  auto warmup = run();
  EXPECT_EQ(warmup.second.data_ptr(), state.data_ptr());
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  torch::Tensor output;
  {
    torch_mlu::mlu::MLUStreamGuard stream(
        torch_mlu::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0));
    graph.capture_begin();
    output = run().first;
    graph.capture_end();
  }
  state.copy_(original_state);
  for (int32_t round = 0; round < rounds; ++round) {
    SCOPED_TRACE(round);
    if (remap_requests) {
      // Evict a request, recycle its slot with clean state, and change both
      // active request count and row order while graph addresses stay fixed.
      live_sequences = bucket - round % 3;
      for (int64_t row = 0; row < bucket; ++row) {
        slots[row] = row < live_sequences
                         ? static_cast<int32_t>((row + round) % bucket + 1)
                         : -1;
      }
      indices.copy_(torch::tensor(slots, ints));
      state[1].zero_();
      expected_state[1].zero_();
    }
    // Change inputs while keeping graph addresses fixed, and carry recurrent
    // state across replays instead of resetting it before each comparison.
    q_cpu = torch::randn_like(q_cpu);
    q.copy_(q_cpu);
    torch::Tensor expected_output =
        torch::zeros({1, bucket, kHeads, kDim}, fp32);
    for (int64_t row = 0; row < live_sequences; ++row) {
      if (offsets[row] == offsets[row + 1]) {
        continue;
      }
      const int64_t token = offsets[row];
      const int64_t slot = slots[row];
      torch::Tensor recurrence = expected_state[slot].clone();
      torch::Tensor step = kda_reference_step(
          q_cpu[0][token].to(torch::kFloat32),
          k_cpu[0][token].to(torch::kFloat32),
          v_cpu[0][token].to(torch::kFloat32),
          gate_cpu[token].to(torch::kFloat32).view({kHeads, kDim}),
          beta_cpu[token].to(torch::kFloat32),
          a_log_cpu,
          bias_cpu.view({kHeads, kDim}),
          recurrence,
          scale,
          kLowerBound);
      expected_output[0][token].copy_(step);
      expected_state[slot].copy_(recurrence);
    }
    graph.replay();
    torch_mlu::synchronize();
    torch::Tensor actual_output = output.to(torch::kCPU).to(torch::kFloat32);
    torch::Tensor actual_state = state.to(torch::kCPU);
    EXPECT_TRUE(torch::allclose(actual_output,
                                expected_output,
                                /*rtol=*/0.03,
                                /*atol=*/0.003));
    EXPECT_TRUE(torch::allclose(actual_state,
                                expected_state,
                                /*rtol=*/5e-4,
                                /*atol=*/5e-5));
    EXPECT_TRUE(torch::equal(actual_state[0], original_state[0]));
    EXPECT_TRUE(
        torch::equal(actual_state[bucket + 1], original_state[bucket + 1]));
    EXPECT_TRUE(torch::isfinite(actual_output).all().item<bool>());
    EXPECT_TRUE(torch::isfinite(actual_state).all().item<bool>());
    for (int64_t token = offsets[live_sequences]; token < bucket; ++token) {
      EXPECT_TRUE(torch::equal(actual_output[0][token],
                               torch::zeros_like(actual_output[0][token])));
    }
    if (include_empty_query) {
      EXPECT_TRUE(
          torch::equal(actual_state[slots[1]], original_state[slots[1]]));
    }
  }
}

TEST(FusedSigmoidUpdateTest, Tp4DecodeGraphBucketsMatchFp32Recurrence) {
  for (int64_t bucket : {1, 2, 4, 8, 16, 32}) {
    SCOPED_TRACE(bucket);
    expect_tp4_decode_graph_matches_reference(bucket,
                                              bucket,
                                              /*include_empty_query=*/false);
  }
}

TEST(FusedSigmoidUpdateTest, Tp4DecodeGraphPaddingAndEmptyRowsPreserveState) {
  expect_tp4_decode_graph_matches_reference(/*bucket=*/4,
                                            /*live_sequences=*/3,
                                            /*include_empty_query=*/false);
  expect_tp4_decode_graph_matches_reference(/*bucket=*/8,
                                            /*live_sequences=*/5,
                                            /*include_empty_query=*/false);
  expect_tp4_decode_graph_matches_reference(/*bucket=*/8,
                                            /*live_sequences=*/5,
                                            /*include_empty_query=*/true);
}

TEST(FusedSigmoidUpdateTest, EverySmallBatchPreservesFourTokenCheckpoints) {
  for (int32_t batch = 1; batch <= 16; ++batch) {
    for (const int64_t heads : {8, 16, 32, 64}) {
      SCOPED_TRACE(batch);
      SCOPED_TRACE(heads);
      const std::vector<int32_t> lengths(batch, 4);
      std::vector<int32_t> accepted;
      accepted.reserve(batch);
      for (int32_t sequence = 0; sequence < batch; ++sequence) {
        accepted.emplace_back(sequence % 4 + 1);
      }
      expect_sparse_kda_matches_reference(lengths,
                                          accepted,
                                          /*inplace_final_state=*/true,
                                          /*include_invalid_slots=*/false,
                                          /*checkpoint_capacity=*/4,
                                          heads);
    }
  }
}

TEST(FusedSigmoidUpdateTest,
     HeadAlignedVerifyKeepsCheckpointsAcrossHeadsAndGateBatches) {
  // Forty four-column rows put more than 32 tokens of one head on a core for
  // every local head count, so gate batches and head switches repeat inside
  // a core; empty, short and invalid rows break the state prefetch chain.
  constexpr int32_t kBatch = 40;
  std::vector<int32_t> lengths(kBatch, 4);
  std::vector<int32_t> accepted;
  accepted.reserve(kBatch);
  for (int32_t row = 0; row < kBatch; ++row) {
    accepted.emplace_back(row % 4 + 1);
  }
  for (const int32_t row : {0, 9, 21}) {
    lengths[row] = 0;
  }
  for (const int32_t row : {3, 14, 30}) {
    lengths[row] = 1;
  }
  for (const int32_t row : {6, 25}) {
    lengths[row] = 3;
  }
  for (const int64_t heads : {8, 16, 32, 64}) {
    SCOPED_TRACE(heads);
    expect_sparse_kda_matches_reference(lengths,
                                        accepted,
                                        /*inplace_final_state=*/true,
                                        /*include_invalid_slots=*/true,
                                        /*checkpoint_capacity=*/4,
                                        heads,
                                        /*padded_token_count=*/kBatch * 4 + 8);
  }
}

TEST(FusedSigmoidUpdateTest, LowTpLongQueriesPreserveMaskedWindowBoundaries) {
  for (const int64_t heads : {16, 32, 64}) {
    for (const int32_t query : {5, 6, 7, 8}) {
      SCOPED_TRACE(heads);
      SCOPED_TRACE(query);
      expect_sparse_kda_matches_reference({query, 0, query - 1, 2, query},
                                          {1, 1, 2, 1, 3},
                                          /*inplace_final_state=*/true,
                                          /*include_invalid_slots=*/true,
                                          /*checkpoint_capacity=*/8,
                                          heads,
                                          /*padded_token_count=*/40);
    }
  }
}

TEST(FusedSigmoidUpdateTest, EverySmallDecodeBatchUsesLiveGraphMetadata) {
  for (int64_t batch = 1; batch <= 16; ++batch) {
    SCOPED_TRACE(batch);
    expect_tp4_decode_graph_matches_reference(batch,
                                              batch,
                                              /*include_empty_query=*/false);
  }
}

TEST(FusedSigmoidUpdateTest, WideNegativeMetadataCannotAliasValidStateSlots) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  constexpr int64_t kHeads = 8;
  constexpr int64_t kDim = 128;
  const double scale = 1.0 / std::sqrt(static_cast<double>(kDim));
  torch::manual_seed(20261020);
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const auto fp32 = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor q = torch::randn({1, 3, kHeads, kDim}, bf16).to(device);
  torch::Tensor k = torch::randn_like(q);
  torch::Tensor v = torch::randn_like(q);
  torch::Tensor gate = torch::randn({3, kHeads * kDim}, bf16).to(device);
  torch::Tensor beta = torch::randn({3, kHeads}, bf16).to(device);
  torch::Tensor log = torch::zeros({kHeads}, fp32).to(device);
  torch::Tensor bias = torch::zeros({kHeads * kDim}, fp32).to(device);
  torch::Tensor initial = torch::randn({2, kHeads, kDim, kDim}, fp32);
  torch::Tensor cu = torch::tensor({0, 1, 2, 3}, torch::kInt64).to(device);
  torch::Tensor negative_slots =
      torch::tensor({std::numeric_limits<int64_t>::min(),
                     -(int64_t{1} << 32) + 1,
                     int64_t{-1}},
                    torch::kInt64);
  torch::Tensor invalid_accepted =
      torch::tensor({-(int64_t{1} << 32) + 1,
                     (int64_t{1} << 32) + 1,
                     std::numeric_limits<int64_t>::min()},
                    torch::kInt64);
  for (const int64_t mode : {0, 1, 2}) {
    SCOPED_TRACE(mode);
    torch::Tensor indices =
        (mode == 2 ? torch::ones({3, 1}, torch::kInt64)
                   : (mode == 1 ? negative_slots.view({3, 1}) : negative_slots))
            .to(device);
    const std::optional<torch::Tensor> accepted =
        mode == 2 ? std::make_optional(invalid_accepted.to(device))
                  : std::nullopt;
    torch::Tensor state = initial.to(device);
    const auto result =
        fused_sigmoid_gating_delta_rule_update(log,
                                               gate,
                                               beta,
                                               bias,
                                               q,
                                               k,
                                               v,
                                               state,
                                               indices,
                                               cu,
                                               scale,
                                               /*use_qk_l2norm_in_kernel=*/true,
                                               /*softplus_beta=*/1.0f,
                                               /*softplus_threshold=*/20.0f,
                                               accepted,
                                               /*inplace_final_state=*/true,
                                               /*is_kda=*/true,
                                               /*kda_use_safe_gate=*/true,
                                               /*kda_gate_lower_bound=*/-5.0f);
    torch_mlu::synchronize();
    EXPECT_TRUE(torch::equal(result.first.cpu(),
                             torch::zeros_like(result.first.cpu())));
    EXPECT_TRUE(torch::equal(state.cpu(), initial));
  }
}

TEST(FusedSigmoidUpdateTest, Int32AcceptedCountsOutsideQueryWindowSkipWrites) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  torch::manual_seed(20261022);
  constexpr int64_t kDim = 128;
  const auto bf16 =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  const auto fp32 = bf16.dtype(torch::kFloat32);
  const auto integer = bf16.dtype(torch::kInt32);
  for (const int64_t heads : {8, 32}) {
    for (const int64_t columns : {1, 4}) {
      SCOPED_TRACE(heads);
      SCOPED_TRACE(columns);
      const int64_t tokens = 3 * columns;
      torch::Tensor q = torch::randn({1, tokens, heads, kDim}, bf16);
      torch::Tensor k = torch::randn_like(q), v = torch::randn_like(q);
      torch::Tensor gate = torch::randn({tokens, heads * kDim}, bf16);
      torch::Tensor beta = torch::randn({tokens, heads}, bf16);
      torch::Tensor log = torch::zeros({heads}, fp32);
      torch::Tensor bias = torch::zeros({heads * kDim}, fp32);
      torch::Tensor initial =
          torch::randn({tokens + 1, heads, kDim, kDim}, fp32);
      torch::Tensor state = initial.clone();
      torch::Tensor slots =
          torch::arange(1, tokens + 1, integer).view({3, columns});
      torch::Tensor cu = torch::arange(0, tokens + 1, columns, integer);
      torch::Tensor accepted =
          torch::tensor({std::numeric_limits<int32_t>::min(),
                         int32_t{0},
                         static_cast<int32_t>(columns + 1)},
                        torch::kInt32)
              .to(device);
      auto result = fused_sigmoid_gating_delta_rule_update(
          log,
          gate,
          beta,
          bias,
          q,
          k,
          v,
          state,
          slots,
          cu,
          1.0 / std::sqrt(static_cast<double>(kDim)),
          true,
          1.0f,
          20.0f,
          accepted,
          true,
          true,
          true,
          -5.0f);
      torch_mlu::synchronize();
      EXPECT_TRUE(torch::equal(result.first.cpu(),
                               torch::zeros_like(result.first.cpu())));
      EXPECT_TRUE(torch::equal(state.cpu(), initial.cpu()));
    }
  }
}

TEST(FusedSigmoidUpdateTest,
     NoncontiguousWideMetadataRemainsLiveOnCapturedStream) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  torch_mlu::mlu::MLUStreamGuard stream(
      torch_mlu::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0));
  constexpr int64_t kHeads = 8;
  constexpr int64_t kDim = 128;
  const double scale = 1.0 / std::sqrt(static_cast<double>(kDim));
  torch::manual_seed(20261021);
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const auto fp32 = torch::TensorOptions().dtype(torch::kFloat32);
  const torch::Tensor q_cpu = torch::randn({1, 3, kHeads, kDim}, bf16);
  const torch::Tensor k_cpu = torch::randn_like(q_cpu);
  const torch::Tensor v_cpu = torch::randn_like(q_cpu);
  const torch::Tensor gate_cpu = torch::randn({3, kHeads * kDim}, bf16);
  const torch::Tensor beta_cpu = torch::randn({3, kHeads}, bf16);
  const torch::Tensor log_cpu = torch::zeros({kHeads}, fp32);
  const torch::Tensor bias_cpu = torch::zeros({kHeads, kDim}, fp32);
  const torch::Tensor initial = torch::randn({5, kHeads, kDim, kDim}, fp32);
  torch::Tensor q = q_cpu.to(device), k = k_cpu.to(device);
  torch::Tensor v = v_cpu.to(device), gate = gate_cpu.to(device);
  torch::Tensor beta = beta_cpu.to(device), log = log_cpu.to(device);
  torch::Tensor bias = bias_cpu.flatten().to(device);
  torch::Tensor state = initial.to(device);
  torch::Tensor cu_backing =
      torch::tensor({0, 91, 1, 92, 2, 93, 3, 94}, torch::kInt64).to(device);
  torch::Tensor cu = cu_backing.slice(0, 0, 8, 2);
  torch::Tensor indices_backing =
      torch::tensor({1, 91, 2, 92, 3, 93}, torch::kInt64)
          .view({3, 2})
          .to(device);
  torch::Tensor indices = indices_backing.slice(1, 0, 1);
  torch::Tensor accepted_backing =
      torch::ones({3, 2}, torch::kInt64).to(device);
  torch::Tensor accepted = accepted_backing.select(1, 0);
  ASSERT_FALSE(cu.is_contiguous());
  ASSERT_FALSE(indices.is_contiguous());
  ASSERT_FALSE(accepted.is_contiguous());
  const auto run = [&]() {
    return fused_sigmoid_gating_delta_rule_update(log,
                                                  gate,
                                                  beta,
                                                  bias,
                                                  q,
                                                  k,
                                                  v,
                                                  state,
                                                  indices,
                                                  cu,
                                                  scale,
                                                  true,
                                                  1.0f,
                                                  20.0f,
                                                  accepted,
                                                  true,
                                                  true,
                                                  true,
                                                  -5.0f);
  };
  auto warmup = run();
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  graph.capture_begin();
  torch::Tensor output = run().first;
  graph.capture_end();
  for (int64_t round = 0; round < 6; ++round) {
    SCOPED_TRACE(round);
    torch::Tensor slots = torch::tensor({1, 2, 3}, torch::kInt64);
    torch::Tensor counts = torch::ones({3}, torch::kInt64);
    torch::Tensor offsets = torch::tensor({0, 1, 2, 3}, torch::kInt64);
    if (round == 1) {
      counts[1] = -(int64_t{1} << 32) + 1;
      offsets = torch::tensor({0, 0, 1, 2}, torch::kInt64);
    } else if (round == 2) {
      slots[0] = std::numeric_limits<int64_t>::min();
      offsets = torch::tensor({0, 1, 1, 2}, torch::kInt64);
    } else if (round == 3) {
      slots = torch::tensor({3, 1, 2}, torch::kInt64);
      offsets = torch::zeros({4}, torch::kInt64);
    } else if (round == 4) {
      counts[2] = std::numeric_limits<int64_t>::min();
    } else if (round == 5) {
      slots[1] = -(int64_t{1} << 32) + 1;
      offsets = torch::tensor({0, 1, 2, 2}, torch::kInt64);
    }
    cu.copy_(offsets);
    indices.copy_(slots.view({3, 1}));
    accepted.copy_(counts);
    state.copy_(initial);
    torch::Tensor expected_state = initial.clone();
    torch::Tensor expected_output = torch::zeros({1, 3, kHeads, kDim}, fp32);
    for (int64_t row = 0; row < 3; ++row) {
      const int64_t slot = slots[row].item<int64_t>();
      const int64_t token = offsets[row].item<int64_t>();
      if (token == offsets[row + 1].item<int64_t>() || slot <= 0 ||
          counts[row].item<int64_t>() != 1) {
        continue;
      }
      torch::Tensor recurrence = initial[slot].clone();
      expected_output[0][token].copy_(kda_reference_step(
          q_cpu[0][token].to(torch::kFloat32),
          k_cpu[0][token].to(torch::kFloat32),
          v_cpu[0][token].to(torch::kFloat32),
          gate_cpu[token].to(torch::kFloat32).view({kHeads, kDim}),
          beta_cpu[token].to(torch::kFloat32),
          log_cpu,
          bias_cpu,
          recurrence,
          scale,
          -5.0f));
      expected_state[slot].copy_(recurrence);
    }
    graph.replay();
    torch_mlu::synchronize();
    EXPECT_TRUE(torch::allclose(
        output.cpu().to(torch::kFloat32), expected_output, 0.03, 0.003));
    EXPECT_TRUE(torch::allclose(state.cpu(), expected_state, 5e-4, 5e-5));
    EXPECT_TRUE(torch::equal(state[0].cpu(), initial[0]));
    EXPECT_TRUE(torch::equal(state[4].cpu(), initial[4]));
    EXPECT_TRUE(torch::equal(indices.cpu(), slots.view({3, 1})));
    EXPECT_TRUE(torch::equal(accepted.cpu(), counts));
    EXPECT_TRUE(torch::equal(cu.cpu(), offsets));
    EXPECT_TRUE(torch::equal(cu_backing.slice(0, 1, 8, 2).cpu(),
                             torch::tensor({91, 92, 93, 94}, torch::kInt64)));
  }
}

TEST(FusedSigmoidUpdateTest, OneDimensionalSlotsKeepRaggedFinalStates) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  torch::manual_seed(20261019);
  constexpr int64_t kDim = 128;
  const double scale = 1.0 / std::sqrt(static_cast<double>(kDim));
  const auto fp32 = torch::TensorOptions().dtype(torch::kFloat32);
  const auto ints = torch::TensorOptions().dtype(torch::kInt64);
  for (int64_t heads : {8, 16, 32, 64}) {
    SCOPED_TRACE(heads);
    // The total token count equals N, but one request is empty and another has
    // two tokens. A 1D table names one final state slot per request.
    const std::vector<int64_t> offsets = {0, 0, 2, 3};
    const std::vector<int64_t> slots = {3, 1, 2};
    torch::Tensor q_cpu =
        torch::randn({1, 3, heads, kDim}, fp32).to(torch::kBFloat16);
    torch::Tensor k_cpu = torch::randn_like(q_cpu);
    torch::Tensor v_cpu = torch::randn_like(q_cpu);
    torch::Tensor gate_cpu =
        torch::randn({3, heads * kDim}, fp32).to(torch::kBFloat16);
    torch::Tensor beta_cpu =
        torch::randn({3, heads}, fp32).to(torch::kBFloat16);
    torch::Tensor log_cpu = torch::randn({heads}, fp32) * 0.1;
    torch::Tensor bias_cpu = torch::randn({heads * kDim}, fp32) * 0.1;
    torch::Tensor initial = torch::randn({5, heads, kDim, kDim}, fp32) * 0.05;
    torch::Tensor expected_state = initial.clone();
    torch::Tensor q = q_cpu.to(device);
    torch::Tensor k = k_cpu.to(device);
    torch::Tensor v = v_cpu.to(device);
    torch::Tensor gate = gate_cpu.to(device);
    torch::Tensor beta = beta_cpu.to(device);
    torch::Tensor log = log_cpu.to(device);
    torch::Tensor bias = bias_cpu.to(device);
    torch::Tensor state = initial.to(device);
    torch::Tensor indices = torch::tensor(slots, ints).to(device);
    torch::Tensor cu = torch::tensor(offsets, ints).to(device);
    for (int32_t repeat = 0; repeat < 3; ++repeat) {
      SCOPED_TRACE(repeat);
      torch::Tensor expected_output = torch::zeros({1, 3, heads, kDim}, fp32);
      for (int64_t sequence = 0; sequence < 3; ++sequence) {
        torch::Tensor current = expected_state[slots[sequence]].clone();
        for (int64_t token = offsets[sequence]; token < offsets[sequence + 1];
             ++token) {
          expected_output[0][token].copy_(kda_reference_step(
              q_cpu[0][token].to(torch::kFloat32),
              k_cpu[0][token].to(torch::kFloat32),
              v_cpu[0][token].to(torch::kFloat32),
              gate_cpu[token].to(torch::kFloat32).view({heads, kDim}),
              beta_cpu[token].to(torch::kFloat32),
              log_cpu,
              bias_cpu.view({heads, kDim}),
              current,
              scale,
              /*gate_lower_bound=*/-5.0f));
        }
        expected_state[slots[sequence]].copy_(current);
      }
      auto result = fused_sigmoid_gating_delta_rule_update(
          log,
          gate,
          beta,
          bias,
          q,
          k,
          v,
          state,
          indices,
          cu,
          scale,
          /*use_qk_l2norm_in_kernel=*/true,
          /*softplus_beta=*/1.0f,
          /*softplus_threshold=*/20.0f,
          /*num_accepted_tokens_opt=*/std::nullopt,
          /*inplace_final_state=*/true,
          /*is_kda=*/true,
          /*kda_use_safe_gate=*/true,
          /*kda_gate_lower_bound=*/-5.0f);
      torch_mlu::synchronize();
      EXPECT_EQ(result.second.data_ptr(), state.data_ptr());
      EXPECT_TRUE(
          torch::allclose(result.first.to(torch::kCPU).to(torch::kFloat32),
                          expected_output,
                          /*rtol=*/0.03,
                          /*atol=*/0.003));
      const torch::Tensor actual_state = state.to(torch::kCPU);
      EXPECT_TRUE(torch::allclose(actual_state,
                                  expected_state,
                                  /*rtol=*/5e-4,
                                  /*atol=*/5e-5));
      EXPECT_TRUE(torch::equal(actual_state[0], initial[0]));
      EXPECT_TRUE(torch::equal(actual_state[3], initial[3]));
      EXPECT_TRUE(torch::equal(actual_state[4], initial[4]));
    }
  }
}

// Sweep effective requests, not the packed tensor's leading dimension.
TEST(FusedSigmoidUpdateTest, DecodeAndVerifyCoverEveryBatchThrough128) {
  for (int32_t batch = 1; batch <= 128; ++batch) {
    for (const int32_t query : {1, 4}) {
      SCOPED_TRACE(::testing::Message() << batch << ":" << query);
      std::vector<int32_t> accepted;
      accepted.reserve(batch);
      for (int32_t row = 0; row < batch; ++row) {
        accepted.emplace_back(row % query + 1);
      }
      expect_sparse_kda_matches_reference(std::vector<int32_t>(batch, query),
                                          accepted,
                                          /*inplace_final_state=*/true,
                                          /*include_invalid_slots=*/false,
                                          /*checkpoint_capacity=*/query,
                                          /*num_heads=*/8);
    }
  }
}

TEST(FusedSigmoidUpdateTest,
     VerifySupportsLargerBatchesAndSixEightTokenWindows) {
  for (const int32_t batch : {1, 17, 64, 128, 129, 160}) {
    for (const int32_t query : {1, 4, 6, 8}) {
      SCOPED_TRACE(::testing::Message() << batch << ":" << query);
      std::vector<int32_t> accepted;
      accepted.reserve(batch);
      for (int32_t row = 0; row < batch; ++row) {
        accepted.emplace_back(row % query + 1);
      }
      expect_sparse_kda_matches_reference(std::vector<int32_t>(batch, query),
                                          accepted,
                                          /*inplace_final_state=*/true,
                                          /*include_invalid_slots=*/false,
                                          /*checkpoint_capacity=*/query,
                                          /*num_heads=*/8);
    }
  }
}

TEST(FusedSigmoidUpdateTest, LargeDecodeGraphsPreserveStateAcrossRounds) {
  for (const int64_t batch : {64, 127, 128, 129, 160}) {
    SCOPED_TRACE(batch);
    expect_tp4_decode_graph_matches_reference(batch,
                                              batch - 1,
                                              /*include_empty_query=*/true);
  }
}

TEST(FusedSigmoidUpdateTest, DraftQ1StepsPreserveReorderedAndReusedStateSlots) {
  for (const int32_t steps : {3, 5, 7}) {
    SCOPED_TRACE(steps);
    expect_tp4_decode_graph_matches_reference(
        /*bucket=*/17,
        /*live_sequences=*/17,
        /*include_empty_query=*/false,
        /*remap_requests=*/true,
        steps);
  }
}

}  // namespace
}  // namespace xllm
