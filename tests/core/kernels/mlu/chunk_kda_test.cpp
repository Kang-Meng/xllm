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

#include "kernels/mlu/chunk_kda.h"

#include <framework/core/device.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <vector>

#include "kernels/mlu/mlu_ops_api.h"

namespace xllm::kernel::mlu {
namespace {

constexpr int64_t kNumHeads = 8;
constexpr int64_t kHeadDim = 128;

torch::Tensor make_cu_seqlens(const std::vector<int64_t>& sequence_lengths,
                              const torch::Device& device) {
  std::vector<int32_t> cumulative_lengths = {0};
  cumulative_lengths.reserve(sequence_lengths.size() + 1);
  int64_t cumulative_length = 0;
  for (const int64_t sequence_length : sequence_lengths) {
    cumulative_length += sequence_length;
    cumulative_lengths.push_back(static_cast<int32_t>(cumulative_length));
  }
  return torch::tensor(
      cumulative_lengths,
      torch::TensorOptions().dtype(torch::kInt32).device(device));
}

torch::Tensor make_chunk_indices(const torch::Tensor& cu_seqlens,
                                 int64_t chunk_size) {
  torch::Tensor lengths = cu_seqlens.diff();
  torch::Tensor chunk_counts =
      ((lengths + chunk_size - 1) / chunk_size).to(torch::kLong);
  torch::Tensor chunk_offsets = torch::cumsum(chunk_counts, /*dim=*/0);
  const int64_t total_chunks = chunk_offsets[-1].item<int64_t>();
  torch::Tensor flat_indices =
      torch::arange(total_chunks, cu_seqlens.options());
  torch::Tensor prefixes =
      torch::cat({torch::zeros({1}, chunk_offsets.options()),
                  chunk_offsets.slice(/*dim=*/0, /*start=*/0, /*end=*/-1)});
  torch::Tensor local_indices =
      flat_indices - torch::repeat_interleave(prefixes, chunk_counts);
  torch::Tensor sequence_indices = (local_indices == 0).cumsum(/*dim=*/0) - 1;
  return torch::stack({sequence_indices, local_indices}, /*dim=*/1)
      .to(torch::kInt32);
}

torch::Tensor make_scaled_identity_states(const torch::Tensor& state_scales,
                                          int64_t num_sequences,
                                          const torch::TensorOptions& options) {
  return torch::eye(kHeadDim, options).view({1, 1, kHeadDim, kHeadDim}) *
         state_scales.view({num_sequences, kNumHeads, 1, 1});
}

TEST(ChunkKDAConfigTest, ChunkSizeDefaultsTo64AndSupports16Override) {
  const char* configured_chunk_size = std::getenv("XLLM_MLU_KDA_CHUNK_SIZE");
  const int64_t expected_chunk_size = configured_chunk_size == nullptr
                                          ? 64
                                          : std::strtoll(configured_chunk_size,
                                                         /*str_end=*/nullptr,
                                                         /*base=*/10);
  ASSERT_TRUE(expected_chunk_size == 16 || expected_chunk_size == 64);
  EXPECT_EQ(kda_prefill_chunk_size(kNumHeads, /*use_qk_l2norm=*/true),
            expected_chunk_size);
  EXPECT_EQ(kda_prefill_chunk_size(/*num_heads=*/1,
                                   /*use_qk_l2norm=*/false),
            expected_chunk_size);
}

TEST(ChunkKDATest,
     PackedHighConcurrencyPreservesStateIsolationAcrossChunkBoundaries) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);

  constexpr int64_t kNumSequences = 128;
  const int64_t chunk_size =
      kda_prefill_chunk_size(kNumHeads, /*use_qk_l2norm=*/true);
  const std::vector<int64_t> boundary_lengths = {1,
                                                 chunk_size - 1,
                                                 chunk_size,
                                                 chunk_size + 1,
                                                 2 * chunk_size - 1,
                                                 2 * chunk_size,
                                                 2 * chunk_size + 1};
  std::vector<int64_t> sequence_lengths;
  sequence_lengths.reserve(kNumSequences);
  for (int64_t sequence_id = 0; sequence_id < kNumSequences; ++sequence_id) {
    sequence_lengths.push_back(
        boundary_lengths[static_cast<size_t>(sequence_id) %
                         boundary_lengths.size()]);
  }
  const int64_t total_tokens = std::accumulate(
      sequence_lengths.begin(), sequence_lengths.end(), int64_t{0});
  const torch::TensorOptions bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const torch::TensorOptions fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  const torch::Tensor qkv =
      torch::ones({1, total_tokens, kNumHeads, kHeadDim}, bf16_options);
  const torch::Tensor raw_gate =
      torch::full({total_tokens, kNumHeads, kHeadDim}, -100.0f, bf16_options);
  const torch::Tensor raw_beta =
      torch::full({total_tokens, kNumHeads}, -100.0f, bf16_options);
  const torch::Tensor sequence_scales =
      torch::arange(kNumSequences, fp32_options).view({kNumSequences, 1}) /
      static_cast<double>(2 * kNumSequences);
  const torch::Tensor head_scales =
      torch::arange(kNumHeads, fp32_options).view({1, kNumHeads}) /
      static_cast<double>(4 * kNumHeads);
  const torch::Tensor state_scales = 0.25f + sequence_scales + head_scales;
  const torch::Tensor a_log = torch::zeros({kNumHeads}, fp32_options);
  const torch::Tensor dt_bias =
      torch::zeros({kNumHeads * kHeadDim}, fp32_options);
  const torch::Tensor initial_state =
      make_scaled_identity_states(state_scales, kNumSequences, fp32_options);
  const torch::Tensor cu_seqlens = make_cu_seqlens(sequence_lengths, device);

  ChunkKDA chunk_kda(kNumHeads);
  chunk_kda->to(device);
  auto [output, final_state] =
      chunk_kda->forward_raw_gate(qkv,
                                  qkv,
                                  qkv,
                                  raw_gate,
                                  a_log,
                                  dt_bias,
                                  /*gate_lower_bound=*/-5.0f,
                                  raw_beta,
                                  initial_state,
                                  cu_seqlens,
                                  /*chunk_indices=*/torch::Tensor(),
                                  /*output_final_state=*/true,
                                  /*use_qk_l2norm=*/true);
  torch_mlu::synchronize();

  const torch::Tensor length_tensor =
      torch::tensor(sequence_lengths,
                    torch::TensorOptions().dtype(torch::kLong).device(device));
  const torch::Tensor token_scales =
      torch::repeat_interleave(state_scales, length_tensor, /*dim=*/0);
  const torch::Tensor expected_output =
      (token_scales / static_cast<double>(kHeadDim))
          .unsqueeze(/*dim=*/-1)
          .expand({total_tokens, kNumHeads, kHeadDim});

  ASSERT_EQ(output.sizes(),
            torch::IntArrayRef({1, total_tokens, kNumHeads, kHeadDim}));
  ASSERT_EQ(final_state.sizes(), initial_state.sizes());
  EXPECT_TRUE(torch::allclose(output.squeeze(/*dim=*/0).to(torch::kFloat32),
                              expected_output,
                              /*rtol=*/0.0,
                              /*atol=*/1e-4));
  EXPECT_TRUE(torch::allclose(final_state,
                              initial_state,
                              /*rtol=*/0.0,
                              /*atol=*/1e-6));
}

TEST(ChunkKDATest, RawGatePreservesStateAcross128K) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);

  constexpr int64_t kLongContextTokens = 128 * 1024;
  const int64_t chunk_size =
      kda_prefill_chunk_size(kNumHeads, /*use_qk_l2norm=*/true);
  const torch::TensorOptions bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const torch::TensorOptions fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  const torch::Tensor qkv =
      torch::ones({1, kLongContextTokens, kNumHeads, kHeadDim}, bf16_options);
  const torch::Tensor raw_gate =
      torch::full({1, 1, 1}, -100.0f, bf16_options)
          .expand({kLongContextTokens, kNumHeads, kHeadDim});
  const torch::Tensor raw_beta =
      torch::full({kLongContextTokens, kNumHeads}, -100.0f, bf16_options);
  const torch::Tensor a_log = torch::zeros({kNumHeads}, fp32_options);
  const torch::Tensor dt_bias =
      torch::zeros({kNumHeads * kHeadDim}, fp32_options);
  const torch::Tensor initial_state =
      torch::eye(kHeadDim, fp32_options)
          .view({1, 1, kHeadDim, kHeadDim})
          .expand({1, kNumHeads, kHeadDim, kHeadDim})
          .contiguous();
  const std::vector<int64_t> sequence_lengths = {kLongContextTokens};
  const torch::Tensor cu_seqlens = make_cu_seqlens(sequence_lengths, device);
  const torch::Tensor chunk_indices =
      make_chunk_indices(cu_seqlens, chunk_size);

  ChunkKDA chunk_kda(kNumHeads);
  chunk_kda->to(device);
  auto [output, final_state] =
      chunk_kda->forward_raw_gate(qkv,
                                  qkv,
                                  qkv,
                                  raw_gate,
                                  a_log,
                                  dt_bias,
                                  /*gate_lower_bound=*/-5.0f,
                                  raw_beta,
                                  initial_state,
                                  cu_seqlens,
                                  chunk_indices,
                                  /*output_final_state=*/true,
                                  /*use_qk_l2norm=*/true);
  torch_mlu::synchronize();

  ASSERT_EQ(output.sizes(),
            torch::IntArrayRef({1, kLongContextTokens, kNumHeads, kHeadDim}));
  ASSERT_EQ(final_state.sizes(), initial_state.sizes());
  const float output_min = output.amin().item<float>();
  const float output_max = output.amax().item<float>();
  constexpr float kExpectedOutput = 1.0f / static_cast<float>(kHeadDim);
  EXPECT_TRUE(std::isfinite(output_min));
  EXPECT_TRUE(std::isfinite(output_max));
  EXPECT_NEAR(output_min, kExpectedOutput, 1e-4f);
  EXPECT_NEAR(output_max, kExpectedOutput, 1e-4f);
  EXPECT_TRUE(torch::allclose(final_state,
                              initial_state,
                              /*rtol=*/0.0,
                              /*atol=*/1e-6));
}

TEST(ChunkKDATest, Random128KMatchesEightKSegmentedStateChaining) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  torch::manual_seed(20260904);

  constexpr int64_t kLongContextTokens = 128 * 1024;
  constexpr int64_t kSegmentTokens = 8 * 1024;
  const int64_t chunk_size =
      kda_prefill_chunk_size(kNumHeads, /*use_qk_l2norm=*/true);
  const torch::TensorOptions bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const torch::TensorOptions fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const std::vector<int64_t> input_shape = {
      1, kLongContextTokens, kNumHeads, kHeadDim};

  // Share one immutable random tensor for q/k/v so the 128K regression stays
  // below the per-process MLU memory quota.
  const torch::Tensor qkv = torch::randn(input_shape, bf16_options) * 0.1f;
  const torch::Tensor raw_gate =
      torch::randn({kLongContextTokens, kNumHeads, kHeadDim}, bf16_options) *
          0.25f -
      6.0f;
  const torch::Tensor raw_beta =
      torch::randn({kLongContextTokens, kNumHeads}, bf16_options) * 0.2f - 1.5f;
  const torch::Tensor a_log = torch::zeros({kNumHeads}, fp32_options);
  const torch::Tensor dt_bias =
      torch::zeros({kNumHeads * kHeadDim}, fp32_options);
  const torch::Tensor initial_state =
      torch::randn({1, kNumHeads, kHeadDim, kHeadDim}, fp32_options) * 0.01f;

  const std::vector<int64_t> full_sequence_lengths = {kLongContextTokens};
  const torch::Tensor full_cu_seqlens =
      make_cu_seqlens(full_sequence_lengths, device);
  const torch::Tensor full_chunk_indices =
      make_chunk_indices(full_cu_seqlens, chunk_size);
  const std::vector<int64_t> segment_sequence_lengths = {kSegmentTokens};
  const torch::Tensor segment_cu_seqlens =
      make_cu_seqlens(segment_sequence_lengths, device);
  const torch::Tensor segment_chunk_indices =
      make_chunk_indices(segment_cu_seqlens, chunk_size);
  ASSERT_EQ(kLongContextTokens % kSegmentTokens, 0);
  ASSERT_GT(full_chunk_indices.size(0), segment_chunk_indices.size(0));

  ChunkKDA chunk_kda(kNumHeads);
  chunk_kda->to(device);
  auto [full_output, full_final_state] =
      chunk_kda->forward_raw_gate(qkv,
                                  qkv,
                                  qkv,
                                  raw_gate,
                                  a_log,
                                  dt_bias,
                                  /*gate_lower_bound=*/-5.0f,
                                  raw_beta,
                                  initial_state,
                                  full_cu_seqlens,
                                  full_chunk_indices,
                                  /*output_final_state=*/true,
                                  /*use_qk_l2norm=*/true);

  torch::Tensor segmented_state = initial_state;
  for (int64_t segment_begin = 0; segment_begin < kLongContextTokens;
       segment_begin += kSegmentTokens) {
    const int64_t segment_end = segment_begin + kSegmentTokens;
    auto [segment_output, segment_final_state] =
        chunk_kda->forward_raw_gate(qkv.slice(/*dim=*/1,
                                              /*start=*/segment_begin,
                                              /*end=*/segment_end),
                                    qkv.slice(/*dim=*/1,
                                              /*start=*/segment_begin,
                                              /*end=*/segment_end),
                                    qkv.slice(/*dim=*/1,
                                              /*start=*/segment_begin,
                                              /*end=*/segment_end),
                                    raw_gate.slice(/*dim=*/0,
                                                   /*start=*/segment_begin,
                                                   /*end=*/segment_end),
                                    a_log,
                                    dt_bias,
                                    /*gate_lower_bound=*/-5.0f,
                                    raw_beta.slice(/*dim=*/0,
                                                   /*start=*/segment_begin,
                                                   /*end=*/segment_end),
                                    segmented_state,
                                    segment_cu_seqlens,
                                    segment_chunk_indices,
                                    /*output_final_state=*/true,
                                    /*use_qk_l2norm=*/true);
    EXPECT_TRUE(torch::allclose(full_output.slice(/*dim=*/1,
                                                  /*start=*/segment_begin,
                                                  /*end=*/segment_end),
                                segment_output,
                                /*rtol=*/1e-3,
                                /*atol=*/1e-3))
        << "output mismatch in segment starting at token " << segment_begin;
    segmented_state = segment_final_state;
  }
  torch_mlu::synchronize();

  EXPECT_TRUE(torch::allclose(full_final_state,
                              segmented_state,
                              /*rtol=*/1e-4,
                              /*atol=*/1e-4));
}

TEST(ChunkKDATest, OmittingFinalStateKeepsOutputContract) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);

  const int64_t chunk_size =
      kda_prefill_chunk_size(kNumHeads, /*use_qk_l2norm=*/true);
  const int64_t token_count = chunk_size + 1;
  const torch::TensorOptions bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const torch::TensorOptions fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const torch::Tensor qkv =
      torch::zeros({1, token_count, kNumHeads, kHeadDim}, bf16_options);
  const torch::Tensor raw_gate =
      torch::full({token_count, kNumHeads, kHeadDim}, -100.0f, bf16_options);
  const torch::Tensor raw_beta =
      torch::full({token_count, kNumHeads}, -100.0f, bf16_options);
  const torch::Tensor a_log = torch::zeros({kNumHeads}, fp32_options);
  const torch::Tensor dt_bias =
      torch::zeros({kNumHeads * kHeadDim}, fp32_options);
  const torch::Tensor initial_state =
      torch::zeros({1, kNumHeads, kHeadDim, kHeadDim}, fp32_options);
  const std::vector<int64_t> sequence_lengths = {token_count};
  const torch::Tensor cu_seqlens = make_cu_seqlens(sequence_lengths, device);
  const torch::Tensor chunk_indices =
      make_chunk_indices(cu_seqlens, chunk_size);

  ChunkKDA chunk_kda(kNumHeads);
  chunk_kda->to(device);
  auto [output, final_state] =
      chunk_kda->forward_raw_gate(qkv,
                                  qkv,
                                  qkv,
                                  raw_gate,
                                  a_log,
                                  dt_bias,
                                  /*gate_lower_bound=*/-5.0f,
                                  raw_beta,
                                  initial_state,
                                  cu_seqlens,
                                  chunk_indices,
                                  /*output_final_state=*/false,
                                  /*use_qk_l2norm=*/true);
  torch_mlu::synchronize();

  EXPECT_EQ(output.sizes(),
            torch::IntArrayRef({1, token_count, kNumHeads, kHeadDim}));
  EXPECT_TRUE(torch::equal(output, torch::zeros_like(output)));
  EXPECT_FALSE(final_state.defined());
}

TEST(ChunkKDATest, RawSoftplusMatchesIndependentRecurrence) {
  const torch::Device device(torch::kPrivateUse1, /*index=*/0);
  const torch::DeviceGuard guard(device);
  torch::manual_seed(/*seed=*/20261028);
  constexpr int64_t kHeads = 2;
  constexpr int64_t kTokens = 3;
  constexpr float kSoftplusBeta = 1.5f;
  constexpr float kSoftplusThreshold = 2.0f;
  const auto fp32 = torch::TensorOptions().dtype(torch::kFloat32);
  for (const int64_t dimension : {128, 32}) {
    SCOPED_TRACE(dimension);
    const auto options =
        fp32.dtype(dimension == 128 ? torch::kBFloat16 : torch::kFloat32);
    const torch::Tensor q_cpu =
        torch::randn({1, kTokens, kHeads, dimension}, options);
    const torch::Tensor k_cpu = torch::randn_like(q_cpu);
    const torch::Tensor v_cpu = torch::randn_like(q_cpu);
    const torch::Tensor gate_cpu =
        torch::randn({kTokens, kHeads * dimension}, options) * 5.0f;
    const torch::Tensor beta_cpu = torch::randn({kTokens, kHeads}, options);
    const torch::Tensor a_log_cpu = torch::tensor({0.1f, -0.2f}, fp32);
    const torch::Tensor bias_cpu =
        torch::randn({kHeads * dimension}, fp32) * 0.1f;
    const torch::Tensor initial_cpu =
        torch::randn({1, kHeads, dimension, dimension}, fp32) * 0.01f;
    torch::Tensor current = initial_cpu[0].clone();
    torch::Tensor expected_output =
        torch::empty({1, kTokens, kHeads, dimension}, fp32);
    torch::Tensor expected_state =
        torch::empty({kTokens, kHeads, dimension, dimension}, fp32);
    const double scale = 1.0 / std::sqrt(static_cast<double>(dimension));
    for (int64_t token = 0; token < kTokens; ++token) {
      torch::Tensor query = q_cpu[0][token].to(torch::kFloat32);
      torch::Tensor key = k_cpu[0][token].to(torch::kFloat32);
      query = query *
              torch::rsqrt(query.square().sum(/*dim=*/-1, /*keepdim=*/true) +
                           1e-6) *
              scale;
      key = key *
            torch::rsqrt(key.square().sum(/*dim=*/-1, /*keepdim=*/true) + 1e-6);
      const torch::Tensor x =
          gate_cpu[token].to(torch::kFloat32).view({kHeads, dimension}) +
          bias_cpu.view({kHeads, dimension});
      const torch::Tensor bx = x * kSoftplusBeta;
      const torch::Tensor log_gate =
          -torch::exp(a_log_cpu).unsqueeze(/*dim=*/-1) *
          torch::where(bx <= kSoftplusThreshold,
                       torch::log1p(torch::exp(bx)) / kSoftplusBeta,
                       x);
      current *= torch::exp(log_gate).unsqueeze(/*dim=*/-2);
      const torch::Tensor prediction =
          (current * key.unsqueeze(/*dim=*/-2)).sum(/*dim=*/-1);
      const torch::Tensor delta =
          (v_cpu[0][token].to(torch::kFloat32) - prediction) *
          torch::sigmoid(beta_cpu[token].to(torch::kFloat32))
              .unsqueeze(/*dim=*/-1);
      current += delta.unsqueeze(/*dim=*/-1) * key.unsqueeze(/*dim=*/-2);
      expected_output[0][token].copy_(
          (current * query.unsqueeze(/*dim=*/-2)).sum(/*dim=*/-1));
      expected_state[token].copy_(current);
    }
    torch::Tensor q = q_cpu.to(device);
    torch::Tensor k = k_cpu.to(device);
    torch::Tensor v = v_cpu.to(device);
    torch::Tensor gate = gate_cpu.to(device);
    torch::Tensor beta = beta_cpu.to(device);
    torch::Tensor initial = initial_cpu.to(device);
    torch::Tensor indices;
    torch::Tensor cu =
        torch::tensor({0, static_cast<int32_t>(kTokens)}, torch::kInt32)
            .to(device);
    const auto [output, state] = fused_sigmoid_gating_delta_rule_update(
        a_log_cpu.to(device),
        gate,
        beta,
        bias_cpu.to(device),
        q,
        k,
        v,
        initial,
        indices,
        cu,
        scale,
        /*use_qk_l2norm_in_kernel=*/true,
        kSoftplusBeta,
        kSoftplusThreshold,
        /*num_accepted_tokens_opt=*/std::nullopt,
        /*inplace_final_state=*/false,
        /*is_kda=*/true,
        /*kda_use_safe_gate=*/false,
        /*kda_gate_lower_bound=*/-5.0f);
    torch_mlu::synchronize();
    EXPECT_TRUE(torch::allclose(output.cpu().to(torch::kFloat32),
                                expected_output,
                                /*rtol=*/0.02,
                                /*atol=*/0.002));
    EXPECT_TRUE(torch::allclose(state.cpu(),
                                expected_state,
                                /*rtol=*/0.001,
                                /*atol=*/0.0001));
    EXPECT_TRUE(torch::equal(initial.cpu(), initial_cpu));
  }
}

// Non-empty collinear_lengths replace the default lengths with nearly equal
// keys, beta close to one and almost no gate decay. (I + L)^-1 stays O(1)
// there while every power of L has binomial entries.
void check_raw_gate_against_recurrence(
    int64_t num_heads,
    bool use_qk_l2norm,
    bool use_padded_batch = false,
    const std::vector<int64_t>& collinear_lengths = {}) {
  const bool collinear = !collinear_lengths.empty();
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  torch::manual_seed(20260915);
  const int64_t chunk_size = kda_prefill_chunk_size(num_heads, use_qk_l2norm);
  const std::vector<int64_t> sequence_lengths =
      collinear ? collinear_lengths
      : use_padded_batch
          ? std::vector<int64_t>{0,
                                 1,
                                 31,
                                 32,
                                 33,
                                 63,
                                 64,
                                 65,
                                 0,
                                 2,
                                 7,
                                 15,
                                 16,
                                 17,
                                 127,
                                 129}
          : std::vector<int64_t>{1, chunk_size - 1, 2 * chunk_size + 1};
  const int64_t total_tokens = std::accumulate(
      sequence_lengths.begin(), sequence_lengths.end(), int64_t{0});
  const int64_t num_sequences = static_cast<int64_t>(sequence_lengths.size());
  const auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32);
  const std::vector<int64_t> shape = {total_tokens, num_heads, kHeadDim};
  // Round the inputs to the model dtype before evaluating the independent
  // token-by-token recurrence on CPU.
  const torch::Tensor q =
      (torch::randn(shape, cpu_options) * 0.1f).to(torch::kBFloat16);
  const torch::Tensor k =
      (collinear ? torch::randn({1, num_heads, kHeadDim}, cpu_options) * 0.1f +
                       torch::randn(shape, cpu_options) * 0.002f
                 : torch::randn(shape, cpu_options) * 0.1f)
          .to(torch::kBFloat16);
  const torch::Tensor v =
      (torch::randn(shape, cpu_options) * 0.1f).to(torch::kBFloat16);
  const torch::Tensor raw_gate =
      (collinear ? torch::full(shape, -30.0f, cpu_options)
                 : torch::randn(shape, cpu_options) * 2.0f - 2.0f)
          .to(torch::kBFloat16);
  const torch::Tensor raw_beta =
      (collinear ? torch::full({total_tokens, num_heads}, 10.0f, cpu_options)
                 : torch::randn({total_tokens, num_heads}, cpu_options))
          .to(torch::kBFloat16);
  const torch::Tensor a_log = torch::randn({num_heads}, cpu_options) * 0.2f;
  const torch::Tensor dt_bias =
      torch::randn({num_heads * kHeadDim}, cpu_options) * 0.1f;
  constexpr float kGateLowerBound = -5.0f;
  const torch::Tensor gate =
      kGateLowerBound *
      torch::sigmoid(torch::exp(a_log).view({1, num_heads, 1}) *
                     (raw_gate.to(torch::kFloat32) +
                      dt_bias.view({1, num_heads, kHeadDim})));
  const torch::Tensor beta = torch::sigmoid(raw_beta.to(torch::kFloat32));
  const torch::Tensor initial_state =
      torch::randn({num_sequences, num_heads, kHeadDim, kHeadDim},
                   cpu_options) *
      0.01f;
  torch::Tensor q_reference = q.to(torch::kFloat32);
  torch::Tensor k_reference = k.to(torch::kFloat32);
  if (use_qk_l2norm) {
    q_reference *=
        torch::rsqrt(torch::sum(q_reference.square(), -1, true) + 1e-6f);
    k_reference *=
        torch::rsqrt(torch::sum(k_reference.square(), -1, true) + 1e-6f);
  }
  q_reference /= std::sqrt(static_cast<float>(kHeadDim));
  const torch::Tensor v_reference = v.to(torch::kFloat32);
  torch::Tensor expected_output = torch::empty(shape, cpu_options);
  torch::Tensor expected_state = initial_state.clone();
  int64_t token = 0;
  for (int64_t sequence = 0; sequence < num_sequences; ++sequence) {
    torch::Tensor state = expected_state[sequence];
    for (int64_t local_token = 0; local_token < sequence_lengths[sequence];
         ++local_token, ++token) {
      state *= torch::exp(gate[token]).unsqueeze(/*dim=*/-2);
      const torch::Tensor key = k_reference[token].unsqueeze(/*dim=*/-2);
      const torch::Tensor delta =
          (v_reference[token] - torch::sum(state * key, /*dim=*/-1)) *
          beta[token].unsqueeze(/*dim=*/-1);
      state += delta.unsqueeze(/*dim=*/-1) * key;
      expected_output[token].copy_(torch::sum(
          state * q_reference[token].unsqueeze(/*dim=*/-2), /*dim=*/-1));
    }
  }
  const torch::Tensor cu_seqlens =
      make_cu_seqlens(sequence_lengths, device)
          .to(use_padded_batch ? torch::kInt64 : torch::kInt32);
  torch::Tensor chunk_indices;
  if (use_padded_batch) {
    std::vector<int64_t> indices;
    indices.reserve(2 * (total_tokens / chunk_size + num_sequences));
    for (int64_t sequence = 0; sequence < num_sequences; ++sequence) {
      const int64_t chunks =
          (sequence_lengths[sequence] + chunk_size - 1) / chunk_size;
      for (int64_t chunk = 0; chunk < chunks; ++chunk) {
        indices.emplace_back(sequence);
        indices.emplace_back(chunk);
      }
    }
    chunk_indices = torch::tensor(indices, cu_seqlens.options()).view({-1, 2});
  } else {
    chunk_indices = make_chunk_indices(cu_seqlens, chunk_size);
  }

  ChunkKDA chunk_kda(num_heads);
  auto [output, final_state] =
      chunk_kda->forward_raw_gate(q.unsqueeze(/*dim=*/0).to(device),
                                  k.unsqueeze(/*dim=*/0).to(device),
                                  v.unsqueeze(/*dim=*/0).to(device),
                                  raw_gate.to(device),
                                  a_log.to(device),
                                  dt_bias.to(device),
                                  kGateLowerBound,
                                  raw_beta.to(device),
                                  initial_state.to(device),
                                  cu_seqlens,
                                  chunk_indices,
                                  /*output_final_state=*/true,
                                  use_qk_l2norm);
  torch_mlu::synchronize();
  const torch::Tensor actual_output =
      output.squeeze(/*dim=*/0).to(torch::kCPU).to(torch::kFloat32);
  const torch::Tensor actual_state = final_state.to(torch::kCPU);
  EXPECT_TRUE(torch::allclose(actual_output,
                              expected_output,
                              /*rtol=*/5e-3,
                              /*atol=*/2e-5))
      << "num_heads=" << num_heads << " max_error="
      << (actual_output - expected_output).abs().max().item<float>()
      << " max_expected=" << expected_output.abs().max().item<float>();
  // Nearly equal keys amplify ordinary FP32 rounding in the carried state to
  // about 1e-4 even for single 16-token blocks; the old inverse missed by
  // 1e9 or more.
  EXPECT_TRUE(torch::allclose(actual_state,
                              expected_state,
                              /*rtol=*/1e-3,
                              /*atol=*/collinear ? 2e-4 : 2e-5))
      << "num_heads=" << num_heads << " max_error="
      << (actual_state - expected_state).abs().max().item<float>()
      << " max_expected=" << expected_state.abs().max().item<float>();
}

TEST(ChunkKDATest, RawGatePackedBoundariesMatchTokenRecurrence) {
  check_raw_gate_against_recurrence(kNumHeads, /*use_qk_l2norm=*/true);
}

TEST(ChunkKDATest, RawGateSmallHeadCountRetainsGenericPath) {
  check_raw_gate_against_recurrence(/*num_heads=*/2, /*use_qk_l2norm=*/true);
}

TEST(ChunkKDATest, RawGateWithoutNormalizationMatchesTokenRecurrence) {
  check_raw_gate_against_recurrence(kNumHeads, /*use_qk_l2norm=*/false);
}

TEST(ChunkKDATest, CollinearKeysMatchTokenRecurrence) {
  // One long sequence takes the adaptive small-batch path; packed batches
  // take the staged paths for every native head count.
  check_raw_gate_against_recurrence(/*num_heads=*/8,
                                    /*use_qk_l2norm=*/true,
                                    /*use_padded_batch=*/false,
                                    /*collinear_lengths=*/{600});
  for (const int64_t num_heads : {8, 16, 32, 64}) {
    check_raw_gate_against_recurrence(num_heads,
                                      /*use_qk_l2norm=*/true,
                                      /*use_padded_batch=*/false,
                                      /*collinear_lengths=*/{300, 64, 37, 5});
  }
}

TEST(ChunkKDATest, RawGatePaddedBatchMatchesTokenRecurrence) {
  check_raw_gate_against_recurrence(/*num_heads=*/8,
                                    /*use_qk_l2norm=*/true,
                                    /*use_padded_batch=*/true);
  check_raw_gate_against_recurrence(/*num_heads=*/32,
                                    /*use_qk_l2norm=*/true,
                                    /*use_padded_batch=*/true);
}

void check_raw_gate_recurrence(const std::vector<int64_t>& sequence_lengths,
                               int64_t num_heads,
                               int64_t suffix_padding = 0,
                               torch::Tensor* reference_carry = nullptr,
                               torch::Tensor* device_carry = nullptr,
                               bool use_decode = false,
                               int64_t seed = 20261019) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  torch::manual_seed(seed);
  constexpr bool use_qk_l2norm = true;
  constexpr bool use_padded_batch = true;
  const int64_t chunk_size = kda_prefill_chunk_size(num_heads, use_qk_l2norm);
  const int64_t total_tokens = std::accumulate(
      sequence_lengths.begin(), sequence_lengths.end(), int64_t{0});
  const int64_t num_sequences = static_cast<int64_t>(sequence_lengths.size());
  const auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32);
  const int64_t storage_tokens = total_tokens + suffix_padding;
  const std::vector<int64_t> shape = {storage_tokens, num_heads, kHeadDim};
  // Round the inputs to the model dtype before evaluating the independent
  // token-by-token recurrence on CPU.
  const torch::Tensor q =
      (torch::randn(shape, cpu_options) * 0.1f).to(torch::kBFloat16);
  const torch::Tensor k =
      (torch::randn(shape, cpu_options) * 0.1f).to(torch::kBFloat16);
  const torch::Tensor v =
      (torch::randn(shape, cpu_options) * 0.1f).to(torch::kBFloat16);
  const torch::Tensor raw_gate =
      (torch::randn(shape, cpu_options) * 2.0f - 2.0f).to(torch::kBFloat16);
  const torch::Tensor raw_beta =
      torch::randn({storage_tokens, num_heads}, cpu_options)
          .to(torch::kBFloat16);
  // Keep model parameters identical across differently sized chunks.
  torch::manual_seed(20261026);
  const torch::Tensor a_log = torch::randn({num_heads}, cpu_options) * 0.2f;
  const torch::Tensor dt_bias =
      torch::randn({num_heads * kHeadDim}, cpu_options) * 0.1f;
  constexpr float kGateLowerBound = -5.0f;
  const torch::Tensor gate =
      kGateLowerBound *
      torch::sigmoid(torch::exp(a_log).view({1, num_heads, 1}) *
                     (raw_gate.to(torch::kFloat32) +
                      dt_bias.view({1, num_heads, kHeadDim})));
  const torch::Tensor beta = torch::sigmoid(raw_beta.to(torch::kFloat32));
  const torch::Tensor initial_state =
      reference_carry != nullptr && reference_carry->defined()
          ? reference_carry->clone()
          : torch::randn({num_sequences, num_heads, kHeadDim, kHeadDim},
                         cpu_options) *
                0.01f;
  const torch::Tensor input_state =
      device_carry != nullptr && device_carry->defined()
          ? *device_carry
          : initial_state.to(device);
  torch::Tensor q_reference = q.to(torch::kFloat32);
  torch::Tensor k_reference = k.to(torch::kFloat32);
  if (use_qk_l2norm) {
    q_reference *=
        torch::rsqrt(torch::sum(q_reference.square(), -1, true) + 1e-6f);
    k_reference *=
        torch::rsqrt(torch::sum(k_reference.square(), -1, true) + 1e-6f);
  }
  q_reference /= std::sqrt(static_cast<float>(kHeadDim));
  const torch::Tensor v_reference = v.to(torch::kFloat32);
  torch::Tensor expected_output = torch::zeros(shape, cpu_options);
  torch::Tensor expected_state = initial_state.clone();
  int64_t token = 0;
  for (int64_t sequence = 0; sequence < num_sequences; ++sequence) {
    torch::Tensor state = expected_state[sequence];
    for (int64_t local_token = 0; local_token < sequence_lengths[sequence];
         ++local_token, ++token) {
      state *= torch::exp(gate[token]).unsqueeze(/*dim=*/-2);
      const torch::Tensor key = k_reference[token].unsqueeze(/*dim=*/-2);
      const torch::Tensor delta =
          (v_reference[token] - torch::sum(state * key, /*dim=*/-1)) *
          beta[token].unsqueeze(/*dim=*/-1);
      state += delta.unsqueeze(/*dim=*/-1) * key;
      expected_output[token].copy_(torch::sum(
          state * q_reference[token].unsqueeze(/*dim=*/-2), /*dim=*/-1));
    }
  }
  const torch::Tensor cu_seqlens =
      make_cu_seqlens(sequence_lengths, device)
          .to(use_padded_batch ? torch::kInt64 : torch::kInt32);
  torch::Tensor chunk_indices;
  if (use_padded_batch) {
    std::vector<int64_t> indices;
    indices.reserve(2 * (total_tokens / chunk_size + num_sequences));
    for (int64_t sequence = 0; sequence < num_sequences; ++sequence) {
      const int64_t chunks =
          (sequence_lengths[sequence] + chunk_size - 1) / chunk_size;
      for (int64_t chunk = 0; chunk < chunks; ++chunk) {
        indices.emplace_back(sequence);
        indices.emplace_back(chunk);
      }
    }
    chunk_indices = torch::tensor(indices, cu_seqlens.options()).view({-1, 2});
  } else {
    chunk_indices = make_chunk_indices(cu_seqlens, chunk_size);
  }

  ChunkKDA chunk_kda(num_heads);
  auto [output, final_state] =
      chunk_kda->forward_raw_gate(q.unsqueeze(/*dim=*/0).to(device),
                                  k.unsqueeze(/*dim=*/0).to(device),
                                  v.unsqueeze(/*dim=*/0).to(device),
                                  raw_gate.to(device),
                                  a_log.to(device),
                                  dt_bias.to(device),
                                  kGateLowerBound,
                                  raw_beta.to(device),
                                  input_state,
                                  cu_seqlens,
                                  chunk_indices,
                                  /*output_final_state=*/true,
                                  use_qk_l2norm);
  if (use_decode) {
    torch::Tensor decode_q = q.unsqueeze(0).to(device);
    torch::Tensor decode_k = k.unsqueeze(0).to(device);
    torch::Tensor decode_v = v.unsqueeze(0).to(device);
    torch::Tensor decode_gate =
        raw_gate.reshape({storage_tokens, -1}).to(device);
    torch::Tensor decode_beta = raw_beta.to(device);
    torch::Tensor state_pool = torch::cat(
        {torch::zeros_like(input_state.narrow(0, 0, 1)), input_state}, 0);
    torch::Tensor slots =
        torch::arange(1, num_sequences + 1, cu_seqlens.options());
    torch::Tensor starts = cu_seqlens.clone();
    auto decoded = fused_sigmoid_gating_delta_rule_update(
        a_log.to(device),
        decode_gate,
        decode_beta,
        dt_bias.to(device),
        decode_q,
        decode_k,
        decode_v,
        state_pool,
        slots,
        starts,
        /*scale=*/1.0 / std::sqrt(static_cast<double>(kHeadDim)),
        /*use_qk_l2norm_in_kernel=*/true,
        /*softplus_beta=*/1.0f,
        /*softplus_threshold=*/20.0f,
        /*num_accepted_tokens_opt=*/std::nullopt,
        /*inplace_final_state=*/true,
        /*is_kda=*/true,
        /*kda_use_safe_gate=*/true,
        kGateLowerBound);
    output = decoded.first;
    final_state = decoded.second.narrow(0, 1, num_sequences);
  }
  torch_mlu::synchronize();
  const torch::Tensor actual_output =
      output.squeeze(/*dim=*/0).to(torch::kCPU).to(torch::kFloat32);
  EXPECT_TRUE(torch::allclose(
      actual_output.slice(/*dim=*/0, /*start=*/0, /*end=*/total_tokens),
      expected_output.slice(/*dim=*/0, /*start=*/0, /*end=*/total_tokens),
      /*rtol=*/5e-3,
      /*atol=*/2e-5));
  if (suffix_padding > 0) {
    EXPECT_EQ(
        torch::count_nonzero(
            actual_output.slice(
                /*dim=*/0, /*start=*/total_tokens, /*end=*/storage_tokens))
            .item<int64_t>(),
        0);
  }
  EXPECT_TRUE(torch::allclose(final_state.to(torch::kCPU),
                              expected_state,
                              /*rtol=*/1e-3,
                              /*atol=*/2e-5));
  if (reference_carry != nullptr) {
    *reference_carry = expected_state;
  }
  if (device_carry != nullptr) {
    *device_carry = final_state;
  }
}

// These are kernel capability cases with an explicit 32K batch token budget;
// they do not assume that a serving scheduler uses that budget.
TEST(ChunkKDATest, RawGateBudgetedPrefillMatchesIndependentRecurrence) {
  const std::vector<std::vector<int64_t>> batches = {{8192},
                                                     {16384},
                                                     {8192, 8192},
                                                     {8192, 8192, 8192, 8192},
                                                     {16384, 16384},
                                                     {8192, 8192, 3001},
                                                     {16384, 8192, 511, 17}};
  for (const auto& lengths : batches) {
    SCOPED_TRACE(::testing::PrintToString(lengths));
    ASSERT_LE(std::accumulate(lengths.begin(), lengths.end(), int64_t{0}),
              32768);
    check_raw_gate_recurrence(lengths, /*num_heads=*/8);
  }
  check_raw_gate_recurrence({16384}, /*num_heads=*/16);
}

TEST(ChunkKDATest, RawGateLongDispatchBoundariesMatchIndependentRecurrence) {
  for (const int64_t batch : {1, 2, 3, 8, 9}) {
    for (const int64_t length : {255, 256, 257}) {
      SCOPED_TRACE(::testing::Message() << batch << ":" << length);
      check_raw_gate_recurrence(std::vector<int64_t>(batch, length),
                                /*num_heads=*/8);
    }
  }
  for (const int64_t length : {255, 256, 257}) {
    check_raw_gate_recurrence({length}, /*num_heads=*/16);
  }
}

TEST(ChunkKDATest, RawGateChunked264KContinuesThroughTailAndShortQueries) {
  for (const int64_t chunk : {8192, 16384}) {
    SCOPED_TRACE(chunk);
    torch::Tensor reference_state;
    torch::Tensor device_state;
    // Carry the independently computed FP32 state separately from the kernel
    // state. Never seed the oracle with the previous device result.
    int64_t history = 0;
    while (history < 264 * 1024) {
      const int64_t length = std::min(chunk, 264 * 1024 - history);
      SCOPED_TRACE(history);
      check_raw_gate_recurrence({length},
                                /*num_heads=*/8,
                                /*suffix_padding=*/0,
                                &reference_state,
                                &device_state,
                                /*use_decode=*/false,
                                /*seed=*/20261019 + history);
      history += length;
    }
    for (const int64_t length : {37, 1, 4, 1, 6, 8}) {
      check_raw_gate_recurrence({length},
                                /*num_heads=*/8,
                                /*suffix_padding=*/0,
                                &reference_state,
                                &device_state,
                                /*use_decode=*/length <= 8,
                                /*seed=*/20261019 + history);
      history += length;
    }
  }
}

TEST(ChunkKDATest, RawGateEverySmallBatchPreservesRaggedFinalStates) {
  constexpr int64_t kHeads[] = {8, 16, 32, 64};
  constexpr int64_t kLengths[] = {1, 31, 32, 33, 0, 63, 64, 65};
  for (int64_t batch = 1; batch <= 16; ++batch) {
    SCOPED_TRACE(batch);
    std::vector<int64_t> lengths;
    lengths.reserve(batch);
    for (int64_t sequence = 0; sequence < batch; ++sequence) {
      lengths.emplace_back(kLengths[(sequence + batch - 1) % 8]);
    }
    check_raw_gate_recurrence(lengths, kHeads[(batch - 1) % 4]);
  }
}

TEST(ChunkKDATest, RawGateGlobalPaddingIsZeroAndPreservesFinalState) {
  // Exercise both staged and direct paths with unused graph storage after
  // cu[-1].
  for (int32_t repeat = 0; repeat < 3; ++repeat) {
    SCOPED_TRACE(repeat);
    check_raw_gate_recurrence({257}, /*num_heads=*/8, /*suffix_padding=*/17);
    check_raw_gate_recurrence(
        {31, 0, 33}, /*num_heads=*/16, /*suffix_padding=*/7);
  }
}

}  // namespace
}  // namespace xllm::kernel::mlu
