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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "kernels/mlu/mlu_ops_api.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {
namespace {

class CausalConv1dPrefillStateTest
    : public ::testing::TestWithParam<std::tuple<int64_t,
                                                 torch::ScalarType,
                                                 bool,
                                                 bool,
                                                 torch::ScalarType>> {};

TEST_P(CausalConv1dPrefillStateTest,
       MatchesRestoreCopyAndPreservesSharedSources) {
  const auto [width, dtype, production_layout, has_bias, index_dtype] =
      GetParam();
  const int64_t channels = 2065;
  const int64_t state_len = width - 1;
  const torch::Device device(Platform::type_torch(), 0);
  Device xllm_device(device);
  xllm_device.set_seed(42);
  torch::manual_seed(42);
  const auto cpu_options = torch::TensorOptions().dtype(dtype);
  const auto index_options = torch::TensorOptions().dtype(torch::kInt32);
  const std::vector<int64_t> lengths = {1, 2, 7, 65, 1, 2, 1};
  const std::vector<int64_t> reads = {1, 1, 2, 8, 0, 2, 0};
  const std::vector<int64_t> writes = {3, 4, 5, 8, 6, 7, 0};
  const std::vector<bool> valid = {true, true, true, true, false, false, false};
  std::vector<int32_t> starts = {0};
  std::vector<int32_t> batch;
  std::vector<int32_t> offsets;
  starts.reserve(lengths.size() + 1);
  batch.reserve(lengths.size() + 2);
  offsets.reserve(lengths.size() + 2);
  for (int32_t row = 0; row < static_cast<int32_t>(lengths.size()); ++row) {
    starts.emplace_back(starts.back() + static_cast<int32_t>(lengths[row]));
    for (int32_t offset = 0; offset < (lengths[row] + 63) / 64; ++offset) {
      batch.emplace_back(row);
      offsets.emplace_back(offset);
    }
  }
  // A padded task and a null destination must leave output/state untouched.
  batch.emplace_back(-1);
  offsets.emplace_back(-1);
  // Production tensors are views over token-major activations and
  // [slot, history, channel] state storage. Keep contiguous layouts covered
  // too.
  auto x_cpu = production_layout
                   ? torch::randn({starts.back(), channels}, cpu_options)
                             .transpose(/*dim0=*/0, /*dim1=*/1) *
                         0.1
                   : torch::randn({channels, starts.back()}, cpu_options) * 0.1;
  auto weight_cpu = torch::randn({channels, width}, cpu_options) * 0.1;
  auto bias_cpu = torch::randn({channels}, cpu_options) * 0.1;
  auto initial_cpu = production_layout
                         ? torch::randn({9, state_len, channels}, cpu_options)
                               .transpose(/*dim0=*/1, /*dim1=*/2)
                         : torch::randn({9, channels, state_len}, cpu_options);
  auto expected_state = initial_cpu.clone();
  auto expected_output = torch::zeros_like(x_cpu);
  for (int64_t row = 0; row < static_cast<int64_t>(lengths.size()) - 1; ++row) {
    auto history = valid[row]
                       ? initial_cpu[reads[row]]
                       : torch::zeros({channels, state_len}, cpu_options);
    auto sequence = x_cpu.narrow(1, starts[row], lengths[row]);
    auto joined =
        torch::cat({history, sequence}, /*dim=*/1).to(torch::kFloat32);
    expected_state[writes[row]].copy_(
        joined.narrow(1, lengths[row], state_len).to(dtype));
    for (int64_t token = 0; token < lengths[row]; ++token) {
      auto value = has_bias ? bias_cpu.to(torch::kFloat32).clone()
                            : torch::zeros({channels},
                                           cpu_options.dtype(torch::kFloat32));
      for (int64_t tap = 0; tap < width; ++tap) {
        value += joined.select(1, token + tap) *
                 weight_cpu.select(1, tap).to(torch::kFloat32);
      }
      expected_output.select(1, starts[row] + token)
          .copy_((value * torch::sigmoid(value)).to(dtype));
    }
  }

  auto x = x_cpu.to(device);
  auto weight = weight_cpu.to(device);
  auto bias = bias_cpu.to(device);
  auto states = initial_cpu.to(device);
  auto baseline = initial_cpu.clone();
  for (int64_t row = 0; row < 4; ++row) {
    baseline[writes[row]].copy_(initial_cpu[reads[row]]);
  }
  baseline = baseline.to(device);
  // Device transfer and clone must retain the selected layout.
  EXPECT_EQ(x.stride(0), production_layout ? 1 : starts.back());
  EXPECT_EQ(states.stride(1), production_layout ? 1 : state_len);
  EXPECT_EQ(baseline.stride(1), states.stride(1));
  auto start_tensor = torch::tensor(starts, index_options).to(device);
  auto batch_tensor = torch::tensor(batch, index_options).to(device);
  auto offset_tensor = torch::tensor(offsets, index_options).to(device);
  auto read_tensor = torch::tensor(reads, index_dtype).to(device);
  auto write_tensor = torch::tensor(writes, index_dtype).to(device);
  auto valid_tensor =
      torch::tensor({1, 1, 1, 1, 0, 0, 0}, torch::kInt32).to(device);
  const int32_t nt = static_cast<int32_t>(batch.size());
  const std::optional<torch::Tensor> bias_opt =
      has_bias ? std::make_optional(bias) : std::nullopt;
  auto actual = kernel::mlu::causal_conv1d_fn(x,
                                              weight,
                                              states,
                                              start_tensor,
                                              batch_tensor,
                                              offset_tensor,
                                              nt,
                                              bias_opt,
                                              write_tensor,
                                              valid_tensor,
                                              std::nullopt,
                                              std::nullopt,
                                              /*inplace_final_state=*/true,
                                              read_tensor);
  auto restored = kernel::mlu::causal_conv1d_fn(x,
                                                weight,
                                                baseline,
                                                start_tensor,
                                                batch_tensor,
                                                offset_tensor,
                                                nt,
                                                bias_opt,
                                                write_tensor,
                                                valid_tensor,
                                                std::nullopt,
                                                std::nullopt,
                                                /*inplace_final_state=*/true);
  xllm_device.synchronize_default_stream();

  const torch::Tensor actual_cpu = actual.cpu();
  const torch::Tensor states_cpu = states.cpu();
  EXPECT_TRUE(torch::equal(actual_cpu, restored.cpu()));
  EXPECT_TRUE(torch::equal(states_cpu, baseline.cpu()));
  EXPECT_TRUE(torch::equal(states_cpu, expected_state));
  EXPECT_TRUE(torch::equal(states_cpu[1], initial_cpu[1]));
  EXPECT_TRUE(torch::equal(states_cpu[2], initial_cpu[2]));
  EXPECT_TRUE(torch::allclose(actual_cpu.to(torch::kFloat32),
                              expected_output.to(torch::kFloat32),
                              /*rtol=*/0.02,
                              /*atol=*/0.002));
}

TEST(CausalConv1dPrefillStateContractTest,
     RejectsInvalidMetadataBeforeDeviceWork) {
  const auto x = torch::empty({4, 1});
  const auto weight = torch::empty({4, 3});
  const auto states = torch::empty({3, 4, 2});
  const auto indices = torch::ones({1}, torch::kInt32);
  const auto starts = torch::tensor({0, 1}, torch::kInt32);
  const auto task = torch::zeros({1}, torch::kInt32);
  const auto invoke = [&](const torch::Tensor& reads,
                          const std::optional<torch::Tensor>& checkpoint,
                          const std::optional<torch::Tensor>& accepted,
                          bool inplace) {
    return kernel::mlu::causal_conv1d_fn(x,
                                         weight,
                                         states,
                                         starts,
                                         task,
                                         task,
                                         /*nt=*/1,
                                         /*bias_opt=*/std::nullopt,
                                         indices,
                                         indices,
                                         checkpoint,
                                         accepted,
                                         inplace,
                                         reads);
  };
  EXPECT_DEATH(
      invoke(indices.reshape({1, 1}), std::nullopt, std::nullopt, true), "dim");
  EXPECT_DEATH(
      invoke(indices.to(torch::kFloat32), std::nullopt, std::nullopt, true),
      "scalar_type");
  EXPECT_DEATH(invoke(indices, indices, std::nullopt, true),
               "does not support checkpoint indices");
  EXPECT_DEATH(invoke(indices, std::nullopt, indices, true),
               "does not support speculative verification");
  EXPECT_DEATH(invoke(indices, std::nullopt, std::nullopt, false),
               "writes final state to the cache");
}

TEST(CausalConv1dPrefillStateContractTest,
     RejectsUnsupportedWidthsBeforeDeviceWork) {
  const torch::Tensor x = torch::empty({4, 1});
  const torch::Tensor states = torch::empty({3, 4, 5});
  const torch::Tensor indices = torch::ones({1}, torch::kInt32);
  const torch::Tensor starts = torch::tensor({0, 1}, torch::kInt32);
  const torch::Tensor task = torch::zeros({1}, torch::kInt32);
  for (const int64_t width : {0, 1, 6}) {
    const torch::Tensor weight = torch::empty({4, width});
    for (const bool distinct : {false, true}) {
      const std::optional<torch::Tensor> reads =
          distinct ? std::make_optional(indices) : std::nullopt;
      EXPECT_DEATH(kernel::mlu::causal_conv1d_fn(
                       x,
                       weight,
                       states,
                       starts,
                       task,
                       task,
                       /*nt=*/1,
                       /*bias_opt=*/std::nullopt,
                       indices,
                       indices,
                       /*initial_state_idx_opt=*/std::nullopt,
                       /*num_accepted_tokens_opt=*/std::nullopt,
                       /*inplace_final_state=*/true,
                       reads),
                   "weight.size");
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Prefill,
    CausalConv1dPrefillStateTest,
    ::testing::Combine(::testing::Values(2, 3, 4, 5),
                       ::testing::Values(torch::kBFloat16, torch::kFloat16),
                       ::testing::Bool(),
                       ::testing::Bool(),
                       ::testing::Values(torch::kInt32, torch::kInt64)));

}  // namespace
}  // namespace xllm
