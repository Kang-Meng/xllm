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
#include <torch/torch.h>

#include <cmath>
#include <cstdint>

#include "kernels/mlu/bangc/include/ple.h"

namespace xllm::kernel::mlu::bangc {
namespace {
constexpr int64_t kH = 2560;
constexpr int64_t kD = 10240;
constexpr int64_t kN = 12800;

// Keep the reference on CPU and use the original, unpacked projection matrix.
// Explicit BF16 boundaries model the public kernel's reduced-precision math;
// all reductions, the gate and the convolution accumulate in FP32.
torch::Tensor round_bf16(const torch::Tensor& tensor) {
  return tensor.to(torch::kBFloat16).to(torch::kFloat32);
}

struct PleReference {
  torch::Tensor kv;
  torch::Tensor values;
  torch::Tensor normalized;
  torch::Tensor output;
  torch::Tensor history;
};

struct PleHeadReference {
  torch::Tensor values;
  torch::Tensor normalized;
};

PleHeadReference ple_head_reference(const torch::Tensor& kv,
                                    const torch::Tensor& hidden,
                                    const torch::Tensor& key_scale,
                                    const torch::Tensor& query_scale,
                                    const torch::Tensor& conv_scale) {
  const int64_t rows = kv.size(0);
  const torch::Tensor key =
      kv.slice(/*dim=*/1, /*start=*/0, kD).reshape({rows, 4, kH});
  const torch::Tensor query = hidden.to(torch::kFloat32).reshape({rows, 4, kH});
  const torch::Tensor value = kv.slice(/*dim=*/1, kD).reshape({rows, 1, kH});
  const torch::Tensor key_inv = torch::rsqrt(
      round_bf16(key.square()).mean(/*dim=*/-1, /*keepdim=*/true) + 1.0e-6);
  const torch::Tensor query_inv = torch::rsqrt(
      round_bf16(query.square()).mean(/*dim=*/-1, /*keepdim=*/true) + 1.0e-6);
  const torch::Tensor affine =
      round_bf16(key_scale * query_scale).reshape({1, 4, kH});
  const torch::Tensor dot = round_bf16(round_bf16(key * query) * affine)
                                .sum(/*dim=*/-1, /*keepdim=*/true) *
                            key_inv * query_inv /
                            std::sqrt(static_cast<double>(kH));
  const torch::Tensor gate = torch::sigmoid(
      torch::sign(dot) * torch::sqrt(torch::clamp_min(dot.abs(), 1.0e-6)));
  const torch::Tensor values =
      round_bf16(value * round_bf16(gate)).reshape({rows, kD});
  const torch::Tensor value_mean =
      round_bf16(value.square()).mean(/*dim=*/-1, /*keepdim=*/true);
  const torch::Tensor conv_inv =
      gate * torch::rsqrt(value_mean * gate.square() + 1.0e-6);
  const torch::Tensor normalized =
      round_bf16(round_bf16(value * round_bf16(conv_inv)) *
                 round_bf16(conv_scale).reshape({1, 4, kH}))
          .reshape({rows, kD});
  return {values, normalized};
}

PleReference ple_reference(const torch::Tensor& embedding,
                           const torch::Tensor& hidden,
                           const torch::Tensor& weight,
                           const torch::Tensor& key_scale,
                           const torch::Tensor& query_scale,
                           const torch::Tensor& conv_scale,
                           const torch::Tensor& taps,
                           const torch::Tensor& history) {
  const int64_t rows = embedding.size(0);
  const int64_t batch = rows / 4;
  const torch::Tensor kv = round_bf16(torch::matmul(
      embedding.to(torch::kFloat32), weight.to(torch::kFloat32).t()));
  const PleHeadReference head =
      ple_head_reference(kv, hidden, key_scale, query_scale, conv_scale);
  const torch::Tensor updated_history = torch::cat(
      {history.slice(/*dim=*/1, /*start=*/0, /*end=*/9).to(torch::kFloat32),
       head.normalized.reshape({batch, 4, kD})},
      /*dim=*/1);
  // Four causal taps with dilation 3: token q reads q, q+3, q+6, q+9
  // from the concatenation of nine old states and four current states.
  torch::Tensor convolved = torch::zeros({batch, 4, kD}, torch::kFloat32);
  const torch::Tensor narrow_taps = round_bf16(taps);
  for (int64_t tap = 0; tap < 4; ++tap) {
    convolved += updated_history.slice(/*dim=*/1, 3 * tap, 3 * tap + 4) *
                 narrow_taps.select(/*dim=*/0, tap);
  }
  convolved = round_bf16(convolved).reshape({rows, kD});
  const torch::Tensor activated =
      round_bf16(convolved * torch::sigmoid(convolved));
  const torch::Tensor output =
      round_bf16(activated + head.values + hidden.to(torch::kFloat32));
  return {kv, head.values, head.normalized, output, updated_history};
}

void expect_close(const torch::Tensor& actual,
                  const torch::Tensor& expected,
                  const char* name,
                  double rtol = 0.008,
                  double atol = 0.001) {
  SCOPED_TRACE(name);
  const torch::Tensor cpu_actual = actual.cpu().to(torch::kFloat32);
  EXPECT_TRUE(torch::allclose(cpu_actual, expected, rtol, atol))
      << "max absolute error: "
      << (cpu_actual - expected).abs().max().item<float>();
}

TEST(Qwen4PleBangcTest, NonzeroInputsMatchIndependentReferenceForEveryBatch) {
  torch::NoGradGuard no_grad;
  torch::manual_seed(/*seed=*/20261019);
  const torch::Device device(torch::kPrivateUse1, /*index=*/0);
  const torch::DeviceGuard guard(device);
  // Seeded dense weights distinguish both projection axes, including every
  // packing tile and HC group. Inputs and references are rounded on CPU first.
  const torch::Tensor weight =
      (torch::randn({kN, kH}, torch::kFloat32) / 32).to(torch::kBFloat16);
  const torch::Tensor channel = torch::arange(kD, torch::kFloat32);
  const torch::Tensor key_scale = 0.5 + channel.remainder(17) / 32;
  const torch::Tensor query_scale = 0.25 + channel.remainder(23) / 32;
  const torch::Tensor conv_scale =
      (0.5 + channel.remainder(29) / 32) * (1 - 2 * channel.remainder(2));
  const torch::Tensor taps =
      (torch::arange(4 * kD, torch::kFloat32).remainder(31) - 15)
          .reshape({4, kD}) /
      32;
  const torch::Tensor device_weight = weight.to(device);
  torch::Tensor packed = torch::empty_like(device_weight);
  torch::Tensor affine = torch::empty({4, 2, kH}, device_weight.options());
  ple_prepare_weights_out(device_weight,
                          key_scale.to(device),
                          query_scale.to(device),
                          conv_scale.to(device),
                          packed,
                          affine);
  const torch::Tensor cpu_affine = affine.cpu();
  EXPECT_TRUE(torch::equal(
      cpu_affine.select(/*dim=*/1, /*index=*/0),
      (key_scale * query_scale).to(torch::kBFloat16).reshape({4, kH})));
  EXPECT_TRUE(torch::equal(cpu_affine.select(/*dim=*/1, /*index=*/1),
                           conv_scale.to(torch::kBFloat16).reshape({4, kH})));
  const torch::Tensor device_taps = taps.to(device);

  // B=1 exercises the cooperative head; B=32 selects the larger projection.
  for (const int64_t batch : {1, 2, 4, 8, 16, 32}) {
    SCOPED_TRACE(batch);
    const int64_t rows = batch * 4;
    const torch::Tensor embedding =
        torch::randn({rows, kH}, torch::kFloat32).to(torch::kBFloat16);
    const torch::Tensor hidden =
        torch::randn({rows, kD}, torch::kFloat32).to(torch::kBFloat16);
    const torch::Tensor initial_history =
        torch::randn({batch, 13, kD}, torch::kFloat32).to(torch::kBFloat16);
    const PleReference expected = ple_reference(embedding,
                                                hidden,
                                                weight,
                                                key_scale,
                                                query_scale,
                                                conv_scale,
                                                taps,
                                                initial_history);
    torch::Tensor history = initial_history.to(device);
    torch::Tensor kv = torch::empty({rows, kN}, device_weight.options());
    torch::Tensor values = torch::empty({rows, kD}, device_weight.options());
    torch::Tensor normalized = torch::empty_like(values);
    torch::Tensor output = torch::empty_like(values);
    ple_forward_out(embedding.to(device),
                    hidden.to(device),
                    packed,
                    affine,
                    device_taps,
                    history,
                    kv,
                    values,
                    normalized,
                    output);
    expect_close(kv, expected.kv, "kv");
    // Projection reduction order can differ by one BF16 ULP. Allow accumulated
    // rounding after the head's additional BF16 products, and check the head
    // separately against actual kv with the tighter single-stage tolerance.
    expect_close(values, expected.values, "values", /*rtol=*/0.016);
    expect_close(normalized, expected.normalized, "normalized", /*rtol=*/0.016);
    const PleHeadReference expected_head =
        ple_head_reference(kv.cpu().to(torch::kFloat32),
                           hidden,
                           key_scale,
                           query_scale,
                           conv_scale);
    expect_close(values, expected_head.values, "head values");
    expect_close(normalized, expected_head.normalized, "head normalized");
    // Also allow cancellation near zero after convolution and both residuals.
    expect_close(
        output, expected.output, "output", /*rtol=*/0.016, /*atol=*/0.008);
    expect_close(history, expected.history, "history", /*rtol=*/0.016);
    const torch::Tensor cpu_history = history.cpu();
    EXPECT_TRUE(
        torch::equal(cpu_history.slice(/*dim=*/1, /*start=*/0, /*end=*/9),
                     initial_history.slice(/*dim=*/1, /*start=*/0, /*end=*/9)));
    EXPECT_TRUE(torch::equal(cpu_history.slice(/*dim=*/1, /*start=*/9),
                             normalized.cpu().reshape({batch, 4, kD})));
    for (int64_t accepted = 0; accepted <= 4; ++accepted) {
      SCOPED_TRACE(accepted);
      expect_close(cpu_history.slice(/*dim=*/1, accepted, accepted + 9),
                   expected.history.slice(/*dim=*/1, accepted, accepted + 9),
                   "accepted history prefix",
                   /*rtol=*/0.016);
    }
  }
}

TEST(Qwen4PleBangcTest, TensorApiRetainsOuterResidualAndEveryAcceptedPrefix) {
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, /*index=*/0);
  const torch::DeviceGuard guard(device);
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const torch::Tensor weight = torch::zeros({kN, kH}, options);
  const torch::Tensor key_scale =
      torch::ones({kD}, options.dtype(torch::kFloat32));
  const torch::Tensor query_scale = torch::full_like(key_scale, 2);
  const torch::Tensor conv_scale = torch::full_like(key_scale, 3);
  const torch::Tensor taps =
      torch::zeros({4, kD}, options.dtype(torch::kFloat32));
  torch::Tensor packed = torch::empty_like(weight);
  torch::Tensor affine = torch::empty({4, 2, kH}, options);
  ple_prepare_weights_out(
      weight, key_scale, query_scale, conv_scale, packed, affine);
  EXPECT_TRUE(
      torch::equal(packed.cpu(), torch::zeros({kN, kH}, torch::kBFloat16)));
  EXPECT_TRUE(torch::equal(affine.select(/*dim=*/1, /*index=*/0).cpu(),
                           torch::full({4, kH}, 2, torch::kBFloat16)));
  EXPECT_TRUE(torch::equal(affine.select(/*dim=*/1, /*index=*/1).cpu(),
                           torch::full({4, kH}, 3, torch::kBFloat16)));

  for (const int64_t batch : {1, 2, 4, 8, 16, 32}) {
    SCOPED_TRACE(batch);
    const int64_t rows = batch * 4;
    const torch::Tensor embedding = torch::ones({rows, kH}, options);
    const torch::Tensor hidden =
        (torch::arange(rows * kD, torch::kFloat32).remainder(23) / 8 - 1)
            .reshape({rows, kD})
            .to(options);
    const torch::Tensor initial_history = torch::ones({batch, 13, kD}, options);
    torch::Tensor history = initial_history.clone();
    torch::Tensor kv = torch::empty({rows, kN}, options);
    torch::Tensor values = torch::empty({rows, kD}, options);
    torch::Tensor normalized = torch::empty_like(values);
    torch::Tensor output = torch::empty_like(values);
    ple_forward_out(embedding,
                    hidden,
                    packed,
                    affine,
                    taps,
                    history,
                    kv,
                    values,
                    normalized,
                    output);
    EXPECT_TRUE(torch::equal(output.cpu(), hidden.cpu()));
    EXPECT_TRUE(
        torch::equal(kv.cpu(), torch::zeros({rows, kN}, torch::kBFloat16)));
    EXPECT_TRUE(
        torch::equal(values.cpu(), torch::zeros({rows, kD}, torch::kBFloat16)));
    EXPECT_TRUE(torch::equal(normalized.cpu(),
                             torch::zeros({rows, kD}, torch::kBFloat16)));
    const torch::Tensor expected_history = initial_history.clone();
    expected_history.slice(/*dim=*/1, /*start=*/9).zero_();
    EXPECT_TRUE(torch::equal(history.cpu(), expected_history.cpu()));
    for (int64_t accepted = 0; accepted <= 4; ++accepted) {
      SCOPED_TRACE(accepted);
      EXPECT_TRUE(torch::equal(
          history.slice(/*dim=*/1, accepted, accepted + 9).cpu(),
          expected_history.slice(/*dim=*/1, accepted, accepted + 9).cpu()));
    }
  }
}

TEST(Qwen4PleBangcDeathTest, RejectsOverlappingHeadOutputs) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        const torch::Device device(torch::kPrivateUse1, /*index=*/0);
        const torch::DeviceGuard guard(device);
        const torch::TensorOptions options =
            torch::TensorOptions().dtype(torch::kBFloat16).device(device);
        const torch::Tensor kv = torch::empty({4, kN}, options);
        const torch::Tensor hidden = torch::empty({4, kD}, options);
        const torch::Tensor affine = torch::empty({4, 2, kH}, options);
        torch::Tensor values = torch::empty({4, kD}, options);
        torch::Tensor normalized = values;
        ple_head_out(kv, hidden, affine, values, normalized);
      },
      "values must not overlap normalized");
}

TEST(Qwen4PleBangcDeathTest, RejectsPartiallyOverlappingHeadInput) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        const torch::Device device(torch::kPrivateUse1, /*index=*/0);
        const torch::DeviceGuard guard(device);
        const torch::TensorOptions options =
            torch::TensorOptions().dtype(torch::kBFloat16).device(device);
        const torch::Tensor kv = torch::empty({4, kN}, options);
        const torch::Tensor storage = torch::empty({5, kD}, options);
        const torch::Tensor hidden =
            storage.narrow(/*dim=*/0, /*start=*/0, /*length=*/4);
        torch::Tensor values =
            storage.narrow(/*dim=*/0, /*start=*/1, /*length=*/4);
        const torch::Tensor affine = torch::empty({4, 2, kH}, options);
        torch::Tensor normalized = torch::empty({4, kD}, options);
        ple_head_out(kv, hidden, affine, values, normalized);
      },
      "values must not overlap hidden");
}

TEST(Qwen4PleBangcDeathTest, RejectsCrossStageForwardOverlap) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        const torch::Device device(torch::kPrivateUse1, /*index=*/0);
        const torch::DeviceGuard guard(device);
        const torch::TensorOptions options =
            torch::TensorOptions().dtype(torch::kBFloat16).device(device);
        torch::Tensor normalized = torch::empty({4, kD}, options);
        const torch::Tensor embedding = normalized.flatten()
                                            .narrow(/*dim=*/0,
                                                    /*start=*/0,
                                                    /*length=*/4 * kH)
                                            .reshape({4, kH});
        const torch::Tensor hidden = torch::empty({4, kD}, options);
        const torch::Tensor packed = torch::empty({kN, kH}, options);
        const torch::Tensor affine = torch::empty({4, 2, kH}, options);
        const torch::Tensor taps =
            torch::empty({4, kD}, options.dtype(torch::kFloat32));
        torch::Tensor history = torch::empty({1, 13, kD}, options);
        torch::Tensor kv = torch::empty({4, kN}, options);
        torch::Tensor values = torch::empty({4, kD}, options);
        torch::Tensor output = torch::empty({4, kD}, options);
        ple_forward_out(embedding,
                        hidden,
                        packed,
                        affine,
                        taps,
                        history,
                        kv,
                        values,
                        normalized,
                        output);
      },
      "normalized must not overlap embedding");
}

TEST(Qwen4PleBangcDeathTest, RejectsCpuPreparationBeforeKernelLaunch) {
  const torch::Tensor scale = torch::ones({kD}, torch::kFloat32);
  torch::Tensor affine = torch::empty({4, 2, kH}, torch::kBFloat16);
  EXPECT_DEATH(ple_prepare_head_out(scale, scale, scale, affine),
               "Check failed");
}

}  // namespace
}  // namespace xllm::kernel::mlu::bangc
