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

#include "core/layers/common/dflash2_grouped_conv.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace xllm::layer {
namespace {

TEST(DFlash2GroupedConvCommonTest, LoadsBothSidesWithoutDeviceBackend) {
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  DFlash2GroupedConv module(4, 2, 2, 4, options);
  torch::Tensor base = torch::ones({2, 2, 4}, options);
  base.select(0, 1).select(0, 0).fill_(2);
  base.select(0, 1).select(0, 1).fill_(3);
  torch::Tensor projection = torch::zeros({8, 4}, options);
  // Each side has a different input-dependent current-token coefficient.
  projection.slice(0, 0, 2).select(1, 0).fill_(1);
  projection.slice(0, 4, 6).select(1, 0).fill_(2);
  module->load_state_dict(StateDict(
      {{"base_kernel", base}, {"kernel_projection.weight", projection}}));
  module->verify_loaded_weights("attention_conv.");
  const auto parameters = module->named_parameters();
  ASSERT_EQ(parameters.size(), 2);
  EXPECT_TRUE(torch::equal(parameters["base_kernel"], base));
  EXPECT_TRUE(torch::equal(parameters["kernel_projection.weight"], projection));
  torch::Tensor hidden =
      torch::arange(1, 9, options).view({8, 1}).repeat({1, 4});
  const torch::Tensor original = hidden.clone();
  auto [prepared, coefficients] = module->prepare(hidden);
  const torch::Tensor expected_prepare =
      torch::tensor({2, 7, 14, 23, 30, 47, 62, 79}, options)
          .view({8, 1})
          .repeat({1, 4});
  EXPECT_TRUE(torch::equal(prepared, expected_prepare));
  // Finish retains the projection from the original input, not a new projection
  // of the prepared tensor. Anchors at rows 0 and 4 have no history.
  const torch::Tensor expected_finish =
      torch::tensor({8, 48, 133, 272, 360, 748, 1133, 1608}, options)
          .view({8, 1})
          .repeat({1, 4});
  EXPECT_TRUE(
      torch::equal(module->finish(prepared, coefficients), expected_finish));
  EXPECT_TRUE(torch::equal(hidden, original));
  EXPECT_TRUE(prepared.device().is_cpu());
}

}  // namespace
}  // namespace xllm::layer
