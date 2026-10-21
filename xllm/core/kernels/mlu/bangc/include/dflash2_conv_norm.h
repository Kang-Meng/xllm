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

#pragma once
#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <tuple>
namespace xllm::kernel::mlu::bangc {
// On supported inputs, updates residual in place and returns the same alias.
// Unsupported inputs return nullopt without launching or modifying tensors.
std::optional<std::tuple<torch::Tensor, torch::Tensor>> dflash2_conv_norm(
    const torch::Tensor& hidden,
    const torch::Tensor& delta,
    const torch::Tensor& base,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    int32_t block_size,
    double eps);
}  // namespace xllm::kernel::mlu::bangc
