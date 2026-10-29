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

#include <torch/torch.h>

#include <cmath>
#include <limits>

#include "framework/core/MLUStream.h"
#include "kernels/mlu/bangc/include/bangc_kda_norm_gate_api.h"
#include "kernels/mlu/bangc/src/kda/launchers.h"

namespace xllm::kernel::mlu {

std::optional<torch::Tensor> bangc_kda_norm_gate(const torch::Tensor& input,
                                                 const torch::Tensor& gate,
                                                 const torch::Tensor& weight,
                                                 double eps) {
  if (!input.defined() || !gate.defined() || !weight.defined() ||
      !input.device().is_privateuseone() ||
      input.scalar_type() != torch::kBFloat16 ||
      gate.scalar_type() != torch::kBFloat16 ||
      weight.scalar_type() != torch::kBFloat16 || input.dim() != 2 ||
      input.size(1) != 128 || gate.sizes() != input.sizes() ||
      weight.dim() != 1 || weight.numel() != 128 ||
      gate.device() != input.device() || weight.device() != input.device() ||
      !input.is_contiguous() || !gate.is_contiguous() ||
      !weight.is_contiguous() || !std::isfinite(eps) || eps < 0.0 ||
      eps > std::numeric_limits<float>::max()) {
    return std::nullopt;
  }
  const torch::DeviceGuard device_guard(input.device());
  torch::Tensor output = torch::empty_like(input);
  if (input.numel() == 0) {
    return output;
  }
  ::kda_norm_gate(
      static_cast<void*>(torch_mlu::getCurMLUStream(input.device().index())),
      input.data_ptr(),
      gate.data_ptr(),
      weight.data_ptr(),
      output.data_ptr(),
      input.size(0),
      static_cast<float>(eps));
  return output;
}

}  // namespace xllm::kernel::mlu
