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

#include "kernels/mlu/bangc/include/dflash2_conv_norm.h"

#include <glog/logging.h>
#include <torch/library.h>

#include <utility>

namespace {
std::tuple<torch::Tensor, torch::Tensor> dflash2_conv_norm(
    const torch::Tensor& hidden,
    const torch::Tensor& delta,
    const torch::Tensor& base,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    int64_t block_size,
    double eps) {
  CHECK(block_size == 4 || block_size == 6);
  auto result = xllm::kernel::mlu::bangc::dflash2_conv_norm(
      hidden,
      delta,
      base,
      residual,
      weight,
      static_cast<int32_t>(block_size),
      eps);
  CHECK(result.has_value())
      << "Unsupported DFlash2 conv/RMSNorm fusion contract";
  return std::move(*result);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(xllm_mlu, m) {
  m.def(
      "dflash2_conv_norm(Tensor hidden, Tensor delta, Tensor base, "
      "Tensor(a!) residual, Tensor weight, int block_size, float eps) "
      "-> (Tensor, Tensor(a!))");
}
TORCH_LIBRARY_IMPL(xllm_mlu, PrivateUse1, m) {
  m.impl("dflash2_conv_norm", TORCH_FN(dflash2_conv_norm));
}
