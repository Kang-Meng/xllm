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

#include "core/kernels/mlu/dflash2_grouped_conv.h"

#include <glog/logging.h>

#include "core/kernels/mlu/bangc/include/dflash2_grouped_conv.h"

namespace xllm::kernel::mlu {

bool supports_dflash2_grouped_conv(const torch::Tensor& hidden_states,
                                   const torch::Tensor& delta,
                                   const torch::Tensor& base,
                                   int32_t group_size) {
  const bool dtype_supported =
      hidden_states.scalar_type() == torch::kBFloat16 ||
      hidden_states.scalar_type() == torch::kFloat16 ||
      hidden_states.scalar_type() == torch::kFloat32;
  if (group_size > 2048 || !hidden_states.device().is_privateuseone() ||
      !dtype_supported || delta.scalar_type() != hidden_states.scalar_type() ||
      base.scalar_type() != hidden_states.scalar_type() ||
      delta.device() != hidden_states.device() ||
      base.device() != hidden_states.device()) {
    return false;
  }
  return true;
}

torch::Tensor dflash2_grouped_conv(const torch::Tensor& hidden_states,
                                   const torch::Tensor& delta,
                                   const torch::Tensor& base,
                                   int32_t block_size,
                                   int32_t group_size,
                                   int32_t taps) {
  CHECK_GT(base.dim(), 0);
  CHECK_EQ(base.size(0), taps);
  return bangc::dflash2_grouped_conv(
      hidden_states, delta, base, block_size, group_size);
}

}  // namespace xllm::kernel::mlu
