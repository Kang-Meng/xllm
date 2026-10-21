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

#include "core/layers/mlu/dflash2_grouped_conv.h"

#include <glog/logging.h>

#include <utility>

#include "core/kernels/mlu/bangc/include/dflash2_conv_norm.h"
#include "core/kernels/mlu/dflash2_grouped_conv.h"
#include "core/layers/common/dflash2_grouped_conv.h"

namespace xllm::layer::mlu {

torch::Tensor dflash2_grouped_conv(const torch::Tensor& hidden_states,
                                   const torch::Tensor& delta,
                                   const torch::Tensor& base,
                                   int32_t block_size,
                                   int32_t num_groups,
                                   int32_t group_size,
                                   int32_t taps) {
  CHECK_EQ(hidden_states.dim(), 2);
  CHECK_EQ(delta.dim(), 3);
  CHECK_EQ(base.dim(), 2);
  CHECK_GT(block_size, 0);
  CHECK_GT(num_groups, 0);
  CHECK_GT(group_size, 0);
  CHECK_GT(taps, 0);
  CHECK_EQ(hidden_states.size(1),
           static_cast<int64_t>(num_groups) * group_size);
  CHECK_EQ(delta.size(0), hidden_states.size(0));
  CHECK_EQ(delta.size(1), taps);
  CHECK_EQ(delta.size(2), num_groups);
  CHECK_EQ(base.size(0), taps);
  CHECK_EQ(base.size(1), hidden_states.size(1));

  const int64_t num_tokens = hidden_states.size(0);
  // Rows must be laid out as contiguous, fixed-width blocks per sequence.
  // The modulo position below is only valid when every sequence begins on a
  // block_size boundary.
  CHECK_EQ(num_tokens % block_size, 0)
      << "DFlash2 convolution rows must contain complete sequence blocks.";
  if (kernel::mlu::supports_dflash2_grouped_conv(
          hidden_states, delta, base, group_size)) {
    return kernel::mlu::dflash2_grouped_conv(
        hidden_states, delta, base, block_size, group_size, taps);
  }
  return ::xllm::layer::dflash2_grouped_conv(
      hidden_states, delta, base, block_size, num_groups, group_size, taps);
}

std::optional<std::tuple<torch::Tensor, torch::Tensor>>
DFlash2GroupedConvImpl::finish_with_norm(const torch::Tensor& hidden_states,
                                         const torch::Tensor& coefficients,
                                         const torch::Tensor& residual,
                                         const torch::Tensor& weight,
                                         double eps) {
  if (taps() == 2 && group_size() == 16) {
    return kernel::mlu::bangc::dflash2_conv_norm(hidden_states,
                                                 coefficients,
                                                 output_kernel(),
                                                 residual,
                                                 weight,
                                                 block_size(),
                                                 eps);
  }
  return std::nullopt;
}

torch::Tensor DFlash2GroupedConvImpl::convolve(
    const torch::Tensor& hidden_states,
    const torch::Tensor& delta,
    const torch::Tensor& base) const {
  return dflash2_grouped_conv(hidden_states,
                              delta,
                              base,
                              block_size(),
                              num_groups(),
                              group_size(),
                              taps());
}

}  // namespace xllm::layer::mlu
