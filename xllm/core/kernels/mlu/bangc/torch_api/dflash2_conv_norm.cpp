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

#include <ATen/DeviceGuard.h>
#include <ATen/MemoryOverlap.h>
#include <cnrt.h>
#include <glog/logging.h>

#include <cmath>
#include <limits>
#include <utility>

#include "framework/core/MLUStream.h"
#include "kernels/mlu/bangc/src/dflash2_conv_norm/launch.h"

namespace xllm::kernel::mlu::bangc {
namespace {
bool supported_tensor(const torch::Tensor& tensor,
                      const torch::Device& device) {
  return tensor.defined() && tensor.device() == device &&
         tensor.scalar_type() == torch::kBFloat16 && tensor.is_contiguous() &&
         !tensor.is_conj() && !tensor.is_neg();
}
bool may_overlap(const torch::Tensor& left, const torch::Tensor& right) {
  // A strided projection is not dense: overlap analysis may return TooHard
  // even for distinct allocations. Different storage cannot overlap.
  return left.is_alias_of(right) &&
         torch::get_overlap_status(left, right) != torch::MemOverlapStatus::No;
}
}  // namespace
std::optional<std::tuple<torch::Tensor, torch::Tensor>> dflash2_conv_norm(
    const torch::Tensor& hidden,
    const torch::Tensor& delta,
    const torch::Tensor& base,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    int32_t block_size,
    double eps) {
  if (!hidden.defined() || !hidden.device().is_privateuseone() ||
      hidden.dim() != 2 || hidden.size(1) != 4096 || hidden.size(0) <= 0 ||
      hidden.size(0) > std::numeric_limits<int32_t>::max() ||
      (block_size != 4 && block_size != 6) ||
      hidden.size(0) % block_size != 0 || !std::isfinite(eps) ||
      static_cast<float>(eps) != 1e-5f) {
    return std::nullopt;
  }
  const torch::Device device = hidden.device();
  if (!supported_tensor(hidden, device) || !supported_tensor(base, device) ||
      !supported_tensor(residual, device) ||
      !supported_tensor(weight, device) || !base.sizes().equals({2, 4096}) ||
      residual.sizes() != hidden.sizes() || !weight.sizes().equals({4096}) ||
      !delta.defined() || delta.device() != device ||
      delta.scalar_type() != torch::kBFloat16 ||
      !delta.sizes().equals({hidden.size(0), 2, 256}) ||
      delta.stride(1) != 256 || delta.stride(2) != 1 ||
      (delta.stride(0) != 512 && delta.stride(0) != 1024) || delta.is_conj() ||
      delta.is_neg()) {
    return std::nullopt;
  }
  // Updating residual must not overwrite any convolution input or parameter.
  if (may_overlap(residual, hidden) || may_overlap(residual, delta) ||
      may_overlap(residual, base) || may_overlap(residual, weight)) {
    return std::nullopt;
  }
  const torch::DeviceGuard guard(device);
  torch::Tensor output = torch::empty_like(hidden);
  // Match the existing MLU RMSNorm contract: mutate the original storage.
  torch::Tensor residual_out = residual;
  dflash2_conv_norm_launch(torch_mlu::getCurMLUStream(),
                           hidden.data_ptr(),
                           delta.data_ptr(),
                           base.data_ptr(),
                           residual.data_ptr(),
                           weight.data_ptr(),
                           output.data_ptr(),
                           residual_out.data_ptr(),
                           static_cast<int32_t>(hidden.size(0)),
                           block_size,
                           delta.stride(0),
                           static_cast<float>(eps));
  CHECK_EQ(cnrtSuccess, cnrtGetLastError());
  return std::make_tuple(std::move(output), std::move(residual_out));
}
}  // namespace xllm::kernel::mlu::bangc
