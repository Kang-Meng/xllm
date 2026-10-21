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

#include "kernels/mlu/bangc/include/dflash2_grouped_conv.h"

#include <ATen/DeviceGuard.h>
#include <cnrt.h>
#include <glog/logging.h>

#include <limits>

#include "framework/core/MLUStream.h"
#include "kernels/mlu/bangc/src/dflash2_grouped_conv/launcher.h"
namespace xllm::kernel::mlu::bangc {
torch::Tensor dflash2_grouped_conv(const torch::Tensor& hidden,
                                   const torch::Tensor& delta,
                                   const torch::Tensor& base,
                                   int64_t block_size,
                                   int64_t group_size) {
  CHECK(hidden.defined() && delta.defined() && base.defined());
  CHECK(hidden.device().is_privateuseone());
  CHECK_EQ(hidden.dim(), 2);
  CHECK_EQ(delta.dim(), 3);
  CHECK_EQ(base.dim(), 2);
  CHECK_EQ(delta.device(), hidden.device());
  CHECK_EQ(base.device(), hidden.device());
  CHECK_EQ(delta.scalar_type(), hidden.scalar_type());
  CHECK_EQ(base.scalar_type(), hidden.scalar_type());
  CHECK(hidden.scalar_type() == torch::kFloat16 ||
        hidden.scalar_type() == torch::kBFloat16 ||
        hidden.scalar_type() == torch::kFloat32);
  CHECK(hidden.layout() == torch::kStrided &&
        delta.layout() == torch::kStrided && base.layout() == torch::kStrided);
  CHECK(!hidden.is_conj() && !hidden.is_neg());
  CHECK(!delta.is_conj() && !delta.is_neg());
  CHECK(!base.is_conj() && !base.is_neg());
  CHECK_GT(block_size, 0);
  CHECK_LE(block_size, std::numeric_limits<int32_t>::max());
  CHECK_GT(group_size, 0);
  CHECK_LE(group_size, 2048);
  const int64_t tokens = hidden.size(0);
  const int64_t channels = hidden.size(1);
  const int64_t taps = base.size(0);
  CHECK_GT(channels, 0);
  CHECK_EQ(channels % group_size, 0);
  CHECK_EQ(tokens % block_size, 0);
  CHECK_GT(taps, 0);
  CHECK_LE(taps, std::numeric_limits<int32_t>::max());
  CHECK(taps == 1 || tokens >= taps);
  CHECK_EQ(base.size(1), channels);
  CHECK_EQ(delta.size(0), tokens);
  CHECK_EQ(delta.size(1), taps);
  CHECK_EQ(delta.size(2), channels / group_size);
  const torch::DeviceGuard guard(hidden.device());
  torch::Tensor output = torch::empty(hidden.sizes(), hidden.options());
  if (tokens == 0) {
    return output;
  }
  const int32_t dtype = hidden.scalar_type() == torch::kFloat16    ? 0
                        : hidden.scalar_type() == torch::kBFloat16 ? 1
                                                                   : 2;
  const cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  dflash2_conv_launch(static_cast<void*>(queue),
                      hidden.data_ptr(),
                      delta.data_ptr(),
                      base.data_ptr(),
                      output.data_ptr(),
                      dtype,
                      tokens,
                      channels,
                      static_cast<int32_t>(block_size),
                      static_cast<int32_t>(group_size),
                      static_cast<int32_t>(taps),
                      hidden.stride(0),
                      hidden.stride(1),
                      delta.stride(0),
                      delta.stride(1),
                      delta.stride(2),
                      base.stride(0),
                      base.stride(1));
  CHECK_EQ(cnrtSuccess, cnrtGetLastError());
  return output;
}
}  // namespace xllm::kernel::mlu::bangc
