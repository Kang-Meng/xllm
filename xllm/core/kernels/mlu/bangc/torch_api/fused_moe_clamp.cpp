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

#include "kernels/mlu/bangc/include/fused_moe_clamp.h"

#include <ATen/DeviceGuard.h>
#include <cnrt.h>
#include <glog/logging.h>

#include <cmath>
#include <limits>

#include "framework/core/MLUStream.h"
#include "kernels/mlu/bangc/src/fused_moe_clamp/launcher.h"
namespace xllm::kernel::mlu::bangc {
void fused_moe_clamp(torch::Tensor& input,
                     double limit,
                     const std::optional<torch::Tensor>& token_count) {
  CHECK(input.defined());
  CHECK(input.device().is_privateuseone());
  CHECK(input.is_contiguous());
  CHECK_GE(input.dim(), 2);
  const torch::ScalarType dtype = input.scalar_type();
  CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16 ||
        dtype == torch::kFloat32);
  CHECK(std::isfinite(limit));
  CHECK_GE(limit, 0.0);
  const int64_t width = input.size(-1);
  CHECK_GT(width, 0);
  CHECK_EQ(width % 2, 0);
  CHECK_LE(width * input.element_size(), kFusedMoeClampMaxRowBytes);
  const int64_t rows = input.numel() / width;
  CHECK_LE(rows, std::numeric_limits<int32_t>::max());
  const void* counts = nullptr;
  int32_t count_size = 0;
  int32_t count_bytes = 4;
  if (token_count.has_value()) {
    // Counts bound rows of the flattened 2-D group GEMM output.
    CHECK_EQ(input.dim(), 2);
    CHECK(token_count->device() == input.device());
    CHECK(token_count->is_contiguous());
    CHECK_EQ(token_count->dim(), 1);
    CHECK(token_count->scalar_type() == torch::kInt32 ||
          token_count->scalar_type() == torch::kInt64);
    CHECK_GT(token_count->numel(), 0);
    CHECK_LE(token_count->numel(), kFusedMoeClampMaxCounts);
    counts = token_count->data_ptr();
    count_size = static_cast<int32_t>(token_count->numel());
    count_bytes = static_cast<int32_t>(token_count->element_size());
  }
  if (rows == 0) {
    return;
  }
  // Compare against the limit rounded to the element type, as the reference
  // clamp does.
  float rounded = static_cast<float>(limit);
  if (dtype == torch::kBFloat16) {
    rounded = static_cast<float>(c10::BFloat16(rounded));
  } else if (dtype == torch::kFloat16) {
    rounded = static_cast<float>(c10::Half(rounded));
  }
  const int32_t code = dtype == torch::kFloat16    ? 0
                       : dtype == torch::kBFloat16 ? 1
                                                   : 2;
  const torch::DeviceGuard guard(input.device());
  const cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  fused_moe_clamp_launch(static_cast<void*>(queue),
                         input.data_ptr(),
                         rows,
                         static_cast<int32_t>(width),
                         code,
                         rounded,
                         counts,
                         count_size,
                         count_bytes);
  CHECK_EQ(cnrtSuccess, cnrtGetLastError());
}
}  // namespace xllm::kernel::mlu::bangc
