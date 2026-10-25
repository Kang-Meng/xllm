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

#include "kernels/mlu/bangc/include/dcp_merge_topk.h"

#include <cnrt.h>
#include <framework/core/MLUStream.h>
#include <glog/logging.h>

#include <cstdint>

#include "kernels/mlu/bangc/src/dcp_merge_topk/launch.h"

namespace xllm::kernel::mlu::bangc {
namespace {
void check_tensor(const torch::Tensor& tensor,
                  torch::ScalarType dtype,
                  int64_t dims,
                  const torch::Device& device) {
  CHECK(tensor.defined());
  CHECK_EQ(tensor.dim(), dims);
  CHECK_EQ(tensor.scalar_type(), dtype);
  CHECK(tensor.device() == device);
  CHECK(!tensor.is_conj() && !tensor.is_neg());
  for (int64_t axis = 0; axis < dims; ++axis) {
    CHECK(tensor.size(axis) <= 1 || tensor.stride(axis) > 0)
        << "Expanded and negative strides are unsupported";
  }
}
}  // namespace

void dcp_merge_topk(const torch::Tensor& scores,
                    const torch::Tensor& slots,
                    const torch::Tensor& mapping,
                    int64_t topk,
                    torch::Tensor& output,
                    torch::Tensor& lengths) {
  CHECK(scores.defined());
  CHECK(scores.device().is_privateuseone());
  const torch::Device device = scores.device();
  check_tensor(scores, torch::kBFloat16, 3, device);
  check_tensor(slots, torch::kInt32, 3, device);
  check_tensor(mapping, torch::kInt32, 1, device);
  CHECK(scores.sizes() == slots.sizes());
  CHECK_EQ(scores.size(0), 4);
  CHECK_EQ(scores.size(2), 2048);
  CHECK_EQ(topk, 2048);
  const int64_t rows = scores.size(1);
  CHECK_EQ(mapping.size(0), rows);
  check_tensor(output, torch::kInt32, 2, device);
  check_tensor(lengths, torch::kInt32, 1, device);
  CHECK(output.sizes() == torch::IntArrayRef({rows, topk}));
  CHECK_EQ(lengths.size(0), rows);
  CHECK(output.is_contiguous() && lengths.is_contiguous())
      << "Merge outputs must be contiguous";
  CHECK(!output.is_alias_of(lengths));
  for (const torch::Tensor& input : {scores, slots, mapping}) {
    CHECK(!output.is_alias_of(input));
    CHECK(!lengths.is_alias_of(input));
  }
  if (rows == 0) {
    return;
  }
  const torch::DeviceGuard guard(device);
  launch_dcp_merge_topk(scores.const_data_ptr(),
                        slots.const_data_ptr<int32_t>(),
                        mapping.const_data_ptr<int32_t>(),
                        output.mutable_data_ptr<int32_t>(),
                        lengths.mutable_data_ptr<int32_t>(),
                        rows,
                        scores.stride(0),
                        scores.stride(1),
                        scores.stride(2),
                        slots.stride(0),
                        slots.stride(1),
                        slots.stride(2),
                        mapping.stride(0),
                        torch_mlu::getCurMLUStream());
  CHECK_EQ(cnrtSuccess, cnrtGetLastError());
}
}  // namespace xllm::kernel::mlu::bangc
