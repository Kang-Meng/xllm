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

#include <framework/core/MLUStream.h>
#include <glog/logging.h>

#include <cstdint>
#include <initializer_list>
#include <limits>

#include "kernels/mlu/bangc/include/dcp_merge_topk.h"
#include "kernels/mlu/mlu_ops_api.h"
#include "triton_jit/include/jit_kernel.h"

namespace xllm::kernel::mlu {
namespace {

using triton_jit::JITKernel;
using triton_jit::LaunchCfg;

void check_tensor(const torch::Tensor& tensor,
                  torch::ScalarType dtype,
                  int64_t dims,
                  const torch::Device& device) {
  CHECK(tensor.defined());
  CHECK_EQ(tensor.dim(), dims);
  CHECK_EQ(tensor.scalar_type(), dtype);
  CHECK(tensor.device() == device);
  for (int64_t axis = 0; axis < dims; ++axis) {
    CHECK(tensor.size(axis) <= 1 || tensor.stride(axis) > 0)
        << "Expanded and negative strides are unsupported";
  }
}

void check_outputs(const torch::Tensor& output,
                   const torch::Tensor& lengths,
                   int64_t rows,
                   int64_t width,
                   const torch::Device& device,
                   std::initializer_list<torch::Tensor> inputs) {
  check_tensor(output, torch::kInt32, 2, device);
  check_tensor(lengths, torch::kInt32, 1, device);
  CHECK(output.sizes() == torch::IntArrayRef({rows, width}));
  CHECK_EQ(lengths.size(0), rows);
  CHECK_LE(rows, std::numeric_limits<uint32_t>::max());
  if (rows == 0) {
    return;
  }
  // Accept dense, transposed and sliced tables; reject overlapping writes.
  const int64_t small = output.stride(0) <= output.stride(1) ? 0 : 1;
  const int64_t large = 1 - small;
  CHECK(output.size(large) <= 1 ||
        output.stride(large) > (output.size(small) - 1) * output.stride(small));
  CHECK(!output.is_alias_of(lengths));
  for (const torch::Tensor& input : inputs) {
    CHECK(!output.is_alias_of(input));
    CHECK(!lengths.is_alias_of(input));
  }
}

}  // namespace

void dcp_merge_topk(const torch::Tensor& scores,
                    const torch::Tensor& slots,
                    const torch::Tensor& mapping,
                    int64_t topk,
                    torch::Tensor& output,
                    torch::Tensor& lengths) {
  bangc::dcp_merge_topk(scores, slots, mapping, topk, output, lengths);
}

void dcp_localize_topk(const torch::Tensor& slots,
                       const torch::Tensor& lengths,
                       int32_t rank,
                       int32_t size,
                       int32_t interleave,
                       torch::Tensor& output,
                       torch::Tensor& output_lengths) {
  CHECK(slots.defined());
  CHECK(slots.device().is_privateuseone());
  const torch::Device device = slots.device();
  check_tensor(slots, torch::kInt32, 2, device);
  check_tensor(lengths, torch::kInt32, 1, device);
  const int64_t q = slots.size(0);
  const int64_t width = slots.size(1);
  CHECK_GT(width, 0);
  CHECK_LE(width, std::numeric_limits<int32_t>::max());
  CHECK_EQ(lengths.size(0), q);
  CHECK(lengths.is_contiguous());
  CHECK_GT(size, 0);
  CHECK_GE(rank, 0);
  CHECK_LT(rank, size);
  CHECK_GT(interleave, 0);
  check_outputs(output, output_lengths, q, width, device, {slots, lengths});
  CHECK(output_lengths.is_contiguous());
  if (q == 0) {
    return;
  }

  int64_t block = q <= 128 ? 1024 : 512;
  int32_t stages = q < 1024 ? 3 : 4;
  if (width != 2048 || interleave != 16 || (size != 4 && size != 8)) {
    stages = 3;
  }
  if (slots.stride(1) != 1 || output.stride(1) != 1) {
    block = 256;
    stages = 3;
  }
  LaunchCfg cfg;
  cfg.num_warps = 1;
  cfg.num_stages = stages;
  cfg.force_use_shared_memory = true;
  const torch::DeviceGuard guard(device);
  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  JITKernel::get("xllm.core.kernels.mlu.triton_kernel.dcp.dcp_localize_topk",
                 "tmo_dcp_localize_topk_kernel")
      .launch(static_cast<void*>(queue),
              {static_cast<uint32_t>(q), 1, 1},
              cfg,
              slots,
              lengths,
              output,
              output_lengths,
              rank,
              size,
              interleave,
              static_cast<int32_t>(width),
              slots.stride(0),
              slots.stride(1),
              output.stride(0),
              output.stride(1),
              block);
}

}  // namespace xllm::kernel::mlu
