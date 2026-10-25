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

#include <torch/library.h>

namespace {
void merge_out(const torch::Tensor& scores,
               const torch::Tensor& slots,
               const torch::Tensor& mapping,
               int64_t topk,
               torch::Tensor output,
               torch::Tensor lengths) {
  xllm::kernel::mlu::bangc::dcp_merge_topk(
      scores, slots, mapping, topk, output, lengths);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(xllm_mlu, m) {
  m.def(
      "dcp_merge_topk_out(Tensor scores, Tensor slots, Tensor mapping, "
      "int topk, Tensor(a!) output, Tensor(b!) lengths) -> ()");
}
TORCH_LIBRARY_IMPL(xllm_mlu, PrivateUse1, m) {
  m.impl("dcp_merge_topk_out", TORCH_FN(merge_out));
}
