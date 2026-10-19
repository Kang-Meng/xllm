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

#include "kernels/mlu/bangc/include/ple.h"

#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(xllm_mlu, m) {
  m.def(
      "ple_pack_projection_out(Tensor weight, Tensor(a!) packed_weight) -> ()");
  m.def(
      "ple_prepare_head_out(Tensor key_scale, Tensor query_scale, "
      "Tensor conv_scale, Tensor(a!) affine_pair) -> ()");
  m.def(
      "ple_prepare_weights_out(Tensor weight, Tensor key_scale, "
      "Tensor query_scale, Tensor conv_scale, Tensor(a!) packed_weight, "
      "Tensor(b!) affine_pair) -> ()");
  m.def(
      "ple_projection_out(Tensor embedding, Tensor packed_weight, "
      "Tensor(a!) kv) -> ()");
  m.def(
      "ple_head_out(Tensor kv, Tensor hidden, Tensor affine_pair, "
      "Tensor(a!) values, Tensor(b!) normalized) -> ()");
  m.def(
      "ple_tail_out(Tensor values, Tensor normalized, Tensor(a!) history, "
      "Tensor taps, Tensor hidden, Tensor(b!) output) -> ()");
  m.def(
      "ple_forward_out(Tensor embedding, Tensor hidden, Tensor packed_weight, "
      "Tensor affine_pair, Tensor taps, Tensor(a!) history, Tensor(b!) kv, "
      "Tensor(c!) values, Tensor(d!) normalized, Tensor(e!) output) -> ()");
}

TORCH_LIBRARY_IMPL(xllm_mlu, PrivateUse1, m) {
  m.impl("ple_pack_projection_out",
         TORCH_FN(xllm::kernel::mlu::bangc::ple_pack_projection_out));
  m.impl("ple_prepare_head_out",
         TORCH_FN(xllm::kernel::mlu::bangc::ple_prepare_head_out));
  m.impl("ple_prepare_weights_out",
         TORCH_FN(xllm::kernel::mlu::bangc::ple_prepare_weights_out));
  m.impl("ple_projection_out",
         TORCH_FN(xllm::kernel::mlu::bangc::ple_projection_out));
  m.impl("ple_head_out", TORCH_FN(xllm::kernel::mlu::bangc::ple_head_out));
  m.impl("ple_tail_out", TORCH_FN(xllm::kernel::mlu::bangc::ple_tail_out));
  m.impl("ple_forward_out",
         TORCH_FN(xllm::kernel::mlu::bangc::ple_forward_out));
}
