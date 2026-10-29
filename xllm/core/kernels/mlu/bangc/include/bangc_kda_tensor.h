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

#pragma once
#include <torch/torch.h>

#include <optional>
#include <utility>
namespace xllm::kernel::mlu {
// Typed integration seam for native KDA kernels. All arithmetic is
// performed by BangC. Torch operations below the seam allocate, cast or arrange
// tensors; metadata remains device-live on the current stream during capture.
struct BangcKdaTensorParams {
  torch::Tensor q;
  torch::Tensor k;
  torch::Tensor v;
  torch::Tensor gate;
  torch::Tensor beta;
  torch::Tensor a_log;
  torch::Tensor bias;
  torch::Tensor initial_state;
  torch::Tensor cu;
  torch::Tensor indices;
  torch::Tensor accepted;
  int32_t gate_mode;  // 0 raw safe sigmoid, 1 raw softplus.
  bool prefill;
  bool inplace_final_state;
  bool output_final_state;
  bool qk_l2norm;
  float scale;
  float lower_bound;
  float softplus_beta;
  float softplus_threshold;
};
std::pair<torch::Tensor, torch::Tensor> bangc_kda_forward(
    const BangcKdaTensorParams& parameters);
}  // namespace xllm::kernel::mlu
