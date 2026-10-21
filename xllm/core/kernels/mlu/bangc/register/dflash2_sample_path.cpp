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

#include "kernels/mlu/bangc/include/dflash2_sample_path.h"

#include <torch/library.h>

#include <tuple>
#include <utility>
namespace {
std::tuple<torch::Tensor, torch::Tensor> sample_path_dispatch(
    const torch::Tensor& candidate_ids,
    const torch::Tensor& edge_log_probs,
    const torch::Tensor& gumbel_noise,
    const torch::Tensor& do_sample,
    bool all_random_sample) {
  auto result =
      xllm::kernel::mlu::bangc::sample_dflash2_paths(candidate_ids,
                                                     edge_log_probs,
                                                     gumbel_noise,
                                                     do_sample,
                                                     all_random_sample);
  return {std::move(result.first), std::move(result.second)};
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(xllm_mlu, m) {
  m.def(
      "sample_dflash2_paths(Tensor candidate_ids, Tensor edge_log_probs, "
      "Tensor gumbel_noise, Tensor do_sample, bool all_random_sample) -> "
      "(Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(xllm_mlu, PrivateUse1, m) {
  m.impl("sample_dflash2_paths", TORCH_FN(sample_path_dispatch));
}
