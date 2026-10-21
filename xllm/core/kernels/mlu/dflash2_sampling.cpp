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

#include "kernels/mlu/dflash2_sampling.h"

#include <cstdint>

#include "kernels/mlu/bangc/include/dflash2_sample_path.h"

namespace xllm::kernel::mlu {

bool supports_dflash2_sampling(const torch::Tensor& candidate_ids,
                               const torch::Tensor& edge_log_probs,
                               const torch::Tensor& gumbel_noise,
                               const torch::Tensor& do_sample,
                               bool all_random_sample,
                               bool all_greedy_sample) {
  if (all_greedy_sample || !candidate_ids.defined() ||
      !edge_log_probs.defined() || !gumbel_noise.defined() ||
      candidate_ids.dim() != 3 || !candidate_ids.device().is_privateuseone() ||
      candidate_ids.device() != edge_log_probs.device() ||
      candidate_ids.device() != gumbel_noise.device() ||
      (!all_random_sample && (!do_sample.defined() ||
                              do_sample.device() != candidate_ids.device()))) {
    return false;
  }
  const int64_t top_k = candidate_ids.size(2);
  const int64_t num_steps = candidate_ids.size(1);
  if (top_k <= 0 || top_k > 4096 || num_steps <= 0) {
    return false;
  }
  return true;
}

std::pair<torch::Tensor, torch::Tensor> sample_dflash2_paths(
    const torch::Tensor& candidate_ids,
    const torch::Tensor& edge_log_probs,
    const torch::Tensor& gumbel_noise,
    const torch::Tensor& do_sample,
    bool all_random_sample) {
  return bangc::sample_dflash2_paths(candidate_ids,
                                     edge_log_probs,
                                     gumbel_noise,
                                     do_sample,
                                     all_random_sample);
}
}  // namespace xllm::kernel::mlu
