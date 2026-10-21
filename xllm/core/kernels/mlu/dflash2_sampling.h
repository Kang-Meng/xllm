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

#pragma once

#include <torch/torch.h>

#include <utility>

namespace xllm::kernel::mlu {

bool supports_dflash2_sampling(const torch::Tensor& candidate_ids,
                               const torch::Tensor& edge_log_probs,
                               const torch::Tensor& gumbel_noise,
                               const torch::Tensor& do_sample,
                               bool all_random_sample,
                               bool all_greedy_sample);

// Sample predecessor paths from already normalized edge log-probabilities.
// Returns [B,S] tokens and the full [B,S,K] sparse proposal probabilities.
// Random rows preserve torch.exp; mixed greedy rows expose a one-hot q.
std::pair<torch::Tensor, torch::Tensor> sample_dflash2_paths(
    const torch::Tensor& candidate_ids,
    const torch::Tensor& edge_log_probs,
    const torch::Tensor& gumbel_noise,
    const torch::Tensor& do_sample,
    bool all_random_sample);

}  // namespace xllm::kernel::mlu
