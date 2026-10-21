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

#include "core/kernels/mlu/dflash2_sampling.h"
#include "core/runtime/dflash/dflash2_worker_impl.h"

namespace xllm {

std::optional<DFlash2WorkerImpl::BlockSampleOutput>
DFlash2WorkerImpl::fused_sample(const torch::Tensor& candidate_ids,
                                const torch::Tensor& edge_log_probs,
                                const torch::Tensor& gumbel_noise,
                                const SamplingParameters& sampling_params,
                                int64_t vocab_size) {
  if (!kernel::mlu::supports_dflash2_sampling(
          candidate_ids,
          edge_log_probs,
          gumbel_noise,
          sampling_params.do_sample,
          sampling_params.all_random_sample,
          sampling_params.all_greedy_sample)) {
    return std::nullopt;
  }
  auto sampled =
      kernel::mlu::sample_dflash2_paths(candidate_ids,
                                        edge_log_probs,
                                        gumbel_noise,
                                        sampling_params.do_sample,
                                        sampling_params.all_random_sample);
  torch::Tensor dense_probs =
      torch::zeros({candidate_ids.size(0), candidate_ids.size(1), vocab_size},
                   edge_log_probs.options().dtype(torch::kFloat32));
  dense_probs.scatter_(/*dim=*/-1, candidate_ids, sampled.second);
  return BlockSampleOutput{.token_ids = std::move(sampled.first),
                           .dense_probs = std::move(dense_probs)};
}

}  // namespace xllm
