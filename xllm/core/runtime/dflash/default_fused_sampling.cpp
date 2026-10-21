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

#include "core/runtime/dflash/dflash2_worker_impl.h"

namespace xllm {

std::optional<DFlash2WorkerImpl::BlockSampleOutput>
DFlash2WorkerImpl::fused_sample(const torch::Tensor& /*candidate_ids*/,
                                const torch::Tensor& /*edge_log_probs*/,
                                const torch::Tensor& /*gumbel_noise*/,
                                const SamplingParameters& /*sampling_params*/,
                                int64_t /*vocab_size*/) {
  return std::nullopt;
}

}  // namespace xllm
