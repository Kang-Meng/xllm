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

#include <ATen/DeviceGuard.h>
#include <cnrt.h>
#include <framework/core/MLUStream.h>
#include <glog/logging.h>

#include <cstdint>
#include <utility>

#include "kernels/mlu/bangc/src/dflash2/sample_path_launchers.h"
namespace xllm::kernel::mlu::bangc {
std::pair<torch::Tensor, torch::Tensor> sample_dflash2_paths(
    const torch::Tensor& candidate_ids,
    const torch::Tensor& edge_log_probs,
    const torch::Tensor& gumbel_noise,
    const torch::Tensor& do_sample,
    bool all_random_sample) {
  CHECK(candidate_ids.defined() && edge_log_probs.defined() &&
        gumbel_noise.defined());
  CHECK_EQ(candidate_ids.dim(), 3);
  CHECK_EQ(edge_log_probs.dim(), 4);
  CHECK_EQ(gumbel_noise.dim(), 3);
  CHECK_EQ(candidate_ids.device(), edge_log_probs.device());
  CHECK_EQ(candidate_ids.device(), gumbel_noise.device());
  CHECK_EQ(candidate_ids.device().type(), torch::kPrivateUse1);
  CHECK(candidate_ids.scalar_type() == torch::kLong ||
        candidate_ids.scalar_type() == torch::kInt32);
  CHECK_EQ(edge_log_probs.scalar_type(), torch::kFloat32);
  CHECK_EQ(gumbel_noise.scalar_type(), torch::kFloat32);
  const int64_t batch = candidate_ids.size(0);
  const int64_t steps = candidate_ids.size(1);
  const int64_t top_k = candidate_ids.size(2);
  CHECK_GT(steps, 0);
  CHECK_GT(top_k, 0);
  CHECK_LE(top_k, 4096);
  CHECK_EQ(edge_log_probs.sizes(),
           torch::IntArrayRef({batch, steps, top_k, top_k}));
  CHECK_EQ(gumbel_noise.sizes(), torch::IntArrayRef({batch, steps, top_k}));
  int64_t mask_stride = 0;
  if (!all_random_sample) {
    CHECK(do_sample.defined());
    CHECK_EQ(do_sample.device(), candidate_ids.device());
    CHECK_EQ(do_sample.scalar_type(), torch::kBool);
    CHECK_EQ(do_sample.sizes(), torch::IntArrayRef({batch}));
    mask_stride = do_sample.stride(0);
  }
  const torch::DeviceGuard guard(candidate_ids.device());
  torch::Tensor tokens = torch::empty({batch, steps}, candidate_ids.options());
  torch::Tensor proposal_log_probs =
      torch::empty({batch, steps, top_k}, edge_log_probs.options());
  if (batch == 0) {
    return {std::move(tokens), std::move(proposal_log_probs)};
  }
  const DFlash2SamplePathShape shape{batch,
                                     steps,
                                     top_k,
                                     candidate_ids.stride(0),
                                     candidate_ids.stride(1),
                                     candidate_ids.stride(2),
                                     edge_log_probs.stride(0),
                                     edge_log_probs.stride(1),
                                     edge_log_probs.stride(2),
                                     edge_log_probs.stride(3),
                                     gumbel_noise.stride(0),
                                     gumbel_noise.stride(1),
                                     gumbel_noise.stride(2),
                                     mask_stride};
  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  launch_dflash2_sample_path(
      static_cast<void*>(queue),
      candidate_ids.const_data_ptr(),
      edge_log_probs.const_data_ptr<float>(),
      gumbel_noise.const_data_ptr<float>(),
      all_random_sample ? nullptr : do_sample.const_data_ptr<bool>(),
      tokens.mutable_data_ptr(),
      proposal_log_probs.mutable_data_ptr<float>(),
      &shape,
      candidate_ids.scalar_type() == torch::kLong,
      all_random_sample);
  CHECK_EQ(cnrtSuccess, cnrtGetLastError());
  // Use the same device exp implementation as the original sampler. Mixed
  // greedy rows have log q=0/-inf, so exp produces exact one-hot probabilities.
  return {std::move(tokens), proposal_log_probs.exp()};
}

}  // namespace xllm::kernel::mlu::bangc
