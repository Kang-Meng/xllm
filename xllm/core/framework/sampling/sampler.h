/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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
#include <c10/core/TensorOptions.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "sampling_params.h"

namespace xllm {

class Sampler final {
 public:
  Sampler() = default;

  // operator() allows us to use the module as a function.
  template <typename... Args>
  auto operator()(Args&&... args) const {
    return this->forward(::std::forward<Args>(args)...);
  }

  // logits: [batch_size, vocab_size], updated with the processed sampling rows.
  // The Tensor may be rebound or promoted in dtype; consumers must read it
  // after forward() returns; cached aliases may still refer to old storage.
  // Returned probabilities retain the input dtype.
  // Equal selected/sample index counts mean all logits rows are sampled in
  // their existing order; only unequal counts select rows by sample_idxes.
  SampleOutput forward(
      torch::Tensor& logits,
      const SamplingParameters& params,
      const torch::Tensor& filter_mask = torch::Tensor()) const;

  // Applies the logit processors from the sampling parameters
  // (frequency/presence penalties, repetition penalties, then the token
  // bitmask or the dense filter mask) and returns the row subset to sample
  // from. Masks or sample indices may produce a new Tensor; callers must use
  // the returned scores.
  // forward() builds on this; callers that only need the processed logits
  // (e.g. greedy speculative verification, which samples separately) can
  // call it directly and skip the sampling passes.
  static torch::Tensor apply_logits_processors(
      torch::Tensor& logits,
      const SamplingParameters& params,
      const torch::Tensor& filter_mask = torch::Tensor());

  // Publishes processed rows using the same selection decision as the caller.
  static void publish_logits(torch::Tensor& logits,
                             const torch::Tensor& sample_logits,
                             const torch::Tensor& sample_idxes,
                             bool use_sample_indices);

  // helper functions
  // probs: [..., vocab_size]
  static torch::Tensor greedy_sample(const torch::Tensor& probs);

  // probs: [..., vocab_size]
  static torch::Tensor random_sample(const torch::Tensor& probs);
};

}  // namespace xllm
