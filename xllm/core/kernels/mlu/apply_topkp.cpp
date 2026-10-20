/* Copyright 2025-2026 The xLLM Authors.

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

#include <limits>

#include "mlu_ops_api.h"
namespace xllm::kernel::mlu {
namespace {

// apply_topkp_v2 requires positive top-k values below this exclusive limit.
constexpr int64_t kFusedTopKLimit = 10000;

torch::Tensor filter_logits(const torch::Tensor& logits,
                            const torch::Tensor& temperatures,
                            const torch::Tensor& top_k,
                            const torch::Tensor& top_p) {
  torch::Tensor scores = logits.to(torch::kFloat32);
  if (temperatures.defined()) {
    const torch::Tensor device_temperatures = temperatures.to(scores.options());
    const torch::Tensor divisors =
        torch::where(device_temperatures == 0,
                     torch::ones_like(device_temperatures),
                     device_temperatures);
    scores = scores / divisors.unsqueeze(/*dim=*/1);
  }
  auto [sorted, indices] = scores.sort(/*dim=*/-1, /*descending=*/true);
  const float neg_inf = -std::numeric_limits<float>::infinity();
  const int64_t vocab_size = scores.size(-1);
  const torch::Tensor device_top_k = top_k.to(scores.device());
  const torch::Tensor cutoffs =
      torch::where(device_top_k > 0,
                   device_top_k,
                   torch::full_like(device_top_k, vocab_size));
  const torch::Tensor ranks = torch::arange(vocab_size, device_top_k.options());
  sorted.masked_fill_(ranks >= cutoffs.unsqueeze(/*dim=*/1), neg_inf);
  if (top_p.defined()) {
    const torch::Tensor probs = sorted.softmax(/*dim=*/-1);
    sorted.masked_fill_(probs.cumsum(/*dim=*/-1) - probs >
                            top_p.to(scores.options()).unsqueeze(/*dim=*/1),
                        neg_inf);
  }
  return torch::empty_like(scores).scatter_(/*dim=*/-1, indices, sorted);
}

}  // namespace

torch::Tensor apply_top_k_top_p(const torch::Tensor& logits,
                                const torch::Tensor& temperature_list,
                                const torch::Tensor& topk_list,
                                const torch::Tensor& topp_list) {
  if (!topk_list.defined() && !topp_list.defined()) {
    return logits;
  }
  if (topk_list.defined() &&
      ((topk_list <= 0) | (topk_list >= kFusedTopKLimit)).any().item<bool>()) {
    return filter_logits(logits, temperature_list, topk_list, topp_list);
  }
  torch::Tensor temperature, topk, topp;
  if (!temperature_list.defined()) {
    temperature =
        torch::ones({logits.size(0)},
                    torch::dtype(torch::kFloat32).device(logits.device()));
  } else {
    temperature = temperature_list.to(logits.device());
  }
  if (topk_list.defined()) {
    topk = topk_list.to(torch::dtype(torch::kInt32).device(logits.device()));
  }
  if (topp_list.defined()) {
    topp = topp_list.to(logits.device());
  }

  const int64_t vocab = logits.size(-1);
  torch::Tensor index_in =
      torch::arange(vocab, torch::dtype(torch::kInt32).device(logits.device()));

  // Initialize output tensors if they are empty
  torch::Tensor logits_out = torch::empty(
      logits.sizes(), torch::dtype(torch::kFloat32).device(logits.device()));

  torch::Tensor sorted_logits_out = torch::empty(
      logits.sizes(), torch::dtype(torch::kFloat32).device(logits.device()));

  torch::Tensor index_out = torch::empty(
      logits.sizes(), torch::dtype(torch::kInt32).device(logits.device()));

  torch::Tensor true_select_len = torch::empty(
      {logits.size(0)}, torch::dtype(torch::kInt32).device(logits.device()));

  // Special case handling
  // Create a variable to hold the logits to use (may be modified in special
  // case)
  torch::Tensor logits_for_kernel = logits;
  if (!topk_list.defined() && topp_list.defined()) {
    auto topk_result = torch::topk(logits, logits.size(-1));
    logits_for_kernel = std::get<0>(topk_result);
    auto topk_indices = std::get<1>(topk_result);
    index_out = topk_indices.to(torch::kInt32);
  }

  tmo::torch_api::apply_topkp_v2(logits_for_kernel.to(torch::kFloat32),
                                 index_in,
                                 temperature,
                                 /*min_topp=*/torch::Tensor(),
                                 topk,
                                 topp,
                                 logits_out,
                                 sorted_logits_out,
                                 index_out,
                                 true_select_len);

  return logits_out;
}

}  // namespace xllm::kernel::mlu
