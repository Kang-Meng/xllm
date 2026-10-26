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

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>

namespace xllm::dcp_test {

inline torch::Tensor aligned_golden(const torch::Tensor& query,
                                    const torch::Tensor& weights,
                                    const torch::Tensor& cache,
                                    const torch::Tensor& global_slots,
                                    int64_t dcp_size) {
  const torch::Tensor q =
      query.cpu().reshape({-1, 32, 128}).to(torch::kFloat64);
  const torch::Tensor w =
      (weights.cpu().reshape({-1, 32}).to(torch::kFloat32) / 64)
          .to(torch::kBFloat16)
          .to(torch::kFloat64);
  const torch::Tensor keys = cache.cpu().reshape({-1, 128}).to(torch::kFloat64);
  const torch::Tensor slots = global_slots.cpu().to(torch::kInt64);
  torch::Tensor output = torch::full(slots.sizes(), -INFINITY, torch::kFloat64);
  for (int64_t row = 0; row < slots.size(0); ++row) {
    const torch::Tensor valid = slots[row] >= 0;
    const torch::Tensor selected = slots[row].masked_select(valid);
    if (selected.numel() == 0) {
      continue;
    }
    const torch::Tensor local =
        torch::floor_divide(selected, 16 * dcp_size) * 16 +
        selected.remainder(16);
    const torch::Tensor k = keys.index_select(0, local);
    const torch::Tensor qk = torch::matmul(q[row], k.t())
                                 .relu()
                                 .to(torch::kBFloat16)
                                 .to(torch::kFloat64);
    output[row].masked_scatter_(valid, (qk * w[row].unsqueeze(1)).sum(0));
  }
  return output;
}

// The FP64 oracle retains the QK/weight BF16 rounding, before output rounding.
inline bool score_precision(const torch::Tensor& actual,
                            const torch::Tensor& baseline,
                            const torch::Tensor& golden) {
  CHECK_EQ(actual.scalar_type(), torch::kBFloat16);
  CHECK_EQ(baseline.scalar_type(), torch::kBFloat16);
  CHECK_EQ(golden.scalar_type(), torch::kFloat64);
  CHECK(actual.sizes() == baseline.sizes());
  CHECK(actual.sizes() == golden.sizes());
  const torch::Tensor a = actual.cpu().to(torch::kFloat64);
  const torch::Tensor b = baseline.cpu().to(torch::kFloat64);
  const torch::Tensor g = golden.cpu();
  const torch::Tensor finite = torch::isfinite(g);
  const torch::Tensor same_nonfinite =
      (torch::isneginf(g) & torch::isneginf(a) & torch::isneginf(b)) |
      (torch::isposinf(g) & torch::isposinf(a) & torch::isposinf(b)) |
      (torch::isnan(g) & torch::isnan(a) & torch::isnan(b));
  if (!((finite & torch::isfinite(a) & torch::isfinite(b)) | same_nonfinite)
           .all()
           .item<bool>()) {
    return false;
  }
  const torch::Tensor gold = g.masked_select(finite);
  const torch::Tensor ref = b.masked_select(finite);
  const torch::Tensor result = a.masked_select(finite);
  const torch::Tensor safe = torch::where(gold == 0, 1e-9, gold);
  const double denom1 = safe.abs().sum().item<double>();
  const double denom2 = safe.square().sum().item<double>();
  if (gold.numel() == 0) {
    return true;
  }
  const torch::Tensor ref_error = ref - gold;
  const torch::Tensor error = result - gold;
  const double ref_diff1 = ref_error.abs().sum().item<double>() / denom1;
  const double diff1 = error.abs().sum().item<double>() / denom1;
  const double ref_diff2 =
      std::sqrt(ref_error.square().sum().item<double>() / denom2);
  const double diff2 = std::sqrt(error.square().sum().item<double>() / denom2);
  const int64_t greater = (error > 0).sum().item<int64_t>();
  const int64_t less = (error < 0).sum().item<int64_t>();
  const int64_t ref_greater = (ref_error > 0).sum().item<int64_t>();
  const int64_t ref_less = (ref_error < 0).sum().item<int64_t>();
  const bool diff4 = greater + less < 100 || (greater > 0 && less > 0) ||
                     (ref_greater > 0 && ref_less == 0) ||
                     (ref_less > 0 && ref_greater == 0);
  const bool passed = diff1 <= 10 * std::max(ref_diff1, 1e-4) &&
                      diff2 <= 10 * std::max(ref_diff2, 1e-4) && diff4;
  LOG_IF(ERROR, !passed) << "Dynamic BF16 precision: diff1=" << diff1
                         << " baseline_diff1=" << ref_diff1
                         << " diff2=" << diff2
                         << " baseline_diff2=" << ref_diff2
                         << " diff4_pass=" << diff4;
  return passed;
}

}  // namespace xllm::dcp_test
