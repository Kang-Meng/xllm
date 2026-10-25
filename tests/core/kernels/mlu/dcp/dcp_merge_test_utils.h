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

#include <bit>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace xllm::dcp_test {
constexpr int32_t kDcp = 4;
constexpr int32_t kTopk = 2048;
constexpr int32_t kEdgeRows = 16;
using TopkResult = std::pair<torch::Tensor, torch::Tensor>;
struct Candidates {
  torch::Tensor scores;
  torch::Tensor slots;
  torch::Tensor mapping;
};

// Equal keys deliberately use the same device TopK ordering as dcp_merge_topk.
// Adding a slot-based tie break would change the bitwise regression contract.
inline TopkResult merge_reference(const torch::Tensor& scores,
                                  const torch::Tensor& slots,
                                  const torch::Tensor& mapping,
                                  int64_t topk) {
  CHECK_EQ(scores.scalar_type(), torch::kBFloat16);
  CHECK_EQ(slots.scalar_type(), torch::kInt32);
  CHECK_EQ(mapping.scalar_type(), torch::kInt32);
  CHECK_EQ(scores.dim(), 3);
  CHECK(scores.sizes() == slots.sizes());
  const int64_t rows = scores.size(1);
  CHECK(mapping.sizes() == torch::IntArrayRef({rows}));
  CHECK_GT(topk, 0);
  CHECK_LE(topk, scores.size(0) * scores.size(2));
  auto output = torch::full({rows, topk}, -1, slots.options());
  auto lengths = torch::zeros({rows}, slots.options());
  if (rows == 0) {
    return {output, lengths};
  }

  auto flat_scores = scores.to(torch::kFloat32)
                         .permute({1, 0, 2})
                         .contiguous()
                         .view({rows, -1});
  auto flat_slots = slots.permute({1, 0, 2}).reshape({rows, -1});
  auto bits = flat_scores.view(torch::kInt32);
  auto valid = (flat_slots >= 0) &
               (bits.bitwise_and(/*other=*/0x7fffffff) <= 0x7f800000);
  // Bitwise negation preserves signed zero and subnormals. Positive canonical
  // NaNs sort after valid keys, including the key of a valid -infinity score.
  auto key_bits = bits.bitwise_xor(std::numeric_limits<int32_t>::min());
  key_bits.masked_fill_(~valid, /*value=*/0x7fc00000);
  auto prepared_slots = flat_slots.masked_fill(~valid, /*value=*/-1);
  auto order = std::get<1>(torch::topk(key_bits.view(torch::kFloat32),
                                       topk,
                                       /*dim=*/1,
                                       /*largest=*/false,
                                       /*sorted=*/true));
  output = prepared_slots.gather(/*dim=*/1, order);
  auto selected = (output >= 0) & (mapping.unsqueeze(/*dim=*/1) >= 0);
  output.masked_fill_(~selected, /*value=*/-1);
  lengths = selected.sum(/*dim=*/1).to(torch::kInt32);
  return {output, lengths};
}

inline torch::Tensor bf16_pattern(const std::vector<uint16_t>& bits) {
  std::vector<int16_t> signed_bits;
  signed_bits.reserve(bits.size());
  for (uint16_t value : bits) {
    signed_bits.emplace_back(std::bit_cast<int16_t>(value));
  }
  return torch::tensor(signed_bits, torch::kInt16).view(torch::kBFloat16);
}

inline Candidates make_candidates(int32_t rows,
                                  int32_t shard,
                                  int32_t tp_rank,
                                  int32_t seed,
                                  bool edges) {
  torch::manual_seed(seed + shard * 101 + tp_rank * 1009);
  auto scores =
      torch::randn({rows, kTopk}, torch::kFloat32).to(torch::kBFloat16);
  auto cols = torch::arange(kTopk, torch::kInt32);
  auto ordinal = cols + shard * kTopk;
  auto ids =
      (cols.div(/*other=*/16, /*rounding_mode=*/"floor") * kDcp + shard) * 16 +
      cols.remainder(16) + tp_rank * (1 << 24);
  auto slots = ids.unsqueeze(/*dim=*/0).repeat({rows, 1});
  auto mapping = torch::arange(rows, torch::kInt32);
  if (!edges) {
    auto counts = torch::arange(rows, torch::kInt32).unsqueeze(/*dim=*/1);
    slots.masked_fill_(ordinal.unsqueeze(/*dim=*/0) >= counts, /*value=*/-1);
    mapping.slice(/*dim=*/0, /*start=*/0, rows, /*step=*/127).fill_(-1);
    return {scores, slots, mapping};
  }

  CHECK_EQ(rows, kEdgeRows);
  const std::vector<int32_t> counts{0, 1, 2047, 2048, 2049};
  for (int32_t row = 0; row < static_cast<int32_t>(counts.size()); ++row) {
    slots[row].masked_fill_(ordinal >= counts[row], /*value=*/-1);
  }
  scores[5].fill_(1);
  scores[6].copy_(torch::where(ordinal < kTopk - 1, 2, 1));
  const auto special = bf16_pattern({0x0000,
                                     0x8000,
                                     0x0001,
                                     0x8001,
                                     0x0002,
                                     0x8002,
                                     0x3f80,
                                     0xbf80,
                                     0x7f7f,
                                     0xff7f,
                                     0x7f80,
                                     0xff80,
                                     0x7fc1,
                                     0xffc2});
  scores[7].copy_(special.index_select(
      /*dim=*/0, ordinal.to(torch::kInt64).remainder(special.numel())));
  scores[8].fill_(std::numeric_limits<float>::infinity());
  scores[9].fill_(-std::numeric_limits<float>::infinity());
  scores[10].copy_(bf16_pattern({0x7fc1}).expand({kTopk}));
  scores[11].copy_(bf16_pattern({0xffc2}).expand({kTopk}));
  auto alternating = ordinal.to(torch::kInt64).remainder(2);
  scores[12].copy_(bf16_pattern({0x0000, 0x8000}).index_select(0, alternating));
  scores[13].copy_(bf16_pattern({0x0001, 0x8001}).index_select(0, alternating));
  mapping[14].fill_(-23);
  slots[15].masked_fill_(ordinal >= 3, /*value=*/-7);
  scores[15].copy_(torch::where(ordinal < 3,
                                -std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::infinity()));
  return {scores, slots, mapping};
}

inline torch::Tensor strided_copy(const torch::Tensor& input) {
  auto shape = input.sizes().vec();
  shape.back() = shape.back() * 2 + 2;
  auto storage = torch::full(shape, 17, input.options());
  auto view = storage.slice(-1, 1, shape.back() - 1, 2);
  view.copy_(input);
  return view;
}
}  // namespace xllm::dcp_test
