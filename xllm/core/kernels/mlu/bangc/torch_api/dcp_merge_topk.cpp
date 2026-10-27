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

#include "kernels/mlu/bangc/include/dcp_merge_topk.h"

#include <cnrt.h>
#include <framework/core/MLUStream.h>
#include <glog/logging.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "kernels/mlu/bangc/src/dcp_merge_topk/launch.h"

namespace xllm::kernel::mlu::bangc {
namespace {
void check_tensor(const torch::Tensor& tensor,
                  torch::ScalarType dtype,
                  int64_t dims,
                  const torch::Device& device) {
  CHECK(tensor.defined());
  CHECK_EQ(tensor.dim(), dims);
  CHECK_EQ(tensor.scalar_type(), dtype);
  CHECK(tensor.device() == device);
  CHECK(!tensor.is_conj() && !tensor.is_neg());
  for (int64_t axis = 0; axis < dims; ++axis) {
    CHECK(tensor.size(axis) <= 1 || tensor.stride(axis) > 0)
        << "Expanded and negative strides are unsupported";
  }
}

constexpr int64_t kTopk = 2048;
constexpr int64_t kGroupShards = 4;

bool is_supported_shards(int64_t shards) {
  return shards >= 2 && (shards & (shards - 1)) == 0;
}

void launch_native(const torch::Tensor& scores,
                   const torch::Tensor& slots,
                   const torch::Tensor& mapping,
                   torch::Tensor& output,
                   torch::Tensor& lengths) {
  launch_dcp_merge_topk(scores.size(0),
                        scores.const_data_ptr(),
                        slots.const_data_ptr<int32_t>(),
                        mapping.const_data_ptr<int32_t>(),
                        output.mutable_data_ptr<int32_t>(),
                        lengths.mutable_data_ptr<int32_t>(),
                        scores.size(1),
                        scores.stride(0),
                        scores.stride(1),
                        scores.stride(2),
                        slots.stride(0),
                        slots.stride(1),
                        slots.stride(2),
                        mapping.stride(0),
                        torch_mlu::getCurMLUStream());
  CHECK_EQ(cnrtSuccess, cnrtGetLastError());
}

// Reduces four shards to their exact TopK (score, slot) candidates. Slots are
// replaced by unique ranks in ascending physical slot order, so the native
// tie order is preserved and every selected rank maps back to one candidate.
std::pair<torch::Tensor, torch::Tensor> reduce_group(
    const torch::Tensor& scores,
    const torch::Tensor& slots) {
  const int64_t rows = scores.size(1);
  const int64_t width = kGroupShards * kTopk;
  torch::Tensor flat_scores = scores.permute({1, 0, 2}).reshape({rows, width});
  torch::Tensor flat_slots = slots.permute({1, 0, 2}).reshape({rows, width});
  torch::Tensor order = std::get<1>(flat_slots.sort(/*dim=*/1));
  torch::Tensor ranks = torch::empty_like(order).scatter_(
      /*dim=*/1,
      order,
      torch::arange(width, order.options()).expand({rows, width}));
  torch::Tensor surrogate_slots =
      ranks.to(torch::kInt32)
          .masked_fill_(flat_slots < 0, /*value=*/-1)
          .view({rows, kGroupShards, kTopk})
          .permute({1, 0, 2});
  torch::Tensor all_rows =
      torch::zeros({rows}, slots.options().dtype(torch::kInt32));
  torch::Tensor selected = torch::empty({rows, kTopk}, all_rows.options());
  torch::Tensor selected_lengths = torch::empty({rows}, all_rows.options());
  launch_native(scores, surrogate_slots, all_rows, selected, selected_lengths);
  torch::Tensor valid = selected >= 0;
  torch::Tensor positions = order.gather(
      /*dim=*/1, selected.clamp_min(/*min=*/0).to(torch::kInt64));
  return {flat_scores.gather(/*dim=*/1, positions),
          flat_slots.gather(/*dim=*/1, positions)
              .masked_fill_(~valid, /*value=*/-1)};
}

// Groups of four shards keep the exact union TopK; recursion ends at the
// native two- or four-shard merge.
void merge_shards(const torch::Tensor& scores,
                  const torch::Tensor& slots,
                  const torch::Tensor& mapping,
                  torch::Tensor& output,
                  torch::Tensor& lengths) {
  const int64_t shards = scores.size(0);
  if (shards <= kGroupShards) {
    launch_native(scores, slots, mapping, output, lengths);
    return;
  }
  const int64_t groups = shards / kGroupShards;
  std::vector<torch::Tensor> group_scores;
  std::vector<torch::Tensor> group_slots;
  group_scores.reserve(groups);
  group_slots.reserve(groups);
  for (int64_t group = 0; group < groups; ++group) {
    const int64_t start = group * kGroupShards;
    auto [reduced_scores, reduced_slots] =
        reduce_group(scores.slice(/*dim=*/0, start, start + kGroupShards),
                     slots.slice(/*dim=*/0, start, start + kGroupShards));
    group_scores.emplace_back(std::move(reduced_scores));
    group_slots.emplace_back(std::move(reduced_slots));
  }
  merge_shards(torch::stack(group_scores),
               torch::stack(group_slots),
               mapping,
               output,
               lengths);
}
}  // namespace

void dcp_merge_topk(const torch::Tensor& scores,
                    const torch::Tensor& slots,
                    const torch::Tensor& mapping,
                    int64_t topk,
                    torch::Tensor& output,
                    torch::Tensor& lengths) {
  CHECK(scores.defined());
  CHECK(scores.device().is_privateuseone());
  const torch::Device device = scores.device();
  check_tensor(scores, torch::kBFloat16, 3, device);
  check_tensor(slots, torch::kInt32, 3, device);
  check_tensor(mapping, torch::kInt32, 1, device);
  CHECK(scores.sizes() == slots.sizes());
  CHECK(is_supported_shards(scores.size(0)))
      << "DCP merge supports power-of-two shard counts of at least 2, got "
      << scores.size(0);
  CHECK_EQ(scores.size(2), kTopk);
  CHECK_EQ(topk, kTopk);
  const int64_t rows = scores.size(1);
  CHECK_EQ(mapping.size(0), rows);
  check_tensor(output, torch::kInt32, 2, device);
  check_tensor(lengths, torch::kInt32, 1, device);
  CHECK(output.sizes() == torch::IntArrayRef({rows, topk}));
  CHECK_EQ(lengths.size(0), rows);
  CHECK(output.is_contiguous() && lengths.is_contiguous())
      << "Merge outputs must be contiguous";
  CHECK(!output.is_alias_of(lengths));
  for (const torch::Tensor& input : {scores, slots, mapping}) {
    CHECK(!output.is_alias_of(input));
    CHECK(!lengths.is_alias_of(input));
  }
  if (rows == 0) {
    return;
  }
  const torch::DeviceGuard guard(device);
  merge_shards(scores, slots, mapping, output, lengths);
}
}  // namespace xllm::kernel::mlu::bangc
