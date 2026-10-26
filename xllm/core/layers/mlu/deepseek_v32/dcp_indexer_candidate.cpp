/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/layers/mlu/deepseek_v32/dcp_indexer_candidate.h"

#include <glog/logging.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "framework/parallel_state/process_group.h"

namespace xllm::layer {

DcpIndexerGatherAsyncCtx launch_dcp_indexer_candidate_gather(
    const torch::Tensor& local_scores,
    const torch::Tensor& local_global_slots,
    ProcessGroup* dcp_group) {
  if (dcp_group == nullptr || dcp_group->world_size() == 1) {
    DcpIndexerGatherAsyncCtx ctx;
    ctx.gathered_scores = local_scores.unsqueeze(0);
    ctx.gathered_global_slots = local_global_slots.unsqueeze(0);
    return ctx;
  }

  const int32_t world_size = dcp_group->world_size();
  CHECK_EQ(local_scores.dim(), 2);
  CHECK_EQ(local_scores.scalar_type(), torch::kBFloat16);
  CHECK_EQ(local_global_slots.scalar_type(), torch::kInt32);
  CHECK(local_scores.sizes() == local_global_slots.sizes());
  CHECK(local_scores.device() == local_global_slots.device());
  std::vector<int64_t> gathered_shape = local_scores.sizes().vec();
  gathered_shape.insert(gathered_shape.begin(),
                        static_cast<int64_t>(world_size));
  DcpIndexerGatherAsyncCtx ctx;
  const torch::Tensor score_bytes =
      local_scores.contiguous().view(torch::kInt8).flatten();
  const torch::Tensor slot_bytes =
      local_global_slots.contiguous().view(torch::kInt8).flatten();
  const int64_t score_size = score_bytes.numel();
  const int64_t slot_offset = (score_size + 3) / 4 * 4;
  if (slot_offset == score_size && score_bytes.is_alias_of(slot_bytes) &&
      slot_bytes.storage_offset() ==
          score_bytes.storage_offset() + score_size) {
    ctx.local_packet = score_bytes.as_strided(
        {score_size + slot_bytes.numel()}, {1}, score_bytes.storage_offset());
  } else {
    std::vector<torch::Tensor> parts;
    parts.reserve(3);
    parts.emplace_back(score_bytes);
    if (slot_offset > score_size) {
      parts.emplace_back(
          torch::zeros({slot_offset - score_size}, score_bytes.options()));
    }
    parts.emplace_back(slot_bytes);
    ctx.local_packet = torch::cat(parts);
  }
  torch::Tensor gathered_packet = torch::empty(
      {world_size, ctx.local_packet.numel()}, ctx.local_packet.options());
  ctx.gathered_scores = gathered_packet.narrow(1, 0, score_size)
                            .view(torch::kBFloat16)
                            .view(gathered_shape);
  ctx.gathered_global_slots =
      gathered_packet.narrow(1, slot_offset, slot_bytes.numel())
          .view(torch::kInt32)
          .view(gathered_shape);
  // All-gather copies opaque words; the field views retain BF16 and int32.
  torch::Tensor wire_input = ctx.local_packet.view(torch::kFloat32);
  torch::Tensor wire_output = gathered_packet.view(torch::kFloat32);
  ctx.gather_work = dcp_group->allgather_base_async(wire_input, wire_output);
  return ctx;
}

DcpIndexerGatheredCandidates finish_dcp_indexer_candidate_gather(
    DcpIndexerGatherAsyncCtx&& ctx) {
  if (ctx.gather_work.defined()) {
    ctx.gather_work->wait();
  }
  return DcpIndexerGatheredCandidates{std::move(ctx.gathered_scores),
                                      std::move(ctx.gathered_global_slots)};
}

}  // namespace xllm::layer
