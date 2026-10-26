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

#include "kernels/mlu/bangc/include/dcp_indexer.h"

#include <framework/core/MLUStream.h>
#include <glog/logging.h>

#include <limits>

#include "kernels/mlu/bangc/src/dcp_indexer/host.h"
#include "kernels/mlu/bangc/src/dcp_indexer/params.h"

namespace xllm::kernel::mlu::bangc {
namespace {

void check_tensor(const torch::Tensor& tensor,
                  torch::ScalarType dtype,
                  int64_t dims,
                  const torch::Device& device) {
  CHECK(tensor.defined());
  CHECK_EQ(tensor.scalar_type(), dtype);
  CHECK_EQ(tensor.dim(), dims);
  CHECK(tensor.device() == device);
  CHECK(tensor.is_contiguous());
  CHECK(!tensor.is_conj() && !tensor.is_neg());
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> dcp_indexer(
    const torch::Tensor& query,
    const torch::Tensor& k_cache,
    const torch::Tensor& weights,
    const torch::Tensor& block_table,
    const torch::Tensor& q_cu_seq_lens,
    const torch::Tensor& query_positions,
    const torch::Tensor& kv_seq_lens,
    int64_t dcp_rank,
    int64_t dcp_size,
    bool is_prefill) {
  CHECK(query.defined());
  CHECK(query.device().is_privateuseone());
  const torch::Device device = query.device();
  check_tensor(query, torch::kBFloat16, 3, device);
  check_tensor(k_cache, torch::kBFloat16, 4, device);
  check_tensor(weights, torch::kBFloat16, 2, device);
  check_tensor(block_table, torch::kInt32, 2, device);
  check_tensor(q_cu_seq_lens, torch::kInt32, 1, device);
  check_tensor(query_positions, torch::kInt32, 1, device);
  check_tensor(kv_seq_lens, torch::kInt32, 1, device);
  CHECK_EQ(query.size(1), detail::kHeads);
  CHECK_EQ(query.size(2), detail::kDim);
  const int64_t rows = query.size(0);
  const int64_t batch = kv_seq_lens.size(0);
  CHECK(weights.sizes() == torch::IntArrayRef({rows, detail::kHeads}));
  CHECK_EQ(k_cache.size(1), 1);
  CHECK_EQ(k_cache.size(2), detail::kPage);
  CHECK_EQ(k_cache.size(3), detail::kDim);
  CHECK_EQ(block_table.size(0), batch);
  CHECK_EQ(q_cu_seq_lens.size(0), batch + 1);
  CHECK_EQ(query_positions.size(0), rows);
  CHECK_GE(dcp_size, 1);
  CHECK_LE(dcp_size, std::numeric_limits<int32_t>::max() / detail::kPage);
  CHECK_GE(dcp_rank, 0);
  CHECK_LT(dcp_rank, dcp_size);
  const int64_t slot_limit = int64_t{1} << 31;
  CHECK_LE(k_cache.size(0), slot_limit / (detail::kPage * dcp_size));
  CHECK_LE(rows, std::numeric_limits<int32_t>::max());
  CHECK_LE(batch, std::numeric_limits<int32_t>::max());
  CHECK_LE(block_table.size(1),
           std::numeric_limits<int32_t>::max() / detail::kPage);
  const torch::DeviceGuard guard(device);
  const int64_t elements = rows * detail::kTopk;
  const int64_t score_bytes = elements * static_cast<int64_t>(sizeof(uint16_t));
  torch::Tensor output = torch::empty(
      {score_bytes + elements * static_cast<int64_t>(sizeof(int32_t))},
      query.options().dtype(torch::kUInt8));
  torch::Tensor scores = output.narrow(0, 0, score_bytes)
                             .view(torch::kBFloat16)
                             .view({rows, detail::kTopk});
  torch::Tensor slots =
      output
          .narrow(
              0, score_bytes, elements * static_cast<int64_t>(sizeof(int32_t)))
          .view(torch::kInt32)
          .view({rows, detail::kTopk});
  if (rows == 0) {
    return {scores, slots};
  }
  CHECK_GT(batch, 0);
  const size_t bytes = detail::workspace_bytes(
      static_cast<int32_t>(rows), static_cast<int32_t>(block_table.size(1)));
  torch::Tensor workspace = torch::empty({static_cast<int64_t>(bytes)},
                                         query.options().dtype(torch::kUInt8));
  detail::IndexerArgs args{};
  args.query = query.const_data_ptr();
  args.k_cache = k_cache.const_data_ptr();
  args.weights = weights.const_data_ptr();
  args.block_table = block_table.const_data_ptr();
  args.q_cu_seq_lens = q_cu_seq_lens.const_data_ptr();
  args.query_positions = query_positions.const_data_ptr();
  args.kv_seq_lens = kv_seq_lens.const_data_ptr();
  args.scores = scores.mutable_data_ptr();
  args.slots = slots.mutable_data_ptr();
  args.workspace = bytes == 0 ? nullptr : workspace.mutable_data_ptr();
  args.num_tokens = static_cast<int32_t>(rows);
  args.batch = static_cast<int32_t>(batch);
  args.table_width = static_cast<int32_t>(block_table.size(1));
  args.num_pages = static_cast<int32_t>(k_cache.size(0));
  args.dcp_rank = static_cast<int32_t>(dcp_rank);
  args.dcp_size = static_cast<int32_t>(dcp_size);
  const cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  const cnrtRet_t result = is_prefill ? detail::launch_prefill(queue, args)
                                      : detail::launch_decode(queue, args);
  CHECK_EQ(result, cnrtSuccess);
  return {scores, slots};
}

}  // namespace xllm::kernel::mlu::bangc
