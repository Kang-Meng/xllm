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

#include "core/layers/mlu/deepseek_v32/dcp_batch_metadata.h"

#include <glog/logging.h>

#include <tuple>
#include <utility>

#include "layers/common/attention_metadata.h"

namespace xllm::layer {

std::pair<torch::Tensor, torch::Tensor> build_prefill_slot_order(
    const torch::Tensor& source_slots) {
  CHECK_EQ(source_slots.dim(), 1);
  CHECK(source_slots.scalar_type() == torch::kInt32 ||
        source_slots.scalar_type() == torch::kInt64);
  auto [slots, rows] = torch::sort(source_slots);
  const int64_t token_count = slots.numel();
  if (token_count > 1) {
    torch::Tensor repeated =
        (slots.slice(0, 1) == slots.slice(0, 0, -1)) & (slots.slice(0, 1) >= 0);
    repeated = torch::cat({torch::zeros({1}, repeated.options()), repeated});
    torch::Tensor indices = torch::arange(token_count, rows.options());
    // index_select checks bounds on MLU without copying a predicate to CPU.
    // A duplicate source has no unique K row and must fail, not pick a winner.
    rows = rows.index_select(0, torch::where(repeated, token_count, indices));
  }
  return {std::move(slots), std::move(rows)};
}

KVShardIndexerMetadata build_kv_shard_indexer_metadata(
    const AttentionMetadata& attention_metadata) {
  const torch::Tensor& cu = attention_metadata.q_cu_seq_lens;
  const torch::Tensor& table = attention_metadata.block_table;
  CHECK(cu.defined() && table.defined());
  CHECK(attention_metadata.slot_mapping.defined());
  CHECK_EQ(cu.dim(), 1);
  CHECK_EQ(cu.scalar_type(), torch::kInt32);
  CHECK_EQ(table.dim(), 2);
  const int64_t batch = table.size(0);
  CHECK_EQ(cu.numel(), batch + 1);
  torch::Tensor lens = attention_metadata.kv_seq_lens;
  if (!lens.defined()) {
    CHECK(attention_metadata.kv_cu_seq_lens.defined());
    lens = torch::diff(attention_metadata.kv_cu_seq_lens);
  }
  CHECK_EQ(lens.dim(), 1);
  CHECK_EQ(lens.numel(), batch);
  lens = lens.to(torch::kInt32).contiguous();
  const int64_t rows = attention_metadata.slot_mapping.numel();
  torch::Tensor positions = torch::full({rows}, -1, cu.options());
  if (rows > 0 && batch > 0) {
    torch::Tensor row = torch::arange(rows, cu.options());
    torch::Tensor ends = cu.slice(/*dim=*/0, /*start=*/1).contiguous();
    torch::Tensor requests =
        torch::searchsorted(ends, row, /*out_int32=*/false, /*right=*/true);
    torch::Tensor safe_requests = requests.clamp_max(batch - 1);
    torch::Tensor starts = cu.index_select(0, safe_requests);
    torch::Tensor query_lens = torch::diff(cu).index_select(0, safe_requests);
    positions = lens.index_select(0, safe_requests) - query_lens + row - starts;
    positions.masked_fill_(
        (attention_metadata.slot_mapping < 0) | (requests >= batch), -1);
  }
  return KVShardIndexerMetadata{table.contiguous(),
                                std::move(lens),
                                cu.contiguous(),
                                std::move(positions)};
}

std::shared_ptr<const KVShardBatchMetadata> build_mlu_shard_metadata(
    const AttentionMetadata& attention_metadata,
    const KVShardLayout& layout) {
  auto metadata = std::make_shared<KVShardBatchMetadata>(
      *build_kv_shard_batch_metadata(attention_metadata, layout));
  if (attention_metadata.is_dummy) {
    return metadata;
  }
  if (attention_metadata.is_prefill && !attention_metadata.is_dummy) {
    std::tie(metadata->prefill_sorted_slots, metadata->prefill_sorted_rows) =
        build_prefill_slot_order(attention_metadata.slot_mapping);
    metadata->prefill_sorted_slots =
        metadata->prefill_sorted_slots.to(torch::kInt64);
  }
  if (attention_metadata.q_cu_seq_lens.defined() &&
      attention_metadata.block_table.defined() &&
      (attention_metadata.kv_seq_lens.defined() ||
       attention_metadata.kv_cu_seq_lens.defined())) {
    metadata->indexer = build_kv_shard_indexer_metadata(attention_metadata);
  }
  return metadata;
}

}  // namespace xllm::layer
