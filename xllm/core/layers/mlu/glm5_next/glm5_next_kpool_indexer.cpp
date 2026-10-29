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

#include "layers/mlu/glm5_next/glm5_next_kpool_indexer.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "common/constants.h"
#include "core/framework/model/graph_eager_context.h"
#include "framework/config/kv_cache_config.h"
#include "kernels/mlu/bangc/include/kpool_bangc.h"
#include "kernels/mlu/kpool.h"
#include "util/linalg.h"

namespace xllm::layer {
namespace {

constexpr int64_t kWorkspaceBytes = 64 * 1024 * 1024;

int32_t max_sequence_length(const std::vector<int32_t>& lengths) {
  return lengths.empty() ? 0
                         : *std::max_element(lengths.begin(), lengths.end());
}

std::vector<int64_t> make_query_offsets(const std::vector<int32_t>& lengths) {
  std::vector<int64_t> offsets;
  offsets.reserve(lengths.size() + 1);
  offsets.emplace_back(0);
  for (const int32_t length : lengths) {
    offsets.emplace_back(offsets.back() + length);
  }
  return offsets;
}

std::vector<int32_t> resolve_seq_lens(const std::vector<int32_t>& host_lens,
                                      const torch::Tensor& device_lens,
                                      const char* name) {
  if (!host_lens.empty()) {
    if (device_lens.defined() && host_lens.front() == 0 &&
        host_lens.size() == static_cast<size_t>(device_lens.numel() + 1)) {
      std::vector<int32_t> per_sequence_lens;
      per_sequence_lens.reserve(host_lens.size() - 1);
      for (size_t i = 1; i < host_lens.size(); ++i) {
        per_sequence_lens.emplace_back(host_lens[i] - host_lens[i - 1]);
      }
      return per_sequence_lens;
    }
    return host_lens;
  }
  CHECK(device_lens.defined()) << name << " must be defined.";
  torch::Tensor cpu_lens =
      device_lens.to(torch::kCPU, torch::kInt32).contiguous();
  const int32_t* data = cpu_lens.data_ptr<int32_t>();
  return std::vector<int32_t>(data, data + cpu_lens.numel());
}

torch::Tensor hadamard_after_bf16_roundtrip(const torch::Tensor& input,
                                            const torch::Tensor& hadamard) {
  torch::Tensor rounded = input.to(torch::kBFloat16).to(torch::kFloat32);
  torch::Tensor matrix =
      hadamard.to(input.device(), torch::kFloat32, /*non_blocking=*/false);
  return util::hadamard_transform(rounded, matrix).to(torch::kBFloat16);
}

torch::Tensor make_row_batch(const std::vector<int32_t>& q_seq_lens,
                             const torch::Device& device) {
  std::vector<int64_t> rows;
  const int64_t row_count = std::accumulate(
      q_seq_lens.begin(), q_seq_lens.end(), static_cast<int64_t>(0));
  rows.reserve(static_cast<size_t>(row_count));
  for (size_t batch_id = 0; batch_id < q_seq_lens.size(); ++batch_id) {
    rows.insert(rows.end(),
                static_cast<size_t>(q_seq_lens[batch_id]),
                static_cast<int64_t>(batch_id));
  }
  return torch::tensor(rows, torch::TensorOptions().dtype(torch::kInt64))
      .to(device);
}

torch::Tensor make_kpool_rows(const torch::Tensor& starts, int64_t tokens) {
  CHECK_EQ(starts.dim(), 1);
  CHECK(starts.scalar_type() == torch::kInt32 ||
        starts.scalar_type() == torch::kInt64);
  const torch::Tensor offsets = starts.to(torch::kInt64).contiguous();
  torch::Tensor rows = torch::empty({tokens}, offsets.options());
  if (tokens == 0) {
    return rows;
  }
  CHECK(kernel::mlu::bangc_kpool_rows(offsets, rows))
      << "Unsupported MLU KPool request offsets";
  return rows;
}

std::shared_ptr<KPoolBatchMetadata> make_kpool_batch_metadata(
    const AttentionMetadata& metadata,
    const torch::Device& device,
    const torch::Tensor& graph_rows) {
  auto batch = std::make_shared<KPoolBatchMetadata>();
  const auto options =
      torch::TensorOptions().dtype(torch::kInt64).device(device);
  const bool graph = metadata.enable_cuda_graph;
  batch->q_seq_lens = metadata.kpool_query_lens;
  if (batch->q_seq_lens.empty()) {
    if (graph && metadata.q_seq_lens_vec.empty()) {
      batch->q_seq_lens.assign(graph_rows.numel(), 1);
    } else {
      batch->q_seq_lens = resolve_seq_lens(
          metadata.q_seq_lens_vec, metadata.q_seq_lens, "KPool q lengths");
    }
  }
  const int64_t requests = static_cast<int64_t>(batch->q_seq_lens.size());
  const std::vector<int64_t> offsets = make_query_offsets(batch->q_seq_lens);
  const int64_t tokens = offsets.back();
  if (graph) {
    // The ordinary graph path has live cumulative query offsets. Explicit
    // speculative grouping is prepared before capture, or has uniform spans.
    if (metadata.kpool_query_lens.empty() && metadata.q_cu_seq_lens.defined() &&
        metadata.q_cu_seq_lens.numel() == requests + 1) {
      batch->query_starts = metadata.q_cu_seq_lens.to(torch::kInt64);
    } else {
      const int32_t width = batch->q_seq_lens.front();
      CHECK(std::all_of(batch->q_seq_lens.begin(),
                        batch->q_seq_lens.end(),
                        [width](int32_t len) { return len == width; }))
          << "Prepare ragged KPool metadata before graph capture.";
      batch->query_starts = torch::arange(requests + 1, options) * width;
    }
    batch->row_batch = make_kpool_rows(batch->query_starts, tokens);
  } else {
    batch->query_starts = torch::tensor(offsets, options);
    batch->row_batch = make_row_batch(batch->q_seq_lens, device);
  }
  batch->block_table = metadata.block_table;
  batch->tail_indices = metadata.linear_state_indices;
  const bool expanded = metadata.block_table.defined() &&
                        metadata.block_table.size(0) != requests;
  if (expanded) {
    CHECK_GE(metadata.block_table.size(0), tokens);
    batch->block_table = metadata.block_table.index_select(
        0, batch->query_starts.narrow(0, 0, requests));
  }
  if (batch->tail_indices.defined()) {
    CHECK_GE(batch->tail_indices.numel(), requests);
    batch->tail_indices = batch->tail_indices.narrow(0, 0, requests);
  }
  if (!graph) {
    batch->kv_seq_lens = resolve_seq_lens(
        metadata.kv_seq_lens_vec, metadata.kv_seq_lens, "KPool KV lengths");
    if (expanded) {
      std::vector<int32_t> logical_lens;
      logical_lens.reserve(requests);
      for (int64_t row = 0; row < requests; ++row) {
        logical_lens.emplace_back(batch->kv_seq_lens.at(offsets[row + 1] - 1));
      }
      batch->kv_seq_lens = std::move(logical_lens);
    }
    batch->max_kv_len = max_sequence_length(batch->kv_seq_lens);
  }
  return batch;
}

void update_prefill(const torch::Tensor& key,
                    const torch::Tensor& gate,
                    const torch::Tensor& ape,
                    const torch::Tensor& hadamard,
                    torch::Tensor& cache,
                    torch::Tensor& tail,
                    const torch::Tensor& tail_ids,
                    const torch::Tensor& block_table,
                    const torch::Tensor& positions,
                    const torch::Tensor& row_batch,
                    const torch::Tensor& starts,
                    int64_t block_size,
                    int64_t pool_size,
                    const torch::Tensor& read_tail_indices) {
  kernel::mlu::update_kpool(key,
                            gate,
                            ape,
                            hadamard,
                            cache,
                            tail,
                            tail_ids,
                            block_table,
                            positions,
                            row_batch,
                            starts,
                            block_size,
                            pool_size,
                            /*decode=*/false,
                            read_tail_indices);
}

}  // namespace

void prepare_glm5_next_kpool_metadata(AttentionMetadata& metadata,
                                      const torch::Device& device) {
  metadata.kpool_batch_metadata =
      metadata.is_dummy
          ? nullptr
          : make_kpool_batch_metadata(metadata, device, metadata.kv_seq_lens);
}

torch::Tensor glm5_next_kpool_select(const torch::Tensor& query,
                                     const torch::Tensor& head_weights,
                                     const torch::Tensor& positions,
                                     const torch::Tensor& row_batch,
                                     const torch::Tensor& index_cache,
                                     const torch::Tensor& block_table,
                                     int64_t max_kv_seq_len,
                                     int64_t block_size,
                                     int64_t index_kpool,
                                     int64_t index_topk,
                                     double softmax_scale,
                                     int64_t workspace_bytes,
                                     const torch::Tensor& query_starts) {
  CHECK_GT(index_kpool, 0);
  CHECK_LE((max_kv_seq_len + block_size - 1) / block_size, block_table.size(1));
  const int64_t count = index_topk / index_kpool;
  const int64_t pools = (max_kv_seq_len + index_kpool - 1) / index_kpool;
  if (pools == 0 || query.size(0) == 0 || count == 0) {
    return torch::full(
        {query.size(0), count}, -1, query.options().dtype(torch::kInt64));
  }
  CHECK_GE(workspace_bytes, pools * static_cast<int64_t>(sizeof(float)));
  const int64_t chunk = std::min<int64_t>(
      query.size(0), workspace_bytes / (pools * sizeof(float)));
  torch::Tensor workspace =
      torch::empty({chunk, pools}, query.options().dtype(torch::kFloat32));
  // Scores are finite at most on each row's live prefix: every shape searches
  // only that prefix.
  const auto select = [&](const torch::Tensor& scores,
                          const torch::Tensor& score_positions) {
    return kernel::mlu::select_kpool_live(
        scores, count, score_positions, index_kpool);
  };
  if (chunk == query.size(0)) {
    // select_kpool fully writes its result; no fill and copy are needed.
    kernel::mlu::score_kpool(query,
                             head_weights,
                             index_cache,
                             block_table,
                             positions,
                             row_batch,
                             workspace,
                             block_size,
                             index_kpool,
                             softmax_scale,
                             query_starts);
    return select(workspace, positions);
  }
  torch::Tensor result = torch::empty({query.size(0), count},
                                      query.options().dtype(torch::kInt64));
  // Request offsets address the complete query tensor. Smaller workspaces use
  // the existing independent-query path instead of slicing request metadata.
  for (int64_t start = 0; start < query.size(0); start += chunk) {
    const int64_t rows = std::min(chunk, query.size(0) - start);
    torch::Tensor scores = workspace.narrow(0, 0, rows);
    kernel::mlu::score_kpool(query.narrow(0, start, rows),
                             head_weights.narrow(0, start, rows),
                             index_cache,
                             block_table,
                             positions.narrow(0, start, rows),
                             row_batch.narrow(0, start, rows),
                             scores,
                             block_size,
                             index_kpool,
                             softmax_scale,
                             torch::Tensor());
    result.narrow(0, start, rows)
        .copy_(select(scores, positions.narrow(0, start, rows)));
  }
  return result;
}

torch::Tensor glm5_next_kpool_normalize_key(torch::Tensor key,
                                            RMSNorm& key_norm) {
  const torch::ScalarType projection_dtype = key.scalar_type();
  key = key.to(torch::kFloat32);
  return std::get<0>(key_norm->forward(key)).to(projection_dtype);
}

Glm5NextKPoolIndexerImpl::Glm5NextKPoolIndexerImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const std::shared_ptr<RotaryEmbeddingBase>& rotary_emb,
    const torch::TensorOptions& options)
    : hidden_size_(args.hidden_size()),
      n_heads_(args.index_n_heads()),
      head_dim_(args.index_head_dim()),
      rope_head_dim_(args.qk_rope_head_dim()),
      index_topk_(args.index_topk()),
      index_kpool_(args.index_kpool()),
      block_size_(KVCacheConfig::get_instance().block_size()),
      softmax_scale_(std::pow(static_cast<double>(head_dim_), -0.5) *
                     std::pow(static_cast<double>(n_heads_), -0.5)),
      always_select_tail_(args.index_kpool_always_select_tail()),
      rotary_emb_(rotary_emb) {
  CHECK(options.device().is_privateuseone()) << "KPool requires MLU.";
  CHECK(options.dtype() == torch::kBFloat16) << "KPool requires BF16.";
  CHECK_EQ(n_heads_, 32) << "MLU KPool requires index_n_heads=32";
  CHECK_EQ(head_dim_, 128) << "MLU KPool requires index_head_dim=128";
  CHECK_EQ(index_kpool_, 4) << "MLU KPool requires index_kpool=4";
  CHECK_EQ(index_topk_, 2048) << "MLU KPool requires index_topk=2048";
  CHECK_EQ(block_size_, 16) << "MLU KPool requires block_size=16";
  CHECK_EQ(parallel_args.kv_split_size_effective(), 1)
      << "KPool does not support DCP cache sharding.";

  CHECK(rope_head_dim_ == 0 || rotary_emb_ != nullptr)
      << "GLM5-Next KPool RoPE requires a rotary embedding.";

  wq_b_ = register_module("wq_b",
                          ReplicatedLinear(args.q_lora_rank(),
                                           n_heads_ * head_dim_,
                                           /*bias=*/false,
                                           quant_args,
                                           options));
  wk_ = register_module("wk",
                        ReplicatedLinear(hidden_size_,
                                         head_dim_,
                                         /*bias=*/false,
                                         QuantArgs(),
                                         options));
  weights_proj_ =
      register_module("weights_proj",
                      ReplicatedLinear(hidden_size_,
                                       n_heads_,
                                       /*bias=*/false,
                                       QuantArgs(),
                                       options.dtype(torch::kFloat32)));
  k_norm_ = register_module(
      "k_norm", RMSNorm(head_dim_, 1e-6, options.dtype(torch::kFloat32)));
  k_norm_->set_layernorm_mode();

  index_kpool_compress_gate_ = register_parameter(
      "index_kpool_compress_gate",
      torch::empty({head_dim_, hidden_size_}, options.dtype(torch::kBFloat16)),
      /*requires_grad=*/false);
  index_kpool_compress_ape_ = register_parameter(
      "index_kpool_compress_ape",
      torch::empty({index_kpool_, head_dim_}, options.dtype(torch::kFloat32)),
      /*requires_grad=*/false);
  hadamard_matrix_ = util::create_hadamard_matrix(
      head_dim_, torch::kFloat32, options.device(), /*normalize=*/true);
}

torch::Tensor Glm5NextKPoolIndexerImpl::project_query(
    const torch::Tensor& q_norm,
    const torch::Tensor& positions,
    const AttentionMetadata& attn_metadata) {
  torch::Tensor q =
      wq_b_->forward(q_norm).view({q_norm.size(0), n_heads_, head_dim_});
  if (rope_head_dim_ > 0) {
    torch::Tensor q_pe = q.slice(/*dim=*/-1, 0, rope_head_dim_);
    rotary_emb_->forward(q_pe,
                         positions,
                         attn_metadata.q_cu_seq_lens,
                         attn_metadata.max_query_len,
                         attn_metadata.is_prefill);
  }
  return hadamard_after_bf16_roundtrip(q, hadamard_matrix_);
}

torch::Tensor Glm5NextKPoolIndexerImpl::project_raw_k(
    const torch::Tensor& hidden_states,
    const torch::Tensor& positions,
    const AttentionMetadata& attn_metadata) {
  torch::Tensor k = wk_->forward(hidden_states);
  k = glm5_next_kpool_normalize_key(std::move(k), k_norm_);
  if (rope_head_dim_ > 0) {
    torch::Tensor k_pe =
        k.slice(/*dim=*/-1, 0, rope_head_dim_).unsqueeze(/*dim=*/1);
    rotary_emb_->forward(k_pe,
                         positions,
                         attn_metadata.q_cu_seq_lens,
                         attn_metadata.max_query_len,
                         attn_metadata.is_prefill);
  }
  return k.to(torch::kBFloat16);
}

Glm5NextKPoolIndexerImpl::Execution Glm5NextKPoolIndexerImpl::prepare_execution(
    const AttentionMetadata& metadata,
    const torch::Tensor& positions) const {
  Execution execution;
  execution.graph_decode = metadata.enable_cuda_graph;
  execution.batch = metadata.kpool_batch_metadata;
  if (!execution.batch) {
    execution.batch =
        make_kpool_batch_metadata(metadata, positions.device(), positions);
  }
  if (execution.graph_decode && metadata.kpool_batch_metadata) {
    auto batch = std::make_shared<KPoolBatchMetadata>(*execution.batch);
    const int64_t tokens = positions.numel();
    batch->row_batch = make_kpool_rows(batch->query_starts, tokens);
    execution.batch = std::move(batch);
  }
  // Draft extend carries prefill metadata; its active positions already mask
  // placeholders. Verify always uses the request-local fused ring update.
  execution.prefill = (metadata.is_prefill || metadata.is_chunked_prefill ||
                       max_sequence_length(execution.batch->q_seq_lens) > 1) &&
                      !metadata.is_spec_verify && !execution.graph_decode;
  // This is an update launch choice within the semantic stage. Tiny prefills
  // avoid the gather/softmax launch chain and share paged selection.
  execution.fused_update =
      !execution.prefill ||
      max_sequence_length(execution.batch->q_seq_lens) <= 32;
  execution.score_capacity =
      execution.graph_decode
          ? (metadata.max_seq_len > 0
                 ? metadata.max_seq_len
                 : execution.batch->block_table.size(1) * block_size_)
          : execution.batch->max_kv_len;
  // Reject unsupported work before update_cache mutates persistent state.
  constexpr int64_t kMaxInt32 = std::numeric_limits<int32_t>::max();
  CHECK_LE(positions.numel(), kMaxInt32 / output_width());
  CHECK_EQ(positions.dim(), 1);
  CHECK(positions.scalar_type() == torch::kInt32 ||
        positions.scalar_type() == torch::kInt64);
  const torch::Tensor& table = execution.batch->block_table;
  CHECK_EQ(table.dim(), 2);
  CHECK_GT(table.size(0), 0);
  CHECK_GT(table.size(1), 0);
  CHECK_LE(table.size(1), kMaxInt32 / block_size_);
  CHECK_LE(table.numel(), kMaxInt32);
  CHECK(table.scalar_type() == torch::kInt32 ||
        table.scalar_type() == torch::kInt64);
  CHECK_EQ(table.device(), positions.device());
  CHECK_GE(execution.score_capacity, 0);
  CHECK_LE(execution.score_capacity, table.size(1) * block_size_);
  const int64_t pools =
      (execution.score_capacity + index_kpool_ - 1) / index_kpool_;
  CHECK_LE(pools, kWorkspaceBytes / static_cast<int64_t>(sizeof(float)));
  return execution;
}

void Glm5NextKPoolIndexerImpl::update_cache(
    const torch::Tensor& raw_k,
    const torch::Tensor& gate_bf16,
    const torch::Tensor& positions,
    torch::Tensor& index_cache,
    torch::Tensor& tail_cache,
    const torch::Tensor& linear_state_indices,
    const torch::Tensor& block_table,
    const Execution& execution,
    const torch::Tensor& read_tail_indices) {
  const auto& batch = *execution.batch;
  if (execution.prefill && !execution.fused_update) {
    update_prefill(raw_k,
                   gate_bf16,
                   index_kpool_compress_ape_,
                   hadamard_matrix_,
                   index_cache,
                   tail_cache,
                   linear_state_indices,
                   block_table,
                   positions,
                   batch.row_batch,
                   batch.query_starts,
                   block_size_,
                   index_kpool_,
                   read_tail_indices);
    return;
  }
  const bool decode = execution.fused_update;
  kernel::mlu::update_kpool(raw_k,
                            gate_bf16,
                            index_kpool_compress_ape_,
                            hadamard_matrix_,
                            index_cache,
                            tail_cache,
                            linear_state_indices,
                            block_table,
                            positions,
                            batch.row_batch,
                            batch.query_starts,
                            block_size_,
                            index_kpool_,
                            decode,
                            read_tail_indices);
}

torch::Tensor Glm5NextKPoolIndexerImpl::select_pools(
    const torch::Tensor& hidden_states,
    const torch::Tensor& q_norm,
    const torch::Tensor& positions,
    const torch::Tensor& index_cache,
    const AttentionMetadata& metadata,
    const Execution& execution) {
  const torch::Tensor query =
      project_query(q_norm, torch::clamp_min(positions, 0), metadata);
  const torch::Tensor weights =
      weights_proj_->forward(hidden_states.to(torch::kFloat32));
  return select_projected_pools(
      query, weights, positions, index_cache, metadata, execution);
}

torch::Tensor Glm5NextKPoolIndexerImpl::select_projected_pools(
    const torch::Tensor& query,
    const torch::Tensor& weights,
    const torch::Tensor& positions,
    const torch::Tensor& index_cache,
    const AttentionMetadata& metadata,
    const Execution& execution) {
  if (execution.prefill && !execution.fused_update) {
    // Score directly from the paged pool cache.  The previous prefill path
    // gathered every request's history into dense tensors and then reran an
    // FP32 matmul for each query tile.  The paged score kernel already applies
    // the request-local causal limit from positions and row_batch, so it is
    // equivalent while avoiding the dense gather and host-side grouping loops.
    return glm5_next_kpool_select(query,
                                  weights,
                                  positions,
                                  execution.batch->row_batch,
                                  index_cache,
                                  execution.batch->block_table,
                                  execution.batch->max_kv_len,
                                  block_size_,
                                  index_kpool_,
                                  index_topk_,
                                  softmax_scale_);
  }
  // Every verify query sees only pools complete at its own logical position.
  const torch::Tensor& score_positions = positions;
  const bool four_query_verify =
      metadata.is_spec_verify && metadata.max_query_len == 4 &&
      !execution.batch->q_seq_lens.empty() &&
      std::all_of(execution.batch->q_seq_lens.begin(),
                  execution.batch->q_seq_lens.end(),
                  [](int32_t length) { return length == 4; });
  auto select = [this,
                 query,
                 weights,
                 score_positions,
                 index_cache,
                 batch = execution.batch,
                 four_query_verify](int64_t kv_max_seq_len) {
    return glm5_next_kpool_select(
        query,
        weights,
        score_positions,
        batch->row_batch,
        index_cache,
        batch->block_table,
        kv_max_seq_len,
        block_size_,
        index_kpool_,
        index_topk_,
        softmax_scale_,
        /*workspace_bytes=*/kWorkspaceBytes,
        four_query_verify ? batch->query_starts : torch::Tensor());
  };
  if (metadata.eager_context != nullptr) {
    // The score workspace grows with live history; only the fixed-width pool
    // IDs cross the graph boundary. The callback owns the projected inputs and
    // selected tensor by value, keeping their storage alive across shared-pool
    // captures and replays. Copy into selected without replacing its storage so
    // the following graph segment keeps reading the same address.
    torch::Tensor selected =
        torch::empty({query.size(0), index_topk_ / index_kpool_},
                     query.options().dtype(torch::kInt64));
    metadata.eager_context->run_eager(
        [select = std::move(select), selected](int64_t kv_max_seq_len) mutable {
          selected.copy_(select(kv_max_seq_len));
        });
    return selected;
  }
  return select(execution.score_capacity);
}

std::tuple<torch::Tensor, torch::Tensor> Glm5NextKPoolIndexerImpl::forward(
    const torch::Tensor& hidden_states,
    const torch::Tensor& q_norm,
    const torch::Tensor& positions,
    torch::Tensor& index_cache,
    torch::Tensor& tail_cache,
    const AttentionMetadata& attn_metadata) {
  CHECK(tail_cache.defined()) << "KPool requires framework-owned tail storage.";
  if (hidden_states.size(0) == 0) {
    const auto options = hidden_states.options().dtype(torch::kInt32);
    return {torch::empty({0, output_width()}, options),
            torch::empty({0}, options)};
  }
  const Execution execution = prepare_execution(attn_metadata, positions);
  const bool use_prefill_expand = execution.prefill && !execution.fused_update;
  torch::Tensor raw_k = project_raw_k(hidden_states, positions, attn_metadata);
  torch::Tensor gate_score =
      torch::nn::functional::linear(hidden_states, index_kpool_compress_gate_);
  const torch::Tensor gate_bf16 = gate_score.to(torch::kBFloat16);
  CHECK_EQ(execution.batch->row_batch.numel(), positions.numel());
  CHECK(execution.batch->tail_indices.defined());
  if (attn_metadata.is_spec_verify) {
    CHECK_LE(max_sequence_length(execution.batch->q_seq_lens),
             tail_cache.size(2) - index_kpool_)
        << "KPool tail capacity does not cover the speculative write window.";
  }
  const torch::Tensor cache_positions =
      kernel::mlu::kpool_cache_positions(execution.batch->tail_indices,
                                         execution.batch->row_batch,
                                         positions,
                                         attn_metadata.slot_mapping,
                                         kPaddingLinearStateId);
  update_cache(raw_k,
               gate_bf16,
               cache_positions,
               index_cache,
               tail_cache,
               execution.batch->tail_indices,
               execution.batch->block_table,
               execution,
               attn_metadata.linear_state_read_indices);
  const torch::Tensor pools = select_pools(hidden_states,
                                           q_norm,
                                           cache_positions,
                                           index_cache,
                                           attn_metadata,
                                           execution);
  kernel::mlu::KPoolSelection selection =
      kernel::mlu::expand_kpool(pools,
                                cache_positions,
                                execution.batch->row_batch,
                                execution.batch->block_table,
                                block_size_,
                                index_topk_,
                                index_kpool_,
                                always_select_tail_,
                                use_prefill_expand);
  return {std::move(selection.physical_slots),
          std::move(selection.context_lens)};
}

void Glm5NextKPoolIndexerImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  wq_b_->load_state_dict(state_dict.get_dict_with_prefix("wq_b."));
  wk_->load_state_dict(state_dict.get_dict_with_prefix("wk."));
  weights_proj_->load_state_dict(
      state_dict.get_dict_with_prefix("weights_proj."));
  k_norm_->load_state_dict(state_dict.get_dict_with_prefix("k_norm."));
  LOAD_WEIGHT(index_kpool_compress_gate);
  LOAD_WEIGHT(index_kpool_compress_ape);
}

}  // namespace xllm::layer
