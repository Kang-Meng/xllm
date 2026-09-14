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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/linear.h"
#include "layers/common/rms_norm.h"
#include "layers/common/rotary_embedding.h"

namespace xllm::layer {

inline constexpr int64_t kGlm5NextGraphMaxKvSeqLen = 32768;

// Rebuilt at the model-forward boundary; never retained across batches.
void prepare_glm5_next_kpool_metadata(AttentionMetadata& metadata,
                                      const torch::Device& device);

struct Glm5NextKPoolSelection final {
  torch::Tensor physical_slots;
  torch::Tensor context_lens;
};

struct Glm5NextKPoolHistory final {
  torch::Tensor keys;
  torch::Tensor valid;
};

// Apply FP32 LayerNorm and restore the projection dtype.
torch::Tensor glm5_next_kpool_normalize_key(torch::Tensor key,
                                            RMSNorm& key_norm);

// Score compressed keys per query head. GLM5-Next applies ReLU before the
// learned head reduction, so the head weights cannot be folded into Q.
torch::Tensor glm5_next_kpool_score_queries(const torch::Tensor& query,
                                            const torch::Tensor& head_weights,
                                            const torch::Tensor& pooled_key,
                                            double softmax_scale);

// Feature-wise softmax pooling, rounded through BF16 before Hadamard.
torch::Tensor glm5_next_kpool_compress_keys(const torch::Tensor& raw_k,
                                            const torch::Tensor& gate_score,
                                            const torch::Tensor& ape,
                                            const torch::Tensor& hadamard,
                                            int64_t index_kpool);

// Complete pools before overwriting the raw K/gate tail shared across steps,
// then stash the latest tail. Callers pass raw layouts; row-major inputs are
// enforced here. index_cache: [blocks, 1, block_size / index_kpool, D] BF16.
// tail_cache: [state_slots, 2, tail_len, D] BF16.
void launch_kpool_update(const torch::Tensor& k,
                         const torch::Tensor& gate,
                         const torch::Tensor& ape,
                         const torch::Tensor& hadamard,
                         torch::Tensor& index_cache,
                         torch::Tensor& tail_cache,
                         const torch::Tensor& tail_block_ids,
                         const torch::Tensor& block_table,
                         const torch::Tensor& positions,
                         const torch::Tensor& row_batch,
                         const torch::Tensor& starts,
                         int64_t block_size,
                         int64_t index_kpool,
                         bool single_token = false);

// Reads paged compressed pools up to max_kv_seq_len. Runtime KV lengths only
// affect the validity mask, so a fixed maximum keeps graph shapes stable.
Glm5NextKPoolHistory glm5_next_kpool_read_compressed_cache(
    const torch::Tensor& compressed_pool_cache,
    const torch::Tensor& block_table,
    const torch::Tensor& kv_seq_lens,
    int64_t max_kv_seq_len,
    int64_t block_size,
    int64_t index_kpool);

// Select causal top-k pools with Triton scoring and exact selection. Long
// prefill gathers one request at a time; query chunks reuse bounded scores.
// Output order is unspecified. Equal scores prefer lower logical pool ids.
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
                                     int64_t workspace_bytes = 64 * 1024 * 1024,
                                     const KPoolBatchMetadata* batch = nullptr);

// Expand logical pool ids back to their token slots, append the incomplete
// causal tail, compact valid entries to the left, then map through block_table.
// MLU-only, with strided int32/int64 inputs and int32 outputs. Scratch remains
// bounded across query counts and output widths; selected pools must fit top-k.
Glm5NextKPoolSelection glm5_next_kpool_expand_to_physical_slots(
    const torch::Tensor& selected_pool_ids,
    const torch::Tensor& query_positions,
    const torch::Tensor& row_batch,
    const torch::Tensor& block_table,
    int64_t block_size,
    int64_t index_topk,
    int64_t index_kpool,
    bool always_select_tail);

class Glm5NextKPoolIndexerImpl final : public torch::nn::Module {
 public:
  Glm5NextKPoolIndexerImpl(
      const ModelArgs& args,
      const QuantArgs& quant_args,
      const ParallelArgs& parallel_args,
      const std::shared_ptr<RotaryEmbeddingBase>& rotary_emb,
      const torch::TensorOptions& options);

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& q_norm,
      const torch::Tensor& positions,
      torch::Tensor& index_cache,
      torch::Tensor& tail_cache,
      const AttentionMetadata& attn_metadata);

  void load_state_dict(const StateDict& state_dict);

  int64_t output_width() const { return index_topk_ + index_kpool_ - 1; }

 private:
  struct Execution;

  Execution prepare_execution(const AttentionMetadata& metadata,
                              const torch::Tensor& positions) const;
  void update_cache(const torch::Tensor& raw_k,
                    const torch::Tensor& gate_bf16,
                    const torch::Tensor& positions,
                    torch::Tensor& index_cache,
                    torch::Tensor& tail_cache,
                    const torch::Tensor& linear_state_indices,
                    const torch::Tensor& block_table,
                    const Execution& execution);
  torch::Tensor select_pools(const torch::Tensor& hidden_states,
                             const torch::Tensor& q_norm,
                             const torch::Tensor& positions,
                             const torch::Tensor& index_cache,
                             const AttentionMetadata& metadata,
                             const Execution& execution);
  torch::Tensor project_query(const torch::Tensor& q_norm,
                              const torch::Tensor& positions,
                              const AttentionMetadata& attn_metadata);
  torch::Tensor project_raw_k(const torch::Tensor& hidden_states,
                              const torch::Tensor& positions,
                              const AttentionMetadata& attn_metadata);

  int64_t hidden_size_ = 0;
  int64_t n_heads_ = 0;
  int64_t head_dim_ = 0;
  int64_t rope_head_dim_ = 0;
  int64_t index_topk_ = 0;
  int64_t index_kpool_ = 0;
  int64_t block_size_ = 0;
  int64_t graph_max_kv_seq_len_ = kGlm5NextGraphMaxKvSeqLen;
  double softmax_scale_ = 1.0;
  bool always_select_tail_ = true;

  ReplicatedLinear wq_b_{nullptr};
  ReplicatedLinear wk_{nullptr};
  ReplicatedLinear weights_proj_{nullptr};
  RMSNorm k_norm_{nullptr};
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;
  DEFINE_WEIGHT(index_kpool_compress_gate);
  DEFINE_WEIGHT(index_kpool_compress_ape);
  torch::Tensor hadamard_matrix_;
};

TORCH_MODULE(Glm5NextKPoolIndexer);

}  // namespace xllm::layer
