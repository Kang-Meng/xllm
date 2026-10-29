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

#include <torch/torch.h>

#include <cstdint>
#include <tuple>

namespace xllm::kernel::mlu {

// PCP partition and convolution scheduling granularity, default 64;
// XLLM_MLU_KDA_CHUNK_SIZE may select 16. Staged KDA prefill uses fixed
// 32-token blocks independently of this setting.
int64_t kda_prefill_chunk_size(int64_t num_heads, bool use_qk_l2norm);

// Inference-only, variable-length KDA prefill using BangC kernels with
// per-key log gates and V-first recurrent-state layout.
class ChunkKDAImpl final : public torch::nn::Module {
 public:
  static constexpr int64_t kDefaultChunkSize = 64;

  explicit ChunkKDAImpl(int64_t num_heads);
  ~ChunkKDAImpl() override = default;

  // Fuses safe-gate and beta activation in the staged or generic BangC path.
  // Chunk metadata must be sequence-major, with
  // every local chunk present in order, as produced by the KDA layer.
  std::tuple<torch::Tensor, torch::Tensor> forward_raw_gate(
      const torch::Tensor& q,
      const torch::Tensor& k,
      const torch::Tensor& v,
      const torch::Tensor& raw_gate,
      const torch::Tensor& a_log,
      const torch::Tensor& dt_bias,
      float gate_lower_bound,
      const torch::Tensor& raw_beta,
      const torch::Tensor& initial_state,
      const torch::Tensor& cu_seqlens,
      // Compatibility argument; BangC derives blocks from cu_seqlens.
      const torch::Tensor& chunk_indices,
      bool output_final_state,
      bool use_qk_l2norm);

 private:
  int64_t num_heads_ = 0;
};

TORCH_MODULE(ChunkKDA);

}  // namespace xllm::kernel::mlu
