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

namespace xllm::kernel::mlu::bangc {

// BF16 query[N,32,128], weights[N,32], cache[P,1,16,128]; contiguous tensors.
// Request-level INT32 metadata uses global KV lengths and absolute positions.
// Returns scaled BF16 scores and INT32 global physical slots, both [N,2048].
// Scale is 1/64. QK and scaled weights round to BF16 before FP32 head
// reduction.
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
    bool is_prefill);

}  // namespace xllm::kernel::mlu::bangc
