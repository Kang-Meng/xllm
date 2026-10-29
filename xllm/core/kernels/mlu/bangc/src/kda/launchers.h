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

#include <cstdint>

// ---- bangc_kda_copy_written ----

namespace xllm::kernel::mlu {
// Non-owning dense rows. Copy complete original-dtype rows only when marked.
extern "C" int32_t bangc_kda_copy_written(void* queue,
                                          void* destination,
                                          const void* source,
                                          const int32_t* written_rows,
                                          int64_t rows,
                                          int64_t row_bytes);
}  // namespace xllm::kernel::mlu

// ---- bangc_kda_full ----

namespace xllm::kernel::mlu {
// Non-owning contiguous BF16 q/k/v and output, FP32 state, normalized int32
// metadata. The generic ABI below handles other recurrent dimensions, data
// types and metadata layouts.
struct BangcKdaFullParams {
  const void* q;
  const void* k;
  const void* v;
  const void* gate;
  const void* beta;
  const float* a_log;
  const float* bias;
  const float* initial_state;
  float* final_state;
  void* output;
  const int32_t* cu;
  const int32_t* indices;
  const int32_t* accepted;
  int32_t sequences;
  int32_t query_heads;
  int32_t value_heads;
  int32_t columns;
  int32_t speculative;
  int32_t
      state_mode;  // 0 token output, 1 checkpoint, 2 per-request, 3 final slot.
  int32_t gate_mode;   // 0 raw safe sigmoid, 1 raw softplus.
  int32_t gate_dtype;  // 0 BF16, 1 FP32.
  int32_t beta_dtype;
  int32_t qk_l2norm;
  float scale;
  float lower_bound;
  float softplus_beta;
  float softplus_threshold;
};
extern "C" int32_t bangc_kda_full_launch(void* queue,
                                         const BangcKdaFullParams* params);
}  // namespace xllm::kernel::mlu

// ---- bangc_kda_generic ----

namespace xllm::kernel::mlu {
// Generic FP32 arithmetic ABI. The tensor wrapper performs only dtype/layout
// conversion. Carry is caller-owned [32,key_dimension] FP32, live until the
// current-stream launch completes. No workspace is shared between calls.
struct BangcKdaGenericParams {
  const float* q;
  const float* k;
  const float* v;
  const float* gate;
  const float* beta;
  const float* a_log;
  const float* bias;
  const float* initial_state;
  float* final_state;
  float* output;
  float* carry;
  const void* cu;
  const void* indices;
  const void* accepted;
  int64_t sequences;
  int64_t query_heads;
  int64_t value_heads;
  int64_t key_dimension;
  int64_t value_dimension;
  int64_t columns;
  int64_t index_sequence_stride;
  int64_t index_token_stride;
  int64_t tokens;
  int64_t initial_slots;
  int64_t final_slots;
  int32_t metadata_wide_mask;
  int32_t speculative;
  int32_t state_mode;
  int32_t gate_mode;
  int32_t qk_l2norm;
  float scale;
  float lower_bound;
  float softplus_beta;
  float softplus_threshold;
  // Optional caller-zeroed [final_slots,value_heads,value_dimension] flags.
  // A row is marked only by the IPU that actually writes that complete row.
  int32_t* written_rows;
};
extern "C" int32_t bangc_kda_generic_launch(
    void* queue,
    const BangcKdaGenericParams* params);
}  // namespace xllm::kernel::mlu

// ---- bangc_kda_norm_gate ----

// All buffers are caller-owned and remain live on the supplied queue.
extern "C" void kda_norm_gate(void* queue,
                              const void* input,
                              const void* gate,
                              const void* weight,
                              void* output,
                              int64_t rows,
                              float eps);
