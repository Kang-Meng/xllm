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

namespace xllm::kernel::mlu {

inline constexpr int64_t kBangcKdaWorkspaceFloatsPerHeadBlock = 20608;

// Bits0/1/2 select contiguous int64 CU/slot/accepted sources; other sources
// and all distinct destination buffers are int32. Caller bounds CU/slots.
extern "C" int32_t bangc_kda_normalize_metadata(void* queue,
                                                const void* cu_source,
                                                const void* indices_source,
                                                const void* accepted_source,
                                                void* cu,
                                                void* indices,
                                                void* accepted,
                                                int32_t num_sequences,
                                                int32_t checkpoint_columns,
                                                int32_t wide_mask);

// Non-owning pointers use contiguous BF16 inputs and FP32 recurrent state.
// Metadata is int32, after the caller's existing normalization.
extern "C" int32_t bangc_kda_recurrent(void* queue,
                                       const void* q,
                                       const void* k,
                                       const void* v,
                                       const void* gate,
                                       const void* beta,
                                       const void* a_log,
                                       const void* bias,
                                       const void* initial_state,
                                       void* final_state,
                                       void* output,
                                       const void* cu_seqlens,
                                       const void* state_indices,
                                       const void* accepted_tokens,
                                       int32_t num_sequences,
                                       int32_t num_heads,
                                       int32_t checkpoint_columns,
                                       int32_t speculative,
                                       int32_t inplace);

// Caller selects this entry when physical T > INT32_MAX / num_heads.
extern "C" int32_t bangc_kda_recurrent_wide(void* queue,
                                            const void* q,
                                            const void* k,
                                            const void* v,
                                            const void* gate,
                                            const void* beta,
                                            const void* a_log,
                                            const void* bias,
                                            const void* initial_state,
                                            void* final_state,
                                            void* output,
                                            const void* cu_seqlens,
                                            const void* state_indices,
                                            const void* accepted_tokens,
                                            int32_t num_sequences,
                                            int32_t num_heads,
                                            int32_t checkpoint_columns,
                                            int32_t speculative,
                                            int32_t inplace);

extern "C" int32_t bangc_kda_prefill(void* queue,
                                     const void* q,
                                     const void* k,
                                     const void* v,
                                     const void* gate,
                                     const void* beta,
                                     const void* a_log,
                                     const void* bias,
                                     const void* initial_state,
                                     void* final_state,
                                     void* output,
                                     const void* cu_seqlens,
                                     int32_t num_sequences,
                                     int32_t num_heads);

extern "C" int32_t bangc_kda_prefill_staged(void* queue,
                                            const void* q,
                                            const void* k,
                                            const void* v,
                                            const void* gate,
                                            const void* beta,
                                            const void* a_log,
                                            const void* bias,
                                            const void* initial_state,
                                            void* final_state,
                                            void* output,
                                            const void* cu_seqlens,
                                            void* workspace,
                                            int32_t num_sequences,
                                            int32_t num_heads,
                                            int32_t total_storage_tokens);

}  // namespace xllm::kernel::mlu
