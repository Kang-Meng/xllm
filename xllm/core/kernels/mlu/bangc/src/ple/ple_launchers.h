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

// Non-owning pointers use the current MLU queue. Tensor validation belongs to
// the public tensor API; buffers must outlive queued kernel completion.
extern "C" {
void launch_pack_projection_bf16(void* queue, const void* weight, void* packed);
void launch_projection_packed_bf16(void* queue,
                                   const void* embedding,
                                   const void* packed,
                                   void* kv,
                                   int32_t rows);
void launch_prepare_head_scale_packed_hc(void* queue,
                                         const void* key_scale,
                                         const void* query_scale,
                                         const void* conv_scale,
                                         void* affine_pair_bf16,
                                         int32_t count);
void launch_head_prepared_packed_hc(void* queue,
                                    const void* kv,
                                    const void* hidden,
                                    const void* affine_pair_bf16,
                                    void* values,
                                    void* normalized,
                                    int32_t rows);
void launch_tail(void* queue,
                 const void* values,
                 const void* normalized,
                 void* history,
                 const void* taps,
                 const void* hidden,
                 void* output,
                 int32_t batch);
}
