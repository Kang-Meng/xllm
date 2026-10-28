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

// Row bytes are bounded by the NRAM pattern buffer; counts by the staging
// buffers. The Tensor wrapper validates both before launching.
constexpr int32_t kFusedMoeClampMaxRowBytes = 32 * 1024;
constexpr int32_t kFusedMoeClampMaxCounts = 1024;

// In place on contiguous [rows, width]; dtype: 0 fp16, 1 bf16, 2 fp32.
// token_count (int32 or int64 by token_count_bytes, may be null) bounds the
// rows that must be clamped to its sum.
extern "C" void fused_moe_clamp_launch(void* queue,
                                       void* data,
                                       int64_t rows,
                                       int32_t width,
                                       int32_t dtype,
                                       float limit,
                                       const void* token_count,
                                       int32_t token_count_size,
                                       int32_t token_count_bytes);
