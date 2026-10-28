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
#include <torch/types.h>

#include <optional>
namespace xllm::kernel::mlu::bangc {
// In place on a contiguous fp16/bf16/fp32 [..., width] tensor whose last
// dimension is gate/up concatenated: gate = min(gate, limit),
// up = clamp(up, -limit, limit); NaN is preserved. With token_count (1-D
// int32/int64 per-expert row counts of a 2-D input), only rows
// [0, sum(token_count)) are guaranteed to be clamped; later rows may or may
// not be. Enqueues on the guarded device's current stream and supports graph
// capture.
void fused_moe_clamp(torch::Tensor& input,
                     double limit,
                     const std::optional<torch::Tensor>& token_count);
}  // namespace xllm::kernel::mlu::bangc
