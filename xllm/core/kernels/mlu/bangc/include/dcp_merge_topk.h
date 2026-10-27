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

namespace xllm::kernel::mlu::bangc {
// BF16 scores[D,Q,2048], INT32 slots/mapping, topk=2048; contiguous outputs.
// D is a power of two of at least 2. D=2 and D=4 run one native kernel;
// larger D first reduces each group of four shards, then merges the groups.
// Equal scores use ascending global physical slots. Positive input strides
// are supported. Outputs cannot alias any input.
void dcp_merge_topk(const torch::Tensor& scores,
                    const torch::Tensor& slots,
                    const torch::Tensor& mapping,
                    int64_t topk,
                    torch::Tensor& output,
                    torch::Tensor& lengths);
}  // namespace xllm::kernel::mlu::bangc
