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
#include <cnrt.h>

#include <cstdint>
namespace xllm::kernel::mlu::bangc {
// Native shard counts are two and four.
void launch_dcp_merge_topk(int64_t shards,
                           const void* scores,
                           const int32_t* slots,
                           const int32_t* mapping,
                           int32_t* output,
                           int32_t* lengths,
                           int64_t rows,
                           int64_t sd,
                           int64_t sq,
                           int64_t sc,
                           int64_t vd,
                           int64_t vq,
                           int64_t vc,
                           int64_t ms,
                           cnrtQueue_t queue);
}  // namespace xllm::kernel::mlu::bangc
