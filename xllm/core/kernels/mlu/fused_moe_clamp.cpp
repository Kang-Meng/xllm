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

#include "core/kernels/mlu/bangc/include/fused_moe_clamp.h"

#include "kernels/mlu/mlu_ops_api.h"

namespace xllm::kernel::mlu {

void fused_moe_clamp(torch::Tensor& input,
                     float limit,
                     const std::optional<torch::Tensor>& token_count) {
  bangc::fused_moe_clamp(input, limit, token_count);
}

}  // namespace xllm::kernel::mlu
