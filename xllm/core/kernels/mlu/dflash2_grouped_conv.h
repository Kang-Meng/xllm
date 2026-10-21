/* Copyright 2026 The xLLM Authors.

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

namespace xllm::kernel::mlu {

bool supports_dflash2_grouped_conv(const torch::Tensor& hidden_states,
                                   const torch::Tensor& delta,
                                   const torch::Tensor& base,
                                   int32_t group_size);

torch::Tensor dflash2_grouped_conv(const torch::Tensor& hidden_states,
                                   const torch::Tensor& delta,
                                   const torch::Tensor& base,
                                   int32_t block_size,
                                   int32_t group_size,
                                   int32_t taps);

}  // namespace xllm::kernel::mlu
