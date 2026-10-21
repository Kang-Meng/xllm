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
#include <cstdint>
extern "C" void dflash2_conv_norm_launch(void* queue,
                                         const void* hidden,
                                         const void* delta,
                                         const void* base,
                                         const void* residual,
                                         const void* weight,
                                         void* output,
                                         void* residual_out,
                                         int32_t tokens,
                                         int32_t block_size,
                                         int64_t delta_row,
                                         float eps);
