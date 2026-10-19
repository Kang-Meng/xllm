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

// Multimodal-processor registration for DeepSeek-V4.1-Flash served as a VLM
// via --backend=vlm --model_impl=python (PyCausalLM +
// xllm.python.models.deepseek_v41.DeepseekV41ForCausalLM). The vision tower
// and language model both live in Python, so the framework only needs a
// processor to satisfy VLMMaster's create_multimodal_processor CHECK:
//   * DeepseekV41ImageProcessor — DeepSeek resize/patchify -> pixel_values +
//     spatial image_grid_thw (see the reference load_image).
//   * DeepseekV41PromptProcessor — expands the `<｜deepseek_image｜>` /
//     `<image>` placeholder into the 129264 sentinel block with 129265 pads.

#include "models/model_registry.h"
#include "processors/deepseek_v41_image_processor.h"
#include "processors/deepseek_v41_prompt_processor.h"
#include "processors/multimodal_processor.h"

namespace xllm {

using DeepseekV41MultimodalProcessor =
    MultimodalProcessor<DeepseekV41PromptProcessor, DeepseekV41ImageProcessor>;

REGISTER_MULTIMODAL_PROCESSOR(deepseek_v41, DeepseekV41MultimodalProcessor);

}  // namespace xllm
