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

#include "processors/kimi25_image_processor.h"
#include "processors/multimodal_processor.h"

namespace xllm {
REGISTER_MODEL_BACKEND(kimi_k3, "vlm");
using KimiK3MultimodalProcessor =
    MultimodalProcessor<KimiK25PromptProcessor, KimiK25ImageProcessor>;
REGISTER_MULTIMODAL_PROCESSOR(kimi_k3, KimiK3MultimodalProcessor);
}  // namespace xllm
