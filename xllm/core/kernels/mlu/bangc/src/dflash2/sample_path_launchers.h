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
namespace xllm::kernel::mlu::bangc {
struct DFlash2SamplePathShape {
  int64_t batch, steps, top_k;
  int64_t cb, cs, ck, eb, es, ep, ek, nb, ns, nk, ms;
};
void launch_dflash2_sample_path(void* queue,
                                const void* ids,
                                const float* edge,
                                const float* noise,
                                const bool* mask,
                                void* tokens,
                                float* proposal,
                                const DFlash2SamplePathShape* shape,
                                bool ids_int64,
                                bool all_random);
}  // namespace xllm::kernel::mlu::bangc
