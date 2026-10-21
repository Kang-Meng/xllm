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

#include "kernels/mlu/bangc/include/dflash2_grouped_conv.h"

#include <torch/library.h>
TORCH_LIBRARY_FRAGMENT(xllm_mlu, m) {
  m.def(
      "dflash2_grouped_conv(Tensor hidden, Tensor delta, Tensor base, "
      "int block_size, int group_size) -> Tensor");
}
TORCH_LIBRARY_IMPL(xllm_mlu, PrivateUse1, m) {
  m.impl("dflash2_grouped_conv",
         TORCH_FN(xllm::kernel::mlu::bangc::dflash2_grouped_conv));
}
