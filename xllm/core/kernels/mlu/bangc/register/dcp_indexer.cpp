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

#include "kernels/mlu/bangc/include/dcp_indexer.h"

#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(xllm_mlu, m) {
  m.def(
      "dcp_indexer(Tensor query, Tensor k_cache, Tensor weights, "
      "Tensor block_table, Tensor q_cu_seq_lens, Tensor query_positions, "
      "Tensor kv_seq_lens, int dcp_rank, int dcp_size, bool is_prefill) "
      "-> (Tensor(a), Tensor(a))");
}
TORCH_LIBRARY_IMPL(xllm_mlu, PrivateUse1, m) {
  m.impl("dcp_indexer", TORCH_FN(xllm::kernel::mlu::bangc::dcp_indexer));
}
