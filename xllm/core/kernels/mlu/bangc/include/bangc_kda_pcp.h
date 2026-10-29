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
#include <type_traits>

namespace xllm::kernel::mlu {

// Non-owning contiguous tensor views. The caller retains their owners on queue.
// Data kinds: 0=FP32, 1=BF16, 2=FP16. Metadata: 0=int32, 1=int64.
struct BangcKdaPcpParams final {
  const void* q;
  const void* k;
  const void* v;
  const void* raw_gate;
  const void* raw_beta;
  const float* a_log;
  const float* dt_bias;
  const void* cu;
  const void* indices;
  float* inverse;
  float* w;
  float* u;
  float* qg;
  float* kg;
  float* aq;
  float* gate_cumsum;
  float* summary;
  const float* initial;
  float* final_state;
  void* output;
  const float* gathered;
  int64_t tokens;
  int64_t chunks;
  int64_t sequences;
  int64_t heads;
  int32_t block;
  int32_t q_kind;
  int32_t k_kind;
  int32_t v_kind;
  int32_t gate_kind;
  int32_t beta_kind;
  int32_t cu_kind;
  int32_t indices_kind;
  int32_t rank;
  float lower_bound;
};
// Shared host/device launcher layout; C linkage controls symbol naming only.
static_assert(std::is_standard_layout_v<BangcKdaPcpParams>);
static_assert(std::is_trivially_copyable_v<BangcKdaPcpParams>);

extern "C" void bangc_kda_pcp_prepare_canonical(
    cnrtQueue_t queue,
    const BangcKdaPcpParams* params);
extern "C" void bangc_kda_pcp_prepare_head_affinity(
    cnrtQueue_t queue,
    const BangcKdaPcpParams* params);
extern "C" void bangc_kda_pcp_summary(cnrtQueue_t queue,
                                      const BangcKdaPcpParams* params);
extern "C" void bangc_kda_pcp_summary_rows128(cnrtQueue_t queue,
                                              const BangcKdaPcpParams* params);
extern "C" void bangc_kda_pcp_replay_rows32(cnrtQueue_t queue,
                                            const BangcKdaPcpParams* params);
extern "C" void bangc_kda_pcp_replay_rows64(cnrtQueue_t queue,
                                            const BangcKdaPcpParams* params);
extern "C" void bangc_kda_pcp_merge(cnrtQueue_t queue,
                                    const BangcKdaPcpParams* params);

}  // namespace xllm::kernel::mlu
