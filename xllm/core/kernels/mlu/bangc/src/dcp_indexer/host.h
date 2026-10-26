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

#include <cstddef>
#include <cstdint>

namespace xllm::kernel::mlu::bangc::detail {

// Non-owning device pointers and dimensions for one launch.
struct IndexerArgs {
  const void* query;            // [N,32,128] BF16
  const void* k_cache;          // [P,1,16,128] BF16
  const void* weights;          // [N,32] BF16
  const void* block_table;      // [B,M] INT32
  const void* q_cu_seq_lens;    // [B+1] INT32
  const void* query_positions;  // [N] INT32
  const void* kv_seq_lens;      // [B] INT32
  void* scores;                 // [N,2048] BF16
  void* slots;                  // [N,2048] INT32
  void* workspace;
  int32_t num_tokens;
  int32_t batch;
  int32_t table_width;
  int32_t num_pages;
  int32_t dcp_rank;
  int32_t dcp_size;
};

// Workspace bytes for a call; zero when no row can need selection.
size_t workspace_bytes(int32_t num_tokens, int32_t table_width);

// Launches the prefill/chunked_prefill BangC kernel on the queue.
cnrtRet_t launch_prefill(cnrtQueue_t queue, const IndexerArgs& args);

// Launches the decode BangC kernel (per-token scheduling, same workspace).
cnrtRet_t launch_decode(cnrtQueue_t queue, const IndexerArgs& args);

}  // namespace xllm::kernel::mlu::bangc::detail
