/* Copyright 2025-2026 The xLLM Authors.

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

#include "block_manager_impl.h"

namespace xllm {

// Sliding-window leaf of CompositeBlockManager. Reuses BlockManagerImpl's
// physical pool and flat-append growth. SWA-specific behavior:
//   - release_out_of_window() drops leading slid-out blocks; released
//     positions stay as invalid placeholders so DSA modulo indexing
//     (`(pos/block_size) % semantic_cols`) remains stable.
//   - Prefix cache is gap-tolerant (LinearStatePrefixCache backend); the
//     tail-continuity check lives on the composite. Supports TEXT, MTP_TEXT,
//     and MM hashers; MTP_MM is rejected.
class SlidingWindowBlockManager : public BlockManagerImpl {
 public:
  explicit SlidingWindowBlockManager(const Options& options);
  ~SlidingWindowBlockManager() override = default;

  std::optional<std::vector<Block>> allocate_for_sequence(
      Sequence* seq,
      KVCacheState& kv_state,
      size_t num_tokens) override;
  bool allocate_for_prefetch(Sequence* seq, size_t num_tokens) override;
  void trim_prefetch_blocks(Sequence* seq, size_t max_hit_tokens) override;

  uint32_t swa_blocks_per_seq() const { return options_.swa_blocks_per_seq(); }

 private:
  size_t num_out_of_window_blocks(const KVCacheState& kv_state,
                                  size_t cached_tokens) const;
  void release_out_of_window(Sequence* seq, KVCacheState& kv_state);
};

}  // namespace xllm
