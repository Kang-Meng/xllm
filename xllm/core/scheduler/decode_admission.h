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

#include <absl/time/time.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

#include "framework/block/kv_cache_manager.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"

namespace xllm {

class BlockManagerPool;

enum class DecodeAdmissionResult : int8_t {
  ADMITTED = 0,
  // Transient: admission is paused or the decode reserve is short. Prefill
  // may retry the request later.
  RETRY_LATER = 1,
  // Blocks could not be allocated. The caller decides whether the prompt is
  // permanently too long.
  REJECTED = 2,
};

// Per-DP-rank decode work the scheduler queues own in one step.
struct DecodeRankLoad {
  // Blocks promised to scheduled requests: each decode reserve, plus the
  // blocks an in-flight restore still needs to reach its full length.
  std::vector<size_t> reserved_blocks;
  // Requests holding or about to hold device blocks.
  std::vector<size_t> num_requests;
  // Every request the scheduler owns, including restore waiters.
  std::unordered_set<const Request*> requests;
};

// Blocks on `dp_rank` not held by any sequence, including evictable
// prefix-cache blocks.
size_t decode_headroom_blocks(const KVCacheManager& kv_cache_manager,
                              int32_t dp_rank);

// Blocks a decode-stage sequence keeps free so it can generate up to
// `reserved_tokens` more tokens, capped by its remaining max tokens.
size_t decode_reserve_blocks(const Sequence& sequence,
                             int32_t reserved_tokens,
                             size_t block_size);

// Gate for PD decode admission. The scheduler thread pauses it while
// preempted requests restore and publishes the per-rank decode reserve each
// step; brpc threads admit new requests against that reserve. A request
// admitted here stays counted until the scheduler queues own it, so its
// reserve is never lost between admission and its first schedule.
class DecodeAdmission final {
 public:
  struct Options {
    int32_t dp_size = 1;
    // 0 disables the reserve.
    int32_t reserved_tokens_per_request = 0;
    int32_t num_speculative_tokens = 0;
    double max_kv_cache_utilization = 1.0;
  };

  DecodeAdmission(BlockManagerPool* block_manager_pool, Options options);

  int32_t reserved_tokens_per_request() const {
    return options_.reserved_tokens_per_request;
  }

  // Blocks `sequence` keeps free for its own growth. On an idle rank it keeps
  // only its next decode step, so a request that fits alone is never refused
  // for a reserve that no other request competes for.
  size_t own_reserve_blocks(const Sequence& sequence, bool idle_rank) const;

  bool paused() const;

  // ===== brpc threads =====
  // Allocates the prompt blocks of `request` unless admission is paused, KV
  // usage is over the threshold, or the rank cannot keep the decode reserve.
  // Serialized so concurrent RPCs cannot spend the same headroom.
  DecodeAdmissionResult admit(const std::shared_ptr<Request>& request);

  // ===== Scheduler thread =====
  void pause();
  void reopen();
  // Per-rank reserve of admitted requests not yet in `load`.
  std::vector<size_t> unscheduled_reserve_blocks(const DecodeRankLoad& load);
  // Replaces the reserve admission checks against with `load` plus admitted
  // requests not yet in it.
  void publish(const DecodeRankLoad& load);

 private:
  // Admitted request the scheduler queues do not own yet.
  struct PendingAdmission {
    std::weak_ptr<Request> request;
    int32_t dp_rank = 0;
    size_t reserve_blocks = 0;
  };

  // Drops pending admissions that ended or that `load` now owns.
  void prune_pending_locked(const DecodeRankLoad& load);

  BlockManagerPool* block_manager_pool_;
  const Options options_;

  mutable std::mutex mutex_;
  bool paused_ = false;
  absl::Time paused_at_;
  std::vector<size_t> reserved_blocks_;
  std::vector<size_t> num_requests_;
  std::vector<PendingAdmission> pending_;
};

}  // namespace xllm
