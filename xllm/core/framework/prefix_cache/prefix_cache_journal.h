/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <absl/container/flat_hash_set.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

#include "core/framework/block/block.h"
#include "core/util/blockingconcurrentqueue.h"
#include "core/util/hash_util.h"

namespace xllm {

// Engine-local prefix-cache records. Identity, contract version, event
// reasons, and RPC failure state are filled by XServiceClient.
enum class PrefixCacheEventKind : int8_t {
  STORED = 1,
  REMOVED = 2,
  CLEARED = 3,
};

struct PrefixCacheEvent {
  PrefixCacheEventKind kind = PrefixCacheEventKind::STORED;
  XXH3Key hash;
};

struct PrefixCacheEvents {
  std::vector<PrefixCacheEvent> events;
  uint64_t commit_version = 0;
};

struct PrefixCacheSnapshot {
  std::vector<XXH3Key> hashes;
  uint64_t commit_version = 0;
};

// Engine-local event source. Prefix-cache mutations enqueue and return.
// The report thread applies the queue to the resident hashes and advances the
// commit version under mutex_. Snapshot copies the same hashes. RPC stays
// outside this object. KV, C1, and C4 never coexist, so one hash set is the
// whole published cache.
class PrefixCacheJournal final {
 public:
  static PrefixCacheJournal* instance();

  void publish_stored(const XXH3Key& hash, BlockType type);
  void publish_removed(const XXH3Key& hash, BlockType type);
  void publish_cleared();

  PrefixCacheSnapshot get_snapshot();

  PrefixCacheEvents get_prefixcache_events(size_t max_events);

 private:
  friend class PrefixCacheJournalTestPeer;

  using ResidentSet =
      absl::flat_hash_set<XXH3Key, FixedStringKeyHash, FixedStringKeyEqual>;

  void enqueue_event(PrefixCacheEventKind kind, std::optional<XXH3Key> hash);
  void apply_resident(const PrefixCacheEvent& event);

  mutable std::mutex mutex_;
  uint64_t commit_version_ = 0;
  std::atomic<size_t> pending_events_{0};
  moodycamel::BlockingConcurrentQueue<PrefixCacheEvent> pending_;
  // Current published hashes. KV, C1, and C4 never share a process, so one
  // set is enough. The commit version advances in the same critical section
  // as the mutation that produced a batch.
  ResidentSet resident_;
};

}  // namespace xllm
