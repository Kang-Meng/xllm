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

#include "prefix_cache_journal.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <optional>
#include <utility>

#include "core/util/hash_util.h"

namespace xllm {
namespace {

// Floor for one drain. The effective capacity is max(this, resident_.size()).
constexpr size_t kDefaultDrainCapacity = 8192;

// KV, C1, and C4 never coexist, so the journal keeps one resident hash map.
bool publishes_prefix_cache(BlockType type) {
  switch (type) {
    case BlockType::KV:
    case BlockType::C1:
    case BlockType::C4:
      return true;
    case BlockType::SWA:
    case BlockType::C128:
    case BlockType::C2:
    case BlockType::EMBEDDING:
    case BlockType::LINEAR:
      return false;
  }
  return false;
}

// Steady occupancy is 60% of the queue. Each further 10% occupied, rounded
// up, adds another 10% of capacity on top of max_events. One drain then
// leaves the queue near 60%. A capacity under 10 still adds one event per
// occupied slot above the line.
size_t drain_batch_size(size_t pending, size_t capacity, size_t max_events) {
  const size_t high_watermark = capacity * 60 / 100;
  if (pending <= high_watermark) {
    return std::min(pending, max_events);
  }
  const size_t band = std::max<size_t>(capacity / 10, 1);
  const size_t above = pending - high_watermark;
  const size_t extra_bands = (above + band - 1) / band;
  if (band > 1 &&
      extra_bands > (std::numeric_limits<size_t>::max() - max_events) / band) {
    return pending;
  }
  return std::min(pending, max_events + extra_bands * band);
}

}  // namespace

PrefixCacheJournal* PrefixCacheJournal::instance() {
  static PrefixCacheJournal journal;
  return &journal;
}

void PrefixCacheJournal::publish_stored(const XXH3Key& hash, BlockType type) {
  if (!publishes_prefix_cache(type)) {
    return;
  }
  // Host, HBM, and each DP rank store the same hash independently. Only the
  // first holder publishes STORED; later stores just raise the count.
  std::lock_guard<std::mutex> lock(mutex_);
  const auto [it, inserted] = resident_.emplace(hash, 1);
  if (!inserted) {
    if (it->second < std::numeric_limits<uint32_t>::max()) {
      ++it->second;
    }
    return;
  }
  enqueue_event(PrefixCacheEventKind::STORED, hash);
}

void PrefixCacheJournal::publish_removed(const XXH3Key& hash, BlockType type) {
  if (!publishes_prefix_cache(type)) {
    return;
  }
  // One DP rank dropping the hash leaves the others. Publish REMOVED only
  // when the last holder is gone.
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = resident_.find(hash);
  if (it == resident_.end() || it->second == 0) {
    return;
  }
  --it->second;
  if (it->second != 0) {
    return;
  }
  resident_.erase(it);
  enqueue_event(PrefixCacheEventKind::REMOVED, hash);
}

void PrefixCacheJournal::publish_cleared() {
  std::lock_guard<std::mutex> lock(mutex_);
  resident_.clear();
  enqueue_event(PrefixCacheEventKind::CLEARED, std::nullopt);
}

void PrefixCacheJournal::enqueue_event(PrefixCacheEventKind kind,
                                       std::optional<XXH3Key> hash) {
  PrefixCacheEvent event;
  event.kind = kind;
  if (hash.has_value()) {
    event.hash = *hash;
  }

  if (pending_.enqueue(std::move(event))) {
    pending_events_.fetch_add(1, std::memory_order_acq_rel);
  }
}

PrefixCacheEvents PrefixCacheJournal::get_prefixcache_events(
    size_t max_events) {
  PrefixCacheEvents result;
  if (max_events == 0) {
    return result;
  }
  // One lock covers the capacity sample, the dequeue, and the resident
  // update. try_dequeue does not wait, so the lock is not held on an empty
  // queue. Publish only takes the queue lock and does not take mutex_.
  std::lock_guard<std::mutex> lock(mutex_);
  // resident_ is a holder count, not the unpublished queue. Size the drain
  // from the events still waiting to be reported.
  const size_t pending = pending_events_.load(std::memory_order_acquire);
  const size_t capacity = std::max(kDefaultDrainCapacity, pending);
  const size_t batch_size = drain_batch_size(pending, capacity, max_events);
  std::vector<PrefixCacheEvent> taken;
  taken.reserve(batch_size);
  while (taken.size() < batch_size) {
    PrefixCacheEvent event;
    if (!pending_.try_dequeue(event)) {
      break;
    }
    taken.push_back(std::move(event));
  }

  if (taken.empty()) {
    result.commit_version = commit_version_;
    return result;
  }
  pending_events_.fetch_sub(taken.size(), std::memory_order_acq_rel);
  if (commit_version_ == std::numeric_limits<uint64_t>::max()) {
    result.commit_version = commit_version_;
    return result;
  }
  ++commit_version_;
  result.commit_version = commit_version_;
  result.events = std::move(taken);
  return result;
}

PrefixCacheSnapshot PrefixCacheJournal::get_snapshot() {
  std::lock_guard<std::mutex> lock(mutex_);
  PrefixCacheSnapshot snapshot;
  snapshot.commit_version = commit_version_;
  snapshot.hashes.reserve(resident_.size());
  for (const auto& [hash, holders] : resident_) {
    if (holders > 0) {
      snapshot.hashes.push_back(hash);
    }
  }
  return snapshot;
}

}  // namespace xllm
