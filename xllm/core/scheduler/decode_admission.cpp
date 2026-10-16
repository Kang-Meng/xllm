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

#include "scheduler/decode_admission.h"

#include <absl/time/clock.h>
#include <glog/logging.h>

#include <algorithm>
#include <utility>

#include "framework/block/block_manager_pool.h"
#include "util/utils.h"

namespace xllm {

namespace {

// Block 0 is reserved as padding and never holds a sequence.
constexpr size_t kNumPaddingBlocks = 1;

}  // namespace

size_t decode_headroom_blocks(const KVCacheManager& kv_cache_manager,
                              int32_t dp_rank) {
  const std::vector<size_t> used_blocks = kv_cache_manager.num_used_blocks();
  if (dp_rank < 0 || static_cast<size_t>(dp_rank) >= used_blocks.size()) {
    return 0;
  }
  const size_t num_blocks = kv_cache_manager.num_blocks();
  const size_t usable_blocks =
      num_blocks > kNumPaddingBlocks ? num_blocks - kNumPaddingBlocks : 0;
  return usable_blocks - std::min(used_blocks[dp_rank], usable_blocks);
}

size_t decode_reserve_blocks(const Sequence& sequence,
                             int32_t reserved_tokens,
                             size_t block_size) {
  if (reserved_tokens <= 0 || block_size == 0) {
    return 0;
  }
  size_t tokens = static_cast<size_t>(reserved_tokens);
  const StoppingChecker* stopping_checker = sequence.stopping_checker();
  if (stopping_checker != nullptr) {
    const size_t max_generated = stopping_checker->get_max_generated_tokens();
    const size_t generated = sequence.num_generated_tokens();
    tokens = std::min(
        tokens,
        max_generated > generated ? max_generated - generated : size_t{0});
  }
  return util::ceil_div(tokens, block_size);
}

DecodeAdmission::DecodeAdmission(BlockManagerPool* block_manager_pool,
                                 Options options)
    : block_manager_pool_(block_manager_pool),
      options_(std::move(options)),
      reserved_blocks_(static_cast<size_t>(std::max(options_.dp_size, 1)), 0),
      num_requests_(static_cast<size_t>(std::max(options_.dp_size, 1)), 0) {
  CHECK(block_manager_pool_ != nullptr);
}

size_t DecodeAdmission::own_reserve_blocks(const Sequence& sequence,
                                           bool idle_rank) const {
  // The next decode step writes the sampled token plus the speculative ones.
  const int32_t reserved_tokens = idle_rank
                                      ? options_.num_speculative_tokens + 1
                                      : options_.reserved_tokens_per_request;
  return decode_reserve_blocks(
      sequence,
      reserved_tokens,
      static_cast<size_t>(block_manager_pool_->block_size()));
}

bool DecodeAdmission::paused() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return paused_;
}

DecodeAdmissionResult DecodeAdmission::admit(
    const std::shared_ptr<Request>& request) {
  CHECK(request != nullptr);
  CHECK(!request->sequences().empty());
  Sequence* sequence = request->sequences()[0].get();
  std::lock_guard<std::mutex> lock(mutex_);
  if (paused_) {
    return DecodeAdmissionResult::RETRY_LATER;
  }
  // When the KV Cache usage reaches the threshold, prefill requests will no
  // longer be scheduled to avoid frequent preemption.
  if (block_manager_pool_->kv_cache_utilization() >=
      options_.max_kv_cache_utilization) {
    return DecodeAdmissionResult::REJECTED;
  }
  if (!block_manager_pool_->try_allocate(sequence)) {
    return DecodeAdmissionResult::REJECTED;
  }
  if (options_.reserved_tokens_per_request <= 0) {
    return DecodeAdmissionResult::ADMITTED;
  }

  // Keep room for admitted decodes to grow, so new arrivals do not force
  // them into preemption.
  const int32_t dp_rank = sequence->dp_rank();
  CHECK(dp_rank >= 0 && static_cast<size_t>(dp_rank) < reserved_blocks_.size())
      << "admitted dp_rank=" << dp_rank << " out of range";
  const size_t reserve_blocks =
      own_reserve_blocks(*sequence,
                         /*idle_rank=*/num_requests_[dp_rank] == 0);
  if (decode_headroom_blocks(*block_manager_pool_, dp_rank) <
      reserved_blocks_[dp_rank] + reserve_blocks) {
    // The blocks hold no KV yet, so they must not enter the prefix cache.
    block_manager_pool_->deallocate_without_cache(sequence);
    return DecodeAdmissionResult::RETRY_LATER;
  }
  reserved_blocks_[dp_rank] += reserve_blocks;
  ++num_requests_[dp_rank];
  pending_.emplace_back(PendingAdmission{.request = request,
                                         .dp_rank = dp_rank,
                                         .reserve_blocks = reserve_blocks});
  return DecodeAdmissionResult::ADMITTED;
}

void DecodeAdmission::pause() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (paused_) {
    return;
  }
  paused_ = true;
  paused_at_ = absl::Now();
  LOG(INFO) << "Decode admission paused";
}

void DecodeAdmission::reopen() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!paused_) {
    return;
  }
  paused_ = false;
  LOG(INFO) << "Decode admission reopened, paused_seconds="
            << absl::ToDoubleSeconds(absl::Now() - paused_at_);
}

std::vector<size_t> DecodeAdmission::unscheduled_reserve_blocks(
    const DecodeRankLoad& load) {
  std::lock_guard<std::mutex> lock(mutex_);
  prune_pending_locked(load);
  std::vector<size_t> reserve(reserved_blocks_.size(), 0);
  for (const PendingAdmission& pending : pending_) {
    reserve[pending.dp_rank] += pending.reserve_blocks;
  }
  return reserve;
}

void DecodeAdmission::publish(const DecodeRankLoad& load) {
  CHECK_EQ(load.reserved_blocks.size(), reserved_blocks_.size());
  CHECK_EQ(load.num_requests.size(), num_requests_.size());
  std::lock_guard<std::mutex> lock(mutex_);
  prune_pending_locked(load);
  reserved_blocks_ = load.reserved_blocks;
  num_requests_ = load.num_requests;
  for (const PendingAdmission& pending : pending_) {
    reserved_blocks_[pending.dp_rank] += pending.reserve_blocks;
    ++num_requests_[pending.dp_rank];
  }
}

void DecodeAdmission::prune_pending_locked(const DecodeRankLoad& load) {
  // A request dropped before reaching the scheduler (released reservation,
  // failed transfer) expires here; one the scheduler owns is in `load`.
  pending_.erase(std::remove_if(pending_.begin(),
                                pending_.end(),
                                [&load](const PendingAdmission& pending) {
                                  const std::shared_ptr<Request> request =
                                      pending.request.lock();
                                  return request == nullptr ||
                                         load.requests.count(request.get()) > 0;
                                }),
                 pending_.end());
}

}  // namespace xllm
