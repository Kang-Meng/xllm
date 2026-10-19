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

#include "sliding_window_block_manager.h"

#include <algorithm>
#include <iterator>

#include "framework/config/kv_cache_config.h"
#include "framework/prefix_cache/prefix_cache.h"

namespace xllm {

SlidingWindowBlockManager::SlidingWindowBlockManager(const Options& options)
    : BlockManagerImpl(options) {
  CHECK_GT(options_.swa_blocks_per_seq(), 0u)
      << "swa_blocks_per_seq must be positive";
  if (options_.enable_prefix_cache()) {
    // SWA prefix cache uses Sequence::block_hashes_ (TEXT chain). VLM MM
    // hasher is not compatible; fail loud instead of silently corrupting hits.
    CHECK(options_.hasher_type() == BlockHasherType::TEXT ||
          options_.hasher_type() == BlockHasherType::MTP_TEXT)
        << "SWA prefix cache does not yet support VLM (MM hasher). "
           "Disable prefix cache for VLM DSV4 or wait for VLM support.";
  }
}

std::optional<std::vector<Block>>
SlidingWindowBlockManager::allocate_for_sequence(Sequence* seq,
                                                 KVCacheState& kv_state,
                                                 size_t num_tokens) {
  if (seq == nullptr) {
    return std::nullopt;
  }

  release_out_of_window(seq, kv_state);

  const size_t block_size = options_.block_size();
  CHECK_GT(block_size, 0u);
  const size_t needed = (num_tokens + block_size - 1) / block_size;

  // Both cases pad invalid blocks through the window start, then the growth
  // below allocates only the missing tail. A cursor behind the sequence uses
  // the cached length; decode admission has not advanced its cursor yet, so
  // it uses the sequence length.
  const size_t seq_cached_tokens = seq->kv_cache_tokens_num();
  const size_t state_cached_tokens = kv_state.kv_cache_tokens_num();
  size_t window_basis = 0;
  if (seq_cached_tokens > state_cached_tokens) {
    window_basis = seq_cached_tokens;
  } else if (kv_state.num_blocks(block_type()) == 0 &&
             options_.instance_is_decode()) {
    window_basis = num_tokens;
  }
  if (window_basis > 0) {
    const size_t sliding_window_tokens =
        std::max<size_t>(options_.sliding_window_size(), 1);
    const size_t window_start = window_basis > sliding_window_tokens
                                    ? window_basis - sliding_window_tokens
                                    : 0;
    // Checkpoint offload keeps swa_blocks_per_seq blocks at the tail. That
    // span can start before the token window when the two options disagree.
    const size_t token_inactive = window_start / block_size;
    const size_t retained_blocks = std::max<size_t>(
        needed - token_inactive, options_.swa_blocks_per_seq());
    const size_t align_len =
        needed > retained_blocks ? needed - retained_blocks : 0;
    std::vector<Block>* held_blocks = kv_state.mutable_blocks(block_type());
    if (held_blocks->size() < align_len) {
      held_blocks->resize(align_len);
    }
  }

  const size_t held = kv_state.num_blocks(block_type());
  if (needed <= held) {
    return std::vector<Block>{};
  }

  const size_t missing = needed - held;
  std::vector<Block> allocated = allocate(missing);
  if (allocated.size() != missing) {
    deallocate(std::move(allocated));
    return std::nullopt;
  }
  return allocated;
}

bool SlidingWindowBlockManager::allocate_for_prefetch(Sequence* seq,
                                                      size_t num_tokens) {
  if (seq == nullptr) {
    return false;
  }
  const size_t block_size = options_.block_size();
  CHECK_GT(block_size, 0u);
  const size_t target_blocks = num_tokens / block_size;
  size_t checkpoint_span_blocks = 1;
  size_t checkpoint_window_blocks = 1;
  for (uint32_t compress_ratio : options_.compress_ratios()) {
    if (compress_ratio == kDsv4C128CompressRatio) {
      const int64_t compressed_unit_tokens = KVCacheConfig::get_instance()
                                                 .dsv4_cache_geometry()
                                                 .compressed_block_token_size();
      CHECK_GT(compressed_unit_tokens, 0);
      CHECK_EQ(compressed_unit_tokens % block_size, 0);
      checkpoint_span_blocks =
          static_cast<size_t>(compressed_unit_tokens / block_size);
      checkpoint_window_blocks = std::min<size_t>(
          checkpoint_span_blocks, options_.swa_blocks_per_seq());
      break;
    }
  }

  KVCacheState& host_state = seq->host_kv_state();
  const size_t cached_cursor =
      std::min(host_state.num_cached_blocks(BlockType::SWA), target_blocks);
  const size_t shared_blocks = host_state.shared_blocks_num(BlockType::SWA);
  std::vector<Block> old = host_state.take_blocks(BlockType::SWA);
  old.resize(target_blocks);

  std::vector<Block> blocks(target_blocks);
  std::vector<Block> dropped;
  dropped.reserve(old.size());
  auto is_checkpoint = [checkpoint_span_blocks,
                        checkpoint_window_blocks](size_t index) {
    return index % checkpoint_span_blocks >=
           checkpoint_span_blocks - checkpoint_window_blocks;
  };
  for (size_t index = 0; index < old.size(); ++index) {
    if (!old[index].is_valid()) {
      continue;
    }
    // Keep the window tail at each compressed boundary. Existing prefix
    // probes may be dense, so collapse them before adding Store destinations.
    if (is_checkpoint(index)) {
      blocks[index] = std::move(old[index]);
    } else {
      dropped.emplace_back(std::move(old[index]));
    }
  }
  if (!dropped.empty()) {
    deallocate(std::move(dropped));
  }

  seq->update_block_hashes(static_cast<uint32_t>(block_size),
                           options_.hasher_type());
  const Slice<XXH3Key> hashes = seq->block_hashes();
  CHECK_GE(hashes.size(), target_blocks);
  std::vector<size_t> missing;
  for (size_t index = cached_cursor; index < target_blocks; ++index) {
    if (is_checkpoint(index) && !blocks[index].is_valid()) {
      missing.emplace_back(index);
    }
  }

  const size_t allocatable = std::min(
      missing.size(), num_free_blocks() + num_blocks_in_prefix_cache());
  std::vector<Block> allocated = allocate(allocatable);
  if (allocated.empty() && allocatable > 0) {
    allocated = allocate(std::min(missing.size(), num_free_blocks()));
  }
  for (size_t i = 0; i < allocated.size(); ++i) {
    allocated[i].set_hash_value(hashes[missing[i]].data);
    blocks[missing[i]] = std::move(allocated[i]);
  }
  if (!blocks.empty()) {
    host_state.replace_composite_blocks(BlockType::SWA,
                                        std::move(blocks),
                                        std::min(shared_blocks, target_blocks),
                                        target_blocks);
  }
  return allocated.size() == missing.size();
}

void SlidingWindowBlockManager::trim_prefetch_blocks(Sequence* seq,
                                                     size_t max_hit_tokens) {
  if (seq == nullptr || block_size() == 0) {
    return;
  }
  KVCacheState& host_state = seq->host_kv_state();
  std::vector<Block> blocks = host_state.take_blocks(BlockType::SWA);
  const size_t keep = std::min(max_hit_tokens / block_size(), blocks.size());
  const size_t window_begin = keep > options_.swa_blocks_per_seq()
                                  ? keep - options_.swa_blocks_per_seq()
                                  : 0;

  std::vector<Block> to_release;
  to_release.reserve(blocks.size());
  for (size_t index = 0; index < blocks.size(); ++index) {
    if (!blocks[index].is_valid() || (index >= window_begin && index < keep)) {
      continue;
    }
    to_release.emplace_back(std::move(blocks[index]));
  }
  if (!to_release.empty()) {
    deallocate(std::move(to_release));
  }
  blocks.resize(keep);
  std::vector<Block> retained(keep);
  std::vector<Block> published;
  seq->update_block_hashes(static_cast<uint32_t>(block_size()),
                           options_.hasher_type());
  const Slice<XXH3Key> hashes = seq->block_hashes();
  for (size_t index = window_begin; index < keep; ++index) {
    if (!blocks[index].is_valid()) {
      continue;
    }
    CHECK_GT(hashes.size(), index);
    blocks[index].set_hash_value(hashes[index].data);
    published.push_back(blocks[index]);
    retained[index] = std::move(blocks[index]);
  }
  if (published.empty()) {
    host_state.erase_blocks(BlockType::SWA);
    return;
  }
  // The regular token-chain insert assumes a dense vector. SWA's logical
  // vector is sparse by design, so publish only the surviving absolute
  // checkpoint block through the block-identity overload.
  BlockManagerImpl::cache(published);
  host_state.replace_composite_blocks(
      BlockType::SWA, std::move(retained), keep, keep);
}

void SlidingWindowBlockManager::release_out_of_window(Sequence* seq,
                                                      KVCacheState& kv_state) {
  if (seq == nullptr) {
    return;
  }
  std::vector<Block>& swa_blocks = *kv_state.mutable_blocks(block_type());
  const size_t release_blocks =
      num_out_of_window_blocks(kv_state, seq->kv_cache_tokens_num());
  if (release_blocks == 0) {
    return;
  }
  // Move slid-out blocks out (leaving invalid placeholders so positional
  // indexing stays stable). Cache alias still pins the physical block, so
  // deallocate walks the ref<=2u branch and only clears usage bookkeeping.
  std::vector<Block> blocks_to_release;
  blocks_to_release.reserve(release_blocks);
  for (size_t j = 0; j < release_blocks; ++j) {
    if (swa_blocks[j].is_valid()) {
      blocks_to_release.emplace_back(std::move(swa_blocks[j]));
    }
  }
  if (!blocks_to_release.empty()) {
    deallocate(std::move(blocks_to_release));
  }
}

size_t SlidingWindowBlockManager::num_out_of_window_blocks(
    const KVCacheState& kv_state,
    size_t cached_tokens) const {
  const size_t block_size = options_.block_size();
  const size_t held = kv_state.num_blocks(block_type());
  if (block_size == 0 || held == 0) {
    return 0;
  }
  // cached_tokens is the number of committed KV tokens. The retained window
  // consists of the last `sliding_window_tokens` tokens, whose first token is
  // at `cached_tokens - sliding_window_tokens`. Release every block before
  // the block containing that token.
  const size_t sliding_window_tokens =
      std::max<size_t>(options_.sliding_window_size(), 1);
  if (cached_tokens <= sliding_window_tokens) {
    return 0;
  }
  const size_t window_start_token = cached_tokens - sliding_window_tokens;
  const size_t first_live_block = window_start_token / block_size;
  return std::min(first_live_block, held);
}

std::vector<Block> SlidingWindowBlockManager::allocate_shared(
    const Slice<int32_t>& token_ids,
    const Slice<Block>& /*existed_shared_blocks*/,
    const MMData& mm_data,
    const Slice<XXH3Key>& block_hashes) {
  if (!options_.enable_prefix_cache() || options_.block_size() == 0 ||
      prefix_cache_ == nullptr) {
    return {};
  }
  AUTO_COUNTER(prefix_cache_latency_seconds_match);
  std::vector<Block> result = prefix_cache_->match(token_ids,
                                                   /*existed_shared_blocks=*/{},
                                                   mm_data,
                                                   block_hashes);
  if (result.empty()) {
    return {};
  }

  // Bookkeeping: mark_used only for valid positions. mark_used is idempotent
  // per block id, so blocks shared across sequences are only counted once.
  size_t added = 0;
  for (const auto& b : result) {
    if (b.is_valid() && mark_used(&usage_accounted_ids_, b.id())) {
      ++added;
    }
  }
  num_used_blocks_.fetch_add(added, std::memory_order_relaxed);

  return result;
}

}  // namespace xllm
