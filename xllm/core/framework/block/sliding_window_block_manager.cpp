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
  std::vector<Block> allocated = block_allocator()->allocate(missing);
  if (allocated.size() != missing) {
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
  const size_t cached_blocks =
      seq->host_kv_state().num_cached_blocks(BlockType::SWA);
  CHECK_LT(cached_blocks, target_blocks);

  const int64_t compressed_unit_tokens = KVCacheConfig::get_instance()
                                             .dsv4_cache_geometry()
                                             .compressed_block_token_size();
  CHECK_GT(compressed_unit_tokens, 0);
  CHECK_EQ(compressed_unit_tokens % block_size, 0);
  const size_t blocks_per_unit =
      static_cast<size_t>(compressed_unit_tokens / block_size);
  const size_t valid_per_unit =
      static_cast<size_t>(options_.swa_blocks_per_seq());
  const size_t placeholders_per_unit = blocks_per_unit - valid_per_unit;
  // Host admission ends on a compressed-unit boundary, and the prefetch target
  // does too, so the tail is a whole number of units.
  CHECK_EQ(cached_blocks % blocks_per_unit, 0u);
  CHECK_EQ(target_blocks % blocks_per_unit, 0u);
  const size_t units = (target_blocks - cached_blocks) / blocks_per_unit;
  const size_t needed = units * valid_per_unit;

  std::vector<Block> allocated = block_allocator()->allocate(needed);
  if (allocated.empty()) {
    allocated =
        block_allocator()->allocate(std::min(needed, num_free_blocks()));
  }

  KVCacheState& host_state = seq->host_kv_state();
  std::vector<Block> blocks = host_state.take_blocks(BlockType::SWA);
  CHECK_EQ(blocks.size(), cached_blocks);
  blocks.reserve(cached_blocks + units * blocks_per_unit);
  size_t used = 0;
  for (size_t unit = 0; unit < units && used < allocated.size(); ++unit) {
    blocks.resize(blocks.size() + placeholders_per_unit);
    const size_t take = std::min(valid_per_unit, allocated.size() - used);
    for (size_t index = 0; index < take; ++index) {
      blocks.emplace_back(std::move(allocated[used + index]));
    }
    used += take;
  }

  seq->update_block_hashes(static_cast<uint32_t>(block_size),
                           options_.hasher_type());
  const Slice<XXH3Key> hashes =
      seq->block_hashes(static_cast<uint32_t>(block_size));
  CHECK_GE(hashes.size(), blocks.size());
  for (size_t index = cached_blocks; index < blocks.size(); ++index) {
    if (!blocks[index].is_valid()) {
      continue;
    }
    blocks[index].set_hash_value(hashes[index].data);
  }
  if (!blocks.empty()) {
    host_state.replace_composite_blocks(
        BlockType::SWA, std::move(blocks), cached_blocks, cached_blocks);
  }
  return used == needed;
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

  for (size_t index = 0; index < keep; ++index) {
    if (!blocks[index].is_valid() || index >= window_begin) {
      continue;
    }
    blocks[index] = Block();
  }
  blocks.resize(keep);
  std::vector<Block> retained(keep);
  std::vector<Block> published;
  seq->update_block_hashes(static_cast<uint32_t>(block_size()),
                           options_.hasher_type());
  const Slice<XXH3Key> hashes =
      seq->block_hashes(static_cast<uint32_t>(block_size()));
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
  // Replace slid-out blocks with invalid placeholders so positional indexing
  // stays stable. The assignment releases the sequence reference; a cache
  // alias stays evictable when it is the only owner.
  for (size_t j = 0; j < release_blocks; ++j) {
    if (swa_blocks[j].is_valid()) {
      swa_blocks[j] = Block();
    }
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

}  // namespace xllm
