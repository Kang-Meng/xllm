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

#include "prefix_cache.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <iostream>
#include <thread>

#include "core/framework/multimodal/mm_data.h"

namespace xllm {

PrefixCache::~PrefixCache() {
  Node* node = lru_lst_.get_first();
  while (!lru_lst_.is_last(node)) {
    Node* next = lru_lst_.remove_node(node);
    if (node->block.is_valid() && node->block.ref_count() == 1) {
      release_evictable_slot();
    }
    if (node->block.is_valid()) {
      blocks_by_id_.erase(node->block.id());
    }
    delete node;
    node = next;
  }
  cached_blocks_.clear();
  blocks_by_id_.clear();
  num_blocks_ = 0;
}

void PrefixCache::check_cached_block_count() const {
  CHECK_EQ(blocks_by_id_.size(), cached_blocks_.size());
}

bool PrefixCache::update_evictable(const Block* block) {
  if (block == nullptr || block->id() < 0) {
    return false;
  }
  const bool resident = blocks_by_id_.find(block->id()) != blocks_by_id_.end();
  if (resident && block->ref_count() == 1) {
    ++num_evictable_;
  }
  return resident;
}

void PrefixCache::release_evictable_slot() {
  // update_evictable() counts a resident only after the owning manager drops
  // the last external alias. Unit tests that construct a PrefixCache beside
  // the manager never make that call, so the counter stays zero and there is
  // no slot to release.
  if (num_evictable_ == 0) {
    return;
  }
  --num_evictable_;
}

void PrefixCache::claim_matched_block(Node* node) {
  if (node == nullptr || !node->block.is_valid() ||
      node->block.ref_count() != 1) {
    return;
  }
  release_evictable_slot();
}

std::vector<Block> PrefixCache::match(const Slice<int32_t>& token_ids,
                                      const Slice<Block>& existed_shared_blocks,
                                      const MMData& mm_data,
                                      const Slice<XXH3Key>& block_hashes) {
  const size_t n_blocks =
      num_hash_blocks(hasher_type_, token_ids.size(), block_size_);
  if (n_blocks == 0) {
    return std::vector<Block>();
  }

  total_blocks_.fetch_add(n_blocks);

  std::vector<Block> blocks;
  blocks.reserve(n_blocks);
  const size_t start_block = std::min(existed_shared_blocks.size(), n_blocks);
  blocks.insert(blocks.end(),
                existed_shared_blocks.begin(),
                existed_shared_blocks.begin() + start_block);

  DNodeList node_list;

  // Look up one block by its chained hash; on hit, record the block and move
  // its LRU node to the front of the working list. Returns false on miss.
  auto match_block = [&](const XXH3Key& token_hash_key) -> bool {
    auto iter = cached_blocks_.find(token_hash_key);
    if (iter == cached_blocks_.end()) {
      return false;
    }
    claim_matched_block(iter->second);
    blocks.emplace_back(iter->second->block);
    lru_lst_.remove_node(iter->second);
    node_list.push_front(iter->second);
    return true;
  };

  // Fast path: precomputed chained hash covers every matchable block, so we
  // only do hash-table lookups and never recompute a hash.
  if (block_hashes.size() >= n_blocks) {
    for (size_t b = start_block; b < n_blocks; ++b) {
      if (!match_block(block_hashes[b])) {
        break;
      }
    }
  } else {
    // Fallback: compute the chained hash on the fly.
    XXH3Key token_hash_key =
        existed_shared_blocks.empty()
            ? XXH3Key{}
            : XXH3Key{existed_shared_blocks.back().get_immutable_hash_value()};
    auto hasher =
        BlockHasher::create(hasher_type_, mm_data, start_block * block_size_);
    for (size_t b = start_block; b < n_blocks; ++b) {
      const size_t i = b * block_size_;
      const uint8_t* pre_hash_value = (b == 0) ? nullptr : token_hash_key.data;
      hasher->compute(
          token_ids, i, i + block_size_, pre_hash_value, token_hash_key);
      if (!match_block(token_hash_key)) {
        break;
      }
    }
  }

  // update LRU list
  while (!node_list.is_empty()) {
    Node* node = node_list.pop_front();
    lru_lst_.push_back(node);
  }

  matched_blocks_.fetch_add(blocks.size());

  return blocks;
}

size_t PrefixCache::insert(const Slice<int32_t>& token_ids,
                           std::vector<Block>& blocks,
                           size_t existed_shared_blocks_num,
                           const MMData& mm_data,
                           const Slice<XXH3Key>& block_hashes) {
  check_cached_block_count();
  const int64_t now = absl::ToUnixMicros(absl::Now());
  // align tokens to block boundary
  const size_t n_blocks =
      std::min(num_hash_blocks(hasher_type_, token_ids.size(), block_size_),
               blocks.size());

  if (n_blocks <= existed_shared_blocks_num) {
    check_cached_block_count();
    return n_blocks * block_size_;
  }
  // truncate the token ids and blocks to boundary

  DNodeList node_list;

  // Fill `token_hash_key` with the chained hash of block `block_idx`, reusing
  // the precomputed hash when it covers all blocks, otherwise computing it.
  // The KV / C4 / C128 prefix is solid (no placeholders), so the compute path
  // seeds the chain in O(1) from the previous block's stamped hash. SWA's
  // slid-out placeholders are handled by LinearStatePrefixCache::insert, which
  // overrides this.
  const bool use_precomputed = block_hashes.size() >= n_blocks;
  XXH3Key token_hash_key = existed_shared_blocks_num == 0
                               ? XXH3Key{}
                               : XXH3Key{blocks[existed_shared_blocks_num - 1]
                                             .get_immutable_hash_value()};
  std::unique_ptr<BlockHasher> hasher;
  if (!use_precomputed) {
    hasher = BlockHasher::create(
        hasher_type_, mm_data, existed_shared_blocks_num * block_size_);
  }
  auto fill_block_hash = [&](size_t block_idx) {
    if (use_precomputed) {
      token_hash_key = block_hashes[block_idx];
      return;
    }
    const size_t i = block_idx * block_size_;
    const uint8_t* pre_hash_value =
        (block_idx == 0) ? nullptr : token_hash_key.data;
    hasher->compute(
        token_ids, i, i + block_size_, pre_hash_value, token_hash_key);
  };

  for (size_t block_idx = existed_shared_blocks_num; block_idx < n_blocks;
       ++block_idx) {
    fill_block_hash(block_idx);
    blocks[block_idx].set_hash_value(token_hash_key.data);

    auto iter = cached_blocks_.find(token_hash_key);
    if (iter != cached_blocks_.end()) {
      iter->second->last_access_time = now;

      lru_lst_.remove_node(iter->second);
      node_list.push_front(iter->second);
    } else {
      // A new hash must name a block that is not already registered under
      // another hash. One physical block, one chain key.
      CHECK(blocks_by_id_.find(blocks[block_idx].id()) == blocks_by_id_.end())
          << "Duplicate block id " << blocks[block_idx].id();
      Node* new_node = new Node();

      new_node->block = blocks[block_idx];
      new_node->last_access_time = now;

      node_list.push_front(new_node);

      cached_blocks_.emplace(std::make_pair(token_hash_key, new_node));
      blocks_by_id_.insert(new_node->block.id());

      num_blocks_++;
    }
  }

  const size_t n_tokens = n_blocks * block_size_;
  while (!node_list.is_empty()) {
    Node* node = node_list.pop_front();
    lru_lst_.push_back(node);
  }

  check_cached_block_count();
  return n_tokens;
}

size_t PrefixCache::insert(const std::vector<Block>& blocks) {
  Slice<Block> slice(blocks);
  return insert(slice);
}

size_t PrefixCache::insert(Slice<Block>& blocks) {
  check_cached_block_count();
  const int64_t now = absl::ToUnixMicros(absl::Now());
  DNodeList node_list;
  XXH3Key token_hash_key;

  for (size_t i = 0; i < blocks.size(); i++) {
    if (!blocks[i].is_valid()) {
      continue;
    }
    token_hash_key.set(blocks[i].get_immutable_hash_value());

    auto iter = cached_blocks_.find(token_hash_key);
    if (iter != cached_blocks_.end()) {
      iter->second->last_access_time = now;

      lru_lst_.remove_node(iter->second);
      node_list.push_front(iter->second);
    } else {
      // A new hash must name a block that is not already registered under
      // another hash. One physical block, one chain key.
      CHECK(blocks_by_id_.find(blocks[i].id()) == blocks_by_id_.end())
          << "Duplicate block id " << blocks[i].id();
      Node* new_node = new Node();

      new_node->block = blocks[i];
      new_node->last_access_time = now;

      node_list.push_front(new_node);

      cached_blocks_.emplace(std::make_pair(token_hash_key, new_node));
      blocks_by_id_.insert(new_node->block.id());

      num_blocks_++;
    }
  }

  while (!node_list.is_empty()) {
    Node* node = node_list.pop_front();
    lru_lst_.push_back(node);
  }

  check_cached_block_count();
  return blocks.size() * block_size_;
}

size_t PrefixCache::evict(size_t n_blocks) {
  check_cached_block_count();
  if (num_blocks_ == 0 || lru_lst_.is_empty()) {
    check_cached_block_count();
    return 0;
  }

  size_t evict_count = 0;
  Node* iter_node = lru_lst_.get_first();
  while (evict_count < n_blocks) {
    if (lru_lst_.is_last(iter_node)) {
      break;
    }

    if (iter_node->block.is_shared()) {  // in use
      iter_node = iter_node->next;

      continue;
    }

    Node* del_node = iter_node;

    iter_node = lru_lst_.remove_node(del_node);

    XXH3Key token_hash_key(del_node->block.get_immutable_hash_value());

    release_evictable_slot();
    blocks_by_id_.erase(del_node->block.id());
    cached_blocks_.erase(token_hash_key);

    delete del_node;
    ++evict_count;
    --num_blocks_;
  }

  check_cached_block_count();
  return evict_count;
}

uint32_t PrefixCache::compute_hash_keys(const Slice<int32_t>& token_ids,
                                        std::vector<Block>& blocks,
                                        const size_t cached_blocks) {
  if (blocks.size() == 0) {
    return 0;
  }
  int32_t block_size = blocks[0].size();
  const size_t n_blocks = (token_ids.size() + block_size - 1) / block_size;
  if (blocks.size() > n_blocks) {
    LOG(ERROR) << "token ids do not cover the allocate block.";
    return 0;
  }
  size_t full_block_size =
      std::min(token_ids.size() / block_size, blocks.size());

  for (size_t i = cached_blocks; i < full_block_size; i++) {
    if (i == 0) {
      xxh3_128bits_hash(nullptr,
                        token_ids.slice(i * block_size, (i + 1) * block_size),
                        blocks[i].get_mutable_hash_value());
    } else {
      xxh3_128bits_hash(blocks[i - 1].get_mutable_hash_value(),
                        token_ids.slice(i * block_size, (i + 1) * block_size),
                        blocks[i].get_mutable_hash_value());
    }
  }

  return full_block_size;
}

}  // namespace xllm
