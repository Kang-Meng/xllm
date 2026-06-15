/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "sequence_kv_state.h"

#include <algorithm>
#include <limits>

namespace xllm {

namespace {
void try_replace_unique_blocks(std::vector<Block>&& matched_shared_blocks,
                               CacheGroupState* c1) {
  uint32_t num_matched_shared_blocks = matched_shared_blocks.size();
  if (c1->shared_blocks_num < num_matched_shared_blocks) {
    CHECK_GE(c1->blocks.size(), num_matched_shared_blocks);
    std::move(matched_shared_blocks.begin(),
              matched_shared_blocks.begin() + num_matched_shared_blocks,
              c1->blocks.begin());
    c1->shared_blocks_num = num_matched_shared_blocks;
  }
}
}  // namespace

const CacheGroupState* KVCacheState::c1_view_group() const {
  for (const CacheGroupState& group : groups_) {
    if (group.state_id == CacheStateId::C1) {
      return &group;
    }
  }
  return nullptr;
}

CacheGroupState* KVCacheState::mutable_c1_group() {
  for (CacheGroupState& group : groups_) {
    if (group.state_id == CacheStateId::C1) {
      return &group;
    }
  }
  CacheGroupState c1;
  c1.state_id = CacheStateId::C1;
  groups_.emplace_back(std::move(c1));
  return &groups_.back();
}

size_t KVCacheState::shared_kv_blocks_num() const {
  if (const CacheGroupState* c1 = c1_view_group()) {
    return c1->shared_blocks_num;
  }
  return 0;
}

CacheGroupState* KVCacheState::group_state(CacheStateId state_id) {
  for (CacheGroupState& group : groups_) {
    if (group.state_id == state_id) {
      return &group;
    }
  }
  return nullptr;
}

Slice<Block> KVCacheState::group_blocks(CacheStateId state_id) const {
  for (const CacheGroupState& group : groups_) {
    if (group.state_id == state_id) {
      return group.blocks;
    }
  }
  return {};
}

std::vector<const CacheGroupState*> KVCacheState::multi_block_table_groups()
    const {
  std::vector<const CacheGroupState*> exported;
  for (const CacheGroupState& group : groups_) {
    if (group.export_index >= 0) {
      exported.push_back(&group);
    }
  }
  std::sort(exported.begin(),
            exported.end(),
            [](const CacheGroupState* a, const CacheGroupState* b) {
              return a->export_index < b->export_index;
            });
  return exported;
}

size_t KVCacheState::shared_kv_tokens_num() const {
  const CacheGroupState* c1 = c1_view_group();
  if (c1 == nullptr || c1->blocks.empty() || c1->shared_blocks_num == 0) {
    return 0;
  }
  return c1->shared_blocks_num * c1->blocks[0].size();
}

size_t KVCacheState::kv_cache_tokens_num() const {
  return kv_cache_tokens_num_;
}

void KVCacheState::set_kv_cache_tokens_num(size_t num) {
  kv_cache_tokens_num_ = num;
}

void KVCacheState::incr_kv_cache_tokens_num(size_t num) {
  CHECK(kv_cache_tokens_num_ + num <= current_max_tokens_capacity());
  kv_cache_tokens_num_ += num;
}

void KVCacheState::add_kv_blocks(const std::vector<Block>& new_blocks) {
  // Append to the C1 attention group, creating it on demand. Used by the test
  // seeding helpers and any path that grows attention KV outside the composite
  // manager's own per-group policies.
  CacheGroupState* c1 = mutable_c1_group();
  c1->blocks.insert(c1->blocks.end(), new_blocks.begin(), new_blocks.end());
}

void KVCacheState::incr_shared_kv_blocks_num(size_t num) {
  // Bump the C1 group's shared-block count. The decode path (disagg PD) treats
  // the whole prefilled prompt as a prefix-cache hit; prefetch results land
  // here too. Only C1 is handled for now -- SINGLE_RES / compressed groups need
  // finer accounting once prefetch returns matched tokens instead of blocks.
  CacheGroupState* c1 = mutable_c1_group();
  CHECK(c1->shared_blocks_num + num <= c1->blocks.size());
  c1->shared_blocks_num += num;
}

void KVCacheState::add_shared_kv_blocks(std::vector<Block>&& blocks,
                                        size_t current_total_num_tokens) {
  // Attach matched prefix-cache blocks to the C1 attention group, creating it
  // on demand.
  if (blocks.empty()) {
    return;
  }
  CacheGroupState* c1 = mutable_c1_group();
  // The number of matched blocks may be fewer than the number of blocks held by
  // the sequence itself. In this case, try to replace the blocks computed by
  // the sequence with blocks from the prefix_cache and release the computed
  // blocks to save kv_cache as much as possible.
  if (blocks.size() <= c1->blocks.size()) {
    try_replace_unique_blocks(std::move(blocks), c1);
    return;
  }

  c1->blocks.clear();
  c1->shared_blocks_num = blocks.size();
  c1->blocks = std::move(blocks);

  // update the kv cache position
  size_t num_shared_tokens = c1->blocks.size() * c1->blocks[0].size();
  // It is possible that num_shared_tokens == current_total_num_tokens,
  // indicating that the exact same prompt has been received again. In this
  // case, it becomes necessary to adjust the kv cache position to the
  // previous token, allowing the model proceed. While the shared blocks
  // should be immutable ideally, but it remains safe to regenerate the kv
  // cache in this context, given the utiliztion of the exact same token.
  if (num_shared_tokens == current_total_num_tokens) {
    size_t block_size = c1->blocks[0].size();
    CHECK_GT(block_size, 0);
    num_shared_tokens =
        ((current_total_num_tokens - 1) / block_size) * block_size;
    if (c1->shared_blocks_num > 0) {
      c1->shared_blocks_num--;
      c1->blocks.pop_back();
    }
  }
  CHECK_LT(num_shared_tokens, current_total_num_tokens);
  // update the kv cache position
  kv_cache_tokens_num_ = num_shared_tokens;
}

size_t KVCacheState::current_max_tokens_capacity() const {
  if (const CacheGroupState* c1 = c1_view_group()) {
    // Composite path with a C1 attention group (normal / Qwen3.5+): only the
    // incremental C1 group carries a linear token capacity; an empty group
    // contributes none yet.
    if (c1->blocks.empty()) {
      return 0;
    }
    return c1->blocks.size() * c1->blocks[0].size();
  }
  if (!groups_.empty()) {
    // Composite path with no C1 group (DSV4): only the incremental compressed
    // groups carry a token-linear capacity, and the binding constraint is the
    // tighter of the two. The SWA ring's windowed capacity must not join the
    // min -- committed tokens keep growing past N * block_size by replacement,
    // so folding it in would fail the capacity CHECK on long sequences.
    // SINGLE_RES (one block per sequence) is equally excluded.
    size_t capacity = std::numeric_limits<size_t>::max();
    bool has_compressed = false;
    for (const CacheStateId state_id : {CacheStateId::C4, CacheStateId::C128}) {
      const Slice<Block> blocks = group_blocks(state_id);
      if (!blocks.empty()) {
        has_compressed = true;
        capacity = std::min(capacity, blocks.size() * blocks[0].size());
      }
    }
    return has_compressed ? capacity : 0;
  }
  return 0;
}

// returns allocated cache blocks
Slice<Block> KVCacheState::kv_blocks() const {
  if (const CacheGroupState* c1 = c1_view_group()) {
    return c1->blocks;
  }
  return {};
}

// get the number of blocks
size_t KVCacheState::num_kv_blocks() const {
  if (const CacheGroupState* c1 = c1_view_group()) {
    return c1->blocks.size();
  }
  return 0;
}

std::vector<int32_t> KVCacheState::kv_cache_slots(int32_t pos_start,
                                                  int32_t pos_end) {
  const Slice<Block> blocks = kv_blocks();
  CHECK(!blocks.empty()) << "no cache blocks available";

  std::vector<int32_t> slots;
  slots.reserve(pos_end - pos_start);

  const size_t block_size = blocks[0].size();
  for (int32_t i = pos_start; i < pos_end; ++i) {
    const int32_t block_id = blocks[i / block_size].id();
    const int32_t block_offset = i % block_size;
    slots.push_back(block_id * block_size + block_offset);
  }
  return slots;
}

void KVCacheState::set_transfer_kv_info(TransferKVInfo&& info) {
  transfer_kv_info_ = std::move(info);
}

std::optional<TransferKVInfo>& KVCacheState::transfer_kv_info() {
  return transfer_kv_info_;
}

size_t KVCacheState::next_transfer_block_idx() const {
  return next_transfer_block_idx_;
}

void KVCacheState::set_next_transfer_block_idx(size_t idx) {
  next_transfer_block_idx_ = idx;
}

void KVCacheState::advance_transfer_block_idx(size_t idx) {
  next_transfer_block_idx_ = std::max(next_transfer_block_idx_, idx);
}

void KVCacheState::reset() {
  kv_cache_tokens_num_ = 0;
  pushed_local_block_count_ = 0;
  groups_.clear();
  transfer_kv_info_.reset();
  next_transfer_block_idx_ = 0;
}

void KVCacheState::reset_single_resource_group() {
  if (CacheGroupState* group = group_state(CacheStateId::SINGLE_RES)) {
    group->blocks.clear();
    group->shared_blocks_num = 0;
    group->prefix_cached_tokens = 0;
  }
}

void KVCacheState::process_beam_search(std::optional<Block> new_block) {
  // Adopt the scored source beam's KV blocks as this sequence's own. The live
  // attention view is the C1 group, so the swap targets c1->blocks, creating
  // the group on demand. When new_block is set (need_swap COW) the shared last
  // block is replaced by the freshly allocated copy so the new token does not
  // overwrite a block another beam still reads.
  CacheGroupState* c1 = mutable_c1_group();
  c1->blocks.clear();
  c1->blocks = std::move(src_blocks_);
  if (new_block.has_value()) {
    c1->blocks.pop_back();
    c1->blocks.emplace_back(new_block.value());
  }
}

}  // namespace xllm
