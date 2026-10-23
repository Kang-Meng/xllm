/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "block_manager.h"

namespace xllm {

class BlockManagerImpl : public BlockManager {
 public:
  explicit BlockManagerImpl(const Options& options);
  virtual ~BlockManagerImpl() {
    prefix_cache_.reset();
    CHECK_EQ(num_free_blocks_, free_blocks_.size() - 1)
        << "Not all blocks have been freed";
  };

  // Try to allocate blocks with num_blocks,
  // return {} if not enough blocks
  std::vector<Block> allocate(size_t num_blocks) override;

  // Flat incremental growth: allocate ceil(num_tokens/block_size) - held blocks
  // and return them (does not insert into the sequence). The shared default for
  // flat-KV / compressed / xtensor leaves; SlidingWindow and Single override.
  std::optional<std::vector<Block>> allocate_for_sequence(
      Sequence* seq,
      KVCacheState& kv_state,
      size_t num_tokens) override;
  bool allocate_for_prefetch(Sequence* seq, size_t num_tokens) override;
  void trim_prefetch_blocks(Sequence* seq, size_t max_hit_tokens) override;

  // allocate shared blocks when enable prefix cache
  std::vector<Block> allocate_shared(
      const Slice<int32_t>& token_ids,
      const Slice<Block>& existed_shared_blocks = {},
      const MMData& mm_data = MMData(),
      const Slice<XXH3Key>& block_hashes = {}) override;

  // cache blocks when enable prefix cache
  void cache(const Slice<int32_t>& token_ids,
             std::vector<Block>& blocks,
             size_t existed_shared_blocks_num = 0,
             const MMData& mm_data = MMData(),
             const Slice<XXH3Key>& block_hashes = {}) override;
  void cache(const std::vector<Block>& blocks) override;

  size_t num_blocks_in_prefix_cache() const override {
    if (options_.enable_prefix_cache()) {
      CHECK(prefix_cache_);
      return prefix_cache_->num_blocks();
    }
    return 0;
  }

  void reset_prefix_cache() override {
    if (options_.enable_prefix_cache() && prefix_cache_) {
      prefix_cache_->evict(prefix_cache_->num_blocks());
    }
  }

  // free blocks num
  size_t num_free_blocks() const override { return num_free_blocks_; }

  // Sequence-held blocks: the usable pool fixed at init, minus free blocks,
  // minus prefix-cache residents that can be evicted.
  size_t num_used_blocks() const override;

  size_t num_evictable_blocks() const override;

  // current kv cache utilization.
  double kv_cache_utilization() const override;

  // See BlockManager::free.
  void free(const Block* block) override;

  // allocate a list of blocks
  // std::vector<Block> allocate(uint32_t n_blocks) override;

  // allocate a block
  Block allocate() override;

  // total blocks num
  size_t num_total_blocks() const override { return free_blocks_.size() - 1; }

 private:
  // check if has enough slots, if not, try to evict some blocks
  // from the prefix cache
  bool has_enough_blocks(uint32_t num_blocks);

 protected:
  // prefix cache
  std::unique_ptr<PrefixCache> prefix_cache_;

  // reserved block id for padding
  Block padding_block_;

  // free block count
  std::atomic<size_t> num_free_blocks_{0};

  // block size
  size_t block_size_ = 0;

  // free block list
  std::vector<int32_t> free_blocks_;

  // 1 once a block id has been handed out and not yet returned to the free
  // list.
  std::vector<uint8_t> usage_accounted_ids_;
};

}  // namespace xllm
