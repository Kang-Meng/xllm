#pragma once

#include <vector>

#include "block_manager.h"
#include "core/framework/model/parameters.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"

namespace xllm {

class BlockManagerPool {
 public:
  explicit BlockManagerPool(const BlockManager::Options& options,
                            int32_t dp_size = 1);

  ~BlockManagerPool() = default;

  BlockManager* get_block_manager(Sequence* sequence, bool is_host);

  bool allocate(Sequence* sequence);
  bool allocate(std::vector<Sequence*>& sequences);
  bool allocate(Sequence* sequence, size_t num_tokens);

  // Try to allocate blocks with num_tokens,
  // return {} if not enough blocks
  std::vector<Block> allocate(size_t num_tokens, int32_t& dp_rank);

  void deallocate(Request* request);
  void deallocate(std::vector<Sequence*>& sequences);
  void deallocate(Sequence* sequence);

  void allocate_shared(Sequence* sequence);
  void cache(Sequence* sequence);
  void copy_in_blocks_for(Request* request);
  void copy_in_blocks_for(std::vector<Sequence*>& sequences);
  void copy_in_blocks_for(Sequence* sequence);

  void copy_out_blocks_for(Request* request, bool is_preempted = false);
  void copy_out_blocks_for(std::vector<Sequence*>& sequences,
                           bool is_preempted = false);
  void copy_out_blocks_for(Sequence* sequence, bool is_preempted = false);

  std::vector<std::vector<CacheContent>>* get_copy_in_content();
  std::vector<std::vector<CacheContent>>* get_copy_out_content();
  void reset_copy_content();

  void get_merged_kvcache_event(KvCacheEvent* event) const;
  float get_gpu_cache_usage_perc() const;

  std::vector<size_t> num_blocks_in_prefix_cache() const;
  std::vector<size_t> num_free_blocks() const;
  std::vector<size_t> num_used_blocks() const;
  double kv_cache_utilization() const;

  // get the options for the block manager
  const BlockManager::Options& options() const { return options_; }

 private:
  int32_t get_manager_with_max_free_blocks() const;
  int32_t get_dp_rank(Sequence* sequence) const;

 private:
  std::vector<std::unique_ptr<BlockManager>> block_managers_;
  std::vector<std::unique_ptr<BlockManager>> host_block_managers_;

  // the options for the block manager
  Options options_;

  // cachecontent per step
  std::vector<std::vector<CacheContent>> copy_in_cache_contents_;
  std::vector<std::vector<CacheContent>> copy_out_cache_contents_;
  std::vector<std::vector<Block>> evict_host_blocks_;
};

}  // namespace xllm
