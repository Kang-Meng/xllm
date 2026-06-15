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

#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "core/common/types.h"
#include "core/util/slice.h"
#include "framework/block/block.h"
#include "framework/block/cache_group.h"

namespace xllm {

class KVCacheState {
 public:
  // get the number of tokens in the kvcache
  size_t kv_cache_tokens_num() const;
  void set_kv_cache_tokens_num(size_t num);
  void incr_kv_cache_tokens_num(size_t num);
  // get the number of shared blocks.
  size_t shared_kv_blocks_num() const;
  size_t shared_kv_tokens_num() const;

  // Append attention KV blocks to the C1 group (created on demand). Used by the
  // test seeding helpers and any path that grows attention KV outside the
  // composite manager's own per-group policies.
  void add_kv_blocks(const std::vector<Block>& new_blocks);
  void add_shared_kv_blocks(std::vector<Block>&& blocks,
                            size_t current_total_num_tokens);
  // Bump the C1 group's shared-block count (the decode / prefetch prefix-hit
  // accounting). Only C1 is handled; compressed / SINGLE_RES groups need finer
  // accounting once prefetch returns matched tokens instead of blocks.
  void incr_shared_kv_blocks_num(size_t num);

  size_t current_max_tokens_capacity() const;

  // Allocated attention KV blocks, read through the C1 group; empty when this
  // sequence holds no C1 group (e.g. DSV4).
  Slice<Block> kv_blocks() const;

  Slice<Block> src_blocks() const { return src_blocks_; };

  void set_src_blocks(const std::vector<Block>& src_blocks,
                      bool need_swap = false) {
    src_blocks_ = std::move(src_blocks);
    need_swap_ = need_swap;
  };

  bool need_swap() const { return need_swap_; }

  // get the number of blocks
  size_t num_kv_blocks() const;
  std::vector<int32_t> kv_cache_slots(int32_t pos_start, int32_t pos_end);

  // Per-cache-group runtime state, index-aligned with the owning composite
  // manager's CacheGroupRuntime entries. This is the sole KV storage: the flat
  // read views above (kv_blocks / shared counts / capacity) project the C1
  // attention group out of this vector.
  const std::vector<CacheGroupState>& groups() const { return groups_; }
  std::vector<CacheGroupState>* mutable_groups() { return &groups_; }
  // Returns the group state for `state_id`, or nullptr when this sequence holds
  // no such group.
  CacheGroupState* group_state(CacheStateId state_id);
  // Blocks currently held by `state_id`; empty when the group is absent.
  Slice<Block> group_blocks(CacheStateId state_id) const;

  // Groups that export to the worker's multi_block_tables (export_index >= 0),
  // ordered by export_index. Empty for the normal/Qwen flat path (C1 has
  // export_index == -1) and for non-composite sequences. DSV4 returns its
  // SWA / C4 / C128 groups in worker export order.
  std::vector<const CacheGroupState*> multi_block_table_groups() const;

  // True once this sequence is managed by the group-composite manager (its
  // per-group state vector has been materialized). The flat views above then
  // read through to the C1 attention group.
  bool on_composite_path() const { return !groups_.empty(); }

  void set_transfer_kv_info(TransferKVInfo&& info);
  std::optional<TransferKVInfo>& transfer_kv_info();

  uint32_t pushed_local_block_count() const {
    return pushed_local_block_count_;
  }
  void set_pushed_local_block_count(uint32_t n) {
    pushed_local_block_count_ = n;
  }

  size_t next_transfer_block_idx() const;
  void set_next_transfer_block_idx(size_t idx);
  void advance_transfer_block_idx(size_t idx);

  void reset();

  // Drop the per-sequence SINGLE_RES resource block while leaving every shared
  // KV group (C1 / compressed) untouched. A forked sequence (the beam / best_of
  // copy constructor) shares the prompt prefix by ref-counting those KV blocks,
  // but its linear / embedding state is private -- it must allocate its own
  // SINGLE_RES block on the next allocate rather than alias the source's. The
  // group entry itself is kept (only its block payload is cleared) so the
  // composite manager's per-group state vector still matches its runtime count.
  void reset_single_resource_group();

  void process_beam_search(std::optional<Block> new_block = std::nullopt);

 private:
  // The C1 attention group when this sequence holds one, else nullptr. Backs the
  // flat read views (kv_blocks / num_kv_blocks / kv_cache_slots / capacity /
  // shared counts) for normal and Qwen3.5+ models.
  const CacheGroupState* c1_view_group() const;

  // The C1 attention group, created on demand if this sequence holds none.
  // Backs the write paths (add_kv_blocks / add_shared_kv_blocks /
  // incr_shared_kv_blocks_num / process_beam_search).
  CacheGroupState* mutable_c1_group();

  // number of tokens in kv cache
  size_t kv_cache_tokens_num_ = 0;

  // source kv cache blocks for swap (beam fork / COW reads this to retarget the
  // C1 group to the scored source beam's blocks).
  std::vector<Block> src_blocks_;

  // if need to swap last block
  bool need_swap_ = false;

  // transfer kv info for disaggregated PD mode.
  std::optional<TransferKVInfo> transfer_kv_info_;

  // next logical prompt block index that needs PD PUSH transfer.
  size_t next_transfer_block_idx_ = 0;

  // Per-cache-group runtime state for the CacheGroupRuntime-based composite
  // manager. Index-aligned with the manager's states_ (worker export order).
  // DSV4 keeps its SWA / C4 / C128 groups here, each assembling its own block
  // table independently of the flat `blocks_` slot.
  std::vector<CacheGroupState> groups_;

  // Number of local KV blocks already pushed to the decode instance.
  // Used for incremental push in chunked prefill + PD disagg mode.
  uint32_t pushed_local_block_count_ = 0;
};

}  // namespace xllm
