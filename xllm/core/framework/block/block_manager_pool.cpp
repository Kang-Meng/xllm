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

#include "block_manager_pool.h"

#include <algorithm>

#include "common/global_flags.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/service_config.h"
#include "framework/block/block_manager_context.h"
#include "framework/block/cache_group.h"
#include "framework/block/cache_group_spec_builder.h"
#include "framework/request/sequence.h"
#include "framework/xtensor/page_allocator.h"

namespace xllm {
namespace {

// Normal / Qwen3.5+ composite layout: C1 + SINGLE_RES, built by the shared
// spec builder so the linear-state gating applies on the live path -- a
// linear-attention model (enable_linear_state) must never expose a cacheable
// C1, because a prefix hit that skips prefill strands the unrecoverable
// recurrent state. The builder CHECK-fails on any violation.
//
// xtensor is just another form of the C1 group: when enable_xtensor is set the
// C1 leaf is an XTensorBlockManagerImpl (VMM page allocator) instead of the
// default BlockManagerImpl, prefix cache is forced off, and only the C1-only
// shape is allowed. `dp_rank` is the per-rank index the xtensor leaf needs for
// its page allocator; ignored for the non-xtensor leaf.
std::vector<CacheGroupSpec> make_normal_composite_specs(
    const BlockManagerPool::Options& options,
    uint32_t single_res_num_blocks,
    int32_t dp_rank) {
  ModelCacheGroupConfig config;
  config.base_block_size = static_cast<uint32_t>(options.block_size());
  config.c1_num_blocks = options.num_blocks();
  config.single_res_num_blocks = single_res_num_blocks;
  config.has_linear_state = options.enable_linear_state();
  config.enable_prefix_cache = options.enable_prefix_cache();
  if (options.enable_xtensor()) {
    CHECK_GT(options.num_layers(), 0)
        << "num_layers must be set when enable_xtensor is true";
    CHECK_GT(options.slot_size(), 0)
        << "slot_size must be set when enable_xtensor is true";
    config.c1_leaf_xtensor = true;
    config.xtensor_params.num_layers = options.num_layers();
    // K and V share one slot size, so halve it for a single side's block.
    config.xtensor_params.block_mem_size =
        static_cast<size_t>(options.block_size()) * options.slot_size() / 2;
    config.xtensor_params.page_size =
        ::xllm::KVCacheConfig::get_instance().phy_page_granularity_size();
    config.xtensor_params.dp_rank = dp_rank;
    config.xtensor_params.model_id = options.model_id();
  }
  return build_cache_group_specs(config);
}

// DeepSeek-V4 cache-group composite: a rolling-window SWA attention group
// followed by one incremental compressed group per compress ratio (C4, C128),
// in worker multi_block_tables export order, plus the trailing SINGLE_RES
// per-sequence resource group every model shape carries. Allocator sizing
// preserves the original DSV4 cache-state pool formula:
//   - SWA shares a burst pool across the in-flight batch.
//   - Each compressed pool shrinks by its ratio (one compressed block covers
//     `ratio` base blocks' worth of tokens).
// The SWA ring exports exactly `swa_window_blocks` columns, which is the modulo
// base the worker reads as semantic_cols = raw_bt.size(1)
// (dsa_metadata_builder.cpp), so no worker change is needed.
std::vector<CacheGroupSpec> make_dsv4_composite_specs(
    const BlockManagerPool::Options& options,
    uint32_t single_res_num_blocks) {
  const uint32_t block_size = static_cast<uint32_t>(options.block_size());
  CHECK_GT(block_size, 0u) << "block_size must be positive for DSV4";
  const uint32_t swa_blocks_per_seq = options.swa_blocks_per_seq();
  CHECK_GT(swa_blocks_per_seq, 0u)
      << "swa_blocks_per_seq must be positive for DSV4";

  ModelCacheGroupConfig config;
  config.base_block_size = block_size;
  config.single_res_num_blocks = single_res_num_blocks;
  config.enable_prefix_cache = options.enable_prefix_cache();

  const uint32_t max_seqs = std::max(options.max_seqs_per_batch(), 1u);
  const uint32_t burst_blocks =
      (std::max(options.max_tokens_per_batch(), 1u) + block_size - 1) /
      block_size;
  config.swa_window_blocks = swa_blocks_per_seq;
  config.swa_num_blocks =
      swa_blocks_per_seq * max_seqs + burst_blocks + max_seqs + 2;

  // The engine prepends a placeholder ratio (0) for the SWA slot; keep only the
  // real compressed ratios (C4=4, C128=128).
  for (const uint32_t ratio : options.compress_ratios()) {
    if (ratio <= 1) {
      continue;
    }
    config.compress_ratios.push_back(ratio);
    config.compress_num_blocks.push_back(options.num_blocks() / ratio);
  }

  return build_cache_group_specs(config);
}

// Bind a sequence's device KV state to one composite manager call. The normal
// pool only ever targets the device role; the host pool is unaffected here.
// The token view and prefix-hash chain let the composite insert completed
// blocks into its prefix caches from inside allocate/deallocate (see
// GroupCompositeBlockManager::insert_committed_blocks). Callers that must
// release without caching (preempt) use make_bare_device_context instead.
BlockManagerContext make_device_context(Sequence* sequence, int32_t dp_rank) {
  BlockManagerContext context;
  context.sequence = sequence;
  context.kv_state = &sequence->kv_state();
  context.role = CacheStorageRole::DEVICE;
  context.device_dp_rank = dp_rank;
  context.tokens = sequence->tokens();
  context.hash_state = &sequence->prefix_hash_state();
  return context;
}

// Like make_device_context but without the token view / hash chain, so the
// composite's internal prefix-cache insert is skipped. Used by the preempt
// path, which must release uncomputed blocks without writing them to a cache.
BlockManagerContext make_bare_device_context(Sequence* sequence,
                                             int32_t dp_rank) {
  BlockManagerContext context;
  context.sequence = sequence;
  context.kv_state = &sequence->kv_state();
  context.role = CacheStorageRole::DEVICE;
  context.device_dp_rank = dp_rank;
  return context;
}

}  // namespace

BlockManagerPool::BlockManagerPool(const Options& options, int32_t dp_size)
    : options_(options) {
  CHECK(dp_size > 0) << "dp_size must be greater than 0";

  const uint32_t max_single_block_sequences =
      options_.max_concurrent_requests() > 0
          ? options_.max_concurrent_requests()
          : static_cast<uint32_t>(std::max(
                ::xllm::ServiceConfig::get_instance().max_concurrent_requests(),
                0));
  const uint32_t num_single_blocks = std::max<uint32_t>(
      options_.num_single_blocks(), max_single_block_sequences + 2);
  CHECK_GT(num_single_blocks, 0u) << "num_single_blocks must be positive";

  // The cache-group composite is the only device-side allocation architecture.
  // xtensor is folded in as a C1 group whose leaf is an XTensorBlockManagerImpl
  // (see make_normal_composite_specs). The hierarchy host modes (host blocks /
  // kvcache store) are temporarily disabled during the block-manager refactor
  // (B5 shell-disabled HierarchyBlockManagerPool; Phase C' rebuilds them on the
  // composite base), so every sequence is served by a composite manager.
  const bool is_dsv4 = !options_.manager_types().empty();
  composite_managers_.reserve(dp_size);

  for (int32_t i = 0; i < dp_size; ++i) {
    // One composite manager per DP rank: DSV4 SWA/C4/C128 + SINGLE_RES, or the
    // normal/Qwen/xtensor C1 + SINGLE_RES. Only disagg-PD enters sequence-level
    // calls from off-scheduler threads (prefill threadpool cache/deallocate,
    // decode-side handlers), so only that mode pays for the locked subclass.
    const std::vector<CacheGroupSpec> specs =
        is_dsv4 ? make_dsv4_composite_specs(options_, num_single_blocks)
                : make_normal_composite_specs(
                      options_, num_single_blocks, /*dp_rank=*/i);
    if (options_.enable_disagg_pd()) {
      composite_managers_.emplace_back(
          std::make_unique<ConcurrentCompositeBlockManager>(specs));
    } else {
      composite_managers_.emplace_back(
          std::make_unique<GroupCompositeBlockManager>(specs));
    }
  }
  swap_block_transfer_infos_.clear();
  swap_block_transfer_infos_.resize(manager_count());
}

size_t BlockManagerPool::manager_count() const {
  return composite_managers_.size();
}

int32_t BlockManagerPool::get_manager_with_max_free_blocks() const {
  const size_t count = manager_count();
  if (count == 0) {
    return 0;
  }

  size_t max_index = 0;
  size_t max_free = composite_managers_[0]->num_free_blocks();
  for (size_t i = 1; i < count; ++i) {
    const size_t current_free = composite_managers_[i]->num_free_blocks();
    if (current_free > max_free) {
      max_free = current_free;
      max_index = i;
    }
  }
  return max_index;
}

int32_t BlockManagerPool::get_dp_rank(Sequence* sequence) const {
  int32_t dp_rank;
  if (sequence->dp_rank() >= 0) {
    dp_rank = sequence->dp_rank();
  } else {
    dp_rank = get_manager_with_max_free_blocks();
    sequence->set_dp_rank(dp_rank);
  }
  return dp_rank;
}

void BlockManagerPool::deallocate(Request* request) {
  DCHECK(request != nullptr);
  for (auto& sequence : request->sequences()) {
    deallocate(sequence.get());
  }
}

void BlockManagerPool::deallocate(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  int32_t dp_rank = get_dp_rank(sequence);

  // The composite inserts the final completed blocks into the prefix cache
  // from inside deallocate (the context carries the token view + hash chain),
  // then releases every group -- SINGLE_RES included. A beam sequence skips
  // that final insert (bare context): its decode blocks carry rewritten-token
  // content that does not match the token hash, matching the allocate path.
  BlockManagerContext context =
      sequence->check_beam_search()
          ? make_bare_device_context(sequence, dp_rank)
          : make_device_context(sequence, dp_rank);
  composite_managers_[dp_rank]->deallocate(&context);
  sequence->reset();
}

std::vector<std::vector<BlockTransferInfo>>*
BlockManagerPool::get_swap_block_transfer_infos() {
  return &swap_block_transfer_infos_;
}

bool BlockManagerPool::allocate(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  return allocate(sequence, sequence->num_tokens());
}

// First scheduling of a sequence on the composite path: no per-group state has
// been materialized and no legacy flat blocks exist. This is the only point
// where a prefix match may run -- the composite has no mid-stream re-match
// (and therefore no block replacement) semantics.
bool BlockManagerPool::is_first_schedule(Sequence* sequence) const {
  return sequence->kv_state().groups().empty() &&
         sequence->kv_state().num_kv_blocks() == 0;
}

bool BlockManagerPool::allocate(Sequence* sequence, size_t num_tokens) {
  AUTO_COUNTER(allocate_blocks_latency_seconds);
  DCHECK(sequence != nullptr);
  int32_t dp_rank = get_dp_rank(sequence);

  const bool first_schedule = is_first_schedule(sequence);
  if (options_.enable_prefix_cache() && first_schedule) {
    // First prefill: restore any cached prefix before growing; growth appends
    // on top of the matched C1 blocks. The lazy flush for subsequent grows
    // (decode / chunked prefill) now happens inside the composite's allocate,
    // so the scheduler never drives a separate flush.
    composite_match_shared(sequence, dp_rank);
  }
  // Beam fork/COW adopts the scored source beam into the C1 group before the
  // composite grows it. A no-op for non-beam sequences and for the beam prefill
  // (no source blocks yet); fails only when the COW swap block is unavailable,
  // which an in-flight decode treats as a preempt-on-next signal.
  if (!composite_process_beam_search(sequence, dp_rank, num_tokens)) {
    return false;
  }
  // A beam sequence must never lazy-flush into the prefix cache: after the
  // per-beam swap/COW the block content no longer matches the token hash, so a
  // bare context (no token view / hash chain) keeps the composite's internal
  // insert from corrupting the cache. Prefix matching on first prefill still
  // ran above (read-only, against the original prompt tokens).
  BlockManagerContext context =
      sequence->check_beam_search()
          ? make_bare_device_context(sequence, dp_rank)
          : make_device_context(sequence, dp_rank);
  if (!composite_managers_[dp_rank]->allocate(&context, num_tokens)) {
    // Only a fresh sequence is fully unwound; an in-flight decode keeps its
    // existing blocks for the scheduler to preempt (matches the legacy path).
    if (first_schedule) {
      composite_managers_[dp_rank]->deallocate(&context);
      sequence->reset();
    }
    return false;
  }
  return true;
}

bool BlockManagerPool::allocate(Sequence* sequence,
                                size_t num_tokens,
                                size_t needed_copy_in_blocks_num) {
  LOG(FATAL)
      << "allocate(Sequence* sequence, size_t num_tokens, size_t "
         "needed_copy_in_blocks_num) is not implemented in BlockManagerPool.";
  return false;
}

bool BlockManagerPool::try_allocate(Sequence* sequence) {
  int32_t dp_rank = get_dp_rank(sequence);
  BlockManagerContext context = make_device_context(sequence, dp_rank);
  if (options_.enable_prefix_cache() && is_first_schedule(sequence)) {
    composite_managers_[dp_rank]->match_prefix_cache(
        &context, sequence->tokens(), &sequence->mm_data());
  }
  if (!composite_managers_[dp_rank]->allocate(&context,
                                              sequence->tokens().size())) {
    composite_managers_[dp_rank]->deallocate(&context);
    sequence->reset();
    return false;
  }
  // try_allocate reserves the whole prompt: mark every token as cached, the same
  // end-state the legacy path reaches via incr_kv_cache_tokens_num.
  sequence->kv_state().set_kv_cache_tokens_num(sequence->tokens().size());
  return true;
}

bool BlockManagerPool::composite_process_beam_search(Sequence* sequence,
                                                     int32_t dp_rank,
                                                     size_t num_tokens) {
  if (!sequence->check_beam_search()) {
    return true;
  }

  KVCacheState& kv_state = sequence->kv_state();
  const Slice<Block> src_blocks = kv_state.src_blocks();
  if (src_blocks.size() == 0) {
    return true;
  }

  // Phase-1 scope: beam fork/COW is defined only over the C1 incremental
  // attention group. DSV4 (no C1) and SWA reject beam upstream; this guards the
  // contract instead of silently writing the disconnected flat block table.
  CHECK(kv_state.group_state(CacheStateId::C1) != nullptr)
      << "beam search on the composite path requires a C1 attention group";

  // Copy-on-write the shared last block only when this step does NOT cross a
  // block boundary: the new token then overwrites the (shared) last block in
  // place, so a beam sharing it needs a private copy. When the step grows, the
  // new token lands in the fresh block the composite allocate appends below, so
  // the now-full last block stays safely shared (mirrors the legacy need_swap
  // gate, which is true only in the no-growth branch).
  const size_t block_size = static_cast<size_t>(options_.block_size());
  const size_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;
  const bool no_growth = num_blocks_needed <= src_blocks.size();

  if (no_growth && kv_state.need_swap()) {
    std::vector<Block> new_blocks =
        composite_managers_[dp_rank]->allocate_blocks(CacheStateId::C1,
                                                      /*num_blocks=*/1);
    if (new_blocks.empty()) {
      return false;
    }
    swap_block_transfer_infos_[dp_rank].emplace_back(src_blocks.back().id(),
                                                     new_blocks[0].id());
    kv_state.process_beam_search(new_blocks[0]);
  } else {
    kv_state.process_beam_search(std::nullopt);
  }
  return true;
}

void BlockManagerPool::allocate_shared(Sequence* sequence) {
  // only allocate shared blocks for prefill sequences
  if (!options_.enable_prefix_cache()) {
    return;
  }
  int32_t dp_rank = get_dp_rank(sequence);
  // Prefix match runs exactly once, on the sequence's first scheduling.
  // A mid-stream re-match (e.g. the chunked-prefill scheduler's periodic
  // allocate_shared) would have to replace blocks the sequence already
  // computed; the composite path deliberately has no such semantics.
  if (!is_first_schedule(sequence)) {
    return;
  }
  composite_match_shared(sequence, dp_rank);
}

void BlockManagerPool::composite_match_shared(Sequence* sequence,
                                              int32_t dp_rank) {
  BlockManagerContext context = make_device_context(sequence, dp_rank);
  CompositeMatchResult matched =
      composite_managers_[dp_rank]->match_prefix_cache(
          &context, sequence->tokens(), &sequence->mm_data());
  if (matched.matched_tokens == 0) {
    return;
  }

  size_t matched_tokens = matched.matched_tokens;
  const size_t total_tokens = sequence->num_tokens();
  // Whole-prompt cache hit: drop the last shared block so the forward pass has
  // at least one token to (re)compute. Mirrors KVCacheState::add_shared_kv_-
  // blocks, which pops the last block and rewinds the cached-token position.
  if (matched_tokens >= total_tokens) {
    CacheGroupState* c1 = sequence->kv_state().group_state(CacheStateId::C1);
    if (c1 != nullptr && !c1->blocks.empty()) {
      const size_t block_size = c1->blocks.back().size();
      c1->blocks.pop_back();
      c1->shared_blocks_num = c1->blocks.size();
      c1->prefix_cached_tokens = c1->blocks.size() * block_size;
      matched_tokens = c1->prefix_cached_tokens;
    }
  }
  sequence->kv_state().set_kv_cache_tokens_num(matched_tokens);
}

void BlockManagerPool::cache(Sequence* sequence) {
  // The composite inserts completed blocks into its prefix caches internally
  // from allocate/deallocate; the scheduler no longer drives a separate cache
  // step.
}

void BlockManagerPool::cache(Sequence* sequence, size_t num_tokens) {
  // In-batch prefix cache (upstream #1743): publishes the first `num_tokens`
  // worth of completed blocks mid-batch so later requests in the same batch can
  // share them. On the composite-only path the composite owns prefix-cache
  // insertion internally (at allocate/deallocate) and has no on-demand bounded
  // publish entry yet, so this is a no-op: same-batch sharing falls back to the
  // post-deallocate prefix cache. Correctness is preserved; only the in-batch
  // sharing optimization is disabled.
  // TODO(block-refactor): expose a num_tokens-bounded insert on the composite
  // (insert_committed_blocks bounded by num_tokens) to restore in-batch
  // prefix-cache publishing.
}

float BlockManagerPool::get_gpu_cache_usage_perc() const {
  const size_t count = manager_count();
  if (count == 0) {
    return 0.0f;
  }
  float perc = 0.0;
  for (size_t i = 0; i < count; ++i) {
    perc += composite_managers_[i]->kv_cache_utilization();
  }
  return perc / count;
}

uint32_t BlockManagerPool::num_blocks() const { return options_.num_blocks(); }

int32_t BlockManagerPool::block_size() const { return options_.block_size(); }

std::vector<size_t> BlockManagerPool::num_blocks_in_prefix_cache() const {
  std::vector<size_t> num_blocks_in_prefix_cache(manager_count());
  if (!options_.enable_prefix_cache()) {
    return num_blocks_in_prefix_cache;
  }
  for (size_t dp_rank = 0; dp_rank < manager_count(); ++dp_rank) {
    num_blocks_in_prefix_cache[dp_rank] =
        composite_managers_[dp_rank]->num_blocks_in_prefix_cache();
  }
  return num_blocks_in_prefix_cache;
}

std::vector<size_t> BlockManagerPool::num_free_blocks() const {
  std::vector<size_t> num_free_blocks(manager_count());
  for (size_t dp_rank = 0; dp_rank < manager_count(); ++dp_rank) {
    num_free_blocks[dp_rank] =
        composite_managers_[dp_rank]->num_free_blocks();
  }
  return num_free_blocks;
}

std::vector<size_t> BlockManagerPool::num_used_blocks() const {
  std::vector<size_t> num_used_blocks(manager_count());
  for (size_t dp_rank = 0; dp_rank < manager_count(); ++dp_rank) {
    num_used_blocks[dp_rank] =
        composite_managers_[dp_rank]->num_used_blocks();
  }
  return num_used_blocks;
}

double BlockManagerPool::kv_cache_utilization() const {
  int32_t dp_rank = get_manager_with_max_free_blocks();
  return composite_managers_[dp_rank]->kv_cache_utilization();
}

// currently use only for profile, which not need prefix cache.
// If more often used in the future, can be integrated into deallocate function.
void BlockManagerPool::deallocate_without_cache(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  int32_t dp_rank = get_dp_rank(sequence);

  // Release every group without inserting into the prefix cache: a bare context
  // carries no token view / hash chain, so the composite's internal insert is
  // skipped and uncomputed blocks are never cached.
  BlockManagerContext context = make_bare_device_context(sequence, dp_rank);
  composite_managers_[dp_rank]->deallocate(&context);
  sequence->reset();
}

void BlockManagerPool::reserve_xtensor_padding_blocks() {
  if (!options_.enable_xtensor()) {
    return;
  }

  // xtensor is folded into the composite as a C1 leaf; each composite manager
  // reserves the padding block on its own C1 group (no-op when that leaf is not
  // xtensor). The leaf type stays hidden behind the composite.
  for (auto& manager : composite_managers_) {
    manager->reserve_xtensor_padding_blocks();
  }

  // Start prealloc thread once (PageAllocator is shared by all managers)
  PageAllocator::get_instance().start_prealloc_thread();
}

}  // namespace xllm
