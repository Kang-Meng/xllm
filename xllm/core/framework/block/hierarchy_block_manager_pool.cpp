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

#include "hierarchy_block_manager_pool.h"

#include <folly/executors/InlineExecutor.h>

#include <algorithm>
#include <atomic>
#include <exception>
#include <iomanip>
#include <map>
#include <optional>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "block_manager_impl.h"
#include "composite_block_manager.h"
#include "concurrent_block_manager_impl.h"
#include "core/framework/block/block_utils.h"
#include "core/framework/config/scheduler_config.h"
#include "linear_state_block_manager.h"
#include "sliding_window_block_manager.h"
#include "util/env_var.h"

namespace xllm {

namespace {

void finalize_host_blocks(bool publish,
                          std::vector<Block> host_blocks,
                          const std::vector<BlockType>& block_types,
                          CompositeBlockManager* host_manager) {
  CHECK(host_manager);
  CHECK_EQ(host_blocks.size(), block_types.size());
  std::unordered_map<BlockType, std::vector<Block>> blocks_by_type;
  for (size_t i = 0; i < host_blocks.size(); ++i) {
    blocks_by_type[block_types[i]].emplace_back(std::move(host_blocks[i]));
  }

  for (auto& [type, blocks] : blocks_by_type) {
    VLOG(1) << "[HostCache][OffloadComplete] type="
            << static_cast<int32_t>(type) << " blocks=" << blocks.size()
            << " publish=" << publish;
    if (publish) {
      host_manager->cache_blocks(type, blocks);
    }
  }
}

// Wrap a leaf in the concurrency adapter when the D2H offload callback frees
// blocks off-thread. Host-offload leaves always need this wrap.
std::unique_ptr<BlockManager> wrap_for_offload(std::unique_ptr<BlockManager> l,
                                               const BlockManager::Options& o) {
  if (o.enable_disagg_pd() || o.enable_kvcache_store() ||
      o.enable_host_offload()) {
    return std::make_unique<ConcurrentBlockManagerImpl>(std::move(l));
  }
  return l;
}

bool has_valid_swa_window(size_t tokens,
                          size_t block_size,
                          size_t blocks_per_window,
                          const Slice<Block>& blocks) {
  size_t index = tokens / block_size;
  if (index < blocks_per_window) {
    return false;
  }
  for (size_t i = 0; i < blocks_per_window; ++i) {
    --index;
    if (index >= blocks.size() || !blocks[index].is_valid()) {
      return false;
    }
  }
  return true;
}

BlockType checkpoint_block_type(const CompositeBlockManager::LeafMap& leaves) {
  if (leaves.contains(BlockType::C128)) {
    return BlockType::C128;
  }
  if (leaves.contains(BlockType::C2)) {
    return BlockType::C2;
  }
  return BlockType::C1;
}

size_t prefetch_unit_size(CompositeBlockManager::LeafCombination combination,
                          const CompositeBlockManager::LeafMap& leaves) {
  // FLAT_KV_LINEAR builds one Store unit per KV block. The LINEAR checkpoint
  // of that same block is the optional hit. Using the LINEAR stride here
  // would skip every KV block inside a larger checkpoint.
  BlockType unit_type = BlockType::KV;
  if (combination == CompositeBlockManager::LeafCombination::SWA_COMPRESSED) {
    unit_type = checkpoint_block_type(leaves);
  }
  const auto unit_leaf = leaves.find(unit_type);
  CHECK(unit_leaf != leaves.end());
  CHECK_GT(unit_leaf->second.leaf->block_size(), 0u);
  return unit_leaf->second.leaf->block_size();
}

bool append_prefetch_transfer(PrefetchUnit* unit,
                              BlockType type,
                              size_t block_index,
                              const KVCacheState& host_state) {
  CHECK(unit != nullptr);
  const Slice<Block> blocks = host_state.blocks(type);
  if (block_index >= blocks.size() || !blocks[block_index].is_valid()) {
    return false;
  }
  std::vector<BlockTransferInfo>& transfers = is_prefetch_gate_block_type(type)
                                                  ? unit->gated_blocks
                                                  : unit->non_gated_blocks;
  transfers.emplace_back(
      /*src_id=*/-1,
      /*dst_id=*/blocks[block_index].id(),
      blocks[block_index].get_immutable_hash_value(),
      TransferType::G2H,
      type);
  return true;
}

size_t valid_block_size(const Slice<Block>& blocks) {
  for (const Block& block : blocks) {
    if (block.is_valid()) {
      return block.size();
    }
  }
  return 0;
}

StoragePrefetchRequest build_flat_kv_prefetch_request(
    const Sequence* sequence) {
  CHECK(sequence != nullptr);
  const KVCacheState& host_state = sequence->host_kv_state();
  const Slice<Block> kv_blocks = host_state.blocks(BlockType::KV);
  const size_t block_size = valid_block_size(kv_blocks);
  if (block_size == 0 || kv_blocks.empty()) {
    return {};
  }
  const size_t base_units = host_state.shared_tokens_num() / block_size;
  if (base_units >= kv_blocks.size()) {
    return {};
  }

  StoragePrefetchRequest request;
  request.units.reserve(kv_blocks.size() - base_units);
  for (size_t unit = base_units; unit < kv_blocks.size(); ++unit) {
    PrefetchUnit prefetch_unit;
    if (!append_prefetch_transfer(
            &prefetch_unit, BlockType::KV, unit, host_state)) {
      break;
    }
    request.units.emplace_back(std::move(prefetch_unit));
  }
  return request;
}

StoragePrefetchRequest build_flat_kv_linear_prefetch_request(
    const Sequence* sequence) {
  CHECK(sequence != nullptr);
  const KVCacheState& host_state = sequence->host_kv_state();
  const Slice<Block> kv_blocks = host_state.blocks(BlockType::KV);
  const Slice<Block> linear_blocks = host_state.blocks(BlockType::LINEAR);
  const size_t kv_block_size = valid_block_size(kv_blocks);
  if (kv_block_size == 0 || linear_blocks.empty()) {
    return {};
  }
  const size_t base_units = host_state.shared_tokens_num() / kv_block_size;
  if (base_units >= kv_blocks.size()) {
    return {};
  }

  StoragePrefetchRequest request;
  request.units.reserve(kv_blocks.size() - base_units);
  for (size_t unit = base_units; unit < kv_blocks.size(); ++unit) {
    PrefetchUnit prefetch_unit;
    prefetch_unit.has_non_gated = true;
    const bool available = append_prefetch_transfer(
        &prefetch_unit, BlockType::KV, unit, host_state);
    if (!available) {
      break;
    }
    if (unit < linear_blocks.size() && linear_blocks[unit].is_valid()) {
      append_prefetch_transfer(
          &prefetch_unit, BlockType::LINEAR, unit, host_state);
    }
    request.units.emplace_back(std::move(prefetch_unit));
  }
  return request;
}

StoragePrefetchRequest build_swa_compressed_prefetch_request(
    const Sequence* sequence,
    const CompositeBlockManager::LeafMap& leaves) {
  CHECK(sequence != nullptr);
  const KVCacheState& host_state = sequence->host_kv_state();
  const BlockType checkpoint_type = checkpoint_block_type(leaves);
  const BlockType companion_type =
      checkpoint_type == BlockType::C128 ? BlockType::C4 : BlockType::C1;
  const Slice<Block> checkpoint_blocks = host_state.blocks(checkpoint_type);
  const auto& swa_options = leaves.at(BlockType::SWA).leaf->options();
  const size_t swa_block_size = swa_options.block_size();
  const size_t swa_blocks_per_window = swa_options.swa_blocks_per_seq();
  const size_t unit_size = leaves.at(checkpoint_type).leaf->block_size();
  const auto companion_leaf = leaves.find(companion_type);
  const bool has_companion =
      companion_type != checkpoint_type && companion_leaf != leaves.end();
  const size_t companion_block_size =
      has_companion ? companion_leaf->second.leaf->block_size() : unit_size;
  CHECK_GT(swa_block_size, 0u);
  CHECK_GT(companion_block_size, 0u);
  CHECK_GT(unit_size, 0u);
  CHECK_EQ(unit_size % companion_block_size, 0u);
  CHECK_EQ(unit_size % swa_block_size, 0u);
  const size_t base_units = host_state.shared_tokens_num() / unit_size;
  if (base_units >= checkpoint_blocks.size()) {
    return {};
  }

  StoragePrefetchRequest request;
  request.units.reserve(checkpoint_blocks.size() - base_units);
  for (size_t unit = base_units; unit < checkpoint_blocks.size(); ++unit) {
    PrefetchUnit prefetch_unit;
    prefetch_unit.has_non_gated = true;
    const size_t swa_end = (unit + 1) * unit_size / swa_block_size;
    const size_t swa_begin =
        swa_end - swa_tail_blocks(swa_end, swa_blocks_per_window);
    // V4.1 can checkpoint before filling its entire SWA window.
    bool complete_window =
        checkpoint_type != BlockType::C128 || swa_end >= swa_blocks_per_window;
    for (size_t block_index = swa_begin; block_index < swa_end; ++block_index) {
      complete_window =
          append_prefetch_transfer(
              &prefetch_unit, BlockType::SWA, block_index, host_state) &&
          complete_window;
    }
    if (!complete_window) {
      prefetch_unit.non_gated_blocks.clear();
    }

    bool gate_available = true;
    if (has_companion) {
      const size_t companion_begin = unit * unit_size / companion_block_size;
      const size_t companion_end =
          (unit + 1) * unit_size / companion_block_size;
      for (size_t block_index = companion_begin; block_index < companion_end;
           ++block_index) {
        const bool available = append_prefetch_transfer(
            &prefetch_unit, companion_type, block_index, host_state);
        gate_available = gate_available && available;
      }
    }
    const bool checkpoint_available = append_prefetch_transfer(
        &prefetch_unit, checkpoint_type, unit, host_state);
    gate_available = gate_available && checkpoint_available;
    if (!gate_available) {
      break;
    }
    request.units.emplace_back(std::move(prefetch_unit));
  }
  return request;
}

StoragePrefetchRequest build_prefetch_request(
    const Sequence* sequence,
    CompositeBlockManager::LeafCombination combination,
    const CompositeBlockManager::LeafMap& leaves) {
  switch (combination) {
    case CompositeBlockManager::LeafCombination::FLAT_KV:
      return build_flat_kv_prefetch_request(sequence);
    case CompositeBlockManager::LeafCombination::FLAT_KV_LINEAR:
      return build_flat_kv_linear_prefetch_request(sequence);
    case CompositeBlockManager::LeafCombination::SWA_COMPRESSED:
      return build_swa_compressed_prefetch_request(sequence, leaves);
    case CompositeBlockManager::LeafCombination::UNSUPPORTED:
      return {};
  }
  return {};
}

void finalize_prefetch(Sequence* sequence,
                       CompositeBlockManager* host_manager,
                       size_t final_tokens,
                       bool publish) {
  CHECK(sequence != nullptr);
  CHECK(host_manager);
  const auto& leaves = host_manager->leaf_entries();
  KVCacheState& host_state = sequence->host_kv_state();
  if (!publish) {
    host_state.reset();
    return;
  }

  // Each Host leaf releases overflow aliases, publishes the retained range and
  // rebuilds its own cursor. The pool only commits the sequence-level token
  // boundary after all leaves have completed that transaction.
  host_manager->trim_prefetch_blocks(sequence, final_tokens);
  host_state.set_kv_cache_tokens_num(final_tokens);
  host_state.set_prefix_cache_matched();
}

size_t finalize_prefetch_tokens(
    CompositeBlockManager::LeafCombination combination,
    const CompositeBlockManager::LeafMap& leaves,
    size_t base_tokens,
    const PrefetchSummary& summary,
    const Slice<Block>& linear_blocks) {
  const size_t unit_size = prefetch_unit_size(combination, leaves);
  const uint8_t rank_count = summary.rank_count;
  size_t gated_units = 0;
  while (gated_units < summary.gated_hits.size() &&
         summary.gated_hits[gated_units] == rank_count) {
    ++gated_units;
  }
  if (combination == CompositeBlockManager::LeafCombination::FLAT_KV) {
    return base_tokens + gated_units * unit_size;
  }
  if (combination == CompositeBlockManager::LeafCombination::FLAT_KV_LINEAR) {
    if (unit_size == 0) {
      return base_tokens;
    }
    const size_t base_blocks = base_tokens / unit_size;
    const size_t kept_blocks =
        last_reusable_linear_checkpoint(linear_blocks,
                                        base_blocks,
                                        base_blocks + gated_units,
                                        summary.non_gated_hits,
                                        rank_count);
    return kept_blocks * unit_size;
  }

  // SWA publishes the deepest fully hit optional unit inside the contiguous
  // KV prefix. A missing checkpoint drops that unit.
  size_t deepest_optional_unit = 0;
  const size_t inspect = std::min(gated_units, summary.non_gated_hits.size());
  for (size_t index = 0; index < inspect; ++index) {
    if (summary.non_gated_hits[index] == rank_count) {
      deepest_optional_unit = index + 1;
    }
  }
  if (deepest_optional_unit == 0) {
    return base_tokens;
  }
  return base_tokens + deepest_optional_unit * unit_size;
}

// One line per prefetched request; see XLLM_STORE_PREFETCH_STATS. Per-rank
// read volume and tail probes are logged by each worker with the same
// request_id. prefetch_latency includes the per-batch replica-tier queries
// that the measurement itself adds; see tier_query_time in the worker lines.
void log_prefetch_summary(const Request& request,
                          const Sequence& sequence,
                          int32_t dp_rank,
                          size_t host_hit_tokens,
                          size_t target_tokens,
                          size_t fetched_tokens,
                          const PrefetchSummary& summary) {
  LOG(INFO) << "[StorePrefetch] request_id: " << request.request_id()
            << ", dp_rank: " << dp_rank
            << ", prompt_tokens: " << sequence.num_prompt_tokens()
            << ", host_hit_tokens: " << host_hit_tokens
            << ", target_tokens: " << target_tokens
            << ", fetched_tokens: " << fetched_tokens << ", stop_reason: "
            << prefetch_stop_reason_name(summary.stop_reason)
            << ", prefetch_latency: " << std::fixed << std::setprecision(1)
            << summary.latency_ms << "ms";
}

uint8_t prefetch_hit_count(size_t position,
                           size_t base_blocks,
                           const std::vector<uint8_t>& hit_counts) {
  if (position < base_blocks) {
    return 0;
  }
  const size_t unit = position - base_blocks;
  if (unit >= hit_counts.size()) {
    return 0;
  }
  return hit_counts[unit];
}

}  // namespace

// A hit past the KV common prefix has overwritten that host page. Incomplete
// prefetch tails stay invalid at allocate time, so they never enter the
// overwrite set. Indexes are absolute block positions. hit_counts covers the
// prefetched tail and starts at base_blocks. Positions below base_blocks are
// the existing host prefix and are not inspected.
size_t last_reusable_linear_checkpoint(const Slice<Block>& linear_blocks,
                                       size_t base_blocks,
                                       size_t kv_common_blocks,
                                       const std::vector<uint8_t>& hit_counts,
                                       uint8_t rank_count) {
  if (rank_count == 0 || linear_blocks.empty()) {
    return base_blocks;
  }
  const size_t common_end = std::min(kv_common_blocks, linear_blocks.size());
  const size_t selectable_begin = std::min(base_blocks, common_end);
  std::unordered_set<int32_t> overwritten_ids;
  for (size_t index = linear_blocks.size(); index > common_end; --index) {
    const size_t position = index - 1;
    const Block& block = linear_blocks[position];
    if (!block.is_valid()) {
      continue;
    }
    if (prefetch_hit_count(position, base_blocks, hit_counts) == 0) {
      continue;
    }
    overwritten_ids.insert(block.id());
  }
  for (size_t index = common_end; index > selectable_begin; --index) {
    const size_t position = index - 1;
    const Block& block = linear_blocks[position];
    if (!block.is_valid()) {
      continue;
    }
    if (prefetch_hit_count(position, base_blocks, hit_counts) < rank_count) {
      continue;
    }
    if (overwritten_ids.find(block.id()) != overwritten_ids.end()) {
      continue;
    }
    return position + 1;
  }
  return base_blocks;
}

HierarchyBlockManagerPool::HierarchyBlockManagerPool(
    const BlockManagerPool::Options& options,
    Engine* engine,
    int32_t dp_size)
    : engine_(engine), BlockManagerPool(options, dp_size) {
  CHECK(dp_size > 0) << "dp_size must be greater than 0";
  host_block_managers_.reserve(dp_size);

  for (int32_t i = 0; i < dp_size; ++i) {
    CompositeBlockManager::LeafMap per_type;
    auto* composite =
        static_cast<CompositeBlockManager*>(block_managers_[i].get());
    const CompositeBlockManager::LeafCombination combination =
        composite->leaf_combination();
    switch (combination) {
      case CompositeBlockManager::LeafCombination::FLAT_KV:
      case CompositeBlockManager::LeafCombination::FLAT_KV_LINEAR:
      case CompositeBlockManager::LeafCombination::SWA_COMPRESSED:
        break;
      case CompositeBlockManager::LeafCombination::UNSUPPORTED: {
        std::ostringstream leaf_summary;
        for (const auto& [type, entry] : composite->leaf_entries()) {
          leaf_summary << static_cast<int32_t>(type)
                       << ":prefix=" << entry.supports_prefix_cache << ",";
        }
        CHECK(!options_.enable_prefix_cache())
            << "Unsupported prefix-cache leaf combination: instance_is_decode="
            << options_.instance_is_decode() << ", leaves=["
            << leaf_summary.str() << "]";
        break;
      }
      default:
        LOG(FATAL) << "HierarchyBlockManagerPool supports only FLAT_KV, "
                      "FLAT_KV_LINEAR and SWA_COMPRESSED cache layouts; got "
                   << static_cast<int32_t>(combination);
    }
    const CompositeBlockManager::LeafMap& device_leaves =
        composite->leaf_entries();
    std::map<BlockType, uint32_t> host_capacities =
        options_.host_num_blocks_by_type();
    if (host_capacities.empty() && options_.host_num_blocks() > 0) {
      host_capacities.emplace(BlockType::KV, options_.host_num_blocks());
    }
    for (const auto& [type, num_blocks] : host_capacities) {
      const auto device_leaf_it = device_leaves.find(type);
      if (device_leaf_it == device_leaves.end() || num_blocks == 0) {
        continue;
      }
      BlockManager::Options host_options =
          device_leaf_it->second.leaf->options();
      host_options.num_blocks(num_blocks)
          .enable_disagg_pd(options_.enable_disagg_pd())
          .enable_kvcache_store(options_.enable_kvcache_store())
          .enable_host_offload(options_.enable_host_offload())
          .prefix_cache_publish_mode(PrefixCachePublishMode::EXPLICIT);
      if (type == BlockType::LINEAR && host_options.instance_is_decode() &&
          options_.enable_host_offload()) {
        // Decode HBM deliberately does not probe LINEAR prefix cache, so its
        // block size is max(KV block, chunk). The Host copy is the restore
        // source and must publish on the KV block grid.
        host_options.enable_prefix_cache(true);
        host_options.block_size(options_.block_size());
      }

      std::unique_ptr<BlockManager> leaf;
      if (type == BlockType::SWA) {
        leaf = std::make_unique<SlidingWindowBlockManager>(host_options);
      } else if (type == BlockType::LINEAR) {
        host_options.num_blocks(num_blocks);
        host_options.block_type(BlockType::LINEAR);
        host_options.prefetch_batch_size(options_.prefetch_batch_size());
        host_options.prefetch_valid_blocks_per_batch(
            options_.prefetch_valid_blocks_per_batch());
        host_options.preserve_decode_checkpoint(
            host_options.instance_is_decode() &&
            options_.enable_host_offload());
        leaf = std::make_unique<LinearStateBlockManager>(host_options);
      } else {
        leaf = std::make_unique<BlockManagerImpl>(host_options);
      }
      leaf = wrap_for_offload(std::move(leaf), host_options);
      per_type.emplace(
          type,
          CompositeBlockManager::LeafEntry{
              std::move(leaf),
              /*participates_in_admission=*/false,
              /*supports_prefix_cache=*/host_options.enable_prefix_cache()});
    }
    if (per_type.empty()) {
      host_block_managers_.emplace_back(nullptr);
    } else {
      host_block_managers_.emplace_back(std::make_unique<CompositeBlockManager>(
          std::move(per_type),
          composite->options(),
          CompositeBlockManager::PrefixCachePublishMode::EXPLICIT));
    }
  }

  load_block_transfer_infos_.resize(host_block_managers_.size());
  offload_block_pair_queues_.resize(host_block_managers_.size());
}

void HierarchyBlockManagerPool::reset_prefix_cache() {
  BlockManagerPool::reset_prefix_cache();
  if (!options_.enable_prefix_cache()) {
    return;
  }
  for (auto& host_manager : host_block_managers_) {
    if (host_manager != nullptr) {
      host_manager->reset_prefix_cache();
    }
  }
}

std::vector<size_t> HierarchyBlockManagerPool::num_blocks_in_prefix_cache()
    const {
  std::vector<size_t> counts = BlockManagerPool::num_blocks_in_prefix_cache();
  for (size_t dp_rank = 0; dp_rank < host_block_managers_.size(); ++dp_rank) {
    if (dp_rank >= counts.size() || host_block_managers_[dp_rank] == nullptr) {
      continue;
    }
    counts[dp_rank] +=
        host_block_managers_[dp_rank]->num_blocks_in_prefix_cache();
  }
  return counts;
}

void HierarchyBlockManagerPool::deallocate(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  int32_t dp_rank = BlockManagerPool::get_dp_rank(sequence);
  auto* composite =
      static_cast<CompositeBlockManager*>(block_managers_[dp_rank].get());
  composite->cache_full_blocks_for_sequence(sequence);

  collect_offload_pairs(sequence);

  // Release the host blocks still held by the sequence. Decode LINEAR source
  // and destination blocks moved into the offload queue remain owned there
  // until D2H completes.
  auto* host_manager = host_block_managers_[dp_rank].get();
  CHECK(host_manager);
  host_manager->deallocate_for_sequence(sequence, sequence->host_kv_state());

  // Release device blocks via the composite (includes prefix cache flush).
  composite->deallocate_for_sequence(sequence);
  sequence->reset();
}

void HierarchyBlockManagerPool::collect_offload_pairs(Sequence* sequence) {
  const int32_t dp_rank = sequence->dp_rank();
  const auto* host_manager = host_block_managers_[dp_rank].get();
  CHECK(host_manager);

  KVCacheState& hbm_state = sequence->kv_state();
  KVCacheState& host_state = sequence->host_kv_state();
  // The first allocate can see a longer HBM prefix while the previous D2H is
  // still in flight. Collecting that prefix again would offload the same span.
  if (hbm_state.shared_tokens_num() > host_state.shared_tokens_num() &&
      hbm_state.kv_cache_tokens_num() == hbm_state.shared_tokens_num()) {
    return;
  }
  bool queued_offload = false;
  const size_t host_cached_tokens = host_state.kv_cache_tokens_num();
  size_t completed_tokens = hbm_state.kv_cache_tokens_num();
  size_t cache_unit_size = 0;
  const auto checkpoint_entry = host_manager->leaf_entries().find(
      checkpoint_block_type(host_manager->leaf_entries()));
  if (checkpoint_entry != host_manager->leaf_entries().end()) {
    cache_unit_size = checkpoint_entry->second.leaf->block_size();
    CHECK_GT(cache_unit_size, 0u);
    for (const auto& [type, entry] : host_manager->leaf_entries()) {
      const size_t block_size = entry.leaf->block_size();
      const BlockHasherType hasher_type = entry.leaf->options().hasher_type();
      completed_tokens =
          std::min(completed_tokens,
                   num_hash_blocks(hasher_type,
                                   sequence->hash_tokens(hasher_type).size(),
                                   block_size) *
                       block_size);
    }
    completed_tokens = completed_tokens / cache_unit_size * cache_unit_size;
  }
  for (const auto& [type, entry] : host_manager->leaf_entries()) {
    std::vector<Block>* hbm_blocks = hbm_state.mutable_blocks(type);
    std::vector<Block>* host_blocks = host_state.mutable_blocks(type);
    const size_t block_size = entry.leaf->block_size();
    CHECK_GT(block_size, 0u);

    if (type == BlockType::LINEAR) {
      VLOG(1) << "[HostCache][LinearOffload] seq=" << sequence->seq_id()
              << " stage=" << static_cast<int>(sequence->stage())
              << " hbm_tokens=" << hbm_state.kv_cache_tokens_num()
              << " host_tokens=" << host_state.kv_cache_tokens_num()
              << " hbm_blocks=" << hbm_blocks->size()
              << " host_blocks=" << host_blocks->size();
      auto two_valid_blocks = [](const std::vector<Block>* blocks) {
        return blocks->size() == 2 && blocks->front().is_valid() &&
               blocks->back().is_valid();
      };
      if (!two_valid_blocks(hbm_blocks) || !two_valid_blocks(host_blocks)) {
        continue;
      }

      auto pair = std::make_shared<OffloadBlockPair>();
      pair->block_type = BlockType::LINEAR;
      size_t checkpoint_index = 0;
      // Offload covers tokens already in the HBM cache. Host may already be
      // ahead from an earlier pair, so the boundary is the HBM cursor.
      const size_t cached_tokens = hbm_state.kv_cache_tokens_num();
      if (sequence->is_prefill_stage()) {
        // Prefill copies the completed HBM front and keeps both HBM blocks
        // valid. Host moves the front into the pair, so that slot becomes
        // invalid while both vectors stay length 2.
        if (cached_tokens < block_size) {
          continue;
        }
        checkpoint_index = cached_tokens / block_size - 1;
        pair->src = hbm_blocks->front();
        pair->dst = std::move(host_blocks->front());
        pair->checkpoint_row = 0;
      } else {
        // Decode with host cache keeps the first block of the crossed
        // checkpoint through allocate. collect runs after that allocate and
        // moves the preserved front slot. Each vector keeps the live block.
        const std::optional<size_t> crossed_row =
            last_decode_step_checkpoint_row(*sequence, block_size);
        if (!crossed_row.has_value() || cached_tokens < block_size) {
          continue;
        }
        checkpoint_index = cached_tokens / block_size - 1;
        pair->src = std::move(hbm_blocks->front());
        pair->dst = std::move(host_blocks->front());
        pair->checkpoint_row = crossed_row.value();
        hbm_blocks->erase(hbm_blocks->begin());
        host_blocks->erase(host_blocks->begin());
      }

      // Stamp the sequence chain before enqueue. The D2H callback has no
      // sequence and reads this value from the destination block.
      const std::optional<const XXH3Key*> checkpoint_hash =
          sequence->block_hash_at(entry.leaf->options().hasher_type(),
                                  checkpoint_index,
                                  static_cast<uint32_t>(block_size));
      if (!checkpoint_hash.has_value()) {
        continue;
      }
      pair->dst.set_hash_value((*checkpoint_hash)->data);

      VLOG(1) << "[HostCache][LinearOffloadEnqueue] seq=" << sequence->seq_id()
              << " src=" << pair->src.id() << " dst=" << pair->dst.id()
              << " row=" << pair->checkpoint_row
              << " cached_index=" << checkpoint_index
              << " hash=" << (*checkpoint_hash)->to_hex();
      offload_block_pair_queues_[dp_rank].enqueue(std::move(pair));
      queued_offload = true;
      continue;
    }

    const size_t unit_size = cache_unit_size > 0 ? cache_unit_size : block_size;
    CHECK_EQ(unit_size % block_size, 0u);
    // The Host cursor may be inside a unit after an earlier pair was queued.
    if (completed_tokens / unit_size <= host_cached_tokens / unit_size) {
      continue;
    }
    const size_t first_block =
        (host_cached_tokens / unit_size) * (unit_size / block_size);
    size_t completed_blocks = completed_tokens / block_size;
    const BlockHasherType hasher_type = entry.leaf->options().hasher_type();
    // completed_tokens is the HBM cached-token cursor. MTP needs one token
    // past the block, so the last closed block is hashable only when that
    // token is already in the sequence.
    if (block_hash_lookahead(hasher_type) > 0) {
      completed_blocks =
          std::min(completed_blocks,
                   num_hash_blocks(hasher_type,
                                   sequence->hash_tokens(hasher_type).size(),
                                   block_size));
    }
    sequence->update_block_hashes(static_cast<uint32_t>(block_size),
                                  hasher_type);
    completed_blocks = std::min(
        completed_blocks,
        sequence->block_hashes(static_cast<uint32_t>(block_size)).size());
    const size_t comparable_blocks =
        std::min({hbm_blocks->size(), host_blocks->size(), completed_blocks});
    const size_t blocks_per_unit = unit_size / block_size;
    const size_t placeholders_per_unit =
        type == BlockType::SWA && cache_unit_size > 0
            ? blocks_per_unit -
                  swa_tail_blocks(blocks_per_unit,
                                  entry.leaf->options().swa_blocks_per_seq())
            : 0;
    for (size_t i = first_block; i < comparable_blocks; ++i) {
      if (placeholders_per_unit > 0 &&
          i % blocks_per_unit < placeholders_per_unit) {
        continue;
      }
      Block& hbm_block = (*hbm_blocks)[i];
      Block& host_block = (*host_blocks)[i];
      // Prefix-capable HBM leaves are held by both the sequence and the device
      // prefix cache. Decode SWA deliberately skips prefix insertion, so its
      // completed block is sequence-only until this offload pair retains it.
      // Host is the sequence's unfilled destination block in both cases.
      const uint32_t expected_hbm_refs = entry.supports_prefix_cache ? 2 : 1;
      if (hbm_block.ref_count() != expected_hbm_refs ||
          host_block.ref_count() != 1) {
        continue;
      }
      // Stamp before enqueue. The HBM block hash is not the chain hash.
      const std::optional<const XXH3Key*> block_hash = sequence->block_hash_at(
          hasher_type, i, static_cast<uint32_t>(block_size));
      if (!block_hash.has_value()) {
        break;
      }
      host_block.set_hash_value((*block_hash)->data);
      VLOG(1) << "[HostCache][OffloadHash] seq=" << sequence->seq_id()
              << " type=" << static_cast<int32_t>(type)
              << " src=" << hbm_block.id() << " dst=" << host_block.id()
              << " cached_index=" << i << " hash=" << (*block_hash)->to_hex();
      auto pair = std::make_shared<OffloadBlockPair>(
          OffloadBlockPair{/*src=*/hbm_block,
                           /*dst=*/std::move(host_block),
                           /*block_type=*/type,
                           /*checkpoint_row=*/0});
      offload_block_pair_queues_[dp_rank].enqueue(std::move(pair));
      queued_offload = true;
    }
  }
  if (queued_offload) {
    // Host advances only after this round has produced a concrete D2H pair.
    // The queue owns the destination block until transfer completion, so the
    // cursor represents a reserved logical checkpoint rather than a published
    // prefix-cache entry.
    host_state.set_kv_cache_tokens_num(hbm_state.kv_cache_tokens_num());
  }
}

HierarchyBlockManagerPool::~HierarchyBlockManagerPool() {
  CHECK_EQ(prefetching_requests_.load(std::memory_order_acquire), 0u)
      << "HierarchyBlockManagerPool destroyed with pending prefetch callbacks";
}

void HierarchyBlockManagerPool::trim_host_cache(
    Sequence* sequence,
    const HostCacheRestorePoint& selected_restore) {
  CHECK(sequence != nullptr);
  const int32_t dp_rank = BlockManagerPool::get_dp_rank(sequence);
  const size_t restore_tokens = selected_restore.restore_target_tokens;
  KVCacheState& hbm_state = sequence->kv_state();
  KVCacheState& host_state = sequence->host_kv_state();
  const size_t matched_tokens = sequence->kv_cache_tokens_num();
  CHECK_LE(hbm_state.kv_cache_tokens_num(), restore_tokens);
  CHECK_LE(restore_tokens, matched_tokens);

  const auto* composite =
      static_cast<const CompositeBlockManager*>(block_managers_[dp_rank].get());
  for (const auto& [type, entry] : composite->leaf_entries()) {
    if (type == BlockType::EMBEDDING || type == BlockType::LINEAR) {
      continue;
    }
    const size_t block_size = entry.leaf->block_size();
    CHECK_GT(block_size, 0u);
    const size_t keep = restore_tokens / block_size;
    host_state.set_num_cached_blocks(
        type, std::min(host_state.num_cached_blocks(type), keep));
    const Slice<Block> current = hbm_state.blocks(type);
    if (current.size() <= keep) {
      continue;
    }

    const size_t shared = std::min(hbm_state.shared_blocks_num(type), keep);
    const size_t cached = std::min(hbm_state.num_cached_blocks(type), keep);
    std::vector<Block> blocks = std::move(*hbm_state.mutable_blocks(type));
    blocks.resize(keep);
    if (blocks.empty()) {
      hbm_state.erase_blocks(type);
    } else {
      hbm_state.replace_composite_blocks(
          type, std::move(blocks), shared, cached);
    }
  }

  host_state.set_kv_cache_tokens_num(
      std::min(host_state.kv_cache_tokens_num(), restore_tokens));
  sequence->set_host_cache_restore(restore_tokens, selected_restore.copy_units);
}

bool HierarchyBlockManagerPool::try_allocate(Sequence* sequence) {
  CHECK(sequence != nullptr);
  if (!BlockManagerPool::try_allocate(sequence)) {
    return false;
  }

  const int32_t dp_rank = BlockManagerPool::get_dp_rank(sequence);
  auto* host_manager = host_block_managers_[dp_rank].get();
  CHECK(host_manager);
  KVCacheState& host_state = sequence->host_kv_state();
  const size_t num_tokens = sequence->num_tokens();
  const size_t hbm_cached_tokens = sequence->kv_state().kv_cache_tokens_num();
  if (!host_manager->allocate_sequence(sequence, host_state, num_tokens)) {
    auto* composite =
        static_cast<CompositeBlockManager*>(block_managers_[dp_rank].get());
    composite->deallocate_for_sequence(sequence);
    host_manager->deallocate_for_sequence(sequence, host_state);
    sequence->reset();
    return false;
  }
  // Prefill already offloaded this prompt. Advance the Host cursor to the
  // last complete KV or grouped history block so the first compute step does
  // not collect that span again. A partial trailing block stays below the
  // cursor.
  const auto& host_leaves = host_manager->leaf_entries();
  auto unit_leaf = host_leaves.find(checkpoint_block_type(host_leaves));
  if (unit_leaf == host_leaves.end()) {
    unit_leaf = host_leaves.find(BlockType::KV);
  }
  CHECK(unit_leaf != host_leaves.end());
  const size_t unit_block_size = unit_leaf->second.leaf->block_size();
  CHECK_GT(unit_block_size, 0u);
  host_state.set_kv_cache_tokens_num(hbm_cached_tokens / unit_block_size *
                                     unit_block_size);
  return true;
}

bool HierarchyBlockManagerPool::allocate(Sequence* sequence,
                                         size_t num_tokens) {
  CHECK(sequence != nullptr);
  const int32_t dp_rank = BlockManagerPool::get_dp_rank(sequence);
  const bool started_empty = !sequence->has_any_blocks();
  if (should_probe_prefix_cache(sequence)) {
    allocate_shared(sequence);
  }

  const size_t restore_tokens = sequence->kv_cache_tokens_num();
  KVCacheState& hbm_state = sequence->kv_state();
  CHECK_LE(hbm_state.kv_cache_tokens_num(), restore_tokens);
  CHECK_LE(restore_tokens, num_tokens);

  auto* composite =
      static_cast<CompositeBlockManager*>(block_managers_[dp_rank].get());
  KVCacheState& host_state = sequence->host_kv_state();
  auto* host_manager = host_block_managers_[dp_rank].get();
  CHECK(host_manager);

  if (!composite->allocate_sequence(sequence, num_tokens)) {
    if (started_empty) {
      composite->deallocate_for_sequence(sequence);
      host_manager->deallocate_for_sequence(sequence, host_state);
      sequence->reset();
    }
    return false;
  }

  if (!host_manager->allocate_sequence(sequence, host_state, num_tokens)) {
    if (started_empty) {
      composite->deallocate_for_sequence(sequence);
      host_manager->deallocate_for_sequence(sequence, host_state);
      sequence->reset();
    }
    return false;
  }

  collect_load_block_transfer_infos(sequence);
  CHECK_GE(hbm_state.current_max_tokens_capacity(), restore_tokens);
  collect_offload_pairs(sequence);
  sequence->clear_host_cache_match();
  return true;
}

void HierarchyBlockManagerPool::collect_load_block_transfer_infos(
    Sequence* sequence) {
  const int32_t dp_rank = sequence->dp_rank();
  const auto* host_manager = host_block_managers_[dp_rank].get();
  CHECK(host_manager);
  KVCacheState& host_state = sequence->host_kv_state();
  KVCacheState& hbm_state = sequence->kv_state();
  const size_t host_cached_tokens = host_state.kv_cache_tokens_num();
  const size_t hbm_cached_tokens = hbm_state.kv_cache_tokens_num();
  if (host_cached_tokens <= hbm_cached_tokens) {
    return;
  }
  std::vector<BlockTransferInfo>& load_infos =
      load_block_transfer_infos_[dp_rank];
  const size_t load_info_begin = load_infos.size();
  for (const auto& [type, entry] : host_manager->leaf_entries()) {
    if (type == BlockType::LINEAR) {
      std::vector<Block>* host_blocks = host_state.mutable_blocks(type);
      std::vector<Block>* hbm_blocks = hbm_state.mutable_blocks(type);
      if (hbm_blocks->size() < 2 || host_blocks->size() < 2) {
        continue;
      }

      // allocate_prefill() has already reduced the Host probe to
      // [restore, live]. The restore slot is the prefix-cache source; the
      // back slot is private forward capacity and must never be loaded.
      Block* host_source = &(*host_blocks)[0];
      if (!host_source->is_valid() || !(*hbm_blocks)[0].is_valid()) {
        continue;
      }

      // The loaded span ends at the Host cursor. collect_load copies that
      // span onto HBM and then advances the HBM cursor to match. The Host
      // front slot is the checkpoint of that span, one block before the
      // Host cursor. The pre-load HBM cursor can be zero on a cold device.
      const size_t block_size = entry.leaf->block_size();
      CHECK_GT(block_size, 0u);
      if (host_cached_tokens < block_size) {
        continue;
      }
      const size_t checkpoint_index = host_cached_tokens / block_size - 1;
      const std::optional<const XXH3Key*> checkpoint_hash =
          sequence->block_hash_at(entry.leaf->options().hasher_type(),
                                  checkpoint_index,
                                  static_cast<uint32_t>(block_size));
      if (!checkpoint_hash.has_value()) {
        continue;
      }

      Block& restore_block = (*hbm_blocks)[0];
      restore_block.set_hash_value((*checkpoint_hash)->data);
      host_source->set_hash_value((*checkpoint_hash)->data);
      // The loaded HBM slot is the checkpoint for this prefix. Publish it so
      // the next HBM match can hit the block just restored from Host.
      auto* hbm_manager =
          static_cast<CompositeBlockManager*>(block_managers_[dp_rank].get());
      const auto hbm_linear = hbm_manager->leaf_entries().find(type);
      CHECK(hbm_linear != hbm_manager->leaf_entries().end());
      std::vector<Block> publish_blocks{restore_block};
      hbm_linear->second.leaf->cache(publish_blocks);
      VLOG(1) << "[HostCache][LinearLoadEnqueue] seq=" << sequence->seq_id()
              << " src=" << host_source->id() << " dst=" << restore_block.id()
              << " cached_tokens=" << hbm_cached_tokens
              << " cached_index=" << checkpoint_index
              << " hash=" << (*checkpoint_hash)->to_hex();
      load_infos.emplace_back(host_source->id(),
                              restore_block.id(),
                              (*checkpoint_hash)->data,
                              TransferType::H2D,
                              BlockType::LINEAR);
      continue;
    }

    const size_t block_size = entry.leaf->block_size();
    CHECK_GT(block_size, 0u);
    std::vector<Block>* host_blocks = host_state.mutable_blocks(type);
    std::vector<Block>* hbm_blocks = hbm_state.mutable_blocks(type);
    // Copy the span Host already holds past the HBM cursor. Both cursors are
    // the sequence's cached-token counts for that tier.
    const size_t begin_block = hbm_cached_tokens / block_size;
    const size_t end_block = std::min({host_cached_tokens / block_size,
                                       host_blocks->size(),
                                       hbm_blocks->size()});
    const BlockHasherType hasher_type = entry.leaf->options().hasher_type();
    for (size_t block_index = begin_block; block_index < end_block;
         ++block_index) {
      Block& host_block = (*host_blocks)[block_index];
      Block& hbm_block = (*hbm_blocks)[block_index];
      if (!hbm_block.is_valid() || !host_block.is_valid()) {
        continue;
      }
      const std::optional<const XXH3Key*> block_hash = sequence->block_hash_at(
          hasher_type, block_index, static_cast<uint32_t>(block_size));
      if (!block_hash.has_value()) {
        break;
      }
      // Shared Host sources can restore multiple private HBM destinations.
      // Stamp both sides before the H2D info is built.
      host_block.set_hash_value((*block_hash)->data);
      hbm_block.set_hash_value((*block_hash)->data);
      VLOG(1) << "[HostCache][LoadHash] seq=" << sequence->seq_id()
              << " type=" << static_cast<int32_t>(type)
              << " src=" << host_block.id() << " dst=" << hbm_block.id()
              << " cached_index=" << block_index
              << " hash=" << (*block_hash)->to_hex();
      load_infos.emplace_back(host_block.id(),
                              hbm_block.id(),
                              (*block_hash)->data,
                              TransferType::H2D,
                              type);
    }
  }
  if (load_infos.size() != load_info_begin) {
    hbm_state.set_kv_cache_tokens_num(host_cached_tokens);
  }
}

void HierarchyBlockManagerPool::allocate_shared(Sequence* sequence) {
  CHECK(sequence != nullptr);
  if (!should_probe_prefix_cache(sequence)) {
    return;
  }

  const int32_t dp_rank = BlockManagerPool::get_dp_rank(sequence);
  auto* composite =
      static_cast<CompositeBlockManager*>(block_managers_[dp_rank].get());
  if (sequence->stage() == SequenceStage::DECODE) {
    composite->allocate_shared_for_sequence(sequence);
    return;
  }
  KVCacheState& hbm_state = sequence->kv_state();
  if (!hbm_state.prefix_cache_matched()) {
    composite->allocate_shared_for_sequence(sequence);
  }

  KVCacheState& host_state = sequence->host_kv_state();
  if (!host_state.prefix_cache_matched()) {
    auto* host_manager = host_block_managers_[dp_rank].get();
    CHECK(host_manager);
    host_manager->allocate_shared_for_sequence(sequence, host_state);
    host_state.set_prefix_cache_matched();
  }

  const size_t hbm_tokens = hbm_state.kv_cache_tokens_num();
  const size_t host_tokens = host_state.kv_cache_tokens_num();

  sequence->clear_host_cache_match();
  if (host_tokens > hbm_tokens) {
    const size_t unit_size = prefetch_unit_size(composite->leaf_combination(),
                                                composite->leaf_entries());
    sequence->set_host_cache_match(host_tokens,
                                   (host_tokens - hbm_tokens) / unit_size);
  }
}

HostCacheRestorePoint HierarchyBlockManagerPool::select_host_cache_restore(
    Sequence* sequence,
    size_t max_copy_units) {
  CHECK(sequence != nullptr);
  const size_t restore_tokens = sequence->kv_cache_tokens_num();

  const int32_t dp_rank = BlockManagerPool::get_dp_rank(sequence);
  const auto* composite =
      static_cast<const CompositeBlockManager*>(block_managers_[dp_rank].get());
  const bool is_swa_compressed =
      composite->leaf_combination() ==
      CompositeBlockManager::LeafCombination::SWA_COMPRESSED;
  const BlockType unit_type =
      !is_swa_compressed ? BlockType::KV
                         : checkpoint_block_type(composite->leaf_entries());
  const auto unit_leaf = composite->leaf_entries().find(unit_type);
  CHECK(unit_leaf != composite->leaf_entries().end());
  const size_t copy_unit_tokens = unit_leaf->second.leaf->block_size();
  CHECK_GT(copy_unit_tokens, 0u);
  const size_t restore_units = restore_tokens / copy_unit_tokens;
  const size_t hbm_cached_units = std::min(
      sequence->kv_state().num_cached_blocks(unit_type), restore_units);
  const size_t full_copy_units = restore_units - hbm_cached_units;
  if (full_copy_units == 0) {
    return HostCacheRestorePoint{/*restore_target_tokens=*/restore_tokens,
                                 /*copy_units=*/0};
  }

  size_t selected_copy_units = std::min(max_copy_units, full_copy_units);
  size_t selected_tokens =
      (hbm_cached_units + selected_copy_units) * copy_unit_tokens;

  if (!is_swa_compressed) {
    return HostCacheRestorePoint{/*restore_target_tokens=*/selected_tokens,
                                 /*copy_units=*/selected_copy_units};
  }

  const auto swa_leaf = composite->leaf_entries().find(BlockType::SWA);
  CHECK(swa_leaf != composite->leaf_entries().end());
  const size_t swa_block_size = swa_leaf->second.leaf->block_size();
  const size_t blocks_per_window = static_cast<size_t>(
      swa_leaf->second.leaf->options().swa_blocks_per_seq());
  CHECK_GT(swa_block_size, 0u);
  CHECK_GT(blocks_per_window, 0u);

  const Slice<Block> host_swa =
      sequence->host_kv_state().blocks(BlockType::SWA);
  // V4.1 may checkpoint before the window fills; require every existing
  // tail block, while V4 retains its full-window admission rule.
  while (selected_copy_units > 0 &&
         !has_valid_swa_window(
             selected_tokens,
             swa_block_size,
             unit_type == BlockType::C1 || unit_type == BlockType::C2
                 ? swa_tail_blocks(selected_tokens / swa_block_size,
                                   blocks_per_window)
                 : blocks_per_window,
             host_swa)) {
    --selected_copy_units;
    selected_tokens -= copy_unit_tokens;
  }

  return HostCacheRestorePoint{/*restore_target_tokens=*/selected_tokens,
                               /*copy_units=*/selected_copy_units};
}

bool HierarchyBlockManagerPool::should_probe_prefix_cache(
    Sequence* sequence) const {
  if (!options_.enable_prefix_cache() || sequence == nullptr) {
    return false;
  }
  // Decode only has a device cache.  Prefill may enter with Host already
  // matched by Mooncake, so the tier completion flags, rather than block
  // presence, define whether another probe is needed.
  KVCacheState& hbm_state = sequence->kv_state();
  if (sequence->stage() == SequenceStage::DECODE) {
    return !hbm_state.prefix_cache_matched();
  }
  if (!hbm_state.prefix_cache_matched()) {
    return true;
  }
  // A Host match released after an HBM allocation failure may be re-probed
  // while HBM still contains only its immutable prefix aliases. Once forward
  // has advanced beyond that prefix, re-probing would detach computed DSV4
  // SWA/C4/C128 blocks and replace the cursor with a shorter cache match.
  if (hbm_state.kv_cache_tokens_num() > hbm_state.shared_tokens_num()) {
    return false;
  }
  return !sequence->host_kv_state().prefix_cache_matched();
}

BlockManager* HierarchyBlockManagerPool::leaf_of(BlockType type,
                                                 int32_t dp_rank) const {
  const auto* host_manager = host_block_managers_[dp_rank].get();
  CHECK(host_manager);
  const auto& leaves = host_manager->leaf_entries();
  const auto leaf = leaves.find(type);
  return leaf == leaves.end() ? nullptr : leaf->second.leaf.get();
}

void HierarchyBlockManagerPool::prefetch_from_storage(
    std::shared_ptr<Request> request,
    PrefetchDoneCallback done) {
  CHECK(request != nullptr);
  CHECK(done != nullptr);
  if (!options_.enable_prefix_cache() || !options_.enable_kvcache_store() ||
      request->sequences().empty()) {
    done(std::move(request));
    return;
  }

  prefetching_requests_.fetch_add(1, std::memory_order_relaxed);
  const std::unique_ptr<Sequence>& prefill_sequence =
      request->sequences().front();
  Sequence* sequence = prefill_sequence.get();
  CHECK(sequence != nullptr);
  CHECK(!sequence->has_any_blocks())
      << "Mooncake prefetch admission requires an empty Sequence state.";

  const int32_t dp_rank = BlockManagerPool::get_dp_rank(sequence);
  const auto* composite =
      static_cast<const CompositeBlockManager*>(block_managers_[dp_rank].get());
  const CompositeBlockManager::LeafCombination combination =
      composite->leaf_combination();
  auto* host_manager = host_block_managers_[dp_rank].get();
  CHECK(host_manager);
  const CompositeBlockManager::LeafMap& host_leaves =
      host_manager->leaf_entries();
  const size_t unit_size = prefetch_unit_size(combination, host_leaves);
  const size_t max_prefix_tokens =
      sequence->tokens().empty() ? 0 : sequence->tokens().size() - 1;
  const size_t target_tokens = (max_prefix_tokens / unit_size) * unit_size;

  // Host admission owns the shared-prefix probe and cross-leaf trim. Keep
  // that transaction in the composite so prefetch sees the same shared
  // cursors as ordinary Host prefix-cache admission.
  host_manager->allocate_shared_for_sequence(sequence,
                                             sequence->host_kv_state());
  const size_t base_tokens = sequence->host_kv_state().kv_cache_tokens_num();
  // Length already in Host before this request's Store GET. A later publish
  // of fetched blocks must not be counted as this hit.
  VLOG(1) << "[HostCache][PrefetchHit] sequence_id=" << sequence->seq_id()
          << " host_hit_tokens=" << base_tokens;
  // Host already covers every cacheable token. Nothing remains to fetch.
  if (base_tokens >= target_tokens) {
    done(std::move(request));
    const size_t previous =
        prefetching_requests_.fetch_sub(1, std::memory_order_acq_rel);
    CHECK_GT(previous, 0u);
    return;
  }

  // Prefetch allocation is a Host-only transaction. Every leaf receives the
  // full target once. The resulting state vectors carry both the shared
  // prefix and every successfully allocated destination; request
  // construction turns that state directly into complete Store units.
  for (const auto& [type, entry] : host_leaves) {
    entry.leaf->allocate_for_prefetch(sequence, target_tokens);
  }

  StoragePrefetchRequest storage_request =
      build_prefetch_request(sequence, combination, host_leaves);
  storage_request.request_id = request->request_id();
  auto storage_request_ptr = std::make_shared<const StoragePrefetchRequest>(
      std::move(storage_request));

  auto finalize = [this,
                   request,
                   sequence,
                   dp_rank,
                   base_tokens,
                   target_tokens,
                   combination,
                   storage_request_ptr,
                   done = std::move(done)](PrefetchSummary summary) mutable {
    const bool transfer_ok =
        storage_request_ptr->units.empty() ||
        summary.gated_hits.size() == storage_request_ptr->units.size();
    const bool publish =
        transfer_ok && !request->finished() && !request->cancelled();
    const size_t final_tokens = finalize_prefetch_tokens(
        combination,
        host_block_managers_[dp_rank]->leaf_entries(),
        base_tokens,
        summary,
        sequence->host_kv_state().blocks(BlockType::LINEAR));
    if (util::store_prefetch_stats_enabled() &&
        !storage_request_ptr->units.empty()) {
      const size_t fetched_tokens =
          publish ? final_tokens - std::min(final_tokens, base_tokens) : 0;
      request->record_store_prefetch(fetched_tokens, summary.latency_ms);
      log_prefetch_summary(*request,
                           *sequence,
                           dp_rank,
                           base_tokens,
                           target_tokens,
                           fetched_tokens,
                           summary);
    }
    finalize_prefetch(
        sequence, host_block_managers_[dp_rank].get(), final_tokens, publish);
    done(std::move(request));
    const size_t previous =
        prefetching_requests_.fetch_sub(1, std::memory_order_acq_rel);
    CHECK_GT(previous, 0u);
  };

  if (storage_request_ptr->units.empty()) {
    finalize(PrefetchSummary{});
    return;
  }

  CHECK(storage_request_ptr->valid());
  CHECK(engine_ != nullptr) << "Mooncake prefetch requires an Engine.";
  engine_->prefetch_from_storage(
      dp_rank,
      storage_request_ptr,
      [request]() { return request->finished() || request->cancelled(); },
      std::move(finalize));
}

void HierarchyBlockManagerPool::transfer_blocks(std::vector<Batch>& batches) {
  for (size_t i = 0; i < load_block_transfer_infos_.size(); ++i) {
    if (load_block_transfer_infos_[i].empty()) {
      continue;
    }
    CHECK_LT(i, batches.size())
        << "Missing batch for pending H2D transfer at dp_rank=" << i;
    batches[i].set_batch_id();
    batches[i].require_host_restore();
    engine_->transfer_kv_blocks(
        i, batches[i].batch_id(), std::move(load_block_transfer_infos_[i]));
    load_block_transfer_infos_[i].clear();
  }

  transfer_offload_blocks();
}

void HierarchyBlockManagerPool::transfer_blocks() { transfer_offload_blocks(); }

bool HierarchyBlockManagerPool::has_pending_async_block_release() const {
  if (offload_transfers_.has_pending()) {
    return true;
  }
  for (const OffloadBlockPairQueue& queue : offload_block_pair_queues_) {
    if (queue.size_approx() > 0) {
      return true;
    }
  }
  return false;
}

void HierarchyBlockManagerPool::transfer_offload_blocks() {
  for (size_t i = 0; i < offload_block_pair_queues_.size(); i++) {
    std::vector<BlockTransferInfo> transfer_infos;
    std::vector<Block> src_blocks;
    std::vector<Block> dst_blocks;
    std::vector<BlockType> block_types;

    std::shared_ptr<OffloadBlockPair> block_pair;
    while (offload_block_pair_queues_[i].try_dequeue(block_pair)) {
      src_blocks.emplace_back(std::move(block_pair->src));
      dst_blocks.emplace_back(std::move(block_pair->dst));
      transfer_infos.emplace_back(
          BlockTransferInfo(src_blocks.back().id(),
                            dst_blocks.back().id(),
                            dst_blocks.back().get_immutable_hash_value(),
                            TransferType::D2H2G));
      // Preserve the BlockType so the completion callback publishes to the
      // right host leaf. The engine transfer path stamps the outbound info's
      // block_type from device layer coverage; this side just needs it to
      // route publish/free.
      transfer_infos.back().block_type = block_pair->block_type;
      transfer_infos.back().checkpoint_row = block_pair->checkpoint_row;
      block_types.emplace_back(block_pair->block_type);
      block_pair.reset();
    }

    if (!transfer_infos.empty()) {
      const uint32_t expected_workers = engine_->host_transfer_worker_count();
      CHECK_GT(expected_workers, 0U);
      std::shared_ptr<KVTransferTracker::Completion> completion =
          offload_transfers_.track();
      folly::collectAll(
          std::move(engine_->transfer_kv_blocks(i, std::move(transfer_infos))))
          .via(&folly::InlineExecutor::instance())
          .thenValue([device_blocks = std::move(src_blocks),
                      host_blocks = std::move(dst_blocks),
                      block_types_vec = std::move(block_types),
                      host_manager = host_block_managers_[i].get(),
                      dp_rank = i,
                      expected_workers](
                         std::vector<folly::Try<uint32_t>>&& results) mutable {
            bool copy_ok = results.size() == expected_workers;
            if (!copy_ok) {
              LOG(ERROR) << "Incomplete Host offload at dp_rank=" << dp_rank
                         << ", expected workers=" << expected_workers
                         << ", actual results=" << results.size();
            }
            for (auto&& result : results) {
              if (result.hasException()) {
                const std::exception* exception =
                    result.exception().get_exception();
                LOG(ERROR) << "Offload RPC failed at dp_rank=" << dp_rank
                           << ": "
                           << (exception != nullptr ? exception->what()
                                                    : "non-standard exception");
                copy_ok = false;
                continue;
              }
              if (result.value() != host_blocks.size()) {
                LOG(ERROR) << "Offload copy fail, expected "
                           << host_blocks.size() << ", got " << result.value();
                copy_ok = false;
              }
            }

            // Always release the reserved ids so the block pools do not leak.
            device_blocks.clear();

            // Successful copies become visible in the Host prefix cache.
            // Failures publish nothing, but both paths release every reserved
            // Host id through the same type-aware ownership path.
            finalize_host_blocks(
                copy_ok, std::move(host_blocks), block_types_vec, host_manager);

            return 0;
          })
          .ensure([completion = std::move(completion)]() mutable {
            completion.reset();
          });
    }
  }
}

}  // namespace xllm
