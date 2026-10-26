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

#include "core/framework/block/linear_state_block_manager.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "core/framework/config/scheduler_config.h"
#include "core/framework/request/sequence.h"

namespace xllm {

namespace {

bool crosses_checkpoint(size_t begin, size_t end, size_t checkpoint_stride) {
  CHECK_LE(begin, end);
  CHECK_GT(checkpoint_stride, 0u);
  return begin / checkpoint_stride != end / checkpoint_stride;
}

}  // namespace

bool last_decode_step_crosses_checkpoint(const Sequence& sequence,
                                         size_t checkpoint_stride) {
  return last_decode_step_checkpoint_row(sequence, checkpoint_stride)
      .has_value();
}

std::optional<size_t> last_decode_step_checkpoint_row(
    const Sequence& sequence,
    size_t checkpoint_stride) {
  CHECK_GT(checkpoint_stride, 0u);
  // HBM cache only. Host cache is a separate tier and does not decide the
  // decode checkpoint row.
  const size_t cached_tokens = sequence.kv_state().kv_cache_tokens_num();
  const size_t accepted_length = sequence.accepted_length();
  if (cached_tokens <= accepted_length + 1) {
    return std::nullopt;
  }
  const size_t begin = cached_tokens - accepted_length - 1;
  if (!crosses_checkpoint(begin, cached_tokens, checkpoint_stride)) {
    return std::nullopt;
  }
  return accepted_length;
}

LinearStateBlockManager::LinearStateBlockManager(const Options& options)
    : BlockManagerImpl(([](Options linear_options) {
        linear_options.block_type(BlockType::LINEAR);
        return linear_options;
      })(options)) {
  CHECK_GT(options_.num_blocks(), 1u)
      << "linear-state leaf needs at least one usable slot (plus padding)";
  CHECK_GT(options_.block_size(), 0)
      << "linear-state leaf needs a positive chunk stride";
}

std::optional<std::vector<Block>>
LinearStateBlockManager::allocate_for_sequence(Sequence* seq,
                                               KVCacheState& kv_state,
                                               size_t /*num_tokens*/) {
  if (seq == nullptr) {
    return std::nullopt;
  }
  if (seq->is_prefill_stage()) {
    return allocate_prefill(seq, kv_state);
  }
  return allocate_decode(seq, kv_state);
}

bool LinearStateBlockManager::allocate_for_prefetch(Sequence* seq,
                                                    size_t num_tokens) {
  if (seq == nullptr || block_size() == 0) {
    return false;
  }

  KVCacheState& host_state = seq->host_kv_state();
  const size_t target_blocks = num_tokens / block_size();
  const size_t matched_blocks = host_state.kv_cache_tokens_num() / block_size();
  // The caller only reaches here when the host hit is shorter than the target.
  CHECK_LT(matched_blocks, target_blocks);

  std::vector<Block> blocks = host_state.take_blocks(BlockType::LINEAR);
  if (matched_blocks == 0) {
    CHECK(blocks.empty());
  } else {
    // Host admission keeps one restore source. It stays at its logical
    // position and is not part of the reused prefetch pool.
    CHECK_EQ(blocks.size(), 1u);
    CHECK(blocks.front().is_valid());
    blocks.insert(blocks.begin(), matched_blocks - 1, Block());
  }
  blocks.resize(target_blocks);

  const size_t prefetch_blocks = target_blocks - matched_blocks;
  const uint32_t batch_size = std::max(options_.prefetch_batch_size(), 1u);
  uint32_t valid_per_batch =
      std::min(options_.prefetch_valid_blocks_per_batch(), batch_size);
  // Published LINEAR checkpoints are evictable. Counting only the free list
  // shrinks to one group once earlier requests fill the host prefix cache,
  // and the next batch then overwrites the only copy of the previous batch.
  const uint32_t reusable_slots = static_cast<uint32_t>(num_available_blocks());
  // Keep the largest valid count that divides the batch. Use two groups when
  // they fit; otherwise keep one group and leave later batches empty.
  while (valid_per_batch > 0 && (batch_size % valid_per_batch != 0 ||
                                 valid_per_batch > reusable_slots)) {
    --valid_per_batch;
  }
  const uint32_t groups =
      valid_per_batch == 0 ? 0 : (reusable_slots < valid_per_batch * 2 ? 1 : 2);

  const uint32_t stride =
      valid_per_batch == 0 ? 0 : batch_size / valid_per_batch;
  const size_t pool_size = static_cast<size_t>(groups) * valid_per_batch;
  std::vector<Block> pool = block_allocator()->allocate(pool_size);
  if (pool.size() != pool_size) {
    pool.clear();
  }

  seq->update_block_hashes(static_cast<uint32_t>(block_size()),
                           options_.hasher_type());
  const Slice<XXH3Key> hashes =
      seq->block_hashes(static_cast<uint32_t>(block_size()));
  CHECK_GE(hashes.size(), target_blocks);

  size_t placed = 0;
  if (!pool.empty() && stride > 0) {
    const size_t placeable = prefetch_blocks / stride * stride;
    for (size_t offset = 0; offset < placeable; ++offset) {
      const size_t batch = (matched_blocks + offset) / batch_size;
      const size_t group = groups == 0 ? 0 : batch % groups;
      const size_t in_batch = offset % batch_size;
      if ((in_batch + 1) % stride != 0) {
        continue;
      }
      const size_t slot = group * valid_per_batch + in_batch / stride;
      Block alias = pool[slot];
      alias.set_hash_value(hashes[matched_blocks + offset].data);
      blocks[matched_blocks + offset] = std::move(alias);
      ++placed;
    }
    // A partial stride has no LINEAR checkpoint. Store writes one checkpoint
    // per full stride (batch_size / V KV blocks). Pinning the last KV block
    // onto that page asks for a hash the prefill never published.
  }

  host_state.replace_composite_blocks(BlockType::LINEAR,
                                      std::move(blocks),
                                      /*num_shared_blocks=*/0,
                                      /*cache_publish_cursor=*/0);
  const size_t aligned = stride == 0 ? 0 : prefetch_blocks / stride;
  return placed == aligned;
}

void LinearStateBlockManager::trim_prefetch_blocks(Sequence* seq,
                                                   size_t max_hit_tokens) {
  if (seq == nullptr || block_size() == 0) {
    return;
  }

  KVCacheState& host_state = seq->host_kv_state();
  std::vector<Block> blocks = host_state.take_blocks(BlockType::LINEAR);
  const size_t keep = std::min(max_hit_tokens / block_size(), blocks.size());

  if (VLOG_IS_ON(10)) {
    size_t valid_count = 0;
    for (const auto& block : blocks) {
      if (block.is_valid()) {
        ++valid_count;
      }
    }
    VLOG(10) << "[trim_prefetch_blocks] max_hit_tokens=" << max_hit_tokens
             << " block_size=" << block_size() << " keep=" << keep
             << " blocks.size=" << blocks.size()
             << " valid_count=" << valid_count;
  }

  // A Linear checkpoint vector is a logical index space. Keep only the
  // deepest valid checkpoint in the accepted gate range; all other physical
  // aliases, including valid blocks after the range, are temporary Store
  // state and must be released by this leaf.
  size_t source_index = keep;
  for (size_t index = keep; index > 0; --index) {
    if (blocks[index - 1].is_valid()) {
      source_index = index - 1;
      break;
    }
  }

  VLOG(10) << "[trim_prefetch_blocks] source_index=" << source_index
           << " (source_index < keep)=" << (source_index < keep);

  Block source;
  if (source_index < keep) {
    source = blocks[source_index];
    VLOG(10) << "[trim_prefetch_blocks] source extracted, is_valid="
             << source.is_valid();
  }

  if (!source.is_valid()) {
    host_state.erase_blocks(BlockType::LINEAR);
    return;
  }

  VLOG(10) << "[trim_prefetch_blocks] retaining source, publishing to cache";

  // Publish only the last hit checkpoint. Earlier valid blocks in the
  // accepted range are released with the temporary vector below.
  // LinearStatePrefixCache needs the absolute checkpoint index to derive the
  // chained key, so the published vector stays sparse up to that index.
  seq->update_block_hashes(static_cast<uint32_t>(block_size()),
                           options_.hasher_type());
  const Slice<XXH3Key> hashes =
      seq->block_hashes(static_cast<uint32_t>(block_size()));
  CHECK_GT(hashes.size(), source_index);
  std::vector<Block> publish(source_index + 1);
  publish[source_index] = std::move(source);
  source = publish[source_index];
  BlockManagerImpl::cache(seq->hash_tokens(options_.hasher_type()),
                          publish,
                          /*existed_shared_blocks_num=*/0,
                          seq->mm_data(),
                          hashes);

  std::vector<Block> retained(1);
  retained[0] = std::move(source);
  host_state.replace_composite_blocks(BlockType::LINEAR,
                                      std::move(retained),
                                      /*num_shared_blocks=*/0,
                                      /*cache_publish_cursor=*/0);
}

std::optional<std::vector<Block>> LinearStateBlockManager::allocate_prefill(
    Sequence* seq,
    KVCacheState& kv_state) {
  CHECK_LE(kv_state.blocks(BlockType::LINEAR).size(), 2u);

  if (seq->kv_cache_tokens_num() > kv_state.kv_cache_tokens_num()) {
    kv_state.erase_blocks(BlockType::LINEAR);
    std::vector<Block> allocated = block_allocator()->allocate(2);
    if (allocated.size() != 2) {
      return std::nullopt;
    }
    return allocated;
  }

  std::vector<Block>* blocks = kv_state.mutable_blocks(BlockType::LINEAR);
  if (blocks->size() == 2) {
    blocks->erase(blocks->begin());
  }

  std::vector<Block> allocated = block_allocator()->allocate(1);
  if (allocated.empty()) {
    if (blocks->empty()) {
      VLOG(1)
          << "LinearStateBlockManager::allocate_prefill: no blocks to allocate";
      return std::nullopt;
    }
    return std::vector<Block>{};
  }

  // The block that remains is the completed read source. Publish it only when
  // this leaf owns automatic prefix-cache insertion and the HBM cache lands
  // on a checkpoint boundary.
  if (options_.prefix_cache_publish_mode() ==
          PrefixCachePublishMode::AUTOMATIC &&
      !seq->is_graph_warmup() && prefix_cache_ != nullptr && !blocks->empty()) {
    const size_t stride = block_size();
    const size_t cached_tokens = seq->kv_cache_tokens_num();
    if (stride > 0 && cached_tokens % stride == 0) {
      const size_t checkpoint_index = cached_tokens / stride;
      if (checkpoint_index > 0) {
        seq->update_block_hashes(static_cast<uint32_t>(stride),
                                 options_.hasher_type());
        const Slice<XXH3Key> hashes =
            seq->block_hashes(static_cast<uint32_t>(stride));
        CHECK(blocks->front().is_valid());
        blocks->front().set_hash_value(hashes[checkpoint_index - 1].data);
        prefix_cache_->insert(Slice<Block>(*blocks).slice(0, 1));
      }
    }
  }
  return allocated;
}

std::optional<std::vector<Block>> LinearStateBlockManager::allocate_decode(
    Sequence* seq,
    KVCacheState& kv_state) {
  std::vector<Block>* blocks = kv_state.mutable_blocks(BlockType::LINEAR);
  CHECK_LE(blocks->size(), 2u);

  const size_t checkpoint_stride = block_size();
  const bool completed_checkpoint =
      blocks->size() == 2 &&
      last_decode_step_crosses_checkpoint(*seq, checkpoint_stride);

  if (blocks->size() == 2) {
    if (completed_checkpoint) {
      if (options_.preserve_decode_checkpoint()) {
        // Host cache is on. collect_offload_pairs runs after allocate and
        // moves the front slot. Overlap keeps the n-2 result on the front
        // block, so that slot is already the one to save. Overlap off writes
        // the previous step into the back block, so swap that block to the
        // front before collect moves it.
        if (!options_.enable_schedule_overlap()) {
          std::iter_swap(blocks->begin(), blocks->begin() + 1);
        }
        return std::vector<Block>{};
      }
      blocks->erase(blocks->begin());
    } else {
      // The previous decode step has not yet confirmed a checkpoint. Keep
      // both slots alive and alternate their roles so the next out-of-place
      // step reads the slot written by the preceding step. This is also
      // what makes overlap allocation safe: the old read slot cannot be
      // returned to the global pool while its in-flight step may still use it.
      std::iter_swap(blocks->begin(), blocks->begin() + 1);
      return std::vector<Block>{};
    }
  }

  if (blocks->empty()) {
    std::vector<Block> allocated = block_allocator()->allocate(1);
    return allocated.empty()
               ? std::nullopt
               : std::optional<std::vector<Block>>(std::move(allocated));
  }

  // The sequence's HBM cache decides the next checkpoint. The KVCacheState
  // argument may be the Host tier, whose cached length is independent.
  const size_t cached_tokens = seq->kv_state().kv_cache_tokens_num();
  const size_t predicted_tokens =
      cached_tokens + static_cast<size_t>(options_.num_speculative_tokens()) +
      1;
  if (!crosses_checkpoint(cached_tokens, predicted_tokens, checkpoint_stride)) {
    return std::vector<Block>{};
  }

  std::vector<Block> allocated = block_allocator()->allocate(1);
  if (allocated.empty()) {
    // The extra slot only preserves a checkpoint for the future offload path;
    // recurrent decode itself supports reading and writing the live slot in
    // place. Under pressure, keep decoding instead of turning an optional
    // checkpoint copy into a sequence preemption.
    return std::vector<Block>{};
  }
  return allocated;
}

void LinearStateBlockManager::cache(const Slice<int32_t>& token_ids,
                                    std::vector<Block>& blocks,
                                    size_t existed_shared_blocks_num,
                                    const MMData& mm_data,
                                    const Slice<XXH3Key>& block_hashes) {
  if (blocks.empty() || prefix_cache_ == nullptr) {
    return;
  }
  const size_t publish_end =
      std::min(token_ids.size() / block_size(), blocks.size() - 1);
  VLOG(1) << "[LinearCache] manager=" << this
          << " decode=" << options_.instance_is_decode()
          << " blocks=" << blocks.size() << " publish_end=" << publish_end;
  if (publish_end <= existed_shared_blocks_num) {
    return;
  }
  CHECK_GE(block_hashes.size(), publish_end);
  BlockManagerImpl::cache(token_ids.slice(0, publish_end * block_size()),
                          blocks,
                          existed_shared_blocks_num,
                          mm_data,
                          block_hashes);
}

}  // namespace xllm
