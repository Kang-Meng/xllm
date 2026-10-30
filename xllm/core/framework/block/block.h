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

#include <openssl/sha.h>
#include <string.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <optional>
#include <vector>

#include "util/hash_util.h"
#include "util/slice.h"

namespace xllm {

class BlockManager;

// Identity of a KV block's cache role inside a sequence's KVCacheState. Used as
// the key of the per-sequence block map: the legacy flat attention KV lives
// under KV; the grouped-cache models use one triple each -- V4 under
// SWA/C4/C128 and V4.1 under SWA/C2/C1. EMBEDDING and LINEAR are
// per-sequence single-resource slots (one block per sequence): EMBEDDING backs
// the spec-decode embedding-row id, LINEAR backs the GDN recurrent state (its
// slot id also indexes the conv/ssm KV tensors, which are tagged as the LINEAR
// cache group). A block carries no type identity itself; the owning
// BlockManager decides which key to store it under when it fills the state.
// APPEND-ONLY: never renumber or remove an existing enumerator. This type is
// serialized by number -- as proto BlockType over RPC (worker.proto) and as the
// cache-group id in persisted Mooncake Store keys -- so a renumber is silently
// misread by a rolling/mixed-version peer and cold-invalidates store keys. C1
// and C2 are appended values; both names are new and carry no historical wire
// or persistence meaning.
enum class BlockType : int8_t {
  KV = 0,         // normal/Qwen flat attention KV, exported to block_tables
  SWA = 1,        // DSV4 sliding window, exported to multi_block_tables[0]
  C4 = 2,         // DSV4 compressed, exported to multi_block_tables[1]
  C128 = 3,       // DSV4 compressed, exported to multi_block_tables[2]
  EMBEDDING = 4,  // per-sequence spec-decode embedding-row slot, exported via
                  // get_embedding_block_id() (embedding_ids). Value kept at 4
                  // for proto BlockType wire compatibility.
  LINEAR = 5,     // per-sequence linear-state (GDN recurrent) live slot, drawn
                  // from LinearStateBlockManager; exported via
                  // get_linear_block_id() (linear_state_ids). Also the cache
                  // group for the conv/ssm recurrent-state KV tensors.
  C1 = 6,         // appended: DSV4.1 compressed TOKEN group (kv-source layers,
                  // ratio 1), exported to multi_block_tables[2] on V4.1 models
  C2 = 7,         // appended: DSV4.1 compressed TOKEN group (kv-source layers,
                  // ratio 2), exported to multi_block_tables[1] on V4.1 models
};

// Fixed column order of worker multi_block_tables. The exported tables must
// follow this order so they line up with the worker-side DSA group_infos; it
// must never depend on std::map iteration order or config traversal order.
// A model only ever populates its own triple: V4 exports [SWA, C4, C128] and
// V4.1 exports [SWA, C2, C1], so each model's tables stay contiguous and line
// up with its block-manager registration order. The order below is the
// registration order, NOT the enum order: build_cache_specs_v41 registers the
// V4.1 TOKEN groups as [C2 (ratio 2), C1 (ratio 1)] (first kv-source
// appearance for the standard layer plan), while V4 registers [C4, C128].
inline constexpr std::array<BlockType, 5> kMultiBlockExportOrder = {
    BlockType::SWA,
    BlockType::C4,
    BlockType::C128,
    BlockType::C2,
    BlockType::C1};

// Stable cache-group identity used by PD transfer. BlockType remains a local
// storage key; the serialized group id is intentionally opaque to transfer
// backends.
inline constexpr int32_t cache_group_id(BlockType type) {
  return static_cast<int32_t>(type);
}

// KV-split widens one source-side logical block across multiple destination
// blocks. This applies to ordinary KV and every grouped attention-cache pool,
// but not to the sequence-scoped embedding or recurrent-state slots.
inline constexpr bool is_kv_split_cache_block_type(BlockType type) {
  switch (type) {
    case BlockType::KV:
    case BlockType::SWA:
    case BlockType::C1:
    case BlockType::C2:
    case BlockType::C4:
    case BlockType::C128:
      return true;
    case BlockType::EMBEDDING:
    case BlockType::LINEAR:
      return false;
  }
  return false;
}

inline constexpr bool is_prefetch_gate_block_type(BlockType type) {
  switch (type) {
    case BlockType::KV:
    case BlockType::C4:
    case BlockType::C128:
    case BlockType::C1:
    case BlockType::C2:
      return true;
    case BlockType::SWA:
    case BlockType::EMBEDDING:
    case BlockType::LINEAR:
      return false;
  }
  return false;
}

// Canonical lowercase label for a block type, used as a monitoring-series key.
// Declared beside the enum so every exporter shares one spelling.
const char* block_type_label(BlockType type);

inline constexpr std::optional<BlockType> block_type_from_cache_group_id(
    int32_t group_id) {
  switch (group_id) {
    case cache_group_id(BlockType::KV):
      return BlockType::KV;
    case cache_group_id(BlockType::SWA):
      return BlockType::SWA;
    case cache_group_id(BlockType::C1):
      return BlockType::C1;
    case cache_group_id(BlockType::C2):
      return BlockType::C2;
    case cache_group_id(BlockType::C4):
      return BlockType::C4;
    case cache_group_id(BlockType::C128):
      return BlockType::C128;
    case cache_group_id(BlockType::EMBEDDING):
      return BlockType::EMBEDDING;
    case cache_group_id(BlockType::LINEAR):
      return BlockType::LINEAR;
    default:
      return std::nullopt;
  }
}

class Block final {
 public:
  ~Block();

  Block() = default;
  Block(int32_t id, BlockManager* allocator);

  Block(const Block& other);
  Block& operator=(const Block& other);

  Block(Block&& other) noexcept;
  Block& operator=(Block&& other) noexcept;

  // get the block id
  constexpr int32_t id() const { return id_; }

  // get the block size
  constexpr uint32_t size() const { return size_; }

  // Shared refcount. 0 when the block was moved from, or when it has no
  // manager. A manager-less block does not own a counter.
  uint32_t ref_count() const {
    return ref_count_ == nullptr
               ? 0
               : ref_count_->total.load(std::memory_order_acquire);
  }

  // check if the block is shared
  bool is_shared() const { return ref_count() > 1; }

  // Inflight references keep the block alive without adding a logical owner.
  Block retain_for_inflight() const;

  // Owners and total are updated independently, not as a joint snapshot.
  // Only total controls reclamation; this query does not synchronize access.
  uint32_t owner_count() const {
    return ref_count_ == nullptr
               ? 0
               : ref_count_->owners.load(std::memory_order_acquire);
  }

  // This is not a lock or an exclusive-write grant. Callers making write
  // decisions must serialize ownership changes and access through the existing
  // scheduler/manager synchronization protocol, not this atomic observation.
  bool has_multiple_owners() const { return owner_count() > 1; }

  // Valid when this alias still owns an id. A manager-less block can be valid
  // without a refcount. Move clears the id.
  bool is_valid() const { return id_ >= 0; }

  // owner manager that allocated this block.
  BlockManager* manager() const { return manager_; }

  // Reassign this block's owning manager. Used by the concurrency wrapper to
  // route Block dtor -> free() through the wrapper lock. The shared refcount
  // stays with this alias; do not use this to move a block onto another leaf.
  void set_manager(BlockManager* manager) { manager_ = manager; }

  // NOTE: Below block `hash_value_` is used for prefix cache and
  // for recording the hash value of the current block and previous blocks.
  // hash_value_ = Hash(current_block, Hash(pre_block)).
  const uint8_t* get_immutable_hash_value() const { return hash_value_; }
  uint8_t* get_mutable_hash_value() { return hash_value_; }

  void set_hash_value(const uint8_t* hash_value) {
    memcpy(hash_value_, hash_value, XXH3_128BITS_HASH_VALUE_LEN);
  }

  constexpr uint32_t get_hash_value_len() const {
    return XXH3_128BITS_HASH_VALUE_LEN;
  }

 private:
  struct RefCounts {
    // All references, including inflight keepalives. Reclaim only at zero.
    // Creating an inflight reference increments total but not owners.
    std::atomic<uint32_t> total{1};
    // State owners, excluding inflight keepalives. LINEAR allocation uses
    // multiple owners to decide write isolation; this count never reclaims.
    // With owners == 0 and total > 0, only keepalives remain and the block
    // must stay allocated until those references are released.
    std::atomic<uint32_t> owners{1};
  };

  Block(const Block& other, bool inflight);

  // Asks the manager to fetch_add the shared count. A null manager has no
  // shared counter and returns without touching one.
  void inc_ref_count();

  // Asks the manager to fetch_sub the shared count, then deletes the atomic
  // when this alias was the last one. A null manager returns immediately.
  void dec_ref_count();

  // With a manager, free() reads this alias and calls dec_ref_count() before
  // returning. The concurrent wrapper holds its lock across that whole call.
  // The padding block has no manager and does not enter free().
  void release();

  friend class BlockManager;

  // block id
  int32_t id_ = -1;

  // block size
  uint32_t size_ = 0;

  // reference count, shared across Block aliases of the same physical block.
  // Atomic because aliases are copied and destroyed across threads.
  RefCounts* ref_count_ = nullptr;

  bool inflight_ = false;

  // manager that manages this block
  BlockManager* manager_ = nullptr;

  uint8_t hash_value_[XXH3_128BITS_HASH_VALUE_LEN];
};

// equal operator, mainly used for testing
inline constexpr bool operator==(const Block& lhs, const Block& rhs) {
  return lhs.id() == rhs.id();
}

}  // namespace xllm
