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

#include "block.h"

#include <glog/logging.h>
#include <string.h>

#include <cstdint>

#include "block_manager.h"

namespace xllm {
const char* block_type_label(BlockType type) {
  switch (type) {
    case BlockType::KV:
      return "kv";
    case BlockType::SWA:
      return "swa";
    case BlockType::C4:
      return "c4";
    case BlockType::C128:
      return "c128";
    case BlockType::EMBEDDING:
      return "embedding";
    case BlockType::LINEAR:
      return "linear";
    case BlockType::C1:
      return "c1";
    case BlockType::C2:
      return "c2";
  }
  LOG(FATAL) << "Unhandled block type " << static_cast<int32_t>(type);
  return "";
}

Block::Block(int32_t id, BlockManager* manager) : id_(id), manager_(manager) {
  // A block with no manager is an id holder. It has no shared refcount.
  if (manager_ != nullptr) {
    ref_count_ = new RefCounts;
    size_ = manager_->block_size();
  }
}

Block::~Block() { release(); }

// copy constructor
Block::Block(const Block& other) : Block(other, other.inflight_) {}

Block::Block(const Block& other, bool inflight)
    : id_(other.id_),
      size_(other.size_),
      ref_count_(other.manager_ == nullptr ? nullptr : other.ref_count_),
      inflight_(inflight),
      manager_(other.manager_) {
  memcpy(hash_value_, other.hash_value_, XXH3_128BITS_HASH_VALUE_LEN);

  inc_ref_count();
}

Block Block::retain_for_inflight() const {
  return Block(*this, /*inflight=*/true);
}

// copy assignment
Block& Block::operator=(const Block& other) {
  if (this != &other) {
    release();

    id_ = other.id_;
    size_ = other.size_;
    manager_ = other.manager_;
    // A manager-less alias does not share the source counter. Padding is the
    // only manager-less block that still owns one, and copying it must not
    // alias that counter.
    ref_count_ = manager_ == nullptr ? nullptr : other.ref_count_;
    inflight_ = other.inflight_;

    memcpy(hash_value_, other.hash_value_, XXH3_128BITS_HASH_VALUE_LEN);

    inc_ref_count();
  }
  return *this;
}

Block::Block(Block&& other) noexcept
    : id_(other.id_),
      size_(other.size_),
      ref_count_(other.ref_count_),
      inflight_(other.inflight_),
      manager_(other.manager_) {
  memcpy(hash_value_, other.hash_value_, XXH3_128BITS_HASH_VALUE_LEN);

  // reset other without adjusting the reference count
  other.id_ = -1;
  other.size_ = 0;
  other.ref_count_ = nullptr;
  other.manager_ = nullptr;
}

Block& Block::operator=(Block&& other) noexcept {
  if (this != &other) {
    release();

    id_ = other.id_;
    size_ = other.size_;
    manager_ = other.manager_;
    ref_count_ = other.ref_count_;
    inflight_ = other.inflight_;

    memcpy(hash_value_, other.hash_value_, XXH3_128BITS_HASH_VALUE_LEN);

    other.id_ = -1;
    other.size_ = 0;
    other.ref_count_ = nullptr;
    other.manager_ = nullptr;
  }

  return *this;
}

void Block::inc_ref_count() {
  // No manager means this alias does not share a refcount.
  if (ref_count_ == nullptr || manager_ == nullptr) {
    return;
  }
  manager_->incr_ref_count(&ref_count_->total);
  if (!inflight_) {
    ref_count_->owners.fetch_add(1, std::memory_order_acq_rel);
  }
}

void Block::dec_ref_count() {
  // No manager means this alias does not share a refcount.
  if (ref_count_ == nullptr || manager_ == nullptr) {
    return;
  }
  if (!inflight_) {
    ref_count_->owners.fetch_sub(1, std::memory_order_acq_rel);
  }
  // The manager fetch_subs. previous == 1 means this alias was the last one.
  const uint32_t previous = manager_->dec_ref_count(&ref_count_->total);
  if (previous == 1) {
    delete ref_count_;
    ref_count_ = nullptr;
  }
}

void Block::release() {
  if (ref_count_ == nullptr) {
    return;
  }
  // free() reads this alias and calls dec_ref_count() before returning.
  if (manager_ != nullptr) {
    manager_->free(this);
    return;
  }
  // Padding keeps a sole counter and set_manager(nullptr), so it must not
  // enter free() (that checks id != 0) and does not call dec_ref_count().
  // Copies of a manager-less block do not alias this counter.
  delete ref_count_;
  ref_count_ = nullptr;
}

}  // namespace xllm
