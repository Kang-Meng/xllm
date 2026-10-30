/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <functional>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xllm {

// Assigns items to key-local batches without allowing any batch to exceed the
// configured item bound. The caller owns payload storage and uses the returned
// stable batch index to append the corresponding item.
template <typename Key, typename Hash = std::hash<Key>>
class KeyedBoundedBatchPlanner final {
 public:
  explicit KeyedBoundedBatchPlanner(size_t max_items_per_batch)
      : max_items_per_batch_(max_items_per_batch) {}

  std::optional<size_t> assign(const Key& key) {
    if (max_items_per_batch_ == 0) {
      return std::nullopt;
    }
    auto current = current_batch_by_key_.find(key);
    if (current == current_batch_by_key_.end() ||
        batch_sizes_[current->second] >= max_items_per_batch_) {
      const size_t batch_index = batch_sizes_.size();
      batch_sizes_.push_back(0);
      current = current_batch_by_key_.insert_or_assign(key, batch_index).first;
    }
    ++batch_sizes_[current->second];
    return current->second;
  }

  size_t batch_count() const { return batch_sizes_.size(); }

  size_t batch_size(size_t batch_index) const {
    return batch_index < batch_sizes_.size() ? batch_sizes_[batch_index] : 0;
  }

 private:
  const size_t max_items_per_batch_;
  std::vector<size_t> batch_sizes_;
  std::unordered_map<Key, size_t, Hash> current_batch_by_key_;
};

}  // namespace xllm
