/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace xllm {

struct StoreRegisteredRange {
  void* address = nullptr;
  size_t bytes = 0;
};

// Fastest complete replica tier currently holding an object, following
// Mooncake's read preference: MEMORY before any SSD-backed replica. This is an
// approximation of the replica Mooncake actually reads: Mooncake also prefers a
// local NoF SSD replica over a remote MEMORY one, but RealClient does not
// expose the local endpoints needed to tell them apart.
enum class MooncakeReplicaTier : uint8_t {
  MISSING = 0,
  MEMORY = 1,
  DISK = 2,
};

struct MultiBuffer {
  std::vector<void*> addresses;
  std::vector<size_t> sizes;
};

class KVCacheStoreTransport {
 public:
  virtual ~KVCacheStoreTransport() = default;
  // Buffers are aligned with keys. Results have keys.size() entries in key
  // order: zero means failure, nonzero means the entire object succeeded.
  // Unexpected GET exceptions propagate; a failed GET may have written some
  // bytes.
  virtual std::vector<uint8_t> batch_put(
      const std::vector<std::string>& keys,
      const std::vector<MultiBuffer>& buffers) = 0;
  virtual std::vector<uint8_t> batch_get(
      const std::vector<std::string>& keys,
      const std::vector<MultiBuffer>& buffers) = 0;
  // Metadata-only lookup; returns one tier per key in key order.
  virtual std::vector<MooncakeReplicaTier> batch_query_tiers(
      const std::vector<std::string>& keys) = 0;
};

}  // namespace xllm
