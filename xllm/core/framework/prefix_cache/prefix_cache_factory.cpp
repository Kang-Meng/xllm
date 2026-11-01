#include "prefix_cache_factory.h"

#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>

#include "linear_state_prefix_cache.h"

namespace xllm {

std::unique_ptr<PrefixCache> create_prefix_cache(PrefixCache::Options options) {
  // Two block types need the gap-tolerant probe:
  //   LINEAR -- block_size here is the KV hash stride; the checkpoint stride
  //             remains on LinearStateBlockManager.
  //   SWA    -- block_size here is the SWA base block. Middle misses are
  //             harmless (attention only reads the tail; the composite
  //             enforces the tail-continuity check).
  // Both share the same subclass; the type name is retained to keep the
  // SWA-adoption diff narrow.
  if (options.block_type() == BlockType::LINEAR ||
      options.block_type() == BlockType::SWA) {
    return std::make_unique<LinearStatePrefixCache>(options);
  }
  return std::make_unique<PrefixCache>(options);
}

}  // namespace xllm
