/* Copyright 2026 The xLLM Authors.

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

#include "core/distributed_runtime/engine_utils.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "core/util/utils.h"
#include "framework/block/block_utils.h"

namespace xllm {

void configure_dsv4_block_manager_options(const ModelArgs& args,
                                          BlockManagerPool::Options& options,
                                          const KVCacheCapacity& kv_cache_cap,
                                          int32_t block_size,
                                          int32_t max_tokens_per_batch) {
  if (!util::is_deepseek_v4_model_type(args.model_type())) {
    return;
  }

  // Integer codes stored in `Options::manager_types()`, mirroring the values
  // the composite manager dispatches on (see composite_block_manager.cpp): 0
  // builds a plain BlockManagerImpl leaf (KV / compressed TOKEN groups); 1
  // builds a SlidingWindowBlockManager leaf. Moved here with the option
  // builder itself so the engine no longer redeclares them inline.
  constexpr uint32_t kManagerTypeBlockManagerImpl = 0;
  constexpr uint32_t kManagerTypeSlidingWindowBlockManager = 1;

  std::vector<uint32_t> manager_types{kManagerTypeSlidingWindowBlockManager};
  std::vector<uint32_t> manager_compress_ratios{
      0};  // unused for sliding window manager
  std::vector<uint32_t> token_manager_ratios;
  token_manager_ratios.reserve(2);
  if (util::is_deepseek_v41_model_type(args.model_type())) {
    // V4.1: one TOKEN manager per distinct kv-source compress ratio, in
    // first-kv-source-appearance order (contract §2: group order
    // [SWA, TOKEN(2), TOKEN(1)] for the standard layer plan). Reuse layers
    // read the source layer's tensors and the TOKEN group's block table,
    // so only the kv-source ratios get managers.
    const std::vector<int32_t>& compress_ratios = args.compress_ratios();
    for (const int32_t layer_id : args.kv_source_layer_ids()) {
      if (layer_id < 0 ||
          layer_id >= static_cast<int64_t>(compress_ratios.size())) {
        continue;
      }
      const int32_t ratio = compress_ratios[static_cast<size_t>(layer_id)];
      if (ratio != 2 && ratio != 1) {
        continue;
      }
      const uint32_t ratio_u32 = static_cast<uint32_t>(ratio);
      if (std::find(token_manager_ratios.begin(),
                    token_manager_ratios.end(),
                    ratio_u32) == token_manager_ratios.end()) {
        token_manager_ratios.push_back(ratio_u32);
      }
    }
  } else {
    for (const int32_t ratio : args.compress_ratios()) {
      if (ratio == 4 || ratio == 128) {
        const uint32_t ratio_u32 = static_cast<uint32_t>(ratio);
        if (std::find(token_manager_ratios.begin(),
                      token_manager_ratios.end(),
                      ratio_u32) == token_manager_ratios.end()) {
          token_manager_ratios.push_back(ratio_u32);
        }
      }
    }
  }
  for (const uint32_t ratio : token_manager_ratios) {
    manager_types.push_back(kManagerTypeBlockManagerImpl);
    manager_compress_ratios.push_back(ratio);
  }

  const int64_t semantic_window = std::max(args.window_size(), 1);
  const int64_t max_model_len = args.max_seq_len();
  const int64_t effective_window =
      max_model_len > 0 ? std::min<int64_t>(semantic_window, max_model_len)
                        : semantic_window;
  // The pool exposes the window as uint32_t; a value beyond that would silently
  // truncate on the cast below, so reject it like the block-count checks.
  CHECK_LE(effective_window,
           static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      << "DSV4 effective_window exceeds uint32_t range: " << effective_window;
  const int64_t swa_blocks_per_seq =
      get_swa_blocks_per_seq(effective_window, block_size);
  CHECK_LE(swa_blocks_per_seq,
           static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      << "DSV4 swa_blocks_per_seq exceeds uint32_t range: "
      << swa_blocks_per_seq;
  const uint32_t swa_blocks_per_seq_u32 =
      static_cast<uint32_t>(swa_blocks_per_seq);
  CHECK_LE(kv_cache_cap.swa_count(),
           static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      << "DSV4 swa_count exceeds uint32_t range: " << kv_cache_cap.swa_count();
  CHECK_LE(kv_cache_cap.c4_count(),
           static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      << "DSV4 c4_count exceeds uint32_t range: " << kv_cache_cap.c4_count();
  CHECK_LE(kv_cache_cap.c128_count(),
           static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      << "DSV4 c128_count exceeds uint32_t range: "
      << kv_cache_cap.c128_count();
  // The pool exposes this as uint32_t; a negative value would wrap to a huge
  // capacity, so reject it up front like the block-count checks above.
  CHECK_GE(max_tokens_per_batch, 0)
      << "DSV4 max_tokens_per_batch must be >= 0, got " << max_tokens_per_batch;

  options.sliding_window_size(static_cast<uint32_t>(effective_window))
      .swa_blocks_per_seq(swa_blocks_per_seq_u32)
      .swa_num_blocks(static_cast<uint32_t>(kv_cache_cap.swa_count()))
      .c4_num_blocks(static_cast<uint32_t>(kv_cache_cap.c4_count()))
      .c128_num_blocks(static_cast<uint32_t>(kv_cache_cap.c128_count()))
      .max_tokens_per_batch(static_cast<uint32_t>(max_tokens_per_batch))
      .manager_types(std::move(manager_types))
      .compress_ratios(std::move(manager_compress_ratios));
}

}  // namespace xllm
