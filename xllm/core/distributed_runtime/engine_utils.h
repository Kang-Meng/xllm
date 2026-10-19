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

#pragma once

#include <cstdint>

#include "framework/block/block_manager_pool.h"
#include "framework/kv_cache/kv_cache_capacity.h"
#include "framework/model/model_args.h"

namespace xllm {

// Configures `options` for the DeepSeek-V4 family (V4 and V4.1): DeepSeek-V4
// serves its compressed KV through a composite block manager (SWA + one TOKEN
// manager per kv-source compress ratio) plus the sliding-window pool sizing
// derived from `kv_cache_cap`. Without this, a plain single-KV pool is built,
// the sequence never owns composite (C1/C2/C4/C128) blocks,
// `multi_block_tables` stays empty, and the DSA attention loses its block
// tables -> garbage output even for pure-text requests.
//
// Extracted from LLMEngine's inline KV-block-manager configuration and shared
// by both LLMEngine and VLMEngine (a past VLM-side omission of this block
// caused exactly that drift). No-op for non-DSV4 model types.
void configure_dsv4_block_manager_options(const ModelArgs& args,
                                          BlockManagerPool::Options& options,
                                          const KVCacheCapacity& kv_cache_cap,
                                          int32_t block_size,
                                          int32_t max_tokens_per_batch);

}  // namespace xllm
