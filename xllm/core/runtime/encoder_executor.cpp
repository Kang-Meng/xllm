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

#include "core/runtime/encoder_executor.h"

#include <glog/logging.h>

#include "core/framework/multimodal/mm_visitor.h"

namespace xllm {

EncoderExecutor::EncoderExecutor(int64_t max_cache_bytes) {
  if (max_cache_bytes > 0) {
    cache_ = std::make_unique<EncoderCache>(max_cache_bytes);
  }
}

void EncoderExecutor::prepare(const ModelInputParams& params,
                              const EncodeFunction& encode) {
  auto& data = params.multimodal.mm_data;
  if (!data.valid()) {
    return;
  }
  if (cache_) {
    EncoderCacheLookupVisitor lookup(cache_.get());
    CHECK(data.foreach (lookup));
  }
  EncoderInputGatherVisitor inputs;
  CHECK(data.foreach (inputs));
  CHECK(inputs.finish(data));

  MMDict outputs = encode(params);
  EncoderOutputScatterVisitor scatter(outputs);
  CHECK(data.foreach (scatter));
  CHECK(scatter.finish());

  if (cache_) {
    EncoderCacheInsertVisitor insert(cache_.get());
    CHECK(data.foreach (insert));
  }
  EncoderEmbeddingGatherVisitor gather(params.attention.host.kv_seq_lens,
                                       params.attention.host.q_seq_lens);
  CHECK(data.foreach (gather));
  CHECK(gather.finish(data));
}

}  // namespace xllm
