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

#include <torch/torch.h>

#include <cstdint>
#include <functional>
#include <memory>

#include "core/framework/encoder_cache/encoder_cache.h"
#include "core/framework/model/model_input_params.h"

namespace xllm {

class EncoderExecutor final {
 public:
  using EncodeFunction = std::function<MMDict(const ModelInputParams&)>;

  explicit EncoderExecutor(int64_t max_cache_bytes);

  void prepare(const ModelInputParams& params, const EncodeFunction& encode);

 private:
  std::unique_ptr<EncoderCache> cache_;
};

}  // namespace xllm
