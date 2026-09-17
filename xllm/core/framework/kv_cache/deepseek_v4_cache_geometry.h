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

#include <cstdint>

namespace xllm {

constexpr int32_t kDsv4C4CompressRatio = 4;
constexpr int32_t kDsv4C128CompressRatio = 128;
constexpr int64_t kDsv4C4PhysicalBlockSize = 512;
constexpr int64_t kDsv4C128PhysicalBlockSize = 16;
constexpr int64_t kDsv4CompressedBlockTokenSpan = 2048;

constexpr int64_t dsv4_compressed_physical_block_size(int32_t compress_ratio) {
  if (compress_ratio == kDsv4C4CompressRatio) {
    return kDsv4C4PhysicalBlockSize;
  }
  if (compress_ratio == kDsv4C128CompressRatio) {
    return kDsv4C128PhysicalBlockSize;
  }
  return 0;
}

}  // namespace xllm
