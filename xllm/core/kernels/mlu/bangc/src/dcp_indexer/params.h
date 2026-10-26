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

namespace xllm::kernel::mlu::bangc::detail {

// Supported tensor dimensions and page layout.
constexpr int32_t kHeads = 32;
constexpr int32_t kDim = 128;
constexpr int32_t kPage = 16;
constexpr int32_t kTopk = 2048;

// mtp_592 per-core on-chip capacity; 32 KiB of NRAM stays with the stack.
constexpr int32_t kNramBytes = 480 * 1024;
constexpr int32_t kWramBytes = 512 * 1024;

// Prefill tiling: query rows per work item and local keys per compute step.
constexpr int32_t kTileQ = 16;
constexpr int32_t kTileK = 128;
constexpr int32_t kQkChannels = kTileQ * kHeads;  // conv.nram output channels
constexpr int32_t kScoreChannels = 64;  // head-reduction output, padded to 64

// On-chip selection keeps a whole local row (in physical slot order).
constexpr int32_t kMaxLocal = 33 * 1024;
constexpr int32_t kMaxPages = kMaxLocal / kPage;
static_assert(kMaxPages % 64 == 0, "page-sort padding fits page buffers");

// Diagnostic bit definitions; production selects none at compile time.
constexpr int32_t kDiagNoKeyLoad = 1;
constexpr int32_t kDiagNoStore = 2;
constexpr int32_t kDiagNoSelect = 4;
constexpr int32_t kDiagNoQk = 8;
constexpr int32_t kDiagNoHeads = 16;
constexpr int32_t kDiagNoMask = 32;
constexpr int32_t kDiagNoTranspose = 64;
constexpr int32_t kDiagNoSort = 128;
constexpr int32_t kDiagNoGather = 256;
constexpr int32_t kDiagNoPageSort = 512;
constexpr int32_t kDiagContigKeys = 1024;

}  // namespace xllm::kernel::mlu::bangc::detail
