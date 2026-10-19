/* Copyright 2025-2026 The xLLM Authors.

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

inline constexpr char kInferContentLength[] = "Infer-Content-Length";
inline constexpr char kContentLength[] = "Content-Length";

// Reserved row 0 of the linear / SSM state cache, used as the padding slot for
// padded decode batch rows. This mirrors the KV cache convention, where block
// id 0 is permanently held for padding (see BlockManagerImpl and
// EmbeddingBlockManager). Keeping both caches padded to row 0 makes the padding
// contract uniform across attention and linear-attention layers; real
// sequences are handed ids in [1, num_blocks - 1] only.
inline constexpr int32_t kPaddingLinearStateId = 0;

// The linear-state slot pool reserves the padding row plus one extra guard slot
// so cache sizing and live-slot accounting keep a non-cacheable baseline.
inline constexpr int64_t kPaddingLinearStateBlocks = 2;

// DeepSeek-V4.1 native image placeholder (`<` + U+FF5C fullwidth bar +
// `deepseek_image` + U+FF5C fullwidth bar + `>`). The C++ chat template injects
// one occurrence per `image_url` content item when the message text carries no
// marker of its own, and the prompt processor expands each occurrence (native
// or alias) into the image's sentinel block. A single definition keeps the
// template and processor byte-for-byte consistent by construction instead of
// relying on a cross-module test.
inline constexpr char kDeepseekV41ImagePlaceholder[] =
    "<\xEF\xBD\x9C"
    "deepseek_image\xEF\xBD\x9C>";

// Generic image marker (`<image>`) many vision harnesses put in the request
// text. The chat template treats it as an existing placeholder and then must
// avoid injecting a second native placeholder per `image_url`; the prompt
// processor expands it into the image's sentinel block.
inline constexpr char kDeepseekV41ImageAlias[] = "<image>";

// Backslash-escaped form of the generic marker. The chat template protects it
// with kDeepseekV41ImageLiteralSentinel so it is not counted as a placeholder,
// and the prompt processor restores it to kDeepseekV41ImageAlias after image
// expansion. This lets plain request text contain a literal `<image>`.
inline constexpr char kDeepseekV41ImageAliasEscape[] = "\\<image>";

// Internal placeholder used only between chat-template rendering and prompt
// processing. It must not contain kDeepseekV41ImageAlias, or the prompt
// processor would expand it as an image marker before restoring it.
inline constexpr char kDeepseekV41ImageLiteralSentinel[] =
    "\x1F"
    "deepseek_v41_literal_image"
    "\x1F";

// DeepSeek-V4.1 splice-time image-pad placeholder. It is `image_token_id + 1`
// (see the LOCKED PAIR note on `kDeepseekV41CompressPadTo`): the compressor
// reserves this span id in the checkpoint tokenizer, and the prompt processor
// inserts it before image blocks so each block's first token lands at
// `kDeepseekV41CompressPadTo - 1 (mod kDeepseekV41CompressPadTo)`. A single
// definition keeps the processor and its tests referencing one spelling
// instead of duplicated literals.
inline constexpr char kDeepseekV41ImagePadPlaceholder[] =
    "<|place_holder_mm_span_0436|>";

// Compressor-alignment pad unit (COMPRESS_PAD_TO in vLLM's mm_preprocess.py):
// a V4.1 image block is padded to a multiple of this many compressor slots so
// the aligner's spatial merge lines up with the compression mask. The prompt
// processor reserves `kDeepseekV41CompressPadTo - 1` splice-time pad slots and
// the image processor trims its token budget by the same amount, so this value
// is a LOCKED PAIR with the Python `_vision_block_slots` block layout; the two
// C++ users must reference this single definition to stay in lockstep.
inline constexpr int32_t kDeepseekV41CompressPadTo = 4;

}  // namespace xllm
