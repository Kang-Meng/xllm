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

#include "processors/deepseek_v41_prompt_processor.h"

#include <absl/strings/str_replace.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "core/framework/tokenizer/tokenizer.h"
#include "core/util/utils.h"

namespace xllm {

namespace {

// Role codes matching vllm mm_preprocess.py (IMAGE_START..IMAGE_END).
enum class ImageBlockRole : char {
  START = 0,
  PAD = 1,
  IMAGE = 2,
  NEWLINE = 3,
  END = 4,
};

// Undo the chat template's protection of escaped `\<image>` text. This must
// happen after the image-placeholder scan, otherwise the restored alias would
// be expanded as a real image marker.
void restore_image_literals(std::string& prompt) {
  prompt = absl::StrReplaceAll(
      prompt, {{kDeepseekV41ImageLiteralSentinel, kDeepseekV41ImageAlias}});
}

}  // namespace

// `process` locates both placeholders with `std::string::find`. An empty needle
// matches at `begin` on every iteration (pos == begin, matched_len == 0), so
// `begin` never advances and the loop spins forever; a shorter-than-begin match
// would also make `pos - begin` underflow. Both placeholders are non-empty
// literals in `core/common/constants.h`, so pin that invariant at compile time
// rather than relying on the find loop to detect it at request time.
static_assert(sizeof(kDeepseekV41ImagePlaceholder) > 1,
              "DeepSeek-V4.1 image placeholder must be non-empty");
static_assert(sizeof(kDeepseekV41ImageAlias) > 1,
              "DeepSeek-V4.1 image alias must be non-empty");
static_assert(sizeof(kDeepseekV41ImageLiteralSentinel) > 1,
              "DeepSeek-V4.1 image literal sentinel must be non-empty");

bool validate_deepseek_v41_tokenizer_binding(const Tokenizer& tokenizer,
                                             int32_t image_token_id) {
  if (image_token_id < 0) {
    return true;
  }
  const auto native_id = tokenizer.token_to_id(kDeepseekV41ImagePlaceholder);
  const auto pad_id = tokenizer.token_to_id(kDeepseekV41ImagePadPlaceholder);
  return native_id.has_value() && pad_id.has_value() &&
         *native_id == image_token_id && *pad_id == image_token_id + 1;
}

DeepseekV41PromptProcessor::DeepseekV41PromptProcessor(const ModelArgs& args) {
  image_token_id_ = args.image_token_id();
  image_pad_id_ = image_token_id_ >= 0 ? image_token_id_ + 1 : -1;
  if (args.mm_downsample_ratio() > 0) {
    downsample_ratio_ = args.mm_downsample_ratio();
  }
}

// LOCKED PAIR: this layout is implemented twice — here (C++, consumed by
// `process` / `adjust_token_layout` / `find_mm_spans`) and in Python
// `_vision_block_slots` (xllm/python/models/deepseek_v41.py, consumed by
// `encode`). The two MUST stay byte-for-byte consistent: a mismatch shifts
// token positions vs embed rows silently.
// tests/processors/deepseek_v41_prompt_processor_test.cpp and
// tests/python/test_deepseek_v41_vision_layout.py pin both sides to the same
// golden role sequences — update them together.
std::vector<char> DeepseekV41PromptProcessor::image_block_roles(
    int64_t n_vit_h,
    int64_t n_vit_w) const {
  const int64_t r = downsample_ratio_;
  const int64_t n_llm_h = util::ceil_div(n_vit_h, r);
  const int64_t n_llm_w = util::ceil_div(n_vit_w, r);
  const int64_t pad_h = n_llm_h % 2;
  const int64_t rows = n_llm_h + pad_h;
  const int64_t row_len = n_llm_w + 1;
  const int64_t pad_last = (((rows / 2) * row_len) % 2) * 2;

  // Pre-reorder types (row-major over the padded rows).
  std::vector<char> pre;
  pre.reserve(rows * row_len);
  for (int64_t i = 0; i < n_llm_h; ++i) {
    for (int64_t j = 0; j < n_llm_w; ++j) {
      pre.push_back(static_cast<char>(ImageBlockRole::IMAGE));
    }
    pre.push_back(static_cast<char>(ImageBlockRole::NEWLINE));
  }
  for (int64_t i = 0; i < row_len * pad_h; ++i) {
    pre.push_back(static_cast<char>(ImageBlockRole::PAD));
  }

  // Row-pair interleave: arange(rows*row_len).view(rows/2, 2, row_len)
  // .transpose(1, 2).reshape(-1).
  std::vector<int64_t> order;
  order.reserve(rows * row_len);
  for (int64_t a = 0; a < rows / 2; ++a) {
    for (int64_t c = 0; c < row_len; ++c) {
      for (int64_t b = 0; b < 2; ++b) {
        order.push_back((a * 2 + b) * row_len + c);
      }
    }
  }

  std::vector<char> roles;
  roles.reserve(rows * row_len + 2);
  roles.push_back(static_cast<char>(ImageBlockRole::START));
  for (int64_t o : order) {
    roles.push_back(pre[o]);
  }
  for (int64_t i = 0; i < pad_last; ++i) {
    roles.push_back(static_cast<char>(ImageBlockRole::PAD));
  }
  roles.push_back(static_cast<char>(ImageBlockRole::END));
  return roles;
}

int32_t DeepseekV41PromptProcessor::block_sentinel_count(
    const torch::Tensor& grid_thw) const {
  if (!grid_thw.defined() || grid_thw.numel() < 2) {
    return -1;
  }
  auto grid = grid_thw.to(torch::kCPU).to(torch::kLong).contiguous();
  const int64_t* g = grid.data_ptr<int64_t>();
  const int64_t n_vit_h = g[grid.numel() - 2];
  const int64_t n_vit_w = g[grid.numel() - 1];
  int32_t cnt = 0;
  for (char role : image_block_roles(n_vit_h, n_vit_w)) {
    if (role != static_cast<char>(ImageBlockRole::PAD)) {
      ++cnt;
    }
  }
  return cnt;
}

bool DeepseekV41PromptProcessor::process(std::string& prompt,
                                         const MMData& mm_data) {
  torch::Tensor image_grid;
  if (auto res = mm_data.get<torch::Tensor>("image_grid_thw")) {
    image_grid = res.value();
  }
  if (!image_grid.defined()) {
    restore_image_literals(prompt);
    return true;
  }
  auto grid = image_grid.cpu().to(torch::kLong);
  if (grid.dim() == 1) {
    grid = grid.reshape({1, -1});
  }
  const int64_t grid_cols = grid.size(1);
  if (grid_cols < 2) {
    LOG(ERROR) << "DeepSeek-V4.1 image_grid_thw needs at least 2 columns "
                  "[n_vit_h, n_vit_w], got "
               << grid_cols;
    return false;
  }

  const int64_t num_images = grid.size(0);
  std::string data;
  data.reserve(prompt.size() * 2);
  int64_t image_index = 0;
  size_t begin = 0;
  while (image_index < num_images) {
    size_t img_pos = prompt.find(image_token_, begin);
    size_t alias_pos = prompt.find(image_alias_, begin);
    size_t pos = std::min(img_pos, alias_pos);
    if (pos == std::string::npos) {
      break;
    }
    const size_t matched_len =
        (pos == img_pos) ? image_token_.size() : image_alias_.size();
    data.append(prompt, begin, pos - begin);
    // Read the trailing two columns (n_vit_h, n_vit_w) so the layout stays
    // consistent with block_sentinel_count and the Python `encode` side even
    // if the grid later grows a leading THW column.
    const int64_t n_vit_h = grid[image_index][grid_cols - 2].item<int64_t>();
    const int64_t n_vit_w = grid[image_index][grid_cols - 1].item<int64_t>();
    for (char role : image_block_roles(n_vit_h, n_vit_w)) {
      data.append(role == static_cast<char>(ImageBlockRole::PAD)
                      ? image_pad_token_
                      : image_token_);
    }
    ++image_index;
    begin = pos + matched_len;
  }
  if (begin < prompt.size()) {
    size_t leftover = std::min(prompt.find(image_token_, begin),
                               prompt.find(image_alias_, begin));
    if (leftover == std::string::npos) {
      data.append(prompt, begin, std::string::npos);
    } else {
      // Surplus placeholders: the prompt carries more markers than images. This
      // happens when a caller puts a marker in the text and the chat template
      // also injects one per image_url. A leftover native placeholder would
      // tokenize to the image sentinel id with no matching embed row, so strip
      // every unmatched placeholder from the tail instead of crashing later.
      std::string tail = prompt.substr(begin);
      size_t stripped_native = 0;
      size_t stripped_alias = 0;
      for (size_t pos = 0;
           (pos = tail.find(image_token_, pos)) != std::string::npos;) {
        tail.erase(pos, image_token_.size());
        ++stripped_native;
      }
      for (size_t pos = 0;
           (pos = tail.find(image_alias_, pos)) != std::string::npos;) {
        tail.erase(pos, image_alias_.size());
        ++stripped_alias;
      }
      LOG(WARNING) << "DeepSeek-V4.1 prompt carries more image placeholders "
                      "than the "
                   << num_images << " image(s) in the request; stripped "
                   << (stripped_native + stripped_alias)
                   << " surplus placeholder(s) from the tail ("
                   << stripped_native << " native, " << stripped_alias
                   << " alias). A stripped placeholder would tokenize to the "
                      "image sentinel id with no matching embed row, so the "
                      "prompt is silently corrected; edit it if that is "
                      "unintended.";
      data += tail;
    }
  }
  // Every image must consume exactly one placeholder: a shortfall would leave
  // the surplus images without any span and silently drop their embeddings.
  if (image_index != num_images) {
    LOG(ERROR) << "DeepSeek-V4.1 prompt carries " << num_images
               << " image(s) but only " << image_index
               << " image placeholder(s); each image needs one '<image>' or "
                  "'<｜deepseek_image｜>' placeholder. Rejecting the request "
                  "instead of silently dropping images.";
    return false;
  }
  prompt = std::move(data);
  restore_image_literals(prompt);
  return true;
}

bool DeepseekV41PromptProcessor::adjust_token_layout(
    std::vector<int32_t>& token_ids,
    const MMData& mm_data) {
  if (image_token_id_ < 0) {
    return true;
  }
  const int32_t image_id = image_token_id_;
  const int32_t pad_id = image_pad_id_;
  const auto in_block = [&](int32_t t) { return t == image_id || t == pad_id; };

  // Expected sentinel count per image, so adjacent blocks (no text between
  // images) are aligned independently instead of collapsing onto one run.
  std::vector<int32_t> expected;
  if (mm_data.hold<MMItemVec>()) {
    const auto& items = mm_data.items<MMItemVec>();
    expected.reserve(items.size());
    for (const auto& item : items) {
      if (item.type() != MMType::IMAGE) {
        // Only images contribute a sentinel block; video/audio tokens are not
        // aligned here and must not consume an entry in `expected`.
        continue;
      }
      int32_t cnt = -1;
      if (auto grid_opt = item.get<torch::Tensor>("image_grid_thw")) {
        cnt = block_sentinel_count(grid_opt.value());
      } else {
        LOG(WARNING) << "DeepseekV41PromptProcessor: multimodal item carries "
                        "no usable image_grid_thw; adjacent image blocks may "
                        "collapse onto one run during layout adjustment.";
      }
      expected.push_back(cnt);
    }
  }

  // Emit into a fresh buffer instead of inserting in place: every
  // `token_ids.insert(begin() + i, ...)` shifts the whole tail, so a prompt
  // with many image blocks costs O(n^2). One pre-reserved pass moves each token
  // at most once.
  std::vector<int32_t> out;
  out.reserve(token_ids.size() +
              static_cast<size_t>(expected.size()) *
                  static_cast<size_t>(kDeepseekV41CompressPadTo));

  size_t i = 0;
  size_t item_idx = 0;
  size_t copied = 0;
  while (i < token_ids.size() && item_idx < expected.size()) {
    if (!in_block(token_ids[i])) {
      ++i;
      continue;
    }
    out.insert(out.end(), token_ids.begin() + copied, token_ids.begin() + i);
    // The compressor consumes the sequence in runs of
    // `kDeepseekV41CompressPadTo` (= 4). `out.size()` is this block's absolute
    // start index (everything before it is already emitted), so inserting
    // `kDeepseekV41CompressPadTo - 1 - start % kDeepseekV41CompressPadTo` pads
    // lands the block's first token at index
    // `start + pads == kDeepseekV41CompressPadTo - 1 (mod
    // kDeepseekV41CompressPadTo)` -- the boundary the reference splice pins it
    // to (`mm_preprocess.py:_apply_token_matches_with_placeholders`). The
    // equivalent in-place algorithm read the same index as `i %
    // kDeepseekV41CompressPadTo`, so the target is unchanged; keep this exact
    // formula (not a generic `(-n) % m`) to stay bit-identical to the golden
    // layout.
    const size_t block_start = out.size();
    const int32_t compress_pad =
        kDeepseekV41CompressPadTo - 1 -
        static_cast<int32_t>(block_start % kDeepseekV41CompressPadTo);
    if (compress_pad > 0) {
      out.insert(out.end(), static_cast<size_t>(compress_pad), pad_id);
    }
    // Advance past this block over the source vector: its sentinel count when
    // the grid is usable, otherwise the whole consecutive in-block run. The
    // pads just emitted live only in `out`, so they are never re-examined.
    const int32_t want = expected[item_idx];
    size_t j = i;
    int32_t seen = 0;
    if (want > 0) {
      while (j < token_ids.size() && seen < want && in_block(token_ids[j])) {
        if (token_ids[j] == image_id) {
          ++seen;
        }
        ++j;
      }
    } else {
      while (j < token_ids.size() && in_block(token_ids[j])) {
        ++j;
      }
    }
    out.insert(out.end(), token_ids.begin() + i, token_ids.begin() + j);
    i = j;
    copied = j;
    ++item_idx;
  }
  // Each image item consumes exactly one sentinel block above, so a sentinel or
  // pad token still present past the last consumed block means the token stream
  // carries more image blocks than `expected` has items. Emitting that tail
  // verbatim would leave the extra block unaligned and unbound to any embed
  // row, so fail loudly instead of producing a silently misaligned layout. A
  // shortfall (fewer blocks than items) is intentionally tolerated: `process`
  // strips surplus placeholders and `find_mm_spans` rejects a missing block, so
  // this stays a pure alignment pass. Only check when alignment was attempted.
  if (!expected.empty()) {
    for (size_t k = i; k < token_ids.size(); ++k) {
      if (in_block(token_ids[k])) {
        LOG(ERROR)
            << "DeepSeek-V4.1 token layout carries more image sentinel "
               "blocks than the "
            << expected.size()
            << " image item(s) in the request; refusing to continue with "
               "an ambiguous block-to-image mapping.";
        return false;
      }
    }
  }
  out.insert(out.end(), token_ids.begin() + copied, token_ids.end());
  token_ids = std::move(out);
  return true;
}

bool DeepseekV41PromptProcessor::find_mm_spans(
    const std::vector<int32_t>& token_ids,
    MMData& mm_data) {
  auto& mm_items = mm_data.items<MMItemVec>();
  const int32_t num_items = static_cast<int32_t>(mm_items.size());
  int32_t num_image_items = 0;
  for (const auto& item : mm_items) {
    if (item.type() == MMType::IMAGE) {
      ++num_image_items;
    }
  }
  int32_t global_mm_index = 0;
  int32_t assigned = 0;
  size_t i = 0;
  const size_t n = token_ids.size();
  const auto in_block = [&](int32_t t) {
    return t == image_token_id_ || (image_pad_id_ >= 0 && t == image_pad_id_);
  };
  while (i < n && global_mm_index < num_items) {
    auto& item = mm_items[global_mm_index];
    if (item.type() != MMType::IMAGE) {
      // Non-image items (video/audio) carry no image sentinel block; skip them
      // without consuming token positions so a following image keeps its span.
      ++global_mm_index;
      continue;
    }
    if (!in_block(token_ids[i])) {
      ++i;
      continue;
    }
    // Adjacent blocks (no text between images) form one contiguous run of
    // sentinel/pad tokens. Segment the run per image using the expected
    // sentinel count derived from each image's grid, so every item gets its
    // own span instead of the whole run collapsing onto the first item.
    int32_t expected = -1;
    if (auto grid_opt = item.get<torch::Tensor>("image_grid_thw")) {
      expected = block_sentinel_count(grid_opt.value());
    }
    if (expected <= 0) {
      // No usable per-image grid (absent, or too short to read
      // n_vit_h/n_vit_w): fall back to taking the whole contiguous sentinel run
      // as this item's span. That is exact for a lone or text-separated block;
      // for adjacent images it under-serves the later items, which the
      // `assigned != num_image_items` check below turns into a hard failure, so
      // a bad grid can never silently mis-assign spans. Still logged: the image
      // processor always attaches the grid, so reaching here means the
      // multimodal item was built inconsistently.
      LOG(WARNING)
          << "DeepseekV41PromptProcessor: image item #" << global_mm_index
          << " has no usable image_grid_thw; falling back to "
             "assigning its whole contiguous sentinel run as one span.";
    }
    const size_t start = i;
    int32_t sentinel_count = 0;
    if (expected > 0) {
      while (i < n && in_block(token_ids[i]) && sentinel_count < expected) {
        if (token_ids[i] == image_token_id_) {
          ++sentinel_count;
        }
        ++i;
      }
    } else {
      while (i < n && in_block(token_ids[i])) {
        if (token_ids[i] == image_token_id_) {
          ++sentinel_count;
        }
        ++i;
      }
    }
    const int32_t length = static_cast<int32_t>(i - start);
    item.mutable_state().mutable_token_pos() = {static_cast<int32_t>(start),
                                                length};
    std::vector<uint8_t> mask_data(length);
    for (int32_t k = 0; k < length; ++k) {
      mask_data[k] = token_ids[start + k] == image_token_id_ ? 1 : 0;
    }
    item.mutable_state().mutable_mm_token_mask() =
        torch::from_blob(mask_data.data(),
                         {length},
                         torch::TensorOptions().dtype(torch::kBool))
            .clone();
    item.mutable_state().mutable_mm_token_num() = sentinel_count;
    ++assigned;
    ++global_mm_index;
  }
  // Defense in depth: if `process` admitted the request, every image item must
  // have found its block above. A shortfall here means a layout mismatch, so
  // fail loudly instead of leaving images without a token span.
  if (assigned != num_image_items) {
    LOG(ERROR) << "DeepSeek-V4.1 prompt processor assigned a span to only "
               << assigned << " of " << num_image_items
               << " image item(s); refusing to continue with unassigned image "
                  "embeddings.";
    return false;
  }
  return true;
}

}  // namespace xllm
