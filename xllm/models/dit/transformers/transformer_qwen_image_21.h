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

#include <glog/logging.h>
#include <torch/torch.h>
#if defined(USE_NPU)
#include <torch_npu/csrc/aten/CustomFunctions.h>

#include "core/framework/parallel_state/parallel_state.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/dit_cache/dit_cache_config.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/state_dict/utils.h"
#include "core/layers/common/add_matmul.h"

namespace xllm::qwen_image_21 {

// Joint sequence layout and rotary positions.

using ImageShape = std::array<int64_t, 3>;

struct SequenceLayout {
  torch::Tensor source_indices;  // expanded joint position -> original VLM slot
  torch::Tensor image_positions;
  torch::Tensor image_ids;
  torch::Tensor target_mask;
  torch::Tensor allowed_attention;  // [1,1,L,L], true means allowed
  torch::Tensor rope_cos;           // [L,head_dim/2], interleaved pairs
  torch::Tensor rope_sin;
  int64_t target_tokens = 0;
  int64_t prefix_tokens = 0;
  int64_t prompt_tokens = 0;
  bool has_padding = false;
  int64_t attention_query_offset = 0;
};

// Layout is constructed once per positive/negative branch, outside the layers.
// Both masks cover only the VLM sequence; target slots are appended here.
inline SequenceLayout prepare_sequence_layout(
    const torch::Tensor& image_pad_mask,
    const torch::Tensor& prompt_mask,
    const std::vector<ImageShape>& image_shapes,
    const std::vector<int64_t>& axes_dims,
    const torch::Device& device,
    bool causal_block = true) {
  CHECK_EQ(image_pad_mask.dim(), 2);
  CHECK_EQ(image_pad_mask.size(0), 1)
      << "QwenImage21 initially supports batch=1";
  CHECK(image_pad_mask.scalar_type() == torch::kBool);
  CHECK(!image_shapes.empty());
  CHECK_EQ(axes_dims.size(), 3);
  const int64_t prompt_tokens = image_pad_mask.size(1);
  CHECK_GT(prompt_tokens, 0);
  auto slots = image_pad_mask.to(torch::kCPU).contiguous();
  auto valid = prompt_mask.defined() ? prompt_mask.to(torch::kCPU).contiguous()
                                     : torch::ones_like(slots);
  CHECK(valid.sizes() == slots.sizes());
  CHECK((valid.eq(0) | valid.eq(1)).all().item<bool>());
  valid = valid.to(torch::kBool);
  // The upstream encoder strips left padding and right-pads the final embeds.
  CHECK(valid[0][0].item<bool>())
      << "Expected right-padded nonempty VLM sequence";
  bool seen_padding = false;
  const bool* slot_data = slots.data_ptr<bool>();
  const bool* valid_data = valid.data_ptr<bool>();
  std::vector<int64_t> source, positions, ids, frame, rows, cols;
  std::vector<uint8_t> key_valid;
  int64_t image_tokens = 0;
  for (const auto& shape : image_shapes) {
    CHECK_EQ(shape[0], 1);
    CHECK_GT(shape[1], 0);
    CHECK_GT(shape[2], 0);
    CHECK_EQ(shape[1] % 2, 0);
    CHECK_EQ(shape[2] % 2, 0);
    image_tokens += shape[1] * shape[2];
  }
  const int64_t target_tokens = image_shapes.back()[1] * image_shapes.back()[2];
  const int64_t condition_tokens = image_tokens - target_tokens;
  CHECK_EQ(slots.sum().item<int64_t>() * 4, condition_tokens)
      << "VLM image slots and VAE image shapes disagree";
  const int64_t joint_tokens =
      prompt_tokens + 3 * (condition_tokens / 4) + target_tokens;
  source.reserve(joint_tokens);
  positions.reserve(image_tokens);
  ids.reserve(joint_tokens);
  key_valid.reserve(joint_tokens);
  frame.reserve(joint_tokens);
  rows.reserve(joint_tokens);
  cols.reserve(joint_tokens);
  // Keep image blocks separate using their declared lengths, even when their
  // slot runs are adjacent and have no intervening text tokens.
  int64_t block = 0;
  int64_t within_block = 0;
  int64_t position = 0;
  const int64_t slots_count = prompt_tokens + target_tokens / 4;
  for (int64_t slot = 0; slot < slots_count; ++slot) {
    const bool is_target = slot >= prompt_tokens;
    const bool is_image = is_target || slot_data[slot];
    const bool is_valid = is_target || valid_data[slot];
    if (!is_target) {
      CHECK(!seen_padding || !is_valid) << "Expected right padding";
      seen_padding = seen_padding || !is_valid;
      CHECK(!is_image || is_valid) << "Image slots must be valid VLM tokens";
    }
    if (!is_image) {
      CHECK_EQ(within_block, 0) << "Text token splits a condition-image block";
      source.emplace_back(slot);
      ids.emplace_back(-1);
      key_valid.emplace_back(is_valid);
      frame.emplace_back(position);
      rows.emplace_back(position);
      cols.emplace_back(position);
      ++position;
      continue;
    }
    CHECK_LT(block, static_cast<int64_t>(image_shapes.size()));
    const int64_t height = image_shapes[block][1];
    const int64_t width = image_shapes[block][2];
    for (int64_t repeat = 0; repeat < 4; ++repeat) {
      positions.emplace_back(static_cast<int64_t>(ids.size()));
      source.emplace_back(slot);
      ids.emplace_back(block);
      key_valid.emplace_back(1);
      frame.emplace_back(position);
      rows.emplace_back(within_block / width - (height - height / 2));
      cols.emplace_back(within_block % width - (width - width / 2));
      ++within_block;
    }
    if (within_block == height * width) {
      position += std::max(height, width);
      within_block = 0;
      ++block;
    }
  }
  CHECK_EQ(block, static_cast<int64_t>(image_shapes.size()));
  CHECK_EQ(within_block, 0);
  CHECK_EQ(static_cast<int64_t>(source.size()), joint_tokens);
  // Reference tables cover [0,8192) and [-1024,0); reject out-of-range
  // coordinates instead of inadvertently indexing the negative-position tail.
  for (int64_t coordinate : frame) {
    CHECK_GE(coordinate, 0);
    CHECK_LT(coordinate, 8192);
  }
  for (const auto* coordinates : {&rows, &cols}) {
    for (int64_t coordinate : *coordinates) {
      CHECK_GE(coordinate, -1024);
      CHECK_LT(coordinate, 8192);
    }
  }
  auto cpu_long =
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
  auto image_ids = torch::tensor(ids, cpu_long).to(device);
  auto indices = torch::arange(joint_tokens, cpu_long).to(device);
  auto same_image = image_ids.unsqueeze(1).eq(image_ids.unsqueeze(0)) &
                    image_ids.unsqueeze(1).ge(0);
  auto allowed =
      causal_block
          ? (indices.unsqueeze(1).ge(indices.unsqueeze(0)) | same_image)
          : torch::ones({joint_tokens, joint_tokens},
                        image_ids.options().dtype(torch::kBool));
  auto key_mask =
      torch::tensor(key_valid, torch::TensorOptions().dtype(torch::kUInt8))
          .to(device, torch::kBool);
  allowed = allowed & key_mask.unsqueeze(0);
  std::vector<torch::Tensor> cos_parts, sin_parts;
  cos_parts.reserve(3);
  sin_parts.reserve(3);
  const std::array<const std::vector<int64_t>*, 3> coordinates{
      &frame, &rows, &cols};
  for (int64_t axis = 0; axis < 3; ++axis) {
    CHECK_GT(axes_dims[axis], 0);
    CHECK_EQ(axes_dims[axis] % 2, 0);
    auto exponent =
        torch::arange(0, axes_dims[axis], 2, torch::kFloat32) / axes_dims[axis];
    auto frequency = 1.0 / torch::pow(10000.0, exponent);
    auto angle = torch::tensor(*coordinates[axis], cpu_long)
                     .to(torch::kFloat32)
                     .unsqueeze(1) *
                 frequency;
    cos_parts.emplace_back(torch::cos(angle));
    sin_parts.emplace_back(torch::sin(angle));
  }
  return {torch::tensor(source, cpu_long).to(device),
          torch::tensor(positions, cpu_long).to(device),
          image_ids,
          indices.ge(joint_tokens - target_tokens),
          allowed.unsqueeze(0).unsqueeze(0).contiguous(),
          torch::cat(cos_parts, -1).to(device),
          torch::cat(sin_parts, -1).to(device),
          target_tokens,
          joint_tokens - target_tokens,
          prompt_tokens,
          seen_padding};
}

inline torch::Tensor apply_rotary(const torch::Tensor& input,
                                  const SequenceLayout& layout) {
  CHECK_EQ(input.dim(), 4);  // [B,S,H,D]
  CHECK_EQ(input.size(1), layout.rope_cos.size(0));
  CHECK_EQ(input.size(3), layout.rope_cos.size(1) * 2);
  auto pairs =
      input.to(torch::kFloat32)
          .reshape({input.size(0), input.size(1), input.size(2), -1, 2});
  auto even = pairs.select(-1, 0);
  auto odd = pairs.select(-1, 1);
  auto cosine = layout.rope_cos.unsqueeze(1);
  auto sine = layout.rope_sin.unsqueeze(1);
  return torch::stack({even * cosine - odd * sine, even * sine + odd * cosine},
                      -1)
      .flatten(-2)
      .to(input.scalar_type());
}

// Exact request-local prefix KV cache.

struct PrefixLayerKV {
  // Owned, contiguous [B,H,prefix,D]. Never views into full-sequence storage.
  torch::Tensor key;
  torch::Tensor value;
};

// One instance per request AND CFG branch. The caller must not modify the
// prompt, condition latents or model weights within its lifetime. This is not
// a shared cache or an approximate residual/activation cache.
class PrefixKVCache final {
 public:
  explicit PrefixKVCache(int64_t num_layers) {
    CHECK_GT(num_layers, 0);
    layers_.resize(static_cast<size_t>(num_layers));
  }
  PrefixKVCache(const PrefixKVCache&) = delete;
  PrefixKVCache& operator=(const PrefixKVCache&) = delete;
  PrefixKVCache(PrefixKVCache&&) = default;
  PrefixKVCache& operator=(PrefixKVCache&&) = default;

  bool ready() const { return ready_; }
  int64_t num_layers() const { return static_cast<int64_t>(layers_.size()); }
  PrefixLayerKV& layer(int64_t index) { return layers_.at(index); }
  const PrefixLayerKV& layer(int64_t index) const { return layers_.at(index); }
  const SequenceLayout& decode_layout() const { return decode_layout_; }

  // Bind once to the stable branch inputs. Reject accidental reuse with a
  // different branch/layout rather than serving from stale K/V.
  void bind(const SequenceLayout& layout, const torch::Tensor& prompt) {
    if (source_identity_.defined()) {
      CHECK(source_identity_.is_same(layout.source_indices))
          << "Prefix cache layout changed";
      CHECK(prompt_identity_.is_same(prompt))
          << "Prefix cache prompt/CFG branch changed";
      return;
    }
    CHECK_GT(layout.prefix_tokens, 0);
    CHECK_GT(layout.target_tokens, 0);
    source_identity_ = layout.source_indices;
    prompt_identity_ = prompt;
    decode_layout_.target_tokens = layout.target_tokens;
    decode_layout_.prompt_tokens = layout.prompt_tokens;
    // No prefix rows are recomputed. All query rows are target-image tokens.
    decode_layout_.prefix_tokens = 0;
    decode_layout_.attention_query_offset = layout.prefix_tokens;
    decode_layout_.has_padding = layout.has_padding;
    decode_layout_.target_mask =
        layout.target_mask.slice(0, layout.prefix_tokens).clone();
    decode_layout_.rope_cos =
        layout.rope_cos.slice(0, layout.prefix_tokens).clone();
    decode_layout_.rope_sin =
        layout.rope_sin.slice(0, layout.prefix_tokens).clone();
    // Target queries may attend to every valid prefix and target key. For an
    // unpadded sequence this is full rectangular attention, requiring no mask.
    if (layout.has_padding) {
      decode_layout_.allowed_attention =
          layout.allowed_attention.slice(2, layout.prefix_tokens).clone();
    }
  }

  void mark_ready(int64_t prefix_tokens) {
    CHECK(!ready_);
    for (const auto& layer : layers_) {
      CHECK(layer.key.defined() && layer.value.defined());
      CHECK_EQ(layer.key.dim(), 4);
      CHECK_EQ(layer.key.size(2), prefix_tokens);
      CHECK(layer.key.sizes() == layer.value.sizes());
    }
    ready_ = true;  // Publish only after all layers have completed prefill.
  }

  int64_t bytes() const {
    int64_t total = 0;
    for (const auto& layer : layers_) {
      if (layer.key.defined()) {
        total += layer.key.numel() * layer.key.element_size();
        total += layer.value.numel() * layer.value.element_size();
      }
    }
    return total;
  }

 private:
  std::vector<PrefixLayerKV> layers_;
  SequenceLayout decode_layout_;
  torch::Tensor source_identity_;
  torch::Tensor prompt_identity_;
  bool ready_ = false;
};

// Sequence partitioning and Ulysses transport.

// QwenImage21 uses equal Q/K/V head counts. Ulysses redistributes whole heads,
// so any positive divisor is valid; this is not a fixed SP degree whitelist.
// Validate before allocating model weights or entering request collectives.
inline void validate_ulysses_head_partition(int64_t num_heads,
                                            int64_t sp_size) {
  CHECK_GT(num_heads, 0) << "Ulysses num_heads must be positive";
  CHECK_GT(sp_size, 0) << "Ulysses sp_size must be positive";
  CHECK_EQ(num_heads % sp_size, 0)
      << "Ulysses requires num_heads divisible by sp_size: num_heads="
      << num_heads << ", sp_size=" << sp_size;
}

// Per-request/per-CFG-branch geometry. Original prefix is replicated, only the
// target tail is padded/sharded. Coordinate tensors are never locally rebased.
class SpSequenceLayout final {
 public:
  SpSequenceLayout(const SequenceLayout& full, int32_t rank, int32_t size)
      : rank_(rank),
        size_(size),
        target_tokens_(full.target_tokens),
        prefix_tokens_(full.prefix_tokens),
        source_identity_(full.source_indices) {
    CHECK_GT(size, 1);
    CHECK_GE(rank, 0);
    CHECK_LT(rank, size);
    CHECK_GT(target_tokens_, 0);
    shard_tokens_ = (target_tokens_ + size - 1) / size;
    padding_ = shard_tokens_ * size - target_tokens_;
    local_decode_.target_tokens = shard_tokens_;
    local_decode_.prompt_tokens = full.prompt_tokens;
    local_decode_.target_mask =
        torch::ones({shard_tokens_}, full.target_mask.options());
    local_decode_.rope_cos =
        shard(full.rope_cos.slice(0, prefix_tokens_), 0, 1.0);
    local_decode_.rope_sin =
        shard(full.rope_sin.slice(0, prefix_tokens_), 0, 0.0);
    local_decode_.has_padding = full.has_padding || padding_ != 0;
    local_decode_.attention_query_offset = prefix_tokens_;
    local_prefill_ = local_decode_;
    local_prefill_.prefix_tokens = prefix_tokens_;
    local_prefill_.attention_query_offset = 0;
    local_prefill_.target_mask =
        torch::cat({full.target_mask.slice(0, 0, prefix_tokens_),
                    local_decode_.target_mask});
    local_prefill_.rope_cos = torch::cat(
        {full.rope_cos.slice(0, 0, prefix_tokens_), local_decode_.rope_cos});
    local_prefill_.rope_sin = torch::cat(
        {full.rope_sin.slice(0, 0, prefix_tokens_), local_decode_.rope_sin});
    const int64_t total = prefix_tokens_ + target_tokens_;
    if (padding_ == 0) {
      prefill_mask_ = full.allowed_attention;
    } else {
      prefill_mask_ = torch::zeros({1, 1, total + padding_, total + padding_},
                                   full.allowed_attention.options());
      prefill_mask_.slice(2, 0, total)
          .slice(3, 0, total)
          .copy_(full.allowed_attention);
      // Dummy target queries attend valid real keys; dummy keys are ALWAYS
      // masked. Avoid all-masked softmax rows and discard padded outputs later.
      prefill_mask_.slice(2, total)
          .slice(3, 0, total)
          .copy_(full.allowed_attention.slice(
              2, prefix_tokens_, prefix_tokens_ + 1));
    }
    if (full.has_padding || padding_ != 0) {
      decode_mask_ = prefill_mask_.slice(2, prefix_tokens_).contiguous();
    }
  }

  torch::Tensor shard(const torch::Tensor& target,
                      int64_t axis,
                      double pad_value = 0.0) const {
    CHECK_EQ(target.size(axis), target_tokens_);
    auto padded = target;
    if (padding_ != 0) {
      auto shape = target.sizes().vec();
      shape[axis] = padding_;
      padded = torch::cat(
          {target, torch::full(shape, pad_value, target.options())}, axis);
    }
    return padded.narrow(axis, rank_ * shard_tokens_, shard_tokens_)
        .contiguous();
  }
  torch::Tensor split_joint(const torch::Tensor& joint) const {
    CHECK_EQ(joint.size(1), prefix_tokens_ + target_tokens_);
    return torch::cat({joint.slice(1, 0, prefix_tokens_),
                       shard(joint.slice(1, prefix_tokens_), 1)},
                      1);
  }
  void validate(const SequenceLayout& full) const {
    CHECK(source_identity_.is_same(full.source_indices))
        << "SP branch/layout changed";
  }
  const SequenceLayout& local(bool decode) const {
    return decode ? local_decode_ : local_prefill_;
  }
  const torch::Tensor& attention_mask(bool decode) const {
    return decode ? decode_mask_ : prefill_mask_;
  }
  int32_t rank() const { return rank_; }
  int32_t size() const { return size_; }
  int64_t target_tokens() const { return target_tokens_; }
  int64_t prefix_tokens() const { return prefix_tokens_; }
  int64_t shard_tokens() const { return shard_tokens_; }
  int64_t padding() const { return padding_; }

 private:
  int32_t rank_;
  int32_t size_;
  int64_t target_tokens_;
  int64_t prefix_tokens_;
  int64_t shard_tokens_;
  int64_t padding_;
  torch::Tensor source_identity_;
  SequenceLayout local_prefill_;
  SequenceLayout local_decode_;
  torch::Tensor prefill_mask_;
  torch::Tensor decode_mask_;
};

class UlyssesContext final {
 public:
#if defined(USE_NPU)
  UlyssesContext(const SequenceLayout& full, ProcessGroup* group)
      : layout_(full, group->rank(), group->world_size()), group_(group) {}
#else
  UlyssesContext(const SequenceLayout& full, int32_t rank, int32_t size)
      : layout_(full, rank, size) {}
#endif
  const SpSequenceLayout& layout() const { return layout_; }

  // BNSD local-sequence/full-head -> BNSD global-sequence/local-head.
  torch::Tensor to_head_shard(const torch::Tensor& input) const {
#if defined(USE_NPU)
    return parallel_state::all_to_all_4D(input.transpose(1, 2).contiguous(),
                                         /*scatter_idx=*/2,
                                         /*gather_idx=*/1,
                                         /*async_ops=*/false,
                                         group_)()
        .transpose(1, 2);
#else
    LOG(FATAL) << "Ulysses communication requires an NPU build";
    return {};
#endif
  }
  torch::Tensor to_sequence_shard(const torch::Tensor& input) const {
#if defined(USE_NPU)
    return parallel_state::all_to_all_4D(input.transpose(1, 2).contiguous(),
                                         /*scatter_idx=*/1,
                                         /*gather_idx=*/2,
                                         /*async_ops=*/false,
                                         group_)()
        .transpose(1, 2);
#else
    LOG(FATAL) << "Ulysses communication requires an NPU build";
    return {};
#endif
  }
  torch::Tensor gather_prefix_heads(const torch::Tensor& input) const {
#if defined(USE_NPU)
    return parallel_state::gather(input.contiguous(), group_, /*dim=*/1);
#else
    LOG(FATAL) << "Ulysses communication requires an NPU build";
    return {};
#endif
  }
  torch::Tensor gather_target(const torch::Tensor& input) const {
#if defined(USE_NPU)
    return parallel_state::gather(input.contiguous(), group_, /*dim=*/1)
        .slice(1, 0, layout_.target_tokens())
        .contiguous();
#else
    LOG(FATAL) << "Ulysses communication requires an NPU build";
    return {};
#endif
  }

 private:
  SpSequenceLayout layout_;
#if defined(USE_NPU)
  ProcessGroup* group_;  // Non-owning; communicator outlives the request.
#endif
};

// Approximate request-local TaylorSeer; behavior unchanged.

inline void validate_taylorseer_options(const TaylorSeerOptions& options) {
  CHECK_GE(options.warmup_steps, 0);
  CHECK_GT(options.skip_interval_steps, 0)
      << "TaylorSeer interval must be positive";
  CHECK_GE(options.n_derivatives, 0);
  CHECK_LE(options.n_derivatives, 4)
      << "QwenImage21 TaylorSeer supports order 0..4";
}

// Approximate block-stack OUTPUT cache, before norm_out/proj_out. One object
// per request AND CFG branch AND SP rank. Fixed step-index scheduling requires
// no rank-local similarity decision and therefore cannot diverge within SP.
// With prefix KV enabled, first-step prefill is deliberately not recorded:
// its [prefix,target] output must never be used as target-only history.
class RequestTaylorSeer final {
 public:
  RequestTaylorSeer(TaylorSeerOptions options, int64_t total_steps)
      : options_(options), total_steps_(total_steps) {
    validate_taylorseer_options(options_);
    CHECK_GT(total_steps_, 0);
    derivatives_.reserve(static_cast<size_t>(options_.n_derivatives) + 1);
  }
  RequestTaylorSeer(const RequestTaylorSeer&) = delete;
  RequestTaylorSeer& operator=(const RequestTaylorSeer&) = delete;
  RequestTaylorSeer(RequestTaylorSeer&&) = default;
  RequestTaylorSeer& operator=(RequestTaylorSeer&&) = default;

  void bind(const torch::Tensor& source_indices, const torch::Tensor& prompt) {
    if (!source_identity_.defined()) {
      source_identity_ = source_indices;
      prompt_identity_ = prompt;
    } else {
      CHECK(source_identity_.is_same(source_indices))
          << "TaylorSeer layout changed";
      CHECK(prompt_identity_.is_same(prompt))
          << "TaylorSeer CFG branch changed";
    }
  }

  // A false result means execute ALL blocks; caller must then call record().
  bool begin_step(int64_t step, bool prefix_prefill) {
    CHECK(!awaiting_record_) << "Previous full step was not recorded";
    CHECK_EQ(step, current_step_ + 1)
        << "TaylorSeer needs consecutive actual denoise step ids";
    CHECK_LE(step, total_steps_);
    current_step_ = step;
    prefix_prefill_ = prefix_prefill;
    if (prefix_prefill) {
      CHECK_EQ(step, 1) << "Prefix prefill can only be the first denoise step";
      first_eligible_step_ = 2;
    }
    const int64_t anchor =
        std::max<int64_t>(options_.warmup_steps, first_eligible_step_);
    const bool full = prefix_prefill || derivatives_.empty() ||
                      step <= anchor || step == total_steps_ ||
                      options_.skip_interval_steps == 1 ||
                      (step - anchor) % options_.skip_interval_steps == 0;
    if (full) {
      ++full_steps_;
      awaiting_record_ = true;
    } else {
      ++hit_steps_;
    }
    return !full;
  }

  void record(const torch::Tensor& output) {
    CHECK(awaiting_record_);
    awaiting_record_ = false;
    if (prefix_prefill_) {
      return;
    }
    CHECK(output.defined());
    auto value = output.detach().to(torch::kFloat32).clone();
    if (!derivatives_.empty()) {
      CHECK(value.sizes() == derivatives_[0].sizes())
          << "TaylorSeer phase/shape changed";
      CHECK(value.device() == derivatives_[0].device());
    }
    std::vector<torch::Tensor> next;
    next.reserve(static_cast<size_t>(options_.n_derivatives) + 1);
    next.emplace_back(std::move(value));
    if (!derivatives_.empty()) {
      const int64_t elapsed = current_step_ - last_full_step_;
      CHECK_GT(elapsed, 0);
      for (int64_t order = 1;
           order <= options_.n_derivatives &&
           order <= static_cast<int64_t>(derivatives_.size());
           ++order) {
        next.emplace_back((next.back() - derivatives_[order - 1]) /
                          static_cast<double>(elapsed));
      }
    }
    derivatives_ = std::move(next);
    last_full_step_ = current_step_;
  }

  torch::Tensor predict(const torch::Tensor& current_input) const {
    CHECK(!awaiting_record_ && !prefix_prefill_ && !derivatives_.empty());
    CHECK(current_input.sizes() == derivatives_[0].sizes())
        << "TaylorSeer phase/shape changed";
    CHECK(current_input.device() == derivatives_[0].device());
    const int64_t elapsed = current_step_ - last_full_step_;
    CHECK_GT(elapsed, 0);
    auto result = derivatives_[0].clone();
    double coefficient = 1.0;
    for (size_t order = 1; order < derivatives_.size(); ++order) {
      coefficient *= static_cast<double>(elapsed) / static_cast<double>(order);
      result.add_(derivatives_[order] * coefficient);
    }
    return result.to(current_input.scalar_type());
  }

  int64_t full_steps() const { return full_steps_; }
  int64_t hit_steps() const { return hit_steps_; }
  int64_t available_order() const {
    return static_cast<int64_t>(derivatives_.size()) - 1;
  }
  int64_t bytes() const {
    int64_t size = 0;
    for (const auto& derivative : derivatives_) {
      size += derivative.numel() * derivative.element_size();
    }
    return size;
  }

 private:
  TaylorSeerOptions options_;
  int64_t total_steps_;
  int64_t current_step_ = 0;
  int64_t last_full_step_ = 0;
  int64_t first_eligible_step_ = 1;
  int64_t full_steps_ = 0;
  int64_t hit_steps_ = 0;
  bool prefix_prefill_ = false;
  bool awaiting_record_ = false;
  torch::Tensor source_identity_;
  torch::Tensor prompt_identity_;
  std::vector<torch::Tensor> derivatives_;
};

}  // namespace xllm::qwen_image_21

namespace xllm {

struct QwenImage21TransformerConfig {
  int64_t in_channels = 64;
  int64_t out_channels = 64;
  int64_t context_in_dim = 4096;
  int64_t num_layers = 32;
  int64_t num_heads = 32;
  int64_t head_dim = 128;
  int64_t mlp_ratio = 3;
  int64_t patch_size = 1;
  double eps = 1e-6;
  bool causal_condition = true;
  bool causal_block = true;
  std::vector<int64_t> axes_dims_rope = {16, 56, 56};
};

namespace qwen_image_21 {

class RmsNormImpl final : public torch::nn::Module {
 public:
  RmsNormImpl(int64_t dim,
              double eps,
              bool zero_centered,
              const torch::TensorOptions& options)
      : eps_(eps), zero_centered_(zero_centered) {
    weight_ = register_parameter("weight", torch::empty({dim}, options));
  }
  torch::Tensor forward(const torch::Tensor& input) {
    auto x = input.to(torch::kFloat32);
    x = x * torch::rsqrt(x.square().mean(-1, /*keepdim=*/true) + eps_);
    if (zero_centered_) {
      return (x * (weight_.to(torch::kFloat32) + 1)).to(input.scalar_type());
    }
    // Q/K norm casts BEFORE applying the checkpoint scale.
    return x.to(input.scalar_type()) * weight_;
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "weight", weight_, weight_is_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(weight_is_loaded_)
        << "weight is not loaded for " << prefix + "weight";
  }

 private:
  double eps_;
  bool zero_centered_;
  torch::Tensor weight_;
  bool weight_is_loaded_ = false;
};
TORCH_MODULE(RmsNorm);

class TextProjectionImpl final : public torch::nn::Module {
 public:
  TextProjectionImpl(int64_t context_dim,
                     int64_t dim,
                     double eps,
                     const torch::TensorOptions& options) {
    norm_ =
        register_module("text_norm", RmsNorm(context_dim, eps, true, options));
    input_ = register_module(
        "in_layer",
        layer::AddMatmul(context_dim, dim, /*with_bias=*/false, options));
    output_ = register_module(
        "out_layer", layer::AddMatmul(dim, dim, /*with_bias=*/false, options));
  }
  torch::Tensor forward(const torch::Tensor& input) {
    return output_->forward(
        torch::gelu(input_->forward(norm_->forward(input)), "tanh"));
  }

  void load_state_dict(const StateDict& state_dict) {
    norm_->load_state_dict(state_dict.get_dict_with_prefix("text_norm."));
    input_->load_state_dict(state_dict.get_dict_with_prefix("in_layer."));
    output_->load_state_dict(state_dict.get_dict_with_prefix("out_layer."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    norm_->verify_loaded_weights(prefix + "text_norm.");
    input_->verify_loaded_weights(prefix + "in_layer.");
    output_->verify_loaded_weights(prefix + "out_layer.");
  }

 private:
  RmsNorm norm_{nullptr};
  layer::AddMatmul input_{nullptr}, output_{nullptr};
};
TORCH_MODULE(TextProjection);

class TimestepEmbedderImpl final : public torch::nn::Module {
 public:
  TimestepEmbedderImpl(int64_t dim, const torch::TensorOptions& options) {
    input_ = register_module(
        "linear_1", layer::AddMatmul(256, dim, /*with_bias=*/false, options));
    output_ = register_module(
        "linear_2", layer::AddMatmul(dim, dim, /*with_bias=*/false, options));
  }
  torch::Tensor forward(const torch::Tensor& input) {
    return output_->forward(torch::silu(input_->forward(input)));
  }

  void load_state_dict(const StateDict& state_dict) {
    input_->load_state_dict(state_dict.get_dict_with_prefix("linear_1."));
    output_->load_state_dict(state_dict.get_dict_with_prefix("linear_2."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    input_->verify_loaded_weights(prefix + "linear_1.");
    output_->verify_loaded_weights(prefix + "linear_2.");
  }

 private:
  layer::AddMatmul input_{nullptr}, output_{nullptr};
};
TORCH_MODULE(TimestepEmbedder);

class TimeProjectionImpl final : public torch::nn::Module {
 public:
  TimeProjectionImpl(int64_t dim, const torch::TensorOptions& options) {
    embedder_ =
        register_module("timestep_embedder", TimestepEmbedder(dim, options));
    // CPU FP32 matches the reference table; this is not a checkpoint parameter.
    frequencies_ = torch::exp(-std::log(10000.0) *
                              torch::arange(128, torch::kFloat32) / 128);
  }
  torch::Tensor forward(const torch::Tensor& timestep,
                        const torch::Tensor& hidden) {
    auto angles = timestep.to(torch::kFloat32).unsqueeze(1) * 1000.0 *
                  frequencies_.to(timestep.device());
    auto embedding = torch::cat({torch::cos(angles), torch::sin(angles)}, -1);
    // Preserve the reference's cast through the timestep dtype.
    return embedder_->forward(
        embedding.to(timestep.scalar_type()).to(hidden.scalar_type()));
  }

  void load_state_dict(const StateDict& state_dict) {
    embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("timestep_embedder."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    embedder_->verify_loaded_weights(prefix + "timestep_embedder.");
  }

 private:
  TimestepEmbedder embedder_{nullptr};
  torch::Tensor frequencies_;
};
TORCH_MODULE(TimeProjection);

inline torch::Tensor transformer_attention(const torch::Tensor& q,
                                           const torch::Tensor& k,
                                           const torch::Tensor& v,
                                           const torch::Tensor& allowed,
                                           int64_t query_offset = 0) {
  const double scale = 1.0 / std::sqrt(static_cast<double>(q.size(-1)));
  if (q.device().is_cpu()) {
    // Explicit CPU reference path, not a recovery path for failed NPU ops.
    auto scores = torch::matmul(q.to(torch::kFloat32),
                                k.to(torch::kFloat32).transpose(-1, -2)) *
                  scale;
    if (allowed.defined()) {
      scores = scores.masked_fill(allowed.logical_not(),
                                  -std::numeric_limits<float>::infinity());
    }
    return torch::matmul(torch::softmax(scores, -1), v.to(torch::kFloat32))
        .to(q.scalar_type());
  }
#if defined(USE_NPU)
  CHECK(q.device().type() == torch::kPrivateUse1);
  CHECK(q.scalar_type() == torch::kBFloat16 ||
        q.scalar_type() == torch::kFloat16);
  CHECK_EQ(q.size(-1), 128)
      << "The initial NPU path targets the QwenImage21 checkpoint";
  // Preserve the full-sequence target's offset within CANN's query tile.
  // Starting a rectangular decode at row zero can select different BF16
  // reduction rounding than prefill when the prefix is not tile-aligned.
  // Dummy query rows have no effect on any real row (attention is row-wise);
  // discard them immediately. This pads Q only, never K/V or model tokens.
  constexpr int64_t kQueryTileRows = 128;
  CHECK_GE(query_offset, 0);
  const int64_t query_padding = query_offset % kQueryTileRows;
  auto query = q.contiguous();
  auto mask = allowed;
  if (query_padding != 0) {
    auto dummy = torch::zeros({q.size(0), q.size(1), query_padding, q.size(3)},
                              q.options());
    query = torch::cat({dummy, query}, 2);
    if (mask.defined()) {
      // Use the first real row's valid keys for dummy rows; no all-masked rows.
      auto dummy_mask = mask.slice(2, 0, 1).expand(
          {mask.size(0), mask.size(1), query_padding, mask.size(3)});
      mask = torch::cat({dummy_mask, mask}, 2);
    }
  }
  // CANN bool masks use true=blocked, unlike the reference's true=allowed.
  const std::optional<torch::Tensor> blocked =
      mask.defined()
          ? std::optional<torch::Tensor>(mask.logical_not().contiguous())
          : std::nullopt;
  auto result = at_npu::native::custom_ops::npu_fusion_attention_v3(
      query,
      k.contiguous(),
      v.contiguous(),
      q.size(1),
      "BNSD",
      /*pse=*/std::nullopt,
      /*padding_mask=*/std::nullopt,
      /*atten_mask=*/blocked,
      scale,
      /*keep_prob=*/1.0,
      /*pre_tockens=*/2147483647,
      /*next_tockens=*/2147483647);
  return std::get<0>(result).slice(2, query_padding);

#else
  LOG(FATAL) << "QwenImage21 requires an NPU build for non-CPU execution";
  return {};
#endif
}

class AttentionImpl final : public torch::nn::Module {
 public:
  AttentionImpl(int64_t heads,
                int64_t head_dim,
                double eps,
                const torch::TensorOptions& options)
      : heads_(heads), head_dim_(head_dim) {
    const int64_t dim = heads * head_dim;
    query_ = register_module(
        "to_q", layer::AddMatmul(dim, dim, /*with_bias=*/false, options));
    key_ = register_module(
        "to_k", layer::AddMatmul(dim, dim, /*with_bias=*/false, options));
    value_ = register_module(
        "to_v", layer::AddMatmul(dim, dim, /*with_bias=*/false, options));
    norm_q_ = register_module("norm_q", RmsNorm(head_dim, eps, false, options));
    norm_k_ = register_module("norm_k", RmsNorm(head_dim, eps, false, options));
    output_ = register_module("to_out",
                              torch::nn::Sequential(layer::AddMatmul(
                                  dim, dim, /*with_bias=*/false, options)));
  }
  torch::Tensor forward(const torch::Tensor& input,
                        const SequenceLayout& layout,
                        PrefixLayerKV* cache = nullptr,
                        bool decode = false,
                        const UlyssesContext* sp = nullptr) {
    const std::vector<int64_t> shape{
        input.size(0), input.size(1), heads_, head_dim_};
    auto q =
        apply_rotary(norm_q_->forward(query_->forward(input).reshape(shape)),
                     layout)
            .transpose(1, 2);
    auto k = apply_rotary(norm_k_->forward(key_->forward(input).reshape(shape)),
                          layout)
                 .transpose(1, 2);
    auto v = value_->forward(input).reshape(shape).transpose(1, 2);
    if (!decode && cache != nullptr) {
      CHECK(!cache->key.defined() && !cache->value.defined());
      cache->key = k.slice(2, 0, layout.prefix_tokens).clone().contiguous();
      cache->value = v.slice(2, 0, layout.prefix_tokens).clone().contiguous();
    }
    torch::Tensor result;
    if (sp != nullptr) {
      const auto& geometry = sp->layout();
      validate_ulysses_head_partition(heads_, geometry.size());
      const int64_t head_count = heads_ / geometry.size();
      const int64_t head_begin = geometry.rank() * head_count;
      const int64_t prefix = decode ? 0 : geometry.prefix_tokens();
      auto prefix_q =
          q.slice(2, 0, prefix).slice(1, head_begin, head_begin + head_count);
      torch::Tensor prefix_k, prefix_v;
      if (decode) {
        CHECK(cache != nullptr && cache->key.defined() &&
              cache->value.defined());
        prefix_k = cache->key.slice(1, head_begin, head_begin + head_count);
        prefix_v = cache->value.slice(1, head_begin, head_begin + head_count);
      } else {
        prefix_k =
            k.slice(2, 0, prefix).slice(1, head_begin, head_begin + head_count);
        prefix_v =
            v.slice(2, 0, prefix).slice(1, head_begin, head_begin + head_count);
      }
      auto target_q = sp->to_head_shard(q.slice(2, prefix));
      auto target_k = sp->to_head_shard(k.slice(2, prefix));
      auto target_v = sp->to_head_shard(v.slice(2, prefix));
      auto joint_q = decode ? target_q : torch::cat({prefix_q, target_q}, 2);
      auto joint_k = torch::cat({prefix_k, target_k}, 2);
      auto joint_v = torch::cat({prefix_v, target_v}, 2);
      auto head_output =
          transformer_attention(joint_q,
                                joint_k,
                                joint_v,
                                geometry.attention_mask(decode),
                                decode ? geometry.prefix_tokens() : 0);
      auto target_output = sp->to_sequence_shard(head_output.slice(2, prefix));
      if (decode) {
        result = target_output;
      } else {
        auto prefix_output =
            sp->gather_prefix_heads(head_output.slice(2, 0, prefix));
        result = torch::cat({prefix_output, target_output}, 2);
      }
    } else {
      if (decode) {
        CHECK(cache != nullptr && cache->key.defined() &&
              cache->value.defined());
        k = torch::cat({cache->key, k}, 2);
        v = torch::cat({cache->value, v}, 2);
      }
      result = transformer_attention(
          q, k, v, layout.allowed_attention, layout.attention_query_offset);
    }
    return output_->forward(result.transpose(1, 2).reshape(input.sizes()));
  }

  void load_state_dict(const StateDict& state_dict) {
    query_->load_state_dict(state_dict.get_dict_with_prefix("to_q."));
    key_->load_state_dict(state_dict.get_dict_with_prefix("to_k."));
    value_->load_state_dict(state_dict.get_dict_with_prefix("to_v."));
    norm_q_->load_state_dict(state_dict.get_dict_with_prefix("norm_q."));
    norm_k_->load_state_dict(state_dict.get_dict_with_prefix("norm_k."));
    output_->at<layer::AddMatmulImpl>(0).load_state_dict(
        state_dict.get_dict_with_prefix("to_out.0."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    query_->verify_loaded_weights(prefix + "to_q.");
    key_->verify_loaded_weights(prefix + "to_k.");
    value_->verify_loaded_weights(prefix + "to_v.");
    norm_q_->verify_loaded_weights(prefix + "norm_q.");
    norm_k_->verify_loaded_weights(prefix + "norm_k.");
    output_->at<layer::AddMatmulImpl>(0).verify_loaded_weights(prefix +
                                                               "to_out.0.");
  }

 private:
  int64_t heads_, head_dim_;
  layer::AddMatmul query_{nullptr}, key_{nullptr}, value_{nullptr};
  RmsNorm norm_q_{nullptr}, norm_k_{nullptr};
  torch::nn::Sequential output_{nullptr};
};
TORCH_MODULE(Attention);

class SwiGluImpl final : public torch::nn::Module {
 public:
  SwiGluImpl(int64_t dim, int64_t hidden, const torch::TensorOptions& options) {
    projection_ = register_module(
        "proj", layer::AddMatmul(dim, hidden, /*with_bias=*/false, options));
    gate_ = register_module(
        "gate_layer",
        layer::AddMatmul(dim, hidden, /*with_bias=*/false, options));
    output_ = register_module(
        "out", layer::AddMatmul(hidden, dim, /*with_bias=*/false, options));
  }
  torch::Tensor forward(const torch::Tensor& input) {
    return output_->forward(torch::silu(gate_->forward(input)) *
                            projection_->forward(input));
  }

  void load_state_dict(const StateDict& state_dict) {
    projection_->load_state_dict(state_dict.get_dict_with_prefix("proj."));
    gate_->load_state_dict(state_dict.get_dict_with_prefix("gate_layer."));
    output_->load_state_dict(state_dict.get_dict_with_prefix("out."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    projection_->verify_loaded_weights(prefix + "proj.");
    gate_->verify_loaded_weights(prefix + "gate_layer.");
    output_->verify_loaded_weights(prefix + "out.");
  }

 private:
  layer::AddMatmul projection_{nullptr}, gate_{nullptr}, output_{nullptr};
};
TORCH_MODULE(SwiGlu);

inline torch::Tensor select_modulation(const torch::Tensor& parameters,
                                       const torch::Tensor& target_mask,
                                       bool causal_condition) {
  if (!causal_condition) {
    return parameters.unsqueeze(1);
  }
  CHECK_EQ(parameters.size(0), 2);  // batch=1 plus the t=0 row
  return torch::where(target_mask.reshape({1, -1, 1}),
                      parameters.slice(0, 0, 1).unsqueeze(1),
                      parameters.slice(0, 1, 2).unsqueeze(1));
}

class TransformerBlockImpl final : public torch::nn::Module {
 public:
  TransformerBlockImpl(const QwenImage21TransformerConfig& config,
                       const torch::TensorOptions& options)
      : dim_(config.num_heads * config.head_dim),
        eps_(config.eps),
        causal_condition_(config.causal_condition) {
    attention_ = register_module(
        "attn",
        Attention(config.num_heads, config.head_dim, config.eps, options));
    mlp_ = register_module("img_mlp",
                           SwiGlu(dim_, dim_ * config.mlp_ratio, options));
  }
  torch::Tensor forward(torch::Tensor hidden,
                        const torch::Tensor& modulation,
                        const SequenceLayout& layout,
                        PrefixLayerKV* cache = nullptr,
                        bool decode = false,
                        const UlyssesContext* sp = nullptr) {
    auto parts = modulation.chunk(4, -1);
    auto scale1 =
        select_modulation(parts[0], layout.target_mask, causal_condition_);
    auto gate1 =
        select_modulation(parts[1], layout.target_mask, causal_condition_);
    auto scale2 =
        select_modulation(parts[2], layout.target_mask, causal_condition_);
    auto gate2 =
        select_modulation(parts[3], layout.target_mask, causal_condition_);
    auto norm_options =
        torch::nn::functional::LayerNormFuncOptions({dim_}).eps(eps_);
    auto normalized =
        torch::nn::functional::layer_norm(hidden, norm_options) * (1 + scale1);
    hidden = hidden + gate1.tanh() * attention_->forward(
                                         normalized, layout, cache, decode, sp);
    normalized =
        torch::nn::functional::layer_norm(hidden, norm_options) * (1 + scale2);
    hidden = hidden + gate2.tanh() * mlp_->forward(normalized);
    return hidden.scalar_type() == torch::kFloat16 ? hidden.clamp(-65504, 65504)
                                                   : hidden;
  }

  void load_state_dict(const StateDict& state_dict) {
    attention_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
    mlp_->load_state_dict(state_dict.get_dict_with_prefix("img_mlp."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    attention_->verify_loaded_weights(prefix + "attn.");
    mlp_->verify_loaded_weights(prefix + "img_mlp.");
  }

 private:
  int64_t dim_;
  double eps_;
  bool causal_condition_;
  Attention attention_{nullptr};
  SwiGlu mlp_{nullptr};
};
TORCH_MODULE(TransformerBlock);

class OutputNormImpl final : public torch::nn::Module {
 public:
  OutputNormImpl(int64_t dim,
                 double eps,
                 bool causal_condition,
                 const torch::TensorOptions& options)
      : dim_(dim), eps_(eps), causal_condition_(causal_condition) {
    linear_ = register_module(
        "linear", layer::AddMatmul(dim, dim, /*with_bias=*/false, options));
  }
  torch::Tensor forward(const torch::Tensor& hidden,
                        const torch::Tensor& time,
                        const SequenceLayout& layout) {
    auto scale = select_modulation(linear_->forward(torch::silu(time)),
                                   layout.target_mask,
                                   causal_condition_);
    return torch::nn::functional::layer_norm(
               hidden,
               torch::nn::functional::LayerNormFuncOptions({dim_}).eps(eps_)) *
           (1 + scale);
  }

  void load_state_dict(const StateDict& state_dict) {
    linear_->load_state_dict(state_dict.get_dict_with_prefix("linear."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    linear_->verify_loaded_weights(prefix + "linear.");
  }

 private:
  int64_t dim_;
  double eps_;
  bool causal_condition_;
  layer::AddMatmul linear_{nullptr};
};
TORCH_MODULE(OutputNorm);

}  // namespace qwen_image_21

// Component-only implementation. The pipeline owns preprocessing, scheduler,
// CFG, device movement and per-request layouts; no VLM is constructed here.
class QwenImage21Transformer2DModelImpl final : public torch::nn::Module {
 public:
  explicit QwenImage21Transformer2DModelImpl(
      QwenImage21TransformerConfig config,
      const torch::TensorOptions& options =
          torch::TensorOptions().dtype(torch::kFloat32))
      : config_(std::move(config)) {
    CHECK_EQ(config_.patch_size, 1);
    CHECK_GT(config_.num_heads, 0);
    CHECK_GT(config_.num_layers, 0);
    CHECK_GT(config_.mlp_ratio, 0);
    CHECK_GT(config_.eps, 0);
    CHECK_EQ(std::accumulate(config_.axes_dims_rope.begin(),
                             config_.axes_dims_rope.end(),
                             int64_t{0}),
             config_.head_dim);
    const int64_t dim = config_.num_heads * config_.head_dim;
    image_in_ = register_module(
        "img_in",
        layer::AddMatmul(
            config_.in_channels, dim, /*with_bias=*/false, options));
    text_in_ =
        register_module("txt_in",
                        qwen_image_21::TextProjection(
                            config_.context_in_dim, dim, config_.eps, options));
    time_ = register_module("time_text_embed",
                            qwen_image_21::TimeProjection(dim, options));
    modulation_ = register_module(
        "modulation",
        torch::nn::Sequential(
            torch::nn::SiLU(),
            layer::AddMatmul(dim, 4 * dim, /*with_bias=*/false, options)));
    blocks_ = register_module("transformer_blocks", torch::nn::ModuleList());
    for (int64_t i = 0; i < config_.num_layers; ++i) {
      blocks_->push_back(qwen_image_21::TransformerBlock(config_, options));
    }
    norm_ = register_module(
        "norm_out",
        qwen_image_21::OutputNorm(
            dim, config_.eps, config_.causal_condition, options));
    output_ = register_module(
        "proj_out",
        layer::AddMatmul(
            dim, config_.out_channels, /*with_bias=*/false, options));
  }

  qwen_image_21::SequenceLayout prepare_layout(
      const torch::Tensor& image_pad_mask,
      const torch::Tensor& prompt_mask,
      const std::vector<qwen_image_21::ImageShape>& shapes,
      const torch::Device& device) const {
    return qwen_image_21::prepare_sequence_layout(image_pad_mask,
                                                  prompt_mask,
                                                  shapes,
                                                  config_.axes_dims_rope,
                                                  device,
                                                  config_.causal_block);
  }

  // Prefill/uncached: hidden is [condition latents, target latents]. Cached
  // decode: hidden contains only the target. The same branch layout and prompt
  // tensor must be supplied throughout the request. Return target noise only.
  torch::Tensor forward(const torch::Tensor& hidden,
                        const torch::Tensor& encoder_hidden,
                        const torch::Tensor& timestep,
                        const qwen_image_21::SequenceLayout& layout,
                        qwen_image_21::PrefixKVCache* cache = nullptr,
                        const qwen_image_21::UlyssesContext* sp = nullptr,
                        qwen_image_21::RequestTaylorSeer* taylorseer = nullptr,
                        int64_t step_id = 0) {
    CHECK_EQ(hidden.dim(), 3);
    CHECK_EQ(hidden.size(0), 1);
    CHECK_EQ(hidden.size(2), config_.in_channels);
    CHECK_EQ(encoder_hidden.dim(), 3);
    CHECK_EQ(encoder_hidden.size(0), 1);
    CHECK_EQ(encoder_hidden.size(1), layout.prompt_tokens);
    CHECK_EQ(encoder_hidden.size(2), config_.context_in_dim);
    CHECK_EQ(timestep.numel(), 1);
    CHECK(hidden.device() == encoder_hidden.device());
    CHECK(hidden.device() == layout.source_indices.device());
    if (sp != nullptr) {
      qwen_image_21::validate_ulysses_head_partition(config_.num_heads,
                                                     sp->layout().size());
    }
    if (cache != nullptr) {
      CHECK(config_.causal_condition && config_.causal_block)
          << "Prefix KV cache requires causal_condition and causal_block";
      CHECK_EQ(cache->num_layers(), config_.num_layers);
      cache->bind(layout, encoder_hidden);
    }
    const bool decode = cache != nullptr && cache->ready();
    CHECK_EQ(hidden.size(1),
             decode ? layout.target_tokens : layout.image_positions.numel());
    if (sp != nullptr) {
      sp->layout().validate(layout);
    }
    const auto& active_layout =
        sp != nullptr ? sp->layout().local(decode)
                      : (decode ? cache->decode_layout() : layout);
    torch::Tensor joint;
    if (decode) {
      // No text projection, condition-image projection, or prefix FFN work.
      joint = image_in_->forward(sp != nullptr ? sp->layout().shard(hidden, 1)
                                               : hidden);
    } else {
      auto text = text_in_->forward(encoder_hidden);
      joint =
          torch::cat({text,
                      torch::zeros({1, layout.target_tokens / 4, text.size(2)},
                                   text.options())},
                     1)
              .index_select(1, layout.source_indices);
      joint.index_copy_(1, layout.image_positions, image_in_->forward(hidden));
      if (sp != nullptr) {
        joint = sp->layout().split_joint(joint);
      }
    }
    auto t = timestep.to(hidden.options()).reshape({1});
    if (config_.causal_condition) {
      t = torch::cat({t, torch::zeros_like(t)});
    }
    auto time = time_->forward(t, joint);
    auto modulation = modulation_->forward(time);
    bool approximate = false;
    if (taylorseer != nullptr) {
      taylorseer->bind(layout.source_indices, encoder_hidden);
      approximate = taylorseer->begin_step(
          step_id, /*prefix_prefill=*/cache != nullptr && !decode);
    }
    if (approximate) {
      joint = taylorseer->predict(joint);
    } else {
      for (int64_t index = 0; index < config_.num_layers; ++index) {
        auto* layer_cache = cache == nullptr ? nullptr : &cache->layer(index);
        joint = blocks_->at<qwen_image_21::TransformerBlockImpl>(index).forward(
            std::move(joint),
            modulation,
            active_layout,
            layer_cache,
            decode,
            sp);
      }
      if (taylorseer != nullptr) {
        taylorseer->record(joint);
      }
    }
    auto normalized = norm_->forward(joint, time, active_layout);
    auto result =
        output_->forward(normalized.slice(1, active_layout.prefix_tokens));
    if (cache != nullptr && !decode) {
      cache->mark_ready(layout.prefix_tokens);
    }
    return sp != nullptr ? sp->gather_target(result) : result;
  }

  const QwenImage21TransformerConfig& config() const { return config_; }

  void load_state_dict(const StateDict& state_dict) {
    image_in_->load_state_dict(state_dict.get_dict_with_prefix("img_in."));
    text_in_->load_state_dict(state_dict.get_dict_with_prefix("txt_in."));
    time_->load_state_dict(state_dict.get_dict_with_prefix("time_text_embed."));
    modulation_->at<layer::AddMatmulImpl>(1).load_state_dict(
        state_dict.get_dict_with_prefix("modulation.1."));
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm_out."));
    output_->load_state_dict(state_dict.get_dict_with_prefix("proj_out."));
    for (size_t i = 0; i < blocks_->size(); ++i) {
      blocks_->at<qwen_image_21::TransformerBlockImpl>(i).load_state_dict(
          state_dict.get_dict_with_prefix("transformer_blocks." +
                                          std::to_string(i) + "."));
    }
  }

  void verify_loaded_weights(const std::string& prefix = "") const {
    image_in_->verify_loaded_weights(prefix + "img_in.");
    text_in_->verify_loaded_weights(prefix + "txt_in.");
    time_->verify_loaded_weights(prefix + "time_text_embed.");
    modulation_->at<layer::AddMatmulImpl>(1).verify_loaded_weights(
        prefix + "modulation.1.");
    norm_->verify_loaded_weights(prefix + "norm_out.");
    output_->verify_loaded_weights(prefix + "proj_out.");
    for (size_t i = 0; i < blocks_->size(); ++i) {
      blocks_->at<qwen_image_21::TransformerBlockImpl>(i).verify_loaded_weights(
          prefix + "transformer_blocks." + std::to_string(i) + ".");
    }
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    CHECK(loader != nullptr);
    for (const auto& state_dict : loader->get_state_dicts()) {
      load_state_dict(*state_dict);
    }
    verify_loaded_weights("");
  }

 private:
  QwenImage21TransformerConfig config_;
  layer::AddMatmul image_in_{nullptr}, output_{nullptr};
  qwen_image_21::TextProjection text_in_{nullptr};
  qwen_image_21::TimeProjection time_{nullptr};
  qwen_image_21::OutputNorm norm_{nullptr};
  torch::nn::Sequential modulation_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
};
TORCH_MODULE(QwenImage21Transformer2DModel);

}  // namespace xllm
