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

#include <ATen/CPUGeneratorImpl.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "core/framework/config/dit_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/dit_model_context.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_args.h"
#include "core/framework/parallel_state/mapping_npu.h"
#include "core/framework/parallel_state/parallel_state.h"
#include "core/framework/request/dit_request_state.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/layers/common/rotary_embedding_util.h"
#include "core/runtime/dit_forward_params.h"
#include "models/dit/autoencoders/autoencoder_kl_qwenimage21.h"
#include "models/dit/schedulers/flowmatch_euler_discrete_scheduler.h"
#include "models/dit/transformers/transformer_qwen_image_21.h"
#include "models/model_registry.h"
#include "models/vlm/npu/qwen3_vl.h"
#include "processors/qwen2_vl_image_processor.h"
#include "processors/qwen3_vl_prompt_processor.h"

namespace xllm {

// Prompt types, image geometry and tokenization.

using QwenImage21ImageSize = std::array<int64_t, 2>;  // width, height

// Shared pixel geometry for condition and output images. Aspect-ratio
// restrictions apply only to the internal VLM path.
inline bool is_valid_qwen_image_21_image_size(int64_t width, int64_t height) {
  return width >= 32 && height >= 32 && width <= 32768 && height <= 32768 &&
         width % 32 == 0 && height % 32 == 0;
}

struct QwenImage21PromptTokens {
  std::vector<int32_t> ids;
  std::vector<int64_t> image_mask;  // Full sequence, including system prefix.
  int64_t drop_idx = 0;
};

inline std::string qwen_image_21_system_prefix() {
  return "<|im_start|>system\nComprehend and analyze the provided prompt."
         "<|im_end|>\n";
}

// Worker-side tokenization. Do not send token IDs through DiTTensorSources:
// that transport casts all tensors to the inference dtype.
// Image count and sizes are validated once by the pipeline before branching.
inline std::string prepare_qwen_image_21_prompt_tokens(
    const Tokenizer& tokenizer,
    const std::string& prompt,
    const std::vector<QwenImage21ImageSize>& sizes,
    QwenImage21PromptTokens& output) {
  output = {};
  for (const char* reserved : {"<|im_start|>",
                               "<|im_end|>",
                               "<|vision_start|>",
                               "<|vision_end|>",
                               "<|image_pad|>",
                               "<|video_pad|>"}) {
    if (prompt.find(reserved) != std::string::npos) {
      return "QwenImage21 prompt contains a reserved multimodal/chat token";
    }
  }
  const auto image_id = tokenizer.token_to_id("<|image_pad|>");
  if (!image_id.has_value()) {
    return "QwenImage21 tokenizer has no image_pad token";
  }
  std::vector<int64_t> slots;
  slots.reserve(sizes.size());
  int64_t image_positions = 0;
  int64_t image_slots = 0;
  std::string vision_prefix;
  for (size_t i = 0; i < sizes.size(); ++i) {
    const int64_t width = sizes[i][0];
    const int64_t height = sizes[i][1];
    const int64_t count = (width / 32) * (height / 32);
    slots.emplace_back(count);
    image_slots += count;
    image_positions += std::max(width, height) / 16;
    vision_prefix += (i == 0 ? "" : " ") + std::string("<image") +
                     std::to_string(i + 1) +
                     "><|vision_start|><|image_pad|><|vision_end|>";
  }
  const std::string system = qwen_image_21_system_prefix();
  const std::string suffix = "<|im_end|>\n<|im_start|>assistant\n";
  const std::string text = system + "<|im_start|>user\n" + vision_prefix +
                           (prompt.empty() ? " " : prompt) + suffix;
  std::vector<int32_t> prefix_ids, ids;
  if (!tokenizer.encode(system, &prefix_ids, /*add_special_tokens=*/false) ||
      !tokenizer.encode(text, &ids, /*add_special_tokens=*/false)) {
    return "QwenImage21 prompt tokenization failed";
  }
  if (prefix_ids.empty() || ids.size() <= prefix_ids.size() ||
      !std::equal(prefix_ids.begin(), prefix_ids.end(), ids.begin())) {
    return "QwenImage21 tokenizer returned an invalid system prefix";
  }
  if (static_cast<size_t>(std::count(ids.begin(), ids.end(), *image_id)) !=
      sizes.size()) {
    return "QwenImage21 image placeholders do not match condition images";
  }
  const int64_t length = static_cast<int64_t>(ids.size() - prefix_ids.size()) +
                         image_slots - static_cast<int64_t>(sizes.size());
  if (length - image_slots + image_positions >= 8192) {
    return "QwenImage21 prompt exceeds the reference RoPE position range";
  }
  output.drop_idx = static_cast<int64_t>(prefix_ids.size());
  output.ids.reserve(static_cast<size_t>(length + output.drop_idx));
  output.image_mask.reserve(static_cast<size_t>(length + output.drop_idx));
  size_t image_index = 0;
  for (const int32_t id : ids) {
    const bool is_image = id == *image_id;
    const int64_t count = is_image ? slots[image_index++] : 1;
    output.ids.insert(output.ids.end(), static_cast<size_t>(count), id);
    output.image_mask.insert(
        output.image_mask.end(), static_cast<size_t>(count), is_image ? 1 : 0);
  }
  return {};
}

// External prompt embeddings and generation parameters.

struct QwenImage21PromptInput {
  torch::Tensor embeddings;  // [1,L,4096]
  torch::Tensor valid_mask;  // CPU bool [1,L]
  torch::Tensor image_mask;  // CPU bool [1,L], before 4x expansion
};

// Empty string means success. Clients supply unbatched [L,D] embeddings.
// The pipeline receives [1,L,D] after DiTBatch adds the request batch
// dimension. Image sizes have already been validated by the pipeline;
// this function checks each branch's embeddings and metadata against them.
inline std::string read_qwen_image_21_prompt(
    const NamedTensor& source,
    const std::vector<QwenImage21ImageSize>& sizes,
    QwenImage21PromptInput& output) {
  auto tensor = source.tensor;
  if (tensor.dim() == 2) {
    tensor = tensor.unsqueeze(0);
  }
  if (tensor.dim() != 3 || tensor.size(0) != 1 || tensor.size(2) != 4096 ||
      tensor.size(1) <= 0) {
    return source.name + " must be [1,L,4096] (or unbatched [L,4096])";
  }
  if (!tensor.is_floating_point() ||
      !torch::isfinite(tensor).all().item<bool>()) {
    return source.name + " must contain finite floating point embeddings";
  }
  const int64_t length = tensor.size(1);
  std::vector<int64_t> valid(static_cast<size_t>(length), 1);
  std::vector<int64_t> image(static_cast<size_t>(length), 0);
  auto read_mask = [&](const std::string& name,
                       std::vector<int64_t>& values,
                       bool required) -> bool {
    const auto item = source.parameters.find(name);
    if (item == source.parameters.end()) {
      return !required;
    }
    const auto* data = std::get_if<std::vector<int64_t>>(&item->second);
    if (data == nullptr || static_cast<int64_t>(data->size()) != length ||
        !std::all_of(data->begin(), data->end(), [](int64_t v) {
          return v == 0 || v == 1;
        })) {
      return false;
    }
    values = *data;
    return true;
  };
  if (!read_mask("prompt_embeds_mask", valid, false) ||
      !read_mask("image_pad_mask", image, !sizes.empty())) {
    return source.name +
           " has a missing/invalid mask parameter (expected L values of 0/1)";
  }
  std::vector<int64_t> expected_sizes, slot_counts;
  expected_sizes.reserve(sizes.size() * 2);
  slot_counts.reserve(sizes.size());
  for (const auto& size : sizes) {
    expected_sizes.emplace_back(size[0]);
    expected_sizes.emplace_back(size[1]);
    slot_counts.emplace_back((size[0] / 32) * (size[1] / 32));
  }
  const auto declared = source.parameters.find("input_image_sizes");
  if (declared != source.parameters.end()) {
    const auto* data = std::get_if<std::vector<int64_t>>(&declared->second);
    if (data == nullptr || *data != expected_sizes) {
      return source.name +
             " input_image_sizes must match the uploaded images in [w,h] order";
    }
  } else if (!sizes.empty()) {
    return source.name +
           " requires input_image_sizes for externally encoded condition "
           "images";
  }
  if (valid.front() != 1) {
    return source.name +
           " must be nonempty and right-padded (strip VLM left padding first)";
  }
  bool padding = false;
  size_t block = 0;
  int64_t consumed = 0;
  int64_t position = 0;
  for (int64_t i = 0; i < length; ++i) {
    if ((padding && valid[i]) || (image[i] && !valid[i])) {
      return source.name +
             " contains left/interior padding or a padded image slot";
    }
    padding = padding || !valid[i];
    if (image[i]) {
      if (block >= slot_counts.size()) {
        return source.name + " has too many image slots";
      }
      ++consumed;
      if (consumed == slot_counts[block]) {
        position += std::max(sizes[block][0], sizes[block][1]) / 16;
        consumed = 0;
        ++block;
      }
    } else {
      if (consumed != 0) {
        return source.name + " splits an image block with a text token";
      }
      ++position;
    }
  }
  if (block != slot_counts.size() || consumed != 0) {
    return source.name +
           " image slots do not match the condition-image latent grid";
  }
  if (position >= 8192) {
    return source.name + " exceeds the reference RoPE position range";
  }
  output = {
      tensor,
      torch::tensor(
          valid, torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
          .to(torch::kBool)
          .unsqueeze(0),
      torch::tensor(
          image, torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
          .to(torch::kBool)
          .unsqueeze(0)};
  return {};
}

inline std::string validate_qwen_image_21_generation(
    const DiTGenerationParams& params) {
  if (!is_valid_qwen_image_21_image_size(params.width, params.height)) {
    return "QwenImage21 output width/height must be positive multiples of 32";
  }
  if (params.num_inference_steps < 2) {
    return "QwenImage21 requires at least 2 steps with shift_terminal=0.02";
  }
  if (params.num_images_per_prompt != 1 || params.seed < 0 ||
      !std::isfinite(params.true_cfg_scale) || params.true_cfg_scale < 1.0f) {
    return "QwenImage21 requires num_images_per_prompt=1, seed>=0 and finite "
           "true_cfg_scale>=1";
  }
  return {};
}

}  // namespace xllm

namespace xllm::qwen_image_21 {

inline torch::Tensor pack_latents(const torch::Tensor& latent) {
  CHECK_EQ(latent.dim(), 5);
  CHECK_EQ(latent.size(2), 1);
  return latent.flatten(2).transpose(1, 2).contiguous();
}

inline torch::Tensor unpack_latents(const torch::Tensor& latent,
                                    int64_t height,
                                    int64_t width) {
  CHECK_EQ(latent.dim(), 3);
  CHECK_GT(height, 0);
  CHECK_GT(width, 0);
  CHECK_EQ(latent.size(1), height * width);
  return latent.transpose(1, 2).reshape(
      {latent.size(0), latent.size(2), 1, height, width});
}

// Pure CFG branch routing and noise composition.

// Pure branch routing, independent of process-group ownership. cfg_size=1
// computes both branches locally; cfg_size=2 uses rank0=cond, rank1=uncond.
class CfgExecutionPlan final {
 public:
  CfgExecutionPlan(int32_t cfg_size, int32_t cfg_rank, bool use_cfg)
      : cfg_size_(cfg_size), cfg_rank_(cfg_rank), use_cfg_(use_cfg) {
    CHECK(cfg_size == 1 || cfg_size == 2);
    CHECK_GE(cfg_rank, 0);
    CHECK_LT(cfg_rank, cfg_size);
  }
  bool leader() const { return cfg_rank_ == 0; }
  bool distributed() const { return cfg_size_ == 2 && use_cfg_; }
  bool positive() const { return leader(); }
  bool negative() const {
    return use_cfg_ && (cfg_size_ == 1 || cfg_rank_ == 1);
  }
  bool idle() const { return !positive() && !negative(); }

 private:
  int32_t cfg_size_;
  int32_t cfg_rank_;
  bool use_cfg_;
};

inline torch::Tensor combine_cfg_noise(const torch::Tensor& positive,
                                       const torch::Tensor& negative,
                                       float scale) {
  CHECK(positive.sizes() == negative.sizes());
  CHECK(positive.device() == negative.device());
  CHECK(positive.scalar_type() == negative.scalar_type());
  CHECK(std::isfinite(scale) && scale > 1.0f);
  // Keep BF16 operation order identical to the single-device baseline. In
  // particular, do NOT copy EditPlus's cond_norm / noise_norm renormalization.
  return negative + scale * (positive - negative);
}

// all-gather along dimension 0 of two [1,N,64] branch outputs. Ordering is
// group-local rank order, not completion order or a presumed global device id.
inline torch::Tensor combine_gathered_cfg_noise(const torch::Tensor& gathered,
                                                float scale) {
  CHECK_EQ(gathered.dim(), 3);
  CHECK_EQ(gathered.size(0), 2);
  return combine_cfg_noise(
      gathered.slice(0, 0, 1), gathered.slice(0, 1, 2), scale);
}

}  // namespace xllm::qwen_image_21

namespace xllm {

// Checkpoint argument conversion and registration.

inline QwenImage21TransformerConfig qwen_image_21_transformer_config(
    const ModelArgs& args) {
  QwenImage21TransformerConfig config;
  config.in_channels = args.in_channels();
  config.out_channels = args.out_channels();
  config.context_in_dim = args.context_in_dim();
  config.num_layers = args.num_layers();
  config.num_heads = args.n_heads();
  config.head_dim = args.head_dim();
  CHECK_EQ(args.mlp_ratio(), static_cast<int64_t>(args.mlp_ratio()));
  config.mlp_ratio = static_cast<int64_t>(args.mlp_ratio());
  config.patch_size = args.patch_size();
  config.eps = args.eps();
  config.causal_condition = args.causal_condition();
  config.causal_block = args.causal_block();
  config.axes_dims_rope = args.axes_dims_rope();
  return config;
}

inline QwenImage21VaeConfig qwen_image_21_vae_config(const ModelArgs& args) {
  QwenImage21VaeConfig config;
  config.in_channels = args.in_channels();
  config.out_channels = args.out_channels();
  config.base_dim = args.base_dim();
  config.decoder_base_dim = args.decoder_base_dim();
  config.z_dim = args.z_dim();
  config.num_res_blocks = args.num_res_blocks();
  config.is_residual = args.vae_is_residual();
  config.patch_size = args.patch_size();
  config.dim_mult = args.dim_mult();
  config.temperal_downsample = args.temperal_downsample();
  config.latents_mean = args.latents_mean();
  config.latents_std = args.latents_std();
  CHECK(args.attn_scales().empty())
      << "Only the checkpoint's empty attn_scales is supported";
  CHECK_EQ(args.vae_scale_factor_spatial(),
           int64_t{1} << config.temperal_downsample.size());
  CHECK_EQ(static_cast<int64_t>(config.latents_mean.size()), config.z_dim);
  CHECK_EQ(static_cast<int64_t>(config.latents_std.size()), config.z_dim);
  return config;
}

// Component argument registration only. Do not advertise a pipeline factory
// before its scheduler, external-embedding request contract and serving path
// exist.
REGISTER_MODEL_ARGS(QwenImage21Transformer2DModel, [&] {
  LOAD_ARG_OR(model_type, "_class_name", "QwenImage21Transformer2DModel");
  LOAD_ARG_OR(dtype, "dtype", "bfloat16");
  LOAD_ARG_OR(in_channels, "in_channels", 64);
  LOAD_ARG_OR(out_channels, "out_channels", 64);
  LOAD_ARG_OR(context_in_dim, "context_in_dim", 4096);
  LOAD_ARG_OR(num_layers, "num_layers", 32);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  LOAD_ARG_OR(head_dim, "attention_head_dim", 128);
  LOAD_ARG_OR(mlp_ratio, "mlp_ratio", 3.0f);
  LOAD_ARG_OR(patch_size, "patch_size", 1);
  LOAD_ARG_OR(eps, "eps", 1e-6);
  LOAD_ARG_OR(causal_condition, "causal_condition", true);
  LOAD_ARG_OR(causal_block, "causal_block", true);
  LOAD_ARG_OR(
      axes_dims_rope, "axes_dims_rope", (std::vector<int64_t>{16, 56, 56}));
});

REGISTER_MODEL_ARGS(AutoencoderKLQwenImage21, [&] {
  LOAD_ARG_OR(model_type, "_class_name", "AutoencoderKLQwenImage21");
  LOAD_ARG_OR(dtype, "dtype", "float32");
  LOAD_ARG_OR(in_channels, "in_channels", 4);
  LOAD_ARG_OR(out_channels, "out_channels", 4);
  LOAD_ARG_OR(base_dim, "base_dim", 96);
  LOAD_ARG_OR(decoder_base_dim, "decoder_base_dim", 144);
  LOAD_ARG_OR(z_dim, "z_dim", 64);
  LOAD_ARG_OR(num_res_blocks, "num_res_blocks", 2);
  LOAD_ARG_OR(vae_is_residual, "is_residual", true);
  // JsonReader treats null as absent: null becomes the no-patch value 1.
  LOAD_ARG_OR(patch_size, "patch_size", 1);
  LOAD_ARG_OR(vae_scale_factor_spatial, "scale_factor_spatial", 16);
  LOAD_ARG_OR(vae_scale_factor_temporal, "scale_factor_temporal", 8);
  LOAD_ARG_OR(dim_mult, "dim_mult", (std::vector<int64_t>{1, 2, 4, 8, 8}));
  LOAD_ARG_OR(temperal_downsample,
              "temperal_downsample",
              (std::vector<bool>{false, true, true, true}));
  LOAD_ARG_OR(attn_scales, "attn_scales", (std::vector<double>{}));
  LOAD_ARG(latents_mean, "latents_mean");
  LOAD_ARG(latents_std, "latents_std");
});

// Eager, batch=1 pipeline with CFG/SP parallelism. Prompt inputs come from
// external embeddings or the existing rank-local ATB Qwen3VL component.
// All ranks run VAE encode/decode within their own VAE groups and initialize
// request-seeded CPU latents locally.
// Prefix KV caching and TaylorSeer remain request-local.
class QwenImage21PipelineImpl final : public torch::nn::Module {
 public:
  explicit QwenImage21PipelineImpl(const DiTModelContext& context)
      : options_(context.get_tensor_options()),
        scheduler_context_(context.get_model_context("scheduler")) {
    const auto& parallel = context.get_parallel_args();
    cfg_size_ = parallel.cfg_size();
    CHECK(cfg_size_ == 1 || cfg_size_ == 2)
        << "QwenImage21 supports cfg_size=1 or 2";
    CHECK_EQ(parallel.tp_size(), 1);
    sp_size_ = parallel.sp_size();
    qwen_image_21::validate_ulysses_head_partition(
        context.get_model_args("transformer").n_heads(), sp_size_);
    CHECK_EQ(parallel.dp_size(), 1);
    vae_size_ = parallel.vae_size();
    CHECK_GT(vae_size_, 0);
    CHECK_EQ(parallel.world_size() % vae_size_, 0)
        << "vae_size must divide the CFG x SP worker count";
    CHECK_EQ(parallel.text_encoder_tp_size(), 1);
    CHECK_EQ(static_cast<int64_t>(parallel.world_size()),
             static_cast<int64_t>(cfg_size_) * sp_size_)
        << "Only CFG x SP supported";
    sp_group_ = parallel.dit_sp_group_;
    if (sp_size_ > 1) {
      CHECK(sp_group_ != nullptr);
      CHECK_EQ(sp_group_->world_size(), sp_size_);
      sp_rank_ = sp_group_->rank();
    }
    cfg_group_ = parallel.dit_cfg_group_;
    cfg_rank_ = 0;
    if (cfg_size_ == 2) {
      CHECK(cfg_group_ != nullptr);
      CHECK_EQ(cfg_group_->world_size(), 2);
      cfg_rank_ = cfg_group_->rank();
    }
    CHECK_EQ(static_cast<int64_t>(parallel.rank()),
             static_cast<int64_t>(cfg_rank_) * sp_size_ + sp_rank_)
        << "Expected SP then CFG rank order";
    is_driver_ = parallel.rank() == 0;
    const auto& cache_config = context.get_dit_config();
    CHECK(cache_config.selected_policy == PolicyType::None ||
          cache_config.selected_policy == PolicyType::TaylorSeer)
        << "QwenImage21 supports only None/TaylorSeer DiT cache policies";
    use_taylorseer_ = cache_config.selected_policy == PolicyType::TaylorSeer;
    taylorseer_options_ = cache_config.taylorseer;
    if (use_taylorseer_) {
      qwen_image_21::validate_taylorseer_options(taylorseer_options_);
    }
    if (!options_.device().is_cpu()) {
      const torch::ScalarType input_dtype = options_.dtype().toScalarType();
      CHECK(input_dtype == torch::kBFloat16 || input_dtype == torch::kFloat32)
          << "QwenImage21 requires BF16/FP32 input transport; avoid a lossy "
             "FP16 cast before the pipeline";
      const std::string& transformer_dtype =
          context.get_model_args("transformer").dtype();
      CHECK(transformer_dtype == "bfloat16" || transformer_dtype == "bf16")
          << "QwenImage21 NPU Transformer requires a BF16 checkpoint";
      options_ = options_.dtype(torch::kBFloat16);
    }
    auto transformer_config =
        qwen_image_21_transformer_config(context.get_model_args("transformer"));
    auto vae_config = qwen_image_21_vae_config(context.get_model_args("vae"));
    CHECK_EQ(transformer_config.in_channels, 64);
    CHECK_EQ(transformer_config.out_channels, 64);
    CHECK_EQ(transformer_config.context_in_dim, 4096);
    CHECK_EQ(vae_config.z_dim, 64);
    CHECK(transformer_config.causal_condition &&
          transformer_config.causal_block);
    transformer_ = register_module(
        "transformer",
        QwenImage21Transformer2DModel(std::move(transformer_config), options_));
    // Every VAE group processes the inputs independently, with width shards
    // selected by group-local rank. All workers load the FP32 VAE weights.
    vae_ = register_module(
        "vae",
        AutoencoderKLQwenImage21(context.get_model_context("vae"),
                                 std::move(vae_config),
                                 options_.dtype(torch::kFloat32)));
    CHECK_EQ(vae_->spatial_scale(), 16);
    if (DiTConfig::get_instance().dit_enable_vae_tiling()) {
      vae_->enable_tiling();
      const auto& tiling_config = vae_->tiling_config();
      LOG(INFO) << "QwenImage21 VAE tiling enabled: tile="
                << tiling_config.tile_height
                << " stride=" << tiling_config.stride_height
                << "; encode/decode with overlap blending (approximate)";
    }
    if (DiTConfig::get_instance().dit_enable_text_encoder()) {
      CHECK_EQ(DiTConfig::get_instance().max_requests_per_batch(), 1)
          << "QwenImage21 internal text encoder requires request batch size 1";
      text_encoder_args_ = context.get_model_args("text_encoder");
      text_encoder_processor_args_ = context.get_model_args("processor");
      text_encoder_options_ =
          context.get_tensor_options().dtype(torch::kBFloat16);
      text_encoder_token_capacity_ =
          SchedulerConfig::get_instance().max_tokens_per_batch();
      CHECK_EQ(text_encoder_args_.hidden_size(), 4096);
      CHECK_EQ(text_encoder_args_.mm_patch_size(), 16);
      CHECK_EQ(text_encoder_args_.mm_spatial_merge_size(), 2);
      CHECK_EQ(text_encoder_processor_args_.mm_image_patch_size(), 16);
      CHECK_EQ(text_encoder_processor_args_.mm_image_merge_size(), 2);
      text_encoder_args_.layers_to_capture() = {
          static_cast<int32_t>(text_encoder_args_.n_layers() - 1)};
      text_encoder_args_.encoder_embedding_mode() = false;
      text_encoder_args_.embedding_mode() = true;
      auto* group = context.get_parallel_args().dit_text_encoder_tp_group_;
      CHECK(group != nullptr);
      CHECK_EQ(group->world_size(), 1);
      CHECK_EQ(group->rank(), 0);
      ParallelArgs local(/*rank=*/0, /*world_size=*/1, group);
      local.tp_group_ = group;
      local.dp_local_process_group_ = group;
      // Reuse xLLM's ATB mapping builder, but keep the encoder's world strictly
      // local (TP=1), independent of the DiT CFG/SP communicator.
      MappingNPU::Options mapping_options;
      mapping_options.dp_size(1)
          .tp_size(1)
          .moe_tp_size(1)
          .moe_ep_size(1)
          .pp_size(1)
          .sp_size(1)
          .cp_size(1)
          .kv_split_size(1);
      MappingNPU mapping(
          /*rank_tablefile=*/"", /*world_size=*/1, /*rank=*/0, mapping_options);
      local.mapping_data() = mapping.to_json();
      local.mapping().ParseParam(local.mapping_data());
      local.mapping().InitGlobalCommDomain(
          ParallelConfig::get_instance().communication_backend());
      text_encoder_ = register_module(
          "text_encoder",
          npu::model::Qwen3_VLForConditionalGeneration(
              ModelContext(local,
                           text_encoder_args_,
                           context.get_quant_args("text_encoder"),
                           text_encoder_options_)));
    }
    const auto& scheduler_args = scheduler_context_.get_model_args();
    CHECK_EQ(scheduler_args.num_train_timesteps(), 1000);
    CHECK(scheduler_args.use_dynamic_shifting());
    CHECK_GT(scheduler_args.max_image_seq_len(),
             scheduler_args.base_image_seq_len());
    CHECK_GE(scheduler_args.shift_terminal(), 0.0f);
    CHECK_LT(scheduler_args.shift_terminal(), 1.0f);
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    torch::NoGradGuard guard;
    CHECK(loader != nullptr);
    // Each component loader releases its mmap after the device copies finish.
    // Disabling the internal text encoder avoids materializing its weights.
    transformer_->load_model(loader->take_component_loader("transformer"));
    vae_->load_model(loader->take_component_loader("vae"));
    if (text_encoder_) {
      auto weights = loader->take_component_loader("text_encoder");
      auto processor = loader->take_component_loader("processor");
      CHECK(weights != nullptr);
      CHECK(processor != nullptr);
      prompt_tokenizer_ = processor->tokenizer();
      CHECK(prompt_tokenizer_ != nullptr);
      text_encoder_->load_model(std::move(weights));
    }
    loaded_ = true;
    eval();
    LOG(INFO) << "QwenImage21 loaded: cfg_size=" << cfg_size_
              << " cfg_rank=" << cfg_rank_ << " sp_size=" << sp_size_
              << " sp_rank=" << sp_rank_ << " vae_size=" << vae_size_
              << "; rank_local_text_encoder="
              << static_cast<bool>(text_encoder_);
  }

  DiTForwardOutput forward(const DiTForwardInput& input) {
    torch::NoGradGuard guard;
    CHECK(loaded_) << "Load QwenImage21 weights before inference";
    CHECK_EQ(input.batch_size, 1);
    const auto& params = input.generation_params;
    std::string error = validate_qwen_image_21_generation(params);
    CHECK(error.empty()) << error;
    const int64_t height = params.height;
    const int64_t width = params.width;
    const int64_t target_tokens = (height / 16) * (width / 16);
    const bool internal_prompt =
        !input.tensor_sources.contains("prompt_embed") &&
        !input.tensor_sources.contains("negative_prompt_embed");
    const bool use_cfg = params.true_cfg_scale > 1.0f &&
                         (internal_prompt || input.tensor_sources.contains(
                                                 "negative_prompt_embed"));
    // Without CFG, both SP groups run the positive branch so every rank enters
    // VAE decoding with final latents, including members of cross-CFG VAE
    // groups.
    const qwen_image_21::CfgExecutionPlan plan(
        cfg_size_, use_cfg ? cfg_rank_ : 0, use_cfg);
    // Validate and collect the shared images once, before either prompt path
    // performs preprocessing or iterates over the CFG branches.
    std::vector<QwenImage21ImageSize> sizes;
    std::vector<qwen_image_21::ImageShape> shapes;
    sizes.reserve(input.media_sources.size());
    shapes.reserve(input.media_sources.size() + 1);
    CHECK_LE(input.media_sources.size(), 4);
    for (const auto& source : input.media_sources.entries()) {
      const auto& image = source.tensor;
      CHECK_EQ(source.modality, "image");
      CHECK_EQ(image.dim(), 4);
      CHECK_EQ(image.size(0), 1);
      CHECK(image.size(1) == 3 || image.size(1) == 4);
      CHECK_EQ(image.scalar_type(), torch::kUInt8);
      CHECK(is_valid_qwen_image_21_image_size(image.size(3), image.size(2)))
          << "QwenImage21 condition images require 32-aligned width/height "
             "in [32, 32768]";
      sizes.emplace_back(QwenImage21ImageSize{image.size(3), image.size(2)});
      shapes.emplace_back(
          qwen_image_21::ImageShape{1, image.size(2) / 16, image.size(3) / 16});
    }
    shapes.emplace_back(qwen_image_21::ImageShape{1, height / 16, width / 16});
    QwenImage21PromptInput positive, negative;
    if (internal_prompt) {
      CHECK(text_encoder_)
          << "QwenImage21 requests without external prompt embeddings require "
             "--dit_enable_text_encoder=true";
      // Preserve this constraint even when the image processor disables resize.
      for (const auto& size : sizes) {
        CHECK_LE(std::max(size[0], size[1]), 200 * std::min(size[0], size[1]))
            << "QwenImage21 condition images require an aspect ratio <= 200";
      }
      // Encode one CFG branch per call for identical VLM batch shapes across
      // cfg_size=1 and cfg_size=2, including request-local encoder K/V.
      if (plan.positive()) {
        positive = encode_prompt(input, sizes, /*encode_negative=*/false);
      }
      if (plan.negative()) {
        negative = encode_prompt(input, sizes, /*encode_negative=*/true);
      }
    }
    const auto prompt_source =
        input.tensor_sources.get_named_tensor("prompt_embed");
    bool cache_enabled = true;
    if (prompt_source.has_value()) {
      const auto flag =
          prompt_source->get().parameters.find("enable_prefix_kv_cache");
      if (flag != prompt_source->get().parameters.end()) {
        CHECK(std::holds_alternative<bool>(flag->second));
        cache_enabled = std::get<bool>(flag->second);
      }
    }
    std::unique_ptr<qwen_image_21::PrefixKVCache> positive_cache;
    std::unique_ptr<qwen_image_21::PrefixKVCache> negative_cache;
    if (cache_enabled) {
      if (plan.positive()) {
        positive_cache = std::make_unique<qwen_image_21::PrefixKVCache>(
            transformer_->config().num_layers);
      }
      if (plan.negative()) {
        negative_cache = std::make_unique<qwen_image_21::PrefixKVCache>(
            transformer_->config().num_layers);
      }
    }
    std::unique_ptr<qwen_image_21::RequestTaylorSeer> positive_ts, negative_ts;
    if (use_taylorseer_) {
      if (plan.positive()) {
        positive_ts = std::make_unique<qwen_image_21::RequestTaylorSeer>(
            taylorseer_options_, params.num_inference_steps);
      }
      if (plan.negative()) {
        negative_ts = std::make_unique<qwen_image_21::RequestTaylorSeer>(
            taylorseer_options_, params.num_inference_steps);
      }
    }
    qwen_image_21::SequenceLayout positive_layout, negative_layout;
    if (plan.positive()) {
      if (!internal_prompt) {
        positive = read_branch(input.tensor_sources, "prompt_embed", sizes);
      }
      positive_layout = transformer_->prepare_layout(
          positive.image_mask, positive.valid_mask, shapes, options_.device());
    }
    if (plan.negative()) {
      if (!internal_prompt) {
        negative =
            read_branch(input.tensor_sources, "negative_prompt_embed", sizes);
      }
      CHECK(negative.embeddings.defined());
      negative_layout = transformer_->prepare_layout(
          negative.image_mask, negative.valid_mask, shapes, options_.device());
    }
    std::unique_ptr<qwen_image_21::UlyssesContext> positive_sp, negative_sp;
    if (sp_size_ > 1) {
      if (plan.positive()) {
        positive_sp = std::make_unique<qwen_image_21::UlyssesContext>(
            positive_layout, sp_group_);
      }
      if (plan.negative()) {
        negative_sp = std::make_unique<qwen_image_21::UlyssesContext>(
            negative_layout, sp_group_);
      }
    }
    auto conditions = encode_condition_images(input.media_sources);
    auto latents = prepare_initial_latents(
        input.tensor_sources, height, width, params.seed);
    CHECK_EQ(latents.size(1), target_tokens);

    // Scheduler state is local to this invocation, so a new request cannot
    // inherit a step index or a sigma schedule from the previous request.
    auto scheduler = FlowMatchEulerDiscreteScheduler(scheduler_context_);
    const auto& scheduler_args = scheduler_context_.get_model_args();
    const double slope =
        (scheduler_args.max_shift() - scheduler_args.base_shift()) /
        static_cast<double>(scheduler_args.max_image_seq_len() -
                            scheduler_args.base_image_seq_len());
    const float mu = static_cast<float>(
        scheduler_args.base_shift() +
        slope * (target_tokens - scheduler_args.base_image_seq_len()));
    std::vector<float> sigmas;
    sigmas.reserve(params.num_inference_steps);
    const double end = 1.0 / params.num_inference_steps;
    for (int64_t i = 0; i < params.num_inference_steps; ++i) {
      sigmas.emplace_back(static_cast<float>(
          1.0 + (end - 1.0) * i / (params.num_inference_steps - 1)));
    }
    scheduler->set_timesteps(
        params.num_inference_steps, options_.device(), sigmas, mu);
    scheduler->set_begin_index(0);
    const auto& timesteps = scheduler->timesteps();
    for (int64_t i = 0; i < timesteps.size(0); ++i) {
      const auto* local_cache =
          plan.positive() ? positive_cache.get() : negative_cache.get();
      const bool decode = local_cache != nullptr && local_cache->ready();
      auto model_input = !decode && conditions.defined()
                             ? torch::cat({conditions, latents}, 1)
                             : latents;
      auto timestep = timesteps[i].expand({1}).to(options_) / 1000.0;
      torch::Tensor positive_noise, negative_noise, noise;
      if (plan.positive()) {
        positive_noise = transformer_->forward(model_input,
                                               positive.embeddings,
                                               timestep,
                                               positive_layout,
                                               positive_cache.get(),
                                               positive_sp.get(),
                                               positive_ts.get(),
                                               i + 1);
      }
      if (plan.negative()) {
        negative_noise = transformer_->forward(model_input,
                                               negative.embeddings,
                                               timestep,
                                               negative_layout,
                                               negative_cache.get(),
                                               negative_sp.get(),
                                               negative_ts.get(),
                                               i + 1);
      }
      if (plan.distributed()) {
        auto local_noise =
            (plan.positive() ? positive_noise : negative_noise).contiguous();
        // Same gather API as EditPlus, but no CFG renormalization. Both ranks
        // receive [cond, uncond] and advance identical local scheduler states.
        auto gathered =
            parallel_state::gather(local_noise, cfg_group_, /*dim=*/0);
        noise = qwen_image_21::combine_gathered_cfg_noise(
            gathered, params.true_cfg_scale);
      } else {
        noise = use_cfg
                    ? qwen_image_21::combine_cfg_noise(
                          positive_noise, negative_noise, params.true_cfg_scale)
                    : positive_noise;
      }
      // The 2.1 reference explicitly disables CFG norm renormalization.
      latents = scheduler->step(noise, timesteps[i], latents).to(options_);
    }
    // Do not retain several GiB of K/V during FP32 VAE decoding, and never
    // persist it on the shared model between requests.
    auto report_taylorseer = [&](const char* branch, const auto& state) {
      if (state) {
        LOG(INFO) << "QwenImage21 TaylorSeer branch=" << branch
                  << " cfg_rank=" << cfg_rank_ << " sp_rank=" << sp_rank_
                  << " full_steps=" << state->full_steps()
                  << " hit_steps=" << state->hit_steps()
                  << " history_bytes=" << state->bytes()
                  << " order=" << taylorseer_options_.n_derivatives
                  << " warmup=" << taylorseer_options_.warmup_steps
                  << " interval=" << taylorseer_options_.skip_interval_steps;
      }
    };
    report_taylorseer("positive", positive_ts);
    report_taylorseer("negative", negative_ts);
    positive_ts.reset();
    negative_ts.reset();
    positive_cache.reset();
    negative_cache.reset();
    return finish_vae_output(latents, height, width);
  }

 private:
  QwenImage21PromptInput encode_prompt(
      const DiTForwardInput& input,
      const std::vector<QwenImage21ImageSize>& sizes,
      bool encode_negative) {
    torch::NoGradGuard guard;
    CHECK(prompt_tokenizer_ != nullptr);
    CHECK_EQ(input.prompts.size(), 1);
    CHECK(!encode_negative || input.generation_params.true_cfg_scale > 1.0f);
    CHECK(!encode_negative || input.negative_prompts.size() <= 1);
    std::vector<MMDataItem> items;
    items.reserve(input.media_sources.size());
    for (const auto& source : input.media_sources.entries()) {
      const auto image = source.tensor.squeeze(0).to(torch::kCPU);
      auto rgb = image.slice(0, 0, 3).to(torch::kFloat32);
      if (image.size(0) == 4) {
        const auto alpha = image.slice(0, 3, 4).to(torch::kFloat32);
        // PIL's integer alpha composition over white, not RGB channel drop.
        rgb = torch::floor((rgb * alpha + 255.0f * (255.0f - alpha) + 127.0f) /
                           255.0f);
      }
      // The uploaded VLM/VAE size is one contract. Pin the processor's pixel
      // budget to this exact, already 32-aligned image; never resize it again.
      auto processor_args = text_encoder_processor_args_;
      const int64_t area = image.size(1) * image.size(2);
      processor_args.mm_image_min_pixels() = area;
      processor_args.mm_image_max_pixels() = area;
      processor_args.mm_image_min_tokens() = 0;
      processor_args.mm_image_max_tokens() = 0;
      Qwen2VLImageProcessor processor(processor_args);
      std::vector<torch::Tensor> pixels, grids;
      CHECK(processor.process_image({rgb}, pixels, grids));
      CHECK_EQ(pixels.size(), 1);
      CHECK_EQ(grids.size(), 1);
      CHECK_EQ(grids[0][0][1].item<int64_t>(), image.size(1) / 16);
      CHECK_EQ(grids[0][0][2].item<int64_t>(), image.size(2) / 16);
      items.emplace_back(
          MMType::IMAGE,
          MMDict{{"pixel_values", pixels[0]}, {"image_grid_thw", grids[0]}});
    }
    MMData mm_data;
    if (!items.empty()) {
      mm_data = MMData(MMType::IMAGE, items);
    }
    // DiTBatch drops empty strings. Preserve the unconditional CFG branch.
    const std::string prompt =
        encode_negative
            ? (input.negative_prompts.empty() ? " "
                                              : input.negative_prompts.front())
            : input.prompts.front();
    QwenImage21PromptTokens branch;
    const std::string error = prepare_qwen_image_21_prompt_tokens(
        *prompt_tokenizer_, prompt, sizes, branch);
    CHECK(error.empty()) << error;
    const auto& ids = branch.ids;
    const int32_t length = static_cast<int32_t>(ids.size());
    // Each CFG branch must fit in the configured encoder capacity separately.
    CHECK_LE(length, text_encoder_token_capacity_)
        << "QwenImage21 input exceeds the configured encoder capture capacity";
    Qwen3VLMPositionGenerator position_generator;
    auto [position, delta] =
        position_generator.generate(ids, mm_data, text_encoder_args_);
    const auto tokens =
        torch::tensor(ids, torch::kInt).to(text_encoder_options_.device());
    const auto positions =
        position.to(text_encoder_options_.device(), torch::kInt).contiguous();
    ModelInputParams params;
    params.meta.batch_forward_type = BatchForwardType::PREFILL;
    params.meta.num_sequences = 1;
    params.meta.actual_num_sequences = 1;
    params.meta.q_max_seq_len = length;
    params.meta.kv_max_seq_len = length;
    params.prefill_without_cache = false;
    params.attention.host.q_seq_lens = {length};
    params.attention.host.kv_seq_lens = {length};
    params.attention.host.q_cu_seq_lens = {length};
    params.attention.host.kv_cache_tokens_nums = {0};
    const auto sequence_lengths =
        torch::tensor({length}, torch::kInt).to(text_encoder_options_.device());
    params.attention.device.q_seq_lens = sequence_lengths;
    params.attention.device.kv_seq_lens = sequence_lengths;
    params.attention.device.q_cu_seq_lens = sequence_lengths;
    params.parallel.dp_global_token_nums = {length};
    if (!items.empty()) {
      // ATB splits vision output by MMItemState::mm_token_num. Populate the
      // same per-image token counts and spans as the standalone VLM service.
      Qwen3VLPromptProcessor prompt_processor(text_encoder_args_);
      CHECK(prompt_processor.find_mm_spans(ids, mm_data));
      ModelInputParams vision_params;
      vision_params.multimodal.mm_data =
          MMBatchData::to(MMBatchData(std::vector<MMData>{mm_data}),
                          text_encoder_options_.device());
      const auto visual =
          text_encoder_->get_multimodal_embeddings(vision_params);
      const auto found = visual.find("image|embedding");
      CHECK(found != visual.end());
      auto embeddings =
          std::holds_alternative<torch::Tensor>(found->second)
              ? std::get<torch::Tensor>(found->second)
              : torch::cat(std::get<std::vector<torch::Tensor>>(found->second),
                           /*dim=*/0);
      embeddings = embeddings.contiguous();
      const auto image_mask = tokens.eq(text_encoder_args_.image_token_id());
      CHECK_EQ(image_mask.sum().item<int64_t>(), embeddings.size(0));
      params.multimodal.mm_data = MMBatchData(
          MMType::IMAGE,
          MMDict{{"image|embedding", embeddings}, {"image|mask", image_mask}});
    }
    params.embedding.input_embedding =
        text_encoder_->get_input_embeddings(tokens, params);
    // The existing ATB component uses paged K/V even for a complete prefill.
    // Allocate only this request's blocks, never a scheduler/global cache pool.
    const int64_t block_size = KVCacheConfig::get_instance().block_size();
    CHECK_GT(block_size, 0);
    const int32_t block_count =
        static_cast<int32_t>((length + block_size - 1) / block_size);
    const auto tables = torch::arange(block_count, torch::kInt).unsqueeze(0);
    params.attention.host.block_tables = tables;
    params.attention.device.block_tables =
        tables.to(text_encoder_options_.device());
    params.attention.device.new_cache_slots =
        torch::arange(length, torch::kInt).to(text_encoder_options_.device());
    std::vector<KVCache> caches;
    caches.reserve(static_cast<size_t>(text_encoder_args_.n_layers()));
    KVCacheCapacity cache_capacity;
    cache_capacity.n_blocks(block_count).block_size(block_size);
    const KVCacheShape cache_shape(
        cache_capacity, text_encoder_args_, /*world_size=*/1);
    KVCacheCreateOptions cache_options;
    cache_options.device(text_encoder_options_.device())
        .dtype(text_encoder_options_.dtype().toScalarType())
        .model_type(text_encoder_args_.model_type())
        .num_layers(text_encoder_args_.n_layers());
    // Use the existing allocator: ATB ReshapeAndCache requires ND K/V.
    // Wrapping plain four-dimensional torch::empty tensors bypasses that
    // format conversion even when the shape and BF16 dtype are correct.
    for (int64_t layer = 0; layer < text_encoder_args_.n_layers(); ++layer) {
      caches.emplace_back(cache_shape, cache_options, layer);
    }
    const auto model_output =
        text_encoder_->forward(tokens, positions, caches, params);
    const auto& hidden = model_output.aux_hidden_states;
    CHECK(hidden.defined())
        << "QwenImage21 requires final-layer pre-norm capture";
    CHECK_EQ(hidden.size(0), length);
    CHECK_EQ(hidden.size(1), 4096);
    const int64_t prompt_length =
        static_cast<int64_t>(length) - branch.drop_idx;
    QwenImage21PromptInput result;
    // Strip the system prefix and own storage before encoding another branch.
    result.embeddings =
        hidden.slice(0, branch.drop_idx, length).clone().unsqueeze(0);
    result.valid_mask =
        torch::ones({1, prompt_length}, torch::dtype(torch::kBool));
    result.image_mask = torch::tensor(branch.image_mask, torch::kBool)
                            .slice(0, branch.drop_idx)
                            .unsqueeze(0);
    return result;
  }

  DiTForwardOutput finish_vae_output(const torch::Tensor& latents,
                                     int64_t height,
                                     int64_t width) {
    // All VAE groups decode their own final latents; only the driver returns
    // an image to the caller.
    auto unpacked =
        qwen_image_21::unpack_latents(latents, height / 16, width / 16)
            .to(torch::kFloat32);
    auto decoded = vae_->decode(vae_->denormalize_latents(unpacked)).squeeze(2);
    if (!is_driver_) {
      return {};
    }
    auto image = (decoded / 2.0 + 0.5)
                     .clamp(0.0, 1.0)
                     .to(torch::kCPU, torch::kFloat32)
                     .contiguous();
    CHECK_EQ(image.size(1), 4);
    DiTForwardOutput result;
    result.tensors.emplace_back(std::move(image));
    return result;
  }

  QwenImage21PromptInput read_branch(
      const DiTTensorSources& sources,
      const std::string& name,
      const std::vector<QwenImage21ImageSize>& sizes) const {
    auto source = sources.get_named_tensor(name);
    CHECK(source.has_value()) << "Missing QwenImage21 " << name;
    QwenImage21PromptInput branch;
    const std::string error =
        read_qwen_image_21_prompt(source->get(), sizes, branch);
    CHECK(error.empty()) << error;
    branch.embeddings = branch.embeddings.to(options_);
    return branch;
  }

  torch::Tensor encode_condition_images(const DiTMediaSources& sources) {
    if (sources.empty()) {
      return {};
    }
    std::vector<torch::Tensor> encoded;
    encoded.reserve(sources.size());
    for (const auto& source : sources.entries()) {
      auto image = source.tensor;
      if (image.size(1) == 3) {
        auto alpha = torch::full(
            {1, 1, image.size(2), image.size(3)}, 255, image.options());
        image = torch::cat({image, alpha}, 1);
      }
      // Images are already resized jointly with the VLM. Never resize them
      // again here: the uploaded dimensions and slot grid are a contract.
      auto pixels =
          (image.to(options_.dtype(torch::kFloat32)) / 255.0 * 2.0 - 1.0)
              .unsqueeze(2);
      auto latent = vae_->normalize_latents(vae_->encode_mode(pixels));
      encoded.emplace_back(qwen_image_21::pack_latents(latent).to(options_));
    }
    return torch::cat(encoded, 1);
  }

  torch::Tensor prepare_initial_latents(const DiTTensorSources& sources,
                                        int64_t height,
                                        int64_t width,
                                        int64_t seed) const {
    const auto initial = sources.get("latent");
    if (initial.has_value()) {
      CHECK(initial->sizes() ==
            torch::IntArrayRef({1, (height / 16) * (width / 16), 64}));
      CHECK(torch::isfinite(*initial).all().item<bool>());
      return initial->to(options_);
    }
    // Reset a request-local CPU generator to the same seed on every rank.
    auto generator = torch::make_generator<torch::CPUGeneratorImpl>();
    generator.set_current_seed(static_cast<uint64_t>(seed));
    auto noise = torch::randn({1, 64, 1, height / 16, width / 16},
                              generator,
                              options_.device(torch::kCPU));
    return qwen_image_21::pack_latents(noise).to(options_);
  }

  int32_t vae_size_ = 1;
  int32_t sp_size_ = 1;
  int32_t sp_rank_ = 0;
  bool is_driver_ = true;
  ProcessGroup* sp_group_ = nullptr;  // Non-owning.
  int32_t cfg_size_ = 1;
  int32_t cfg_rank_ = 0;
  ProcessGroup* cfg_group_ =
      nullptr;  // Non-owning; communicator outlives pipeline.
  torch::TensorOptions options_;
  ModelContext scheduler_context_;
  ModelArgs text_encoder_args_;
  ModelArgs text_encoder_processor_args_;
  torch::TensorOptions text_encoder_options_;
  int64_t text_encoder_token_capacity_ = 0;
  std::unique_ptr<Tokenizer> prompt_tokenizer_;
  npu::model::Qwen3_VLForConditionalGeneration text_encoder_{nullptr};
  QwenImage21Transformer2DModel transformer_{nullptr};
  AutoencoderKLQwenImage21 vae_{nullptr};
  bool use_taylorseer_ = false;
  TaylorSeerOptions taylorseer_options_;
  bool loaded_ = false;
};
TORCH_MODULE(QwenImage21Pipeline);
REGISTER_DIT_MODEL(QwenImage21Pipeline, QwenImage21Pipeline);
REGISTER_IMAGE_ALPHA_POLICY(QwenImage21Pipeline, true);

}  // namespace xllm
