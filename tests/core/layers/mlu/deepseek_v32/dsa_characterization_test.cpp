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

#include <c10/util/ScopeExit.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/config/kv_cache_config.h"
#include "layers/mlu/deepseek_v32/deepseek_v32_attention.h"
#include "layers/mlu/deepseek_v32/deepseek_v32_decoder_layer.h"
#include "layers/mlu/deepseek_v32/dsa_topk_relay.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm::layer {
namespace {
constexpr int32_t kCapacity = 64;

torch::Tensor active_topk(const torch::Tensor& table,
                          const torch::Tensor& lengths) {
  const auto columns = torch::arange(table.size(1), lengths.options());
  return torch::where(
      columns < lengths.unsqueeze(1), table, torch::full_like(table, -1));
}

struct Glm5DcpConfig {
  ModelArgs args;
  QuantArgs quant;
};

enum class DsaModelVariant { V32, GLM5, GLM52_SHARE, GLM5_MTP };

enum class DsaWeightFormat { BF16, SMOOTHQUANT, COMPRESSED_TENSORS };

using Weights = std::unordered_map<std::string, torch::Tensor>;

ModelArgs create_decoder_model_args();
ModelArgs create_production_mla_args(
    const std::string& rope_type = "deepseek_yarn");
Weights create_mla_weights(const ModelArgs& model_args,
                           const torch::TensorOptions& options);
ModelArgs create_dsa_model_args(DsaModelVariant variant,
                                bool production = false);
Weights create_seeded_linear_weights(int64_t out,
                                     int64_t in,
                                     DsaWeightFormat format,
                                     const torch::TensorOptions& options,
                                     const std::string& seed,
                                     const std::string& prefix = "");
Weights create_seeded_indexer_weights(const ModelArgs& args,
                                      const torch::TensorOptions& options,
                                      const std::string& prefix = "");
Weights create_seeded_dense_weights(const ModelArgs& args,
                                    DsaWeightFormat format,
                                    const torch::TensorOptions& options,
                                    const std::string& prefix = "");
Weights create_seeded_mla_weights(const ModelArgs& args,
                                  DsaWeightFormat format,
                                  const torch::TensorOptions& options,
                                  const std::string& prefix = "");
void expect_close(const torch::Tensor& actual,
                  const torch::Tensor& expected,
                  double rtol = 0.02,
                  double atol = 0.025,
                  const std::string& label = "");
KVCache create_indexed_kv_cache(torch::Tensor key_cache,
                                torch::Tensor index_cache);
ModelArgs create_attention_model_args(int64_t q_lora_rank = 0,
                                      bool enable_indexer = false);
Glm5DcpConfig create_glm5_dcp_config();

ModelArgs create_decoder_model_args() {
  ModelArgs args = ::xllm::layer::test::create_default_model_args();
  // Use a small but structurally valid DeepSeek-style config so the tests
  // exercise Dense/MoE/MLA wiring without allocating near-production tensors.
  args.model_type() = "deepseek_v3";
  args.dtype() = "";  // default empty
  args.vocab_size() = 4096;
  args.hidden_size() = 512;
  args.n_layers() = 8;
  args.n_heads() = 8;
  args.n_kv_heads() = 8;
  args.intermediate_size() = 1024;
  args.hidden_act() = "silu";
  args.rms_norm_eps() = 1e-6f;
  args.max_position_embeddings() = 4096;
  args.eos_token_id() = 1;
  args.bos_token_id() = 0;

  // Decoder layer specific routing between MoE and Dense
  args.first_k_dense_replace() = 1;
  args.moe_layer_freq() = 1;

  // MoE-related params
  args.n_routed_experts() = 8;
  args.n_shared_experts() = 1;
  args.num_experts_per_tok() = 2;
  args.moe_intermediate_size() = 128;
  args.routed_scaling_factor() = 2.5f;
  args.norm_topk_prob() = true;
  args.n_group() = 2;
  args.topk_group() = 1;
  args.scoring_func() = "sigmoid";
  args.topk_method() = "noaux_tc";

  // Q/K/V dims used by DeepseekV32Attention
  args.qk_nope_head_dim() = 64;
  args.qk_rope_head_dim() = 64;
  args.v_head_dim() = 64;
  args.head_dim() = 128;  // qk_nope_head_dim + qk_rope_head_dim
  args.rotary_dim() = args.qk_rope_head_dim();

  // Rope scaling related
  args.rope_scaling_rope_type() = "deepseek_yarn";
  // The following values may be model/export dependent; set common defaults.
  args.rope_scaling_beta_fast() = 32;
  args.rope_scaling_beta_slow() = 1;
  args.rope_scaling_factor() = 40;
  args.rope_extrapolation_factor() = 1.0f;
  args.rope_scaling_mscale() = 1.0f;
  args.rope_scaling_mscale_all_dim() = 1.0f;
  args.rope_scaling_original_max_position_embeddings() = 1024;
  args.rope_scaling_attn_factor() = 1.0f;

  // Sliding window
  args.use_sliding_window() = false;
  args.sliding_window() = 512;
  args.max_window_layers() = 8;

  // LORA ranks for DeepSeek-V3
  args.q_lora_rank() = 128;
  args.kv_lora_rank() = 64;  // qk_rope_head_dim + kv_lora_rank = 128

  // extra parameters for DeepSeek-V3.2-Exp
  args.index_head_dim() = 128;
  args.index_n_heads() = 0;
  args.index_topk() = 0;
  args.enable_mla() = true;

  return args;
}

ModelArgs create_production_mla_args(const std::string& rope_type) {
  ModelArgs model_args;
  model_args.q_lora_rank() = 1536;
  model_args.kv_lora_rank() = 512;
  model_args.qk_nope_head_dim() = 128;
  model_args.qk_rope_head_dim() = 64;
  model_args.v_head_dim() = 128;
  model_args.hidden_size() = 7168;
  model_args.n_heads() = 128;
  model_args.max_position_embeddings() = 163840;
  model_args.rope_theta() = 10000;
  model_args.rms_norm_eps() = 1e-06;

  // rope_scaling config
  model_args.rope_scaling_original_max_position_embeddings() = 4096;
  model_args.rope_scaling_factor() = 40;
  model_args.rope_extrapolation_factor() = 1.;
  model_args.rope_scaling_attn_factor() = 1.;
  model_args.rope_scaling_beta_fast() = 32;
  model_args.rope_scaling_beta_slow() = 1;
  model_args.rope_scaling_mscale() = 1.;
  model_args.rope_scaling_mscale_all_dim() = 1.;
  model_args.rope_scaling_rope_type() = rope_type;

  // indexer
  model_args.index_head_dim() = 128;
  model_args.index_n_heads() = 64;
  model_args.index_topk() = 2048;
  model_args.enable_mla() = true;

  return model_args;
}

Weights create_mla_weights(const ModelArgs& model_args,
                           const torch::TensorOptions& options) {
  Weights weights;
  int64_t q_lora_rank = model_args.q_lora_rank();
  int64_t kv_lora_rank = model_args.kv_lora_rank();
  int64_t qk_nope_head_dim = model_args.qk_nope_head_dim();
  int64_t qk_rope_head_dim = model_args.qk_rope_head_dim();
  int64_t index_topk = model_args.index_topk();
  int64_t index_n_heads = model_args.index_n_heads();
  int64_t index_head_dim = model_args.index_head_dim();
  int64_t v_head_dim = model_args.v_head_dim();
  int64_t hidden_size = model_args.hidden_size();
  int64_t num_heads = model_args.n_heads();
  int64_t max_position_embeddings = model_args.max_position_embeddings();
  int64_t qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
  std::unordered_map<std::string, std::vector<int64_t>> qweight_map = {
      {"model.layers.0.self_attn.o_proj.qweight",
       {hidden_size, num_heads * v_head_dim}},
      {"model.layers.0.self_attn.q_b_proj.qweight",
       {num_heads * qk_head_dim, q_lora_rank}},
  };
  std::unordered_map<std::string, std::vector<int64_t>> scale_map = {
      {"model.layers.0.self_attn.o_proj.per_channel_scale", {hidden_size}},
      {"model.layers.0.self_attn.q_b_proj.per_channel_scale",
       {num_heads * qk_head_dim}},
      {"model.layers.0.self_attn.o_proj.smooth", {num_heads * v_head_dim}},
      {"model.layers.0.self_attn.q_b_proj.smooth", {q_lora_rank}},
  };
  std::unordered_map<std::string, std::vector<int64_t>> weight_map = {
      {"model.layers.0.self_attn.indexer.k_norm.bias", {index_head_dim}},
      {"model.layers.0.self_attn.indexer.k_norm.weight", {index_head_dim}},
      {"model.layers.0.self_attn.kv_a_layernorm.weight", {kv_lora_rank}},
      {"model.layers.0.self_attn.q_a_layernorm.weight", {q_lora_rank}},
      {"model.layers.0.self_attn.indexer.weights_proj.weight",
       {index_n_heads, hidden_size}},
      {"model.layers.0.self_attn.indexer.wk.weight",
       {index_head_dim, hidden_size}},
      {"model.layers.0.self_attn.indexer.wq_b.weight",
       {index_n_heads * index_head_dim, q_lora_rank}},
      {"model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
       {kv_lora_rank + qk_rope_head_dim, hidden_size}},
      {"model.layers.0.self_attn.kv_b_proj.weight",
       {num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank}},
      {"model.layers.0.self_attn.q_a_proj.weight", {q_lora_rank, hidden_size}},
  };

  auto float_option = options.dtype(torch::kFloat32);
  for (auto& [key, shape] : qweight_map) {
    auto tensor =
        test::seeded_tensor(key, shape, torch::kInt8, options.device());
    weights[key] = tensor;
  }
  for (auto& [key, shape] : scale_map) {
    auto tensor =
        test::seeded_tensor(key, shape, torch::kFloat32, options.device()) *
            0.01f +
        0.03f;
    weights[key] = tensor;
  }
  for (auto& [key, shape] : weight_map) {
    auto tensor =
        test::seeded_tensor(key, shape, torch::kBFloat16, options.device()) *
            0.01f +
        0.02f;
    weights[key] = tensor;
  }

  return weights;
}

ModelArgs create_dsa_model_args(DsaModelVariant variant, bool production) {
  ModelArgs args = production
                       ? create_production_mla_args()
                       : create_attention_model_args(/*q_lora_rank=*/64,
                                                     /*enable_indexer=*/true);
  if (!production) {
    const ModelArgs decoder = create_decoder_model_args();
    args.n_layers() = decoder.n_layers();
    args.vocab_size() = decoder.vocab_size();
    args.intermediate_size() = decoder.intermediate_size();
    args.moe_intermediate_size() = decoder.moe_intermediate_size();
    args.first_k_dense_replace() = decoder.first_k_dense_replace();
    args.moe_layer_freq() = decoder.moe_layer_freq();
    args.n_routed_experts() = decoder.n_routed_experts();
    args.n_shared_experts() = decoder.n_shared_experts();
    args.num_experts_per_tok() = decoder.num_experts_per_tok();
    args.routed_scaling_factor() = decoder.routed_scaling_factor();
    args.norm_topk_prob() = decoder.norm_topk_prob();
    args.scoring_func() = decoder.scoring_func();
    args.topk_method() = decoder.topk_method();
    args.n_group() = decoder.n_group();
    args.topk_group() = decoder.topk_group();
    args.hidden_act() = decoder.hidden_act();
  }
  args.model_type() = "deepseek_v32";
  args.index_n_heads() = production ? 64 : 32;
  args.index_topk() = production ? 2048 : 16;
  if (variant == DsaModelVariant::V32) {
    return args;
  }
  if (production) {
    args = create_glm5_dcp_config().args;
    args.n_layers() = 8;
  }
  args.model_type() = "glm_moe_dsa";
  args.qk_nope_head_dim() = 192;
  args.v_head_dim() = 256;
  args.index_n_heads() = 32;
  args.rope_theta() = production ? 8000000.0f : 1000000.0f;
  args.rope_scaling_rope_type() = "default";
  args.indexer_rope_interleave() = true;
  args.head_dim() = args.qk_nope_head_dim() + args.qk_rope_head_dim();
  if (variant == DsaModelVariant::GLM52_SHARE) {
    args.index_topk_pattern() = "FSFSFSFS";
  }
  if (variant == DsaModelVariant::GLM5_MTP) {
    args.model_type() = "glm_moe_dsa_mtp";
    args.index_share_for_mtp_iteration() = true;
    args.mtp_start_layer_idx() = args.n_layers();
  }
  return args;
}

Weights create_seeded_linear_weights(int64_t out,
                                     int64_t in,
                                     DsaWeightFormat format,
                                     const torch::TensorOptions& options,
                                     const std::string& seed,
                                     const std::string& prefix) {
  Weights weights;
  if (format == DsaWeightFormat::BF16) {
    weights[prefix + "weight"] =
        ::xllm::layer::test::seeded_tensor(
            seed + ".weight", {out, in}, torch::kBFloat16, options.device()) *
        0.01f;
    return weights;
  }
  const bool compressed = format == DsaWeightFormat::COMPRESSED_TENSORS;
  weights[prefix + (compressed ? "weight" : "qweight")] =
      ::xllm::layer::test::seeded_tensor(
          seed + ".qweight", {out, in}, torch::kInt8, options.device());
  torch::Tensor scale =
      ::xllm::layer::test::seeded_tensor(
          seed + ".scale", {out}, torch::kFloat32, options.device())
              .abs() *
          0.01f +
      0.03f;
  weights[prefix + (compressed ? "weight_scale" : "per_channel_scale")] =
      compressed ? scale.reshape({out, 1}) : scale;
  weights[prefix + "smooth"] =
      ::xllm::layer::test::seeded_tensor(
          seed + ".smooth", {in}, torch::kFloat32, options.device())
              .abs() *
          0.01f +
      0.03f;
  return weights;
}

Weights create_seeded_indexer_weights(const ModelArgs& args,
                                      const torch::TensorOptions& options,
                                      const std::string& prefix) {
  Weights weights;
  auto add = [&](const std::string& name, std::vector<int64_t> shape) {
    weights[prefix + name] =
        ::xllm::layer::test::seeded_tensor(
            prefix + name, shape, torch::kBFloat16, options.device()) *
            0.01f +
        0.02f;
  };
  add("wq_b.weight",
      {args.index_n_heads() * args.index_head_dim(), args.q_lora_rank()});
  add("wk.weight", {args.index_head_dim(), args.hidden_size()});
  add("weights_proj.weight", {args.index_n_heads(), args.hidden_size()});
  add("k_norm.weight", {args.index_head_dim()});
  add("k_norm.bias", {args.index_head_dim()});
  return weights;
}

Weights create_seeded_dense_weights(const ModelArgs& args,
                                    DsaWeightFormat format,
                                    const torch::TensorOptions& options,
                                    const std::string& prefix) {
  Weights weights = create_seeded_linear_weights(args.intermediate_size(),
                                                 args.hidden_size(),
                                                 format,
                                                 options,
                                                 prefix + "gate_proj",
                                                 prefix + "gate_proj.");
  weights.merge(create_seeded_linear_weights(args.intermediate_size(),
                                             args.hidden_size(),
                                             format,
                                             options,
                                             prefix + "up_proj",
                                             prefix + "up_proj."));
  weights.merge(create_seeded_linear_weights(args.hidden_size(),
                                             args.intermediate_size(),
                                             format,
                                             options,
                                             prefix + "down_proj",
                                             prefix + "down_proj."));
  return weights;
}

Weights create_seeded_mla_weights(const ModelArgs& args,
                                  DsaWeightFormat format,
                                  const torch::TensorOptions& options,
                                  const std::string& prefix) {
  Weights weights;
  auto add = [&](const std::string& name,
                 int64_t out,
                 int64_t in,
                 DsaWeightFormat projection_format) {
    weights.merge(create_seeded_linear_weights(out,
                                               in,
                                               projection_format,
                                               options,
                                               prefix + name,
                                               prefix + name + "."));
  };
  add("q_a_proj",
      args.q_lora_rank(),
      args.hidden_size(),
      DsaWeightFormat::BF16);
  add("q_b_proj",
      args.n_heads() * (args.qk_nope_head_dim() + args.qk_rope_head_dim()),
      args.q_lora_rank(),
      format);
  add("kv_a_proj_with_mqa",
      args.kv_lora_rank() + args.qk_rope_head_dim(),
      args.hidden_size(),
      DsaWeightFormat::BF16);
  add("kv_b_proj",
      args.n_heads() * (args.qk_nope_head_dim() + args.v_head_dim()),
      args.kv_lora_rank(),
      DsaWeightFormat::BF16);
  add("o_proj", args.hidden_size(), args.n_heads() * args.v_head_dim(), format);
  weights[prefix + "q_a_layernorm.weight"] =
      torch::ones({args.q_lora_rank()}, options.dtype(torch::kFloat32));
  weights[prefix + "kv_a_layernorm.weight"] =
      torch::ones({args.kv_lora_rank()}, options.dtype(torch::kFloat32));
  weights.merge(
      create_seeded_indexer_weights(args, options, prefix + "indexer."));
  return weights;
}

void expect_close(const torch::Tensor& actual,
                  const torch::Tensor& expected,
                  double rtol,
                  double atol,
                  const std::string& label) {
  const torch::Tensor ref = expected.cpu().to(torch::kFloat32);
  const torch::Tensor value = actual.cpu().to(torch::kFloat32);
  ASSERT_GT(ref.abs().max().item<float>(), atol)
      << label << " reference is too small for a meaningful absolute tolerance";
  EXPECT_TRUE(torch::allclose(value, ref, rtol, atol))
      << label << " max error " << (value - ref).abs().max().item<float>();
}

KVCache create_indexed_kv_cache(torch::Tensor key_cache,
                                torch::Tensor index_cache) {
  return KVCache(IndexedKVCacheTensors{
      KVCacheTensors{std::move(key_cache), torch::Tensor()},
      std::move(index_cache)});
}

ModelArgs create_attention_model_args(int64_t q_lora_rank,
                                      bool enable_indexer) {
  ModelArgs args;
  args.model_type() = "deepseek_v32";
  args.hidden_size() = 256;
  args.n_heads() = 4;
  args.max_position_embeddings() = 128;
  args.rope_theta() = 10000.0f;
  args.rms_norm_eps() = 1e-6f;
  args.q_lora_rank() = q_lora_rank;
  args.kv_lora_rank() = enable_indexer ? 256 : 128;
  args.qk_nope_head_dim() = 128;
  args.qk_rope_head_dim() = 64;
  args.v_head_dim() = 128;
  args.index_n_heads() = enable_indexer ? 64 : 0;
  args.index_head_dim() = enable_indexer ? 128 : 0;
  args.index_topk() = enable_indexer ? 8 : 0;
  args.sliding_window() = 0;
  args.enable_mla() = true;
  args.rope_scaling_rope_type() = "";
  args.rope_scaling_original_max_position_embeddings() = 128;
  args.rope_scaling_factor() = 1.0f;
  args.rope_extrapolation_factor() = 1.0f;
  args.rope_scaling_attn_factor() = 1.0f;
  args.rope_scaling_beta_fast() = 1;
  args.rope_scaling_beta_slow() = 1;
  args.rope_scaling_mscale() = 1.0f;
  args.rope_scaling_mscale_all_dim() = 1.0f;
  return args;
}

Glm5DcpConfig create_glm5_dcp_config() {
  Glm5DcpConfig result;
  ModelArgs& args = result.args;
  args.model_type() = "glm_moe_dsa";
  args.hidden_size() = 6144;
  args.n_heads() = 64;
  args.q_lora_rank() = 2048;
  args.kv_lora_rank() = 512;
  args.qk_nope_head_dim() = 192;
  args.qk_rope_head_dim() = 64;
  args.v_head_dim() = 256;
  args.index_n_heads() = 32;
  args.index_head_dim() = 128;
  args.index_topk() = 2048;
  args.max_position_embeddings() = 1048576;
  args.rms_norm_eps() = 1e-5f;
  args.rope_theta() = 8000000.0f;
  args.rope_scaling_rope_type() = "default";
  args.indexer_rope_interleave() = true;
  args.enable_mla() = true;
  result.quant.quant_method() = "smoothquant";
  result.quant.bits() = 8;
  result.quant.only_expert_per_group() = true;
  result.quant.activation_dynamic() = true;
  return result;
}

enum class AttentionPhase { PREFILL, CHUNKED, DECODE };
struct AttentionCase {
  DsaModelVariant variant;
  AttentionPhase phase;
  bool fused;
  std::string name;
  bool production = false;
  bool legacy = false;
  int32_t tokens = 8;
  float hidden_scale = 1.0f;
};

constexpr double kBf16Epsilon = 1.0 / 128.0;

::testing::AssertionResult fused_output_close(const torch::Tensor& actual,
                                              const torch::Tensor& expected) {
  if (actual.sizes() != expected.sizes() ||
      actual.scalar_type() != expected.scalar_type() || actual.dim() != 2 ||
      actual.numel() == 0) {
    return ::testing::AssertionFailure()
           << "Fused output requires matching nonempty matrix shape and dtype";
  }
  const torch::Tensor values =
      actual.detach().to(torch::kCPU).to(torch::kFloat64);
  const torch::Tensor reference =
      expected.detach().to(torch::kCPU).to(torch::kFloat64);
  if (!torch::isfinite(values).all().item<bool>() ||
      !torch::isfinite(reference).all().item<bool>()) {
    return ::testing::AssertionFailure() << "Fused output must be finite";
  }
  const torch::Tensor reference_rms =
      reference.square().mean(/*dim=*/-1, /*keepdim=*/true).sqrt();
  if (reference_rms.le(/*other=*/1e-6).any().item<bool>()) {
    return ::testing::AssertionFailure()
           << "Fused output reference must be nondegenerate";
  }
  const torch::Tensor error = values - reference;
  const double row_relative_l2 =
      (error.square().sum(/*dim=*/-1) / reference.square().sum(/*dim=*/-1))
          .sqrt()
          .max()
          .item<double>();
  const double local_scaled_error =
      (error.abs() / torch::maximum(reference.abs(), reference_rms))
          .max()
          .item<double>();
  if (row_relative_l2 > kBf16Epsilon) {
    return ::testing::AssertionFailure()
           << "Fused output row relative L2 " << row_relative_l2 << " exceeds "
           << kBf16Epsilon << "; local scaled error=" << local_scaled_error;
  }
  if (local_scaled_error > 2.0 * kBf16Epsilon) {
    return ::testing::AssertionFailure()
           << "Fused output local scaled error " << local_scaled_error
           << " exceeds " << 2.0 * kBf16Epsilon
           << "; row relative L2=" << row_relative_l2;
  }
  return ::testing::AssertionSuccess();
}

ModelArgs model_args(DsaModelVariant variant, bool production = false) {
  ModelArgs args = create_dsa_model_args(variant, production);
  if (variant == DsaModelVariant::V32 && !production) {
    args.rope_scaling_rope_type() = "deepseek_yarn";
    args.rope_scaling_original_max_position_embeddings() = 128;
    args.rope_scaling_factor() = 1.0f;
    args.rope_extrapolation_factor() = 1.0f;
    args.rope_scaling_attn_factor() = 1.0f;
    args.rope_scaling_beta_fast() = 32;
    args.rope_scaling_beta_slow() = 1;
    args.rope_scaling_mscale() = 1.0f;
    args.rope_scaling_mscale_all_dim() = 1.0f;
  }
  return args;
}

AttentionMetadata metadata(const torch::TensorOptions& options_,
                           int32_t prefix,
                           int32_t rows,
                           bool prefill) {
  AttentionMetadata value;
  auto ints = options_.dtype(torch::kInt32);
  const int32_t total = prefix + rows;
  value.q_cu_seq_lens = torch::tensor({0, rows}, ints);
  value.kv_cu_seq_lens = torch::tensor({0, total}, ints);
  value.kv_seq_lens = torch::tensor({total}, ints);
  value.block_table = torch::arange(kCapacity, ints).reshape({1, kCapacity});
  value.slot_mapping = torch::arange(prefix, total, ints);
  value.max_query_len = rows;
  value.max_seq_len = total;
  value.total_kv_len = total;
  value.compute_dtype = "bfloat16";
  value.is_prefill = prefill;
  value.is_chunked_prefill = !prefill && rows > 1;
  value.is_dummy = false;
  return value;
}

KVCache cache(const torch::TensorOptions& options_, const ModelArgs& args) {
  return create_indexed_kv_cache(
      test::seeded_tensor(
          "characterization.k_guard",
          {kCapacity, 1, 1, args.kv_lora_rank() + args.qk_rope_head_dim()},
          torch::kBFloat16,
          options_.device()) *
              0.01f +
          0.125f,
      test::seeded_tensor("characterization.index_guard",
                          {kCapacity, 1, 1, args.index_head_dim()},
                          torch::kBFloat16,
                          options_.device()) *
              0.01f +
          0.125f);
}

DeepseekV32Attention attention(const ParallelArgs& parallel_,
                               const torch::TensorOptions& options_,
                               const ModelArgs& args,
                               const Weights& weights,
                               bool fused,
                               bool indexer = true,
                               const QuantArgs& quant = QuantArgs{}) {
  OptimizationConfig optimization;
  optimization.enable_fused_mla_kernel = fused;
  optimization.enable_fused_indexer_qk = false;
  DeepseekV32Attention result(
      args, quant, parallel_, options_, optimization, indexer);
  result->load_state_dict(StateDict(weights));
  result->verify_loaded_weights();
  return result;
}

torch::Tensor forward(const torch::TensorOptions& options_,
                      DeepseekV32Attention& attention,
                      KVCache& cache,
                      const torch::Tensor& hidden,
                      int32_t prefix,
                      int32_t rows,
                      bool prefill,
                      DsaTopkTransfer& transfer) {
  torch::Tensor positions =
      torch::arange(prefix, prefix + rows, options_.dtype(torch::kInt32));
  return attention
      ->forward(positions,
                hidden.slice(/*dim=*/0, prefix, prefix + rows),
                metadata(options_, prefix, rows, prefill),
                cache,
                nullptr,
                &transfer)
      .output;
}

void check_selection(const DsaTopkState& state, int32_t prefix, int32_t rows) {
  torch::Tensor tables = state.flattened().block_tables().cpu();
  torch::Tensor lens = state.context_lens().cpu();
  ASSERT_EQ(lens.numel(), rows);
  for (int32_t row = 0; row < rows; ++row) {
    const int32_t expected_len = prefix + row + 1;
    ASSERT_EQ(lens[row].item<int32_t>(), expected_len);
    torch::Tensor selected = std::get<0>(
        tables[row].slice(/*dim=*/0, /*start=*/0, expected_len).sort());
    EXPECT_TRUE(
        torch::equal(selected, torch::arange(expected_len, torch::kInt32)));
  }
}

Weights decoder_weights(const torch::TensorOptions& options_,
                        const ModelArgs& args) {
  Weights weights = create_seeded_mla_weights(
      args, DsaWeightFormat::BF16, options_, "self_attn.");
  weights.merge(create_seeded_dense_weights(
      args, DsaWeightFormat::BF16, options_, "mlp."));
  for (auto& [name, tensor] : weights) {
    if (tensor.dim() != 2) {
      continue;
    }
    tensor =
        tensor - (name.starts_with("self_attn.indexer.") ? 0.025f : 0.005f);
  }
  weights["self_attn.indexer.k_norm.weight"] =
      torch::ones({args.index_head_dim()}, options_.dtype(torch::kFloat32));
  weights["self_attn.indexer.k_norm.bias"] =
      torch::zeros({args.index_head_dim()}, options_.dtype(torch::kFloat32));
  weights["input_layernorm.weight"] =
      torch::ones({args.hidden_size()}, options_.dtype(torch::kFloat32));
  weights["post_attention_layernorm.weight"] =
      torch::ones({args.hidden_size()}, options_.dtype(torch::kFloat32));
  return weights;
}

ModelInputParams params(const torch::TensorOptions& options_,
                        int32_t prefix,
                        int32_t rows) {
  ModelInputParams result;
  result.meta.batch_forward_type =
      rows == 1 ? BatchForwardType::DECODE : BatchForwardType::PREFILL;
  result.meta.num_sequences = 1;
  result.meta.q_max_seq_len = rows;
  result.meta.kv_max_seq_len = prefix + rows;
  result.parallel.dp_global_token_nums = {rows};
  return result;
}

TEST(DsaFusedOutputCalibrationTest, RejectsWholeRowDrift) {
  const torch::Tensor reference =
      torch::ones({1, 256}, torch::dtype(torch::kBFloat16));
  const torch::Tensor actual = reference * (1.0 + 1.0 / 64.0);
  // Every error is 1/64: relative L2 is 1/64, while local error meets 2 eps.
  const auto result = fused_output_close(actual, reference);
  EXPECT_FALSE(result);
  EXPECT_TRUE(std::string(result.message())
                  .starts_with("Fused output row relative L2"));
  EXPECT_TRUE(fused_output_close(reference, reference));
}

TEST(DsaFusedOutputCalibrationTest, RejectsIsolatedErrorHiddenByWideRow) {
  const torch::Tensor reference =
      torch::ones({1, 8192}, torch::dtype(torch::kBFloat16));
  torch::Tensor actual = reference.clone();
  actual[0][0] = 1.0 + 1.0 / 32.0;
  // Relative L2 is (1/32)/sqrt(8192) < eps; local error is 1/32 > 2 eps.
  const auto result = fused_output_close(actual, reference);
  EXPECT_FALSE(result);
  EXPECT_TRUE(std::string(result.message())
                  .starts_with("Fused output local scaled error"));
  EXPECT_TRUE(fused_output_close(reference, reference));
}

torch::Tensor cpu_linear(const torch::Tensor& x, const torch::Tensor& weight) {
  return torch::matmul(x.to(torch::kFloat64),
                       weight.cpu().to(torch::kFloat64).t())
      .to(torch::kBFloat16)
      .to(torch::kFloat64);
}

torch::Tensor cpu_rms(const torch::Tensor& x,
                      const torch::Tensor& gamma,
                      double eps) {
  return (x * torch::rsqrt(x.square().mean(-1, true) + eps) *
          gamma.cpu().to(torch::kFloat64))
      .to(torch::kBFloat16)
      .to(torch::kFloat64);
}

// Each adjacent coordinate pair rotates through position / theta^(2i/d).
// The small YaRN cases use factor one, so interpolation is the identity.
torch::Tensor cpu_rope(const torch::Tensor& x, double theta) {
  torch::Tensor result = torch::empty_like(x);
  const int64_t width = x.size(-1);
  for (int64_t row = 0; row < x.size(0); ++row) {
    for (int64_t pair = 0; pair < width / 2; ++pair) {
      const double angle = row / std::pow(theta, 2.0 * pair / width);
      const double cosine =
          static_cast<double>(static_cast<torch::BFloat16>(std::cos(angle)));
      const double sine =
          static_cast<double>(static_cast<torch::BFloat16>(std::sin(angle)));
      result[row].select(-1, 2 * pair) = x[row].select(-1, 2 * pair) * cosine -
                                         x[row].select(-1, 2 * pair + 1) * sine;
      result[row].select(-1, 2 * pair + 1) =
          x[row].select(-1, 2 * pair) * sine +
          x[row].select(-1, 2 * pair + 1) * cosine;
    }
  }
  return result.to(torch::kBFloat16).to(torch::kFloat64);
}

torch::Tensor cpu_attention(const torch::Tensor& hidden,
                            const ModelArgs& args,
                            const Weights& weights) {
  const int64_t rows = hidden.size(0);
  const int64_t heads = args.n_heads();
  const int64_t nope = args.qk_nope_head_dim();
  const int64_t rope = args.qk_rope_head_dim();
  const int64_t latent = args.kv_lora_rank();
  const int64_t value_dim = args.v_head_dim();
  const torch::Tensor q =
      cpu_linear(
          cpu_rms(cpu_linear(hidden.cpu(), weights.at("q_a_proj.weight")),
                  weights.at("q_a_layernorm.weight"),
                  args.rms_norm_eps()),
          weights.at("q_b_proj.weight"))
          .reshape({rows, heads, nope + rope});
  const torch::Tensor kv =
      cpu_linear(hidden.cpu(), weights.at("kv_a_proj_with_mqa.weight"));
  const torch::Tensor content = cpu_rms(kv.narrow(-1, 0, latent),
                                        weights.at("kv_a_layernorm.weight"),
                                        args.rms_norm_eps());
  const torch::Tensor expanded =
      cpu_linear(content, weights.at("kv_b_proj.weight"))
          .reshape({rows, heads, nope + value_dim});
  const torch::Tensor query_rope =
      cpu_rope(q.narrow(-1, nope, rope), args.rope_theta());
  const torch::Tensor key_rope =
      cpu_rope(kv.narrow(-1, latent, rope), args.rope_theta());
  torch::Tensor result =
      torch::empty({rows, heads, value_dim}, torch::kFloat64);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t head = 0; head < heads; ++head) {
      torch::Tensor scores =
          torch::matmul(expanded.narrow(0, 0, row + 1)
                            .select(1, head)
                            .narrow(-1, 0, nope),
                        q[row][head].narrow(0, 0, nope)) +
          torch::matmul(key_rope.narrow(0, 0, row + 1), query_rope[row][head]);
      scores /= std::sqrt(static_cast<double>(nope + rope));
      result[row][head] = torch::matmul(torch::softmax(scores, 0),
                                        expanded.narrow(0, 0, row + 1)
                                            .select(1, head)
                                            .narrow(-1, nope, value_dim));
    }
  }
  return cpu_linear(result.reshape({rows, heads * value_dim}),
                    weights.at("o_proj.weight"))
      .to(torch::kBFloat16);
}

TEST(DsaAttentionCharacterizationTest, MatchesIndependentTemporalBatching) {
  for (const auto& param : std::vector<AttentionCase>{
           AttentionCase{.variant = DsaModelVariant::V32,
                         .phase = AttentionPhase::PREFILL,
                         .fused = false,
                         .name = "V32Prefill"},
           AttentionCase{.variant = DsaModelVariant::V32,
                         .phase = AttentionPhase::CHUNKED,
                         .fused = false,
                         .name = "V32Chunked"},
           AttentionCase{.variant = DsaModelVariant::V32,
                         .phase = AttentionPhase::DECODE,
                         .fused = false,
                         .name = "V32Decode"},
           AttentionCase{.variant = DsaModelVariant::GLM5,
                         .phase = AttentionPhase::PREFILL,
                         .fused = false,
                         .name = "Glm5Prefill"},
           AttentionCase{.variant = DsaModelVariant::GLM5,
                         .phase = AttentionPhase::CHUNKED,
                         .fused = false,
                         .name = "Glm5Chunked"},
           AttentionCase{.variant = DsaModelVariant::GLM5,
                         .phase = AttentionPhase::DECODE,
                         .fused = false,
                         .name = "Glm5Decode"},
           AttentionCase{.variant = DsaModelVariant::V32,
                         .phase = AttentionPhase::DECODE,
                         .fused = true,
                         .name = "V32FusedDecode"},
           AttentionCase{.variant = DsaModelVariant::GLM5,
                         .phase = AttentionPhase::DECODE,
                         .fused = true,
                         .name = "Glm5FusedDecode"},
           AttentionCase{.variant = DsaModelVariant::V32,
                         .phase = AttentionPhase::DECODE,
                         .fused = true,
                         .name = "V32LegacySmallControl",
                         .production = false,
                         .legacy = true,
                         .tokens = 8,
                         .hidden_scale = 0.02f},
           AttentionCase{.variant = DsaModelVariant::GLM5,
                         .phase = AttentionPhase::DECODE,
                         .fused = true,
                         .name = "Glm5LegacySmallControl",
                         .production = false,
                         .legacy = true,
                         .tokens = 8,
                         .hidden_scale = 0.02f},
           AttentionCase{.variant = DsaModelVariant::V32,
                         .phase = AttentionPhase::DECODE,
                         .fused = true,
                         .name = "V32LegacyProductionControl",
                         .production = true,
                         .legacy = true,
                         .tokens = 8,
                         .hidden_scale = 0.02f},
           AttentionCase{.variant = DsaModelVariant::GLM5,
                         .phase = AttentionPhase::DECODE,
                         .fused = true,
                         .name = "Glm5LegacyProductionControl",
                         .production = true,
                         .legacy = true,
                         .tokens = 8,
                         .hidden_scale = 0.02f},
           AttentionCase{.variant = DsaModelVariant::V32,
                         .phase = AttentionPhase::DECODE,
                         .fused = true,
                         .name = "V32GenericProductionControl",
                         .production = true,
                         .legacy = false,
                         .tokens = 8,
                         .hidden_scale = 1.0f},
           AttentionCase{.variant = DsaModelVariant::GLM5,
                         .phase = AttentionPhase::DECODE,
                         .fused = true,
                         .name = "Glm5GenericProductionControl",
                         .production = true,
                         .legacy = false,
                         .tokens = 8,
                         .hidden_scale = 1.0f}}) {
    torch::set_num_threads(1);
    Device device(/*device_index=*/0);
    device.set_device();
    const auto options_ =
        torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
    auto& cache_config = KVCacheConfig::get_instance();
    const int64_t saved_block = cache_config.block_size();
    const std::string saved_dtype = cache_config.kv_cache_dtype();
    const std::string saved_index_dtype = cache_config.indexer_cache_dtype();
    auto restore_config = c10::make_scope_exit([&] {
      cache_config.block_size(saved_block);
      cache_config.kv_cache_dtype(saved_dtype);
      cache_config.indexer_cache_dtype(saved_index_dtype);
    });
    cache_config.block_size(1);
    cache_config.kv_cache_dtype("auto");
    cache_config.indexer_cache_dtype("auto");
    std::unique_ptr<ProcessGroup> group_;
    const ParallelArgs parallel_ = test::create_default_parallel_args(group_);

    torch::NoGradGuard no_grad;
    const AttentionCase& config = param;
    const ModelArgs args = model_args(config.variant, config.production);
    const DsaWeightFormat format =
        config.fused ? DsaWeightFormat::SMOOTHQUANT : DsaWeightFormat::BF16;
    Weights weights;
    if (config.legacy) {
      const Weights checkpoint = create_mla_weights(args, options_);
      const std::string prefix = "model.layers.0.self_attn.";
      for (const auto& [key, tensor] : checkpoint) {
        weights.emplace(key.substr(prefix.size()), tensor);
      }
    } else {
      weights = create_seeded_mla_weights(args, format, options_);
    }
    const QuantArgs quant =
        config.fused ? test::create_default_quant_args() : QuantArgs{};
    auto actual = attention(parallel_,
                            options_,
                            args,
                            weights,
                            config.fused,
                            /*indexer=*/true,
                            quant);
    auto reference = attention(parallel_,
                               options_,
                               args,
                               weights,
                               /*fused=*/false,
                               /*indexer=*/true,
                               quant);
    auto actual_cache = cache(options_, args);
    auto reference_cache = cache(options_, args);
    const torch::Tensor initial_k = actual_cache.get_k_cache().clone();
    const torch::Tensor initial_index = actual_cache.get_index_cache().clone();
    const torch::Tensor hidden =
        test::seeded_tensor("characterization.attention.hidden",
                            {config.tokens, args.hidden_size()},
                            torch::kBFloat16,
                            options_.device()) *
        config.hidden_scale;
    std::vector<torch::Tensor> reference_rows;
    reference_rows.reserve(config.tokens);
    for (int32_t row = 0; row < config.tokens; ++row) {
      auto reference_capture = DsaTopkTransfer::capture_output();
      reference_rows.emplace_back(forward(options_,
                                          reference,
                                          reference_cache,
                                          hidden,
                                          row,
                                          /*rows=*/1,
                                          /*prefill=*/false,
                                          reference_capture));
      ASSERT_NE(reference_capture.output(), nullptr);
      check_selection(*reference_capture.output(), row, /*rows=*/1);
    }
    const torch::Tensor expected = torch::cat(reference_rows, /*dim=*/0);
    if (!config.production && !config.fused) {
      expect_close(expected,
                   cpu_attention(hidden, args, weights),
                   /*rtol=*/0.04,
                   /*atol=*/0.001,
                   config.name + " CPU attention equation");
    }
    int32_t prefix = 0;
    int32_t rows = config.tokens;
    if (config.phase != AttentionPhase::PREFILL) {
      prefix = config.phase == AttentionPhase::CHUNKED ? 3 : config.tokens - 1;
      rows -= prefix;
      if (prefix > 0) {
        auto prefix_capture = DsaTopkTransfer::capture_output();
        forward(options_,
                actual,
                actual_cache,
                hidden,
                /*prefix=*/0,
                prefix,
                /*prefill=*/true,
                prefix_capture);
      }
    }
    torch::Tensor expected_output =
        expected.slice(/*dim=*/0, prefix, config.tokens);
    if (config.fused) {
      reference_cache =
          create_indexed_kv_cache(actual_cache.get_k_cache().clone(),
                                  actual_cache.get_index_cache().clone());
      auto reference_capture = DsaTopkTransfer::capture_output();
      expected_output = forward(options_,
                                reference,
                                reference_cache,
                                hidden,
                                prefix,
                                rows,
                                /*prefill=*/false,
                                reference_capture);
    }
    auto capture = DsaTopkTransfer::capture_output();
    const torch::Tensor output =
        forward(options_,
                actual,
                actual_cache,
                hidden,
                prefix,
                rows,
                config.phase == AttentionPhase::PREFILL,
                capture);
    ASSERT_NE(capture.output(), nullptr);
    check_selection(*capture.output(), prefix, rows);
    ASSERT_TRUE(torch::isfinite(output).all().item<bool>());

    if (config.fused) {
      EXPECT_TRUE(fused_output_close(output, expected_output)) << config.name;
    } else {
      expect_close(
          output, expected_output, /*rtol=*/0.04, /*atol=*/0.001, config.name);
    }
    expect_close(
        actual_cache.get_k_cache().slice(/*dim=*/0, /*start=*/0, config.tokens),
        reference_cache.get_k_cache().slice(
            /*dim=*/0, /*start=*/0, config.tokens),
        /*rtol=*/0.02,
        /*atol=*/0.01);
    expect_close(actual_cache.get_index_cache().slice(
                     /*dim=*/0, /*start=*/0, config.tokens),
                 reference_cache.get_index_cache().slice(
                     /*dim=*/0, /*start=*/0, config.tokens),
                 /*rtol=*/0.02,
                 /*atol=*/0.01);
    EXPECT_TRUE(
        torch::equal(actual_cache.get_k_cache().slice(/*dim=*/0, config.tokens),
                     initial_k.slice(/*dim=*/0, config.tokens)));
    EXPECT_TRUE(torch::equal(
        actual_cache.get_index_cache().slice(/*dim=*/0, config.tokens),
        initial_index.slice(/*dim=*/0, config.tokens)));
    EXPECT_FALSE(torch::equal(
        actual_cache.get_k_cache().slice(/*dim=*/0, /*start=*/0, config.tokens),
        initial_k.slice(/*dim=*/0, /*start=*/0, config.tokens)));
    EXPECT_FALSE(torch::equal(
        actual_cache.get_index_cache().slice(
            /*dim=*/0, /*start=*/0, config.tokens),
        initial_index.slice(/*dim=*/0, /*start=*/0, config.tokens)));
  }
}

TEST(DsaSingleCardTest, C12CompleteWeightsVerifyAndRun) {
  torch::set_num_threads(1);
  Device device(/*device_index=*/0);
  device.set_device();
  const auto options_ =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
  auto& cache_config = KVCacheConfig::get_instance();
  const int64_t saved_block = cache_config.block_size();
  const std::string saved_dtype = cache_config.kv_cache_dtype();
  const std::string saved_index_dtype = cache_config.indexer_cache_dtype();
  auto restore_config = c10::make_scope_exit([&] {
    cache_config.block_size(saved_block);
    cache_config.kv_cache_dtype(saved_dtype);
    cache_config.indexer_cache_dtype(saved_index_dtype);
  });
  cache_config.block_size(1);
  cache_config.kv_cache_dtype("auto");
  cache_config.indexer_cache_dtype("auto");
  std::unique_ptr<ProcessGroup> group_;
  const ParallelArgs parallel_ = test::create_default_parallel_args(group_);

  torch::NoGradGuard no_grad;
  const ModelArgs args = model_args(DsaModelVariant::V32);
  const Weights weights =
      create_seeded_mla_weights(args, DsaWeightFormat::BF16, options_);
  auto loaded = attention(parallel_, options_, args, weights, /*fused=*/false);
  auto resident = cache(options_, args);
  const torch::Tensor hidden =
      test::seeded_tensor("characterization.loading.hidden",
                          {1, args.hidden_size()},
                          torch::kBFloat16,
                          options_.device());
  auto capture = DsaTopkTransfer::capture_output();
  const torch::Tensor output = forward(options_,
                                       loaded,
                                       resident,
                                       hidden,
                                       /*prefix=*/0,
                                       /*rows=*/1,
                                       /*prefill=*/false,
                                       capture);
  ASSERT_NE(capture.output(), nullptr);
  check_selection(*capture.output(), /*prefix=*/0, /*rows=*/1);
  ASSERT_TRUE(torch::isfinite(output).all().item<bool>());
  EXPECT_GT(output.abs().max().item<float>(), 0.0f);
  expect_close(output,
               cpu_attention(hidden, args, weights),
               /*rtol=*/0.04,
               /*atol=*/0.001,
               "complete checkpoint CPU equation");
}

TEST(DsaSingleCardTest, C12SharedIgnoresIndexerCheckpoint) {
  torch::set_num_threads(1);
  Device device(/*device_index=*/0);
  device.set_device();
  const auto options_ =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
  auto& cache_config = KVCacheConfig::get_instance();
  const int64_t saved_block = cache_config.block_size();
  const std::string saved_dtype = cache_config.kv_cache_dtype();
  const std::string saved_index_dtype = cache_config.indexer_cache_dtype();
  auto restore_config = c10::make_scope_exit([&] {
    cache_config.block_size(saved_block);
    cache_config.kv_cache_dtype(saved_dtype);
    cache_config.indexer_cache_dtype(saved_index_dtype);
  });
  cache_config.block_size(1);
  cache_config.kv_cache_dtype("auto");
  cache_config.indexer_cache_dtype("auto");
  std::unique_ptr<ProcessGroup> group_;
  const ParallelArgs parallel_ = test::create_default_parallel_args(group_);

  torch::NoGradGuard no_grad;
  const ModelArgs args = model_args(DsaModelVariant::GLM5);
  Weights complete =
      create_seeded_mla_weights(args, DsaWeightFormat::BF16, options_);
  Weights omitted;
  for (const auto& [key, tensor] : complete) {
    if (!key.starts_with("indexer.")) {
      omitted.emplace(key, tensor);
    }
  }
  auto supplied = attention(
      parallel_, options_, args, complete, /*fused=*/false, /*indexer=*/false);
  auto absent = attention(
      parallel_, options_, args, omitted, /*fused=*/false, /*indexer=*/false);
  auto supplied_cache = cache(options_, args);
  auto absent_cache = cache(options_, args);
  const torch::Tensor index_before = supplied_cache.get_index_cache().clone();
  const torch::Tensor hidden =
      test::seeded_tensor("characterization.loading.hidden",
                          {1, args.hidden_size()},
                          torch::kBFloat16,
                          options_.device());
  const DsaTopkState fixed(
      torch::zeros({1, args.index_topk()}, options_.dtype(torch::kInt32)),
      torch::ones({1}, options_.dtype(torch::kInt32)));
  auto supplied_transfer = DsaTopkTransfer::reuse(fixed);
  auto absent_transfer = DsaTopkTransfer::reuse(fixed);
  const torch::Tensor actual = forward(options_,
                                       supplied,
                                       supplied_cache,
                                       hidden,
                                       /*prefix=*/0,
                                       /*rows=*/1,
                                       /*prefill=*/false,
                                       supplied_transfer);
  const torch::Tensor expected = forward(options_,
                                         absent,
                                         absent_cache,
                                         hidden,
                                         /*prefix=*/0,
                                         /*rows=*/1,
                                         /*prefill=*/false,
                                         absent_transfer);
  EXPECT_TRUE(torch::equal(actual, expected));
  EXPECT_TRUE(torch::equal(supplied_cache.get_index_cache(), index_before));
  EXPECT_TRUE(torch::equal(absent_cache.get_index_cache(), index_before));
  EXPECT_TRUE(
      torch::equal(supplied_cache.get_k_cache(), absent_cache.get_k_cache()));
}

TEST(DsaSingleCardTest, DISABLED_C12FullMissingIndexerMustBeRejected) {
  torch::set_num_threads(1);
  Device device(/*device_index=*/0);
  device.set_device();
  const auto options_ =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
  auto& cache_config = KVCacheConfig::get_instance();
  const int64_t saved_block = cache_config.block_size();
  const std::string saved_dtype = cache_config.kv_cache_dtype();
  const std::string saved_index_dtype = cache_config.indexer_cache_dtype();
  auto restore_config = c10::make_scope_exit([&] {
    cache_config.block_size(saved_block);
    cache_config.kv_cache_dtype(saved_dtype);
    cache_config.indexer_cache_dtype(saved_index_dtype);
  });
  cache_config.block_size(1);
  cache_config.kv_cache_dtype("auto");
  cache_config.indexer_cache_dtype("auto");
  std::unique_ptr<ProcessGroup> group_;
  const ParallelArgs parallel_ = test::create_default_parallel_args(group_);

  const ModelArgs args = model_args(DsaModelVariant::V32);
  const Weights complete =
      create_seeded_mla_weights(args, DsaWeightFormat::BF16, options_);
  Weights omitted;
  for (const auto& [key, tensor] : complete) {
    if (!key.starts_with("indexer.")) {
      omitted.emplace(key, tensor);
    }
  }
  OptimizationConfig optimization;
  DeepseekV32Attention full(
      args, QuantArgs{}, parallel_, options_, optimization);
  full->load_state_dict(StateDict(omitted));
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ASSERT_DEATH(full->verify_loaded_weights(), "indexer");
}

TEST(DsaRelayCharacterizationTest,
     C4SparseFssfsMatchesFixedSelectionReference) {
  for (const auto& param : std::vector<bool>{false, true}) {
    torch::set_num_threads(1);
    Device device(/*device_index=*/0);
    device.set_device();
    const auto options_ =
        torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
    auto& cache_config = KVCacheConfig::get_instance();
    const int64_t saved_block = cache_config.block_size();
    const std::string saved_dtype = cache_config.kv_cache_dtype();
    const std::string saved_index_dtype = cache_config.indexer_cache_dtype();
    auto restore_config = c10::make_scope_exit([&] {
      cache_config.block_size(saved_block);
      cache_config.kv_cache_dtype(saved_dtype);
      cache_config.indexer_cache_dtype(saved_index_dtype);
    });
    cache_config.block_size(1);
    cache_config.kv_cache_dtype("auto");
    cache_config.indexer_cache_dtype("auto");
    std::unique_ptr<ProcessGroup> group_;
    const ParallelArgs parallel_ = test::create_default_parallel_args(group_);

    torch::NoGradGuard no_grad;
    const bool decode = param;
    ModelArgs args = model_args(DsaModelVariant::GLM52_SHARE);
    args.n_layers() = 5;
    args.first_k_dense_replace() = 5;
    args.index_topk_pattern() = "FSSFS";
    args.index_topk() = 8;
    args.intermediate_size() = 256;
    const ModelContext context(parallel_, args, QuantArgs{}, options_);
    const DsaTopkSharePlan plan(args);
    const Weights common_weights = decoder_weights(options_, args);
    std::vector<DeepseekV32DecoderLayer> layers;
    std::vector<Weights> layer_weights;
    std::vector<KVCache> caches;
    layers.reserve(5);
    layer_weights.reserve(5);
    caches.reserve(5);
    for (int32_t layer = 0; layer < 5; ++layer) {
      Weights weights = common_weights;
      if (layer == 3) {
        weights["self_attn.indexer.weights_proj.weight"] =
            -weights.at("self_attn.indexer.weights_proj.weight");
      }
      DeepseekV32DecoderLayer decoder(context, layer, plan);
      decoder->load_state_dict(StateDict(weights));
      decoder->verify_loaded_weights();
      layers.emplace_back(std::move(decoder));
      layer_weights.emplace_back(std::move(weights));
      caches.emplace_back(cache(options_, args));
    }
    const torch::Tensor hidden =
        test::seeded_tensor("characterization.relay.hidden",
                            {33, args.hidden_size()},
                            torch::kBFloat16,
                            options_.device()) -
        0.5f;
    if (decode) {
      torch::Tensor history =
          hidden.slice(/*dim=*/0, /*start=*/0, /*end=*/32).clone();
      torch::Tensor positions =
          torch::arange(32, options_.dtype(torch::kInt32));
      const AttentionMetadata prefill =
          metadata(options_, /*prefix=*/0, /*rows=*/32, /*prefill=*/true);
      const ModelInputParams input =
          params(options_, /*prefix=*/0, /*rows=*/32);
      DsaTopkRelay history_relay;
      for (int32_t layer = 0; layer < 5; ++layer) {
        std::optional<torch::Tensor> residual;
        history = layers[layer]->forward(history,
                                         residual,
                                         positions,
                                         prefill,
                                         caches[layer],
                                         input,
                                         std::nullopt,
                                         &history_relay);
      }
    }
    const int32_t prefix = decode ? 32 : 0;
    const int32_t rows = decode ? 1 : 32;
    torch::Tensor x = hidden.slice(/*dim=*/0, prefix, prefix + rows).clone();
    torch::Tensor positions =
        torch::arange(prefix, prefix + rows, options_.dtype(torch::kInt32));
    const AttentionMetadata meta = metadata(options_, prefix, rows, !decode);
    const ModelInputParams input = params(options_, prefix, rows);
    DsaTopkRelay relay;
    DsaTopkShareDecision reuse;
    reuse.reuse_topk = true;
    DsaTopkShareDecision publish;
    publish.output_topk = true;
    torch::Tensor first_selection;

    for (int32_t layer = 0; layer < 5; ++layer) {
      const std::string tag = "layer" + std::to_string(layer);
      const torch::Tensor index_before =
          caches[layer].get_index_cache().clone();
      auto reference_cache = create_indexed_kv_cache(
          caches[layer].get_k_cache().clone(), index_before.clone());
      torch::Tensor reference_x = x.clone();
      std::optional<DsaTopkTransfer> upstream;
      DsaTopkRelay reference_relay;
      if (plan.decision_for(layer).reuse_topk) {
        upstream = relay.prepare_layer(reuse);
        ASSERT_NE(upstream->input(), nullptr);
        auto captured = DsaTopkTransfer::capture_output();
        captured.publish_output(*upstream->input());
        reference_relay.finish_layer(publish, captured);
      }
      std::optional<torch::Tensor> residual;
      x = layers[layer]->forward(x,
                                 residual,
                                 positions,
                                 meta,
                                 caches[layer],
                                 input,
                                 std::nullopt,
                                 &relay);
      EXPECT_FALSE(residual.has_value());
      ASSERT_TRUE(torch::isfinite(x).all().item<bool>());
      auto observed = relay.prepare_layer(reuse);
      ASSERT_NE(observed->input(), nullptr);
      const DsaTopkState& state = *observed->input();
      EXPECT_EQ(state.context_lens()[-1].item<int32_t>(), args.index_topk());
      if (layer == 0) {
        first_selection =
            active_topk(state.flattened().block_tables(), state.context_lens());
      }
      if (layer == 3) {
        const torch::Tensor prior = std::get<0>(first_selection[-1].sort());
        const torch::Tensor refreshed =
            std::get<0>(active_topk(state.flattened().block_tables(),
                                    state.context_lens())[-1]
                            .sort());
        ASSERT_FALSE(torch::equal(refreshed, prior))
            << "the sparse producer fixture must refresh its selection";
      }
      if (!upstream.has_value()) {
        OptimizationConfig optimization;
        optimization.enable_fused_mla_kernel = false;
        optimization.enable_fused_indexer_qk = false;
        DeepseekV32Attention fixed_attention(args,
                                             QuantArgs{},
                                             parallel_,
                                             options_,
                                             optimization,
                                             /*enable_indexer=*/false);
        fixed_attention->load_state_dict(
            StateDict(layer_weights[layer]).get_dict_with_prefix("self_attn."));
        auto fixed = DsaTopkTransfer::reuse(state);
        const torch::Tensor normalized =
            cpu_rms(reference_x.cpu().to(torch::kFloat64),
                    layer_weights[layer].at("input_layernorm.weight"),
                    args.rms_norm_eps());
        const torch::Tensor attn_out = fixed_attention
                                           ->forward(positions,
                                                     normalized.to(options_),
                                                     meta,
                                                     reference_cache,
                                                     /*cp_ctx=*/nullptr,
                                                     &fixed)
                                           .output;
        const torch::Tensor skip = (reference_x.cpu().to(torch::kFloat64) +
                                    attn_out.cpu().to(torch::kFloat64))
                                       .to(torch::kBFloat16)
                                       .to(torch::kFloat64);
        const torch::Tensor ffn_in =
            cpu_rms(skip,
                    layer_weights[layer].at("post_attention_layernorm.weight"),
                    args.rms_norm_eps());
        const torch::Tensor gate =
            cpu_linear(ffn_in, layer_weights[layer].at("mlp.gate_proj.weight"));
        const torch::Tensor up =
            cpu_linear(ffn_in, layer_weights[layer].at("mlp.up_proj.weight"));
        const torch::Tensor activation = (gate * torch::sigmoid(gate) * up)
                                             .to(torch::kBFloat16)
                                             .to(torch::kFloat64);
        const torch::Tensor expected =
            (skip + cpu_linear(activation,
                               layer_weights[layer].at("mlp.down_proj.weight")))
                .to(torch::kBFloat16);
        expect_close(x,
                     expected,
                     /*rtol=*/0.02,
                     /*atol=*/0.001,
                     tag + " fixed selection CPU FFN");
        EXPECT_TRUE(torch::equal(caches[layer].get_k_cache(),
                                 reference_cache.get_k_cache()));
      }
      if (upstream.has_value()) {
        EXPECT_TRUE(torch::equal(
            active_topk(state.flattened().block_tables(), state.context_lens()),
            active_topk(upstream->input()->flattened().block_tables(),
                        upstream->input()->context_lens())));
        EXPECT_TRUE(torch::equal(state.context_lens(),
                                 upstream->input()->context_lens()));
        DeepseekV32DecoderLayer reference(context, layer, plan);
        reference->load_state_dict(StateDict(layer_weights[layer]));
        std::optional<torch::Tensor> reference_residual;
        reference_x = reference->forward(reference_x,
                                         reference_residual,
                                         positions,
                                         meta,
                                         reference_cache,
                                         input,
                                         std::nullopt,
                                         &reference_relay);
        expect_close(x, reference_x, /*rtol=*/0.02, /*atol=*/0.001, tag);
        EXPECT_TRUE(torch::equal(caches[layer].get_k_cache(),
                                 reference_cache.get_k_cache()));
        EXPECT_TRUE(
            torch::equal(caches[layer].get_index_cache(), index_before));
      }
    }
  }
}

TEST(DsaRelayCharacterizationTest, C4DummyDoesNotRequireUpstreamSelection) {
  for (const auto& param : std::vector<bool>{false, true}) {
    torch::set_num_threads(1);
    Device device(/*device_index=*/0);
    device.set_device();
    const auto options_ =
        torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
    auto& cache_config = KVCacheConfig::get_instance();
    const int64_t saved_block = cache_config.block_size();
    const std::string saved_dtype = cache_config.kv_cache_dtype();
    const std::string saved_index_dtype = cache_config.indexer_cache_dtype();
    auto restore_config = c10::make_scope_exit([&] {
      cache_config.block_size(saved_block);
      cache_config.kv_cache_dtype(saved_dtype);
      cache_config.indexer_cache_dtype(saved_index_dtype);
    });
    cache_config.block_size(1);
    cache_config.kv_cache_dtype("auto");
    cache_config.indexer_cache_dtype("auto");
    std::unique_ptr<ProcessGroup> group_;
    const ParallelArgs parallel_ = test::create_default_parallel_args(group_);

    torch::NoGradGuard no_grad;
    ModelArgs args = model_args(DsaModelVariant::GLM52_SHARE);
    args.n_layers() = 5;
    args.first_k_dense_replace() = 5;
    args.index_topk_pattern() = "FSSFS";
    const ModelContext context(parallel_, args, QuantArgs{}, options_);
    DeepseekV32DecoderLayer shared(
        context, /*layer_id=*/1, DsaTopkSharePlan(args));
    shared->load_state_dict(StateDict(decoder_weights(options_, args)));
    auto resident = cache(options_, args);
    const torch::Tensor k_before = resident.get_k_cache().clone();
    const torch::Tensor index_before = resident.get_index_cache().clone();
    torch::Tensor hidden = test::seeded_tensor("characterization.dummy.hidden",
                                               {1, args.hidden_size()},
                                               torch::kBFloat16,
                                               options_.device());
    torch::Tensor positions = torch::zeros({1}, options_.dtype(torch::kInt32));
    AttentionMetadata meta =
        metadata(options_, /*prefix=*/0, /*rows=*/1, /*prefill=*/!param);
    meta.is_dummy = true;
    DsaTopkRelay empty;
    std::optional<torch::Tensor> residual;
    const torch::Tensor output =
        shared->forward(hidden,
                        residual,
                        positions,
                        meta,
                        resident,
                        params(options_, /*prefix=*/0, /*rows=*/1),
                        std::nullopt,
                        &empty);
    EXPECT_EQ(output.sizes(), torch::IntArrayRef({1, args.hidden_size()}));
    EXPECT_TRUE(torch::equal(resident.get_k_cache(), k_before));
    EXPECT_TRUE(torch::equal(resident.get_index_cache(), index_before));
  }
}

TEST(DsaQuantCacheCharacterizationTest,
     C6Int8PrefillThenDecodeMatchesBf16Cache) {
  for (const auto& param : std::vector<DsaModelVariant>{
           DsaModelVariant::V32, DsaModelVariant::GLM5}) {
    torch::set_num_threads(1);
    Device device(/*device_index=*/0);
    device.set_device();
    const auto options_ =
        torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
    auto& cache_config = KVCacheConfig::get_instance();
    const int64_t saved_block = cache_config.block_size();
    const std::string saved_dtype = cache_config.kv_cache_dtype();
    const std::string saved_index_dtype = cache_config.indexer_cache_dtype();
    auto restore_config = c10::make_scope_exit([&] {
      cache_config.block_size(saved_block);
      cache_config.kv_cache_dtype(saved_dtype);
      cache_config.indexer_cache_dtype(saved_index_dtype);
    });
    cache_config.block_size(1);
    cache_config.kv_cache_dtype("auto");
    cache_config.indexer_cache_dtype("auto");
    std::unique_ptr<ProcessGroup> group_;
    const ParallelArgs parallel_ = test::create_default_parallel_args(group_);

    torch::NoGradGuard no_grad;
    ModelArgs args = model_args(param);
    args.kv_lora_rank() = 512;
    const Weights weights =
        create_seeded_mla_weights(args, DsaWeightFormat::BF16, options_);
    KVCacheConfig::get_instance().kv_cache_dtype("int8");
    auto quantized =
        attention(parallel_, options_, args, weights, /*fused=*/false);
    KVCacheConfig::get_instance().kv_cache_dtype("auto");
    auto reference =
        attention(parallel_, options_, args, weights, /*fused=*/false);
    auto reference_cache = cache(options_, args);
    const torch::Tensor initial_index =
        reference_cache.get_index_cache().clone();
    const torch::Tensor initial_quant =
        torch::full(reference_cache.get_k_cache().sizes(),
                    73,
                    options_.dtype(torch::kInt8));
    const torch::Tensor initial_scale =
        torch::full({kCapacity, 1, 1}, 17.0f, options_.dtype(torch::kFloat32));
    IndexedKVCacheTensors tensors;
    tensors.kv_cache_tensors.key_cache = initial_quant.clone();
    tensors.index_cache = initial_index.clone();
    tensors.key_cache_scale = initial_scale.clone();
    KVCache quantized_cache(tensors);
    const torch::Tensor hidden =
        test::seeded_tensor("characterization.int8.hidden",
                            {8, args.hidden_size()},
                            torch::kBFloat16,
                            options_.device());

    for (int32_t step = 0; step < 2; ++step) {
      const int32_t prefix = step == 0 ? 0 : 7;
      const int32_t rows = step == 0 ? 7 : 1;
      const int32_t total = prefix + rows;
      const std::string tag = step == 0 ? "prefill" : "decode";
      auto actual_capture = DsaTopkTransfer::capture_output();
      auto reference_capture = DsaTopkTransfer::capture_output();
      const torch::Tensor output = forward(options_,
                                           quantized,
                                           quantized_cache,
                                           hidden,
                                           prefix,
                                           rows,
                                           /*prefill=*/step == 0,
                                           actual_capture);
      const torch::Tensor expected = forward(options_,
                                             reference,
                                             reference_cache,
                                             hidden,
                                             prefix,
                                             rows,
                                             /*prefill=*/step == 0,
                                             reference_capture);
      ASSERT_NE(actual_capture.output(), nullptr);
      check_selection(*actual_capture.output(), prefix, rows);
      expect_close(output, expected, /*rtol=*/0.1, /*atol=*/0.02, tag);
      const auto scale = quantized_cache.get_k_cache_scale();
      ASSERT_TRUE(scale.has_value());
      const torch::Tensor active_scale =
          scale->slice(/*dim=*/0, /*start=*/0, total);
      EXPECT_TRUE(torch::isfinite(active_scale).all().item<bool>());
      EXPECT_TRUE((active_scale > 0).all().item<bool>());
      EXPECT_TRUE(torch::equal(scale->slice(/*dim=*/0, total),
                               initial_scale.slice(/*dim=*/0, total)));
      EXPECT_TRUE(
          torch::equal(quantized_cache.get_k_cache().slice(/*dim=*/0, total),
                       initial_quant.slice(/*dim=*/0, total)));
      EXPECT_TRUE(torch::equal(quantized_cache.get_index_cache(),
                               reference_cache.get_index_cache()));
      const torch::Tensor dequantized =
          quantized_cache.get_k_cache()
              .slice(/*dim=*/0, /*start=*/0, total)
              .to(torch::kFloat32) *
          active_scale.unsqueeze(/*dim=*/-1);
      const torch::Tensor original = reference_cache.get_k_cache()
                                         .slice(/*dim=*/0, /*start=*/0, total)
                                         .to(torch::kFloat32);
      const torch::Tensor quant_error = (dequantized - original).abs();
      EXPECT_TRUE(
          (quant_error <= active_scale.unsqueeze(/*dim=*/-1) * 0.51f + 0.0001f)
              .all()
              .item<bool>())
          << "dynamic int8 cache exceeds half a quantization step";
    }
  }
}

}  // namespace
}  // namespace xllm::layer
