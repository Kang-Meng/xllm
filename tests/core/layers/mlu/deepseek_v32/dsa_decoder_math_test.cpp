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

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/config/kv_cache_config.h"
#include "core/framework/model_context.h"
#include "core/layers/mlu/deepseek_v32/deepseek_v32_decoder_layer.h"
#include "platform/device.h"
#include "tests/core/layers/mlu/tests_utils.h"

namespace xllm::layer {
namespace {
using Weights = std::unordered_map<std::string, torch::Tensor>;

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

ModelArgs create_attention_model_args() {
  ModelArgs args;
  args.model_type() = "deepseek_v32";
  args.hidden_size() = 256;
  args.n_heads() = 4;
  args.max_position_embeddings() = 128;
  args.rope_theta() = 10000.0f;
  args.rms_norm_eps() = 1e-6f;
  args.q_lora_rank() = 64;
  args.kv_lora_rank() = 256;
  args.qk_nope_head_dim() = 128;
  args.qk_rope_head_dim() = 64;
  args.v_head_dim() = 128;
  args.index_n_heads() = 64;
  args.index_head_dim() = 128;
  args.index_topk() = 8;
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

ModelArgs create_glm5_model_args() {
  ModelArgs args = create_attention_model_args();
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
  args.model_type() = "glm_moe_dsa";
  args.index_topk() = 16;
  args.qk_nope_head_dim() = 192;
  args.v_head_dim() = 256;
  args.index_n_heads() = 32;
  args.rope_theta() = 1000000.0f;
  args.rope_scaling_rope_type() = "default";
  args.indexer_rope_interleave() = true;
  args.head_dim() = args.qk_nope_head_dim() + args.qk_rope_head_dim();
  return args;
}

Weights create_seeded_moe_weights(const ModelArgs& model_args,
                                  const torch::TensorOptions& options) {
  std::unordered_map<std::string, torch::Tensor> weight_dict;
  const int64_t num_experts = model_args.n_routed_experts();
  const int64_t hidden_size = model_args.hidden_size();
  const int64_t intermediate_size = model_args.moe_intermediate_size();

  for (int64_t expert_id = 0; expert_id < num_experts; ++expert_id) {
    const std::string expert_prefix =
        "experts." + std::to_string(expert_id) + ".";
    const std::string seed_prefix =
        "deepseek_v2_sparse_moe_block.expert_" + std::to_string(expert_id);
    weight_dict[expert_prefix + "gate_proj.weight"] =
        test::seeded_tensor(seed_prefix + ".gate_proj",
                            {intermediate_size, hidden_size},
                            torch::kBFloat16,
                            options.device());
    weight_dict[expert_prefix + "up_proj.weight"] =
        test::seeded_tensor(seed_prefix + ".up_proj",
                            {intermediate_size, hidden_size},
                            torch::kBFloat16,
                            options.device());
    weight_dict[expert_prefix + "down_proj.weight"] =
        test::seeded_tensor(seed_prefix + ".down_proj",
                            {hidden_size, intermediate_size},
                            torch::kBFloat16,
                            options.device());
  }

  weight_dict["gate.weight"] =
      test::seeded_tensor("deepseek_v2_sparse_moe_block.gate",
                          {num_experts, hidden_size},
                          torch::kBFloat16,
                          options.device());

  const int64_t n_shared_experts = model_args.n_shared_experts();
  if (n_shared_experts > 0) {
    const int64_t shared_size = intermediate_size * n_shared_experts;
    weight_dict["shared_experts.gate_proj.weight"] =
        test::seeded_tensor("deepseek_v2_sparse_moe_block.shared.gate_proj",
                            {shared_size, hidden_size},
                            torch::kBFloat16,
                            options.device());
    weight_dict["shared_experts.up_proj.weight"] =
        test::seeded_tensor("deepseek_v2_sparse_moe_block.shared.up_proj",
                            {shared_size, hidden_size},
                            torch::kBFloat16,
                            options.device());
    weight_dict["shared_experts.down_proj.weight"] =
        test::seeded_tensor("deepseek_v2_sparse_moe_block.shared.down_proj",
                            {hidden_size, shared_size},
                            torch::kBFloat16,
                            options.device());
  }

  weight_dict["gate.e_score_correction_bias"] =
      torch::zeros({num_experts}, options.dtype(torch::kFloat32));
  return weight_dict;
}

Weights create_seeded_linear_weights(int64_t out,
                                     int64_t in,
                                     const torch::TensorOptions& options,
                                     const std::string& seed,
                                     const std::string& prefix) {
  Weights weights;
  weights[prefix + "weight"] =
      test::seeded_tensor(
          seed + ".weight", {out, in}, torch::kBFloat16, options.device()) *
      0.01f;
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
                                    const torch::TensorOptions& options,
                                    const std::string& prefix = "") {
  Weights weights = create_seeded_linear_weights(args.intermediate_size(),
                                                 args.hidden_size(),
                                                 options,
                                                 prefix + "gate_proj",
                                                 prefix + "gate_proj.");
  weights.merge(create_seeded_linear_weights(args.intermediate_size(),
                                             args.hidden_size(),
                                             options,
                                             prefix + "up_proj",
                                             prefix + "up_proj."));
  weights.merge(create_seeded_linear_weights(args.hidden_size(),
                                             args.intermediate_size(),
                                             options,
                                             prefix + "down_proj",
                                             prefix + "down_proj."));
  return weights;
}

Weights create_seeded_mla_weights(const ModelArgs& args,
                                  const torch::TensorOptions& options,
                                  const std::string& prefix) {
  Weights weights;
  auto add = [&](const std::string& name, int64_t out, int64_t in) {
    weights.merge(create_seeded_linear_weights(
        out, in, options, prefix + name, prefix + name + "."));
  };
  add("q_a_proj", args.q_lora_rank(), args.hidden_size());
  add("q_b_proj",
      args.n_heads() * (args.qk_nope_head_dim() + args.qk_rope_head_dim()),
      args.q_lora_rank());
  add("kv_a_proj_with_mqa",
      args.kv_lora_rank() + args.qk_rope_head_dim(),
      args.hidden_size());
  add("kv_b_proj",
      args.n_heads() * (args.qk_nope_head_dim() + args.v_head_dim()),
      args.kv_lora_rank());
  add("o_proj", args.hidden_size(), args.n_heads() * args.v_head_dim());
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

constexpr int32_t kBlocks = 16;
constexpr int32_t kHistory = 6;
constexpr int32_t kPrefill = 8;
constexpr double kRtol = 0.03;
constexpr double kAtol = 0.003;

class BlockSizeGuard final {
 public:
  BlockSizeGuard() : saved_(KVCacheConfig::get_instance().block_size()) {
    KVCacheConfig::get_instance().block_size(/*value=*/1);
  }
  ~BlockSizeGuard() { KVCacheConfig::get_instance().block_size(saved_); }

 private:
  int64_t saved_;
};

torch::Tensor round_bf16(const torch::Tensor& value) {
  return value.to(torch::kBFloat16).to(torch::kFloat32);
}

torch::Tensor rms_reference(const torch::Tensor& input,
                            const torch::Tensor& weight,
                            double eps) {
  const torch::Tensor x = input.to(torch::kFloat32);
  return round_bf16(x *
                    torch::rsqrt(x.square().mean(/*dim=*/-1,
                                                 /*keepdim=*/true) +
                                 eps) *
                    weight.to(torch::kFloat32));
}

torch::Tensor linear_reference(const torch::Tensor& input,
                               const torch::Tensor& weight) {
  return round_bf16(
      torch::matmul(input.to(torch::kFloat32), weight.to(torch::kFloat32).t()));
}

torch::Tensor swiglu_reference(const torch::Tensor& input,
                               const Weights& weights,
                               const std::string& prefix) {
  const torch::Tensor gate =
      linear_reference(input, weights.at(prefix + "gate_proj.weight"));
  const torch::Tensor up =
      linear_reference(input, weights.at(prefix + "up_proj.weight"));
  const torch::Tensor activated = round_bf16(gate * torch::sigmoid(gate) * up);
  return linear_reference(activated, weights.at(prefix + "down_proj.weight"));
}

torch::Tensor moe_reference(const torch::Tensor& input,
                            const Weights& weights,
                            float route_scale) {
  const torch::Tensor logits =
      linear_reference(input, weights.at("mlp.gate.weight"));
  const torch::Tensor probs = torch::softmax(logits, /*dim=*/-1);
  const auto sorted = probs.sort(/*dim=*/-1, /*descending=*/true);
  const torch::Tensor values = std::get<0>(sorted);
  const torch::Tensor ids = std::get<1>(sorted).select(/*dim=*/1, /*index=*/0);
  EXPECT_GT((values.select(/*dim=*/1, /*index=*/0) -
             values.select(/*dim=*/1, /*index=*/1))
                .min()
                .item<float>(),
            0.0f)
      << "Routing reference requires a unique selected expert";
  if (input.size(/*dim=*/0) > 1) {
    EXPECT_LT(ids.min().item<int64_t>(), ids.max().item<int64_t>());
  }
  torch::Tensor routed = torch::zeros_like(input);
  for (int64_t row = 0; row < input.size(/*dim=*/0); ++row) {
    const int64_t expert = ids[row].item<int64_t>();
    const torch::Tensor expert_out =
        swiglu_reference(input.slice(/*dim=*/0, row, row + 1),
                         weights,
                         "mlp.experts." + std::to_string(expert) + ".");
    routed[row] =
        expert_out.squeeze(/*dim=*/0) * probs[row][expert] * route_scale;
  }
  const torch::Tensor shared =
      swiglu_reference(input, weights, "mlp.shared_experts.");
  return round_bf16(round_bf16(routed) + shared);
}

ModelArgs decoder_args() {
  ModelArgs args = create_glm5_model_args();
  args.n_layers() = 4;
  args.index_topk_pattern() = "FSFS";
  args.index_topk() = 4;
  args.first_k_dense_replace() = 2;
  args.intermediate_size() = 256;
  args.moe_intermediate_size() = 256;
  args.n_routed_experts() = 4;
  args.num_experts_per_tok() = 1;
  args.n_group() = 1;
  args.topk_group() = 1;
  args.scoring_func() = "softmax";
  args.topk_method() = "greedy";
  args.norm_topk_prob() = false;
  args.routed_scaling_factor() = 1.25f;
  return args;
}

Weights decoder_weights(const ModelArgs& args, bool moe) {
  const auto cpu_options = torch::TensorOptions().dtype(torch::kBFloat16);
  Weights weights = create_seeded_mla_weights(args, cpu_options, "self_attn.");
  for (auto& [key, tensor] : weights) {
    if (tensor.dim() != 2) {
      continue;
    }
    const float midpoint =
        key.find("indexer.") == std::string::npos ? 0.005f : 0.025f;
    tensor = (tensor - midpoint) * 4.0f;
  }
  for (const std::string& norm :
       {"input_layernorm.weight", "post_attention_layernorm.weight"}) {
    weights[norm] = (test::seeded_tensor("C9/" + norm,
                                         {args.hidden_size()},
                                         torch::kBFloat16,
                                         torch::kCPU) *
                         0.2f +
                     0.9f);
  }
  Weights ffn = moe ? create_seeded_moe_weights(args, cpu_options)
                    : create_seeded_dense_weights(args, cpu_options);
  for (auto& [key, tensor] : ffn) {
    if (tensor.dim() == 2) {
      tensor = moe ? (tensor - 0.5f) * 0.12f : (tensor - 0.005f) * 12.0f;
    }
    weights.emplace("mlp." + key, std::move(tensor));
  }
  return weights;
}

AttentionMetadata metadata(const torch::TensorOptions& options, bool decode) {
  const int32_t rows = decode ? 1 : kPrefill;
  const int32_t length = decode ? kHistory + 1 : kPrefill;
  const auto ints = options.dtype(torch::kInt32);
  AttentionMetadata result;
  result.q_cu_seq_lens = torch::tensor({0, rows}, ints);
  result.kv_cu_seq_lens = torch::tensor({0, length}, ints);
  result.kv_seq_lens = torch::tensor({length}, ints);
  result.block_table = torch::arange(kBlocks, ints).view(/*size=*/{1, kBlocks});
  result.slot_mapping = torch::arange(decode ? kHistory : 0, length, ints);
  result.max_query_len = rows;
  result.max_seq_len = length;
  result.total_kv_len = length;
  result.compute_dtype = "bfloat16";
  result.is_prefill = !decode;
  result.is_chunked_prefill = false;
  result.is_dummy = false;
  return result;
}

DsaTopkState fixed_topk(const torch::Device& device, bool decode) {
  const int32_t rows = decode ? 1 : kPrefill;
  torch::Tensor table =
      torch::full(/*size=*/{rows, 4}, /*fill_value=*/-1, torch::kInt32);
  torch::Tensor lens = torch::empty(/*size=*/{rows}, torch::kInt32);
  for (int32_t row = 0; row < rows; ++row) {
    const int32_t count = decode ? 4 : std::min(row + 1, 4);
    lens[row] = count;
    for (int32_t col = 0; col < count; ++col) {
      table[row][col] = decode ? 2 * col : col;
    }
  }
  return DsaTopkState(table.to(device), lens.to(device));
}

void run_decoder_math(bool moe, bool decode) {
  torch::NoGradGuard no_grad;
  BlockSizeGuard block_size;
  Device dev(/*device_index=*/0);
  dev.set_device();
  const torch::Device device = dev.unwrap();
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  ProcessGroup group(/*rank=*/0, /*world_size=*/1, device);
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group);
  parallel.tp_group_ = &group;
  parallel.cp_group_ = &group;
  parallel.single_rank_group_ = &group;
  parallel.kv_split_size() = 1;
  const ModelArgs args = decoder_args();
  const Weights weights = decoder_weights(args, moe);
  const ModelContext context(parallel, args, QuantArgs(), options);
  const int32_t layer_id = moe ? 3 : 1;
  DeepseekV32DecoderLayer decoder(context, layer_id, DsaTopkSharePlan(args));
  decoder->load_state_dict(StateDict(weights));
  decoder->verify_loaded_weights();
  DeepseekV32Attention attention(context, /*enable_indexer=*/false);
  attention->load_state_dict(
      StateDict(weights).get_dict_with_prefix("self_attn."));
  attention->verify_loaded_weights();
  const int32_t rows = decode ? 1 : kPrefill;
  const torch::Tensor input = (test::seeded_tensor("C9/input",
                                                   {rows, args.hidden_size()},
                                                   torch::kBFloat16,
                                                   torch::kCPU) -
                               0.5f) *
                              0.6f;
  const torch::Tensor initial_key =
      (test::seeded_tensor(
           "C9/cache",
           {kBlocks, 1, 1, args.kv_lora_rank() + args.qk_rope_head_dim()},
           torch::kBFloat16,
           torch::kCPU) -
       0.5f) *
      0.02f;
  const torch::Tensor initial_index =
      torch::full({kBlocks, 1, 1, args.index_head_dim()},
                  /*fill_value=*/-2.0f,
                  torch::kBFloat16);
  auto actual_cache = create_indexed_kv_cache(initial_key.clone().to(device),
                                              initial_index.clone().to(device));
  auto reference_cache = create_indexed_kv_cache(
      initial_key.clone().to(device), initial_index.clone().to(device));
  const auto attn_metadata = metadata(options, decode);
  const auto ints = options.dtype(torch::kInt32);
  torch::Tensor positions = torch::arange(
      decode ? kHistory : 0, decode ? kHistory + 1 : kPrefill, ints);
  const DsaTopkState selection = fixed_topk(device, decode);
  auto capture = DsaTopkTransfer::capture_output();
  capture.publish_output(selection);
  DsaTopkRelay relay;
  relay.finish_layer(DsaTopkShareDecision{.output_topk = true}, capture);
  ModelInputParams params;
  params.meta.batch_forward_type =
      decode ? BatchForwardType::DECODE : BatchForwardType::PREFILL;
  torch::Tensor x = input.to(device);
  std::optional<torch::Tensor> residual;
  const torch::Tensor output = decoder->forward(x,
                                                residual,
                                                positions,
                                                attn_metadata,
                                                actual_cache,
                                                params,
                                                /*input_ids=*/std::nullopt,
                                                &relay);
  const torch::Tensor normalized = rms_reference(
      input, weights.at("input_layernorm.weight"), args.rms_norm_eps());
  auto reuse = DsaTopkTransfer::reuse(selection);
  const auto attn_out =
      attention->forward(positions,
                         normalized.to(device, torch::kBFloat16),
                         attn_metadata,
                         reference_cache,
                         /*cp_ctx=*/nullptr,
                         &reuse);
  dev.synchronize_default_stream();
  const torch::Tensor skip = round_bf16(
      attn_out.output.cpu().to(torch::kFloat32) + input.to(torch::kFloat32));
  const torch::Tensor ffn_in = rms_reference(
      skip, weights.at("post_attention_layernorm.weight"), args.rms_norm_eps());
  const torch::Tensor ffn_out =
      moe ? moe_reference(ffn_in, weights, args.routed_scaling_factor())
          : swiglu_reference(ffn_in, weights, "mlp.");
  const torch::Tensor expected = round_bf16(ffn_out + skip);
  EXPECT_GT(ffn_out.abs().max().item<float>(), 0.02f);
  EXPECT_GT(output.to(torch::kFloat32).std().item<float>(), 0.03f);
  EXPECT_TRUE(torch::isfinite(output).all().item<bool>());
  EXPECT_FALSE(residual.has_value());
  expect_close(output, expected, kRtol, kAtol, "CPU decoder composition");
  expect_close(actual_cache.get_k_cache(),
               reference_cache.get_k_cache(),
               kRtol,
               kAtol,
               "normalized attention cache");
  EXPECT_TRUE(
      torch::equal(actual_cache.get_index_cache().cpu(), initial_index));
  torch::Tensor untouched = torch::ones({kBlocks}, torch::kBool);
  untouched.index_fill_(/*dim=*/0,
                        attn_metadata.slot_mapping.cpu().to(torch::kInt64),
                        /*value=*/false);
  const torch::Tensor actual_key = actual_cache.get_k_cache().cpu();
  EXPECT_TRUE(torch::equal(actual_key.index({untouched}),
                           initial_key.index({untouched})));
  EXPECT_FALSE(torch::equal(actual_key, initial_key));
  auto state = relay.prepare_layer(DsaTopkShareDecision{.reuse_topk = true});
  ASSERT_TRUE(state.has_value());
  EXPECT_TRUE(
      torch::equal(state->input()->block_tables(), selection.block_tables()));
  EXPECT_TRUE(
      torch::equal(state->input()->context_lens(), selection.context_lens()));
}

TEST(DsaDecoderMathTest, DensePrefillMatchesCpuComposition) {
  run_decoder_math(/*moe=*/false, /*decode=*/false);
}
TEST(DsaDecoderMathTest, DenseDecodeMatchesCpuComposition) {
  run_decoder_math(/*moe=*/false, /*decode=*/true);
}
TEST(DsaDecoderMathTest, MoePrefillMatchesCpuRoutingAndComposition) {
  run_decoder_math(/*moe=*/true, /*decode=*/false);
}
TEST(DsaDecoderMathTest, MoeDecodeMatchesCpuRoutingAndComposition) {
  run_decoder_math(/*moe=*/true, /*decode=*/true);
}

}  // namespace
}  // namespace xllm::layer
