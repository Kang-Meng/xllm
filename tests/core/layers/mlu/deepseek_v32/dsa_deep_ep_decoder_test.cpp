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
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "core/framework/config/eplb_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/model_context.h"
#include "core/layers/common/deep_ep.h"
#include "core/layers/mlu/deepseek_v32/deepseek_v32_decoder_layer.h"
#include "core/layers/mlu/deepseek_v32/deepseek_v32_sparse_moe_block.h"
#include "framework/parallel_state/context_parallel_topology.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "tests/core/layers/mlu/tests_utils.h"
#include "util/net.h"

namespace xllm::layer {
namespace {
bool child_failed(int32_t initial_parts) {
  const auto* result =
      ::testing::UnitTest::GetInstance()->current_test_info()->result();
  for (int32_t idx = initial_parts; idx < result->total_part_count(); ++idx) {
    if (result->GetTestPartResult(idx).failed()) {
      return true;
    }
  }
  return false;
}

void stop_ranks(const std::vector<pid_t>& pids,
                const std::vector<bool>& finished) {
  for (size_t idx = 0; idx < pids.size(); ++idx) {
    if (!finished[idx]) {
      ::kill(pids[idx], SIGKILL);
    }
  }
  for (size_t idx = 0; idx < pids.size(); ++idx) {
    if (finished[idx]) {
      continue;
    }
    int status = 0;
    while (::waitpid(pids[idx], &status, 0) < 0 && errno == EINTR) {
    }
  }
}

bool wait_ranks(const std::vector<pid_t>& pids,
                std::chrono::milliseconds timeout,
                bool& skipped) {
  std::vector<bool> finished(pids.size(), false);
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  size_t remaining = pids.size();
  int32_t skip_count = 0;
  while (remaining > 0 && std::chrono::steady_clock::now() < deadline) {
    for (size_t idx = 0; idx < pids.size(); ++idx) {
      if (finished[idx]) {
        continue;
      }
      int status = 0;
      const pid_t pid = ::waitpid(pids[idx], &status, WNOHANG);
      if (pid == 0 || (pid < 0 && errno == EINTR)) {
        continue;
      }
      finished[idx] = true;
      --remaining;
      if (pid < 0 || !WIFEXITED(status) ||
          (WEXITSTATUS(status) != 0 && WEXITSTATUS(status) != 77)) {
        LOG(ERROR) << "Rank " << idx << " wait status=" << status
                   << " waitpid=" << pid << " errno=" << errno;
        stop_ranks(pids, finished);
        return false;
      }
      skip_count += WEXITSTATUS(status) == 77 ? 1 : 0;
    }
    if (remaining > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds{10});
    }
  }
  if (remaining > 0) {
    stop_ranks(pids, finished);
    return false;
  }
  skipped = skip_count == static_cast<int32_t>(pids.size());
  return skip_count == 0 || skipped;
}

std::unique_ptr<ProcessGroup> create_test_process_group(
    int32_t rank,
    int32_t world_size,
    int32_t port,
    const std::string& host,
    const torch::Device& device) {
  auto group = create_process_group(rank,
                                    world_size,
                                    world_size,
                                    port,
                                    /*trans=*/false,
                                    host,
                                    "multi_rank_test",
                                    device);
  CHECK(group);
  torch::Tensor ready = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  group->allreduce(ready);
  CHECK_EQ(ready.item<float>(), static_cast<float>(world_size));
  return group;
}
std::unique_ptr<ProcessGroup> create_test_self_group(
    int32_t rank,
    int32_t world_size,
    const std::string& host,
    const torch::Device& device) {
  return create_process_group(rank,
                              world_size,
                              /*rank_size=*/1,
                              /*port=*/0,
                              /*trans=*/false,
                              host,
                              "multi_rank_self_" + std::to_string(rank),
                              device);
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
Weights create_moe_bf16_weights(const ModelArgs& model_args,
                                const torch::TensorOptions& options,
                                int64_t n_shared_experts);
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
Weights create_seeded_moe_weights(const ModelArgs& args,
                                  DsaWeightFormat format,
                                  const torch::TensorOptions& options);
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

Weights create_moe_bf16_weights(const ModelArgs& model_args,
                                const torch::TensorOptions& options,
                                int64_t n_shared_experts) {
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

  return weight_dict;
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

Weights create_seeded_moe_weights(const ModelArgs& args,
                                  DsaWeightFormat format,
                                  const torch::TensorOptions& options) {
  if (format == DsaWeightFormat::BF16) {
    Weights weights =
        create_moe_bf16_weights(args, options, args.n_shared_experts());
    weights["gate.e_score_correction_bias"] =
        torch::zeros({args.n_routed_experts()}, options.dtype(torch::kFloat32));
    return weights;
  }
  Weights weights;
  for (int32_t expert = 0; expert < args.n_routed_experts(); ++expert) {
    ModelArgs dense_args = args;
    dense_args.intermediate_size() = args.moe_intermediate_size();
    weights.merge(
        create_seeded_dense_weights(dense_args,
                                    format,
                                    options,
                                    "experts." + std::to_string(expert) + "."));
  }
  if (args.n_shared_experts() > 0) {
    ModelArgs dense_args = args;
    dense_args.intermediate_size() =
        args.moe_intermediate_size() * args.n_shared_experts();
    weights.merge(create_seeded_dense_weights(
        dense_args, format, options, "shared_experts."));
  }
  weights["gate.weight"] = ::xllm::layer::test::seeded_tensor(
      "deepseek_v2_sparse_moe_block.gate",
      {args.n_routed_experts(), args.hidden_size()},
      torch::kBFloat16,
      options.device());
  weights["gate.e_score_correction_bias"] =
      torch::zeros({args.n_routed_experts()}, options.dtype(torch::kFloat32));
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

constexpr int32_t kBlocks = 16;
constexpr int32_t kHistory = 6;
constexpr int32_t kPrefill = 8;
// BF16 routed GEMM may accumulate in a different order across the EP and
// unsharded implementations.
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

ModelArgs decoder_args() {
  ModelArgs args = create_dsa_model_args(DsaModelVariant::GLM5);
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
  Weights weights = create_seeded_mla_weights(
      args, DsaWeightFormat::BF16, cpu_options, "self_attn.");
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
    weights[norm] = (test::seeded_tensor("C8/" + norm,
                                         {args.hidden_size()},
                                         torch::kBFloat16,
                                         torch::kCPU) *
                         0.2f +
                     0.9f);
  }
  Weights ffn =
      moe ? create_seeded_moe_weights(args, DsaWeightFormat::BF16, cpu_options)
          : create_seeded_dense_weights(
                args, DsaWeightFormat::BF16, cpu_options);
  for (auto& [key, tensor] : ffn) {
    if (tensor.dim() == 2) {
      tensor = moe ? (tensor - 0.5f) * 0.12f : (tensor - 0.005f) * 12.0f;
    }
    weights.emplace("mlp." + key, std::move(tensor));
  }
  weights["mlp.gate.weight"] = torch::zeros_like(weights.at("mlp.gate.weight"));
  weights.at("mlp.gate.weight")[0][0] = -4.0f;
  weights.at("mlp.gate.weight")[2][0] = 4.0f;
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

struct DecoderResult {
  torch::Tensor output;
  torch::Tensor key;
  torch::Tensor index;
};

DecoderResult forward_decoder(const ModelContext& context,
                              const Weights& weights,
                              const torch::Tensor& input,
                              const torch::Tensor& key,
                              const torch::Tensor& index,
                              const ModelInputParams& params) {
  const auto& options = context.get_tensor_options();
  const torch::Device device = options.device();
  const ModelArgs& args = context.get_model_args();
  DeepseekV32DecoderLayer decoder(
      context, /*layer_id=*/3, DsaTopkSharePlan(args));
  decoder->load_state_dict(StateDict(weights));
  decoder->verify_loaded_weights();
  auto cache =
      create_indexed_kv_cache(key.clone().to(device), index.clone().to(device));
  const DsaTopkState selection = fixed_topk(device, /*decode=*/true);
  auto transfer = DsaTopkTransfer::capture_output();
  transfer.publish_output(selection);
  DsaTopkRelay relay;
  relay.finish_layer(DsaTopkShareDecision{.output_topk = true}, transfer);
  torch::Tensor x = input.to(device);
  std::optional<torch::Tensor> residual;
  torch::Tensor positions =
      torch::tensor({kHistory}, options.dtype(torch::kInt32));
  torch::Tensor output = decoder->forward(x,
                                          residual,
                                          positions,
                                          metadata(options, /*decode=*/true),
                                          cache,
                                          params,
                                          /*input_ids=*/std::nullopt,
                                          &relay);
  EXPECT_FALSE(residual.has_value());
  return {
      output.cpu(), cache.get_k_cache().cpu(), cache.get_index_cache().cpu()};
}

int32_t run_remote_decoder(int32_t rank, int32_t port) {
  torch::NoGradGuard no_grad;
  BlockSizeGuard block_size;
  EPLBConfig::get_instance().expert_parallel_degree(/*value=*/2);
  SchedulerConfig::get_instance().max_seqs_per_batch(/*value=*/4);
  Device device_guard(rank);
  device_guard.set_device();
  const torch::Device device = device_guard.unwrap();
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  auto world = create_test_process_group(
      rank, /*world_size=*/2, port, "127.0.0.1", device);
  auto self =
      create_test_self_group(rank, /*world_size=*/2, "127.0.0.1", device);
  ParallelArgs parallel(rank, /*world_size=*/2, /*dp_size=*/2, world.get());
  parallel.ep_size() = 2;
  parallel.tp_group_ = self.get();
  parallel.cp_group_ = self.get();
  parallel.single_rank_group_ = self.get();
  parallel.moe_tp_group_ = self.get();
  parallel.moe_ep_group_ = world.get();
  parallel.dp_local_process_group_ = world.get();
  parallel.kv_split_size() = 1;
  ParallelArgs single(/*rank=*/0, /*world_size=*/1, self.get());
  single.tp_group_ = self.get();
  single.cp_group_ = self.get();
  single.single_rank_group_ = self.get();
  single.moe_tp_group_ = self.get();
  single.kv_split_size() = 1;
  const ModelArgs args = decoder_args();
  const ModelContext distributed_context(parallel, args, QuantArgs(), options);
  const ModelContext reference_context(single, args, QuantArgs(), options);
  // Own the public communication instance with stable configuration lifetimes.
  // All decoder instances reuse this exact production manager configuration.
  DeepEP deep_ep = DeepEPManager::get_instance(args.hidden_size() * 2,
                                               args.hidden_size() * 2,
                                               /*max_num_tokens_per_rank=*/4,
                                               args.n_routed_experts(),
                                               parallel,
                                               options);
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.parallel.dp_global_token_nums = {1, 1};
  params.parallel.dp_is_decode = {1, 1};
  {
    const auto& streams = distributed_context.stream_registry();
    DeepseekV32SparseMoEBlock block(
        args,
        QuantArgs(),
        parallel,
        options,
        streams->get(ExecutionStreamRole::COMMUNICATION),
        streams->get(ExecutionStreamRole::AUXILIARY_COMPUTE));
    const auto exec = block->plan_exec(params);
    EXPECT_TRUE(exec.enable_all2all);
    EXPECT_FALSE(exec.need_dp_gather);
    DeepseekV32SparseMoEBlock reference_block(
        args,
        QuantArgs(),
        single,
        options,
        streams->get(ExecutionStreamRole::COMMUNICATION),
        streams->get(ExecutionStreamRole::AUXILIARY_COMPUTE));
    EXPECT_FALSE(reference_block->plan_exec(params).enable_all2all);
  }
  ModelInputParams reference_params;
  reference_params.meta.batch_forward_type = BatchForwardType::DECODE;
  reference_params.parallel.dp_global_token_nums = {1};
  reference_params.parallel.dp_is_decode = {1};
  const Weights weights = decoder_weights(args, /*moe=*/true);
  torch::Tensor input =
      (test::seeded_tensor(
           "C8/input", {2, args.hidden_size()}, torch::kBFloat16, torch::kCPU) -
       0.5f) *
      0.6f;
  input[0][0] = 3.0f;
  input[1][0] = -3.0f;
  input = input.narrow(0, rank, 1).clone();
  const torch::Tensor key =
      (test::seeded_tensor(
           "C8/key",
           {2, kBlocks, 1, 1, args.kv_lora_rank() + args.qk_rope_head_dim()},
           torch::kBFloat16,
           torch::kCPU)
           .select(0, rank)
           .clone() -
       0.5f) *
      0.02f;
  const torch::Tensor index =
      torch::full({kBlocks, 1, 1, args.index_head_dim()},
                  /*fill_value=*/-2.0f,
                  torch::kBFloat16);
  const DecoderResult expected = forward_decoder(
      reference_context, weights, input, key, index, reference_params);
  const DecoderResult actual =
      forward_decoder(distributed_context, weights, input, key, index, params);
  // The decoder reuses the stable public manager instance initialized above.
  // Received token counters establish nonempty remote expert dispatch, while
  // remote-only weight sensitivity below requires the return/combine path.
  const torch::Tensor counts =
      deep_ep->get_params().dispatch_recv_token_num.cpu();
  EXPECT_EQ(counts.sum().item<int32_t>(), 1);
  expect_close(actual.output,
               expected.output,
               kRtol,
               kAtol,
               "EP2 decoder versus independent EP1 decoder");
  expect_close(
      actual.key, expected.key, kRtol, kAtol, "EP2 decoder attention cache");
  EXPECT_TRUE(torch::equal(actual.index, index));
  EXPECT_TRUE(torch::equal(actual.key, expected.key));
  torch::Tensor untouched = torch::ones({kBlocks}, torch::kBool);
  untouched[kHistory] = false;
  EXPECT_TRUE(
      torch::equal(actual.key.index({untouched}), key.index({untouched})));
  EXPECT_EQ(actual.output.sizes(), input.sizes());
  EXPECT_EQ(actual.key.sizes(), key.sizes());
  EXPECT_FALSE(torch::equal(actual.key, key));
  EXPECT_TRUE(torch::isfinite(actual.output).all().item<bool>());
  Weights changed = weights;
  changed.at("mlp.experts.2.down_proj.weight") =
      weights.at("mlp.experts.2.down_proj.weight") * 1.5f;
  const DecoderResult changed_expected = forward_decoder(
      reference_context, changed, input, key, index, reference_params);
  const DecoderResult changed_actual =
      forward_decoder(distributed_context, changed, input, key, index, params);
  expect_close(changed_actual.output,
               changed_expected.output,
               kRtol,
               kAtol,
               "remote expert perturbation versus EP1");
  const float change = (changed_actual.output.to(torch::kFloat32) -
                        actual.output.to(torch::kFloat32))
                           .abs()
                           .max()
                           .item<float>();
  if (rank == 0) {
    EXPECT_GT(change, 0.003f) << "rank0 must observe rank1-owned expert2";
  } else {
    EXPECT_EQ(change, 0.0f) << "rank1 routes to unchanged rank0-owned expert0";
  }

  device_guard.synchronize_default_stream();
  deep_ep = DeepEP(nullptr);
  DeepEPManager::destroy();
  return 0;
}

TEST(DsaDeepEpDecoderTest, RemoteExpertsMatchUnshardedDecode) {
  const int32_t port = net::get_local_free_port();

  const int32_t child_count = 2;
  const int32_t initial_parts = ::testing::UnitTest::GetInstance()
                                    ->current_test_info()
                                    ->result()
                                    ->total_part_count();
  std::vector<pid_t> child_pids;
  child_pids.reserve(child_count);
  bool children_ok = true;
  bool children_skipped = false;
  for (int32_t child_rank = 0; child_rank < child_count; ++child_rank) {
    const pid_t pid = ::fork();
    if (pid == 0) {
      try {
        if (Platform::device_count() < child_count) {
          ::_exit(77);
        }
        const auto rank_body = [port](int32_t rank) {
          return run_remote_decoder(rank, port);
        };
        const int32_t code = rank_body(child_rank);
        std::fflush(nullptr);
        ::_exit(child_failed(initial_parts) ? 1 : code);
      } catch (const std::exception& error) {
        LOG(ERROR) << "Rank " << child_rank << ": " << error.what();
        ::_exit(1);
      }
    }
    if (pid < 0) {
      children_ok = false;
      break;
    }
    child_pids.emplace_back(pid);
  }
  if (!children_ok) {
    stop_ranks(child_pids, std::vector<bool>(child_pids.size(), false));
  } else {
    children_ok = wait_ranks(child_pids,
                             std::chrono::milliseconds(/*count=*/180000),
                             children_skipped);
  }

  ASSERT_EQ(children_ok, true)
      << "forked ranks failed, timed out, or require more physical MLU cards";
}
}  // namespace
}  // namespace xllm::layer
