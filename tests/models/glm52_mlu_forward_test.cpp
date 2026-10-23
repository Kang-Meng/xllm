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

#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/config/kv_cache_config.h"
#include "core/framework/model_loader.h"
#include "framework/parallel_state/context_parallel_topology.h"
#include "models/llm/mlu/glm52.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "tests/core/layers/mlu/tests_utils.h"

namespace xllm {
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

struct CpTpTestGroups {
  std::unique_ptr<ProcessGroup> world;
  std::unique_ptr<ProcessGroup> self;
  std::unique_ptr<ProcessGroup> tp;
  std::unique_ptr<ProcessGroup> cp;
  std::unique_ptr<ParallelArgs> parallel_args;
};
int32_t parallel_base_port(int32_t world_size, int32_t cp_size) {
  CHECK_GT(cp_size, 0);
  CHECK_EQ(world_size % cp_size, 0);
  const int32_t tp_size = world_size / cp_size;
  int32_t ephemeral_low = 0;
  int32_t ephemeral_high = 0;
  std::ifstream range("/proc/sys/net/ipv4/ip_local_port_range");
  CHECK(range >> ephemeral_low >> ephemeral_high)
      << "Cannot read Linux ephemeral port range";
  CHECK_GT(ephemeral_high, ephemeral_low);
  // Administratively reserved ports can have routing/proxy ownership even
  // when a local bind probe succeeds.
  std::ifstream reserved_config("/proc/sys/net/ipv4/ip_local_reserved_ports");
  CHECK(reserved_config.is_open()) << "Cannot read Linux reserved ports";
  std::string reserved_text;
  std::getline(reserved_config, reserved_text);
  CHECK(!reserved_config.fail() || reserved_config.eof())
      << "Cannot read Linux reserved ports";
  std::array<bool, 65536> reserved{};
  std::istringstream entries(reserved_text);
  std::string entry;
  while (std::getline(entries, entry, ',')) {
    std::istringstream interval(entry);
    int32_t first_port = 0;
    CHECK(interval >> first_port) << "Malformed reserved port: " << entry;
    int32_t last_port = first_port;
    if (interval.peek() == '-') {
      interval.get();
      CHECK(interval >> last_port) << "Malformed reserved interval: " << entry;
    }
    interval >> std::ws;
    CHECK(interval.eof()) << "Malformed reserved port: " << entry;
    CHECK_GE(first_port, 0);
    CHECK_GE(last_port, first_port);
    CHECK_LT(last_port, reserved.size());
    for (int32_t port = first_port; port <= last_port; ++port) {
      reserved[port] = true;
    }
  }
  std::vector<int32_t> offsets;
  offsets.reserve(1 + cp_size + tp_size);
  offsets.emplace_back(0);
  for (int32_t group = 0; group < cp_size; ++group) {
    offsets.emplace_back(group + 1);
  }
  for (int32_t group = 0; group < tp_size; ++group) {
    offsets.emplace_back(group + 11);
  }
  constexpr int32_t kFirstPort = 1024;
  const int32_t last_base = ephemeral_low - offsets.back() - 1;
  CHECK_GE(last_base, kFirstPort)
      << "No nonprivileged port block below ephemeral range";
  const int32_t candidates = last_base - kFirstPort + 1;
  static int32_t sequence = 0;
  const int32_t first =
      (static_cast<int32_t>(getpid()) + sequence++) % candidates;
  for (int32_t attempt = 0; attempt < candidates; ++attempt) {
    const int32_t base = kFirstPort + (first + attempt) % candidates;
    std::vector<int32_t> sockets;
    sockets.reserve(offsets.size());
    bool available = true;
    for (int32_t offset : offsets) {
      if (reserved[base + offset]) {
        available = false;
        break;
      }
      const int32_t fd =
          socket(/*domain=*/AF_INET, /*type=*/SOCK_STREAM, /*protocol=*/0);
      PCHECK(fd >= 0);
      sockets.emplace_back(fd);
      sockaddr_in address{};
      address.sin_family = AF_INET;
      address.sin_addr.s_addr = htonl(INADDR_ANY);
      address.sin_port = htons(static_cast<uint16_t>(base + offset));
      const int32_t status =
          bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address));
      if (status != 0) {
        PCHECK(errno == EADDRINUSE)
            << "Cannot probe test TCPStore port " << base + offset;
        available = false;
        break;
      }
    }
    for (int32_t fd : sockets) {
      PCHECK(close(fd) == 0);
    }
    if (available) {
      LOG(INFO) << "test base_port=" << base << " world=" << world_size
                << " cp=" << cp_size << " ephemeral=" << ephemeral_low << "-"
                << ephemeral_high << " reserved=" << reserved_text;
      return base;
    }
  }
  LOG(FATAL) << "No free TCPStore port block below ephemeral range";
  return -1;
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
CpTpTestGroups create_cp_tp_groups(int32_t rank,
                                   int32_t world_size,
                                   int32_t cp_size,
                                   int32_t port,
                                   const std::string& host,
                                   const torch::Device& device) {
  CHECK_GT(cp_size, 0);
  CHECK_EQ(world_size % cp_size, 0);
  const int32_t tp_size = world_size / cp_size;
  const parallel_state::ContextParallelTopology topology(
      rank, world_size, /*dp_size=*/1, cp_size, /*dcp_size=*/1);
  CpTpTestGroups groups;
  groups.world =
      create_test_process_group(rank, world_size, port, host, device);
  groups.tp = create_process_group(rank,
                                   world_size,
                                   tp_size,
                                   port + rank / tp_size + 1,
                                   /*trans=*/false,
                                   host,
                                   "cp_tp_test_tp",
                                   device);
  groups.cp = create_process_group(rank,
                                   topology.pcp_rank(),
                                   topology.pcp_group_ranks(),
                                   world_size,
                                   cp_size,
                                   port + 10 + topology.tp_rank() + 1,
                                   host,
                                   "cp_tp_test_cp",
                                   device);
  // Fixed subgroup ports must bind before single-rank stores choose port 0.
  groups.self = create_test_self_group(rank, world_size, host, device);
  CHECK(groups.tp);
  CHECK(groups.cp);
  groups.parallel_args =
      std::make_unique<ParallelArgs>(rank, world_size, groups.world.get());
  groups.parallel_args->tp_size() = tp_size;
  groups.parallel_args->kv_split_size() = 1;
  groups.parallel_args->tp_group_ = groups.tp.get();
  groups.parallel_args->cp_group_ = groups.cp.get();
  groups.parallel_args->dcp_group_ = groups.self.get();
  groups.parallel_args->single_rank_group_ = groups.self.get();
  groups.parallel_args->cp_size() = cp_size;
  return groups;
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
KVCache create_mla_cache(const ModelArgs& model_args,
                         const torch::TensorOptions& options,
                         int64_t block_num,
                         int64_t block_size);
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
        ::xllm::layer::test::seeded_tensor(seed_prefix + ".gate_proj",
                                           {intermediate_size, hidden_size},
                                           torch::kBFloat16,
                                           options.device());
    weight_dict[expert_prefix + "up_proj.weight"] =
        ::xllm::layer::test::seeded_tensor(seed_prefix + ".up_proj",
                                           {intermediate_size, hidden_size},
                                           torch::kBFloat16,
                                           options.device());
    weight_dict[expert_prefix + "down_proj.weight"] =
        ::xllm::layer::test::seeded_tensor(seed_prefix + ".down_proj",
                                           {hidden_size, intermediate_size},
                                           torch::kBFloat16,
                                           options.device());
  }

  weight_dict["gate.weight"] =
      ::xllm::layer::test::seeded_tensor("deepseek_v2_sparse_moe_block.gate",
                                         {num_experts, hidden_size},
                                         torch::kBFloat16,
                                         options.device());

  if (n_shared_experts > 0) {
    const int64_t shared_size = intermediate_size * n_shared_experts;
    weight_dict["shared_experts.gate_proj.weight"] =
        ::xllm::layer::test::seeded_tensor(
            "deepseek_v2_sparse_moe_block.shared.gate_proj",
            {shared_size, hidden_size},
            torch::kBFloat16,
            options.device());
    weight_dict["shared_experts.up_proj.weight"] =
        ::xllm::layer::test::seeded_tensor(
            "deepseek_v2_sparse_moe_block.shared.up_proj",
            {shared_size, hidden_size},
            torch::kBFloat16,
            options.device());
    weight_dict["shared_experts.down_proj.weight"] =
        ::xllm::layer::test::seeded_tensor(
            "deepseek_v2_sparse_moe_block.shared.down_proj",
            {hidden_size, shared_size},
            torch::kBFloat16,
            options.device());
  }

  return weight_dict;
}

KVCache create_mla_cache(const ModelArgs& model_args,
                         const torch::TensorOptions& options,
                         int64_t block_num,
                         int64_t block_size) {
  auto k_cache =
      torch::zeros({block_num,
                    1,
                    block_size,
                    model_args.qk_rope_head_dim() + model_args.kv_lora_rank()},
                   options);
  auto index_cache = torch::zeros(
      {block_num, 1, block_size, model_args.index_head_dim()}, options);
  return create_indexed_kv_cache(k_cache, index_cache);
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

namespace fixture = layer::test;
constexpr int32_t kLayers = 4;
constexpr int32_t kPrefill = 12;
constexpr int32_t kBlocks = 32;
constexpr float kIndexSentinel = -2.0f;

class MemoryModelLoader final : public ModelLoader {
 public:
  MemoryModelLoader(ModelArgs args, Weights weights) {
    args_ = std::move(args);
    states_.reserve(/*new_cap=*/1);
    states_.emplace_back(std::make_unique<StateDict>(std::move(weights)));
  }
  std::unique_ptr<Tokenizer> tokenizer() const override { return nullptr; }
  std::vector<std::unique_ptr<StateDict>>& get_state_dicts() override {
    return states_;
  }
  std::string model_weights_path() const override { return "memory://glm52"; }

 private:
  std::vector<std::unique_ptr<StateDict>> states_;
};

ModelArgs model_args() {
  ModelArgs args = create_dsa_model_args(DsaModelVariant::GLM52_SHARE);
  args.n_layers() = kLayers;
  args.index_topk_pattern() = "FSFS";
  args.index_topk() = 4;
  args.vocab_size() = 64;
  args.n_kv_heads() = args.n_heads();
  return args;
}

Weights model_weights(const ModelArgs& args) {
  const auto options = torch::TensorOptions().dtype(torch::kBFloat16);
  Weights weights;
  weights["model.embed_tokens.weight"] =
      fixture::seeded_tensor("c10.embed",
                             {args.vocab_size(), args.hidden_size()},
                             torch::kBFloat16,
                             torch::kCPU) *
      0.2f;
  weights["lm_head.weight"] =
      fixture::seeded_tensor("c10.head",
                             {args.vocab_size(), args.hidden_size()},
                             torch::kBFloat16,
                             torch::kCPU) *
      0.02f;
  weights["model.norm.weight"] = torch::ones({args.hidden_size()}, options);
  for (int32_t layer = 0; layer < kLayers; ++layer) {
    const std::string prefix = "model.layers." + std::to_string(layer) + ".";
    weights[prefix + "input_layernorm.weight"] =
        torch::ones({args.hidden_size()}, options);
    weights[prefix + "post_attention_layernorm.weight"] =
        torch::ones({args.hidden_size()}, options);
    weights.merge(create_seeded_mla_weights(
        args, DsaWeightFormat::BF16, options, prefix + "self_attn."));
    Weights ffn =
        layer == 0
            ? create_seeded_dense_weights(
                  args, DsaWeightFormat::BF16, options, /*prefix=*/"")
            : create_seeded_moe_weights(args, DsaWeightFormat::BF16, options);
    for (auto& [key, tensor] : ffn) {
      weights.emplace(
          prefix + "mlp." + key,
          tensor * (layer == 0 ? 1.0f : 0.01f) * (1.0f + 0.05f * layer));
    }
  }
  return weights;
}

Glm52ForCausalLM create_model(const ModelArgs& args,
                              const ParallelArgs& parallel,
                              const torch::TensorOptions& options,
                              Weights weights) {
  Glm52ForCausalLM model(ModelContext(parallel, args, QuantArgs(), options));
  model->load_model(
      std::make_unique<MemoryModelLoader>(args, std::move(weights)));
  return model;
}

std::vector<KVCache> create_caches(const ModelArgs& args,
                                   const torch::TensorOptions& options,
                                   int32_t splits = 1) {
  std::vector<KVCache> caches;
  caches.reserve(kLayers);
  for (int32_t layer = 0; layer < kLayers; ++layer) {
    auto cache =
        create_mla_cache(args, options, kBlocks / splits, /*block_size=*/1);
    cache.get_index_cache().fill_(kIndexSentinel);
    caches.emplace_back(std::move(cache));
  }
  return caches;
}

ModelInputParams input_params(const torch::Device& device, bool decode) {
  const int32_t rows = decode ? 1 : kPrefill;
  const int32_t length = decode ? kPrefill + 1 : kPrefill;
  const auto ints = torch::TensorOptions().dtype(torch::kInt32).device(device);
  ModelInputParams params;
  params.meta.batch_forward_type =
      decode ? BatchForwardType::DECODE : BatchForwardType::PREFILL;
  params.meta.num_sequences = 1;
  params.meta.q_max_seq_len = rows;
  params.meta.kv_max_seq_len = length;
  params.attention.host.q_seq_lens = {0, rows};
  params.attention.host.kv_seq_lens = {0, length};
  params.attention.device.q_seq_lens = torch::tensor({0, rows}, ints);
  params.attention.device.kv_seq_lens = torch::tensor({0, length}, ints);
  params.attention.device.new_cache_slots =
      torch::arange(decode ? kPrefill : 0, length, ints);
  params.attention.device.block_tables =
      torch::arange(kBlocks, ints).view(/*size=*/{1, -1});
  return params;
}

ModelOutput forward(Glm52ForCausalLM& model,
                    std::vector<KVCache>& caches,
                    const torch::Device& device,
                    bool decode) {
  const auto ints = torch::TensorOptions().dtype(torch::kInt32).device(device);
  const int32_t begin = decode ? kPrefill : 0;
  const int32_t end = decode ? kPrefill + 1 : kPrefill;
  const torch::Tensor positions = torch::arange(begin, end, ints);
  const torch::Tensor tokens = (positions * 3 + 7).remainder(/*other=*/64);
  return model->forward(
      tokens, positions, caches, input_params(device, decode));
}

void check_output(const ModelOutput& output, int32_t rows) {
  ASSERT_TRUE(output.hidden_states.defined());
  ASSERT_EQ(output.hidden_states.sizes(), torch::IntArrayRef({rows, 256}));
  EXPECT_TRUE(torch::isfinite(output.hidden_states).all().item<bool>());
  EXPECT_GT(output.hidden_states.abs().max().item<float>(), 0.1f);
}

void check_caches(const std::vector<KVCache>& caches, int32_t length) {
  for (int32_t layer = 0; layer < kLayers; ++layer) {
    const torch::Tensor index =
        caches[layer].get_index_cache().cpu().view(/*size=*/{-1, 128});
    const torch::Tensor key = caches[layer].get_k_cache().cpu().flatten(
        /*start_dim=*/0, /*end_dim=*/2);
    EXPECT_GT(
        key.slice(/*dim=*/0, /*start=*/0, length).abs().max().item<float>(),
        0.0f);
    EXPECT_TRUE(torch::equal(key.slice(/*dim=*/0, length),
                             torch::zeros_like(key.slice(/*dim=*/0, length))));
    if (layer % 2 == 0) {
      EXPECT_TRUE(
          (index.slice(/*dim=*/0, /*start=*/0, length) != kIndexSentinel)
              .any(/*dim=*/-1)
              .all()
              .item<bool>());
    } else {
      EXPECT_TRUE((index == kIndexSentinel).all().item<bool>());
    }
    EXPECT_TRUE(
        (index.slice(/*dim=*/0, length) == kIndexSentinel).all().item<bool>());
  }
}

ParallelArgs single_parallel(ProcessGroup* group) {
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, group);
  parallel.tp_group_ = group;
  parallel.cp_group_ = group;
  parallel.dcp_group_ = group;
  parallel.single_rank_group_ = group;
  parallel.kv_split_size() = 1;
  return parallel;
}

void run_single(bool mutate_shared, bool mutate_full) {
  torch::NoGradGuard no_grad;
  Device dev(/*device_index=*/0);
  dev.set_device();
  const torch::Device device = dev.unwrap();
  KVCacheConfig::get_instance().block_size(/*value=*/1);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  ProcessGroup group(/*rank=*/0, /*world_size=*/1, device);
  const ParallelArgs parallel = single_parallel(&group);
  const ModelArgs args = model_args();
  const Weights weights = model_weights(args);
  auto model = create_model(args, parallel, options, weights);
  auto caches = create_caches(args, options);
  auto temporal_model = create_model(args, parallel, options, weights);
  auto temporal_cache = create_caches(args, options);
  std::vector<torch::Tensor> temporal_rows;
  temporal_rows.reserve(kPrefill + 1);
  for (int32_t row = 0; row <= kPrefill; ++row) {
    ModelInputParams params = input_params(device, true);
    params.attention.host.kv_seq_lens = {0, row + 1};
    params.attention.device.kv_seq_lens =
        torch::tensor({0, row + 1}, torch::kInt32).to(device);
    params.meta.kv_max_seq_len = row + 1;
    params.attention.device.new_cache_slots =
        torch::tensor({row}, torch::kInt32).to(device);
    const torch::Tensor positions =
        torch::tensor({row}, torch::kInt32).to(device);
    const torch::Tensor tokens = (positions * 3 + 7).remainder(64);
    temporal_rows.emplace_back(
        temporal_model->forward(tokens, positions, temporal_cache, params)
            .hidden_states);
  }
  auto prefill = forward(model, caches, device, /*decode=*/false);
  check_output(prefill, kPrefill);
  check_caches(caches, kPrefill);

  auto decode = forward(model, caches, device, /*decode=*/true);
  check_output(decode, /*rows=*/1);
  check_caches(caches, kPrefill + 1);
  expect_close(
      prefill.hidden_states,
      torch::cat(std::vector<torch::Tensor>(temporal_rows.begin(),
                                            temporal_rows.begin() + kPrefill),
                 0),
      /*rtol=*/0.02,
      /*atol=*/0.02,
      "prefill versus tokenwise model");
  expect_close(decode.hidden_states,
               temporal_rows.back(),
               /*rtol=*/0.02,
               /*atol=*/0.02,
               "decode versus tokenwise model");
  for (int32_t layer = 0; layer < kLayers; ++layer) {
    expect_close(caches[layer].get_k_cache(),
                 temporal_cache[layer].get_k_cache(),
                 /*rtol=*/0.02,
                 /*atol=*/0.02,
                 "tokenwise model key cache");
    expect_close(caches[layer].get_index_cache(),
                 temporal_cache[layer].get_index_cache(),
                 /*rtol=*/0.02,
                 /*atol=*/0.02,
                 "tokenwise model index cache");
  }

  if (!mutate_shared && !mutate_full) {
    return;
  }
  Weights changed = model_weights(args);
  for (auto& [key, tensor] : changed) {
    const bool shared = key.starts_with("model.layers.1.") ||
                        key.starts_with("model.layers.3.");
    if (key.find(".indexer.") != std::string::npos &&
        ((mutate_shared && shared) ||
         (mutate_full &&
          key == "model.layers.2.self_attn.indexer.wq_b.weight"))) {
      tensor = -tensor * 3.0f;
    }
  }
  auto other = create_model(args, parallel, options, std::move(changed));
  auto other_caches = create_caches(args, options);
  const auto other_prefill =
      forward(other, other_caches, device, /*decode=*/false);
  const auto other_decode =
      forward(other, other_caches, device, /*decode=*/true);
  dev.synchronize_default_stream();
  if (mutate_shared) {
    EXPECT_TRUE(
        torch::equal(prefill.hidden_states, other_prefill.hidden_states));
    EXPECT_TRUE(torch::equal(decode.hidden_states, other_decode.hidden_states));
    for (int32_t layer = 0; layer < kLayers; ++layer) {
      EXPECT_TRUE(torch::equal(caches[layer].get_k_cache(),
                               other_caches[layer].get_k_cache()));
      EXPECT_TRUE(torch::equal(caches[layer].get_index_cache(),
                               other_caches[layer].get_index_cache()));
    }
  } else {
    EXPECT_TRUE(
        torch::equal(caches[2].get_k_cache(), other_caches[2].get_k_cache()));
    EXPECT_TRUE(torch::equal(caches[2].get_index_cache(),
                             other_caches[2].get_index_cache()));
    LOG(INFO) << "Later Full query-only perturbation max output delta: "
              << (decode.hidden_states.to(torch::kFloat32) -
                  other_decode.hidden_states.to(torch::kFloat32))
                     .abs()
                     .max()
                     .item<float>();
    EXPECT_FALSE(torch::equal(decode.hidden_states, other_decode.hidden_states))
        << "Fixture must expose later Full selection sensitivity";
  }
}

void run_parallel(int32_t rank, int32_t port, bool dcp) {
  torch::NoGradGuard no_grad;
  Device dev(rank);
  dev.set_device();
  const torch::Device device = dev.unwrap();
  KVCacheConfig::get_instance().block_size(/*value=*/1);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  auto groups = create_cp_tp_groups(
      rank, /*world_size=*/2, dcp ? 1 : 2, port, /*host=*/"127.0.0.1", device);
  ParallelArgs actual_parallel = *groups.parallel_args;
  const ParallelArgs reference_parallel = single_parallel(groups.self.get());
  if (dcp) {
    actual_parallel.kv_split_size() = 2;
    actual_parallel.dcp_group_ = groups.world.get();
  }
  const ModelArgs args = model_args();
  Weights weights = model_weights(args);
  // Distinct row gains keep LayerNorm variance above epsilon across CP
  // reductions.
  const torch::Tensor row_gains =
      torch::linspace(0.005f, 0.035f, args.index_head_dim(), torch::kFloat32)
          .to(torch::kBFloat16);
  for (int32_t layer = 0; layer < kLayers; layer += 2) {
    auto& key_weight = weights.at("model.layers." + std::to_string(layer) +
                                  ".self_attn.indexer.wk.weight");
    key_weight = row_gains.unsqueeze(1).expand_as(key_weight).clone() +
                 (key_weight - 0.025f) * 0.1f;
  }
  auto reference = create_model(args, reference_parallel, options, weights);
  auto actual = create_model(args, actual_parallel, options, weights);
  auto expected_cache = create_caches(args, options);
  auto actual_cache = create_caches(args, options, dcp ? 2 : 1);
  for (const bool decode : {false, true}) {
    const auto expected = forward(reference, expected_cache, device, decode);
    const auto output = forward(actual, actual_cache, device, decode);
    dev.synchronize_default_stream();
    check_output(output, decode ? 1 : kPrefill);
    expect_close(output.hidden_states,
                 expected.hidden_states,
                 /*rtol=*/0.02,
                 /*atol=*/0.02,
                 dcp ? "model DCP" : "model CP");

    for (int32_t layer = 0; layer < kLayers; ++layer) {
      torch::Tensor keys = expected_cache[layer].get_k_cache();
      torch::Tensor index = expected_cache[layer].get_index_cache();
      if (dcp) {
        const torch::Tensor slots =
            torch::arange(rank, kBlocks, 2, torch::kInt64).to(device);
        keys = keys.index_select(0, slots);
        index = index.index_select(0, slots);
      }
      const int32_t length = decode ? kPrefill + 1 : kPrefill;
      const torch::Tensor slots =
          torch::arange(keys.size(0), torch::kInt64).to(device) *
              (dcp ? 2 : 1) +
          (dcp ? rank : 0);
      const torch::Tensor written = slots < length;
      const torch::Tensor untouched = written.logical_not();
      const torch::Tensor actual_keys = actual_cache[layer].get_k_cache();
      const torch::Tensor actual_index = actual_cache[layer].get_index_cache();
      expect_close(actual_keys.index({written}),
                   keys.index({written}),
                   /*rtol=*/0.02,
                   /*atol=*/0.02,
                   "parallel model written key cache");
      EXPECT_TRUE(torch::equal(actual_keys.index({untouched}),
                               keys.index({untouched})));
      EXPECT_TRUE(torch::equal(actual_index.index({untouched}),
                               index.index({untouched})));
      if (layer % 2 == 0) {
        expect_close(actual_index.index({written}),
                     index.index({written}),
                     /*rtol=*/0.02,
                     /*atol=*/0.002,
                     "parallel model written index cache");
      } else {
        EXPECT_TRUE(torch::equal(actual_index, index));
      }
    }
    ASSERT_EQ(output.residual.defined(), expected.residual.defined());
    if (output.residual.defined()) {
      expect_close(output.residual,
                   expected.residual,
                   /*rtol=*/0.02,
                   /*atol=*/0.02,
                   dcp ? "model DCP residual" : "model CP residual");
    }
  }
}

TEST(Glm52MluForwardTest, PrefillAndDecodeExposeFullAndSharedCacheWrites) {
  {
    const auto ranks = /*ranks=*/1;
    const auto cp_size = /*cp_size=*/1;
    const auto fn = [](int32_t /*rank*/, int32_t /*port*/) {
      run_single(/*mutate_shared=*/false, /*mutate_full=*/false);
    };

    const int32_t port = parallel_base_port(ranks, cp_size);

    const int32_t child_count = ranks;
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
          const auto rank_body = [&](int32_t rank) {
            fn(rank, port);
            return 0;
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
        LOG(ERROR) << "fork failed: " << std::strerror(errno);
        children_ok = false;
        break;
      }
      child_pids.emplace_back(pid);
    }
    if (!children_ok) {
      stop_ranks(child_pids, std::vector<bool>(child_pids.size(), false));
    } else {
      children_ok = wait_ranks(
          child_pids, std::chrono::minutes(/*count=*/3), children_skipped);
    }

    if (children_ok && children_skipped) {
      GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                      "physical MLU cards";
    }
    ASSERT_EQ(children_ok, true)
        << "forked ranks failed, timed out, or require more physical MLU cards";
  };
}
TEST(Glm52MluForwardTest, SharedIndexerCheckpointDoesNotAffectStack) {
  {
    const auto ranks = /*ranks=*/1;
    const auto cp_size = /*cp_size=*/1;
    const auto fn = [](int32_t /*rank*/, int32_t /*port*/) {
      run_single(/*mutate_shared=*/true, /*mutate_full=*/false);
    };

    const int32_t port = parallel_base_port(ranks, cp_size);

    const int32_t child_count = ranks;
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
          const auto rank_body = [&](int32_t rank) {
            fn(rank, port);
            return 0;
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
        LOG(ERROR) << "fork failed: " << std::strerror(errno);
        children_ok = false;
        break;
      }
      child_pids.emplace_back(pid);
    }
    if (!children_ok) {
      stop_ranks(child_pids, std::vector<bool>(child_pids.size(), false));
    } else {
      children_ok = wait_ranks(
          child_pids, std::chrono::minutes(/*count=*/3), children_skipped);
    }

    if (children_ok && children_skipped) {
      GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                      "physical MLU cards";
    }
    ASSERT_EQ(children_ok, true)
        << "forked ranks failed, timed out, or require more physical MLU cards";
  };
}
TEST(Glm52MluForwardTest, LaterFullIndexerChangesSparseDecodeOutput) {
  {
    const auto ranks = /*ranks=*/1;
    const auto cp_size = /*cp_size=*/1;
    const auto fn = [](int32_t /*rank*/, int32_t /*port*/) {
      run_single(/*mutate_shared=*/false, /*mutate_full=*/true);
    };

    const int32_t port = parallel_base_port(ranks, cp_size);

    const int32_t child_count = ranks;
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
          const auto rank_body = [&](int32_t rank) {
            fn(rank, port);
            return 0;
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
        LOG(ERROR) << "fork failed: " << std::strerror(errno);
        children_ok = false;
        break;
      }
      child_pids.emplace_back(pid);
    }
    if (!children_ok) {
      stop_ranks(child_pids, std::vector<bool>(child_pids.size(), false));
    } else {
      children_ok = wait_ranks(
          child_pids, std::chrono::minutes(/*count=*/3), children_skipped);
    }

    if (children_ok && children_skipped) {
      GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                      "physical MLU cards";
    }
    ASSERT_EQ(children_ok, true)
        << "forked ranks failed, timed out, or require more physical MLU cards";
  };
}
TEST(Glm52MluForwardTest, CpStackMatchesUnshardedModel) {
  {
    const auto ranks = /*ranks=*/2;
    const auto cp_size = /*cp_size=*/2;
    const auto fn = [](int32_t rank, int32_t port) {
      run_parallel(rank, port, /*dcp=*/false);
    };

    const int32_t port = parallel_base_port(ranks, cp_size);

    const int32_t child_count = ranks;
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
          const auto rank_body = [&](int32_t rank) {
            fn(rank, port);
            return 0;
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
        LOG(ERROR) << "fork failed: " << std::strerror(errno);
        children_ok = false;
        break;
      }
      child_pids.emplace_back(pid);
    }
    if (!children_ok) {
      stop_ranks(child_pids, std::vector<bool>(child_pids.size(), false));
    } else {
      children_ok = wait_ranks(
          child_pids, std::chrono::minutes(/*count=*/3), children_skipped);
    }

    if (children_ok && children_skipped) {
      GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                      "physical MLU cards";
    }
    ASSERT_EQ(children_ok, true)
        << "forked ranks failed, timed out, or require more physical MLU cards";
  };
}
TEST(Glm52MluForwardTest, KvSplitStackMatchesUnsplitModel) {
  {
    const auto ranks = /*ranks=*/2;
    const auto cp_size = /*cp_size=*/1;
    const auto fn = [](int32_t rank, int32_t port) {
      run_parallel(rank, port, /*dcp=*/true);
    };

    const int32_t port = parallel_base_port(ranks, cp_size);

    const int32_t child_count = ranks;
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
          const auto rank_body = [&](int32_t rank) {
            fn(rank, port);
            return 0;
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
        LOG(ERROR) << "fork failed: " << std::strerror(errno);
        children_ok = false;
        break;
      }
      child_pids.emplace_back(pid);
    }
    if (!children_ok) {
      stop_ranks(child_pids, std::vector<bool>(child_pids.size(), false));
    } else {
      children_ok = wait_ranks(
          child_pids, std::chrono::minutes(/*count=*/3), children_skipped);
    }

    if (children_ok && children_skipped) {
      GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                      "physical MLU cards";
    }
    ASSERT_EQ(children_ok, true)
        << "forked ranks failed, timed out, or require more physical MLU cards";
  };
}

}  // namespace
}  // namespace xllm
