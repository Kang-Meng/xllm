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
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/config/kv_cache_config.h"
#include "core/layers/mlu/deepseek_v32/dcp_batch_metadata.h"
#include "core/layers/mlu/deepseek_v32/dcp_decode_context.h"
#include "core/layers/mlu/deepseek_v32/deepseek_v32_attention.h"
#include "core/layers/mlu/deepseek_v32/deepseek_v32_decoder_layer.h"
#include "core/layers/mlu/deepseek_v32/deepseek_v32_indexer.h"
#include "framework/parallel_state/context_parallel_topology.h"
#include "layers/common/attention_metadata_builder.h"
#include "layers/common/linear.h"
#include "layers/common/rms_norm.h"
#include "layers/common/rotary_embedding.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm::layer::test {
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

torch::Tensor active_topk(const torch::Tensor& table,
                          const torch::Tensor& lengths) {
  const auto columns = torch::arange(table.size(1), lengths.options());
  return torch::where(
      columns < lengths.unsqueeze(1), table, torch::full_like(table, -1));
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

namespace fixture = ::xllm::layer::test;

constexpr int32_t kBlock = 16;
constexpr int32_t kSuperblocks = 8;
constexpr std::array<int32_t, 3> kPages = {3, 1, 5};
enum class Topology {
  CP,
  CPTP,
  DCP,
  CPDCP,
  DCP4,
  CPTPDCP4,
  CPTPDCP2,
  CP4TP2DCP4
};
enum class Shape { PREFILL, CHUNKS, BOUNDARY15, BOUNDARY16, BOUNDARY33 };
struct PrefillCase {
  DsaModelVariant variant;
  Topology topology;
  Shape shape;
  std::string name;
};

::testing::AssertionResult bf16_output_close(const torch::Tensor& actual,
                                             const torch::Tensor& reference) {
  if (actual.sizes() != reference.sizes() || actual.dim() != 2 ||
      actual.scalar_type() != torch::kBFloat16 ||
      reference.scalar_type() != torch::kBFloat16 || actual.numel() == 0) {
    return ::testing::AssertionFailure()
           << "Expected matching nonempty BF16 rows";
  }
  const torch::Tensor a = actual.cpu().to(torch::kFloat64);
  const torch::Tensor r = reference.cpu().to(torch::kFloat64);
  if (!torch::isfinite(a).all().item<bool>() ||
      !torch::isfinite(r).all().item<bool>()) {
    return ::testing::AssertionFailure() << "Nonfinite output";
  }
  const torch::Tensor delta = a - r;
  const torch::Tensor reference_norm = r.square().sum(/*dim=*/-1).sqrt();
  const torch::Tensor delta_norm = delta.square().sum(/*dim=*/-1).sqrt();
  const torch::Tensor zero_rows = reference_norm.eq(/*other=*/0);
  if (delta_norm.masked_select(zero_rows).ne(/*other=*/0).any().item<bool>()) {
    return ::testing::AssertionFailure()
           << "Zero reference rows require exact output";
  }
  const torch::Tensor nonzero_rows = zero_rows.logical_not();
  const torch::Tensor relative_l2 = delta_norm.masked_select(nonzero_rows) /
                                    reference_norm.masked_select(nonzero_rows);
  const double max_relative_l2 =
      relative_l2.numel() == 0 ? 0.0 : relative_l2.max().item<double>();
  const torch::Tensor rms =
      r.square().mean(/*dim=*/-1, /*keepdim=*/true).sqrt();
  const torch::Tensor bound = 0.0002 + 0.02 * torch::maximum(r.abs(), rms);
  const int64_t pointwise_failures =
      delta.abs().gt(bound).count_nonzero().item<int64_t>();
  constexpr double kBf16Epsilon = 1.0 / 128.0;
  if (pointwise_failures != 0 || max_relative_l2 > kBf16Epsilon) {
    return ::testing::AssertionFailure()
           << "pointwise failures=" << pointwise_failures
           << " token relativeL2 max=" << max_relative_l2
           << " limit=" << kBf16Epsilon;
  }
  return ::testing::AssertionSuccess();
}

::testing::AssertionResult score_selection_legal(
    const torch::Tensor& query,
    const torch::Tensor& weights,
    const torch::Tensor& keys,
    const torch::Tensor& causal_slots,
    const torch::Tensor& selected,
    int64_t topk) {
  if (query.dim() != 2 || weights.dim() != 1 || keys.dim() != 2 ||
      causal_slots.dim() != 1 || selected.dim() != 1 || topk <= 0 ||
      query.size(/*dim=*/0) != weights.size(/*dim=*/0) ||
      query.size(/*dim=*/1) != keys.size(/*dim=*/1) ||
      query.size(/*dim=*/0) == 0 || query.size(/*dim=*/1) == 0 ||
      keys.size(/*dim=*/0) == 0 ||
      causal_slots.numel() != keys.size(/*dim=*/0) ||
      selected.numel() != std::min(topk, keys.size(/*dim=*/0)) ||
      query.scalar_type() != torch::kBFloat16 ||
      weights.scalar_type() != torch::kBFloat16 ||
      keys.scalar_type() != torch::kBFloat16 ||
      (causal_slots.scalar_type() != torch::kInt32 &&
       causal_slots.scalar_type() != torch::kInt64) ||
      (selected.scalar_type() != torch::kInt32 &&
       selected.scalar_type() != torch::kInt64)) {
    return ::testing::AssertionFailure()
           << "Invalid BF16 score/selection dimensions";
  }
  const torch::Tensor q = query.cpu().to(torch::kFloat64);
  const torch::Tensor w = weights.cpu().to(torch::kFloat64);
  const torch::Tensor k = keys.cpu().to(torch::kFloat64);
  if (!torch::isfinite(q).all().item<bool>() ||
      !torch::isfinite(w).all().item<bool>() ||
      !torch::isfinite(k).all().item<bool>()) {
    return ::testing::AssertionFailure() << "Nonfinite score operands";
  }
  const torch::Tensor slots = causal_slots.cpu().to(torch::kInt64).contiguous();
  const torch::Tensor chosen = selected.cpu().to(torch::kInt64).contiguous();
  if (slots.lt(/*other=*/0).any().item<bool>()) {
    return ::testing::AssertionFailure() << "Negative causal slot";
  }
  const int64_t* slot_ptr = slots.const_data_ptr<int64_t>();
  const int64_t* chosen_ptr = chosen.const_data_ptr<int64_t>();
  const std::unordered_set<int64_t> legal(slot_ptr, slot_ptr + slots.numel());
  const std::unordered_set<int64_t> unique(chosen_ptr,
                                           chosen_ptr + chosen.numel());
  if (static_cast<int64_t>(legal.size()) != slots.numel() ||
      static_cast<int64_t>(unique.size()) != chosen.numel()) {
    return ::testing::AssertionFailure() << "Duplicate causal or selected slot";
  }
  for (int64_t slot : unique) {
    if (!legal.contains(slot)) {
      return ::testing::AssertionFailure()
             << "Selected slot is not causal: " << slot;
    }
  }
  if (selected.numel() == keys.size(/*dim=*/0)) {
    return ::testing::AssertionSuccess();
  }
  const torch::Tensor products =
      torch::matmul(q, k.transpose(/*dim0=*/0, /*dim1=*/1))
          .clamp_min(/*min=*/0) *
      w.unsqueeze(/*dim=*/1);
  const torch::Tensor scores = products.sum(/*dim=*/0);
  const torch::Tensor head_abs = products.abs().sum(/*dim=*/0);
  const torch::Tensor dot_abs =
      (torch::matmul(q.abs(), k.abs().transpose(/*dim0=*/0, /*dim1=*/1)) *
       w.abs().unsqueeze(/*dim=*/1))
          .sum(/*dim=*/0);
  constexpr double kFp32Roundoff = 1.0 / 16777216.0;
  constexpr double kBf16Roundoff = 1.0 / 256.0;
  constexpr int32_t kBf16Stages = 3;
  const double dot_roundoff = query.size(/*dim=*/1) * kFp32Roundoff;
  const double head_roundoff = (query.size(/*dim=*/0) - 1) * kFp32Roundoff;
  if (dot_roundoff >= 1.0 || head_roundoff >= 1.0) {
    return ::testing::AssertionFailure()
           << "Score dimensions exceed roundoff model";
  }
  const double gamma_dot = dot_roundoff / (1.0 - dot_roundoff);
  const double gamma_head = head_roundoff / (1.0 - head_roundoff);
  // Engineering budget: at most three BF16 score stages plus FP32 reductions;
  // this is not a vendor guarantee about the internal kernel arithmetic.
  const double factor =
      std::pow(1.0 + kBf16Roundoff, kBf16Stages) * (1.0 + gamma_head);
  const torch::Tensor budget =
      (factor - 1.0) * head_abs + factor * gamma_dot * dot_abs;
  if (!torch::isfinite(scores).all().item<bool>() ||
      !torch::isfinite(budget).all().item<bool>()) {
    return ::testing::AssertionFailure() << "Nonfinite score interval";
  }
  torch::Tensor mask = torch::zeros({slots.numel()}, torch::kBool);
  bool* mask_ptr = mask.data_ptr<bool>();
  for (int64_t i = 0; i < slots.numel(); ++i) {
    mask_ptr[i] = unique.contains(slot_ptr[i]);
  }
  const double selected_min =
      (scores + budget).masked_select(mask).min().item<double>();
  const double omitted_max =
      (scores - budget).masked_select(mask.logical_not()).max().item<double>();
  if (omitted_max > selected_min) {
    return ::testing::AssertionFailure()
           << "Cutoff interval violation: omitted lower=" << omitted_max
           << " selected upper=" << selected_min;
  }
  return ::testing::AssertionSuccess();
}

TEST(AttentionReferenceContractTest, OutputRejectsCoherentDrift) {
  const torch::Tensor reference = torch::ones({1, 256}, torch::kBFloat16);
  EXPECT_FALSE(bf16_output_close(reference + 0.015625, reference));
}

TEST(AttentionReferenceContractTest, OutputRejectsSingleOutlier) {
  const torch::Tensor reference = torch::ones({1, 256}, torch::kBFloat16);
  torch::Tensor actual = reference.clone();
  actual[0][0] = 1.03125;
  EXPECT_FALSE(bf16_output_close(actual, reference));
}

TEST(AttentionReferenceContractTest, OutputAcceptsCancellationFloor) {
  torch::Tensor reference = torch::ones({1, 256}, torch::kBFloat16);
  reference[0][0] = 0;
  torch::Tensor actual = reference.clone();
  actual[0][0] = std::ldexp(/*arg=*/1.0, /*exp=*/-10);
  EXPECT_TRUE(bf16_output_close(actual, reference));
}

TEST(AttentionReferenceContractTest, ZeroReferenceRowsRequireExactOutput) {
  const torch::Tensor reference = torch::zeros({1, 256}, torch::kBFloat16);
  EXPECT_TRUE(bf16_output_close(reference, reference));
  EXPECT_FALSE(bf16_output_close(reference + 0.00001, reference));
}

TEST(AttentionReferenceContractTest, NonfiniteValuesAreRejected) {
  const torch::Tensor nonfinite = torch::full(
      {1, 1}, std::numeric_limits<double>::infinity(), torch::kBFloat16);
  EXPECT_FALSE(bf16_output_close(nonfinite, nonfinite));
  EXPECT_FALSE(score_selection_legal(nonfinite,
                                     torch::ones({1}, torch::kBFloat16),
                                     torch::ones({1, 1}, torch::kBFloat16),
                                     torch::tensor({96}),
                                     torch::tensor({96}),
                                     /*topk=*/1));
}

TEST(AttentionReferenceContractTest, SelectionAcceptsCutoffRounding) {
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  EXPECT_TRUE(
      score_selection_legal(torch::ones({1, 1}, bf16),
                            torch::ones({1}, bf16),
                            torch::tensor({{1.0}, {0.99609375}, {0.5}}, bf16),
                            torch::tensor({96, 97, 98}),
                            torch::tensor({97}),
                            /*topk=*/1));
}

TEST(AttentionReferenceContractTest, SelectionRejectsWorstSwap) {
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  EXPECT_FALSE(
      score_selection_legal(torch::ones({1, 1}, bf16),
                            torch::ones({1}, bf16),
                            torch::tensor({{1.0}, {0.99609375}, {0.5}}, bf16),
                            torch::tensor({96, 97, 98}),
                            torch::tensor({98}),
                            /*topk=*/1));
}

TEST(AttentionReferenceContractTest, SelectionRejectsInvalidStructure) {
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const torch::Tensor query = torch::ones({1, 1}, bf16);
  const torch::Tensor weights = torch::ones({1}, bf16);
  const torch::Tensor keys = torch::ones({3, 1}, bf16);
  const torch::Tensor slots = torch::tensor({96, 97, 98});
  EXPECT_FALSE(score_selection_legal(
      query, weights, keys, slots, torch::tensor({96, 96}), /*topk=*/2));
  EXPECT_FALSE(score_selection_legal(
      query, weights, keys, slots, torch::tensor({96, 99}), /*topk=*/2));
  EXPECT_FALSE(score_selection_legal(
      query, weights, keys, slots, torch::tensor({96}), /*topk=*/2));
}

TEST(AttentionReferenceContractTest, SelectionAcceptsCompleteCausalSet) {
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  EXPECT_TRUE(score_selection_legal(torch::ones({1, 1}, bf16),
                                    torch::ones({1}, bf16),
                                    torch::ones({3, 1}, bf16),
                                    torch::tensor({96, 97, 98}),
                                    torch::tensor({98, 96, 97}),
                                    /*topk=*/8));
}

ModelArgs prefill_args(DsaModelVariant variant) {
  ModelArgs args = create_dsa_model_args(variant);
  args.index_topk() = 8;
  args.max_position_embeddings() = 256;
  if (variant == DsaModelVariant::V32) {
    args.rope_scaling_rope_type() = "deepseek_yarn";
    args.rope_scaling_original_max_position_embeddings() = 128;
    args.rope_scaling_factor() = 2.0f;
    args.rope_extrapolation_factor() = 1.0f;
    args.rope_scaling_attn_factor() = 1.0f;
    args.rope_scaling_beta_fast() = 32;
    args.rope_scaling_beta_slow() = 1;
    args.rope_scaling_mscale() = 1.0f;
    args.rope_scaling_mscale_all_dim() = 1.0f;
    args.indexer_rope_interleave() = false;
  }
  return args;
}

torch::Tensor valid_selection(const DsaTopkState& state) {
  return active_topk(state.block_tables(), state.context_lens());
}

Weights centered_weights(Weights weights, const torch::TensorOptions& options) {
  for (auto& [name, value] : weights) {
    if (name.ends_with(".weight") && name.find("norm") == std::string::npos) {
      // The weights use .02+.01U for indexer matrices and .01U
      // for ordinary linear matrices. Center each source explicitly.
      const double center =
          name.find("indexer.") != std::string::npos ? 0.025 : 0.005;
      value =
          ((value.to(torch::kFloat32) - center) * 4.0).to(value.scalar_type());
      CHECK_LT(value.min().item<float>(), 0.0f) << name;
      CHECK_GT(value.max().item<float>(), 0.0f) << name;
    }
    value = value.to(options.device());
  }
  return weights;
}

std::vector<int32_t> phase_lengths(Shape shape) {
  switch (shape) {
    case Shape::CHUNKS:
      return {8, 15, 16, 33};
    case Shape::BOUNDARY15:
      return {15};
    case Shape::BOUNDARY16:
      return {16};
    case Shape::BOUNDARY33:
      return {33};
    case Shape::PREFILL:
      return {64};
  }
  LOG(FATAL) << "Invalid prefill shape";
  return {};
}

std::vector<int32_t> phase_slots(int32_t prefix,
                                 int32_t count,
                                 int32_t dcp_size = 2) {
  const int32_t logical_block = dcp_size * kBlock;
  std::vector<int32_t> slots;
  slots.reserve(count);
  for (int32_t pos = prefix; pos < prefix + count; ++pos) {
    slots.emplace_back(kPages.at(pos / logical_block) * logical_block +
                       pos % logical_block);
  }
  return slots;
}

ModelInputParams phase_params(int32_t prefix,
                              int32_t count,
                              bool sharded,
                              const torch::TensorOptions& options,
                              int32_t dcp_size = 2) {
  const auto ints = options.dtype(torch::kInt32);
  ModelInputParams params;
  params.meta.batch_forward_type = prefix == 0
                                       ? BatchForwardType::PREFILL
                                       : BatchForwardType::CHUNKED_PREFILL;
  params.meta.num_sequences = 1;
  params.meta.q_max_seq_len = count;
  params.meta.kv_max_seq_len = prefix + count;
  params.attention.device.q_seq_lens = torch::tensor({0, count}, ints);
  params.attention.device.kv_seq_lens =
      torch::tensor({0, prefix + count}, ints);
  params.attention.host.q_seq_lens = {count};
  params.attention.host.kv_seq_lens = {prefix + count};
  params.attention.device.new_cache_slots =
      torch::tensor(phase_slots(prefix, count, dcp_size), ints);
  std::vector<int32_t> blocks;
  blocks.reserve(kPages.size() * dcp_size);
  for (int32_t page : kPages) {
    if (sharded) {
      blocks.emplace_back(page);
      continue;
    }
    for (int32_t stripe = 0; stripe < dcp_size; ++stripe) {
      blocks.emplace_back(dcp_size * page + stripe);
    }
  }
  params.attention.device.block_tables =
      torch::tensor(blocks, ints).unsqueeze(/*dim=*/0);
  return params;
}

torch::Tensor resident_view(const torch::Tensor& full,
                            int32_t shard_rank,
                            int32_t dcp_size = 2) {
  if (shard_rank < 0) {
    return full.clone();
  }
  return full.view({kSuperblocks, dcp_size, kBlock, full.size(/*dim=*/-1)})
      .select(/*dim=*/1, shard_rank)
      .unsqueeze(/*dim=*/1)
      .contiguous();
}

KVCache initial_cache(const ModelArgs& args,
                      const torch::TensorOptions& options,
                      int32_t shard_rank,
                      int32_t dcp_size = 2) {
  const torch::Tensor key =
      seeded_tensor("v32.parallel.initial.key",
                    {dcp_size * kSuperblocks,
                     1,
                     kBlock,
                     args.kv_lora_rank() + args.qk_rope_head_dim()},
                    torch::kBFloat16,
                    torch::Device(torch::kCPU));
  const torch::Tensor index =
      seeded_tensor("v32.parallel.initial.index",
                    {dcp_size * kSuperblocks, 1, kBlock, args.index_head_dim()},
                    torch::kBFloat16,
                    torch::Device(torch::kCPU));
  return create_indexed_kv_cache(
      resident_view(key, shard_rank, dcp_size).to(options.device()),
      resident_view(index, shard_rank, dcp_size).to(options.device()));
}

void check_cache(const torch::Tensor& actual,
                 const torch::Tensor& reference,
                 const torch::Tensor& before,
                 const std::vector<int32_t>& slots,
                 int32_t shard_rank,
                 const std::string& label,
                 int32_t dcp_size = 2) {
  const int32_t logical_block = dcp_size * kBlock;
  torch::Tensor mask =
      torch::zeros({actual.size(/*dim=*/0) * kBlock}, torch::kBool);
  for (int32_t slot : slots) {
    if (shard_rank >= 0 && (slot % logical_block) / kBlock != shard_rank) {
      continue;
    }
    const int32_t local =
        shard_rank < 0 ? slot : slot / logical_block * kBlock + slot % kBlock;
    mask[local] = true;
  }
  const torch::Tensor flat =
      actual.cpu().reshape({-1, actual.size(/*dim=*/-1)});
  const torch::Tensor old = before.cpu().reshape_as(flat);
  const torch::Tensor expected =
      resident_view(reference.cpu(), shard_rank, dcp_size).reshape_as(flat);
  const torch::Tensor written = mask.unsqueeze(/*dim=*/1).expand_as(flat);
  if (mask.any().item<bool>()) {
    expect_close(flat.masked_select(written),
                 expected.masked_select(written),
                 /*rtol=*/0.02,
                 /*atol=*/0.002,
                 label + " written");
  }
  const torch::Tensor untouched = written.logical_not();
  EXPECT_GT(untouched.count_nonzero().item<int64_t>(), 0);
  EXPECT_TRUE(
      torch::equal(flat.masked_select(untouched), old.masked_select(untouched)))
      << label << " changed a prefix or unwritten slot";
}

torch::Tensor restore_output(
    DeepseekV32AttentionImpl::ForwardResult result,
    const std::optional<v32_cp::DeepseekV32CPContext>& cp,
    ProcessGroup* tp_group) {
  if (result.layout == PostAttnLayout::PACKED_LOCAL) {
    CHECK(cp.has_value());
    return v32_cp::gather_and_restore_global(result.output, cp.value());
  }
  if (result.layout == PostAttnLayout::TP_SHARD) {
    return parallel_state::reduce(result.output, tp_group);
  }
  return result.output;
}

void check_projection_shards(const DeepseekV32Attention& attention,
                             const DeepseekV32Attention& reference,
                             int32_t tp_rank,
                             int32_t tp_size) {
  const auto local_modules = attention->named_children();
  const auto full_modules = reference->named_children();
  for (const std::string& name : {"q_b_proj", "kv_b_proj"}) {
    const auto local = local_modules[name]->as<ColumnParallelLinearImpl>();
    const auto full = full_modules[name]->as<ColumnParallelLinearImpl>();
    EXPECT_TRUE(torch::equal(local->weight(),
                             full->weight().chunk(tp_size, /*dim=*/0)[tp_rank]))
        << name;
    if (full->per_channel_scale().defined()) {
      EXPECT_TRUE(
          torch::equal(local->per_channel_scale(),
                       full->per_channel_scale().chunk(tp_size)[tp_rank]))
          << name;
      ASSERT_TRUE(local->smooth().has_value());
      EXPECT_TRUE(torch::equal(*local->smooth(), *full->smooth())) << name;
    }
  }
  const auto local = local_modules["o_proj"]->as<RowParallelLinearImpl>();
  const auto full = full_modules["o_proj"]->as<RowParallelLinearImpl>();
  EXPECT_TRUE(torch::equal(local->weight(),
                           full->weight().chunk(tp_size, /*dim=*/1)[tp_rank]));
  if (full->per_channel_scale().defined()) {
    EXPECT_TRUE(
        torch::equal(local->per_channel_scale(), full->per_channel_scale()));
    ASSERT_TRUE(local->smooth().has_value());
    EXPECT_TRUE(torch::equal(*local->smooth(),
                             full->smooth()->chunk(tp_size)[tp_rank]));
  }
}

void run_prefill_case(const PrefillCase& config,
                      int32_t rank,
                      int32_t world_size,
                      int32_t port) {
  torch::NoGradGuard no_grad;
  Device device(rank);
  device.set_device();
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
  KVCacheConfig::get_instance().block_size(kBlock);
  const bool cp_enabled =
      config.topology != Topology::DCP && config.topology != Topology::DCP4;
  const int32_t cp_size =
      config.topology == Topology::CP4TP2DCP4 ? 4 : (cp_enabled ? 2 : 1);
  const int32_t dcp_size = config.topology == Topology::DCP4 ||
                                   config.topology == Topology::CPTPDCP4 ||
                                   config.topology == Topology::CP4TP2DCP4
                               ? 4
                               : 2;
  const bool sharded = config.topology == Topology::DCP ||
                       config.topology == Topology::CPDCP ||
                       config.topology == Topology::DCP4 ||
                       config.topology == Topology::CPTPDCP4 ||
                       config.topology == Topology::CPTPDCP2 ||
                       config.topology == Topology::CP4TP2DCP4;
  auto groups = create_cp_tp_groups(rank,
                                    world_size,
                                    cp_size,
                                    port,
                                    /*host=*/"127.0.0.1",
                                    device.unwrap());
  ParallelArgs parallel = *groups.parallel_args;
  parallel.kv_split_size() = sharded ? dcp_size : 1;
  const bool dcp_within_cp = config.topology == Topology::CPTPDCP2 ||
                             config.topology == Topology::CP4TP2DCP4;
  parallel.dcp_group_ =
      dcp_within_cp ? groups.cp.get()
                    : (sharded ? groups.world.get() : groups.self.get());
  ParallelArgs single(/*rank=*/0, /*world_size=*/1, groups.self.get());
  single.tp_group_ = groups.self.get();
  single.cp_group_ = groups.self.get();
  single.dcp_group_ = groups.self.get();
  single.single_rank_group_ = groups.self.get();
  single.kv_split_size() = 1;
  const int32_t shard_rank = sharded ? parallel.dcp_group_->rank() : -1;
  const ModelArgs args = prefill_args(config.variant);
  StateDict weights(centered_weights(
      create_seeded_mla_weights(
          args, DsaWeightFormat::BF16, options.device(torch::kCPU)),
      options));
  const ModelContext context(parallel, args, QuantArgs(), options);
  const ModelContext reference_context(single, args, QuantArgs(), options);
  DeepseekV32Attention attention(context);
  DeepseekV32Attention reference(reference_context);
  if (config.topology == Topology::CPTPDCP4) {
    EXPECT_EQ(parallel.cp_size(), 2);
    EXPECT_EQ(parallel.tp_size(), 2);
    EXPECT_EQ(parallel.kv_split_size_effective(), 4);
    EXPECT_EQ(groups.cp->world_size(), 2);
    EXPECT_EQ(groups.tp->world_size(), 2);
    EXPECT_EQ(parallel.dcp_group_->world_size(), 4);
    EXPECT_GT(parallel.dcp_group_->world_size(), parallel.cp_size());
  }
  attention->load_state_dict(weights);
  reference->load_state_dict(weights);
  attention->verify_loaded_weights();
  reference->verify_loaded_weights();
  check_projection_shards(
      attention, reference, groups.tp->rank(), groups.tp->world_size());
  if (groups.tp->world_size() > 1) {
    const QuantArgs quant = create_default_quant_args();
    const StateDict quant_weights(create_seeded_mla_weights(
        args, DsaWeightFormat::SMOOTHQUANT, options.device(torch::kCPU)));
    DeepseekV32Attention quant_attention(
        ModelContext(parallel, args, quant, options));
    DeepseekV32Attention quant_reference(
        ModelContext(single, args, quant, options));
    quant_attention->load_state_dict(quant_weights);
    quant_reference->load_state_dict(quant_weights);
    quant_attention->verify_loaded_weights();
    quant_reference->verify_loaded_weights();
    check_projection_shards(quant_attention,
                            quant_reference,
                            groups.tp->rank(),
                            groups.tp->world_size());
  }
  KVCache cache = initial_cache(args, options, shard_rank, dcp_size);
  KVCache reference_cache =
      initial_cache(args, options, /*shard_rank=*/-1, dcp_size);
  DeepseekV32Attention fixed_reference(args,
                                       QuantArgs(),
                                       single,
                                       options,
                                       context.get_optimization_config(),
                                       /*enable_indexer=*/false);
  fixed_reference->load_state_dict(weights);
  fixed_reference->verify_loaded_weights();
  KVCache fixed_cache =
      initial_cache(args, options, /*shard_rank=*/-1, dcp_size);
  ReplicatedLinear query_projection(args.hidden_size(),
                                    args.q_lora_rank(),
                                    /*bias=*/false,
                                    QuantArgs(),
                                    options);
  RMSNorm query_norm(args.q_lora_rank(), args.rms_norm_eps(), options);
  query_projection->load_state_dict(weights.get_dict_with_prefix("q_a_proj."));
  query_norm->load_state_dict(weights.get_dict_with_prefix("q_a_layernorm."));
  const auto rope = create_mla_rotary_embedding(args,
                                                args.qk_rope_head_dim(),
                                                args.max_position_embeddings(),
                                                args.indexer_rope_interleave(),
                                                options);
  const bool fused_qk =
      context.get_optimization_config().enable_fused_indexer_qk;
  DeepseekV32Indexer diagnostic(args.hidden_size(),
                                args.index_n_heads(),
                                args.index_head_dim(),
                                args.qk_rope_head_dim(),
                                args.index_topk(),
                                args.q_lora_rank(),
                                fused_qk,
                                rope,
                                QuantArgs(),
                                parallel,
                                options);
  DeepseekV32Indexer ordinary(args.hidden_size(),
                              args.index_n_heads(),
                              args.index_head_dim(),
                              args.qk_rope_head_dim(),
                              args.index_topk(),
                              args.q_lora_rank(),
                              fused_qk,
                              rope,
                              QuantArgs(),
                              single,
                              options);
  diagnostic->load_state_dict(weights.get_dict_with_prefix("indexer."));
  ordinary->load_state_dict(weights.get_dict_with_prefix("indexer."));
  torch::Tensor diagnostic_cache = cache.get_index_cache().clone();
  torch::Tensor ordinary_cache = reference_cache.get_index_cache().clone();
  std::optional<DcpDecodeContext> dcp;
  if (sharded) {
    dcp.emplace(KVShardLayout(kBlock, dcp_size, shard_rank),
                parallel.dcp_group_);
  }
  const torch::Tensor hidden = (seeded_tensor("v32.parallel.centered.hidden",
                                              {72, args.hidden_size()},
                                              torch::kBFloat16,
                                              torch::Device(torch::kCPU)) *
                                    2.0 -
                                1.0)
                                   .to(options.device());

  int32_t prefix = 0;
  for (int32_t count : phase_lengths(config.shape)) {
    SCOPED_TRACE(config.name + " prefix=" + std::to_string(prefix));
    const auto params = phase_params(prefix, count, sharded, options, dcp_size);
    const auto reference_params =
        phase_params(prefix, count, /*sharded=*/false, options, dcp_size);
    auto metadata = AttentionMetadataBuilder::build(
        params, /*enable_mla=*/true, /*compute_dtype=*/"bfloat16");
    const auto reference_metadata = AttentionMetadataBuilder::build(
        reference_params, /*enable_mla=*/true, /*compute_dtype=*/"bfloat16");
    std::optional<KVShardLayout> layout;
    if (sharded) {
      layout.emplace(kBlock, dcp_size, shard_rank);
      metadata.kv_shard_batch_metadata =
          build_mlu_shard_metadata(metadata, layout.value());
    }
    const auto ints = options.dtype(torch::kInt32);
    const torch::Tensor positions = torch::arange(prefix, prefix + count, ints);
    const torch::Tensor input = hidden.narrow(/*dim=*/0, prefix, count);
    const torch::Tensor tokens = torch::arange(count, ints);
    std::optional<v32_cp::DeepseekV32CPContext> cp;
    if (cp_enabled) {
      cp = v32_cp::build_deepseek_v32_cp_context(cp_size,
                                                 metadata,
                                                 params.meta.batch_forward_type,
                                                 tokens,
                                                 groups.cp.get(),
                                                 groups.cp->rank(),
                                                 groups.cp->world_size(),
                                                 layout);
      CHECK(cp.has_value()) << "Expected a real CP forward";
    }
    const torch::Tensor key_before = cache.get_k_cache().clone();
    const torch::Tensor index_before = cache.get_index_cache().clone();
    auto reference_transfer = DsaTopkTransfer::capture_output();
    auto transfer = DsaTopkTransfer::capture_output();
    const torch::Tensor expected =
        restore_output(reference->forward(positions,
                                          input,
                                          reference_metadata,
                                          reference_cache,
                                          /*cp_ctx=*/nullptr,
                                          &reference_transfer),
                       /*cp=*/std::nullopt,
                       groups.self.get());
    const torch::Tensor local_input =
        cp ? v32_cp::reorder_to_local_shard(input, *cp) : input;
    const torch::Tensor local_positions =
        cp ? v32_cp::reorder_to_local_shard(positions, *cp) : positions;
    auto actual_result = attention->forward(local_positions,
                                            local_input,
                                            metadata,
                                            cache,
                                            cp ? &cp.value() : nullptr,
                                            &transfer);
    if (cp) {
      EXPECT_EQ(actual_result.layout, PostAttnLayout::PACKED_LOCAL);
      EXPECT_EQ(actual_result.output.size(/*dim=*/0),
                cp->comm_plan.tokens_per_rank.at(cp->rank));
      EXPECT_EQ(actual_result.output.size(/*dim=*/1), args.hidden_size());
    }
    const torch::Tensor actual =
        restore_output(std::move(actual_result), cp, groups.tp.get());
    device.synchronize_default_stream();
    EXPECT_TRUE(torch::isfinite(actual).all().item<bool>());
    const torch::Tensor all_outputs =
        parallel_state::gather(actual, groups.world.get(), /*dim=*/0);
    for (int32_t peer = 0; peer < world_size; ++peer) {
      EXPECT_TRUE(torch::equal(
          all_outputs.narrow(/*dim=*/0, peer * count, count), actual))
          << "Restored output differs across ranks: peer=" << peer;
    }
    CHECK(transfer.output() != nullptr);
    CHECK(reference_transfer.output() != nullptr);
    torch::Tensor selected = valid_selection(*transfer.output());
    torch::Tensor lengths = transfer.output()->context_lens();
    if (cp) {
      selected = v32_cp::gather_and_restore_global(selected, *cp);
      lengths = v32_cp::gather_and_restore_global(lengths, *cp);
    }
    EXPECT_TRUE(torch::equal(
        lengths.cpu(), reference_transfer.output()->context_lens().cpu()));
    const DsaTopkState global_state(selected.clone(), lengths.clone());
    auto fixed_transfer = DsaTopkTransfer::reuse(global_state);
    const torch::Tensor fixed_output =
        restore_output(fixed_reference->forward(positions,
                                                input,
                                                reference_metadata,
                                                fixed_cache,
                                                /*cp_ctx=*/nullptr,
                                                &fixed_transfer),
                       /*cp=*/std::nullopt,
                       groups.self.get());
    EXPECT_TRUE(torch::equal(global_state.block_tables(), selected));
    EXPECT_TRUE(torch::equal(global_state.context_lens(), lengths));
    EXPECT_TRUE(
        torch::equal(fixed_cache.get_k_cache(), reference_cache.get_k_cache()))
        << "Fixed-selection reference MLA cache differs";
    EXPECT_GT(fixed_output.cpu()
                  .to(torch::kFloat64)
                  .square()
                  .sum(/*dim=*/-1)
                  .min()
                  .item<double>(),
              0.0)
        << "Every reference token must carry a nonzero signal";
    EXPECT_TRUE(bf16_output_close(actual, fixed_output))
        << "Fixed actual-selection output engineering guard";

    torch::Tensor projected = query_projection(input);
    const torch::Tensor normalized = std::get<0>(query_norm(projected));
    const auto [ordinary_table, ordinary_lengths] =
        ordinary->forward(input,
                          normalized,
                          positions,
                          ordinary_cache,
                          reference_metadata,
                          /*is_prefill=*/true);
    EXPECT_TRUE(torch::equal(ordinary_cache, reference_cache.get_index_cache()))
        << "Public ordinary indexer cache differs";
    const torch::Tensor ordinary_selected =
        active_topk(ordinary_table, ordinary_lengths);
    EXPECT_TRUE(torch::equal(
        std::get<0>(ordinary_selected.sort(/*dim=*/-1)),
        std::get<0>(
            valid_selection(*reference_transfer.output()).sort(/*dim=*/-1))))
        << "Public ordinary indexer selection differs";

    const auto normal_pre =
        ordinary->cp_pre(input, normalized, positions, reference_metadata);
    torch::Tensor local_projected = query_projection(local_input);
    const torch::Tensor local_normalized =
        std::get<0>(query_norm(local_projected));
    auto pre = diagnostic->cp_pre(local_input,
                                  local_normalized,
                                  local_positions,
                                  cp ? cp->local_attn_metadata : metadata);
    torch::Tensor diagnostic_selected;
    if (cp) {
      const auto handle = diagnostic->cp_comm(pre.k_local, *cp);
      const torch::Tensor gathered_keys =
          diagnostic->cp_wait_k(pre.k_local, handle);
      if (!sharded) {
        const auto [table, lens] =
            diagnostic->cp_post(pre,
                                gathered_keys,
                                diagnostic_cache,
                                metadata,
                                cp->gathered_slot_mapping,
                                *cp);
        diagnostic_selected =
            v32_cp::gather_and_restore_global(active_topk(table, lens), *cp);
      }
      pre.q = v32_cp::gather_and_restore_global(pre.q, *cp);
      pre.weights = v32_cp::gather_and_restore_global(pre.weights, *cp);
      pre.k_local =
          v32_cp::restore_gathered_to_global_order(gathered_keys, *cp);
    }
    if (sharded) {
      const auto& causal = metadata.kv_shard_batch_metadata->causal_selector;
      AttentionMetadata local_metadata = metadata;
      local_metadata.slot_mapping =
          metadata.kv_shard_batch_metadata->local_slot_mapping;
      AttentionMetadata selector = local_metadata;
      selector.q_cu_seq_lens = causal.q_cu_seq_lens;
      selector.kv_cu_seq_lens = causal.q_cu_seq_lens;
      selector.kv_seq_lens = causal.local_context_lens;
      selector.block_table = causal.block_table;
      selector.max_query_len = 1;
      selector.max_seq_len = args.index_topk();
      selector.total_kv_len = 0;
      selector.is_prefill = false;
      selector.is_chunked_prefill = false;
      const auto candidates =
          cp ? diagnostic->forward_dcp_local_prefill_from_cp(
                   pre, diagnostic_cache, local_metadata, selector)
             : diagnostic->forward_dcp_local_prefill(input,
                                                     normalized,
                                                     positions,
                                                     diagnostic_cache,
                                                     local_metadata,
                                                     selector);
      auto gathered = dcp->launch_indexer_candidate_gather(
          candidates.scores, candidates.global_slots);
      const auto merged = dcp->finish_indexer_candidate_merge(
          std::move(gathered), args.index_topk(), metadata.slot_mapping);
      diagnostic_selected = valid_selection(merged);
    }
    device.synchronize_default_stream();
    EXPECT_TRUE(torch::equal(diagnostic_cache, cache.get_index_cache()))
        << "Public parallel indexer cache differs";
    EXPECT_TRUE(torch::equal(std::get<0>(diagnostic_selected.sort(/*dim=*/-1)),
                             std::get<0>(selected.sort(/*dim=*/-1))))
        << "Public parallel indexer selection differs";
    const torch::Tensor full_index =
        reference_cache.get_index_cache().cpu().reshape(
            {-1, args.index_head_dim()});
    const torch::Tensor suffix_slots =
        torch::tensor(phase_slots(prefix, count, dcp_size), torch::kInt64);
    const torch::Tensor suffix_keys =
        full_index.index_select(/*dim=*/0, suffix_slots);
    EXPECT_TRUE(torch::equal(pre.k_local.cpu(), suffix_keys))
        << "Parallel score keys differ from independent full cache";
    EXPECT_TRUE(torch::equal(normal_pre.k_local.cpu(), suffix_keys))
        << "Ordinary score keys differ from independent full cache";
    const torch::Tensor causal_slots = torch::tensor(
        phase_slots(/*prefix=*/0, prefix + count, dcp_size), torch::kInt64);
    const torch::Tensor q = pre.q.cpu();
    const torch::Tensor w = pre.weights.cpu();
    const torch::Tensor normal_q = normal_pre.q.cpu();
    const torch::Tensor normal_w = normal_pre.weights.cpu();
    const torch::Tensor chosen = selected.cpu();
    const torch::Tensor normal_chosen = ordinary_selected.cpu();
    const torch::Tensor lens = lengths.cpu();
    const torch::Tensor normal_lens = ordinary_lengths.cpu();
    for (int32_t row = 0; row < count; ++row) {
      SCOPED_TRACE("score row=" + std::to_string(row));
      const int64_t valid_count =
          std::min<int64_t>(args.index_topk(), prefix + row + 1);
      ASSERT_EQ(lens[row].item<int32_t>(), valid_count);
      ASSERT_EQ(normal_lens[row].item<int32_t>(), valid_count);
      const torch::Tensor row_slots =
          causal_slots.narrow(/*dim=*/0, /*start=*/0, prefix + row + 1);
      const torch::Tensor row_keys =
          full_index.index_select(/*dim=*/0, row_slots);
      EXPECT_TRUE(score_selection_legal(
          q[row],
          w[row],
          row_keys,
          row_slots,
          chosen[row].narrow(/*dim=*/0, /*start=*/0, valid_count),
          args.index_topk()))
          << "Parallel selected scores";
      EXPECT_TRUE(score_selection_legal(
          normal_q[row],
          normal_w[row],
          row_keys,
          row_slots,
          normal_chosen[row].narrow(/*dim=*/0, /*start=*/0, valid_count),
          args.index_topk()))
          << "Ordinary selected scores";
    }
    const auto slots = phase_slots(prefix, count, dcp_size);
    check_cache(cache.get_k_cache(),
                reference_cache.get_k_cache(),
                key_before,
                slots,
                shard_rank,
                "MLA cache",
                dcp_size);
    check_cache(cache.get_index_cache(),
                reference_cache.get_index_cache(),
                index_before,
                slots,
                shard_rank,
                "index cache",
                dcp_size);
    const std::string phase = "prefix" + std::to_string(prefix);

    const torch::Tensor reference_written_index =
        reference_cache.get_index_cache()
            .cpu()
            .reshape({-1, args.index_head_dim()})
            .index_select(/*dim=*/0, torch::tensor(slots, torch::kInt64));

    prefix += count;
  }
}

std::vector<PrefillCase> prefill_cases() {
  std::vector<PrefillCase> result;
  result.reserve(40);
  const std::array<std::string, 4> topology_names = {
      "CP2", "CP2TP2", "DCP2", "CP2DCP2"};
  const std::array<std::string, 5> shape_names = {
      "Prefill", "Chunks3", "Boundary15", "Boundary16", "Boundary33"};
  for (DsaModelVariant variant :
       {DsaModelVariant::V32, DsaModelVariant::GLM5}) {
    for (int32_t topology = 0; topology < 4; ++topology) {
      for (int32_t shape = 0; shape < 5; ++shape) {
        const std::string name =
            (variant == DsaModelVariant::V32 ? "V32_" : "Glm5_") +
            topology_names[topology] + "_" + shape_names[shape];
        result.emplace_back(PrefillCase{variant,
                                        static_cast<Topology>(topology),
                                        static_cast<Shape>(shape),
                                        name});
      }
    }
  }
  return result;
}

void run_prefill_cases(const std::vector<PrefillCase>& cases) {
  for (const auto& param : cases) {
    const PrefillCase config = param;
    const bool tp2 = config.topology == Topology::CPTP ||
                     config.topology == Topology::CPTPDCP2 ||
                     config.topology == Topology::CPTPDCP4;
    const int32_t ranks =
        config.topology == Topology::CP4TP2DCP4 ? 8 : (tp2 ? 4 : 2);
    const int32_t port =
        parallel_base_port(ranks,
                           config.topology == Topology::CP4TP2DCP4
                               ? 4
                               : (config.topology == Topology::DCP ? 1 : 2));
    LOG(INFO) << config.name << " base_port=" << port;

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
            run_prefill_case(config, rank, ranks, port);
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
          child_pids, std::chrono::milliseconds{180000}, children_skipped);
    }

    if (children_ok && children_skipped) {
      GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                      "physical MLU cards";
    }
    EXPECT_EQ(children_ok, true)
        << "forked ranks failed, timed out, or require more physical MLU cards";
  }
}

TEST(AttentionPrefillParallelTest,
     MatchesSingleRankAndPreservesUnwrittenSlots) {
  run_prefill_cases(prefill_cases());
}

TEST(AttentionCpTpDcpTest, HistoryMatchesSingleRankAndPreservesUnwrittenSlots) {
  run_prefill_cases({
      {DsaModelVariant::GLM5,
       Topology::CPTPDCP2,
       Shape::CHUNKS,
       "Glm5_CP2TP2DCP2_Chunks3"},
      {DsaModelVariant::GLM5,
       Topology::CPTPDCP4,
       Shape::CHUNKS,
       "Glm5_CP2TP2DCP4_Chunks3"},
      {DsaModelVariant::GLM5,
       Topology::CP4TP2DCP4,
       Shape::CHUNKS,
       "Glm5_CP4TP2DCP4_Chunks3"},
  });
}

TEST(DcpChunkedParallelTest, HistoryMatchesSingleRankAcrossFourShards) {
  for (const auto& param : std::vector<DsaModelVariant>{
           DsaModelVariant::V32, DsaModelVariant::GLM5}) {
    // C5: four physical cards, DCP4 and a real historical prefix plus three
    // chunks.
    const DsaModelVariant variant = param;
    const PrefillCase config{variant,
                             Topology::DCP4,
                             Shape::CHUNKS,
                             variant == DsaModelVariant::V32
                                 ? "V32_DCP4_Chunks3"
                                 : "Glm5_DCP4_Chunks3"};
    const int32_t port = parallel_base_port(/*world_size=*/4, /*cp_size=*/1);

    const int32_t child_count = 4;
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
            run_prefill_case(config, rank, /*world_size=*/4, port);
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
          child_pids, std::chrono::milliseconds{180000}, children_skipped);
    }

    if (children_ok && children_skipped) {
      GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                      "physical MLU cards";
    }
    EXPECT_EQ(children_ok, true)
        << "forked ranks failed, timed out, or require more physical MLU cards";
  }
}

TEST(V32AttentionParallelTest, CpTpDcp4PrefillMatchesSingleRank) {
  const PrefillCase config{.variant = DsaModelVariant::V32,
                           .topology = Topology::CPTPDCP4,
                           .shape = Shape::PREFILL,
                           .name = "V32_CP2TP2DCP4_Prefill"};
  const int32_t port = parallel_base_port(/*world_size=*/4, /*cp_size=*/2);

  const int32_t child_count = 4;
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
          run_prefill_case(config, rank, /*world_size=*/4, port);
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
      children_ok = false;
      break;
    }
    child_pids.emplace_back(pid);
  }
  if (!children_ok) {
    stop_ranks(child_pids, std::vector<bool>(child_pids.size(), false));
  } else {
    children_ok = wait_ranks(
        child_pids, std::chrono::milliseconds{180000}, children_skipped);
  }

  if (children_ok && children_skipped) {
    GTEST_SKIP()
        << "forked ranks failed, timed out, or require more physical MLU cards";
  }
  EXPECT_EQ(children_ok, true)
      << "forked ranks failed, timed out, or require more physical MLU cards";
}

Weights decoder_weights(const ModelArgs& args,
                        bool moe,
                        int32_t layer_id,
                        const torch::TensorOptions& options) {
  const auto cpu_options = options.device(torch::kCPU);
  Weights weights = create_seeded_mla_weights(
      args, DsaWeightFormat::BF16, cpu_options, "self_attn.");
  if (moe) {
    ModelArgs expert_args = args;
    expert_args.intermediate_size() = args.moe_intermediate_size();
    for (int32_t expert = 0; expert < args.n_routed_experts(); ++expert) {
      weights.merge(create_seeded_dense_weights(
          expert_args,
          DsaWeightFormat::BF16,
          cpu_options,
          "mlp.experts." + std::to_string(expert) + "."));
    }
    expert_args.intermediate_size() =
        args.moe_intermediate_size() * args.n_shared_experts();
    weights.merge(create_seeded_dense_weights(expert_args,
                                              DsaWeightFormat::BF16,
                                              cpu_options,
                                              "mlp.shared_experts."));
    weights["mlp.gate.weight"] =
        seeded_tensor("v32.parallel.gate",
                      {args.n_routed_experts(), args.hidden_size()},
                      torch::kBFloat16,
                      torch::Device(torch::kCPU)) *
        0.01;
    weights["mlp.gate.e_score_correction_bias"] = torch::zeros(
        {args.n_routed_experts()}, cpu_options.dtype(torch::kFloat32));
  } else {
    weights.merge(create_seeded_dense_weights(
        args, DsaWeightFormat::BF16, cpu_options, "mlp."));
  }
  weights = centered_weights(std::move(weights), options);
  if (layer_id == 3) {
    weights.at("self_attn.indexer.weights_proj.weight").neg_();
  }
  weights["input_layernorm.weight"] =
      torch::ones({args.hidden_size()}, options.dtype(torch::kFloat32));
  weights["post_attention_layernorm.weight"] =
      torch::ones({args.hidden_size()}, options.dtype(torch::kFloat32));
  return weights;
}

ParallelArgs self_parallel(ProcessGroup* group) {
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, group);
  parallel.tp_group_ = group;
  parallel.cp_group_ = group;
  parallel.dcp_group_ = group;
  parallel.moe_tp_group_ = group;
  parallel.moe_ep_group_ = group;
  parallel.single_rank_group_ = group;
  parallel.kv_split_size() = 1;
  return parallel;
}

void run_cp_decoder(int32_t rank,
                    int32_t port,
                    bool moe,
                    int32_t count,
                    bool expected_rs) {
  torch::NoGradGuard no_grad;
  Device device(rank);
  device.set_device();
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
  KVCacheConfig::get_instance().block_size(kBlock);
  auto groups = create_cp_tp_groups(rank,
                                    /*world_size=*/2,
                                    /*cp_size=*/2,
                                    port,
                                    "127.0.0.1",
                                    device.unwrap());
  ParallelArgs parallel = *groups.parallel_args;
  if (moe) {
    parallel.ep_size() = 2;
    parallel.moe_ep_group_ = groups.world.get();
    parallel.moe_tp_group_ = groups.self.get();
  }
  const ParallelArgs single = self_parallel(groups.self.get());
  ModelArgs args = prefill_args(DsaModelVariant::GLM5);
  args.index_topk() = 64;
  const int32_t layer_id = moe ? 1 : 0;
  StateDict weights(decoder_weights(args, moe, layer_id, options));
  DeepseekV32DecoderLayer actual(
      ModelContext(parallel, args, QuantArgs(), options), layer_id);
  DeepseekV32DecoderLayer reference(
      ModelContext(single, args, QuantArgs(), options), layer_id);
  actual->load_state_dict(weights);
  reference->load_state_dict(weights);
  actual->verify_loaded_weights();
  reference->verify_loaded_weights();
  const auto params =
      phase_params(/*prefix=*/0, count, /*sharded=*/false, options);
  const auto metadata = AttentionMetadataBuilder::build(
      params, /*enable_mla=*/true, /*compute_dtype=*/"bfloat16");
  const auto ints = options.dtype(torch::kInt32);
  const torch::Tensor positions = torch::arange(count, ints);
  const torch::Tensor tokens = torch::arange(count, ints);
  auto cp = v32_cp::build_deepseek_v32_cp_context(
      /*cp_size=*/2,
      metadata,
      BatchForwardType::PREFILL,
      tokens,
      groups.cp.get(),
      groups.cp->rank(),
      groups.cp->world_size());
  CHECK(cp.has_value());
  // Even/odd token counts select balanced/unbalanced FFN collective layouts.
  EXPECT_EQ(cp->comm_plan.ffn_can_rs, expected_rs);
  actual->set_context_parallel_context(&cp.value());
  torch::Tensor hidden = (seeded_tensor("v32.parallel.decoder",
                                        {count, args.hidden_size()},
                                        torch::kBFloat16,
                                        torch::Device(torch::kCPU)) *
                              2.0 -
                          1.0)
                             .to(options.device());
  torch::Tensor full_input = hidden.clone();
  torch::Tensor local_input =
      v32_cp::reorder_to_local_shard(hidden, cp.value());
  torch::Tensor full_positions = positions.clone();
  torch::Tensor local_positions =
      v32_cp::reorder_to_local_shard(positions, cp.value());
  KVCache cache = initial_cache(args, options, /*shard_rank=*/-1);
  KVCache reference_cache = initial_cache(args, options, /*shard_rank=*/-1);
  const torch::Tensor before = cache.get_k_cache().clone();
  std::optional<torch::Tensor> residual;
  std::optional<torch::Tensor> reference_residual;
  const torch::Tensor expected = reference->forward(full_input,
                                                    reference_residual,
                                                    full_positions,
                                                    metadata,
                                                    reference_cache,
                                                    params);
  const torch::Tensor output = actual->forward(
      local_input, residual, local_positions, metadata, cache, params);
  const torch::Tensor restored =
      v32_cp::gather_and_restore_global(output, cp.value());
  device.synchronize_default_stream();
  expect_close(restored, expected, /*rtol=*/0.02, /*atol=*/0.002, "CP decoder");
  EXPECT_FALSE(residual.has_value());
  check_cache(cache.get_k_cache(),
              reference_cache.get_k_cache(),
              before,
              phase_slots(/*prefix=*/0, count),
              /*shard_rank=*/-1,
              "decoder KV");
}

struct DecoderCpCase {
  bool moe;
  int32_t count;
  bool can_rs;
  std::string name;
};

TEST(DecoderCpParallelTest, MatchesSingleRank) {
  for (const auto& param : std::vector<DecoderCpCase>{
           DecoderCpCase{.moe = false,
                         .count = 16,
                         .can_rs = true,
                         .name = "DenseBalanced"},
           DecoderCpCase{.moe = false,
                         .count = 15,
                         .can_rs = false,
                         .name = "DenseUnbalanced"},
           DecoderCpCase{
               .moe = true, .count = 16, .can_rs = true, .name = "MoeBalanced"},
           DecoderCpCase{.moe = true,
                         .count = 15,
                         .can_rs = false,
                         .name = "MoeUnbalanced"}}) {
    // Two physical cards, including equal and unequal token counts per CP rank.
    const DecoderCpCase config = param;
    const int32_t port = parallel_base_port(/*world_size=*/2, /*cp_size=*/2);
    LOG(INFO) << config.name << " base_port=" << port;

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
          const auto rank_body = [&](int32_t rank) {
            run_cp_decoder(rank, port, config.moe, config.count, config.can_rs);
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
          child_pids, std::chrono::milliseconds{180000}, children_skipped);
    }

    if (children_ok && children_skipped) {
      GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                      "physical MLU cards";
    }
    EXPECT_EQ(children_ok, true)
        << "forked ranks failed, timed out, or require more physical MLU cards";
  }
}

ModelInputParams dp_params(int32_t rank, const torch::TensorOptions& options) {
  const auto ints = options.dtype(torch::kInt32);
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::PREFILL;
  params.meta.num_sequences = rank < 0 ? 2 : 1;
  params.meta.q_max_seq_len = rank == 0 ? 4 : 6;
  params.meta.kv_max_seq_len = params.meta.q_max_seq_len;
  const std::vector<int32_t> lengths =
      rank < 0 ? std::vector<int32_t>{4, 6}
               : std::vector<int32_t>{rank == 0 ? 4 : 6};
  const std::vector<int32_t> starts = rank < 0
                                          ? std::vector<int32_t>{0, 4, 10}
                                          : std::vector<int32_t>{0, lengths[0]};
  params.attention.host.q_seq_lens = lengths;
  params.attention.host.kv_seq_lens = lengths;
  params.attention.device.q_seq_lens = torch::tensor(starts, ints);
  params.attention.device.kv_seq_lens = params.attention.device.q_seq_lens;
  params.attention.device.block_tables =
      rank < 0 ? torch::tensor({{6}, {2}}, ints)
               : torch::tensor({{rank == 0 ? 6 : 2}}, ints);
  std::vector<int32_t> slots;
  slots.reserve(rank < 0 ? 10 : lengths[0]);
  for (int32_t sequence = rank < 0 ? 0 : rank;
       sequence < (rank < 0 ? 2 : rank + 1);
       ++sequence) {
    const int32_t count = sequence == 0 ? 4 : 6;
    const int32_t begin = sequence == 0 ? 96 : 32;
    for (int32_t offset = 0; offset < count; ++offset) {
      slots.emplace_back(begin + offset);
    }
  }
  params.attention.device.new_cache_slots = torch::tensor(slots, ints);
  params.parallel.dp_global_token_nums = {4, 6};
  return params;
}

void run_tp_dp_decoder(int32_t rank, int32_t port, bool dp) {
  torch::NoGradGuard no_grad;
  Device device(rank);
  device.set_device();
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
  KVCacheConfig::get_instance().block_size(kBlock);
  auto groups = create_cp_tp_groups(rank,
                                    /*world_size=*/2,
                                    /*cp_size=*/1,
                                    port,
                                    "127.0.0.1",
                                    device.unwrap());
  ParallelArgs parallel = *groups.parallel_args;
  if (dp) {
    parallel.dp_size() = 2;
    parallel.ep_size() = 2;
    parallel.tp_size() = 1;
    parallel.tp_group_ = groups.self.get();
    parallel.dp_local_process_group_ = groups.world.get();
    parallel.moe_ep_group_ = groups.world.get();
    parallel.moe_tp_group_ = groups.self.get();
  }
  const ParallelArgs single = self_parallel(groups.self.get());
  ModelArgs args = prefill_args(DsaModelVariant::GLM5);
  args.index_topk() = 64;
  StateDict weights(
      decoder_weights(args, /*moe=*/true, /*layer_id=*/1, options));
  DeepseekV32DecoderLayer actual(
      ModelContext(parallel, args, QuantArgs(), options), /*layer_id=*/1);
  DeepseekV32DecoderLayer reference(
      ModelContext(single, args, QuantArgs(), options), /*layer_id=*/1);
  actual->load_state_dict(weights);
  reference->load_state_dict(weights);
  actual->verify_loaded_weights();
  reference->verify_loaded_weights();
  const auto params =
      dp ? dp_params(rank, options)
         : phase_params(/*prefix=*/0, /*count=*/16, /*sharded=*/false, options);
  const auto reference_params = dp ? dp_params(/*rank=*/-1, options) : params;
  const auto metadata = AttentionMetadataBuilder::build(
      params, /*enable_mla=*/true, /*compute_dtype=*/"bfloat16");
  const auto reference_metadata = AttentionMetadataBuilder::build(
      reference_params, /*enable_mla=*/true, /*compute_dtype=*/"bfloat16");
  const int32_t total = dp ? 10 : 16;
  const int32_t begin = dp && rank == 1 ? 4 : 0;
  const int32_t count = dp ? (rank == 0 ? 4 : 6) : total;
  torch::Tensor full_input = (seeded_tensor("v32.parallel.dp",
                                            {total, args.hidden_size()},
                                            torch::kBFloat16,
                                            torch::Device(torch::kCPU)) *
                                  2.0 -
                              1.0)
                                 .to(options.device());
  torch::Tensor local_input =
      full_input.narrow(/*dim=*/0, begin, count).clone();
  const auto ints = options.dtype(torch::kInt32);
  torch::Tensor full_positions =
      dp ? torch::cat(
               {torch::arange(/*end=*/4, ints), torch::arange(/*end=*/6, ints)})
         : torch::arange(total, ints);
  torch::Tensor local_positions =
      full_positions.narrow(/*dim=*/0, begin, count).clone();
  KVCache cache = initial_cache(args, options, /*shard_rank=*/-1);
  KVCache reference_cache = initial_cache(args, options, /*shard_rank=*/-1);
  const torch::Tensor before = cache.get_k_cache().clone();
  std::optional<torch::Tensor> residual;
  std::optional<torch::Tensor> reference_residual;
  const torch::Tensor expected = reference->forward(full_input,
                                                    reference_residual,
                                                    full_positions,
                                                    reference_metadata,
                                                    reference_cache,
                                                    reference_params);
  const torch::Tensor output = actual->forward(
      local_input, residual, local_positions, metadata, cache, params);
  device.synchronize_default_stream();
  expect_close(output,
               expected.narrow(/*dim=*/0, begin, count),
               /*rtol=*/0.02,
               /*atol=*/0.002,
               dp ? "DP2 decoder" : "TP2 decoder");
  const torch::Tensor slots_cpu = params.attention.device.new_cache_slots.cpu();
  std::vector<int32_t> slots(
      slots_cpu.const_data_ptr<int32_t>(),
      slots_cpu.const_data_ptr<int32_t>() + slots_cpu.numel());
  check_cache(cache.get_k_cache(),
              reference_cache.get_k_cache(),
              before,
              slots,
              /*shard_rank=*/-1,
              "TP/DP decoder cache");
}

TEST(DecoderDpParallelTest, MatchesSingleRank) {
  for (const auto& param : std::vector<bool>{false, true}) {
    // Two physical cards. DP2 uses EP2 so the real DP gather/merge path
    // executes.
    const bool dp = param;
    const int32_t port = parallel_base_port(/*world_size=*/2, /*cp_size=*/1);
    LOG(INFO) << (dp ? "DP2" : "TP2") << " base_port=" << port
              << " tp_port=" << port + 1 << " cp_ports=" << port + 11 << ","
              << port + 12;

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
          const auto rank_body = [&](int32_t rank) {
            run_tp_dp_decoder(rank, port, dp);
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
          child_pids, std::chrono::milliseconds{180000}, children_skipped);
    }

    if (children_ok && children_skipped) {
      GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                      "physical MLU cards";
    }
    EXPECT_EQ(children_ok, true)
        << "forked ranks failed, timed out, or require more physical MLU cards";
  }
}

void run_cp_relay(int32_t rank, int32_t port) {
  torch::NoGradGuard no_grad;
  Device device(rank);
  device.set_device();
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
  KVCacheConfig::get_instance().block_size(kBlock);
  auto groups = create_cp_tp_groups(rank,
                                    /*world_size=*/2,
                                    /*cp_size=*/2,
                                    port,
                                    "127.0.0.1",
                                    device.unwrap());
  const ParallelArgs parallel = *groups.parallel_args;
  const ParallelArgs single = self_parallel(groups.self.get());
  ModelArgs args = prefill_args(DsaModelVariant::GLM5);
  args.n_layers() = 5;
  args.first_k_dense_replace() = 5;
  // The observing S makes the final F publish its refreshed selection.
  args.index_topk_pattern() = "FSSFS";
  const DsaTopkSharePlan plan(args);
  const auto params =
      phase_params(/*prefix=*/0, /*count=*/33, /*sharded=*/false, options);
  const auto metadata = AttentionMetadataBuilder::build(
      params, /*enable_mla=*/true, /*compute_dtype=*/"bfloat16");
  const auto ints = options.dtype(torch::kInt32);
  const torch::Tensor tokens = torch::arange(/*end=*/33, ints);
  const torch::Tensor positions = torch::arange(/*end=*/33, ints);
  auto cp = v32_cp::build_deepseek_v32_cp_context(/*cp_size=*/2,
                                                  metadata,
                                                  BatchForwardType::PREFILL,
                                                  tokens,
                                                  groups.cp.get(),
                                                  groups.cp->rank(),
                                                  groups.cp->world_size());
  CHECK(cp.has_value());
  torch::Tensor hidden = (seeded_tensor("v32.parallel.relay",
                                        {33, args.hidden_size()},
                                        torch::kBFloat16,
                                        torch::Device(torch::kCPU)) *
                              2.0 -
                          1.0)
                             .to(options.device());
  hidden = v32_cp::reorder_to_local_shard(hidden, cp.value());
  torch::Tensor local_positions =
      v32_cp::reorder_to_local_shard(positions, cp.value());
  DsaTopkRelay relay;
  torch::Tensor first_selection;

  for (int32_t layer_id = 0; layer_id < args.n_layers(); ++layer_id) {
    StateDict weights(decoder_weights(args, /*moe=*/false, layer_id, options));
    DeepseekV32DecoderLayer actual(
        ModelContext(parallel, args, QuantArgs(), options), layer_id, plan);
    actual->load_state_dict(weights);
    actual->verify_loaded_weights();
    actual->set_context_parallel_context(&cp.value());
    KVCache cache = initial_cache(args, options, /*shard_rank=*/-1);
    const torch::Tensor initial_index = cache.get_index_cache().clone();
    torch::Tensor expected;
    const bool shared = plan.decision_for(layer_id).reuse_topk;
    if (shared) {
      auto upstream =
          relay.prepare_layer(DsaTopkShareDecision{.reuse_topk = true});
      CHECK(upstream.has_value());
      CHECK(upstream->input() != nullptr);
      const torch::Tensor global_topk = v32_cp::gather_and_restore_global(
          upstream->input()->block_tables(), cp.value());
      const torch::Tensor global_lens = v32_cp::gather_and_restore_global(
          upstream->input()->context_lens(), cp.value());
      DsaTopkRelay reference_relay;
      auto fixed = DsaTopkTransfer::capture_output();
      fixed.publish_output(DsaTopkState(global_topk, global_lens));
      reference_relay.finish_layer(DsaTopkShareDecision{.output_topk = true},
                                   fixed);
      DeepseekV32DecoderLayer reference(
          ModelContext(single, args, QuantArgs(), options), layer_id, plan);
      reference->load_state_dict(weights);
      reference->verify_loaded_weights();
      KVCache reference_cache = initial_cache(args, options, /*shard_rank=*/-1);
      torch::Tensor full_input =
          v32_cp::gather_and_restore_global(hidden, cp.value());
      torch::Tensor full_positions = positions.clone();
      std::optional<torch::Tensor> reference_residual;
      expected = reference->forward(full_input,
                                    reference_residual,
                                    full_positions,
                                    metadata,
                                    reference_cache,
                                    params,
                                    /*input_ids=*/std::nullopt,
                                    &reference_relay);
    }
    std::optional<torch::Tensor> residual;
    hidden = actual->forward(hidden,
                             residual,
                             local_positions,
                             metadata,
                             cache,
                             params,
                             /*input_ids=*/std::nullopt,
                             &relay);
    const torch::Tensor restored =
        v32_cp::gather_and_restore_global(hidden, cp.value());
    if (shared) {
      expect_close(restored,
                   expected,
                   /*rtol=*/0.02,
                   /*atol=*/0.002,
                   "CP shared fixed-selection reference");
      EXPECT_TRUE(torch::equal(cache.get_index_cache(), initial_index));
    }
    auto current =
        relay.prepare_layer(DsaTopkShareDecision{.reuse_topk = true});
    CHECK(current.has_value());
    const torch::Tensor selected =
        std::get<0>(valid_selection(*current->input()).sort(/*dim=*/-1));
    if (layer_id == 0) {
      first_selection = selected.clone();
    } else if (layer_id < 3) {
      EXPECT_TRUE(torch::equal(selected, first_selection));
    } else if (layer_id == 3) {
      EXPECT_FALSE(torch::equal(selected, first_selection))
          << "Full layer must refresh a sensitive sparse fixture";
    }
    const std::string tag = "layer" + std::to_string(layer_id);
  }
  // Dummy shared layers skip relay preparation even with an empty relay.
  DeepseekV32DecoderLayer dummy(
      ModelContext(parallel, args, QuantArgs(), options), /*layer_id=*/1, plan);
  dummy->load_state_dict(
      StateDict(decoder_weights(args, /*moe=*/false, /*layer_id=*/1, options)));
  auto dummy_metadata = metadata;
  dummy_metadata.is_dummy = true;
  auto dummy_cp = v32_cp::build_deepseek_v32_cp_context(
      /*cp_size=*/2,
      dummy_metadata,
      BatchForwardType::PREFILL,
      tokens,
      groups.cp.get(),
      groups.cp->rank(),
      groups.cp->world_size());
  CHECK(dummy_cp.has_value());
  dummy->set_context_parallel_context(&dummy_cp.value());
  KVCache cache = initial_cache(args, options, /*shard_rank=*/-1);
  const torch::Tensor key_before = cache.get_k_cache().clone();
  const torch::Tensor index_before = cache.get_index_cache().clone();
  DsaTopkRelay empty;
  std::optional<torch::Tensor> residual;
  dummy->forward(hidden,
                 residual,
                 local_positions,
                 dummy_metadata,
                 cache,
                 params,
                 /*input_ids=*/std::nullopt,
                 &empty);
  EXPECT_TRUE(torch::equal(cache.get_k_cache(), key_before));
  EXPECT_TRUE(torch::equal(cache.get_index_cache(), index_before));
}

TEST(V32DecoderParallelTest, CpRelayReusesAndRefreshesFixedSelections) {
  // Two physical cards, with the required F -> S -> S -> F chain.
  const int32_t port = parallel_base_port(/*world_size=*/2, /*cp_size=*/2);
  LOG(INFO) << "rank test base_port=" << port;

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
        const auto rank_body = [&](int32_t rank) {
          run_cp_relay(rank, port);
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
      children_ok = false;
      break;
    }
    child_pids.emplace_back(pid);
  }
  if (!children_ok) {
    stop_ranks(child_pids, std::vector<bool>(child_pids.size(), false));
  } else {
    children_ok = wait_ranks(
        child_pids, std::chrono::milliseconds{180000}, children_skipped);
  }

  if (children_ok && children_skipped) {
    GTEST_SKIP()
        << "forked ranks failed, timed out, or require more physical MLU cards";
  }
  EXPECT_EQ(children_ok, true)
      << "forked ranks failed, timed out, or require more physical MLU cards";
}

void run_dummy_attention(int32_t rank, int32_t port) {
  torch::NoGradGuard no_grad;
  Device device(rank);
  device.set_device();
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
  KVCacheConfig::get_instance().block_size(kBlock);
  auto groups = create_cp_tp_groups(rank,
                                    /*world_size=*/2,
                                    /*cp_size=*/2,
                                    port,
                                    "127.0.0.1",
                                    device.unwrap());
  ParallelArgs parallel = *groups.parallel_args;
  parallel.kv_split_size() = 2;
  parallel.dcp_group_ = groups.world.get();
  for (DsaModelVariant variant :
       {DsaModelVariant::V32, DsaModelVariant::GLM5}) {
    const ModelArgs args = prefill_args(variant);
    for (bool shared : {false, true}) {
      DeepseekV32Attention attention(
          ModelContext(parallel, args, QuantArgs(), options),
          /*enable_indexer=*/!shared);
      attention->load_state_dict(StateDict(
          create_seeded_mla_weights(args, DsaWeightFormat::BF16, options)));
      const auto params =
          phase_params(/*prefix=*/0, /*count=*/33, /*sharded=*/true, options);
      auto metadata = AttentionMetadataBuilder::build(
          params, /*enable_mla=*/true, /*compute_dtype=*/"bfloat16");
      metadata.is_dummy = true;
      const KVShardLayout layout(kBlock, /*dcp_size=*/2, rank);
      const auto ints = options.dtype(torch::kInt32);
      const torch::Tensor tokens = torch::arange(/*end=*/33, ints);
      auto cp = v32_cp::build_deepseek_v32_cp_context(/*cp_size=*/2,
                                                      metadata,
                                                      BatchForwardType::PREFILL,
                                                      tokens,
                                                      groups.cp.get(),
                                                      rank,
                                                      /*world_size=*/2,
                                                      layout);
      CHECK(cp.has_value());
      torch::Tensor positions =
          v32_cp::reorder_to_local_shard(tokens, cp.value());
      torch::Tensor hidden = seeded_tensor("v32.parallel.dummy",
                                           {33, args.hidden_size()},
                                           torch::kBFloat16,
                                           options.device());
      hidden = v32_cp::reorder_to_local_shard(hidden, cp.value());
      KVCache cache = initial_cache(args, options, rank);
      const torch::Tensor key_before = cache.get_k_cache().clone();
      const torch::Tensor index_before = cache.get_index_cache().clone();
      auto transfer = DsaTopkTransfer::capture_output();
      const auto output = attention->forward(
          positions, hidden, metadata, cache, &cp.value(), &transfer);
      device.synchronize_default_stream();
      EXPECT_EQ(output.output.size(/*dim=*/0), hidden.size(/*dim=*/0));
      EXPECT_EQ(output.output.size(/*dim=*/1), args.hidden_size());
      EXPECT_EQ(transfer.output(), nullptr);
      EXPECT_EQ(transfer.localized_output(), nullptr);
      EXPECT_TRUE(torch::equal(cache.get_k_cache(), key_before));
      EXPECT_TRUE(torch::equal(cache.get_index_cache(), index_before));
    }
  }
}

TEST(V32AttentionParallelTest, CpDcpDummyPreservesCachesAndPublishesNothing) {
  // Two physical cards, both Full and Shared attention under CP2 + DCP2.
  const int32_t port = parallel_base_port(/*world_size=*/2, /*cp_size=*/2);
  LOG(INFO) << "rank test base_port=" << port;

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
        const auto rank_body = [&](int32_t rank) {
          run_dummy_attention(rank, port);
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
      children_ok = false;
      break;
    }
    child_pids.emplace_back(pid);
  }
  if (!children_ok) {
    stop_ranks(child_pids, std::vector<bool>(child_pids.size(), false));
  } else {
    children_ok = wait_ranks(
        child_pids, std::chrono::milliseconds{180000}, children_skipped);
  }

  if (children_ok && children_skipped) {
    GTEST_SKIP()
        << "forked ranks failed, timed out, or require more physical MLU cards";
  }
  EXPECT_EQ(children_ok, true)
      << "forked ranks failed, timed out, or require more physical MLU cards";
}

void run_row_quantization(int32_t rank, int32_t port) {
  torch::set_num_threads(1);
  torch::NoGradGuard no_grad;
  Device dev(rank);
  dev.set_device();
  const torch::Device device = dev.unwrap();
  auto groups = create_cp_tp_groups(rank,
                                    /*world_size=*/2,
                                    /*cp_size=*/1,
                                    port,
                                    "127.0.0.1",
                                    device);
  constexpr int64_t kInputs = 256;
  constexpr int64_t kOutputs = 128;
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  QuantArgs quant;
  quant.quant_method() = "smoothquant";
  quant.bits() = 8;
  quant.activation_dynamic() = true;
  RowParallelLinear full(kInputs,
                         kOutputs,
                         /*bias=*/false,
                         /*input_is_parallelized=*/true,
                         /*enable_result_reduction=*/false,
                         quant,
                         groups.self.get(),
                         options,
                         LinearExtraArgs(),
                         /*share_token_scale=*/true);
  RowParallelLinear sharded(kInputs,
                            kOutputs,
                            /*bias=*/false,
                            /*input_is_parallelized=*/true,
                            /*enable_result_reduction=*/false,
                            quant,
                            groups.tp.get(),
                            options,
                            LinearExtraArgs(),
                            /*share_token_scale=*/true);
  RowParallelLinear local(kInputs,
                          kOutputs,
                          /*bias=*/false,
                          /*input_is_parallelized=*/true,
                          /*enable_result_reduction=*/false,
                          quant,
                          groups.tp.get(),
                          options);
  torch::Tensor weight = torch::zeros({kOutputs, kInputs}, torch::kInt8);
  weight.index_put_({0, kInputs / 2}, 127);
  weight.index_put_({1, 0}, 1);
  StateDict weights(
      Weights{{"qweight", weight},
              {"per_channel_scale", torch::ones({kOutputs}, torch::kFloat32)},
              {"smooth", torch::ones({kInputs}, torch::kFloat32)}});
  full->load_state_dict(weights);
  sharded->load_state_dict(weights);
  local->load_state_dict(weights);
  ASSERT_TRUE(full->is_weight_loaded());
  ASSERT_TRUE(sharded->is_weight_loaded());
  torch::Tensor input = torch::zeros({4, kInputs}, torch::kBFloat16);
  input.index_put_({0, 0}, 1.0);
  input.index_put_({0, kInputs / 2}, 1.0 / 256.0);
  input.index_put_({1, 0}, 1.0 / 256.0);
  input.index_put_({1, kInputs / 2}, 1.0);
  input.index_put_({2, 0}, 1.0);
  input.index_put_({2, kInputs / 2}, 1.0);
  input = input.to(device);
  torch::Tensor expected = torch::zeros({4, kOutputs}, torch::kBFloat16);
  expected.index_put_({0, 1}, 1.0);
  expected.index_put_({1, 0}, 127.0);
  expected.index_put_({2, 0}, 127.0);
  expected.index_put_({2, 1}, 1.0);
  const torch::Tensor reference = full->forward(input);
  // Each output channel selects one input channel, so partial GEMM rounding
  // cannot hide a change in the per-token activation quantization domain.
  ASSERT_TRUE(torch::equal(reference.cpu(), expected));
  const torch::Tensor local_input =
      input.chunk(/*chunks=*/2, /*dim=*/1)[rank].contiguous();
  torch::Tensor local_expected = expected.clone();
  local_expected.index_put_({0, 0}, 127.0 / 256.0);
  local_expected.index_put_({1, 1}, 1.0 / 256.0);
  torch::Tensor local_output = local->forward(local_input);
  groups.tp->allreduce(local_output);
  EXPECT_TRUE(torch::equal(local_output.cpu(), local_expected));
  torch::Tensor actual = sharded->forward(local_input);
  groups.tp->allreduce(actual);
  dev.synchronize_default_stream();
  EXPECT_EQ(actual.scalar_type(), torch::kBFloat16);
  EXPECT_TRUE(torch::isfinite(actual).all().item<bool>());
  EXPECT_TRUE(torch::equal(actual.cpu(), expected))
      << "Row-parallel INT8 quantization changed the full-token scale domain: "
      << actual.cpu().slice(/*dim=*/1, /*start=*/0, /*end=*/2);
}

TEST(RowParallelLinearTest, PreservesGlobalTokenQuantization) {
  constexpr int32_t kRanks = 2;
  const int32_t port = parallel_base_port(kRanks, /*cp_size=*/1);
  const int32_t initial_parts = ::testing::UnitTest::GetInstance()
                                    ->current_test_info()
                                    ->result()
                                    ->total_part_count();
  std::vector<pid_t> pids;
  pids.reserve(kRanks);
  bool passed = true;
  bool skipped = false;
  for (int32_t rank = 0; rank < kRanks; ++rank) {
    const pid_t pid = ::fork();
    if (pid == 0) {
      try {
        if (Platform::device_count() < kRanks) {
          ::_exit(77);
        }
        run_row_quantization(rank, port);
        std::fflush(nullptr);
        ::_exit(child_failed(initial_parts) ? 1 : 0);
      } catch (const std::exception& error) {
        LOG(ERROR) << "Rank " << rank << ": " << error.what();
        ::_exit(1);
      }
    }
    if (pid < 0) {
      passed = false;
      break;
    }
    pids.emplace_back(pid);
  }
  if (passed) {
    passed = wait_ranks(pids, std::chrono::milliseconds{180000}, skipped);
  } else {
    stop_ranks(pids, std::vector<bool>(pids.size(), false));
  }
  if (passed && skipped) {
    GTEST_SKIP() << "Two physical MLU cards are required";
  }
  EXPECT_TRUE(passed) << "Row-parallel quantization ranks failed or timed out";
}

}  // namespace
}  // namespace xllm::layer::test
