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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <torch/torch.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

#include "core/framework/config/kv_cache_config.h"
#include "core/framework/model_context.h"
#include "core/layers/mlu/deepseek_v32/dcp_batch_metadata.h"
#include "framework/batch/batch_forward_type.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/context_parallel_topology.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/process_group.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/attention_metadata_builder.h"
#include "layers/mlu/deepseek_v32/dcp_decode_context.h"
#include "layers/mlu/deepseek_v32/deepseek_v32_attention.h"
#include "layers/mlu/deepseek_v32/deepseek_v32_cp_context.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "util/net.h"

namespace xllm {
namespace layer {
namespace test {

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

constexpr int32_t EXIT_CODE_SKIP = 77;

constexpr std::chrono::seconds kChildProcessTimeout{120};
constexpr std::chrono::seconds kChildTerminationGracePeriod{2};
constexpr std::chrono::milliseconds kChildPollInterval{10};

struct ChildWaitResult {
  bool any_failed = false;
  bool any_skipped = false;
  bool timed_out = false;
};

pid_t wait_for_pid(pid_t pid, int* status, int options) {
  pid_t result;
  do {
    result = ::waitpid(pid, status, options);
  } while (result < 0 && errno == EINTR);
  return result;
}

void record_child_status(size_t rank, int status, ChildWaitResult& result) {
  if (WIFEXITED(status)) {
    const int32_t exit_code = WEXITSTATUS(status);
    if (exit_code == EXIT_CODE_SKIP) {
      result.any_skipped = true;
    } else if (exit_code != 0) {
      result.any_failed = true;
      LOG(ERROR) << "Rank " << rank << " failed with code " << exit_code;
    }
    return;
  }

  result.any_failed = true;
  if (WIFSIGNALED(status)) {
    LOG(ERROR) << "Rank " << rank << " terminated by signal "
               << WTERMSIG(status);
  } else {
    LOG(ERROR) << "Rank " << rank << " ended unexpectedly";
  }
}

bool reap_children_until(
    const std::vector<pid_t>& child_pids,
    std::vector<bool>& finished,
    const std::chrono::steady_clock::time_point& deadline) {
  while (std::chrono::steady_clock::now() < deadline) {
    bool all_finished = true;
    for (size_t rank = 0; rank < child_pids.size(); ++rank) {
      if (finished[rank]) {
        continue;
      }

      int status = 0;
      const pid_t wait_result =
          wait_for_pid(child_pids[rank], &status, WNOHANG);
      if (wait_result == child_pids[rank] ||
          (wait_result < 0 && errno == ECHILD)) {
        finished[rank] = true;
        continue;
      }
      if (wait_result < 0) {
        LOG(ERROR) << "Failed to reap rank " << rank << ": "
                   << std::strerror(errno);
      }
      all_finished = false;
    }
    if (all_finished) {
      return true;
    }
    std::this_thread::sleep_for(kChildPollInterval);
  }
  return false;
}

void signal_unfinished_children(const std::vector<pid_t>& child_pids,
                                const std::vector<bool>& finished,
                                int signal_number) {
  for (size_t rank = 0; rank < child_pids.size(); ++rank) {
    if (finished[rank]) {
      continue;
    }
    if (::kill(child_pids[rank], signal_number) != 0 && errno != ESRCH) {
      LOG(ERROR) << "Failed to signal rank " << rank << ": "
                 << std::strerror(errno);
    }
  }
}

void terminate_and_reap_children(const std::vector<pid_t>& child_pids,
                                 std::vector<bool>& finished) {
  signal_unfinished_children(child_pids, finished, SIGTERM);
  const auto term_deadline =
      std::chrono::steady_clock::now() + kChildTerminationGracePeriod;
  if (reap_children_until(child_pids, finished, term_deadline)) {
    return;
  }

  signal_unfinished_children(child_pids, finished, SIGKILL);
  const auto kill_deadline =
      std::chrono::steady_clock::now() + kChildTerminationGracePeriod;
  reap_children_until(child_pids, finished, kill_deadline);
  for (size_t rank = 0; rank < child_pids.size(); ++rank) {
    if (!finished[rank]) {
      LOG(ERROR) << "Unable to reap timed-out rank " << rank;
    }
  }
}

ChildWaitResult wait_for_children(
    const std::vector<pid_t>& child_pids,
    std::chrono::steady_clock::duration timeout = kChildProcessTimeout) {
  ChildWaitResult result;
  std::vector<bool> finished(child_pids.size(), false);
  size_t remaining = child_pids.size();
  const auto deadline = std::chrono::steady_clock::now() + timeout;

  while (remaining > 0 && std::chrono::steady_clock::now() < deadline) {
    bool made_progress = false;
    for (size_t rank = 0; rank < child_pids.size(); ++rank) {
      if (finished[rank]) {
        continue;
      }

      int status = 0;
      const pid_t wait_result =
          wait_for_pid(child_pids[rank], &status, WNOHANG);
      if (wait_result == child_pids[rank]) {
        finished[rank] = true;
        --remaining;
        made_progress = true;
        record_child_status(rank, status, result);
      } else if (wait_result < 0) {
        result.any_failed = true;
        finished[rank] = true;
        --remaining;
        made_progress = true;
        LOG(ERROR) << "Failed waiting for rank " << rank << ": "
                   << std::strerror(errno);
      }
    }
    if (remaining > 0 && !made_progress) {
      std::this_thread::sleep_for(kChildPollInterval);
    }
  }

  if (remaining == 0) {
    return result;
  }

  result.any_failed = true;
  result.timed_out = true;
  const int64_t timeout_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(timeout).count();
  LOG(ERROR) << "Timed out waiting for " << remaining
             << " multi-device test rank(s) after " << timeout_ms << " ms";
  terminate_and_reap_children(child_pids, finished);
  return result;
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
ModelArgs create_glm5_attention_model_args();
StateDict create_attention_state_dict(const ModelArgs& args,
                                      const torch::TensorOptions& options);
AttentionMetadata create_decode_metadata(const torch::TensorOptions& options);
AttentionMetadata create_batch_decode_metadata(
    const torch::TensorOptions& options,
    int32_t batch_size);
AttentionMetadata create_prefill_metadata(const torch::TensorOptions& options,
                                          int32_t seq_len);
AttentionMetadata create_chunked_metadata(const torch::TensorOptions& options,
                                          int32_t prefix_len,
                                          int32_t chunk_len);
KVCache create_decode_kv_cache(const ModelArgs& args,
                               const torch::TensorOptions& options);
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

ModelArgs create_glm5_attention_model_args() {
  ModelArgs args =
      create_attention_model_args(/*q_lora_rank=*/64, /*enable_indexer=*/true);
  args.model_type() = "glm_moe_dsa";
  args.qk_nope_head_dim() = 192;
  args.v_head_dim() = 256;
  args.index_n_heads() = 32;
  args.rope_theta() = 1000000.0f;
  args.rope_scaling_rope_type() = "default";
  args.indexer_rope_interleave() = true;
  return args;
}

StateDict create_attention_state_dict(const ModelArgs& args,
                                      const torch::TensorOptions& options) {
  const int64_t hidden_size = args.hidden_size();
  const int64_t num_heads = args.n_heads();
  const int64_t kv_lora_rank = args.kv_lora_rank();
  const int64_t qk_nope_head_dim = args.qk_nope_head_dim();
  const int64_t qk_rope_head_dim = args.qk_rope_head_dim();
  const int64_t v_head_dim = args.v_head_dim();
  const int64_t qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
  const int64_t q_lora_rank = args.q_lora_rank();
  const int64_t index_n_heads = args.index_n_heads();
  const int64_t index_head_dim = args.index_head_dim();

  std::unordered_map<std::string, torch::Tensor> weights;
  if (q_lora_rank > 0) {
    weights["q_a_proj.weight"] = ::xllm::layer::test::seeded_tensor(
        "attention_multi_device/q_a_proj/weight",
        {q_lora_rank, hidden_size},
        torch::kBFloat16,
        options.device());
    weights["q_a_layernorm.weight"] =
        ::xllm::layer::test::seeded_tensor(
            "attention_multi_device/q_a_layernorm/weight",
            {q_lora_rank},
            torch::kFloat32,
            options.device())
            .abs()
            .add_(0.5f);
    weights["q_b_proj.qweight"] = ::xllm::layer::test::seeded_tensor(
        "attention_multi_device/q_b_proj/qweight",
        {num_heads * qk_head_dim, q_lora_rank},
        torch::kInt8,
        options.device());
    weights["q_b_proj.per_channel_scale"] =
        ::xllm::layer::test::seeded_tensor(
            "attention_multi_device/q_b_proj/per_channel_scale",
            {num_heads * qk_head_dim},
            torch::kFloat32,
            options.device())
            .abs()
            .add_(0.25f);
    weights["q_b_proj.smooth"] = ::xllm::layer::test::seeded_tensor(
                                     "attention_multi_device/q_b_proj/smooth",
                                     {q_lora_rank},
                                     torch::kFloat32,
                                     options.device())
                                     .abs()
                                     .add_(0.25f);
  } else {
    weights["q_proj.qweight"] = ::xllm::layer::test::seeded_tensor(
        "attention_multi_device/q_proj/qweight",
        {num_heads * qk_head_dim, hidden_size},
        torch::kInt8,
        options.device());
    weights["q_proj.per_channel_scale"] =
        ::xllm::layer::test::seeded_tensor(
            "attention_multi_device/q_proj/per_channel_scale",
            {num_heads * qk_head_dim},
            torch::kFloat32,
            options.device())
            .abs()
            .add_(0.25f);
    weights["q_proj.smooth"] = ::xllm::layer::test::seeded_tensor(
                                   "attention_multi_device/q_proj/smooth",
                                   {hidden_size},
                                   torch::kFloat32,
                                   options.device())
                                   .abs()
                                   .add_(0.25f);
  }

  weights["kv_a_proj_with_mqa.weight"] = ::xllm::layer::test::seeded_tensor(
      "attention_multi_device/kv_a_proj_with_mqa/weight",
      {kv_lora_rank + qk_rope_head_dim, hidden_size},
      torch::kBFloat16,
      options.device());
  weights["kv_a_layernorm.weight"] =
      ::xllm::layer::test::seeded_tensor(
          "attention_multi_device/kv_a_layernorm/weight",
          {kv_lora_rank},
          torch::kFloat32,
          options.device())
          .abs()
          .add_(0.5f);
  weights["kv_b_proj.weight"] = ::xllm::layer::test::seeded_tensor(
      "attention_multi_device/kv_b_proj/weight",
      {num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank},
      torch::kBFloat16,
      options.device());

  weights["o_proj.qweight"] = ::xllm::layer::test::seeded_tensor(
      "attention_multi_device/o_proj/qweight",
      {hidden_size, num_heads * v_head_dim},
      torch::kInt8,
      options.device());
  weights["o_proj.per_channel_scale"] =
      ::xllm::layer::test::seeded_tensor(
          "attention_multi_device/o_proj/per_channel_scale",
          {hidden_size},
          torch::kFloat32,
          options.device())
          .abs()
          .add_(0.25f);
  weights["o_proj.smooth"] =
      ::xllm::layer::test::seeded_tensor("attention_multi_device/o_proj/smooth",
                                         {num_heads * v_head_dim},
                                         torch::kFloat32,
                                         options.device())
          .abs()
          .add_(0.25f);

  if (index_n_heads > 0) {
    weights["indexer.k_norm.bias"] =
        ::xllm::layer::test::seeded_tensor(
            "attention_multi_device/indexer/k_norm/bias",
            {index_head_dim},
            torch::kFloat32,
            options.device())
            .abs()
            .add_(0.5f);
    weights["indexer.k_norm.weight"] =
        ::xllm::layer::test::seeded_tensor(
            "attention_multi_device/indexer/k_norm/weight",
            {index_head_dim},
            torch::kFloat32,
            options.device())
            .abs()
            .add_(0.5f);
    weights["indexer.weights_proj.weight"] = ::xllm::layer::test::seeded_tensor(
        "attention_multi_device/indexer/weights_proj/weight",
        {index_n_heads, hidden_size},
        torch::kBFloat16,
        options.device());
    weights["indexer.wk.weight"] = ::xllm::layer::test::seeded_tensor(
        "attention_multi_device/indexer/wk/weight",
        {index_head_dim, hidden_size},
        torch::kBFloat16,
        options.device());
    weights["indexer.wq_b.weight"] = ::xllm::layer::test::seeded_tensor(
        "attention_multi_device/indexer/wq_b/weight",
        {index_n_heads * index_head_dim, q_lora_rank},
        torch::kBFloat16,
        options.device());
  }

  return StateDict(std::move(weights));
}

AttentionMetadata create_decode_metadata(const torch::TensorOptions& options) {
  AttentionMetadata metadata;
  auto int_options = options.dtype(torch::kInt32);
  metadata.q_cu_seq_lens = torch::tensor({0, 1}, int_options);
  metadata.kv_cu_seq_lens = torch::tensor({0, 0}, int_options);
  metadata.kv_seq_lens = torch::tensor({1}, int_options);
  metadata.block_table = torch::zeros({1, 4}, int_options);
  metadata.block_table[0][0] = 1;
  metadata.slot_mapping = torch::tensor({1}, int_options);
  metadata.max_query_len = 1;
  metadata.max_seq_len = 1;
  metadata.total_kv_len = 1;
  metadata.compute_dtype = "half";
  metadata.is_prefill = false;
  metadata.is_chunked_prefill = false;
  metadata.is_dummy = false;
  return metadata;
}

AttentionMetadata create_batch_decode_metadata(
    const torch::TensorOptions& options,
    int32_t batch_size) {
  AttentionMetadata metadata;
  auto int_options = options.dtype(torch::kInt32);
  metadata.q_cu_seq_lens = torch::arange(batch_size + 1, int_options);
  metadata.kv_cu_seq_lens = torch::zeros({batch_size + 1}, int_options);
  metadata.kv_seq_lens = torch::ones({batch_size}, int_options);
  metadata.block_table = torch::zeros({batch_size, 4}, int_options);
  for (int32_t i = 0; i < batch_size; ++i) {
    metadata.block_table[i][0] = i + 1;
  }
  metadata.slot_mapping = torch::arange(1, batch_size + 1, int_options);
  metadata.max_query_len = 1;
  metadata.max_seq_len = 1;
  metadata.total_kv_len = batch_size;
  metadata.compute_dtype = "half";
  metadata.is_prefill = false;
  metadata.is_chunked_prefill = false;
  metadata.is_dummy = false;
  return metadata;
}

AttentionMetadata create_prefill_metadata(const torch::TensorOptions& options,
                                          int32_t seq_len) {
  AttentionMetadata metadata;
  auto int_options = options.dtype(torch::kInt32);
  metadata.q_cu_seq_lens = torch::tensor({0, seq_len}, int_options);
  metadata.kv_cu_seq_lens = torch::tensor({0, seq_len}, int_options);
  metadata.kv_seq_lens = torch::tensor({seq_len}, int_options);
  metadata.block_table = torch::zeros({1, 4}, int_options);
  // slot_mapping starts at zero, so the first physical block is zero.
  metadata.block_table[0][0] = 0;
  metadata.slot_mapping = torch::arange(seq_len, int_options);
  metadata.max_query_len = seq_len;
  metadata.max_seq_len = seq_len;
  metadata.total_kv_len = seq_len;
  metadata.compute_dtype = "half";
  metadata.is_prefill = true;
  metadata.is_chunked_prefill = false;
  metadata.is_dummy = false;
  return metadata;
}

AttentionMetadata create_chunked_metadata(const torch::TensorOptions& options,
                                          int32_t prefix_len,
                                          int32_t chunk_len) {
  AttentionMetadata metadata;
  auto int_options = options.dtype(torch::kInt32);
  const int32_t ctx_len = prefix_len + chunk_len;
  metadata.q_cu_seq_lens = torch::tensor({0, chunk_len}, int_options);
  metadata.kv_cu_seq_lens = torch::tensor({0, ctx_len}, int_options);
  metadata.kv_seq_lens = torch::tensor({ctx_len}, int_options);
  metadata.block_table = torch::zeros({1, 4}, int_options);
  metadata.block_table[0][0] = 1;
  metadata.slot_mapping = torch::arange(prefix_len, ctx_len, int_options);
  metadata.max_query_len = chunk_len;
  metadata.max_seq_len = ctx_len;
  metadata.total_kv_len = ctx_len;
  metadata.compute_dtype = "half";
  metadata.is_prefill = false;
  metadata.is_chunked_prefill = true;
  metadata.is_dummy = false;
  return metadata;
}

KVCache create_decode_kv_cache(const ModelArgs& args,
                               const torch::TensorOptions& options) {
  const int64_t cache_width = args.qk_rope_head_dim() + args.kv_lora_rank();
  torch::Tensor k_cache = ::xllm::layer::test::seeded_tensor(
      "attention_multi_device/k_cache",
      {8, 1, KVCacheConfig::get_instance().block_size(), cache_width},
      torch::kBFloat16,
      options.device());
  torch::Tensor index_cache = torch::Tensor();
  if (args.index_head_dim() > 0) {
    index_cache = ::xllm::layer::test::seeded_tensor(
        "attention_multi_device/index_cache",
        {8,
         1,
         KVCacheConfig::get_instance().block_size(),
         args.index_head_dim()},
        torch::kBFloat16,
        options.device());
  }
  return create_indexed_kv_cache(std::move(k_cache), std::move(index_cache));
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

class DistributedAttentionChildContext final {
 public:
  DistributedAttentionChildContext(int32_t rank,
                                   int32_t world_size,
                                   int32_t port,
                                   const std::string& host,
                                   bool enable_context_parallel)
      : xllm_device_(rank),
        device_(xllm_device_.unwrap()),
        options_(torch::TensorOptions()
                     .dtype(torch::kBFloat16)
                     .device(device_)
                     .requires_grad(false)) {
    xllm_device_.set_device();
    process_group_ =
        create_test_process_group(rank, world_size, port, host, device_);
    self_group_ = create_test_self_group(rank, world_size, host, device_);
    CHECK(process_group_) << "Rank " << rank
                          << ": failed to create process group";
    CHECK(self_group_) << "Rank " << rank << ": failed to create self group";

    parallel_args_ =
        std::make_unique<ParallelArgs>(rank, world_size, process_group_.get());
    // Keep the CP output-baseline tests on the supported replicated-KV path.
    // Dedicated DCP tests must opt into KV sharding explicitly.
    parallel_args_->kv_split_size() = 1;
    parallel_args_->tp_group_ = process_group_.get();
    parallel_args_->single_rank_group_ = self_group_.get();
    parallel_args_->cp_group_ = process_group_.get();
    if (enable_context_parallel) {
      parallel_args_->cp_size() = world_size;
    }
  }

  xllm::Device& xllm_device() { return xllm_device_; }

  const torch::Device& device() const { return device_; }

  ParallelArgs& parallel_args() { return *parallel_args_; }

  const torch::TensorOptions& options() const { return options_; }

  xllm::ProcessGroup* process_group() const { return process_group_.get(); }

 private:
  xllm::Device xllm_device_;
  torch::Device device_;
  std::unique_ptr<xllm::ProcessGroup> process_group_;
  std::unique_ptr<xllm::ProcessGroup> self_group_;
  std::unique_ptr<ParallelArgs> parallel_args_;
  torch::TensorOptions options_;
};

template <typename ChildBody>
int32_t run_distributed_attention_child(int32_t rank,
                                        int32_t world_size,
                                        int32_t port,
                                        const std::string& host,
                                        bool enable_context_parallel,
                                        const ChildBody& child_body) {
  try {
    const int32_t device_count = xllm::Platform::device_count();
    if (device_count < world_size) {
      LOG(WARNING) << "Rank " << rank << ": insufficient devices. Skipping.";
      return EXIT_CODE_SKIP;
    }

    KVCacheConfig::get_instance().block_size(16);
    DistributedAttentionChildContext context(
        rank, world_size, port, host, enable_context_parallel);
    return child_body(context);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Rank " << rank << ": Exception: " << e.what();
    return 1;
  }
}

void check_tensors_close(const torch::Tensor& actual,
                         const torch::Tensor& expected,
                         double rtol,
                         double atol,
                         const std::string& label) {
  CHECK_EQ(actual.sizes(), expected.sizes())
      << label << " shape mismatch: actual=" << actual.sizes()
      << ", expected=" << expected.sizes();
  torch::Tensor actual_fp32 = actual.to(torch::kFloat32);
  torch::Tensor expected_fp32 = expected.to(torch::kFloat32);
  torch::Tensor diff = torch::abs(actual_fp32 - expected_fp32);
  CHECK(torch::allclose(actual_fp32, expected_fp32, rtol, atol))
      << label << " mismatch. max_diff=" << torch::max(diff).item<float>()
      << ", mean_diff=" << torch::mean(diff).item<float>();
}

struct AttentionRunResult {
  torch::Tensor local_output;
  torch::Tensor global_output;
  torch::Tensor k_cache;
  torch::Tensor index_cache;
  PostAttnLayout layout = PostAttnLayout::TP_SHARD;
  bool used_cp = false;
  int64_t local_token_num = 0;
};

void check_repl_output_contract(const AttentionRunResult& result,
                                int64_t token_num,
                                int64_t hidden_size,
                                const std::string& label) {
  CHECK(!result.used_cp) << label << " unexpectedly used context parallel";
  CHECK(result.layout == PostAttnLayout::REPLICATED)
      << label << " must report replicated attention output";
  CHECK_EQ(result.local_token_num, token_num)
      << label << " local token count mismatch";
  CHECK_EQ(result.local_output.sizes(), result.global_output.sizes())
      << label << " local/global output shape mismatch";
  CHECK_EQ(result.local_output.size(0), token_num)
      << label << " output token count mismatch";
  CHECK_EQ(result.local_output.size(1), hidden_size)
      << label << " output hidden size mismatch";
  check_tensors_close(result.local_output,
                      result.global_output,
                      /*rtol=*/0.0,
                      /*atol=*/0.0,
                      label + " local/global");
}

void check_tensor_same_across_ranks(const torch::Tensor& tensor,
                                    xllm::ProcessGroup* process_group,
                                    const std::string& label) {
  if (process_group == nullptr || process_group->world_size() <= 1) {
    return;
  }

  torch::Tensor gathered =
      xllm::parallel_state::gather(tensor, process_group, /*dim=*/0);
  const int64_t rows = tensor.size(0);
  torch::Tensor ref = gathered.slice(0, 0, rows);
  for (int64_t rank = 1; rank < process_group->world_size(); ++rank) {
    torch::Tensor curr = gathered.slice(0, rank * rows, (rank + 1) * rows);
    check_tensors_close(curr,
                        ref,
                        /*rtol=*/0.0,
                        /*atol=*/0.0,
                        label + " rank " + std::to_string(rank));
  }
}

torch::Tensor run_attention_decode_once(const ModelArgs& args,
                                        const QuantArgs& quant_args,
                                        const ParallelArgs& parallel_args,
                                        const torch::TensorOptions& options,
                                        const StateDict& state_dict,
                                        const torch::Tensor& positions,
                                        const torch::Tensor& hidden_states,
                                        KVCache& kv_cache,
                                        bool enable_full_weight_path,
                                        bool enable_fused_mla_kernel) {
  ParallelArgs effective_parallel_args = parallel_args;
  effective_parallel_args.cp_size() =
      enable_full_weight_path ? parallel_args.world_size() : 1;
  OptimizationConfig optimization_config;
  optimization_config.enable_fused_mla_kernel = enable_fused_mla_kernel;
  optimization_config.enable_fused_indexer_qk = false;
  DeepseekV32Attention attention(
      args, quant_args, effective_parallel_args, options, optimization_config);
  attention->load_state_dict(state_dict);
  const auto ints = options.dtype(torch::kInt32);
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.meta.num_sequences = 1;
  params.meta.q_max_seq_len = 1;
  params.meta.kv_max_seq_len = 1;
  params.attention.host.q_seq_lens = {1};
  params.attention.host.kv_seq_lens = {1};
  params.attention.device.q_seq_lens = torch::tensor({0, 1}, ints);
  params.attention.device.kv_seq_lens = torch::tensor({0, 1}, ints);
  params.attention.device.block_tables = torch::tensor({{1}}, ints);
  params.attention.device.new_cache_slots = torch::tensor(
      {static_cast<int32_t>(KVCacheConfig::get_instance().block_size())}, ints);
  const auto metadata = AttentionMetadataBuilder::build(
      params, /*enable_mla=*/true, /*compute_dtype=*/"bfloat16");
  return attention->forward(positions, hidden_states, metadata, kv_cache)
      .output;
}

AttentionRunResult run_attention_prefill_once(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options,
    const StateDict& state_dict,
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    KVCache& kv_cache,
    bool enable_full_weight_path,
    BatchForwardType batch_forward_type = BatchForwardType::PREFILL,
    int32_t prefix_len = 0,
    bool build_cp_context = true,
    DsaTopkTransfer* topk_transfer = nullptr,
    bool enable_indexer = true,
    std::optional<layer::v32_cp::DeepseekV32CPContext>* cp_ctx_cache =
        nullptr) {
  ParallelArgs effective_parallel_args = parallel_args;
  effective_parallel_args.cp_size() =
      enable_full_weight_path ? parallel_args.world_size() : 1;
  ModelContext model_context(
      effective_parallel_args, args, quant_args, options);
  DeepseekV32Attention attention(model_context, enable_indexer);
  attention->load_state_dict(state_dict);
  const int32_t token_num = static_cast<int32_t>(tokens.numel());
  AttentionMetadata metadata =
      batch_forward_type.is_chunked_prefill()
          ? create_chunked_metadata(options, prefix_len, token_num)
          : create_prefill_metadata(options, token_num);
  std::optional<layer::v32_cp::DeepseekV32CPContext> cp_ctx;
  torch::Tensor local_positions = positions;
  torch::Tensor local_hidden_states = hidden_states;
  if (build_cp_context && enable_full_weight_path && args.index_n_heads() > 0) {
    if (cp_ctx_cache != nullptr && cp_ctx_cache->has_value()) {
      cp_ctx = *cp_ctx_cache;
    } else {
      ProcessGroup* cp_group = parallel_args.cp_group_ != nullptr
                                   ? parallel_args.cp_group_
                                   : parallel_args.process_group_;
      std::optional<KVShardLayout> kv_shard_layout;
      if (effective_parallel_args.kv_split_size_effective() > 1) {
        kv_shard_layout.emplace(
            KVCacheConfig::get_instance().block_size(),
            effective_parallel_args.kv_split_size_effective(),
            effective_parallel_args.kv_split_rank());
      }
      cp_ctx = layer::v32_cp::build_deepseek_v32_cp_context(
          effective_parallel_args.cp_size(),
          metadata,
          batch_forward_type,
          tokens,
          cp_group,
          parallel_args.rank(),
          parallel_args.world_size(),
          kv_shard_layout);
      if (cp_ctx_cache != nullptr) {
        *cp_ctx_cache = cp_ctx;
      }
    }
    if (cp_ctx.has_value()) {
      local_positions =
          layer::v32_cp::reorder_to_local_shard(positions, cp_ctx.value());
      local_hidden_states =
          layer::v32_cp::reorder_to_local_shard(hidden_states, cp_ctx.value());
    }
  }
  AttentionRunResult result;
  auto attention_result = attention->forward(local_positions,
                                             local_hidden_states,
                                             metadata,
                                             kv_cache,
                                             cp_ctx ? &cp_ctx.value() : nullptr,
                                             topk_transfer);
  result.local_output = std::move(attention_result.output);
  result.layout = attention_result.layout;
  result.global_output = result.local_output;
  result.used_cp = cp_ctx.has_value();
  result.local_token_num =
      cp_ctx.has_value() ? cp_ctx->comm_plan.tokens_per_rank.at(cp_ctx->rank)
                         : result.local_output.size(0);
  if (cp_ctx.has_value()) {
    result.global_output = layer::v32_cp::restore_gathered_to_global_order(
        layer::v32_cp::all_gather_across_ranks(result.local_output,
                                               cp_ctx.value()),
        cp_ctx.value());
  }
  result.k_cache = kv_cache.get_k_cache().clone();
  if (kv_cache.get_index_cache().defined()) {
    result.index_cache = kv_cache.get_index_cache().clone();
  }
  return result;
}

int32_t run_glm_decode_child(int32_t rank,
                             int32_t world_size,
                             int32_t port,
                             const std::string& host,
                             bool enable_fused_mla_kernel) {
  return run_distributed_attention_child(
      rank,
      world_size,
      port,
      host,
      /*enable_context_parallel=*/false,
      [enable_fused_mla_kernel](DistributedAttentionChildContext& context) {
        xllm::Device& xllm_device = context.xllm_device();
        const torch::Device& device = context.device();
        ParallelArgs& parallel_args = context.parallel_args();
        const torch::TensorOptions& options = context.options();
        ModelArgs model_args = create_glm5_attention_model_args();
        QuantArgs quant_args = create_default_quant_args();
        StateDict state_dict = create_attention_state_dict(model_args, options);

        torch::Tensor hidden_states =
            seeded_tensor("attention_multi_device/hidden_states",
                          {1, model_args.hidden_size()},
                          torch::kBFloat16,
                          device);
        torch::Tensor positions =
            torch::tensor({0}, options.dtype(torch::kInt32).device(device));

        KVCache full_weight_kv_cache =
            create_decode_kv_cache(model_args, options);
        const torch::Tensor initial_key =
            full_weight_kv_cache.get_k_cache().clone();
        const torch::Tensor initial_index =
            full_weight_kv_cache.get_index_cache().clone();
        KVCache reference_cache =
            create_indexed_kv_cache(initial_key.clone(), initial_index.clone());
        ParallelArgs single(
            /*rank=*/0, /*world_size=*/1, parallel_args.single_rank_group_);
        single.tp_group_ = parallel_args.single_rank_group_;
        single.cp_group_ = parallel_args.single_rank_group_;
        single.dcp_group_ = parallel_args.single_rank_group_;
        single.single_rank_group_ = parallel_args.single_rank_group_;
        single.kv_split_size() = 1;
        const torch::Tensor expected =
            run_attention_decode_once(model_args,
                                      quant_args,
                                      single,
                                      options,
                                      state_dict,
                                      positions,
                                      hidden_states,
                                      reference_cache,
                                      /*enable_full_weight_path=*/true,
                                      enable_fused_mla_kernel);

        torch::Tensor full_weight_output =
            run_attention_decode_once(model_args,
                                      quant_args,
                                      parallel_args,
                                      options,
                                      state_dict,
                                      positions,
                                      hidden_states,
                                      full_weight_kv_cache,
                                      /*enable_full_weight_path=*/true,
                                      enable_fused_mla_kernel);

        xllm_device.synchronize_default_stream();
        CHECK_EQ(full_weight_output.size(0), 1)
            << "DeepseekV32Attention decode replicated token count mismatch";
        CHECK_EQ(full_weight_output.size(1), model_args.hidden_size())
            << "DeepseekV32Attention decode replicated hidden size mismatch";
        check_tensor_same_across_ranks(
            full_weight_output,
            parallel_args.tp_group_,
            "DeepseekV32Attention decode replicated");
        check_tensors_close(full_weight_output,
                            expected,
                            /*rtol=*/0.0,
                            /*atol=*/0.0,
                            "GLM5 full-weight decode vs single-rank reference");
        CHECK(torch::equal(full_weight_kv_cache.get_k_cache(),
                           reference_cache.get_k_cache()));
        CHECK(torch::equal(full_weight_kv_cache.get_index_cache(),
                           reference_cache.get_index_cache()));
        const int64_t slot = KVCacheConfig::get_instance().block_size();
        const torch::Tensor key_rows =
            full_weight_kv_cache.get_k_cache().cpu().reshape(
                {-1,
                 model_args.kv_lora_rank() + model_args.qk_rope_head_dim()});
        const torch::Tensor index_rows =
            full_weight_kv_cache.get_index_cache().cpu().reshape(
                {-1, model_args.index_head_dim()});
        torch::Tensor untouched =
            torch::ones({key_rows.size(/*dim=*/0)}, torch::kBool);
        untouched[slot] = false;
        CHECK(torch::equal(
            key_rows.index({untouched}),
            initial_key.cpu().reshape_as(key_rows).index({untouched})));
        CHECK(torch::equal(
            index_rows.index({untouched}),
            initial_index.cpu().reshape_as(index_rows).index({untouched})));
        const std::string name = enable_fused_mla_kernel
                                     ? "Glm5FusedDecodeFullWeights"
                                     : "Glm5DecodeFullWeights";

        return 0;
      });
}

int32_t run_attention_prefill_fallback_test_child(int32_t rank,
                                                  int32_t world_size,
                                                  int32_t port,
                                                  const std::string& host) {
  return run_distributed_attention_child(
      rank,
      world_size,
      port,
      host,
      /*enable_context_parallel=*/false,
      [](DistributedAttentionChildContext& context) {
        xllm::Device& xllm_device = context.xllm_device();
        const torch::Device& device = context.device();
        ParallelArgs& parallel_args = context.parallel_args();
        const torch::TensorOptions& options = context.options();
        ModelArgs model_args =
            create_attention_model_args(/*q_lora_rank=*/64,
                                        /*enable_indexer=*/true);
        QuantArgs quant_args = create_default_quant_args();
        StateDict state_dict = create_attention_state_dict(model_args, options);

        constexpr int32_t seq_len = 1;
        torch::Tensor tokens =
            torch::arange(seq_len, options.dtype(torch::kInt32).device(device));
        torch::Tensor hidden_states = seeded_tensor(
            "attention_multi_device/prefill_hidden_states_fallback",
            {seq_len, model_args.hidden_size()},
            torch::kBFloat16,
            device);
        torch::Tensor positions =
            torch::arange(seq_len, options.dtype(torch::kInt32).device(device));

        KVCache full_weight_kv_cache =
            create_decode_kv_cache(model_args, options);
        full_weight_kv_cache.get_k_cache().zero_();

        auto repl_result = run_attention_prefill_once(model_args,
                                                      quant_args,
                                                      parallel_args,
                                                      options,
                                                      state_dict,
                                                      tokens,
                                                      positions,
                                                      hidden_states,
                                                      full_weight_kv_cache,
                                                      true);

        xllm_device.synchronize_default_stream();
        check_repl_output_contract(repl_result,
                                   seq_len,
                                   model_args.hidden_size(),
                                   "DeepseekV32Attention prefill fallback");
        check_tensor_same_across_ranks(
            repl_result.global_output,
            parallel_args.tp_group_,
            "DeepseekV32Attention prefill fallback output");
        return 0;
      });
}

int32_t run_prefill_share_child(int32_t rank,
                                int32_t world_size,
                                int32_t port,
                                const std::string& host,
                                bool missing_slot = false,
                                bool use_cp = true) {
  bool checking_missing_slot = false;
  try {
    const int32_t dev_count = xllm::Platform::device_count();
    if (dev_count < world_size) {
      LOG(WARNING) << "Rank " << rank << ": insufficient devices. Skipping.";
      return EXIT_CODE_SKIP;
    }

    KVCacheConfig::get_instance().block_size(16);
    const int32_t device_index = rank;
    xllm::Device xllm_device(device_index);
    xllm_device.set_device();
    torch::Device device = xllm_device.unwrap();
    auto process_group =
        create_test_process_group(rank, world_size, port, host, device);
    CHECK(process_group) << "Rank " << rank
                         << ": failed to create process group";
    auto self_group = create_test_self_group(rank, world_size, host, device);
    CHECK(self_group) << "Rank " << rank << ": failed to create self group";

    ParallelArgs parallel_args(rank, world_size, process_group.get());
    // This GLM-5.2 case validates rank-local top-k reuse with DCP KV shards.
    parallel_args.kv_split_size() = world_size;
    parallel_args.dcp_group_ = process_group.get();
    parallel_args.tp_group_ = process_group.get();
    parallel_args.single_rank_group_ = self_group.get();
    parallel_args.cp_group_ = process_group.get();

    auto options = torch::TensorOptions()
                       .dtype(torch::kBFloat16)
                       .device(device)
                       .requires_grad(false);
    ModelArgs model_args = create_glm5_attention_model_args();
    QuantArgs quant_args = create_default_quant_args();
    StateDict state_dict = create_attention_state_dict(model_args, options);

    constexpr int32_t kSeqLen = 4;
    torch::Tensor tokens =
        torch::arange(kSeqLen, options.dtype(torch::kInt32).device(device));
    torch::Tensor hidden_states = seeded_tensor(
        "attention_multi_device/glm52_cp_topk_share_hidden_states",
        {kSeqLen, model_args.hidden_size()},
        torch::kBFloat16,
        device);
    torch::Tensor positions =
        torch::arange(kSeqLen, options.dtype(torch::kInt32).device(device));

    std::optional<layer::v32_cp::DeepseekV32CPContext> cp_ctx_cache;
    KVCache full_layer_cache = create_decode_kv_cache(model_args, options);
    full_layer_cache.get_k_cache().zero_();
    DsaTopkTransfer full_transfer = DsaTopkTransfer::capture_output();
    AttentionRunResult full_result =
        run_attention_prefill_once(model_args,
                                   quant_args,
                                   parallel_args,
                                   options,
                                   state_dict,
                                   tokens,
                                   positions,
                                   hidden_states,
                                   full_layer_cache,
                                   /*enable_full_weight_path=*/true,
                                   BatchForwardType::PREFILL,
                                   /*prefix_len=*/0,
                                   /*build_cp_context=*/use_cp,
                                   &full_transfer,
                                   /*enable_indexer=*/true,
                                   &cp_ctx_cache);
    xllm_device.synchronize_default_stream();

    const DsaTopkState* full_topk = full_transfer.output();
    CHECK(full_topk != nullptr)
        << "GLM5.2 PCP Full layer must publish rank-local DSA top-k state";
    CHECK_EQ(full_topk->block_tables().size(0),
             use_cp ? kSeqLen / world_size : kSeqLen)
        << "GLM5.2 PCP top-k rows must match rank-local query rows";
    CHECK_EQ(full_topk->block_tables().size(1), model_args.index_topk())
        << "GLM5.2 PCP top-k width mismatch";
    CHECK(torch::isfinite(full_result.local_output.to(torch::kFloat32))
              .all()
              .item<bool>())
        << "GLM5.2 PCP Full layer output must be finite";

    KVCache shared_layer_cache = create_decode_kv_cache(model_args, options);
    shared_layer_cache.get_k_cache().zero_();
    DsaTopkState shared_topk(
        missing_slot ? torch::full_like(full_topk->block_tables(), kSeqLen + 1)
                     : full_topk->block_tables(),
        full_topk->context_lens());
    std::optional<DsaPrefillRemap> shared_remap;
    if (use_cp && !missing_slot) {
      const DsaPrefillRemap* full_remap = full_transfer.prefill_remap_output();
      CHECK(full_remap != nullptr);
      shared_remap.emplace(full_remap->block_table(),
                           shared_topk.block_tables(),
                           shared_topk.context_lens(),
                           cp_ctx_cache->sorted_gathered_slot_mapping_int64,
                           cp_ctx_cache->sorted_gathered_slot_rows);
    }
    DsaTopkTransfer shared_transfer =
        DsaTopkTransfer::reuse_and_capture(shared_topk, shared_remap);
    checking_missing_slot = missing_slot;
    AttentionRunResult shared_result =
        run_attention_prefill_once(model_args,
                                   quant_args,
                                   parallel_args,
                                   options,
                                   state_dict,
                                   tokens,
                                   positions,
                                   hidden_states,
                                   shared_layer_cache,
                                   /*enable_full_weight_path=*/true,
                                   BatchForwardType::PREFILL,
                                   /*prefix_len=*/0,
                                   /*build_cp_context=*/use_cp,
                                   &shared_transfer,
                                   /*enable_indexer=*/missing_slot && use_cp,
                                   &cp_ctx_cache);
    xllm_device.synchronize_default_stream();

    if (missing_slot) {
      LOG(ERROR) << "Prefill accepted a top-k slot absent from the input K";
      return 1;
    }
    const DsaTopkState* reused_topk = shared_transfer.output();
    CHECK(reused_topk != nullptr)
        << "GLM5.2 PCP Shared layer must publish the state it consumed";
    CHECK_EQ(reused_topk->block_tables().data_ptr(),
             full_topk->block_tables().data_ptr())
        << "GLM5.2 PCP Shared layer must reuse the Full layer block table "
           "storage";
    CHECK_EQ(reused_topk->context_lens().data_ptr(),
             full_topk->context_lens().data_ptr())
        << "GLM5.2 PCP Shared layer must reuse the Full layer context lens "
           "storage";
    CHECK(torch::isfinite(shared_result.local_output.to(torch::kFloat32))
              .all()
              .item<bool>())
        << "GLM5.2 PCP Shared layer output must be finite";
    check_tensors_close(shared_result.global_output,
                        full_result.global_output,
                        /*rtol=*/0.0,
                        /*atol=*/0.0,
                        "GLM5.2 PCP Full/Shared output");
    return 0;
  } catch (const std::exception& e) {
    if (checking_missing_slot &&
        std::string(e.what()).find("Device-side assert") != std::string::npos) {
      return 0;
    }
    LOG(ERROR) << "Rank " << rank << ": Exception: " << e.what();
    return 1;
  }
}

int32_t run_sparse_prefill_child(int32_t rank,
                                 int32_t world_size,
                                 int32_t port,
                                 const std::string& host,
                                 bool use_cp,
                                 int32_t second_length = 33,
                                 bool duplicate_slot = false) {
  return run_distributed_attention_child(
      rank,
      world_size,
      port,
      host,
      /*enable_context_parallel=*/true,
      [rank, world_size, use_cp, second_length, duplicate_slot](
          DistributedAttentionChildContext& context) {
        const auto& options = context.options();
        auto int_options = options.dtype(torch::kInt32);
        ModelArgs args = create_glm5_attention_model_args();
        QuantArgs quant_args = create_default_quant_args();
        StateDict weights = create_attention_state_dict(args, options);
        // Two requests cross both a physical block and a logical DCP block.
        // Their logical blocks are deliberately high and out of token order.
        const int32_t token_count = 17 + second_length;
        AttentionMetadata metadata =
            create_prefill_metadata(options, token_count);
        metadata.q_cu_seq_lens =
            torch::tensor({0, 17, token_count}, int_options);
        metadata.kv_cu_seq_lens = metadata.q_cu_seq_lens;
        metadata.kv_seq_lens = torch::tensor({17, second_length}, int_options);
        metadata.block_table = torch::tensor({{6, 0}, {2, 5}}, int_options);
        metadata.slot_mapping = torch::cat(
            {torch::arange(192, 209, int_options),
             torch::arange(
                 64, 64 + std::min(second_length, int32_t{32}), int_options),
             torch::arange(160,
                           160 + std::max(second_length - 32, int32_t{0}),
                           int_options)});
        metadata.max_query_len = std::max(int32_t{17}, second_length);
        metadata.max_seq_len = metadata.max_query_len;
        torch::Tensor positions =
            torch::cat({torch::arange(17, int_options),
                        torch::arange(second_length, int_options)});
        torch::Tensor hidden = seeded_tensor("oom/sparse_prefill",
                                             {token_count, args.hidden_size()},
                                             torch::kBFloat16,
                                             context.device());
        torch::Tensor tokens = torch::arange(token_count, int_options);
        // Hand-built causal top-k, in reverse token order, independent of the
        // production indexer and remapping algorithm. Padding tests include a
        // high positive slot outside context length and a zero-context row.
        torch::Tensor slots_cpu = metadata.slot_mapping.cpu();
        torch::Tensor table_cpu =
            torch::full({token_count, args.index_topk()}, -1, torch::kInt32);
        torch::Tensor lens_cpu = torch::zeros({token_count}, torch::kInt32);
        auto slot_data = slots_cpu.accessor<int32_t, 1>();
        auto table_data = table_cpu.accessor<int32_t, 2>();
        auto lens_data = lens_cpu.accessor<int32_t, 1>();
        for (int32_t row = 0; row < token_count; ++row) {
          const int32_t start = row < 17 ? 0 : 17;
          const int32_t count = std::min(
              row - start + 1, static_cast<int32_t>(args.index_topk()));
          lens_data[row] = count;
          for (int32_t col = 0; col < count; ++col) {
            table_data[row][col] = slot_data[row - col];
          }
        }
        lens_data[0] = 0;
        table_data[0][0] = 1000000;
        table_data[1][7] = 1000000;
        // Repeated padding slots are legal; these queries have no valid keys
        // and must not write the resident KV.
        metadata.slot_mapping.slice(/*dim=*/0, token_count - 2).fill_(-1);
        lens_cpu.slice(/*dim=*/0, token_count - 2).zero_();
        DsaTopkState global_topk(table_cpu.to(context.device()),
                                 lens_cpu.to(context.device()));
        const torch::Tensor original_table = global_topk.block_tables().clone();
        const torch::Tensor original_lens = global_topk.context_lens().clone();

        auto run = [&](bool sharded) {
          ParallelArgs parallel = context.parallel_args();
          if (sharded) {
            parallel.cp_size() = world_size;
            parallel.kv_split_size() = world_size;
            parallel.dcp_group_ = parallel.process_group_;
          } else {
            ProcessGroup* self = parallel.single_rank_group_;
            parallel = ParallelArgs(/*rank=*/0, /*world_size=*/1, self);
            parallel.tp_group_ = self;
            parallel.cp_group_ = self;
            parallel.dcp_group_ = self;
            parallel.single_rank_group_ = self;
            parallel.kv_split_size() = 1;
          }
          ModelContext model_context(parallel, args, quant_args, options);
          DeepseekV32Attention attention(model_context,
                                         /*enable_indexer=*/false);
          attention->load_state_dict(weights);
          AttentionMetadata batch = metadata;
          std::optional<v32_cp::DeepseekV32CPContext> cp;
          const KVShardLayout layout(
              /*physical_block_size=*/16, world_size, rank);
          if (sharded && duplicate_slot) {
            batch.slot_mapping = batch.slot_mapping.clone();
            batch.slot_mapping[1] = batch.slot_mapping[0];
          } else if (sharded) {
            batch.kv_shard_batch_metadata =
                build_mlu_shard_metadata(batch, layout);
          }
          if (sharded && use_cp) {
            cp =
                v32_cp::build_deepseek_v32_cp_context(world_size,
                                                      batch,
                                                      BatchForwardType::PREFILL,
                                                      tokens,
                                                      parallel.cp_group_,
                                                      rank,
                                                      world_size,
                                                      layout);
            CHECK(cp.has_value());
          }
          torch::Tensor query_hidden =
              cp ? v32_cp::reorder_to_local_shard(hidden, *cp) : hidden;
          torch::Tensor query_positions =
              cp ? v32_cp::reorder_to_local_shard(positions, *cp) : positions;
          DsaTopkState topk =
              cp ? DsaTopkState(v32_cp::reorder_to_local_shard(
                                    global_topk.block_tables(), *cp),
                                v32_cp::reorder_to_local_shard(
                                    global_topk.context_lens(), *cp))
                 : global_topk;
          torch::Tensor initial_topk = topk.block_tables().clone();
          std::optional<DsaPrefillRemap> prefill_remap;
          if (cp.has_value()) {
            const torch::Tensor& sorted_slots =
                cp->sorted_gathered_slot_mapping_int64;
            const torch::Tensor& sorted_rows = cp->sorted_gathered_slot_rows;
            const torch::Tensor& table = topk.block_tables();
            const torch::Tensor columns =
                torch::arange(table.size(1), table.options());
            const torch::Tensor valid =
                (columns < topk.context_lens().unsqueeze(1)) & (table >= 0);
            const torch::Tensor slots = table.to(torch::kInt64).contiguous();
            const torch::Tensor positions =
                torch::searchsorted(sorted_slots, slots);
            const torch::Tensor safe_positions =
                positions.clamp_max(sorted_slots.numel() - 1);
            const torch::Tensor matches =
                (positions < sorted_slots.numel()) &
                (sorted_slots.index_select(0, safe_positions.flatten())
                     .view_as(table) == slots);
            const torch::Tensor checked_positions = torch::where(
                valid,
                torch::where(matches, positions, sorted_slots.numel()),
                0);
            const torch::Tensor rows =
                sorted_rows.index_select(0, checked_positions.flatten())
                    .view_as(table);
            const torch::Tensor mapped =
                torch::where(valid, rows, torch::where(table < 0, -1, 0))
                    .to(table.scalar_type());
            prefill_remap.emplace(
                mapped, table, topk.context_lens(), sorted_slots, sorted_rows);
          }
          DsaTopkTransfer transfer =
              DsaTopkTransfer::reuse_and_capture(topk, prefill_remap);
          KVCache cache(KVCacheTensors{
              torch::zeros({sharded ? 8 : 16,
                            1,
                            16,
                            args.kv_lora_rank() + args.qk_rope_head_dim()},
                           options),
              torch::Tensor()});
          auto output = attention->forward(query_positions,
                                           query_hidden,
                                           batch,
                                           cache,
                                           cp ? &*cp : nullptr,
                                           &transfer);
          torch::Tensor global_output =
              cp ? v32_cp::gather_and_restore_global(output.output, *cp)
                 : output.output;
          context.xllm_device().synchronize_default_stream();
          CHECK(torch::equal(topk.block_tables(), initial_topk));
          CHECK(transfer.output() != nullptr);
          CHECK(torch::equal(transfer.output()->block_tables(), initial_topk));
          return std::make_pair(global_output, cache.get_k_cache());
        };
        auto [reference, reference_cache] = run(/*sharded=*/false);
        if (duplicate_slot) {
          try {
            run(/*sharded=*/true);
          } catch (const std::exception& error) {
            if (std::string(error.what()).find("Device-side assert") !=
                std::string::npos) {
              return 0;
            }
            throw;
          }
          LOG(ERROR) << "Prefill accepted duplicate valid source slots";
          return 1;
        }
        auto [actual, actual_cache] = run(/*sharded=*/true);
        check_tensors_close(
            actual,
            reference,
            /*rtol=*/1e-3,
            /*atol=*/1e-2,
            "Sparse prefill output against unsharded attention");
        // Independent physical layout: each logical block holds one 16-token
        // strip from each rank. Derive written slots from the input mapping so
        // holes and padding remain subject to exact checks.
        torch::Tensor expected_cache =
            reference_cache.view({8, world_size, 16, -1})
                .select(1, rank)
                .unsqueeze(1);
        torch::Tensor valid_slots =
            metadata.slot_mapping.cpu().to(torch::kInt64);
        valid_slots = valid_slots.masked_select(valid_slots.ge(0));
        torch::Tensor written_mask = torch::zeros({256}, torch::kBool);
        written_mask.index_fill_(/*dim=*/0, valid_slots, true);
        written_mask = written_mask.view({8, world_size, 16, 1})
                           .select(1, rank)
                           .unsqueeze(1)
                           .to(actual_cache.device())
                           .expand_as(actual_cache);
        // CP projects smaller, reordered batches independently. Allow BF16
        // rounding differences with a 1% relative tolerance and a small floor.
        check_tensors_close(actual_cache.masked_select(written_mask),
                            expected_cache.masked_select(written_mask),
                            /*rtol=*/1e-2,
                            /*atol=*/1e-2,
                            "Sparse prefill populated resident KV cache");
        torch::Tensor untouched_cache =
            actual_cache.masked_select(written_mask.logical_not());
        CHECK(torch::equal(untouched_cache, torch::zeros_like(untouched_cache)))
            << "Sparse prefill modified untouched resident KV cache slots";
        CHECK(torch::equal(global_topk.block_tables(), original_table));
        CHECK(torch::equal(global_topk.context_lens(), original_lens));
        return 0;
      });
}

int32_t run_dcp_remote_prefix_cache_merge_test_child(int32_t rank,
                                                     int32_t world_size,
                                                     int32_t port,
                                                     const std::string& host) {
  return run_distributed_attention_child(
      rank,
      world_size,
      port,
      host,
      /*enable_context_parallel=*/false,
      [world_size](DistributedAttentionChildContext& context) {
        xllm::Device& xllm_device = context.xllm_device();
        const torch::Device& device = context.device();
        const torch::TensorOptions& options = context.options();
        const int32_t rank = context.process_group()->rank();
        DcpDecodeContext dcp_context(
            KVShardLayout(
                KVCacheConfig::get_instance().block_size(), world_size, rank),
            context.process_group());

        // A cached prefix can put a valid suffix query on another rank's
        // local KV page. The -1 below is an ownership marker, not padding.
        torch::Tensor global_slot_mapping =
            torch::zeros({1}, options.dtype(torch::kInt32).device(device));
        torch::Tensor local_slot_mapping =
            dcp_context.localize_slots(global_slot_mapping);
        CHECK_EQ(global_slot_mapping.item<int32_t>(), 0);
        CHECK_EQ(local_slot_mapping.item<int32_t>(),
                 rank == 0 ? 0 : KVShardLayout::kInvalidSlot);

        torch::Tensor local_output =
            torch::full({1, 1, 1, 1}, static_cast<float>(rank + 2), options);
        torch::Tensor local_lse = torch::zeros(
            {1, 1, 1}, options.dtype(torch::kFloat32).device(device));
        torch::Tensor merged_output = dcp_context.merge(local_output,
                                                        local_lse,
                                                        global_slot_mapping,
                                                        /*head_sharded=*/false);

        xllm_device.synchronize_default_stream();
        torch::Tensor expected = torch::full_like(merged_output, 2.5f);
        check_tensors_close(
            merged_output,
            expected,
            /*rtol=*/0.0,
            /*atol=*/0.0,
            "DCP merge preserves a remote prefix-cache contribution");
        return 0;
      });
}

int32_t run_attention_topk_share_test_child(int32_t rank,
                                            int32_t world_size,
                                            int32_t port,
                                            const std::string& host) {
  return run_distributed_attention_child(
      rank,
      world_size,
      port,
      host,
      /*enable_context_parallel=*/true,
      [](DistributedAttentionChildContext& context) {
        xllm::Device& xllm_device = context.xllm_device();
        const torch::Device& device = context.device();
        ParallelArgs& parallel_args = context.parallel_args();
        const torch::TensorOptions& options = context.options();
        ModelArgs model_args =
            create_attention_model_args(/*q_lora_rank=*/64,
                                        /*enable_indexer=*/true);
        QuantArgs quant_args = create_default_quant_args();
        StateDict state_dict = create_attention_state_dict(model_args, options);

        // Mirror the production Full/Shared split. The Shared layer owns no
        // DeepseekV32Indexer, so its forward can only succeed by consuming the
        // Full layer's exported sparse metadata.
        OptimizationConfig optimization_config;
        optimization_config.enable_fused_mla_kernel = false;
        optimization_config.enable_fused_indexer_qk = false;
        DeepseekV32Attention full_attention(model_args,
                                            quant_args,
                                            parallel_args,
                                            options,
                                            optimization_config,
                                            /*enable_indexer=*/true);
        DeepseekV32Attention shared_attention(model_args,
                                              quant_args,
                                              parallel_args,
                                              options,
                                              optimization_config,
                                              /*enable_indexer=*/false);
        full_attention->load_state_dict(state_dict);
        shared_attention->load_state_dict(state_dict);
        CHECK(full_attention->named_children().contains("indexer"))
            << "Full layer must own an DeepseekV32Indexer";
        CHECK(!shared_attention->named_children().contains("indexer"))
            << "Shared layer must not own an DeepseekV32Indexer";

        torch::Tensor hidden_states =
            seeded_tensor("attention_multi_device/topk_share_hidden_states",
                          {1, model_args.hidden_size()},
                          torch::kBFloat16,
                          device);
        torch::Tensor positions =
            torch::tensor({0}, options.dtype(torch::kInt32).device(device));

        // Use independent, identically initialized caches so the Shared result
        // is compared with a clean Full-layer recomputation rather than with
        // state mutated by the baseline forward.
        KVCache full_kv_cache = create_decode_kv_cache(model_args, options);
        KVCache shared_kv_cache = create_decode_kv_cache(model_args, options);
        full_kv_cache.get_k_cache().zero_();
        shared_kv_cache.get_k_cache().zero_();

        // Full-layer baseline: independently compute the sparse block table and
        // the attention output.
        AttentionMetadata full_metadata = create_decode_metadata(options);
        DsaTopkTransfer full_transfer = DsaTopkTransfer::capture_output();
        torch::Tensor full_out = full_attention
                                     ->forward(positions,
                                               hidden_states,
                                               full_metadata,
                                               full_kv_cache,
                                               /*cp_ctx=*/nullptr,
                                               &full_transfer)
                                     .output;
        xllm_device.synchronize_default_stream();

        const DsaTopkState* full_topk = full_transfer.output();
        CHECK(full_topk != nullptr)
            << "Output layer must publish DSA top-k state";
        const torch::Tensor& full_block_tables = full_topk->block_tables();
        const torch::Tensor& full_context_lens = full_topk->context_lens();
        CHECK(full_block_tables.defined())
            << "Output layer must export a sparse block table";
        CHECK_EQ(full_block_tables.dim(), 2)
            << "sparse block table must be 2D {num_tokens, index_topk}";
        CHECK_EQ(full_block_tables.size(0), 1) << "num_tokens mismatch";
        CHECK_EQ(full_block_tables.size(1), model_args.index_topk())
            << "index_topk mismatch";
        CHECK(full_block_tables.scalar_type() == torch::kInt32)
            << "sparse block table must be int32";
        CHECK(torch::isfinite(full_out.to(torch::kFloat32)).all().item<bool>())
            << "Full layer attention output must be finite";

        // Shared-layer pass: use the production reuse-only relay. Pointer
        // checks below verify the relay does not copy metadata; numerical
        // equivalence with the independently recomputed Full result verifies
        // the no-DeepseekV32Indexer Shared path produces the expected attention
        // result.
        AttentionMetadata shared_metadata = create_decode_metadata(options);
        DsaTopkTransfer shared_transfer = DsaTopkTransfer::reuse(*full_topk);
        const DsaTopkState* shared_input = shared_transfer.input();
        CHECK(shared_input != nullptr) << "Shared layer must receive DSA top-k";
        CHECK_EQ(shared_input->block_tables().data_ptr(),
                 full_block_tables.data_ptr())
            << "Shared relay must reuse the Full layer's sparse block table";
        CHECK_EQ(shared_input->context_lens().data_ptr(),
                 full_context_lens.data_ptr())
            << "Shared relay must reuse the Full layer's context lens";
        torch::Tensor shared_out = shared_attention
                                       ->forward(positions,
                                                 hidden_states,
                                                 shared_metadata,
                                                 shared_kv_cache,
                                                 /*cp_ctx=*/nullptr,
                                                 &shared_transfer)
                                       .output;
        xllm_device.synchronize_default_stream();

        CHECK(shared_transfer.output() == nullptr)
            << "A production Shared relay must not self-echo its input";
        check_tensors_close(shared_out,
                            full_out,
                            /*rtol=*/1e-3,
                            /*atol=*/1e-2,
                            "Shared attention vs Full indexer baseline");
        return 0;
      });
}

// Dummy runs (DP sync / MTP step_empty / graph warmup) skip the indexer in
// attention, so an Output layer cannot export a valid sparse block table. This
// pins that contract: a dummy Output-layer forward must not produce a sparse
// {num_tokens, index_topk} table. It is the precondition for the decoder
// bypassing cross-layer sharing on dummy runs, without which a following Shared
// layer would assert on a missing previous top-k.
int32_t run_attention_topk_share_dummy_run_test_child(int32_t rank,
                                                      int32_t world_size,
                                                      int32_t port,
                                                      const std::string& host) {
  return run_distributed_attention_child(
      rank,
      world_size,
      port,
      host,
      /*enable_context_parallel=*/true,
      [](DistributedAttentionChildContext& context) {
        xllm::Device& xllm_device = context.xllm_device();
        const torch::Device& device = context.device();
        ParallelArgs& parallel_args = context.parallel_args();
        const torch::TensorOptions& options = context.options();
        ModelArgs model_args =
            create_attention_model_args(/*q_lora_rank=*/64,
                                        /*enable_indexer=*/true);
        QuantArgs quant_args = create_default_quant_args();
        StateDict state_dict = create_attention_state_dict(model_args, options);

        OptimizationConfig optimization_config;
        optimization_config.enable_fused_mla_kernel = false;
        optimization_config.enable_fused_indexer_qk = false;
        DeepseekV32Attention attention(model_args,
                                       quant_args,
                                       parallel_args,
                                       options,
                                       optimization_config);
        attention->load_state_dict(state_dict);

        torch::Tensor hidden_states = seeded_tensor(
            "attention_multi_device/topk_share_dummy_hidden_states",
            {1, model_args.hidden_size()},
            torch::kBFloat16,
            device);
        torch::Tensor positions =
            torch::tensor({0}, options.dtype(torch::kInt32).device(device));

        KVCache kv_cache = create_decode_kv_cache(model_args, options);
        kv_cache.get_k_cache().zero_();

        // Output-layer pass on a dummy run: the indexer is skipped, so no fresh
        // sparse block table is computed.
        AttentionMetadata dummy_metadata = create_decode_metadata(options);
        dummy_metadata.is_dummy = true;
        DsaTopkTransfer output_transfer = DsaTopkTransfer::capture_output();
        attention->forward(positions,
                           hidden_states,
                           dummy_metadata,
                           kv_cache,
                           /*cp_ctx=*/nullptr,
                           &output_transfer);
        xllm_device.synchronize_default_stream();

        // A dummy run must not yield a usable sparse top-k table (width
        // index_topk), so a Shared layer cannot legally reuse it. The decoder
        // therefore has to disable cross-layer sharing whenever
        // attn_metadata.is_dummy is set.
        const DsaTopkState* dummy_topk = output_transfer.output();
        const bool produced_sparse_topk =
            dummy_topk != nullptr && dummy_topk->block_tables().dim() == 2 &&
            dummy_topk->block_tables().size(1) == model_args.index_topk();
        CHECK(!produced_sparse_topk) << "Dummy Output-layer run must not "
                                        "export a sparse top-k block table";
        return 0;
      });
}

// Same Full->Shared numerical and relay contract as above, but on a multi-token
// decode bucket (batch_size > 1). This also pins the graph invariant that the
// reused sparse block table's shape strictly equals the captured num_tokens
// bucket, since every layer in one forward processes the same bucketed count.
int32_t run_attention_topk_share_bucket_test_child(int32_t rank,
                                                   int32_t world_size,
                                                   int32_t port,
                                                   const std::string& host) {
  return run_distributed_attention_child(
      rank,
      world_size,
      port,
      host,
      /*enable_context_parallel=*/true,
      [](DistributedAttentionChildContext& context) {
        xllm::Device& xllm_device = context.xllm_device();
        const torch::Device& device = context.device();
        ParallelArgs& parallel_args = context.parallel_args();
        const torch::TensorOptions& options = context.options();
        ModelArgs model_args =
            create_attention_model_args(/*q_lora_rank=*/64,
                                        /*enable_indexer=*/true);
        QuantArgs quant_args = create_default_quant_args();
        StateDict state_dict = create_attention_state_dict(model_args, options);

        OptimizationConfig optimization_config;
        optimization_config.enable_fused_mla_kernel = false;
        optimization_config.enable_fused_indexer_qk = false;
        DeepseekV32Attention full_attention(model_args,
                                            quant_args,
                                            parallel_args,
                                            options,
                                            optimization_config,
                                            /*enable_indexer=*/true);
        DeepseekV32Attention shared_attention(model_args,
                                              quant_args,
                                              parallel_args,
                                              options,
                                              optimization_config,
                                              /*enable_indexer=*/false);
        full_attention->load_state_dict(state_dict);
        shared_attention->load_state_dict(state_dict);
        CHECK(full_attention->named_children().contains("indexer"))
            << "Full layer must own an DeepseekV32Indexer";
        CHECK(!shared_attention->named_children().contains("indexer"))
            << "Shared layer must not own an DeepseekV32Indexer";

        constexpr int32_t kBucketTokens = 4;
        torch::Tensor hidden_states = seeded_tensor(
            "attention_multi_device/topk_share_bucket_hidden_states",
            {kBucketTokens, model_args.hidden_size()},
            torch::kBFloat16,
            device);
        torch::Tensor positions =
            torch::zeros({kBucketTokens}, options.dtype(torch::kInt32));

        KVCache full_kv_cache = create_decode_kv_cache(model_args, options);
        KVCache shared_kv_cache = create_decode_kv_cache(model_args, options);
        full_kv_cache.get_k_cache().zero_();
        shared_kv_cache.get_k_cache().zero_();

        AttentionMetadata full_metadata =
            create_batch_decode_metadata(options, kBucketTokens);
        DsaTopkTransfer full_transfer = DsaTopkTransfer::capture_output();
        torch::Tensor full_out = full_attention
                                     ->forward(positions,
                                               hidden_states,
                                               full_metadata,
                                               full_kv_cache,
                                               /*cp_ctx=*/nullptr,
                                               &full_transfer)
                                     .output;
        xllm_device.synchronize_default_stream();

        const DsaTopkState* full_topk = full_transfer.output();
        CHECK(full_topk != nullptr)
            << "Output layer must publish DSA top-k state";
        const torch::Tensor& full_block_tables = full_topk->block_tables();
        CHECK(full_block_tables.defined())
            << "Output layer must export a sparse block table";
        CHECK_EQ(full_block_tables.size(0), kBucketTokens)
            << "sparse block table row count must equal the decode bucket";
        CHECK_EQ(full_block_tables.size(1), model_args.index_topk())
            << "index_topk mismatch";

        AttentionMetadata shared_metadata =
            create_batch_decode_metadata(options, kBucketTokens);
        DsaTopkTransfer shared_transfer = DsaTopkTransfer::reuse(*full_topk);
        const DsaTopkState* shared_input = shared_transfer.input();
        CHECK(shared_input != nullptr) << "Shared layer must receive DSA top-k";
        CHECK_EQ(shared_input->block_tables().sizes(),
                 full_block_tables.sizes())
            << "Shared layer reuse shape must strictly equal the captured "
               "bucket";
        CHECK_EQ(shared_input->block_tables().data_ptr(),
                 full_block_tables.data_ptr())
            << "Shared relay must reuse the Full layer's sparse block table";
        torch::Tensor shared_out = shared_attention
                                       ->forward(positions,
                                                 hidden_states,
                                                 shared_metadata,
                                                 shared_kv_cache,
                                                 /*cp_ctx=*/nullptr,
                                                 &shared_transfer)
                                       .output;
        xllm_device.synchronize_default_stream();

        CHECK(shared_transfer.output() == nullptr)
            << "A production Shared relay must not self-echo its input";
        check_tensors_close(
            shared_out,
            full_out,
            /*rtol=*/1e-3,
            /*atol=*/1e-2,
            "Bucketed Shared attention vs Full indexer baseline");
        return 0;
      });
}

int32_t run_cp_position_cache_child(int32_t rank,
                                    int32_t world_size,
                                    int32_t port,
                                    const std::string& host,
                                    DsaModelVariant variant) {
  return run_distributed_attention_child(
      rank,
      world_size,
      port,
      host,
      /*enable_context_parallel=*/true,
      [variant](DistributedAttentionChildContext& context) {
        torch::NoGradGuard no_grad;
        const auto& options = context.options();
        ParallelArgs& parallel = context.parallel_args();
        ParallelArgs single(
            /*rank=*/0, /*world_size=*/1, parallel.single_rank_group_);
        single.tp_group_ = parallel.single_rank_group_;
        single.cp_group_ = parallel.single_rank_group_;
        single.dcp_group_ = parallel.single_rank_group_;
        single.single_rank_group_ = parallel.single_rank_group_;
        single.kv_split_size() = 1;
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
        }
        Weights values = create_seeded_mla_weights(
            args, DsaWeightFormat::BF16, options.device(torch::kCPU));
        auto& key_projection = values.at("indexer.wk.weight");
        key_projection = ((key_projection.to(torch::kFloat32) - 0.025) * 4.0)
                             .to(torch::kBFloat16);
        for (auto& [name, value] : values) {
          value = value.to(options.device());
        }
        StateDict weights(std::move(values));
        DeepseekV32Attention actual(
            ModelContext(parallel, args, QuantArgs(), options));
        DeepseekV32Attention reference(
            ModelContext(single, args, QuantArgs(), options));
        actual->load_state_dict(weights);
        reference->load_state_dict(weights);
        actual->verify_loaded_weights();
        reference->verify_loaded_weights();
        constexpr int32_t kTokens = 33;
        const auto ints = options.dtype(torch::kInt32);
        const torch::Tensor positions = torch::arange(kTokens, ints);
        const torch::Tensor hidden =
            (seeded_tensor("cp.position.input",
                           {kTokens, args.hidden_size()},
                           torch::kBFloat16,
                           torch::Device(torch::kCPU)) *
                 2.0 -
             1.0)
                .to(options.device());
        ModelInputParams params;
        params.meta.batch_forward_type = BatchForwardType::PREFILL;
        params.meta.num_sequences = 1;
        params.meta.q_max_seq_len = kTokens;
        params.meta.kv_max_seq_len = kTokens;
        params.attention.host.q_seq_lens = {kTokens};
        params.attention.host.kv_seq_lens = {kTokens};
        params.attention.device.q_seq_lens = torch::tensor({0, kTokens}, ints);
        params.attention.device.kv_seq_lens =
            params.attention.device.q_seq_lens;
        params.attention.device.new_cache_slots = positions;
        params.attention.device.block_tables = torch::tensor({{0, 1, 2}}, ints);
        const auto metadata = AttentionMetadataBuilder::build(
            params, /*enable_mla=*/true, /*compute_dtype=*/"bfloat16");
        auto cp = v32_cp::build_deepseek_v32_cp_context(
            parallel.cp_size(),
            metadata,
            BatchForwardType::PREFILL,
            positions,
            parallel.cp_group_,
            parallel.cp_group_->rank(),
            parallel.cp_group_->world_size());
        CHECK(cp.has_value());
        KVCache cache = create_indexed_kv_cache(
            seeded_tensor(
                "cp.position.key",
                {4, 1, 16, args.kv_lora_rank() + args.qk_rope_head_dim()},
                torch::kBFloat16,
                options.device()),
            seeded_tensor("cp.position.index",
                          {4, 1, 16, args.index_head_dim()},
                          torch::kBFloat16,
                          options.device()));
        const torch::Tensor initial_key = cache.get_k_cache().clone();
        const torch::Tensor initial_index = cache.get_index_cache().clone();
        KVCache reference_cache =
            create_indexed_kv_cache(initial_key.clone(), initial_index.clone());
        reference->forward(positions, hidden, metadata, reference_cache);
        actual->forward(v32_cp::reorder_to_local_shard(positions, cp.value()),
                        v32_cp::reorder_to_local_shard(hidden, cp.value()),
                        metadata,
                        cache,
                        &cp.value());
        context.xllm_device().synchronize_default_stream();
        const torch::Tensor actual_index =
            cache.get_index_cache().reshape({-1, args.index_head_dim()});
        const torch::Tensor expected_index =
            reference_cache.get_index_cache().reshape_as(actual_index);
        expect_close(actual_index.narrow(/*dim=*/0, /*start=*/0, kTokens),
                     expected_index.narrow(/*dim=*/0, /*start=*/0, kTokens),
                     /*rtol=*/0.02,
                     /*atol=*/0.002,
                     "CP explicit-position index cache");
        EXPECT_TRUE(
            torch::equal(actual_index.narrow(/*dim=*/0, kTokens, /*length=*/31),
                         initial_index.reshape_as(actual_index)
                             .narrow(/*dim=*/0, kTokens, /*length=*/31)));
        const torch::Tensor actual_key = cache.get_k_cache().reshape(
            {-1, args.kv_lora_rank() + args.qk_rope_head_dim()});
        expect_close(actual_key.narrow(/*dim=*/0, /*start=*/0, kTokens),
                     reference_cache.get_k_cache()
                         .reshape_as(actual_key)
                         .narrow(/*dim=*/0, /*start=*/0, kTokens),
                     /*rtol=*/0.02,
                     /*atol=*/0.002,
                     "CP MLA cache");
        EXPECT_TRUE(
            torch::equal(actual_key.narrow(/*dim=*/0, kTokens, /*length=*/31),
                         initial_key.reshape_as(actual_key)
                             .narrow(/*dim=*/0, kTokens, /*length=*/31)));
        return 0;
      });
}

}  // namespace

TEST(AttentionMultiDeviceProcessTest, TimedOutChildIsTerminatedAndReaped) {
  const pid_t child_pid = ::fork();
  ASSERT_GE(child_pid, 0) << "Failed to fork timeout test child";
  if (child_pid == 0) {
    while (true) {
      ::pause();
    }
  }

  const std::vector<pid_t> child_pids{child_pid};
  const ChildWaitResult result =
      wait_for_children(child_pids, std::chrono::milliseconds{50});
  EXPECT_TRUE(result.any_failed);
  EXPECT_TRUE(result.timed_out);

  int status = 0;
  errno = 0;
  EXPECT_EQ(wait_for_pid(child_pid, &status, WNOHANG), -1);
  EXPECT_EQ(errno, ECHILD) << "Timed-out child was not fully reaped";
}

// Two physical cards; each rank also runs its own single-rank reference.
TEST(AttentionMultiDeviceTest, CpIndexerDefaultPositionsMatchSingleRankCache) {
  const int32_t world_size_ = 2;
  const std::string host_ = "127.0.0.1";

  {
    const auto child_fn = [](int32_t rank,
                             int32_t world_size,
                             int32_t port,
                             const std::string& host) {
      return run_cp_position_cache_child(
          rank, world_size, port, host, DsaModelVariant::GLM5);
    };
    const std::string failure_message =
        "Default RoPE CP index cache differs from single rank.";

    const int32_t port = net::get_local_free_port();
    ASSERT_GT(port, 0) << "Failed to allocate a TCP rendezvous port";

    const int32_t child_count = world_size_;
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
            return child_fn(rank, world_size_, port, host_);
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
          child_pids, std::chrono::milliseconds{120000}, children_skipped);
    }

    if (children_ok && children_skipped) {
      GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                      "physical MLU cards";
    }
    ASSERT_EQ(children_ok, true)
        << failure_message << " "
        << "forked ranks failed, timed out, or require more physical MLU cards";
  };
}
TEST(AttentionMultiDeviceTest, CpIndexerYarnPositionsMatchSingleRankCache) {
  const int32_t world_size_ = 2;
  const std::string host_ = "127.0.0.1";

  {
    const auto child_fn = [](int32_t rank,
                             int32_t world_size,
                             int32_t port,
                             const std::string& host) {
      return run_cp_position_cache_child(
          rank, world_size, port, host, DsaModelVariant::V32);
    };
    const std::string failure_message =
        "YaRN CP index cache differs from single rank.";

    const int32_t port = net::get_local_free_port();
    ASSERT_GT(port, 0) << "Failed to allocate a TCP rendezvous port";

    const int32_t child_count = world_size_;
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
            return child_fn(rank, world_size_, port, host_);
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
          child_pids, std::chrono::milliseconds{120000}, children_skipped);
    }

    if (children_ok && children_skipped) {
      GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                      "physical MLU cards";
    }
    ASSERT_EQ(children_ok, true)
        << failure_message << " "
        << "forked ranks failed, timed out, or require more physical MLU cards";
  };
}
TEST(AttentionMultiDeviceTest, Glm5DecodeFullWeightsMatchSingleRank) {
  const int32_t world_size_ = 2;
  const std::string host_ = "127.0.0.1";

  {
    {
      const auto child_fn = [enable_fused_mla_kernel = false](
                                int32_t rank,
                                int32_t world_size,
                                int32_t port,
                                const std::string& host) {
        return run_glm_decode_child(
            rank, world_size, port, host, (/*enable_fused_mla_kernel=*/false));
      };
      const std::string failure_message =
          "Attention multi-device equivalence test failed.";

      const int32_t port = net::get_local_free_port();
      ASSERT_GT(port, 0) << "Failed to allocate a TCP rendezvous port";

      const int32_t child_count = world_size_;
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
              return child_fn(rank, world_size_, port, host_);
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
            child_pids, std::chrono::milliseconds{120000}, children_skipped);
      }

      if (children_ok && children_skipped) {
        GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                        "physical MLU cards";
      }
      ASSERT_EQ(children_ok, true) << failure_message << " "
                                   << "forked ranks failed, timed out, or "
                                      "require more physical MLU cards";
    };
  };
}

// Two physical cards using the fused MLA decode kernel.
TEST(AttentionMultiDeviceTest, Glm5FusedDecodeFullWeightsMatchSingleRank) {
  const int32_t world_size_ = 2;
  const std::string host_ = "127.0.0.1";

  {
    {
      const auto child_fn = [enable_fused_mla_kernel = true](
                                int32_t rank,
                                int32_t world_size,
                                int32_t port,
                                const std::string& host) {
        return run_glm_decode_child(
            rank, world_size, port, host, (/*enable_fused_mla_kernel=*/true));
      };
      const std::string failure_message =
          "Attention multi-device equivalence test failed.";

      const int32_t port = net::get_local_free_port();
      ASSERT_GT(port, 0) << "Failed to allocate a TCP rendezvous port";

      const int32_t child_count = world_size_;
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
              return child_fn(rank, world_size_, port, host_);
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
            child_pids, std::chrono::milliseconds{120000}, children_skipped);
      }

      if (children_ok && children_skipped) {
        GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                        "physical MLU cards";
      }
      ASSERT_EQ(children_ok, true) << failure_message << " "
                                   << "forked ranks failed, timed out, or "
                                      "require more physical MLU cards";
    };
  };
}

TEST(AttentionMultiDeviceTest, PrefillShortSeqFallsBackToReplicatedPath) {
  const int32_t world_size_ = 2;
  const std::string host_ = "127.0.0.1";

  {
    {
      const auto child_fn = [](int32_t rank,
                               int32_t world_size,
                               int32_t port,
                               const std::string& host) {
        return run_attention_prefill_fallback_test_child(
            rank, world_size, port, host);
      };
      const std::string failure_message =
          "Attention multi-device prefill fallback test failed.";

      const int32_t port = net::get_local_free_port();
      ASSERT_GT(port, 0) << "Failed to allocate a TCP rendezvous port";

      const int32_t child_count = world_size_;
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
              return child_fn(rank, world_size_, port, host_);
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
            child_pids, std::chrono::milliseconds{120000}, children_skipped);
      }

      if (children_ok && children_skipped) {
        GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                        "physical MLU cards";
      }
      ASSERT_EQ(children_ok, true) << failure_message << " "
                                   << "forked ranks failed, timed out, or "
                                      "require more physical MLU cards";
    };
  };
}

TEST(AttentionMultiDeviceTest, PrefillRejectsMissingTopkSlot) {
  const int32_t world_size_ = 2;
  const std::string host_ = "127.0.0.1";

  {
    const auto child_fn = [](int32_t rank,
                             int32_t world_size,
                             int32_t port,
                             const std::string& host) {
      return run_prefill_share_child(
          rank, world_size, port, host, /*missing_slot=*/true);
    };
    const std::string failure_message =
        "Prefill must reject a top-k slot absent from the input K.";

    const int32_t port = net::get_local_free_port();
    ASSERT_GT(port, 0) << "Failed to allocate a TCP rendezvous port";

    const int32_t child_count = world_size_;
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
            return child_fn(rank, world_size_, port, host_);
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
          child_pids, std::chrono::milliseconds{120000}, children_skipped);
    }

    if (children_ok && children_skipped) {
      GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                      "physical MLU cards";
    }
    ASSERT_EQ(children_ok, true)
        << failure_message << " "
        << "forked ranks failed, timed out, or require more physical MLU cards";
  };
}

TEST(AttentionMultiDeviceTest, EmptyDcpPrefillLeavesResidentCacheUnchanged) {
  const int32_t world_size_ = 2;
  const std::string host_ = "127.0.0.1";

  {
    const auto child_fn = [](int32_t rank,
                             int32_t world_size,
                             int32_t port,
                             const std::string& host) {
      return run_distributed_attention_child(
          rank,
          world_size,
          port,
          host,
          /*enable_context_parallel=*/true,
          [world_size](DistributedAttentionChildContext& context) {
            ParallelArgs parallel = context.parallel_args();
            parallel.kv_split_size() = world_size;
            parallel.dcp_group_ = parallel.process_group_;
            const auto& options = context.options();
            ModelArgs args = create_glm5_attention_model_args();
            QuantArgs quant_args = create_default_quant_args();
            StateDict weights = create_attention_state_dict(args, options);
            KVCache cache = create_decode_kv_cache(args, options);
            torch::Tensor original = cache.get_k_cache().clone();
            torch::Tensor tokens =
                torch::empty({0}, options.dtype(torch::kInt32));
            AttentionRunResult result = run_attention_prefill_once(
                args,
                quant_args,
                parallel,
                options,
                weights,
                tokens,
                tokens,
                torch::empty({0, args.hidden_size()}, options),
                cache,
                /*enable_full_weight_path=*/true,
                BatchForwardType::PREFILL,
                /*prefix_len=*/0,
                /*build_cp_context=*/false);
            CHECK_EQ(result.global_output.size(0), 0);
            CHECK_EQ(result.global_output.size(1), args.hidden_size());
            CHECK(torch::equal(cache.get_k_cache(), original));
            return 0;
          });
    };
    const std::string failure_message =
        "Empty DCP prefill must leave the resident cache unchanged.";

    const int32_t port = net::get_local_free_port();
    ASSERT_GT(port, 0) << "Failed to allocate a TCP rendezvous port";

    const int32_t child_count = world_size_;
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
            return child_fn(rank, world_size_, port, host_);
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
          child_pids, std::chrono::milliseconds{120000}, children_skipped);
    }

    if (children_ok && children_skipped) {
      GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                      "physical MLU cards";
    }
    ASSERT_EQ(children_ok, true)
        << failure_message << " "
        << "forked ranks failed, timed out, or require more physical MLU cards";
  };
}

TEST(AttentionMultiDeviceTest, DcpPrefillRejectsMissingTopkSlot) {
  const int32_t world_size_ = 2;
  const std::string host_ = "127.0.0.1";

  {
    const auto child_fn = [](int32_t rank,
                             int32_t world_size,
                             int32_t port,
                             const std::string& host) {
      return run_prefill_share_child(rank,
                                     world_size,
                                     port,
                                     host,
                                     /*missing_slot=*/true,
                                     /*use_cp=*/false);
    };
    const std::string failure_message =
        "DCP prefill must reject a top-k slot absent from the input K.";

    const int32_t port = net::get_local_free_port();
    ASSERT_GT(port, 0) << "Failed to allocate a TCP rendezvous port";

    const int32_t child_count = world_size_;
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
            return child_fn(rank, world_size_, port, host_);
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
          child_pids, std::chrono::milliseconds{120000}, children_skipped);
    }

    if (children_ok && children_skipped) {
      GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                      "physical MLU cards";
    }
    ASSERT_EQ(children_ok, true)
        << failure_message << " "
        << "forked ranks failed, timed out, or require more physical MLU cards";
  };
}

TEST(AttentionMultiDeviceTest, DcpPrefillRejectsDuplicateSourceSlots) {
  const int32_t world_size_ = 2;
  const std::string host_ = "127.0.0.1";

  {
    const auto child_fn = [](int32_t rank,
                             int32_t world_size,
                             int32_t port,
                             const std::string& host) {
      return run_sparse_prefill_child(rank,
                                      world_size,
                                      port,
                                      host,
                                      /*use_cp=*/false,
                                      /*second_length=*/33,
                                      /*duplicate_slot=*/true);
    };
    const std::string failure_message =
        "DCP prefill must reject duplicate valid source slots.";

    const int32_t port = net::get_local_free_port();
    ASSERT_GT(port, 0) << "Failed to allocate a TCP rendezvous port";

    const int32_t child_count = world_size_;
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
            return child_fn(rank, world_size_, port, host_);
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
          child_pids, std::chrono::milliseconds{120000}, children_skipped);
    }

    if (children_ok && children_skipped) {
      GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                      "physical MLU cards";
    }
    ASSERT_EQ(children_ok, true)
        << failure_message << " "
        << "forked ranks failed, timed out, or require more physical MLU cards";
  };
}

TEST(AttentionMultiDeviceTest, DcpSparsePrefillMatchesUnshardedAttention) {
  const int32_t world_size_ = 2;
  const std::string host_ = "127.0.0.1";

  {
    const auto child_fn = [](int32_t rank,
                             int32_t world_size,
                             int32_t port,
                             const std::string& host) {
      return run_sparse_prefill_child(rank,
                                      world_size,
                                      port,
                                      host,
                                      /*use_cp=*/false);
    };
    const std::string failure_message =
        "DCP sparse prefill must match unsharded attention.";

    const int32_t port = net::get_local_free_port();
    ASSERT_GT(port, 0) << "Failed to allocate a TCP rendezvous port";

    const int32_t child_count = world_size_;
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
            return child_fn(rank, world_size_, port, host_);
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
          child_pids, std::chrono::milliseconds{120000}, children_skipped);
    }

    if (children_ok && children_skipped) {
      GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                      "physical MLU cards";
    }
    ASSERT_EQ(children_ok, true)
        << failure_message << " "
        << "forked ranks failed, timed out, or require more physical MLU cards";
  };
}

TEST(AttentionMultiDeviceTest, CpDcpSparsePrefillMatchesUnshardedAttention) {
  const int32_t world_size_ = 2;
  const std::string host_ = "127.0.0.1";

  {
    const auto child_fn = [](int32_t rank,
                             int32_t world_size,
                             int32_t port,
                             const std::string& host) {
      return run_sparse_prefill_child(rank,
                                      world_size,
                                      port,
                                      host,
                                      /*use_cp=*/true);
    };
    const std::string failure_message =
        "CP+DCP sparse prefill must match unsharded attention.";

    const int32_t port = net::get_local_free_port();
    ASSERT_GT(port, 0) << "Failed to allocate a TCP rendezvous port";

    const int32_t child_count = world_size_;
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
            return child_fn(rank, world_size_, port, host_);
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
          child_pids, std::chrono::milliseconds{120000}, children_skipped);
    }

    if (children_ok && children_skipped) {
      GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                      "physical MLU cards";
    }
    ASSERT_EQ(children_ok, true)
        << failure_message << " "
        << "forked ranks failed, timed out, or require more physical MLU cards";
  };
}

TEST(AttentionMultiDeviceTest, DcpPrefillAtBlockBoundaryMatchesUnsharded) {
  const int32_t world_size_ = 2;
  const std::string host_ = "127.0.0.1";

  {
    const auto child_fn = [](int32_t rank,
                             int32_t world_size,
                             int32_t port,
                             const std::string& host) {
      return run_sparse_prefill_child(rank,
                                      world_size,
                                      port,
                                      host,
                                      /*use_cp=*/false,
                                      /*second_length=*/15);
    };
    const std::string failure_message =
        "DCP block-aligned prefill must match unsharded attention.";

    const int32_t port = net::get_local_free_port();
    ASSERT_GT(port, 0) << "Failed to allocate a TCP rendezvous port";

    const int32_t child_count = world_size_;
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
            return child_fn(rank, world_size_, port, host_);
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
          child_pids, std::chrono::milliseconds{120000}, children_skipped);
    }

    if (children_ok && children_skipped) {
      GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                      "physical MLU cards";
    }
    ASSERT_EQ(children_ok, true)
        << failure_message << " "
        << "forked ranks failed, timed out, or require more physical MLU cards";
  };
}

TEST(AttentionMultiDeviceTest, CpDcpPrefillTailBlockMatchesUnsharded) {
  const int32_t world_size_ = 2;
  const std::string host_ = "127.0.0.1";

  {
    const auto child_fn = [](int32_t rank,
                             int32_t world_size,
                             int32_t port,
                             const std::string& host) {
      return run_sparse_prefill_child(rank,
                                      world_size,
                                      port,
                                      host,
                                      /*use_cp=*/true,
                                      /*second_length=*/16);
    };
    const std::string failure_message =
        "CP+DCP tail-block prefill must match unsharded attention.";

    const int32_t port = net::get_local_free_port();
    ASSERT_GT(port, 0) << "Failed to allocate a TCP rendezvous port";

    const int32_t child_count = world_size_;
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
            return child_fn(rank, world_size_, port, host_);
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
          child_pids, std::chrono::milliseconds{120000}, children_skipped);
    }

    if (children_ok && children_skipped) {
      GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                      "physical MLU cards";
    }
    ASSERT_EQ(children_ok, true)
        << failure_message << " "
        << "forked ranks failed, timed out, or require more physical MLU cards";
  };
}

TEST(AttentionMultiDeviceTest, Glm52PrefillCpReusesRankLocalTopkState) {
  const int32_t world_size_ = 2;
  const std::string host_ = "127.0.0.1";

  {
    {
      const auto child_fn = [](int32_t rank,
                               int32_t world_size,
                               int32_t port,
                               const std::string& host) {
        return run_prefill_share_child(rank, world_size, port, host);
      };
      const std::string failure_message =
          "GLM5.2 PCP cross-layer top-k share test failed.";

      const int32_t port = net::get_local_free_port();
      ASSERT_GT(port, 0) << "Failed to allocate a TCP rendezvous port";

      const int32_t child_count = world_size_;
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
              return child_fn(rank, world_size_, port, host_);
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
            child_pids, std::chrono::milliseconds{120000}, children_skipped);
      }

      if (children_ok && children_skipped) {
        GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                        "physical MLU cards";
      }
      ASSERT_EQ(children_ok, true) << failure_message << " "
                                   << "forked ranks failed, timed out, or "
                                      "require more physical MLU cards";
    };
  };
}

TEST(AttentionMultiDeviceTest, DcpMergePreservesRemotePrefixCacheContribution) {
  const int32_t world_size_ = 2;
  const std::string host_ = "127.0.0.1";

  {
    {
      const auto child_fn = [](int32_t rank,
                               int32_t world_size,
                               int32_t port,
                               const std::string& host) {
        return run_dcp_remote_prefix_cache_merge_test_child(
            rank, world_size, port, host);
      };
      const std::string failure_message =
          "DCP remote prefix-cache merge test failed.";

      const int32_t port = net::get_local_free_port();
      ASSERT_GT(port, 0) << "Failed to allocate a TCP rendezvous port";

      const int32_t child_count = world_size_;
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
              return child_fn(rank, world_size_, port, host_);
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
            child_pids, std::chrono::milliseconds{120000}, children_skipped);
      }

      if (children_ok && children_skipped) {
        GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                        "physical MLU cards";
      }
      ASSERT_EQ(children_ok, true) << failure_message << " "
                                   << "forked ranks failed, timed out, or "
                                      "require more physical MLU cards";
    };
  };
}

TEST(AttentionMultiDeviceTest, DummyRunOutputLayerYieldsNoSparseTopk) {
  const int32_t world_size_ = 2;
  const std::string host_ = "127.0.0.1";

  {
    {
      const auto child_fn = [](int32_t rank,
                               int32_t world_size,
                               int32_t port,
                               const std::string& host) {
        return run_attention_topk_share_dummy_run_test_child(
            rank, world_size, port, host);
      };
      const std::string failure_message =
          "Attention dummy-run top-k share test failed.";

      const int32_t port = net::get_local_free_port();
      ASSERT_GT(port, 0) << "Failed to allocate a TCP rendezvous port";

      const int32_t child_count = world_size_;
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
              return child_fn(rank, world_size_, port, host_);
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
            child_pids, std::chrono::milliseconds{120000}, children_skipped);
      }

      if (children_ok && children_skipped) {
        GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                        "physical MLU cards";
      }
      ASSERT_EQ(children_ok, true) << failure_message << " "
                                   << "forked ranks failed, timed out, or "
                                      "require more physical MLU cards";
    };
  };
}

TEST(AttentionMultiDeviceTest, SharedLayerReusesFullLayerSparseBlockTable) {
  const int32_t world_size_ = 2;
  const std::string host_ = "127.0.0.1";

  {
    {
      const auto child_fn = [](int32_t rank,
                               int32_t world_size,
                               int32_t port,
                               const std::string& host) {
        return run_attention_topk_share_test_child(
            rank, world_size, port, host);
      };
      const std::string failure_message =
          "Attention cross-layer top-k share test failed.";

      const int32_t port = net::get_local_free_port();
      ASSERT_GT(port, 0) << "Failed to allocate a TCP rendezvous port";

      const int32_t child_count = world_size_;
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
              return child_fn(rank, world_size_, port, host_);
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
            child_pids, std::chrono::milliseconds{120000}, children_skipped);
      }

      if (children_ok && children_skipped) {
        GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                        "physical MLU cards";
      }
      ASSERT_EQ(children_ok, true) << failure_message << " "
                                   << "forked ranks failed, timed out, or "
                                      "require more physical MLU cards";
    };
  };
}

TEST(AttentionMultiDeviceTest, SharedLayerReuseShapeMatchesDecodeBucket) {
  const int32_t world_size_ = 2;
  const std::string host_ = "127.0.0.1";

  {
    {
      const auto child_fn = [](int32_t rank,
                               int32_t world_size,
                               int32_t port,
                               const std::string& host) {
        return run_attention_topk_share_bucket_test_child(
            rank, world_size, port, host);
      };
      const std::string failure_message =
          "Attention cross-layer top-k share bucket test failed.";

      const int32_t port = net::get_local_free_port();
      ASSERT_GT(port, 0) << "Failed to allocate a TCP rendezvous port";

      const int32_t child_count = world_size_;
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
              return child_fn(rank, world_size_, port, host_);
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
            child_pids, std::chrono::milliseconds{120000}, children_skipped);
      }

      if (children_ok && children_skipped) {
        GTEST_SKIP() << "forked ranks failed, timed out, or require more "
                        "physical MLU cards";
      }
      ASSERT_EQ(children_ok, true) << failure_message << " "
                                   << "forked ranks failed, timed out, or "
                                      "require more physical MLU cards";
    };
  };
}

}  // namespace test
}  // namespace layer
}  // namespace xllm
