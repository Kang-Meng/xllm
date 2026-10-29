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

#include "kernels/mlu/chunk_kda.h"

#include <framework/core/MLUStream.h>
#include <glog/logging.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <tuple>

#include "kernels/mlu/bangc/include/bangc_kda.h"
#include "kernels/mlu/bangc/include/bangc_kda_tensor.h"

namespace xllm::kernel::mlu {
namespace {

constexpr int64_t kHeadDim = 128;
constexpr int64_t kWorkspaceLimitBytes = 2LL * 1024 * 1024 * 1024;

int64_t configured_prefill_chunk_size() {
  static const int64_t chunk_size = [] {
    const char* env = std::getenv("XLLM_MLU_KDA_CHUNK_SIZE");
    if (env == nullptr) {
      return ChunkKDAImpl::kDefaultChunkSize;
    }
    char* parse_end = nullptr;
    const int64_t parsed =
        static_cast<int64_t>(std::strtoll(env, &parse_end, 10));
    CHECK(parse_end != env && *parse_end == '\0')
        << "XLLM_MLU_KDA_CHUNK_SIZE must be 16 or 64";
    CHECK(parsed == 16 || parsed == 64)
        << "XLLM_MLU_KDA_CHUNK_SIZE must be 16 or 64";
    return parsed;
  }();
  return chunk_size;
}

}  // namespace

int64_t kda_prefill_chunk_size(int64_t num_heads, bool use_qk_l2norm) {
  CHECK_GT(num_heads, 0);
  static_cast<void>(use_qk_l2norm);
  return configured_prefill_chunk_size();
}

namespace {

void validate_inputs(const torch::Tensor& q,
                     const torch::Tensor& k,
                     const torch::Tensor& v,
                     const torch::Tensor& initial_state,
                     const torch::Tensor& cu_seqlens,
                     int64_t num_heads) {
  CHECK_EQ(q.dim(), 4) << "Chunk KDA q must be [1,T,H,K]";
  CHECK_EQ(k.sizes(), q.sizes()) << "Chunk KDA q/k shape mismatch";
  CHECK_EQ(v.dim(), 4) << "Chunk KDA v must be [1,T,H,V]";
  CHECK_EQ(q.size(0), 1) << "Chunk KDA only supports packed batch size 1";
  CHECK_EQ(v.size(0), 1) << "Chunk KDA v batch size mismatch";
  CHECK_EQ(v.size(1), q.size(1)) << "Chunk KDA v token count mismatch";
  CHECK_EQ(q.size(2), num_heads) << "Chunk KDA q head count mismatch";
  CHECK_EQ(v.size(2), num_heads) << "Chunk KDA v head count mismatch";
  CHECK_EQ(q.size(3), kHeadDim) << "Chunk KDA requires key dimension 128";
  CHECK_EQ(v.size(3), kHeadDim) << "Chunk KDA requires value dimension 128";
  CHECK_EQ(q.scalar_type(), torch::kBFloat16) << "Chunk KDA q must be bfloat16";
  CHECK_EQ(k.scalar_type(), torch::kBFloat16) << "Chunk KDA k must be bfloat16";
  CHECK_EQ(v.scalar_type(), torch::kBFloat16) << "Chunk KDA v must be bfloat16";
  CHECK_EQ(initial_state.scalar_type(), torch::kFloat32)
      << "Chunk KDA state must be float32";
  CHECK_EQ(initial_state.dim(), 4) << "Chunk KDA state must be [N,H,V,K]";
  CHECK_EQ(initial_state.size(0), cu_seqlens.size(0) - 1)
      << "Chunk KDA requires one initial state per sequence";
  CHECK_EQ(initial_state.size(1), num_heads)
      << "Chunk KDA state head count mismatch";
  CHECK_EQ(initial_state.size(2), v.size(3))
      << "Chunk KDA state value dimension mismatch";
  CHECK_EQ(initial_state.size(3), q.size(3))
      << "Chunk KDA state key dimension mismatch";
  CHECK_EQ(cu_seqlens.dim(), 1) << "Chunk KDA cu_seqlens must be 1D";
  CHECK_GT(q.size(1), 0) << "Chunk KDA requires at least one token";
}

}  // namespace

ChunkKDAImpl::ChunkKDAImpl(int64_t num_heads) : num_heads_(num_heads) {
  CHECK_GT(num_heads_, 0) << "Chunk KDA requires at least one head";
}

std::tuple<torch::Tensor, torch::Tensor> ChunkKDAImpl::forward_raw_gate(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& raw_gate,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_bias,
    float gate_lower_bound,
    const torch::Tensor& raw_beta,
    const torch::Tensor& initial_state,
    const torch::Tensor& cu_seqlens,
    const torch::Tensor& /*chunk_indices*/,
    bool output_final_state,
    bool use_qk_l2norm) {
  const auto forward_bangc = [&]() {
    CHECK_EQ(q.dim(), 4);
    CHECK_EQ(v.dim(), 4);
    CHECK_EQ(q.size(2), num_heads_);
    CHECK_EQ(v.size(2), num_heads_);
    BangcKdaTensorParams parameters{};
    parameters.q = q;
    parameters.k = k;
    parameters.v = v;
    parameters.gate = raw_gate;
    parameters.beta = raw_beta;
    parameters.a_log = a_log;
    parameters.bias = dt_bias;
    parameters.initial_state = initial_state;
    parameters.cu = cu_seqlens;
    parameters.gate_mode = 0;
    parameters.prefill = true;
    parameters.output_final_state = output_final_state;
    parameters.qk_l2norm = use_qk_l2norm;
    parameters.scale = 1.f / std::sqrt(static_cast<float>(q.size(3)));
    parameters.lower_bound = gate_lower_bound;
    auto result = bangc_kda_forward(parameters);
    return std::make_tuple(result.first, result.second);
  };
  // The staged prefill path requires BF16 inputs and FP32 state. Other typed
  // inputs, including empty packed batches, use the generic BangC path.
  const bool staged_prefill_shape =
      q.dim() == 4 && k.sizes() == q.sizes() && v.dim() == 4 &&
      q.size(0) == 1 && v.size(0) == 1 && q.size(1) > 0 &&
      v.size(1) == q.size(1) && q.size(2) == num_heads_ &&
      v.size(2) == num_heads_ && q.size(3) == kHeadDim &&
      v.size(3) == kHeadDim && q.scalar_type() == torch::kBFloat16 &&
      k.scalar_type() == torch::kBFloat16 &&
      v.scalar_type() == torch::kBFloat16 &&
      initial_state.scalar_type() == torch::kFloat32 &&
      initial_state.dim() == 4 && cu_seqlens.dim() == 1 &&
      cu_seqlens.numel() > 1;
  if (!staged_prefill_shape) {
    return forward_bangc();
  }

  validate_inputs(q, k, v, initial_state, cu_seqlens, num_heads_);
  CHECK_LT(gate_lower_bound, 0.0f)
      << "Chunk KDA gate lower bound must be negative";
  torch::Tensor gate_work =
      (raw_gate.dim() == 3 ? raw_gate.unsqueeze(/*dim=*/0) : raw_gate)
          .contiguous();
  torch::Tensor beta_work =
      (raw_beta.dim() == 2 ? raw_beta.unsqueeze(/*dim=*/0) : raw_beta)
          .contiguous();
  CHECK_EQ(gate_work.dim(), 4) << "Chunk KDA raw gate must be [1,T,H,K]";
  CHECK_EQ(gate_work.sizes(), q.sizes()) << "Chunk KDA raw gate shape mismatch";
  CHECK_EQ(beta_work.dim(), 3) << "Chunk KDA raw beta must be [1,T,H]";
  CHECK_EQ(beta_work.size(0), 1) << "Chunk KDA raw beta batch size mismatch";
  CHECK_EQ(beta_work.size(1), q.size(1))
      << "Chunk KDA raw beta token count mismatch";
  CHECK_EQ(beta_work.size(2), num_heads_)
      << "Chunk KDA raw beta head count mismatch";
  CHECK_EQ(a_log.dim(), 1) << "Chunk KDA A_log must be [H]";
  CHECK_EQ(a_log.size(0), num_heads_) << "Chunk KDA A_log head count mismatch";
  CHECK_EQ(dt_bias.dim(), 1) << "Chunk KDA dt_bias must be [H*K]";
  CHECK_EQ(dt_bias.size(0), num_heads_ * kHeadDim)
      << "Chunk KDA dt_bias dimension mismatch";

  const int64_t num_sequences = cu_seqlens.size(0) - 1;
  // Every native job and normalized CU value must fit the int32 ABI.
  const int64_t max_native_index = std::numeric_limits<int32_t>::max();
  const bool native_heads = num_heads_ == 8 || num_heads_ == 16 ||
                            num_heads_ == 32 || num_heads_ == 64;
  const bool native_capacity =
      native_heads && num_sequences > 0 && num_heads_ > 0 &&
      q.size(1) <= max_native_index && num_heads_ <= max_native_index / 4 &&
      num_sequences <= max_native_index / (num_heads_ * 4);
  // The native block recurrence carries FP32 state across arbitrary packed
  // sequence boundaries; context length does not change its working set.
  const bool same_device =
      k.device() == q.device() && v.device() == q.device() &&
      gate_work.device() == q.device() && beta_work.device() == q.device() &&
      a_log.device() == q.device() && dt_bias.device() == q.device() &&
      initial_state.device() == q.device() && cu_seqlens.device() == q.device();
  const bool use_staged_prefill = same_device && native_capacity &&
                                  use_qk_l2norm && gate_lower_bound == -5.0f &&
                                  gate_work.scalar_type() == torch::kBFloat16 &&
                                  beta_work.scalar_type() == torch::kBFloat16 &&
                                  a_log.scalar_type() == torch::kFloat32 &&
                                  dt_bias.scalar_type() == torch::kFloat32;
  if (use_staged_prefill) {
    const torch::DeviceGuard device_guard(q.device());
    torch::Tensor q_work = q.contiguous();
    torch::Tensor k_work = k.contiguous();
    torch::Tensor v_work = v.contiguous();
    torch::Tensor state_work = initial_state.contiguous();
    torch::Tensor log_work = a_log.contiguous();
    torch::Tensor bias_work = dt_bias.contiguous();
    torch::Tensor cu_work = cu_seqlens.contiguous().to(torch::kInt32);
    torch::Tensor output = torch::empty_like(v_work);
    torch::Tensor final_state = torch::empty_like(state_work);
    torch::Tensor workspace;
    const int64_t total_tokens = q_work.size(1);
    const bool underfilled_long =
        num_sequences * num_heads_ <= 16 && total_tokens >= num_sequences * 256;
    const bool adaptive_long = num_heads_ == 8 && num_sequences >= 3 &&
                               num_sequences <= 8 &&
                               total_tokens >= num_sequences * 256;
    const int64_t slots = (total_tokens + 31) / 32 + num_sequences;
    if ((underfilled_long || adaptive_long) &&
        slots <= kWorkspaceLimitBytes / sizeof(float) / num_heads_ /
                     kBangcKdaWorkspaceFloatsPerHeadBlock) {
      workspace = torch::empty(
          {slots * num_heads_ * kBangcKdaWorkspaceFloatsPerHeadBlock},
          q_work.options().dtype(torch::kFloat32));
    }
    cnrtQueue_t queue = torch_mlu::getCurMLUStream(q.device().index());
    CHECK_EQ(bangc_kda_prefill_staged(
                 static_cast<void*>(queue),
                 q_work.data_ptr(),
                 k_work.data_ptr(),
                 v_work.data_ptr(),
                 gate_work.data_ptr(),
                 beta_work.data_ptr(),
                 log_work.data_ptr(),
                 bias_work.data_ptr(),
                 state_work.data_ptr(),
                 final_state.data_ptr(),
                 output.data_ptr(),
                 cu_work.data_ptr(),
                 workspace.defined() ? workspace.data_ptr() : nullptr,
                 static_cast<int32_t>(num_sequences),
                 static_cast<int32_t>(num_heads_),
                 static_cast<int32_t>(total_tokens)),
             0);
    return {output, output_final_state ? final_state : torch::Tensor()};
  }

  return forward_bangc();
}

}  // namespace xllm::kernel::mlu
