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

#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include "kernels/mlu/bangc/include/bangc_kda.h"
#include "kernels/mlu/bangc/include/bangc_kda_tensor.h"
#include "kernels/mlu/mlu_ops_api.h"
#include "triton_jit/include/jit_kernel.h"

namespace xllm::kernel::mlu {

namespace {

// GLM5-Next KDA: scale = 1/sqrt(128), safe-gate lower bound = -5.
constexpr float kBangcKdaScale = 0.08838834764831845f;
constexpr float kBangcKdaGateLowerBound = -5.0f;

constexpr int64_t kMaxBlockHv = 32;
constexpr int64_t kMaxGdnBlockHv = 16;
constexpr int64_t kMinGdnBlockHv = 2;
constexpr int64_t kMaxBlockN = 4;
constexpr int64_t kBlockQueryLen = 4;

int64_t choose_block_hv(int64_t num_k_heads,
                        int64_t num_v_heads,
                        int64_t max_block_hv) {
  CHECK_GT(num_k_heads, 0);
  CHECK_GE(num_v_heads, num_k_heads);
  CHECK_EQ(num_v_heads % num_k_heads, 0) << "HV must be divisible by H";

  int64_t heads_per_query = num_v_heads / num_k_heads;
  int64_t candidate = std::min<int64_t>(
      (max_block_hv / heads_per_query) * heads_per_query, num_v_heads);
  for (; candidate >= heads_per_query; candidate -= heads_per_query) {
    if (num_v_heads % candidate == 0) {
      return candidate;
    }
  }
  LOG(FATAL) << "Failed to select BLOCK_HV for H=" << num_k_heads
             << ", HV=" << num_v_heads;
  return heads_per_query;
}

}  // namespace

using xllm::triton_jit::JITKernel;

std::pair<torch::Tensor, torch::Tensor> fused_sigmoid_gating_delta_rule_update(
    const torch::Tensor& A_log,
    torch::Tensor& a,
    torch::Tensor& b,
    const torch::Tensor& dt_bias,
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& initial_state,
    torch::Tensor& ssm_state_indices,
    torch::Tensor& cu_seqlens,
    double scale,
    bool use_qk_l2norm_in_kernel,
    float softplus_beta,
    float softplus_threshold,
    const std::optional<torch::Tensor>& num_accepted_tokens_opt,
    bool inplace_final_state,
    bool is_kda,
    bool kda_use_safe_gate,
    float kda_gate_lower_bound) {
  CHECK(q.device().is_privateuseone()) << "q must be an MLU tensor";
  CHECK_EQ(q.dim(), 4) << "q must be 4D [B, T, H, K]";
  CHECK_EQ(k.dim(), 4) << "k must be 4D [B, T, H, K]";
  CHECK_EQ(v.dim(), 4) << "v must be 4D [B, T, HV, V]";
  CHECK_EQ(A_log.dim(), 1) << "A_log must be 1D [HV]";
  CHECK_EQ(dt_bias.dim(), 1) << "dt_bias must be 1D [HV] or [HV * K]";
  CHECK_EQ(a.dim(), 2) << "a must be 2D [tokens, HV] or [tokens, HV * K]";
  CHECK_EQ(b.dim(), 2) << "b must be 2D [tokens, HV]";
  CHECK(!kda_use_safe_gate || is_kda) << "safe KDA gate requires is_kda=true";
  if (kda_use_safe_gate) {
    CHECK_LT(kda_gate_lower_bound, 0.0f)
        << "KDA gate lower bound must be negative";
  }

  q = q.contiguous();
  k = k.contiguous();
  v = v.contiguous();
  a = a.contiguous();
  b = b.contiguous();
  initial_state = initial_state;
  int64_t batch_size = k.size(0);
  int64_t seq_len = k.size(1);
  int64_t num_k_heads = k.size(2);
  int64_t head_k_dim = k.size(3);
  int64_t num_v_heads = v.size(2);
  int64_t head_v_dim = v.size(3);
  torch::Tensor query_start_loc =
      cu_seqlens.defined()
          ? cu_seqlens.contiguous()
          : torch::arange(
                0,
                (batch_size + 1) * std::max<int64_t>(seq_len, 1),
                std::max<int64_t>(seq_len, 1),
                torch::TensorOptions().dtype(torch::kInt32).device(q.device()));
  int64_t num_sequences = query_start_loc.size(0) - 1;
  if (cu_seqlens.defined()) {
    CHECK_EQ(batch_size, 1) << "cu_seqlens path expects flattened batch size 1";
  }
  CHECK_EQ(q.sizes(), k.sizes()) << "q/k shape mismatch";
  CHECK_EQ(v.size(0), batch_size) << "v batch dimension mismatch";
  CHECK_EQ(v.size(1), seq_len) << "v sequence dimension mismatch";
  CHECK_EQ(A_log.size(0), num_v_heads) << "A_log head dimension mismatch";
  CHECK_EQ(dt_bias.size(0), is_kda ? num_v_heads * head_k_dim : num_v_heads)
      << "dt_bias head dimension mismatch";
  CHECK_EQ(a.size(0), batch_size * seq_len) << "a token dimension mismatch";
  CHECK_EQ(b.size(0), batch_size * seq_len) << "b token dimension mismatch";
  CHECK_EQ(a.size(1), is_kda ? num_v_heads * head_k_dim : num_v_heads)
      << "a head dimension mismatch";
  CHECK_EQ(b.size(1), num_v_heads) << "b head dimension mismatch";
  CHECK_EQ(initial_state.dim(), 4)
      << "initial_state must be 4D [slots, HV, V, K]";
  CHECK_EQ(initial_state.size(1), num_v_heads)
      << "initial_state head dimension mismatch";
  CHECK_EQ(initial_state.size(2), head_v_dim)
      << "initial_state value dimension mismatch";
  CHECK_EQ(initial_state.size(3), head_k_dim)
      << "initial_state key dimension mismatch";

  // Invalid continuous-batching rows are intentionally skipped by the
  // kernels. Pre-zero the backing storage so skipped/padded rows cannot
  // expose uninitialized device memory.
  torch::Tensor out_storage =
      torch::zeros({1, batch_size, seq_len, num_v_heads, head_v_dim},
                   v.options().dtype(v.dtype()));
  torch::Tensor out = out_storage.select(/*dim=*/0, /*index=*/0);
  torch::Tensor final_state =
      inplace_final_state
          ? initial_state
          : torch::empty(
                {batch_size * seq_len, num_v_heads, head_v_dim, head_k_dim},
                initial_state.options());
  if (seq_len == 0 || num_sequences == 0) {
    return std::make_pair(out, final_state);
  }

  torch::Tensor num_accepted_tokens =
      num_accepted_tokens_opt.has_value()
          ? num_accepted_tokens_opt.value().contiguous()
          : torch::Tensor();
  torch::Tensor state_indices;
  if (ssm_state_indices.defined()) {
    state_indices = ssm_state_indices.contiguous();
  } else if (inplace_final_state) {
    torch::Tensor token_offsets =
        torch::arange(seq_len, query_start_loc.options());
    torch::Tensor seq_offsets =
        query_start_loc.slice(/*dim=*/0, /*start=*/0, /*end=*/num_sequences)
            .unsqueeze(/*dim=*/1);
    state_indices = seq_offsets + token_offsets.unsqueeze(/*dim=*/0);
  } else {
    state_indices = torch::zeros({num_sequences, 1}, query_start_loc.options());
  }
  if (ssm_state_indices.defined()) {
    CHECK(state_indices.dim() == 1 || state_indices.dim() == 2)
        << "state indices must be 1D or 2D";
    CHECK_GE(state_indices.size(0), num_sequences)
        << "state indices must cover each active sequence";
    if (state_indices.dim() == 2) {
      CHECK_GT(state_indices.size(1), 0)
          << "state index table must contain an initial slot";
    }
  }
  if (num_accepted_tokens_opt.has_value()) {
    CHECK_EQ(num_accepted_tokens.dim(), 1);
    CHECK_GE(num_accepted_tokens.size(0), num_sequences)
        << "accepted tokens must cover each active sequence";
  }
  int64_t stride_indices_seq = 1;
  int64_t stride_indices_tok = 1;
  if (state_indices.dim() == 1) {
    stride_indices_seq = state_indices.stride(0);
  } else {
    stride_indices_seq = state_indices.stride(0);
    stride_indices_tok = state_indices.stride(1);
  }

  const bool explicit_index_table =
      ssm_state_indices.defined() && state_indices.dim() == 2;
  const bool bangc_fp32_state =
      q.scalar_type() == torch::kBFloat16 &&
      k.scalar_type() == torch::kBFloat16 &&
      v.scalar_type() == torch::kBFloat16 &&
      a.scalar_type() == torch::kBFloat16 &&
      b.scalar_type() == torch::kBFloat16 &&
      initial_state.scalar_type() == torch::kFloat32 &&
      A_log.scalar_type() == torch::kFloat32 &&
      dt_bias.scalar_type() == torch::kFloat32;
  const int64_t bangc_columns =
      state_indices.dim() == 1 ? 1 : state_indices.size(1);
  const int64_t max_native_index = std::numeric_limits<int32_t>::max();
  const bool bangc_heads = num_v_heads == 8 || num_v_heads == 16 ||
                           num_v_heads == 32 || num_v_heads == 64;
  const bool bangc_capacity =
      bangc_heads && num_sequences > 0 && num_v_heads > 0 &&
      bangc_columns > 0 && num_v_heads <= max_native_index / 4 &&
      num_sequences <= max_native_index / (num_v_heads * 4) &&
      bangc_columns <= max_native_index / num_sequences &&
      seq_len <= max_native_index;
  const bool bangc_verify = num_accepted_tokens_opt.has_value();
  const bool bangc_checkpoint_layout = explicit_index_table || !bangc_verify;
  const auto integer_metadata = [](const torch::Tensor& tensor) {
    return tensor.scalar_type() == torch::kInt32 ||
           tensor.scalar_type() == torch::kLong;
  };
  const bool bangc_metadata_types =
      integer_metadata(query_start_loc) && integer_metadata(state_indices) &&
      (!bangc_verify || integer_metadata(num_accepted_tokens));
  // Select the specialized BangC schedule before any state mutation.
  const bool bangc_same_device =
      k.device() == q.device() && v.device() == q.device() &&
      a.device() == q.device() && b.device() == q.device() &&
      A_log.device() == q.device() && dt_bias.device() == q.device() &&
      initial_state.device() == q.device() &&
      query_start_loc.device() == q.device() &&
      state_indices.device() == q.device() &&
      (!num_accepted_tokens_opt.has_value() ||
       num_accepted_tokens.device() == q.device());
  const bool use_bangc =
      bangc_same_device && is_kda && kda_use_safe_gate &&
      use_qk_l2norm_in_kernel &&
      kda_gate_lower_bound == kBangcKdaGateLowerBound &&
      static_cast<float>(scale) == kBangcKdaScale && bangc_capacity &&
      bangc_fp32_state && num_k_heads == num_v_heads && head_k_dim == 128 &&
      head_v_dim == 128 && batch_size == 1 && ssm_state_indices.defined() &&
      inplace_final_state && initial_state.is_contiguous() &&
      A_log.is_contiguous() && dt_bias.is_contiguous() &&
      bangc_checkpoint_layout && bangc_metadata_types;
  if (use_bangc) {
    const torch::DeviceGuard device_guard(q.device());
    cnrtQueue_t queue = torch_mlu::getCurMLUStream(q.device().index());
    const auto launch_recurrent = seq_len <= max_native_index / num_v_heads
                                      ? bangc_kda_recurrent
                                      : bangc_kda_recurrent_wide;
    const int32_t wide_mask =
        (query_start_loc.scalar_type() == torch::kLong ? 1 : 0) |
        (state_indices.scalar_type() == torch::kLong ? 2 : 0) |
        (bangc_verify && num_accepted_tokens.scalar_type() == torch::kLong ? 4
                                                                           : 0);
    if (wide_mask != 0) {
      torch::Tensor native_cu = torch::empty(
          {num_sequences + 1}, query_start_loc.options().dtype(torch::kInt32));
      torch::Tensor native_indices =
          torch::empty({num_sequences, bangc_columns},
                       state_indices.options().dtype(torch::kInt32));
      torch::Tensor native_accepted =
          bangc_verify
              ? torch::empty({num_sequences},
                             num_accepted_tokens.options().dtype(torch::kInt32))
              : torch::Tensor();
      CHECK_EQ(bangc_kda_normalize_metadata(
                   static_cast<void*>(queue),
                   query_start_loc.data_ptr(),
                   state_indices.data_ptr(),
                   bangc_verify ? num_accepted_tokens.data_ptr() : nullptr,
                   native_cu.data_ptr(),
                   native_indices.data_ptr(),
                   bangc_verify ? native_accepted.data_ptr() : nullptr,
                   static_cast<int32_t>(num_sequences),
                   static_cast<int32_t>(bangc_columns),
                   wide_mask),
               0);
      query_start_loc = std::move(native_cu);
      state_indices = std::move(native_indices);
      num_accepted_tokens = std::move(native_accepted);
    }
    CHECK_EQ(launch_recurrent(
                 static_cast<void*>(queue),
                 q.data_ptr(),
                 k.data_ptr(),
                 v.data_ptr(),
                 a.data_ptr(),
                 b.data_ptr(),
                 A_log.data_ptr(),
                 dt_bias.data_ptr(),
                 initial_state.data_ptr(),
                 final_state.data_ptr(),
                 out.data_ptr(),
                 query_start_loc.data_ptr(),
                 state_indices.data_ptr(),
                 bangc_verify ? num_accepted_tokens.data_ptr() : nullptr,
                 static_cast<int32_t>(num_sequences),
                 static_cast<int32_t>(num_v_heads),
                 static_cast<int32_t>(bangc_columns),
                 /*speculative=*/bangc_verify ? 1 : 0,
                 /*inplace=*/explicit_index_table ? 1 : 3),
             0);
    return std::make_pair(out, final_state);
  }

  if (is_kda) {
    BangcKdaTensorParams parameters{};
    parameters.q = q;
    parameters.k = k;
    parameters.v = v;
    parameters.gate = a;
    parameters.beta = b;
    parameters.a_log = A_log;
    parameters.bias = dt_bias;
    parameters.initial_state = initial_state;
    parameters.cu = cu_seqlens;
    parameters.indices = ssm_state_indices;
    parameters.accepted = num_accepted_tokens;
    parameters.gate_mode = kda_use_safe_gate ? 0 : 1;
    parameters.inplace_final_state = inplace_final_state;
    parameters.output_final_state = true;
    parameters.qk_l2norm = use_qk_l2norm_in_kernel;
    parameters.scale = static_cast<float>(scale);
    parameters.lower_bound = kda_gate_lower_bound;
    parameters.softplus_beta = softplus_beta;
    parameters.softplus_threshold = softplus_threshold;
    return bangc_kda_forward(parameters);
  }

  query_start_loc = query_start_loc.to(torch::kInt32);
  state_indices = state_indices.to(torch::kInt32);
  if (num_accepted_tokens_opt.has_value()) {
    num_accepted_tokens = num_accepted_tokens.to(torch::kInt32);
  }
  std::optional<torch::Tensor> num_accepted_tokens_arg =
      num_accepted_tokens_opt.has_value()
          ? std::make_optional(num_accepted_tokens)
          : std::nullopt;

  torch_mlu::DeviceProp* prop =
      torch_mlu::getDeviceProperties(torch_mlu::current_device());
  CHECK(prop != nullptr);
  int64_t core_count = prop->cluster_count * prop->core_num_per_cluster;

  int64_t block_k = head_k_dim;
  int64_t block_v = std::min<int64_t>(head_v_dim, 128);
  int64_t block_n = 1;
  if (num_sequences > core_count * 2) {
    block_n = kMaxBlockN;
  } else if (num_sequences > core_count) {
    block_n = 2;
  }
  int64_t max_block_hv = std::min(kMaxBlockHv / block_n, kMaxGdnBlockHv);
  int64_t block_hv = choose_block_hv(num_k_heads, num_v_heads, max_block_hv);
  // MTP verification has only a few sequences. Give each Q/K head pair an
  // independent state tile so GDN can distribute value heads across MLU cores.
  const bool split_hv = num_v_heads > 1;
  if (split_hv) {
    const int64_t heads_per_query = num_v_heads / num_k_heads;
    block_hv = choose_block_hv(
        num_k_heads, num_v_heads, std::max(kMinGdnBlockHv, heads_per_query));
  }
  int64_t num_hv_blocks =
      split_hv ? (num_v_heads + block_hv - 1) / block_hv : 1;
  int64_t total_blocks = ((head_k_dim + block_k - 1) / block_k) *
                         ((head_v_dim + block_v - 1) / block_v) *
                         num_sequences * num_hv_blocks;

  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  JITKernel& f = JITKernel::get(
      /*py_path=*/
      "xllm.core.kernels.mlu.triton_kernel.fused_sigmoid_gating_delta_rule_"
      "update",
      /*fn_name=*/"tmo_fused_sigmoid_gating_delta_rule_update_kernel");

  f.launch(static_cast<void*>(queue),
           /*grid=*/
           {static_cast<uint32_t>(std::min(total_blocks, core_count)), 1, 1},
           /*cfg=*/{/*num_warps=*/1, /*num_stages=*/3},
           A_log,
           a,
           b,
           dt_bias,
           softplus_beta,
           softplus_threshold,
           q,
           k,
           v,
           out_storage,
           initial_state,
           final_state,
           query_start_loc,
           state_indices,
           num_accepted_tokens_arg,
           static_cast<float>(scale),
           static_cast<int64_t>(num_sequences),
           static_cast<int64_t>(seq_len),
           static_cast<int32_t>(batch_size),
           static_cast<int32_t>(num_k_heads),
           static_cast<int32_t>(num_v_heads),
           /*BLOCK_HV=*/static_cast<int32_t>(block_hv),
           static_cast<int32_t>(head_k_dim),
           static_cast<int32_t>(head_v_dim),
           /*BK=*/static_cast<int32_t>(block_k),
           /*BV=*/static_cast<int32_t>(block_v),
           static_cast<int64_t>(initial_state.stride(0)),
           static_cast<int64_t>(final_state.stride(0)),
           stride_indices_seq,
           stride_indices_tok,
           /*USE_INITIAL_STATE=*/1,
           /*INPLACE_FINAL_STATE=*/inplace_final_state ? 1 : 0,
           /*USE_QK_L2NORM_IN_KERNEL=*/use_qk_l2norm_in_kernel ? 1 : 0,
           /*IS_VARLEN=*/cu_seqlens.defined() ? 1 : 0,
           /*IS_CONTINUOUS_BATCHING=*/ssm_state_indices.defined() ? 1 : 0,
           /*IS_SPEC_DECODING=*/num_accepted_tokens_opt.has_value() ? 1 : 0,
           /*SPLIT_HV=*/split_hv ? 1 : 0,
           /*BLOCK_N=*/static_cast<int32_t>(block_n),
           /*BLOCK_QUERY_LEN=*/static_cast<int32_t>(kBlockQueryLen),
           /*FACTORED_REDUCE=*/head_k_dim == 128 ? 1 : 0);

  return std::make_pair(out, final_state);
}

}  // namespace xllm::kernel::mlu
