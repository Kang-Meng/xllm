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

#include "kernels/mlu/chunk_kda_pcp.h"

#include <framework/core/MLUStream.h>
#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "kernels/mlu/bangc/include/bangc_kda_pcp.h"

namespace xllm::kernel::mlu {
namespace {

constexpr int32_t kDim = 128;
static_assert(std::is_standard_layout_v<BangcKdaPcpParams>);
static_assert(sizeof(BangcKdaPcpParams) == 240);
static_assert(offsetof(BangcKdaPcpParams, tokens) == 168);
static_assert(offsetof(BangcKdaPcpParams, block) == 200);
static_assert(offsetof(BangcKdaPcpParams, lower_bound) == 236);

void check_data(const torch::Tensor& tensor) {
  CHECK(tensor.defined());
  CHECK(tensor.device().is_privateuseone()) << "KDA PCP requires MLU tensors";
  CHECK(tensor.scalar_type() == torch::kFloat32 ||
        tensor.scalar_type() == torch::kBFloat16 ||
        tensor.scalar_type() == torch::kFloat16)
      << "KDA PCP supports FP32, BF16, and FP16 tensors";
}

void launch_bangc_prepare(cnrtQueue_t queue, const BangcKdaPcpParams* params) {
  if (params->heads > 32 && params->heads % 32 == 0 && params->chunks > 1) {
    bangc_kda_pcp_prepare_head_affinity(queue, params);
  } else {
    bangc_kda_pcp_prepare_canonical(queue, params);
  }
}

void launch_bangc_summary(cnrtQueue_t queue, const BangcKdaPcpParams* params) {
  if (params->heads >= 16 && params->heads % 16 == 0) {
    bangc_kda_pcp_summary_rows128(queue, params);
  } else {
    bangc_kda_pcp_summary(queue, params);
  }
}

void launch_bangc_replay(cnrtQueue_t queue, const BangcKdaPcpParams* params) {
  int64_t remainder = ((params->sequences % 16) * (params->heads % 16)) % 16;
  if (params->sequences > 0 && params->heads > 0 && remainder > 0 &&
      remainder <= 8) {
    bangc_kda_pcp_replay_rows32(queue, params);
  } else {
    bangc_kda_pcp_replay_rows64(queue, params);
  }
}
int32_t data_kind(const torch::Tensor& tensor) {
  if (tensor.scalar_type() == torch::kFloat32) {
    return 0;
  }
  if (tensor.scalar_type() == torch::kBFloat16) {
    return 1;
  }
  CHECK_EQ(tensor.scalar_type(), torch::kFloat16);
  return 2;
}

int32_t metadata_kind(const torch::Tensor& tensor) {
  if (tensor.scalar_type() == torch::kInt32) {
    return 0;
  }
  CHECK_EQ(tensor.scalar_type(), torch::kInt64);
  return 1;
}

void check_product(int64_t first, int64_t second) {
  CHECK_GE(first, 0);
  CHECK_GE(second, 0);
  CHECK(first == 0 || second <= std::numeric_limits<int64_t>::max() / first);
}

BangcKdaPcpParams factor_params(const ChunkKDAPcpPrepared& prepared) {
  BangcKdaPcpParams params{};
  params.inverse = prepared.lower_inverse.data_ptr<float>();
  params.w = prepared.w.data_ptr<float>();
  params.u = prepared.u.data_ptr<float>();
  params.qg = prepared.qg.data_ptr<float>();
  params.kg = prepared.kg.data_ptr<float>();
  params.aq = prepared.aq.data_ptr<float>();
  params.gate_cumsum = prepared.gate_cumsum.data_ptr<float>();
  params.cu = prepared.cu_seqlens.data_ptr();
  params.indices = prepared.chunk_indices.data_ptr();
  params.cu_kind = metadata_kind(prepared.cu_seqlens);
  params.indices_kind = metadata_kind(prepared.chunk_indices);
  params.tokens = prepared.v.size(1);
  params.chunks = prepared.chunks;
  params.heads = prepared.heads;
  params.sequences = prepared.sequences;
  params.block = prepared.chunk_size;
  params.v_kind = data_kind(prepared.v);
  return params;
}
}  // namespace

ChunkKDAPcpPrepared chunk_kda_pcp_prepare(const torch::Tensor& q,
                                          const torch::Tensor& k,
                                          const torch::Tensor& v,
                                          const torch::Tensor& raw_gate,
                                          const torch::Tensor& a_log,
                                          const torch::Tensor& dt_bias,
                                          float gate_lower_bound,
                                          const torch::Tensor& raw_beta,
                                          const torch::Tensor& cu_seqlens,
                                          const torch::Tensor& chunk_indices,
                                          int64_t chunk_size) {
  for (const auto& tensor : {q, k, v, raw_gate, raw_beta, a_log, dt_bias}) {
    check_data(tensor);
  }
  CHECK(chunk_size == 16 || chunk_size == 64);
  CHECK_EQ(q.dim(), 4);
  CHECK_EQ(q.size(0), 1);
  CHECK_EQ(q.size(3), 128);
  CHECK_EQ(k.sizes(), q.sizes());
  CHECK_EQ(v.sizes(), q.sizes());
  CHECK_EQ(raw_gate.numel(), q.numel());
  CHECK_EQ(raw_beta.numel(), q.size(1) * q.size(2));
  CHECK_EQ(cu_seqlens.dim(), 1);
  CHECK_GE(cu_seqlens.numel(), 1);
  CHECK_EQ(chunk_indices.dim(), 2);
  CHECK_EQ(chunk_indices.size(1), 2);
  CHECK(cu_seqlens.scalar_type() == torch::kInt32 ||
        cu_seqlens.scalar_type() == torch::kInt64)
      << "cu_seqlens must have int32 or int64 dtype";
  CHECK(chunk_indices.scalar_type() == torch::kInt32 ||
        chunk_indices.scalar_type() == torch::kInt64)
      << "chunk_indices must have int32 or int64 dtype";
  for (const auto& tensor :
       {k, v, raw_gate, a_log, dt_bias, raw_beta, cu_seqlens, chunk_indices}) {
    CHECK_EQ(tensor.device(), q.device());
  }
  CHECK_GE(a_log.numel(), q.size(2));
  CHECK_GE(dt_bias.numel(), q.size(2) * 128);
  const torch::DeviceGuard guard(q.device());
  const int64_t chunks = chunk_indices.size(0);
  const int64_t heads = q.size(2);
  const int64_t sequences = cu_seqlens.numel() - 1;
  check_product(chunks, heads);
  check_product(chunks * heads, chunk_size * 128 * sizeof(float));
  check_product(sequences, heads);
  check_product(sequences * heads, 256 * 128 * sizeof(float));
  auto fp32 = q.options().dtype(torch::kFloat32);
  ChunkKDAPcpPrepared prepared;
  prepared.v = v;
  prepared.chunks = chunks;
  prepared.heads = heads;
  prepared.sequences = sequences;
  prepared.chunk_size = static_cast<int32_t>(chunk_size);
  prepared.summary = torch::empty({sequences, heads, 256, 128}, fp32);
  if (chunks == 0) {
    prepared.summary.zero_();
    prepared.summary.slice(/*dim=*/2, kDim, 2 * kDim)
        .copy_(torch::eye(kDim, fp32));
    return prepared;
  }
  prepared.lower_inverse =
      torch::empty({chunks, heads, chunk_size, chunk_size}, fp32);
  prepared.aq = torch::empty_like(prepared.lower_inverse);
  prepared.w = torch::empty({chunks, heads, chunk_size, 128}, fp32);
  prepared.u = torch::empty({chunks, heads, 128, chunk_size}, fp32);
  prepared.qg = torch::empty_like(prepared.w);
  prepared.kg = torch::empty_like(prepared.w);
  prepared.gate_cumsum = torch::empty({chunks, heads, 128}, fp32);
  prepared.cu_seqlens = cu_seqlens.contiguous().to(torch::kInt32);
  prepared.chunk_indices = chunk_indices.contiguous().to(torch::kInt32);
  const auto q_work = q.contiguous();
  const auto k_work = k.contiguous();
  const auto v_work = v.contiguous();
  const auto gate_work = raw_gate.contiguous();
  const auto beta_work = raw_beta.contiguous();
  // Convert gate parameters to FP32 before arithmetic.
  const auto log_work = a_log.to(torch::kFloat32).contiguous();
  const auto bias_work = dt_bias.to(torch::kFloat32).contiguous();
  auto params = factor_params(prepared);
  params.q = q_work.data_ptr();
  params.k = k_work.data_ptr();
  params.v = v_work.data_ptr();
  params.raw_gate = gate_work.data_ptr();
  params.raw_beta = beta_work.data_ptr();
  params.a_log = log_work.data_ptr<float>();
  params.dt_bias = bias_work.data_ptr<float>();
  params.q_kind = data_kind(q_work);
  params.k_kind = data_kind(k_work);
  params.v_kind = data_kind(v_work);
  params.gate_kind = data_kind(gate_work);
  params.beta_kind = data_kind(beta_work);
  params.lower_bound = gate_lower_bound;
  params.summary = prepared.summary.data_ptr<float>();
  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  if (chunks > 0 && heads > 0) {
    launch_bangc_prepare(queue, &params);
  }
  if (sequences > 0 && heads > 0) {
    launch_bangc_summary(queue, &params);
  }
  return prepared;
}
std::tuple<torch::Tensor, torch::Tensor> chunk_kda_pcp_replay(
    const ChunkKDAPcpPrepared& prepared,
    const torch::Tensor& initial_state) {
  check_data(prepared.v);
  check_data(initial_state);
  CHECK_EQ(initial_state.device(), prepared.v.device());
  if (prepared.chunks == 0) {
    return {torch::empty_like(prepared.v), initial_state};
  }
  CHECK_EQ(initial_state.dim(), 4);
  CHECK_EQ(initial_state.size(0), prepared.sequences);
  CHECK_EQ(initial_state.size(1), prepared.heads);
  CHECK_EQ(initial_state.size(2), 128);
  CHECK_EQ(initial_state.size(3), 128);
  const torch::DeviceGuard guard(initial_state.device());
  const auto initial = initial_state.to(torch::kFloat32).contiguous();
  auto final = torch::empty(initial.sizes(), initial.options());
  auto output = torch::empty(prepared.v.sizes(), prepared.v.options());
  auto params = factor_params(prepared);
  params.initial = initial.data_ptr<float>();
  params.final_state = final.data_ptr<float>();
  params.output = output.data_ptr();
  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  launch_bangc_replay(queue, &params);
  return {output, final.to(initial_state.scalar_type())};
}
torch::Tensor chunk_kda_pcp_merge(const torch::Tensor& gathered_summary,
                                  const torch::Tensor& prefix_state,
                                  int32_t cp_rank,
                                  int64_t local_token_count) {
  check_data(gathered_summary);
  check_data(prefix_state);
  CHECK_EQ(gathered_summary.dim(), 5);
  CHECK_EQ(prefix_state.dim(), 4);
  CHECK_EQ(gathered_summary.size(1), prefix_state.size(0));
  CHECK_EQ(gathered_summary.size(2), prefix_state.size(1));
  CHECK_EQ(gathered_summary.size(3), 256);
  CHECK_EQ(gathered_summary.size(4), 128);
  CHECK_EQ(prefix_state.size(2), 128);
  CHECK_EQ(prefix_state.size(3), 128);
  CHECK_GE(cp_rank, 0);
  CHECK_LT(cp_rank, gathered_summary.size(0));
  CHECK_GE(local_token_count, 0);
  CHECK_EQ(gathered_summary.device(), prefix_state.device());
  const torch::DeviceGuard guard(prefix_state.device());
  const auto gathered = gathered_summary.to(torch::kFloat32).contiguous();
  const auto initial = prefix_state.to(torch::kFloat32).contiguous();
  auto output = torch::empty(initial.sizes(), initial.options());
  BangcKdaPcpParams params{};
  params.sequences = prefix_state.size(0);
  params.heads = prefix_state.size(1);
  params.rank = cp_rank;
  params.initial = initial.data_ptr<float>();
  params.gathered = gathered.data_ptr<float>();
  params.final_state = output.data_ptr<float>();
  check_product(params.sequences, params.heads);
  check_product(params.sequences * params.heads, 256 * 128 * sizeof(float));
  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  bangc_kda_pcp_merge(queue, &params);
  return output.to(prefix_state.scalar_type());
}

torch::Tensor chunk_kda_pcp_summary(const torch::Tensor& q,
                                    const torch::Tensor& k,
                                    const torch::Tensor& v,
                                    const torch::Tensor& raw_gate,
                                    const torch::Tensor& a_log,
                                    const torch::Tensor& dt_bias,
                                    float gate_lower_bound,
                                    const torch::Tensor& raw_beta,
                                    const torch::Tensor& cu_seqlens,
                                    const torch::Tensor& chunk_indices,
                                    int64_t chunk_size) {
  return chunk_kda_pcp_prepare(q,
                               k,
                               v,
                               raw_gate,
                               a_log,
                               dt_bias,
                               gate_lower_bound,
                               raw_beta,
                               cu_seqlens,
                               chunk_indices,
                               chunk_size)
      .summary;
}

}  // namespace xllm::kernel::mlu
