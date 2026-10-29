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

#include "kernels/mlu/bangc/include/bangc_kda_tensor.h"

#include <framework/core/MLUStream.h>
#include <glog/logging.h>

#include <algorithm>
#include <climits>
#include <cstdint>

#include "kernels/mlu/bangc/src/kda/launchers.h"
namespace xllm::kernel::mlu {
namespace {
bool integer_metadata(const torch::Tensor& tensor) {
  return tensor.scalar_type() == torch::kInt32 ||
         tensor.scalar_type() == torch::kInt64;
}
void same_device(const torch::Tensor& tensor, const torch::Device& device) {
  CHECK(tensor.defined());
  CHECK_EQ(tensor.device(), device);
}
}  // namespace
std::pair<torch::Tensor, torch::Tensor> bangc_kda_forward(
    const BangcKdaTensorParams& p) {
  CHECK(p.q.defined());
  const torch::Device device = p.q.device();
  const torch::DeviceGuard device_guard(device);
  same_device(p.k, device);
  same_device(p.v, device);
  same_device(p.gate, device);
  same_device(p.beta, device);
  same_device(p.initial_state, device);
  CHECK_EQ(p.q.dim(), 4);
  CHECK_EQ(p.q.sizes(), p.k.sizes());
  CHECK_EQ(p.v.dim(), 4);
  CHECK_EQ(p.v.size(0), p.q.size(0));
  CHECK_EQ(p.v.size(1), p.q.size(1));
  CHECK_GT(p.q.size(2), 0);
  CHECK_GT(p.v.size(2), 0);
  CHECK_EQ(p.v.size(2) % p.q.size(2), 0);
  CHECK_GT(p.q.size(3), 0);
  CHECK_GT(p.v.size(3), 0);
  CHECK_EQ(p.initial_state.dim(), 4);
  CHECK_EQ(p.initial_state.size(1), p.v.size(2));
  CHECK_EQ(p.initial_state.size(2), p.v.size(3));
  CHECK_EQ(p.initial_state.size(3), p.q.size(3));
  CHECK_GE(p.gate_mode, 0);
  CHECK_LE(p.gate_mode, 1);
  if (p.gate_mode == 0) {
    CHECK_LT(p.lower_bound, 0.f);
  }
  const int64_t batch = p.q.size(0);
  const int64_t length = p.q.size(1);
  const int64_t tokens = batch * length;
  const int64_t query_heads = p.q.size(2);
  const int64_t heads = p.v.size(2);
  const int64_t key_dimension = p.q.size(3);
  const int64_t value_dimension = p.v.size(3);
  CHECK_EQ(p.gate.numel(), tokens * heads * key_dimension);
  CHECK_EQ(p.beta.numel(), tokens * heads);
  same_device(p.a_log, device);
  same_device(p.bias, device);
  CHECK_EQ(p.a_log.numel(), heads);
  CHECK_EQ(p.bias.numel(), heads * key_dimension);
  torch::Tensor cu =
      p.cu.defined() ? p.cu.contiguous()
                     : torch::arange(0,
                                     (batch + 1) * std::max<int64_t>(length, 1),
                                     std::max<int64_t>(length, 1),
                                     p.q.options().dtype(torch::kInt64));
  same_device(cu, device);
  CHECK_EQ(cu.dim(), 1);
  CHECK_GE(cu.numel(), 1);
  CHECK(integer_metadata(cu));
  if (p.cu.defined()) {
    CHECK_EQ(batch, 1);
  }
  const int64_t sequences = cu.numel() - 1;
  torch::Tensor indices =
      p.indices.defined() ? p.indices.contiguous() : torch::Tensor();
  torch::Tensor accepted =
      p.accepted.defined() ? p.accepted.contiguous() : torch::Tensor();
  int64_t columns = 1;
  int64_t index_sequence_stride = 1;
  int64_t index_token_stride = 1;
  if (indices.defined()) {
    same_device(indices, device);
    CHECK(integer_metadata(indices));
    CHECK(indices.dim() == 1 || indices.dim() == 2);
    CHECK_GE(indices.size(0), sequences);
    index_sequence_stride = indices.stride(0);
    if (indices.dim() == 2) {
      CHECK_GT(indices.size(1), 0);
      columns = indices.size(1);
      index_token_stride = indices.stride(1);
    }
  }
  if (accepted.defined()) {
    same_device(accepted, device);
    CHECK(integer_metadata(accepted));
    CHECK_EQ(accepted.dim(), 1);
    CHECK_GE(accepted.numel(), sequences);
  }
  if (p.prefill) {
    CHECK_EQ(batch, 1);
    CHECK_EQ(p.initial_state.size(0), sequences);
    CHECK(!indices.defined());
    CHECK(!accepted.defined());
  }
  const int32_t state_mode = p.prefill
                                 ? 2
                                 : (p.inplace_final_state && indices.defined()
                                        ? (indices.dim() == 1 ? 3 : 1)
                                        : 0);
  torch::Tensor output = torch::zeros(p.v.sizes(), p.v.options());
  torch::Tensor returned_state =
      p.prefill
          ? torch::empty_like(p.initial_state,
                              p.initial_state.options(),
                              torch::MemoryFormat::Contiguous)
          : (p.inplace_final_state
                 ? p.initial_state
                 : torch::zeros({tokens, heads, value_dimension, key_dimension},
                                p.initial_state.options()));
  if (sequences == 0 || tokens == 0) {
    if (p.prefill) {
      returned_state.copy_(p.initial_state);
    }
    return {
        output,
        p.prefill && !p.output_final_state ? torch::Tensor() : returned_state};
  }
  torch::Tensor q = p.q.contiguous();
  torch::Tensor k = p.k.contiguous();
  torch::Tensor v = p.v.contiguous();
  torch::Tensor gate = p.gate.contiguous().to(torch::kFloat32);
  torch::Tensor beta = p.beta.contiguous().to(torch::kFloat32);
  torch::Tensor a_log = p.a_log.contiguous().to(torch::kFloat32);
  torch::Tensor bias = p.bias.contiguous().to(torch::kFloat32);
  torch::Tensor initial = p.initial_state.contiguous().to(torch::kFloat32);
  // Preserve every untouched original-dtype byte on in-place generic writes.
  const bool selective_copy = !p.prefill && p.inplace_final_state &&
                              returned_state.scalar_type() != torch::kFloat32;
  torch::Tensor final =
      returned_state.scalar_type() == torch::kFloat32 &&
              returned_state.is_contiguous()
          ? returned_state
          : (p.prefill || selective_copy
                 ? torch::zeros(
                       p.initial_state.sizes(),
                       p.initial_state.options().dtype(torch::kFloat32))
                 : returned_state.contiguous().to(torch::kFloat32));
  const bool narrow_metadata =
      cu.scalar_type() == torch::kInt32 &&
      (!indices.defined() || indices.scalar_type() == torch::kInt32) &&
      (!accepted.defined() || accepted.scalar_type() == torch::kInt32);
  const bool native128 =
      key_dimension == 128 && value_dimension == 128 &&
      q.scalar_type() == torch::kBFloat16 &&
      k.scalar_type() == torch::kBFloat16 &&
      v.scalar_type() == torch::kBFloat16 && !selective_copy &&
      narrow_metadata && sequences <= INT32_MAX && tokens <= INT32_MAX &&
      columns <= INT32_MAX && heads <= INT32_MAX / 512 &&
      query_heads <= INT32_MAX / 512 &&
      (!indices.defined() || index_sequence_stride == columns) &&
      (!indices.defined() || index_token_stride == 1);
  cnrtQueue_t queue = torch_mlu::getCurMLUStream(device.index());
  if (native128) {
    BangcKdaFullParams params{};
    params.q = q.data_ptr();
    params.k = k.data_ptr();
    params.v = v.data_ptr();
    params.gate = gate.data_ptr();
    params.beta = beta.data_ptr();
    params.a_log = a_log.defined() ? a_log.data_ptr<float>() : nullptr;
    params.bias = bias.defined() ? bias.data_ptr<float>() : nullptr;
    params.initial_state = initial.data_ptr<float>();
    params.final_state = final.data_ptr<float>();
    // Native output is BF16 even if the caller's state uses another dtype.
    params.output = output.data_ptr();
    params.cu = cu.data_ptr<int32_t>();
    params.indices = indices.defined() ? indices.data_ptr<int32_t>() : nullptr;
    params.accepted =
        accepted.defined() ? accepted.data_ptr<int32_t>() : nullptr;
    params.sequences = static_cast<int32_t>(sequences);
    params.query_heads = static_cast<int32_t>(query_heads);
    params.value_heads = static_cast<int32_t>(heads);
    params.columns = static_cast<int32_t>(columns);
    params.speculative = accepted.defined() ? 1 : 0;
    params.state_mode = state_mode;
    params.gate_mode = p.gate_mode;
    params.gate_dtype = 1;
    params.beta_dtype = 1;
    params.qk_l2norm = p.qk_l2norm ? 1 : 0;
    params.scale = p.scale;
    params.lower_bound = p.lower_bound;
    params.softplus_beta = p.softplus_beta;
    params.softplus_threshold = p.softplus_threshold;
    CHECK_EQ(bangc_kda_full_launch(static_cast<void*>(queue), &params), 0);
  } else {
    q = q.to(torch::kFloat32);
    k = k.to(torch::kFloat32);
    v = v.to(torch::kFloat32);
    torch::Tensor output_float =
        torch::zeros(p.v.sizes(), p.v.options().dtype(torch::kFloat32));
    torch::Tensor carry =
        torch::empty({32, key_dimension}, p.q.options().dtype(torch::kFloat32));
    torch::Tensor written_rows =
        selective_copy
            ? torch::zeros({final.size(0), heads, value_dimension},
                           p.initial_state.options().dtype(torch::kInt32))
            : torch::Tensor();
    BangcKdaGenericParams params{};
    params.q = q.data_ptr<float>();
    params.k = k.data_ptr<float>();
    params.v = v.data_ptr<float>();
    params.gate = gate.data_ptr<float>();
    params.beta = beta.data_ptr<float>();
    params.a_log = a_log.defined() ? a_log.data_ptr<float>() : nullptr;
    params.bias = bias.defined() ? bias.data_ptr<float>() : nullptr;
    params.initial_state = initial.data_ptr<float>();
    params.final_state = final.data_ptr<float>();
    params.output = output_float.data_ptr<float>();
    params.carry = carry.data_ptr<float>();
    params.cu = cu.data_ptr();
    params.indices = indices.defined() ? indices.data_ptr() : nullptr;
    params.accepted = accepted.defined() ? accepted.data_ptr() : nullptr;
    params.sequences = sequences;
    params.query_heads = query_heads;
    params.value_heads = heads;
    params.key_dimension = key_dimension;
    params.value_dimension = value_dimension;
    params.columns = columns;
    params.index_sequence_stride = index_sequence_stride;
    params.index_token_stride = index_token_stride;
    params.tokens = tokens;
    params.initial_slots = initial.size(0);
    params.final_slots = final.size(0);
    params.metadata_wide_mask =
        (cu.scalar_type() == torch::kInt64 ? 1 : 0) |
        (indices.defined() && indices.scalar_type() == torch::kInt64 ? 2 : 0) |
        (accepted.defined() && accepted.scalar_type() == torch::kInt64 ? 4 : 0);
    params.speculative = accepted.defined() ? 1 : 0;
    params.state_mode = state_mode;
    params.gate_mode = p.gate_mode;
    params.qk_l2norm = p.qk_l2norm ? 1 : 0;
    params.scale = p.scale;
    params.lower_bound = p.lower_bound;
    params.softplus_beta = p.softplus_beta;
    params.softplus_threshold = p.softplus_threshold;
    params.written_rows =
        written_rows.defined() ? written_rows.data_ptr<int32_t>() : nullptr;
    CHECK_EQ(bangc_kda_generic_launch(static_cast<void*>(queue), &params), 0);
    output.copy_(output_float);
    if (selective_copy) {
      // Keep the original-dtype snapshot separate from FP32 arithmetic.
      // Noncontiguous callers receive only a same-dtype layout copy afterward.
      torch::Tensor destination = returned_state.contiguous();
      torch::Tensor converted = final.to(returned_state.scalar_type());
      CHECK_EQ(
          bangc_kda_copy_written(static_cast<void*>(queue),
                                 destination.data_ptr(),
                                 converted.data_ptr(),
                                 written_rows.data_ptr<int32_t>(),
                                 written_rows.numel(),
                                 key_dimension * returned_state.element_size()),
          0);
      if (destination.data_ptr() != returned_state.data_ptr()) {
        returned_state.copy_(destination);
      }
    }
  }
  if (!selective_copy && final.data_ptr() != returned_state.data_ptr()) {
    returned_state.copy_(final);
  }
  return {
      output,
      p.prefill && !p.output_final_state ? torch::Tensor() : returned_state};
}
}  // namespace xllm::kernel::mlu
