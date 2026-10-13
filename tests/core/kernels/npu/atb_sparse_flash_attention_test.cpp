/* Copyright 2026 The xLLM Authors.

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

// NPU acceptance test for the atb SparseFlashAttentionOperation adapter.
//
// Drives the real atb adapter (atb VariantPack -> aclnnSparseFlashAttention)
// with the production parameter combination from the deepseekv2 sparse latent
// attention route and compares against a dense CPU reference. Inputs and
// tolerances mirror the python wrapper's sparse attention cases so the atb
// adapter and the python wrapper are validated against the same cases, which
// is the drop-in replacement contract between the two op paths.

#include <acl/acl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "atb/context.h"
#include "atb/types.h"
#include "operations/aclnn/ops/sparse_flash_attention_operation.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace {

constexpr int64_t kHeadDim = 512;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kQHeads = 8;
constexpr int64_t kKvHeads = 1;
constexpr int64_t kBlockSize = 16;
constexpr int64_t kTopk = 4;
constexpr int64_t kInputNum =
    9;  // query, key, value, sparse, block_table, seq_q, seq_kv, q_rope, k_rope
constexpr double kRtol = 2e-2;
constexpr double kAtol = 2e-2;
const std::vector<int64_t> kSeqQ = {2, 1};
const std::vector<int64_t> kSeqKv = {8, 4};

struct SparseCase {
  torch::Tensor query;        // [tq, hq, d], TND
  torch::Tensor key;          // TND [tk, hkv, d] or PA_BSND [pages, bs, hkv, d]
  torch::Tensor value;        // same layout as key
  torch::Tensor sparse;       // [tq, hkv, topk], int32
  torch::Tensor block_table;  // [b, 1], int32; undefined when non-paged
  torch::Tensor seq_q;        // [b], int32, prefix sums
  torch::Tensor seq_kv;       // [b], int32, prefix sums
  torch::Tensor query_rope;   // [tq, hq, rope]
  torch::Tensor key_rope;     // TND [tk, hkv, rope] or PA_BSND pages
};

aclDataType to_acl_dtype(const torch::ScalarType type) {
  if (type == torch::kBFloat16) {
    return ACL_BF16;
  }
  if (type == torch::kInt32) {
    return ACL_INT32;
  }
  if (type == torch::kInt64) {
    return ACL_INT64;
  }
  if (type == torch::kFloat32) {
    return ACL_FLOAT;
  }
  LOG(FATAL) << "unsupported dtype for atb tensor: " << type;
  return ACL_DT_UNDEFINED;
}

atb::Tensor to_atb_tensor(const torch::Tensor& tensor) {
  atb::Tensor atb_tensor = {};
  atb_tensor.desc.dtype = to_acl_dtype(tensor.scalar_type());
  atb_tensor.desc.format = ACL_FORMAT_ND;
  const auto sizes = tensor.sizes();
  atb_tensor.desc.shape.dimNum = static_cast<uint64_t>(sizes.size());
  for (int64_t i = 0; i < sizes.size(); ++i) {
    atb_tensor.desc.shape.dims[i] = sizes[i];
  }
  atb_tensor.deviceData = tensor.data_ptr();
  atb_tensor.dataSize =
      static_cast<uint64_t>(tensor.numel() * tensor.element_size());
  return atb_tensor;
}

torch::Tensor prefix_sums(const std::vector<int64_t>& seq_lens) {
  auto out = torch::zeros({static_cast<int64_t>(seq_lens.size())},
                          torch::TensorOptions().dtype(torch::kInt32));
  int64_t total = 0;
  for (size_t i = 0; i < seq_lens.size(); ++i) {
    total += seq_lens[i];
    out[static_cast<int64_t>(i)] = static_cast<int32_t>(total);
  }
  return out;
}

SparseCase make_case(const std::vector<int64_t>& seq_q = kSeqQ,
                     const std::vector<int64_t>& seq_kv = kSeqKv) {
  torch::manual_seed(20260909);
  const int64_t tokens_q = std::accumulate(seq_q.begin(), seq_q.end(), 0LL);
  const int64_t tokens_kv = std::accumulate(seq_kv.begin(), seq_kv.end(), 0LL);
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);

  SparseCase c;
  c.query = torch::randn({tokens_q, kQHeads, kHeadDim}, bf16);
  c.key = torch::randn({tokens_kv, kKvHeads, kHeadDim}, bf16);
  // The kernel pairs V with K (MLA-absorb), so keep value identical to key.
  c.value = c.key.clone();
  c.query_rope = torch::randn({tokens_q, kQHeads, kRopeDim}, bf16);
  c.key_rope = torch::randn({tokens_kv, kKvHeads, kRopeDim}, bf16);
  c.sparse = torch::full({tokens_q, kKvHeads, kTopk},
                         -1,
                         torch::TensorOptions().dtype(torch::kInt32));
  c.seq_q = prefix_sums(seq_q);
  c.seq_kv = prefix_sums(seq_kv);

  int64_t token = 0;
  for (size_t batch = 0; batch < seq_kv.size(); ++batch) {
    const int64_t valid = std::min<int64_t>(kTopk, seq_kv[batch]);
    for (int64_t local_q = 0; local_q < seq_q[batch]; ++local_q) {
      for (int64_t j = 0; j < valid; ++j) {
        c.sparse.index({token, 0, j}) = static_cast<int32_t>(j);
      }
      ++token;
    }
  }
  return c;
}

SparseCase pack_pa(const SparseCase& t,
                   const std::vector<int64_t>& seq_kv = kSeqKv) {
  std::vector<torch::Tensor> pages;
  std::vector<torch::Tensor> value_pages;
  std::vector<torch::Tensor> rope_pages;
  auto block_table = torch::full({static_cast<int64_t>(seq_kv.size()), 1},
                                 -1,
                                 torch::TensorOptions().dtype(torch::kInt32));
  int64_t offset = 0;
  int64_t page_id = 0;
  for (const auto kv_len : seq_kv) {
    auto seq = t.key.slice(0, offset, offset + kv_len);
    auto val = t.value.slice(0, offset, offset + kv_len);
    auto rope = t.key_rope.slice(0, offset, offset + kv_len);
    const int64_t remainder = kv_len % kBlockSize;
    const int64_t pad = remainder == 0 ? 0 : kBlockSize - remainder;
    if (pad != 0) {
      seq = torch::cat(
          {seq, torch::zeros({pad, kKvHeads, kHeadDim}, seq.options())}, 0);
      val = torch::cat(
          {val, torch::zeros({pad, kKvHeads, kHeadDim}, val.options())}, 0);
      rope = torch::cat(
          {rope, torch::zeros({pad, kKvHeads, kRopeDim}, rope.options())}, 0);
    }
    pages.push_back(seq);
    value_pages.push_back(val);
    rope_pages.push_back(rope);
    block_table[page_id][0] = static_cast<int32_t>(page_id);
    ++page_id;
    offset += kv_len;
  }

  SparseCase packed = t;
  packed.key = torch::stack(pages, 0);
  packed.value = torch::stack(value_pages, 0);
  packed.key_rope = torch::stack(rope_pages, 0);
  packed.block_table = block_table;
  return packed;
}

SparseCase to_device(const SparseCase& t, const torch::Device& device) {
  SparseCase out = t;
  out.query = t.query.to(device);
  out.key = t.key.to(device);
  out.value = t.value.to(device);
  out.sparse = t.sparse.to(device);
  out.seq_q = t.seq_q.to(device);
  out.seq_kv = t.seq_kv.to(device);
  out.query_rope = t.query_rope.to(device);
  out.key_rope = t.key_rope.to(device);
  if (t.block_table.defined()) {
    out.block_table = t.block_table.to(device);
  }
  return out;
}

torch::Tensor sparse_reference(const SparseCase& t) {
  const auto query_f = t.query.to(torch::kFloat32);
  const auto key_f = t.key.to(torch::kFloat32);
  const auto value_f = t.value.to(torch::kFloat32);
  const auto q_rope_f = t.query_rope.to(torch::kFloat32);
  const auto k_rope_f = t.key_rope.to(torch::kFloat32);
  const double scale = std::pow(static_cast<double>(kHeadDim), -0.5);
  auto out = torch::zeros_like(query_f);

  const auto seq_q_ends = t.seq_q.to(torch::kCPU);
  const auto seq_kv_ends = t.seq_kv.to(torch::kCPU);
  const int64_t num_batches = seq_q_ends.numel();
  int64_t q_beg = 0;
  int64_t kv_beg = 0;
  for (int64_t batch = 0; batch < num_batches; ++batch) {
    const int64_t q_end = seq_q_ends[batch].item<int64_t>();
    const int64_t kv_end = seq_kv_ends[batch].item<int64_t>();
    for (int64_t q_pos = q_beg; q_pos < q_end; ++q_pos) {
      const auto idx = t.sparse.index({q_pos, 0}).to(torch::kLong);
      const auto valid = (idx >= 0) & (idx < (kv_end - kv_beg));
      const auto local = idx.clamp_min(0);
      const auto gathered_k = key_f.index_select(0, kv_beg + local).squeeze(1);
      const auto gathered_v =
          value_f.index_select(0, kv_beg + local).squeeze(1);
      const auto gathered_kr =
          k_rope_f.index_select(0, kv_beg + local).squeeze(1);
      auto scores = query_f.index({q_pos}).matmul(gathered_k.t()) * scale;
      scores = scores + q_rope_f.index({q_pos}).matmul(gathered_kr.t()) * scale;
      scores = scores.masked_fill(~valid.unsqueeze(0),
                                  -std::numeric_limits<double>::infinity());
      const auto max_s = std::get<0>(scores.max(/*dim=*/-1, /*keepdim=*/true));
      auto exp_s = (scores - max_s).exp().masked_fill(~valid.unsqueeze(0), 0.0);
      const auto sum_s = exp_s.sum(-1, /*keepdim=*/true).clamp_min(1e-12);
      out.index({q_pos}) = (exp_s / sum_s).matmul(gathered_v);
    }
    q_beg = q_end;
    kv_beg = kv_end;
  }
  return out.to(t.query.scalar_type());
}

class AtbContextGuard {
 public:
  atb::Context* ctx = nullptr;
  ~AtbContextGuard() {
    if (ctx != nullptr) {
      atb::DestroyContext(ctx);
    }
  }
};

class AclWorkspaceGuard {
 public:
  void* ptr = nullptr;
  ~AclWorkspaceGuard() {
    if (ptr != nullptr) {
      EXPECT_EQ(aclrtFree(ptr), ACL_SUCCESS);
    }
  }
};

bool run_adapter(const SparseCase& t,
                 const std::string& kv_layout,
                 const int64_t sparse_mode,
                 const bool has_block_table,
                 torch::Tensor& output) {
  atb_speed::common::SparseFlashAttentionParam param;
  param.queryLayout = "TND";
  param.kvLayout = kv_layout;
  param.scaleValue = std::pow(static_cast<double>(kHeadDim), -0.5);
  param.sparseBlockSize = 1;
  param.sparseMode = sparse_mode;
  param.hasBlockTable = has_block_table;
  atb_speed::common::SparseFlashAttentionOperation op(
      "atb_sparse_flash_attention_test", param);

  atb::VariantPack variant_pack;
  variant_pack.inTensors.resize(kInputNum);
  variant_pack.inTensors.at(0) = to_atb_tensor(t.query);
  variant_pack.inTensors.at(1) = to_atb_tensor(t.key);
  variant_pack.inTensors.at(2) = to_atb_tensor(t.value);
  variant_pack.inTensors.at(3) = to_atb_tensor(t.sparse);
  // Index 4 is the block_table slot; the adapter skips it when hasBlockTable
  // is false, so a default-constructed placeholder is never read.
  variant_pack.inTensors.at(4) =
      has_block_table ? to_atb_tensor(t.block_table) : atb::Tensor{};
  variant_pack.inTensors.at(5) = to_atb_tensor(t.seq_q);
  variant_pack.inTensors.at(6) = to_atb_tensor(t.seq_kv);
  variant_pack.inTensors.at(7) = to_atb_tensor(t.query_rope);
  variant_pack.inTensors.at(8) = to_atb_tensor(t.key_rope);

  output = torch::empty_like(t.query);
  variant_pack.outTensors.resize(1);
  variant_pack.outTensors.at(0) = to_atb_tensor(output);

  AtbContextGuard context_guard;
  if (atb::CreateContext(&context_guard.ctx) != atb::NO_ERROR) {
    ADD_FAILURE() << "atb CreateContext failed";
    return false;
  }
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  context_guard.ctx->SetExecuteStream(stream);

  uint64_t workspace_size = 0;
  if (op.Setup(variant_pack, workspace_size, context_guard.ctx) !=
      atb::NO_ERROR) {
    ADD_FAILURE() << "atb Setup failed";
    return false;
  }
  AclWorkspaceGuard workspace_guard;
  if (workspace_size > 0 &&
      aclrtMalloc(&workspace_guard.ptr,
                  workspace_size,
                  ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) {
    ADD_FAILURE() << "workspace alloc failed";
    return false;
  }
  if (op.Execute(variant_pack,
                 static_cast<uint8_t*>(workspace_guard.ptr),
                 workspace_size,
                 context_guard.ctx) != atb::NO_ERROR) {
    ADD_FAILURE() << "atb Execute failed";
    return false;
  }
  if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    ADD_FAILURE() << "aclrtSynchronizeStream failed";
    return false;
  }
  return true;
}

void expect_matches_reference(const torch::Tensor& result,
                              const SparseCase& cpu_case) {
  const auto result_f = result.to(torch::kCPU).to(torch::kFloat32);
  const auto reference = sparse_reference(cpu_case).to(torch::kFloat32);
  const auto max_diff = (result_f - reference).abs().max().item<double>();
  EXPECT_TRUE(torch::allclose(result_f, reference, kRtol, kAtol))
      << "atb adapter output mismatch: max diff " << max_diff
      << " exceeds rtol " << kRtol << " / atol " << kAtol;
}

TEST(AtbSparseFlashAttentionTest, PagedRopeTndPaBsndMode3MatchesReference) {
  const torch::Device device("npu:0");
  const SparseCase cpu_case = make_case();
  const SparseCase packed = pack_pa(cpu_case);
  const SparseCase device_case = to_device(packed, device);
  torch::Tensor result;
  ASSERT_TRUE(run_adapter(device_case,
                          "PA_BSND",
                          /*sparse_mode=*/3,
                          /*has_block_table=*/true,
                          result));
  expect_matches_reference(result, cpu_case);
}

TEST(AtbSparseFlashAttentionTest, NonPagedRopeTndTndMode3MatchesReference) {
  const torch::Device device("npu:0");
  const SparseCase cpu_case = make_case();
  const SparseCase device_case = to_device(cpu_case, device);
  torch::Tensor result;
  ASSERT_TRUE(run_adapter(device_case,
                          "TND",
                          /*sparse_mode=*/3,
                          /*has_block_table=*/false,
                          result));
  expect_matches_reference(result, cpu_case);
}

TEST(AtbSparseFlashAttentionTest,
     PagedRopeTndPaBsndMode3InvalidTailMatchesReference) {
  // The second batch's kv length (2) is shorter than topk (4): its sparse
  // index rows carry -1 tail padding (valid values in the front half, invalid
  // in the back half), exercising the op's invalid-index handling that the
  // full-length cases never trigger.
  const torch::Device device("npu:0");
  const std::vector<int64_t> short_kv = {8, 2};
  const SparseCase cpu_case = make_case(kSeqQ, short_kv);
  const SparseCase packed = pack_pa(cpu_case, short_kv);
  const SparseCase device_case = to_device(packed, device);
  torch::Tensor result;
  ASSERT_TRUE(run_adapter(device_case,
                          "PA_BSND",
                          /*sparse_mode=*/3,
                          /*has_block_table=*/true,
                          result));
  expect_matches_reference(result, cpu_case);
}

}  // namespace
