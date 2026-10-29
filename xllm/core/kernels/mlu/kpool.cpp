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

#include "kernels/mlu/kpool.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>

#include "kernels/mlu/bangc/include/kpool_bangc.h"

namespace xllm::kernel::mlu {
namespace {
constexpr int64_t kMaxInt32 = std::numeric_limits<int32_t>::max();
// These checks inspect metadata only; no device synchronization is required.
void check_tail_indices(const torch::Tensor& indices,
                        const torch::Tensor& tail,
                        int64_t requests) {
  CHECK(indices.defined());
  CHECK_EQ(indices.dim(), 1) << "tail indices must be one-dimensional";
  CHECK_EQ(indices.numel(), requests);
  CHECK(indices.scalar_type() == torch::kInt32 ||
        indices.scalar_type() == torch::kInt64)
      << "tail indices must have integer dtype";
  CHECK(indices.device() == tail.device());
}

void check_tail_sizes(const torch::Tensor& tail, int64_t requests) {
  CHECK_GT(tail.size(3), 0);
  CHECK_LE(tail.size(3), kMaxInt32);
  CHECK_GT(tail.size(2), 0);
  CHECK_LE(tail.size(2), kMaxInt32);
  CHECK_LE(requests, kMaxInt32);
  // The generic BangC path snapshots one ring per request. Bound its byte
  // count before any cache or tail mutation, without reading device data.
  CHECK_LE(requests,
           std::numeric_limits<int64_t>::max() / 2 / tail.size(2) /
               tail.size(3) / static_cast<int64_t>(sizeof(uint16_t)));
}

}  // namespace

void update_kpool(const torch::Tensor& k,
                  const torch::Tensor& gate,
                  const torch::Tensor& ape,
                  const torch::Tensor& hadamard,
                  torch::Tensor& cache,
                  torch::Tensor& tail,
                  const torch::Tensor& tail_ids,
                  const torch::Tensor& table,
                  const torch::Tensor& positions,
                  const torch::Tensor& rows,
                  const torch::Tensor& starts,
                  int64_t block_size,
                  int64_t pool_size,
                  bool decode,
                  const torch::Tensor& read_tail_ids) {
  CHECK_EQ(pool_size, 4) << "MLU KPool requires pool_size=4";
  CHECK_EQ(block_size, 16) << "MLU KPool requires block_size=16";
  CHECK_LE(k.numel(), kMaxInt32);
  CHECK_LE(cache.numel(), kMaxInt32);
  CHECK_EQ(table.dim(), 2);
  CHECK_LE(table.numel(), kMaxInt32);
  CHECK_GT(table.size(1), 0);
  CHECK_LE(table.size(1), kMaxInt32 / 16);
  CHECK_EQ(k.dim(), 2);
  CHECK_EQ(k.scalar_type(), torch::kBFloat16);
  CHECK_EQ(gate.scalar_type(), torch::kBFloat16);
  CHECK_EQ(cache.scalar_type(), torch::kBFloat16);
  CHECK_EQ(tail.scalar_type(), torch::kBFloat16);
  CHECK_EQ(ape.dim(), 2);
  CHECK_EQ(ape.size(0), pool_size);
  CHECK_EQ(ape.size(1), k.size(1));
  CHECK_EQ(hadamard.dim(), 2);
  CHECK_EQ(hadamard.size(0), k.size(1));
  CHECK_EQ(hadamard.size(1), k.size(1));
  CHECK_EQ(table.size(0), tail_ids.numel());
  CHECK(k.sizes() == gate.sizes());
  CHECK_EQ(tail.dim(), 4);
  CHECK_EQ(tail.size(1), 2);
  CHECK_GE(tail.size(2), pool_size);
  CHECK_EQ(tail.size(3), k.size(1));
  CHECK_EQ(cache.dim(), 4);
  CHECK_EQ(cache.size(1), 1);
  CHECK_EQ(cache.size(2), block_size / pool_size);
  CHECK_EQ(cache.size(3), k.size(1));
  CHECK_EQ(k.size(1), 128) << "MLU KPool requires D=128";
  CHECK_EQ(hadamard.scalar_type(), torch::kFloat32);
  CHECK_LE(cache.size(0), kMaxInt32);
  CHECK_LE(table.size(1), kMaxInt32);
  check_tail_sizes(tail, tail_ids.numel());
  check_tail_indices(tail_ids, tail, tail_ids.numel());
  check_tail_indices(positions, tail, k.size(0));
  check_tail_indices(rows, tail, k.size(0));
  check_tail_indices(starts, tail, tail_ids.numel() + 1);
  CHECK(table.scalar_type() == torch::kInt32 ||
        table.scalar_type() == torch::kInt64)
      << "table must have integer dtype";
  CHECK_EQ(table.device(), tail.device());
  CHECK_EQ(k.device(), tail.device());
  CHECK_EQ(gate.device(), tail.device());
  CHECK_EQ(cache.device(), tail.device());
  CHECK_EQ(hadamard.device(), tail.device());
  CHECK(cache.is_contiguous());
  CHECK(tail.is_contiguous());
  if (read_tail_ids.defined()) {
    check_tail_indices(read_tail_ids, tail, tail_ids.numel());
  }
  CHECK(tail.device().is_privateuseone()) << "KPool requires MLU tensors";
  if (k.size(0) == 0) {
    return;
  }
  CHECK_GT(tail_ids.numel(), 0);
  const torch::DeviceGuard device_guard(k.device());
  const torch::Tensor raw = k.contiguous();
  const torch::Tensor gates = gate.contiguous();
  const torch::Tensor a = ape.to(k.device(), torch::kFloat32).contiguous();
  const torch::Tensor h = hadamard.contiguous();
  const torch::Tensor ids = tail_ids.contiguous();
  const torch::Tensor read_ids =
      read_tail_ids.defined() ? read_tail_ids.contiguous() : ids;
  CHECK_EQ(read_ids.numel(), ids.numel());
  const torch::Tensor pages = table.contiguous();
  const torch::Tensor pos = positions.contiguous();
  const torch::Tensor batch = rows.contiguous();
  const torch::Tensor offsets = starts.contiguous();
  CHECK(bangc_kpool_update(raw,
                           gates,
                           a,
                           h,
                           cache,
                           tail,
                           ids,
                           read_ids,
                           pages,
                           pos,
                           batch,
                           offsets,
                           block_size,
                           pool_size,
                           decode,
                           read_tail_ids.defined()))
      << "Unsupported MLU KPool update inputs";
}

void stash_kpool_from_source(const torch::Tensor& k,
                             const torch::Tensor& gate,
                             torch::Tensor& tail,
                             const torch::Tensor& tail_ids,
                             const torch::Tensor& read_tail_ids,
                             const torch::Tensor& positions,
                             const torch::Tensor& starts) {
  CHECK_EQ(k.dim(), 2);
  CHECK(k.sizes() == gate.sizes());
  CHECK_EQ(k.scalar_type(), torch::kBFloat16);
  CHECK_EQ(gate.scalar_type(), k.scalar_type());
  CHECK_EQ(k.stride(1), 1);
  CHECK_EQ(gate.stride(1), 1);
  CHECK_EQ(tail.dim(), 4);
  CHECK_EQ(tail.size(1), 2);
  CHECK_GT(tail.size(2), 0);
  CHECK_EQ(tail.size(3), k.size(1));
  CHECK_EQ(tail.scalar_type(), k.scalar_type());
  CHECK(tail.is_contiguous());
  CHECK(k.device() == tail.device());
  CHECK(gate.device() == tail.device());
  check_tail_indices(tail_ids, tail, tail_ids.numel());
  check_tail_indices(read_tail_ids, tail, tail_ids.numel());
  check_tail_indices(positions, tail, k.size(0));
  check_tail_indices(starts, tail, tail_ids.numel() + 1);
  CHECK(tail_ids.is_contiguous());
  CHECK(read_tail_ids.is_contiguous());
  CHECK(positions.is_contiguous());
  CHECK(starts.is_contiguous());
  check_tail_sizes(tail, tail_ids.numel());
  CHECK(tail.device().is_privateuseone()) << "KPool requires MLU tensors";
  if (tail_ids.numel() == 0) {
    return;
  }
  const torch::DeviceGuard device_guard(k.device());
  const torch::Tensor raw = k.contiguous();
  const torch::Tensor gates = gate.contiguous();
  if (bangc_kpool_stash(
          raw, gates, tail, tail_ids, read_tail_ids, positions, starts)) {
    return;
  }
  CHECK(bangc_kpool_tail_generic(
      raw, gates, tail, tail_ids, read_tail_ids, positions, starts, false))
      << "Unsupported KPool tail inputs";
}

void score_kpool(const torch::Tensor& query,
                 const torch::Tensor& weights,
                 const torch::Tensor& cache,
                 const torch::Tensor& table,
                 const torch::Tensor& positions,
                 const torch::Tensor& rows,
                 torch::Tensor& scores,
                 int64_t block_size,
                 int64_t pool_size,
                 double scale,
                 const torch::Tensor& query_starts) {
  CHECK_EQ(pool_size, 4) << "MLU KPool requires pool_size=4";
  CHECK_EQ(block_size, 16) << "MLU KPool requires block_size=16";
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(query.size(1), 32) << "MLU KPool requires H=32";
  CHECK_EQ(query.size(2), 128) << "MLU KPool requires D=128";
  CHECK_EQ(cache.dim(), 4);
  CHECK_EQ(cache.size(1), 1);
  CHECK_EQ(cache.size(2), 4);
  CHECK_EQ(scores.dim(), 2);
  CHECK_GT(scale, 0.0);
  CHECK_EQ(weights.dim(), 2);
  CHECK_EQ(positions.numel(), query.size(0));
  CHECK_EQ(rows.numel(), query.size(0));
  CHECK_EQ(cache.size(-1), query.size(2));
  CHECK(cache.is_contiguous());
  CHECK_EQ(table.dim(), 2);
  CHECK_EQ(weights.size(0), query.size(0));
  CHECK_EQ(weights.size(1), query.size(1));
  CHECK_EQ(scores.size(0), query.size(0));
  CHECK_EQ(scores.scalar_type(), torch::kFloat32);
  CHECK_EQ(scores.stride(1), 1);
  CHECK_LE(query.numel(), kMaxInt32);
  CHECK_LE(scores.numel(), kMaxInt32);
  if (scores.numel() == 0) {
    return;
  }
  const torch::DeviceGuard device_guard(query.device());
  const torch::Tensor q = query.contiguous();
  const torch::Tensor w = weights.contiguous();
  const torch::Tensor pages = table.contiguous();
  const torch::Tensor pos = positions.contiguous();
  const torch::Tensor batch = rows.contiguous();
  torch::Tensor output = scores.is_contiguous()
                             ? scores
                             : torch::empty(scores.sizes(), scores.options());
  CHECK(bangc_kpool_score(q,
                          w,
                          cache,
                          pages,
                          pos,
                          batch,
                          output,
                          block_size,
                          pool_size,
                          scale,
                          query_starts))
      << "Unsupported MLU KPool score inputs";
  if (!scores.is_contiguous()) {
    scores.copy_(output);
  }
}

torch::Tensor select_kpool(const torch::Tensor& scores, int64_t count) {
  CHECK_GE(count, 0);
  if (scores.size(0) > 0 && count > 0) {
    torch::Tensor selected = torch::empty(
        {scores.size(0), count}, scores.options().dtype(torch::kInt64));
    if (bangc_kpool_select(scores, selected)) {
      return selected;
    }
  }
  torch::Tensor result = torch::full(
      {scores.size(0), count}, -1, scores.options().dtype(torch::kInt64));
  const int64_t k = std::min(count, scores.size(1));
  if (scores.size(0) == 0 || k == 0) {
    return result;
  }
  const auto [values, ids] = torch::topk(scores, k, /*dim=*/-1);
  result.narrow(1, 0, k).copy_(torch::where(torch::isfinite(values), ids, -1));
  return result;
}

torch::Tensor select_kpool_live(const torch::Tensor& scores,
                                int64_t count,
                                const torch::Tensor& positions,
                                int64_t pool_size) {
  CHECK_GE(count, 0);
  CHECK_GT(pool_size, 0);
  // The native live-prefix kernel computes the valid pool count with / 4.
  if (pool_size == 4 && scores.size(0) > 0 && count > 0) {
    torch::Tensor selected = torch::empty(
        {scores.size(0), count}, scores.options().dtype(torch::kInt64));
    if (bangc_kpool_select_live(scores, positions, selected)) {
      return selected;
    }
  }
  return select_kpool(scores, count);
}

torch::Tensor kpool_cache_positions(const torch::Tensor& tail_ids,
                                    const torch::Tensor& rows,
                                    const torch::Tensor& positions,
                                    const torch::Tensor& slot_mapping,
                                    int64_t padding_id) {
  CHECK_EQ(rows.numel(), positions.numel());
  const torch::Tensor slots =
      slot_mapping.defined() ? slot_mapping.reshape({-1}) : slot_mapping;
  if (padding_id == 0 && positions.numel() > 0) {
    torch::Tensor output = torch::empty_like(positions);
    if (bangc_kpool_cache_positions(tail_ids, rows, positions, slots, output)) {
      return output;
    }
  }
  torch::Tensor active = tail_ids.index_select(0, rows) > padding_id;
  if (slots.defined()) {
    active = active & (slots > 0);
  }
  return torch::where(active, positions, torch::full_like(positions, -1));
}

KPoolSelection expand_kpool(const torch::Tensor& ids,
                            const torch::Tensor& positions,
                            const torch::Tensor& rows,
                            const torch::Tensor& table,
                            int64_t block_size,
                            int64_t token_budget,
                            int64_t pool_size,
                            bool always_tail,
                            bool prefill) {
  CHECK_EQ(pool_size, 4) << "MLU KPool requires pool_size=4";
  CHECK_EQ(block_size, 16) << "MLU KPool requires block_size=16";
  CHECK_EQ(token_budget, 2048) << "MLU KPool requires token_budget=2048";
  CHECK_EQ(ids.dim(), 2);
  CHECK_EQ(ids.size(1), 512) << "MLU KPool requires 512 selected pools";
  CHECK_LE(ids.size(0), kMaxInt32 / 2051);
  CHECK(ids.scalar_type() == torch::kInt32 ||
        ids.scalar_type() == torch::kInt64);
  CHECK_EQ(positions.dim(), 1);
  CHECK_EQ(rows.dim(), 1);
  CHECK_EQ(ids.size(0), positions.numel());
  CHECK_EQ(ids.size(0), rows.numel());
  CHECK_EQ(table.dim(), 2);
  CHECK_LE(table.numel(), kMaxInt32);
  CHECK_LE(table.size(1), kMaxInt32 / 16);
  CHECK_GT(table.size(1), 0);
  CHECK(ids.size(0) == 0 || table.size(0) > 0)
      << "Nonempty KPool queries require a nonempty block table";
  const auto options = ids.options().dtype(torch::kInt32);
  KPoolSelection result{torch::empty({ids.size(0), 2051}, options),
                        torch::empty({ids.size(0)}, options)};
  if (ids.size(0) == 0) {
    return result;
  }
  const torch::DeviceGuard device_guard(ids.device());
  const torch::Tensor pools = ids.to(torch::kInt64).contiguous();
  const torch::Tensor pos = positions.contiguous();
  const torch::Tensor batch = rows.contiguous();
  const torch::Tensor pages = table.contiguous();
  CHECK(bangc_kpool_expand(pools,
                           pos,
                           batch,
                           pages,
                           result.physical_slots,
                           result.context_lens,
                           block_size,
                           token_budget,
                           pool_size,
                           always_tail,
                           prefill))
      << "Unsupported MLU KPool expansion inputs";
  return result;
}

}  // namespace xllm::kernel::mlu
