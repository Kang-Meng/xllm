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

#include "kernels/mlu/bangc/include/kpool_bangc.h"

#include <glog/logging.h>

#include <limits>

#include "framework/core/MLUStream.h"
#include "kernels/mlu/bangc/src/kpool/launchers.h"

namespace xllm::kernel::mlu {
namespace {
constexpr int64_t kMaxInt32 = std::numeric_limits<int32_t>::max();

bool compatible(const torch::Tensor& tensor,
                torch::ScalarType dtype,
                const torch::Device& device) {
  return tensor.defined() && device.is_privateuseone() &&
         tensor.device() == device && tensor.scalar_type() == dtype &&
         tensor.is_contiguous();
}

bool integer_compatible(const torch::Tensor& tensor,
                        const torch::Device& device) {
  return tensor.defined() &&
         (tensor.scalar_type() == torch::kInt32 ||
          tensor.scalar_type() == torch::kInt64) &&
         compatible(tensor, tensor.scalar_type(), device);
}

struct CanonicalMetadata {
  torch::Tensor table;
  torch::Tensor positions;
  torch::Tensor rows;
  torch::Tensor starts;
};

CanonicalMetadata prepare_metadata(void* queue,
                                   const torch::Tensor& table,
                                   const torch::Tensor& positions,
                                   const torch::Tensor& rows,
                                   const torch::Tensor& starts,
                                   int32_t position_max,
                                   int32_t position_mode) {
  const auto prepare = [](const torch::Tensor& input, torch::ScalarType dtype) {
    return !input.defined() || input.scalar_type() == dtype
               ? input
               : torch::empty(input.sizes(), input.options().dtype(dtype));
  };
  CanonicalMetadata result{prepare(table, torch::kInt32),
                           prepare(positions, torch::kInt32),
                           prepare(rows, torch::kInt64),
                           prepare(starts, torch::kInt64)};
  const auto count = [](const torch::Tensor& input,
                        const torch::Tensor& output) -> int64_t {
    return input.defined() && input.data_ptr() != output.data_ptr()
               ? input.numel()
               : 0;
  };
  const int64_t tables = count(table, result.table);
  const int64_t tokens = count(positions, result.positions);
  const int64_t queries = count(rows, result.rows);
  const int64_t requests = count(starts, result.starts);
  if (tables == 0 && tokens == 0 && queries == 0 && requests == 0) {
    return result;
  }
  const auto data = [](const torch::Tensor& tensor) -> void* {
    return tensor.defined() ? tensor.data_ptr() : nullptr;
  };
  ::kpool_pack_integer_metadata(queue,
                                data(table),
                                data(result.table),
                                tables,
                                data(positions),
                                data(result.positions),
                                tokens,
                                data(rows),
                                data(result.rows),
                                queries,
                                data(starts),
                                data(result.starts),
                                requests,
                                position_max,
                                position_mode);
  return result;
}

bool state_ids_compatible(const torch::Tensor& ids,
                          const torch::Tensor& read_ids,
                          const torch::Device& device) {
  return ids.defined() && ids.dim() == 1 && ids.numel() > 0 &&
         ids.numel() <= kMaxInt32 &&
         (ids.scalar_type() == torch::kInt32 ||
          ids.scalar_type() == torch::kInt64) &&
         compatible(ids, ids.scalar_type(), device) &&
         integer_compatible(read_ids, device) &&
         read_ids.sizes() == ids.sizes();
}
struct CanonicalStateIds {
  torch::Tensor destination;
  torch::Tensor source;
};

CanonicalStateIds prepare_state_ids(const torch::Tensor& ids,
                                    const torch::Tensor& read_ids) {
  if (ids.scalar_type() == read_ids.scalar_type()) {
    return {ids, read_ids};
  }
  // Widen only: every int32 value is exactly represented in int64, and the
  // existing int64 tensor retains all sentinel and allocator-index bits.
  // Keep mixed metadata on the original pool arithmetic instead of the
  // generic Torch reduction/projection path, which has different rounding.
  return {ids.to(torch::kInt64), read_ids.to(torch::kInt64)};
}

// Bound allocation before any cache/tail mutation. Allocation and both kernels
// use the guarded current stream, so the caching allocator keeps its storage
// alive until all queued consumers on that stream have completed.
torch::Tensor make_tail_workspace(const torch::Tensor& tail,
                                  int64_t batch,
                                  bool required) {
  if (!required) {
    return {};
  }
  constexpr int64_t kBytesPerRing = 2 * 128 * sizeof(uint16_t);
  CHECK_GT(batch, 0);
  CHECK_GT(tail.size(2), 0);
  CHECK_LE(batch,
           std::numeric_limits<int64_t>::max() / kBytesPerRing / tail.size(2));
  // The device allocator enforces actual capacity and propagates OOM before
  // any kernel launch; no arbitrary workload-size limit is introduced.
  return torch::empty({batch, 2, tail.size(2), 128}, tail.options());
}

bool stash_workspace_required(int64_t batch, int64_t ring) {
  const int64_t slices = ring / 16 + (ring % 16 != 0);
  return batch > 1024 / slices;
}
}  // namespace

bool bangc_kpool_update(const torch::Tensor& key,
                        const torch::Tensor& gate,
                        const torch::Tensor& ape,
                        const torch::Tensor& matrix,
                        torch::Tensor& cache,
                        torch::Tensor& tail,
                        const torch::Tensor& ids,
                        const torch::Tensor& read_ids,
                        const torch::Tensor& table,
                        const torch::Tensor& positions,
                        const torch::Tensor& rows,
                        const torch::Tensor& starts,
                        int64_t block_size,
                        int64_t pool_size,
                        bool decode,
                        bool inherit) {
  const torch::Device device = key.device();
  if (block_size != 16 || pool_size != 4 || key.dim() != 2 ||
      key.size(1) != 128 || key.size(0) <= 0 || key.numel() > kMaxInt32 ||
      gate.sizes() != key.sizes() ||
      !compatible(key, torch::kBFloat16, device) ||
      !compatible(gate, torch::kBFloat16, device) ||
      !compatible(ape, torch::kFloat32, device) || ape.dim() != 2 ||
      ape.size(0) != 4 || ape.size(1) != 128 ||
      !compatible(matrix, torch::kFloat32, device) || matrix.dim() != 2 ||
      matrix.size(0) != 128 || matrix.size(1) != 128 ||
      !compatible(cache, torch::kBFloat16, device) || cache.dim() != 4 ||
      cache.size(1) != 1 || cache.size(2) != 4 || cache.size(3) != 128 ||
      cache.numel() > kMaxInt32 ||
      !compatible(tail, torch::kBFloat16, device) || tail.dim() != 4 ||
      tail.size(1) != 2 || tail.size(2) < 4 || tail.size(2) > kMaxInt32 ||
      tail.size(3) != 128 || !state_ids_compatible(ids, read_ids, device) ||
      !integer_compatible(table, device) || table.dim() != 2 ||
      table.size(0) != ids.numel() || table.size(1) <= 0 ||
      table.size(1) > kMaxInt32 / 16 || table.numel() > kMaxInt32 ||
      !integer_compatible(positions, device) ||
      positions.numel() != key.size(0) || !integer_compatible(rows, device) ||
      rows.numel() != key.size(0) || !integer_compatible(starts, device) ||
      starts.numel() != ids.numel() + 1) {
    return false;
  }
  const torch::DeviceGuard guard(device);
  const CanonicalStateIds state_ids = prepare_state_ids(ids, read_ids);
  const torch::Tensor& native_ids = state_ids.destination;
  const torch::Tensor& native_read_ids = state_ids.source;
  void* queue = static_cast<void*>(torch_mlu::getCurMLUStream(device.index()));
  const CanonicalMetadata metadata = prepare_metadata(
      queue,
      table,
      positions.scalar_type() == torch::kInt64 ? torch::Tensor{} : positions,
      rows,
      starts,
      static_cast<int32_t>(kMaxInt32),
      1);
  if (positions.scalar_type() == torch::kInt64) {
    torch::Tensor wide_workspace =
        make_tail_workspace(tail, native_ids.numel(), decode);
    const auto launch = native_ids.scalar_type() == torch::kInt32
                            ? ::kpool_update_wide_positions_i32
                            : ::kpool_update_wide_positions;
    launch(queue,
           key.data_ptr(),
           gate.data_ptr(),
           ape.data_ptr(),
           matrix.data_ptr(),
           cache.data_ptr(),
           tail.data_ptr(),
           native_ids.data_ptr(),
           native_read_ids.data_ptr(),
           metadata.table.data_ptr(),
           positions.data_ptr(),
           metadata.starts.data_ptr(),
           metadata.rows.data_ptr(),
           wide_workspace.defined() ? wide_workspace.data_ptr() : nullptr,
           static_cast<int32_t>(native_ids.numel()),
           static_cast<int32_t>(key.size(0)),
           static_cast<int32_t>(table.size(1)),
           static_cast<int32_t>(tail.size(2)),
           inherit ? 1 : 0,
           decode ? 0 : 1);
    if (!decode) {
      const CanonicalMetadata stash_metadata = prepare_metadata(
          queue, {}, positions, {}, {}, static_cast<int32_t>(tail.size(2)), 2);
      torch::Tensor stash_workspace = make_tail_workspace(
          tail,
          native_ids.numel(),
          stash_workspace_required(native_ids.numel(), tail.size(2)));
      ::kpool_stash_workspace(
          queue,
          key.data_ptr(),
          gate.data_ptr(),
          tail.data_ptr(),
          native_ids.data_ptr(),
          inherit ? native_read_ids.data_ptr() : native_ids.data_ptr(),
          stash_metadata.positions.data_ptr(),
          metadata.starts.data_ptr(),
          stash_workspace.defined() ? stash_workspace.data_ptr() : nullptr,
          static_cast<int32_t>(native_ids.numel()),
          static_cast<int32_t>(tail.size(2)),
          native_ids.scalar_type() == torch::kInt32 ? 32 : 64);
    }
    return true;
  }
  const int64_t batch = native_ids.numel();
  const int64_t ring = tail.size(2);
  const bool parallel = key.size(0) >= 128 && key.size(0) > batch * 32;
  const bool distinct_sources =
      inherit && native_read_ids.data_ptr() != native_ids.data_ptr();
  const bool saturated_short = batch >= 8 && key.size(0) <= batch * 64;
  const int64_t owned_slots = (batch / 32 + (batch % 32 != 0)) * ring;
  const bool cooperative = distinct_sources && batch > 1 && ring <= 64 &&
                           owned_slots <= 256 && (!parallel || saturated_short);
  const bool private_sources =
      !parallel && (!distinct_sources || (batch == 1 && ring <= 64));
  const bool required = decode ? (!cooperative && !private_sources)
                               : stash_workspace_required(batch, ring);
  torch::Tensor workspace =
      make_tail_workspace(tail, native_ids.numel(), required);
  void* workspace_data = workspace.defined() ? workspace.data_ptr() : nullptr;
  if (decode) {
    const auto launch = native_ids.scalar_type() == torch::kInt32
                            ? ::kpool_update_tokens_workspace_i32
                            : ::kpool_update_tokens_workspace;
    launch(queue,
           key.data_ptr(),
           gate.data_ptr(),
           ape.data_ptr(),
           matrix.data_ptr(),
           cache.data_ptr(),
           tail.data_ptr(),
           native_ids.data_ptr(),
           native_read_ids.data_ptr(),
           metadata.table.data_ptr(),
           metadata.positions.data_ptr(),
           metadata.starts.data_ptr(),
           workspace_data,
           static_cast<int32_t>(native_ids.numel()),
           static_cast<int32_t>(table.size(1)),
           static_cast<int32_t>(tail.size(2)),
           inherit ? 1 : 0,
           static_cast<int32_t>(key.size(0)));
  } else {
    const auto launch = native_ids.scalar_type() == torch::kInt32
                            ? ::kpool_prefill_workspace_i32
                            : ::kpool_prefill_workspace;
    launch(queue,
           key.data_ptr(),
           gate.data_ptr(),
           ape.data_ptr(),
           matrix.data_ptr(),
           cache.data_ptr(),
           tail.data_ptr(),
           native_ids.data_ptr(),
           native_read_ids.data_ptr(),
           metadata.table.data_ptr(),
           metadata.positions.data_ptr(),
           metadata.starts.data_ptr(),
           metadata.rows.data_ptr(),
           workspace_data,
           static_cast<int32_t>(native_ids.numel()),
           static_cast<int32_t>(key.size(0)),
           static_cast<int32_t>(table.size(1)),
           static_cast<int32_t>(tail.size(2)),
           inherit ? 1 : 0);
  }
  return true;
}

bool bangc_kpool_stash(const torch::Tensor& key,
                       const torch::Tensor& gate,
                       torch::Tensor& tail,
                       const torch::Tensor& ids,
                       const torch::Tensor& read_ids,
                       const torch::Tensor& positions,
                       const torch::Tensor& starts) {
  const torch::Device device = key.device();
  if (key.dim() != 2 || key.size(1) != 128 || key.numel() > kMaxInt32 ||
      gate.sizes() != key.sizes() ||
      !compatible(key, torch::kBFloat16, device) ||
      !compatible(gate, torch::kBFloat16, device) ||
      !compatible(tail, torch::kBFloat16, device) || tail.dim() != 4 ||
      tail.size(1) != 2 || tail.size(2) <= 0 || tail.size(2) > kMaxInt32 ||
      tail.size(3) != 128 || !state_ids_compatible(ids, read_ids, device) ||
      !integer_compatible(positions, device) ||
      positions.numel() != key.size(0) || !integer_compatible(starts, device) ||
      starts.numel() != ids.numel() + 1) {
    return false;
  }
  const torch::DeviceGuard guard(device);
  const CanonicalStateIds state_ids = prepare_state_ids(ids, read_ids);
  const torch::Tensor& native_ids = state_ids.destination;
  const torch::Tensor& native_read_ids = state_ids.source;
  void* queue = static_cast<void*>(torch_mlu::getCurMLUStream(device.index()));
  const CanonicalMetadata metadata = prepare_metadata(
      queue, {}, positions, {}, starts, static_cast<int32_t>(tail.size(2)), 2);
  torch::Tensor workspace = make_tail_workspace(
      tail,
      native_ids.numel(),
      stash_workspace_required(native_ids.numel(), tail.size(2)));
  ::kpool_stash_workspace(queue,
                          key.data_ptr(),
                          gate.data_ptr(),
                          tail.data_ptr(),
                          native_ids.data_ptr(),
                          native_read_ids.data_ptr(),
                          metadata.positions.data_ptr(),
                          metadata.starts.data_ptr(),
                          workspace.defined() ? workspace.data_ptr() : nullptr,
                          static_cast<int32_t>(native_ids.numel()),
                          static_cast<int32_t>(tail.size(2)),
                          native_ids.scalar_type() == torch::kInt32 ? 32 : 64);
  return true;
}

bool bangc_kpool_score(const torch::Tensor& query,
                       const torch::Tensor& weights,
                       const torch::Tensor& cache,
                       const torch::Tensor& table,
                       const torch::Tensor& positions,
                       const torch::Tensor& rows,
                       torch::Tensor& scores,
                       int64_t block_size,
                       int64_t pool_size,
                       double scale,
                       const torch::Tensor& starts) {
  const torch::Device device = query.device();
  if (block_size != 16 || pool_size != 4 || !(scale > 0.0) ||
      query.dim() != 3 || query.size(1) != 32 || query.size(2) != 128 ||
      query.size(0) <= 0 || query.numel() > kMaxInt32 ||
      !compatible(query, torch::kBFloat16, device) ||
      !compatible(weights, torch::kFloat32, device) || weights.dim() != 2 ||
      weights.size(0) != query.size(0) || weights.size(1) != 32 ||
      !compatible(cache, torch::kBFloat16, device) || cache.dim() != 4 ||
      cache.size(1) != 1 || cache.size(2) != 4 || cache.size(3) != 128 ||
      cache.size(0) > 4194304 || !integer_compatible(table, device) ||
      table.dim() != 2 || table.size(0) <= 0 || table.size(1) <= 0 ||
      table.numel() > kMaxInt32 || table.size(1) > kMaxInt32 / 16 ||
      !integer_compatible(positions, device) ||
      positions.numel() != query.size(0) || !integer_compatible(rows, device) ||
      rows.numel() != query.size(0) ||
      !compatible(scores, torch::kFloat32, device) || scores.dim() != 2 ||
      scores.size(0) != query.size(0) || scores.size(1) <= 0 ||
      scores.numel() > kMaxInt32) {
    return false;
  }
  const torch::DeviceGuard guard(device);
  void* queue = static_cast<void*>(torch_mlu::getCurMLUStream(device.index()));
  const CanonicalMetadata metadata = prepare_metadata(
      queue, table, positions, rows, {}, static_cast<int32_t>(kMaxInt32), true);
  // Pipelined score for decode shapes (at most eight queries per request):
  // consecutive queries of one request share a key gather. The window divides
  // the per-request query count of uniform batches so no window spans two
  // requests; such windows would still be scored correctly per query, so the
  // choice only affects speed. Odd requests of five or more queries share
  // keys better in the per-request kernel. Prefill shapes keep that kernel.
  const int64_t tokens = query.size(0);
  const int64_t requests = table.size(0);
  const int64_t per_request = tokens % requests == 0 ? tokens / requests : 1;
  int32_t window = 0;
  if (tokens <= requests * 8) {
    window = per_request % 4 == 0   ? 4
             : per_request % 2 == 0 ? 2
             : per_request >= 5     ? 0
                                    : 1;
  }
  if (window > 0 &&
      ::kpool_score_pipelined(queue,
                              query.data_ptr(),
                              weights.data_ptr(),
                              cache.data_ptr(),
                              metadata.table.data_ptr(),
                              metadata.positions.data_ptr(),
                              metadata.rows.data_ptr(),
                              scores.data_ptr(),
                              static_cast<int32_t>(tokens),
                              static_cast<int32_t>(scores.size(1)),
                              static_cast<int32_t>(table.size(1)),
                              static_cast<int32_t>(cache.size(0)),
                              static_cast<int32_t>(requests),
                              static_cast<float>(scale),
                              window)) {
    return true;
  }
  // Prefill shapes: pipelined kernel tiling each window's own live prefix.
  if (tokens > requests * 8 &&
      ::kpool_score_prefill(queue,
                            query.data_ptr(),
                            weights.data_ptr(),
                            cache.data_ptr(),
                            metadata.table.data_ptr(),
                            metadata.positions.data_ptr(),
                            metadata.rows.data_ptr(),
                            scores.data_ptr(),
                            static_cast<int32_t>(tokens),
                            static_cast<int32_t>(scores.size(1)),
                            static_cast<int32_t>(table.size(1)),
                            static_cast<int32_t>(cache.size(0)),
                            static_cast<int32_t>(requests),
                            static_cast<float>(scale))) {
    return true;
  }
  // Explicit rows determine each query's table. Request offsets do not prove
  // a common row or a maximum span, so they cannot select a shared-table path.
  const auto launch = query.size(0) < table.size(0) * 4
                          ? ::kpool_score_unshared_bounded
                          : ::kpool_score_bounded;
  launch(queue,
         query.data_ptr(),
         weights.data_ptr(),
         cache.data_ptr(),
         metadata.table.data_ptr(),
         metadata.positions.data_ptr(),
         metadata.rows.data_ptr(),
         scores.data_ptr(),
         static_cast<int32_t>(query.size(0)),
         static_cast<int32_t>(scores.size(1)),
         static_cast<int32_t>(table.size(1)),
         static_cast<int32_t>(cache.size(0)),
         static_cast<int32_t>(table.size(0)),
         static_cast<float>(scale));
  return true;
}

bool bangc_kpool_expand(const torch::Tensor& ids,
                        const torch::Tensor& positions,
                        const torch::Tensor& rows,
                        const torch::Tensor& table,
                        torch::Tensor& slots,
                        torch::Tensor& lengths,
                        int64_t block_size,
                        int64_t token_budget,
                        int64_t pool_size,
                        bool always_tail,
                        bool prefill) {
  const torch::Device device = ids.device();
  if (block_size != 16 || token_budget != 2048 || pool_size != 4 ||
      ids.dim() != 2 || ids.size(1) != 512 || ids.size(0) <= 0 ||
      ids.size(0) > kMaxInt32 / 2051 ||
      !compatible(ids, torch::kInt64, device) ||
      !integer_compatible(positions, device) ||
      positions.numel() != ids.size(0) || !integer_compatible(rows, device) ||
      rows.numel() != ids.size(0) || !integer_compatible(table, device) ||
      table.dim() != 2 || table.size(0) <= 0 || table.size(0) > kMaxInt32 ||
      table.size(1) <= 0 || table.size(1) > kMaxInt32 / 16 ||
      table.numel() > kMaxInt32 || !compatible(slots, torch::kInt32, device) ||
      slots.dim() != 2 || slots.size(0) != ids.size(0) ||
      slots.size(1) != 2051 || !compatible(lengths, torch::kInt32, device) ||
      lengths.numel() != ids.size(0)) {
    return false;
  }
  const torch::DeviceGuard guard(device);
  void* queue = static_cast<void*>(torch_mlu::getCurMLUStream(device.index()));
  const CanonicalMetadata metadata = prepare_metadata(
      queue, table, positions, rows, {}, static_cast<int32_t>(kMaxInt32), true);
  if (!prefill && ::kpool_expand_decode(queue,
                                        ids.data_ptr(),
                                        metadata.positions.data_ptr(),
                                        metadata.rows.data_ptr(),
                                        metadata.table.data_ptr(),
                                        slots.data_ptr(),
                                        lengths.data_ptr(),
                                        static_cast<int32_t>(ids.size(0)),
                                        static_cast<int32_t>(table.size(1)),
                                        static_cast<int32_t>(table.size(0)),
                                        always_tail ? 1 : 0)) {
    return true;
  }
  const auto launch =
      prefill ? ::kpool_prefill_expand_bounded : ::kpool_expand_bounded;
  launch(queue,
         ids.data_ptr(),
         metadata.positions.data_ptr(),
         metadata.rows.data_ptr(),
         metadata.table.data_ptr(),
         slots.data_ptr(),
         lengths.data_ptr(),
         static_cast<int32_t>(ids.size(0)),
         static_cast<int32_t>(table.size(1)),
         static_cast<int32_t>(table.size(0)),
         always_tail ? 1 : 0);
  return true;
}

bool bangc_kpool_cache_positions(const torch::Tensor& tail_ids,
                                 const torch::Tensor& rows,
                                 const torch::Tensor& positions,
                                 const torch::Tensor& slot_mapping,
                                 torch::Tensor& output) {
  const torch::Device device = positions.device();
  const int64_t tokens = positions.numel();
  if (tokens <= 0 || tokens > kMaxInt32 ||
      !compatible(positions, torch::kInt32, device) ||
      !compatible(rows, torch::kInt64, device) || rows.numel() != tokens ||
      !integer_compatible(tail_ids, device) || tail_ids.dim() != 1 ||
      tail_ids.numel() <= 0 || tail_ids.numel() > kMaxInt32 / 8 ||
      (slot_mapping.defined() && (!integer_compatible(slot_mapping, device) ||
                                  slot_mapping.numel() != tokens)) ||
      !compatible(output, torch::kInt32, device) || output.numel() != tokens) {
    return false;
  }
  const torch::DeviceGuard guard(device);
  void* queue = static_cast<void*>(torch_mlu::getCurMLUStream(device.index()));
  const auto bits = [](const torch::Tensor& tensor) {
    return tensor.scalar_type() == torch::kInt32 ? 32 : 64;
  };
  return ::kpool_cache_positions(
      queue,
      rows.data_ptr(),
      tail_ids.data_ptr(),
      bits(tail_ids),
      static_cast<int32_t>(tail_ids.numel()),
      positions.data_ptr(),
      slot_mapping.defined() ? slot_mapping.data_ptr() : nullptr,
      slot_mapping.defined() ? bits(slot_mapping) : 0,
      output.data_ptr(),
      static_cast<int32_t>(tokens));
}

namespace {
// Rows with more live pools than this may be selected by a whole cluster.
constexpr int32_t kLongPools = 12288;

// Selects over each row's live prefix (positions: int32 cache positions of
// the rows; null selects every pool).
bool select_rows(const torch::Tensor& scores,
                 const void* positions,
                 torch::Tensor& selected) {
  const torch::DeviceGuard guard(scores.device());
  void* queue =
      static_cast<void*>(torch_mlu::getCurMLUStream(scores.device().index()));
  const int32_t rows = static_cast<int32_t>(scores.size(0));
  const int32_t pools = static_cast<int32_t>(scores.size(1));
  const int32_t stride = static_cast<int32_t>(scores.stride(0));
  const int32_t count = static_cast<int32_t>(selected.size(1));
  // Short rows; then long or mixed rows (clusters for a few long rows,
  // single cores otherwise); then rows split over grid parts.
  if (::kpool_select_topk_short(queue,
                                scores.data_ptr(),
                                positions,
                                selected.data_ptr(),
                                rows,
                                pools,
                                stride,
                                count) ||
      ::kpool_select_topk_hybrid(queue,
                                 scores.data_ptr(),
                                 positions,
                                 selected.data_ptr(),
                                 rows,
                                 pools,
                                 stride,
                                 count,
                                 kLongPools)) {
    return true;
  }
  int32_t segment = 0;
  const int32_t parts = ::kpool_select_parts(rows, pools, count, &segment);
  if (parts <= 0) {
    return false;
  }
  torch::Tensor workspace;
  if (parts > 1) {
    // Candidate scores and IDs, then two int32 per part.
    workspace =
        torch::empty({static_cast<int64_t>(rows) * parts * (2 * count + 2)},
                     scores.options());
  }
  return ::kpool_select_topk_parts(queue,
                                   scores.data_ptr(),
                                   positions,
                                   selected.data_ptr(),
                                   parts > 1 ? workspace.data_ptr() : nullptr,
                                   rows,
                                   pools,
                                   stride,
                                   count,
                                   parts,
                                   segment);
}
}  // namespace

bool bangc_kpool_select(const torch::Tensor& scores, torch::Tensor& selected) {
  const torch::Device device = scores.device();
  if (scores.dim() != 2 || scores.scalar_type() != torch::kFloat32 ||
      scores.stride(1) != 1 || scores.size(0) <= 0 || scores.size(1) <= 0 ||
      scores.size(0) > kMaxInt32 || scores.stride(0) < scores.size(1) ||
      scores.stride(0) > kMaxInt32 ||
      !compatible(selected, torch::kInt64, device) || selected.dim() != 2 ||
      selected.size(0) != scores.size(0) || selected.size(1) <= 0 ||
      selected.size(1) > kMaxInt32) {
    return false;
  }
  return select_rows(scores, nullptr, selected);
}

bool bangc_kpool_select_live(const torch::Tensor& scores,
                             const torch::Tensor& positions,
                             torch::Tensor& selected) {
  const torch::Device device = scores.device();
  if (scores.dim() != 2 || scores.scalar_type() != torch::kFloat32 ||
      scores.stride(1) != 1 || scores.size(0) <= 0 || scores.size(1) <= 0 ||
      scores.size(0) > kMaxInt32 || scores.stride(0) < scores.size(1) ||
      scores.stride(0) > kMaxInt32 ||
      !compatible(positions, torch::kInt32, device) ||
      positions.numel() != scores.size(0) ||
      !compatible(selected, torch::kInt64, device) || selected.dim() != 2 ||
      selected.size(0) != scores.size(0) || selected.size(1) <= 0 ||
      selected.size(1) > kMaxInt32) {
    return false;
  }
  return select_rows(scores, positions.data_ptr(), selected);
}

bool bangc_kpool_rows(const torch::Tensor& starts, torch::Tensor& rows) {
  const torch::Device device = starts.device();
  const int64_t batch = starts.numel() - 1;
  if (starts.dim() != 1 || batch <= 0 || batch > kMaxInt32 ||
      rows.numel() <= 0 || rows.numel() > kMaxInt32 ||
      (starts.scalar_type() != torch::kInt32 &&
       starts.scalar_type() != torch::kInt64) ||
      !compatible(starts, starts.scalar_type(), device) ||
      !compatible(rows, starts.scalar_type(), device) ||
      (starts.scalar_type() == torch::kInt32 && (batch & (batch - 1)) != 0)) {
    return false;
  }
  const torch::DeviceGuard guard(device);
  ::kpool_rows(static_cast<void*>(torch_mlu::getCurMLUStream(device.index())),
               starts.data_ptr(),
               rows.data_ptr(),
               static_cast<int32_t>(batch),
               static_cast<int32_t>(rows.numel()),
               starts.scalar_type() == torch::kInt32 ? 32 : 64);
  return true;
}
bool bangc_kpool_tail_generic(const torch::Tensor& key,
                              const torch::Tensor& gate,
                              torch::Tensor& tail,
                              const torch::Tensor& ids,
                              const torch::Tensor& read_ids,
                              const torch::Tensor& positions,
                              const torch::Tensor& starts,
                              bool decode) {
  const torch::Device device = key.device();
  if (!compatible(key, torch::kBFloat16, device) ||
      !compatible(gate, torch::kBFloat16, device) ||
      !compatible(tail, torch::kBFloat16, device) ||
      !integer_compatible(ids, device) ||
      !integer_compatible(read_ids, device) || ids.dim() != 1 ||
      read_ids.dim() != 1 || ids.numel() <= 0 || ids.numel() > kMaxInt32 ||
      read_ids.numel() != ids.numel() ||
      !integer_compatible(positions, device) ||
      !integer_compatible(starts, device) || key.size(1) <= 0 ||
      key.size(1) > kMaxInt32 || tail.size(2) <= 0 ||
      tail.size(2) > kMaxInt32) {
    return false;
  }
  constexpr int64_t kMaxInt64 = std::numeric_limits<int64_t>::max();
  CHECK_LE(ids.numel(),
           kMaxInt64 / 2 / tail.size(2) / key.size(1) /
               static_cast<int64_t>(sizeof(uint16_t)));
  const torch::DeviceGuard guard(device);
  torch::Tensor workspace =
      torch::empty({ids.numel(), 2, tail.size(2), key.size(1)}, tail.options());
  int32_t flags = (ids.scalar_type() == torch::kInt64 ? 1 : 0) |
                  (positions.scalar_type() == torch::kInt64 ? 2 : 0) |
                  (starts.scalar_type() == torch::kInt64 ? 8 : 0) |
                  (decode ? 16 : 0) |
                  (ids.scalar_type() != read_ids.scalar_type() ? 32 : 0);
  ::kpool_generic_tail_workspace_bounded(
      static_cast<void*>(torch_mlu::getCurMLUStream(device.index())),
      key.data_ptr(),
      gate.data_ptr(),
      tail.data_ptr(),
      ids.data_ptr(),
      read_ids.data_ptr(),
      positions.data_ptr(),
      starts.data_ptr(),
      workspace.data_ptr(),
      static_cast<int32_t>(ids.numel()),
      static_cast<int32_t>(tail.size(2)),
      static_cast<int32_t>(key.size(1)),
      flags,
      tail.size(0));
  return true;
}

}  // namespace xllm::kernel::mlu
