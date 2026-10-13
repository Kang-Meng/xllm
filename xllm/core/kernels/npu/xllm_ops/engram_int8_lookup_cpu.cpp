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

#include <ATen/Parallel.h>
#include <torch/torch.h>

#include <algorithm>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

// Host CPU int8 lookup for the Engram HBM-offload tables: gathers int8 rows by
// id and applies the FP32 group-32 scale in place into the BF16 pinned staging
// buffer, then the NPU side copies it device-ward. The whole computation runs
// on the host (with an __aarch64__ NEON path), so it is a plain CPU operator
// and not an AICPU/AscendC kernel; this file therefore includes only torch/ATen
// headers and has no CANN/aclnn dependency. It is registered on the CPU device
// key in npu_ops_library.cpp.
//
// Ported from vllm-ascend (dsv41-latest @ e41e0885f) csrc/engram_cpu.cpp, via
// ops-transformer experimental/dsv41/engram_int8_lookup_cpu.

namespace xllm::kernel::npu {

namespace {

constexpr int64_t kGroupSize = 32;
constexpr int64_t kParallelMinRows = 256;
constexpr int64_t kParallelGrain = 128;
// Ceiling on the grain: in the extreme-batch regime the grain below would
// otherwise scale with numel/(4*num_threads). Capping it bounds the value
// handed to ``at::parallel_for`` once numel exceeds
// 4*kParallelGrainMax*num_threads (2048*num_threads). The grain affects the
// serial cutoff (``at::parallel_for`` runs a range in the calling thread when
// ``numel <= grain``) and the backend-dependent partitioning of the range (the
// Native and OpenMP backends chunk it differently), so this cap bounds the
// argument, not the actual callback range or dispatch count -- the actual
// chunks are not guaranteed to be a fixed function of the grain. Rows are
// independent, so it only partitions the work and never changes a result.
constexpr int64_t kParallelGrainMax = kParallelGrain * 4;

#if defined(__aarch64__)
uint16x4_t to_bfloat16(float32x4_t values) {
  const auto bits = vreinterpretq_u32_f32(values);
  const auto bias = vaddq_u32(vdupq_n_u32(0x7fff),
                              vandq_u32(vshrq_n_u32(bits, 16), vdupq_n_u32(1)));
  const uint16x4_t rounded = vshrn_n_u32(vaddq_u32(bits, bias), 16);
  // ACLE: ``vbsl_u16`` takes three ``uint16x4_t`` operands -- the mask is the
  // 32->16-bit narrowed f32 self-compare (an all-ones lane means the lane is a
  // real number), not a ``uint8x8_t``. The types are pinned so a future edit
  // cannot silently rely on the x86 build, which never compiles this branch.
  const uint16x4_t not_nan_mask = vmovn_u32(vceqq_f32(values, values));
  // Match c10::BFloat16's canonical NaN and round-to-nearest-even.
  return vbsl_u16(not_nan_mask, rounded, vdup_n_u16(0x7fc0));
}
#endif

}  // namespace

void engram_int8_lookup_cpu(const torch::Tensor& weight,
                            const torch::Tensor& scale,
                            const torch::Tensor& ids,
                            torch::Tensor& output) {
  // TORCH_CHECK (not glog's CHECK) throughout: a CHECK failure calls abort(),
  // which is neither catchable nor complete for the caller. TORCH_CHECK raises
  // a c10::Error the Python side can surface. The id range check below runs on
  // the calling thread before at::parallel_for, so the parallel region never
  // throws across the parallel runtime's unwind boundary.
  for (const auto& tensor : {weight, scale, ids, output}) {
    TORCH_CHECK(tensor.device().is_cpu() && tensor.is_contiguous(),
                "Engram CPU lookup requires contiguous CPU tensors");
  }
  TORCH_CHECK(
      weight.scalar_type() == torch::kChar &&
          scale.scalar_type() == torch::kFloat &&
          ids.scalar_type() == torch::kLong &&
          output.scalar_type() == torch::kBFloat16,
      "Engram CPU lookup expects INT8 weight, FP32 scale, INT64 IDs and "
      "BF16 output");
  TORCH_CHECK(weight.dim() == 2 && scale.dim() == 2 && ids.dim() == 1 &&
                  output.dim() == 2,
              "Invalid Engram CPU lookup dimensions");
  const int64_t rows = weight.size(0);
  const int64_t width = weight.size(1);
  const int64_t groups = width / kGroupSize;
  TORCH_CHECK(width > 0 && width % kGroupSize == 0 && scale.size(0) == rows &&
                  scale.size(1) == groups && output.size(0) == ids.numel() &&
                  output.size(1) == width,
              "Invalid Engram CPU lookup shapes");
  const auto* codes = weight.const_data_ptr<int8_t>();
  const auto* scales = scale.const_data_ptr<float>();
  const auto* indices = ids.const_data_ptr<int64_t>();
  auto* out = output.mutable_data_ptr<torch::BFloat16>();
  // Validate every id on the calling thread before entering the parallel
  // region: a throw inside an at::parallel_for worker crosses the parallel
  // runtime's unwind boundary, so the error cannot be delivered cleanly to the
  // caller. Defensive only: the Python caller clamps every id into
  // [0, part_num_embeddings) before it reaches the op's `ids` operand (see
  // EngramEmbedding._local_block_offload), so a row outside the shard is
  // unreachable in production and this guard only fires for a hand-built
  // misuse. The pre-pass reads `ids` one extra time (decode-level batches of
  // int64 ids).
  for (int64_t i = 0; i < ids.numel(); ++i) {
    const int64_t row = indices[i];
    TORCH_CHECK(row >= 0 && row < rows, "Engram CPU lookup ID outside shard");
  }
  // Tiny decode batches do not amortize the thread-pool handoff: the grain is
  // the whole range (``ids.numel()``) below ``kParallelMinRows``, and
  // ``at::parallel_for`` runs a range in the calling thread when
  // ``numel <= grain``, so those batches stay serial. At or above the threshold
  // the grain scales with the row count (``numel / (num_threads * 4)``) so a
  // very large batch is handed a coarser grain argument; the floor
  // (``kParallelGrain``) keeps the historical grain for moderate batches and
  // the ceiling (``kParallelGrainMax``) bounds it in the extreme-batch regime.
  // The grain affects the serial cutoff above and the backend-dependent
  // partitioning of the parallel range (the Native and OpenMP backends chunk it
  // differently), so the constants tune that argument rather than a fixed
  // actual callback range or dispatch count -- the actual chunks are not
  // guaranteed to be a fixed function of the grain. Row i is independent, so
  // the grain only partitions the work and never changes a result. The
  // rows-per-grain division is only meaningful in the parallel branch, so it is
  // computed there (the small batches keep their exact previous behaviour).
  // get_num_threads() is floored at 1 so the division below cannot divide by
  // zero on a host that reports no usable thread pool; a normal (>=1) thread
  // count is unchanged.
  const int64_t num_threads =
      std::max<int64_t>(1, static_cast<int64_t>(torch::get_num_threads()));
  const int64_t grain =
      ids.numel() < kParallelMinRows
          ? ids.numel()
          : std::min(std::max(kParallelGrain, ids.numel() / (num_threads * 4)),
                     kParallelGrainMax);
  at::parallel_for(0, ids.numel(), grain, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const int64_t row = indices[i];
      for (int64_t g = 0; g < groups; ++g) {
        const float factor = scales[row * groups + g];
#if defined(__aarch64__)
        for (int64_t j = 0; j < kGroupSize; j += 8) {
          const int64_t column = g * kGroupSize + j;
          const auto values = vmovl_s8(vld1_s8(codes + row * width + column));
          const auto low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(values)));
          const auto high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(values)));
          auto* destination =
              reinterpret_cast<uint16_t*>(out + i * width + column);
          vst1_u16(destination, to_bfloat16(vmulq_n_f32(low, factor)));
          vst1_u16(destination + 4, to_bfloat16(vmulq_n_f32(high, factor)));
        }
#else
        for (int64_t j = 0; j < kGroupSize; ++j) {
          const int64_t column = g * kGroupSize + j;
          out[i * width + column] =
              torch::BFloat16(float(codes[row * width + column]) * factor);
        }
#endif
      }
    }
  });
}

}  // namespace xllm::kernel::npu
