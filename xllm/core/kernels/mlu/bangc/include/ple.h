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

#pragma once

#include <torch/types.h>

namespace xllm::kernel::mlu::bangc {

// Decode contract: Q=4, B in {1,2,4,8,16,32}, rows=B*4. All tensors are
// contiguous on one MLU device. Mutable buffers must not overlap one another
// or read-only inputs. Calls enqueue on the caller's current device stream;
// they neither allocate tensors nor synchronize. Prepared weights are opaque,
// caller-owned, immutable layouts; source weight updates require preparation
// again and recapture of graphs that retain the previous prepared buffers.

// Lossless BF16 [12800,2560] projection layout preparation.
void ple_pack_projection_out(const torch::Tensor& weight,
                             torch::Tensor& packed_weight);

// FP32 scales[10240] -> BF16 affine_pair[4,2,2560]. Its two HC-local rows
// contain the rounded key*query product and rounded convolution scale.
void ple_prepare_head_out(const torch::Tensor& key_scale,
                          const torch::Tensor& query_scale,
                          const torch::Tensor& conv_scale,
                          torch::Tensor& affine_pair);

// Two explicit initialization kernels; taps remain FP32 and are not packed.
void ple_prepare_weights_out(const torch::Tensor& weight,
                             const torch::Tensor& key_scale,
                             const torch::Tensor& query_scale,
                             const torch::Tensor& conv_scale,
                             torch::Tensor& packed_weight,
                             torch::Tensor& affine_pair);

// BF16 embedding[rows,2560] -> BF16 kv[rows,12800].
void ple_projection_out(const torch::Tensor& embedding,
                        const torch::Tensor& packed_weight,
                        torch::Tensor& kv);

// BF16 kv/hidden -> BF16 values/normalized[rows,10240], using the explicitly
// prepared affine_pair. The B1 cooperative and larger-B paths share this ABI.
void ple_head_out(const torch::Tensor& kv,
                  const torch::Tensor& hidden,
                  const torch::Tensor& affine_pair,
                  torch::Tensor& values,
                  torch::Tensor& normalized);

// FP32 taps[4,10240], BF16 history[B,13,10240]. Only history rows9..12
// change, to the same-round normalized values; old9 rows remain unchanged.
// Accepted-prefix selection remains the caller's subsequent state operation.
void ple_tail_out(const torch::Tensor& values,
                  const torch::Tensor& normalized,
                  torch::Tensor& history,
                  const torch::Tensor& taps,
                  const torch::Tensor& hidden,
                  torch::Tensor& output);

// Three compute launches, with every argument checked before the first launch.
// kv/values/normalized are caller-supplied scratch and output is [rows,10240].
void ple_forward_out(const torch::Tensor& embedding,
                     const torch::Tensor& hidden,
                     const torch::Tensor& packed_weight,
                     const torch::Tensor& affine_pair,
                     const torch::Tensor& taps,
                     torch::Tensor& history,
                     torch::Tensor& kv,
                     torch::Tensor& values,
                     torch::Tensor& normalized,
                     torch::Tensor& output);

}  // namespace xllm::kernel::mlu::bangc
