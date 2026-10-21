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

#include <ATen/core/dispatch/Dispatcher.h>
#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/kernels/mlu/bangc/include/dflash2_conv_norm.h"
#include "core/kernels/mlu/bangc/include/dflash2_grouped_conv.h"
#include "core/kernels/mlu/bangc/include/dflash2_sample_path.h"
#include "core/kernels/mlu/dflash2_grouped_conv.h"
#include "core/layers/common/dflash2_grouped_conv.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/mlu/attention.h"
#include "core/layers/mlu/dflash2_context_kv.h"
#include "core/layers/mlu/dflash2_grouped_conv.h"
#include "core/platform/platform.h"

namespace xllm::layer {
namespace {

torch::Tensor reference_grouped_conv(const torch::Tensor& hidden_states,
                                     const torch::Tensor& delta,
                                     const torch::Tensor& base,
                                     int32_t block_size,
                                     int32_t num_groups,
                                     int32_t group_size,
                                     int32_t taps) {
  const int64_t num_tokens = hidden_states.size(0);
  torch::Tensor blocks =
      hidden_states.view({num_tokens, num_groups, group_size});
  torch::Tensor coefficients =
      base.view({1, taps, num_groups, group_size}) + delta.unsqueeze(-1);
  torch::Tensor output = coefficients.select(/*dim=*/1, /*index=*/0) * blocks;
  torch::Tensor positions = torch::arange(num_tokens,
                                          torch::TensorOptions()
                                              .dtype(torch::kLong)
                                              .device(hidden_states.device())) %
                            block_size;

  for (int32_t tap = 1; tap < taps; ++tap) {
    CHECK_GT(num_tokens, tap)
        << "DFlash2 convolution token count must exceed its tap offset.";
    // Aligned-slice add_ replaces the previous zeros+cat shift: the first
    // `tap` output rows would receive padding-block contributions (zero) and
    // are skipped, which the block-boundary mask below already guarantees.
    torch::Tensor valid =
        positions.slice(/*dim=*/0, /*start=*/tap, /*end=*/num_tokens)
            .ge(tap)
            .view({num_tokens - tap, 1, 1})
            .to(hidden_states.dtype());
    output.slice(/*dim=*/0, /*start=*/tap, /*end=*/num_tokens)
        .add_(coefficients.select(/*dim=*/1, /*index=*/tap)
                  .slice(/*dim=*/0, /*start=*/tap, /*end=*/num_tokens) *
              blocks.slice(/*dim=*/0, /*start=*/0, /*end=*/num_tokens - tap) *
              valid);
  }
  return output.flatten(/*start_dim=*/1);
}

ModelArgs make_args() {
  ModelArgs args;
  args.n_layers(2);
  args.hidden_size(16);
  args.head_dim(128);
  args.n_heads(32);
  args.n_kv_heads(8);
  args.max_position_embeddings(32);
  args.rope_theta(10000);
  args.rms_norm_eps(1e-5);
  return args;
}

torch::Tensor make_weight(int64_t offset) {
  return ((torch::arange(8 * 128 * 16, torch::kFloat32) + offset)
              .remainder(29) -
          14)
             .reshape({8 * 128, 16}) /
         32;
}

class DFlash2ContextKVTest
    : public ::testing::TestWithParam<std::pair<int32_t, int32_t>> {};

TEST_P(DFlash2ContextKVTest, WritesOnlySelectedSlotsWithReferenceNormAndRope) {
  const auto [tp_size, rank] = GetParam();
  const ModelArgs args = make_args();
  const auto options = torch::TensorOptions()
                           .dtype(torch::kBFloat16)
                           .device(torch::Device(Platform::type_torch(), 0));
  ModelContext context(
      ParallelArgs(rank, tp_size, nullptr), args, QuantArgs(), options);
  DFlash2ContextKV writer(context);
  const torch::Tensor norm =
      torch::linspace(0.5, 1.5, 128).to(torch::kBFloat16);
  for (int32_t layer = 0; layer < 2; ++layer) {
    const std::string prefix =
        "layers." + std::to_string(layer) + ".self_attn.";
    // Deliberately split K, V and norm across independent checkpoint shards.
    writer.load_state_dict(StateDict(
        {{prefix + "k_proj.weight", make_weight(layer).to(torch::kBFloat16)}}));
    writer.load_state_dict(
        StateDict({{prefix + "v_proj.weight",
                    make_weight(layer + 7).to(torch::kBFloat16)}}));
    writer.load_state_dict(StateDict({{prefix + "k_norm.weight", norm}}));
  }
  writer.finalize_loaded_weights();

  const int64_t heads = std::max<int64_t>(8 / tp_size, 1);
  const int64_t first_head = tp_size <= 8 ? rank * heads : rank / (tp_size / 8);
  std::vector<KVCache> caches;
  caches.reserve(2);
  for (int32_t layer = 0; layer < 2; ++layer) {
    caches.emplace_back(
        KVCacheTensors{torch::zeros({3, heads, 16, 128}, options),
                       torch::zeros({3, heads, 16, 128}, options)});
  }
  const torch::Tensor hidden =
      (torch::arange(48, torch::kFloat32).reshape({3, 16}).remainder(13) / 16)
          .to(torch::kBFloat16);
  const torch::Tensor positions = torch::tensor({2, 7, 4}, torch::kInt32);
  const torch::Tensor slots = torch::tensor({17, 34, 21}, torch::kInt32);
  ASSERT_TRUE(writer.write(hidden.to(options),
                           positions.to(options.device()),
                           slots.to(options.device()),
                           caches,
                           ModelInputParams()));

  const torch::Tensor inv_freq = torch::exp(
      -std::log(10000.0) * torch::arange(0, 128, 2, torch::kFloat32) / 128);
  const torch::Tensor angles =
      positions.to(torch::kFloat32).view({3, 1}) * inv_freq;
  const torch::Tensor cosine = torch::cat({angles.cos(), angles.cos()}, -1)
                                   .to(torch::kBFloat16)
                                   .to(torch::kFloat32)
                                   .unsqueeze(1);
  const torch::Tensor sine = torch::cat({angles.sin(), angles.sin()}, -1)
                                 .to(torch::kBFloat16)
                                 .to(torch::kFloat32)
                                 .unsqueeze(1);
  for (int32_t layer = 0; layer < 2; ++layer) {
    const auto project = [&](int64_t offset) {
      const torch::Tensor weight =
          make_weight(offset)
              .to(torch::kBFloat16)
              .slice(0, first_head * 128, (first_head + heads) * 128);
      return torch::matmul(hidden.to(torch::kFloat32),
                           weight.to(torch::kFloat32).t())
          .to(torch::kBFloat16)
          .to(torch::kFloat32)
          .view({3, heads, 128});
    };
    torch::Tensor key = project(layer);
    key = (key * torch::rsqrt(key.pow(2).mean(-1, true) + 1e-5) *
           norm.to(torch::kFloat32))
              .to(torch::kBFloat16)
              .to(torch::kFloat32);
    const torch::Tensor rotated =
        torch::cat({-key.slice(-1, 64, 128), key.slice(-1, 0, 64)}, -1);
    key = (key * cosine + rotated * sine)
              .to(torch::kBFloat16)
              .to(torch::kFloat32);
    const torch::Tensor value = project(layer + 7);
    torch::Tensor expected_k = torch::zeros({3, heads, 16, 128});
    torch::Tensor expected_v = torch::zeros_like(expected_k);
    for (int64_t row = 0; row < 3; ++row) {
      const int32_t slot = slots[row].item<int32_t>();
      expected_k.select(0, slot / 16).select(1, slot % 16).copy_(key[row]);
      expected_v.select(0, slot / 16).select(1, slot % 16).copy_(value[row]);
    }
    EXPECT_TRUE(
        torch::allclose(caches[layer].get_k_cache().cpu().to(torch::kFloat32),
                        expected_k,
                        0.02,
                        0.02));
    EXPECT_TRUE(
        torch::allclose(caches[layer].get_v_cache().cpu().to(torch::kFloat32),
                        expected_v,
                        0.02,
                        0.02));
  }
}

INSTANTIATE_TEST_SUITE_P(ShardedAndReplicatedHeads,
                         DFlash2ContextKVTest,
                         ::testing::Values(std::pair<int32_t, int32_t>{1, 0},
                                           std::pair<int32_t, int32_t>{2, 0},
                                           std::pair<int32_t, int32_t>{2, 1},
                                           std::pair<int32_t, int32_t>{16, 0},
                                           std::pair<int32_t, int32_t>{16, 1},
                                           std::pair<int32_t, int32_t>{16, 14},
                                           std::pair<int32_t, int32_t>{16,
                                                                       15}));

TEST(DFlash2ContextKVDeathTest, RejectsInvalidHeadTopologyBeforeAllocation) {
  ModelArgs args = make_args();
  args.n_heads(30);
  ModelContext context(
      ParallelArgs(0, 16, nullptr), args, QuantArgs(), torch::TensorOptions());
  EXPECT_DEATH(
      { DFlash2ContextKV writer(context); }, "args.n_heads\\(\\) % tp_size");
}

class DFlash2AttentionTest : public ::testing::TestWithParam<bool> {};

TEST_P(DFlash2AttentionTest,
       NonCausalWindowIsOptInAndRespectsSequenceBoundaries) {
  const auto options = torch::TensorOptions()
                           .dtype(torch::kBFloat16)
                           .device(torch::Device(Platform::type_torch(), 0));
  Attention attention(/*num_heads=*/1,
                      /*head_size=*/128,
                      /*scale=*/1.0f,
                      /*num_kv_heads=*/1,
                      /*sliding_window=*/3);
  AttentionMetadata metadata{};
  metadata.is_dummy = false;
  metadata.is_prefill = !GetParam();
  metadata.is_chunked_prefill = GetParam();
  metadata.block_table =
      torch::tensor({1, 2}, options.dtype(torch::kInt32)).reshape({2, 1});
  metadata.is_causal = false;  // Legacy callers did not control MLU causality.
  metadata.max_query_len = 4;
  metadata.max_seq_len = 4;
  metadata.q_cu_seq_lens =
      torch::tensor({0, 4, 8}, options.dtype(torch::kInt32));
  metadata.kv_cu_seq_lens = metadata.q_cu_seq_lens;
  metadata.slot_mapping = torch::tensor({16, 17, 18, 19, 32, 33, 34, 35},
                                        options.dtype(torch::kInt32));
  KVCache cache(KVCacheTensors{torch::zeros({3, 1, 16, 128}, options),
                               torch::zeros({3, 1, 16, 128}, options)});
  torch::Tensor query = torch::zeros({8, 128}, options);
  torch::Tensor key = torch::zeros_like(query);
  torch::Tensor value =
      torch::tensor({1., 2., 3., 4., 10., 20., 30., 40.}, options)
          .view({8, 1})
          .expand({8, 128})
          .contiguous();
  const torch::Tensor input_value = value;
  const torch::Tensor causal =
      std::get<0>(attention->forward(metadata, query, key, value, cache));
  const torch::Tensor expected_causal =
      torch::tensor({1., 1.5, 2., 3., 10., 15., 20., 30.});
  EXPECT_TRUE(torch::allclose(causal.cpu().to(torch::kFloat32).select(1, 0),
                              expected_causal,
                              0.01,
                              0.01));

  // Paged prefill rebinds key/value to the cache tensors. Each invocation
  // must receive fresh query-block inputs, as the decoder layer supplies.
  key = torch::zeros({8, 128}, options);
  value = input_value;
  metadata.non_causal_window_right = 3;
  const torch::Tensor bidirectional =
      std::get<0>(attention->forward(metadata, query, key, value, cache));
  const torch::Tensor expected_bidirectional =
      torch::tensor({2.5, 2.5, 2.5, 3., 25., 25., 25., 30.});
  EXPECT_TRUE(
      torch::allclose(bidirectional.cpu().to(torch::kFloat32).select(1, 0),
                      expected_bidirectional,
                      0.01,
                      0.01));
}

INSTANTIATE_TEST_SUITE_P(DenseAndPaged,
                         DFlash2AttentionTest,
                         ::testing::Bool());

TEST_P(DFlash2AttentionTest, ShorterBlocksDoNotReadPastTheirLogicalKVLength) {
  const auto options = torch::TensorOptions()
                           .dtype(torch::kBFloat16)
                           .device(torch::Device(Platform::type_torch(), 0));
  Attention attention(/*num_heads=*/1,
                      /*head_size=*/128,
                      /*scale=*/1.0f,
                      /*num_kv_heads=*/1,
                      /*sliding_window=*/2048);
  AttentionMetadata metadata{};
  metadata.is_dummy = false;
  metadata.is_prefill = !GetParam();
  metadata.is_chunked_prefill = GetParam();
  metadata.non_causal_window_right = 5;
  metadata.max_query_len = 6;  // One anchor and five candidate positions.
  metadata.max_seq_len = 6;
  metadata.q_cu_seq_lens =
      torch::tensor({0, 6, 12}, options.dtype(torch::kInt32));
  metadata.kv_cu_seq_lens = metadata.q_cu_seq_lens;
  metadata.block_table =
      torch::tensor({1, 2}, options.dtype(torch::kInt32)).reshape({2, 1});
  metadata.slot_mapping =
      torch::tensor({16, 17, 18, 19, 20, 21, 32, 33, 34, 35, 36, 37},
                    options.dtype(torch::kInt32));
  // Poison unused cache rows: a window extending past the six valid tokens
  // must not attend to stale speculative tokens or the adjacent sequence.
  KVCache cache(KVCacheTensors{torch::zeros({3, 1, 16, 128}, options),
                               torch::full({3, 1, 16, 128}, 1000, options)});
  torch::Tensor query = torch::zeros({12, 128}, options);
  torch::Tensor key = torch::zeros_like(query);
  torch::Tensor value =
      torch::tensor({1., 2., 3., 4., 5., 6., 10., 20., 30., 40., 50., 60.},
                    options)
          .view({12, 1})
          .expand({12, 128})
          .contiguous();
  const torch::Tensor output =
      std::get<0>(attention->forward(metadata, query, key, value, cache));
  const torch::Tensor expected = torch::tensor(
      {3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 35., 35., 35., 35., 35., 35.});
  EXPECT_TRUE(torch::allclose(
      output.cpu().to(torch::kFloat32).select(1, 0), expected, 0.01, 0.01));
}

TEST(DFlash2GroupedConvTest, FiveCandidateBlocksResetAtEachAnchor) {
  // Four six-row sequences: the total row count is also divisible by eight,
  // so using the trained block size would silently mix neighboring requests.
  const torch::Tensor hidden = torch::arange(1, 25, torch::kFloat32)
                                   .view({24, 1})
                                   .expand({24, 4})
                                   .contiguous();
  const torch::Tensor output = dflash2_grouped_conv(hidden,
                                                    torch::zeros({24, 2, 2}),
                                                    torch::ones({2, 4}),
                                                    /*block_size=*/6,
                                                    /*num_groups=*/2,
                                                    /*group_size=*/2,
                                                    /*taps=*/2);
  const torch::Tensor expected =
      torch::tensor({1.,  3.,  5.,  7.,  9.,  11., 7.,  15.,
                     17., 19., 21., 23., 13., 27., 29., 31.,
                     33., 35., 19., 39., 41., 43., 45., 47.})
          .view({24, 1})
          .expand({24, 4});
  EXPECT_TRUE(torch::equal(output, expected));
}

class DFlash2GroupedConvDeviceTest
    : public ::testing::TestWithParam<std::tuple<torch::ScalarType, int32_t>> {
};

TEST_P(DFlash2GroupedConvDeviceTest,
       MatchesEagerWithStridedProjectionAndHistory) {
  const auto [dtype, taps] = GetParam();
  constexpr int32_t kBlockSize = 6;
  constexpr int32_t kNumGroups = 7;
  constexpr int32_t kGroupSize = 5;
  constexpr int64_t kTokens = 18;
  constexpr int64_t kHiddenSize = kNumGroups * kGroupSize;
  const auto cpu_options = torch::TensorOptions().dtype(dtype);
  torch::manual_seed(217);
  // Retain the projection's [token, side, tap, group] strides after selecting
  // one side. Also exercise channel-strided hidden states and base weights.
  const torch::Tensor hidden_storage =
      torch::randn({kTokens, 2 * kHiddenSize}, cpu_options);
  const torch::Tensor delta_storage =
      torch::randn({kTokens, 2, taps, kNumGroups}, cpu_options);
  const torch::Tensor base_storage =
      torch::randn({taps, 2 * kHiddenSize}, cpu_options);
  const torch::Tensor expected =
      dflash2_grouped_conv(hidden_storage.slice(/*dim=*/1,
                                                /*start=*/0,
                                                /*end=*/2 * kHiddenSize,
                                                /*step=*/2),
                           delta_storage.select(/*dim=*/1, /*index=*/1),
                           base_storage.slice(/*dim=*/1,
                                              /*start=*/0,
                                              /*end=*/2 * kHiddenSize,
                                              /*step=*/2),
                           kBlockSize,
                           kNumGroups,
                           kGroupSize,
                           taps);
  const torch::Device device(Platform::type_torch(), 0);
  const torch::Tensor hidden = hidden_storage.to(device).slice(
      /*dim=*/1, /*start=*/0, /*end=*/2 * kHiddenSize, /*step=*/2);
  const torch::Tensor delta =
      delta_storage.to(device).select(/*dim=*/1, /*index=*/1);
  const torch::Tensor base = base_storage.to(device).slice(
      /*dim=*/1, /*start=*/0, /*end=*/2 * kHiddenSize, /*step=*/2);
  const torch::Tensor eager = reference_grouped_conv(
      hidden, delta, base, kBlockSize, kNumGroups, kGroupSize, taps);
  const torch::Tensor fused = mlu::dflash2_grouped_conv(
      hidden, delta, base, kBlockSize, kNumGroups, kGroupSize, taps);
  EXPECT_TRUE(torch::equal(fused.cpu(), eager.cpu()));
  EXPECT_TRUE(torch::equal(fused.cpu(), expected));
}

INSTANTIATE_TEST_SUITE_P(
    FloatingPointTaps,
    DFlash2GroupedConvDeviceTest,
    ::testing::Combine(
        ::testing::Values(torch::kBFloat16, torch::kFloat16, torch::kFloat32),
        ::testing::Values(1, 2, 3, 4)),
    [](const ::testing::TestParamInfo<DFlash2GroupedConvDeviceTest::ParamType>&
           info) {
      return std::to_string(static_cast<int32_t>(std::get<0>(info.param))) +
             "Taps" + std::to_string(std::get<1>(info.param));
    });

TEST(DFlash2GroupedConvTest,
     GraphReplayReadsUpdatedHiddenStatesAndCoefficients) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  torch::Tensor hidden = torch::ones({8, 12}, options);
  torch::Tensor delta_storage = torch::zeros({8, 2, 2, 3}, options);
  const torch::Tensor delta = delta_storage.select(/*dim=*/1, /*index=*/1);
  torch::Tensor base = torch::ones({2, 12}, options);
  const auto convolve = [&]() {
    return mlu::dflash2_grouped_conv(hidden,
                                     delta,
                                     base,
                                     /*block_size=*/4,
                                     /*num_groups=*/3,
                                     /*group_size=*/4,
                                     /*taps=*/2);
  };
  convolve();
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  torch::Tensor captured;
  {
    torch_mlu::mlu::MLUStreamGuard guard(
        torch_mlu::getStreamFromPool(/*isHighPriority=*/false, device.index()));
    graph.capture_begin();
    captured = convolve();
    graph.capture_end();
  }
  hidden.fill_(3);
  delta_storage.fill_(2);
  base.fill_(4);
  graph.replay();
  torch_mlu::synchronize();
  const torch::Tensor expected =
      torch::tensor({18., 36., 36., 36., 18., 36., 36., 36.}, options)
          .view({8, 1})
          .expand({8, 12});
  EXPECT_TRUE(torch::equal(captured, expected));
}

// This opt-in regression reserves 4 GiB of BF16 storage but initializes only
// four values. Keep it disabled in ordinary small-device test runs.
TEST(DFlash2GroupedConvTest, DISABLED_LargeRowStrideUses64BitAddressing) {
  constexpr int64_t kRowStride = int64_t{1} << 31;
  const auto options = torch::TensorOptions()
                           .dtype(torch::kBFloat16)
                           .device(torch::Device(Platform::type_torch(), 0));
  const torch::Tensor storage = torch::empty({kRowStride + 2}, options);
  storage.narrow(/*dim=*/0, /*start=*/0, /*length=*/2)
      .copy_(torch::tensor({1., 2.}, options));
  storage.narrow(/*dim=*/0, /*start=*/kRowStride, /*length=*/2)
      .copy_(torch::tensor({3., 4.}, options));
  const torch::Tensor hidden = storage.as_strided({2, 2}, {kRowStride, 1});
  const torch::Tensor delta = torch::zeros({2, 1, 2}, options);
  const torch::Tensor base = torch::tensor({3., 4.}, options).view({1, 2});
  const torch::Tensor output = mlu::dflash2_grouped_conv(hidden,
                                                         delta,
                                                         base,
                                                         /*block_size=*/1,
                                                         /*num_groups=*/2,
                                                         /*group_size=*/1,
                                                         /*taps=*/1);
  // Avoid the CNNL eager path and a full-storage host copy: the oracle is a
  // small CPU tensor, and only the contiguous four-element output is copied.
  const torch::Tensor expected =
      torch::tensor({3., 8., 9., 16.}, torch::kBFloat16).view({2, 2});
  EXPECT_TRUE(torch::equal(output.cpu(), expected));
}

// Limit test memory with three disjoint read-only views of one 4 GiB
// allocation.
TEST(DFlash2GroupedConvTest, DISABLED_LargeColumnStridesUse64BitAddressing) {
  constexpr int64_t kStride = int64_t{1} << 31;
  const auto options = torch::TensorOptions()
                           .dtype(torch::kBFloat16)
                           .device(torch::Device(Platform::type_torch(), 0));
  torch::Tensor storage = torch::empty({kStride + 3}, options);
  storage.narrow(0, 0, 3).copy_(torch::tensor({1., 3., 5.}, options));
  storage.narrow(0, kStride, 3).copy_(torch::tensor({2., 4., 6.}, options));
  const torch::Tensor hidden =
      storage.as_strided({1, 2}, {0, kStride}, /*storage_offset=*/0);
  const torch::Tensor base =
      storage.as_strided({1, 2}, {0, kStride}, /*storage_offset=*/1);
  const torch::Tensor delta =
      storage.as_strided({1, 1, 2}, {0, 0, kStride}, /*storage_offset=*/2);
  const torch::Tensor actual = mlu::dflash2_grouped_conv(hidden,
                                                         delta,
                                                         base,
                                                         /*block_size=*/1,
                                                         /*num_groups=*/2,
                                                         /*group_size=*/1,
                                                         /*taps=*/1);
  EXPECT_TRUE(torch::equal(
      actual.cpu(), torch::tensor({8., 20.}, torch::kBFloat16).view({1, 2})));
}

// Cached convolution must dispatch before narrowing the coefficient row stride.
TEST(DFlash2GroupedConvTest,
     DISABLED_LargeCoefficientRowStrideUsesGenericPath) {
  constexpr int64_t kStride = int64_t{1} << 31;
  const auto options = torch::TensorOptions()
                           .dtype(torch::kBFloat16)
                           .device(torch::Device(Platform::type_torch(), 0));
  torch::Tensor storage = torch::empty({kStride + 512}, options);
  storage.narrow(0, 0, 512).fill_(1);
  storage.narrow(0, kStride, 512).fill_(2);
  const torch::Tensor delta =
      storage.as_strided({2, 2, 256}, {kStride, 256, 1});
  const torch::Tensor hidden = torch::ones({2, 4096}, options);
  const torch::Tensor base = torch::ones({2, 4096}, options);
  const torch::Tensor actual = mlu::dflash2_grouped_conv(hidden,
                                                         delta,
                                                         base,
                                                         /*block_size=*/1,
                                                         /*num_groups=*/256,
                                                         /*group_size=*/16,
                                                         /*taps=*/2);
  const torch::Tensor expected =
      torch::tensor({2., 3.}, torch::kBFloat16).view({2, 1}).expand({2, 4096});
  EXPECT_TRUE(torch::equal(actual.cpu(), expected));
}

TEST(DFlash2GroupedConvModuleTest, PrepareAndFinishLoadCheckpointSides) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  constexpr int32_t kHiddenSize = 16;
  constexpr int32_t kGroups = 4;
  constexpr int32_t kTaps = 3;
  constexpr int32_t kBlock = 6;
  mlu::DFlash2GroupedConv module(kHiddenSize, kTaps, 4, kBlock, options);
  DFlash2GroupedConv common(kHiddenSize, kTaps, 4, kBlock, options);
  torch::Tensor base = torch::arange(2 * kTaps * kHiddenSize, options)
                           .reshape({2, kTaps, kHiddenSize}) /
                       128;
  torch::Tensor projection =
      torch::arange(2 * kTaps * kGroups * kHiddenSize, options)
          .remainder(7)
          .reshape({2 * kTaps * kGroups, kHiddenSize}) /
      128;
  module->load_state_dict(StateDict(
      {{"base_kernel", base}, {"kernel_projection.weight", projection}}));
  module->verify_loaded_weights("attention_conv.");
  common->load_state_dict(StateDict(
      {{"base_kernel", base}, {"kernel_projection.weight", projection}}));
  common->verify_loaded_weights("attention_conv.");
  const auto common_parameters = common->named_parameters();
  ASSERT_EQ(common_parameters.size(), 2);
  EXPECT_TRUE(torch::equal(common_parameters["base_kernel"], base));
  EXPECT_TRUE(
      torch::equal(common_parameters["kernel_projection.weight"], projection));

  const auto parameters = module->named_parameters();
  ASSERT_EQ(parameters.size(), 2);
  EXPECT_TRUE(torch::equal(parameters["base_kernel"], base));
  EXPECT_TRUE(torch::equal(parameters["kernel_projection.weight"], projection));
  torch::Tensor hidden = torch::arange(2 * kBlock * kHiddenSize, options)
                             .reshape({2 * kBlock, kHiddenSize}) /
                         64;
  const torch::Tensor original = hidden.clone();
  const torch::Tensor coefficients = torch::matmul(hidden, projection.t())
                                         .view({2 * kBlock, 2, kTaps, kGroups});
  auto [prepared, finish_coefficients] = module->prepare(hidden);
  auto [common_prepared, common_coefficients] = common->prepare(hidden);
  EXPECT_TRUE(torch::equal(finish_coefficients, common_coefficients));
  EXPECT_TRUE(torch::allclose(prepared, common_prepared, 1e-5, 1e-5));
  EXPECT_TRUE(
      torch::allclose(module->finish(prepared, finish_coefficients),
                      common->finish(common_prepared, common_coefficients),
                      1e-5,
                      1e-5));
  EXPECT_TRUE(torch::equal(finish_coefficients, coefficients.select(1, 1)));
  const torch::Tensor expected_prepare =
      reference_grouped_conv(hidden,
                             coefficients.select(1, 0),
                             base.select(0, 0),
                             kBlock,
                             kGroups,
                             4,
                             kTaps);
  EXPECT_TRUE(torch::allclose(prepared, expected_prepare, 1e-5, 1e-5));
  const torch::Tensor expected_finish =
      reference_grouped_conv(prepared,
                             coefficients.select(1, 1),
                             base.select(0, 1),
                             kBlock,
                             kGroups,
                             4,
                             kTaps);
  EXPECT_TRUE(torch::allclose(module->finish(prepared, finish_coefficients),
                              expected_finish,
                              1e-5,
                              1e-5));
  EXPECT_TRUE(torch::equal(hidden, original));
}

TEST(DFlash2GroupedConvModuleTest, UnsupportedFusionLeavesInputsForEagerNorm) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  mlu::DFlash2GroupedConv module(32, 3, 4, 6, options);
  torch::Tensor base = torch::ones({2, 3, 32}, options);
  module->load_state_dict(StateDict(
      {{"base_kernel", base},
       {"kernel_projection.weight", torch::zeros({48, 32}, options)}}));
  torch::Tensor hidden = torch::randn({12, 32}, options);
  auto [prepared, coefficients] = module->prepare(hidden);
  torch::Tensor residual = torch::randn_like(hidden);
  const torch::Tensor original_residual = residual.clone();
  const torch::Tensor original_prepared = prepared.clone();
  const torch::Tensor original_coefficients = coefficients.clone();
  RMSNorm norm(32, 1e-5, options);
  norm->weight().fill_(1);
  EXPECT_FALSE(module
                   ->finish_with_norm(
                       prepared, coefficients, residual, norm->weight(), 1e-5)
                   .has_value());
  EXPECT_TRUE(torch::equal(residual, original_residual));
  EXPECT_TRUE(torch::equal(prepared, original_prepared));
  EXPECT_TRUE(torch::equal(coefficients, original_coefficients));
  torch::Tensor expected_residual = original_residual.clone();
  torch::Tensor eager = reference_grouped_conv(
      prepared, coefficients, base.select(0, 1), 6, 8, 4, 3);
  auto [expected, expected_updated] = norm->forward(eager, expected_residual);
  torch::Tensor finished = module->finish(prepared, coefficients);
  auto [actual, updated] = norm->forward(finished, residual);
  ASSERT_TRUE(updated.has_value());
  ASSERT_TRUE(expected_updated.has_value());
  EXPECT_EQ(updated->data_ptr(), residual.data_ptr());
  EXPECT_TRUE(torch::equal(*updated, *expected_updated));
  EXPECT_TRUE(torch::allclose(actual, expected, 1e-2, 1e-2));
}

TEST(DFlash2GroupedConvModuleTest, PreparedProjectionFeedsFusedFinishNorm) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  torch::manual_seed(2026);
  for (int32_t block : {4, 6}) {
    SCOPED_TRACE(block);
    mlu::DFlash2GroupedConv module(4096, 2, 16, block, options);
    torch::Tensor base = torch::randn({2, 2, 4096}, options) / 8;
    torch::Tensor projection = torch::randn({1024, 4096}, options) / 128;
    module->load_state_dict(StateDict(
        {{"base_kernel", base}, {"kernel_projection.weight", projection}}));
    module->verify_loaded_weights("mlp_conv.");
    torch::Tensor hidden = torch::randn({3 * block, 4096}, options);
    auto [prepared, coefficients] = module->prepare(hidden);
    torch::Tensor residual = torch::randn_like(hidden);
    torch::Tensor expected_residual = residual.clone();
    RMSNorm norm(4096, 1e-5, options);
    norm->weight().copy_(torch::randn({4096}, options));
    torch::Tensor eager = reference_grouped_conv(
        prepared, coefficients, base.select(0, 1), block, 256, 16, 2);
    auto [expected, updated] = norm->forward(eager, expected_residual);
    auto actual = module->finish_with_norm(
        prepared, coefficients, residual, norm->weight(), 1e-5);
    ASSERT_TRUE(actual.has_value());
    ASSERT_TRUE(updated.has_value());
    EXPECT_EQ(std::get<1>(*actual).data_ptr(), residual.data_ptr());
    EXPECT_TRUE(torch::equal(std::get<1>(*actual), *updated));
    EXPECT_TRUE(torch::allclose(std::get<0>(*actual), expected, 1e-2, 1e-2));
  }
}

TEST(DFlash2NormTest, InitializesResidualAliasAndAccumulatesNextLayer) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  RMSNorm norm(/*dim=*/128, /*eps=*/1e-5, options);
  norm->weight().fill_(1);
  torch::Tensor input =
      (torch::arange(256, options).reshape({2, 128}) + 1) / 128;
  const torch::Tensor original = input.clone();
  torch::Tensor reference_input = input.clone();
  const auto expected_first = norm->forward(reference_input);
  auto [first, residual] = norm->forward_with_residual(input, std::nullopt);
  ASSERT_TRUE(residual.has_value());
  EXPECT_EQ(residual->data_ptr(), input.data_ptr());
  EXPECT_TRUE(torch::equal(*residual, original));
  EXPECT_TRUE(torch::equal(first, std::get<0>(expected_first)));

  torch::Tensor next = torch::full_like(input, 0.5);
  torch::Tensor reference_next = next.clone();
  const torch::Tensor reference_residual = residual->clone();
  const auto expected_next = norm->forward(reference_next, reference_residual);
  auto [output, updated] = norm->forward_with_residual(next, residual);
  ASSERT_TRUE(updated.has_value());
  ASSERT_TRUE(std::get<1>(expected_next).has_value());
  EXPECT_TRUE(torch::equal(output, std::get<0>(expected_next)));
  EXPECT_TRUE(torch::equal(*updated, *std::get<1>(expected_next)));
  EXPECT_TRUE(torch::equal(*updated, original + 0.5));
}

TEST(DFlash2DispatchTest, GroupedConvMatchesDirectApi) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const torch::Tensor hidden = torch::arange(48, options).reshape({8, 6}) / 16;
  const torch::Tensor delta = torch::full({8, 2, 3}, 0.25, options);
  const torch::Tensor base = torch::ones({2, 6}, options);
  const torch::Tensor expected = kernel::mlu::bangc::dflash2_grouped_conv(
      hidden, delta, base, /*block_size=*/4, /*group_size=*/2);
  const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "xllm_mlu::dflash2_grouped_conv", "");
  std::vector<c10::IValue> stack{hidden, delta, base, int64_t{4}, int64_t{2}};
  op.callBoxed(&stack);
  ASSERT_EQ(stack.size(), 1);
  EXPECT_TRUE(torch::equal(stack[0].toTensor(), expected));
}

TEST(DFlash2DispatchTest, SamplePathMatchesDirectApi) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const torch::Tensor ids =
      torch::arange(12, options.dtype(torch::kLong)).reshape({2, 3, 2});
  const torch::Tensor edges =
      torch::arange(24, options).reshape({2, 3, 2, 2}) / 16;
  const torch::Tensor noise =
      torch::tensor({0., 2., 3., 0., 0., 4., 2., 0., 0., 1., 3., 0.}, options)
          .reshape({2, 3, 2});
  const torch::Tensor mask =
      torch::tensor({true, false}, options.dtype(torch::kBool));
  const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "xllm_mlu::sample_dflash2_paths", "");
  for (const bool all_random : {false, true}) {
    const auto expected = kernel::mlu::bangc::sample_dflash2_paths(
        ids, edges, noise, mask, all_random);
    std::vector<c10::IValue> stack{ids, edges, noise, mask, all_random};
    op.callBoxed(&stack);
    ASSERT_EQ(stack.size(), 2);
    EXPECT_TRUE(torch::equal(stack[0].toTensor(), expected.first));
    EXPECT_TRUE(torch::equal(stack[1].toTensor(), expected.second));
  }
}

TEST(DFlash2DispatchTest, ConvNormUpdatesAndAliasesResidual) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const torch::Tensor hidden = torch::ones({8, 4096}, options);
  const torch::Tensor delta = torch::zeros({8, 2, 256}, options);
  const torch::Tensor base = torch::ones({2, 4096}, options);
  const torch::Tensor weight = torch::ones({4096}, options);
  const torch::Tensor residual = torch::full_like(hidden, 3);
  const torch::Tensor direct_residual = residual.clone();
  const auto expected = kernel::mlu::bangc::dflash2_conv_norm(hidden,
                                                              delta,
                                                              base,
                                                              direct_residual,
                                                              weight,
                                                              /*block_size=*/4,
                                                              /*eps=*/1e-5);
  ASSERT_TRUE(expected.has_value());
  const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "xllm_mlu::dflash2_conv_norm", "");
  std::vector<c10::IValue> stack{
      hidden, delta, base, residual, weight, int64_t{4}, 1e-5};
  op.callBoxed(&stack);
  ASSERT_EQ(stack.size(), 2);
  EXPECT_TRUE(torch::equal(stack[0].toTensor(), std::get<0>(*expected)));
  EXPECT_EQ(stack[1].toTensor().data_ptr(), residual.data_ptr());
  EXPECT_TRUE(torch::equal(residual, direct_residual));
  const torch::Tensor updated = torch::tensor({4., 5., 5., 5.}, options)
                                    .repeat({2})
                                    .view({8, 1})
                                    .expand_as(residual);
  EXPECT_TRUE(torch::equal(residual, updated));
}

TEST(DFlash2DispatchDeathTest, ConvNormRejectsUnsupportedFusion) {
  const std::string original_style = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_DEATH(
      ([] {
        const torch::Device device(Platform::type_torch(), 0);
        const auto options =
            torch::TensorOptions().dtype(torch::kBFloat16).device(device);
        const torch::Tensor hidden = torch::ones({4, 4096}, options);
        const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
            "xllm_mlu::dflash2_conv_norm", "");
        std::vector<c10::IValue> stack{hidden,
                                       torch::zeros({4, 2, 256}, options),
                                       torch::ones({2, 4096}, options),
                                       hidden.clone(),
                                       torch::ones({4096}, options),
                                       int64_t{4},
                                       1e-6};
        op.callBoxed(&stack);
      }()),
      "Unsupported DFlash2 conv/RMSNorm fusion contract");
  ::testing::FLAGS_gtest_death_test_style = original_style;
}

TEST(DFlash2DispatchDeathTest, GroupedConvRejectsUnsupportedGroupSize) {
  const std::string original_style = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_DEATH(
      ([] {
        const torch::Device device(Platform::type_torch(), 0);
        const auto options =
            torch::TensorOptions().dtype(torch::kFloat32).device(device);
        const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
            "xllm_mlu::dflash2_grouped_conv", "");
        std::vector<c10::IValue> stack{torch::ones({1, 1}, options),
                                       torch::zeros({1, 1, 1}, options),
                                       torch::ones({1, 1}, options),
                                       int64_t{1},
                                       int64_t{2049}};
        op.callBoxed(&stack);
      }()),
      "group_size <= 2048");
  ::testing::FLAGS_gtest_death_test_style = original_style;
}

TEST(DFlash2DispatchDeathTest, SamplePathRejectsEmptyCandidates) {
  const std::string original_style = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_DEATH(
      ([] {
        const torch::Device device(Platform::type_torch(), 0);
        const auto options =
            torch::TensorOptions().dtype(torch::kFloat32).device(device);
        const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
            "xllm_mlu::sample_dflash2_paths", "");
        std::vector<c10::IValue> stack{
            torch::empty({1, 1, 0}, options.dtype(torch::kLong)),
            torch::empty({1, 1, 0, 0}, options),
            torch::empty({1, 1, 0}, options),
            torch::ones({1}, options.dtype(torch::kBool)),
            true};
        op.callBoxed(&stack);
      }()),
      "top_k > 0");
  ::testing::FLAGS_gtest_death_test_style = original_style;
}

TEST(DFlash2GroupedConvDeathTest, RejectsExplicitTapMismatch) {
  const std::string original_style = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_DEATH(
      ([] {
        const torch::Device device(Platform::type_torch(), 0);
        const auto options =
            torch::TensorOptions().dtype(torch::kFloat32).device(device);
        kernel::mlu::dflash2_grouped_conv(torch::ones({4, 6}, options),
                                          torch::zeros({4, 2, 3}, options),
                                          torch::ones({2, 6}, options),
                                          /*block_size=*/4,
                                          /*group_size=*/2,
                                          /*taps=*/3);
      }()),
      "base.size\\(0\\) == taps");
  ::testing::FLAGS_gtest_death_test_style = original_style;
}

TEST(DFlash2ConvNormTest, MatchesRmsNormAndPreservesResidualAlias) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  torch_mlu::mlu::MLUStreamGuard guard(
      torch_mlu::getStreamFromPool(/*isHighPriority=*/false, device.index()));
  torch::manual_seed(2026);
  RMSNorm norm(/*dim=*/4096, /*eps=*/1e-5, options);
  norm->weight().copy_(torch::randn({4096}, options));
  for (int32_t batch :
       {1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,
        13,  14,  15,  16,  31,  32,  33,  63,  64,  65,  66,  67,
        127, 128, 129, 131, 193, 257, 319, 385, 513, 769, 1025}) {
    for (int32_t block : {4, 6}) {
      for (int32_t sides : {1, 2}) {
        SCOPED_TRACE(::testing::Message()
                     << "batch=" << batch << " block=" << block
                     << " delta_row=" << sides * 512);
        const int64_t tokens = batch * block;
        torch::Tensor hidden = torch::randn({tokens, 4096}, options);
        torch::Tensor delta_storage =
            torch::randn({tokens, sides, 2, 256}, options);
        torch::Tensor delta = delta_storage.select(/*dim=*/1, sides - 1);
        torch::Tensor base = torch::randn({2, 4096}, options);
        torch::Tensor residual = torch::randn_like(hidden);
        torch::Tensor expected_residual = residual.clone();
        torch::Tensor convolution =
            mlu::dflash2_grouped_conv(hidden,
                                      delta,
                                      base,
                                      block,
                                      /*num_groups=*/256,
                                      /*group_size=*/16,
                                      /*taps=*/2);
        torch::Tensor eager = reference_grouped_conv(hidden,
                                                     delta,
                                                     base,
                                                     block,
                                                     /*num_groups=*/256,
                                                     /*group_size=*/16,
                                                     /*taps=*/2);
        EXPECT_TRUE(torch::equal(convolution, eager));
        auto [expected, updated] = norm->forward(eager, expected_residual);
        auto actual = kernel::mlu::bangc::dflash2_conv_norm(
            hidden, delta, base, residual, norm->weight(), block, /*eps=*/1e-5);
        ASSERT_TRUE(actual.has_value());
        ASSERT_TRUE(updated.has_value());
        EXPECT_EQ(std::get<1>(*actual).data_ptr(), residual.data_ptr());
        EXPECT_TRUE(torch::equal(std::get<1>(*actual), *updated));
        // Match the existing MLU normalization tolerance; residual remains
        // exact.
        EXPECT_TRUE(torch::allclose(
            std::get<0>(*actual), expected, /*rtol=*/1e-2, /*atol=*/1e-2));
      }
    }
  }
  torch_mlu::synchronize();
}

TEST(DFlash2ConvNormTest,
     GraphReplayUpdatesInputsAndRejectsUnsupportedAliasing) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  torch::Tensor hidden = torch::ones({64, 4096}, options);
  torch::Tensor delta = torch::zeros({64, 2, 256}, options);
  torch::Tensor base = torch::ones({2, 4096}, options);
  torch::Tensor residual = torch::full_like(hidden, 3);
  torch::Tensor weight = torch::ones({4096}, options);
  const auto run = [&]() {
    return kernel::mlu::bangc::dflash2_conv_norm(
        hidden, delta, base, residual, weight, /*block_size=*/4, /*eps=*/1e-5);
  };
  ASSERT_TRUE(run().has_value());
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  std::optional<std::tuple<torch::Tensor, torch::Tensor>> captured;
  {
    torch_mlu::mlu::MLUStreamGuard guard(
        torch_mlu::getStreamFromPool(/*isHighPriority=*/false, device.index()));
    graph.capture_begin();
    captured = run();
    graph.capture_end();
  }
  hidden.fill_(2);
  residual.fill_(3);
  graph.replay();
  torch_mlu::synchronize();
  ASSERT_TRUE(captured.has_value());
  EXPECT_EQ(std::get<1>(*captured).data_ptr(), residual.data_ptr());
  EXPECT_TRUE(torch::equal(std::get<0>(*captured), torch::ones_like(hidden)));
  const torch::Tensor expected = torch::tensor({5., 7., 7., 7.}, options)
                                     .repeat({16})
                                     .view({64, 1})
                                     .expand_as(hidden);
  EXPECT_TRUE(torch::equal(residual, expected));
  torch::Tensor before = hidden.clone();
  EXPECT_FALSE(
      kernel::mlu::bangc::dflash2_conv_norm(
          hidden, delta, base, hidden, weight, /*block_size=*/4, /*eps=*/1e-5)
          .has_value());
  EXPECT_TRUE(torch::equal(hidden, before));
  before = residual.clone();
  const torch::Tensor overlapping_delta =
      residual.as_strided({64, 2, 256}, {1024, 256, 1});
  EXPECT_FALSE(kernel::mlu::bangc::dflash2_conv_norm(hidden,
                                                     overlapping_delta,
                                                     base,
                                                     residual,
                                                     weight,
                                                     /*block_size=*/4,
                                                     /*eps=*/1e-5)
                   .has_value());
  EXPECT_TRUE(torch::equal(residual, before));
  EXPECT_FALSE(
      kernel::mlu::bangc::dflash2_conv_norm(
          hidden, delta, base, residual, weight, /*block_size=*/4, /*eps=*/1e-6)
          .has_value());
  EXPECT_FALSE(kernel::mlu::bangc::dflash2_conv_norm(hidden.narrow(0, 0, 33),
                                                     delta.narrow(0, 0, 33),
                                                     base,
                                                     residual.narrow(0, 0, 33),
                                                     weight,
                                                     /*block_size=*/4,
                                                     /*eps=*/1e-5)
                   .has_value());
  EXPECT_FALSE(kernel::mlu::bangc::dflash2_conv_norm(hidden.narrow(0, 0, 0),
                                                     delta.narrow(0, 0, 0),
                                                     base,
                                                     residual.narrow(0, 0, 0),
                                                     weight,
                                                     /*block_size=*/4,
                                                     /*eps=*/1e-5)
                   .has_value());
  constexpr int64_t kTooManyTokens = int64_t{1} << 31;
  EXPECT_FALSE(kernel::mlu::bangc::dflash2_conv_norm(
                   hidden.as_strided({kTooManyTokens, 4096}, {0, 0}),
                   delta.as_strided({kTooManyTokens, 2, 256}, {0, 0, 0}),
                   base,
                   residual.as_strided({kTooManyTokens, 4096}, {0, 0}),
                   weight,
                   /*block_size=*/4,
                   /*eps=*/1e-5)
                   .has_value());
  EXPECT_TRUE(torch::equal(residual, before));
}

TEST(DFlash2ConvNormTest, TailWavesReplayUpdatedInputsOnNondefaultStream) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  for (int32_t batch :
       {1, 15, 65, 66, 67, 129, 131, 193, 257, 319, 385, 513, 769, 1025}) {
    for (int32_t block : {4, 6}) {
      for (int32_t sides : {1, 2}) {
        SCOPED_TRACE(::testing::Message()
                     << "batch=" << batch << " block=" << block
                     << " delta_row=" << sides * 512);
        const int64_t tokens = batch * block;
        torch::Tensor hidden = torch::ones({tokens, 4096}, options);
        torch::Tensor delta_storage =
            torch::zeros({tokens, sides, 2, 256}, options);
        const torch::Tensor delta = delta_storage.select(/*dim=*/1, sides - 1);
        const torch::Tensor base = torch::ones({2, 4096}, options);
        torch::Tensor residual = torch::full_like(hidden, 3);
        const torch::Tensor weight = torch::ones({4096}, options);
        const auto run = [&]() {
          return kernel::mlu::bangc::dflash2_conv_norm(
              hidden, delta, base, residual, weight, block, /*eps=*/1e-5);
        };
        ASSERT_TRUE(run().has_value());
        torch_mlu::synchronize();
        torch_mlu::MLUGraph graph;
        std::optional<std::tuple<torch::Tensor, torch::Tensor>> captured;
        {
          torch_mlu::mlu::MLUStreamGuard guard(torch_mlu::getStreamFromPool(
              /*isHighPriority=*/false, device.index()));
          graph.capture_begin();
          captured = run();
          graph.capture_end();
        }
        hidden.fill_(2);
        delta_storage.fill_(1);
        residual.fill_(3);
        graph.replay();
        torch_mlu::synchronize();
        ASSERT_TRUE(captured.has_value());
        EXPECT_EQ(std::get<1>(*captured).data_ptr(), residual.data_ptr());
        EXPECT_TRUE(
            torch::equal(std::get<0>(*captured), torch::ones_like(hidden)));
        const torch::Tensor expected =
            torch::where(torch::arange(tokens, options.dtype(torch::kLong))
                             .remainder(block)
                             .eq(0),
                         7,
                         11)
                .to(torch::kBFloat16)
                .view({tokens, 1})
                .expand_as(hidden);
        EXPECT_TRUE(torch::equal(residual, expected));
      }
    }
  }
}

}  // namespace
}  // namespace xllm::layer
