/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/sampling/rejection_sampler.h"
#include "core/kernels/mlu/dflash2_sampling.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "core/runtime/dflash/dflash2_worker_impl.h"
#include "core/runtime/llm_worker_impl.h"

namespace xllm {

class DFlash2WorkerImplTestPeer final {
 public:
  static auto fused_sample(const torch::Tensor& candidate_ids,
                           const torch::Tensor& edge_log_probs,
                           const torch::Tensor& gumbel_noise,
                           const SamplingParameters& sampling_params,
                           int64_t vocab_size) {
    return DFlash2WorkerImpl::fused_sample(candidate_ids,
                                           edge_log_probs,
                                           gumbel_noise,
                                           sampling_params,
                                           vocab_size);
  }

  static std::pair<torch::Tensor, torch::Tensor> sample_path(
      const DFlash2WorkerImpl& worker,
      const DFlash2CandidateOutput& candidates,
      const SamplingParameters& sampling,
      const torch::Tensor& noise,
      int64_t vocab_size) {
    auto result = worker.sample_path(candidates, sampling, noise, vocab_size);
    return {std::move(result.token_ids), std::move(result.dense_probs)};
  }
};

class DFlashWorkerImplTestPeer final {
 public:
  static void set_target(DFlashWorkerImpl& worker,
                         std::unique_ptr<WorkerImpl> target) {
    worker.impl_ = std::move(target);
    worker.target_is_hybrid_recurrent_ = true;
    worker.dtype_ = torch::kFloat32;
  }

  static ForwardInput prepare_validate(DFlashWorkerImpl& worker,
                                       const ForwardInput& input) {
    ForwardInput output;
    worker.prepare_validate_inputs(input, output);
    CHECK_EQ(worker.prepare_stream_->synchronize(), 0);
    return output;
  }
};

namespace {

constexpr int64_t kDraftSteps = 3;
constexpr int64_t kCandidates = 4;
constexpr int64_t kVocabSize = 16;

std::pair<torch::Tensor, torch::Tensor> reference_sample_path(
    const DFlash2CandidateOutput& candidates,
    const SamplingParameters& sampling,
    const torch::Tensor& noise) {
  const torch::Tensor candidate_ids = candidates.candidate_ids.cpu();
  torch::Tensor scores = candidates.edge_logits.cpu().to(torch::kFloat32);
  const int64_t batch = candidate_ids.size(0);
  if (sampling.temperatures.defined()) {
    torch::Tensor temperatures = sampling.temperatures.cpu();
    temperatures = temperatures.masked_fill(temperatures.eq(0), 1.0);
    scores = scores / temperatures.view({batch, 1, 1, 1});
  }
  scores = torch::log_softmax(scores, /*dim=*/-1);
  const torch::Tensor host_noise = noise.cpu();
  const torch::Tensor do_sample = sampling.do_sample.cpu();
  torch::Tensor tokens = torch::empty({batch, kDraftSteps}, torch::kLong);
  torch::Tensor probabilities =
      torch::zeros({batch, kDraftSteps, kVocabSize}, torch::kFloat32);
  using ISlice = torch::indexing::Slice;
  for (int64_t row = 0; row < batch; ++row) {
    int64_t predecessor = 0;
    for (int64_t step = 0; step < kDraftSteps; ++step) {
      const torch::Tensor row_scores = scores.index({row, step, predecessor});
      const int64_t successor = (row_scores + host_noise.index({row, step}))
                                    .argmax(/*dim=*/-1)
                                    .item<int64_t>();
      const torch::Tensor step_candidates = candidate_ids.index({row, step});
      tokens.index_put_({row, step}, step_candidates.index({successor}));
      const bool random =
          sampling.all_random_sample ||
          (!sampling.all_greedy_sample && do_sample.index({row}).item<bool>());
      torch::Tensor step_probs = torch::zeros({kCandidates}, torch::kFloat32);
      if (random) {
        step_probs = row_scores.exp();
      } else {
        step_probs.index_put_({successor}, 1.0);
      }
      probabilities.index({row, step, ISlice()})
          .scatter_(/*dim=*/0, step_candidates, step_probs);
      predecessor = successor;
    }
  }
  return {std::move(tokens), std::move(probabilities)};
}

std::pair<torch::Tensor, torch::Tensor> reference_sparse_paths(
    const torch::Tensor& ids,
    const torch::Tensor& log_probs,
    const torch::Tensor& noise,
    const torch::Tensor& do_sample,
    bool all_random_sample) {
  const int64_t batch = ids.size(0);
  const int64_t steps = ids.size(1);
  const int64_t top_k = ids.size(2);
  torch::Tensor tokens = torch::empty({batch, steps}, ids.options());
  torch::Tensor probs =
      torch::empty({batch, steps, top_k}, log_probs.options());
  torch::Tensor previous = torch::zeros({batch}, ids.options());
  torch::Tensor offsets = torch::arange(batch, ids.options()) * (steps * top_k);
  torch::Tensor rows = log_probs.contiguous().view({-1, top_k});
  for (int64_t step = 0; step < steps; ++step) {
    torch::Tensor row = rows.index_select(
        /*dim=*/0, previous + (offsets + step * top_k));
    torch::Tensor selected =
        (row + noise.select(/*dim=*/1, /*index=*/step)).argmax(/*dim=*/-1);
    tokens.select(/*dim=*/1, /*index=*/step)
        .copy_(ids.select(/*dim=*/1, /*index=*/step)
                   .gather(/*dim=*/1, selected.view({-1, 1}))
                   .view({-1}));
    torch::Tensor step_probs = row.exp();
    if (!all_random_sample) {
      torch::Tensor onehot = torch::zeros_like(step_probs);
      onehot.scatter_(/*dim=*/1, selected.view({-1, 1}), /*value=*/1.0);
      step_probs = torch::where(do_sample.view({batch, 1}), step_probs, onehot);
    }
    probs.select(/*dim=*/1, /*index=*/step).copy_(step_probs);
    previous = selected;
  }
  return {std::move(tokens), std::move(probs)};
}

TEST(MluDFlash2SamplingHelperTest, HostInputsAreRejectedWithoutMutation) {
  const torch::Tensor ids = torch::arange(12, torch::kLong).view({1, 3, 4});
  const torch::Tensor scores = torch::randn({1, 3, 4, 4});
  const torch::Tensor noise = torch::randn({1, 3, 4});
  SamplingParameters sampling;
  sampling.all_random_sample = true;
  sampling.all_greedy_sample = false;
  sampling.do_sample = torch::ones({1}, torch::kBool);
  const torch::Tensor original_ids = ids.clone();
  const torch::Tensor original_scores = scores.clone();
  const torch::Tensor original_noise = noise.clone();
  const torch::Tensor original_do_sample = sampling.do_sample.clone();
  EXPECT_FALSE(
      DFlash2WorkerImplTestPeer::fused_sample(ids, scores, noise, sampling, 16)
          .has_value());
  EXPECT_TRUE(torch::equal(ids, original_ids));
  EXPECT_TRUE(torch::equal(scores, original_scores));
  EXPECT_TRUE(torch::equal(noise, original_noise));
  EXPECT_TRUE(torch::equal(sampling.do_sample, original_do_sample));
}

class MluDFlash2SamplingGuardTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_.set_device();
    device_.init_device_context();
    inputs_ = {
        torch::arange(12, torch::kLong).view({1, 3, 4}).to(device_.unwrap()),
        torch::full({1, 3, 4, 4}, -std::log(4.0F)).to(device_.unwrap()),
        torch::zeros({1, 3, 4}).to(device_.unwrap()),
        torch::ones({1}, torch::kBool).to(device_.unwrap())};
  }

  void expect_rejected(const std::array<torch::Tensor, 4>& inputs) {
    std::array<torch::Tensor, 4> original;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (!inputs[i].defined()) {
        continue;
      }
      original[i] = inputs[i].cpu().clone();
    }
    EXPECT_FALSE(
        kernel::mlu::supports_dflash2_sampling(inputs[0],
                                               inputs[1],
                                               inputs[2],
                                               inputs[3],
                                               /*all_random_sample=*/false,
                                               /*all_greedy_sample=*/false));
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (!inputs[i].defined()) {
        continue;
      }
      EXPECT_TRUE(torch::equal(inputs[i].cpu(), original[i]));
    }
  }

  Device device_{/*device_index=*/0};
  std::array<torch::Tensor, 4> inputs_;
};

TEST_F(MluDFlash2SamplingGuardTest, MissingOrHostInputsAreRejected) {
  for (size_t i = 0; i < inputs_.size(); ++i) {
    SCOPED_TRACE(i);
    auto missing = inputs_;
    missing[i] = torch::Tensor();
    expect_rejected(missing);
    auto host = inputs_;
    host[i] = host[i].cpu();
    expect_rejected(host);
  }
  auto invalid_rank = inputs_;
  invalid_rank[0] = invalid_rank[0].flatten();
  expect_rejected(invalid_rank);
}

TEST_F(MluDFlash2SamplingGuardTest, RandomSamplingDoesNotRequireDeviceMask) {
  for (const torch::Tensor& mask : {torch::Tensor(), inputs_[3].cpu()}) {
    ASSERT_TRUE(
        kernel::mlu::supports_dflash2_sampling(inputs_[0],
                                               inputs_[1],
                                               inputs_[2],
                                               mask,
                                               /*all_random_sample=*/true,
                                               /*all_greedy_sample=*/false));
    auto result = kernel::mlu::sample_dflash2_paths(inputs_[0],
                                                    inputs_[1],
                                                    inputs_[2],
                                                    mask,
                                                    /*all_random_sample=*/true);
    EXPECT_TRUE(
        torch::equal(result.first.cpu(),
                     torch::tensor({0, 4, 8}, torch::kLong).view({1, 3})));
    EXPECT_TRUE(
        torch::allclose(result.second.cpu(), torch::full({1, 3, 4}, 0.25F)));
  }
}

TEST_F(MluDFlash2SamplingGuardTest, OtherDeviceInputsAreRejected) {
  if (Platform::device_count() < 2) {
    GTEST_SKIP() << "Requires two MLU devices";
  }
  Device other_device(/*device_index=*/1);
  other_device.set_device();
  other_device.init_device_context();
  for (size_t i = 0; i < inputs_.size(); ++i) {
    SCOPED_TRACE(i);
    auto inputs = inputs_;
    inputs[i] = inputs[i].cpu().to(other_device.unwrap());
    expect_rejected(inputs);
  }
  device_.set_device();
}

TEST(MluDFlash2RandomPathKernelTest,
     ExactSparseProbabilitiesWithStridedInputs) {
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  torch::manual_seed(42);
  struct Shape {
    int64_t batch;
    int64_t steps;
    int64_t candidates;
  };
  const std::vector<Shape> shapes = {
      {0, 3, 16}, {1, 1, 1}, {15, 3, 5}, {16, 3, 16}, {33, 7, 33}};
  const std::vector<torch::ScalarType> id_types = {torch::kInt32, torch::kLong};
  const std::vector<bool> modes = {true, false};
  using ISlice = torch::indexing::Slice;
  for (const Shape& shape : shapes) {
    for (torch::ScalarType id_type : id_types) {
      for (bool all_random_sample : modes) {
        SCOPED_TRACE(::testing::Message()
                     << "B=" << shape.batch << " S=" << shape.steps
                     << " K=" << shape.candidates << " dtype=" << id_type
                     << " random=" << all_random_sample);
        torch::Tensor ids =
            torch::arange(shape.batch * shape.steps * shape.candidates * 2,
                          torch::TensorOptions().dtype(id_type))
                .view({shape.batch, shape.steps, shape.candidates * 2})
                .to(device.unwrap())
                .index(
                    {ISlice(), ISlice(), ISlice(0, torch::indexing::None, 2)});
        torch::Tensor log_probs =
            torch::log_softmax(torch::randn({shape.batch,
                                             shape.steps,
                                             shape.candidates,
                                             shape.candidates}),
                               /*dim=*/-1)
                .to(device.unwrap());
        // Repeated/interleaved backing storage preserves normalized values
        // while exposing a non-contiguous last dimension to the kernel.
        log_probs = log_probs.repeat_interleave(/*repeats=*/2, /*dim=*/3)
                        .index({ISlice(),
                                ISlice(),
                                ISlice(),
                                ISlice(0, torch::indexing::None, 2)});
        torch::Tensor noise =
            torch::randn({shape.batch, shape.steps, shape.candidates * 2})
                .to(device.unwrap())
                .index(
                    {ISlice(), ISlice(), ISlice(0, torch::indexing::None, 2)});
        torch::Tensor mask =
            (torch::arange(shape.batch * 2, torch::kLong) % 4 == 0)
                .to(device.unwrap())
                .index({ISlice(0, torch::indexing::None, 2)});
        torch::Tensor input_mask = all_random_sample ? torch::Tensor() : mask;
        auto expected = reference_sparse_paths(
            ids, log_probs, noise, input_mask, all_random_sample);
        auto result = kernel::mlu::sample_dflash2_paths(
            ids, log_probs, noise, input_mask, all_random_sample);
        EXPECT_EQ(result.first.scalar_type(), id_type);
        EXPECT_TRUE(torch::equal(result.first.cpu(), expected.first.cpu()));
        EXPECT_TRUE(torch::equal(result.second.cpu(), expected.second.cpu()));
        EXPECT_TRUE(torch::allclose(result.second.sum(/*dim=*/-1).cpu(),
                                    torch::ones({shape.batch, shape.steps}),
                                    /*rtol=*/1e-5,
                                    /*atol=*/1e-6));
      }
    }
  }
}

TEST(MluDFlash2RandomPathKernelTest, CompactPathsMatchReferenceAcrossTiles) {
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  torch::manual_seed(2026);
  const auto options =
      torch::TensorOptions().device(device.unwrap()).dtype(torch::kFloat32);
  for (int64_t steps : {3, 5}) {
    for (int64_t batch : {1, 63, 64, 65, 127, 128, 129}) {
      torch::Tensor ids =
          torch::arange(batch * steps * 16, options.dtype(torch::kLong))
              .view({batch, steps, 16});
      for (bool ties : {false, true}) {
        torch::Tensor logits =
            ties ? torch::zeros({batch, steps, 16, 16}, options)
                 : torch::randn({batch, steps, 16, 16}, options);
        torch::Tensor logs = torch::log_softmax(logits, /*dim=*/-1);
        torch::Tensor noise = ties ? torch::zeros({batch, steps, 16}, options)
                                   : torch::randn({batch, steps, 16}, options);
        torch::Tensor mask = torch::arange(batch, options.dtype(torch::kLong))
                                 .remainder(2)
                                 .eq(0);
        for (bool all_random : {false, true}) {
          SCOPED_TRACE(::testing::Message()
                       << "batch=" << batch << " steps=" << steps
                       << " ties=" << ties << " random=" << all_random);
          torch::Tensor active_mask = all_random ? torch::Tensor() : mask;
          torch::Tensor active_noise =
              all_random ? noise
                         : torch::where(mask.view({batch, 1, 1}),
                                        noise,
                                        torch::zeros_like(noise));
          auto expected = reference_sparse_paths(
              ids, logs, active_noise, active_mask, all_random);
          auto actual = kernel::mlu::sample_dflash2_paths(
              ids, logs, active_noise, active_mask, all_random);
          EXPECT_TRUE(torch::equal(actual.first.cpu(), expected.first.cpu()));
          EXPECT_TRUE(torch::equal(actual.second.cpu(), expected.second.cpu()));
          if (ties) {
            EXPECT_TRUE(torch::equal(actual.first, ids.select(/*dim=*/2, 0)));
          }
        }
      }
    }
  }
}

// Reserves 12 GiB but initializes only six scalar values. Run explicitly on
// large devices to cover the original 64-bit Tensor stride contract.
TEST(MluDFlash2RandomPathKernelTest,
     DISABLED_LargeColumnStridesUse64BitAddressing) {
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const auto options = torch::TensorOptions().device(device.unwrap());
  constexpr int64_t kIdStride = int64_t{1} << 29;
  constexpr int64_t kFloatStride = int64_t{1} << 30;
  torch::Tensor id_storage =
      torch::empty({kIdStride + 1}, options.dtype(torch::kLong));
  torch::Tensor edge_storage =
      torch::empty({kFloatStride + 1}, options.dtype(torch::kFloat32));
  torch::Tensor noise_storage = torch::empty_like(edge_storage);
  id_storage.narrow(0, 0, 1).fill_(11);
  id_storage.narrow(0, kIdStride, 1).fill_(22);
  edge_storage.narrow(0, 0, 1).fill_(-1);
  edge_storage.narrow(0, kFloatStride, 1).fill_(-2);
  noise_storage.narrow(0, 0, 1).fill_(0);
  noise_storage.narrow(0, kFloatStride, 1).fill_(2);
  const torch::Tensor ids = id_storage.as_strided({1, 1, 2}, {0, 0, kIdStride});
  const torch::Tensor edges =
      edge_storage.as_strided({1, 1, 2, 2}, {0, 0, 0, kFloatStride});
  const torch::Tensor noise =
      noise_storage.as_strided({1, 1, 2}, {0, 0, kFloatStride});
  auto result = kernel::mlu::sample_dflash2_paths(
      ids, edges, noise, torch::Tensor(), /*all_random_sample=*/true);
  EXPECT_TRUE(torch::equal(result.first.cpu(),
                           torch::tensor({22}, torch::kLong).view({1, 1})));
  const torch::Tensor expected =
      torch::tensor({-1., -2.}, options.dtype(torch::kFloat32))
          .exp()
          .view({1, 1, 2});
  EXPECT_TRUE(torch::equal(result.second, expected));
}

TEST(MluDFlash2RandomPathKernelTest, FirstTieNanAndInfinityMatchTorch) {
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  torch::Tensor ids =
      torch::arange(60, torch::kLong).view({4, 3, 5}).to(device.unwrap());
  torch::Tensor scores = torch::zeros({4, 3, 5, 5});
  scores.select(/*dim=*/0, /*index=*/1)
      .fill_(-std::numeric_limits<float>::infinity());
  scores.select(/*dim=*/0, /*index=*/2)
      .select(/*dim=*/2, /*index=*/2)
      .fill_(std::numeric_limits<float>::quiet_NaN());
  scores.select(/*dim=*/0, /*index=*/2)
      .select(/*dim=*/2, /*index=*/3)
      .fill_(std::numeric_limits<float>::quiet_NaN());
  scores.select(/*dim=*/0, /*index=*/3)
      .select(/*dim=*/2, /*index=*/1)
      .fill_(std::numeric_limits<float>::infinity());
  scores = scores.to(device.unwrap());
  torch::Tensor noise = torch::zeros({4, 3, 5}, scores.options());
  torch::Tensor mask =
      torch::tensor({true, false, true, false}).to(device.unwrap());
  const std::vector<bool> modes = {true, false};
  for (bool all_random_sample : modes) {
    auto expected =
        reference_sparse_paths(ids, scores, noise, mask, all_random_sample);
    auto result = kernel::mlu::sample_dflash2_paths(
        ids, scores, noise, mask, all_random_sample);
    EXPECT_TRUE(torch::equal(result.first.cpu(), expected.first.cpu()));
    EXPECT_TRUE(torch::allclose(result.second.cpu(),
                                expected.second.cpu(),
                                /*rtol=*/0,
                                /*atol=*/0,
                                /*equal_nan=*/true));
  }
}

class StatefulTargetWorker final : public LLMWorkerImpl {
 public:
  StatefulTargetWorker(const ParallelArgs& parallel_args,
                       const torch::Device& device,
                       const runtime::Options& options,
                       bool has_state)
      : LLMWorkerImpl(parallel_args, device, options) {
    if (has_state) {
      kv_caches_.emplace_back(
          LinearAttentionKVCacheTensors{torch::Tensor(), torch::zeros({1})});
    }
  }
};

class MluDFlashVerifyInputTest : public ::testing::TestWithParam<bool> {};

TEST_P(MluDFlashVerifyInputTest, ReplacesDecodeSpansOnlyForStatefulTargets) {
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  runtime::Options options;
  options.num_speculative_tokens(kDraftSteps)
      .speculative_algorithm(GetParam() ? "DFlash2" : "DFlash")
      .block_size(16)
      .enable_schedule_overlap(false);
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  for (bool has_state : {false, true}) {
    std::unique_ptr<DFlashWorkerImpl> worker;
    if (GetParam()) {
      worker = std::make_unique<DFlash2WorkerImpl>(
          parallel_args, device.unwrap(), options);
    } else {
      worker = std::make_unique<DFlashWorkerImpl>(
          parallel_args, device.unwrap(), options);
    }
    DFlashWorkerImplTestPeer::set_target(
        *worker,
        std::make_unique<StatefulTargetWorker>(
            parallel_args, device.unwrap(), options, has_state));
    for (int32_t batch : {15, 16}) {
      ForwardInput input;
      input.token_ids_host = torch::arange(batch, torch::kInt32);
      input.positions_host = torch::full({batch}, 4, torch::kInt32);
      input.token_ids = input.token_ids_host.to(device.unwrap());
      input.positions = input.positions_host.to(device.unwrap());
      auto& params = input.input_params;
      params.meta.num_sequences = batch;
      params.meta.batch_forward_type = BatchForwardType::DECODE;
      params.meta.q_max_seq_len = 1;
      params.meta.kv_max_seq_len = 5;
      params.execution_batch.num_reqs = batch;
      params.execution_batch.num_tokens = batch;
      params.execution_batch.num_scheduled_tokens.assign(batch, 1);
      params.execution_batch.num_computed_tokens.assign(batch, 4);
      params.execution_batch.is_prefilling.assign(batch, 0);
      params.attention.host.block_tables =
          torch::arange(batch, torch::kInt32).view({batch, 1});
      params.attention.host.kpool_query_lens.assign(batch, 1);
      for (int32_t row = 0; row <= batch; ++row) {
        params.attention.host.q_seq_lens.emplace_back(row);
        params.attention.host.kv_seq_lens.emplace_back(row * 5);
      }
      params.execution_batch.query_start_loc = params.attention.host.q_seq_lens;
      auto output = DFlashWorkerImplTestPeer::prepare_validate(*worker, input);
      const auto& prepared = output.input_params;
      EXPECT_TRUE(prepared.is_spec_verify);
      EXPECT_TRUE(prepared.meta.batch_forward_type.is_chunked_prefill());
      EXPECT_EQ(prepared.meta.num_sequences, batch);
      EXPECT_EQ(prepared.meta.q_max_seq_len, kDraftSteps + 1);
      EXPECT_EQ(output.token_ids.numel(), batch * (kDraftSteps + 1));
      EXPECT_EQ(prepared.attention.host.kpool_query_lens,
                std::vector<int32_t>(batch, has_state ? kDraftSteps + 1 : 1));
      std::vector<int32_t> expected_offsets;
      expected_offsets.reserve(batch + 1);
      for (int32_t row = 0; row <= batch; ++row) {
        expected_offsets.emplace_back(row * (kDraftSteps + 1));
      }
      EXPECT_EQ(prepared.attention.host.q_seq_lens, expected_offsets);
      EXPECT_TRUE(torch::equal(prepared.attention.device.q_seq_lens.cpu(),
                               torch::tensor(expected_offsets, torch::kInt32)));
      EXPECT_TRUE(torch::equal(prepared.num_accepted_tokens.cpu(),
                               torch::ones({batch}, torch::kInt32)));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(WorkerVersions,
                         MluDFlashVerifyInputTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "DFlash2" : "DFlash";
                         });

struct SamplingInputs {
  DFlash2CandidateOutput candidates;
  SamplingParameters sampling;
  torch::Tensor noise;
  torch::Tensor tokens;
  torch::Tensor probabilities;
};

class MluDFlash2SamplingTest : public ::testing::TestWithParam<int32_t> {
 protected:
  void SetUp() override {
    device_.set_device();
    device_.init_device_context();
    runtime::Options options;
    options.num_speculative_tokens(kDraftSteps)
        .speculative_algorithm("DFlash2")
        .enable_schedule_overlap(false);
    const ParallelArgs parallel_args(
        /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
    worker_ = std::make_unique<DFlash2WorkerImpl>(
        parallel_args, device_.unwrap(), options);
  }

  std::unique_ptr<SamplingInputs> make_inputs(int64_t batch) {
    auto entry = std::make_unique<SamplingInputs>();
    const torch::TensorOptions integer_options =
        torch::TensorOptions().device(device_.unwrap()).dtype(torch::kLong);
    const torch::TensorOptions float_options =
        integer_options.dtype(torch::kFloat);
    entry->candidates.candidate_ids =
        (torch::arange(kDraftSteps * kCandidates, integer_options) + 1)
            .view({1, kDraftSteps, kCandidates})
            .repeat({batch, 1, 1});
    entry->candidates.edge_logits =
        torch::empty({batch, kDraftSteps, kCandidates, kCandidates},
                     float_options.dtype(torch::kBFloat16));
    entry->noise =
        torch::zeros({batch, kDraftSteps, kCandidates}, float_options);
    entry->sampling.all_greedy_sample = GetParam() == 0;
    entry->sampling.all_random_sample = GetParam() == 1;
    entry->sampling.do_sample =
        torch::zeros({batch}, integer_options.dtype(torch::kBool));
    entry->sampling.temperatures = torch::ones({batch}, float_options);
    entry->tokens = torch::empty({batch, kDraftSteps}, integer_options);
    entry->probabilities =
        torch::empty({batch, kDraftSteps, kVocabSize}, float_options);
    update_inputs(*entry, /*round=*/0);
    return entry;
  }

  void update_inputs(SamplingInputs& entry, int32_t round) {
    const int64_t batch = entry.noise.size(0);
    auto host_logits = torch::empty(
        {batch, kDraftSteps, kCandidates, kCandidates}, torch::kFloat);
    auto host_noise =
        torch::zeros({batch, kDraftSteps, kCandidates}, torch::kFloat);
    auto host_sample = torch::zeros({batch}, torch::kBool);
    auto host_temperature = torch::empty({batch}, torch::kFloat);
    auto logits = host_logits.accessor<float, 4>();
    auto noise = host_noise.accessor<float, 3>();
    auto sample = host_sample.accessor<bool, 1>();
    auto temperature = host_temperature.accessor<float, 1>();
    for (int64_t row = 0; row < batch; ++row) {
      sample[row] =
          GetParam() == 1 || (GetParam() == 2 && (row + round) % 2 == 0);
      temperature[row] = 0.5F + 0.25F * static_cast<float>((row + round) % 5);
      for (int64_t step = 0; step < kDraftSteps; ++step) {
        const int64_t chosen = (row + step + round) % kCandidates;
        noise[row][step][chosen] = sample[row] ? 20.0F : 0.0F;
        for (int64_t predecessor = 0; predecessor < kCandidates;
             ++predecessor) {
          for (int64_t successor = 0; successor < kCandidates; ++successor) {
            logits[row][step][predecessor][successor] =
                -static_cast<float>(successor) +
                0.25F * static_cast<float>(predecessor * successor) +
                0.1F * static_cast<float>((round + row) % 3 * successor);
          }
        }
      }
    }
    entry.candidates.edge_logits.copy_(host_logits);
    entry.noise.copy_(host_noise);
    entry.sampling.do_sample.copy_(host_sample);
    entry.sampling.temperatures.copy_(host_temperature);
  }

  void expect_sampling_matches_reference(SamplingInputs& entry, int32_t round) {
    update_inputs(entry, round);
    auto eager = DFlash2WorkerImplTestPeer::sample_path(
        *worker_, entry.candidates, entry.sampling, entry.noise, kVocabSize);
    entry.tokens = eager.first;
    entry.probabilities = eager.second;
    auto reference =
        reference_sample_path(entry.candidates, entry.sampling, entry.noise);
    EXPECT_TRUE(torch::equal(entry.tokens.cpu(), reference.first));
    EXPECT_TRUE(torch::allclose(entry.probabilities.cpu(),
                                reference.second,
                                /*rtol=*/1e-5,
                                /*atol=*/1e-6));
    EXPECT_TRUE(
        torch::allclose(entry.probabilities.sum(-1).cpu(),
                        torch::ones({entry.tokens.size(0), kDraftSteps}),
                        /*rtol=*/1e-5,
                        /*atol=*/1e-6));
    if (GetParam() == 1) {
      const torch::Tensor host_tokens = entry.tokens.cpu();
      auto tokens = host_tokens.accessor<int64_t, 2>();
      for (int64_t row = 0; row < entry.tokens.size(0); ++row) {
        for (int64_t step = 0; step < kDraftSteps; ++step) {
          EXPECT_EQ(
              tokens[row][step],
              step * kCandidates + (row + step + round) % kCandidates + 1);
        }
      }
    }
  }

  Device device_{/*device_index=*/0};
  std::unique_ptr<DFlash2WorkerImpl> worker_;
};

TEST_P(MluDFlash2SamplingTest,
       MatchesReferenceWithChangingNoiseAndTemperature) {
  for (int64_t batch : {15, 16}) {
    auto entry = make_inputs(batch);
    for (int32_t round = 0; round < 6; ++round) {
      SCOPED_TRACE(::testing::Message() << "B=" << batch << " round=" << round);
      expect_sampling_matches_reference(*entry, round);
    }
  }
}

TEST_P(MluDFlash2SamplingTest, HostCandidatesRetainEagerSamplingSemantics) {
  auto entry = make_inputs(/*batch=*/2);
  entry->candidates.candidate_ids = entry->candidates.candidate_ids.cpu();
  entry->candidates.edge_logits = entry->candidates.edge_logits.cpu();
  entry->noise = entry->noise.cpu();
  entry->sampling.temperatures = entry->sampling.temperatures.cpu();
  entry->sampling.do_sample = entry->sampling.do_sample.cpu();
  auto actual = DFlash2WorkerImplTestPeer::sample_path(
      *worker_, entry->candidates, entry->sampling, entry->noise, kVocabSize);
  auto expected =
      reference_sample_path(entry->candidates, entry->sampling, entry->noise);
  EXPECT_TRUE(actual.first.device().is_cpu());
  EXPECT_TRUE(actual.second.device().is_cpu());
  EXPECT_TRUE(torch::equal(actual.first, expected.first));
  EXPECT_TRUE(torch::allclose(actual.second, expected.second));
}

TEST_P(MluDFlash2SamplingTest, DenseProposalSupportsRejectedTokenRecovery) {
  auto entry = make_inputs(/*batch=*/16);
  auto result = DFlash2WorkerImplTestPeer::sample_path(
      *worker_, entry->candidates, entry->sampling, entry->noise, kVocabSize);
  DraftProposal proposal(result.first, result.second);
  proposal.validate(/*expected_batch_size=*/16,
                    /*expected_vocab_size=*/kVocabSize,
                    /*expected_num_speculative_tokens=*/kDraftSteps);
  // Candidates occupy ids 1..12; target support at 14 forces rejection and
  // requires the complete q distribution to recover the unique target token.
  torch::Tensor target_probs = torch::zeros_like(result.second);
  target_probs.select(/*dim=*/2, /*index=*/14).fill_(1.0);
  torch::Tensor uniform =
      torch::full({16, kDraftSteps}, 0.5, target_probs.options());
  torch::Tensor bonus = torch::full({16, 1}, 15, result.first.options());
  auto recovered =
      RejectionSampler::random_sample(proposal,
                                      target_probs,
                                      uniform,
                                      bonus,
                                      /*mask_out_rejected_tokens=*/true);
  torch::Tensor expected =
      torch::full({16, kDraftSteps + 1}, -1, result.first.options());
  expected.select(/*dim=*/1, /*index=*/0).fill_(14);
  EXPECT_TRUE(torch::equal(std::get<1>(recovered).cpu(), expected.cpu()));
}

INSTANTIATE_TEST_SUITE_P(GreedyRandomAndMixed,
                         MluDFlash2SamplingTest,
                         ::testing::Values(0, 1, 2));

}  // namespace
}  // namespace xllm
