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

#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "core/framework/sampling/rec_sampler.h"
#include "core/framework/sampling/sampler.h"
#include "platform/platform.h"

namespace xllm {
namespace {

SamplingParameters make_greedy_params(int64_t batch_size) {
  SamplingParameters params;
  params.selected_token_idxes =
      torch::arange(batch_size, torch::TensorOptions().dtype(torch::kInt));
  params.sample_idxes =
      torch::arange(batch_size, torch::TensorOptions().dtype(torch::kInt));
  params.do_sample = torch::zeros({batch_size}, torch::kBool);
  params.all_greedy_sample = true;
  params.all_random_sample = false;
  return params;
}

bool uses_device_random_sample() {
  return Platform::is_mlu() || Platform::is_cuda() || Platform::is_dcu();
}

torch::Device get_random_sample_device() {
  if (uses_device_random_sample()) {
    return torch::Device(Platform::type_torch(), 0);
  }
  return torch::Device(torch::kCPU);
}

TEST(SamplerFilterMaskTest, GreedySamplingHonorsMixedRows) {
  SamplingParameters params = make_greedy_params(/*batch_size=*/2);
  params.filter_mask =
      torch::tensor({{0.0F, -1.0e9F, 0.0F}, {0.0F, 0.0F, -1.0e9F}});
  torch::Tensor logits =
      torch::tensor({{1.0F, 100.0F, 2.0F}, {1.0F, 2.0F, 100.0F}});

  Sampler sampler;
  SampleOutput output = sampler.forward(logits, params);

  EXPECT_EQ(output.next_tokens.index({0}).item<int64_t>(), 2);
  EXPECT_EQ(output.next_tokens.index({1}).item<int64_t>(), 1);
}

TEST(SamplerFilterMaskTest, RandomSamplingCannotSelectDisallowedToken) {
  if (uses_device_random_sample() && Platform::device_count() == 0) {
    GTEST_SKIP() << "Random sampling backend device is unavailable";
  }

  SamplingParameters params = make_greedy_params(/*batch_size=*/1);
  params.do_sample = torch::ones({1}, torch::kBool);
  params.all_greedy_sample = false;
  params.all_random_sample = true;
  params.filter_mask = torch::tensor({{-1.0e9F, 0.0F, -1.0e9F}});
  torch::Tensor logits = torch::tensor({{100.0F, 1.0F, 100.0F}});

  const torch::Device device = get_random_sample_device();
  params = params.to(device, torch::kFloat32);
  logits = logits.to(device);

  Sampler sampler;
  SampleOutput output = sampler.forward(logits, params);

  EXPECT_EQ(output.next_tokens.index({0}).item<int64_t>(), 1);
}

TEST(SamplerFilterMaskTest, PackedMaskFiltersCallerLogitsInPlace) {
  SamplingParameters params = make_greedy_params(/*batch_size=*/1);
  params.filter_bitmask =
      torch::tensor({{static_cast<int32_t>(0b0101)}},
                    torch::TensorOptions().dtype(torch::kInt32));
  torch::Tensor logits = torch::tensor({{1.0F, 100.0F, 2.0F, 100.0F}});

  Sampler sampler;
  SampleOutput output = sampler.forward(logits, params);

  EXPECT_EQ(output.next_tokens.index({0}).item<int64_t>(), 2);
  EXPECT_EQ(logits.index({0, 0}).item<float>(), 1.0F);
  EXPECT_LT(logits.index({0, 1}).item<float>(), -1.0F);
  EXPECT_EQ(logits.index({0, 2}).item<float>(), 2.0F);
  EXPECT_LT(logits.index({0, 3}).item<float>(), -1.0F);
}

TEST(SamplerFilterMaskTest, TopKSamplingPublishesDistribution) {
  if (!Platform::is_mlu() || Platform::device_count() == 0) {
    GTEST_SKIP() << "MLU top-k sampling requires a device";
  }
  const torch::Device device = get_random_sample_device();
  SamplingParameters params = make_greedy_params(/*batch_size=*/2);
  params.do_sample = torch::ones({2}, torch::kBool);
  params.all_greedy_sample = false;
  params.all_random_sample = true;
  params.top_k = torch::ones({2}, torch::kLong);
  params.top_p = torch::ones({2});
  params = params.to(device, torch::kFloat32);
  torch::Tensor logits =
      torch::tensor({{3.0F, 2.0F, 1.0F}, {1.0F, 3.0F, 2.0F}}).to(device);

  const SampleOutput output = Sampler().forward(logits, params);
  const torch::Tensor expected =
      torch::tensor({{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}}).to(device);
  EXPECT_TRUE(torch::equal(output.probs, expected));
  EXPECT_TRUE(torch::equal(logits.softmax(-1), expected));
}

TEST(SamplerFilterMaskTest, GreedySamplingPublishesDenseMask) {
  for (bool return_probs : {false, true}) {
    SCOPED_TRACE(return_probs);
    SamplingParameters params = make_greedy_params(/*batch_size=*/1);
    params.return_probs = return_probs;
    params.filter_mask = torch::tensor({{-20.0F, 0.0F, 0.0F}});
    torch::Tensor logits = torch::tensor({{10.0F, 2.0F, 1.0F}});

    const SampleOutput output = Sampler().forward(logits, params);

    EXPECT_EQ(output.next_tokens.item<int64_t>(), 1);
    EXPECT_TRUE(torch::equal(logits, torch::tensor({{-10.0F, 2.0F, 1.0F}})));
  }
}

TEST(SamplerFilterMaskTest, SubsetSamplingPreservesOtherRows) {
  if (!Platform::is_mlu() || Platform::device_count() == 0) {
    GTEST_SKIP() << "MLU top-k sampling requires a device";
  }
  const torch::Device device = get_random_sample_device();
  const float neg_inf = -std::numeric_limits<float>::infinity();
  for (torch::ScalarType dtype : {torch::kFloat32, torch::kBFloat16}) {
    for (bool packed_mask : {false, true}) {
      SCOPED_TRACE(dtype);
      SCOPED_TRACE(packed_mask);
      SamplingParameters params = make_greedy_params(/*batch_size=*/3);
      params.sample_idxes = torch::tensor({2, 0}, torch::kLong);
      params.do_sample = torch::ones({2}, torch::kBool);
      params.all_greedy_sample = false;
      params.all_random_sample = true;
      params.top_k = torch::tensor({1, 1, 2}, torch::kLong);
      params.top_p = torch::ones({3});
      if (packed_mask) {
        params.filter_bitmask = torch::tensor({{15}, {15}, {13}}, torch::kInt);
      } else {
        params.filter_mask = torch::tensor(
            {{0.0F, neg_inf, 0.0F, 0.0F}, {0.0F, neg_inf, 0.0F, 0.0F}});
      }
      params = params.to(device, torch::kFloat32);
      torch::Tensor logits = torch::tensor({{4.0F, 3.0F, 2.0F, 1.0F},
                                            {3.0F, 2.0F, 1.0F, 0.0F},
                                            {1.0F, 2.0F, 4.0F, 3.0F}})
                                 .to(device, dtype);

      const SampleOutput output = Sampler().forward(logits, params);

      const torch::Tensor expected =
          torch::tensor({{4.0F, neg_inf, neg_inf, neg_inf},
                         {3.0F, 2.0F, 1.0F, 0.0F},
                         {neg_inf, neg_inf, 4.0F, 3.0F}})
              .to(device);
      EXPECT_EQ(logits.scalar_type(), torch::kFloat32);
      EXPECT_EQ(output.probs.scalar_type(), dtype);
      EXPECT_TRUE(torch::equal(logits, expected));
      EXPECT_TRUE(torch::equal(
          output.probs,
          expected.index_select(0, params.sample_idxes).softmax(-1).to(dtype)));
      EXPECT_EQ(output.next_tokens[1].item<int64_t>(), 0);
    }
  }
}

TEST(SamplerFilterMaskTest, FullRowsKeepOrderWithSparseSelection) {
  for (bool return_probs : {false, true}) {
    for (bool logprobs : {false, true}) {
      SamplingParameters params = make_greedy_params(/*batch_size=*/3);
      params.selected_token_idxes = torch::tensor({2, 5, 9}, torch::kLong);
      params.return_probs = return_probs;
      params.logprobs = logprobs;
      params.filter_mask = torch::tensor(
          {{0.F, -10.F, 0.F}, {0.F, 0.F, -10.F}, {-10.F, 0.F, 0.F}});
      torch::Tensor logits =
          torch::tensor({{1.F, 3.F, 2.F}, {3.F, 2.F, 1.F}, {2.F, 3.F, 1.F}});
      const torch::Tensor old_alias = logits;
      const torch::Tensor expected = logits + params.filter_mask;
      const SampleOutput output = Sampler().forward(logits, params);
      EXPECT_TRUE(torch::equal(logits, expected));
      EXPECT_TRUE(torch::equal(output.next_tokens, expected.argmax(-1)));
      EXPECT_FALSE(torch::equal(old_alias, logits));
    }
  }
}

TEST(SamplerFilterMaskTest, DenseProcessorReturnsSelectedRows) {
  SamplingParameters params = make_greedy_params(/*batch_size=*/3);
  params.sample_idxes = torch::tensor({2, 0}, torch::kLong);
  params.filter_mask = torch::tensor({{-10.F, 0.F}, {0.F, -10.F}});
  torch::Tensor logits = torch::tensor({{1.F, 2.F}, {3.F, 4.F}, {5.F, 6.F}});
  const torch::Tensor original = logits.clone();
  const torch::Tensor scores = Sampler::apply_logits_processors(logits, params);
  EXPECT_TRUE(torch::equal(logits, original));
  EXPECT_TRUE(torch::equal(scores, torch::tensor({{-5.F, 6.F}, {1.F, -8.F}})));
}

TEST(SamplerFilterMaskTest, OneRecPublishesFullAndSubsetRows) {
  const torch::Device device(torch::kCPU);
  const float neg_inf = -std::numeric_limits<float>::infinity();
  for (bool subset : {false, true}) {
    for (torch::ScalarType dtype : {torch::kFloat32, torch::kBFloat16}) {
      SamplingParameters params = make_greedy_params(/*batch_size=*/3);
      params.all_greedy_sample = false;
      params.all_random_sample = true;
      params.use_beam_search = true;
      params.logprobs = true;
      params.max_top_logprobs = 1;
      params.sample_idxes = subset ? torch::tensor({2, 0}, torch::kLong)
                                   : torch::arange(3, torch::kLong);
      params.do_sample =
          torch::ones({params.sample_idxes.numel()}, torch::kBool);
      params = params.to(device, torch::kFloat32);
      torch::Tensor logits =
          torch::tensor({{10.F, 0.F}, {0.F, 10.F}, {0.F, 10.F}})
              .to(device, dtype);
      const torch::Tensor original = logits.clone();
      const torch::Tensor mask =
          torch::tensor({{0.F, neg_inf}, {neg_inf, 0.F}, {neg_inf, 0.F}})
              .index_select(/*dim=*/0, params.sample_idxes);
      const SampleOutput output = RecSampler(RecPipelineType::kOneRecDefault)
                                      .forward(logits, params, mask);
      const torch::Tensor sampled =
          logits.index_select(/*dim=*/0, params.sample_idxes);
      EXPECT_TRUE(torch::equal(output.probs, sampled.softmax(-1).to(dtype)));
      EXPECT_EQ(output.probs.scalar_type(), dtype);
      EXPECT_EQ(sampled.isneginf().sum().item<int64_t>(),
                params.sample_idxes.numel());
      if (subset) {
        EXPECT_TRUE(
            torch::equal(logits[1], original[1].to(logits.scalar_type())));
      }
    }
  }
}

void check_request_filters(int64_t top_k, float top_p, bool mixed) {
  if (!Platform::is_mlu() || Platform::device_count() == 0) {
    GTEST_SKIP() << "MLU integration test requires a device";
  }
  const torch::Device device(Platform::type_torch(), 0);
  // Qwen3.5 and GLM-5.3-Flash vocabulary sizes.
  for (int64_t vocab_size : {248320, 154880}) {
    for (torch::ScalarType dtype : {torch::kFloat32, torch::kBFloat16}) {
      SCOPED_TRACE(vocab_size);
      SCOPED_TRACE(dtype);
      RequestSamplingParam first;
      first.temperature = 1.F;
      first.top_k = top_k;
      first.top_p = top_p;
      RequestSamplingParam second = first;
      if (mixed) {
        second.top_k = 8;
        second.top_p = 1.F;
      }
      SamplingParameters params;
      params.init({&first, &second}, {0, 1}, {0, 1}, {}, {}, {});
      EXPECT_EQ(params.top_k.defined(), top_k > 0 || mixed);
      EXPECT_EQ(params.top_p.defined(), top_p != 1.F);
      ASSERT_TRUE(params.all_random_sample);
      torch::Tensor original = torch::full({2, vocab_size}, -100.F);
      const torch::Tensor scores = -torch::arange(32, torch::kFloat32) * 0.2F;
      const torch::Tensor indices =
          torch::arange(32, torch::kLong) * 13 + vocab_size / 2;
      original[0].index_copy_(/*dim=*/0, indices, scores);
      original[1].index_copy_(/*dim=*/0, indices + 7, scores.flip({0}));
      original = original.to(dtype).to(torch::kFloat32);
      torch::Tensor expected = original.clone();
      const std::vector<const RequestSamplingParam*> requests{&first, &second};
      for (int64_t row = 0; row < 2; ++row) {
        auto [sorted, order] =
            original[row].sort(/*dim=*/-1, /*descending=*/true);
        const RequestSamplingParam& request = *requests[row];
        if (request.top_k > 0) {
          sorted.slice(/*dim=*/0, /*start=*/request.top_k)
              .fill_(-std::numeric_limits<float>::infinity());
        }
        const torch::Tensor probabilities = sorted.softmax(/*dim=*/-1);
        const torch::Tensor excluded =
            probabilities.cumsum(/*dim=*/-1) - probabilities > request.top_p;
        sorted.masked_fill_(excluded, -std::numeric_limits<float>::infinity());
        expected[row].scatter_(/*dim=*/0, order, sorted);
      }
      torch::Tensor logits = original.to(device, dtype);
      params = params.to(device, torch::kFloat32);
      const SampleOutput output = Sampler().forward(logits, params);
      const torch::Tensor actual = logits.to(torch::kCPU);
      const torch::Tensor expected_probs = expected.softmax(/*dim=*/-1);
      EXPECT_EQ(logits.scalar_type(), torch::kFloat32);
      EXPECT_EQ(output.probs.scalar_type(), dtype);
      EXPECT_TRUE(torch::equal(actual.isneginf(), expected.isneginf()));
      EXPECT_TRUE(torch::allclose(actual.softmax(/*dim=*/-1),
                                  expected_probs,
                                  /*rtol=*/1e-5,
                                  /*atol=*/1e-7));
      EXPECT_TRUE(
          torch::allclose(output.probs.to(torch::kCPU).to(torch::kFloat32),
                          expected_probs.to(dtype).to(torch::kFloat32),
                          /*rtol=*/1e-5,
                          /*atol=*/1e-7));
      const torch::Tensor tokens = output.next_tokens.to(torch::kCPU);
      EXPECT_TRUE(expected_probs.gather(/*dim=*/1, tokens.view({2, 1}))
                      .gt(0)
                      .all()
                      .item<bool>());
    }
  }
}

TEST(SamplerFilterMaskTest, RequestTopPOnlyUsesModelVocabulary) {
  check_request_filters(/*top_k=*/-1, /*top_p=*/0.9F, /*mixed=*/false);
}

TEST(SamplerFilterMaskTest, RequestTopKOnlyUsesModelVocabulary) {
  check_request_filters(/*top_k=*/8, /*top_p=*/1.F, /*mixed=*/false);
}

TEST(SamplerFilterMaskTest, RequestTopKTopPUsesModelVocabulary) {
  check_request_filters(/*top_k=*/8, /*top_p=*/0.9F, /*mixed=*/false);
}

TEST(SamplerFilterMaskTest, MixedRequestsUseModelVocabulary) {
  check_request_filters(/*top_k=*/-1, /*top_p=*/0.9F, /*mixed=*/true);
}

TEST(SamplerFilterMaskTest, MixedRequestsTreatZeroTopKAsDisabled) {
  check_request_filters(/*top_k=*/0, /*top_p=*/0.9F, /*mixed=*/true);
}

TEST(SamplerFilterMaskTest, RequestLargeTopKUsesModelVocabulary) {
  check_request_filters(/*top_k=*/10000, /*top_p=*/0.9F, /*mixed=*/false);
}

TEST(SamplerFilterMaskTest, MixedTopKOnlyUsesModelVocabulary) {
  check_request_filters(/*top_k=*/-1, /*top_p=*/1.F, /*mixed=*/true);
}

TEST(SamplerFilterMaskTest, TopPFilteringPreservesTargetLogitsAlias) {
  if (uses_device_random_sample() && Platform::device_count() == 0) {
    GTEST_SKIP() << "Random sampling backend device is unavailable";
  }

  SamplingParameters params = make_greedy_params(/*batch_size=*/1);
  params.do_sample = torch::ones({1}, torch::kBool);
  params.all_greedy_sample = false;
  params.all_random_sample = true;
  params.top_p = torch::tensor({0.95F});
  const torch::Device device = get_random_sample_device();
  params = params.to(device, torch::kFloat32);
  torch::Tensor logits = torch::tensor({{3.0F, 2.0F, 1.0F, 0.0F}}).to(device);
  // The target worker retains this alias before sampling. Speculative
  // verification must see the same filtered distribution as the sampler.
  const torch::Tensor target_logits = logits;

  Sampler sampler;
  SampleOutput output = sampler.forward(logits, params);

  EXPECT_EQ(target_logits.data_ptr(), logits.data_ptr());
  EXPECT_TRUE(torch::allclose(output.probs, target_logits.softmax(-1)));
  EXPECT_EQ(target_logits.softmax(-1).index({0, 3}).item<float>(), 0.0F);
}

}  // namespace
}  // namespace xllm
