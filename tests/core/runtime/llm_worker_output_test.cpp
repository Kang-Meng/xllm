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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/config/load_config.h"
#include "core/framework/config/model_config.h"
#include "framework/model/causal_lm.h"
#include "platform/device.h"
#include "runtime/llm_worker_impl.h"

namespace xllm {
namespace {

class OutputTestModel final : public CausalLM {
 public:
  OutputTestModel(ModelOutput output, torch::TensorOptions options)
      : output_(std::move(output)), options_(std::move(options)) {}

  ModelOutput forward(const torch::Tensor&,
                      const torch::Tensor&,
                      std::vector<KVCache>&,
                      const ModelInputParams& params) override {
    received_sampling_ = params.sampling.has_value();
    return output_;
  }

  torch::Tensor logits(const torch::Tensor& hidden,
                       const torch::Tensor& indices) override {
    ++logits_calls_;
    return hidden.index_select(0, indices.to(torch::kLong));
  }

  torch::Tensor logits(const torch::Tensor& hidden,
                       const torch::Tensor& indices,
                       torch::Tensor& selected_hidden) override {
    selected_hidden = logits(hidden, indices);
    return selected_hidden;
  }

  void load_model(std::unique_ptr<ModelLoader>) override {}
  torch::Device device() const override { return options_.device(); }
  const torch::TensorOptions& options() const override { return options_; }
  void prepare_expert_weight(int32_t, const std::vector<int32_t>&) override {}
  void update_expert_weight(int32_t) override {}

  int32_t logits_calls() const { return logits_calls_; }
  bool received_sampling() const { return received_sampling_; }

 private:
  ModelOutput output_;
  torch::TensorOptions options_;
  int32_t logits_calls_ = 0;
  bool received_sampling_ = false;
};

class OutputTestWorker final : public LLMWorkerImpl {
 public:
  OutputTestWorker(const ParallelArgs& parallel_args,
                   const torch::Device& device,
                   const runtime::Options& options,
                   std::unique_ptr<OutputTestModel> model)
      : LLMWorkerImpl(parallel_args, device, options) {
    ModelArgs args;
    args.model_type("llm_output_test").n_layers(1).hidden_size(4);
    context_ = ModelContext(parallel_args, args, QuantArgs(), model->options());
    model_ = std::move(model);
    model_executor_ =
        std::make_unique<Executor>(model_.get(), args, device, options);
    is_spec_draft_ = options.is_draft_engine();
  }

  std::optional<ForwardOutput> run(const ForwardInput& input,
                                   bool record_ready_event) {
    return execute_no_sync_on_stream(
        input, *compute_stream_, record_ready_event);
  }
};

class LLMWorkerOutputTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    saved_model_impl_ = ModelConfig::get_instance().model_impl();
    saved_manual_loader_ = LoadConfig::get_instance().enable_manual_loader();
    ModelConfig::get_instance().model_impl("native");
    LoadConfig::get_instance().enable_manual_loader(false);
    device_.set_device();
    device_.init_device_context();
    options_.enable_graph(false)
        .backend("llm")
        .enable_schedule_overlap(true)
        .enable_speculative_decode(true)
        .is_draft_engine(true)
        .num_speculative_tokens(3);
  }

  void TearDown() override {
    EXPECT_EQ(device_.synchronize_default_stream(), 0);
    ModelConfig::get_instance().model_impl(saved_model_impl_);
    LoadConfig::get_instance().enable_manual_loader(saved_manual_loader_);
  }

  ForwardInput make_input() {
    ForwardInput input;
    input.token_ids =
        torch::tensor({1, 0, 0}, torch::kInt).to(device_.unwrap());
    input.positions = torch::arange(3, input.token_ids.options());
    input.input_params.meta.num_sequences = 1;
    input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
    input.input_params.sampling.emplace();
    input.sampling_params.selected_token_idxes =
        torch::arange(3, input.token_ids.options().dtype(torch::kLong));
    input.return_selected_hidden = true;
    auto stream = device_.current_stream();
    input.metadata_ready_event = stream->record_event();
    return input;
  }

  void expect_retained_input(const ForwardOutput& output,
                             const ForwardInput& input) {
    ASSERT_EQ(output.retained_inputs.size(), 1);
    EXPECT_EQ(output.retained_inputs.front()->token_ids.data_ptr(),
              input.token_ids.data_ptr());
    EXPECT_TRUE(output.failed_request_ids.empty());
    EXPECT_EQ(output.ready_event != nullptr, GetParam());
    if (output.ready_event != nullptr) {
      EXPECT_TRUE(output.ready_event->synchronize());
    }
  }

  Device device_{0};
  runtime::Options options_;
  std::string saved_model_impl_;
  bool saved_manual_loader_ = false;
};

TEST_P(LLMWorkerOutputTest,
       CompletedProposalSkipsLogitsAndKeepsCommonRetirement) {
  const auto tensor_options =
      torch::TensorOptions().device(device_.unwrap()).dtype(torch::kFloat);
  torch::Tensor token_ids =
      torch::tensor({{2, 3, 4}}, torch::kLong).to(device_.unwrap());
  torch::Tensor probabilities = torch::full({1, 3, 5}, 0.2, tensor_options);
  torch::Tensor confidence = torch::full({1, 3}, 0.75, tensor_options);
  ModelOutput model_output;
  model_output.draft_proposal.emplace(token_ids, probabilities);
  model_output.draft_confidence_probs = confidence;
  auto model = std::make_unique<OutputTestModel>(std::move(model_output),
                                                 tensor_options);
  OutputTestModel* model_ptr = model.get();
  const ParallelArgs parallel_args(0, 1, nullptr);
  OutputTestWorker worker(
      parallel_args, device_.unwrap(), options_, std::move(model));
  ForwardInput input = make_input();

  std::optional<ForwardOutput> output = worker.run(input, GetParam());

  ASSERT_TRUE(output.has_value());
  ASSERT_TRUE(output->draft_proposal.has_value());
  EXPECT_EQ(model_ptr->logits_calls(), 0);
  EXPECT_TRUE(model_ptr->received_sampling());
  EXPECT_FALSE(output->logits.defined());
  EXPECT_FALSE(output->selected_hidden.defined());
  EXPECT_FALSE(output->sample_output.next_tokens.defined());
  EXPECT_TRUE(torch::equal(output->draft_proposal->token_ids(), token_ids));
  ASSERT_TRUE(output->draft_proposal->draft_probs().has_value());
  EXPECT_TRUE(
      torch::equal(*output->draft_proposal->draft_probs(), probabilities));
  EXPECT_TRUE(torch::equal(output->draft_confidence_probs, confidence));
  expect_retained_input(*output, input);
}

TEST_P(LLMWorkerOutputTest, HiddenStatesKeepTheExistingLogitsPath) {
  const auto tensor_options =
      torch::TensorOptions().device(device_.unwrap()).dtype(torch::kFloat);
  torch::Tensor hidden = torch::arange(12, tensor_options).view({3, 4});
  auto model =
      std::make_unique<OutputTestModel>(ModelOutput(hidden), tensor_options);
  OutputTestModel* model_ptr = model.get();
  const ParallelArgs parallel_args(0, 1, nullptr);
  OutputTestWorker worker(
      parallel_args, device_.unwrap(), options_, std::move(model));
  ForwardInput input = make_input();
  input.skip_sampling_for_logits_only = true;
  input.sampling_params.selected_token_idxes =
      torch::tensor({0, 2}, torch::kLong).to(device_.unwrap());

  std::optional<ForwardOutput> output = worker.run(input, GetParam());

  ASSERT_TRUE(output.has_value());
  EXPECT_FALSE(output->draft_proposal.has_value());
  EXPECT_EQ(model_ptr->logits_calls(), 1);
  EXPECT_TRUE(model_ptr->received_sampling());
  torch::Tensor expected =
      hidden.index_select(0, input.sampling_params.selected_token_idxes);
  EXPECT_TRUE(torch::equal(output->logits, expected));
  EXPECT_TRUE(torch::equal(output->selected_hidden, expected));
  EXPECT_TRUE(torch::equal(output->sample_output.embeddings, expected));
  EXPECT_FALSE(output->sample_output.next_tokens.defined());
  expect_retained_input(*output, input);
}

TEST_P(LLMWorkerOutputTest, OrdinaryHiddenStatesStillRunSampling) {
  const auto tensor_options =
      torch::TensorOptions().device(device_.unwrap()).dtype(torch::kFloat);
  torch::Tensor hidden = torch::arange(12, tensor_options).view({3, 4});
  auto model =
      std::make_unique<OutputTestModel>(ModelOutput(hidden), tensor_options);
  OutputTestModel* model_ptr = model.get();
  const ParallelArgs parallel_args(0, 1, nullptr);
  options_.enable_speculative_decode(false)
      .is_draft_engine(false)
      .num_speculative_tokens(0);
  OutputTestWorker worker(
      parallel_args, device_.unwrap(), options_, std::move(model));
  ForwardInput input = make_input();
  input.input_params.sampling.reset();
  input.return_selected_hidden = false;
  input.sampling_params.sample_idxes =
      input.sampling_params.selected_token_idxes;

  std::optional<ForwardOutput> output = worker.run(input, GetParam());

  ASSERT_TRUE(output.has_value());
  EXPECT_FALSE(output->draft_proposal.has_value());
  EXPECT_EQ(model_ptr->logits_calls(), 1);
  EXPECT_FALSE(model_ptr->received_sampling());
  EXPECT_TRUE(torch::equal(output->logits, hidden));
  EXPECT_TRUE(
      torch::equal(output->sample_output.next_tokens, hidden.argmax(-1)));
  expect_retained_input(*output, input);
}

INSTANTIATE_TEST_SUITE_P(Readiness,
                         LLMWorkerOutputTest,
                         ::testing::Values(false, true));

}  // namespace
}  // namespace xllm
