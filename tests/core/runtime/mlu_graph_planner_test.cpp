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

#include "core/runtime/mlu_graph_planner.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "core/framework/model/mtp_topk_state.h"
#include "core/runtime/mlu_graph_executor_impl.h"
#include "models/llm/mlu/mtp_topk_state.h"

namespace xllm::mlu {
namespace {

GraphPlannerConfig qwen_config() {
  GraphPlannerConfig config;
  config.options.block_size(16).num_decoding_tokens(1);
  config.capabilities.graph_history =
      MtpGraphHistoryPolicy::QWEN_FULL_ATTENTION;
  config.history_capacity = 32;
  config.max_tokens = 64;
  return config;
}

// A draft engine whose model supports both grouped graphs and native index
// sharing is the only configuration that admits top-k reuse steps.
GraphPlannerConfig topk_reuse_config() {
  GraphPlannerConfig config = qwen_config();
  config.options.is_draft_engine(true);
  config.capabilities.supports_grouped_mtp_graph = true;
  config.capabilities.supports_native_index_share_for_iteration = true;
  return config;
}

ModelInputParams decode_input(int64_t history, int64_t rows = 3) {
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.meta.num_sequences = rows;
  params.meta.q_max_seq_len = 1;
  params.meta.kv_max_seq_len = history;
  params.parallel.dp_global_token_nums = {static_cast<int32_t>(rows)};
  params.parallel.dp_is_decode = {1};
  for (int64_t row = 0; row <= rows; ++row) {
    params.attention.host.q_seq_lens.push_back(static_cast<int32_t>(row));
  }
  params.attention.device.block_tables = torch::zeros({rows, 2}, torch::kInt32);
  return params;
}

ModelInputParams grouped_input(int64_t rows = 3) {
  ModelInputParams params = decode_input(/*history=*/16, rows);
  params.attention.device.q_seq_lens = torch::zeros({rows + 1}, torch::kInt32);
  params.attention.device.kv_seq_lens = torch::zeros({rows + 1}, torch::kInt32);
  params.attention.device.new_cache_slots = torch::zeros({rows}, torch::kInt32);
  return params;
}

void expect_eager(const MluGraphPlanner& planner,
                  int64_t tokens,
                  const ModelInputParams& params,
                  EagerReason reason) {
  const GraphDecision decision = planner.plan({tokens, params});
  ASSERT_TRUE(std::holds_alternative<EagerReason>(decision));
  EXPECT_EQ(std::get<EagerReason>(decision), reason);
}

class PlannerTopkState final : public MtpTopkState {
 public:
  int64_t num_rows() const override { return 1; }
  torch::Device device() const override { return torch::Device(torch::kCPU); }
  MtpTopkStatePtr to(const torch::Device& /*device*/) const override {
    return nullptr;
  }
  MtpTopkStatePtr index_select_rows(
      const torch::Tensor& /*index*/) const override {
    return nullptr;
  }
};

// A non-native implementation proves admission depends on the graph contract,
// not the MLU model's concrete carrier or per-layer tensor layout.
class PlannerGraphTopkState final : public MtpTopkGraphState {
 public:
  explicit PlannerGraphTopkState(std::vector<int64_t> schema = {7, 11})
      : schema_(std::move(schema)) {}

  int64_t num_rows() const override { return 3; }
  torch::Device device() const override { return torch::Device(torch::kCPU); }
  MtpTopkStatePtr to(const torch::Device& /*device*/) const override {
    return std::make_shared<PlannerGraphTopkState>(schema_);
  }
  MtpTopkStatePtr index_select_rows(
      const torch::Tensor& /*index*/) const override {
    return nullptr;
  }
  std::vector<int64_t> graph_schema() const override { return schema_; }
  std::unique_ptr<MtpTopkGraphBuffer> make_graph_buffer(
      int64_t /*rows*/) const override {
    return nullptr;
  }

 private:
  std::vector<int64_t> schema_;
};

TEST(MtpTopkGraphBufferDeathTest, RejectsMissingBufferFromGraphContract) {
  const torch::Tensor tokens = torch::zeros({3}, torch::kInt32);
  ModelInputParams params = grouped_input();
  params.mtp_topk_state = std::make_shared<PlannerGraphTopkState>();
  GraphLayout layout;
  layout.num_reqs = 3;
  layout.padded_num_reqs = 3;
  layout.padded_num_tokens = 3;
  EXPECT_DEATH(GraphPersistentParam(tokens,
                                    tokens,
                                    params,
                                    layout,
                                    /*graph_max_kv_seq_len=*/32,
                                    /*main_block_table_columns=*/2),
               "graph contract must return a buffer");
}

TEST(MtpTopkGraphBufferDeathTest, RejectsProducerWithoutGraphContract) {
  const torch::Tensor tokens = torch::zeros({3}, torch::kInt32);
  GraphLayout layout;
  layout.num_reqs = 3;
  layout.padded_num_reqs = 3;
  layout.padded_num_tokens = 3;
  GraphPersistentParam buffers(tokens,
                               tokens,
                               grouped_input(),
                               layout,
                               /*graph_max_kv_seq_len=*/32,
                               /*main_block_table_columns=*/2);
  const auto state = std::make_shared<PlannerTopkState>();
  EXPECT_DEATH(buffers.store_topk_output(state),
               "state must implement the graph contract");
}

TEST(MtpTopkGraphBufferDeathTest, RejectsIncompatibleUpdatesBeforeCopying) {
  const auto make_state =
      [](int64_t rows, int64_t columns, bool first_present) {
        model::MluMtpTopkState::LayerStates layers;
        layers.reserve(2);
        layer::DsaTopkState layer(torch::ones({rows, columns}, torch::kInt32),
                                  torch::ones({rows}, torch::kInt32));
        if (first_present) {
          layers.emplace_back(std::move(layer));
          layers.emplace_back(std::nullopt);
        } else {
          layers.emplace_back(std::nullopt);
          layers.emplace_back(std::move(layer));
        }
        return std::make_shared<model::MluMtpTopkState>(std::move(layers));
      };
  const auto source =
      make_state(/*rows=*/3, /*columns=*/2, /*first_present=*/true);
  const auto buffer = source->make_graph_buffer(/*rows=*/4);
  buffer->update(*source);
  const PlannerTopkState incompatible;
  EXPECT_DEATH(buffer->update(incompatible), "incompatible MTP top-k state");
  const auto wide =
      make_state(/*rows=*/3, /*columns=*/4, /*first_present=*/true);
  EXPECT_DEATH(buffer->update(*wide), "schema changed");
  const auto missing =
      make_state(/*rows=*/3, /*columns=*/2, /*first_present=*/false);
  EXPECT_DEATH(buffer->update(*missing), "schema changed");
  const auto other_device = source->to(torch::Device(torch::kMeta));
  EXPECT_DEATH(buffer->update(*other_device), "device changed");
  const auto large =
      make_state(/*rows=*/5, /*columns=*/2, /*first_present=*/true);
  EXPECT_DEATH(buffer->update(*large), "exceeds row capacity");
}

TEST(MluGraphPlannerTest, AcceptsBackendIndependentTopkGraphContract) {
  GraphPlannerConfig config = topk_reuse_config();
  ModelInputParams params = grouped_input();
  params.mtp_topk_state = std::make_shared<PlannerGraphTopkState>();
  const auto decision = MluGraphPlanner(config).plan({3, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(decision));
  EXPECT_EQ(std::get<GraphPlan>(decision).key.mtp_topk_schema,
            (std::vector<int64_t>{7, 11}));
  params.mtp_topk_state = std::make_shared<PlannerTopkState>();
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
}

// Data-parallel ranks compare the whole key in one fixed-size packet, so a key
// that cannot fit is rejected here instead of being dropped at agreement time.
TEST(MluGraphPlannerTest, KeyLayoutBeyondAgreementPacketStaysEager) {
  const MluGraphPlanner planner(topk_reuse_config());
  ModelInputParams params = grouped_input();
  params.mtp_topk_state = std::make_shared<PlannerGraphTopkState>(
      std::vector<int64_t>(kMaxGraphKeyLayoutFields, /*value=*/8));
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(planner.plan({3, params})));
  params.mtp_topk_state = std::make_shared<PlannerGraphTopkState>(
      std::vector<int64_t>(kMaxGraphKeyLayoutFields + 1, /*value=*/8));
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
}

TEST(MluGraphPlannerTest, MixedKeyFieldsKeepAgreementBudget) {
  const MluGraphPlanner planner(topk_reuse_config());
  ModelInputParams params = grouped_input();
  params.multi_block_tables = {torch::zeros({3, 2}, torch::kInt32)};
  params.mtp_topk_state = std::make_shared<PlannerGraphTopkState>(
      std::vector<int64_t>(kMaxGraphKeyLayoutFields - 1, /*value=*/8));
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(planner.plan({3, params})));
  params.multi_block_tables.emplace_back(torch::zeros({3, 4}, torch::kInt32));
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
}

TEST(MluGraphPlannerTest, KeyVectorsKeepFieldBoundariesAndOrder) {
  const MluGraphPlanner planner(topk_reuse_config());
  ModelInputParams params = grouped_input();
  params.multi_block_tables = {torch::zeros({3, 2}, torch::kInt32)};
  params.mtp_topk_state =
      std::make_shared<PlannerGraphTopkState>(std::vector<int64_t>{3, 4});
  const GraphKey first = std::get<GraphPlan>(planner.plan({3, params})).key;
  const GraphKey copy = std::get<GraphPlan>(planner.plan({3, params})).key;
  EXPECT_TRUE(first == copy);
  EXPECT_EQ(GraphKeyHash{}(first), GraphKeyHash{}(copy));

  params.multi_block_tables.emplace_back(torch::zeros({3, 3}, torch::kInt32));
  params.mtp_topk_state =
      std::make_shared<PlannerGraphTopkState>(std::vector<int64_t>{4});
  const GraphKey split = std::get<GraphPlan>(planner.plan({3, params})).key;
  params.multi_block_tables = {torch::zeros({3, 3}, torch::kInt32),
                               torch::zeros({3, 2}, torch::kInt32)};
  const GraphKey reordered = std::get<GraphPlan>(planner.plan({3, params})).key;
  EXPECT_FALSE(first == split);
  EXPECT_FALSE(split == reordered);
  const std::unordered_set<GraphKey, GraphKeyHash> keys{
      first, copy, split, reordered};
  EXPECT_EQ(keys.size(), 3);
}

// The empty schema keys producer graphs, so a reuse step reporting it must not
// be admitted into the producer's cache entry.
TEST(MluGraphPlannerTest, ReuseStateWithReservedEmptySchemaStaysEager) {
  const MluGraphPlanner planner(topk_reuse_config());
  ModelInputParams params = grouped_input();
  params.mtp_topk_state =
      std::make_shared<PlannerGraphTopkState>(std::vector<int64_t>{});
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
}

TEST(MluGraphPlannerTest, ProducerDoesNotRequireTopkReuseCapability) {
  GraphPlannerConfig config = topk_reuse_config();
  config.capabilities.supports_native_index_share_for_iteration = false;
  const MluGraphPlanner planner(config);
  ModelInputParams params = grouped_input();
  const auto producer = planner.plan({3, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(producer));
  EXPECT_TRUE(std::get<GraphPlan>(producer).key.mtp_topk_schema.empty());
  params.mtp_topk_state = std::make_shared<PlannerGraphTopkState>();
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
}

TEST(MluGraphPlannerTest, StatefulTargetGroupingDoesNotEnableTopkReuse) {
  GraphPlannerConfig config = qwen_config();
  config.state_required = true;
  config.capabilities.supports_grouped_mtp_graph = true;
  config.capabilities.supports_native_index_share_for_iteration = true;
  ModelInputParams params = grouped_input();
  params.embedding.linear_state_ids = {1, 2, 3};
  const MluGraphPlanner target(config);
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(target.plan({3, params})));
  params.mtp_topk_state = std::make_shared<PlannerGraphTopkState>();
  expect_eager(target, 3, params, EagerReason::UNSUPPORTED_INPUT);
  config.options.is_draft_engine(true);
  const auto reuse = MluGraphPlanner(config).plan({3, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(reuse));
  EXPECT_EQ(std::get<GraphPlan>(reuse).key.mtp_topk_schema,
            (std::vector<int64_t>{7, 11}));
  config.capabilities.supports_grouped_mtp_graph = false;
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
}

TEST(MluGraphPlannerTest, TopkInputValidationPreservesReasonPriority) {
  GraphPlannerConfig config = topk_reuse_config();
  const MluGraphPlanner planner(config);
  ModelInputParams params = grouped_input();
  params.mtp_topk_state = std::make_shared<PlannerTopkState>();
  params.meta.batch_forward_type = BatchForwardType::PREFILL;
  params.attention.host.q_seq_lens = {0, 1};
  expect_eager(planner, 3, params, EagerReason::NON_DECODE);
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.mtp_topk_state = std::make_shared<PlannerGraphTopkState>();
  expect_eager(planner, 3, params, EagerReason::INVALID_LAYOUT);

  params.attention.host.q_seq_lens = {0, 1, 2, 3};
  params.attention.device.q_seq_lens = torch::Tensor();
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.attention.device.q_seq_lens = torch::empty(
      {4}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kMeta));
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.attention.device.q_seq_lens = torch::zeros({4}, torch::kInt32);
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(planner.plan({3, params})));
}

TEST(MluGraphPlannerTest, OrdinaryStagesKeepReasonPriority) {
  GraphPlannerConfig config = qwen_config();
  ModelInputParams params = decode_input(16);
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(
      MluGraphPlanner(config).plan({3, params})));

  params.meta.batch_forward_type = BatchForwardType::PREFILL;
  expect_eager(MluGraphPlanner(config), 3, params, EagerReason::NON_DECODE);
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.mtp_topk_state = std::make_shared<PlannerTopkState>();
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.mtp_topk_state.reset();
  params.embedding.input_embedding = torch::ones({3, 2});
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(
      MluGraphPlanner(config).plan({3, params})));
  params.embedding.input_embedding = torch::Tensor();
  params.is_spec_verify = true;
  expect_eager(MluGraphPlanner(config), 3, params, EagerReason::NON_DECODE);

  params.is_spec_verify = false;
  config.options.is_draft_engine(true);
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
}

TEST(MluGraphPlannerTest, OrdinaryDpAdmissionChecksCountsAndDecodeFlags) {
  GraphPlannerConfig config = qwen_config();
  config.options.dp_size(2).world_size(8);
  const MluGraphPlanner planner(config);
  ModelInputParams params = decode_input(16);
  params.parallel.dp_global_token_nums = {3, 3};
  params.parallel.dp_is_decode = {1, 1};
  const GraphDecision equal = planner.plan({3, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(equal));
  EXPECT_EQ(std::get<GraphPlan>(equal).layout.padded_num_tokens, 4);

  params.parallel.dp_global_token_nums = {2, 3};
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(planner.plan({3, params})));
  params.meta.q_max_seq_len = 2;
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.meta.q_max_seq_len = 1;
  params.parallel.dp_global_token_nums = {3, 0};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.parallel.dp_global_token_nums = {3, 3};
  params.parallel.dp_is_decode = {1, 0};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.parallel.dp_is_decode = {1};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.parallel.dp_is_decode = {1, 1};
  params.meta.q_max_seq_len = 0;
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
}

TEST(MluGraphPlannerTest, OrdinaryDpRejectsCountsOutsideConfiguredGroup) {
  GraphPlannerConfig config = qwen_config();
  config.options.dp_size(2).world_size(2);
  const MluGraphPlanner planner(config);
  ModelInputParams params = decode_input(16);
  // The former RunMode skipped DP validation for a singleton count and never
  // compared a longer count vector with dp_size. Both are malformed for DP=2.
  params.parallel.dp_global_token_nums = {3};
  params.parallel.dp_is_decode = {1};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.parallel.dp_global_token_nums = {3, 3, 3};
  params.parallel.dp_is_decode = {1, 1, 1};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
}

TEST(MluGraphPlannerTest, GroupedStagesAndRestrictionsStayDistinct) {
  GraphPlannerConfig config = qwen_config();
  config.options.is_draft_engine(true);
  config.capabilities.supports_grouped_mtp_graph = true;
  ModelInputParams params = grouped_input();
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(
      MluGraphPlanner(config).plan({3, params})));
  params.meta.batch_forward_type = BatchForwardType::PREFILL;
  expect_eager(MluGraphPlanner(config), 3, params, EagerReason::NON_DECODE);
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  config.options.cp_size(2);
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
  config.options.cp_size(1);
  params.prefill_without_cache = true;
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.prefill_without_cache = false;
  params.mtp_topk_state = std::make_shared<PlannerTopkState>();
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.mtp_topk_state.reset();
  params.attention.device.new_cache_slots = torch::Tensor();
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.attention.device.new_cache_slots = torch::zeros({3}, torch::kInt32);
  params.embedding.linear_state_ids = {1, 2};
  expect_eager(MluGraphPlanner(config), 3, params, EagerReason::INVALID_LAYOUT);
}

TEST(MluGraphPlannerTest, GroupedVerifyChecksStageAndDpMetadata) {
  GraphPlannerConfig config = qwen_config();
  config.options.enable_speculative_decode(true).speculative_algorithm("MTP");
  config.options.dp_size(2).world_size(2);
  config.capabilities.supports_grouped_mtp_graph = true;
  const MluGraphPlanner planner(config);
  ModelInputParams params = grouped_input();
  params.is_spec_verify = true;
  params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  params.num_accepted_tokens = torch::ones({3}, torch::kInt32);
  params.parallel.dp_global_token_nums = {3, 3};
  params.parallel.dp_is_decode = {1, 1};
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(planner.plan({3, params})));
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  expect_eager(planner, 3, params, EagerReason::NON_DECODE);
  params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  params.parallel.dp_global_token_nums = {3, 2};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.parallel.dp_global_token_nums = {3};
  params.parallel.dp_is_decode = {1};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.parallel.dp_global_token_nums = {3, 3};
  params.parallel.dp_is_decode = {1, 0};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.parallel.dp_is_decode = {1, 1};
  params.num_accepted_tokens = torch::ones({2}, torch::kInt32);
  expect_eager(planner, 3, params, EagerReason::INVALID_LAYOUT);
}

TEST(MluGraphPlannerTest, GroupedPaddingKeepsRequestsAndTpAlignment) {
  GraphPlannerConfig config = qwen_config();
  config.options.is_draft_engine(true).dp_size(2).world_size(12);
  config.capabilities.supports_grouped_mtp_graph = true;
  const MluGraphPlanner planner(config);
  ModelInputParams params = grouped_input(/*rows=*/12);
  params.meta.num_sequences = 3;
  params.attention.host.kpool_query_lens = {4, 4, 4};
  params.parallel.dp_global_token_nums = {12, 12};
  params.parallel.dp_is_decode = {1, 1};
  const GraphDecision decision = planner.plan({12, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(decision));
  const GraphPlan& plan = std::get<GraphPlan>(decision);
  EXPECT_EQ(plan.layout.tokens_per_request, 4);
  EXPECT_EQ(plan.layout.num_reqs, 3);
  EXPECT_EQ(plan.layout.padded_num_reqs, 6);
  EXPECT_EQ(plan.layout.padded_num_tokens, 24);

  config.options.enable_graph_mode_decode_no_padding(true);
  const GraphDecision no_padding = MluGraphPlanner(config).plan({12, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(no_padding));
  EXPECT_EQ(std::get<GraphPlan>(no_padding).layout.padded_num_reqs, 3);
  EXPECT_EQ(std::get<GraphPlan>(no_padding).layout.padded_num_tokens, 12);
}

TEST(MluGraphPlannerTest, HistoryAtCapacityReusesLayoutAcrossSteps) {
  MluGraphPlanner planner(qwen_config());
  ModelInputParams first = decode_input(16);
  ModelInputParams boundary = decode_input(32);
  GraphDecision first_decision = planner.plan({3, first});
  GraphDecision boundary_decision = planner.plan({3, boundary});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(first_decision));
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(boundary_decision));
  EXPECT_EQ(std::get<GraphPlan>(first_decision).history_capacity, 32);
  EXPECT_EQ(std::get<GraphPlan>(first_decision).main_block_table_columns, 3);
  EXPECT_TRUE(std::get<GraphPlan>(first_decision).key ==
              std::get<GraphPlan>(boundary_decision).key);
  EXPECT_EQ(std::get<GraphPlan>(first_decision).layout.padded_num_tokens, 4);
}

TEST(MluGraphPlannerTest, ExternalEmbeddingGetsSeparateGraphKey) {
  MluGraphPlanner planner(qwen_config());
  ModelInputParams tokens_only = decode_input(/*history=*/16);
  ModelInputParams external = decode_input(/*history=*/16);
  external.embedding.input_embedding = torch::ones({3, 2});

  const GraphDecision tokens_decision = planner.plan({3, tokens_only});
  const GraphDecision external_decision = planner.plan({3, external});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(tokens_decision));
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(external_decision));
  const GraphKey& token_key = std::get<GraphPlan>(tokens_decision).key;
  const GraphKey& external_key = std::get<GraphPlan>(external_decision).key;
  EXPECT_FALSE(token_key.has_input_embedding);
  EXPECT_TRUE(external_key.has_input_embedding);
  EXPECT_FALSE(token_key == external_key);
  EXPECT_NE(GraphKeyHash{}(token_key), GraphKeyHash{}(external_key));
  EXPECT_TRUE(external_key ==
              std::get<GraphPlan>(planner.plan({3, external})).key);
}

TEST(MluGraphPlannerTest, HistoryAboveCapacityFallsBackAndCanReturn) {
  MluGraphPlanner planner(qwen_config());
  ModelInputParams over = decode_input(33);
  ModelInputParams back = decode_input(16);
  EXPECT_EQ(std::get<EagerReason>(planner.plan({3, over})),
            EagerReason::HISTORY_LIMIT);
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(planner.plan({3, back})));
}

TEST(MluGraphPlannerTest, InvalidOffsetsAndDistinctMultiTablesStaySeparate) {
  MluGraphPlanner planner(qwen_config());
  ModelInputParams first = decode_input(16);
  ModelInputParams second = decode_input(16);
  first.multi_block_tables.push_back(torch::zeros({3, 4}, torch::kInt32));
  second.multi_block_tables.push_back(torch::zeros({3, 8}, torch::kInt32));
  EXPECT_FALSE(std::get<GraphPlan>(planner.plan({3, first})).key ==
               std::get<GraphPlan>(planner.plan({3, second})).key);
  second.attention.host.q_seq_lens = {1, 2, 3, 4};
  EXPECT_EQ(std::get<EagerReason>(planner.plan({3, second})),
            EagerReason::INVALID_LAYOUT);
}

TEST(MluGraphPlannerTest, NonDecodeAndTokenLimitReturnReasons) {
  GraphPlannerConfig config = qwen_config();
  config.max_tokens = 3;
  MluGraphPlanner planner(config);
  ModelInputParams params = decode_input(16);
  EXPECT_EQ(std::get<EagerReason>(planner.plan({3, params})),
            EagerReason::TOKEN_LIMIT);
  params.meta.batch_forward_type = BatchForwardType::PREFILL;
  EXPECT_EQ(std::get<EagerReason>(planner.plan({3, params})),
            EagerReason::NON_DECODE);
}

TEST(MluGraphPlannerTest, DraftAndVerifyUseCurrentStageMetadata) {
  GraphPlannerConfig draft_config = qwen_config();
  draft_config.options.is_draft_engine(true);
  draft_config.capabilities.supports_grouped_mtp_graph = true;
  MluGraphPlanner draft(draft_config);
  ModelInputParams params = decode_input(32);
  params.attention.device.q_seq_lens = torch::zeros({4}, torch::kInt32);
  params.attention.device.kv_seq_lens = torch::zeros({4}, torch::kInt32);
  params.attention.device.new_cache_slots = torch::zeros({3}, torch::kInt32);
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(draft.plan({3, params})));
  params.meta.kv_max_seq_len = 33;
  EXPECT_EQ(std::get<EagerReason>(draft.plan({3, params})),
            EagerReason::HISTORY_LIMIT);

  GraphPlannerConfig verify_config = qwen_config();
  verify_config.options.enable_speculative_decode(true).speculative_algorithm(
      "MTP");
  verify_config.capabilities.supports_grouped_mtp_graph = true;
  MluGraphPlanner verify(verify_config);
  params.meta.kv_max_seq_len = 32;
  params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  params.is_spec_verify = true;
  params.num_accepted_tokens = torch::ones({3}, torch::kInt32);
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(verify.plan({3, params})));
}

TEST(MluGraphPlannerTest, NativeDraftSharingSeparatesProducerAndReuseGraphs) {
  GraphPlannerConfig config = topk_reuse_config();
  ModelInputParams params = grouped_input();
  const auto producer = MluGraphPlanner(config).plan({3, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(producer));
  const auto make_state = [](int64_t rows, int64_t columns) {
    model::MluMtpTopkState::LayerStates layers;
    layers.reserve(2);
    layers.emplace_back(
        layer::DsaTopkState(torch::zeros({rows, columns}, torch::kInt32),
                            torch::ones({rows}, torch::kInt32)));
    layers.emplace_back(std::nullopt);
    return std::make_shared<model::MluMtpTopkState>(std::move(layers));
  };
  params.mtp_topk_state = make_state(/*rows=*/3, /*columns=*/8);
  const auto reuse = MluGraphPlanner(config).plan({3, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(reuse));
  EXPECT_FALSE(std::get<GraphPlan>(producer).key ==
               std::get<GraphPlan>(reuse).key);
  params.mtp_topk_state = make_state(/*rows=*/3, /*columns=*/16);
  const auto changed_width = MluGraphPlanner(config).plan({3, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(changed_width));
  EXPECT_FALSE(std::get<GraphPlan>(reuse).key ==
               std::get<GraphPlan>(changed_width).key);
  params.mtp_topk_state = make_state(/*rows=*/2, /*columns=*/8);
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.mtp_topk_state = make_state(/*rows=*/3, /*columns=*/8);
  config.capabilities.supports_native_index_share_for_iteration = false;
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
  config.capabilities.supports_native_index_share_for_iteration = true;
  config.options.is_draft_engine(false);
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
}

TEST(MluGraphPlannerTest, ModelContextHistoryReusesKeyAcrossContextLengths) {
  auto config = qwen_config();
  config.history_capacity = 32768;
  config.capabilities.graph_history = MtpGraphHistoryPolicy::MODEL_CONTEXT;
  MluGraphPlanner planner(config);
  std::optional<GraphKey> first_key;
  for (int64_t history : {1, 2048, 2049, 4097, 32768, 16}) {
    auto params = decode_input(history);
    params.attention.device.block_tables =
        torch::zeros({3, (history + 15) / 16}, torch::kInt32);
    const auto decision = planner.plan({3, params});
    ASSERT_TRUE(std::holds_alternative<GraphPlan>(decision));
    const auto& plan = std::get<GraphPlan>(decision);
    EXPECT_EQ(plan.history_capacity, config.history_capacity);
    EXPECT_EQ(plan.main_block_table_columns, 2048);
    if (first_key.has_value()) {
      EXPECT_TRUE(*first_key == plan.key);
    } else {
      first_key = plan.key;
    }
  }
  expect_eager(planner, 3, decode_input(32769), EagerReason::HISTORY_LIMIT);
}

}  // namespace
}  // namespace xllm::mlu
