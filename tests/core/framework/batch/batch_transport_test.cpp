/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <unistd.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "core/framework/batch/batch.h"
#include "core/framework/batch/batch_input_builder.h"
#include "core/framework/block/block_manager_impl.h"
#include "core/framework/model/model_args.h"
#include "core/framework/request/stopping_checker.h"
#include "core/runtime/forward_shared_memory_manager.h"

namespace xllm {
namespace {

TEST(BatchTransportTest, MultimodalPreviewMatchesMaterializedPayloadSize) {
  BlockManager::Options options;
  options.num_blocks(4).block_size(4);
  BlockManagerImpl manager(options);
  RequestSamplingParam sampling;
  StoppingChecker stopping;
  stopping.set_max_generated_tokens(4);
  SequenceParams params;
  params.seq_capacity = 32;
  params.sampling_param = &sampling;
  params.stopping_checker = &stopping;
  params.request_id = "multimodal_preview";
  Sequence sequence(
      0, {1, 2, 3, 4}, {}, {}, IncrementalDecoder("", 1, false, false), params);
  sequence.kv_state().add_blocks(BlockType::KV, manager.allocate(1));
  const auto first = torch::ones({2, 3}, torch::kFloat32);
  const auto second = torch::full({4, 3}, 2.0, torch::kFloat32);
  const std::vector<MMData> multimodal = {
      MMData(MMType::IMAGE, MMDict{{"pixel_values", first}}),
      MMData(MMType::IMAGE, MMDict{{"pixel_values", second}})};
  const std::vector<Sequence*> sequences = {&sequence};
  const std::vector<uint32_t> budgets = {4};
  const std::vector<torch::Tensor> embeddings;
  ModelArgs args;
  BatchInputBuilder preview_builder(sequences,
                                    budgets,
                                    embeddings,
                                    multimodal,
                                    /*swap_block_transfer_infos=*/nullptr,
                                    /*batch_id=*/0,
                                    &args,
                                    BatchForwardType::PREFILL,
                                    /*cp_size=*/1,
                                    /*thread_pool=*/nullptr,
                                    /*commit_sequence_state=*/false);
  auto preview = preview_builder.build_forward_input(
      /*num_decoding_tokens=*/0, /*min_decoding_batch_size=*/0);
  auto real = preview;
  real.input_params.multimodal.mm_data.batch(multimodal);
  const auto placeholder =
      preview.input_params.multimodal.mm_data.get<torch::Tensor>(
          "pixel_values");
  ASSERT_TRUE(placeholder.has_value());
  EXPECT_EQ(placeholder->device().type(), torch::kMeta);
  EXPECT_EQ(placeholder->numel(), 18);
  const auto materialized =
      real.input_params.multimodal.mm_data.get<torch::Tensor>("pixel_values");
  ASSERT_TRUE(materialized.has_value());
  EXPECT_TRUE(torch::equal(*materialized, torch::cat({first, second})));
  proto::PackedForwardInput packed;
  ASSERT_TRUE(forward_input_to_packed_proto(real, &packed));
  EXPECT_EQ(
      ForwardSharedMemoryManager::input_size(preview),
      packed.payload().size() + sizeof(ControlMetadata) + sizeof(uint64_t));
}

TEST(BatchTransportTest, MultimodalPreviewPreservesNonconcatenatedLayout) {
  const auto first = torch::ones({2, 3}, torch::kFloat32);
  const std::vector<torch::Tensor> alternatives = {
      torch::ones({4, 5}, torch::kFloat32),
      torch::ones({4, 3}, torch::kInt32),
      torch::empty(
          {4, 3},
          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMeta))};
  for (const auto& second : alternatives) {
    const std::vector<MMData> multimodal = {
        MMData(MMType::IMAGE, MMDict{{"pixel_values", first}}),
        MMData(MMType::IMAGE, MMDict{{"pixel_values", second}})};
    ForwardInput preview;
    ForwardInput real;
    preview.input_params.multimodal.mm_data.batch(
        multimodal, /*materialize_tensors=*/false);
    real.input_params.multimodal.mm_data.batch(multimodal);
    const auto& value =
        preview.input_params.multimodal.mm_data.data().at("pixel_values");
    ASSERT_TRUE(std::holds_alternative<std::vector<torch::Tensor>>(value));
    EXPECT_EQ(std::get<std::vector<torch::Tensor>>(value).size(), 2);
    EXPECT_EQ(ForwardSharedMemoryManager::input_size(preview),
              ForwardSharedMemoryManager::input_size(real));
  }
}

TEST(BatchTransportTest,
     CapacityCheckMatchesWriteAndRejectDoesNotPoisonChannel) {
  ForwardInput small;
  small.token_ids = torch::tensor({1, 2}, torch::kInt32);
  ForwardInput large;
  large.token_ids = torch::zeros({8192}, torch::kInt32);
  proto::PackedForwardInput packed;
  ASSERT_TRUE(forward_input_to_packed_proto(large, &packed));
  const uint64_t required =
      packed.payload().size() + sizeof(ControlMetadata) + sizeof(uint64_t);
  EXPECT_EQ(ForwardSharedMemoryManager::input_size(large), required);
  const std::string prefix = "transport_capacity_" + std::to_string(getpid());
  bool creator = false;
  ForwardSharedMemoryManager exact(
      prefix + "_exact", required, creator, ForwardType::RAW_INPUT);
  ASSERT_TRUE(exact.input_write(large));
  ForwardSharedMemoryManager writer(
      prefix, required - 1, creator, ForwardType::RAW_INPUT);
  ForwardSharedMemoryManager reader(
      prefix, required - 1, creator, ForwardType::RAW_INPUT);
  EXPECT_FALSE(writer.input_write(large));
  EXPECT_FALSE(writer.cancelled());
  ASSERT_TRUE(writer.input_write(small));
  ForwardInput restored;
  ASSERT_TRUE(reader.input_read(restored, torch::Device(torch::kCPU)));
  EXPECT_TRUE(torch::equal(restored.host_token_ids(), small.token_ids));
}

TEST(BatchTransportTest, InputCancellationIsVisibleToEveryReader) {
  const std::string name = "transport_cancel_" + std::to_string(getpid());
  bool creator = false;
  ForwardSharedMemoryManager writer(
      name, 4096, creator, ForwardType::RAW_INPUT);
  ForwardSharedMemoryManager reader(
      name, 4096, creator, ForwardType::RAW_INPUT);
  writer.cancel();
  EXPECT_TRUE(reader.cancelled());
  ForwardInput input;
  EXPECT_FALSE(reader.input_read(input, torch::Device(torch::kCPU)));
  EXPECT_FALSE(writer.input_write(input));
}

TEST(BatchTransportTest, MissingOutputTimesOutAndGroupCancellationStopsWait) {
  const std::string prefix = "transport_timeout_" + std::to_string(getpid());
  bool creator = false;
  ForwardSharedMemoryManager output(
      prefix + "_out", 4096, creator, ForwardType::RAW_OUTPUT);
  ForwardSharedMemoryManager input(
      prefix + "_in", 4096, creator, ForwardType::RAW_INPUT);
  RawForwardOutput result;
  EXPECT_FALSE(output.raw_output_read(result, std::chrono::milliseconds(5)));
  input.cancel();
  const auto start = std::chrono::steady_clock::now();
  EXPECT_FALSE(
      output.raw_output_read(result, std::chrono::seconds(30), &input));
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(1));
}

TEST(BatchTransportTest, PreviewAndSplitPreserveUnexecutedSequenceState) {
  BlockManager::Options options;
  options.num_blocks(4).block_size(4);
  BlockManagerImpl manager(options);
  RequestSamplingParam sampling;
  StoppingChecker stopping;
  stopping.set_max_generated_tokens(4);
  SequenceParams params;
  params.seq_capacity = 32;
  params.sampling_param = &sampling;
  params.stopping_checker = &stopping;
  params.request_id = "first";
  Sequence first(
      0, {1, 2, 3, 4}, {}, {}, IncrementalDecoder("", 1, false, false), params);
  params.request_id = "second";
  Sequence second(
      0, {5, 6, 7, 8}, {}, {}, IncrementalDecoder("", 1, false, false), params);
  first.kv_state().add_blocks(BlockType::KV, manager.allocate(1));
  second.kv_state().add_blocks(BlockType::KV, manager.allocate(1));
  Batch batch;
  batch.add(&first, 2);
  batch.add(&second, 4);
  ModelArgs args;
  Batch preview_batch = batch;
  auto prepared = preview_batch.prepare_forward_input_candidate(args, nullptr);
  const auto& preview = prepared.input();
  EXPECT_EQ(first.kv_state().kv_cache_tokens_num(), 0);
  EXPECT_EQ(second.kv_state().kv_cache_tokens_num(), 0);
  EXPECT_FALSE(preview.input_params.meta.requires_host_restore);
  auto parts = batch.split_by_request();
  ASSERT_EQ(parts.size(), 2);
  ASSERT_EQ(parts[0].size(), 1);
  ASSERT_EQ(parts[1].size(), 1);
  EXPECT_EQ(parts[0].get_allowed_max_tokens()[0], 2);
  EXPECT_EQ(parts[1].get_allowed_max_tokens()[0], 4);
  const auto first_input = parts[0].prepare_forward_input(args, nullptr);
  EXPECT_FALSE(first_input.input_params.meta.requires_host_restore);
  EXPECT_EQ(first.kv_state().kv_cache_tokens_num(), 2);
  EXPECT_EQ(second.kv_state().kv_cache_tokens_num(), 0);
  const auto second_input = parts[1].prepare_forward_input(args, nullptr);
  EXPECT_EQ(second.kv_state().kv_cache_tokens_num(), 4);
  EXPECT_EQ(first_input.host_token_ids().numel() +
                second_input.host_token_ids().numel(),
            preview.host_token_ids().numel());
}

TEST(BatchTransportTest, HostRestoreBatchRemainsAtomicAfterPreview) {
  Batch batch;
  batch.require_host_restore();
  ModelArgs args;
  Batch preview_batch = batch;
  auto prepared = preview_batch.prepare_forward_input_candidate(args, nullptr);
  const auto& preview = prepared.input();
  EXPECT_TRUE(preview.input_params.meta.requires_host_restore);
  auto parts = batch.split_by_request();
  ASSERT_EQ(parts.size(), 1);
  const auto input = parts[0].prepare_forward_input(args, nullptr);
  EXPECT_TRUE(input.input_params.meta.requires_host_restore);
}

TEST(BatchTransportTest,
     CandidateCommitsMultimodalScheduleWithoutRebuildingTokens) {
  BlockManager::Options options;
  options.num_blocks(4).block_size(4);
  BlockManagerImpl manager(options);
  RequestSamplingParam sampling;
  StoppingChecker stopping;
  stopping.set_max_generated_tokens(4);
  SequenceParams params;
  params.seq_capacity = 32;
  params.sampling_param = &sampling;
  params.stopping_checker = &stopping;
  params.request_id = "candidate_mm";
  MMDataItem item(MMType::IMAGE, MMDict{{"pixel_values", torch::ones({2, 3})}});
  item.mutable_state().mutable_token_pos().length = 4;
  item.mutable_state().mutable_seq_index() = 77;
  item.mutable_state().mutable_schedule_data().start_pos = 91;
  item.mutable_state().mutable_schedule_data().end_pos = 92;
  MMData data(MMType::IMAGE, MMItemVec{item, item});
  Sequence sequence(0,
                    {1, 2, 3, 4},
                    {},
                    data,
                    IncrementalDecoder("", 1, false, false),
                    params);
  sequence.kv_state().add_blocks(BlockType::KV, manager.allocate(1));
  sequence.set_mrope_positions(torch::arange(4, torch::kInt32).expand({3, -1}));
  Batch batch;
  batch.add(&sequence, 2);
  batch.require_host_restore();
  ModelArgs args;
  args.rope_scaling_rope_type("mrope");
  auto candidate = batch.prepare_forward_input_candidate(args, nullptr);
  const auto& before = sequence.mm_data().items<MMItemVec>().front().state();
  EXPECT_EQ(before.seq_index(), 77);
  EXPECT_EQ(before.schedule_data().start_pos, 91);
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 0);
  const auto pixel_values =
      candidate.input().input_params.multimodal.mm_data.get<torch::Tensor>(
          "pixel_values");
  ASSERT_TRUE(pixel_values.has_value());
  EXPECT_EQ(pixel_values->device().type(), torch::kMeta);
  const uint64_t size =
      ForwardSharedMemoryManager::input_size(candidate.input());
  const void* token_storage = candidate.input().token_ids.data_ptr();
  const auto input = batch.commit_forward_input(std::move(candidate));
  EXPECT_EQ(input.token_ids.data_ptr(), token_storage);
  EXPECT_EQ(ForwardSharedMemoryManager::input_size(input), size);
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 2);
  EXPECT_TRUE(input.input_params.meta.requires_host_restore);
  const auto& after = sequence.mm_data().items<MMItemVec>().front().state();
  EXPECT_EQ(after.seq_index(), 0);
  EXPECT_EQ(after.schedule_data().start_pos, 0);
  EXPECT_EQ(after.schedule_data().end_pos, 2);
  EXPECT_TRUE(torch::equal(
      *input.input_params.multimodal.mm_data.get<torch::Tensor>("pixel_values"),
      torch::ones({4, 3})));
  Batch preview_batch = batch;
  auto prepared = preview_batch.prepare_forward_input_candidate(args, nullptr);
  EXPECT_FALSE(prepared.input().input_params.meta.requires_host_restore);
}

TEST(BatchTransportTest, DecodeCandidateDefersHistoryTransferAndMropeUpdates) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(options);
  RequestSamplingParam sampling;
  StoppingChecker stopping;
  stopping.set_max_generated_tokens(4);
  SequenceParams params;
  params.seq_capacity = 32;
  params.sampling_param = &sampling;
  params.stopping_checker = &stopping;
  Sequence sequence(
      0, {1, 2, 3}, {}, {}, IncrementalDecoder("", 1, false, false), params);
  sequence.kv_state().add_blocks(BlockType::KV, manager.allocate(2));
  sequence.kv_state().add_blocks(BlockType::SWA, manager.allocate(2));
  sequence.kv_state().incr_kv_cache_tokens_num(/*size=*/3);
  sequence.append_token(4);
  sequence.set_mrope_positions(torch::zeros({3, 3}, torch::kInt32));
  sequence.update_mtp_bootstrap_embedding(torch::ones({1, 8}));
  TransferKVInfo transfer;
  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::KV);
  mapping.remote_ids = {100, 101};
  transfer.mappings.emplace_back(std::move(mapping));
  sequence.kv_state().set_transfer_kv_info(std::move(transfer));
  ModelArgs args;
  args.engram_layer_ids({1});
  args.engram_max_ngram_size(4);
  args.rope_scaling_rope_type("mrope");
  Batch batch;
  batch.add(&sequence, 1);
  {
    auto rejected = batch.prepare_forward_input_candidate(args, nullptr);
    EXPECT_FALSE(
        rejected.input().input_params.embedding.engram_history.empty());
  }
  EXPECT_FALSE(sequence.is_engram_history_restored());
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 3);
  EXPECT_EQ(sequence.kv_state().next_transfer_block_idx(), 0);
  EXPECT_TRUE(sequence.mrope_positions().defined());
  EXPECT_TRUE(sequence.get_mtp_bootstrap_embedding().defined());
  auto accepted = batch.prepare_forward_input_candidate(args, nullptr);
  const uint64_t size =
      ForwardSharedMemoryManager::input_size(accepted.input());
  const auto input = batch.commit_forward_input(std::move(accepted));
  EXPECT_EQ(ForwardSharedMemoryManager::input_size(input), size);
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 4);
  EXPECT_EQ(sequence.kv_state().next_transfer_block_idx(), 1);
  EXPECT_TRUE(sequence.is_engram_history_restored());
  EXPECT_FALSE(sequence.mrope_positions().defined());
  EXPECT_FALSE(sequence.get_mtp_bootstrap_embedding().defined());
  EXPECT_TRUE(
      torch::equal(input.input_params.embedding.mtp_bootstrap_embeddings,
                   torch::ones({1, 8})));
}

TEST(BatchTransportTest,
     ThreadedCandidatesCommitAllSequencesOnlyAfterAcceptance) {
  constexpr int32_t kSequences = 4;
  constexpr int32_t kTokens = 16384;
  BlockManager::Options options;
  options.num_blocks(128).block_size(1024);
  BlockManagerImpl manager(options);
  RequestSamplingParam sampling;
  StoppingChecker stopping;
  stopping.set_max_generated_tokens(4);
  SequenceParams params;
  params.seq_capacity = kTokens + 4;
  params.sampling_param = &sampling;
  params.stopping_checker = &stopping;
  std::vector<std::unique_ptr<Sequence>> sequences;
  sequences.reserve(kSequences);
  Batch batch;
  for (int32_t i = 0; i < kSequences; ++i) {
    sequences.emplace_back(
        std::make_unique<Sequence>(i,
                                   std::vector<int32_t>(kTokens, 1),
                                   torch::Tensor(),
                                   MMData(),
                                   IncrementalDecoder("", 1, false, false),
                                   params));
    sequences.back()->kv_state().add_blocks(BlockType::KV,
                                            manager.allocate(16));
    batch.add(sequences.back().get(), kTokens);
  }
  ThreadPool pool(/*num_threads=*/2);
  ModelArgs args;
  auto candidate = batch.prepare_forward_input_candidate(args, &pool);
  for (const auto& sequence : sequences) {
    EXPECT_EQ(sequence->kv_state().kv_cache_tokens_num(), 0);
  }
  batch.commit_forward_input(std::move(candidate));
  for (const auto& sequence : sequences) {
    EXPECT_EQ(sequence->kv_state().kv_cache_tokens_num(), kTokens);
  }
}

}  // namespace
}  // namespace xllm
