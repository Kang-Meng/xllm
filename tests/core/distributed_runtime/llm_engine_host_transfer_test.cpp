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

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "core/distributed_runtime/llm_engine.h"

namespace xllm {
namespace {

class RecordingWorker final : public WorkerClient {
 public:
  void prefetch_from_storage(
      const std::shared_ptr<const StoragePrefetchRequest>& request,
      std::shared_ptr<PrefetchResult> result,
      size_t worker_index) override {
    prefetch_request_ = request;
    prefetch_result_ = std::move(result);
    prefetch_index_ = worker_index;
  }

  folly::SemiFuture<uint32_t> transfer_kv_blocks(
      const std::vector<BlockTransferInfo>& infos) override {
    ++offload_count_;
    offload_infos_ = infos;
    return folly::makeSemiFuture(static_cast<uint32_t>(infos.size()));
  }

  void transfer_kv_blocks(
      uint64_t batch_id,
      const std::vector<BlockTransferInfo>& infos) override {
    ++load_count_;
    load_batch_id_ = batch_id;
    load_infos_ = infos;
  }

  uint32_t offload_count_ = 0;
  uint32_t load_count_ = 0;
  uint64_t load_batch_id_ = 0;
  std::vector<BlockTransferInfo> offload_infos_;
  std::vector<BlockTransferInfo> load_infos_;
  std::shared_ptr<const StoragePrefetchRequest> prefetch_request_;
  std::shared_ptr<PrefetchResult> prefetch_result_;
  size_t prefetch_index_ = 0;
};

class ClientEngine final : public LLMEngine {
 public:
  ClientEngine(runtime::Options options,
               std::vector<std::shared_ptr<WorkerClient>> clients)
      : LLMEngine(std::move(options), std::move(clients)) {}
};

struct TransferTopology {
  uint32_t workers_;
  uint32_t dp_;
  uint32_t cp_;
  uint32_t rank_;
  std::vector<uint32_t> expected_;
};

class EngineHostTransferTest
    : public ::testing::TestWithParam<TransferTopology> {};

TEST_P(EngineHostTransferTest, TransfersEveryShardInOnlyTheRequestedDpGroup) {
  const auto& topology = GetParam();
  std::vector<std::shared_ptr<WorkerClient>> clients;
  clients.reserve(topology.workers_);
  std::vector<std::shared_ptr<RecordingWorker>> workers;
  workers.reserve(topology.workers_);
  for (uint32_t i = 0; i < topology.workers_; ++i) {
    auto worker = std::make_shared<RecordingWorker>();
    clients.emplace_back(worker);
    workers.emplace_back(std::move(worker));
  }
  runtime::Options options;
  options.world_size(static_cast<int32_t>(topology.workers_))
      .dp_size(static_cast<int32_t>(topology.dp_))
      .cp_size(static_cast<int32_t>(topology.cp_));
  ClientEngine engine(std::move(options), std::move(clients));
  EXPECT_EQ(engine.host_transfer_worker_count(), topology.expected_.size());
  BlockTransferInfo info(/*src_block_id=*/3, /*dst_block_id=*/7);
  info.transfer_type = TransferType::D2H2G;
  info.block_type = BlockType::C4;
  std::fill(std::begin(info.hash_key), std::end(info.hash_key), 42);
  auto futures = engine.transfer_kv_blocks(topology.rank_, {info});
  EXPECT_EQ(futures.size(), topology.expected_.size());
  for (auto& future : futures) {
    EXPECT_EQ(std::move(future).get(), 1U);
  }
  info.transfer_type = TransferType::H2D;
  engine.transfer_kv_blocks(topology.rank_, /*batch_id=*/123, {info});
  std::vector<uint32_t> offloaded;
  std::vector<uint32_t> loaded;
  offloaded.reserve(topology.workers_);
  loaded.reserve(topology.workers_);
  for (uint32_t i = 0; i < topology.workers_; ++i) {
    const auto& worker = *workers[i];
    EXPECT_LE(worker.offload_count_, 1U);
    EXPECT_LE(worker.load_count_, 1U);
    if (worker.offload_count_ != 0) {
      offloaded.emplace_back(i);
      ASSERT_EQ(worker.offload_infos_.size(), 1U);
      EXPECT_EQ(worker.offload_infos_[0].src_block_id, 3);
      EXPECT_EQ(worker.offload_infos_[0].dst_block_id, 7);
      EXPECT_EQ(worker.offload_infos_[0].transfer_type, TransferType::D2H2G);
    }
    if (worker.load_count_ == 0) {
      continue;
    }
    loaded.emplace_back(i);
    EXPECT_EQ(worker.load_batch_id_, 123U);
    ASSERT_EQ(worker.load_infos_.size(), 1U);
    EXPECT_EQ(worker.load_infos_[0].src_block_id, 3);
    EXPECT_EQ(worker.load_infos_[0].dst_block_id, 7);
    EXPECT_EQ(worker.load_infos_[0].block_type, BlockType::C4);
    EXPECT_EQ(worker.load_infos_[0].transfer_type, TransferType::H2D);
    EXPECT_TRUE(std::equal(std::begin(info.hash_key),
                           std::end(info.hash_key),
                           std::begin(worker.load_infos_[0].hash_key)));
  }
  EXPECT_EQ(offloaded, topology.expected_);
  EXPECT_EQ(loaded, topology.expected_);
}

TEST_P(EngineHostTransferTest, PrefetchWaitsForEveryShardAndStopsAtFirstMiss) {
  const auto& topology = GetParam();
  std::vector<std::shared_ptr<WorkerClient>> clients;
  std::vector<std::shared_ptr<RecordingWorker>> workers;
  clients.reserve(topology.workers_);
  workers.reserve(topology.workers_);
  for (uint32_t i = 0; i < topology.workers_; ++i) {
    auto worker = std::make_shared<RecordingWorker>();
    clients.emplace_back(worker);
    workers.emplace_back(std::move(worker));
  }
  runtime::Options options;
  options.world_size(static_cast<int32_t>(topology.workers_))
      .dp_size(static_cast<int32_t>(topology.dp_))
      .cp_size(static_cast<int32_t>(topology.cp_))
      .prefetch_timeout(0)
      .prefetch_batch_size(5);
  ClientEngine engine(std::move(options), std::move(clients));
  auto request = std::make_shared<StoragePrefetchRequest>();
  for (int32_t i = 0; i < 5; ++i) {
    BlockTransferInfo info(/*src_block_id=*/i, /*dst_block_id=*/i);
    info.transfer_type = TransferType::G2H;
    PrefetchUnit unit;
    unit.gated_blocks.emplace_back(info);
    request->units.emplace_back(std::move(unit));
  }
  size_t callbacks = 0;
  size_t common_prefix = 0;
  engine.prefetch_from_storage(
      topology.rank_,
      request,
      [] { return false; },
      [&](PrefetchSummary summary) {
        ++callbacks;
        common_prefix = 0;
        while (common_prefix < summary.gated_hits.size() &&
               summary.gated_hits[common_prefix] != 0) {
          ++common_prefix;
        }
      });
  std::vector<uint32_t> prefetched;
  prefetched.reserve(topology.workers_);
  for (uint32_t i = 0; i < topology.workers_; ++i) {
    if (workers[i]->prefetch_request_ != nullptr) {
      prefetched.emplace_back(i);
    }
  }
  ASSERT_EQ(prefetched, topology.expected_);
  for (size_t i = 0; i < topology.expected_.size(); ++i) {
    const auto& worker = *workers[topology.expected_[i]];
    EXPECT_EQ(worker.prefetch_request_, request);
    EXPECT_EQ(worker.prefetch_index_, i);
    EXPECT_EQ(worker.prefetch_result_->worker_count(),
              topology.expected_.size());
    EXPECT_EQ(callbacks, 0U);
    const bool last = i + 1 == topology.expected_.size();
    const std::vector<uint8_t> gated_hits =
        last ? std::vector<uint8_t>{1, 1, 1, 0, 1}
             : std::vector<uint8_t>{1, 1, 1, 1, 1};
    const std::vector<uint8_t> non_gated_hits(5, 1);
    EXPECT_EQ(worker.prefetch_result_->record_batch_result(
                  i, gated_hits, non_gated_hits),
              PrefetchControl::STOP);
    EXPECT_EQ(callbacks, 0U);
    worker.prefetch_result_->mark_worker_ended(i, /*worker_ok=*/true);
  }
  EXPECT_EQ(callbacks, 1U);
  EXPECT_EQ(common_prefix, 3U);
}

INSTANTIATE_TEST_SUITE_P(
    DpCp,
    EngineHostTransferTest,
    ::testing::Values(
        TransferTopology{8, 1, 1, 0, {0, 1, 2, 3, 4, 5, 6, 7}},
        TransferTopology{8, 1, 4, 0, {0, 1, 2, 3, 4, 5, 6, 7}},
        TransferTopology{16, 2, 1, 1, {8, 9, 10, 11, 12, 13, 14, 15}},
        TransferTopology{16, 2, 4, 0, {0, 1, 2, 3, 4, 5, 6, 7}},
        TransferTopology{16, 2, 4, 1, {8, 9, 10, 11, 12, 13, 14, 15}}));

}  // namespace
}  // namespace xllm
