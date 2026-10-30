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

#include "core/util/verbose_trace_logger.h"

#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace xllm {
namespace {

TEST(VerboseTraceLoggerTest, FlushesQueuedEntryOnShutdown) {
  const std::filesystem::path log_path =
      std::filesystem::temp_directory_path() /
      "xllm_verbose_trace_logger_test.log";
  std::error_code ec;
  std::filesystem::remove(log_path, ec);

  VerboseTraceLogger& logger = VerboseTraceLogger::get_instance();
  logger.initialize(/*enabled=*/true,
                    log_path.string(),
                    /*max_size_mb=*/1,
                    /*max_files=*/1,
                    /*queue_size=*/37);
  logger.log("event=test_entry");
  const VerboseTraceStats active_stats = logger.stats();
  EXPECT_TRUE(active_stats.enabled);
  EXPECT_EQ(active_stats.attempted, 1);
  EXPECT_EQ(active_stats.queue_overruns, 0);
  EXPECT_EQ(active_stats.queue_capacity, 37);
  logger.shutdown();

  const VerboseTraceStats stopped_stats = logger.stats();
  EXPECT_FALSE(stopped_stats.enabled);
  EXPECT_EQ(stopped_stats.attempted, 1);
  EXPECT_EQ(stopped_stats.queue_overruns, 0);
  EXPECT_EQ(stopped_stats.queue_capacity, 37);

  std::ifstream log_file(log_path);
  std::string line;
  ASSERT_TRUE(std::getline(log_file, line));
  EXPECT_NE(line.find("event=test_entry"), std::string::npos);

  log_file.close();
  std::filesystem::remove(log_path, ec);
}

TEST(VerboseTraceLoggerTest, ConcurrentLoggingAndShutdownAreLifetimeSafe) {
  const std::filesystem::path log_path =
      std::filesystem::temp_directory_path() /
      "xllm_verbose_trace_logger_shutdown_test.log";
  std::error_code ec;
  std::filesystem::remove(log_path, ec);

  VerboseTraceLogger& logger = VerboseTraceLogger::get_instance();
  logger.initialize(/*enabled=*/true,
                    log_path.string(),
                    /*max_size_mb=*/8,
                    /*max_files=*/1);
  std::atomic<bool> start{false};
  std::atomic<bool> race_shutdown{false};
  std::atomic<int32_t> writers_ready{0};
  std::vector<std::thread> writers;
  writers.reserve(16);
  for (int32_t writer = 0; writer < 16; ++writer) {
    writers.emplace_back(
        [&logger, &start, &race_shutdown, &writers_ready, writer]() {
          while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
          }
          logger.log("event=shutdown_race writer=" + std::to_string(writer) +
                     " entry=0");
          writers_ready.fetch_add(1, std::memory_order_acq_rel);
          while (!race_shutdown.load(std::memory_order_acquire)) {
            std::this_thread::yield();
          }
          for (int32_t entry = 1; entry < 1000; ++entry) {
            logger.log("event=shutdown_race writer=" + std::to_string(writer) +
                       " entry=" + std::to_string(entry));
          }
        });
  }
  start.store(true, std::memory_order_release);
  while (writers_ready.load(std::memory_order_acquire) !=
         static_cast<int32_t>(writers.size())) {
    std::this_thread::yield();
  }
  race_shutdown.store(true, std::memory_order_release);
  logger.shutdown();
  for (std::thread& writer : writers) {
    writer.join();
  }
  const VerboseTraceStats stopped_stats = logger.stats();
  EXPECT_FALSE(logger.enabled());
  EXPECT_FALSE(stopped_stats.enabled);

  std::ifstream log_file(log_path);
  uint64_t physical_lines = 0;
  std::string line;
  while (std::getline(log_file, line)) {
    ++physical_lines;
  }
  EXPECT_EQ(physical_lines,
            stopped_stats.attempted - stopped_stats.queue_overruns);
  log_file.close();
  std::filesystem::remove(log_path, ec);
}

TEST(VerboseTraceLogPathTest, SingleRankPreservesConfiguredPath) {
  EXPECT_EQ(resolve_verbose_trace_log_path(
                "log/verbose_trace.log", /*nnodes=*/1, /*node_rank=*/0),
            "log/verbose_trace.log");
}

TEST(VerboseTraceLogPathTest, MultiRankInsertsRankBeforeLastExtension) {
  EXPECT_EQ(resolve_verbose_trace_log_path(
                "log.v1/verbose.trace.log", /*nnodes=*/4, /*node_rank=*/2),
            "log.v1/verbose.trace_rank_2.log");
}

TEST(VerboseTraceLogPathTest, MultiRankSuffixesRankZero) {
  EXPECT_EQ(resolve_verbose_trace_log_path(
                "log/verbose_trace.log", /*nnodes=*/4, /*node_rank=*/0),
            "log/verbose_trace_rank_0.log");
}

TEST(VerboseTraceLogPathTest, MultiRankAppendsRankWithoutExtension) {
  EXPECT_EQ(resolve_verbose_trace_log_path(
                "log/verbose_trace", /*nnodes=*/4, /*node_rank=*/3),
            "log/verbose_trace_rank_3");
}

TEST(VerboseTraceLogPathTest, EmptyPathRemainsEmpty) {
  EXPECT_TRUE(resolve_verbose_trace_log_path("", /*nnodes=*/4, /*node_rank=*/1)
                  .empty());
}

}  // namespace
}  // namespace xllm
