/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <folly/MPMCQueue.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>

#include "framework/batch/batch_forward_type.h"

namespace xllm {

struct StepTraceRecord {
  int64_t ts_us = 0;
  BatchForwardType batch_forward_type;
  int64_t qlen = 0;
  std::string batch_ids;
  double step_latency_ms = 0.0;
};

class StepTraceDumper final {
 public:
  StepTraceDumper(const std::string& file_path,
                  int32_t queue_size,
                  int32_t flush_interval_ms,
                  int32_t flush_batch_size);
  ~StepTraceDumper();

  bool try_enqueue(StepTraceRecord record);

  uint64_t dropped_records() const {
    return dropped_records_.load(std::memory_order_relaxed);
  }

  static std::string format_record_tsv(const StepTraceRecord& record);

 private:
  void run();
  void flush(bool force_flush);

  std::unique_ptr<folly::MPMCQueue<StepTraceRecord>> queue_;
  std::string file_path_;
  int32_t flush_interval_ms_ = 1000;
  int32_t flush_batch_size_ = 1024;
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> dropped_records_{0};
  std::thread writer_thread_;
  std::string write_buffer_;
  FILE* file_ = nullptr;
};

}  // namespace xllm
