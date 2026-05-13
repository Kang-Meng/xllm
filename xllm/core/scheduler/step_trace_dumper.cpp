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

#include "scheduler/step_trace_dumper.h"

#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <sstream>
#include <string>

namespace xllm {
namespace {

constexpr char kStepTraceHeader[] = "ts_us\tstep_kind\tqlen\tstep_latency_ms\n";

std::string to_step_kind(const BatchForwardType& batch_forward_type) {
  switch (batch_forward_type.value()) {
    case BatchForwardType::PREFILL:
    case BatchForwardType::CHUNKED_PREFILL:
      return "prefill";
    case BatchForwardType::DECODE:
      return "decode";
    case BatchForwardType::MIXED:
      return "mixed";
    case BatchForwardType::EMPTY:
      return "empty";
    default:
      return "unknown";
  }
}

}  // namespace

StepTraceDumper::StepTraceDumper(const std::string& file_path,
                                 int32_t queue_size,
                                 int32_t flush_interval_ms,
                                 int32_t flush_batch_size)
    : queue_(std::make_unique<folly::MPMCQueue<StepTraceRecord>>(
          static_cast<size_t>(std::max(queue_size, 1)))),
      file_path_(file_path),
      flush_interval_ms_(std::max(flush_interval_ms, 1)),
      flush_batch_size_(std::max(flush_batch_size, 1)) {
  writer_thread_ = std::thread(&StepTraceDumper::run, this);
}

StepTraceDumper::~StepTraceDumper() {
  stop_.store(true, std::memory_order_release);
  if (writer_thread_.joinable()) {
    writer_thread_.join();
  }
}

bool StepTraceDumper::try_enqueue(const StepTraceRecord& record) {
  const bool written = queue_->write(record);
  if (!written) {
    dropped_records_.fetch_add(1, std::memory_order_relaxed);
  }
  return written;
}

std::string StepTraceDumper::format_record_tsv(const StepTraceRecord& record) {
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(3);
  os << record.ts_us << '\t' << to_step_kind(record.batch_forward_type) << '\t'
     << record.qlen << '\t' << record.step_latency_ms << '\n';
  return os.str();
}

void StepTraceDumper::run() {
  file_ = fopen(file_path_.c_str(), "ab+");
  if (file_ == nullptr) {
    LOG(ERROR) << "Failed to open step trace dump file: " << file_path_;
    return;
  }

  fseek(file_, 0, SEEK_END);
  const int64_t file_size = static_cast<int64_t>(ftell(file_));
  if (file_size == 0) {
    fwrite(kStepTraceHeader, 1, sizeof(kStepTraceHeader) - 1, file_);
  }

  const auto flush_interval = std::chrono::milliseconds(flush_interval_ms_);
  auto next_flush_deadline = std::chrono::steady_clock::now() + flush_interval;
  int32_t pending_records = 0;

  while (!stop_.load(std::memory_order_acquire)) {
    StepTraceRecord record;
    if (queue_->read(record)) {
      write_buffer_.append(format_record_tsv(record));
      ++pending_records;
      if (pending_records >= flush_batch_size_) {
        flush(true);
        pending_records = 0;
        next_flush_deadline = std::chrono::steady_clock::now() + flush_interval;
      }
      continue;
    }

    const auto now = std::chrono::steady_clock::now();
    if (pending_records > 0 && now >= next_flush_deadline) {
      flush(true);
      pending_records = 0;
      next_flush_deadline = now + flush_interval;
      continue;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  StepTraceRecord record;
  while (queue_->read(record)) {
    write_buffer_.append(format_record_tsv(record));
  }
  flush(true);
  fclose(file_);
  file_ = nullptr;
}

void StepTraceDumper::flush(bool force_flush) {
  if (write_buffer_.empty() || file_ == nullptr) {
    return;
  }

  fwrite(write_buffer_.data(), 1, write_buffer_.size(), file_);
  if (force_flush) {
    fflush(file_);
  }
  write_buffer_.clear();
}

}  // namespace xllm
