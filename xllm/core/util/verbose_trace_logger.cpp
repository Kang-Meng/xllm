/* Copyright 2025-2026 The xLLM Authors.

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

#include <glog/logging.h>
#include <spdlog/async_logger.h>
#include <spdlog/details/thread_pool.h>
#include <spdlog/sinks/rotating_file_sink.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <thread>

namespace xllm {

struct VerboseTraceLogger::State {
  std::shared_ptr<spdlog::details::thread_pool> thread_pool;
  std::shared_ptr<spdlog::logger> logger;
  uint64_t queue_capacity = 0;
};

namespace {

// One writer keeps file output strictly ordered.
constexpr size_t kWriterThreads = 1;

constexpr int64_t kBytesPerMiB = 1024 * 1024;

class ActiveLogCall final {
 public:
  explicit ActiveLogCall(std::atomic<uint64_t>* active) : active_(active) {}
  ~ActiveLogCall() { active_->fetch_sub(1, std::memory_order_seq_cst); }

 private:
  std::atomic<uint64_t>* active_;
};

}  // namespace

bool level_one_trace_enabled() { return VLOG_IS_ON(1); }

void log_level_one_trace(const char* file,
                         int32_t line,
                         const std::string& message) {
  if (level_one_trace_enabled()) {
    google::LogMessage(file, line, google::GLOG_INFO).stream() << message;
  }
}

std::string resolve_verbose_trace_log_path(const std::string& configured_path,
                                           int32_t nnodes,
                                           int32_t node_rank) {
  if (configured_path.empty() || nnodes <= 1) {
    return configured_path;
  }

  const std::filesystem::path path(configured_path);
  const std::string extension = path.extension().string();
  const size_t extension_position = configured_path.size() - extension.size();
  std::string resolved_path = configured_path;
  resolved_path.insert(extension_position,
                       "_rank_" + std::to_string(node_rank));
  return resolved_path;
}

VerboseTraceLogger& VerboseTraceLogger::get_instance() {
  static VerboseTraceLogger instance;
  return instance;
}

VerboseTraceLogger::~VerboseTraceLogger() { shutdown(); }

void VerboseTraceLogger::initialize(bool enabled,
                                    const std::string& file_path,
                                    int32_t max_size_mb,
                                    int32_t max_files,
                                    int32_t queue_size) {
  if (!enabled) {
    LOG(INFO) << "Verbose trace logging is disabled.";
    return;
  }
  if (std::atomic_load_explicit(&state_, std::memory_order_acquire) !=
      nullptr) {
    LOG(WARNING) << "Verbose trace logger already initialized; ignoring.";
    return;
  }
  if (file_path.empty()) {
    LOG(ERROR) << "Verbose trace logging enabled but no file path provided; "
                  "verbose trace logging stays disabled.";
    return;
  }

  std::error_code ec;
  const std::filesystem::path path(file_path);
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
      LOG(ERROR) << "Failed to create directory for verbose trace log "
                 << file_path << ": " << ec.message()
                 << "; verbose trace logging stays disabled.";
      return;
    }
  }

  // Guard against non-positive configuration that spdlog would reject.
  const size_t max_size_bytes =
      static_cast<size_t>(std::max(1, max_size_mb)) * kBytesPerMiB;
  const size_t total_files = static_cast<size_t>(std::max(1, max_files));
  const size_t queue_capacity = static_cast<size_t>(std::max(1, queue_size));
  // spdlog's max_files parameter counts rotated backups and excludes the
  // active file. The service option is the total on-disk file limit.
  const size_t backup_files = total_files - 1;

  std::shared_ptr<State> state = std::make_shared<State>();
  try {
    state->queue_capacity = queue_capacity;
    state->thread_pool = std::make_shared<spdlog::details::thread_pool>(
        queue_capacity, kWriterThreads);
    auto sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        file_path, max_size_bytes, backup_files);
    state->logger = std::make_shared<spdlog::async_logger>(
        "verbose_trace",
        std::move(sink),
        state->thread_pool,
        // Never block the serving hot path: drop the oldest queued entry if the
        // background writer cannot keep up.
        spdlog::async_overflow_policy::overrun_oldest);
    // Prefix each entry with a millisecond wall-clock timestamp; the entry body
    // is emitted verbatim and spdlog appends the trailing newline.
    state->logger->set_pattern("%Y-%m-%d %H:%M:%S.%e %v");
    state->logger->set_level(spdlog::level::info);
    // Flush each entry so a live tail and post-crash inspection see it
    // promptly; this runs on the background thread, off the serving hot path.
    state->logger->flush_on(spdlog::level::info);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to open verbose trace log file " << file_path << ": "
               << e.what() << "; verbose trace logging stays disabled.";
    return;
  }

  attempted_.store(0, std::memory_order_release);
  last_queue_overruns_.store(0, std::memory_order_release);
  last_queue_capacity_.store(queue_capacity, std::memory_order_release);
  std::atomic_store_explicit(
      &state_, std::move(state), std::memory_order_release);
  enabled_.store(true, std::memory_order_release);
  LOG(INFO) << "Verbose trace logging enabled, writing to " << file_path
            << " (rotating at " << max_size_mb << " MiB, keeping up to "
            << total_files << " files, queue capacity " << queue_capacity
            << " entries).";
}

void VerboseTraceLogger::log(const std::string& entry) {
  if (!enabled_.load(std::memory_order_relaxed)) {
    return;
  }
  active_log_calls_.fetch_add(1, std::memory_order_seq_cst);
  ActiveLogCall active_call(&active_log_calls_);
  // Close the race in which shutdown disables the logger after the first
  // fast-path read but before this call advertises itself as active.
  if (!enabled_.load(std::memory_order_seq_cst)) {
    return;
  }
  const std::shared_ptr<State> state =
      std::atomic_load_explicit(&state_, std::memory_order_acquire);
  if (state == nullptr) {
    return;
  }
  attempted_.fetch_add(1, std::memory_order_relaxed);
  // spdlog copies the message into its async ring buffer internally, so passing
  // by const-ref is safe (the thread-local source buffer can be reused
  // immediately after this returns).
  state->logger->info(entry);
}

VerboseTraceStats VerboseTraceLogger::stats() const {
  VerboseTraceStats result;
  result.attempted = attempted_.load(std::memory_order_acquire);
  result.enabled = enabled_.load(std::memory_order_acquire);
  const std::shared_ptr<State> state =
      std::atomic_load_explicit(&state_, std::memory_order_acquire);
  result.queue_overruns =
      state == nullptr ? last_queue_overruns_.load(std::memory_order_acquire)
                       : state->thread_pool->overrun_counter();
  result.queue_capacity =
      state == nullptr ? last_queue_capacity_.load(std::memory_order_acquire)
                       : state->queue_capacity;
  return result;
}

void VerboseTraceLogger::shutdown() {
  // The sequentially-consistent gate orders a producer's second enabled read
  // with the active-call sample: shutdown either observes that producer or the
  // producer observes disabled and cannot enqueue.
  enabled_.store(false, std::memory_order_seq_cst);
  while (active_log_calls_.load(std::memory_order_seq_cst) != 0) {
    std::this_thread::yield();
  }
  std::shared_ptr<State> state = std::atomic_exchange_explicit(
      &state_, std::shared_ptr<State>(), std::memory_order_acq_rel);
  if (state == nullptr) {
    return;
  }
  state->logger->flush();
  last_queue_overruns_.store(state->thread_pool->overrun_counter(),
                             std::memory_order_release);
  // Destroying State drains its queue and joins the writer. The active-call
  // fence above guarantees no producer can enqueue after these final counters
  // are sampled.
}

}  // namespace xllm
