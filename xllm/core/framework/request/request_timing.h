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

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

namespace xllm {

// Values 0..6 are PD wire-stage reasons; 7 and above are local derived reasons
// and must not be serialized as wire-stage reasons.
enum class RequestTimingReason : uint32_t {
  MISSING_START = 0,
  COMPLETE = 1,
  WAITING_END = 2,
  INTERRUPTED = 3,
  PD_LOCAL = 4,
  UNKNOWN_BOUNDARY = 5,
  MISSING_END = 6,
  UNSUPPORTED_OBJECT = 7,
};

struct RequestTimingSample {
  std::optional<double> milliseconds;
  bool valid = false;
  bool partial = false;
  RequestTimingReason reason = RequestTimingReason::MISSING_START;
};

struct RequestTimingSnapshot {
  RequestTimingSample server_ttft;
  // Match vLLM V1 FinishedRequestStats names; samples use milliseconds.
  RequestTimingSample queued_time;
  RequestTimingSample prefill_time;
  RequestTimingSample decode_time;
  bool terminal = false;
  // PD only correlates role-local samples, never remote clock timestamps.
  bool pd_local = false;
  // The original ingress observed a successful first-generation handoff ACK.
  bool pd_handoff_accepted = false;
  // Role-local observations never contain remote clock values and are not
  // interchangeable with imported upstream or complete-request stages.
  bool pd_imported = false;
  RequestTimingSample pd_queued_time;
  RequestTimingSample pd_decode_time;
  std::optional<uint64_t> pd_decode_tokens_at_start;
};

// Supply the committed token count only for a single supported output. The
// derived TPOT follows vLLM V1's core Decode span, not E2E minus frontend TTFT.
RequestTimingSample request_tpot_sample(
    const RequestTimingSnapshot& snapshot,
    std::optional<uint64_t> num_generation_tokens);

// Receipt-to-last-output duration per newly committed local token. Imported
// tokens form the baseline, not part of the denominator.
RequestTimingSample request_pd_tpot_sample(
    const RequestTimingSnapshot& snapshot,
    uint64_t num_generation_tokens);

// Shared API/Request/Sequence timing on one steady clock. Generation terminal
// is immutable; first-token observation can follow successful completion.
// PD imports durations, never timestamps; OOC return starts a new local span.
class RequestTiming final {
 public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;
  // Create timing for direct callers or retain the API's earlier arrival.
  static std::shared_ptr<RequestTiming> start_ingress(
      std::shared_ptr<RequestTiming> timing = nullptr,
      TimePoint now = Clock::now());
  void record_arrival(TimePoint now = Clock::now());
  void record_queued(TimePoint now = Clock::now());
  void record_scheduled(TimePoint now = Clock::now());
  bool has_scheduled() const {
    return scheduled_.load(std::memory_order_acquire);
  }
  bool needs_pd_scheduling() const {
    return pd_needs_scheduling_.load(std::memory_order_acquire);
  }
  // Like vLLM's stats update_from_output, consume confirmed token output, not
  // an unaccepted speculative candidate or an overlap placeholder.
  void update_from_output(TimePoint now = Clock::now());
  // Token mutation only signals confirmed output. The core result batch owns
  // the timestamp, after result processing, not individual append_token calls.
  void note_generated_token() {
    pending_output_.store(true, std::memory_order_release);
  }
  bool record_core_output(TimePoint now) {
    // Sequences share timing: only the first consumes the pending batch output
    // and takes mutex_; repeated calls without new output are lock-free.
    if (!pending_output_.exchange(false, std::memory_order_acq_rel)) {
      return false;
    }
    update_from_output(now);
    return true;
  }
  bool has_generated_token() const {
    return generated_.load(std::memory_order_acquire);
  }
  void record_first_token_observed(TimePoint now = Clock::now());
  bool has_observed_token() const {
    return observed_.load(std::memory_order_acquire);
  }
  void update_from_finished_request(bool generation_complete,
                                    TimePoint now = Clock::now());
  void mark_generation_events_incomplete();
  void mark_pd_local();
  void mark_pd_handoff_accepted();
  // After appending received tokens, before publishing to a scheduler.
  // Repeated calls within a role visit are idempotent.
  void record_pd_decode_start(uint64_t num_generation_tokens,
                              TimePoint now = Clock::now());

  RequestTimingSnapshot snapshot(TimePoint now = Clock::now()) const;

  // Preserve source-local Queue/Prefill only. Destination token events start
  // a new local span, not a cross-host Decode. Missing samples stay invalid.
  void restore_pd(const RequestTimingSnapshot& upstream);

 private:
  // Caller holds mutex_; publish the mirror only after updating its timestamp.
  static void set_timestamp_locked(std::optional<TimePoint>& timestamp,
                                   std::atomic<bool>& mirror,
                                   std::optional<TimePoint> value);

  mutable std::mutex mutex_;
  std::optional<TimePoint> arrival_ts_;
  std::optional<TimePoint> queued_ts_;
  std::optional<TimePoint> scheduled_ts_;
  std::optional<TimePoint> first_token_ts_;
  std::optional<TimePoint> last_token_ts_;
  std::optional<TimePoint> first_token_observed_ts_;
  std::optional<TimePoint> finished_ts_;
  std::optional<TimePoint> pd_queued_ts_;
  std::optional<TimePoint> pd_scheduled_ts_;
  std::optional<uint64_t> pd_decode_tokens_at_start_;
  // These flags mirror the matching *_ts_ optionals so the scheduler/token hot
  // paths can test them lock-free. Writers use set_timestamp_locked, including
  // restore_pd when clearing generation. pending_output_ is independent state,
  // not a timestamp mirror.
  std::atomic<bool> scheduled_{false};
  std::atomic<bool> generated_{false};
  std::atomic<bool> observed_{false};
  std::atomic<bool> pending_output_{false};
  std::atomic<bool> pd_needs_scheduling_{false};
  bool generation_complete_ = false;
  bool generation_events_complete_ = true;
  bool pd_local_ = false;
  bool pd_imported_ = false;
  bool pd_handoff_accepted_ = false;
  RequestTimingSample upstream_queue_;
  RequestTimingSample upstream_prefill_;
};

}  // namespace xllm
