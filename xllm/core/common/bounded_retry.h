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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace xllm {

class BoundedRetryBackoff final {
 public:
  using Duration = std::chrono::milliseconds;

  struct Config {
    Duration initial_delay{0};
    Duration maximum_delay{0};
  };

  explicit BoundedRetryBackoff(Config config) : config_(config) {}

  bool valid() const {
    return config_.initial_delay.count() > 0 &&
           config_.maximum_delay >= config_.initial_delay;
  }

  Duration maximum_delay() const { return config_.maximum_delay; }

  // consecutive_failures is one-based. retry_after is a lower-bound hint.
  // Remote hints are advisory across rolling versions and are clamped to the
  // local safety envelope instead of turning a temporary rejection terminal.
  std::optional<Duration> delay(
      uint32_t consecutive_failures,
      std::optional<Duration> retry_after = std::nullopt) const {
    if (!valid() || consecutive_failures == 0 ||
        (retry_after.has_value() && retry_after->count() < 0)) {
      return std::nullopt;
    }

    Duration result = config_.initial_delay;
    uint32_t remaining_shifts = consecutive_failures - 1;
    while (remaining_shifts > 0 && result < config_.maximum_delay) {
      if (result.count() > config_.maximum_delay.count() / 2) {
        result = config_.maximum_delay;
        break;
      }
      result *= 2;
      --remaining_shifts;
    }
    result = std::min(result, config_.maximum_delay);
    if (retry_after.has_value()) {
      result = std::max(result, std::min(*retry_after, config_.maximum_delay));
    }
    return result;
  }

 private:
  Config config_;
};

// Single-threaded preference bit for fair sharing between fresh ingress and
// ready retries. A successful take gives the other source first choice next.
class FairRetryPreference final {
 public:
  bool prefer_retry() const { return prefer_retry_; }

  void record_fresh_taken() { prefer_retry_ = true; }

  void record_retry_taken() { prefer_retry_ = false; }

 private:
  bool prefer_retry_ = false;
};

enum class FailureReportAction : uint8_t {
  kFirstFailure = 0,
  kPeriodicSummary = 1,
  kSuppressed = 2,
};

struct FailureObservation {
  FailureReportAction action = FailureReportAction::kSuppressed;
  BoundedRetryBackoff::Duration retry_delay{0};
  uint64_t consecutive_failures = 0;
  uint64_t suppressed_since_last_report = 0;
  uint64_t total_suppressed = 0;
  uint64_t duration_ms = 0;
  std::string first_reason;
  std::string last_reason;
};

struct FailureRecovery {
  uint64_t consecutive_failures = 0;
  uint64_t total_suppressed = 0;
  uint64_t duration_ms = 0;
  std::string first_reason;
  std::string last_reason;
};

// Tracks one single-threaded failure episode. The first failure is reportable,
// repeats are suppressed with periodic summaries, and success emits exactly
// one recovery record. Retry delay grows independently of log frequency.
class FailureEpisodeTracker final {
 public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;
  using Duration = std::chrono::milliseconds;

  struct Config {
    BoundedRetryBackoff::Config backoff;
    Duration summary_interval{0};
  };

  explicit FailureEpisodeTracker(Config config)
      : backoff_(config.backoff), summary_interval_(config.summary_interval) {}

  bool valid() const {
    return backoff_.valid() && summary_interval_.count() > 0;
  }

  FailureObservation record_failure(
      std::string reason,
      TimePoint now = Clock::now(),
      std::optional<Duration> retry_after = std::nullopt) {
    FailureObservation observation;
    if (!active_) {
      active_ = true;
      first_failure_at_ = now;
      last_report_at_ = now;
      first_reason_ = reason;
    }
    last_reason_ = std::move(reason);
    if (consecutive_failures_ < std::numeric_limits<uint64_t>::max()) {
      ++consecutive_failures_;
    }

    const uint32_t backoff_failures = static_cast<uint32_t>(std::min<uint64_t>(
        consecutive_failures_, std::numeric_limits<uint32_t>::max()));
    observation.retry_delay = backoff_.delay(backoff_failures, retry_after)
                                  .value_or(backoff_.maximum_delay());
    observation.consecutive_failures = consecutive_failures_;
    observation.duration_ms = elapsed_ms(first_failure_at_, now);
    observation.first_reason = first_reason_;
    observation.last_reason = last_reason_;

    if (consecutive_failures_ == 1) {
      observation.action = FailureReportAction::kFirstFailure;
    } else if (now >= last_report_at_ + summary_interval_) {
      observation.action = FailureReportAction::kPeriodicSummary;
      observation.suppressed_since_last_report = suppressed_since_last_report_;
      suppressed_since_last_report_ = 0;
      last_report_at_ = now;
    } else {
      observation.action = FailureReportAction::kSuppressed;
      if (suppressed_since_last_report_ <
          std::numeric_limits<uint64_t>::max()) {
        ++suppressed_since_last_report_;
      }
      if (total_suppressed_ < std::numeric_limits<uint64_t>::max()) {
        ++total_suppressed_;
      }
    }
    observation.total_suppressed = total_suppressed_;
    return observation;
  }

  std::optional<FailureRecovery> record_success(TimePoint now = Clock::now()) {
    if (!active_) {
      return std::nullopt;
    }
    FailureRecovery recovery;
    recovery.consecutive_failures = consecutive_failures_;
    recovery.total_suppressed = total_suppressed_;
    recovery.duration_ms = elapsed_ms(first_failure_at_, now);
    recovery.first_reason = std::move(first_reason_);
    recovery.last_reason = std::move(last_reason_);
    reset();
    return recovery;
  }

  bool active() const { return active_; }

 private:
  static uint64_t elapsed_ms(TimePoint start, TimePoint end) {
    if (end <= start) {
      return 0;
    }
    return static_cast<uint64_t>(
        std::chrono::duration_cast<Duration>(end - start).count());
  }

  void reset() {
    active_ = false;
    consecutive_failures_ = 0;
    suppressed_since_last_report_ = 0;
    total_suppressed_ = 0;
    first_reason_.clear();
    last_reason_.clear();
  }

  BoundedRetryBackoff backoff_;
  Duration summary_interval_;
  bool active_ = false;
  TimePoint first_failure_at_;
  TimePoint last_report_at_;
  uint64_t consecutive_failures_ = 0;
  uint64_t suppressed_since_last_report_ = 0;
  uint64_t total_suppressed_ = 0;
  std::string first_reason_;
  std::string last_reason_;
};

}  // namespace xllm
