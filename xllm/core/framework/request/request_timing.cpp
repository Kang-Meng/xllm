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

#include "core/framework/request/request_timing.h"

#include <algorithm>

namespace xllm {
namespace {

RequestTimingSample span(const std::optional<RequestTiming::TimePoint>& start,
                         const std::optional<RequestTiming::TimePoint>& end,
                         RequestTiming::TimePoint partial_end,
                         RequestTimingReason incomplete_reason,
                         bool complete = true) {
  RequestTimingSample result;
  if (!start.has_value()) {
    return result;
  }
  const RequestTiming::TimePoint stop = end.value_or(partial_end);
  if (stop < *start) {
    result.reason = RequestTimingReason::UNKNOWN_BOUNDARY;
    return result;
  }
  result.milliseconds =
      std::chrono::duration<double, std::milli>(stop - *start).count();
  result.valid = complete && end.has_value();
  result.partial = !result.valid;
  result.reason =
      result.valid ? RequestTimingReason::COMPLETE : incomplete_reason;
  return result;
}

void make_pd_partial(RequestTimingSample& sample) {
  sample.valid = false;
  sample.partial = sample.milliseconds.has_value();
  sample.reason = RequestTimingReason::PD_LOCAL;
}

}  // namespace

RequestTimingSample request_tpot_sample(
    const RequestTimingSnapshot& snapshot,
    std::optional<uint64_t> num_generation_tokens) {
  RequestTimingSample sample;
  if (snapshot.pd_local) {
    // The request token count can include an imported upstream token. It is
    // not a reliable denominator for the role-local Decode span.
    sample.reason = RequestTimingReason::PD_LOCAL;
  } else if (!num_generation_tokens.has_value()) {
    sample.reason = RequestTimingReason::UNSUPPORTED_OBJECT;
  } else if (*num_generation_tokens > 0) {
    sample = snapshot.decode_time;
    if (sample.milliseconds.has_value()) {
      // Match vLLM V1's single-token zero without making a missing or
      // interrupted generation into a complete sample.
      sample.milliseconds =
          *num_generation_tokens > 1
              ? *sample.milliseconds /
                    static_cast<double>(*num_generation_tokens - 1)
              : 0.0;
    }
  }
  return sample;
}

RequestTimingSample request_pd_tpot_sample(
    const RequestTimingSnapshot& snapshot,
    uint64_t num_generation_tokens) {
  if (!snapshot.pd_local || !snapshot.pd_imported ||
      !snapshot.pd_decode_tokens_at_start.has_value() ||
      num_generation_tokens <= *snapshot.pd_decode_tokens_at_start) {
    return {};
  }
  RequestTimingSample sample = snapshot.pd_decode_time;
  if (sample.milliseconds.has_value()) {
    sample.milliseconds =
        *sample.milliseconds /
        static_cast<double>(num_generation_tokens -
                            *snapshot.pd_decode_tokens_at_start);
  }
  return sample;
}

void RequestTiming::set_timestamp_locked(std::optional<TimePoint>& timestamp,
                                         std::atomic<bool>& mirror,
                                         std::optional<TimePoint> value) {
  timestamp = value;
  mirror.store(timestamp.has_value(), std::memory_order_release);
}

std::shared_ptr<RequestTiming> RequestTiming::start_ingress(
    std::shared_ptr<RequestTiming> timing,
    TimePoint now) {
  if (!timing) {
    timing = std::make_shared<RequestTiming>();
  }
  timing->record_arrival(now);
  return timing;
}

void RequestTiming::record_arrival(TimePoint now) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!finished_ts_.has_value() && !arrival_ts_.has_value()) {
    arrival_ts_ = now;
  }
}

void RequestTiming::record_queued(TimePoint now) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!finished_ts_.has_value() && pd_local_ && !pd_queued_ts_.has_value()) {
    pd_queued_ts_ = now;
  }
  if (!finished_ts_.has_value() && !queued_ts_.has_value()) {
    queued_ts_ = now;
  }
}

void RequestTiming::record_scheduled(TimePoint now) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!finished_ts_.has_value() && pd_local_ && !pd_scheduled_ts_.has_value()) {
    pd_scheduled_ts_ = now;
    pd_needs_scheduling_.store(false, std::memory_order_release);
  }
  if (!finished_ts_.has_value() && !scheduled_ts_.has_value()) {
    set_timestamp_locked(scheduled_ts_, scheduled_, now);
  }
}

void RequestTiming::update_from_output(TimePoint now) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (finished_ts_.has_value()) {
    return;
  }
  // Several Sequences can commit concurrently. Aggregate the earliest/latest
  // local result event, regardless of lock acquisition order.
  const TimePoint first_token =
      first_token_ts_.has_value() ? std::min(*first_token_ts_, now) : now;
  last_token_ts_ =
      last_token_ts_.has_value() ? std::max(*last_token_ts_, now) : now;
  set_timestamp_locked(first_token_ts_, generated_, first_token);
}

void RequestTiming::record_first_token_observed(TimePoint now) {
  std::lock_guard<std::mutex> lock(mutex_);
  // A successful single-token generation may finish before its frontend
  // result is processed. Observation is independent of generation completion.
  if (!first_token_ts_.has_value() ||
      (finished_ts_.has_value() && !generation_complete_)) {
    return;
  }
  if (!first_token_observed_ts_.has_value() ||
      now < *first_token_observed_ts_) {
    set_timestamp_locked(first_token_observed_ts_, observed_, now);
  }
}

void RequestTiming::update_from_finished_request(bool generation_complete,
                                                 TimePoint now) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!finished_ts_.has_value()) {
    finished_ts_ = now;
    generation_complete_ = generation_complete;
  }
}

void RequestTiming::mark_pd_local() {
  std::lock_guard<std::mutex> lock(mutex_);
  pd_local_ = true;
}

void RequestTiming::mark_pd_handoff_accepted() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (pd_local_ && !pd_imported_ && !finished_ts_.has_value() &&
      first_token_ts_.has_value() && !first_token_observed_ts_.has_value()) {
    pd_handoff_accepted_ = true;
  }
}

void RequestTiming::mark_generation_events_incomplete() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!finished_ts_.has_value()) {
    generation_events_complete_ = false;
  }
}

void RequestTiming::record_pd_decode_start(uint64_t num_generation_tokens,
                                           TimePoint now) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!pd_imported_ || finished_ts_.has_value() ||
      pd_decode_tokens_at_start_.has_value() || num_generation_tokens == 0) {
    return;
  }
  pd_decode_tokens_at_start_ = num_generation_tokens;
  set_timestamp_locked(first_token_ts_, generated_, now);
  last_token_ts_ = now;
  pending_output_.store(false, std::memory_order_release);
}

RequestTimingSnapshot RequestTiming::snapshot(TimePoint now) const {
  std::lock_guard<std::mutex> lock(mutex_);
  RequestTimingSnapshot result;
  result.terminal = finished_ts_.has_value();
  result.pd_local = pd_local_;
  result.pd_handoff_accepted = pd_handoff_accepted_;
  result.pd_imported = pd_imported_;
  result.pd_decode_tokens_at_start = pd_decode_tokens_at_start_;
  // Freeze diagnostic partial spans at terminal. A missing end still yields
  // valid=false/partial=true and is never admitted as a complete metric.
  now = finished_ts_.value_or(now);
  const RequestTimingReason incomplete_reason =
      !result.terminal       ? RequestTimingReason::WAITING_END
      : generation_complete_ ? RequestTimingReason::MISSING_END
                             : RequestTimingReason::INTERRUPTED;
  result.server_ttft =
      span(arrival_ts_, first_token_observed_ts_, now, incomplete_reason);
  result.queued_time = span(queued_ts_, scheduled_ts_, now, incomplete_reason);
  result.prefill_time =
      span(scheduled_ts_, first_token_ts_, now, incomplete_reason);
  // Never extend Decode to terminal/response time, even for a partial sample.
  result.decode_time =
      span(first_token_ts_,
           last_token_ts_,
           now,
           incomplete_reason,
           generation_complete_ && generation_events_complete_);
  if (!generation_events_complete_) {
    result.decode_time.reason = RequestTimingReason::UNKNOWN_BOUNDARY;
    if (!first_token_ts_.has_value()) {
      result.prefill_time.reason = RequestTimingReason::UNKNOWN_BOUNDARY;
      result.server_ttft.reason = RequestTimingReason::UNKNOWN_BOUNDARY;
    }
  }
  if (pd_imported_) {
    result.queued_time = upstream_queue_;
    result.prefill_time = upstream_prefill_;
  }
  if (pd_local_) {
    result.pd_queued_time =
        span(pd_queued_ts_, pd_scheduled_ts_, now, incomplete_reason);
    // Save the complete receipt-to-output local span before classifying the
    // request-level Decode as partial. Never subtract P and D clocks.
    if (pd_decode_tokens_at_start_.has_value()) {
      result.pd_decode_time = result.decode_time;
    }
    // PR #2588 measures ingress admission through successful handoff ACK on
    // one local clock. Decode/other role-local samples remain partial.
    if (!pd_handoff_accepted_) {
      make_pd_partial(result.server_ttft);
    }
    make_pd_partial(result.decode_time);
  }
  return result;
}

void RequestTiming::restore_pd(const RequestTimingSnapshot& upstream) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (finished_ts_.has_value()) {
    return;
  }
  pd_local_ = true;
  pd_imported_ = true;
  pd_queued_ts_.reset();
  pd_scheduled_ts_.reset();
  pd_decode_tokens_at_start_.reset();
  pd_needs_scheduling_.store(true, std::memory_order_release);
  upstream_queue_ = upstream.queued_time;
  upstream_prefill_ = upstream.prefill_time;
  // OOC can return to this node. Don't append its old generation timestamps to
  // the new local role's span, or turn a handoff gap into a complete duration.
  last_token_ts_.reset();
  generation_events_complete_ = true;
  set_timestamp_locked(first_token_ts_, generated_, std::nullopt);
  pending_output_.store(false, std::memory_order_release);
  pd_handoff_accepted_ = false;
}

}  // namespace xllm
