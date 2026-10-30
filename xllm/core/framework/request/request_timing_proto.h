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

#include <cmath>

#include "core/framework/request/request_timing.h"
#include "disagg_pd.pb.h"

namespace xllm {
namespace detail {

// These numeric values are the PD wire contract, not enum declaration order.
static_assert(
    static_cast<uint32_t>(RequestTimingReason::MISSING_START) == 0 &&
        static_cast<uint32_t>(RequestTimingReason::COMPLETE) == 1 &&
        static_cast<uint32_t>(RequestTimingReason::WAITING_END) == 2 &&
        static_cast<uint32_t>(RequestTimingReason::INTERRUPTED) == 3 &&
        static_cast<uint32_t>(RequestTimingReason::PD_LOCAL) == 4 &&
        static_cast<uint32_t>(RequestTimingReason::UNKNOWN_BOUNDARY) == 5 &&
        static_cast<uint32_t>(RequestTimingReason::MISSING_END) == 6,
    "PD wire timing reason values must remain stable");

// Wire-constructed samples bypass span(); normalize them before PD restore.
inline RequestTimingSample request_timing_sample_from_proto(
    const proto::RequestPhaseTimingSample& source) {
  RequestTimingSample sample;
  // Only the contiguous wire-stage reasons 0..6 are accepted. The derived
  // UNSUPPORTED_OBJECT reason (7) and unknown future values stay invalid.
  sample.reason =
      source.reason() <= static_cast<uint32_t>(RequestTimingReason::MISSING_END)
          ? static_cast<RequestTimingReason>(source.reason())
          : RequestTimingReason::UNKNOWN_BOUNDARY;
  if (source.has_milliseconds() && std::isfinite(source.milliseconds()) &&
      source.milliseconds() >= 0.0) {
    sample.milliseconds = source.milliseconds();
    sample.valid = source.valid() && !source.partial() &&
                   sample.reason == RequestTimingReason::COMPLETE;
    sample.partial = !sample.valid;
    if (source.valid() != (sample.reason == RequestTimingReason::COMPLETE)) {
      sample.reason = RequestTimingReason::UNKNOWN_BOUNDARY;
      sample.valid = false;
      sample.partial = true;
    }
  } else if (source.has_milliseconds() || source.valid() || source.partial()) {
    sample.reason = RequestTimingReason::UNKNOWN_BOUNDARY;
  }
  return sample;
}

}  // namespace detail

inline void request_timing_to_proto(const RequestTimingSnapshot& snapshot,
                                    proto::RequestPhaseTiming* output) {
  output->Clear();
  const auto write_sample = [](const RequestTimingSample& sample,
                               proto::RequestPhaseTimingSample* target) {
    if (sample.milliseconds.has_value()) {
      target->set_milliseconds(*sample.milliseconds);
    }
    target->set_valid(sample.valid);
    target->set_partial(sample.partial);
    target->set_reason(static_cast<uint32_t>(sample.reason));
  };
  write_sample(snapshot.queued_time, output->mutable_queued_time());
  write_sample(snapshot.prefill_time, output->mutable_prefill_time());
  write_sample(snapshot.decode_time, output->mutable_decode_time());
}

inline RequestTimingSnapshot request_timing_from_proto(
    const proto::RequestPhaseTiming* input) {
  RequestTimingSnapshot result;
  if (input == nullptr) {
    return result;
  }
  result.queued_time =
      detail::request_timing_sample_from_proto(input->queued_time());
  result.prefill_time =
      detail::request_timing_sample_from_proto(input->prefill_time());
  result.decode_time =
      detail::request_timing_sample_from_proto(input->decode_time());
  return result;
}

}  // namespace xllm
