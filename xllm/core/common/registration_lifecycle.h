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

#include <chrono>
#include <cstdint>

namespace xllm {

enum class RegistrationLookupState {
  kPresent,
  kMissing,
  kUnavailable,
};

enum class RegistrationReconcileAction {
  kKeepServing,
  kRetryLookup,
  kFailStop,
};

// Membership loss is fail-stop because an incarnation is inseparable from its
// in-memory attempt ledger. The lease therefore must tolerate ordinary etcd
// elections and short control-plane stalls; a three-second lease is too close
// to those normal operating delays.
inline constexpr int32_t kMinRegistrationLeaseTtlSeconds = 15;

constexpr bool valid_registration_lease_timing(
    int32_t ttl_seconds,
    double reconcile_interval_seconds) {
  return ttl_seconds >= kMinRegistrationLeaseTtlSeconds &&
         reconcile_interval_seconds > 0.0 &&
         reconcile_interval_seconds <= static_cast<double>(ttl_seconds) / 3.0;
}

// An Engine incarnation is bound to its in-memory attempt ledger. Once etcd
// has authoritatively lost the membership key, the process must not republish
// that incarnation or rotate identity while retaining the old ledger. A fresh
// process is the only safe owner of a new incarnation.
constexpr RegistrationReconcileAction registration_reconcile_action(
    RegistrationLookupState state) {
  switch (state) {
    case RegistrationLookupState::kPresent:
      return RegistrationReconcileAction::kKeepServing;
    case RegistrationLookupState::kUnavailable:
      return RegistrationReconcileAction::kRetryLookup;
    case RegistrationLookupState::kMissing:
      return RegistrationReconcileAction::kFailStop;
  }
  return RegistrationReconcileAction::kFailStop;
}

// A lease proof is measured from the start of the authoritative TTL lookup,
// never from its completion. A delayed response cannot revive expired
// ownership. The owner must check this deadline independently of the blocking
// etcd client.
class RegistrationLeaseDeadline final {
 public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  RegistrationLeaseDeadline(TimePoint request_started,
                            std::chrono::seconds remaining_ttl)
      : deadline_(request_started + remaining_ttl) {}

  bool expired(TimePoint now) const { return now >= deadline_; }

  bool refresh(TimePoint request_started,
               std::chrono::seconds remaining_ttl,
               TimePoint now) {
    const TimePoint candidate = request_started + remaining_ttl;
    if (expired(now) || remaining_ttl.count() <= 0 || request_started > now ||
        candidate <= now) {
      return false;
    }
    deadline_ = candidate;
    return true;
  }

 private:
  TimePoint deadline_;
};

}  // namespace xllm
