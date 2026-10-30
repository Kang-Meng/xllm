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

#include "core/common/registration_lifecycle.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(RegistrationLifecycleTest, KeepsServingOnlyWhileMembershipExists) {
  EXPECT_EQ(registration_reconcile_action(RegistrationLookupState::kPresent),
            RegistrationReconcileAction::kKeepServing);
}

TEST(RegistrationLifecycleTest, RetriesAnUnavailableRegistryWithoutRotation) {
  EXPECT_EQ(
      registration_reconcile_action(RegistrationLookupState::kUnavailable),
      RegistrationReconcileAction::kRetryLookup);
}

TEST(RegistrationLifecycleTest, FailsStopAfterAuthoritativeMembershipLoss) {
  EXPECT_EQ(registration_reconcile_action(RegistrationLookupState::kMissing),
            RegistrationReconcileAction::kFailStop);
}

TEST(RegistrationLifecycleTest, AcceptsProductionLeaseTiming) {
  EXPECT_TRUE(
      valid_registration_lease_timing(kMinRegistrationLeaseTtlSeconds, 0.5));
  EXPECT_TRUE(valid_registration_lease_timing(30, 10.0));
}

TEST(RegistrationLifecycleTest, RejectsLeaseBelowAvailabilityFloor) {
  EXPECT_FALSE(valid_registration_lease_timing(
      kMinRegistrationLeaseTtlSeconds - 1, 0.5));
}

TEST(RegistrationLifecycleTest, RejectsInvalidReconcileInterval) {
  EXPECT_FALSE(valid_registration_lease_timing(15, 0.0));
  EXPECT_FALSE(valid_registration_lease_timing(15, -0.5));
  EXPECT_FALSE(valid_registration_lease_timing(15, 5.1));
}

TEST(RegistrationLifecycleTest, KeepsMultipleRefreshOpportunitiesPerLease) {
  // The checked-out etcd client refreshes on its own cadence and does not
  // expose KeepAlive::RefreshInterval. The local contract is that a valid
  // lease still leaves at least three reconcile windows inside one TTL.
  EXPECT_TRUE(valid_registration_lease_timing(15, 5.0));
  EXPECT_TRUE(valid_registration_lease_timing(30, 10.0));
  EXPECT_FALSE(valid_registration_lease_timing(15, 6.0));
}

TEST(RegistrationLifecycleTest, UnavailableRegistryCannotExtendDeadline) {
  const RegistrationLeaseDeadline::TimePoint start{};
  RegistrationLeaseDeadline lease(start, std::chrono::seconds(15));
  EXPECT_FALSE(lease.expired(start + std::chrono::seconds(14)));
  EXPECT_TRUE(lease.expired(start + std::chrono::seconds(15)));
}

TEST(RegistrationLifecycleTest, SlowProofUsesRequestStartNotCompletion) {
  const RegistrationLeaseDeadline::TimePoint start{};
  RegistrationLeaseDeadline lease(start, std::chrono::seconds(15));
  EXPECT_TRUE(lease.refresh(start + std::chrono::seconds(5),
                            std::chrono::seconds(10),
                            start + std::chrono::seconds(8)));
  EXPECT_TRUE(lease.expired(start + std::chrono::seconds(15)));
}

TEST(RegistrationLifecycleTest, ExpiredOwnerCannotBeRevivedByLateResponse) {
  const RegistrationLeaseDeadline::TimePoint start{};
  RegistrationLeaseDeadline lease(start, std::chrono::seconds(15));
  EXPECT_FALSE(lease.refresh(start + std::chrono::seconds(14),
                             std::chrono::seconds(15),
                             start + std::chrono::seconds(15)));
  EXPECT_TRUE(lease.expired(start + std::chrono::seconds(15)));
}

TEST(RegistrationLifecycleTest, LiveLeaseNeedsUnexpiredAuthoritativeTtl) {
  const RegistrationLeaseDeadline::TimePoint start{};
  RegistrationLeaseDeadline lease(start, std::chrono::seconds(15));
  EXPECT_FALSE(lease.refresh(
      start, std::chrono::seconds(1), start + std::chrono::seconds(2)));
  EXPECT_TRUE(lease.refresh(start + std::chrono::seconds(10),
                            std::chrono::seconds(15),
                            start + std::chrono::seconds(11)));
  EXPECT_FALSE(lease.expired(start + std::chrono::seconds(24)));
  EXPECT_TRUE(lease.expired(start + std::chrono::seconds(25)));
}

}  // namespace
}  // namespace xllm
