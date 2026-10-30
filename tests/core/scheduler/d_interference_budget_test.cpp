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

#include "scheduler/d_interference_budget.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

DInterferenceConfig config() {
  return DInterferenceConfig{
      .default_tpot_guard_units = 100,
      .local_prefill_share_permyriad = 6000,
      .store_copy_share_permyriad = 3000,
      .reservation_capacity = 8,
  };
}

DInterferenceCandidate local(uint64_t delta) {
  return {DInterferenceClass::LOCAL_PREFILL, delta};
}

DInterferenceCandidate store(uint64_t delta) {
  return {DInterferenceClass::STORE_COPY, delta};
}

TEST(DInterferenceBudgetTest, FourModesRespectSharedAbsoluteGuard) {
  DInterferenceBudget off(config());
  off.update_decode_tpot_ub(40);
  EXPECT_EQ(off.snapshot().reservations, 0);

  DInterferenceBudget local_only(config());
  local_only.update_decode_tpot_ub(40);
  EXPECT_EQ(local_only.reserve("local", local(30)), DInterferenceStatus::OK);

  DInterferenceBudget store_only(config());
  store_only.update_decode_tpot_ub(40);
  EXPECT_EQ(store_only.reserve("store", store(15)), DInterferenceStatus::OK);

  DInterferenceBudget both(config());
  both.update_decode_tpot_ub(40);
  EXPECT_EQ(both.reserve("local", local(30)), DInterferenceStatus::OK);
  EXPECT_EQ(both.reserve("store", store(15)), DInterferenceStatus::OK);
  EXPECT_EQ(both.reserve("overflow", local(20)),
            DInterferenceStatus::ABSOLUTE_GUARD);
}

TEST(DInterferenceBudgetTest, ArrivalOrderCannotBypassClassShares) {
  DInterferenceBudget local_first(config());
  local_first.update_decode_tpot_ub(20);
  EXPECT_EQ(local_first.reserve("l1", local(40)), DInterferenceStatus::OK);
  EXPECT_EQ(local_first.reserve("s1", store(24)), DInterferenceStatus::OK);
  EXPECT_EQ(local_first.reserve("s2", store(1)),
            DInterferenceStatus::CLASS_SHARE);

  DInterferenceBudget store_first(config());
  store_first.update_decode_tpot_ub(20);
  EXPECT_EQ(store_first.reserve("s1", store(24)), DInterferenceStatus::OK);
  EXPECT_EQ(store_first.reserve("l1", local(40)), DInterferenceStatus::OK);
  EXPECT_EQ(store_first.reserve("s2", store(1)),
            DInterferenceStatus::CLASS_SHARE);
}

TEST(DInterferenceBudgetTest, TightenedDecodeGuardStopsNewOptionalWork) {
  DInterferenceBudget budget(config());
  budget.update_decode_tpot_ub(10);
  ASSERT_EQ(budget.reserve("existing", local(30)), DInterferenceStatus::OK);
  budget.update_decode_tpot_ub(80);
  EXPECT_EQ(budget.reserve("new", local(1)),
            DInterferenceStatus::ABSOLUTE_GUARD);
  EXPECT_EQ(budget.snapshot().reservations, 1);
  EXPECT_EQ(budget.release("existing"), DInterferenceStatus::OK);
}

TEST(DInterferenceBudgetTest, DuplicateIsIdempotentButConflictFails) {
  DInterferenceBudget budget(config());
  ASSERT_EQ(budget.reserve("same", local(10)), DInterferenceStatus::OK);
  EXPECT_EQ(budget.reserve("same", local(10)), DInterferenceStatus::OK);
  EXPECT_EQ(budget.snapshot().local_prefill_reserved_tpot, 10);
  EXPECT_EQ(budget.reserve("same", store(10)),
            DInterferenceStatus::DUPLICATE_CONFLICT);
  EXPECT_EQ(budget.release("same"), DInterferenceStatus::OK);
  EXPECT_EQ(budget.release("same"), DInterferenceStatus::UNKNOWN_RESERVATION);
}

TEST(DInterferenceBudgetTest, RejectsInvalidConfigurationAndCapacity) {
  DInterferenceConfig invalid = config();
  invalid.store_copy_share_permyriad = 7000;
  EXPECT_FALSE(DInterferenceBudget(invalid).valid());

  DInterferenceConfig limited = config();
  limited.reservation_capacity = 1;
  DInterferenceBudget budget(limited);
  ASSERT_EQ(budget.reserve("one", local(1)), DInterferenceStatus::OK);
  EXPECT_EQ(budget.reserve("two", local(1)),
            DInterferenceStatus::RECORD_CAPACITY);
}

}  // namespace
}  // namespace xllm
