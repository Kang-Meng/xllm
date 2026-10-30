/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/scheduler/prefill_footprint_budget.h"

#include <gtest/gtest.h>

#include <limits>

namespace xllm {
namespace {
TEST(PrefillFootprintBudgetTest, CapacityIsIndependentAcrossDpRanks) {
  PrefillFootprintBudget budget(2);
  ASSERT_TRUE(budget.fits(0, 2, 9));
  budget.reserve(0, 2);
  EXPECT_TRUE(budget.fits(1, 8, 9));
  EXPECT_FALSE(budget.fits(0, 8, 9));
}

TEST(PrefillFootprintBudgetTest, ContinuationsAndDecodeMarginShareTheirRank) {
  PrefillFootprintBudget budget(2);
  budget.set_reserved(0, 11);
  budget.reserve(1, 8);
  EXPECT_FALSE(budget.fits(0, 1, 10));
  EXPECT_TRUE(budget.fits(1, 2, 10));
  budget.reserve(1, 2);
  EXPECT_FALSE(budget.fits(1, 1, 10));
}

TEST(PrefillFootprintBudgetTest, AdditionCannotWrapIntoAdmission) {
  PrefillFootprintBudget budget(1);
  budget.set_reserved(0, std::numeric_limits<size_t>::max() - 1);
  EXPECT_FALSE(budget.fits(0, 3, std::numeric_limits<size_t>::max()));
}

TEST(PrefillFootprintBudgetTest, InvalidTopologyFailsBeforeIndexing) {
  EXPECT_DEATH(PrefillFootprintBudget(0), "dp_size");
  PrefillFootprintBudget budget(2);
  EXPECT_DEATH(budget.fits(2, 1, 9), "dp_rank");
}
}  // namespace
}  // namespace xllm
