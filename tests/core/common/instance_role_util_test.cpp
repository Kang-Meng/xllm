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

#include "core/common/instance_role_util.h"

#include <gtest/gtest.h>

#include <optional>

#include "core/common/options.h"
#include "core/common/types.h"

namespace xllm {
namespace {

constexpr InstanceRole::RoleType kAllRoles[] = {InstanceRole::DEFAULT,
                                                InstanceRole::PREFILL,
                                                InstanceRole::DECODE,
                                                InstanceRole::MIX};

// Mirrors the ContinuousScheduler::Options accessor shape, whose instance_role
// is a std::optional, without pulling the scheduler target into this CPU-only
// test.
class OptionalRoleOptions final {
 public:
  OptionalRoleOptions(bool enable_disagg_pd,
                      bool enable_pd_ooc,
                      std::optional<InstanceRole> instance_role)
      : enable_disagg_pd_(enable_disagg_pd),
        enable_pd_ooc_(enable_pd_ooc),
        instance_role_(instance_role) {}

  bool enable_disagg_pd() const { return enable_disagg_pd_; }
  bool enable_pd_ooc() const { return enable_pd_ooc_; }
  const std::optional<InstanceRole>& instance_role() const {
    return instance_role_;
  }

 private:
  bool enable_disagg_pd_ = false;
  bool enable_pd_ooc_ = false;
  std::optional<InstanceRole> instance_role_;
};

Options make_options(bool enable_disagg_pd,
                     bool enable_pd_ooc,
                     InstanceRole instance_role) {
  Options options;
  options.enable_disagg_pd(enable_disagg_pd);
  options.enable_pd_ooc(enable_pd_ooc);
  options.instance_role(instance_role);
  return options;
}

TEST(InstanceRoleUtilTest, PrefillRoleAloneIsNotThePdPrefillSide) {
  // Without disaggregated PD the continuous scheduler keeps decode on the same
  // instance whatever --instance_role says, so the role alone must not pass.
  for (const InstanceRole::RoleType role : kAllRoles) {
    const Options options = make_options(
        /*enable_disagg_pd=*/false, /*enable_pd_ooc=*/false, role);
    EXPECT_FALSE(is_pd_prefill_role(options)) << InstanceRole(role).to_string();
    EXPECT_FALSE(is_pd_prefill_only(options)) << InstanceRole(role).to_string();
  }
}

TEST(InstanceRoleUtilTest, PdPrefillSideRequiresBothDisaggPdAndPrefillRole) {
  for (const InstanceRole::RoleType role : kAllRoles) {
    const Options options = make_options(
        /*enable_disagg_pd=*/true, /*enable_pd_ooc=*/false, role);
    const bool expected = role == InstanceRole::PREFILL;
    EXPECT_EQ(is_pd_prefill_role(options), expected)
        << InstanceRole(role).to_string();
    EXPECT_EQ(is_pd_prefill_only(options), expected)
        << InstanceRole(role).to_string();
  }
}

TEST(InstanceRoleUtilTest, OnlineOfflineColocationIsNotPrefillOnly) {
  // A PD-OOC prefill instance decodes its offline requests locally, so it is
  // the PD prefill side but not a prefill-only instance.
  const Options options = make_options(/*enable_disagg_pd=*/true,
                                       /*enable_pd_ooc=*/true,
                                       InstanceRole::PREFILL);
  EXPECT_TRUE(is_pd_prefill_role(options));
  EXPECT_FALSE(is_pd_prefill_only(options));
}

TEST(InstanceRoleUtilTest, DefaultOptionsAreNeverExempt) {
  // Not const: Options has a defaulted (not user-provided) constructor and
  // PROPERTY fields without in-class initializers, so GCC rejects
  // default-initializing a const instance.
  Options options;
  EXPECT_EQ(options.instance_role(), InstanceRole::DEFAULT);
  EXPECT_FALSE(is_pd_prefill_role(options));
  EXPECT_FALSE(is_pd_prefill_only(options));
}

TEST(InstanceRoleUtilTest, OptionalRoleFollowsTheSameMatrix) {
  EXPECT_TRUE(is_pd_prefill_role(OptionalRoleOptions(
      /*enable_disagg_pd=*/true,
      /*enable_pd_ooc=*/false,
      InstanceRole(InstanceRole::PREFILL))));
  EXPECT_TRUE(is_pd_prefill_only(OptionalRoleOptions(
      /*enable_disagg_pd=*/true,
      /*enable_pd_ooc=*/false,
      InstanceRole(InstanceRole::PREFILL))));
  EXPECT_FALSE(is_pd_prefill_only(OptionalRoleOptions(
      /*enable_disagg_pd=*/true,
      /*enable_pd_ooc=*/true,
      InstanceRole(InstanceRole::PREFILL))));
  EXPECT_FALSE(is_pd_prefill_role(OptionalRoleOptions(
      /*enable_disagg_pd=*/false,
      /*enable_pd_ooc=*/false,
      InstanceRole(InstanceRole::PREFILL))));
  // An unset role (the scheduler options carry std::optional) is not prefill.
  EXPECT_FALSE(is_pd_prefill_role(OptionalRoleOptions(
      /*enable_disagg_pd=*/true, /*enable_pd_ooc=*/false, std::nullopt)));
}

}  // namespace
}  // namespace xllm
