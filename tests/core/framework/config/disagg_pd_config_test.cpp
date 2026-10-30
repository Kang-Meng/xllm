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

#include "core/framework/config/disagg_pd_config.h"

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/common/global_flags.h"

namespace xllm {
namespace {

// super_node_id is a per-instance identity resolved in initialize() as:
// explicit non-empty --super_node_id flag wins; otherwise (flag unset or
// explicitly "") fall back to the SUPER_POD_ID env the launcher injects; both
// empty leaves "" (wildcard, opts out of cross super-node isolation). These
// tests pin that precedence and the wildcard semantics because a regression
// would silently break physical super-pod isolation at scheduling time.
//
// google::FlagSaver restores flag values and their is_default bit, and
// SetCommandLineOption flips is_default so the "explicit flag" branch is
// exercised exactly as a real command line would. config_json_file stays at its
// default empty value, so initialize() skips from_json() and never touches the
// process-global parsed JSON config.
class SuperNodeIdScope final {
 public:
  SuperNodeIdScope() {
    const char* previous = std::getenv("SUPER_POD_ID");
    if (previous != nullptr) {
      had_env_ = true;
      previous_env_ = previous;
    }
  }

  ~SuperNodeIdScope() {
    if (had_env_) {
      setenv("SUPER_POD_ID", previous_env_.c_str(), /*overwrite=*/1);
    } else {
      unsetenv("SUPER_POD_ID");
    }
  }

  void set_env(const std::string& value) {
    setenv("SUPER_POD_ID", value.c_str(), /*overwrite=*/1);
  }

  void clear_env() { unsetenv("SUPER_POD_ID"); }

 private:
  google::FlagSaver flag_saver_;
  bool had_env_ = false;
  std::string previous_env_;
};

TEST(DisaggPDConfigSuperNodeIdTest, ExplicitFlagWinsOverEnv) {
  SuperNodeIdScope scope;
  google::SetCommandLineOption("super_node_id", "pod-A");
  scope.set_env("pod-ENV");

  DisaggPDConfig config;
  config.initialize();

  EXPECT_EQ(config.super_node_id(), "pod-A");
}

TEST(DisaggPDConfigSuperNodeIdTest, EnvFallbackWhenFlagUnset) {
  SuperNodeIdScope scope;
  scope.set_env("pod-ENV");

  DisaggPDConfig config;
  config.initialize();

  EXPECT_EQ(config.super_node_id(), "pod-ENV");
}

TEST(DisaggPDConfigSuperNodeIdTest, ExplicitEmptyFlagFallsBackToEnv) {
  SuperNodeIdScope scope;
  // An explicitly empty flag is treated as "not chosen", so the env still wins;
  // this keeps launchers that pass --super_node_id="" able to inject via env.
  google::SetCommandLineOption("super_node_id", "");
  scope.set_env("pod-ENV");

  DisaggPDConfig config;
  config.initialize();

  EXPECT_EQ(config.super_node_id(), "pod-ENV");
}

TEST(DisaggPDConfigSuperNodeIdTest, BothUnsetLeavesWildcard) {
  SuperNodeIdScope scope;
  scope.clear_env();

  DisaggPDConfig config;
  config.initialize();

  EXPECT_TRUE(config.super_node_id().empty());
}

TEST(DisaggPDConfigSuperNodeIdTest, RegisteredInOptionCategory) {
  const std::vector<std::string>& option_names =
      DisaggPDConfig::option_category().option_names;
  EXPECT_NE(
      std::find(option_names.begin(), option_names.end(), "super_node_id"),
      option_names.end());
}

}  // namespace
}  // namespace xllm
