#pragma once

#include <absl/time/time.h>

#include <cstdint>
#include <string>

namespace xllm {

struct TimeTraceEntry {
  std::string stage;
  absl::Time timestamp = absl::UnixEpoch();
  int32_t token_budget = 0;
};

}  // namespace xllm
