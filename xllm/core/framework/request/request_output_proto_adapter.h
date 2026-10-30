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

#include <string>

#include "disagg_pd.pb.h"
#include "request_output.h"

namespace xllm {

// Converts an Engine-local output to the single authoritative Service wire
// representation. The same conversion is used for live delivery and for D's
// bounded seq=0 recovery payload.
bool request_output_to_proto(const RequestOutput& output,
                             const std::string& sender_engine_uid,
                             proto::DisaggStreamGeneration* generation);

}  // namespace xllm
