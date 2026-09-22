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

#include "mm_embedding_handler.h"

#include <absl/strings/ascii.h>
#include <absl/strings/escaping.h>
#include <openssl/rand.h>

#include <algorithm>
#include <cstdint>
#include <string>

#include "core/framework/multimodal/embedding_output.h"
#include "core/util/hash_util.h"
#include "core/util/utils.h"
#include "mm_input.h"

namespace xllm {

namespace {

bool parse_tensor(const xllm::proto::Tensor& in_tensor,
                  const BinaryPayload& payload,
                  torch::Tensor& out_tensor) {
  const auto& parameters = in_tensor.parameters();
  const auto binary_it = parameters.find("is_binary");
  const bool is_binary = binary_it != parameters.end() &&
                         binary_it->second.has_bool_param() &&
                         binary_it->second.bool_param();
  out_tensor = is_binary ? util::proto_to_torch(in_tensor, payload)
                         : util::proto_to_torch(in_tensor);
  return out_tensor.defined();
}

bool parse_embedding_output(const xllm::proto::Embedding& in_embedding_output,
                            const BinaryPayload& payload,
                            EmbeddingOutput& out_embedding_output) {
  if (!parse_tensor(in_embedding_output.embedding(),
                    payload,
                    out_embedding_output.embedding)) {
    return false;
  }

  out_embedding_output.hash_key = in_embedding_output.hash_key();
  out_embedding_output.metadata.clear();

  for (const auto& [key, proto_tensor] : in_embedding_output.metadata()) {
    torch::Tensor tensor;

    if (!parse_tensor(proto_tensor, payload, tensor)) {
      return false;
    }

    out_embedding_output.metadata.emplace(key, std::move(tensor));
  }

  return true;
}

}  // namespace

MMEmbeddingHandler::MMEmbeddingHandler(MMType::Value mm_type)
    : mm_type_(mm_type) {};

MMErrCode MMEmbeddingHandler::load(const MMContent& content,
                                   MMInputItem& input,
                                   const BinaryPayload& payload) {
  MMInputItem parsed_input;
  parsed_input.type = mm_type_;
  if (!parse_embedding_output(
          content.embedding, payload, parsed_input.embedding)) {
    LOG(ERROR) << "parse embedding failed";
    return MMErrCode::PARSE_EMB_ERR;
  }

  const std::string& hash_key = parsed_input.embedding.hash_key;
  if (hash_key.empty()) {
    // Generate once per keyless item, not per block or decode step. This is
    // a submission identity, not a content hash: entries still occupy cache
    // space, but later keyless submissions normally cannot reuse them.
    // Full-width random keys have negligible, not zero, collision risk.
    XXH3Key random_key;
    if (RAND_bytes(random_key.data,
                   static_cast<int32_t>(sizeof(random_key.data))) != 1) {
      // Reject the request instead of aborting or using uninitialized key
      // bytes.
      LOG(ERROR) << "Failed to generate a multimodal embedding cache key";
      return MMErrCode::PARSE_EMB_ERR;
    }
    parsed_input.hash_key = random_key;
  } else {
    if (hash_key.size() != 2 * XXH3_128BITS_HASH_VALUE_LEN ||
        !std::all_of(hash_key.begin(), hash_key.end(), absl::ascii_isxdigit)) {
      LOG(ERROR) << "embedding hash_key must contain exactly "
                 << 2 * XXH3_128BITS_HASH_VALUE_LEN
                 << " hexadecimal characters";
      return MMErrCode::PARSE_EMB_ERR;
    }
    const std::string key_bytes = absl::HexStringToBytes(hash_key);
    parsed_input.hash_key =
        XXH3Key(reinterpret_cast<const uint8_t*>(key_bytes.data()));
  }
  input = std::move(parsed_input);
  return MMErrCode::SUCCESS;
}

MMErrCode MMEmbeddingHandler::decode(MMInputItem& input) {
  return MMErrCode::SUCCESS;
}

}  // namespace xllm
