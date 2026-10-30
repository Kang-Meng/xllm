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

#include <openssl/evp.h>
#include <openssl/pem.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace xllm {

// Only the independent termination authority holds the Ed25519 private key.
// Engines pin its public key. etcd is a durable delivery channel, NOT proof:
// a compromised/expired Engine credential cannot manufacture a valid receipt.
// Signing attests ALL ranks AND device resources of this exact incarnation
// have stopped writing. No timeout, registry event, or master waitpid suffices.
struct WriterTerminationProof {
  std::string engine_uid;
  std::string proof_id;
  std::string signature;
};

inline std::string termination_hex(const std::string& value) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string encoded;
  encoded.reserve(value.size() * 2);
  for (unsigned char byte : value) {
    encoded += kHex[byte >> 4];
    encoded += kHex[byte & 15];
  }
  return encoded;
}

inline std::optional<std::string> termination_unhex(const std::string& value) {
  if (value.size() % 2 != 0) {
    return std::nullopt;
  }
  std::string decoded;
  decoded.reserve(value.size() / 2);
  const auto digit = [](char character) -> int32_t {
    if (character >= '0' && character <= '9') {
      return character - '0';
    }
    if (character >= 'a' && character <= 'f') {
      return character - 'a' + 10;
    }
    return -1;
  };
  for (size_t index = 0; index < value.size(); index += 2) {
    const int32_t high = digit(value[index]);
    const int32_t low = digit(value[index + 1]);
    if (high < 0 || low < 0) {
      return std::nullopt;
    }
    decoded += static_cast<char>((high << 4) | low);
  }
  return decoded;
}

inline std::string writer_termination_key(const std::string& uid) {
  return "XLLM:WRITER_TERMINATED:" + termination_hex(uid);
}

inline bool valid_writer_termination(const WriterTerminationProof& proof) {
  const auto valid = [](const std::string& value, size_t limit) {
    if (value.empty() || value.size() > limit) {
      return false;
    }
    for (unsigned char character : value) {
      if (character < 0x21 || character > 0x7e) {
        return false;
      }
    }
    return true;
  };
  return valid(proof.engine_uid, 512) && valid(proof.proof_id, 128);
}

inline std::string writer_termination_payload(
    const WriterTerminationProof& proof) {
  // Version and scope are signed domain separators. Hex encoding gives one
  // canonical representation with no delimiter/identity ambiguity.
  return "XLLM_WRITER_TERMINATION_V1_ALL_EXECUTION_PROCESSES_AND_RESOURCES\n" +
         termination_hex(proof.engine_uid) + "\n" +
         termination_hex(proof.proof_id) + "\n";
}

inline std::string serialize_writer_termination(
    const WriterTerminationProof& proof) {
  return writer_termination_payload(proof) + termination_hex(proof.signature);
}

inline std::optional<WriterTerminationProof> parse_writer_termination(
    const std::string& wire) {
  if (wire.size() > 4096) {
    return std::nullopt;
  }
  std::vector<std::string> fields;
  fields.reserve(5);
  size_t start = 0;
  for (size_t index = 0; index <= wire.size(); ++index) {
    if (index == wire.size() || wire[index] == '\n') {
      fields.emplace_back(wire.substr(start, index - start));
      start = index + 1;
    }
  }
  if (fields.size() != 4) {
    return std::nullopt;
  }
  auto uid = termination_unhex(fields[1]);
  auto id = termination_unhex(fields[2]);
  auto signature = termination_unhex(fields[3]);
  if (!uid || !id || !signature) {
    return std::nullopt;
  }
  WriterTerminationProof proof{*uid, *id, *signature};
  if (!valid_writer_termination(proof) || proof.signature.size() != 64 ||
      serialize_writer_termination(proof) != wire) {
    return std::nullopt;
  }
  return proof;
}

inline std::shared_ptr<EVP_PKEY> load_termination_key(const std::string& pem,
                                                      bool private_key) {
  if (pem.empty() || pem.size() > 4096) {
    return nullptr;
  }
  std::unique_ptr<BIO, decltype(&BIO_free)> input(
      BIO_new_mem_buf(pem.data(), static_cast<int32_t>(pem.size())), BIO_free);
  if (!input) {
    return nullptr;
  }
  std::shared_ptr<EVP_PKEY> key(
      private_key ? PEM_read_bio_PrivateKey(
                        input.get(),
                        nullptr,
                        [](char*, int, int, void*) { return 0; },
                        nullptr)
                  : PEM_read_bio_PUBKEY(input.get(), nullptr, nullptr, nullptr),
      EVP_PKEY_free);
  return key && EVP_PKEY_id(key.get()) == EVP_PKEY_ED25519 ? key : nullptr;
}

inline bool sign_writer_termination(const std::string& private_key_pem,
                                    WriterTerminationProof* proof) {
  if (proof == nullptr || !valid_writer_termination(*proof)) {
    return false;
  }
  auto key = load_termination_key(private_key_pem, true);
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!key || !context ||
      EVP_DigestSignInit(context.get(), nullptr, nullptr, nullptr, key.get()) !=
          1) {
    return false;
  }
  const std::string payload = writer_termination_payload(*proof);
  std::string signature(64, '\0');
  size_t size = signature.size();
  if (EVP_DigestSign(context.get(),
                     reinterpret_cast<unsigned char*>(signature.data()),
                     &size,
                     reinterpret_cast<const unsigned char*>(payload.data()),
                     payload.size()) != 1 ||
      size != 64) {
    return false;
  }
  proof->signature = std::move(signature);
  return true;
}

inline bool verify_writer_termination(const std::string& public_key_pem,
                                      const WriterTerminationProof& proof) {
  if (!valid_writer_termination(proof) || proof.signature.size() != 64) {
    return false;
  }
  auto key = load_termination_key(public_key_pem, false);
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!key || !context ||
      EVP_DigestVerifyInit(
          context.get(), nullptr, nullptr, nullptr, key.get()) != 1) {
    return false;
  }
  const std::string payload = writer_termination_payload(proof);
  return EVP_DigestVerify(
             context.get(),
             reinterpret_cast<const unsigned char*>(proof.signature.data()),
             proof.signature.size(),
             reinterpret_cast<const unsigned char*>(payload.data()),
             payload.size()) == 1;
}

}  // namespace xllm
