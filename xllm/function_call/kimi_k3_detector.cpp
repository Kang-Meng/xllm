/* Copyright 2026 The xLLM Authors.

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

#include "function_call/kimi_k3_detector.h"

#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <initializer_list>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace xllm {
namespace function_call {
namespace {

constexpr std::string_view kOpenToken = "<|open|>";
constexpr std::string_view kSepToken = "<|sep|>";
constexpr std::string_view kEndOfMessageToken = "<|end_of_msg|>";
constexpr std::string_view kResponseStart = "<|open|>response<|sep|>";
constexpr std::string_view kResponseEnd = "<|close|>response<|sep|>";
constexpr std::string_view kMessageEnd = "<|close|>message<|sep|>";

// XTML markers are structural token sequences, not escaped body text. Bodies
// must not contain literal protocol markers: the first matching close marker
// terminates the block, including inside JSON/string arguments. Attribute
// entity decoding below does not apply to bodies; do not invent a body codec.
constexpr std::string_view kToolsStart = "<|open|>tools<|sep|>";
constexpr std::string_view kToolsEnd = "<|close|>tools<|sep|>";
constexpr std::string_view kCallEnd = "<|close|>call<|sep|>";
constexpr std::string_view kArgumentEnd = "<|close|>argument<|sep|>";
constexpr std::string_view kJsonStart = "<|open|>json";
constexpr std::string_view kJsonEnd = "<|close|>json<|sep|>";

struct OpenTag {
  std::unordered_map<std::string, std::string> attributes;
  size_t body_start = 0;
};

struct ParseSnapshot {
  std::string content;
  std::vector<ToolCallItem> calls;
  bool protocol_complete = false;
};

bool is_attribute_key_char(char character) {
  return std::isalnum(static_cast<unsigned char>(character)) != 0 ||
         character == '_' || character == '.' || character == '-';
}

// Match KimiK3CppTemplate::escape_attribute: only &quot; and &amp; are
// decoded. Reject other entities instead of guessing an attribute codec.
std::string decode_attribute(std::string_view value) {
  std::string decoded;
  decoded.reserve(value.size());
  for (size_t index = 0; index < value.size();) {
    if (value.substr(index).starts_with("&quot;")) {
      decoded.push_back('"');
      index += 6;
    } else if (value.substr(index).starts_with("&amp;")) {
      decoded.push_back('&');
      index += 5;
    } else if (value[index] == '&') {
      throw std::runtime_error("Unsupported XTML attribute entity.");
    } else {
      decoded.push_back(value[index]);
      ++index;
    }
  }
  return decoded;
}

// XTML attributes follow the input template: key="value", with double
// quotes and entity escaping. Single-quoted/bare values are not this grammar.
std::unordered_map<std::string, std::string> parse_attributes(
    std::string_view attributes) {
  // Error offsets are relative to this attribute list; never log raw values.
  std::unordered_map<std::string, std::string> result;
  size_t cursor = 0;
  while (cursor < attributes.size()) {
    while (cursor < attributes.size() &&
           std::isspace(static_cast<unsigned char>(attributes[cursor])) != 0) {
      ++cursor;
    }
    if (cursor == attributes.size()) {
      break;
    }

    const size_t key_start = cursor;
    while (cursor < attributes.size() &&
           is_attribute_key_char(attributes[cursor])) {
      ++cursor;
    }
    if (key_start == cursor || cursor + 1 >= attributes.size() ||
        attributes[cursor] != '=' || attributes[cursor + 1] != '"') {
      throw std::runtime_error("Malformed XTML attributes at byte " +
                               std::to_string(cursor) + ".");
    }
    const std::string key(attributes.substr(key_start, cursor - key_start));
    cursor += 2;
    const size_t value_start = cursor;
    const size_t quote = attributes.find('"', cursor);
    if (quote == std::string_view::npos) {
      throw std::runtime_error("Unclosed XTML attribute value at byte " +
                               std::to_string(value_start) + ".");
    }
    if (result.contains(key)) {
      throw std::runtime_error("Duplicate XTML attribute at byte " +
                               std::to_string(key_start) + ".");
    }
    result.emplace(
        key,
        decode_attribute(attributes.substr(value_start, quote - value_start)));
    cursor = quote + 1;
  }
  return result;
}

OpenTag parse_open_tag(std::string_view text,
                       size_t cursor,
                       std::string_view tag) {
  const std::string_view remaining = text.substr(cursor);
  if (!remaining.starts_with(kOpenToken) ||
      !remaining.substr(kOpenToken.size()).starts_with(tag)) {
    throw std::runtime_error("Unexpected XTML tag.");
  }
  const size_t attributes_start = cursor + kOpenToken.size() + tag.size();
  const size_t separator = text.find(kSepToken, attributes_start);
  if (separator == std::string_view::npos) {
    throw std::runtime_error("Unclosed XTML start tag.");
  }

  OpenTag result;
  result.attributes = parse_attributes(
      text.substr(attributes_start, separator - attributes_start));
  result.body_start = separator + kSepToken.size();
  return result;
}

void require_attributes(
    const std::unordered_map<std::string, std::string>& attributes,
    std::initializer_list<std::string_view> required) {
  if (attributes.size() != required.size()) {
    throw std::runtime_error("Unexpected XTML attributes.");
  }
  for (const std::string_view key : required) {
    if (!attributes.contains(std::string(key))) {
      throw std::runtime_error("Missing required XTML attribute.");
    }
  }
}

// Keep the longest suffix that can still become an incomplete delimiter.
size_t partial_token_overlap(std::string_view text, std::string_view marker) {
  if (marker.empty()) {
    return 0;
  }
  const size_t maximum = std::min(text.size(), marker.size() - 1);
  for (size_t overlap = maximum; overlap > 0; --overlap) {
    if (text.substr(text.size() - overlap) == marker.substr(0, overlap)) {
      return overlap;
    }
  }
  return 0;
}

std::string safe_prefix(std::string_view text, std::string_view marker) {
  const size_t overlap = partial_token_overlap(text, marker);
  return std::string(text.substr(0, text.size() - overlap));
}

void skip_whitespace(std::string_view text, size_t* cursor) {
  while (*cursor < text.size() &&
         std::isspace(static_cast<unsigned char>(text[*cursor])) != 0) {
    ++*cursor;
  }
}

nlohmann::json parse_json_value(std::string_view value) {
  std::vector<std::unordered_set<std::string>> object_keys;
  bool has_duplicate_key = false;
  auto reject_duplicate_keys = [&object_keys, &has_duplicate_key](
                                   int /*depth*/,
                                   nlohmann::json::parse_event_t event,
                                   nlohmann::json& parsed_value) {
    if (event == nlohmann::json::parse_event_t::object_start) {
      object_keys.emplace_back();
    } else if (event == nlohmann::json::parse_event_t::key) {
      const std::string key = parsed_value.get<std::string>();
      if (!object_keys.back().insert(key).second) {
        has_duplicate_key = true;
      }
    } else if (event == nlohmann::json::parse_event_t::object_end) {
      object_keys.pop_back();
    }
    return true;
  };
  // allow_exceptions=false covers syntax errors, but conversion errors such
  // as numeric overflow can still throw. Normalize library errors here; the
  // public entrypoints reject the entire call batch before publishing calls.
  nlohmann::json parsed;
  try {
    parsed = nlohmann::json::parse(
        value.begin(), value.end(), reject_duplicate_keys, false);
  } catch (const nlohmann::json::exception& error) {
    throw std::runtime_error(
        std::string("Invalid JSON in K3 tool arguments: ") + error.what());
  }
  if (has_duplicate_key) {
    throw std::runtime_error("Duplicate JSON key in K3 arguments.");
  }
  if (parsed.is_discarded()) {
    throw std::runtime_error("Invalid JSON in K3 tool arguments.");
  }
  return parsed;
}

nlohmann::json parse_typed_argument(std::string_view value,
                                    const std::string& type) {
  if (type == "string") {
    return std::string(value);
  }

  nlohmann::json parsed = parse_json_value(value);
  const bool valid = (type == "number" && parsed.is_number()) ||
                     (type == "boolean" && parsed.is_boolean()) ||
                     (type == "null" && parsed.is_null()) ||
                     (type == "object" && parsed.is_object()) ||
                     (type == "array" && parsed.is_array());
  if (!valid) {
    throw std::runtime_error("K3 typed argument has an invalid value.");
  }
  return parsed;
}

std::string parse_call_arguments(std::string_view body) {
  size_t cursor = 0;
  skip_whitespace(body, &cursor);
  if (cursor == body.size()) {
    return "{}";
  }

  if (body.substr(cursor).starts_with(kJsonStart)) {
    const OpenTag json_tag = parse_open_tag(body, cursor, "json");
    require_attributes(json_tag.attributes, {"type"});
    if (json_tag.attributes.at("type") != "object") {
      throw std::runtime_error("K3 json arguments must have object type.");
    }
    const size_t json_end = body.find(kJsonEnd, json_tag.body_start);
    if (json_end == std::string_view::npos) {
      throw std::runtime_error("K3 json argument block is not closed.");
    }
    const nlohmann::json value = parse_json_value(
        body.substr(json_tag.body_start, json_end - json_tag.body_start));
    if (!value.is_object()) {
      throw std::runtime_error("K3 tool arguments must be a JSON object.");
    }
    size_t tail = json_end + kJsonEnd.size();
    skip_whitespace(body, &tail);
    if (tail != body.size()) {
      throw std::runtime_error("K3 call mixes argument encodings.");
    }
    return value.dump();
  }

  nlohmann::ordered_json arguments = nlohmann::ordered_json::object();
  while (cursor < body.size()) {
    const OpenTag argument_tag = parse_open_tag(body, cursor, "argument");
    require_attributes(argument_tag.attributes, {"key", "type"});
    const std::string& key = argument_tag.attributes.at("key");
    if (key.empty() || arguments.contains(key)) {
      throw std::runtime_error("K3 argument key is empty or duplicated.");
    }
    const size_t argument_end =
        body.find(kArgumentEnd, argument_tag.body_start);
    if (argument_end == std::string_view::npos) {
      throw std::runtime_error("K3 typed argument is not closed.");
    }
    arguments[key] = parse_typed_argument(
        body.substr(argument_tag.body_start,
                    argument_end - argument_tag.body_start),
        argument_tag.attributes.at("type"));
    cursor = argument_end + kArgumentEnd.size();
    skip_whitespace(body, &cursor);
  }
  return arguments.dump();
}

std::vector<ToolCallItem> parse_calls(
    std::string_view body,
    const std::unordered_set<std::string>& allowed_tool_names) {
  std::vector<ToolCallItem> calls;
  size_t cursor = 0;
  while (cursor < body.size()) {
    skip_whitespace(body, &cursor);
    if (cursor == body.size()) {
      break;
    }

    const OpenTag call_tag = parse_open_tag(body, cursor, "call");
    require_attributes(call_tag.attributes, {"tool", "index"});
    const std::string& name = call_tag.attributes.at("tool");
    const std::string& index_text = call_tag.attributes.at("index");
    if (name.empty()) {
      throw std::runtime_error("K3 call name must not be empty.");
    }
    // Indices use canonical positive decimal strings, without leading zeros
    // or signs. Even arbitrarily long values need no integer conversion.
    if (index_text != std::to_string(calls.size() + 1)) {
      throw std::runtime_error(
          "K3 call indices must be sequential and start at one.");
    }
    if (!allowed_tool_names.contains(name)) {
      throw std::runtime_error("K3 attempted to call an unknown tool.");
    }

    const size_t call_end = body.find(kCallEnd, call_tag.body_start);
    if (call_end == std::string_view::npos) {
      throw std::runtime_error("K3 call block is not closed.");
    }
    const std::string arguments = parse_call_arguments(
        body.substr(call_tag.body_start, call_end - call_tag.body_start));
    calls.emplace_back(static_cast<int32_t>(calls.size()), name, arguments);
    cursor = call_end + kCallEnd.size();
  }
  return calls;
}

void parse_response_tail(std::string_view text,
                         const std::vector<JsonTool>& tools,
                         ParseSnapshot& snapshot) {
  size_t cursor = 0;
  // Whitespace between blocks carries no response or argument content.
  skip_whitespace(text, &cursor);
  const std::string_view tail = text.substr(cursor);
  // Calls are published atomically after all arguments and the envelope have
  // validated. Executing a valid prefix of a malformed batch is unsafe.
  std::vector<ToolCallItem> calls;
  if (tail.starts_with(kToolsStart)) {
    const size_t tools_body_start = cursor + kToolsStart.size();
    const size_t tools_end = text.find(kToolsEnd, tools_body_start);
    if (tools_end == std::string_view::npos) {
      return;
    }
    std::unordered_set<std::string> allowed_tool_names;
    for (const JsonTool& tool : tools) {
      allowed_tool_names.insert(tool.function.name);
    }
    if (!tools.empty()) {
      calls = parse_calls(
          text.substr(tools_body_start, tools_end - tools_body_start),
          allowed_tool_names);
    }
    cursor = tools_end + kToolsEnd.size();
  } else if (kToolsStart.starts_with(tail)) {
    // No calls have been parsed on this branch; wait for the tools opener.
    return;
  }

  skip_whitespace(text, &cursor);
  const std::string_view message_tail(text.data() + cursor,
                                      text.size() - cursor);
  if (!message_tail.starts_with(kMessageEnd)) {
    if (kMessageEnd.starts_with(message_tail)) {
      return;
    }
    throw std::runtime_error("K3 output is missing its message end marker.");
  }
  cursor += kMessageEnd.size();
  // The required structural message close completes the protocol. The optional
  // end_of_msg stop token may be withheld by sequence output, absent or split
  // across chunks. Its prefix is ignored, not used to establish completion.
  std::string_view remainder = text.substr(cursor);
  if (remainder.starts_with(kEndOfMessageToken)) {
    remainder.remove_prefix(kEndOfMessageToken.size());
  } else if (kEndOfMessageToken.starts_with(remainder)) {
    remainder = {};
  }
  if (std::any_of(remainder.begin(), remainder.end(), [](char character) {
        return std::isspace(static_cast<unsigned char>(character)) == 0;
      })) {
    throw std::runtime_error("Unexpected content after the K3 message.");
  }
  if (tools.empty() && tail.starts_with(kToolsStart)) {
    // Warn only after validating the complete envelope, not once per chunk.
    LOG(WARNING) << "Ignoring Kimi K3 tools block: no tools were declared.";
  }
  snapshot.protocol_complete = true;
  snapshot.calls = std::move(calls);
}

void parse_snapshot(std::string_view text,
                    const std::vector<JsonTool>& tools,
                    ParseSnapshot& snapshot) {
  size_t cursor = 0;

  if (text.starts_with(kResponseStart)) {
    cursor += kResponseStart.size();
  } else if (kResponseStart.starts_with(text)) {
    return;
  }

  // The generation prompt already opens the response block when thinking is
  // disabled, so generated text legitimately starts with response content.
  const size_t response_end = text.find(kResponseEnd, cursor);
  if (response_end == std::string_view::npos) {
    const std::string_view response_body(text.data() + cursor,
                                         text.size() - cursor);
    // Length-limited generation may end inside a response. Return its safe
    // prefix, but never expose an incomplete protocol delimiter as content.
    snapshot.content = safe_prefix(response_body, kResponseEnd);
    return;
  }
  snapshot.content = text.substr(cursor, response_end - cursor);
  parse_response_tail(
      text.substr(response_end + kResponseEnd.size()), tools, snapshot);
}

}  // namespace

StreamingParseResult KimiK3Detector::detect_and_parse(
    const std::string& text,
    const std::vector<JsonTool>& tools) {
  ParseSnapshot snapshot;
  try {
    parse_snapshot(text, tools, snapshot);
  } catch (const std::exception& error) {
    LOG(ERROR) << "Invalid Kimi K3 XTML output: " << error.what();
  }
  return StreamingParseResult(std::move(snapshot.content),
                              std::move(snapshot.calls));
}

bool KimiK3Detector::has_tool_call(const std::string& text) {
  return text.find(kToolsStart) != std::string::npos;
}

StreamingParseResult KimiK3Detector::parse_streaming_increment(
    const std::string& new_text,
    const std::vector<JsonTool>& tools) {
  if (stream_state_ == StreamState::DONE) {
    return StreamingParseResult();
  }
  buffer_.append(new_text);
  if (stream_state_ == StreamState::RESPONSE_START) {
    if (kResponseStart.starts_with(buffer_)) {
      return StreamingParseResult();
    }
    if (buffer_.starts_with(kResponseStart)) {
      buffer_.erase(0, kResponseStart.size());
    }
    stream_state_ = StreamState::RESPONSE;
  }

  std::string content_delta;
  if (stream_state_ == StreamState::RESPONSE) {
    const size_t response_end = buffer_.find(kResponseEnd);
    if (response_end == std::string::npos) {
      // Release emitted text; only a possible split delimiter stays buffered.
      const size_t safe_size =
          buffer_.size() - partial_token_overlap(buffer_, kResponseEnd);
      content_delta = buffer_.substr(0, safe_size);
      buffer_.erase(0, safe_size);
      return StreamingParseResult(std::move(content_delta));
    }
    content_delta = buffer_.substr(0, response_end);
    buffer_.erase(0, response_end + kResponseEnd.size());
    // This is the only transition into TAIL. Its offset is still the initial
    // zero; subsequent TAIL calls append without erasing the searched bytes.
    stream_state_ = StreamState::TAIL;
  }

  // Search using the previous call's offset before computing the next one.
  // Tool arguments need complete blocks. Scan only newly appended bytes and
  // defer argument parsing until the message closes, rather than reparsing
  // all accumulated arguments on every generated token.
  const size_t message_end = buffer_.find(kMessageEnd, tail_search_offset_);
  if (message_end == std::string::npos) {
    // Only an unsuccessful search proves old bytes contain no complete marker.
    // Retain marker_length - 1 bytes to catch a close split across chunks.
    // A found marker must remain searchable if tail validation is incomplete.
    tail_search_offset_ = buffer_.size() > kMessageEnd.size() - 1
                              ? buffer_.size() - (kMessageEnd.size() - 1)
                              : 0;
    return StreamingParseResult(std::move(content_delta));
  }

  // A partial tools opener alone cannot reach this point: wait above for
  // the message close before validating the complete tail.
  ParseSnapshot snapshot;
  try {
    parse_response_tail(buffer_, tools, snapshot);
  } catch (const std::exception& error) {
    LOG(ERROR) << "Invalid streaming Kimi K3 XTML output: " << error.what();
    stream_state_ = StreamState::DONE;
    buffer_.clear();
    return StreamingParseResult(std::move(content_delta));
  }
  // Keep the search offset unchanged if validation is incomplete: a message
  // close can occur inside an unclosed tools block. Later chunks must retry
  // that candidate. Completion or an exception ends parsing, so neither path
  // needs to advance the offset.
  if (snapshot.protocol_complete) {
    stream_state_ = StreamState::DONE;
    buffer_.clear();
  }
  return StreamingParseResult(std::move(content_delta),
                              std::move(snapshot.calls));
}

}  // namespace function_call
}  // namespace xllm
