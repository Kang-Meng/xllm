#pragma once
#include <string>
#include <variant>

#include "audio_generation.pb.h"
#include "core/common/types.h"
#include "core/util/binary_payload.h"
#include "dit_request_state.h"
#include "image_generation.pb.h"
#include "text_generation.pb.h"
#include "video_generation.pb.h"

namespace xllm {

class DiTRequestParser;

using DiTInputProto = std::variant<std::monostate,
                                   const proto::Input*,
                                   const proto::AudioInput*,
                                   const proto::VideoInput*,
                                   const proto::TextInput*>;

struct DiTRawInputParams {
  // Non-owning pointer. Online callbacks retain the request until parsing ends.
  DiTInputProto input;
  BinaryPayload binary_payload;
};

class DiTRequestParams final {
 public:
  // Offline callers provide prepared input and generation parameters.
  explicit DiTRequestParams(
      DiTRequestKind request_kind = DiTRequestKind::kNone);

  DiTRequestParams(const proto::ImageGenerationRequest& request,
                   const std::string& x_rid,
                   const std::string& x_rtime,
                   BinaryPayload request_payload = {});
  DiTRequestParams(const proto::AudioGenerationRequest& request,
                   const std::string& x_rid,
                   const std::string& x_rtime,
                   BinaryPayload request_payload = {});
  DiTRequestParams(const proto::VideoGenerationRequest& request,
                   const std::string& x_rid,
                   const std::string& x_rtime,
                   BinaryPayload request_payload = {});
  DiTRequestParams(const proto::TextGenerationRequest& request,
                   const std::string& x_rid,
                   const std::string& x_rtime);

  DiTRequestKind request_kind() const { return request_kind_; }

  bool parse_input_params(Status& status);
  bool verify_params(Status& status) const;

  // request id
  std::string request_id;
  std::string x_request_id;
  std::string x_request_time;

  DiTRawInputParams raw_input_params;
  DiTInputParams input_params;

  DiTGenerationParams generation_params;

  Status request_parse_status;

 private:
  DiTRequestKind request_kind_ = DiTRequestKind::kNone;
  // Non-owning; points to an immutable parser with static lifetime.
  const DiTRequestParser* parser_ = nullptr;
};

}  // namespace xllm
