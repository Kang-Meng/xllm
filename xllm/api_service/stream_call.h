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

#pragma once

#include <brpc/controller.h>
#include <butil/iobuf.h>
#include <glog/logging.h>
#include <google/protobuf/message.h>
#include <json2pb/pb_to_json.h>

#include <memory>
#include <string>
#include <string_view>
#include <type_traits>

#include "anthropic.pb.h"
#include "api_service/anthropic_json.h"
#include "api_service/call.h"
#include "api_service/public_error.h"
#include "core/common/types.h"
#include "core/util/verbose_trace_logger.h"

namespace xllm {

template <typename Request, typename Response>
class StreamCall : public Call {
 public:
  using ReqType = Request;
  using ResType = Response;

  StreamCall(brpc::Controller* controller,
             ::google::protobuf::Closure* done,
             Request* request,
             Response* response,
             bool use_arena = false,
             bool is_http_request = false)
      : Call(controller, request_body_x_request_id(request), is_http_request),
        done_(done),
        request_(request),
        response_(response),
        use_arena_(use_arena) {
    stream_ = request_->stream();
    if (!stream_) {
      controller_->http_response().set_content_type(
          "application/json; charset=utf-8");
    }

    json_options_.bytes_to_base64 = false;
    json_options_.jsonify_empty_array = false;
  }

  ~StreamCall() override {
    record_result(stream_started_
                      ? connection_status_ == 0
                      : !controller_->Failed() && !controller_->IsCanceled());
    // A stream is opened only on the first successful event. A rejection before
    // that point is a regular HTTP JSON response and still needs its done Run.
    if (!stream_started_) {
      done_->Run();
    }
    if (!use_arena_) {
      delete request_;
      delete response_;
    }
  }

  bool write_and_finish(Response& response) {
    butil::IOBufAsZeroCopyOutputStream json_output(
        &controller_->response_attachment());
    std::string err_msg;
    if (!json2pb::ProtoMessageToJson(
            response, &json_output, json_options_, &err_msg)) {
      return finish_with_error(StatusCode::UNKNOWN, err_msg);
    }
    XLLM_VERBOSE_TRACE() << "event=response_serialized x-request-id="
                         << x_request_id_;
    response_complete_ = true;
    return true;
  }

  // For non-streaming plain-text responses.
  bool write_text_and_finish(std::string_view text) {
    controller_->http_response().set_content_type("text/plain; charset=utf-8");
    controller_->response_attachment().append(text.data(), text.size());
    response_complete_ = true;
    XLLM_VERBOSE_TRACE() << "event=response_serialized x-request-id="
                         << x_request_id_;
    return true;
  }

  bool finish_with_error(const StatusCode& code,
                         const std::string& error_message) {
    return finish_with_error(Status(code, error_message));
  }

  bool finish_with_error(const Status& status) {
    XLLM_VERBOSE_TRACE() << "event=request_error x-request-id=" << x_request_id_
                         << " message=" << status.message();
    if (stream_ && stream_finished_) {
      return false;
    }
    record_error();
    if (!is_http_request_) {
      controller_->SetFailed(status.message());
      return true;
    }

    constexpr bool kAnthropic =
        std::is_same_v<Request, proto::AnthropicMessagesRequest>;
    if (!stream_ || !stream_started_) {
      api_service::write_http_error(controller_, status, kAnthropic);
    } else {
      const api_service::PublicHttpError error =
          api_service::public_error_from_status(status);
      io_buf_.clear();
      if constexpr (kAnthropic) {
        io_buf_.append(api_service::anthropic_error_sse_event(error));
      } else {
        io_buf_.append(api_service::openai_error_sse_event(error));
      }
      connection_status_ |= pa_->Write(io_buf_);
    }
    stream_finished_ = stream_;
    return connection_status_ == 0;
  }

  // Writes one SSE event ("data: {json}\n\n"). Stream chunks may use a
  // different proto type from the non-streaming response. Returns false on
  // serialization failure or client disconnect.
  bool write(const google::protobuf::Message& message) {
    if (stream_finished_) {
      return false;
    }
    io_buf_.clear();
    io_buf_.append("data: ");
    butil::IOBufAsZeroCopyOutputStream json_output(&io_buf_);
    std::string err_msg;
    if (!json2pb::ProtoMessageToJson(
            message, &json_output, json_options_, &err_msg)) {
      LOG(ERROR) << "Failed to convert proto to json: " << err_msg;
      finish_with_error(StatusCode::UNKNOWN, err_msg);
      return false;
    }
    io_buf_.append("\n\n");

    start_stream();
    connection_status_ |= pa_->Write(io_buf_);
    return connection_status_ == 0;
  }

  // For stream response
  bool finish() {
    if (stream_finished_) {
      return connection_status_ == 0;
    }
    start_stream();
    stream_finished_ = true;
    io_buf_.clear();
    io_buf_.append("data: [DONE]\n\n");

    connection_status_ |= pa_->Write(io_buf_);
    XLLM_VERBOSE_TRACE() << "event=stream_closed x-request-id="
                         << x_request_id_;
    response_complete_ = connection_status_ == 0;
    return connection_status_ == 0;
  }

  bool is_disconnected() const override {
    if (stream_) {
      return connection_status_ != 0 ||
             (!stream_started_ && controller_->IsCanceled());
    } else {
      if (controller_) {
        return controller_->IsCanceled();
      }
      return true;
    }
  }

  const Request& request() const { return *request_; }
  Response& response() { return *response_; }
  ::google::protobuf::Closure* done() { return done_; }

 protected:
  void start_stream() {
    if (stream_started_) {
      return;
    }
    pa_ = controller_->CreateProgressiveAttachment();
    controller_->http_response().set_content_type(
        "text/event-stream; charset=utf-8");
    controller_->http_response().set_status_code(200);
    controller_->http_response().SetHeader("Connection", "keep-alive");
    controller_->http_response().SetHeader("Cache-Control", "no-cache");
    stream_started_ = true;
    done_->Run();
  }

  ::google::protobuf::Closure* done_;

  Request* request_ = nullptr;
  Response* response_ = nullptr;

  bool stream_ = false;
  bool stream_started_ = false;
  bool stream_finished_ = false;
  bool use_arena_ = false;
  butil::intrusive_ptr<brpc::ProgressiveAttachment> pa_;
  butil::IOBuf io_buf_;

  json2pb::Pb2JsonOptions json_options_;

  int connection_status_ = 0;
};

// Anthropic SSE stream call with custom event formatting
class AnthropicCall : public StreamCall<proto::AnthropicMessagesRequest,
                                        proto::AnthropicMessagesResponse> {
 public:
  AnthropicCall(brpc::Controller* controller,
                ::google::protobuf::Closure* done,
                proto::AnthropicMessagesRequest* request,
                proto::AnthropicMessagesResponse* response,
                bool use_arena = false,
                bool is_http_request = false)
      : StreamCall<proto::AnthropicMessagesRequest,
                   proto::AnthropicMessagesResponse>(controller,
                                                     done,
                                                     request,
                                                     response,
                                                     use_arena,
                                                     is_http_request) {
    // Anthropic responses require empty content arrays to remain visible.
    this->json_options_.jsonify_empty_array = true;
  }

  ~AnthropicCall() {}

  bool write_and_finish(proto::AnthropicMessagesResponse& response) {
    std::string json;
    std::string err_msg;
    if (!api_service::proto_to_anthropic_json(
            response, this->json_options_, &json, &err_msg)) {
      return this->finish_with_error(StatusCode::UNKNOWN, err_msg);
    }
    this->controller_->response_attachment().append(json);
    this->response_complete_ = true;
    XLLM_VERBOSE_TRACE() << "event=response_serialized x-request-id="
                         << this->x_request_id_;
    return true;
  }

  // Write SSE event with Anthropic format: event: <type>\ndata: <json>\n\n
  bool write(const std::string& event_type, const std::string& json_data) {
    if (this->stream_finished_) {
      return false;
    }
    this->io_buf_.clear();
    this->io_buf_.append("event: ");
    this->io_buf_.append(event_type);
    this->io_buf_.append("\ndata: ");
    this->io_buf_.append(json_data);
    this->io_buf_.append("\n\n");

    this->start_stream();
    this->connection_status_ |= this->pa_->Write(this->io_buf_);
    return this->connection_status_ == 0;
  }

  // Write SSE event with proto message
  template <typename ProtoMessage>
  bool write(const std::string& event_type, const ProtoMessage& message) {
    if (this->stream_finished_) {
      return false;
    }
    this->io_buf_.clear();
    this->io_buf_.append("event: ");
    this->io_buf_.append(event_type);
    this->io_buf_.append("\ndata: ");
    std::string json;
    std::string err_msg;
    if (!api_service::proto_to_anthropic_json(
            message, this->json_options_, &json, &err_msg)) {
      LOG(ERROR) << "Failed to convert proto to json: " << err_msg;
      this->finish_with_error(StatusCode::UNKNOWN, err_msg);
      return false;
    }
    this->io_buf_.append(json);
    this->io_buf_.append("\n\n");
    this->start_stream();
    this->connection_status_ |= this->pa_->Write(this->io_buf_);
    return this->connection_status_ == 0;
  }
};

template <typename T>
struct is_stream_call : std::false_type {};

template <typename... Args>
struct is_stream_call<StreamCall<Args...>> : std::true_type {};

template <typename T>
inline constexpr bool is_stream_call_v = is_stream_call<T>::value;

}  // namespace xllm
