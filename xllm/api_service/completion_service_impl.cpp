/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "api_service/completion_service_impl.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "api_service/request_admission.h"
#include "api_service/utils.h"
#include "completion.pb.h"
#include "core/common/instance_name.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/framework/request/request_output.h"
#include "core/framework/request/request_timing.h"
#include "core/util/utils.h"

#ifdef likely
#undef likely
#endif
#define likely(x) __builtin_expect(!!(x), 1)

#ifdef unlikely
#undef unlikely
#endif
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace xllm {
namespace {
void set_logprobs(proto::Choice* choice,
                  const std::optional<std::vector<LogProb>>& logprobs,
                  const std::vector<int32_t>& query_ids = {}) {
  if (!logprobs.has_value() || logprobs.value().empty()) {
    return;
  }

  auto* proto_logprobs = choice->mutable_logprobs();
  for (const auto& logprob : logprobs.value()) {
    proto_logprobs->add_tokens(logprob.token);
    proto_logprobs->add_token_ids(logprob.token_id);
    proto_logprobs->add_token_logprobs(logprob.logprob);
  }
  if (!query_ids.empty()) {
    const auto& queried = *logprobs->front().top_logprobs;
    for (const auto& entry : queried) {
      proto_logprobs->add_queried_token_ids(entry.token_id);
      proto_logprobs->add_queried_token_logprobs(entry.logprob);
    }
  }
}

bool send_delta_to_client_brpc(std::shared_ptr<CompletionCall> call,
                               bool include_usage,
                               const std::string& request_id,
                               int64_t created_time,
                               const std::string& model,
                               const RequestOutput& output) {
  auto& response = call->response();
  output.token_submission_tracked = true;

  for (const auto& seq_output : output.outputs) {
    auto submission = call->token_submission();
    if (!seq_output.text.empty()) {
      response.Clear();
      response.set_object("text_completion");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(seq_output.index);
      choice->set_text(seq_output.text);
      set_logprobs(choice, seq_output.logprobs);
      if (!call->write(response)) {
        return false;
      }
    }

    if (!submission.commit()) {
      return false;
    }
    seq_output.token_submission_confirmed = true;

    if (seq_output.finish_reason.has_value()) {
      response.Clear();
      response.set_object("text_completion");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(seq_output.index);
      choice->set_text("");
      choice->set_finish_reason(seq_output.finish_reason.value());
      if (!call->write(response)) {
        return false;
      }
    }
  }

  if (include_usage && output.usage.has_value()) {
    const auto& usage = output.usage.value();
    response.Clear();
    response.set_object("text_completion");
    response.set_id(request_id);
    response.set_created(created_time);
    response.set_model(model);
    response.mutable_choices();
    auto* proto_usage = response.mutable_usage();
    api_service::set_proto_usage(proto_usage, usage);
    if (!call->write(response)) {
      return false;
    }
  }

  if (output.finished || output.cancelled) {
    response.Clear();
    return call->finish();
  }
  return true;
}

bool send_result_to_client_brpc(std::shared_ptr<CompletionCall> call,
                                const std::string& request_id,
                                int64_t created_time,
                                const std::string& model,
                                const RequestOutput& req_output,
                                const std::vector<int32_t>& query_ids) {
  auto& response = call->response();
  response.set_object("text_completion");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);

  response.mutable_choices()->Reserve(req_output.outputs.size());
  for (const auto& output : req_output.outputs) {
    if (!query_ids.empty()) {
      if (!output.logprobs.has_value() || output.logprobs->size() != 1 ||
          !output.logprobs->front().top_logprobs.has_value() ||
          output.logprobs->front().top_logprobs->size() != query_ids.size()) {
        return call->finish_with_error(
            Status(StatusCode::UNKNOWN, "Missing requested token logprobs"));
      }
      const auto& queried = *output.logprobs->front().top_logprobs;
      for (size_t i = 0; i < query_ids.size(); ++i) {
        if (queried[i].token_id != query_ids[i]) {
          return call->finish_with_error(Status(
              StatusCode::UNKNOWN, "Requested token logprobs out of order"));
        }
      }
    }
    auto* choice = response.add_choices();
    choice->set_index(output.index);
    choice->set_text(output.text);
    set_logprobs(choice, output.logprobs, query_ids);
    if (output.finish_reason.has_value()) {
      choice->set_finish_reason(output.finish_reason.value());
    }
  }

  if (req_output.usage.has_value()) {
    const auto& usage = req_output.usage.value();
    auto* proto_usage = response.mutable_usage();
    api_service::set_proto_usage(proto_usage, usage);
  }

  return call->write_and_finish(response);
}

}  // namespace

CompletionServiceImpl::CompletionServiceImpl(
    LLMMaster* master,
    const std::vector<std::string>& models)
    : APIServiceImpl(models), master_(master) {
  CHECK(master_ != nullptr);
  add_model_master(models[0], master);
}

void CompletionServiceImpl::add_model_master(const std::string& model,
                                             LLMMaster* master) {
  CHECK(master != nullptr);
  std::unique_lock<std::shared_mutex> lock(llm_model_to_master_mutex_);
  llm_model_to_master_.insert_or_assign(model, master);
  models_.insert(model);
}

LLMMaster* CompletionServiceImpl::get_model_master(
    const std::string& model) const {
  std::shared_lock<std::shared_mutex> lock(llm_model_to_master_mutex_);
  auto it = llm_model_to_master_.find(model);
  if (it == llm_model_to_master_.end()) {
    return nullptr;
  }
  return it->second;
}

// complete_async for brpc from xllm_service
void CompletionServiceImpl::process_async_rpc_impl(
    const proto::CompletionRequest* request) {
  const RequestTiming::TimePoint arrival_time = RequestTiming::Clock::now();
  const auto& service_request_id = request->service_request_id();
  const auto& target_xservice_addr = request->source_xservice_addr();
  auto callback = [master = master_](const RequestOutput& req_output) -> bool {
    req_output.log_request_status();
    return master->handle_rpc_response(req_output);
  };

  const auto& rpc_request = *request;
  const auto& model = rpc_request.model();
  Status admission_status = api_service_internal::admit_rpc_request(
      model, models_, master_->get_rate_limiter());
  if (unlikely(!admission_status.ok())) {
    callback({std::move(admission_status),
              service_request_id,
              target_xservice_addr});
    return;
  }

  RequestParams request_params(rpc_request, "", "");
  request_params.timing = std::make_shared<RequestTiming>();
  request_params.timing->record_arrival(arrival_time);

  std::optional<std::vector<int>> prompt_tokens = std::nullopt;
  if (rpc_request.token_ids_size() > 0) {
    prompt_tokens = std::vector<int>(rpc_request.token_ids().begin(),
                                     rpc_request.token_ids().end());
  }
  if (rpc_request.has_routing()) {
    request_params.decode_address = rpc_request.routing().decode_name();
  }

  // schedule the request
  master_->handle_request(std::move(rpc_request.prompt()),
                          std::move(prompt_tokens),
                          std::move(request_params),
                          std::nullopt,
                          callback);
}

// complete_async for brpc
void CompletionServiceImpl::process_async_impl(
    std::shared_ptr<CompletionCall> call) {
  // Route input preparation starts here, after transport/JSON parsing.
  call->request_timing()->record_arrival();
  const auto& rpc_request = call->request();
  const auto& model = rpc_request.model();
  LLMMaster* master = get_model_master(model);
  if (unlikely(master == nullptr)) {
    call->finish_with_error(api_service_internal::model_not_supported_status());
    return;
  }

  // Preserve the rejection cause: sleep and configured concurrency are not
  // interchangeable from the caller's perspective.
  if (unlikely(master->get_rate_limiter()->is_limited())) {
    call->finish_with_error(api_service_internal::rate_limiter_rejection_status(
        *master->get_rate_limiter()));
    return;
  }

  RequestParams request_params(
      rpc_request, call->get_x_request_id(), call->get_x_request_time());
  request_params.timing = call->request_timing();
  bool include_usage = false;
  if (rpc_request.has_stream_options()) {
    include_usage = rpc_request.stream_options().include_usage();
  }

  std::optional<std::vector<int>> prompt_tokens = std::nullopt;
  if (rpc_request.token_ids_size() > 0) {
    prompt_tokens = std::vector<int>(rpc_request.token_ids().begin(),
                                     rpc_request.token_ids().end());
  }
  if (rpc_request.has_routing()) {
    request_params.decode_address = rpc_request.routing().decode_name();
  }

  auto saved_streaming = request_params.streaming;
  auto saved_request_id = request_params.request_id;
  std::vector<int32_t> query_ids(rpc_request.logprob_token_ids().begin(),
                                 rpc_request.logprob_token_ids().end());
  // schedule the request
  master->handle_request(
      std::move(rpc_request.prompt()),
      std::move(prompt_tokens),
      std::move(request_params),
      call.get(),
      [call,
       model,
       master = master,
       stream = std::move(saved_streaming),
       include_usage = include_usage,
       query_ids = std::move(query_ids),
       request_id = std::move(saved_request_id),
       created_time = absl::ToUnixSeconds(absl::Now())](
          const RequestOutput& req_output) -> bool {
        req_output.log_request_status();
        if (req_output.status.has_value()) {
          const auto& status = req_output.status.value();
          if (!status.ok()) {
            return call->finish_with_error(status);
          }
        }

        if (stream) {
          return send_delta_to_client_brpc(
              call, include_usage, request_id, created_time, model, req_output);
        }
        // NOTE: maybe need to refactor along with service, currently for
        // non-stream request in prefill instance, we send a virtual response
        return send_result_to_client_brpc(
            call, request_id, created_time, model, req_output, query_ids);
      });
}

}  // namespace xllm
