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

#include "core/scheduler/async_response_processor.h"

#include <absl/synchronization/notification.h>
#include <absl/time/clock.h>
#include <glog/logging.h>

#include <algorithm>
#include <memory>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "core/framework/config/service_config.h"
#include "framework/request/finish_reason.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "util/blocking_counter.h"
#include "util/env_var.h"

namespace xllm {

AsyncResponseProcessor::AsyncResponseProcessor(
    const Tokenizer* tokenizer,
    const std::optional<InstanceRole>& role,
    bool enable_service_routing,
    bool disable_log_stats,
    std::function<void(std::shared_ptr<Request>)> cancel_request)
    : response_threadpool_(
          /*num_threads=*/::xllm::ServiceConfig::get_instance()
              .num_response_handling_threads(),
          /*cpu_binding=*/false,
          /*pool_name=*/"AsyncResponseProcessor.response"),
      rpc_threadpool_(/*num_threads=*/1,
                      /*cpu_binding=*/false,
                      /*pool_name=*/"AsyncResponseProcessor.rpc"),
      generate_output_threadpool_(
          /*num_threads=*/16,
          /*cpu_binding=*/false,
          /*pool_name=*/"AsyncResponseProcessor.generate_output"),
      tokenizer_(tokenizer->clone()),
      role_(role.value_or(InstanceRole::DEFAULT)),
      enable_batch_response_(enable_service_routing),
      disable_log_stats_(disable_log_stats),
      cancel_request_(std::move(cancel_request)) {}

AsyncResponseProcessor::~AsyncResponseProcessor() { wait_completion(); }

bool AsyncResponseProcessor::claim_and_log_terminal(
    Request& request,
    const std::optional<Status>& status,
    double total_latency) {
  const RequestTerminalOutcome outcome = request.claim_terminal_outcome(status);
  if (outcome == RequestTerminalOutcome::NONE) {
    return false;
  }
  if (!disable_log_stats_) {
    request.log_terminal_statistic(outcome, status.value(), total_latency);
  }
  return true;
}

bool AsyncResponseProcessor::prepare_completed_output(
    Request& request,
    bool terminal,
    ThreadPool* generation_threadpool,
    RequestOutput* output) {
  if (terminal && request.has_terminal_outcome()) {
    return false;
  }
  const double end_2_end_latency_seconds = request.end_to_end_latency_seconds();
  *output = request.generate_output(*tokenizer_, generation_threadpool);
  if (terminal && !claim_and_log_terminal(
                      request, output->status, end_2_end_latency_seconds)) {
    return false;
  }
  if (!terminal && output->status.has_value() && output->status->ok()) {
    output->finished_on_prefill_instance = true;
  }
  HISTOGRAM_OBSERVE(end_2_end_latency_milliseconds,
                    static_cast<int64_t>(end_2_end_latency_seconds * 1000.0));
  return true;
}

void AsyncResponseProcessor::process_failed_request(
    std::shared_ptr<Request> request,
    Status status,
    bool send_response) {
  // schedule the response handling
  auto runnable = [this, request, status = std::move(status), send_response]() {
    if (!claim_and_log_terminal(
            *request, status, request->end_to_end_latency_seconds())) {
      return;
    }
    if (!send_response) {
      return;
    }
    RequestOutput output;
    output.request_id = request->request_id();
    output.service_request_id = request->service_request_id();
    output.target_xservice_addr = request->source_xservice_addr();
    output.status = status;
    request->state().output_func(output);
  };
  if (request->state().response_thread_id < 0) {
    request->state().response_thread_id =
        response_threadpool_.schedule(runnable);
  } else {
    response_threadpool_.schedule_with_tid(runnable,
                                           request->state().response_thread_id);
  }
}

void AsyncResponseProcessor::process_completed_request(
    std::shared_ptr<Request> request,
    bool terminal) {
  // schedule the response handling

  // NOTE: Maybe refactor later.
  // For prefill instance in 'ENABLE_SERVICE_DISPATCH_REQUEST' scenario,
  // Currently, the xLLM service uses the BRPC HTTP interface to forward
  // requests. When the prefill process returns the first token to the xLLM
  // service, the result is only returned after the OutputFunc callback function
  // is destructed. Therefore, to ensure the TTFT, the Output callback function
  // needs to be recycled once its execution is complete, allowing the first
  // token to be returned immediately. Additionally, the processing here runs in
  // parallel with the subsequent step where the prefill sends the first token
  // to the decode instance.
  //  When this completes, it destructs the request, which in turn causes the
  //  Output object
  // to be destructed. To prevent the scenario where the request is recycled
  // before the response is fully returned to the xLLM service, the Output
  // object must be detached to avoid premature destruction.
  auto runnable = [this, request, terminal]() mutable {
    AUTO_COUNTER(responsing_latency_seconds_non_stream);

    RequestOutput req_output;
    if (!prepare_completed_output(
            *request, terminal, &generate_output_threadpool_, &req_output)) {
      return;
    }
    const bool submitted = request->state().output_func(req_output);
    request->record_token_output(req_output, submitted);
  };
  if (request->state().response_thread_id < 0) {
    request->state().response_thread_id =
        response_threadpool_.schedule(runnable);
  } else {
    response_threadpool_.schedule_with_tid(runnable,
                                           request->state().response_thread_id);
  }
}

void AsyncResponseProcessor::batch_process_completed_requests(
    std::vector<std::shared_ptr<Request>>& requests,
    bool terminal) {
  size_t requests_size = requests.size();
  auto counter = new BlockingCounter(requests_size);
  std::vector<RequestOutput> request_outputs;
  request_outputs.resize(requests_size);
  std::vector<uint8_t> valid_outputs(requests_size, 0);
  for (size_t i = 0; i < requests_size; ++i) {
    auto& request = requests[i];
    auto runnable = [counter,
                     this,
                     request = request,
                     terminal,
                     valid_output = &valid_outputs[i],
                     request_output = &request_outputs[i]]() mutable {
      AUTO_COUNTER(responsing_latency_seconds_non_stream);
      if (prepare_completed_output(*request,
                                   terminal,
                                   /*generation_threadpool=*/nullptr,
                                   request_output)) {
        *valid_output = 1;
      }
      counter->decrement_count();
    };
    if (request->state().response_thread_id < 0) {
      request->state().response_thread_id =
          response_threadpool_.schedule(runnable);
    } else {
      response_threadpool_.schedule_with_tid(
          runnable, request->state().response_thread_id);
    }
  }

  rpc_threadpool_.schedule(
      [counter = std::unique_ptr<BlockingCounter>(counter),
       requests = std::move(requests),
       request_outputs = std::move(request_outputs),
       valid_outputs = std::move(valid_outputs)]() mutable {
        counter->wait();
        auto& resp_callback = requests[0]->state().outputs_func;
        // Common case: no request was skipped, so forward in place and avoid
        // re-moving every RequestOutput into a filtered vector.
        if (std::all_of(valid_outputs.begin(),
                        valid_outputs.end(),
                        [](uint8_t valid) { return valid != 0; })) {
          const std::vector<bool> submitted = resp_callback(request_outputs);
          CHECK_EQ(submitted.size(), request_outputs.size());
          for (size_t i = 0; i < requests.size(); ++i) {
            requests[i]->record_token_output(request_outputs[i], submitted[i]);
          }
          return;
        }
        std::vector<RequestOutput> outputs;
        outputs.reserve(request_outputs.size());
        std::vector<size_t> request_indexes;
        request_indexes.reserve(request_outputs.size());
        for (size_t i = 0; i < request_outputs.size(); ++i) {
          if (valid_outputs[i] != 0) {
            outputs.emplace_back(std::move(request_outputs[i]));
            request_indexes.emplace_back(i);
          }
        }
        if (outputs.empty()) {
          return;
        }
        const std::vector<bool> submitted = resp_callback(outputs);
        CHECK_EQ(submitted.size(), outputs.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
          requests[request_indexes[i]]->record_token_output(outputs[i],
                                                            submitted[i]);
        }
      });
}

// process non-stream requests
void AsyncResponseProcessor::process_completed_requests(
    std::vector<std::shared_ptr<Request>>& requests,
    bool terminal) {
  std::vector<std::shared_ptr<Request>> service_requests;
  service_requests.reserve(enable_batch_response_ ? requests.size() : 0);
  for (std::shared_ptr<Request>& request : requests) {
    if (!enable_batch_response_ || request->service_request_id().empty()) {
      process_completed_request(std::move(request), terminal);
    } else {
      service_requests.emplace_back(std::move(request));
    }
  }
  if (!service_requests.empty()) {
    batch_process_completed_requests(service_requests, terminal);
  }
}

void AsyncResponseProcessor::process_stream_request(
    std::shared_ptr<Request> request) {
  CHECK(request->state().stream) << "request is not a streaming request";

  std::vector<size_t> indexes;
  std::vector<size_t> num_tokens;
  bool is_all_seqs_closed = true;
  for (size_t i = 0; i < request->sequences().size(); ++i) {
    auto& seq = request->sequences()[i];
    is_all_seqs_closed &= seq->is_closed();
    if (seq->is_closed()) {
      // skip already closed sequences
      continue;
    }

    // check if the sequence has enough tokens to output
    if (seq->has_new_tokens_generated() || seq->finished()) {
      indexes.push_back(i);
      num_tokens.push_back(seq->num_tokens());
    }

    // close the sequence after sending finish reason
    if (seq->finished()) {
      seq->close();
    }
  }

  if (!is_all_seqs_closed) {
    // output the delta text til the end of the sequence to the client

    auto runnable = [cancel_request = cancel_request_,
                     request,
                     this,
                     indexes = std::move(indexes),
                     num_tokens = std::move(num_tokens)]() {
      AUTO_COUNTER(responsing_latency_seconds_stream);

      RequestOutput req_output;
      req_output.request_id = request->request_id();
      for (size_t i = 0; i < indexes.size(); ++i) {
        const size_t index = indexes[i];
        const size_t size = num_tokens[i];
        auto& seq = request->sequences()[index];
        auto seq_output = seq->generate_streaming_output(size, *tokenizer_);
        if (seq_output.has_value()) {
          req_output.outputs.push_back(std::move(seq_output.value()));
        }
      }
      const bool submitted = request->state().output_func(req_output);
      request->record_token_output(req_output, submitted);
      if (!submitted) {
        cancel_request(request);
      }
    };
    if (request->state().response_thread_id < 0) {
      request->state().response_thread_id =
          response_threadpool_.schedule(runnable);
    } else {
      response_threadpool_.schedule_with_tid(
          runnable, request->state().response_thread_id);
    }
  }
}

void AsyncResponseProcessor::batch_process_stream_requests(
    std::vector<std::shared_ptr<Request>>& requests) {
  size_t requests_size = requests.size();
  auto counter = new BlockingCounter(requests_size);
  std::vector<RequestOutput> request_outputs;
  request_outputs.resize(requests_size);
  for (int i = 0; i < requests_size; ++i) {
    auto& request = requests[i];
    CHECK(request->state().stream) << "request is not a streaming request";

    std::vector<size_t> indexes;
    std::vector<size_t> num_tokens;
    for (size_t i = 0; i < request->sequences().size(); ++i) {
      auto& seq = request->sequences()[i];
      if (seq->is_closed()) {
        // skip already closed sequences
        continue;
      }

      // check if the sequence has enough tokens to output
      if (seq->has_new_tokens_generated() || seq->finished()) {
        indexes.push_back(i);
        num_tokens.push_back(seq->num_tokens());
      }

      // close the sequence after sending finish reason
      if (seq->finished()) {
        seq->close();
      }
    }

    // output the delta text til the end of the sequence to the client
    auto runnable = [this,
                     counter,
                     request,
                     indexes = std::move(indexes),
                     num_tokens = std::move(num_tokens),
                     req_output = &request_outputs[i]]() mutable {
      AUTO_COUNTER(responsing_latency_seconds_stream);
      const absl::Time response_start_time = absl::Now();

      // RequestOutput req_output;
      req_output->request_id = request->request_id();
      req_output->service_request_id = request->service_request_id();
      req_output->target_xservice_addr = request->source_xservice_addr();
      for (size_t i = 0; i < indexes.size(); ++i) {
        const size_t index = indexes[i];
        const size_t size = num_tokens[i];
        auto& seq = request->sequences()[index];

        auto seq_output = seq->generate_streaming_output(size, *tokenizer_);
        if (seq_output.has_value()) {
          req_output->outputs.push_back(std::move(seq_output.value()));
        }
        if (seq->num_generated_tokens() == 1) {
          // currently only support one sequence when enable_service_routing
          // IMPROVE LATER: support enable_schedule_overlap in Default mode
          // for stream request
          req_output->finished_on_prefill_instance = true;
        }
      }
      if (req_output->finished_on_prefill_instance) {
        VLOG(1) << "Prefill response generation request_id="
                << request->request_id() << ", response_thread_id="
                << request->state().response_thread_id << ", total_ms="
                << absl::ToDoubleMilliseconds(absl::Now() -
                                              response_start_time);
      }
      counter->decrement_count();
    };
    if (request->state().response_thread_id < 0) {
      request->state().response_thread_id =
          response_threadpool_.schedule(runnable);
    } else {
      response_threadpool_.schedule_with_tid(
          runnable, request->state().response_thread_id);
    }
  }

  rpc_threadpool_.schedule([cancel_request = cancel_request_,
                            counter = std::unique_ptr<BlockingCounter>(counter),
                            requests = std::move(requests),
                            request_outputs =
                                std::move(request_outputs)]() mutable {
    auto& resp_callback = requests[0]->state().outputs_func;
    const absl::Time wait_start_time = absl::Now();
    counter->wait();
    const double wait_ms =
        absl::ToDoubleMilliseconds(absl::Now() - wait_start_time);
    const absl::Time rpc_start_time = absl::Now();
    std::vector<bool> status_set = resp_callback(request_outputs);
    CHECK_EQ(status_set.size(), request_outputs.size());
    if (!request_outputs.empty() &&
        request_outputs[0].finished_on_prefill_instance) {
      VLOG(1) << "Prefill response RPC request_id=" << requests[0]->request_id()
              << ", response_wait_ms=" << wait_ms << ", rpc_ms="
              << absl::ToDoubleMilliseconds(absl::Now() - rpc_start_time);
    }
    for (size_t i = 0; i < requests.size(); ++i) {
      requests[i]->record_token_output(request_outputs[i], status_set[i]);
      if (!status_set[i]) {
        cancel_request(requests[i]);
      }
    }
  });
}

// process stream requests
void AsyncResponseProcessor::process_stream_requests(
    std::vector<std::shared_ptr<Request>>& requests) {
  std::vector<std::shared_ptr<Request>> service_requests;
  service_requests.reserve(enable_batch_response_ ? requests.size() : 0);
  for (std::shared_ptr<Request>& request : requests) {
    if (!enable_batch_response_ || request->service_request_id().empty()) {
      process_stream_request(std::move(request));
    } else {
      service_requests.emplace_back(std::move(request));
    }
  }
  if (!service_requests.empty()) {
    batch_process_stream_requests(service_requests);
  }
}

// for batch generate, wait all response done.
void AsyncResponseProcessor::wait_completion() {
  auto wait_threadpool = [](ThreadPool& threadpool) {
    size_t thread_num = threadpool.size();
    // Add a task to each thread, and when all tasks are completed, it indicates
    // that all previously scheduled tasks in the thread pool have finished.
    BlockingCounter counter(thread_num);
    for (size_t i = 0; i < thread_num; ++i) {
      auto runnable = [&counter]() mutable { counter.decrement_count(); };
      threadpool.schedule_with_tid(std::move(runnable), i);
    }
    counter.wait();
  };
  wait_threadpool(response_threadpool_);
  wait_threadpool(rpc_threadpool_);
}

}  // namespace xllm
