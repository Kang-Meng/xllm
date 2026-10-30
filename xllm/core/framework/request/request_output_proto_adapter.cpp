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

#include "request_output_proto_adapter.h"

#include <cstdint>
#include <limits>

#include "core/common/public_error.h"

namespace xllm {

bool request_output_to_proto(const RequestOutput& output,
                             const std::string& sender_engine_uid,
                             proto::DisaggStreamGeneration* generation) {
  if (generation == nullptr || sender_engine_uid.empty() ||
      output.outputs.size() >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
    return false;
  }

  generation->Clear();
  generation->set_req_id(output.request_id);
  generation->set_service_req_id(output.service_request_id);
  if (output.status.has_value()) {
    proto::GenerationStatus* status = generation->mutable_gen_status();
    status->set_status_code(static_cast<int32_t>(output.status->code()));
    status->set_status_msg(output.status->message());
    if (output.status->reason() != FailureReason::UNSPECIFIED) {
      status->set_failure_reason(static_cast<int32_t>(output.status->reason()));
    }
    if (output.status->param().has_value()) {
      status->set_param(*output.status->param());
    }
    if (!output.status->ok()) {
      const PublicHttpError error = public_error_from_status(*output.status);
      proto::PublicError* public_error = status->mutable_error();
      public_error->set_http_status_code(error.status_code);
      public_error->set_type(error.type);
      public_error->set_code(error.code);
      public_error->set_message(error.message);
      public_error->set_retryable(error.retryable);
      if (error.param.has_value()) {
        public_error->set_param(*error.param);
      }
    }
  }
  generation->set_finished(output.finished);
  generation->set_finished_on_prefill_instance(
      output.finished_on_prefill_instance);
  generation->set_sender_engine_uid(sender_engine_uid);
  if (output.usage.has_value()) {
    const Usage& usage = *output.usage;
    if (usage.num_prompt_tokens < 0 || usage.num_generated_tokens < 0 ||
        usage.num_total_tokens < 0 || usage.num_cached_tokens < 0) {
      return false;
    }
    proto::OutputUsage* proto_usage = generation->mutable_usage();
    proto_usage->set_num_prompt_tokens(usage.num_prompt_tokens);
    proto_usage->set_num_generated_tokens(usage.num_generated_tokens);
    proto_usage->set_num_total_tokens(usage.num_total_tokens);
    proto_usage->set_num_cached_tokens(usage.num_cached_tokens);
  }

  generation->mutable_outputs()->Reserve(
      static_cast<int>(output.outputs.size()));
  for (const SequenceOutput& sequence_output : output.outputs) {
    if (sequence_output.index >
            static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        sequence_output.token_ids.size() >
            static_cast<size_t>(std::numeric_limits<int>::max())) {
      return false;
    }
    proto::SequenceOutput* proto_output = generation->add_outputs();
    proto_output->set_index(static_cast<int32_t>(sequence_output.index));
    proto_output->set_text(sequence_output.text);
    proto_output->set_finish_reason(sequence_output.finish_reason.value_or(""));
    proto_output->mutable_token_ids()->Reserve(
        static_cast<int>(sequence_output.token_ids.size()));
    for (int32_t token_id : sequence_output.token_ids) {
      proto_output->add_token_ids(token_id);
    }
    if (!sequence_output.logprobs.has_value()) {
      continue;
    }
    if (sequence_output.logprobs->size() >
        static_cast<size_t>(std::numeric_limits<int>::max())) {
      return false;
    }
    proto_output->mutable_logprobs()->Reserve(
        static_cast<int>(sequence_output.logprobs->size()));
    for (const LogProb& logprob : *sequence_output.logprobs) {
      proto::LogProb* proto_logprob = proto_output->add_logprobs();
      proto::LogProbData* data = proto_logprob->mutable_log_prob_data();
      data->set_token(logprob.token);
      data->set_token_id(logprob.token_id);
      data->set_logprob(logprob.logprob);
      data->set_finished_token(logprob.finished_token);
      if (!logprob.top_logprobs.has_value()) {
        continue;
      }
      if (logprob.top_logprobs->size() >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
        return false;
      }
      proto_logprob->mutable_top_logprobs()->Reserve(
          static_cast<int>(logprob.top_logprobs->size()));
      for (const LogProbData& top_logprob : *logprob.top_logprobs) {
        proto::LogProbData* proto_top = proto_logprob->add_top_logprobs();
        proto_top->set_token(top_logprob.token);
        proto_top->set_token_id(top_logprob.token_id);
        proto_top->set_logprob(top_logprob.logprob);
        proto_top->set_finished_token(top_logprob.finished_token);
      }
    }
  }
  return true;
}

}  // namespace xllm
