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

#include "core/framework/request/request.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "api_service/call.h"
#include "core/common/metrics.h"
#include "core/framework/request/sequence.h"
#include "core/util/env_var.h"
#include "core/util/timer.h"

namespace xllm {
namespace {

// Write optional fields directly into the log stream without allocating a
// concatenated string. The label includes its separator.
class OptionalLogField final {
 public:
  OptionalLogField(std::string_view label, std::string_view value)
      : label_(label), value_(value) {}

  friend std::ostream& operator<<(std::ostream& stream,
                                  const OptionalLogField& field) {
    if (!field.value_.empty()) {
      stream << field.label_ << field.value_;
    }
    return stream;
  }

 private:
  std::string_view label_;
  std::string_view value_;
};

OptionalLogField optional_field(std::string_view label,
                                std::string_view value) {
  return OptionalLogField(label, value);
}

bool complete(const RequestTimingSample& sample) {
  // Constructed snapshots, including wire imports, can bypass span().
  // Revalidate complete samples at the recorder boundary.
  return sample.valid && !sample.partial && sample.milliseconds.has_value() &&
         std::isfinite(*sample.milliseconds) && *sample.milliseconds >= 0 &&
         *sample.milliseconds * 1000.0 <
             static_cast<double>(std::numeric_limits<int64_t>::max()) &&
         sample.reason == RequestTimingReason::COMPLETE;
}

const char* skipped_sample_reason(const RequestTimingSample& sample) {
  return sample.reason == RequestTimingReason::MISSING_START ? "missing_start"
         : sample.reason == RequestTimingReason::MISSING_END ? "missing_end"
         : sample.reason == RequestTimingReason::UNKNOWN_BOUNDARY
             ? "unknown_boundary"
         : sample.reason == RequestTimingReason::INTERRUPTED ? "interrupted"
                                                             : "invalid_sample";
}

const char* segment_exclusion(const RequestTimingSample& sample,
                              const char* exclusion) {
  return exclusion != nullptr || complete(sample)
             ? exclusion
             : skipped_sample_reason(sample);
}

void observe(bvar::LatencyRecorder& recorder,
             const RequestTimingSample& sample,
             const char* phase = nullptr,
             const char* exclusion = nullptr) {
  if (exclusion == nullptr && complete(sample)) {
    recorder << static_cast<int64_t>(*sample.milliseconds * 1000.0);
  } else if (phase != nullptr) {
    const char* reason =
        exclusion != nullptr ? exclusion : skipped_sample_reason(sample);
    bvar::Adder<int64_t>* counter =
        MULTI_COUNTER_xllm_request_phase_skipped_total.get_stats(
            {phase, reason});
    if (counter != nullptr) {
      *counter << 1;
    }
  }
}

void observe(bvar::LatencyRecorder* recorder,
             const RequestTimingSample& sample,
             const char* phase,
             const char* exclusion) {
  if (recorder != nullptr) {
    observe(*recorder, sample, phase, exclusion);
  }
}

}  // namespace

void observe_request_ttft_metric(const RequestTimingSnapshot& snapshot,
                                 bool single_output) {
  if (single_output && !snapshot.pd_local) {
    observe(HISTOGRAM_xllm_request_ttft_microseconds, snapshot.server_ttft);
  }
}

void observe_request_phase_metrics(const RequestTimingSnapshot& snapshot,
                                   bool single_output,
                                   uint64_t num_generation_tokens,
                                   const char* exclusion) {
  if (!snapshot.terminal) {
    return;
  }
  if (exclusion == nullptr && !single_output) {
    exclusion = "unsupported_output";
  }
  if (snapshot.pd_local) {
    // An original P request has no local Decode segment to sample or skip.
    if (!snapshot.pd_imported) {
      return;
    }
    const RequestTimingSample tpot =
        request_pd_tpot_sample(snapshot, num_generation_tokens);
    if (exclusion == nullptr) {
      if (!snapshot.pd_decode_tokens_at_start.has_value()) {
        exclusion = "missing_start";
      } else if (num_generation_tokens <= *snapshot.pd_decode_tokens_at_start) {
        exclusion = "no_local_tokens";
      }
    }
    // No local output means no Decode segment. Validate the raw span before
    // TPOT division so overflow cannot be hidden by the token denominator.
    const char* decode_exclusion =
        segment_exclusion(snapshot.pd_decode_time, exclusion);
    observe(MULTI_HISTOGRAM_xllm_pd_request_queue_microseconds.get_stats(
                {"decode"}),
            snapshot.pd_queued_time,
            "pd_decode_queue",
            decode_exclusion);
    observe(MULTI_HISTOGRAM_xllm_pd_request_decode_microseconds.get_stats(
                {"decode"}),
            snapshot.pd_decode_time,
            "pd_decode",
            decode_exclusion);
    observe(
        MULTI_HISTOGRAM_xllm_pd_request_tpot_microseconds.get_stats({"decode"}),
        tpot,
        "pd_tpot",
        decode_exclusion);
    return;
  }
  observe(HISTOGRAM_xllm_request_queue_microseconds,
          snapshot.queued_time,
          "queue",
          exclusion);
  observe(HISTOGRAM_xllm_request_prefill_microseconds,
          snapshot.prefill_time,
          "prefill",
          exclusion);
  if (exclusion == nullptr && num_generation_tokens == 1) {
    exclusion = "single_token";
  }
  observe(HISTOGRAM_xllm_request_decode_microseconds,
          snapshot.decode_time,
          "decode",
          exclusion);
  observe(HISTOGRAM_xllm_request_tpot_microseconds,
          request_tpot_sample(snapshot, num_generation_tokens),
          "tpot",
          exclusion);
}

void observe_request_pd_prefill_metrics(const RequestTimingSnapshot& snapshot,
                                        bool single_output,
                                        const char* exclusion) {
  if (!snapshot.pd_local || snapshot.pd_imported) {
    return;
  }
  if (exclusion == nullptr && !single_output) {
    exclusion = "unsupported_output";
  }
  // Prefill completes before generation terminal. Publish at the first
  // confirmed core result; later handoff success is a different boundary.
  observe(
      MULTI_HISTOGRAM_xllm_pd_request_queue_microseconds.get_stats({"prefill"}),
      snapshot.queued_time,
      "pd_prefill_queue",
      segment_exclusion(snapshot.prefill_time, exclusion));
  observe(MULTI_HISTOGRAM_xllm_pd_request_prefill_microseconds.get_stats(
              {"prefill"}),
          snapshot.prefill_time,
          "pd_prefill",
          exclusion);
}

Request::Request(const std::string& request_id,
                 const std::string& x_request_id,
                 const std::string& x_request_time,
                 RequestState state,
                 const std::string& service_request_id,
                 const std::string& source_xservice_addr,
                 RateLimiter* rate_limiter)
    : RequestBase(request_id,
                  x_request_id,
                  x_request_time,
                  service_request_id,
                  source_xservice_addr,
                  rate_limiter),
      state_(std::move(state)) {
  if (!state_.timing) {
    state_.timing = std::make_shared<RequestTiming>();
  }
  create_sequences_group();
}

Request::~Request() {
  if (!VLOG_IS_ON(1)) {
    return;
  }
  const RequestTokenCounts counts = request_tokens_.snapshot();
  if (counts.enabled) {
    VLOG(1) << "record_type: request_token_summary, request_id: " << request_id_
            << ", service_request_id: " << service_request_id_
            << ", input_tokens: " << counts.input_tokens
            << ", output_tokens: " << counts.output_tokens;
  }
}

void Request::create_sequences_group() {
  SequenceParams sequence_params;
  sequence_params.seq_capacity = state_.seq_capacity;
  sequence_params.skip_special_tokens = state_.skip_special_tokens;
  sequence_params.include_stop_str_in_output =
      state_.include_stop_str_in_output;
  sequence_params.echo = state_.echo;
  sequence_params.logprobs = state_.logprobs;
  sequence_params.n = state_.n;
  sequence_params.best_of = state_.best_of;
  sequence_params.streaming = state_.stream;
  sequence_params.enable_schedule_overlap = state_.enable_schedule_overlap;
  sequence_params.is_graph_warmup = state_.is_graph_warmup;
  sequence_params.rec_type = state_.rec_type;
  sequence_params.bos_token_id = state_.bos_token_id;
  sequence_params.request_id = request_id_;
  sequence_params.sample_slots = &(state_.sample_slots);
  sequence_params.sampling_param = &(state_.sampling_param);
  sequence_params.stopping_checker = &(state_.stopping_checker);
  sequence_params.json_object_grammar = state_.json_object_grammar;
  sequence_params.json_reasoning_enabled = state_.json_reasoning_enabled;
  sequence_params.request_failure_state = failure_state_;
  sequence_params.speculative_token_stats = speculative_token_stats_;
  sequence_params.request_timing = state_.timing;
  sequences_group_ =
      std::make_unique<SequencesGroup>(state_.prompt,
                                       state_.prompt_tokens,
                                       state_.input_embedding,
                                       state_.mm_data,
                                       std::move(sequence_params));
}

bool Request::finished() const {
  return error_status().has_value() || sequences_group_->finished();
}

std::optional<Status> Request::error_status() const {
  return failure_state_->status;
}

bool Request::expand_sequences(bool share_prefix) {
  return sequences_group_->expand_sequences(share_prefix);
}

void Request::reset() {
  // These blocks never entered a forward, so they must not be published.
  // Sequence::reset drops the aliases without the pool's prefix-cache flush.
  for (const auto& sequence : sequences()) {
    if (sequence != nullptr) {
      sequence->reset();
    }
  }
}

void Request::record_token_admission() {
  if (offline() || state_.is_graph_warmup) {
    return;
  }
  const uint64_t delta = request_tokens_.admit(state_.prompt_tokens.size());
  COUNTER_ADD(request_input_tokens_total, delta);
  if (delta > 0) {
    VLOG(1) << "record_type: request_tokens, request_id: " << request_id_
            << ", event: admission, input_delta: " << delta
            << ", output_delta: 0";
  }
}

void Request::enable_forwarded_token_output() {
  request_tokens_.forward(state_.stream);
}

void Request::record_token_output(const RequestOutput& output, bool submitted) {
  if (!request_tokens_.enabled()) {
    return;
  }
  if (output.cancelled || (output.status.has_value() && !output.status->ok()) ||
      (!state_.stream && output.finished_on_prefill_instance)) {
    return;
  }
  const bool per_sequence_submission = output.token_submission_tracked;
  if (!per_sequence_submission) {
    // Custom/batch callbacks acknowledge the whole output via their bool.
    // Successfully submitting an error response is not a token submission.
    if (!submitted || (state_.call_.has_value() &&
                       state_.call_.value()->has_response_error())) {
      return;
    }
  }
  // HTTP streaming acknowledges each sequence separately. A later write may
  // return false without invalidating earlier confirmed deltas.
  token_progress_buf_.clear();
  token_progress_buf_.reserve(output.outputs.size());
  for (const SequenceOutput& sequence : output.outputs) {
    if (per_sequence_submission && !sequence.token_submission_confirmed) {
      continue;
    }
    token_progress_buf_.emplace_back(
        OutputTokenProgress{sequence.index, sequence.generated_token_count});
  }
  const uint64_t delta = request_tokens_.commit(token_progress_buf_);
  COUNTER_ADD(request_output_tokens_total, delta);
  if (delta > 0) {
    VLOG(1) << "record_type: request_tokens, request_id: " << request_id_
            << ", event: submission, input_delta: 0, output_delta: " << delta;
  }
}

// Nested timing locks always follow timing_metrics_mutex_ -> RequestTiming's
// mutex_. Timing methods release their lock before any callback into Request.
void Request::update_timing_after_execution(bool generation_terminal) {
  if (state_.is_graph_warmup) {
    return;
  }
  const bool terminal = generation_terminal && finished();
  const bool prefill =
      !generation_terminal && state_.timing->has_generated_token() &&
      !pd_prefill_metrics_observed_.load(std::memory_order_acquire);
  if (!terminal && !prefill && !cancelled() && !error_status().has_value()) {
    return;
  }
  // Recheck under the same lock used to publish cancellation.
  std::lock_guard<std::mutex> lock(timing_metrics_mutex_);
  if (cancelled() || error_status().has_value()) {
    state_.timing->update_from_finished_request(/*generation_complete=*/false);
    observe_terminal_timing_locked(cancelled() ? "cancelled" : "failed");
    return;
  }
  if (terminal) {
    state_.timing->update_from_finished_request(/*generation_complete=*/true);
    observe_terminal_timing_locked(/*exclusion=*/nullptr);
  } else if (!pd_prefill_metrics_observed_.load(std::memory_order_relaxed)) {
    observe_request_pd_prefill_metrics(state_.timing->snapshot(),
                                       supports_single_output_timing());
    pd_prefill_metrics_observed_.store(true, std::memory_order_release);
  }
}

void Request::observe_terminal_timing_locked(const char* exclusion) {
  if (phase_metrics_observed_ || state_.is_graph_warmup) {
    return;
  }
  const RequestTimingSnapshot snapshot = state_.timing->snapshot();
  const bool single_output =
      supports_single_output_timing() && sequences().size() == 1;
  if (snapshot.pd_local && !snapshot.pd_imported) {
    if (!pd_prefill_metrics_observed_.exchange(true,
                                               std::memory_order_acq_rel)) {
      observe_request_pd_prefill_metrics(snapshot, single_output, exclusion);
    }
  } else {
    observe_request_phase_metrics(
        snapshot,
        single_output,
        single_output ? sequences().front()->num_committed_generated_tokens()
                      : 0,
        exclusion);
  }
  phase_metrics_observed_ = true;
}

void Request::record_timing_failure(const Status& status) {
  std::lock_guard<std::mutex> lock(timing_metrics_mutex_);
  state_.timing->update_from_finished_request(/*generation_complete=*/false);
  observe_terminal_timing_locked(
      status.code() == StatusCode::CANCELLED ? "cancelled" : "failed");
}

bool Request::supports_single_output_timing() const {
  return state_.n == 1 && state_.best_of == 1 && !check_beam_search() &&
         state_.sample_slots.empty() && !state_.sampling_param.is_embeddings &&
         !state_.is_graph_warmup;
}

bool Request::record_server_first_token(RequestTiming::TimePoint observed_at) {
  if (cancelled() || error_status().has_value() ||
      !state_.sample_slots.empty() || state_.sampling_param.is_embeddings ||
      state_.is_graph_warmup || state_.timing->has_observed_token() ||
      !state_.timing->has_generated_token()) {
    return false;
  }
  std::lock_guard<std::mutex> lock(timing_metrics_mutex_);
  // The lock-free checks are hints; recheck before publishing a sample.
  if (cancelled() || error_status().has_value() ||
      !state_.timing->has_generated_token() ||
      state_.timing->has_observed_token()) {
    return false;
  }
  state_.timing->record_first_token_observed(observed_at);
  const RequestTimingSnapshot snapshot = state_.timing->snapshot(observed_at);
  // Unlike the single-output vLLM comparison cohort below, this metric is
  // once per Request, including n/best_of. Never publish an unfinished wait
  // as TTFT. PD is complete only on the original ingress after handoff ACK.
  if (snapshot.server_ttft.valid && !snapshot.server_ttft.partial &&
      snapshot.server_ttft.milliseconds.has_value()) {
    HISTOGRAM_OBSERVE(server_time_to_first_token_latency_milliseconds,
                      static_cast<int64_t>(*snapshot.server_ttft.milliseconds));
  }
  observe_request_ttft_metric(snapshot, supports_single_output_timing());
  return snapshot.server_ttft.valid;
}

void Request::set_upstream_latency_seconds(double upstream_latency_seconds) {
  const double latency_offset = upstream_latency_seconds - elapsed_seconds();
  end_to_end_latency_offset_seconds_ =
      std::max(end_to_end_latency_offset_seconds_, latency_offset);
}

double Request::end_to_end_latency_seconds() const {
  return elapsed_seconds() + end_to_end_latency_offset_seconds_;
}

RequestTerminalOutcome Request::claim_terminal_outcome(
    const std::optional<Status>& status) {
  RequestTerminalOutcome outcome = RequestTerminalOutcome::NONE;
  if (status.has_value() && !status->ok()) {
    outcome = status->code() == StatusCode::CANCELLED
                  ? RequestTerminalOutcome::CANCELLED
                  : RequestTerminalOutcome::FAILURE;
  } else if (cancelled()) {
    outcome = RequestTerminalOutcome::CANCELLED;
  } else if (finished()) {
    outcome = RequestTerminalOutcome::SUCCESS;
  }
  if (outcome == RequestTerminalOutcome::NONE) {
    return outcome;
  }
  RequestTerminalOutcome expected = RequestTerminalOutcome::NONE;
  if (!terminal_outcome_.compare_exchange_strong(
          expected, outcome, std::memory_order_relaxed)) {
    return RequestTerminalOutcome::NONE;
  }
  if (outcome != RequestTerminalOutcome::SUCCESS) {
    record_timing_failure(outcome == RequestTerminalOutcome::CANCELLED
                              ? Status(StatusCode::CANCELLED)
                              : status.value());
  }
  return outcome;
}

void Request::log_terminal_statistic(RequestTerminalOutcome outcome,
                                     const Status& status,
                                     double total_latency) {
  // Client-side cancellation can leave the output status OK. Normalize the
  // terminal log without changing the status sent to the caller.
  const Status logged_status =
      (outcome == RequestTerminalOutcome::CANCELLED && status.ok())
          ? Status(StatusCode::CANCELLED, "Cancelled")
          : status;
  const char* outcome_name =
      outcome == RequestTerminalOutcome::SUCCESS     ? "success"
      : outcome == RequestTerminalOutcome::CANCELLED ? "cancelled"
                                                     : "failure";
  LOG(INFO) << "record_type: request_terminal, request_id: " << request_id_
            << optional_field(", x-request-id: ", x_request_id_)
            << ", outcome: " << outcome_name
            << ", status_code: " << static_cast<int32_t>(logged_status.code())
            << optional_field(", x-request-time: ", x_request_time_);
  if (outcome == RequestTerminalOutcome::SUCCESS) {
    log_statistic(total_latency, /*prefill_stage=*/false);
  } else {
    log_error_statistic(logged_status, outcome_name);
  }
}

void Request::log_prefill_statistic(double total_latency) {
  if (prefill_statistic_logged_.exchange(true, std::memory_order_relaxed)) {
    return;
  }
  log_statistic(total_latency, /*prefill_stage=*/true);
}

void Request::record_first_batched_time() {
  if (first_batched_time_ == absl::InfinitePast()) {
    first_batched_time_ = absl::Now();
  }
}

double Request::queue_to_batch_latency_ms() const {
  if (first_batched_time_ == absl::InfinitePast()) {
    return -1.0;
  }
  return absl::ToDoubleMilliseconds(first_batched_time_ - created_time_);
}

void Request::log_statistic(double total_latency, bool prefill_stage) {
  // Keep the client ID on every record for grep; request time belongs to the
  // request terminal or the non-terminal PD prefill stage, not each sequence.
  const std::string_view request_time =
      prefill_stage ? std::string_view(x_request_time_) : std::string_view();
  int32_t idx = 0;
  for (const auto& seq : sequences()) {
    const double ttft = seq->time_to_first_token_latency_seconds();
    const size_t gen_tokens = seq->num_committed_generated_tokens();
    double tpot = 0.0;
    double gen_speed = 0.0;
    // Preserve these legacy response-inclusive statistics. The request-level
    // xllm_request_tpot_microseconds uses the core first-to-last token span.
    if (gen_tokens > 1 && total_latency > ttft && ttft > 0) {
      const double generation_latency = total_latency - ttft;
      tpot = (generation_latency * 1000.0) / (gen_tokens - 1);
      gen_speed = gen_tokens / generation_latency;
    }
    std::string queue_to_batch_log;
    {
      std::ostringstream stream;
      stream << std::fixed << std::setprecision(1)
             << "queue_to_batch: " << queue_to_batch_latency_ms() << "ms, ";
      queue_to_batch_log = stream.str();
    }
    std::string speculative_stats_log;
    if (speculative_token_stats_->proposed_tokens > 0) {
      const double acceptance_rate =
          static_cast<double>(speculative_token_stats_->accepted_tokens) /
          static_cast<double>(speculative_token_stats_->proposed_tokens);
      std::ostringstream stream;
      stream << ", speculative_accepted_tokens: "
             << speculative_token_stats_->accepted_tokens
             << ", speculative_proposed_tokens: "
             << speculative_token_stats_->proposed_tokens << std::fixed
             << std::setprecision(4)
             << ", speculative_token_acceptance_rate: " << acceptance_rate;
      speculative_stats_log = stream.str();
    }
    std::string store_prefetch_log;
    if (util::store_prefetch_stats_enabled()) {
      std::ostringstream stream;
      stream << ", store_prefetch_tokens: " << store_prefetch_tokens_
             << std::fixed << std::setprecision(1)
             << ", store_prefetch_latency: " << store_prefetch_latency_ms_
             << "ms";
      store_prefetch_log = stream.str();
    }
    const std::string finish_reason =
        seq->finish_reason().to_string().value_or("");
    LOG(INFO) << "record_type: "
              << (prefill_stage ? "pd_prefill_stage" : "sequence_terminal")
              << ", request_id: " << request_id_
              << optional_field(", x-request-id: ", x_request_id_)
              << ", sequence: " << idx++
              << optional_field(", finish_reason: ", finish_reason)
              << ", prompt_tokens: " << seq->num_prompt_tokens() << ", "
              << "generated_tokens: " << gen_tokens << ", "
              << queue_to_batch_log << std::fixed << std::setprecision(1)
              << "ttft: " << ttft * 1000 << "ms, "
              << "total_latency: " << total_latency * 1000 << "ms, "
              << "avg tpot: " << tpot << "ms, "
              << "generation speed: " << gen_speed << " tokens/s"
              << ", max_tokens: "
              << seq->stopping_checker()->get_max_generated_tokens()
              << ", temperature: " << std::defaultfloat << std::setprecision(6)
              << seq->sampling_param()->temperature
              << ", num_prefix_cache_tokens: " << num_prefix_cache_tokens_
              << speculative_stats_log << store_prefetch_log
              << optional_field(", x-request-time: ", request_time);
    // only log once when beam search is enabled
    if (check_beam_search()) {
      break;
    }
  }
}

void Request::log_error_statistic(const Status& status,
                                  const char* outcome_name) {
  // log the request statistics
  int32_t idx = 0;
  for (const auto& seq : sequences()) {
    LOG(INFO) << "record_type: sequence_terminal, request_id: " << request_id_
              << optional_field(", x-request-id: ", x_request_id_)
              << ", sequence: " << idx++ << ", "
              << "outcome: " << outcome_name << ", "
              << "status_code: " << static_cast<int32_t>(status.code())
              << optional_field(", status_msg: ", status.message())
              << ", prompt_tokens: " << seq->num_prompt_tokens() << ", "
              << "max_tokens: "
              << seq->stopping_checker()->get_max_generated_tokens() << ", "
              << "temperature: " << seq->sampling_param()->temperature;
  }
}

size_t Request::total_num_blocks() {
  size_t num = 0;
  for (auto& seq : sequences()) {
    num += seq->kv_state().num_blocks(BlockType::KV);
  }
  return num;
}

void Request::set_cancel() {
  {
    // Serialize cancellation publication with successful metric sampling.
    std::lock_guard<std::mutex> lock(timing_metrics_mutex_);
    state_.timing->update_from_finished_request(/*generation_complete=*/false);
    cancelled_.store(true, std::memory_order_relaxed);
    observe_terminal_timing_locked("cancelled");
  }
  for (const auto& seq : sequences()) {
    seq->set_cancel();
  }
}

void Request::handle_last_token() {
  state_.handle_last_token_done = true;
  for (const auto& seq : sequences()) {
    seq->handle_last_token();
  }
}

RequestOutput Request::generate_output(const Tokenizer& tokenizer,
                                       ThreadPool* thread_pool) {
  // summarize statistics for all sequences
  Usage usage;
  usage.num_prompt_tokens = state_.prompt_tokens.size();
  for (const auto& seq : sequences()) {
    usage.num_generated_tokens +=
        static_cast<int32_t>(seq->num_committed_generated_tokens());
  }
  CHECK_LE(num_prefix_cache_tokens_,
           static_cast<size_t>(std::numeric_limits<int32_t>::max()));
  usage.num_cached_tokens = static_cast<int32_t>(num_prefix_cache_tokens_);
  usage.num_total_tokens = usage.num_prompt_tokens + usage.num_generated_tokens;

  RequestOutput output;
  output.request_id = request_id_;
  output.service_request_id = service_request_id_;
  output.target_xservice_addr = source_xservice_addr_;
  output.usage = usage;
  output.finished = finished();
  output.cancelled = cancelled();
  const std::optional<Status> request_error = error_status();
  if (request_error.has_value()) {
    output.status = request_error.value();
    return output;
  }
  if (output.cancelled) {
    output.status = Status(StatusCode::CANCELLED);
    return output;
  }
  output.status = Status(StatusCode::OK);
  sequences_group_->generate_outputs(output.outputs, tokenizer, thread_pool);
  return output;
}

void Request::record_num_prefix_cache_tokens() {
  size_t current_max = 0;
  for (const auto& seq : sequences()) {
    current_max = std::max(current_max, seq->num_prefix_cache_tokens());
  }
  record_num_prefix_cache_tokens(current_max);
  if (prefix_cache_hit_metrics_recorded_) {
    return;
  }

  record_prefix_cache_hit_metrics(state_.prompt_tokens.size(),
                                  num_prefix_cache_tokens_);
  prefix_cache_hit_metrics_recorded_ = true;
}

void Request::record_num_prefix_cache_tokens(size_t num_prefix_cache_tokens) {
  num_prefix_cache_tokens_ =
      std::max(num_prefix_cache_tokens_, num_prefix_cache_tokens);
}

void Request::update_connection_status() {
  if (!state_.call_.has_value()) {
    return;
  }
  Call* call = state_.call_.value();
  CHECK(call != nullptr);
  bool is_disconnected = call->is_disconnected();
  if (!is_disconnected) {
    return;
  }
  set_cancel();
}

}  // namespace xllm
