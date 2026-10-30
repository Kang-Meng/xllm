/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "scheduler/async_response_processor.h"

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "framework/request/request.h"
#include "framework/request/request_output_proto_adapter.h"
#include "framework/tokenizer/tokenizer.h"

namespace xllm {
namespace {

using namespace std::chrono_literals;

class FakeTokenizer final : public Tokenizer {
 public:
  std::string decode(const Slice<int32_t>& ids,
                     bool /*skip_special_tokens*/) const override {
    return std::string(ids.size(), 'x');
  }

  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeTokenizer>();
  }
};

std::shared_ptr<Request> make_failed_request(const std::string& request_id,
                                             OutputsFunc outputs_func) {
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(1);
  stopping_checker.set_max_context_len(16);

  RequestState state("prompt",
                     /*prompt_tokens=*/{1},
                     sampling_param,
                     scheduler_param,
                     stopping_checker,
                     /*seq_capacity=*/16,
                     /*n=*/1,
                     /*best_of=*/1,
                     /*logprobs=*/false,
                     /*stream=*/false,
                     /*echo=*/false,
                     /*skip_special_tokens=*/false,
                     /*enable_schedule_overlap=*/false,
                     /*output_func=*/nullptr,
                     std::move(outputs_func));
  return std::make_shared<Request>(request_id,
                                   request_id,
                                   "request-time",
                                   std::move(state),
                                   "service-" + request_id,
                                   "service-address");
}

TEST(AsyncResponseProcessorTest, WaitCompletionIsFencedAndIsolatedByRequest) {
  std::mutex mutex;
  std::condition_variable cv;
  bool first_started = false;
  bool second_started = false;
  bool release_first = false;
  bool release_second = false;

  auto blocking_delivery = [&](bool* started, bool* release) {
    return [&, started, release](const std::vector<RequestOutput>& outputs) {
      EXPECT_EQ(outputs.size(), 1u);
      std::unique_lock<std::mutex> lock(mutex);
      *started = true;
      cv.notify_all();
      cv.wait(lock, [&]() { return *release; });
      return std::vector<OutputDeliveryStatus>{OutputDeliveryStatus::DELIVERED};
    };
  };

  auto first = make_failed_request(
      "first", blocking_delivery(&first_started, &release_first));
  auto second = make_failed_request(
      "second", blocking_delivery(&second_started, &release_second));
  FakeTokenizer tokenizer;
  AsyncResponseProcessor processor(
      &tokenizer,
      InstanceRole::PREFILL,
      /*enable_service_routing=*/true,
      /*disable_log_stats=*/true,
      [](std::shared_ptr<Request>) { ADD_FAILURE() << "unexpected abort"; });

  processor.process_failed_request(
      first, Status(StatusCode::UNAVAILABLE, "first failed"));
  {
    std::unique_lock<std::mutex> lock(mutex);
    EXPECT_TRUE(cv.wait_for(lock, 1s, [&]() { return first_started; }));
  }

  processor.process_failed_request(
      second, Status(StatusCode::UNAVAILABLE, "second failed"));
  std::future<void> second_waiter = std::async(
      std::launch::async, [&]() { processor.wait_completion(second); });
  {
    std::unique_lock<std::mutex> lock(mutex);
    EXPECT_TRUE(cv.wait_for(lock, 1s, [&]() { return second_started; }));
  }
  EXPECT_EQ(second_waiter.wait_for(20ms), std::future_status::timeout);

  {
    std::lock_guard<std::mutex> lock(mutex);
    release_second = true;
  }
  cv.notify_all();
  EXPECT_EQ(second_waiter.wait_for(1s), std::future_status::ready);
  second_waiter.get();
  {
    std::lock_guard<std::mutex> lock(mutex);
    EXPECT_TRUE(first_started);
    EXPECT_FALSE(release_first);
    release_first = true;
  }
  cv.notify_all();
  processor.wait_completion();
}

// The supplied token replaces model execution; Request stopping, asynchronous
// output production and protobuf encoding remain the production path.
void check_remote_output(InstanceRole role,
                         bool stream,
                         bool eos,
                         bool overlap,
                         bool completed_path,
                         bool prepared_first = false) {
  std::mutex mutex;
  std::vector<RequestOutput> observed;
  OutputsFunc deliver = [&](const std::vector<RequestOutput>& outputs) {
    std::lock_guard<std::mutex> lock(mutex);
    observed.insert(observed.end(), outputs.begin(), outputs.end());
    return std::vector<OutputDeliveryStatus>(outputs.size(),
                                             OutputDeliveryStatus::DELIVERED);
  };
  StoppingChecker stopping;
  stopping.set_max_generated_tokens(eos ? 64 : 1);
  stopping.set_max_context_len(128);
  stopping.set_eos_token(9);
  RequestState state("prompt",
                     {1, 2, 3},
                     RequestSamplingParam{},
                     SchedulerParam{},
                     stopping,
                     /*seq_capacity=*/128,
                     /*n=*/1,
                     /*best_of=*/1,
                     /*logprobs=*/false,
                     stream,
                     /*echo=*/false,
                     /*skip_special_tokens=*/false,
                     overlap,
                     /*output_func=*/nullptr,
                     deliver);
  state.execution_mode = proto::EXECUTION_MODE_REMOTE_PD;
  auto request = std::make_shared<Request>("engine-request",
                                           "review",
                                           "time",
                                           std::move(state),
                                           "remote-output",
                                           "router");
  auto& sequence = request->sequences().front();
  sequence->kv_state().set_kv_cache_tokens_num(3);
  if (role == InstanceRole::DECODE && stream) {
    sequence->enable_checking_prefill_token();
  }
  const Token token(eos ? 9 : 8);
  if (overlap) {
    sequence->append_token(-1);
    sequence->update_last_step_token(token);
  } else {
    sequence->append_token(token);
  }
  ASSERT_TRUE(request->finished());
  FakeTokenizer tokenizer;
  AsyncResponseProcessor processor(
      &tokenizer,
      role,
      /*enable_service_routing=*/true,
      /*disable_log_stats=*/true,
      [](std::shared_ptr<Request>) { ADD_FAILURE() << "unexpected abort"; });
  std::vector<std::shared_ptr<Request>> requests{request};
  if (prepared_first) {
    auto output = processor.prepare_first_output(request);
    ASSERT_TRUE(output.has_value());
    processor.process_prepared_output(request, std::move(*output));
  } else if (completed_path) {
    processor.process_completed_requests(requests);
  } else {
    processor.process_stream_requests(requests);
  }
  processor.wait_completion(request);
  std::lock_guard<std::mutex> lock(mutex);
  ASSERT_EQ(observed.size(), 1u);
  proto::DisaggStreamGeneration encoded;
  ASSERT_TRUE(request_output_to_proto(observed.front(), "engine", &encoded));
  EXPECT_EQ(encoded.finished_on_prefill_instance(),
            role == InstanceRole::PREFILL);
  EXPECT_EQ(encoded.service_req_id(), "remote-output");
  EXPECT_EQ(proto::DisaggStreamGeneration::descriptor()->FindFieldByName(
                "output_event_seq"),
            nullptr);
  if (prepared_first) {
    // P hands off a remote request; only D owns the request terminal.
    EXPECT_FALSE(encoded.finished());
  }
}

TEST(AsyncResponseProcessorTest, DecodeSingleTokenRetainsDecodeIdentity) {
  for (const bool stream : {false, true}) {
    for (const bool eos : {false, true}) {
      for (const bool overlap : {false, true}) {
        SCOPED_TRACE(::testing::Message() << stream << eos << overlap);
        check_remote_output(InstanceRole::DECODE,
                            stream,
                            eos,
                            overlap,
                            /*completed_path=*/true);
        if (stream) {
          check_remote_output(InstanceRole::DECODE,
                              stream,
                              eos,
                              overlap,
                              /*completed_path=*/false);
        }
      }
    }
  }
}

TEST(AsyncResponseProcessorTest, PrefillHandoffNeverTerminatesRemoteRequest) {
  for (const bool stream : {false, true}) {
    for (const bool eos : {false, true}) {
      check_remote_output(InstanceRole::PREFILL,
                          stream,
                          eos,
                          /*overlap=*/false,
                          /*completed_path=*/true,
                          /*prepared_first=*/true);
    }
  }
}

}  // namespace
}  // namespace xllm
