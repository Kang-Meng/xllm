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

#include "core/scheduler/disagg_pd_scheduler.h"

#include <brpc/closure_guard.h>
#include <brpc/server.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/common/public_error.h"
#include "distributed_runtime/engine.h"
#include "framework/batch/batch.h"
#include "framework/block/block_manager_impl.h"
#include "framework/block/block_manager_pool.h"
#include "framework/config/rec_config.h"
#include "framework/config/scheduler_config.h"
#include "framework/kv_cache_transfer/kv_transfer_completion.h"
#include "framework/model/model_args.h"
#include "framework/request/request.h"
#include "framework/request/request_state.h"
#include "framework/tokenizer/tokenizer.h"
#include "runtime/xservice_client.h"
#include "scheduler/async_response_processor.h"
#include "scheduler/pd_ooc_scheduler.h"
#include "util/scope_guard.h"

namespace xllm {
namespace {

// allocate_decode reads the process-global chunk size. Scheduler Options do
// not publish it, so a test that declares a chunk must install the same value
// here and restore the previous one when it returns.
class ScopedPrefillChunkSize final {
 public:
  explicit ScopedPrefillChunkSize(int32_t chunk_size)
      : previous_(SchedulerConfig::get_instance()
                      .max_tokens_per_chunk_for_prefill()) {
    SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
        chunk_size;
  }

  ~ScopedPrefillChunkSize() {
    SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
        previous_;
  }

  ScopedPrefillChunkSize(const ScopedPrefillChunkSize&) = delete;
  ScopedPrefillChunkSize& operator=(const ScopedPrefillChunkSize&) = delete;

 private:
  int32_t previous_;
};

class FakeTokenizer final : public Tokenizer {
 public:
  bool encode(const std::string_view& /*text*/,
              std::vector<int32_t>* /*ids*/,
              bool /*add_special_tokens*/) const override {
    NOT_IMPLEMENTED();
  }

  std::string decode(const Slice<int32_t>& ids,
                     bool /*skip_special_tokens*/) const override {
    std::string text;
    for (int32_t id : ids) {
      text += std::to_string(id);
    }
    return text;
  }

  std::optional<int32_t> token_to_id(
      const std::string_view& /*token*/) const override {
    NOT_IMPLEMENTED();
  }

  std::string id_to_token(int32_t /*id*/) const override { NOT_IMPLEMENTED(); }

  size_t vocab_size() const override { NOT_IMPLEMENTED(); }

  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeTokenizer>();
  }
};

class FakeEngine final : public Engine {
 public:
  FakeEngine(int32_t num_blocks,
             int32_t block_size,
             int32_t num_speculative_tokens = 0,
             int32_t dp_size = 1,
             int32_t embedding_blocks = 0,
             bool enable_linear_state = false) {
    BlockManagerPool::Options options;
    options.num_blocks(num_blocks)
        .block_size(block_size)
        .enable_prefix_cache(true)
        .enable_disagg_pd(true)
        .num_speculative_tokens(num_speculative_tokens)
        .num_embedding_blocks(embedding_blocks == 0 ? num_blocks
                                                    : embedding_blocks);
    if (enable_linear_state) {
      options.enable_linear_state(true)
          .linear_state_num_slots(num_blocks)
          .instance_is_decode(true);
      model_args_.model_type("qwen3_5")
          .num_speculative_tokens(num_speculative_tokens)
          .layer_types({"linear_attention"});
    }
    tokenizer_ = std::make_unique<FakeTokenizer>();
    block_manager_ = std::make_unique<BlockManagerPool>(options, dp_size);
  }

  std::function<ForwardOutput(std::vector<Batch>&)> forward;
  std::function<void(std::vector<Batch>&)> complete_forward;
  std::function<void()> on_link_cluster;

  ForwardOutput step(std::vector<Batch>& batch) override {
    if (forward) {
      return forward(batch);
    }
    NOT_IMPLEMENTED();
  }

  void update_last_step_result(std::vector<Batch>& batch) override {
    if (complete_forward) {
      complete_forward(batch);
      return;
    }
    NOT_IMPLEMENTED();
  }

  const Tokenizer* tokenizer() const override { return tokenizer_.get(); }

  BlockManagerPool* block_manager_pool() const override {
    return block_manager_.get();
  }

  const ModelArgs& model_args() const override { return model_args_; }

  void set_model_args(ModelArgs model_args) {
    model_args_ = std::move(model_args);
  }

  const TokenizerArgs& tokenizer_args() const override { NOT_IMPLEMENTED(); }

  std::vector<int64_t> get_active_activation_memory() const override {
    return {0};
  }

  bool init() override { return true; }

  bool link_cluster(const std::vector<uint64_t>& /*cluster_ids*/,
                    const std::vector<std::string>& /*addrs*/,
                    const std::vector<uint16_t>& /*ports*/,
                    int32_t /*src_dp_size*/,
                    int32_t /*src_kv_split_size*/) override {
    if (on_link_cluster) {
      on_link_cluster();
    }
    return true;
  }

  bool unlink_cluster(const std::vector<uint64_t>& /*cluster_ids*/,
                      const std::vector<std::string>& /*addrs*/,
                      const std::vector<uint16_t>& /*ports*/,
                      int32_t /*src_dp_size*/,
                      int32_t /*src_kv_split_size*/) override {
    return true;
  }

  bool pull_kv_blocks(int32_t /*src_dp_size*/,
                      int32_t /*src_dp_rank*/,
                      const std::vector<uint64_t>& /*src_cluster_ids*/,
                      const std::vector<std::string>& /*src_addrs*/,
                      int32_t /*dst_dp_rank*/,
                      const std::vector<KVTransferMapping>& mappings) override {
    pulled_mappings = mappings;
    if (pull) {
      return pull();
    }
    return true;
  }

  std::function<bool()> pull;
  std::vector<KVTransferMapping> pulled_mappings;

 private:
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<BlockManagerPool> block_manager_;
  ModelArgs model_args_;
};

class TestDisaggPDScheduler final : public DisaggPDScheduler {
 public:
  TestDisaggPDScheduler(Engine* engine,
                        const Options& options,
                        bool start_services = true)
      : DisaggPDScheduler(engine, options, start_services) {}

  void admit_prefill(std::shared_ptr<Request> request,
                     proto::DisaggPDService_Stub* stub = nullptr) {
    {
      std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
      req_to_channel_map_[request->request_id()] = stub;
    }
    request_queue_.write(std::move(request));
  }

  bool reject_queued_for_test(const std::shared_ptr<Request>& request) {
    return reject_queued_request(request);
  }

  bool has_channel(const std::string& req_id) {
    std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
    return req_to_channel_map_.contains(req_id);
  }

  bool reservations_empty() {
    std::lock_guard<std::mutex> lock(received_request_map_mutex_);
    return received_request_map_.empty() &&
           instance_to_received_requests_map_.empty() &&
           request_to_instance_map_.empty();
  }

  void report_queues() { report_request_queue_metrics(); }

  void enqueue_dispatch(std::shared_ptr<Request> request) {
    enqueue_ready_request(std::move(request));
  }

  void stop_dispatch() { prefill_request_queue_.enqueue(nullptr); }
  size_t dispatch_waiting() const {
    return dispatch_waiting_requests_.load(std::memory_order_relaxed) +
           dispatch_active_requests_.load(std::memory_order_relaxed);
  }

  void wait_notifications() {
    std::promise<void> done;
    auto future = done.get_future();
    reservation_release_threadpool_.schedule([&done] { done.set_value(); });
    future.get();
  }

  void wait_responses() { response_processor_->wait_completion(); }

  std::shared_ptr<CancelRequestQueue> cancel_queue_for_test() {
    return cancel_request_queue_;
  }

  std::future<void> prefill_completion() {
    std::promise<void> done;
    auto future = done.get_future();
    prefill_threadpool_.schedule(
        [done = std::move(done)]() mutable { done.set_value(); });
    return future;
  }

  void emit_stream_for_test(const std::shared_ptr<Request>& request) {
    std::vector<std::shared_ptr<Request>> requests{request};
    response_processor_->process_stream_requests(requests);
  }

  void cache_prefill_blocks_for_test(Request* request) {
    cache_prefill_blocks(request);
  }

  using DisaggPDScheduler::do_permanent_rejection;

  bool pop_decode_request_for_test(std::shared_ptr<Request>* request) {
    return request_queue_.read(*request);
  }

  static int64_t amortized_token_latency_for_test(int64_t latency,
                                                  size_t num_tokens) {
    return amortized_token_latency(latency, num_tokens);
  }

  void update_metrics(std::vector<Sequence*>& sequences) {
    update_token_latency_metrics(sequences);
  }
};

class FirstGenerationOrderService final : public proto::DisaggPDService {
 public:
  std::future<void> received() { return received_.get_future(); }

  void FirstGeneration(google::protobuf::RpcController* /*controller*/,
                       const proto::DisaggGenerationsRequests* /*request*/,
                       proto::Status* response,
                       google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    received_.set_value();
    response->set_ok(true);
  }

 private:
  std::promise<void> received_;
};

class DelayedReservationService final : public proto::DisaggPDService {
 public:
  explicit DelayedReservationService(DisaggPDScheduler* scheduler)
      : scheduler_(scheduler) {
    pending_releases_.reserve(3);
  }

  std::future<void> release_started() { return release_started_.get_future(); }

  void ReleaseReservation(google::protobuf::RpcController* /*controller*/,
                          const proto::ReleaseReservationRequest* request,
                          proto::ReleaseReservationResponse* response,
                          google::protobuf::Closure* done) override {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!delay_releases_) {
      lock.unlock();
      complete_release(request, response, done);
      return;
    }
    // Keep the server-side RPC alive without blocking a brpc worker.
    pending_releases_.emplace_back([this, request, response, done] {
      complete_release(request, response, done);
    });
    if (pending_releases_.size() == 1) {
      release_started_.set_value();
    }
  }

  void FirstGeneration(google::protobuf::RpcController* /*controller*/,
                       const proto::DisaggGenerationsRequests* request,
                       proto::Status* response,
                       google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    CHECK_EQ(request->multi_gens_size(), 1);
    const auto& gen = request->multi_gens(0);
    CHECK_EQ(gen.tokens_size(), 1);
    CHECK_EQ(gen.kv_cache_transfer_mode(), "PUSH");
    const auto& token = gen.tokens(0);
    response->set_ok(scheduler_->decode_recv_first_generation(
        gen.req_id(),
        token.token_id(),
        token.has_logprob(),
        token.logprob(),
        token.time_to_first_token_latency_seconds(),
        gen.upstream_elapsed_seconds(),
        /*top_tokens=*/{},
        /*top_logprobs=*/{},
        gen.kv_cache_transfer_mode(),
        /*src_cluster_ids=*/{},
        /*src_addrs=*/{},
        /*source_mappings=*/{},
        gen.dp_size(),
        gen.dp_rank()));
  }

  void resume_releases() {
    std::vector<std::function<void()>> pending;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      delay_releases_ = false;
      pending.swap(pending_releases_);
    }
    for (auto& release : pending) {
      release();
    }
  }

 private:
  void complete_release(const proto::ReleaseReservationRequest* request,
                        proto::ReleaseReservationResponse* response,
                        google::protobuf::Closure* done) {
    brpc::ClosureGuard done_guard(done);
    const bool released = scheduler_->release_reservation(
        request->req_id(), request->reservation_id());
    response->set_result(released
                             ? proto::ReleaseReservationResponse::RELEASED
                             : proto::ReleaseReservationResponse::NOT_WAITING);
  }

  DisaggPDScheduler* scheduler_;
  std::mutex mutex_;
  bool delay_releases_ = true;
  std::promise<void> release_started_;
  std::vector<std::function<void()>> pending_releases_;
};

class ReservationService final : public proto::DisaggPDService {
 public:
  explicit ReservationService(DisaggPDScheduler* scheduler)
      : scheduler_(scheduler) {}

  enum class Failure { NONE, LOST_ACK, TRANSPORT, UNSUPPORTED };
  Failure failure = Failure::NONE;
  std::atomic<int32_t> calls{0};
  std::atomic<int32_t> released{0};

  void ReleaseReservation(google::protobuf::RpcController* controller,
                          const proto::ReleaseReservationRequest* request,
                          proto::ReleaseReservationResponse* response,
                          google::protobuf::Closure* done) override {
    const int32_t attempt = ++calls;
    if (failure == Failure::UNSUPPORTED) {
      proto::DisaggPDService::ReleaseReservation(
          controller, request, response, done);
      return;
    }
    if (failure == Failure::TRANSPORT) {
      controller->SetFailed("injected transport error");
    } else {
      const bool freed = scheduler_->release_reservation(
          request->req_id(), request->reservation_id());
      released += freed ? 1 : 0;
      response->set_result(
          freed ? proto::ReleaseReservationResponse::RELEASED
                : proto::ReleaseReservationResponse::NOT_WAITING);
      if (failure == Failure::LOST_ACK && attempt == 1) {
        controller->SetFailed("injected lost release confirmation");
      }
    }
    if (done != nullptr) {
      done->Run();
    }
  }

 private:
  DisaggPDScheduler* scheduler_;
};

class TestPDOOCScheduler final : public PDOOCScheduler {
 public:
  TestPDOOCScheduler(Engine* engine, const Options& options)
      : PDOOCScheduler(engine, options, false) {}

  void enqueue_dispatch(std::shared_ptr<Request> request) {
    enqueue_ready_request(std::move(request));
  }
  void report_queues() { report_request_queue_metrics(); }
  bool enqueue_ingress(std::shared_ptr<Request> request) {
    return request_queue_.write(std::move(request));
  }
  void stop_dispatch() { prefill_request_queue_.enqueue(nullptr); }
  size_t dispatch_waiting() const {
    return dispatch_waiting_requests_.load(std::memory_order_relaxed) +
           dispatch_active_requests_.load(std::memory_order_relaxed);
  }
  void set_prefill_channel(const std::string& name,
                           proto::DisaggPDService_Stub* stub) {
    instance_channel_map_[name] = stub;
  }
  void set_decode_channel(const std::string& name,
                          proto::DisaggPDService_Stub* stub) {
    set_prefill_channel(name, stub);
    InstanceInfo remote;
    remote.name = name;
    remote_instances_info_[name] = std::move(remote);
    xservice_client_ = XServiceClient::get_instance();
  }
  bool pop_decode_request_for_test(std::shared_ptr<Request>* request) {
    return request_queue_.read(*request);
  }
};

class RetryingAllocationChannel final : public google::protobuf::RpcChannel {
 public:
  explicit RetryingAllocationChannel(
      int32_t retry_status = kDecodeAddNewPromptTooLongStatusCode)
      : retry_status_(retry_status) {}
  std::future<void> retry_started() { return retry_started_.get_future(); }
  void finish_retry() { finish_retry_.set_value(); }
  int32_t attempts() const { return attempts_.load(); }

  void CallMethod(const google::protobuf::MethodDescriptor*,
                  google::protobuf::RpcController*,
                  const google::protobuf::Message*,
                  google::protobuf::Message* response,
                  google::protobuf::Closure* done) override {
    const int32_t attempt = attempts_.fetch_add(1) + 1;
    if (attempt == 2) {
      retry_started_.set_value();
      finish_retry_.get_future().wait();
    }
    auto* responses = static_cast<proto::DisaggResponses*>(response);
    responses->add_resps()->set_status_code(attempt == 1 ? 503 : retry_status_);
    if (done != nullptr) {
      done->Run();
    }
  }

 private:
  const int32_t retry_status_;
  std::atomic<int32_t> attempts_{0};
  std::promise<void> retry_started_;
  std::promise<void> finish_retry_;
};

DisaggPDScheduler::Options make_options() {
  DisaggPDScheduler::Options options;
  options.enable_pd_ooc(true)
      .enable_disagg_pd(true)
      .enable_schedule_overlap(false)
      .instance_role(InstanceRole::PREFILL)
      .max_tokens_per_batch(32)
      .max_seqs_per_batch(4)
      .max_tokens_per_chunk_for_prefill(32)
      .dp_size(1);
  return options;
}

DisaggPDScheduler::Options make_mtp_decode_options(int32_t dp_size = 1) {
  DisaggPDScheduler::Options options = make_options();
  options.instance_role(InstanceRole::DECODE)
      .num_speculative_tokens(1)
      .dp_size(dp_size);
  return options;
}

DisaggPDScheduler::Options make_decode_options() {
  DisaggPDScheduler::Options options = make_options();
  options.instance_role(InstanceRole::DECODE);
  return options;
}

std::shared_ptr<Request> make_request(
    const std::vector<int32_t>& prompt_token_ids,
    const std::string& req_id = "req",
    size_t num_sequences = 1,
    const std::string& x_request_id = "x-request-id",
    const std::string& x_request_time = "x-request-time") {
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  stopping_checker.set_max_context_len(64);
  stopping_checker.set_ignore_eos(true);

  RequestState state("prompt",
                     prompt_token_ids,
                     sampling_param,
                     scheduler_param,
                     stopping_checker,
                     prompt_token_ids.size() + 8,
                     /*n=*/num_sequences,
                     /*best_of=*/num_sequences,
                     /*stream=*/false,
                     /*echo=*/false,
                     /*logprobs=*/false,
                     /*skip_special_tokens=*/false,
                     /*include_usage=*/false,
                     /*mm_data=*/nullptr,
                     /*service_request_id=*/nullptr);

  return std::make_shared<Request>(req_id,
                                   x_request_id,
                                   x_request_time,
                                   std::move(state),
                                   "service-req",
                                   /*source_xservice_addr=*/"service-source");
}

void finish_prefill(Sequence* sequence) {
  CHECK(sequence != nullptr);
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  sequence->append_token(Token(999));
}

class TerminalLogSink final : public google::LogSink {
 public:
  TerminalLogSink() { google::AddLogSink(this); }
  ~TerminalLogSink() override { google::RemoveLogSink(this); }

  void send(google::LogSeverity severity,
            const char* /*full_filename*/,
            const char* /*base_filename*/,
            int /*line*/,
            const google::LogMessageTime& /*logmsgtime*/,
            const char* message,
            size_t message_len) override {
    if (severity != google::GLOG_INFO) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    messages_.append(message, message_len);
    messages_.push_back('\n');
  }

  std::string messages() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return messages_;
  }

 private:
  mutable std::mutex mutex_;
  std::string messages_;
};

size_t first_cache_size(const BlockManagerPool& block_manager) {
  const std::vector<size_t> cache_sizes =
      block_manager.num_blocks_in_prefix_cache();
  CHECK(!cache_sizes.empty());
  return cache_sizes[0];
}

void release_prefix_cache(BlockManagerPool* block_manager) {
  CHECK(block_manager != nullptr);
  const size_t num_data_blocks = block_manager->num_blocks() - 1;
  std::vector<int32_t> token_ids;
  token_ids.reserve(num_data_blocks * block_manager->block_size());
  for (size_t i = 0; i < num_data_blocks * block_manager->block_size(); ++i) {
    token_ids.push_back(static_cast<int32_t>(1000 + i));
  }

  std::shared_ptr<Request> request = make_request(token_ids);
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(block_manager->allocate(sequence));
  block_manager->deallocate(sequence);
  EXPECT_EQ(first_cache_size(*block_manager), 0u);
}

bool recv_first_generation(DisaggPDScheduler* scheduler,
                           const torch::Tensor& mtp_embedding,
                           int32_t num_cached_tokens = 0,
                           double time_to_first_token_latency_seconds = 0.1,
                           double upstream_elapsed_seconds = 0.0,
                           const std::string& transfer_mode = "PUSH") {
  return scheduler->decode_recv_first_generation(
      "req",
      /*token_id=*/42,
      /*has_logprob=*/false,
      /*logprob=*/0.0f,
      time_to_first_token_latency_seconds,
      upstream_elapsed_seconds,
      /*top_tokens=*/{},
      /*top_logprobs=*/{},
      transfer_mode,
      /*src_cluster_ids=*/{},
      /*src_addrs=*/{},
      /*source_mappings=*/{},
      /*src_dp_size=*/1,
      /*src_dp_rank=*/0,
      mtp_embedding,
      num_cached_tokens);
}

void expect_queue_sample(
    const std::string& label,
    std::initializer_list<std::pair<std::string, double>> queues,
    double waiting_count) {
  for (const auto& [queue, count] : queues) {
    auto* gauge =
        MULTI_GAUGE_scheduler_queue_requests.get_stats({queue, label});
    ASSERT_NE(gauge, nullptr);
    EXPECT_DOUBLE_EQ(gauge->get_value(), count) << queue;
  }
  auto* waiting = MULTI_GAUGE_scheduler_waiting_requests.get_stats({label});
  ASSERT_NE(waiting, nullptr);
  EXPECT_DOUBLE_EQ(waiting->get_value(), waiting_count);
}

void set_metrics_model(FakeEngine& engine) {
  ModelArgs model;
  model.n_layers(1)
      .vocab_size(128)
      .hidden_size(64)
      .intermediate_size(128)
      .n_heads(4)
      .n_kv_heads(4);
  engine.set_model_args(std::move(model));
}

template <typename Scheduler, typename Receive>
void check_kv_prepare_metrics(Scheduler& scheduler,
                              FakeEngine& engine,
                              const std::string& label,
                              bool succeeds,
                              Receive receive) {
  scheduler.report_queues();
  expect_queue_sample(label, {{"pd_reservation", 1}, {"pd_kv_prepare", 0}}, 1);
  engine.pull = [&] {
    scheduler.report_queues();
    expect_queue_sample(
        label,
        {{"pd_reservation", 0}, {"pd_kv_prepare", 1}, {"ingress", 0}},
        0);
    return succeeds;
  };
  ScopeGuard clear_pull([&] { engine.pull = nullptr; });
  EXPECT_EQ(receive(), succeeds);
  scheduler.report_queues();
  expect_queue_sample(label,
                      {{"pd_reservation", 0},
                       {"pd_kv_prepare", 0},
                       {"ingress", succeeds ? 1.0 : 0.0}},
                      succeeds ? 1.0 : 0.0);
  EXPECT_EQ(engine.block_manager_pool()->num_used_blocks()[0],
            succeeds ? 1u : 0u);
  std::shared_ptr<Request> queued;
  EXPECT_EQ(scheduler.pop_decode_request_for_test(&queued), succeeds);
  if (queued != nullptr) {
    engine.block_manager_pool()->deallocate(queued.get());
  }
  scheduler.report_queues();
  expect_queue_sample(label, {{"ingress", 0}}, 0);
}

}  // namespace

class KVPrepareMetricsTest : public ::testing::TestWithParam<bool> {};

TEST_P(KVPrepareMetricsTest, FirstGenerationTracksPullUntilCompletion) {
  FakeEngine engine(8, 2);
  auto options = make_decode_options();
  options.server_idx(421);
  TestDisaggPDScheduler scheduler(&engine, options, false);
  auto request = make_request({1, 2});
  request->sequences().front()->kv_state().set_kv_cache_tokens_num(
      request->sequences().front()->num_prompt_tokens());
  ASSERT_TRUE(
      engine.block_manager_pool()->allocate(request->sequences()[0].get()));
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));
  check_kv_prepare_metrics(scheduler, engine, "421", GetParam(), [&] {
    return recv_first_generation(
        &scheduler, torch::Tensor(), 0, 0.0, 0.0, "PULL");
  });
}

TEST_P(KVPrepareMetricsTest, OfflineMigrationTracksPullUntilCompletion) {
  FakeEngine engine(8, 2);
  set_metrics_model(engine);
  auto options = make_decode_options();
  options.server_idx(422).nnodes(1);
  TestPDOOCScheduler scheduler(&engine, options);
  proto::DisaggPDService_Stub stub(nullptr);
  scheduler.set_prefill_channel("prefill", &stub);
  auto request = make_request({1, 2});
  request->state().scheduler_param.offline = true;
  request->sequences().front()->kv_state().set_kv_cache_tokens_num(
      request->sequences().front()->num_prompt_tokens());
  ASSERT_TRUE(
      engine.block_manager_pool()->allocate(request->sequences()[0].get()));
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));
  proto::RemoteToken token;
  token.set_token_id(42);
  check_kv_prepare_metrics(scheduler, engine, "422", GetParam(), [&] {
    return scheduler.decode_recv_multi_generations(
        request->request_id(), {token}, 0.0, "PULL", {}, {}, {}, 1, 0);
  });
}

INSTANTIATE_TEST_SUITE_P(PullResults, KVPrepareMetricsTest, ::testing::Bool());

class DispatchRetryMetricsTest : public ::testing::TestWithParam<bool> {};

TEST_P(DispatchRetryMetricsTest, RetryStopsOnRejectionOrCancellation) {
  FakeEngine engine(8, 2);
  set_metrics_model(engine);
  auto options = make_options();
  options.server_idx(423).nnodes(1);
  RetryingAllocationChannel channel(
      GetParam() ? 503 : kDecodeAddNewPromptTooLongStatusCode);
  proto::DisaggPDService_Stub stub(&channel);
  TestPDOOCScheduler scheduler(&engine, options);
  scheduler.set_decode_channel("decode", &stub);
  auto request = make_request({1, 2});
  request->state().decode_address = "decode";
  auto failed = std::make_shared<std::promise<Status>>();
  auto failure = failed->get_future();
  request->state().output_func = [failed](const RequestOutput& output) {
    failed->set_value(output.status.value());
    return true;
  };
  scheduler.enqueue_dispatch(request);
  scheduler.report_queues();
  expect_queue_sample(
      "423", {{"pd_dispatch", 1}, {"pd_dispatch_active", 0}}, 1);
  auto retry = channel.retry_started();
  auto result =
      std::async(std::launch::async, [&] { scheduler.dispatch_requests(); });
  bool retry_finished = false;
  ScopeGuard stop_dispatch([&] {
    scheduler.stop_dispatch();
    if (!retry_finished) {
      channel.finish_retry();
    }
    result.get();
  });
  ASSERT_EQ(retry.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  EXPECT_EQ(channel.attempts(), 2);
  scheduler.report_queues();
  expect_queue_sample(
      "423", {{"pd_dispatch", 0}, {"pd_dispatch_active", 1}}, 0);
  if (GetParam()) {
    request->set_cancel();
  }
  channel.finish_retry();
  retry_finished = true;
  ASSERT_EQ(failure.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  const Status status = failure.get();
  if (GetParam()) {
    EXPECT_EQ(status.code(), StatusCode::CANCELLED);
  } else {
    EXPECT_EQ(status.reason(), FailureReason::PROMPT_TOO_LONG);
  }
  scheduler.stop_dispatch();
  result.get();
  stop_dispatch.dismiss();
  EXPECT_EQ(channel.attempts(), 2);
  scheduler.report_queues();
  EXPECT_EQ(scheduler.dispatch_waiting(), 0u);
  expect_queue_sample(
      "423", {{"pd_dispatch", 0}, {"pd_dispatch_active", 0}}, 0);
  std::shared_ptr<Request> queued;
  EXPECT_FALSE(scheduler.pop_decode_request_for_test(&queued));
}

INSTANTIATE_TEST_SUITE_P(TerminalResults,
                         DispatchRetryMetricsTest,
                         ::testing::Bool());

class DispatchRemovalMetricsTest : public ::testing::TestWithParam<bool> {};

TEST_P(DispatchRemovalMetricsTest, CancelledOrInvalidDispatchFallsBackToIdle) {
  FakeEngine engine(8, 2);
  auto options = make_options();
  options.server_idx(424);
  TestDisaggPDScheduler scheduler(&engine, options, false);
  auto request = make_request({1, 2});
  auto failed = std::make_shared<std::promise<Status>>();
  auto failure = failed->get_future();
  request->state().output_func = [failed](const RequestOutput& output) {
    failed->set_value(output.status.value());
    return true;
  };
  if (GetParam()) {
    ASSERT_TRUE(
        engine.block_manager_pool()->allocate(request->sequences()[0].get()));
    request->set_cancel();
  }
  scheduler.enqueue_dispatch(request);
  scheduler.report_queues();
  expect_queue_sample("424", {{"pd_dispatch", 1}}, 1);
  auto result =
      std::async(std::launch::async, [&] { scheduler.dispatch_requests(); });
  ScopeGuard stop_dispatch([&] {
    scheduler.stop_dispatch();
    result.get();
  });
  ASSERT_EQ(failure.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  EXPECT_EQ(failure.get().code(),
            GetParam() ? StatusCode::CANCELLED : StatusCode::INVALID_ARGUMENT);
  scheduler.stop_dispatch();
  result.get();
  stop_dispatch.dismiss();
  scheduler.report_queues();
  expect_queue_sample(
      "424", {{"pd_dispatch", 0}, {"pd_dispatch_active", 0}}, 0);
  EXPECT_EQ(engine.block_manager_pool()->num_used_blocks()[0], 0u);
}

INSTANTIATE_TEST_SUITE_P(TerminalStates,
                         DispatchRemovalMetricsTest,
                         ::testing::Bool());

TEST(PDOOCSchedulerTest, OfflineRequestMovesBetweenDistinctWaitingQueues) {
  auto previous_perf_model = perf_model::get_perf_model();
  ScopeGuard restore_perf_model(
      [&] { perf_model::set_perf_model(previous_perf_model); });
  FakeEngine engine(8, 2);
  set_metrics_model(engine);
  auto options = make_options();
  options.server_idx(419).nnodes(1);
  TestPDOOCScheduler scheduler(&engine, options);
  auto offline = make_request({1, 2, 3, 4}, "offline_metrics");
  offline->state().scheduler_param.offline = true;
  offline->state().output_func = [](const RequestOutput&) { return true; };
  scheduler.enqueue_dispatch(offline);
  scheduler.report_queues();
  expect_queue_sample(
      "419", {{"pd_dispatch", 1}, {"ingress", 0}, {"offline_prefill", 0}}, 1);
  auto result =
      std::async(std::launch::async, [&] { scheduler.dispatch_requests(); });
  ScopeGuard stop_dispatch([&] {
    scheduler.stop_dispatch();
    result.get();
  });
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (scheduler.dispatch_waiting() != 0 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  ASSERT_EQ(scheduler.dispatch_waiting(), 0u);
  scheduler.report_queues();
  expect_queue_sample("419", {{"pd_dispatch", 0}, {"ingress", 1}}, 1);
  auto online = make_request({11, 12}, "online_metrics");
  online->state().output_func = [](const RequestOutput&) { return true; };
  ASSERT_TRUE(scheduler.enqueue_ingress(online));
  ASSERT_EQ(scheduler.prepare_batch_test().front().size(), 1u);
  expect_queue_sample("419", {{"ingress", 0}, {"offline_prefill", 1}}, 1);
  online->set_cancel();
  ASSERT_EQ(scheduler.prepare_batch_test().front().size(), 1u);
  expect_queue_sample("419", {{"offline_prefill", 0}}, 0);
  offline->set_cancel();
  scheduler.prepare_batch_test();
  expect_queue_sample("419", {{"offline_prefill", 0}}, 0);
}

TEST(DisaggPDSchedulerTest, RelinkedDecodeRetainsOldEpochRejection) {
  FakeEngine engine(/*num_blocks=*/16, /*block_size=*/2);
  auto options = make_options();
  options.enable_pd_ooc(false).disable_log_stats(true);
  TestDisaggPDScheduler scheduler(&engine, options, /*start_services=*/false);
  auto old_request = make_request({11, 12}, "old-reservation");
  old_request->state().decode_address = "decode-1";
  old_request->state().pd_reservation_id = "old-reservation";
  old_request->state().output_func = [](const RequestOutput&) { return true; };
  scheduler.admit_prefill(old_request);

  ASSERT_TRUE(scheduler.unlink_instance("decode-1",
                                        /*cluster_ids=*/{},
                                        /*addrs=*/{},
                                        /*ports=*/{},
                                        /*dp_size=*/1,
                                        /*src_kv_split_size=*/1));
  ASSERT_TRUE(scheduler.link_instance("decode-1",
                                      /*cluster_ids=*/{},
                                      /*addrs=*/{},
                                      /*ports=*/{},
                                      /*dp_size=*/1,
                                      /*src_kv_split_size=*/1));

  auto new_request = make_request({21, 22}, "new-reservation");
  new_request->state().decode_address = "decode-1";
  new_request->state().pd_reservation_id = "new-reservation";
  new_request->state().decode_link_epoch = 1;
  scheduler.admit_prefill(new_request);
  const auto batches = scheduler.prepare_batch_test();
  scheduler.wait_responses();

  EXPECT_TRUE(old_request->has_terminal_outcome());
  EXPECT_FALSE(new_request->has_terminal_outcome());
  EXPECT_EQ(scheduler.get_running_requests().size(), 1u);
  EXPECT_TRUE(std::any_of(batches.begin(),
                          batches.end(),
                          [](const Batch& batch) { return !batch.empty(); }));
}

TEST(DisaggPDSchedulerTest, LinkBeforePendingUnlinkKeepsDecodeDown) {
  FakeEngine engine(/*num_blocks=*/16, /*block_size=*/2);
  auto options = make_options();
  options.enable_pd_ooc(false).disable_log_stats(true);
  TestDisaggPDScheduler scheduler(&engine, options, /*start_services=*/false);

  std::promise<void> link_entered;
  std::promise<void> release_link;
  engine.on_link_cluster = [&] {
    link_entered.set_value();
    release_link.get_future().wait();
  };
  auto link = std::async(std::launch::async, [&] {
    return scheduler.link_instance("decode-1", {}, {}, {}, 1, 1);
  });
  link_entered.get_future().wait();
  auto unlink = std::async(std::launch::async, [&] {
    return scheduler.unlink_instance("decode-1", {}, {}, {}, 1, 1);
  });

  auto reserved = [](const std::string& id) {
    auto request = make_request({1, 2}, id);
    request->state().decode_address = "decode-1";
    request->state().pd_reservation_id = id;
    request->state().decode_link_epoch = 1;
    request->state().output_func = [](const RequestOutput&) { return true; };
    return request;
  };
  auto probe = reserved("probe");
  bool saw_down = false;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!saw_down && std::chrono::steady_clock::now() < deadline) {
    saw_down = scheduler.reject_queued_for_test(probe);
    std::this_thread::yield();
  }
  release_link.set_value();
  EXPECT_TRUE(link.get());
  EXPECT_TRUE(unlink.get());
  EXPECT_TRUE(saw_down);
  EXPECT_TRUE(scheduler.reject_queued_for_test(reserved("after-link")));
  scheduler.wait_responses();
}

TEST(RequestTerminalOutcomeTest, FirstTerminalEventWins) {
  auto failed = make_request({1, 2}, "failed");
  failed->log_prefill_statistic(/*total_latency=*/0.01);
  EXPECT_EQ(failed->claim_terminal_outcome(std::nullopt),
            RequestTerminalOutcome::NONE);
  EXPECT_EQ(failed->claim_terminal_outcome(
                Status(StatusCode::UNAVAILABLE, "handoff failed")),
            RequestTerminalOutcome::FAILURE);
  EXPECT_EQ(failed->claim_terminal_outcome(Status(StatusCode::OK)),
            RequestTerminalOutcome::NONE);

  auto cancelled = make_request({1, 2}, "cancelled");
  cancelled->set_cancel();
  EXPECT_EQ(cancelled->claim_terminal_outcome(Status(StatusCode::OK)),
            RequestTerminalOutcome::CANCELLED);
  EXPECT_EQ(cancelled->claim_terminal_outcome(
                Status(StatusCode::UNAVAILABLE, "late error")),
            RequestTerminalOutcome::NONE);

  auto status_cancelled = make_request({1, 2}, "status-cancelled");
  EXPECT_EQ(
      status_cancelled->claim_terminal_outcome(Status(StatusCode::CANCELLED)),
      RequestTerminalOutcome::CANCELLED);

  auto multi = make_request({1, 2}, "multi", /*num_sequences=*/2);
  ASSERT_TRUE(multi->expand_sequences(/*share_prefix=*/false));
  ASSERT_EQ(multi->sequences().size(), 2u);
  finish_prefill(multi->sequences()[0].get());
  for (int32_t i = 1; i < 4; ++i) {
    multi->sequences()[0]->append_token(Token(100 + i));
  }
  EXPECT_EQ(multi->claim_terminal_outcome(Status(StatusCode::OK)),
            RequestTerminalOutcome::NONE);
  finish_prefill(multi->sequences()[1].get());
  for (int32_t i = 1; i < 4; ++i) {
    multi->sequences()[1]->append_token(Token(100 + i));
  }
  EXPECT_EQ(multi->claim_terminal_outcome(Status(StatusCode::OK)),
            RequestTerminalOutcome::SUCCESS);
  EXPECT_EQ(multi->claim_terminal_outcome(Status(StatusCode::CANCELLED)),
            RequestTerminalOutcome::NONE);
}

TEST(RequestTerminalOutcomeTest, IdentityPrecedesResultsAndParameters) {
  TerminalLogSink sink;
  auto request = make_request({11, 12}, "readable", /*num_sequences=*/2);
  request->state().sampling_param.temperature = 0.123456;
  ASSERT_TRUE(request->expand_sequences(/*share_prefix=*/false));
  for (const auto& sequence : request->sequences()) {
    finish_prefill(sequence.get());
    for (int32_t i = 1; i < 4; ++i) {
      sequence->append_token(Token(100 + i));
    }
  }
  ASSERT_EQ(request->claim_terminal_outcome(Status(StatusCode::OK)),
            RequestTerminalOutcome::SUCCESS);
  request->log_terminal_statistic(
      RequestTerminalOutcome::SUCCESS, Status(StatusCode::OK), 0.01);
  const std::string logged = sink.messages();
  EXPECT_EQ(std::count(logged.begin(), logged.end(), '\n'), 3);
  EXPECT_EQ(logged.find("record_type: request_terminal, request_id: readable, "
                        "x-request-id: x-request-id, outcome: success, "
                        "status_code: 0"),
            0u);
  for (int32_t i = 0; i < 2; ++i) {
    EXPECT_NE(
        logged.find("record_type: sequence_terminal, request_id: readable, "
                    "x-request-id: x-request-id, sequence: " +
                    std::to_string(i) +
                    ", finish_reason: "
                    "length, prompt_tokens: 2, generated_tokens: 4, "
                    "queue_to_batch: -1.0ms, ttft: "),
        std::string::npos);
  }
  EXPECT_NE(logged.find("temperature: 0.123456, num_prefix_cache_tokens: 0\n"),
            std::string::npos);
  const size_t time = logged.find("x-request-time: x-request-time");
  ASSERT_NE(time, std::string::npos);
  EXPECT_EQ(logged.find("x-request-time:", time + 1), std::string::npos);
}

TEST(RequestTerminalOutcomeTest, ErrorsStayVisibleWithoutEmptyMetadata) {
  TerminalLogSink sink;
  auto request = make_request({11, 12},
                              "empty-metadata",
                              /*num_sequences=*/1,
                              /*x_request_id=*/"",
                              /*x_request_time=*/"");
  request->log_terminal_statistic(
      RequestTerminalOutcome::CANCELLED, Status(StatusCode::OK), 0.01);
  const std::string logged = sink.messages();
  EXPECT_NE(logged.find("record_type: request_terminal, request_id: "
                        "empty-metadata, outcome: cancelled, status_code: 1\n"),
            std::string::npos);
  EXPECT_NE(
      logged.find("record_type: sequence_terminal, request_id: empty-metadata, "
                  "sequence: 0, outcome: cancelled, "
                  "status_code: 1, status_msg: Cancelled, prompt_tokens: 2"),
      std::string::npos);
  EXPECT_EQ(logged.find("x-request-id:"), std::string::npos);
  EXPECT_EQ(logged.find("x-request-time:"), std::string::npos);
  EXPECT_EQ(logged.find("finish_reason:"), std::string::npos);
  EXPECT_EQ(logged.find("generated_tokens:"), std::string::npos);
}

TEST(AsyncResponseProcessorTest,
     BatchedDuplicatePreservesValidRequestIdentity) {
  FakeTokenizer tokenizer;
  AsyncResponseProcessor processor(&tokenizer,
                                   std::nullopt,
                                   /*enable_service_routing=*/true,
                                   /*disable_log_stats=*/true,
                                   [](std::shared_ptr<Request>) {});
  auto duplicate = make_request({1, 2}, "already-claimed");
  auto valid = make_request({1, 2}, "still-valid");
  duplicate->set_cancel();
  valid->set_cancel();
  ASSERT_EQ(duplicate->claim_terminal_outcome(Status(StatusCode::OK)),
            RequestTerminalOutcome::CANCELLED);

  std::atomic<int32_t> callbacks{0};
  duplicate->state().outputs_func =
      [&callbacks](const std::vector<RequestOutput>& outputs) {
        EXPECT_EQ(outputs.size(), 1u);
        if (!outputs.empty()) {
          EXPECT_EQ(outputs[0].request_id, "still-valid");
        }
        ++callbacks;
        return std::vector<bool>{true};
      };
  std::vector<std::shared_ptr<Request>> requests{duplicate, valid};
  processor.process_completed_requests(requests);
  processor.wait_completion();
  std::vector<std::shared_ptr<Request>> invalid_only{duplicate};
  processor.process_completed_requests(invalid_only);
  processor.wait_completion();
  EXPECT_EQ(callbacks.load(), 1);
}

TEST(DisaggPDSchedulerTest, NonStreamHandoffFailureHasOneTerminalOutcome) {
  TerminalLogSink sink;
  FakeEngine engine(
      /*num_blocks=*/16, /*block_size=*/2, /*num_speculative_tokens=*/1);
  auto options = make_options();
  options.enable_chunked_prefill(false)
      .kv_cache_transfer_mode("PUSH")
      .num_speculative_tokens(1);
  TestDisaggPDScheduler prefill(&engine, options, /*start_services=*/false);
  auto request = make_request({11, 12}, "failed-nonstream-handoff");
  std::atomic<int32_t> callbacks{0};
  request->state().output_func = [&callbacks](const RequestOutput& output) {
    EXPECT_TRUE(output.status.has_value());
    EXPECT_TRUE(output.status->ok());
    EXPECT_TRUE(output.finished_on_prefill_instance);
    ++callbacks;
    return true;
  };
  prefill.admit_prefill(request);
  auto batches = prefill.prepare_batch_test();
  ASSERT_FALSE(batches[0].empty());
  finish_prefill(request->sequences()[0].get());
  prefill.prefill_send_first_generation();
  prefill.prefill_completion().get();

  EXPECT_EQ(callbacks.load(), 1);
  EXPECT_TRUE(request->has_terminal_outcome());
  EXPECT_FALSE(prefill.has_channel(request->request_id()));
  const std::string logged = sink.messages();
  const std::string terminal_record = "record_type: request_terminal";
  const size_t first = logged.find(terminal_record);
  ASSERT_NE(first, std::string::npos) << logged;
  EXPECT_EQ(logged.find(terminal_record, first + 1), std::string::npos)
      << logged;
  const std::string terminal_log =
      logged.substr(first, logged.find('\n', first) - first);
  EXPECT_NE(terminal_log.find("outcome: failure"), std::string::npos);
  EXPECT_NE(terminal_log.find("request_id: failed-nonstream-handoff"),
            std::string::npos);
  const size_t prefill_record = logged.find("record_type: pd_prefill_stage");
  ASSERT_NE(prefill_record, std::string::npos);
  const std::string prefill_log = logged.substr(
      prefill_record, logged.find('\n', prefill_record) - prefill_record);
  EXPECT_EQ(prefill_log.find("record_type: pd_prefill_stage, request_id: "
                             "failed-nonstream-handoff, "
                             "x-request-id: x-request-id, sequence: 0"),
            0u);
  EXPECT_NE(prefill_log.find("sequence: 0"), std::string::npos);
  EXPECT_NE(prefill_log.find("request_id: failed-nonstream-handoff"),
            std::string::npos);
  EXPECT_NE(prefill_log.find("x-request-time: x-request-time"),
            std::string::npos);
  EXPECT_EQ(prefill_log.find("finish_reason:"), std::string::npos);
}

TEST(DisaggPDSchedulerTest, CachesPrefillBlocksBeforeRelease) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  BlockManagerPool* block_manager = engine.block_manager_pool();

  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(block_manager->allocate(sequence));
  finish_prefill(sequence);

  scheduler.cache_prefill_blocks_for_test(request.get());
  EXPECT_EQ(first_cache_size(*block_manager), 2u);

  block_manager->deallocate(request.get());
  EXPECT_EQ(sequence->kv_state().num_blocks(BlockType::KV), 0u);
  EXPECT_EQ(first_cache_size(*block_manager), 2u);

  std::shared_ptr<Request> matched_request = make_request({1, 2, 3, 4, 5});
  Sequence* matched_sequence = matched_request->sequences()[0].get();
  block_manager->allocate_shared(matched_sequence);

  EXPECT_EQ(matched_sequence->kv_state().shared_blocks_num(BlockType::KV), 2u);
  block_manager->deallocate(matched_sequence);
  release_prefix_cache(block_manager);
}

TEST(DisaggPDSchedulerTest, CacheSkipsExistingSharedBlocks) {
  FakeEngine engine(/*num_blocks=*/10, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  BlockManagerPool* block_manager = engine.block_manager_pool();

  std::shared_ptr<Request> seed_request = make_request({1, 2, 3, 4});
  Sequence* seed_sequence = seed_request->sequences()[0].get();
  ASSERT_TRUE(block_manager->allocate(seed_sequence));
  finish_prefill(seed_sequence);
  scheduler.cache_prefill_blocks_for_test(seed_request.get());
  block_manager->deallocate(seed_request.get());
  ASSERT_EQ(first_cache_size(*block_manager), 2u);

  std::shared_ptr<Request> extended_request = make_request({1, 2, 3, 4, 5, 6});
  Sequence* extended_sequence = extended_request->sequences()[0].get();
  block_manager->allocate_shared(extended_sequence);
  ASSERT_EQ(extended_sequence->kv_state().shared_blocks_num(BlockType::KV), 2u);
  ASSERT_TRUE(block_manager->allocate(extended_sequence,
                                      extended_sequence->num_prompt_tokens()));
  finish_prefill(extended_sequence);

  scheduler.cache_prefill_blocks_for_test(extended_request.get());
  EXPECT_EQ(first_cache_size(*block_manager), 3u);

  block_manager->deallocate(extended_request.get());
  EXPECT_EQ(first_cache_size(*block_manager), 3u);
  release_prefix_cache(block_manager);
}

TEST(DisaggPDSchedulerTest, MtpFirstGenerationRequiresBootstrapBeforeQueue) {
  FakeEngine engine(/*num_blocks=*/8,
                    /*block_size=*/2,
                    /*num_speculative_tokens=*/1);
  TestDisaggPDScheduler scheduler(&engine, make_mtp_decode_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  ASSERT_TRUE(
      engine.block_manager_pool()->allocate(request->sequences()[0].get()));
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  EXPECT_FALSE(recv_first_generation(&scheduler, torch::Tensor()));
  std::shared_ptr<Request> queued;
  EXPECT_FALSE(scheduler.pop_decode_request_for_test(&queued));
}

TEST(DisaggPDSchedulerTest, MtpFirstGenerationStoresBootstrapThenQueues) {
  FakeEngine engine(/*num_blocks=*/8,
                    /*block_size=*/2,
                    /*num_speculative_tokens=*/1);
  TestDisaggPDScheduler scheduler(&engine, make_mtp_decode_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(engine.block_manager_pool()->allocate(sequence));
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  ASSERT_GE(sequence->get_embedding_block_id(), 0);
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  torch::Tensor embedding = torch::tensor({1.0f, 2.0f});
  EXPECT_TRUE(recv_first_generation(&scheduler, embedding));

  std::shared_ptr<Request> queued;
  ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
  EXPECT_EQ(queued->request_id(), "req");
  EXPECT_EQ(queued->sequences()[0]->tokens().back(), 42);
  EXPECT_TRUE(torch::equal(
      queued->sequences()[0]->get_mtp_bootstrap_embedding(), embedding));
}

TEST(DisaggPDSchedulerTest, GroupedPullAlignsActiveSwaSuffix) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_decode_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(engine.block_manager_pool()->allocate(sequence));
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());

  BlockManager::Options swa_options;
  swa_options.num_blocks(8).block_size(2);
  BlockManagerImpl swa_manager(swa_options);
  std::vector<Block> live_swa_blocks = swa_manager.allocate(2);
  const std::vector<uint64_t> live_swa_ids = {
      static_cast<uint64_t>(live_swa_blocks[0].id()),
      static_cast<uint64_t>(live_swa_blocks[1].id())};
  std::vector<Block> logical_swa_blocks(2);
  logical_swa_blocks.insert(logical_swa_blocks.end(),
                            std::make_move_iterator(live_swa_blocks.begin()),
                            std::make_move_iterator(live_swa_blocks.end()));
  sequence->kv_state().add_blocks(BlockType::SWA,
                                  std::move(logical_swa_blocks));
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  KVTransferMapping source_mapping;
  source_mapping.group_id = cache_group_id(BlockType::SWA);
  source_mapping.remote_ids = {101, 102};
  ASSERT_TRUE(scheduler.decode_recv_first_generation(
      "req",
      /*token_id=*/42,
      /*has_logprob=*/false,
      /*logprob=*/0.0f,
      /*time_to_first_token_latency_seconds=*/0.1,
      /*upstream_elapsed_seconds=*/0.0,
      /*top_tokens=*/{},
      /*top_logprobs=*/{},
      /*kv_cache_transfer_mode=*/"PULL",
      /*src_cluster_ids=*/{1},
      /*src_addrs=*/{"remote"},
      /*source_mappings=*/{source_mapping},
      /*src_dp_size=*/1,
      /*src_dp_rank=*/0));

  ASSERT_EQ(engine.pulled_mappings.size(), 1U);
  EXPECT_EQ(engine.pulled_mappings[0].group_id, cache_group_id(BlockType::SWA));
  EXPECT_EQ(engine.pulled_mappings[0].local_ids, live_swa_ids);
  EXPECT_EQ(engine.pulled_mappings[0].remote_ids,
            (std::vector<uint64_t>{101, 102}));

  std::shared_ptr<Request> queued;
  ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
  engine.block_manager_pool()->deallocate(queued.get());
  queued->sequences()[0]->kv_state().erase_blocks(BlockType::SWA);
}

TEST(DisaggPDSchedulerTest, LinearStatePullFeedsFrameworkDecodeSlot) {
  ScopedPrefillChunkSize chunk_size(/*chunk_size=*/32);
  FakeEngine engine(/*num_blocks=*/8,
                    /*block_size=*/2,
                    /*num_speculative_tokens=*/1,
                    /*dp_size=*/1,
                    /*embedding_blocks=*/8,
                    /*enable_linear_state=*/true);
  TestDisaggPDScheduler scheduler(&engine, make_mtp_decode_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(engine.block_manager_pool()->allocate(sequence));
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  const int32_t local_linear_state_id = sequence->get_linear_state_slot_id();
  ASSERT_GE(local_linear_state_id, 0);
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  KVTransferMapping source_mapping;
  source_mapping.group_id = cache_group_id(BlockType::LINEAR);
  source_mapping.remote_ids = {101};
  torch::Tensor bootstrap_embedding = torch::tensor({1.0f, 2.0f});
  ASSERT_TRUE(scheduler.decode_recv_first_generation(
      "req",
      /*token_id=*/42,
      /*has_logprob=*/false,
      /*logprob=*/0.0f,
      /*time_to_first_token_latency_seconds=*/0.1,
      /*upstream_elapsed_seconds=*/0.0,
      /*top_tokens=*/{},
      /*top_logprobs=*/{},
      /*kv_cache_transfer_mode=*/"PULL",
      /*src_cluster_ids=*/{1},
      /*src_addrs=*/{"remote"},
      /*source_mappings=*/{source_mapping},
      /*src_dp_size=*/1,
      /*src_dp_rank=*/0,
      bootstrap_embedding));

  ASSERT_EQ(engine.pulled_mappings.size(), 1u);
  EXPECT_EQ(engine.pulled_mappings[0].group_id,
            cache_group_id(BlockType::LINEAR));
  EXPECT_EQ(engine.pulled_mappings[0].remote_ids, std::vector<uint64_t>({101}));
  EXPECT_EQ(
      engine.pulled_mappings[0].local_ids,
      std::vector<uint64_t>({static_cast<uint64_t>(local_linear_state_id)}));

  std::vector<Batch> batches = scheduler.prepare_batch_test();
  ASSERT_EQ(batches.size(), 1u);
  ASSERT_EQ(batches[0].size(), 1u);
  ForwardInput input = batches[0].prepare_forward_input(
      /*num_decoding_tokens=*/2,
      /*min_decoding_bach_size=*/0,
      engine.model_args());
  EXPECT_EQ(input.input_params.embedding.linear_state_ids,
            std::vector<int32_t>({local_linear_state_id}));
  EXPECT_EQ(input.input_params.embedding.linear_state_read_ids,
            std::vector<int32_t>({local_linear_state_id}));
  ASSERT_EQ(input.input_params.attention.host.kv_cache_tokens_nums.size(), 1u);
  EXPECT_GT(input.input_params.attention.host.kv_cache_tokens_nums[0], 0);
}

TEST(DisaggPDSchedulerTest, FirstDecodeTokenLatencyIsNonNegative) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(engine.block_manager_pool()->allocate(sequence));
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  EXPECT_TRUE(recv_first_generation(&scheduler, torch::Tensor()));

  std::shared_ptr<Request> queued;
  ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
  // Base rebuilt in decode_recv_first_generation must not sit in the future:
  // pre-fix it was created_time + ttft (~now+100ms), yielding a negative ITL.
  int64_t first_itl = queued->sequences()[0]->tbt(absl::Now());
  EXPECT_GE(first_itl, 0);
}

TEST(DisaggPDSchedulerTest, DecodeLatencyIncludesPrefillTtft) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(engine.block_manager_pool()->allocate(sequence));
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  constexpr double kPrefillTtftSeconds = 10.0;
  EXPECT_TRUE(recv_first_generation(&scheduler,
                                    torch::Tensor(),
                                    /*num_cached_tokens=*/0,
                                    kPrefillTtftSeconds));

  std::shared_ptr<Request> queued;
  ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
  EXPECT_GE(queued->end_to_end_latency_seconds(), kPrefillTtftSeconds);
  EXPECT_LT(queued->elapsed_seconds(), kPrefillTtftSeconds);
}

TEST(DisaggPDSchedulerTest, DecodeLatencyUsesCumulativeUpstreamElapsed) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(engine.block_manager_pool()->allocate(sequence));
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  constexpr double kTtftSeconds = 2.0;
  constexpr double kUpstreamElapsedSeconds = 10.0;
  EXPECT_TRUE(recv_first_generation(&scheduler,
                                    torch::Tensor(),
                                    /*num_cached_tokens=*/0,
                                    kTtftSeconds,
                                    kUpstreamElapsedSeconds));

  std::shared_ptr<Request> queued;
  ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
  EXPECT_DOUBLE_EQ(
      queued->sequences()[0]->time_to_first_token_latency_seconds(),
      kTtftSeconds);
  EXPECT_GE(queued->end_to_end_latency_seconds(), kUpstreamElapsedSeconds);
  EXPECT_LT(queued->elapsed_seconds(), kTtftSeconds);

  queued->sequences()[0]->append_token(Token(43));
  queued->sequences()[0]->append_token(Token(44));
  queued->sequences()[0]->append_token(Token(45));
  const size_t generated_tokens =
      queued->sequences()[0]->num_generated_tokens();
  ASSERT_EQ(generated_tokens, 4U);
  const double generation_latency_seconds =
      queued->end_to_end_latency_seconds() - kTtftSeconds;
  const double average_tpot_milliseconds =
      generation_latency_seconds * 1000.0 / (generated_tokens - 1);
  EXPECT_GE(average_tpot_milliseconds,
            (kUpstreamElapsedSeconds - kTtftSeconds) * 1000.0 /
                (generated_tokens - 1));
}

TEST(DisaggPDSchedulerTest, PreservesPrefillCachedTokensOnDecodeRequest) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(engine.block_manager_pool()->allocate(sequence));
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  ASSERT_TRUE(recv_first_generation(
      &scheduler, torch::Tensor(), /*num_cached_tokens=*/2));

  std::shared_ptr<Request> queued;
  ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
  EXPECT_EQ(queued->num_prefix_cache_tokens(), 2u);
}

TEST(DisaggPDSchedulerTest, PromptAtDecodeBlockCapacityIsNotPermanent) {
  EXPECT_FALSE(exceeds_decode_capacity(
      /*num_prompt_tokens=*/6, /*block_size=*/2, /*num_blocks=*/4));
}

TEST(DisaggPDSchedulerTest, OnlyOversizedDecodeResponseIsTerminal) {
  EXPECT_FALSE(is_permanent_rejection(/*status_code=*/404));
  EXPECT_TRUE(is_permanent_rejection(kDecodeAddNewPromptTooLongStatusCode));
  EXPECT_FALSE(is_permanent_rejection(/*status_code=*/500));
}

TEST(DisaggPDSchedulerTest, DetectsRankPreservingFlatKvGroups) {
  proto::DisaggResponse response;
  response.add_groups()->set_group_id(cache_group_id(BlockType::KV));
  response.add_groups()->set_group_id(cache_group_id(BlockType::LINEAR));

  EXPECT_TRUE(has_rank_preserving_kv_groups(response));
}

TEST(DisaggPDSchedulerTest, KeepsExpandedGroupedCacheMappings) {
  proto::DisaggResponse response;
  response.add_groups()->set_group_id(cache_group_id(BlockType::C4));

  EXPECT_FALSE(has_rank_preserving_kv_groups(response));
}

// The shared dispatch resolution (resolve_kv_split_plan) is what BOTH
// dispatch overrides -- DisaggPDScheduler::dispatch_requests and
// PDOOCScheduler::dispatch_requests/dispatch_offline_requests -- compose the
// TransferKVInfo mapping claim from. The OOC overrides historically built
// the info without the width plan, so an equal-width pair (decode declaring
// kv_split_size > 1) was always remapped by the kv_split stride and an
// unreconcilable pair was never rejected; these pins hold the helper to the
// contract that closed both holes.
namespace {

constexpr int32_t kTestSrcKvSplitSize = 2;

InstanceInfo make_remote_info(int32_t kv_split_size) {
  InstanceInfo remote;
  remote.name = "decode";
  remote.kv_split_size = kv_split_size;
  return remote;
}

InstanceInfo make_remote_info(int32_t kv_split_size,
                              int32_t cp_index_write_mode) {
  InstanceInfo remote = make_remote_info(kv_split_size);
  remote.cp_index_write_mode = cp_index_write_mode;
  return remote;
}

proto::DisaggResponse make_rank_preserving_response() {
  proto::DisaggResponse response;
  response.add_groups()->set_group_id(cache_group_id(BlockType::KV));
  response.add_groups()->set_group_id(cache_group_id(BlockType::LINEAR));
  return response;
}

}  // namespace

TEST(DisaggPDSchedulerTest,
     ResolveKvSplitPlanLicensesRankLocalMappingForEqualWidthPair) {
  const proto::DisaggResponse response = make_rank_preserving_response();

  // The pre-RPC resolution: the plan is already exact while the mapping
  // claim stays false because the destination's group layout is not yet
  // known -- this is the state the dispatch gates consume.
  const KvSplitDispatchPlan pre_rpc =
      resolve_kv_split_plan(kTestSrcKvSplitSize,
                            kCpIndexWriteModeReplicated,
                            make_remote_info(kTestSrcKvSplitSize),
                            /*response=*/nullptr);
  EXPECT_EQ(pre_rpc.plan, KvSplitWidthPlan::RANK_LOCAL);
  EXPECT_FALSE(pre_rpc.rank_local_mapping);
  EXPECT_TRUE(pre_rpc.reason.empty());

  // The post-RPC resolution with a rank-preserving allocation: the 1:1
  // mapping claim the OOC dispatch must now set on its TransferKVInfo.
  const KvSplitDispatchPlan post_rpc =
      resolve_kv_split_plan(kTestSrcKvSplitSize,
                            kCpIndexWriteModeReplicated,
                            make_remote_info(kTestSrcKvSplitSize),
                            &response);
  EXPECT_EQ(post_rpc.plan, KvSplitWidthPlan::RANK_LOCAL);
  EXPECT_TRUE(post_rpc.rank_local_mapping);
  EXPECT_TRUE(post_rpc.reason.empty());

  // A group layout that shards KV across ranks must not claim the rank-local
  // mapping even at equal widths: the strided filter remaps each rank's own
  // slice instead.
  proto::DisaggResponse expanded = make_rank_preserving_response();
  expanded.add_groups()->set_group_id(cache_group_id(BlockType::C4));
  const KvSplitDispatchPlan expanded_plan =
      resolve_kv_split_plan(kTestSrcKvSplitSize,
                            kCpIndexWriteModeReplicated,
                            make_remote_info(kTestSrcKvSplitSize),
                            &expanded);
  EXPECT_EQ(expanded_plan.plan, KvSplitWidthPlan::RANK_LOCAL);
  EXPECT_FALSE(expanded_plan.rank_local_mapping);
}

TEST(DisaggPDSchedulerTest,
     ResolveKvSplitPlanRefusesUnreconcilablePairsWithReason) {
  // An inexpressible width pair: no mapping may be derived, and the reason
  // names both widths for the INVALID_ARGUMENT the dispatch paths emit.
  const KvSplitDispatchPlan mismatched =
      resolve_kv_split_plan(kTestSrcKvSplitSize,
                            kCpIndexWriteModeReplicated,
                            make_remote_info(/*kv_split_size=*/3),
                            /*response=*/nullptr);
  EXPECT_EQ(mismatched.plan, KvSplitWidthPlan::UNRECONCILABLE);
  EXPECT_FALSE(mismatched.rank_local_mapping);
  EXPECT_NE(mismatched.reason.find(std::to_string(kTestSrcKvSplitSize)),
            std::string::npos)
      << mismatched.reason;
  EXPECT_NE(mismatched.reason.find("3"), std::string::npos)
      << mismatched.reason;

  // An undeclared width (proto3's 0 on the registration record) is equally
  // refused rather than read as 1 -- for a SPLIT source, whose stride
  // remapping depends on the destination's declared width.
  const KvSplitDispatchPlan undeclared =
      resolve_kv_split_plan(kTestSrcKvSplitSize,
                            kCpIndexWriteModeReplicated,
                            make_remote_info(/*kv_split_size=*/0),
                            /*response=*/nullptr);
  EXPECT_EQ(undeclared.plan, KvSplitWidthPlan::UNRECONCILABLE);
  EXPECT_FALSE(undeclared.reason.empty());

  // A width-1 source against an undeclared destination stays expressible:
  // the source never splits, skips the stride filter, and maps onto the
  // destination's table one-to-one whatever the destination declares or
  // fails to declare -- the pre-field legacy semantics (a non-CP prefill
  // serving an old-service-registered decode transferred with stride=1).
  const KvSplitDispatchPlan legacy_pair =
      resolve_kv_split_plan(/*src_kv_split_size=*/1,
                            kCpIndexWriteModeReplicated,
                            make_remote_info(/*kv_split_size=*/0),
                            /*response=*/nullptr);
  EXPECT_EQ(legacy_pair.plan, KvSplitWidthPlan::RANK_LOCAL);
  EXPECT_TRUE(legacy_pair.reason.empty());

  // An undeclared SOURCE is always refused: nothing can be validated about
  // how its blocks are laid out.
  const KvSplitDispatchPlan undeclared_source =
      resolve_kv_split_plan(/*src_kv_split_size=*/0,
                            kCpIndexWriteModeReplicated,
                            make_remote_info(kTestSrcKvSplitSize),
                            /*response=*/nullptr);
  EXPECT_EQ(undeclared_source.plan, KvSplitWidthPlan::UNRECONCILABLE);
  EXPECT_FALSE(undeclared_source.reason.empty());

  // A sharded-write source against an equal-width destination: refused with
  // the write mode named (the registration-side half of the contract).
  const proto::DisaggResponse rank_preserving = make_rank_preserving_response();
  const KvSplitDispatchPlan sharded =
      resolve_kv_split_plan(kTestSrcKvSplitSize,
                            kCpIndexWriteModeSharded,
                            make_remote_info(kTestSrcKvSplitSize),
                            &rank_preserving);
  EXPECT_EQ(sharded.plan, KvSplitWidthPlan::UNRECONCILABLE);
  EXPECT_FALSE(sharded.rank_local_mapping);
  EXPECT_NE(sharded.reason.find("write mode"), std::string::npos)
      << sharded.reason;

  // A sharded-write DESTINATION against a replicated source: the same
  // refusal from the other direction. resolve_kv_split_plan reads
  // remote_info.cp_index_write_mode (the field the registration record
  // already publishes), so a destination's sharded declaration is enforced,
  // not silently ignored.
  const KvSplitDispatchPlan sharded_dst = resolve_kv_split_plan(
      kTestSrcKvSplitSize,
      kCpIndexWriteModeReplicated,
      make_remote_info(kTestSrcKvSplitSize, kCpIndexWriteModeSharded),
      &rank_preserving);
  EXPECT_EQ(sharded_dst.plan, KvSplitWidthPlan::UNRECONCILABLE);
  EXPECT_FALSE(sharded_dst.rank_local_mapping);
  EXPECT_NE(sharded_dst.reason.find("write mode"), std::string::npos)
      << sharded_dst.reason;
  EXPECT_NE(sharded_dst.reason.find("the destination"), std::string::npos)
      << sharded_dst.reason;

  // The settled heterogeneous pair stays expressible and strided.
  const KvSplitDispatchPlan settled =
      resolve_kv_split_plan(kTestSrcKvSplitSize,
                            kCpIndexWriteModeReplicated,
                            make_remote_info(/*kv_split_size=*/1),
                            &rank_preserving);
  EXPECT_EQ(settled.plan, KvSplitWidthPlan::STRIDED);
  EXPECT_FALSE(settled.rank_local_mapping);
  EXPECT_TRUE(settled.reason.empty());
}

TEST(DisaggPDSchedulerTest, PromptBeyondDecodeBlockCapacityIsPermanent) {
  EXPECT_TRUE(exceeds_decode_capacity(
      /*num_prompt_tokens=*/7, /*block_size=*/2, /*num_blocks=*/4));
}

TEST(DisaggPDSchedulerTest, TemporaryDecodeBlockPressureIsNotPermanent) {
  FakeEngine engine(/*num_blocks=*/4, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  BlockManagerPool* block_manager = engine.block_manager_pool();
  std::shared_ptr<Request> holder = make_request({1, 2, 3, 4, 5, 6});
  ASSERT_TRUE(block_manager->try_allocate(holder->sequences()[0].get()));
  std::shared_ptr<Request> request = make_request({7, 8});
  Sequence* sequence = request->sequences()[0].get();

  EXPECT_EQ(scheduler.try_admit(request), DecodeAdmissionResult::REJECTED);
  EXPECT_FALSE(scheduler.exceeds_decode_capacity(sequence));

  block_manager->deallocate(holder.get());
}

TEST(DisaggPDSchedulerTest, OversizedDecodePromptIsPermanent) {
  FakeEngine engine(/*num_blocks=*/4, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  BlockManagerPool* block_manager = engine.block_manager_pool();
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4, 5, 6, 7});
  Sequence* sequence = request->sequences()[0].get();

  EXPECT_EQ(scheduler.try_admit(request), DecodeAdmissionResult::REJECTED);
  EXPECT_TRUE(scheduler.exceeds_decode_capacity(sequence));

  block_manager->deallocate(request.get());
}

TEST(DisaggPDSchedulerTest, PermanentDecodeRejectionIsPublicContextError) {
  FakeEngine engine(/*num_blocks=*/4, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4, 5, 6, 7});
  std::promise<Status> result;
  auto future = result.get_future();
  request->state().output_func = [&result](const RequestOutput& output) {
    result.set_value(output.status.value());
    return true;
  };
  scheduler.do_permanent_rejection(request);
  ASSERT_EQ(future.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  const Status status = future.get();
  EXPECT_EQ(status.code(), StatusCode::RESOURCE_EXHAUSTED);
  EXPECT_EQ(status.reason(), FailureReason::PROMPT_TOO_LONG);
  const PublicHttpError error = public_error_from_status(status);
  EXPECT_EQ(error.status_code, 400);
  EXPECT_EQ(error.code, "context_length_exceeded");
  EXPECT_FALSE(scheduler.has_channel(request->request_id()));
}

TEST(DisaggPDSchedulerTest, InvalidPrefillCachedTokensFallBackToZero) {
  for (int32_t num_cached_tokens : {-1, 5}) {
    FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
    TestDisaggPDScheduler scheduler(&engine, make_options());
    std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
    Sequence* sequence = request->sequences()[0].get();
    ASSERT_TRUE(engine.block_manager_pool()->allocate(sequence));
    sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
    ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

    ASSERT_TRUE(
        recv_first_generation(&scheduler, torch::Tensor(), num_cached_tokens));

    std::shared_ptr<Request> queued;
    ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
    EXPECT_EQ(queued->num_prefix_cache_tokens(), 0u);
  }
}

TEST(DisaggPDSchedulerTest, AmortizedTokenLatencyRoundsHalfUp) {
  // Amortized per-token latency is round(latency / n) via (latency + n/2) / n.
  EXPECT_EQ(TestDisaggPDScheduler::amortized_token_latency_for_test(100, 4),
            25);
  EXPECT_EQ(TestDisaggPDScheduler::amortized_token_latency_for_test(101, 4),
            25);
  EXPECT_EQ(TestDisaggPDScheduler::amortized_token_latency_for_test(102, 4),
            26);
  EXPECT_EQ(TestDisaggPDScheduler::amortized_token_latency_for_test(50, 5), 10);
  EXPECT_EQ(TestDisaggPDScheduler::amortized_token_latency_for_test(53, 5), 11);
  // With a single committed token amortized latency equals the raw latency.
  EXPECT_EQ(TestDisaggPDScheduler::amortized_token_latency_for_test(37, 1), 37);
}

TEST(DisaggPDSchedulerTest, SchedulerDoesNotOverwriteSpeculativeOutputGauge) {
  FakeEngine engine(/*num_blocks=*/8,
                    /*block_size=*/2,
                    /*num_speculative_tokens=*/1);
  TestDisaggPDScheduler scheduler(&engine, make_mtp_decode_options());
  GAUGE_SET(speculative_mean_acceptance_length, 4.25);

  std::shared_ptr<Request> first_request = make_request({1, 2, 3, 4});
  Sequence* first_sequence = first_request->sequences()[0].get();
  first_sequence->kv_state().set_kv_cache_tokens_num(
      first_sequence->num_prompt_tokens());
  for (int32_t token_id = 10; token_id < 15; ++token_id) {
    first_sequence->append_token(Token(token_id));
  }

  std::shared_ptr<Request> second_request = make_request({5, 6, 7, 8});
  Sequence* second_sequence = second_request->sequences()[0].get();
  second_sequence->kv_state().set_kv_cache_tokens_num(
      second_sequence->num_prompt_tokens());
  for (int32_t token_id = 20; token_id < 23; ++token_id) {
    second_sequence->append_token(Token(token_id));
  }

  std::vector<Sequence*> sequences = {first_sequence, second_sequence};
  scheduler.update_metrics(sequences);

  EXPECT_DOUBLE_EQ(GAUGE_speculative_mean_acceptance_length.get_value(), 4.25);
  EXPECT_EQ(first_sequence->generated_tokens_since_latency(), 0u);
  EXPECT_EQ(second_sequence->generated_tokens_since_latency(), 0u);
}

TEST(DisaggPDSchedulerTest, SpeculativeMetricsSilentWhenDisabled) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  // make_options() keeps num_speculative_tokens at its default of 0.
  TestDisaggPDScheduler scheduler(&engine, make_options());

  GAUGE_SET(speculative_mean_acceptance_length, -1.0);

  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  sequence->append_token(Token(10));
  sequence->append_token(Token(11));
  std::vector<Sequence*> sequences = {sequence};

  scheduler.update_metrics(sequences);

  EXPECT_DOUBLE_EQ(GAUGE_speculative_mean_acceptance_length.get_value(), -1.0);
}

TEST(DisaggPDSchedulerTest, StructuredOutputFieldsPreserveWireTags) {
  proto::DisaggRequest request;
  request.set_include_stop_str_in_output(true);
  request.set_json_object(true);
  request.set_json_reasoning_enabled(true);

  std::string serialized;
  ASSERT_TRUE(request.SerializeToString(&serialized));

  proto::DisaggRequest decoded;
  ASSERT_TRUE(decoded.ParseFromString(serialized));
  EXPECT_TRUE(decoded.include_stop_str_in_output());
  EXPECT_TRUE(decoded.json_object());
  EXPECT_TRUE(decoded.json_reasoning_enabled());
  EXPECT_EQ(proto::DisaggRequest::kIncludeStopStrInOutputFieldNumber, 39);
  EXPECT_EQ(proto::DisaggRequest::kJsonObjectFieldNumber, 40);
  EXPECT_EQ(proto::DisaggRequest::kJsonReasoningEnabledFieldNumber, 41);
}

TEST(DisaggPDSchedulerTest, GenerationLatencyFieldsPreserveWireTags) {
  proto::DisaggGenerationsRequest request;
  request.set_upstream_elapsed_seconds(12.5);
  proto::RemoteToken* token = request.add_tokens();
  token->set_time_to_first_token_latency_seconds(2.5);

  std::string serialized;
  ASSERT_TRUE(request.SerializeToString(&serialized));

  proto::DisaggGenerationsRequest decoded;
  ASSERT_TRUE(decoded.ParseFromString(serialized));
  ASSERT_TRUE(decoded.has_upstream_elapsed_seconds());
  EXPECT_DOUBLE_EQ(decoded.upstream_elapsed_seconds(), 12.5);
  ASSERT_EQ(decoded.tokens_size(), 1);
  EXPECT_DOUBLE_EQ(decoded.tokens(0).time_to_first_token_latency_seconds(),
                   2.5);
  EXPECT_EQ(proto::DisaggGenerationsRequest::kUpstreamElapsedSecondsFieldNumber,
            21);
  EXPECT_EQ(proto::RemoteToken::kTimeToFirstTokenLatencySecondsFieldNumber, 6);
}

TEST(DisaggPDSchedulerTest, LocalFailureReturnsDecodeReservation) {
  FakeEngine decode_engine(/*num_blocks=*/16,
                           /*block_size=*/2,
                           /*num_speculative_tokens=*/1);
  TestDisaggPDScheduler decode(&decode_engine, make_mtp_decode_options());
  auto remote = make_request({1, 2, 3, 4, 5, 6, 7, 8});
  auto* pool = decode_engine.block_manager_pool();
  const auto free_before = pool->num_free_blocks();
  ASSERT_EQ(decode.try_admit(remote), DecodeAdmissionResult::ADMITTED);
  remote->state().pd_reservation_id = "reservation";
  ASSERT_TRUE(decode.decode_schedule(remote, "prefill"));
  ASSERT_GE(remote->sequences()[0]->get_embedding_block_id(), 0);

  ReservationService service(&decode);
  brpc::Server server;
  ASSERT_EQ(server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE), 0);
  ASSERT_EQ(server.Start("127.0.0.1:0", nullptr), 0);

  FakeEngine prefill_engine(/*num_blocks=*/2, /*block_size=*/2);
  TestDisaggPDScheduler prefill(&prefill_engine, make_options());
  auto local = make_request({1, 2, 3, 4, 5, 6, 7, 8});
  TransferKVInfo info;
  info.request_id = local->request_id();
  info.remote_instance_info.rpc_address =
      "127.0.0.1:" + std::to_string(server.listen_address().port);
  local->state().pd_reservation_id = "reservation";
  local->state().decode_rpc_address = info.remote_instance_info.rpc_address;
  local->sequences()[0]->kv_state().set_transfer_kv_info(std::move(info));
  local->state().output_func = [](const RequestOutput&) { return true; };
  prefill.admit_prefill(local);
  auto batch = prefill.prepare_batch_test();
  EXPECT_TRUE(batch[0].empty());
  prefill.wait_notifications();

  EXPECT_FALSE(prefill.has_channel(local->request_id()));
  EXPECT_TRUE(decode.reservations_empty());
  EXPECT_EQ(service.calls, 1);
  EXPECT_EQ(pool->num_used_blocks()[0], 0u);
  EXPECT_EQ(pool->num_free_blocks(), free_before);
  EXPECT_EQ(remote->sequences()[0]->get_embedding_block_id(), -1);
  EXPECT_EQ(first_cache_size(*pool), 0u);
  server.Stop(0);
  server.Join();
}

TEST(DisaggPDSchedulerTest, CancelledPrefillReturnsDecodeReservation) {
  for (const bool running : {false, true}) {
    SCOPED_TRACE(running);
    FakeEngine decode_engine(/*num_blocks=*/16, /*block_size=*/2);
    TestDisaggPDScheduler decode(&decode_engine, make_decode_options());
    auto remote = make_request({1, 2, 3, 4, 5, 6, 7, 8});
    remote->state().pd_reservation_id = "reservation";
    const auto free_before =
        decode_engine.block_manager_pool()->num_free_blocks();
    ASSERT_EQ(decode.try_admit(remote), DecodeAdmissionResult::ADMITTED);
    ASSERT_TRUE(decode.decode_schedule(remote, "prefill"));
    ReservationService service(&decode);
    brpc::Server server;
    ASSERT_EQ(server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE), 0);
    ASSERT_EQ(server.Start("127.0.0.1:0", nullptr), 0);

    FakeEngine prefill_engine(/*num_blocks=*/16, /*block_size=*/2);
    TestDisaggPDScheduler prefill(&prefill_engine, make_options());
    auto local = make_request({1, 2, 3, 4, 5, 6, 7, 8});
    local->state().pd_reservation_id = "reservation";
    local->state().decode_rpc_address =
        "127.0.0.1:" + std::to_string(server.listen_address().port);
    local->state().output_func = [](const RequestOutput&) { return true; };
    prefill.admit_prefill(local);
    if (running) {
      auto batch = prefill.prepare_batch_test();
      ASSERT_FALSE(batch[0].empty());
    }
    local->set_cancel();
    auto batch = prefill.prepare_batch_test();
    EXPECT_TRUE(batch[0].empty());
    prefill.wait_notifications();
    EXPECT_EQ(service.calls, 1);
    EXPECT_TRUE(decode.reservations_empty());
    EXPECT_EQ(decode_engine.block_manager_pool()->num_free_blocks(),
              free_before);
    EXPECT_FALSE(prefill.has_channel(local->request_id()));
    server.Stop(0);
    server.Join();
  }
}
TEST(DisaggPDSchedulerTest, PendingReleaseDoesNotDelayFirstGeneration) {
  FakeEngine decode_engine(/*num_blocks=*/16, /*block_size=*/2);
  TestDisaggPDScheduler decode(&decode_engine, make_decode_options());
  auto remote_failed = make_request({1, 2, 3, 4, 5, 6, 7, 8}, "failed");
  remote_failed->state().pd_reservation_id = "failed-reservation";
  ASSERT_EQ(decode.try_admit(remote_failed), DecodeAdmissionResult::ADMITTED);
  ASSERT_TRUE(decode.decode_schedule(remote_failed, "prefill"));
  auto remote_healthy = make_request({11, 12}, "healthy");
  remote_healthy->state().stream = true;
  ASSERT_EQ(decode.try_admit(remote_healthy), DecodeAdmissionResult::ADMITTED);
  ASSERT_TRUE(decode.decode_schedule(remote_healthy, "prefill"));

  DelayedReservationService service(&decode);
  brpc::Server server;
  ASSERT_EQ(server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE), 0);
  ASSERT_EQ(server.Start("127.0.0.1:0", nullptr), 0);
  const std::string address =
      "127.0.0.1:" + std::to_string(server.listen_address().port);
  brpc::Channel channel;
  ASSERT_EQ(channel.Init(address.c_str(), nullptr), 0);
  proto::DisaggPDService_Stub stub(&channel);

  FakeEngine prefill_engine(/*num_blocks=*/4, /*block_size=*/2);
  auto options = make_options();
  options.enable_chunked_prefill(false).kv_cache_transfer_mode("PUSH");
  TestDisaggPDScheduler prefill(&prefill_engine, options);
  auto local_failed = make_request({1, 2, 3, 4, 5, 6, 7, 8}, "failed");
  local_failed->state().pd_reservation_id = "failed-reservation";
  local_failed->state().decode_rpc_address = address;
  local_failed->state().output_func = [](const RequestOutput&) { return true; };
  auto release_started = service.release_started();
  prefill.admit_prefill(local_failed);
  EXPECT_TRUE(prefill.prepare_batch_test()[0].empty());
  // Always resume the deferred server RPCs below, including on test failure.
  EXPECT_EQ(release_started.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);

  auto local_healthy = make_request({11, 12}, "healthy");
  local_healthy->state().stream = true;
  prefill.admit_prefill(local_healthy, &stub);
  auto batch = prefill.prepare_batch_test();
  EXPECT_FALSE(batch[0].empty());
  if (!batch[0].empty()) {
    finish_prefill(local_healthy->sequences()[0].get());
    // Supply the engine's PD output without enabling global PD configuration.
    local_healthy->sequences()[0]->first_token() = RemoteToken{.token_id = 999};
    prefill.prefill_send_first_generation();
  }
  auto prefill_done = prefill.prefill_completion();
  // Shorter than the release retry budget: handoff and local reclamation must
  // finish while the release RPC is still awaiting its server-side completion.
  const std::future_status status =
      prefill_done.wait_for(std::chrono::seconds(1));
  EXPECT_EQ(status, std::future_status::ready);
  std::shared_ptr<Request> queued;
  if (status == std::future_status::ready) {
    EXPECT_TRUE(decode.pop_decode_request_for_test(&queued));
    EXPECT_EQ(queued, remote_healthy);
    EXPECT_EQ(remote_healthy->sequences()[0]->num_generated_tokens(), 1u);
    EXPECT_EQ(prefill_engine.block_manager_pool()->num_used_blocks()[0], 1u);
    batch.clear();
    EXPECT_EQ(prefill_engine.block_manager_pool()->num_used_blocks()[0], 0u);
    EXPECT_FALSE(decode.reservations_empty());
  }

  service.resume_releases();
  prefill.wait_notifications();
  prefill_done.get();
  EXPECT_TRUE(decode.reservations_empty());
  if (queued == nullptr) {
    decode.pop_decode_request_for_test(&queued);
  }
  remote_healthy->sequences()[0]->reset();
  server.Stop(0);
  server.Join();
}

namespace {
class ReservationTest : public ::testing::Test {
 protected:
  FakeEngine engine_{/*num_blocks=*/16,
                     /*block_size=*/2,
                     /*num_speculative_tokens=*/1,
                     /*dp_size=*/2};
  TestDisaggPDScheduler decode_{&engine_,
                                make_mtp_decode_options(/*dp_size=*/2)};
  ReservationService service_{&decode_};
  brpc::Server server_;

  void SetUp() override {
    ASSERT_EQ(server_.AddService(&service_, brpc::SERVER_DOESNT_OWN_SERVICE),
              0);
    ASSERT_EQ(server_.Start("127.0.0.1:0", nullptr), 0);
  }

  void TearDown() override {
    server_.Stop(0);
    server_.Join();
  }

  std::shared_ptr<Request> reserve(
      const std::string& id = "req",
      const std::string& identity = "reservation") {
    auto request = make_request({1, 2, 3, 4, 5, 6, 7, 8}, id);
    request->state().pd_reservation_id = identity;
    CHECK(decode_.try_admit(request) == DecodeAdmissionResult::ADMITTED);
    CHECK(decode_.decode_schedule(request, "prefill"));
    return request;
  }

  std::shared_ptr<Request> accepted(const std::string& id = "req") {
    auto request = make_request({1, 2, 3, 4, 5, 6, 7, 8}, id);
    request->state().pd_reservation_id = "reservation";
    request->state().decode_rpc_address =
        "127.0.0.1:" + std::to_string(server_.listen_address().port);
    request->state().output_func = [](const RequestOutput&) { return true; };
    return request;
  }

  void fail_prefill() {
    FakeEngine engine(/*num_blocks=*/2, /*block_size=*/2);
    auto options = make_options();
    options.enable_chunked_prefill(false);
    TestDisaggPDScheduler prefill(&engine, options);
    auto request = accepted();
    std::promise<Status> error;
    auto status = error.get_future();
    request->state().output_func = [&error](const RequestOutput& output) {
      error.set_value(output.status.value());
      return true;
    };
    prefill.admit_prefill(request);
    EXPECT_TRUE(prefill.prepare_batch_test()[0].empty());
    prefill.wait_notifications();
    EXPECT_EQ(status.get().code(), StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_FALSE(prefill.has_channel(request->request_id()));
    EXPECT_TRUE(request->state().decode_rpc_address.empty());
    EXPECT_EQ(engine.block_manager_pool()->num_used_blocks()[0], 0u);
  }
};
}  // namespace

TEST_F(ReservationTest, UnlinkedDecodeReleasesAcceptedQueuedReservation) {
  auto remote = reserve();
  FakeEngine engine(/*num_blocks=*/16, /*block_size=*/2);
  auto options = make_options();
  options.enable_pd_ooc(false).disable_log_stats(true);
  TestDisaggPDScheduler prefill(&engine, options, /*start_services=*/false);
  auto local = accepted();
  local->state().decode_address = "decode-1";
  auto* pool = engine.block_manager_pool();
  ASSERT_TRUE(pool->allocate(local->sequences()[0].get(), /*num_tokens=*/4));
  ASSERT_GT(pool->num_used_blocks()[0], 0u);
  std::atomic<int32_t> failures{0};
  local->state().output_func = [&failures](const RequestOutput& output) {
    EXPECT_TRUE(output.status.has_value());
    if (output.status.has_value()) {
      EXPECT_EQ(output.status->code(), StatusCode::UNAVAILABLE);
    }
    ++failures;
    return true;
  };
  prefill.admit_prefill(local);
  ASSERT_TRUE(prefill.unlink_instance("decode-1",
                                      /*cluster_ids=*/{},
                                      /*addrs=*/{},
                                      /*ports=*/{},
                                      /*dp_size=*/1,
                                      /*src_kv_split_size=*/1));
  const auto batches = prefill.prepare_batch_test();
  EXPECT_TRUE(std::all_of(batches.begin(),
                          batches.end(),
                          [](const Batch& batch) { return batch.empty(); }));
  prefill.wait_notifications();
  prefill.wait_responses();
  EXPECT_EQ(failures.load(), 1);
  EXPECT_TRUE(local->has_terminal_outcome());
  EXPECT_EQ(service_.released, 1);
  EXPECT_TRUE(decode_.reservations_empty());
  EXPECT_FALSE(prefill.has_channel(local->request_id()));
  EXPECT_TRUE(prefill.get_running_requests().empty());
  EXPECT_EQ(pool->num_used_blocks()[0], 0u);
  EXPECT_EQ(remote->sequences()[0]->get_embedding_block_id(), -1);
}

TEST_F(ReservationTest, LostConfirmationRetriesWithoutDoubleFree) {
  auto request = reserve();
  BlockManager* embedding = request->sequences()[0]
                                ->kv_state()
                                .blocks(BlockType::EMBEDDING)[0]
                                .manager();
  ASSERT_EQ(embedding->num_used_blocks(), 1u);
  service_.failure = ReservationService::Failure::LOST_ACK;
  fail_prefill();
  EXPECT_EQ(service_.calls, 2);
  EXPECT_EQ(service_.released, 1);
  EXPECT_TRUE(decode_.reservations_empty());
  EXPECT_EQ(engine_.block_manager_pool()->num_used_blocks(),
            (std::vector<size_t>{0, 0}));
  EXPECT_EQ(embedding->num_used_blocks(), 0u);
  EXPECT_EQ(embedding->num_free_blocks(), 15u);
}

TEST_F(ReservationTest, TransportFailureHasBoundedRetries) {
  auto request = reserve();
  service_.failure = ReservationService::Failure::TRANSPORT;
  fail_prefill();
  EXPECT_EQ(service_.calls, 3);
  EXPECT_EQ(service_.released, 0);
  EXPECT_FALSE(decode_.reservations_empty());
  EXPECT_TRUE(decode_.release_reservation("req", "reservation"));
}

TEST_F(ReservationTest, UnsupportedRpcHasBoundedRetries) {
  auto request = reserve();
  service_.failure = ReservationService::Failure::UNSUPPORTED;
  fail_prefill();
  EXPECT_EQ(service_.calls, 3);
  EXPECT_EQ(service_.released, 0);
  EXPECT_FALSE(decode_.reservations_empty());
  EXPECT_TRUE(decode_.release_reservation("req", "reservation"));
}

TEST_F(ReservationTest, ReleaseNeverConsumesAnotherReservation) {
  auto first = reserve();
  auto other = reserve("other");
  decode_.report_queues();
  auto* reservations =
      MULTI_GAUGE_scheduler_queue_requests.get_stats({"pd_reservation", "0"});
  ASSERT_NE(reservations, nullptr);
  EXPECT_DOUBLE_EQ(reservations->get_value(), 2.0);
  EXPECT_FALSE(decode_.release_reservation("unknown", "reservation"));
  EXPECT_FALSE(decode_.release_reservation("req", "wrong"));
  EXPECT_FALSE(decode_.release_reservation("req", ""));
  EXPECT_TRUE(decode_.release_reservation("req", "reservation"));
  EXPECT_FALSE(decode_.release_reservation("req", "reservation"));
  decode_.report_queues();
  EXPECT_DOUBLE_EQ(reservations->get_value(), 1.0);
  auto replacement = reserve("req", "replacement");
  EXPECT_FALSE(decode_.release_reservation("req", "reservation"));
  EXPECT_GE(replacement->sequences()[0]->get_embedding_block_id(), 0);
  EXPECT_GE(other->sequences()[0]->get_embedding_block_id(), 0);
  EXPECT_TRUE(decode_.release_reservation("other", "reservation"));
  EXPECT_TRUE(decode_.release_reservation("req", "replacement"));
  EXPECT_TRUE(decode_.reservations_empty());
  decode_.report_queues();
  EXPECT_DOUBLE_EQ(reservations->get_value(), 0.0);
  EXPECT_EQ(engine_.block_manager_pool()->num_used_blocks(),
            (std::vector<size_t>{0, 0}));
}

TEST_F(ReservationTest, ReleaseBeforeLateAllocationPreventsReservation) {
  EXPECT_FALSE(decode_.release_reservation("late", "expired-reservation"));
  auto expired = make_request({1, 2, 3, 4, 5, 6, 7, 8}, "late");
  expired->state().pd_reservation_id = "expired-reservation";
  ASSERT_EQ(decode_.try_admit(expired), DecodeAdmissionResult::ADMITTED);
  EXPECT_FALSE(decode_.decode_schedule(expired, "prefill"));
  EXPECT_TRUE(decode_.reservations_empty());
  EXPECT_EQ(engine_.block_manager_pool()->num_used_blocks(),
            (std::vector<size_t>{0, 0}));

  reserve("late", "fresh-reservation");
  EXPECT_TRUE(decode_.release_reservation("late", "fresh-reservation"));
  auto replayed = make_request({1, 2, 3, 4, 5, 6, 7, 8}, "late");
  replayed->state().pd_reservation_id = "fresh-reservation";
  ASSERT_EQ(decode_.try_admit(replayed), DecodeAdmissionResult::ADMITTED);
  EXPECT_FALSE(decode_.decode_schedule(replayed, "prefill"));
  EXPECT_TRUE(decode_.reservations_empty());
  EXPECT_EQ(engine_.block_manager_pool()->num_used_blocks(),
            (std::vector<size_t>{0, 0}));
}

TEST_F(ReservationTest, RepeatedLateAllocationsRemainCancelled) {
  auto released = reserve(/*id=*/"late", /*identity=*/"released-reservation");
  const Request& released_request = *released;
  EXPECT_EQ(released_request.state().pd_reservation_id, "released-reservation");
  ASSERT_TRUE(
      decode_.release_reservation(released_request.request_id(),
                                  released_request.state().pd_reservation_id));

  for (int32_t attempt = 0; attempt < 2; ++attempt) {
    SCOPED_TRACE(attempt);
    auto replayed = make_request(/*prompt_token_ids=*/{1, 2, 3, 4, 5, 6, 7, 8},
                                 /*req_id=*/"late");
    replayed->state().pd_reservation_id =
        released_request.state().pd_reservation_id;
    ASSERT_EQ(decode_.try_admit(replayed), DecodeAdmissionResult::ADMITTED);
    EXPECT_FALSE(
        decode_.decode_schedule(replayed, /*prefill_instance_name=*/"prefill"));
    EXPECT_TRUE(decode_.reservations_empty());
    EXPECT_EQ(engine_.block_manager_pool()->num_used_blocks(),
              (std::vector<size_t>{0, 0}));
    EXPECT_EQ(first_cache_size(*engine_.block_manager_pool()), 0u);
  }

  reserve(/*id=*/"late", /*identity=*/"fresh-reservation");
  EXPECT_TRUE(
      decode_.release_reservation(/*req_id=*/"late",
                                  /*reservation_id=*/"fresh-reservation"));
}

TEST(ReservationOOC, ChannelFailureReleasesUncomputedBlocksWithoutCaching) {
  FakeEngine engine(/*num_blocks=*/16, /*block_size=*/2);
  set_metrics_model(engine);
  TestPDOOCScheduler decode(&engine, make_decode_options());
  decode.set_prefill_channel("prefill", nullptr);
  auto request = make_request({1, 2, 3, 4}, "channel-failure");
  ASSERT_TRUE(
      engine.block_manager_pool()->allocate(request->sequences()[0].get()));
  ASSERT_GT(engine.block_manager_pool()->num_used_blocks()[0], 0u);
  EXPECT_FALSE(decode.decode_schedule(request, "prefill"));
  EXPECT_EQ(engine.block_manager_pool()->num_used_blocks()[0], 0u);
  EXPECT_EQ(first_cache_size(*engine.block_manager_pool()), 0u);
}

TEST(ReservationOOC, ReleaseBeforeLateAllocationPreventsReservation) {
  FakeEngine engine(/*num_blocks=*/16, /*block_size=*/2);
  set_metrics_model(engine);
  TestPDOOCScheduler decode(&engine, make_decode_options());
  proto::DisaggPDService_Stub stub(nullptr);
  decode.set_prefill_channel("prefill", &stub);

  EXPECT_FALSE(decode.release_reservation("late", "expired-reservation"));
  auto expired = make_request({1, 2}, "late");
  expired->state().pd_reservation_id = "expired-reservation";
  ASSERT_TRUE(
      engine.block_manager_pool()->allocate(expired->sequences()[0].get()));
  EXPECT_FALSE(decode.decode_schedule(expired, "prefill"));
  EXPECT_EQ(engine.block_manager_pool()->num_used_blocks()[0], 0u);
  EXPECT_FALSE(decode.release_reservation("late", "expired-reservation"));

  auto fresh = make_request({1, 2}, "late");
  fresh->state().pd_reservation_id = "fresh-reservation";
  ASSERT_TRUE(
      engine.block_manager_pool()->allocate(fresh->sequences()[0].get()));
  ASSERT_TRUE(decode.decode_schedule(fresh, "prefill"));
  EXPECT_TRUE(decode.release_reservation("late", "fresh-reservation"));
  auto replayed = make_request({1, 2}, "late");
  replayed->state().pd_reservation_id = "fresh-reservation";
  ASSERT_TRUE(
      engine.block_manager_pool()->allocate(replayed->sequences()[0].get()));
  EXPECT_FALSE(decode.decode_schedule(replayed, "prefill"));
  EXPECT_EQ(engine.block_manager_pool()->num_used_blocks()[0], 0u);
}

TEST_F(ReservationTest, UnlinkClearsOnlyMatchingReservationMetrics) {
  auto first = reserve();
  auto other = make_request({11, 12}, "other");
  other->state().pd_reservation_id = "other_reservation";
  ASSERT_EQ(decode_.try_admit(other), DecodeAdmissionResult::ADMITTED);
  ASSERT_TRUE(decode_.decode_schedule(other, "other_prefill"));
  decode_.report_queues();
  auto* reservations =
      MULTI_GAUGE_scheduler_queue_requests.get_stats({"pd_reservation", "0"});
  ASSERT_NE(reservations, nullptr);
  EXPECT_DOUBLE_EQ(reservations->get_value(), 2.0);
  EXPECT_TRUE(decode_.unlink_instance("prefill", {}, {}, {}, 1, 1));
  decode_.report_queues();
  EXPECT_DOUBLE_EQ(reservations->get_value(), 1.0);
  EXPECT_FALSE(decode_.release_reservation("req", "reservation"));
  // An unlinked waiting request must not block retries while still alive.
  auto replacement = reserve(/*id=*/"req", /*identity=*/"replacement");
  EXPECT_GE(first->sequences()[0]->get_embedding_block_id(), 0);
  EXPECT_FALSE(decode_.release_reservation("req", "reservation"));
  EXPECT_GE(replacement->sequences()[0]->get_embedding_block_id(), 0);
  EXPECT_TRUE(decode_.release_reservation("req", "replacement"));
  engine_.block_manager_pool()->deallocate(first.get());
  EXPECT_TRUE(decode_.release_reservation("other", "other_reservation"));
  decode_.report_queues();
  EXPECT_DOUBLE_EQ(reservations->get_value(), 0.0);
}

TEST_F(ReservationTest, ReleasePreventsLateHandoff) {
  auto request = reserve();
  decode_.report_queues();
  auto* reservations =
      MULTI_GAUGE_scheduler_queue_requests.get_stats({"pd_reservation", "0"});
  ASSERT_NE(reservations, nullptr);
  EXPECT_DOUBLE_EQ(reservations->get_value(), 1.0);
  EXPECT_TRUE(decode_.release_reservation("req", "reservation"));
  decode_.report_queues();
  EXPECT_DOUBLE_EQ(reservations->get_value(), 0.0);
  EXPECT_FALSE(recv_first_generation(&decode_, torch::tensor({1.0f})));
  std::shared_ptr<Request> queued;
  EXPECT_FALSE(decode_.pop_decode_request_for_test(&queued));
  EXPECT_TRUE(decode_.reservations_empty());
}

TEST_F(ReservationTest, HandoffPreventsReleaseOfRunningResources) {
  auto request = reserve();
  EXPECT_TRUE(recv_first_generation(&decode_, torch::tensor({1.0f})));
  decode_.report_queues();
  auto* reservations =
      MULTI_GAUGE_scheduler_queue_requests.get_stats({"pd_reservation", "0"});
  auto* ingress =
      MULTI_GAUGE_scheduler_queue_requests.get_stats({"ingress", "0"});
  ASSERT_NE(reservations, nullptr);
  ASSERT_NE(ingress, nullptr);
  EXPECT_DOUBLE_EQ(reservations->get_value(), 0.0);
  EXPECT_DOUBLE_EQ(ingress->get_value(), 1.0);
  EXPECT_FALSE(decode_.release_reservation("req", "reservation"));
  EXPECT_GE(request->sequences()[0]->get_embedding_block_id(), 0);
  std::shared_ptr<Request> queued;
  ASSERT_TRUE(decode_.pop_decode_request_for_test(&queued));
  EXPECT_EQ(queued, request);
  EXPECT_TRUE(decode_.reservations_empty());
  queued->sequences()[0]->reset();
}

TEST(DisaggPDSchedulerTest, FullFirstGenerationQueueRollsBackOnlyTarget) {
  int32_t& queue_capacity = RecConfig::get_instance().request_queue_size();
  const int32_t previous_capacity = queue_capacity;
  ScopeGuard restore_capacity([&] { queue_capacity = previous_capacity; });
  queue_capacity = 1;

  FakeEngine engine(/*num_blocks=*/16,
                    /*block_size=*/2,
                    /*num_speculative_tokens=*/1);
  auto options = make_mtp_decode_options();
  options.enable_pd_ooc(false);
  TestDisaggPDScheduler decode(&engine, options, /*start_services=*/false);
  auto blocker = make_request({11, 12, 13, 14}, "blocker");
  ASSERT_EQ(decode.try_admit(blocker), DecodeAdmissionResult::ADMITTED);
  decode.admit_prefill(blocker);
  decode.report_queues();
  auto* ingress =
      MULTI_GAUGE_scheduler_queue_requests.get_stats({"ingress", "0"});
  ASSERT_NE(ingress, nullptr);
  ASSERT_DOUBLE_EQ(ingress->get_value(), 1.0);
  auto* pool = engine.block_manager_pool();
  const auto used_with_blocker = pool->num_used_blocks();
  const int32_t blocker_embedding =
      blocker->sequences()[0]->get_embedding_block_id();

  auto request = make_request({1, 2, 3, 4});
  request->state().pd_reservation_id = "reservation";
  ASSERT_EQ(decode.try_admit(request), DecodeAdmissionResult::ADMITTED);
  ASSERT_TRUE(decode.decode_schedule(request, "prefill"));
  EXPECT_FALSE(decode.reservations_empty());
  proto::CancelRequestRequest cancellation;
  cancellation.set_req_id(request->request_id());
  cancellation.set_service_req_id(request->service_request_id());
  cancellation.set_source_xservice_addr(request->source_xservice_addr());
  EXPECT_EQ(decode.cancel_request(cancellation),
            proto::CancelRequestResponse::NOT_READY);

  EXPECT_FALSE(recv_first_generation(&decode, torch::tensor({1.0f})));
  EXPECT_TRUE(decode.reservations_empty());
  EXPECT_EQ(decode.cancel_request(cancellation),
            proto::CancelRequestResponse::NOT_FOUND);
  EXPECT_EQ(request->sequences()[0]->get_embedding_block_id(), -1);
  EXPECT_EQ(request->sequences()[0]->kv_state().num_blocks(BlockType::KV), 0u);
  // Restored prompt KV may remain in the evictable prefix cache.
  EXPECT_EQ(pool->num_used_blocks(), used_with_blocker);
  EXPECT_EQ(blocker->sequences()[0]->get_embedding_block_id(),
            blocker_embedding);
  std::shared_ptr<Request> queued;
  ASSERT_TRUE(decode.pop_decode_request_for_test(&queued));
  EXPECT_EQ(queued, blocker);
  EXPECT_FALSE(decode.pop_decode_request_for_test(&queued));

  auto retry = make_request({1, 2, 3, 4});
  retry->state().pd_reservation_id = "reservation";
  ASSERT_EQ(decode.try_admit(retry), DecodeAdmissionResult::ADMITTED);
  ASSERT_TRUE(decode.decode_schedule(retry, "prefill"));
  EXPECT_TRUE(recv_first_generation(&decode, torch::tensor({1.0f})));
  EXPECT_TRUE(decode.reservations_empty());
  ASSERT_TRUE(decode.pop_decode_request_for_test(&queued));
  EXPECT_EQ(queued, retry);
  pool->deallocate(retry.get());
  pool->deallocate(blocker.get());
  EXPECT_EQ(pool->num_used_blocks(), (std::vector<size_t>{0}));
}

TEST(DisaggPDSchedulerTest, CancelAfterHandoffUsesSchedulerAndOneTerminal) {
  using Response = proto::CancelRequestResponse;
  FakeEngine engine(/*num_blocks=*/16,
                    /*block_size=*/2,
                    /*num_speculative_tokens=*/1);
  auto options = make_mtp_decode_options();
  options.enable_pd_ooc(false);
  TestDisaggPDScheduler decode(&engine, options, /*start_services=*/false);
  auto request = make_request({1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_EQ(decode.try_admit(request), DecodeAdmissionResult::ADMITTED);
  ASSERT_TRUE(decode.decode_schedule(request, "prefill"));
  std::atomic<int32_t> callbacks{0};
  request->state().output_func = [&callbacks](const RequestOutput& output) {
    EXPECT_EQ(output.status->code(), StatusCode::CANCELLED);
    ++callbacks;
    return true;
  };
  proto::CancelRequestRequest cancellation;
  EXPECT_EQ(decode.cancel_request(cancellation), Response::INVALID_ARGUMENT);
  cancellation.set_req_id(request->request_id());
  cancellation.set_service_req_id(request->service_request_id());
  cancellation.set_source_xservice_addr("wrong-service");
  EXPECT_EQ(decode.cancel_request(cancellation), Response::NOT_FOUND);
  cancellation.set_source_xservice_addr(request->source_xservice_addr());
  EXPECT_EQ(decode.cancel_request(cancellation), Response::NOT_READY);
  ASSERT_TRUE(recv_first_generation(&decode, torch::tensor({1.0f})));
  ASSERT_TRUE(decode.unlink_instance(/*instance_name=*/"prefill",
                                     /*cluster_ids=*/{},
                                     /*addrs=*/{},
                                     /*ports=*/{},
                                     /*dp_size=*/1,
                                     /*src_kv_split_size=*/1));
  EXPECT_FALSE(decode.release_reservation(/*req_id=*/"req",
                                          /*reservation_id=*/"old-attempt"));
  cancellation.set_service_req_id("wrong-request");
  EXPECT_EQ(decode.cancel_request(cancellation), Response::NOT_FOUND);
  cancellation.set_service_req_id(request->service_request_id());
  auto concurrent_cancellation = std::async(
      std::launch::async, [&] { return decode.cancel_request(cancellation); });
  EXPECT_EQ(decode.cancel_request(cancellation), Response::ACCEPTED);
  EXPECT_EQ(concurrent_cancellation.get(), Response::ACCEPTED);
  EXPECT_EQ(decode.cancel_request(cancellation), Response::ACCEPTED);
  auto cancellation_queue = decode.cancel_queue_for_test();
  auto queued_cancellations = cancellation_queue->take_all();
  ASSERT_EQ(queued_cancellations.size(), 1);
  EXPECT_EQ(queued_cancellations.front(), request);
  cancellation_queue->submit(std::move(queued_cancellations.front()));
  EXPECT_FALSE(request->cancelled());
  EXPECT_GE(request->sequences()[0]->get_embedding_block_id(), 0);
  decode.step(absl::ZeroDuration());
  decode.wait_responses();
  EXPECT_TRUE(request->cancelled());
  EXPECT_TRUE(request->has_terminal_outcome());
  EXPECT_EQ(callbacks.load(), 1);
  EXPECT_EQ(engine.block_manager_pool()->num_used_blocks(),
            (std::vector<size_t>{0}));
  EXPECT_EQ(decode.cancel_request(cancellation), Response::ALREADY_FINISHED);
}

TEST_F(ReservationTest, ConcurrentHandoffAndReleaseHaveOneOwner) {
  auto request = reserve();
  std::barrier start(2);
  auto release = std::async(std::launch::async, [&] {
    start.arrive_and_wait();
    return decode_.release_reservation("req", "reservation");
  });
  start.arrive_and_wait();
  const bool handed_off =
      recv_first_generation(&decode_, torch::tensor({1.0f}));
  EXPECT_NE(release.get(), handed_off);
  EXPECT_TRUE(decode_.reservations_empty());
  std::shared_ptr<Request> queued;
  EXPECT_EQ(decode_.pop_decode_request_for_test(&queued), handed_off);
  if (handed_off) {
    queued->sequences()[0]->reset();
  }
  EXPECT_EQ(engine_.block_manager_pool()->num_used_blocks(),
            (std::vector<size_t>{0, 0}));
}

TEST_F(ReservationTest, ReleasePreservesValidPrefixOnly) {
  auto* pool = engine_.block_manager_pool();
  auto seed = make_request({1, 2, 3, 4}, "seed");
  auto* seq = seed->sequences()[0].get();
  ASSERT_TRUE(pool->allocate(seq));
  const int32_t dp_rank = seq->dp_rank();
  finish_prefill(seq);
  decode_.cache_prefill_blocks_for_test(seed.get());
  pool->deallocate(seed.get());
  const auto cache_before = pool->num_blocks_in_prefix_cache();
  auto request = make_request({1, 2, 3, 4, 5, 6, 7, 8});
  request->state().pd_reservation_id = "reservation";
  request->sequences()[0]->set_dp_rank(dp_rank);
  ASSERT_EQ(decode_.try_admit(request), DecodeAdmissionResult::ADMITTED);
  ASSERT_EQ(
      request->sequences()[0]->kv_state().shared_blocks_num(BlockType::KV), 2u);
  ASSERT_TRUE(decode_.decode_schedule(request, "prefill"));
  EXPECT_TRUE(decode_.release_reservation("req", "reservation"));
  EXPECT_EQ(pool->num_blocks_in_prefix_cache(), cache_before);
  EXPECT_EQ(pool->num_used_blocks(), (std::vector<size_t>{0, 0}));
  auto probe = make_request({1, 2, 3, 4, 5, 6, 7, 8}, "probe");
  probe->sequences()[0]->set_dp_rank(dp_rank);
  pool->allocate_shared(probe->sequences()[0].get());
  EXPECT_EQ(probe->sequences()[0]->kv_state().shared_blocks_num(BlockType::KV),
            2u);
  probe->sequences()[0]->reset();
}

TEST_F(ReservationTest, RejectedAllocationRollsBackWithoutReservation) {
  auto* pool = engine_.block_manager_pool();
  const auto free_before = pool->num_free_blocks();
  auto rejected = make_request(std::vector<int32_t>(40, 1));
  EXPECT_NE(decode_.try_admit(rejected), DecodeAdmissionResult::ADMITTED);
  EXPECT_EQ(pool->num_free_blocks(), free_before);
  EXPECT_EQ(pool->num_used_blocks(), (std::vector<size_t>{0, 0}));
  EXPECT_EQ(rejected->sequences()[0]->get_embedding_block_id(), -1);
  EXPECT_TRUE(decode_.reservations_empty());
  EXPECT_FALSE(decode_.release_reservation("req", "reservation"));
}

TEST_F(ReservationTest, UnacceptedLocalFailureSendsNoRelease) {
  FakeEngine engine(/*num_blocks=*/2, /*block_size=*/2);
  TestDisaggPDScheduler prefill(&engine, make_options());
  auto local = accepted();
  local->state().decode_rpc_address.clear();
  prefill.admit_prefill(local);
  EXPECT_TRUE(prefill.prepare_batch_test()[0].empty());
  prefill.wait_notifications();
  EXPECT_EQ(service_.calls, 0);
  EXPECT_FALSE(prefill.has_channel(local->request_id()));
}

class ChunkReservationTest : public ReservationTest,
                             public ::testing::WithParamInterface<bool> {};

TEST_P(ChunkReservationTest,
       ChunkFailureReleasesAfterPushWithoutWaitingForOutput) {
  auto remote = reserve();
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  auto options = make_options();
  options.enable_schedule_overlap(GetParam())
      .enable_chunked_prefill(true)
      .max_tokens_per_chunk_for_prefill(2)
      .max_tokens_per_batch(2);
  TestDisaggPDScheduler prefill(&engine, options);
  auto local = accepted();
  prefill.admit_prefill(local);
  folly::Promise<std::vector<KVTransferTaskResult>> push;
  std::promise<void> started;
  auto entered = started.get_future();
  engine.forward = [&](std::vector<Batch>& batch) {
    KVTransferCompletion completion;
    completion.add(push.getSemiFuture(), {local->request_id()});
    started.set_value();
    CHECK(completion.wait().empty());
    batch[0][0]->kv_state().set_kv_cache_tokens_num(2);
    return ForwardOutput{};
  };
  auto step = std::async(std::launch::async,
                         [&] { prefill.step(absl::ZeroDuration()); });
  entered.get();
  EXPECT_EQ(service_.calls, 0);
  EXPECT_FALSE(decode_.reservations_empty());
  EXPECT_GE(remote->sequences()[0]->get_embedding_block_id(), 0);
  push.setValue(std::vector<KVTransferTaskResult>{});
  step.get();
  auto blocker = make_request(std::vector<int32_t>(12, 99), "blocker");
  ASSERT_TRUE(
      engine.block_manager_pool()->allocate(blocker->sequences()[0].get()));
  EXPECT_TRUE(prefill.prepare_batch_test()[0].empty());
  prefill.wait_notifications();
  EXPECT_EQ(service_.released, 1);
  EXPECT_TRUE(decode_.reservations_empty());
  if (GetParam()) {
    engine.complete_forward = [&](std::vector<Batch>&) {
      EXPECT_EQ(service_.released, 1);
    };
    prefill.step(absl::ZeroDuration());
    EXPECT_TRUE(prefill.prepare_batch_test()[0].empty());
  }
  prefill.wait_notifications();
  EXPECT_EQ(service_.released, 1);
  EXPECT_TRUE(decode_.reservations_empty());
  EXPECT_EQ(engine_.block_manager_pool()->num_used_blocks(),
            (std::vector<size_t>{0, 0}));
  blocker->sequences()[0]->reset();
}
INSTANTIATE_TEST_SUITE_P(OverlapModes, ChunkReservationTest, ::testing::Bool());

TEST_F(ReservationTest, BudgetFailureReleasesOnlyTarget) {
  auto remote = reserve();
  auto other = reserve("other");
  FakeEngine engine(/*num_blocks=*/16, /*block_size=*/2);
  auto options = make_options();
  options.enable_chunked_prefill(false).max_tokens_per_batch(2);
  TestDisaggPDScheduler prefill(&engine, options);
  auto local = accepted();
  prefill.admit_prefill(local);
  EXPECT_TRUE(prefill.prepare_batch_test()[0].empty());
  prefill.wait_notifications();
  EXPECT_EQ(service_.released, 1);
  EXPECT_EQ(remote->sequences()[0]->get_embedding_block_id(), -1);
  EXPECT_GE(other->sequences()[0]->get_embedding_block_id(), 0);
  EXPECT_TRUE(decode_.release_reservation("other", "reservation"));
  EXPECT_TRUE(decode_.reservations_empty());
}

TEST_F(ReservationTest, DuplicateAdmissionRollsBackUncomputedResources) {
  auto original = reserve();
  auto duplicate = make_request({11, 12, 13, 14});
  duplicate->state().pd_reservation_id = "duplicate";
  ASSERT_EQ(decode_.try_admit(duplicate), DecodeAdmissionResult::ADMITTED);
  BlockManager* embedding = duplicate->sequences()[0]
                                ->kv_state()
                                .blocks(BlockType::EMBEDDING)[0]
                                .manager();
  const size_t embedding_used = embedding->num_used_blocks();
  EXPECT_FALSE(decode_.decode_schedule(duplicate, "prefill"));
  decode_.report_queues();
  auto* reservations =
      MULTI_GAUGE_scheduler_queue_requests.get_stats({"pd_reservation", "0"});
  ASSERT_NE(reservations, nullptr);
  EXPECT_DOUBLE_EQ(reservations->get_value(), 1.0);
  EXPECT_EQ(duplicate->sequences()[0]->get_embedding_block_id(), -1);
  EXPECT_EQ(embedding->num_used_blocks(), embedding_used - 1);
  EXPECT_EQ(engine_.block_manager_pool()->num_blocks_in_prefix_cache(),
            (std::vector<size_t>{0, 0}));
  EXPECT_GE(original->sequences()[0]->get_embedding_block_id(), 0);
  EXPECT_TRUE(decode_.release_reservation("req", "reservation"));

  // Handoff removes the waiting entry but must still reject the active ID.
  auto handed_off = reserve(/*id=*/"req", /*identity=*/"handed-off");
  ASSERT_TRUE(recv_first_generation(&decode_, torch::tensor({1.0f})));
  EXPECT_TRUE(decode_.reservations_empty());
  auto handed_off_duplicate = make_request({11, 12, 13, 14});
  ASSERT_EQ(decode_.try_admit(handed_off_duplicate),
            DecodeAdmissionResult::ADMITTED);
  EXPECT_FALSE(decode_.decode_schedule(handed_off_duplicate, "prefill"));
  EXPECT_GE(handed_off->sequences()[0]->get_embedding_block_id(), 0);
  EXPECT_FALSE(decode_.release_reservation("req", "reservation"));

  // An expired cancellation entry must not reject a new request with that ID.
  std::shared_ptr<Request> queued;
  ASSERT_TRUE(decode_.pop_decode_request_for_test(&queued));
  EXPECT_EQ(queued, handed_off);
  engine_.block_manager_pool()->deallocate(queued.get());
  queued.reset();
  handed_off.reset();
  auto retry = reserve(/*id=*/"req", /*identity=*/"retry");
  EXPECT_GE(retry->sequences()[0]->get_embedding_block_id(), 0);
  EXPECT_TRUE(decode_.release_reservation("req", "retry"));
}
TEST(DisaggPDSchedulerTest, MtpExhaustionRollsBackKvReservation) {
  FakeEngine engine(/*num_blocks=*/8,
                    /*block_size=*/2,
                    /*num_speculative_tokens=*/1,
                    /*dp_size=*/1,
                    /*embedding_blocks=*/2);
  TestDisaggPDScheduler decode(&engine, make_mtp_decode_options());
  auto original = make_request({1, 2, 3, 4});
  original->state().pd_reservation_id = "reservation";
  ASSERT_EQ(decode.try_admit(original), DecodeAdmissionResult::ADMITTED);
  ASSERT_TRUE(decode.decode_schedule(original, "prefill"));
  auto* pool = engine.block_manager_pool();
  const auto free_before = pool->num_free_blocks();
  auto rejected = make_request({11, 12, 13, 14}, "rejected");
  EXPECT_NE(decode.try_admit(rejected), DecodeAdmissionResult::ADMITTED);
  EXPECT_EQ(rejected->sequences()[0]->get_embedding_block_id(), -1);
  EXPECT_EQ(pool->num_free_blocks(), free_before);
  EXPECT_EQ(pool->num_used_blocks()[0], 2u);
  EXPECT_EQ(first_cache_size(*pool), 0u);
  EXPECT_FALSE(decode.release_reservation("rejected", "reservation"));
  EXPECT_TRUE(decode.release_reservation("req", "reservation"));
  EXPECT_TRUE(decode.reservations_empty());
  EXPECT_EQ(pool->num_used_blocks()[0], 0u);
}

TEST(DisaggPDSchedulerTest, PrefillResponseCompletesBeforeDecodeHandoff) {
  FirstGenerationOrderService service;
  auto received = service.received();
  brpc::Server server;
  ASSERT_EQ(server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE), 0);
  ASSERT_EQ(server.Start("127.0.0.1:0", nullptr), 0);
  const std::string address =
      "127.0.0.1:" + std::to_string(server.listen_address().port);
  brpc::Channel channel;
  ASSERT_EQ(channel.Init(address.c_str(), nullptr), 0);
  proto::DisaggPDService_Stub stub(&channel);

  FakeEngine engine(/*num_blocks=*/16, /*block_size=*/2);
  auto options = make_options();
  options.enable_chunked_prefill(false).kv_cache_transfer_mode("PUSH");
  TestDisaggPDScheduler prefill(&engine, options, /*start_services=*/false);
  auto request = make_request({11, 12}, "first-token-order");
  request->state().stream = true;
  std::atomic<bool> first_token_observed{false};
  std::promise<void> response_started;
  auto started = response_started.get_future();
  std::promise<void> release_response;
  auto released = release_response.get_future().share();
  request->state().output_func = [&response_started,
                                  &first_token_observed,
                                  released](const RequestOutput& output) {
    first_token_observed =
        output.outputs.size() == 1 && output.outputs[0].text == "999";
    response_started.set_value();
    released.wait();
    return true;
  };
  prefill.admit_prefill(request, &stub);
  auto batch = prefill.prepare_batch_test();
  ASSERT_FALSE(batch[0].empty());
  finish_prefill(request->sequences()[0].get());
  request->sequences()[0]->first_token() = RemoteToken{.token_id = 999};
  prefill.emit_stream_for_test(request);
  const bool output_started =
      started.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
  EXPECT_TRUE(output_started);
  EXPECT_TRUE(first_token_observed.load());
  if (output_started) {
    prefill.prefill_send_first_generation();
    const std::future_status before_response_completed =
        received.wait_for(std::chrono::milliseconds(250));
    EXPECT_EQ(before_response_completed, std::future_status::timeout)
        << "Decode received FirstGeneration while the Prefill response "
           "callback "
           "was still blocked; its later output can overtake the first token.";
  }
  release_response.set_value();
  if (output_started) {
    EXPECT_EQ(received.wait_for(std::chrono::seconds(5)),
              std::future_status::ready);
    prefill.prefill_completion().get();
  }
  server.Stop(0);
  server.Join();
}
}  // namespace xllm
