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

#include "scheduler/output_delivery_queue.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace xllm {
namespace {

using namespace std::chrono_literals;

RequestOutput make_output(uint64_t sequence, std::string text = "event") {
  RequestOutput output;
  output.request_id = "engine-request";
  output.service_request_id = "service-request";
  output.target_xservice_addr = "127.0.0.1:28888";
  output.output_event_seq = sequence;
  SequenceOutput sequence_output;
  sequence_output.index = 0;
  sequence_output.text = std::move(text);
  output.outputs.emplace_back(std::move(sequence_output));
  return output;
}

OutputDeliveryQueue::Config make_config() {
  return OutputDeliveryQueue::Config{
      .worker_threads = 2,
      .max_pending_requests = 16,
      .max_events_per_request = 8,
      .max_bytes_per_request = 4096,
      .subscriber_unreachable_abort = 40ms,
      .retry_interval = 2ms,
      .delivery_timeout = 5ms,
  };
}

TEST(OutputDeliveryQueueTest, DeliversEachRequestInFifoOrder) {
  std::mutex mutex;
  std::vector<uint64_t> delivered;
  std::atomic<int32_t> aborts{0};
  OutputDeliveryQueue queue(make_config());
  auto deliver = [&](const RequestOutput& output) {
    std::lock_guard<std::mutex> lock(mutex);
    delivered.emplace_back(*output.output_event_seq);
    return OutputDeliveryStatus::DELIVERED;
  };
  auto abort = [&]() { aborts.fetch_add(1); };

  for (uint64_t sequence = 0; sequence < 5; ++sequence) {
    ASSERT_TRUE(
        queue.enqueue("request", make_output(sequence), deliver, abort));
  }
  queue.wait_empty();

  EXPECT_EQ(delivered, (std::vector<uint64_t>{0, 1, 2, 3, 4}));
  EXPECT_EQ(aborts.load(), 0);
  EXPECT_EQ(queue.pending_events(), 0u);
}

TEST(OutputDeliveryQueueTest, WaitRequestIgnoresUnrelatedTraffic) {
  OutputDeliveryQueue::Config config = make_config();
  config.subscriber_unreachable_abort = 2s;
  std::atomic<bool> release_background{false};
  std::atomic<int32_t> background_attempts{0};
  std::atomic<int32_t> target_deliveries{0};
  std::mutex background_mutex;
  std::condition_variable background_cv;
  OutputDeliveryQueue queue(config);

  ASSERT_TRUE(queue.enqueue(
      "background",
      make_output(0),
      [&](const RequestOutput&) {
        ++background_attempts;
        background_cv.notify_all();
        return release_background.load(std::memory_order_acquire)
                   ? OutputDeliveryStatus::DELIVERED
                   : OutputDeliveryStatus::SUBSCRIBER_UNREACHABLE;
      },
      []() {}));
  {
    std::unique_lock<std::mutex> lock(background_mutex);
    ASSERT_TRUE(background_cv.wait_for(lock, 1s, [&]() {
      return background_attempts.load(std::memory_order_acquire) > 0;
    }));
  }
  RequestOutput terminal = make_output(0);
  terminal.finished = true;
  ASSERT_TRUE(queue.enqueue_terminal(
      "target",
      std::move(terminal),
      [&](const RequestOutput&) {
        ++target_deliveries;
        return OutputDeliveryStatus::DELIVERED;
      },
      []() {}));

  queue.wait_request("target");
  EXPECT_EQ(target_deliveries.load(), 1);
  EXPECT_GT(background_attempts.load(), 0);
  EXPECT_GE(queue.pending_requests(), 1u);

  release_background.store(true, std::memory_order_release);
  queue.wait_empty();
}

TEST(OutputDeliveryQueueTest, WaitRequestIncludesRegisteredProducer) {
  OutputDeliveryQueue queue(make_config());
  queue.begin_request_production("target");
  std::atomic<bool> wait_finished{false};
  std::thread waiter([&]() {
    queue.wait_request("target");
    wait_finished.store(true, std::memory_order_release);
  });

  std::this_thread::sleep_for(10ms);
  EXPECT_FALSE(wait_finished.load(std::memory_order_acquire));
  RequestOutput terminal = make_output(0);
  terminal.finished = true;
  ASSERT_TRUE(queue.enqueue_terminal(
      "target",
      std::move(terminal),
      [](const RequestOutput&) { return OutputDeliveryStatus::DELIVERED; },
      []() {}));
  queue.finish_request_production("target");
  waiter.join();

  EXPECT_TRUE(wait_finished.load(std::memory_order_acquire));
  EXPECT_EQ(queue.pending_events(), 0u);
}

TEST(OutputDeliveryQueueTest, WaitRequestReturnsForUnknownRequest) {
  OutputDeliveryQueue queue(make_config());
  const auto started_at = std::chrono::steady_clock::now();

  queue.wait_request("not-enqueued");

  EXPECT_LT(std::chrono::steady_clock::now() - started_at, 100ms);
}

TEST(OutputDeliveryStateTest, ConcurrentTransitionsHaveExactlyOneWinner) {
  OutputDeliveryState state;
  constexpr size_t kThreads = 64;
  std::atomic<size_t> abort_winners{0};
  std::atomic<size_t> finalize_winners{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (size_t index = 0; index < kThreads; ++index) {
    threads.emplace_back([&]() {
      if (state.mark_aborted()) {
        ++abort_winners;
      }
      if (state.mark_finalized()) {
        ++finalize_winners;
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(state.aborted());
  EXPECT_EQ(abort_winners.load(), 1u);
  EXPECT_EQ(finalize_winners.load(), 1u);
}

TEST(OutputDeliveryQueueTest, BlockedSubscriberDoesNotBlockAnotherRequest) {
  std::mutex mutex;
  std::condition_variable cv;
  bool release_blocked = false;
  bool other_delivered = false;
  OutputDeliveryQueue queue(make_config());
  auto blocked_delivery = [&](const RequestOutput&) {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait_for(lock, 1s, [&]() { return release_blocked; });
    return OutputDeliveryStatus::DELIVERED;
  };
  auto other_delivery = [&](const RequestOutput&) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      other_delivered = true;
    }
    cv.notify_all();
    return OutputDeliveryStatus::DELIVERED;
  };

  ASSERT_TRUE(
      queue.enqueue("blocked", make_output(0), blocked_delivery, [] {}));
  ASSERT_TRUE(queue.enqueue("other", make_output(0), other_delivery, [] {}));
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, 1s, [&]() { return other_delivered; }));
    release_blocked = true;
  }
  cv.notify_all();
  queue.wait_empty();
}

TEST(OutputDeliveryQueueTest, RetriesTransientUnreachableWithoutReordering) {
  std::atomic<int32_t> attempts{0};
  std::atomic<int32_t> aborts{0};
  std::vector<uint64_t> delivered;
  std::mutex mutex;
  OutputDeliveryQueue queue(make_config());
  auto deliver = [&](const RequestOutput& output) {
    if (attempts.fetch_add(1) < 2) {
      return OutputDeliveryStatus::SUBSCRIBER_UNREACHABLE;
    }
    std::lock_guard<std::mutex> lock(mutex);
    delivered.emplace_back(*output.output_event_seq);
    return OutputDeliveryStatus::DELIVERED;
  };

  ASSERT_TRUE(
      queue.enqueue("request", make_output(0), deliver, [&]() { ++aborts; }));
  ASSERT_TRUE(
      queue.enqueue("request", make_output(1), deliver, [&]() { ++aborts; }));
  queue.wait_empty();

  EXPECT_EQ(attempts.load(), 4);
  EXPECT_EQ(delivered, (std::vector<uint64_t>{0, 1}));
  EXPECT_EQ(aborts.load(), 0);
}

TEST(OutputDeliveryQueueTest, SustainedOutageWithinConfiguredWindowRecovers) {
  OutputDeliveryQueue::Config config = make_config();
  config.subscriber_unreachable_abort = 150ms;
  config.retry_interval = 2ms;
  const auto started = std::chrono::steady_clock::now();
  std::atomic<int32_t> attempts{0};
  std::atomic<int32_t> aborts{0};
  OutputDeliveryQueue queue(config);

  ASSERT_TRUE(queue.enqueue(
      "request",
      make_output(0),
      [&](const RequestOutput&) {
        ++attempts;
        if (std::chrono::steady_clock::now() - started < 75ms) {
          return OutputDeliveryStatus::SUBSCRIBER_UNREACHABLE;
        }
        return OutputDeliveryStatus::DELIVERED;
      },
      [&]() { ++aborts; }));
  queue.wait_empty();

  EXPECT_GE(std::chrono::steady_clock::now() - started, 75ms);
  EXPECT_GE(attempts.load(), 2);
  EXPECT_EQ(aborts.load(), 0);
}

TEST(OutputDeliveryQueueTest, AbortsAfterContinuousUnreachable) {
  OutputDeliveryQueue::Config config = make_config();
  config.subscriber_unreachable_abort = 15ms;
  std::atomic<int32_t> attempts{0};
  std::atomic<int32_t> aborts{0};
  OutputDeliveryQueue queue(config);

  ASSERT_TRUE(queue.enqueue(
      "request",
      make_output(0),
      [&](const RequestOutput&) {
        ++attempts;
        return OutputDeliveryStatus::SUBSCRIBER_UNREACHABLE;
      },
      [&]() { ++aborts; }));
  queue.wait_empty();

  EXPECT_GE(attempts.load(), 2);
  EXPECT_EQ(aborts.load(), 1);
}

TEST(OutputDeliveryQueueTest, SubscriberRejectIsTerminalWithoutRetry) {
  std::atomic<int32_t> attempts{0};
  std::atomic<int32_t> aborts{0};
  OutputDeliveryQueue queue(make_config());

  ASSERT_TRUE(queue.enqueue(
      "request",
      make_output(0),
      [&](const RequestOutput&) {
        ++attempts;
        return OutputDeliveryStatus::SUBSCRIBER_REJECTED;
      },
      [&]() { ++aborts; }));
  queue.wait_empty();

  EXPECT_EQ(attempts.load(), 1);
  EXPECT_EQ(aborts.load(), 1);
}

TEST(OutputDeliveryQueueTest, CapacityFailureAbortsOnlyAffectedRequest) {
  OutputDeliveryQueue::Config config = make_config();
  config.worker_threads = 1;
  config.max_events_per_request = 1;
  std::mutex mutex;
  std::condition_variable cv;
  bool started = false;
  bool release = false;
  std::atomic<int32_t> aborts{0};
  OutputDeliveryQueue queue(config);
  auto blocked = [&](const RequestOutput&) {
    std::unique_lock<std::mutex> lock(mutex);
    started = true;
    cv.notify_all();
    cv.wait_for(lock, 1s, [&]() { return release; });
    return OutputDeliveryStatus::DELIVERED;
  };

  ASSERT_TRUE(
      queue.enqueue("request", make_output(0), blocked, [&]() { ++aborts; }));
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, 1s, [&]() { return started; }));
  }
  EXPECT_FALSE(
      queue.enqueue("request", make_output(1), blocked, [&]() { ++aborts; }));
  EXPECT_EQ(aborts.load(), 1);
  {
    std::lock_guard<std::mutex> lock(mutex);
    release = true;
  }
  cv.notify_all();
  queue.wait_empty();
}

TEST(OutputDeliveryQueueTest,
     CapacityAbortDeliversTerminalAfterInFlightOutputExactlyOnce) {
  OutputDeliveryQueue::Config config = make_config();
  config.worker_threads = 1;
  config.max_events_per_request = 1;
  std::mutex mutex;
  std::condition_variable cv;
  bool delivery_started = false;
  bool release_delivery = false;
  std::vector<uint64_t> delivered;
  std::atomic<int32_t> aborts{0};
  OutputDeliveryQueue queue(config);
  auto deliver = [&](const RequestOutput& output) {
    const uint64_t sequence = *output.output_event_seq;
    std::unique_lock<std::mutex> lock(mutex);
    if (sequence == 0) {
      delivery_started = true;
      cv.notify_all();
      cv.wait_for(lock, 1s, [&]() { return release_delivery; });
    }
    delivered.emplace_back(sequence);
    return OutputDeliveryStatus::DELIVERED;
  };
  auto abort = [&]() { ++aborts; };

  ASSERT_TRUE(queue.enqueue("request", make_output(0), deliver, abort));
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, 1s, [&]() { return delivery_started; }));
  }
  EXPECT_FALSE(queue.enqueue("request", make_output(1), deliver, abort));
  RequestOutput terminal = make_output(2);
  terminal.cancelled = true;
  const bool terminal_accepted =
      queue.enqueue_terminal("request", std::move(terminal), deliver, abort);
  EXPECT_TRUE(terminal_accepted);
  RequestOutput duplicate_terminal = make_output(3);
  duplicate_terminal.cancelled = true;
  EXPECT_FALSE(queue.enqueue_terminal(
      "request", std::move(duplicate_terminal), deliver, abort));
  EXPECT_FALSE(queue.enqueue("request", make_output(4), deliver, abort));
  {
    std::lock_guard<std::mutex> lock(mutex);
    release_delivery = true;
  }
  cv.notify_all();
  queue.wait_empty();

  EXPECT_EQ(delivered, (std::vector<uint64_t>{0, 2}));
  EXPECT_EQ(aborts.load(), 1);
  EXPECT_EQ(queue.pending_requests(), 0u);
  EXPECT_EQ(queue.pending_events(), 0u);
}

TEST(OutputDeliveryQueueTest,
     CapacityAbortCanReceiveTerminalWhileAbortCallbackIsRunning) {
  OutputDeliveryQueue::Config config = make_config();
  config.worker_threads = 1;
  config.max_events_per_request = 1;
  std::mutex mutex;
  std::condition_variable cv;
  bool delivery_started = false;
  bool release_delivery = false;
  bool abort_started = false;
  bool release_abort = false;
  bool terminal_delivered = false;
  std::atomic<bool> overflow_rejected{false};
  std::atomic<int32_t> aborts{0};
  OutputDeliveryQueue queue(config);
  auto deliver = [&](const RequestOutput& output) {
    std::unique_lock<std::mutex> lock(mutex);
    if (*output.output_event_seq == 0) {
      delivery_started = true;
      cv.notify_all();
      cv.wait_for(lock, 1s, [&]() { return release_delivery; });
    } else {
      terminal_delivered = true;
      cv.notify_all();
    }
    return OutputDeliveryStatus::DELIVERED;
  };
  auto abort = [&]() {
    ++aborts;
    std::unique_lock<std::mutex> lock(mutex);
    abort_started = true;
    cv.notify_all();
    cv.wait_for(lock, 1s, [&]() { return release_abort; });
  };

  ASSERT_TRUE(queue.enqueue("request", make_output(0), deliver, abort));
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, 1s, [&]() { return delivery_started; }));
  }
  std::thread overflow([&]() {
    overflow_rejected.store(
        !queue.enqueue("request", make_output(1), deliver, abort),
        std::memory_order_release);
  });
  bool observed_abort = false;
  {
    std::unique_lock<std::mutex> lock(mutex);
    observed_abort = cv.wait_for(lock, 1s, [&]() { return abort_started; });
  }
  bool terminal_accepted = false;
  if (observed_abort) {
    RequestOutput terminal = make_output(2);
    terminal.cancelled = true;
    terminal_accepted =
        queue.enqueue_terminal("request", std::move(terminal), deliver, abort);
  }
  {
    std::lock_guard<std::mutex> lock(mutex);
    release_delivery = true;
  }
  cv.notify_all();
  bool observed_terminal = false;
  if (terminal_accepted) {
    std::unique_lock<std::mutex> lock(mutex);
    observed_terminal =
        cv.wait_for(lock, 1s, [&]() { return terminal_delivered; });
  }
  {
    std::lock_guard<std::mutex> lock(mutex);
    release_abort = true;
  }
  cv.notify_all();
  overflow.join();
  queue.wait_empty();

  EXPECT_TRUE(observed_abort);
  EXPECT_TRUE(terminal_accepted);
  EXPECT_TRUE(observed_terminal);
  EXPECT_TRUE(overflow_rejected.load(std::memory_order_acquire));
  EXPECT_EQ(aborts.load(), 1);
  EXPECT_EQ(queue.pending_requests(), 0u);
  EXPECT_EQ(queue.pending_events(), 0u);
}

TEST(OutputDeliveryQueueTest, InitialCapacityRejectCanStillDeliverTerminal) {
  OutputDeliveryQueue::Config config = make_config();
  // Reject the data-bearing output while leaving enough room for the compact
  // terminal identity that closes the request lifecycle.
  config.max_bytes_per_request = 128;
  std::atomic<int32_t> deliveries{0};
  std::atomic<int32_t> aborts{0};
  OutputDeliveryQueue queue(config);
  auto deliver = [&](const RequestOutput&) {
    ++deliveries;
    return OutputDeliveryStatus::DELIVERED;
  };
  auto abort = [&]() { ++aborts; };

  EXPECT_FALSE(queue.enqueue("request", make_output(0), deliver, abort));
  RequestOutput terminal = make_output(1);
  terminal.cancelled = true;
  EXPECT_TRUE(
      queue.enqueue_terminal("request", std::move(terminal), deliver, abort));
  queue.wait_empty();

  EXPECT_EQ(deliveries.load(), 1);
  EXPECT_EQ(aborts.load(), 1);
  EXPECT_EQ(queue.pending_requests(), 0u);
  EXPECT_EQ(queue.pending_events(), 0u);
}

TEST(OutputDeliveryQueueTest, TerminalRetriesThenAbortsAndDrains) {
  OutputDeliveryQueue::Config config = make_config();
  config.subscriber_unreachable_abort = 15ms;
  std::atomic<int32_t> attempts{0};
  std::atomic<int32_t> aborts{0};
  OutputDeliveryQueue queue(config);
  RequestOutput terminal = make_output(0);
  terminal.cancelled = true;

  ASSERT_TRUE(queue.enqueue_terminal(
      "request",
      std::move(terminal),
      [&](const RequestOutput&) {
        ++attempts;
        return OutputDeliveryStatus::SUBSCRIBER_UNREACHABLE;
      },
      [&]() { ++aborts; }));
  queue.wait_empty();

  EXPECT_GE(attempts.load(), 2);
  EXPECT_EQ(aborts.load(), 1);
  EXPECT_EQ(queue.pending_requests(), 0u);
  EXPECT_EQ(queue.pending_events(), 0u);
}

TEST(OutputDeliveryQueueTest,
     CapacityAbortTerminalTimeoutDoesNotInvokeAbortTwice) {
  OutputDeliveryQueue::Config config = make_config();
  config.worker_threads = 1;
  config.max_events_per_request = 1;
  config.subscriber_unreachable_abort = 15ms;
  std::mutex mutex;
  std::condition_variable cv;
  bool delivery_started = false;
  bool release_delivery = false;
  std::atomic<int32_t> terminal_attempts{0};
  std::atomic<int32_t> aborts{0};
  OutputDeliveryQueue queue(config);
  auto deliver = [&](const RequestOutput& output) {
    if (*output.output_event_seq == 0) {
      std::unique_lock<std::mutex> lock(mutex);
      delivery_started = true;
      cv.notify_all();
      cv.wait_for(lock, 1s, [&]() { return release_delivery; });
      return OutputDeliveryStatus::DELIVERED;
    }
    ++terminal_attempts;
    return OutputDeliveryStatus::SUBSCRIBER_UNREACHABLE;
  };
  auto abort = [&]() { ++aborts; };

  ASSERT_TRUE(queue.enqueue("request", make_output(0), deliver, abort));
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, 1s, [&]() { return delivery_started; }));
  }
  EXPECT_FALSE(queue.enqueue("request", make_output(1), deliver, abort));
  RequestOutput terminal = make_output(2);
  terminal.cancelled = true;
  const bool terminal_accepted =
      queue.enqueue_terminal("request", std::move(terminal), deliver, abort);
  {
    std::lock_guard<std::mutex> lock(mutex);
    release_delivery = true;
  }
  cv.notify_all();
  queue.wait_empty();

  EXPECT_TRUE(terminal_accepted);
  EXPECT_GE(terminal_attempts.load(), 2);
  EXPECT_EQ(aborts.load(), 1);
  EXPECT_EQ(queue.pending_requests(), 0u);
  EXPECT_EQ(queue.pending_events(), 0u);
}

TEST(OutputDeliveryQueueTest, AcceptedDataPrecedesTerminal) {
  OutputDeliveryQueue::Config config = make_config();
  config.worker_threads = 1;
  std::mutex mutex;
  std::condition_variable cv;
  bool first_started = false;
  bool release_first = false;
  std::vector<uint64_t> delivered;
  OutputDeliveryQueue queue(config);
  auto deliver = [&](const RequestOutput& output) {
    const uint64_t sequence = *output.output_event_seq;
    std::unique_lock<std::mutex> lock(mutex);
    if (sequence == 0) {
      first_started = true;
      cv.notify_all();
      cv.wait_for(lock, 1s, [&]() { return release_first; });
    }
    delivered.emplace_back(sequence);
    return OutputDeliveryStatus::DELIVERED;
  };

  ASSERT_TRUE(queue.enqueue("request", make_output(0), deliver, [] {}));
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, 1s, [&]() { return first_started; }));
  }
  const bool data_accepted =
      queue.enqueue("request", make_output(1), deliver, [] {});
  RequestOutput terminal = make_output(2);
  terminal.finished = true;
  const bool terminal_accepted =
      queue.enqueue_terminal("request", std::move(terminal), deliver, [] {});
  {
    std::lock_guard<std::mutex> lock(mutex);
    release_first = true;
  }
  cv.notify_all();
  queue.wait_empty();

  EXPECT_TRUE(data_accepted);
  EXPECT_TRUE(terminal_accepted);
  EXPECT_EQ(delivered, (std::vector<uint64_t>{0, 1, 2}));
  EXPECT_EQ(queue.pending_requests(), 0u);
  EXPECT_EQ(queue.pending_events(), 0u);
}

TEST(OutputDeliveryQueueTest, AbortingRequestCannotRecreateQueue) {
  OutputDeliveryQueue::Config config = make_config();
  config.worker_threads = 1;
  config.max_events_per_request = 1;
  std::mutex mutex;
  std::condition_variable cv;
  bool delivery_started = false;
  bool release_delivery = false;
  bool abort_started = false;
  bool release_abort = false;
  std::atomic<bool> overflow_rejected{false};
  std::atomic<int32_t> deliveries{0};
  std::atomic<int32_t> aborts{0};
  OutputDeliveryQueue queue(config);
  auto blocked_delivery = [&](const RequestOutput&) {
    ++deliveries;
    std::unique_lock<std::mutex> lock(mutex);
    delivery_started = true;
    cv.notify_all();
    cv.wait_for(lock, 1s, [&]() { return release_delivery; });
    return OutputDeliveryStatus::DELIVERED;
  };
  auto blocked_abort = [&]() {
    ++aborts;
    std::unique_lock<std::mutex> lock(mutex);
    abort_started = true;
    cv.notify_all();
    cv.wait_for(lock, 1s, [&]() { return release_abort; });
  };

  ASSERT_TRUE(queue.enqueue(
      "request", make_output(0), blocked_delivery, blocked_abort));
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, 1s, [&]() { return delivery_started; }));
  }
  std::thread overflow([&]() {
    overflow_rejected.store(
        !queue.enqueue(
            "request", make_output(1), blocked_delivery, blocked_abort),
        std::memory_order_release);
  });
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, 1s, [&]() { return abort_started; }));
  }
  EXPECT_FALSE(queue.enqueue(
      "request", make_output(2), blocked_delivery, blocked_abort));
  {
    std::lock_guard<std::mutex> lock(mutex);
    release_abort = true;
    release_delivery = true;
  }
  cv.notify_all();
  overflow.join();
  queue.wait_empty();

  EXPECT_TRUE(overflow_rejected.load(std::memory_order_acquire));
  EXPECT_EQ(deliveries.load(), 1);
  EXPECT_EQ(aborts.load(), 1);
}

TEST(OutputDeliveryQueueTest, RejectsOversizedEventBeforeDelivery) {
  OutputDeliveryQueue::Config config = make_config();
  config.max_bytes_per_request = 1;
  std::atomic<int32_t> deliveries{0};
  std::atomic<int32_t> aborts{0};
  OutputDeliveryQueue queue(config);

  EXPECT_FALSE(queue.enqueue(
      "request",
      make_output(0, "too-large"),
      [&](const RequestOutput&) {
        ++deliveries;
        return OutputDeliveryStatus::DELIVERED;
      },
      [&]() { ++aborts; }));
  EXPECT_EQ(deliveries.load(), 0);
  EXPECT_EQ(aborts.load(), 1);
}

TEST(OutputDeliveryQueueTest, OversizedTerminalDeliversBoundedErrorSkeleton) {
  OutputDeliveryQueue::Config config = make_config();
  config.max_bytes_per_request = 512;
  std::atomic<int32_t> deliveries{0};
  std::atomic<int32_t> aborts{0};
  OutputDeliveryQueue queue(config);
  RequestOutput terminal = make_output(0, std::string(4096, 'x'));
  terminal.finished = true;

  ASSERT_TRUE(queue.enqueue_terminal(
      "request",
      std::move(terminal),
      [&](const RequestOutput& output) {
        ++deliveries;
        EXPECT_TRUE(output.cancelled);
        EXPECT_FALSE(output.finished);
        EXPECT_TRUE(output.status.has_value());
        if (output.status.has_value()) {
          EXPECT_EQ(output.status->code(), StatusCode::RESOURCE_EXHAUSTED);
        }
        EXPECT_TRUE(output.outputs.empty());
        EXPECT_EQ(output.outputs.capacity(), 0u);
        EXPECT_FALSE(output.prompt.has_value());
        EXPECT_EQ(output.request_id, "engine-request");
        EXPECT_EQ(output.service_request_id, "service-request");
        EXPECT_EQ(output.target_xservice_addr, "127.0.0.1:28888");
        EXPECT_LE(OutputDeliveryQueue::retained_dynamic_size_bytes(output),
                  config.max_bytes_per_request);
        return OutputDeliveryStatus::DELIVERED;
      },
      [&]() { ++aborts; }));
  queue.wait_empty();

  EXPECT_EQ(deliveries.load(), 1);
  EXPECT_EQ(aborts.load(), 0);
}

TEST(OutputDeliveryQueueTest,
     TerminalIdentityBeyondRetentionLimitStillClosesDirectly) {
  OutputDeliveryQueue::Config config = make_config();
  config.max_bytes_per_request = 1;
  std::atomic<int32_t> deliveries{0};
  std::atomic<int32_t> aborts{0};
  OutputDeliveryQueue queue(config);
  RequestOutput terminal = make_output(0, std::string(4096, 'x'));
  terminal.finished = true;

  EXPECT_TRUE(queue.enqueue_terminal(
      "request",
      std::move(terminal),
      [&](const RequestOutput& output) {
        ++deliveries;
        EXPECT_TRUE(output.cancelled);
        EXPECT_TRUE(output.status.has_value());
        EXPECT_EQ(output.service_request_id, "service-request");
        return OutputDeliveryStatus::DELIVERED;
      },
      [&]() { ++aborts; }));
  queue.wait_empty();

  EXPECT_EQ(deliveries.load(), 1);
  EXPECT_EQ(aborts.load(), 0);
  EXPECT_EQ(queue.pending_requests(), 0u);
}

TEST(OutputDeliveryQueueTest, DuplicateTerminalAbortsExactlyOnceAndDrains) {
  OutputDeliveryQueue::Config config = make_config();
  config.worker_threads = 1;
  std::mutex mutex;
  std::condition_variable cv;
  bool delivery_started = false;
  bool release_delivery = false;
  std::atomic<int32_t> aborts{0};
  OutputDeliveryQueue queue(config);
  std::atomic<size_t> deliveries{0};
  auto deliver = [&](const RequestOutput&) {
    const size_t delivery = deliveries.fetch_add(1);
    if (delivery != 0) {
      return OutputDeliveryStatus::DELIVERED;
    }
    std::unique_lock<std::mutex> lock(mutex);
    delivery_started = true;
    cv.notify_all();
    cv.wait_for(lock, 1s, [&]() { return release_delivery; });
    return OutputDeliveryStatus::DELIVERED;
  };
  ASSERT_TRUE(
      queue.enqueue("request", make_output(0), deliver, [&]() { ++aborts; }));
  ASSERT_TRUE(
      queue.enqueue("request", make_output(1), deliver, [&]() { ++aborts; }));
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, 1s, [&]() { return delivery_started; }));
  }
  RequestOutput terminal = make_output(0);
  terminal.finished = true;
  ASSERT_TRUE(queue.enqueue_terminal(
      "request", std::move(terminal), deliver, [&]() { ++aborts; }));
  RequestOutput duplicate = make_output(1);
  duplicate.finished = true;
  EXPECT_FALSE(queue.enqueue_terminal(
      "request", std::move(duplicate), deliver, [&]() { ++aborts; }));
  {
    std::lock_guard<std::mutex> lock(mutex);
    release_delivery = true;
  }
  cv.notify_all();
  queue.wait_empty();

  EXPECT_EQ(aborts.load(), 1);
  EXPECT_EQ(deliveries.load(), 2u);
  EXPECT_EQ(queue.pending_requests(), 0u);
  EXPECT_EQ(queue.pending_events(), 0u);
}

TEST(OutputDeliveryQueueTest, StopAbortsPendingTerminalExactlyOnce) {
  OutputDeliveryQueue::Config config = make_config();
  config.worker_threads = 1;
  config.retry_interval = 1s;
  std::mutex mutex;
  std::condition_variable cv;
  bool delivery_started = false;
  bool release_abort = false;
  std::atomic<int32_t> aborts{0};
  std::atomic<int32_t> terminal_attempts{0};
  auto queue = std::make_unique<OutputDeliveryQueue>(config);
  ASSERT_TRUE(queue->enqueue(
      "abort-then-terminal",
      make_output(0),
      [&](const RequestOutput& output) {
        if (output.finished) {
          ++terminal_attempts;
          return OutputDeliveryStatus::DELIVERED;
        }
        return OutputDeliveryStatus::SUBSCRIBER_REJECTED;
      },
      [&]() {
        ++aborts;
        std::unique_lock<std::mutex> lock(mutex);
        delivery_started = true;
        cv.notify_all();
        cv.wait(lock, [&]() { return release_abort; });
      }));
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, 1s, [&]() { return delivery_started; }));
  }
  RequestOutput terminal = make_output(0);
  terminal.finished = true;
  ASSERT_TRUE(queue->enqueue_terminal(
      "abort-then-terminal",
      std::move(terminal),
      [](const RequestOutput&) { return OutputDeliveryStatus::DELIVERED; },
      []() {}));

  std::thread stopping([owned = std::move(queue)]() mutable { owned.reset(); });
  {
    std::lock_guard<std::mutex> lock(mutex);
    release_abort = true;
  }
  cv.notify_all();
  stopping.join();

  EXPECT_EQ(aborts.load(), 1);
  EXPECT_EQ(terminal_attempts.load(), 1);
}

TEST(OutputDeliveryQueueTest, StopRetriesOneFailedInflightTerminal) {
  OutputDeliveryQueue::Config config = make_config();
  config.worker_threads = 1;
  std::mutex mutex;
  std::condition_variable cv;
  bool first_delivery_started = false;
  bool release_first_delivery = false;
  std::atomic<int32_t> attempts{0};
  std::atomic<int32_t> aborts{0};
  auto queue = std::make_unique<OutputDeliveryQueue>(config);
  RequestOutput terminal = make_output(0);
  terminal.finished = true;
  ASSERT_TRUE(queue->enqueue_terminal(
      "terminal",
      std::move(terminal),
      [&](const RequestOutput&) {
        const int32_t attempt = attempts.fetch_add(1);
        if (attempt == 0) {
          std::unique_lock<std::mutex> lock(mutex);
          first_delivery_started = true;
          cv.notify_all();
          cv.wait(lock, [&]() { return release_first_delivery; });
          return OutputDeliveryStatus::SUBSCRIBER_UNREACHABLE;
        }
        return OutputDeliveryStatus::DELIVERED;
      },
      [&]() { ++aborts; }));
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(
        cv.wait_for(lock, 1s, [&]() { return first_delivery_started; }));
  }

  std::thread stopping([owned = std::move(queue)]() mutable { owned.reset(); });
  {
    std::lock_guard<std::mutex> lock(mutex);
    release_first_delivery = true;
  }
  cv.notify_all();
  stopping.join();

  EXPECT_EQ(attempts.load(), 2);
  EXPECT_EQ(aborts.load(), 1);
}

TEST(OutputDeliveryQueueTest, StopUsesSharedTerminalRetryDeadline) {
  OutputDeliveryQueue::Config config = make_config();
  config.worker_threads = 1;
  config.retry_interval = 1s;
  config.delivery_timeout = 10ms;
  config.shutdown_terminal_delivery_timeout = 25ms;
  std::mutex mutex;
  std::condition_variable cv;
  bool blocker_started = false;
  bool release_blocker = false;
  std::atomic<int32_t> terminal_attempts{0};
  std::atomic<int32_t> aborts{0};
  auto queue = std::make_unique<OutputDeliveryQueue>(config);

  ASSERT_TRUE(queue->enqueue(
      "blocker",
      make_output(0),
      [&](const RequestOutput&) {
        std::unique_lock<std::mutex> lock(mutex);
        blocker_started = true;
        cv.notify_all();
        cv.wait(lock, [&]() { return release_blocker; });
        return OutputDeliveryStatus::SUBSCRIBER_UNREACHABLE;
      },
      [&]() { ++aborts; }));
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, 1s, [&]() { return blocker_started; }));
  }

  constexpr int32_t kTerminalCount = 20;
  for (int32_t index = 0; index < kTerminalCount; ++index) {
    RequestOutput terminal = make_output(static_cast<uint64_t>(index));
    terminal.finished = true;
    ASSERT_TRUE(queue->enqueue_terminal(
        "terminal-" + std::to_string(index),
        std::move(terminal),
        [&](const RequestOutput&) {
          ++terminal_attempts;
          std::this_thread::sleep_for(5ms);
          return OutputDeliveryStatus::SUBSCRIBER_UNREACHABLE;
        },
        [&]() { ++aborts; }));
  }

  const auto stop_start = std::chrono::steady_clock::now();
  std::thread stopping([owned = std::move(queue)]() mutable { owned.reset(); });
  {
    std::lock_guard<std::mutex> lock(mutex);
    release_blocker = true;
  }
  cv.notify_all();
  stopping.join();
  const auto stop_elapsed = std::chrono::steady_clock::now() - stop_start;

  EXPECT_GT(terminal_attempts.load(), 0);
  EXPECT_LT(terminal_attempts.load(), kTerminalCount);
  EXPECT_EQ(aborts.load(), kTerminalCount + 1);
  EXPECT_LT(stop_elapsed, 250ms);
}

TEST(OutputDeliveryQueueTest, ConcurrentTerminalProducersDeliverExactlyOnce) {
  OutputDeliveryQueue::Config config = make_config();
  config.worker_threads = 4;
  config.max_pending_requests = 8;
  config.max_terminal_only_requests = 128;
  constexpr size_t kRequests = 128;
  std::atomic<size_t> deliveries{0};
  std::atomic<size_t> aborts{0};
  std::atomic<size_t> accepted{0};
  std::mutex start_mutex;
  std::condition_variable start_cv;
  bool start = false;
  OutputDeliveryQueue queue(config);

  constexpr size_t kProducers = 8;
  std::vector<std::thread> producers;
  producers.reserve(kProducers);
  for (size_t producer = 0; producer < kProducers; ++producer) {
    producers.emplace_back([&, producer]() {
      {
        std::unique_lock<std::mutex> lock(start_mutex);
        start_cv.wait(lock, [&]() { return start; });
      }
      for (size_t offset = producer; offset < kRequests; offset += kProducers) {
        RequestOutput terminal = make_output(offset);
        terminal.finished = true;
        if (queue.enqueue_terminal(
                "terminal-" + std::to_string(offset),
                std::move(terminal),
                [&](const RequestOutput&) {
                  ++deliveries;
                  return OutputDeliveryStatus::DELIVERED;
                },
                [&]() { ++aborts; })) {
          ++accepted;
        }
      }
    });
  }
  {
    std::lock_guard<std::mutex> lock(start_mutex);
    start = true;
  }
  start_cv.notify_all();
  for (std::thread& producer : producers) {
    producer.join();
  }
  queue.wait_empty();

  EXPECT_EQ(accepted.load(), kRequests);
  EXPECT_EQ(deliveries.load(), kRequests);
  EXPECT_EQ(aborts.load(), 0u);
  EXPECT_EQ(queue.pending_requests(), 0u);
  EXPECT_EQ(queue.pending_events(), 0u);
}

TEST(OutputDeliveryQueueTest, TerminalReserveOverflowUsesDirectClosure) {
  OutputDeliveryQueue::Config config = make_config();
  config.worker_threads = 1;
  config.max_pending_requests = 1;
  config.max_terminal_only_requests = 2;
  std::mutex mutex;
  std::condition_variable cv;
  bool delivery_started = false;
  bool release_delivery = false;
  std::atomic<size_t> rejected_aborts{0};
  std::atomic<size_t> terminal_direct_deliveries{0};
  OutputDeliveryQueue queue(config);

  ASSERT_TRUE(queue.enqueue(
      "blocked",
      make_output(0),
      [&](const RequestOutput&) {
        std::unique_lock<std::mutex> lock(mutex);
        delivery_started = true;
        cv.notify_all();
        cv.wait_for(lock, 1s, [&]() { return release_delivery; });
        return OutputDeliveryStatus::DELIVERED;
      },
      []() {}));
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, 1s, [&]() { return delivery_started; }));
  }
  for (size_t request = 0; request < 2; ++request) {
    RequestOutput terminal = make_output(request);
    terminal.finished = true;
    ASSERT_TRUE(queue.enqueue_terminal(
        "terminal-" + std::to_string(request),
        std::move(terminal),
        [](const RequestOutput&) { return OutputDeliveryStatus::DELIVERED; },
        []() {}));
  }
  RequestOutput overflow = make_output(2);
  overflow.finished = true;
  EXPECT_TRUE(queue.enqueue_terminal(
      "terminal-overflow",
      std::move(overflow),
      [&](const RequestOutput&) {
        ++terminal_direct_deliveries;
        return OutputDeliveryStatus::DELIVERED;
      },
      [&]() { ++rejected_aborts; }));
  {
    std::lock_guard<std::mutex> lock(mutex);
    release_delivery = true;
  }
  cv.notify_all();
  queue.wait_empty();

  EXPECT_EQ(rejected_aborts.load(), 0u);
  EXPECT_EQ(terminal_direct_deliveries.load(), 1u);
  EXPECT_EQ(queue.pending_requests(), 0u);
  EXPECT_EQ(queue.pending_events(), 0u);
}

TEST(OutputDeliveryQueueTest, CombinedByteLimitAbortsDataAndKeepsTerminal) {
  RequestOutput data = make_output(0, std::string(512, 'd'));
  RequestOutput terminal = make_output(1, std::string(512, 't'));
  terminal.finished = true;
  const size_t data_bytes =
      OutputDeliveryQueue::retained_dynamic_size_bytes(data);
  const size_t terminal_bytes =
      OutputDeliveryQueue::retained_dynamic_size_bytes(terminal);
  OutputDeliveryQueue::Config config = make_config();
  config.worker_threads = 1;
  config.max_bytes_per_request = std::max(data_bytes, terminal_bytes) + 1;
  std::mutex mutex;
  std::condition_variable cv;
  bool data_started = false;
  bool release_data = false;
  std::atomic<int32_t> terminal_deliveries{0};
  std::atomic<int32_t> aborts{0};
  OutputDeliveryQueue queue(config);
  auto deliver = [&](const RequestOutput& output) {
    if (output.finished) {
      ++terminal_deliveries;
      return OutputDeliveryStatus::DELIVERED;
    }
    std::unique_lock<std::mutex> lock(mutex);
    data_started = true;
    cv.notify_all();
    cv.wait(lock, [&]() { return release_data; });
    return OutputDeliveryStatus::DELIVERED;
  };

  ASSERT_TRUE(
      queue.enqueue("request", std::move(data), deliver, [&]() { ++aborts; }));
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, 1s, [&]() { return data_started; }));
  }
  ASSERT_TRUE(queue.enqueue_terminal(
      "request", std::move(terminal), deliver, [&]() { ++aborts; }));
  {
    std::lock_guard<std::mutex> lock(mutex);
    release_data = true;
  }
  cv.notify_all();
  queue.wait_empty();

  EXPECT_EQ(aborts.load(), 1);
  EXPECT_EQ(terminal_deliveries.load(), 1);
  EXPECT_EQ(queue.pending_events(), 0u);
}

TEST(OutputDeliveryQueueTest, CountsRetainedTorchCpuStorageForTensorViews) {
  OutputDeliveryQueue::Config config = make_config();
  config.max_bytes_per_request = 1024;
  std::atomic<int32_t> deliveries{0};
  std::atomic<int32_t> aborts{0};
  RequestOutput output = make_output(0);
  torch::Tensor storage = torch::ones({32, 32}, torch::kFloat32);
  output.outputs[0].embedding = storage.transpose(0, 1);
  ASSERT_FALSE(output.outputs[0].embedding.is_contiguous());
  OutputDeliveryQueue queue(config);

  EXPECT_FALSE(queue.enqueue(
      "request",
      std::move(output),
      [&](const RequestOutput&) {
        ++deliveries;
        return OutputDeliveryStatus::DELIVERED;
      },
      [&]() { ++aborts; }));
  EXPECT_EQ(deliveries.load(), 0);
  EXPECT_EQ(aborts.load(), 1);
}

TEST(OutputDeliveryQueueTest, RetainsTorchCpuTensorUntilDeliveryCompletes) {
  OutputDeliveryQueue::Config config = make_config();
  config.worker_threads = 1;
  config.max_bytes_per_request = 1024 * 1024;
  std::mutex mutex;
  std::condition_variable cv;
  bool delivery_started = false;
  bool inspect_tensor = false;
  std::atomic<bool> tensor_valid{false};
  RequestOutput output = make_output(0);
  torch::Tensor source = torch::arange(4, torch::kFloat32);
  output.outputs[0].embedding = source;
  OutputDeliveryQueue queue(config);

  ASSERT_TRUE(queue.enqueue(
      "request",
      std::move(output),
      [&](const RequestOutput& pending_output) {
        std::unique_lock<std::mutex> lock(mutex);
        delivery_started = true;
        cv.notify_all();
        cv.wait_for(lock, 1s, [&]() { return inspect_tensor; });
        tensor_valid.store(
            pending_output.outputs[0].embedding.device().is_cpu() &&
                pending_output.outputs[0].embedding.scalar_type() ==
                    torch::kFloat32 &&
                pending_output.outputs[0].embedding.sizes().vec() ==
                    std::vector<int64_t>{4} &&
                pending_output.outputs[0].embedding.sum().item<float>() == 6.0f,
            std::memory_order_release);
        return OutputDeliveryStatus::DELIVERED;
      },
      []() {}));
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, 1s, [&]() { return delivery_started; }));
    source = torch::Tensor();
    inspect_tensor = true;
  }
  cv.notify_all();
  queue.wait_empty();

  EXPECT_TRUE(tensor_valid.load(std::memory_order_acquire));
}

TEST(OutputDeliveryQueueTest, AcceptsEmptyTorchCpuTensorStorage) {
  std::atomic<int32_t> deliveries{0};
  std::atomic<bool> empty_tensor_valid{false};
  RequestOutput output = make_output(0);
  output.outputs[0].embedding = torch::empty({0}, torch::kFloat32);
  OutputDeliveryQueue queue(make_config());

  ASSERT_TRUE(queue.enqueue(
      "request",
      std::move(output),
      [&](const RequestOutput& pending_output) {
        empty_tensor_valid.store(
            pending_output.outputs[0].embedding.device().is_cpu() &&
                pending_output.outputs[0].embedding.numel() == 0,
            std::memory_order_release);
        ++deliveries;
        return OutputDeliveryStatus::DELIVERED;
      },
      []() {}));
  queue.wait_empty();

  EXPECT_EQ(deliveries.load(), 1);
  EXPECT_TRUE(empty_tensor_valid.load(std::memory_order_acquire));
}

TEST(OutputDeliveryQueueTest, CountsNestedVectorCapacity) {
  OutputDeliveryQueue::Config config = make_config();
  config.max_bytes_per_request = 1024;
  std::atomic<int32_t> aborts{0};
  RequestOutput output = make_output(0);
  output.outputs[0].item_infos_list.resize(128);
  OutputDeliveryQueue queue(config);

  EXPECT_FALSE(queue.enqueue(
      "request",
      std::move(output),
      [](const RequestOutput&) { return OutputDeliveryStatus::DELIVERED; },
      [&]() { ++aborts; }));
  EXPECT_EQ(aborts.load(), 1);
}

TEST(OutputDeliveryQueueTest, ConcurrentRequestsRemainIsolatedAndOrdered) {
  OutputDeliveryQueue::Config config = make_config();
  config.worker_threads = 4;
  config.max_pending_requests = 64;
  config.max_events_per_request = 64;
  config.max_bytes_per_request = 1024 * 1024;
  constexpr size_t kRequests = 16;
  constexpr size_t kEvents = 32;
  std::mutex mutex;
  std::vector<std::vector<uint64_t>> delivered(kRequests);
  std::atomic<int32_t> aborts{0};
  OutputDeliveryQueue queue(config);
  std::vector<std::thread> producers;

  for (size_t request = 0; request < kRequests; ++request) {
    producers.emplace_back([&, request]() {
      for (size_t event = 0; event < kEvents; ++event) {
        ASSERT_TRUE(queue.enqueue(
            "request-" + std::to_string(request),
            make_output(event),
            [&, request](const RequestOutput& output) {
              std::lock_guard<std::mutex> lock(mutex);
              delivered[request].emplace_back(*output.output_event_seq);
              return OutputDeliveryStatus::DELIVERED;
            },
            [&]() { ++aborts; }));
      }
    });
  }
  for (std::thread& producer : producers) {
    producer.join();
  }
  queue.wait_empty();

  EXPECT_EQ(aborts.load(), 0);
  for (const std::vector<uint64_t>& events : delivered) {
    ASSERT_EQ(events.size(), kEvents);
    for (size_t index = 0; index < events.size(); ++index) {
      EXPECT_EQ(events[index], index);
    }
  }
}

}  // namespace
}  // namespace xllm
