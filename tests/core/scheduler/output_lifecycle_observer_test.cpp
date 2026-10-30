/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/request/output_lifecycle_observer.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace xllm {
namespace {

struct ObservedEvent {
  OutputLifecycleEventType type;
  uint64_t sequence;
};

RequestOutput make_output(uint64_t sequence) {
  RequestOutput output;
  output.output_event_seq = sequence;
  return output;
}

TEST(OutputLifecycleObserverTest, EmitsFirstAndTerminalInOrderOnce) {
  std::vector<ObservedEvent> events;
  OutputLifecycleObserver observer(
      [&](OutputLifecycleEventType type, const RequestOutput& output) {
        events.push_back(ObservedEvent{type, *output.output_event_seq});
      });

  observer.observe(make_output(0));
  observer.observe(make_output(1));
  RequestOutput terminal = make_output(2);
  terminal.finished = true;
  observer.observe(terminal);
  observer.observe(terminal);

  ASSERT_EQ(events.size(), 2u);
  EXPECT_EQ(events[0].type, OutputLifecycleEventType::FIRST_OUTPUT);
  EXPECT_EQ(events[0].sequence, 0u);
  EXPECT_EQ(events[1].type, OutputLifecycleEventType::TERMINAL);
  EXPECT_EQ(events[1].sequence, 2u);
}

TEST(OutputLifecycleObserverTest, TreatsFailureAsFirstAndTerminal) {
  std::vector<ObservedEvent> events;
  OutputLifecycleObserver observer(
      [&](OutputLifecycleEventType type, const RequestOutput& output) {
        events.push_back(ObservedEvent{type, *output.output_event_seq});
      });
  RequestOutput output = make_output(7);
  output.status = Status(StatusCode::UNKNOWN, "failure");

  observer.observe(output);

  ASSERT_EQ(events.size(), 2u);
  EXPECT_EQ(events[0].type, OutputLifecycleEventType::FIRST_OUTPUT);
  EXPECT_EQ(events[1].type, OutputLifecycleEventType::TERMINAL);
  EXPECT_EQ(events[0].sequence, 7u);
  EXPECT_EQ(events[1].sequence, 7u);
}

TEST(OutputLifecycleObserverTest, SerializesConcurrentTerminalObservation) {
  std::vector<ObservedEvent> events;
  OutputLifecycleObserver observer(
      [&](OutputLifecycleEventType type, const RequestOutput& output) {
        events.push_back(ObservedEvent{type, *output.output_event_seq});
      });
  std::atomic<bool> start{false};
  std::vector<std::thread> workers;
  for (uint64_t sequence = 0; sequence < 32; ++sequence) {
    workers.emplace_back([&observer, &start, sequence]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      RequestOutput output = make_output(sequence);
      output.finished = true;
      observer.observe(output);
    });
  }
  start.store(true, std::memory_order_release);
  for (std::thread& worker : workers) {
    worker.join();
  }

  ASSERT_EQ(events.size(), 2u);
  EXPECT_EQ(events[0].type, OutputLifecycleEventType::FIRST_OUTPUT);
  EXPECT_EQ(events[1].type, OutputLifecycleEventType::TERMINAL);
  EXPECT_EQ(events[0].sequence, events[1].sequence);
}

}  // namespace
}  // namespace xllm
