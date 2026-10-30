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

#include "core/distributed_runtime/spawn_worker_server/spawn_worker_shutdown.h"

#include <gtest/gtest.h>
#include <signal.h>
#include <unistd.h>

#include <chrono>

namespace xllm::spawn_worker_lifecycle {
namespace {

TEST(SpawnWorkerShutdownTest, ReapsChildThatAcceptsTerminate) {
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    while (true) {
      pause();
    }
  }

  const TerminationResult result =
      terminate_and_reap(child,
                         std::chrono::seconds(1),
                         std::chrono::seconds(1),
                         std::chrono::milliseconds(10));
  EXPECT_EQ(result.outcome, TerminationOutcome::TERMINATED);
  ASSERT_TRUE(WIFSIGNALED(result.status));
  EXPECT_EQ(WTERMSIG(result.status), SIGTERM);
}

TEST(SpawnWorkerShutdownTest, EscalatesAndReapsChildStuckDuringStartup) {
  int ready_pipe[2] = {-1, -1};
  ASSERT_EQ(pipe(ready_pipe), 0);
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    close(ready_pipe[0]);
    struct sigaction action = {};
    action.sa_handler = SIG_IGN;
    sigemptyset(&action.sa_mask);
    sigaction(SIGTERM, &action, nullptr);
    const char ready = '1';
    (void)write(ready_pipe[1], &ready, sizeof(ready));
    while (true) {
      pause();
    }
  }
  close(ready_pipe[1]);
  char ready = 0;
  ASSERT_EQ(read(ready_pipe[0], &ready, sizeof(ready)),
            static_cast<ssize_t>(sizeof(ready)));
  close(ready_pipe[0]);

  const auto started = std::chrono::steady_clock::now();
  const TerminationResult result =
      terminate_and_reap(child,
                         std::chrono::milliseconds(50),
                         std::chrono::seconds(1),
                         std::chrono::milliseconds(5));
  const auto elapsed = std::chrono::steady_clock::now() - started;

  EXPECT_EQ(result.outcome, TerminationOutcome::KILLED);
  ASSERT_TRUE(WIFSIGNALED(result.status));
  EXPECT_EQ(WTERMSIG(result.status), SIGKILL);
  EXPECT_LT(elapsed, std::chrono::seconds(1));
}

}  // namespace
}  // namespace xllm::spawn_worker_lifecycle
