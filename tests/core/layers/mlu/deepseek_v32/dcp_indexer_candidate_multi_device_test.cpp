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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <torch/torch.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

#include "framework/parallel_state/context_parallel_topology.h"
#include "framework/parallel_state/process_group.h"
#include "layers/mlu/deepseek_v32/dcp_indexer_candidate.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "util/net.h"

namespace xllm::layer {
namespace {
bool child_failed(int32_t initial_parts) {
  const auto* result =
      ::testing::UnitTest::GetInstance()->current_test_info()->result();
  for (int32_t idx = initial_parts; idx < result->total_part_count(); ++idx) {
    if (result->GetTestPartResult(idx).failed()) {
      return true;
    }
  }
  return false;
}

void stop_ranks(const std::vector<pid_t>& pids,
                const std::vector<bool>& finished) {
  for (size_t idx = 0; idx < pids.size(); ++idx) {
    if (!finished[idx]) {
      ::kill(pids[idx], SIGKILL);
    }
  }
  for (size_t idx = 0; idx < pids.size(); ++idx) {
    if (finished[idx]) {
      continue;
    }
    int status = 0;
    while (::waitpid(pids[idx], &status, 0) < 0 && errno == EINTR) {
    }
  }
}

bool wait_ranks(const std::vector<pid_t>& pids,
                std::chrono::milliseconds timeout,
                bool& skipped) {
  std::vector<bool> finished(pids.size(), false);
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  size_t remaining = pids.size();
  int32_t skip_count = 0;
  while (remaining > 0 && std::chrono::steady_clock::now() < deadline) {
    for (size_t idx = 0; idx < pids.size(); ++idx) {
      if (finished[idx]) {
        continue;
      }
      int status = 0;
      const pid_t pid = ::waitpid(pids[idx], &status, WNOHANG);
      if (pid == 0 || (pid < 0 && errno == EINTR)) {
        continue;
      }
      finished[idx] = true;
      --remaining;
      if (pid < 0 || !WIFEXITED(status) ||
          (WEXITSTATUS(status) != 0 && WEXITSTATUS(status) != 77)) {
        LOG(ERROR) << "Rank " << idx << " wait status=" << status
                   << " waitpid=" << pid << " errno=" << errno;
        stop_ranks(pids, finished);
        return false;
      }
      skip_count += WEXITSTATUS(status) == 77 ? 1 : 0;
    }
    if (remaining > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds{10});
    }
  }
  if (remaining > 0) {
    stop_ranks(pids, finished);
    return false;
  }
  skipped = skip_count == static_cast<int32_t>(pids.size());
  return skip_count == 0 || skipped;
}

constexpr int32_t kExitCodeSkip = 77;

// Rank-encoded local fixtures: score[r] = 10r + {2, 1}, slot[r] = 100r + {5,
// 7}.
torch::Tensor make_local_scores(int32_t rank,
                                const torch::TensorOptions& options) {
  const float rank_score = static_cast<float>(rank * 10);
  return torch::tensor({{rank_score + 2.0f, rank_score + 1.0f}}, options);
}

torch::Tensor make_local_slots(int32_t rank,
                               const torch::TensorOptions& options) {
  const int32_t rank_slot = rank * 100;
  return torch::tensor({{rank_slot + 5, rank_slot + 7}}, options);
}

void verify_gathered_candidates(const DcpIndexerGatheredCandidates& gathered,
                                int32_t world_size,
                                const torch::Device& device) {
  const torch::TensorOptions score_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const torch::TensorOptions slot_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  CHECK_EQ(gathered.scores.sizes(), (torch::IntArrayRef{world_size, 1, 2}));
  CHECK_EQ(gathered.global_slots.sizes(),
           (torch::IntArrayRef{world_size, 1, 2}));
  for (int32_t source_rank = 0; source_rank < world_size; ++source_rank) {
    const torch::Tensor expected_scores =
        make_local_scores(source_rank, score_options);
    const torch::Tensor expected_slots =
        make_local_slots(source_rank, slot_options);
    CHECK(torch::equal(gathered.scores[source_rank], expected_scores));
    CHECK(torch::equal(gathered.global_slots[source_rank], expected_slots));
  }
}

int32_t run_candidate_collective_child(int32_t rank,
                                       int32_t world_size,
                                       int32_t port,
                                       const std::string& host) {
  try {
    if (Platform::device_count() < world_size) {
      LOG(WARNING) << "DCP candidate collective requires " << world_size
                   << " MLU devices";
      return kExitCodeSkip;
    }
    Device xllm_device(rank);
    xllm_device.set_device();
    const torch::Device device = xllm_device.unwrap();
    std::unique_ptr<ProcessGroup> process_group =
        create_process_group(rank,
                             world_size,
                             world_size,
                             port,
                             /*use_dp_master=*/false,
                             host,
                             "dcp_indexer_candidate_collective_test",
                             device);
    CHECK(process_group) << "Failed to create DCP candidate process group";

    torch::Tensor rendezvous = torch::ones(
        {1}, torch::TensorOptions().dtype(torch::kBFloat16).device(device));
    process_group->allreduce(rendezvous);
    CHECK_EQ(rendezvous.item<float>(), static_cast<float>(world_size));

    const torch::TensorOptions score_options =
        torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    const torch::TensorOptions slot_options =
        torch::TensorOptions().dtype(torch::kInt32).device(device);

    DcpIndexerGatherAsyncCtx gather = launch_dcp_indexer_candidate_gather(
        make_local_scores(rank, score_options),
        make_local_slots(rank, slot_options),
        process_group.get());
    const DcpIndexerGatheredCandidates gathered =
        finish_dcp_indexer_candidate_gather(std::move(gather));
    xllm_device.synchronize_default_stream();
    verify_gathered_candidates(gathered, world_size, device);
    return 0;
  } catch (const std::exception& error) {
    LOG(ERROR) << "DCP candidate rank " << rank << " failed: " << error.what();
    return 1;
  }
}

// Exercises the launch/finish overlap contract: independent device compute is
// enqueued while the candidate all-gather is in flight, and both the compute
// result and the gathered buffers must stay intact after the finish.
int32_t run_candidate_async_collective_child(int32_t rank,
                                             int32_t world_size,
                                             int32_t port,
                                             const std::string& host) {
  constexpr int64_t kOverlapMatrixSize = 8;
  try {
    if (Platform::device_count() < world_size) {
      LOG(WARNING) << "DCP candidate collective requires " << world_size
                   << " MLU devices";
      return kExitCodeSkip;
    }
    Device xllm_device(rank);
    xllm_device.set_device();
    const torch::Device device = xllm_device.unwrap();
    std::unique_ptr<ProcessGroup> process_group =
        create_process_group(rank,
                             world_size,
                             world_size,
                             port,
                             /*use_dp_master=*/false,
                             host,
                             "dcp_indexer_candidate_async_collective_test",
                             device);
    CHECK(process_group) << "Failed to create DCP candidate process group";

    torch::Tensor rendezvous = torch::ones(
        {1}, torch::TensorOptions().dtype(torch::kBFloat16).device(device));
    process_group->allreduce(rendezvous);
    CHECK_EQ(rendezvous.item<float>(), static_cast<float>(world_size));

    const torch::TensorOptions score_options =
        torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    const torch::TensorOptions slot_options =
        torch::TensorOptions().dtype(torch::kInt32).device(device);

    DcpIndexerGatherAsyncCtx gather = launch_dcp_indexer_candidate_gather(
        make_local_scores(rank, score_options),
        make_local_slots(rank, slot_options),
        process_group.get());
    const torch::Tensor overlap_probe = torch::matmul(
        torch::ones({kOverlapMatrixSize, kOverlapMatrixSize}, score_options),
        torch::full({kOverlapMatrixSize, kOverlapMatrixSize},
                    static_cast<float>(rank + 1),
                    score_options));
    const DcpIndexerGatheredCandidates gathered =
        finish_dcp_indexer_candidate_gather(std::move(gather));
    xllm_device.synchronize_default_stream();
    CHECK_EQ(overlap_probe.sum().item<float>(),
             static_cast<float>(kOverlapMatrixSize * kOverlapMatrixSize *
                                kOverlapMatrixSize * (rank + 1)));
    verify_gathered_candidates(gathered, world_size, device);
    return 0;
  } catch (const std::exception& error) {
    LOG(ERROR) << "DCP candidate rank " << rank << " failed: " << error.what();
    return 1;
  }
}

int32_t run_candidate_bits_child(int32_t rank,
                                 int32_t world_size,
                                 int32_t port,
                                 const std::string& host) {
  try {
    if (Platform::device_count() < world_size) {
      return kExitCodeSkip;
    }
    Device xllm_device(rank);
    xllm_device.set_device();
    const torch::Device device = xllm_device.unwrap();
    auto group = create_process_group(rank,
                                      world_size,
                                      world_size,
                                      port,
                                      /*use_dp_master=*/false,
                                      host,
                                      "dcp_bf16_wire_bits_test",
                                      device);
    CHECK(group);
    const std::array<std::pair<int32_t, int32_t>, 4> shapes{
        {{16, 2048}, {32, 2048}, {64, 2048}, {1, 3}}};
    const std::array<int32_t, 4> slot_masks{std::numeric_limits<int32_t>::min(),
                                            std::numeric_limits<int32_t>::max(),
                                            0x40000001,
                                            -1};
    for (const auto& [rows, cols] : shapes) {
      const int32_t count = rows * cols;
      const torch::Tensor bits = (torch::arange(count, torch::kInt32) + rank)
                                     .to(torch::kInt16)
                                     .reshape({rows, cols});
      for (const bool adjacent : {false, true}) {
        torch::Tensor scores = bits.to(device).view(torch::kBFloat16);
        torch::Tensor slots =
            (torch::arange(count, torch::kInt32) + rank * count)
                .bitwise_xor(slot_masks[rank % 4])
                .reshape({rows, cols})
                .to(device);
        if (adjacent) {
          const int64_t score_bytes = static_cast<int64_t>(count) * 2;
          const int64_t slot_offset = (score_bytes + 3) / 4 * 4;
          torch::Tensor packet =
              torch::empty({8 + slot_offset + static_cast<int64_t>(count) * 4},
                           scores.options().dtype(torch::kUInt8));
          torch::Tensor packed_scores = packet.narrow(0, 8, score_bytes)
                                            .view(torch::kBFloat16)
                                            .view({rows, cols});
          torch::Tensor packed_slots =
              packet.narrow(0, 8 + slot_offset, static_cast<int64_t>(count) * 4)
                  .view(torch::kInt32)
                  .view({rows, cols});
          packed_scores.copy_(scores);
          packed_slots.copy_(slots);
          scores = std::move(packed_scores);
          slots = std::move(packed_slots);
        }
        const torch::Tensor input_slots = slots.cpu();
        auto ctx =
            launch_dcp_indexer_candidate_gather(scores, slots, group.get());
        CHECK_EQ(ctx.local_packet.is_alias_of(scores),
                 adjacent && count % 2 == 0);
        const auto gathered =
            finish_dcp_indexer_candidate_gather(std::move(ctx));
        CHECK_EQ(gathered.scores.scalar_type(), torch::kBFloat16);
        CHECK(gathered.scores.sizes() ==
              torch::IntArrayRef({world_size, rows, cols}));
        for (int32_t source_rank = 0; source_rank < world_size; ++source_rank) {
          const torch::Tensor expected_bits =
              (torch::arange(count, torch::kInt32) + source_rank)
                  .to(torch::kInt16)
                  .reshape({rows, cols});
          const torch::Tensor expected_slots =
              (torch::arange(count, torch::kInt32) + source_rank * count)
                  .bitwise_xor(slot_masks[source_rank % 4])
                  .reshape({rows, cols});
          CHECK(torch::equal(
              gathered.scores[source_rank].view(torch::kInt16).cpu(),
              expected_bits));
          CHECK(torch::equal(gathered.global_slots[source_rank].cpu(),
                             expected_slots));
        }
        CHECK(torch::equal(scores.view(torch::kInt16).cpu(), bits));
        CHECK(torch::equal(slots.cpu(), input_slots));
      }
    }
    return 0;
  } catch (const std::exception& error) {
    LOG(ERROR) << "DCP candidate bits rank " << rank
               << " failed: " << error.what();
    return 1;
  }
}

using CandidateChildFn = int32_t (*)(int32_t,
                                     int32_t,
                                     int32_t,
                                     const std::string&);

void run_candidate_collective_test(int32_t world_size,
                                   CandidateChildFn child_fn) {
  const int32_t port = net::get_local_free_port();
  ASSERT_GT(port, 0) << "Failed to allocate a DCP candidate rendezvous port";

  const int32_t child_count = world_size;
  const int32_t initial_parts = ::testing::UnitTest::GetInstance()
                                    ->current_test_info()
                                    ->result()
                                    ->total_part_count();
  std::vector<pid_t> child_pids;
  child_pids.reserve(child_count);
  bool children_ok = true;
  bool children_skipped = false;
  for (int32_t child_rank = 0; child_rank < child_count; ++child_rank) {
    const pid_t pid = ::fork();
    if (pid == 0) {
      try {
        if (Platform::device_count() < child_count) {
          ::_exit(77);
        }
        const auto rank_body = [&](int32_t rank) {
          return child_fn(rank, world_size, port, /*host=*/"127.0.0.1");
        };
        const int32_t code = rank_body(child_rank);
        std::fflush(nullptr);
        ::_exit(child_failed(initial_parts) ? 1 : code);
      } catch (const std::exception& error) {
        LOG(ERROR) << "Rank " << child_rank << ": " << error.what();
        ::_exit(1);
      }
    }
    if (pid < 0) {
      children_ok = false;
      break;
    }
    child_pids.emplace_back(pid);
  }
  if (!children_ok) {
    stop_ranks(child_pids, std::vector<bool>(child_pids.size(), false));
  } else {
    children_ok = wait_ranks(
        child_pids, std::chrono::milliseconds{120000}, children_skipped);
  }

  if (children_ok && children_skipped) {
    GTEST_SKIP()
        << "forked ranks failed, timed out, or require more physical MLU cards";
  }
  EXPECT_EQ(children_ok, true)
      << "forked ranks failed, timed out, or require more physical MLU cards";
}

// The four-rank sweep covers the richer interleave topology on real CNCL
// groups; the two-rank overlap contract below covers the launch/finish split.
TEST(DcpIndexerCandidateMultiDeviceTest, ExchangesCandidatesAcrossFourRanks) {
  run_candidate_collective_test(/*world_size=*/4,
                                run_candidate_collective_child);
}

TEST(DcpIndexerCandidateMultiDeviceTest,
     OverlapsComputeWithLaunchedCandidateGather) {
  run_candidate_collective_test(/*world_size=*/2,
                                run_candidate_async_collective_child);
}

TEST(DcpIndexerCandidateMultiDeviceTest, PreservesEveryBf16BitPattern) {
  run_candidate_collective_test(/*world_size=*/4, run_candidate_bits_child);
}

}  // namespace
}  // namespace xllm::layer
