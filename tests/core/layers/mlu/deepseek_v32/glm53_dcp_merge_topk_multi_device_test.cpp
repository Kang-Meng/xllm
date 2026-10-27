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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <bit>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/config/parallel_config.h"
#include "core/framework/parallel_state/collective_communicator.h"
#include "core/kernels/mlu/dcp/dcp_merge_test_utils.h"
#include "core/layers/mlu/deepseek_v32/multi_device_test_utils.h"
#include "core/platform/device.h"
#include "kernels/mlu/mlu_ops_api.h"

namespace xllm {
namespace {

constexpr int32_t kRanks = 8;
constexpr int32_t kDcp = 4;
constexpr int32_t kPage = 16;
constexpr int32_t kStripe = kDcp * kPage;
constexpr int32_t kTopk = 2048;
constexpr int32_t kQueries = 8192;
constexpr int32_t kEdgeRows = 16;
constexpr int32_t kSentinel = 123456789;
using TopkResult = std::pair<torch::Tensor, torch::Tensor>;

using dcp_test::bf16_pattern;
using dcp_test::Candidates;
using dcp_test::make_candidates;
using dcp_test::merge_reference;
using dcp_test::strided_copy;

enum class Deployment { PREFILL, DECODE };

struct MergeCase {
  std::string name;
  int32_t batch;
  int32_t query_width;
  int32_t history = 0;
  bool edges = false;
};

Candidates make_history_candidates(const MergeCase& test,
                                   int32_t shard,
                                   int32_t replica,
                                   int32_t seed) {
  CHECK_GE(test.history, 64 * 1024);
  CHECK_GT(test.batch, 0);
  CHECK_GT(test.query_width, 0);
  const int32_t rows = test.batch * test.query_width;
  torch::manual_seed(seed + shard * 101 + replica * 1009 + test.history / 1024);
  auto scores =
      torch::randn({rows, kTopk}, torch::kFloat32).to(torch::kBFloat16);
  auto query = torch::arange(rows, torch::kInt64).unsqueeze(/*dim=*/1);
  auto request = query.div(test.query_width, /*rounding_mode=*/"floor");
  auto position = query.remainder(test.query_width) + test.history;
  auto visible = position + 1;
  auto local_visible = visible.div(kStripe, /*rounding_mode=*/"floor") * kPage +
                       (visible.remainder(kStripe) - shard * kPage)
                           .clamp(/*min=*/0, /*max=*/kPage);
  auto cols = torch::arange(kTopk, torch::kInt64).unsqueeze(/*dim=*/0);
  constexpr int32_t kSpread = kTopk - kPage;
  // Spread candidates over history and retain the last local page, including
  // newly appended visible tokens. Every shard supplies exactly 2048 entries.
  auto local_pos =
      torch::where(cols < kSpread,
                   (cols * (local_visible - kPage - 1))
                       .div(kSpread - 1, /*rounding_mode=*/"floor"),
                   local_visible - kPage + cols - kSpread);
  const int64_t pages =
      (static_cast<int64_t>(test.history) + test.query_width + kStripe - 1) /
      kStripe;
  const int64_t base = replica * (test.batch * pages + 7);
  // Reverse each request's pages to exercise physical slots independently of
  // logical token order. Replicas and requests have disjoint page ranges.
  auto physical_page = base + request * pages + pages - 1 -
                       local_pos.div(kPage, /*rounding_mode=*/"floor");
  auto slots =
      (physical_page * kStripe + shard * kPage + local_pos.remainder(kPage))
          .to(torch::kInt32);
  auto mapping = ((base + request * pages + pages - 1 -
                   position.div(kStripe, /*rounding_mode=*/"floor")) *
                      kStripe +
                  position.remainder(kStripe))
                     .flatten()
                     .to(torch::kInt32);
  return {scores, slots, mapping};
}

std::vector<MergeCase> history_cases(Deployment deployment, int32_t batch) {
  std::vector<MergeCase> cases;
  cases.reserve(deployment == Deployment::PREFILL ? 3 : 6);
  for (int32_t history : {64 * 1024, 128 * 1024, 256 * 1024}) {
    if (deployment == Deployment::PREFILL) {
      cases.emplace_back(MergeCase{
          "prefill_b" + std::to_string(batch) + "_h" + std::to_string(history),
          batch,
          kQueries / batch,
          history});
      continue;
    }
    // Global batch is split equally between the two DP replicas.
    // K=3 verifies the current plus three draft tokens.
    for (int32_t width : {1, 4}) {
      const std::string stage = width == 1 ? "decode" : "mtp_verify_k3";
      cases.emplace_back(MergeCase{
          stage + "_b" + std::to_string(batch) + "_h" + std::to_string(history),
          batch / 2,
          width,
          history});
    }
  }
  return cases;
}

void log_candidate(const Candidates& input, int64_t row, int32_t slot) {
  auto slots = input.slots.select(/*dim=*/1, row).cpu().contiguous().flatten();
  auto matches = torch::nonzero(slots == slot).flatten();
  if (slot < 0 || matches.numel() == 0) {
    LOG(ERROR) << "slot=" << slot << " has no valid candidate";
    return;
  }
  const int64_t index = matches[0].item<int64_t>();
  auto scores =
      input.scores.select(/*dim=*/1, row).cpu().contiguous().flatten();
  const int16_t bf16_bits = scores.view(torch::kInt16)[index].item<int16_t>();
  const int32_t fp32_bits =
      scores.to(torch::kFloat32).view(torch::kInt32)[index].item<int32_t>();
  LOG(ERROR) << "slot=" << slot << " shard=" << index / kTopk
             << " candidate=" << index % kTopk
             << " bf16_bits=" << static_cast<uint16_t>(bf16_bits)
             << " fp32_bits=" << static_cast<uint32_t>(fp32_bits);
}

void compare_result(const TopkResult& actual,
                    const TopkResult& expected,
                    const Candidates& input,
                    int32_t rank,
                    const std::string& scenario,
                    int32_t seed) {
  for (const auto& tensor : {actual.first, actual.second}) {
    ASSERT_EQ(tensor.scalar_type(), torch::kInt32);
    ASSERT_EQ(tensor.device(), input.scores.device());
  }
  ASSERT_EQ(actual.first.sizes(), expected.first.sizes());
  ASSERT_EQ(actual.second.sizes(), expected.second.sizes());
  const std::vector<torch::Tensor> got{actual.first.cpu().contiguous(),
                                       actual.second.cpu().contiguous()};
  const std::vector<torch::Tensor> want{expected.first.cpu().contiguous(),
                                        expected.second.cpu().contiguous()};
  for (int32_t field = 0; field < 2; ++field) {
    const int64_t count = got[field].numel();
    const auto* left = got[field].data_ptr<int32_t>();
    const auto* right = want[field].data_ptr<int32_t>();
    int64_t mismatches = 0;
    int64_t bit_diff = 0;
    int64_t first = -1;
    for (int64_t i = 0; i < count; ++i) {
      const uint32_t diff =
          static_cast<uint32_t>(left[i]) ^ static_cast<uint32_t>(right[i]);
      bit_diff += std::popcount(diff);
      mismatches += diff != 0;
      if (diff != 0 && first < 0) {
        first = i;
      }
    }
    LOG(INFO) << "GLM53_DCP_MERGE rank=" << rank << " scenario=" << scenario
              << " seed=" << seed << " rows=" << input.scores.size(1)
              << " field=" << (field == 0 ? "slots" : "lengths")
              << " elements=" << count << " mismatches=" << mismatches
              << " bit_diff=" << bit_diff << " element_ratio="
              << (count == 0 ? 1.0
                             : 1.0 - static_cast<double>(mismatches) / count)
              << " bit_ratio="
              << (count == 0
                      ? 1.0
                      : 1.0 - static_cast<double>(bit_diff) / (count * 32));
    if (first < 0) {
      continue;
    }
    const int64_t row = field == 0 ? first / kTopk : first;
    LOG(ERROR) << "First difference query=" << row
               << " column=" << (field == 0 ? first % kTopk : -1)
               << " actual=" << left[first] << " expected=" << right[first];
    if (field == 0) {
      log_candidate(input, row, left[first]);
      log_candidate(input, row, right[first]);
      const auto got_row = got[0][row];
      const auto want_row = want[0][row];
      std::unordered_set<int32_t> selected(
          want_row.data_ptr<int32_t>(), want_row.data_ptr<int32_t>() + kTopk);
      int64_t set_diff = 0;
      for (int32_t col = 0; col < kTopk; ++col) {
        set_diff += selected.count(got_row.data_ptr<int32_t>()[col]) == 0;
      }
      LOG(ERROR) << "query=" << row << " selected_set_diff=" << set_diff;
    }
    ADD_FAILURE() << "rank=" << rank << " scenario=" << scenario
                  << " seed=" << seed << " field=" << field
                  << " mismatches=" << mismatches << " bit_diff=" << bit_diff;
  }
}

void check_merge(Candidates input,
                 int32_t rank,
                 const std::string& scenario,
                 int32_t seed,
                 bool strided,
                 bool full = false) {
  auto scores = input.scores;
  if (strided) {
    input.scores = strided_copy(input.scores);
    scores = input.scores;
    input.slots = strided_copy(input.slots);
    input.mapping = strided_copy(input.mapping);
    ASSERT_FALSE(scores.is_contiguous());
    ASSERT_FALSE(input.slots.is_contiguous());
    ASSERT_FALSE(input.mapping.is_contiguous());
  }
  const auto reference =
      merge_reference(input.scores, input.slots, input.mapping, kTopk);
  const int64_t rows = input.scores.size(1);
  TopkResult actual{
      torch::full({rows, kTopk}, kSentinel, input.slots.options()),
      torch::full({rows}, kSentinel, input.slots.options())};
  kernel::mlu::dcp_merge_topk(
      scores, input.slots, input.mapping, kTopk, actual.first, actual.second);
  compare_result(actual, reference, input, rank, scenario, seed);
  if (full) {
    EXPECT_TRUE(actual.second.eq(kTopk).all().item<bool>());
    EXPECT_TRUE(actual.first.ge(0).all().item<bool>());
  }
}

void run_case(int32_t rank,
              ProcessGroup& group,
              const torch::Device& device,
              const MergeCase& test,
              int32_t seed,
              int32_t replica) {
  const int32_t rows = test.batch * test.query_width;
  auto make_local = [&](int32_t shard) {
    return test.history > 0
               ? make_history_candidates(test, shard, replica, seed)
               : make_candidates(rows, shard, replica, seed, test.edges);
  };
  auto local = make_local(group.rank());
  auto scores = group.allgather_base_sync(local.scores.to(device));
  auto slots = group.allgather_base_sync(local.slots.to(device));
  ASSERT_EQ(scores.scalar_type(), torch::kBFloat16);
  ASSERT_EQ(slots.scalar_type(), torch::kInt32);
  ASSERT_EQ(scores.sizes(), torch::IntArrayRef({kDcp, rows, kTopk}));
  ASSERT_EQ(slots.sizes(), scores.sizes());
  auto cpu_scores = scores.cpu();
  auto cpu_slots = slots.cpu();
  for (int32_t shard = 0; shard < kDcp; ++shard) {
    const auto expected = make_local(shard);
    ASSERT_TRUE(torch::equal(cpu_scores[shard].view(torch::kInt16),
                             expected.scores.view(torch::kInt16)))
        << "BF16 AllGather bits shard=" << shard;
    ASSERT_TRUE(torch::equal(cpu_slots[shard], expected.slots))
        << "INT32 AllGather slots shard=" << shard;
  }
  Candidates gathered{scores, slots, local.mapping.to(device)};
  check_merge(gathered,
              rank,
              test.name,
              seed,
              /*strided=*/false,
              /*full=*/test.history > 0);
  if (test.edges) {
    check_merge(gathered, rank, "edges_strided", seed, /*strided=*/true);
    Candidates empty{
        scores.narrow(/*dim=*/1, /*start=*/0, /*length=*/0),
        slots.narrow(/*dim=*/1, /*start=*/0, /*length=*/0),
        gathered.mapping.narrow(/*dim=*/0, /*start=*/0, /*length=*/0)};
    check_merge(std::move(empty), rank, "empty", seed, /*strided=*/false);
  }
}

void run_rank(int32_t rank,
              int32_t port,
              Deployment deployment,
              const std::vector<MergeCase>& cases) {
  torch::set_num_threads(1);
  torch::NoGradGuard guard;
  Device dev(rank);
  dev.set_device();
  const auto device = dev.unwrap();
  ParallelConfig::get_instance().kv_split_size(kDcp);
  const bool prefill = deployment == Deployment::PREFILL;
  const int32_t dp = prefill ? 1 : 2;
  const int32_t pcp = prefill ? kDcp : 1;
  const int32_t tp = kRanks / (dp * pcp);
  CollectiveCommunicator comm(rank,
                              kRanks,
                              dp,
                              /*ep_size=*/kRanks,
                              pcp);
  comm.create_process_groups("127.0.0.1:" + std::to_string(port), device);
  const auto& parallel = *comm.parallel_args();
  CHECK_EQ(parallel.dp_size(), dp);
  CHECK_EQ(parallel.tp_group_->world_size(), tp);
  CHECK_EQ(parallel.tp_group_->rank(), rank % tp);
  CHECK_EQ(parallel.dcp_group_->world_size(), kDcp);
  CHECK_EQ(parallel.dcp_group_->rank(), prefill ? rank / tp : rank % tp);
  if (prefill) {
    CHECK_EQ(parallel.cp_group_->world_size(), pcp);
    CHECK_EQ(parallel.cp_group_->rank(), rank / tp);
    CHECK_EQ(parallel.dcp_group_, parallel.cp_group_);
  } else {
    CHECK_EQ(parallel.dcp_group_, parallel.tp_group_);
    CHECK_EQ(parallel.dp_local_process_group_->world_size(), dp);
    CHECK_EQ(parallel.dp_local_process_group_->rank(), rank / tp);
  }
  const int32_t replica = prefill ? rank % tp : rank / tp;
  LOG(INFO) << "GLM53_DCP_MERGE_SETUP rank=" << rank << " dp=" << dp
            << " pcp=" << pcp << " tp=" << tp << " dcp=" << kDcp
            << " ep=" << kRanks << " replica=" << replica;
  for (const auto& test : cases) {
    for (int32_t seed : {42, 2026}) {
      run_case(rank, *parallel.dcp_group_, device, test, seed, replica);
      CHECK_EQ(dev.synchronize_default_stream(), 0);
    }
  }
}

void run_distributed(Deployment deployment,
                     const std::vector<MergeCase>& cases) {
  test::run_multi_device(kRanks,
                         [deployment, &cases](int32_t rank, int32_t port) {
                           run_rank(rank, port, deployment, cases);
                         });
}

TEST(Glm53DcpMergeTopkMultiDeviceTest, Bf16CandidatesQ8192) {
  run_distributed(Deployment::PREFILL, {{"q8192", 1, kQueries}});
}

TEST(Glm53DcpMergeTopkMultiDeviceTest, Bf16EdgesAndStrides) {
  run_distributed(Deployment::PREFILL,
                  {{"edges", 1, kEdgeRows, /*history=*/0, /*edges=*/true}});
}

TEST(Glm53DcpMergeTopkMultiDeviceTest, PrefillLongHistoryBatch1) {
  run_distributed(Deployment::PREFILL,
                  history_cases(Deployment::PREFILL, /*batch=*/1));
}

TEST(Glm53DcpMergeTopkMultiDeviceTest, PrefillLongHistoryBatch8) {
  run_distributed(Deployment::PREFILL,
                  history_cases(Deployment::PREFILL, /*batch=*/8));
}

TEST(Glm53DcpMergeTopkMultiDeviceTest, DecodeDp2Batch16) {
  run_distributed(Deployment::DECODE,
                  history_cases(Deployment::DECODE, /*batch=*/16));
}

TEST(Glm53DcpMergeTopkMultiDeviceTest, DecodeDp2Batch32) {
  run_distributed(Deployment::DECODE,
                  history_cases(Deployment::DECODE, /*batch=*/32));
}

// CPU fixtures keep the parent free of MLU initialization before fork.
TEST(Glm53DcpMergeTopkReferenceTest, LongHistoryOwnershipAndCausality) {
  for (int32_t history : {64 * 1024, 128 * 1024, 256 * 1024}) {
    const MergeCase test{
        "history", /*batch=*/2, /*query_width=*/kStripe + 1, history};
    const int32_t pages = (history + test.query_width + kStripe - 1) / kStripe;
    for (int32_t replica : {0, 1}) {
      const int32_t base = replica * (test.batch * pages + 7);
      std::vector<Candidates> inputs;
      inputs.reserve(kDcp);
      for (int32_t shard = 0; shard < kDcp; ++shard) {
        inputs.emplace_back(
            make_history_candidates(test, shard, replica, /*seed=*/42));
        ASSERT_EQ(inputs.back().scores.scalar_type(), torch::kBFloat16);
        ASSERT_EQ(inputs.back().slots.scalar_type(), torch::kInt32);
        ASSERT_EQ(inputs.back().mapping.scalar_type(), torch::kInt32);
        ASSERT_TRUE(inputs.back().scores.isfinite().all().item<bool>());
      }
      for (int32_t row = 0; row < test.batch * test.query_width; ++row) {
        SCOPED_TRACE(::testing::Message()
                     << "history=" << history << " replica=" << replica
                     << " row=" << row);
        const int32_t request = row / test.query_width;
        const int32_t position = history + row % test.query_width;
        const int32_t page_start = base + request * pages;
        const int32_t new_slot =
            (page_start + pages - 1 - position / kStripe) * kStripe +
            position % kStripe;
        std::unordered_set<int32_t> positions;
        positions.reserve(kDcp * kTopk);
        for (int32_t shard = 0; shard < kDcp; ++shard) {
          ASSERT_EQ(inputs[shard].mapping[row].item<int32_t>(), new_slot);
          const auto slots = inputs[shard].slots.accessor<int32_t, 2>();
          for (int32_t col = 0; col < kTopk; ++col) {
            const int32_t slot = slots[row][col];
            ASSERT_GE(slot, 0);
            ASSERT_EQ(slot / kPage % kDcp, shard);
            ASSERT_GE(slot / kStripe, page_start);
            ASSERT_LT(slot / kStripe, page_start + pages);
            const int32_t logical =
                (pages - 1 - (slot / kStripe - page_start)) * kStripe +
                slot % kStripe;
            ASSERT_GE(logical, 0);
            ASSERT_LE(logical, position);
            ASSERT_TRUE(positions.emplace(logical).second);
          }
        }
        EXPECT_EQ(positions.size(), kDcp * kTopk);
        EXPECT_EQ(positions.count(0), 1);
        EXPECT_EQ(positions.count(position), 1);
      }
    }
  }
}

TEST(Glm53DcpMergeTopkReferenceTest, UniqueScoresLengthsAndPadding) {
  auto scores = torch::full(
      {2, 3, 3}, std::numeric_limits<float>::quiet_NaN(), torch::kBFloat16);
  auto slots = torch::arange(18, torch::kInt32).view({2, 3, 3});
  auto mapping = torch::tensor({0, -7, 2}, torch::kInt32);
  scores.select(/*dim=*/1, /*index=*/0)
      .copy_(torch::tensor(
          {{3.0f,
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN()},
           {2.0f, 1.0f, std::numeric_limits<float>::infinity()}}));
  scores.select(/*dim=*/1, /*index=*/1).fill_(9);
  scores[0][2][0].fill_(-std::numeric_limits<float>::infinity());
  const auto merged = merge_reference(scores, slots, mapping, /*topk=*/4);
  EXPECT_TRUE(torch::equal(
      merged.first,
      torch::tensor({{11, 0, 9, 10}, {-1, -1, -1, -1}, {6, -1, -1, -1}},
                    torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(merged.second, torch::tensor({4, 0, 1}, torch::kInt32)));
}

TEST(Glm53DcpMergeTopkReferenceTest, SignedValuesAndInvalidSlots) {
  auto scores =
      bf16_pattern(
          {0x8000, 0x8001, 0xff80, 0x0001, 0x7f80, 0x7fc1, 0xffc2, 0x7f80})
          .view({1, 1, 8});
  auto slots =
      torch::tensor({0, 1, 2, 3, 4, 5, 6, -9}, torch::kInt32).view({1, 1, 8});
  auto mapping = torch::zeros({1}, torch::kInt32);
  const auto merged = merge_reference(scores, slots, mapping, /*topk=*/6);
  EXPECT_TRUE(torch::equal(
      merged.first, torch::tensor({{4, 3, 0, 1, 2, -1}}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(merged.second, torch::tensor({5}, torch::kInt32)));
}

TEST(Glm53DcpMergeTopkReferenceTest, EmptyQueries) {
  const auto merged =
      merge_reference(torch::empty({kDcp, 0, kTopk}, torch::kBFloat16),
                      torch::empty({kDcp, 0, kTopk}, torch::kInt32),
                      torch::empty({0}, torch::kInt32),
                      kTopk);
  EXPECT_EQ(merged.first.scalar_type(), torch::kInt32);
  EXPECT_EQ(merged.second.scalar_type(), torch::kInt32);
  EXPECT_EQ(merged.first.sizes(), torch::IntArrayRef({0, kTopk}));
  EXPECT_EQ(merged.second.sizes(), torch::IntArrayRef({0}));
}

}  // namespace
}  // namespace xllm
