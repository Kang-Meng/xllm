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

#include "core/util/binary_payload.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <string>
#include <utility>

namespace xllm {
namespace {

butil::IOBuf make_segmented_payload() {
  butil::IOBuf payload;
  for (const std::string& block : {"abcd", "efgh", "ijkl"}) {
    char* data = new char[block.size()];
    std::copy(block.begin(), block.end(), data);
    CHECK_EQ(payload.append_user_data(
                 data,
                 block.size(),
                 [](void* data) { delete[] static_cast<char*>(data); }),
             0);
  }
  return payload;
}

TEST(BinaryPayloadTest, CopiesRangeAcrossBlocks) {
  BinaryPayload payload(make_segmented_payload());
  std::array<char, 6> output{};

  ASSERT_TRUE(payload.copy_to(/*offset=*/3, output.size(), output.data()));
  EXPECT_EQ(std::string(output.data(), output.size()), "defghi");
}

TEST(BinaryPayloadTest, CopiesRangeToString) {
  BinaryPayload payload(make_segmented_payload());
  std::string output;

  ASSERT_TRUE(payload.copy_to(/*offset=*/4, /*length=*/5, output));
  EXPECT_EQ(output, "efghi");
}

TEST(BinaryPayloadTest, ViewsRangeWithinBlock) {
  BinaryPayload payload(make_segmented_payload());

  const std::optional<std::string_view> view =
      payload.contiguous_view(/*offset=*/5, /*length=*/2);

  ASSERT_TRUE(view.has_value());
  EXPECT_EQ(*view, "fg");
}

TEST(BinaryPayloadTest, CannotViewRangeAcrossBlocks) {
  BinaryPayload payload(make_segmented_payload());

  EXPECT_FALSE(payload.contiguous_view(/*offset=*/3, /*length=*/2).has_value());
}

TEST(BinaryPayloadTest, ViewsEmptyValidRange) {
  BinaryPayload payload(make_segmented_payload());

  const std::optional<std::string_view> view =
      payload.contiguous_view(payload.size(), /*length=*/0);

  ASSERT_TRUE(view.has_value());
  EXPECT_TRUE(view->empty());
}

TEST(BinaryPayloadTest, RejectsInvalidViewRange) {
  BinaryPayload payload(make_segmented_payload());

  EXPECT_FALSE(
      payload.contiguous_view(/*offset=*/11, /*length=*/2).has_value());
  EXPECT_FALSE(
      payload.contiguous_view(/*offset=*/13, /*length=*/0).has_value());
}

TEST(BinaryPayloadTest, RejectsInvalidRange) {
  BinaryPayload payload(make_segmented_payload());
  std::string output = "unchanged";

  EXPECT_FALSE(payload.copy_to(/*offset=*/11, /*length=*/2, output));
  EXPECT_EQ(output, "unchanged");
  EXPECT_FALSE(payload.copy_to(/*offset=*/13, /*length=*/0, output));
}

TEST(BinaryPayloadTest, KeepsBlocksAliveAfterSourceRelease) {
  butil::IOBuf source = make_segmented_payload();
  BinaryPayload payload(source);
  source.clear();
  std::string output;

  ASSERT_TRUE(payload.copy_to(/*offset=*/0, payload.size(), output));
  EXPECT_EQ(output, "abcdefghijkl");
}

TEST(BinaryPayloadTest, MoveTransfersBlocks) {
  BinaryPayload source(make_segmented_payload());
  BinaryPayload payload(std::move(source));
  std::string output;

  EXPECT_TRUE(source.empty());
  ASSERT_TRUE(payload.copy_to(/*offset=*/0, payload.size(), output));
  EXPECT_EQ(output, "abcdefghijkl");

  BinaryPayload assigned;
  assigned = std::move(payload);
  EXPECT_TRUE(payload.empty());
  ASSERT_TRUE(assigned.copy_to(/*offset=*/0, assigned.size(), output));
  EXPECT_EQ(output, "abcdefghijkl");
}

}  // namespace
}  // namespace xllm
