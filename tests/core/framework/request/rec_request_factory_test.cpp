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

#include "framework/request/rec_request_factory.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/options.h"
#include "common/rate_limiter.h"
#include "core/common/types.h"
#include "framework/model/model_args.h"
#include "framework/request/rec_type.h"
#include "framework/request/request_output.h"
#include "framework/request/request_params.h"
#include "framework/tokenizer/tokenizer.h"
#include "rec.pb.h"
#include "util/rec_model_utils.h"

namespace xllm {
namespace {

// Deterministic tokenizer for factory tests. Encoding fails for any text that
// contains the marker "FAIL", so tokenize/stop-sequence error paths can be
// exercised independently.
class FakeTokenizer final : public Tokenizer {
 public:
  explicit FakeTokenizer(int32_t vocab_size) : vocab_size_(vocab_size) {}

  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool /*add_special_tokens*/ = true) const override {
    if (text.find("FAIL") != std::string_view::npos) {
      return false;
    }
    ids->clear();
    for (const char c : text) {
      ids->push_back(static_cast<int32_t>(static_cast<unsigned char>(c)) %
                     vocab_size_);
    }
    if (ids->empty()) {
      ids->push_back(1);
    }
    return true;
  }

  size_t vocab_size() const override {
    return static_cast<size_t>(vocab_size_);
  }

  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeTokenizer>(*this);
  }

 private:
  int32_t vocab_size_ = 1000;
};

// Records the last error surfaced through the OutputCallback.
struct CallbackCapture {
  bool called = false;
  std::optional<Status> status;
};

OutputCallback make_capture_callback(CallbackCapture* capture) {
  return [capture](RequestOutput output) {
    capture->called = true;
    capture->status = output.status;
    return false;
  };
}

class RecRequestFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Simulate the caller (service entry) having acquired a rate-limit slot.
    rate_limiter_.is_limited();
    ASSERT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
  }

  void configure_model(int32_t vocab_size, int32_t max_position) {
    model_args_.vocab_size(vocab_size)
        .max_position_embeddings(max_position)
        .hidden_size(4)
        .eos_token_id(-1)
        .bos_token_id(0);
    // enable_chunked_prefill defaults true; keep it so the prompt length limit
    // is exactly max_position_embeddings.
    options_.enable_service_routing(false).num_speculative_tokens(0);
  }

  std::unique_ptr<RecRequestFactory> make_llmrec_factory(
      int32_t vocab_size = 1000,
      int32_t max_position = 2048) {
    configure_model(vocab_size, max_position);
    tokenizer_ = std::make_unique<FakeTokenizer>(vocab_size);
    return std::make_unique<RecRequestFactory>(&model_args_,
                                               tokenizer_.get(),
                                               &options_,
                                               &rate_limiter_,
                                               RecType::kLlmRec,
                                               RecPipelineType::kLlmRecDefault);
  }

  std::unique_ptr<RecRequestFactory> make_onerec_factory(
      int32_t max_position = 2048) {
    configure_model(/*vocab_size=*/1000, max_position);
    // OneRec models do not require a tokenizer for the input paths tested here.
    return std::make_unique<RecRequestFactory>(&model_args_,
                                               /*tokenizer=*/nullptr,
                                               &options_,
                                               &rate_limiter_,
                                               RecType::kOneRec,
                                               RecPipelineType::kOneRecDefault);
  }

  std::unique_ptr<FakeTokenizer> tokenizer_;
  ModelArgs model_args_;
  Options options_;
  RateLimiter rate_limiter_;
};

// -------------------------- LlmRec: prompt overload -------------------------

TEST_F(RecRequestFactoryTest, LlmRecRejectsEmptyInputReleasesRateLimitSlot) {
  auto factory = make_llmrec_factory();
  CallbackCapture capture;
  RequestParams sp;

  auto request = factory->create(/*prompt=*/"",
                                 /*prompt_tokens=*/std::nullopt,
                                 /*input_tensors=*/std::nullopt,
                                 sp,
                                 make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.called);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(capture.status->message(),
            "LlmRec requires prompt or prompt_tokens to be provided");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(RecRequestFactoryTest, LlmRecFailsWhenPromptTokenizationFails) {
  auto factory = make_llmrec_factory();
  CallbackCapture capture;
  RequestParams sp;

  auto request = factory->create(/*prompt=*/"please FAIL here",
                                 /*prompt_tokens=*/std::nullopt,
                                 /*input_tensors=*/std::nullopt,
                                 sp,
                                 make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->message(), "Failed to tokenize prompt");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(RecRequestFactoryTest, LlmRecRejectsPromptLongerThanContext) {
  auto factory = make_llmrec_factory(/*vocab_size=*/1000, /*max_position=*/8);
  CallbackCapture capture;
  RequestParams sp;

  auto request = factory->create(
      /*prompt=*/"",
      /*prompt_tokens=*/std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8},
      /*input_tensors=*/std::nullopt,
      sp,
      make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->message(), "Prompt is too long");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(RecRequestFactoryTest, LlmRecFailsWhenStopSequenceEncodingFails) {
  auto factory = make_llmrec_factory();
  CallbackCapture capture;
  RequestParams sp;
  sp.stop = std::vector<std::string>{"FAIL-STOP"};

  auto request = factory->create(/*prompt=*/"",
                                 /*prompt_tokens=*/std::vector<int>{1, 2, 3},
                                 /*input_tensors=*/std::nullopt,
                                 sp,
                                 make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->message(), "Failed to encode stop sequence");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(RecRequestFactoryTest, LlmRecCreatesRequestForValidPromptTokens) {
  auto factory = make_llmrec_factory();
  CallbackCapture capture;
  RequestParams sp;
  sp.request_id = "req-tokens";
  sp.max_tokens = 16;

  auto request = factory->create(/*prompt=*/"",
                                 /*prompt_tokens=*/std::vector<int>{1, 2, 3},
                                 /*input_tensors=*/std::nullopt,
                                 sp,
                                 make_capture_callback(&capture));

  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(capture.called);
  // On success the slot stays held; released later by the scheduler.
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
}

TEST_F(RecRequestFactoryTest, LlmRecCreatesRequestForValidPromptString) {
  auto factory = make_llmrec_factory();
  CallbackCapture capture;
  RequestParams sp;
  sp.request_id = "req-prompt";
  sp.max_tokens = 16;

  auto request = factory->create(/*prompt=*/"hello world",
                                 /*prompt_tokens=*/std::nullopt,
                                 /*input_tensors=*/std::nullopt,
                                 sp,
                                 make_capture_callback(&capture));

  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(capture.called);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
}

// ------------------------- LlmRec: mm_data overload -------------------------

TEST_F(RecRequestFactoryTest, LlmRecMmDataOverloadCreatesRequestWithoutMmData) {
  auto factory = make_llmrec_factory();
  CallbackCapture capture;
  RequestParams sp;
  sp.request_id = "req-mm";
  sp.max_tokens = 16;

  auto request = factory->create(/*prompt_tokens=*/std::vector<int>{1, 2, 3},
                                 /*mm_data=*/std::nullopt,
                                 sp,
                                 make_capture_callback(&capture));

  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(capture.called);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
}

// -------------------------- OneRec: prompt overload -------------------------

TEST_F(RecRequestFactoryTest, OneRecRejectsBothPromptTokensAndInputTensors) {
  auto factory = make_onerec_factory();
  CallbackCapture capture;
  RequestParams sp;

  auto request = factory->create(
      /*prompt=*/"",
      /*prompt_tokens=*/std::vector<int>{1},
      /*input_tensors=*/std::vector<proto::InferInputTensor>{},
      sp,
      make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->message(),
            "prompt_tokens and input_tensors cannot both be set");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(RecRequestFactoryTest, OneRecRejectsEmptyInput) {
  auto factory = make_onerec_factory();
  CallbackCapture capture;
  RequestParams sp;

  auto request = factory->create(/*prompt=*/"",
                                 /*prompt_tokens=*/std::nullopt,
                                 /*input_tensors=*/std::nullopt,
                                 sp,
                                 make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->message(),
            "Rec model requires prompt_tokens or input_tensors to be provided");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(RecRequestFactoryTest, OneRecRejectsEmptyInputTensors) {
  auto factory = make_onerec_factory();
  CallbackCapture capture;
  RequestParams sp;

  auto request = factory->create(
      /*prompt=*/"",
      /*prompt_tokens=*/std::nullopt,
      /*input_tensors=*/std::vector<proto::InferInputTensor>{},
      sp,
      make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->message(), "OneRec input_tensors cannot be empty");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(RecRequestFactoryTest, OneRecCreatesRequestForPromptTokens) {
  auto factory = make_onerec_factory();
  CallbackCapture capture;
  RequestParams sp;
  sp.request_id = "req-onerec";
  sp.max_tokens = 16;

  auto request = factory->create(/*prompt=*/"",
                                 /*prompt_tokens=*/std::vector<int>{1, 2, 3},
                                 /*input_tensors=*/std::nullopt,
                                 sp,
                                 make_capture_callback(&capture));

  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(capture.called);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
}

}  // namespace
}  // namespace xllm
