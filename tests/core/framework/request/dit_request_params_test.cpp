/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/request/dit_request_params.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "audio_generation.pb.h"
#include "butil/base64.h"
#include "core/framework/multimodal/mm_codec.h"
#include "core/util/binary_payload.h"
#include "image_generation.pb.h"
#include "models/model_registry.h"
#include "text_generation.pb.h"
#include "video_generation.pb.h"

namespace xllm {
namespace {

// Helper to build a minimal TextGenerationRequest.
proto::TextGenerationRequest MakeTextRequest(const std::string& model,
                                             const std::string& prompt) {
  proto::TextGenerationRequest request;
  request.set_model(model);
  auto* input = request.mutable_input();
  input->set_prompt(prompt);
  return request;
}

constexpr char kTinyPngBase64[] =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVQIHWP4z8AAAAMB"
    "AQBb2/lEAAAAAElFTkSuQmCC";

const std::string& TinyPngBytes() {
  static const std::string bytes = [] {
    std::string result;
    CHECK(butil::Base64Decode(kTinyPngBase64, &result));
    return result;
  }();
  return bytes;
}

void AppendUint16Le(uint16_t value, std::string& bytes) {
  bytes.push_back(static_cast<char>(value & 0xff));
  bytes.push_back(static_cast<char>((value >> 8) & 0xff));
}

void AppendUint32Le(uint32_t value, std::string& bytes) {
  bytes.push_back(static_cast<char>(value & 0xff));
  bytes.push_back(static_cast<char>((value >> 8) & 0xff));
  bytes.push_back(static_cast<char>((value >> 16) & 0xff));
  bytes.push_back(static_cast<char>((value >> 24) & 0xff));
}

const std::string& TinyWavBytes() {
  static const std::string bytes = [] {
    constexpr uint32_t kSampleRate = 24000;
    constexpr uint16_t kChannelCount = 1;
    constexpr uint16_t kBitsPerSample = 16;
    constexpr int16_t kSamples[] = {0, 1024, -1024, 0};
    constexpr uint32_t kDataSize = sizeof(kSamples);

    std::string result;
    result.reserve(44 + kDataSize);
    result.append("RIFF", 4);
    AppendUint32Le(36 + kDataSize, result);
    result.append("WAVEfmt ", 8);
    AppendUint32Le(16, result);
    AppendUint16Le(/*value=*/1, result);
    AppendUint16Le(kChannelCount, result);
    AppendUint32Le(kSampleRate, result);
    AppendUint32Le(kSampleRate * kChannelCount * kBitsPerSample / 8, result);
    AppendUint16Le(kChannelCount * kBitsPerSample / 8, result);
    AppendUint16Le(kBitsPerSample, result);
    result.append("data", 4);
    AppendUint32Le(kDataSize, result);
    for (const int16_t sample : kSamples) {
      AppendUint16Le(static_cast<uint16_t>(sample), result);
    }
    return result;
  }();
  return bytes;
}

proto::AudioGenerationRequest MakeAudioRequest() {
  proto::AudioGenerationRequest request;
  request.set_model("LongCat-AudioDiT");
  request.mutable_input()->set_prompt("generate speech");
  request.mutable_parameters()->set_sampling_rate(24000);
  return request;
}

proto::ImageGenerationRequest MakeImageRequest() {
  proto::ImageGenerationRequest request;
  request.set_model("QwenImageEditPlus");
  request.mutable_input()->set_prompt("edit image");
  return request;
}

proto::VideoGenerationRequest MakeVideoRequest() {
  proto::VideoGenerationRequest request;
  request.set_model("WanI2V");
  request.mutable_input()->set_prompt("animate image");
  return request;
}

const DiTInputParams& decode_input(DiTRequestParams& params) {
  Status status;
  EXPECT_TRUE(params.parse_input_params(status));
  return params.input_params;
}

bool verify_fails_with_invalid_argument(DiTRequestParams& params) {
  Status parse_status;
  if (!params.parse_input_params(parse_status)) {
    return parse_status.code() == StatusCode::INVALID_ARGUMENT;
  }
  Status status;
  const bool valid = params.verify_params(status);
  return !valid && status.code() == StatusCode::INVALID_ARGUMENT;
}

bool decode_fails_with_invalid_argument(DiTRequestParams& params) {
  Status status;
  const bool decoded = params.parse_input_params(status);
  return !decoded && status.code() == StatusCode::INVALID_ARGUMENT;
}

TEST(DiTRequestParamsTest, LegacyImagesMapToOrderedUnknownSources) {
  proto::ImageGenerationRequest request = MakeImageRequest();
  request.mutable_input()->add_images(kTinyPngBase64);
  request.mutable_input()->add_images(kTinyPngBase64);

  DiTRequestParams params(request, "rid", "rtime");

  ASSERT_TRUE(params.request_parse_status.ok());
  const DiTInputParams& input_params = decode_input(params);
  ASSERT_EQ(input_params.media_sources.size(), 2u);
  EXPECT_EQ(input_params.media_sources.at(0).name, "unknown");
  EXPECT_EQ(input_params.media_sources.at(1).name, "unknown");
  EXPECT_EQ(input_params.media_sources.at(0).modality, "image");
  EXPECT_TRUE(torch::equal(input_params.media_sources.at(0).tensor,
                           input_params.media_sources.at(1).tensor));
}

TEST(DiTRequestParamsTest, ConstructorCollectsMediaWithoutDecoding) {
  proto::ImageGenerationRequest request = MakeImageRequest();
  proto::MediaSource* source = request.mutable_input()->add_media_sources();
  source->set_type("base64");
  source->set_name("image");
  source->set_modality("image");
  source->set_base64("not-base64");

  DiTRequestParams params(request, "rid", "rtime");

  ASSERT_TRUE(params.request_parse_status.ok());
  const auto* input =
      std::get_if<const proto::Input*>(&params.raw_input_params.input);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(*input, nullptr);
  ASSERT_EQ((*input)->media_sources_size(), 1);
  EXPECT_EQ((*input)->media_sources(0).base64(), "not-base64");
}

TEST(DiTRequestParamsTest, ImageSourcesSupportBase64AndBinaryInOrder) {
  proto::ImageGenerationRequest request = MakeImageRequest();
  proto::Input* input = request.mutable_input();
  proto::MediaSource* base64 = input->add_media_sources();
  base64->set_type("base64");
  base64->set_name("unknown");
  base64->set_modality("image");
  base64->set_base64(kTinyPngBase64);
  proto::MediaSource* binary = input->add_media_sources();
  binary->set_type("binary");
  binary->set_name("control_image");
  binary->set_modality("image");
  binary->mutable_binary()->set_offset(2);
  binary->mutable_binary()->set_length(TinyPngBytes().size());
  const std::string payload = "xx" + TinyPngBytes() + "trailing bytes";

  DiTRequestParams params(request, "rid", "rtime", BinaryPayload(payload));

  ASSERT_TRUE(params.request_parse_status.ok());
  const DiTInputParams& input_params = decode_input(params);
  ASSERT_EQ(input_params.media_sources.size(), 2u);
  EXPECT_EQ(input_params.media_sources.at(0).name, "unknown");
  EXPECT_EQ(input_params.media_sources.at(1).name, "control_image");
  EXPECT_TRUE(torch::equal(input_params.media_sources.at(0).tensor,
                           input_params.media_sources.at(1).tensor));
}

TEST(DiTRequestParamsTest, VideoImageSourcesAreDecoded) {
  proto::VideoGenerationRequest request = MakeVideoRequest();
  proto::VideoInput* input = request.mutable_input();
  const size_t image_size = TinyPngBytes().size();
  proto::MediaSource* first = input->add_media_sources();
  first->set_type("binary");
  first->set_name("image");
  first->set_modality("image");
  first->mutable_binary()->set_length(image_size);
  proto::MediaSource* last = input->add_media_sources();
  last->set_type("binary");
  last->set_name("last_image");
  last->set_modality("image");
  last->mutable_binary()->set_offset(image_size);
  last->mutable_binary()->set_length(image_size);
  const std::string payload = TinyPngBytes() + TinyPngBytes();

  DiTRequestParams params(request, "rid", "rtime", BinaryPayload(payload));

  ASSERT_TRUE(params.request_parse_status.ok());
  std::vector<torch::Tensor> images =
      decode_input(params).media_sources.get({"image", "last_image"});
  EXPECT_EQ(images.size(), 2u);
}

TEST(DiTRequestParamsTest,
     MixedImageSourcesAndLegacyImagesPreserveExistingOrder) {
  proto::ImageGenerationRequest request = MakeImageRequest();
  proto::Input* input = request.mutable_input();
  proto::MediaSource* base64 = input->add_media_sources();
  base64->set_type("base64");
  base64->set_name("first");
  base64->set_modality("image");
  base64->set_base64(kTinyPngBase64);
  proto::MediaSource* binary = input->add_media_sources();
  binary->set_type("binary");
  binary->set_name("second");
  binary->set_modality("image");
  binary->mutable_binary()->set_length(TinyPngBytes().size());
  input->add_images(kTinyPngBase64);

  DiTRequestParams params(
      request, "rid", "rtime", BinaryPayload(TinyPngBytes()));

  ASSERT_TRUE(params.request_parse_status.ok());
  const DiTInputParams& input_params = decode_input(params);
  ASSERT_EQ(input_params.media_sources.size(), 3u);
  EXPECT_EQ(input_params.media_sources.at(0).name, "first");
  EXPECT_EQ(input_params.media_sources.at(1).name, "second");
  EXPECT_EQ(input_params.media_sources.at(2).name, "unknown");
  EXPECT_TRUE(torch::equal(input_params.media_sources.at(0).tensor,
                           input_params.media_sources.at(1).tensor));
  EXPECT_TRUE(torch::equal(input_params.media_sources.at(1).tensor,
                           input_params.media_sources.at(2).tensor));
}

TEST(DiTRequestParamsTest, FailedParallelImageDecodePreservesExistingInput) {
  proto::ImageGenerationRequest request = MakeImageRequest();
  proto::Input* input = request.mutable_input();
  proto::MediaSource* valid = input->add_media_sources();
  valid->set_type("base64");
  valid->set_name("valid");
  valid->set_modality("image");
  valid->set_base64(kTinyPngBase64);
  proto::MediaSource* invalid = input->add_media_sources();
  invalid->set_type("base64");
  invalid->set_name("invalid");
  invalid->set_modality("image");
  std::string invalid_base64;
  butil::Base64Encode("not an image", &invalid_base64);
  invalid->set_base64(invalid_base64);

  DiTRequestParams params(request, "rid", "rtime");
  ASSERT_TRUE(params.request_parse_status.ok());
  params.input_params.prompt = "existing prompt";
  const torch::Tensor existing_tensor = torch::ones({1});
  params.input_params.media_sources.add("existing", "image", existing_tensor);
  Status status;

  EXPECT_FALSE(params.parse_input_params(status));
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(params.input_params.prompt, "existing prompt");
  ASSERT_EQ(params.input_params.media_sources.size(), 1);
  EXPECT_TRUE(params.input_params.media_sources.contains("existing"));
  EXPECT_TRUE(torch::equal(params.input_params.media_sources.at(0).tensor,
                           existing_tensor));
}

TEST(DiTRequestParamsTest, DuplicateImageSourceNamesArePreserved) {
  proto::ImageGenerationRequest request = MakeImageRequest();
  proto::Input* input = request.mutable_input();
  for (int32_t index = 0; index < 2; ++index) {
    proto::MediaSource* source = input->add_media_sources();
    source->set_type("base64");
    source->set_name("image");
    source->set_modality("image");
    source->set_base64(kTinyPngBase64);
  }

  DiTRequestParams params(request, "rid", "rtime");

  ASSERT_TRUE(params.request_parse_status.ok());
  const DiTInputParams& input_params = decode_input(params);
  EXPECT_EQ(input_params.media_sources.get({"image", "image"}).size(), 2u);
}

TEST(DiTRequestParamsTest, RejectsOutOfBoundsBinaryImage) {
  proto::ImageGenerationRequest request = MakeImageRequest();
  proto::MediaSource* source = request.mutable_input()->add_media_sources();
  source->set_type("binary");
  source->set_name("image");
  source->set_modality("image");
  proto::BinaryRef* binary = source->mutable_binary();
  binary->set_offset(TinyPngBytes().size());
  binary->set_length(1);

  DiTRequestParams params(
      request, "rid", "rtime", BinaryPayload(TinyPngBytes()));

  ASSERT_TRUE(params.request_parse_status.ok());
  EXPECT_TRUE(decode_fails_with_invalid_argument(params));
}

TEST(DiTRequestParamsTest, RejectsOverflowingBinaryImageRange) {
  proto::ImageGenerationRequest request = MakeImageRequest();
  proto::MediaSource* source = request.mutable_input()->add_media_sources();
  source->set_type("binary");
  source->set_name("image");
  source->set_modality("image");
  proto::BinaryRef* binary = source->mutable_binary();
  binary->set_offset(1);
  binary->set_length(std::numeric_limits<uint64_t>::max());

  DiTRequestParams params(
      request, "rid", "rtime", BinaryPayload(TinyPngBytes()));

  ASSERT_TRUE(params.request_parse_status.ok());
  EXPECT_TRUE(decode_fails_with_invalid_argument(params));
}

TEST(DiTRequestParamsTest, RejectsEmptyImageSource) {
  proto::ImageGenerationRequest request = MakeImageRequest();
  request.mutable_input()->add_media_sources();

  DiTRequestParams params(request, "rid", "rtime");

  ASSERT_TRUE(params.request_parse_status.ok());
  EXPECT_TRUE(decode_fails_with_invalid_argument(params));
}

TEST(DiTRequestParamsTest, BinaryTensorInputMatchesContentsInput) {
  proto::ImageGenerationRequest contents_request = MakeImageRequest();
  proto::Tensor* contents_tensor =
      contents_request.mutable_input()->mutable_prompt_embed();
  contents_tensor->set_datatype("FP32");
  contents_tensor->add_shape(2);
  contents_tensor->mutable_contents()->add_fp32_contents(1.25f);
  contents_tensor->mutable_contents()->add_fp32_contents(-2.5f);
  DiTRequestParams contents_params(contents_request, "rid", "rtime");

  proto::ImageGenerationRequest binary_request = MakeImageRequest();
  proto::Tensor* binary_tensor =
      binary_request.mutable_input()->mutable_prompt_embed();
  binary_tensor->set_datatype("FP32");
  binary_tensor->add_shape(2);
  (*binary_tensor->mutable_parameters())["is_binary"].set_bool_param(true);
  (*binary_tensor->mutable_parameters())["offset"].set_int64_param(2);
  (*binary_tensor->mutable_parameters())["len"].set_int64_param(2 *
                                                                sizeof(float));
  const float values[] = {1.25f, -2.5f};
  std::string payload = "xx";
  payload.append(reinterpret_cast<const char*>(values), sizeof(values));
  DiTRequestParams binary_params(
      binary_request, "rid", "rtime", BinaryPayload(payload));

  ASSERT_TRUE(binary_params.request_parse_status.ok());
  const DiTInputParams& binary_input = decode_input(binary_params);
  const DiTInputParams& contents_input = decode_input(contents_params);
  EXPECT_TRUE(torch::equal(*binary_input.tensor_sources.get("prompt_embed"),
                           *contents_input.tensor_sources.get("prompt_embed")));
}

TEST(DiTRequestParamsTest, VideoBinaryTensorInputMatchesContentsInput) {
  proto::VideoGenerationRequest contents_request = MakeVideoRequest();
  proto::Tensor* contents_tensor =
      contents_request.mutable_input()->mutable_prompt_embed();
  contents_tensor->set_datatype("FP32");
  contents_tensor->add_shape(2);
  contents_tensor->mutable_contents()->add_fp32_contents(1.25f);
  contents_tensor->mutable_contents()->add_fp32_contents(-2.5f);
  DiTRequestParams contents_params(contents_request, "rid", "rtime");

  proto::VideoGenerationRequest binary_request = MakeVideoRequest();
  proto::Tensor* binary_tensor =
      binary_request.mutable_input()->mutable_prompt_embed();
  binary_tensor->set_datatype("FP32");
  binary_tensor->add_shape(2);
  (*binary_tensor->mutable_parameters())["is_binary"].set_bool_param(true);
  (*binary_tensor->mutable_parameters())["offset"].set_int64_param(2);
  (*binary_tensor->mutable_parameters())["len"].set_int64_param(2 *
                                                                sizeof(float));
  const float values[] = {1.25f, -2.5f};
  std::string payload = "xx";
  payload.append(reinterpret_cast<const char*>(values), sizeof(values));
  DiTRequestParams binary_params(
      binary_request, "rid", "rtime", BinaryPayload(payload));

  ASSERT_TRUE(binary_params.request_parse_status.ok());
  const DiTInputParams& binary_input = decode_input(binary_params);
  const DiTInputParams& contents_input = decode_input(contents_params);
  EXPECT_TRUE(torch::equal(*binary_input.tensor_sources.get("prompt_embed"),
                           *contents_input.tensor_sources.get("prompt_embed")));
}

TEST(DiTRequestParamsTest, PromptTokenTagsDecodeFromPromptEmbedParameters) {
  proto::VideoGenerationRequest request = MakeVideoRequest();
  proto::VideoInput* input = request.mutable_input();
  proto::Tensor* prompt_embed = input->mutable_prompt_embed();
  prompt_embed->set_datatype("FP32");
  prompt_embed->add_shape(2);
  prompt_embed->add_shape(2);
  for (int32_t index = 0; index < 4; ++index) {
    prompt_embed->mutable_contents()->add_fp32_contents(
        static_cast<float>(index));
  }
  proto::Parameter& prompt_token_tags =
      (*prompt_embed->mutable_parameters())["prompt_token_tags"];
  prompt_token_tags.add_int64_tensor_param(0);
  prompt_token_tags.add_int64_tensor_param(1);

  DiTRequestParams params(request, "rid", "rtime");

  ASSERT_TRUE(params.request_parse_status.ok());
  const DiTInputParams& input_params = decode_input(params);
  const std::optional<NamedTensorConstRef> parsed_prompt_embed =
      input_params.tensor_sources.get_named_tensor("prompt_embed");
  ASSERT_TRUE(parsed_prompt_embed.has_value());
  const std::vector<int64_t>* tags = get_tensor_parameter<std::vector<int64_t>>(
      parsed_prompt_embed->get().parameters, "prompt_token_tags");
  ASSERT_NE(tags, nullptr);
  EXPECT_EQ(*tags, std::vector<int64_t>({0, 1}));
}

TEST(DiTRequestParamsTest, InvalidResolutionReturnsInvalidArgument) {
  for (const std::string& size : {"invalid",
                                  "1x1",
                                  "1*",
                                  "*1",
                                  "-1*1",
                                  "1*-1",
                                  "1*1*1",
                                  "2147483648*1",
                                  "1*2147483648"}) {
    proto::ImageGenerationRequest request = MakeImageRequest();
    request.mutable_parameters()->set_size(size);

    DiTRequestParams params(request, "rid", "rtime");

    EXPECT_TRUE(verify_fails_with_invalid_argument(params)) << size;
  }
}

TEST(DiTRequestParamsTest, ImageOutputTypeDefaultsAndValidates) {
  proto::ImageGenerationRequest default_request = MakeImageRequest();
  DiTRequestParams default_params(default_request, "rid", "rtime");
  EXPECT_EQ(default_params.generation_params.output_type, "base64");

  proto::ImageGenerationRequest binary_request = MakeImageRequest();
  binary_request.mutable_parameters()->set_output_type("binary");
  DiTRequestParams binary_params(binary_request, "rid", "rtime");
  EXPECT_EQ(binary_params.generation_params.output_type, "binary");
  EXPECT_TRUE(binary_params.request_parse_status.ok());

  proto::ImageGenerationRequest invalid_request = MakeImageRequest();
  invalid_request.mutable_parameters()->set_output_type("url");
  DiTRequestParams invalid_params(invalid_request, "rid", "rtime");
  EXPECT_TRUE(invalid_params.request_parse_status.ok());
  EXPECT_TRUE(verify_fails_with_invalid_argument(invalid_params));
}

TEST(DiTRequestParamsTest, ImageGenerationCountValidatesAfterParsing) {
  proto::ImageGenerationRequest request = MakeImageRequest();
  request.mutable_parameters()->set_num_images_per_prompt(-1);

  DiTRequestParams params(request, "rid", "rtime");

  EXPECT_TRUE(params.request_parse_status.ok());
  EXPECT_EQ(params.generation_params.num_images_per_prompt, -1);
  EXPECT_TRUE(verify_fails_with_invalid_argument(params));
}

TEST(DiTRequestParamsTest, VideoOutputTypeDefaultsAndValidates) {
  proto::VideoGenerationRequest default_request = MakeVideoRequest();
  DiTRequestParams default_params(default_request, "rid", "rtime");
  EXPECT_EQ(default_params.generation_params.output_type, "base64");

  proto::VideoGenerationRequest binary_request = MakeVideoRequest();
  binary_request.mutable_parameters()->set_output_type("binary");
  DiTRequestParams binary_params(binary_request, "rid", "rtime");
  EXPECT_EQ(binary_params.generation_params.output_type, "binary");
  EXPECT_TRUE(binary_params.request_parse_status.ok());

  proto::VideoGenerationRequest invalid_request = MakeVideoRequest();
  invalid_request.mutable_parameters()->set_output_type("url");
  DiTRequestParams invalid_params(invalid_request, "rid", "rtime");
  EXPECT_TRUE(invalid_params.request_parse_status.ok());
  EXPECT_TRUE(verify_fails_with_invalid_argument(invalid_params));
}

TEST(DiTRequestParamsTest, VideoGenerationCountValidatesAfterParsing) {
  proto::VideoGenerationRequest request = MakeVideoRequest();
  request.mutable_parameters()->set_num_videos_per_prompt(-1);

  DiTRequestParams params(request, "rid", "rtime");

  EXPECT_TRUE(params.request_parse_status.ok());
  EXPECT_EQ(params.generation_params.num_videos_per_prompt, -1);
  EXPECT_TRUE(verify_fails_with_invalid_argument(params));
}

TEST(DiTRequestParamsTest, PromptAudioUsesCanonicalSourceIdentity) {
  std::string wav_base64;
  butil::Base64Encode(TinyWavBytes(), &wav_base64);
  proto::AudioGenerationRequest request = MakeAudioRequest();
  proto::MediaSource* source = request.mutable_input()->mutable_prompt_audio();
  source->set_type("base64");
  source->set_name("ignored-name");
  source->set_modality("image");
  source->set_base64(wav_base64);
  const std::string original_request = request.SerializeAsString();

  DiTRequestParams params(request, "rid", "rtime");
  const DiTInputParams& input_params = decode_input(params);

  ASSERT_EQ(input_params.media_sources.size(), 1u);
  EXPECT_EQ(input_params.media_sources.at(0).name, "prompt_audio");
  EXPECT_EQ(input_params.media_sources.at(0).modality, "audio");
  EXPECT_EQ(request.SerializeAsString(), original_request);
}

TEST(DiTRequestParamsTest, BinaryPromptAudioUsesCanonicalSourceIdentity) {
  proto::AudioGenerationRequest request = MakeAudioRequest();
  proto::MediaSource* source = request.mutable_input()->mutable_prompt_audio();
  source->set_type("binary");
  source->set_name("ignored-name");
  source->set_modality("image");
  source->mutable_binary()->set_offset(2);
  source->mutable_binary()->set_length(TinyWavBytes().size());
  const std::string original_request = request.SerializeAsString();

  DiTRequestParams params(
      request, "rid", "rtime", BinaryPayload("xx" + TinyWavBytes()));
  const DiTInputParams& input_params = decode_input(params);

  ASSERT_EQ(input_params.media_sources.size(), 1u);
  EXPECT_EQ(input_params.media_sources.at(0).name, "prompt_audio");
  EXPECT_EQ(input_params.media_sources.at(0).modality, "audio");
  EXPECT_EQ(request.SerializeAsString(), original_request);
}

TEST(DiTRequestParamsTest, VideoDefaultAudioSamplingRateIs32000) {
  proto::VideoGenerationRequest request = MakeVideoRequest();

  DiTRequestParams params(request, "rid", "rtime");

  EXPECT_EQ(params.generation_params.audio_sampling_rate, 32000);
}

TEST(DiTRequestParamsTest, PromptAudioSupportsBase64AndBinary) {
  std::string wav_base64;
  butil::Base64Encode(TinyWavBytes(), &wav_base64);

  proto::AudioGenerationRequest base64_request = MakeAudioRequest();
  proto::MediaSource* base64_source =
      base64_request.mutable_input()->mutable_prompt_audio();
  base64_source->set_type("base64");
  base64_source->set_base64(wav_base64);
  const std::string original_base64_request =
      base64_request.SerializeAsString();
  DiTRequestParams base64_params(base64_request, "rid", "rtime");

  proto::AudioGenerationRequest binary_request = MakeAudioRequest();
  proto::MediaSource* binary_source =
      binary_request.mutable_input()->mutable_prompt_audio();
  binary_source->set_type("binary");
  binary_source->mutable_binary()->set_offset(2);
  binary_source->mutable_binary()->set_length(TinyWavBytes().size());
  const std::string original_binary_request =
      binary_request.SerializeAsString();
  DiTRequestParams binary_params(
      binary_request, "rid", "rtime", BinaryPayload("xx" + TinyWavBytes()));

  ASSERT_TRUE(base64_params.request_parse_status.ok());
  ASSERT_TRUE(binary_params.request_parse_status.ok());
  const DiTInputParams& base64_input = decode_input(base64_params);
  const DiTInputParams& binary_input = decode_input(binary_params);
  ASSERT_EQ(base64_input.media_sources.size(), 1u);
  ASSERT_EQ(binary_input.media_sources.size(), 1u);
  EXPECT_EQ(base64_input.media_sources.at(0).name, "prompt_audio");
  EXPECT_EQ(binary_input.media_sources.at(0).name, "prompt_audio");
  EXPECT_EQ(base64_input.media_sources.at(0).modality, "audio");
  EXPECT_EQ(binary_input.media_sources.at(0).modality, "audio");
  const torch::Tensor base64_audio = base64_input.media_sources.at(0).tensor;
  EXPECT_EQ(base64_audio.scalar_type(), torch::kFloat32);
  EXPECT_TRUE(
      torch::equal(base64_audio, binary_input.media_sources.at(0).tensor));
  EXPECT_EQ(base64_request.SerializeAsString(), original_base64_request);
  EXPECT_EQ(binary_request.SerializeAsString(), original_binary_request);
}

TEST(DiTRequestParamsTest, RejectsOutOfBoundsBinaryPromptAudio) {
  proto::AudioGenerationRequest request = MakeAudioRequest();
  proto::MediaSource* source = request.mutable_input()->mutable_prompt_audio();
  source->set_type("binary");
  source->set_name("prompt_audio");
  source->set_modality("audio");
  source->mutable_binary()->set_offset(1);
  source->mutable_binary()->set_length(std::numeric_limits<uint64_t>::max());

  DiTRequestParams params(
      request, "rid", "rtime", BinaryPayload(TinyWavBytes()));

  ASSERT_TRUE(params.request_parse_status.ok());
  EXPECT_TRUE(decode_fails_with_invalid_argument(params));
}

TEST(DiTRequestParamsTest, AudioOutputTypeDefaultsAndValidates) {
  proto::AudioGenerationRequest default_request = MakeAudioRequest();
  DiTRequestParams default_params(default_request, "rid", "rtime");
  EXPECT_EQ(default_params.generation_params.output_type, "base64");

  proto::AudioGenerationRequest binary_request = MakeAudioRequest();
  binary_request.mutable_parameters()->set_output_type("binary");
  DiTRequestParams binary_params(binary_request, "rid", "rtime");
  EXPECT_EQ(binary_params.generation_params.output_type, "binary");
  EXPECT_TRUE(binary_params.request_parse_status.ok());

  proto::AudioGenerationRequest invalid_request = MakeAudioRequest();
  invalid_request.mutable_parameters()->set_output_type("url");
  DiTRequestParams invalid_params(invalid_request, "rid", "rtime");
  EXPECT_TRUE(invalid_params.request_parse_status.ok());
  EXPECT_TRUE(verify_fails_with_invalid_argument(invalid_params));
}

TEST(DiTRequestParamsTest, RejectsBinaryTensorLengthMismatch) {
  proto::ImageGenerationRequest request = MakeImageRequest();
  proto::Tensor* tensor = request.mutable_input()->mutable_prompt_embed();
  tensor->set_datatype("FP32");
  tensor->add_shape(2);
  (*tensor->mutable_parameters())["is_binary"].set_bool_param(true);
  (*tensor->mutable_parameters())["offset"].set_int64_param(0);
  (*tensor->mutable_parameters())["len"].set_int64_param(sizeof(float));

  const std::string payload(sizeof(float), 0);
  DiTRequestParams params(request, "rid", "rtime", BinaryPayload(payload));
  ASSERT_TRUE(params.request_parse_status.ok());
  Status status;

  EXPECT_FALSE(params.parse_input_params(status));
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
}

TEST(DiTRequestParamsTest, RejectsMissingOrInvalidBinaryImagePayload) {
  proto::ImageGenerationRequest request = MakeImageRequest();
  proto::MediaSource* source = request.mutable_input()->add_media_sources();
  source->set_type("binary");
  source->set_name("image");
  source->set_modality("image");
  source->mutable_binary()->set_length(4);

  DiTRequestParams missing_params(request, "rid", "rtime");
  DiTRequestParams invalid_params(
      request, "rid", "rtime", BinaryPayload("nope"));

  ASSERT_TRUE(missing_params.request_parse_status.ok());
  EXPECT_TRUE(decode_fails_with_invalid_argument(missing_params));
  ASSERT_TRUE(invalid_params.request_parse_status.ok());
  Status status;
  EXPECT_FALSE(invalid_params.parse_input_params(status));
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
}

TEST(DiTRequestParamsTest, NullInputPointersPreserveExistingInput) {
  const std::vector<std::pair<DiTRequestKind, DiTInputProto>> null_inputs = {
      {DiTRequestKind::kImage, static_cast<const proto::Input*>(nullptr)},
      {DiTRequestKind::kAudio, static_cast<const proto::AudioInput*>(nullptr)},
      {DiTRequestKind::kVideo, static_cast<const proto::VideoInput*>(nullptr)},
      {DiTRequestKind::kText, static_cast<const proto::TextInput*>(nullptr)}};
  for (const auto& [request_kind, input] : null_inputs) {
    SCOPED_TRACE(input.index());
    DiTRequestParams params(request_kind);
    params.raw_input_params.input = input;
    params.input_params.prompt = "existing prompt";
    const torch::Tensor existing_tensor = torch::ones({1});
    params.input_params.media_sources.add("existing", "image", existing_tensor);
    Status status;

    EXPECT_FALSE(params.parse_input_params(status));
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.message(), "input is missing");
    EXPECT_EQ(params.input_params.prompt, "existing prompt");
    ASSERT_EQ(params.input_params.media_sources.size(), 1);
    EXPECT_TRUE(params.input_params.media_sources.contains("existing"));
    EXPECT_TRUE(torch::equal(params.input_params.media_sources.at(0).tensor,
                             existing_tensor));
  }
}

TEST(DiTRequestParamsTest, RejectsUnknownRequestKind) {
  DiTRequestParams params(static_cast<DiTRequestKind>(42));
  params.input_params.prompt = "prompt";
  Status status;

  EXPECT_FALSE(params.verify_params(status));
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.message(), "request kind is invalid");
}

TEST(DiTRequestParamsTest, DefaultRequestRequiresConcreteKind) {
  DiTRequestParams params;
  params.input_params.prompt = "offline prompt";
  EXPECT_EQ(params.request_kind(), DiTRequestKind::kNone);
  Status status;

  EXPECT_TRUE(params.parse_input_params(status));
  EXPECT_FALSE(params.verify_params(status));
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.message(), "request kind is invalid");
}

TEST(DiTRequestParamsTest, UnknownRequestKindCannotParseRawInput) {
  proto::Input input;
  input.set_prompt("raw prompt");
  for (const DiTRequestKind request_kind :
       {DiTRequestKind::kNone, static_cast<DiTRequestKind>(42)}) {
    SCOPED_TRACE(static_cast<int32_t>(request_kind));
    DiTRequestParams params(request_kind);
    params.raw_input_params.input = &input;
    params.input_params.prompt = "existing prompt";
    Status status;

    EXPECT_FALSE(params.parse_input_params(status));
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.message(), "request kind is invalid");
    EXPECT_EQ(params.input_params.prompt, "existing prompt");
  }
}

TEST(DiTRequestParamsTest, PreparsedOfflineInputRemainsUnchanged) {
  for (const DiTRequestKind request_kind : {DiTRequestKind::kImage,
                                            DiTRequestKind::kAudio,
                                            DiTRequestKind::kVideo,
                                            DiTRequestKind::kText}) {
    SCOPED_TRACE(static_cast<int32_t>(request_kind));
    DiTRequestParams params(request_kind);
    params.input_params.prompt = "offline prompt";
    const torch::Tensor existing_tensor = torch::ones({1});
    params.input_params.media_sources.add("existing", "image", existing_tensor);
    params.generation_params.guidance_scale = 4.0f;
    params.generation_params.max_new_tokens = 48;
    params.generation_params.audio_sampling_rate = 48000;
    params.generation_params.output_type = "binary";
    const DiTGenerationParams expected_generation_params =
        params.generation_params;
    Status status;

    EXPECT_EQ(params.request_kind(), request_kind);
    ASSERT_TRUE(
        std::holds_alternative<std::monostate>(params.raw_input_params.input));
    ASSERT_TRUE(params.parse_input_params(status));
    EXPECT_TRUE(params.verify_params(status)) << status.message();
    EXPECT_TRUE(status.ok());
    EXPECT_EQ(params.input_params.prompt, "offline prompt");
    ASSERT_EQ(params.input_params.media_sources.size(), 1);
    EXPECT_EQ(params.input_params.media_sources.at(0).name, "existing");
    EXPECT_TRUE(torch::equal(params.input_params.media_sources.at(0).tensor,
                             existing_tensor));
    EXPECT_EQ(params.generation_params, expected_generation_params);
    EXPECT_EQ(params.generation_params.output_type, "binary");
  }
}

TEST(DiTRequestParamsTest, OfflineImageRejectsInvalidImageCount) {
  DiTRequestParams params(DiTRequestKind::kImage);
  params.input_params.prompt = "offline prompt";
  params.generation_params.num_images_per_prompt = 0;
  Status status;

  ASSERT_TRUE(params.parse_input_params(status));
  EXPECT_FALSE(params.verify_params(status));
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.message(), "num_images_per_prompt must be greater than 0.");
}

TEST(DiTRequestParamsTest, OfflineAudioRejectsInvalidOutputType) {
  DiTRequestParams params(DiTRequestKind::kAudio);
  params.input_params.prompt = "offline prompt";
  params.generation_params.output_type = "invalid";
  Status status;

  ASSERT_TRUE(params.parse_input_params(status));
  EXPECT_FALSE(params.verify_params(status));
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.message(), "output_type must be either base64 or binary");
}

TEST(DiTRequestParamsTest, OfflineVideoRejectsInvalidVideoCount) {
  DiTRequestParams params(DiTRequestKind::kVideo);
  params.input_params.prompt = "offline prompt";
  params.generation_params.num_videos_per_prompt = 0;
  Status status;

  ASSERT_TRUE(params.parse_input_params(status));
  EXPECT_FALSE(params.verify_params(status));
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.message(), "num_videos_per_prompt must be greater than 0.");
}

TEST(DiTRequestParamsTest, OfflineTextRejectsInvalidTokenCount) {
  DiTRequestParams params(DiTRequestKind::kText);
  params.input_params.prompt = "offline prompt";
  params.generation_params.max_new_tokens = 0;
  Status status;

  ASSERT_TRUE(params.parse_input_params(status));
  EXPECT_FALSE(params.verify_params(status));
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.message(), "max_new_tokens must be positive.");
}

TEST(DiTRequestParamsTest, VerifyUsesConstructedRequestKind) {
  DiTRequestParams image_params(DiTRequestKind::kImage);
  image_params.input_params.prompt = "offline prompt";
  image_params.generation_params.num_images_per_prompt = 0;
  Status status;
  ASSERT_FALSE(image_params.verify_params(status));

  DiTRequestParams text_params(DiTRequestKind::kText);
  text_params.input_params.prompt = "offline prompt";
  text_params.generation_params.num_images_per_prompt = 0;
  EXPECT_TRUE(text_params.verify_params(status));
  EXPECT_TRUE(status.ok());

  text_params.generation_params.max_new_tokens = 0;
  EXPECT_FALSE(text_params.verify_params(status));
  EXPECT_EQ(status.message(), "max_new_tokens must be positive.");
}

TEST(DiTRequestParamsTest, ParseUsesConstructedRequestKind) {
  proto::TextInput input;
  input.set_prompt("text prompt");
  DiTRequestParams image_params(DiTRequestKind::kImage);
  image_params.raw_input_params.input = &input;
  image_params.input_params.prompt = "existing prompt";
  Status status;

  ASSERT_FALSE(image_params.parse_input_params(status));
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.message(), "input type does not match parser");
  EXPECT_EQ(image_params.input_params.prompt, "existing prompt");

  DiTRequestParams text_params(DiTRequestKind::kText);
  text_params.raw_input_params.input = &input;
  Status text_status;
  ASSERT_TRUE(text_params.parse_input_params(text_status));
  EXPECT_TRUE(text_status.ok());
  EXPECT_EQ(text_params.input_params.prompt, "text prompt");
}

// ===========================================================================
// DiTRequestParams constructor from TextGenerationRequest
// ===========================================================================

TEST(DiTRequestParamsTest, TextRequestSetsKindToText) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  DiTRequestParams params(req, "rid", "rtime");

  EXPECT_EQ(params.request_kind(), DiTRequestKind::kText);
}

TEST(DiTRequestParamsTest, TextRequestMapsPrompt) {
  auto req = MakeTextRequest("Cola-DLM", "hello world");
  DiTRequestParams params(req, "rid", "rtime");

  const auto* input =
      std::get_if<const proto::TextInput*>(&params.raw_input_params.input);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(*input, nullptr);
  EXPECT_EQ((*input)->prompt(), "hello world");
}

TEST(DiTRequestParamsTest, TextRequestUsesProvidedRequestId) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  req.set_request_id("custom-id");
  DiTRequestParams params(req, "rid", "rtime");

  EXPECT_EQ(params.request_id, "custom-id");
}

TEST(DiTRequestParamsTest, TextRequestGeneratesRequestIdWhenAbsent) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  DiTRequestParams params(req, "rid", "rtime");

  EXPECT_FALSE(params.request_id.empty());
  EXPECT_EQ(params.request_id.substr(0, 8), "textgen-");
}

TEST(DiTRequestParamsTest, TextRequestMapsXRequestIdAndTime) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  DiTRequestParams params(req, "xrid-123", "xrtime-456");

  EXPECT_EQ(params.x_request_id, "xrid-123");
  EXPECT_EQ(params.x_request_time, "xrtime-456");
}

TEST(DiTRequestParamsTest, TextRequestDefaultGenerationParams) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  DiTRequestParams params(req, "rid", "rtime");

  // Default values when no parameters set.
  EXPECT_FALSE(params.generation_params.seed_is_set);
  EXPECT_EQ(params.generation_params.max_new_tokens, 32);
  EXPECT_EQ(params.generation_params.diffusion_steps, 16);
  EXPECT_FLOAT_EQ(params.generation_params.guidance_scale, 7.0f);
}

TEST(DiTRequestParamsTest, TextRequestMapsSeedAndSetsFlag) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  auto* p = req.mutable_parameters();
  p->set_seed(42);
  DiTRequestParams params(req, "rid", "rtime");

  EXPECT_EQ(params.generation_params.seed, 42);
  EXPECT_TRUE(params.generation_params.seed_is_set);
}

TEST(DiTRequestParamsTest, TextRequestMapsAllSamplingParams) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  auto* p = req.mutable_parameters();
  p->set_max_new_tokens(128);
  p->set_diffusion_steps(32);
  p->set_guidance_scale(5.0f);
  p->set_temperature(0.8f);
  p->set_top_k(50);
  p->set_top_p(0.95f);
  p->set_repetition_penalty(1.2f);

  DiTRequestParams params(req, "rid", "rtime");

  EXPECT_EQ(params.generation_params.max_new_tokens, 128);
  EXPECT_EQ(params.generation_params.diffusion_steps, 32);
  EXPECT_FLOAT_EQ(params.generation_params.guidance_scale, 5.0f);
  EXPECT_FLOAT_EQ(params.generation_params.temperature, 0.8f);
  EXPECT_EQ(params.generation_params.top_k, 50);
  EXPECT_FLOAT_EQ(params.generation_params.top_p, 0.95f);
  EXPECT_FLOAT_EQ(params.generation_params.repetition_penalty, 1.2f);
}

TEST(DiTRequestParamsTest, TextRequestWithoutParametersUsesDefaults) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  // Don't set parameters at all.
  DiTRequestParams params(req, "rid", "rtime");

  EXPECT_EQ(params.generation_params.max_new_tokens, 32);
  EXPECT_EQ(params.generation_params.diffusion_steps, 16);
  EXPECT_FLOAT_EQ(params.generation_params.guidance_scale, 7.0f);
  EXPECT_FALSE(params.generation_params.seed_is_set);
}

// ===========================================================================
// verify_params for kText
// ===========================================================================

TEST(DiTRequestParamsVerifyTest, TextValidParamsReturnsTrue) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  auto* p = req.mutable_parameters();
  p->set_max_new_tokens(128);
  p->set_diffusion_steps(16);
  DiTRequestParams params(req, "rid", "rtime");
  decode_input(params);

  Status status;
  const bool result = params.verify_params(status);

  EXPECT_TRUE(result);
  EXPECT_TRUE(status.ok());
}

TEST(DiTRequestParamsVerifyTest, TextEmptyPromptReturnsFalse) {
  auto req = MakeTextRequest("Cola-DLM", "");
  DiTRequestParams params(req, "rid", "rtime");
  decode_input(params);

  Status status;
  const bool result = params.verify_params(status);

  EXPECT_FALSE(result);
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.message(), "prompt is empty");
}

TEST(DiTRequestParamsVerifyTest, TextValidationDoesNotRequireModel) {
  auto req = MakeTextRequest("", "hello");
  auto* p = req.mutable_parameters();
  p->set_max_new_tokens(128);
  p->set_diffusion_steps(16);
  DiTRequestParams params(req, "rid", "rtime");
  decode_input(params);

  Status status;
  const bool result = params.verify_params(status);

  EXPECT_TRUE(result);
  EXPECT_TRUE(status.ok());
}

TEST(DiTRequestParamsVerifyTest, TextZeroMaxNewTokensReturnsFalse) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  auto* p = req.mutable_parameters();
  p->set_max_new_tokens(0);
  p->set_diffusion_steps(16);
  DiTRequestParams params(req, "rid", "rtime");
  decode_input(params);

  Status status;
  const bool result = params.verify_params(status);

  EXPECT_FALSE(result);
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
}

TEST(DiTRequestParamsVerifyTest, TextNegativeDiffusionStepsReturnsFalse) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  auto* p = req.mutable_parameters();
  p->set_max_new_tokens(128);
  p->set_diffusion_steps(-1);
  DiTRequestParams params(req, "rid", "rtime");
  decode_input(params);

  Status status;
  const bool result = params.verify_params(status);

  EXPECT_FALSE(result);
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
}

// ===========================================================================
// DiTGenerationParams equality with new text fields
// ===========================================================================

TEST(DiTGenerationParamsEqualityTest, EqualParamsAreEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;

  EXPECT_EQ(a, b);
}

TEST(DiTGenerationParamsEqualityTest, DifferentMaxNewTokensAreNotEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;
  a.max_new_tokens = 128;
  b.max_new_tokens = 256;

  EXPECT_NE(a, b);
}

TEST(DiTGenerationParamsEqualityTest, DifferentSeedIsSetAreNotEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;
  a.seed_is_set = true;
  b.seed_is_set = false;

  EXPECT_NE(a, b);
}

TEST(DiTGenerationParamsEqualityTest, DifferentDiffusionStepsAreNotEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;
  a.diffusion_steps = 16;
  b.diffusion_steps = 32;

  EXPECT_NE(a, b);
}

TEST(DiTGenerationParamsEqualityTest, DifferentTemperatureAreNotEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;
  a.temperature = 0.0f;
  b.temperature = 1.0f;

  EXPECT_NE(a, b);
}

TEST(DiTGenerationParamsEqualityTest, DifferentTopKAreNotEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;
  a.top_k = 0;
  b.top_k = 50;

  EXPECT_NE(a, b);
}

TEST(DiTGenerationParamsEqualityTest, DifferentTopPAreNotEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;
  a.top_p = 1.0f;
  b.top_p = 0.9f;

  EXPECT_NE(a, b);
}

TEST(DiTGenerationParamsEqualityTest, DifferentRepetitionPenaltyAreNotEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;
  a.repetition_penalty = 1.0f;
  b.repetition_penalty = 1.1f;

  EXPECT_NE(a, b);
}

}  // namespace
}  // namespace xllm

namespace xllm {
namespace {
// Keep decoder tests independent of which model backends were built.
REGISTER_IMAGE_ALPHA_POLICY(TestRgbaPipeline, true);

DiTRequestParams parse_model_image_request(
    const proto::ImageGenerationRequest& request,
    const std::string& rid,
    const std::string& rtime,
    const std::string& model_type = "TestRgbaPipeline") {
  DiTRequestParams params(request, rid, rtime);
  Status status;
  params.parse_input_params(
      status, ModelRegistry::get_preserve_image_alpha(model_type));
  params.request_parse_status = status;
  return params;
}

proto::ImageGenerationRequest make_qwen_image_21_request(int64_t length = 2) {
  proto::ImageGenerationRequest request;
  request.set_model("QwenImage21Pipeline");
  auto* embed = request.mutable_input()->mutable_prompt_embed();
  embed->set_datatype("FP32");
  embed->add_shape(length);
  embed->add_shape(4096);
  for (int64_t i = 0; i < length * 4096; ++i) {
    embed->mutable_contents()->add_fp32_contents(0.0f);
  }
  return request;
}

TEST(DiTRequestParamsTest, QwenImage21DoesNotApplyModelAdmissionRules) {
  auto request = make_qwen_image_21_request();
  request.mutable_parameters()->set_size("33*32");
  request.mutable_parameters()->set_true_cfg_scale(4.0f);
  auto params = parse_model_image_request(request, "rid", "time");
  ASSERT_TRUE(params.request_parse_status.ok());
  // Model constraints are enforced by the pipeline, not the transport parser.
  Status status;
  EXPECT_TRUE(params.verify_params(status));
  EXPECT_EQ(params.generation_params.width, 33);
  EXPECT_FALSE(
      params.input_params.tensor_sources.contains("negative_prompt_embed"));
}

TEST(DiTRequestParamsTest, QwenImage21PreservesClientTensorShapes) {
  for (bool client_batch : {false, true}) {
    auto request = make_qwen_image_21_request();
    request.mutable_parameters()->set_size("32*32");
    auto* embedding = request.mutable_input()->mutable_prompt_embed();
    embedding->clear_shape();
    auto* latent = request.mutable_input()->mutable_latent();
    latent->set_datatype("FP32");
    if (client_batch) {
      embedding->add_shape(1);
      latent->add_shape(1);
    }
    embedding->add_shape(2);
    embedding->add_shape(4096);
    latent->add_shape(4);
    latent->add_shape(64);
    for (int64_t i = 0; i < 4 * 64; ++i) {
      latent->mutable_contents()->add_fp32_contents(0.0f);
    }
    (*embedding->mutable_parameters())["enable_prefix_kv_cache"].set_bool_param(
        false);
    auto params = parse_model_image_request(request, "rid", "time");
    ASSERT_TRUE(params.request_parse_status.ok());
    const auto parsed_embedding =
        params.input_params.tensor_sources.get("prompt_embed");
    const auto parsed_latent = params.input_params.tensor_sources.get("latent");
    ASSERT_TRUE(parsed_embedding.has_value());
    ASSERT_TRUE(parsed_latent.has_value());
    EXPECT_EQ(parsed_embedding->dim(), client_batch ? 3 : 2);
    EXPECT_EQ(parsed_latent->dim(), client_batch ? 3 : 2);
    const auto named_embedding =
        params.input_params.tensor_sources.get_named_tensor("prompt_embed");
    ASSERT_TRUE(named_embedding.has_value());
    EXPECT_FALSE(std::get<bool>(
        named_embedding->get().parameters.at("enable_prefix_kv_cache")));
  }
}

TEST(DiTRequestParamsTest, QwenImage21LeavesEmptyNegativePromptForPipeline) {
  auto request = make_qwen_image_21_request();
  request.mutable_input()->clear_prompt_embed();
  request.mutable_input()->set_prompt("draw a cat");
  request.mutable_parameters()->set_true_cfg_scale(4.0f);
  auto params = parse_model_image_request(request, "rid", "time");
  ASSERT_TRUE(params.request_parse_status.ok());
  EXPECT_EQ(params.input_params.prompt, "draw a cat");
  EXPECT_TRUE(params.input_params.negative_prompt.empty());
  Status status;
  EXPECT_TRUE(params.verify_params(status));
}

TEST(DiTRequestParamsTest,
     QwenImage21PreservesAlphaWithoutInferringOutputSize) {
  OpenCVImageEncoder encoder;
  auto rgba = torch::zeros({4, 32, 32}, torch::kFloat32);
  rgba[0].fill_(1.0f);
  rgba[3].fill_(0.25f);
  std::string png;
  ASSERT_TRUE(encoder.encode(rgba, png));
  OpenCVImageDecoder decoder;
  torch::Tensor roundtrip, legacy_rgb;
  ASSERT_TRUE(decoder.decode(png, roundtrip, /*preserve_alpha=*/true));
  ASSERT_EQ(roundtrip.size(0), 4);
  EXPECT_EQ(roundtrip[3][0][0].item<int64_t>(), 64);
  ASSERT_TRUE(decoder.decode(png, legacy_rgb));
  EXPECT_EQ(legacy_rgb.size(0), 3);
  EXPECT_EQ(legacy_rgb[1][0][0].item<int64_t>(), 191);

  auto request = make_qwen_image_21_request(3);
  request.set_model("my-image-service");
  std::string encoded;
  butil::Base64Encode(png, &encoded);
  request.mutable_input()->add_images(encoded);
  auto* parameters =
      request.mutable_input()->mutable_prompt_embed()->mutable_parameters();
  for (int64_t bit : {0, 1, 0}) {
    (*parameters)["image_pad_mask"].add_int64_tensor_param(bit);
  }
  (*parameters)["input_image_sizes"].add_int64_tensor_param(32);
  (*parameters)["input_image_sizes"].add_int64_tensor_param(32);
  auto params = parse_model_image_request(request, "rid", "time");
  ASSERT_TRUE(params.request_parse_status.ok())
      << params.request_parse_status.message();
  const auto& image = params.input_params.media_sources.at(0).tensor;
  EXPECT_EQ(image.sizes(), (torch::IntArrayRef{4, 32, 32}));
  EXPECT_EQ(image[3][0][0].item<int64_t>(), 64);
  EXPECT_EQ(params.generation_params.width, 512);
  EXPECT_EQ(params.generation_params.height, 512);

  auto rgb_params = parse_model_image_request(
      request, "rid", "time", "UnregisteredRgbPipeline");
  ASSERT_TRUE(rgb_params.request_parse_status.ok());
  const auto& rgb = rgb_params.input_params.media_sources.at(0).tensor;
  EXPECT_EQ(rgb.sizes(), (torch::IntArrayRef{3, 32, 32}));
  EXPECT_EQ(rgb[1][0][0].item<int64_t>(), 191);
}
}  // namespace
}  // namespace xllm
