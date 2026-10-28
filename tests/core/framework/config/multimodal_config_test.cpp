/* Copyright 2026 The xLLM Authors.

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

#include "core/framework/config/multimodal_config.h"

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

namespace xllm {
namespace {

MultimodalConfig service_config(const std::string& mm_args) {
  google::SetCommandLineOption("mm_args", mm_args.c_str());
  MultimodalConfig config;
  config.initialize();
  return config;
}

class MultimodalConfigTest : public ::testing::Test {
 protected:
  void SetUp() override {
    google::SetCommandLineOption("config_json_file", "");
  }

 private:
  google::FlagSaver flag_saver_;
};

TEST_F(MultimodalConfigTest, EmptyOverridesPreserveAllModalities) {
  for (const std::string mm_args : {"", "{}", R"({"image":{}})"}) {
    const auto config = service_config(mm_args);
    EXPECT_TRUE(config.processor_args("image").empty());
    EXPECT_TRUE(config.processor_args("video").empty());
    EXPECT_TRUE(config.processor_args("audio").empty());
  }
}

TEST_F(MultimodalConfigTest, RoutesArbitraryFieldsWithoutModelKnowledge) {
  const auto config = service_config(
      R"({"image":{"max_pixels":4096,"size":{"longest_edge":64}},
          "video":{"fps":2.5,"max_frames":32},
          "audio":{"padding":true,"normalization":[0.1,0.2]}})");
  EXPECT_EQ(config.processor_args("image").at("max_pixels"), 4096);
  EXPECT_EQ(config.processor_args("image").at("size").at("longest_edge"), 64);
  EXPECT_EQ(config.processor_args("video").at("fps"), 2.5);
  EXPECT_EQ(config.processor_args("audio").at("padding"), true);
  EXPECT_FALSE(config.processor_args("image").contains("fps"));
}

TEST_F(MultimodalConfigTest, RejectsMalformedObjectsAndUnknownModalities) {
  EXPECT_DEATH(service_config("{"), "mm_args must be valid JSON");
  EXPECT_DEATH(service_config("[]"),
               "mm_args must be a JSON object, got array");
  EXPECT_DEATH(service_config("null"),
               "mm_args must be a JSON object, got null");
  EXPECT_DEATH(service_config("42"),
               "mm_args must be a JSON object, got number");
  EXPECT_DEATH(service_config(R"({"max_pixels":4096})"),
               "Unsupported mm_args modality");
  EXPECT_DEATH(service_config(R"({"images":{}})"),
               "Unsupported mm_args modality");
}

TEST_F(MultimodalConfigTest, RequiresObjectForEachModality) {
  for (const std::string mm_args :
       {R"({"image":null})", R"({"video":[]})", R"({"audio":true})"}) {
    EXPECT_DEATH(service_config(mm_args), "must be an object");
  }
}

TEST_F(MultimodalConfigTest, AppearsInHelpAndConfigurationExport) {
  const std::string mm_args = R"({"image":{"max_pixels":1048576}})";
  const auto config = service_config(mm_args);
  nlohmann::ordered_json output;
  config.append_config_json(output);
  EXPECT_EQ(output.at("mm_args").get<std::string>(), mm_args);
  const auto& names = MultimodalConfig::option_category().option_names;
  EXPECT_NE(std::find(names.begin(), names.end(), "mm_args"), names.end());
}

TEST_F(MultimodalConfigTest, LoadsConfigFileAndHonorsCommandLinePrecedence) {
  const auto path =
      std::filesystem::temp_directory_path() / "xllm-mm-args-config-test.json";
  {
    std::ofstream stream(path);
    stream << nlohmann::json(
        {{"mm_args", R"({"image":{"max_pixels":1048576}})"}});
  }
  google::SetCommandLineOption("config_json_file", path.c_str());
  MultimodalConfig from_file;
  from_file.initialize();
  EXPECT_EQ(from_file.processor_args("image").at("max_pixels"), 1048576);
  google::SetCommandLineOption("mm_args", R"({"image":{"max_pixels":262144}})");
  MultimodalConfig from_cli;
  from_cli.initialize();
  EXPECT_EQ(from_cli.processor_args("image").at("max_pixels"), 262144);
  std::filesystem::remove(path);
}

TEST_F(MultimodalConfigTest, ReinitializationClearsPreviousOverrides) {
  auto config = service_config(R"({"image":{"max_pixels":1048576}})");
  google::SetCommandLineOption("mm_args", "{}");
  config.initialize();
  EXPECT_TRUE(config.processor_args("image").empty());
}

}  // namespace
}  // namespace xllm
