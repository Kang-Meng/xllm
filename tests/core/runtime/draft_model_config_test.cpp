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

#include "core/runtime/draft_model_config.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/model/model_args.h"
#include "core/runtime/options.h"
#include "core/util/json_reader.h"

namespace xllm {
namespace {

#if defined(USE_MLU)
TEST(DraftModelConfigTest, ShortensDFlash2BlockBeforeModelConstruction) {
  ModelArgs args;
  args.model_type("qwen3")
      .dflash2_block_size(8)
      .dflash2_conv_kernel_size(4)
      .layers_to_capture({1, 3});
  runtime::Options options;
  options.is_draft_engine(true)
      .speculative_algorithm("DFlash2")
      .draft_graph_query_width(6);

  configure_block_diffusion_model(args, options);

  EXPECT_EQ(args.model_type(), "DFlash2DraftModel");
  EXPECT_EQ(args.dflash2_block_size(), 6);
  EXPECT_EQ(args.num_speculative_tokens(), 5);
  EXPECT_EQ(args.dummy_token_count(), 6);
  EXPECT_EQ(args.dflash2_conv_kernel_size(), 4);
  EXPECT_TRUE(args.layers_to_capture().empty());
  EXPECT_TRUE(args.requires_eager_execution());
}

TEST(DraftModelConfigTest, AcceptsFullBlockAndConvolutionBoundary) {
  for (const int32_t width : {8, 4}) {
    ModelArgs args;
    args.model_type("qwen3").dflash2_block_size(8).dflash2_conv_kernel_size(4);
    runtime::Options options;
    options.is_draft_engine(true)
        .speculative_algorithm("DFlash2")
        .draft_graph_query_width(width);
    configure_block_diffusion_model(args, options);
    EXPECT_EQ(args.dflash2_block_size(), width);
    EXPECT_EQ(args.dummy_token_count(), width);
    EXPECT_EQ(args.num_speculative_tokens(), width == 8 ? 7 : 3);
  }
}

TEST(DraftModelConfigDeathTest, RejectsInvalidRuntimeGeometry) {
  struct Geometry {
    int32_t trained;
    int32_t runtime;
    const char* error;
  };
  const std::vector<Geometry> geometries = {
      {8, 1, "requires candidate tokens"},
      {8, 9, "must not exceed"},
      {8, 3, "must cover the convolution kernel"},
      {0, 4, "trained_block_size > 0"},
      {-1, 4, "trained_block_size > 0"}};
  for (const Geometry& geometry : geometries) {
    SCOPED_TRACE(geometry.trained);
    SCOPED_TRACE(geometry.runtime);
    ModelArgs args;
    args.model_type("qwen3")
        .dflash2_block_size(geometry.trained)
        .dflash2_conv_kernel_size(4);
    runtime::Options options;
    options.is_draft_engine(true)
        .speculative_algorithm("DFlash2")
        .draft_graph_query_width(geometry.runtime);
    EXPECT_DEATH(configure_block_diffusion_model(args, options),
                 geometry.error);
  }
}

TEST(DraftModelConfigDeathTest, RejectsUnsupportedDraftAlgorithmsOnMlu) {
  for (const std::string& algorithm : {"DFlash", "DSpark"}) {
    ModelArgs args;
    args.model_type("qwen3");
    runtime::Options options;
    options.is_draft_engine(true).speculative_algorithm(algorithm);
    EXPECT_DEATH(configure_block_diffusion_model(args, options),
                 "not supported on mlu");
  }
}
#endif

class DraftCaptureConfigTest : public ::testing::Test {
 protected:
  void SetUp() override {
    char path[] = "/tmp/xllm-draft-config-XXXXXX";
    ASSERT_NE(mkdtemp(path), nullptr);
    model_path_ = path;
  }

  void TearDown() override { std::filesystem::remove_all(model_path_); }

  void write_config(const std::string& json) {
    std::ofstream file(model_path_ + "/config.json");
    ASSERT_TRUE(file.is_open());
    file << json;
    ASSERT_TRUE(file.good());
  }

  std::string model_path_;
};

TEST_F(DraftCaptureConfigTest, ConfiguresTargetWithoutChangingModelGeometry) {
  write_config(R"({"dflash_config":{"target_layer_ids":[1,3]}})");
  ModelArgs args;
  args.model_type("glm_moe_dsa")
      .n_layers(8)
      .dflash2_block_size(8)
      .num_speculative_tokens(5);
  runtime::Options options;
  options.is_draft_engine(false)
      .speculative_algorithm("DFlash2")
      .draft_model_path(model_path_);
  configure_block_diffusion_model(args, options);
  EXPECT_EQ(args.layers_to_capture(), (std::vector<int32_t>{1, 3}));
  EXPECT_EQ(args.model_type(), "glm_moe_dsa");
  EXPECT_EQ(args.dflash2_block_size(), 8);
  EXPECT_EQ(args.num_speculative_tokens(), 5);
}

TEST_F(DraftCaptureConfigTest, ConvertsBoundaryIndicesForSharedDraftCapture) {
  write_config(R"({"aux_hidden_state_layer_ids":[2,4]})");
  EXPECT_EQ(read_capture_layer_ids(model_path_), (std::vector<int32_t>{1, 3}));
}

TEST_F(DraftCaptureConfigTest, PrefersLegacyPostLayerIndices) {
  write_config(R"({"dspark_target_layer_ids":[1,3],
                    "aux_hidden_state_layer_ids":[3,5]})");
  EXPECT_EQ(read_capture_layer_ids(model_path_), (std::vector<int32_t>{1, 3}));
}

TEST_F(DraftCaptureConfigTest, AllowsEagleCaptureDefaultsForMissingConfig) {
  EXPECT_TRUE(read_capture_layer_ids(model_path_, /*required=*/false).empty());
  write_config("{}");
  EXPECT_TRUE(read_capture_layer_ids(model_path_, /*required=*/false).empty());
}

TEST_F(DraftCaptureConfigTest, RejectsMissingRequiredCaptureConfig) {
  EXPECT_DEATH(read_capture_layer_ids(model_path_), "Failed to parse");
  write_config("{}");
  EXPECT_DEATH(read_capture_layer_ids(model_path_),
               "requires dspark_target_layer_ids");
}

TEST_F(DraftCaptureConfigTest, RejectsEmbeddingCaptureBoundary) {
  write_config(R"({"aux_hidden_state_layer_ids":[0,4]})");
  EXPECT_DEATH(read_capture_layer_ids(model_path_), "Embedding capture");
}

TEST_F(DraftCaptureConfigTest, RejectsInvalidTargetCaptureLayers) {
  for (const std::string& json : {R"({"target_layer_ids":[-1,3]})",
                                  R"({"target_layer_ids":[1,8]})",
                                  R"({"target_layer_ids":[1,1]})"}) {
    write_config(json);
    ModelArgs args;
    args.n_layers(8);
    runtime::Options options;
    options.is_draft_engine(false).draft_model_path(model_path_);
    EXPECT_DEATH(configure_block_diffusion_model(args, options),
                 "Invalid target capture layer|out of range|Duplicate target "
                 "capture layer");
  }
}

TEST(DraftModelConfigDeathTest, RequiresDraftPathForTargetCapture) {
  ModelArgs args;
  runtime::Options options;
  options.is_draft_engine(false);
  EXPECT_DEATH(configure_block_diffusion_model(args, options),
               "requires --draft_model");
}

TEST_F(DraftCaptureConfigTest, MaskRootAliasWinsOverTheTextConfigValue) {
  write_config(R"({"dspark_noise_token_id": 128799,
                    "text_config": {"dspark_noise_token_id": 1}})");
  JsonReader reader;
  ASSERT_TRUE(reader.parse(model_path_ + "/config.json"));
  EXPECT_EQ(read_mask_token_id(reader), 128799);
}

TEST_F(DraftCaptureConfigTest, MaskTextConfigFallbackResolves) {
  write_config(R"({"text_config": {"mask_token_id": 128799}})");
  JsonReader reader;
  ASSERT_TRUE(reader.parse(model_path_ + "/config.json"));
  EXPECT_EQ(read_mask_token_id(reader), 128799);
}

TEST_F(DraftCaptureConfigTest, MaskMissingEverywhereReturnsInvalid) {
  write_config(R"({"text_config": {"num_hidden_layers": 40}})");
  JsonReader reader;
  ASSERT_TRUE(reader.parse(model_path_ + "/config.json"));
  EXPECT_EQ(read_mask_token_id(reader), -1);
}

TEST_F(DraftCaptureConfigTest, MaskRootAliasesKeepTheirPrecedence) {
  write_config(R"({"dflash_config": {"mask_token_id": 1},
                    "mask_token_id": 2,
                    "dspark_noise_token_id": 3})");
  JsonReader reader;
  ASSERT_TRUE(reader.parse(model_path_ + "/config.json"));
  EXPECT_EQ(read_mask_token_id(reader), 1);
  write_config(R"({"mask_token_id": 2, "dspark_noise_token_id": 3})");
  ASSERT_TRUE(reader.parse(model_path_ + "/config.json"));
  EXPECT_EQ(read_mask_token_id(reader), 2);
  write_config(R"({"dspark_noise_token_id": 3})");
  ASSERT_TRUE(reader.parse(model_path_ + "/config.json"));
  EXPECT_EQ(read_mask_token_id(reader), 3);
}

TEST_F(DraftCaptureConfigTest, BlockSizeRootKeyWinsOverTheTextConfigValue) {
  write_config(R"({"dspark_block_size": 5,
                    "text_config": {"dspark_block_size": 7}})");
  EXPECT_EQ(read_block_size(model_path_), 5);
}

TEST_F(DraftCaptureConfigTest, BlockSizeTextConfigFallbackResolves) {
  write_config(R"({"text_config": {"dspark_block_size": 5}})");
  EXPECT_EQ(read_block_size(model_path_), 5);
}

TEST_F(DraftCaptureConfigTest, BlockSizeMissingEverywhereStaysZero) {
  write_config(R"({"text_config": {"num_hidden_layers": 40}})");
  EXPECT_EQ(read_block_size(model_path_), 0);
}

#if defined(USE_NPU)
TEST(DraftModelConfigTest, AllowsPythonDSparkGraphsWithoutChangingOtherDrafts) {
  auto& model_config = ModelConfig::get_instance();
  const std::string original_model_impl = model_config.model_impl();
  for (const std::string& model_impl : {"python", "native"}) {
    model_config.model_impl(model_impl);
    for (const std::string& algorithm : {"DSpark", "DFlash", "DFlash2"}) {
      SCOPED_TRACE(model_impl);
      SCOPED_TRACE(algorithm);
      ModelArgs args;
      args.model_type("qwen3").dflash2_block_size(8);
      runtime::Options options;
      options.is_draft_engine(true)
          .speculative_algorithm(algorithm)
          .num_speculative_tokens(6);

      configure_block_diffusion_model(args, options);

      EXPECT_EQ(args.requires_eager_execution(),
                model_impl != "python" || algorithm != "DSpark");
      EXPECT_FALSE(args.draft_query_uses_decode());
      if (algorithm == "DSpark") {
        EXPECT_EQ(args.dummy_token_count(), model_impl == "python" ? 6 : 1);
      }
    }
  }
  model_config.model_impl(original_model_impl);
}

TEST_F(DraftCaptureConfigTest, PythonV41UsesBlockRowsAndCompleteDummyQueries) {
  write_config(R"({"dspark_block_size":3})");
  auto& model_config = ModelConfig::get_instance();
  auto& kernel_config = KernelConfig::get_instance();
  const std::string original_model_impl = model_config.model_impl();
  const bool original_native_sas = kernel_config.enable_dspark_native_sas();
  model_config.model_impl("python");
  kernel_config.enable_dspark_native_sas(true);
  ModelArgs args;
  args.model_type("deepseek_v41").n_layers(4).dspark_num_layers(2);
  runtime::Options options;
  options.is_draft_engine(true)
      .speculative_algorithm("DSpark")
      .num_speculative_tokens(3)
      .model_path(model_path_);
  configure_block_diffusion_model(args, options);
  EXPECT_EQ(args.model_type(), "deepseek_v41_dspark");
  EXPECT_EQ(args.dummy_token_count(), 3);
  EXPECT_EQ(args.n_layers(), 2);
  EXPECT_EQ(args.compress_ratios(), (std::vector<int32_t>{0, 0}));
  EXPECT_FALSE(args.requires_eager_execution());
  EXPECT_FALSE(args.dspark_use_native_sas());
  EXPECT_TRUE(args.draft_query_uses_decode());
  model_config.model_impl(original_model_impl);
  kernel_config.enable_dspark_native_sas(original_native_sas);
}

class DraftQueryInputConfigTest
    : public DraftCaptureConfigTest,
      public ::testing::WithParamInterface<
          std::tuple<std::string, std::string, bool, bool>> {};

TEST_P(DraftQueryInputConfigTest, PreservesInputContractAcrossGraphModes) {
  const auto& [model_type, model_impl, native_sas, enable_graph] = GetParam();
  write_config(R"({"dspark_block_size":3})");
  auto& model_config = ModelConfig::get_instance();
  auto& kernel_config = KernelConfig::get_instance();
  const std::string original_model_impl = model_config.model_impl();
  const bool original_native_sas = kernel_config.enable_dspark_native_sas();
  model_config.model_impl(model_impl);
  kernel_config.enable_dspark_native_sas(native_sas);
  ModelArgs args;
  args.model_type(model_type).n_layers(4).dspark_num_layers(2);
  runtime::Options options;
  options.is_draft_engine(true)
      .speculative_algorithm("DSpark")
      .num_speculative_tokens(3)
      .model_path(model_path_)
      .enable_graph(enable_graph);

  configure_block_diffusion_model(args, options);

  const bool python_v41_draft =
      model_type == "deepseek_v41" && model_impl == "python";
  EXPECT_EQ(args.dspark_use_native_sas(), native_sas && !python_v41_draft);
  EXPECT_EQ(args.draft_query_uses_decode(), native_sas || python_v41_draft);
  model_config.model_impl(original_model_impl);
  kernel_config.enable_dspark_native_sas(original_native_sas);
}

INSTANTIATE_TEST_SUITE_P(
    DraftImplementations,
    DraftQueryInputConfigTest,
    ::testing::Combine(::testing::Values(std::string("deepseek_v4"),
                                         std::string("deepseek_v41")),
                       ::testing::Values(std::string("native"),
                                         std::string("python")),
                       ::testing::Bool(),
                       ::testing::Bool()));

TEST(DraftModelConfigTest, PreservesNpuCheckpointBlockGeometry) {
  ModelArgs args;
  args.model_type("qwen3")
      .dflash2_block_size(8)
      .dflash2_conv_kernel_size(4)
      .num_speculative_tokens(7);
  runtime::Options options;
  options.is_draft_engine(true)
      .speculative_algorithm("DFlash2")
      .draft_graph_query_width(6);
  configure_block_diffusion_model(args, options);
  EXPECT_EQ(args.dflash2_block_size(), 8);
  EXPECT_EQ(args.num_speculative_tokens(), 7);
  EXPECT_EQ(args.dummy_token_count(), 8);
  EXPECT_EQ(args.model_type(), "DFlash2DraftModel");
}

namespace {

// Fixed inputs shared by the DSpark draft-geometry tests: a 40-decoder V4
// checkpoint whose compress_ratios carries 43 entries -- 40 decoder ratios
// plus a three-entry draft tail encoded V4.1-style as 0 (pure SWA) -- plus
// the DSpark draft fields the config reader consumes. The config file on
// disk stays required even though ModelArgs carries the same geometry
// in memory: configure_deepseek_v4_dspark_args re-reads config.json for the
// draft block size (read_block_size), mirroring how the deployment loader
// points options.model_path() at the real checkpoint directory.
bool write_v4_dspark_config(const std::string& model_path) {
  std::ofstream file(model_path + "/config.json");
  if (!file.is_open()) {
    return false;
  }
  file << R"json({
    "model_type": "deepseek_v4",
    "num_hidden_layers": 40,
    "compress_ratios": [1, 4, 128, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    "dspark_block_size": 5,
    "dspark_markov_rank": 256,
    "dspark_noise_token_id": 128799,
    "dspark_target_layer_ids": [40, 41, 42]
  })json";
  return file.good();
}

// RAII guard for the process-wide model_impl selection. The draft
// model_type override consults both FLAGS_model_impl and the cached
// ModelConfig singleton, so the guard saves and restores both values
// directly (never re-deriving them via from_flags(), which would rewrite
// every ModelConfig field and clobber state other tests rely on). Without
// it, an ASSERT_* early return would leave the process on "python" and
// silently reroute later tests through the python draft path.
class ModelImplGuard final {
 public:
  explicit ModelImplGuard(const std::string& model_impl) {
    old_flag_ = FLAGS_model_impl;
    old_config_impl_ = ModelConfig::get_instance().model_impl();
    FLAGS_model_impl = model_impl;
    ModelConfig::get_instance().model_impl(model_impl);
  }

  ModelImplGuard(const ModelImplGuard&) = delete;
  ModelImplGuard& operator=(const ModelImplGuard&) = delete;

  ~ModelImplGuard() {
    FLAGS_model_impl = old_flag_;
    ModelConfig::get_instance().model_impl(old_config_impl_);
  }

 private:
  std::string old_flag_;
  std::string old_config_impl_;
};

}  // namespace

// The C++ deepseek_v4_dspark draft constructs DSAttentionImpl per layer, and
// its constructor CHECKs the layer's compress ratio against {1, 4, 128}. A V4
// checkpoint that encodes its trailing draft-stage entries as 0 (the V4.1
// pure-SWA spelling) must therefore have the checkpoint-tail ratios
// normalized to 1 for the V4 draft, instead of passing the raw 0 through and
// failing at layer 0.
TEST(DraftModelConfigTest, NormalizesV4DraftTailZeroCompressRatios) {
  const std::string model_path =
      testing::TempDir() + "xllm-draft-v4-dspark-ratios";
  std::filesystem::remove_all(model_path);
  ASSERT_TRUE(std::filesystem::create_directories(model_path));
  ASSERT_TRUE(write_v4_dspark_config(model_path));

  ModelArgs args;
  args.model_type("deepseek_v4")
      .n_layers(40)
      .dspark_num_layers(3)
      .compress_ratios({1, 4, 128, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 0, 0, 0});
  runtime::Options options;
  options.is_draft_engine(true).speculative_algorithm("DSpark").model_path(
      model_path);

  configure_block_diffusion_model(args, options);

  ASSERT_EQ(args.compress_ratios().size(), 3U);
  for (const int32_t ratio : args.compress_ratios()) {
    EXPECT_TRUE(ratio == 1 || ratio == 4 || ratio == 128)
        << "DSpark draft ratio " << ratio
        << " must satisfy DSAttention's compress_ratio CHECK";
  }
  EXPECT_EQ(args.compress_ratios(), (std::vector<int32_t>{1, 1, 1}))
      << "V4.1-style 0 draft tail must normalize to pure-SWA ratio 1 for the "
         "C++ V4 draft";
  EXPECT_EQ(args.model_type(), "deepseek_v4_dspark");
}

// The V4.1 python-path draft keeps the checkpoint's 0 pure-SWA encoding: its
// attention path treats 0 as a distinct SWA group identity, so the tail must
// be copied verbatim. The python-path draft is only selected when the
// executor's model_impl is python, so the guard pins the selection for the
// duration of the test and restores both the flag and the cached
// ModelConfig value even on early returns.
TEST(DraftModelConfigTest, PreservesV41DraftTailZeroCompressRatios) {
  const ModelImplGuard model_impl_guard("python");
  const std::string model_path =
      testing::TempDir() + "xllm-draft-v41-dspark-ratios";
  std::filesystem::remove_all(model_path);
  ASSERT_TRUE(std::filesystem::create_directories(model_path));
  ASSERT_TRUE(write_v4_dspark_config(model_path));

  ModelArgs args;
  args.model_type("deepseek_v41")
      .n_layers(40)
      .dspark_num_layers(3)
      .compress_ratios({1, 4, 128, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 0, 0, 0});
  runtime::Options options;
  options.is_draft_engine(true).speculative_algorithm("DSpark").model_path(
      model_path);

  configure_block_diffusion_model(args, options);

  EXPECT_EQ(args.compress_ratios(), (std::vector<int32_t>{0, 0, 0}))
      << "V4.1 draft keeps the checkpoint's 0 pure-SWA encoding verbatim";
  EXPECT_EQ(args.model_type(), "deepseek_v41_dspark");
}

// A V4.1 DSpark checkpoint under a non-python model_impl resolves the C++
// deepseek_v4_dspark draft (see the model_type override in
// configure_block_diffusion_model), so the 0-encoded tail must be
// normalized to 1 exactly like a V4 checkpoint. The normalization decision
// follows the resolved draft model type, not the checkpoint generation:
// otherwise the combination would carry a 0 ratio into DSAttentionImpl's
// {1, 4, 128} CHECK and crash at layer 0.
TEST(DraftModelConfigTest, NormalizesV41DraftTailForNativeImpl) {
  const ModelImplGuard model_impl_guard("TORCH");
  const std::string model_path =
      testing::TempDir() + "xllm-draft-v41-native-dspark-ratios";
  std::filesystem::remove_all(model_path);
  ASSERT_TRUE(std::filesystem::create_directories(model_path));
  ASSERT_TRUE(write_v4_dspark_config(model_path));

  ModelArgs args;
  args.model_type("deepseek_v41")
      .n_layers(40)
      .dspark_num_layers(3)
      .compress_ratios({1, 4, 128, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 0, 0, 0});
  runtime::Options options;
  options.is_draft_engine(true).speculative_algorithm("DSpark").model_path(
      model_path);

  configure_block_diffusion_model(args, options);

  EXPECT_EQ(args.compress_ratios(), (std::vector<int32_t>{1, 1, 1}))
      << "A V4.1 checkpoint on the native impl takes the C++ V4 draft, whose "
         "0-encoded tail must normalize to the pure-SWA ratio 1";
  EXPECT_EQ(args.model_type(), "deepseek_v4_dspark");
}
#endif

}  // namespace
}  // namespace xllm
