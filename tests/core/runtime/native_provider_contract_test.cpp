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

#include "core/runtime/native_provider_contract.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>

namespace xllm {
namespace {

NativeProviderDescriptorConfig make_config(proto::EngineRole role) {
  return NativeProviderDescriptorConfig{
      .engine_uid =
          role == proto::ENGINE_ROLE_PREFILL ? "prefill-0" : "decode-0",
      .api_address = "127.0.0.1:19000",
      .role = role,
      .runtime_version = "0.11.0",
      .hardware_runtime_version = "cpu;torch=2.8.0",
      .model_revision = "model-r1",
      .tokenizer_revision = "model-r1",
      .chat_template_contract = "{{ messages }}",
      .renderer_contract = "sentencepiece|tokenizer.model|{{ messages }}",
      .quantization = "compute=bf16",
      .soc = "cpu",
      .device_count = 2,
      .tp = 1,
      .dp = 2,
      .pp = 1,
      .ep = 1,
      .cp = 1,
      .kv_layout_contract = "model=llama|layers=32|heads=32",
      .kv_cache_dtype = "auto",
      .block_size = 128,
      .logical_kv_bytes_per_token = 4096,
      .cache_groups = {"group-0:type=kv:ratio=1"},
      .head_shard_mapping_contract = "heads=32|kv_heads=8|tp=1",
      .connector = "mooncake",
      .connector_version = "builtin-0.11.0",
      .scheduler_class = "disagg_pd",
      .max_num_seqs = 256,
      .max_num_batched_tokens = 4096,
      .scheduler_policy_contract = "policy=fcfs|chunked=true",
      .api_features = {"chat_completions", "completions"},
  };
}

proto::ProviderDescriptor make_descriptor(proto::EngineRole role) {
  std::string error;
  std::optional<proto::ProviderDescriptor> descriptor =
      build_native_provider_descriptor(make_config(role), &error);
  EXPECT_TRUE(descriptor.has_value()) << error;
  return descriptor.value_or(proto::ProviderDescriptor{});
}

TEST(NativeProviderContractTest, ApiFeaturesFollowServingModeRegistry) {
  // The published api_features must mirror ServiceImplFactory's per-mode
  // registry so the Service-side fail-closed subset check can select a VLM
  // Engine for multimodal traffic and never matches a text request to a VLM
  // Engine (which has no Completions service). VLM modes publish the extra
  // "multimodal_chat_completions" word: plain "chat_completions" is also
  // published by LLM and REC Engines and cannot route multimodal traffic
  // alone.
  EXPECT_EQ(api_features_for_backend("llm"),
            (std::vector<std::string>{"chat_completions", "completions"}));
  EXPECT_EQ(api_features_for_backend("LLM"),
            (std::vector<std::string>{"chat_completions", "completions"}));
  EXPECT_EQ(api_features_for_backend("ssm"),
            (std::vector<std::string>{"chat_completions", "completions"}));
  EXPECT_EQ(api_features_for_backend("vlm"),
            (std::vector<std::string>{"chat_completions",
                                      "multimodal_chat_completions"}));
  EXPECT_EQ(api_features_for_backend("vlmssm"),
            (std::vector<std::string>{"chat_completions",
                                      "multimodal_chat_completions"}));
  EXPECT_EQ(api_features_for_backend("rec"),
            (std::vector<std::string>{"completions", "chat_completions"}));
  EXPECT_EQ(api_features_for_backend("dit"),
            (std::vector<std::string>{"image_generation",
                                      "audio_generation",
                                      "text_generation",
                                      "video_generation"}));
  // Unknown backends must yield an empty surface so collect time fails
  // closed instead of publishing an unverifiable api_features list.
  EXPECT_TRUE(api_features_for_backend("unknown").empty());
  EXPECT_TRUE(api_features_for_backend("").empty());
}

TEST(NativeProviderContractTest, BuildsDeterministicRemotePdDescriptors) {
  const proto::ProviderDescriptor prefill =
      make_descriptor(proto::ENGINE_ROLE_PREFILL);
  const proto::ProviderDescriptor decode =
      make_descriptor(proto::ENGINE_ROLE_DECODE);
  const proto::ProviderDescriptor second_prefill =
      make_descriptor(proto::ENGINE_ROLE_PREFILL);

  EXPECT_EQ(prefill.contract_version(), kNativeProviderContractVersion);
  EXPECT_EQ(prefill.identity().provider_id(), proto::PROVIDER_ID_XLLM_NATIVE);
  EXPECT_EQ(prefill.endpoint().address(), "127.0.0.1:19000");
  EXPECT_EQ(prefill.serving().role(), proto::ENGINE_ROLE_PREFILL);
  ASSERT_EQ(prefill.serving().execution_modes_size(), 1);
  EXPECT_EQ(prefill.serving().execution_modes(0).mode(),
            proto::EXECUTION_MODE_REMOTE_PD);
  EXPECT_EQ(prefill.serving().execution_modes(0).transfer_mode(),
            proto::TRANSFER_MODE_LAYERWISE_PUSH);
  EXPECT_EQ(prefill.profile_digest(), second_prefill.profile_digest());
  EXPECT_NE(prefill.profile_digest(), decode.profile_digest());
  EXPECT_EQ(prefill.model().model_revision(), decode.model().model_revision());
  EXPECT_EQ(prefill.kv().kv_layout_digest(), decode.kv().kv_layout_digest());
  EXPECT_EQ(prefill.kv().head_shard_mapping_digest(),
            decode.kv().head_shard_mapping_digest());
  EXPECT_EQ(prefill.kv().hash_version(), 1u);
  EXPECT_EQ(prefill.kv().hash_seed(), 1024u);
  ASSERT_TRUE(prefill.kv().has_logical_bytes_per_token());
  EXPECT_EQ(prefill.kv().logical_bytes_per_token(), 4096u);
  EXPECT_EQ(prefill.topology().SerializeAsString(),
            decode.topology().SerializeAsString());
  // Capability 19 did not exist in the original contract v1 schema. Emitting
  // it before version negotiation would make an old Service reject an otherwise
  // compatible descriptor during an Engine-first rolling upgrade.
  EXPECT_EQ(std::find(prefill.capabilities().begin(),
                      prefill.capabilities().end(),
                      proto::PROVIDER_CAPABILITY_AUTHORITATIVE_CAPACITY),
            prefill.capabilities().end());
}

TEST(NativeProviderContractTest, TransferContractPreservesLocalHashIdentity) {
  auto config = make_config(proto::ENGINE_ROLE_PREFILL);
  std::string error;
  const auto legacy = build_native_provider_descriptor(config, &error);
  ASSERT_TRUE(legacy.has_value()) << error;
  EXPECT_TRUE(legacy->kv().transfer_compatibility_digest().empty());
  config.kv_transfer_layout_contract = "llama|layers=32|heads=32";
  const auto upgraded = build_native_provider_descriptor(config, &error);
  ASSERT_TRUE(upgraded.has_value()) << error;
  EXPECT_EQ(upgraded->kv().kv_layout_digest(), legacy->kv().kv_layout_digest());
  EXPECT_NE(upgraded->profile_digest(), legacy->profile_digest());

  // CP folding changes local block/hash units, not the semantic KV payload.
  config.block_size *= 2;
  config.kv_layout_contract += "|logical-block=256";
  const auto folded = build_native_provider_descriptor(config, &error);
  ASSERT_TRUE(folded.has_value()) << error;
  EXPECT_EQ(upgraded->kv().transfer_compatibility_digest(),
            folded->kv().transfer_compatibility_digest());

  config.cache_isolation_domain = "other-tenant";
  const auto isolated = build_native_provider_descriptor(config, &error);
  ASSERT_TRUE(isolated.has_value()) << error;
  EXPECT_NE(folded->kv().transfer_compatibility_digest(),
            isolated->kv().transfer_compatibility_digest());
}

TEST(NativeProviderContractTest, PropagatesSuperNodeIdIntoTopology) {
  // Data flow A: the resolved DisaggPDConfig::super_node_id is carried into the
  // provider descriptor's topology (proto tag 8), which is the field the
  // Coordinator/Router super-node guards actually read. An empty id must stay
  // empty in the topology so it keeps its wildcard (opt-out) meaning.
  NativeProviderDescriptorConfig config =
      make_config(proto::ENGINE_ROLE_PREFILL);
  config.super_node_id = "pod-A";
  std::string error;
  const std::optional<proto::ProviderDescriptor> descriptor =
      build_native_provider_descriptor(config, &error);
  ASSERT_TRUE(descriptor.has_value()) << error;
  EXPECT_EQ(descriptor->topology().super_node_id(), "pod-A");

  NativeProviderDescriptorConfig wildcard_config =
      make_config(proto::ENGINE_ROLE_PREFILL);
  const std::optional<proto::ProviderDescriptor> wildcard =
      build_native_provider_descriptor(wildcard_config, &error);
  ASSERT_TRUE(wildcard.has_value()) << error;
  EXPECT_TRUE(wildcard->topology().super_node_id().empty());
}

TEST(NativeProviderContractTest, OmitsUnknownLogicalKvBytesPerToken) {
  NativeProviderDescriptorConfig config =
      make_config(proto::ENGINE_ROLE_PREFILL);
  config.logical_kv_bytes_per_token = 0;
  std::string error;
  const std::optional<proto::ProviderDescriptor> descriptor =
      build_native_provider_descriptor(config, &error);

  ASSERT_TRUE(descriptor.has_value()) << error;
  EXPECT_FALSE(descriptor->kv().has_logical_bytes_per_token());
}

TEST(NativeProviderContractTest, PublishesOnlyRoleValidLocalModes) {
  NativeProviderDescriptorConfig prefill_config =
      make_config(proto::ENGINE_ROLE_PREFILL);
  prefill_config.supports_prefill_only = true;
  NativeProviderDescriptorConfig decode_config =
      make_config(proto::ENGINE_ROLE_DECODE);
  decode_config.supports_local_prefill_decode = true;
  std::string error;
  const std::optional<proto::ProviderDescriptor> prefill =
      build_native_provider_descriptor(prefill_config, &error);
  const std::optional<proto::ProviderDescriptor> decode =
      build_native_provider_descriptor(decode_config, &error);
  ASSERT_TRUE(prefill.has_value()) << error;
  ASSERT_TRUE(decode.has_value()) << error;
  ASSERT_EQ(prefill->serving().execution_modes_size(), 2);
  EXPECT_EQ(prefill->serving().execution_modes(1).mode(),
            proto::EXECUTION_MODE_PREFILL_ONLY);
  EXPECT_EQ(prefill->serving().execution_modes(1).selection_order(),
            proto::SELECTION_ORDER_P_ONLY);
  ASSERT_EQ(decode->serving().execution_modes_size(), 2);
  EXPECT_EQ(decode->serving().execution_modes(1).mode(),
            proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE);
  EXPECT_EQ(decode->serving().execution_modes(1).selection_order(),
            proto::SELECTION_ORDER_D_ONLY);

  prefill_config.supports_local_prefill_decode = true;
  EXPECT_FALSE(
      build_native_provider_descriptor(prefill_config, &error).has_value());
  decode_config.supports_prefill_only = true;
  EXPECT_FALSE(
      build_native_provider_descriptor(decode_config, &error).has_value());
}

TEST(NativeProviderContractTest, InvalidInputsFailClosed) {
  NativeProviderDescriptorConfig config =
      make_config(proto::ENGINE_ROLE_PREFILL);
  std::string error;
  config.role = proto::ENGINE_ROLE_UNSPECIFIED;
  EXPECT_FALSE(build_native_provider_descriptor(config, &error).has_value());
  EXPECT_FALSE(error.empty());

  config = make_config(proto::ENGINE_ROLE_PREFILL);
  config.dp = 0;
  EXPECT_FALSE(build_native_provider_descriptor(config, &error).has_value());

  config = make_config(proto::ENGINE_ROLE_PREFILL);
  config.cache_groups.emplace_back(config.cache_groups.front());
  EXPECT_FALSE(build_native_provider_descriptor(config, &error).has_value());

  config = make_config(proto::ENGINE_ROLE_PREFILL);
  config.hash_version = 0;
  EXPECT_FALSE(build_native_provider_descriptor(config, &error).has_value());

  config = make_config(proto::ENGINE_ROLE_PREFILL);
  config.hash_seed = 0;
  EXPECT_FALSE(build_native_provider_descriptor(config, &error).has_value());
}

TEST(NativeProviderContractTest, BuildsFullPerDpStateFromBlockCounts) {
  NativeEngineStateBuilder builder(make_descriptor(proto::ENGINE_ROLE_DECODE));
  NativeEngineStateSnapshot snapshot{
      .kv_free_blocks = {85, 3},
      .kv_used_blocks = {25, 7},
      .kv_total_blocks = {110, 10},
  };
  std::string error;
  const std::optional<proto::EngineState> state =
      builder.build(snapshot, &error);

  ASSERT_TRUE(state.has_value()) << error;
  EXPECT_EQ(state->state_quality(), proto::STATE_QUALITY_FULL);
  EXPECT_EQ(state->lifecycle(), proto::ENGINE_LIFECYCLE_READY);
  EXPECT_EQ(state->ownership(), proto::ENGINE_OWNERSHIP_OWNED);
  EXPECT_EQ(state->deep_health(), proto::HEALTH_STATUS_UNKNOWN);
  ASSERT_EQ(state->per_dp_size(), 2);
  EXPECT_DOUBLE_EQ(state->per_dp(0).kv_used_ratio(), 25.0 / 110.0);
  EXPECT_EQ(state->per_dp(0).kv_free_blocks(), 85u);
  EXPECT_EQ(state->per_dp(0).kv_total_blocks(), 110u);
  EXPECT_EQ(state->per_dp(0).admission_credit(), 85u);
  EXPECT_DOUBLE_EQ(state->per_dp(1).kv_used_ratio(), 0.7);
  EXPECT_EQ(state->per_dp(1).kv_free_blocks(), 3u);
  EXPECT_EQ(state->per_dp(1).kv_total_blocks(), 10u);
  // READY credit follows free blocks. Drain no longer zeroes it.
  EXPECT_EQ(state->per_dp(1).admission_credit(), 3u);
  EXPECT_TRUE(builder.build(snapshot, &error).has_value());
}

TEST(NativeProviderContractTest, RejectsAvailableBeyondPhysicalCapacity) {
  NativeEngineStateBuilder builder(make_descriptor(proto::ENGINE_ROLE_DECODE));
  std::string error;
  EXPECT_FALSE(builder
                   .build(
                       NativeEngineStateSnapshot{
                           .kv_free_blocks = {7, 0},
                           .kv_used_blocks = {1, 1},
                           .kv_total_blocks = {6, 6},
                       },
                       &error)
                   .has_value());
  EXPECT_FALSE(error.empty());
}

TEST(NativeProviderContractTest, IncompleteSnapshotDoesNotPublishState) {
  NativeEngineStateBuilder builder(make_descriptor(proto::ENGINE_ROLE_PREFILL));
  std::string error;
  EXPECT_FALSE(builder
                   .build(
                       NativeEngineStateSnapshot{
                           .kv_free_blocks = {1},
                           .kv_used_blocks = {0},
                           .kv_total_blocks = {1},
                       },
                       &error)
                   .has_value());

  const std::optional<proto::EngineState> state = builder.build(
      NativeEngineStateSnapshot{
          .kv_free_blocks = {1, 2},
          .kv_used_blocks = {0, 0},
          .kv_total_blocks = {1, 2},
      },
      &error);
  ASSERT_TRUE(state.has_value()) << error;
  EXPECT_EQ(state->per_dp_size(), 2);
}

TEST(NativeProviderContractTest, DrainingLifecycleIsRejected) {
  NativeEngineStateBuilder builder(make_descriptor(proto::ENGINE_ROLE_DECODE));
  std::string error;
  EXPECT_FALSE(builder
                   .build(
                       NativeEngineStateSnapshot{
                           .kv_free_blocks = {10, 20},
                           .kv_used_blocks = {5, 5},
                           .kv_total_blocks = {15, 25},
                           .lifecycle = proto::ENGINE_LIFECYCLE_DRAINING,
                       },
                       &error)
                   .has_value());

  const std::optional<proto::EngineState> ready = builder.build(
      NativeEngineStateSnapshot{
          .kv_free_blocks = {10, 20},
          .kv_used_blocks = {5, 5},
          .kv_total_blocks = {15, 25},
      },
      &error);
  ASSERT_TRUE(ready.has_value()) << error;
  EXPECT_EQ(ready->lifecycle(), proto::ENGINE_LIFECYCLE_READY);
}

TEST(NativeProviderContractTest, StateBuilderRejectsMissingPerDpCapability) {
  proto::ProviderDescriptor descriptor =
      make_descriptor(proto::ENGINE_ROLE_PREFILL);
  descriptor.clear_capabilities();
  NativeEngineStateBuilder builder(std::move(descriptor));
  std::string error;
  EXPECT_FALSE(builder
                   .build(
                       NativeEngineStateSnapshot{
                           .kv_free_blocks = {1, 2},
                           .kv_used_blocks = {0, 0},
                           .kv_total_blocks = {1, 2},
                       },
                       &error)
                   .has_value());
  EXPECT_FALSE(error.empty());
}

}  // namespace
}  // namespace xllm
