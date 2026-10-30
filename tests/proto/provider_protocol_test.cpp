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

#include <gtest/gtest.h>

#include <string>

#include "disagg_pd.pb.h"
#include "provider.pb.h"
#include "xllm_service.pb.h"
#include "xservice.pb.h"

namespace xllm::proto {
namespace {

ProviderDescriptor make_xllm_descriptor() {
  ProviderDescriptor descriptor;
  descriptor.set_contract_version(1);
  ProviderIdentity* identity = descriptor.mutable_identity();
  identity->set_engine_uid("engine-p-0");
  identity->set_provider_id(PROVIDER_ID_XLLM_NATIVE);
  identity->set_runtime_family("xllm");
  identity->set_runtime_version("08d6de0c");
  identity->set_plugin_version("builtin");
  identity->set_hardware_runtime_version("cann-8");
  identity->set_protocol_version(1);

  ProviderEndpoint* endpoint = descriptor.mutable_endpoint();
  endpoint->set_control_transport("brpc");
  endpoint->set_data_transport("mooncake");
  endpoint->set_address("127.0.0.1:19000");

  ServingDescriptor* serving = descriptor.mutable_serving();
  serving->set_role(ENGINE_ROLE_PREFILL);
  ExecutionModeSpec* mode = serving->add_execution_modes();
  mode->set_mode(EXECUTION_MODE_REMOTE_PD);
  mode->set_transfer_mode(TRANSFER_MODE_LAYERWISE_PUSH);
  mode->set_selection_order(SELECTION_ORDER_P_FIRST);
  mode->set_binding_stage(BINDING_STAGE_BEFORE_PREFILL);
  serving->add_api_features("chat_completions");

  ModelDescriptor* model = descriptor.mutable_model();
  model->set_model_revision("model-r1");
  model->set_tokenizer_revision("tokenizer-r1");
  model->set_chat_template_digest("template-sha256");
  model->set_quantization("bf16");
  model->set_renderer_digest("renderer-sha256");

  TopologyDescriptor* topology = descriptor.mutable_topology();
  topology->set_soc("Ascend910B");
  topology->set_device_count(8);
  topology->set_tp(8);
  topology->set_dp(1);
  topology->set_pp(1);
  topology->set_ep(1);
  topology->set_cp(1);

  KVDescriptor* kv = descriptor.mutable_kv();
  kv->set_kv_layout_digest("kv-layout-sha256");
  kv->set_cache_dtype("bf16");
  kv->set_block_size(128);
  kv->add_cache_groups("full_attention");
  kv->set_head_shard_mapping_digest("head-map-sha256");
  kv->set_connector("mooncake");
  kv->set_connector_version("1");
  kv->add_transfer_modes(TRANSFER_MODE_LAYERWISE_PUSH);
  kv->set_logical_bytes_per_token(4096);

  SchedulerDescriptor* scheduler = descriptor.mutable_scheduler();
  scheduler->set_scheduler_class("disagg_pd");
  scheduler->set_max_num_seqs(256);
  scheduler->set_max_num_batched_tokens(32768);
  scheduler->set_scheduler_policy_digest("scheduler-sha256");

  descriptor.add_capabilities(PROVIDER_CAPABILITY_REMOTE_PD_LAYERWISE_PUSH);
  descriptor.add_capabilities(PROVIDER_CAPABILITY_NATIVE_RESERVATION);
  descriptor.add_capabilities(PROVIDER_CAPABILITY_ATTEMPT_QUERY);
  descriptor.add_capabilities(PROVIDER_CAPABILITY_CANCEL_FENCE);
  descriptor.add_capabilities(PROVIDER_CAPABILITY_ENGINE_LOCAL_DEADLINE);
  descriptor.add_capabilities(PROVIDER_CAPABILITY_SELF_FENCING);
  descriptor.set_profile_digest("profile-sha256");
  return descriptor;
}

TEST(ProviderProtocolTest, DescriptorRoundtripPreservesCompatibilityFields) {
  ProviderDescriptor descriptor = make_xllm_descriptor();
  std::string encoded;
  ASSERT_TRUE(descriptor.SerializeToString(&encoded));

  ProviderDescriptor decoded;
  ASSERT_TRUE(decoded.ParseFromString(encoded));
  EXPECT_EQ(decoded.identity().provider_id(), PROVIDER_ID_XLLM_NATIVE);
  EXPECT_EQ(decoded.identity().engine_uid(), "engine-p-0");
  EXPECT_EQ(decoded.model().model_revision(), "model-r1");
  EXPECT_EQ(decoded.model().renderer_digest(), "renderer-sha256");
  EXPECT_EQ(decoded.kv().kv_layout_digest(), "kv-layout-sha256");
  ASSERT_TRUE(decoded.kv().has_logical_bytes_per_token());
  EXPECT_EQ(decoded.kv().logical_bytes_per_token(), 4096u);
  EXPECT_EQ(decoded.serving().execution_modes(0).selection_order(),
            SELECTION_ORDER_P_FIRST);
  EXPECT_EQ(decoded.capabilities_size(), 6);
}

TEST(ProviderProtocolTest, OptionalMetricsDistinguishUnknownFromZero) {
  EngineState state;
  state.set_engine_uid("engine-d-0");
  state.set_lifecycle(ENGINE_LIFECYCLE_READY);
  state.set_ownership(ENGINE_OWNERSHIP_OWNED);
  state.set_shallow_health(HEALTH_STATUS_HEALTHY);
  state.set_state_quality(STATE_QUALITY_PARTIAL);
  PerDpEngineState* dp = state.add_per_dp();
  dp->set_dp_rank(0);
  dp->set_running(0);

  std::string encoded;
  ASSERT_TRUE(state.SerializeToString(&encoded));
  EngineState decoded;
  ASSERT_TRUE(decoded.ParseFromString(encoded));
  ASSERT_EQ(decoded.per_dp_size(), 1);
  EXPECT_TRUE(decoded.per_dp(0).has_running());
  EXPECT_EQ(decoded.per_dp(0).running(), 0);
  EXPECT_FALSE(decoded.per_dp(0).has_kv_used_ratio());
  EXPECT_EQ(decoded.deep_health(), HEALTH_STATUS_UNKNOWN);
}

TEST(ProviderProtocolTest, KVEventBatchPreservesOrderingAndAgePresence) {
  KVEventBatch batch;
  batch.set_contract_version(1);
  batch.mutable_identity()->mutable_engine()->set_provider_id(
      PROVIDER_ID_XLLM_NATIVE);
  batch.mutable_identity()->mutable_engine()->set_profile_digest("profile");
  batch.mutable_identity()->mutable_engine()->set_engine_uid("engine");
  batch.mutable_identity()->set_model_revision("model");
  batch.mutable_identity()->set_cache_epoch(3);
  KVEvent* event = batch.add_events();
  event->set_event_seq(7);
  event->set_kind(KV_EVENT_KIND_CLEARED);
  event->set_reason(KV_EVENT_REASON_CACHE_RESET);
  event->set_event_age_ms_at_publish(0);
  batch.set_last_event_seq(7);
  batch.set_batch_age_ms_at_publish(0);

  KVEventBatch decoded;
  ASSERT_TRUE(decoded.ParseFromString(batch.SerializeAsString()));
  EXPECT_EQ(decoded.identity().cache_epoch(), 3);
  ASSERT_EQ(decoded.events_size(), 1);
  EXPECT_EQ(decoded.events(0).event_seq(), 7);
  EXPECT_TRUE(decoded.events(0).has_event_age_ms_at_publish());
  EXPECT_EQ(decoded.events(0).event_age_ms_at_publish(), 0u);
  EXPECT_TRUE(decoded.has_batch_age_ms_at_publish());
  EXPECT_EQ(decoded.batch_age_ms_at_publish(), 0u);
}

TEST(ProviderProtocolTest, XServiceStatusPreservesRemoteFailureReason) {
  xllm_service::proto::Status status;
  status.set_ok(false);
  status.set_status_msg("not-current-master");

  xllm_service::proto::Status decoded;
  ASSERT_TRUE(decoded.ParseFromString(status.SerializeAsString()));
  EXPECT_FALSE(decoded.ok());
  EXPECT_EQ(decoded.status_msg(), "not-current-master");
  EXPECT_EQ(xllm_service::proto::Status::descriptor()
                ->FindFieldByName("status_msg")
                ->number(),
            3);
}

TEST(ProviderProtocolTest, CriticalFieldNumbersRemainStable) {
  const google::protobuf::Descriptor* descriptor =
      ProviderDescriptor::descriptor();
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->FindFieldByName("identity")->number(), 2);
  EXPECT_EQ(descriptor->FindFieldByName("capabilities")->number(), 9);
  EXPECT_EQ(descriptor->FindFieldByName("profile_digest")->number(), 10);

  EXPECT_EQ(ModelDescriptor::descriptor()
                ->FindFieldByName("renderer_digest")
                ->number(),
            5);
  EXPECT_EQ(EngineState::descriptor()->FindFieldByName("provider_id")->number(),
            15);
  EXPECT_EQ(EngineState::descriptor()
                ->FindFieldByName("state_age_ms_at_publish")
                ->number(),
            19);
  EXPECT_EQ(EngineState::descriptor()->FindFieldByName("drain")->number(), 20);
  EXPECT_EQ(EngineState::descriptor()
                ->FindFieldByName("lifecycle_desired_generation")
                ->number(),
            23);
  EXPECT_EQ(xllm_service::proto::HeartbeatRequest::descriptor()
                ->FindFieldByName("engine_state")
                ->number(),
            7);
  EXPECT_EQ(PerDpEngineState::descriptor()
                ->FindFieldByName("kv_total_blocks")
                ->number(),
            8);
  EXPECT_EQ(PerDpEngineState::descriptor()
                ->FindFieldByName("admission_capacity")
                ->number(),
            9);
  EXPECT_EQ(PerDpEngineState::descriptor()->FindFieldByName(
                "prefix_cache_reclaimable_blocks"),
            nullptr);
  EXPECT_EQ(PerDpEngineState::descriptor()->FindFieldByName(
                "admission_reserve_blocks"),
            nullptr);
  EXPECT_EQ(
      EngineState::descriptor()->FindFieldByName("latency_histogram_delta"),
      nullptr);
  EXPECT_EQ(EngineState::descriptor()->FindFieldByName(
                "throughput_tokens_per_second"),
            nullptr);
  EXPECT_EQ(EngineState::descriptor()->FindFieldByName("connector_state"),
            nullptr);
  EXPECT_EQ(EngineState::descriptor()->FindFieldByName("failure_counters"),
            nullptr);
  EXPECT_EQ(
      KVStreamIdentity::descriptor()->FindFieldByName("cache_epoch")->number(),
      4);
  EXPECT_EQ(KVDescriptor::descriptor()->FindFieldByName("kv_namespace"),
            nullptr);
  EXPECT_EQ(KVStreamIdentity::descriptor()->FindFieldByName("kv_namespace"),
            nullptr);
  EXPECT_EQ(
      KVDescriptor::descriptor()->FindFieldByName("hash_version")->number(),
      11);
  EXPECT_EQ(KVDescriptor::descriptor()->FindFieldByName("hash_seed")->number(),
            12);
  EXPECT_EQ(KVDescriptor::descriptor()
                ->FindFieldByName("logical_bytes_per_token")
                ->number(),
            13);
  EXPECT_EQ(KVEvent::descriptor()->FindFieldByName("event_seq")->number(), 1);
  EXPECT_EQ(KVEvent::descriptor()
                ->FindFieldByName("event_age_ms_at_publish")
                ->number(),
            5);
  EXPECT_EQ(KVEventBatch::descriptor()
                ->FindFieldByName("batch_age_ms_at_publish")
                ->number(),
            6);
  EXPECT_EQ(KVCacheSnapshotPage::descriptor()
                ->FindFieldByName("serialized_bytes")
                ->number(),
            10);
  const google::protobuf::ServiceDescriptor* xservice =
      xllm_service::proto::XllmRpcService::descriptor();
  ASSERT_NE(xservice, nullptr);
  EXPECT_NE(xservice->FindMethodByName("PushKVEvents"), nullptr);
  EXPECT_EQ(xservice->FindMethodByName("PushKVState"), nullptr);
  const google::protobuf::DescriptorPool* pool =
      google::protobuf::DescriptorPool::generated_pool();
  ASSERT_NE(pool, nullptr);
  EXPECT_EQ(pool->FindMessageTypeByName("xllm.proto.StateBatch"), nullptr);
  EXPECT_EQ(pool->FindMessageTypeByName("xllm.proto.CompactLinkStateFull"),
            nullptr);
  EXPECT_EQ(pool->FindMessageTypeByName("xllm.proto.KVStateBatch"), nullptr);
  const google::protobuf::ServiceDescriptor* engine_api =
      XllmAPIService::descriptor();
  ASSERT_NE(engine_api, nullptr);
  EXPECT_EQ(engine_api->FindMethodByName("ExecuteProviderLifecycle"), nullptr);
  EXPECT_EQ(engine_api->FindMethodByName("QueryProviderLifecycle"), nullptr);
  EXPECT_NE(engine_api->FindMethodByName("GetKVCacheSnapshot"), nullptr);
  const google::protobuf::ServiceDescriptor* disagg =
      DisaggPDService::descriptor();
  ASSERT_NE(disagg, nullptr);
  EXPECT_EQ(disagg->FindMethodByName("GetKVCacheSnapshot"), nullptr);
  EXPECT_EQ(disagg->FindMethodByName("BeginTransfer"), nullptr);
  EXPECT_EQ(disagg->FindMethodByName("CompleteTransfer"), nullptr);
  EXPECT_NE(disagg->FindMethodByName("CancelRequest"), nullptr);
  EXPECT_EQ(disagg->FindMethodByName("QueryRequest"), nullptr);
  EXPECT_EQ(disagg->FindMethodByName("CommitGeneration"), nullptr);
  EXPECT_EQ(disagg->FindMethodByName("EnsureLink"), nullptr);
  EXPECT_EQ(disagg->FindMethodByName("DrainLink"), nullptr);
  EXPECT_NE(disagg->FindMethodByName("LinkInstance"), nullptr);
  EXPECT_NE(disagg->FindMethodByName("UnlinkInstance"), nullptr);
  EXPECT_NE(pool->FindMessageTypeByName("xllm.proto.ExecutionAttemptId"),
            nullptr);
  EXPECT_NE(pool->FindEnumTypeByName("xllm.proto.AdmissionReason"), nullptr);
  EXPECT_NE(pool->FindEnumTypeByName("xllm.proto.AttemptLifecycleState"),
            nullptr);
  EXPECT_NE(pool->FindEnumTypeByName("xllm.proto.HolderConvergenceProof"),
            nullptr);
}

}  // namespace
}  // namespace xllm::proto
