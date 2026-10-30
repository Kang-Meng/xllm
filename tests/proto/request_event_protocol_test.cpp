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

#include <atomic>
#include <chrono>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "chat.pb.h"
#include "completion.pb.h"
#include "disagg_pd.pb.h"
#include "observability.pb.h"

namespace xllm::proto {
namespace {

RequestEvent make_ingress_event() {
  RequestEvent event;
  event.set_schema_version(1);
  event.mutable_correlation()->set_global_request_id("g");
  event.mutable_correlation()->set_trace_id("t");
  event.mutable_correlation()->set_request_uid("r");
  event.mutable_correlation()->set_attempt_seq(0);
  event.mutable_correlation()->set_global_request_id_source(
      CORRELATION_ID_SOURCE_UPSTREAM);
  event.mutable_correlation()->set_trace_id_source(
      CORRELATION_ID_SOURCE_UPSTREAM);
  event.set_event_seq(0);
  event.set_event_type(REQUEST_EVENT_TYPE_INGRESS);
  event.set_owner_role(EVENT_OWNER_ROLE_SERVICE);
  event.set_result(EVENT_RESULT_SUCCEEDED);
  event.set_build_id("b");
  event.set_stage_duration_validity(METRIC_VALIDITY_NOT_APPLICABLE);
  return event;
}

TEST(RequestEventProtocolTest, MinimalIngressGoldenWireIsStable) {
  const std::string golden(
      "\x08\x01"
      "\x12\x0f\x0a\x01g\x12\x01t\x1a\x01r\x20\x00\x28\x01\x30\x01"
      "\x18\x00"
      "\x20\x01"
      "\x28\x02"
      "\x50\x02"
      "\x6a\x01"
      "b"
      "\x88\x01\x02",
      33);

  std::string encoded;
  ASSERT_TRUE(make_ingress_event().SerializeToString(&encoded));
  EXPECT_EQ(encoded, golden);

  RequestEvent decoded;
  ASSERT_TRUE(decoded.ParseFromString(golden));
  EXPECT_TRUE(decoded.correlation().has_attempt_seq());
  EXPECT_EQ(decoded.correlation().attempt_seq(), 0);
  EXPECT_TRUE(decoded.has_event_seq());
  EXPECT_EQ(decoded.event_seq(), 0);
  EXPECT_EQ(decoded.correlation().global_request_id_source(),
            CORRELATION_ID_SOURCE_UPSTREAM);
}

TEST(RequestEventProtocolTest, DisaggV2ResourceFieldsRoundtrip) {
  DisaggRequests batch;
  batch.set_prefill_name("prefill-1");
  DisaggRequests decoded_batch;
  ASSERT_TRUE(decoded_batch.ParseFromString(batch.SerializeAsString()));
  EXPECT_EQ(decoded_batch.prefill_name(), "prefill-1");

  DisaggRequest request;
  request.set_service_req_id("01234567-89ab-7cde-bf01-23456789abcd");
  DisaggRequest decoded;
  ASSERT_TRUE(decoded.ParseFromString(request.SerializeAsString()));
  EXPECT_EQ(decoded.service_req_id(), request.service_req_id());

  DisaggGenerationsRequest generation;
  generation.set_upstream_elapsed_seconds(1.25);
  DisaggGenerationsRequest decoded_generation;
  ASSERT_TRUE(
      decoded_generation.ParseFromString(generation.SerializeAsString()));
  EXPECT_DOUBLE_EQ(decoded_generation.upstream_elapsed_seconds(), 1.25);

  DisaggStreamGeneration output;
  output.set_service_req_id("01234567-89ab-7cde-bf01-23456789abcd");
  output.set_sender_engine_uid("decode-1");
  DisaggStreamGeneration decoded_output;
  ASSERT_TRUE(decoded_output.ParseFromString(output.SerializeAsString()));
  EXPECT_EQ(decoded_output.service_req_id(), output.service_req_id());
  EXPECT_EQ(decoded_output.sender_engine_uid(), "decode-1");

  InstanceClusterInfo cluster_info;
  cluster_info.set_instance_name("prefill-0");
  cluster_info.add_cluster_ids(7);
  cluster_info.add_addrs("10.0.0.1:12345");
  cluster_info.add_ports(12345);
  cluster_info.set_dp_size(1);
  cluster_info.set_kv_split_size(1);
  InstanceClusterInfo decoded_cluster_info;
  ASSERT_TRUE(
      decoded_cluster_info.ParseFromString(cluster_info.SerializeAsString()));
  EXPECT_EQ(decoded_cluster_info.instance_name(), "prefill-0");
}

TEST(RequestEventProtocolTest, ApiRequestKeepsCacheNamespaceWithoutDeadline) {
  EXPECT_EQ(
      CompletionRequest::descriptor()->FindFieldByName("remaining_deadline_ms"),
      nullptr);
  EXPECT_EQ(ChatRequest::descriptor()->FindFieldByName("remaining_deadline_ms"),
            nullptr);
  EXPECT_EQ(CompletionRequest::descriptor()->FindFieldByName(
                "first_event_retry_budget_ms"),
            nullptr);
  EXPECT_EQ(CompletionRequest::descriptor()->FindFieldByName(
                "first_event_dispatch_margin_ms"),
            nullptr);
  EXPECT_EQ(
      ChatRequest::descriptor()->FindFieldByName("first_event_retry_budget_ms"),
      nullptr);
  EXPECT_EQ(ChatRequest::descriptor()->FindFieldByName(
                "first_event_dispatch_margin_ms"),
            nullptr);
  EXPECT_EQ(CompletionRequest::descriptor()->FindFieldByName("execution_plan"),
            nullptr);
  EXPECT_EQ(ChatRequest::descriptor()->FindFieldByName("execution_plan"),
            nullptr);
}

TEST(RequestEventProtocolTest, OptionalWorkloadMetricsPreserveUnknownAndZero) {
  RequestEvent event = make_ingress_event();
  event.mutable_workload()->set_prompt_tokens(0);
  RequestMetric* metric = event.mutable_metric();
  metric->set_kind(REQUEST_METRIC_KIND_SERVER_ITL);
  metric->set_validity(METRIC_VALIDITY_VALID);
  metric->set_invalid_reason(INVALID_METRIC_REASON_NONE);
  metric->set_duration_ns(0);
  metric->set_measurement_boundary("service_response_write");

  RequestEvent decoded;
  ASSERT_TRUE(decoded.ParseFromString(event.SerializeAsString()));
  EXPECT_TRUE(decoded.workload().has_prompt_tokens());
  EXPECT_EQ(decoded.workload().prompt_tokens(), 0);
  EXPECT_FALSE(decoded.workload().has_output_tokens());
  EXPECT_TRUE(decoded.metric().has_duration_ns());
  EXPECT_EQ(decoded.metric().duration_ns(), 0);

  event.mutable_workload()->set_predicted_prefill_hit_tokens(0);
  event.mutable_workload()->set_predicted_decode_hit_tokens(7);
  event.mutable_workload()->set_predicted_transfer_bytes(4096);
  event.mutable_workload()->set_actual_prefill_hit_tokens(0);
  event.mutable_workload()->set_actual_decode_hit_tokens(6);
  event.mutable_workload()->set_skipped_transfer_bytes(3072);
  event.mutable_workload()->set_shadow_prefill_host_hit_tokens_ub(5);
  event.mutable_workload()->set_shadow_decode_host_hit_tokens_ub(2);
  ASSERT_TRUE(decoded.ParseFromString(event.SerializeAsString()));
  EXPECT_TRUE(decoded.workload().has_predicted_prefill_hit_tokens());
  EXPECT_EQ(decoded.workload().predicted_prefill_hit_tokens(), 0);
  EXPECT_EQ(decoded.workload().predicted_decode_hit_tokens(), 7);
  EXPECT_EQ(decoded.workload().predicted_transfer_bytes(), 4096);
  EXPECT_TRUE(decoded.workload().has_actual_prefill_hit_tokens());
  EXPECT_EQ(decoded.workload().actual_prefill_hit_tokens(), 0);
  EXPECT_EQ(decoded.workload().actual_decode_hit_tokens(), 6);
  EXPECT_EQ(decoded.workload().skipped_transfer_bytes(), 3072);
  EXPECT_EQ(decoded.workload().shadow_prefill_host_hit_tokens_ub(), 5);
  EXPECT_EQ(decoded.workload().shadow_decode_host_hit_tokens_ub(), 2);
}

TEST(RequestEventProtocolTest, RuntimeProfileRoundtripKeepsCalibrationKeys) {
  RequestEvent event = make_ingress_event();
  RuntimeProfileIdentity* profile = event.mutable_runtime_profile();
  profile->set_provider_id(PROVIDER_ID_XLLM_NATIVE);
  profile->set_runtime_version("xllm-v1");
  profile->set_plugin_version("native-v1");
  profile->set_hardware_runtime_version("cann-v1");
  profile->set_profile_digest("profile-sha256");
  profile->set_model_revision("model-r1");
  profile->set_execution_mode(EXECUTION_MODE_REMOTE_PD);

  RequestEvent decoded;
  ASSERT_TRUE(decoded.ParseFromString(event.SerializeAsString()));
  EXPECT_EQ(decoded.runtime_profile().provider_id(), PROVIDER_ID_XLLM_NATIVE);
  EXPECT_EQ(decoded.runtime_profile().profile_digest(), "profile-sha256");
  EXPECT_EQ(decoded.runtime_profile().execution_mode(),
            EXECUTION_MODE_REMOTE_PD);
}

TEST(RequestEventProtocolTest, CriticalFieldsAndLowCardinalityEnumsStayStable) {
  const google::protobuf::Descriptor* descriptor = RequestEvent::descriptor();
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->FindFieldByName("correlation")->number(), 2);
  EXPECT_EQ(descriptor->FindFieldByName("stage_duration_ns")->number(), 8);
  EXPECT_EQ(descriptor->FindFieldByName("metric")->number(), 16);
  EXPECT_EQ(descriptor->FindFieldByName("stage_duration_validity")->number(),
            17);
  const google::protobuf::Descriptor* workload = WorkloadShape::descriptor();
  ASSERT_NE(workload, nullptr);
  EXPECT_EQ(workload->FindFieldByName("predicted_prefill_hit_tokens")->number(),
            6);
  EXPECT_EQ(workload->FindFieldByName("skipped_transfer_bytes")->number(), 11);
  EXPECT_EQ(
      workload->FindFieldByName("shadow_prefill_host_hit_tokens_ub")->number(),
      12);
  EXPECT_EQ(workload->FindFieldByName("shadow_decode_store_hit_tokens_ub"),
            nullptr);
  EXPECT_EQ(descriptor->FindFieldByName("prompt"), nullptr);
  EXPECT_EQ(descriptor->FindFieldByName("output"), nullptr);
  EXPECT_EQ(descriptor->FindFieldByName("token_ids"), nullptr);
  EXPECT_EQ(REQUEST_EVENT_TYPE_RESOURCE_RELEASE, 14);
  EXPECT_EQ(INVALID_METRIC_REASON_EVENT_DROPPED, 7);
  EXPECT_EQ(CompletionRequest::descriptor()->FindFieldByName("correlation"),
            nullptr);
  EXPECT_EQ(ChatRequest::descriptor()->FindFieldByName("correlation"), nullptr);
  EXPECT_EQ(CompletionRequest::descriptor()->FindFieldByName("kv_namespace"),
            nullptr);
  EXPECT_EQ(ChatRequest::descriptor()->FindFieldByName("kv_namespace"),
            nullptr);
  EXPECT_EQ(
      ChatRequest::descriptor()->FindFieldByName("response_format")->number(),
      47);
  EXPECT_EQ(
      CompletionRequest::descriptor()->FindFieldByName("remaining_deadline_ms"),
      nullptr);
  EXPECT_EQ(CompletionRequest::descriptor()->FindFieldByName("execution_plan"),
            nullptr);
  EXPECT_EQ(ChatRequest::descriptor()->FindFieldByName("remaining_deadline_ms"),
            nullptr);
  EXPECT_EQ(ChatRequest::descriptor()->FindFieldByName("execution_plan"),
            nullptr);
  EXPECT_EQ(DisaggRequest::descriptor()->FindFieldByName("correlation"),
            nullptr);
  EXPECT_EQ(DisaggRequest::descriptor()->FindFieldByName(
                "expected_decode_incarnation_id"),
            nullptr);
  EXPECT_EQ(
      DisaggRequest::descriptor()->FindFieldByName("remaining_deadline_ms"),
      nullptr);
  EXPECT_EQ(
      DisaggRequests::descriptor()->FindFieldByName("prefill_incarnation_id"),
      nullptr);
  EXPECT_EQ(DisaggRequests::descriptor()->FindFieldByName("link_generation"),
            nullptr);
  EXPECT_EQ(Routing::descriptor()->FindFieldByName("prefill_name")->number(),
            1);
  EXPECT_EQ(Routing::descriptor()->FindFieldByName("decode_name")->number(), 2);
  EXPECT_EQ(Routing::descriptor()->FindFieldByName("decode_incarnation_id"),
            nullptr);
  EXPECT_EQ(Routing::descriptor()->FindFieldByName("prefill_incarnation_id"),
            nullptr);
  EXPECT_EQ(Routing::descriptor()->FindFieldByName("coordinator_epoch"),
            nullptr);
  EXPECT_EQ(Routing::descriptor()->FindFieldByName("adjacency_generation"),
            nullptr);
  EXPECT_EQ(Routing::descriptor()->FindFieldByName("link_generation"), nullptr);
  EXPECT_EQ(DisaggResponse::descriptor()->FindFieldByName("admission"),
            nullptr);
  EXPECT_NE(
      google::protobuf::DescriptorPool::generated_pool()->FindEnumTypeByName(
          "xllm.proto.AdmissionReason"),
      nullptr);
  EXPECT_EQ(DisaggGenerationsRequest::descriptor()->FindFieldByName(
                "handoff_fingerprint"),
            nullptr);
  EXPECT_EQ(
      DisaggGenerationsRequest::descriptor()->FindFieldByName("correlation"),
      nullptr);
  EXPECT_EQ(DisaggGenerationsRequest::descriptor()
                ->FindFieldByName("upstream_elapsed_seconds")
                ->number(),
            21);
  EXPECT_EQ(
      DisaggStreamGeneration::descriptor()->FindFieldByName("output_event_seq"),
      nullptr);
  EXPECT_EQ(
      DisaggStreamGeneration::descriptor()->FindFieldByName("attempt_seq"),
      nullptr);
  EXPECT_EQ(DisaggStreamGeneration::descriptor()
                ->FindFieldByName("sender_engine_uid")
                ->number(),
            10);
  EXPECT_EQ(DisaggStreamGeneration::descriptor()->FindFieldByName(
                "sender_incarnation_id"),
            nullptr);
  EXPECT_EQ(
      InstanceClusterInfo::descriptor()->FindFieldByName("incarnation_id"),
      nullptr);
  EXPECT_EQ(Status::descriptor()->FindFieldByName("delivery_code")->number(),
            2);
  EXPECT_EQ(Status::descriptor()->FindFieldByName("status_msg")->number(), 3);
  EXPECT_EQ(GENERATION_DELIVERY_CODE_ACCEPTED, 1);
  EXPECT_EQ(GENERATION_DELIVERY_CODE_AFFINITY_MISSING, 10);

  const google::protobuf::ServiceDescriptor* disagg_service =
      DisaggPDService::descriptor();
  ASSERT_NE(disagg_service, nullptr);
  EXPECT_EQ(disagg_service->FindMethodByName("CommitGeneration"), nullptr);
  EXPECT_EQ(disagg_service->FindMethodByName("BeginTransfer"), nullptr);
  EXPECT_NE(disagg_service->FindMethodByName("CancelRequest"), nullptr);
  EXPECT_EQ(disagg_service->FindMethodByName("QueryRequest"), nullptr);
  EXPECT_EQ(disagg_service->FindMethodByName("EnsureLink"), nullptr);
  EXPECT_EQ(disagg_service->FindMethodByName("DrainLink"), nullptr);
  EXPECT_NE(disagg_service->FindMethodByName("AddNewRequests"), nullptr);
  EXPECT_NE(disagg_service->FindMethodByName("FirstGeneration"), nullptr);
  EXPECT_NE(disagg_service->FindMethodByName("LinkInstance"), nullptr);
  EXPECT_NE(disagg_service->FindMethodByName("UnlinkInstance"), nullptr);
  EXPECT_NE(disagg_service->FindMethodByName("ReleaseReservation"), nullptr);
}

}  // namespace
}  // namespace xllm::proto
