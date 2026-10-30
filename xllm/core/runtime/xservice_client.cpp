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

#include "core/runtime/xservice_client.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
#include <unistd.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <string_view>
#include <unordered_map>

#include "common/bounded_batch.h"
#include "common/bounded_retry.h"
#include "common/metrics.h"
#include "core/common/etcd_schema.h"
#include "core/common/public_error.h"
#include "core/framework/config/distributed_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/service_config.h"
#include "core/framework/prefix_cache/prefix_cache_journal.h"
#include "core/framework/request/request_output_proto_adapter.h"
#include "core/runtime/native_provider_runtime.h"
#include "scheduler/continuous_scheduler.h"
#include "util/env_var.h"
#include "util/hash_util.h"
#include "util/net.h"
#include "util/uuid.h"
#include "util/verbose_trace_logger.h"

namespace xllm {
namespace {

inline constexpr uint32_t kKVEventContractVersion = 1;

proto::KVEventKind prefix_cache_kind_to_proto(PrefixCacheEventKind kind) {
  switch (kind) {
    case PrefixCacheEventKind::STORED:
      return proto::KV_EVENT_KIND_STORED;
    case PrefixCacheEventKind::REMOVED:
      return proto::KV_EVENT_KIND_REMOVED;
    case PrefixCacheEventKind::CLEARED:
      return proto::KV_EVENT_KIND_CLEARED;
  }
  return proto::KV_EVENT_KIND_UNSPECIFIED;
}

proto::KVEventReason prefix_cache_reason_to_proto(PrefixCacheEventKind kind) {
  switch (kind) {
    case PrefixCacheEventKind::STORED:
      return proto::KV_EVENT_REASON_PREFIX_PUBLISHED;
    case PrefixCacheEventKind::REMOVED:
      return proto::KV_EVENT_REASON_CAPACITY_EVICTED;
    case PrefixCacheEventKind::CLEARED:
      return proto::KV_EVENT_REASON_CACHE_RESET;
  }
  return proto::KV_EVENT_REASON_UNSPECIFIED;
}

bool same_kv_identity(const proto::KVStreamIdentity& left,
                      const proto::KVStreamIdentity& right) {
  return left.engine().provider_id() == right.engine().provider_id() &&
         left.engine().profile_digest() == right.engine().profile_digest() &&
         left.engine().engine_uid() == right.engine().engine_uid() &&
         left.model_revision() == right.model_revision();
}

}  // namespace
namespace {
constexpr const char* kEtcdUsernameEnvVar = "ETCD_USERNAME";
constexpr const char* kEtcdPasswordEnvVar = "ETCD_PASSWORD";
static std::unordered_map<xllm_service::proto::InstanceType, std::string>
    ETCD_KEYS_PREFIX_MAP = {
        {xllm_service::proto::InstanceType::DEFAULT, "XLLM:DEFAULT:"},
        {xllm_service::proto::InstanceType::PREFILL, "XLLM:PREFILL:"},
        {xllm_service::proto::InstanceType::DECODE, "XLLM:DECODE:"},
        {xllm_service::proto::InstanceType::MIX, "XLLM:MIX:"},
};

struct ControlPlaneFailure {
  std::string reason;
  std::string detail;
};

ControlPlaneFailure control_plane_failure(
    bool issued,
    const brpc::Controller& controller,
    const xllm_service::proto::Status& response) {
  if (!issued) {
    return {.reason = "MASTER_STUB_UNAVAILABLE",
            .detail = "master RPC stub is unavailable"};
  }
  if (controller.Failed()) {
    return {.reason = "RPC_FAILED", .detail = controller.ErrorText()};
  }
  return {.reason = "REMOTE_REJECTED",
          .detail = response.status_msg().empty()
                        ? "remote Service returned ok=false"
                        : response.status_msg()};
}

void log_control_plane_failure(const char* lane,
                               const std::string& master_addr,
                               const ControlPlaneFailure& failure,
                               const FailureObservation& observation) {
  if (observation.action == FailureReportAction::kFirstFailure) {
    LOG(WARNING) << "engine_control_state_change lane=" << lane
                 << " lifecycle=DEGRADED master=" << master_addr
                 << " reason=" << failure.reason
                 << " retry_delay_ms=" << observation.retry_delay.count()
                 << " detail=" << failure.detail;
  } else if (observation.action == FailureReportAction::kPeriodicSummary) {
    LOG(WARNING) << "engine_control_suppressed_summary lane=" << lane
                 << " lifecycle=DEGRADED consecutive_failures="
                 << observation.consecutive_failures
                 << " suppressed_since_last_report="
                 << observation.suppressed_since_last_report
                 << " total_suppressed=" << observation.total_suppressed
                 << " duration_ms=" << observation.duration_ms
                 << " last_reason=" << observation.last_reason;
  }
}

void log_control_plane_recovery(const char* lane,
                                const std::string& master_addr,
                                const FailureRecovery& recovery) {
  LOG(INFO) << "engine_control_state_change lane=" << lane
            << " lifecycle=READY master=" << master_addr
            << " recovered_after_failures=" << recovery.consecutive_failures
            << " suppressed=" << recovery.total_suppressed
            << " duration_ms=" << recovery.duration_ms
            << " last_reason=" << recovery.last_reason;
}

std::string parse_instance_name(const std::string& name) {
  if (name.empty()) {
    return "";
  }
  const size_t pos = name.find(':');
  if (pos == std::string::npos) {
    int32_t port = 0;
    const std::string_view port_text(name);
    const auto parsed = std::from_chars(
        port_text.data(), port_text.data() + port_text.size(), port);
    if (parsed.ec != std::errc() ||
        parsed.ptr != port_text.data() + port_text.size() || port <= 0 ||
        port > std::numeric_limits<uint16_t>::max()) {
      return name;
    }
    return xllm::net::format_host_port(xllm::net::get_local_ip_addr(), port);
  }
  return name;
}

bool check_instance_name(const std::string& name) {
  if (!xllm::net::is_valid_host_port(name)) {
    LOG(ERROR) << "Invalid instance endpoint; expected host:port or "
                  "[ipv6]:port.";
    return false;
  }

  return true;
}

bool add_provider_descriptor_to_registration(
    const proto::ProviderDescriptor& descriptor,
    nlohmann::json* registration,
    std::string* error) {
  std::string descriptor_text;
  google::protobuf::util::JsonPrintOptions options;
  options.preserve_proto_field_names = true;
  const google::protobuf::util::Status status =
      google::protobuf::util::MessageToJsonString(
          descriptor, &descriptor_text, options);
  if (!status.ok()) {
    if (error != nullptr) {
      *error =
          "Failed to encode Native Provider Descriptor: " + status.ToString();
    }
    return false;
  }
  nlohmann::json descriptor_json = nlohmann::json::parse(
      descriptor_text, nullptr, /*allow_exceptions=*/false);
  if (descriptor_json.is_discarded()) {
    if (error != nullptr) {
      *error = "Native Provider Descriptor JSON is invalid";
    }
    return false;
  }
  (*registration)["provider_id"] =
      static_cast<int32_t>(descriptor.identity().provider_id());
  (*registration)["provider_contract_version"] = descriptor.contract_version();
  (*registration)["provider_profile_digest"] = descriptor.profile_digest();
  (*registration)["backend_type"] = "xllm";
  (*registration)["provider_descriptor"] = std::move(descriptor_json);
  return true;
}

}  // namespace

bool XServiceClient::init(const std::string& etcd_addr,
                          const std::string& instance_name,
                          BlockManagerPool* block_manager_pool,
                          const std::string& etcd_namespace) {
  if (initialize_done_) {
    LOG(INFO) << "XServiceClient is already initialized, skipping.";
    return true;
  }

  if (etcd_addr.empty()) {
    LOG(ERROR) << "etcd_addr address is empty.";
    return false;
  }

  const DistributedConfig& distributed_config =
      DistributedConfig::get_instance();
  if (!etcd_schema::valid_control_plane_shard_id(
          distributed_config.control_plane_shard_id())) {
    LOG(ERROR) << "Invalid control_plane_shard_id; expected 1-64 ASCII "
                  "letters, digits, '.', '_' or '-'";
    return false;
  }
  master_service_key_ = etcd_schema::master_service_key_for_shard(
      distributed_config.control_plane_shard_id());
  xservices_key_prefix_ = etcd_schema::router_member_prefix_for_shard(
      distributed_config.control_plane_shard_id());
  if (!valid_registration_lease_timing(
          distributed_config.etcd_ttl(),
          distributed_config.heart_beat_interval())) {
    LOG(ERROR) << "Invalid service-routing membership timing: etcd_ttl="
               << distributed_config.etcd_ttl() << "s, heart_beat_interval="
               << distributed_config.heart_beat_interval()
               << "s. etcd_ttl must be at least "
               << kMinRegistrationLeaseTtlSeconds
               << "s and the reconcile interval must be in (0, ttl/3].";
    return false;
  }

  instance_name_ = instance_name;
  chan_options_.max_retry = 3;
  chan_options_.timeout_ms =
      ::xllm::ServiceConfig::get_instance().rpc_channel_timeout_ms();

  const std::string etcd_username =
      util::get_optional_string_env(kEtcdUsernameEnvVar).value_or("");
  const std::string etcd_password =
      util::get_optional_string_env(kEtcdPasswordEnvVar).value_or("");
  const bool has_etcd_auth_user = !etcd_username.empty();
  const bool has_etcd_auth_password = !etcd_password.empty();
  if (has_etcd_auth_user != has_etcd_auth_password) {
    LOG(ERROR) << "Both " << kEtcdUsernameEnvVar << " and "
               << kEtcdPasswordEnvVar << " must be set together.";
    return false;
  }
  if (has_etcd_auth_user) {
    etcd_client_ = std::make_unique<EtcdClient>(
        etcd_addr, etcd_username, etcd_password, etcd_namespace);
  } else {
    etcd_client_ = std::make_unique<EtcdClient>(etcd_addr, etcd_namespace);
  }

  // connect master xllm_service
  while (!etcd_client_->get_master_service(master_service_key_,
                                           &master_xservice_addr_)) {
    LOG(ERROR) << "Master service not set, wait 2s!";
    sleep(2);
  }

  if (!check_instance_name(master_xservice_addr_)) {
    LOG(FATAL) << "Invalid master service name format, now only support "
                  "`ip:port` style.";
    return false;
  }

  if (!connect_to_xservice(master_xservice_addr_)) {
    LOG(FATAL) << "Fail to initialize connection to master xservice server "
               << master_xservice_addr_;
    return false;
  }

  // Get and connect to all existing xllm_service instances.
  std::vector<std::string> all_services;
  if (etcd_client_->get_all_xservices(xservices_key_prefix_, &all_services)) {
    for (const auto& service_addr : all_services) {
      if (service_addr != master_xservice_addr_ &&
          check_instance_name(service_addr)) {
        connect_to_xservice(service_addr);
      }
    }
  }

  block_manager_pool_ = block_manager_pool;

  // heartbeat thread
  heartbeat_thread_ =
      std::make_unique<std::thread>(&XServiceClient::heartbeat, this);
  reconcile_thread_ = std::make_unique<std::thread>(
      &XServiceClient::reconcile_registration_loop, this);

  // watch master xllm_service change
  auto master_func = [this](const etcd::Response& response,
                            uint64_t prefix_len) {
    handle_master_service_watch(response, prefix_len);
  };
  etcd_client_->add_watch(master_service_key_, master_func);

  // watch all xllm_service changes
  auto xservices_func = [this](const etcd::Response& response,
                               uint64_t prefix_len) {
    handle_xservices_watch(response, prefix_len);
  };
  etcd_client_->add_watch(xservices_key_prefix_, xservices_func);

  initialize_done_ = true;
  return true;
}

void XServiceClient::set_scheduler(Scheduler* scheduler) {
  scheduler_ = scheduler;
}

void XServiceClient::set_engine(Engine* engine) { engine_ = engine; }

XServiceClient::~XServiceClient() { shutdown(); }

void XServiceClient::shutdown() {
  std::call_once(shutdown_once_, [this]() {
    LOG(INFO) << "XServiceClient background workers stopping...";
    exited_.store(true, std::memory_order_release);
    kv_event_wait_cv_.notify_all();
    if (etcd_client_ != nullptr) {
      etcd_client_->stop_watch();
    }
    if (heartbeat_thread_ && heartbeat_thread_->joinable()) {
      heartbeat_thread_->join();
    }
    if (reconcile_thread_ && reconcile_thread_->joinable()) {
      reconcile_thread_->join();
    }
    if (kv_event_thread_ && kv_event_thread_->joinable()) {
      kv_event_thread_->join();
    }
    heartbeat_thread_.reset();
    reconcile_thread_.reset();
    kv_event_thread_.reset();

    {
      std::lock_guard<std::mutex> lock(registration_mutex_);
      register_done_.store(false, std::memory_order_release);
      registration_key_.clear();
      engine_state_builder_.reset();
      provider_descriptor_.reset();
    }
    {
      std::unique_lock<std::shared_mutex> lock(mutex_);
      master_xservice_addr_.clear();
      xservice_stubs_.clear();
      xservice_channels_.clear();
    }
    etcd_client_.reset();
    block_manager_pool_ = nullptr;
    scheduler_ = nullptr;
    engine_ = nullptr;
    LOG(INFO) << "XServiceClient background workers stopped.";
  });
}

std::string XServiceClient::get_instance_name() { return instance_name_; }

std::optional<proto::ProviderDescriptor>
XServiceClient::get_provider_descriptor() {
  std::lock_guard<std::mutex> lock(registration_mutex_);
  return provider_descriptor_;
}

bool XServiceClient::register_instance_with_retry(const std::string& key,
                                                  const std::string& value) {
  int retry_cnt = 0;
  while (!etcd_client_->register_instance(
      key, value, ::xllm::DistributedConfig::get_instance().etcd_ttl())) {
    if (retry_cnt >= 30) {
      LOG(ERROR) << "Register instance failed! key: " << key;
      return false;
    }

    LOG(WARNING) << "Register instance failed, wait 2s! key: " << key;
    sleep(2);
    retry_cnt++;
  }
  return true;
}

bool XServiceClient::reconcile_registration() {
  std::string registration_key;
  {
    std::lock_guard<std::mutex> lock(registration_mutex_);
    if (!register_done_.load() || registration_key_.empty()) {
      return true;
    }
    registration_key = registration_key_;
  }

  const RegistrationReconcileAction action = registration_reconcile_action(
      etcd_client_->lookup_registration(registration_key));
  if (action == RegistrationReconcileAction::kKeepServing) {
    return true;
  }
  if (action == RegistrationReconcileAction::kRetryLookup) {
    LOG(WARNING) << "Could not verify instance membership in etcd; retaining "
                    "the current registration while lookup is retried: "
                 << registration_key;
    return false;
  }

  LOG(FATAL) << "Instance membership was lost in etcd. The current process "
                "must stop and a fresh process must register again. key="
             << registration_key;
  return false;
}

void XServiceClient::reconcile_registration_loop() {
  while (!exited_.load()) {
    const auto interval = std::chrono::milliseconds(static_cast<int64_t>(
        ::xllm::DistributedConfig::get_instance().heart_beat_interval() *
        1000));
    std::unique_lock<std::mutex> lock(kv_event_wait_mutex_);
    if (kv_event_wait_cv_.wait_for(
            lock, interval, [this]() { return exited_.load(); })) {
      break;
    }
    lock.unlock();
    if (!register_done_.load()) continue;

    if (!reconcile_registration()) {
      LOG(ERROR) << "Failed to reconcile instance registration in etcd.";
    }
  }
}

void XServiceClient::register_instance(const InstanceInfo& instance_info) {
  InstanceInfo registered_info = instance_info;
  if (registered_info.register_ts_ms == 0) {
    registered_info.register_ts_ms =
        static_cast<uint64_t>(absl::ToUnixMillis(absl::Now()));
  }

  std::string key_prefix = "";
  if (InstanceRole(registered_info.type) == InstanceRole::DEFAULT) {
    key_prefix =
        ETCD_KEYS_PREFIX_MAP[xllm_service::proto::InstanceType::DEFAULT];
  } else if (InstanceRole(registered_info.type) == InstanceRole::PREFILL) {
    key_prefix =
        ETCD_KEYS_PREFIX_MAP[xllm_service::proto::InstanceType::PREFILL];
  } else if (InstanceRole(registered_info.type) == InstanceRole::DECODE) {
    key_prefix =
        ETCD_KEYS_PREFIX_MAP[xllm_service::proto::InstanceType::DECODE];
  } else if (InstanceRole(registered_info.type) == InstanceRole::MIX) {
    key_prefix = ETCD_KEYS_PREFIX_MAP[xllm_service::proto::InstanceType::MIX];
  } else {
    LOG(ERROR) << "Unsupported instance type: " << registered_info.type;
    return;
  }

  const std::string key = key_prefix + registered_info.name;
  nlohmann::json registration = registered_info.serialize_to_json();
  std::shared_ptr<NativeEngineStateBuilder> engine_state_builder;
  std::optional<proto::ProviderDescriptor> provider_descriptor;
  bool journal_ready = false;
  if (engine_ != nullptr && block_manager_pool_ != nullptr) {
    std::string error;
    const std::optional<NativeProviderDescriptorConfig> config =
        collect_native_provider_descriptor_config(
            registered_info, *engine_, *block_manager_pool_, &error);
    if (config.has_value()) {
      provider_descriptor = build_native_provider_descriptor(*config, &error);
      if (!provider_descriptor.has_value() ||
          !add_provider_descriptor_to_registration(
              *provider_descriptor, &registration, &error)) {
        LOG(FATAL) << "Failed to build Native V2 registration: " << error;
        return;
      }
      engine_state_builder =
          std::make_shared<NativeEngineStateBuilder>(*provider_descriptor);

      // The journal is ready for the life of the process. Its drain watermark
      // starts at 8192 and grows with the resident hash set; it does not need
      // the block-manager capacity.
      journal_ready = block_manager_pool_->options().enable_prefix_cache();
    } else if (!error.empty()) {
      LOG(FATAL) << "Failed to collect Native V2 registration: " << error;
      return;
    }
  }
  const std::string value = registration.dump();
  {
    std::lock_guard<std::mutex> lock(registration_mutex_);
    instance_name_ = registered_info.name;
    registration_key_ = key;
    engine_state_builder_ = std::move(engine_state_builder);
    provider_descriptor_ = std::move(provider_descriptor);
  }

  if (!register_instance_with_retry(key, value)) {
    LOG(FATAL) << "Register instance to etcd failed!";
    return;
  }

  register_done_.store(true);
  // Membership is the fencing authority for KV events. Starting this sender
  // before etcd registration succeeds lets a valid journal race the Service
  // registry and produces a deterministic REMOTE_REJECTED first batch.
  if (journal_ready && kv_event_thread_ == nullptr) {
    kv_event_thread_ =
        std::make_unique<std::thread>(&XServiceClient::kv_event_loop, this);
  }
  LOG(INFO) << "Success register instance to etcd.";
}

InstanceInfo instance_info_from_proto(
    const xllm_service::proto::InstanceMetaInfo& response) {
  InstanceInfo result;
  result.name = response.name();
  result.rpc_address = response.rpc_address();
  result.register_ts_ms = response.register_ts_ms();
  if (response.type() == xllm_service::proto::InstanceType::PREFILL) {
    result.type = "PREFILL";
  } else if (response.type() == xllm_service::proto::InstanceType::DECODE) {
    result.type = "DECODE";
  } else if (response.type() == xllm_service::proto::InstanceType::MIX) {
    result.type = "MIX";
  } else {
    result.type = "DEFAULT";
  }
  // parse kv cache info
  result.cluster_ids.reserve(static_cast<size_t>(response.cluster_ids_size()));
  for (auto& cluster_id : response.cluster_ids()) {
    result.cluster_ids.emplace_back(cluster_id);
  }
  result.addrs.reserve(static_cast<size_t>(response.addrs_size()));
  for (auto& addr : response.addrs()) {
    result.addrs.emplace_back(addr);
  }
  result.dp_size = response.dp_size();
  // Copied verbatim, including proto3's 0 for "the field was not present": a
  // peer that never declared a width must stay distinguishable from one that
  // declares 1, because the PUSH transfer plan keys on the DECLARED width.
  // Folding 0 into the record's 1 default is what made a decode intending
  // dcp=4 (behind a service that does not carry InstanceMetaInfo.kv_split_size)
  // silently take the 1:1 mapping path. See plan_kv_split_widths().
  result.kv_split_size = response.kv_split_size();
  // Same verbatim contract for the CP index write mode: 0 is an instance
  // (or a serving master) that predates the field, and admission keeps the
  // legacy behavior for it rather than guessing a mode.
  result.cp_index_write_mode = response.cp_index_write_mode();
  // super_node_id: empty is a wildcard (engine opts out of cross super-node
  // isolation). Round-trip the wire value verbatim; InstanceInfo.super_node_id
  // defaults to "" so a peer that never declared one stays a wildcard.
  result.super_node_id = response.super_node_id();
  result.ports.reserve(static_cast<size_t>(response.ports_size()));
  for (auto& port : response.ports()) {
    result.ports.emplace_back(port);
  }

  return result;
}

proto::KVStreamIdentity XServiceClient::kv_stream_identity() const {
  proto::KVStreamIdentity identity;
  std::lock_guard<std::mutex> lock(registration_mutex_);
  if (!provider_descriptor_.has_value()) {
    return identity;
  }
  identity.mutable_engine()->set_provider_id(
      provider_descriptor_->identity().provider_id());
  identity.mutable_engine()->set_profile_digest(
      provider_descriptor_->profile_digest());
  identity.mutable_engine()->set_engine_uid(
      provider_descriptor_->identity().engine_uid());
  identity.set_model_revision(provider_descriptor_->model().model_revision());
  identity.set_cache_epoch(1);
  return identity;
}

proto::KVEventBatch XServiceClient::make_kv_event_batch(
    const PrefixCacheEvents& events) {
  proto::KVEventBatch message;
  message.set_contract_version(kKVEventContractVersion);
  *message.mutable_identity() = kv_stream_identity();
  message.set_commit_version(events.commit_version);
  message.set_previous_commit_version(kv_last_pushed_version_);
  if (kv_failed_commit_version_.has_value()) {
    message.set_failed_commit_version(*kv_failed_commit_version_);
    message.set_failed_commit_error(kv_failed_commit_error_);
  }
  for (const PrefixCacheEvent& event : events.events) {
    proto::KVEvent* proto_event = message.add_events();
    proto_event->set_event_seq(message.events_size());
    proto_event->set_kind(prefix_cache_kind_to_proto(event.kind));
    proto_event->set_reason(prefix_cache_reason_to_proto(event.kind));
    if (event.kind != PrefixCacheEventKind::CLEARED) {
      proto_event->mutable_block()->set_block_hash(event.hash.data,
                                                   XXH3_128BITS_HASH_VALUE_LEN);
    }
  }
  kv_last_pushed_version_ = events.commit_version;
  return message;
}

proto::KVCacheSnapshotPage XServiceClient::prefix_cache_snapshot_to_proto(
    const PrefixCacheSnapshot& snapshot) const {
  proto::KVCacheSnapshotPage page;
  page.set_contract_version(kKVEventContractVersion);
  page.set_status(proto::KV_SNAPSHOT_STATUS_OK);
  *page.mutable_identity() = kv_stream_identity();
  page.set_commit_version(snapshot.commit_version);
  uint64_t bytes = 0;
  for (const XXH3Key& hash : snapshot.hashes) {
    proto::KVBlockEntry* entry = page.add_entries();
    entry->set_block_hash(hash.data, XXH3_128BITS_HASH_VALUE_LEN);
    bytes += entry->ByteSizeLong();
  }
  page.set_serialized_bytes(bytes);
  return page;
}

void XServiceClient::note_kv_push_result(bool success,
                                         uint64_t commit_version,
                                         uint64_t service_commit_version,
                                         const std::string& error) {
  if (!success) {
    if (commit_version > 0) {
      kv_failed_commit_version_ = commit_version;
      kv_failed_commit_error_ = error;
    }
    return;
  }
  kv_failed_commit_version_.reset();
  kv_failed_commit_error_.clear();
  static_cast<void>(service_commit_version);
}

void XServiceClient::kv_event_loop() {
  constexpr size_t kMaxBatchEvents = 256;
  constexpr std::chrono::milliseconds kIdleWait(10);
  constexpr uint64_t kKeepaliveIntervalMs = 1000;
  uint64_t last_successful_publish_ms = 0;
  FailureEpisodeTracker failure_episode(
      {.backoff = {.initial_delay = std::chrono::milliseconds(50),
                   .maximum_delay = std::chrono::seconds(1)},
       .summary_interval = std::chrono::seconds(30)});
  CHECK(failure_episode.valid());
  while (!exited_.load()) {
    const uint64_t now_monotonic_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
    const PrefixCacheEvents drained =
        PrefixCacheJournal::instance()->get_prefixcache_events(kMaxBatchEvents);
    const bool due_keepalive =
        drained.events.empty() && drained.commit_version > 0 &&
        (last_successful_publish_ms == 0 ||
         now_monotonic_ms < last_successful_publish_ms ||
         now_monotonic_ms - last_successful_publish_ms >= kKeepaliveIntervalMs);
    if (drained.events.empty() && !due_keepalive) {
      std::unique_lock<std::mutex> lock(kv_event_wait_mutex_);
      kv_event_wait_cv_.wait_for(
          lock, kIdleWait, [this]() { return exited_.load(); });
      continue;
    }

    proto::KVEventBatch batch = make_kv_event_batch(drained);

    brpc::Controller controller;
    controller.set_timeout_ms(200);
    xllm_service::proto::Status response;
    std::string master_addr;
    const bool issued = with_master_stub(
        [&](xllm_service::proto::XllmRpcService_Stub* master_stub) {
          master_stub->PushKVEvents(&controller, &batch, &response, nullptr);
        },
        &master_addr,
        /*quiet=*/true);
    const bool success = issued && !controller.Failed() && response.ok();
    const ControlPlaneFailure failure =
        control_plane_failure(issued, controller, response);
    const uint64_t service_version =
        response.has_kv_commit_version() ? response.kv_commit_version() : 0;
    note_kv_push_result(
        success, batch.commit_version(), service_version, failure.detail);
    if (success) {
      last_successful_publish_ms = now_monotonic_ms;
      const std::optional<FailureRecovery> recovery =
          failure_episode.record_success();
      if (recovery.has_value()) {
        COUNTER_INC(kv_event_publish_recoveries_total);
        log_control_plane_recovery("KV_EVENT", master_addr, *recovery);
      }
    }
    if (!success) {
      COUNTER_INC(kv_event_publish_failures_total);
      const ControlPlaneFailure failure =
          control_plane_failure(issued, controller, response);
      const FailureObservation observation =
          failure_episode.record_failure(failure.reason);
      log_control_plane_failure("KV_EVENT", master_addr, failure, observation);
      if (observation.action == FailureReportAction::kSuppressed) {
        COUNTER_INC(kv_event_publish_suppressed_total);
      }
      std::unique_lock<std::mutex> lock(kv_event_wait_mutex_);
      kv_event_wait_cv_.wait_for(
          lock, observation.retry_delay, [this]() { return exited_.load(); });
    }
  }
}

proto::KVCacheSnapshotPage XServiceClient::get_kv_cache_snapshot(
    const proto::KVCacheSnapshotRequest& request) {
  const proto::KVStreamIdentity local = kv_stream_identity();
  if (local.engine().engine_uid().empty() ||
      !same_kv_identity(request.identity(), local)) {
    proto::KVCacheSnapshotPage page;
    page.set_contract_version(kKVEventContractVersion);
    page.set_status(proto::KV_SNAPSHOT_STATUS_INVALID_REQUEST);
    page.set_status_message("KV event journal is unavailable");
    return page;
  }
  return prefix_cache_snapshot_to_proto(
      PrefixCacheJournal::instance()->get_snapshot());
}

InstanceInfo XServiceClient::get_instance_info(
    const std::string& instance_name) {
  InstanceInfo result;
  brpc::Controller cntl;
  xllm_service::proto::InstanceID req;
  xllm_service::proto::InstanceMetaInfo resp;
  req.set_name(instance_name);

  std::string master_addr;
  if (!with_master_stub(
          [&](xllm_service::proto::XllmRpcService_Stub* master_stub) {
            master_stub->GetInstanceInfo(&cntl, &req, &resp, nullptr);
          },
          &master_addr)) {
    return result;
  }

  if (cntl.Failed()) {
    LOG(ERROR) << "Fail to get instance info from xservice server "
               << master_addr << ", error text: " << cntl.ErrorText();
    return result;
  }
  return instance_info_from_proto(resp);
}

void XServiceClient::heartbeat() {
  const auto heartbeat_interval = std::chrono::milliseconds(std::max<int64_t>(
      1,
      static_cast<int64_t>(
          ::xllm::DistributedConfig::get_instance().heart_beat_interval() *
          1000)));
  FailureEpisodeTracker failure_episode(
      {.backoff = {.initial_delay = heartbeat_interval,
                   .maximum_delay = heartbeat_interval},
       .summary_interval = std::chrono::seconds(30)});
  CHECK(failure_episode.valid());
  while (!exited_.load()) {
    std::unique_lock<std::mutex> lock(kv_event_wait_mutex_);
    if (kv_event_wait_cv_.wait_for(
            lock, heartbeat_interval, [this]() { return exited_.load(); })) {
      break;
    }
    lock.unlock();
    if (!register_done_.load()) continue;

    if (block_manager_pool_ == nullptr || scheduler_ == nullptr) continue;

    brpc::Controller cntl;
    xllm_service::proto::HeartbeatRequest req;
    req.set_name(instance_name_);

    req.mutable_load_metrics()->set_gpu_cache_usage_perc(
        block_manager_pool_->get_gpu_cache_usage_perc());

    req.mutable_load_metrics()->set_waiting_requests_num(
        scheduler_->get_waiting_requests_num());

    std::shared_ptr<NativeEngineStateBuilder> engine_state_builder;
    {
      std::lock_guard<std::mutex> lock(registration_mutex_);
      engine_state_builder = engine_state_builder_;
    }
    if (engine_state_builder != nullptr) {
      const std::vector<BlockCapacitySnapshot> capacity =
          block_manager_pool_->capacity_snapshots();
      NativeEngineStateSnapshot snapshot;
      snapshot.kv_free_blocks.reserve(capacity.size());
      snapshot.kv_used_blocks.reserve(capacity.size());
      snapshot.kv_total_blocks.reserve(capacity.size());
      for (const BlockCapacitySnapshot& value : capacity) {
        snapshot.kv_free_blocks.emplace_back(
            static_cast<uint64_t>(value.free_blocks));
        snapshot.kv_used_blocks.emplace_back(
            static_cast<uint64_t>(value.used_blocks));
        snapshot.kv_total_blocks.emplace_back(
            static_cast<uint64_t>(value.total_blocks));
      }
      // Idleness is decided by Service from waiting and running counts. The
      // heartbeat always publishes READY; DRAINING lives only in etcd.
      snapshot.lifecycle = proto::ENGINE_LIFECYCLE_READY;
      std::string error;
      const std::optional<proto::EngineState> state =
          engine_state_builder->build(snapshot, &error);
      if (state.has_value()) {
        *req.mutable_engine_state() = *state;
      } else {
        LOG(ERROR) << "Failed to build Native EngineState: " << error;
      }
    }

    std::vector<int64_t> ttft;
    std::vector<int64_t> tbt;
    scheduler_->get_latency_metrics(ttft, tbt);
    if (!ttft.empty()) {
      auto max_ttft = std::max_element(ttft.begin(), ttft.end());
      req.mutable_latency_metrics()->set_recent_max_ttft(*max_ttft);
    }

    if (!tbt.empty()) {
      auto max_tbt = std::max_element(tbt.begin(), tbt.end());
      req.mutable_latency_metrics()->set_recent_max_tbt(*max_tbt);
    }

    // XTensor page and weight-segment fields stay unset. Scheduling does not
    // consume them, and the registration payload must not invent values.

    xllm_service::proto::Status resp;
    std::string master_addr;
    const bool issued = with_master_stub(
        [&](xllm_service::proto::XllmRpcService_Stub* master_stub) {
          master_stub->Heartbeat(&cntl, &req, &resp, nullptr);
        },
        &master_addr,
        /*quiet=*/true);
    const bool success = issued && !cntl.Failed() && resp.ok();
    if (success) {
      const std::optional<FailureRecovery> recovery =
          failure_episode.record_success();
      if (recovery.has_value()) {
        COUNTER_INC(engine_heartbeat_recoveries_total);
        log_control_plane_recovery("HEARTBEAT", master_addr, *recovery);
      }
    } else {
      COUNTER_INC(engine_heartbeat_failures_total);
      const ControlPlaneFailure failure =
          control_plane_failure(issued, cntl, resp);
      const FailureObservation observation =
          failure_episode.record_failure(failure.reason);
      log_control_plane_failure("HEARTBEAT", master_addr, failure, observation);
      if (observation.action == FailureReportAction::kSuppressed) {
        COUNTER_INC(engine_heartbeat_suppressed_total);
      }
    }
  }
}

std::vector<std::string> XServiceClient::get_static_decode_list() {
  brpc::Controller cntl;
  xllm_service::proto::InstanceID req;
  xllm_service::proto::InstanceIDs resp;
  req.set_name(instance_name_);

  std::string master_addr;
  if (!with_master_stub(
          [&](xllm_service::proto::XllmRpcService_Stub* master_stub) {
            master_stub->GetStaticDecodeList(&cntl, &req, &resp, nullptr);
          },
          &master_addr)) {
    return {};
  }

  if (cntl.Failed()) {
    LOG(ERROR) << "Fail to get static decode list from master xservice server "
               << master_addr << ", error text: " << cntl.ErrorText();
    return {};
  }
  return std::vector<std::string>(resp.names().begin(), resp.names().end());
}

std::vector<std::string> XServiceClient::get_static_prefill_list() {
  brpc::Controller cntl;
  xllm_service::proto::InstanceID req;
  xllm_service::proto::InstanceIDs resp;
  req.set_name(instance_name_);

  std::string master_addr;
  if (!with_master_stub(
          [&](xllm_service::proto::XllmRpcService_Stub* master_stub) {
            master_stub->GetStaticPrefillList(&cntl, &req, &resp, nullptr);
          },
          &master_addr)) {
    return {};
  }

  if (cntl.Failed()) {
    LOG(ERROR) << "Fail to get static prefill list from master xservice server "
               << master_addr << ", error text: " << cntl.ErrorText();
    return {};
  }
  return std::vector<std::string>(resp.names().begin(), resp.names().end());
}

std::vector<std::string> XServiceClient::get_all_xservice_addrs() {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  std::vector<std::string> addrs;
  for (const auto& pair : xservice_stubs_) {
    addrs.push_back(pair.first);
  }
  return addrs;
}

std::vector<bool> XServiceClient::generations(
    const std::vector<RequestOutput>& outputs) {
  // true means the owning service accepted the output. A transport failure or
  // a rejected status is false so the baseline scheduler cancels the request.
  std::vector<bool> results(outputs.size(), false);

  // Keep each callback under Service's bounded Generations RPC contract.
  // Multiple chunks for the same Service are still fired before any join, so
  // a large scheduler step does not serialize transport latency.
  constexpr size_t kMaxGenerationsPerRpc = 256;
  struct PendingBatch {
    std::string service_addr;
    std::vector<size_t> output_indices;
    proto::DisaggStreamGenerations gens;
  };
  std::vector<PendingBatch> pending_batches;
  KeyedBoundedBatchPlanner<std::string> batch_planner(kMaxGenerationsPerRpc);

  auto mark_batch_failed = [&](const std::vector<size_t>& indices) {
    for (size_t idx : indices) {
      results[idx] = false;
    }
  };

  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto& output = outputs[i];
    if (output.target_xservice_addr.empty()) {
      LOG(ERROR) << "Reject service-routed output without its owning xservice "
                    "address, request_id="
                 << output.request_id;
      continue;
    }
    const std::string& target_service = output.target_xservice_addr;

    proto::DisaggStreamGeneration generation;
    if (!request_output_to_proto(output, instance_name_, &generation)) {
      LOG(ERROR) << "Failed to encode generation output for request_id: "
                 << output.request_id;
      continue;
    }
    const std::optional<size_t> batch_index =
        batch_planner.assign(target_service);
    if (!batch_index.has_value()) {
      LOG(ERROR) << "Generation batch planner rejected request_id: "
                 << output.request_id;
      continue;
    }
    if (*batch_index == pending_batches.size()) {
      pending_batches.emplace_back(
          PendingBatch{.service_addr = target_service});
    }
    PendingBatch& batch = pending_batches[*batch_index];
    batch.output_indices.push_back(i);
    *batch.gens.add_gens() = std::move(generation);
  }

  // Use brpc semi-synchronous RPC pattern (DoNothing + Join) instead of
  // folly thread pool. brpc::Join is bthread-aware: it yields the brpc worker
  // thread via butex_wait, avoiding the deadlock that occurs when
  // folly::SemiFuture::wait() blocks all brpc workers with a pthread futex.
  struct AsyncCallContext {
    std::string service_addr;
    brpc::Controller cntl;
    proto::StatusSet resp;
    proto::DisaggStreamGenerations gens;
    std::vector<size_t> output_indices;
    bool issued = false;
  };

  std::vector<AsyncCallContext> contexts(pending_batches.size());

  // Fire all RPCs asynchronously
  for (size_t index = 0; index < pending_batches.size(); ++index) {
    PendingBatch& batch = pending_batches[index];
    AsyncCallContext& ctx = contexts[index];
    ctx.service_addr = std::move(batch.service_addr);
    ctx.gens = std::move(batch.gens);
    ctx.output_indices = std::move(batch.output_indices);

    if (!connect_to_xservice(ctx.service_addr)) {
      LOG(ERROR) << "Failed to connect target xservice: " << ctx.service_addr;
      continue;
    }

    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto* service_stub = find_stub_locked(ctx.service_addr);
    if (service_stub == nullptr) {
      LOG(ERROR) << "No stub available for xservice: " << ctx.service_addr;
      continue;
    }
    ctx.issued = true;
    ctx.cntl.set_timeout_ms(
        ::xllm::ServiceConfig::get_instance().output_delivery_rpc_timeout_ms());
    service_stub->Generations(
        &ctx.cntl, &ctx.gens, &ctx.resp, brpc::DoNothing());
  }

  // Wait for all RPCs — bthread-aware, yields brpc worker properly
  for (auto& ctx : contexts) {
    if (ctx.issued) {
      brpc::Join(ctx.cntl.call_id());
    }
  }

  // Process results
  for (auto& ctx : contexts) {
    const std::string& service_addr = ctx.service_addr;
    const std::vector<size_t>& indices = ctx.output_indices;

    if (!ctx.issued || ctx.cntl.Failed()) {
      if (ctx.cntl.Failed()) {
        LOG(ERROR) << "Fail to response tokens to xservice server "
                   << service_addr << ", error text: " << ctx.cntl.ErrorText();
      }
      mark_batch_failed(indices);
      continue;
    }

    if (ctx.resp.all_status_size() != static_cast<int>(indices.size())) {
      LOG(ERROR) << "Generation delivery returned "
                 << ctx.resp.all_status_size() << " statuses for "
                 << indices.size() << " outputs, service=" << service_addr;
      continue;
    }

    for (size_t j = 0; j < indices.size(); ++j) {
      size_t original_idx = indices[j];
      const proto::Status& status = ctx.resp.all_status(j);
      results[original_idx] = status.ok();
      if (!status.ok()) {
        LOG(INFO) << "Output subscriber rejected request_id="
                  << outputs[original_idx].request_id << ", delivery_code="
                  << (status.has_delivery_code()
                          ? static_cast<int>(status.delivery_code())
                          : 0)
                  << ", message=" << status.status_msg();
      }
    }
  }

  if (level_one_trace_enabled()) {
    for (size_t index = 0; index < outputs.size(); ++index) {
      const RequestOutput& output = outputs[index];
      // Record first/terminal delivery boundaries and failures, not every
      // successful token. This joins Engine ownership to Service's ACK.
      if (!output.finished && !output.finished_on_prefill_instance &&
          results[index]) {
        continue;
      }
      const char* outcome = results[index] ? "DELIVERED" : "REJECTED";
      std::ostringstream diagnostic;
      diagnostic << "engine_output_delivery request_uid="
                 << output.service_request_id
                 << " engine_request_id=" << output.request_id
                 << " engine_uid=" << instance_name_
                 << " service=" << output.target_xservice_addr
                 << " finished_on_prefill="
                 << output.finished_on_prefill_instance
                 << " finished=" << output.finished << " outcome=" << outcome;
      log_level_one_trace(__FILE__, __LINE__, diagnostic.str());
    }
  }

  return results;
}

bool XServiceClient::connect_to_xservice(const std::string& xservice_addr,
                                         bool quiet) {
  if (!check_instance_name(xservice_addr)) {
    if (quiet) {
      VLOG(1) << "Invalid xservice address format: " << xservice_addr;
    } else {
      LOG(ERROR) << "Invalid xservice address format: " << xservice_addr;
    }
    return false;
  }

  std::unique_lock<std::shared_mutex> lock(mutex_);

  // If already connected, directly return true
  if (xservice_channels_.find(xservice_addr) != xservice_channels_.end()) {
    return true;
  }

  auto channel = std::make_unique<brpc::Channel>();
  if (channel->Init(xservice_addr.c_str(), "", &chan_options_) != 0) {
    if (quiet) {
      VLOG(1) << "Fail to initialize xservice channel to server "
              << xservice_addr;
    } else {
      LOG(ERROR) << "Fail to initialize xservice channel to server "
                 << xservice_addr;
    }
    return false;
  }

  xservice_channels_[xservice_addr] = std::move(channel);
  xservice_stubs_[xservice_addr] =
      std::make_unique<xllm_service::proto::XllmRpcService_Stub>(
          xservice_channels_[xservice_addr].get());

  LOG(INFO) << "Successfully connected to xservice: " << xservice_addr;
  return true;
}

bool XServiceClient::with_master_stub(
    const std::function<void(xllm_service::proto::XllmRpcService_Stub*)>& fn,
    std::string* master_addr,
    bool quiet) {
  if (master_addr == nullptr) {
    return false;
  }

  // wrapper in a whole lambda function
  auto run_with_current_master_stub = [&](bool* has_master_addr) -> bool {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    *master_addr = master_xservice_addr_;
    *has_master_addr = !master_addr->empty();
    if (!*has_master_addr) {
      if (quiet) {
        VLOG(1) << "Master xservice address is empty";
      } else {
        LOG(ERROR) << "Master xservice address is empty";
      }
      return false;
    }

    auto* master_stub = find_stub_locked(*master_addr);
    if (master_stub == nullptr) {
      return false;
    }

    fn(master_stub);
    return true;
  };

  bool has_master_addr = false;
  if (run_with_current_master_stub(&has_master_addr)) {
    return true;
  }
  if (!has_master_addr) {
    return false;
  }

  // try re-connecting once
  if (!connect_to_xservice(*master_addr, quiet)) {
    if (quiet) {
      VLOG(1) << "Failed to connect to master xservice: " << *master_addr;
    } else {
      LOG(ERROR) << "Failed to connect to master xservice: " << *master_addr;
    }
    return false;
  }

  if (run_with_current_master_stub(&has_master_addr)) {
    return true;
  }
  if (!has_master_addr) {
    return false;
  }

  if (quiet) {
    VLOG(1) << "No master stub available for address: " << *master_addr;
  } else {
    LOG(ERROR) << "No master stub available for address: " << *master_addr;
  }
  return false;
}

xllm_service::proto::XllmRpcService_Stub* XServiceClient::find_stub_locked(
    const std::string& xservice_addr) {
  auto it = xservice_stubs_.find(xservice_addr);
  if (it == xservice_stubs_.end() || it->second == nullptr) {
    return nullptr;
  }
  return it->second.get();
}

void XServiceClient::disconnect_xservice(const std::string& xservice_addr) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  if (xservice_stubs_.erase(xservice_addr) > 0) {
    xservice_channels_.erase(xservice_addr);
    LOG(INFO) << "Disconnected from xservice: " << xservice_addr;

    // if master disconnected，need to update master address
    if (xservice_addr == master_xservice_addr_) {
      LOG(WARNING) << "Master xservice disconnected: " << master_xservice_addr_;
    }
  }
}

void XServiceClient::handle_master_service_watch(const etcd::Response& response,
                                                 const uint64_t& prefix_len) {
  if (response.events().empty() || exited_.load()) {
    return;
  }

  for (const auto& event : response.events()) {
    if (event.event_type() == etcd::Event::EventType::PUT) {
      auto new_master_addr = event.kv().as_string();

      {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (master_xservice_addr_.compare(new_master_addr) == 0) {
          continue;
        }

        LOG(INFO) << "Master service changed from " << master_xservice_addr_
                  << " to " << new_master_addr;

        master_xservice_addr_ = new_master_addr;
      }

      if (!connect_to_xservice(new_master_addr)) {
        LOG(ERROR) << "Failed to connect to new master: " << new_master_addr;
      }
    } else if (event.event_type() == etcd::Event::EventType::DELETE_) {
      std::unique_lock<std::shared_mutex> lock(mutex_);
      if (!master_xservice_addr_.empty()) {
        LOG(WARNING) << "Master service key deleted, clear cached master addr: "
                     << master_xservice_addr_;
        master_xservice_addr_.clear();
      }
    }
  }
}

void XServiceClient::handle_xservices_watch(const etcd::Response& response,
                                            const uint64_t& prefix_len) {
  if (response.events().empty() || exited_.load()) {
    return;
  }

  for (const auto& event : response.events()) {
    std::string event_key;
    std::string service_addr;
    if (event.event_type() == etcd::Event::EventType::PUT) {
      if (event.has_kv()) {
        event_key = event.kv().key().substr(prefix_len);
        service_addr = event.kv().as_string();
      }
    } else if (event.event_type() == etcd::Event::EventType::DELETE_) {
      if (event.has_prev_kv()) {
        event_key = event.prev_kv().key().substr(prefix_len);
        service_addr = event.prev_kv().as_string();
      }
      if (service_addr.empty() && event.has_kv()) {
        if (event_key.empty()) {
          event_key = event.kv().key().substr(prefix_len);
        }
        service_addr = event.kv().as_string();
      }
    }

    if (xservices_key_prefix_ == etcd_schema::kDefaultRouterMemberPrefix &&
        (event_key == etcd_schema::kDefaultMasterServiceName ||
         event_key == etcd_schema::kDefaultMasterServiceKey)) {
      continue;
    }

    if (service_addr.empty() && !event_key.empty() &&
        event_key.rfind(xservices_key_prefix_, 0) == 0) {
      service_addr = event_key.substr(xservices_key_prefix_.size());
    }

    if (service_addr.empty()) {
      continue;
    }

    if (!check_instance_name(service_addr)) {
      continue;
    }

    if (event.event_type() == etcd::Event::EventType::PUT) {
      std::string master_xservice_addr;
      {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        master_xservice_addr = master_xservice_addr_;
      }

      if (service_addr != master_xservice_addr) {
        connect_to_xservice(service_addr);
      }
    } else if (event.event_type() == etcd::Event::EventType::DELETE_) {
      disconnect_xservice(service_addr);
    }
  }
}

std::optional<WriterTerminationProof>
XServiceClient::get_writer_termination_proof(const std::string& engine_uid) {
  std::string wire;
  if (etcd_client_ == nullptr ||
      !etcd_client_->get(writer_termination_key(engine_uid),
                         &wire,
                         /*quiet_missing=*/true)) {
    return std::nullopt;
  }
  auto proof = parse_writer_termination(wire);
  if (!proof || proof->engine_uid != engine_uid) {
    return std::nullopt;
  }
  return proof;
}

}  // namespace xllm
