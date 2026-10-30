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

#pragma once

#include <brpc/channel.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/etcd_client.h"
#include "core/runtime/native_provider_contract.h"
#include "distributed_runtime/engine.h"
#include "distributed_runtime/writer_termination_proof.h"
#include "forward_params.h"
#include "framework/block/block_manager_pool.h"
#include "framework/prefix_cache/prefix_cache_journal.h"
#include "framework/request/request_output.h"
#include "scheduler/scheduler.h"
#include "xservice.pb.h"
namespace xllm {

class XServiceClient {
 public:
  static XServiceClient* get_instance() {
    static XServiceClient xservice_client;
    return &xservice_client;
  }

  ~XServiceClient();
  bool init(const std::string& etcd_addr,
            const std::string& instance_name = "",
            BlockManagerPool* block_manager_pool = nullptr,
            const std::string& etcd_namespace = "");
  void shutdown();
  void set_scheduler(Scheduler* scheduler);
  void set_engine(Engine* engine);
  bool initialize_done() { return initialize_done_; }

  std::string get_instance_name();
  std::optional<proto::ProviderDescriptor> get_provider_descriptor();
  void register_instance(const InstanceInfo& instance_info);
  void heartbeat();
  InstanceInfo get_instance_info(const std::string& instance_name);
  std::vector<std::string> get_static_decode_list();
  std::vector<std::string> get_static_prefill_list();

  // get all xllm_service addrs
  std::vector<std::string> get_all_xservice_addrs();
  // Untrusted durable transport only. The attempt table verifies the pinned
  // authority signature before any ownership transition.
  std::optional<WriterTerminationProof> get_writer_termination_proof(
      const std::string& engine_uid);

  // response generation tokens to xllm service
  std::vector<bool> generations(const std::vector<RequestOutput>& outputs);
  proto::KVCacheSnapshotPage get_kv_cache_snapshot(
      const proto::KVCacheSnapshotRequest& request);

 private:
  bool register_instance_with_retry(const std::string& key,
                                    const std::string& value);
  bool reconcile_registration();
  void reconcile_registration_loop();
  void kv_event_loop();

  void handle_master_service_watch(const etcd::Response& response,
                                   const uint64_t& prefix_len);
  void handle_xservices_watch(const etcd::Response& response,
                              const uint64_t& prefix_len);

  // connect to specific xllm_service
  bool connect_to_xservice(const std::string& xservice_addr,
                           bool quiet = false);

  // remove the connection to specific xllm_service
  void disconnect_xservice(const std::string& xservice_addr);

  proto::KVStreamIdentity kv_stream_identity() const;
  proto::KVEventBatch make_kv_event_batch(const PrefixCacheEvents& events);
  proto::KVCacheSnapshotPage prefix_cache_snapshot_to_proto(
      const PrefixCacheSnapshot& snapshot) const;
  void note_kv_push_result(bool success,
                           uint64_t commit_version,
                           uint64_t service_commit_version,
                           const std::string& error);

  // call rpc with current master stub atomically.
  bool with_master_stub(
      const std::function<void(xllm_service::proto::XllmRpcService_Stub*)>& fn,
      std::string* master_addr,
      bool quiet = false);

  // find stub by address, caller should hold mutex_
  xllm_service::proto::XllmRpcService_Stub* find_stub_locked(
      const std::string& xservice_addr);

 private:
  XServiceClient() = default;

  std::atomic_bool exited_{false};
  std::atomic_bool register_done_{false};
  std::once_flag shutdown_once_;
  bool initialize_done_ = false;
  std::string instance_name_;
  std::string registration_key_;

  std::string master_xservice_addr_;
  std::string master_service_key_ = "XLLM:SERVICE:MASTER";
  std::string xservices_key_prefix_ = "XLLM:SERVICE:";
  std::unordered_map<std::string, std::unique_ptr<brpc::Channel>>
      xservice_channels_;
  std::unordered_map<std::string,
                     std::unique_ptr<xllm_service::proto::XllmRpcService_Stub>>
      xservice_stubs_;
  std::unique_ptr<std::thread> heartbeat_thread_;
  std::unique_ptr<std::thread> reconcile_thread_;
  std::unique_ptr<std::thread> kv_event_thread_;

  std::shared_mutex mutex_;
  mutable std::mutex registration_mutex_;
  std::mutex kv_event_wait_mutex_;
  std::condition_variable kv_event_wait_cv_;
  // Only the previous push is remembered. A failed push replaces this
  // record. A successful push clears it. The journal only supplies events
  // and the resident version.
  std::optional<uint64_t> kv_failed_commit_version_;
  std::string kv_failed_commit_error_;
  uint64_t kv_last_pushed_version_ = 0;
  brpc::ChannelOptions chan_options_;
  std::unique_ptr<EtcdClient> etcd_client_;
  BlockManagerPool* block_manager_pool_ = nullptr;  // not own
  Scheduler* scheduler_ = nullptr;                  // not own
  Engine* engine_ = nullptr;                        // not own, for xtensor info
  std::shared_ptr<NativeEngineStateBuilder> engine_state_builder_;
  std::optional<proto::ProviderDescriptor> provider_descriptor_;
};

// Builds the in-process instance record from a GetInstanceInfo response.
//
// `kv_split_size` is copied verbatim so that proto3's 0 -- "the field was not
// present" -- stays distinguishable from a declared 1. Collapsing the two is a
// silent mis-addressing risk: the transfer plan keys on the peer's DECLARED
// width, and a peer that never declared one must be refused, not assumed to be
// 1. See plan_kv_split_widths(). `cp_index_write_mode` follows the same
// verbatim contract for the write mode.
InstanceInfo instance_info_from_proto(
    const xllm_service::proto::InstanceMetaInfo& response);

}  // namespace xllm
