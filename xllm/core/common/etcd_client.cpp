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

#include "etcd_client.h"

#include <glog/logging.h>

#include <etcd/Client.hpp>
#include <etcd/Response.hpp>
#include <etcd/v3/Transaction.hpp>
#include <exception>
#include <nlohmann/json.hpp>

#include "xllm/core/common/etcd_utils.h"

namespace xllm {

EtcdClient::EtcdClient(const std::string& etcd_addr,
                       const std::string& etcd_namespace)
    : client_(etcd_addr),
      etcd_addr_(etcd_addr),
      etcd_namespace_prefix_(normalize_etcd_namespace(etcd_namespace)) {
  client_.set_grpc_timeout(std::chrono::seconds(1));
  auto response = client_.put(namespaced_key("XLLM_PING"), "PING");
  if (!response.is_ok()) {
    LOG(FATAL) << "etcd connect to etcd server failed: "
               << response.error_message();
  }
}

EtcdClient::EtcdClient(const std::string& etcd_addr,
                       const std::string& username,
                       const std::string& password,
                       const std::string& etcd_namespace)
    : client_(etcd_addr, username, password),
      etcd_addr_(etcd_addr),
      etcd_namespace_prefix_(normalize_etcd_namespace(etcd_namespace)) {
  client_.set_grpc_timeout(std::chrono::seconds(1));
  auto response = client_.put(namespaced_key("XLLM_PING"), "PING");
  if (!response.is_ok()) {
    LOG(FATAL) << "etcd connect to etcd server failed: "
               << response.error_message();
  }
}

std::string EtcdClient::namespaced_key(const std::string& logical_key) const {
  if (etcd_namespace_prefix_.empty()) {
    return logical_key;
  }
  return etcd_namespace_prefix_ + logical_key;
}

EtcdClient::~EtcdClient() {
  {
    std::lock_guard<std::mutex> lock(keep_alives_mutex_);
    stopping_ = true;
    ownership_cv_.notify_all();
  }
  if (ownership_thread_.joinable()) {
    ownership_thread_.join();
  }
  stop_watch();
  for (auto& [key, registration] : keep_alives_) {
    registration.keep_alive->Cancel();
  }
}

void EtcdClient::monitor_registration_ownership() {
  std::unique_lock<std::mutex> lock(keep_alives_mutex_);
  while (!stopping_) {
    const auto now = RegistrationLeaseDeadline::Clock::now();
    for (const auto& [key, registration] : keep_alives_) {
      CHECK(!registration.ownership.expired(now))
          << "registration lifecycle=SELF_FENCE reason=OWNERSHIP_DEADLINE "
          << "key=" << key << " lease_id=" << registration.keep_alive->Lease();
    }
    ownership_cv_.wait_for(
        lock, std::chrono::milliseconds(100), [this]() { return stopping_; });
  }
}

void EtcdClient::add_watch(const std::string& key_prefix,
                           Callback callback,
                           bool recursive) {
  std::lock_guard<std::mutex> lock(watchers_mutex_);

  if (watchers_.find(key_prefix) != watchers_.end()) {
    watchers_[key_prefix].watcher->Cancel();
  }

  uint64_t prefix_len = etcd_namespace_prefix_.size();
  auto watcher = std::make_unique<etcd::Watcher>(
      client_,
      namespaced_key(key_prefix),
      [callback, prefix_len](const etcd::Response& response) {
        callback(response, prefix_len);
      },
      recursive);

  watchers_[key_prefix] = {std::move(watcher), callback};
}

void EtcdClient::remove_watch(const std::string& key_prefix) {
  std::lock_guard<std::mutex> lock(watchers_mutex_);

  auto it = watchers_.find(key_prefix);
  if (it != watchers_.end()) {
    it->second.watcher->Cancel();
    watchers_.erase(it);
  }
}

void EtcdClient::stop_watch() {
  std::lock_guard<std::mutex> lock(watchers_mutex_);

  for (auto& pair : watchers_) {
    pair.second.watcher->Cancel();
  }

  watchers_.clear();
}

bool EtcdClient::get_master_service(const std::string& key,
                                    std::string* values) {
  auto response = client_.get(namespaced_key(key));
  if (!response.is_ok()) {
    LOG(ERROR) << "etcd get " << key << " failed: " << response.error_message();
    return false;
  }
  *values = response.value().as_string();
  return true;
}

bool EtcdClient::get(const std::string& key,
                     std::string* value,
                     bool quiet_missing) {
  auto response = client_.get(namespaced_key(key));
  if (!response.is_ok()) {
    if (quiet_missing && response.error_code() == 100) {
      return false;
    }
    LOG(ERROR) << "etcd get " << key << " failed: " << response.error_message();
    return false;
  }

  if (value != nullptr) {
    *value = response.value().as_string();
  }
  return true;
}

RegistrationLookupState EtcdClient::lookup_registration(
    const std::string& key) {
  int64_t lease_id = 0;
  int32_t ttl = 0;
  std::shared_ptr<etcd::KeepAlive> current_keep_alive;
  std::shared_ptr<std::atomic<bool>> keep_alive_failed;
  std::string expected_value;
  {
    std::lock_guard<std::mutex> lock(keep_alives_mutex_);
    const auto it = keep_alives_.find(key);
    if (it == keep_alives_.end()) {
      return RegistrationLookupState::kMissing;
    }
    current_keep_alive = it->second.keep_alive;
    keep_alive_failed = it->second.keep_alive_failed;
    ttl = it->second.ttl;
    lease_id = current_keep_alive->Lease();
    expected_value = it->second.value;
  }
  const auto response = client_.get(namespaced_key(key));
  if (!response.is_ok()) {
    if (response.error_code() == etcdv3::ERROR_KEY_NOT_FOUND) {
      return RegistrationLookupState::kMissing;
    }
    LOG(ERROR) << "etcd membership lookup unavailable, key=" << key
               << ", error=" << response.error_message();
    return RegistrationLookupState::kUnavailable;
  }
  if (response.value().lease() != lease_id ||
      response.value().as_string() != expected_value) {
    return RegistrationLookupState::kMissing;
  }
  const auto ttl_started = RegistrationLeaseDeadline::Clock::now();
  const auto ttl_response = client_.leasetimetolive(lease_id);
  if (!ttl_response.is_ok()) {
    return RegistrationLookupState::kUnavailable;
  }
  {
    std::lock_guard<std::mutex> lock(keep_alives_mutex_);
    const auto it = keep_alives_.find(key);
    if (it == keep_alives_.end() ||
        it->second.keep_alive != current_keep_alive ||
        !it->second.ownership.refresh(
            ttl_started,
            std::chrono::seconds(ttl_response.value().ttl()),
            RegistrationLeaseDeadline::Clock::now())) {
      return RegistrationLookupState::kMissing;
    }
  }
  // A failed KeepAlive stream never restarts itself. Reconnect only the exact
  // still-owned lease; do not create a new lease/incarnation or revive an
  // expired owner. This is a control-plane operation, not a model retry.
  if (keep_alive_failed->load(std::memory_order_acquire)) {
    auto replacement_failed = std::make_shared<std::atomic<bool>>(false);
    auto replacement = make_keep_alive(key, ttl, lease_id, replacement_failed);
    // The background stream waits ttl / 3 before its first refresh. Renew
    // immediately: a recovered connection can have less than that TTL left.
    try {
      replacement->Check();
    } catch (const std::exception& error) {
      LOG(ERROR) << "etcd lease reconnect failed, key=" << key
                 << ", lease_id=" << lease_id << ", error=" << error.what();
      return RegistrationLookupState::kUnavailable;
    }
    {
      std::lock_guard<std::mutex> lock(keep_alives_mutex_);
      const auto it = keep_alives_.find(key);
      if (it == keep_alives_.end() ||
          it->second.keep_alive != current_keep_alive ||
          it->second.ownership.expired(
              RegistrationLeaseDeadline::Clock::now())) {
        return RegistrationLookupState::kMissing;
      }
      it->second.keep_alive = std::move(replacement);
      it->second.keep_alive_failed = std::move(replacement_failed);
    }
    current_keep_alive->Cancel();
    LOG(INFO) << "registration lifecycle=RENEWING key=" << key
              << " lease_id=" << lease_id;
  }
  return RegistrationLookupState::kPresent;
}

bool EtcdClient::get_all_xservices(const std::string& key_prefix,
                                   std::vector<std::string>* values) {
  auto response = client_.ls(namespaced_key(key_prefix));
  if (!response.is_ok()) {
    LOG(ERROR) << "etcd get " << key_prefix
               << " failed: " << response.error_message();
    return false;
  }

  if (values == nullptr) {
    return true;
  }

  values->clear();
  values->reserve(response.keys().size());
  for (int i = 0; i < response.keys().size(); ++i) {
    values->emplace_back(response.value(i).as_string());
  }

  return true;
}

std::shared_ptr<etcd::KeepAlive> EtcdClient::make_keep_alive(
    const std::string& key,
    int32_t ttl,
    int64_t lease_id,
    const std::shared_ptr<std::atomic<bool>>& failed) {
  const auto keep_alive_failure = [key, ttl, failed](std::exception_ptr error) {
    failed->store(true, std::memory_order_release);
    try {
      if (error != nullptr) {
        std::rethrow_exception(error);
      }
      LOG(ERROR) << "etcd keepalive stopped without an exception, key=" << key
                 << ", ttl_seconds=" << ttl;
    } catch (const std::exception& exception) {
      LOG(ERROR) << "etcd keepalive failed, key=" << key
                 << ", ttl_seconds=" << ttl << ", error=" << exception.what();
    } catch (...) {
      LOG(ERROR) << "etcd keepalive failed with an unknown exception, key="
                 << key << ", ttl_seconds=" << ttl;
    }
  };
  return std::make_shared<etcd::KeepAlive>(
      client_, keep_alive_failure, ttl, lease_id);
}

bool EtcdClient::register_instance(const std::string& key,
                                   const std::string& value,
                                   const int ttl) {
  const auto registration_started = RegistrationLeaseDeadline::Clock::now();
  auto failed = std::make_shared<std::atomic<bool>>(false);
  auto keep_alive = make_keep_alive(key, ttl, /*lease_id=*/0, failed);
  if (keep_alive->Lease() <= 0 || failed->load(std::memory_order_acquire)) {
    LOG(ERROR) << "etcd lease grant failed, key=" << key;
    keep_alive->Cancel();
    return false;
  }
  etcdv3::Transaction transaction;
  transaction.add_compare_create(namespaced_key(key), 0);
  transaction.add_success_put(namespaced_key(key), value, keep_alive->Lease());
  auto response = client_.txn(transaction);
  if (!response.is_ok()) {
    LOG(ERROR) << "etcd create " << key
               << " failed: " << response.error_message();
    keep_alive->Cancel();
    return false;
  }

  std::shared_ptr<etcd::KeepAlive> old_keep_alive;
  {
    std::lock_guard<std::mutex> lock(keep_alives_mutex_);
    auto it = keep_alives_.find(key);
    if (it != keep_alives_.end()) {
      old_keep_alive = it->second.keep_alive;
    }
    keep_alives_.insert_or_assign(
        key,
        Registration{keep_alive,
                     failed,
                     ttl,
                     value,
                     RegistrationLeaseDeadline(registration_started,
                                               std::chrono::seconds(ttl))});
    if (!ownership_thread_.joinable()) {
      ownership_thread_ =
          std::thread([this]() { monitor_registration_ownership(); });
    }
  }
  if (old_keep_alive != nullptr) {
    old_keep_alive->Cancel();
  }

  return true;
}

}  // namespace xllm
