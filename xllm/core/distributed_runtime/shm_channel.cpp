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

#include "shm_channel.h"

#include "common/global_flags.h"
#include "core/common/health_check_manager.h"
#include "core/framework/config/distributed_config.h"
#include "core/framework/config/execution_config.h"
#include "util/net.h"

namespace xllm {

ShmChannel::ShmChannel(int dp_group,
                       int rank,
                       bool is_driver,
                       const runtime::Options& options)
    : enable_shm_(options.enable_shm()), is_driver_(is_driver) {
  bool is_creator;

  // Options carries the address for this channel. The process-wide config is
  // only the fallback because several static libraries each contain a copy.
  const std::string configured_addr = options.master_node_addr().value_or(
      DistributedConfig::get_instance().master_node_addr());
  std::string name_prefix = "xllm_" + net::extract_port(configured_addr);
  if (enable_shm_) {
    auto name = ForwardSharedMemoryManager::create_unique_name(
        name_prefix, dp_group, ForwardType::RAW_INPUT, rank);
    input_shm_manager_ = std::make_unique<ForwardSharedMemoryManager>(
        name, options.input_shm_size(), is_creator, ForwardType::RAW_INPUT);
    LOG(INFO) << "Create input shared memory manager with name: " << name
              << " and size: " << options.input_shm_size();
  }

  auto name = ForwardSharedMemoryManager::create_unique_name(
      name_prefix, dp_group, ForwardType::RAW_OUTPUT, rank);
  output_shm_manager_ = std::make_unique<ForwardSharedMemoryManager>(
      name, options.output_shm_size(), is_creator, ForwardType::RAW_OUTPUT);
  LOG(INFO) << "Create output shared memory manager with name: " << name
            << " and size: " << options.output_shm_size();
}

bool ShmChannel::execute_model_with_shm(const ForwardInput& input,
                                        RawForwardOutput& raw_output) {
  if (is_driver_) {
    bool use_shm_ret = input_shm_manager_->input_write(input);
    if (!use_shm_ret) {
      input_shm_manager_->cancel();
      HealthCheckManager::instance().set_unhealthy(
          "Worker SHM input write failed");
      LOG(ERROR)
          << "Worker SHM input write failed; cancelling the worker group";
      return false;
    }
  }
  const bool success = output_shm_manager_->raw_output_read(
      raw_output,
      std::chrono::milliseconds(
          ExecutionConfig::get_instance().forward_execution_timeout_ms()),
      input_shm_manager_.get());
  if (!success) {
    input_shm_manager_->cancel();
    HealthCheckManager::instance().set_unhealthy("Worker SHM execution failed");
  }
  return success;
}

void ShmChannel::execute_model_async(
    const ForwardInput& input,
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  if (enable_shm_) {
    RawForwardOutput raw_output;
    bool shm_success = execute_model_with_shm(input, raw_output);
    if (shm_success) {
      promise.setValue(raw_output);
    } else {
      promise.setValue(std::nullopt);
    }
    return;
  }
  execute_model_with_brpc(input, promise);
}
}  // namespace xllm
