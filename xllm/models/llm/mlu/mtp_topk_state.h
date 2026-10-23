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

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "core/framework/model/mtp_topk_state.h"
#include "core/layers/mlu/deepseek_v32/dsa_topk_state.h"

namespace xllm::mlu::model {

// Model-level MLU adapter for sparse-attention state emitted by each MTP
// decoder layer. Runtime only sees the common state and graph contracts.
class MluMtpTopkState final : public ::xllm::MtpTopkGraphState {
 public:
  using LayerStates = std::vector<std::optional<layer::DsaTopkState>>;

  explicit MluMtpTopkState(LayerStates layer_states)
      : layer_states_(std::move(layer_states)) {
    CHECK(!layer_states_.empty())
        << "MLU MTP top-k state must contain decoder-layer entries.";
    bool found_state = false;
    for (std::optional<layer::DsaTopkState>& state : layer_states_) {
      if (!state.has_value()) {
        continue;
      }
      state = state->flattened();
      if (!found_state) {
        num_rows_ = state->block_tables().size(0);
        device_ = state->block_tables().device();
        found_state = true;
        continue;
      }
      CHECK_EQ(state->block_tables().size(0), num_rows_)
          << "MLU per-layer MTP top-k states must have equal row counts.";
      CHECK(state->block_tables().device() == device_)
          << "MLU per-layer MTP top-k states must use the same device.";
    }
    CHECK(found_state)
        << "MLU MTP top-k state must contain at least one populated layer.";
  }

  const LayerStates& layer_states() const { return layer_states_; }

  // Rebuilds every populated layer through `fn` and keeps absent layers absent.
  template <typename Fn>
  static LayerStates map_layers(const LayerStates& layer_states, Fn&& fn) {
    LayerStates mapped;
    mapped.reserve(layer_states.size());
    for (const std::optional<layer::DsaTopkState>& state : layer_states) {
      mapped.emplace_back(state.has_value()
                              ? std::optional<layer::DsaTopkState>(fn(*state))
                              : std::nullopt);
    }
    return mapped;
  }

  // The graph key includes absent layers and column counts; a populated layer
  // always has at least one column. Row counts vary within a bucket, while all
  // non-row dimensions must remain fixed.
  std::vector<int64_t> graph_schema() const override {
    std::vector<int64_t> schema;
    schema.reserve(layer_states_.size());
    for (const auto& state : layer_states_) {
      schema.emplace_back(state.has_value() ? state->block_tables().size(1)
                                            : 0);
    }
    return schema;
  }

  std::unique_ptr<MtpTopkGraphBuffer> make_graph_buffer(
      int64_t rows) const override;

  int64_t num_rows() const override { return num_rows_; }

  torch::Device device() const override { return device_; }

  MtpTopkStatePtr to(const torch::Device& device) const override {
    return std::make_shared<MluMtpTopkState>(
        map_layers(layer_states_, [&device](const layer::DsaTopkState& state) {
          return state.to(device);
        }));
  }

  MtpTopkStatePtr index_select_rows(const torch::Tensor& index) const override {
    return std::make_shared<MluMtpTopkState>(
        map_layers(layer_states_, [&index](const layer::DsaTopkState& state) {
          return layer::DsaTopkState(
              state.block_tables().index_select(/*dim=*/0, index),
              state.context_lens().index_select(/*dim=*/0, index));
        }));
  }

 private:
  LayerStates layer_states_;
  int64_t num_rows_ = 0;
  torch::Device device_{torch::kCPU};
};

// Keep native tensor layout knowledge on the model side of the graph contract.
class MluMtpTopkGraphBuffer final : public MtpTopkGraphBuffer {
 public:
  MluMtpTopkGraphBuffer(const MluMtpTopkState& source, int64_t rows) {
    CHECK_GE(rows, source.num_rows());
    state_ = std::make_shared<MluMtpTopkState>(MluMtpTopkState::map_layers(
        source.layer_states(), [rows](const layer::DsaTopkState& state) {
          return layer::DsaTopkState(
              torch::empty({rows, state.block_tables().size(1)},
                           state.block_tables().options()),
              torch::empty({rows}, state.context_lens().options()));
        }));
  }

  MluMtpTopkGraphBuffer(const MluMtpTopkGraphBuffer&) = delete;
  MluMtpTopkGraphBuffer& operator=(const MluMtpTopkGraphBuffer&) = delete;

  MtpTopkStatePtr state() const override { return state_; }

  // Only this graph-owned buffer can update the shared read-only view. Validate
  // the complete layout before changing any tensor, and never replace storage.
  void update(const MtpTopkState& source) override {
    const auto* native = dynamic_cast<const MluMtpTopkState*>(&source);
    CHECK(native != nullptr)
        << "MLU graph received incompatible MTP top-k state";
    CHECK(state_->graph_schema() == native->graph_schema())
        << "MLU MTP graph state schema changed";
    CHECK(state_->device() == native->device())
        << "MLU MTP graph state device changed";
    CHECK_GE(state_->num_rows(), native->num_rows())
        << "MLU MTP graph state exceeds row capacity";
    // Copies stay on one device and use the current stream. The executor
    // orders capture with events and launches replay on this same stream.
    const int64_t rows = native->num_rows();
    const int64_t padding = state_->num_rows() - rows;
    for (std::size_t i = 0; i < state_->layer_states().size(); ++i) {
      if (!state_->layer_states()[i].has_value()) {
        continue;
      }
      const auto& target = *state_->layer_states()[i];
      const auto& input = *native->layer_states()[i];
      target.block_tables()
          .narrow(/*dim=*/0, /*start=*/0, rows)
          .copy_(input.block_tables(), /*non_blocking=*/true);
      target.context_lens()
          .narrow(/*dim=*/0, /*start=*/0, rows)
          .copy_(input.context_lens(), /*non_blocking=*/true);
      // Zero-length padded rows cannot read the sentinel block-table entries.
      target.block_tables()
          .narrow(/*dim=*/0, rows, padding)
          .fill_(/*value=*/-1);
      target.context_lens().narrow(/*dim=*/0, rows, padding).zero_();
    }
  }

  MtpTopkStatePtr snapshot(int64_t rows) const override {
    CHECK_LE(rows, state_->num_rows())
        << "MLU MTP graph snapshot exceeds row capacity";
    return std::make_shared<MluMtpTopkState>(MluMtpTopkState::map_layers(
        state_->layer_states(), [rows](const layer::DsaTopkState& state) {
          return layer::DsaTopkState(
              state.block_tables().narrow(/*dim=*/0, /*start=*/0, rows).clone(),
              state.context_lens()
                  .narrow(/*dim=*/0, /*start=*/0, rows)
                  .clone());
        }));
  }

  std::size_t tensor_bytes() const override {
    std::size_t total = 0;
    for (const auto& layer : state_->layer_states()) {
      if (!layer.has_value()) {
        continue;
      }
      total +=
          layer->block_tables().numel() * layer->block_tables().element_size();
      total +=
          layer->context_lens().numel() * layer->context_lens().element_size();
    }
    return total;
  }

 private:
  std::shared_ptr<MluMtpTopkState> state_;
};

inline std::unique_ptr<MtpTopkGraphBuffer> MluMtpTopkState::make_graph_buffer(
    int64_t rows) const {
  return std::make_unique<MluMtpTopkGraphBuffer>(*this, rows);
}

}  // namespace xllm::mlu::model
