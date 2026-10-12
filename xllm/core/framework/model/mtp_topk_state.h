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

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace xllm {

class MtpTopkState;
using MtpTopkStatePtr = std::shared_ptr<const MtpTopkState>;

// Backend-neutral state carried between MTP draft steps. Runtime code owns
// row-selection orchestration; backend implementations only expose the common
// row axis and interpret index_select_rows() for their native payload.
class MtpTopkState {
 public:
  virtual ~MtpTopkState() = default;

  virtual int64_t num_rows() const = 0;
  virtual torch::Device device() const = 0;
  virtual MtpTopkStatePtr to(const torch::Device& device) const = 0;
  virtual MtpTopkStatePtr index_select_rows(
      const torch::Tensor& index) const = 0;
};

// A graph owns the writable storage; model inputs only receive a read-only
// state view. Updates preserve addresses and clear unused rows.
class MtpTopkGraphBuffer {
 public:
  virtual ~MtpTopkGraphBuffer() = default;
  virtual MtpTopkStatePtr state() const = 0;
  virtual void update(const MtpTopkState& source) = 0;
  // Copies the leading rows into storage that later updates cannot reach.
  virtual MtpTopkStatePtr snapshot(int64_t rows) const = 0;
  virtual std::size_t tensor_bytes() const = 0;
};

// Optional graph contract. Eager-only backends continue to implement just
// MtpTopkState. The schema is an opaque, deterministic sequence of layout
// identifiers: equal schemas must support the same captured operations at a
// fixed row capacity. Runtime compares it without interpreting payload fields.
// The empty schema is reserved for steps without shared state, so a state that
// can be captured must return a non-empty one. A step that receives a state
// must return that selection unchanged: graphs hand the caller's input back
// instead of buffering a second copy.
class MtpTopkGraphState : public MtpTopkState {
 public:
  virtual std::vector<int64_t> graph_schema() const = 0;
  virtual std::unique_ptr<MtpTopkGraphBuffer> make_graph_buffer(
      int64_t rows) const = 0;
};

}  // namespace xllm
