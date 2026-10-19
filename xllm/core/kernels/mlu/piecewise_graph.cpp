/* Copyright 2026 The xLLM Authors.

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

#include "core/kernels/mlu/piecewise_graph.h"

#include <glog/logging.h>

#include <algorithm>
#include <utility>

namespace xllm::kernel::mlu {

PiecewiseGraph::PiecewiseGraph(const int32_t& kv_max_seq_len)
    : kv_max_seq_len_(kv_max_seq_len) {}

void PiecewiseGraph::begin_segment() {
  current_graph_ = std::make_unique<torch_mlu::MLUGraph>();
  current_graph_->capture_begin(pool_, cnrtQueueCaptureModeRelaxed);
}

void PiecewiseGraph::end_segment() {
  current_graph_->capture_end();
  segments_.emplace_back(Segment{std::move(current_graph_), {}});
}

void PiecewiseGraph::capture_begin(const torch_mlu::MempoolId_t& pool) {
  CHECK(!capturing_ && segments_.empty());
  pool_ = pool;
  capturing_ = true;
  begin_segment();
}

void PiecewiseGraph::capture_end() {
  CHECK(capturing_);
  end_segment();
  capturing_ = false;
}

void PiecewiseGraph::run_eager(std::function<void(int64_t)> function) {
  if (!capturing_) {
    function(kv_max_seq_len_);
    return;
  }
  end_segment();
  // Capture does not execute the preceding graph. Record the callback without
  // reading its not-yet-computed inputs or advancing request state a second
  // time.
  segments_.emplace_back(Segment{nullptr, std::move(function)});
  begin_segment();
}

void PiecewiseGraph::replay() {
  CHECK(!capturing_);
  for (const auto& segment : segments_) {
    if (segment.graph != nullptr) {
      segment.graph->replay();
    } else {
      segment.eager(kv_max_seq_len_);
    }
  }
}

std::size_t PiecewiseGraph::num_graphs() const {
  return std::count_if(
      segments_.begin(), segments_.end(), [](const auto& segment) {
        return segment.graph != nullptr;
      });
}

std::size_t PiecewiseGraph::num_eager_segments() const {
  return segments_.size() - num_graphs();
}

}  // namespace xllm::kernel::mlu
