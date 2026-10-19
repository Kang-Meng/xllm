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

#pragma once

#include <framework/graphs/MLUGraph.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "core/framework/model/graph_eager_context.h"

namespace xllm::kernel::mlu {

// A graph and its eager boundaries share fixed input/output addresses.
// The owner keeps the history length alive and refreshes it before every
// replay. All graph segments and eager callbacks must enqueue work on the
// current replay stream, preserving stream FIFO ordering across boundaries.
// Callbacks must not leave dependent work on another stream or host thread.
class PiecewiseGraph final : public GraphEagerContext {
 public:
  explicit PiecewiseGraph(const int32_t& kv_max_seq_len);

  void capture_begin(const torch_mlu::MempoolId_t& pool);
  void capture_end();
  void run_eager(std::function<void(int64_t)> function) override;
  void replay();
  std::size_t num_graphs() const;
  std::size_t num_eager_segments() const;

 private:
  void begin_segment();
  void end_segment();

  struct Segment {
    std::unique_ptr<torch_mlu::MLUGraph> graph;
    std::function<void(int64_t)> eager;
  };

  const int32_t& kv_max_seq_len_;
  torch_mlu::MempoolId_t pool_;
  bool capturing_ = false;
  std::vector<Segment> segments_;
  std::unique_ptr<torch_mlu::MLUGraph> current_graph_;
};

}  // namespace xllm::kernel::mlu
