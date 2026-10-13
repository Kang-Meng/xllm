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

#include "core/runtime/py_multimodal_executor.h"

#include <glog/logging.h>
#include <pybind11/stl.h>
#include <torch/python.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/util/pybind_helper.h"

namespace py = pybind11;

namespace xllm {
namespace {

using EncoderTokenCounts =
    std::unordered_map<std::string, std::vector<int32_t>>;

EncoderTokenCounts collect_encoder_token_counts(const MMBatchData& data) {
  EncoderTokenCounts token_counts;
  for (const auto& entry : data.mm_data_vec()) {
    CHECK(entry.hold<MMItemVec>())
        << "Python VLM requires scheduled multimodal items";
    for (const auto& item : entry.items<MMItemVec>()) {
      if (item.is_embedded()) {
        continue;
      }
      const auto modality = item.type().to_string();
      CHECK(modality.has_value())
          << "unexpected multimodal item type: " << item.type();
      token_counts[modality.value()].emplace_back(item.state().mm_token_num());
    }
  }
  return token_counts;
}

}  // namespace

PyMultimodalExecutor::PyMultimodalExecutor(py::object model,
                                           int64_t max_cache_bytes)
    : model_(std::move(model)) {
  py::gil_scoped_acquire gil;
  py::object protocol = py::module_::import("xllm.python.models.multimodal")
                            .attr("SupportsMultimodal");
  CHECK(py::isinstance(model_, protocol))
      << "Python VLM requires encode_multimodal and prepare_multimodal_inputs";
  const bool enable_encoder_cache =
      model_.attr("cache_multimodal_embeddings").cast<bool>();
  executor_ = std::make_unique<EncoderExecutor>(
      enable_encoder_cache ? max_cache_bytes : 0);
}

PyMultimodalExecutor::~PyMultimodalExecutor() { clear_python_object(model_); }

MMDict PyMultimodalExecutor::encode(const MMBatchData& data) {
  const EncoderTokenCounts token_counts = collect_encoder_token_counts(data);
  return model_
      .attr("encode_multimodal")(py::cast(data.data()), py::cast(token_counts))
      .cast<MMDict>();
}

py::object PyMultimodalExecutor::prepare(const torch::Tensor& tokens,
                                         const ModelInputParams& params) {
  py::gil_scoped_acquire gil;
  auto& data = params.multimodal.mm_data;
  if (!data.valid()) {
    return py::none();
  }
  executor_->prepare(params, [this](const ModelInputParams& inputs) {
    return encode(inputs.multimodal.mm_data);
  });
  return model_.attr("prepare_multimodal_inputs")(tokens,
                                                  py::cast(data.data()));
}

}  // namespace xllm
