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

#include "core/runtime/py_runtime_module.h"

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <torch/python.h>

#include <cstdint>
#include <string>
#include <vector>

#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/runtime/py_input_batch.h"
#include "models/py_model_helper.h"

namespace py = pybind11;

namespace xllm {
namespace {

class PyRuntimeModuleTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!Py_IsInitialized()) {
      py::initialize_interpreter(/*init_signal_handlers=*/false);
      thread_state_ = PyEval_SaveThread();
    }
  }

  // The registered module owns pybind11 types. Finalize the interpreter with
  // the GIL held so those types are released before the process tears down its
  // statics; otherwise pybind11 aborts the exit with "dec_ref() is being called
  // while the GIL is either not held or invalid".
  static void TearDownTestSuite() {
    if (thread_state_ == nullptr) {
      return;
    }
    PyEval_RestoreThread(thread_state_);
    thread_state_ = nullptr;
    py::finalize_interpreter();
  }

  static PyThreadState* thread_state_;
};

PyThreadState* PyRuntimeModuleTest::thread_state_ = nullptr;

TEST_F(PyRuntimeModuleTest, RegistersInRunningInterpreterAndPreservesIdentity) {
  py::gil_scoped_acquire gil;
  ensure_xllm_runtime_module();
  py::module_ module = py::module_::import("xllm_runtime");
  py::object metadata_type = module.attr("InputBatchMetadataView");

  ensure_xllm_runtime_module();
  EXPECT_TRUE(module.is(py::module_::import("xllm_runtime")));
  EXPECT_TRUE(metadata_type.is(module.attr("InputBatchMetadataView")));
  EXPECT_TRUE(py::hasattr(module, "AttentionMetadataView"));
  EXPECT_TRUE(py::hasattr(module, "ExpandedDecodeMetadataView"));
  EXPECT_TRUE(py::hasattr(module, "LayerLoadSynchronizer"));
}

TEST_F(PyRuntimeModuleTest, CallsCollectiveWithoutActiveModel) {
  py::gil_scoped_acquire gil;
  py::module_::import("torch");
  ensure_xllm_runtime_module();
  py::module_ module = py::module_::import("xllm_runtime");
  torch::Tensor input = torch::arange(4, torch::kFloat32);
  torch::Tensor output =
      module.attr("tp_all_reduce")(input).cast<torch::Tensor>();
  EXPECT_TRUE(output.is_same(input));
  EXPECT_TRUE(module.attr("tp_all_to_all")(input, 1).is_none());
}

TEST_F(PyRuntimeModuleTest, ExposesInputBatchMetadata) {
  py::gil_scoped_acquire gil;
  ensure_xllm_runtime_module();
  ModelInputParams params;
  params.execution_batch.num_reqs = 2;
  params.execution_batch.num_tokens = 3;
  params.execution_batch.num_scheduled_tokens = {1, 2};
  py::object view = py::cast(PyInputBatchMetadataView(params));
  EXPECT_EQ(view.attr("num_reqs").cast<int32_t>(), 2);
  EXPECT_EQ(view.attr("num_tokens").cast<int64_t>(), 3);
  EXPECT_EQ(view.attr("num_scheduled_tokens").cast<std::vector<int32_t>>(),
            (std::vector<int32_t>{1, 2}));
}

TEST_F(PyRuntimeModuleTest, RegistersWeightLoaderAndReadsStateDict) {
  py::gil_scoped_acquire gil;
  py::module_::import("torch");
  ensure_xllm_weight_loader_module();
  py::module_ module = py::module_::import("xllm_weight_loader");
  py::object state_dict_type = module.attr("StateDict");
  ensure_xllm_weight_loader_module();
  EXPECT_TRUE(module.is(py::module_::import("xllm_weight_loader")));
  EXPECT_TRUE(state_dict_type.is(module.attr("StateDict")));

  torch::Tensor weight = torch::arange(4, torch::kFloat32);
  StateDict state_dict({{"weight", weight}});
  py::object view = py::cast(PyStateDict(&state_dict));
  EXPECT_TRUE(view.attr("has")("weight").cast<bool>());
  EXPECT_FALSE(view.attr("has")("missing").cast<bool>());
  EXPECT_EQ(view.attr("keys")().cast<std::vector<std::string>>(),
            (std::vector<std::string>{"weight"}));
  EXPECT_TRUE(
      view.attr("get_tensor")("weight").cast<torch::Tensor>().is_same(weight));
}

}  // namespace
}  // namespace xllm
