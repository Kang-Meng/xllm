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

#include "core/runtime/py_executor_impl.h"

#include <glog/logging.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/metrics.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/kv_cache/kv_shard_layout.h"
#include "core/framework/model/model_sampling_input.h"
#include "core/framework/speculative/mtp_async_state.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/kv_shard_batch_metadata.h"
#include "core/runtime/py_attention_metadata.h"
#include "core/runtime/py_input_batch.h"
#include "core/util/pybind_helper.h"
#include "models/llm/py_causal_lm.h"

namespace py = pybind11;

namespace xllm {

PyExecutorImpl::PyExecutorImpl(CausalLM* model,
                               const ModelArgs& args,
                               const torch::Device& device,
                               const runtime::Options& options)
    : py_causal_lm_(dynamic_cast<PyCausalLM*>(model)),
      args_(args),
      device_(device),
      options_(options),
      enable_mla_(args.enable_mla()) {
  CHECK(py_causal_lm_ != nullptr) << "PyExecutorImpl requires PyCausalLM";

  py::gil_scoped_acquire gil;
  if (py_causal_lm_->config_dict()["is_multimodal"].cast<bool>()) {
    multimodal_executor_ = std::make_unique<PyMultimodalExecutor>(
        py_causal_lm_->python_model(),
        options.max_encoder_cache_size() * 1024 * 1024);
  }
  py::module_::import("xllm_runtime");
  py::module_ executor_module =
      py::module_::import("xllm.python.model_executor.executor");
  py_model_output_type_ =
      py::module_::import("xllm.python.model_executor.model_output")
          .attr("ModelOutput");
  const bool is_spec_draft =
      options_.is_draft_engine() || (options_.enable_speculative_decode() &&
                                     options_.num_decoding_tokens() == 1);
  py_executor_ = executor_module.attr("ModelExecutor")(
      py_causal_lm_->python_model(),
      py_causal_lm_->config_dict(),
      options_.max_seqs_per_batch(),
      options_.num_decoding_tokens(),
      ExecutionConfig::get_instance().acl_graph_decode_batch_size_limit(),
      py::arg("is_spec_draft") = is_spec_draft);
}

PyExecutorImpl::~PyExecutorImpl() {
  if (PyCausalLM::active_instance() == py_causal_lm_) {
    PyCausalLM::set_active_instance(nullptr);
  }
  clear_python_object(py_executor_);
  clear_python_object(py_model_output_type_);
}

bool PyExecutorImpl::supports_device_only_decode_inputs() const {
  py::gil_scoped_acquire gil;
  return py_executor_
      .attr("supports_device_only_decode_inputs")(options_.block_size())
      .cast<bool>();
}

void PyExecutorImpl::set_device_only_decode_inputs(bool enabled) {
  py::gil_scoped_acquire gil;
  py_executor_.attr("set_device_only_decode_inputs")(enabled);
}

ForwardInput PyExecutorImpl::prepare_inputs(Batch& batch) {
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

ModelOutput PyExecutorImpl::run(const torch::Tensor& tokens,
                                const torch::Tensor& positions,
                                std::vector<KVCache>& kv_caches,
                                const ModelInputParams& params) {
  torch::NoGradGuard no_grad;
  COUNTER_INC(num_model_execution_total_eager);
  PyCausalLM::set_active_instance(py_causal_lm_);

  // Build or reuse attention metadata.
  std::shared_ptr<layer::AttentionMetadata> attn_metadata =
      params.attn_metadata;
  if (!attn_metadata) {
    attn_metadata = std::make_shared<layer::AttentionMetadata>(
        layer::AttentionMetadataBuilder::build(
            params, enable_mla_, std::nullopt, device_));
  }
  // The dense-DCP case is a model with no MLA latent cache (Qwen3.5), which
  // needs KV-shard metadata on every step. An MLA model with cp_size == 1 has
  // no CP group for the DSA prefill to shard across, so attaching the metadata
  // there only trips the sharded-MLA-prefill guard in npu_paged_attention.py;
  // keep the pre-main behavior for MLA and require the active CP group.
  const bool is_decode_context_parallel = !enable_mla_ &&
                                          py_causal_lm_->cp_size() == 1 &&
                                          py_causal_lm_->kv_split_size() > 1;
  const bool is_mla_prefill_kv_shard =
      enable_mla_ && py_causal_lm_->cp_size() > 1 &&
      py_causal_lm_->kv_split_size() > 1 &&
      (attn_metadata->is_prefill || attn_metadata->is_chunked_prefill);
  if (is_decode_context_parallel || is_mla_prefill_kv_shard) {
    const KVShardLayout layout(options_.block_size(),
                               py_causal_lm_->kv_split_size(),
                               py_causal_lm_->kv_split_rank());
    attn_metadata->kv_shard_batch_metadata =
        layer::build_kv_shard_batch_metadata(*attn_metadata, layout);
  }

  py::gil_scoped_acquire gil;

  // Lazy bind KV caches on first call.
  int64_t num_layers = static_cast<int64_t>(kv_caches.size());
  if (!kv_bound_) {
    py::list kv_caches_py;
    for (auto& kv : kv_caches) {
      // Slot order must match ``LayerCache`` on the Python side.
      // Keep this order synchronized with LayerCache/_LAYER_CACHE_SLOTS.
      // Generic caches use the first five entries; DeepSeek-V4 uses the
      // next six, and GLM5 uses the final KPOOL_TAIL entry.
      kv_caches_py.append(
          py::make_tuple(optional_tensor(kv.get_k_cache()),
                         optional_tensor(kv.get_v_cache()),
                         optional_tensor(kv.get_index_cache()),
                         optional_tensor(kv.get_conv_cache()),
                         optional_tensor(kv.get_ssm_cache()),
                         optional_tensor(kv.get_swa_cache()),
                         optional_tensor(kv.get_compress_kv_state()),
                         optional_tensor(kv.get_compress_score_state()),
                         optional_tensor(kv.get_compress_index_kv_state()),
                         optional_tensor(kv.get_compress_index_score_state()),
                         optional_tensor(kv.get_indexer_cache_scale()),
                         optional_tensor(kv.get_kpool_tail())));
    }
    py_executor_.attr("bind_kv_caches")(kv_caches_py);
    kv_bound_ = true;
    kv_layer_count_ = num_layers;
  } else {
    CHECK_EQ(num_layers, kv_layer_count_)
        << "KV cache layer count changed after initial bind";
  }

  py::object py_metadata = py::cast(PyAttentionMetadataView(
      attn_metadata, params, args_.dummy_token_count()));
  py::object py_input_batch_metadata =
      py::cast(PyInputBatchMetadataView(params));
  py::object sampling_input =
      params.sampling.has_value() ? py::cast(*params.sampling) : py::none();
  py::object input_embedding =
      optional_tensor(params.embedding.input_embedding);
  torch::Tensor execution_tokens =
      mtp_async::materialize_graph_speculative_verify_tokens(tokens,
                                                             params.graph);

  py::object multimodal_inputs =
      multimodal_executor_
          ? multimodal_executor_->prepare(execution_tokens, params)
          : py::none();

  // --- mRoPE: collapse [3, N] decode positions to 1-D ---
  // Only PURE decode collapses to 1-D: decode rows are identical
  // (batch_input_builder get_mrope_positions), and mRoPE(p,p,p) == standard
  // RoPE at p, so a single row feeds the captured aclgraph's 1-D
  // static_positions unchanged. Chunked/mixed prefill (is_prefill=false but
  // is_chunked_prefill=true) still needs the full [3, N] for the Python mRoPE
  // path, so it is excluded here. The 2-D shape itself is the mRoPE signal:
  // non-mRoPE models never receive 2-D positions, so no config flag is needed.
  torch::Tensor positions_arg = positions;
  if (positions.dim() == 2 && params.meta.batch_forward_type.is_decode()) {
    positions_arg = positions.select(/*dim=*/0, /*index=*/0);
  }

  py::object py_sync = py::none();
  // Host-load waits are separate from the PD push completion notifications.
  // Python eager waits before each attention; graph execution waits for all
  // loads before warmup/capture/replay, outside the recorded device graph.
  py::object py_load_sync = py::none();
  if (params.parallel.layer_wise_load_synchronizer) {
    py_load_sync = py::cast(params.parallel.layer_wise_load_synchronizer);
  }
#if defined(USE_NPU)
  if (params.parallel.layer_synchronizer) {
    py_sync = py::cast(params.parallel.layer_synchronizer);
  }
#endif

  py::object expert_load_data = optional_tensor(params.expert.expert_load_data);
  py::object eplb_decode_token_mask =
      optional_tensor(params.expert.eplb_decode_token_mask);

  py::object py_output =
      py_executor_.attr("execute")(execution_tokens,
                                   positions_arg,
                                   py_metadata,
                                   input_embedding,
                                   py_sync,
                                   expert_load_data,
                                   eplb_decode_token_mask,
                                   params.meta.is_graph_warmup,
                                   py_input_batch_metadata,
                                   py_load_sync,
                                   params.parallel.layers_per_event,
                                   params.parallel.draft_load_event_index,
                                   multimodal_inputs,
                                   py::arg("sampling_input") = sampling_input);
  if (py::isinstance(py_output, py_model_output_type_)) {
    ModelOutput output;
    auto tensor = [&](const char* name) {
      py::object value = py_output.attr(name);
      return value.is_none() ? torch::Tensor() : value.cast<torch::Tensor>();
    };
    output.hidden_states = tensor("hidden_states");
    output.residual = tensor("residual");
    output.aux_hidden_states = tensor("aux_hidden_states");
    py::object proposal = py_output.attr("draft_proposal");
    if (!proposal.is_none()) {
      torch::Tensor tokens = proposal.attr("token_ids").cast<torch::Tensor>();
      py::object probabilities = proposal.attr("draft_probs");
      output.draft_proposal.emplace(
          std::move(tokens),
          probabilities.is_none()
              ? std::nullopt
              : std::make_optional(probabilities.cast<torch::Tensor>()));
    }
    output.draft_confidence_probs = tensor("draft_confidence_probs");
    return output;
  }
  if (py::isinstance<py::tuple>(py_output)) {
    py::tuple output = py_output.cast<py::tuple>();
    CHECK_EQ(output.size(), 2) << "Python model tuple output must be "
                                  "(hidden_states, aux_hidden_states)";
    return ModelOutput(output[0].cast<torch::Tensor>(),
                       torch::Tensor(),
                       output[1].cast<torch::Tensor>());
  }
  return ModelOutput(py_output.cast<torch::Tensor>());
}

}  // namespace xllm
