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

#include <memory>
#include <string>
#include <vector>

#include "core/framework/model/causal_lm.h"
#include "core/layers/common/attention_metadata.h"
#include "models/llm/llm_model_base.h"
#include "models/llm/mlu/deepseek_mtp.h"
#include "models/llm/mlu/glm5_next_graph.h"
#include "models/model_registry.h"

namespace xllm::mlu::model {

// Native GLM checkpoints retain the appended MTP layer's original index. The
// native draft shares vocabulary weights with the target after loading its
// transformer body; Python drafts use their own exported vocabulary weights.
template <typename MtpModelType>
class Glm5NextMtpCheckpointImplBase
    : public LlmForCausalLMImplBase<MtpModelType> {
 public:
  explicit Glm5NextMtpCheckpointImplBase(const ModelContext& context)
      : LlmForCausalLMImplBase<MtpModelType>(context) {}

  bool reports_loaded_vocab_weights() const { return true; }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    for (const auto& state_dict : loader->get_state_dicts()) {
      StateDict model_state = state_dict->get_dict_with_prefix(
          std::vector<std::string>{"model.language_model.", prefix});
      this->model_->load_state_dict(model_state);
    }
    this->model_->verify_loaded_weights();
  }
};

class Glm5NextMtpForCausalLMImpl final
    : public Glm5NextMtpCheckpointImplBase<DeepseekMtpModel> {
 public:
  explicit Glm5NextMtpForCausalLMImpl(const ModelContext& context)
      : Glm5NextMtpCheckpointImplBase<DeepseekMtpModel>(
            check_context(context)) {}

  bool requires_graph_forward_metadata() { return true; }

  std::unique_ptr<ModelGraphMetadataState>
  create_graph_forward_metadata_state() {
    return std::make_unique<Glm5NextGraphMetadataState>();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& params) {
    Glm5NextGraphMetadata::prepare(state, positions, params);
  }

 private:
  static const ModelContext& check_context(const ModelContext& context) {
    const auto& args = context.get_model_args();
    CHECK_GE(args.mtp_start_layer_idx(), 0)
        << "Native MLU GLM5 MTP requires a full target checkpoint with an "
           "appended MTP layer; exported draft checkpoints are not supported.";
    CHECK_EQ(args.n_layers(), 1);
    CHECK_EQ(args.num_nextn_predict_layers(), 1);
    return context;
  }
};
TORCH_MODULE(Glm5NextMtpForCausalLM);

const bool glm5_next_mtp_capabilities_registered = []() {
  MtpModelCapabilities capabilities;
  capabilities.supports_native_index_share_for_iteration = false;
  capabilities.supports_accepted_span_replay = true;
  capabilities.python_draft_owns_embedding_and_lm_head = true;
  ModelRegistry::register_mtp_capabilities("glm5_next", capabilities);
  ModelRegistry::register_mtp_capabilities("glm5_next_mtp", capabilities);
  return true;
}();

REGISTER_CAUSAL_MODEL(glm5_next_mtp, Glm5NextMtpForCausalLM);

}  // namespace xllm::mlu::model
