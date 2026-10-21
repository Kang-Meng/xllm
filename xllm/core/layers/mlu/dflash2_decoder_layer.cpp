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

#include "core/layers/mlu/dflash2_decoder_layer.h"

namespace xllm::layer::mlu {

DFlash2DecoderLayerImpl::DFlash2DecoderLayerImpl(const ModelContext& context,
                                                 int32_t layer_id)
    : parallel_args_(context.get_parallel_args()) {
  const auto& model_args = context.get_model_args();
  CHECK(is_dflash2_draft_model_type(model_args.model_type()));
  const auto& quant_args = context.get_quant_args();
  const auto& options = context.get_tensor_options();
  const std::string mlp_module_prefix =
      layer_id >= 0 ? "model.layers." + std::to_string(layer_id) + ".mlp" : "";

  // Initialize attention layers
  attention_ = register_module("self_attn", Qwen2Attention(context));

  // Initialize norm layers
  input_norm_ = register_module(
      "input_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  post_norm_ = register_module(
      "post_attention_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  // Initialize mlp
  mlp_ = register_module("mlp",
                         DenseMLP(model_args.hidden_size(),
                                  model_args.intermediate_size(),
                                  true,
                                  false,
                                  model_args.hidden_act(),
                                  /*enable_result_reduction=*/true,
                                  quant_args,
                                  parallel_args_.tp_group_,
                                  options,
                                  mlp_module_prefix));

  attention_conv_ =
      register_module("attention_conv",
                      DFlash2GroupedConv(model_args.hidden_size(),
                                         model_args.dflash2_conv_kernel_size(),
                                         model_args.dflash2_conv_group_size(),
                                         model_args.dflash2_block_size(),
                                         options));
  mlp_conv_ =
      register_module("mlp_conv",
                      DFlash2GroupedConv(model_args.hidden_size(),
                                         model_args.dflash2_conv_kernel_size(),
                                         model_args.dflash2_conv_group_size(),
                                         model_args.dflash2_block_size(),
                                         options));
}

void DFlash2DecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  attention_->load_state_dict(state_dict.get_dict_with_prefix("self_attn."));
  input_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("input_layernorm."));
  post_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("post_attention_layernorm."));
  mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));

  attention_conv_->load_state_dict(
      state_dict.get_dict_with_prefix("attention_conv."));
  mlp_conv_->load_state_dict(state_dict.get_dict_with_prefix("mlp_conv."));
}

void DFlash2DecoderLayerImpl::verify_loaded_weights(
    const std::string& prefix) const {
  attention_->verify_loaded_weights(prefix + "self_attn.");
  mlp_->verify_loaded_weights(prefix + "mlp.");
  input_norm_->verify_loaded_weights(prefix + "input_layernorm.");
  post_norm_->verify_loaded_weights(prefix + "post_attention_layernorm.");

  attention_conv_->verify_loaded_weights(prefix + "attention_conv.");
  mlp_conv_->verify_loaded_weights(prefix + "mlp_conv.");
}

torch::Tensor DFlash2DecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  auto pre_fp8_scale = attention_->get_fp8_input_scale();
  auto post_fp8_scale = mlp_->get_fp8_input_scale();

  // Pre-attention norm
  std::tie(x, residual) =
      input_norm_->forward_with_residual(x, residual, pre_fp8_scale);

  torch::Tensor attention_coefficients;
  std::tie(x, attention_coefficients) = attention_conv_->prepare(x);

  // Attention
  x = attention_->forward(positions, x, attn_metadata, kv_cache);
  bool post_norm_done = false;
  if (residual.has_value() && !post_fp8_scale.has_value() &&
      !post_norm_->bias().defined()) {
    if (auto fused = attention_conv_->finish_with_norm(x,
                                                       attention_coefficients,
                                                       residual.value(),
                                                       post_norm_->weight(),
                                                       post_norm_->eps())) {
      x = std::get<0>(*fused);
      residual = std::get<1>(*fused);
      post_norm_done = true;
    }
  }
  if (!post_norm_done) {
    x = attention_conv_->finish(x, attention_coefficients);

    // Post-attention norm for the explicitly unsupported fusion contracts.
    std::tie(x, residual) =
        post_norm_->forward_with_residual(x, residual, post_fp8_scale);
  }

  torch::Tensor mlp_coefficients;
  std::tie(x, mlp_coefficients) = mlp_conv_->prepare(x);

  // MLP
  x = mlp_->forward(x);
  x = mlp_conv_->finish(x, mlp_coefficients);

  return x;
}

}  // namespace xllm::layer::mlu
