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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <string>
#include <type_traits>
#include <unordered_map>

#include "core/framework/parallel_state/process_group.h"
#include "core/layers/mlu/dflash2_decoder_layer.h"
#include "core/platform/platform.h"
#include "models/llm/mlu/qwen3_dflash2.h"

namespace xllm {
namespace {

using MluDraftModel = mlu::model::DFlash2Qwen3ModelImpl;
static_assert(
    std::is_base_of_v<QWen3ModelImplBase<layer::mlu::DFlash2DecoderLayer>,
                      MluDraftModel>);

ModelArgs make_draft_args() {
  ModelArgs args;
  args.model_type("DFlash2DraftModel")
      .n_layers(1)
      .hidden_size(128)
      .intermediate_size(256)
      .hidden_act("silu")
      .vocab_size(16)
      .head_dim(128)
      .n_heads(1)
      .n_kv_heads(1)
      .attention_bias(false)
      .max_position_embeddings(32)
      .rope_theta(10000)
      .rms_norm_eps(1e-5)
      .sliding_window(16)
      .dflash2_block_size(4)
      .dflash2_conv_group_size(16)
      .dflash2_conv_kernel_size(2)
      .dflash2_selector_rank(8)
      .dflash2_selector_top_k(4);
  return args;
}

std::unordered_map<std::string, torch::Tensor> make_decoder_weights(
    const torch::TensorOptions& options) {
  std::unordered_map<std::string, torch::Tensor> weights;
  for (const std::string& projection :
       {"q_proj", "k_proj", "v_proj", "o_proj"}) {
    weights.emplace("self_attn." + projection + ".weight",
                    torch::full({128, 128}, 0.125, options));
  }
  for (const std::string& norm : {"self_attn.q_norm",
                                  "self_attn.k_norm",
                                  "input_layernorm",
                                  "post_attention_layernorm"}) {
    weights.emplace(norm + ".weight", torch::ones({128}, options));
  }
  weights.emplace("mlp.gate_proj.weight",
                  torch::full({256, 128}, 0.25, options));
  weights.emplace("mlp.up_proj.weight", torch::full({256, 128}, 0.5, options));
  weights.emplace("mlp.down_proj.weight",
                  torch::full({128, 256}, 0.125, options));
  for (const std::string& conv : {"attention_conv", "mlp_conv"}) {
    weights.emplace(conv + ".base_kernel",
                    torch::full({2, 2, 128}, 0.75, options));
    weights.emplace(conv + ".kernel_projection.weight",
                    torch::full({32, 128}, 0.25, options));
  }
  return weights;
}

TEST(DFlash2MluModelTest, DedicatedDecoderRetainsCheckpointModuleKeys) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  ProcessGroup group(0, 1, device);
  ParallelArgs parallel(0, 1, &group);
  parallel.tp_group_ = &group;
  ModelArgs args = make_draft_args();
  ModelContext context(parallel, args, QuantArgs(), options);
  layer::mlu::DFlash2DecoderLayer decoder(context, 0);
  const auto weights = make_decoder_weights(options);
  decoder->load_state_dict(StateDict(weights));
  decoder->verify_loaded_weights("layers.0.");
  const auto parameters = decoder->named_parameters();
  for (const std::string& conv : {"attention_conv", "mlp_conv"}) {
    for (const std::string& suffix :
         {"base_kernel", "kernel_projection.weight"}) {
      const std::string name = conv + "." + suffix;
      ASSERT_TRUE(parameters.contains(name)) << name;
      EXPECT_TRUE(torch::equal(parameters[name], weights.at(name))) << name;
    }
  }

  // Ordinary Qwen3 must retain the common decoder without local convolutions.
  args.model_type("qwen3");
  ModelContext ordinary_context(parallel, args, QuantArgs(), options);
  layer::Qwen3DecoderLayer ordinary(ordinary_context, 0);
  const auto ordinary_modules = ordinary->named_modules();
  EXPECT_FALSE(ordinary_modules.contains("attention_conv"));
  EXPECT_FALSE(ordinary_modules.contains("mlp_conv"));
}

TEST(DFlash2MluModelTest, ActualDraftModelLoadsUnchangedLayerPrefixes) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  ProcessGroup group(0, 1, device);
  ParallelArgs parallel(0, 1, &group);
  parallel.tp_group_ = &group;
  ModelContext context(parallel, make_draft_args(), QuantArgs(), options);
  MluDraftModel model(context);
  std::unordered_map<std::string, torch::Tensor> weights;
  for (const auto& [name, value] : make_decoder_weights(options)) {
    weights.emplace("layers.0." + name, value);
  }
  weights.emplace("fc.weight", torch::ones({128, 128}, options));
  weights.emplace("hidden_norm.weight", torch::ones({128}, options));
  weights.emplace("norm.weight", torch::ones({128}, options));
  weights.emplace("candidate_selector.hidden_projection.weight",
                  torch::ones({8, 128}, options));
  weights.emplace("candidate_selector.predecessor_codebook",
                  torch::ones({16, 8}, options));
  weights.emplace("candidate_selector.successor_codebook",
                  torch::ones({16, 8}, options));
  model.load_state_dict(StateDict(weights));
  // Verifies the real model's decoder vector, including both convolution
  // modules. The vector was never registered as torch named_modules.
  model.verify_loaded_weights();
  const auto modules =
      model.named_modules(/*name_prefix=*/"", /*include_self=*/false);
  EXPECT_FALSE(modules.contains("embed_tokens"));
  EXPECT_TRUE(modules.contains("fc"));
  EXPECT_TRUE(modules.contains("hidden_norm"));
}

}  // namespace
}  // namespace xllm
