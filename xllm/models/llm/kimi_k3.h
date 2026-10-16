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

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "models/model_registry.h"

namespace xllm {

REGISTER_MODEL_ARGS(kimi_k3, [&] {
  LOAD_ARG_OR(model_type, "model_type", "kimi_k3");
  LOAD_ARG_OR_FUNC(dtype, "dtype", [&] {
    return json.value_or<std::string>("torch_dtype", "bfloat16");
  });

  LOAD_ARG_OR(n_layers, "text_config.num_hidden_layers", 93);
  LOAD_ARG_OR(hidden_size, "text_config.hidden_size", 7168);
  LOAD_ARG_OR(intermediate_size, "text_config.intermediate_size", 33792);
  LOAD_ARG_OR(rms_norm_eps, "text_config.rms_norm_eps", 1e-5);
  LOAD_ARG_OR(tie_word_embeddings, "text_config.tie_word_embeddings", false);
  LOAD_ARG_OR(first_k_dense_replace, "text_config.first_k_dense_replace", 1);
  LOAD_ARG_OR(hidden_act, "text_config.hidden_act", "situ");
  LOAD_ARG_OR(n_routed_experts, "text_config.num_experts", 896);
  LOAD_ARG_OR(n_shared_experts, "text_config.num_shared_experts", 2);
  LOAD_ARG_OR(num_experts_per_tok, "text_config.num_experts_per_token", 16);
  LOAD_ARG_OR(moe_intermediate_size, "text_config.moe_intermediate_size", 3072);
  LOAD_ARG_OR(routed_scaling_factor, "text_config.routed_scaling_factor", 1.0f);
  LOAD_ARG_OR(norm_topk_prob, "text_config.moe_renormalize", true);
  LOAD_ARG_OR(n_group, "text_config.num_expert_group", 1);
  LOAD_ARG_OR(topk_group, "text_config.topk_group", 1);
  LOAD_ARG_OR(
      scoring_func, "text_config.moe_router_activation_func", "sigmoid");
  LOAD_ARG_OR(topk_method, "text_config.topk_method", "noaux_tc");
  LOAD_ARG_OR(n_heads, "text_config.num_attention_heads", 96);
  LOAD_ARG_OR(n_kv_heads, "text_config.num_key_value_heads", 96);
  LOAD_ARG_OR(qk_nope_head_dim, "text_config.qk_nope_head_dim", 128);
  LOAD_ARG_OR(qk_rope_head_dim, "text_config.qk_rope_head_dim", 64);
  LOAD_ARG_OR(v_head_dim, "text_config.v_head_dim", 128);
  LOAD_ARG_OR(q_lora_rank, "text_config.q_lora_rank", 1536);
  LOAD_ARG_OR(kv_lora_rank, "text_config.kv_lora_rank", 512);
  SET_ARG(head_dim, args->qk_nope_head_dim() + args->qk_rope_head_dim());
  SET_ARG(enable_mla, true);
  SET_ARG(rotary_dim, args->qk_rope_head_dim());
  LOAD_ARG_OR(vocab_size, "text_config.vocab_size", 163840);
  LOAD_ARG_OR(
      max_position_embeddings, "text_config.max_position_embeddings", 1048576);
  LOAD_ARG_OR(eos_token_id, "text_config.eos_token_id", 163586);
  LOAD_ARG_OR(pad_token_id, "text_config.pad_token_id", 163839);
  LOAD_ARG_OR(bos_token_id, "text_config.bos_token_id", 163584);
  LOAD_ARG_OR(
      num_nextn_predict_layers, "text_config.num_nextn_predict_layers", 0);

  // Kimi K3 Delta Attention. These flat linear_*
  // fields drive the shared C++ linear-attention cache path
  // (has_linear_attention_layers -> KVCacheShape conv/ssm allocation). Kimi-K3
  // stores its KDA config nested under text_config.linear_attn_config, and it
  // shares one head_dim / num_heads across q/k/v (unlike Qwen GDN), so key and
  // value dims/heads map to the same source field.
  LOAD_ARG_OR(linear_conv_kernel_dim,
              "text_config.linear_attn_config.short_conv_kernel_size",
              4);
  LOAD_ARG_OR(
      linear_key_head_dim, "text_config.linear_attn_config.head_dim", 128);
  LOAD_ARG_OR(
      linear_value_head_dim, "text_config.linear_attn_config.head_dim", 128);
  LOAD_ARG_OR(
      linear_num_key_heads, "text_config.linear_attn_config.num_heads", 96);
  LOAD_ARG_OR(
      linear_num_value_heads, "text_config.linear_attn_config.num_heads", 96);
  // The recurrent (ssm) state must stay fp32 to match the KDA layer contract.
  // The KVCache create-option default is bf16.
  SET_ARG(mamba_ssm_dtype, "float32");
  // Build per-layer types from the 1-based kda_layers list so both
  // has_linear_attention_layers and the per-layer cache dispatch classify
  // Kimi-K3's irregular KDA layout correctly.
  [&] {
    int64_t n_layers = args->n_layers();
    auto kda_layers = json.value<std::vector<int64_t>>(
        "text_config.linear_attn_config.kda_layers");
    if (!kda_layers.has_value() || kda_layers->empty() || n_layers <= 0) {
      return;
    }
    std::unordered_set<int64_t> kda_layer_set(kda_layers->begin(),
                                              kda_layers->end());
    std::vector<std::string> layer_types;
    layer_types.reserve(n_layers);
    for (int64_t layer_id = 0; layer_id < n_layers; ++layer_id) {
      // kda_layers is 1-based; a 0-based layer_id maps to layer_id + 1.
      const bool is_kda = kda_layer_set.count(layer_id + 1) > 0;
      layer_types.emplace_back(is_kda ? "linear_attention" : "full_attention");
    }
    args->layer_types() = std::move(layer_types);
  }();

  LOAD_ARG_OR(mm_num_channels, "vision_config.in_chans", 3);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
  LOAD_ARG_OR(mm_hidden_size, "vision_config.vt_hidden_size", 1024);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.vt_intermediate_size", 4096);
  LOAD_ARG_OR(
      mm_num_attention_heads, "vision_config.vt_num_attention_heads", 12);
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.vt_num_hidden_layers", 27);
  LOAD_ARG_OR_FUNC(mm_projection_dim, "vision_config.text_hidden_size", [&] {
    return json.value_or<int64_t>("text_config.hidden_size", 7168);
  });
  LOAD_ARG_OR(
      mm_projector_type, "vision_config.mm_projector_type", "patchmergerv2");
  LOAD_ARG_OR(
      mm_projector_hidden_act, "vision_config.projector_hidden_act", "gelu");
  LOAD_ARG_OR(mm_layer_norm_eps, "vision_config.projector_ln_eps", 1e-5f);
  [&] {
    auto merge_kernel_size =
        json.value<std::vector<int64_t>>("vision_config.merge_kernel_size");
    args->mm_spatial_merge_size() =
        merge_kernel_size.has_value() && !merge_kernel_size->empty()
            ? (*merge_kernel_size)[0]
            : int64_t(2);
  }();
  SET_ARG(mm_image_merge_size, args->mm_spatial_merge_size());
  LOAD_ARG_OR(mm_init_pos_emb_time, "vision_config.init_pos_emb_time", 4);
  LOAD_ARG_OR(mm_init_pos_emb_width, "vision_config.init_pos_emb_width", 64);
  LOAD_ARG_OR(mm_init_pos_emb_height, "vision_config.init_pos_emb_height", 64);

  SET_ARG(vision_start_token_id, 163602);
  SET_ARG(vision_token_id, 163603);
  SET_ARG(vision_end_token_id, 163604);
  SET_ARG(image_token_id, 163605);
  SET_ARG(video_token_id, 163605);
  SET_ARG(mm_km_patch_size, 14);
  SET_ARG(mm_km_merge_kernel_size, 2);
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({0, 163585, 163586}));
});

REGISTER_TOKENIZER_ARGS(kimi_k3, [&] {
  SET_ARG(tokenizer_type, "tiktoken");
  SET_ARG(vocab_file, "tiktoken.model");
  SET_ARG(special_tokens,
          std::vector<SpecialToken>({{"[BOS]", 163584},
                                     {"[EOS]", 163585},
                                     {"<|end_of_msg|>", 163586},
                                     {"<|open|>", 163587},
                                     {"<|close|>", 163588},
                                     {"<|sep|>", 163589},
                                     {"[start_header_id]", 163590},
                                     {"[end_header_id]", 163591},
                                     {"[EOT]", 163593},
                                     {"<|media_begin|>", 163602},
                                     {"<|media_content|>", 163603},
                                     {"<|media_end|>", 163604},
                                     {"<|media_pad|>", 163605},
                                     {"[UNK]", 163838},
                                     {"[PAD]", 163839}}));
  SET_ARG(visible_special_tokens,
          std::vector<std::string>(
              {"<|end_of_msg|>", "<|open|>", "<|close|>", "<|sep|>"}));
  SET_ARG(
      pattern,
      R"([\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+[^\s]|\s+)");
});

}  // namespace xllm
