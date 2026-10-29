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

#include "models/model_registry.h"
#include "models/vlm/mposition/mposition.h"
#include "processors/image_processor.h"
#include "processors/joytalker_audio_processor.h"
#include "processors/multimodal_processor.h"
#include "processors/qwen3_omni_prompt_processor.h"
#include "processors/video_processor.h"

namespace xllm {

using JoyTalkerMultimodalProcessor =
    MultimodalProcessor<Qwen3OmniPromptProcessor,
                        ImageNoneProcessor,
                        VideoNoneProcessor,
                        JoyTalkerAudioProcessor>;

// HF transformers PaddingStrategy::MAX_LENGTH.
inline constexpr int64_t kPaddingMaxLength = 2;

REGISTER_MODEL_BACKEND(joytalker, "vlm");
REGISTER_MULTIMODAL_PROCESSOR(joytalker, JoyTalkerMultimodalProcessor);
REGISTER_MPOSITION_GENERATOR(joytalker, Qwen3VLMPositionGenerator);

// llm_config wins over the bare key, which stays as the flattened fallback.
#define LOAD_JOYTALKER_LLM_ARG(arg_name, json_key, default_value) \
  LOAD_ARG_OR(arg_name, json_key, default_value);                 \
  LOAD_ARG_OR(arg_name, "llm_config." json_key, args->arg_name())

#define LOAD_JOYTALKER_LLM_ROPE_ARG(arg_name, json_key, default_value) \
  LOAD_ARG_OR(arg_name, "rope_parameters." json_key, default_value);   \
  LOAD_ARG_OR(                                                         \
      arg_name, "llm_config.rope_parameters." json_key, args->arg_name())

REGISTER_MODEL_ARGS(joytalker, [&] {
  SET_ARG(model_type, "joytalker");
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(dtype, "dtype", args->dtype());

  LOAD_JOYTALKER_LLM_ARG(hidden_size, "hidden_size", 2048);
  LOAD_JOYTALKER_LLM_ARG(n_layers, "num_hidden_layers", 40);
  LOAD_JOYTALKER_LLM_ARG(n_heads, "num_attention_heads", 16);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 2);
  LOAD_ARG_OR(n_kv_heads,
              "llm_config.num_key_value_heads",
              args->n_kv_heads().value_or(2));
  LOAD_JOYTALKER_LLM_ARG(head_dim, "head_dim", 256);
  LOAD_JOYTALKER_LLM_ARG(intermediate_size, "intermediate_size", 0);
  LOAD_JOYTALKER_LLM_ARG(vocab_size, "vocab_size", 248320);
  LOAD_JOYTALKER_LLM_ARG(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_JOYTALKER_LLM_ARG(attention_bias, "attention_bias", false);
  LOAD_JOYTALKER_LLM_ARG(
      max_position_embeddings, "max_position_embeddings", 262144);
  LOAD_JOYTALKER_LLM_ARG(eos_token_id, "eos_token_id", 248044);
  LOAD_JOYTALKER_LLM_ARG(tie_word_embeddings, "tie_word_embeddings", false);

  LOAD_JOYTALKER_LLM_ARG(
      layer_types, "layer_types", std::vector<std::string>());
  LOAD_JOYTALKER_LLM_ARG(full_attention_interval, "full_attention_interval", 4);
  LOAD_JOYTALKER_LLM_ARG(attn_output_gate, "attn_output_gate", true);
  LOAD_JOYTALKER_LLM_ARG(linear_conv_kernel_dim, "linear_conv_kernel_dim", 4);
  LOAD_JOYTALKER_LLM_ARG(linear_key_head_dim, "linear_key_head_dim", 128);
  LOAD_JOYTALKER_LLM_ARG(linear_value_head_dim, "linear_value_head_dim", 128);
  LOAD_JOYTALKER_LLM_ARG(linear_num_key_heads, "linear_num_key_heads", 16);
  LOAD_JOYTALKER_LLM_ARG(linear_num_value_heads, "linear_num_value_heads", 32);

  LOAD_JOYTALKER_LLM_ARG(num_experts, "num_experts", 256);
  LOAD_JOYTALKER_LLM_ARG(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_JOYTALKER_LLM_ARG(decoder_sparse_step, "decoder_sparse_step", 1);
  LOAD_JOYTALKER_LLM_ARG(
      mlp_only_layers, "mlp_only_layers", std::vector<int32_t>());
  LOAD_JOYTALKER_LLM_ARG(moe_intermediate_size, "moe_intermediate_size", 512);
  LOAD_JOYTALKER_LLM_ARG(
      shared_expert_intermediate_size, "shared_expert_intermediate_size", 512);
  LOAD_JOYTALKER_LLM_ARG(norm_topk_prob, "norm_topk_prob", true);
  LOAD_JOYTALKER_LLM_ARG(mamba_ssm_dtype, "mamba_ssm_dtype", "float32");

  LOAD_JOYTALKER_LLM_ROPE_ARG(rope_theta, "rope_theta", 10000000.0f);
  LOAD_JOYTALKER_LLM_ROPE_ARG(
      partial_rotary_factor, "partial_rotary_factor", 0.25f);
  LOAD_JOYTALKER_LLM_ARG(rope_scaling_mrope_section,
                         "rope_parameters.mrope_section",
                         std::vector<int64_t>());
  if (args->rope_scaling_mrope_section().empty()) {
    // Release mrope fallback; parentheses keep the comma list in one arg.
    args->rope_scaling_mrope_section() = std::vector<int64_t>({11, 11, 10});
  }
  LOAD_JOYTALKER_LLM_ARG(rope_scaling_mrope_interleaved,
                         "rope_parameters.mrope_interleaved",
                         true);
  LOAD_ARG_OR(rope_scaling_rope_type,
              "llm_config.rope_parameters.rope_type",
              "default");
  if (args->rope_scaling_rope_type() == "default") {
    args->rope_scaling_rope_type() = "mrope";
  }

  LOAD_ARG_OR(audio_token_id, "audio_token_id", 248076);
  LOAD_ARG_OR(audio_start_token_id, "audio_start_token_id", 248070);
  LOAD_ARG_OR(audio_end_token_id, "audio_end_token_id", 248071);
  LOAD_ARG_OR(mm_projector_bias, "projector_bias", false);
  LOAD_ARG_OR(
      mm_audio_num_attention_heads, "audio_config.encoder_attention_heads", 20);
  LOAD_ARG_OR(mm_audio_hidden_size, "audio_config.d_model", 1280);
  LOAD_ARG_OR(mm_audio_downsample_hidden_size,
              "audio_config.downsample_hidden_size",
              480);
  // Both mel-bin fields share the loader's num_mel_bins/feature_size chain.
  LOAD_ARG_OR(mm_audio_num_mel_bins, "audio_config.num_mel_bins", 128);
  LOAD_ARG_OR(mm_audio_feature_size, "audio_config.num_mel_bins", 128);
  LOAD_ARG_OR(
      mm_audio_max_source_positions, "audio_config.max_source_positions", 1500);
  LOAD_ARG_OR(mm_audio_n_window, "audio_config.n_window", 50);
  // Token formula assumes 50-frame chunks; reject other checkpoints.
  CHECK_EQ(args->mm_audio_n_window(), 50)
      << "JoyTalker currently requires audio_config.n_window == 50";
  LOAD_ARG_OR(mm_audio_n_window_infer, "audio_config.n_window_infer", 800);
  LOAD_ARG_OR(mm_audio_conv_chunksize, "audio_config.conv_chunksize", 500);
  // Reject values that zero the window ratio or empty the conv split.
  CHECK_GE(args->mm_audio_n_window_infer(), args->mm_audio_n_window() * 2)
      << "JoyTalker requires audio_config.n_window_infer >= n_window * 2";
  CHECK_GT(args->mm_audio_conv_chunksize(), 0)
      << "JoyTalker requires a positive audio_config.conv_chunksize";
  LOAD_ARG_OR(mm_audio_encoder_layers, "audio_config.encoder_layers", 32);
  LOAD_ARG_OR(mm_audio_output_dim, "audio_config.output_dim", 2048);
  LOAD_ARG_OR(mm_audio_encoder_ffn_dim, "audio_config.encoder_ffn_dim", 5120);

  // HF defaults: pad/truncate at 30s; transformers 5.x ignores n_samples.
  SET_ARG(mm_audio_chunk_length, 30);
  SET_ARG(mm_audio_padding_strategy, kPaddingMaxLength);
  SET_ARG(mm_audio_truncation, true);
  SET_ARG(mm_audio_return_attention_mask, true);

  SET_ARG(vision_start_token_id, -1);
  SET_ARG(vision_end_token_id, -1);
  SET_ARG(image_token_id, -1);
  SET_ARG(video_token_id, -1);

  // <|im_end|>; generation_config.json lists it as an eos token.
  constexpr int32_t kImEndTokenId = 248046;
  SET_ARG(stop_token_ids,
          std::unordered_set<int32_t>({args->eos_token_id(), kImEndTokenId}));
});

#undef LOAD_JOYTALKER_LLM_ROPE_ARG
#undef LOAD_JOYTALKER_LLM_ARG

}  // namespace xllm
