/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "framework/config/model_config.h"
#include "models/vlm/npu/qwen3_omni_moe_thinker.h"

namespace xllm::npu::model {

// JoyAI omni-duplex audio tower: stock qwen3-omni audio tower plus an
// optional CTC head (Linear d_model -> vocab, from ctc.safetensors) tapped
// between ln_post and the adaptor.
class JoyaiDuplexAudioTransformerImpl
    : public Qwen3OmniMoe_Thinker_AudioTransformerImpl {
 public:
  explicit JoyaiDuplexAudioTransformerImpl(const ModelContext& context)
      : Qwen3OmniMoe_Thinker_AudioTransformerImpl(context) {}

  void clear_ctc() { ctc_.clear(); }

  void on_encoder_output(const torch::Tensor& hidden_states,
                         const torch::Tensor& clip_lens) override {
    clear_ctc();
    if (!ctc_lo_.is_empty()) {
      ctc_ = ctc_greedy_decode(hidden_states, clip_lens);
    }
  }

  const MMDict& ctc() const { return ctc_; }

  void load_state_dict(const StateDict& state_dict) override {
    Qwen3OmniMoe_Thinker_AudioTransformerImpl::load_state_dict(state_dict);
    // load_state_dict is invoked per checkpoint shard; only the shard carrying
    // ctc.safetensors defines these tensors.
    auto ctc_w = state_dict.get_tensor("ctc_lo.weight");
    if (!ctc_w.defined()) {
      return;
    }
    if (ctc_lo_.is_empty()) {
      ctc_lo_ = register_module(
          "ctc_lo",
          torch::nn::Linear(
              torch::nn::LinearOptions(static_cast<int64_t>(ctc_w.size(1)),
                                       static_cast<int64_t>(ctc_w.size(0)))
                  .bias(true)));
    }
    weight::load_weight(
        state_dict, "ctc_lo.weight", ctc_lo_->weight, is_ctc_lo_weight_loaded_);
    weight::load_weight(
        state_dict, "ctc_lo.bias", ctc_lo_->bias, is_ctc_lo_bias_loaded_);
    LOG(INFO) << "JoyAI duplex CTC head loaded: in=" << ctc_w.size(1)
              << " out=" << ctc_w.size(0);
  }

 private:
  // Greedy CTC decode (blank 0, collapse repeats). encoder_outs:
  // [total_tokens, d_model] concatenated across
  // clips; clip_lens: per-clip token counts.
  // Returns the sealed side-output dict: "audio|ctc_ids" -> one id tensor
  // per clip, "audio|ctc_conf" -> one fp32 scalar per clip.
  MMDict ctc_greedy_decode(const torch::Tensor& encoder_outs,
                           const torch::Tensor& clip_lens) {
    MMDict ctc;
    std::vector<torch::Tensor> ids_per_clip;
    std::vector<torch::Tensor> conf_per_clip;
    constexpr int32_t kBlankId = 0;
    if (!encoder_outs.defined() || encoder_outs.size(0) == 0) {
      return ctc;
    }
    auto lens_cpu = clip_lens.to(torch::kCPU).contiguous();
    const int64_t* lens_ptr = lens_cpu.data_ptr<int64_t>();
    const int64_t total = encoder_outs.size(0);

    // Top-1 of every clip is computed on device first, then one cat + D2H;
    // a per-clip .to(kCPU) syncs the stream repeatedly and dominates the
    // overhead.
    std::vector<torch::Tensor> dev_probs;
    std::vector<torch::Tensor> dev_idx;
    std::vector<int64_t> lens;
    int64_t offset = 0;
    for (int64_t clip = 0; clip < lens_cpu.size(0); ++clip) {
      int64_t len = std::max<int64_t>(1, lens_ptr[clip]);
      len = std::min(len, total - offset);
      if (len <= 0) {
        break;
      }
      // Per-clip forward keeps the [len, vocab] workspace in model dtype.
      auto clip_logits =
          ctc_lo_->forward(encoder_outs.narrow(/*dim=*/0, offset, len));
      auto probs = torch::softmax(clip_logits, /*dim=*/-1);
      auto max_res = probs.max(/*dim=*/-1);
      dev_probs.push_back(std::get<0>(max_res));
      dev_idx.push_back(std::get<1>(max_res));
      lens.push_back(len);
      offset += len;
    }
    if (dev_probs.empty()) {
      return ctc;
    }
    auto all_probs =
        torch::cat(dev_probs).to(torch::kCPU, torch::kFloat32).contiguous();
    auto all_idx = torch::cat(dev_idx).to(torch::kCPU).contiguous();
    const float* prob_ptr = all_probs.data_ptr<float>();
    const int64_t* idx_ptr = all_idx.data_ptr<int64_t>();

    offset = 0;
    for (size_t clip = 0; clip < lens.size(); ++clip) {
      const int64_t len = lens[clip];
      auto ids =
          torch::empty({len}, torch::TensorOptions().dtype(torch::kInt32));
      int32_t* ids_ptr = ids.data_ptr<int32_t>();
      int64_t ids_count = 0;
      double conf_sum = 0.0;
      int64_t valid_count = 0;
      int64_t prev = -1;
      for (int64_t i = 0; i < len; ++i) {
        const int64_t id = idx_ptr[offset + i];
        if (id != kBlankId) {
          conf_sum += prob_ptr[offset + i];
          ++valid_count;
        }
        if (id != kBlankId && id != prev) {
          ids_ptr[ids_count++] = static_cast<int32_t>(id);
        }
        prev = id;
      }
      float confidence = 0.0f;
      if (valid_count > 0) {
        confidence = static_cast<float>(conf_sum / valid_count);
        // Clamp out-of-range confidence produced by bf16 softmax.
        if (confidence < 0.0f || confidence > 1.0f) {
          confidence = 0.0f;
        }
      }
      ids_per_clip.push_back(ids.narrow(/*dim=*/0, 0, ids_count).clone());
      conf_per_clip.push_back(torch::tensor({confidence}));
      offset += len;
    }
    // One single-element confidence tensor per clip.
    if (!ids_per_clip.empty()) {
      ctc["audio|ctc_ids"] = std::move(ids_per_clip);
      ctc["audio|ctc_conf"] = std::move(conf_per_clip);
    }
    return ctc;
  }

  torch::nn::Linear ctc_lo_{nullptr};
  MMDict ctc_;
  bool is_ctc_lo_weight_loaded_ = false;
  bool is_ctc_lo_bias_loaded_ = false;
};
TORCH_MODULE(JoyaiDuplexAudioTransformer);

// Duplex thinker: stock thinker with the CTC-augmented audio tower swapped
// in, packing per-clip CTC tensors into the MMDict for the executor.
class JoyaiDuplexThinkerImpl
    : public Qwen3OmniMoe_Thinker_ForConditionalGenerationImpl {
 public:
  explicit JoyaiDuplexThinkerImpl(const ModelContext& context)
      : Qwen3OmniMoe_Thinker_ForConditionalGenerationImpl(context) {
    auto tower = JoyaiDuplexAudioTransformer(context);
    replace_module("audio_tower", tower);
    audio_tower_ = Qwen3OmniMoe_Thinker_AudioTransformer(
        std::static_pointer_cast<Qwen3OmniMoe_Thinker_AudioTransformerImpl>(
            tower.ptr()));
  }

  MMDict get_multimodal_embeddings(const ModelInputParams& input_params) {
    // The duplex model is audio-only and the tower is always the CTC one.
    auto* tower =
        static_cast<JoyaiDuplexAudioTransformerImpl*>(audio_tower_.get());
    // The tower only refills CTC state when it actually encodes; clear
    // beforehand so a round without fresh audio never reports stale clips.
    tower->clear_ctc();
    MMDict multimodal_embeds =
        Qwen3OmniMoe_Thinker_ForConditionalGenerationImpl::
            get_multimodal_embeddings(input_params);
    const auto& ctc = tower->ctc();
    const auto ids_it = ctc.find("audio|ctc_ids");
    if (ids_it == ctc.end()) {
      return multimodal_embeds;
    }
    const auto& all_ids = std::get<std::vector<torch::Tensor>>(ids_it->second);
    const auto& all_conf =
        std::get<std::vector<torch::Tensor>>(ctc.at("audio|ctc_conf"));
    CHECK_EQ(all_ids.size(), all_conf.size());
    if (!all_ids.empty()) {
      multimodal_embeds["audio|ctc_ids"] = all_ids;
      multimodal_embeds["audio|ctc_conf"] = all_conf;
    }
    return multimodal_embeds;
  }
};
TORCH_MODULE(JoyaiDuplexThinker);

REGISTER_MULTIMODAL_PROCESSOR(joyai_duplex, Qwen3OmniMoeMultimodalProcessor);
REGISTER_CAUSAL_VLM_MODEL(joyai_duplex, JoyaiDuplexThinker);
REGISTER_MPOSITION_GENERATOR(joyai_duplex, Qwen3OmniMPositionGenerator);
REGISTER_MODEL_ARGS_LOADER(joyai_duplex, load_qwen3_omni_moe_args);

}  // namespace xllm::npu::model
