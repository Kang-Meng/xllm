/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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
#include <cmath>
#include <tuple>
#include <unordered_map>

#include "core/framework/state_dict/utils.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/common/word_embedding.h"
#include "core/layers/deepseek_v4_decoder_layer.h"
#include "layers/common/rotary_embedding_util.h"
#include "llm_model_base.h"

namespace xllm {

// DSA cache type enum for DeepSeek V4 multi-cache management
enum class DSACacheType : int32_t {
  TOKEN = 0,           // block allocated by token count / ratio
  SEQUENCE = 1,        // one block per sequence
  SLIDING_WINDOW = 2,  // sliding window, fixed number of blocks per seq
};

// Per-cache metadata within a layer
struct DSACacheInfo {
  int32_t group_id;    // which block manager group this cache belongs to
  DSACacheType type;   // cache type
  int32_t ratio;       // compression ratio
  int32_t block_size;  // block size for this cache
};

// Group key: (ratio, type, block_size) -> group_id
struct DSAGroupKey {
  int32_t ratio;
  DSACacheType type;
  int32_t block_size;
  bool operator==(const DSAGroupKey& o) const {
    return ratio == o.ratio && type == o.type && block_size == o.block_size;
  }
};

struct DSAGroupKeyHash {
  size_t operator()(const DSAGroupKey& k) const {
    size_t h = std::hash<int32_t>()(k.ratio);
    h ^= std::hash<int32_t>()(static_cast<int32_t>(k.type)) << 16;
    h ^= std::hash<int32_t>()(k.block_size) << 8;
    return h;
  }
};

// Group-level info
struct DSAGroupInfo {
  DSACacheType type;
  int32_t ratio;
  int32_t block_size;
};

class DeepseekV4ModelImpl
    : public LlmModelImplBase<layer::DeepseekV4DecoderLayer> {
 public:
  explicit DeepseekV4ModelImpl(const ModelContext& context)
      : LlmModelImplBase<layer::DeepseekV4DecoderLayer>(
            "deepseek_v4",
            context.get_model_args()) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    layers_.reserve(model_args.n_layers());
    norm_ = register_module("norm", layer::RMSNorm(context));
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));

    hc_mult_ = std::max<int64_t>(model_args.hc_mult(), 1);
    hc_eps_ = static_cast<double>(model_args.hc_eps());
    norm_eps_ = static_cast<double>(model_args.rms_norm_eps());

    const int64_t hc_dim = hc_mult_ * model_args.hidden_size();
    auto hc_options = options.dtype(torch::kFloat32);
    hc_head_fn_ =
        register_parameter("hc_head_fn",
                           torch::empty({hc_mult_, hc_dim}, hc_options),
                           /*requires_grad=*/false);
    hc_head_base_ = register_parameter("hc_head_base",
                                       torch::empty({hc_mult_}, hc_options),
                                       /*requires_grad=*/false);
    hc_head_scale_ = register_parameter("hc_head_scale",
                                        torch::empty({1}, hc_options),
                                        /*requires_grad=*/false);

    const int64_t rope_head_dim = model_args.rope_head_dim();
    const int64_t max_pos = model_args.max_position_embeddings();
    if (rope_head_dim > 0 && max_pos > 0) {
      const int64_t original_max_pos =
          model_args.rope_scaling_original_max_position_embeddings() > 0
              ? model_args.rope_scaling_original_max_position_embeddings()
              : max_pos;
      const float scaling_factor =
          model_args.factor() > 0.0f ? model_args.factor() : 1.0f;
      const float attn_factor = model_args.rope_scaling_attn_factor() > 0.0f
                                    ? model_args.rope_scaling_attn_factor()
                                    : 1.0f;
      dsa_cos_sin_ = layer::rotary::get_deepseek_rotary_embedding(
          /*head_size=*/model_args.head_dim(),
          /*rotary_dim=*/rope_head_dim,
          /*max_position_embeddings=*/max_pos,
          /*rope_scaling_original_max_position_embeddings=*/original_max_pos,
          /*rope_theta=*/model_args.rope_theta(),
          /*interleaved=*/false,
          /*scaling_factor=*/scaling_factor,
          /*extrapolation_factor=*/model_args.rope_extrapolation_factor(),
          /*attn_factor=*/attn_factor,
          /*beta_fast=*/model_args.beta_fast(),
          /*beta_slow=*/model_args.beta_slow(),
          /*mscale=*/model_args.rope_scaling_mscale(),
          /*mscale_all_dim=*/model_args.rope_scaling_mscale_all_dim(),
          options);
    }

    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      auto layer = layer::DeepseekV4DecoderLayer(context);
      layers_.push_back(layer);
    }

    // Build DSA caches_info from compress_ratios
    const auto& compress_ratios = model_args.compress_ratios();
    const int32_t window_size =
        model_args.window_size() > 0 ? model_args.window_size() : 128;
    const int32_t base_block_size = 128;  // default block size

    std::unordered_map<DSAGroupKey, int32_t, DSAGroupKeyHash> group_key_map;
    caches_info_.resize(model_args.n_layers());

    for (int32_t layer_id = 0; layer_id < model_args.n_layers(); ++layer_id) {
      int32_t cr = (layer_id < static_cast<int32_t>(compress_ratios.size()))
                       ? compress_ratios[layer_id]
                       : 1;
      // Build per-layer cache specs based on compress_ratio
      struct CacheEntry {
        DSACacheType type;
        int32_t ratio;
        int32_t block_size;
      };
      std::vector<CacheEntry> layer_caches;

      if (cr == 1) {
        // C1: 1 cache (swa)
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
      } else if (cr == 4) {
        // C4: 8 caches
        // compress_kv(TOKEN,4,128), compress_index(TOKEN,4,128),
        // swa(SW,1,window), kv_state(SW,1,window), score_state(SW,1,window),
        // idx_kv_state(SW,1,window), idx_score_state(SW,1,window),
        // indexer_scale(TOKEN,4,128)
        layer_caches.push_back({DSACacheType::TOKEN, 4, base_block_size});
        layer_caches.push_back({DSACacheType::TOKEN, 4, base_block_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::TOKEN, 4, base_block_size});
      } else if (cr == 128) {
        // C128: 4 caches
        // compress_kv(TOKEN,128,128), swa(SW,1,window),
        // kv_state(SW,1,window), score_state(SW,1,window)
        layer_caches.push_back({DSACacheType::TOKEN, 128, base_block_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
      }

      for (const auto& ce : layer_caches) {
        DSAGroupKey gk{ce.ratio, ce.type, ce.block_size};
        int32_t gid;
        auto it = group_key_map.find(gk);
        if (it == group_key_map.end()) {
          gid = static_cast<int32_t>(group_infos_.size());
          group_key_map[gk] = gid;
          group_infos_.push_back({ce.type, ce.ratio, ce.block_size});
        } else {
          gid = it->second;
        }
        caches_info_[layer_id].push_back(
            {gid, ce.type, ce.ratio, ce.block_size});
      }
    }
  }

  void load_state_dict(const StateDict& state_dict) override {
    LlmModelImplBase<layer::DeepseekV4DecoderLayer>::load_state_dict(
        state_dict);
    LOAD_WEIGHT(hc_head_fn);
    LOAD_WEIGHT(hc_head_base);
    LOAD_WEIGHT(hc_head_scale);
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) override {
    if (tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
    }

    auto inputs_embeds = input_params.input_embedding;
    torch::Tensor h =
        inputs_embeds.defined() ? inputs_embeds : embed_tokens_(tokens);

    if (h.dim() == 2) {
      h = h.unsqueeze(1).repeat({1, hc_mult_, 1});
    }

    auto modified_input_params = input_params;
    auto& dp_token_nums = modified_input_params.dp_global_token_nums;
    // DP helper: keep zero entries at least 1 to avoid empty slices/padding
    // in xllm DP utilities. DeepSeek V4 not use DP today.
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);

    if (!modified_input_params.attn_metadata) {
      modified_input_params.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              layer::AttentionMetadataBuilder::build(modified_input_params));
    }
    auto& attn_metadata = *(modified_input_params.attn_metadata);

    if (dsa_cos_sin_.defined() && positions.defined()) {
      torch::Tensor cos_sin = dsa_cos_sin_;
      if (cos_sin.device() != positions.device()) {
        cos_sin = cos_sin.to(positions.device());
      }
      auto target = cos_sin.index({positions});
      auto chunks = target.chunk(/*chunks=*/2, /*dim=*/-1);
      attn_metadata.dsa_cos = chunks[0].contiguous();
      attn_metadata.dsa_sin = chunks[1].contiguous();
    }

    // Build DSA per-layer block_tables and slot_mappings from
    // multi_block_tables (block -> slot expansion, layer expansion,
    // processing).
    if (!modified_input_params.multi_block_tables.empty() &&
        !caches_info_.empty()) {
      // Set input_positions for c4/c128 compressed RoPE computation
      attn_metadata.dsa_input_positions = positions;
      build_dsa_metadata(modified_input_params, attn_metadata);
    }

    std::optional<torch::Tensor> residual;
    for (size_t i = 0; i < layers_.size(); i++) {
      h = layers_[i](h,
                     residual,
                     positions,
                     attn_metadata,
                     kv_caches[i],
                     modified_input_params);
    }
    h = hc_head(h);
    auto [hidden_states, residual_out] = norm_(h, std::nullopt);
    return ModelOutput(hidden_states, residual_out);
  }

 private:
  torch::Tensor hc_head(const torch::Tensor& x) {
    auto x_float = x.to(torch::kFloat32);
    auto x_flatten = x_float.flatten(-2, -1);
    auto rsqrt = torch::rsqrt(x_flatten.pow(2).mean(-1, true) + norm_eps_);
    auto mixes = torch::matmul(x_flatten, hc_head_fn_.transpose(0, 1));
    mixes = mixes * rsqrt;
    auto pre = torch::sigmoid(mixes * hc_head_scale_ + hc_head_base_) + hc_eps_;
    auto y = (pre.unsqueeze(-1) * x_float).sum(-2);
    return y.to(x.dtype());
  }

  // Build DSA block_tables and slot_mappings per layer per cache.
  //   Step 1: block -> slot expansion per manager
  //   Step 2: per-group processing (TOKEN sort+truncate / SWA left-pad)
  //   Step 3: expand manager-level to [layer][cache] using group_id
  void build_dsa_metadata(const ModelInputParams& params,
                          layer::AttentionMetadata& attn_metadata) const {
    const int32_t manager_num =
        static_cast<int32_t>(params.multi_block_tables.size());
    const int32_t n_layers = static_cast<int32_t>(caches_info_.size());
    const int32_t batch_size =
        static_cast<int32_t>(params.kv_seq_lens_vec.size());
    const auto& ctx_lens = params.kv_seq_lens_vec;
    int64_t total_tokens = 0;
    for (auto len : ctx_lens) total_tokens += len;

    // Step 1: block -> slot expansion per manager
    std::vector<torch::Tensor> mgr_slots(manager_num);
    for (int32_t m = 0; m < manager_num; ++m) {
      mgr_slots[m] = expand_blocks_to_slots(params.multi_block_tables[m],
                                            group_infos_[m],
                                            ctx_lens,
                                            batch_size,
                                            total_tokens);
    }

    // Step 2: per-group processing (done once per group, shared across layers)
    std::vector<torch::Tensor> proc_slots(manager_num);
    std::vector<torch::Tensor> proc_bt(manager_num);
    for (int32_t m = 0; m < manager_num; ++m) {
      process_group(params.multi_block_tables[m],
                    mgr_slots[m],
                    group_infos_[m],
                    ctx_lens,
                    batch_size,
                    total_tokens,
                    proc_bt[m],
                    proc_slots[m]);
    }

    // Step 3: expand by layer using group_id
    attn_metadata.dsa_block_tables.resize(n_layers);
    attn_metadata.dsa_slot_mappings.resize(n_layers);
    for (int32_t lid = 0; lid < n_layers; ++lid) {
      const auto& lci = caches_info_[lid];
      attn_metadata.dsa_block_tables[lid].resize(lci.size());
      attn_metadata.dsa_slot_mappings[lid].resize(lci.size());
      for (size_t ci = 0; ci < lci.size(); ++ci) {
        int32_t gid = lci[ci].group_id;
        if (gid < manager_num) {
          attn_metadata.dsa_block_tables[lid][ci] = proc_bt[gid];
          attn_metadata.dsa_slot_mappings[lid][ci] = proc_slots[gid];
        }
      }
    }

    // Build actual_seq_lengths_kv and actual_seq_lengths_query
    build_dsa_seq_lengths(params, batch_size, attn_metadata);

    // Build compressed positions (c4/c128) and input_positions
    build_dsa_positions(params, batch_size, total_tokens, attn_metadata);

    // Attach cache spec pointer
    attn_metadata.dsa_caches_info = &caches_info_;
  }

  // Step 1: expand block_table to slot array for one manager.
  // slot_id = block_id * block_size + offset
  // Returns tensor of shape (total_tokens,), unfilled positions are -1.
  static torch::Tensor expand_blocks_to_slots(const torch::Tensor& block_table,
                                              const DSAGroupInfo& gi,
                                              const std::vector<int>& ctx_lens,
                                              int32_t batch_size,
                                              int64_t total_tokens) {
    const int32_t bs = gi.block_size;
    auto slots = torch::full({total_tokens}, -1, torch::kInt32);
    auto slots_acc = slots.accessor<int32_t, 1>();
    auto bt_acc = block_table.accessor<int32_t, 2>();
    const int32_t max_blocks = block_table.size(1);

    int64_t start_idx = 0;
    for (int32_t seq = 0; seq < batch_size; ++seq) {
      int64_t token_len = ctx_lens[seq];
      int64_t slot_num = compute_slot_num(gi, token_len);

      int64_t filled = 0;
      for (int32_t blk = 0; blk < max_blocks && filled < slot_num; ++blk) {
        int32_t block_id = bt_acc[seq][blk];
        if (block_id < 0) break;
        for (int32_t off = 0; off < bs && filled < slot_num; ++off) {
          slots_acc[start_idx + filled] =
              static_cast<int32_t>(static_cast<int64_t>(block_id) * bs + off);
          ++filled;
        }
      }
      start_idx += token_len;
    }
    return slots;
  }

  // Compute how many slots a single seq needs for this group.
  static int64_t compute_slot_num(const DSAGroupInfo& gi, int64_t token_len) {
    if (gi.type == DSACacheType::TOKEN) {
      return token_len / gi.ratio;
    }
    // SLIDING_WINDOW
    const int32_t bs = gi.block_size;
    if (token_len > bs) {
      return token_len % bs + bs;
    }
    int64_t n = token_len % bs;
    return (n == 0 && token_len > 0) ? bs : n;
  }

  // Step 2: per-group processing.
  // TOKEN:  sort slots (valid first), truncate, replace -1 with 0.
  //         block_tables unchanged.
  // SWA:    replace -1 with 0 in slots.
  //         left-pad block_tables with 0.
  static void process_group(const torch::Tensor& raw_bt,
                            const torch::Tensor& raw_slots,
                            const DSAGroupInfo& gi,
                            const std::vector<int>& ctx_lens,
                            int32_t batch_size,
                            int64_t total_tokens,
                            torch::Tensor& out_bt,
                            torch::Tensor& out_slots) {
    if (gi.type == DSACacheType::TOKEN) {
      process_token_group(raw_bt,
                          raw_slots,
                          gi.ratio,
                          batch_size,
                          total_tokens,
                          out_bt,
                          out_slots);
    } else if (gi.type == DSACacheType::SLIDING_WINDOW) {
      process_swa_group(raw_bt,
                        raw_slots,
                        gi.block_size,
                        ctx_lens,
                        batch_size,
                        out_bt,
                        out_slots);
    } else {
      out_slots = torch::where(
          raw_slots.eq(-1), torch::zeros_like(raw_slots), raw_slots);
      out_bt = raw_bt;
    }
  }

  // TOKEN group: sort slots (valid first, -1 last), truncate, -1 -> 0.
  static void process_token_group(const torch::Tensor& raw_bt,
                                  const torch::Tensor& raw_slots,
                                  int32_t ratio,
                                  int32_t batch_size,
                                  int64_t total_tokens,
                                  torch::Tensor& out_bt,
                                  torch::Tensor& out_slots) {
    int64_t op_need_length = std::min(
        total_tokens / ratio + static_cast<int64_t>(batch_size), total_tokens);
    auto sort_key = torch::where(raw_slots.eq(-1),
                                 torch::ones_like(raw_slots),
                                 torch::zeros_like(raw_slots));
    auto sorted_idx =
        sort_key.argsort(/*dim=*/0, /*descending=*/false, /*stable=*/true);
    auto slots = raw_slots.index_select(0, sorted_idx)
                     .slice(/*dim=*/0, /*start=*/0, /*end=*/op_need_length)
                     .contiguous();
    out_slots = torch::where(slots.eq(-1), torch::zeros_like(slots), slots);
    out_bt = raw_bt;  // keep original right-padded block_tables
  }

  // SWA group: replace -1 with 0 in slots; left-pad block_tables.
  static void process_swa_group(const torch::Tensor& raw_bt,
                                const torch::Tensor& raw_slots,
                                int32_t block_size,
                                const std::vector<int>& ctx_lens,
                                int32_t batch_size,
                                torch::Tensor& out_bt,
                                torch::Tensor& out_slots) {
    out_slots =
        torch::where(raw_slots.eq(-1), torch::zeros_like(raw_slots), raw_slots);

    int32_t current_cols = raw_bt.size(1);
    int32_t max_dst_len = 0;
    std::vector<int32_t> dst_lens(batch_size);
    for (int32_t s = 0; s < batch_size; ++s) {
      dst_lens[s] = static_cast<int32_t>(
          std::ceil(static_cast<double>(ctx_lens[s]) / block_size));
      max_dst_len = std::max(max_dst_len, dst_lens[s]);
    }
    max_dst_len = std::max(max_dst_len, current_cols);

    auto new_bt = torch::zeros({batch_size, max_dst_len}, raw_bt.options());
    auto new_acc = new_bt.accessor<int32_t, 2>();
    auto old_acc = raw_bt.accessor<int32_t, 2>();

    for (int32_t s = 0; s < batch_size; ++s) {
      int32_t pad_len = dst_lens[s] - current_cols;
      if (pad_len > 0) {
        for (int32_t j = 0; j < current_cols; ++j)
          new_acc[s][pad_len + j] = old_acc[s][j];
      } else if (pad_len < 0) {
        for (int32_t j = 0; j < dst_lens[s]; ++j) new_acc[s][j] = old_acc[s][j];
      } else {
        for (int32_t j = 0; j < current_cols; ++j)
          new_acc[s][j] = old_acc[s][j];
      }
    }
    out_bt = new_bt;
  }

  // Build actual_seq_lengths_kv and actual_seq_lengths_query.
  // Prefill: kv = context_length, query = pad(cumsum(kv), (1,0), 0)
  // Decode:  kv = context_length, query = pad(cumsum(ones), (1,0), 0)
  static void build_dsa_seq_lengths(const ModelInputParams& params,
                                    int32_t batch_size,
                                    layer::AttentionMetadata& attn_metadata) {
    auto kv_lens =
        torch::tensor(std::vector<int32_t>(params.kv_seq_lens_vec.begin(),
                                           params.kv_seq_lens_vec.end()),
                      torch::kInt32);
    attn_metadata.dsa_actual_seq_lengths_kv = kv_lens;

    torch::Tensor q_lens;
    if (params.is_prefill) {
      // prefill: query lengths = context lengths
      q_lens = kv_lens;
    } else {
      // decode: each seq has query length = 1
      q_lens = torch::ones({batch_size}, torch::kInt32);
    }
    // cumsum with leading 0: shape (batch_size+1,)
    auto cumsum = torch::cumsum(q_lens, /*dim=*/0, /*dtype=*/torch::kInt32);
    attn_metadata.dsa_actual_seq_lengths_query =
        torch::cat({torch::zeros({1}, torch::kInt32), cumsum});
  }

  // Build input_positions, c4_pad_positions, c128_pad_positions.
  // c4_pad_positions: positions where (pos+1) % 4 == 0, adjusted, padded.
  // c128_pad_positions: positions where (pos+1) % 128 == 0, adjusted, padded.
  static void build_dsa_positions(const ModelInputParams& params,
                                  int32_t batch_size,
                                  int64_t total_tokens,
                                  layer::AttentionMetadata& attn_metadata) {
    // input_positions from q_seq_lens_vec (flatten positions)
    // Already available as model forward `positions` arg, but we also store it
    // in attn_metadata for DSA operator use.
    // Note: this will be set from the `positions` tensor passed to forward().
    // Here we only compute c4/c128 pad positions if input_positions is set.
    if (!attn_metadata.dsa_input_positions.defined()) return;

    auto input_positions = attn_metadata.dsa_input_positions;
    int64_t num_tokens = input_positions.size(0);

    // C4 compressed positions
    auto c4_mask = ((input_positions + 1) % 4).eq(0);
    auto c4_pos = input_positions.index({c4_mask});
    c4_pos = (c4_pos + 1) - 4;
    int64_t c4_target = std::min(num_tokens, num_tokens / 4 + batch_size);
    int64_t c4_pad_right = c4_target - c4_pos.size(0);
    if (c4_pad_right > 0) {
      attn_metadata.dsa_c4_pad_positions =
          torch::cat({c4_pos, torch::zeros({c4_pad_right}, c4_pos.options())});
    } else {
      attn_metadata.dsa_c4_pad_positions = c4_pos.slice(0, 0, c4_target);
    }

    // C128 compressed positions
    auto c128_mask = ((input_positions + 1) % 128).eq(0);
    auto c128_pos = input_positions.index({c128_mask});
    c128_pos = (c128_pos + 1) - 128;
    int64_t c128_target = std::min(num_tokens, num_tokens / 128 + batch_size);
    int64_t c128_pad_right = c128_target - c128_pos.size(0);
    if (c128_pad_right > 0) {
      attn_metadata.dsa_c128_pad_positions = torch::cat(
          {c128_pos, torch::zeros({c128_pad_right}, c128_pos.options())});
    } else {
      attn_metadata.dsa_c128_pad_positions = c128_pos.slice(0, 0, c128_target);
    }
  }

  torch::Tensor dsa_cos_sin_;

  int64_t hc_mult_ = 1;
  double hc_eps_ = 0.0;
  double norm_eps_ = 1e-6;

  // DSA cache group info: built once at model init from compress_ratios
  // caches_info_[layer_id] = vector of DSACacheInfo for each cache in that
  // layer
  std::vector<std::vector<DSACacheInfo>> caches_info_;
  // group_infos_[group_id] = DSAGroupInfo
  std::vector<DSAGroupInfo> group_infos_;

  DEFINE_WEIGHT(hc_head_fn);
  DEFINE_WEIGHT(hc_head_base);
  DEFINE_WEIGHT(hc_head_scale);
};
TORCH_MODULE(DeepseekV4Model);

class DeepseekV4ForCausalLMImpl
    : public LlmForCausalLMImplBase<DeepseekV4Model> {
 public:
  explicit DeepseekV4ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DeepseekV4Model>(context) {}
};
TORCH_MODULE(DeepseekV4ForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(deepseek_v4, DeepseekV4ForCausalLM);

// register the model args
REGISTER_MODEL_ARGS(deepseek_v4, [&] {
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v4");
  LOAD_ARG_OR(dtype, "torch_dtype", "");

  // Basic model structure
  LOAD_ARG_OR_FUNC(hidden_size, "dim", [&] { return args->hidden_size(); });
  LOAD_ARG_OR_FUNC(
      hidden_size, "hidden_size", [&] { return args->hidden_size(); });
  LOAD_ARG_OR_FUNC(
      n_layers, "num_hidden_layers", [&] { return args->n_layers(); });
  LOAD_ARG_OR_FUNC(n_heads, "n_heads", [&] { return args->n_heads(); });
  LOAD_ARG_OR_FUNC(
      n_heads, "num_attention_heads", [&] { return args->n_heads(); });
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 1);
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    if (args->head_dim() > 0) {
      return args->head_dim();
    }
    if (args->hidden_size() > 0 && args->n_heads() > 0) {
      return args->hidden_size() / args->n_heads();
    }
    return int64_t{0};
  });
  LOAD_ARG_OR_FUNC(
      vocab_size, "vocab_size", [&] { return args->vocab_size(); });
  LOAD_ARG_OR_FUNC(max_position_embeddings, "max_position_embeddings", [&] {
    return args->max_position_embeddings();
  });
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR_FUNC(intermediate_size, "intermediate_size", [&] {
    if (args->intermediate_size() > 0) {
      return args->intermediate_size();
    }
    if (args->moe_intermediate_size() > 0) {
      return static_cast<int64_t>(args->moe_intermediate_size());
    }
    if (args->hidden_size() > 0) {
      return args->hidden_size() * 4;
    }
    return int64_t{0};
  });

  // Norm / RoPE
  LOAD_ARG_OR_FUNC(
      rms_norm_eps, "norm_eps", [&] { return args->rms_norm_eps(); });
  LOAD_ARG_OR_FUNC(
      rms_norm_eps, "rms_norm_eps", [&] { return args->rms_norm_eps(); });
  LOAD_ARG_OR_FUNC(
      rope_theta, "rope_theta", [&] { return args->rope_theta(); });
  LOAD_ARG_OR_FUNC(
      rope_head_dim, "rope_head_dim", [&] { return args->rope_head_dim(); });

  // LoRA / groups
  LOAD_ARG_OR_FUNC(
      q_lora_rank, "q_lora_rank", [&] { return args->q_lora_rank(); });
  LOAD_ARG_OR_FUNC(
      o_lora_rank, "o_lora_rank", [&] { return args->o_lora_rank(); });
  LOAD_ARG_OR_FUNC(o_groups, "o_groups", [&] { return args->o_groups(); });

  // KV compression / windowing
  LOAD_ARG(compress_ratios, "compress_ratios");
  LOAD_ARG_OR_FUNC(compress_rope_theta, "compress_rope_theta", [&] {
    return args->compress_rope_theta();
  });
  LOAD_ARG_OR_FUNC(
      window_size, "window_size", [&] { return args->window_size(); });

  // MoE routing (DeepSeek V4)
  LOAD_ARG_OR_FUNC(n_routed_experts, "n_routed_experts", [&] {
    return args->n_routed_experts();
  });
  LOAD_ARG_OR_FUNC(n_activated_experts, "n_activated_experts", [&] {
    return args->n_activated_experts();
  });
  LOAD_ARG_OR_FUNC(
      n_hash_layers, "n_hash_layers", [&] { return args->n_hash_layers(); });
  LOAD_ARG_OR_FUNC(
      route_scale, "route_scale", [&] { return args->route_scale(); });
  LOAD_ARG_OR_FUNC(
      score_func, "score_func", [&] { return args->score_func(); });

  // Indexer
  LOAD_ARG_OR_FUNC(
      index_head_dim, "index_head_dim", [&] { return args->index_head_dim(); });
  LOAD_ARG_OR_FUNC(
      index_n_heads, "index_n_heads", [&] { return args->index_n_heads(); });
  LOAD_ARG_OR_FUNC(
      index_topk, "index_topk", [&] { return args->index_topk(); });

  // HC / DSA helpers
  LOAD_ARG_OR_FUNC(hc_mult, "hc_mult", [&] { return args->hc_mult(); });
  LOAD_ARG_OR_FUNC(hc_sinkhorn_iters, "hc_sinkhorn_iters", [&] {
    return args->hc_sinkhorn_iters();
  });
  LOAD_ARG_OR_FUNC(hc_eps, "hc_eps", [&] { return args->hc_eps(); });
  LOAD_ARG_OR_FUNC(factor, "factor", [&] { return args->factor(); });
  LOAD_ARG_OR_FUNC(beta_fast, "beta_fast", [&] { return args->beta_fast(); });
  LOAD_ARG_OR_FUNC(beta_slow, "beta_slow", [&] { return args->beta_slow(); });
  LOAD_ARG_OR_FUNC(scale_fmt, "scale_fmt", [&] { return args->scale_fmt(); });

  // Runtime sizing hints
  LOAD_ARG_OR_FUNC(
      max_batch_size, "max_batch_size", [&] { return args->max_batch_size(); });
  LOAD_ARG_OR_FUNC(
      max_seq_len, "max_seq_len", [&] { return args->max_seq_len(); });

  // Token ids
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm
