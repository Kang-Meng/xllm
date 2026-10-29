# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python import kernels
from xllm.python.layers import ColumnParallelLinear
from xllm.python.model_loader import (
    ParallelLoadContext,
    ScopedWeightLoader,
    load_causal_lm_weights,
)
from xllm.python.models.base import PyModelBase
from xllm.python.models.multimodal import (
    EncoderOutputs,
    EncoderTokenCounts,
    MultimodalData,
    MultimodalInputs,
    merge_multimodal_embeddings,
)
from xllm.python.models.qwen3_5 import Qwen3_5Config, Qwen3_5Model

# Ordered Qwen3.5 backbone roots; exact keys keep sibling towers out.
_LLM_SRC_PREFIXES = ("model.language_model.", "language_model.", "model.", "")


@dataclass
class JoyTalkerAudioConfig:
    """Flat ModelArgs view of the Qwen3-Omni audio encoder args."""

    d_model: int = 1280
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    num_mel_bins: int = 128
    downsample_hidden_size: int = 480
    output_dim: int = 2048
    max_source_positions: int = 1500
    n_window: int = 50
    n_window_infer: int = 800
    conv_chunksize: int = 500

    @classmethod
    def from_dict(cls, d: dict) -> JoyTalkerAudioConfig:
        def pick(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in d and d[key] is not None:
                    return d[key]
            return default

        num_mel_bins = int(pick("mm_audio_num_mel_bins", default=128))
        if int(pick("mm_audio_feature_size", default=num_mel_bins)) != num_mel_bins:
            raise ValueError("audio num_mel_bins and feature_size must match")
        return cls(
            d_model=int(pick("mm_audio_hidden_size", default=1280)),
            encoder_layers=int(pick("mm_audio_encoder_layers", default=32)),
            encoder_attention_heads=int(pick("mm_audio_num_attention_heads", default=20)),
            encoder_ffn_dim=int(pick("mm_audio_encoder_ffn_dim", default=5120)),
            num_mel_bins=num_mel_bins,
            downsample_hidden_size=int(pick("mm_audio_downsample_hidden_size", default=480)),
            output_dim=int(pick("mm_audio_output_dim", default=2048)),
            max_source_positions=int(pick("mm_audio_max_source_positions", default=1500)),
            n_window=int(pick("mm_audio_n_window", default=50)),
            n_window_infer=int(pick("mm_audio_n_window_infer", default=800)),
            conv_chunksize=int(pick("mm_audio_conv_chunksize", default=500)),
        )


def _get_feat_extract_output_lengths(input_lengths: torch.Tensor, n_window: int = 50) -> torch.Tensor:
    """Post-CNN token count, matching transformers' Qwen3-Omni encoder."""
    chunk_len = n_window * 2
    input_lengths_leave = input_lengths % chunk_len
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    # Full 100 frames -> 50 -> 25 -> 13 tokens via convs; tail collapses to // 4 + 1.
    return (feat_lengths - 1) // 4 + 1 + (input_lengths // chunk_len) * 13


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length: int, channels: int, max_timescale: float = 10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length).unsqueeze(1) * inv_timescales.unsqueeze(0)
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int) -> torch.Tensor:
        if seqlen > self.positional_embedding.shape[0]:
            raise ValueError(
                f"audio encoder needs {seqlen} positional steps but only "
                f"{self.positional_embedding.shape[0]} are registered"
            )
        return self.positional_embedding[:seqlen, :]


class JoyTalkerAudioAttention(nn.Module):
    """Multi-head self-attention over packed audio windows (non-causal)."""

    def __init__(self, cfg: JoyTalkerAudioConfig):
        super().__init__()
        self.embed_dim = cfg.d_model
        self.num_heads = cfg.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("d_model must be divisible by encoder_attention_heads")
        self.scaling = self.head_dim**-0.5
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: list[int]) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        num_heads = self.num_heads
        head_dim = self.head_dim

        query = self.q_proj(hidden_states).view(seq_length, num_heads, head_dim)
        key = self.k_proj(hidden_states).view(seq_length, num_heads, head_dim)
        value = self.v_proj(hidden_states).view(seq_length, num_heads, head_dim)

        # Packed TND attention takes cumulative ends; fp32 softmax matches HF.
        attn_output = kernels.packed_infer_attention(
            query=query,
            key=key,
            value=value,
            actual_seq_lengths=cu_seqlens[1:],
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            scale=self.scaling,
            input_layout="TND",
        )
        return self.out_proj(attn_output.reshape(seq_length, num_heads * head_dim))


class JoyTalkerAudioEncoderLayer(nn.Module):
    def __init__(self, cfg: JoyTalkerAudioConfig, activation: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.self_attn = JoyTalkerAudioAttention(cfg)
        self.self_attn_layer_norm = nn.LayerNorm(cfg.d_model)
        self.final_layer_norm = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.encoder_ffn_dim)
        self.fc2 = nn.Linear(cfg.encoder_ffn_dim, cfg.d_model)
        self.activation_fn = activation

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: list[int]) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cu_seqlens)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return residual + hidden_states


class JoyTalkerAudioEncoder(nn.Module):
    """Qwen3-Omni audio tower; parameter names match the HF checkpoint."""

    def __init__(self, cfg: JoyTalkerAudioConfig):
        super().__init__()
        embed_dim = cfg.d_model
        self.n_window = cfg.n_window
        self.n_window_infer = cfg.n_window_infer
        self.conv_chunksize = cfg.conv_chunksize

        self.positional_embedding = SinusoidsPositionEmbedding(cfg.max_source_positions, embed_dim)
        activation = F.gelu
        self.layers = nn.ModuleList(JoyTalkerAudioEncoderLayer(cfg, activation) for _ in range(cfg.encoder_layers))
        self.ln_post = nn.LayerNorm(embed_dim)
        self.conv2d1 = nn.Conv2d(1, cfg.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(cfg.downsample_hidden_size, cfg.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(cfg.downsample_hidden_size, cfg.downsample_hidden_size, 3, 2, padding=1)
        conv_bins = (((cfg.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        self.conv_out = nn.Linear(cfg.downsample_hidden_size * conv_bins, embed_dim, bias=False)
        self.proj1 = nn.Linear(embed_dim, embed_dim)
        self.proj2 = nn.Linear(embed_dim, cfg.output_dim)

    def _chunk_lengths(self, feature_lens: list[int]) -> list[int]:
        """Frame count of every ``n_window * 2`` chunk, tail chunk included."""
        chunk_size = self.n_window * 2
        chunk_lengths: list[int] = []
        for length in feature_lens:
            chunk_lengths.extend([chunk_size] * (length // chunk_size))
            remainder = length % chunk_size
            if remainder != 0:
                chunk_lengths.append(remainder)
        return chunk_lengths

    def _chunk_and_pad(
        self,
        input_features: torch.Tensor,
        chunk_lengths: list[int],
    ) -> torch.Tensor:
        """Gather the packed frames into zero-padded chunks on device.

        Returns ``[num_chunks, num_mel_bins, max_chunk_len]``. This replaces a
        host-side ``split`` plus ``pad_sequence`` loop with one gather.
        """
        if not chunk_lengths:
            raise ValueError("audio encoder received no mel frames; every audio item is empty")
        width = max(chunk_lengths)
        frames = input_features.t()
        expected_frames = sum(chunk_lengths)
        if frames.shape[0] != expected_frames:
            raise ValueError(f"audio encoder expected {expected_frames} mel frames, got {frames.shape[0]}")
        device = frames.device
        lengths = torch.tensor(chunk_lengths, dtype=torch.long, device=device)
        offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)[:-1]])
        columns = torch.arange(width, device=device)
        positions = offsets[:, None] + columns[None, :]
        valid = columns[None, :] < lengths[:, None]
        # positions [chunks, width] gathers frames; pad slots are zeroed.
        padded = frames[positions.masked_fill(~valid, 0)]
        return padded.masked_fill(~valid.unsqueeze(-1), 0).transpose(1, 2)

    def forward(self, input_features: torch.Tensor, feature_lens: torch.Tensor) -> torch.Tensor:
        """Args:
        input_features: ``[num_mel_bins, total_frames]`` packed audio features.
        feature_lens: ``[num_audios]`` per-item frame counts.

        Returns ``[total_audio_tokens, output_dim]``.
        """
        # Fused attention needs host cu_seqlens; lengths are read once.
        lengths = feature_lens.to(dtype=torch.long).tolist()
        chunk_lengths = self._chunk_lengths(lengths)
        padded_feature = self._chunk_and_pad(input_features, chunk_lengths)

        chunk_token_lens = _get_feat_extract_output_lengths(
            torch.tensor(chunk_lengths, dtype=torch.long), self.n_window
        )
        max_len_after_cnn = int(chunk_token_lens.max())
        device = padded_feature.device
        padded_mask_after_cnn = torch.arange(max_len_after_cnn, device=device) < chunk_token_lens.to(device)[:, None]
        padded_feature = padded_feature.unsqueeze(1)
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            embed = F.gelu(self.conv2d1(chunk))
            embed = F.gelu(self.conv2d2(embed))
            embed = F.gelu(self.conv2d3(embed))
            padded_embeds.append(embed)
        padded_embed = torch.cat(padded_embeds, dim=0)

        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
        positional_embedding = self.positional_embedding(padded_embed.shape[1]).unsqueeze(0)
        padded_embed = padded_embed + positional_embedding.to(padded_embed.dtype)
        hidden_states = padded_embed[padded_mask_after_cnn]

        # Vectorized cu_seqlens; empty items add no window, as in HF.
        aftercnn_lens = _get_feat_extract_output_lengths(torch.tensor(lengths, dtype=torch.long), self.n_window)
        window_aftercnn = max_len_after_cnn * (self.n_window_infer // (self.n_window * 2))
        full_windows = aftercnn_lens // window_aftercnn
        remainders = aftercnn_lens % window_aftercnn
        window_counts = full_windows + (remainders > 0).long()
        columns = torch.arange(int(window_counts.max()))
        window_lengths = torch.where(
            columns[None, :] == full_windows[:, None],
            remainders[:, None],
            remainders.new_full((), window_aftercnn),
        )
        window_lengths = window_lengths[columns[None, :] < window_counts[:, None]]
        cu_seqlens = [0, *window_lengths.cumsum(0).tolist()]

        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens)

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states


class JoyTalkerAudioProjector(nn.Module):
    def __init__(self, audio_hidden_size: int, text_hidden_size: int, bias: bool):
        super().__init__()
        # Release projector is a fixed 2x MLP; no config field.
        intermediate_size = audio_hidden_size * 2
        self.proj1 = nn.Linear(audio_hidden_size, intermediate_size, bias=bias)
        self.proj2 = nn.Linear(intermediate_size, text_hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(F.gelu(self.proj1(x)))


class JoyTalkerForConditionalGeneration(PyModelBase):
    # Stateless per-item audio embeddings can use the encoder cache.
    cache_multimodal_embeddings = True

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.audio_cfg = JoyTalkerAudioConfig.from_dict(config)

        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "cuda"))
        # Share Qwen3.5 policy so metadata builders see the same flag.
        self.text_cfg = Qwen3_5Config.prepare_execution_config(config, device)
        self.dtype = dtype
        self.device = device

        self.audio_tower = JoyTalkerAudioEncoder(self.audio_cfg).to(dtype=dtype, device=device)
        self.audio_projector = JoyTalkerAudioProjector(
            self.audio_cfg.output_dim,
            self.text_cfg.hidden_size,
            bool(config.get("mm_projector_bias", False)),
        ).to(dtype=dtype, device=device)

        tp = self.text_cfg.tp_size
        self.model = Qwen3_5Model(self.text_cfg, dtype, device)
        self.lm_head = ColumnParallelLinear(
            self.text_cfg.hidden_size,
            self.text_cfg.vocab_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

    def encode(self, input_features: torch.Tensor, feature_lens: torch.Tensor) -> torch.Tensor:
        """Run the audio tower + projector over packed features.

        Args:
            input_features: ``[total_frames, num_mel_bins]`` (audio processor
                output, items concatenated on the frame dimension).
            feature_lens: ``[num_audios]`` per-item valid frame counts.

        Returns:
            ``[total_audio_tokens, text_hidden_size]``.
        """
        features = input_features.to(dtype=self.dtype, device=self.device)
        # Keep lengths on their device; the tower reads them once.
        feature_lens = feature_lens.to(dtype=torch.long)
        audio_features = self.audio_tower(features.t(), feature_lens)
        return self.audio_projector(audio_features)

    def encode_multimodal(self, data: MultimodalData, token_counts: EncoderTokenCounts) -> EncoderOutputs:
        counts = token_counts.get("audio", [])
        if not counts:
            return {}
        input_features = data["input_features"]
        feature_lens = data["feat_origin_lens"]
        if not isinstance(input_features, torch.Tensor) or not isinstance(feature_lens, torch.Tensor):
            raise TypeError("batched audio features and lengths must be tensors")
        features = self.encode(input_features, feature_lens)
        if features.shape[0] != sum(counts):
            raise ValueError(f"audio encoder produced {features.shape[0]} tokens but the prompt reserves {sum(counts)}")
        return {"audio|embedding": list(features.split(counts, dim=0))}

    def prepare_multimodal_inputs(self, input_ids: torch.Tensor, data: MultimodalData) -> MultimodalInputs:
        inputs_embeds = self.model.embed_tokens(input_ids)
        return MultimodalInputs(merge_multimodal_embeddings(inputs_embeds, data))

    def load_weights(
        self,
        state_dicts: list,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        tower_weights = ScopedWeightLoader(state_dicts, src_prefixes=("audio_tower.encoder.",))
        self._load_replicated(self.audio_tower, tower_weights)

        projector_weights = ScopedWeightLoader(state_dicts, src_prefixes=("audio_projector.",))
        self._load_replicated(self.audio_projector, projector_weights)

        llm_weights = ScopedWeightLoader(state_dicts, src_prefixes=_LLM_SRC_PREFIXES)
        load_causal_lm_weights(
            self.model,
            self.lm_head.weight,
            llm_weights,
            ParallelLoadContext.from_config(self.text_cfg, tp_rank, tp_size),
            tie_word_embeddings=self.text_cfg.tie_word_embeddings,
            embed_fallback=True,
        )

    def _load_replicated(
        self,
        module: nn.Module,
        weights: ScopedWeightLoader,
    ) -> None:
        """Load every parameter of ``module`` from a replicated checkpoint
        namespace, mirroring checkpoint names one-to-one."""
        for name, param in module.named_parameters():
            weights.load_tensor(param, name)
