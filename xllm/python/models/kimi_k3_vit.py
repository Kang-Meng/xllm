# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kimi K3 MoonViT tower and PatchMergerV2 projector."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python import distributed, kernels
from xllm.python.layers import ColumnParallelLinear, RowParallelLinear
from xllm.python.model_loader import ScopedWeightLoader
from xllm.python.models.base import PyModelBase


def _encoder_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: Sequence[int],
) -> torch.Tensor:
    """Run Kimi packed attention for independent media items.

    Kimi uses fused infer attention, distinct from the mainline
    vision_fusion_attention API's fusion attention operator.
    NPU uses packed attention. The CPU/reference path processes each item
    separately for small-scale validation, avoiding quadratic cross-item masks.
    """
    if query.ndim != 3 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("vision attention expects equal [tokens, heads, dim] tensors")
    if not cu_seqlens or cu_seqlens[0] != 0:
        raise ValueError("vision cu_seqlens must start with zero")
    if cu_seqlens[-1] != query.shape[0]:
        raise ValueError(f"vision cu_seqlens cover {cu_seqlens[-1]} tokens, got {query.shape[0]}")
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        if end <= start:
            raise ValueError(f"vision cu_seqlens must increase, got boundary {start}, {end}")

    actual_seq_lengths = list(cu_seqlens[1:])
    if query.device.type in ("npu", "privateuseone"):
        return kernels.packed_infer_attention(
            query=query,
            key=key,
            value=value,
            actual_seq_lengths=actual_seq_lengths,
            num_heads=query.shape[1],
            num_key_value_heads=key.shape[1],
            scale=query.shape[2] ** -0.5,
            input_layout="TND",
        )

    outputs: list[torch.Tensor] = []
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        q = query[start:end].transpose(0, 1).unsqueeze(0).contiguous()
        k = key[start:end].transpose(0, 1).unsqueeze(0).contiguous()
        v = value[start:end].transpose(0, 1).unsqueeze(0).contiguous()
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        )
        outputs.append(attended.squeeze(0).transpose(0, 1))
    if not outputs:
        return torch.empty_like(query)
    return torch.cat(outputs, dim=0)


def _grid_thw_list(
    grid_thws: torch.Tensor | Sequence[Sequence[int]],
) -> list[tuple[int, int, int]]:
    values = grid_thws.detach().cpu().tolist() if isinstance(grid_thws, torch.Tensor) else grid_thws
    grids: list[tuple[int, int, int]] = []
    for grid in values:
        if len(grid) != 3:
            raise ValueError(f"Kimi K3 grid must contain [t, h, w], got {grid}")
        time, height, width = (int(value) for value in grid)
        if time <= 0 or height <= 0 or width <= 0:
            raise ValueError(f"Kimi K3 grid dimensions must be positive, got {grid}")
        grids.append((time, height, width))
    return grids


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    frequencies: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.shape != key.shape or query.shape[0] != frequencies.shape[0]:
        raise ValueError("Kimi K3 RoPE input shapes do not match")
    if query.shape[-1] % 2 != 0:
        raise ValueError("Kimi K3 RoPE head dimension must be even")

    frequency_pairs = frequencies.unsqueeze(-2)
    query_complex = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
    key_complex = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
    query_out = torch.view_as_real(query_complex * frequency_pairs).flatten(-2)
    key_out = torch.view_as_real(key_complex * frequency_pairs).flatten(-2)
    return query_out.to(query.dtype), key_out.to(key.dtype)


@dataclass
class KimiK3VisionConfig:
    patch_size: int = 14
    in_channels: int = 3
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 12
    num_hidden_layers: int = 27
    qkv_hidden_size: int = 1536
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    init_pos_emb_time: int = 4
    patch_embed_proj_bias: bool = False
    attn_bias: bool = False
    linear_bias: bool = False
    activation_func: str = "gelu_pytorch_tanh"
    norm_type: str = "rmsnorm"
    mlp_type: str = "mlp2"
    pos_emb_type: str = "divided_fixed"
    merge_kernel_size: tuple[int, int] = (2, 2)
    merge_type: str = "sd2_tpool"
    mm_hidden_size: int = 1024
    text_hidden_size: int = 7168
    mm_projector_type: str = "patchmergerv2"
    projector_hidden_act: str = "gelu"
    projector_ln_eps: float = 1e-5
    tp_size: int = 1
    tp_rank: int = 0
    encoder_dp_size: int = 1
    encoder_dp_rank: int = 0

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> KimiK3VisionConfig:
        if "vision_config" in config:
            raw = config["vision_config"]
        elif "mm_hidden_size" in config and "vt_hidden_size" not in config:
            # ModelArgs uses mm_* for vision and unprefixed fields for text.
            # Do not interpret the text hidden_size/layer count as ViT values.
            aliases = {
                "mm_hidden_size": "vt_hidden_size",
                "mm_intermediate_size": "vt_intermediate_size",
                "mm_num_attention_heads": "vt_num_attention_heads",
                "mm_num_hidden_layers": "vt_num_hidden_layers",
                "mm_num_channels": "in_chans",
                "mm_patch_size": "patch_size",
                "mm_init_pos_emb_height": "init_pos_emb_height",
                "mm_init_pos_emb_width": "init_pos_emb_width",
                "mm_init_pos_emb_time": "init_pos_emb_time",
                "mm_projection_dim": "text_hidden_size",
                "mm_projector_type": "mm_projector_type",
                "mm_projector_hidden_act": "projector_hidden_act",
                # The Kimi loader publishes the projector epsilon here; it
                # must not be applied to the encoder's dtype-dependent norms.
                "mm_layer_norm_eps": "projector_ln_eps",
            }
            raw = {
                field: config[key]
                for key, field in aliases.items()
                if key in config and config[key] not in (None, 0, "")
            }
            merge_size = config.get("mm_spatial_merge_size") or config.get("mm_km_merge_kernel_size")
            if merge_size:
                raw["merge_kernel_size"] = (merge_size, merge_size)
            raw.setdefault("text_hidden_size", config.get("hidden_size", 7168))
        else:
            raw = config
        if not isinstance(raw, dict):
            raise TypeError("Kimi K3 vision_config must be a dictionary")

        merge_kernel_size = raw.get("merge_kernel_size", (2, 2))
        if not isinstance(merge_kernel_size, Sequence) or isinstance(merge_kernel_size, str):
            raise TypeError("Kimi K3 merge_kernel_size must be a sequence")
        if len(merge_kernel_size) != 2:
            raise ValueError("Kimi K3 merge_kernel_size must contain two values")
        hidden_size = int(raw.get("vt_hidden_size", raw.get("hidden_size", 1024)))
        text_config = config.get("text_config", {})
        if not isinstance(text_config, dict):
            text_config = {}
        mm_hidden_size = raw.get("mm_hidden_size")
        # Replicate the complete vision tower on each text-TP rank. Distribute
        # whole media items across that existing TP group, then gather their
        # embeddings; vision layers themselves do not shard weights or heads.
        encoder_dp_size = int(config.get("tp_size", raw.get("tp_size", 1)))
        encoder_dp_rank = int(config.get("tp_rank", raw.get("tp_rank", 0)))
        return cls(
            patch_size=int(raw.get("patch_size", 14)),
            in_channels=int(raw.get("in_chans", raw.get("in_channels", 3))),
            hidden_size=hidden_size,
            intermediate_size=int(raw.get("vt_intermediate_size", raw.get("intermediate_size", 4096))),
            num_attention_heads=int(
                raw.get(
                    "vt_num_attention_heads",
                    raw.get("num_attention_heads", 12),
                )
            ),
            num_hidden_layers=int(raw.get("vt_num_hidden_layers", raw.get("num_hidden_layers", 27))),
            qkv_hidden_size=int(raw.get("qkv_hidden_size", 1536)),
            init_pos_emb_height=int(raw.get("init_pos_emb_height", 64)),
            init_pos_emb_width=int(raw.get("init_pos_emb_width", 64)),
            init_pos_emb_time=int(raw.get("init_pos_emb_time", 4)),
            patch_embed_proj_bias=bool(raw.get("patch_embed_proj_bias", False)),
            attn_bias=bool(raw.get("attn_bias", False)),
            linear_bias=bool(raw.get("linear_bias", False)),
            activation_func=str(raw.get("activation_func", "gelu_pytorch_tanh")),
            norm_type=str(raw.get("norm_type", "rmsnorm")),
            mlp_type=str(raw.get("mlp_type", "mlp2")),
            pos_emb_type=str(raw.get("pos_emb_type", "divided_fixed")),
            merge_kernel_size=(
                int(merge_kernel_size[0]),
                int(merge_kernel_size[1]),
            ),
            merge_type=str(raw.get("merge_type", "sd2_tpool")),
            mm_hidden_size=(hidden_size if mm_hidden_size is None else int(mm_hidden_size)),
            text_hidden_size=int(raw.get("text_hidden_size", text_config.get("hidden_size", 7168))),
            mm_projector_type=str(raw.get("mm_projector_type", "patchmergerv2")),
            projector_hidden_act=str(raw.get("projector_hidden_act", "gelu")),
            projector_ln_eps=float(raw.get("projector_ln_eps", 1e-5)),
            tp_size=1,
            tp_rank=0,
            encoder_dp_size=encoder_dp_size,
            encoder_dp_rank=encoder_dp_rank,
        )

    def validate(self) -> None:
        if self.patch_size <= 0 or self.in_channels <= 0:
            raise ValueError("Kimi K3 patch size and input channels must be positive")
        if self.hidden_size <= 0 or self.hidden_size % 2 != 0:
            raise ValueError("Kimi K3 vision hidden size must be positive and even")
        if self.intermediate_size <= 0 or self.qkv_hidden_size <= 0:
            raise ValueError("Kimi K3 vision projection sizes must be positive")
        if self.num_hidden_layers <= 0:
            raise ValueError("Kimi K3 vision layer count must be positive")
        if (
            min(
                self.init_pos_emb_height,
                self.init_pos_emb_width,
                self.init_pos_emb_time,
            )
            <= 0
        ):
            raise ValueError("Kimi K3 initial position dimensions must be positive")
        if self.tp_size <= 0 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("Kimi K3 TP rank and size are invalid")
        if self.encoder_dp_size <= 0 or not 0 <= self.encoder_dp_rank < self.encoder_dp_size:
            raise ValueError("Kimi K3 encoder DP rank and size are invalid")
        if self.num_attention_heads <= 0:
            raise ValueError("Kimi K3 vision head count must be positive")
        if self.qkv_hidden_size % self.num_attention_heads != 0:
            raise ValueError("Kimi K3 qkv_hidden_size must be divisible by heads")
        if self.qkv_hidden_size // self.num_attention_heads % 4 != 0:
            raise ValueError("Kimi K3 vision head dimension must be divisible by four")
        if self.num_attention_heads % self.tp_size != 0:
            raise ValueError("Kimi K3 vision heads must be divisible by tp_size")
        if self.intermediate_size % self.tp_size != 0:
            raise ValueError("Kimi K3 intermediate_size must be divisible by tp_size")
        if self.activation_func != "gelu_pytorch_tanh":
            raise ValueError(f"Unsupported Kimi K3 activation: {self.activation_func}")
        if self.norm_type != "rmsnorm":
            raise ValueError(f"Unsupported Kimi K3 norm: {self.norm_type}")
        if self.mlp_type != "mlp2":
            raise ValueError(f"Unsupported Kimi K3 MLP: {self.mlp_type}")
        if self.pos_emb_type != "divided_fixed":
            raise ValueError(f"Unsupported Kimi K3 position embedding: {self.pos_emb_type}")
        if self.merge_type != "sd2_tpool":
            raise ValueError(f"Unsupported Kimi K3 merge type: {self.merge_type}")
        if self.mm_projector_type != "patchmergerv2":
            raise ValueError(f"Unsupported Kimi K3 projector: {self.mm_projector_type}")
        if self.projector_hidden_act != "gelu":
            raise ValueError(f"Unsupported Kimi K3 projector activation: {self.projector_hidden_act}")
        if any(size <= 0 for size in self.merge_kernel_size):
            raise ValueError("Kimi K3 merge dimensions must be positive")
        if self.mm_hidden_size != self.hidden_size:
            raise ValueError("Kimi K3 projector input must match vision hidden size")
        if self.text_hidden_size <= 0 or self.projector_ln_eps <= 0:
            raise ValueError("Kimi K3 projector dimensions and epsilon must be positive")


class KimiK3VisionPosEmbDivided(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.height = config.init_pos_emb_height
        self.width = config.init_pos_emb_width
        self.num_frames = config.init_pos_emb_time
        self.dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.height, self.width, self.dim, dtype=dtype, device=device))
        nn.init.normal_(self.weight)

        float_options = {"dtype": torch.float32, "device": device}
        omega = torch.arange(self.dim // 2, **float_options)
        omega = 1.0 / 10000.0 ** (omega / (self.dim / 2.0))
        positions = torch.arange(self.num_frames, **float_options)
        phases = torch.einsum("m,d->md", positions, omega)
        time_weight = torch.cat([phases.sin(), phases.cos()], dim=1)
        self.register_buffer(
            "time_weight",
            time_weight.unsqueeze(1).to(dtype=dtype),
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thws: torch.Tensor | Sequence[Sequence[int]],
    ) -> torch.Tensor:
        position_embeddings: list[torch.Tensor] = []
        for time, height, width in _grid_thw_list(grid_thws):
            if time > self.num_frames:
                raise ValueError(f"Kimi K3 grid time {time} exceeds {self.num_frames}")
            if height == self.height and width == self.width:
                spatial = self.weight.flatten(0, 1)
            else:
                # [H0, W0, D] -> [1, D, H0, W0] for spatial interpolation,
                # then [H, W, D] -> [H*W, D] in patch row-major order.
                spatial = (
                    F.interpolate(
                        self.weight.permute(2, 0, 1).unsqueeze(0),
                        size=(height, width),
                        mode="bicubic",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .flatten(0, 1)
                )
            if time > 1:
                # [H*W, D] -> [T, H*W, D]; [T, 1, D] broadcasts the
                # temporal encoding over patches. Single images retain only
                # spatial positions, matching the checkpoint's convention.
                spatial = spatial.unsqueeze(0).repeat(time, 1, 1)
                spatial = spatial + self.time_weight[:time]
            # Flatten frames before patches to match packed pixel/grid order.
            position_embeddings.append(spatial.reshape(-1, self.dim))

        if not position_embeddings:
            return hidden_states
        positions = torch.cat(position_embeddings, dim=0).to(hidden_states)
        if positions.shape != hidden_states.shape:
            raise ValueError(f"Kimi K3 positions {positions.shape} do not match patches {hidden_states.shape}")
        return hidden_states + positions


class KimiK3VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=config.patch_embed_proj_bias,
            dtype=dtype,
            device=device,
        )
        self.pos_emb = KimiK3VisionPosEmbDivided(config, dtype, device)

    def forward(
        self,
        pixels: torch.Tensor,
        grid_thws: torch.Tensor | Sequence[Sequence[int]],
    ) -> torch.Tensor:
        # The multimodal processor already emits individual [C, P, P] patches.
        # Accepting whole images here would also change the supplied grid order.
        expected_shape = (self.proj.in_channels, self.proj.kernel_size[0], self.proj.kernel_size[1])
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != expected_shape:
            raise ValueError(f"Kimi K3 pixels must be pre-patched [tokens, {expected_shape}], got {pixels.shape}")
        hidden_states = self.proj(pixels).flatten(1)
        return self.pos_emb(hidden_states, grid_thws)


class KimiK3VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_height: int | None = None,
        max_width: int | None = None,
    ) -> None:
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("Kimi K3 vision head dimension must be divisible by four")
        if any(limit is not None and limit <= 0 for limit in (max_height, max_width)):
            raise ValueError("Kimi K3 RoPE spatial limits must be positive when specified")
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self.register_buffer("_frequencies", None, persistent=False)

    def _build_frequencies(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(
            height * width,
            dtype=torch.float32,
            device=device,
        )
        x_positions = positions.remainder(width)
        y_positions = positions.floor_divide(width)
        dim_range = torch.arange(0, self.dim, 4, dtype=torch.float32, device=device)
        frequencies = 1.0 / 10000.0 ** (dim_range / self.dim)
        x_phases = torch.outer(x_positions, frequencies)
        y_phases = torch.outer(y_positions, frequencies)
        x_complex = torch.polar(torch.ones_like(x_phases), x_phases)
        y_complex = torch.polar(torch.ones_like(y_phases), y_phases)
        return torch.stack([x_complex, y_complex], dim=-1).reshape(
            height,
            width,
            self.dim // 2,
        )

    def forward(
        self,
        grid_thws: torch.Tensor | Sequence[Sequence[int]],
        device: torch.device,
    ) -> torch.Tensor:
        grids = _grid_thw_list(grid_thws)
        if not grids:
            return torch.empty(0, self.dim // 2, dtype=torch.complex64, device=device)
        for _, height, width in grids:
            if (self.max_height is not None and height > self.max_height) or (
                self.max_width is not None and width > self.max_width
            ):
                raise ValueError(
                    f"Kimi K3 grid {(height, width)} exceeds RoPE cache {(self.max_height, self.max_width)}"
                )
        height = max(grid[1] for grid in grids)
        width = max(grid[2] for grid in grids)
        if self._frequencies is None or self._frequencies.device != device:
            self._frequencies = self._build_frequencies(height, width, device)
        elif height > self._frequencies.shape[0] or width > self._frequencies.shape[1]:
            self._frequencies = self._build_frequencies(
                max(height, self._frequencies.shape[0]),
                max(width, self._frequencies.shape[1]),
                device,
            )

        outputs: list[torch.Tensor] = []
        for time, height, width in grids:
            frequencies = self._frequencies[:height, :width].reshape(-1, self.dim // 2)
            outputs.append(frequencies.repeat(time, 1))
        return torch.cat(outputs, dim=0)


class KimiK3VisionMLP(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        intermediate_per_rank = config.intermediate_size // config.tp_size
        self.fc0 = ColumnParallelLinear(
            config.hidden_size,
            intermediate_per_rank,
            config.tp_size,
            bias=config.linear_bias,
            dtype=dtype,
            device=device,
        )
        self.fc1 = RowParallelLinear(
            intermediate_per_rank,
            config.hidden_size,
            config.tp_size,
            bias=config.linear_bias,
            dtype=dtype,
            device=device,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.gelu(self.fc0(hidden_states), approximate="tanh")
        return self.fc1(hidden_states)


class KimiK3VisionEncoderLayer(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads // config.tp_size
        self.head_dim = config.qkv_hidden_size // config.num_attention_heads
        qkv_hidden_per_rank = self.num_heads * self.head_dim
        self.norm0 = nn.RMSNorm(
            config.hidden_size,
            # The released checkpoint uses dtype-dependent encoder epsilon;
            # projector_ln_eps is a separate projector-only contract.
            eps=None,
            dtype=dtype,
            device=device,
        )
        self.norm1 = nn.RMSNorm(
            config.hidden_size,
            eps=None,
            dtype=dtype,
            device=device,
        )
        self.mlp = KimiK3VisionMLP(config, dtype, device)
        self.wqkv = ColumnParallelLinear(
            config.hidden_size,
            3 * qkv_hidden_per_rank,
            config.tp_size,
            bias=config.attn_bias,
            dtype=dtype,
            device=device,
        )
        self.wo = RowParallelLinear(
            qkv_hidden_per_rank,
            config.hidden_size,
            config.tp_size,
            bias=config.attn_bias,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Sequence[int],
        frequencies: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        normalized = self.norm0(hidden_states)
        qkv = self.wqkv(normalized).reshape(
            hidden_states.shape[0],
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=1)
        query, key = _apply_rope(query, key, frequencies)
        attended = _encoder_attention(
            query,
            key,
            value,
            cu_seqlens,
        )
        hidden_states = residual + self.wo(attended.flatten(1))
        return hidden_states + self.mlp(self.norm1(hidden_states))


class KimiK3VisionEncoder(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        head_dim = config.qkv_hidden_size // config.num_attention_heads
        self.rope_2d = KimiK3VisionRotaryEmbedding(head_dim)
        self.blocks = nn.ModuleList(
            [KimiK3VisionEncoderLayer(config, dtype, device) for _ in range(config.num_hidden_layers)]
        )
        self.final_layernorm = nn.RMSNorm(
            config.hidden_size,
            eps=None,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thws: torch.Tensor | Sequence[Sequence[int]],
    ) -> torch.Tensor:
        grids = _grid_thw_list(grid_thws)
        sequence_lengths = [time * height * width for time, height, width in grids]
        cu_seqlens = [0]
        for sequence_length in sequence_lengths:
            cu_seqlens.append(cu_seqlens[-1] + sequence_length)
        frequencies = self.rope_2d(grids, hidden_states.device)
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, frequencies)
        return self.final_layernorm(hidden_states)


def _merged_rows_per_item(grids: Sequence[Sequence[int]], merge_kernel_size: tuple[int, int]) -> list[int]:
    merge_height, merge_width = merge_kernel_size
    rows: list[int] = []
    for _, height, width in grids:
        if height % merge_height != 0 or width % merge_width != 0:
            raise ValueError(f"Kimi K3 grid {(height, width)} is not divisible by merge kernel {merge_kernel_size}")
        # sd2_tpool removes the time axis; output rows depend only on space.
        rows.append((height // merge_height) * (width // merge_width))
    return rows


def tpool_patch_merger(
    hidden_states: torch.Tensor,
    grid_thws: torch.Tensor | Sequence[Sequence[int]],
    merge_kernel_size: tuple[int, int],
) -> list[torch.Tensor]:
    merge_height, merge_width = merge_kernel_size
    grids = _grid_thw_list(grid_thws)
    output_rows = _merged_rows_per_item(grids, merge_kernel_size)
    outputs: list[torch.Tensor] = []
    offset = 0
    for (time, height, width), rows in zip(grids, output_rows):
        length = time * height * width
        sequence = hidden_states[offset : offset + length].reshape(
            time,
            height // merge_height,
            merge_height,
            width // merge_width,
            merge_width,
            hidden_states.shape[-1],
        )
        sequence = sequence.mean(dim=0).permute(0, 2, 1, 3, 4)
        outputs.append(
            sequence.reshape(
                rows,
                merge_height * merge_width,
                hidden_states.shape[-1],
            )
        )
        offset += length
    if offset != hidden_states.shape[0]:
        raise ValueError(f"Kimi K3 grids cover {offset} tokens, got {hidden_states.shape[0]}")
    return outputs


class KimiK3VisionTower(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_embed = KimiK3VisionPatchEmbed(config, dtype, device)
        self.encoder = KimiK3VisionEncoder(config, dtype, device)

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thws: torch.Tensor | Sequence[Sequence[int]],
    ) -> list[torch.Tensor]:
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws)
        return tpool_patch_merger(
            hidden_states,
            grid_thws,
            self.merge_kernel_size,
        )


def _get_load_balance_assignment(
    sizes: list[int],
    num_ranks: int,
) -> tuple[list[int], list[int], list[int]]:
    assignments: list[list[int]] = [[] for _ in range(num_ranks)]
    loads = [0] * num_ranks
    for image_index in sorted(
        range(len(sizes)),
        key=lambda index: sizes[index],
        reverse=True,
    ):
        # Largest media first, assigned to the least-loaded rank by patch count.
        target_rank = min(range(num_ranks), key=lambda rank: loads[rank])
        assignments[target_rank].append(image_index)
        loads[target_rank] += sizes[image_index]

    # This is a rank-major permutation of original media indices, not a map
    # from media index to rank. Each rank owns the next image_counts[rank] items.
    image_to_rank: list[int] = []
    image_counts: list[int] = []
    for rank_assignment in assignments:
        image_to_rank.extend(rank_assignment)
        image_counts.append(len(rank_assignment))
    return image_to_rank, image_counts, loads


def _run_data_parallel_vision_tower(
    vision_tower: KimiK3VisionTower,
    pixel_values: torch.Tensor,
    grid_thws: torch.Tensor | Sequence[Sequence[int]],
    rank: int,
    world_size: int,
) -> list[torch.Tensor]:
    grids = _grid_thw_list(grid_thws)
    if not grids:
        return []
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("Kimi K3 encoder DP rank and size are invalid")
    if world_size == 1:
        return vision_tower(pixel_values, grids)

    patches_per_image = [math.prod(grid) for grid in grids]
    output_lengths = _merged_rows_per_item(grids, vision_tower.merge_kernel_size)
    # Input patches are media-major. Prefix sums locate each original item's
    # patch slice; assignment prefix sums locate this rank's media indices.
    cumulative_patches = [0, *itertools.accumulate(patches_per_image)]
    image_to_rank, image_counts, _ = _get_load_balance_assignment(patches_per_image, world_size)
    cumulative_image_counts = [0, *itertools.accumulate(image_counts)]
    rows_per_rank = [
        sum(
            output_lengths[index]
            for index in image_to_rank[cumulative_image_counts[r] : cumulative_image_counts[r + 1]]
        )
        for r in range(world_size)
    ]
    local_image_indices = image_to_rank[cumulative_image_counts[rank] : cumulative_image_counts[rank + 1]]
    local_grids = [grids[index] for index in local_image_indices]

    if local_image_indices:
        local_pixel_values = torch.cat(
            [pixel_values[cumulative_patches[index] : cumulative_patches[index + 1]] for index in local_image_indices],
            dim=0,
        )
    else:
        local_pixel_values = pixel_values.new_empty((0, *pixel_values.shape[1:]))

    merge_size = math.prod(vision_tower.merge_kernel_size)
    if local_image_indices:
        local_outputs = vision_tower(local_pixel_values, local_grids)
        local_embeddings = torch.cat(local_outputs, dim=0)
    else:
        local_embeddings = pixel_values.new_empty((0, merge_size, vision_tower.config.hidden_size))
    if local_embeddings.shape[0] != rows_per_rank[rank]:
        raise ValueError("Kimi K3 encoder output rows do not match temporally pooled grids")
    max_embeddings_per_rank = max(rows_per_rank)
    # AllGather needs equal physical rows. Pad to the busiest rank's merged
    # row count, including ranks assigned no media, so every rank participates.
    padding_size = max_embeddings_per_rank - local_embeddings.shape[0]
    if padding_size < 0:
        raise ValueError("Kimi K3 encoder output exceeds its AllGather bucket")
    if padding_size > 0:
        padding = local_embeddings.new_empty(
            (
                padding_size,
                local_embeddings.shape[1],
                local_embeddings.shape[2],
            )
        )
        local_embeddings = torch.cat([local_embeddings, padding], dim=0)

    # Encoder data parallelism uses the text TP ranks and their mainline group.
    gathered = distributed.tp_all_gather(
        local_embeddings.contiguous(),
        dim=0,
        world_size=world_size,
    )
    rank_embeddings: list[torch.Tensor] = []
    # Gather concatenates equal buckets in rank order; discard each bucket's
    # padding using its real pooled row count before splitting individual media.
    for source_rank in range(world_size):
        start = source_rank * max_embeddings_per_rank
        length = rows_per_rank[source_rank]
        rank_embeddings.append(gathered[start : start + length])

    outputs: list[torch.Tensor | None] = [None] * len(grids)
    assignment_offset = 0
    # Undo the rank-major assignment: split each rank's packed embeddings by
    # media length and write them back at their original request/media index.
    for source_rank in range(world_size):
        rank_image_indices = image_to_rank[assignment_offset : assignment_offset + image_counts[source_rank]]
        embedding_offset = 0
        for image_index in rank_image_indices:
            output_length = output_lengths[image_index]
            outputs[image_index] = rank_embeddings[source_rank][embedding_offset : embedding_offset + output_length]
            embedding_offset += output_length
        assignment_offset += image_counts[source_rank]

    if any(output is None for output in outputs):
        raise RuntimeError("Kimi K3 encoder DP left an image output unassigned")
    return [output for output in outputs if output is not None]


class KimiK3MultiModalProjector(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        merge_size = config.merge_kernel_size[0] * config.merge_kernel_size[1]
        self.input_size = config.mm_hidden_size * merge_size
        self.linear_1 = nn.Linear(
            self.input_size,
            self.input_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.linear_2 = nn.Linear(
            self.input_size,
            config.text_hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.post_norm = nn.RMSNorm(
            config.text_hidden_size,
            eps=config.projector_ln_eps,
            dtype=dtype,
            device=device,
        )
        self.rot_proj: nn.Linear | None = None

    def add_rot_proj(self) -> None:
        if self.rot_proj is not None:
            return
        reference = self.post_norm.weight
        self.rot_proj = nn.Linear(
            reference.shape[0],
            reference.shape[0],
            bias=False,
            dtype=reference.dtype,
            device=reference.device,
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = image_features.reshape(-1, self.input_size)
        hidden_states = F.gelu(self.linear_1(hidden_states))
        hidden_states = self.post_norm(self.linear_2(hidden_states))
        if self.rot_proj is not None:
            hidden_states = self.rot_proj(hidden_states)
        return hidden_states


class KimiK3VisionModel(nn.Module):
    """Standalone vision path, ready to be composed with the K3 language model."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = KimiK3VisionConfig.from_dict(config)
        self.config.validate()
        dtype = PyModelBase.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "cuda"))
        self.vision_tower = KimiK3VisionTower(self.config, dtype, device)
        self.mm_projector = KimiK3MultiModalProjector(self.config, dtype, device)

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thws: torch.Tensor | Sequence[Sequence[int]],
    ) -> list[torch.Tensor]:
        reference = next(self.parameters())
        # Convert device grids to host once; submodules receive host tuples.
        grids = _grid_thw_list(grid_thws)
        pixel_values = pixel_values.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        tower_outputs = _run_data_parallel_vision_tower(
            self.vision_tower,
            pixel_values,
            grids,
            self.config.encoder_dp_rank,
            self.config.encoder_dp_size,
        )
        if not tower_outputs:
            return []
        lengths = [output.shape[0] for output in tower_outputs]
        gathered_tower_output = torch.cat(tower_outputs, dim=0)
        projected = self.mm_projector(gathered_tower_output)
        outputs = list(projected.split(lengths, dim=0))
        return outputs

    def load_weights(
        self,
        state_dicts: list[Any],
        tp_rank: int,
        tp_size: int,
    ) -> set[str]:
        rank_context = (
            f"loader TP {tp_rank}/{tp_size}, vision TP {self.config.tp_rank}/{self.config.tp_size}, "
            f"encoder DP {self.config.encoder_dp_rank}/{self.config.encoder_dp_size}"
        )
        if tp_rank != self.config.encoder_dp_rank or tp_size != self.config.encoder_dp_size:
            raise ValueError(f"Kimi K3 loader rank/size must match encoder DP construction ({rank_context})")
        weights = ScopedWeightLoader(
            state_dicts,
            src_prefixes=("", "model."),
            name_aliases={
                "mm_projector.linear_1.weight": ("mm_projector.linear_1.weight", "mm_projector.proj.0.weight"),
                "mm_projector.linear_2.weight": ("mm_projector.linear_2.weight", "mm_projector.proj.2.weight"),
            },
        )
        if weights.has("mm_projector.rot_proj.weight"):
            self.mm_projector.add_rot_proj()

        loaded: set[str] = set()
        for name, parameter in self.named_parameters():
            try:
                weights.load_tensor(parameter, name)
            except (KeyError, ValueError) as error:
                # Keep the shared loader's failure and identify the replica
                # whose checkpoint loading failed; do not continue loading.
                raise type(error)(f"{error} ({rank_context})") from error
            loaded.add(name)
        return loaded


__all__ = [
    "KimiK3MultiModalProjector",
    "KimiK3VisionConfig",
    "KimiK3VisionEncoderLayer",
    "KimiK3VisionModel",
    "KimiK3VisionTower",
    "tpool_patch_merger",
]
