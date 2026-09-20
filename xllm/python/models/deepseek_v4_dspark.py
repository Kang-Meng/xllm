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

"""DeepSeek-V4 DSpark draft model for the Python NPU executor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from xllm.python.attention.backend import LayerCacheInput, normalize_layer_caches
from xllm.python.layers.layernorm import RMSNorm
from xllm.python.model_executor.forward_context import LayerSynchronizer
from xllm.python.models.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4Model,
    _expand_half_rope_cos_sin,
)


@dataclass
class DeepseekV4DSparkConfig(DeepseekV4Config):
    """DeepSeek-V4 target configuration narrowed to its DSpark draft."""

    dspark_num_layers: int = 0
    dspark_target_layer_ids: tuple[int, ...] = ()
    dspark_block_size: int = 0
    dspark_noise_token_id: int = 0
    markov_rank: int = 0
    enable_confidence_head: bool = False
    confidence_head_with_markov: bool = True
    dspark_use_native_sas: bool = False

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> DeepseekV4DSparkConfig:
        base = DeepseekV4Config.from_dict(config)
        target_layer_ids = tuple(int(layer_id) for layer_id in config.get("dspark_target_layer_ids", ()))
        num_layers = int(config.get("dspark_num_layers", len(target_layer_ids)))
        values = dict(base.__dict__)
        values.update(
            model_type="deepseek_v4_dspark",
            n_layers=num_layers,
            n_hash_layers=0,
            compress_ratios=[1] * num_layers,
            dspark_num_layers=num_layers,
            dspark_target_layer_ids=target_layer_ids,
            dspark_block_size=int(config.get("dspark_block_size", 0)),
            dspark_noise_token_id=int(config.get("dspark_noise_token_id", 0)),
            markov_rank=int(config.get("markov_rank", config.get("dspark_markov_rank", 0))),
            enable_confidence_head=bool(config.get("enable_confidence_head", False)),
            confidence_head_with_markov=bool(config.get("confidence_head_with_markov", True)),
            dspark_use_native_sas=bool(config.get("dspark_use_native_sas", False)),
        )
        return cls(**values)

    def validate(self) -> None:
        if self.dspark_num_layers <= 0:
            raise ValueError("DeepSeek-V4 DSpark requires dspark_num_layers > 0")
        if self.dspark_target_layer_ids and len(self.dspark_target_layer_ids) != self.dspark_num_layers:
            raise ValueError("DeepSeek-V4 DSpark target layer count must match dspark_num_layers")
        if self.markov_rank <= 0:
            raise ValueError("DeepSeek-V4 DSpark requires dspark_markov_rank > 0")


class DeepseekV4DSparkModel(DeepseekV4Model):
    """DeepSeek-V4 decoder layers with shared target-context projection."""

    def __init__(
        self,
        cfg: DeepseekV4DSparkConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(cfg, dtype, device)
        self.cfg = cfg
        self.main_proj = nn.Linear(
            cfg.hidden_size * cfg.dspark_num_layers,
            cfg.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.main_norm = RMSNorm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            dtype=dtype,
            device=device,
        )

    def write_context_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_slots: torch.Tensor,
        kv_caches: list[LayerCacheInput],
        layer_synchronizer: LayerSynchronizer | None,
    ) -> torch.Tensor | None:
        if len(kv_caches) != self.cfg.n_layers:
            raise ValueError("DeepSeek-V4 DSpark cache/layer count mismatch")
        if cache_slots.numel() != target_hidden.size(0):
            raise ValueError("DeepSeek-V4 DSpark context slot count mismatch")
        expected_width = self.cfg.hidden_size * self.cfg.dspark_num_layers
        if target_hidden.dim() != 2 or target_hidden.size(-1) != expected_width:
            raise ValueError(
                f"DeepSeek-V4 DSpark target hidden width must equal dspark_num_layers * hidden_size ({expected_width})"
            )

        projected = self.main_norm(self.main_proj(target_hidden))
        positions = positions.to(torch.int64).contiguous()
        cos_sin = self.rotary.cos_sin_cache.index_select(0, positions)
        cos, sin = _expand_half_rope_cos_sin(cos_sin)
        layer_caches = normalize_layer_caches(kv_caches)
        for layer_id, (layer, layer_cache) in enumerate(zip(self.layers, layer_caches, strict=True)):
            if layer_cache.swa is None:
                raise RuntimeError(f"DeepSeek-V4 DSpark SWA cache is missing for layer {layer_id}")
            layer.write_context_kv(
                projected,
                cos,
                sin,
                cache_slots,
                layer_cache.swa,
            )
            if layer_synchronizer is not None and not layer_synchronizer.record_event(layer_id):
                return None
        return projected
