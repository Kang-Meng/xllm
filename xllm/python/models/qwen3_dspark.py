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

"""Qwen3 DSpark draft model for the Python NPU executor."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from xllm.python.model_executor.forward_context import LayerSynchronizer
from xllm.python.model_loader import ParallelLoadContext, load_draft_lm_head_if_present
from xllm.python.models.dspark import DSparkForCausalLMBase
from xllm.python.models.qwen3_dflash import (
    DFlashQwen3Config,
    DFlashQwen3Model,
)


@dataclass
class Qwen3DSparkConfig(DFlashQwen3Config):
    markov_rank: int = 0
    enable_confidence_head: bool = False
    confidence_head_with_markov: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> Qwen3DSparkConfig:
        base = DFlashQwen3Config.from_dict(d)
        return cls(
            **base.__dict__,
            markov_rank=int(d.get("markov_rank", 0)),
            enable_confidence_head=bool(d.get("enable_confidence_head", False)),
            confidence_head_with_markov=bool(d.get("confidence_head_with_markov", False)),
        )

    def validate(self) -> None:
        super().validate()
        if self.markov_rank <= 0:
            raise ValueError("Qwen3 DSpark requires markov_rank > 0")
        if self.use_sliding_window:
            raise ValueError("sliding-window DSpark drafts are not supported yet")

    def _allows_reduced_vocab(self) -> bool:
        return True


class Qwen3DSparkModel(DFlashQwen3Model):
    """DFlash draft backbone used with DSpark-specific output heads."""

class Qwen3DSparkForCausalLM(DSparkForCausalLMBase):

    def __init__(self, config: dict) -> None:
        cfg = Qwen3DSparkConfig.from_dict(config)
        cfg.validate()
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "npu"))
        super().__init__(
            backbone=Qwen3DSparkModel(cfg, dtype, device),
            lm_head=None,
            query_width=int(config["draft_query_width"]),
            vocab_size=cfg.vocab_size,
            draft_vocab_size=cfg.draft_vocab_size,
            markov_rank=cfg.markov_rank,
            hidden_size=cfg.hidden_size,
            enable_confidence_head=cfg.enable_confidence_head,
            confidence_head_with_markov=cfg.confidence_head_with_markov,
            dtype=dtype,
            device=device,
        )
        self.cfg = cfg
        self.dtype = dtype
        self.device = device

    def supports_device_only_decode_inputs(self, kv_page_size: int) -> bool:
        import xllm_runtime

        return (
            type(self) is Qwen3DSparkForCausalLM
            and not self.cfg.use_sliding_window
            and self.cfg.n_heads % self.cfg.tp_size == 0
            and self.cfg.n_kv_heads % self.cfg.tp_size == 0
            and self.cfg.head_dim == 128
            and kv_page_size == 128
            and xllm_runtime.is_ascend_a3()
        )

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        pipeline = self.model
        all_weights = pipeline.backbone.load_weights(state_dicts, tp_rank, tp_size)
        if all_weights.has("d2t"):
            offsets = all_weights.get_tensor("d2t").to(device=self.device, dtype=torch.long)
            pipeline.hot_token_ids = offsets + torch.arange(offsets.numel(), device=self.device)
        load_draft_lm_head_if_present(
            pipeline,
            all_weights,
            context=ParallelLoadContext(tp_rank, tp_size),
        )
        all_weights.load_tensor(pipeline.markov_head.markov_w1.weight, "markov_head.markov_w1.weight")
        all_weights.load_tensor(pipeline.markov_head.markov_w2.weight, "markov_head.markov_w2.weight")
        if pipeline.confidence_head is not None:
            all_weights.load_tensor(pipeline.confidence_head.proj.weight, "confidence_head.proj.weight")
            all_weights.load_tensor(pipeline.confidence_head.proj.bias, "confidence_head.proj.bias")

    def adapt_weights_for_reference_model(
        self,
        reference_model_path: str,
    ) -> None:
        self.model.backbone.adapt_weights_for_reference_model(reference_model_path)

    def write_context_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_slots: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor | None, ...]],
        layer_synchronizer: LayerSynchronizer | None,
    ) -> torch.Tensor | None:
        return self.model.backbone.write_context_kv(
            target_hidden,
            positions,
            cache_slots,
            kv_caches,
            layer_synchronizer,
        )
