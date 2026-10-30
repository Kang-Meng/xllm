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

"""Qwen3.5 MTP draft computation; scheduling and verification remain in C++."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python.layers.linear import ColumnParallelLinear
from xllm.python.layers.qwen3_5.factory import get_qwen3_5_decoder_layer_class
from xllm.python.layers.sequence_parallel import sp_all_gather, sp_shard
from xllm.python.layers.sequence_parallel_metadata import SpMetadata
from xllm.python.model_executor.forward_context import get_execution_context, record_layer_event
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader, StateDictLike, load_lm_head
from xllm.python.models.base import PyModelBase
from xllm.python.models.qwen3_5 import Qwen3_5Config, Qwen3_5Model


class _MtpStateDict:
    """Expose the checkpoint's individual MTP experts in the shared MoE layout."""

    def __init__(self, state_dicts: list[StateDictLike], num_experts: int) -> None:
        self._weights = ScopedWeightLoader(state_dicts)
        self._num_experts = num_experts

    def has(self, name: str) -> bool:
        if self._weights.has(name):
            return True
        if name.endswith(".mlp.experts.gate_up_proj"):
            root = name.removesuffix("gate_up_proj")
            return self._weights.has(root + "0.gate_proj.weight") and self._weights.has(root + "0.up_proj.weight")
        if name.endswith(".mlp.experts.down_proj"):
            return self._weights.has(name.removesuffix("down_proj") + "0.down_proj.weight")
        return False

    def get_tensor(self, name: str) -> torch.Tensor:
        if self._weights.has(name):
            return self._weights.get_tensor(name)
        if name.endswith(".mlp.experts.gate_up_proj"):
            root = name.removesuffix("gate_up_proj")
            return torch.stack(
                [
                    torch.cat(
                        (
                            self._weights.get_tensor(f"{root}{expert}.gate_proj.weight"),
                            self._weights.get_tensor(f"{root}{expert}.up_proj.weight"),
                        ),
                        dim=0,
                    )
                    for expert in range(self._num_experts)
                ]
            )
        if name.endswith(".mlp.experts.down_proj"):
            root = name.removesuffix("down_proj")
            return torch.stack(
                [self._weights.get_tensor(f"{root}{expert}.down_proj.weight") for expert in range(self._num_experts)]
            )
        raise KeyError(f"MTP checkpoint tensor not found: {name}")


class Qwen3_5MtpModel(Qwen3_5Model):
    def __init__(self, cfg: Qwen3_5Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__(cfg, dtype, device)
        normalization_cls = get_qwen3_5_decoder_layer_class(device).normalization_cls
        self.pre_fc_norm_embedding = normalization_cls(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.pre_fc_norm_hidden = normalization_cls(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.fc = nn.Linear(cfg.hidden_size * 2, cfg.hidden_size, bias=False, dtype=dtype, device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._comm_policy is not None:
            self.layers[0].validate_ep_level2(self._comm_policy, force_mc2=self.cfg.eplv2_comm_mode == "mc2")
        sp_layout: SpMetadata | None = None
        if self.cfg.sequence_parallel_enabled:
            if input_ids.ndim != 1 or positions.shape != input_ids.shape:
                raise ValueError("Qwen3.5 MTP SP requires flattened, token-aligned positions")
            sp_layout = get_execution_context(SpMetadata)
            if sp_layout is None:
                raise RuntimeError("Qwen3.5 MTP SP layout metadata is unavailable")
            self.layers[0].validate_sp(sp_layout, device=self.embed_tokens.weight.device)

        embedding = self.embed_tokens(input_ids)
        hidden = embedding if input_embedding is None else input_embedding
        if hidden.shape != embedding.shape:
            raise ValueError("Qwen3.5 MTP carried hidden must match token embeddings")
        normalized_embedding = self.pre_fc_norm_embedding(embedding)
        normalized_hidden = self.pre_fc_norm_hidden(hidden)
        hidden = self.fc(torch.cat((normalized_embedding, normalized_hidden), dim=-1))
        if sp_layout is not None:
            hidden = sp_shard(hidden, sp_layout)
        residual: torch.Tensor | None = None
        for layer_id, layer in enumerate(self.layers):
            hidden, residual = layer(hidden, residual, positions)
            record_layer_event(layer_id)
        hidden, _ = self.norm(hidden, residual)
        if sp_layout is not None:
            hidden = sp_all_gather(hidden, sp_layout)
        return hidden


class Qwen3_5MtpForCausalLM(PyModelBase):
    def __init__(self, config: dict) -> None:
        super().__init__()
        if int(config.get("cp_size", 1)) != 1:
            raise NotImplementedError("Qwen3.5 MTP requires CP=1")
        if bool(config.get("enable_mega_moe", False)):
            raise NotImplementedError("Qwen3.5 MTP does not support MegaMoe")
        self.dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        self.device = torch.device(config.get("device", "npu"))
        self.cfg = Qwen3_5Config.prepare_execution_config(
            config, self.device, allow_speculative_decoding=True, allow_draft_engine=True
        )
        text_config = config.get("text_config", config)
        self.cfg.n_layers = max(
            int(config.get("num_nextn_predict_layers", text_config.get("mtp_num_hidden_layers", 1))), 1
        )
        self.cfg.layer_types = ["full_attention"] * self.cfg.n_layers
        self.cfg.layers_to_capture = ()
        self.model = Qwen3_5MtpModel(self.cfg, self.dtype, self.device)
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // self.cfg.tp_size,
            self.cfg.tp_size,
            gather_output=True,
            dtype=self.dtype,
            device=self.device,
        )

    def load_weights(self, state_dicts: list[StateDictLike], tp_rank: int, tp_size: int) -> None:
        context = ParallelLoadContext.from_config(self.cfg, tp_rank, tp_size)
        shared = ScopedWeightLoader(state_dicts, src_prefixes=("model.language_model.", "model.", ""))
        backbone = shared.bind_source_root("embed_tokens.weight")
        backbone.load_tensor(
            self.model.embed_tokens.weight, "embed_tokens.weight", dim=1, rank=tp_rank, world_size=tp_size
        )
        load_lm_head(
            self.lm_head.weight,
            backbone,
            shared,
            context,
            tie_word_embeddings=self.cfg.tie_word_embeddings,
            embed_fallback=True,
        )

        mtp = ScopedWeightLoader(
            [_MtpStateDict(state_dicts, self.cfg.num_experts)], src_prefixes=("mtp.", "model.mtp.")
        ).bind_source_root("fc.weight")
        mtp.load_tensor(self.model.fc.weight, "fc.weight")
        mtp.load_tensor(self.model.pre_fc_norm_embedding.weight, "pre_fc_norm_embedding.weight")
        mtp.load_tensor(self.model.pre_fc_norm_hidden.weight, "pre_fc_norm_hidden.weight")
        mtp.load_tensor(self.model.norm.weight, "norm.weight")
        for layer_id, layer in enumerate(self.model.layers):
            layer.load_weights(mtp.with_prefix(f"layers.{layer_id}."), context)
