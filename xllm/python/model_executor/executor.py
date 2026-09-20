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

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python import distributed
from xllm.python.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    LayerCacheInput,
    normalize_layer_caches,
)
from xllm.python.layers.attention import Attention
from xllm.python.layers.npu.mega_moe_context_provider import (
    create_token_owner_mega_moe_context_provider,
)
from xllm.python.model_executor.forward_context import EplbRuntimeState, LayerSynchronizer
from xllm.python.model_executor.runners.base import ModelExecutionOutput
from xllm.python.model_executor.runners.eager import EagerRunner
from xllm.python.platform import current_platform


def _is_deepseek_v4_model_type(model_type: str) -> bool:
    return model_type.startswith("deepseek_v4")


def _resolve_graph_backend(config: dict) -> str:
    graph_backend = str(config.get("python_graph_backend", "off")).lower()
    graph_disabled = graph_backend in ("", "off", "none", "0")
    if graph_disabled and config.get("enable_graph", False):
        if current_platform.is_npu():
            return "aclgraph"
    return graph_backend


def _create_attention_backend(
    first_attention: Attention,
    device: torch.device,
    dtype: torch.dtype,
    config: dict | None = None,
    max_num_reqs: int = 1,
    num_decoding_tokens: int = 1,
) -> AttentionBackend:
    config = config or {}
    model_type = config.get("model_type", "")
    if _is_deepseek_v4_model_type(model_type) and current_platform.is_npu():
        from xllm.python.attention.dsa_attention import DsaAttentionBackend

        return DsaAttentionBackend(
            compress_ratios=list(config.get("compress_ratios", [])),
            window_size=int(config.get("window_size", 128)),
            n_layers=int(config.get("n_layers", config.get("num_hidden_layers", 0))),
            num_heads=first_attention.num_heads,
            attn_head_dim=first_attention.head_dim,
            index_topk=int(config.get("index_topk", 512)),
            index_n_heads=int(config.get("index_n_heads", 64)),
            index_head_dim=int(config.get("index_head_dim", 128)),
            rope_head_dim=int(config.get("qk_rope_head_dim", 64)),
            device=device,
            dtype=dtype,
            dspark_block_size=int(config.get("dspark_block_size", 0)),
            dspark_use_native_sas=bool(config.get("dspark_use_native_sas", False)),
        )
    if current_platform.is_npu():
        dcp_group = distributed.dcp_group(device)
        if int(config.get("cp_size", 1)) == 1 and dcp_group is not None and dcp_group.size() > 1:
            from xllm.python.attention.sfa_dcp_backend import (
                SfaDcpAttentionBackend,
                dcp_layer_options,
            )

            index_topk = dcp_layer_options(first_attention)
            return SfaDcpAttentionBackend(
                num_heads=first_attention.num_heads,
                num_kv_heads=first_attention.num_kv_heads,
                head_dim=first_attention.head_dim,
                scale=first_attention.scale,
                sliding_window=first_attention.sliding_window,
                device=device,
                dtype=dtype,
                dcp_group=dcp_group,
                index_topk=index_topk,
                max_num_reqs=max(max_num_reqs, 1),
                num_decoding_tokens=max(num_decoding_tokens, 1),
            )
        from xllm.python.attention.npu_paged_attention import (
            NpuPagedAttentionBackend,
        )

        return NpuPagedAttentionBackend(
            num_heads=first_attention.num_heads,
            num_kv_heads=first_attention.num_kv_heads,
            head_dim=first_attention.head_dim,
            scale=first_attention.scale,
            sliding_window=first_attention.sliding_window,
            is_mla=bool(config.get("enable_mla", False)),
            device=device,
            dtype=dtype,
            num_decoding_tokens=num_decoding_tokens,
        )
    if current_platform.is_cuda():
        from xllm.python.attention.flashinfer import FlashInferBackend

        return FlashInferBackend(
            num_heads=first_attention.num_heads,
            num_kv_heads=first_attention.num_kv_heads,
            head_dim=first_attention.head_dim,
            scale=first_attention.scale,
            sliding_window=first_attention.sliding_window,
            device=device,
            dtype=dtype,
        )
    raise NotImplementedError(f"No attention backend available for device type '{device.type}'")


class ModelExecutor:
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        max_seqs_per_batch: int,
        num_decoding_tokens: int = 1,
        acl_graph_decode_batch_size_limit: int | None = None,
    ) -> None:
        self.model = model
        self._kv_bound = False
        self._requires_framework_kpool_tail = bool(config.get("requires_framework_kpool_tail", False))

        attention_layers = [module for module in model.modules() if isinstance(module, Attention)]
        if not attention_layers:
            raise ValueError("Python model does not contain an Attention layer")

        # GLM-Next mixes DSA (MLA) and KDA (linear-attention) layers with
        # different head/dim configs; the paged backend only serves the DSA
        # layers, so it is built from the first DSA layer and the "identical
        # config across all layers" check is skipped. Non-GLM-Next models keep
        # the upstream behavior: backend from the first layer plus the
        # identical-config check. DSA layers are tagged with the
        # ``is_glm_next_mla`` class attribute so this dispatch does not have to
        # import glm5_next (that import pulls KDA kernel transitive deps and
        # fails on builds without them).
        dsa_layers = [layer for layer in attention_layers if getattr(layer, "is_glm_next_mla", False)]
        has_kda_layers = any(getattr(layer, "is_glm_next_kda", False) for layer in attention_layers)
        is_draft_engine = bool(config.get("is_draft_engine", False))
        adaptive_speculative_decode_enabled = bool(config.get("runtime_adaptive_speculative_decode_enabled", False))
        if has_kda_layers and not is_draft_engine and adaptive_speculative_decode_enabled:
            raise ValueError(
                "GLM5 KDA does not support adaptive speculative decode; disable adaptive speculative decoding."
            )
        if dsa_layers:
            first_attention = dsa_layers[0]
        else:
            first_attention = attention_layers[0]
            expected_config = self._attention_config(first_attention)
            for layer in attention_layers[1:]:
                if self._attention_config(layer) != expected_config:
                    raise ValueError("Attention backend requires identical attention configuration across all layers")

        first_parameter = next(model.parameters())
        device = first_parameter.device
        self._num_attention_layers = len(attention_layers)
        num_decoding_tokens = max(
            int(num_decoding_tokens),
            int(config.get("num_speculative_tokens", 0)) + 1,
        )
        self.attention_backend = _create_attention_backend(
            first_attention,
            device,
            first_parameter.dtype,
            config,
            max_seqs_per_batch,
            num_decoding_tokens=num_decoding_tokens,
        )

        execution_model = model.model
        self.eager_runner = EagerRunner(execution_model, self.attention_backend, device)
        # Context-Parallel: shard prefill sequences across the CP group. Decode
        # stays on the non-CP path (CP is prefill-only, eager-only in v1).
        self.eager_runner.cp_size = int(config.get("cp_size", 1))
        self.eager_runner.cp_rank = int(config.get("cp_rank", 0))
        self.layerwise_split_size = int(config.get("layerwise_split_size", 1))
        self.layerwise_split_rank = int(config.get("layerwise_split_rank", 0))
        if self.layerwise_split_size > 1 and config.get("model_type") != "glm_moe_dsa":
            raise NotImplementedError("Python layerwise split is supported only for GLM5.2")
        self.decode_graph_runner = None
        self.inductor_runner = None

        graph_backend = _resolve_graph_backend(config)
        if self.layerwise_split_size > 1 and graph_backend not in ("", "off", "none", "0"):
            raise NotImplementedError(
                "Python GLM5.2 layerwise split requires eager execution; "
                f"graph backend '{graph_backend}' is not supported."
            )
        dp_size = int(config.get("dp_size", 1))
        dp_rank = int(config.get("dp_rank", 0))
        self.dp_size = dp_size
        token_owner_mega_moe_provider = create_token_owner_mega_moe_context_provider(execution_model)
        execution_context_providers = (
            (token_owner_mega_moe_provider,) if token_owner_mega_moe_provider is not None else ()
        )
        self.eager_runner.bind_execution_context_providers(execution_context_providers)
        if dp_size > 1 and graph_backend not in (
            "",
            "off",
            "none",
            "0",
            "cudagraphs",
            "aclgraph",
        ):
            raise NotImplementedError("Python data parallel graph execution supports cudagraphs and aclgraph only")
        if graph_backend in ("", "off", "none", "0"):
            pass
        elif graph_backend == "cudagraphs":
            from xllm.python.model_executor.runners.decode_cuda_graph import (
                DecodeCudaGraphRunner,
            )

            self.decode_graph_runner = DecodeCudaGraphRunner(
                execution_model,
                self.attention_backend,
                device,
                max_seqs_per_batch,
                int(config["max_position_embeddings"]),
                dp_size,
                dp_rank,
            )
        elif graph_backend == "aclgraph":
            from xllm.python.model_executor.runners.decode_acl_graph import (
                DecodeAclGraphRunner,
            )

            self.decode_graph_runner = DecodeAclGraphRunner(
                execution_model,
                self.attention_backend,
                device,
                max_seqs_per_batch,
                int(config["max_position_embeddings"]),
                dp_size,
                dp_rank,
                acl_graph_decode_batch_size_limit,
                # MTP spec-verify packs (num_speculative_tokens+1) token rows
                # per logical sequence; the runner sizes its static paging
                # buffer in token rows, so pass the row multiplier so capacity
                # matches the captured expanded-verify graph (item 二). Take
                # the max of the explicit ctor param and the config-derived
                # width so a caller that passes num_decoding_tokens still wins
                # (single source of truth, no param/config divergence).
                num_decoding_tokens=num_decoding_tokens,
                enable_mega_moe_token_mask=bool(
                    config.get("enable_mega_moe", False) and token_owner_mega_moe_provider is None
                ),
            )
        else:
            if self.layerwise_split_size > 1:
                raise NotImplementedError(
                    "Python layerwise split requires eager execution; graph "
                    f"backend '{graph_backend}' is not supported."
                )
            if self.eager_runner.cp_size > 1:
                # CP is prefill-only and lives on eager_runner; a compile
                # backend serves prefill through InductorRunner, which carries
                # no cp_context, so CP would silently no-op. Reject rather than
                # run without the requested sharding.
                raise NotImplementedError(
                    "Context-Parallel (cp_size > 1) is not supported with the "
                    f"'{graph_backend}' graph backend; CP is eager-only. Use "
                    "graph_backend=off/aclgraph, or set cp_size=1."
                )
            from xllm.python.model_executor.runners.inductor import InductorRunner

            self.inductor_runner = InductorRunner(execution_model, self.attention_backend, device, graph_backend)

        if self.decode_graph_runner is not None:
            self.decode_graph_runner.bind_execution_context_providers(execution_context_providers)

    @staticmethod
    def _attention_config(layer: Attention) -> tuple[int, int, int, float, int]:
        return (
            layer.num_heads,
            layer.num_kv_heads,
            layer.head_dim,
            layer.scale,
            layer.sliding_window,
        )

    def bind_kv_caches(self, kv_caches: list[LayerCacheInput]) -> None:
        layer_caches = normalize_layer_caches(kv_caches)
        required_layers = max(layer.layer_id for layer in self.model.modules() if isinstance(layer, Attention)) + 1
        if len(layer_caches) < required_layers:
            raise ValueError("cache layer count does not match the model layer layout")
        if self._kv_bound:
            return
        if self._requires_framework_kpool_tail:
            indexed_caches = [cache for cache in layer_caches if cache.index is not None]
            if not indexed_caches or any(cache.kpool_tail is None for cache in indexed_caches):
                raise ValueError("model requires a framework-managed kPool tail for every indexer layer")
        self.attention_backend.bind_kv_caches(layer_caches)
        self.eager_runner.bind_layer_caches(layer_caches)
        if self.decode_graph_runner is not None:
            self.decode_graph_runner.bind_layer_caches(layer_caches)
        if self.inductor_runner is not None:
            self.inductor_runner.bind_layer_caches(layer_caches)
        self._kv_bound = True

    def _reset_kda_spec_state_on_pd_handoff(self, metadata: AttentionMetadata) -> None:
        """Reset process-local KDA speculative state on first PD decode."""
        reset_mask = getattr(metadata, "pd_handoff_reset_mask", None)
        if not isinstance(reset_mask, torch.Tensor) or reset_mask.numel() == 0:
            return

        slots = getattr(metadata, "linear_state_indices", None)
        if not isinstance(slots, torch.Tensor) or slots.numel() == 0:
            raise RuntimeError("PD handoff reset requires linear-state indices")
        if reset_mask.numel() != slots.numel():
            raise RuntimeError(
                "PD handoff reset mask is not aligned with linear-state "
                f"slots: mask={reset_mask.numel()}, slots={slots.numel()}"
            )

        reset_fn = getattr(self.attention_backend, "reset_kda_spec_slots", None)
        if reset_fn is None:
            raise RuntimeError("PD handoff reset requires KDA speculative-state support")
        reset_slots = slots.reshape(-1)[reset_mask.reshape(-1).to(device=slots.device, dtype=torch.bool)]
        if reset_slots.numel() > 0:
            reset_fn(reset_slots)

    @torch.inference_mode()
    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        layer_synchronizer: LayerSynchronizer | None = None,
        expert_load_data: torch.Tensor | None = None,
        eplb_decode_token_mask: torch.Tensor | None = None,
        is_graph_warmup: bool = False,
    ) -> ModelExecutionOutput:
        if not self._kv_bound:
            raise RuntimeError("KV caches are not bound")
        if self.layerwise_split_size > 1 and (metadata.is_prefill or metadata.is_chunked_prefill):
            raise NotImplementedError("Python GLM5.2 layerwise split is decode-only")

        self._reset_kda_spec_state_on_pd_handoff(metadata)

        eplb = None
        if expert_load_data is not None:
            eplb = EplbRuntimeState(
                expert_load_data=expert_load_data,
                decode_token_mask=eplb_decode_token_mask,
                is_graph_warmup=is_graph_warmup,
            )
        graph_runner = self.decode_graph_runner
        if (
            graph_runner is not None
            and (eplb is None or current_platform.is_npu())
            and graph_runner.can_execute(input_ids, metadata, input_embedding)
        ):
            graph_runner.warmup(
                input_ids,
                positions,
                metadata,
                input_embedding,
            )
            return graph_runner.execute(
                input_ids,
                positions,
                metadata,
                input_embedding,
                eplb=eplb,
            )
        if self.inductor_runner is not None:
            return self.inductor_runner.execute(
                input_ids,
                positions,
                metadata,
                input_embedding,
                layer_synchronizer,
                eplb,
            )
        return self.eager_runner.execute(
            input_ids,
            positions,
            metadata,
            input_embedding,
            layer_synchronizer,
            eplb,
        )
