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

"""Upstream allocation and validation for NPU Token Owner MegaMoe."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from xllm.python.attention.backend import AttentionMetadata
from xllm.python.layers.npu.mega_moe_context import (
    MegaMoeContext,
    MegaMoeLayerContext,
    MegaMoeLayerSpec,
)

_MEGA_MOE_MAX_TOKENS = 4096


class TokenOwnerMegaMoeContextProvider:
    """Creates eager-forward and graph-entry-scoped MegaMoe addresses."""

    context_type = MegaMoeContext

    def __init__(self, layer_specs: Sequence[MegaMoeLayerSpec]) -> None:
        specs = tuple(layer_specs)
        if not specs:
            raise ValueError("Token Owner MegaMoe requires at least one layer spec")

        first = specs[0]
        if first.dp_size <= 0:
            raise ValueError("MegaMoe DP size must be positive")
        if first.dp_rank < 0 or first.dp_rank >= first.dp_size:
            raise ValueError(f"MegaMoe DP rank {first.dp_rank} is outside DP size {first.dp_size}")
        if first.tp_rank < 0:
            raise ValueError("MegaMoe TP rank must be non-negative")

        layer_ids = tuple(spec.layer_id for spec in specs)
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError(f"MegaMoe layer ids must be unique, got {layer_ids}")

        topology = (first.dp_size, first.dp_rank, first.tp_rank)
        for spec in specs:
            if (spec.dp_size, spec.dp_rank, spec.tp_rank) != topology:
                raise ValueError(
                    "Token Owner MegaMoe layers must use one execution topology: "
                    f"expected={topology}, got={(spec.dp_size, spec.dp_rank, spec.tp_rank)}"
                )
            if spec.hidden_size <= 0 or spec.top_k <= 0 or spec.token_limit <= 0:
                raise ValueError(
                    "MegaMoe layer dimensions must be positive: "
                    f"layer={spec.layer_id}, hidden_size={spec.hidden_size}, "
                    f"top_k={spec.top_k}, token_limit={spec.token_limit}"
                )

        self._layer_specs = specs
        self._dp_size = first.dp_size
        self._dp_rank = first.dp_rank
        self._tp_rank = first.tp_rank

    def build_eager(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> MegaMoeContext:
        execution_token_counts = self._execution_token_counts(
            metadata,
            input_ids.shape[0],
        )
        if any(count <= 0 for count in execution_token_counts):
            raise RuntimeError(f"MegaMoe requires positive DP execution token counts, got {execution_token_counts}")
        local_token_count = execution_token_counts[self._dp_rank]
        if local_token_count != input_ids.shape[0]:
            raise RuntimeError(
                "DP execution token count does not match the local input: "
                f"rank={self._dp_rank}, rows={input_ids.shape[0]}, token_counts={execution_token_counts}"
            )
        return self._build_context(
            max(execution_token_counts),
            local_token_count,
            input_ids.device,
            share_writable_layer_buffers=True,
        )

    def allocate_graph(
        self,
        token_capacity: int,
        device: torch.device,
        metadata: AttentionMetadata,
    ) -> MegaMoeContext:
        del metadata
        if token_capacity <= 0:
            raise ValueError("MegaMoe graph token capacity must be positive")
        return self._build_context(
            token_capacity,
            token_capacity,
            device,
            share_writable_layer_buffers=False,
        )

    def update_graph(
        self,
        context: object,
        metadata: AttentionMetadata,
        local_token_count: int,
    ) -> None:
        if not isinstance(context, MegaMoeContext):
            raise TypeError("Token Owner MegaMoe received an invalid graph execution context")

        execution_token_counts = self._execution_token_counts(
            metadata,
            local_token_count,
        )
        if execution_token_counts[self._dp_rank] != local_token_count:
            raise RuntimeError(
                "DP execution token count does not match the local graph input: "
                f"rank={self._dp_rank}, rows={local_token_count}, token_counts={execution_token_counts}"
            )
        if any(count < 0 or count > context.token_capacity for count in execution_token_counts):
            raise RuntimeError(
                "DP token counts must fit the ACL graph batch bucket: "
                f"counts={execution_token_counts}, bucket={context.token_capacity}"
            )

        if self._tp_rank != 0:
            return
        active_token_count = execution_token_counts[self._dp_rank]
        context.active_token_mask.zero_()
        context.active_token_mask[:active_token_count].fill_(1)

    def _execution_token_counts(
        self,
        metadata: AttentionMetadata,
        local_token_count: int,
    ) -> tuple[int, ...]:
        execution_token_counts = tuple(
            int(count)
            for count in getattr(
                metadata,
                "dp_execution_token_counts",
                (),
            )
        )
        if not execution_token_counts and self._dp_size == 1:
            execution_token_counts = (local_token_count,)
        if len(execution_token_counts) != self._dp_size:
            raise RuntimeError(f"expected {self._dp_size} DP execution token counts, got {execution_token_counts}")
        return execution_token_counts

    def _build_context(
        self,
        token_capacity: int,
        local_token_count: int,
        device: torch.device,
        *,
        share_writable_layer_buffers: bool,
    ) -> MegaMoeContext:
        active_token_mask = self._allocate_active_token_mask(
            token_capacity,
            local_token_count,
            device,
        )
        self._validate_active_token_mask(
            active_token_mask,
            token_capacity,
            device,
        )
        shared_contexts: dict[tuple[int, int, torch.dtype, torch.device], MegaMoeLayerContext] = {}
        layer_contexts: dict[int, MegaMoeLayerContext | None] = {}
        for spec in self._layer_specs:
            if spec.device != device:
                raise RuntimeError(
                    f"MegaMoe layer {spec.layer_id} is on {spec.device}, but execution input is on {device}"
                )
            if token_capacity > min(_MEGA_MOE_MAX_TOKENS, spec.token_limit):
                layer_contexts[spec.layer_id] = None
                continue

            buffer_signature = (
                spec.hidden_size,
                spec.top_k,
                spec.dtype,
                spec.device,
            )
            shared_context = shared_contexts.get(buffer_signature)
            if shared_context is None:
                shared_context = self._allocate_layer_context(
                    spec,
                    token_capacity,
                    local_token_count,
                )
                self._validate_layer_context(
                    spec,
                    shared_context,
                    token_capacity,
                    local_token_count,
                )
                shared_contexts[buffer_signature] = shared_context

            # Eager layers execute serially and may reuse all scratch. During
            # graph replay, only immutable dummy inputs are shared; each
            # non-owner layer keeps an independent broadcast destination.
            if share_writable_layer_buffers or shared_context.input_buffer is None:
                layer_context = shared_context
            else:
                layer_context = self._allocate_graph_layer_context(
                    shared_context,
                    spec,
                    local_token_count,
                )
                self._validate_layer_context(
                    spec,
                    layer_context,
                    token_capacity,
                    local_token_count,
                )
            layer_contexts[spec.layer_id] = layer_context

        return MegaMoeContext(
            token_capacity=token_capacity,
            active_token_mask=active_token_mask,
            layers=layer_contexts,
        )

    def _allocate_active_token_mask(
        self,
        token_capacity: int,
        local_token_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        active_token_mask = torch.zeros(
            token_capacity,
            dtype=torch.int8,
            device=device,
        )
        if self._tp_rank == 0:
            active_token_mask[:local_token_count].fill_(1)
        else:
            active_token_mask[0] = 1
        return active_token_mask

    def _allocate_graph_layer_context(
        self,
        shared_context: MegaMoeLayerContext,
        spec: MegaMoeLayerSpec,
        local_token_count: int,
    ) -> MegaMoeLayerContext:
        return MegaMoeLayerContext(
            input_buffer=shared_context.input_buffer,
            topk_weights_buffer=shared_context.topk_weights_buffer,
            topk_ids_buffer=shared_context.topk_ids_buffer,
            output_buffer=torch.empty(
                local_token_count,
                spec.hidden_size,
                dtype=spec.dtype,
                device=spec.device,
            ),
        )

    def _allocate_layer_context(
        self,
        spec: MegaMoeLayerSpec,
        token_capacity: int,
        local_token_count: int,
    ) -> MegaMoeLayerContext:
        if self._tp_rank == 0:
            if local_token_count == token_capacity:
                return MegaMoeLayerContext(
                    input_buffer=None,
                    topk_weights_buffer=None,
                    topk_ids_buffer=None,
                    output_buffer=None,
                )
            return MegaMoeLayerContext(
                input_buffer=torch.zeros(
                    token_capacity,
                    spec.hidden_size,
                    dtype=spec.dtype,
                    device=spec.device,
                ),
                topk_weights_buffer=torch.zeros(
                    token_capacity,
                    spec.top_k,
                    dtype=torch.float32,
                    device=spec.device,
                ),
                topk_ids_buffer=torch.zeros(
                    token_capacity,
                    spec.top_k,
                    dtype=torch.int32,
                    device=spec.device,
                ),
                output_buffer=None,
            )

        return MegaMoeLayerContext(
            input_buffer=torch.zeros(
                token_capacity,
                spec.hidden_size,
                dtype=spec.dtype,
                device=spec.device,
            ),
            topk_weights_buffer=torch.full(
                (token_capacity, spec.top_k),
                1.0 / spec.top_k,
                dtype=torch.float32,
                device=spec.device,
            ),
            topk_ids_buffer=(
                torch.arange(
                    spec.top_k,
                    dtype=torch.int32,
                    device=spec.device,
                )
                .view(1, spec.top_k)
                .expand(token_capacity, spec.top_k)
                .contiguous()
            ),
            output_buffer=torch.empty(
                local_token_count,
                spec.hidden_size,
                dtype=spec.dtype,
                device=spec.device,
            ),
        )

    def _validate_layer_context(
        self,
        spec: MegaMoeLayerSpec,
        layer_context: MegaMoeLayerContext,
        token_capacity: int,
        local_token_count: int,
    ) -> None:
        input_buffers = (
            layer_context.input_buffer,
            layer_context.topk_weights_buffer,
            layer_context.topk_ids_buffer,
        )
        has_all_input_buffers = all(buffer is not None for buffer in input_buffers)
        has_any_input_buffer = any(buffer is not None for buffer in input_buffers)
        if has_any_input_buffer and not has_all_input_buffers:
            raise RuntimeError(f"MegaMoe layer {spec.layer_id} input buffers are incomplete")
        if self._tp_rank == 0 and has_all_input_buffers != (local_token_count < token_capacity):
            raise RuntimeError(f"MegaMoe owner layer {spec.layer_id} has an invalid padding-buffer layout")
        if self._tp_rank != 0 and not has_all_input_buffers:
            raise RuntimeError(f"MegaMoe non-owner layer {spec.layer_id} input buffers are unavailable")
        if layer_context.input_buffer is not None:
            if layer_context.input_buffer.shape != (token_capacity, spec.hidden_size):
                raise RuntimeError(f"MegaMoe layer {spec.layer_id} input buffer has an invalid shape")
            if layer_context.input_buffer.dtype != spec.dtype or layer_context.input_buffer.device != spec.device:
                raise RuntimeError(f"MegaMoe layer {spec.layer_id} input buffer has an invalid dtype or device")
            if layer_context.topk_weights_buffer is None or layer_context.topk_weights_buffer.shape != (
                token_capacity,
                spec.top_k,
            ):
                raise RuntimeError(f"MegaMoe layer {spec.layer_id} top-k weight buffer has an invalid shape")
            if layer_context.topk_weights_buffer.dtype != torch.float32:
                raise RuntimeError(f"MegaMoe layer {spec.layer_id} top-k weight buffer must use float32")
            if layer_context.topk_weights_buffer.device != spec.device:
                raise RuntimeError(f"MegaMoe layer {spec.layer_id} top-k weight buffer is on the wrong device")
            if layer_context.topk_ids_buffer is None or layer_context.topk_ids_buffer.shape != (
                token_capacity,
                spec.top_k,
            ):
                raise RuntimeError(f"MegaMoe layer {spec.layer_id} top-k id buffer has an invalid shape")
            if layer_context.topk_ids_buffer.dtype != torch.int32:
                raise RuntimeError(f"MegaMoe layer {spec.layer_id} top-k id buffer must use int32")
            if layer_context.topk_ids_buffer.device != spec.device:
                raise RuntimeError(f"MegaMoe layer {spec.layer_id} top-k id buffer is on the wrong device")

        output_buffer = layer_context.output_buffer
        if self._tp_rank == 0:
            if output_buffer is not None:
                raise RuntimeError(f"MegaMoe owner layer {spec.layer_id} must not have an output buffer")
            return
        if output_buffer is None or output_buffer.shape != (local_token_count, spec.hidden_size):
            raise RuntimeError(f"MegaMoe non-owner layer {spec.layer_id} output buffer has an invalid shape")
        if output_buffer.dtype != spec.dtype or output_buffer.device != spec.device:
            raise RuntimeError(f"MegaMoe non-owner layer {spec.layer_id} output buffer has an invalid dtype or device")

    @staticmethod
    def _validate_active_token_mask(
        active_token_mask: torch.Tensor,
        token_capacity: int,
        device: torch.device,
    ) -> None:
        if active_token_mask.shape != (token_capacity,):
            raise RuntimeError("MegaMoe active-token mask has an invalid shape")
        if active_token_mask.dtype != torch.int8:
            raise RuntimeError("MegaMoe active-token mask must use int8")
        if active_token_mask.device != device:
            raise RuntimeError("MegaMoe active-token mask is on the wrong device")


def create_token_owner_mega_moe_context_provider(
    model: nn.Module,
) -> TokenOwnerMegaMoeContextProvider | None:
    """Collect MegaMoe resource declarations from an initialized model."""
    layer_specs: list[MegaMoeLayerSpec] = []
    for module in model.modules():
        layer_spec = getattr(module, "mega_moe_execution_spec", None)
        if layer_spec is None:
            continue
        if not isinstance(layer_spec, MegaMoeLayerSpec):
            raise TypeError("mega_moe_execution_spec must be a MegaMoeLayerSpec")
        layer_specs.append(layer_spec)

    if not layer_specs:
        return None
    return TokenOwnerMegaMoeContextProvider(layer_specs)
