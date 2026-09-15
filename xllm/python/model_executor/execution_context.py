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

"""Forward-scoped execution contexts contributed by optional features."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from xllm.python.attention.backend import AttentionMetadata


ExecutionContexts = dict[type[object], object]


class ExecutionContextProvider(Protocol):
    """Builds mode-specific storage without exposing feature logic to runners."""

    context_type: type[object]

    def build_eager(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> object: ...

    def allocate_graph(
        self,
        token_capacity: int,
        device: torch.device,
        metadata: AttentionMetadata,
    ) -> object: ...

    def update_graph(
        self,
        context: object,
        metadata: AttentionMetadata,
        local_token_count: int,
    ) -> None: ...


def _insert_context(
    contexts: ExecutionContexts,
    provider: ExecutionContextProvider,
    context: object,
) -> None:
    context_type = provider.context_type
    if context_type in contexts:
        raise RuntimeError(f"duplicate execution context provider for {context_type.__name__}")
    contexts[context_type] = context


def build_eager_execution_contexts(
    providers: Sequence[ExecutionContextProvider],
    input_ids: torch.Tensor,
    metadata: AttentionMetadata,
) -> ExecutionContexts:
    contexts: ExecutionContexts = {}
    for provider in providers:
        _insert_context(
            contexts,
            provider,
            provider.build_eager(input_ids, metadata),
        )
    return contexts


def allocate_graph_execution_contexts(
    providers: Sequence[ExecutionContextProvider],
    token_capacity: int,
    device: torch.device,
    metadata: AttentionMetadata,
) -> ExecutionContexts:
    contexts: ExecutionContexts = {}
    for provider in providers:
        _insert_context(
            contexts,
            provider,
            provider.allocate_graph(token_capacity, device, metadata),
        )
    return contexts


def update_graph_execution_contexts(
    providers: Sequence[ExecutionContextProvider],
    contexts: ExecutionContexts,
    metadata: AttentionMetadata,
    local_token_count: int,
) -> None:
    for provider in providers:
        context = contexts.get(provider.context_type)
        if context is None:
            raise RuntimeError(f"missing graph execution context for {provider.context_type.__name__}")
        provider.update_graph(context, metadata, local_token_count)
