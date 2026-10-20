# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""Intermediate residual-stream capture for Python target models."""

from __future__ import annotations

from collections.abc import Callable

import torch


class AuxHiddenCapture:
    """Captures selected layer outputs into a buffer in config order.

    Two buffer modes:

    * transient (default) — a fresh buffer per forward, returned to the caller
      as the second element of a ``(hidden, aux)`` tuple;
    * persistent — the buffer is allocated once per (shape, dtype, device) and
      reused across forwards. Graph capture requires fixed addresses it can
      capture and replay, and the graph runner only accepts a single Tensor
      output, so in this mode the capture is published out of band through
      :attr:`buffer` while ``finalize`` returns the plain hidden states.
    """

    def __init__(
        self,
        layers_to_capture: tuple[int, ...],
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        persistent_buffer: bool = False,
    ) -> None:
        self._layers_to_capture = layers_to_capture
        self._capture_slots = {layer_id: slot_id for slot_id, layer_id in enumerate(layers_to_capture)}
        # Applied to a captured layer output before writing, only for captured
        # layers — defers an expensive per-layer collapse past the slot check.
        self._transform = transform
        self._persistent_buffer = bool(persistent_buffer)
        self._buffers: dict[tuple[tuple[int, ...], torch.dtype, torch.device], torch.Tensor] = {}
        self._buffer: torch.Tensor | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._layers_to_capture)

    @property
    def persistent_buffer(self) -> bool:
        """Whether the capture buffer is reused across forwards (graph mode)."""
        return self._persistent_buffer

    @property
    def buffer(self) -> torch.Tensor | None:
        """The most recent persistent capture buffer, if any.

        Out-of-band access to the captured layer outputs when the model output
        must stay a single Tensor (graph capture/replay).
        """
        return self._buffer

    def persistent_buffer_for(self, rows: int) -> torch.Tensor | None:
        """The memoized persistent buffer whose leading axis is exactly ``rows``.

        An ACL-graph replay does not re-run the Python forward, so the graph's
        aux buffer is the one its captured forward allocated. ``buffer``
        (the most recently created) can meanwhile point at a SMALLER buffer
        written by an interleaved 1-row forward -- e.g. the empty DP shard's
        single dummy row. Slicing ``[:rows]`` out of that smaller buffer
        silently clamps to its own row count and hands the drafter a
        per-sequence-shaped tensor where a per-token-row one is expected
        (``gather_v3`` then indexes a 1-row source with ``0..rows-1``). Look the
        buffer up by the row count the graph actually captured instead.
        """
        if rows <= 0:
            return None
        for (shape, _dtype, _device), buffer in self._buffers.items():
            if len(shape) >= 2 and shape[0] == rows:
                return buffer
        return None

    def should_capture(self, layer_id: int) -> bool:
        """Whether ``layer_id`` is one of the configured capture layers.

        Lets a caller skip the per-layer argument preparation (e.g. the
        ``hc_mult`` collapse the V4.1 target needs) for every other layer.
        """
        return layer_id in self._capture_slots

    def create_buffer(self, hidden: torch.Tensor) -> torch.Tensor | None:
        if not self.enabled:
            return None
        shape = (*hidden.shape[:-1], hidden.shape[-1] * len(self._layers_to_capture))
        if not self._persistent_buffer:
            return hidden.new_empty(shape)
        key = (tuple(shape), hidden.dtype, hidden.device)
        buffer = self._buffers.get(key)
        if buffer is None:
            buffer = hidden.new_empty(shape)
            self._buffers[key] = buffer
        self._buffer = buffer
        return buffer

    def capture_layer(
        self,
        layer_id: int,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        buffer: torch.Tensor | None,
    ) -> None:
        slot_id = self._capture_slots.get(layer_id)
        if slot_id is None:
            return
        assert buffer is not None
        if self._transform is not None:
            hidden = self._transform(hidden)
        hidden_size = hidden.shape[-1]
        slot = buffer.narrow(-1, slot_id * hidden_size, hidden_size)
        if residual is None:
            slot.copy_(hidden)
        else:
            torch.add(hidden, residual, out=slot)

    def finalize(
        self,
        hidden: torch.Tensor,
        buffer: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if buffer is None:
            return hidden
        if self._persistent_buffer:
            # Keep the model output a single Tensor so the graph runner can
            # capture it; the capture is consumed through ``buffer``.
            return hidden
        return hidden, buffer
