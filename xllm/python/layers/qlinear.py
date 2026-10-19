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

"""Transparent linear wrapper: fp or W8A8, resolved at load time.

Mirrors libtorch #1426's resolved_weight_quant_method_ pattern in pure Python.
The model graph constructs ``QLinear`` everywhere; whether it actually quantizes
is decided at ``load_weights`` by probing the checkpoint for quant tensors
(``deq_scale`` / ``weight_scale``). Forward is identical either way.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        kind: str = "static",
        row_parallel: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kind = kind
        self.row_parallel = row_parallel
        self.use_w8a8: bool | None = None
        # fp weight (used iff use_w8a8 is False)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)
        self._w8a8: nn.Module | None = None  # built in Task 2

    def resolve_quant(self, has_quant_tensors: bool, *, quant_format: str | None = None) -> None:
        """Resolve fp vs W8A8 and (optionally) pin the quant format.

        ``quant_format`` overrides the construction-time ``kind`` when the
        loader has probed the checkpoint: ``"static"`` (deq_scale/input_scale
        suffixes, internal exports) or ``"dynamic"`` (weight_scale/
        weight_offset suffixes, HF-ecosystem per-channel exports such as
        GLM-5.3-w8a8). Without it the constructor's ``kind`` stands, preserving
        existing GLM-5.3-Flash call sites that pin static/dynamic per
        projection.
        """
        if quant_format is not None and not has_quant_tensors:
            raise ValueError(
                f"resolve_quant(quant_format={quant_format!r}) requires "
                "has_quant_tensors=True: pinning a quant format on an fp "
                "projection is a caller bug"
            )
        self.use_w8a8 = bool(has_quant_tensors)
        if self.use_w8a8:
            if quant_format is not None:
                self.kind = quant_format
            # The fp weight is dead in W8A8 mode (forward reads only the
            # _w8a8 submodule); release its storage so quantized deployments
            # do not pay for both layouts.
            self.weight.data = torch.empty(0, dtype=self.weight.dtype, device=self.weight.device)
            self._build_w8a8()

    def _build_w8a8(self) -> None:
        from xllm.python.models.deepseek_v32 import (
            W8A8DynamicLinear,
            W8A8StaticLinear,
        )

        device = self.weight.device
        cls = W8A8DynamicLinear if self.kind == "dynamic" else W8A8StaticLinear
        if cls is W8A8StaticLinear:
            self._w8a8 = cls(self.in_features, self.out_features, device, row_parallel=self.row_parallel)
        else:
            self._w8a8 = cls(self.in_features, self.out_features, device)

    def load_weight(self, tensor: torch.Tensor) -> None:
        self.weight.data.copy_(tensor.to(dtype=self.weight.dtype, device=self.weight.device))

    def load_w8a8(self, loader, prefix: str, proj: str, shard_dims=None, *, quant_format: str | None = None) -> None:
        """Load w8a8 tensors into this QLinear's ``_w8a8`` submodule.

        ``prefix + proj`` is this module's dotted path inside the model (the
        caller passes the path up to the projection attribute, e.g. ``attn`` +
        ``o_proj``). The loader targets ``<prefix+proj>._w8a8.<suffix>`` so the
        int8 weights land on the submodule ``forward`` actually reads, not this
        QLinear's fp ``weight`` slot. ``quant_format`` (when the caller has
        probed the checkpoint) makes the loader validate that the projection
        ships the full suffix set of that format.
        """
        assert self._w8a8 is not None, "resolve_quant(True) before load_w8a8"
        loader.load_w8a8_into_qlinear(prefix + proj, prefix, proj, shard_dims, quant_format=quant_format)

    def process_weights_after_loading(self) -> None:
        if self.use_w8a8:
            self._w8a8.process_weights_after_loading()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_w8a8 is None:
            raise RuntimeError("QLinear.forward before resolve_quant")
        if self.use_w8a8:
            return self._w8a8(x)
        return torch.nn.functional.linear(x, self.weight, self.bias)


class QLinearWeightLoader:
    """Probe-driven fp/w8a8 load router + TP shard, reusing W8A8WeightLoader."""

    # Required suffixes per W8A8 checkpoint format. A projection shipping one
    # format's scale suffix must carry the full set; a mixed or truncated
    # export would otherwise leave the torch.empty buffers uninitialized and
    # silently corrupt forward numerics (the pre-QLinear loader failed loudly
    # with KeyError instead). weight_offset is deliberately outside the
    # dynamic required set: a symmetric per-channel export may omit it, and
    # W8A8DynamicLinear zero-initializes the buffer so "not provided" reads
    # as symmetric.
    _W8A8_REQUIRED_SUFFIXES = {
        "static": ("weight", "deq_scale", "quant_bias", "input_scale", "input_offset"),
        "dynamic": ("weight", "weight_scale"),
    }

    def __init__(self, model: nn.Module, state_dicts, tp_size: int, tp_rank: int):
        from xllm.python.models.glm5_next_weight import W8A8WeightLoader

        self._w8 = W8A8WeightLoader(model, state_dicts, tp_size, tp_rank)
        self.tp_size = tp_size
        self.tp_rank = tp_rank

    # expose the underlying helpers glm5_next's load_weights still needs directly
    def find(self, name):
        return self._w8.find(name)

    def load_tensor(self, name):
        return self._w8.load_tensor(name)

    def shard(self, t, dim, world=None, rank=None):
        return self._w8.shard(t, dim, world, rank)

    def copy_in(self, name, tensor):
        self._w8.copy_in(name, tensor)

    def probe_quant(self, prefix: str, proj: str) -> bool:
        """True iff the checkpoint carries w8a8 tensors for ``prefix+proj``."""
        return (
            self._w8.find(f"{prefix}{proj}.deq_scale") is not None
            or self._w8.find(f"{prefix}{proj}.weight_scale") is not None
        )

    def load_fp(self, name: str, dim=None) -> None:
        t = self.load_tensor(name)
        if dim is not None:
            t = self.shard(t, dim=dim)
        self.copy_in(name, t)

    def load_w8a8_into_qlinear(
        self,
        qlinear_name: str,
        prefix: str,
        proj: str,
        shard_dims=None,
        *,
        quant_format: str | None = None,
    ) -> None:
        """Load static W8A8 projection tensors into a QLinear's ``_w8a8`` module.

        Source checkpoint keys are ``prefix + proj + ".<suffix>"`` (the
        checkpoint names the QLinear directly, e.g. ``attn.o_proj.<suffix>``).
        The tensors are written to the *int8* submodule at qualified name
        ``qlinear_name + "._w8a8.<suffix>"`` (``qlinear_name`` is the QLinear's
        own dotted path, e.g. ``model.layers.0.self_attn.o_proj``). QLinear's
        ``forward`` reads ``self._w8a8``, so that int8 slot is the only one that
        must be populated. Unlike a loader that targets the module's own name,
        this lands on the ``_w8a8`` path segment.
        """
        # Internal static exports carry deq_scale/quant_bias/input_scale/
        # input_offset; HF-ecosystem dynamic exports (GLM-5.3-w8a8) carry
        # per-channel weight_scale/weight_offset. Load whichever suffix set the
        # checkpoint provides -- resolve_quant has already built the matching
        # W8A8StaticLinear or W8A8DynamicLinear submodule. The suffix-skip
        # below is only meant for the *other* format's keys; when the caller
        # probed the format, require this projection to ship the full suffix
        # set of that format. Structure-only loads (no W8A8 key at all, e.g.
        # empty state dicts in unit tests) keep the silent skip; a projection
        # carrying any known suffix but missing a required one (including
        # weight itself) fails loudly instead of shipping torch.empty
        # garbage into forward.
        if quant_format is not None:
            if quant_format not in self._W8A8_REQUIRED_SUFFIXES:
                raise ValueError(
                    f"unknown W8A8 quant_format {quant_format!r}; expected one of "
                    f"{sorted(self._W8A8_REQUIRED_SUFFIXES)}"
                )
            known_suffixes = (
                self._W8A8_REQUIRED_SUFFIXES["static"] + self._W8A8_REQUIRED_SUFFIXES["dynamic"] + ("weight_offset",)
            )
            if any(self.find(f"{prefix}{proj}.{s}") is not None for s in known_suffixes):
                for suffix in self._W8A8_REQUIRED_SUFFIXES[quant_format]:
                    key = f"{prefix}{proj}.{suffix}"
                    if self.find(key) is None:
                        raise KeyError(
                            f"checkpoint tensor not found: {key} (required by the {quant_format} W8A8 format)"
                        )
        for suffix in (
            "weight",
            "deq_scale",
            "quant_bias",
            "input_scale",
            "input_offset",
            "weight_scale",
            "weight_offset",
        ):
            key = f"{prefix}{proj}.{suffix}"
            if self.find(key) is None:
                continue  # absent in this checkpoint format
            t = self.load_tensor(key)
            dim = (shard_dims or {}).get(suffix)
            if dim is not None:
                t = self.shard(t, dim=dim)
            self.copy_in(f"{qlinear_name}._w8a8.{suffix}", t)

    def load_w8a8_mlp_into_qlinear(self, mlp_pfx: str, *, world: int | None = None, rank: int | None = None) -> None:
        """Load dynamic W8A8 MLP tensors into the QLinear-wrapped ``_w8a8``.

        Cats gate+up on dim 0 and shards down on dim 1, but writes into
        ``gate_up_proj._w8a8`` / ``down_proj._w8a8`` instead of the QLinear fp
        slots. ``mlp_pfx`` is the dotted MLP path, e.g.
        ``model.layers.0.mlp.``.
        """
        g = mlp_pfx + "gate_proj."
        u = mlp_pfx + "up_proj."
        d = mlp_pfx + "down_proj."
        self.copy_in(
            mlp_pfx + "gate_up_proj._w8a8.weight",
            torch.cat(
                [
                    self.shard(self.load_tensor(g + "weight"), 0, world, rank),
                    self.shard(self.load_tensor(u + "weight"), 0, world, rank),
                ],
                dim=0,
            ).contiguous(),
        )
        self.copy_in(
            mlp_pfx + "gate_up_proj._w8a8.weight_scale",
            torch.cat(
                [
                    self.shard(self.load_tensor(g + "weight_scale"), 0, world, rank),
                    self.shard(self.load_tensor(u + "weight_scale"), 0, world, rank),
                ],
                dim=0,
            ).contiguous(),
        )
        self.copy_in(
            mlp_pfx + "gate_up_proj._w8a8.weight_offset",
            torch.cat(
                [
                    self.shard(self.load_tensor(g + "weight_offset"), 0, world, rank),
                    self.shard(self.load_tensor(u + "weight_offset"), 0, world, rank),
                ],
                dim=0,
            ).contiguous(),
        )
        self.copy_in(mlp_pfx + "down_proj._w8a8.weight", self.shard(self.load_tensor(d + "weight"), 1, world, rank))
        self.copy_in(mlp_pfx + "down_proj._w8a8.weight_scale", self.load_tensor(d + "weight_scale"))
        self.copy_in(mlp_pfx + "down_proj._w8a8.weight_offset", self.load_tensor(d + "weight_offset"))
