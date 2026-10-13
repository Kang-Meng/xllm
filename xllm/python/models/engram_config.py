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

"""Strict parsing for the ``--engram-config`` payload.

The C++ ``ModelConfig.engram_config`` gflag carries the raw JSON string through
to the Python executor untouched (``py_causal_lm.cpp`` injects it verbatim into
the config dict), so this module is the single place that decodes and validates
it. Parsing is deliberately strict: a malformed value raises instead of being
coerced, so a typo cannot silently disable a requested switch.

The two switches are ``cpu_offload`` (keep the int8 table in host memory -- the
offload switch) and ``dp_shared_memory`` (share the host table across the DP
group). ``dp_shared_memory`` is only meaningful together with ``cpu_offload``,
so requesting it without ``cpu_offload`` is rejected at parse time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# The only keys the engram config accepts, in the order they are named in
# error messages. Anything else is a hard error so a typo in
# ``--engram-config`` fails loudly instead of being ignored.
_ALLOWED_KEYS = ("cpu_offload", "dp_shared_memory")


@dataclass(frozen=True)
class EngramConfig:
    """Immutable engram host-offload switches parsed from ``--engram-config``.

    ``cpu_offload`` selects the host-resident table; ``dp_shared_memory``
    requests the DP-shared host table and is only legal together with
    ``cpu_offload``.
    """

    cpu_offload: bool = False
    dp_shared_memory: bool = False

    @classmethod
    def from_value(cls, value: Any) -> EngramConfig:
        """Build from ``None`` / bool / dict / JSON string.

        ``None`` disables both switches. A native ``bool`` (or a JSON string
        decoding to one) sets ``cpu_offload`` only. A ``dict`` -- or a JSON
        string decoding to one -- is read key-by-key; field values must be real
        booleans (``"true"``/``0``/``1`` are rejected).
        """
        if value is None:
            return cls()
        if isinstance(value, str):
            value = cls._decode_json(value)
        if isinstance(value, bool):
            # ``bool`` is a subclass of ``int``; this check must precede any
            # numeric handling so ``True``/``False`` are not read as 1/0.
            return cls(cpu_offload=value)
        if not isinstance(value, dict):
            raise ValueError(f"engram_config must be a JSON object or boolean, got {type(value).__name__}")
        return cls._from_object(value)

    @staticmethod
    def _decode_json(text: str) -> Any:
        """Strictly decode ``text``; reject duplicate keys.

        ``object_pairs_hook`` runs before a repeated key is silently
        overwritten (JSON's usual last-wins), so a duplicate key is a hard error
        instead of a silent one.
        """
        duplicates: list[str] = []

        def _collect_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            seen: set[str] = set()
            for key, _ in pairs:
                if key in seen:
                    duplicates.append(key)
                seen.add(key)
            return dict(pairs)

        try:
            decoded = json.loads(text, object_pairs_hook=_collect_duplicates)
        except json.JSONDecodeError as exc:
            raise ValueError(f"engram_config is not valid JSON: {exc}") from exc
        if duplicates:
            raise ValueError(f"engram_config JSON has duplicate key(s): {', '.join(duplicates)}")
        return decoded

    @classmethod
    def _from_object(cls, obj: dict[str, Any]) -> EngramConfig:
        unknown = [key for key in obj if key not in _ALLOWED_KEYS]
        if unknown:
            raise ValueError(f"engram_config has unknown key(s) {unknown}; allowed keys are {list(_ALLOWED_KEYS)}")
        cpu_offload = cls._bool_field(obj, "cpu_offload")
        dp_shared_memory = cls._bool_field(obj, "dp_shared_memory")
        if dp_shared_memory and not cpu_offload:
            raise ValueError("engram_config.dp_shared_memory=true requires engram_config.cpu_offload=true")
        return cls(cpu_offload=cpu_offload, dp_shared_memory=dp_shared_memory)

    @staticmethod
    def _bool_field(obj: dict[str, Any], key: str) -> bool:
        value = obj.get(key, False)
        if not isinstance(value, bool):
            # Only a real boolean is accepted: ``"true"``, ``0`` and ``1`` are
            # rejected so a stringly-typed config never flips a switch.
            raise ValueError(f"engram_config.{key} must be a boolean, got {type(value).__name__}")
        return value
