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

"""Engram config parsing and validation.

Two independent surfaces are pinned, both pure enough to run without an NPU:

* :meth:`EngramConfig.from_value` -- the strict ``--engram-config`` decoder.
* :func:`_validate_engram_config` -- the TP/PP/DCP/load-format gate on
  :class:`DeepseekV41Config`.

``EngramConfig`` is imported from its leaf module (stdlib only). The validator
lives in ``deepseek_v41`` (which pulls in torch), so it is imported lazily
inside its tests; a cfg is a plain namespace, avoiding a full model build.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from xllm.python.models.engram_config import EngramConfig

# ---------------------------------------------------------------------------
# EngramConfig.from_value
# ---------------------------------------------------------------------------


def test_from_value_none_and_defaults_disable_both_switches() -> None:
    assert EngramConfig.from_value(None) == EngramConfig(cpu_offload=False, dp_shared_memory=False)
    assert EngramConfig.from_value({}) == EngramConfig()
    assert EngramConfig.from_value("{}") == EngramConfig()


def test_from_value_bool_sets_cpu_offload_only() -> None:
    assert EngramConfig.from_value(True) == EngramConfig(cpu_offload=True)
    assert EngramConfig.from_value(False) == EngramConfig(cpu_offload=False)
    # A JSON string decoding to a bare boolean is treated the same way.
    assert EngramConfig.from_value("true") == EngramConfig(cpu_offload=True)
    assert EngramConfig.from_value("false") == EngramConfig(cpu_offload=False)


def test_from_value_object_reads_real_booleans() -> None:
    assert EngramConfig.from_value({"cpu_offload": True}) == EngramConfig(cpu_offload=True)
    assert EngramConfig.from_value({"cpu_offload": True, "dp_shared_memory": True}) == EngramConfig(
        cpu_offload=True, dp_shared_memory=True
    )
    assert EngramConfig.from_value('{"cpu_offload": true, "dp_shared_memory": false}') == EngramConfig(cpu_offload=True)


@pytest.mark.parametrize("value", [1, 0, [], [True], "1", "0", 1.0])
def test_from_value_rejects_non_object_non_bool(value: object) -> None:
    with pytest.raises(ValueError, match="must be a JSON object or boolean"):
        EngramConfig.from_value(value)


@pytest.mark.parametrize(
    "obj",
    [
        {"cpu_offload": "true"},
        {"cpu_offload": 1},
        {"cpu_offload": 0},
        {"dp_shared_memory": "yes"},
        {"cpu_offload": True, "dp_shared_memory": 0},
    ],
)
def test_from_value_rejects_stringly_typed_fields(obj: dict) -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        EngramConfig.from_value(obj)


def test_from_value_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        EngramConfig.from_value({"cpu_offload": True, "bogus": True})


def test_from_value_rejects_duplicate_json_keys() -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        EngramConfig.from_value('{"cpu_offload": true, "cpu_offload": false}')


def test_from_value_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        EngramConfig.from_value('{"cpu_offload": tru}')


@pytest.mark.parametrize(
    "obj",
    [
        {"dp_shared_memory": True},
        {"cpu_offload": False, "dp_shared_memory": True},
    ],
)
def test_from_value_rejects_shared_without_offload(obj: dict) -> None:
    with pytest.raises(ValueError, match=r"dp_shared_memory=true requires engram_config.cpu_offload=true"):
        EngramConfig.from_value(obj)


# ---------------------------------------------------------------------------
# _validate_engram_config
# ---------------------------------------------------------------------------


def _validate(cfg: SimpleNamespace) -> None:
    # Imported lazily so the pure EngramConfig cases above run without torch.
    from xllm.python.models.deepseek_v41 import _validate_engram_config

    _validate_engram_config(cfg)


def _engram_cfg(**overrides: object) -> SimpleNamespace:
    fields: dict[str, object] = {
        "engram_layer_ids": [1],
        "tp_size": 1,
        "pp_size": 1,
        "dcp_size": 1,
        "load_format": "safetensors",
        "engram_config": EngramConfig(),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_validate_is_skipped_without_engram_layers() -> None:
    # A non-engram checkpoint ignores the switches entirely, even a bogus TP.
    _validate(_engram_cfg(engram_layer_ids=[], tp_size=16, pp_size=2, dcp_size=2, load_format="gguf"))


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_validate_accepts_supported_tp(tp_size: int) -> None:
    _validate(_engram_cfg(tp_size=tp_size))


@pytest.mark.parametrize("tp_size", [3, 6, 12, 16])
def test_validate_rejects_unsupported_tp(tp_size: int) -> None:
    # 3/6/12 are the values the uniform-shard divisibility rule alone would
    # admit (for a 6-head set), yet the model-level whitelist still refuses
    # them; 16 is simply outside the set. All raise the same whitelist message.
    with pytest.raises(ValueError, match=r"TP in \{1, 2, 4, 8\}"):
        _validate(_engram_cfg(tp_size=tp_size))


@pytest.mark.parametrize("field", ["pp_size", "dcp_size"])
def test_validate_rejects_non_singleton_pp_or_dcp(field: str) -> None:
    with pytest.raises(ValueError, match="PP=1 and DCP=1"):
        _validate(_engram_cfg(**{field: 2}))


def test_validate_rejects_dcp_above_one_with_its_value() -> None:
    # The C++ bridge forwards the live DCP (KV-split) size, so a DCP>1
    # deployment reaches this gate instead of the old hardcoded 1; the message
    # must name the offending value.
    with pytest.raises(ValueError, match=r"DCP=2"):
        _validate(_engram_cfg(dcp_size=2))


@pytest.mark.parametrize("load_format", ["auto", "safetensors", "dummy"])
def test_validate_accepts_known_load_formats(load_format: str) -> None:
    _validate(_engram_cfg(load_format=load_format))


def test_validate_rejects_unknown_load_format() -> None:
    with pytest.raises(ValueError, match="load_format=gguf"):
        _validate(_engram_cfg(load_format="gguf"))


def test_validate_rechecks_shared_implies_offload() -> None:
    # from_value already rejects this, but a future construction path could
    # bypass it; the config gate re-checks the invariant.
    cfg = _engram_cfg(engram_config=EngramConfig(cpu_offload=False, dp_shared_memory=True))
    with pytest.raises(ValueError, match="dp_shared_memory=true requires engram_config.cpu_offload=true"):
        _validate(cfg)
