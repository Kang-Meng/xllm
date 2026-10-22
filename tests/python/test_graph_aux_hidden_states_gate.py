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

"""Contract tests for the graph aux-hidden-state opt-in.

The deepseek_v41 target's transient aux capture (a ``(hidden, aux)`` tuple)
cannot replay through its CSA2 + fused-ops graph path, so ``py_causal_lm.cpp``
pins that one target to eager unless ``enable_graph_aux_hidden_states`` turns
on the persistent-buffer publication. The incompatibility is a per-model
property: every other capture target - e.g. the glm5_next DFlash2 draft tap
since #240 - allocates its fresh per-forward buffer inside the graph's private
memory pool at fixed addresses, and the runner replays the tuple fine. The
aux-capture eager gate must therefore stay scoped to ``deepseek_v41`` and
never spill onto models whose capture coexists with the graph.

The C++ cannot be compiled here (no toolchain), so these tests pin the two
sides of the contract: the flag wiring in ``execution_config.{h,cpp}`` and the
predicate structure in ``py_causal_lm.cpp``, plus a truth table for the
predicate itself (which must reproduce the pre-aux-gate mainline decision for
every non-v41 model).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXECUTION_CONFIG_H = _REPO_ROOT / "xllm/core/framework/config/execution_config.h"
_EXECUTION_CONFIG_CPP = _REPO_ROOT / "xllm/core/framework/config/execution_config.cpp"
_PY_CAUSAL_LM = _REPO_ROOT / "xllm/models/llm/py_causal_lm.cpp"

_FLAG = "enable_graph_aux_hidden_states"


def _read(path: Path) -> str:
    return path.read_text()


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _extract_assignment(text: str, marker: str) -> str:
    """Return the initializer of ``<marker> = ...;`` (no nested semicolons)."""
    start = text.index(marker)
    end = text.index(";", start)
    return _collapse(text[start:end])


# --------------------------------------------------------------------------
# Predicate truth table (mirror of py_causal_lm.cpp::build_config_dict)
# --------------------------------------------------------------------------


def _requires_eager(
    *,
    eager_draft_model: bool,
    layers_to_capture: bool,
    allow_graph_aux_hidden_states: bool,
    model_type: str,
) -> bool:
    return eager_draft_model or (layers_to_capture and model_type in _V41_FAMILY and not allow_graph_aux_hidden_states)


def _mainline_requires_eager(*, eager_draft_model: bool) -> bool:
    """The pre-aux-gate mainline predicate: the draft set alone decides."""
    return eager_draft_model


# Mirror of util::is_deepseek_v41_family_model_type (core/util/utils.h):
# the v41 target plus its python-path DSpark draft spelling. The draft member
# is eager anyway through the draft set, so family membership changes nothing
# for it; both spellings are kept so the mirror tracks the C++ predicate.
_V41_FAMILY = frozenset({"deepseek_v41", "deepseek_v41_dspark"})

_MODEL_TYPES = ("deepseek_v41", "deepseek_v41_dspark", "glm5_next", "qwen3", "DFlashDraftModel")


@pytest.mark.parametrize("model_type", _MODEL_TYPES)
@pytest.mark.parametrize("allow_graph_aux_hidden_states", [False, True])
def test_non_v41_models_reproduce_the_mainline_decision(model_type: str, allow_graph_aux_hidden_states: bool) -> None:
    """Every non-v41 model ignores the aux gate entirely (glm5_next graph
    coexistence since #240 must not regress)."""
    if model_type in _V41_FAMILY:
        pytest.skip("covered by the v41-specific tests below")
    for layers_to_capture in (False, True):
        for eager_draft_model in (False, True):
            assert _requires_eager(
                eager_draft_model=eager_draft_model,
                layers_to_capture=layers_to_capture,
                allow_graph_aux_hidden_states=allow_graph_aux_hidden_states,
                model_type=model_type,
            ) is _mainline_requires_eager(eager_draft_model=eager_draft_model)


@pytest.mark.parametrize("model_type", sorted(_V41_FAMILY))
@pytest.mark.parametrize("eager_draft_model", [False, True])
@pytest.mark.parametrize("layers_to_capture", [False, True])
def test_flag_off_pins_only_the_v41_family_capture_target_eager(
    model_type: str, eager_draft_model: bool, layers_to_capture: bool
) -> None:
    expected = eager_draft_model or layers_to_capture
    assert (
        _requires_eager(
            eager_draft_model=eager_draft_model,
            layers_to_capture=layers_to_capture,
            allow_graph_aux_hidden_states=False,
            model_type=model_type,
        )
        is expected
    )


@pytest.mark.parametrize("model_type", sorted(_V41_FAMILY))
@pytest.mark.parametrize("eager_draft_model", [False, True])
@pytest.mark.parametrize("layers_to_capture", [False, True])
def test_flag_on_lets_the_v41_family_capture_target_use_the_graph(
    model_type: str, eager_draft_model: bool, layers_to_capture: bool
) -> None:
    expected = eager_draft_model
    assert (
        _requires_eager(
            eager_draft_model=eager_draft_model,
            layers_to_capture=layers_to_capture,
            allow_graph_aux_hidden_states=True,
            model_type=model_type,
        )
        is expected
    )


def test_flag_alone_never_forces_eager() -> None:
    for model_type in _MODEL_TYPES:
        assert not _requires_eager(
            eager_draft_model=False,
            layers_to_capture=False,
            allow_graph_aux_hidden_states=True,
            model_type=model_type,
        )


def test_every_draft_stays_eager() -> None:
    for model_type in ("DFlashDraftModel", "DSparkDraftModel", "deepseek_v41_dspark"):
        assert _requires_eager(
            eager_draft_model=True,
            layers_to_capture=False,
            allow_graph_aux_hidden_states=True,
            model_type=model_type,
        )


# --------------------------------------------------------------------------
# C++ source contract: py_causal_lm.cpp
# --------------------------------------------------------------------------


def test_py_causal_lm_scopes_the_aux_gate_to_the_v41_target() -> None:
    text = _read(_PY_CAUSAL_LM)

    assert f"ExecutionConfig::get_instance().{_FLAG}()" in text
    expression = _extract_assignment(text, "bool requires_eager_execution =")
    assert "eager_draft_model" in expression
    assert "!model_args_.layers_to_capture().empty()" in expression
    # The gate must AND the v41 family with the flag, never OR it: an OR
    # spills the eager pin onto every non-v41 capture target (the glm5_next
    # regression this file locks out). The family predicate - not a bare
    # string - is the required form so future v41 spellings inherit it.
    assert "util::is_deepseek_v41_family_model_type(model_args_.model_type())" in expression
    assert "!allow_graph_aux_hidden_states" in expression
    assert 'model_args_.model_type() != "deepseek_v41"' not in expression
    assert 'model_args_.model_type() == "deepseek_v41" &&' not in expression


def test_py_causal_lm_keeps_the_draft_set_unchanged() -> None:
    text = _read(_PY_CAUSAL_LM)
    draft_set = _extract_assignment(text, "const bool eager_draft_model =")

    assert "model_args_.model_type() == kDFlashDraftModelType" in draft_set
    assert "model_args_.model_type() == kDSparkDraftModelType" in draft_set
    assert "util::is_deepseek_v4_dspark_model_type(model_args_.model_type())" in draft_set


def test_py_causal_lm_bridges_the_flag_to_the_python_config() -> None:
    text = _read(_PY_CAUSAL_LM)

    assert f'd["{_FLAG}"] = allow_graph_aux_hidden_states;' in text


# --------------------------------------------------------------------------
# C++ source contract: execution_config.{h,cpp}
# --------------------------------------------------------------------------


def test_flag_property_defaults_to_false() -> None:
    header = _read(_EXECUTION_CONFIG_H)

    assert f"PROPERTY(bool, {_FLAG}) = false;" in header
    assert f'"{_FLAG}"' in header  # listed in option_category()


def test_flag_gflag_defaults_to_false_and_wires_every_category() -> None:
    source = _collapse(_read(_EXECUTION_CONFIG_CPP))

    define = _extract_assignment(source, f"DEFINE_bool( {_FLAG},")
    assert "false" in define
    assert f"XLLM_CONFIG_ASSIGN_FROM_FLAG({_FLAG});" in source
    assert f"XLLM_CONFIG_ASSIGN_FROM_JSON({_FLAG});" in source
    assert f"APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT( config_json, default_config, {_FLAG});" in source
