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

from types import SimpleNamespace

import pytest
import torch

from xllm.python.models.deepseek_v4 import DeepseekV4ForCausalLM


@pytest.mark.parametrize(
    ("hash_layer", "metadata_suffix", "parameter_name"),
    [
        (True, "tid2eid.weight", "model.layers.0.mlp.tid2eid"),
        (False, "e_score_correction_bias", "model.layers.0.mlp.e_score_correction_bias"),
    ],
)
def test_moe_routing_loader_accepts_native_metadata_aliases(
    hash_layer: bool,
    metadata_suffix: str,
    parameter_name: str,
) -> None:
    gate_prefix = "layers.0.gate."
    tensors = {
        gate_prefix + "weight": torch.arange(6).reshape(2, 3),
        gate_prefix + metadata_suffix: torch.arange(2),
    }

    class FakeLoader:
        def __init__(self) -> None:
            self.loaded: dict[str, torch.Tensor] = {}

        def load_tensor(self, name: str) -> torch.Tensor:
            return tensors[name]

        def has(self, name: str) -> bool:
            return name in tensors

        def copy_in(self, name: str, tensor: torch.Tensor) -> None:
            self.loaded[name] = tensor

    loader = FakeLoader()
    DeepseekV4ForCausalLM._load_dsv4_moe_routing(
        loader,
        gate_prefix,
        "model.layers.0.",
        SimpleNamespace(hash_layer=hash_layer),
        layer_id=0,
    )

    torch.testing.assert_close(
        loader.loaded["model.layers.0.mlp.gate.weight"],
        tensors[gate_prefix + "weight"],
    )
    torch.testing.assert_close(loader.loaded[parameter_name], tensors[gate_prefix + metadata_suffix])
