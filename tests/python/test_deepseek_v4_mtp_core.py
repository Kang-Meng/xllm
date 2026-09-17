# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytest.importorskip("torch_npu")

from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.model_executor.runners.decode_acl_graph import _StaticAttentionMetadata
from xllm.python.models.deepseek_v4_mtp import DeepseekV4MtpLayer, DeepseekV4MtpModel


def _fusion_layer() -> DeepseekV4MtpLayer:
    layer = DeepseekV4MtpLayer.__new__(DeepseekV4MtpLayer)
    nn.Module.__init__(layer)
    layer.cfg = SimpleNamespace(hidden_size=2, hc_mult=2, rms_norm_eps=0.0, hc_eps=0.0)
    layer.enorm = nn.Identity()
    layer.hnorm = nn.Identity()
    layer.e_proj = nn.Identity()
    layer.h_proj = nn.Identity()
    return layer


@pytest.mark.parametrize(
    ("previous_hidden", "expected"),
    [
        (
            torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            torch.tensor([[[11.0, 22.0], [31.0, 42.0]]]),
        ),
        (
            torch.tensor([[10.0, 20.0]]),
            torch.tensor([[[11.0, 22.0], [11.0, 22.0]]]),
        ),
    ],
)
def test_mtp_fuses_target_hidden_layouts(
    previous_hidden: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    fused = _fusion_layer()._fuse_hidden_states(torch.tensor([[1.0, 2.0]]), previous_hidden)

    torch.testing.assert_close(fused, expected)


def test_mtp_rejects_unknown_hidden_width() -> None:
    with pytest.raises(ValueError, match=r"hc_mult \* hidden_size \(4\), but got 3"):
        _fusion_layer()._fuse_hidden_states(torch.ones(1, 2), torch.ones(1, 3))


def test_mtp_dummy_embedding_matches_target_hidden_width() -> None:
    model = DeepseekV4MtpModel.__new__(DeepseekV4MtpModel)
    nn.Module.__init__(model)
    model.cfg = SimpleNamespace(hc_mult=2, hidden_size=3)
    model.norm = SimpleNamespace(weight=torch.empty(3, dtype=torch.float32))

    embedding = model.make_dummy_input_embedding(torch.tensor([1, 2], dtype=torch.int32))

    assert tuple(embedding.shape) == (2, 6)
    assert embedding.dtype == torch.float32
    assert torch.count_nonzero(embedding) == 0


def test_mtp_forward_uses_dummy_flag_from_graph_metadata() -> None:
    class Embedding(nn.Module):
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return torch.ones((input_ids.shape[0], 2))

    class Layer(nn.Module):
        def forward(
            self,
            inputs_embeds: torch.Tensor,
            previous_hidden_states: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            input_ids: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del positions, cos_sin_cache, input_ids
            assert torch.count_nonzero(previous_hidden_states) == 0
            return inputs_embeds, previous_hidden_states.reshape(-1, 2, 2)

    class Norm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2), requires_grad=False)

        def forward(
            self,
            hidden: torch.Tensor,
            residual: torch.Tensor | None,
        ) -> torch.Tensor:
            del residual
            return hidden

    model = DeepseekV4MtpModel.__new__(DeepseekV4MtpModel)
    nn.Module.__init__(model)
    model.cfg = SimpleNamespace(
        compress_ratios=(1,),
        cp_size=1,
        hc_mult=2,
        hidden_size=2,
    )
    model.embed_tokens = Embedding()
    model.layers = nn.ModuleList([Layer()])
    model.norm = Norm()
    model.rotary = SimpleNamespace(cos_sin_cache=torch.empty(0))
    model.compress_rotary_c4 = SimpleNamespace(cos_sin_cache=torch.empty(0))
    model.compress_rotary_c128 = SimpleNamespace(cos_sin_cache=torch.empty(0))
    model.attach_rope_tables_to_backend = lambda *_args, **_kwargs: None
    metadata = _StaticAttentionMetadata(
        slot_mapping=torch.zeros(1, dtype=torch.int32),
        paged_kv_indptr=torch.zeros(2, dtype=torch.int32),
        paged_kv_indices=torch.zeros(1, dtype=torch.int32),
        paged_kv_last_page_len=torch.ones(1, dtype=torch.int32),
        is_dummy=True,
    )
    backend = SimpleNamespace(
        reset_forward=lambda _metadata: None,
        prepare_dsa_metadata_for_forward=lambda _metadata: None,
        select_dsa_layer_rope=lambda *_args: None,
    )

    with forward_context(ForwardContext(backend, torch.device("cpu"), metadata, [])):
        hidden, aux_hidden = model(
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            None,
        )

    assert hidden.shape == (1, 2)
    assert aux_hidden.shape == (1, 4)
    assert torch.count_nonzero(aux_hidden) == 0
