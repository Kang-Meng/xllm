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

"""CPU behavior tests for the Kimi K3 Python vision tower."""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from xllm.python.models import kimi_k3_vit as _vit_module
from xllm.python.models.kimi_k3_vit import (
    KimiK3VisionConfig,
    KimiK3VisionModel,
    KimiK3VisionRotaryEmbedding,
    _encoder_attention,
    _get_load_balance_assignment,
    _run_data_parallel_vision_tower,
    tpool_patch_merger,
)


def test_rope_cache_grows_with_grid_and_preserves_phases() -> None:
    rope = KimiK3VisionRotaryEmbedding(128)
    device = torch.device("cpu")
    first = rope([[2, 2, 3]], device)
    assert rope._frequencies.shape == (2, 3, 64)
    cached = rope._frequencies
    small = rope([[1, 1, 2]], device)
    assert rope._frequencies is cached
    torch.testing.assert_close(small, first[:2])
    rope([[1, 4, 2], [1, 2, 5]], device)
    assert rope._frequencies.shape == (4, 5, 64)
    torch.testing.assert_close(rope([[2, 2, 3]], device), first)
    # An independent position formula catches cache-width-dependent indexing.
    coordinates = torch.tensor([[x, y] for y in range(2) for x in range(3)])
    inv_freq = 10000.0 ** (-torch.arange(0, 128, 4).float() / 128)
    phase = coordinates.float().unsqueeze(-1) * inv_freq
    expected = torch.polar(torch.ones_like(phase), phase).transpose(1, 2).reshape(6, 64)
    torch.testing.assert_close(first, expected.repeat(2, 1))


def test_rope_empty_and_invalid_grids_do_not_allocate_cache() -> None:
    rope = KimiK3VisionRotaryEmbedding(128, max_height=8, max_width=8)
    assert rope([], torch.device("cpu")).shape == (0, 64)
    assert rope._frequencies is None
    with pytest.raises(ValueError, match="exceeds"):
        rope([[1, 9, 1]], torch.device("cpu"))
    assert rope._frequencies is None


def test_rope_default_cache_can_grow_beyond_512_patches() -> None:
    rope = KimiK3VisionRotaryEmbedding(8)
    first = rope([[1, 2, 2]], torch.device("cpu"))
    assert rope([[1, 513, 1]], torch.device("cpu")).shape == (513, 4)
    assert rope._frequencies.shape == (513, 2, 4)
    torch.testing.assert_close(rope([[1, 2, 2]], torch.device("cpu")), first)


def _tiny_config() -> dict:
    return {
        "device": "cpu",
        "dtype": "float32",
        "vision_config": {
            "patch_size": 2,
            "in_chans": 3,
            "vt_hidden_size": 4,
            "vt_intermediate_size": 8,
            "vt_num_attention_heads": 2,
            "vt_num_hidden_layers": 1,
            "qkv_hidden_size": 8,
            "init_pos_emb_height": 2,
            "init_pos_emb_width": 2,
            "init_pos_emb_time": 2,
            "merge_kernel_size": [2, 2],
            "mm_hidden_size": 4,
            "text_hidden_size": 8,
        },
    }


def test_config_reads_nested_k3_vision_fields() -> None:
    config = KimiK3VisionConfig.from_dict(_tiny_config())

    assert config.hidden_size == 4
    assert config.qkv_hidden_size == 8
    assert config.merge_kernel_size == (2, 2)
    config.validate()


def test_config_replicates_vision_weights_across_runtime_tp_ranks() -> None:
    raw_config = _tiny_config()
    raw_config["tp_size"] = 4
    raw_config["tp_rank"] = 2

    config = KimiK3VisionConfig.from_dict(raw_config)

    assert config.tp_size == 1
    assert config.tp_rank == 0
    assert config.encoder_dp_size == 4
    assert config.encoder_dp_rank == 2
    config.validate()


def test_tpool_merger_pools_time_before_spatial_reorder() -> None:
    hidden_states = torch.arange(16, dtype=torch.float32).reshape(16, 1)

    outputs = tpool_patch_merger(hidden_states, [[2, 2, 4]], (2, 2))

    assert len(outputs) == 1
    assert outputs[0].shape == (2, 4, 1)
    expected = torch.tensor([[4.0, 5.0, 8.0, 9.0], [6.0, 7.0, 10.0, 11.0]])
    torch.testing.assert_close(outputs[0].squeeze(-1), expected)


def test_tiny_vision_model_projects_each_media_item() -> None:
    torch.manual_seed(7)
    model = KimiK3VisionModel(_tiny_config())
    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, -0.1, 0.1)
    pixel_values = torch.randn(8, 3, 2, 2)
    grid_thws = torch.tensor([[2, 2, 2]], dtype=torch.int64)

    outputs = model(pixel_values, grid_thws)

    assert len(outputs) == 1
    # Temporal pooling removes the two-frame axis before projection.
    assert outputs[0].shape == (1, 8)
    assert torch.isfinite(outputs[0]).all()


def test_single_encoder_rank_preserves_media_attention_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_boundaries: list[list[int]] = []

    def capture_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: Sequence[int],
    ) -> torch.Tensor:
        captured_boundaries.append(list(cu_seqlens))
        return _encoder_attention(query, key, value, cu_seqlens)

    monkeypatch.setattr(_vit_module, "_encoder_attention", capture_attention)
    config = _tiny_config()
    config["vision_config"]["merge_kernel_size"] = [1, 1]
    model = KimiK3VisionModel(config)
    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, -0.1, 0.1)
    pixel_values = torch.randn(3, 3, 2, 2)
    grid_thws = torch.tensor([[1, 1, 1], [1, 1, 2]], dtype=torch.int64)

    outputs = model(pixel_values, grid_thws)

    assert captured_boundaries == [[0, 1, 3]]
    assert [output.shape[0] for output in outputs] == [1, 2]


def test_encoder_data_mode_matches_vllm_greedy_assignment() -> None:
    assignment, image_counts, rank_loads = _get_load_balance_assignment(
        [1000, 100, 200, 50],
        num_ranks=2,
    )

    assert assignment == [0, 2, 1, 3]
    assert image_counts == [1, 3]
    assert rank_loads == [1000, 350]


def test_encoder_data_parallel_reuses_mainline_tp_group(monkeypatch) -> None:
    class _VisionTower:
        merge_kernel_size = (1, 1)
        config = MagicMock(hidden_size=2)

        def __call__(self, pixel_values, _grids):
            return [pixel_values]

    gathered = torch.tensor([[10.0, 11.0], [20.0, 21.0]], dtype=torch.float32).unsqueeze(1)

    def all_gather(tensor, *, dim, world_size):
        assert tensor.shape == (1, 1, 2)
        assert dim == 0
        assert world_size == 2
        return gathered

    monkeypatch.setattr(_vit_module.distributed, "tp_all_gather", all_gather)
    outputs = _run_data_parallel_vision_tower(
        _VisionTower(),
        torch.tensor([[10.0, 11.0], [20.0, 21.0]]).unsqueeze(1),
        [[1, 1, 1], [1, 1, 1]],
        rank=0,
        world_size=2,
    )

    assert [output.squeeze(0).tolist() for output in outputs] == [
        [[10.0, 11.0]],
        [[20.0, 21.0]],
    ]


def test_packed_attention_keeps_media_items_independent() -> None:
    torch.manual_seed(11)
    query, key, value = [torch.randn(5, 2, 4) for _ in range(3)]
    packed = _encoder_attention(query, key, value, [0, 2, 5])
    expected = []
    for start, end in [(0, 2), (2, 5)]:
        attended = F.scaled_dot_product_attention(
            query[start:end].transpose(0, 1),
            key[start:end].transpose(0, 1),
            value[start:end].transpose(0, 1),
            is_causal=False,
        )
        expected.append(attended.transpose(0, 1))
    torch.testing.assert_close(packed, torch.cat(expected))


@pytest.mark.parametrize("boundaries", [[], [1, 5], [0, 4], [0, 3, 2, 5], [0, 2, 2, 5]])
def test_packed_attention_rejects_invalid_boundaries(boundaries: list[int]) -> None:
    tensor = torch.randn(5, 2, 4)
    with pytest.raises(ValueError, match="cu_seqlens"):
        _encoder_attention(tensor, tensor, tensor, boundaries)


def test_packed_attention_supports_empty_encoder_rank() -> None:
    tensor = torch.empty(0, 2, 4)
    output = _encoder_attention(tensor, tensor, tensor, [0])
    assert output.shape == tensor.shape
    assert output.dtype == tensor.dtype


def test_multiple_media_match_separate_vision_forward() -> None:
    torch.manual_seed(17)
    config = _tiny_config()
    config["vision_config"]["merge_kernel_size"] = [1, 1]
    model = KimiK3VisionModel(config)
    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, -0.1, 0.1)
    pixels = torch.randn(3, 3, 2, 2)
    packed = model(pixels, [[1, 1, 1], [1, 1, 2]])
    separate = [model(pixels[:1], [[1, 1, 1]])[0], model(pixels[1:], [[1, 1, 2]])[0]]
    for actual, expected in zip(packed, separate):
        torch.testing.assert_close(actual, expected)


def test_vision_weight_loading_accepts_checkpoint_projector_aliases() -> None:
    model = KimiK3VisionModel(_tiny_config())
    checkpoint = {}
    expected = {}
    for name, parameter in model.named_parameters():
        expected[name] = torch.full_like(parameter, 0.125)
        checkpoint_name = name.replace("mm_projector.linear_1.", "mm_projector.proj.0.").replace(
            "mm_projector.linear_2.", "mm_projector.proj.2."
        )
        checkpoint[f"model.{checkpoint_name}"] = expected[name]
    state_dict = MagicMock()
    state_dict.has.side_effect = checkpoint.__contains__
    state_dict.get_tensor.side_effect = checkpoint.__getitem__
    loaded = model.load_weights([state_dict], tp_rank=0, tp_size=1)
    assert loaded == set(expected)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, expected[name])


@pytest.mark.parametrize("missing", [True, False])
def test_vision_weight_errors_identify_loader_and_encoder_rank(missing: bool) -> None:
    config = _tiny_config()
    config.update(tp_size=4, tp_rank=2)
    model = KimiK3VisionModel(config)
    name, _ = next(model.named_parameters())
    state_dict = MagicMock()
    state_dict.has.side_effect = lambda key: not missing and key == name
    state_dict.get_tensor.return_value = torch.empty(1)
    expected_error = KeyError if missing else ValueError
    with pytest.raises(expected_error, match="loader TP 2/4, vision TP 0/1, encoder DP 2/4"):
        model.load_weights([state_dict], tp_rank=2, tp_size=4)


def test_config_reads_flat_model_args_without_using_text_dimensions() -> None:
    config = KimiK3VisionConfig.from_dict(
        {
            "mm_hidden_size": 16,
            "mm_intermediate_size": 32,
            "mm_num_hidden_layers": 2,
            "mm_num_attention_heads": 12,
            "mm_num_channels": 1,
            "mm_patch_size": 4,
            "mm_init_pos_emb_height": 3,
            "mm_init_pos_emb_width": 6,
            "mm_init_pos_emb_time": 2,
            "mm_spatial_merge_size": 3,
            "mm_projection_dim": 48,
            "mm_projector_type": "patchmergerv2",
            "mm_projector_hidden_act": "gelu",
            "mm_layer_norm_eps": 1e-4,
            "hidden_size": 96,
            "num_hidden_layers": 64,
            "intermediate_size": 192,
            "tp_size": 4,
            "tp_rank": 2,
        }
    )
    config.validate()
    assert (config.hidden_size, config.intermediate_size, config.num_hidden_layers) == (16, 32, 2)
    assert config.patch_size == 4 and config.in_channels == 1
    assert (config.init_pos_emb_height, config.init_pos_emb_width, config.init_pos_emb_time) == (3, 6, 2)
    assert config.merge_kernel_size == (3, 3)
    assert config.text_hidden_size == 48 and config.projector_ln_eps == 1e-4
    assert (config.encoder_dp_size, config.encoder_dp_rank) == (4, 2)
    nested = _tiny_config()
    nested.update(mm_hidden_size=999, hidden_size=777)
    assert KimiK3VisionConfig.from_dict(nested).hidden_size == 4


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_encoder_norms_keep_released_checkpoint_epsilon(dtype: torch.dtype) -> None:
    config = _tiny_config()
    config["dtype"] = dtype
    model = KimiK3VisionModel(config)
    encoder = model.vision_tower.encoder
    assert all(norm.eps is None for norm in (encoder.blocks[0].norm0, encoder.blocks[0].norm1, encoder.final_layernorm))
    assert model.mm_projector.post_norm.eps == 1e-5
    x = torch.full((1, 4), 0.01, dtype=dtype)
    with torch.no_grad():
        encoder.blocks[0].norm0.weight.fill_(1)
    reference = torch.nn.RMSNorm(4, dtype=dtype)(x)
    torch.testing.assert_close(encoder.blocks[0].norm0(x), reference)


def test_flat_model_args_unset_fields_do_not_replace_vision_defaults() -> None:
    config = KimiK3VisionConfig.from_dict(
        {
            "mm_hidden_size": 1024,
            "mm_num_hidden_layers": 0,
            "mm_num_attention_heads": 0,
            "mm_spatial_merge_size": 0,
            "mm_km_merge_kernel_size": 0,
            "mm_projector_type": "",
            "hidden_size": 7168,
        }
    )
    config.validate()
    assert (config.hidden_size, config.text_hidden_size) == (1024, 7168)
    assert config.num_hidden_layers == 27 and config.num_attention_heads == 12
    assert config.merge_kernel_size == (2, 2) and config.mm_projector_type == "patchmergerv2"


@pytest.mark.parametrize("shape", [(1, 3, 4, 4), (1, 1, 2, 2), (3, 2, 2)])
def test_patch_embed_rejects_whole_images_and_invalid_patch_shapes(shape: tuple[int, ...]) -> None:
    model = KimiK3VisionModel(_tiny_config())
    with pytest.raises(ValueError, match="pre-patched"):
        model.vision_tower.patch_embed(torch.ones(shape), [[1, 2, 2]])


@pytest.mark.parametrize("grid", [[[1, 0, 2]], [[1, 2, 0]], [[0, 2, 2]]])
def test_position_embedding_rejects_nonpositive_grids_before_interpolation(grid: list[list[int]]) -> None:
    model = KimiK3VisionModel(_tiny_config())
    with pytest.raises(ValueError, match="positive"):
        model.vision_tower.patch_embed.pos_emb(torch.zeros(1, 4), grid)


def test_rope_rejects_mismatched_token_count() -> None:
    with pytest.raises(ValueError, match="shapes do not match"):
        _vit_module._apply_rope(torch.ones(2, 1, 4), torch.ones(2, 1, 4), torch.ones(1, 2, dtype=torch.complex64))


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_encoder_dp_tracks_pooled_video_rows_and_empty_ranks(monkeypatch: pytest.MonkeyPatch, rank: int) -> None:
    grids = [[2, 2, 2], [1, 2, 2], [1, 2, 4]]
    pixels = torch.cat([torch.full((8, 1), 10.0), torch.full((4, 1), 20.0), torch.full((8, 1), 30.0)])

    class Tower:
        merge_kernel_size = (2, 2)
        config = MagicMock(hidden_size=1)

        def __call__(self, x: torch.Tensor, local_grids: Sequence[Sequence[int]]) -> list[torch.Tensor]:
            return tpool_patch_merger(x, local_grids, self.merge_kernel_size)

    # Patch-load assignment: rank0=item0, rank1=item2, rank2=item1, rank3=empty.
    # Pooled rows: [1, 2, 1, 0], independent of each item's frame count.
    gathered = torch.full((8, 4, 1), float("nan"))
    gathered[0] = 10
    gathered[2:4] = 30
    gathered[4] = 20

    def all_gather(x: torch.Tensor, *, dim: int, world_size: int) -> torch.Tensor:
        assert x.shape == (2, 4, 1) and dim == 0 and world_size == 4
        expected = {0: (1, 10), 1: (2, 30), 2: (1, 20), 3: (0, 0)}[rank]
        if expected[0]:
            torch.testing.assert_close(x[: expected[0]], torch.full_like(x[: expected[0]], expected[1]))
        return gathered

    monkeypatch.setattr(_vit_module.distributed, "tp_all_gather", all_gather)
    result = _run_data_parallel_vision_tower(Tower(), pixels, grids, rank, 4)
    assert [x.shape[0] for x in result] == [1, 1, 2]
    for x, label in zip(result, [10, 20, 30]):
        torch.testing.assert_close(x, torch.full_like(x, label))
        assert torch.isfinite(x).all()


def test_single_encoder_rank_keeps_original_pixels_and_media_order() -> None:
    grids = [[1, 2, 2], [2, 2, 2]]
    pixels = torch.cat([torch.full((4, 1), 10.0), torch.full((8, 1), 20.0)])

    class Tower:
        def __call__(self, x: torch.Tensor, host_grids: Sequence[Sequence[int]]) -> list[torch.Tensor]:
            assert x is pixels and host_grids == [(1, 2, 2), (2, 2, 2)]
            return tpool_patch_merger(x, host_grids, (2, 2))

    result = _run_data_parallel_vision_tower(Tower(), pixels, grids, 0, 1)
    assert [x.shape[0] for x in result] == [1, 1]
    torch.testing.assert_close(result[0], torch.full_like(result[0], 10))
    torch.testing.assert_close(result[1], torch.full_like(result[1], 20))


def test_encoder_dp_rejects_wrong_tower_row_count_before_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    class Tower:
        merge_kernel_size = (2, 2)
        config = MagicMock(hidden_size=1)

        def __call__(self, x: torch.Tensor, grids: Sequence[Sequence[int]]) -> list[torch.Tensor]:
            return [torch.ones(2, 4, 1)]

    gather = MagicMock()
    monkeypatch.setattr(_vit_module.distributed, "tp_all_gather", gather)
    with pytest.raises(ValueError, match="pooled grids"):
        _run_data_parallel_vision_tower(Tower(), torch.ones(8, 1), [[2, 2, 2]], 0, 2)
    gather.assert_not_called()


def test_vision_forward_converts_grid_tensor_to_host_once(monkeypatch: pytest.MonkeyPatch) -> None:
    model = KimiK3VisionModel(_tiny_config())
    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, -0.1, 0.1)
    grids = torch.tensor([[1, 2, 2]])
    original_cpu = torch.Tensor.cpu
    calls = []

    def record_cpu(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        calls.append(tensor.shape)
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", record_cpu)
    model(torch.ones(4, 3, 2, 2), grids)
    assert calls == [torch.Size([1, 3])]
