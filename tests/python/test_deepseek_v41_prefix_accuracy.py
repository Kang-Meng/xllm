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

"""NPU forward precision after completed prefix blocks are aliased.

Uses the existing tiny six-layer model geometry and production CSA2 backend.
A source forwards a complete C2-aligned prefix; a fresh request aliases its
SWA tail/C1/C2 (including INT8 index codes and FP16 scales) and allocates
private suffix blocks. Compares suffix logits to an independently bound
full-prefill baseline and checks shared tensors remain unchanged. This does
NOT exercise the C++ allocator, W8A8 checkpoint or full serving topology;
those still require separate integration/performance runs.

Wide-window/block128 cases verify that binding physical SWA pages updates
metadata strides without changing the attention window. The default metadata
path and the optional host physical-slot interface must address the same pages.
No cache spec or production builder is overridden by these tests.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.python.test_deepseek_v41_dspark import _build_checkpoint, _config_dict, _load
from tests.python.test_deepseek_v41_model import _DEEPSEEK_V41_FORWARD_CONFIG, _make_csa2_backend, _random_init_v41
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.csa2_attention import Csa2AttentionBackend
from xllm.python.attention.csa_attention import _scatter_by_slot
from xllm.python.attention.dsa_metadata import DSA_CACHE_SLIDING_WINDOW, DSA_CACHE_TOKEN
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.models.deepseek_v41 import DeepseekV41Config, DeepseekV41ForCausalLM
from xllm.python.models.deepseek_v41_dspark import DeepseekV41DSparkForCausalLM

_NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()
npu_only = pytest.mark.skipif(not _NPU_AVAILABLE, reason="restored-prefix logits require a healthy NPU")
_BLOCK_SIZE = 128
_COMPRESS_STATE_ROWS = 32
_SHORT_PREFIX_CASES = [
    pytest.param(128, 256, [0, 0, 1, 1, 1, 1], id="c1-prefix128-window256"),
    pytest.param(256, 512, [0, 0, 2, 2, 1, 1], id="c1-c2-prefix256-window512"),
]


def _model(config: dict[str, Any]) -> DeepseekV41ForCausalLM:
    torch.manual_seed(41)
    model = DeepseekV41ForCausalLM(config)
    _random_init_v41(model)
    # Distinct finite FP8 table rows make the Engram test sensitive to incorrect
    # hash history; constant embeddings would hide hash-ID corruption.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.dtype == torch.uint8 and not name.endswith("weight_scale"):
                values = torch.arange(parameter.numel(), device=parameter.device) % 24 + 0x28
                parameter.copy_(values.reshape(parameter.shape).to(torch.uint8))
    return model


def _caches(model: DeepseekV41ForCausalLM, total: int) -> list[LayerCache]:
    cfg = model.cfg
    count = (total + _BLOCK_SIZE - 1) // _BLOCK_SIZE + 3
    caches: list[LayerCache] = []
    for layer in range(cfg.n_layers):
        swa = torch.zeros(count, _BLOCK_SIZE, cfg.head_dim, dtype=torch.bfloat16, device="npu:0")
        ratio = cfg.compress_ratios[layer]
        source = layer in cfg.kv_source_layer_ids
        # State blocks follow SWA block IDs, but use the production ring-32 stride.
        state_shape = (count, _COMPRESS_STATE_ROWS, cfg.head_dim)
        caches.append(
            LayerCache(
                key=torch.zeros(count, _BLOCK_SIZE, cfg.head_dim, dtype=swa.dtype, device=swa.device)
                if source
                else None,
                value=None,
                index=torch.zeros(count, _BLOCK_SIZE, cfg.index_head_dim, dtype=torch.int8, device=swa.device)
                if source
                else None,
                swa=swa,
                compress_kv_state=torch.zeros(state_shape, dtype=torch.float32, device=swa.device)
                if source and ratio > 1
                else None,
                compress_score_state=torch.zeros(state_shape, dtype=torch.float32, device=swa.device)
                if source and ratio > 1
                else None,
                indexer_scale=torch.zeros(count, _BLOCK_SIZE, 1, dtype=torch.float16, device=swa.device)
                if source
                else None,
            )
        )
    return caches


def _shared_prefix_views(cache: LayerCache, prefix: int, ratio: int) -> dict[str, torch.Tensor]:
    """Views of every completed shared block, with each plane's own row stride."""
    views: dict[str, torch.Tensor] = {}
    swa_count = prefix // _BLOCK_SIZE
    for name in ("swa", "compress_kv_state", "compress_score_state"):
        tensor = getattr(cache, name)
        if tensor is not None:
            views[name] = tensor[1 : 1 + swa_count]
    if ratio > 0:
        token_count = prefix // (_BLOCK_SIZE * ratio)
        for name in ("key", "index", "indexer_scale"):
            tensor = getattr(cache, name)
            if tensor is not None:
                views[name] = tensor[1 : 1 + token_count]
    return views


def _tables(total: int, *, prefix: int = 0, retain_swa_history: bool = False) -> list[torch.Tensor]:
    tables = []
    for ratio in (1, 2, 1):
        span = _BLOCK_SIZE * ratio
        count = (total + span - 1) // span
        table = torch.arange(1, count + 1, dtype=torch.int32).view(1, -1)
        if prefix:
            shared = prefix // span
            # Leave one unused block between source and private continuation.
            table[0, shared:] += 1
        tables.append(table)
    if prefix and not retain_swa_history:
        # SWA readers need only the last window; logical placeholders keep
        # absolute positions stable, matching the composite export contract.
        tables[0][0, : prefix // _BLOCK_SIZE - 1] = -1
    return tables


def _swa_slots(table: torch.Tensor, start: int, count: int) -> list[int]:
    """Host slots supplied by the framework for physical block128 SWA pages."""
    assert table.device.type == "cpu" and table.size(0) == 1
    positions = torch.arange(start, start + count, dtype=torch.int64)
    columns = (positions // _BLOCK_SIZE) % table.size(1)
    blocks = table[0, columns].to(torch.int64)
    assert (blocks >= 0).all()
    return (blocks * _BLOCK_SIZE + positions % _BLOCK_SIZE).tolist()


def _short_prefix_config(window: int, ratios: list[int]) -> dict[str, Any]:
    return {
        **_DEEPSEEK_V41_FORWARD_CONFIG,
        "compress_ratios": ratios,
        "window_size": window,
        "sliding_window": window,
        "max_position_embeddings": 1024,
        "engram_layer_ids": [],
        "engram_num_embeddings": [],
        "num_nextn_predict_layers": 0,
        "device": "npu:0",
        "dtype": torch.bfloat16,
    }


def _short_prefix_tables(total: int, ratios: list[int], *, prefix: int = 0) -> list[torch.Tensor]:
    tables = _tables(total, prefix=prefix, retain_swa_history=True)
    return tables if 2 in ratios else [tables[0], tables[2]]


def _step(
    model: DeepseekV41ForCausalLM,
    backend: Csa2AttentionBackend,
    caches: list[LayerCache],
    ids: torch.Tensor,
    start: int,
    tables: list[torch.Tensor],
    *,
    spec_verify: bool = False,
    captured: list[torch.Tensor] | None = None,
    new_cache_slots: list[int] | None = None,
) -> torch.Tensor:
    metadata = SimpleNamespace(
        is_prefill=start == 0 and not spec_verify,
        is_chunked_prefill=start > 0 and not spec_verify,
        is_mixed=False,
        is_spec_verify=spec_verify,
        multi_block_tables=tables,
        kv_seq_lens_host=torch.tensor([start + ids.numel()], dtype=torch.int32),
        kv_seq_lens_host_values=[start + ids.numel()],
        q_seq_lens_host=torch.tensor([ids.numel()], dtype=torch.int32),
        dsa_metadata=None,
        new_cache_slots_host_values=new_cache_slots,
    )
    context = ForwardContext(backend, torch.device("npu:0"), metadata, caches)
    positions = torch.arange(start, start + ids.numel(), device=ids.device)
    with torch.inference_mode(), forward_context(context):
        result = model.model(ids, positions)
        if isinstance(result, tuple):
            hidden, aux = result
            if captured is not None:
                captured.append(aux.detach().clone())
        else:
            hidden = result
        return model.compute_logits(hidden, None).float()


@pytest.mark.parametrize("engram_enabled", [False, True])
@pytest.mark.parametrize("prefix", [256, 512])
@pytest.mark.parametrize("suffix", [1, 3, 129])
@npu_only
def test_prefix_alias_logits_match_full_prefill(prefix: int, suffix: int, engram_enabled: bool) -> None:
    config = {
        **_DEEPSEEK_V41_FORWARD_CONFIG,
        "window_size": _BLOCK_SIZE,
        "sliding_window": _BLOCK_SIZE,
        "max_position_embeddings": 1024,
        "device": "npu:0",
        "dtype": torch.bfloat16,
        "num_nextn_predict_layers": 0,
    }
    if not engram_enabled:
        config["engram_layer_ids"] = []
        config["engram_num_embeddings"] = []
    baseline_model, cached_model = _model(config), _model(config)
    cached_model.load_state_dict(baseline_model.state_dict())
    baseline_caches = _caches(baseline_model, prefix + suffix)
    cached_caches = _caches(cached_model, prefix + suffix)
    baseline_backend = _make_csa2_backend(baseline_model.cfg, torch.device("npu:0"))
    cached_backend = _make_csa2_backend(cached_model.cfg, torch.device("npu:0"))
    baseline_backend.bind_kv_caches(baseline_caches)
    cached_backend.bind_kv_caches(cached_caches)
    ids = (torch.arange(prefix + suffix, device="npu:0", dtype=torch.int64) * 7) % 63 + 1
    golden = _step(baseline_model, baseline_backend, baseline_caches, ids, 0, _tables(prefix + suffix))[prefix:]
    _step(cached_model, cached_backend, cached_caches, ids[:prefix], 0, _tables(prefix))

    shared_views = [
        _shared_prefix_views(cache, prefix, cached_model.cfg.compress_ratios[layer])
        for layer, cache in enumerate(cached_caches)
    ]
    saved = [{name: tensor.clone() for name, tensor in views.items()} for views in shared_views]
    hash_history = cached_model.model.engram_hash._cache.clone() if engram_enabled else None

    # A new binding must restore both index codes and scales, not retain a
    # source forward's shared runtime or silently copy a TOKEN plane.
    restored_caches = [replace(cache) for cache in cached_caches]
    for layer, (source, restored) in enumerate(zip(cached_caches, restored_caches, strict=True)):
        assert restored is not source
        if layer in cached_model.cfg.kv_source_layer_ids:
            assert source.index is not None and source.index.dtype == torch.int8
            assert source.indexer_scale is not None and source.indexer_scale.dtype == torch.float16
            assert source.indexer_scale.shape == (*source.index.shape[:2], 1)
        assert restored.index is source.index
        assert restored.indexer_scale is source.indexer_scale
        views = _shared_prefix_views(restored, prefix, cached_model.cfg.compress_ratios[layer])
        for name, tensor in shared_views[layer].items():
            assert views[name].data_ptr() == tensor.data_ptr(), f"layer {layer} lost shared {name} alias"
    restored_backend = _make_csa2_backend(cached_model.cfg, torch.device("npu:0"))
    restored_backend.bind_kv_caches(restored_caches)
    actual = _step(
        cached_model, restored_backend, restored_caches, ids[prefix:], prefix, _tables(prefix + suffix, prefix=prefix)
    )
    assert torch.isfinite(actual).all()
    error = (actual - golden).abs().max().item()
    assert error <= 0.02, f"restored-cache suffix logits max_abs_error={error}"
    assert torch.equal(actual.argmax(dim=-1), golden.argmax(dim=-1))
    for layer, (source, restored, before) in enumerate(zip(cached_caches, restored_caches, saved, strict=True)):
        source_views = _shared_prefix_views(source, prefix, cached_model.cfg.compress_ratios[layer])
        restored_views = _shared_prefix_views(restored, prefix, cached_model.cfg.compress_ratios[layer])
        for name, tensor in before.items():
            assert restored_views[name].data_ptr() == source_views[name].data_ptr()
            assert torch.equal(source_views[name], tensor), f"layer {layer} mutated shared {name}"
            assert torch.equal(restored_views[name], tensor), f"layer {layer} changed restored {name}"
    if hash_history is not None:
        count = (1 + prefix // _BLOCK_SIZE) * _BLOCK_SIZE
        assert torch.equal(cached_model.model.engram_hash._cache[:count], hash_history[:count])


@pytest.mark.parametrize("suffix", [1, 129])
@npu_only
def test_short_prefix_c1_alias_logits_match_full_prefill(suffix: int) -> None:
    """Non-fused short-KV masking with a completed C1 block and private SWA.

    Python SWA tensors and DSA specs use 256 rows per block here, unlike the
    C++ allocator's block_size=128/window_blocks=2. Only C1 planes are shared;
    this is a kernel oracle, not validation of that allocator's geometry.
    """
    prefix, window = 128, 256
    total = prefix + suffix
    config = {
        **_DEEPSEEK_V41_FORWARD_CONFIG,
        "compress_ratios": [0, 0, 1, 1, 1, 1],
        "window_size": window,
        "sliding_window": window,
        "max_position_embeddings": 1024,
        "engram_layer_ids": [],
        "engram_num_embeddings": [],
        "num_nextn_predict_layers": 0,
        "device": "npu:0",
        "dtype": torch.bfloat16,
    }
    baseline, cached = _model(config), _model(config)
    cached.load_state_dict(baseline.state_dict())
    # Tiny index/rope widths select the real non-fused CSA2 path, not mocks.
    assert cached.cfg.n_layers == len(config["compress_ratios"]) == 6
    assert cached.cfg.index_head_dim == 32 and cached.cfg.qk_rope_head_dim == 16
    baseline_caches, source_caches = _caches(baseline, total), _caches(cached, total)
    for caches, poison in ((baseline_caches, 11), (source_caches, -13)):
        for layer, cache in enumerate(caches):
            cache = replace(
                cache,
                swa=torch.full(
                    (cache.swa.size(0), window, cached.cfg.head_dim),
                    poison,
                    dtype=cache.swa.dtype,
                    device=cache.swa.device,
                ),
            )
            caches[layer] = cache
            for name in ("key", "index", "indexer_scale"):
                tensor = getattr(cache, name)
                if tensor is not None:
                    tensor.fill_(abs(poison) / 32 if name == "indexer_scale" else poison)
    baseline_backend = _make_csa2_backend(baseline.cfg, torch.device("npu:0"))
    source_backend = _make_csa2_backend(cached.cfg, torch.device("npu:0"))
    baseline_backend.bind_kv_caches(baseline_caches)
    source_backend.bind_kv_caches(source_caches)
    ids = (torch.arange(total, device="npu:0", dtype=torch.int64) * 7) % 63 + 1
    swa_table = torch.arange(1, (total + window - 1) // window + 1, dtype=torch.int32).view(1, -1)
    golden = _step(baseline, baseline_backend, baseline_caches, ids, 0, [swa_table, _tables(total)[2]])[prefix:]
    _step(cached, source_backend, source_caches, ids[:prefix], 0, [swa_table[:, :1], _tables(prefix)[2]])
    shared = [
        {
            name: tensor
            for name, tensor in _shared_prefix_views(cache, prefix, cached.cfg.compress_ratios[layer]).items()
            if name != "swa"
        }
        for layer, cache in enumerate(source_caches)
    ]
    for layer in cached.cfg.kv_source_layer_ids:
        assert set(shared[layer]) == {"key", "index", "indexer_scale"}
    saved = [{name: tensor.clone() for name, tensor in views.items()} for views in shared]
    # No completed SWA block: copy only valid history into a private block.
    # Unwritten rows/blocks keep distinct finite poison to expose stray reads.
    restored_caches = [replace(cache, swa=torch.full_like(cache.swa, -17)) for cache in source_caches]
    for source, restored in zip(source_caches, restored_caches, strict=True):
        restored.swa[2, :prefix].copy_(source.swa[1, :prefix])
    restored_backend = _make_csa2_backend(cached.cfg, torch.device("npu:0"))
    restored_backend.bind_kv_caches(restored_caches)
    actual = _step(
        cached,
        restored_backend,
        restored_caches,
        ids[prefix:],
        prefix,
        [swa_table + 1, _tables(total, prefix=prefix)[2]],
    )
    assert torch.isfinite(golden).all() and torch.isfinite(actual).all()
    error = (actual - golden).abs().max().item()
    assert error <= 0.02, f"short-prefix suffix logits max_abs_error={error}"
    assert torch.equal(actual.argmax(dim=-1), golden.argmax(dim=-1))
    for layer, (cache, before) in enumerate(zip(restored_caches, saved, strict=True)):
        views = _shared_prefix_views(cache, prefix, cached.cfg.compress_ratios[layer])
        for name, tensor in before.items():
            assert views[name].data_ptr() == shared[layer][name].data_ptr()
            assert torch.equal(views[name], tensor), f"layer {layer} mutated shared {name}"


@pytest.mark.parametrize("prefix,window,ratios", _SHORT_PREFIX_CASES)
@pytest.mark.parametrize("suffix", [1, 129])
def test_short_prefix_physical_swa_slot_contract(prefix: int, window: int, ratios: list[int], suffix: int) -> None:
    """Real CPU metadata/scatter/gather contract, not an allocator/logits oracle.

    Bound SWA descriptors use the physical block128 width, not the attention
    window. Default slot generation must match the optional explicit host
    slots while preserving shared history and private suffix writes.
    """
    total = prefix + suffix
    cfg = DeepseekV41Config.from_dict(_short_prefix_config(window, ratios))
    backend = _make_csa2_backend(cfg, torch.device("cpu"))
    tables = _short_prefix_tables(total, ratios, prefix=prefix)
    source_tables = _short_prefix_tables(prefix, ratios)
    for table, source in zip(tables, source_tables, strict=True):
        assert torch.equal(table[:, : source.size(1)], source)
        assert set(table[0, source.size(1) :].tolist()).isdisjoint(source[0].tolist())
    caches = [LayerCache(key=None, value=None, swa=torch.full((12, _BLOCK_SIZE, 8), -13.0)) for _ in ratios]
    backend.bind_kv_caches(caches)
    assert backend._swa_ring_blocks == window // _BLOCK_SIZE
    assert backend.group_infos[0].block_size == caches[0].swa.size(1) == _BLOCK_SIZE
    assert [gi.ratio for gi in backend.group_infos if gi.cache_type == DSA_CACHE_TOKEN] == (
        [2, 1] if 2 in ratios else [1]
    )
    for layer_specs in backend.caches_info:
        for spec in layer_specs:
            assert spec.block_size == _BLOCK_SIZE
    specs_before = [[vars(spec).copy() for spec in layer] for layer in backend.caches_info]
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=True,
        is_mixed=False,
        is_spec_verify=False,
        multi_block_tables=tables,
        kv_seq_lens_host=torch.tensor([total], dtype=torch.int32),
        kv_seq_lens_host_values=[total],
        q_seq_lens_host=torch.tensor([suffix], dtype=torch.int32),
        dsa_positions=torch.arange(prefix, total),
        dsa_metadata=None,
    )
    expected_slots = torch.tensor(_swa_slots(tables[0], prefix, suffix), dtype=torch.int32)
    source_metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        multi_block_tables=source_tables,
        kv_seq_lens_host=torch.tensor([prefix], dtype=torch.int32),
        kv_seq_lens_host_values=[prefix],
        q_seq_lens_host=torch.tensor([prefix], dtype=torch.int32),
        dsa_positions=torch.arange(prefix),
        dsa_metadata=None,
    )
    default_source = backend._build_dsa_metadata_for_forward(source_metadata)
    source_slots = torch.tensor(_swa_slots(source_tables[0], 0, prefix), dtype=torch.int32)
    assert torch.equal(default_source.slot_mappings[0][0], source_slots)
    default_suffix = backend._build_dsa_metadata_for_forward(metadata)
    assert torch.equal(default_suffix.slot_mappings[0][0], expected_slots)
    metadata.new_cache_slots_host_values = expected_slots.tolist()
    explicit = backend._build_dsa_metadata_for_forward(metadata)
    for layer, cache in enumerate(caches):
        mapping = backend._resolve_cache_mapping(layer, ratios[layer])
        read_table = explicit.block_tables[layer][mapping.ori_cache_idx]
        slots = explicit.slot_mappings[layer][mapping.ori_cache_idx]
        assert torch.equal(read_table, tables[0]), "short history table must not be expanded or shifted"
        assert torch.equal(slots, expected_slots)
        history_positions = torch.arange(prefix)
        history = torch.arange(prefix * 8, dtype=cache.swa.dtype).view(prefix, 8)
        _scatter_by_slot(cache.swa, torch.tensor(_swa_slots(source_tables[0], 0, prefix)), history)
        restored = replace(cache)
        assert restored.swa is cache.swa
        shared = cache.swa[1 : 1 + prefix // _BLOCK_SIZE]
        pointer, before = shared.data_ptr(), shared.clone()
        updates = torch.arange(suffix * 8, dtype=cache.swa.dtype).view(suffix, 8) + 4096
        _scatter_by_slot(restored.swa, slots, updates)
        assert shared.data_ptr() == pointer and restored.swa.data_ptr() == cache.swa.data_ptr()
        assert torch.equal(shared, before)
        flat = restored.swa.view(-1, 8)
        assert torch.equal(backend._gather_swa_positions(flat, read_table, 0, history_positions, _BLOCK_SIZE), history)
        assert torch.equal(
            backend._gather_swa_positions(flat, read_table, 0, torch.arange(prefix, total), _BLOCK_SIZE), updates
        )
        if ratios[layer] > 1:
            state_slots = backend._v41_state_slots(
                explicit, layer, mapping.ori_cache_idx, 0, torch.arange(prefix, total), _COMPRESS_STATE_ROWS
            )
            expected_state = (expected_slots.long() // _BLOCK_SIZE) * _COMPRESS_STATE_ROWS
            expected_state += torch.arange(prefix, total) % _COMPRESS_STATE_ROWS
            assert torch.equal(state_slots, expected_state)
    assert specs_before == [[vars(spec).copy() for spec in layer] for layer in backend.caches_info]


@pytest.mark.parametrize("prefix,window,ratios", _SHORT_PREFIX_CASES)
@pytest.mark.parametrize("suffix", [1, 129])
@npu_only
def test_short_prefix_shared_swa_alias_logits_match_full_prefill(
    prefix: int, window: int, ratios: list[int], suffix: int
) -> None:
    """Block128/full-history aliases using default production slot metadata.

    SWA, completed C1/C2 and index/scale planes alias the source unchanged;
    only suffix IDs are private. No historical rows are copied. This is a
    synthetic CSA2 component oracle, NOT execution of the C++ allocator.
    Bound descriptors must address physical blocks independently of window.
    """
    total = prefix + suffix
    config = _short_prefix_config(window, ratios)
    baseline, cached = _model(config), _model(config)
    cached.load_state_dict(baseline.state_dict())
    assert cached.cfg.n_layers == len(ratios) == 6
    assert cached.cfg.index_head_dim == 32 and cached.cfg.qk_rope_head_dim == 16
    baseline_caches, source_caches = _caches(baseline, total), _caches(cached, total)
    for caches, poison in ((baseline_caches, 11), (source_caches, -13)):
        for cache in caches:
            assert cache.swa.size(1) == _BLOCK_SIZE
            for name in ("swa", "key", "index", "indexer_scale", "compress_kv_state", "compress_score_state"):
                tensor = getattr(cache, name)
                if tensor is not None:
                    tensor.fill_(abs(poison) / 32 if name == "indexer_scale" else poison)
    full_tables = _short_prefix_tables(total, ratios)
    source_tables = _short_prefix_tables(prefix, ratios)
    private_tables = _short_prefix_tables(total, ratios, prefix=prefix)
    for table, source in zip(private_tables, source_tables, strict=True):
        shared_count = source.size(1)
        assert torch.equal(table[:, :shared_count], source), "all historical IDs must survive"
        assert (table >= 0).all()
        assert set(table[0, shared_count:].tolist()).isdisjoint(source[0].tolist())
    baseline_backend = _make_csa2_backend(baseline.cfg, torch.device("npu:0"))
    source_backend = _make_csa2_backend(cached.cfg, torch.device("npu:0"))
    baseline_backend.bind_kv_caches(baseline_caches)
    source_backend.bind_kv_caches(source_caches)
    ids = (torch.arange(total, device="npu:0", dtype=torch.int64) * 7) % 63 + 1
    golden = _step(
        baseline,
        baseline_backend,
        baseline_caches,
        ids,
        0,
        full_tables,
    )[prefix:]
    _step(
        cached,
        source_backend,
        source_caches,
        ids[:prefix],
        0,
        source_tables,
    )
    shared = [_shared_prefix_views(cache, prefix, ratios[layer]) for layer, cache in enumerate(source_caches)]
    for layer in cached.cfg.kv_source_layer_ids:
        assert {"swa", "key", "index", "indexer_scale"} <= shared[layer].keys()
    saved = [{name: (tensor.data_ptr(), tensor.clone()) for name, tensor in views.items()} for views in shared]
    restored_caches = [replace(cache) for cache in source_caches]
    for layer, (source, restored) in enumerate(zip(source_caches, restored_caches, strict=True)):
        assert restored is not source
        for name in shared[layer]:
            assert getattr(restored, name) is getattr(source, name), f"layer {layer} copied historical {name}"
    restored_backend = _make_csa2_backend(cached.cfg, torch.device("npu:0"))
    restored_backend.bind_kv_caches(restored_caches)
    assert restored_backend._swa_ring_blocks == window // _BLOCK_SIZE
    assert restored_backend.group_infos[0].block_size == _BLOCK_SIZE
    actual = _step(
        cached,
        restored_backend,
        restored_caches,
        ids[prefix:],
        prefix,
        private_tables,
    )
    assert torch.isfinite(golden).all() and torch.isfinite(actual).all()
    error = (actual - golden).abs().max().item()
    assert error <= 0.02, f"shared block128/window{window} suffix logits max_abs_error={error}"
    assert torch.equal(actual.argmax(dim=-1), golden.argmax(dim=-1))
    for layer, (source, restored, before) in enumerate(zip(source_caches, restored_caches, saved, strict=True)):
        for cache in (source, restored):
            views = _shared_prefix_views(cache, prefix, ratios[layer])
            for name, (pointer, content) in before.items():
                assert views[name].data_ptr() == pointer, f"layer {layer} lost shared {name} pointer"
                assert torch.equal(views[name], content), f"layer {layer} mutated shared {name}"


@pytest.mark.parametrize("prefix", [256, 512])
@pytest.mark.parametrize("accepted", [1, 2, 6])
@pytest.mark.parametrize("engram_enabled", [False, True])
@npu_only
def test_prefix_verify_rejection_replays_only_committed_rows(prefix: int, accepted: int, engram_enabled: bool) -> None:
    """Actual CSA2 verify writes followed by rollback, not mocked acceptance.

    The six-layer synthetic target uses C1/C2 and the ring-32 residual state.
    This is a component oracle, not full-checkpoint DSpark acceptance.
    """
    config = {
        **_DEEPSEEK_V41_FORWARD_CONFIG,
        "window_size": _BLOCK_SIZE,
        "sliding_window": _BLOCK_SIZE,
        "max_position_embeddings": 1024,
        "device": "npu:0",
        "dtype": torch.bfloat16,
        "num_nextn_predict_layers": 0,
    }
    if not engram_enabled:
        config["engram_layer_ids"] = []
        config["engram_num_embeddings"] = []
    full, restored = _model(config), _model(config)
    restored.load_state_dict(full.state_dict())
    total = prefix + 140
    full_caches, restored_caches = _caches(full, total), _caches(restored, total)
    full_backend = _make_csa2_backend(full.cfg, torch.device("npu:0"))
    restored_backend = _make_csa2_backend(restored.cfg, torch.device("npu:0"))
    full_backend.bind_kv_caches(full_caches)
    restored_backend.bind_kv_caches(restored_caches)
    ids = (torch.arange(total, device="npu:0", dtype=torch.int64) * 7) % 63 + 1
    _step(restored, restored_backend, restored_caches, ids[:prefix], 0, _tables(prefix))
    saved = [
        {
            name: view.clone()
            for name, view in _shared_prefix_views(cache, prefix, restored.cfg.compress_ratios[layer]).items()
        }
        for layer, cache in enumerate(restored_caches)
    ]
    private_tables = _tables(total, prefix=prefix)
    # Complete the prompt through an odd C2 boundary, at the end of an SWA
    # block. The next verify spans both a block boundary and a ring wrap.
    start = prefix + 127
    _step(full, full_backend, full_caches, ids[:start], 0, _tables(start))
    _step(restored, restored_backend, restored_caches, ids[prefix:start], prefix, private_tables)
    proposal = ids[start : start + 6].clone()
    proposal[accepted:] = proposal[accepted:].roll(1) + 1
    expected_verify = _step(full, full_backend, full_caches, ids[start : start + accepted], start, _tables(total))
    actual_verify = _step(
        restored, restored_backend, restored_caches, proposal, start, private_tables, spec_verify=True
    )
    torch.testing.assert_close(actual_verify[:accepted], expected_verify, atol=0.02, rtol=0)
    # Roll back the logical cursor exactly as the V4 accepted-token path does.
    # Rejected rows can stay in private planes, but must never affect replay.
    resume = start + accepted
    expected = _step(full, full_backend, full_caches, ids[resume : resume + 2], resume, _tables(total))
    actual = _step(restored, restored_backend, restored_caches, ids[resume : resume + 2], resume, private_tables)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0)
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))
    for layer, (cache, before) in enumerate(zip(restored_caches, saved, strict=True)):
        current = _shared_prefix_views(cache, prefix, restored.cfg.compress_ratios[layer])
        for name, value in before.items():
            assert torch.equal(current[name], value), f"verify mutated shared layer {layer} {name}"


@pytest.mark.parametrize("prefix", [256, 512])
@npu_only
def test_dspark_context_prefix_alias_uses_target_capture_and_private_suffix(prefix: int) -> None:
    """Real target capture -> existing draft context projection -> SWA aliases.

    Reuses V4's joint block-ID convention. Neither source hidden states nor
    draft context rows for a cache hit are recomputed for the restored prefix.
    """
    config = {
        **_DEEPSEEK_V41_FORWARD_CONFIG,
        "window_size": _BLOCK_SIZE,
        "sliding_window": _BLOCK_SIZE,
        "max_position_embeddings": 1024,
        "engram_layer_ids": [],
        "engram_num_embeddings": [],
        "layers_to_capture": [4, 5],
        "enable_graph_aux_hidden_states": False,
        "device": "npu:0",
        "dtype": torch.bfloat16,
        "num_nextn_predict_layers": 0,
    }
    total = prefix + 3
    target, cold_target = _model(config), _model(config)
    cold_target.load_state_dict(target.state_dict())
    source_caches, cold_caches = _caches(target, total), _caches(cold_target, total)
    source_backend = _make_csa2_backend(target.cfg, torch.device("npu:0"))
    cold_backend = _make_csa2_backend(cold_target.cfg, torch.device("npu:0"))
    source_backend.bind_kv_caches(source_caches)
    cold_backend.bind_kv_caches(cold_caches)
    ids = (torch.arange(total, device="npu:0", dtype=torch.int64) * 7) % 63 + 1
    prefix_aux: list[torch.Tensor] = []
    cold_prefix_aux: list[torch.Tensor] = []
    # Same genuine multi-forward geometry on the cache-off control. A full
    # vs split comparison additionally measures BF16 batch-shape rounding;
    # this test isolates alias restoration and draft context ownership.
    _step(cold_target, cold_backend, cold_caches, ids[:prefix], 0, _tables(prefix), captured=cold_prefix_aux)
    _step(target, source_backend, source_caches, ids[:prefix], 0, _tables(prefix), captured=prefix_aux)
    restored_backend = _make_csa2_backend(target.cfg, torch.device("npu:0"))
    restored_backend.bind_kv_caches(source_caches)
    suffix_aux: list[torch.Tensor] = []
    cold_suffix_aux: list[torch.Tensor] = []
    golden = _step(
        cold_target, cold_backend, cold_caches, ids[prefix:], prefix, _tables(total), captured=cold_suffix_aux
    )
    actual = _step(
        target,
        restored_backend,
        source_caches,
        ids[prefix:],
        prefix,
        _tables(total, prefix=prefix),
        captured=suffix_aux,
    )
    assert len(cold_prefix_aux) == len(prefix_aux) == len(suffix_aux) == len(cold_suffix_aux) == 1
    torch.testing.assert_close(prefix_aux[0], cold_prefix_aux[0], atol=0, rtol=0)
    torch.testing.assert_close(suffix_aux[0], cold_suffix_aux[0], atol=0, rtol=0)
    torch.testing.assert_close(actual, golden, atol=0.02, rtol=0)
    draft_config = _config_dict(
        hidden_size=config["hidden_size"],
        window_size=_BLOCK_SIZE,
        sliding_window=_BLOCK_SIZE,
        max_position_embeddings=1024,
        dspark_target_layer_ids=[4, 5],
        device="npu:0",
        dtype=torch.bfloat16,
    )
    draft = DeepseekV41DSparkForCausalLM(draft_config)
    _load(draft, _build_checkpoint(draft.cfg, lambda key, shape: torch.randn(shape) * 0.05))
    draft_caches = [
        LayerCache(key=None, value=None, swa=torch.zeros_like(source_caches[0].swa))
        for _ in range(draft.cfg.num_draft_layers)
    ]
    cold_draft_caches = [replace(cache, swa=torch.zeros_like(cache.swa)) for cache in draft_caches]
    positions = torch.arange(total, device="npu:0", dtype=torch.int64)

    def slots(table: torch.Tensor, begin: int, end: int) -> torch.Tensor:
        pos = positions[begin:end]
        return (table.to(pos.device)[0, pos // _BLOCK_SIZE] * _BLOCK_SIZE + pos % _BLOCK_SIZE).to(torch.int32)

    with torch.inference_mode():
        draft.model.write_context_kv(
            cold_prefix_aux[0], positions[:prefix], slots(_tables(prefix)[0], 0, prefix), cold_draft_caches
        )
        draft.model.write_context_kv(
            cold_suffix_aux[0], positions[prefix:], slots(_tables(total)[0], prefix, total), cold_draft_caches
        )
        draft.model.write_context_kv(
            prefix_aux[0], positions[:prefix], slots(_tables(prefix)[0], 0, prefix), draft_caches
        )
        saved = [cache.swa[1 : 1 + prefix // _BLOCK_SIZE].clone() for cache in draft_caches]
        restored = [replace(cache) for cache in draft_caches]
        draft.model.write_context_kv(
            suffix_aux[0], positions[prefix:], slots(_tables(total, prefix=prefix)[0], prefix, total), restored
        )
    for source, alias, cold, before in zip(draft_caches, restored, cold_draft_caches, saved, strict=True):
        assert alias.swa.data_ptr() == source.swa.data_ptr()
        assert torch.equal(source.swa[1 : 1 + prefix // _BLOCK_SIZE], before)
        # The prefix shared ID is identical for target and draft; only the
        # suffix uses the next private block, as in V4's joint pool.
        actual = alias.swa[2 + prefix // _BLOCK_SIZE, :3]
        expected = cold.swa[1 + prefix // _BLOCK_SIZE, :3]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert torch.isfinite(actual).all()

    # Run the real pure-SWA draft body on both bindings, rather than only
    # comparing the context scatter. It consumes the same joint block IDs
    # after a warm target prefill and predicts the anchor-first block once.
    query_start = total - 1
    block = draft.model.anchor_first_query_block(ids[-1:]).reshape(-1)
    query_positions = torch.arange(query_start, query_start + block.numel(), device="npu:0")

    def draft_logits(caches: list[LayerCache], tables: list[torch.Tensor]) -> torch.Tensor:
        backend = _make_csa2_backend(draft.model.layer_cfg, torch.device("npu:0"))
        backend.bind_kv_caches(caches)
        metadata = SimpleNamespace(
            is_prefill=False,
            is_chunked_prefill=False,
            is_mixed=False,
            is_spec_verify=False,
            multi_block_tables=tables,
            kv_seq_lens_host=torch.tensor([query_start + block.numel()], dtype=torch.int32),
            kv_seq_lens_host_values=[query_start + block.numel()],
            q_seq_lens_host=torch.tensor([block.numel()], dtype=torch.int32),
            dsa_metadata=None,
        )
        context = ForwardContext(backend, torch.device("npu:0"), metadata, caches)
        with torch.inference_mode(), forward_context(context):
            hidden = draft.model(block, query_positions)
            return draft.compute_logits(hidden, None).float()

    cold_logits = draft_logits(cold_draft_caches, _tables(total + 3))
    restored_logits = draft_logits(restored, _tables(total + 3, prefix=prefix))
    torch.testing.assert_close(restored_logits, cold_logits, atol=0.02, rtol=0)
    assert torch.equal(restored_logits.argmax(-1), cold_logits.argmax(-1))
    for source, before in zip(draft_caches, saved, strict=True):
        assert torch.equal(source.swa[1 : 1 + prefix // _BLOCK_SIZE], before)
