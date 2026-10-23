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

"""A3 SMLA integration: eager, CP query shards, and live decode graph lengths."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from xllm.python.attention import csa2_attention as csa2
from xllm.python.attention.csa2_attention import (
    Csa2AttentionBackend,
    Csa2QueryLayout,
    CSA2SharedRuntime,
    _rebase_qli_picks,
    _supports_smla_va,
)
from xllm.python.model_executor.dsa_cp import build_dsa_cp_context
from xllm.python.model_executor.forward_context import ForwardContext, forward_context


@pytest.fixture(autouse=True)
def _require_smla() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("smla_va requires an A3 NPU")
    if not _supports_smla_va():
        pytest.skip("smla_va is packaged for A3 only")
    if not hasattr(torch.ops.xllm_ops, "smla_va"):
        pytest.skip("smla_va native registration is required")


def _case(q_lens: list[int], device: str, ratio: int = 2, history: int = 12, topk: int = 8) -> tuple[Any, ...]:
    torch.manual_seed(42)
    heads, width = 16, 512
    capacity = max(128, ((history + max(q_lens)) // ratio + 127) // 128 * 128)
    lengths = torch.tensor(q_lens, dtype=torch.int32) + history
    q = (torch.randn(sum(q_lens), heads, width) * 0.3).bfloat16()
    kv = torch.randn(sum(q_lens), width).bfloat16()
    swa = torch.randn(len(q_lens), 128, width).bfloat16()
    compressed = torch.randn(len(q_lens), capacity, width).bfloat16()
    table = torch.arange(len(q_lens), dtype=torch.int32).view(-1, 1)
    picks = torch.full((len(q_lens), max(q_lens), topk), -1, dtype=torch.int32)
    for seq, length in enumerate(q_lens):
        for row in range(length):
            count = min(topk, (history + row + 1) // ratio)
            picks[seq, row, :count] = torch.arange(count) + 128 + length
    layer = SimpleNamespace(
        mode=SimpleNamespace(compress_ratio=ratio), layer_id=0, attn_sink=torch.linspace(-1, 1, heads).to(device)
    )
    backend = Csa2AttentionBackend(
        compress_ratios=[ratio],
        window_size=128,
        n_layers=1,
        num_heads=heads,
        attn_head_dim=width,
        index_topk=picks.size(-1),
        index_n_heads=32,
        index_head_dim=128,
        rope_head_dim=64,
        device=torch.device(device),
        dtype=torch.bfloat16,
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0],
        candidate_source_layer_id=-1,
        candidate_topk_blocks=0,
        candidate_block_size=128,
    )
    # Supply deterministic paged-cache fixtures for this attention-only test.
    backend._kv_caches = [SimpleNamespace(swa=swa.to(device))]
    backend._resolve_cache_mapping = lambda *_: SimpleNamespace(ori_cache_idx=0)
    backend._committed_capacity = lambda *_: capacity
    dsa = SimpleNamespace(seq_lens=lengths.to(device), block_tables=[[table.to(device)]])
    shared = CSA2SharedRuntime(key_cache=compressed.to(device), key_block_table=table.to(device))
    return backend, layer, dsa, shared, q.to(device), kv.to(device), picks.to(device)


def _context(backend: Csa2AttentionBackend, dsa: Any) -> ForwardContext:
    return ForwardContext(
        attention_backend=backend,
        device=torch.device("npu"),
        metadata=SimpleNamespace(dsa_metadata=dsa),
        layer_caches=[],
    )


@pytest.mark.parametrize("shared_window", [False, True])
def test_smla_dispatch_preserves_non_causal_block_attention(
    shared_window: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    query_lens = [3]

    def make_case(device: str) -> tuple[Any, ...]:
        backend, layer, dsa, shared, q, kv, _ = _case(query_lens, device, ratio=1)
        layer.mode.compress_ratio = 0
        dsa.is_acl_graph = False
        shared.query_layout = Csa2QueryLayout(3, shared_window=shared_window)
        return backend, layer, dsa, shared, q, kv

    backend, layer, dsa, shared, q, kv = make_case("cpu")
    expected = backend._csa2_attention(q, kv, layer, dsa, query_lens, dsa.seq_lens, None, shared)
    backend, layer, dsa, shared, q, kv = make_case("npu")
    calls = []
    native_attention = backend._smla_attention

    def tracked(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(True)
        return native_attention(*args, **kwargs)

    monkeypatch.setattr(backend, "_smla_attention", tracked)
    with forward_context(_context(backend, dsa)):
        actual = backend._csa2_attention(q, kv, layer, dsa, query_lens, dsa.seq_lens, None, shared)
    torch.npu.synchronize()
    assert len(calls) == (0 if shared_window else 1)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.02, atol=0.003)


@pytest.mark.parametrize("q_lens", [[1], [3, 8], [1, 3, 1, 2]])
@pytest.mark.parametrize("rank", [None, 0, 1, 3])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("history", [12, 140])
def test_smla_backend_matches_reference(
    q_lens: list[int], rank: int | None, ratio: int, history: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = torch.ops.xllm_ops.smla_va
    calls = []

    def tracked(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(torch.ops.xllm_ops, "smla_va", tracked)
    cpu = _case(q_lens, "cpu", ratio, history)
    backend, layer, dsa, shared, q, kv, picks = cpu
    expected = backend._csa2_attention(q, kv, layer, dsa, q_lens, dsa.seq_lens, picks, shared)
    backend, layer, dsa, shared, q, kv, picks = _case(q_lens, "npu", ratio, history)
    plan = None
    local_q_lens = q_lens
    local_lengths = dsa.seq_lens
    if rank is not None:
        plan = build_dsa_cp_context(sum(q_lens), q_lens, 4, rank)
        q, expected = plan.shard(q), plan.shard(expected)
        local_q_lens = list(plan.local_q_lens)
        local_lengths = plan.local_context_lengths(dsa.seq_lens)
        local_picks = torch.full((len(q_lens), max(local_q_lens), picks.size(-1)), -1, dtype=torch.int32, device="npu")
        offset = 0
        for seq, length in enumerate(q_lens):
            start = max(0, plan.start - offset)
            count = local_q_lens[seq]
            if count:
                selected = picks[seq, start : start + count]
                # Use the production indexer hand-off, which encodes logical
                # compressed rows after this rank's local query buffer.
                logical = torch.where(selected >= 0, selected - backend.window_size - length, selected)
                positions = local_lengths[seq] - count + torch.arange(1, count + 1, device=q.device)
                visible = torch.div(positions, ratio, rounding_mode="floor").unsqueeze(-1)
                local_picks[seq, :count] = _rebase_qli_picks(logical, visible, backend.window_size + count)
            offset += length
        picks = local_picks
    with forward_context(_context(backend, dsa)):
        actual = backend._csa2_attention(q, kv, layer, dsa, local_q_lens, local_lengths, picks, shared, plan)
    torch.npu.synchronize()
    assert len(calls) == sum(length > 0 for length in local_q_lens)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.02, atol=0.003)


@pytest.mark.parametrize("rank", [None, 0, 1, 3])
def test_smla_saturated_ring_large_chunk(rank: int | None) -> None:
    backend, layer, dsa, shared, q, kv, picks = _case([256], "npu", history=140, topk=512)
    plan = build_dsa_cp_context(256, [256], 4, rank) if rank is not None else None
    lengths = plan.local_context_lengths(dsa.seq_lens) if plan is not None else dsa.seq_lens
    local_lens = list(plan.local_q_lens) if plan is not None else [256]
    if plan is not None:
        q = plan.shard(q)
        logical = picks[0, plan.start : plan.end] - backend.window_size - 256
        logical = torch.where(picks[0, plan.start : plan.end] >= 0, logical, -1)
        positions = lengths[0] - local_lens[0] + torch.arange(1, local_lens[0] + 1, device=q.device)
        picks = _rebase_qli_picks(
            logical, (positions // 2).unsqueeze(-1), backend.window_size + local_lens[0]
        ).unsqueeze(0)
    with forward_context(_context(backend, dsa)):
        actual = backend._csa2_attention(q, kv, layer, dsa, local_lens, lengths, picks, shared, plan)
    # The oracle consumes exactly the same production-style local indices.
    from unittest.mock import patch

    with patch.object(csa2, "_supports_smla_va", return_value=False):
        expected = backend._csa2_attention(q, kv, layer, dsa, local_lens, lengths, picks, shared, plan)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.003)


@pytest.mark.parametrize("topk", [576, 1024])
def test_smla_rounds_compressed_index_width(topk: int, monkeypatch: pytest.MonkeyPatch) -> None:
    backend, layer, dsa, shared, q, kv, picks = _case([3], "npu", history=140, topk=topk)
    layer.attn_sink = layer.attn_sink.bfloat16()
    native = torch.ops.xllm_ops.smla_va
    widths = []

    def tracked(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        widths.append(kwargs["cmp_sparse_indices"].size(-1))
        assert kwargs["sinks"].dtype == torch.float32
        return native(*args, **kwargs)

    monkeypatch.setattr(torch.ops.xllm_ops, "smla_va", tracked)
    with forward_context(_context(backend, dsa)):
        actual = backend._csa2_attention(q, kv, layer, dsa, [3], dsa.seq_lens, picks, shared)
    monkeypatch.setattr(csa2, "_supports_smla_va", lambda: False)
    expected = backend._csa2_attention(q, kv, layer, dsa, [3], dsa.seq_lens, picks, shared)
    assert widths == [1024]
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.003)


@pytest.mark.parametrize("name", ["smla_va", "smla_va_metadata"])
def test_smla_missing_schema_uses_reference(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    backend, layer, dsa, shared, q, kv, picks = _case([3], "npu", history=140)
    monkeypatch.setattr(torch.ops.xllm_ops, name, None)
    assert not _supports_smla_va()
    actual = backend._csa2_attention(q, kv, layer, dsa, [3], dsa.seq_lens, picks, shared)
    cpu, cpu_layer, cpu_dsa, cpu_shared, cpu_q, cpu_kv, cpu_picks = _case([3], "cpu", history=140)
    expected = cpu._csa2_attention(cpu_q, cpu_kv, cpu_layer, cpu_dsa, [3], cpu_dsa.seq_lens, cpu_picks, cpu_shared)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.02, atol=0.003)


def test_smla_large_topk_uses_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, layer, dsa, shared, q, kv, picks = _case([3], "npu", history=140, topk=1025)
    monkeypatch.setattr(
        torch.ops.xllm_ops, "smla_va", lambda *args, **kwargs: pytest.fail("unsupported top-k reached SMLA")
    )
    actual = backend._csa2_attention(q, kv, layer, dsa, [3], dsa.seq_lens, picks, shared)
    cpu, cpu_layer, cpu_dsa, cpu_shared, cpu_q, cpu_kv, cpu_picks = _case([3], "cpu", history=140, topk=1025)
    expected = cpu._csa2_attention(cpu_q, cpu_kv, cpu_layer, cpu_dsa, [3], cpu_dsa.seq_lens, cpu_picks, cpu_shared)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.02, atol=0.003)


def test_smla_metadata_without_tensor_arguments() -> None:
    metadata = torch.ops.xllm_ops.smla_va_metadata(
        16,
        1,
        512,
        batch_size=1,
        max_seqlen_q=3,
        max_seqlen_ori_kv=128,
        layout_q="BSND",
        layout_kv="BSND",
        has_ori_kv=True,
        has_cmp_kv=False,
        ori_mask_mode=4,
        ori_win_left=127,
        ori_win_right=0,
    )
    torch.npu.synchronize()
    assert metadata.device == torch.device("npu", torch.npu.current_device())
    assert metadata.shape == (1024,)


def test_smla_metadata_is_independent_of_allocator_contents() -> None:
    def metadata() -> torch.Tensor:
        return torch.ops.xllm_ops.smla_va_metadata(
            16,
            1,
            512,
            batch_size=1,
            max_seqlen_q=3,
            max_seqlen_ori_kv=128,
            layout_q="BSND",
            layout_kv="BSND",
            has_ori_kv=True,
            has_cmp_kv=False,
            ori_mask_mode=4,
            ori_win_left=127,
            ori_win_right=0,
        )

    expected = metadata().cpu()
    for sentinel in (0x55555555, 0x2AAAAAAA):
        dirty = torch.full((1024,), sentinel, dtype=torch.int32, device="npu")
        torch.npu.synchronize()
        del dirty
        torch.testing.assert_close(metadata().cpu(), expected, rtol=0, atol=0)


def test_smla_decode_graph_reads_live_lengths() -> None:
    backend, layer, dsa, shared, q, kv, picks = _case([1], "npu")
    for _ in range(3):
        with forward_context(_context(backend, dsa)):
            backend._csa2_attention(q, kv, layer, dsa, [1], dsa.seq_lens, picks, shared)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph), forward_context(_context(backend, dsa)):
        actual = backend._csa2_attention(q, kv, layer, dsa, [1], dsa.seq_lens, picks, shared)
    for length in (14, 127, 128, 129, 254, 255):
        # Reuse the same captured graph across parity and SWA ring boundaries.
        dsa.seq_lens.fill_(length)
        q.mul_(0.9)
        kv.add_(0.01)
        graph.replay()
        torch.npu.synchronize()
        with forward_context(_context(backend, dsa)):
            expected = backend._csa2_attention(q, kv, layer, dsa, [1], dsa.seq_lens, picks, shared)
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        reference, cpu_layer, cpu_dsa, cpu_shared, _, _, cpu_picks = _case([1], "cpu")
        cpu_dsa.seq_lens.fill_(length)
        reference_output = reference._csa2_attention(
            q.cpu(), kv.cpu(), cpu_layer, cpu_dsa, [1], cpu_dsa.seq_lens, cpu_picks, cpu_shared
        )
        torch.testing.assert_close(actual.cpu(), reference_output, rtol=0.02, atol=0.003)


def test_smla_forward_cache_separates_distinct_length_tensors() -> None:
    backend, layer, dsa, shared, q, kv, picks = _case([1], "npu")
    lengths = (dsa.seq_lens, dsa.seq_lens + 5)
    window_kv = torch.cat((backend._kv_caches[0].swa[0], kv))
    expected = []
    for ctx_lens in lengths:
        with forward_context(_context(backend, dsa)):
            expected.append(
                backend._smla_attention(
                    q, window_kv, layer, ctx_lens, ctx_lens, 1, 1, 0, picks, shared, layer.attn_sink.float()
                )
            )
    with forward_context(_context(backend, dsa)):
        for ctx_lens, reference in zip(lengths, expected):
            actual = backend._smla_attention(
                q, window_kv, layer, ctx_lens, ctx_lens, 1, 1, 0, picks, shared, layer.attn_sink.float()
            )
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_smla_forward_cache_invalidates_versioned_lengths(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, layer, dsa, shared, q, kv, picks = _case([1], "npu")
    native = torch.ops.xllm_ops.smla_va_metadata
    calls = []

    def tracked(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(torch.ops.xllm_ops, "smla_va_metadata", tracked)
    with forward_context(_context(backend, dsa)):
        backend._csa2_attention(q, kv, layer, dsa, [1], dsa.seq_lens, picks, shared)
        backend._csa2_attention(q, kv, layer, dsa, [1], dsa.seq_lens, picks, shared)
        assert len(calls) == 1
        dsa.seq_lens.add_(5)
        actual = backend._csa2_attention(q, kv, layer, dsa, [1], dsa.seq_lens, picks, shared)
        assert len(calls) == 2
    with forward_context(_context(backend, dsa)):
        expected = backend._csa2_attention(q, kv, layer, dsa, [1], dsa.seq_lens, picks, shared)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_smla_forward_cache_refreshes_reused_inference_lengths() -> None:
    backend, layer, dsa, shared, q, kv, picks = _case([1], "npu")
    assert torch.is_inference(dsa.seq_lens)
    with forward_context(_context(backend, dsa)):
        backend._csa2_attention(q, kv, layer, dsa, [1], dsa.seq_lens, picks, shared)
    # The executor starts a new ForwardContext for each step, even when it
    # reuses the inference tensor's storage for the next step's lengths.
    dsa.seq_lens.add_(5)
    with forward_context(_context(backend, dsa)):
        actual = backend._csa2_attention(q, kv, layer, dsa, [1], dsa.seq_lens, picks, shared)
    lengths_copy = dsa.seq_lens.clone()
    with forward_context(_context(backend, dsa)):
        expected = backend._csa2_attention(q, kv, layer, dsa, [1], lengths_copy, picks, shared)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(1, 128, 2, 512), (1, 128, 256)])
def test_smla_rejects_incompatible_compressed_cache(shape: tuple[int, ...]) -> None:
    backend, layer, dsa, shared, q, kv, picks = _case([3], "npu")
    shared.key_cache = torch.empty(shape, dtype=q.dtype, device=q.device)
    with (
        forward_context(_context(backend, dsa)),
        pytest.raises(ValueError, match="compressed KV cache must have shape"),
    ):
        backend._csa2_attention(q, kv, layer, dsa, [3], dsa.seq_lens, picks, shared)


def test_smla_original_lengths_stay_within_supplied_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, layer, dsa, shared, q, kv, picks = _case([3], "npu")
    window_kv = torch.cat((backend._kv_caches[0].swa[0], kv))
    calls = []

    def metadata(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(kwargs["seqused_ori_kv"].cpu())
        return torch.zeros(1024, dtype=torch.int32, device=q.device)

    monkeypatch.setattr(torch.ops.xllm_ops, "smla_va_metadata", metadata)
    monkeypatch.setattr(torch.ops.xllm_ops, "smla_va", lambda *args, **kwargs: (torch.empty_like(q), None))
    # An inconsistent restored context must not advertise rows beyond the
    # actual original-KV view (or a negative sequence length) to the kernel.
    for local_length in (-200, 1000):
        local = dsa.seq_lens.new_tensor([local_length])
        with forward_context(_context(backend, dsa)):
            backend._smla_attention(q, window_kv, layer, local, dsa.seq_lens, 3, 3, 0, picks, shared, layer.attn_sink)
    assert calls[0].item() == 0
    assert calls[1].item() == window_kv.size(0)


def test_smla_invalid_compressed_picks_are_padding() -> None:
    backend, layer, dsa, shared, q, kv, picks = _case([3], "npu")
    # Keep valid picks first, then exercise a window-row index and an index
    # beyond the logical compressed sequence. Neither may alias valid KV.
    picks[:, :, -2] = backend.window_size + 2
    picks[:, :, -1] = backend.window_size + 3 + 128
    padded = picks.clone()
    padded[:, :, -2:] = -1
    window_kv = torch.cat((backend._kv_caches[0].swa[0], kv))
    with forward_context(_context(backend, dsa)):
        actual = backend._smla_attention(
            q, window_kv, layer, dsa.seq_lens, dsa.seq_lens, 3, 3, 0, picks, shared, layer.attn_sink.float()
        )
    with forward_context(_context(backend, dsa)):
        expected = backend._smla_attention(
            q, window_kv, layer, dsa.seq_lens, dsa.seq_lens, 3, 3, 0, padded, shared, layer.attn_sink.float()
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["TND", "BSND"])
@pytest.mark.parametrize("return_lse", [False, True])
def test_smla_native_schema_supports_fake_tensors(layout: str, return_lse: bool) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    from xllm.python.kernels_npu import _custom_op  # noqa: F401

    with FakeTensorMode():
        shape = (3, 16, 512) if layout == "TND" else (1, 3, 16, 512)
        q = torch.empty(shape, dtype=torch.bfloat16, device="npu")
        kv = torch.empty((1, 128, 1, 512), dtype=torch.bfloat16, device="npu")
        starts = torch.empty((2,), dtype=torch.int32, device="npu")
        metadata = torch.ops.xllm_ops.smla_va_metadata(16, 1, 512, cu_seqlens_q=starts)
        output, lse = torch.ops.xllm_ops.smla_va(
            q,
            ori_kv=kv,
            metadata=metadata,
            layout_q=layout,
            layout_kv="PA_BBND",
            return_softmax_lse=return_lse,
        )
        assert output.shape == q.shape and output.dtype == q.dtype
        assert metadata.shape == (1024,) and metadata.dtype == torch.int32
        expected_lse = (1, 3, 16) if layout == "TND" else (1, 1, 3, 16)
        assert lse.shape == (expected_lse if return_lse else (0,))
        assert lse.dtype == torch.float32
