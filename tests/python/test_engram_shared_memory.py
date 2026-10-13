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

"""Node-local DP sharing of the host-resident Engram int8 tables.

``SharedHostEngramStorage`` normally rides a gloo ``engram_dp`` group. These
cases drive it in a *single* process: the group is a rank/size stub and the
three collectives (broadcast / engram_all_gather_object / engram_barrier) are
replaced with no-op / identity stand-ins. That exercises the real create,
``torch.frombuffer`` view, write/read aliasing, load marking and idempotent
close paths -- everything except the cross-process attach.
"""

from __future__ import annotations

import multiprocessing.resource_tracker

import pytest
import torch

from xllm.python.models import engram_shared_memory as shm


class _StubGroup:
    """Minimal stand-in for a ProcessGroup: only rank()/size() are read."""

    def __init__(self, *, rank: int = 0, size: int = 1) -> None:
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def _stub_collectives(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Replace the gloo collectives with single-process stand-ins."""
    calls = {"broadcast": 0, "gather": 0, "barrier": 0}

    def _broadcast(object_list: list, src: int = 0, group_name: str | None = None) -> None:
        calls["broadcast"] += 1

    def _all_gather(object_list: list, obj: object, group_name: str | None = None) -> None:
        calls["gather"] += 1
        # Single process: this rank is the only member of the group.
        assert len(object_list) == 1
        object_list[0] = obj

    def _barrier(group_name: str | None = None) -> None:
        calls["barrier"] += 1

    monkeypatch.setattr(shm, "engram_broadcast_object_list", _broadcast)
    monkeypatch.setattr(shm, "engram_all_gather_object", _all_gather)
    monkeypatch.setattr(shm, "engram_barrier", _barrier)
    return calls


def test_estimate_shared_bytes_counts_codes_and_scales() -> None:
    # rows*dim int8 codes + rows*(dim//quant_block) float32 scales.
    assert shm.estimate_shared_bytes(10, 64, 32) == 10 * 64 + 10 * 2 * 4
    assert shm.estimate_shared_bytes(0, 64, 32) == 0


def test_validate_descriptor_requires_a_wellformed_token() -> None:
    """A descriptor without a full-length token is rejected before attach."""
    storage = shm.SharedHostEngramStorage(
        layer_id=1,
        rows=8,
        dim=32,
        quant_block=32,
        group=_StubGroup(rank=0, size=1),
    )
    descriptor = {
        "layer_id": 1,
        "rows": 8,
        "dim": 32,
        "quant_block": 32,
        "codes_bytes": 8 * 32,
        "scales_bytes": 8 * 1 * 4,
        "token": "a" * shm._TOKEN_BYTES,
    }
    storage._validate_descriptor(descriptor)  # well-formed: no raise

    with pytest.raises(RuntimeError, match="missing/malformed token"):
        storage._validate_descriptor({**descriptor, "token": None})
    with pytest.raises(RuntimeError, match="missing/malformed token"):
        storage._validate_descriptor({**descriptor, "token": "abc"})


def test_verify_segment_token_reads_back_the_written_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """The leader's token round-trips through the codes segment tail.

    An attacher compares that tail to the descriptor token (the authoritative
    attach evidence); a foreign/stale segment would not carry it.
    """
    _stub_collectives(monkeypatch)
    storage = shm.SharedHostEngramStorage(
        layer_id=2,
        rows=8,
        dim=32,
        quant_block=32,
        group=_StubGroup(rank=0, size=1),
    )
    storage.create_or_attach()
    assert storage._codes_handle is not None
    assert storage.token is not None

    storage._verify_segment_token(storage._codes_handle, storage.token)  # matches
    with pytest.raises(RuntimeError, match="token mismatch"):
        storage._verify_segment_token(storage._codes_handle, "0" * shm._TOKEN_BYTES)

    # The token lives past the table view, so the codes geometry is unchanged.
    assert storage.codes is not None and storage.codes.shape == (8, 32)
    storage.close()


def test_ensure_dev_shm_capacity_skips_without_dev_shm(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(path: str) -> object:
        raise OSError("no /dev/shm here")

    monkeypatch.setattr(shm.os, "statvfs", _raise)
    # Best-effort: a host without /dev/shm is not a hard failure.
    shm.ensure_dev_shm_capacity(10**12)


def test_ensure_dev_shm_capacity_rejects_short_host(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Stat:
        f_bavail = 1
        f_frsize = 4096

    monkeypatch.setattr(shm.os, "statvfs", lambda path: _Stat())
    with pytest.raises(RuntimeError, match="/dev/shm"):
        shm.ensure_dev_shm_capacity(4096 * 2)


def test_shared_host_engram_storage_single_process_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_collectives(monkeypatch)
    storage = shm.SharedHostEngramStorage(
        layer_id=3,
        rows=8,
        dim=32,
        quant_block=32,
        group=_StubGroup(rank=0, size=1),
    )
    assert storage.state == shm.STATE_INIT

    storage.create_or_attach()
    assert storage.state == shm.STATE_ATTACHED
    assert storage.is_loaded is False

    # frombuffer views: CPU, contiguous, exact geometry.
    assert storage.codes is not None and storage.codes.shape == (8, 32)
    assert storage.scales is not None and storage.scales.shape == (8, 1)
    assert storage.codes.dtype == torch.int8 and storage.scales.dtype == torch.float32
    assert storage.codes.device.type == "cpu" and storage.scales.device.type == "cpu"
    assert storage.codes.is_contiguous() and storage.scales.is_contiguous()

    # The views alias the shared mapping: a write reads back through the view.
    storage.codes[0, :] = torch.arange(32, dtype=torch.int8)
    storage.scales[0, 0] = 0.375
    assert torch.equal(storage.codes[0], torch.arange(32, dtype=torch.int8))
    assert float(storage.scales[0, 0]) == 0.375

    storage.mark_loaded()
    assert storage.is_loaded is True
    assert storage.state == shm.STATE_LOADED

    storage.close()
    assert storage.state == shm.STATE_CLOSED
    assert storage.codes is None and storage.scales is None

    # close is idempotent and collective-free.
    storage.close()
    assert storage.state == shm.STATE_CLOSED
    assert calls == {"broadcast": 1, "gather": 1, "barrier": 1}


def test_shared_host_engram_storage_creation_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_collectives(monkeypatch)

    def _boom(*args: object, **kwargs: object) -> object:
        raise OSError("shared memory exhausted")

    monkeypatch.setattr(shm.shared_memory, "SharedMemory", _boom)
    storage = shm.SharedHostEngramStorage(
        layer_id=4,
        rows=8,
        dim=32,
        quant_block=32,
        group=_StubGroup(rank=0, size=1),
    )
    with pytest.raises(RuntimeError, match="creation failed for layer 4"):
        storage.create_or_attach()


def test_shared_host_engram_storage_failure_path_still_unlinks(monkeypatch: pytest.MonkeyPatch) -> None:
    """I-1 regression: the leader unlinks before close on the failure path.

    The old order ran ``close()`` first, which nulled the handles, so the
    leader's ``_unlink_created`` became a no-op and the created segments
    lingered in /dev/shm. Force a post-broadcast failure (the same branch a
    peer attach failure lands in) and count the unlinks.
    """
    _stub_collectives(monkeypatch)
    real_unlink = shm.shared_memory.SharedMemory.unlink
    unlinked: list[str] = []

    def _counting_unlink(self: shm.shared_memory.SharedMemory) -> None:
        unlinked.append(self.name)
        real_unlink(self)

    monkeypatch.setattr(shm.shared_memory.SharedMemory, "unlink", _counting_unlink)

    def _boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("view construction boom")

    monkeypatch.setattr(shm.torch, "frombuffer", _boom)

    storage = shm.SharedHostEngramStorage(
        layer_id=6,
        rows=8,
        dim=32,
        quant_block=32,
        group=_StubGroup(rank=0, size=1),
    )
    with pytest.raises(RuntimeError, match="initialization failed for layer 6"):
        storage.create_or_attach()
    # The leader created both segments and unlinked both before closing.
    assert len(unlinked) == 2
    assert storage.state == shm.STATE_CLOSED


def test_shared_host_engram_storage_mark_failed_after_attach(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_collectives(monkeypatch)
    storage = shm.SharedHostEngramStorage(
        layer_id=5,
        rows=4,
        dim=32,
        quant_block=32,
        group=_StubGroup(rank=0, size=1),
    )
    # mark_failed is a no-op before the segments exist.
    storage.mark_failed()
    assert storage.state == shm.STATE_INIT

    storage.create_or_attach()
    storage.mark_failed()
    assert storage.state == shm.STATE_FAILED
    assert storage.is_loaded is False

    storage.close()
    assert storage.state == shm.STATE_CLOSED


def test_shared_host_engram_storage_rejects_bad_geometry() -> None:
    group = _StubGroup(rank=0, size=1)
    # dim not divisible by quant_block.
    with pytest.raises(ValueError, match="invalid shared Engram geometry"):
        shm.SharedHostEngramStorage(layer_id=0, rows=4, dim=48, quant_block=32, group=group)
    # non-positive dim.
    with pytest.raises(ValueError, match="invalid shared Engram geometry"):
        shm.SharedHostEngramStorage(layer_id=0, rows=4, dim=0, quant_block=32, group=group)


def test_attach_restores_resource_tracker_register_on_success() -> None:
    """``_attach`` swaps the resource_tracker hook out and always restores it.

    The suppression is meant to be local to one attach call: a leaked no-op
    would silently stop this process from ever reclaiming *created* segments.
    Attach to a real (self-created) segment and pin the restore.
    """
    original = multiprocessing.resource_tracker.register
    created = shm.shared_memory.SharedMemory(create=True, size=64)
    try:
        attached = shm.SharedHostEngramStorage._attach(created.name, 64)
        try:
            assert attached.size >= 64
            assert multiprocessing.resource_tracker.register is original
        finally:
            attached.close()
    finally:
        created.unlink()
        created.close()
    assert multiprocessing.resource_tracker.register is original


def test_attach_restores_resource_tracker_register_on_failure() -> None:
    """A failed attach (unknown name) still restores the tracker hook."""
    original = multiprocessing.resource_tracker.register
    with pytest.raises(FileNotFoundError):
        shm.SharedHostEngramStorage._attach("engram_attach_missing_segment", 64)
    assert multiprocessing.resource_tracker.register is original
