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

When ``engram_config.dp_shared_memory`` is set, the DP replicas of one node
that map the same TP shard share ONE physical copy of the host table instead of
allocating one per rank (~27.7GB/rank of host memory saved). vllm-ascend does
this with a CANN-registered UVA range (``SharedUvaBuffer``); xLLM reads the
table on the CPU (``xllm_ops.engram_int8_lookup_cpu``), so the device pointer /
host-registration layer is dropped: the sharing mechanism is a plain
``multiprocessing.shared_memory`` segment plus a ``torch.frombuffer`` CPU view.

One :class:`SharedHostEngramStorage` owns the two segments of one (engram layer,
TP shard): ``codes`` (``[rows, dim]`` int8) and ``scales``
(``[rows, dim // quant_block]`` float32). Creation mirrors vllm-ascend's
``SharedUvaBuffer`` timing:

1. the group leader (group rank 0) creates both segments with random names;
2. it broadcasts a structured descriptor (names/shape/dtype/size/layer id/TP
   shard/leader global rank/random token) over the gloo group;
3. non-leaders attach and validate size/shape/dtype/token, then every rank
   builds its ``torch.frombuffer`` CPU view;
4. an ``engram_all_gather_object`` collects per-rank errors, an
   ``engram_barrier`` fences every mapping (or every failure), and the leader
   unlinks both names while its own mapping -- and every attacher's -- stays
   valid;
5. only then is the storage ``ATTACHED``; the leader-only load moves it to
   ``LOADED``.

Every collective here rides the gloo ``engram_dp`` group (HCCL has no object
collective). ``close()`` is idempotent and collective-free: the caller drops the
``nn.Parameter`` aliases first (readers stop) and then closes the mappings.
"""

from __future__ import annotations

import multiprocessing.resource_tracker
import os
import secrets
from multiprocessing import shared_memory
from typing import Any

import torch
from torch.distributed import ProcessGroup

from xllm.python.distributed import (
    engram_all_gather_object,
    engram_barrier,
    engram_broadcast_object_list,
)

# Lifecycle states. INIT -> ATTACHED -> LOADED; FAILED/CLOSED are terminal.
STATE_INIT = "init"
STATE_ATTACHED = "attached"
STATE_LOADED = "loaded"
STATE_FAILED = "failed"
STATE_CLOSED = "closed"

# The group name the collectives resolve; only one Engram sharing group exists.
ENGRAM_DP_GROUP_NAME = "engram_dp"

# /dev/shm is where multiprocessing.shared_memory allocates on Linux. The
# capacity check below is a precheck; an actual attach with a matching token is
# the authoritative evidence that a group really shares host memory.
_SHM_DIR = "/dev/shm"

# Length of the leader's random token (``secrets.token_hex(16)``) in bytes. The
# leader writes it into the tail of the ``codes`` segment; an attacher reads it
# back and compares it to the descriptor's token, so a stale/foreign segment of
# the same size cannot be attached silently (see ``_verify_segment_token``).
_TOKEN_BYTES = 32


def estimate_shared_bytes(rows: int, dim: int, quant_block: int) -> int:
    """Bytes of the codes+scales segment pair for one (layer, TP shard)."""
    scale_cols = dim // quant_block
    return int(rows) * int(dim) + int(rows) * scale_cols * 4


def ensure_dev_shm_capacity(required_bytes: int) -> None:
    """Raise unless ``/dev/shm`` has room for ``required_bytes``.

    A best-effort precheck: on hosts without ``/dev/shm`` it is skipped, and
    even when it passes, a successful attach with a matching token is the only
    authoritative evidence that a sharing group really shares host memory.
    """
    try:
        stat = os.statvfs(_SHM_DIR)
    except OSError:
        return
    free = stat.f_bavail * stat.f_frsize
    if required_bytes > free:
        raise RuntimeError(
            f"engram DP-shared host table needs {required_bytes} bytes in {_SHM_DIR} but only "
            f"{free} bytes are free; reduce the shared engram footprint on this node or "
            "disable engram_config.dp_shared_memory"
        )


class SharedHostEngramStorage:
    """Codes + scales host segments shared by a node-local DP shard group.

    The leader creates the segments and the group shares them until every rank
    :meth:`close`\\ s (or the process exits). The class never scans or deletes
    ``/dev/shm`` entries it did not create.
    """

    def __init__(
        self,
        *,
        layer_id: int,
        rows: int,
        dim: int,
        quant_block: int,
        group: ProcessGroup,
    ) -> None:
        if rows < 0 or dim <= 0 or dim % quant_block != 0:
            raise ValueError(f"invalid shared Engram geometry rows={rows}, dim={dim}, quant_block={quant_block}")
        self.layer_id = int(layer_id)
        self.rows = int(rows)
        self.dim = int(dim)
        self.quant_block = int(quant_block)
        self.scale_cols = self.dim // self.quant_block
        self.codes_bytes = self.rows * self.dim
        self.scales_bytes = self.rows * self.scale_cols * 4
        self.group = group
        self.rank = int(group.rank())
        self.world_size = int(group.size())
        self.owner = self.rank == 0
        self.token: str | None = None
        self._codes_handle: shared_memory.SharedMemory | None = None
        self._scales_handle: shared_memory.SharedMemory | None = None
        self._state = STATE_INIT
        # CPU views over the shared mappings (the table the lookup op reads).
        self.codes: torch.Tensor | None = None
        self.scales: torch.Tensor | None = None

    # -- lifecycle ----------------------------------------------------------

    def create_or_attach(self) -> None:
        """Create (leader) or attach (peers) both segments, then return ATTACHED."""
        if self._state != STATE_INIT:
            raise RuntimeError(f"engram shared storage for layer {self.layer_id} is already {self._state}")
        self._attach_or_create()
        self._state = STATE_ATTACHED

    def mark_loaded(self) -> None:
        """Mark the shared table fully populated (after the leader-only load)."""
        if self._state == STATE_ATTACHED:
            self._state = STATE_LOADED

    def mark_failed(self) -> None:
        if self._state in (STATE_ATTACHED, STATE_LOADED):
            self._state = STATE_FAILED

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_loaded(self) -> bool:
        return self._state == STATE_LOADED

    def _attach_or_create(self) -> None:
        payload: list[dict[str, Any] | None] = [None]
        if self.owner:
            try:
                self._check_dev_shm_capacity()
                codes = shared_memory.SharedMemory(create=True, size=self.codes_bytes + _TOKEN_BYTES)
                self._codes_handle = codes
                scales = shared_memory.SharedMemory(create=True, size=self.scales_bytes)
                self._scales_handle = scales
                token = secrets.token_hex(16)
                self.token = token
                # Write the token into the codes segment tail: the attachers read
                # it back and compare (``_verify_segment_token``), which is the
                # only authoritative check that they mapped THIS leader's
                # segment. The table view is bounded by ``count`` below, so the
                # token bytes are never read as table data.
                codes_buf = codes.buf
                assert codes_buf is not None
                codes_buf[self.codes_bytes : self.codes_bytes + _TOKEN_BYTES] = token.encode("ascii")
                payload = [
                    {
                        "codes": codes.name,
                        "scales": scales.name,
                        "codes_bytes": self.codes_bytes,
                        "scales_bytes": self.scales_bytes,
                        "token": self.token,
                        "layer_id": self.layer_id,
                        "rows": self.rows,
                        "dim": self.dim,
                        "quant_block": self.quant_block,
                    }
                ]
            except Exception as exc:  # noqa: BLE001 - surfaced to every rank below
                self._discard_partial_creation()
                payload = [{"error": f"{type(exc).__name__}: {exc}"}]
        engram_broadcast_object_list(payload, src=0, group_name=ENGRAM_DP_GROUP_NAME)
        descriptor = payload[0]
        if not isinstance(descriptor, dict):
            raise RuntimeError(
                f"engram shared host-table for layer {self.layer_id} got a malformed descriptor {descriptor!r}"
            )
        if "error" in descriptor:
            raise RuntimeError(
                f"engram shared host-table creation failed for layer {self.layer_id}: {descriptor['error']}"
            )
        error: str | None = None
        try:
            # Validate inside the aggregate window: a descriptor mismatch must
            # ride the ``engram_all_gather_object``/``engram_barrier`` below so
            # every rank fails together, not raise here and leave the peers
            # blocked on a collective that never comes.
            self._validate_descriptor(descriptor)
            if not self.owner:
                # The codes segment carries the leader's token in its tail, so
                # size the attach check to cover it too.
                self._codes_handle = self._attach(descriptor["codes"], self.codes_bytes + _TOKEN_BYTES)
                self._scales_handle = self._attach(descriptor["scales"], self.scales_bytes)
                self._verify_segment_token(self._codes_handle, descriptor["token"])
            assert self._codes_handle is not None and self._scales_handle is not None
            # ``count`` bounds the view to the exact element count: some
            # platforms round a SharedMemory allocation up to a page multiple,
            # so the raw buffer can be larger than requested and an unbounded
            # ``frombuffer`` + reshape would fail.
            codes = torch.frombuffer(self._codes_handle.buf, dtype=torch.int8, count=self.rows * self.dim).reshape(
                self.rows, self.dim
            )
            scales = torch.frombuffer(
                self._scales_handle.buf, dtype=torch.float32, count=self.rows * self.scale_cols
            ).reshape(self.rows, self.scale_cols)
            if not codes.is_contiguous() or not scales.is_contiguous():
                raise RuntimeError("engram shared host-table view is not contiguous")
            self.codes = codes
            self.scales = scales
            self.token = descriptor["token"]
        except Exception as exc:  # noqa: BLE001 - aggregated below
            error = f"{type(exc).__name__}: {exc}"

        errors: list[str | None] = [None] * self.world_size
        engram_all_gather_object(errors, error, group_name=ENGRAM_DP_GROUP_NAME)
        failures = "; ".join(f"rank {rank}: {failure}" for rank, failure in enumerate(errors) if failure is not None)
        # Fence every mapping (or every failure) before the leader unlinks.
        engram_barrier(group_name=ENGRAM_DP_GROUP_NAME)
        if failures:
            # Unlink BEFORE close on the leader: unlink keeps this rank's own
            # mapping valid, whereas ``close()`` nulls the handles and would
            # turn the leader's unlink into a no-op (leaving the names behind in
            # /dev/shm). Mirrors the unlink -> close order of
            # ``_discard_partial_creation``.
            if self.owner:
                self._unlink_created()
            try:
                self.close()
            except RuntimeError as exc:
                failures = f"{failures}; release failed: {exc}"
            raise RuntimeError(f"engram shared host-table initialization failed for layer {self.layer_id}: {failures}")
        if self.owner:
            # The leader's own mapping -- and every attacher's -- stays valid
            # after unlink, so the names can go before any reader exists.
            self._unlink_created()

    def _validate_descriptor(self, descriptor: dict[str, Any]) -> None:
        expected = {
            "layer_id": self.layer_id,
            "rows": self.rows,
            "dim": self.dim,
            "quant_block": self.quant_block,
            "codes_bytes": self.codes_bytes,
            "scales_bytes": self.scales_bytes,
        }
        for key, value in expected.items():
            if descriptor.get(key) != value:
                raise RuntimeError(
                    f"engram shared host-table descriptor mismatch for layer {self.layer_id}: "
                    f"{key}={descriptor.get(key)!r}, expected {value!r}"
                )
        # The token must be present and well-formed: the owner writes it into
        # the codes segment tail and every attacher verifies it against the
        # attached bytes (``_verify_segment_token``). A descriptor without it
        # cannot be trusted, so reject it here rather than attach blind.
        token = descriptor.get("token")
        if not isinstance(token, str) or len(token) != _TOKEN_BYTES:
            raise RuntimeError(
                f"engram shared host-table descriptor for layer {self.layer_id} has a "
                f"missing/malformed token {token!r}"
            )

    def _verify_segment_token(self, codes: shared_memory.SharedMemory, token: str) -> None:
        """Confirm the attached ``codes`` segment carries the descriptor's token.

        The leader writes its random token into the segment tail (see
        ``_attach_or_create``); matching it is the authoritative evidence that
        this rank mapped the segment the descriptor names, not a stale or
        foreign segment of the same size.
        """
        buf = codes.buf
        if buf is None:
            raise RuntimeError(f"engram shared segment for layer {self.layer_id} is already closed")
        attached = bytes(buf[self.codes_bytes : self.codes_bytes + _TOKEN_BYTES])
        if attached != token.encode("ascii"):
            raise RuntimeError(
                f"engram shared segment token mismatch for layer {self.layer_id}: "
                "the attached codes segment does not carry this leader's token"
            )

    @staticmethod
    def _attach(name: str, expected_size: int) -> shared_memory.SharedMemory:
        """Attach to a peer's segment without becoming a tracked owner.

        Python >= 3.13 registers an *attacher* with ``multiprocessing``'s
        resource_tracker, which would keep the name alive after the creator
        unlinks it (3.12 only registers the create path). Swap the registration
        hook out only for this attach call -- never permanently -- so only the
        creator owns the segment.
        """
        real_register = multiprocessing.resource_tracker.register
        multiprocessing.resource_tracker.register = lambda *args, **kwargs: None
        try:
            shm = shared_memory.SharedMemory(name=name)
        finally:
            multiprocessing.resource_tracker.register = real_register
        if shm.size < expected_size:
            shm.close()
            raise RuntimeError(f"engram shared segment {name} is {shm.size} bytes, expected at least {expected_size}")
        return shm

    def _check_dev_shm_capacity(self) -> None:
        ensure_dev_shm_capacity(self.codes_bytes + self.scales_bytes + _TOKEN_BYTES)

    def _discard_partial_creation(self) -> None:
        for handle in (self._codes_handle, self._scales_handle):
            if handle is None:
                continue
            try:
                handle.unlink()
            except OSError:
                pass
            try:
                handle.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup of a failed create
                pass
        self._codes_handle = None
        self._scales_handle = None

    def _unlink_created(self) -> None:
        """Leader-only: unlink the created names; mappings stay valid."""
        for handle in (self._codes_handle, self._scales_handle):
            if handle is None:
                continue
            try:
                handle.unlink()
            except OSError as exc:
                self._warn(f"engram shared backing unlink failed for layer {self.layer_id}: {exc}")

    def close(self) -> None:
        """Release the shared mappings; idempotent, no collective, no cross-rank sync.

        The caller must drop the ``nn.Parameter`` aliases (and any other reader)
        before calling this. A failed close keeps the remaining handles for a
        retry instead of leaving a half-released storage.
        """
        if self._state == STATE_CLOSED:
            return
        # Stop readers first: drop the storage's own CPU views (the caller has
        # already dropped the ``nn.Parameter`` aliases) before closing mappings.
        self.codes = None
        self.scales = None
        failures: list[str] = []
        for attr in ("_codes_handle", "_scales_handle"):
            handle = getattr(self, attr)
            if handle is None:
                continue
            try:
                handle.close()
                setattr(self, attr, None)
            except Exception as exc:  # noqa: BLE001 - aggregated; state stays retryable
                failures.append(f"{attr}: {type(exc).__name__}: {exc}")
        if failures:
            raise RuntimeError(
                f"engram shared host-table close failed for layer {self.layer_id}: " + "; ".join(failures)
            )
        self._state = STATE_CLOSED

    def __del__(self) -> None:
        # Best-effort only: never run a distributed collective from teardown /
        # GC. The mappings die with the process even if this is skipped.
        try:
            self.close()
        except Exception:  # noqa: BLE001 - teardown must never raise
            pass

    @staticmethod
    def _warn(message: str) -> None:
        try:
            from scripts.logger import logger

            logger.warning(message)
        except Exception:  # noqa: BLE001 - logging must never block teardown
            pass


__all__ = [
    "ENGRAM_DP_GROUP_NAME",
    "STATE_ATTACHED",
    "STATE_CLOSED",
    "STATE_FAILED",
    "STATE_INIT",
    "STATE_LOADED",
    "SharedHostEngramStorage",
    "ensure_dev_shm_capacity",
    "estimate_shared_bytes",
]
