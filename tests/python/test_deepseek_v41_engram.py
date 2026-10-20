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

"""Component tests for ``deepseek_v41_engram.py``: the standalone
``EngramEmbedding`` (layout/hash helpers, TP sharding, int8 storage, host
offload) plus the host-offload CPU int8 lookup op/kernel.
"""

from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.python._engram_reference import _reference_engram_lookup
from xllm.python.models.deepseek_v41 import DeepseekV41Config
from xllm.python.models.deepseek_v41_engram import (
    ENGRAM_IMAGE_SENTINEL_BASE_ID,
    ENGRAM_STORAGE_FP8,
    ENGRAM_STORAGE_INT8,
    Engram,
    EngramEmbedding,
    EngramHashState,
    EngramLayout,
    _load_engram_hash_state_sidecar,
    build_compressed_token_map_from_texts,
    build_engram_hash_state,
    normalize_token_text,
)


def _engram_lookup_op_registered() -> bool:
    """Whether the in-repo ``xllm_ops`` CPU lookup op is registered on this host.

    Needs a full xLLM binary build; a pure-Python test host has no such key.
    """
    try:
        import xllm.python.models.deepseek_v41_engram as engram

        return engram._get_engram_lookup_op() is not None
    except Exception:  # noqa: BLE001 - a missing op must not error collection
        return False


def _host_pinning_available() -> bool:
    """Whether ``_host_buffer`` can hand out pinned staging on this host."""
    try:
        return torch.empty(1, dtype=torch.int64, pin_memory=True).is_pinned()
    except Exception:  # noqa: BLE001
        return False


def _make_offload_engram(seed: int, dim: int = 64, heads: int = 4, head_size: int = 128) -> Any:
    """A host-offloaded int8 engram embedding with random quantized tables."""
    import xllm.python.models.deepseek_v41_engram as engram

    torch.manual_seed(seed)
    embed = engram.EngramEmbedding(
        num_embeddings=heads * head_size,
        dim=dim,
        head_sizes=[head_size] * heads,
        tp_size=2,
        tp_rank=0,
        quant_block=32,
        storage=engram.ENGRAM_STORAGE_INT8,
        device=torch.device("cpu"),
        offload=True,
    )
    assert embed.offload
    with torch.no_grad():
        embed.weight.copy_(torch.randint(-127, 127, embed.weight.shape, dtype=torch.int8))
        embed.weight_scale.copy_(torch.rand(embed.weight_scale.shape) * 0.05)
    return embed


def _assert_bf16_within_one_ulp(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    """Bitwise, or at most one bf16 ulp, with ``max_err`` on failure."""
    if torch.equal(actual, expected):
        return
    scale = expected.float().abs().clamp_min(2.0**-30)
    ulp = torch.pow(2.0, torch.floor(torch.log2(scale)) - 7)
    diff = (actual.float() - expected.float()).abs()
    max_err = diff.max().item()
    assert not bool((diff > ulp).any()), f"{label}: exceeds 1 bf16 ulp, max_err={max_err:.3e}"


# --- JIT-built execution of the in-repo kernel -----------------------------
#
# ``engram_int8_lookup_cpu`` is the only dequant path for the host-offloaded
# table, but a pure-Python host never registers it, so the op-registered case
# below is permanently skipped there. This case compiles the very source the
# xLLM binary builds (``_ENGRAM_KERNEL_SOURCE``) into a throwaway extension and
# runs it, so the fused host dequant is genuinely executed rather than only
# compared python-vs-python.

_ENGRAM_KERNEL_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "xllm"
    / "core"
    / "kernels"
    / "npu"
    / "xllm_ops"
    / "engram_int8_lookup_cpu.cpp"
)

# The repo kernel is a plain C++ free function (no ``PYBIND11_MODULE``: the
# binary registers it on the ``xllm_ops`` CPU key). This shim adds the binding
# so the same translation unit becomes importable here.
_ENGRAM_PYBIND_SHIM = """
#include <torch/extension.h>

namespace xllm { namespace kernel { namespace npu {
void engram_int8_lookup_cpu(const torch::Tensor& weight, const torch::Tensor& scale,
                            const torch::Tensor& ids, torch::Tensor& output);
}}}

namespace {
void engram_int8_lookup_cpu_entry(const torch::Tensor& weight, const torch::Tensor& scale,
                                  const torch::Tensor& ids, torch::Tensor& output) {
  xllm::kernel::npu::engram_int8_lookup_cpu(weight, scale, ids, output);
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("engram_int8_lookup_cpu", &engram_int8_lookup_cpu_entry,
        "In-repo Engram CPU int8 lookup dequant");
}
"""

_ENGRAM_KERNEL_MODULE_NAME = "engram_kernel_under_test"
_engram_kernel_cache: dict[str, Any] = {}


def _cpp_compiler() -> str | None:
    """A usable C++ compiler on PATH, or None.

    Its absence is the only reason to skip the kernel case: the build is a
    direct compiler invocation (no ninja, no xllm_ops registration needed).
    """
    for candidate in ("c++", "clang++", "g++", "cc"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def _neon_coverage_note(machine: str) -> str:
    """Architecture note: the kernel's NEON path only exists on aarch64.

    The in-repo kernel's hand-written NEON fast path is compiled only on
    aarch64; elsewhere the same translation unit runs its scalar path. That is
    fine -- the case still validates the golden and the device-reference rows --
    but it is not NEON coverage, so the note states it explicitly rather than
    letting a green run on x86 be misread as exercising NEON. Returns "" on
    aarch64 (real coverage) so the note only appears where NEON is absent.
    """
    if machine.lower() in ("aarch64", "arm64"):
        return ""
    return f" [NEON not covered: host arch is {machine}, the kernel falls back to its scalar path]"


def _build_engram_kernel_module(source: Path) -> Any:
    """Compile ``source`` into an importable module; return the op callable.

    Drives the C++ compiler directly with the installed torch's own
    include/library paths, so the real kernel runs on any host that has a
    compiler plus torch -- no ``ninja`` and no full xLLM build. The extension is
    a throwaway under a temp dir and is cached per source path.
    """
    cache_key = str(source)
    if cache_key in _engram_kernel_cache:
        return _engram_kernel_cache[cache_key]
    if not source.exists():
        raise FileNotFoundError(f"engram CPU kernel source missing: {source}")
    compiler = _cpp_compiler()
    if compiler is None:
        raise RuntimeError("no C++ compiler on PATH")

    import hashlib
    import importlib.util
    import subprocess
    import sysconfig
    import tempfile

    from torch.utils import cpp_extension

    # The pybind module name must match ``PyInit_<name>``; keying it on the
    # source digest keeps distinct builds (e.g. the mutation self-check) from
    # resolving to an already-loaded extension of the same name.
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    module_name = f"{_ENGRAM_KERNEL_MODULE_NAME}_{digest}"

    build_dir = tempfile.mkdtemp(prefix="engram_kernel_")
    shim = Path(build_dir) / "engram_pybind_shim.cpp"
    shim.write_text(_ENGRAM_PYBIND_SHIM, encoding="utf-8")
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    module_path = Path(build_dir) / f"{module_name}{suffix}"

    if sys.platform == "darwin":
        command = [compiler, "-bundle", "-undefined", "dynamic_lookup", "-fPIC", "-O2"]
    else:
        command = [compiler, "-shared", "-fPIC", "-O2"]
    command += [
        f"-DTORCH_EXTENSION_NAME={module_name}",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
    ]
    for include in [*cpp_extension.include_paths(), sysconfig.get_paths()["include"]]:
        command += ["-I", include]
    library_dirs = cpp_extension.library_paths()
    for library_dir in library_dirs:
        command += ["-L", library_dir]
    link = ["-ltorch", "-ltorch_cpu", "-lc10", "-ltorch_python"]
    if sys.platform != "darwin":
        link += [f"-Wl,-rpath,{library_dir}" for library_dir in library_dirs]
    tail = [str(source), str(shim), *link, "-o", str(module_path)]

    # torch's default C++ standard tracks its headers (C++20 on current torch,
    # C++17 on older); try the newer first and fall back.
    error = ""
    for std_flag in ("-std=c++20", "-std=c++17"):
        completed = subprocess.run([*command, std_flag, *tail], capture_output=True, text=True)
        if completed.returncode == 0:
            break
        error = completed.stderr[-4000:]
    else:
        raise RuntimeError(f"failed to compile {source}:\n{error}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import compiled engram kernel {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    op = module.engram_int8_lookup_cpu
    _engram_kernel_cache[cache_key] = op
    return op


@pytest.mark.skipif(
    _cpp_compiler() is None,
    reason=(
        "no C++ compiler on PATH; cannot JIT-build the in-repo engram CPU kernel"
        + _neon_coverage_note(platform.machine())
    ),
)
def test_engram_lookup_op_kernel_matches_golden() -> None:
    """The real in-repo CPU int8 lookup kernel executes and matches goldens.

    Case 1 pins a fixed golden: power-of-two scales make every ``int8 * scale``
    product exactly bf16-representable, so the expected rows (and two literal
    bf16 encodings) are hand-derivable and the comparison is bitwise. Case 2
    compares against the device-path reference on random tables with fp32 group
    scales. Case 3 repeats the exact-golden idea with ``numel >= 256`` so the
    dynamic-grain parallel branch runs, deriving its expected rows independently
    in Python.
    """
    op = _build_engram_kernel_module(_ENGRAM_KERNEL_SOURCE)

    # Case 1: exact power-of-two scales -> fixed golden.
    width = 32
    weight = torch.empty(2, width, dtype=torch.int8)
    weight[0] = torch.arange(1, width + 1, dtype=torch.int8)
    weight[1] = -torch.arange(1, width + 1, dtype=torch.int8)
    scale = torch.tensor([[2.0**-8], [2.0**-7]], dtype=torch.float32)
    ids = torch.tensor([0, 1], dtype=torch.int64)
    out = torch.empty(ids.numel(), width, dtype=torch.bfloat16)
    op(weight, scale, ids, out)

    expected = torch.tensor(
        [
            [value / 256.0 for value in range(1, width + 1)],
            [-value / 128.0 for value in range(1, width + 1)],
        ],
        dtype=torch.bfloat16,
    )
    assert torch.equal(out, expected), (
        f"golden mismatch: {out[0, :4].tolist()}{_neon_coverage_note(platform.machine())}"
    )
    bits = out.view(torch.int16).to(torch.int32) & 0xFFFF
    assert int(bits[0, 0]) == 0x3B80  # bf16(2**-8)
    assert int(bits[1, 0]) == 0xBC00  # bf16(-(2**-7))

    # Case 2: random int8 + fp32 group scales vs the device-path reference.
    generator = torch.Generator().manual_seed(77)
    rows, dim = 37, 64
    weight = torch.randint(-127, 128, (rows, dim), dtype=torch.int8, generator=generator)
    scale = (torch.rand(rows, dim // 32, generator=generator) * 0.05 + 1e-3).float()
    ids = torch.randint(0, rows, (11,), dtype=torch.int64)
    out = torch.empty(ids.numel(), dim, dtype=torch.bfloat16)
    op(weight, scale, ids, out)
    reference = torch.empty_like(out)
    _reference_engram_lookup(weight, scale, ids, reference)
    _assert_bf16_within_one_ulp(out, reference, "engram CPU kernel vs device reference")

    # Case 3: numel == 256 pins the ``kParallelMinRows`` boundary and runs the
    # dynamic-grain parallel branch (Case 2's numel == 11 stays on the serial
    # path). Per-group scales are powers of two, so every ``int8 * scale`` value
    # is exactly bf16-representable and the golden is a strict bitwise match.
    # The expected rows are built independently in Python (index the raw int8
    # rows and multiply by the raw scales) -- neither the kernel under test nor
    # the eager reference helper is reused.
    width = 64
    rows = 259
    generator = torch.Generator().manual_seed(4242)
    weight = torch.randint(-127, 128, (rows, width), dtype=torch.int8, generator=generator)
    groups = width // 32
    scale = torch.empty(rows, groups, dtype=torch.float32)
    for row in range(rows):
        for group in range(groups):
            scale[row, group] = 2.0 ** -((row + group) % 8 + 1)
    ids = torch.randint(0, rows, (256,), dtype=torch.int64, generator=generator)
    out = torch.empty(ids.numel(), width, dtype=torch.bfloat16)
    op(weight, scale, ids, out)

    weight_rows = weight.tolist()
    scale_rows = scale.tolist()
    expected = torch.tensor(
        [
            [weight_rows[id_row][column] * scale_rows[id_row][column // 32] for column in range(width)]
            for id_row in ids.tolist()
        ],
        dtype=torch.bfloat16,
    )
    assert torch.equal(out, expected), (
        f"parallel golden mismatch (numel={ids.numel()}): {out[0, :4].tolist()} vs {expected[0, :4].tolist()}"
    )


@pytest.mark.skipif(
    _cpp_compiler() is None,
    reason=(
        "no C++ compiler on PATH; cannot JIT-build the in-repo engram CPU kernel"
        + _neon_coverage_note(platform.machine())
    ),
)
def test_engram_lookup_op_kernel_boundary_goldens() -> None:
    """Exact goldens at the numel boundaries of the grain computation.

    ``kParallelMinRows == 256`` is the only control-flow boundary in the
    kernel's grain: below it the whole range is handed to ``at::parallel_for``
    (which runs it serially when ``numel <= grain``), at or above it the
    row-scaled/capped branch is taken. These sizes pin the empty range (0), the
    smallest real batch (1), the last serial size (255), the first parallel size
    (256) and one past it (257) against a bitwise golden built independently in
    Python. Per-group scales are powers of two, so every ``int8 * scale``
    product is exactly bf16-representable and the comparison is bitwise.
    """
    op = _build_engram_kernel_module(_ENGRAM_KERNEL_SOURCE)

    width = 64
    rows = 259
    generator = torch.Generator().manual_seed(31337)
    weight = torch.randint(-127, 128, (rows, width), dtype=torch.int8, generator=generator)
    groups = width // 32
    scale = torch.empty(rows, groups, dtype=torch.float32)
    for row in range(rows):
        for group in range(groups):
            scale[row, group] = 2.0 ** -((row + group) % 8 + 1)
    weight_rows = weight.tolist()
    scale_rows = scale.tolist()

    for numel in (0, 1, 255, 256, 257):
        ids = torch.zeros(numel, dtype=torch.int64)
        if numel:
            ids = torch.randint(0, rows, (numel,), dtype=torch.int64, generator=torch.Generator().manual_seed(numel))
        out = torch.empty(numel, width, dtype=torch.bfloat16)
        op(weight, scale, ids, out)

        # Expected rows built from the raw int8 rows and raw scales; neither the
        # kernel under test nor the eager reference helper is reused.
        expected = torch.tensor(
            [
                [weight_rows[id_row][column] * scale_rows[id_row][column // 32] for column in range(width)]
                for id_row in ids.tolist()
            ],
            dtype=torch.bfloat16,
        ).reshape(numel, width)
        assert torch.equal(out, expected), (
            f"boundary golden mismatch (numel={numel}): {out[0, :4].tolist()} vs {expected[0, :4].tolist()}"
        )


@pytest.mark.skipif(
    not _engram_lookup_op_registered() or not _host_pinning_available(),
    reason=(
        "regular CI does not run the real CPU lookup kernel: it needs a full "
        "xLLM binary registration plus pinnable host staging. The real kernel "
        "is covered by test_engram_lookup_op_kernel_matches_golden, which "
        "JIT-compiles and runs the in-repo source; this case is the op-call "
        "routing/contract check that only runs on such a host. Its skipif "
        "probe (like the case body) touches the function under test, so a "
        "green skip here is NOT evidence the op path works."
    ),
)
def test_engram_lookup_op_matches_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    """The offload path always runs the real CPU op, matching the eager rows."""
    import xllm.python.models.deepseek_v41_engram as engram

    embed = _make_offload_engram(seed=201)
    ids = torch.randint(0, embed.num_embeddings, (9, embed.n_hash_cols), dtype=torch.int64)
    ids[0, 0] = -1  # padding / hash id outside every head bucket
    ids[1, 1] = embed.num_embeddings + 7  # foreign head id

    real_op = engram._get_engram_lookup_op()
    calls: list[int] = []

    def counting_op(weight: torch.Tensor, scale: torch.Tensor, flat: torch.Tensor, out: torch.Tensor) -> None:
        calls.append(int(flat.numel()))
        return real_op(weight, scale, flat, out)

    monkeypatch.setattr(engram, "_get_engram_lookup_op", lambda: counting_op)

    fused = embed._local_block_offload(ids).clone()
    flat = embed._ids_host[: ids.shape[0]].reshape(-1)
    assert calls == [int(flat.numel())], "the offload path must route through the op"

    # Reference: the eager host dequant the op fuses (group-32 scale, bf16),
    # with the unowned columns zeroed like the device path's torch.where.
    rows_out = embed._rows_host[: ids.shape[0]].view(-1, embed.dim)
    _reference_engram_lookup(embed.weight, embed.weight_scale, flat, rows_out)
    rows_out.masked_fill_(~embed._owned_host[: ids.shape[0]].reshape(-1, 1), 0.0)
    expected = rows_out.view(ids.shape[0], embed.part_n_hash_cols, embed.dim).clone()

    _assert_bf16_within_one_ulp(fused, expected, "engram lookup op vs eager")

    # Non-vacuity: the masked (unowned) rows are exactly zero while an owned
    # row is not, so the comparison is not against an all-zero block.
    local_ids = ids[:, embed.head_start : embed.head_start + embed.part_n_hash_cols]
    owned = (local_ids >= embed.vocab_start) & (local_ids < embed.vocab_end)
    assert not bool(owned.all())
    assert torch.equal(fused[~owned], torch.zeros_like(fused[~owned]))
    assert float(fused[owned].abs().max()) > 0.0


def test_engram_lookup_op_missing_is_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing op registration must raise, never silently fall back to eager.

    Forcing the op-missing environment (rather than probing
    ``_get_engram_lookup_op`` itself) is deliberate: a probe of the function
    under test would ``skip`` exactly when the regression this case guards -- a
    silent eager fallback -- is reintroduced, leaving the guard with zero power.
    Here the resolver is reset and the ``torch.ops.xllm_ops`` namespace is
    swapped for one whose every attribute access raises ``AttributeError``, so
    the case runs unconditionally and fails if the resolver stops raising.
    """
    import xllm.python.models.deepseek_v41_engram as engram

    class _MissingOpNamespace:
        def __getattr__(self, name: str) -> object:
            raise AttributeError(name)

    monkeypatch.setattr(engram, "_engram_lookup_op", None)
    monkeypatch.setattr(torch.ops, "xllm_ops", _MissingOpNamespace(), raising=False)
    with pytest.raises(RuntimeError, match="not registered"):
        engram._get_engram_lookup_op()


def test_engram_offload_guards_are_mockable_without_compiling_the_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Python-layer fail-loud guards run without JIT-compiling the C++ kernel.

    ``test_engram_lookup_op_kernel_matches_golden`` compiles the in-repo source
    with the host C++ toolchain, so it is skipped when there is no compiler (or an
    ABI mismatch) -- it cannot by itself guard the Python contract. This
    case pins that layer with a mocked op and no C++ at all: (1) a contract
    violation must raise *before* the op is ever called, and (2) a missing op
    registration must propagate its loud ``RuntimeError`` rather than a silent
    eager fallback. It needs neither a compiler nor pinnable host staging.

    Discriminating: dropping the ``if reason: raise`` guard makes ``mock_op`` run
    (``calls`` non-empty) and no ``RuntimeError`` fire, so the first half fails;
    turning a missing resolver into a silent fallback makes the second
    ``pytest.raises`` fail.
    """
    import xllm.python.models.deepseek_v41_engram as engram

    embed = _make_offload_engram(seed=207)
    ids = torch.randint(0, embed.num_embeddings, (6, embed.n_hash_cols), dtype=torch.int64)
    calls: list[int] = []

    def mock_op(weight: torch.Tensor, scale: torch.Tensor, flat: torch.Tensor, out: torch.Tensor) -> None:
        calls.append(int(flat.numel()))

    monkeypatch.setattr(engram, "_get_engram_lookup_op", lambda: mock_op)

    # (1) A forced contract violation (quant_block != 32, checked before the
    # pinning checks so this runs even on a host that cannot pin) must raise
    # before the op is reached.
    monkeypatch.setattr(embed, "quant_block", 16)
    with pytest.raises(RuntimeError, match="cannot run the CPU int8 lookup op"):
        embed._local_block_offload(ids)
    assert calls == [], "the op must not be called when the contract is violated"

    # (2) With the contract satisfied (stubbed, to stay independent of host
    # pinning), a missing resolver must still surface its loud error.
    def missing_op() -> object:
        raise RuntimeError("engram host offload requires the in-repo CPU int8 lookup op; not registered")

    monkeypatch.setattr(embed, "quant_block", 32)
    monkeypatch.setattr(embed, "_engram_lookup_op_contract_error", lambda ids, out: "")
    monkeypatch.setattr(engram, "_get_engram_lookup_op", missing_op)
    with pytest.raises(RuntimeError, match="not registered"):
        embed._local_block_offload(ids)
    assert calls == [], "a missing op must not be papered over by calling anything else"


def test_engram_lookup_op_contract_rejects_mismatches(monkeypatch: pytest.MonkeyPatch) -> None:
    """The contract only passes for the exact op operands; every miss is flagged."""
    if not _host_pinning_available():
        pytest.skip("host staging cannot be pinned; the op contract cannot be met")

    embed = _make_offload_engram(seed=203)
    ids = torch.randint(0, embed.num_embeddings, (5, embed.n_hash_cols), dtype=torch.int64)
    embed._ensure_host_buffers(ids.shape[0])
    flat = embed._ids_host[: ids.shape[0]].reshape(-1)
    rows_out = embed._rows_host[: ids.shape[0]].view(-1, embed.dim)

    assert embed._engram_lookup_op_contract_error(flat, rows_out) == ""

    # The op's group size is fixed at 32.
    monkeypatch.setattr(embed, "quant_block", 16)
    assert embed._engram_lookup_op_contract_error(flat, rows_out)
    monkeypatch.setattr(embed, "quant_block", 32)

    # A device-resident table is not the offload contract.
    monkeypatch.setattr(embed, "offload", False)
    assert embed._engram_lookup_op_contract_error(flat, rows_out)
    monkeypatch.setattr(embed, "offload", True)

    # Dtype / shape / contiguity / pinning mismatches on the operands.
    assert embed._engram_lookup_op_contract_error(flat.to(torch.int32), rows_out)
    assert embed._engram_lookup_op_contract_error(flat[:1], rows_out)
    assert embed._engram_lookup_op_contract_error(flat, rows_out.to(torch.float32))
    assert embed._engram_lookup_op_contract_error(flat, torch.empty(rows_out.shape, dtype=rows_out.dtype))
    # An unpinned ids operand (a pageable clone of the pinned
    # staging view) must be rejected by the ids-pinned contract check. Without
    # the ``flat_unpinned`` guard the ``flat`` and ``flat_unpinned`` assertions
    # could both pass for the wrong reason on a host whose clone happens to stay
    # pinned, so the explicit ``not ... is_pinned()`` pins the premise.
    flat_unpinned = flat.clone()
    assert not flat_unpinned.is_pinned()
    assert embed._engram_lookup_op_contract_error(flat_unpinned, rows_out)
    assert embed._engram_lookup_op_contract_error(flat, rows_out) == ""


def test_offload_construction_rejects_non_32_group() -> None:
    """An offloaded table with ``quant_block != 32`` is refused at construction.

    The fused CPU int8 lookup op hard-codes a 32-wide group, so such a table
    could never be served; failing in ``__init__`` avoids loading the whole host
    table first (the runtime ``_engram_lookup_op_contract_error`` check stays as
    defence in depth). A non-offload table is unaffected -- any block size is
    fine on the device/HBM path.
    """
    import xllm.python.models.deepseek_v41_engram as engram

    def make(*, quant_block: int, offload: bool) -> engram.EngramEmbedding:
        return engram.EngramEmbedding(
            12,
            64,
            [5, 7],
            1,
            0,
            quant_block=quant_block,
            storage=engram.ENGRAM_STORAGE_INT8,
            device=torch.device("cpu"),
            offload=offload,
        )

    with pytest.raises(ValueError, match="quant_block=32"):
        make(quant_block=64, offload=True)

    # Supported: any block while on the device path, block 32 while offloading.
    assert make(quant_block=64, offload=False).offload is False
    assert make(quant_block=32, offload=True).offload is True


def test_kernel_case_neon_note_is_architecture_gated() -> None:
    """The kernel case must not claim NEON coverage off aarch64.

    The case is skipped only when there is no compiler -- never by host
    architecture -- so an aarch64 host always runs it (the NEON path). On other
    hosts the same translation unit runs its scalar fallback, which is valid but
    is not NEON, so the note must say so. Driving an explicit machine string
    keeps the check independent of the host running the suite.
    """
    assert _neon_coverage_note("aarch64") == ""
    assert _neon_coverage_note("arm64") == ""
    x86 = _neon_coverage_note("x86_64")
    assert "NEON" in x86
    assert "x86_64" in x86


# --- fallback_to_device: two-phase commit ----------------------------------
#
# ``fallback_to_device`` copies ``weight`` then ``weight_scale`` to the
# accelerator. The copies must run into locals and only commit once both have
# succeeded: an injected failure on either ``.to()`` has to leave the embedding
# exactly as it was. The failure is driven by monkeypatching ``Tensor.to`` so the
# first copy "succeeds" into fresh storage and the second raises, which under the
# old per-parameter commit left ``weight`` moved and ``weight_scale`` on the
# host -- the half-demoted state the tests below reject.
#
# Scope of these cases: they cover only a *synchronous* exception raised by
# ``.to()`` itself. An asynchronous NPU error that only surfaces at a later sync
# point is not observable here and is outside what the test (or the two-phase
# commit state guarantee) covers.


def _copy_to_fresh_storage(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    """Stand-in for ``Tensor.to(device)``: a copy into fresh storage."""
    return self.detach().clone()


@pytest.mark.parametrize("fail_on_call", [1, 2])
def test_fallback_to_device_leaves_everything_untouched_when_a_copy_fails(
    monkeypatch: pytest.MonkeyPatch, fail_on_call: int
) -> None:
    """A failed ``.to()`` must leave both parameters, the flag and scratch intact.

    ``fail_on_call == 1`` is the first-copy failure and ``fail_on_call == 2`` the
    second -- the case the old per-parameter commit got wrong (``weight`` moved,
    ``weight_scale`` not). Both must leave ``weight``'s and ``weight_scale``'s
    storage, contents, ``offload`` and the host scratch/staging state byte-for-byte
    as they were. Discriminating: restoring the old ordering fails the address
    assertion for ``fail_on_call == 2`` (see the module note above).

    The injected failure is a synchronous exception from ``.to()`` itself. An
    asynchronous NPU error that only surfaces at a later sync point would not be
    seen here and is outside this case and the two-phase commit state guarantee.
    """
    embed = _make_offload_engram(seed=219)
    # Non-trivial scratch + eager staging to assert untouched. Graph staging must
    # stay unallocated: ``fallback_to_device`` refuses once a captured-graph block
    # exists (it is the pre-capture-only guard).
    ids = torch.randint(0, embed.num_embeddings, (3, embed.n_hash_cols), dtype=torch.int64)
    embed._ensure_host_buffers(ids.shape[0])

    weight_before = embed.weight.detach().clone()
    scale_before = embed.weight_scale.detach().clone()
    weight_ptr = embed.weight.data.data_ptr()
    scale_ptr = embed.weight_scale.data.data_ptr()
    ids_host, owned_host, rows_host = embed._ids_host, embed._owned_host, embed._rows_host
    eager_block = embed._eager_staged_device
    state_before = (
        embed.offload,
        embed._host_capacity,
        embed._ids_host_pinned,
        embed._rows_host_pinned,
        embed._graph_staged_device,
        embed._graph_capacity,
        embed._eager_capacity,
    )

    calls = {"n": 0}

    def faulty_to(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        calls["n"] += 1
        if calls["n"] == fail_on_call:
            raise RuntimeError("injected .to() failure")
        return _copy_to_fresh_storage(self)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", faulty_to)
        with pytest.raises(RuntimeError, match="injected .to\\(\\) failure"):
            embed.fallback_to_device()

    assert calls["n"] == fail_on_call, "fallback must stop at the first failing copy"
    assert embed.offload is True
    assert embed.weight.data.data_ptr() == weight_ptr
    torch.testing.assert_close(embed.weight.detach(), weight_before, check_dtype=True)
    assert embed.weight_scale.data.data_ptr() == scale_ptr
    torch.testing.assert_close(embed.weight_scale.detach(), scale_before, check_dtype=True)
    assert embed._ids_host is ids_host
    assert embed._owned_host is owned_host
    assert embed._rows_host is rows_host
    assert embed._eager_staged_device is eager_block
    assert (
        embed.offload,
        embed._host_capacity,
        embed._ids_host_pinned,
        embed._rows_host_pinned,
        embed._graph_staged_device,
        embed._graph_capacity,
        embed._eager_capacity,
    ) == state_before


def test_fallback_to_device_commits_both_tables_and_clears_host_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Success path: both tables move, ``offload`` flips and host scratch drops."""
    embed = _make_offload_engram(seed=220)
    ids = torch.randint(0, embed.num_embeddings, (3, embed.n_hash_cols), dtype=torch.int64)
    embed._ensure_host_buffers(ids.shape[0])
    weight_before = embed.weight.detach().clone()
    scale_before = embed.weight_scale.detach().clone()
    weight_ptr = embed.weight.data.data_ptr()
    scale_ptr = embed.weight_scale.data.data_ptr()

    # Both copies succeed into fresh storage; the commit must adopt exactly those.
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", _copy_to_fresh_storage)
        assert embed.fallback_to_device() is True

    assert embed.offload is False
    assert embed.weight.data.data_ptr() != weight_ptr
    assert embed.weight_scale.data.data_ptr() != scale_ptr
    torch.testing.assert_close(embed.weight.detach(), weight_before, check_dtype=True)
    torch.testing.assert_close(embed.weight_scale.detach(), scale_before, check_dtype=True)
    assert (embed._ids_host, embed._owned_host, embed._rows_host) == (None, None, None)
    assert embed._host_capacity == 0
    assert embed._ids_host_pinned is False
    assert embed._rows_host_pinned is False
    assert (embed._graph_staged_device, embed._graph_capacity) == (None, 0)
    assert (embed._eager_staged_device, embed._eager_capacity) == (None, 0)


# ---------------------------------------------------------------------------
# EngramEmbedding component tests
#
# Migrated from test_deepseek_v41_model.py: these exercise only the standalone
# ``EngramEmbedding`` / ``deepseek_v41_engram`` helpers (layout, hash, TP
# sharding, int8 storage, host-offload config / staging / fallback), never a
# constructed ``DeepseekV41Model``. Definitions shared with the
# model-integration tests are duplicated here -- not imported -- so this module
# stays importable on its own.
# ---------------------------------------------------------------------------


def _engram_test_token_map() -> list[int]:
    # Same tiny token map as test_deepseek_v41_model.py (duplicated so this
    # module does not import the model test module).
    return (torch.arange(64) % 16).tolist()


def test_engram_layout_prime_buckets() -> None:
    layout = EngramLayout(
        layer_ids=[1, 14],
        num_embeddings=[384006168, 384016682],
        max_ngram_size=3,
        n_heads=2,
        head_dim=32,
        vocab_size=64,
        compressed_vocab_size=16,
        pad_token_id=0,
    )
    # Primes are drawn from 63 upward, never reused across layers/heads.
    assert layout.primes[0][0] == (67, 71)
    assert layout.primes[0][1] == (73, 79)
    # Layer 14 continues the shared search: 83, 89, 97, 101.
    assert layout.primes[1][0] == (83, 89)
    assert layout.primes[1][1] == (97, 101)
    assert layout.n_hash_cols == 4
    # Offsets are exclusive prefix sums of the flattened head sizes.
    assert torch.equal(layout.offsets, torch.tensor([[0, 67, 138, 211], [0, 83, 172, 269]]))


def test_engram_token_map_normalization_chain() -> None:
    # Case / accent / whitespace folding collapses onto one compressed id.
    lookup, size = build_compressed_token_map_from_texts(["The", "the", "THE", " thé ", "x", "y"])
    assert size == 3
    assert lookup[0] == lookup[1] == lookup[2] == lookup[3]
    assert lookup[4] != lookup[5]
    assert normalize_token_text("  A\tB\nC ") == "a b c"
    # A lone space survives Strip via the private-use sentinel.
    assert normalize_token_text(" ") == " "


def _tiny_engram_hash_state() -> EngramHashState:
    from xllm.python.models.deepseek_v41_engram import compute_hash_multipliers

    layout = EngramLayout(
        layer_ids=[1],
        num_embeddings=[290],
        max_ngram_size=3,
        n_heads=2,
        head_dim=32,
        vocab_size=64,
        compressed_vocab_size=16,
        pad_token_id=0,
    )
    multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, 16)
    return EngramHashState(layout, torch.tensor(_engram_test_token_map(), dtype=torch.int64), multipliers)


def test_engram_hash_rolling_xor_and_dead_tokens() -> None:
    state = _tiny_engram_hash_state()
    pad_id = state.pad_id
    # Two sequences of 3 tokens; sequence 0's token at position 1 is dead.
    ids = torch.tensor([5, 6, 7, 10, 11, 12])
    positions = torch.tensor([0, 1, 2, 0, 1, 2])
    q_cu = torch.tensor([0, 3, 6])
    dead = torch.tensor([False, True, False, False, False, False])
    hashes = state(ids, positions, q_cu, dead, None, None, None, 0)
    assert hashes.shape == (6, 1, 4)

    comp = torch.tensor(_engram_test_token_map(), dtype=torch.int64)
    m = state.multipliers[0]
    # Column layout: shift 1 -> cols 0/1 (primes 67/71), shift 2 -> cols 2/3
    # (primes 73/79). `blocked` is sticky: once a lookback is out of range or
    # dead, every further shift pools pad_id into the rolling hash.
    r_pos0_2gram = (comp[5] * m[0]) ^ (pad_id * m[1])
    r_pos0_3gram = r_pos0_2gram ^ (pad_id * m[2])
    assert torch.equal(hashes[0, 0, 0], (r_pos0_2gram % 67).to(torch.int32))
    assert torch.equal(hashes[0, 0, 2], ((r_pos0_3gram % 73) + 138).to(torch.int32))
    # Position 2: the dead token at position 1 blocks shifts 1 AND 2 (sticky),
    # even though position 0 itself is a healthy chunk member.
    r_pos2_2gram = (comp[7] * m[0]) ^ (pad_id * m[1])
    r_pos2_3gram = r_pos2_2gram ^ (pad_id * m[2])
    assert torch.equal(hashes[2, 0, 0], (r_pos2_2gram % 67).to(torch.int32))
    assert torch.equal(hashes[2, 0, 2], ((r_pos2_3gram % 73) + 138).to(torch.int32))
    # Sequence 1 (no dead tokens): the 2-gram at position 1 pools position 0.
    r = (comp[11] * m[0]) ^ (comp[10] * m[1])
    assert torch.equal(hashes[4, 0, 0], (r % 67).to(torch.int32))


def test_engram_hash_slot_cache_roundtrip() -> None:
    """Generated tokens read their lookback back from the slot cache."""
    state = _tiny_engram_hash_state()
    state._cache = torch.zeros(64, dtype=torch.int32)
    # Step 1: a 2-token chunk parks its compressed ids at slots 0/1.
    ids1 = torch.tensor([3, 9])
    bt = torch.tensor([[0, 0]], dtype=torch.int32)
    slot1 = torch.tensor([0, 1], dtype=torch.int32)
    q_cu = torch.tensor([0, 2])
    state(
        ids1,
        torch.tensor([0, 1]),
        q_cu,
        torch.zeros(2, dtype=torch.bool),
        None,
        slot1,
        bt,
        32,
    )
    # Step 2: one more token; positions 0/1 precede the chunk start, so their
    # compressed ids must come from the slot cache (slots 0/1).
    comp = torch.tensor(_engram_test_token_map(), dtype=torch.int64)
    m = state.multipliers[0]
    ids2 = torch.tensor([20])
    q_cu2 = torch.tensor([0, 1])
    slot2 = torch.tensor([2], dtype=torch.int32)
    hashes = state(
        ids2,
        torch.tensor([2]),
        q_cu2,
        torch.zeros(1, dtype=torch.bool),
        None,
        slot2,
        bt,
        32,
    )
    r = (comp[20] * m[0]) ^ (comp[9] * m[1]) ^ (comp[3] * m[2])
    assert torch.equal(hashes[0, 0, 2], ((r % 73) + 138).to(torch.int32))


def test_engram_mixed_batch_deduplicates_ring_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty sequences must not hide a multi-token ring collision."""
    import xllm.python.models.deepseek_v41_engram as engram

    state = _tiny_engram_hash_state()
    state._cache = torch.zeros(16, dtype=torch.int32)
    ids = torch.tensor([3, 9, 20])
    # q_lens=[2, 0, 1]: total tokens equals batch size, which made the old
    # aggregate-count heuristic incorrectly classify this as decode-only.
    query_start_loc = torch.tensor([0, 2, 2, 3])
    slots = torch.tensor([4, 4, 5], dtype=torch.int32)
    calls: list[torch.Tensor] = []
    original = engram.last_ring_writes

    def recording_last_writes(physical_slots: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(physical_slots.clone())
        return original(physical_slots, values)

    monkeypatch.setattr(engram, "last_ring_writes", recording_last_writes)
    state(
        ids,
        torch.tensor([0, 1, 0]),
        query_start_loc,
        torch.zeros(3, dtype=torch.bool),
        None,
        slots,
        None,
        8,
    )

    assert len(calls) == 1
    compressed = state.token_map[ids].to(torch.int32)
    assert state._cache[4] == compressed[1]
    assert state._cache[5] == compressed[2]


@pytest.mark.parametrize("device_name", ["cpu", "npu"])
def test_engram_graph_writes_keep_active_slots_and_repeated_padding(
    device_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise deduplicate=False and real graph replay on valid row slots.

    Active decode/verify rows address distinct slots. Multiple padding rows
    may map to reserved slot 0, but all publish its unchanged value. Conflicting
    active slots are not a valid input and have no last-writer graph guarantee.
    """
    import xllm.python.models.deepseek_v41_engram as engram
    from xllm.python.model_executor.forward_context import (
        AclGraphCaptureContext,
        AclGraphExecutionState,
        ForwardContext,
        forward_context,
    )

    if device_name == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU is unavailable")
    device = torch.device("npu:0" if device_name == "npu" else "cpu")
    state = _tiny_engram_hash_state().to(device)
    state._cache = torch.full((24,), 91, dtype=torch.int32, device=device)
    ids = torch.tensor([3, 9, 20, 7, 11, 12], device=device)
    positions = torch.zeros(6, dtype=torch.int64, device=device)
    starts = torch.arange(7, device=device)
    dead = torch.tensor([False, True, False, False, False, False], device=device)
    slots = torch.tensor([8, 11, -1, -1, 14, -1], dtype=torch.int32, device=device)

    def unexpected_eager_dedup(*args: object) -> None:
        raise AssertionError("Graph writes must retain their fixed operand shapes")

    monkeypatch.setattr(engram, "last_ring_writes", unexpected_eager_dedup)
    context = ForwardContext(None, device, None, [], execution_state=AclGraphExecutionState({}))

    def execute() -> torch.Tensor:
        with forward_context(context):
            return state(ids, positions, starts, dead, None, slots, None, 8)

    def check_cache() -> None:
        valid = slots >= 0
        expected = torch.full_like(state._cache, 91)
        compressed = state.token_map[ids].to(torch.int32)
        compressed = torch.where(dead, -1, compressed)
        expected[slots[valid].long()] = compressed[valid]
        torch.testing.assert_close(state._cache, expected, atol=0, rtol=0)

    with torch.inference_mode():
        execute()
        check_cache()
        if device_name == "cpu":
            return
        graph = torch.npu.NPUGraph()
        stream = torch.npu.Stream()
        torch.npu.synchronize()
        context = ForwardContext(
            None,
            device,
            None,
            [],
            execution_state=AclGraphExecutionState({}),
            acl_graph=AclGraphCaptureContext(stream, []),
        )
        with torch.npu.graph(graph, stream=stream):
            captured = execute()
        for active_slots, next_ids in (
            ([9, 12, -1, -1, 15, -1], [5, 6, 7, 8, 9, 10]),
            ([8, 11, -1, -1, 14, -1], [10, 9, 8, 7, 6, 5]),
        ):
            state._cache.fill_(91)
            slots.copy_(torch.tensor(active_slots, dtype=torch.int32, device=device))
            ids.copy_(torch.tensor(next_ids, device=device))
            graph.replay()
            torch.npu.synchronize()
            check_cache()
            # With position=0 each request's history is padding; hashes are
            # independently computed without a slot cache or graph dedup.
            reference = _tiny_engram_hash_state().to(device)
            expected_hashes = reference(ids, positions, starts, dead, None, None, None, 8)
            torch.testing.assert_close(captured, expected_hashes, atol=0, rtol=0)


@pytest.mark.parametrize(
    "prefix,chunk,window", [(7, 2, 8), (8, 8, 8), (9, 17, 8), (31, 33, 8), (129, 3, 128), (5000, 3, 128)]
)
@pytest.mark.parametrize("dead_prefix", [False, True])
def test_engram_chunked_hash_preserves_history_across_ring_wrap(
    prefix: int,
    chunk: int,
    window: int,
    dead_prefix: bool,
) -> None:
    """Chunk history and following decode retain absolute ring addressing."""
    state = _tiny_engram_hash_state()
    reference = _tiny_engram_hash_state()
    torch.manual_seed(611)
    ids = torch.randint(1, 64, (prefix + chunk + 3,))
    dead = torch.zeros_like(ids, dtype=torch.bool)
    dead[prefix - 1] = dead_prefix
    state._cache = torch.zeros(window * 2, dtype=torch.int32)
    table = torch.tensor([[1]], dtype=torch.int32)
    contexts = [0, prefix, prefix + chunk, prefix + chunk + 1, prefix + chunk + 2, prefix + chunk + 3]
    for start, end in zip(contexts, contexts[1:]):
        positions = torch.arange(start, end)
        actual = state(
            ids[start:end],
            positions,
            torch.tensor([0, end - start]),
            dead[start:end],
            None,
            window + positions % window,
            table,
            window,
        )
        expected = reference(
            ids[:end], torch.arange(end), torch.tensor([0, end]), dead[:end], None, None, None, window
        )[start:end]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        tail = torch.arange(max(0, end - window), end)
        compressed = state.token_map[ids[tail]].to(torch.int32)
        compressed = torch.where(dead[tail], torch.full_like(compressed, -1), compressed)
        torch.testing.assert_close(state._cache[window + tail % window], compressed, atol=0, rtol=0)


def test_engram_explicit_lookback_beats_slot_cache() -> None:
    """Resolution order is chunk -> explicit lookback -> slot cache.

    A real ``lookback_token_ids`` id must win even when the cache holds a
    different value for the same position; an unknown (``-1``) id must fall
    through to the cache; an image-dead id must block the n-gram instead of
    letting the cache resurrect a live lookback.
    """
    state = _tiny_engram_hash_state()
    state._cache = torch.zeros(64, dtype=torch.int32)
    comp = torch.tensor(_engram_test_token_map(), dtype=torch.int64)
    m = state.multipliers[0]
    ids = torch.tensor([5])
    positions = torch.tensor([1])
    q_cu = torch.tensor([0, 1])
    dead = torch.zeros(1, dtype=torch.bool)
    bt = torch.tensor([[0, 0]], dtype=torch.int32)
    # Chunk start is position 1, so shift 1 (position 0) is out of chunk and
    # reads slot 0: seed it with a live value distinct from any lookback id.
    state._cache[0] = int(comp[9])

    def _hash_with_lookback(token_id: int) -> torch.Tensor:
        return state(
            ids,
            positions,
            q_cu,
            dead,
            torch.tensor([[token_id]], dtype=torch.int64),
            None,
            bt,
            32,
        )

    chunk_roll = comp[5] * m[0]
    # A supplied id (7) outranks the cache's 9.
    explicit = _hash_with_lookback(7)
    explicit_roll = chunk_roll ^ (comp[7] * m[1])
    assert torch.equal(explicit[0, 0, 0], (explicit_roll % 67).to(torch.int32))
    assert torch.equal(explicit[0, 0, 1], ((explicit_roll % 71) + 67).to(torch.int32))
    # -1 means unknown, so the slot cache supplies the lookback (9).
    unknown = _hash_with_lookback(-1)
    cache_roll = chunk_roll ^ (comp[9] * m[1])
    assert torch.equal(unknown[0, 0, 0], (cache_roll % 67).to(torch.int32))
    assert torch.equal(unknown[0, 0, 1], ((cache_roll % 71) + 67).to(torch.int32))
    # An image-dead id blocks the n-gram (pads) rather than reading the cache.
    dead_lookback = _hash_with_lookback(ENGRAM_IMAGE_SENTINEL_BASE_ID)
    assert torch.equal(dead_lookback[0, 0, 0], (chunk_roll % 67).to(torch.int32))
    assert not torch.equal(dead_lookback[0, 0, 0], (cache_roll % 67).to(torch.int32))


def test_engram_hash_state_builder_roundtrip(tmp_path) -> None:
    """build_engram_hash_state() writes a sidecar from_config can load.

    Builds a tiny tokenizer (Hello/hello fold onto one compressed id), runs
    the in-module builder end to end, and loads the result through the
    serving path. Guards the builder the from_config error message points at.
    """
    pytest.importorskip("tokenizers")
    pytest.importorskip("numpy")
    import json

    from tokenizers import Tokenizer, models

    tok = Tokenizer(
        models.WordLevel(
            vocab={"[PAD]": 0, "Hello": 1, "hello": 2, "WORLD": 3, "x": 4},
            unk_token="[PAD]",
        )
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tok.save(str(tokenizer_path))
    config = {
        "model_type": "deepseek_v41",
        "vocab_size": 5,
        "engram_layer_ids": [1],
        "engram_num_embeddings": [1000],
        "engram_max_ngram_size": 3,
        "engram_compressed_vocab_size": 4,  # [pad], hello, world, x
        "engram_pad_token_id": 0,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    written = build_engram_hash_state(str(tmp_path))
    assert written == str(tmp_path / "engram_hash_state.bin")

    # First-seen compressed ids: [PAD]->0, Hello->1, hello->1, WORLD->2, x->3.
    sidecar = _load_engram_hash_state_sidecar(str(tmp_path))
    assert sidecar is not None
    assert sidecar["token_map"].tolist() == [0, 1, 1, 2, 3]
    assert sidecar["multipliers"].shape == (1, 3)
    # The serving path accepts the generated sidecar.
    layout = EngramLayout(
        layer_ids=[1],
        num_embeddings=[1000],
        max_ngram_size=3,
        n_heads=2,
        head_dim=32,
        vocab_size=5,
        compressed_vocab_size=4,
        pad_token_id=0,
    )
    state = EngramHashState.from_config(layout, {"model_path": str(tmp_path)})
    assert state.token_map.tolist() == [0, 1, 1, 2, 3]
    assert torch.equal(state.multipliers, sidecar["multipliers"])


def test_engram_embedding_tp_sharding_rejects_nondivisible() -> None:
    """A TP size that does not divide the hash-column count is refused.

    Uniform TP shards are a hard contract: each rank owns exactly
    ``n_hash_cols // tp_size`` complete heads so the strict all-gather
    reassembles the real columns with no padding. The old
    padded-shard behaviour (and its reassembly test) is gone; a non-dividing
    world size now raises at construction and names both quantities.
    """
    head_sizes = [7, 11, 13, 5, 3]  # n_hash_cols = 5
    num_rows = sum(head_sizes)
    for tp_size in (2, 3, 4, 8):
        with pytest.raises(ValueError) as excinfo:
            EngramEmbedding(num_rows, 32, head_sizes, tp_size, 0)
        message = str(excinfo.value)
        assert "n_hash_cols=5" in message
        assert f"tp_size={tp_size}" in message
        assert "dp_shared_memory does not relax this requirement" in message

    # A 6-column head set is not divisible by 4 either.
    six = [7, 11, 13, 5, 3, 2]
    with pytest.raises(ValueError, match="n_hash_cols=6 is not divisible by tp_size=4"):
        EngramEmbedding(sum(six), 32, six, 4, 0)


def test_engram_embedding_divisible_tp_shards_reassemble() -> None:
    """A dividing tp_size: gathered per-rank blocks equal the unsharded lookup.

    Each rank owns ``n_hash_cols // tp_size`` complete heads, so concatenating
    the per-rank blocks along the head axis reproduces the full lookup exactly
    -- no padding and no post-gather trim.
    """
    head_sizes = [7, 11, 13, 5, 3, 9, 4, 6]  # n_hash_cols = 8
    dim = 32
    num_rows = sum(head_sizes)
    g = torch.Generator().manual_seed(1234)
    payload = torch.randint(0, 256, (num_rows, dim), dtype=torch.uint8, generator=g)
    # E4M3FN 0x7F/0xFF decode to NaN; NaN != NaN breaks the exact-equality
    # checks below, so remap those byte patterns to finite neighbours.
    payload = torch.where((payload == 0x7F) | (payload == 0xFF), payload - 1, payload)
    scales = torch.randint(126, 130, (num_rows, dim // 32), dtype=torch.uint8, generator=g)

    def make(tp_size: int, tp_rank: int) -> EngramEmbedding:
        emb = EngramEmbedding(num_rows, dim, head_sizes, tp_size, tp_rank)
        with torch.no_grad():
            emb.weight.copy_(payload[emb.vocab_start : emb.vocab_end])
            emb.weight_scale.copy_(scales[emb.vocab_start : emb.vocab_end])
        return emb

    ref = make(1, 0)
    t = 6
    ids = torch.zeros(t, len(head_sizes), dtype=torch.int64)
    for j, size in enumerate(head_sizes):
        ids[:, j] = torch.randint(0, size, (t,), generator=g) + sum(head_sizes[:j])

    for tp_size in (1, 2, 4, 8):
        part = len(head_sizes) // tp_size
        blocks = [make(tp_size, rank).local_block(ids) for rank in range(tp_size)]
        assert all(block.shape == (t, part, dim) for block in blocks)
        gathered = torch.cat(blocks, dim=1)
        assert gathered.shape == (t, ref.n_hash_cols, dim)
        assert torch.equal(gathered, ref.lookup(ids))


def test_engram_embedding_int8_storage_dequant() -> None:
    """int8_sym storage (Eco-Tech W8A8 export) dequantizes payload * fp32 group scales."""
    head_sizes = [7, 11, 13, 5, 3]
    dim = 32
    num_rows = sum(head_sizes)
    g = torch.Generator().manual_seed(4321)
    payload = torch.randint(-127, 128, (num_rows, dim), dtype=torch.int8, generator=g)
    scales = torch.rand(num_rows, dim // 32, generator=g).float() * 0.05 + 0.01

    emb = EngramEmbedding(num_rows, dim, head_sizes, 1, 0, storage=ENGRAM_STORAGE_INT8)
    assert emb.weight.dtype == torch.int8
    assert emb.weight_scale.dtype == torch.float32
    with torch.no_grad():
        emb.weight.copy_(payload)
        emb.weight_scale.copy_(scales)

    t = 9
    ids = torch.zeros(t, len(head_sizes), dtype=torch.int64)
    for j, size in enumerate(head_sizes):
        ids[:, j] = torch.randint(0, size, (t,), generator=g) + sum(head_sizes[:j])
    out = emb.lookup(ids)
    expected = (payload[ids].float() * scales[ids].repeat_interleave(32, dim=-1)).to(torch.bfloat16)
    assert out.shape == (t, len(head_sizes), dim)
    assert torch.equal(out, expected)


# Minimal V4.1 config carrying one engram layer, standing in for the engram
# fields of ``_DEEPSEEK_V41_TINY_CONFIG`` in test_deepseek_v41_model.py.
# ``DeepseekV41Config.from_dict`` fills every other field with its production
# default, so only the fields that drive ``engram_config`` validation need to
# be present.
_ENGRAM_OFFLOAD_CONFIG: dict[str, Any] = {
    "model_type": "deepseek_v41",
    "engram_layer_ids": [1],
    "engram_num_embeddings": [290],
}


def test_engram_offload_active_is_config_controlled() -> None:
    """``engram_config.cpu_offload`` selects the host-offload path.

    ``DeepseekV41Model`` derives ``engram_offload_active`` from
    ``engram_config.cpu_offload`` (deepseek_v41.py). Pins that the parsed
    config field -- which drives ``engram_offload_active`` for an
    engram-carrying checkpoint -- follows the config, and that the removed
    env hook is truly gone.
    """
    from xllm.python.models import deepseek_v41_engram as engram_mod
    from xllm.python.models.engram_config import EngramConfig

    # The removed env hook is truly gone: no constant, no reader.
    assert not hasattr(engram_mod, "ENGRAM_OFFLOAD_ENV")
    assert not hasattr(engram_mod, "engram_offload_requested")

    on = DeepseekV41Config.from_dict({**_ENGRAM_OFFLOAD_CONFIG, "engram_config": {"cpu_offload": True}})
    off = DeepseekV41Config.from_dict({**_ENGRAM_OFFLOAD_CONFIG, "engram_config": {"cpu_offload": False}})
    default = DeepseekV41Config.from_dict(_ENGRAM_OFFLOAD_CONFIG)
    assert on.engram_config == EngramConfig(cpu_offload=True)
    assert off.engram_config == EngramConfig(cpu_offload=False, dp_shared_memory=False)
    assert default.engram_config.cpu_offload is False

    # This config field is the sole input to the model's switch (the model ANDs
    # it with the engram-layer presence and the int8 storage demotion).
    assert bool(on.engram_layer_ids) and on.engram_config.cpu_offload is True
    assert bool(off.engram_layer_ids) and off.engram_config.cpu_offload is False
    assert bool(default.engram_layer_ids) and default.engram_config.cpu_offload is False


# ``_int8_engram_payload`` / ``_make_int8_embedding`` / ``_engram_ids`` are
# duplicated (minimal) from test_deepseek_v41_model.py: the model-integration
# tests use the same builders, and this module stays standalone rather than
# importing the model test module.
def _int8_engram_payload(head_sizes: list[int], dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_rows = sum(head_sizes)
    g = torch.Generator().manual_seed(seed)
    payload = torch.randint(-127, 128, (num_rows, dim), dtype=torch.int8, generator=g)
    scales = torch.rand(num_rows, dim // 32, generator=g).float() * 0.05 + 0.01
    return payload, scales


def _make_int8_embedding(
    head_sizes: list[int],
    dim: int,
    payload: torch.Tensor,
    scales: torch.Tensor,
    *,
    tp_size: int = 1,
    tp_rank: int = 0,
    offload: bool = False,
) -> EngramEmbedding:
    emb = EngramEmbedding(
        sum(head_sizes),
        dim,
        head_sizes,
        tp_size,
        tp_rank,
        storage=ENGRAM_STORAGE_INT8,
        device=torch.device("cpu"),
        offload=offload,
    )
    with torch.no_grad():
        emb.weight.copy_(payload[emb.vocab_start : emb.vocab_end])
        emb.weight_scale.copy_(scales[emb.vocab_start : emb.vocab_end])
    return emb


def _engram_ids(head_sizes: list[int], num_tokens: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    ids = torch.zeros(num_tokens, len(head_sizes), dtype=torch.int64)
    for j, size in enumerate(head_sizes):
        ids[:, j] = torch.randint(0, size, (num_tokens,), generator=g) + sum(head_sizes[:j])
    return ids


@pytest.fixture
def _engram_offload_lookup_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the CPU lookup op available to the offload cases when it is absent.

    Duplicated (minimal) from test_deepseek_v41_model.py: the model-integration
    tests keep their own copy and this module must not import it. The offload
    dequant path (:meth:`EngramEmbedding._local_block_offload`) always calls the
    in-repo ``xllm_ops.engram_int8_lookup_cpu`` op and fails loudly when it is
    missing (no eager fallback). A standalone pytest process never links the
    xLLM binary, so every offload case here would otherwise raise. This fixture
    substitutes the bitwise-equivalent eager reference
    (:func:`_reference_engram_lookup`) for the duration of each case and
    restores the real resolver afterwards via ``monkeypatch``.

    When the real op *is* registered (a full xLLM build) the probe succeeds and
    the fixture is a **no-op**, so those runs keep exercising the real kernel.
    """
    import xllm.python.models.deepseek_v41_engram as engram

    try:
        engram._get_engram_lookup_op()
    except RuntimeError:
        monkeypatch.setattr(engram, "_get_engram_lookup_op", lambda: _reference_engram_lookup)


def test_engram_offload_lookup_matches_hbm_bitwise(_engram_offload_lookup_op) -> None:
    """Host gather + CPU dequant equals the device index_select bit for bit."""
    head_sizes = [7, 11, 13, 5, 3]
    dim = 32
    payload, scales = _int8_engram_payload(head_sizes, dim, 2024)
    ref = _make_int8_embedding(head_sizes, dim, payload, scales)
    off = _make_int8_embedding(head_sizes, dim, payload, scales, offload=True)
    assert off.offload is True
    assert off.weight.device.type == "cpu"
    assert off.weight.dtype == torch.int8

    ids = _engram_ids(head_sizes, 13, 7)
    out_off = off.lookup(ids)
    out_ref = ref.lookup(ids)
    assert out_off.shape == (13, len(head_sizes), dim)
    assert torch.equal(out_off, out_ref)

    # The eager staging block is reused within a capacity bucket and only grows
    # on demand; its address is free to move (nothing captures it). The
    # graph-visible block's stability is covered by the dedicated tests below.
    staged = off._eager_staged_device
    assert staged is not None
    _ = off.lookup(_engram_ids(head_sizes, 40, 8))
    assert off._eager_staged_device is staged
    assert off._eager_staged_device.shape[0] >= 256  # floor, no shrink on small batches
    _ = off.lookup(_engram_ids(head_sizes, 300, 9))
    assert off._eager_staged_device.shape[0] >= 300


def test_engram_offload_rejects_table_less_rank() -> None:
    """No rank may own zero heads: tp_size 8 over 5 heads is refused.

    The old ``part_num_embeddings == 0`` branch (a rank whose
    head slice started past the last head, e.g. tp_size 8 over 5 heads -> ranks
    5/6/7) is no longer reachable -- uniform head shards are enforced at
    construction -- so its resident-zero-block graph contract is obsolete. This
    pins the replacement behaviour: the world size that would have produced a
    table-less rank now raises instead, so the offloaded graph path can never
    receive a phantom zero block.
    """
    head_sizes = [7, 11, 13, 5, 3]  # n_hash_cols = 5
    with pytest.raises(ValueError, match="n_hash_cols=5 is not divisible by tp_size=8"):
        EngramEmbedding(
            sum(head_sizes),
            32,
            head_sizes,
            8,
            5,
            storage=ENGRAM_STORAGE_INT8,
            offload=True,
        )


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_engram_offload_tp_shards_match_hbm(tp_size: int, _engram_offload_lookup_op) -> None:
    """Per-rank offload blocks (incl. foreign-row zeroing) match the HBM path.

    The ids sweep the full global table rather than one head's range, so most
    columns fall outside a given rank's head slice and the owned/zeroing guard
    is exercised on both the offload and HBM paths. ``tp_size`` must divide the
    8-column head set (uniform-shard contract).
    """
    head_sizes = [7, 11, 13, 5, 3, 9, 4, 6]  # n_hash_cols = 8
    dim = 32
    payload, scales = _int8_engram_payload(head_sizes, dim, 99)
    g = torch.Generator().manual_seed(11)
    ids = torch.randint(0, sum(head_sizes), (9, len(head_sizes)), generator=g)
    for rank in range(tp_size):
        ref = _make_int8_embedding(head_sizes, dim, payload, scales, tp_size=tp_size, tp_rank=rank)
        off = _make_int8_embedding(head_sizes, dim, payload, scales, tp_size=tp_size, tp_rank=rank, offload=True)
        assert torch.equal(off.local_block(ids), ref.local_block(ids))


def test_engram_offload_unsupported_storage_falls_back() -> None:
    """Offload is int8-only; the fp8 byte table keeps the HBM path."""
    emb = EngramEmbedding(
        12,
        32,
        [5, 7],
        1,
        0,
        storage=ENGRAM_STORAGE_FP8,
        device=torch.device("cpu"),
        offload=True,
    )
    assert emb.offload is False
    assert emb.weight.dtype == torch.uint8
    assert emb.weight.device.type == "cpu"


def test_engram_offload_fallback_to_device() -> None:
    """A failed host setup demotes the table to the HBM parameter in place."""
    head_sizes = [5, 7]
    dim = 32
    payload, scales = _int8_engram_payload(head_sizes, dim, 5)
    emb = _make_int8_embedding(head_sizes, dim, payload, scales, offload=True)
    assert emb.offload is True
    assert emb.fallback_to_device() is True
    assert emb.offload is False
    assert emb.weight.dtype == torch.int8
    # setup_device is CPU for this construction, so the demotion is a no-op move.
    assert emb.weight.device.type == "cpu"
    assert torch.equal(emb.weight, payload)
    assert torch.equal(emb.weight_scale, scales)
    assert emb.fallback_to_device() is False


def test_staging_operand_pinned_accepts_nonzero_offset_view() -> None:
    """``_staging_operand_pinned`` matches by storage, so a shifted view passes.

    The op's operands are views of the pinned ``_host_buffer`` staging blocks. A
    contiguous slice that starts partway into the block has a different
    ``data_ptr()`` than the block head, so a bare pointer compare would reject a
    legitimate staging slice; comparing the underlying storage accepts it. A
    foreign tensor (different storage) is still rejected, and the ``pinned``
    record still gates the result.
    """
    staging = torch.zeros(8, 4, dtype=torch.int64)
    shifted = staging[2:]
    assert shifted.data_ptr() != staging.data_ptr()
    assert EngramEmbedding._staging_operand_pinned(shifted, staging, pinned=True) is True
    assert EngramEmbedding._staging_operand_pinned(staging, staging, pinned=True) is True

    foreign = torch.zeros(6, 4, dtype=torch.int64)
    assert EngramEmbedding._staging_operand_pinned(foreign, staging, pinned=True) is False
    assert EngramEmbedding._staging_operand_pinned(shifted, staging, pinned=False) is False
    assert EngramEmbedding._staging_operand_pinned(shifted, None, pinned=True) is False


def test_first_host_scratch_allocation_skips_device_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first pinned scratch allocation must not pay a device drain.

    ``_ensure_host_scratch`` drains the accelerator before dropping the old
    pinned buffers on a grow, because the non-blocking H2D at the end of the
    previous offload pass may still be reading them. The first allocation has no
    old buffer (``_ids_host is None``) and hence no copy that could reference
    one, so that drain is pure startup cost and is skipped. The offload path
    allocates the scratch once per rank at load time, so skipping the drain
    removes a whole-device synchronize from startup.

    Discriminating: dropping the ``if self._ids_host is not None:`` guard makes
    the first allocation drain unconditionally, so ``sync_calls == 0`` fails
    with ``1``.
    """
    import xllm.python.models.deepseek_v41_engram as engram_mod

    head_sizes = [7, 11, 13, 5, 3]
    payload, scales = _int8_engram_payload(head_sizes, 32, 4242)
    emb = _make_int8_embedding(head_sizes, 32, payload, scales, offload=True)
    assert emb.offload is True
    assert emb._ids_host is None

    sync_calls = 0
    real_sync = engram_mod._sync_accelerator

    def _counting_sync(device: torch.device | None) -> None:
        nonlocal sync_calls
        sync_calls += 1
        real_sync(device)

    # ``_ensure_host_scratch`` calls the module-global ``_sync_accelerator``, so
    # patching the module attribute intercepts every drain it issues.
    monkeypatch.setattr(engram_mod, "_sync_accelerator", _counting_sync)

    # First allocation: no old buffer to drop, so no drain.
    emb._ensure_host_scratch(4)
    assert emb._ids_host is not None
    assert sync_calls == 0

    # Within capacity: the guard short-circuits before any drain.
    emb._ensure_host_scratch(4)
    assert sync_calls == 0

    # Growth: the old pinned buffers are about to be dropped -> exactly one drain.
    emb._ensure_host_scratch(emb._host_capacity + 1)
    assert sync_calls == 1


def test_host_pin_failure_warns_once_and_raises_with_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host that cannot pin logs the real cause once, then fails loudly.

    ``_host_buffer`` swallows the ``pin_memory`` failure and falls back to an
    unpinned buffer, so the only signal used to be a bare ``pinned=False`` far
    from the cause. It now records why and warns once per failure episode: the
    three sibling buffers of one (re)allocation log only on the first failure,
    and the host-wide ``_host_pin_warned`` latch suppresses every later failure
    until a pin actually succeeds (which re-arms the warning). The recorded
    cause is per embedding (``self._host_pin_failure``), so a sibling layer's
    allocation cannot overwrite it. The first lookup still refuses unpinned
    staging and names the recorded cause
    (``_engram_lookup_op_contract_error`` -> ``_pin_failure_hint``).

    Discriminating: dropping the ``_host_pin_warned`` gate logs all three
    sibling buffers, so ``len(warnings) == 1`` becomes ``3`` on the first round;
    never re-arming the latch on a successful pin keeps the second episode from
    warning again, so the final ``== 2`` fails. Keeping the cause on a
    module-global (rather than the instance) makes the sibling-pin assertion
    ``emb._host_pin_failure is not None`` fail.
    """
    import scripts.logger as shared_logger
    import xllm.python.models.deepseek_v41_engram as engram_mod

    # Reset the shared warning latch so an earlier case cannot mask this one.
    monkeypatch.setattr(engram_mod, "_host_pin_warned", False)

    warnings: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class _WarningRecorder:
        """Minimal stand-in for the shared logger: records ``warning`` calls."""

        def warning(self, *args: object, **kwargs: object) -> None:
            warnings.append((args, kwargs))

    # ``_host_buffer`` imports ``logger`` lazily from ``scripts.logger``, so
    # replacing the module attribute intercepts the real caller (and nothing
    # else: every importer that bound the old object keeps the real logger).
    monkeypatch.setattr(shared_logger, "logger", _WarningRecorder(), raising=False)

    pin = {"fails": True}
    real_empty = torch.empty

    def _pin_aware_empty(*args: object, **kwargs: object) -> torch.Tensor:
        if kwargs.get("pin_memory") and pin["fails"]:
            # Pin injection: mirror the accelerator-less host failure.
            raise RuntimeError("no registered accelerator")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", _pin_aware_empty)

    head_sizes = [7, 11, 13, 5, 3]
    payload, scales = _int8_engram_payload(head_sizes, 32, 4242)
    emb = _make_int8_embedding(head_sizes, 32, payload, scales, offload=True)
    assert emb.offload is True

    # Round 1: all three sibling buffers fail; only the first logs.
    emb._ensure_host_scratch(4)
    assert emb._ids_host_pinned is False
    assert emb._rows_host_pinned is False
    assert len(warnings) == 1
    assert "cannot allocate pinned CPU staging" in warnings[0][0][0]
    assert emb._host_pin_failure is not None

    # Round 2: a grow re-allocates the siblings and still fails; the latch holds.
    emb._ensure_host_scratch(emb._host_capacity + 1)
    assert len(warnings) == 1

    # A successful pin clears the stale cause (a success never warns)...
    pin["fails"] = False
    emb._ensure_host_scratch(emb._host_capacity + 1)
    assert emb._ids_host_pinned is True
    assert emb._host_pin_failure is None
    assert len(warnings) == 1

    # ...so a fresh failure warns again instead of staying silent forever.
    pin["fails"] = True
    emb._ensure_host_scratch(emb._host_capacity + 1)
    assert emb._ids_host_pinned is False
    assert len(warnings) == 2

    # The first lookup refuses unpinned staging and names the real cause.
    ids = torch.zeros(4, emb.n_hash_cols, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="no registered accelerator"):
        emb._local_block_offload(ids)

    # The recorded cause is per embedding: a sibling layer that pins cleanly
    # must not clear this layer's cause (the old module-global did).
    other = _make_int8_embedding(head_sizes, 32, payload, scales, offload=True)
    pin["fails"] = False
    other._ensure_host_scratch(4)
    assert other._ids_host_pinned is True
    assert other._host_pin_failure is None
    assert emb._host_pin_failure is not None


@pytest.mark.parametrize("with_chunk", [False, True])
def test_engram_cache_writes_use_resolved_compressed_ids(with_chunk: bool) -> None:
    state = _tiny_engram_hash_state()
    reference = _tiny_engram_hash_state()
    state._cache = torch.full((16,), 91, dtype=torch.int32)
    ids = torch.tensor([ENGRAM_IMAGE_SENTINEL_BASE_ID, -3, 7, 100])
    positions = torch.arange(4) if with_chunk else torch.zeros(4, dtype=torch.int64)
    query_start_loc = torch.tensor([0, 4]) if with_chunk else torch.arange(5)
    dead = torch.tensor([True, False, False, False])
    slots = torch.tensor([8, 9, 10, 11], dtype=torch.int32)
    actual = state(ids, positions, query_start_loc, dead, None, slots, None, 8)
    expected = reference(ids, positions, query_start_loc, dead, None, None, None, 8)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    compressed = torch.tensor([-1, state.token_map[0], state.token_map[7], state.token_map[-1]], dtype=torch.int32)
    torch.testing.assert_close(state._cache[slots.long()], compressed, atol=0, rtol=0)
    assert torch.equal(state._cache[:8], torch.full((8,), 91, dtype=torch.int32))
