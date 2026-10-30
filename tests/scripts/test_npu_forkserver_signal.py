# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_XLLM_ENTRYPOINT = _REPOSITORY_ROOT / "xllm" / "xllm.cpp"
_FORKSERVER_PROBE = textwrap.dedent(
    """
    import multiprocessing
    import multiprocessing.forkserver
    import os
    import sys


    def child_main() -> None:
        os._exit(0)


    if __name__ == "__main__":
        start_early = sys.argv[1] == "early"
        if start_early:
            multiprocessing.forkserver.ensure_running()
        multiprocessing.forkserver.set_forkserver_preload(
            ["invalid_signal_state"]
        )

        context = multiprocessing.get_context("forkserver")
        process = context.Process(target=child_main)
        process.start()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()
            raise SystemExit(91)
        multiprocessing.forkserver._forkserver._stop()
        raise SystemExit(process.exitcode or 0)
    """
)
_INVALID_SIGNAL_STATE = textwrap.dedent(
    """
    import signal


    _original_signal = signal.signal


    def _signal_with_unrestorable_native_handler(signum, handler):
        previous_handler = _original_signal(signum, handler)
        if signum == signal.SIGINT and handler == signal.SIG_IGN:
            return None
        return previous_handler


    signal.signal = _signal_with_unrestorable_native_handler
    """
)
_NATIVE_SIGNAL_LIBRARY = textwrap.dedent(
    """
    #include <signal.h>

    static void native_signal_handler(int signal_number) {
      (void)signal_number;
    }

    __attribute__((constructor)) static void install_signal_handler(void) {
      struct sigaction action = {0};
      action.sa_handler = native_signal_handler;
      sigemptyset(&action.sa_mask);
      sigaction(SIGINT, &action, 0);
    }
    """
)
_LD_PRELOAD_PROBE = textwrap.dedent(
    """
    import multiprocessing
    import multiprocessing.forkserver
    import os
    import sys


    def child_main():
        os._exit(0)


    if __name__ == "__main__":
        if sys.argv[1] == "sanitize":
            unsafe_libraries = {"libasio.so", "libcust_opapi.so"}
            preload_entries = os.environ.get("LD_PRELOAD", "").split(":")
            safe_entries = [
                entry
                for entry in preload_entries
                if entry
                and os.path.basename(entry) not in unsafe_libraries
            ]
            if safe_entries:
                os.environ["LD_PRELOAD"] = ":".join(safe_entries)
            else:
                os.environ.pop("LD_PRELOAD", None)

        multiprocessing.forkserver.ensure_running()
        context = multiprocessing.get_context("forkserver")
        process = context.Process(target=child_main)
        process.start()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()
            raise SystemExit(91)
        multiprocessing.forkserver._forkserver._stop()
        raise SystemExit(process.exitcode or 0)
    """
)


class NpuForkserverSignalTest(unittest.TestCase):
    def _run_probe(self, mode: str) -> subprocess.CompletedProcess[str]:
        if "forkserver" not in multiprocessing.get_all_start_methods():
            self.skipTest("Python forkserver is not available on this platform")
        with tempfile.TemporaryDirectory() as temporary_directory:
            probe_path = Path(temporary_directory) / "forkserver_probe.py"
            probe_path.write_text(_FORKSERVER_PROBE, encoding="utf-8")
            invalid_state_path = Path(temporary_directory) / "invalid_signal_state.py"
            invalid_state_path.write_text(
                _INVALID_SIGNAL_STATE,
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(
                    None,
                    (
                        temporary_directory,
                        environment.get("PYTHONPATH", ""),
                    ),
                )
            )
            return subprocess.run(
                [sys.executable, str(probe_path), mode],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )

    def _run_ld_preload_probe(self, mode: str) -> subprocess.CompletedProcess[str]:
        if not sys.platform.startswith("linux"):
            self.skipTest("LD_PRELOAD regression requires Linux")
        if "forkserver" not in multiprocessing.get_all_start_methods():
            self.skipTest("Python forkserver is not available on this platform")
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler is required for LD_PRELOAD regression")

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "native_signal_handler.c"
            library_path = temporary_path / "libasio.so"
            probe_path = temporary_path / "forkserver_preload_probe.py"
            source_path.write_text(_NATIVE_SIGNAL_LIBRARY, encoding="utf-8")
            probe_path.write_text(_LD_PRELOAD_PROBE, encoding="utf-8")
            subprocess.run(
                [
                    compiler,
                    "-shared",
                    "-fPIC",
                    "-o",
                    str(library_path),
                    str(source_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            environment = os.environ.copy()
            environment["LD_PRELOAD"] = str(library_path)
            return subprocess.run(
                [sys.executable, str(probe_path), mode],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )

    def test_late_forkserver_reproduces_native_signal_failure(self) -> None:
        result = self._run_probe("late")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "signal handler must be signal.SIG_IGN, signal.SIG_DFL, or a callable object",
            result.stderr,
        )

    def test_early_forkserver_survives_later_native_signal_handler(self) -> None:
        result = self._run_probe("early")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("TypeError", result.stderr)

    def test_preloaded_native_signal_handler_reproduces_failure(self) -> None:
        result = self._run_ld_preload_probe("unsanitized")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "signal handler must be signal.SIG_IGN, signal.SIG_DFL, or a callable object",
            result.stderr,
        )

    def test_sanitized_preload_allows_forkserver_child(self) -> None:
        result = self._run_ld_preload_probe("sanitize")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("TypeError", result.stderr)

    def test_npu_runtime_bootstraps_interpreter_before_native_runtime(self) -> None:
        # The serving entry calls initialize_python_runtime() before any model
        # work. That helper owns the embedded interpreter and must create it,
        # then aclInit, then import torch. A forkserver preload is not part of
        # this path.
        entry = _XLLM_ENTRYPOINT.read_text(encoding="utf-8")
        runtime = (_REPOSITORY_ROOT / "xllm" / "core" / "platform" / "npu" / "npu_python_runtime.cpp").read_text(
            encoding="utf-8"
        )

        self.assertIn("init_npu_python_runtime();", entry)
        self.assertLess(
            entry.index("init_npu_python_runtime();"),
            entry.index("return run();"),
        )
        self.assertIn("py::initialize_interpreter", runtime)
        self.assertLess(
            runtime.index("py::initialize_interpreter"),
            runtime.index("aclInit(nullptr)"),
        )
        self.assertLess(
            runtime.index("aclInit(nullptr)"),
            runtime.index("py::exec(kInitializeTorchNpuScript, scope)"),
        )
        self.assertIn("import torch\n", runtime)


if __name__ == "__main__":
    unittest.main()
