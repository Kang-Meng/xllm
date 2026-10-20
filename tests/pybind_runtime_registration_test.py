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

"""Reject built extensions that cannot be loaded into a running interpreter."""

import os
import subprocess
import sys
from pathlib import Path


def test_import_into_running_python_does_not_abort(tmp_path: Path) -> None:
    artifact = Path(os.environ["XLLM_EXPORT_PATH"]).resolve()
    assert artifact.is_file(), f"built extension does not exist: {artifact}"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util, sys; import torch; "
            "spec = importlib.util.spec_from_file_location('xllm_export', sys.argv[1]); "
            "module = importlib.util.module_from_spec(spec); "
            "sys.modules['xllm_export'] = module; "
            "spec.loader.exec_module(module)",
            str(artifact),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
