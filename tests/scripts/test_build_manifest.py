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

import argparse
import re
import subprocess
from pathlib import Path


def read_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in entries:
            raise AssertionError(f"invalid build manifest entry: {line!r}")
        entries[key] = value
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--use-npu", required=True)
    parser.add_argument("--build-type", required=True)
    arguments = parser.parse_args()

    entries = read_manifest(arguments.manifest)
    assert entries == {
        "source_dir": str(arguments.source_dir),
        "source_commit": entries["source_commit"],
        "source_dirty": entries["source_dirty"],
        "use_npu": arguments.use_npu,
        "build_type": arguments.build_type,
        "binary_relative_path": "xllm/core/server/xllm",
    }
    assert re.fullmatch(r"[0-9a-f]{40}", entries["source_commit"])
    assert entries["source_dirty"] in {"true", "false"}

    git_command = [
        "git",
        "-c",
        f"safe.directory={arguments.source_dir}",
        "-C",
        str(arguments.source_dir),
    ]
    current_commit = subprocess.run(
        [*git_command, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    current_status = subprocess.run(
        [
            *git_command,
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert entries["source_commit"] == current_commit
    assert entries["source_dirty"] == ("true" if current_status else "false")


if __name__ == "__main__":
    main()
