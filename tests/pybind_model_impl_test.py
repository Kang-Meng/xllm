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

import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

_ROOT = Path(__file__).resolve().parents[1]


class PublicModelImplTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._model = Path(self._directory.name)
        (self._model / "config.json").write_text(json.dumps({"model_type": "qwen3"}))

        package = types.ModuleType("xllm")
        package.__path__ = [str(_ROOT / "xllm")]
        bindings = types.ModuleType("xllm.pybind")
        bindings.__path__ = [str(_ROOT / "xllm" / "pybind")]
        self._export = types.ModuleType("xllm_export")
        self._export.Options = types.SimpleNamespace
        self._export.RequestParams = types.SimpleNamespace
        self._export.RequestOutput = types.SimpleNamespace
        self._export.LLMMaster = Mock()
        self._export.VLMMaster = Mock()
        self._export.get_model_backend = Mock(return_value="llm")
        self._export.configure_cpp_chat_template = Mock()
        modules = {
            "xllm": package,
            "xllm.pybind": bindings,
            "xllm_export": self._export,
            "psutil": types.ModuleType("psutil"),
        }
        self._modules = patch.dict(sys.modules, modules)
        self._modules.start()
        self.addCleanup(self._modules.stop)
        for name in tuple(sys.modules):
            if name.startswith("xllm.pybind."):
                del sys.modules[name]
        self._signals = patch("signal.signal")
        self._signals.start()
        self.addCleanup(self._signals.stop)
        utils = importlib.import_module("xllm.pybind.utils")
        ports = patch.object(utils, "get_free_port", return_value=29500)
        ports.start()
        self.addCleanup(ports.stop)
        # These cases exercise the public API surface; bringing the real
        # accelerator up would need an NPU and would initialise torch_npu
        # inside the test process.
        runtime = patch.object(utils, "_configure_npu_runtime")
        self._npu_runtime = runtime.start()
        self.addCleanup(runtime.stop)

    def _construct(self, module_name: str, class_name: str, **kwargs: Any) -> Any:
        module = importlib.import_module(f"xllm.pybind.{module_name}")
        return getattr(module, class_name)(str(self._model), **kwargs)

    def test_constructors_forward_model_implementation_to_master(self) -> None:
        for module_name, class_name, master_name in (
            ("llm", "LLM", "LLMMaster"),
            ("vlm", "VLM", "VLMMaster"),
            ("embedding", "Embedding", "LLMMaster"),
        ):
            for value, expected in (("native", "native"), ("python", "python"), ("py", "python"), ("", "native")):
                with self.subTest(api=class_name, model_impl=value):
                    self._construct(module_name, class_name, model_impl=value)
                    options = getattr(self._export, master_name).call_args.args[0]
                    self.assertEqual(options.model_impl, expected)

    def test_constructors_preserve_native_default(self) -> None:
        for module_name, class_name, master_name in (
            ("llm", "LLM", "LLMMaster"),
            ("vlm", "VLM", "VLMMaster"),
            ("embedding", "Embedding", "LLMMaster"),
        ):
            with self.subTest(api=class_name):
                self._construct(module_name, class_name)
                self.assertEqual(getattr(self._export, master_name).call_args.args[0].model_impl, "native")

    def test_invalid_implementation_does_not_create_master(self) -> None:
        for module_name, class_name in (("llm", "LLM"), ("vlm", "VLM"), ("embedding", "Embedding")):
            with self.subTest(api=class_name), self.assertRaisesRegex(ValueError, "unsupported model_impl"):
                self._construct(module_name, class_name, model_impl="unknown")
        self._export.LLMMaster.assert_not_called()
        self._export.VLMMaster.assert_not_called()

    def test_python_implementation_bootstraps_the_process_runtime(self) -> None:
        utilities = importlib.import_module("xllm.pybind.utils")
        with patch.object(utilities, "_configure_python_model_path") as model_path:
            self._construct("llm", "LLM", model_impl="python")
        model_path.assert_called_once_with()
        self._npu_runtime.assert_called_once_with(0)

    def test_native_implementation_leaves_the_process_runtime_alone(self) -> None:
        self._construct("llm", "LLM")
        self._npu_runtime.assert_not_called()

    def test_cli_selects_implementation_and_rejects_unknown_value(self) -> None:
        parser = importlib.import_module("xllm.pybind.args").ArgumentParser().parser
        self.assertEqual(parser.parse_args([]).model_impl, "native")
        self.assertEqual(parser.parse_args(["--model_impl", "python"]).model_impl, "python")
        self.assertEqual(parser.parse_args(["--model_impl", "py"]).model_impl, "py")
        with self.assertRaises(SystemExit):
            parser.parse_args(["--model_impl", "unknown"])

    def test_offline_entries_reject_multi_node_before_master(self) -> None:
        # nnodes > 1 is not constructible offline: reject it with a Python
        # exception in the entry points instead of the C++ CHECK aborting the
        # process after the Options object has been built.
        for module_name, class_name in (("llm", "LLM"), ("vlm", "VLM"), ("embedding", "Embedding")):
            with self.subTest(api=class_name), self.assertRaisesRegex(ValueError, "nnodes"):
                self._construct(module_name, class_name, nnodes=2)
        self._export.LLMMaster.assert_not_called()
        self._export.VLMMaster.assert_not_called()


class PythonModelPathTest(unittest.TestCase):
    """XLLM_PYTHON_MODEL_PATH must name the directory that holds the ``xllm``
    package for both shapes the package ships in: a wheel install (the package
    lives in site-packages) and an in-tree run (the package lives in the
    repository). Spawned workers create their own interpreter and only prepend
    this one directory, so a wrong answer breaks the Python backend in exactly
    one of the two shapes."""

    _VARIABLE = "XLLM_PYTHON_MODEL_PATH"

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._root = Path(self._directory.name)
        modules = {
            "psutil": types.ModuleType("psutil"),
            "xllm_export": types.SimpleNamespace(),
        }
        self._modules = patch.dict(sys.modules, modules)
        self._modules.start()
        self.addCleanup(self._modules.stop)
        environment = patch.dict(os.environ, {}, clear=False)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop(self._VARIABLE, None)
        self.addCleanup(os.environ.pop, self._VARIABLE, None)

    def _load_utils(self, package_root: Path) -> types.ModuleType:
        """Import ``utils.py`` from ``<package_root>/xllm/pybind/utils.py``."""
        source = _ROOT / "xllm" / "pybind" / "utils.py"
        target = package_root / "xllm" / "pybind" / "utils.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        spec = importlib.util.spec_from_file_location(f"_model_path_probe_{abs(hash(str(package_root)))}", target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_site_packages_layout_points_at_the_installation_prefix(self) -> None:
        site_packages = self._root / "lib" / "python3.12" / "site-packages"
        (site_packages / "xllm" / "python").mkdir(parents=True)
        utils = self._load_utils(site_packages)

        utils._configure_python_model_path()

        self.assertEqual(os.environ[self._VARIABLE], str(site_packages.resolve()))
        self.assertTrue((Path(os.environ[self._VARIABLE]) / "xllm" / "python").is_dir())

    def test_source_tree_layout_points_at_the_repository_root(self) -> None:
        utils = self._load_utils(self._root)

        utils._configure_python_model_path()

        self.assertEqual(os.environ[self._VARIABLE], str(self._root.resolve()))
        self.assertTrue((Path(os.environ[self._VARIABLE]) / "xllm" / "pybind" / "utils.py").is_file())

    def test_preset_value_is_not_overwritten(self) -> None:
        site_packages = self._root / "lib" / "python3.12" / "site-packages"
        (site_packages / "xllm" / "python").mkdir(parents=True)
        os.environ[self._VARIABLE] = str(site_packages)
        utils = self._load_utils(self._root / "build")

        utils._configure_python_model_path()

        self.assertEqual(os.environ[self._VARIABLE], str(site_packages))

    def test_invalid_preset_value_fails_instead_of_deferring(self) -> None:
        os.environ[self._VARIABLE] = "/opt/custom/does-not-hold-the-package"
        utils = self._load_utils(self._root / "build")

        with self.assertRaisesRegex(RuntimeError, "/opt/custom"):
            utils._configure_python_model_path()
        # The rejected preset is left untouched: this process did not publish
        # it, so it has no business cleaning it up either.
        self.assertEqual(
            os.environ[self._VARIABLE], "/opt/custom/does-not-hold-the-package"
        )

    def test_relocated_utils_fails_instead_of_guessing(self) -> None:
        # Drop utils.py two directories below the temporary root without any
        # 'xllm' package around it, so the three-level backtracking lands
        # inside the temporary root and finds no package there: it must fail
        # loudly instead of publishing the directory to spawned workers.
        source = _ROOT / "xllm" / "pybind" / "utils.py"
        target = self._root / "any" / "nested" / "utils.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        spec = importlib.util.spec_from_file_location("_model_path_probe_stray", target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with self.assertRaisesRegex(RuntimeError, "XLLM_PYTHON_MODEL_PATH"):
            module._configure_python_model_path()
        self.assertNotIn(self._VARIABLE, os.environ)


if __name__ == "__main__":
    unittest.main()
