# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0. See the repository LICENSE.
"""Compile real GTest macros and compare the executed CTest inventory."""

import argparse
import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--ctest", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--gtest-dir", required=True)
    args = parser.parse_args()
    expected = {
        "DiscoveryContractTest.MultilineDeclarationRemainsVisible",
        "DiscoveryContractTest.SingleLineDeclarationRemainsVisible",
        "DiscoveryContractTest.RequiresExternalServer",
        "Values/DiscoveryParameterTest.MultilineParameterRemainsVisible/0",
        "Values/DiscoveryParameterTest.MultilineParameterRemainsVisible/1",
        "DiscoveryTypedTest.MultilineTypeRemainsVisible<0>",
        "DiscoveryTypedTest.MultilineTypeRemainsVisible<1>",
    }
    for with_environment in (False, True):
        with tempfile.TemporaryDirectory(prefix="xllm-discovery-") as temporary:
            build = Path(temporary)
            subprocess.run(
                [
                    args.cmake,
                    "-S",
                    str(args.source_dir / "tests/cmake/discovery"),
                    "-B",
                    str(build),
                    "-DCMAKE_CXX_COMPILER=" + args.compiler,
                    "-DFIXTURE_WITH_ENVIRONMENT=" + ("ON" if with_environment else "OFF"),
                    "-DGTest_DIR=" + args.gtest_dir,
                    "-DXLLM_CC_TEST=" + str(args.source_dir / "cmake/cc_test.cmake"),
                ],
                check=True,
            )
            subprocess.run([args.cmake, "--build", str(build)], check=True)
            command = [args.ctest, "--test-dir", str(build)]
            plan = json.loads(subprocess.check_output([*command, "--show-only=json-v1"], text=True))
            assert {test["name"] for test in plan["tests"]} == expected, plan
            for test in plan["tests"]:
                properties = {item["name"]: item["value"] for item in test["properties"]}
                assert properties.get("ENVIRONMENT", []) == (
                    ["XLLM_DISCOVERY_CONTRACT=present"] if with_environment else []
                )
                assert "--gtest_shuffle" in test["command"]
                assert "--gtest_random_seed=7391" in test["command"]
                assert bool(properties.get("DISABLED")) == test["name"].endswith(".RequiresExternalServer")
            result = build / "result.xml"
            subprocess.run(
                [*command, "--output-on-failure", "--no-tests=error", "--output-junit", str(result)], check=True
            )
            rows = list(ET.parse(result).getroot().iter("testcase"))
            assert len(rows) == len(expected)
            assert {row.attrib["name"] for row in rows} == expected
            disabled = [row.attrib["name"] for row in rows if row.attrib.get("status") == "disabled"]
            assert disabled == ["DiscoveryContractTest.RequiresExternalServer"]
            assert not any(row.find("failure") is not None or row.find("error") is not None for row in rows)


if __name__ == "__main__":
    main()
