"""The compile request: how the Build Server compiles one descriptor.

The Build Server runs ``python -m uniserve_build_server compile`` in the
build module's directory with the request as JSON on standard input
(``Request`` in ``crates/build-server/src/source.rs``). These tests run the
program exactly so, with a CUDA-kind package of their own (``recording``)
whose build module records what it was handed, so the protocol is checked
without a compiler or a device.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest
from uniserve_build_server.request import Target
from uniserve_build_server.toolchain import available, missing

#: The recording package: its build module writes what it was handed.
RECORDING = pathlib.Path(__file__).parent / "recording"


def request(package: pathlib.Path, output: pathlib.Path, **changes) -> dict:
    """A request as the Build Server writes one for the recording module."""
    target = Target.named("sm_100a")
    written = {
        "op": "recording",
        "backend": "cuda",
        "descriptor": "dtype_bf16_columns_128",
        "fields": {"dtype": "bf16", "columns": "128"},
        "parameters": {"tile": 64, "selected": {"mActType": "silu"}},
        "source": "cuda",
        "module": "build",
        "unit": None,
        "package": str(package),
        "output": str(output),
        "target": {
            "name": target.name,
            "backend": target.backend,
            "capability": target.capability,
            "specific": target.specific,
            "families": list(target.families),
            "warp_size": target.warp_size,
            "shared_memory": target.shared_memory,
            "tensor_memory": target.tensor_memory,
            "dependent_launch": target.dependent_launch,
            "fp8": target.fp8,
        },
        "jobs": 3,
        "cache": str(output / "cache"),
        "toolchain": {
            "interpreter": sys.executable,
            "driver": "ab" * 32,
            "distributions": {},
            "triton": available("3.7.1"),
            "cute_dsl": available("4.6.2"),
            "cuda": available(
                {
                    "version": "13.0.48",
                    "root": "/cuda",
                    "nvcc": "/cuda/bin/nvcc",
                    "include": "/cuda/include",
                    "lib": "/cuda/lib",
                    "stubs": "/cuda/lib/stubs",
                }
            ),
            "host_compiler": missing("no c++ on this host's search path"),
        },
    }
    written.update(changes)
    return written


def compile_in(module_directory: pathlib.Path, stated: dict):
    """Run the compilation as the Build Server runs it."""
    return subprocess.run(
        [sys.executable, "-m", "uniserve_build_server", "compile"],
        cwd=module_directory,
        input=json.dumps(stated),
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_build_module_is_handed_the_typed_request(tmp_path):
    output = tmp_path / "output"
    compiled = compile_in(RECORDING / "src", request(RECORDING, output))
    assert compiled.returncode == 0, compiled.stderr

    handed = json.loads((output / "handed.json").read_text())
    assert handed == {
        "descriptor": "dtype_bf16_columns_128",
        "fields": {"dtype": "bf16", "columns": "128"},
        "parameters": {"tile": 64, "selected": {"mActType": "silu"}},
        "target": "sm_100a",
        "capability": 100,
        "families": ["sm_100a", "sm_100f"],
        "jobs": 3,
        "nvcc": "/cuda/bin/nvcc",
    }


def test_a_missing_toolchain_refuses_the_build_with_its_reason(tmp_path):
    stated = request(RECORDING, tmp_path / "output")
    stated["toolchain"]["cuda"] = missing("no CUDA 13.0 toolkit")
    compiled = compile_in(RECORDING / "src", stated)
    assert compiled.returncode != 0
    assert "no CUDA 13.0 toolkit" in compiled.stderr


def test_a_source_without_a_build_module_is_refused(tmp_path):
    compiled = compile_in(
        RECORDING / "src",
        request(RECORDING, tmp_path / "output", source="cubin"),
    )
    assert compiled.returncode != 0
    assert "cubin" in compiled.stderr


def test_a_target_the_description_does_not_state_is_refused():
    with pytest.raises(SystemExit, match="no target 'sm_75'"):
        Target.named("sm_75")
