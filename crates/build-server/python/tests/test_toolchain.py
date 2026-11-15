"""The toolchain description the Build Server reads once when it starts.

The Build Server runs ``uniserve-build-server toolchain --json`` and parses
the object it prints (``Report`` in ``crates/build-server/src/toolchain.rs``),
then hands the same object back to every compilation in its request; these
tests state that contract from the program's side.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

from uniserve_build_server.toolchain import Toolchain

#: The fields the Build Server's `Report` reads, each exactly once.
FIELDS = {
    "interpreter",
    "driver",
    "distributions",
    "triton",
    "cute_dsl",
    "cuda",
    "host_compiler",
}


def described() -> dict:
    """The description the program prints, as the Build Server runs it."""
    printed = subprocess.run(
        [sys.executable, "-m", "uniserve_build_server", "toolchain", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(printed.stdout)


def test_the_description_names_this_environment_and_its_compilers():
    description = described()
    assert set(description) == FIELDS
    assert description["interpreter"] == sys.executable
    assert re.fullmatch(r"[0-9a-f]{64}", description["driver"])
    # The two Python compilers are distributions this environment locks.
    assert description["triton"] == {
        "available": description["distributions"]["triton"]
    }
    assert description["cute_dsl"] == {
        "available": description["distributions"]["nvidia-cutlass-dsl"]
    }
    # A toolchain is available with what a build needs, or missing with why.
    for name in ("cuda", "host_compiler"):
        entry = description[name]
        assert list(entry) in (["available"], ["missing"]), entry
    if "available" in description["cuda"]:
        cuda = description["cuda"]["available"]
        assert re.fullmatch(r"\d+\.\d+\.\d+", cuda["version"])
        assert set(cuda) == {
            "version",
            "root",
            "nvcc",
            "include",
            "lib",
            "stubs",
        }


def test_the_description_reads_back_as_a_compilation_is_handed_it():
    description = described()
    toolchain = Toolchain.from_json(description)
    assert json.loads(toolchain.to_json()) == description


def test_the_driver_digest_is_stable_for_unchanged_files():
    assert described()["driver"] == described()["driver"]
