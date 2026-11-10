"""The kernels' argument lists against the operators that launch them.

``tools/kernel_arguments.py`` compares the three statements of a kernel's
arguments: the kernel's own parameters, the package's ``spec`` and the
operator's Rust ``Arguments`` chain. It is sound only against a built store,
whose cells' ``metadata.json`` are what a launch reads; its source mode
reports an operator's arms against each other. So this test runs only when a
store and the key fixture it was built from are named, and runs both of the
tool's passes over them: the cells (``--check``) and the native call
structures (``--check --native``).
"""

import os
import pathlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit

REPOSITORY = pathlib.Path(__file__).resolve().parents[3]

#: The key fixture the store was built from: `op<TAB>descriptor[<TAB>backend]`.
KEYS = "UNISERVE_KERNEL_KEYS"

#: The built store whose cells the comparison reads.
STORE = "UNISERVE_KERNEL_STORE"

#: The architecture the store was built for, as the GPU tests name it.
TARGET = "UNISERVE_PROGRAM_TARGET"


@pytest.mark.parametrize("native", [False, True], ids=["cells", "native"])
def test_every_built_cell_takes_the_arguments_its_operator_packs(native):
    keys, store = os.environ.get(KEYS), os.environ.get(STORE)
    if not keys or not store:
        pytest.skip(f"{KEYS} and {STORE} name the built store this compares")
    if not any(pathlib.Path(store).rglob("metadata.json")):
        pytest.fail(f"{STORE}={store} holds no built cell", pytrace=False)

    arguments = ["--check", "--fixture", keys, "--store", store]
    arguments += ["--target", os.environ.get(TARGET, "sm_100a")]
    if native:
        arguments.append("--native")
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY / "tools" / "kernel_arguments.py")]
        + arguments,
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    output = (completed.stdout + completed.stderr).strip()
    assert completed.returncode == 0, output
    # A comparison that read no cell compared nothing, which is no agreement.
    assert not output.startswith("0 cell(s) compared"), output
