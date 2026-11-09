"""The host's Program Server of a test module, with every program package.

A deployment's epoch is resolved at this host's Program Server before its main
processes serve, and a program's module exists nowhere else, so every test that
serves needs one. :func:`program_server` starts one for a test module over an
empty store, with every kernel package of the tree and the scheduling policy
installed. The engine's recorded walk then asks the server to build exactly the
kernels the deployment launches at startup.

The kernels build with the Program Server's own Python environment
(``crates/program-server/python``), which holds the compilers and the packages
kernel programs import (Triton, CuTe DSL, FlashInfer). The repository's
development environment holds the development tools and the evaluator, and none
of those, so a build run with it stops at its first import.

``UNISERVE_PROGRAM_SERVER_PYTHON`` names that environment's interpreter,
``UNISERVE_SCHEDULING_PRECOMPILE`` the wasm precompiler, and ``UNISERVE_BINARY``
the ``uniserve`` binary.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.python.e2e.http_helpers import (
    require_uniserve_binary,
    start_server,
    stop_server,
)

# The packages this file installs: the scheduling policy, and every kernel
# package of the tree. A deployment's epoch is resolved at this host's Program
# Server before its main processes serve (design 5.1, 6.1): the policy fills the
# scheduling slot, and the engine's recorded walk names every kernel the
# deployment launches, which the server builds from the installed packages on
# the first prepare. A server that holds the policy alone refuses the epoch by
# the first kernel it lacks.
#
# A program's backend is a package's name: the Program Server selects a program
# by the backend its manifest declares, and no package is named ``auto``.
KERNEL_PACKAGES = "crates/ops/kernels"
POLICY_PACKAGE = "crates/ops/program-fixtures/policy"


def repository() -> Path:
    """The repository root: the workspace the policy package is built in."""
    return Path(__file__).resolve().parents[3]


def compiler() -> Path:
    """The interpreter of the Program Server's own environment.

    It compiles every kernel program, and nothing else provides what the
    programs import. ``UNISERVE_PROGRAM_SERVER_PYTHON`` names it.
    """
    named = os.environ.get("UNISERVE_PROGRAM_SERVER_PYTHON")
    if not named:
        pytest.fail(
            "UNISERVE_PROGRAM_SERVER_PYTHON must name the Program Server's "
            "interpreter: `uv sync --project crates/program-server/python` "
            "makes one"
        )
    return Path(named)


def precompiler() -> Path:
    """The wasm precompiler the Program Server turns a policy module into.

    A policy is built for wasm32 and precompiled for this host's CPU, so the
    server needs the precompiler before it can install a policy (design 5.3).
    ``UNISERVE_SCHEDULING_PRECOMPILE`` names it.
    """
    named = os.environ.get("UNISERVE_SCHEDULING_PRECOMPILE")
    if not named:
        pytest.fail(
            "UNISERVE_SCHEDULING_PRECOMPILE must name the precompiler: "
            "`cargo build -p scheduling --bin scheduling-precompile` builds it"
        )
    return Path(named)


def program_target() -> str:
    """The target this host's modules are built for.

    The engine derives it from the device it serves — ``sm_{major}{minor}a``
    (``src/model_runner.rs``) — and a store built for another target describes
    nothing to overlay: the precopy then fails at the module lane rather than
    the build. So the default is the served device's own compute capability,
    read the way this file already reads that device; the environment's
    ``UNISERVE_PROGRAM_TARGET`` names a target instead, for a host whose store
    is provisioned elsewhere.
    """
    named = os.environ.get("UNISERVE_PROGRAM_TARGET")
    if named:
        return named
    device = os.environ.get("UNISERVE_LIVE_DEVICE", "0")
    reported = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device}",
            "--query-gpu=compute_cap",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    major, _, minor = reported.partition(".")
    if not major.isdigit() or not minor.isdigit():
        raise AssertionError(
            f"nvidia-smi reported no compute capability for GPU {device}: "
            f"{reported!r}"
        )
    return f"sm_{major}{minor}a"


@contextmanager
def program_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Path]:
    """The host's Program Server, with every program package installed.

    The daemon's leader resolves each epoch's programs at this server, and a
    program's module exists nowhere else, so every deployment of a test module
    needs it. The server starts over an empty store and is handed two sets of
    packages:

    * the kernels — every package under ``KERNEL_PACKAGES``. Installing copies
      a package's source and registers its revision; the engine's recorded walk
      names every descriptor key the deployment launches, and the server builds
      each on its first prepare, so a deployment's startup builds exactly what
      it launches;
    * the policy — ``POLICY_PACKAGE``, a wasm package this host precompiles.

    The package root and the store live as long as the module, and every test
    of the module deploys against this one server, so a module is built once and
    every later deployment of the module reuses it (design 5.3). They are not
    shared with another run: a store entry is keyed by operator, backend,
    revision, target and descriptor, and a revision counts from 1 in a fresh
    package root, so a store kept across runs would answer a changed package
    with the module the older source built.
    """
    binary = require_uniserve_binary()
    work = tmp_path_factory.mktemp("program-server")
    socket = work / "program-server.sock"
    log = work / "program-server.log"
    packages = work / "packages"
    store = work / "modules"
    command = [
        str(binary),
        "program-server",
        "serve",
        "--socket",
        str(socket),
        "--package-root",
        str(packages),
        "--store-root",
        str(store),
        "--target",
        program_target(),
        "--wasm",
        str(precompiler()),
    ]
    command += ["--python", str(compiler())]
    process = start_server(command, log)
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"the Program Server exited: {log.read_text()}")
        if socket.exists():
            break
        time.sleep(0.2)
    else:
        raise TimeoutError(f"the Program Server left no socket at {socket}")

    try:
        manifests = (repository() / KERNEL_PACKAGES).glob("*/*/package.toml")
        kernels = sorted(
            str(manifest.parent.relative_to(repository()))
            for manifest in manifests
        )
        for package in (*kernels, POLICY_PACKAGE):
            installed = subprocess.run(
                [
                    str(binary),
                    "program-server",
                    "install",
                    "--socket",
                    str(socket),
                    str(repository() / package),
                ],
                capture_output=True,
                text=True,
                # A kernel package builds its modules on the first prepare, not
                # here: installing copies the source and registers its revision.
                timeout=3600,
            )
            if installed.returncode != 0:
                raise AssertionError(
                    f"installing {package} failed: "
                    f"{installed.stdout}\n{installed.stderr}"
                )
        yield socket
    finally:
        stop_server(process)
