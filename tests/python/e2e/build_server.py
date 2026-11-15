"""The host's Build Server of a test module, with every package.

A deployment's epoch is resolved at this host's Build Server before its main
processes serve, and a package's module exists nowhere else, so every test that
serves needs one. :func:`build_server` starts one for a test module over an
empty store, with every kernel package of the tree and the scheduling policy
installed. The engine's profile runs then ask the server to build exactly the
kernels the deployment launches at startup.

The kernels build with the Build Server's own Python environment
(``crates/build-server/python``), which holds the compilers and the packages
kernel packages import (Triton, CuTe DSL, FlashInfer). The repository's
development environment holds the development tools and the evaluator, and none
of those, so a build run with it stops at its first import.

``UNISERVE_BUILD_SERVER_PYTHON`` names that environment's interpreter (the
tree's ``crates/build-server/python/.venv`` when unset),
``UNISERVE_SCHEDULING_PRECOMPILE`` the wasm precompiler (a ``uniserve-engine``
executable), and ``UNISERVE_BINARY``
the ``uniserve`` binary.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.python.e2e.http_helpers import (
    granted_devices,
    require_uniserve_binary,
    start_server,
    stop_server,
)

# The packages this file installs: the scheduling policy, and every kernel
# package of the tree. A deployment's epoch is resolved at this host's Build
# Server before its main processes serve (``specs/design.md`` §3): the policy
# fills the scheduling slot, and the engine's profile runs name every kernel the
# deployment launches, which the server builds from the installed packages on
# the first prepare. A server that holds the policy alone refuses the epoch by
# the first kernel it lacks.
#
# A package's backend is a package's name: the Build Server selects a package
# by the backend its manifest declares, and no package is named ``auto``.
KERNEL_PACKAGES = "crates/ops/kernels"
POLICY_PACKAGE = "crates/scheduling/policies/fixture"

# The engine package: the executable every deployment's main processes run
# as. The daemon fixes its revision when a deployment starts, and the server
# builds it with cargo from the installed snapshot the first time it is asked
# for; cargo's own target directory keeps what it compiled between servers.
ENGINE_PACKAGE = "crates/engine"


def repository() -> Path:
    """The repository root: the workspace the policy package is built in."""
    return Path(__file__).resolve().parents[3]


def compiler() -> Path:
    """The interpreter of the Build Server's own environment.

    It compiles every kernel package, and nothing else provides what the
    modules import. ``UNISERVE_BUILD_SERVER_PYTHON`` names it; unset, it
    is the environment `uv sync --locked --project crates/build-server/python`
    creates in the tree.
    """
    named = os.environ.get("UNISERVE_BUILD_SERVER_PYTHON")
    path = (
        Path(named)
        if named
        else repository() / "crates/build-server/python/.venv/bin/python"
    )
    if not path.exists():
        pytest.fail(
            f"the Build Server's interpreter {path} does not exist: "
            "`uv sync --locked --project crates/build-server/python` "
            "makes one, or UNISERVE_BUILD_SERVER_PYTHON names another"
        )
    return path


def precompiler() -> Path:
    """The wasm precompiler the Build Server turns a policy module into.

    A policy is built for wasm32 and precompiled for this host's CPU, so the
    server needs the precompiler before it can install a policy: a
    ``uniserve-engine`` executable, whose ``precompile`` command the server
    runs. ``UNISERVE_SCHEDULING_PRECOMPILE`` names it.
    """
    named = os.environ.get("UNISERVE_SCHEDULING_PRECOMPILE")
    if not named:
        pytest.fail(
            "UNISERVE_SCHEDULING_PRECOMPILE must name the precompiler, a "
            "uniserve-engine executable: `cargo build -p engine --bin "
            "uniserve-engine` builds it"
        )
    return Path(named)


def module_target() -> str:
    """The target this host's modules are built for.

    The engine derives it from the device it serves (``sm_{major}{minor}a``),
    and a store built for another target holds no module the engine can load:
    the epoch then fails when its modules are resolved rather than at the
    build. So the default is the compute capability of the first GPU the
    lease granted (:func:`granted_devices`), which the module serves on; the
    environment's ``UNISERVE_MODULE_TARGET`` names a target instead, for a
    host whose store is provisioned elsewhere.
    """
    named = os.environ.get("UNISERVE_MODULE_TARGET")
    if named:
        return named
    device = granted_devices(1)
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
def build_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Path]:
    """The host's Build Server, with every package installed.

    The daemon's leader resolves each epoch's modules at this server, and a
    package's module exists nowhere else, so every deployment of a test module
    needs it. The server starts over an empty store and is handed two sets of
    packages:

    * the kernels — every package under ``KERNEL_PACKAGES``. Installing copies
      a package's source and registers its revision; the engine's profile runs
      name every descriptor key the deployment launches, and the server builds
      each on its first prepare, so a deployment's startup builds exactly what
      it launches;
    * the policy — ``POLICY_PACKAGE``, a wasm package this host precompiles.

    The package root and the store live as long as the module, and every test
    of the module deploys against this one server, so a module is built once and
    every later deployment of the module reuses it. They are not
    shared with another run: a store entry is keyed by operator, backend,
    revision, target and descriptor, and a revision counts from 1 in a fresh
    package root, so a store kept across runs would answer a changed package
    with the module the older source built.
    """
    binary = require_uniserve_binary()
    work = tmp_path_factory.mktemp("build-server")
    # A Unix socket's path is bounded (`SUN_LEN`, 108 bytes), and the test's
    # directory may lie deep in a run's artifacts, so the socket lives in a
    # short directory of its own; the log, packages and store stay with the
    # test.
    sockets = Path(tempfile.mkdtemp(prefix="uniserve-modules-"))
    socket = sockets / "build-server.sock"
    log = work / "build-server.log"
    packages = work / "packages"
    store = work / "modules"
    command = [
        str(binary),
        "build-server",
        "serve",
        "--socket",
        str(socket),
        "--package-root",
        str(packages),
        "--store-root",
        str(store),
        "--target",
        module_target(),
        "--wasm",
        str(precompiler()),
    ]
    command += ["--python", str(compiler())]
    process = start_server(command, log)
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"the Build Server exited: {log.read_text()}")
        if socket.exists():
            break
        time.sleep(0.2)
    else:
        raise TimeoutError(f"the Build Server left no socket at {socket}")

    try:
        for package in packages_in_order():
            installed = subprocess.run(
                [
                    str(binary),
                    "build-server",
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
        shutil.rmtree(sockets, ignore_errors=True)


def packages_in_order() -> list[str]:
    """Every package a test's Build Server holds, in the order it installs
    them: a revision counts from 1 in a fresh package root, so two servers
    that install the same packages in this order number them alike.
    """  # noqa: D205
    manifests = (repository() / KERNEL_PACKAGES).glob("*/*/package.toml")
    kernels = sorted(
        str(manifest.parent.relative_to(repository())) for manifest in manifests
    )
    packages = [*kernels, POLICY_PACKAGE, ENGINE_PACKAGE]
    # An engine package a run selects with `UNISERVE_ENGINE`, such as the
    # sanitizer's wrapper, is installed from the directory this names.
    selected = os.environ.get("UNISERVE_ENGINE_PACKAGE")
    if selected:
        packages.append(selected)
    return packages


@contextmanager
def remote_build_server(host: str) -> Iterator[str]:
    """A Build Server on the ssh destination `host`, as
    :func:`build_server` starts one on this host, and its socket there.

    A main process resolves its modules from its own host's Build Server,
    which the daemon of a deployment's second host names (its
    ``build_config.socket``). The leader pins the epoch's revisions at this
    host's server; this one installs the same packages in the same order, so
    it numbers them alike and builds each module the second host's main
    process asks for. The checkout lies at the same path on `host`; the
    server's sockets, packages and store lie in a directory of its own there,
    removed with the server.
    """  # noqa: D205
    binary = require_uniserve_binary()
    work = f"/tmp/uniserve-modules-{os.getpid()}-{time.monotonic_ns()}"
    socket = f"{work}/build-server.sock"
    serve = [
        str(binary),
        "build-server",
        "serve",
        "--socket",
        socket,
        "--package-root",
        f"{work}/packages",
        "--store-root",
        f"{work}/modules",
        "--target",
        module_target(),
        "--wasm",
        str(precompiler()),
        "--python",
        str(compiler()),
    ]
    # The server runs in this test's environment: its search paths, native
    # libraries and CUDA toolkit, the FFmpeg prefix and libclang (with its
    # bindgen arguments) its build of the engine package reads
    # (tools/gpu_e2e.sh), and its compilations' parallelism.
    variables = " ".join(
        f"{name}={shlex.quote(os.environ.get(name, default))}"
        for name, default in (
            ("PATH", ""),
            ("LD_LIBRARY_PATH", ""),
            ("CUDA_HOME", ""),
            ("UNISERVE_FFMPEG_DIR", ""),
            ("LIBCLANG_PATH", ""),
            ("BINDGEN_EXTRA_CLANG_ARGS", ""),
            ("MAX_JOBS", "16"),
        )
    )
    remote = (
        f"mkdir -m 700 {work} && cd {shlex.quote(str(repository()))} "
        "|| exit 1; "
        f"{variables} nohup {shlex.join(serve)} > {work}/build-server.log "
        f"2>&1 < /dev/null & echo $!"
    )

    def on_host(command: str, timeout: float = 120) -> str:
        return subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host, command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        ).stdout

    pid = on_host(remote).strip()
    try:
        deadline = time.monotonic() + 120.0
        while on_host(f"test -S {socket} && echo up || true").strip() != "up":
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"the Build Server on {host} left no socket at {socket}: "
                    + on_host(f"cat {work}/build-server.log")
                )
            time.sleep(0.5)
        for package in packages_in_order():
            on_host(
                f"{variables} "
                + shlex.join(
                    [
                        str(binary),
                        "build-server",
                        "install",
                        "--socket",
                        socket,
                        str(repository() / package),
                    ]
                ),
                timeout=3600,
            )
        yield socket
    finally:
        on_host(f"kill {pid}; sleep 1; rm -rf {work}")
