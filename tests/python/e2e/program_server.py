"""The host's Program Server of a test module, with a deployment's programs.

A deployment's epoch is resolved at this host's Program Server before its main
processes serve, and a program's module exists nowhere else, so every test that
serves needs one. :func:`program_server` starts one for a test module, built
from the kernels the bf16 dense Qwen3 deployment launches.

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

# The packages this file installs: the scheduling policy, and one kernel
# package for every operator the deployment selects. A deployment's epoch is
# resolved at this host's Program Server before its main processes serve
# (design 5.1, 6.1): the policy fills the scheduling slot, and the operators
# the build config names fill theirs, so a server that has the policy alone
# refuses the epoch by the first kernel it lacks.
#
# ``FilledPrograms::of`` (``crates/daemon/src/epoch.rs``) fills paged attention
# with the attention program's backend; ``moe_runner_backend`` is unset here, so
# no grouped expert GEMM is filled, and the expert exchange is filled only for
# an expert-parallel deployment — the model launches no exchange when every
# expert is local, so this one-rank deployment names no such program.
#
# A program's backend is a package's name: the Program Server selects a program
# by the backend its manifest declares, and no package is named ``auto``. The
# deployment therefore states the attention backend it means (``OPTIONS``), and
# it is the one design 9's default choice resolves to for this shape — BF16, no
# window, head_dim 128, 64-token pages, one rank: trtllm-gen.
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


def deployment_keys(store: Path, checkpoint: Path) -> Path:
    """The key lines this file's deployment serves, generated beside `store`.

    The model's own rows come from its walk over the checkpoint's metadata
    (``qwen3-moe --example keys``, which opens no shard and no device), and the
    worker's rows (the sampler's chain and the logits gathers) are merged in by
    ``ops --example qwen3_moe_keys``. So the list is the code's own statement of
    this bf16 deployment, and no stored file can drift from it.
    """
    model = subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "--package",
            "qwen3-moe",
            "--example",
            "keys",
            "--",
            "--checkpoint",
            str(checkpoint),
        ],
        cwd=repository(),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if model.returncode != 0:
        raise AssertionError(
            f"the model's key walk failed: {model.stderr[-4000:]}"
        )
    walk = store.parent / "model-keys.txt"
    walk.write_text(model.stdout)
    merged = subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "--package",
            "ops",
            "--example",
            "qwen3_moe_keys",
            "--",
            "--model-keys",
            str(walk),
        ],
        cwd=repository(),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if merged.returncode != 0:
        raise AssertionError(
            f"merging the worker's keys failed: {merged.stderr[-4000:]}"
        )
    rows = [
        line
        for line in merged.stdout.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not rows:
        raise AssertionError(
            "the key generators stated no row for this deployment"
        )
    path = store.parent / "deployment-keys.txt"
    path.write_text("\n".join(rows) + "\n")
    return path


def build_store(
    store: Path, target: str, python: Path, checkpoint: Path
) -> None:
    """Builds a module for every kernel key this deployment launches.

    `store` is ``build_fixtures``'s store, which it fills in its own shape: the
    installed package sources under ``<store>/packages`` and the prepared
    modules under ``<store>/modules``. The directory a Program Server serves is
    the second of those — a store root holds the modules, not the store's
    parent.

    ``build_fixtures`` (``crates/program-server/examples``) is the store's own
    producer: for each ``operator<TAB>key`` line of the key file it installs the
    operator's package, prepares that key for `target`, and resolves it to the
    module it names. It is given :func:`deployment_keys`: this deployment's
    rows only, so a package another deployment owes fails that deployment's run
    and not this one. The builds are real — Triton, nvcc and
    CuTe DSL all run — so this takes minutes rather than seconds, once per
    module. The Program Server then
    serves the store as it stands, which is why ``--store-root`` below is this
    directory and no kernel package is installed a second time.

    Every row it is given is one this deployment launches, so a row that does
    not build is this deployment's failure and the file stops here: a short
    store would only move the same news to the precopy a minute later, and the
    build's own output names the key and the reason.
    """
    command = [
        "cargo",
        "run",
        "--package",
        "program-server",
        "--example",
        "build_fixtures",
        "--",
        "--keys",
        str(deployment_keys(store, checkpoint)),
        "--packages",
        str(repository() / "crates/ops/kernels"),
        "--target",
        target,
        "--store",
        str(store),
        "--python",
        str(python),
    ]
    built = subprocess.run(
        command,
        cwd=repository(),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if built.returncode != 0:
        missing = [
            line.removeprefix("FAILED\t").replace("\t", " ")
            for line in built.stdout.splitlines()
            if line.startswith("FAILED\t")
        ]
        raise AssertionError(
            "the deployment's own kernels did not build, so nothing can serve"
            + (f": {', '.join(missing)}" if missing else "")
            + f"\n{built.stdout[-4000:]}\n{built.stderr[-4000:]}"
        )


@contextmanager
def program_server(
    tmp_path_factory: pytest.TempPathFactory,
    checkpoint: Path | None = None,
) -> Iterator[Path]:
    """The host's Program Server, with the deployment's programs installed.

    Autouse and module-scoped because every deployment of this file needs it:
    the daemon's leader resolves each epoch's programs at this server, and a
    program's module exists nowhere else. Two sets of programs are here, built
    once for the module:

    * the kernels — a module for every descriptor key the deployment launches,
      built into the store by ``build_fixtures`` from :func:`deployment_keys`
      over `checkpoint`, which defaults to the one ``UNISERVE_QWEN3_MODEL``
      names. The epoch is
      refused by the first slot no installed program fills, and a main process
      serves nothing until it can install the kernels the model launches;
    * the policy — ``POLICY_PACKAGE``, installed below, because a policy is a
      wasm package this host precompiles rather than a kernel key.

    The package root and the store live as long as the module, and every test
    of the module deploys against this one server, so a module is built once and
    every later deployment of the module reuses it (design 5.3). They are not
    shared with another run: a store entry is keyed by operator, backend,
    revision, target and descriptor, and a revision counts from 1 in a fresh
    package root, so a store kept across runs would answer a changed package
    with the module the older source built. The environment's
    ``UNISERVE_PROGRAM_STORE`` names a store ``build_fixtures`` built from this
    tree instead, for a run of several modules: a build takes minutes.
    """
    binary = require_uniserve_binary()
    work = tmp_path_factory.mktemp("program-server")
    socket = work / "program-server.sock"
    log = work / "program-server.log"
    target = program_target()
    interpreter = compiler()
    named = os.environ.get("UNISERVE_PROGRAM_STORE")
    built = Path(named) if named else work / "programs"
    if not named:
        if checkpoint is None:
            stated = os.environ.get("UNISERVE_QWEN3_MODEL")
            if not stated:
                pytest.fail(
                    "UNISERVE_QWEN3_MODEL must name the model checkpoint"
                )
            checkpoint = Path(stated)
        build_store(built, target, interpreter, checkpoint)
    # The builder's two directories are the two the server is given: installed
    # sources for a rebuild, prepared modules for what it serves.
    packages = built / "packages"
    store = built / "modules"
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
        target,
        "--wasm",
        str(precompiler()),
    ]
    command += ["--python", str(interpreter)]
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

    for package in (POLICY_PACKAGE,):
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
    try:
        yield socket
    finally:
        stop_server(process)


