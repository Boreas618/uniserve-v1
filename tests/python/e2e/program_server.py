"""The host's Program Server of a test module, with a deployment's programs.

A deployment's epoch is resolved at this host's Program Server before its main
processes serve, and a program's module exists nowhere else, so every test that
serves needs one. :func:`program_server` starts one for a test module, built
from the kernels the bf16 dense Qwen3 deployment launches.

The kernels build with the Program Server's own Python environment
(``crates/program-server/python``), which holds the compilers and the packages
kernel programs import (Triton, CuTe DSL, FlashInfer). The repository's
``.venv`` holds the development tools and the evaluator, and none of those, so a
build run with it stops at its first import.
"""

from __future__ import annotations

import os
import shutil
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

#: The descriptor keys the deployment launches: the model's own ``keys()`` plus
#: the executor's sampler launches, as ``crates/ops`` generators state them
#: (``notes/programs/qwen3-moe-keys.txt``). The store the fixture serves holds a
#: module for every one of them.
#:
#: This file used to install one kernel package, the attention one the daemon
#: names. That is enough for the epoch to *resolve* — the daemon names one
#: operator and the Program Server fills that slot — but not to serve: a main
#: process installs the kernels the model actually launches, and it resolves
#: them from this host's store (``listed_programs`` in ``src/engine_main.rs``).
#: With two packages there, the listing described nothing to overlay and the
#: precopy stood in ``this node's devices`` until its 300 s budget ended it.
KEYS = "notes/programs/qwen3-moe-keys.txt"

#: The precisions whose rows in ``KEYS`` belong to deployments other than this
#: file's. The key file covers three deployments — bf16 dense weights, NVFP4
#: weights, and an fp8 KV variant — in one store, as its own header states.
#: This file serves the bf16 dense one, so a row that names an NVFP4 encoding or
#: an ``f8e4m3`` scale or page is another deployment's, and one operator
#: (``kv_write_scaled``) stands in ``kv_write``'s place there. Building those
#: rows here fails on packages that deployment's owner owes, which says nothing
#: about whether this deployment can serve.
FOREIGN_PRECISIONS = ("nvfp4", "f8e4m3")
FOREIGN_OPERATORS = frozenset({"kv_write_scaled"})

def repository() -> Path:
    """The repository root: the workspace the policy package is built in."""
    return Path(__file__).resolve().parents[3]


def compiler() -> Path:
    """The interpreter of the Program Server's own environment.

    It compiles every kernel program, and nothing else provides what the
    programs import, so a host without it is told how to make it.
    """
    interpreter = (
        repository() / "crates/program-server/python/.venv/bin/python"
    )
    if not interpreter.exists():
        pytest.fail(
            f"no interpreter at {interpreter}: run `uv sync --project "
            "crates/program-server/python --python /usr/bin/python3.12`"
        )
    return interpreter


def precompiler() -> Path:
    """The wasm precompiler the Program Server turns a policy module into.

    A policy is built for wasm32 and precompiled for this host's CPU, so the
    server needs the precompiler before it can install a policy (design 5.3).
    It is named by the environment, found on the path, shipped beside the
    ``uniserve`` binary, or built from this tree, in that order.
    """
    named = os.environ.get("UNISERVE_SCHEDULING_PRECOMPILE")
    if named:
        return Path(named)
    found = shutil.which("scheduling-precompile")
    if found:
        return Path(found)
    beside = Path(require_uniserve_binary()).with_name("scheduling-precompile")
    if beside.exists():
        return beside
    root = repository()
    done = subprocess.run(
        [
            "cargo",
            "build",
            "--package",
            "scheduling",
            "--bin",
            "scheduling-precompile",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if done.returncode != 0:
        pytest.fail(
            f"scheduling-precompile did not build: {done.stderr[-2000:]}"
        )
    return root / "target" / "debug" / "scheduling-precompile"


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


def deployment_keys(store: Path) -> Path:
    """The key lines this file's deployment serves, written beside `store`.

    ``KEYS`` is the union of three deployments' rows; ``build_fixtures`` fails
    on any row that does not build, so handing it the union would let another
    deployment's missing package stop this one from serving. The rows kept here
    are the bf16 dense deployment's: everything whose key names none of
    ``FOREIGN_PRECISIONS`` and whose operator is not ``FOREIGN_OPERATORS``.
    Comments and blank lines are dropped — the store's build reads rows, and
    the selection remains readable in the file it came from.
    """
    kept = []
    for line in (repository() / KEYS).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        operator, _, key = stripped.partition("\t")
        if operator in FOREIGN_OPERATORS:
            continue
        lowered = stripped.lower()
        if any(precision in lowered for precision in FOREIGN_PRECISIONS):
            continue
        kept.append(stripped)
    if not kept:
        raise AssertionError(
            f"{KEYS} states no row for this deployment: "
            f"{sorted(FOREIGN_PRECISIONS)} and "
            f"{sorted(FOREIGN_OPERATORS)} excluded all of them"
        )
    path = store.parent / "deployment-keys.txt"
    path.write_text("\n".join(kept) + "\n")
    return path


def build_store(store: Path, target: str, python: Path) -> None:
    """Builds a module for every kernel key this deployment launches.

    `store` is ``build_fixtures``'s store, which it fills in its own shape: the
    installed package sources under ``<store>/packages`` and the prepared
    modules under ``<store>/modules``. The directory a Program Server serves is
    the second of those — a store root holds the modules, not the store's
    parent.

    ``build_fixtures`` (``crates/program-server/examples``) is the store's own
    producer: for each ``operator<TAB>key`` line of the key file it installs the
    operator's package, prepares that key for `target`, and resolves it to the
    module it names. It is given :func:`deployment_keys`, not ``KEYS``: this
    deployment's rows only, so a package another deployment owes fails that
    deployment's run and not this one. The builds are real — Triton, nvcc and
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
        str(deployment_keys(store)),
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
) -> Iterator[Path]:
    """The host's Program Server, with the deployment's programs installed.

    Autouse and module-scoped because every deployment of this file needs it:
    the daemon's leader resolves each epoch's programs at this server, and a
    program's module exists nowhere else. Two sets of programs are here, built
    once for the module:

    * the kernels — a module for every descriptor key the deployment launches,
      built into the store by ``build_fixtures`` (``KEYS``). The epoch is
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
        build_store(built, target, interpreter)
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


