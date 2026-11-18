"""A Build Server of a test session, through its command-line interface.

A deployment's epoch is resolved at its host's Build Server before its main
processes serve, and a package's module exists nowhere else, so every test
that serves names one, as does a kernel check that loads a built module.
:func:`build_server` starts one over a fresh package root and store under a
runtime directory of its own, where it binds its default socket, which its
readiness record states (``endpoints.build``), with the repository as its
source tree (``--source``): its Build Server environment compiles, and the
server installs the tree's serving packages (every kernel package of the
operator tree, the scheduling policy and the engine) before it serves. It
builds for the architectures of this host's devices, which it reads from the
driver itself. :class:`BuildServer` installs further packages and resolves
modules with the ``--json`` forms of ``uniserve build-server install`` and
``resolve``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import tempfile
import tomllib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from uniserve_testing.servers import Server, environment, launch, socket_path

#: How long a Build Server takes to serve, in seconds: uv brings its
#: environment to the tree's lock, it describes its toolchains, which imports
#: the compilers of the environment, and it installs the tree's packages,
#: which snapshots the engine's whole workspace.
START_S = 300.0

#: The tree's packages, relative to the repository, that tests install or
#: read themselves: the kernel packages' tree, the scheduling policy and the
#: engine.
KERNEL_TREE = Path("crates", "ops", "kernels")
POLICY_PACKAGE = Path("crates", "scheduling", "policies", "arrival")
ENGINE_PACKAGE = Path("crates", "engine")


@dataclasses.dataclass(frozen=True)
class Installed:
    """A package revision a Build Server installed."""

    op: str
    backend: str
    revision: int


@dataclasses.dataclass(frozen=True)
class Resolved:
    """A module a Build Server resolved: its form, its files in the store,
    and for an executable module the file its process runs as.
    """  # noqa: D205

    form: str
    files: tuple[Path, ...]
    executable: Path | None

    @property
    def directory(self) -> Path:
        """The store directory that holds the module's files: their common
        parent, since a module may keep files in directories of its own (a
        trtllm-gen module's ``cubins/``).
        """  # noqa: D205
        return (
            Path(os.path.commonpath(self.files))
            if len(self.files) > 1
            else self.files[0].parent
        )


class BuildServer:
    """A running Build Server and the commands a test sends it."""

    def __init__(self, uniserve: Path, server: Server) -> None:
        """The server `server` started with the executable `uniserve`."""
        self.uniserve = uniserve
        self.server = server

    @property
    def socket(self) -> Path:
        """The server's socket, as its readiness record states it."""
        return socket_path(self.server.endpoint("build"))

    @property
    def settings(self) -> dict[str, Any]:
        """The settings the server serves with, as its readiness record
        states them (``build_server::command::Settings``, its socket
        explicit): what a deployment at this server states as its
        ``build_config`` (:func:`build_config`).
        """  # noqa: D205
        return self.server.record["settings"]

    def install(self, package: Path) -> Installed:
        """Install the package directory `package` and return its revision.

        Raises:
            RuntimeError: With the server's reason, when it refuses.
        """
        installed = self._command("install", str(package))
        return Installed(**installed)

    def resolve(
        self,
        op: str,
        backend: str,
        revision: int,
        key: str | None = None,
        target: str | None = None,
    ) -> Resolved:
        """The module of descriptor `key` (a single-module package's when
        unset) of a package revision, built for `target` (this host's
        platform when unset, which an executable package is built for),
        which the server builds first when its store misses it.

        Raises:
            RuntimeError: With the server's reason, when it cannot build or
                resolve the module.
        """  # noqa: D205
        keyed = ["--key", key] if key is not None else []
        targeted = ["--target", target] if target is not None else []
        resolved = self._command(
            "resolve", op, backend, str(revision), *keyed, *targeted
        )
        executable = resolved["executable"]
        return Resolved(
            form=resolved["form"],
            files=tuple(Path(file) for file in resolved["files"]),
            executable=Path(executable) if executable is not None else None,
        )

    def list(self) -> list[dict]:
        """Every package revision the server installed, with the modules
        its store holds of it, as ``build-server list --json`` prints them:
        each ``{op, backend, revision, modules}``, each module ``{target,
        descriptor, form, interface_version}``.
        """  # noqa: D205
        return self._command("list")

    def _command(self, command: str, *arguments: str) -> Any:
        """The JSON answer of one ``build-server`` command at the server."""
        completed = subprocess.run(
            [
                str(self.uniserve),
                "build-server",
                command,
                "--socket",
                str(self.socket),
                "--json",
                *arguments,
            ],
            env=environment(),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"build-server {command} {' '.join(arguments)} failed: "
                f"{completed.stderr.strip()}"
            )
        return json.loads(completed.stdout)


def build_config(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The options of a package's ``config.toml`` that state a Build
    Server's `settings` as a deployment's ``build_config``.

    The deployment states its host's Build Server, and its daemon refuses a
    server that serves with other settings (decision D15), so a deployment
    at a server a test started states exactly the settings that server's
    readiness record states.
    """  # noqa: D205
    return {"build_config": dict(settings)}


@contextmanager
def build_server(
    uniserve: Path, repository: Path, work: Path, *, scale: float = 1.0
) -> Iterator[BuildServer]:
    """A Build Server over a fresh package root and store under `work`,
    with `repository` as its source tree, until the block ends.

    The server binds its default socket in a runtime directory of its own,
    which this creates in the system's temporary directory: a Unix socket's
    path holds at most 107 bytes, and `work` may lie deep in a run's
    artifacts. Its package root and store live as long as the block: a
    revision counts from 1 in a fresh package root, so a store kept across
    runs would answer a changed package with the module older source built.
    It serves once it has installed the tree's packages, so a deployment's
    start builds what it launches and later deployments reuse it.
    """  # noqa: D205
    runtime = Path(tempfile.mkdtemp(prefix="uniserve-"))
    server = launch(
        [
            str(uniserve),
            "build-server",
            "serve",
            "--source",
            str(repository),
            "--package-root",
            str(work / "packages"),
            "--store-root",
            str(work / "modules"),
        ],
        work / "build-server.log",
        environment(XDG_RUNTIME_DIR=str(runtime)),
        scale=scale,
    )
    try:
        server.wait("serving", START_S)
        yield BuildServer(uniserve, server)
    finally:
        server.stop()
        shutil.rmtree(runtime, ignore_errors=True)


def derived_package(
    repository: Path, mutations: Path, variant: str, directory: Path
) -> Path:
    """A revision of a serving package, derived by a declared mutation and
    written into `directory`.

    `mutations` is a TOML file that names the serving package (``package``,
    relative to the repository) and, for each variant, the files the variant
    adds from the mutation file's own directory (``add``: a package path to a
    fixture path) and the exact texts it replaces in the package's files
    (``replace``). The revision is the serving package's tree with those
    changes, so it never holds a copy that drifts from what serves.

    Raises:
        AssertionError: When a replaced text does not occur exactly once in
            the serving package, naming the mutation, the file and the text.
    """  # noqa: D205
    declared = tomllib.loads(mutations.read_text())
    mutation = declared["variants"][variant]
    package = directory / f"{mutations.stem}-{variant}"
    shutil.copytree(
        repository / declared["package"],
        package,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for target, source in mutation.get("add", {}).items():
        shutil.copyfile(mutations.parent / source, package / target)
    for replaced in mutation.get("replace", []):
        path = package / replaced["file"]
        text = path.read_text()
        found = text.count(replaced["old"])
        if found != 1:
            raise AssertionError(
                f"{mutations.stem} {variant}: {replaced['file']} of "
                f"{declared['package']} holds {found} copies of "
                f"{replaced['old']!r}, and the variant replaces one"
            )
        path.write_text(text.replace(replaced["old"], replaced["new"]))
    return package
