"""The Build Servers of a test session, through their command-line interface.

A deployment's epoch is resolved at its own Build Server on its node before its
main processes serve: the deployment's daemon starts it, and it compiles and
keeps its modules apart from every other deployment's. A run starts one Build
Server of its own (:func:`build_server`) over a fresh store under its work
directory (``store/packages`` and ``store/modules``, the layout of a
``build-server build --store`` store), with the repository as its source tree
(``--source``): its Build Server environment compiles, and the server installs
the tree's serving packages (every kernel package of the operator tree, the
scheduling policy, the processing program and the engine) before it serves, as
a kernel check that loads a built module needs. Every deployment of the run
reads that store as a read-only store (``build_config.read_only_stores``,
:func:`deployment_config`), so its own Build Server finds the tree's packages
installed and installs none of them again, and keeps what it compiles in the
run's data directory for deployments (:attr:`BuildServer.data`, which holds the
``XDG_DATA_HOME`` and ``XDG_CACHE_HOME`` of the run's daemons), under the
deployment's name: the cases of a run that serve a deployment of one name
compile its modules once, and deployments of other names never read them. A
deployment's packages are its case's own (``build_config.package_root``), so a
revision one case installs is selected by no other case. :class:`BuildServer`
installs further packages and resolves modules with the ``--json`` forms of
``uniserve build-server install`` and ``resolve``, at the run's server or at a
deployment's (:meth:`BuildServer.of`).
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

from uniserve_testing.servers import environment, launch, socket_path

#: The deployment a check's Build Server serves: a server builds for one
#: deployment, and a check's is the only one in its runtime directory.
DEPLOYMENT = "checks"

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


#: The directory of a Build Server's store under its work directory, as
#: ``build-server build --store`` lays one out: its ``packages`` and
#: ``modules``.
STORE = "store"


@dataclasses.dataclass(frozen=True)
class BuildServer:
    """A running Build Server and the commands a test sends it: the run's
    (:func:`build_server`), whose `store` and `data` its deployments read and
    write, or a deployment's (:meth:`of`).
    """  # noqa: D205

    #: The executable the commands run.
    uniserve: Path
    #: The server's socket.
    socket: Path
    #: The settings it serves with, as its readiness record states them
    #: (``build_server::command::Settings``).
    settings: Mapping[str, Any]
    #: The store it writes, which the run's deployments read; ``None`` for a
    #: deployment's server.
    store: Path | None = None
    #: The data directory of the run's deployments, which holds the
    #: ``XDG_DATA_HOME`` (``data``) and ``XDG_CACHE_HOME`` (``cache``) of
    #: their daemons, where each deployment's Build Server keeps its modules
    #: and caches under the deployment's name; ``None`` for a deployment's
    #: server.
    data: Path | None = None

    @classmethod
    def of(cls, uniserve: Path, runtime: Path, deployment: str) -> BuildServer:
        """The Build Server of the deployment `deployment` serving under the
        runtime directory `runtime`, which its daemon started: its socket in
        the deployment's runtime directory and the settings the readiness
        record beside it states.
        """  # noqa: D205
        socket = deployment_socket(runtime, deployment, SERVICE)
        record = json.loads(socket.with_suffix(".ready").read_text())
        return cls(uniserve, socket, record["settings"])

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


#: The server name of the Build Server: its command, and its socket's name
#: in a deployment's runtime directory.
SERVICE = "build-server"


def deployment_socket(runtime: Path, deployment: str, service: str) -> Path:
    """The socket of the server `service` (``build-server`` or
    ``storage-server``) of the deployment `deployment` under the runtime
    directory `runtime`: its daemon starts it there, in the deployment's own
    directory.
    """  # noqa: D205
    return runtime / "uniserve" / "deployments" / deployment / f"{service}.sock"


def deployment_config(run: BuildServer, packages: Path) -> dict[str, Any]:
    """The options of a package's ``config.toml`` that state the Build Server
    of a deployment of the run whose server is `run`: the run's source tree,
    targets and limits, the run's store as a read-only store beside the ones
    the run's server reads, and `packages`, the case's own directory, as its
    package root. Its modules are kept in its default store root, the
    deployment's directory of the run's data directory
    (:attr:`BuildServer.data`).
    """  # noqa: D205
    assert run.store is not None, "a deployment's server derives from a run's"
    settings = {
        key: value
        for key, value in run.settings.items()
        if key not in ("package_root", "store_roots", "upstream")
    }
    settings["read_only_stores"] = [
        *settings.get("read_only_stores", []),
        str(run.store),
    ]
    settings["package_root"] = str(packages)
    return {"build_config": settings}


@contextmanager
def build_server(
    uniserve: Path, repository: Path, work: Path, *, scale: float = 1.0
) -> Iterator[BuildServer]:
    """A Build Server over a fresh store under `work` (:data:`STORE`), with
    `repository` as its source tree, until the block ends; its deployments'
    data directory is ``work/deployments``.

    The server serves the deployment :data:`DEPLOYMENT`, whose socket it binds
    in a runtime directory of its own, which this creates in the system's
    temporary directory: a Unix socket's path holds at most 107 bytes, and
    `work` may lie deep in a run's artifacts. Its package root and store live
    as long as the block: a revision counts from 1 in a fresh package root, so
    a store kept across runs would answer a changed package with the module
    older source built.
    It serves once it has installed the tree's packages, which the run's
    deployments then read from its store.
    """  # noqa: D205
    store = work / STORE
    data = work / "deployments"
    data.mkdir(parents=True, exist_ok=True)
    runtime = Path(tempfile.mkdtemp(prefix="uniserve-"))
    server = launch(
        [
            str(uniserve),
            "build-server",
            "serve",
            "--deployment",
            DEPLOYMENT,
            "--source",
            str(repository),
            "--package-root",
            str(store / "packages"),
            "--store-root",
            str(store / "modules"),
        ],
        work / "build-server.log",
        environment(XDG_RUNTIME_DIR=str(runtime)),
        scale=scale,
    )
    try:
        server.wait("serving", START_S)
        yield BuildServer(
            uniserve,
            socket_path(server.endpoint("build")),
            server.record["settings"],
            store=store,
            data=data,
        )
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
