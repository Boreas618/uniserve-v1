"""What every package compile script reads.

Its build's arguments, the descriptor's fields, the checked nvcc, and the
entries a native module must export.

The Build Server runs a package's ``src/build.py`` with its working directory
the package root and ``UNISERVE_PACKAGE``, ``UNISERVE_DESCRIPTOR``,
``UNISERVE_FIELDS``, ``UNISERVE_TARGET`` and ``UNISERVE_OUT`` in the
environment (``UNISERVE_NVCC`` too for a CUDA package); a direct run states
the same five as ``--package``, ``--descriptor``, ``--fields``, ``--target``
and ``--out``.

The key grammar is the Build Server's alone (``Descriptors::fields``,
``crates/build-server/src/manifest.rs``): a script is handed the fields the
server read from the key and never reads the key's text.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import tomllib


def arguments() -> tuple[pathlib.Path, str, str, pathlib.Path]:
    """The package root, descriptor key, target and output of one build."""
    reader = argparse.ArgumentParser()
    reader.add_argument("--package")
    reader.add_argument("--descriptor")
    reader.add_argument("--target")
    reader.add_argument("--out")
    known, _ = reader.parse_known_args()
    values = {
        "package": known.package or os.environ.get("UNISERVE_PACKAGE"),
        "descriptor": (
            known.descriptor or os.environ.get("UNISERVE_DESCRIPTOR")
        ),
        "target": known.target or os.environ.get("UNISERVE_TARGET"),
        "out": known.out or os.environ.get("UNISERVE_OUT"),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise SystemExit(f"missing {', '.join(missing)}")
    return (
        pathlib.Path(values["package"]).resolve(),
        values["descriptor"],
        values["target"],
        pathlib.Path(values["out"]),
    )


def manifest(package: pathlib.Path) -> dict:
    """The package's ``package.toml``."""
    return tomllib.loads((package / "package.toml").read_text())


def fields() -> dict[str, str]:
    """The descriptor's fields, as the Build Server read them from its key.

    The server reads a key against the package's ``[kernel.descriptors]`` and
    hands the build ``{dimension: value}`` as a JSON object in
    ``UNISERVE_FIELDS``, in the key's order; a direct run states the same
    object as ``--fields``, which ``uniserve build-server fields <package>
    <key>`` prints. A dimension the key leaves out is absent: the package's
    compilation accounts for a field it needs and the key lacks.
    """
    reader = argparse.ArgumentParser()
    reader.add_argument("--fields")
    known, _ = reader.parse_known_args()
    stated = known.fields or os.environ.get("UNISERVE_FIELDS")
    if not stated:
        raise SystemExit(
            "missing fields: the Build Server states them in UNISERVE_FIELDS, "
            "and `uniserve build-server fields <package> <key>` prints them"
        )
    return json.loads(stated)


def read_fields(package: pathlib.Path, key: str) -> dict[str, str]:
    """A descriptor key's fields, read by the Build Server's executable.

    For a tool that needs a key's fields outside a build: it asks the same
    reader a build is handed its fields by (``uniserve build-server fields``),
    through the executable ``UNISERVE_BINARY`` names (``uniserve`` on the
    search path when unset).

    Raises:
        ValueError: If the package's declared dimensions do not read the key.
        SystemExit: If the executable is not there to run.
    """
    executable = os.environ.get("UNISERVE_BINARY", "uniserve")
    try:
        completed = subprocess.run(
            [executable, "build-server", "fields", str(package), key],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise SystemExit(
            f"no {executable} to read the key {key} with: build the base "
            "executable (cargo build -p uniserve --bin uniserve) and name it "
            "in UNISERVE_BINARY"
        ) from None
    if completed.returncode != 0:
        raise ValueError(f"{key}: {completed.stderr.strip()}")
    return json.loads(completed.stdout)


def nvcc() -> tuple[str, str]:
    """The nvcc the Build Server checked, and the version line it prints.

    The Build Server compared it with the package's declared version.
    A compiler that does not print CUDA's own line for ``--version`` is
    refused here rather than producing a module from the host compiler.
    """
    compiler = os.environ["UNISERVE_NVCC"]
    completed = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True
    )
    if (
        completed.returncode != 0
        or "Cuda compilation tools" not in completed.stdout
    ):
        raise SystemExit(
            f"{compiler} is not CUDA's nvcc:\n"
            f"{completed.stdout[-300:]}{completed.stderr[-300:]}"
        )
    return compiler, completed.stdout.strip().splitlines()[-1].strip()


def require_exports(module: pathlib.Path, entries: list[str]) -> None:
    """Refuses a native module that does not export every one of ``entries``.

    The main process resolves them by name after loading it.
    """
    listed = subprocess.run(
        ["nm", "-D", "--defined-only", str(module)],
        capture_output=True,
        text=True,
        check=True,
    )
    symbols = {
        line.split()[-1] for line in listed.stdout.splitlines() if line.split()
    }
    missing = [entry for entry in entries if entry not in symbols]
    if missing:
        raise SystemExit(f"{module.name} does not export {', '.join(missing)}")
