"""The toolchain description: what this environment compiles with.

The Build Server reads this description once when it starts, by running
``uniserve-build-server toolchain --json``, and hands the part a build needs
back to every compilation in its request (:mod:`uniserve_build_server.request`),
so a compile script never looks a compiler up itself.

Every entry is resolved through a declared mechanism, never by guessing a path:

- the Python distributions of this environment, and the versions of the two
  Python compilers among them (Triton and the CuTe DSL), through
  :mod:`importlib.metadata`;
- the compile driver itself (this package) as the digest of its files, read
  through :mod:`importlib.resources`: a module is stored under the driver it
  was built through, so an edited driver or pin is another identity;
- the kernel packages' shared sources (the ``uniserve_kernels`` library of
  ``crates/ops/python``, which the packages import) as the digest of their
  files, read the same way, so an edit of what a package imports is another
  identity too;
- the CUDA toolkit of the release this environment's lock fixes (the release
  of the locked ``nvidia-cuda-runtime`` distribution, such as ``13.0``),
  through the ``cudart-<release>`` and ``cuda-<release>`` pkg-config modules
  NVIDIA's distribution packages register; nvcc's own version is read from the
  macros it predefines (``__CUDACC_VER_MAJOR__`` and its two siblings), not from
  the text of ``--version``;
- the host C++ compiler, the program ``c++`` names on this host's search path,
  resolved once to the file it runs.

A toolchain this host lacks is described as missing, with the reason, rather
than failing the description: a host without a CUDA toolkit still compiles
Triton packages, and only a build that needs the missing toolchain fails.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import importlib.resources
import json
import re
import shutil
import subprocess
import sys
from importlib.resources.abc import Traversable
from typing import Any

#: The distribution whose release fixes the CUDA toolkit this environment
#: compiles with: the CUDA runtime the locked compilers and libraries are built
#: against.
CUDA_RUNTIME = "nvidia-cuda-runtime"

#: The distributions of the two Python compilers.
TRITON = "triton"
CUTE_DSL = "nvidia-cutlass-dsl"

#: The program that names the host C++ compiler, as POSIX systems install it.
HOST_COMPILER = "c++"

#: The kernel packages' shared sources, a library this environment installs.
KERNELS = "uniserve_kernels"


@dataclasses.dataclass(frozen=True)
class Cuda:
    """The CUDA toolkit a build compiles CUDA C++ with.

    Every path is absolute. ``version`` is nvcc's release and build, such as
    ``13.0.48``: the version a CUDA package declares in its manifest.
    """

    version: str
    root: str
    nvcc: str
    include: str
    lib: str
    #: The driver API's link-time stub (``libcuda.so``): a module that calls
    #: the driver directly links against it, and the serving process loads the
    #: driver's own library.
    stubs: str


@dataclasses.dataclass(frozen=True)
class Toolchain:
    """This environment's toolchains, as the Build Server reads them.

    A toolchain is either available, with what a build needs of it, or
    missing, with the reason (:func:`available` and :func:`missing`): the
    serialized form is ``{"available": ...}`` or ``{"missing": "..."}``.
    """

    #: The interpreter of this environment, which runs every compilation.
    interpreter: str
    #: The SHA-256 of the compile driver's files, in hex.
    driver: str
    #: The SHA-256 of the kernel packages' shared sources, in hex.
    kernels: dict[str, Any]
    #: Every installed distribution by its normalized name (PEP 503), with its
    #: version.
    distributions: dict[str, str]
    triton: dict[str, Any]
    cute_dsl: dict[str, Any]
    cuda: dict[str, Any]
    host_compiler: dict[str, Any]

    def to_json(self) -> str:
        """The description as the Build Server reads it."""
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Toolchain:
        """The description a build request carries back."""
        return cls(**data)

    def require_cuda(self) -> Cuda:
        """The CUDA toolkit, or a refusal naming why this host has none."""
        return Cuda(**_require(self.cuda, "the CUDA toolkit"))

    def require_host_compiler(self) -> str:
        """The host C++ compiler, or a refusal naming why this host has none."""
        return _require(self.host_compiler, "the host C++ compiler")


def available(value: Any) -> dict[str, Any]:
    """A toolchain entry this host has."""
    return {"available": value}


def missing(reason: str) -> dict[str, Any]:
    """A toolchain entry this host lacks, with the reason."""
    return {"missing": reason}


def _require(entry: dict[str, Any], what: str) -> Any:
    if "available" in entry:
        return entry["available"]
    raise SystemExit(f"{what} is missing on this host: {entry['missing']}")


def normalized(name: str) -> str:
    """A distribution name as PEP 503 normalizes it."""
    return re.sub(r"[-_.]+", "-", name).lower()


def distributions() -> dict[str, str]:
    """Every installed distribution, by normalized name, with its version."""
    return {
        normalized(one.metadata["Name"]): one.version
        for one in importlib.metadata.distributions()
    }


def _files(
    root: Traversable, prefix: str = ""
) -> list[tuple[str, Traversable]]:
    """The files under ``root`` in name order, each with its relative name.

    Python's bytecode is derived from the sources and is no input of a build.
    """
    found: list[tuple[str, Traversable]] = []
    for entry in sorted(root.iterdir(), key=lambda one: one.name):
        if entry.name == "__pycache__":
            continue
        name = f"{prefix}{entry.name}"
        if entry.is_dir():
            found.extend(_files(entry, f"{name}/"))
        else:
            found.append((name, entry))
    return found


def digest(package: str) -> str:
    """The SHA-256 of an installed package's files, framed by names and lengths."""
    digest = hashlib.sha256()
    for name, entry in _files(importlib.resources.files(package)):
        data = entry.read_bytes()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def _version(distribution: str) -> dict[str, Any]:
    try:
        return available(importlib.metadata.version(distribution))
    except importlib.metadata.PackageNotFoundError:
        return missing(f"this environment has no {distribution} distribution")


def _pkg_config(module: str, variable: str) -> str:
    completed = subprocess.run(
        ["pkg-config", f"--variable={variable}", module],
        capture_output=True,
        text=True,
        check=False,
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or not value:
        raise LookupError(
            f"pkg-config states no {variable} of {module}: "
            f"{completed.stderr.strip() or 'no such variable'}"
        )
    return value


def _nvcc_version(nvcc: str) -> str:
    """The release and build of ``nvcc``, from the macros it predefines."""
    completed = subprocess.run(
        [nvcc, "-x", "cu", "-E", "-Xcompiler", "-dM", "/dev/null"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise LookupError(
            f"{nvcc} cannot preprocess: {completed.stderr.strip()}"
        )
    macros = dict(
        re.findall(
            r"^#define (__CUDACC_VER_\w+__) (\d+)$", completed.stdout, re.M
        )
    )
    try:
        return ".".join(
            macros[f"__CUDACC_VER_{part}__"]
            for part in ("MAJOR", "MINOR", "BUILD")
        )
    except KeyError as absent:
        raise LookupError(f"{nvcc} predefines no {absent}") from None


def cuda() -> dict[str, Any]:
    """The CUDA toolkit of the release this environment's lock fixes."""
    try:
        runtime = importlib.metadata.version(CUDA_RUNTIME)
    except importlib.metadata.PackageNotFoundError:
        return missing(f"this environment locks no {CUDA_RUNTIME} release")
    release = ".".join(runtime.split(".")[:2])
    try:
        root = _pkg_config(f"cudart-{release}", "cudaroot")
        lib = _pkg_config(f"cudart-{release}", "libdir")
        include = _pkg_config(f"cudart-{release}", "includedir")
        # The driver module's library directory holds the driver API's stub, in
        # the `stubs` directory NVIDIA's toolkit layout puts beside the runtime.
        stubs = f"{_pkg_config(f'cuda-{release}', 'libdir')}/stubs"
        nvcc = f"{root}/bin/nvcc"
        version = _nvcc_version(nvcc)
    except (LookupError, OSError) as error:
        return missing(
            f"no CUDA {release} toolkit registered with pkg-config "
            f"(NVIDIA's cuda-toolkit-{release.replace('.', '-')} package "
            f"provides it): {error}"
        )
    return available(
        dataclasses.asdict(
            Cuda(
                version=version,
                root=root,
                nvcc=nvcc,
                include=include,
                lib=lib,
                stubs=stubs,
            )
        )
    )


def kernels() -> dict[str, Any]:
    """The digest of the kernel packages' shared sources."""
    try:
        return available(digest(KERNELS))
    except ModuleNotFoundError:
        return missing(
            f"this environment has no {KERNELS} library (crates/ops/python)"
        )


def host_compiler() -> dict[str, Any]:
    """The host C++ compiler, resolved to the file it runs."""
    found = shutil.which(HOST_COMPILER)
    if found is None:
        return missing(f"no {HOST_COMPILER} on this host's search path")
    return available(found)


def describe() -> Toolchain:
    """This environment's toolchains."""
    return Toolchain(
        interpreter=sys.executable,
        driver=digest(__package__),
        kernels=kernels(),
        distributions=distributions(),
        triton=_version(TRITON),
        cute_dsl=_version(CUTE_DSL),
        cuda=cuda(),
        host_compiler=host_compiler(),
    )
