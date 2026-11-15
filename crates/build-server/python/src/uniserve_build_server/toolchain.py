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
  ``crates/ops/python``, which the packages import and compile against) as the
  digest of their files, read the same way, so an edit of what a package
  imports or includes is another identity too, and as the directory of the C
  ABI headers the library ships, which every native compilation includes;
- the CUDA toolkit of the release this environment's lock fixes (the release
  of the locked ``nvidia-cuda-runtime`` distribution, such as ``13.0``),
  through the ``cudart-<release>`` and ``cuda-<release>`` pkg-config modules
  NVIDIA's distribution packages register; nvcc's own version is read from the
  macros it predefines (``__CUDACC_VER_MAJOR__`` and its two siblings), not from
  the text of ``--version``;
- cuDNN of the release this environment's lock fixes (the major release of the
  locked ``nvidia-cudnn-cu13`` distribution), as the host's distribution package
  installs it on the default search paths: the host C++ compiler finds its
  header, whose macros state the version, and resolves its library, the file
  the serving process's loader opens by the same search;
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
import pathlib
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

#: The distribution whose major release fixes the cuDNN this environment
#: compiles against, and the header and library of that release's distribution
#: package.
CUDNN = "nvidia-cudnn-cu13"
CUDNN_HEADER = "cudnn_version.h"
CUDNN_LIBRARY = "libcudnn.so"

#: The program that names the host C++ compiler, as POSIX systems install it.
HOST_COMPILER = "c++"

#: The kernel packages' shared sources, a library this environment installs.
KERNELS = "uniserve_kernels"

#: The directory of that library which holds its C ABI headers.
INCLUDE = "include"


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
class Cudnn:
    """The cuDNN a module that calls it compiles and links against.

    ``version`` is the header's release, such as ``9.12.0``; ``include`` holds
    its headers and ``lib`` its library, both directories of the default
    search paths the distribution package installs into, so a module links the
    library by its soname and the serving process's loader finds the same file.
    """

    version: str
    include: str
    lib: str


@dataclasses.dataclass(frozen=True)
class Kernels:
    """The kernel packages' shared sources, as this environment installs them.

    ``include`` is absolute: the directory of the C ABI headers generated from
    the operator crate's call structures (``uniserve_<operator>.h`` and the
    ``uniserve_abi.h`` they share), which a native compilation passes as an
    include directory. The digest covers those headers with the rest of the
    library's files.
    """

    #: The SHA-256 of the library's files, in hex.
    digest: str
    include: str


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
    #: The kernel packages' shared sources (:class:`Kernels`).
    kernels: dict[str, Any]
    #: Every installed distribution by its normalized name (PEP 503), with its
    #: version.
    distributions: dict[str, str]
    triton: dict[str, Any]
    cute_dsl: dict[str, Any]
    cuda: dict[str, Any]
    cudnn: dict[str, Any]
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

    def require_cudnn(self) -> Cudnn:
        """cuDNN, or a refusal naming why this host has none."""
        return Cudnn(**_require(self.cudnn, "cuDNN"))

    def require_kernels(self) -> Kernels:
        """The kernel library, or a refusal naming why this host has none."""
        return Kernels(**_require(self.kernels, "the kernel library"))

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
    """The SHA-256 of an installed package's files, framed by names and
    lengths.
    """  # noqa: D205
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


def cudnn(compiler: dict[str, Any]) -> dict[str, Any]:
    """The cuDNN of the release the lock fixes, as the host compiler finds it.

    The header is found by preprocessing an include of it, whose line markers
    name the file and whose macros state the release; the library is the file
    the compiler driver resolves on the linker's search path.
    """
    try:
        locked = importlib.metadata.version(CUDNN).split(".")[0]
    except importlib.metadata.PackageNotFoundError:
        return missing(f"this environment locks no {CUDNN} release")
    if "available" not in compiler:
        return missing(
            f"no host C++ compiler finds cuDNN: {compiler['missing']}"
        )
    host = compiler["available"]
    preprocessed = subprocess.run(
        [host, "-x", "c++", "-E", "-dD", "-"],
        input=f"#include <{CUDNN_HEADER}>\n",
        capture_output=True,
        text=True,
        check=False,
    )
    if preprocessed.returncode != 0:
        return missing(
            f"the host compiler finds no {CUDNN_HEADER} (NVIDIA's "
            f"libcudnn{locked}-dev package provides it): "
            f"{preprocessed.stderr.strip()}"
        )
    header = re.search(
        rf'^# \d+ "(/[^"]*)/{re.escape(CUDNN_HEADER)}"',
        preprocessed.stdout,
        re.M,
    )
    macros = dict(
        re.findall(
            r"^#define (CUDNN_(?:MAJOR|MINOR|PATCHLEVEL)) (\d+)$",
            preprocessed.stdout,
            re.M,
        )
    )
    resolved = subprocess.run(
        [host, f"-print-file-name={CUDNN_LIBRARY}"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if header is None or len(macros) != 3:
        return missing(f"{CUDNN_HEADER} states no cuDNN release")
    version = ".".join(
        macros[f"CUDNN_{part}"] for part in ("MAJOR", "MINOR", "PATCHLEVEL")
    )
    if macros["CUDNN_MAJOR"] != locked:
        return missing(
            f"the host's cuDNN is {version}, and this environment's lock fixes "
            f"release {locked}"
        )
    # The driver prints the bare name for a library its search does not find.
    if not resolved.startswith("/"):
        return missing(
            f"the host compiler resolves no {CUDNN_LIBRARY} (NVIDIA's "
            f"libcudnn{locked}-dev package provides it)"
        )
    return available(
        dataclasses.asdict(
            Cudnn(
                version=version,
                include=header.group(1),
                lib=str(pathlib.Path(resolved).resolve().parent),
            )
        )
    )


def kernels() -> dict[str, Any]:
    """The kernel library: its digest and its C ABI headers' directory."""
    try:
        include = importlib.resources.files(KERNELS) / INCLUDE
        return available(
            dataclasses.asdict(
                Kernels(digest=digest(KERNELS), include=str(include))
            )
        )
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
    compiler = host_compiler()
    return Toolchain(
        interpreter=sys.executable,
        driver=digest(__package__),
        kernels=kernels(),
        distributions=distributions(),
        triton=_version(TRITON),
        cute_dsl=_version(CUTE_DSL),
        cuda=cuda(),
        cudnn=cudnn(compiler),
        host_compiler=compiler,
    )
