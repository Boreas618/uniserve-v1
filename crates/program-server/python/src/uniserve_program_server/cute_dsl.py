"""The CuTe DSL compile driver every CuTe DSL package's ``src/build.py`` calls.

The Program Server's CuTe DSL source runs a package's ``build.py`` for one
(descriptor key, target) pair and keeps what it writes into the build's output
directory: ``module.so``, a shared object with the C ABI entry the package
declares (``src/entry.h``) and the kernel's cubin embedded.

Steps: ``cute.compile`` the kernel for fake arguments on an explicit
architecture (no GPU is needed: fake tensors from ``cutlass.cute.runtime`` and
a fake stream describe the runtime signature, and the architecture comes from
the target), ``export_to_c`` to emit the host launcher object and its header,
then link the object with the package's entry translation unit and the CuTe
DSL's ``libcuda_dialect_runtime_static.a`` plus the CUDA runtime. The archive
carries the device-dialect runtime; the CUDA runtime must come with it, or the
module loads with undefined ``cuda*`` symbols.

A package's ``build.py`` states only ``spec(fields, parameters, target)`` and
ends with ``main(spec)``. The spec names:

- ``name``: the exported launcher's name, which ``entry.cpp`` calls;
- ``entry``: the C entry symbol the module must export;
- either ``kernel``, ``args`` and ``options`` for ``cute.compile``, or
  ``compile``, a callable returning the compiled object, for a package whose
  kernel is compiled through its vendor's own selection (FlashAttention-4);
- ``defines`` (optional): preprocessor definitions of the entry unit.

The module is a **shared library with an entry symbol**, not a cubin ``ops``
launches: the call's parameters are the ``Call`` struct ``entry.h`` declares,
whose layout a ``static_assert`` fixes, so no launch signature is written to a
``metadata.json``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import tempfile
from collections.abc import Callable

import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream

from uniserve_program_server import package

#: What a package's ``spec`` is: the descriptor's fields, the manifest's
#: ``[parameters]`` and the target, to the compilation of one descriptor.
Spec = Callable[[dict[str, str], dict, str], dict]


def cutlass_library() -> pathlib.Path:
    """The CuTe DSL static runtime archive for this host's CUDA."""
    found = importlib.util.find_spec("nvidia_cutlass_dsl")
    # The distribution is a namespace package: its directory is in the search
    # locations, not in an origin module's path.
    locations = list(getattr(found, "submodule_search_locations", None) or ())
    if not locations:
        raise SystemExit(
            "the CuTe DSL package nvidia_cutlass_dsl is not installed"
        )
    root = pathlib.Path(locations[0])
    for major in ("cu13", "cu12"):
        archive = root / major / "lib" / "libcuda_dialect_runtime_static.a"
        if archive.exists():
            return archive
    raise SystemExit(
        "no libcuda_dialect_runtime_static.a in nvidia_cutlass_dsl"
    )


def cuda_home() -> pathlib.Path:
    """The CUDA toolkit `CUDA_HOME` names: the runtime's headers and library."""
    home = os.environ.get("CUDA_HOME")
    if not home:
        raise SystemExit(
            "CUDA_HOME names the CUDA toolkit whose runtime the module "
            "links against"
        )
    return pathlib.Path(home)


def link(
    out: pathlib.Path,
    objects: list[pathlib.Path],
    header_dir: pathlib.Path,
    entry: str,
    archive: pathlib.Path,
) -> None:
    """Link the exported object and the entry unit into ``out/module.so``.

    The dialect runtime archive is linked whole and the CUDA runtime
    dynamically with ``--no-as-needed``: without the runtime the module loads
    with undefined ``cuda*`` symbols, and without ``--no-as-needed`` the linker
    drops it because the archive references it only indirectly.
    """
    command = [
        os.environ.get("CXX", "g++"),
        "-shared",
        "-O2",
        "-std=c++20",
        "-fPIC",
        "-o",
        str(out / "module.so"),
        *(str(one) for one in objects),
        f"-I{header_dir}",
        f"-I{cuda_home() / 'include'}",
        "-Wl,--whole-archive",
        str(archive),
        "-Wl,--no-whole-archive",
        "-Wl,--no-as-needed",
        f"-L{cuda_home() / 'lib64'}",
        "-lcudart",
    ]
    built = subprocess.run(command, capture_output=True, text=True, check=False)
    if built.returncode != 0:
        raise SystemExit(f"linking module.so failed:\n{built.stderr}")
    package.require_exports(out / "module.so", [entry])


def compile_one(
    root: pathlib.Path, key: str, target: str, out: pathlib.Path, spec: Spec
) -> None:
    """Compile descriptor ``key`` of the package at ``root`` into ``out``."""
    # The compile describes the architecture itself: the target names it, so
    # no device is read and none is needed.
    os.environ["CUTE_DSL_ARCH"] = target
    fields = package.fields(root, key)
    try:
        stated = spec(
            fields, package.manifest(root).get("parameters", {}), target
        )
    except KeyError as missing:
        raise SystemExit(f"{key}: the descriptor names no {missing}") from None
    if "compile" in stated:
        compiled = stated["compile"]()
    else:
        compiled = cute.compile(
            stated["kernel"],
            *stated["args"],
            **stated["options"],
            stream=make_fake_stream(),
        )

    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as staging:
        staging = pathlib.Path(staging)
        name = stated["name"]
        compiled.export_to_c(str(staging), name)
        # The package's entry unit is compiled beside the generated header, so
        # it includes both the export's launcher declarations and its own ABI.
        entry_unit = staging / "entry.cpp"
        entry_unit.write_text((root / "src" / "entry.cpp").read_text())
        (staging / "entry.h").write_text((root / "src" / "entry.h").read_text())
        object_file = staging / f"{name}.o"
        built = subprocess.run(
            [
                os.environ.get("CXX", "g++"),
                "-c",
                "-O2",
                "-std=c++20",
                "-fPIC",
                "-o",
                str(entry_unit.with_suffix(".o")),
                str(entry_unit),
                *(f"-D{one}" for one in stated.get("defines", [])),
                f"-I{staging}",
                f"-I{cuda_home() / 'include'}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if built.returncode != 0:
            raise SystemExit(
                f"compiling the entry unit failed:\n{built.stderr}"
            )
        link(
            out,
            [object_file, entry_unit.with_suffix(".o")],
            staging,
            stated["entry"],
            cutlass_library(),
        )
    print(
        json.dumps(
            {"descriptor": key, "target": target, "entry": stated["entry"]}
        )
    )


def main(spec: Spec) -> None:
    """Compile the descriptor the build names into its output directory."""
    root, key, target, out = package.arguments()
    compile_one(root, key, target, out, spec)
