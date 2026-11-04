"""Compile one descriptor of a CuTe DSL kernel program into a native module.

The Program Server's CuTe DSL source runs this package's ``build.py`` for one
(descriptor key, target) pair and keeps what it writes into ``--out``:
``module.so``, a shared object with the C ABI entry the package declares
(``src/entry.h``) and the kernel's cubin embedded.

Steps: ``cute.compile`` the kernel for fake arguments on an explicit
architecture (no GPU is needed: fake tensors from ``cutlass.cute.runtime`` and
a fake stream describe the runtime signature, and the architecture comes from
the target), ``export_to_c`` to emit the host launcher object and its header,
then link the object with the package's entry translation unit and the CuTe
DSL's ``libcuda_dialect_runtime_static.a`` plus the CUDA runtime. The archive
carries the device-dialect runtime; the CUDA runtime must come with it, or the
module loads with undefined ``cuda*`` symbols.

The descriptor dimensions and their values come from the package manifest, so
the keys this script accepts are exactly the range the manifest declares.
"""

import json
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream

#: Threads in a warp of every target these packages build for.
WARP_SIZE = 32


def capability(target):
    """The compute capability of an architecture target, ``"sm_100a"`` -> 100."""
    digits = target.removeprefix("sm_")
    while digits and digits[-1].isalpha():
        digits = digits[:-1]
    if not digits.isdigit():
        raise SystemExit(f"target {target!r} is not an sm_<capability> architecture")
    return int(digits)


def parse_key(key, dimensions):
    """Read a descriptor key into ``{dimension: value}``, order-free.

    A key is ``field_value`` pairs joined with ``_`` in the order the
    operator's ``Descriptor`` prints. Field names and declared values may
    contain ``_``, so a field is the longest declared name the rest starts with
    and a value the longest declared value that follows it whole.
    """
    names = sorted(dimensions, key=len, reverse=True)
    remaining = key
    fields = {}
    while remaining:
        name = next((one for one in names if remaining.startswith(one + "_")), None)
        if name is None:
            raise SystemExit(f"{key}: no dimension among {names} starts {remaining!r}")
        rest = remaining[len(name) + 1 :]
        values = sorted(dimensions[name], key=len, reverse=True)
        value = next(
            (one for one in values if rest == one or rest.startswith(one + "_")),
            None,
        )
        if value is None:
            raise SystemExit(
                f"{key}: {name} takes none of {dimensions[name]} at {rest!r}"
            )
        fields[name] = value
        remaining = rest[len(value) :].removeprefix("_")
    missing = sorted(set(dimensions) - set(fields))
    if missing:
        raise SystemExit(f"{key}: the descriptor names no {missing}")
    return fields


def cutlass_library():
    """The CuTe DSL static runtime archive for this host's CUDA."""
    import importlib.util  # noqa: PLC0415 - resolved from the toolchain's package

    found = importlib.util.find_spec("nvidia_cutlass_dsl")
    # The distribution is a namespace package: its directory is in the search
    # locations, not in an origin module's path.
    locations = list(getattr(found, "submodule_search_locations", None) or ())
    if not locations:
        raise SystemExit("the CuTe DSL package nvidia_cutlass_dsl is not installed")
    root = Path(locations[0])
    for major in ("cu13", "cu12"):
        archive = root / major / "lib" / "libcuda_dialect_runtime_static.a"
        if archive.exists():
            return archive
    raise SystemExit("no libcuda_dialect_runtime_static.a in nvidia_cutlass_dsl")


def cuda_home():
    """The CUDA toolkit's directory: the runtime's headers and library."""
    return Path(os.environ.get("CUDA_HOME") or "/usr/local/cuda")


def link(out, objects, header_dir, entry, archive):
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
    if entry not in exported_symbols(out / "module.so"):
        raise SystemExit(f"module.so does not export {entry}")


def exported_symbols(shared_object):
    """The dynamic symbols ``shared_object`` defines."""
    listed = subprocess.run(
        ["nm", "-D", "--defined-only", str(shared_object)],
        capture_output=True,
        text=True,
        check=True,
    )
    return {line.split()[-1] for line in listed.stdout.splitlines() if line.split()}


def compile_one(package, target, key, out, spec):
    """Compile descriptor ``key`` of ``package`` for ``target`` into ``out``."""
    # The compile describes the architecture itself: the target names it, so
    # no device is read and none is needed.
    os.environ["CUTE_DSL_ARCH"] = target
    manifest = tomllib.loads((package / "package.toml").read_text())
    fields = parse_key(key, manifest["kernel"]["descriptors"]["dimensions"])
    described = spec(fields, manifest.get("parameters", {}), target)
    # A package whose kernel is not its own source compiles it through the
    # vendor's own selection: `compile` hands back the compiled object.
    if "compile" in described:
        compiled = described["compile"]()
    else:
        compiled = cute.compile(
            described["kernel"],
            *described["args"],
            **described["options"],
            stream=make_fake_stream(),
        )

    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as staging:
        staging = Path(staging)
        name = described["name"]
        compiled.export_to_c(str(staging), name)
        # The package's entry unit is compiled beside the generated header, so
        # it includes both the export's launcher declarations and its own ABI.
        entry_unit = staging / "entry.cpp"
        entry_unit.write_text((package / "src" / "entry.cpp").read_text())
        (staging / "entry.h").write_text((package / "src" / "entry.h").read_text())
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
                f"-I{staging}",
                f"-I{cuda_home() / 'include'}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if built.returncode != 0:
            raise SystemExit(f"compiling the entry unit failed:\n{built.stderr}")
        link(
            out,
            [object_file, entry_unit.with_suffix(".o")],
            staging,
            described["entry"],
            cutlass_library(),
        )
    print(json.dumps({"descriptor": key, "target": target, "entry": described["entry"]}))


def arguments():
    """The build's arguments: the Program Server's environment, or flags.

    The Program Server runs a package's script with `UNISERVE_PACKAGE`,
    `UNISERVE_DESCRIPTOR`, `UNISERVE_TARGET` and `UNISERVE_OUT` set, and
    `MAX_JOBS` capping the job's own parallelism. A direct run states the same
    four as flags, which is how a package is proved to compile outside it.
    """
    if os.environ.get("UNISERVE_OUT"):
        package = Path(os.environ["UNISERVE_PACKAGE"])
        manifest = tomllib.loads((package / "package.toml").read_text())
        target = os.environ.get("UNISERVE_TARGET") or manifest["kernel"]["architectures"][0]
        return (
            package,
            target,
            os.environ["UNISERVE_DESCRIPTOR"],
            Path(os.environ["UNISERVE_OUT"]),
        )
    flags = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    return (
        Path(flags["--package"]),
        flags["--target"],
        flags["--descriptor"],
        Path(flags["--out"]),
    )


def main(spec):
    """Compile the descriptor the build names into its output directory."""
    compile_one(*arguments(), spec)
