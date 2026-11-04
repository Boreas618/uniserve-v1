"""Compile one descriptor of a Triton kernel program with ``triton.compile``.

The Program Server's Triton source runs this package's ``build.py`` for one
(descriptor key, target) pair and keeps what it writes into ``--out``:
``module.cubin`` and ``metadata.json``, the metadata JSON ``triton.compile``
itself wrote for the module. Triton's own AOT tool expresses neither clusters
nor PDL nor global scratch (``triton/tools/compile.py``), so the module is
compiled through ``triton.compile`` on an explicit target, which needs no GPU.

The descriptor dimensions and their values come from the package manifest, so
the keys this script accepts are exactly the range the manifest declares.
"""

import json
import os
import sys
import tomllib
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource

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
    """Read a descriptor key into ``{dimension: value}``.

    A key is ``field_value`` pairs joined with ``_`` in the order the
    operator's ``Descriptor`` prints, which this parser does not depend on.
    Field names and declared values may themselves contain ``_``
    (``float8_e4m3``), so a field is the longest declared name the rest starts
    with and a value the longest declared value that follows it whole.
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


def compile_one(package, target, key, out, spec):
    """Compile descriptor ``key`` of ``package`` for ``target`` into ``out``.

    ``package`` is the package directory, which holds ``package.toml`` and
    ``src/``. ``spec`` states what one descriptor of the package needs: the
    kernel function, its argument signature, the compile-time values and the
    compilation options.
    """
    manifest = tomllib.loads((package / "package.toml").read_text())
    fields = parse_key(key, manifest["kernel"]["descriptors"]["dimensions"])
    described = spec(fields, manifest.get("parameters", {}), target)
    compiled = triton.compile(
        ASTSource(
            described["kernel"],
            described["signature"],
            described["constexprs"],
        ),
        target=GPUTarget("cuda", capability(target), WARP_SIZE),
        options=described["options"],
    )

    out.mkdir(parents=True, exist_ok=True)
    (out / "module.cubin").write_bytes(compiled.asm["cubin"])
    metadata = next(
        path for name, path in compiled.metadata_group.items() if name.endswith(".json")
    )
    (out / "metadata.json").write_text(Path(metadata).read_text())
    print(json.dumps({"descriptor": key, "target": target, "symbol": compiled.name}))


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
