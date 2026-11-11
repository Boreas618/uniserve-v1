"""The Triton compile driver every Triton package's ``src/build.py`` calls.

The Build Server's Triton source runs a package's ``build.py`` for one
(descriptor key, target) pair and keeps what it writes into the build's output
directory: ``module.cubin`` and ``metadata.json``, the metadata JSON
``triton.compile`` itself wrote for the module, with the launch signature added.
Triton's own AOT tool expresses neither clusters nor PDL nor global scratch
(``triton/tools/compile.py``), so the module is compiled through
``triton.compile`` on an explicit target, which needs no GPU.

A package's ``build.py`` states only ``spec(fields, parameters, target)``, what
one descriptor of its kernel needs, and ends with ``main(spec)``:

- ``kernel``: the ``@triton.jit`` function;
- ``signature``: its arguments in launch order, each a Triton type or
  ``"constexpr"``;
- ``constexprs``: the compile-time values, by argument name;
- ``options``: ``triton.compile``'s options (``num_warps``, ``launch_pdl``);
- ``aligned`` (optional): the arguments the package states are multiples of
  sixteen, which lets the compiler load and store whole vectors.

The descriptor's fields come from
:func:`uniserve_build_server.package.fields`, so the keys a package builds
are exactly the range its manifest declares.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource

from uniserve_build_server import package

#: Threads in a warp of every target these packages build for.
WARP_SIZE = 32

#: What a package's ``spec`` is: the descriptor's fields, the manifest's
#: ``[parameters]`` and the target, to the compilation of one descriptor.
Spec = Callable[[dict[str, str], dict, str], dict]


def capability(target: str) -> int:
    """The compute capability of an architecture target: ``sm_100a`` is 100."""
    digits = target.removeprefix("sm_")
    while digits and digits[-1].isalpha():
        digits = digits[:-1]
    if not digits.isdigit():
        raise SystemExit(
            f"target {target!r} is not an sm_<capability> architecture"
        )
    return int(digits)


def described(spec: Spec, root: pathlib.Path, key: str, target: str) -> dict:
    """What ``spec`` states for descriptor ``key`` of the package at ``root``.

    A dimension the spec needs and the key leaves out is refused by name, as
    the key's own parse would refuse a misspelled one.
    """
    fields = package.fields()
    try:
        return spec(
            fields, package.manifest(root).get("parameters", {}), target
        )
    except KeyError as missing:
        raise SystemExit(f"{key}: the descriptor names no {missing}") from None


def compile_one(
    root: pathlib.Path, key: str, target: str, out: pathlib.Path, spec: Spec
) -> None:
    """Compile descriptor ``key`` of the package at ``root`` into ``out``."""
    stated = described(spec, root, key, target)
    signature = stated["signature"]
    # An aligned argument is attributed by its position in the signature,
    # which is the order of the kernel's parameters. A module is compiled
    # ahead of its launches, so nothing observes a pointer here: the statement
    # is the package's, about its operator's views.
    attributes = {
        (list(signature).index(name),): [["tt.divisibility", 16]]
        for name in stated.get("aligned", ())
    }
    compiled = triton.compile(
        ASTSource(
            stated["kernel"], signature, stated["constexprs"], attributes
        ),
        target=GPUTarget("cuda", capability(target), WARP_SIZE),
        options=stated["options"],
    )

    out.mkdir(parents=True, exist_ok=True)
    (out / "module.cubin").write_bytes(compiled.asm["cubin"])
    metadata = next(
        path
        for name, path in compiled.metadata_group.items()
        if name.endswith(".json")
    )
    # The module states the parameters it takes at launch, in order and with
    # their types, so `ops` can compare the operator's packed words against
    # them and refuse a disagreement by name instead of leaving it to the
    # driver. Triton drops a kernel's constexprs from its parameters, and so
    # does this list: the signature the source was compiled from marks them
    # `"constexpr"`.
    written = json.loads(pathlib.Path(metadata).read_text())
    written["signature"] = {
        name: kind for name, kind in signature.items() if kind != "constexpr"
    }
    (out / "metadata.json").write_text(json.dumps(written))
    print(
        json.dumps(
            {"descriptor": key, "target": target, "symbol": compiled.name}
        )
    )


def main(spec: Spec) -> None:
    """Compile the descriptor the build names into its output directory."""
    root, key, target, out = package.arguments()
    compile_one(root, key, target, out, spec)
