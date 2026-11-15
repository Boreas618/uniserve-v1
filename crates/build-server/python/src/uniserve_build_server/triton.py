"""The Triton compile driver: a ``spec`` to ``module.cubin`` and its metadata.

The Build Server compiles a Triton package's descriptor by handing its request
to :func:`build` with the package's ``spec``, and keeps what it writes into the
request's output directory: ``module.cubin`` and ``metadata.json``, the metadata
``triton.compile`` itself recorded for the module, with the launch signature
added. Triton's own AOT tool expresses neither clusters nor PDL nor global
scratch (``triton/tools/compile.py``), so the module is compiled through
``triton.compile`` on the request's target, which needs no GPU.

A package's build module states only ``spec(fields, parameters, target)``, what
one descriptor of its kernel needs:

- ``kernel``: the ``@triton.jit`` function;
- ``signature``: its arguments in launch order, each a Triton type or
  ``"constexpr"``;
- ``constexprs``: the compile-time values, by argument name;
- ``options``: ``triton.compile``'s options (``num_warps``, ``launch_pdl``);
- ``aligned`` (optional): the arguments the package states are multiples of
  sixteen, which lets the compiler load and store whole vectors.

``fields`` are the descriptor's fields as the Build Server read them from its
key, ``parameters`` the manifest's, and ``target`` the
:class:`~uniserve_build_server.request.Target` with its hardware facts.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import tomllib
from collections.abc import Callable
from typing import Any

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource

from uniserve_build_server.request import Request, Target

#: What a package's ``spec`` is: the descriptor's fields, the manifest's
#: ``[parameters]`` and the target, to the compilation of one descriptor.
Spec = Callable[[dict[str, str], dict[str, Any], Target], dict[str, Any]]


def capability(target: Target) -> int:
    """The compute capability of ``target``: 100 for ``sm_100a``."""
    return target.capability


def described(
    spec: Spec,
    descriptor: str,
    fields: dict[str, str],
    parameters: dict[str, Any],
    target: Target,
) -> dict[str, Any]:
    """What ``spec`` states for ``descriptor``.

    A dimension the spec needs and the key leaves out is refused by name, as
    the key's own reading would refuse a misspelled one.
    """
    try:
        return spec(fields, parameters, target)
    except KeyError as absent:
        raise SystemExit(
            f"{descriptor}: the descriptor names no {absent}"
        ) from None


def stated_for(
    spec: Spec,
    package: pathlib.Path,
    descriptor: str,
    target: Target,
    uniserve: str,
) -> dict[str, Any]:
    """What ``spec`` states for ``descriptor`` outside a build, for a check.

    The fields are the ones a build is handed, read by the Build Server's own
    reader of the key grammar, ``build-server fields`` of the ``uniserve``
    executable the caller names; the parameters are the package manifest's.

    Raises:
        ValueError: If the package's declared dimensions do not read the key.
    """
    read = subprocess.run(
        [uniserve, "build-server", "fields", str(package), descriptor],
        capture_output=True,
        text=True,
        check=False,
    )
    if read.returncode != 0:
        raise ValueError(f"{descriptor}: {read.stderr.strip()}")
    parameters = tomllib.loads((package / "package.toml").read_text()).get(
        "parameters", {}
    )
    return described(
        spec, descriptor, json.loads(read.stdout), parameters, target
    )


def build(request: Request, spec: Spec) -> None:
    """Compile the descriptor ``request`` names into its output directory."""
    stated = described(
        spec,
        request.descriptor,
        request.fields,
        request.parameters,
        request.target,
    )
    signature = stated["signature"]
    # An aligned argument is attributed by its position in the signature,
    # which is the order of the kernel's parameters. A module is compiled
    # ahead of its launches, so nothing observes a pointer here: the statement
    # is the package's, about its operator's views.
    attributes = {
        (list(signature).index(name),): [["tt.divisibility", 16]]
        for name in stated.get("aligned", ())
    }
    target = request.target
    compiled = triton.compile(
        ASTSource(
            stated["kernel"], signature, stated["constexprs"], attributes
        ),
        target=GPUTarget(target.backend, target.capability, target.warp_size),
        options=stated["options"],
    )

    out = request.output
    (out / "module.cubin").write_bytes(compiled.asm["cubin"])
    # The compilation's metadata is the JSON member of its group, the record
    # `triton.compile` wrote of `compiled.metadata` and the one `CompiledKernel`
    # reads back: its own serialization keeps the target as an object and the
    # backend's field order, which a re-serialization of the parsed tuple would
    # not.
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
            {
                "descriptor": request.descriptor,
                "target": target.name,
                "symbol": compiled.name,
            }
        )
    )
