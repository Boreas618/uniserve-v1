"""The Triton compile driver: a launch table to ``module.cubin`` and metadata.

The Build Server compiles a Triton package's descriptor by handing its request
to :func:`build`, and keeps what it writes into the request's output directory:
``module.cubin`` and ``metadata.json``, the metadata ``triton.compile`` itself
recorded for the module, with the launch signature added. Triton's own AOT
tool expresses neither clusters nor PDL nor global scratch
(``triton/tools/compile.py``), so the module is compiled through
``triton.compile`` on the request's target, which needs no GPU.

A Triton package writes no build code. Its kernel is a ``@triton.jit``
function of its source module ``src/<op>.py``, whose parameter list states the
names and the order of its arguments, and its manifest's launch table
(``[launch]``, ``crates/build-server/src/launch.rs``) states what the list does
not: each launch argument's type, each compile-time parameter's value, the warp
count and the aligned arguments, each a literal or read from the descriptor's
fields. This driver is the one implementation of the rest:

- an element type a field names is spelled as Triton spells it
  (:data:`ELEMENTS`, keyed by the names ``ops::abi`` prints);
- the kernel's ``PDL`` parameter is its programmatic dependent launch, true
  when the descriptor's ``pdl`` field is and the target has dependent launch,
  and the module's launch carries the attribute exactly then;
- an aligned integer argument whose value the descriptor's field of the same
  name fixes is attributed as a multiple of sixteen only when that value is.

:func:`stated` gives the compilation a launch table states for one descriptor
without compiling it, and :func:`stated_for` reads a package's table and a
descriptor's fields for a check that compares a kernel's arguments.
"""

from __future__ import annotations

import functools
import importlib
import json
import pathlib
import subprocess
import tomllib
from collections.abc import Mapping
from typing import Any

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource

from uniserve_build_server.request import Request, Target

#: Triton's spelling of each element type, by the name ``ops::abi`` prints.
ELEMENTS = {
    "u8": "u8",
    "u32": "u32",
    "i16": "i16",
    "i32": "i32",
    "i64": "i64",
    "f16": "fp16",
    "bf16": "bf16",
    "f32": "fp32",
    "f64": "fp64",
    "f8e4m3": "fp8e4nv",
}

#: The kernel parameter that carries the programmatic dependent launch, and
#: the descriptor field that asks for it (``crates/build-server/src/launch.rs``).
DEPENDENT_LAUNCH_PARAMETER = "PDL"
DEPENDENT_LAUNCH_FIELD = "pdl"

#: The divisor an aligned argument is attributed with.
ALIGNMENT = 16


def element(name: str) -> str:
    """Triton's spelling of the element type ``ops::abi`` spells ``name``."""
    try:
        return ELEMENTS[name]
    except KeyError:
        raise SystemExit(
            f"{name!r} is no element type a Triton kernel takes; the element "
            f"types are {sorted(ELEMENTS)}"
        ) from None


def field(fields: Mapping[str, Any], name: str, descriptor: str) -> Any:
    """The value of field ``name``, or a refusal naming the field."""
    try:
        return fields[name]
    except KeyError:
        raise SystemExit(f"{descriptor}: the descriptor names no {name}") from None


def argument_type(
    stated: Any, fields: Mapping[str, Any], descriptor: str
) -> str:
    """The Triton type one launch argument's table entry states."""
    if isinstance(stated, str):
        return stated
    if "pointer" in stated:
        return "*" + element(field(fields, stated["pointer"], descriptor))
    if "element" in stated:
        return element(field(fields, stated["element"], descriptor))
    return selected(stated, fields, descriptor)


def selected(stated: Mapping[str, Any], fields: Mapping[str, Any], descriptor: str) -> Any:
    """The value a field's value selects from a table entry's ``values``.

    A key of a TOML table is text, so the field's value is looked up as the
    text a descriptor key prints for it.
    """
    value = field(fields, stated["field"], descriptor)
    printed = str(value).lower() if isinstance(value, bool) else str(value)
    try:
        return stated["values"][printed]
    except KeyError:
        raise SystemExit(
            f"{descriptor}: {stated['field']} = {printed} selects nothing of "
            f"{sorted(stated['values'])}"
        ) from None


def constant(
    stated: Any,
    fields: Mapping[str, Any],
    parameters: Mapping[str, Any],
    descriptor: str,
) -> Any:
    """The value one compile-time parameter's table entry states."""
    if not isinstance(stated, dict):
        return stated
    if "parameter" in stated:
        return parameters[stated["parameter"]]
    if "values" in stated:
        return selected(stated, fields, descriptor)
    if stated["field"] not in fields and "absent" in stated:
        return stated["absent"]
    return field(fields, stated["field"], descriptor)


def warps(stated: Any, fields: Mapping[str, Any], descriptor: str) -> int:
    """The warp count the table states for one descriptor."""
    if isinstance(stated, int):
        return stated
    value = field(fields, stated["field"], descriptor)
    if "steps" not in stated:
        return value
    for threshold, count in stated["steps"]:
        if value >= threshold:
            return count
    return stated["otherwise"]


def stated(
    launch: Mapping[str, Any],
    kernels: Any,
    descriptor: str,
    fields: Mapping[str, Any],
    parameters: Mapping[str, Any],
    target: Target,
) -> dict[str, Any]:
    """The compilation ``launch`` states for one descriptor of a package.

    ``kernels`` is the package's source module and ``fields`` the descriptor's
    typed fields. Returns the kernel, its signature in parameter order (each a
    Triton type, or ``"constexpr"``), the compile-time values, the attributes
    of the aligned arguments by position, and ``triton.compile``'s options.

    Raises:
        SystemExit: When the table leaves a parameter of the kernel unbound,
            binds a name the kernel lacks, or reads a field the descriptor
            does not name.
    """
    fields = dict(fields)
    if launch.get("derive"):
        # A derivation is named as the source module reaches it: one of its
        # own functions, or a function of a module it imports, such as the
        # kernel library's (`strided.addressed`).
        derive = functools.reduce(
            getattr, launch["derive"].split("."), kernels
        )
        fields.update(derive(fields, parameters, target))

    # A table that selects among several kernels states the types and
    # constants of all of them: each binds what its kernel's parameters name.
    stated_kernel = launch["kernel"]
    if isinstance(stated_kernel, str):
        kernel = getattr(kernels, stated_kernel)
        candidates = [kernel]
    else:
        kernel = getattr(kernels, selected(stated_kernel, fields, descriptor))
        candidates = [
            getattr(kernels, name) for name in stated_kernel["values"].values()
        ]

    types = launch.get("types", {})
    constants = launch.get("constants", {})
    # A module launches dependently only when its kernel runs the prologue
    # that waits for the grid before it: a kernel without a `PDL` parameter
    # reads its inputs at once, so its launch never carries the attribute.
    dependent = (
        any(
            parameter.name == DEPENDENT_LAUNCH_PARAMETER
            for parameter in kernel.params
        )
        and bool(fields.get(DEPENDENT_LAUNCH_FIELD, False))
        and target.dependent_launch
    )
    signature: dict[str, str] = {}
    constexprs: dict[str, Any] = {}
    for parameter in kernel.params:
        name = parameter.name
        if parameter.is_constexpr:
            signature[name] = "constexpr"
            if name == DEPENDENT_LAUNCH_PARAMETER:
                constexprs[name] = dependent
            elif name in constants:
                constexprs[name] = constant(
                    constants[name], fields, parameters, descriptor
                )
            else:
                raise SystemExit(
                    f"{descriptor}: the launch table gives the compile-time "
                    f"parameter {name} no value"
                )
        elif name in types:
            signature[name] = argument_type(types[name], fields, descriptor)
        else:
            raise SystemExit(
                f"{descriptor}: the launch table gives the argument {name} "
                f"no type"
            )
    taken = {
        parameter.name for candidate in candidates for parameter in candidate.params
    }
    unbound = (set(types) | set(constants)) - taken
    if unbound:
        raise SystemExit(
            f"{descriptor}: the launch table binds {sorted(unbound)}, which "
            f"no kernel it compiles takes"
        )

    names = list(signature)
    attributes = {}
    aligned = launch.get("aligned", [])
    if isinstance(aligned, dict):
        aligned = selected(aligned, fields, descriptor)
    for name in aligned:
        if name not in signature:
            raise SystemExit(
                f"{descriptor}: the aligned argument {name} is no argument of "
                f"the kernel"
            )
        # An integer argument the descriptor fixes is a multiple of sixteen
        # only when the descriptor's value is; a pointer is the operator's
        # statement about its views.
        if name in fields and not signature[name].startswith("*"):
            if fields[name] % ALIGNMENT != 0:
                continue
        attributes[(names.index(name),)] = [["tt.divisibility", ALIGNMENT]]

    return {
        "kernel": kernel,
        "signature": signature,
        "constexprs": constexprs,
        "attributes": attributes,
        "options": {
            "num_warps": warps(launch["warps"], fields, descriptor),
            **{
                name: constant(value, fields, parameters, descriptor)
                for name, value in launch.get("options", {}).items()
            },
            "launch_pdl": dependent,
        },
    }


def stated_for(
    kernels: Any,
    package: pathlib.Path,
    descriptor: str,
    target: Target,
    uniserve: str,
) -> dict[str, Any]:
    """What a package's launch table states for ``descriptor``, outside a build.

    ``kernels`` is the package's source module as the caller imported it. The
    fields are the ones a build is handed, read by the Build Server's own
    reader of the key grammar, ``build-server fields`` of the ``uniserve``
    executable the caller names; the table and the parameters are the
    package manifest's.

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
    manifest = tomllib.loads((package / "package.toml").read_text())
    return stated(
        manifest["launch"],
        kernels,
        descriptor,
        json.loads(read.stdout),
        manifest.get("parameters", {}),
        target,
    )


def build(request: Request) -> None:
    """Compile the descriptor ``request`` names into its output directory."""
    if request.launch is None:
        raise SystemExit("a Triton package states its launch table in [launch]")
    compilation = stated(
        request.launch,
        importlib.import_module(request.module),
        request.descriptor,
        request.fields,
        request.parameters,
        request.target,
    )
    signature = compilation["signature"]
    target = request.target
    compiled = triton.compile(
        ASTSource(
            compilation["kernel"],
            signature,
            compilation["constexprs"],
            compilation["attributes"],
        ),
        target=GPUTarget(target.backend, target.capability, target.warp_size),
        options=compilation["options"],
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
    # does this list.
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
