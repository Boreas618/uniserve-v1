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
fields (:mod:`uniserve_build_server.binding` evaluates those statements). This
driver is the one implementation of the rest:

- an element type a field names is spelled as Triton spells it
  (:data:`uniserve_build_server.binding.ELEMENTS`, keyed by the names
  ``ops::abi`` prints);
- the kernel's ``PDL`` parameter is its programmatic dependent launch, true
  when the descriptor's ``pdl`` field is and the target has dependent launch,
  and the module's launch carries the attribute exactly then;
- an aligned integer argument whose value the descriptor's field of the same
  name fixes is attributed as a multiple of sixteen only when that value is.

:func:`stated` gives the compilation a launch table states for one descriptor
without compiling it, and :func:`launches` answers it as JSON for descriptors
outside a build (``uniserve build-server launch``), which a check that launches
a package's kernel from its source reads.
"""

from __future__ import annotations

import functools
import importlib
import importlib.machinery
import importlib.util
import json
import pathlib
from collections.abc import Mapping
from typing import IO, Any

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource

from uniserve_build_server.binding import (
    argument_type,
    constant,
    field,
    selected,
)
from uniserve_build_server.request import Request, Target

#: The kernel parameter that carries the programmatic dependent launch, and
#: the descriptor field that asks for it (``crates/build-server/src/launch.rs``).
DEPENDENT_LAUNCH_PARAMETER = "PDL"
DEPENDENT_LAUNCH_FIELD = "pdl"

#: The divisor an aligned argument is attributed with.
ALIGNMENT = 16


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
        derive = functools.reduce(getattr, launch["derive"].split("."), kernels)
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
        parameter.name
        for candidate in candidates
        for parameter in candidate.params
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


def launches(requests: IO[str], answers: IO[str]) -> None:
    """Answer the descriptors ``uniserve build-server launch`` writes.

    The first line of ``requests`` is what every descriptor of one package
    shares: the manifest's parameters and launch table, the package's source
    directory and its kernel module's name, and the target's name, whose facts
    are read from this package's description (:meth:`Target.named`). The
    kernel module is imported once, from the source directory by its location.
    Each later line is one descriptor, its key and its typed fields as the
    Build Server's reader gives them, and is answered by one JSON line of
    ``answers`` before the next is read: ``{"key": ..., "launch": {...}}`` with
    what :func:`stated` gives (the kernel named rather than held, since a
    caller launches it from its own import of the source, and each aligned
    argument's attributes by its name), or ``{"key": ..., "refused": "..."}``
    with the reason :func:`stated` refuses the descriptor.

    Raises:
        SystemExit: When the source directory holds no such module.
    """
    package = json.loads(requests.readline())
    source, name = package["source"], package["module"]
    spec = importlib.machinery.PathFinder.find_spec(name, [source])
    if spec is None:
        raise SystemExit(f"{source} holds no kernel module {name}")
    kernels = importlib.util.module_from_spec(spec)
    # A launch is stated for the compiled kernel, whose parameters only a
    # compiled `@triton.jit` function holds: the module is decorated in
    # Triton's compiled mode, whichever mode the caller's process runs in (a
    # kernel check on Triton's interpreter starts this program).
    triton.knobs.runtime.interpret = False
    spec.loader.exec_module(kernels)
    target = Target.named(package["target"])

    while line := requests.readline():
        descriptor = json.loads(line)
        key = descriptor["descriptor"]
        try:
            compilation = stated(
                package["launch"],
                kernels,
                key,
                descriptor["fields"],
                package["parameters"],
                target,
            )
        except SystemExit as refusal:
            answer = {"key": key, "refused": str(refusal)}
        else:
            names = list(compilation["signature"])
            answer = {
                "key": key,
                "launch": {
                    "kernel": compilation["kernel"].__name__,
                    "signature": compilation["signature"],
                    "constexprs": compilation["constexprs"],
                    # `triton.compile` keys an attribute by the argument's
                    # position; a description outside the compilation names
                    # the argument instead.
                    "attributes": {
                        names[position]: attributes
                        for (position,), attributes in compilation[
                            "attributes"
                        ].items()
                    },
                    "options": compilation["options"],
                },
            }
        # Compact, as the command prints its own refusals.
        answers.write(json.dumps(answer, separators=(",", ":")) + "\n")
        answers.flush()


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
