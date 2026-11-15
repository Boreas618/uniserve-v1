"""How a manifest's table binds a value for one descriptor.

A Triton package's launch table (``[launch]``,
``crates/build-server/src/launch.rs``) and a CUDA package's declared
compilation (``[compile]``, ``crates/build-server/src/compile.rs``) state each
value they need the same way: a literal, a descriptor field's value, the value
a field's value selects, or a manifest parameter, and each type of a launch
argument as a pointer to the element type a field names, a type a field
selects, or a type spelled as a module's launch signature spells one. The
Build Server's manifest reader types and validates those statements; this
module evaluates them against one descriptor's typed fields, for both drivers.

A module's launch signature spells an element type as Triton does
(:data:`ELEMENTS`), which is also what ``ops`` reads back from a cubin
module's metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: The launch signature's spelling of each element type, by the name
#: ``ops::abi`` prints: Triton's, which ``ops::CubinMetadata`` reads.
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


def element(name: str) -> str:
    """The signature's spelling of the element type ``ops::abi`` calls
    ``name``.
    """  # noqa: D205
    try:
        return ELEMENTS[name]
    except KeyError:
        raise SystemExit(
            f"{name!r} is no element type a kernel takes; the element types "
            f"are {sorted(ELEMENTS)}"
        ) from None


def field(fields: Mapping[str, Any], name: str, descriptor: str) -> Any:
    """The value of field ``name``, or a refusal naming the field."""
    try:
        return fields[name]
    except KeyError:
        raise SystemExit(
            f"{descriptor}: the descriptor names no {name}"
        ) from None


def selected(
    stated: Mapping[str, Any], fields: Mapping[str, Any], descriptor: str
) -> Any:
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


def argument_type(
    stated: Any, fields: Mapping[str, Any], descriptor: str
) -> str:
    """The signature type one launch argument's table entry states."""
    if isinstance(stated, str):
        return stated
    if "pointer" in stated:
        return "*" + element(field(fields, stated["pointer"], descriptor))
    if "element" in stated:
        return element(field(fields, stated["element"], descriptor))
    return selected(stated, fields, descriptor)


def constant(
    stated: Any,
    fields: Mapping[str, Any],
    parameters: Mapping[str, Any],
    descriptor: str,
) -> Any:
    """The value one table entry states: a literal, a field's value, the value
    a field selects, or a manifest parameter.

    A field the descriptor does not name takes the entry's ``absent`` value
    when it states one (a variant the operator prints only when it is set).
    """  # noqa: D205
    if not isinstance(stated, dict):
        return stated
    if "parameter" in stated:
        return parameters[stated["parameter"]]
    if "values" in stated:
        return selected(stated, fields, descriptor)
    if stated["field"] not in fields and "absent" in stated:
        return stated["absent"]
    return field(fields, stated["field"], descriptor)
