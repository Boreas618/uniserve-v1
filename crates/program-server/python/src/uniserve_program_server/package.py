"""What every package compile script reads.

Its build's arguments, the descriptor's fields, the checked nvcc, and the
entries a native module must export.

The Program Server runs a package's ``src/build.py`` with its working directory
the package root and ``UNISERVE_PACKAGE``, ``UNISERVE_DESCRIPTOR``,
``UNISERVE_TARGET`` and ``UNISERVE_OUT`` in the environment (``UNISERVE_NVCC``
too for a CUDA package); a direct run states the same four as ``--package``,
``--descriptor``, ``--target`` and ``--out``.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import tomllib


def arguments() -> tuple[pathlib.Path, str, str, pathlib.Path]:
    """The package root, descriptor key, target and output of one build."""
    reader = argparse.ArgumentParser()
    reader.add_argument("--package")
    reader.add_argument("--descriptor")
    reader.add_argument("--target")
    reader.add_argument("--out")
    known = reader.parse_args()
    values = {
        "package": known.package or os.environ.get("UNISERVE_PACKAGE"),
        "descriptor": (
            known.descriptor or os.environ.get("UNISERVE_DESCRIPTOR")
        ),
        "target": known.target or os.environ.get("UNISERVE_TARGET"),
        "out": known.out or os.environ.get("UNISERVE_OUT"),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise SystemExit(f"missing {', '.join(missing)}")
    return (
        pathlib.Path(values["package"]).resolve(),
        values["descriptor"],
        values["target"],
        pathlib.Path(values["out"]),
    )


def manifest(package: pathlib.Path) -> dict:
    """The package's ``package.toml``."""
    return tomllib.loads((package / "package.toml").read_text())


def fields(package: pathlib.Path, key: str) -> dict[str, str]:
    """A descriptor key read into ``{dimension: value}``.

    The dimensions are the manifest's ``[kernel.descriptors]``: one table, or
    several ``[[kernel.descriptors]]`` shapes when an operator prints some
    fields only for the variant that has them (a narrowed rotation names its
    width, a sectioned one its axes). The key is read against each shape in
    turn and the first that reads it whole answers.

    A key is ``field_value`` pairs joined with ``_``, in whatever order the
    operator prints them. Names and declared values may contain ``_``, so a
    field is the longest declared name the rest starts with and a value the
    longest declared value that follows it whole, or, for a dimension declared
    by a constraint table, the text up to the next ``_`` when it meets the
    constraint.

    A declared dimension the key leaves out is not refused here, which is the
    rule the Program Server's coverage (``Descriptors::covers``) applies: the
    package's compilation answers for a field it needs and the key lacks.
    """
    declared = manifest(package)["kernel"]["descriptors"]
    shapes = declared if isinstance(declared, list) else [declared]
    refusals = []
    for shape in shapes:
        dimensions = shape.get("dimensions") or {}
        if not dimensions:
            continue
        try:
            return read_key(key, dimensions)
        except DescriptorKeyError as refusal:
            refusals.append(str(refusal))
    raise SystemExit(
        f"{key}: no declared descriptor shape reads it: " + "; ".join(refusals)
    )


class DescriptorKeyError(ValueError):
    """A key one descriptor shape does not read."""


def read_key(key: str, dimensions: dict) -> dict[str, str]:
    """``key`` read against one shape's ``dimensions``.

    Raises :class:`DescriptorKeyError` when the shape does not read it.
    """
    names = sorted(dimensions, key=len, reverse=True)
    found: dict[str, str] = {}
    remaining = key
    while remaining:
        name = next(
            (one for one in names if remaining.startswith(one + "_")), None
        )
        if name is None:
            raise DescriptorKeyError(
                f"no dimension among {names} starts {remaining!r}"
            )
        rest = remaining[len(name) + 1 :]
        value = declared_value(dimensions[name], rest)
        if value is None:
            raise DescriptorKeyError(f"{name} admits no value at {rest!r}")
        found[name] = value
        remaining = rest[len(value) :].removeprefix("_")
    return found


def declared_value(declared, rest: str) -> str | None:
    """The value a declared dimension reads at the start of ``rest``, or None.

    An enumeration reads the longest of its values that ``rest`` starts with
    whole. A constraint table (``{ multiple = 16 }``, ``{ bits = 32 }``,
    ``{ separator = "x" }``) reads the text up to the next ``_``, which no
    integer, extent tuple or float bit pattern holds, and admits it when it
    meets the constraint, as the Program Server's manifest does.
    """
    if isinstance(declared, list):
        return next(
            (
                one
                for one in sorted(declared, key=len, reverse=True)
                if rest == one or rest.startswith(one + "_")
            ),
            None,
        )
    value = rest.split("_", 1)[0]
    return value if admits(declared, value) else None


def admits(constraint: dict, value: str) -> bool:
    """Whether ``value``, as a key spells it, meets a constraint table."""
    if "bits" in constraint:
        # A float enters a key as its bit pattern: `0x`, then lowercase
        # hexadecimal without leading zeros.
        digits = value.removeprefix("0x")
        return (
            value.startswith("0x")
            and 0 < len(digits) <= constraint["bits"] // 4
            and all(digit in "0123456789abcdef" for digit in digits)
            and (digits == "0" or not digits.startswith("0"))
        )
    separator = constraint.get("separator")
    for part in value.split(separator) if separator else [value]:
        # An integer enters a key in canonical decimal: `064` names nothing.
        if not part or any(digit not in "0123456789" for digit in part):
            return False
        if part != "0" and part.startswith("0"):
            return False
        number = int(part)
        if number < constraint.get("min", 1):
            return False
        if number > constraint.get("max", number):
            return False
        if number % constraint.get("multiple", 1) != 0:
            return False
        power = number > 0 and number & (number - 1) == 0
        if constraint.get("power_of_two") and not power:
            return False
    return True


def nvcc() -> tuple[str, str]:
    """The nvcc the Program Server checked, and the version line it prints.

    The Program Server compared it with the package's declared version.
    A compiler that does not answer ``--version`` with CUDA's own line is
    refused here rather than producing a module from the host compiler.
    """
    compiler = os.environ["UNISERVE_NVCC"]
    answered = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True
    )
    if (
        answered.returncode != 0
        or "Cuda compilation tools" not in answered.stdout
    ):
        raise SystemExit(
            f"{compiler} is not CUDA's nvcc:\n"
            f"{answered.stdout[-300:]}{answered.stderr[-300:]}"
        )
    return compiler, answered.stdout.strip().splitlines()[-1].strip()


def require_exports(module: pathlib.Path, entries: list[str]) -> None:
    """Refuses a native module that does not export every one of ``entries``.

    The main process resolves them by name after loading it.
    """
    listed = subprocess.run(
        ["nm", "-D", "--defined-only", str(module)],
        capture_output=True,
        text=True,
        check=True,
    )
    symbols = {
        line.split()[-1] for line in listed.stdout.splitlines() if line.split()
    }
    missing = [entry for entry in entries if entry not in symbols]
    if missing:
        raise SystemExit(f"{module.name} does not export {', '.join(missing)}")
