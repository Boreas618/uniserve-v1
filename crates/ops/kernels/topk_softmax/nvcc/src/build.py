"""The topk_softmax package's compile script: one descriptor -> one cubin.

The descriptor bakes the two values the kernel needs at compile time: the
scores a lane walks (the expert count rounded up to whole warps) and the
butterfly width of the renormalization (the top-k width rounded up to a power of
two). One warp routes one token, so the module declares four warps and no
dynamic shared memory.

The Build Server runs this script with the package, descriptor, target and
output directory in `UNISERVE_PACKAGE`, `UNISERVE_DESCRIPTOR`, `UNISERVE_TARGET`
and `UNISERVE_OUT`; a direct run passes the same four as flags.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys

#: The warps one block holds, which the operator's grid counts tokens by.
WARPS_PER_BLOCK = 4

#: The storage kinds the kernel reads, keyed by descriptor value.
KINDS = {
    "f32": 0,
    "bf16": 1,
    "f16": 2,
    "fp32": 0,
    "fp16": 2,
    "none": -1,
}

#: The element type each storage kind reads, spelled as a module's signature spells one.
POINTERS = {"f32": "fp32", "fp32": "fp32", "bf16": "bf16", "f16": "fp16", "fp16": "fp16"}


def arguments():
    """The package, descriptor, target and output directory of one build."""
    reader = argparse.ArgumentParser(description=__doc__)
    reader.add_argument("--package")
    reader.add_argument("--descriptor")
    reader.add_argument("--target")
    reader.add_argument("--out")
    known = reader.parse_args()
    values = {
        "package": known.package or os.environ.get("UNISERVE_PACKAGE"),
        "descriptor": known.descriptor or os.environ.get("UNISERVE_DESCRIPTOR"),
        "target": known.target or os.environ.get("UNISERVE_TARGET"),
        "out": known.out or os.environ.get("UNISERVE_OUT"),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise SystemExit(f"missing {', '.join(missing)}")
    return pathlib.Path(values["package"]), values["descriptor"], values["target"], pathlib.Path(values["out"])


def fields_of(package, descriptor):
    """Parse a descriptor key against the package manifest's dimensions.

    A key is `name_value` joined by `_` in any order; the longest matching name
    wins, so a value may itself contain an underscore.
    """
    import tomllib

    manifest = tomllib.loads((package / "package.toml").read_text())
    dimensions = manifest["kernel"]["descriptors"]["dimensions"]
    names = sorted(dimensions, key=len, reverse=True)
    fields = {}
    remaining = descriptor
    while remaining:
        name = next((one for one in names if remaining.startswith(one + "_")), None)
        if name is None:
            raise SystemExit(f"{descriptor}: no dimension starts {remaining!r}")
        rest = remaining[len(name) + 1 :]
        value = declared_value(dimensions[name], rest)
        if value is None:
            raise SystemExit(
                f"{descriptor}: {name} admits no value at {rest!r}"
            )
        fields[name] = value
        remaining = rest[len(value) :].removeprefix("_")
    missing = sorted(set(dimensions) - set(fields))
    if missing:
        raise SystemExit(f"{descriptor}: the descriptor names no {missing}")
    return fields


def declared_value(declared, rest):
    """The value a declared dimension reads at the start of ``rest``, or None.

    An enumeration reads the longest of its values that ``rest`` starts with
    whole. A constraint table (``{ multiple = 16 }``, ``{ bits = 32 }``,
    ``{ separator = "x" }``) reads the text up to the next ``_``, which no
    integer, extent tuple or float bit pattern holds, and admits it when it
    meets the constraint, as the Build Server's manifest does.
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


def admits(constraint, value):
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


def capability(target):
    """The SM number a target names (`sm_100a` -> 100)."""
    digits = "".join(character for character in target.split("_", 1)[1] if character.isdigit())
    return int(digits[:3])


def main():
    """Compile one descriptor into `module.cubin` and its metadata."""
    package, descriptor, target, out = arguments()
    fields = fields_of(package, descriptor)
    experts = int(fields["experts"])
    top = int(fields["top"])
    iterations = max(1, -(-experts // 32))
    butterfly = 1 << max(0, (top - 1).bit_length())
    if fields["renormalize"] not in ("true", "false"):
        raise SystemExit(f"renormalize {fields['renormalize']!r} is not a flag")
    if fields["scores"] not in KINDS:
        raise SystemExit(f"scores {fields['scores']!r} has no storage kind")
    if fields["scale"] not in KINDS:
        raise SystemExit(f"scale {fields['scale']!r} has no storage kind")

    out.mkdir(parents=True, exist_ok=True)
    # The nvcc the Build Server checked against this package's declared version.
    compiler = os.environ["UNISERVE_NVCC"]
    command = [
        compiler,
        "-cubin",
        f"-arch={target}",
        "-O3",
        "-std=c++20",
        "--threads=1",
        f"-DUNISERVE_ITERATIONS={iterations}",
        f"-DUNISERVE_TOP={butterfly}",
        # The descriptor's own numbers, so the kernel can refuse a call that
        # arrives with a different expert count or a wider top-k: `experts` and
        # `top` are launch arguments, and a cubin has no error channel, so the
        # kernel traps instead of reducing over part of the vocabulary.
        f"-DUNISERVE_EXPERTS={experts}",
        f"-DUNISERVE_MAX_TOP={top}",
        "-o",
        str(out / "module.cubin"),
        str(package / "src" / "topk_softmax.cu"),
    ]
    print(" ".join(command), file=sys.stderr)
    subprocess.run(command, check=True)

    # The kernel routes one token per warp of a four-warp block in registers, so
    # its metadata is the fixed shape the operator's launch checks.
    metadata = {
        "name": "uniserve_topk_softmax",
        "num_warps": WARPS_PER_BLOCK,
        "num_ctas": 1,
        "shared": 0,
        "launch_pdl": False,
        "launch_cooperative_grid": False,
        "global_scratch_size": 0,
        "global_scratch_align": 1,
        # The parameters the kernel takes at launch, in order and with their types, so `ops` can
        # compare the operator's packed words against them: this cubin's list is hand-written on both
        # sides and one word of it is optional, which is the kind that drifts. The last two are the
        # scale, a pointer even when the call has none (the operator passes a null one), and its
        # storage kind.
        "signature": {
            "scores": "*" + POINTERS[fields["scores"]],
            "score_kind": "i32",
            "row_stride": "i64",
            "rows": "i32",
            "experts": "i32",
            "top": "i32",
            "renormalize": "i32",
            # A pointer even when the call has no scale: the operator passes a null one, and a
            # pointer slot filled with a scalar is exactly what `ops` refuses by name.
            "scale": "*" + POINTERS.get(fields["scale"], "fp32"),
            "scale_kind": "i32",
            "ids": "*i32",
            "weights": "*fp32",
            # The logical-to-physical expert map, a pointer the operator passes null without one.
            "map": "*i32",
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # The constants this cubin was built with, beside it: a check compares them
    # with its own flags and refuses a module built for another descriptor
    # before it allocates anything, which is the only way to observe the guard
    # in the kernel without a device. Kept out of `metadata.json`, whose fields
    # the loader reads as the launch's own.
    (out / "baked.json").write_text(
        json.dumps(
            {
                "experts": experts,
                "top": top,
                "iterations": iterations,
                "butterfly": butterfly,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "descriptor": descriptor,
                "target": target,
                "symbol": metadata["name"],
                "iterations": iterations,
                "butterfly": butterfly,
                "experts": experts,
                "max_top": top,
            }
        )
    )


if __name__ == "__main__":
    main()
