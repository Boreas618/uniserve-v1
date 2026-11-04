"""The topk_softmax package's compile script: one descriptor -> one cubin.

The descriptor bakes the two values the kernel needs at compile time: the
scores a lane walks (the expert count rounded up to whole warps) and the
butterfly width of the renormalization (the top-k width rounded up to a power of
two). One warp routes one token, so the module declares four warps and no
dynamic shared memory.

The Program Server runs this script with the package, descriptor, target and
output directory in `UNISERVE_PACKAGE`, `UNISERVE_DESCRIPTOR`, `UNISERVE_TARGET`
and `UNISERVE_OUT`; a direct run passes the same four as flags.
"""

import argparse
import json
import os
import pathlib
import shutil
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
        for name in names:
            for value in dimensions[name]:
                text = f"{name}_{value}"
                if remaining == text or remaining.startswith(text + "_"):
                    fields[name] = value
                    remaining = remaining[len(text) + 1 :]
                    break
            else:
                continue
            break
        else:
            raise SystemExit(f"descriptor {descriptor!r} names no declared dimension at {remaining!r}")
    missing = set(dimensions) - set(fields)
    if missing:
        raise SystemExit(f"descriptor {descriptor!r} is missing {', '.join(sorted(missing))}")
    return fields


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
    compiler = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    command = [
        compiler,
        "-cubin",
        f"-arch={target}",
        "-O3",
        "-std=c++20",
        "--threads=1",
        f"-DUNISERVE_ITERATIONS={iterations}",
        f"-DUNISERVE_TOP={butterfly}",
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
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "descriptor": descriptor,
                "target": target,
                "symbol": metadata["name"],
                "iterations": iterations,
                "butterfly": butterfly,
            }
        )
    )


if __name__ == "__main__":
    main()
