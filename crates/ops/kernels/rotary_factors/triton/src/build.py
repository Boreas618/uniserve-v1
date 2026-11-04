"""The rotary_factors package's compile script: one descriptor -> one module.

A descriptor is the factors' element type, the positions' element type, the
frequencies per position, the program's block, the factor every angle is
multiplied by, the dependent-launch flag and — for a multimodal rotation — how
many position axes the factors read, the rotated width each axis owns and
whether the axes interleave. The token count is a runtime word.
"""

import struct

import _compile
import rotary_factors as kernels

#: Element types of the factor tables and of the positions.
DTYPES = {"f32": "fp32", "bf16": "bf16", "f16": "fp16"}
POSITIONS = {"i32": "i32", "i64": "i64"}

#: The most position axes a sectioned rotation reads, as the operator allows.
MAX_AXES = 4


def bits_of(value):
    """The float a descriptor's hexadecimal bit pattern names."""
    return struct.unpack("<f", struct.pack("<I", int(value, 16)))[0]


def widths_of(fields, axes):
    """The rotated elements each axis owns, padded or cut to `axes`."""
    if "sections" not in fields:
        return [0] * axes
    widths = [int(part) for part in fields["sections"].split("-")]
    return (widths + [0] * axes)[:axes]


def spec(fields, parameters, target):
    """The compilation of one rotary_factors descriptor."""
    columns = int(fields["columns"])
    axes = int(fields.get("axes", "1"))
    if axes < 1 or axes > MAX_AXES:
        raise SystemExit(f"{axes} position axes, at most {MAX_AXES}")
    interleaved = fields.get("interleaved") == "true"
    if interleaved and axes < 2:
        raise SystemExit("an interleaved rotation needs at least two axes")
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100

    # Each axis owns `width / 2` frequencies, so the sections must cover the
    # row for a call to be launchable; the operator refuses a call whose
    # sections cover anything else. A descriptor states both fields, and one
    # whose sections do not cover its columns still names a module here: the
    # columns past the last section read the last axis.
    halves = [width // 2 for width in widths_of(fields, axes)]
    # The contiguous cut's thresholds: a column at or past threshold `a` reads
    # axis `a` or later.
    cuts = []
    total = 0
    for half in halves:
        total += half
        cuts.append(total)

    signature = {
        "positions": "*" + POSITIONS[fields["positions"]],
        "frequencies": "*fp32",
        "cosine": "*" + DTYPES[fields["dtype"]],
        "sine": "*" + DTYPES[fields["dtype"]],
        "total": "u64",
        "columns": "u32",
        "scale": "fp32",
        "AXES": "constexpr",
        "HALVES": "constexpr",
        "CUTS": "constexpr",
        "INTERLEAVED": "constexpr",
        "BLOCK": "constexpr",
        "PDL": "constexpr",
    }
    if axes > 1:
        # A sectioned program's arguments carry the axes and the widths the
        # operator always passes; the module compiled its own from the key, so
        # the words are taken for the call's order and not read.
        signature["axes"] = "u32"
        for index in range(MAX_AXES):
            signature[f"section_{index}"] = "u32"

    return {
        "kernel": (
            kernels._rotary_factors_kernel
            if axes == 1
            else kernels._rotary_factors_sectioned_kernel
        ),
        "signature": signature,
        "constexprs": {
            "AXES": axes,
            "HALVES": tuple(halves[:axes]),
            "CUTS": tuple(cuts[:axes]),
            "INTERLEAVED": interleaved,
            "BLOCK": int(fields["block"]),
            "PDL": pdl,
        },
        "options": {"num_warps": 4, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
