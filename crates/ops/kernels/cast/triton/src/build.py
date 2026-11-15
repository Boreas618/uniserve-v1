"""The cast package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the
kernel function, its argument signature in the order the operator launches
them, the compile-time values the descriptor dimensions give, and the
compilation options. The Build Server's Triton source drives
``uniserve_build_server.triton`` with it.

The two element types and the number of axes are compiled in; the shape and
the input's strides are words. The steps a conversion takes follow from the
two types alone (``steps``): whether it passes FP32, and whether its last
step narrows a float.
"""

import cast as kernels
import strided
from uniserve_build_server import triton as driver

#: The half-precision element types, which convert only to and from FP32.
HALVES = ("f16", "bf16")

#: The floating element types, by width in bits.
FLOATS = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64}


def steps(source, target):
    """Whether a conversion passes FP32, and whether its last step narrows.

    A half-precision type on either side makes FP32 the middle of the
    conversion unless FP32 is already the other side. The last step narrows
    when it takes a float to a narrower float.
    """
    through = (source in HALVES and target != "f32") or (
        target in HALVES and source != "f32"
    )
    last = "f32" if through else source
    narrows = (
        last in FLOATS and target in FLOATS and FLOATS[target] < FLOATS[last]
    )
    return through, narrows


def spec(fields, parameters, target):
    """The compilation of one cast descriptor."""
    if fields["input"] == fields["output"]:
        raise SystemExit(
            f"a conversion from {fields['input']} to itself is copy_strided's"
        )
    block = int(fields["block"])
    through, narrows = steps(fields["input"], fields["output"])
    pdl = fields["pdl"] == "true" and driver.capability(target) >= 100
    return {
        "kernel": kernels._cast_kernel,
        "signature": {
            "x": strided.pointer(fields["input"]),
            "out": strided.pointer(fields["output"]),
            "elements": "i64",
            **strided.extents("d"),
            **strided.strides("s"),
            "pdl": "i32",
            "THROUGH": "constexpr",
            "NARROWS": "constexpr",
            "RANK": "constexpr",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "THROUGH": through,
            "NARROWS": narrows,
            "RANK": int(fields["rank"]),
            "BLOCK": block,
            "PDL": pdl,
        },
        "options": {"num_warps": strided.warps_of(block), "launch_pdl": pdl},
    }
