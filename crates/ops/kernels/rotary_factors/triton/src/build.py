"""The rotary_factors package's compile script: one descriptor -> one module.

A descriptor is the factor dtype, the position dtype, the number of frequencies
per row, the program's block, the scale's bit pattern and the dependent-launch
flag. The position count stays a runtime value and the row width comes from the
launch, so one module serves every token count.
"""

import struct

import _compile
import rotary_factors as kernels

#: Position dtypes: token offsets arrive as int32 or int64.
POSITIONS = {"i32": "i32", "i64": "i64"}

#: Factor dtypes: FP32 keeps the FP32 phase, BF16 and FP16 halve the table.
FACTORS = {"fp32": "fp32", "f32": "fp32", "bf16": "bf16", "fp16": "fp16"}


def float_of(bits):
    """The float a descriptor's hexadecimal bit pattern names."""
    return struct.unpack("<f", struct.pack("<I", int(bits, 16)))[0]


def spec(fields, parameters, target):
    """The compilation of one rotary_factors descriptor."""
    factor = "*" + FACTORS[fields["dtype"]]
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._rotary_factors_kernel,
        "signature": {
            "positions": "*" + POSITIONS[fields["positions"]],
            "frequencies": "*fp32",
            "cosine": factor,
            "sine": factor,
            "total": "i64",
            "columns": "i32",
            "scale": "fp32",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "BLOCK": int(fields["block"]),
            "PDL": pdl,
        },
        "options": {"num_warps": 4, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
