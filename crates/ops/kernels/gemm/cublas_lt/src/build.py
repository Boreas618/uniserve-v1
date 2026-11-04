#!/usr/bin/env python3
"""Compile the gemm cuBLASLt module for one descriptor.

The Program Server runs this script in the package's root with the build's
environment (design 5.2):

    UNISERVE_DESCRIPTOR   the descriptor key, e.g.
                          input_bf16_weight_bf16_accumulate_f32_output_bf16_order_nxk_epilogue_none_k_2048
    UNISERVE_TARGET       the architecture, e.g. sm_100a
    UNISERVE_PACKAGE      this package's directory
    UNISERVE_OUT          where the module is written (module.so)

The element types, the weight order and the epilogue are the descriptor's; this
script turns them into the preprocessor definitions the host code reads, so one
module is one point of the operator's descriptor matrix. The library itself is
cuBLASLt, which the module links and the deployment provides.
"""

import os
import pathlib
import shutil
import subprocess
import sys

# The descriptor value's name in `ops` to the CUDA type name in cublasLt.h.
DATA_TYPES = {"bf16": "CUDA_R_16BF", "f16": "CUDA_R_16F", "f32": "CUDA_R_32F"}
# The accumulate type is cuBLASLt's compute type, not a data type.
COMPUTE_TYPES = {
    "f32": "CUBLAS_COMPUTE_32F",
    "bf16": "CUBLAS_COMPUTE_32F_FAST_16BF",
    "f16": "CUBLAS_COMPUTE_32F_FAST_16F",
}
WEIGHT_ORDERS = {"kxn": "0", "nxk": "1"}
EPILOGUES = {"none": "0", "bias": "1"}


def parse(descriptor: str) -> dict[str, str]:
    """The descriptor key's `name_value` pairs, in the order it declares them."""
    # A descriptor key joins `name_value` pairs with underscores, and neither a
    # name nor a value here contains one, so the tokens pair up in order.
    tokens = descriptor.split("_")
    if len(tokens) % 2 != 0:
        raise SystemExit(f"the descriptor {descriptor!r} does not pair names with values")
    return {tokens[index]: tokens[index + 1] for index in range(0, len(tokens), 2)}


def definitions(fields: dict[str, str]) -> list[str]:
    """The `-D` definitions of one descriptor."""
    required = {"input", "weight", "accumulate", "output", "order", "epilogue", "k"}
    missing = required - fields.keys()
    if missing:
        raise SystemExit(f"the descriptor names no {sorted(missing)}")
    unknown = fields.keys() - required
    if unknown:
        raise SystemExit(f"the descriptor names what gemm does not have: {sorted(unknown)}")

    def named(table: dict[str, str], field: str, macro: str) -> str:
        value = fields[field]
        if value not in table:
            raise SystemExit(f"{field} = {value!r} is not one of {sorted(table)}")
        return f"-D{macro}={table[value]}"

    return [
        named(DATA_TYPES, "input", "UNISERVE_INPUT_TYPE"),
        named(DATA_TYPES, "weight", "UNISERVE_WEIGHT_TYPE"),
        named(COMPUTE_TYPES, "accumulate", "UNISERVE_ACCUMULATE_TYPE"),
        named(DATA_TYPES, "output", "UNISERVE_OUTPUT_TYPE"),
        named(WEIGHT_ORDERS, "order", "UNISERVE_WEIGHT_ORDER"),
        named(EPILOGUES, "epilogue", "UNISERVE_EPILOGUE"),
    ]


def main() -> int:
    descriptor = os.environ["UNISERVE_DESCRIPTOR"]
    target = os.environ["UNISERVE_TARGET"]
    package = pathlib.Path(os.environ["UNISERVE_PACKAGE"])
    output = pathlib.Path(os.environ["UNISERVE_OUT"]) / "module.so"
    output.parent.mkdir(parents=True, exist_ok=True)

    nvcc = os.environ.get("UNISERVE_NVCC") or shutil.which("nvcc")
    if not nvcc:
        print("no nvcc on this host: the CUDA toolkit is required", file=sys.stderr)
        return 1

    command = [
        nvcc,
        f"-arch={target}",
        "-std=c++20",
        "-O3",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        *definitions(parse(descriptor)),
        "-I",
        str(package / "src"),
        str(package / "src" / "gemm.cpp"),
        "-lcublasLt",
        "-lcudart",
        "-o",
        str(output),
    ]
    print(" ".join(command), flush=True)
    completed = subprocess.run(command, cwd=package, check=False)
    if completed.returncode != 0 or not output.is_file():
        print(f"nvcc failed with {completed.returncode}", file=sys.stderr)
        return completed.returncode or 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
