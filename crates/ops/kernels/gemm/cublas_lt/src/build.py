"""Compile the gemm cuBLASLt module for one descriptor.

The Build Server compiles a descriptor, such as
`input_bf16_weight_bf16_accumulate_f32_output_bf16_order_nxk_epilogue_none`,
by calling :func:`compile` with the compile request (design 5.2).

The element types, the weight order and the epilogue are the descriptor's; this
module turns them into the preprocessor definitions the host code reads, so one
module is one point of the operator's descriptor matrix. The library itself is
cuBLASLt, which the module links and the deployment provides.
"""

import subprocess

from uniserve_build_server.request import Request

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


def definitions(fields: dict[str, str]) -> list[str]:
    """The `-D` definitions of one descriptor."""
    # No contraction length: cublasLt takes `m`, `n` and `k` with the call, so
    # one module of a given type set and order serves every weight shape.
    required = {"input", "weight", "accumulate", "output", "order", "epilogue"}
    missing = required - fields.keys()
    if missing:
        raise SystemExit(f"the descriptor names no {sorted(missing)}")
    unknown = fields.keys() - required
    if unknown:
        raise SystemExit(
            f"the descriptor names what gemm does not have: {sorted(unknown)}"
        )

    def named(table: dict[str, str], field: str, macro: str) -> str:
        value = fields[field]
        if value not in table:
            raise SystemExit(
                f"{field} = {value!r} is not one of {sorted(table)}"
            )
        return f"-D{macro}={table[value]}"

    return [
        named(DATA_TYPES, "input", "UNISERVE_INPUT_TYPE"),
        named(DATA_TYPES, "weight", "UNISERVE_WEIGHT_TYPE"),
        named(COMPUTE_TYPES, "accumulate", "UNISERVE_ACCUMULATE_TYPE"),
        named(DATA_TYPES, "output", "UNISERVE_OUTPUT_TYPE"),
        named(WEIGHT_ORDERS, "order", "UNISERVE_WEIGHT_ORDER"),
        named(EPILOGUES, "epilogue", "UNISERVE_EPILOGUE"),
    ]


def compile(request: Request) -> None:
    """Compile the module for the request's descriptor into ``module.so``.

    A compiler failure refuses the build, with the compiler's output in the
    build log.
    """
    package = request.package
    output = request.output / "module.so"
    # The nvcc the Build Server checked against this package's declared version.
    nvcc = request.toolchain.require_cuda().nvcc

    command = [
        nvcc,
        f"-arch={request.target}",
        "-std=c++20",
        "-O3",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        *definitions(request.fields),
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
    if completed.returncode != 0:
        raise SystemExit(f"nvcc failed with {completed.returncode}")
