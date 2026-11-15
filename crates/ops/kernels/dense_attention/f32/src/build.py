"""The dense_attention f32 backend's build module: one descriptor -> a module.

Two things this backend compiles in. The mask, so a causal module refuses a
dense call and the other way round and the masking kernel carries no branch on
it. And the sequence layout, so a module built for the launch that carries
`[batch + 1]` offsets never branches on a missing table and the no-offsets
module never reads a pointer it was not given — `cu_seq_lens_false` is the only
form `ops` prints that field in, so a key without it is the scheduled batch
path. Everything else the descriptor names (heads, head width) arrives in the
call and is checked there, and the history window is a bound the call carries,
so one module serves a windowed and an unwindowed call of the same descriptor.
A launch's extents — the packed query and key rows and the descriptor's length
cap — are not compiled in at all: `ops` keys one program per geometry and hands
the extents over.

The manifest admits FP32 alone, so the element types compile to nothing; a
descriptor whose heads do not divide is refused here by name rather than
compiled into a module that would refuse every launch.

The Build Server compiles a descriptor by calling :func:`compile` with the
compile request.
"""

import subprocess
import sys

from uniserve_build_server.request import Request

#: The translation unit of the module.
SOURCES = ["src/dense.cu"]


def compile(request: Request) -> None:
    """Compile the module for the request's descriptor into ``module.so``.

    The module exports the launch entry the manifest resolves; a compiler
    failure refuses the build, with the compiler's output in the build log.
    """
    package = request.package
    if request.field("q_heads") % request.field("kv_heads") != 0:
        raise SystemExit(
            f"{request.field('q_heads')} query heads do not divide over "
            f"{request.field('kv_heads')} key heads"
        )
    causal = request.field("mask") == "causal"
    # A key that names `cu_seq_lens` (always `false`) attends one sequence; a
    # key without it reads the call's offset tables.
    sequences = 0 if "cu_seq_lens" in request.fields else 1

    command = [
        request.toolchain.require_cuda().nvcc,
        "-O3",
        "-std=c++17",
        f"-arch={request.target}",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        f"-DUNISERVE_ENTRY={request.entries['launch']}",
        f"-DUNISERVE_CAUSAL={int(causal)}",
        f"-DUNISERVE_SEQUENCES={sequences}",
        "-I",
        str(package / "include"),
        *[str(package / source) for source in SOURCES],
        "-lcublasLt",
        "-lcudart",
        "-o",
        str(request.output / "module.so"),
    ]
    print(" ".join(command), file=sys.stderr)
    built = subprocess.run(command, check=False)
    if built.returncode != 0:
        raise SystemExit(f"nvcc failed with {built.returncode}")
