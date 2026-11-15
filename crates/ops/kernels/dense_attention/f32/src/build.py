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

The export name is read from the package's manifest, so the symbol the module
exports and the symbol the package declares cannot disagree.

A descriptor this backend does not implement — an element type other than
FP32, or heads that do not divide — is refused here by name rather than
compiled into a module that would refuse every launch.

The Build Server compiles a descriptor by calling :func:`compile` with the
compile request.
"""

import json
import subprocess
import sys
import tomllib

from uniserve_build_server.request import Request

#: The translation unit of the module.
SOURCES = ["src/dense.cu"]

#: The operator crate's package ABI numbering (`ops::abi::code`).
ELEMENT_TYPES = {
    "u8": 0,
    "i32": 1,
    "i64": 2,
    "f16": 3,
    "bf16": 4,
    "f32": 5,
    "f8e4m3": 6,
    "f4e2m1": 7,
}

#: The operator's masks, as `Mask`'s `Display` prints them, to the compiled
#: flag.
MASKS = {"causal": 1, "dense": 0}

#: The fields every descriptor of this operator carries.
FIELDS = (
    "dtype",
    "kv",
    "out",
    "head_dim",
    "q_heads",
    "kv_heads",
    "mask",
    "window",
)

#: The field `ops` prints only for the form without sequence offsets, and the
#: flag it compiles to: 1 reads the call's tables, 0 attends one sequence.
SEQUENCES = {"false": 0}


def launch_entry(package):
    """The launch symbol the package's manifest declares."""
    manifest = tomllib.loads((package / "package.toml").read_text())
    entry = manifest.get("module", {}).get("entries", {}).get("launch")
    if not entry:
        raise SystemExit(
            "the package's [module.entries] names no launch symbol"
        )
    return entry


def compile(request: Request) -> None:
    """Compile the module for the request's descriptor into ``module.so``.

    The descriptor's fields are checked first, so a descriptor this backend
    does not implement is refused by name; a compiler failure refuses the
    build, with the compiler's output in the build log.
    """
    package = request.package
    entry = launch_entry(package)
    # Every field of `FIELDS` is required; `cu_seq_lens` is present only in
    # the one-sequence form of the key.
    named = {field: request.field(field) for field in FIELDS}
    if "cu_seq_lens" in request.fields:
        named["cu_seq_lens"] = request.fields["cu_seq_lens"]

    for field in ("dtype", "kv", "out"):
        if named[field] != "f32":
            raise SystemExit(
                f"{field} = {named[field]!r}: this backend serves FP32 only"
            )
    if named["mask"] not in MASKS:
        raise SystemExit(
            f"mask = {named['mask']!r} is not one of {sorted(MASKS)}"
        )
    if named["window"] not in {"false", "true"}:
        raise SystemExit(
            f"window = {named['window']!r} is not one of ['false', 'true']"
        )
    sequences = 1
    if "cu_seq_lens" in named:
        if named["cu_seq_lens"] not in SEQUENCES:
            raise SystemExit(
                f"cu_seq_lens = {named['cu_seq_lens']!r} is not one of "
                f"{sorted(SEQUENCES)}"
            )
        sequences = SEQUENCES[named["cu_seq_lens"]]
    counts = {}
    for field in ("head_dim", "q_heads", "kv_heads"):
        try:
            counts[field] = int(named[field])
        except ValueError:
            raise SystemExit(f"{field} = {named[field]!r} is not a number")
        if counts[field] <= 0:
            raise SystemExit(f"{field} = {counts[field]} names no work")
    if counts["q_heads"] % counts["kv_heads"] != 0:
        raise SystemExit(
            f"{counts['q_heads']} query heads do not divide over "
            f"{counts['kv_heads']} key heads"
        )
    # The value of an element type in the descriptor is the operator's
    # numbering; the module compares against it, so it is passed rather than
    # assumed.
    for field in ("dtype", "kv", "out"):
        if named[field] not in ELEMENT_TYPES:
            raise SystemExit(
                f"{field} = {named[field]!r} is not an element type"
            )

    # The nvcc the Build Server checked against this package's declared version.
    compiler = request.toolchain.require_cuda().nvcc
    command = [
        compiler,
        "-O3",
        "-std=c++17",
        f"-arch={request.target}",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        f"-DUNISERVE_ENTRY={entry}",
        f"-DUNISERVE_CAUSAL={MASKS[named['mask']]}",
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

    print(
        json.dumps(
            {
                "descriptor": request.descriptor,
                "target": str(request.target),
                "entry": entry,
                "mask": named["mask"],
                "window": named["window"],
                "cu_seq_lens": sequences == 1,
            }
        )
    )
