#!/usr/bin/env python3
"""Check that a dense-attention module **refuses** the calls it was not built for — no GPU needed.

    crates/program-server/python/.venv/bin/python refusal_check.py <causal module dir> <dense module dir>

Each module bakes two of its descriptor's fields into itself: the mask (`UNISERVE_CAUSAL`) and the
sequence layout (`UNISERVE_SEQUENCES`). A module built causal and called dense would mask the future
away and return a perfectly plausible answer, and `resolve` normally hands back the right module — so
the danger is a fast path or a runner reusing one across descriptors, which is exactly what a fast
path does. The module answers that with guards at the top of its entry, and this script is what keeps
them honest.

**It runs on a host with no device.** Every guard reads only the call's scalars and the null-ness of
its pointers, and each sits before the first CUDA call, so a refusal is observable with nothing but
the loaded library — which is also why a refusal costs nothing at run time.

The cases are the mismatches the descriptor can express:
- a causal module called dense, and a dense module called causal, each naming the mask it compiles;
- a module that attends one sequence called with sequence offsets, and one called with two sequences;
- a call that leaves one of its buffers null;
- a call whose element types are not FP32;
- and, for contrast, a call that **matches** the module: on a host without a device it gets past
  every guard and fails at the device instead, which the check asserts by name so a guard that
  silently widened would show up as the device message disappearing.
"""

import argparse
import ctypes
import pathlib
import sys

#: `DType::as_i64()`: this backend serves FP32 only.
F32 = 5
F16 = 3


class Call(ctypes.Structure):
    """`attention::dense::DenseCall`, field for field (`uniserve_dense_attention.h`)."""

    _fields_ = [
        ("out", ctypes.c_void_p),
        ("query", ctypes.c_void_p),
        ("key", ctypes.c_void_p),
        ("value", ctypes.c_void_p),
        ("cu_seq_lens_q", ctypes.c_void_p),
        ("cu_seq_lens_k", ctypes.c_void_p),
        ("batch", ctypes.c_int64),
        ("q_tokens", ctypes.c_int64),
        ("kv_tokens", ctypes.c_int64),
        ("max_q_len", ctypes.c_int64),
        ("max_kv_len", ctypes.c_int64),
        ("q_heads", ctypes.c_int64),
        ("kv_heads", ctypes.c_int64),
        ("head_dim", ctypes.c_int64),
        ("q_stride_tokens", ctypes.c_int64),
        ("q_stride_heads", ctypes.c_int64),
        ("kv_stride_tokens", ctypes.c_int64),
        ("kv_stride_heads", ctypes.c_int64),
        ("o_stride_tokens", ctypes.c_int64),
        ("o_stride_heads", ctypes.c_int64),
        ("scale", ctypes.c_double),
        ("window_left", ctypes.c_int64),
        ("causal", ctypes.c_int64),
        ("dtype_query", ctypes.c_int64),
        ("dtype_key_value", ctypes.c_int64),
        ("dtype_output", ctypes.c_int64),
        ("error", ctypes.c_char * 512),
    ]


DENSE_CALL_SIZE = 720
assert ctypes.sizeof(Call) == DENSE_CALL_SIZE, (
    f"the check's Call mirror is {ctypes.sizeof(Call)} bytes; the operator's "
    f"DENSE_ATTENTION_CALL_LAYOUT says {DENSE_CALL_SIZE} (`crates/ops/src/attention/dense.rs`)"
)


def module_entry(directory):
    """The entry a compiled module exports, and the directory's name as the descriptor it carries."""
    path = pathlib.Path(directory)
    if not (path / "module.so").is_file():
        raise SystemExit(f"{path} holds no module.so, so there is nothing to call")
    library = ctypes.CDLL(str(path / "module.so"))
    entry = library.uniserve_dense_attention
    entry.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    entry.restype = ctypes.c_int32
    return entry, path.name


def buffers():
    """Four small host allocations: the guards read their addresses, never their contents."""
    return [ctypes.create_string_buffer(64) for _ in range(6)]


def refuse(entry, *, causal, offsets, batch, dtypes, buffers_, nulls=()):
    """Call `entry` with the given call shape; return (status, message).

    `nulls` names buffer fields to leave null — a call that omits one of its tensors, which the
    entry must refuse rather than dereference.
    """
    out, query, key, value, offsets_q, offsets_k = buffers_
    call = Call(
        out=ctypes.addressof(out),
        query=ctypes.addressof(query),
        key=ctypes.addressof(key),
        value=ctypes.addressof(value),
        cu_seq_lens_q=ctypes.addressof(offsets_q) if offsets else None,
        cu_seq_lens_k=ctypes.addressof(offsets_k) if offsets else None,
        batch=batch,
        q_tokens=64,
        kv_tokens=64,
        max_q_len=64,
        max_kv_len=1024,
        q_heads=32,
        kv_heads=8,
        head_dim=128,
        q_stride_tokens=32 * 128,
        q_stride_heads=128,
        kv_stride_tokens=8 * 128,
        kv_stride_heads=128,
        o_stride_tokens=32 * 128,
        o_stride_heads=128,
        scale=128**-0.5,
        window_left=-1,
        causal=causal,
        dtype_query=dtypes[0],
        dtype_key_value=dtypes[1],
        dtype_output=dtypes[2],
        error=b"",
    )
    for name in nulls:
        setattr(call, name, None)
    status = entry(ctypes.byref(call), None)
    return status, call.error.decode("utf-8", "replace").rstrip("\x00")


def expect(what, status, message, *, must):
    """One refusal, with the words that say which guard answered."""
    if status == 0:
        raise SystemExit(f"{what}: the module accepted a call it was not built for")
    for phrase in must:
        if phrase not in message:
            raise SystemExit(f"{what}: the refusal does not say {phrase!r}: {message!r}")
    return f"{what}: {message}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("causal", type=pathlib.Path, help="a module compiled for the causal mask")
    parser.add_argument("dense", type=pathlib.Path, help="a module compiled for the dense mask")
    options = parser.parse_args()

    causal_entry, causal_key = module_entry(options.causal)
    dense_entry, dense_key = module_entry(options.dense)
    if "mask_causal" not in causal_key or "mask_dense" not in dense_key:
        raise SystemExit(
            f"the two directories must be the causal and the dense cells of one descriptor; they "
            f"are {causal_key!r} and {dense_key!r}"
        )
    held = buffers()
    reported = []

    # The mask, both ways.
    reported.append(expect(
        f"{causal_key} launched dense",
        *refuse(causal_entry, causal=0, offsets=False, batch=1, dtypes=(F32, F32, F32), buffers_=held),
        must=("compiles the causal mask", "causal = 0"),
    ))
    reported.append(expect(
        f"{dense_key} launched causal",
        *refuse(dense_entry, causal=1, offsets=False, batch=1, dtypes=(F32, F32, F32), buffers_=held),
        must=("compiles the dense mask", "causal = 1"),
    ))
    # The sequence layout: these modules attend one sequence and read no offsets.
    for entry, key in ((causal_entry, causal_key), (dense_entry, dense_key)):
        reported.append(expect(
            f"{key} given sequence offsets",
            *refuse(entry, causal=1 if "causal" in key else 0, offsets=True, batch=2,
                    dtypes=(F32, F32, F32), buffers_=held),
            must=("attends one sequence", "offsets"),
        ))
        reported.append(expect(
            f"{key} given two sequences",
            *refuse(entry, causal=1 if "causal" in key else 0, offsets=False, batch=2,
                    dtypes=(F32, F32, F32), buffers_=held),
            must=("attends one sequence", "names 2"),
        ))
    # The element types.
    reported.append(expect(
        f"{causal_key} given FP16 rows",
        *refuse(causal_entry, causal=1, offsets=False, batch=1, dtypes=(F16, F16, F16), buffers_=held),
        must=("serves FP32 activations only",),
    ))
    # A null buffer: the field is left null rather than pointed at a stand-in, so the entry has to
    # look at it. The first draft passed `ctypes.cast(None, c_void_p)` through `addressof`, which
    # produced a non-null address and the case silently measured nothing.
    reported.append(expect(
        f"{causal_key} given a null buffer",
        *refuse(causal_entry, causal=1, offsets=False, batch=1, dtypes=(F32, F32, F32),
                buffers_=held, nulls=("value",)),
        must=("leaves one of its buffers null",),
    ))
    # The matching call gets past every guard: on a host with no device it fails at the device.
    status, message = refuse(causal_entry, causal=1, offsets=False, batch=1,
                             dtypes=(F32, F32, F32), buffers_=held)
    if status == 0:
        reported.append(f"{causal_key} with its own call: ran")
    else:
        guards = ("compiles", "attends one sequence", "FP32 activations", "buffers null")
        if any(phrase in message for phrase in guards):
            raise SystemExit(
                f"a call that matches the module was refused by a guard: {message!r}"
            )
        reported.append(f"{causal_key} with its own call reached the device: {message}")

    for line in reported:
        print(line)
    print(f"ok: {len(reported)} refusals and one matching call, on a module pair with no device")
    return 0


if __name__ == "__main__":
    sys.exit(main())
