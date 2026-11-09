#!/usr/bin/env python3
"""Report every tensor of a checkpoint that holds no nonzero element.

A weight that is all zero does not fail: a projection built from it answers zeros for every
activation, so the model runs and produces a row of zeros that reads like a kernel fault far from the
loading boundary. The packed path refuses this at load already (`crates/nn/src/weights.rs`: "every
packed E4M1 byte is zero"), and this script answers the same question about the *file*, for every
representation, without a device - which is the half a load-time refusal cannot report: whether the
zeros are in the checkpoint or were made by the loader.

    # every tensor of a checkpoint, by name, with its shape and the count of zero elements
    python tools/checkpoint_zeros.py <checkpoint directory>

    # one layer, or one kind of tensor
    python tools/checkpoint_zeros.py <root> --tensors 'model.layers.0.'
    python tools/checkpoint_zeros.py <root> --tensors 'v_proj'

A tensor is reported when every element is zero, and with `--fraction` when more than that share of
it is: a plane that is half zero is a different defect from one that is empty, and the count says
which. The comparison is on the stored bytes, not on a value: a bfloat16 zero is the bit pattern
`0x0000` or `0x8000`, and reading the bytes avoids an element type this script would otherwise have
to know (`safetensors` states it, and an unknown one is skipped by name rather than guessed).

Bfloat16, float16, float32, float8 and integer tensors are all read as fixed-width unsigned words,
one width per dtype, so nothing is converted. A `F4` (packed E2M1) tensor is a byte of two codes and
is scanned as bytes, which answers "every byte zero" - the packed path's own test.

`--values` answers the other question a zero count cannot: **what a tensor holds**. A calibration
scalar is one element and is never zero, so the scan above passes it, while its magnitude is what
decides whether an activation encoded against it survives the quantization at all:

    # every scale of layer 0's attention, as numbers
    python tools/checkpoint_zeros.py <root> --tensors 'layers.0.attention.' --values 8

The decode is the element type's own: a float8 byte is E4M3 or E5M2 by the tensor's dtype, a
bfloat16 is the high half of a float32, and an integer prints as an integer. Nothing is converted
that the file does not state, and a dtype this script cannot decode is skipped by name.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

#: One element's stored width and the words its zero is, per `safetensors` dtype.
#:
#: A float's zero is its sign-magnitude pattern, so both signed zeros count: `-0.0` is zero and a
#: plane of `-0.0` answers zeros for every activation exactly as `+0.0` does. An integer's zero is
#: the word zero. A dtype absent here is skipped by name rather than guessed.
ELEMENTS = {
    "BOOL": (1, (0x00,)),
    "U8": (1, (0x00,)),
    "I8": (1, (0x00,)),
    "F8_E4M3": (1, (0x00, 0x80)),
    "F8_E5M2": (1, (0x00, 0x80)),
    "I16": (2, (0x0000,)),
    "U16": (2, (0x0000,)),
    "F16": (2, (0x0000, 0x8000)),
    "BF16": (2, (0x0000, 0x8000)),
    "I32": (4, (0x00000000,)),
    "U32": (4, (0x00000000,)),
    "F32": (4, (0x00000000, 0x80000000)),
    "I64": (8, (0,)),
    "U64": (8, (0,)),
    "F64": (8, (0, 0x8000000000000000)),
}


def headers(root: pathlib.Path):
    """Every shard's header, by path: `(dtype, shape, begin, end)` per tensor name."""
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"{root}: no .safetensors shards")
    for shard in shards:
        with shard.open("rb") as handle:
            (length,) = np.frombuffer(handle.read(8), dtype="<u8")
            header = json.loads(handle.read(int(length)))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            begin, end = entry["data_offsets"]
            yield shard, name, entry["dtype"], tuple(entry["shape"]), begin, end


def decoded(dtype: str, raw: bytes):
    """The first `len(raw)` elements of `dtype` as numbers, or `None` when this script cannot."""
    array = np.frombuffer(raw, dtype=np.uint8)
    if dtype == "F32":
        return [float(v) for v in np.frombuffer(raw, dtype="<f4")]
    if dtype == "F16":
        return [float(v) for v in np.frombuffer(raw, dtype="<f2")]
    if dtype == "BF16":
        # A bfloat16 is a float32's high half: widening shifts the mantissa into place.
        words = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return [float(v) for v in words.view(np.float32)]
    if dtype in ("F8_E4M3", "F8_E5M2"):
        # E4M3: four exponent bits biased by 7 over three mantissa bits, no infinity and `0x7f`/`0xff`
        # NaN. E5M2: five exponent bits biased by 15 over two mantissa bits, with both, as IEEE does.
        # The subnormal step is the smallest exponent's ULP: `2**(1 - bias - mantissa_bits)`.
        values = []
        for byte in array:
            byte = int(byte)
            sign = -1.0 if byte & 0x80 else 1.0
            if dtype == "F8_E4M3":
                exponent, mantissa, bias, bits = (byte >> 3) & 0xF, byte & 0x7, 7, 3
                if exponent == 0xF and mantissa == 0x7:
                    values.append(float("nan"))
                elif exponent == 0:
                    values.append(sign * mantissa * 2.0**-9)
                else:
                    values.append(sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - bias))
            else:
                exponent, mantissa, bias, bits = (byte >> 2) & 0x1F, byte & 0x3, 15, 2
                if exponent == 0x1F:
                    values.append(float("nan") if mantissa else sign * float("inf"))
                elif exponent == 0:
                    values.append(sign * mantissa * 2.0**-16)
                else:
                    values.append(sign * (1.0 + mantissa / 4.0) * 2.0 ** (exponent - bias))
        return values
    integers = {
        "BOOL": "<u1",
        "U8": "<u1",
        "I8": "<i1",
        "U16": "<u2",
        "I16": "<i2",
        "U32": "<u4",
        "I32": "<i4",
        "U64": "<u8",
        "I64": "<i8",
    }
    if dtype in integers:
        return [int(v) for v in np.frombuffer(raw, dtype=integers[dtype])]
    return None


def values_of(shard: pathlib.Path, dtype: str, begin: int, end: int, count: int):
    """Up to `count` decoded elements of one tensor, or `None` when its dtype is not decoded."""
    width = ELEMENTS.get(dtype, (1, ()))[0]
    with shard.open("rb") as handle:
        (length,) = np.frombuffer(handle.read(8), dtype="<u8")
        handle.seek(8 + int(length) + begin)
        raw = handle.read(min(end - begin, count * width))
    return decoded(dtype, raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="the checkpoint directory holding the safetensors shards")
    parser.add_argument("--tensors", help="scan only the tensor names containing this text")
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="report a tensor when at least this share of its words is zero (default: all of it)",
    )
    parser.add_argument("--chunk", type=int, default=1 << 26, help="bytes read at a time")
    parser.add_argument(
        "--values",
        type=int,
        default=0,
        metavar="N",
        help="also print the first N elements of each matched tensor, decoded by its own dtype",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="with --values, print at most this many tensors (default: 40)",
    )
    arguments = parser.parse_args()

    root = pathlib.Path(arguments.root)
    empty = 0
    scanned_tensors = 0
    skipped = set()
    printed = 0
    for shard, name, dtype, shape, begin, end in headers(root):
        if arguments.tensors and arguments.tensors not in name:
            continue
        if arguments.values and printed < arguments.limit:
            shown = values_of(shard, dtype, begin, end, arguments.values)
            if shown is None:
                skipped.add(dtype)
            else:
                printed += 1
                body = ", ".join(f"{value:g}" for value in shown)
                print(f"{name}\t{dtype}\t{shape}\t[{body}]")
        if dtype not in ELEMENTS:
            skipped.add(dtype)
            continue
        width, zeros = ELEMENTS[dtype]
        words = (end - begin) // width
        if words == 0:
            continue
        empty_words = 0
        with shard.open("rb") as handle:
            # The payload starts after the header, whose length is the first eight bytes.
            (length,) = np.frombuffer(handle.read(8), dtype="<u8")
            handle.seek(8 + int(length) + begin)
            remaining = words
            while remaining:
                want = min(remaining, max(1, arguments.chunk // width))
                raw = handle.read(want * width)
                if not raw:
                    raise SystemExit(f"{shard}: {name} ended early")
                array = np.frombuffer(raw, dtype=f"<u{width}")
                found = array == zeros[0]
                for alternate in zeros[1:]:
                    found |= array == alternate
                empty_words += int(np.count_nonzero(found))
                remaining -= array.size
        scanned_tensors += 1
        share = empty_words / words
        if share >= arguments.fraction:
            if empty_words == words:
                empty += 1
            print(f"{name}\t{dtype}\t{shape}\t{empty_words} of {words} words are zero")
    if skipped:
        print(f"# skipped element types this script does not read: {sorted(skipped)}", file=sys.stderr)
    print(
        f"# {scanned_tensors} tensor(s) scanned, {empty} of them hold no nonzero element"
        + (f", reported at a zero share of {arguments.fraction:g}" if arguments.fraction < 1 else "")
        + (f", {printed} printed as values" if arguments.values else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
