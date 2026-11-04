"""The cross_fade package's compile script: what this program would be, and why it cannot be built.

`cross_fade` blends the seam between two decoded tiles: over the leading `extent` voxels along one
axis, `out = previous · (1 − i/extent) + current · (i/extent)`, and `current` everywhere after them.
The blend is a position's ratio, so a kernel must know **where a voxel sits along the blended
axis**, and that needs the tile's inner extents.

**The operator's launch does not carry them.** `cross_fade.rs` passes
`previous, current, out, extent, axis, channels, frames, block, pdl`: the height and the width of
the `[channels, frames, height, width]` tile are not in the argument pack. The only thing a kernel
could recover is their *product* — the grid's voxel count divided by `channels · frames` — and the
seam needs each extent separately: for `axis = 3` the width alone, for `axis = 2` both. Deriving one
from the other, or from a stride, or fading a neighbouring grid instead, would compute a different
filter from the one the reference applies, and a wrong fade is worse than a refusal.

So this script refuses every descriptor with that reason, in the message the store shows the
caller. The package exists so the refusal is articulate and attributed: a `prepare` of any
`cross_fade` key fails here, naming what is missing, rather than reporting the operator as one
nobody packaged or, worse, building a module that blends the wrong axis.

When `cross_fade.rs` passes `dims[2]` and `dims[3]`, this script becomes the usual Triton build —
the kernel walks the output, decodes the voxel into (channel, frame, row, column), blends inside the
leading `extent` positions of the named axis and copies elsewhere — and the package needs an hour.

The Program Server runs this script with `UNISERVE_DESCRIPTOR`, `UNISERVE_TARGET`,
`UNISERVE_PACKAGE` and `UNISERVE_OUT` in the environment; `--package`, `--descriptor`, `--target`
and `--out` name the same values on the command line.
"""

import argparse
import os

#: The two words the operator's launch does not carry.
MISSING = "dims[2] and dims[3] (the frame grid's inner extents)"


def arguments():
    """The descriptor and target this build was asked for, for the refusal's message."""
    reader = argparse.ArgumentParser(description=__doc__)
    for flag in ("package", "descriptor", "target", "out"):
        reader.add_argument(f"--{flag}")
    known, _unused = reader.parse_known_args()
    descriptor = known.descriptor or os.environ.get("UNISERVE_DESCRIPTOR", "a descriptor")
    target = known.target or os.environ.get("UNISERVE_TARGET", "a target")
    return descriptor, target


def main():
    """Refuse, naming the two values the operator's call structure is missing."""
    descriptor, target = arguments()
    raise SystemExit(
        f"cross_fade {descriptor} for {target}: the operator's launch does not carry {MISSING}, so "
        "no kernel can address the blend band along axis 2 or 3. This package is written and "
        "waiting for those two words in `crates/ops/src/cross_fade.rs`; until then it refuses "
        "rather than fade a grid it cannot locate."
    )


if __name__ == "__main__":
    main()
