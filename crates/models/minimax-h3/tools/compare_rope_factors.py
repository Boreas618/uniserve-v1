"""Compare a prompt's text-encoder M-RoPE factors against the reference.

**This tool needs the retired Python tree.** It reads `uniserve`, the Python package that left this
repository with the Python worker, as the reference half of design 10's diagnosis. Run it with a
checkout of the last tree that held the package and that tree's interpreter, with the tree on the
path; `UNISERVE_REFERENCE_TREE` names that checkout:

    PYTHONPATH=$UNISERVE_REFERENCE_TREE:$UNISERVE_REFERENCE_TREE/uniserve_kernels/src \\
        $UNISERVE_REFERENCE_TREE/.venv/bin/python <this file> ...

`examples/encoder_rope.rs` writes the encode's own host tables for a prompt - the interleaved
coordinates, the cosine and the sine - and this recomputes them from the ids alone, with the
reference's `rope_index` and `RotaryEmbedding`, then prints the worst element of each. No device and
no weights are read: both sides are host arithmetic, which is why this can run anywhere the
checkpoint's `config.json` and the reference package are.

The instrument exists because a wrong spectrum is invisible to every other check. The tables have the
right shape, the right element type and the right count whatever the frequencies or the axis
assignment are, so a rotation can turn the wrong way by a factor and read as a tolerance problem in a
page. The DiT's rotary had exactly that defect, found only by comparing factors.

Three verdicts, by exit code, the same contract `compare_positions.py` uses:

    0  every table agrees to the floor
    1  a table differs, and the worst element is named
    2  the two sides are about different requests, so no comparison is possible

Usage:

    cargo run -q -p minimax-h3 --example encoder_rope -- --root <checkpoint> \\
        --prompt-ids <ids> --out artifacts/h3-stage4/rope
    python crates/models/minimax-h3/tools/compare_rope_factors.py \\
        --case artifacts/h3-stage4/rope --checkpoint <checkpoint>

The comparator has been seen to say no, three ways, and a check that has only ever been green is
not evidence:

- one coordinate of the case's `positions.h3t` raised by 1 -> ``positions (3, 8): 1 apart at
  element 19: ours 5, reference 4``, verdict ``differ``, exit 1;
- one element of `cosine.h3t` multiplied by 1.001 -> ``cosine (8, 64): max |difference| 1.000e-03
  at element 100``, verdict ``differ``, exit 1;
- a case whose `role.json` states a different `theta` -> ``refused the case states theta 10000.0 and
  the checkpoint's config gives 5000000.0; the pair is two different requests``, exit 2.

And the same two holds for the other two modes: `--dit` reports swapped axis blocks as ``the columns
take different coordinates`` and a spectrum value times 1.01 as ``5.623e-04 at element 5``; `--vision`
reports a column coordinate raised by 1 as ``positions (16, 2): 1 apart at element 7`` and a cosine
times 1.001 as ``1.000e-03 at element 7``.

**A text-only prompt cannot exercise the interleaving**, because every axis carries the token's own
position and the three coordinates are equal - a wrong axis assignment then produces identical
factors. The case that does exercise it states an image grid (`--image-grid`), which is the only
thing that makes the height and width axes differ from time.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import struct
import sys
import types
from pathlib import Path

# The floor this comparator reports against. Both sides are FP32 arithmetic over the same two
# inputs, so the only difference left is the order of two multiplications: a cosine is
# `cos(coordinate * frequency)` on both. Anything above this is a different spectrum, not rounding.
TOLERANCE = 1e-6

# The dtype codes the container uses, shared with the crate's dumps.
DTYPES = {1: ("f", 4), 4: ("bf16", 2), 3: ("i", 4)}


class Differ(SystemExit):
    """A comparison that must stop with a verdict rather than a traceback."""


def load_module(name: str, path: Path):
    """Load one reference module by path, without importing its package.

    The package's `__init__` pulls in the compiled kernels, which are not needed to evaluate a
    rotation and are not present everywhere this runs; these two modules import only `torch` and
    each other.
    """
    import importlib.util as importlib_util
    spec = importlib_util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise Differ(f"{path} is not a module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_container(path: Path):
    """Read one `H3D1` tensor: its dtype code, its extents and its values."""
    payload = path.read_bytes()
    if payload[:4] != b"H3D1":
        raise Differ(f"{path} is not an H3D1 container")
    code = payload[4]
    rank = payload[5]
    if code not in DTYPES:
        raise Differ(f"{path} holds dtype code {code}, which this reader does not know")
    kind, width = DTYPES[code]
    shape = tuple(struct.unpack_from("<" + "Q" * rank, payload, 6))
    body = payload[6 + 8 * rank :]
    count = 1
    for extent in shape:
        count *= extent
    if len(body) != count * width:
        raise Differ(
            f"{path} states {shape} of {width}-byte elements, which is"
            f" {count * width} bytes, and holds {len(body)}"
        )
    if kind == "f":
        values = struct.unpack("<" + "f" * count, body)
    elif kind == "i":
        values = struct.unpack("<" + "i" * count, body)
    else:
        raise Differ(f"{path} holds {kind}, which this comparator does not compare")
    return shape, values


def install_kernel_stub() -> bool:
    """Let the reference's CPU factor path run where its kernel package is not installed.

    `rope.RotaryEmbedding._factors` imports `uniserve_kernels.triton` before it branches, and the
    compiled package is not present on every host that has the reference's Python. On the CPU path
    that module is never *called* - it is only imported, and used under `positions.is_cuda` - so a
    stub that refuses if called keeps the reference's own arithmetic while removing an import that
    a CPU comparison does not need. Returns whether the stub was installed, so the verdict can say
    which branch produced it.
    """
    try:
        importlib.import_module("uniserve_kernels.triton")

        return False
    except ModuleNotFoundError:
        pass
    kernel = types.ModuleType("uniserve_kernels.triton")

    def require_kernel(name, unsupported, **kwargs):
        raise Differ(
            f"the reference asked for the {name} kernel, which this host does not have;"
            f" the comparison needs its CPU path"
        )

    kernel.require_kernel = require_kernel
    # `_factors` also imports `uniserve_kernels.rope`, the compiled bindings, immediately after; it
    # is used on the same device-only branch, and a stub here refuses on use rather than at import.
    bindings = types.ModuleType("uniserve_kernels.rope")

    def unused(*args, **kwargs):
        raise Differ(
            "the reference called the compiled rope bindings; the comparison needs its CPU path"
        )

    bindings.rotary_factors = unused
    bindings.unsupported_rotary_factors = unused
    package = types.ModuleType("uniserve_kernels")
    package.triton = kernel
    package.rope = bindings
    sys.modules["uniserve_kernels.rope"] = bindings
    sys.modules["uniserve_kernels"] = package
    sys.modules["uniserve_kernels.triton"] = kernel
    return True


def reference_factors(role: dict, case: Path, checkpoint: Path):
    """The reference's coordinates and factors for the case's prompt."""
    python = Path(os.environ["UNISERVE_REFERENCE_TREE"])
    # `uniserve.nn.rope` reads one dataclass off `uniserve.tensors`, and importing the package
    # itself pulls in the compiled kernels; the reference tree goes on the path so that one import
    # resolves and nothing else is loaded.
    if str(python) not in sys.path:
        sys.path.insert(0, str(python))
    torch = __import__("torch")
    positions_module = load_module(
        "reference_positions", python / "uniserve/models/qwen3_vl/positions.py"
    )
    rope_module = load_module("reference_rope", python / "uniserve/nn/rope.py")

    stubbed = install_kernel_stub()
    if stubbed:
        print(
            "note    the reference's compiled kernel module is absent here; its CPU factor path"
            " is the one that ran (that module is imported unconditionally and called only on a"
            " device)"
        )
    config = json.loads((checkpoint / "text_encoder" / "config.json").read_text())
    text = config.get("text_config", config)

    def value(name: str):
        """One field, from the text config or the file's top level.

        The export states the placeholder ids, the vocabulary and the position bound at the top
        level and the layer geometry under `text_config`; the crate reads both from one normalized
        config (`config/text.rs`), and this reader has to know which side each field is on.
        """
        if name in text:
            return text[name]
        return config[name]
    scaling = text.get("rope_scaling") or {}
    sections = tuple(scaling.get("mrope_section") or ())
    interleaved = bool(scaling.get("mrope_interleaved"))
    merge = int(
        (config.get("vision_config") or text.get("vision_config") or {}).get(
            "spatial_merge_size", 2
        )
    )
    ids = [int(value) for value in role["prompt_ids"]]

    # The case is the crate's own statement of what it computed; a difference here is a mismatched
    # pair, not a numeric verdict.
    for field, ours in (
        ("head_dim", int(text["head_dim"])),
        ("sections", list(sections)),
        ("theta", float(text["rope_theta"])),
        ("merge", merge),
    ):
        theirs = role.get(field)
        if field == "sections":
            theirs = [int(value) for value in theirs or []]
        elif field == "theta":
            theirs = float(theirs)
        else:
            theirs = int(theirs)
        if theirs != ours:
            raise Differ(
                f"the case states {field} {theirs} and the checkpoint's config gives {ours};"
                f" the pair is two different requests"
            )
    if not interleaved:
        raise Differ(
            "the checkpoint's rope_scaling does not state mrope_interleaved; this comparator"
            " compares the interleaved form the crate implements"
        )

    # The case's grids, in the shape the reference's own signature takes: one per image block, in
    # prompt order. A case that states none is a text-only prompt and every axis carries the token's
    # own position - which is why an interleaving defect is invisible on such a case and a grid is
    # what exercises it.
    grids = [tuple(int(extent) for extent in grid) for grid in role.get("image_grids") or ()]
    coordinates = positions_module.rope_index(
        ids,
        image_grids=grids,
        video_grids=(),
        image_token_id=int(value("image_token_id")),
        video_token_id=int(value("video_token_id")),
        spatial_merge_size=merge,
    )
    rotary = rope_module.RotaryEmbedding(
        int(text["head_dim"]),
        theta=float(text["rope_theta"]),
        sections=sections,
        scaling=None,
        max_position_embeddings=int(value("max_position_embeddings")),
    )
    # `rope_index` answers `[3, tokens]` - one row per axis - and the rotation takes that form
    # directly; a reader that assumed one triple per token would transpose a tensor that is already
    # axis-major and pass `[tokens, 3]`, which the reference refuses by name.
    packed = torch.as_tensor(coordinates, dtype=torch.float32)
    if packed.shape[0] != len(sections):
        packed = packed.t().contiguous()
    tokens = packed.shape[-1]
    with torch.inference_mode():
        cosine, sine = rotary(packed, dtype=torch.float32, sequence_length=tokens)
    triples = [[int(packed[axis, token]) for axis in range(3)] for token in range(tokens)]
    return triples, cosine.reshape(-1).tolist(), sine.reshape(-1).tolist()


def compare(name: str, mine, theirs, shape, *, integer: bool = False) -> float:
    """Print one table's worst element and return the size of the difference."""
    if len(mine) != len(theirs):
        raise Differ(f"{name}: {len(mine)} values against the reference's {len(theirs)}")
    if integer:
        worst = max((abs(int(a) - int(b)), index) for index, (a, b) in enumerate(zip(mine, theirs)))
        if worst[0] == 0:
            print(f"{name:<9} {shape}: every coordinate equal ({len(mine)} values)")
        else:
            index = worst[1]
            print(
                f"{name:<9} {shape}: {worst[0]} apart at element {index}:"
                f" ours {mine[index]}, reference {theirs[index]}"
            )
        return float(worst[0])
    worst = max((abs(float(a) - float(b)), index) for index, (a, b) in enumerate(zip(mine, theirs)))
    if worst[0] <= TOLERANCE:
        print(
            f"{name:<9} {shape}: max |difference| {worst[0]:.3e} over {len(mine)} values"
            f" (floor {TOLERANCE:.0e})"
        )
    else:
        index = worst[1]
        print(
            f"{name:<9} {shape}: max |difference| {worst[0]:.3e} at element {index}:"
            f" ours {mine[index]:.6f}, reference {theirs[index]:.6f}"
        )
    return worst[0]


def compare_spectrum(case: Path, checkpoint: Path, *, stated_as: str, label: str) -> int:
    """Compare one component's rotary spectrum and axis order with the reference's.

    The factors of both components are device operators, so what the host can compare is what feeds
    them: the table (one spectrum per axis) and the axis each of its columns takes. Both are wrong in
    ways no shape can see - distinct frequencies where the reference repeats one spectrum, or the axes
    in another order - and the DiT's rotary has had exactly that class of defect.

    `stated_as` is the file the probe wrote; `label` names the component in the report.
    """
    if install_kernel_stub():
        print(
            "note    the reference's compiled kernel module is absent here; only its host inverse"
            " frequency table is read, which needs no kernel"
        )
    stated = json.loads((case / stated_as).read_text())
    pairs = int(stated["pairs"])
    axes = int(stated["axes"])
    theta = float(stated["theta"])
    table = [float(value) for value in stated["table"]]

    python = Path(".").resolve()
    if str(python) not in sys.path:
        sys.path.insert(0, str(python))
    rope_module = load_module("reference_rope", python / "uniserve/nn/rope.py")
    reference = rope_module.RotaryEmbedding(
        2 * pairs, theta=theta, sections=None, scaling=None, max_position_embeddings=32768
    )
    spectrum = [float(value) for value in reference.inv_freq.reshape(-1).tolist()]
    if len(spectrum) != pairs:
        raise Differ(
            f"the reference's RotaryEmbedding({2 * pairs}) gives {len(spectrum)} frequencies and"
            f" the case states {pairs}"
        )
    # The reference's factors are `[axes, tokens, dim / 2]`, flattened axis-major
    # (`rope.py:355-380`), so column `a * pairs + i` is the `i`-th frequency of axis `a` and the
    # spectrum repeats once per axis.
    repeated = spectrum * axes
    implied = "".join(str(index // pairs) for index in range(axes * pairs))
    print(f"case    {label}: {pairs} frequencies, {axes} axes over {len(table)} columns, theta {theta}")
    worst = compare("spectrum", table, repeated, (axes * pairs,))
    if stated["spectrum_axes"] != implied:
        print(
            f"axes    ours {stated['spectrum_axes']} and the reference's implied {implied}:"
            f" the columns take different coordinates"
        )
        return 1
    print(f"axes    {implied}: the axes occupy contiguous blocks of {pairs} columns, as ours do")
    if worst <= TOLERANCE:
        print("verdict equal")
        return 0
    print("verdict differ")
    return 1


def compare_vision(case: Path, checkpoint: Path) -> int:
    """Compare the vision tower's 2-D rotary factors with the reference's.

    The reference builds one `RotaryEmbedding(head_dim / 2)` without sections
    (`qwen3_vl/vision.py:168`) and calls it with `[patches, 2]` merged-grid coordinates (`:253`),
    flattening `[patches, 2, head_dim / 4]` to `[patches, head_dim / 2]` - so each axis uses the
    whole 18-frequency spectrum and the axes are contiguous halves. The crate computes the same
    tables on the host (`positions::VisionRotary::factors`), which is why this can be compared at
    all.
    """
    if install_kernel_stub():
        print(
            "note    the reference's compiled kernel module is absent here; only its host factor"
            " arithmetic is used, which needs no kernel"
        )
    stated = json.loads((case / "vision_rope.json").read_text())
    head_dim = int(stated["head_dim"])
    theta = float(stated["theta"])
    height, width, merge = int(stated["height"]), int(stated["width"]), int(stated["merge"])
    patches = int(stated["patches"])
    shape, coordinates = read_container(case / "coordinates.h3t")
    _, cosine = read_container(case / "cosine.h3t")
    _, sine = read_container(case / "sine.h3t")
    if shape != (patches, 2):
        raise Differ(f"the case's coordinates are {shape} for {patches} patches")

    python = Path(".").resolve()
    if str(python) not in sys.path:
        sys.path.insert(0, str(python))
    torch = importlib.import_module("torch")
    position_module = load_module(
        "reference_vision_position", python / "uniserve/nn/vision/position.py"
    )
    rope_module = load_module("reference_rope", python / "uniserve/nn/rope.py")
    rows, columns = position_module.merged_grid_coordinates(height, width, merge)
    theirs = torch.stack((rows.reshape(-1), columns.reshape(-1)), dim=-1).to(torch.float32)
    rotary = rope_module.RotaryEmbedding(
        head_dim // 2,
        theta=theta,
        sections=None,
        scaling=None,
        max_position_embeddings=32768,
    )
    with torch.inference_mode():
        their_cosine, their_sine = rotary(
            theirs, dtype=torch.float32, sequence_length=0
        )
    theirs_coordinates = [[int(a), int(b)] for a, b in theirs.tolist()]
    ours_coordinates = [
        [coordinates[2 * index], coordinates[2 * index + 1]] for index in range(patches)
    ]
    flat_theirs = [value for pair in theirs_coordinates for value in pair]
    print(f"case    {patches} patches of a {height}x{width} grid merged by {merge}, head {head_dim}")
    worst = compare("positions", [v for pair in ours_coordinates for v in pair], flat_theirs, shape, integer=True)
    factors = (patches, head_dim // 2)
    worst = max(
        worst,
        compare("cosine", cosine, their_cosine.flatten().tolist(), factors),
    )
    worst = max(worst, compare("sine", sine, their_sine.flatten().tolist(), factors))
    if worst <= TOLERANCE:
        print("verdict equal")
        return 0
    print("verdict differ")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, help="a prompt's factors, as encoder_rope wrote them")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dit",
        type=Path,
        help="compare the DiT's spectrum and axis order instead of a prompt's factors",
    )
    parser.add_argument(
        "--vae",
        type=Path,
        help="compare the video decoder's spectrum and axis order",
    )
    parser.add_argument(
        "--vision",
        type=Path,
        help="compare the vision tower's 2-D rotary factors",
    )
    arguments = parser.parse_args(argv)
    if arguments.dit is not None:
        return compare_spectrum(
            arguments.dit,
            arguments.checkpoint,
            stated_as="dit_rope.json",
            label="the DiT's",
        )
    if arguments.vae is not None:
        return compare_spectrum(
            arguments.vae,
            arguments.checkpoint,
            stated_as="vae_rope.json",
            label="the video decoder's",
        )
    if arguments.vision is not None:
        return compare_vision(arguments.vision, arguments.checkpoint)
    if arguments.case is None:
        parser.error("one of --case, --dit, --vae or --vision is required")

    role = json.loads((arguments.case / "role.json").read_text())
    shape, coordinates = read_container(arguments.case / "positions.h3t")
    _, cosine = read_container(arguments.case / "cosine.h3t")
    _, sine = read_container(arguments.case / "sine.h3t")
    tokens = len(role["prompt_ids"])
    if shape != (3, tokens):
        raise Differ(f"the case's positions are {shape} for {tokens} ids")

    theirs_coordinates, theirs_cosine, theirs_sine = reference_factors(
        role, arguments.case, arguments.checkpoint
    )
    # The case is axis-major (`[3, tokens]`, which is what `ops::rope` documents) and the
    # reference's list is one triple per token; this transposition is the one convention difference
    # between the two and it is stated rather than hidden.
    ours_coordinates = [
        coordinates[axis * tokens + token]
        for token in range(tokens)
        for axis in range(3)
    ]
    flat_theirs = [value for row in theirs_coordinates for value in row]

    print(f"case    {tokens} ids, sections {role['sections']}, theta {role['theta']}")
    columns = len(cosine) // tokens
    factors = (tokens, columns)
    worst = compare("positions", list(ours_coordinates), flat_theirs, shape, integer=True)
    worst = max(worst, compare("cosine", cosine, theirs_cosine, factors))
    worst = max(worst, compare("sine", sine, theirs_sine, factors))
    # The verdict is the contract, not `worst == 0`: two FP32 evaluations of one formula differ in
    # the last bit, and a comparator that calls one ulp a disagreement is a comparator that will be
    # argued with rather than read.
    if worst <= TOLERANCE:
        print("verdict equal")
        return 0
    print("verdict differ")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Differ as stop:
        print(f"refused {stop}")
        sys.exit(2)
