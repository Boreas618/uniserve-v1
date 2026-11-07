#!/usr/bin/env python3
"""Check the topk_softmax module and pin the routing it must produce.

    python topk_check.py --module <store>/modules/topk_softmax/nvcc/1/sm_100a/<descriptor>
    python topk_check.py --module ... --write artifacts/topk          # inputs and the command
    python topk_check.py --verify artifacts/topk                      # the harness's own output

**This operator is a cubin with no launcher of its own**, so the check has two halves and each
one says what it is.

**Without a device** it checks the module against its own build: the constants in `baked.json`
are the ones this run states and the ones the kernel derives from them (the iterations a lane
walks are `ceil(experts / 32)` and the renormalizing butterfly is the top-k rounded up to a power
of two), and the twelve words in `metadata.json`'s signature are the operator's, in order and by
kind — the fourth pointer is the scale, and a `scale_none` module still declares it as one, and the
last is the expert map, which the operator passes null without one. A module built for another
descriptor is refused here, which is the observable form of the kernel's `__trap()`.

**With a device** the launch is the harness's, not this script's:

```text
python crates/ops/kernels/topk_softmax/nvcc/test/topk_check.py --module <module> --write <dir>
cargo run -p program-server --bin cubin-launch -- --recipe <dir>/recipe.json --dump <dir>
python crates/ops/kernels/topk_softmax/nvcc/test/topk_check.py --verify <dir>
```

so the arithmetic is exercised through `platform`'s launch surface — the same route the executor
takes — and this script's device half is a comparison of two dumped buffers against numbers it
computed before the launch, not a second launcher. **That half is unproven until a device takes
it**: what is proven here is the module half, the pinned numbers, and the comparison's power to
fail against wrong dumps.

The module's own descriptor key states what it was built for, so `--module` alone is enough:
`experts`, `top`, the scores' storage and the renormalization flag are derived from the last path
component and the flags stay as overrides. The run prints what it derived.

**The rule it pins.** The scores are a softmax over the vocabulary, and the picks are the top
`top` of it, descending, **equal probabilities in ascending expert id** — the kernel compares
`(value, expert)` pairs, so a tie must resolve by id and not by whichever lane spoke last.
Renormalized, the weights are the picks' share of the picks' own sum, floored at FP32's epsilon.
The fixed input states both cases: one row whose second and third picks are exactly equal in
`bf16`, and every row's padding past `experts` filled with a value that would win if it were
ever read.

**With `--mapped`** a device run also passes the logical-to-physical map, reversed, and turns the
last row into a padding row of NaN scores, as a captured bucket's rows past the batch hold. The ids
must be the map's entries of the same picks, and the padding row must keep the id that names no
expert: it picks none, and a read of the map at that id is an illegal address.
"""

import argparse
import json
import pathlib
import re
import sys

import torch

#: The eleven parameters the operator launches with, in declaration order.
SIGNATURE = (
    ("scores", "pointer"),
    ("score_kind", "i32"),
    ("row_stride", "i64"),
    ("rows", "i32"),
    ("experts", "i32"),
    ("top", "i32"),
    ("renormalize", "i32"),
    ("scale", "pointer"),
    ("scale_kind", "i32"),
    ("ids", "pointer"),
    ("weights", "pointer"),
    ("map", "pointer"),
)

#: The storage kinds the kernel reads, as its `score_kind` and `scale_kind` take them.
KINDS = {"f32": 0, "bf16": 1, "f16": 2, "none": -1}

TORCH_TYPES = {
    "f32": torch.float32,
    "bf16": torch.bfloat16,
    "f16": torch.float16,
}

#: FP32's epsilon, the floor the renormalization divides by, and its unit roundoff.
FLT_EPSILON = 1.1920928955078125e-07
UNIT_ROUNDOFF = 2.0**-24

#: The warps one block holds, which the operator's grid counts tokens by.
WARPS_PER_BLOCK = 4


def derive(module):
    """The descriptor a module directory names, from its last path component.

    A store path ends in the descriptor's key —
    `scores_bf16_experts_128_top_8_renormalize_true_scale_none` — so the run takes the
    vocabulary, the top-k, the scores' storage and the renormalization flag from it. A key that
    does not say them is not a module this check can pin, and the run says so rather than asking
    for the same numbers twice.
    """
    key = module.name
    fields = {}
    for name, pattern in (
        ("experts", r"(?:^|_)experts_(\d+)(?=_|$)"),
        ("top", r"(?:^|_)top_(\d+)(?=_|$)"),
    ):
        found = re.search(pattern, key)
        if found:
            fields[name] = int(found.group(1))
    storage = re.search(r"(?:^|_)scores_([a-z0-9]+)(?=_|$)", key)
    if storage and storage.group(1) in TORCH_TYPES:
        fields["dtype"] = storage.group(1)
    renormalize = re.search(r"(?:^|_)renormalize_(true|false)(?=_|$)", key)
    if renormalize:
        fields["renormalize"] = renormalize.group(1) == "true"
    return fields


def fixed_case(experts, top, rows, row_stride, renormalize=True):
    """The input every phase agrees on, and the routing it must produce.

    The scores are built in `bf16`, so a tie made in the values is a tie the kernel sees. Row 0
    carries its second and third picks exactly equal, which only the ascending-id rule orders
    correctly; every row's lanes past `experts` hold a value large enough to win if the padding
    were ever read.
    """
    generator = torch.Generator().manual_seed(20240704)
    scores = torch.randn(rows, row_stride, generator=generator) * 0.5
    if row_stride > experts:
        # Nothing may read these: if a lane did, its pick would be the pad.
        scores[:, experts:] = 1000.0
    scores[0, 0] = 10.0
    scores[0, 1] = 5.0
    scores[0, 2] = 5.0
    scores = scores.to(torch.bfloat16)
    exact = scores.double()[:, :experts]

    # The softmax the kernel computes, in FP64: `exp(x - max)` over the vocabulary, normalized.
    maximum = exact.max(dim=-1, keepdim=True).values
    probabilities = torch.exp(exact - maximum)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)

    # The picks: descending, ties ascending by id. `torch.topk` does not promise that order, so
    # the reference sorts explicitly rather than trusting a library's tie-breaking.
    ids = torch.empty(rows, top, dtype=torch.int64)
    picked = torch.empty(rows, top, dtype=torch.float64)
    for row in range(rows):
        order = sorted(
            range(experts),
            key=lambda expert: (-probabilities[row, expert].item(), expert),
        )[:top]
        ids[row] = torch.tensor(order, dtype=torch.int64)
        picked[row] = probabilities[row][order]

    # Renormalized over the picks, as the kernel's butterfly does; its floor is FP32's epsilon.
    # A module built without renormalization reports the vocabulary's own probabilities.
    weights = picked
    if renormalize:
        total = picked.sum(dim=-1, keepdim=True)
        total = torch.where(
            total < FLT_EPSILON, torch.full_like(total, FLT_EPSILON), total
        )
        weights = picked / total
    return scores, ids, weights


def reference_bound(experts):
    """The operator's own tolerance: `γₙ` over the FP32 softmax plus four roundings."""
    terms = experts * UNIT_ROUNDOFF
    return terms / (1.0 - terms) + 4.0 * UNIT_ROUNDOFF


def check_module(module, experts, top, dtype, row_stride):
    """Refuse a module built for another descriptor, and check its published words."""
    problems = []
    baked_path = module / "baked.json"
    if not baked_path.exists():
        problems.append(
            f"{module} holds no baked.json, so its build constants are unobservable"
        )
    else:
        baked = json.loads(baked_path.read_text())
        expected = {
            "experts": experts,
            "top": top,
            "iterations": -(-experts // 32),
            "butterfly": 1 << (top - 1).bit_length(),
        }
        for key, value in expected.items():
            if baked.get(key) != value:
                problems.append(
                    f"baked.json says {key} = {baked.get(key)}, this run needs {value}"
                )
        print(
            f"baked: experts {baked.get('experts')}, top {baked.get('top')}, "
            f"iterations {baked.get('iterations')}, butterfly {baked.get('butterfly')}"
        )

    metadata_path = module / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    )
    signature = metadata.get("signature")
    if not signature:
        problems.append(
            f"{module} publishes no signature, so its words cannot be checked against it"
        )
    else:
        names = list(signature)
        if names != [name for name, _ in SIGNATURE]:
            problems.append(
                f"the module's parameters are {names}, the operator's are "
                f"{[name for name, _ in SIGNATURE]}"
            )
        else:
            for (name, kind), published in zip(SIGNATURE, signature.values()):
                if kind == "pointer":
                    if not published.startswith("*"):
                        problems.append(
                            f"parameter {name} is {published}, the operator passes a pointer"
                        )
                elif published != kind:
                    problems.append(
                        f"parameter {name} is {published}, the operator passes an {kind}"
                    )
            scores_kind = signature["scores"]
            wanted = "*" + {"f32": "fp32", "bf16": "bf16", "f16": "fp16"}[dtype]
            if scores_kind != wanted:
                problems.append(
                    f"the module reads {scores_kind} scores, this run builds {wanted}"
                )
        print(
            f"signature: {' '.join(f'{name} {kind}' for name, kind in signature.items())}"
        )
    return problems


def recipe_of(module, experts, top, rows, row_stride, dtype, mapped=False):
    """The harness's recipe for one launch of the module on the fixed input."""
    kind = {"f32": "i32", "bf16": "i32", "f16": "i32"}[dtype]
    recipe = {
        # Absolute: the harness resolves a recipe's relative module path against the recipe's own
        # directory, and a case written from the repo root is read from anywhere.
        "module": str(pathlib.Path(module).absolute()),
        "entry": "uniserve_topk_softmax",
        "grid": [-(-rows // WARPS_PER_BLOCK), 1, 1],
        "buffers": {
            "scores": {
                "bytes": rows
                * row_stride
                * torch.empty(0, dtype=TORCH_TYPES[dtype]).element_size(),
                "fill": "scores.bin",
            },
            "scratch": {"bytes": 64},
            "ids": {"bytes": rows * top * 4, "dump": "ids.bin"},
            "weights": {"bytes": rows * top * 4, "dump": "weights.bin"},
        },
        "parameters": [
            {"pointer": "scores"},
            {"i32": KINDS[dtype]},
            {"i64": row_stride},
            {"i32": rows},
            {"baked": "experts"},
            {"baked": "top"},
            {"i32": 1},
            {"pointer": "scratch"},
            {"i32": KINDS["none"]},
            {"pointer": "ids"},
            {"pointer": "weights"},
            {"i64": 0},
        ],
        "kinds": kind,
    }
    if mapped:
        recipe["buffers"]["map"] = {"bytes": experts * 4, "fill": "map.bin"}
        recipe["parameters"][-1] = {"pointer": "map"}
    return recipe


def write_inputs(
    directory, scores, module, experts, top, rows, row_stride, dtype, mapped=False
):
    """Write the fixed input and the recipe a device run uses."""
    directory.mkdir(parents=True, exist_ok=True)
    if mapped:
        scores = scores.clone()
        scores[-1, :] = float("nan")
        (directory / "map.bin").write_bytes(
            reversed_map(experts).to(torch.int32).numpy().tobytes()
        )
    # The bytes of the stored dtype, in the order the kernel reads them: one row of `row_stride`.
    stored = scores.to(TORCH_TYPES[dtype]).contiguous()
    (directory / "scores.bin").write_bytes(
        stored.view(torch.uint8).numpy().tobytes()
    )
    recipe = recipe_of(module, experts, top, rows, row_stride, dtype, mapped)
    recipe.pop("kinds", None)
    (directory / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    print(f"wrote {directory}/scores.bin and {directory}/recipe.json")


def reversed_map(experts):
    """The map a `--mapped` run passes: logical expert `e` lies in slot `experts - 1 - e`."""
    return torch.arange(experts - 1, -1, -1, dtype=torch.int64)


def verify_mapped(directory, ids, experts, rows, top):
    """Compare a `--mapped` device run's ids: slots for real picks, none for the padding row."""
    path = directory / "ids.bin"
    if not path.exists():
        return [f"{path} is missing: the harness did not dump it"]
    got = torch.from_file(str(path), dtype=torch.int32, size=rows * top).reshape(rows, top)
    problems = []
    expected = reversed_map(experts)[ids[:-1]]
    wrong = (got[:-1].long() != expected).sum().item()
    if wrong:
        problems.append(f"{wrong} routes do not hold the slot the map names")
    if ((got[-1].long() >= 0) & (got[-1].long() < experts)).any():
        problems.append(f"the padding row names a slot: {got[-1].tolist()}")
    print(f"verified: {(rows - 1) * top} mapped routes, {wrong} wrong; padding {got[-1].tolist()}")
    return problems


def verify(directory, ids, weights, experts, rows, top):
    """Compare a device run's two buffers against the numbers pinned before it."""
    ids_path, weights_path = directory / "ids.bin", directory / "weights.bin"
    problems = []
    for path in (ids_path, weights_path):
        if not path.exists():
            problems.append(f"{path} is missing: the harness did not dump it")
    if problems:
        return problems

    got_ids = torch.from_file(
        str(ids_path), dtype=torch.int32, size=rows * top
    ).reshape(rows, top)
    got_weights = torch.from_file(
        str(weights_path), dtype=torch.float32, size=rows * top
    ).reshape(rows, top)

    wrong_ids = (got_ids.long() != ids).sum().item()
    if wrong_ids:
        problems.append(
            f"{wrong_ids} of {rows * top} picks are not the ones the scores rank"
        )
        for row, slot in (got_ids.long() != ids).nonzero()[:3].tolist():
            problems.append(
                f"  [{row}, {slot}]: the module selected {got_ids[row, slot].item()}, "
                f"the ranking says {ids[row, slot].item()}"
            )
    bound = reference_bound(experts)
    difference = (got_weights.double() - weights).abs()
    outside = (~(difference <= bound * weights.abs())).sum().item()
    if outside:
        problems.append(
            f"{outside} of {rows * top} weights are outside {bound:.3e} relative"
        )
        for row, slot in (
            (~(difference <= bound * weights.abs())).nonzero()[:3].tolist()
        ):
            problems.append(
                f"  [{row}, {slot}]: the module weighed {got_weights[row, slot].item():.9e}, "
                f"the reference {weights[row, slot].item():.9e}"
            )
    sums = got_weights.double().sum(dim=-1)
    for row, total in enumerate(sums.tolist()):
        if abs(total - 1.0) > bound:
            problems.append(
                f"row {row}'s renormalized weights sum to {total:.9e}, not one"
            )
    print(
        f"verified: {rows * top} picks and weights, {wrong_ids} ids wrong, "
        f"{outside} weights outside {bound:.3e}, renormalized sums "
        f"{[round(total, 9) for total in sums.tolist()]}"
    )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=pathlib.Path, default=None)
    parser.add_argument(
        "--experts",
        type=int,
        default=None,
        help="the vocabulary; taken from the module's descriptor key when omitted",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="the picks per token; taken from the module's descriptor key when omitted",
    )
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument(
        "--row-stride",
        type=int,
        default=None,
        help="elements between the score rows; the vocabulary when omitted, since a row that "
        "is wider than the vocabulary exercises the padding a lane must not read",
    )
    parser.add_argument("--dtype", choices=sorted(TORCH_TYPES), default=None)
    parser.add_argument(
        "--renormalize", choices=["true", "false"], default=None
    )
    parser.add_argument("--write", type=pathlib.Path, default=None)
    parser.add_argument("--verify", type=pathlib.Path, default=None)
    parser.add_argument(
        "--mapped",
        action="store_true",
        help="pass the reversed expert map and a padding row of NaN scores",
    )
    arguments = parser.parse_args()

    if arguments.module is None and arguments.verify is None:
        print(
            "this check needs a module directory (--module) or a device run's output "
            "(--verify): nothing was compared",
            file=sys.stderr,
        )
        return 1

    # The module's own descriptor key states what it was built for, so the natural invocation
    # needs no numbers: the flags stay as overrides, and the run says where the rest came from.
    fields = derive(arguments.module) if arguments.module is not None else {}
    derived = []
    for name in ("experts", "top", "dtype", "renormalize"):
        value = getattr(arguments, name)
        if value is None:
            value = fields.get(name)
            if name in fields:
                derived.append(f"{name} {value}")
        if value is None:
            print(
                f"--{name} is not given and the module's path does not state it: "
                f"{arguments.module if arguments.module else arguments.verify}",
                file=sys.stderr,
            )
            return 1
        setattr(arguments, name, value)
    if arguments.row_stride is None:
        arguments.row_stride = arguments.experts
    if arguments.renormalize == "true":
        arguments.renormalize = True
    elif arguments.renormalize == "false":
        arguments.renormalize = False
    if derived:
        print(f"derived {', '.join(derived)} from the descriptor key")
    if arguments.top > 32:
        parser.error(
            "a warp routes one token, so the top-k width is at most 32"
        )
    if arguments.row_stride < arguments.experts:
        parser.error("a row narrower than the vocabulary cannot hold it")

    problems = []
    if arguments.module is not None:
        problems += check_module(
            arguments.module,
            arguments.experts,
            arguments.top,
            arguments.dtype,
            arguments.row_stride,
        )

    scores, ids, weights = fixed_case(
        arguments.experts,
        arguments.top,
        arguments.rows,
        arguments.row_stride,
        arguments.renormalize,
    )
    # The pinned expectation: printed so a device run compares against numbers rather than
    # re-deriving them, and so a reader can see the tie resolved the right way.
    print("pinned: ids " + str(ids.tolist()))
    print(
        "pinned: weights "
        + str([[round(value, 9) for value in row] for row in weights.tolist()])
    )
    print(
        f"pinned: renormalized sums {[round(total, 9) for total in weights.sum(dim=-1).tolist()]}"
    )
    if int(ids[0, 1]) > int(ids[0, 2]):
        problems.append(
            f"the tie in row 0 resolved to {ids[0, 1].item()} before {ids[0, 2].item()}"
        )

    if arguments.write is not None and arguments.module is not None:
        write_inputs(
            arguments.write,
            scores,
            arguments.module,
            arguments.experts,
            arguments.top,
            arguments.rows,
            arguments.row_stride,
            arguments.dtype,
            arguments.mapped,
        )

    if arguments.verify is not None and arguments.mapped:
        problems += verify_mapped(
            arguments.verify, ids, arguments.experts, arguments.rows, arguments.top
        )
    elif arguments.verify is not None:
        problems += verify(
            arguments.verify,
            ids,
            weights,
            arguments.experts,
            arguments.rows,
            arguments.top,
        )

    for problem in problems:
        print(f"FAIL: {problem}")
    if problems:
        return 1
    print(
        "the module and its pinned routing agree"
        + (
            ""
            if arguments.verify
            else "; the device half is the harness's own command"
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
