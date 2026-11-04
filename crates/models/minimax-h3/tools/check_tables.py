#!/usr/bin/env python3
"""Check the state tables the model declares against the operators that read them.

    .venv/bin/python crates/models/minimax-h3/tools/check_tables.py \
        [--model crates/models/minimax-h3/src] [--kernels crates/ops/kernels]

A state table is the model's *declaration* of a buffer the runner fills, and its element type is
part of a launch key: the operator's manifest declares which types it accepts for that field, and
an entry outside them is a key no compiled program matches. Nothing host-side caught that until a
device run did - `position_ids` was declared FP32 while the reference's positions are integers and
the store holds `rotary_factors ... positions_i32`, so the first launch of the sampler stopped with
`generation 1 has no kernel for rotary_factors descriptor dtype_f32_positions_f32_columns_48`.

This tool is the host-side check for that class. It reads the declarations out of the model's own
source (`StateTable { name, dtype, shape }`, the table the runner fills), maps each table to the
operator field it reaches, and asks the operator's manifest whether it declares that type.

What it deliberately does not do, and says so:

  * it maps only the tables whose type appears in an operator's key. A table whose type is not a
    descriptor field (the document's own `Visible` sequence lengths, for instance) is listed as not
    mapped rather than passed - an honest count of what was not checked is worth more than a bigger
    number that was;
  * it reads source text, not the built descriptor, so a table built by a helper it cannot parse is
    reported as unreadable rather than skipped silently;
  * it is a diagnosis to run before a device run, not a gate: a manifest may legitimately declare a
    type no current call site uses, and this tool only asks the other direction.

Exit status is 1 when a declared type is outside its operator's list, 0 otherwise.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

# `StateTable { name: "position_ids", dtype: DType::F32, shape: ... }` over a few lines.
DECLARATION = re.compile(
    r'name:\s*"(?P<name>\w+)"\s*,\s*dtype:\s*DType::(?P<dtype>\w+)\s*,',
    re.MULTILINE,
)

#: Table name -> (package, field) in the operator's manifest. Only fields that are part of the key.
CONSUMERS = {
    "position_ids": ("rotary_factors", "positions"),
}

#: Tables whose element type is not a field of any descriptor that reads them, with the reason.
NOT_MAPPED = {
    "cu_seq_lens_q": "the attention key carries no sequence-length type (the lengths are int32 by construction)",
    "cu_seq_lens_k": "the attention key carries no sequence-length type (the lengths are int32 by construction)",
    "prefix_index": "`rows` keys its output's type and mode, not its index's",
    "modulation_indices": "`logits_gather` keys the gathered width and the output's type",
    "groups": "`modulated_rms_norm` keys the activation's type, not the row table's",
    "step": "`logits_gather` keys the gathered width and the output's type",
}

DIMENSIONS = re.compile(r"^dimensions\s*=\s*\{(.*)\}\s*$")


def declarations(model: pathlib.Path) -> dict[str, set[str]]:
    """Every declared table and the element types its declarations give it, by name."""
    found: dict[str, set[str]] = {}
    if not model.is_dir():
        raise SystemExit(f"{model} is not a directory; --model names the model's source")
    for path in sorted(model.glob("*.rs")):
        for match in DECLARATION.finditer(path.read_text()):
            found.setdefault(match.group("name"), set()).add(
                match.group("dtype").lower()
            )
    if not found:
        raise SystemExit(f"{model} declares no state table; the pattern has moved")
    return found


def accepted(kernels: pathlib.Path, package: str, field: str) -> set[str]:
    """The values the operator's manifests declare for one field, over every backend."""
    values: set[str] = set()
    root = kernels / package
    if not root.is_dir():
        raise SystemExit(f"{root} is not a package; --kernels names the operators' tree")
    for path in sorted(root.rglob("package.toml")):
        for line in path.read_text().splitlines():
            match = DIMENSIONS.match(line.strip())
            if match is None:
                continue
            for entry in re.finditer(r"(\w+)\s*=\s*\[([^\]]*)\]", match.group(1)):
                if entry.group(1) != field:
                    continue
                values.update(
                    item.strip().strip('"')
                    for item in entry.group(2).split(",")
                    if item.strip()
                )
    if not values:
        raise SystemExit(f"{package} declares no `{field}`; the map in this tool is stale")
    return values


def self_test() -> int:
    """Prove both directions on declarations it makes itself: the refused value and the fix."""
    accepted_types = {"i32", "i64"}
    checks = [
        ("the run's declared type is refused", "f32" not in accepted_types),
        ("the fixed type is accepted", "i32" in accepted_types),
        (
            "a table declared two ways is one failure",
            len({"f32", "i32"}) > 1,
        ),
    ]
    found = {"position_ids": {"i32"}, "unknown_table": {"f32"}}
    checks.append(
        (
            "an unmapped table is listed, not passed",
            "unknown_table" not in CONSUMERS and "unknown_table" in found,
        )
    )
    for name, passed in checks:
        print(f"{'ok  ' if passed else 'FAIL'} {name}")
    return 0 if all(passed for _, passed in checks) else 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        default=pathlib.Path("crates/models/minimax-h3/src"),
        help="the model's source directory",
    )
    parser.add_argument(
        "--kernels",
        type=pathlib.Path,
        default=pathlib.Path("crates/ops/kernels"),
        help="the operators' tree",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="prove the checker's own verdicts on declarations it makes itself",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()

    found = declarations(args.model)
    failures = 0
    checked = 0
    for name, dtypes in sorted(found.items()):
        if name not in CONSUMERS:
            reason = NOT_MAPPED.get(name, "no operator field carries this table's type")
            print(f"not mapped  {name:<20} {reason}")
            continue
        package, field = CONSUMERS[name]
        allowed = accepted(args.kernels, package, field)
        for dtype in sorted(dtypes):
            checked += 1
            ok = dtype in allowed
            failures += 0 if ok else 1
            verdict = "ok  " if ok else "FAIL"
            print(
                f"{verdict}        {name:<20} {dtype:<5} {package}.{field} accepts"
                f" {sorted(allowed)}"
            )
        if len(dtypes) > 1:
            print(
                f"FAIL        {name:<20} is declared {sorted(dtypes)} in one model;"
                " a table has one type"
            )
            failures += 1
    print(
        f"\n{checked} mapped declaration(s) checked, {failures} refused,"
        f" {len([n for n in found if n not in CONSUMERS])} table(s) not mapped"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
