#!/usr/bin/env python3
"""What a forward binds and the store list does not hold - both directions.

The store a caller builds comes from a fixture under `notes/programs/`, one per deployment, and it is
a **walk's rows**: `cargo run -p minimax-h3 --example programs -- <root> --dtype <t> ...` prints them
from the model's own layers, and `h3_fixture --walk <print> --commit <rev>` writes the file with the
tree it came from. Three checks read that file, and they answer three different questions:

* `key_coverage` asks whether a **listed** key is buildable for a target;
* `check_manifests.py` asks whether a **descriptor the code derives** has a manifest cell;
* this one asks whether a **key a forward binds** is in the list.

The first cannot see a bound key that is not listed, and the second cannot see it either unless the
descriptor walk knows about it - so a store can be complete over everything anyone wrote down and
still stop a device run one key later. That is what happened twice (`logits_gather`, then `gemm`
`epilogue_bias`).

The walk file comes from a forward that reports its launches. `ops::capture::Sink` is that recorder:
every model layer launches through `Forward::operate`, which reports `(operator, key)` to the sink
installed on the thread, so a harness installs one around the forward and writes what it recorded.
One line per launch, `operator<TAB>key`, in launch order - duplicates are fine and are counted.

Usage:

    walk_gaps.py --walk artifacts/h3-stage1/walk.txt \\
                 --fixture notes/programs/minimax-h3-keys.txt [--unbound]

Exit status: 1 when a bound key has no fixture line (the run would stop with `no directory`/`no
kernel`), 0 otherwise. Unbound fixture lines are printed but never failed on by default.

**A file is one deployment's rows, so an unbound line means one of two things, and they are worth
telling apart before it is dismissed:** the fixture is another deployment's (a `--dtype` or a degree
that does not match this walk), or the walk is narrower than the file because a component it cannot
reach - a stage that has never run - contributed its rows to a *bootstrap* list and not to a walk. The
first is a wrong file; the second is a component this run never exercised, and neither is "normal
coverage". `--allow-unbound` silences the count entirely, so it is for a reader who has already said
which of the two it is.
"""

import argparse
import sys
from collections import Counter


def read_lines(path):
    """`(operator, key)` pairs of one file, and how many times each appears."""
    pairs = Counter()
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            fields = text.split("\t")
            if len(fields) != 2:
                raise SystemExit(f"{path}:{number}: expected `operator<TAB>key`, found {text!r}")
            pairs[(fields[0], fields[1])] += 1
    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--walk", required=True, help="what the forward reported, one launch a line")
    parser.add_argument("--fixture", required=True, help="the store list the caller builds from")
    parser.add_argument(
        "--unbound",
        action="store_true",
        help="list every fixture line no launch bound (the normal case for one deployment, so the "
        "default prints the count and a few)",
    )
    args = parser.parse_args(argv)

    walk = read_lines(args.walk)
    fixture = read_lines(args.fixture)

    bound = sorted(walk)
    unbound = sorted(set(fixture) - set(walk))
    missing = sorted(set(walk) - set(fixture))

    calls = sum(walk.values())
    print(f"walk: {len(bound)} distinct keys over {calls} launches")
    print(f"fixture: {len(fixture)} keys")
    for operator, key in missing:
        print(f"  MISSING from the fixture: {operator}\t{key}  ({walk[(operator, key)]} launches)")
    if unbound:
        print(f"fixture lines no launch bound ({len(unbound)}; normal for one deployment):")
        shown = unbound if args.unbound else unbound[:8]
        for operator, key in shown:
            print(f"  unbound: {operator}\t{key}")
        if len(shown) < len(unbound):
            print(f"  … {len(unbound) - len(shown)} more with --unbound")

    if missing:
        print(f"\n{len(missing)} key(s) a forward binds and the store list does not hold")
        return 1
    print("\nevery key the walk bound is in the fixture")
    return 0


if __name__ == "__main__":
    sys.exit(main())
