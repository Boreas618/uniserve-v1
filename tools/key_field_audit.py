#!/usr/bin/env python3
"""Which descriptor fields are earned: does a second value produce a second artifact?

    python tools/key_field_audit.py                    # every Triton package in the fixtures
    python tools/key_field_audit.py --operator gelu    # one operator
    python tools/key_field_audit.py --fields columns,extent

A key field exists because two of its values compile two different programs. A field whose values
compile the **same bytes** is a declaration the store pays for and no launch benefits from: every
deployment that states a new value needs a new store cell holding a copy of the same module. The
measurement is the two-build check — build the fixture's key, build it again with one field moved to
another declared value, and compare what the package wrote.

The comparison is over the artifact's own bytes (`module.so`, its metadata and headers): two keys
that differ only in a runtime word write identical modules, and the field is not earned. Compiled
backends (cuda, cutlass, cute_dsl, nvcc) are not audited here — their artifacts are host code where
an extent is baked by construction, and their builds are minutes rather than seconds.
"""

import argparse
import hashlib
import importlib.util
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = (
    HERE.parent / 'notes/programs/qwen3-moe-keys.txt',
    HERE.parent / 'notes/programs/minimax-h3-keys.txt',
    HERE.parent / 'notes/programs/minimax-h3-keys-bf16.txt',
    HERE.parent / 'notes/programs/minimax-h3-sparse-keys.txt',
)

#: Fields an extent is likely to be named by, most specific first.
CANDIDATES = (
    "extent", "columns", "rows", "tokens", "frames", "elements", "width",
    "block", "tile", "sequences", "samples", "packed", "channels", "head_dim",
)


def load_build_store():
    """`tools/build_store.py` as a module: the audit borrows its environment, not a copy of it."""
    spec = importlib.util.spec_from_file_location('build_store', HERE / 'build_store.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows_of(path):
    """The `operator<TAB>key` rows of a fixture."""
    rows = []
    for line in pathlib.Path(path).read_text().split('\n'):
        if not line or line.startswith('#'):
            continue
        operator, key = line.split('\t')
        rows.append((operator, key))
    return rows


def tables_of(package):
    """Every descriptor table of a manifest, as the checker's own reader sees them."""
    kernel = package["manifest"].get("kernel", {})
    kernels = kernel if isinstance(kernel, list) else [kernel]
    tables = []
    for one in kernels:
        described = one.get("descriptors", {})
        for table in described if isinstance(described, list) else [described]:
            tables.append(table.get("dimensions", {}))
    return tables


def moved(key, field, dimensions):
    """The key with `field`'s value moved to another declared one, or None.

    The value is found the way a key reader finds it: the longest declared value the key starts
    with after the field's name. A value that contains an underscore (`swizzled_128x4`) is a
    value, not a boundary, so nothing here splits the key on underscores.
    """
    marker = field + "_"
    start = key.find(marker)
    if start < 0:
        return None
    head = start + len(marker)
    rest = key[head:]
    matches = [value for value in dimensions[field] if rest.startswith(value)]
    if not matches:
        return None
    current = max(matches, key=len)
    others = [value for value in dimensions[field] if value != current]
    if not others:
        return None
    return key[:head] + others[0] + rest[len(current):]


def digest(directory):
    """Every file the build wrote, by name, as one hash."""
    entries = []
    for path in sorted(pathlib.Path(directory).rglob('*')):
        if path.is_file():
            entries.append((str(path.relative_to(directory)),
                            hashlib.sha256(path.read_bytes()).hexdigest()))
    return entries


def audit(store, package, key, fields):
    """Build `key`, then one moved key per candidate field; report which fields are earned."""
    reported = []
    with tempfile.TemporaryDirectory() as scratch:
        base = pathlib.Path(scratch) / "base"
        failure = store.build(package, key, "sm_100a", base)
        if failure is not None:
            return [("base", key, "the fixture's key does not build: " + failure)]
        reference = digest(base)
        for table in tables_of(package):
            for field in fields:
                if field not in table or len(table[field]) < 2:
                    continue
                moved_key = moved(key, field, table)
                if moved_key is None or moved_key == key:
                    continue
                out = pathlib.Path(scratch) / f"moved-{field}"
                failure = store.build(package, moved_key, "sm_100a", out)
                if failure is not None:
                    reported.append((field, moved_key, "does not build: " + failure))
                    continue
                same = digest(out) == reference
                reported.append((field, moved_key,
                                 "NOT EARNED (identical artifact)" if same else "earned"))
    return reported


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", default=None, help="audit one operator")
    parser.add_argument("--fields", default=",".join(CANDIDATES),
                        help="comma-separated field names to move, in priority order")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many operators")
    arguments = parser.parse_args()

    store = load_build_store()
    packages = store.load_packages(HERE.parent / "crates/ops/kernels")
    fields = [one.strip() for one in arguments.fields.split(",") if one.strip()]

    seen = set()
    unearned = 0
    audited = 0
    for fixture in FIXTURES:
        for operator, key in rows_of(fixture):
            if arguments.operator and operator != arguments.operator:
                continue
            if operator in seen:
                continue
            candidates = [one for one in packages.get(operator, [])
                          if one["manifest"]["source"]["kind"] == "triton"]
            if not candidates:
                continue
            seen.add(operator)
            package = candidates[0]
            audited += 1
            print(f"{operator} ({package['backend']}):")
            for field, moved_key, verdict in audit(store, package, key, fields):
                print(f"  {field:12} {verdict}")
                if "NOT EARNED" in verdict:
                    unearned += 1
            if arguments.limit and audited >= arguments.limit:
                return 0 if unearned == 0 else 1
    print(f"{audited} operator(s) audited, {unearned} field value(s) not earned")
    return 0 if unearned == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
