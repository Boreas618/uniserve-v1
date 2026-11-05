#!/usr/bin/env python3
"""Assemble a program store for a key fixture, offline, without a GPU.

For every fixture row whose descriptor some package covers, run that package's compile script with
the environment the Program Server passes (`UNISERVE_OP`, `UNISERVE_BACKEND`, `UNISERVE_DESCRIPTOR`,
`UNISERVE_TARGET`, `UNISERVE_PACKAGE`, `UNISERVE_OUT`), then publish the result under the store
layout `<root>/<op>/<backend>/<revision>/<target>/<descriptor>/` with the `.committed` mark.

The output is a store a reader can use: `program_server::store::Store::open([root])`, or the
`qwen3_moe_slice` example's `--store`. It is a stopgap for a host where the Program Server has not
run; the published layout and the environment are the Program Server's own, so the two cannot
disagree about what a module looks like.

Every cell states the digest of the source it was built from, every build ends by verifying what it
wrote, and `--verify` answers whether the
cells a fixture needs were built from the source this tree holds. A store outlives the tree it was
built from, and a cell that says nothing about its source cannot be told from one built yesterday
against a package that has changed since - which is how two runs stopped inside kernels whose
constants and argument order had moved.
"""

import argparse
import hashlib
import json
import os
import ctypes
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

TARGETS = {"sm_100a", "sm_90a"}


def load_packages(kernels: Path):
    """Every package under <kernels>/<op>/<backend>/, by op."""
    packages = {}
    for manifest_path in sorted(kernels.glob("*/*/package.toml")):
        with open(manifest_path, "rb") as handle:
            manifest = tomllib.load(handle)
        package = {
            "root": manifest_path.parent,
            "manifest": manifest,
            "op": manifest["op"],
            "backend": manifest["backend"],
        }
        packages.setdefault(package["op"], []).append(package)
    return packages


def split_key(dimensions, key):
    """The key's `dimension_value` pairs, longest dimension name first."""
    names = sorted(dimensions, key=len, reverse=True)
    pairs, rest = [], key
    while rest:
        name = next((n for n in names if rest.startswith(n)), None)
        if name is None or not rest[len(name):].startswith("_"):
            return None
        after = rest[len(name) + 1:]
        ends = [after.find("_" + n + "_") for n in names]
        ends = [end for end in ends if end >= 0]
        end = min(ends) if ends else len(after)
        if not after[:end]:
            return None
        pairs.append((name, after[:end]))
        rest = after[end:].lstrip("_")
    return pairs


def shapes(kernel):
    """The descriptor tables a package declares: one, or one per shape."""
    descriptors = kernel.get("descriptors")
    if descriptors is None:
        return []
    return descriptors if isinstance(descriptors, list) else [descriptors]


def covers_one(descriptors, key):
    if key in descriptors.get("keys", []):
        return True
    dimensions = descriptors.get("dimensions") or {}
    if not dimensions:
        return False
    pairs = split_key(dimensions, key)
    if pairs is None:
        return False
    return all(value in dimensions.get(name, []) for name, value in pairs)


def prune_empty_directories(root: Path) -> None:
    """Remove the empty ancestors a failed attempt leaves behind.

    Only directories that hold nothing at all are removed, deepest first, so a store a failed build
    touched is left as the build found it. Anything a cell wrote keeps its directory.
    """
    if not root.is_dir():
        return
    for directory in sorted((one for one in root.rglob("*") if one.is_dir()), reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()

def covers(package, key):
    """Whether the package covers this key: it covers one of the shapes it declares."""
    kernel = package["manifest"].get("kernel")
    if kernel is None:
        return False
    return any(covers_one(descriptors, key) for descriptors in shapes(kernel))


def build(package, key, target, out_dir, python):
    """Compile one descriptor into `out_dir`; returns a failure message or None."""
    manifest = package["manifest"]
    source = manifest["source"]
    script = source.get("script")
    if not script:
        return "the package declares no compile script"
    environment = dict(os.environ)
    environment.update({
        "UNISERVE_OP": package["op"],
        "UNISERVE_BACKEND": package["backend"],
        "UNISERVE_DESCRIPTOR": key,
        "UNISERVE_TARGET": target,
        "UNISERVE_PACKAGE": str(package["root"].resolve()),
        "UNISERVE_OUT": str(out_dir.resolve()),
    })
    if source["kind"] == "triton":
        # Triton's cache is build scratch, not part of the module: keep it out of the store so a store
        # can be copied or published without a compilation cache inside it.
        environment.setdefault("TRITON_CACHE_DIR", str(Path(tempfile.gettempdir()) / "uniserve-triton-cache"))
    # A package that fetches a vendored artifact asks for the Program Server's own interpreter by
    # importing `uniserve_program_server`; making that package importable lets the script keep this
    # interpreter, which is the one that carries the toolchain (Triton, CuTe DSL, nvcc).
    python_path = str((Path(__file__).resolve().parents[1]
                       / "crates/program-server/python/src").resolve())
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = python_path + (os.pathsep + existing if existing else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    command = {
        "triton": [python, script],
        "cuda": [python, script],
        "cutlass": [python, script],
        "cute_dsl": [python, script],
        "nvcc": [python, script],
    }.get(source["kind"])
    if command is None:
        command = [python, script]
    try:
        finished = subprocess.run(
            command,
            cwd=package["root"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=1200,
        )
    except subprocess.TimeoutExpired:
        return "the compile script did not finish in 20 minutes"
    if finished.returncode != 0:
        tail = (finished.stderr or finished.stdout).strip().splitlines()[-3:]
        return "compile script failed: " + " | ".join(tail)
    form = manifest["module"]["form"]
    if form == "cubin" and not (out_dir / "module.cubin").is_file():
        return "no module.cubin was written"
    if form == "native" and not any(out_dir.glob("*.so")):
        return "no shared object was written"
    if form == "native":
        # A shared object that references symbols no library provides still appears in the store:
        # the file exists, the manifest names its entry, and "built N of N" reads as a complete
        # store. The failure then arrives at the first launch as "cannot load native library",
        # which names neither a symbol nor a package - and a run that stops there looks like a
        # model defect. `RTLD_NOW` resolves the whole object at load, so an undefined reference
        # is a build failure with the symbol in it.
        for library in sorted(out_dir.glob("*.so")):
            failure, unresolved = inspect_module(library)
            if failure:
                return failure
            if unresolved:
                # Loadable, and one call away from failing. Named rather than merged into the
                # failure count: a store that cannot be built is a run that cannot start, and
                # these are cells the engine loads today.
                print(f"# warning: {library.name} loads lazily and references {unresolved}")
        # The link inputs are not part of the module: an object file beside the library is build
        # scratch, and a store is what a host loads rather than what a compiler read.
        for scratch in out_dir.glob("*.o"):
            scratch.unlink()
    # A native module's C entry is declared by its manifest and is not in the shared object; publish
    # it beside the library so the store directory is enough to load the module from.
    entries = manifest["module"].get("entries")
    if form == "native" and entries:
        (out_dir / "launch.json").write_text(
            json.dumps({"name": entries.get("launch"), "entries": entries}, indent=1) + "\n"
        )
    return None


def inspect_module(library: Path):
    """Whether the runtime could load `library`, and which symbols it would leave unresolved.

    Two questions, kept apart because they have different answers and different cures.

    The first is what the runtime does: `platform::native` opens with `libloading::Library::new`,
    which is `dlopen(RTLD_LAZY | RTLD_LOCAL)` in `libloading` 0.8.9. A module that fails THAT
    fails for the engine too, and it is a build failure.

    The second is `RTLD_NOW`: every reference resolved at load. A module can pass the first and
    fail the second, and then it works until the unresolved path is taken. That is worth a
    warning naming the symbol, and it is not worth failing a store the engine can load.

    Each probe runs in its own process. A second `dlopen` of a file already loaded returns the
    existing handle without re-resolving, so two modes measured in one process report the first
    mode twice - a mistake that reads as "both load".
    """
    probe = (
        "import ctypes, os, sys\n"
        "ctypes.CDLL(sys.argv[1], mode=int(sys.argv[2]))\n"
    )
    def run(mode):
        finished = subprocess.run(
            [sys.executable, "-c", probe, str(library), str(mode)],
            capture_output=True,
            text=True,
        )
        if finished.returncode == 0:
            return None
        tail = (finished.stderr or finished.stdout).strip().splitlines()[-1:]
        return tail[0] if tail else "the probe failed"

    # `RTLD_LAZY` is named, not implied: `ctypes` forces `RTLD_NOW` when the mode carries no
    # binding flag, so `RTLD_LOCAL` alone is an eager load wearing a lazy load's name.
    fatal = run(os.RTLD_LAZY | os.RTLD_LOCAL)
    if fatal:
        return f"{library.name} does not load under the runtime's own mode: {fatal}", None
    eager = run(os.RTLD_NOW | os.RTLD_LOCAL)
    return None, eager


def module_digest(directory):
    """One program's digest and byte size: every file the package wrote, by relative name."""
    digest = hashlib.sha256()
    total = 0
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == ".committed":
            continue
        data = path.read_bytes()
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(b"\0")
        digest.update(data)
        total += len(data)
    return digest.hexdigest(), total


def source_digest(package):
    """The digest of the source a cell's module was built from.

    A package's own directory is its source: the manifest that declares it, the compile script that
    runs, and whatever that script compiles. A store outlives the tree it was built from, and a cell
    said nothing about which source produced it - so a store older than the source answered a changed
    package with the module its older source built, and two runs in a row stopped inside kernels for
    exactly that (`bf16 = 3` in an entry header, a word order in a signature).

    Build scratch is excluded: the compile script runs inside the package directory, so an import
    leaves `__pycache__` behind and its bytes shift with nothing the module depends on.
    """
    digest = hashlib.sha256()
    root = package["root"]
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix == ".pyc" or "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def write_source_record(directory: Path, digest: str, target: str) -> None:
    """State, in the cell, which source built it."""
    (directory / "source.json").write_text(
        json.dumps({"digest": digest, "target": target}, indent=1) + "\n"
    )


def verify(root: Path, rows, packages, target, revision, fixture, refuse_absent: bool = True) -> int:
    """Answer whether every cell this fixture needs was built from the source this tree holds.

    This is where a stale store is refused by name. The install cannot ask the question - a
    deployment host holds the store and not the package trees it was built from - and the build
    cannot either, because a build that finds a cell already there is not the tree's statement about
    it. The fixture holds both: it names the rows, and it runs in the tree.
    """
    print(f"# verify: {root} (revision {revision}, {target})")
    print(f"# fixture: {fixture} ({len(rows)} rows)")
    stale, absent, fresh = [], [], 0
    for op, key, stated_backend in rows:
        candidates = [
            package for package in packages.get(op, [])
            if covers(package, key)
            and target in package["manifest"]["kernel"]["architectures"]
            and (stated_backend is None or package["backend"] == stated_backend)
        ]
        if not candidates:
            absent.append((op, "-", key, "no package covers this key for the target"))
            continue
        verdict, found = None, False
        for package in candidates:
            directory = root / op / package["backend"] / str(revision) / target / key
            if not (directory / ".committed").is_file():
                continue
            found = True
            record = directory / "source.json"
            if not record.is_file():
                verdict = (op, package["backend"], key,
                           "the cell states no source digest, so it cannot be told from a stale one")
                break
            stated = json.loads(record.read_text()).get("digest")
            current = source_digest(package)
            if stated != current:
                verdict = (op, package["backend"], key,
                           f"built from {str(stated)[:12]}, and this tree's source is "
                           f"{current[:12]}")
                break
        if verdict is not None:
            stale.append(verdict)
        elif found:
            fresh += 1
        else:
            absent.append((op, "-", key, "the store holds no cell for this key"))
    for op, backend, key, reason in stale:
        print(f"stale\t{op}\t{backend}\t{key}: {reason}")
    for op, backend, key, reason in absent:
        print(f"absent\t{op}\t{backend}\t{key}: {reason}")
    print(
        f"# {fresh} of {len(rows)} cells were built from this tree's source; "
        f"{len(stale)} are stale, {len(absent)} absent"
    )
    # Two callers ask two questions. A build is asked whether the store it just wrote is this tree's,
    # and a row it could not build was already named as SKIPPED there. A check is asked whether this
    # store can serve this fixture, and a store short of a row cannot.
    if stale:
        return 1
    return 1 if absent and refuse_absent else 0


def read_rows(fixture):
    """The fixture's rows, as `(operator, key, backend-or-None)`.

    A row is `op<TAB>descriptor`, and it may state a third field: the backend the deployment launches
    this key on. That field is not decoration - `paged_attention` declares `bf16`/`f16` on `trtllm`
    and `f8e4m3` on `triton`, so which package covers a key depends on the cache dtype, and a
    deployment that names one backend while the fixture states no backend can be served a cell from
    the other. A stated backend is held to: the store builds that package or refuses, and never
    quietly substitutes the other.
    """
    rows = []
    for number, line in enumerate(Path(fixture).read_text().splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) not in (2, 3) or not all(one.strip() for one in fields):
            raise SystemExit(
                f"{fixture}:{number} states {len(fields)} field(s); a row is `op<TAB>descriptor` "
                f"or `op<TAB>descriptor<TAB>backend`"
            )
        rows.append((fields[0], fields[1], fields[2] if len(fields) == 3 else None))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--root", required=True, help="the store root to assemble")
    parser.add_argument("--kernels", default="crates/ops/kernels")
    parser.add_argument("--target", default="sm_100a")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="the interpreter that runs each package's compile script; it must carry the "
             "toolchain the package declares (Triton, CuTe DSL, nvcc), so it is named rather "
             "than assumed to be this one",
    )
    parser.add_argument("--revision", type=int, default=1)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="answer whether every cell this fixture needs was built from the source this tree "
             "holds, and exit non-zero naming the stale and absent ones; builds nothing",
    )
    args = parser.parse_args()

    if args.target not in TARGETS:
        sys.exit(f"target {args.target} is not one a package here declares ({sorted(TARGETS)})")

    # Named, not assumed: a plain interpreter without the venv's Triton builds none of the PDL
    # packages and raises `module 'triton.language.extra.cuda' has no attribute 'gdc_wait'` from
    # inside a compile script, which reads like a kernel defect rather than a wrong toolchain.
    print(f"# interpreter: {args.python}")

    packages = load_packages(Path(args.kernels))
    root = Path(args.root)
    # The fixture is read once: a second read of a pipe or a stream answers nothing, and a guard that
    # inspects one read while the rows come from another cannot see an empty one.
    rows = read_rows(args.fixture)
    # A fixture with no rows is a walk that failed to print, and a store built from it is empty and
    # says so as a success: "built 0 of 0" reads like a complete store and is no evidence at all.
    if not rows:
        raise SystemExit(
            f"{args.fixture} names no rows: an empty fixture is a walk that printed nothing, and "
            f"a store built from it would be empty rather than correct"
        )

    # A key names one program, so exactly one package may claim it. Two packages claiming one key
    # would let two hosts that build the same fixture store different modules under the same name,
    # with nothing in the build log saying so — and the loader, which keys modules by (operator, key),
    # would then run whichever the directory walk offered last. The claim is checked over the whole
    # fixture before anything compiles, because a build that succeeds differently on another host is
    # worse than one that stops.
    claims = {}
    for op, key, stated_backend in rows:
        claiming = [
            package for package in packages.get(op, [])
            if covers(package, key)
            and (stated_backend is None or package["backend"] == stated_backend)
            and args.target in package["manifest"]["kernel"]["architectures"]
        ]
        if len(claiming) > 1:
            claims.setdefault((op, key), []).append(claiming)
    if claims:
        report = []
        for (op, key), _ in sorted(claims.items()):
            directories = " and ".join(
                str(package["root"]) for package in packages[op] if covers(package, key)
                and args.target in package["manifest"]["kernel"]["architectures"]
            )
            report.append(f"  {op}\t{key}\n    claimed by {directories}")
        sys.exit(
            "the store cannot be built: a key must name one program, and these name two or more\n"
            + "\n".join(report)
        )

    if args.verify:
        sys.exit(verify(root, rows, packages, args.target, args.revision, args.fixture))

    built, skipped = [], []
    for op, key, stated_backend in rows:
        candidates = [
            package for package in packages.get(op, [])
            if covers(package, key)
            and args.target in package["manifest"]["kernel"]["architectures"]
            and (stated_backend is None or package["backend"] == stated_backend)
        ]
        if not candidates:
            skipped.append(
                (
                    op,
                    key,
                    f"no package covers this key for the target (this row states the backend "
                    f"{stated_backend!r})"
                    if stated_backend
                    else "no package covers this key for the target",
                )
            )
            continue
        # A package may cover a key by its dimensions and still refuse the descriptor's variant
        # (the FA4 attention backend serves prefill only), so every candidate is tried before the
        # key is reported uncovered.
        failures = []
        for package in candidates:
            directory = root / op / package["backend"] / str(args.revision) / args.target / key
            # A cell exists only once it has committed. A failed attempt can leave a directory with
            # a manifest and a source and no `.committed` mark, and a later reader cannot tell that
            # from a cell built by an older revision — so an attempt that created one removes it.
            existed = directory.exists()
            failure = build(package, key, args.target, directory, args.python)
            if failure:
                if not existed:
                    shutil.rmtree(directory, ignore_errors=True)
                failures.append(f"{package['backend']}: {failure}")
                continue
            (directory / ".committed").write_text("")
            write_source_record(directory, source_digest(package), args.target)
            built.append((op, package["backend"], key))
            break
        else:
            skipped.append((op, key, "; ".join(failures)))
            continue
    # A store that built nothing is not a store: the count line says "0 of N", and a reader who
    # stops at it can carry on with an empty directory. The first failure is the evidence, and a
    # missing capability in the interpreter is the common cause.
    if not built:
        prune_empty_directories(root)
        first = skipped[0][2] if skipped else "no candidate package"
        sys.exit(
            f"the store is empty: none of {len(rows)} keys built for {args.target} with "
            f"{args.python}\n  first failure: {skipped[0][0]} {skipped[0][1]}: {first}"
            if skipped else f"the store is empty: no package covers a key of {args.fixture}"
        )
    print(
        f"# built {len(built)} of {len(rows)} keys into {root} "
        f"(revision {args.revision}, {args.target})"
    )
    # The fixture is named in the summary because a build is evidence of exactly one file: two
    # fixtures built into one log read as one count, and a row of one fixture's failures then looks
    # like a failure of the other's deployment.
    print(f"# fixture: {args.fixture} ({len(rows)} rows)")
    stated = sum(1 for _, _, backend in rows if backend)
    if stated:
        # A row that states its backend is the deployment's own statement, and the store held it: the
        # count says how much of this store is that claim rather than a coverage choice.
        print(f"# {stated} of {len(rows)} row(s) state the backend their deployment launches")

    # How many of those cells hold the same program. A key field that reaches no compiled word
    # writes an identical module under its own name (tools/key_field_audit.py proves which), so the
    # store pays a cell, a directory and the compile for code it already has. The count is printed
    # rather than acted on: collapsing cells is a decision about descriptor identity, and this tool
    # only says what it would save.
    digests = {}
    for op, backend, key in built:
        directory = root / op / backend / str(args.revision) / args.target / key
        digest, size = module_digest(directory)
        digests.setdefault(digest, []).append((op, key, size))
    distinct = len(digests)
    duplicated = len(built) - distinct
    cells_bytes = sum(one[2] for group in digests.values() for one in group)
    module_bytes = sum(group[0][2] for group in digests.values())
    if duplicated:
        print(
            f"# {len(built)} cells hold {distinct} distinct programs: {duplicated} cells duplicate one "
            f"({cells_bytes} bytes stored, {module_bytes} distinct)"
        )
    else:
        print(f"# {len(built)} cells, {distinct} distinct programs")
    for op, backend, key in built:
        print(f"built\t{op}\t{backend}\t{key}")
    for op, key, reason in skipped:
        print(f"SKIPPED\t{op}\t{reason}\t{key}")
    # Every build ends by stating the store's age, so a cell that another source built cannot be read
    # as this one's: the check has a caller by default rather than only when someone remembers it.
    verdict = verify(root, rows, packages, args.target, args.revision, args.fixture,
                     refuse_absent=False)
    if verdict:
        sys.exit(verdict)
    # A skipped key may have left an empty chain behind; the store holds cells, not the shape of an
    # attempt.
    prune_empty_directories(root)
    return 0 if not skipped else 1


if __name__ == "__main__":
    sys.exit(main())
