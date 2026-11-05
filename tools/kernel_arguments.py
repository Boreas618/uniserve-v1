#!/usr/bin/env python3
"""Compare the three statements of one Triton kernel's argument list.

A Triton operator's arguments are written down three times, by three different hands:

1. the kernel's ``def`` in ``<op>/triton/src/*.py`` - the parameters the compiled entry reads;
2. the package's ``build.py``, whose ``spec()["signature"]`` states them **in the order the operator
   launches them**, with their types;
3. the operator's Rust ``fn launch`` in ``crates/ops/src/*.rs``, which packs one ``Arguments`` chain
   in launch order, and whose constexprs come from the descriptor key.

Nothing in the tree compares them. ``key_coverage`` says whether a package declares a descriptor
key, ``build_store`` says whether it builds, and the recorder says whether the model launches it -
and a kernel that reads its stage from its draw's slot satisfies all three. This script reports the
three orders side by side and names where they disagree.

**The package is asked, not parsed.** Each ``build.py`` is imported and its ``spec(fields,
parameters, target)`` called once per declared descriptor shape, which is what the Program Server
does: nine packages build their signature incrementally in code, so a source-level parse reads them
as having none, and a signature assembled in a loop is exactly the case where a hand-check is most
likely to be wrong.

Read-only: it opens sources, imports them and prints. ``--check`` exits 1 on any disagreement.

    tools/kernel_arguments.py                     # every Triton operator
    tools/kernel_arguments.py copy_block --verbose
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: The compile driver every package imports; a key is parsed by the package's own copy.
_COMPILE = "_compile"
KERNELS = ROOT / "crates/ops/kernels"
OPS = ROOT / "crates/ops/src"

#: The `Arguments` methods that pack one launch word. `flag` is a word like any other: Triton's
#: `pdl` is a kernel parameter the entry reads, not a launch attribute.
PACKERS = {
    "view": "tensor",
    "tensor": "tensor",
    "pointer": "pointer",
    "u32": "u32",
    "u64": "u64",
    "i32": "i32",
    "i64": "i64",
    "f32": "f32",
    "scalar": "scalar",
    "flag": "flag",
}


def _native_package(package: pathlib.Path) -> bool:
    """Whether a package states a native module, which builds no Triton cell."""
    manifest = package / "package.toml"
    if not manifest.exists():
        return False
    declared = tomllib.loads(manifest.read_text())
    # A package states its toolchain, and the one Triton cell a sweep can read is the one whose
    # toolchain is Triton's: `nvcc` and the CuTe DSL build native modules, whose parameters live in
    # an entry's C declaration.
    kind = (declared.get("toolchain") or {}).get("kind")
    return kind is not None and kind != "triton"


def load_spec(build: pathlib.Path):
    """The package's `spec` function, imported under a name of its own.

    Each package's `src` is put first on the path for the import, because `build.py` imports its
    kernel module and `_compile` by bare name. The module is loaded fresh per package so two
    packages never share one `kernels` module.
    """
    source = build.parent
    sys.path.insert(0, str(source))
    try:
        # `build.py` imports `_compile` and its kernel module by bare name, so `sys.modules` would
        # hand every package after the first the first package's copy. The copies differ between
        # operators - one carries `parse_key` and no `descriptor_fields`, another the reverse - so a
        # shared one refuses keys its own operator declares. Everything imported from a package's
        # source is dropped, which is what a fresh process per package would do.
        for cached, module in list(sys.modules.items()):
            origin = getattr(module, "__file__", None)
            if origin and str(KERNELS) in origin:
                del sys.modules[cached]
        name = f"h3_package_{build.parent.parent.parent.name}_{build.parent.parent.name}"
        found = importlib.util.spec_from_file_location(name, build)
        module = importlib.util.module_from_spec(found)
        # A package's source may print while it is imported or leave by `SystemExit`; neither is
        # this report's output and neither may end it, so the import's stdout is captured and every
        # base exception becomes a gap against that package.
        noise = io.StringIO()
        try:
            with contextlib.redirect_stdout(noise):
                found.loader.exec_module(module)
        except BaseException as error:  # noqa: BLE001 - an unimportable package is a gap
            raise RuntimeError(f"{type(error).__name__}: {error}") from None
        # The purge above is the repair; this is the statement that it held. A package that bound
        # another package's `_compile` would compare a key against a parser its own operator never
        # used, and the symptom is a refusal, not a wrong number - so the invariant is checked where
        # the binding happens rather than trusted to the purge being repeated correctly.
        bound = sys.modules.get(_COMPILE)
        origin = getattr(bound, "__file__", None)
        if origin and source not in pathlib.Path(origin).parents:
            raise RuntimeError(
                f"{build} bound `_compile` from {origin}, which is not this package's; two "
                f"packages in one process share a bare module name unless each is loaded fresh"
            )
        return module.spec
    finally:
        sys.path.remove(str(source))


def shapes(package: pathlib.Path) -> list[dict]:
    """The descriptor shapes a package declares, flattened."""
    manifest = tomllib.loads((package / "package.toml").read_text())
    kernel = manifest.get("kernel", {})
    declared = kernel.get("descriptors", {})
    declared = declared if isinstance(declared, list) else [declared]
    return [
        shape
        for shape in declared
        if isinstance(shape, dict) and "dimensions" in shape
    ]


def parameters(package: pathlib.Path) -> tuple[dict, str]:
    """The build parameters a package states, and its first architecture.

    `[parameters]` is a top-level table in most packages and `[kernel.parameters]` in some; both are
    read, because the compile script is handed `manifest.get("parameters", {})` and a package that
    put them under `[kernel]` would otherwise be compiled with none here and fail where the server
    succeeds.
    """
    manifest = tomllib.loads((package / "package.toml").read_text())
    kernel = manifest.get("kernel", {})
    stated = manifest.get("parameters") or kernel.get("parameters") or {}
    architectures = kernel.get("architectures") or ["sm_100a"]
    return stated, architectures[0]


def sampled(dimensions: dict) -> list[dict]:
    """A shape's baseline fields, then one field varied at a time.

    Combinatorially complete sampling is not wanted - a package with six dimensions of four values
    is four thousand compilations - and it is not needed: what a signature's length depends on is
    one declared value at a time (a pointer added by `affine`, a form that takes an extra stride).
    """
    baseline = {name: values[0] for name, values in dimensions.items()}
    seen = [baseline]
    for name, values in dimensions.items():
        for value in values[1:]:
            fields = dict(baseline)
            fields[name] = value
            if fields not in seen:
                seen.append(fields)
    return seen


def kernel_entry(spec: dict):
    """The JIT function a spec names, when it names one."""
    entry = spec.get("kernel")
    while isinstance(entry, (tuple, list)):
        entry = entry[0]
    return entry


def rust_launch(op: str) -> tuple[pathlib.Path, str] | None:
    """The Rust file that impls `op`, and the text of the chain its `fn launch` packs."""
    needle = re.compile(rf'const NAME: &\'static str = "{re.escape(op)}";')
    next_impl = re.compile(r"const NAME: &'static str = ")
    for path in sorted(OPS.rglob("*.rs")):
        text = path.read_text()
        found = needle.search(text)
        if found is None:
            continue
        # The operator's own impl, not the file: `activation.rs` and `segments.rs` hold several, and
        # a search that ran past this one would attribute a neighbour's chain to this operator.
        following = next_impl.search(text, found.end())
        body = text[found.end() : following.start() if following else len(text)]
        start = body.find("Arguments::new()")
        if start < 0:
            return path, ""
        end = body.find("kernel.launch", start)
        return path, body[start : end if end > 0 else len(body)]
    return None


def packed(chain: str) -> tuple[list[str], bool, bool]:
    """The word kinds one chain packs, whether a loop packs an unstated run, and whether an `else`.

    A chain may choose its word types by descriptor - `act_and_mul` packs three `u64` words for its
    wide form and three `i32` words for its narrow one - so both arms of an `if` are not two sets of
    arguments. The `if` arm is read and the `else` arm is dropped, and the conditional is reported so
    a reader knows a second packing exists.
    """
    conditional = "} else {" in chain
    if conditional:
        start = chain.index("} else {")
        depth, index = 1, start + len("} else {")
        while index < len(chain) and depth:
            if chain[index] == "{":
                depth += 1
            elif chain[index] == "}":
                depth -= 1
            index += 1
        chain = chain[:start] + chain[index:]
    words = [
        PACKERS[name]
        for name in re.findall(r"\.([a-z0-9_]+)\(", chain)
        if name in PACKERS
    ]
    dynamic = bool(re.search(r"\bfor\b[^{]*\{", chain))
    return words, dynamic, conditional


def report(op: str, verbose: bool) -> tuple[list[str], list[str]]:
    """One operator's three lists: `(problems, gaps)`."""
    problems: list[str] = []
    gaps: list[str] = []
    for backend in sorted(
        p.name for p in (KERNELS / op).iterdir() if p.is_dir()
    ):
        package = KERNELS / op / backend
        if backend != "triton":
            if verbose:
                print(
                    f"  {backend:9} native: its arguments are its entry's C signature"
                )
            continue
        build = package / "src/build.py"
        if not build.exists():
            gaps.append(f"{op}/{backend}: no build.py, so no signature to read")
            continue
        declared, target = parameters(package)
        try:
            spec = load_spec(build)
        except Exception as error:  # noqa: BLE001 - a package that cannot be imported is a gap
            gaps.append(
                f"{op}/{backend}: `spec` could not be imported ({error})"
            )
            continue
        launch = rust_launch(op)
        for shape in shapes(package):
            for fields in sampled(shape["dimensions"]):
                stated = read_shape(
                    spec, fields, declared, target, op, backend, gaps
                )
                if stated is None:
                    continue
                signature = stated.get("signature") or {}
                order = list(signature)
                # Two packages state their compile-time values two ways: marked `"constexpr"` inside
                # the signature, or listed in a separate `constexprs` table. Both are the same
                # statement and `_compile` hands Triton both, so they are read together.
                marked = [
                    name
                    for name, kind in signature.items()
                    if kind == "constexpr"
                ]
                separate = [
                    name
                    for name in (stated.get("constexprs") or {})
                    if name not in marked
                ]
                constexprs = marked + separate
                words = [name for name in order if name not in constexprs]
                entry = kernel_entry(stated)
                names = list(getattr(entry, "arg_names", []) or [])
                if names and sorted(names) != sorted(order + separate):
                    problems.append(
                        f"{op}/{backend} {fields}: the signature states {order}"
                        f"{' + ' + str(separate) if separate else ''} and the kernel reads {names}"
                    )
                if launch is None:
                    gaps.append(
                        f"{op}/{backend}: no `Op` impl under crates/ops/src names it"
                    )
                    continue
                path, chain = launch
                kinds, dynamic, conditional = packed(chain)
                if not dynamic and len(kinds) != len(words):
                    problems.append(
                        f"{op}/{backend} {fields}: {path.name} packs {len(kinds)} word(s) {kinds} "
                        f"and the signature states {len(words)} {words}"
                    )
                if dynamic and len(kinds) >= len(words):
                    problems.append(
                        f"{op}/{backend} {fields}: {path.name} packs {len(kinds)} stated word(s) "
                        f"plus a loop, and the signature states {len(words)} {words}"
                    )
                if verbose:
                    print(f"{op}/{backend} {fields}")
                    print(f"  kernel    {names or '(no arg_names)'}")
                    print(f"  signature {order}  constexpr {constexprs}")
                    print(
                        f"  packed    {kinds}{' + a loop' if dynamic else ''}"
                        f"{' (one of two arms)' if conditional else ''}  ({path.name})"
                    )
    return problems, gaps


def read_shape(spec, fields, declared, target, op, backend, gaps):
    """One descriptor shape's `spec` output, or `None` when the package refuses it."""
    try:
        return spec(fields, declared, target)
    except BaseException as error:  # noqa: BLE001 - a shape a package refuses is a gap to report
        # `SystemExit` is how a package's `spec` refuses a shape it does not serve, so it is caught
        # with the rest: a refusal is a gap against that package, not the end of the report.
        gaps.append(
            f"{op}/{backend} {fields}: `spec` refused this shape ({error})"
        )
        return None


def sweep(
    fixture: pathlib.Path, store: pathlib.Path, target: str
) -> tuple[list[str], list[str]]:
    """Compare every built cell's recorded signature with the package that wrote it, in order.

    This is `ops::Kernel::check`'s comparison made **before a launch exists**: the cell's
    `metadata.json` is what `check` holds the packed words to, and the package's own `spec()` is what
    the compile script handed Triton. Both are keyed by one descriptor, so the comparison is made for
    **the phase the key names** rather than for the union of an operator's arms - which is the one
    thing a source-level reader gets wrong (`paged_attention` packs two phases' lists from one table).
    Each position is compared by name, so a disagreement says which word is where.
    """
    problems: list[str] = []
    gaps: list[str] = []
    cells = 0
    for line in fixture.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = line.split("\t")
        op, descriptor = row[0], row[1]
        backend = row[2] if len(row) > 2 else None
        if backend is None:
            # A fixture row may name only the operator and its descriptor: 152 rows over 46
            # operators would repeat a backend on every line, and the operator's directory states
            # it once. Triton's is the default a cell can be read from; a native operator has no
            # cell at all, which the gap below then says in those words.
            stated = sorted(
                p.name for p in (KERNELS / op).iterdir() if p.is_dir()
            )
            triton = [
                name
                for name in stated
                if not _native_package(KERNELS / op / name)
            ]
            backend = (
                triton[0]
                if len(triton) == 1
                else (stated[0] if len(stated) == 1 else "triton")
            )
            finding = f"{op} (backends: {', '.join(stated)})"
        else:
            finding = f"{op}/{backend}"
        package = KERNELS / op / backend
        build = package / "src/build.py"
        cell = store / op / backend
        recorded = sorted(cell.glob(f"*/{target}/{descriptor}/metadata.json"))
        if not recorded:
            # A native package records no Triton metadata: its parameters are its entry's C
            # declaration and `launch.json` names that entry. That is a different statement rather
            # than a missing one, and it is reported as such.
            if sorted(cell.glob(f"*/{target}/{descriptor}/launch.json")):
                gaps.append(
                    f"{finding} {descriptor}: a native cell, whose parameters are its "
                    f"entry's C declaration rather than a recorded signature"
                )
            elif _native_package(package):
                # A native package is not a Triton cell, so no build produces a `metadata.json` for
                # it: its parameters are its entry's C declaration and `launch.json` names that
                # entry. Nothing here can compare that, and saying "not built" would under-describe
                # a gap that is a different statement rather than a missing one.
                gaps.append(
                    f"{finding} {descriptor}: a native package, whose parameters are its "
                    f"entry's C declaration rather than a recorded signature"
                )
            else:
                # A row that names no backend is attributed only when the store holds its cell:
                # the store is the build's own record, so which backend a row belongs to is the
                # store's answer and not this reader's guess.
                gaps.append(
                    f"{finding} {descriptor}: no built cell under {store}"
                )
            continue
        cells += 1
        if not build.exists():
            gaps.append(
                f"{op}/{backend}: a built cell with no package under crates/ops/kernels"
            )
            continue
        try:
            spec = load_spec(build)
            manifest = tomllib.loads((package / "package.toml").read_text())
            compile_mod = spec.__globals__[_COMPILE]
        except Exception as error:  # noqa: BLE001
            gaps.append(
                f"{op}/{backend}: the package could not be read ({error})"
            )
            continue
        declared = manifest["kernel"]["descriptors"]
        declared = declared if isinstance(declared, list) else [declared]
        fields = None
        for shape in declared:
            try:
                fields = compile_mod.parse_key(
                    descriptor, shape.get("dimensions", {})
                )
                break
            except BaseException:  # noqa: BLE001 - a shape that does not parse this key
                continue
        if fields is None:
            gaps.append(
                f"{op}/{backend} {descriptor}: no declared shape parses this key"
            )
            continue
        parameters_of, first_target = parameters(package)
        try:
            stated = spec(fields, parameters_of, target)
        except BaseException as error:  # noqa: BLE001
            gaps.append(
                f"{op}/{backend} {descriptor}: `spec` refused it ({error})"
            )
            continue
        constexprs = set(stated.get("constexprs") or {})
        expected = [
            (name, kind)
            for name, kind in (stated.get("signature") or {}).items()
            if kind != "constexpr" and name not in constexprs
        ]
        module = json.loads(recorded[0].read_text()).get("signature") or {}
        actual = list(module.items())
        if len(actual) != len(expected):
            problems.append(
                f"{op}/{backend} {descriptor}: the module records {len(actual)} parameter(s) "
                f"{[name for name, _ in actual]} and the package states {len(expected)} "
                f"{[name for name, _ in expected]}"
            )
            continue
        for position, (
            (packed_name, packed_ty),
            (kernel_name, kernel_ty),
        ) in enumerate(zip(actual, expected)):
            if packed_name != kernel_name:
                problems.append(
                    f"{op}/{backend} {descriptor}: position {position}: the kernel reads "
                    f"`{kernel_name}`, the package states `{packed_name}`"
                )
                break
            if packed_ty != kernel_ty:
                problems.append(
                    f"{op}/{backend} {descriptor}: position {position}: `{kernel_name}` is "
                    f"`{kernel_ty}` in the kernel and `{packed_ty}` in the module"
                )
                break
    if cells:
        print(
            f"{cells} cell(s) compared in order against the packages that wrote them",
            file=sys.stderr,
        )
    return problems, gaps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "operator", nargs="*", help="operators to read; default all"
    )
    parser.add_argument(
        "--check", action="store_true", help="exit 1 on any disagreement"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="print the three lists"
    )
    parser.add_argument(
        "--fixture",
        help="a key fixture: each row is `op<TAB>descriptor[<TAB>backend]`",
    )
    parser.add_argument(
        "--store", help="the built store whose cells --fixture is compared to"
    )
    parser.add_argument(
        "--target",
        default="sm_100a",
        help="the architecture the store was built for",
    )
    arguments = parser.parse_args()

    if arguments.fixture:
        if not arguments.store:
            print(
                "--fixture needs --store: the cells are what the comparison reads"
            )
            return 2
        problems, gaps = sweep(
            pathlib.Path(arguments.fixture),
            pathlib.Path(arguments.store),
            arguments.target,
        )
        print(f"{len(problems)} problem(s), {len(gaps)} gap(s)")
        for problem in problems:
            print(f"  problem: {problem}")
        for gap in gaps:
            print(f"  gap:     {gap}")
        return 1 if (arguments.check and problems) else 0

    operators = arguments.operator or sorted(
        p.name for p in KERNELS.iterdir() if p.is_dir()
    )
    problems: list[str] = []
    gaps: list[str] = []
    for op in operators:
        if not (KERNELS / op).is_dir():
            problems.append(f"{op}: no such operator under crates/ops/kernels")
            continue
        found, missing = report(op, arguments.verbose)
        problems.extend(found)
        gaps.extend(missing)

    # A defect of one operator repeats for every descriptor shape it declares, so the report is
    # keyed by cause: the field set is dropped from the key and the number of shapes that showed it
    # is counted, which is what a reader needs to judge whether one fix settles it.
    causes: dict[str, int] = {}
    for problem in problems:
        key = re.sub(r" \{[^}]*\}", " {..}", problem)
        causes[key] = causes.get(key, 0) + 1
    print(
        f"{len(operators)} operator(s) read, {len(problems)} shape(s) with a problem in "
        f"{len(causes)} cause(s), {len(gaps)} gap(s)"
    )
    for cause, count in causes.items():
        print(f"  problem: {cause}  [{count} shape(s)]")
    for gap in gaps:
        print(f"  gap:     {gap}")
    if arguments.check and problems:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
