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
    # `} else {` is one spelling of the join; a chain that writes the brace on its own line is the
    # same conditional, and reading both arms counted four words where the operator packs two.
    join = re.search(r"}\s*else\s*\{", chain)
    conditional = join is not None
    if join is not None:
        start = join.start()
        depth, index = 1, join.end()
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


def _resolve_length(kind: str, constants: dict[str, str]) -> str:
    """One field type with a named array length replaced by the number the file gives it.

    An array whose length is a constant is the same field width as a literal, so both sides' names
    are resolved before the comparison: a constant compared by its name reports a difference
    wherever the two files spell one number differently.
    """
    return re.sub(
        r"\[([\w:<>, ]+);\s*(\w+)\]",
        lambda found: (
            f"[{found.group(1)}; {constants.get(found.group(2), found.group(2))}]"
        ),
        kind,
    )


def rust_calls(op: str) -> tuple[pathlib.Path, list[tuple[str, list[tuple[str, str]]]]] | None:
    """The Rust file that impls `op`, and every call structure it declares, in declaration order.

    A native module's arguments are not packed into an `Arguments` chain: the operator fills a
    `#[repr(C)]` structure the package's entry receives, and the package mirrors it by hand in its
    header. So the two statements to compare are the structure on each side.

    A file may declare more than one (`DispatchCall` and `CombineCall` are the dispatch and the
    combine pass of one exchange), and the structure that pins a layout as constants is not a call
    structure at all. Which one a header mirrors is decided by comparison, not by name.
    """
    found = rust_launch(op)
    if found is None:
        return None
    path = found[0]
    text = path.read_text()
    constants = dict(re.findall(r"const (\w+):\s*usize\s*=\s*(\d+);", text))
    structures = []
    for match in re.finditer(r"pub struct (\w*Call) \{(.*?)\n\}", text, re.S):
        if match.group(1).endswith("Layout"):
            continue
        fields = [
            (name, _resolve_length(kind, constants))
            for name, kind in re.findall(r"^\s*pub (\w+): ([^,\n]+),", match.group(2), re.M)
        ]
        structures.append((match.group(1), fields))
    return (path, structures) if structures else None


#: The Rust and C spellings of one storage type. A layout disagreement is a field read on one side
#: and not filled on the other, and a type that widened moves every field after it - so the two
#: spellings are compared, not only the names.
_STORAGE = {
    "c_void": "void",
    "i8": "int8_t",
    "u8": "uint8_t",
    "i16": "int16_t",
    "u16": "uint16_t",
    "i32": "int32_t",
    "u32": "uint32_t",
    "i64": "int64_t",
    "u64": "uint64_t",
    "f32": "float",
    "f64": "double",
    "char": "uint8_t",
}


def _storage(kind: str, rust: bool) -> str:
    """One side's field type, in a form both sides can be compared in."""
    text = kind.strip().removeprefix("pub ").strip()
    # Rust spells an array `[T; N]` and C spells it `T[N]`; both are the same field width.
    array = re.search(r"\[(.+?);\s*(\d+)\]$", text) or re.search(
        r"^(.*?)\[(\d+)\]$", text.strip()
    )
    if array and rust == (";" in array.group(0)):
        element, count = array.group(1), array.group(2)
        return f"{_storage(element, rust)}[{count}]"
    if rust:
        pointer = re.match(r"\*(mut|const)\s+(\w+)$", text)
        if pointer:
            qualifier = "" if pointer.group(1) == "mut" else "const "
            return f"{qualifier}{_storage(pointer.group(2), rust)}*"
        return _STORAGE.get(text, text)
    text = text.replace(" *", "*").replace("* ", "*")
    qualifier = "const " if text.startswith("const ") else ""
    text = text.removeprefix("const ").strip()
    if text.endswith("*"):
        return f"{qualifier}{_storage(text[:-1].strip(), rust)}*"
    return _STORAGE.get(text, text)


def _differences(
    rust: list[tuple[str, str]], declared: list[tuple[str, str]]
) -> tuple[int, int]:
    """The field positions that differ, and the number of fields one side has and the other does not."""
    wrong = sum(
        1
        for (left_name, left), (right_name, right) in zip(rust, declared)
        if left_name != right_name or _storage(left, True) != _storage(right, False)
    )
    return wrong, abs(len(rust) - len(declared))


def header_structs(header: pathlib.Path) -> list[tuple[str, list[tuple[str, str]]]]:
    """Every field-carrying structure a header declares, named after its tag or its typedef.

    A package spells the same thing four ways - `struct uniserve_conv_call {`, a `struct ..._call`
    in `src/entry.h`, `typedef struct UniserveGemmCall { ... } UniserveGemmCall;`, and
    `typedef struct { ... } SomeParams;` - so the reader takes them all and lets the comparison
    decide which one the module receives.

    Read line by line rather than by one pattern: a pointer field (`void* out;`) and an array field
    (`int64_t input_shape[5];`) differ in where the name sits, and a single pattern that spans lines
    loses the pointer fields and then reports every later field one position early.
    """
    text = re.sub(r"//[^\n]*", "", header.read_text())
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    defines = dict(re.findall(r"#define\s+(\w+)\s+(\d+)", text))
    found = []
    for block in re.finditer(
        r"(?:typedef\s+)?struct\s*(\w*)\s*\{(.*?)\n\}", text, re.S
    ):
        typedef = re.match(r"\s*(\w+)\s*;", text[block.end() :])
        name = block.group(1) or (typedef.group(1) if typedef else "")
        fields = []
        for line in block.group(2).splitlines():
            line = line.strip()
            if not line.endswith(";"):
                continue
            field = re.search(
                r"(?:^|\s)((?:\w+[\s*]+)*?)(\w+)(\[[^\]]*\])?;$", line
            )
            if field:
                length = field.group(3) or ""
                if length:
                    symbol = length.strip("[]")
                    length = f"[{defines.get(symbol, symbol)}]"
                fields.append((field.group(2), f"{field.group(1).strip()}{length}"))
        if name and fields:
            found.append((name, fields))
    return found


#: The word a C parameter occupies, in the same vocabulary the Rust chain packs. An address is an
#: address: `view` and `pointer` differ in what the launch records, not in the word it passes.
_C_WORDS = {
    "void*": "address",
    "int": "i32",
    "int32_t": "i32",
    "int64_t": "i64",
    "unsigned": "u32",
    "unsigned int": "u32",
    "uint32_t": "u32",
    "uint64_t": "u64",
    "float": "f32",
    "double": "f64",
}

_WORDS = {"view": "address", "tensor": "address", "pointer": "address"}


def entry_parameters(source: pathlib.Path) -> tuple[str, list[str]] | None:
    """A cubin module's entry and the words its parameter list takes, in order.

    A package that ships no call structure passes its arguments as kernel parameters - `topk_softmax`
    takes `const void* scores, int score_kind, ...` - so the declared side is that list and its
    comparison is against the words the operator's `Arguments` chain packs.
    """
    text = source.read_text()
    entry = re.search(
        r"__global__\s+void\s+(\w+)\s*\(([^)]*)\)", text, re.S
    )
    if entry is None:
        return None
    parameters = [
        re.sub(r"\b(const|volatile|__restrict__)\b", " ", parameter)
        .replace("*", "*")
        .strip()
        for parameter in entry.group(2).split(",")
        if parameter.strip()
    ]
    words = []
    for parameter in parameters:
        kind = re.split(r"(?<=\w)\s+|\s+", parameter)
        spelling = " ".join(kind[:-1]).strip() if len(kind) > 1 else parameter
        spelling = re.sub(r"\s*\*\s*", "*", spelling).strip()
        if spelling.endswith("*"):
            words.append("address")
        else:
            words.append(_C_WORDS.get(spelling, spelling))
    return entry.group(1), words


def native_parameters(
    op: str, package: pathlib.Path, verbose: bool
) -> tuple[str, str] | None:
    """One package's entry parameter list against the words the operator's chain packs.

    Answers `(kind, text)` where kind is `ok`, `problem` or `gap`, or `None` when the package has no
    entry to read at all.
    """
    found = rust_launch(op)
    if found is None:
        return None
    chain = found[1]
    if not chain:
        return None
    # A chain that chooses its words in a `match` arm is not one list: counting every arm counted
    # thirteen words for `topk_softmax`, whose entry takes eleven. Which arm a call takes is a
    # descriptor's business, so the comparison stops and says so rather than picking one.
    if re.search(r"\bmatch\b", chain):
        arms = {
            tuple(packed(arm)[0])
            for arm in re.findall(r"=>\s*\{([^{}]*)\}", chain)
        }
        if len(arms) == 1 and not any(len(arm) == 0 for arm in arms):
            # Every arm packs the same words, so the arms are not a choice about the entry's
            # parameters and the list they agree on is the chain's one packing.
            # Keep the words the chain packs outside the `match` and take one arm's for the rest.
            opening = chain.index("{", chain.index("match"))
            depth, index = 1, opening + 1
            while index < len(chain) and depth:
                if chain[index] == "{":
                    depth += 1
                elif chain[index] == "}":
                    depth -= 1
                index += 1
            arm = next(iter(arms))
            chain = chain[:opening] + "{" + "".join(f".{word}()" for word in arm) + "}" + chain[index:]
        else:
            words = packed(chain)[0]
            return (
                "gap",
                f"{op}/{package.name}: the chain packs {len(words)} word(s) across "
                f"{len(arms)} distinct match arm(s), and its entry's parameter list is one arm's "
                f"packing - not compared rather than guessed",
            )
    words = packed(chain)[0]
    if not words:
        return None
    declared = None
    for source in sorted(package.rglob("*.cu")):
        declared = entry_parameters(source)
        if declared is not None:
            break
    if declared is None:
        return None
    entry, words = declared
    packed_words = [_WORDS.get(word, word) for word in packed(chain)[0]]
    if len(words) != len(packed_words):
        return (
            "problem",
            f"{op}/{package.name} {entry}: {len(words)} parameter(s) and {len(packed_words)} "
            f"packed word(s)",
        )
    for index, (declared_word, packed_word) in enumerate(zip(words, packed_words)):
        if declared_word != packed_word:
            return (
                "problem",
                f"{op}/{package.name} {entry}: parameter {index} is {declared_word} and the "
                f"chain packs {packed_word}",
            )
    if verbose:
        print(f"  {op}/{package.name}: {entry} = {len(words)} word(s), the chain agrees")
    return ("ok", "")


def native(verbose: bool) -> tuple[list[str], list[str]]:
    """Every native package's call structure, compared with the Rust structure it mirrors.

    The comparison is positional, by name and by storage type, which is what a layout disagreement
    looks like: a field read on one side and not filled on the other is uninitialised memory inside
    the module, where a Triton disagreement is only a wrong argument.

    **Which structure on each side is the one the module receives is decided by comparison, and the
    choice is printed.** A file may declare several (dispatch and combine passes; a package's params
    and its call), and a wrong pick reads exactly like a layout defect - which is how the first
    version of this reader reported a disagreement that was its own.
    """
    problems: list[str] = []
    gaps: list[str] = []
    compared = 0
    for package in sorted(KERNELS.glob("*/*")):
        op = package.parent.name
        # A Triton package's arguments are its `spec()` and the recorded signature its cell carries,
        # which the sweep compares; this mode is the native package, whose arguments are a C struct.
        if not _native_package(package):
            continue
        headers = sorted(package.rglob("*.h"))
        if not headers:
            # A package that ships no call structure may pass its arguments as kernel parameters:
            # `topk_softmax` is a cubin whose entry takes `const void* scores, int score_kind, ...`,
            # and its declared side is that list rather than a struct.
            outcome = native_parameters(op, package, verbose)
            if outcome is not None:
                kind, text = outcome
                if kind == "gap":
                    gaps.append(text)
                    continue
                compared += 1
                if kind == "problem":
                    problems.append(text)
                continue
            gaps.append(
                f"{op}/{package.name}: no header, so no call structure to compare - the module "
                f"states its arguments somewhere this reader has not found"
            )
            continue
        where = rust_calls(op)
        structures = [
            (header, name, fields)
            for header in headers
            for name, fields in header_structs(header)
        ]
        if not structures:
            gaps.append(
                f"{op}/{package.name}: {len(headers)} header(s) and no field-carrying structure"
            )
            continue
        if where is None:
            # Name what the Rust side does declare instead of only reporting that nothing was
            # found: a structure called `GemmArgs` is a different statement about a module's
            # arguments, not a missing one, and the difference is what a reader needs.
            declared = []
            source = rust_launch(op)
            if source is not None:
                declared = re.findall(r"pub struct (\w+)", source[0].read_text())
            gaps.append(
                f"{op}/{package.name}: {len(structures)} structure(s) in the header, and no Rust "
                f"call structure from the operator's name"
                + (
                    f" - {source[0].name} declares {', '.join(declared)} and no `*Call`"
                    if declared
                    else ""
                )
                + " - the reader's reach, not a disagreement"
            )
            continue
        path, rusts = where
        scored = sorted(
            (
                (_differences(rust, fields), structure, header, rust, fields)
                for header, structure, fields in structures
                for structure, rust in rusts
            ),
            key=lambda entry: entry[0],
        )
        best, structure, header, rust, fields = scored[0]
        compared += 1
        if best == (0, 0):
            if verbose:
                print(
                    f"  {op}/{package.name}: {structure} = {len(rust)} field(s), "
                    f"{header.relative_to(KERNELS)} agrees"
                )
            continue
        wrong, missing = best
        first = next(
            (
                f"position {index}: {left_name} {left} in {path.name}, "
                f"{right_name} {right} in {header.name}"
                for index, ((left_name, left), (right_name, right)) in enumerate(
                    zip(rust, fields)
                )
                if left_name != right_name
                or _storage(left, True) != _storage(right, False)
            ),
            "no position differs; the two lists differ in length",
        )
        problems.append(
            f"{op}/{package.name} {structure} vs {header.relative_to(KERNELS)}: "
            f"{wrong} position(s) differ, {missing} field(s) unpaired, from {path.name}; "
            f"best of {len(scored)} pairing(s) - {first}"
        )
    print(
        f"{compared} call structure(s) compared, {len(problems)} disagreement(s), "
        f"{len(gaps)} not compared"
    )
    return problems, gaps


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
            if len(triton) == 1:
                backend = triton[0]
            elif len(stated) == 1:
                backend = stated[0]
            else:
                # No backend can be named for this row: either every backend the operator states
                # is native, or several Triton ones exist and the row does not say which. Guessing
                # `triton` invented a package name and reported "no built cell" for a native
                # operator, so an unattributable row says so instead.
                backend = None
            finding = f"{op} (backends: {', '.join(stated)})"
        else:
            finding = f"{op}/{backend}"
        package = KERNELS / op / backend if backend else KERNELS / op
        build = package / "src/build.py"
        cell = store / op / (backend or ".")
        recorded = sorted(cell.glob(f"*/{target}/{descriptor}/metadata.json"))
        if not recorded:
            # A native package records no Triton metadata: its parameters are its entry's C
            # declaration and `launch.json` names that entry. That is a different statement rather
            # than a missing one, and it is reported as such.
            if sorted(cell.glob(f"*/{target}/{descriptor}/launch.json")):
                gaps.append(
                    f"{finding} {descriptor}: a native cell, whose parameters are its "
                    f"entry's C declaration rather than a recorded signature - `--native` "
                    f"compares that declaration with the Rust structure it mirrors"
                )
            elif backend is None:
                # The row names no backend and the operator states none a cell could be read from,
                # so the honest report is that the row is unattributable rather than unbuilt.
                gaps.append(
                    f"{finding} {descriptor}: the row names no backend, and the store holds no "
                    f"cell for it under any backend the operator states"
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
    parser.add_argument(
        "--native",
        action="store_true",
        help="compare each native package's call structure with the Rust one it mirrors",
    )
    arguments = parser.parse_args()

    if arguments.native:
        problems, gaps = native(arguments.verbose)
        for problem in problems:
            print(f"  problem: {problem}")
        for gap in gaps:
            print(f"  gap:     {gap}")
        return 1 if (arguments.check and problems) else 0

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
