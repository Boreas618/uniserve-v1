"""Check an operator's packed parameters against the parameters its module actually takes.

A launch hands the driver one pointer per parameter and no count of its own: the driver reads as many
values as the cubin declares. Two facts make that dangerous, and neither is checked before the launch.

* **A Triton module takes two parameters beyond its declared signature.** The compiler's launcher
  appends a global scratch address and a profile scratch address, and the cubin declares both whatever
  their sizes are, so `embedding`'s seven-parameter signature is nine parameters in its PTX.
* **A packed word the module does not declare does not sit idle.** It takes the position of a parameter
  that follows it, so the driver reads a flag or a width where a device address belongs.

The ground truth here is the PTX: the package's `build.py` `spec()` is executed for the fixture key and
the module compiled (no GPU needed), and the `.param` lines are counted. The operator's packed words are
read from its `launch` in `crates/ops/src`, separating the words it always packs from those it packs
only under a condition (`rotary_factors` appends its section widths only for a sectioned rotation), so
the comparison is a range rather than a number.

    python3 tools/audit_parameters.py --fixture <key file>

Rows whose package cannot be imported here (cutlass, cute and native packages, whose contract is a
`#[repr(C)]` call structure) are reported as unchecked rather than as passing.
"""

import argparse
import importlib.util
import os
import pathlib
import re
import sys
import tempfile
import tomllib

# The compiler picks `ptxas` off `PATH`, and the system one may not know this host's architecture;
# the CUDA toolkit `CUDA_HOME` names, which the store is built with, does. The Triton cache goes
# outside the tree.
if not os.environ.get("CUDA_HOME"):
    sys.exit("CUDA_HOME names the CUDA toolkit whose ptxas the compiler runs")
os.environ["PATH"] = (
    os.path.join(os.environ["CUDA_HOME"], "bin") + os.pathsep + os.environ.get("PATH", "")
)
os.environ.setdefault(
    "TRITON_CACHE_DIR", str(pathlib.Path(tempfile.gettempdir()) / "uniserve-audit-cache")
)

#: This repository's operators and packages, found from this file's own place in it.
REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
OPS = REPOSITORY / "crates/ops/src"
KERNELS = REPOSITORY / "crates/ops/kernels"

#: The parameters the compiler's launcher appends to every Triton kernel.
IMPLICIT = 2

#: How each packing method's argument is read by the driver.
PACKING = re.compile(r"\.(view|pointer|u64|u32|i32|i64|f32|flag|scratch)\(")


def masked(text):
    """The source with the inside of string literals and comments blanked, length preserved.

    A `format!("{}")` in a launch body would otherwise count as code, and its braces would close the
    block being read.
    """
    characters = list(text)
    index = 0
    while index < len(text):
        if text.startswith("//", index):
            end = text.find("\n", index)
            end = len(text) if end < 0 else end
            for at in range(index, end):
                characters[at] = " "
            index = end
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            end = len(text) if end < 0 else end + 2
            for at in range(index, end):
                characters[at] = " "
            index = end
        elif text[index] == '"':
            index += 1
            while index < len(text) and text[index] != '"':
                characters[index] = " "
                index += 2 if text[index] == "\\" else 1
            index += 1
        else:
            index += 1
    return "".join(characters)


def impl_blocks(text):
    """Every `impl Op for X { ... }` block, by brace matching on the masked source."""
    masked_text = masked(text)
    for match in re.finditer(r"impl Op for (\w+) \{", masked_text):
        depth = 0
        for index in range(match.end() - 1, len(masked_text)):
            if masked_text[index] == "{":
                depth += 1
            elif masked_text[index] == "}":
                depth -= 1
                if depth == 0:
                    yield match.group(1), text[match.start() : index + 1]
                    break


def packed(operator):
    """How many words an operator packs: (always, at most), and the file it packs them in."""
    for path in OPS.rglob("*.rs"):
        text = path.read_text()
        if f'"{operator}"' not in text:
            continue
        for _, body in impl_blocks(text):
            named = re.search(r"const NAME: &'static str = \"([\w]+)\"", body)
            if not named or named.group(1) != operator:
                continue
            match = re.search(r"fn launch\(", body)
            if not match:
                continue
            launch = body[match.start() :]
            following = re.search(r"\n    fn ", launch)
            launch = launch[: following.start()] if following else launch
            depth = 0
            always = conditional = 0
            looped = False
            for index, character in enumerate(launch):
                if character == "{":
                    depth += 1
                elif character == "}":
                    depth -= 1
                elif character == "." and PACKING.match(launch, index):
                    # The launch function's body is depth 1; anything deeper is inside an `if` or a
                    # `for`, and a loop's word count is not a fact about the source.
                    if depth <= 1:
                        always += 1
                    elif re.search(r"for [^\n]*\{", launch[:index][-200:]) and "for " in launch[:index][-200:]:
                        looped = True
                    else:
                        conditional += 1
            return always, None if looped else always + conditional, path
    return None, None, None



def split_key_or_none(dimensions, key):
    """The key's pairs when every dimension it names is declared, else None."""
    if not dimensions:
        return None
    names = sorted(dimensions, key=len, reverse=True)
    rest, pairs = key, []
    while rest:
        name = next((n for n in names if rest.startswith(n) and rest[len(n):].startswith("_")), None)
        if name is None:
            return None
        after = rest[len(name) + 1:]
        ends = [end for end in (after.find("_" + n + "_") for n in names) if end >= 0]
        end = min(ends) if ends else len(after)
        if not after[:end]:
            return None
        pairs.append((name, after[:end]))
        rest = after[end:].lstrip("_")
    return pairs

def module_parameters(operator, key):
    """The parameter count a module's PTX declares, and its declared signature's length."""
    for manifest_path in KERNELS.glob(f"{operator}/*/package.toml"):
        source = manifest_path.parent / "src"
        if not (source / "build.py").exists():
            continue
        manifest = tomllib.loads(manifest_path.read_text())
        descriptors = manifest.get("kernel", {}).get("descriptors", {})
        # A package may declare one table of shapes or one per shape (`[[kernel.descriptors]]`);
        # a key is parsed against the table of the shape it covers.
        if isinstance(descriptors, list):
            tables = descriptors
        else:
            tables = [descriptors] if descriptors else []
        dimensions = {}
        for table in tables:
            names = table.get("dimensions") or {}
            if split_key_or_none(names, key) is not None:
                dimensions = names
                break
        if not dimensions and tables:
            dimensions = tables[0].get("dimensions") or {}
        sys.path.insert(0, str(source))
        try:
            sys.modules.pop("_compile", None)
            spec_module = importlib.util.spec_from_file_location("build", source / "build.py")
            module = importlib.util.module_from_spec(spec_module)
            spec_module.loader.exec_module(module)
            import _compile

            fields = _compile.parse_key(key, dimensions)
            described = module.spec(fields, manifest.get("parameters", {}), "sm_100a")
            import triton
            from triton.backends.compiler import GPUTarget
            from triton.compiler.compiler import ASTSource

            compiled = triton.compile(
                ASTSource(
                    described["kernel"], described["signature"], described["constexprs"]
                ),
                target=GPUTarget("cuda", 100, 32),
                options=described["options"],
            )
            declared = sum(
                1 for ty in described["signature"].values() if ty != "constexpr"
            )
            parameters = sum(
                1
                for line in compiled.asm["ptx"].splitlines()
                if line.strip().startswith(".param")
            )
            return parameters, declared
        except SystemExit as error:
            # The package's own key reader refused the key: the row is not covered by that package, so
            # there is no module to compare against.
            return f"not covered: {error}", None
        except KeyError:
            # A package whose build script describes its module another way (a `#[repr(C)]` call
            # structure rather than a parameter list) has no signature to compare against.
            return "no parameter signature in this package", None
        except Exception as error:  # noqa: BLE001 - "not checked" is not "passes"
            return f"{type(error).__name__}: {error}", None
        finally:
            sys.path.remove(str(source))
    return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, help="the key file whose rows are audited")
    arguments = parser.parse_args()

    rows = [
        line.split("\t")
        for line in pathlib.Path(arguments.fixture).read_text().strip().split("\n")
    ]
    print(f"# {len(rows)} keys of {arguments.fixture}")
    checked = bad = 0
    for operator, key in rows:
        always, most, rust = packed(operator)
        if always is None:
            print(f"  ?  {operator:18} no `impl Op` in {OPS} ({rust})")
            continue
        parameters, declared = module_parameters(operator, key)
        if isinstance(parameters, str):
            print(f"  !  {operator:18} not checked: {parameters}")
            continue
        if parameters is None:
            print(f"  ?  {operator:18} no package")
            continue
        checked += 1
        problems = []
        if parameters != declared + IMPLICIT:
            problems.append(
                f"module declares {declared} but takes {parameters}: not {IMPLICIT} implicit"
            )
        # The operator's words plus the two the launcher appends are what the driver reads.
        if most is None:
            print(f"?   {operator:18} packs {always} words plus a loop, module takes {parameters}")
            continue
        if not always + IMPLICIT <= parameters <= most + IMPLICIT:
            problems.append(
                f"module takes {parameters}, operator packs {always + IMPLICIT}..{most + IMPLICIT}"
            )
        if problems:
            bad += 1
            print(f"BAD {operator:18} {'; '.join(problems)}")
        else:
            print(f"OK  {operator:18} packs {always}..{most}, module takes {parameters}")
    print(f"\n{checked} checked, {bad} misaligned, {len(rows) - checked} not checked")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
