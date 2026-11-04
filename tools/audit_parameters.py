"""Check every operator's packed parameters against the module its package would build.

A launch hands the driver one pointer per parameter, in order, and stops at the module's count: an
extra trailing word is inert, but a word that is missing or of the wrong kind shifts every parameter
after it, so a scalar is read where the program expects a device address. Nothing catches that before
the launch because Triton's metadata record carries no signature (`ops::CubinMetadata` reads one when a
package writes it, and none does yet), so this script makes the same comparison on the host.

For each fixture row it reads the operator's packed words from its `launch` in `crates/ops/src` and the
module's runtime signature by *executing* the package's own `build.py` `spec()` with the row's key --
so the signature is the one the compile script would hand the driver, constexprs removed.

    python3 tools/audit_parameters.py [--fixture notes/programs/qwen3-moe-keys.txt]

A row whose package cannot be imported on this host (a cutlass or cute package, or one whose driver is
not `_compile`) is reported as unchecked rather than as passing: those packages take a `#[repr(C)]`
call struct instead of a parameter array, and the comparison below does not apply to them.
"""

import argparse
import importlib.util
import pathlib
import re
import sys
import tomllib

OPS = pathlib.Path("crates/ops/src")
KERNELS = pathlib.Path("crates/ops/kernels")

#: How each packing method's argument is read by the driver.
KINDS = {"view": "ptr", "pointer": "ptr", "scratch": "scratch"}


def impl_blocks(text):
    """Every `impl Op for X { ... }` block, by brace matching."""
    for match in re.finditer(r"impl Op for (\w+) \{", text):
        depth = 0
        for index in range(match.end() - 1, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    yield match.group(1), text[match.start() : index + 1]
                    break


def packed(operator):
    """The words an operator packs, in order, and the file it packs them in."""
    for path in OPS.rglob("*.rs"):
        text = path.read_text()
        if f'"{operator}"' not in text:
            continue
        for _, body in impl_blocks(text):
            if f"const NAME: &'static str = \"{operator}\"" not in body:
                continue
            match = re.search(r"fn launch\(", body)
            if not match:
                continue
            launch = body[match.start() :]
            following = re.search(r"\n    fn ", launch)
            launch = launch[: following.start()] if following else launch
            kinds = [
                KINDS.get(call.group(1), "scalar")
                for call in re.finditer(
                    r"\.(view|pointer|u64|u32|i32|i64|f32|flag|scratch)\(", launch
                )
            ]
            return kinds, path
    return None, None


def signature(operator, key):
    """The module's runtime signature, from the build script that would produce it."""
    for manifest_path in KERNELS.glob(f"{operator}/*/package.toml"):
        source = manifest_path.parent / "src"
        manifest = tomllib.loads(manifest_path.read_text())
        dimensions = manifest.get("kernel", {}).get("descriptors", {}).get("dimensions", {})
        sys.path.insert(0, str(source))
        try:
            sys.modules.pop("_compile", None)
            spec = importlib.util.spec_from_file_location(
                f"build_{operator}_{manifest_path.parent.parent.name}", source / "build.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            import _compile

            fields = _compile.parse_key(key, dimensions)
            described = module.spec(fields, manifest.get("parameters", {}), "sm_100a")
            return [(name, ty) for name, ty in described["signature"].items() if ty != "constexpr"]
        except Exception as error:  # noqa: BLE001 - any failure means "not checked", not "passes"
            return f"{type(error).__name__}: {error}"
        finally:
            sys.path.remove(str(source))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="notes/programs/qwen3-moe-keys.txt")
    arguments = parser.parse_args()

    rows = [
        line.split("\t")
        for line in pathlib.Path(arguments.fixture).read_text().strip().split("\n")
    ]
    checked = mismatched = 0
    for operator, key in rows:
        kinds, rust = packed(operator)
        if kinds is None:
            print(f"  ?  {operator:16} no `impl Op` in {OPS} ({rust})")
            continue
        module = signature(operator, key)
        if isinstance(module, str):
            print(f"  !  {operator:16} not checked: {module}")
            continue
        if module is None:
            print(f"  ?  {operator:16} no package")
            continue
        checked += 1
        expected = ["ptr" if ty.startswith("*") else "scalar" for _, ty in module]
        head, tail = kinds[: len(expected)], kinds[len(expected) :]
        if head == expected:
            print(f"OK  {operator:16} {len(kinds)} words, module {len(expected)}, tail {tail}")
        else:
            mismatched += 1
            print(f"BAD {operator:16} module wants {expected}, operator packs {head}")
    print(f"\n{checked} checked, {mismatched} misaligned, {len(rows) - checked} not checked")
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
