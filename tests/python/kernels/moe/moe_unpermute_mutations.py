#!/usr/bin/env python3
"""The two mutations `combine_check.py` exists to catch: build each one, require a red.

    python mutations.py [--build-only] [--keep]

Each mutant is a package of its own, which `uniserve build-server build` installs and builds into
a store of its own, as the Build Server builds every package; the `uniserve` program and the Build
Server's toolchain environment (`uniserve-build-server`) are on the search path.

`combine_check.py` says the combine it measures is correct. That green is evidence only if a
*plausible-wrong* combine reads red through the same script, so this builds two of them and runs
the check against each.

The combine is NVIDIA's kernel (`csrc/nv_internal/tensorrt_llm/kernels/cuteDslKernels/moeUtils.cu`
in the FlashInfer distribution), so a mutant is one expression of that file, changed, in a copy of
the package that is compiled into a module of its own. Nothing in the tree's packages is edited:

* **`dropped-route-reads-row-zero`** — the `permuted_idx < 0` skip becomes a clamp to row zero: the
  combine adds another rank's product for a route this rank does not hold. Caught by the case whose
  routes are dropped, and by the padding case, where row zero need not even be a number.
* **`routes-read-in-reverse`** — the expanded index counts routes downward inside a token, so each
  route's scale is applied to another route's row. Caught by every case with more than one visible
  route, because the reference accumulates in the operator's own order.

Then the module `resolve` hands a runner must pass the same check, unmutated: without that control a
red mutant would say only that the harness disagrees with everything.

`--build-only` builds both mutants and stops before a device is needed — it is what proves the edits
compile, and it runs on a host with no GPU. Without it the script needs the device the module was
built for, and says so in one sentence.
"""

import argparse
import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

#: The kernel packages' root: this check lives in `tests/python/kernels/<family>/`.
KERNELS = (
    pathlib.Path(__file__).resolve().parents[4] / "crates" / "ops" / "kernels"
)
#: The package this check covers.
PACKAGE = KERNELS / "moe_unpermute" / "trtllm"

#: The target the mutants are built for: the device the check runs them on.
TARGET = "sm_100a"

#: The fixture's descriptor: the cell this package serves.
FIXTURE_KEY = "hidden_2048_top_8_experts_128_input_bf16_scales_f32_output_bf16"

#: The translation unit every mutation edits, relative to the distribution's `data` tree.
KERNEL = "csrc/nv_internal/tensorrt_llm/kernels/cuteDslKernels/moeUtils.cu"

#: Each mutation: its name, the text it replaces, the text it puts there, and why that is plausible.
MUTATIONS = [
    (
        "dropped-route-reads-row-zero",
        """      int32_t const permuted_idx = expanded_idx_to_permuted_idx[expanded_idx];
      if (permuted_idx < 0) {
        continue;
      }
""",
        """      // MUTATION: a route this rank does not hold reads the first permuted row.
      int32_t const permuted_idx = expanded_idx_to_permuted_idx[expanded_idx] < 0
                                       ? 0
                                       : expanded_idx_to_permuted_idx[expanded_idx];
""",
        "a `-1` route adds row zero's product instead of contributing nothing",
    ),
    (
        "routes-read-in-reverse",
        """      int32_t const expanded_idx = token_idx * top_k + k;
""",
        """      // MUTATION: the routes inside a token are read in reverse order.
      int32_t const expanded_idx = token_idx * top_k + (top_k - 1 - k);
""",
        "each route's scale is applied to another route's row",
    ),
]


def package() -> pathlib.Path:
    """The package this script belongs to."""
    return PACKAGE


def distribution() -> pathlib.Path:
    """The FlashInfer `data` tree the build compiles from, as the build resolves it."""
    from uniserve_build_server import trtllm

    return trtllm.data_tree()


def mutant(name: str, old: str, new: str, root: pathlib.Path) -> pathlib.Path:
    """A package tree whose one package builds `old` -> `new` of the kernel instead.

    The copy is the mutant's own package, under the operator and backend of the tree's: its
    `src/build.py` is the one edited, so the tree's build module keeps compiling the vendor's kernel
    unchanged. Returns the tree, `<root>/<name>`.
    """
    tree = root / name
    copy = tree / PACKAGE.parent.name / PACKAGE.name
    shutil.copytree(
        package(),
        copy,
        ignore=shutil.ignore_patterns("test", "__pycache__", "*.o", "*.so"),
    )
    source = distribution() / KERNEL
    text = source.read_text()
    if text.count(old) != 1:
        raise SystemExit(
            f"{KERNEL} holds the mutated text {text.count(old)} times, not once: the distribution "
            f"is not the one this mutation was written against"
        )
    edited = root / f"{name}.cu"
    edited.write_text(text.replace(old, new))
    script = copy / "src/build.py"
    build = script.read_text()
    if f'"{KERNEL}"' not in build:
        raise SystemExit(
            f"{script} does not name {KERNEL}, so a mutation has nowhere to go"
        )
    script.write_text(build.replace(f'"{KERNEL}"', f'"{edited}"'))
    return tree


def build(tree: pathlib.Path, store: pathlib.Path) -> pathlib.Path:
    """Build the fixture key of a mutant's package tree into `store`; returns the module directory.

    The store is the mutant's own, so its package is revision 1 there, and the module lies under
    `modules/<op>/<backend>/1/<inputs digest>/<target>/<key>`.
    """
    keys = tree / "keys.tsv"
    keys.write_text(f"{PACKAGE.parent.name}\t{FIXTURE_KEY}\n")
    built = subprocess.run(
        [
            "uniserve",
            "build-server",
            "build",
            "--keys",
            str(keys),
            "--packages",
            str(tree),
            "--target",
            TARGET,
            "--store",
            str(store),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if built.returncode != 0:
        raise SystemExit(f"building {tree.name} failed:\n{built.stdout}\n{built.stderr}")
    found = list(
        (store / "modules" / PACKAGE.parent.name / PACKAGE.name / "1").glob(
            f"*/{TARGET}/{FIXTURE_KEY}"
        )
    )
    if len(found) != 1:
        raise SystemExit(f"{store} holds {len(found)} modules of {FIXTURE_KEY}, not one")
    return found[0]


def check(module: pathlib.Path | None) -> tuple[int, str]:
    """Run the check against a module (or the store's, with no argument): its status and last line."""
    ran = subprocess.run(
        [
            sys.executable,
            str(
                pathlib.Path(__file__).resolve().parent
                / "moe_unpermute_check.py"
            ),
        ]
        + ([str(module)] if module is not None else []),
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [
        line for line in (ran.stdout + ran.stderr).splitlines() if line.strip()
    ]
    return ran.returncode, lines[-1] if lines else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="build the mutants and stop: the edits compile, and no device is needed",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the mutants' directories after the run",
    )
    options = parser.parse_args()

    root = pathlib.Path(
        tempfile.mkdtemp(prefix="uniserve-moe-unpermute-mutations-")
    )
    try:
        built = []
        for name, old, new, why in MUTATIONS:
            tree = mutant(name, old, new, root)
            print(f"building {name}: {why}", file=sys.stderr)
            out = build(tree, root / "stores" / name)
            built.append((name, why, out))
        if options.build_only:
            print(
                f"built {len(built)} mutants under {root}: "
                + ", ".join(one[0] for one in built)
            )
            return 0

        import torch

        if not torch.cuda.is_available():
            raise SystemExit(
                "these modules are compiled kernels: the red this script requires needs the device "
                f"they were built for; its mutants are built under {root}"
            )
        control, control_line = check(None)
        if control != 0:
            raise SystemExit(
                "the unmutated module does not pass the check, so a red mutant would prove "
                f"nothing: {control_line}"
            )
        print(f"control: the unmutated module passes — {control_line}")
        for name, why, out in built:
            status, line = check(out)
            if status == 0:
                raise SystemExit(
                    f"{name} ({why}) passed the check: the check does not catch this defect"
                )
            print(f"{name}: red as it must be — {line}")
        print(
            f"ok: {len(built)} mutants, each red, on the same check the control passed"
        )
        return 0
    finally:
        if options.keep:
            print(f"kept {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
