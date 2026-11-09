"""The deepep package's compile script: one descriptor -> one module.

The descriptor fixes every template argument of DeepEP's kernels, so this compiles
them here, and a serving host compiles nothing. The Program Server runs this script
with its working directory the package root and `UNISERVE_DESCRIPTOR`,
`UNISERVE_TARGET`, `UNISERVE_PACKAGE` and `UNISERVE_OUT` in the environment.
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import sysconfig

SOURCES = ["src/exchange.cu"]

ENTRIES = [
    "uniserve_expert_exchange_launch",
    "uniserve_expert_exchange_create",
    "uniserve_expert_exchange_step",
    "uniserve_expert_exchange_suspend",
    "uniserve_expert_exchange_restore",
    "uniserve_expert_exchange_release",
    "uniserve_expert_exchange_info",
]


def arguments():
    """The package, descriptor, target and output directory of one build."""
    reader = argparse.ArgumentParser(description=__doc__)
    reader.add_argument("--package")
    reader.add_argument("--descriptor")
    reader.add_argument("--target")
    reader.add_argument("--out")
    known = reader.parse_args()
    values = {
        "package": known.package or os.environ.get("UNISERVE_PACKAGE"),
        "descriptor": known.descriptor or os.environ.get("UNISERVE_DESCRIPTOR"),
        "target": known.target or os.environ.get("UNISERVE_TARGET"),
        "out": known.out or os.environ.get("UNISERVE_OUT"),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise SystemExit(f"missing {', '.join(missing)}")
    return (
        pathlib.Path(values["package"]).resolve(),
        values["descriptor"],
        values["target"],
        pathlib.Path(values["out"]),
    )


def constants(descriptor):
    """The compile-time values a descriptor states."""
    found = re.fullmatch(
        r"mode_(?:dispatch|combine)_hidden_(\d+)_intermediate_\d+_top_(\d+)_experts_(\d+)"
        r"_max_tokens_(\d+)_ranks_(\d+)_attention_(\d+)_dtype_bf16",
        descriptor,
    )
    if found is None:
        raise SystemExit(f"this package serves no descriptor {descriptor}")
    hidden, top, experts, max_tokens, ranks, attention = found.groups()
    return {
        "UNISERVE_HIDDEN": hidden,
        "UNISERVE_TOP": top,
        "UNISERVE_EXPERTS": experts,
        "UNISERVE_MAX_TOKENS": max_tokens,
        "UNISERVE_RANKS": ranks,
        "UNISERVE_ATTENTION_RANKS": attention,
    }


def nccl_library():
    """The directory of a `libnccl.so.2` to name in the module's needed list.

    The module takes the serving process's NCCL at run time, which
    `tools/check_nccl.sh` holds to the project's version; this one only states the
    library's name to the linker.
    """
    site = pathlib.Path(sysconfig.get_paths()["purelib"])
    directory = site / "nvidia" / "nccl" / "lib"
    if not (directory / "libnccl.so.2").exists():
        raise SystemExit(f"no libnccl.so.2 in {directory}")
    return directory


def main():
    """Compile the exchange for one descriptor into ``module.so``."""
    package, descriptor, target, out = arguments()
    out.mkdir(parents=True, exist_ok=True)
    # The nvcc the Program Server checked against this package's declared version.
    compiler = os.environ["UNISERVE_NVCC"]
    defines = [f"-D{name}={value}" for name, value in constants(descriptor).items()]
    command = [
        compiler,
        "-O3",
        "-std=c++20",
        f"-arch={target}",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        "--extended-lambda",
        "--expt-relaxed-constexpr",
        "--diag-suppress=39,128,161,174,177,186,940,2417,3012",
        "--ptxas-options=--register-usage-level=10",
        "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
        *defines,
        f"-I{package / 'include'}",
        f"-I{package / 'vendor'}",
        f"-I{package / 'vendor' / 'nccl'}",
        *[str(package / source) for source in SOURCES],
        "-lcudart",
        "-lcuda",
        f"-L{nccl_library()}",
        "-Xlinker",
        "--no-as-needed",
        "-l:libnccl.so.2",
        "-o",
        str(out / "module.so"),
    ]
    print(" ".join(command), file=sys.stderr)
    built = subprocess.run(command, capture_output=True, text=True, check=False)
    if built.returncode != 0:
        raise SystemExit(f"nvcc failed with {built.returncode}:\n{built.stderr[-6000:]}")
    exported = subprocess.run(
        ["nm", "-D", "--defined-only", str(out / "module.so")],
        capture_output=True,
        text=True,
        check=True,
    )
    symbols = {line.split()[-1] for line in exported.stdout.splitlines() if line.split()}
    missing = [entry for entry in ENTRIES if entry not in symbols]
    if missing:
        raise SystemExit(f"module.so does not export {', '.join(missing)}")
    print(json.dumps({"descriptor": descriptor, "target": target, "entries": ENTRIES}))


if __name__ == "__main__":
    main()
