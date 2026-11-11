"""The deepep package's compile script: one descriptor -> one module.

The descriptor fixes every template argument of DeepEP's kernels, so this
compiles them here, and a serving host compiles nothing. The Build Server runs
this script with its working directory the package root and
`UNISERVE_DESCRIPTOR`, `UNISERVE_TARGET`, `UNISERVE_PACKAGE` and `UNISERVE_OUT`
in the environment.

DeepEP's kernels and the NCCL device API they are built against are pinned
sources (`uniserve_build_server.fetch`'s `deepep` and `nccl` pins): the
compile includes them from their verified cache directories.
"""

import json
import pathlib
import re
import subprocess
import sys
import sysconfig

from uniserve_build_server import fetch, package

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
    `tools/check_nccl.sh` holds to the project's version; this one only states
    the library's name to the linker.
    """
    site = pathlib.Path(sysconfig.get_paths()["purelib"])
    directory = site / "nvidia" / "nccl" / "lib"
    if not (directory / "libnccl.so.2").exists():
        raise SystemExit(f"no libnccl.so.2 in {directory}")
    return directory


def main():
    """Compile the exchange for one descriptor into ``module.so``."""
    root, descriptor, target, out = package.arguments()
    out.mkdir(parents=True, exist_ok=True)
    try:
        deepep = fetch.fetch(fetch.pin("deepep"))
        nccl = fetch.fetch(fetch.pin("nccl"))
    except fetch.FetchError as error:
        raise SystemExit(str(error)) from error
    compiler, _ = package.nvcc()
    defines = [
        f"-D{name}={value}" for name, value in constants(descriptor).items()
    ]
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
        f"-I{root / 'include'}",
        f"-I{deepep / 'deep_ep' / 'include'}",
        f"-I{nccl / 'nvidia' / 'nccl' / 'include'}",
        *[str(root / source) for source in SOURCES],
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
        raise SystemExit(
            f"nvcc failed with {built.returncode}:\n{built.stderr[-6000:]}"
        )
    package.require_exports(out / "module.so", ENTRIES)
    print(
        json.dumps(
            {"descriptor": descriptor, "target": target, "entries": ENTRIES}
        )
    )


if __name__ == "__main__":
    main()
