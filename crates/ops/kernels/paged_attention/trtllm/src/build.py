"""The paged_attention package's trtllm compile script: one descriptor -> one module.

The module is trtllm-gen's FMHA launch code — the artifact's kernel table plus
`TllmGenFmhaRunner` — compiled against the generated header, with the cubins the
descriptor's configurations can select published beside the module in `cubins/`.

Steps:
1. stage the artifact: the generated `flashInferMetaInfo.h` and its manifest, through
   `python -m uniserve_program_server.trtllm_gen --out DIR` (the Program Server's own
   fetcher, which verifies every file against the pin);
2. read the artifact's kernel table and keep the entries the descriptor can select —
   the element types, `head_dim`, `page`, the mask and the phase, on the
   architecture families the target runs;
3. fetch those cubins and publish them beside the module, under `cubins/`;
4. compile the entry unit and the vendor's reduction unit against the staged
   header, and link `module.so`.

The module needs two things besides the runtime: the CUDA driver (`libcuda`),
because trtllm-gen's launcher calls `cuModuleLoadData`, `cuModuleGetFunction` and
`cuLaunchKernelEx` directly, and the vendor's own `runFmhaReduction`, whose
definition ships in the same data tree as the headers (`csrc/fmhaReduction.cu`)
and is compiled beside the entry. The driver is a *link-time stub* and a
*run-time host library*: the toolkit's `libcuda.so` contributes the symbols and
the soname `libcuda.so.1`, which the host's driver satisfies when the module is
loaded — the module carries no driver of its own.

A descriptor is the operator's: phase, dtype, kv, out, head_dim, page, q_heads,
kv_heads, mask, window, len.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

#: The operator's element types, as the artifact names them in a kernel.
NAMES = {"bf16": "Bfloat16", "f16": "Fp16", "f8e4m3": "E4m3", "f4e2m1": "E2m1"}

#: The architecture families each target runs. A GB200 reports SM 100 and runs the
#: family-conditional `Sm100f` kernels beside the specific `Sm100a` ones.
FAMILIES = {"sm_100a": ("Sm100a", "Sm100f"), "sm_100f": ("Sm100f",)}

#: The compile defines the artifact's own build uses.
DEFINES = [
    "TLLM_GEN_EXPORT_INTERFACE",
    "FLASHINFER_ENABLE_F16",
    "FLASHINFER_ENABLE_BF16",
    "FLASHINFER_ENABLE_FP8_E4M3",
    "FLASHINFER_ENABLE_FP8_E5M2",
    "FLASHINFER_ENABLE_FP8_E8M0",
    "FLASHINFER_ENABLE_FP4_E2M1",
    "ENABLE_BF16",
    "ENABLE_FP8",
    "ENABLE_FP4",
    "NDEBUG",
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
        pathlib.Path(values["package"]),
        values["descriptor"],
        values["target"],
        pathlib.Path(values["out"]),
    )


def fields_of(package, descriptor):
    """Parse a descriptor key against the package manifest's dimensions."""
    import tomllib

    manifest = tomllib.loads((package / "package.toml").read_text())
    dimensions = manifest["kernel"]["descriptors"]["dimensions"]
    names = sorted(dimensions, key=len, reverse=True)
    fields = {}
    remaining = descriptor
    while remaining:
        name = next((one for one in names if remaining.startswith(one + "_")), None)
        if name is None:
            raise SystemExit(f"{descriptor}: no dimension starts {remaining!r}")
        rest = remaining[len(name) + 1 :]
        values = sorted(dimensions[name], key=len, reverse=True)
        value = next((one for one in values if rest == one or rest.startswith(one + "_")), None)
        if value is None:
            raise SystemExit(f"{descriptor}: {name} takes none of {dimensions[name]} at {rest!r}")
        fields[name] = value
        remaining = rest[len(value) :].removeprefix("_")
    missing = sorted(set(dimensions) - set(fields))
    if missing:
        raise SystemExit(f"{descriptor}: the descriptor names no {missing}")
    return fields


def distribution():
    """The FlashInfer distribution's data tree: the launch code's headers."""
    found = importlib.util.find_spec("flashinfer")
    roots = []
    if found is not None and found.submodule_search_locations:
        roots.extend(pathlib.Path(one) for one in found.submodule_search_locations)
    repository = pathlib.Path(__file__).resolve().parents[6]
    roots.extend(repository.glob("crates/program-server/python/.venv/lib/python*/site-packages/flashinfer"))
    roots.extend(repository.glob(".venv/lib/python*/site-packages/flashinfer"))
    for root in roots:
        data = root / "data"
        if (data / "include/flashinfer/trtllm/fmha/fmhaRunner.cuh").is_file():
            return data
    raise SystemExit("no FlashInfer distribution with the trtllm-gen launch code was found")


def fetcher_python():
    """The interpreter that carries the Program Server's artifact fetcher.

    The build normally runs under the Program Server's own environment, whose
    `uniserve_program_server` package fetches and verifies the artifact; a direct
    run under another interpreter falls back to that environment explicitly.
    """
    if importlib.util.find_spec("uniserve_program_server") is not None:
        return sys.executable
    repository = pathlib.Path(__file__).resolve().parents[6]
    for candidate in sorted(
        repository.glob("crates/program-server/python/.venv/bin/python*")
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise SystemExit("no interpreter carries uniserve_program_server's artifact fetcher")


def stage_artifact(staged):
    """Stage the artifact's generated header and manifest into ``staged``."""
    if (staged / "include/flashInferMetaInfo.h").is_file():
        return staged
    staged.mkdir(parents=True, exist_ok=True)
    staged_run = subprocess.run(
        [fetcher_python(), "-m", "uniserve_program_server.trtllm_gen", "--out", str(staged)],
        capture_output=True,
        text=True,
        check=False,
    )
    if staged_run.returncode != 0:
        raise SystemExit(f"staging the trtllm-gen artifact failed:\n{staged_run.stderr[-4000:]}")
    return staged


def matching_kernels(header, fields, target):
    """The kernel names of the artifact the descriptor's configurations select.

    The table is the artifact's own: an entry names its element types, head width,
    page size, mask, phase and architecture family, so the filter is the
    descriptor's values, not a guess.
    """
    if fields["window"] != "false":
        raise SystemExit("the trtllm package is compiled without an attention window")
    families = FAMILIES.get(target)
    if families is None:
        raise SystemExit(f"{target} names no trtllm-gen architecture family")
    query = NAMES[fields["dtype"]]
    key_value = NAMES[fields["kv"]]
    out_type = NAMES[fields["out"]]
    prefix = f"Q{query}Kv{key_value}O{out_type}" if query != key_value else f"Qkv{query}O{out_type}"
    head = f"H{fields['head_dim']}"
    page = f"P{fields['page']}"
    decode = fields["phase"] == "decode"
    # The artifact's generation kernels are named causally whatever mask a call
    # asks for: with one query token per request, a dense and a causal mask
    # cover the same keys, which is why the vendor's own launcher hands its
    # generation kernels a causal mask either way. The prefill's mask does pick
    # the name, since its tiles see whole queries.
    mask = "Causal" if decode or fields["mask"] == "causal" else "Dense"
    kept = []
    for line in header.splitlines():
        if not line.startswith("{ DATA_TYPE_"):
            continue
        name = re.search(r'"([^"]+)"', line)
        if name is None:
            continue
        name = name.group(1)
        if prefix not in name or head not in name:
            continue
        if f"PagedKv{mask}{page}" not in name:
            continue
        if not any(name.startswith(f"fmha{family}Kernel") for family in families):
            continue
        # The artifact's prefill kernels end in `Context`; the decode ones carry
        # the generation phase in their name.
        if ("Context" in name) == decode:
            continue
        # A skipped softmax is a variant this module never asks for, and the
        # sliding-window ones only serve a window the descriptor does not have.
        if "SkipsSoftmax" in name:
            continue
        # A window is its own trade: the sliding kernels read a window size the
        # operator's call states, and this package does not serve one.
        if "SlidingOrChunked" in name:
            continue
        kept.append(name)
    if not kept:
        raise SystemExit(f"the artifact has no {prefix}{head} {mask}{page} kernel for {target}")
    return sorted(set(kept))


def fetch_cubin(name, artifact, out):
    """Publish one cubin beside the module, as bytes rather than a link."""
    target = out / "cubins" / f"{name}.cubin"
    if target.is_file():
        return target
    staged = artifact
    run = subprocess.run(
        [fetcher_python(), "-m", "uniserve_program_server.trtllm_gen", "--out", str(staged), "--cubin", f"{name}.cubin"],
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode != 0:
        raise SystemExit(f"fetching cubin {name} failed:\n{run.stderr[-2000:]}")
    staged_file = staged / f"{name}.cubin"
    if not staged_file.is_file():
        raise SystemExit(f"the fetcher staged no {staged_file}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(staged_file.resolve(), target)
    return target


def main():
    """Compile one descriptor into ``module.so`` with its cubins."""
    package, descriptor, target, out = arguments()
    fields = fields_of(package, descriptor)
    data = distribution()
    out.mkdir(parents=True, exist_ok=True)
    # The artifact is the build's input, not the module: it is staged where the
    # fetcher puts it, compiled against, and never enters the directory a runner
    # is handed.
    staging = pathlib.Path(tempfile.mkdtemp(prefix="uniserve-trtllm-gen-"))
    try:
        build(package, descriptor, fields, target, out, data, stage_artifact(staging))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def compiler_and_version():
    """The CUDA compiler this build names, and the version line it prints.

    ``CUDACXX`` wins over ``nvcc`` on ``PATH`` over the toolkit's own path, so a
    host with several toolkits can name the one it means. A candidate that does
    not answer ``--version`` with CUDA's own line is refused here rather than
    producing a module from the host compiler.
    """
    candidates = [
        os.environ.get("CUDACXX"),
        shutil.which("nvcc"),
        "/usr/local/cuda/bin/nvcc",
    ]
    tried = []
    for candidate in candidates:
        if not candidate or not pathlib.Path(candidate).is_file():
            continue
        answered = subprocess.run([candidate, "--version"], capture_output=True, text=True)
        if answered.returncode == 0 and "Cuda compilation tools" in answered.stdout:
            return candidate, answered.stdout.strip().splitlines()[-1].strip()
        tried.append(f"{candidate}: {answered.stdout[-300:]}{answered.stderr[-300:]}")
    raise SystemExit(
        "no CUDA compiler: name one with CUDACXX, or put nvcc on PATH"
        + (f"\ntried:\n" + "\n".join(tried) if tried else "")
    )


def build(package, descriptor, fields, target, out, data, artifact):
    """Compile one descriptor's module and publish the cubins it can select."""
    header_path = artifact / "include/flashInferMetaInfo.h"
    header = header_path.read_text()
    names = matching_kernels(header, fields, target)
    for name in names:
        fetch_cubin(name, artifact, out)
    print(f"published {len(names)} cubins", file=sys.stderr)

    metainfo_hash = hashlib.sha256(header_path.read_bytes()).hexdigest()
    compiler, version = compiler_and_version()
    includes = [
        artifact / "include",
        data / "include",
        data / "csrc",
        data / "csrc/nv_internal",
        data / "csrc/nv_internal/include",
        data / "cutlass/include",
        data / "cutlass/tools/util/include",
        data / "cccl/libcudacxx/include",
        data / "cccl/cub",
        data / "cccl/thrust",
    ]
    unit = out / "entry.o"
    command = [
        compiler,
        "-c",
        "-x",
        "cu",
        "-O3",
        "-std=c++17",
        f"-arch={target}",
        "--expt-relaxed-constexpr",
        "-Xcompiler",
        "-fPIC",
        *[f"-D{one}" for one in DEFINES],
        f'-DTLLM_GEN_FMHA_CUBIN_PATH="."',
        f'-DTLLM_GEN_FMHA_METAINFO_HASH="{metainfo_hash}"',
        *[f"-I{one}" for one in includes],
        "-o",
        str(unit),
        str(package / "src/entry.cpp"),
    ]
    compiled = subprocess.run(command, capture_output=True, text=True, check=False)
    if compiled.returncode != 0:
        # The command and the compiler's identity are part of the failure: a
        # module that builds on one host and not another is a defect of the
        # package, and this is what the next reader needs to see it.
        raise SystemExit(
            "compiling the entry unit failed with "
            f"{compiler} ({version}):\n{' '.join(command)}\n{compiled.stderr[-6000:]}"
        )

    # trtllm-gen's launcher calls the CUDA driver API directly
    # (`cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernelEx`), so the
    # module links against `libcuda` as well as the runtime. The toolkit's stub
    # `libcuda.so` is a link-time input only: the module carries no dependency
    # of that name, and the host's own driver satisfies the symbols when the
    # module is loaded.
    stub = pathlib.Path("/usr/local/cuda/lib64/stubs")
    if (stub / "libcuda.so").is_file():
        driver = [f"-L{stub}", "-lcuda"]
    elif any(
        path.is_file()
        for directory in ("/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu")
        for path in (
            pathlib.Path(directory) / "libcuda.so",
            pathlib.Path(directory) / "libcuda.so.1",
        )
    ):
        driver = ["-lcuda"]
    else:
        raise SystemExit(
            "no libcuda to link against: the CUDA toolkit's stub is missing from "
            f"{stub} and the host has no libcuda; a module whose driver symbols "
            "stay undefined cannot be loaded"
        )

    # trtllm-gen's launcher also calls the vendor's own reduction
    # (`tensorrt_llm::kernels::runFmhaReduction`), whose definition ships in the
    # same data tree as the headers: it is compiled beside the entry, with the
    # entry's own flags, and linked into the module.
    reduction = out / "fmhaReduction.o"
    reduction_command = [
        *command[:-3],
        "-o",
        str(reduction),
        str(data / "csrc/fmhaReduction.cu"),
    ]
    compiled = subprocess.run(
        reduction_command, capture_output=True, text=True, check=False
    )
    if compiled.returncode != 0:
        raise SystemExit(
            "compiling the vendor's reduction unit failed with "
            f"{compiler} ({version}):\n{' '.join(reduction_command)}\n"
            f"{compiled.stderr[-6000:]}"
        )

    module = out / "module.so"
    linked = subprocess.run(
        [
            compiler,
            "-shared",
            f"-arch={target}",
            str(unit),
            str(reduction),
            "-L/usr/local/cuda/lib64",
            *driver,
            "-lcudart",
            "-o",
            str(module),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if linked.returncode != 0:
        raise SystemExit(f"linking module.so failed:\n{linked.stderr[-4000:]}")
    unit.unlink()
    reduction.unlink()

    exported = subprocess.run(
        ["nm", "-D", "--defined-only", str(module)], capture_output=True, text=True, check=True
    )
    if "uniserve_paged_attention" not in {
        line.split()[-1] for line in exported.stdout.splitlines() if line.split()
    }:
        raise SystemExit("module.so does not export uniserve_paged_attention")
    print(
        json.dumps(
            {
                "descriptor": descriptor,
                "target": target,
                "entry": "uniserve_paged_attention",
                "cubins": len(names),
            }
        )
    )


if __name__ == "__main__":
    main()
