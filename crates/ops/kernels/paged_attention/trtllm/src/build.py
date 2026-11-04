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
4. compile the entry unit against the staged header and link `module.so`.

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
    families = FAMILIES.get(target)
    if families is None:
        raise SystemExit(f"{target} names no trtllm-gen architecture family")
    query = NAMES[fields["dtype"]]
    key_value = NAMES[fields["kv"]]
    out_type = NAMES[fields["out"]]
    prefix = f"Q{query}Kv{key_value}O{out_type}" if query != key_value else f"Qkv{query}O{out_type}"
    head = f"H{fields['head_dim']}"
    page = f"P{fields['page']}"
    mask = "Causal" if fields["mask"] == "causal" else "Dense"
    decode = fields["phase"] == "decode"
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
        if "SlidingOrChunked" in name and fields["window"] == "false":
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


def build(package, descriptor, fields, target, out, data, artifact):
    """Compile one descriptor's module and publish the cubins it can select."""
    header_path = artifact / "include/flashInferMetaInfo.h"
    header = header_path.read_text()
    names = matching_kernels(header, fields, target)
    for name in names:
        fetch_cubin(name, artifact, out)
    print(f"published {len(names)} cubins", file=sys.stderr)

    metainfo_hash = hashlib.sha256(header_path.read_bytes()).hexdigest()
    compiler = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
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
    compiled = subprocess.run(
        [
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
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if compiled.returncode != 0:
        raise SystemExit(f"compiling the entry unit failed:\n{compiled.stderr[-6000:]}")

    module = out / "module.so"
    linked = subprocess.run(
        [
            compiler,
            "-shared",
            f"-arch={target}",
            str(unit),
            "-L/usr/local/cuda/lib64",
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
