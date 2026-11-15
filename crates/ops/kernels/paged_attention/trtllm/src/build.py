"""The paged_attention package's trtllm build module.

One descriptor -> one module.

The module is trtllm-gen's FMHA launch code — the artifact's kernel table plus
`TllmGenFmhaRunner` — compiled against the generated header, with the cubins
the descriptor's configurations can select published beside the module in
`cubins/`. The Build Server compiles a descriptor by calling :func:`compile`
with the compile request.

Steps:
1. stage the pinned `trtllm-gen-fmha` artifact's generated
   `flashInferMetaInfo.h` (`uniserve_build_server.trtllm.stage`, which
   verifies every file against the pin);
2. read the artifact's kernel table and keep the entries the descriptor can
   select — the element types, `head_dim`, `page`, the mask and the phase, on
   the architecture families the target runs;
3. fetch those cubins and publish them beside the module, under `cubins/`;
4. compile the entry unit and the vendor's reduction unit against the staged
   header, and link `module.so`.

The module needs two things besides the runtime: the CUDA driver (`libcuda`),
because trtllm-gen's launcher calls `cuModuleLoadData`, `cuModuleGetFunction`
and `cuLaunchKernelEx` directly, and the vendor's own `runFmhaReduction`, whose
definition ships in the same data tree as the headers (`csrc/fmhaReduction.cu`)
and is compiled beside the entry. The driver is a *link-time stub* and a
*run-time host library*: the toolkit's `libcuda.so` contributes the symbols and
the soname `libcuda.so.1`, which the host's driver satisfies when the module is
loaded — the module carries no driver of its own.

A descriptor is the operator's: phase, dtype, kv, out, head_dim, page, q_heads,
kv_heads, mask, window, len.
"""

import hashlib
import pathlib
import re
import shutil
import sys
import tempfile

from uniserve_build_server import trtllm
from uniserve_build_server.request import Request

#: The operator's element types, as the artifact names them in a kernel.
NAMES = {"bf16": "Bfloat16", "f16": "Fp16", "f8e4m3": "E4m3", "f4e2m1": "E2m1"}

#: The pinned trtllm-gen library this package launches.
LIBRARY = "trtllm-gen-fmha"

#: The artifact's generated kernel table.
METADATA_HEADER = "include/flashInferMetaInfo.h"


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


def matching_kernels(header, fields, target):
    """The kernel names of the artifact the descriptor's configurations select.

    The table is the artifact's own: an entry names its element types, head
    width, page size, mask, phase and architecture family, so the filter is the
    descriptor's values, not a guess. A target none of whose families the
    artifact was built for matches no kernel, which refuses the descriptor.
    """
    families = trtllm.families(target)
    query = NAMES[fields["dtype"]]
    key_value = NAMES[fields["kv"]]
    out_type = NAMES[fields["out"]]
    prefix = (
        f"Q{query}Kv{key_value}O{out_type}"
        if query != key_value
        else f"Qkv{query}O{out_type}"
    )
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
        if not any(
            name.startswith(f"fmha{family}Kernel") for family in families
        ):
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
        raise SystemExit(
            f"the artifact has no {prefix}{head} {mask}{page} kernel for "
            f"{target}"
        )
    return sorted(set(kept))


def publish_cubins(request, names, artifact):
    """Publish the named cubins beside the module, as bytes, not links."""
    trtllm.stage(
        request, LIBRARY, artifact, [f"{name}.cubin" for name in names]
    )
    for name in names:
        staged = artifact / f"{name}.cubin"
        if not staged.is_file():
            raise SystemExit(f"the fetcher staged no {staged}")
        target = request.output / "cubins" / f"{name}.cubin"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staged.resolve(), target)


def compile(request: Request) -> None:
    """Compile the request's descriptor into ``module.so`` with its cubins.

    The artifact is the build's input, not the module: it is staged into a dot
    directory of the request's output, which the Build Server does not
    publish, compiled against, and never enters the directory a runner is
    handed. The objects are staged there too.
    """
    data = trtllm.data_tree()
    with tempfile.TemporaryDirectory(dir=request.output, prefix=".") as scratch:
        staging = pathlib.Path(scratch)
        trtllm.stage(request, LIBRARY, staging, [METADATA_HEADER])
        build(request, data, staging)


def build(request, data, artifact):
    """Compile one descriptor's module and publish the cubins it can select.

    ``artifact`` is the scratch directory the artifact's header is staged in;
    the cubins and the objects are staged beside it.
    """
    package = request.package
    target = request.target
    header_path = artifact / METADATA_HEADER
    header = header_path.read_text()
    names = matching_kernels(header, request.fields, target)
    publish_cubins(request, names, artifact)
    print(f"published {len(names)} cubins", file=sys.stderr)

    metainfo_hash = hashlib.sha256(header_path.read_bytes()).hexdigest()
    cuda = request.toolchain.require_cuda()
    # The compiler's identity is part of a failure: a module that builds on one
    # host and not another is a defect of the package, and this is what the
    # next reader needs to see it, beside the command `trtllm.run` logs.
    identity = f"{cuda.nvcc} ({cuda.version})"
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
    unit = artifact / "entry.o"
    command = [
        cuda.nvcc,
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
        '-DTLLM_GEN_FMHA_CUBIN_PATH="."',
        f'-DTLLM_GEN_FMHA_METAINFO_HASH="{metainfo_hash}"',
        *[f"-I{one}" for one in includes],
        "-o",
        str(unit),
        str(package / "src/entry.cpp"),
    ]
    trtllm.run(command, f"compiling the entry unit with {identity}")

    # trtllm-gen's launcher also calls the vendor's own reduction
    # (`tensorrt_llm::kernels::runFmhaReduction`), whose definition ships in the
    # same data tree as the headers: it is compiled beside the entry, with the
    # entry's own flags, and linked into the module.
    reduction = artifact / "fmhaReduction.o"
    reduction_command = [
        *command[:-3],
        "-o",
        str(reduction),
        str(data / "csrc/fmhaReduction.cu"),
    ]
    trtllm.run(
        reduction_command,
        f"compiling the vendor's reduction unit with {identity}",
    )

    # trtllm-gen's launcher calls the CUDA driver API directly
    # (`cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernelEx`), so the
    # module links against `libcuda` as well as the runtime. The toolkit's stub
    # `libcuda.so` is a link-time input only, named by its directory: the
    # module carries no dependency of that name, and the host's own driver
    # satisfies the symbols when the module is loaded.
    trtllm.run(
        [
            cuda.nvcc,
            "-shared",
            f"-arch={target}",
            str(unit),
            str(reduction),
            f"-L{cuda.stubs}",
            "-lcuda",
            "-lcudart",
            "-o",
            str(request.output / "module.so"),
        ],
        "linking module.so",
    )
