"""The megamoe package's compile script: one descriptor -> one module.

The Program Server runs this script with its working directory the package root and
`UNISERVE_DESCRIPTOR`, `UNISERVE_TARGET`, `UNISERVE_PACKAGE` and `UNISERVE_OUT` in the
environment. It compiles FlashInfer's MegaMoE kernel (`vendor/megamoe`) for one group
and one shape with the CuTe DSL, exports the kernel with its host launcher, and links
them with `src/entry.cpp` and `src/stage.cu` into `module.so`. No device is read: the
target names the architecture, and the kernel's tuning is the table below.
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

ENTRIES = [
    "uniserve_expert_exchange_launch",
    "uniserve_expert_exchange_create",
    "uniserve_expert_exchange_step",
    "uniserve_expert_exchange_suspend",
    "uniserve_expert_exchange_restore",
    "uniserve_expert_exchange_release",
    "uniserve_expert_exchange_info",
]

#: The SMs the launch's clusters are sized for. A persistent cluster that does not fit
#: on the device would wait for one that never starts, so this is the smallest SM100
#: part's count, and a larger part leaves SMs idle rather than hanging.
SM_COUNT = 148

#: The kernel's measured profiles by the largest step's token count: design 5.4 keeps
#: tuning data in the program package. These are FlashInfer's built-in table.
MID_TOKENS = {
    "mma_tiler_mnk": (256, 128, 256),
    "cluster_shape_mnk": (2, 1, 1),
    "group_hint": 512,
    "flag_batch": 4,
    "epi_flag_batch": (2, 4),
    "token_back_mode": "reuse_dispatch_warps",
    "load_balance_mode": "atomic_counter",
}
SMALL_TOKENS = dict(MID_TOKENS, token_back_mode="epi_warps")
MID_LARGE_TOKENS = dict(
    MID_TOKENS, mma_tiler_mnk=(256, 256, 256), token_back_mode="standalone_warps"
)
LARGE_TOKENS = dict(MID_TOKENS, mma_tiler_mnk=(256, 256, 256), flag_batch=8)


def knobs_for(tokens):
    """The profile of a step of `tokens` rows, as `default_knobs` chooses it."""
    if tokens < 512:
        return SMALL_TOKENS
    if tokens < 1024:
        return MID_TOKENS
    if tokens < 2048:
        return MID_LARGE_TOKENS
    return LARGE_TOKENS


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


def shape_of(descriptor):
    """The extents a descriptor states."""
    found = re.fullmatch(
        r"mode_fused_hidden_(\d+)_intermediate_(\d+)_top_(\d+)_experts_(\d+)"
        r"_max_tokens_(\d+)_ranks_(\d+)_attention_0_dtype_bf16",
        descriptor,
    )
    if found is None:
        raise SystemExit(f"this package serves no descriptor {descriptor}")
    hidden, intermediate, top, experts, tokens, ranks = map(int, found.groups())
    return hidden, intermediate, top, experts, tokens, ranks


#: The kernel's entry, spelled per group size: the exported launcher takes scalars, so a
#: peer's offset is a parameter of its own.
EXPORT = """
class Export:
    def __init__(self, kernel, clusters):
        self.kernel = kernel
        self.clusters = clusters

    @cute.jit
    def __call__(
        self,
        activation: cute.Tensor,
        activation_sf: cute.Tensor,
        topk_idx: cute.Tensor,
        topk_weights: cute.Tensor,
        fc1_weight: cute.Tensor,
        fc1_weight_sf: cute.Tensor,
        fc2_weight: cute.Tensor,
        fc2_weight_sf: cute.Tensor,
        fc1_alpha: cute.Tensor,
        fc2_alpha: cute.Tensor,
        fc1_norm_const: cute.Tensor,
        output_activation: cute.Tensor,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        base: cutlass.Int64,
        {parameters}
        rank: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        host = SymBufferHost(
            base_addr=base,
            offsets=({offsets},),
            rank_idx=rank,
            num_max_ranks={ranks},
        )
        self.kernel(
            activation,
            activation_sf,
            topk_idx,
            topk_weights,
            fc1_weight,
            fc1_weight_sf,
            fc2_weight,
            fc2_weight_sf,
            fc1_alpha,
            fc2_alpha,
            fc1_norm_const,
            output_activation,
            local_workspace,
            shared_workspace,
            host,
            self.clusters,
            stream,
        )
"""


def compile_kernel(package, target, shape, staging):
    """Compile and export the kernel; answers its two workspace sizes."""
    hidden, intermediate, top, experts, tokens, ranks = shape
    os.environ["CUTE_DSL_ARCH"] = target
    sys.path.insert(0, str(package / "vendor" / "megamoe"))
    import cuda.bindings.driver as cuda  # noqa: PLC0415
    import cutlass  # noqa: PLC0415
    import cutlass.cute as cute  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from common.megamoe_constants import SfPaddingBlock  # noqa: PLC0415
    from cutlass import torch as cutlass_torch  # noqa: PLC0415
    from cutlass.cute.runtime import make_fake_stream, make_ptr  # noqa: PLC0415
    from cutlass.cute.typing import AddressSpace  # noqa: PLC0415
    from moe_nvfp4_swapab.epilogue_refactor import (  # noqa: PLC0415
        SwapABSwigluFp4Epilogue,
    )
    from moe_nvfp4_swapab.megamoe_kernel import Sm100MegaMoEKernel  # noqa: PLC0415
    from moe_nvfp4_swapab.runner_common import (  # noqa: PLC0415
        _DataDtype,
        _ScaleDtype,
    )
    from src.sym_buffer import SymBufferHost  # noqa: PLC0415
    from src.token_comm import CombineFormat  # noqa: PLC0415

    knobs = knobs_for(tokens)
    tiler = knobs["mma_tiler_mnk"]
    cluster = knobs["cluster_shape_mnk"]
    local = experts // ranks
    width = 2 * intermediate
    kernel = Sm100MegaMoEKernel(
        mma_tiler_mnk=tiler,
        cluster_shape_mnk=cluster,
        use_2cta_instrs=tiler[0] == 256,
        group_hint=knobs["group_hint"],
        token_padding_block=SwapABSwigluFp4Epilogue._EpilogueTokenTileSize,
        sf_padding_block=SfPaddingBlock,
        load_balance_mode=knobs["load_balance_mode"],
        static_expert_shape=(local, width, hidden),
        force_static_sched=True,
        clc_bundle_size=None,
        num_sched_stages=None,
        world_size=ranks,
        num_topk=top,
        max_tokens_per_rank=tokens,
        hidden=hidden,
        fc2_output_dtype=cutlass.BFloat16,
        non_ubulk_fc2_store=True,
        in_kernel_fc2_reduce=False,
        token_back_mode=knobs["token_back_mode"],
        apply_topk_in_fc1=False,
        gate_up_clamp=None,
        flag_batch=knobs["flag_batch"],
        epi_flag_batch=knobs["epi_flag_batch"],
        combine_format=CombineFormat.parse("bf16"),
        activation="silu",
    )

    def described(tensor, align=16):
        """The tensor's description: every extent and stride a run-time value."""
        wrapped = cutlass_torch.from_dlpack(tensor, assumed_align=align)
        return wrapped.mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(tensor))

    scale_columns = (hidden // 16 + 3) // 4 * 4
    packed = torch.zeros((tokens, hidden // 2), dtype=torch.uint8).view(_DataDtype)
    scales = torch.zeros((tokens, scale_columns), dtype=torch.uint8).view(_ScaleDtype)
    routes = torch.zeros((tokens, top), dtype=torch.int64)
    weights = torch.zeros((tokens, top), dtype=torch.float32)
    first = torch.zeros((local, width, hidden // 2), dtype=torch.uint8)
    first = first.transpose(1, 2).view(_DataDtype)
    second = torch.zeros((local, hidden, intermediate // 2), dtype=torch.uint8)
    second = second.transpose(1, 2).view(_DataDtype)
    first_scales = torch.zeros((local, width * (hidden // 16)), dtype=torch.uint8)
    first_scales = first_scales.view(_ScaleDtype)
    second_scales = torch.zeros((local, hidden * (intermediate // 16)), dtype=torch.uint8)
    second_scales = second_scales.view(_ScaleDtype)
    alpha = torch.zeros((local,), dtype=torch.float32)
    output = torch.zeros((tokens, hidden), dtype=torch.bfloat16)

    names = ", ".join(f"off{rank}" for rank in range(ranks))
    parameters = "\n        ".join(f"off{rank}: cutlass.Int64," for rank in range(ranks))
    source = EXPORT.format(parameters=parameters, offsets=names, ranks=ranks)
    # `cute.jit` reads the function's source, so the entry is a real file on the path.
    module_path = staging / "mega_export.py"
    module_path.write_text(
        "import cuda.bindings.driver as cuda\n"
        "import cutlass\nimport cutlass.cute as cute\n"
        "from src.sym_buffer import SymBufferHost\n" + source
    )
    sys.path.insert(0, str(staging))
    import mega_export  # noqa: PLC0415

    def pointer():
        return make_ptr(cutlass.Uint8, 16, AddressSpace.gmem, assumed_align=16)

    cluster_size = cluster[0] * cluster[1]
    clusters = max(1, SM_COUNT // max(cluster_size, 1))
    compiled = cute.compile(
        mega_export.Export(kernel, clusters),
        described(packed),
        described(scales),
        described(routes),
        described(weights),
        described(first),
        described(first_scales),
        described(second),
        described(second_scales),
        described(alpha, 4),
        described(alpha, 4),
        described(alpha, 4),
        described(output),
        pointer(),
        pointer(),
        cutlass.Int64(0),
        *[cutlass.Int64(0) for _ in range(ranks)],
        cutlass.Int32(0),
        make_fake_stream(),
    )
    compiled.export_to_c(str(staging), "mega")
    return kernel.get_workspace_sizes()


def cutlass_library():
    """The CuTe DSL static runtime archive for this host's CUDA."""
    import importlib.util  # noqa: PLC0415

    found = importlib.util.find_spec("nvidia_cutlass_dsl")
    locations = list(getattr(found, "submodule_search_locations", None) or ())
    if not locations:
        raise SystemExit("the CuTe DSL package nvidia_cutlass_dsl is not installed")
    root = pathlib.Path(locations[0])
    for major in ("cu13", "cu12"):
        archive = root / major / "lib" / "libcuda_dialect_runtime_static.a"
        if archive.exists():
            return archive
    raise SystemExit("no libcuda_dialect_runtime_static.a in nvidia_cutlass_dsl")


def cuda_home():
    """The CUDA toolkit's directory, which `CUDA_HOME` names: the runtime's headers and library."""
    home = os.environ.get("CUDA_HOME")
    if not home:
        raise SystemExit("CUDA_HOME names the CUDA toolkit whose runtime the module links against")
    return pathlib.Path(home)


def run(command, what):
    """Runs one compile step and refuses with its output when it fails."""
    built = subprocess.run(command, capture_output=True, text=True, check=False)
    if built.returncode != 0:
        raise SystemExit(f"{what} failed:\n{built.stderr[-6000:]}")


def main():
    """Compile the fused exchange for one descriptor into ``module.so``."""
    package, descriptor, target, out = arguments()
    out.mkdir(parents=True, exist_ok=True)
    shape = shape_of(descriptor)
    hidden, intermediate, top, experts, tokens, ranks = shape
    with tempfile.TemporaryDirectory() as directory:
        staging = pathlib.Path(directory)
        local_bytes, shared_bytes = compile_kernel(package, target, shape, staging)
        offsets = ", ".join(f"offsets[{rank}]" for rank in range(ranks))
        defines = [
            f"-DUNISERVE_RANKS={ranks}",
            f"-DUNISERVE_HIDDEN={hidden}",
            f"-DUNISERVE_INTERMEDIATE={intermediate}",
            f"-DUNISERVE_TOP={top}",
            f"-DUNISERVE_EXPERTS={experts}",
            f"-DUNISERVE_MAX_TOKENS={tokens}",
            f"-DUNISERVE_LOCAL_WORKSPACE={local_bytes}ull",
            f"-DUNISERVE_SHARED_WORKSPACE={shared_bytes}ull",
            f"-DUNISERVE_OFFSETS={ranks}",
            f"-DUNISERVE_OFFSET_ARGUMENTS={offsets}",
        ]
        nvcc = str(cuda_home() / "bin" / "nvcc")
        stage = staging / "stage.o"
        run(
            [
                nvcc,
                "-c",
                "-O3",
                "-std=c++17",
                f"-arch={target}",
                "-Xcompiler",
                "-fPIC",
                str(package / "src" / "stage.cu"),
                "-o",
                str(stage),
            ],
            "compiling the staging kernel",
        )
        # The stage's fatbin registers itself through a device link, so it is linked
        # by nvcc into one object the host linker takes.
        stage_link = staging / "stage_link.o"
        run(
            [
                nvcc,
                "-dlink",
                f"-arch={target}",
                "-Xcompiler",
                "-fPIC",
                str(stage),
                "-o",
                str(stage_link),
            ],
            "linking the staging kernel",
        )
        entry = staging / "entry.o"
        run(
            [
                os.environ.get("CXX", "g++"),
                "-c",
                "-O2",
                "-std=c++20",
                "-fPIC",
                *defines,
                f"-I{staging}",
                f"-I{package / 'include'}",
                f"-I{cuda_home() / 'include'}",
                f"-iquote{package / 'src'}",
                str(package / "src" / "entry.cpp"),
                "-o",
                str(entry),
            ],
            "compiling the entry unit",
        )
        run(
            [
                os.environ.get("CXX", "g++"),
                "-shared",
                "-O2",
                "-fPIC",
                "-o",
                str(out / "module.so"),
                str(staging / "mega.o"),
                str(entry),
                str(stage),
                str(stage_link),
                "-Wl,--whole-archive",
                str(cutlass_library()),
                "-Wl,--no-whole-archive",
                "-Wl,--no-as-needed",
                f"-L{cuda_home() / 'lib64'}",
                "-lcudart",
                "-lcuda",
            ],
            "linking module.so",
        )
    listed = subprocess.run(
        ["nm", "-D", "--defined-only", str(out / "module.so")],
        capture_output=True,
        text=True,
        check=True,
    )
    symbols = {line.split()[-1] for line in listed.stdout.splitlines() if line.split()}
    missing = [entry for entry in ENTRIES if entry not in symbols]
    if missing:
        raise SystemExit(f"module.so does not export {', '.join(missing)}")
    print(json.dumps({"descriptor": descriptor, "target": target, "entries": ENTRIES}))


if __name__ == "__main__":
    main()
