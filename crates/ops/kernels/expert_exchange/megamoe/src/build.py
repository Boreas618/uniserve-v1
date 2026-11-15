"""The megamoe package's build module: one descriptor -> one module.

The Build Server compiles a descriptor by calling :func:`compile` with the
compile request. It compiles FlashInfer's MegaMoE kernel (the pinned `megamoe`
source with `patches/megamoe.patch`, see `load_kernel_sources`) for one group
and one shape with the CuTe DSL, exports it with its host launch
(`src/launcher.py`), and links them with `src/entry.cpp` and `src/stage.cu`
into `module.so`. No device is read: the target names the architecture, and
the kernel's tuning is the one FlashInfer's frontend selects for the
descriptor's token count.
"""

import pathlib
import subprocess
import tempfile

from uniserve_build_server import cute_dsl, fetch
from uniserve_build_server.request import Request


def shape_of(request):
    """The extents a descriptor states.

    The answer is `(hidden, intermediate, top, experts, max_tokens, ranks)`;
    the mode, the attention ranks and the element type are the manifest's
    single choices, so they shape nothing here.
    """
    names = ("hidden", "intermediate", "top", "experts", "max_tokens", "ranks")
    return tuple(int(request.field(name)) for name in names)


def load_kernel_sources(request, staging):
    """Import the kernel's packages as this package compiles them.

    They are the pinned FlashInfer copy (`uniserve_build_server.fetch`'s
    `megamoe` pin) with `patches/megamoe.patch` applied: the token exchange
    addresses a peer's route metadata by that metadata's strides, which its
    capacity fixes, rather than by this launch's token extent, which differs
    between ranks and launches. The patch is applied to copies in the build's
    staging directory, so neither the distribution nor the Build Server's
    verified cache of pins is changed, and the pin's packages are imported from
    those copies.
    """
    pinned = fetch.pin("megamoe")
    staged = staging / "megamoe"
    patch = request.package / "patches" / "megamoe.patch"
    try:
        names = list(fetch.manifest(pinned, request.cache))
        fetch.fetch(pinned, request.cache, names, request.jobs)
        fetch.stage(pinned, request.cache, staged, names, copy=True)
        fetch.apply_patch(staged, patch)
        fetch.load(pinned, staged)
    except fetch.FetchError as error:
        raise SystemExit(str(error)) from error


def compile_kernel(request, shape, staging):
    """Compile and export the kernel.

    Answers the kernel's local and shared workspace sizes in bytes and the
    width of a row of its activation scales, which the entry unit lays out.
    """
    hidden, intermediate, top, experts, tokens, ranks = shape
    # The kernel is traced for the request's target, never for a probed device.
    cute_dsl.configure(request.target)
    load_kernel_sources(request, staging)
    import cutlass  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from common.megamoe_constants import SfPaddingBlock  # noqa: PLC0415
    from cutlass import torch as cutlass_torch  # noqa: PLC0415
    from cutlass.cute.runtime import make_ptr  # noqa: PLC0415
    from cutlass.cute.typing import AddressSpace  # noqa: PLC0415
    from launcher import Launcher  # noqa: PLC0415
    from moe_nvfp4_swapab.epilogue_refactor import (  # noqa: PLC0415
        SwapABSwigluFp4Epilogue,
    )
    from moe_nvfp4_swapab.megamoe_kernel import (
        Sm100MegaMoEKernel,  # noqa: PLC0415
    )
    from moe_nvfp4_swapab.runner_common import (  # noqa: PLC0415
        Nvfp4DataDtype,
        Nvfp4ScaleDtype,
    )
    from shim.tuner import default_knobs  # noqa: PLC0415
    from src.token_comm import CombineFormat  # noqa: PLC0415

    # The tile, cluster and schedule FlashInfer's frontend chooses for a kernel
    # compiled for `tokens` rows per rank: its profiles measured by token count.
    knobs = default_knobs(tokens, dtype="nvfp4")
    tiler = knobs["mma_tiler_mnk"]
    cluster = knobs["cluster_shape_mnk"]
    local = experts // ranks
    width = 2 * intermediate
    kernel = Sm100MegaMoEKernel(
        mma_tiler_mnk=tiler,
        cluster_shape_mnk=cluster,
        use_2cta_instrs=tiler[0] == 256,
        group_hint=knobs["group_hint"],
        # The pool's token block is the epilogue's token tile. The pinned
        # sources state it only as this class attribute, which their own
        # runner reads the same way (`moe_nvfp4_swapab/mega_runner.py`).
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
    )

    def described(tensor, align=16):
        """The tensor's description: every extent and stride at run time."""
        wrapped = cutlass_torch.from_dlpack(tensor, assumed_align=align)
        return wrapped.mark_layout_dynamic(
            leading_dim=cutlass_torch.get_leading_dim(tensor)
        )

    # A row's activation scales as the kernel reads them: whole 32-bit words
    # of four FP8 scales, one scale per `sf_vec_size` elements.
    scale_columns = kernel.sf_uint32_per_token * 4
    vector = kernel.sf_vec_size
    packed = torch.zeros((tokens, hidden // 2), dtype=torch.uint8).view(
        Nvfp4DataDtype
    )
    scales = torch.zeros((tokens, scale_columns), dtype=torch.uint8).view(
        Nvfp4ScaleDtype
    )
    routes = torch.zeros((tokens, top), dtype=torch.int64)
    weights = torch.zeros((tokens, top), dtype=torch.float32)
    first = torch.zeros((local, width, hidden // 2), dtype=torch.uint8)
    first = first.transpose(1, 2).view(Nvfp4DataDtype)
    second = torch.zeros((local, hidden, intermediate // 2), dtype=torch.uint8)
    second = second.transpose(1, 2).view(Nvfp4DataDtype)
    first_scales = torch.zeros(
        (local, width * (hidden // vector)), dtype=torch.uint8
    )
    first_scales = first_scales.view(Nvfp4ScaleDtype)
    second_scales = torch.zeros(
        (local, hidden * (intermediate // vector)), dtype=torch.uint8
    )
    second_scales = second_scales.view(Nvfp4ScaleDtype)
    alpha = torch.zeros((local,), dtype=torch.float32)
    output = torch.zeros((tokens, hidden), dtype=torch.bfloat16)

    def pointer():
        return make_ptr(cutlass.Uint8, 16, AddressSpace.gmem, assumed_align=16)

    # The persistent grid synchronizes grid-wide, so every cluster of it must
    # be resident at once: as many clusters as the device's multiprocessors
    # hold, one block on each.
    cluster_size = cluster[0] * cluster[1]
    clusters = request.field("multiprocessors") // cluster_size
    if clusters == 0:
        raise SystemExit(
            f"a device of {request.field('multiprocessors')} multiprocessors "
            f"holds no cluster of {cluster_size}"
        )
    # The launcher is traced with one offset per rank: its tuple's length is
    # the group's size, and the exported function takes the offsets as one
    # array.
    compiled = cute_dsl.kernel(
        Launcher(kernel, clusters),
        [
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
            tuple(cutlass.Int64(0) for _ in range(ranks)),
            cutlass.Int32(0),
        ],
        {},
    )()
    compiled.export_to_c(str(staging), "mega")
    local_bytes, shared_bytes = kernel.get_workspace_sizes()
    return local_bytes, shared_bytes, scale_columns


def run(command, what):
    """Runs one compile step; its whole output is the build's log."""
    built = subprocess.run(command, check=False)
    if built.returncode != 0:
        raise SystemExit(f"{what} failed with {built.returncode}")


def compile(request: Request) -> None:
    """Compile the fused exchange for the request's descriptor into a module.

    Only ``module.so`` is written into the request's output directory: the
    staged sources, the kernel's export and the objects are the build's own
    intermediates, in a dot directory of the output the Build Server does not
    publish.
    """
    package = request.package
    target = request.target
    shape = shape_of(request)
    hidden, intermediate, top, experts, tokens, ranks = shape
    cuda = request.toolchain.require_cuda()
    compiler = request.toolchain.require_host_compiler()
    with tempfile.TemporaryDirectory(dir=request.output, prefix=".") as scratch:
        staging = pathlib.Path(scratch)
        local_bytes, shared_bytes, scale_columns = compile_kernel(
            request, shape, staging
        )
        defines = [
            f"-DUNISERVE_RANKS={ranks}",
            f"-DUNISERVE_HIDDEN={hidden}",
            f"-DUNISERVE_INTERMEDIATE={intermediate}",
            f"-DUNISERVE_TOP={top}",
            f"-DUNISERVE_EXPERTS={experts}",
            f"-DUNISERVE_MAX_TOKENS={tokens}",
            f"-DUNISERVE_SCALE_COLUMNS={scale_columns}",
            f"-DUNISERVE_LOCAL_WORKSPACE={local_bytes}ull",
            f"-DUNISERVE_SHARED_WORKSPACE={shared_bytes}ull",
        ]
        stage = staging / "stage.o"
        run(
            [
                cuda.nvcc,
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
        # The stage's fatbin registers itself through a device link, so it is
        # linked by nvcc into one object the host linker takes.
        stage_link = staging / "stage_link.o"
        run(
            [
                cuda.nvcc,
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
                compiler,
                "-c",
                "-O2",
                "-std=c++20",
                "-fPIC",
                *defines,
                f"-I{staging}",
                f"-I{package / 'include'}",
                f"-I{cuda.include}",
                f"-iquote{package / 'src'}",
                str(package / "src" / "entry.cpp"),
                "-o",
                str(entry),
            ],
            "compiling the entry unit",
        )
        # The CuTe DSL's dialect runtime archive is linked whole. The driver
        # library is the toolkit's link-time stub: the serving process loads
        # the driver itself.
        run(
            [
                compiler,
                "-shared",
                "-O2",
                "-fPIC",
                "-o",
                str(request.output / "module.so"),
                str(staging / "mega.o"),
                str(entry),
                str(stage),
                str(stage_link),
                "-Wl,--whole-archive",
                str(cute_dsl.dialect_runtime(cuda)),
                "-Wl,--no-whole-archive",
                "-Wl,--no-as-needed",
                f"-L{cuda.lib}",
                "-lcudart",
                f"-L{cuda.stubs}",
                "-lcuda",
            ],
            "linking module.so",
        )
