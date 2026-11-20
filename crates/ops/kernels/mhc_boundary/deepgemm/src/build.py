"""The mhc_boundary package's DeepGEMM build module: a descriptor, a module.

The module is the operator's fused form, DeepGEMM's SM100 mega mHC kernel in
its shifted form (``sm100_mega_mhc_impl`` of the pinned headers), compiled with
the package's entry unit (``src/boundary.cu``), which encodes the kernel's
tensor maps and launches it. DeepGEMM compiles one instance per split count at
run time, choosing the count from the call's tokens and the device; the
operator states that rule for a fixed plan of programs
(``ops::mhc::BoundaryDescriptor::fused_splits``), so the counts a descriptor's
token class can take are known here, and one module holds an instance of each.
A serving host compiles nothing.

The steps:

1. fetch the ``deepgemm`` pin and stage a copy with
   ``patches/launched-grid.patch`` applied: the persistent kernel strides its
   tasks by the launched grid rather than a compiled multiprocessor count, and
   each block's two arrival lines lie side by side;
2. evaluate the split rule (:func:`splits`) over the blocks of the
   descriptor's token class: one to eight blocks of 64 tokens for a call of at
   most 512 tokens, more for a longer one;
3. compile the entry unit with the width and the split counts as definitions,
   against the staged headers and the CUTLASS of the ``flashinfer-python``
   distribution, as DeepGEMM's run-time compiler compiles its kernels (the
   pin's ``[compile]``), and link it with the driver API into ``module.so``.

The Build Server compiles a descriptor by calling :func:`compile` with the
compile request.
"""

from __future__ import annotations

from uniserve_build_server import cuda, fetch
from uniserve_build_server.request import Request

#: The package's unit: the operator's entry and the kernel's instances.
UNIT = "src/boundary.cu"

#: The package's patch of the pinned kernel and its scheduler.
PATCH = "patches/launched-grid.patch"

#: Where the CUTLASS the kernel includes is installed.
CUTLASS = ("flashinfer-python", "flashinfer/data/cutlass/include")

#: The kernel's tokens of a block and columns of a step (``layout::mega_mhc``).
BLOCK_TOKENS = 64
BLOCK_COLUMNS = 64

#: The operator's split rule (``ops::mhc``): the programs it plans for, and
#: the fewest and the most splits of a block.
PROGRAMS = 148
FEWEST_SPLITS = 16
MOST_SPLITS = 64

#: The most tokens a call of the smaller token class holds
#: (``ops::mhc::BOUNDARY_FEW``), and the class's token block.
FEW_TOKENS = 512
FEW_BLOCK = 16

#: The instances one module holds: the entry dispatches among four.
INSTANCES = 4


def splits(blocks: int, hidden: int) -> int:
    """The split count of a call of ``blocks`` blocks of 64 tokens over
    streams of ``hidden`` elements: the fewest equal shares of the column steps
    that still bring the blocks' tasks to the planned programs, between 16 and
    64 (``BoundaryDescriptor::fused_splits``, DeepGEMM's ``get_num_splits``).
    """  # noqa: D205
    steps = max(hidden // BLOCK_COLUMNS, 1)
    planned = min(max(PROGRAMS // max(blocks, 1), FEWEST_SPLITS), MOST_SPLITS)
    shares = -(-steps // planned)
    return -(-steps // shares)


def instances(tokens: int, hidden: int) -> list[int]:
    """The split counts a call of the descriptor's token class can take, as
    many as the entry dispatches among (the last repeated): every block count
    of a call of at most 512 tokens for the small class, and the counts from
    nine blocks on, where the rule gives the fewest splits, for the large one.
    """  # noqa: D205
    if tokens == FEW_BLOCK:
        blocks = range(1, FEW_TOKENS // BLOCK_TOKENS + 1)
    else:
        blocks = range(FEW_TOKENS // BLOCK_TOKENS + 1, PROGRAMS + 2)
    counts = sorted({splits(one, hidden) for one in blocks}, reverse=True)
    if len(counts) > INSTANCES:
        raise SystemExit(
            f"streams of {hidden} elements take {len(counts)} split counts, "
            f"more than the entry's {INSTANCES} instances"
        )
    return counts + [counts[-1]] * (INSTANCES - len(counts))


def definitions(request: Request) -> dict[str, cuda.Definition]:
    """The entry's compile-time values: the width and the split counts."""
    hidden = request.field("hidden")
    counts = instances(request.field("tokens"), hidden)
    return {
        "UNISERVE_HIDDEN": hidden,
        **{
            f"UNISERVE_SPLITS_{index}": count
            for index, count in enumerate(counts)
        },
        "UNISERVE_PLANNED_MULTIPROCESSORS": PROGRAMS,
    }


def compile(request: Request) -> None:
    """Compile the fused boundary of the request's descriptor: ``module.so``.

    A fetch, patch or compiler failure refuses the build, with its output in
    the build log.
    """
    values = definitions(request)
    print(f"the instances: {values}", flush=True)

    pinned = fetch.pin("deepgemm")
    with cuda.scratch(request) as staged:
        try:
            names = list(fetch.manifest(pinned, request.cache))
            fetch.fetch(pinned, request.cache, names, request.jobs)
            fetch.stage(pinned, request.cache, staged, names, copy=True)
            fetch.apply_patch(staged, request.package / PATCH)
            headers = [
                *fetch.includes(pinned, staged),
                fetch.installed(*CUTLASS),
            ]
        except fetch.FetchError as error:
            raise SystemExit(
                f"staging the deepgemm pin failed: {error}"
            ) from error

        # The kernel's body is tcgen05 code behind a guard of the architecture
        # number alone, which no portable PTX can hold.
        unit = cuda.Unit(
            source=request.package / UNIT,
            definitions=values,
            includes=headers,
            specific=True,
            **pinned.compile,
        )
        # The entry encodes the kernel's tensor maps through the driver API.
        cuda.native(request, [unit], libraries=["cuda"])
