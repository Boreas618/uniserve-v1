"""The expert_exchange DeepGEMM build module: one descriptor -> one module.

The module is DeepGEMM's MegaMoE over MXFP4 experts,
``sm100_fp8_fp4_mega_moe_impl`` of the pinned headers, compiled ahead of time.
DeepGEMM's Python extension compiles it at run time for the call's own rows:
its scheduler picks the tile height (BLOCK_M) from the rows a rank sends
(``get_block_config_for_mega_moe``, ``csrc/jit_kernels/heuristics/mega_moe.hpp``
at the pinned commit), and the rest of the tile, the pipeline depth and the
ring capacity follow from it. A module serves every step its transport holds,
so the build compiles the instance of every BLOCK_M the rule picks for a
step of 1 to ``max_tokens`` rows a rank, and the entry launches the one the
rule picks for its step's capacity.

The steps:

1. refuse a descriptor the kernel cannot run (:func:`check`);
2. find the BLOCK_M runs (:func:`runs`): the tile height of each run of
   steps and the largest step of the run, ascending;
3. fetch the ``deepgemm`` pin and stage it;
4. compile ``src/kernel.cu`` once per height (``src/config.cuh`` derives the
   rest of the tile and the ring from BLOCK_M with DeepGEMM's own constexpr
   helpers), ``src/entry.cu`` with the class table, and ``src/stage.cu``, as
   DeepGEMM's run-time compiler compiles its kernels (the pin's
   ``[compile]``), against the CUTLASS of the ``flashinfer-python``
   distribution, and link them with the driver API into ``module.so``.

The Build Server compiles a descriptor by calling :func:`compile` with the
compile request.
"""

from __future__ import annotations

import math
import struct

from uniserve_build_server import cuda, fetch
from uniserve_build_server.request import Request

#: The package's units: the transport and launch, the staging kernel, and the
#: kernel instance compiled once per BLOCK_M.
ENTRY = "src/entry.cu"
STAGE = "src/stage.cu"
KERNEL = "src/kernel.cu"

#: The header of the module's tile heights and runs the build writes for the
#: entry.
RUNS = "uniserve_mega_runs.h"

#: Where the CUTLASS the kernel includes is installed.
CUTLASS = ("flashinfer-python", "flashinfer/data/cutlass/include")

#: DeepGEMM's tile heights (``layout::kCandidateBlockM``).
CANDIDATES = (8, 16, 32, 64, 96, 128, 192, 240)

#: The bounds of the kernel's signal block and dispatch
#: (``layout/mega_moe.cuh``, ``layout/sym_buffer.cuh``): experts, experts a
#: rank, ranks, routes a token.
MAX_EXPERTS, MAX_LOCAL_EXPERTS, MAX_RANKS, MAX_TOP = 2048, 512, 72, 32

#: A local expert count from which one 240-row block of an expert pays for
#: itself (DeepGEMM's rule: about 14 blocks a rank).
WIDE_BLOCK_EXPERTS = 14


def f32(value: float) -> float:
    """``value`` rounded to the nearest FP32, as DeepGEMM's ``float`` holds
    it.
    """  # noqa: D205
    return struct.unpack("<f", struct.pack("<f", value))[0]


def block_m(tokens: int, ranks: int, experts: int, top: int) -> int:
    """The BLOCK_M DeepGEMM's scheduler picks for ``tokens`` rows a rank.

    A port of ``get_block_config_for_mega_moe``, its single-precision
    arithmetic rounded step for step: the rows an expert expects plus one
    standard deviation of the routing, covered by the fewest blocks of the
    smallest height that holds them.
    """
    expected = f32(f32(f32(float(tokens) * ranks) * top) / experts)
    covered = f32(expected + f32(math.sqrt(expected)))
    height = 16 if expected <= 10 else 32
    if expected > 24:
        blocks = math.ceil(f32(covered / 240))
        if (
            blocks == 1
            and covered > 192
            and experts // ranks < WIDE_BLOCK_EXPERTS
        ):
            blocks = 2
        for candidate in (64, 128, 192, 240):
            height = candidate
            if blocks * candidate >= covered:
                break
    return height


def runs(request: Request) -> list[tuple[int, int]]:
    """The BLOCK_M a step of 1 to ``max_tokens`` rows a rank picks, as runs of
    steps: each run's height and the largest step of the run, in ascending
    order of the steps.

    The rule is not monotone in the rows (a step that needs a second block an
    expert takes a lower height than the step before), so a height may head
    several runs.
    """  # noqa: D205
    ranks, experts, top = (
        request.field("ranks"),
        request.field("experts"),
        request.field("top"),
    )
    found: list[tuple[int, int]] = []
    for tokens in range(1, request.field("max_tokens") + 1):
        height = block_m(tokens, ranks, experts, top)
        if found and found[-1][0] == height:
            found[-1] = (height, tokens)
        else:
            found.append((height, tokens))
    assert all(height in CANDIDATES for height, _ in found)
    return found


def limit_of(request: Request) -> float:
    """The clamp the kernel is compiled with: the descriptor's, or none."""
    limit = float(request.field("limit"))
    if not math.isfinite(limit) or limit < 0:
        raise SystemExit(f"a clamp of {limit} is no finite magnitude")
    return limit


def check(request: Request) -> None:
    """Refuse a descriptor the kernel's static assertions or its signal block
    refuse, naming the rule.
    """  # noqa: D205
    ranks, experts, top = (
        request.field("ranks"),
        request.field("experts"),
        request.field("top"),
    )
    intermediate, shared = (
        request.field("intermediate"),
        request.field("shared"),
    )
    if experts % ranks != 0:
        raise SystemExit(f"{experts} experts do not divide over {ranks} ranks")
    if experts // ranks > MAX_LOCAL_EXPERTS:
        raise SystemExit(
            f"{experts // ranks} experts a rank, and the signal block holds "
            f"{MAX_LOCAL_EXPERTS}"
        )
    if (intermediate * shared) % 128 != 0:
        raise SystemExit("the shared experts' width is whole 128-element words")
    if request.field("multiprocessors") % 2 != 0:
        raise SystemExit("the persistent grid is whole clusters of two blocks")
    assert ranks <= MAX_RANKS and experts <= MAX_EXPERTS and top <= MAX_TOP
    limit_of(request)


def definitions(request: Request) -> dict[str, cuda.Definition]:
    """The extents every unit is compiled with."""
    limit = limit_of(request)
    return {
        "UNISERVE_HIDDEN": request.field("hidden"),
        "UNISERVE_INTERMEDIATE": request.field("intermediate"),
        "UNISERVE_TOP": request.field("top"),
        "UNISERVE_EXPERTS": request.field("experts"),
        "UNISERVE_MAX_TOKENS": request.field("max_tokens"),
        "UNISERVE_RANKS": request.field("ranks"),
        "UNISERVE_MULTIPROCESSORS": request.field("multiprocessors"),
        "UNISERVE_SHARED": request.field("shared"),
        # The clamp's FP32 bit pattern, zero for none.
        "UNISERVE_LIMIT_BITS": hex(
            struct.unpack("<I", struct.pack("<f", limit))[0]
        ),
    }


def compile(request: Request) -> None:
    """Compile the fused exchange of the request's descriptor into the module.

    A refused descriptor, a fetch failure or a compiler failure refuses the
    build, with its output in the build log.
    """
    check(request)
    found = runs(request)
    heights = sorted({height for height, _ in found})
    print(f"the BLOCK_M runs (height, largest step): {found}", flush=True)

    pinned = fetch.pin("deepgemm")
    with cuda.scratch(request) as staged:
        try:
            names = list(fetch.manifest(pinned, request.cache))
            fetch.fetch(pinned, request.cache, names, request.jobs)
            fetch.stage(pinned, request.cache, staged, names)
            headers = [
                *fetch.includes(pinned, staged),
                fetch.installed(*CUTLASS),
            ]
        except fetch.FetchError as error:
            raise SystemExit(
                f"staging the deepgemm pin failed: {error}"
            ) from error

        common = definitions(request)
        # The instances, once each, and the runs of steps that take them, as
        # a header the entry includes (nvcc splits a definition's value at
        # its commas).
        generated = staged / "generated"
        generated.mkdir()
        (generated / RUNS).write_text(
            "#pragma once\n"
            "#define UNISERVE_INSTANCES "
            + " ".join(f"X({height})" for height in heights)
            + "\n#define UNISERVE_RUNS "
            + " ".join(f"X({height}, {largest})" for height, largest in found)
            + "\n"
        )

        def unit(source: str, extra: dict[str, cuda.Definition]) -> cuda.Unit:
            # The kernel's body is tcgen05 code behind a guard of the
            # architecture number alone, which no portable PTX can hold.
            return cuda.Unit(
                source=request.package / source,
                definitions={**common, **extra},
                includes=[generated, *headers],
                specific=True,
                **pinned.compile,
            )

        units = [
            unit(ENTRY, {}),
            unit(STAGE, {}),
            *(
                unit(
                    KERNEL,
                    {
                        "UNISERVE_BLOCK_M": height,
                        "UNISERVE_INSTANCE": f"uniserve_mega_moe_{height}",
                    },
                )
                for height in heights
            ),
        ]
        # The entry encodes the kernel's tensor maps through the driver API.
        cuda.native(request, units, libraries=["cuda"])
