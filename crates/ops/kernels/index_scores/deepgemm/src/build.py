"""The index_scores package's DeepGEMM build module: one descriptor -> one module.

The module is DeepGEMM's SM100 paged MQA logits over MXFP4 operands,
``sm100_paged_mqa_logits`` of the pinned headers, instantiated by the
package's entry unit (``src/scores.cu``) for the descriptor's heads, head
width and keys of a page with DeepGEMM's own geometry for MXFP4 operands (its
``MQALogitsConfig``, ``csrc/jit_kernels/impls/sm100_mqa_logits.hpp`` at the
pinned commit), which the entry restates.

The steps:

1. fetch the ``deepgemm`` pin and stage a copy with
   ``patches/paged-logits.patch`` applied: the epilogue's BF16 rounding, the
   page-table row of a request's tokens, the request ends the schedule
   writes, and the schedule over the launched multiprocessors;
2. compile the entry unit with the descriptor's extents as its definitions,
   against the staged headers and the CUTLASS of the ``flashinfer-python``
   distribution, as DeepGEMM's run-time compiler compiles its kernels (the
   pin's ``[compile]``), and link it with the driver API into ``module.so``.

The Build Server compiles a descriptor by calling :func:`compile` with the
compile request.
"""

from __future__ import annotations

from uniserve_build_server import cuda, fetch
from uniserve_build_server.request import Request

#: The package's unit: the operator's entry and the kernel's instance.
UNIT = "src/scores.cu"

#: The package's patch of the pinned kernel and scheduler.
PATCH = "patches/paged-logits.patch"

#: Where the CUTLASS the kernel includes is installed.
CUTLASS = ("flashinfer-python", "flashinfer/data/cutlass/include")


def compile(request: Request) -> None:
    """Compile the scoring of the request's descriptor into ``module.so``.

    A fetch, patch or compiler failure refuses the build, with its output in
    the build log.
    """
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
            definitions={
                "UNISERVE_HEADS": request.field("heads"),
                "UNISERVE_WIDTH": request.field("width"),
                "UNISERVE_GROUPS": request.field("groups"),
            },
            includes=headers,
            specific=True,
            **pinned.compile,
        )
        # The entry encodes the kernel's tensor maps through the driver API.
        cuda.native(request, [unit], libraries=["cuda"])
