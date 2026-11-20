"""The latent_attention flashmla package's build module.

One descriptor -> one module: FlashMLA's SM100 sparse decoding kernel for
DeepSeek-V4.1 (``head64``, the ``flashmla`` pin of
:mod:`uniserve_build_server.fetch`), instantiated by this package's unit
``src/attention.cu``, which also holds the operator's entry and the program
that writes each query's entry slots into the call's workspace.

The pin carries headers alone: the kernel's host launch is a template whose
instantiation the unit states, so the build stages the pin, compiles the unit
against it (and the CUTLASS the ``flashinfer-python`` distribution installs)
as FlashMLA compiles its own kernels (the pin's ``[compile]``), and links the
module with the driver API, which the kernel's tensor maps are encoded
through.

The descriptor decides one thing the compilation cannot vary: whether the
layer reads compressed entries beside its window, and in which format. The
pair of record formats is the kernel's ``ModelType`` pair -- MXFP8 window
records (``V41``) alone, or with NVFP4 compressed records (``V41_FP4``) --
and the unit instantiates the kernel for the pair this descriptor names. The
window, the pages, the selection's columns, the head count and the token
count are call extents the module reads.

The Build Server compiles a descriptor by calling :func:`compile` with the
compile request.
"""

from __future__ import annotations

from uniserve_build_server import cuda, fetch
from uniserve_build_server.request import Request

#: The package's unit: the operator's entry, the slot program and the kernel's
#: instantiation.
UNIT = "src/attention.cu"

#: Where the CUTLASS the kernel includes is installed.
CUTLASS = ("flashinfer-python", "flashinfer/data/cutlass/include")

#: The kernel's record format of a compressed side, by the format the
#: descriptor names; a window-only layer reads the window's format alone.
EXTRA_FORMATS = {"none": "V41", "nvfp4": "V41_FP4"}


def compile(request: Request) -> None:
    """Compile the attention for the request's descriptor into ``module.so``.

    The pin is fetched into the Build Server's cache of pins when the cache
    does not hold it yet; a fetch or compiler failure refuses the build, with
    its output in the build log.
    """
    compressed = request.field("compressed")
    pinned = fetch.pin("flashmla")
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
                f"staging the flashmla pin failed: {error}"
            ) from error

        # The kernel's body is tcgen05 code behind a guard of the architecture
        # number alone, which no portable PTX can hold; `NDEBUG` and
        # `_USE_MATH_DEFINES` are the definitions FlashMLA's own build gives
        # every one of its units.
        unit = cuda.Unit(
            source=request.package / UNIT,
            definitions={
                "NDEBUG": None,
                "_USE_MATH_DEFINES": None,
                "FLASH_MLA_IS_BUILD_ON_CUDA": None,
                "UNISERVE_EXTRA_FORMAT": EXTRA_FORMATS[compressed],
                "UNISERVE_COMPRESSED": compressed != "none",
            },
            includes=headers,
            specific=True,
            **pinned.compile,
        )
        # The kernel's tensor maps are encoded through the driver API.
        cuda.native(request, [unit], libraries=["cuda"])
