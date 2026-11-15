"""The deepep package's build module: one descriptor -> one module.

The descriptor fixes every template argument of DeepEP's kernels, so this
compiles them here, and a serving host compiles nothing. The Build Server
compiles a descriptor by calling :func:`compile` with the compile request.

DeepEP's kernels and the NCCL they are built against are pinned sources
(`uniserve_build_server.fetch`'s `deepep` and `nccl` pins). The pins state
what a compilation against them is given: their include roots, NCCL's library
and how DeepEP compiles its kernels. The CUDA driver
(`uniserve_build_server.cuda`) compiles the package's one unit with them and
links it against the pinned NCCL and the CUDA runtime.
"""

from uniserve_build_server import cuda, fetch
from uniserve_build_server.request import Request

#: The translation unit of the module: every entry, and the kernels' launches.
UNIT = "src/exchange.cu"


def definitions(request):
    """The compile-time values a descriptor states, by the macro each defines.

    The direction and the element type are the manifest's single choices, so
    only the extents reach the compile.
    """
    return {
        "UNISERVE_HIDDEN": request.field("hidden"),
        "UNISERVE_TOP": request.field("top"),
        "UNISERVE_EXPERTS": request.field("experts"),
        "UNISERVE_MAX_TOKENS": request.field("max_tokens"),
        "UNISERVE_RANKS": request.field("ranks"),
        "UNISERVE_ATTENTION_RANKS": request.field("attention"),
    }


def compile(request: Request) -> None:
    """Compile the exchange for the request's descriptor into ``module.so``.

    The pinned sources are fetched into the Build Server's cache of pins when
    it does not hold them yet; a compiler failure refuses the build, with the
    compiler's output in the build log.
    """
    cache = request.cache
    try:
        deepep = fetch.pin("deepep")
        nccl = fetch.pin("nccl")
        fetch.fetch(deepep, cache, jobs=request.jobs)
        fetch.fetch(nccl, cache, jobs=request.jobs)
        library = fetch.library(nccl, nccl.directory(cache))
    except fetch.FetchError as error:
        raise SystemExit(str(error)) from error

    # The unit is compiled as DeepEP compiles its kernels, which its headers
    # are written for.
    unit = cuda.Unit(
        source=request.package / UNIT,
        definitions=definitions(request),
        includes=[
            *fetch.includes(deepep, deepep.directory(cache)),
            *fetch.includes(nccl, nccl.directory(cache)),
        ],
        **deepep.compile,
    )

    with cuda.scratch(request) as directory:
        objects = cuda.objects(request, [unit], directory)
        # The module needs NCCL by the soname the pinned library states: the
        # serving process's loader resolves it to the NCCL `crates/platform`
        # opens.
        cuda.link(
            request,
            objects,
            request.output / cuda.MODULE_SO,
            standard=unit.standard,
            shared=[library],
        )
