"""The sparse_attention tcgen05 package's build module.

One descriptor -> one module: FastVideo's block-sparse attention forward pass
for data-center Blackwell, the `fastvideo` pin of
:mod:`uniserve_build_server.fetch`, launched by this package's unit
`src/attention.cu` and linked into a shared object exporting
`uniserve_sparse_attention`.

The pinned kernel walks a pair of adjacent query tiles per work item and reads
the partner tile's selection count and writes its normalizer unconditionally,
so FastVideo's own launch refuses an odd number of query tiles. The operator's
calls have any number, so the build compiles a staged copy of the pin with
`patches/unpaired-tile.patch` applied: the absent partner of a head's trailing
tile selects no key block, reads no metadata of the next head and writes no
normalizer row past the query rows. The change is owed upstream; until FastVideo
carries it, the patch is the package's.

The kernel reads the geometry, the strides and the selection from the call, so
the descriptor decides only what a compilation cannot vary: the sparse block, a
compile-time constant of the kernel (`VSA_BLK128`), from the descriptor's
`block`.

The Build Server compiles a descriptor by calling :func:`compile` with the
compile request.
"""

from uniserve_build_server import cuda, fetch
from uniserve_build_server.request import Request

#: The package's unit: the operator's entry and the kernel's launch.
SOURCE = "src/attention.cu"

#: The package's patch of the pinned kernel.
PATCH = "patches/unpaired-tile.patch"


def compile(request: Request) -> None:
    """Compile the kernel for the request's descriptor into ``module.so``.

    The pin is fetched into the Build Server's cache of pins when the cache
    does not hold it yet and staged as copies the patch changes; a fetch, patch
    or compiler failure refuses the build, with its output in the build log.
    """
    pinned = fetch.pin("fastvideo")
    with cuda.scratch(request) as staged:
        try:
            names = list(fetch.manifest(pinned, request.cache))
            fetch.fetch(pinned, request.cache, names, request.jobs)
            fetch.stage(pinned, request.cache, staged, names, copy=True)
            fetch.apply_patch(staged, request.package / PATCH)
            headers = fetch.includes(pinned, staged)
        except fetch.FetchError as error:
            raise SystemExit(
                f"staging the fastvideo pin failed: {error}"
            ) from error

        unit = cuda.Unit(
            source=request.package / SOURCE,
            standard="c++17",
            definitions={"VSA_BLK128": request.field("block") == 128},
            includes=headers,
        )
        # The kernel's tensor maps are encoded through the driver API.
        cuda.native(request, [unit], libraries=["cuda"])
