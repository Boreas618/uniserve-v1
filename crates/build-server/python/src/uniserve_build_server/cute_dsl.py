"""The CuTe DSL compile driver: a package's ``spec`` to a native ``module.so``.

The Build Server compiles a CuTe DSL package's descriptor by handing its request
to :func:`build` with the package's ``spec``, and keeps what it writes into the
request's output directory: ``module.so``, a shared object with the C entry the
package declares and the kernel's cubin embedded.

Steps: compile the kernel for the request's target (no GPU is read: the target
is stated to the DSL, and fake tensors and a fake stream describe the runtime
signature), ``export_to_c`` to emit the host launcher object and its header,
then compile the package's entry unit beside that header and link both with the
CuTe DSL's ``libcuda_dialect_runtime_static.a``, whose members the launcher
calls, through the CUDA driver's routines (:mod:`uniserve_build_server.cuda`:
``objects`` and ``link``, which links the CUDA runtime with every module).

A package's build module states ``spec(request)``, the compilation of the
descriptor the request names (:class:`uniserve_build_server.request.Request`:
its typed fields, the manifest's parameters, the target and the installed
package, whose tuning data a package reads), one shape for every package:

- ``name``: the exported launcher's name, which the entry unit calls;
- ``compile``: a callable that returns the compiled launcher, :func:`kernel`
  of the package's launcher (its own kernel, or a vendor's kernel class it
  constructs with an explicit configuration) and the fake arguments
  :func:`pointer`, :func:`tensor` and :func:`scalar` describe;
- ``definitions`` (optional): preprocessor definitions of the entry unit, by
  macro name (:data:`uniserve_build_server.cuda.Definition`).

The entry unit is the C++ translation unit the manifest names in ``[source]
unit``, which defines the C entry ``[module.entries]`` names; it finds the
export's header through the include path, beside the headers every native
compile gets (:func:`uniserve_build_server.cuda.includes`): the operator's ABI
header, generated from the operator crate, and the shared entry header
``uniserve_cute_dsl.h`` of the kernel library.

The module is a **shared library with an entry symbol**, not a cubin ``ops``
launches: the call's parameters are the call structure the ABI header declares,
whose layout its ``static_assert`` lines fix, so no launch signature is written
to a ``metadata.json``.
"""

from __future__ import annotations

import importlib.metadata
import pathlib
from collections.abc import Callable
from typing import Any

import cutlass.cute as cute
from cutlass import Numeric
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor, make_ptr
from cutlass.cute.typing import Pointer, Tensor
from cutlass.cutlass_dsl import CuTeDSL

from uniserve_build_server import cuda
from uniserve_build_server.request import Request, Target
from uniserve_build_server.toolchain import Cuda

#: What a package's ``spec`` is: the request, to the compilation of the one
#: descriptor it names.
Spec = Callable[[Request], dict[str, Any]]

#: The CuTe DSL's static device-dialect runtime, which every module links.
DIALECT_RUNTIME = "libcuda_dialect_runtime_static.a"

#: The C++ standard the entry units are written in.
STANDARD = "c++20"


def configure(target: Target) -> None:
    """State ``target`` as the architecture this process compiles for.

    The DSL reads its architecture from its configuration object when it
    traces a kernel, decides whether a JIT engine can run the result and picks
    the cubin's chip; stating it there means no device is probed, and a
    package whose vendor compiles its kernel compiles for the same target.
    """
    CuTeDSL._get_dsl().envar.arch = target.name


def kernel(
    compiled: Any, args: list[Any], options: dict[str, Any]
) -> Callable[[], Any]:
    """The ``compile`` of a kernel the DSL compiles for fake arguments."""
    return lambda: cute.compile(
        compiled, *args, **options, stream=make_fake_stream()
    )


def pointer(dtype: type[Numeric], align: int) -> Pointer:
    """A fake device pointer to ``dtype`` elements, ``align`` bytes aligned.

    The compiled function takes the pointer at launch: the compilation is told
    its element type, its address space and the alignment the kernel's accesses
    rely on, and the address itself is never read.
    """
    return make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=align)


def tensor(
    dtype: type[Numeric], rank: int, leading: int, align: int | None
) -> Tensor:
    """A fake device tensor of ``rank`` modes whose layout the launch states.

    Every extent is a 32-bit runtime word and every stride a 64-bit one, but
    the ``leading`` mode's stride, which is 1: the compiled function takes the
    tensor's base, extents and strides at launch (the export's tensor struct),
    and only the unit stride is compiled in. ``align`` is the alignment of the
    base in bytes the kernel's accesses rely on; ``None`` states one element's.
    """
    return make_fake_tensor(
        dtype,
        tuple(cute.sym_int32() for _ in range(rank)),
        tuple(
            1 if mode == leading else cute.sym_int64() for mode in range(rank)
        ),
        assumed_align=align,
    )


def scalar(dtype: type[Numeric]) -> Numeric:
    """A fake scalar of ``dtype`` for a parameter the function takes at launch.

    A parameter annotated with a numeric type is a runtime word of the compiled
    function, so the value the compilation is handed only states its type; no
    problem size is compiled in.
    """
    return dtype(0)


def dialect_runtime(cuda: Cuda) -> pathlib.Path:
    """The dialect runtime archive of the toolkit's CUDA major release.

    It is a file of the locked ``nvidia-cutlass-dsl-libs-cu<major>``
    distribution, found through the distribution's own file list.
    """
    distribution = f"nvidia-cutlass-dsl-libs-cu{cuda.version.split('.')[0]}"
    try:
        listed = importlib.metadata.files(distribution) or []
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit(
            f"this environment has no {distribution} distribution"
        ) from None
    for file in listed:
        if file.name == DIALECT_RUNTIME:
            return pathlib.Path(file.locate())
    raise SystemExit(f"{distribution} holds no {DIALECT_RUNTIME}")


def build(request: Request, spec: Spec) -> None:
    """Compile the descriptor ``request`` names into its output directory."""
    configure(request.target)
    stated = spec(request)
    compiled = stated["compile"]()

    # The Build Server resolves a CuTe DSL source's entry unit (``[source]
    # unit``, ``src/entry.cpp`` unless the manifest states another).
    if request.unit is None:
        raise SystemExit("the request names no entry unit")
    runtime = dialect_runtime(request.toolchain.require_cuda())
    # The export and the entry unit's object are the build's own
    # intermediates, in its scratch directory.
    with cuda.scratch(request) as staging:
        name = stated["name"]
        compiled.export_to_c(str(staging), name)
        (entry,) = cuda.objects(
            request,
            [
                cuda.Unit(
                    source=request.package / request.unit,
                    standard=STANDARD,
                    definitions=stated.get("definitions", {}),
                    includes=[staging],
                )
            ],
            staging,
        )
        # The archive gives the dialect runtime's members the launcher calls.
        cuda.link(
            request,
            [staging / f"{name}.o", entry, runtime],
            request.output / cuda.MODULE_SO,
            standard=STANDARD,
        )
