"""The CuTe DSL compile driver: a package's ``spec`` to a native ``module.so``.

The Build Server compiles a CuTe DSL package's descriptor by handing its request
to :func:`build` with the package's ``spec``, and keeps what it writes into the
request's output directory: ``module.so``, a shared object with the C entry the
package declares and the kernel's cubin embedded.

Steps: compile the kernel for the request's target (no GPU is read: the target
is stated to the DSL, and fake tensors and a fake stream describe the runtime
signature), ``export_to_c`` to emit the host launcher object and its header,
then compile the package's entry unit beside that header and link both with the
CuTe DSL's ``libcuda_dialect_runtime_static.a`` and the CUDA runtime. The
archive carries the device-dialect runtime; the CUDA runtime must come with it,
or the module loads with undefined ``cuda*`` symbols.

A package's build module states ``spec(fields, parameters, target)``, one shape
for every package:

- ``name``: the exported launcher's name, which the entry unit calls;
- ``compile``: a callable that returns the compiled kernel. A kernel the DSL
  compiles directly states :func:`kernel`; a package whose kernel is selected
  and compiled by its vendor's own code states that code;
- ``defines`` (optional): preprocessor definitions of the entry unit.

The entry unit is the C++ translation unit the manifest names in ``[source]
unit``, which defines the C entry ``[module.entries]`` names; it is compiled
where it is, so the ABI header beside it is found as its own include, and the
export's header through the include path.

The module is a **shared library with an entry symbol**, not a cubin ``ops``
launches: the call's parameters are the ``Call`` struct ``entry.h`` declares,
whose layout a ``static_assert`` fixes, so no launch signature is written to a
``metadata.json``.
"""

from __future__ import annotations

import importlib.metadata
import pathlib
import subprocess
import tempfile
from collections.abc import Callable
from typing import Any

import cutlass.cute as cute
from cutlass import Numeric
from cutlass.cute.runtime import make_fake_stream, make_ptr
from cutlass.cute.typing import Pointer
from cutlass.cutlass_dsl import CuTeDSL

from uniserve_build_server.request import Request, Target
from uniserve_build_server.toolchain import Cuda

#: What a package's ``spec`` is: the descriptor's fields, the manifest's
#: ``[parameters]`` and the target, to the compilation of one descriptor.
Spec = Callable[[dict[str, str], dict[str, Any], Target], dict[str, Any]]

#: The CuTe DSL's static device-dialect runtime, which every module links whole.
DIALECT_RUNTIME = "libcuda_dialect_runtime_static.a"

#: The flags every C++ translation unit of a module is compiled with.
CXX_FLAGS = ["-O2", "-std=c++20", "-fPIC"]


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


def compile_cxx(compiler: str, arguments: list[str], what: str) -> None:
    """Run the host C++ compiler; a failure refuses the build."""
    built = subprocess.run(
        [compiler, *CXX_FLAGS, *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if built.returncode != 0:
        raise SystemExit(f"{what} failed:\n{built.stderr}")


def build(request: Request, spec: Spec) -> None:
    """Compile the descriptor ``request`` names into its output directory."""
    configure(request.target)
    try:
        stated = spec(request.fields, request.parameters, request.target)
    except KeyError as absent:
        raise SystemExit(
            f"{request.descriptor}: the descriptor names no {absent}"
        ) from None
    compiled = stated["compile"]()

    cuda = request.toolchain.require_cuda()
    compiler = request.toolchain.require_host_compiler()
    # The Build Server resolves a CuTe DSL source's entry unit (``[source]
    # unit``, ``src/entry.cpp`` unless the manifest states another).
    if request.unit is None:
        raise SystemExit("the request names no entry unit")
    out = request.output
    # The export and the entry unit's object are the build's own intermediates:
    # they live in a directory of the build's workspace the module does not
    # publish (a dot directory).
    with tempfile.TemporaryDirectory(dir=out, prefix=".") as scratch:
        staging = pathlib.Path(scratch)
        name = stated["name"]
        compiled.export_to_c(str(staging), name)
        unit = staging / "entry.o"
        compile_cxx(
            compiler,
            [
                "-c",
                "-o",
                str(unit),
                str(request.package / request.unit),
                *(f"-D{one}" for one in stated.get("defines", [])),
                f"-I{staging}",
                f"-I{cuda.include}",
            ],
            "compiling the entry unit",
        )
        # The dialect runtime archive is linked whole and the CUDA runtime
        # dynamically with `--no-as-needed`: without the runtime the module
        # loads with undefined `cuda*` symbols, and without `--no-as-needed`
        # the linker drops it because the archive references it only
        # indirectly.
        compile_cxx(
            compiler,
            [
                "-shared",
                "-o",
                str(out / "module.so"),
                str(staging / f"{name}.o"),
                str(unit),
                "-Wl,--whole-archive",
                str(dialect_runtime(cuda)),
                "-Wl,--no-whole-archive",
                "-Wl,--no-as-needed",
                f"-L{cuda.lib}",
                "-lcudart",
            ],
            "linking module.so",
        )
