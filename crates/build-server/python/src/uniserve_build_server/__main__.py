"""The ``uniserve-build-server`` program: the Build Server's environment.

The Build Server (``uniserve build-server``, ``crates/build-server``) runs it
for four things:

``uniserve-build-server toolchain --json``
    prints the toolchain description (:mod:`uniserve_build_server.toolchain`),
    which the Build Server reads once when it starts.

``python -m uniserve_build_server compile``
    compiles one descriptor from the request on standard input
    (:mod:`uniserve_build_server.request`). The Build Server runs it with this
    environment's interpreter in the package's source directory, so the
    package's modules import as the interpreter imports any program's own. A
    Triton package's launch table is compiled by
    :mod:`uniserve_build_server.triton`; a CuTe DSL package's build module
    states ``spec(fields, parameters, target)``, which
    :mod:`uniserve_build_server.cute_dsl` compiles; a CUDA package's declared
    compilation is compiled by :mod:`uniserve_build_server.cuda`, and the
    build module of a CUDA package whose compilation is computed states
    ``compile(request)``, which calls that driver's routines.

``uniserve-build-server provision PIN --cache DIR``
    fetches and verifies a pinned third-party source into the Build Server's
    cache (:mod:`uniserve_build_server.fetch`).

``uniserve-build-server launch``
    answers each descriptor of one Triton package on standard input with a JSON
    line stating the compilation its launch table gives it
    (:func:`uniserve_build_server.triton.launches`): what
    ``uniserve build-server launch <package> --target <target>`` prints for a
    check that launches the package's kernel outside a build.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys
from collections.abc import Callable

from uniserve_build_server import cuda, cute_dsl, fetch, toolchain, triton
from uniserve_build_server.request import Request


def compile_cuda(request: Request) -> None:
    """A CUDA package's declared compilation, or its build module's."""
    if request.module is None:
        cuda.build(request)
    else:
        importlib.import_module(request.module).compile(request)


#: How each source kind is compiled.
DRIVERS: dict[str, Callable[[Request], None]] = {
    "triton": triton.build,
    "cute_dsl": lambda request: cute_dsl.build(
        request, importlib.import_module(request.module).spec
    ),
    "cuda": compile_cuda,
}


def compile_request(request: Request) -> None:
    """Compile the descriptor ``request`` names into its output directory."""
    driver = DRIVERS.get(request.source)
    if driver is None:
        raise SystemExit(
            f"a {request.source} package is not compiled through this "
            f"environment; the compiled source kinds are {sorted(DRIVERS)}"
        )
    request.output.mkdir(parents=True, exist_ok=True)
    driver(request)


def describe_toolchain(as_json: bool) -> None:
    described = toolchain.describe()
    if as_json:
        print(described.to_json())
        return
    print(f"interpreter    {described.interpreter}")
    print(f"driver         {described.driver}")
    for name in (
        "kernels",
        "triton",
        "cute_dsl",
        "cuda",
        "cudnn",
        "host_compiler",
    ):
        entry = getattr(described, name)
        state = entry.get("available", f"missing: {entry.get('missing')}")
        print(f"{name:<14} {state}")


def provision(options: argparse.Namespace) -> int:
    source = fetch.pin(options.pin)
    try:
        files = fetch.manifest(source, options.cache, options.offline)
        names = list(files) if options.all else options.file
        fetch.fetch(source, options.cache, names, options.jobs, options.offline)
    except fetch.FetchError as error:
        print(f"{source.name}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(fetch.describe(source, options.cache, files), indent=2))
    return 0


def arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="uniserve-build-server",
        description="the Build Server's toolchain environment",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    described = commands.add_parser(
        "toolchain", help="describe this environment's toolchains"
    )
    described.add_argument(
        "--json",
        action="store_true",
        help="print the description the Build Server reads",
    )

    commands.add_parser("compile", help="compile the request on standard input")

    commands.add_parser(
        "launch",
        help="state the launches of the Triton descriptors on standard input",
    )

    provisioned = commands.add_parser(
        "provision", help="fetch a pinned source into the cache"
    )
    provisioned.add_argument(
        "pin", choices=sorted(fetch.pins()), help="the pinned source"
    )
    provisioned.add_argument(
        "--cache",
        type=pathlib.Path,
        required=True,
        help="the Build Server's cache of pins",
    )
    provisioned.add_argument(
        "--file",
        action="append",
        default=[],
        metavar="NAME",
        help="a file, by manifest path",
    )
    provisioned.add_argument(
        "--all", action="store_true", help="every file of the pin"
    )
    provisioned.add_argument(
        "--jobs", type=int, default=1, help="files fetched at once"
    )
    provisioned.add_argument(
        "--offline", action="store_true", help="use only the cache"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    options = arguments(argv)
    if options.command == "toolchain":
        describe_toolchain(options.json)
        return 0
    if options.command == "compile":
        compile_request(Request.read(sys.stdin))
        return 0
    if options.command == "launch":
        triton.launches(sys.stdin, sys.stdout)
        return 0
    return provision(options)


if __name__ == "__main__":
    raise SystemExit(main())
