"""The compile request: everything one compilation is given.

The Build Server compiles one descriptor of one package for one target by
running ``uniserve-build-server compile`` with this request, as a JSON object
on standard input, and the working directory the package's build module lives
in (:mod:`uniserve_build_server.__main__`). The request is the whole interface:
a compilation reads no environment variable and looks nothing up, so what it
builds is a function of the request and of the files the request names.

- the package: its operator and backend, its installed root, its source kind,
  the module the compilation imports (a Triton package's kernel module, or the
  build module ``src/build.py``), a Triton package's launch table and a CUDA
  package's declared compilation;
- the descriptor: its key and the fields the Build Server read out of the key
  against the package's declared dimensions, each typed as its dimension
  declares it (``Descriptors::typed_fields`` in
  ``crates/build-server/src/manifest.rs`` is the one reader of the key
  grammar): a number, a list of numbers, a float, a boolean flag or a name;
  and the manifest's parameters;
- the target, with the hardware facts of its architecture
  (``crates/build-server/targets.toml``), so a package states requirements
  rather than hardware facts;
- the output directory, the number of compiler processes the build may run
  (the compile queue's limit), the Build Server's cache of pinned sources
  (:mod:`uniserve_build_server.fetch`), and the toolchain the Build Server
  checked against the package's declared version
  (:class:`uniserve_build_server.toolchain.Toolchain`).
"""

from __future__ import annotations

import dataclasses
import importlib.resources
import json
import pathlib
import tomllib
from typing import IO, Any

from uniserve_build_server.toolchain import Toolchain


@dataclasses.dataclass(frozen=True)
class Target:
    """What a module is compiled for, as the Build Server describes it.

    ``name`` is the target's name in the module store, such as ``sm_100a``,
    and what ``str()`` gives, so a command line states ``-arch={target}``.
    """

    name: str
    #: The code generator's backend, ``cuda``.
    backend: str
    #: The compute capability as one number: ``100`` for 10.0.
    capability: int
    #: Whether the target is the architecture-specific instruction set of
    #: exactly its capability, which a device of that capability is built for.
    specific: bool
    #: The threads of a warp.
    warp_size: int
    #: The architecture families whose code runs on this target, most specific
    #: first, such as ``("sm_100a", "sm_100f")``.
    families: tuple[str, ...]
    #: The largest shared memory one block may opt into, in bytes.
    shared_memory: int
    #: Whether the target has tensor memory (``tcgen05``).
    tensor_memory: bool
    #: Whether the target has the asynchronous warpgroup matrix instructions
    #: (``wgmma``).
    warpgroup_mma: bool
    #: Whether a kernel may launch dependent grids (programmatic dependent
    #: launch).
    dependent_launch: bool
    #: Whether the tensor cores take FP8 operands.
    fp8: bool

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Target:
        return cls(**{**data, "families": tuple(data["families"])})

    @classmethod
    def named(cls, name: str) -> Target:
        """The target ``name`` with its facts, for a spec outside a build.

        A build is handed its target in its request; this reads the same
        description, ``targets.toml``.
        """
        described = tomllib.loads(
            importlib.resources.files(__package__)
            .joinpath("targets.toml")
            .read_text()
        )
        if name not in described:
            raise SystemExit(
                f"no target {name!r}; the targets are {sorted(described)}"
            )
        return cls.from_json({"name": name, **described[name]})


@dataclasses.dataclass(frozen=True)
class Request:
    """One compilation: one descriptor of one package for one target."""

    op: str
    backend: str
    #: The descriptor key, which names the module in the store.
    descriptor: str
    #: The key's fields, each a declared dimension and its typed value, in the
    #: key's order. A dimension the key leaves out is absent: the package
    #: accounts for a field it needs and the key lacks.
    fields: dict[str, Any]
    #: The manifest's ``[parameters]``.
    parameters: dict[str, Any]
    #: The manifest's source kind: ``triton``, ``cute_dsl`` or ``cuda``.
    source: str
    #: The module the compilation imports from the package's source directory,
    #: its working directory: a Triton package's kernel module ``<op>``, or the
    #: build module ``build``; ``None`` for a CUDA package's declared
    #: compilation, which imports nothing of the package.
    module: str | None
    #: A Triton package's launch table (``crates/build-server/src/launch.rs``).
    launch: dict[str, Any] | None
    #: A CUDA package's declared compilation (``[compile]``,
    #: ``crates/build-server/src/compile.rs``), which
    #: :mod:`uniserve_build_server.cuda` runs.
    compile: dict[str, Any] | None
    #: A native module's C entry symbols by role, as the manifest resolves
    #: them: ``launch`` is the operator's ``uniserve_<op>`` unless the manifest
    #: names another.
    entries: dict[str, str]
    #: The C++ translation unit that defines a native module's C entries, for
    #: a source whose driver compiles it (``[source] unit``), relative to the
    #: package root.
    unit: str | None
    #: The installed package's root.
    package: pathlib.Path
    #: The directory the compilation writes the module's files into.
    output: pathlib.Path
    target: Target
    #: How many compiler processes the compilation may run at once.
    jobs: int
    #: The Build Server's cache of pinned third-party sources.
    cache: pathlib.Path
    toolchain: Toolchain

    @classmethod
    def read(cls, stream: IO[str]) -> Request:
        """The request the Build Server wrote to ``stream``."""
        data = json.load(stream)
        return cls(
            **{
                **data,
                "package": pathlib.Path(data["package"]),
                "output": pathlib.Path(data["output"]),
                "cache": pathlib.Path(data["cache"]),
                "target": Target.from_json(data["target"]),
                "toolchain": Toolchain.from_json(data["toolchain"]),
            }
        )

    def field(self, name: str) -> Any:
        """One descriptor field, or a refusal naming the field the key lacks."""
        try:
            return self.fields[name]
        except KeyError:
            raise SystemExit(
                f"the descriptor {self.descriptor} names no {name}"
            ) from None

    def tuned(self, data: str) -> dict[str, Any]:
        """The configuration the package's tuning data holds for the descriptor.

        Tuning data (design section 5.4, the manifest's ``tuning``) is a TOML
        file of the package, ``data`` relative to its root: an array of
        ``[[configuration]]`` tables in the order a descriptor tries them. A
        table may name ``descriptors``, field values as the descriptor's typed
        fields spell them; the descriptor takes the first table whose
        ``descriptors`` its fields all equal (a table naming none fits every
        descriptor), and the table's other keys are its configuration. A
        descriptor no table fits is refused.
        """
        try:
            tables = tomllib.loads((self.package / data).read_text())
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise SystemExit(f"the tuning data {data}: {error}") from None
        for table in tables.get("configuration", []):
            wanted = table.get("descriptors", {})
            if all(
                self.fields.get(name) == value for name, value in wanted.items()
            ):
                return {
                    name: value
                    for name, value in table.items()
                    if name != "descriptors"
                }
        raise SystemExit(
            f"the tuning data {data} has no configuration of {self.descriptor}"
        )
