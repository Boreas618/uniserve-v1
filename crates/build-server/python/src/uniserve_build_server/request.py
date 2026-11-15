"""The compile request: everything one compilation is given.

The Build Server compiles one descriptor of one package for one target by
running ``uniserve-build-server compile`` with this request, as a JSON object
on standard input, and the working directory the package's build module lives
in (:mod:`uniserve_build_server.__main__`). The request is the whole interface:
a compilation reads no environment variable and looks nothing up, so what it
builds is a function of the request and of the files the request names.

- the package: its operator and backend, its installed root, its source kind
  and the build module its manifest names;
- the descriptor: its key and the fields the Build Server read out of the key
  against the package's declared dimensions (``Descriptors::fields`` in
  ``crates/build-server/src/manifest.rs`` is the one reader of the key grammar),
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
    #: The key's fields, each a declared dimension and its value, in the key's
    #: order. A dimension the key leaves out is absent: the package accounts for
    #: a field it needs and the key lacks.
    fields: dict[str, str]
    #: The manifest's ``[parameters]``.
    parameters: dict[str, Any]
    #: The manifest's source kind: ``triton``, ``cute_dsl`` or ``cuda``.
    source: str
    #: The package's build module, imported from the compilation's working
    #: directory.
    module: str
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

    def field(self, name: str) -> str:
        """One descriptor field, or a refusal naming the field the key lacks."""
        try:
            return self.fields[name]
        except KeyError:
            raise SystemExit(
                f"the descriptor {self.descriptor} names no {name}"
            ) from None
