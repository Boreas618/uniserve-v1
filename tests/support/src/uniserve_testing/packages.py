"""The policy packages a test serves and applies.

A deployment's whole configuration is a policy package
(``docs/configuration.md``): a directory holding ``config.toml``, the tree
of options, and optionally ``programs/``. ``uniserve serve <package>``
starts the deployment it states, the directory's name naming the
deployment, and ``PUT /admin/policy`` takes the next package as a tar
archive of such a directory (:mod:`uniserve_testing.admin`).

A test states a configuration as the tree ``config.toml`` holds: nested
mappings for its tables, lists for its arrays (an array of tables a list of
mappings), and strings, numbers and booleans for its values, as
:mod:`tomllib` reads the file and :mod:`tomli_w` writes it. :func:`merged`
composes such trees and :func:`without` removes options from one,
:func:`write` writes one as a package directory and :func:`archive` as the
archive a ``PUT`` carries.
"""

from __future__ import annotations

import io
import os
import tarfile
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import tomli_w

#: The file of a package that holds its configuration.
CONFIG = "config.toml"

#: The directory of a package that holds its programs' sources.
PROGRAMS = "programs"

#: A configuration tree: what a package's ``config.toml`` states.
Config = Mapping[str, Any]


def merged(*configs: Config) -> dict[str, Any]:
    """The configuration `configs` state together, a later one over an
    earlier one: a table two of them state holds the options of both, and an
    option two of them state, or an array, takes the later value whole.
    """  # noqa: D205
    result: dict[str, Any] = {}
    for config in configs:
        for key, value in config.items():
            earlier = result.get(key)
            if isinstance(earlier, Mapping) and isinstance(value, Mapping):
                result[key] = merged(earlier, value)
            elif isinstance(value, Mapping):
                result[key] = merged(value)
            else:
                result[key] = value
    return result


def without(config: Config, *removed: str) -> dict[str, Any]:
    """`config` less the options `removed` names, each a dotted path of
    tables to an option or a table (``profiling_config.session``); a path
    `config` does not state is no change.
    """  # noqa: D205
    result = merged(config)
    for path in removed:
        *tables, option = path.split(".")
        table: Any = result
        for name in tables:
            table = table.get(name)
            if not isinstance(table, dict):
                break
        else:
            table.pop(option, None)
    return result


def text(config: Config) -> str:
    """`config` as the text of a ``config.toml``."""
    return tomli_w.dumps(dict(config))


def read(package: Path) -> dict[str, Any]:
    """The configuration of the package in the directory `package`."""
    return tomllib.loads((package / CONFIG).read_text())


def write(package: Path, config: Config) -> Path:
    """Write `config` as the ``config.toml`` of the package directory
    `package`, created when missing; the directory.
    """  # noqa: D205
    package.mkdir(parents=True, exist_ok=True)
    (package / CONFIG).write_text(text(config))
    return package


def archive(config: Config, programs: Path | None = None) -> bytes:
    """The package of `config` and the program sources under the directory
    `programs` as the tar archive ``PUT /admin/policy`` carries: its
    ``config.toml`` and every file under its ``programs/``.
    """  # noqa: D205
    buffer = io.BytesIO()
    with tarfile.open(
        fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT
    ) as tar:
        encoded = text(config).encode()
        entry = tarfile.TarInfo(CONFIG)
        entry.size = len(encoded)
        entry.mode = 0o644
        tar.addfile(entry, io.BytesIO(encoded))
        if programs is not None:
            for root, _, files in os.walk(programs):
                for name in sorted(files):
                    path = Path(root, name)
                    relative = path.relative_to(programs)
                    tar.add(path, arcname=str(Path(PROGRAMS, relative)))
    return buffer.getvalue()


def unpacked(archive_bytes: bytes) -> dict[str, Any]:
    """The configuration of the package the tar archive `archive_bytes`
    holds, as ``GET /admin/policy`` answers it.
    """  # noqa: D205
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r") as tar:
        for member in tar.getmembers():
            if Path(member.name) == Path(CONFIG):
                extracted = tar.extractfile(member)
                assert extracted is not None, "config.toml is no file"
                return tomllib.loads(extracted.read().decode())
    raise ValueError(f"the package archive holds no {CONFIG}")


def on_hosts(config: Config, hosts: Sequence[str]) -> dict[str, Any]:
    """`config`, a deployment over the hosts its ``hosts`` lists, moved to
    `hosts`, host for host by position: its ``hosts`` becomes `hosts`, and a
    rank of its ``worker_groups`` placed on its n-th host
    (``<host>/<device>``) is placed on the n-th of `hosts`.

    A checked-in package of several hosts names the hosts it was written
    for; a case serves it on the hosts it runs on.

    Raises:
        ValueError: When the two lists differ in length, or a rank names a
            host the package does not list.
    """  # noqa: D205
    stated = list(config.get("hosts", []))
    if len(stated) != len(hosts):
        raise ValueError(f"the package lists hosts {stated}, not {len(hosts)}")
    renamed = dict(zip(stated, hosts, strict=True))

    def placed(rank: str) -> str:
        host, separator, device = rank.rpartition("/")
        if not separator:
            return rank
        if host not in renamed:
            raise ValueError(f"rank {rank} lies on no host of {stated}")
        return f"{renamed[host]}/{device}"

    groups = [
        {**group, "ranks": [placed(rank) for rank in group.get("ranks", [])]}
        for group in config.get("worker_groups", [])
    ]
    moved = merged(config, {"hosts": list(hosts)})
    if groups:
        moved["worker_groups"] = groups
    return moved
