"""Pinned third-party sources: fetch, verify, cache and stage them.

A kernel package that builds against code it does not own -- TensorRT-LLM's
trtllm-gen kernel libraries, DeepEP's dispatch and combine kernels, NCCL's
device-API headers, FlashInfer's MegaMoE kernel -- names that code by a **pin**
instead of carrying a copy. Every pin is addressed the same way:

- its identity is the sha256 of a **manifest**, one ``<sha256>  <path>`` line
  per file;
- the manifest is either published beside the files (trtllm-gen's
  ``checksums.txt``) or kept in this package (``pins/<name>.sha256``);
- every file is verified against the manifest before it is used, and kept under
  ``~/.cache/uniserve/build-server/artifacts/<identity>/<path>`` (or
  ``$UNISERVE_ARTIFACT_CACHE``), so a cached file is reused offline and a file
  that does not verify is fetched again.

What differs between pins is only where a file comes from (its :class:`Origin`):
a repository that serves each file under a base URL, one archive whose own
sha256 is pinned, or a distribution installed in this interpreter at a pinned
version.

    python -m uniserve_build_server.fetch PIN [--file NAME ...] [--all] [--out
    DIR] [--offline]

fetches the named files (``--all``: every file) into the cache, stages them into
``DIR`` as symbolic links when ``--out`` is given, and prints a JSON description
of the pin.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable
from pathlib import Path

#: Where NVIDIA publishes the trtllm-gen kernel libraries' generic artifacts.
TRTLLM_REPOSITORY = os.environ.get(
    "UNISERVE_ARTIFACT_REPOSITORY",
    "https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/",
)

#: The directory of this package's own manifests.
PINS = Path(__file__).resolve().parent / "pins"

#: How many files are fetched at once.
WORKERS = 16

#: How many times one file is asked for. A repository resets a connection now
#: and then under `WORKERS` parallel fetches, and that says nothing about the
#: file; a reset fails the build only when it keeps happening.
ATTEMPTS = 5


class FetchError(RuntimeError):
    """A pinned source cannot be fetched, verified or staged."""


@dataclasses.dataclass(frozen=True)
class Files:
    """A repository serving each file of the manifest at ``<base><path>``."""

    base: str


@dataclasses.dataclass(frozen=True)
class Archive:
    """One zip archive (a wheel, say) whose own sha256 is pinned.

    The manifest's paths are members of it. The archive is read once and not
    kept: only the files the manifest lists enter the cache.
    """

    url: str
    sha256: str


@dataclasses.dataclass(frozen=True)
class Distribution:
    """A distribution installed in this interpreter at a pinned version.

    The manifest's paths are relative to its installation root
    (``site-packages``).
    """

    name: str
    version: str


Origin = Files | Archive | Distribution


@dataclasses.dataclass(frozen=True)
class Pin:
    """One pinned source."""

    #: The name packages and the command line use.
    name: str
    #: The sha256 of the manifest, which identifies the pin and names its cache
    #: directory.
    sha256: str
    #: Where the files come from; ``None`` when this machine has no pinned
    #: origin.
    origin: Origin | None
    #: The manifest's name in the origin, for a manifest published beside the
    #: files. ``None`` keeps the manifest in this package, as
    #: ``pins/<name>.sha256``.
    manifest: str | None = None


#: NCCL 2.30.4's headers, the device API DeepEP's kernels are built against,
#: from the `nvidia-nccl-cu13` wheel the serving environment pins (`uv.lock`).
#: The Build Server's own environment cannot install that wheel beside its
#: torch, which requires another NCCL, so the headers are read out of the
#: archive. Its headers are the same on every architecture.
NCCL_WHEELS = {
    "aarch64": Archive(
        url="https://files.pythonhosted.org/packages/65/32/ff4e28cbed87f99fed63df446ef1986e0617842258a3535eaa2ee92d6226/nvidia_nccl_cu13-2.30.4-py3-none-manylinux_2_18_aarch64.whl",
        sha256="e99308a3a89fba78918d50886e81072a6c8b0b4199feb02c3903e63713a6525a",
    ),
    "x86_64": Archive(
        url="https://files.pythonhosted.org/packages/3e/93/6d020a69fc37e57fae8a96ab0c53102d96538db256e933e914d100e5a430/nvidia_nccl_cu13-2.30.4-py3-none-manylinux_2_18_x86_64.whl",
        sha256="534dbf3058cadb625f08ab0d17f1dffad3b961a2bfa360d66633fcf21be53f57",
    ),
}


def _pins() -> dict[str, Pin]:
    pins = [
        # trtllm-gen's FMHA library: the attention kernels behind
        # `trtllm_batch_decode_with_kv_cache`, as FlashInfer pins them
        # (`ArtifactPath.TRTLLM_GEN_FMHA`, `CheckSumHash.TRTLLM_GEN_FMHA`).
        Pin(
            name="trtllm-gen-fmha",
            sha256="d79b5c51fc8597fac57dae0da4afa114fb2014575e4ec3df099ad856d97cabc3",
            origin=Files(
                TRTLLM_REPOSITORY.rstrip("/")
                + "/2d6a5a029eefcc388ec0ceb87efb55d8bcce5c3c/fmha/trtllm-gen/"
            ),
            manifest="checksums.txt",
        ),
        # trtllm-gen's batched-GEMM library: the grouped and routed MoE product
        # behind `trtllm_fp4_block_scale_routed_moe`, with its export headers.
        Pin(
            name="trtllm-gen-bmm",
            sha256="011635d3c36756addcdc148eea90c984f2d7611ba375626aaaf466d355c80e50",
            origin=Files(
                TRTLLM_REPOSITORY.rstrip("/")
                + "/8ec29a98612c3670f9f28825d1ed19f09496073b"
                + "/batched_gemm-fa419f4-31ee4e5/"
            ),
            manifest="checksums.txt",
        ),
        # DeepEP's elastic dispatch and combine kernels, at deepseek-ai/DeepEP
        # commit b306af0.
        Pin(
            name="deepep",
            sha256="ee22833b7bf83ebf3da46056dcbe0b936cdb4e5442d22b583f1a0be0eda36a31",
            origin=Files(
                "https://raw.githubusercontent.com/deepseek-ai/DeepEP/b306af06afd412c88e51e71802951606e40b7358/"
            ),
        ),
        Pin(
            name="nccl",
            sha256="67a3f54b593e8361db3c3bc8d3b00cfc0cf36930f60d52a34b7e1b2012551301",
            # A machine with no pinned wheel has no origin, and a fetch refuses
            # it.
            origin=NCCL_WHEELS.get(platform.machine()),
        ),
        # FlashInfer's fused MegaMoE CuTe DSL kernel, from the
        # `flashinfer-python` this environment locks (`pyproject.toml`).
        Pin(
            name="megamoe",
            sha256="dae28b342e055af61128a79efbce8727bbed54228613b8a547ad37134ecf2d97",
            origin=Distribution("flashinfer-python", "0.6.18"),
        ),
    ]
    return {pin.name: pin for pin in pins}


#: Every pin this module knows, by name.
PIN_RECORDS: dict[str, Pin] = _pins()


def pin(name: str) -> Pin:
    """The pin of a name, or a refusal naming the known ones."""
    try:
        return PIN_RECORDS[name]
    except KeyError:
        raise FetchError(
            f"no pin {name!r}; the pins are {sorted(PIN_RECORDS)}"
        ) from None


def cache_root() -> Path:
    """The directory pinned sources are cached under.

    It is made absolute, because a staged file is a symbolic link into it: a
    relative cache path would make every link dangle.
    """
    configured = os.environ.get("UNISERVE_ARTIFACT_CACHE")
    root = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache" / "uniserve" / "build-server" / "artifacts"
    )
    return Path(os.path.abspath(root))


def directory(source: Pin) -> Path:
    """The cache directory of a pin: content-addressed by its manifest."""
    return cache_root() / source.sha256


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_manifest(path: Path) -> dict[str, str]:
    """A manifest's files: each path with its sha256."""
    files: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, name = line.partition(" ")
        name = name.strip()
        if not name or len(digest) != 64:
            raise FetchError(f"{path}: cannot read the line {line!r}")
        files[name] = digest
    if not files:
        raise FetchError(f"{path}: the manifest lists no file")
    return files


def _download(url: str, destination: Path) -> None:
    """Fetches ``url`` into ``destination``.

    The file is written beside its destination and renamed only when complete,
    so a failed fetch never leaves a truncated file in the cache.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "uniserve"})
    for attempt in range(1, ATTEMPTS + 1):
        staged = None
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 200:
                    raise FetchError(f"fetching {url}: HTTP {response.status}")
                with tempfile.NamedTemporaryFile(
                    dir=destination.parent, delete=False
                ) as temporary:
                    staged = Path(temporary.name)
                    shutil.copyfileobj(response, temporary)
            staged.replace(destination)
            return
        except (urllib.error.URLError, OSError) as error:
            if staged is not None:
                staged.unlink(missing_ok=True)
            # The repository's own refusal of the file is its response, and
            # asking again repeats it.
            refused = (
                isinstance(error, urllib.error.HTTPError) and error.code < 500
            )
            if refused or attempt == ATTEMPTS:
                raise FetchError(f"fetching {url}: {error}") from error
            time.sleep(attempt)


def manifest(source: Pin, offline: bool = False) -> dict[str, str]:
    """The pin's files with their sha256, from a manifest verified by the pin.

    A packaged manifest is read from ``pins/``; a published one is cached as
    ``<identity>/<manifest>`` and fetched when the cache does not hold a
    verifying copy.
    """
    if source.manifest is None:
        path = PINS / f"{source.name}.sha256"
        if sha256_of_file(path) != source.sha256:
            raise FetchError(
                f"{path} is not the manifest the {source.name} pin names "
                f"({source.sha256})"
            )
        return parse_manifest(path)

    path = directory(source) / source.manifest
    if path.is_file() and sha256_of_file(path) == source.sha256:
        return parse_manifest(path)
    if offline:
        raise FetchError(
            f"{source.name} is not cached at {directory(source)} and --offline "
            "was given"
        )
    # A wrong or truncated manifest is replaced, not trusted.
    path.unlink(missing_ok=True)
    assert isinstance(source.origin, Files)
    _download(source.origin.base + source.manifest, path)
    digest = sha256_of_file(path)
    if digest != source.sha256:
        path.unlink()
        raise FetchError(
            f"{source.name}'s {source.manifest} has sha256 {digest}, the pin "
            f"expects {source.sha256}"
        )
    return parse_manifest(path)


def cached(source: Pin, files: dict[str, str]) -> set[str]:
    """The files of ``files`` the cache already holds and verifies."""
    root = directory(source)
    return {
        name
        for name, digest in files.items()
        if (root / name).is_file() and sha256_of_file(root / name) == digest
    }


def _verified(path: Path, digest: str, where: str) -> None:
    actual = sha256_of_file(path)
    if actual != digest:
        path.unlink()
        raise FetchError(
            f"{where} has sha256 {actual}, the manifest says {digest}"
        )


def _from_files(source: Pin, origin: Files, wanted: dict[str, str]) -> None:
    def one(name: str, digest: str) -> None:
        path = directory(source) / name
        _download(origin.base + name, path)
        _verified(path, digest, origin.base + name)

    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(one, name, digest): name
            for name, digest in wanted.items()
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except FetchError as error:
                failures.append(str(error))
    if failures:
        raise FetchError(
            f"{len(failures)} of {len(wanted)} files did not verify:\n"
            + "\n".join(failures[:10])
        )


def _from_archive(source: Pin, origin: Archive, wanted: dict[str, str]) -> None:
    cache_root().mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_root()) as scratch:
        archive = Path(scratch) / "archive.zip"
        _download(origin.url, archive)
        _verified(archive, origin.sha256, origin.url)
        with zipfile.ZipFile(archive) as members:
            for name, digest in wanted.items():
                path = directory(source) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    data = members.read(name)
                except KeyError:
                    raise FetchError(
                        f"{origin.url} has no member {name}"
                    ) from None
                path.write_bytes(data)
                _verified(path, digest, f"{origin.url}!{name}")


def _from_distribution(
    source: Pin, origin: Distribution, wanted: dict[str, str]
) -> None:
    try:
        installed = importlib.metadata.distribution(origin.name)
    except importlib.metadata.PackageNotFoundError:
        raise FetchError(
            f"{sys.executable} has no {origin.name} distribution; the pin "
            f"needs {origin.version}"
        ) from None
    if installed.version != origin.version:
        raise FetchError(
            f"{sys.executable} has {origin.name} {installed.version}; the pin "
            f"needs {origin.version}"
        )
    root = Path(installed.locate_file(""))
    for name, digest in wanted.items():
        path = directory(source) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not (root / name).is_file():
            raise FetchError(f"{origin.name} {origin.version} has no {name}")
        shutil.copyfile(root / name, path)
        _verified(path, digest, f"{origin.name} {origin.version}: {name}")


def fetch(
    source: Pin, names: Iterable[str] | None = None, offline: bool = False
) -> Path:
    """Makes the cache hold verified copies of a pin's files.

    ``names`` are manifest paths, every file of the pin when ``None``. Returns
    the pin's cache directory.
    """
    files = manifest(source, offline)
    selected = list(files) if names is None else list(names)
    unknown = [name for name in selected if name not in files]
    if unknown:
        raise FetchError(
            f"the {source.name} pin has no {', '.join(unknown[:5])}"
        )
    missing = {
        name: files[name]
        for name in selected
        if name not in cached(source, {name: files[name]})
    }
    if missing:
        # A local installation is no network fetch, so `--offline` does not
        # refuse it.
        if offline and not isinstance(source.origin, Distribution):
            raise FetchError(
                f"{len(missing)} {source.name} files are not cached at "
                f"{directory(source)} and --offline was given"
            )
        if source.origin is None:
            raise FetchError(
                f"the {source.name} pin has no origin for {platform.machine()}"
            )
        if isinstance(source.origin, Files):
            _from_files(source, source.origin, missing)
        elif isinstance(source.origin, Archive):
            _from_archive(source, source.origin, missing)
        else:
            _from_distribution(source, source.origin, missing)
    return directory(source)


def stage(
    source: Pin, out: Path, names: Iterable[str], copy: bool = False
) -> list[str]:
    """Stages cached files of a pin into ``out`` under their manifest paths.

    A staged file is a symbolic link into the cache, which is persistent and
    content-addressed, so a build points at it rather than copying gigabytes of
    cubins; ``copy`` writes files of its own instead, for a build that changes
    them (a patch) and must not change the cache.
    """
    staged: list[str] = []
    for name in names:
        cached_file = directory(source) / name
        if not cached_file.is_file():
            raise FetchError(
                f"{name} is not in the {source.name} cache; fetch it first"
            )
        destination = out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() or destination.exists():
            destination.unlink()
        if copy:
            shutil.copyfile(cached_file, destination)
        else:
            destination.symlink_to(cached_file)
        staged.append(name)
    return staged


def apply_patch(root: Path, patch: Path) -> None:
    """Applies a unified diff to files under ``root``, exactly.

    The diff names files as ``a/<path>`` and ``b/<path>`` relative to ``root``.
    Every hunk's context and removed lines must match the file at the hunk's own
    line: a pinned source is known byte for byte, so a hunk that does not apply
    where it says is a wrong pin or a wrong patch, and the build refuses rather
    than searching for a place it might fit.
    """
    lines = patch.read_text().splitlines(keepends=True)
    index = 0
    while index < len(lines):
        if not lines[index].startswith("--- "):
            index += 1
            continue
        target = lines[index + 1]
        if not target.startswith("+++ b/"):
            raise FetchError(
                f"{patch}: a file header at line {index + 1} names no b/ path"
            )
        path = root / target[len("+++ b/") :].strip()
        original = path.read_text().splitlines(keepends=True)
        result: list[str] = []
        consumed = 0
        index += 2
        while index < len(lines) and lines[index].startswith("@@ "):
            # `@@ -start,old +start,new @@`: the hunk spans `old` lines of the
            # file and writes `new`.
            old_range, new_range = lines[index].split()[1:3]
            start, _, old_count = old_range.lstrip("-").partition(",")
            new_count = new_range.lstrip("+").partition(",")[2]
            remaining_old = int(old_count or 1)
            remaining_new = int(new_count or 1)
            start = int(start)
            # Lines before the hunk are carried over unchanged.
            result.extend(original[consumed : start - 1])
            consumed = start - 1
            index += 1
            while index < len(lines) and (
                remaining_old or remaining_new or lines[index].startswith("\\")
            ):
                line = lines[index]
                if line[0] in (" ", "-"):
                    remaining_old -= 1
                    if (
                        consumed >= len(original)
                        or original[consumed] != line[1:]
                    ):
                        raise FetchError(
                            f"{patch}: the hunk at line {index + 1} does not "
                            f"match {path} line {consumed + 1}"
                        )
                    if line[0] == " ":
                        result.append(original[consumed])
                    consumed += 1
                if line[0] in (" ", "+"):
                    remaining_new -= 1
                if line[0] == "+":
                    result.append(line[1:])
                if line[0] not in (" ", "-", "+", "\\"):
                    raise FetchError(
                        f"{patch}: line {index + 1} is no hunk line"
                    )
                index += 1
        result.extend(original[consumed:])
        path.write_text("".join(result))


def describe(
    source: Pin, files: dict[str, str], staged: list[str]
) -> dict[str, object]:
    """What a caller needs to know about a pin."""
    return {
        "pin": source.name,
        "sha256": source.sha256,
        "cache": str(directory(source)),
        "files": len(files),
        "cached": len(cached(source, files)),
        "staged": sorted(staged),
    }


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m uniserve_build_server.fetch",
        description="fetch, verify, cache and stage a pinned source",
    )
    parser.add_argument(
        "pin", choices=sorted(PIN_RECORDS), help="the pinned source"
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        metavar="NAME",
        help="fetch one file, by its manifest path",
    )
    parser.add_argument(
        "--all", action="store_true", help="fetch every file of the pin"
    )
    parser.add_argument(
        "--out", type=Path, help="stage the fetched files into this directory"
    )
    parser.add_argument(
        "--offline", action="store_true", help="use only what the cache holds"
    )
    parser.add_argument(
        "--cache",
        type=Path,
        help="the cache root (default $UNISERVE_ARTIFACT_CACHE or "
        "~/.cache/uniserve/build-server/artifacts)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    options = arguments(argv)
    if options.cache:
        os.environ["UNISERVE_ARTIFACT_CACHE"] = str(options.cache)
    source = pin(options.pin)
    try:
        files = manifest(source, options.offline)
        names = list(files) if options.all else options.file
        fetch(source, names, options.offline)
        staged = stage(source, options.out, names) if options.out else []
        print(json.dumps(describe(source, files, staged), indent=2))
        return 0
    except FetchError as error:
        print(f"{source.name}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
