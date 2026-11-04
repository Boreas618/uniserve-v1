"""The trtllm-gen FMHA artifact: fetch, verify, cache and stage it.

TensorRT-LLM's FMHA kernels ship as a directory of cubins plus one generated
header, outside any Python distribution, addressed by a pin in FlashInfer
(``flashinfer.artifacts.ArtifactPath.TRTLLM_GEN_FMHA`` and its
``CheckSumHash.TRTLLM_GEN_FMHA``). A kernel program that launches those kernels
(a native library of the launch code and its cubins, design section 9) needs
two things from it: the header that carries the kernel table, to compile
against, and the cubins, to load at run time.

The artifact has no archive: the pin names a directory, and its identity is
the sha256 of the ``checksums.txt`` manifest inside it. This module addresses
the artifact by that hash, keeps it under
``~/.cache/uniserve/program-server/artifacts/<sha256>/`` (or
``$UNISERVE_ARTIFACT_CACHE``), verifies the manifest against the pin and every
file against the manifest, and works offline on what is already there.

    python -m uniserve_program_server.trtllm_gen --out DIR

stages the header, the checksums and the artifact's description into ``DIR``;
``--all`` also stages every cubin, and ``--cubin NAME`` stages one.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

#: The artifact's name, as the design and the package manifests call it.
ARTIFACT = "trtllm-gen-fmha"

#: Where NVIDIA publishes the kernel library's generic artifacts.
REPOSITORY = os.environ.get(
    "UNISERVE_ARTIFACT_REPOSITORY",
    "https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/",
)

#: The artifact's pin inside the repository: the directory that holds
#: ``checksums.txt``, ``include/`` and the cubins.
PIN = "2d6a5a029eefcc388ec0ceb87efb55d8bcce5c3c/fmha/trtllm-gen/"

#: The sha256 of the pin's ``checksums.txt``, which identifies the artifact.
SHA256 = "d79b5c51fc8597fac57dae0da4afa114fb2014575e4ec3df099ad856d97cabc3"

#: The header the launch code includes.
METADATA_HEADER = "include/flashInferMetaInfo.h"

#: The manifest that lists every file of the artifact with its sha256.
CHECKSUMS = "checksums.txt"

#: How many files are fetched at once.
WORKERS = 16


class ArtifactError(RuntimeError):
    """The artifact cannot be fetched, verified or staged."""


def cache_root() -> Path:
    """The directory the artifacts are cached under.

    It is made absolute, because a staged header or cubin is a symbolic link
    into it: a relative cache path would make every link dangle.
    """
    configured = os.environ.get("UNISERVE_ARTIFACT_CACHE")
    root = Path(configured).expanduser() if configured else Path.home() / ".cache" / "uniserve" / "program-server" / "artifacts"
    return Path(os.path.abspath(root))


def artifact_dir() -> Path:
    """The cache directory of the pinned artifact: content-addressed by its
    checksum manifest."""
    return cache_root() / SHA256


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def url_of(relative: str) -> str:
    return f"{REPOSITORY.rstrip('/')}/{PIN}{relative}"


def download(relative: str, destination: Path) -> None:
    """Fetches one file of the artifact into ``destination``.

    The file is written beside its destination and renamed only when it is
    complete, so a failed fetch never leaves a truncated file in the cache.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url_of(relative), headers={"User-Agent": "uniserve"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 200:
                raise ArtifactError(f"fetching {relative}: HTTP {response.status}")
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, delete=False
            ) as temporary:
                shutil.copyfileobj(response, temporary)
                staged = Path(temporary.name)
    except (urllib.error.URLError, OSError) as error:
        raise ArtifactError(f"fetching {url_of(relative)}: {error}") from error
    staged.replace(destination)


def checksums(offline: bool = False) -> dict[str, str]:
    """The artifact's manifest: each file's path relative to the pin, and its
    sha256.

    The manifest is verified against the pin's sha256 before it is trusted, so
    a cache that somehow holds another artifact's manifest is not used.
    """
    path = artifact_dir() / CHECKSUMS
    if path.is_file():
        digest = sha256_of_file(path)
        if digest == SHA256:
            return parse_checksums(path)
        if offline:
            raise ArtifactError(
                f"{path} is not the pinned {ARTIFACT} manifest: it has {digest}, the pin is "
                f"{SHA256}"
            )
        # A wrong or truncated manifest is replaced, not trusted.
        path.unlink()
    if offline:
        raise ArtifactError(
            f"{ARTIFACT} is not cached at {artifact_dir()} and --offline was given"
        )
    download(CHECKSUMS, path)
    digest = sha256_of_file(path)
    if digest != SHA256:
        path.unlink()
        raise ArtifactError(
            f"{url_of(CHECKSUMS)} has sha256 {digest}, the pin expects {SHA256}"
        )
    return parse_checksums(path)


def parse_checksums(path: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, name = line.partition(" ")
        name = name.strip()
        if not name or len(digest) != 64:
            raise ArtifactError(f"{path}: cannot read the line {line!r}")
        files[name] = digest
    if not files:
        raise ArtifactError(f"{path}: the manifest lists no file")
    return files


def fetch_one(relative: str, digest: str, offline: bool) -> Path:
    """The cached copy of one artifact file, fetched and verified if needed."""
    path = artifact_dir() / relative
    if path.is_file() and sha256_of_file(path) == digest:
        return path
    if offline:
        raise ArtifactError(
            f"{relative} is not in the cache at {artifact_dir()} and --offline was given"
        )
    download(relative, path)
    actual = sha256_of_file(path)
    if actual != digest:
        path.unlink()
        raise ArtifactError(
            f"{url_of(relative)} has sha256 {actual}, the manifest says {digest}"
        )
    return path


def fetch(missing: Iterable[str], files: dict[str, str], offline: bool) -> None:
    """Fetches every named file the cache does not already hold, in parallel."""
    wanted = [(name, files[name]) for name in missing]
    if not wanted:
        return
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(fetch_one, name, digest, offline): name for name, digest in wanted
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except ArtifactError as error:
                failures.append(str(error))
    if failures:
        raise ArtifactError(
            f"{len(failures)} of {len(wanted)} files did not verify:\n"
            + "\n".join(failures[:10])
        )


def cached_files(files: dict[str, str]) -> set[str]:
    """The artifact files the cache already holds and verifies."""
    present: set[str] = set()
    for name, digest in files.items():
        path = artifact_dir() / name
        if path.is_file() and sha256_of_file(path) == digest:
            present.add(name)
    return present


def stage(out: Path, names: Iterable[str]) -> list[str]:
    """Stages cached artifact files into ``out``, as symbolic links.

    The cache is content-addressed and persistent; a build's output directory
    is temporary, so a build points at the cache rather than copying three
    gigabytes of cubins.
    """
    staged: list[str] = []
    for name in names:
        source = artifact_dir() / name
        if not source.is_file():
            raise ArtifactError(f"{name} is not in the cache; fetch it first")
        destination = out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() or destination.exists():
            destination.unlink()
        destination.symlink_to(source)
        staged.append(name)
    return staged


def describe(files: dict[str, str], staged: list[str]) -> dict[str, object]:
    """What a build needs to know about the artifact, for its compile script."""
    cubins = sorted(name for name in files if name.endswith(".cubin"))
    headers = sorted(name for name in files if not name.endswith(".cubin"))
    return {
        "artifact": ARTIFACT,
        "pin": PIN,
        "sha256": SHA256,
        "cache": str(artifact_dir()),
        "manifest": str(artifact_dir() / CHECKSUMS),
        "include": str(artifact_dir() / "include"),
        "cubins": cubins,
        "headers": headers,
        "staged": sorted(staged),
    }


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m uniserve_program_server.trtllm_gen",
        description="fetch, verify, cache and stage the pinned trtllm-gen FMHA artifact",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="stage the header, checksums and description into this directory",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="stage every cubin of the artifact, not only the header",
    )
    parser.add_argument(
        "--cubin",
        action="append",
        default=[],
        metavar="NAME",
        help="stage one cubin, by its name such as fmhaSm100aKernel_....cubin",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify every cached file against the manifest and fetch what is missing",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use only what the cache holds; never reach the repository",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        help="the cache root (default $UNISERVE_ARTIFACT_CACHE or ~/.cache/uniserve/...)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    options = arguments(argv)
    if options.cache:
        os.environ["UNISERVE_ARTIFACT_CACHE"] = str(options.cache)
    try:
        files = checksums(options.offline)
        # The header is what a native launcher compiles against; it is always
        # fetched. Cubins are fetched when asked for, or all of them when the
        # artifact must be complete.
        wanted = {METADATA_HEADER}
        for name in options.cubin:
            if name not in files:
                raise ArtifactError(f"the artifact has no {name}")
            wanted.add(name)
        if options.all or options.verify:
            wanted.update(files)
        missing = wanted - cached_files({name: files[name] for name in wanted})
        fetch(sorted(missing), files, options.offline)
        staged: list[str] = []
        if options.out:
            options.out.mkdir(parents=True, exist_ok=True)
            staged = stage(options.out, sorted(wanted) + [CHECKSUMS])
        description = describe(files, staged)
        description["cached"] = len(cached_files(files))
        print(json.dumps(description, indent=2))
        return 0
    except ArtifactError as error:
        print(f"{ARTIFACT}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
