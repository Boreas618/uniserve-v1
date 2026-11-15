"""Pinned third-party sources: fetch, verify, cache and stage them.

A kernel package that builds against code it does not own -- TensorRT-LLM's
trtllm-gen kernel libraries, DeepEP's dispatch and combine kernels, NCCL's
device-API headers, FlashInfer's MegaMoE kernel -- names that code by a **pin**
instead of carrying a copy. Every pin is one data file of this package,
``pins/<name>.toml``, and is addressed the same way:

- its identity is the sha256 of a **manifest**, one ``<sha256>  <path>`` line
  per file;
- the manifest is either published beside the files (trtllm-gen's
  ``checksums.txt``, named by the pin's ``manifest``) or kept in this package
  (``pins/<name>.sha256``);
- every file is verified against the manifest before it is used, and kept in
  the Build Server's cache of pins under ``<identity>/<path>``, so a cached file
  is reused offline and a file that does not verify is fetched again.

What differs between pins is only where a file comes from, the pin's
``[origin]``: a repository that serves each file under a base URL (``files``),
one archive whose own sha256 is pinned (``archive``, ``archive_sha256``), or a
distribution this environment's lock installs (``distribution``).

The cache is the Build Server's: a build is handed its root in its request
(:attr:`uniserve_build_server.request.Request.cache`), and
``uniserve-build-server provision PIN --cache DIR`` fills it by hand.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import functools
import hashlib
import importlib.metadata
import importlib.resources
import os
import shutil
import sys
import tempfile
import tomllib
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path

import patch_ng
import urllib3

#: How a file is asked for: a reset connection or a server error is asked for
#: again, with a growing pause, and a refusal of the file (another 4xx) is the
#: repository's answer, which asking again repeats.
RETRIES = urllib3.Retry(
    total=5, backoff_factor=1, status_forcelist=(500, 502, 503, 504)
)

#: The seconds one request may take to connect, and to deliver each read.
TIMEOUT = urllib3.Timeout(connect=30, read=120)


class FetchError(RuntimeError):
    """A pinned source cannot be fetched, verified or staged."""


@dataclasses.dataclass(frozen=True)
class Pin:
    """One pinned source, as its data file states it."""

    #: The name packages and the command line use.
    name: str
    #: The sha256 of the manifest, which identifies the pin and names its cache
    #: directory.
    sha256: str
    #: Where the files come from.
    origin: Files | Archive | Distribution
    #: The manifest's name in the origin, for a manifest published beside the
    #: files; ``None`` keeps it in this package as ``pins/<name>.sha256``.
    manifest: str | None = None

    def directory(self, cache: Path) -> Path:
        """The pin's cache directory, content-addressed by its manifest."""
        return cache / self.sha256


@dataclasses.dataclass(frozen=True)
class Files:
    """A repository serving each file of the manifest at ``<base><path>``."""

    base: str

    def fetch(
        self, pin: Pin, cache: Path, wanted: dict[str, str], jobs: int
    ) -> None:
        def one(name: str, digest: str) -> None:
            url = self.base + name
            _place(
                pin.directory(cache) / name,
                lambda staged: _download(url, staged),
                digest,
                url,
            )

        failures: list[str] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, jobs)
        ) as pool:
            futures = [
                pool.submit(one, name, digest)
                for name, digest in wanted.items()
            ]
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


@dataclasses.dataclass(frozen=True)
class Archive:
    """One zip archive (a wheel, say) whose own sha256 is pinned.

    The manifest's paths are members of it. The archive is read once and not
    kept: only the files the manifest lists enter the cache.
    """

    url: str
    sha256: str

    def fetch(
        self, pin: Pin, cache: Path, wanted: dict[str, str], jobs: int
    ) -> None:
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache, prefix=".") as scratch:
            archive = Path(scratch) / "archive.zip"
            _download(self.url, archive)
            _verified(archive, self.sha256, self.url)
            with zipfile.ZipFile(archive) as members:
                for name, digest in wanted.items():
                    try:
                        data = members.read(name)
                    except KeyError:
                        raise FetchError(
                            f"{self.url} has no member {name}"
                        ) from None
                    _place(
                        pin.directory(cache) / name,
                        lambda staged, data=data: staged.write_bytes(data),
                        digest,
                        f"{self.url}!{name}",
                    )


@dataclasses.dataclass(frozen=True)
class Distribution:
    """A distribution this environment's lock installs.

    The manifest's paths are relative to its installation root
    (``site-packages``); the manifest's digests fix its content, so a
    distribution of another version that changed a pinned file is refused.
    """

    name: str

    def fetch(
        self, pin: Pin, cache: Path, wanted: dict[str, str], jobs: int
    ) -> None:
        try:
            installed = importlib.metadata.distribution(self.name)
        except importlib.metadata.PackageNotFoundError:
            raise FetchError(
                f"{sys.executable} has no {self.name} distribution"
            ) from None
        root = Path(installed.locate_file(""))
        for name, digest in wanted.items():
            if not (root / name).is_file():
                raise FetchError(
                    f"{self.name} {installed.version} has no {name}"
                )
            _place(
                pin.directory(cache) / name,
                lambda staged, name=name: shutil.copyfile(root / name, staged),
                digest,
                f"{self.name} {installed.version}: {name}",
            )


def _origin(stated: dict[str, str]) -> Files | Archive | Distribution:
    if "files" in stated:
        return Files(stated["files"])
    if "archive" in stated:
        return Archive(stated["archive"], stated["archive_sha256"])
    return Distribution(stated["distribution"])


@functools.cache
def pins() -> dict[str, Pin]:
    """Every pin this package declares, by name."""
    declared = importlib.resources.files(__package__).joinpath("pins")
    found = {}
    for entry in declared.iterdir():
        if not entry.name.endswith(".toml"):
            continue
        name = entry.name.removesuffix(".toml")
        stated = tomllib.loads(entry.read_text())
        found[name] = Pin(
            name=name,
            sha256=stated["sha256"],
            origin=_origin(stated["origin"]),
            manifest=stated.get("manifest"),
        )
    return found


def pin(name: str) -> Pin:
    """The pin of a name, or a refusal naming the known ones."""
    try:
        return pins()[name]
    except KeyError:
        raise FetchError(
            f"no pin {name!r}; the pins are {sorted(pins())}"
        ) from None


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_manifest(text: str, where: str) -> dict[str, str]:
    """A manifest's files: each path with its sha256."""
    files: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, name = line.partition(" ")
        name = name.strip()
        if not name or len(digest) != 64:
            raise FetchError(f"{where}: cannot read the line {line!r}")
        files[name] = digest
    if not files:
        raise FetchError(f"{where}: the manifest lists no file")
    return files


def _download(url: str, destination: Path) -> None:
    """Fetches ``url`` into ``destination`` with the :data:`RETRIES` policy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib3.PoolManager(retries=RETRIES, timeout=TIMEOUT) as http:
        try:
            response = http.request("GET", url, preload_content=False)
        except urllib3.exceptions.HTTPError as error:
            raise FetchError(f"fetching {url}: {error}") from error
        try:
            if response.status != 200:
                raise FetchError(f"fetching {url}: HTTP {response.status}")
            with destination.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        finally:
            response.release_conn()


def _verified(path: Path, digest: str, where: str) -> None:
    actual = sha256_of_file(path)
    if actual != digest:
        path.unlink()
        raise FetchError(
            f"{where} has sha256 {actual}, the manifest says {digest}"
        )


def _place(
    path: Path, write: Callable[[Path], None], digest: str, where: str
) -> None:
    """Writes one cached file through ``write``, verified before it is visible.

    ``write`` fills a file of its own beside ``path``; that file is verified
    against ``digest`` and renamed onto ``path`` only when it matches. Builds
    that fetch the same pin at once (a Build Server builds its modules
    concurrently, and a cold cache is fetched by each of them) therefore never
    read, verify or remove another's partly written file: each renames a
    complete, verified copy, and a later rename replaces equal contents.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        staged = Path(temporary.name)
    try:
        write(staged)
        _verified(staged, digest, where)
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def manifest(source: Pin, cache: Path, offline: bool = False) -> dict[str, str]:
    """The pin's files with their sha256, from a manifest verified by the pin.

    A packaged manifest is read from ``pins/``; a published one is cached as
    ``<identity>/<manifest>`` and fetched when the cache holds no verifying
    copy.
    """
    if source.manifest is None:
        packaged = importlib.resources.files(__package__).joinpath(
            "pins", f"{source.name}.sha256"
        )
        data = packaged.read_bytes()
        if hashlib.sha256(data).hexdigest() != source.sha256:
            raise FetchError(
                f"{packaged} is not the manifest the {source.name} pin names "
                f"({source.sha256})"
            )
        return parse_manifest(data.decode(), str(packaged))

    path = source.directory(cache) / source.manifest
    if not (path.is_file() and sha256_of_file(path) == source.sha256):
        if offline:
            raise FetchError(
                f"{source.name} is not cached at {source.directory(cache)}"
            )
        if not isinstance(source.origin, Files):
            raise FetchError(
                f"{source.name} publishes its manifest, and only a repository "
                "of files publishes one"
            )
        url = source.origin.base + source.manifest
        _place(path, lambda staged: _download(url, staged), source.sha256, url)
    return parse_manifest(path.read_text(), str(path))


def cached(source: Pin, cache: Path, files: dict[str, str]) -> set[str]:
    """The files of ``files`` the cache already holds and verifies."""
    root = source.directory(cache)
    return {
        name
        for name, digest in files.items()
        if (root / name).is_file() and sha256_of_file(root / name) == digest
    }


def fetch(
    source: Pin,
    cache: Path,
    names: Iterable[str] | None = None,
    jobs: int = 1,
    offline: bool = False,
) -> Path:
    """Makes the cache hold verified copies of a pin's files.

    ``names`` are manifest paths, every file of the pin when ``None``; ``jobs``
    files are fetched at once. Returns the pin's cache directory.
    """
    files = manifest(source, cache, offline)
    selected = list(files) if names is None else list(names)
    unknown = [name for name in selected if name not in files]
    if unknown:
        raise FetchError(
            f"the {source.name} pin has no {', '.join(unknown[:5])}"
        )
    wanted = {name: files[name] for name in selected}
    missing = {
        name: digest
        for name, digest in wanted.items()
        if name not in cached(source, cache, {name: digest})
    }
    if missing:
        # A local installation is no network fetch: offline does not refuse it.
        if offline and not isinstance(source.origin, Distribution):
            raise FetchError(
                f"{len(missing)} {source.name} files are not cached at "
                f"{source.directory(cache)}"
            )
        source.origin.fetch(source, cache, missing, jobs)
    return source.directory(cache)


def stage(
    source: Pin,
    cache: Path,
    out: Path,
    names: Iterable[str],
    copy: bool = False,
) -> list[str]:
    """Stages cached files of a pin into ``out`` under their manifest paths.

    A staged file is a hard link to the cached one: the cache and every build's
    workspace are directories of the Build Server's one cache root, so a build
    holds its inputs inside its own workspace without copying gigabytes of
    cubins. ``copy`` writes files of the build's own instead, for a build that
    changes them (a patch) and must not change the cache.
    """
    staged: list[str] = []
    for name in names:
        cached_file = source.directory(cache) / name
        if not cached_file.is_file():
            raise FetchError(
                f"{name} is not in the {source.name} cache; fetch it first"
            )
        destination = out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.unlink(missing_ok=True)
        if copy:
            shutil.copyfile(cached_file, destination)
        else:
            os.link(cached_file, destination)
        staged.append(name)
    return staged


def apply_patch(root: Path, patch: Path) -> None:
    """Applies a unified diff to files under ``root``.

    The diff names files as ``a/<path>`` and ``b/<path>`` relative to ``root``.
    A pinned source is known byte for byte, so a diff that does not apply is a
    wrong pin or a wrong patch, and the build refuses it.
    """
    parsed = patch_ng.fromfile(str(patch))
    if not parsed or not parsed.apply(strip=1, root=str(root)):
        raise FetchError(
            f"{patch} does not apply to the pinned files under {root}"
        )


def describe(
    source: Pin, cache: Path, files: dict[str, str]
) -> dict[str, object]:
    """What a caller needs to know about a pin."""
    return {
        "pin": source.name,
        "sha256": source.sha256,
        "cache": str(source.directory(cache)),
        "files": len(files),
        "cached": len(cached(source, cache, files)),
    }
