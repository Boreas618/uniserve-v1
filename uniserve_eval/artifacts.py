"""Writes durable benchmark metadata and validated media artifacts.

`ArtifactWriter` writes into one benchmark point's result directory: JSON and
JSON Lines records at its top level and generated media under `samples/`.
`pipeline.run.run_point` is its caller and decides which records exist and
when each is rewritten.

Every file this writer produces is written to a hidden temporary file in its
destination directory, flushed and fsynced, then moved into place with
`os.replace`. A reader therefore sees either the previous or the new complete
version of such a file, never a partial write. `run_point` writes
`summary.md` directly, so that file does not have this guarantee.

`supersede` moves the evidence of an earlier run of a point -- its bundle,
its media directory and its deployment's server logs -- out of the canonical
paths before the point runs again, so a rerun neither overwrites nor mixes
with the earlier run's files and the earlier run stays inspectable.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .types import DecodedImage


class ArtifactWriter:
    """Writes one benchmark result bundle with atomic metadata updates.

    Media samples are content-addressed: `DecodedImage.sample_filename` and
    `DecodedVideo.sample_filename` are the SHA-256 of the encoded bytes plus an
    extension, so identical outputs from different requests share one file.
    """

    def __init__(self, output_dir: str | Path) -> None:
        """Create the result and media-sample directories."""
        self.output_dir = Path(output_dir)
        self.samples_dir = self.output_dir / "samples"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Any) -> Path:
        """Serialize a value as atomically replaced, formatted JSON.

        `name` is relative to the output directory. Keys are sorted, so equal
        payloads produce identical files.
        """
        path = self.output_dir / name
        temporary_path: Path | None = None
        try:
            # The temporary file shares the destination directory so that
            # `os.replace` is a same-filesystem rename, and its bytes reach
            # disk before the rename publishes them.
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            # After a successful rename the temporary name no longer exists
            # and this is a no-op; on failure it removes the partial file.
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return path

    def write_original_video(self, path: Path, data: bytes) -> Path:
        """Durably retain bytes, including bodies rejected by inspection.

        Callers supply a request-qualified path. Existing different bytes are
        never overwritten, so a canonical output cannot silently change.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != data:
                raise FileExistsError(f"original media already exists: {path}")
            return path
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, delete=False
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return path

    def write_jsonl(self, name: str, payloads: list[Any]) -> Path:
        """Serialize values as an atomically replaced JSON Lines file."""
        path = self.output_dir / name
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                for payload in payloads:
                    json.dump(_jsonable(payload), handle, sort_keys=True)
                    handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return path

    def write_image_sample(self, image: DecodedImage) -> Path:
        """Store a content-addressed image sample.

        `image` comes from `inspect_image_bytes`, which decoded its bytes and
        derived the metadata the request record holds, so the stored file
        matches the record that references it.

        Returns:
            The sample path under `samples/`. An existing file with identical
            bytes is reused without rewriting.

        Raises:
            ValueError: If an existing sample of the same name holds
                different bytes.
        """
        path = self.samples_dir / image.sample_filename
        if path.exists():
            if path.read_bytes() != image.data:
                raise ValueError("content-addressed image sample collision")
            return path

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(image.data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return path


#: The directory under an output root that holds superseded runs, as
#: `superseded/<point>/<UTC time of supersession>/<role>`.
SUPERSEDED_DIR = "superseded"


def supersede(
    output_root: Path, point: str, evidence: Mapping[str, Path]
) -> Path | None:
    """Move an earlier run's evidence of `point` out of its canonical paths.

    Args:
        output_root: The run's output root, which holds the archive.
        point: The benchmark point about to run again.
        evidence: The canonical paths an earlier run of the point may have
            left, keyed by the name each keeps in the archive, such as
            `bundle`, `media` or `server-logs/<point>.log`. A missing path
            and an empty directory hold no evidence and are skipped.

    Returns:
        The archive directory `<output_root>/superseded/<point>/<UTC time>`
        that now holds every path that held evidence, each under its key, or
        `None` when no path held any.

    Raises:
        FileExistsError: If an archive of the same point and time exists.
    """
    held = {
        name: path
        for name, path in evidence.items()
        if path.is_file() or (path.is_dir() and any(path.iterdir()))
    }
    if not held:
        return None

    # Microseconds keep two supersessions of one point apart; `mkdir`
    # without `exist_ok` refuses to merge into an existing archive.
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    stamp += f".{time.time_ns() // 1000 % 1_000_000:06d}Z"
    archive = output_root / SUPERSEDED_DIR / point / stamp
    archive.mkdir(parents=True)
    for name, path in held.items():
        destination = archive / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        # `shutil.move` renames within a filesystem and copies across one,
        # so a media directory outside the output root moves as well.
        shutil.move(path, destination)
    return archive


def _jsonable(payload: Any) -> Any:
    """Convert supported structured values into JSON-compatible values.

    Dataclass instances become `dataclasses.asdict` output, which is not
    converted further. `Path` values become strings, and dicts and lists are
    converted element by element. Any other value is returned unchanged for
    `json.dump` to encode or reject.
    """
    if is_dataclass(payload) and not isinstance(payload, type):
        return asdict(payload)
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {key: _jsonable(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_jsonable(value) for value in payload]
    return payload
