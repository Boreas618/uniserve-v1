"""Recognize MiniMax-H3 checkpoints and the DiT partitions they hold.

The base release (``model_index.json`` naming ``MiniMaxH3ModularPipeline``)
holds every component and two DiT partitions: ``transformer`` serves t2va
and fl2va, and ``transformer_ref`` serves ref2va. FastH3 is its fast variant:
a FastVideo student of one partition, whose ``fastvideo_inference.json``
states the student's task, schedule and sparse attention. The task selects
the partition the student replaces, and the student serves every task of
that partition: a ``t2va`` student also takes ``fl2va`` keyframes, as the
partition's network does. An export whose contract pins a base
revision in ``base_model_revision`` (``hf://<repository>@<revision>``) reads
every directory it omits from that revision (``base_checkpoint``).

The loader's catalog selects this package by the pipeline class the root
index names (``MiniMaxH3ModularPipeline``), so a checkpoint is told apart by
its inference contract and the partitions it holds alone. Each partition is
a denoising component named as the checkpoint names it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

# The DiT partition of the base release that serves each task.
TASK_PARTITIONS: Mapping[str, str] = MappingProxyType(
    {"t2va": "transformer", "fl2va": "transformer", "ref2va": "transformer_ref"}
)

# FastVideo states a student's task as ``task`` (``t2av`` for
# text-to-video-and-audio) or as ``model_type``.
_FASTVIDEO_TASKS: Mapping[str, str] = MappingProxyType(
    {"t2av": "t2va", "ref2va": "ref2va"}
)


@dataclass(frozen=True, slots=True)
class Layout:
    """One recognized checkpoint.

    Attributes:
        partitions: Each DiT partition the root holds, mapped to the tasks it
            serves in canonical order.
        contract: A FastH3 export's ``fastvideo_inference.json``, or None for
            the base release.
    """

    partitions: Mapping[str, tuple[str, ...]]
    contract: Mapping[str, Any] | None


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return value


def _contract(root: Path) -> dict[str, Any] | None:
    """Read an export's ``fastvideo_inference.json``, or None without one."""
    path = root / "fastvideo_inference.json"
    return _json(path) if path.is_file() else None


def _partition_tasks(partition: str) -> tuple[str, ...]:
    """The tasks ``partition`` serves, in canonical order."""
    return tuple(
        task for task, owner in TASK_PARTITIONS.items() if owner == partition
    )


def detect(root: Path) -> Layout:
    """Classify ``root`` and list the DiT partitions it holds.

    Raises:
        FileNotFoundError: The root holds neither a FastH3 contract nor a
            pipeline index.
        ValueError: The contract names a task no partition serves, or the
            index lists no DiT partition the root holds.
    """
    contract = _contract(root)
    if contract is not None:
        stated = contract.get("task", contract.get("model_type"))
        task = _FASTVIDEO_TASKS.get(stated) if isinstance(stated, str) else None
        if task is None:
            raise ValueError(
                "unsupported FastH3 checkpoint: its task must be one of "
                f"{sorted(_FASTVIDEO_TASKS)}, got {stated!r}"
            )
        partition = TASK_PARTITIONS[task]
        return Layout(
            MappingProxyType({partition: _partition_tasks(partition)}),
            MappingProxyType(contract),
        )

    index_path = root / "model_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"{root} holds neither fastvideo_inference.json nor "
            "model_index.json"
        )
    index = _json(index_path)
    partitions = {
        partition: _partition_tasks(partition)
        for partition in dict.fromkeys(TASK_PARTITIONS.values())
        if partition in index and (root / partition / "config.json").is_file()
    }
    if not partitions:
        raise ValueError(
            "unsupported MiniMax-H3 checkpoint: the release holds neither "
            "transformer nor transformer_ref"
        )
    return Layout(MappingProxyType(partitions), None)


def _pinned_revision(value: object) -> tuple[str, str]:
    """Parse ``hf://<repository>@<revision>`` into its two parts.

    Raises:
        ValueError: ``value`` is not of that form with an ``<owner>/<name>``
            repository and a non-empty revision.
    """
    prefix = "hf://"
    repository, separator, revision = (
        value[len(prefix) :].rpartition("@")
        if isinstance(value, str) and value.startswith(prefix)
        else ("", "", "")
    )
    if (
        not separator
        or repository.count("/") != 1
        or not all(repository.split("/"))
        or not revision
    ):
        raise ValueError(
            "FastH3 base_model_revision must be "
            f"hf://<repository>@<revision>, got {value!r}"
        )
    return repository, revision


def base_checkpoint(root: Path) -> tuple[str, str] | None:
    """Return the ``(repository, revision)`` an export pins, if any.

    The loader reads every top-level directory the export omits from that
    revision. The base release and an export that pins nothing return
    ``None``. Only the pin is read: ``read_config`` classifies the
    checkpoint, once per load.

    Raises:
        ValueError: The contract is not a JSON object, or the pin is not of
            the form ``hf://<repository>@<revision>``.
    """
    contract = _contract(root)
    if contract is None or contract.get("base_model_revision") is None:
        return None
    return _pinned_revision(contract["base_model_revision"])
