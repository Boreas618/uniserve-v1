"""Immutable file-reading choices.

Separate from model and weight representation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class Config:
    """Checkpoint IO policy: encoding, read mode, concurrency and caching.

    Also integrity. mode selects eager full reads, per-shard layered reads,
    or deterministic dummy values. prefetch, which defaults to mmap when
    unset, reads every source file into the page cache before a reader's
    first tensor read, on num_threads threads (8 when unset);
    drop_cache_after_load evicts the source files' cached pages when the
    reader closes.
    """

    format: Literal["auto", "safetensors", "pt"] = "auto"
    mode: Literal["eager", "layered", "dummy"] = "eager"
    revision: str | None = None
    download_dir: Path | None = None
    ignore_patterns: tuple[str, ...] = ("original/**/*",)
    num_threads: int | None = None
    mmap: bool = True
    prefetch: bool | None = None
    drop_cache_after_load: bool = False
    checksum_manifest: Path | None = None

    def __post_init__(self):
        if self.format not in {"auto", "safetensors", "pt"}:
            raise ValueError(
                "checkpoint format must be auto, safetensors or pt"
            )
        if self.mode not in {"eager", "layered", "dummy"}:
            raise ValueError("checkpoint mode must be eager, layered or dummy")
        if self.num_threads is not None and (
            type(self.num_threads) is not int or self.num_threads < 1
        ):
            raise ValueError(
                "reader concurrency must be a positive integer or None"
            )
        if not isinstance(self.ignore_patterns, tuple) or any(
            not isinstance(pattern, str) or not pattern
            for pattern in self.ignore_patterns
        ):
            raise ValueError(
                "ignore patterns must be a tuple of nonempty patterns"
            )
        for name in ("download_dir", "checksum_manifest"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
