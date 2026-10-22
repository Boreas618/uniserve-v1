"""Immutable numerical configuration of the stub model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Config:
    """Numerical configuration of the stub model.

    Attributes:
        patch_size: Pixel side of one square patch, shared by the vision
            encoder, VAE, denoiser and image decoder. The worker
            builds ``Model`` with the default, which ``image_processor``'s
            fixed 16-pixel transforms assume.
        request_state: Whether the backbone layer also counts each
            request's tokens in a cache layer kept per request slot
            (``TokenCount``), the test implementation of such a cache layer
            type.
    """

    patch_size: int = 16
    request_state: bool = False

    @classmethod
    def read(cls, path: str | None) -> Config:
        """The configuration in ``stub.json`` of the directory ``path``
        names (the worker's ``--model``), the defaults without one.
        """  # noqa: D205
        if path is None or not (Path(path) / "stub.json").is_file():
            return cls()
        return cls(**json.loads((Path(path) / "stub.json").read_text()))

    def __post_init__(self):
        if type(self.request_state) is not bool:
            raise ValueError("request_state is a boolean")
        if type(self.patch_size) is not int or self.patch_size < 1:
            raise ValueError(
                "simulation patches must have a positive pixel width"
            )
