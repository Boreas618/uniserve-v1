"""Borrowed image denoising inputs of the stub model."""

from __future__ import annotations

from dataclasses import dataclass

from uniserve.media import image
from uniserve.models import DenoiserInput as NumericalDenoiserInput
from uniserve.nn.attention import AttentionBatch


@dataclass(frozen=True)
class DenoiserInput(NumericalDenoiserInput[image.Config]):
    """Latents, image sizes and attention inputs for cache writes."""

    # ``Denoiser.forward`` rejects a dense batch or one without host query
    # lengths, and writes zero K/V at the cache layer's table entry's
    # ``slot_mapping`` when it carries them.
    attention: AttentionBatch
