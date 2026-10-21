"""Schedule-stable seeds of a request's noise draws.

The rank's token sampler draws its uniforms in Rust
(``uniserve_ipc::philox``); the Python library draws only normal noise,
from ``torch`` generators seeded here. A seed is a SplitMix64 fold of the
request's seed and a semantic coordinate, so a draw depends on which image
or encoding it belongs to, never on batch order, execution depth or replay.
"""

from __future__ import annotations

_U64 = 0xFFFFFFFFFFFFFFFF
_SPLITMIX64_GAMMA = 0x9E3779B97F4A7C15


def flow_noise_seed(request_seed: int, semantic_image_index: int) -> int:
    """Return the schedule-stable seed for one semantic image's initial
    noise.
    """  # noqa: D205
    return _splitmix_coordinate(request_seed, semantic_image_index)


def posterior_noise_seed(request_seed: int, encode_index: int) -> int:
    """Return the seed of one latent encoding's posterior noise.

    ``encode_index`` counts the request's latent encodings before this one.
    A second mixing step keeps these seeds apart from the flow-noise seeds
    of the same request and index.
    """
    return _splitmix_coordinate(flow_noise_seed(request_seed, encode_index), 0)


def _splitmix_coordinate(seed: int, coordinate: int) -> int:
    """Mix a seed and logical coordinate into one deterministic unsigned
    64-bit value.
    """  # noqa: D205
    if int(coordinate) < 0:
        raise ValueError("random coordinate must not be negative")
    value = (int(seed) + (int(coordinate) + 1) * _SPLITMIX64_GAMMA) & _U64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _U64
    return (value ^ (value >> 31)) & _U64


__all__ = ["flow_noise_seed", "posterior_noise_seed"]
