"""Counter-based uniform draws of the token sampler.

The sampler draws one uniform per candidate row from Philox4x32-10, keyed by
the request's stream and addressed by the row's semantic coordinate, so a
draw is independent of batching, execution depth, completion order and
replay. The engine simulator draws the same values through ``ipc::philox``;
every function here is bit-identical to it.

Layout:

- key: a SplitMix64 fold of ``(session_seed, engine_id, request_id, epoch,
  0)`` split into two 32-bit words, low word first; the final zero is the
  draw space of target sampling. A seeded request folds its seed alone
  (:func:`request_sampling_key`);
- counter: ``[semantic_token_index low, semantic_token_index high,
  processor_stage, draw_index]``, the stage and draw index truncated to
  their low 32 bits;
- uniform: ``(word0 >> 8) * 2**-24``, a 24-bit dyadic float32 in ``[0, 1)``.

Every function is vectorized over numpy arrays: 32-bit lanes are held in
``uint64`` arrays so that the 32x32-bit products and key additions never
overflow, and are masked back to 32 bits after each step.

:class:`DrawWindow` caches a window of each slot's draws so that a decode
step gathers its draws instead of evaluating ten Philox rounds per row.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from uniserve.nn.rng import splitmix_coordinate

__all__ = [
    "DrawWindow",
    "philox4x32_10",
    "request_sampling_key",
    "sampling_key",
    "sampling_uniform",
]

_MASK32 = np.uint64(0xFFFF_FFFF)
_SHIFT32 = np.uint64(32)
_PHILOX_M0 = np.uint64(0xD251_1F53)
_PHILOX_M1 = np.uint64(0xCD9E_8D57)
_KEY_BUMP_0 = np.uint64(0x9E37_79B9)
_KEY_BUMP_1 = np.uint64(0xBB67_AE85)
_ROUNDS = 10

# The draw space of target sampling, folded as a key's last coordinate.
_TARGET_DRAW_SPACE = 0

# A uniform keeps the top 24 bits of its word: exactly representable in
# float32, and so is their product with this power of two.
_UNIFORM_SHIFT = np.uint64(8)
_UNIFORM_SCALE = np.float32(1.0 / 16_777_216.0)


def philox4x32_10(counter: ArrayLike, key: ArrayLike) -> np.ndarray:
    """Apply the ten-round Philox4x32 bijection.

    Args:
        counter: ``[..., 4]`` unsigned 32-bit counter words.
        key: ``[..., 2]`` unsigned 32-bit key words; broadcast against the
            counter's leading dimensions.

    Returns:
        ``[..., 4]`` uint32 output words.
    """
    lanes = np.asarray(counter).astype(np.uint64)
    keys = np.asarray(key).astype(np.uint64)
    c0, c1, c2, c3 = (lanes[..., index] for index in range(4))
    k0, k1 = keys[..., 0], keys[..., 1]
    for round_index in range(_ROUNDS):
        # Round zero consumes the caller's key; every later round advances
        # both key words by the Weyl constants before permuting the counter.
        if round_index > 0:
            k0 = (k0 + _KEY_BUMP_0) & _MASK32
            k1 = (k1 + _KEY_BUMP_1) & _MASK32
        product0 = _PHILOX_M0 * c0
        product1 = _PHILOX_M1 * c2
        c0, c1, c2, c3 = (
            (product1 >> _SHIFT32) ^ c1 ^ k0,
            product1 & _MASK32,
            (product0 >> _SHIFT32) ^ c3 ^ k1,
            product0 & _MASK32,
        )
    words = np.stack(np.broadcast_arrays(c0, c1, c2, c3), axis=-1)
    return words.astype(np.uint32)


def sampling_key(
    session_seed: int, engine_id: int, request_id: int, request_epoch: int
) -> int:
    """Return the 64-bit Philox key of one request lineage.

    The coordinates fold in protocol order, so the same tuple addresses the
    same stream in every runtime.
    """
    key = int(session_seed)
    for coordinate in (engine_id, request_id, request_epoch):
        key = splitmix_coordinate(key, coordinate)
    return splitmix_coordinate(key, _TARGET_DRAW_SPACE)


def request_sampling_key(
    seed: int | None, engine_id: int, request_id: int, request_epoch: int
) -> int:
    """Return the key of one request's stream.

    A seeded request's stream is a function of its seed alone, so it draws
    the same values whenever it is sent; an unseeded request's stream is its
    lineage's, which keeps the streams of distinct requests apart.
    """
    if seed is not None:
        return sampling_key(seed, 0, 0, 0)
    return sampling_key(0, engine_id, request_id, request_epoch)


def sampling_uniform(
    keys: ArrayLike,
    positions: ArrayLike,
    stage: int = 0,
    draw: int = 0,
) -> np.ndarray:
    """Return uniform draws in ``[0, 1)`` at semantic coordinates.

    Args:
        keys: Request stream keys (unsigned 64-bit), broadcast against
            ``positions``.
        positions: Semantic token indices (unsigned 64-bit).
        stage: The processor stage; its low 32 bits address the draw.
        draw: The draw index; its low 32 bits address the draw.

    Returns:
        float32 draws of the broadcast shape of ``keys`` and ``positions``.
    """
    key = np.asarray(keys, dtype=np.uint64)
    position = np.asarray(positions, dtype=np.uint64)
    key, position = np.broadcast_arrays(key, position)
    stage_word = np.uint64(int(stage) & 0xFFFF_FFFF)
    draw_word = np.uint64(int(draw) & 0xFFFF_FFFF)
    counter = np.stack(
        np.broadcast_arrays(
            position & _MASK32,
            position >> _SHIFT32,
            stage_word,
            draw_word,
        ),
        axis=-1,
    )
    key_words = np.stack((key & _MASK32, key >> _SHIFT32), axis=-1)
    first = philox4x32_10(counter, key_words)[..., 0]
    return (first >> np.uint32(_UNIFORM_SHIFT)).astype(np.float32) * (
        _UNIFORM_SCALE
    )


class DrawWindow:
    """Cached draws of each request slot over a window of positions.

    A slot's window holds the draws of one key at ``width`` consecutive
    positions from a base. A lookup gathers the draws its window holds; a
    slot whose window holds another key, or not the position, is refilled
    from its least requested position, and a position outside even the
    refilled window is drawn directly. The draws are therefore exactly
    :func:`sampling_uniform` of the requested coordinates, whatever the
    window held. Windows are keyed by the stream key, so a slot taken by
    another request needs no invalidation.

    Args:
        slots: Request slots; slot ``0`` is padding and has a window too.
        width: Positions per window.
    """

    def __init__(self, slots: int, width: int = 64) -> None:
        if width <= 0:
            raise ValueError("a draw window holds at least one position")
        rows = slots + 1
        self.width = width
        self._keys = np.zeros(rows, dtype=np.uint64)
        self._bases = np.zeros(rows, dtype=np.int64)
        self._filled = np.zeros(rows, dtype=bool)
        self._draws = np.zeros((rows, width), dtype=np.float32)
        self._offsets = np.arange(width, dtype=np.int64)

    def draws(
        self,
        slots: Sequence[int] | np.ndarray,
        keys: Sequence[int] | np.ndarray,
        positions: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        """Return the draws of ``keys`` at ``positions`` for ``slots``.

        Args:
            slots: ``[n]`` request slot of each draw.
            keys: ``[n]`` stream key of each draw's request.
            positions: ``[n]`` non-negative semantic token indices.

        Returns:
            ``[n]`` float32 draws, equal to ``sampling_uniform(keys,
            positions)``.
        """
        slot = np.asarray(slots, dtype=np.int64)
        key = np.asarray(keys, dtype=np.uint64)
        position = np.asarray(positions, dtype=np.int64)
        if np.any(position < 0):
            raise ValueError("a draw position is negative")

        missing = ~self._holds(slot, key, position)
        if missing.any():
            self._refill(slot[missing], key[missing], position[missing])
        values = self._draws[slot, (position - self._bases[slot]) % self.width]

        # A slot drawn at positions farther apart than one window keeps the
        # window of its least position; the others are drawn directly.
        outside = ~self._holds(slot, key, position)
        if outside.any():
            values[outside] = sampling_uniform(key[outside], position[outside])
        return values

    def _holds(
        self, slot: np.ndarray, key: np.ndarray, position: np.ndarray
    ) -> np.ndarray:
        offset = position - self._bases[slot]
        return (
            self._filled[slot]
            & (self._keys[slot] == key)
            & (offset >= 0)
            & (offset < self.width)
        )

    def _refill(
        self, slot: np.ndarray, key: np.ndarray, position: np.ndarray
    ) -> None:
        # One window per missing slot, starting at its least missing
        # position, of the key its first missing draw names.
        refilled, first = np.unique(slot, return_index=True)
        bases = np.full(refilled.shape, np.iinfo(np.int64).max)
        np.minimum.at(bases, np.searchsorted(refilled, slot), position)
        window_keys = key[first]
        window = bases[:, None] + self._offsets[None, :]
        self._draws[refilled] = sampling_uniform(
            window_keys[:, None], window.astype(np.uint64)
        )
        self._keys[refilled] = window_keys
        self._bases[refilled] = bases
        self._filled[refilled] = True
