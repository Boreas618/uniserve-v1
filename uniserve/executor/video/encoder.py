"""A video model's encoders.

:class:`TextEncoders` serves a video model's text encoder component: the
prompt encoder and, for conditioned tasks, the vision tower bound with it,
which computes with the prompt encoder's execution context on its stream.
:class:`ConditionEncoders` serves the condition latent encoders: the video
encoder and the audio encoder, whose runners each serve one input shape.

**Prompts.** A prompt is zero-padded to the smallest configured text
capacity that holds it; causal attention keeps the padding out of the
prompt's rows, and one captured graph per capacity serves every prompt
length up to it. A prompt holding vision tokens takes rotary coordinates,
which the host computes, and its vision rows; under CUDA graphs they are
copied into the encoder's buffers and the capacity's graph of prompts with
vision replays.

**Vision blocks.** The vision tower encodes a request's vision blocks, each
one sample of its own call. Under CUDA graphs startup captures one block of
each keyframe grid the admitted canvases plan; a block of a captured grid
replays its graph, any other block (a reference's) is encoded eagerly.

**Conditions.** A video encoder whose latent encoder tiles encodes every
still frame as tiles of one shape, each many small kernels, so its eager
encoding is bound by the host launching them. With a graph pool, startup
captures one tile's encoding, and every video runner constructed afterwards
replays it for each still-frame tile of that shape inside its eager forward.
Under CUDA graphs startup also captures the encodings of the keyframes an
fl2va request brings at each admitted canvas; condition media of any other
shape (a reference's) are encoded eagerly by a runner of their shapes. A
latent encoder that samples its posterior reads one draw per condition size,
filled once and held for the sizes most recently used.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

import torch

from uniserve.cuda_graph import BreakableCUDAGraph, capture
from uniserve.executor.video.denoiser import EAGER, REPLAY, close_all
from uniserve.executor.video.plan import (
    CANVAS_MULTIPLE,
    Canvas,
    rows_per_frame,
)
from uniserve.runner.encoder import EncoderRunner

__all__ = [
    "ConditionEncoders",
    "ConditionUnit",
    "PromptVision",
    "TextEncoders",
    "text_capacity",
]

# The posterior draws held at most: one per condition size, the oldest
# released first. Sixteen draws of five-second 16:9 references take about
# 230 MB of page-locked storage.
POSTERIORS = 16


def text_capacity(capacities: Sequence[int], tokens: int) -> int:
    """The text capacity a prompt of ``tokens`` is encoded at.

    The smallest configured capacity holding it, or the prompt's own length
    without configured capacities.

    Raises:
        ValueError: The prompt is empty or exceeds every capacity.
    """
    if tokens == 0:
        raise ValueError("text encoding requires a token")
    if not capacities:
        return tokens
    for capacity in sorted(capacities):
        if capacity >= tokens:
            return capacity
    raise ValueError(
        f"prompt of {tokens} tokens exceeds the configured text capacities"
    )


@dataclass(frozen=True)
class PromptVision:
    """The vision tokens of a prompt, for a text encoder that places them."""

    # ``(time, height, width)`` patch grids of the prompt's image blocks and
    # video blocks, each in prompt order.
    image_grids: tuple[tuple[int, int, int], ...]
    video_grids: tuple[tuple[int, int, int], ...]
    # The vision rows of its placeholders, in placeholder order.
    rows: torch.Tensor


class _Blocks:
    """The vision tower's buffers under CUDA graphs.

    Each block's patch rows are copied into ``rows``, its grid into ``grid``
    (``[1, 3]`` int64), and the graph captured at its grid writes its
    features into ``features``.
    """

    def __init__(
        self,
        grids: Sequence[tuple[int, int, int]],
        row_width: int,
        row_dtype: torch.dtype,
        output_width: int,
        output_dtype: torch.dtype,
        downsample: int,
        device: torch.device,
    ) -> None:
        # The grids captured, largest first.
        self.grids = sorted(set(grids), key=lambda grid: (-prod(grid), grid))
        rows = prod(self.grids[0]) if self.grids else 0
        tokens = rows // downsample**2
        # Rows past a block keep earlier values, which no graph reads; they
        # start finite.
        self.rows = torch.zeros(
            (max(rows, 1), row_width), dtype=row_dtype, device=device
        )
        self.features = torch.zeros(
            (max(tokens, 1), output_width), dtype=output_dtype, device=device
        )
        self.grid = torch.zeros((1, 3), dtype=torch.int64, device=device)
        self.graphs: dict[tuple[int, int, int], BreakableCUDAGraph] = {}


class TextEncoders:
    """A video model's text encoder component on its stream.

    Args:
        loaded: The loaded model.
        path: The text encoder's module path.
        stream: The primary stream of the component's device.
        attention: The attention backend the model loaded with.
        scratch: The stream's shared work areas.
        capacities: Configured prompt capacities, ascending; empty to encode
            every prompt at its own length.
        max_text_tokens: The longest admitted prompt.
        row_budget: The most vision patch rows one request carries.
        vision: Whether the component also encodes condition media with its
            vision tower.
        dtype: The model dtype the encoders compute in.
        pool: The graph pool captures allocate in; ``None`` encodes every
            prompt eagerly.
        block_grids: Under CUDA graphs, the keyframe grids whose blocks the
            vision tower captures.
        block_rows: ``(width, dtype)`` of the vision rows the media reader
            packs.
    """

    def __init__(
        self,
        *,
        loaded,
        path: str,
        stream: torch.cuda.Stream,
        attention: str,
        scratch,
        capacities: Sequence[int],
        max_text_tokens: int,
        row_budget: int,
        vision: bool,
        dtype: torch.dtype,
        pool: torch.cuda.MemPool | None,
        block_grids: Sequence[tuple[int, int, int]] | None,
        block_rows: tuple[int, torch.dtype] | None,
    ) -> None:
        self.stream = stream
        self.pool = pool
        self.capacities = sorted(set(capacities))
        device = stream.device
        self.device = device
        hidden = EncoderRunner.output_width(loaded, path)
        places = EncoderRunner.places_vision(loaded, path)
        tower_path = f"{path}.vision" if path else "vision"
        # One vision call encodes every vision block of a request. Their
        # patch rows together are within the row budget, and a block holds
        # at least one feature, so the budget also bounds the blocks.
        downsample = (
            EncoderRunner.downsample(loaded, tower_path) if vision else 1
        )
        max_images = max(row_budget // downsample**2, 1)
        # The context serves both encoders, so its operators are prepared
        # for the larger of their workloads: a prompt padded to the largest
        # capacity, and a request's vision blocks.
        tokens = max(
            self.capacities[-1] if self.capacities else max_text_tokens, 1
        )
        rows = 1
        if vision:
            tokens = max(
                tokens,
                EncoderRunner.image_patches(loaded, tower_path),
                row_budget,
                1,
            )
            rows = max(rows, max_images)
        self.runner = EncoderRunner(
            loaded, path, stream.cuda_stream, attention, scratch, [tokens, rows]
        )
        self.tower = self.runner.tower("vision") if vision else None
        self.final_stage = self.runner.final_stage()
        capacity = tokens if not self.capacities else self.capacities[-1]
        capacity = max(capacity, 1)
        with torch.cuda.device(device):
            # The ``[max capacity]`` int64 prompt, zero-padded.
            self.tokens = torch.zeros(
                capacity, dtype=torch.int64, device=device
            )
            # The ``[max capacity, hidden]`` features of the padded prompt.
            self.features = torch.zeros(
                (capacity, hidden), dtype=dtype, device=device
            )
            # The ``[3, max capacity]`` rotary coordinates of a prompt with
            # vision tokens.
            self.positions = (
                torch.zeros((3, capacity), dtype=torch.int64, device=device)
                if places
                else None
            )
        self.graphs: dict[int, BreakableCUDAGraph] = {}
        self.vision_graphs: dict[int, BreakableCUDAGraph] = {}
        # Under CUDA graphs, the ``[max capacity, width]`` vision rows of a
        # prompt with vision tokens, which every capacity's graph of such a
        # prompt reads, and the tower's block buffers.
        self.visual: torch.Tensor | None = None
        self.blocks: _Blocks | None = None
        if (
            pool is not None
            and self.tower is not None
            and block_grids is not None
            and block_rows is not None
            and self.positions is not None
        ):
            output_width = EncoderRunner.output_width(loaded, tower_path)
            output_dtype = getattr(
                torch, EncoderRunner.output_dtype(loaded, tower_path)
            )
            row_width, row_dtype = block_rows
            self.blocks = _Blocks(
                block_grids,
                row_width,
                row_dtype,
                output_width,
                output_dtype,
                downsample,
                device,
            )
            self.visual = torch.zeros(
                (capacity, output_width), dtype=output_dtype, device=device
            )
        self.downsample = downsample

    def encode_text(
        self,
        tokens: Sequence[int],
        out: torch.Tensor,
        vision: PromptVision | None,
    ) -> int:
        """Encode one prompt into ``out``, its ``[tokens, hidden]`` features.

        A pipeline stage that emits no features writes nothing. Returns how
        the forward ran.
        """
        capacity = text_capacity(self.capacities, len(tokens))
        if capacity > self.tokens.shape[0]:
            raise ValueError("text encoder input exceeds its token capacity")
        padded = [int(token) for token in tokens] + [0] * (
            capacity - len(tokens)
        )
        with torch.cuda.stream(self.stream):
            self.tokens[:capacity].copy_(
                torch.tensor(padded, dtype=torch.int64).pin_memory(),
                non_blocking=True,
            )
            if vision is not None:
                if self.positions is None:
                    raise ValueError("the text encoder places no vision tokens")
                self.runner.position_ids(
                    padded,
                    [extent for grid in vision.image_grids for extent in grid],
                    [extent for grid in vision.video_grids for extent in grid],
                    self.positions,
                )
                if self.visual is not None:
                    rows = vision.rows
                    if (
                        rows.dtype != self.visual.dtype
                        or rows.shape[1:] != self.visual.shape[1:]
                        or rows.shape[0] > capacity
                    ):
                        raise ValueError(
                            "vision rows disagree with the captured prompt "
                            "encodes"
                        )
                    self.visual[: rows.shape[0]].copy_(rows)
                    mode = self._forward(
                        self.vision_graphs,
                        capacity,
                        lambda: self._encode_vision(capacity),
                    )
                else:
                    self.runner.encode(
                        self.tokens[:capacity],
                        self.features[:capacity],
                        self.positions,
                        vision.rows,
                    )
                    mode = EAGER
            elif not self.capacities:
                # Without configured capacities a prompt is encoded at its
                # exact length, which no graph serves.
                self._encode(capacity)
                mode = EAGER
            else:
                mode = self._forward(
                    self.graphs, capacity, lambda: self._encode(capacity)
                )
            if self.final_stage:
                out.copy_(self.features[: out.shape[0]])
        return mode

    def _encode(self, capacity: int) -> None:
        self.runner.encode(self.tokens[:capacity], self.features[:capacity])

    def _encode_vision(self, capacity: int) -> None:
        assert self.positions is not None and self.visual is not None
        self.runner.encode(
            self.tokens[:capacity],
            self.features[:capacity],
            self.positions,
            self.visual[:capacity],
        )

    @staticmethod
    def _forward(graphs: dict, key, eager) -> int:
        graph = graphs.get(key)
        if graph is not None:
            graph.replay()
            return REPLAY
        eager()
        return EAGER

    def encode_patches(
        self,
        rows: torch.Tensor,
        grids: Sequence[tuple[int, int, int]],
        out: torch.Tensor,
    ) -> int:
        """Encode vision blocks into ``out``'s rows in block order.

        The blocks' patch rows are packed back to back in ``rows``, each
        block of its ``(time, height, width)`` grid. Returns how the forwards
        ran: replay when every block replayed.
        """
        tower = self.tower
        if tower is None:
            raise ValueError("this rank's text encoder reads no vision")
        blocks = self.blocks
        with torch.cuda.stream(self.stream):
            if blocks is None:
                tower.encode(
                    rows, out, [extent for grid in grids for extent in grid]
                )
                return EAGER
            if (
                rows.dtype != blocks.rows.dtype
                or rows.shape[1:] != blocks.rows.shape[1:]
            ):
                raise ValueError(
                    "vision rows disagree with the captured blocks' rows"
                )
            mode = REPLAY
            row = 0
            token = 0
            for grid in grids:
                count = prod(grid)
                tokens = count // self.downsample**2
                graph = blocks.graphs.get(grid)
                if graph is None:
                    tower.encode(
                        rows[row : row + count],
                        out[token : token + tokens],
                        list(grid),
                    )
                    mode = EAGER
                else:
                    blocks.rows[:count].copy_(rows[row : row + count])
                    blocks.grid.copy_(
                        torch.tensor([grid], dtype=torch.int64).pin_memory(),
                        non_blocking=True,
                    )
                    graph.replay()
                    out[token : token + tokens].copy_(blocks.features[:tokens])
                row += count
                token += tokens
        return mode

    def _encode_block(self, grid: tuple[int, int, int]) -> None:
        assert self.tower is not None and self.blocks is not None
        count = prod(grid)
        tokens = count // self.downsample**2
        self.tower.encode(
            self.blocks.rows[:count],
            self.blocks.features[:tokens],
            list(grid),
            self.blocks.grid,
        )

    def capture(self) -> None:
        """Warm and, with a graph pool, capture every graph the encoders run.

        The prompt capacities, largest first (with and without vision rows),
        then the tower's block grids, largest first. Without a pool the same
        forwards run once, so lazy initialization precedes serving.
        """
        with torch.cuda.stream(self.stream):
            # Every capacity encodes the zero prompt, and under graphs with
            # zero vision rows at zero coordinates.
            self.tokens.zero_()
            if self.visual is not None:
                assert self.positions is not None
                self.positions.zero_()
                self.visual.zero_()
        for capacity in reversed(self.capacities):
            self._warm_capture(
                self.graphs, capacity, lambda c=capacity: self._encode(c)
            )
            if self.visual is not None:
                self._warm_capture(
                    self.vision_graphs,
                    capacity,
                    lambda c=capacity: self._encode_vision(c),
                )
        if self.blocks is not None:
            for grid in self.blocks.grids:
                with torch.cuda.stream(self.stream):
                    self.blocks.grid.copy_(
                        torch.tensor([grid], dtype=torch.int64),
                        non_blocking=False,
                    )
                self._warm_capture(
                    self.blocks.graphs,
                    grid,
                    lambda g=grid: self._encode_block(g),
                )
        self.stream.synchronize()

    def _warm_capture(self, graphs: dict, key, body) -> None:
        with torch.cuda.stream(self.stream):
            body()
        if self.pool is not None:
            graphs[key] = capture(body, stream=self.stream, pool=self.pool)

    def kernels(self) -> list[str]:
        return list(self.runner.kernels())

    def close(self, aborted: bool) -> None:
        """Release the graphs, then the prompt encoder's execution context.

        The caller drained the stream.
        """
        if aborted:
            return
        graphs = [*self.graphs.values(), *self.vision_graphs.values()]
        if self.blocks is not None:
            graphs.extend(self.blocks.graphs.values())
        close_all(graphs, [self.runner])


# One visual unit as the encoder takes it: ``[pixel_offset, frames, width,
# height, window, first_row, stop_row]``: its condition's first pixel row,
# frame count and raster, the index of the encoder window the unit's frames
# are, and the run of the frames' rows of 32-pixel latent patches it
# encodes.
ConditionUnit = tuple[int, int, int, int, int, int, int]


class _Keyframes:
    """The keyframe encodings startup captured under CUDA graphs.

    They are keyed by input shapes (``[1, height, width, 3]`` per keyframe):
    a call copies its keyframes' pixels into ``pixels``, back to back,
    replays the graph of its shapes and copies its rows out of ``rows``.
    """

    def __init__(self, pixels: torch.Tensor, rows: torch.Tensor) -> None:
        self.pixels = pixels
        self.rows = rows
        self.graphs: dict[tuple[int, ...], BreakableCUDAGraph] = {}
        self.runners: dict[tuple[int, ...], EncoderRunner] = {}


class ConditionEncoders:
    """A video model's condition latent encoders on their stream.

    Args:
        loaded: The loaded model.
        video_path: The video encoder's module path.
        audio_path: The audio encoder's module path.
        stream: The primary stream of the encoders' device.
        attention: The attention backend the model loaded with.
        scratch: The stream's shared work areas.
        pool: The graph pool captures allocate in; ``None`` encodes every
            condition eagerly.
    """

    def __init__(
        self,
        *,
        loaded,
        video_path: str,
        audio_path: str,
        stream: torch.cuda.Stream,
        attention: str,
        scratch,
        pool: torch.cuda.MemPool | None,
    ) -> None:
        self.loaded = loaded
        self.video_path = video_path
        self.audio_path = audio_path
        self.stream = stream
        self.attention = attention
        self.scratch = scratch
        self.pool = pool
        self.device = stream.device
        # Condition encoders are constructed per input shape while serving;
        # the natively built kernels they launch load now, so no encoding
        # compiles one.
        EncoderRunner.load_condition_kernels(loaded)
        # ``[1, height, width, 3]`` of a still frame one tile in size; empty
        # when the video encoder does not tile still frames.
        self.still_tile = tuple(EncoderRunner.still_tile(loaded, video_path))
        self.frames: dict[int, list[int]] = {}
        self.video_runners: dict[tuple[int, ...], EncoderRunner] = {}
        self.audio_runners: dict[tuple[int, ...], EncoderRunner] = {}
        # The posterior draw of each condition size ``(frames, width,
        # height)``, ``None`` when the encoder takes none; oldest first.
        self.posteriors: OrderedDict[
            tuple[int, int, int], torch.Tensor | None
        ] = OrderedDict()
        self.tile_runner: EncoderRunner | None = None
        self.tile_graph: BreakableCUDAGraph | None = None
        self.keyframes: _Keyframes | None = None

    def _runner(self, shapes: tuple[int, ...], video: bool) -> EncoderRunner:
        runners = self.video_runners if video else self.audio_runners
        runner = runners.get(shapes)
        if runner is None:
            path = self.video_path if video else self.audio_path
            tile_runner, tile_launch = None, None
            if (
                video
                and self.tile_runner is not None
                and self.tile_graph is not None
            ):
                # A video runner replays the captured still-frame tile.
                tile_runner, tile_launch = (
                    self.tile_runner,
                    self.tile_graph.replay,
                )
            runner = EncoderRunner(
                self.loaded,
                path,
                self.stream.cuda_stream,
                self.attention,
                self.scratch,
                list(shapes),
                tile_runner,
                tile_launch,
            )
            runners[shapes] = runner
        return runner

    def unit_frames(self, frames: int) -> list[int]:
        """Frames of each visual unit of a condition of ``frames`` frames."""
        lengths = self.frames.get(frames)
        if lengths is None:
            lengths = EncoderRunner.unit_frames(self.loaded, frames)
            self.frames[frames] = lengths
        return lengths

    def _shapes(self, units: Sequence[ConditionUnit]) -> tuple[int, ...]:
        shapes: list[int] = []
        for _, frames, width, height, unit, _, _ in units:
            lengths = self.unit_frames(frames)
            if unit >= len(lengths):
                raise ValueError("a condition unit lies outside its condition")
            shapes.extend([lengths[unit], height, width, 3])
        return tuple(shapes)

    def _posterior(
        self,
        size: tuple[int, int, int],
        current: Sequence[tuple[int, int, int]],
    ) -> torch.Tensor | None:
        """A condition size's posterior draw, filled on first use.

        Draws are held for the most recently used sizes; a size ``current``
        reads or a captured keyframe encoding reads is never released.
        """
        if size in self.posteriors:
            self.posteriors.move_to_end(size)
            return self.posteriors[size]
        frames, width, height = size
        elements = EncoderRunner.posterior_elements(
            self.loaded, frames, width, height
        )
        noise = None
        if elements:
            noise = torch.empty(elements, dtype=torch.float32, pin_memory=True)
            EncoderRunner.draw_posterior(
                self.loaded, frames, width, height, noise
            )
        self.posteriors[size] = noise
        while len(self.posteriors) > POSTERIORS:
            released = next(
                (
                    held
                    for held in self.posteriors
                    if held not in current and not self._captured(held)
                ),
                None,
            )
            if released is None:
                break
            del self.posteriors[released]
        return noise

    def _captured(self, size: tuple[int, int, int]) -> bool:
        frames, width, height = size
        keyframes = self.keyframes
        return keyframes is not None and any(
            shapes[:4] == (frames, height, width, 3)
            for shapes in keyframes.runners
        )

    def encode_units(
        self,
        pixels: torch.Tensor,
        units: Sequence[ConditionUnit],
        out: torch.Tensor,
    ) -> int:
        """Encode visual units into consecutive rows of ``out``.

        Returns how the forward ran.
        """
        shapes = self._shapes(units)
        keyframes = self.keyframes
        with torch.cuda.stream(self.stream):
            graph = None if keyframes is None else keyframes.graphs.get(shapes)
            if graph is not None:
                assert keyframes is not None
                # Each keyframe's pixels are copied into the encoders', back
                # to back, the graph of the call's shapes replays, and the
                # rows are copied out.
                offset = 0
                for first, frames, width, height, *_ in units:
                    count = frames * width * height
                    keyframes.pixels[offset : offset + count].copy_(
                        pixels[first : first + count]
                    )
                    offset += count
                graph.replay()
                out.copy_(keyframes.rows[: out.shape[0]])
                return REPLAY
            sizes = [
                (frames, width, height)
                for _, frames, width, height, *_ in units
            ]
            draws = [self._posterior(size, sizes) for size in sizes]
            runner = self._runner(shapes, True)
            runner.encode(
                pixels,
                out,
                [value for unit in units for value in unit],
                *(draw for draw in draws if draw is not None),
            )
        return EAGER

    def encode_tracks(
        self, samples: torch.Tensor, lengths: Sequence[int], out: torch.Tensor
    ) -> int:
        """Encode audio tracks of ``lengths`` samples into rows of ``out``."""
        shapes = tuple(
            value for length in lengths for value in (int(length), 2)
        )
        with torch.cuda.stream(self.stream):
            self._runner(shapes, False).encode(
                samples, out, [int(length) for length in lengths]
            )
        return EAGER

    # Startup.

    def capture_tile(self) -> None:
        """Capture the encoding of one still-frame tile.

        Nothing is captured for a video encoder that does not tile still
        frames or without a graph pool.
        """
        if (
            self.pool is None
            or not self.still_tile
            or self.tile_graph is not None
        ):
            return
        runner = EncoderRunner.tile(
            self.loaded,
            self.video_path,
            self.stream.cuda_stream,
            self.attention,
            self.scratch,
        )
        # The eager encoding compiles the tile's kernels outside the capture.
        with torch.cuda.stream(self.stream):
            runner.encode_tile()
        self.tile_graph = capture(
            runner.encode_tile, stream=self.stream, pool=self.pool
        )
        self.tile_runner = runner

    def capture_keyframes(
        self,
        canvases: Sequence[tuple[int, int]],
        per_call: int,
        row_width: int,
        row_dtype: torch.dtype,
    ) -> None:
        """Capture the encoding of ``1..=per_call`` keyframes per canvas.

        ``canvases`` are ``(width, height)``; the encodings write rows of
        ``row_width`` values of ``row_dtype``. A size's posterior draw is
        filled here and held for the graphs' life.
        """
        if self.pool is None:
            return
        ordered = sorted(canvases, key=lambda canvas: -canvas[0] * canvas[1])
        largest = ordered[0][0] * ordered[0][1] if ordered else 0
        most_rows = max(
            (rows_per_frame(Canvas(*canvas)) for canvas in ordered),
            default=0,
        )
        per_call = max(per_call, 1)
        device = self.device
        keyframes = _Keyframes(
            torch.zeros(
                (max(largest * per_call, 1), 3),
                dtype=torch.uint8,
                device=device,
            ),
            torch.zeros(
                (max(most_rows * per_call, 1), row_width),
                dtype=row_dtype,
                device=device,
            ),
        )
        self.stream.synchronize()
        for width, height in ordered:
            size = (1, width, height)
            noise = self._posterior(size, [size])
            frame_rows = rows_per_frame(Canvas(width, height))
            for count in range(per_call, 0, -1):
                shapes = tuple(
                    value
                    for _ in range(count)
                    for value in (1, height, width, 3)
                )
                runner = EncoderRunner(
                    self.loaded,
                    self.video_path,
                    self.stream.cuda_stream,
                    self.attention,
                    self.scratch,
                    list(shapes),
                    None,
                    None,
                )
                units = [
                    value
                    for index in range(count)
                    for value in (
                        index * width * height,
                        1,
                        width,
                        height,
                        0,
                        0,
                        height // CANVAS_MULTIPLE,
                    )
                ]
                pixels = keyframes.pixels[: count * width * height]
                rows = keyframes.rows[: count * frame_rows]
                draws = [noise] * count if noise is not None else []

                def encode(
                    runner=runner,
                    pixels=pixels,
                    rows=rows,
                    units=units,
                    draws=draws,
                ):
                    runner.encode(pixels, rows, units, *draws)

                with torch.cuda.stream(self.stream):
                    encode()
                keyframes.graphs[shapes] = capture(
                    encode, stream=self.stream, pool=self.pool
                )
                keyframes.runners[shapes] = runner
        self.keyframes = keyframes
        self.stream.synchronize()

    def kernels(self) -> list[str]:
        kernels: list[str] = []
        runners = [*self.video_runners.values(), *self.audio_runners.values()]
        if self.keyframes is not None:
            runners.extend(self.keyframes.runners.values())
        for runner in runners:
            kernels.extend(runner.kernels())
        return kernels

    def close(self, aborted: bool) -> None:
        """Release the graphs, then close the runners, the tile runner last.

        The video runners replay the tile's graph; the caller drained the
        stream.
        """
        if aborted:
            return
        graphs: list[BreakableCUDAGraph] = []
        runners = [*self.video_runners.values(), *self.audio_runners.values()]
        if self.keyframes is not None:
            graphs.extend(self.keyframes.graphs.values())
            runners.extend(self.keyframes.runners.values())
        if self.tile_graph is not None:
            graphs.append(self.tile_graph)
        if self.tile_runner is not None:
            runners.append(self.tile_runner)
        close_all(graphs, runners)
