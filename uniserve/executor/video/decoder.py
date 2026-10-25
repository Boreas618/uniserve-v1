"""A video model's decoders: completed latents to frames and samples.

A video decoding round hands each rank of the decoder's unit ring one media
unit. :class:`VideoDecoder` decodes it: the forward unpacks the unit's clip
input from the request's packed latent, decodes it at its clip output and
cross-fades and quantizes it into the unit's RGB frames, exchanging decoded
overlaps with its ring neighbours. The request's retained overlap lives in
its slot's row. A forward reads and writes tensors at fixed addresses, so
one captured graph per unit shape serves every request; the request's
values are copied in and out around it.

:class:`AudioDecoder` decodes a contiguous run of a track's audio units into
their sample span, one captured graph per track units.

A runner serves one module at one size, so a decoder holds one per size it
serves: the video decoder one per clip output, the audio decoder one per
track length, all constructed at startup for every admitted video.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from uniserve.cuda_graph import BreakableCUDAGraph, capture
from uniserve.executor.video.denoiser import EAGER, REPLAY
from uniserve.executor.video.plan import (
    Canvas,
    MediaPlan,
    VideoModel,
    VideoSize,
)
from uniserve.runner.decoder import DecoderRunner

__all__ = [
    "AudioDecoder",
    "TrackUnits",
    "UnitShape",
    "VideoDecoder",
    "clip",
    "overlap",
    "units",
]


def units(
    distributed: bool,
    ranks: Sequence[int],
    units_per_rank: int,
    dealt: Sequence[int],
    rank: int,
    cursor: int,
    count: int,
) -> range:
    """This rank's media units of a round covering units ``[cursor, cursor +
    count)``: all of them for an undistributed component, otherwise the
    rank's run of ``units_per_rank`` units from its position among the ranks
    the round is ``dealt`` to, or among the component's ``ranks`` in
    configured order when it names none.

    Raises:
        ValueError: The rank is not a member of the component.
    """  # noqa: D205
    if not distributed:
        return range(cursor, cursor + count)
    if rank not in ranks:
        raise ValueError("media units assigned outside the component")
    order = list(dealt) if dealt else list(ranks)
    if rank not in order:
        return range(cursor, cursor)
    position = order.index(rank)
    first = min(position * units_per_rank, count)
    last = min(position * units_per_rank + units_per_rank, count)
    return range(cursor + first, cursor + last)


@dataclass(frozen=True)
class UnitShape:
    """What fixes a video decoding forward's computation: the unit's clip
    output, whether the unit is its video's first and last, and how many
    ring members hold a unit of its round.
    """  # noqa: D205

    # ``(frames, canvas)`` of the clip output.
    clip_output: tuple[int, Canvas]
    first: bool
    last: bool
    count: int


class VideoDecoder:
    """A video decoder and its RGB post-processor on one stream.

    Args:
        loaded: The loaded model.
        path: The decoder's module path.
        model: The model's facts.
        plan: What the rank admits.
        stream: The decoder's lane stream.
        attention: The attention backend the model loaded with.
        scratch: The stream's shared work areas.
        slots: Request slots.
        pool: The graph pool captures allocate in; ``None`` decodes every
            unit eagerly.
    """

    def __init__(
        self,
        *,
        loaded,
        path: str,
        model: VideoModel,
        plan: MediaPlan,
        stream: torch.cuda.Stream,
        attention: str,
        scratch,
        slots: int,
        pool: torch.cuda.MemPool | None,
    ) -> None:
        self.loaded = loaded
        self.path = path
        self.model = model
        self.stream = stream
        self.attention = attention
        self.scratch = scratch
        self.pool = pool
        device = stream.device
        decoder = model.decoder
        overlap_elements = 0
        overlap_dtype = torch.float16
        rows = width = 0
        height = canvas_width = 0
        longest = max(model.unit_frames(plan.max_frames))
        for canvas in plan.canvases:
            elements, dtype = overlap(model, canvas)
            overlap_elements = max(overlap_elements, elements)
            overlap_dtype = dtype
            layout = model.resolve(
                model.layout_size(
                    VideoSize(plan.max_frames, canvas, plan.max_text_tokens)
                )
            )
            shape = model.denoiser.latent_shape("video", layout)
            rows = max(rows, int(shape[0]))
            width = int(shape[1])
            height = max(height, canvas.height)
            canvas_width = max(canvas_width, canvas.width)
        del decoder
        with torch.cuda.device(device):
            # The retained overlap of each request slot: the decoded tail a
            # request's previous round left for its next.
            self.overlap = torch.zeros(
                (slots + 1, max(overlap_elements, 1)),
                dtype=overlap_dtype,
                device=device,
            )
            # A request's complete packed latent.
            self.latent = torch.zeros(
                (max(rows, 1), max(width, 1)),
                dtype=torch.float32,
                device=device,
            )
            # The latent rows a unit's clip input reads.
            self.rows = torch.zeros(
                max(rows, 1), dtype=torch.int64, device=device
            )
            # The forward's view of the request's retained overlap.
            self.retained = torch.zeros(
                max(overlap_elements, 1), dtype=overlap_dtype, device=device
            )
            # The unit's ``[frames, height, width, 3]`` RGB frames.
            self.frames = torch.zeros(
                max(longest * height * canvas_width * 3, 1),
                dtype=torch.uint8,
                device=device,
            )
        # The runner of each clip output, in preparation order.
        self.runners: dict[tuple[int, Canvas], DecoderRunner] = {}
        # The clip rows and clip output of each ``(frames, canvas, unit)``.
        self.clip_rows: dict[tuple[int, Canvas, int], torch.Tensor] = {}
        self.clip_outputs: dict[
            tuple[int, Canvas, int], tuple[int, Canvas]
        ] = {}
        self.graphs: dict[UnitShape, BreakableCUDAGraph] = {}
        # The unit shapes startup warmed, which it then captures: each with
        # its clip input's latent rows and its video's longest unit.
        self.warmed: list[tuple[UnitShape, int, int]] = []
        self.sealed = False

    def prepare(self, plan: MediaPlan) -> None:
        """Read every admitted video's clip outputs and clip rows and
        construct the runner of every clip output, the largest first.
        """  # noqa: D205
        for canvas in plan.canvases:
            for frames in reversed(plan.frame_counts):
                for unit in range(len(self.model.unit_frames(frames))):
                    key = (frames, canvas, unit)
                    clip_output, rows = clip(self.model, frames, canvas, unit)
                    self.clip_outputs[key] = clip_output
                    self.clip_rows[key] = torch.tensor(
                        rows, dtype=torch.int64
                    ).pin_memory()
                    if clip_output not in self.runners:
                        frames_out, raster = clip_output
                        self.runners[clip_output] = DecoderRunner(
                            self.loaded,
                            self.path,
                            self.stream.cuda_stream,
                            self.attention,
                            self.scratch,
                            [frames_out, raster.width, raster.height],
                        )

    def _frames(self, canvas: Canvas, frames: int) -> torch.Tensor:
        count = frames * canvas.height * canvas.width * 3
        return self.frames[:count].view(frames, canvas.height, canvas.width, 3)

    def _forward(self, shape: UnitShape, rows: int, longest: int) -> None:
        runner = self.runners.get(shape.clip_output)
        if runner is None:
            raise AssertionError("a video clip output has no decoder runner")
        runner.decode(
            self.latent,
            self.rows[:rows],
            self.retained,
            self._frames(shape.clip_output[1], longest),
            shape.first,
            shape.last,
            shape.count,
        )

    def _run(self, shape: UnitShape, rows: int, longest: int) -> int:
        graph = self.graphs.get(shape)
        if graph is not None:
            graph.replay()
            return REPLAY
        if self.pool is not None and self.sealed:
            raise ValueError(f"no captured graph decodes a unit of {shape}")
        self._forward(shape, rows, longest)
        return EAGER

    def decode(
        self,
        slot: int,
        latent: torch.Tensor,
        frames: int,
        canvas: Canvas,
        unit: int,
        shape: UnitShape,
        longest: int,
        written: int,
        out: torch.Tensor,
    ) -> int:
        """Decode and reconstruct media unit ``unit`` of a ``frames``-frame
        video into the leading ``written`` frames of ``out`` (``[longest,
        height, width, 3]`` uint8); the frames past them become zero.
        ``latent`` is the request's complete packed latent.
        """  # noqa: D205
        rows = self.clip_rows.get((frames, canvas, unit))
        if rows is None:
            raise ValueError("the rank admits no video with this media unit")
        count = latent.numel()
        if latent.dtype != torch.float32 or count > self.latent.numel():
            raise ValueError("a video latent exceeds the decoder's latent rows")
        with torch.cuda.stream(self.stream):
            self.rows[: rows.shape[0]].copy_(rows, non_blocking=True)
            self.latent.view(-1)[:count].copy_(latent.reshape(-1))
            overlap = self.overlap[slot]
            self.retained.copy_(overlap[: self.retained.numel()])
            mode = self._run(shape, rows.shape[0], longest)
            overlap[: self.retained.numel()].copy_(self.retained)
            unit_frames = self._frames(canvas, longest)
            out[:written].copy_(unit_frames[:written])
            # A short unit's row keeps zero frames after its own for its
            # reader.
            out[written:].zero_()
        return mode

    def warm(
        self,
        plan: MediaPlan,
        ranks: Sequence[int],
        units_per_rank: int,
        distributed: bool,
        rank: int,
    ) -> None:
        """Warm one round of each shape of the unit ring: a round's shape is
        its canvas, whether it holds the video's first and last units, and
        how many members hold a unit. Every member derives the same rounds
        in the same order, because rounds exchange overlaps.
        """  # noqa: D205
        width = len(ranks) * max(units_per_rank, 1)
        warmed: set[tuple[Canvas, bool, bool, int]] = set()
        for canvas in plan.canvases:
            for frames in reversed(plan.frame_counts):
                lengths = self.model.unit_frames(frames)
                total = len(lengths)
                longest = max(lengths)
                cursor = 0
                while cursor < total:
                    count = min(width, total - cursor)
                    round_shape = (
                        canvas,
                        cursor == 0,
                        cursor + count == total,
                        count,
                    )
                    run = units(
                        distributed,
                        ranks,
                        units_per_rank,
                        (),
                        rank,
                        cursor,
                        count,
                    )
                    if round_shape not in warmed and len(run) > 0:
                        warmed.add(round_shape)
                        unit = run.start
                        shape = UnitShape(
                            self.clip_outputs[(frames, canvas, unit)],
                            unit == 0,
                            unit + 1 == total,
                            count,
                        )
                        rows = self.clip_rows[(frames, canvas, unit)].shape[0]
                        with torch.cuda.stream(self.stream):
                            self.latent.zero_()
                            self.rows.zero_()
                            self.retained.zero_()
                            self._forward(shape, rows, longest)
                        self.stream.synchronize()
                        self.warmed.append((shape, rows, longest))
                    cursor += count

    def capture(self) -> None:
        """Capture every warmed unit shape, then refuse eager forwards."""
        if self.pool is not None:
            for shape, rows, longest in self.warmed:
                if shape not in self.graphs:
                    self.graphs[shape] = capture(
                        functools.partial(self._forward, shape, rows, longest),
                        stream=self.stream,
                        pool=self.pool,
                    )
        self.warmed.clear()
        self.stream.synchronize()
        self.sealed = True

    def kernels(self) -> list[str]:
        kernels: list[str] = []
        for runner in self.runners.values():
            kernels.extend(runner.kernels())
        return kernels

    def close(self, aborted: bool) -> None:
        if aborted:
            return
        _close(self.graphs.values(), self.runners.values())
        self.graphs.clear()


@dataclass(frozen=True)
class TrackUnits:
    """What fixes an audio decoding forward's computation: the track's
    samples and which of how many units the forward decodes.
    """  # noqa: D205

    samples: int
    # Latent frames of the track.
    latent_frames: int
    first: int
    count: int
    total: int
    # Samples the units span.
    span: int


class AudioDecoder:
    """One rank's audio decoder.

    Args:
        loaded: The loaded model.
        path: The decoder's module path.
        model: The model's facts.
        plan: What the rank admits.
        stream: The decoder's lane stream.
        attention: The attention backend the model loaded with.
        scratch: The stream's shared work areas.
        pool: The graph pool captures allocate in; ``None`` decodes every
            track eagerly.
    """

    def __init__(
        self,
        *,
        loaded,
        path: str,
        model: VideoModel,
        plan: MediaPlan,
        stream: torch.cuda.Stream,
        attention: str,
        scratch,
        pool: torch.cuda.MemPool | None,
    ) -> None:
        self.loaded = loaded
        self.path = path
        self.model = model
        self.stream = stream
        self.attention = attention
        self.scratch = scratch
        self.pool = pool
        device = stream.device
        # A track's latent does not depend on the canvas.
        layout = model.resolve(
            model.layout_size(
                VideoSize(
                    plan.max_frames, plan.canvases[0], plan.max_text_tokens
                )
            )
        )
        rows, width = (
            int(extent)
            for extent in model.denoiser.latent_shape("audio", layout)
        )
        samples = model.track_samples(plan.max_frames)
        with torch.cuda.device(device):
            # A request's complete audio latent.
            self.latent = torch.zeros(
                (max(rows, 1), max(width, 1)),
                dtype=torch.float32,
                device=device,
            )
            # The samples of the decoded units.
            self.samples = torch.zeros(
                (max(samples, 1), 2), dtype=torch.int16, device=device
            )
        # The runner of each track length, by its latent frames.
        self.runners: dict[int, DecoderRunner] = {}
        # Each admitted track decoded in some number of units, by
        # ``(samples, units)``: its latent frames and each unit's samples.
        self.tracks: dict[
            tuple[int, int], tuple[int, list[tuple[int, int]]]
        ] = {}
        self.graphs: dict[TrackUnits, BreakableCUDAGraph] = {}
        self.warmed: list[tuple[TrackUnits, int]] = []
        self.sealed = False

    def track(
        self, samples: int, count: int
    ) -> tuple[int, list[tuple[int, int]]]:
        """The latent frames of an admitted ``samples``-sample track and the
        samples of each of its ``count`` audio units.
        """  # noqa: D205
        track = self.tracks.get((samples, count))
        if track is None:
            raise ValueError("the rank admits no track of this length")
        return track

    def _forward(self, track: TrackUnits, rows: int) -> None:
        runner = self.runners.get(track.latent_frames)
        if runner is None:
            raise ValueError("the audio track's length has no decoder runner")
        runner.decode(
            self.latent[:rows],
            self.samples[: track.span],
            track.samples,
            track.first,
            track.count,
            track.total,
        )

    def _run(self, track: TrackUnits, rows: int) -> int:
        graph = self.graphs.get(track)
        if graph is not None:
            graph.replay()
            return REPLAY
        if self.pool is not None and self.sealed:
            raise ValueError(f"no captured graph decodes audio units {track}")
        self._forward(track, rows)
        return EAGER

    def decode(
        self, latent: torch.Tensor, track: TrackUnits, out: torch.Tensor
    ) -> int:
        """Decode ``track``'s units into ``out`` (``[span, 2]`` int16).
        ``latent`` is the request's complete audio latent.
        """  # noqa: D205
        rows = latent.shape[0]
        if (
            latent.dtype != torch.float32
            or latent.numel() > self.latent.numel()
        ):
            raise ValueError(
                "an audio latent exceeds the decoder's latent rows"
            )
        with torch.cuda.stream(self.stream):
            self.latent.view(-1)[: latent.numel()].copy_(latent.reshape(-1))
            mode = self._run(track, rows)
            out.copy_(self.samples[: track.span])
        return mode

    def prepare(
        self,
        plan: MediaPlan,
        ranks: Sequence[int],
        units_per_rank: int,
        distributed: bool,
        rank: int,
    ) -> None:
        """Construct the runner of every admitted track, read its unit spans
        for every number of units a round decodes it in, and warm one
        decode of this rank's units of each track.
        """  # noqa: D205
        audio = self.model.audio
        total = len(ranks) * max(units_per_rank, 1) if distributed else 1
        canvas = plan.canvases[0]
        for frames in reversed(plan.frame_counts):
            samples = self.model.track_samples(frames)
            latent_frames = int(audio.latent_frames(samples))
            if latent_frames not in self.runners:
                self.runners[latent_frames] = DecoderRunner(
                    self.loaded,
                    self.path,
                    self.stream.cuda_stream,
                    self.attention,
                    self.scratch,
                    [latent_frames],
                )
            for count in range(1, total + 1):
                spans = [
                    (int(span.start), int(span.stop))
                    for span in audio.unit_samples(samples, count)
                ]
                self.tracks[(samples, count)] = (latent_frames, spans)
        for frames in reversed(plan.frame_counts):
            samples = self.model.track_samples(frames)
            layout = self.model.resolve(
                self.model.layout_size(
                    VideoSize(frames, canvas, plan.max_text_tokens)
                )
            )
            rows = int(self.model.denoiser.latent_shape("audio", layout)[0])
            run = units(distributed, ranks, units_per_rank, (), rank, 0, total)
            if len(run) == 0:
                continue
            latent_frames, spans = self.tracks[(samples, total)]
            track = TrackUnits(
                samples,
                latent_frames,
                run.start,
                len(run),
                total,
                spans[run.stop - 1][1] - spans[run.start][0],
            )
            with torch.cuda.stream(self.stream):
                self.latent.zero_()
                self._forward(track, rows)
            self.stream.synchronize()
            self.warmed.append((track, rows))

    def capture(self) -> None:
        """Capture every warmed track's units, then refuse eager forwards."""
        if self.pool is not None:
            for track, rows in self.warmed:
                if track not in self.graphs:
                    self.graphs[track] = capture(
                        functools.partial(self._forward, track, rows),
                        stream=self.stream,
                        pool=self.pool,
                    )
        self.warmed.clear()
        self.stream.synchronize()
        self.sealed = True

    def kernels(self) -> list[str]:
        kernels: list[str] = []
        for runner in self.runners.values():
            kernels.extend(runner.kernels())
        return kernels

    def close(self, aborted: bool) -> None:
        if aborted:
            return
        _close(self.graphs.values(), self.runners.values())
        self.graphs.clear()


def _video(frames: int, canvas: Canvas):
    from uniserve.media import image, video

    return video.Config(int(frames), image.Config(canvas.height, canvas.width))


def overlap(model: VideoModel, canvas: Canvas) -> tuple[int, torch.dtype]:
    """Elements and dtype of the overlap a canvas's unit carries to the next."""
    decoder = model.decoder
    frames = 1
    while True:
        try:
            decoder.clips(frames)
        except ValueError:
            frames += 1
            continue
        break
    state = model.clock.state_buffers(_video(frames, canvas))["video_overlap"]
    count = 1
    for extent in state.shape:
        count *= int(extent)
    return count, state.dtype


def clip(
    model: VideoModel, frames: int, canvas: Canvas, unit: int
) -> tuple[tuple[int, Canvas], list[int]]:
    """The clip output media unit ``unit`` of a ``frames``-frame video at
    ``canvas`` decodes to, ``(frames, raster)``, and the rows of the video's
    complete packed latent its clip input reads, in the order the decoder
    reads them.
    """  # noqa: D205
    decoder = model.decoder
    size = _video(frames, canvas)
    unit_clip = decoder.clips(int(frames))[int(unit)]
    output = decoder.clip_output(size, unit_clip)
    raster = Canvas(int(output.frame.width), int(output.frame.height))
    rows = [int(row) for row in decoder.clip_rows(unit_clip, size)]
    return (int(output.num_frames), raster), rows


def overlap_bytes(model: VideoModel, plan: MediaPlan) -> int:
    """Device bytes of one slot's retained decoder overlap."""
    largest = 0
    for canvas in plan.canvases:
        elements, dtype = overlap(model, canvas)
        largest = max(largest, elements * dtype.itemsize)
    return largest


def _close(graphs, runners) -> None:
    failure: BaseException | None = None
    for graph in list(graphs):
        try:
            graph.release()
        except Exception as error:  # noqa: BLE001 - release every graph
            failure = failure or error
    for runner in list(runners):
        try:
            runner.close(False)
        except Exception as error:  # noqa: BLE001 - close every runner
            failure = failure or error
    if failure is not None:
        raise failure
