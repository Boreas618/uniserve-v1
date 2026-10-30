"""The executor of a video model's components on one rank.

:class:`VideoExecutor` serves every media call a video model's rank
executes: the text encoder and its vision tower, the condition latent
encoders, the denoiser, the video and audio decoders and the host media
(condition reading, encoding and muxing). It keeps what the rank knows of
each admitted request (its size, conditions, layout, media units and MP4
under assembly) from the request's start to its slot's release.

A batch is served in two calls. :meth:`VideoExecutor.prepare` states what
each call reads and writes: which inputs, and every output's shape, region and
storage, whether a host task fills it after the batch commits and how it is
published. The worker leases the reads and binds the writes, then
:meth:`VideoExecutor.execute` runs the calls on the batch's stream with
those tensors, and starts the host tasks of host calls. The trajectory
ledger (which latent bank holds a request's committed step) is the
worker's: a latent preparation or a denoising call names the banks it
reads and writes and its evaluation index.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import Any

import torch

from uniserve import numerics
from uniserve.executor.predicates import split
from uniserve.executor.video.decoder import (
    AudioDecoder,
    TrackUnits,
    UnitShape,
    VideoDecoder,
    overlap_bytes,
    units,
)
from uniserve.executor.video.denoiser import (
    EAGER,
    Conditioning,
    Denoiser,
    keyframe_sets,
)
from uniserve.executor.video.encoder import (
    ConditionEncoders,
    PromptVision,
    TextEncoders,
)
from uniserve.executor.video.media import (
    Conditions,
    MediaThread,
    MuxSession,
    Reports,
    Staging,
    encode_audio,
    encode_unit,
    encoded_unit_bytes,
    read_unit,
    run_reported,
    soundtrack,
)
from uniserve.executor.video.plan import (
    CANVAS_MULTIPLE,
    OUTSIDE_PLAN,
    Admission,
    Canvas,
    ConditionShape,
    Layout,
    MediaPlan,
    SamplePages,
    StateEnvelope,
    VideoModel,
    VideoSize,
    admitted_canvases,
    decode_conditioning,
    vision_spans,
)
from uniserve.runner.denoiser import DenoiserRunner
from uniserve.runner.encoder import EncoderRunner
from uniserve.worker import library

__all__ = ["VideoExecutor", "VideoRequest", "condition_spans"]

# The media calls a host component serves: they run host tasks, so the
# worker hands them to the rank whether or not they are active.
_HOST_CALLS = frozenset(
    ("media_reading", "video_encoding", "audio_encoding", "muxing")
)

# The calls that advance a request's trajectory, the ones the engine
# predicates on their predecessor's completion (``CallKind::advances_state``).
_PREDICATED_CALLS = frozenset(("latent_preparation", "denoising"))

# The most keyframes a request brings: its first and its last frame
# (:func:`~uniserve.executor.video.denoiser.keyframe_sets`).
_KEYFRAMES = 2

# The ``ipc::DType`` name of each torch dtype a product takes.
_DTYPES = {
    "uint8": "u8",
    "int32": "i32",
    "int64": "i64",
    "float16": "f16",
    "bfloat16": "bf16",
    "float32": "f32",
    "int16": "i16",
}


def _key(value: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(value["engine_id"]),
        int(value["request_id"]),
        int(value["request_epoch"]),
    )


def _extent(bound: Mapping[str, Any]) -> int:
    """The most elements a product holds along one dimension."""
    if "static" in bound:
        return int(bound["static"])
    return int(bound["device"]["max"])


def _row_width(product: Mapping[str, Any]) -> int:
    """Rows' width of a two-dimensional product."""
    return _extent(product["shape_bound"]["dims"][1])


@dataclass(frozen=True)
class Placed:
    """One component placed on this rank."""

    name: str
    host: bool
    # ``capability -> path`` of the component's declared calls.
    paths: dict[str, str]
    # The call kinds the component serves.
    kinds: frozenset[str]
    ranks: tuple[int, ...]
    distributed: bool
    units_per_rank: int

    @classmethod
    def of(cls, value: Mapping[str, Any]) -> Placed:
        return cls(
            name=str(value["name"]),
            host=bool(value["host"]),
            paths={
                str(call["capability"]): str(call["path"])
                for call in value["calls"]
            },
            kinds=frozenset(str(kind) for kind in value["kinds"]),
            ranks=tuple(int(rank) for rank in value["ranks"]),
            distributed=bool(value["distributed"]),
            units_per_rank=int(value["units_per_rank"]),
        )

    def units(
        self, dealt: Sequence[int], rank: int, cursor: int, count: int
    ) -> range:
        """This rank's media units of a round (:func:`~.decoder.units`)."""
        return units(
            self.distributed,
            self.ranks,
            self.units_per_rank,
            dealt,
            rank,
            cursor,
            count,
        )


@dataclass
class VideoRequest:
    """One admitted video request on this rank."""

    key: tuple[int, int, int]
    frames: int
    canvas: Canvas
    # The prompt's token ids.
    tokens: list[int]
    conditions: list[dict[str, Any]]
    size: VideoSize
    # Frames of each media unit.
    units: list[int]
    # Samples of the generated audio track.
    audio_samples: int
    seed: int
    # The request's layout on a denoising rank.
    admission: Admission | None = None
    # The request's MP4 on the muxer rank.
    mux: MuxSession | None = None

    @property
    def longest(self) -> int:
        """Frames of the request's longest media unit."""
        return max(self.units)


class VideoExecutor:
    """The media calls of a video model's rank.

    Args:
        loaded: The loaded model.
        config: The rank's deployment: its placed components and the video
            settings (see :meth:`uniserve.executor.Executor`).
    """

    def __init__(self, loaded, config: Mapping[str, Any]) -> None:
        self.loaded = loaded
        self.config = config
        self.rank = int(config["rank"])
        self.graphs = bool(config["graphs"])
        self.attention = str(config["attention"])
        self.dtype: torch.dtype = getattr(
            torch, str(config["model_dtype"]).removeprefix("torch.")
        )
        self.placed = {
            placed.name: placed
            for placed in map(Placed.of, config["components"])
        }
        video = config["video"]
        self.model = VideoModel(loaded, str(video["denoiser"]))
        admitted = admitted_canvases(
            self.model.canvases, video["resolutions"], video["aspect_ratios"]
        )
        self.plan = MediaPlan.new(
            self.model,
            float(video["max_seconds"]),
            None
            if video["min_seconds"] is None
            else float(video["min_seconds"]),
            int(config["max_model_len"]),
            [int(capacity) for capacity in video["text_capacities"]],
            None
            if video["max_condition_rows"] is None
            else int(video["max_condition_rows"]),
            admitted,
            self.graphs,
        )
        self.pages = SamplePages.plan(self.model, self.plan.largest)
        self.denoising = self._component("video_denoiser") is not None
        self.decoding = self._component("video_decoder") is not None
        self.envelope = (
            StateEnvelope.of(
                [self.model.fields(layout) for layout in self.plan.startup()],
                self.pages,
            )
            if self.denoising
            else None
        )
        # Every admitted frame count's units and track, read when the rank
        # starts: a request is admitted without asking the model.
        self.durations = {
            frames: (
                self.model.unit_frames(frames),
                self.model.track_samples(frames),
            )
            for frames in self.plan.frame_counts
        }
        self.requests: dict[int, VideoRequest] = {}
        self.text: TextEncoders | None = None
        self.conditions: ConditionEncoders | None = None
        self.denoiser: Denoiser | None = None
        self.decoder: VideoDecoder | None = None
        self.audio: AudioDecoder | None = None
        self.reader: Conditions | None = None
        self.staging: Staging | None = None
        self.media_thread: MediaThread | None = None
        self.reports: Reports | None = None
        # The layout each admitted slot holds on the denoiser.
        self.held: dict[int, Layout] = {}
        # The manifest's products, read once by startup.
        self._outputs: dict[str, list[dict[str, Any]]] | None = None

    def _component(self, capability: str) -> Placed | None:
        for placed in self.placed.values():
            if capability in placed.paths:
                return placed
        return None

    # The manifest.

    def manifest(self) -> dict[str, Any]:
        """What capacity planning and the handshake read of the rank."""
        model, plan = self.model, self.plan
        outputs = DenoiserRunner.products(
            self.loaded,
            model.component,
            plan.max_frames,
            plan.max_text_tokens,
            plan.condition_rows,
            str(self.dtype).removeprefix("torch."),
            [value for canvas in plan.canvases for value in canvas],
        )
        products = _output_infos(outputs.owned)
        longest = max(model.unit_frames(plan.max_frames))
        row = max(
            encoded_unit_bytes(longest, canvas.height, canvas.width)
            for canvas in plan.canvases
        )
        products.setdefault("video_codec", []).append(
            {
                "name": "encoded_units",
                "dtype": "u8",
                "shape_bound": {
                    "dims": [
                        {
                            "device": {
                                "max": len(model.unit_frames(plan.max_frames))
                            }
                        },
                        {"static": row},
                    ]
                },
                "raster_axes": None,
            }
        )
        if plan.condition_rows > 0:
            reader = DenoiserRunner.reader_products(
                self.loaded,
                model.component,
                plan.max_frames,
                plan.max_text_tokens,
                plan.condition_rows,
            )
            for infos in _output_infos(reader.owned).values():
                products.setdefault("media_reader", []).extend(infos)
        return {
            "outputs": products,
            "denoiser": {
                **model.facts(),
                "canvases": [
                    {
                        "short_edge": short_edge,
                        "aspect_ratio": ratio,
                        "canvas": {
                            "width": canvas.width,
                            "height": canvas.height,
                        },
                    }
                    for short_edge, ratio, canvas in plan.admitted
                ],
            },
            "pages": {
                "page_units": self.pages.page_units,
                "pages": self.pages.pages,
            },
            "state_bytes": (
                0 if self.envelope is None else self.envelope.slot_bytes()
            )
            + (overlap_bytes(model, plan) if self.decoding else 0),
            "stores_latent": self.denoising,
            "latent_region": self._latent_region() if self.denoising else None,
            "encoder_identities": self._encoder_identities(),
            "units": len(model.unit_frames(plan.max_frames)),
            "frame_counts": plan.frame_counts,
        }

    def _encoder_identities(self) -> dict[str, str]:
        """What computes each encoder output this rank writes, by product.

        The declared function and format identity
        (``uniserve.numerics.identity``) of the module that computes it: the
        text encoder for the prompt's conditioning, its vision tower for the
        vision features, the video and audio encoders for the condition
        latents. With the weights' values it keys the output in the host
        encoder pool, so another world computing the same finds it.
        """
        module = self.loaded.module

        def identity(path: str) -> str:
            return numerics.identity(
                module.get_submodule(path) if path else module
            )

        identities = {}
        text = self._component("text_encoder")
        if text is not None:
            path = text.paths["text_encoder"]
            identities["conditioning"] = identity(path)
            if "patch_encoder" in text.paths:
                # The vision tower the text encoder's runner encodes patches
                # with (``TextEncoders``).
                identities["vision_features"] = identity(
                    f"{path}.vision" if path else "vision"
                )
        latent = self._component("video_encoder")
        if latent is not None:
            identities["condition_video_latents"] = identity(
                latent.paths["video_encoder"]
            )
            if "audio_encoder" in latent.paths:
                identities["condition_audio_latents"] = identity(
                    latent.paths["audio_encoder"]
                )
        return identities

    def _latent_region(self) -> str:
        """What determines this rank's share of every request's samples.

        A request's samples on this rank are its rows of each modality in
        the layout the plan admits the request to (:meth:`Denoiser.storage`):
        the plan (its canvases, frame counts, capacities and startup
        layouts), the slot's pages, the rank's coordinates in the denoiser's
        mesh and the denoiser's division of a layout's rows among them,
        which the rows it gives this rank of every startup layout state, fix
        them. A rank of another world with the same digest holds the same
        share of any trajectory, so the engine knows which stored pages it
        reads without its layout. Lowercase hexadecimal SHA-256.
        """
        mesh = self.model.denoiser.mesh
        shares = [
            self.model.fields(layout).samples for layout in self.plan.startup()
        ]
        digest = hashlib.sha256()
        for part in (
            "uniserve-latent-region/2",
            repr(self.plan),
            repr(self.pages),
            repr((mesh.axes, mesh.shape, mesh.coordinate(mesh.rank))),
            repr(shares),
        ):
            encoded = part.encode()
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        return digest.hexdigest()

    # Startup.

    def bind(
        self,
        settings: Mapping[str, Any],
        streams: Mapping[str, torch.cuda.Stream],
        pools: Mapping[str, Any],
        scratches: Mapping[str, Any],
        banks: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> None:
        """Construct the rank's runners.

        ``settings`` holds the ``slots``, the ``reports`` pipe host tasks
        report through (``None`` on a rank without host components), the
        ``ffmpeg`` executable and the denoiser's
        ``transient_layouts``; ``streams`` maps the first call kind of each
        component the rank holds to its stream and ``scratches`` to the
        shared work areas of the contexts serialized on it; ``pools`` each
        component capability that captures to its graph pool; ``banks`` are
        the latent banks of a denoising rank.
        """
        slots = int(settings["slots"])
        ffmpeg = str(settings["ffmpeg"])

        # Only a rank holding a host component runs host tasks, so only it
        # is given a pipe to report them through.
        reports = settings["reports"]
        self.reports = None if reports is None else Reports(int(reports))

        plan, model = self.plan, self.model
        text = self._component("text_encoder")
        if text is not None:
            stream = streams["text_encoding"]
            vision = "patch_encoder" in text.paths
            block_grids = None
            block_rows = None
            if vision and self.graphs and plan.condition_rows > 0:
                block_grids = self._keyframe_grids()
                block_rows = self._vision_rows()
            self.text = TextEncoders(
                loaded=self.loaded,
                path=text.paths["text_encoder"],
                stream=stream,
                attention=self.attention,
                scratch=scratches["text_encoding"],
                capacities=plan.text_capacities,
                max_text_tokens=plan.max_text_tokens,
                row_budget=max(self._vision_row_count(), 1),
                vision=vision,
                dtype=self.dtype,
                pool=pools.get("text_encoder"),
                block_grids=block_grids,
                block_rows=block_rows,
            )
        latent = self._component("video_encoder")
        if latent is not None:
            stream = streams["latent_encoding"]
            self.conditions = ConditionEncoders(
                loaded=self.loaded,
                video_path=latent.paths["video_encoder"],
                audio_path=latent.paths["audio_encoder"],
                stream=stream,
                attention=self.attention,
                scratch=scratches["latent_encoding"],
                pool=pools.get("video_encoder"),
            )
        denoiser = self._component("video_denoiser")
        if denoiser is not None:
            assert self.envelope is not None and banks is not None
            stream = streams["denoising"]
            self.denoiser = Denoiser(
                loaded=self.loaded,
                component=denoiser.name,
                model=model,
                plan=plan,
                envelope=self.envelope,
                pages=self.pages,
                slots=slots,
                stream=stream,
                attention=self.attention,
                scratch=scratches["denoising"],
                conditioning=self._conditioning(denoiser),
                condition_latents=self._condition_latents(),
                pool=pools.get("video_denoiser"),
                transient=int(settings["transient_layouts"]),
                banks=banks,
            )
        decoder = self._component("video_decoder")
        if decoder is not None:
            stream = streams["video_decoding"]
            self.decoder = VideoDecoder(
                loaded=self.loaded,
                path=decoder.paths["video_decoder"],
                model=model,
                plan=plan,
                stream=stream,
                attention=self.attention,
                scratch=scratches["video_decoding"],
                slots=slots,
                pool=pools.get("video_decoder"),
            )
        audio = self._component("audio_decoder")
        if audio is not None:
            stream = streams["audio_decoding"]
            self.audio = AudioDecoder(
                loaded=self.loaded,
                path=audio.paths["audio_decoder"],
                model=model,
                plan=plan,
                stream=stream,
                attention=self.attention,
                scratch=scratches["audio_decoding"],
                pool=pools.get("audio_decoder"),
            )
        if any(placed.host for placed in self.placed.values()):
            self.media_thread = MediaThread("uniserve-media")
            if "media_reader" in self.placed:
                self.reader = Conditions(self.loaded, ffmpeg, model.frame_rate)
                device = self.config.get("device")
                if device is not None and str(device).startswith("cuda"):
                    self.staging = Staging(torch.device(str(device)))

    def _conditioning(self, placed: Placed) -> Conditioning | None:
        """The text conditioner this rank's pipeline stage holds, if any."""
        path = placed.paths.get("encoder")
        if path is None:
            return None
        if not EncoderRunner.holds(self.loaded, path):
            return None
        features = self._output("conditioning")
        if features is None:
            raise ValueError(
                "the text conditioner has no text encoder features"
            )
        dims = features["shape_bound"]["dims"]
        return Conditioning(
            path=path,
            capacities=tuple(self.plan.text_capacities),
            width=_extent(dims[1]),
            dtype=_torch_dtype(features["dtype"]),
        )

    def _condition_latents(self) -> tuple[int, torch.dtype] | None:
        """``(width, dtype)`` of the condition latent rows encodings read.

        Under CUDA graphs the captured condition encodings read them;
        ``None`` without conditions.
        """
        output = self._output("condition_video_latents")
        if output is None or self.plan.condition_rows == 0:
            return None
        return (
            _extent(output["shape_bound"]["dims"][1]),
            _torch_dtype(output["dtype"]),
        )

    def _output(
        self, name: str, owner: str | None = None
    ) -> dict[str, Any] | None:
        """The manifest's product ``name``, of component ``owner`` if named."""
        if self._outputs is None:
            self._outputs = self.manifest()["outputs"]
        for component, outputs in self._outputs.items():
            if owner is not None and component != owner:
                continue
            for output in outputs:
                if output["name"] == name:
                    return output
        return None

    def _vision_rows(self) -> tuple[int, torch.dtype] | None:
        """``(width, dtype)`` of the vision rows the media reader packs."""
        output = self._output("vision_pixels", "media_reader")
        if output is None:
            return None
        return (
            _extent(output["shape_bound"]["dims"][1]),
            _torch_dtype(output["dtype"]),
        )

    def _vision_row_count(self) -> int:
        """The most vision patch rows one request carries."""
        output = self._output("vision_pixels", "media_reader")
        if output is None:
            return 0
        return _extent(output["shape_bound"]["dims"][0])

    def _keyframe_grids(self) -> list[tuple[int, int, int]]:
        """The vision grid of an fl2va keyframe at each admitted canvas.

        The server derives it with the checkpoint's processor configs.
        """
        processor = EncoderRunner.vision_processor(self.loaded)
        if len(processor) != 2:
            raise ValueError(
                "under CUDA graphs the vision tower captures the keyframe "
                "grids its processor configs plan, which the model does not "
                "keep"
            )
        grid = library()["video_image_grid"]
        grids = []
        for _, _, canvas in self.plan.admitted:
            t, h, w = grid(
                processor[0], processor[1], canvas.height, canvas.width
            )
            grids.append((int(t), int(h), int(w)))
        return grids

    def warmup(self) -> None:
        """Prepare, warm and capture every component.

        In the order every rank shares: encoders, the denoiser, the
        decoders.
        """
        plan = self.plan
        if self.text is not None:
            self.text.capture()
        if self.conditions is not None:
            self.conditions.capture_tile()
            output = self._output("condition_video_latents")
            if output is not None and self.graphs:
                placed = self._component("video_encoder")
                assert placed is not None
                # A distributed encoder deals a request's keyframes out at
                # most ``units_per_rank`` to a rank.
                per_call = (
                    min(placed.units_per_rank, _KEYFRAMES)
                    if placed.distributed
                    else _KEYFRAMES
                )
                self.conditions.capture_keyframes(
                    [
                        (canvas.width, canvas.height)
                        for _, _, canvas in plan.admitted
                    ],
                    per_call,
                    _extent(output["shape_bound"]["dims"][1]),
                    _torch_dtype(output["dtype"]),
                )
        denoiser = self.denoiser
        if denoiser is not None:
            # A placeholder request in slot one warms every layout before any
            # capture: kernels, plans and scratch are created by a layout's
            # first step.
            for layout in plan.startup():
                denoiser.prepare_startup(layout)
            for layout in plan.startup():
                size = VideoSize(
                    layout.frames,
                    layout.canvas,
                    min(layout.text_rows, plan.max_text_tokens),
                )
                denoiser.prepare_latents(layout, 1, size, 0)
                denoiser.warm(layout, 1)
            denoiser.warm_conditioner(False)
        decoder = self._component("video_decoder")
        if self.decoder is not None and decoder is not None:
            self.decoder.prepare(plan)
            self.decoder.warm(
                plan,
                decoder.ranks,
                decoder.units_per_rank,
                decoder.distributed,
                self.rank,
            )
        audio = self._component("audio_decoder")
        if self.audio is not None and audio is not None:
            self.audio.prepare(
                plan,
                audio.ranks,
                audio.units_per_rank,
                audio.distributed,
                self.rank,
            )
        if denoiser is not None:
            for layout in (*plan.layouts, *plan.conditioned):
                denoiser.capture(layout)
            # Every keyframe set a request may bring, encoded into each
            # layout of requests with conditions.
            for layout in plan.conditioned:
                for conditions in keyframe_sets(layout.canvas):
                    denoiser.capture_conditions(layout, conditions)
            denoiser.warm_conditioner(True)
            denoiser.stream.synchronize()
            denoiser.seal()
        if self.decoder is not None:
            self.decoder.capture()
        if self.audio is not None:
            self.audio.capture()

    # Requests.

    def start(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> list[tuple[tuple[int, int, int], str]]:
        """Admit started video requests with their exact sizes.

        On a denoising rank each one's layout is selected and held,
        preparing a transient layout the denoiser lacks. Returns the refused
        requests with their reasons.
        """
        refused = []
        for request in requests:
            media, video = request.get("diffusion"), request.get("video")
            if media is None or video is None:
                continue
            key = _key(request["request_key"])
            slot = int(request["request_pool_idx"])
            canvas = Canvas(int(media["width"]), int(media["height"]))
            payload = video["conditioning"]
            conditioning = decode_conditioning(
                bytes(payload) if isinstance(payload, list) else payload
            )
            tokens = [int(token) for token in request["prompt_token_ids"]]
            frames = int(media["num_frames"])
            duration = self.durations.get(frames)
            if duration is None or canvas not in self.plan.canvases:
                refused.append((key, OUTSIDE_PLAN))
                continue
            conditions = list(conditioning["conditions"])
            size = VideoSize(
                frames,
                canvas,
                len(tokens),
                tuple(ConditionShape.of(condition) for condition in conditions),
                vision_spans(conditioning["text_tags"]),
            )
            record = VideoRequest(
                key=key,
                frames=frames,
                canvas=canvas,
                tokens=tokens,
                conditions=conditions,
                size=size,
                units=list(duration[0]),
                audio_samples=int(duration[1]),
                seed=int(media["seed"]),
            )
            if self.denoiser is not None:
                admitted = self.plan.admit(self.model, size)
                if isinstance(admitted, str):
                    refused.append((key, admitted))
                    continue
                self.denoiser.acquire(admitted.layout)
                previous = self.held.pop(slot, None)
                if previous is not None:
                    self.denoiser.release(previous)
                self.held[slot] = admitted.layout
                record.admission = admitted
            self.requests[slot] = record
        return refused

    def release(self, slots: Sequence[int]) -> None:
        """Forget retired slots' requests and their layouts."""
        for slot in slots:
            record = self.requests.pop(int(slot), None)
            if record is not None and record.mux is not None:
                record.mux.close()
            layout = self.held.pop(int(slot), None)
            if layout is not None and self.denoiser is not None:
                self.denoiser.release(layout)

    def _request(self, slot: int, key: Mapping[str, Any]) -> VideoRequest:
        record = self.requests.get(slot)
        if record is None or record.key != _key(key):
            raise ValueError("video call has no admitted media dimensions")
        return record

    # Batches.

    def prepare(self, step: Mapping[str, Any]) -> list[dict[str, Any]]:
        """What each call of a batch reads and writes, in call order.

        Each entry: ``reads`` (input indices), ``writes`` (each output's
        ``output`` index, ``shape``, ``region`` or None, ``storage`` ``host``
        or ``device``, whether a host task fills it (``deferred``) and
        whether host readers read its export (``host_export``)),
        ``publication`` (``products``, ``encoded`` with the written
        ``rows``, ``media`` or None) and, for host calls, ``reports``: how
        many reports its host tasks send through the pipe. A call reading a
        round's decoded rows names the round position of each read
        (``read_rows``); a muxing call reads the framed rows of the exports
        that reach the rank (``exported_rows``).
        """
        result = []
        for entry in step["calls"]:
            if entry["refused"]:
                result.append(_nothing())
                continue
            call = entry["call"]
            kind = call["kind"]
            if not entry["active"] and kind not in _HOST_CALLS:
                result.append(_nothing())
                continue
            handler = getattr(self, f"_prepare_{kind}", None)
            if handler is None:
                raise NotImplementedError(
                    f"{kind} is not a video model's media call"
                )
            result.append(handler(entry, step))
        return result

    def execute(
        self, step: Mapping[str, Any], tensors: Sequence[torch.Tensor]
    ) -> list[int]:
        """Run a batch's calls with the tensors the worker leased and bound.

        The tensors follow the order :meth:`prepare` named them, call by
        call: its reads, then its writes; the calls' ``uint8`` predicates
        follow them all (:mod:`uniserve.executor.predicates`). A predicated
        call runs whatever its predicate holds: when it is false, latent
        preparation keeps the bank it would write and the slot's tables and
        conditioning, and denoising writes the samples it read. An encoder
        call whose output lies in the host encoder pool reads or writes the
        pool's pages after its writes, one ``uint8 [page_bytes]`` host
        tensor per copy its ``host["copies"]`` lists: it reads its output
        from them instead of encoding (``host["import"]``), or writes the
        output it encoded to them. A latent preparation that restores a
        stored trajectory reads the stored pages after its writes, one
        ``uint8 [page_bytes]`` host tensor per page its
        ``latent["restore"]["pages"]`` counts, which its copies read.
        Returns how each call's forward ran.
        """
        tensors, predicates = split(step["calls"], tensors)
        modes = []
        cursor = 0
        for entry, prepared in zip(
            step["calls"], step["prepared"], strict=True
        ):
            host = entry.get("host")
            pages = len(host["copies"]) if host is not None else 0
            restore = (entry.get("latent") or {}).get("restore")
            restored = 0 if restore is None else int(restore["pages"])
            reading = len(prepared["reads"])
            writing = reading + len(prepared["writes"])
            count = writing + pages + restored
            own = list(tensors[cursor : cursor + count])
            cursor += count
            if entry["refused"]:
                modes.append(EAGER)
                continue
            kind = entry["call"]["kind"]
            if not entry["active"] and kind not in _HOST_CALLS:
                modes.append(EAGER)
                continue
            reads = own[:reading]
            writes = own[reading:writing]
            if restore is not None:
                # The stored pages follow the writes; only a latent
                # preparation reads them.
                entry = {**entry, "stored": own[writing + pages :]}
            if host is not None and host["import"]:
                # The output is stored: every rank reads its rows instead of
                # encoding them.
                self._copy_host(
                    kind, writes, own[writing : writing + pages], host, False
                )
                modes.append(EAGER)
                continue
            run = getattr(self, f"_run_{kind}")
            predicate = predicates.get(int(entry["index"]))
            if kind in _PREDICATED_CALLS:
                modes.append(
                    run(entry, step, prepared, reads, writes, predicate)
                )
            elif predicate is not None:
                raise ValueError(f"a {kind} call carries no predicate")
            else:
                modes.append(run(entry, step, prepared, reads, writes))
            if host is not None:
                self._copy_host(
                    kind, writes, own[writing : writing + pages], host, True
                )
        return modes

    def _copy_host(
        self,
        kind: str,
        writes: Sequence[torch.Tensor],
        pages: Sequence[torch.Tensor],
        host: Mapping[str, Any],
        export: bool,
    ) -> None:
        """Copy an encoder call's output rows between its output and the
        host encoder pool's pages, on the stream of the encoder that writes
        the output, which orders them after the encoding.

        ``host["copies"]`` holds, for each page of ``pages``, the byte offset
        in the rank's write (its region of the call's output, or the whole
        output; ``writes[0]`` starts at its first row), the byte offset in
        the page and the bytes copied
        (``crates/worker/src/executor/products.rs::host_copies``).
        ``export`` writes the output to the pages, otherwise the output is
        read from them.
        """  # noqa: D205
        if len(writes) != 1:
            raise ValueError(
                "an encoder call with a host output writes one output"
            )
        runner = (
            self.text
            if kind in ("text_encoding", "vision_encoding")
            else self.conditions
        )
        if runner is None:
            raise ValueError(f"this rank runs no encoder of a {kind} call")
        output = writes[0]
        if not output.is_contiguous():
            raise ValueError(
                "an encoder output with a host output is contiguous"
            )
        flat = output.view(-1).view(torch.uint8)
        with torch.cuda.stream(runner.stream):
            for page, (start, offset, length) in zip(
                pages, host["copies"], strict=True
            ):
                stored = page[offset : offset + length]
                rows = flat[start : start + length]
                if export:
                    stored.copy_(rows, non_blocking=True)
                else:
                    rows.copy_(stored, non_blocking=True)

    # Text encoding.

    def _prepare_text_encoding(self, entry, step) -> dict[str, Any]:
        call = entry["call"]
        if len(call["outputs"]) != 1 or len(call["inputs"]) > 1:
            raise ValueError(
                "text encoding reads at most one vision product and writes "
                "its conditioning"
            )
        record = self._request(entry["slot"], call["request_key"])
        if not record.tokens:
            raise ValueError(
                "text conditioning requires admitted prompt tokens"
            )
        width = _row_width(call["outputs"][0])
        return {
            "reads": list(range(len(call["inputs"]))),
            "writes": [_write(0, [len(record.tokens), width])],
            "publication": None,
        }

    def _run_text_encoding(self, entry, step, prepared, reads, writes) -> int:
        assert self.text is not None
        record = self._request(entry["slot"], entry["call"]["request_key"])
        vision = None
        if reads:
            images: list[tuple[int, int, int]] = []
            videos: list[tuple[int, int, int]] = []
            for condition in record.conditions:
                block = condition.get("vision")
                if block is None:
                    continue
                (videos if "video" in condition["media"] else images).append(
                    _grid(block)
                )
            vision = PromptVision(tuple(images), tuple(videos), reads[0])
        return self.text.encode_text(record.tokens, writes[0], vision)

    # Vision encoding.

    def _prepare_vision_encoding(self, entry, step) -> dict[str, Any]:
        call = entry["call"]
        if len(call["inputs"]) != 1 or len(call["outputs"]) != 1:
            raise ValueError(
                "vision encoding requires one input and one output"
            )
        record = self._request(entry["slot"], call["request_key"])
        tokens = sum(
            int(condition["vision"]["tokens"])
            for condition in record.conditions
            if condition.get("vision") is not None
        )
        if tokens == 0:
            raise ValueError("vision encoding requires a vision block")
        return {
            "reads": [0],
            "writes": [_write(0, [tokens, _row_width(call["outputs"][0])])],
            "publication": None,
        }

    def _run_vision_encoding(self, entry, step, prepared, reads, writes) -> int:
        assert self.text is not None
        record = self._request(entry["slot"], entry["call"]["request_key"])
        grids = [
            _grid(condition["vision"])
            for condition in record.conditions
            if condition.get("vision") is not None
        ]
        return self.text.encode_patches(reads[0], grids, writes[0])

    # Condition encoding.

    def _condition_input(self, entry) -> dict[str, Any] | None:
        """What a condition encoding call encodes on this rank.

        This rank's visual units of a round, or every audio track; ``None``
        for an audio call on a rank that is not the round's first.
        """
        call = entry["call"]
        if len(call["inputs"]) != 1 or len(call["outputs"]) != 1:
            raise ValueError(
                "condition encoding requires one input and one output"
            )
        name = entry["output_names"][0]
        record = self._request(entry["slot"], call["request_key"])
        if not record.conditions:
            raise ValueError("condition encoding requires admitted conditions")
        placed = self._component("video_encoder")
        assert placed is not None
        decode = entry["range"]
        if name == "condition_video_latents":
            # Every visual unit as (condition, unit, rows), in encoding order.
            every = [
                (index, unit, int(rows))
                for index, condition in enumerate(record.conditions)
                for unit, rows in enumerate(condition["latent_units"])
            ]
            cursor, count = int(decode["cursor"]), int(decode["max_units"])
            if count == 0 or cursor + count > len(every):
                raise ValueError(
                    "condition encoding exceeds the admitted units"
                )
            round_units = every[cursor : cursor + count]
            run = placed.units(decode["ranks"], self.rank, 0, count)
            if len(run) == 0:
                raise ValueError("condition encoding assigns this rank no unit")
            rows = sum(unit[2] for unit in round_units)
            first = sum(unit[2] for unit in round_units[: run.start])
            last = sum(unit[2] for unit in round_units[: run.stop])
            offsets, pixels = [], 0
            for condition in record.conditions:
                offsets.append(pixels)
                pixels += _pixel_rows(condition)
            units_list = []
            for condition_index, unit, _ in round_units[run.start : run.stop]:
                condition = record.conditions[condition_index]
                frames, canvas = _pixels(condition)
                window, patch_rows = _unit_rows(condition, unit, frames, canvas)
                units_list.append(
                    (
                        offsets[condition_index],
                        frames,
                        canvas.width,
                        canvas.height,
                        window,
                        patch_rows.start,
                        patch_rows.stop,
                    )
                )
            return {
                "units": units_list,
                "rows": rows,
                "region": [[first, last]],
            }
        if name == "condition_audio_latents":
            # The audio call is one unit, the first of its dealt ranks'.
            if len(placed.units(decode["ranks"], self.rank, 0, 1)) == 0:
                return None
            lengths = _track_samples(record.conditions)
            rows = sum(
                int(condition["audio_rows"]) for condition in record.conditions
            )
            return {"tracks": lengths, "rows": rows, "region": None}
        raise ValueError(f"latent encoding does not produce {name}")

    def _prepare_latent_encoding(self, entry, step) -> dict[str, Any]:
        found = self._condition_input(entry)
        if found is None:
            return _nothing()
        width = _row_width(entry["call"]["outputs"][0])
        region = None
        if found["region"] is not None:
            first, last = found["region"][0]
            region = [[first, last], [0, width]]
        return {
            "reads": [0],
            "writes": [_write(0, [found["rows"], width], region)],
            "publication": None,
            "input": found,
        }

    def _run_latent_encoding(self, entry, step, prepared, reads, writes) -> int:
        assert self.conditions is not None
        found = prepared.get("input")
        if found is None:
            return EAGER
        if "units" in found:
            return self.conditions.encode_units(
                reads[0], [tuple(unit) for unit in found["units"]], writes[0]
            )
        return self.conditions.encode_tracks(
            reads[0], found["tracks"], writes[0]
        )

    # The trajectory.

    def _prepare_latent_preparation(self, entry, step) -> dict[str, Any]:
        call = entry["call"]
        record = self._request(entry["slot"], call["request_key"])
        conditioned = bool(record.conditions)
        inputs = call["inputs"]
        if not inputs or (len(inputs) > 1) != conditioned:
            raise ValueError(
                "video preparation reads its conditioning and its conditions"
            )
        if record.admission is None:
            raise AssertionError("a denoising request has no layout")
        assert self.denoiser is not None
        # A pipeline stage without the conditioner leaves the prompt
        # features unread: its forward receives the refined prompt among the
        # preceding stage's hidden rows.
        reads = list(range(1, len(inputs)))
        if self.denoiser.conditions:
            reads = [0, *reads]
        return {
            "reads": reads,
            "writes": [],
            "publication": None,
            # How this rank's slot pages hold the request's samples, which
            # its completion reports for the engine to key and record a
            # stored trajectory by, and which a restore lays the stored
            # values out in.
            "latent_layout": self.denoiser.storage(record.admission.layout),
        }

    def _run_latent_preparation(
        self, entry, step, prepared, reads, writes, predicate=None
    ) -> int:
        assert self.denoiser is not None
        call = entry["call"]
        slot = int(entry["slot"])
        record = self._request(slot, call["request_key"])
        admission = record.admission
        assert admission is not None
        layout = admission.layout
        denoiser = self.denoiser
        # The request's noise and the host sources its slot is initialized
        # from. A resumed request's samples are then those of the stored
        # step the worker commits it at; its tables and conditioning follow
        # from the request as for step zero.
        denoiser.prepare_latents(layout, slot, admission.size, record.seed)
        bank = int(entry["latent"]["write"])
        denoiser.initialize(layout, slot, bank, predicate)
        restore = entry["latent"].get("restore")
        if restore is not None:
            # The stored values, laid out in this rank's pages by the copies
            # the worker planned from the stored ranks' layouts.
            denoiser.restore(
                slot,
                bank,
                [[int(value) for value in copy] for copy in restore["copies"]],
                entry["stored"],
                predicate,
            )
        mode = EAGER
        conditions = reads
        if denoiser.conditions:
            features = reads[0]
            if features.shape[0] != record.size.tokens:
                raise ValueError(
                    "prompt features disagree with the request's prompt"
                )
            mode = denoiser.condition(layout, slot, features, predicate)
            conditions = reads[1:]
        if record.conditions:
            latents = _condition_latents(record.conditions, conditions)
            mode = max(
                mode,
                denoiser.encode_conditions(
                    layout, slot, admission.size, latents, predicate
                ),
            )
        return mode

    def _prepare_denoising(self, entry, step) -> dict[str, Any]:
        call = entry["call"]
        record = self._request(entry["slot"], call["request_key"])
        if record.admission is None:
            raise AssertionError("a denoising request has no layout")
        assert self.denoiser is not None
        # Only the call that completes denoising declares products: the
        # final samples its step publishes (checked against the step when
        # the call runs).
        writes = []
        layout = record.admission.layout
        rows = self.denoiser.sample_rows(layout)
        for index, name in enumerate(entry["output_names"]):
            modality = next(
                (
                    position
                    for position, each in enumerate(self.model.modalities)
                    if f"{each}_latents" == name
                ),
                None,
            )
            if modality is None:
                raise ValueError(f"the denoiser publishes no {name}")
            sample = rows[modality]
            if sample.stop == sample.start:
                # This rank holds none of the modality's rows.
                continue
            writes.append(
                _write(
                    index,
                    [sample.rows, sample.width],
                    [[sample.start, sample.stop], [0, sample.width]],
                    modality=modality,
                )
            )
        return {"reads": [], "writes": writes, "publication": None}

    def _run_denoising(
        self, entry, step, prepared, reads, writes, predicate=None
    ) -> int:
        assert self.denoiser is not None
        record = self._request(entry["slot"], entry["call"]["request_key"])
        admission = record.admission
        assert admission is not None
        latent = entry["latent"]
        # Each call advances the slot's committed trajectory, the worker's
        # ledger, by the next step of the checkpoint's fixed schedule.
        step_index = int(latent["step"])
        num_steps = self.model.num_steps
        if step_index >= num_steps:
            raise ValueError("denoising continues past the fixed schedule")
        if entry["call"]["outputs"] and step_index + 1 != num_steps:
            raise ValueError(
                "final latent products require completed denoising"
            )
        layout = admission.layout
        mode = self.denoiser.denoise(
            layout,
            int(entry["slot"]),
            int(latent["read"]),
            int(latent["write"]),
            step_index,
            predicate,
        )
        if writes:
            samples = self.denoiser.final_samples(layout)
            with torch.cuda.stream(self.denoiser.stream):
                for write, prepared_write in zip(
                    writes, prepared["writes"], strict=True
                ):
                    sample = samples[prepared_write["modality"]]
                    assert sample is not None
                    write.copy_(sample)
        return mode

    # Decoding.

    def _prepare_video_decoding(self, entry, step) -> dict[str, Any]:
        call = entry["call"]
        record = self._request(entry["slot"], call["request_key"])
        decode = entry["range"]
        total = len(record.units)
        if len(call["inputs"]) != 1 or len(call["outputs"]) != 1:
            raise ValueError(
                "media reconstruction requires one tensor input and output"
            )
        cursor, count = int(decode["cursor"]), int(decode["max_units"])
        if count == 0 or cursor + count > total:
            raise ValueError(
                "media decoder assignment exceeds its media unit range"
            )
        placed = self._component("video_decoder")
        assert placed is not None
        run = placed.units(decode["ranks"], self.rank, cursor, count)
        if len(run) == 0:
            # This rank holds no unit of the round and joins no exchange.
            return _nothing()
        if len(run) != 1:
            raise NotImplementedError(
                "video decoding reconstructs one media unit per rank and round"
            )
        row = run.start - cursor
        shape = [
            count,
            record.longest,
            record.canvas.height,
            record.canvas.width,
            3,
        ]
        region = [[0, extent] for extent in shape]
        region[0] = [row, row + 1]
        return {
            "reads": [0],
            "writes": [_write(0, shape, region, host_export=True)],
            "publication": None,
            "unit": run.start,
        }

    def _run_video_decoding(self, entry, step, prepared, reads, writes) -> int:
        assert self.decoder is not None
        unit = prepared.get("unit")
        if unit is None:
            return EAGER
        record = self._request(entry["slot"], entry["call"]["request_key"])
        decode = entry["range"]
        total = len(record.units)
        clip_output = self.decoder.clip_outputs.get(
            (record.frames, record.canvas, unit)
        )
        if clip_output is None:
            raise ValueError("the rank admits no video with this media unit")
        shape = UnitShape(
            clip_output, unit == 0, unit + 1 == total, int(decode["max_units"])
        )
        return self.decoder.decode(
            int(entry["slot"]),
            reads[0],
            record.frames,
            record.canvas,
            unit,
            shape,
            record.longest,
            record.units[unit],
            writes[0][0],
        )

    def _prepare_audio_decoding(self, entry, step) -> dict[str, Any]:
        call = entry["call"]
        record = self._request(entry["slot"], call["request_key"])
        if len(call["inputs"]) != 1 or len(call["outputs"]) != 1:
            raise ValueError(
                "media reconstruction requires one tensor input and output"
            )
        placed = self._component("audio_decoder")
        assert placed is not None and self.audio is not None
        decode = entry["range"]
        total = (
            len(placed.ranks) * max(placed.units_per_rank, 1)
            if placed.distributed
            else 1
        )
        count = max(min(int(decode["max_units"]), total), 1)
        run = placed.units(decode["ranks"], self.rank, 0, count)
        if len(run) == 0:
            return _nothing()
        latent_frames, spans = self.audio.track(record.audio_samples, count)
        first, last = spans[run.start][0], spans[run.stop - 1][1]
        return {
            "reads": [0],
            "writes": [
                _write(
                    0,
                    [record.audio_samples, 2],
                    [[first, last], [0, 2]],
                    host_export=True,
                )
            ],
            "publication": None,
            "track": [
                record.audio_samples,
                latent_frames,
                run.start,
                len(run),
                count,
                last - first,
            ],
        }

    def _run_audio_decoding(self, entry, step, prepared, reads, writes) -> int:
        assert self.audio is not None
        track = prepared.get("track")
        if track is None:
            return EAGER
        return self.audio.decode(reads[0], TrackUnits(*track), writes[0])

    # Host media.

    def _prepare_media_reading(self, entry, step) -> dict[str, Any]:
        call = entry["call"]
        record = self._request(entry["slot"], call["request_key"])
        if not record.conditions:
            raise ValueError("media reading requires admitted conditions")
        if self.reader is None:
            raise ValueError("media reading requires the condition encoders")
        storage = "host" if self.staging is None else "device"
        writes = []
        for index, name in enumerate(entry["output_names"]):
            if name == "condition_pixels":
                # One RGB24 pixel per row.
                rows = sum(map(_pixel_rows, record.conditions))
                width = 3
            elif name == "condition_samples":
                # One stereo sample per row.
                rows = sum(_track_samples(record.conditions))
                width = 2
            elif name == "vision_pixels":
                # One patch per row.
                rows = sum(
                    prod(_grid(condition["vision"]))
                    for condition in record.conditions
                    if condition.get("vision") is not None
                )
                width = _row_width(call["outputs"][index])
            else:
                raise ValueError(f"media reading does not produce {name}")
            writes.append(
                _write(
                    index,
                    [rows, width],
                    storage=storage,
                    deferred=True,
                    name=name,
                )
            )
        return {
            "reads": [],
            "writes": writes,
            "publication": "products",
            "reports": 1,
        }

    def _run_media_reading(self, entry, step, prepared, reads, writes) -> int:
        assert (
            self.reader is not None
            and self.media_thread is not None
            and self.reports is not None
        )
        record = self._request(entry["slot"], entry["call"]["request_key"])
        products = {
            prepared_write["name"]: write
            for prepared_write, write in zip(
                prepared["writes"], writes, strict=True
            )
        }
        reader, staging = self.reader, self.staging
        conditions = record.conditions

        def work() -> tuple[int, bytes]:
            if staging is None:
                reader.read(conditions, products)
            else:
                # The reader decodes into page-locked staging of the
                # products' shapes, copied to them on the reader's stream.
                host = staging.host(products)
                reader.read(conditions, host)
                staging.upload(host, products)
            return 0, b""

        self.media_thread.submit(
            run_reported(
                self.reports,
                int(step["batch_id"]),
                int(entry["index"]),
                0,
                work,
            )
        )
        return EAGER

    def _prepare_video_encoding(self, entry, step) -> dict[str, Any]:
        call = entry["call"]
        decode = entry["range"]
        if len(call["inputs"]) != 1 or len(call["outputs"]) != 1:
            raise ValueError(
                "video encoding requires an input round and one output"
            )
        placed = self.placed.get(call["component"])
        if placed is None:
            raise ValueError("video encoding has no codec component")
        run = placed.units(
            decode["ranks"], self.rank, 0, int(decode["max_units"])
        )
        if len(run) == 0:
            return _nothing()
        row_bytes = _row_width(call["outputs"][0])
        # One read per unit: the unit's row of the round's decoded frames,
        # read in place from its shared-memory export or leased.
        return {
            "reads": [0 for _ in run],
            "read_rows": list(run),
            "writes": [
                _write(
                    0,
                    [int(decode["max_units"]), row_bytes],
                    [[run.start, run.stop], [0, row_bytes]],
                    storage="host",
                    deferred=True,
                )
            ],
            "publication": "encoded",
            "rows": [position - run.start for position in run],
            "reports": len(run),
        }

    def _run_video_encoding(self, entry, step, prepared, reads, writes) -> int:
        """Encode this rank's media units of a round into its encoded rows.

        The rows are this rank's of the round's encoded product. ``reads``
        hold one host tensor of each of the rank's units' frames, in unit
        order.
        """
        assert self.media_thread is not None and self.reports is not None
        if not prepared["writes"]:
            return EAGER
        record = self._request(entry["slot"], entry["call"]["request_key"])
        decode = entry["range"]
        rows = writes[0]
        batch, call = int(step["batch_id"]), int(entry["index"])
        frame_rate = self.model.frame_rate
        canvas = record.canvas
        for local, position in enumerate(prepared["read_rows"]):
            unit = int(decode["cursor"]) + position
            if unit >= len(record.units):
                raise ValueError(
                    "video encoding names a unit outside its video"
                )
            count = record.units[unit]
            frames = (
                reads[local]
                .reshape(-1)[: count * canvas.height * canvas.width * 3]
                .view(count, canvas.height, canvas.width, 3)
            )
            row = rows[local]

            def work(frames=frames, row=row) -> tuple[int, bytes]:
                return encode_unit(frames, frame_rate, row), b""

            self.media_thread.submit(
                run_reported(self.reports, batch, call, local, work)
            )
        return EAGER

    def _prepare_audio_encoding(self, entry, step) -> dict[str, Any]:
        if len(entry["call"]["inputs"]) != 1:
            raise ValueError("audio encoding requires one PCM input")
        return {"reads": [0], "writes": [], "publication": None, "reports": 1}

    def _run_audio_encoding(self, entry, step, prepared, reads, writes) -> int:
        assert self.media_thread is not None and self.reports is not None
        session = self._mux(entry)
        pcm = reads[0]
        rate = self.model.sample_rate

        def work() -> tuple[int, bytes]:
            session.set_audio(encode_audio(pcm, session.samples, rate))
            return 0, b""

        self.media_thread.submit(
            run_reported(
                self.reports,
                int(step["batch_id"]),
                int(entry["index"]),
                0,
                work,
            )
        )
        return EAGER

    def _prepare_muxing(self, entry, step) -> dict[str, Any]:
        # Each input round's framed rows arrive from the exports that reach
        # this rank, as ``[rows, width]`` host bytes.
        finalize = not entry["call"]["inputs"]
        return {
            "reads": list(range(len(entry["call"]["inputs"]))),
            "exported_rows": True,
            "writes": [],
            "publication": "media" if finalize else None,
            "reports": 1,
        }

    def _run_muxing(self, entry, step, prepared, reads, writes) -> int:
        """Append a request's encoded rounds to its MP4, or finish it.

        ``reads`` hold each input round's framed rows as ``[rows, width]``
        uint8 host tensors.
        """
        assert self.media_thread is not None and self.reports is not None
        session = self._mux(entry)
        batch, call = int(step["batch_id"]), int(entry["index"])
        if not reads:

            def finish() -> tuple[int, bytes]:
                data = session.finish()
                return len(data), data

            self.media_thread.submit(
                run_reported(self.reports, batch, call, 0, finish)
            )
            return EAGER
        units = [
            read_unit(memoryview(row.numpy())) for rows in reads for row in rows
        ]

        def append() -> tuple[int, bytes]:
            session.append(units)
            return 0, b""

        self.media_thread.submit(
            run_reported(self.reports, batch, call, 0, append)
        )
        return EAGER

    def _mux(self, entry) -> MuxSession:
        """The MP4 session of a request on the muxer rank, made on first use."""
        record = self._request(entry["slot"], entry["call"]["request_key"])
        if record.mux is None:
            record.mux = MuxSession(
                record.canvas.width,
                record.canvas.height,
                self.model.frame_rate,
                self.model.sample_rate,
                record.units,
            )
        return record.mux

    # Reporting.

    def kernels(self) -> list[tuple[str, list[str]]]:
        """The kernels each component's operators selected."""
        kernels = []
        for placed, owner in (
            (self._component("text_encoder"), self.text),
            (self._component("video_denoiser"), self.denoiser),
            (self._component("video_decoder"), self.decoder),
            (self._component("audio_decoder"), self.audio),
            (self._component("video_encoder"), self.conditions),
        ):
            if placed is not None and owner is not None:
                kernels.append((placed.name, owner.kernels()))
        return kernels

    def close(self, aborted: bool) -> None:
        """Release every runner; the caller drained the streams.

        Every owner is closed before the first failure is raised.
        """
        failure: BaseException | None = None
        if self.media_thread is not None and not aborted:
            self.media_thread.close()
        for owner in (
            self.text,
            self.conditions,
            self.denoiser,
            self.decoder,
            self.audio,
        ):
            if owner is None:
                continue
            try:
                owner.close(aborted)
            except Exception as error:
                failure = failure or error
        for record in self.requests.values():
            if record.mux is not None:
                record.mux.close()
        self.requests.clear()
        if failure is not None:
            raise failure


def _nothing() -> dict[str, Any]:
    return {"reads": [], "writes": [], "publication": None}


def _write(
    output: int,
    shape: Sequence[int],
    region: Sequence[Sequence[int]] | None = None,
    *,
    storage: str = "device",
    deferred: bool = False,
    host_export: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "output": output,
        "shape": [int(extent) for extent in shape],
        "region": None
        if region is None
        else [[int(a), int(b)] for a, b in region],
        "storage": storage,
        "deferred": deferred,
        "host_export": host_export,
        **extra,
    }


def _torch_dtype(name: str) -> torch.dtype:
    for torch_name, protocol in _DTYPES.items():
        if protocol == name:
            return getattr(torch, torch_name)
    raise ValueError(f"unknown protocol dtype {name!r}")


def _output_infos(owned) -> dict[str, list[dict[str, Any]]]:
    """``OutputInfo`` records of the products components publish."""
    result: dict[str, list[dict[str, Any]]] = {}
    for owner, output in owned:
        dtype = _DTYPES.get(str(output["dtype"]))
        if dtype is None:
            raise ValueError(
                f"result {owner}.{output['name']} has no protocol dtype"
            )
        variable = output["variable_axes"][0] if output["variable_axes"] else -1
        dims = [
            {"device": {"max": int(extent)}}
            if axis == variable
            else {"static": int(extent)}
            for axis, extent in enumerate(output["shape"])
        ]
        raster = output["raster_axes"]
        result.setdefault(owner, []).append(
            {
                "name": str(output["name"]),
                "dtype": dtype,
                "shape_bound": {"dims": dims},
                "raster_axes": None
                if raster is None
                else {"height": int(raster[0]), "width": int(raster[1])},
            }
        )
    return result


def _pixels(condition: Mapping[str, Any]) -> tuple[int, Canvas]:
    media = condition["media"]
    if "image" in media:
        size = media["image"]["size"]
        return 1, Canvas(int(size["width"]), int(size["height"]))
    if "video" in media:
        clip = media["video"]["clip"]
        return int(clip["vae_frames"]), Canvas(
            int(clip["canvas"]["width"]), int(clip["canvas"]["height"])
        )
    return 0, Canvas(0, 0)


def _pixel_rows(condition: Mapping[str, Any]) -> int:
    """A condition's rows of ``condition_pixels``: its encoded pixels."""
    frames, canvas = _pixels(condition)
    return frames * canvas.width * canvas.height


def _track_samples(conditions: Sequence[Mapping[str, Any]]) -> list[int]:
    """Samples of each condition's audio track, of those carrying one."""
    tracks = (soundtrack(condition["media"]) for condition in conditions)
    return [int(track["samples"]) for track in tracks if track is not None]


def _grid(vision: Mapping[str, Any]) -> tuple[int, int, int]:
    """A vision block's ``(time, height, width)`` patch grid."""
    grid = vision["grid"]
    return int(grid["t"]), int(grid["h"]), int(grid["w"])


def _unit_rows(
    condition, unit: int, frames: int, canvas: Canvas
) -> tuple[int, range]:
    """The encoder window and the latent patch rows a condition unit encodes.

    Unit ``unit`` encodes rows of 32-pixel latent patches. A still frame
    the request splits into several units is encoded in bands of its patch
    rows, each band reading the whole frame: a band's first row follows the
    rows of the bands before it. Any other unit is its encoder window, over
    every patch row.
    """
    patch_rows = canvas.height // CANVAS_MULTIPLE
    latent_units = [int(rows) for rows in condition["latent_units"]]
    if frames != 1 or len(latent_units) < 2:
        return unit, range(0, patch_rows)
    per_row = canvas.width // CANVAS_MULTIPLE
    before = sum(latent_units[:unit])
    rows = latent_units[unit]
    if per_row == 0 or before % per_row or rows % per_row:
        raise ValueError("an image band is not whole rows of latent patches")
    first = before // per_row
    return 0, range(first, first + rows // per_row)


def condition_spans(
    conditions: Sequence[tuple[int, int]],
    rounds: Sequence[int],
    tracks: int | None,
) -> list[list[tuple[int, int, int]]]:
    """Locate every condition latent in latent preparation's condition reads.

    ``conditions`` are each condition's ``(visual rows, audio rows)`` in
    request order. The reads are the visual rounds' rows in round order
    (``rounds`` holds each round's row count), then, when any condition
    carries audio, one read of every track's rows (``tracks``). A round ends
    wherever its units do, so a condition's rows may continue into the next
    round. Returns the latents in request order, a condition's visual rows
    before its audio rows, each as ``(read, first row, rows)`` spans.

    Raises:
        ValueError: The reads hold other rows than the conditions take.
    """
    visual = sum(rows for rows, _ in conditions)
    audio = sum(rows for _, rows in conditions)
    if sum(rounds) != visual or (tracks or 0) != audio:
        raise ValueError(
            "condition latents disagree with the request's condition rows"
        )
    track_read = len(rounds)
    result: list[list[tuple[int, int, int]]] = []
    round_index, row, audio_row = 0, 0, 0
    for visual_rows, audio_rows in conditions:
        if visual_rows > 0:
            spans = []
            needed = visual_rows
            while needed > 0:
                if round_index >= len(rounds):
                    raise ValueError(
                        "condition latents disagree with the request's "
                        "condition rows"
                    )
                available = rounds[round_index] - row
                taken = min(available, needed)
                if taken > 0:
                    spans.append((round_index, row, taken))
                needed -= taken
                row += taken
                if taken == available:
                    round_index += 1
                    row = 0
            result.append(spans)
        if audio_rows > 0:
            result.append([(track_read, audio_row, audio_rows)])
            audio_row += audio_rows
    return result


def _condition_latents(
    conditions: Sequence[Mapping[str, Any]], reads: Sequence[torch.Tensor]
) -> list[list[torch.Tensor]]:
    """The condition latents of latent preparation's condition reads.

    Each latent is a list of row tensors that together hold its rows.
    """
    audio = any(int(condition["audio_rows"]) > 0 for condition in conditions)
    if audio:
        if not reads:
            raise ValueError("condition latents lack their audio rows")
        rounds, tracks = list(reads[:-1]), reads[-1]
    else:
        rounds, tracks = list(reads), None
    condition_rows = [
        (
            sum(int(rows) for rows in condition["latent_units"]),
            int(condition["audio_rows"]),
        )
        for condition in conditions
    ]
    spans = condition_spans(
        condition_rows,
        [read.shape[0] for read in rounds],
        None if tracks is None else tracks.shape[0],
    )
    every = [*rounds, *([] if tracks is None else [tracks])]
    return [
        [every[read][first : first + count] for read, first, count in latent]
        for latent in spans
    ]
