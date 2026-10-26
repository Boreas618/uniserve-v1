"""The image half of an executor: image models' media work, by computation.

An image model (BAGEL, SenseNova, and any model with a vision tower that is
not a video model) serves its media calls on prepared computations of the
rank's executor: the worker asks first what each call of a batch reads and
writes (:meth:`ImageExecutor.prepare`), reserves it, and then has the
executor compute the batch on what it reserved
(:meth:`ImageExecutor.execute`). :class:`ImageExecutor` serves one
computation per kind of work a placed component holds:

- a *vision* computation encodes input images (and generated images fed
  back) into the features a prompt's vision block places (:mod:`.vision`);
- a *latent* computation encodes images into the latents a prompt's latent
  block conditions on (:mod:`.codec`);
- the *denoiser* computation opens and steps image trajectories in the
  worker's latent ledger and writes latent conditioning for the text
  computations (:mod:`.denoiser`); it attends to the prefix cache, so it
  runs on the device's primary stream, and decodes the finished images when
  the same component holds the image decoder;
- a *decoder* computation decodes images of a component without a denoiser.

The host-side decisions live in the modules beside: :mod:`.layout` (an
image denoiser's latent layout), :mod:`.trajectory` (trajectories, guidance
plans and prefixes), :mod:`.facts` (vision blocks of a text prompt),
:mod:`.rows` (packed denoiser rows), :mod:`.shapes` (graph buckets) and
:mod:`.plan` (what each call reads and writes).

The text computations ask the image half about vision blocks and latent
conditioning (:meth:`ImageExecutor.facts`, :meth:`ImageExecutor.denoiser`),
and the denoiser prefills guidance prefixes on a text computation
(:meth:`ImageExecutor.attach_text`).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from uniserve.errors import SetupError
from uniserve.executor.text.protocol import GraphMode
from uniserve.media.pixels import PreparedImage, Tower, prepare_tensor_image
from uniserve.models import ImageDecoder, ImageDenoiser, PatchEncoder
from uniserve.nn.vae import PatchAutoencoder
from uniserve.processing import BranchSource
from uniserve.runner.context import held_module
from uniserve.runner.encoder import EncoderRunner

from .codec import LatentEncoder, PixelDecoder
from .denoiser import (
    CacheTables,
    Denoiser,
    LatentBanks,
    LatentBinding,
    StepCall,
    framed_prefix,
)
from .facts import ImageFacts
from .layout import ImageLayout
from .plan import (
    NOTHING,
    PlanEntry,
    decoding_entry,
    latent_entry,
    range_interval,
    range_name,
    vision_entry,
)
from .shapes import (
    RowLimits,
    RowShapes,
    condition_rows,
    denoise_rows,
    digest,
    row_shapes,
    token_sizes,
)
from .trajectory import ImageParams
from .vision import VisionEncoder, VisionImage, features

__all__ = [
    "ImageBinding",
    "ImageExecutor",
    "StreamBinding",
    "text_settings",
]

# Each computation's role: the kind of work it serves.
VISION = "vision"
LATENT = "latent"
DENOISER = "denoiser"
DECODER = "decoder"

# The number each graph key's family stands for in the digests.
_FAMILIES = {"vision": 1, "latent": 2, "step": 3, "condition": 4, "decode": 5}

# Latent rows of the largest latent page. A page of the ledger holds one KV
# block of latent rows, and the worker resolves the KV block size to at
# most 64 tokens, after it read the manifest that this bounds.
_MAX_PAGE_UNITS = 64

# Pixels of the largest image a computation bounds when neither the
# configured image sizes nor an image denoiser's longest sequence bound it.
_DEFAULT_MAX_PIXELS = 1024 * 1024


def text_settings(loaded) -> str:
    """JSON of what the text executor's settings read of a model's images.

    ``{"image_processor": bool, "media_axes": 1 | 3 | null}``: whether the
    model declares an image policy, and the rotary axes of the vision rows a
    prompt places (three for a model whose rows carry spatial positions,
    ``null`` without feature injection).
    """
    facts = ImageFacts.from_loaded(loaded)
    axes = None
    if facts.injection is not None:
        axes = 3 if facts.spatial_positions() else 1
    return json.dumps(
        {
            "image_processor": loaded.image_processor is not None,
            "media_axes": axes,
        }
    )


def _dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, str(name).removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown dtype {name!r}")
    return dtype


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _key(value: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(value["engine_id"]),
        int(value["request_id"]),
        int(value["request_epoch"]),
    )


def _device(stream: torch.cuda.Stream | None) -> int:
    """The CUDA ordinal a computation's stream belongs to, -1 on the host."""
    return -1 if stream is None else int(stream.device.index)


def _whole_grid(module: PatchEncoder, shape: Sequence[int]) -> tuple[int, int]:
    """The patch grid tiling whole ``[..., height, width]`` pixels.

    A tower that takes whole images rather than patch rows reports no grid
    of its own; its patches tile the pixels.
    """
    side = module.patch_size
    return int(shape[-2]) // side, int(shape[-1]) // side


@dataclass(frozen=True, slots=True)
class StreamBinding:
    """The storage the worker binds one image computation to.

    Attributes:
        stream: The computation's stream, ``None`` on the host.
        scratch: The stream's shared work areas.
        pool: The stream's graph pool, ``None`` without graphs.
        peers: ``(ordinal, pool)`` of the stream's graph pool on each other
            device of the rank (a generation tower).
        collectives: The process group of each multi-member communicator on
            the stream, by communicator name; ``None`` for none.
    """

    stream: torch.cuda.Stream | None
    scratch: Any
    pool: Any | None = None
    peers: tuple[tuple[int, Any], ...] = ()
    collectives: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ImageBinding:
    """What capacity planning granted the image computations.

    Attributes:
        slots: The request slots.
        computations: Each image computation's storage, in manifest order.
        unit_pool: The prefix cache's K/V pool object, ``None`` without
            one.
        tables: The worker's block tables by name.
        banks: The latent ledger's two banks, ``None`` without one.
    """

    slots: int
    computations: tuple[StreamBinding, ...]
    unit_pool: Any | None = None
    tables: Mapping[str, torch.Tensor] | None = None
    banks: tuple[torch.Tensor, torch.Tensor] | None = None


@dataclass(slots=True)
class _Request:
    """An admitted request's image facts.

    Attributes:
        key: Its identity.
        params: Its image parameters, ``None`` for a request generating no
            image.
        negative: Its admitted negative tokens.
        draws: Its latent encodings recorded so far: the posterior draw
            index of its next one.
    """

    key: tuple[int, int, int]
    params: ImageParams | None
    negative: tuple[int, ...]
    draws: int = 0

    @property
    def seed(self) -> int:
        """The seed its draws are keyed by; 0 without image parameters."""
        return 0 if self.params is None else self.params.draw_seed


@dataclass(slots=True)
class _Computation:
    """One image computation: its component, kinds and modules.

    The module of its role, and a denoiser computation's decoder when its
    component decodes, are constructed at binding; the others stay
    ``None``.

    Attributes:
        role: ``vision``, ``latent``, ``denoiser`` or ``decoder``.
        component: The component it serves.
        kinds: The call kinds it serves.
        paths: The module path of each capability it computes with.
        resources: The worker's resources it uses.
        vision: A vision computation's patch tower.
        latent: A latent computation's latent encoder.
        denoiser: A denoiser computation's denoiser.
        decoder: The image decoder of a computation that decodes.
        rows: A decoding computation's buffer of one trajectory's committed
            latent rows, gathered from the ledger's banks.
        banks: The latent ledger's banks a decoding computation gathers from.
        stream: The computation's stream, ``None`` on the host.
    """

    role: str
    component: str
    kinds: list[str]
    paths: dict[str, str]
    resources: list[str]
    vision: VisionEncoder | None = None
    latent: LatentEncoder | None = None
    denoiser: Denoiser | None = None
    decoder: PixelDecoder | None = None
    rows: torch.Tensor | None = None
    banks: LatentBanks | None = None
    stream: torch.cuda.Stream | None = None

    def label(self, capability: str) -> str:
        path = self.paths[capability]
        return f"{self.component}.{path}" if path else self.component

    def modules(self) -> list[Any]:
        """Its constructed modules, in construction order."""
        return [
            module
            for module in (
                self.vision,
                self.latent,
                self.denoiser,
                self.decoder,
            )
            if module is not None
        ]


def _computations(
    components: Sequence[Mapping[str, Any]],
) -> list[_Computation]:
    """The image computations of the rank's placed device components."""
    computations = []
    for component in components:
        if component["host"]:
            continue
        name = str(component["name"])
        kinds = set(component["kinds"])
        paths = {
            str(call["capability"]): str(call["path"])
            for call in component["calls"]
        }
        if "patch_encoder" in paths and "vision_encoding" in kinds:
            computations.append(
                _Computation(
                    VISION,
                    name,
                    ["vision_encoding"],
                    {"patch_encoder": paths["patch_encoder"]},
                    ["images"],
                )
            )
        if "patch_autoencoder" in paths and "latent_encoding" in kinds:
            computations.append(
                _Computation(
                    LATENT,
                    name,
                    ["latent_encoding"],
                    {"patch_autoencoder": paths["patch_autoencoder"]},
                    ["images"],
                )
            )
        decodes = "image_decoder" in paths and "image_decoding" in kinds
        stepping = [
            kind
            for kind in ("latent_preparation", "denoising")
            if kind in kinds
        ]
        if "image_denoiser" in paths and stepping:
            computation = _Computation(
                DENOISER,
                name,
                stepping,
                {"image_denoiser": paths["image_denoiser"]},
                ["pages", "latents"],
            )
            if decodes:
                computation.kinds.append("image_decoding")
                computation.paths["image_decoder"] = paths["image_decoder"]
                computation.resources.append("images")
            computations.append(computation)
        elif decodes:
            computations.append(
                _Computation(
                    DECODER,
                    name,
                    ["image_decoding"],
                    {"image_decoder": paths["image_decoder"]},
                    ["latents", "images"],
                )
            )
    return computations


class ImageExecutor:
    """The image computations of one rank.

    Args:
        loaded: The loaded model.
        settings: The rank's settings: ``graphs``, ``model_dtype``, the
            placed ``components`` and the ``image`` section
            (``max_batch_calls``, ``max_batch_tokens``,
            ``max_request_slots``, ``max_model_len``, ``flow_shapes``, the
            configured ``[height, width]`` image sizes the codecs capture).
    """

    def __init__(self, loaded, settings: Mapping[str, Any]) -> None:
        self.loaded = loaded
        section = settings["image"]
        self.graphs = bool(settings["graphs"])
        self.attention = loaded.attention
        self.dtype = _dtype(settings["model_dtype"])
        self.max_slots = max(int(section["max_request_slots"]), 1)
        # A batch's calls are distinct requests, so the slots bound them.
        self.max_calls = max(
            min(int(section["max_batch_calls"]), self.max_slots), 1
        )
        self.max_batch_tokens = int(section["max_batch_tokens"])
        self.max_model_len = int(section["max_model_len"])
        self.flow_shapes = [
            (int(height), int(width))
            for height, width in section["flow_shapes"]
        ]
        self.image_facts = ImageFacts.from_loaded(loaded)
        self.layout: ImageLayout | None = self.image_facts.layout
        self.placed = _computations(settings["components"])
        # The dtype the worker's host preparation writes pixels in.
        self.pixel_dtype = _dtype(EncoderRunner.pixel_dtype(loaded))
        self.requests: dict[int, _Request] = {}
        self.slots = self.max_slots
        self._phases: list[tuple[_Computation, str, Any]] = []
        # Each component's graph keys in capture order.
        self._keys: dict[str, list[Any]] = {}
        # Each call kind's preparation (:meth:`prepare`).
        self._preparations: dict[
            str, Callable[[_Computation, Mapping[str, Any]], PlanEntry]
        ] = {
            "vision_encoding": self._prepare_vision,
            "latent_encoding": self._prepare_latent,
            "latent_preparation": self._prepare_trajectory,
            "denoising": self._prepare_trajectory,
            "image_decoding": self._prepare_decoding,
        }

    def _module(self, computation: _Computation, capability: str, kind: type):
        module = held_module(self.loaded.module, computation.paths[capability])
        if not isinstance(module, kind):
            raise SetupError(
                f"{computation.label(capability)} is not a {kind.__name__} on "
                "this rank"
            )
        return module

    # The manifest.

    def computations(self) -> list[dict[str, Any]]:
        """Each computation's manifest entry, in index order."""
        entries = []
        for computation in self.placed:
            entry: dict[str, Any] = {
                "component": computation.component,
                "kinds": list(computation.kinds),
                "prepares": True,
                "commits": False,
                "resources": list(computation.resources),
            }
            if "image_decoding" in computation.kinds:
                # Each call's decoded bytes beside its completion word.
                pixels = 3 * self._max_pixels()
                entry["output_words"] = self.max_calls * (1 + -(-pixels // 8))
            entries.append(entry)
        return entries

    def _max_pixels(self) -> int:
        """Pixels of the largest image a computation encodes or decodes."""
        sizes = [height * width for height, width in self.flow_shapes]
        if self.layout is not None:
            side = self.layout.downsample
            sizes.append(self.layout.max_sequence_tokens * side * side)
        return max(sizes, default=_DEFAULT_MAX_PIXELS)

    def _step_limits(self) -> RowLimits:
        layout = self.layout
        assert layout is not None
        branches = layout.guidance_branches
        return RowLimits(
            min_row_tokens=1 + layout.framing_tokens,
            max_row_tokens=layout.max_row_tokens,
            max_rows=self.max_calls * branches,
            max_tokens=layout.max_row_tokens * branches,
        )

    def _condition_limits(self) -> RowLimits | None:
        """The condition writes a text batch's latent images take.

        ``None`` for a denoiser that does not frame images, whose latent
        images no prompt inserts, or whose shortest framed image exceeds a
        text batch's tokens.
        """
        layout = self.layout
        if layout is None or layout.framing_tokens != 2:
            return None
        longest = min(layout.max_row_tokens, self.max_batch_tokens)
        if longest < 1 + layout.framing_tokens:
            return None
        return RowLimits(
            min_row_tokens=1 + layout.framing_tokens,
            max_row_tokens=longest,
            max_rows=self.max_calls,
            max_tokens=max(self.max_batch_tokens, longest),
        )

    def _row_shapes(
        self, limits: RowLimits, rows: Sequence[int], multiple: int
    ) -> RowShapes:
        return row_shapes(
            token_sizes(
                limits.max_tokens,
                multiple=multiple,
                row_tokens=limits.max_row_tokens,
            ),
            rows,
            limits,
        )

    def manifest(self) -> dict[str, Any]:
        """What capacity planning reads of the image computations.

        ``latent``: the latent rows and width of the longest trajectory and
        their dtype, which size the latent ledger (``null`` without a
        denoiser); the largest vision and latent feature products; and the
        device bytes the computations allocate beside the pools,
        ``fixed_bytes`` plus ``slot_bytes`` per request slot.
        """
        layout = self.layout
        fixed = 0
        latent = None
        latent_bytes = 0
        vision_bytes = 0
        for computation in self.placed:
            if computation.role == VISION:
                product, buffers = self._vision_bytes(computation)
                vision_bytes = max(vision_bytes, product)
                fixed += buffers
            elif computation.role == LATENT:
                product, buffers = self._latent_bytes(computation)
                latent_bytes = max(latent_bytes, product)
                fixed += buffers
            elif computation.role == DENOISER and layout is not None:
                latent = {
                    "rows": layout.max_sequence_tokens,
                    "width": layout.width,
                    "dtype": _dtype_name(self.dtype),
                }
                fixed += self._denoiser_bytes(layout)
            if "image_decoding" in computation.kinds:
                fixed += self._decoder_bytes()
        return {
            "latent": latent,
            "vision_feature_bytes": int(vision_bytes),
            "latent_feature_bytes": int(latent_bytes),
            "fixed_bytes": int(fixed),
            # Each slot's reference time of the solver update.
            "slot_bytes": 4 if latent is not None else 0,
        }

    def _vision_bytes(self, computation: _Computation) -> tuple[int, int]:
        """A vision computation's largest feature product and buffer bytes.

        The buffers hold every slot's patch rows of the largest image the
        image policy admits, and their features.
        """
        module = self._module(computation, "patch_encoder", PatchEncoder)
        patches = int(
            EncoderRunner.image_patches(
                self.loaded, computation.paths["patch_encoder"]
            )
        )
        width, dtype = features(module)
        feature = width * dtype.itemsize
        unit = module.downsample**2
        capacity = patches * (module.max_slots or self.max_calls)
        pixels = capacity * 3 * module.patch_size**2 * self.pixel_dtype.itemsize
        return patches // unit * feature, pixels + capacity // unit * feature

    def _latent_bytes(self, computation: _Computation) -> tuple[int, int]:
        """A latent computation's largest feature product and buffer bytes.

        The buffers hold one image's pixels and posterior noise, each at
        most a float32 per pixel value, and its latent rows.
        """
        module = self._module(
            computation, "patch_autoencoder", PatchAutoencoder
        )
        width = module.patch_size**2 * module.latent_channels
        rows = (
            self.layout.max_sequence_tokens
            if self.layout is not None
            else self._max_pixels() // module.downsample**2
        )
        product = rows * width * module.latent_dtype.itemsize
        pixels = 3 * self._max_pixels()
        return product, 2 * pixels * torch.float32.itemsize + product

    def _denoiser_bytes(self, layout: ImageLayout) -> int:
        """The bytes of the denoiser's buffers beside the ledger.

        The step workspace (every stepping image's committed latent, each at
        most a page past its rows) with the predictions, the initial noise
        draw of the longest image, at most a float32 per value, and the
        latent rows condition writes read.
        """
        itemsize = self.dtype.itemsize
        step = self._step_limits()
        tokens = step.max_tokens + step.max_rows * _MAX_PAGE_UNITS
        total = (
            tokens
            * layout.width
            * (itemsize + layout.prediction_dtype.itemsize)
        )
        total += (
            layout.max_sequence_tokens * layout.width * torch.float32.itemsize
        )
        conditions = self._condition_limits()
        if conditions is not None:
            total += conditions.max_tokens * layout.width * itemsize
        return total

    def _decoder_bytes(self) -> int:
        """The bytes of a decoding computation's buffers.

        The decoded values (float32) and quantized bytes of the largest
        image, and with a denoiser's layout the gathered and captured
        latent rows of its longest trajectory.
        """
        pixels = 3 * self._max_pixels()
        total = pixels * (torch.float32.itemsize + torch.uint8.itemsize)
        layout = self.layout
        if layout is not None:
            rows = layout.max_sequence_tokens + _MAX_PAGE_UNITS
            total += rows * layout.width * self.dtype.itemsize * 2
        return total

    # Binding.

    def bind_from(self, binding, computations: Sequence[int]) -> dict[str, Any]:
        """Bind on the executor's binding (``uniserve.executor._Binding``).

        ``computations`` are the global indices of this executor's
        computations, in the order :meth:`computations` stated them. The
        binding's ``peer_pools`` holds each stream's ``(ordinal, pool)`` on
        the rank's other devices, when it has any. Returns :meth:`bind`'s
        report.
        """
        entries = []
        for placed, index in zip(self.placed, computations, strict=True):
            computation = binding.computations[index]
            streams = [int(stream) for stream in computation["streams"]]
            if not streams:
                # A computation on the host runs on the rank's host scratch.
                entries.append(
                    StreamBinding(stream=None, scratch=binding.host_scratch)
                )
                continue
            stream = streams[0]
            entries.append(
                StreamBinding(
                    stream=binding.streams[stream],
                    scratch=binding.scratches[stream],
                    pool=None
                    if binding.pools is None
                    else binding.pools[stream],
                    peers=tuple(binding.peer_pools.get(stream, ())),
                    collectives=binding.stream_groups(placed.component, 0)
                    if placed.role == DENOISER
                    else None,
                )
            )
        return self.bind(
            ImageBinding(
                slots=int(binding.slots),
                computations=tuple(entries),
                unit_pool=binding.unit_pool,
                tables=binding.tables or None,
                banks=binding.banks,
            )
        )

    def bind(self, binding: ImageBinding) -> dict[str, Any]:
        """Construct every computation's modules on the granted storage.

        Returns what the engine reads of the bound computations:
        ``warmups`` (the image shapes it warms the serving path with),
        ``captured_images`` and ``captured_latents`` (the sizes the codecs
        captured, ``null`` without graphs), ``unconditional_token_ids`` (the
        prefix the image-unconditional branch reads, ``null`` when it reads
        the request's own context), ``capacity`` (each packed forward's rows
        and tokens) and ``digests`` (each component's SHA-256 of its graph
        keys in capture order, which the worker compares across the ranks
        that hold the component).

        Raises:
            SetupError: A computation's module is not on this rank, or its
                storage is missing.
        """
        if len(binding.computations) != len(self.placed):
            raise ValueError(
                "the worker binds other image computations than stated"
            )
        self.slots = int(binding.slots)
        self._keys = {}
        for computation, bound in zip(
            self.placed, binding.computations, strict=True
        ):
            computation.stream = bound.stream
            pool = bound.pool if self.graphs else None
            keys: list[Any] = []
            if computation.role == VISION:
                keys += self._bind_vision(computation, bound, pool)
            elif computation.role == LATENT:
                keys += self._bind_latent(computation, bound, pool)
            elif computation.role == DENOISER:
                keys += self._bind_denoiser(computation, bound, binding, pool)
            if "image_decoding" in computation.kinds:
                keys += self._bind_decoder(computation, bound, binding, pool)
            self._keys.setdefault(computation.component, []).extend(keys)
        return self._report()

    def _bind_vision(
        self, computation: _Computation, bound: StreamBinding, pool: Any | None
    ) -> list[Any]:
        """Construct a vision computation's patch tower; its graph keys."""
        module = self._module(computation, "patch_encoder", PatchEncoder)
        patches = int(
            EncoderRunner.image_patches(
                self.loaded, computation.paths["patch_encoder"]
            )
        )
        computation.vision = VisionEncoder(
            module=module,
            owner=computation.label("patch_encoder"),
            stream=bound.stream,
            device=_device(bound.stream),
            scratch=bound.scratch,
            attention=self.attention,
            graphs=pool is not None,
            pool=pool,
            peers=bound.peers,
            pixel_dtype=self.pixel_dtype,
            row_budget=max(self.max_batch_tokens, patches),
            max_images=self.max_calls,
        )
        return [("vision", *key) for key in computation.vision.keys()]

    def _bind_latent(
        self, computation: _Computation, bound: StreamBinding, pool: Any | None
    ) -> list[Any]:
        """Construct a latent computation's encoder; its graph keys.

        It captures the prepared pixel size of every configured image size.
        """
        module = self._module(
            computation, "patch_autoencoder", PatchAutoencoder
        )
        sizes = []
        for height, width in self.flow_shapes:
            facts = EncoderRunner.image_facts(
                self.loaded, "vae", height, width, 0
            )
            sizes.append((int(facts[-2]), int(facts[-1])))
        computation.latent = LatentEncoder(
            module=module,
            owner=computation.label("patch_autoencoder"),
            stream=bound.stream,
            device=_device(bound.stream),
            scratch=bound.scratch,
            attention=self.attention,
            sizes=sizes,
            pixel_dtype=self.pixel_dtype,
            pool=pool,
            peers=bound.peers,
        )
        return [("latent", *key) for key in computation.latent.keys()]

    def _bind_denoiser(
        self,
        computation: _Computation,
        bound: StreamBinding,
        binding: ImageBinding,
        pool: Any | None,
    ) -> list[Any]:
        """Construct the denoiser computation's denoiser; its graph keys."""
        layout = self.layout
        if layout is None:
            raise SetupError("an image denoiser reports no latent layout")
        if (
            binding.unit_pool is None
            or not binding.tables
            or binding.banks is None
        ):
            raise SetupError(
                "an image denoiser requires the prefix cache and the latent "
                "ledger"
            )
        network = self._module(computation, "image_denoiser", ImageDenoiser)
        # The sequence-parallel degree divides every forward's tokens.
        mesh = network.mesh
        multiple = int(mesh.size("sp")) if "sp" in mesh.axes else 1
        tables = CacheTables.of(
            binding.unit_pool,
            binding.tables,
            max_sequence=self.max_model_len,
            max_query=layout.max_row_tokens,
        )
        step_limits = self._step_limits()
        condition_limits = self._condition_limits()
        graphs = pool is not None
        step_shapes = (
            self._row_shapes(
                step_limits,
                denoise_rows(
                    guidance_branches=layout.guidance_branches,
                    max_images=self.max_calls,
                ),
                multiple,
            )
            if graphs
            else None
        )
        condition_shapes = (
            self._row_shapes(
                condition_limits,
                condition_rows(max_calls=condition_limits.max_rows),
                multiple,
            )
            if graphs and condition_limits is not None
            else None
        )
        computation.denoiser = Denoiser(
            network=network,
            owner=computation.label("image_denoiser"),
            layout=layout,
            stream=bound.stream,
            device=_device(bound.stream),
            scratch=bound.scratch,
            attention=self.attention,
            tables=tables,
            banks=LatentBanks(binding.banks),
            slots=binding.slots,
            step_limits=step_limits,
            condition_limits=condition_limits,
            step_shapes=step_shapes,
            condition_shapes=condition_shapes,
            pool=pool,
            peers=bound.peers,
            collectives=bound.collectives,
            frame=lambda source: framed_prefix(self.loaded, source),
            multiple=multiple,
        )
        keys: list[Any] = []
        for name, shapes in (
            ("step", step_shapes),
            ("condition", condition_shapes),
        ):
            if shapes is not None:
                keys.extend(
                    (name, *shape.key) for shape in shapes.capture_order()
                )
        return keys

    def _bind_decoder(
        self,
        computation: _Computation,
        bound: StreamBinding,
        binding: ImageBinding,
        pool: Any | None,
    ) -> list[Any]:
        """Construct a computation's image decoder and its latent row buffer.

        It captures every configured image size the ledger's longest
        trajectory holds. Returns its graph keys.
        """
        layout = self.layout
        module = self._module(computation, "image_decoder", ImageDecoder)
        if layout is None or binding.banks is None:
            raise SetupError(
                "image decoding requires an image denoiser's layout and the "
                "latent ledger"
            )
        banks = computation.banks = LatentBanks(binding.banks)
        sizes = [
            (height, width, layout.latent_rows(height, width))
            for height, width in self.flow_shapes
            if layout.latent_rows(height, width) <= layout.max_sequence_tokens
        ]
        computation.decoder = PixelDecoder(
            module=module,
            owner=computation.label("image_decoder"),
            stream=bound.stream,
            device=_device(bound.stream),
            scratch=bound.scratch,
            attention=self.attention,
            sizes=sizes,
            latent_width=layout.width,
            latent_dtype=banks.dtype,
            pool=pool,
            peers=bound.peers,
        )
        with torch.cuda.stream(bound.stream):
            # The longest trajectory's rows, page padded.
            computation.rows = torch.zeros(
                (
                    layout.max_sequence_tokens + banks.page_units,
                    layout.width,
                ),
                dtype=banks.dtype,
                device=banks.device,
            )
        return [("decode", *key) for key in computation.decoder.keys()]

    def _capacity(self) -> dict[str, int] | None:
        """Each packed forward's live rows and tokens; ``None`` for none."""
        capacity = dict.fromkeys(
            (
                "max_step_tokens",
                "max_step_rows",
                "max_condition_tokens",
                "max_condition_rows",
                "max_vision_patch_rows",
                "max_vision_images",
            ),
            0,
        )
        for computation in self.placed:
            if computation.vision is not None:
                patches, images = computation.vision.capacity()
                capacity["max_vision_patch_rows"] = max(
                    capacity["max_vision_patch_rows"], patches
                )
                capacity["max_vision_images"] = max(
                    capacity["max_vision_images"], images
                )
            denoiser = computation.denoiser
            if denoiser is not None:
                capacity["max_step_tokens"] = denoiser.steps.live_tokens
                capacity["max_step_rows"] = denoiser.steps.live_rows
                if denoiser.conditioning is not None:
                    family = denoiser.conditioning
                    capacity["max_condition_tokens"] = family.live_tokens
                    capacity["max_condition_rows"] = family.live_rows
        return capacity if any(capacity.values()) else None

    def _report(self) -> dict[str, Any]:
        """What the engine reads of the bound computations (:meth:`bind`)."""
        graphs = self.graphs
        report: dict[str, Any] = {
            "warmups": [],
            "captured_images": None,
            "captured_latents": None,
            "unconditional_token_ids": None,
            "capacity": self._capacity(),
            "digests": {
                component: digest([_numbers(key) for key in keys])
                for component, keys in self._keys.items()
            },
        }
        latents = [
            computation.latent
            for computation in self.placed
            if computation.latent is not None
        ]
        if graphs and latents:
            report["captured_latents"] = [
                [height, width]
                for encoder in latents
                for height, width in encoder.keys()
            ]
        denoiser = self.denoiser()
        layout = self.layout
        if denoiser is None or layout is None:
            return report

        # The captured decode sizes whose image one step holds.
        steps = denoiser.steps.live_tokens
        fitting = [
            (height, width)
            for computation in self.placed
            if computation.decoder is not None
            for height, width in computation.decoder.keys()
            if layout.sequence_length(height, width) <= steps
        ]
        if graphs:
            report["captured_images"] = [
                {"height": height, "width": width, "branches": branches}
                for height, width in sorted(set(fitting))
                for branches in range(1, layout.guidance_branches + 1)
                if branches * layout.sequence_length(height, width) <= steps
            ]
        report["warmups"] = self._warmups(denoiser, layout, fitting)
        source = layout.image_unconditional
        if source is BranchSource.NEGATIVE_OR_START:
            raise SetupError(
                "an image-unconditional branch conditioned on the negative "
                "prompt has no guidance prefix the engine places"
            )
        if source is not BranchSource.CONDITIONING:
            report["unconditional_token_ids"] = list(
                framed_prefix(self.loaded, source)
            )
        return report

    def _warmups(
        self,
        denoiser: Denoiser,
        layout: ImageLayout,
        fitting: Sequence[tuple[int, int]],
    ) -> list[dict[str, Any]]:
        """The image shapes the engine warms the serving path with.

        One per branch count a step evaluates: the largest captured size a
        step holds at that count, or without graphs a square image the
        ledger holds. A guided count needs a second slot for its
        unconditional prefix, and reads the framed empty negative prompt.
        """
        steps = denoiser.steps.live_tokens
        negative = list(
            framed_prefix(self.loaded, BranchSource.NEGATIVE_OR_START)
        )
        units = min(layout.max_sequence_tokens, denoiser.banks.units)
        side = max(int(units**0.5), 1) * layout.downsample
        largest = sorted(fitting, key=lambda size: -size[0] * size[1])
        warmups = []
        for branches in range(1, layout.guidance_branches + 1):
            if branches > 1 and self.slots < 2:
                continue
            shape = next(
                (
                    (height, width)
                    for height, width in largest
                    if branches * layout.sequence_length(height, width) <= steps
                ),
                None,
            )
            if shape is None:
                if self.graphs:
                    continue
                shape = (side, side)
            warmups.append(
                {
                    "height": shape[0],
                    "width": shape[1],
                    "branches": branches,
                    "negative_token_ids": negative if branches > 1 else [],
                }
            )
        return warmups

    # The text computations' questions.

    def facts(self) -> ImageFacts:
        """The image facts text prompts are laid out with."""
        return self.image_facts

    def denoiser(self) -> Denoiser | None:
        """The image denoiser that writes latent conditioning, if any."""
        for computation in self.placed:
            if computation.denoiser is not None:
                return computation.denoiser
        return None

    def attach_text(self, computations: Mapping[str, Any]) -> None:
        """Take each denoiser's component's text computation.

        The text computation prefills the denoiser's guidance prefixes
        (``TextComputation.prefill_cache``).
        """
        for computation in self.placed:
            if computation.denoiser is not None:
                computation.denoiser.attach_text(
                    computations.get(computation.component)
                )

    # Warmup.

    def warmup_phases(self) -> list[dict[str, Any]]:
        """The warmup phases, in the order every rank runs them.

        Vision buckets, latent encode sizes, then the denoiser's steps and
        condition writes over scratch tables, then the decoder's sizes;
        each names its computation (its index in this executor), the scratch
        lengths of slots ``1..`` (``null`` for a phase over no prefix
        cache), each step's attention rows and whether they write.
        """
        phases: list[dict[str, Any]] = []
        self._phases = []
        order = {VISION: 0, LATENT: 1, DENOISER: 2, DECODER: 3}
        indexed = sorted(
            enumerate(self.placed),
            key=lambda item: (order[item[1].role], item[0]),
        )
        decoders = []
        for index, computation in indexed:
            if computation.role in (VISION, LATENT):
                phases.append(
                    {"computation": index, "lengths": None, "steps": []}
                )
                self._phases.append((computation, computation.role, None))
            if computation.denoiser is not None:
                steps, conditions = computation.denoiser.warm_forwards(
                    self.slots
                )
                for name, forwards, writes in (
                    ("steps", steps, False),
                    ("conditions", conditions, True),
                ):
                    if not forwards:
                        continue
                    used = min(
                        max(len(forward.rows) for forward in forwards),
                        max(self.slots, 1),
                    )
                    length = max(
                        prefix + (tokens if writes else 0)
                        for forward in forwards
                        for _, prefix, tokens in forward.rows
                    )
                    phases.append(
                        {
                            "computation": index,
                            "lengths": [length] * used,
                            "steps": [
                                forward.attention() for forward in forwards
                            ],
                            "writes": writes,
                        }
                    )
                    self._phases.append((computation, name, forwards))
            if computation.decoder is not None:
                decoders.append((index, computation))
        for index, computation in decoders:
            phases.append({"computation": index, "lengths": None, "steps": []})
            self._phases.append((computation, DECODER, None))
        return phases

    def warmup_phase(self, index: int, *pages: torch.Tensor) -> None:
        """Run warmup phase ``index`` with each step's page columns."""
        computation, name, forwards = self._phases[int(index)]
        if name == VISION:
            assert computation.vision is not None
            computation.vision.warm()
        elif name == LATENT:
            assert computation.latent is not None
            computation.latent.warm(self._warm_side())
        elif name == DECODER:
            assert computation.decoder is not None and self.layout is not None
            side = self._warm_side()
            computation.decoder.warm(side, self.layout.latent_rows(side, side))
        else:
            denoiser = computation.denoiser
            assert denoiser is not None and forwards is not None
            if len(pages) != len(forwards):
                raise ValueError(
                    "a denoiser warmup phase receives one page column per step"
                )
            for forward, columns in zip(forwards, pages, strict=True):
                denoiser.warm(
                    forward, np.asarray(columns.numpy(), dtype=np.int64)
                )

    def _warm_side(self) -> int:
        """The side of the image the codecs warm their kernels with.

        Two latent patches, or 64 pixels without a denoiser.
        """
        return 64 if self.layout is None else 2 * self.layout.downsample

    def warmup_finish(self) -> None:
        """Wait for the warmup's device work; every capture ran in a phase."""
        for computation in self.placed:
            if computation.stream is not None:
                computation.stream.synchronize()

    def seal(self) -> None:
        """Refuse captures from now on: serving replays only."""
        for computation in self.placed:
            for module in computation.modules():
                module.seal()

    # Requests.

    def start(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> list[tuple[tuple[int, int, int], str]]:
        """Admit started requests (``NewRequest`` JSON); refuses none.

        Each slot keeps its request's image parameters, admitted negative
        tokens and posterior draw count; a reused slot forgets its earlier
        request's trajectory.
        """
        slots = []
        for request in requests:
            slot = int(request["request_pool_idx"])
            params = request.get("image")
            generation = request.get("generation") or {}
            self.requests[slot] = _Request(
                key=_key(request["request_key"]),
                params=None
                if params is None
                else ImageParams.from_json(params),
                negative=tuple(
                    int(token)
                    for token in generation.get("negative_token_ids") or ()
                ),
            )
            slots.append(slot)
        self._forget(slots)
        return []

    def release(self, slots: Sequence[int]) -> None:
        """Forget the retired slots' requests and trajectories."""
        for slot in slots:
            self.requests.pop(int(slot), None)
        self._forget([int(slot) for slot in slots])

    def _forget(self, slots: Sequence[int]) -> None:
        for computation in self.placed:
            if computation.denoiser is not None:
                computation.denoiser.release(slots)

    def _request(self, entry: Mapping[str, Any]) -> _Request:
        record = self.requests.get(int(entry["slot"]))
        if record is None or record.key != _key(entry["call"]["request_key"]):
            raise ValueError("an image call has no admitted request")
        return record

    def _image_params(self, entry: Mapping[str, Any]) -> ImageParams:
        params = self._request(entry).params
        if params is None:
            raise ValueError("an image trajectory requires image parameters")
        return params

    # Preparation.

    def prepare(
        self, computation: int, step: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """What each call of a batch reads and writes, in call order."""
        placed = self.placed[int(computation)]
        result = []
        for entry in step["calls"]:
            if entry["refused"] or not entry["active"]:
                result.append(dict(NOTHING))
                continue
            kind = str(entry["call"]["kind"])
            if kind not in placed.kinds:
                raise ValueError(
                    f"computation {computation} serves no {kind} calls"
                )
            result.append(self._preparations[kind](placed, entry).to_json())
        return result

    def _input(self, entry: Mapping[str, Any]) -> Mapping[str, Any]:
        value = (entry.get("image") or {}).get("input")
        if not value:
            raise ValueError(
                "an image encoding requires exactly one inline or resident "
                "image"
            )
        return value

    def _resident_facts(
        self, tower: str, resident: Mapping[str, Any]
    ) -> list[int]:
        """The facts of a resident image prepared for ``tower``.

        ``[canvas height, width, grid rows, columns, *pixel shape]``
        (``EncoderRunner.image_facts``).

        Raises:
            ValueError: The image's product states no value range, which
                its preparation reads.
        """
        range_interval(resident["value_range"])
        shape = [int(extent) for extent in resident["shape"]]
        height = int(resident["height"]) or shape[-2]
        width = int(resident["width"]) or shape[-1]
        return [
            int(value)
            for value in EncoderRunner.image_facts(
                self.loaded, tower, height, width, 0
            )
        ]

    def _prepare_vision(
        self, computation: _Computation, entry: Mapping[str, Any]
    ) -> PlanEntry:
        module = self._module(computation, "patch_encoder", PatchEncoder)
        value = self._input(entry)
        if "inline" in value:
            facts = value["inline"]
            grid = facts.get("grid")
            shape = [int(extent) for extent in facts["shape"]]
            height, width = int(facts["height"]), int(facts["width"])
            resident = False
        else:
            height, width, rows, columns, *shape = self._resident_facts(
                "vit", value["resident"]
            )
            grid = None if rows == 0 else (rows, columns)
            resident = True
        if grid is None:
            grid = _whole_grid(module, shape)
        return vision_entry(
            grid=(int(grid[0]), int(grid[1])),
            downsample=module.downsample,
            output_size=features(module)[0],
            height=height,
            width=width,
            resident=resident,
        )

    def _prepare_latent(
        self, computation: _Computation, entry: Mapping[str, Any]
    ) -> PlanEntry:
        module = self._module(
            computation, "patch_autoencoder", PatchAutoencoder
        )
        value = self._input(entry)
        if "inline" in value:
            facts = value["inline"]
            shape = [int(extent) for extent in facts["shape"]]
            height, width = int(facts["height"]), int(facts["width"])
            resident = False
        else:
            height, width, _, _, *shape = self._resident_facts(
                "vae", value["resident"]
            )
            resident = True
        side = module.downsample
        return latent_entry(
            rows=(shape[-2] // side) * (shape[-1] // side),
            width=module.patch_size**2 * module.latent_channels,
            height=height,
            canvas_width=width,
            resident=resident,
        )

    def _prepare_trajectory(
        self, computation: _Computation, entry: Mapping[str, Any]
    ) -> PlanEntry:
        """A latent preparation or denoising step: a ledger binding."""
        self._image_params(entry)
        return PlanEntry(latent=True)

    def _prepare_decoding(
        self, computation: _Computation, entry: Mapping[str, Any]
    ) -> PlanEntry:
        decoder = self._module(computation, "image_decoder", ImageDecoder)
        call = entry["call"]
        value_range = range_name(decoder.decoder.value_range)
        if call.get("latent_input") is not None:
            params = self._image_params(entry)
            return decoding_entry(
                height=params.height,
                width=params.width,
                value_range=value_range,
                publishes=call.get("image_output") is not None,
                trajectory=True,
            )
        resident = self._input(entry).get("resident")
        if resident is None:
            raise ValueError(
                "image decoding requires a trajectory or a resident image"
            )
        # Its quantization reads the resident image's value range, which a
        # product without one cannot give.
        range_interval(resident["value_range"])
        shape = [int(extent) for extent in resident["shape"]]
        return decoding_entry(
            height=int(resident["height"]) or shape[-2],
            width=int(resident["width"]) or shape[-1],
            value_range=value_range,
            publishes=False,
            trajectory=False,
        )

    # Execution.

    def execute(
        self,
        computation: int,
        step: Mapping[str, Any],
        tensors: Sequence[torch.Tensor],
    ) -> list[int]:
        """Run a batch on the tensors the worker leased and bound.

        ``step`` is the prepared step: its ``calls`` and the ``prepared``
        entries :meth:`prepare` returned. ``tensors`` holds the generic
        reads and writes of every call (none for image calls), then each
        call's image tensors in call order (its resident read, its prepared
        pixels, each write, its lent bytes), then for a denoising batch the
        ``int64 [5 + 2G, R]`` attention rows. Returns each call's graph
        mode.
        """
        placed = self.placed[int(computation)]
        calls = step["calls"]
        prepared = step["prepared"]
        cursor = sum(
            len(entry.get("reads", ())) + len(entry.get("writes", ()))
            for entry in prepared
        )
        bound: list[
            tuple[Mapping[str, Any], PlanEntry, list[torch.Tensor]]
        ] = []
        for entry, planned in zip(calls, prepared, strict=True):
            plan = PlanEntry.from_json(planned)
            if plan is None or entry["refused"] or not entry["active"]:
                continue
            count = plan.tensors
            bound.append((entry, plan, list(tensors[cursor : cursor + count])))
            cursor += count
        rest = list(tensors[cursor:])
        modes = {int(entry["index"]): GraphMode.EAGER for entry in calls}
        if bound:
            kind = str(bound[0][0]["call"]["kind"])
            if kind == "vision_encoding":
                self._encode_vision(placed, bound, modes)
            elif kind == "latent_encoding":
                self._encode_latents(placed, bound, modes)
            elif kind == "latent_preparation":
                self._prepare_latents(placed, bound, modes)
            elif kind == "denoising":
                if len(rest) != 1:
                    raise ValueError(
                        "a denoising batch receives its attention rows"
                    )
                self._denoise(placed, bound, rest[0], modes)
            elif kind == "image_decoding":
                self._decode(placed, bound, modes)
            else:
                raise ValueError(f"{kind} is not an image model's call")
        return [int(modes[int(entry["index"])]) for entry in calls]

    def _resident_pixels(
        self,
        computation: _Computation,
        tower: Tower,
        image: torch.Tensor,
        entry: Mapping[str, Any],
    ) -> PreparedImage:
        """A resident image prepared for ``tower``.

        It is prepared on the device ``computation`` runs on.
        """
        resident = self._input(entry)["resident"]
        with torch.inference_mode(), torch.cuda.stream(computation.stream):
            return prepare_tensor_image(
                self.loaded.image_processor,
                tower,
                image,
                device=image.device,
                signed_unit=resident["value_range"] == "signed_unit",
            )

    def _encode_vision(self, computation, bound, modes) -> None:
        encoder = computation.vision
        assert encoder is not None
        images = []
        for entry, plan, tensors in bound:
            source, out = tensors[0], tensors[-1]
            if plan.read:
                prepared = self._resident_pixels(
                    computation, "vit", source, entry
                )
                pixels, grid = prepared.pixels, prepared.grid_shape
            else:
                pixels = source
                grid = self._input(entry)["inline"].get("grid")
            if grid is None:
                # A tower taking whole images reads their pixels as they
                # are.
                grid = _whole_grid(encoder.module, pixels.shape)
            images.append(
                VisionImage(pixels, (int(grid[0]), int(grid[1])), out)
            )
        mode = encoder.encode(images)
        for entry, _, _ in bound:
            modes[int(entry["index"])] = mode

    def _encode_latents(self, computation, bound, modes) -> None:
        encoder = computation.latent
        assert encoder is not None
        for entry, plan, tensors in bound:
            source, out = tensors[0], tensors[-1]
            if plan.read:
                pixels = self._resident_pixels(
                    computation, "vae", source, entry
                ).pixels
            else:
                pixels = source
            request = self._request(entry)
            draw = request.draws
            request.draws += 1
            modes[int(entry["index"])] = encoder.encode(
                pixels, out, seed=request.seed, draw=draw
            )

    def _prepare_latents(self, computation, bound, modes) -> None:
        denoiser = computation.denoiser
        assert denoiser is not None
        for entry, _, _ in bound:
            request = self._request(entry)
            params = self._image_params(entry)
            image_index = entry["call"].get("image_index")
            if image_index is None:
                raise ValueError("latent preparation names no image index")
            modes[int(entry["index"])] = denoiser.prepare_latents(
                int(entry["slot"]),
                request.key,
                params,
                request.negative,
                int(image_index),
                LatentBinding.from_json(entry["latent"]),
            )

    def _denoise(
        self, computation, bound, forward: torch.Tensor, modes
    ) -> None:
        denoiser = computation.denoiser
        assert denoiser is not None
        calls = []
        for entry, _, _ in bound:
            request = self._request(entry)
            image = entry["image"]
            calls.append(
                StepCall(
                    index=int(entry["index"]),
                    slot=int(entry["slot"]),
                    key=request.key,
                    params=self._image_params(entry),
                    negative=request.negative,
                    latent=LatentBinding.from_json(entry["latent"]),
                    logical=int(image["progress"]["logical"]),
                    visible=int(image["progress"]["visible"]),
                    allocated={
                        int(slot): int(tokens)
                        for slot, tokens in image.get("allocated") or ()
                    },
                )
            )
        ran = denoiser.denoise(calls, forward.numpy())
        for call, mode in zip(calls, ran, strict=True):
            modes[call.index] = mode

    def _decode(self, computation, bound, modes) -> None:
        decoder = computation.decoder
        assert decoder is not None
        for entry, plan, tensors in bound:
            out = tensors[-1]
            if plan.latent:
                latent = LatentBinding.from_json(entry["latent"])
                product = tensors[0] if plan.writes else None
                rows, banks = computation.rows, computation.banks
                assert rows is not None and banks is not None
                with (
                    torch.inference_mode(),
                    torch.cuda.stream(computation.stream),
                ):
                    banks.gather(latent.read, latent.pages, rows)
                mode = decoder.decode(
                    rows[: latent.units],
                    latent.height,
                    latent.width,
                    product,
                    out,
                )
            else:
                resident = self._input(entry)["resident"]
                mode = decoder.quantize(
                    tensors[0], range_interval(resident["value_range"]), out
                )
            modes[int(entry["index"])] = mode

    # Kernels and close.

    def kernels(self) -> list[tuple[str, list[str]]]:
        """Each module's kernel-table label and selected kernels."""
        listed = []
        for computation in self.placed:
            if computation.vision is not None:
                listed.append(
                    (
                        computation.label("patch_encoder"),
                        computation.vision.kernels(),
                    )
                )
            if computation.latent is not None:
                listed.append(
                    (
                        computation.label("patch_autoencoder"),
                        computation.latent.kernels(),
                    )
                )
            if computation.denoiser is not None:
                listed.append(
                    (
                        computation.label("image_denoiser"),
                        computation.denoiser.kernels(),
                    )
                )
            if computation.decoder is not None:
                listed.append(
                    (
                        computation.label("image_decoder"),
                        computation.decoder.kernels(),
                    )
                )
        return listed

    def close(self, aborted: bool) -> None:
        """Release every computation's modules, newest first.

        The worker drained the streams. Every release is attempted before
        the first failure is raised.
        """
        failure: BaseException | None = None
        for computation in reversed(self.placed):
            for module in reversed(computation.modules()):
                try:
                    module.close(bool(aborted))
                except Exception as error:  # noqa: BLE001 - close every part
                    failure = failure or error
        self.requests.clear()
        if failure is not None:
            raise failure


def _numbers(key: Any) -> list[int]:
    """A graph key as integers, its family's name numbered by ``_FAMILIES``."""
    family, *extents = key
    return [_FAMILIES[family], *(int(extent) for extent in extents)]
