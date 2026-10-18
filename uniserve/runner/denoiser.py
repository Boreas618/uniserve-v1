"""The forwards of the latent denoisers: :class:`DenoiserRunner`.

One runner serves one denoiser on one stream. A request's state lives in
tensors the caller owns; every method names the tensors it reads and writes
as parameters, and a tensor wider than a forward needs is viewed through
its leading elements. The parameters of :meth:`DenoiserRunner.draw_noise`,
:meth:`~DenoiserRunner.prepare_latents`,
:meth:`~DenoiserRunner.encode_conditions` and
:meth:`~DenoiserRunner.denoise` depend on the denoiser's capability class
and are documented there.

**Video denoisers** (``VideoDenoiser``). A request's latent samples, state
tables, retained conditioning and noise are the caller's.

- ``draw_noise`` fills a request's noise tensors from its seed with the
  library's native draw.
- ``prepare_latents`` turns that noise and the request's size into its
  initial samples and state tables.
- ``encode_conditions`` writes a request's encoded conditions into its
  retained conditioning.
- ``denoise`` evaluates one solver step in place. Its shapes are fixed by
  its layout and every changing value is the contents of a tensor, so it is
  captured into one CUDA graph per layout.

Requests of one shape share a *layout* (``VideoDenoiser.layout_size``):
:meth:`DenoiserRunner.prepare_layout` prepares a layout's constants and
returns them as an opaque handle the forwards take.

Sizes cross as flat integer lists:

- a request size: ``[frames, width, height, prompt_tokens]``;
- conditions: ``[role, frames, width, height, audio_samples]`` per
  condition in request order, ``role`` 0 for the first frame, 1 for the
  last frame and 2 for a reference, ``frames`` 0 for a condition without
  pixels;
- vision spans: ``[start, stop]`` per vision block of the presented prompt;
- a layout: ``[frames, width, height, text_rows, condition_rows]``.

**Image denoisers** (``ImageDenoiser``), which attend to the prefix cache.
The caller's resident tensors are the latent pool's step workspace
(``samples``), its per-slot timesteps, the product arena and the request
slots' installed block tables; they are passed once, to
:meth:`DenoiserRunner.bound`. A *row* is one guidance branch of one image,
``[sample, units, height, width, token, tokens, prediction]``: its latent
rows ``samples[sample:sample + units]``, its size, its positions
``positions[:, token:token + tokens]`` and its rows of the predictions;
row ``i`` reads ``timesteps[i]``. Rows cross flattened.

- ``draw_noise`` fills an image's noise from its seed.
- ``prepare_latents`` turns that noise into the image's initial samples.
- ``denoise`` gathers the rows' attention metadata from the slot tables on
  the device and predicts every row. Its shapes are fixed by its rows and
  page widths, so it is captured into one CUDA graph per set of rows;
  :meth:`DenoiserRunner.plan_denoise` builds the plans of attention
  operators that plan a launch on the host.
- ``encode_conditions`` writes the keys and values of latent images
  inserted as conditioning.
- :meth:`DenoiserRunner.step` combines each image's branch predictions
  with its trajectory's guidance and advances its samples by one solver
  step. A trajectory's schedules and guidance are made from the image's
  parameters, which the caller keeps and passes to :meth:`~DenoiserRunner.plan`
  and to every step.

The static methods answer a deployment's sizing questions from the model's
configuration alone, so a rank without the denoiser's weights asks them
too.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch

from uniserve.bindings import capturing
from uniserve.bindings.context import ExecutionContext
from uniserve.bindings.workspace import TensorBuffers
from uniserve.diffusion import (
    Branch,
    DenoisingStep,
    Guidance,
    Renorm,
    Schedule,
    advance_,
    normal_noise,
)
from uniserve.errors import SetupError
from uniserve.loading.rank import capability
from uniserve.media import image, video
from uniserve.models import (
    AudioDecoder,
    AudioEncoder,
    Condition,
    ConditionRole,
    ImageDenoiser,
    LatentInput,
    PatchEncoder,
    TextEncoder,
    TextSize,
    VideoDecoder,
    VideoDenoiser,
    VideoEncoder,
    VideoPostprocessor,
)
from uniserve.nn.attention import (
    AttentionBatch,
)
from uniserve.nn.rng import flow_noise_seed
from uniserve.processing import (
    load_tokenizer,
    resolve_prefix,
)
from uniserve.runner.attention import AttentionMetadata
from uniserve.runner.context import (
    execution_context,
    held_module,
    kernel_records,
)
from uniserve.tensors import BufferConfig

__all__ = ["DenoiserRunner", "Outputs"]

# Guidance branch mask bits, in evaluation order.
_BITS = {
    Branch.CONDITIONED: 1,
    Branch.TEXT_UNCONDITIONAL: 2,
    Branch.IMAGE_UNCONDITIONAL: 4,
}

_ROLES = (
    ConditionRole.FIRST_FRAME,
    ConditionRole.LAST_FRAME,
    ConditionRole.REFERENCE,
)


class _Guided(Protocol):
    """The guidance an image denoiser makes for one trajectory.

    ``ImageDenoiser`` does not declare the method, so the runner names the
    signature it calls.
    """

    def make_guidance(
        self,
        *,
        text_scale: float,
        image_scale: float,
        interval: tuple[float, float],
        renorm: Renorm,
        renorm_min: float,
    ) -> Guidance: ...


def _trajectory(
    network,
    steps: int,
    shift: float,
    text_scale: float,
    image_scale: float,
    interval_low: float,
    interval_high: float,
    renorm: str,
    renorm_min: float,
) -> tuple[dict[str, Schedule], Guidance]:
    """The schedules and guidance of one image trajectory.

    A non-positive ``shift`` keeps the denoiser's default shift.
    """
    shift = float(shift)
    schedules = dict(
        network.make_schedules(
            int(steps), shift=shift if shift > 0 else None, device="cpu"
        )
    )
    guidance = cast(_Guided, network).make_guidance(
        text_scale=float(text_scale),
        image_scale=float(image_scale),
        interval=(float(interval_low), float(interval_high)),
        renorm=Renorm(str(renorm)),
        renorm_min=float(renorm_min),
    )
    return schedules, guidance


class _Buffered(Protocol):
    """The constants and workspace a denoiser declares for a layout.

    The capability class does not declare these methods, so the runner
    names the signatures it calls.
    """

    def constant_buffers(self, size: Any) -> Mapping[str, BufferConfig]: ...

    def prepare_constants(
        self, size: Any, *, out: Mapping[str, torch.Tensor]
    ) -> None: ...

    def workspace_buffers(self, size: Any) -> Mapping[str, BufferConfig]: ...


def _denoiser(loaded, component: str) -> VideoDenoiser:
    """The video denoiser at the root of a declared component."""
    entry = loaded.description.entries.get(component)
    if entry is None:
        raise SetupError(f"the model declares no component {component!r}")
    root = loaded.module
    module = root.get_submodule(entry.component) if entry.component else root
    if not isinstance(module, VideoDenoiser):
        raise SetupError(f"component {component!r} is not a video denoiser")
    return module


def _require(loaded, kind: type):
    module = capability(loaded.module, kind)
    if module is None:
        raise SetupError(f"the model has no {kind.__name__}")
    return module


def _called(loaded, entry) -> list[tuple[Any, str]]:
    """The modules an entry's methods are called on, with each method name.

    A method may be nested in the entry's component (``"vision.encode"``).
    A module this rank does not hold is left out: a rank's share of a
    partitioned component keeps an absent submodule as ``None`` (a pipeline
    stage without the denoiser's conditioner).
    """
    called = []
    for point in entry.points:
        prefix, _, method = point.method.rpartition(".")
        path = ".".join(part for part in (entry.component, prefix) if part)
        module = held_module(loaded.module, path)
        if module is not None:
            called.append((module, method))
    return called


def _owner(loaded, kind: type) -> str:
    """The component whose entry calls a module of ``kind``."""
    for name, entry in loaded.description.entries.items():
        for module, _ in _called(loaded, entry):
            if isinstance(module, kind):
                return name
    raise SetupError(f"no component calls the model's {kind.__name__}")


def _conditions(values: Sequence[int]) -> tuple[Condition, ...]:
    flat = [int(value) for value in values]
    result = []
    for role, frames, width, height, samples in zip(
        *(flat[offset::5] for offset in range(5)), strict=True
    ):
        pixels = (
            None
            if frames == 0
            else video.Config(frames, image.Config(height, width))
        )
        result.append(Condition(_ROLES[role], pixels, samples))
    return tuple(result)


def _spans(values: Sequence[int]) -> tuple[tuple[int, int], ...]:
    flat = [int(value) for value in values]
    return tuple(zip(flat[::2], flat[1::2], strict=True))


def _size(denoiser: VideoDenoiser, size, conditions=(), spans=()):
    frames, width, height, tokens = (int(value) for value in size)
    return denoiser.make_size(
        frames,
        tokens,
        canvas=image.Config(height, width),
        conditions=_conditions(conditions),
        vision_spans=_spans(spans),
    )


def _describe(layout) -> list[int]:
    return [
        int(layout.num_frames),
        int(layout.canvas.width),
        int(layout.canvas.height),
        int(layout.num_text_tokens),
        int(layout.condition_rows),
    ]


def _layout(denoiser: VideoDenoiser, value: Sequence[int]):
    """The denoiser's layout of a flat layout.

    Raises:
        ValueError: ``value`` names no layout of the denoiser.
    """
    frames, width, height, text, rows = (int(item) for item in value)
    layout = denoiser.layout_size(
        denoiser.make_size(frames, text, canvas=image.Config(height, width))
    )
    if rows:
        layout = denoiser.condition_layout(layout, rows)
    if _describe(layout) != [frames, width, height, text, rows]:
        raise ValueError(f"{list(value)} is not a layout of the denoiser")
    return layout


def _elements(shape: Sequence[int]) -> int:
    count = 1
    for extent in shape:
        count *= int(extent)
    return count


def _view(tensor: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """The leading elements of a flat backing as ``shape``."""
    return tensor.reshape(-1)[: _elements(shape)].view(tuple(shape))


def _dtype(dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _image_denoiser(loaded):
    """The loaded model's image denoiser, or None."""
    return capability(loaded.module, ImageDenoiser)


def _image_rows(values: Sequence[int]) -> list[tuple[int, ...]]:
    """Flattened image rows as tuples of their seven fields."""
    flat = [int(value) for value in values]
    return [tuple(flat[start : start + 7]) for start in range(0, len(flat), 7)]


@functools.cache
def _tokenizer(path: Path):
    return load_tokenizer(path)


@dataclass(frozen=True)
class Outputs:
    """The bounded outputs components publish, read column by column.

    Output ``i`` is the tensor ``names()[i]`` of component ``owners()[i]``.
    Its shape bound is the next ``ranks()[i]`` values of ``extents()``; at
    most one of its axes, ``variable_axes()[i]``, varies per request.
    """

    owned: tuple[tuple[str, Mapping[str, Any]], ...]

    @classmethod
    def of(cls, components: Mapping[str, Sequence[Mapping[str, Any]]]):
        """The outputs of ``components``, each in its declared order."""
        owned = tuple(
            (owner, output)
            for owner, outputs in components.items()
            for output in outputs
        )
        for owner, output in owned:
            if len(output["variable_axes"]) > 1:
                raise ValueError(
                    f"{owner}.{output['name']} varies along more than one axis"
                )
        return cls(owned)

    def owners(self) -> list[str]:
        """The component publishing each output."""
        return [owner for owner, _ in self.owned]

    def names(self) -> list[str]:
        """Each output's name, unique within its component."""
        return [str(output["name"]) for _, output in self.owned]

    def dtypes(self) -> list[str]:
        """Each output's element type name."""
        return [str(output["dtype"]) for _, output in self.owned]

    def ranks(self) -> list[int]:
        """Each output's number of axes."""
        return [len(output["shape"]) for _, output in self.owned]

    def extents(self) -> list[int]:
        """Every output's largest extents, concatenated in output order."""
        return [
            int(extent)
            for _, output in self.owned
            for extent in output["shape"]
        ]

    def variable_axes(self) -> list[int]:
        """The axis of each output a request sizes; -1 for a fixed shape."""
        return [
            int(output["variable_axes"][0]) if output["variable_axes"] else -1
            for _, output in self.owned
        ]

    def raster_axes(self) -> list[int]:
        """Each output's ``[height, width]`` axes, flattened.

        The axes hold a video request's raster; ``[-1, -1]`` for an output
        that is not laid out at the canvas.
        """
        return [
            int(axis)
            for _, output in self.owned
            for axis in (output["raster_axes"] or (-1, -1))
        ]


def _product(name, layout, *, raster_axes=None) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": _dtype(layout.dtype),
        "shape": [int(extent) for extent in layout.shape],
        "variable_axes": [int(axis) for axis in layout.variable_axes],
        "raster_axes": raster_axes,
    }


def _rows(name: str, rows: int, sample) -> dict[str, Any]:
    """A product of request-sized rows shaped like one row of ``sample``."""
    return {
        "name": name,
        "dtype": _dtype(sample.dtype),
        "shape": [int(rows), *(int(extent) for extent in sample.shape[1:])],
        "variable_axes": [0],
        "raster_axes": None,
    }


@dataclass
class _Layout:
    """One prepared layout: its constants and its share of the workspace."""

    size: Any
    # The layout's state fields, in state order. Describing them runs the
    # network's rotary tables, so they are read once, outside any capture.
    fields: Mapping[str, BufferConfig]
    # ``(rows, width)`` of the retained conditioning.
    text_condition: tuple[int, int]
    constants: Mapping[str, torch.Tensor]
    workspace: Mapping[str, torch.Tensor]
    # Each modality's ``[1]`` timestep, gathered from its schedule per step.
    timesteps: dict[str, torch.Tensor]
    # Constants allocated for this layout alone (other than the maximum's).
    owned: TensorBuffers | None = None


class DenoiserRunner:
    """The forwards of one denoiser on one stream.

    The constructor serves a video denoiser; :meth:`bound` serves an image
    denoiser of a bound component.

    Args:
        loaded: The loaded model.
        component: The declared component of the video denoiser.
        stream: The CUDA stream handle every forward enqueues on.
        attention: The attention backend the model loaded with.
        scratch: The stream's shared work areas
            (``uniserve.bindings.workspace.Scratch``).
        maximum: The largest layout, whose capacity the operators and the
            workspace are prepared for.
    """

    # The denoiser, a ``VideoDenoiser`` or an ``ImageDenoiser``.
    _network: Any

    # An image denoiser's resident tensors and the tensors only its
    # forwards read and write (see :meth:`bound`).
    _pipeline: Any
    _samples: torch.Tensor
    _slot_timesteps: torch.Tensor
    _predictions: torch.Tensor
    _step: torch.Tensor
    _attention_metadata: AttentionMetadata

    def __init__(
        self,
        loaded,
        component: str,
        stream: int,
        attention: str,
        scratch,
        maximum: Sequence[int],
    ) -> None:
        network = _denoiser(loaded, str(component))
        self._network = network
        self._device = next(network.parameters()).device
        self._context: ExecutionContext = execution_context(
            network, int(stream), str(attention), scratch
        )
        self._maximum = _layout(network, maximum)
        self._context.prepare(self._maximum)
        self._schedules = network.make_schedules(
            network.num_steps, shift=None, device=self._device
        )
        self._modalities = tuple(network.modalities)
        self._draw_noise: Callable[..., None] = self._video_draw_noise
        self._prepare_latents: Callable[..., None] = self._video_prepare_latents
        self._encode_conditions: Callable[..., None] = (
            self._video_encode_conditions
        )
        self._denoise: Callable[..., None] = self._video_denoise

    @classmethod
    def image(
        cls,
        loaded,
        path: str,
        stream: int,
        device: int,
        scratch,
        cache,
        rows: int,
        tokens: int,
        units: int,
        width: int,
        page_tokens: Sequence[int],
        windowed: Sequence[int],
        samples: torch.Tensor,
        slot_timesteps: torch.Tensor,
        unit_tables: torch.Tensor,
        start_pages: torch.Tensor,
        table_shapes: torch.Tensor,
    ) -> DenoiserRunner:
        """A runner of an image model's denoiser.

        Args:
            loaded: The loaded model.
            path: The denoiser module's path in the model.
            stream: The CUDA stream handle every forward enqueues on (0 on
                the host).
            device: The CUDA ordinal ``stream`` belongs to, -1 for the
                host. Modules placed on a generation tower compute there
                between the stream's steps.
            scratch: The stream's shared work areas.
            cache: The prefix cache view object the denoiser's attention
                reads.
            rows: The most rows one forward evaluates.
            tokens: The most tokens (framed latent sequences) of a forward;
                with ``rows``, the size the operators are prepared for.
            units: The most latent rows one forward predicts.
            width: The most pages one row selects from a block table.
            page_tokens: Tokens per page of each numerical block table.
            windowed: Whether each numerical block table retires pages
                behind an attention window.
            samples: The latent pool's step workspace ``[units, width]``,
                where the caller gathers each trajectory's committed pages
                before a step and from which it scatters them after it.
            slot_timesteps: The latent pool's ``[slots + 1, 1]`` float32
                per-slot timesteps the solver update reads.
            unit_tables: ``[tables, S, units]`` int32, each slot's installed
                units from its installed start page on.
            start_pages: ``[groups, S]`` int32, each slot's installed start
                page.
            table_shapes: ``[tables, 3]`` int32 ``(group, page tokens,
                window or -1)``.
        """
        path = str(path)
        network = loaded.module.get_submodule(path) if path else loaded.module
        if not isinstance(network, ImageDenoiser):
            raise SetupError(
                f"{type(network).__qualname__} is not an image denoiser"
            )
        # The denoiser receives exact host lengths with each forward.
        context = execution_context(
            network,
            int(stream),
            loaded.attention,
            scratch,
            device=int(device),
            cache=cache,
        )
        context.prepare(TextSize(int(tokens), int(rows)))
        runner = cls.__new__(cls)
        runner._network = network
        runner._context = context
        runner._device = samples.device
        runner._samples = samples
        runner._slot_timesteps = slot_timesteps
        runner._pipeline = network.mesh.get_group(
            "pp" if "pp" in network.mesh.axes else ()
        )

        # The tensors only the forwards read and write: every row's
        # predictions, packed in row order, the zero evaluation index the
        # network's input carries, and the attention metadata the gather
        # derives from the slot tables. Captured graphs address them, so
        # they live as long as the runner.
        rows, tokens = int(rows), int(tokens)
        target = runner._device
        latent_width = network.patch_size**2 * network.latent_channels
        with torch.inference_mode(), torch.cuda.stream(context.stream):
            runner._predictions = torch.zeros(
                (int(units), latent_width),
                dtype=network.prediction_dtype,
                device=target,
            )
            runner._step = torch.zeros(1, dtype=torch.int64, device=target)
            runner._attention_metadata = AttentionMetadata(
                page_tokens,
                windowed,
                unit_tables,
                start_pages,
                table_shapes,
                rows,
                tokens,
                int(width),
                target,
            )
        runner._draw_noise = runner._image_draw_noise
        runner._prepare_latents = runner._image_prepare_latents
        runner._encode_conditions = runner._image_encode_conditions
        runner._denoise = runner._image_denoise
        return runner

    # Sizing, answered from configuration.

    @staticmethod
    def modalities(loaded, component: str) -> list[str]:
        """The sample modalities in the denoiser's order."""
        return list(_denoiser(loaded, component).modalities)

    @staticmethod
    def tasks(loaded, component: str) -> list[str]:
        """The tasks the denoiser serves."""
        return list(_denoiser(loaded, component).tasks)

    @staticmethod
    def num_steps(loaded, component: str) -> int:
        """Solver steps of one trajectory."""
        return int(_denoiser(loaded, component).num_steps)

    @staticmethod
    def schedule_shifts(loaded, component: str) -> list[float]:
        """The ``[video, audio]`` schedule shifts."""
        shifts = _denoiser(loaded, component).schedule_shifts
        return [float(shifts["video"]), float(shifts["audio"])]

    @staticmethod
    def canvases(loaded, component: str) -> list[int]:
        """``[width, height]`` of each canvas the denoiser offers."""
        return [
            int(extent)
            for canvas in _denoiser(loaded, component).canvases
            for extent in (canvas.width, canvas.height)
        ]

    @staticmethod
    def max_sequence_rows(loaded, component: str) -> int:
        """The most sequence rows of one evaluation; zero when unbounded."""
        return int(_denoiser(loaded, component).max_sequence_rows or 0)

    @staticmethod
    def condition_tiles(loaded, component: str) -> list[int]:
        """``[rows, frames, height, width]`` of a condition tile, or empty."""
        tiles = _denoiser(loaded, component).condition_tiles
        if tiles is None:
            return []
        return [int(tiles.rows), *(int(side) for side in tiles.video)]

    @staticmethod
    def sample_dtype(loaded, component: str) -> str:
        """The dtype of the predicted samples."""
        return _dtype(_denoiser(loaded, component).prediction_dtype)

    @staticmethod
    def text_condition_width(loaded, component: str) -> int:
        """Columns of the retained conditioning."""
        return int(_denoiser(loaded, component).text_condition_width)

    @staticmethod
    def legal_frame_count(loaded, component: str, frames: int) -> int:
        """The smallest frame count the denoiser generates for ``frames``."""
        return int(_denoiser(loaded, component).legal_frame_count(int(frames)))

    @staticmethod
    def layout_size(loaded, component: str, size, conditions, spans):
        """The smallest layout holding a request size."""
        network = _denoiser(loaded, component)
        return _describe(
            network.layout_size(_size(network, size, conditions, spans))
        )

    @staticmethod
    def holds(loaded, component: str, layout, size, conditions, spans) -> bool:
        """Whether ``layout`` holds a request size."""
        network = _denoiser(loaded, component)
        return bool(
            network.holds(
                _layout(network, layout),
                _size(network, size, conditions, spans),
            )
        )

    @staticmethod
    def condition_layout(loaded, component: str, layout, rows: int):
        """``layout`` widened so its condition region holds ``rows``."""
        network = _denoiser(loaded, component)
        return _describe(
            network.condition_layout(_layout(network, layout), int(rows))
        )

    @staticmethod
    def latent_shape(loaded, component: str, modality: str, layout):
        """The complete sample shape of one modality in ``layout``."""
        network = _denoiser(loaded, component)
        shape = network.latent_shape(str(modality), _layout(network, layout))
        return [int(extent) for extent in shape]

    @staticmethod
    def state_names(loaded, component: str, layout) -> list[str]:
        """The state fields of ``layout`` on this rank, in state order.

        The sample modalities are among them; every other field is a state
        table.
        """
        network = _denoiser(loaded, component)
        return list(network.state_buffers(_layout(network, layout)))

    @staticmethod
    def state_dtypes(loaded, component: str, layout) -> list[str]:
        """Each state field's dtype, in state order."""
        network = _denoiser(loaded, component)
        fields = network.state_buffers(_layout(network, layout))
        return [_dtype(config.dtype) for config in fields.values()]

    @staticmethod
    def state_elements(loaded, component: str, layout) -> list[int]:
        """Each state field's elements on this rank, in state order."""
        network = _denoiser(loaded, component)
        fields = network.state_buffers(_layout(network, layout))
        return [_elements(config.shape) for config in fields.values()]

    @staticmethod
    def sample_rows(loaded, component: str, layout) -> list[int]:
        """``[rows, width, first, stop]`` of each modality's sample.

        ``rows`` and ``width`` shape the complete sample; ``[first, stop)``
        are this rank's rows of it.
        """
        network = _denoiser(loaded, component)
        resolved = _layout(network, layout)
        outputs = network.output_layout(resolved)
        result = []
        for name in network.modalities:
            shape = network.latent_shape(name, resolved)
            local = outputs[name].local_slice[0]
            result.extend(
                [
                    int(shape[0]),
                    int(shape[1]),
                    int(local.start),
                    int(local.stop),
                ]
            )
        return result

    @staticmethod
    def noise_elements(loaded, component: str, layout) -> list[int]:
        """Elements of each modality's native draw, in modality order."""
        network = _denoiser(loaded, component)
        resolved = _layout(network, layout)
        return [
            _elements(network.noise_shape(name, resolved))
            for name in network.modalities
        ]

    @staticmethod
    def condition_noise_elements(loaded, component: str, layout) -> int:
        """FP32 elements bounding the condition draws of ``layout``."""
        network = _denoiser(loaded, component)
        return int(network.condition_noise_capacity(_layout(network, layout)))

    @staticmethod
    def text_condition_rows(loaded, component: str, layout) -> int:
        """Rows of the retained conditioning of ``layout``."""
        network = _denoiser(loaded, component)
        return int(network.text_condition_rows(_layout(network, layout)))

    @staticmethod
    def workspace_elements(loaded, component: str, layout) -> int:
        """Elements of the workspace ``layout`` takes on this rank."""
        network = _denoiser(loaded, component)
        buffers = cast(_Buffered, network).workspace_buffers(
            _layout(network, layout)
        )
        return sum(_elements(config.shape) for config in buffers.values())

    @staticmethod
    def products(
        loaded,
        component: str,
        frames: int,
        tokens: int,
        rows: int,
        dtype: str,
        canvases: Sequence[int],
    ) -> Outputs:
        """Products the model's components publish, bounded for any request.

        ``frames`` is the longest admitted video, ``tokens`` the prompt
        capacity, ``rows`` the condition capacity (zero without conditioned
        tasks), ``dtype`` the model dtype the text encoder runs at and
        ``canvases`` the flat ``[width, height]`` canvases the deployment
        prepares. Each video product bounds the video of ``frames`` frames
        at every one of them. Returns each publishing component's products
        (:class:`Outputs`).
        """
        root = loaded.module
        network = _denoiser(loaded, component)
        decoder = _require(loaded, VideoDecoder)
        audio = _require(loaded, AudioDecoder)
        clock = _require(loaded, VideoPostprocessor)
        video_encoder = capability(root, VideoEncoder)
        audio_encoder = capability(root, AudioEncoder)
        vision = capability(root, PatchEncoder)
        text = capability(root, TextEncoder)
        frames = int(frames)
        flat = [int(value) for value in canvases]
        sizes = [
            network.layout_size(
                network.make_size(
                    frames, int(tokens), canvas=image.Config(height, width)
                )
            )
            for width, height in zip(flat[::2], flat[1::2], strict=True)
        ]
        result: dict[str, list[dict[str, Any]]] = {}

        def add(owner, record):
            result.setdefault(owner, []).append(record)

        names = {"video": "video_latents", "audio": "audio_latents"}
        for name in network.modalities:
            shapes = [network.latent_shape(name, size) for size in sizes]
            longest = max(shape[0] for shape in shapes)
            add(
                component,
                {
                    "name": names.get(name, name),
                    "dtype": _dtype(network.prediction_dtype),
                    "shape": [int(longest), int(shapes[0][1])],
                    "variable_axes": [0],
                    "raster_axes": None,
                },
            )
        lengths = [clip.stop - clip.start for clip in decoder.clips(frames)]
        decoders = [
            name
            for name, entry in loaded.description.entries.items()
            if any(
                method == "decode" and isinstance(module, VideoDecoder)
                for module, method in _called(loaded, entry)
            )
        ]
        for name in decoders:
            add(
                name,
                {
                    "name": "video_units",
                    "dtype": "uint8",
                    "shape": [
                        len(lengths),
                        max(lengths),
                        max(int(size.canvas.height) for size in sizes),
                        max(int(size.canvas.width) for size in sizes),
                        3,
                    ],
                    "variable_axes": [0],
                    "raster_axes": [2, 3],
                },
            )
        if text is not None:
            layouts = text.output_layout(
                int(tokens), getattr(torch, str(dtype).removeprefix("torch."))
            )
            for name, layout in layouts.items():
                add(_owner(loaded, TextEncoder), _product(name, layout))
        samples = int(audio.track_samples(frames, clock.frame_rate))
        track = audio.output_layout(samples)["audio"]
        add(_owner(loaded, AudioDecoder), _product("audio_samples", track))

        if rows and video_encoder is not None and audio_encoder is not None:
            sample = video_encoder.output_layout(
                video.Config(1, image.Config(32, 32))
            )["video"]
            owner = _owner(loaded, VideoEncoder)
            add(owner, _rows("condition_video_latents", rows, sample))
            sample = audio_encoder.output_layout(audio_encoder.latent_rate)[
                "audio"
            ]
            add(owner, _rows("condition_audio_latents", rows, sample))
        if rows and vision is not None:
            add(
                _owner(loaded, PatchEncoder),
                _product(
                    "vision_features",
                    vision.features_layout(int(tokens))["features"],
                ),
            )
        return Outputs.of(result)

    @staticmethod
    def reader_products(
        loaded, component: str, frames: int, tokens: int, rows: int
    ) -> Outputs:
        """The media reader's products for a deployment serving conditions.

        A condition's pixels are its rows times the pixels per row of its
        frame count, one the denoiser generates or a single frame; the bound
        is the condition rows times the largest such ratio, read from the
        video encoder's own layout (it does not depend on the raster). A
        track's stereo rows are twice its latent frames, each frame
        ``latent_rate`` samples. Patch rows bound every vision block of the
        presentation.
        """
        root = loaded.module
        network = _denoiser(loaded, component)
        video_encoder = capability(root, VideoEncoder)
        audio_encoder = capability(root, AudioEncoder)
        vision = capability(root, PatchEncoder)
        if video_encoder is None or audio_encoder is None or vision is None:
            raise SetupError(
                "media reading requires the condition encoders and a vision "
                "encoder"
            )
        counts = [1, int(network.legal_frame_count(1))]
        while (count := int(network.legal_frame_count(counts[-1] + 1))) <= int(
            frames
        ):
            counts.append(count)
        patch = image.Config(32, 32)
        per_row = max(
            -(
                -count
                * patch.height
                * patch.width
                // video_encoder.output_layout(video.Config(count, patch))[
                    "video"
                ].shape[0]
            )
            for count in counts
        )
        rows = int(rows)
        products = [
            {
                "name": "condition_pixels",
                "dtype": "uint8",
                "shape": [rows * per_row, 3],
                "variable_axes": [0],
                "raster_axes": None,
            },
            {
                "name": "condition_samples",
                "dtype": "float32",
                "shape": [rows // 2 * int(audio_encoder.latent_rate), 2],
                "variable_axes": [0],
                "raster_axes": None,
            },
            _product("vision_pixels", vision.pixels_layout(int(tokens))),
        ]
        return Outputs.of({"media_reader": products})

    # Layouts.

    def prepare_layout(self, layout: Sequence[int]) -> _Layout:
        """Prepare one layout's constants and return them as a handle.

        The largest layout uses the constants and workspace the runner was
        prepared with; any other allocates its own constants and views the
        shared workspace. Captured graphs address a layout's constants, so
        the caller keeps the handle while any graph of the layout lives.
        """
        network = self._network
        buffered = cast(_Buffered, network)
        context = self._context
        resolved = _layout(network, layout)
        owned = None
        if resolved == self._maximum:
            constants, workspace = context.constants, context.workspace_views
        else:
            requirements = buffered.constant_buffers(resolved)
            owned = TensorBuffers.allocate(requirements, device=self._device)
            constants = owned.view(requirements)
            with context.activate():
                buffered.prepare_constants(resolved, out=constants)
            workspace = TensorBuffers.from_tensors(
                dict(context.workspace_views)
            ).view(buffered.workspace_buffers(resolved))
        timesteps = {
            name: torch.empty((1,), dtype=torch.float32, device=self._device)
            for name in self._modalities
        }
        return _Layout(
            resolved,
            network.state_buffers(resolved),
            (
                network.text_condition_rows(resolved),
                network.text_condition_width,
            ),
            constants,
            workspace,
            timesteps,
            owned,
        )

    # Forwards.

    def draw_noise(self, *arguments: Any) -> None:
        """Fill a request's noise from its seed.

        A video denoiser takes ``(seed, layout, size, conditions, spans,
        condition_noise, *noise)``; an image denoiser takes ``(seed, index,
        height, width, noise)`` and draws image ``index`` of the request's
        seed at the image's native draw shape into the leading elements of
        ``noise``, a tensor of the samples' device and dtype. ``seed``
        carries the unsigned 64-bit seed's bits as a signed integer.
        """
        self._draw_noise(*arguments)

    def prepare_latents(self, *arguments: Any) -> None:
        """Write a request's initial samples.

        A video denoiser takes ``(layout, size, conditions, spans,
        *tensors)``; an image denoiser takes ``(noise, sample, units,
        padded, height, width)``: it zeroes ``samples[sample:sample +
        padded]`` and writes the image's initial latent, patched from its
        drawn ``noise``, into the first ``units`` of those rows.
        """
        self._prepare_latents(*arguments)

    def encode_conditions(self, *arguments: Any) -> None:
        """Write a request's conditions where its denoising reads them.

        A video denoiser takes ``(layout, size, conditions, spans,
        condition_noise, out, counts, *rows)``. An image denoiser takes
        the arguments of :meth:`denoise` and ``*features``, one latent
        feature product tensor per row: row ``i`` reads ``features[i]``
        instead of samples, at timestep zero, and writes its keys and
        values after its slot's visible tokens; its prediction is computed
        and unused.
        """
        self._encode_conditions(*arguments)

    def denoise(self, *arguments: Any) -> None:
        """Evaluate the denoiser.

        A video denoiser takes ``(layout, step, text_condition,
        *tensors)`` and advances a request by one solver step in place.

        An image denoiser takes ``(row_columns, positions, timesteps, rows,
        queries, prefixes, starts, widths)`` and predicts every row:
        ``row_columns`` packs the rows' slots, lengths and selected page
        counts (``gather_request_rows``), from which the forward gathers
        the rows' attention metadata on the device; ``positions`` is
        ``[3, T]`` int64 and ``timesteps`` ``[R, 1]`` float32; ``rows``
        are the flattened rows; ``queries`` and ``prefixes`` mirror the
        rows' lengths on the host, ``starts`` each windowed table's first
        pages (one per row and table) and ``widths`` each table's page
        width. The host mirrors serve an eager forward and the capture; a
        launched graph reads the device columns.
        """
        self._denoise(*arguments)

    def _condition_noise(
        self, flat: torch.Tensor, size
    ) -> tuple[torch.Tensor, ...]:
        # The condition draws occupy the leading elements one after
        # another, each leading with its batch of one.
        views, offset = [], 0
        for shape in self._network.condition_noise_shapes(size):
            count = _elements(shape)
            views.append(flat[offset : offset + count].view(tuple(shape)))
            offset += count
        return tuple(views)

    def _video_draw_noise(
        self,
        seed: int,
        layout: _Layout,
        size: Sequence[int],
        conditions: Sequence[int],
        spans: Sequence[int],
        condition_noise: torch.Tensor,
        *noise: torch.Tensor,
    ) -> None:
        """Fill a request's noise from its seed.

        The seed's stream draws each condition's noise, then each
        modality's, every draw at its native shape: the shapes fix how the
        stream is consumed. ``condition_noise`` and ``noise`` (one flat
        tensor per modality) are host tensors; ``seed`` carries the
        unsigned 64-bit seed's bits as a signed integer.
        """
        network = self._network
        resolved = _size(network, size, conditions, spans)
        with torch.inference_mode():
            normal_noise(
                (int(seed) & 0xFFFF_FFFF_FFFF_FFFF,),
                out=(
                    *self._condition_noise(condition_noise, resolved),
                    *(
                        _view(
                            tensor, network.noise_shape(name, layout.size)
                        ).unsqueeze(0)
                        for name, tensor in zip(
                            self._modalities, noise, strict=True
                        )
                    ),
                ),
            )

    def _video_prepare_latents(
        self,
        layout: _Layout,
        size: Sequence[int],
        conditions: Sequence[int],
        spans: Sequence[int],
        *tensors: torch.Tensor,
    ) -> None:
        """Write a request's initial samples and state tables.

        ``tensors`` are each modality's noise (in modality order), then
        every state field's destination in state order (see
        :meth:`state_names`): a modality's field receives its initial
        sample, patched from the noise and restricted to this rank's rows,
        and every other field its table for the request's size. All are
        host tensors; the call does no device work.
        """
        network = self._network
        resolved = _size(network, size, conditions, spans)
        fields = layout.fields
        count = len(self._modalities)
        noise = {
            name: _view(
                tensor, network.noise_shape(name, layout.size)
            ).unsqueeze(0)
            for name, tensor in zip(
                self._modalities, tensors[:count], strict=True
            )
        }
        sources = {
            name: _view(tensor, config.shape)
            for (name, config), tensor in zip(
                fields.items(), tensors[count:], strict=True
            )
        }
        with torch.inference_mode():
            network.prepare_state(
                (resolved,),
                layouts=(layout.size,),
                out={
                    name: source
                    for name, source in sources.items()
                    if name not in self._modalities
                },
            )
            network.prepare_latents(
                (layout.size,),
                noise=noise,
                state={
                    name: sources[name].unsqueeze(0)
                    for name in self._modalities
                },
                constants=layout.constants,
                workspace=layout.workspace,
            )

    def _video_encode_conditions(
        self,
        layout: _Layout,
        size: Sequence[int],
        conditions: Sequence[int],
        spans: Sequence[int],
        condition_noise: torch.Tensor,
        out: torch.Tensor,
        counts: Sequence[int],
        *rows: torch.Tensor,
    ) -> None:
        """Write a request's encoded conditions into its conditioning.

        The condition latents come in request order, a condition's visual
        rows before its audio rows. Latent ``i`` is the next ``counts[i]``
        tensors of ``rows``, which together hold its rows in order: a
        condition encoded over several rounds has one tensor per round.
        ``condition_noise`` holds the request's condition draws
        (:meth:`draw_noise`); ``out`` is the request's retained
        conditioning, whose rows past the prompt's receive the conditions.
        """
        network = self._network
        resolved = _size(network, size, conditions, spans)
        latents, cursor = [], 0
        for count in (int(value) for value in counts):
            parts = rows[cursor : cursor + count]
            latents.append(parts[0] if count == 1 else torch.cat(parts))
            cursor += count
        target = _view(
            out,
            layout.text_condition,
        )
        with torch.inference_mode(), self._context.activate():
            network.encode_conditions(
                resolved,
                layout.size,
                latents=tuple(latents),
                noise=self._condition_noise(condition_noise, resolved),
                out=target,
            )

    def _video_denoise(
        self,
        layout: _Layout,
        step: torch.Tensor,
        text_condition: torch.Tensor,
        *tensors: torch.Tensor,
    ) -> None:
        """Evaluate one solver step of a request in place.

        ``step`` is the ``[1]`` int64 evaluation index; ``text_condition``
        the request's retained conditioning; ``tensors`` every state field
        in state order (see :meth:`state_names`): each modality's sample of
        this rank, which the step updates toward the next timestep, and
        the request's state tables. The step's endpoints are gathered on
        the device from the schedule, so one captured step serves every
        evaluation.
        """
        network = self._network
        fields = layout.fields
        views = {
            name: _view(tensor, config.shape)
            for (name, config), tensor in zip(
                fields.items(), tensors, strict=True
            )
        }
        state = {
            name: view
            for name, view in views.items()
            if name not in self._modalities
        }
        state["text_condition"] = _view(
            text_condition,
            layout.text_condition,
        )
        inputs = network.bind_inputs(
            latents={
                name: (LatentInput(views[name], layout.timesteps[name][0]),)
                for name in self._modalities
            },
            sizes=(layout.size,),
            step=step,
            text_features=(state["text_condition"],),
        )
        with torch.inference_mode(), self._context.activate():
            for name in self._modalities:
                torch.index_select(
                    self._schedules[name].timesteps,
                    0,
                    step,
                    out=layout.timesteps[name],
                )
            DenoisingStep(
                network,
                inputs,
                self._schedules,
                state,
                layout.constants,
                layout.workspace,
            )()

    # Image denoisers.

    # Facts of a loaded model's image denoiser, answered before a runner
    # of it exists.

    @staticmethod
    def image_layout(loaded) -> list[int]:
        """The image denoiser's latent layout; empty without one.

        ``[patch_size, latent_channels, downsample, framing_tokens,
        max_sequence_tokens, rope_advance]``: ``downsample`` is pixels per
        latent patch on each axis, ``framing_tokens`` the tokens framing a
        generated image, ``max_sequence_tokens`` the longest image sequence
        in latent patch rows, framing excluded, and ``rope_advance`` the
        language positions a generated image spans in its conversation.
        """
        denoiser = _image_denoiser(loaded)
        if denoiser is None:
            return []
        return [
            int(denoiser.patch_size),
            int(denoiser.latent_channels),
            int(denoiser.downsample),
            int(denoiser.framing_tokens),
            int(denoiser.max_sequence_tokens),
            int(denoiser.rope_advance),
        ]

    @staticmethod
    def image_unconditional(loaded) -> str:
        """What the image-unconditional guidance branch conditions on."""
        return str(_image_denoiser(loaded).image_unconditional.value)

    @staticmethod
    def prediction_dtype(loaded) -> str:
        """The dtype of the image denoiser's predictions."""
        return _dtype(_image_denoiser(loaded).prediction_dtype)

    @staticmethod
    def guidance_prefix(
        loaded, source: str, image_prompt: str, negative_prompt: str
    ) -> list[int]:
        """The prefix a guidance branch of ``source`` conditions on.

        The model's prompt framing and tokenizer resolve it (see
        ``uniserve.processing.resolve_prefix``).

        Returns:
            ``[reuses, *tokens]``: whether the branch reuses the request's
            own context, then the prefix tokens.

        Raises:
            ValueError: A prompt override for a model without prompt
                framing.
        """

        def tokenizer():
            return (
                None
                if loaded.tokenizer is None
                else _tokenizer(loaded.tokenizer)
            )

        tokens, reuses = resolve_prefix(
            loaded.flow_prompt,
            tokenizer,
            source,
            image_prompt=image_prompt,
            negative_prompt=negative_prompt,
        )
        return [int(reuses), *(int(token) for token in tokens)]

    def plans_attention(self) -> bool:
        """Whether an attention operator plans each launch on the host.

        A captured :meth:`denoise` of such a model is launched after
        :meth:`plan_denoise` built the plans its graph reads.
        """
        return self._context.plans_attention()

    def plan(
        self,
        steps: int,
        shift: float,
        text_scale: float,
        image_scale: float,
        interval_low: float,
        interval_high: float,
        renorm: str,
        renorm_min: float,
    ) -> list:
        """What one image trajectory evaluates at each solver step.

        The trajectory's schedules and guidance are made from these
        parameters; :meth:`step` takes them again. A non-positive ``shift``
        keeps the denoiser's default shift.

        Returns:
            ``[timesteps, branches]``: each solver step's model timestep and
            the mask of guidance branches it evaluates (bit 0 conditioned, 1
            text-unconditional, 2 image-unconditional).
        """
        schedules, guidance = _trajectory(
            self._network,
            steps,
            shift,
            text_scale,
            image_scale,
            interval_low,
            interval_high,
            renorm,
            renorm_min,
        )
        schedule = schedules["image"]
        masks = [
            sum(_BITS[branch] for branch in guidance.branches(schedule, index))
            for index in range(schedule.num_steps)
        ]
        times = [float(value) for value in schedule.timesteps.tolist()]
        return [times, masks]

    def _image_draw_noise(
        self,
        seed: int,
        index: int,
        height: int,
        width: int,
        noise: torch.Tensor,
    ) -> None:
        size = image.Config(int(height), int(width))
        shape = (1, *self._network.noise_shape("image", size))
        with torch.inference_mode(), torch.cuda.stream(self._context.stream):
            normal_noise(
                (flow_noise_seed(int(seed), int(index)),),
                out=(_view(noise, shape),),
            )

    def _image_prepare_latents(
        self,
        noise: torch.Tensor,
        sample: int,
        units: int,
        padded: int,
        height: int,
        width: int,
    ) -> None:
        network, samples = self._network, self._samples
        sample, units, padded = int(sample), int(units), int(padded)
        size = image.Config(int(height), int(width))
        out = samples[sample : sample + units]
        if (
            padded < units
            or sample + padded > samples.shape[0]
            or tuple(out.shape) != network.latent_shape("image", size)
        ):
            raise ValueError(
                "image trajectory storage must have its canonical patch shape"
            )
        shape = (1, *network.noise_shape("image", size))
        with torch.inference_mode(), torch.cuda.stream(self._context.stream):
            # Physical page padding past the trajectory's rows stays zero.
            samples[sample : sample + padded].zero_()
            network.prepare_latents(
                (size,),
                noise={"image": _view(noise, shape)},
                state={"image": out.unsqueeze(0)},
                constants={},
                workspace={},
            )

    def _attention(
        self,
        row_columns: torch.Tensor,
        count: int,
        tokens: int,
        queries: Sequence[int],
        prefixes: Sequence[int],
        starts: Sequence[int],
        widths: Sequence[int],
        writes: bool,
    ) -> AttentionBatch:
        """Gather ``count`` rows of ``tokens`` tokens and view them.

        The gather fills the block tables, start pages, lengths, offsets
        and slot mappings from the slot tables in one launch. Rows that
        write keys and values read paged inputs with a slot mapping;
        read-only rows read their segments without writing. No row is
        causal.
        """
        metadata = self._attention_metadata
        metadata.gather(row_columns, count, tokens, widths)
        return metadata.batch(
            count,
            tokens,
            widths,
            queries,
            prefixes,
            starts,
            (False,) * count,
            writes=writes,
        )

    def _evaluate(
        self,
        rows: Sequence[tuple[int, ...]],
        positions: torch.Tensor,
        timesteps: torch.Tensor,
        attention: AttentionBatch,
        features: Sequence | None = None,
    ) -> None:
        """Run the network over ``rows`` and write their predictions."""
        network, context = self._network, self._context
        latents = []
        for index, (offset, units, *_) in enumerate(rows):
            values = (
                self._samples[offset : offset + units]
                if features is None
                else features[index]
            )
            latents.append(LatentInput(values, timesteps[index : index + 1]))
        # The denoiser binds its inputs inside the forward: whatever it
        # derives from the samples (conditioning pixels) follows each
        # launch's samples.
        typed = network.bind_inputs(
            latents={"image": tuple(latents)},
            sizes=tuple(image.Config(row[2], row[3]) for row in rows),
            step=self._step,
            positions=tuple(
                positions[:, row[4] : row[4] + row[5]] for row in rows
            ),
            sequence_lengths=tuple(row[5] for row in rows),
            attention=attention,
        )
        result = network(
            typed,
            state={},
            constants=context.constants,
            workspace=context.workspace_views,
        )["image"]
        # Only the last pipeline stage predicts; its broadcast gives every
        # stage the same values. A mesh without ``pp`` broadcasts locally.
        pipeline = self._pipeline
        for prediction, row in zip(result, rows, strict=True):
            target = self._predictions[row[6] : row[6] + row[1]]
            if prediction is not None:
                target.copy_(prediction.tensor.reshape(target.shape))
            pipeline.broadcast(target, src=pipeline.size - 1)

    def plan_denoise(
        self,
        row_columns: torch.Tensor,
        positions: torch.Tensor,
        timesteps: torch.Tensor,
        rows: Sequence[int],
        queries: Sequence[int],
        prefixes: Sequence[int],
        starts: Sequence[int],
        widths: Sequence[int],
    ) -> None:
        """Plan the attention launches of the next captured :meth:`denoise`.

        Operators that plan on the host (:meth:`plans_attention`) read the
        gathered metadata and its host mirrors here; the graph then reads
        the plans. The arguments are those of :meth:`denoise`.
        """
        described = _image_rows(rows)
        tokens = sum(row[5] for row in described)
        with torch.inference_mode(), self._context.activate():
            attention = self._attention(
                row_columns,
                len(described),
                tokens,
                queries,
                prefixes,
                starts,
                widths,
                False,
            )
            self._context.bind_attention(attention)

    def _image_denoise(
        self,
        row_columns: torch.Tensor,
        positions: torch.Tensor,
        timesteps: torch.Tensor,
        rows: Sequence[int],
        queries: Sequence[int],
        prefixes: Sequence[int],
        starts: Sequence[int],
        widths: Sequence[int],
    ) -> None:
        described = _image_rows(rows)
        tokens = sum(row[5] for row in described)
        with torch.inference_mode(), self._context.activate():
            attention = self._attention(
                row_columns,
                len(described),
                tokens,
                queries,
                prefixes,
                starts,
                widths,
                False,
            )
            if not capturing(self._device):
                self._context.bind_attention(attention)
            self._evaluate(described, positions, timesteps, attention)

    def _image_encode_conditions(
        self,
        row_columns: torch.Tensor,
        positions: torch.Tensor,
        timesteps: torch.Tensor,
        rows: Sequence[int],
        queries: Sequence[int],
        prefixes: Sequence[int],
        starts: Sequence[int],
        widths: Sequence[int],
        *features: torch.Tensor,
    ) -> None:
        described = _image_rows(rows)
        if len(features) != len(described):
            raise ValueError("latent conditioning rows and features must align")
        tokens = sum(row[5] for row in described)
        with torch.inference_mode(), self._context.activate():
            attention = self._attention(
                row_columns,
                len(described),
                tokens,
                queries,
                prefixes,
                starts,
                widths,
                True,
            )
            self._context.bind_attention(attention)
            # Conditioning runs at timestep zero.
            timesteps[: len(described)].zero_()
            self._evaluate(described, positions, timesteps, attention, features)

    def step(
        self,
        rows: Sequence[int],
        parameters: Sequence[float],
        renorms: Sequence[str],
    ) -> None:
        """Advance each image's samples by one solver step.

        ``rows`` holds one ``[sample, units, slot, step, prediction,
        branches]`` row per image, flattened: the solver combines the
        image's branch predictions (consecutive from ``prediction``, in
        evaluation order) with its trajectory's guidance and updates
        ``samples[sample:sample + units]`` in place toward the next
        timestep. ``parameters`` holds each image's ``[steps, shift,
        text_scale, image_scale, interval_low, interval_high,
        renorm_min]``, flattened, and ``renorms`` its renormalization: the
        parameters :meth:`plan` was given, which make the trajectory's
        schedules and guidance.
        """
        network = self._network
        described = [
            tuple(int(value) for value in rows[start : start + 6])
            for start in range(0, len(rows), 6)
        ]
        settings = [
            tuple(float(value) for value in parameters[start : start + 7])
            for start in range(0, len(parameters), 7)
        ]
        if not len(described) == len(settings) == len(renorms):
            raise ValueError(
                "integration rows and trajectory parameters must align"
            )
        with torch.inference_mode(), torch.cuda.stream(self._context.stream):
            for (sample, units, slot, index, first, mask), (
                steps,
                shift,
                text_scale,
                image_scale,
                low,
                high,
                renorm_min,
            ), renorm in zip(described, settings, renorms, strict=True):
                schedules, guidance = _trajectory(
                    network,
                    int(steps),
                    shift,
                    text_scale,
                    image_scale,
                    low,
                    high,
                    str(renorm),
                    renorm_min,
                )
                schedule = schedules["image"]
                branches = guidance.branches(schedule, index)
                if sum(_BITS[branch] for branch in branches) != mask:
                    raise ValueError(
                        "guidance branches disagree with the evaluated rows"
                    )
                values = self._samples[sample : sample + units]
                outputs = {
                    branch: self._predictions[
                        first + position * units : first
                        + (position + 1) * units
                    ]
                    for position, branch in enumerate(branches)
                }
                guided = guidance.combine(outputs, schedule, index)
                advance_(
                    network,
                    {
                        "image": (
                            LatentInput(values, self._slot_timesteps[slot]),
                        )
                    },
                    {"image": (guided,)},
                    schedules,
                    schedule.step(index),
                )

    def kernels(self) -> list[str]:
        """The kernels the runner's operators selected, as JSON objects."""
        return kernel_records(self._context.kernels())

    def close(self, aborted: bool) -> None:
        """Release the runner's operators and workspace.

        An aborted close keeps everything device work may still address.
        """
        self._context.close(aborted=bool(aborted))
