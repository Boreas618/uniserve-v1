"""The forwards of a video latent denoiser: :class:`DenoiserRunner`.

One runner serves one video denoiser on one stream. A request's state lives
in tensors the caller owns: its latent samples, its state tables, its
retained conditioning and its noise. Every method names the tensors it
reads and writes as parameters; a tensor wider than the layout needs is
viewed through its leading elements.

- :meth:`DenoiserRunner.draw_noise` fills a request's noise tensors from
  its seed with the library's native draw.
- :meth:`DenoiserRunner.prepare_latents` turns that noise and the request's
  size into its initial samples and state tables.
- :meth:`DenoiserRunner.encode_conditions` writes a request's encoded
  conditions into its retained conditioning.
- :meth:`DenoiserRunner.denoise` evaluates one solver step in place. Its
  shapes are fixed by its layout and every changing value is the contents
  of a tensor, so it is captured into one CUDA graph per layout.

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

The static methods answer a deployment's sizing questions from the model's
configuration alone, so a rank without the denoiser's weights asks them
too.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch

from uniserve_models.bindings.context import ExecutionContext, TensorBuffers
from uniserve_models.diffusion import DenoisingStep, normal_noise
from uniserve_models.interface.errors import SetupError
from uniserve_models.interface.loading import capability
from uniserve_models.interface.workspace import execution_context
from uniserve_models.media import image, video
from uniserve_models.model import (
    AudioDecoder,
    AudioEncoder,
    Condition,
    ConditionRole,
    LatentInput,
    PatchEncoder,
    TextEncoder,
    VideoDecoder,
    VideoDenoiser,
    VideoEncoder,
    VideoPostprocessor,
)
from uniserve_models.tensors import BufferConfig

__all__ = ["DenoiserRunner"]

_ROLES = (
    ConditionRole.FIRST_FRAME,
    ConditionRole.LAST_FRAME,
    ConditionRole.REFERENCE,
)


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
    """
    called = []
    for point in entry.points:
        prefix, _, method = point.method.rpartition(".")
        path = ".".join(part for part in (entry.component, prefix) if part)
        called.append((loaded.module.get_submodule(path), method))
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


def _dtype(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


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
    """The forwards of one video denoiser on one stream.

    Args:
        loaded: The loaded model.
        component: The declared component of the video denoiser.
        stream: The CUDA stream handle every forward enqueues on.
        attention: The attention backend the model loaded with.
        allocate: The caller's allocation function for workspace torch
            cannot provide.
        maximum: The largest layout, whose capacity the operators and the
            workspace are prepared for.
    """

    def __init__(
        self,
        loaded,
        component: str,
        stream: int,
        attention: str,
        allocate,
        maximum: Sequence[int],
    ) -> None:
        network = _denoiser(loaded, str(component))
        self._network = network
        self._device = next(network.parameters()).device
        self._context: ExecutionContext = execution_context(
            network, int(stream), str(attention), allocate, "denoiser"
        )
        self._maximum = _layout(network, maximum)
        self._context.prepare(self._maximum)
        self._schedules = network.make_schedules(
            network.num_steps, shift=None, device=self._device
        )
        self._modalities = tuple(network.modalities)

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
    def condition_rows(loaded, component: str, size, conditions, spans) -> int:
        """Condition rows of a request size."""
        network = _denoiser(loaded, component)
        return int(_size(network, size, conditions, spans).condition_rows)

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

    # FIXME: the two product tables below still cross as records. Each
    # product's shape is derivable from the scalar methods of the runners;
    # the tables remain until the rank composes its product declarations
    # from them.

    @staticmethod
    def products(
        loaded,
        component: str,
        frames: int,
        tokens: int,
        rows: int,
        dtype: str,
        canvases: Sequence[int],
    ) -> dict[str, Any]:
        """Products the model's components publish, bounded for any request.

        ``frames`` is the longest admitted video, ``tokens`` the prompt
        capacity, ``rows`` the condition capacity (zero without conditioned
        tasks), ``dtype`` the model dtype the text encoder runs at and
        ``canvases`` the flat ``[width, height]`` canvases the deployment
        prepares. Each video product bounds the video of ``frames`` frames
        at every one of them. Returns each publishing component's products.
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
        lengths = [
            window.stop - window.start
            for window in decoder.frame_slices(frames)
        ]
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
        return result

    @staticmethod
    def reader_products(
        loaded, component: str, frames: int, tokens: int, rows: int
    ) -> list[Any]:
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
        return [
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

    def draw_noise(
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

    def prepare_latents(
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

    def encode_conditions(
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

    def denoise(
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

    def kernels(self) -> list[dict]:
        """The kernels the runner's operators selected."""
        return list(self._context.kernels())

    def close(self, aborted: bool) -> None:
        """Release the runner's operators and workspace.

        An aborted close keeps everything device work may still address.
        """
        self._context.close(aborted=bool(aborted))
