"""Video generation: model functions and the bindings of its components.

A video model's request runs as media calls the worker schedules on its
components: latent preparation and one denoising call per solver step on the
denoiser, unit rounds on the video decoder (with the RGB post-processor),
the audio decoder and the condition latent encoder. The worker owns every
request, slot, page, buffer, graph and stream around these calls; the
functions here compute on the tensors and integers they are handed.

Interface entries:

- ``video.model(loaded, denoiser) -> {...}``: the model's sizing rules over
  integer descriptors, answered from its configuration (a host rank's
  weight-less description serves them too).
- ``video.bind_denoiser(loaded, request) -> {...}``: denoising steps,
  request preparation, condition encoding and the text conditioner, bound
  once over worker-owned buffers.
- ``video.bind_decoder(loaded, request) -> {...}``: one media unit's decode
  and RGB reconstruction.
- ``video.bind_audio_decoder(loaded, request) -> {...}``: a round of audio
  units.
- ``video.bind_latent_encoder(loaded, request) -> {...}``: condition video
  units and audio tracks.

Descriptors are integer arrays:

- a request size: ``[frames, width, height, prompt_tokens]``;
- conditions: ``[[role, frames, width, height, audio_samples], ...]`` in
  request order, ``role`` 0 for the first frame, 1 for the last frame and
  2 for a reference, ``frames`` 0 for a condition without pixels;
- vision spans: ``[start0, stop0, start1, stop1, ...]``, the presented
  prompt's vision-token ranges;
- a layout: ``[frames, width, height, text_rows, condition_rows]``.

A binding is created for one CUDA stream (an integer handle the worker
owns); every call it makes enqueues its device work there.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from uniserve_models.bindings.context import ExecutionContext, TensorBuffers
from uniserve_models.diffusion import DenoisingStep, normal_noise
from uniserve_models.media import image, video
from uniserve_models.model import (
    AudioDecoder,
    AudioEncoder,
    Condition,
    ConditionRole,
    LatentInput,
    PatchEncoder,
    VideoDecoder,
    VideoDenoiser,
    VideoEncoder,
    VideoPostprocessor,
)

from . import graphs
from .errors import SetupError, translate
from .workspace import RuntimeWorkspace

_ROLES = (
    ConditionRole.FIRST_FRAME,
    ConditionRole.LAST_FRAME,
    ConditionRole.REFERENCE,
)


def functions() -> dict[str, Callable[..., Any]]:
    return {
        "video.model": model,
        "video.bind_denoiser": bind_denoiser,
        "video.bind_decoder": bind_decoder,
        "video.bind_audio_decoder": bind_audio_decoder,
        "video.bind_latent_encoder": bind_latent_encoder,
    }


def _unique(root: nn.Module, kind: type) -> nn.Module | None:
    """The model's unique module of ``kind``, or None."""
    found = list(
        {
            id(module): module
            for module in root.modules()
            if isinstance(module, kind)
        }.values()
    )
    if len(found) > 1:
        raise SetupError(f"the model has more than one {kind.__name__}")
    return found[0] if found else None


def _require(root: nn.Module, kind: type) -> nn.Module:
    module = _unique(root, kind)
    if module is None:
        raise SetupError(f"the model has no {kind.__name__}")
    return module


def _component(loaded, name: str) -> nn.Module:
    """The root module of a declared component."""
    entry = loaded.description.entries.get(name)
    if entry is None:
        raise SetupError(f"the model declares no component {name!r}")
    root = loaded.module
    return root.get_submodule(entry.component) if entry.component else root


def _called(root: nn.Module, entry) -> list[tuple[nn.Module, str]]:
    """The modules an entry's methods are called on, with each method name.

    A method may be nested in the entry's component (``"vision.encode"``).
    """
    called = []
    for point in entry.points:
        prefix, _, method = point.method.rpartition(".")
        path = ".".join(part for part in (entry.component, prefix) if part)
        called.append((root.get_submodule(path), method))
    return called


def _conditions(values: Sequence[Sequence[int]]) -> tuple[Condition, ...]:
    result = []
    for role, frames, width, height, samples in (
        tuple(int(value) for value in item) for item in values
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


def _describe_layout(layout) -> list[int]:
    return [
        int(layout.num_frames),
        int(layout.canvas.width),
        int(layout.canvas.height),
        int(layout.num_text_tokens),
        int(layout.condition_rows),
    ]


def _layout(denoiser: VideoDenoiser, value: Sequence[int]):
    """The denoiser's layout of a layout descriptor.

    Raises:
        ValueError: The descriptor names no layout of the denoiser.
    """
    frames, width, height, text, rows = (int(item) for item in value)
    layout = denoiser.layout_size(
        denoiser.make_size(frames, text, canvas=image.Config(height, width))
    )
    if rows:
        layout = denoiser.condition_layout(layout, rows)
    if _describe_layout(layout) != [frames, width, height, text, rows]:
        raise ValueError(f"{list(value)} is not a layout of the denoiser")
    return layout


def _elements(shape: Sequence[int]) -> int:
    count = 1
    for extent in shape:
        count *= int(extent)
    return count


def _video(frames: int, width: int, height: int) -> video.Config:
    return video.Config(int(frames), image.Config(int(height), int(width)))


def _segment(config: video.Config) -> list[int]:
    return [
        int(config.num_frames),
        int(config.frame.width),
        int(config.frame.height),
    ]


def _dtype(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _output(name, layout, *, raster_axes=None) -> dict[str, Any]:
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


def model(loaded, denoiser: str) -> dict[str, Any]:
    """Return the sizing rules of a video model's components.

    ``denoiser`` names the component of the deployment's video denoiser.
    Every function reads configuration only.
    """
    root = loaded.module
    network = _component(loaded, denoiser)
    if not isinstance(network, VideoDenoiser):
        raise SetupError(f"component {denoiser!r} is not a video denoiser")
    decoder = _require(root, VideoDecoder)
    audio = _require(root, AudioDecoder)
    clock = _require(root, VideoPostprocessor)
    video_encoder = _unique(root, VideoEncoder)
    audio_encoder = _unique(root, AudioEncoder)
    vision = _unique(root, PatchEncoder)
    entries = loaded.description.entries

    def describe() -> dict[str, Any]:
        tiles = network.condition_tiles
        canvases = network.fixed_canvases
        shifts = network.schedule_shifts
        return {
            "modalities": list(network.modalities),
            "num_steps": int(network.num_steps),
            "tasks": list(network.tasks),
            "video_shift": float(shifts["video"]),
            "audio_shift": float(shifts["audio"]),
            "canvases": [
                [int(canvas.width), int(canvas.height)]
                for canvas in network.canvases
            ],
            "fixed_canvases": canvases is not None,
            "max_sequence_rows": int(network.max_sequence_rows or 0),
            "condition_tiles": []
            if tiles is None
            else [int(tiles.rows), *(int(side) for side in tiles.video)],
            "sample_dtype": _dtype(network.prediction_dtype),
            "text_condition_width": int(network.text_condition_width),
            "frame_rate": int(clock.frame_rate),
            "sample_rate": int(audio.sample_rate),
        }

    def legal_frame_count(frames: int) -> int:
        return int(network.legal_frame_count(int(frames)))

    def request_size(size, conditions, spans) -> int:
        return int(_size(network, size, conditions, spans).condition_rows)

    def layout_size(size, conditions, spans) -> list[int]:
        return _describe_layout(
            network.layout_size(_size(network, size, conditions, spans))
        )

    def holds(layout, size, conditions, spans) -> bool:
        return bool(
            network.holds(
                _layout(network, layout),
                _size(network, size, conditions, spans),
            )
        )

    def condition_layout(layout, rows: int) -> list[int]:
        return _describe_layout(
            network.condition_layout(_layout(network, layout), int(rows))
        )

    def layout_fields(layout) -> dict[str, Any]:
        """The request state of one layout on this rank.

        ``state`` lists every field of ``state_buffers`` in order (the
        sample modalities first among them flagged ``sample``); ``samples``
        maps each modality to ``[rows, width, first, stop]``: its complete
        sample's rows and width and this rank's rows; ``noise`` each
        modality's native draw shape; ``condition_noise`` the FP32 elements
        of the condition draws; ``text_condition`` the retained
        conditioning's ``[rows, width]``.
        """
        resolved = _layout(network, layout)
        outputs = network.output_layout(resolved)
        state = [
            {
                "name": name,
                "dtype": _dtype(config.dtype),
                "shape": [int(extent) for extent in config.shape],
                "sample": name in network.modalities,
            }
            for name, config in network.state_buffers(resolved).items()
        ]
        samples = {}
        for name in network.modalities:
            shape = network.latent_shape(name, resolved)
            rows = outputs[name].local_slice[0]
            samples[name] = [
                int(shape[0]),
                int(shape[1]),
                int(rows.start),
                int(rows.stop),
            ]
        return {
            "state": state,
            "samples": samples,
            "noise": {
                name: [
                    int(extent)
                    for extent in network.noise_shape(name, resolved)
                ]
                for name in network.modalities
            },
            "condition_noise": int(network.condition_noise_capacity(resolved)),
            "text_condition": [
                int(network.text_condition_rows(resolved)),
                int(network.text_condition_width),
            ],
        }

    def latent_shape(modality: str, layout) -> list[int]:
        """The complete sample shape of one modality in ``layout``."""
        shape = network.latent_shape(str(modality), _layout(network, layout))
        return [int(extent) for extent in shape]

    def unit_frames(frames: int) -> list[int]:
        return [
            window.stop - window.start
            for window in decoder.frame_slices(int(frames))
        ]

    def segment(frames: int, width: int, height: int, unit: int) -> list[int]:
        """The ``[frames, width, height]`` segment one unit decodes to."""
        size = _video(frames, width, height)
        window = decoder.frame_slices(size.num_frames)[int(unit)]
        return _segment(decoder.segment(size, window))

    def window(frames: int, width: int, height: int) -> list[Any]:
        """A segment's decoder input as ``[elements, dtype]``."""
        config = decoder.window_input(_video(frames, width, height))
        return [_elements(config.shape), _dtype(config.dtype)]

    def overlap(width: int, height: int) -> list[Any]:
        """The overlap a canvas's unit carries, as ``[elements, dtype]``."""
        size = _video(_first_frames(decoder), width, height)
        config = clock.state_buffers(size)["video_overlap"]
        return [_elements(config.shape), _dtype(config.dtype)]

    def track_samples(frames: int) -> int:
        return int(audio.track_samples(int(frames), clock.frame_rate))

    def audio_units(samples: int, units: int) -> list[int]:
        """Each audio unit's ``[start, stop)`` samples, flattened."""
        spans = audio.unit_samples(int(samples), int(units))
        return [
            int(value) for span in spans for value in (span.start, span.stop)
        ]

    def audio_latent_frames(samples: int) -> int:
        return int(audio.latent_frames(int(samples)))

    def outputs(frames: int, tokens: int, rows: int) -> dict[str, Any]:
        """Products the model's components publish, bounded for any request.

        ``frames`` is the longest admitted video, ``tokens`` the prompt
        capacity and ``rows`` the condition capacity (zero without
        conditioned tasks). Each video product bounds the video of
        ``frames`` frames at every canvas the deployment prepares.
        """
        frames = int(frames)
        sizes = [
            network.layout_size(
                network.make_size(frames, int(tokens), canvas=canvas)
            )
            for canvas in network.canvases
        ]
        result: dict[str, list[dict[str, Any]]] = {}

        def add(component, record):
            result.setdefault(component, []).append(record)

        names = {
            "video": "video_latents",
            "audio": "audio_latents",
        }
        for name in network.modalities:
            shapes = [network.latent_shape(name, size) for size in sizes]
            longest = max(shape[0] for shape in shapes)
            add(
                denoiser,
                {
                    "name": names.get(name, name),
                    "dtype": _dtype(network.prediction_dtype),
                    "shape": [int(longest), int(shapes[0][1])],
                    "variable_axes": [0],
                    "raster_axes": None,
                },
            )
        lengths = unit_frames(frames)
        decoders = [
            name
            for name, entry in entries.items()
            if any(
                method == "decode" and isinstance(module, VideoDecoder)
                for module, method in _called(root, entry)
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
        track = audio.output_layout(track_samples(frames))["audio"]
        add(_owner(loaded, AudioDecoder), _output("audio_samples", track))

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
                _output(
                    "vision_features",
                    vision.features_layout(int(tokens))["features"],
                ),
            )
        return result

    def reader_outputs(frames: int, tokens: int, rows: int) -> list[Any]:
        """The media reader's products for a deployment serving conditions.

        A condition's pixels are its rows times the pixels per row of its
        frame count, one the denoiser generates or a single frame; the bound
        is the condition rows times the largest such ratio, read from the
        video encoder's own layout (it does not depend on the raster). A
        track's stereo rows are twice its latent frames, each frame
        ``latent_rate`` samples. Patch rows bound every vision block of the
        presentation.
        """
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
            _output("vision_pixels", vision.pixels_layout(int(tokens))),
        ]

    return _translated(
        {
            "describe": describe,
            "legal_frame_count": legal_frame_count,
            "request_size": request_size,
            "layout_size": layout_size,
            "holds": holds,
            "condition_layout": condition_layout,
            "layout_fields": layout_fields,
            "unit_frames": unit_frames,
            "track_samples": track_samples,
            "audio_units": audio_units,
            "audio_latent_frames": audio_latent_frames,
            "latent_shape": latent_shape,
            "segment": segment,
            "window": window,
            "overlap": overlap,
            "outputs": outputs,
            "reader_outputs": reader_outputs,
        }
    )


def _translated(functions: dict[str, Any]) -> dict[str, Any]:
    """Give a binding's functions the interface's error kinds."""
    return {
        name: translate(value) if callable(value) else value
        for name, value in functions.items()
    }


def _owner(loaded, kind: type) -> str:
    """The component whose entry calls a module of ``kind``."""
    root = loaded.module
    for name, entry in loaded.description.entries.items():
        for module, _ in _called(root, entry):
            if isinstance(module, kind):
                return name
    raise SetupError(f"no component calls the model's {kind.__name__}")


def _context(module, request, name: str) -> ExecutionContext:
    """An execution context of ``module`` on the binding's stream."""
    backends = dict(request.get("backends") or {})
    stream = int(request["stream"])
    return ExecutionContext(
        module,
        stream=stream if stream else None,
        attention=request["attention"],
        vsa=str(backends.get("vsa", "auto")),
        matmul=str(backends.get("matmul", "auto")),
        moe=str(backends.get("moe", "auto")),
        derive_host_lengths=False,
        workspace=RuntimeWorkspace(request["allocate"], name),
    )


def _tensors(values) -> dict[str, torch.Tensor]:
    return {
        str(name): torch.from_dlpack(tensor)
        for name, tensor in dict(values or {}).items()
    }


def _view(tensor: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """The leading elements of a flat backing as ``shape``."""
    return tensor.reshape(-1)[: _elements(shape)].view(tuple(shape))


@dataclass
class _Plan:
    """One prepared layout's numerical views."""

    layout: Any
    constants: Mapping[str, torch.Tensor]
    workspace: Mapping[str, torch.Tensor]
    samples: dict[str, torch.Tensor]
    state: dict[str, torch.Tensor]
    timesteps: dict[str, torch.Tensor]
    inputs: Any
    # Constants allocated for this layout alone (other than the maximum's).
    owned: TensorBuffers | None = None
    graphs: list[graphs.Graph] = field(default_factory=list)


def bind_denoiser(loaded, request: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a video denoiser's serving calls over worker-owned buffers.

    ``request`` holds:

    - ``component``, ``stream``, ``attention``, ``backends`` and
      ``allocate`` (the worker's workspace allocation callback);
    - ``maximum``: the largest layout, whose capacity the denoising context
      is prepared for;
    - ``samples``: the flat FP32 scratch one request's samples occupy while
      a step runs, each modality at the element offset the worker assigns;
    - ``step``: the ``[1]`` int64 evaluation index the worker writes before
      every step;
    - ``state``: flat backings of every state field other than the samples,
      and of ``text_condition``: the fixed inputs a step reads, which the
      worker fills from the request's slot before each step;
    - ``banks``: per-slot device rows ``[slots + 1, elements]`` of the same
      fields (row ``slot`` belongs to request slot ``slot``);
    - ``host``: per-slot pinned host rows of ``{field}_source``,
      ``{modality}_noise`` and ``condition_noise``;
    - ``conditioner``: ``{"input", "lengths", "output"}`` of the text
      conditioner, absent on a pipeline stage without it.

    Returns the binding's functions (see the inner definitions).
    """
    network = _component(loaded, str(request["component"]))
    if not isinstance(network, VideoDenoiser):
        raise SetupError("the denoising component is not a video denoiser")
    device = next(network.parameters()).device
    context = _context(network, request, "denoiser")
    maximum = _layout(network, request["maximum"])
    context.prepare(maximum)

    samples = torch.from_dlpack(request["samples"])
    step = torch.from_dlpack(request["step"])
    state_backing = _tensors(request["state"])
    banks = _tensors(request["banks"])
    host = _tensors(request["host"])
    schedules = network.make_schedules(
        network.num_steps, shift=None, device=device
    )
    modalities = tuple(network.modalities)
    plans: dict[int, _Plan] = {}

    def prepare_layout(identifier: int, layout, offsets) -> None:
        """Prepare one layout's constants and views under ``identifier``.

        ``offsets`` are the element offsets of each modality's local sample
        in the sample scratch.
        """
        resolved = _layout(network, layout)
        owned = None
        if resolved == maximum:
            constants, workspace = context.constants, context.workspace_views
        else:
            requirements = network.constant_buffers(resolved)
            owned = TensorBuffers.allocate(requirements, device=device)
            constants = owned.view(requirements)
            with context.activate():
                network.prepare_constants(resolved, out=constants)
            workspace = TensorBuffers.from_tensors(
                dict(context.workspace_views)
            ).view(network.workspace_buffers(resolved))
        fields = network.state_buffers(resolved)
        views = {}
        for name, offset in zip(modalities, offsets, strict=True):
            shape = fields[name].shape
            start = int(offset)
            if start + _elements(shape) > samples.numel():
                raise ValueError(
                    f"the {name} sample {tuple(shape)} at element {start} "
                    f"exceeds the {samples.numel()}-element sample scratch"
                )
            views[name] = samples[start : start + _elements(shape)].view(shape)
        state = {
            name: _view(state_backing[name], config.shape)
            for name, config in fields.items()
            if name not in modalities
        }
        rows = network.text_condition_rows(resolved)
        state["text_condition"] = _view(
            state_backing["text_condition"],
            (rows, network.text_condition_width),
        )
        timesteps = {
            name: torch.empty((1,), dtype=torch.float32, device=device)
            for name in modalities
        }
        inputs = network.bind_inputs(
            latents={
                name: (LatentInput(views[name], timesteps[name][0]),)
                for name in modalities
            },
            sizes=(resolved,),
            step=step,
            text_features=(state["text_condition"],),
        )
        plans[int(identifier)] = _Plan(
            resolved,
            constants,
            workspace,
            views,
            state,
            timesteps,
            inputs,
            owned,
        )

    def release_layout(identifier: int) -> None:
        """Drop a layout the worker no longer evaluates.

        The worker has waited for the layout's last step.
        """
        plan = plans.pop(int(identifier))
        if plan.owned is not None:
            plan.owned.close()

    def _compute(plan: _Plan) -> None:
        # The step's endpoints are gathered on the device from the bound
        # schedule, so one captured step serves every evaluation.
        for name in modalities:
            torch.index_select(
                schedules[name].timesteps, 0, step, out=plan.timesteps[name]
            )
        DenoisingStep(
            network,
            plan.inputs,
            schedules,
            plan.state,
            plan.constants,
            plan.workspace,
        )()

    def run(identifier: int) -> None:
        """Evaluate one solver step of the layout over the bound buffers."""
        plan = plans[int(identifier)]
        with torch.inference_mode(), context.activate():
            _compute(plan)

    def capture(identifier: int, pool) -> graphs.Graph:
        """Capture the layout's step into the worker's graph pool."""
        plan = plans[int(identifier)]

        def body() -> None:
            with context.activate():
                _compute(plan)

        return graphs.capture(body, stream=context.stream, pool=pool)

    def _host_views(slot: int, plan: _Plan) -> dict[str, torch.Tensor]:
        resolved = plan.layout
        views = {}
        for name, config in network.state_buffers(resolved).items():
            views[f"{name}_source"] = _view(
                host[f"{name}_source"][slot], config.shape
            )
        for name in modalities:
            views[f"{name}_noise"] = _view(
                host[f"{name}_noise"][slot],
                network.noise_shape(name, resolved),
            )
        return views

    def _condition_noise(slot: int, size) -> tuple[torch.Tensor, ...]:
        # The condition draws occupy the leading elements of the slot's
        # condition noise one after another, each leading with its batch
        # of one.
        flat, views, offset = host["condition_noise"][slot], [], 0
        for shape in network.condition_noise_shapes(size):
            count = _elements(shape)
            views.append(flat[offset : offset + count].view(tuple(shape)))
            offset += count
        return tuple(views)

    def prepare_request(
        slot: int, identifier: int, size, conditions, spans, seed: int
    ) -> None:
        """Draw a request's noise and fill its host state for its layout.

        Runs on a host thread without device work: the seed's stream draws
        the conditions' noise, then each modality's; the request's tables
        and samples are prepared into the slot's host sources.
        """
        slot = int(slot)
        plan = plans[int(identifier)]
        resolved = _size(network, size, conditions, spans)
        views = _host_views(slot, plan)
        # The worker passes the unsigned 64-bit seed's bits as a signed
        # integer.
        seed = int(seed) & 0xFFFF_FFFF_FFFF_FFFF
        with torch.inference_mode():
            normal_noise(
                (seed,),
                out=(
                    *_condition_noise(slot, resolved),
                    *(
                        views[f"{name}_noise"].unsqueeze(0)
                        for name in modalities
                    ),
                ),
            )
            tables = [
                name
                for name in network.state_buffers(plan.layout)
                if name not in modalities
            ]
            network.prepare_state(
                (resolved,),
                layouts=(plan.layout,),
                out={name: views[f"{name}_source"] for name in tables},
            )
            network.prepare_latents(
                (plan.layout,),
                noise={
                    name: views[f"{name}_noise"].unsqueeze(0)
                    for name in modalities
                },
                state={
                    name: views[f"{name}_source"].unsqueeze(0)
                    for name in modalities
                },
                constants=plan.constants,
                workspace=plan.workspace,
            )

    def encode_conditions(
        slot: int, identifier: int, size, conditions, spans, latents
    ) -> None:
        """Write a request's encoded conditions into its retained rows.

        ``latents`` are the condition latents in request order, a
        condition's visual rows before its audio rows. The rows follow the
        prompt's in the slot's ``text_condition``.
        """
        slot = int(slot)
        plan = plans[int(identifier)]
        resolved = _size(network, size, conditions, spans)
        rows = network.text_condition_rows(plan.layout)
        out = _view(
            banks["text_condition"][slot],
            (rows, network.text_condition_width),
        )
        with torch.inference_mode(), context.activate():
            network.encode_conditions(
                resolved,
                plan.layout,
                latents=tuple(torch.from_dlpack(value) for value in latents),
                noise=_condition_noise(slot, resolved),
                out=out,
            )

    result: dict[str, Any] = {
        "prepare_layout": prepare_layout,
        "release_layout": release_layout,
        "step": run,
        "capture": capture,
        "replay": lambda graph: graph.replay(),
        "release": lambda graph: graph.release(),
        "prepare_request": prepare_request,
        "encode_conditions": encode_conditions,
    }

    conditioner_buffers = request.get("conditioner")
    contexts: dict[int, ExecutionContext] = {}
    if conditioner_buffers is not None:
        conditioner = network.conditioner
        buffers = _tensors(conditioner_buffers)
        source, lengths, target = (
            buffers["input"],
            buffers["lengths"],
            buffers["output"],
        )

        def prepare_conditioner(capacity: int) -> None:
            """Prepare the conditioner for prompts padded to ``capacity``."""
            capacity = int(capacity)
            prepared = _context(conditioner, request, f"conditioner.{capacity}")
            # The context is keyed by its input's shape, as one numerical
            # signature of the conditioner.
            prepared.prepare((tuple(source[:capacity].shape),))
            contexts[capacity] = prepared

        def _condition(capacity: int) -> None:
            out = conditioner.encode((source[:capacity],), lengths=lengths)[0]
            target[:capacity].copy_(out)

        def condition(capacity: int) -> None:
            """Refine the bound prompt features padded to ``capacity``.

            The worker wrote the prompt's features into the input's leading
            rows, zero padding up to the capacity, and its length into
            ``lengths``; the refined rows land in the output's leading
            ``capacity`` rows.
            """
            capacity = int(capacity)
            with torch.inference_mode(), contexts[capacity].activate():
                _condition(capacity)

        def capture_conditioner(capacity: int, pool) -> graphs.Graph:
            capacity = int(capacity)
            prepared = contexts[capacity]

            def body() -> None:
                with prepared.activate():
                    _condition(capacity)

            return graphs.capture(body, stream=prepared.stream, pool=pool)

        result.update(
            prepare_conditioner=prepare_conditioner,
            condition=condition,
            capture_conditioner=capture_conditioner,
        )

    def kernels() -> list[dict]:
        records = list(context.kernels())
        for prepared in contexts.values():
            records.extend(prepared.kernels())
        return records

    def close(aborted: bool) -> None:
        for prepared in contexts.values():
            prepared.close(aborted=bool(aborted))
        context.close(aborted=bool(aborted))
        if not aborted:
            for plan in plans.values():
                if plan.owned is not None:
                    plan.owned.close()
            plans.clear()
            contexts.clear()

    result.update(kernels=kernels, close=close)
    return _translated(result)


def bind_decoder(loaded, request: Mapping[str, Any]) -> dict[str, Any]:
    """Bind one media unit's decode and RGB reconstruction.

    ``request`` holds ``component``, ``stream``, ``attention``, ``backends``
    and ``allocate``, and the worker's buffers: ``window`` (the flat FP32
    decoder input of the largest segment), ``overlap`` (per-slot rows
    ``[slots + 1, elements]`` of the post-processor's retained overlap) and
    ``exchange`` (the flat overlap the unit ring delivers).
    """
    root = loaded.module
    decoder = _require(root, VideoDecoder)
    postprocessor = _require(root, VideoPostprocessor)
    device = next(decoder.parameters()).device
    window = torch.from_dlpack(request["window"])
    overlap = torch.from_dlpack(request["overlap"])
    exchange = torch.from_dlpack(request["exchange"])
    segments: dict[tuple[int, int, int], ExecutionContext] = {}
    canvases: dict[tuple[int, int], Mapping[str, torch.Tensor]] = {}
    graph_outputs: dict[tuple[int, int, int], graphs.Graph] = {}

    def _segment_key(segment) -> tuple[int, int, int]:
        return tuple(_segment(segment))

    def prepare_segment(frames: int, width: int, height: int) -> None:
        """Prepare decoding of one segment, before any capture."""
        config = _video(frames, width, height)
        key = _segment_key(config)
        if key in segments:
            return
        prepared = _context(decoder, request, f"decoder.{key}")
        prepared.prepare(config)
        segments[key] = prepared

    def prepare_canvas(width: int, height: int) -> None:
        """Prepare the RGB reconstruction constants of one canvas."""
        key = (int(width), int(height))
        if key in canvases:
            return
        size = _video(_first_frames(decoder), width, height)
        backing = TensorBuffers.allocate(
            postprocessor.constant_buffers(size), device=device
        )
        constants = backing.view(postprocessor.constant_buffers(size))
        with torch.inference_mode():
            postprocessor.prepare_constants(size, out=constants)
        canvases[key] = constants

    def _window(config) -> torch.Tensor:
        layout = decoder.window_input(config)
        return _view(window, layout.shape)

    def capture_segment(frames: int, width: int, height: int, pool):
        """Capture one segment's decode of the bound window."""
        config = _video(frames, width, height)
        key = _segment_key(config)
        prepared = segments[key]
        target = _window(config)

        def body():
            with prepared.activate():
                return decoder.decode((target,), segments=(config,))[0]

        graph = graphs.capture(body, stream=prepared.stream, pool=pool)
        graph_outputs[key] = graph
        return graph

    def warm_segment(frames: int, width: int, height: int) -> None:
        """Decode a zero window of one segment eagerly, before capture."""
        config = _video(frames, width, height)
        prepared = segments[_segment_key(config)]
        target = _window(config)
        with torch.inference_mode(), prepared.activate():
            target.zero_()
            decoder.decode((target,), segments=(config,))

    def warm_unit(
        frames: int, width: int, height: int, unit: int, unit_count: int
    ) -> None:
        """Reconstruct a zero segment as one unit of a round.

        Every ring member holding a unit of the round calls it, so the
        ring's exchange of every round shape is prepared before serving.
        The padding slot's overlap row serves as the retained overlap.
        """
        size = _video(frames, width, height)
        frames_slice = decoder.frame_slices(size.num_frames)[int(unit)]
        config = decoder.segment(size, frames_slice)
        layout = decoder.output_layout(size)["video"]
        state_shape = postprocessor.state_buffers(size)["video_overlap"].shape
        longest = max(
            window.stop - window.start
            for window in decoder.frame_slices(size.num_frames)
        )
        prepared = segments[_segment_key(config)]
        with torch.inference_mode(), prepared.activate():
            decoded = torch.zeros(
                (1, *layout.shape[1:]), dtype=layout.dtype, device=device
            )
            rgb = torch.empty(
                (longest, int(height), int(width), 3),
                dtype=torch.uint8,
                device=device,
            )
            postprocessor(
                (decoder.place(decoded, frames_slice, size),),
                frames=(frames_slice,),
                sizes=(size,),
                state={"video_overlap": _view(overlap[0], state_shape)},
                constants=canvases[(int(width), int(height))],
                workspace={
                    "rgb_frames": rgb,
                    "overlap_exchange": _view(exchange, state_shape),
                },
                unit_count=int(unit_count),
            )

    def decode_unit(
        slot: int,
        latent,
        frames: int,
        width: int,
        height: int,
        unit: int,
        unit_count: int,
        out,
        graph,
    ) -> int:
        """Decode and reconstruct one media unit into ``out``.

        ``latent`` is the request's complete packed video latent; ``out``
        the unit's ``[longest, height, width, 3]`` uint8 product row, whose
        leading frames receive the unit's RGB. ``graph`` replays the unit's
        captured segment decode, or is None for an eager decode. The
        post-processor exchanges overlaps with the ``unit_count`` ring
        members holding a unit of this round. Returns the unit's frames.
        """
        size = _video(frames, width, height)
        frames_slice = decoder.frame_slices(size.num_frames)[int(unit)]
        config = decoder.segment(size, frames_slice)
        prepared = segments[_segment_key(config)]
        target = _window(config)
        state_shape = postprocessor.state_buffers(size)["video_overlap"].shape
        with torch.inference_mode(), prepared.activate():
            decoder.unpack_latents(
                torch.from_dlpack(latent), frames_slice, size, out=target
            )
            decoded = (
                graph.replay()
                if graph is not None
                else decoder.decode((target,), segments=(config,))[0]
            )
            postprocessor(
                (decoder.place(decoded, frames_slice, size),),
                frames=(frames_slice,),
                sizes=(size,),
                state={"video_overlap": _view(overlap[int(slot)], state_shape)},
                constants=canvases[(int(width), int(height))],
                workspace={
                    "rgb_frames": torch.from_dlpack(out),
                    "overlap_exchange": _view(exchange, state_shape),
                },
                unit_count=int(unit_count),
            )
        return frames_slice.stop - frames_slice.start

    def kernels() -> list[dict]:
        return [
            record
            for prepared in segments.values()
            for record in prepared.kernels()
        ]

    def close(aborted: bool) -> None:
        for prepared in segments.values():
            prepared.close(aborted=bool(aborted))
        if not aborted:
            segments.clear()
            canvases.clear()
            graph_outputs.clear()

    return _translated(
        {
            "prepare_segment": prepare_segment,
            "prepare_canvas": prepare_canvas,
            "warm_segment": warm_segment,
            "warm_unit": warm_unit,
            "capture_segment": capture_segment,
            "replay": lambda graph: graph.replay(),
            "release": lambda graph: graph.release(),
            "decode_unit": decode_unit,
            "kernels": kernels,
            "close": close,
        }
    )


def _first_frames(decoder: VideoDecoder) -> int:
    """The shortest video the decoder reconstructs.

    Its overlap state and RGB constants are every video's of its canvas.
    """
    frames = 1
    while True:
        try:
            decoder.frame_slices(frames)
        except ValueError:
            frames += 1
            continue
        return frames


def bind_audio_decoder(loaded, request: Mapping[str, Any]) -> dict[str, Any]:
    """Bind audio unit decoding.

    ``request`` holds ``component``, ``stream``, ``attention``, ``backends``
    and ``allocate``.
    """
    audio = _require(loaded.module, AudioDecoder)
    device = next(audio.parameters()).device
    contexts: dict[int, ExecutionContext] = {}

    def prepare(latent_frames: int) -> None:
        """Prepare the decode of tracks of ``latent_frames`` latent frames."""
        frames = int(latent_frames)
        if frames in contexts:
            return
        prepared = _context(audio, request, f"audio.{frames}")
        prepared.prepare(frames)
        contexts[frames] = prepared

    def warm(samples: int, first: int, count: int, total: int, shape) -> None:
        """Decode audio units of a zero track of ``samples`` samples.

        ``shape`` is the track's complete latent shape.
        """
        samples = int(samples)
        frames = audio.latent_frames(samples)
        prepared = contexts[frames]
        windows = audio.unit_frames(samples, int(total))[
            int(first) : int(first) + int(count)
        ]
        with torch.inference_mode(), prepared.activate():
            latent = torch.zeros(
                tuple(int(extent) for extent in shape),
                dtype=torch.float32,
                device=device,
            )
            audio.decode(
                (latent,) * len(windows),
                frames=tuple(windows),
                num_samples=(samples,) * len(windows),
                workspace=prepared.workspace_views,
            )

    def decode(
        latent, samples: int, first: int, count: int, total: int, out
    ) -> int:
        """Decode audio units ``[first, first + count)`` of ``total``.

        ``latent`` is the request's complete audio latent timeline and
        ``samples`` its exact duration; the units' samples, which abut,
        land in ``out`` (``[samples, 2]`` int16) in order. Returns the
        samples written.
        """
        samples = int(samples)
        windows = audio.unit_frames(samples, int(total))[
            int(first) : int(first) + int(count)
        ]
        prepared = contexts[audio.latent_frames(samples)]
        value = torch.from_dlpack(latent)
        target = torch.from_dlpack(out)
        with torch.inference_mode(), prepared.activate():
            outputs = audio.decode(
                (value,) * len(windows),
                frames=tuple(windows),
                num_samples=(samples,) * len(windows),
                workspace=prepared.workspace_views,
            )
            written = sum(int(output.shape[0]) for output in outputs)
            if tuple(target.shape) != (written, 2):
                raise ValueError("audio units disagree with their product rows")
            torch.cat(outputs, out=target)
        return written

    def kernels() -> list[dict]:
        return [
            record
            for prepared in contexts.values()
            for record in prepared.kernels()
        ]

    def close(aborted: bool) -> None:
        for prepared in contexts.values():
            prepared.close(aborted=bool(aborted))
        if not aborted:
            contexts.clear()

    return _translated(
        {
            "prepare": prepare,
            "warm": warm,
            "decode": decode,
            "kernels": kernels,
            "close": close,
        }
    )


def bind_latent_encoder(loaded, request: Mapping[str, Any]) -> dict[str, Any]:
    """Bind condition encoding into denoiser rows.

    ``request`` holds ``component``, ``stream``, ``attention``, ``backends``
    and ``allocate``.
    """
    root = loaded.module
    video_encoder = _require(root, VideoEncoder)
    audio_encoder = _require(root, AudioEncoder)
    contexts: dict[tuple, ExecutionContext] = {}

    def _prepared(module, signature: tuple) -> ExecutionContext:
        key = (id(module), signature)
        prepared = contexts.get(key)
        if prepared is None:
            prepared = _context(module, request, f"encoder.{len(contexts)}")
            prepared.prepare(signature)
            contexts[key] = prepared
        return prepared

    def encode_video(pixels, units, out) -> list[int]:
        """Encode visual units into consecutive rows of ``out``.

        ``pixels`` is the request's ``[pixels, 3]`` uint8 condition pixels;
        each unit is ``[pixel_offset, frames, width, height, unit]``: its
        condition's first pixel row, frame count and raster, and the unit's
        index among the condition's units. Returns each unit's rows.
        """
        source = torch.from_dlpack(pixels)
        target = torch.from_dlpack(out)
        values, windows, counts = [], [], []
        for offset, frames, width, height, unit in (
            tuple(int(value) for value in item) for item in units
        ):
            window = video_encoder.frame_slices(frames)[unit]
            area = width * height
            values.append(
                source[
                    offset + window.start * area : offset + window.stop * area
                ].view(window.stop - window.start, height, width, 3)
            )
            windows.append(window)
            counts.append(frames)
        signature = tuple(tuple(value.shape) for value in values)
        prepared = _prepared(video_encoder, signature)
        with torch.inference_mode(), prepared.activate():
            encoded = video_encoder.encode(
                tuple(values), frames=tuple(windows), num_frames=tuple(counts)
            )
            rows = [int(output.tensor.shape[0]) for output in encoded]
            if sum(rows) != target.shape[0]:
                raise ValueError(
                    "encoded condition units disagree with their rows"
                )
            torch.cat([output.tensor for output in encoded], out=target)
        return rows

    def encode_audio(samples, lengths, out) -> list[int]:
        """Encode every audio track into consecutive rows of ``out``.

        ``samples`` is the request's ``[samples, 2]`` FP32 PCM, the tracks'
        sample counts ``lengths``. Returns each track's rows.
        """
        source = torch.from_dlpack(samples)
        target = torch.from_dlpack(out)
        tracks = source.split([int(length) for length in lengths])
        signature = tuple(tuple(track.shape) for track in tracks)
        prepared = _prepared(audio_encoder, signature)
        with torch.inference_mode(), prepared.activate():
            encoded = audio_encoder.encode(tracks)
            rows = [int(value.shape[0]) for value in encoded]
            if sum(rows) != target.shape[0]:
                raise ValueError(
                    "encoded audio tracks disagree with their rows"
                )
            torch.cat(encoded, out=target)
        return rows

    def kernels() -> list[dict]:
        return [
            record
            for prepared in contexts.values()
            for record in prepared.kernels()
        ]

    def close(aborted: bool) -> None:
        for prepared in contexts.values():
            prepared.close(aborted=bool(aborted))
        if not aborted:
            contexts.clear()

    return _translated(
        {
            "encode_video": encode_video,
            "encode_audio": encode_audio,
            "kernels": kernels,
            "close": close,
        }
    )
