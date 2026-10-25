"""Generate one H3 video in eager library calls.

``generate`` runs a text-conditioned request through the runners serving
uses (``uniserve.runner``): the text encoder's and the denoiser's
conditioner's ``EncoderRunner``, the denoiser's ``DenoiserRunner`` for every
solver step, and the video and audio decoders' ``DecoderRunner``. It holds
the request state the executor holds, sized by the model's facts
(``uniserve.executor.video.plan.VideoModel``), and calls the runners in the
executor's order and with its arguments, eagerly, on one stream of its own.
Every rank that holds the components calls it with the same arguments
(SPMD); a denoiser sharded by sequence
parallelism gathers its final samples before decoding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from uniserve.bindings.workspace import Scratch
from uniserve.executor.video.decoder import clip, overlap
from uniserve.executor.video.plan import Canvas, Layout, VideoModel, VideoSize
from uniserve.media import image
from uniserve.runner.decoder import DecoderRunner
from uniserve.runner.denoiser import DenoiserRunner
from uniserve.runner.encoder import EncoderRunner

if TYPE_CHECKING:
    from uniserve.loading.rank import Loaded

# The module paths of the components ``generate`` runs besides the denoiser
# (``model.entry_points``); a denoiser's conditioner is its ``conditioner``.
_TEXT_ENCODER = "text_encoder"
_VIDEO_DECODER = "video_decoder"
_AUDIO_DECODER = "audio_decoder"


@dataclass(frozen=True, slots=True)
class Generation:
    """One generated video with its stereo track.

    Attributes:
        video_latent: [rows, 96] FP32 final video latent, packed order.
        audio_latent: [2 * frames, 32] FP32 final audio latent,
            channel-major.
        frames: [num_frames, height, width, 3] uint8 RGB frames.
        audio: [samples, 2] int16 PCM at the model's sample rate.
    """

    video_latent: torch.Tensor
    audio_latent: torch.Tensor
    frames: torch.Tensor
    audio: torch.Tensor


@dataclass
class _Runners:
    """The runners of one generation, constructed as a worker does.

    Every runner enqueues on ``stream`` (its handle) and borrows the
    stream's ``scratch``; ``close`` releases them, then the scratch.
    """

    loaded: Any
    model: VideoModel
    stream: torch.cuda.Stream
    scratch: Scratch = field(default_factory=Scratch)
    opened: list[Any] = field(default_factory=list)

    def open(self, runner: type, path: str, size: Sequence[int]) -> Any:
        """Construct a runner of the module at ``path`` for ``size``."""
        opened = runner(
            self.loaded,
            path,
            self.stream.cuda_stream,
            self.loaded.attention_backend,
            self.scratch,
            [int(extent) for extent in size],
        )
        self.opened.append(opened)
        return opened

    def close(self) -> None:
        """Release every runner and the scratch once the stream drained."""
        self.stream.synchronize()
        for runner in reversed(self.opened):
            runner.close(False)
        self.scratch.close()


def _gather(sample: torch.Tensor, rows: int, group) -> torch.Tensor:
    """Concatenate every sequence rank's canonical rows in rank order.

    Each rank holds one contiguous range of the ``rows``-row canonical
    sample and the ranges follow the ranks, so gathering shards padded to the
    whole sample and dropping each rank's padding restores the sample.
    """
    if group.size == 1:
        return sample
    width = rows
    padded = sample.new_zeros((width, sample.shape[1]))
    padded[: sample.shape[0]].copy_(sample)
    counts = torch.tensor([sample.shape[0]], device=sample.device)
    every = group.all_gather(counts)
    shards = group.all_gather(padded).view(group.size, width, -1)
    return torch.cat(
        [shards[rank, : int(every[rank])] for rank in range(group.size)]
    )


def _condition(
    runners: _Runners,
    component: str,
    layout: Sequence[int],
    prompt_token_ids: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Encode and refine the prompt into the request's retained conditioning.

    Returns the flat BF16 conditioning of ``layout``: the refined prompt in
    its leading rows and zeros past them. The text encoder reads the prompt
    padded with token 0 to the layout's text capacity (its causal attention
    keeps the padding out of the prompt's rows); the conditioner refines the
    prompt's features at the same capacity with the padding rows zero, as
    the worker's ``DenoiserRunner::condition`` feeds it.

    Raises:
        ValueError: This rank is not the text encoder's final pipeline
            stage, which alone holds features to condition on.
    """
    loaded = runners.loaded
    tokens = len(prompt_token_ids)
    capacity = int(layout[3])
    width = EncoderRunner.output_width(loaded, _TEXT_ENCODER)
    text = runners.open(EncoderRunner, _TEXT_ENCODER, [capacity, 1])
    padded = torch.tensor(
        (*prompt_token_ids, *(0,) * (capacity - tokens)),
        dtype=torch.long,
        device=device,
    )
    features = torch.empty((capacity, width), dtype=dtype, device=device)
    text.encode(padded, features)
    if not text.final_stage():
        raise ValueError(
            "eager H3 generation needs the text encoder's final pipeline stage"
        )

    rows, columns = runners.model.fields(Layout(*layout)).text_condition
    conditioning = torch.zeros(
        rows * columns, dtype=torch.bfloat16, device=device
    )
    # The conditioner runs on the denoiser's first pipeline stage, whose
    # forward writes the refined prompt into the packed rows; a later stage
    # holds no conditioner and reads no conditioning.
    path = f"{component}.conditioner"
    if EncoderRunner.holds(loaded, path):
        conditioner = runners.open(EncoderRunner, path, [capacity, width])
        features[tokens:].zero_()
        lengths = torch.full((1,), tokens, dtype=torch.int32, device=device)
        refined = torch.empty(
            (capacity, columns), dtype=torch.bfloat16, device=device
        )
        conditioner.encode(features, lengths, refined)
        conditioning.view(rows, columns)[:tokens].copy_(refined[:tokens])
    return conditioning


def _denoise(
    runners: _Runners,
    component: str,
    layout: Sequence[int],
    size: Sequence[int],
    seed: int,
    conditioning: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Run a request's trajectory and return each modality's final sample.

    The request's state is what the worker keeps for its slot: one flat
    tensor per state field of ``layout`` (``VideoModel.fields``),
    prepared on the host from the seed's draws and copied to the device,
    which every solver step advances in place. A sequence-parallel rank
    holds its own rows of each sample; they are gathered over the
    denoiser's sequence group into the complete ``[rows, width]`` sample.
    """
    loaded = runners.loaded
    model = runners.model
    denoiser = runners.open(DenoiserRunner, component, layout)
    prepared = denoiser.prepare_layout(layout)

    fields = model.fields(Layout(*layout))
    names = [field.name for field in fields.state]
    noise = [
        torch.empty(count, dtype=torch.float32) for _, count in fields.noise
    ]
    condition_noise = torch.empty(fields.condition_noise, dtype=torch.float32)
    # Page-locked host sources copy to the device without blocking.
    sources = [
        torch.empty(field.elements, dtype=field.dtype, pin_memory=True)
        for field in fields.state
    ]
    # A text-conditioned request has no conditions and no vision spans.
    denoiser.draw_noise(seed, prepared, size, [], [], condition_noise, *noise)
    denoiser.prepare_latents(prepared, size, [], [], *noise, *sources)
    state = [source.to(device, non_blocking=True) for source in sources]

    step = torch.zeros(1, dtype=torch.int64, device=device)
    for index in range(model.num_steps):
        step.fill_(index)
        denoiser.denoise(prepared, step, conditioning, *state)

    # Each modality's complete sample and this rank's rows of it, which the
    # state field holds.
    group = loaded.module.get_submodule(component).sequence_group()
    latents = {}
    for name, (rows, width, first, stop) in fields.samples:
        local = state[names.index(name)][: (stop - first) * width]
        latents[name] = _gather(local.view(stop - first, width), rows, group)
    return latents


def _decode_video(
    runners: _Runners,
    latent: torch.Tensor,
    num_frames: int,
    canvas: image.Config,
    device: torch.device,
) -> torch.Tensor:
    """Decode and reconstruct a video's media units in order.

    Each unit decodes one clip at its clip output, whose runner the worker
    constructs once, largest first. A unit reads its clip's rows of the
    packed latent and the overlap the previous unit retained; this rank
    reconstructs every unit alone, so each round holds one unit.
    """
    model = runners.model
    width, height = canvas.width, canvas.height
    raster = Canvas(width, height)
    units = model.unit_frames(num_frames)
    clips = [
        clip(model, num_frames, raster, unit) for unit in range(len(units))
    ]
    outputs = [
        (frames, output.width, output.height) for (frames, output), _ in clips
    ]
    decoders = {
        output: runners.open(DecoderRunner, _VIDEO_DECODER, output)
        for output in sorted(
            set(outputs),
            key=lambda output: output[0] * output[1] * output[2],
            reverse=True,
        )
    }
    elements, overlap_dtype = overlap(model, raster)
    retained = torch.zeros(elements, dtype=overlap_dtype, device=device)
    rgb = torch.empty(
        (max(units), height, width, 3), dtype=torch.uint8, device=device
    )
    frames = []
    for unit, (count, output) in enumerate(zip(units, outputs, strict=True)):
        rows = torch.tensor(clips[unit][1], dtype=torch.long, device=device)
        decoders[output].decode(
            latent, rows, retained, rgb, unit == 0, unit + 1 == len(units), 1
        )
        frames.append(rgb[:count].clone())
    return torch.cat(frames)


def _decode_audio(
    runners: _Runners,
    latent: torch.Tensor,
    num_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """Decode a video's stereo track as one audio unit."""
    model = runners.model
    samples = model.track_samples(num_frames)
    frames = int(model.audio.latent_frames(samples))
    decoder = runners.open(DecoderRunner, _AUDIO_DECODER, [frames])
    audio = torch.empty((samples, 2), dtype=torch.int16, device=device)
    decoder.decode(latent, audio, samples, 0, 1, 1)
    return audio


@torch.inference_mode()
def generate(
    loaded: Loaded,
    *,
    prompt_token_ids: Sequence[int],
    num_frames: int,
    canvas: image.Config,
    seed: int,
    component: str = "transformer",
    layout: Sequence[int] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Generation:
    """Generate a text-conditioned video and audio track eagerly.

    Args:
        loaded: An H3 model loaded on this rank
            (``uniserve.loading.rank.load_model``) holding the text encoder,
            ``component``'s denoiser, both decoders and the post-processor,
            on one CUDA device.
        prompt_token_ids: The presented prompt's token ids.
        num_frames: Output frames at 24 fps, of the form ``17 * n + 5``.
        canvas: The output raster, one the denoiser generates.
        seed: The request seed of the native CPU noise draw.
        component: The denoising component to run.
        layout: The capacity layout the request evaluates in, as the
            runners describe it (``[frames, width, height, text_rows,
            condition_rows]``), one that holds the request; by default the
            smallest that does. The prompt is encoded and refined at the
            layout's text capacity, as a worker evaluates it in its own
            capacity layout.
        dtype: The dtype the model computes in (the ``model_dtype`` it was
            loaded with), which the text encoder's features carry.

    Raises:
        ValueError: The denoiser rejects the size (a canvas it does not
            generate, a frame count it does not produce), ``layout`` does
            not hold it, or this rank is not the text encoder's final
            pipeline stage.
    """
    # The denoiser's placement fixes the device the request computes on.
    device = next(loaded.module.get_submodule(component).parameters()).device
    size = [int(num_frames), canvas.width, canvas.height, len(prompt_token_ids)]
    model = VideoModel(loaded, component)
    request = VideoSize(
        int(num_frames),
        Canvas(canvas.width, canvas.height),
        len(prompt_token_ids),
    )
    if layout is None:
        layout = model.layout_size(request)
    elif not model.denoiser.holds(
        model.resolve(Layout(*(int(extent) for extent in layout))),
        model.size(request),
    ):
        raise ValueError("the capacity layout does not hold the request")
    layout = [int(extent) for extent in layout]

    # Every call enqueues on the generation's own stream, which first waits
    # for the caller's work; the runners are released once it drained.
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    runners = _Runners(loaded, model, stream)
    try:
        with torch.cuda.stream(stream):
            conditioning = _condition(
                runners, component, layout, prompt_token_ids, dtype, device
            )
            latents = _denoise(
                runners, component, layout, size, seed, conditioning, device
            )
            frames = _decode_video(
                runners, latents["video"], int(num_frames), canvas, device
            )
            audio = _decode_audio(
                runners, latents["audio"], int(num_frames), device
            )
    finally:
        runners.close()
    return Generation(
        video_latent=latents["video"],
        audio_latent=latents["audio"],
        frames=frames,
        audio=audio,
    )
