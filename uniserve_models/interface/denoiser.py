"""Image denoiser bindings: KV-conditioned latent prediction and updates.

The worker's ``ImageDiffusion`` runner (``crates/worker/src/runner/
diffusion/image.rs``) serves the ``ImageDenoiser`` at ``path`` within a
bound component. ``denoiser.bind(prepared, loaded, path, buffers, layout)``
binds the worker's buffers once:

- ``samples``: the latent pool's step workspace ``[units, width]`` on the
  denoiser's device, where the worker gathers each trajectory's committed
  pages before a step and from which it scatters the updated pages after
  it;
- ``positions`` ``[3, T]`` int64, ``timesteps`` ``[R, 1]`` float32 and
  ``predictions`` ``[P, width]``: the worker's per-row positions and
  timesteps, and the predictions of every row packed in row order;
- ``step``: a ``[1]`` int64 zero, the evaluation index the network's input
  carries;
- ``slot_timesteps``: the latent pool's ``[slots + 1, 1]`` float32
  per-slot timesteps the solver update reads;
- ``products``: the device's product arena (latent feature products);
- the attention columns and slot tables of
  ``uniserve_models.interface.attention``, with ``layout`` their tables'
  page geometry.

A row is one guidance branch of one image: ``[sample, units, height,
width, token, tokens, prediction]``: its latent rows
``samples[sample:sample + units]``, its size, its positions
``positions[:, token:token + tokens]`` and its prediction rows
``predictions[prediction:prediction + units]``. Row ``i`` reads
``timesteps[i]``.

Step functions, each on the binding's stream:

- ``plan(steps, shift, text_scale, image_scale, interval_low,
  interval_high, renorm, renorm_min) -> [plan, timesteps, branches]``:
  the schedules and guidance of one trajectory (an opaque plan the worker
  passes back to ``integrate``), each solver step's model
  timestep and the mask of guidance branches it evaluates (bit 0
  conditioned, 1 text-unconditional, 2 image-unconditional). A
  non-positive ``shift`` keeps the denoiser's default shift.
- ``initial_latent(sample, units, padded, height, width, seed, index)``:
  zero ``samples[sample:sample + padded]`` and draw image
  ``index``'s seeded initial latent into its first ``units`` rows.
- ``predict(graph, rows, attention)``: gather the rows'
  attention metadata (``attention`` = ``[rows, tokens, queries, prefixes,
  starts, widths]``), then evaluate the network (replaying ``graph`` when
  given) and write each row's prediction.
- ``capture(rows, attention, pool, peers) -> graph``: capture the
  evaluation of the geometry the last eager ``predict`` gathered;
  ``peers`` are the ``[device, pool]`` pairs of the other devices the
  network's modules run on (see ``uniserve_models.interface.graphs``).
- ``condition(rows, attention, features)``: write the K/V of
  latent images inserted as conditioning (latent ingest and feedback): row
  ``i`` reads the latent feature product region ``features[i]`` instead of
  samples, at timestep zero, and writes its K/V; its prediction is
  computed and unused.
- ``integrate(calls, plans)``: ``calls`` holds one
  ``[sample, units, slot, step, prediction, branches]`` row per image:
  combine its branch predictions (consecutive from ``prediction``, in
  evaluation order) with its plan's guidance and update its samples in
  place by one solver step toward the next timestep.
- ``release(graph)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from uniserve_models.diffusion import (
    Branch,
    Guidance,
    Renorm,
    Schedule,
    advance_,
    normal_noise,
)
from uniserve_models.media import image
from uniserve_models.model import ImageDenoiser, LatentInput
from uniserve_models.nn.rng import flow_noise_seed

from . import calls, graphs, regions
from .attention import AttentionColumns
from .errors import translate

# Guidance branch mask bits, in evaluation order.
_BITS = {
    Branch.CONDITIONED: 1,
    Branch.TEXT_UNCONDITIONAL: 2,
    Branch.IMAGE_UNCONDITIONAL: 4,
}


def functions() -> dict[str, Callable[..., Any]]:
    return {"denoiser.bind": bind}


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One image trajectory's host schedules and guidance."""

    schedules: dict[str, Schedule]
    guidance: Guidance


def bind(
    prepared,
    loaded,
    path: str,
    buffers: Mapping[str, Any],
    layout: Mapping[str, Any],
) -> dict[str, Callable[..., Any]]:
    """Bind the image denoiser at ``path`` to the worker's buffers.

    Raises:
        UnsupportedCall: The module is not an ``ImageDenoiser``.
    """
    denoiser = calls.module(prepared, path)
    if not isinstance(denoiser, ImageDenoiser):
        raise NotImplementedError(
            f"{type(denoiser).__qualname__} has no image denoiser binding"
        )
    context = prepared.context
    if context is None:
        raise RuntimeError("denoiser binding requires a bound component")
    b = calls.tensors(buffers)
    samples = b["samples"]
    positions, timesteps = b["positions"], b["timesteps"]
    predictions, step = b["predictions"], b["step"]
    slot_timesteps, products = b["slot_timesteps"], b["products"]
    attention = AttentionColumns(b, layout)
    pipeline = denoiser.mesh.get_group(
        "pp" if "pp" in denoiser.mesh.axes else ()
    )

    def plan(
        steps: int,
        shift: float,
        text_scale: float,
        image_scale: float,
        interval_low: float,
        interval_high: float,
        renorm: str,
        renorm_min: float,
    ) -> list:
        shift = float(shift)
        schedules = dict(
            denoiser.make_schedules(
                int(steps), shift=shift if shift > 0 else None, device="cpu"
            )
        )
        guidance = denoiser.make_guidance(
            text_scale=float(text_scale),
            image_scale=float(image_scale),
            interval=(float(interval_low), float(interval_high)),
            renorm=Renorm(str(renorm)),
            renorm_min=float(renorm_min),
        )
        schedule = schedules["image"]
        masks = [
            sum(_BITS[branch] for branch in guidance.branches(schedule, index))
            for index in range(schedule.num_steps)
        ]
        times = [float(value) for value in schedule.timesteps.tolist()]
        return [Trajectory(schedules, guidance), times, masks]

    def initial_latent(
        sample: int,
        units: int,
        padded: int,
        height: int,
        width: int,
        seed: int,
        index: int,
    ) -> None:
        sample, units, padded = int(sample), int(units), int(padded)
        size = image.Config(int(height), int(width))
        out = samples[sample : sample + units]
        if (
            padded < units
            or sample + padded > samples.shape[0]
            or tuple(out.shape) != denoiser.latent_shape("image", size)
        ):
            raise ValueError(
                "image trajectory storage must have its canonical patch shape"
            )
        with torch.inference_mode(), torch.cuda.stream(context.stream):
            # Physical page padding past the trajectory's rows stays zero.
            samples[sample : sample + padded].zero_()
            noise = torch.empty(
                (1, *denoiser.noise_shape("image", size)),
                dtype=out.dtype,
                device=out.device,
            )
            normal_noise(
                (flow_noise_seed(int(seed), int(index)),), out=(noise,)
            )
            denoiser.prepare_latents(
                (size,),
                noise={"image": noise},
                state={"image": out.unsqueeze(0)},
                constants={},
                workspace={},
            )

    def bind_rows(rows, batch, features=None):
        """The network's typed input over the rows' column views."""
        latents = []
        for index, (offset, units, *_) in enumerate(rows):
            values = (
                samples[offset : offset + units]
                if features is None
                else regions.view(products, features[index])
            )
            latents.append(LatentInput(values, timesteps[index : index + 1]))
        return denoiser.bind_inputs(
            latents={"image": tuple(latents)},
            sizes=tuple(image.Config(row[2], row[3]) for row in rows),
            step=step,
            positions=tuple(
                positions[:, row[4] : row[4] + row[5]] for row in rows
            ),
            sequence_lengths=tuple(row[5] for row in rows),
            attention=batch,
        )

    def run(rows, typed) -> None:
        """Evaluate the network and write each row's prediction."""
        result = denoiser(
            typed,
            state={},
            constants=context.constants,
            workspace=context.workspace_views,
        )["image"]
        # Only the last pipeline stage predicts; its broadcast gives every
        # stage the same values. A mesh without ``pp`` broadcasts locally.
        for prediction, row in zip(result, rows, strict=True):
            target = predictions[row[6] : row[6] + row[1]]
            if prediction is not None:
                target.copy_(prediction.tensor.reshape(target.shape))
            pipeline.broadcast(target, src=pipeline.size - 1)

    def evaluate(rows, batch, features=None) -> None:
        """Run the network over the rows and write their predictions."""
        run(rows, bind_rows(rows, batch, features))

    def batch(layout_args, writes: bool):
        """The attention batch the rows read (never causal)."""
        count, tokens, queries, prefixes, starts, widths = layout_args
        return attention.batch(
            count,
            tokens,
            queries,
            prefixes,
            starts,
            [0] * int(count),
            writes,
            widths,
        )

    def gathered(layout_args, writes: bool):
        count, tokens, *_, widths = layout_args
        attention.gather(count, tokens, widths)
        return batch(layout_args, writes)

    def predict(graph, rows, layout_args) -> None:
        rows = _rows(rows)
        with calls.scope(context):
            inputs_batch = gathered(layout_args, False)
            context.bind_attention(inputs_batch, replay=graph is not None)
            if graph is not None:
                graph.replay()
            else:
                evaluate(rows, inputs_batch)

    def condition(rows, layout_args, features) -> None:
        rows = _rows(rows)
        if len(features) != len(rows):
            raise ValueError("latent conditioning rows and features must align")
        with calls.scope(context):
            inputs_batch = gathered(layout_args, True)
            context.bind_attention(inputs_batch, replay=False)
            # Conditioning runs at timestep zero.
            timesteps[: len(rows)].zero_()
            evaluate(rows, inputs_batch, features)

    def capture(rows, layout_args, pool, peers) -> graphs.Graph:
        if context.stream is None:
            raise ValueError("graph capture requires a CUDA binding")
        rows = _rows(rows)
        with torch.inference_mode():
            inputs_batch = batch(layout_args, False)
            context.bind_attention(inputs_batch, replay=False)

        def body():
            # The denoiser binds its inputs inside the step: whatever it
            # derives from the samples (SenseNova's conditioning pixels)
            # must follow each replay's samples.
            with context.activate():
                evaluate(rows, inputs_batch)

        return graphs.capture(
            body, stream=context.stream, pool=pool, peers=peers
        )

    def integrate(rows: Sequence[Sequence[int]], plans: Sequence) -> None:
        rows = [tuple(int(value) for value in row) for row in rows]
        if len(rows) != len(plans):
            raise ValueError("integration rows and plans must align")
        with torch.inference_mode(), torch.cuda.stream(context.stream):
            for (sample, units, slot, index, first, mask), trajectory in zip(
                rows, plans, strict=True
            ):
                schedule = trajectory.schedules["image"]
                guidance = trajectory.guidance
                branches = guidance.branches(schedule, index)
                if sum(_BITS[branch] for branch in branches) != mask:
                    raise ValueError(
                        "guidance branches disagree with the evaluated rows"
                    )
                values = samples[sample : sample + units]
                outputs = {
                    branch: predictions[
                        first + position * units : first
                        + (position + 1) * units
                    ]
                    for position, branch in enumerate(branches)
                }
                guided = guidance.combine(outputs, schedule, index)
                advance_(
                    denoiser,
                    {"image": (LatentInput(values, slot_timesteps[slot]),)},
                    {"image": (guided,)},
                    trajectory.schedules,
                    schedule.step(index),
                )

    steps = {
        "plan": plan,
        "initial_latent": initial_latent,
        "predict": predict,
        "condition": condition,
        "capture": capture,
        "integrate": integrate,
        "release": lambda graph: graph.release(),
    }
    return {name: translate(step) for name, step in steps.items()}


def _rows(rows) -> list[tuple[int, ...]]:
    return [tuple(int(value) for value in row) for row in rows]
