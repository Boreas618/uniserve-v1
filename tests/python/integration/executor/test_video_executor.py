"""A video denoiser's predicated calls over the FastH3 checkpoint.

The tests play the worker's half of a video denoising computation over the
FastH3 DMD student's transformer on one GPU: they own the latent banks,
admit requests of the shortest admitted clip on the 16:9 canvas, a
text-to-video request in slot 1 and a first-frame request in slot 2, and
run each request's latent preparation and denoising calls as the worker
hands them to the video executor, gated or not by the ``uint8`` device
predicate of a call queued behind an unresolved call
(``uniserve.executor.predicates``).

A call whose predicate is false leaves every state it would change as it
was: a latent preparation keeps the bank it would write and the slot's
tables and conditioning (its prompt's refined features and its encoded
conditions), a denoising step writes the samples it read, unadvanced. A
true predicate runs the call as an ungated call does; the steps compare
exactly, as the image executor's do, because a predicated call runs the
same kernels on the same inputs and selects their result.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from uniserve.bindings.workspace import Scratch
from uniserve.executor.video import VideoExecutor
from uniserve.loading.rank import Description, Loaded
from uniserve.models import registry as models
from uniserve.models.components import capability_tags, describe_components
from uniserve.models.minimax_h3 import weight_config

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.model("minimax_h3"),
    pytest.mark.slow,
]

DEVICE = torch.device("cuda", 0)
DENOISER = "transformer"
# Slot 1 holds the text-to-video request, slot 2 the first-frame request.
SLOTS = 2
# A 64-token prompt, below the one text capacity the rank prepares.
TOKENS = 64
# The worker's names of the element types a manifest states.
DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}


def _key(slot: int) -> dict:
    return {"engine_id": 1, "request_id": slot, "request_epoch": 1}


def _output(manifest: dict, name: str) -> dict:
    """The manifest's product ``name``."""
    return next(
        output
        for outputs in manifest["outputs"].values()
        for output in outputs
        if output["name"] == name
    )


class Worker:
    """The worker's half of the denoiser computation, on one stream."""

    def __init__(self, checkpoint: str) -> None:
        config = models.read_config(checkpoint, modules=frozenset({DENOISER}))
        model = models.load_model(
            config,
            device=DEVICE,
            weights=weight_config(config.model, preset="quality"),
        ).model
        calls = describe_components(model, entries=config.entry_points)
        loaded = Loaded(
            module=model,
            description=Description(
                stub=False,
                model=model,
                metadata=config,
                path=checkpoint,
                io=config.io,
                entries=config.entry_points,
                calls=calls,
                selections={},
                expert_paths=(),
            ),
            root=model,
            attention="auto",
        )
        # The placement the worker states of a rank holding the denoiser
        # alone: its calls' capabilities and paths, and the call kinds a
        # video denoiser serves.
        placed = {
            "name": DENOISER,
            "host": False,
            "calls": [
                {"capability": tag, "path": call.path}
                for call in calls[DENOISER]
                for tag in capability_tags(call.module)
            ],
            "kinds": ["latent_preparation", "denoising"],
            "ranks": [0],
            "distributed": False,
            "units_per_rank": 1,
        }
        self.executor = VideoExecutor(
            loaded,
            {
                "rank": 0,
                "device": str(DEVICE),
                "graphs": False,
                "attention": "auto",
                "model_dtype": "bfloat16",
                "max_model_len": 1024,
                "components": [placed],
                "video": {
                    "denoiser": DENOISER,
                    "resolutions": ["768p"],
                    "aspect_ratios": ["16:9"],
                    "max_seconds": 4.0,
                    "min_seconds": None,
                    "text_capacities": [1024],
                    "max_condition_rows": None,
                    "ffmpeg": "ffmpeg",
                    "transient_layouts": SLOTS,
                },
            },
        )
        manifest = self.executor.manifest()
        features = _output(manifest, "conditioning")
        self.feature_width = int(features["shape_bound"]["dims"][1]["static"])
        self.feature_dtype = DTYPES[features["dtype"]]
        latents = _output(manifest, "condition_video_latents")
        self.latent_width = int(latents["shape_bound"]["dims"][1]["static"])
        self.latent_dtype = DTYPES[latents["dtype"]]
        canvas = manifest["denoiser"]["canvases"][0]["canvas"]
        self.width, self.height = int(canvas["width"]), int(canvas["height"])
        # A keyframe fitted to the canvas encodes to one denoiser row per
        # 32x32-pixel patch (``models::video::plan::rows_per_frame``).
        self.keyframe_rows = (self.width // 32) * (self.height // 32)
        self.frames = min(int(frames) for frames in manifest["frame_counts"])

        # The latent banks the worker allocates: page zero is padding, then
        # each request slot's pages of ``page_units`` float32 elements.
        self.slot_pages = int(manifest["pages"]["pages"])
        self.banks = tuple(
            torch.zeros(
                (
                    1 + SLOTS * self.slot_pages,
                    int(manifest["pages"]["page_units"]),
                ),
                dtype=torch.float32,
                device=DEVICE,
            )
            for _ in range(2)
        )
        self.stream = torch.cuda.Stream(DEVICE)
        scratch = Scratch()
        self.executor.bind(
            {
                "slots": SLOTS,
                "reports": None,
                "ffmpeg": "ffmpeg",
                "transient_layouts": SLOTS,
            },
            {"latent_preparation": self.stream, "denoising": self.stream},
            {},
            {"latent_preparation": scratch, "denoising": scratch},
            self.banks,
        )
        keyframe = {
            "role": "first_frame",
            "media": {
                "image": {"size": {"width": self.width, "height": self.height}}
            },
            "vision": None,
            "latent_units": [self.keyframe_rows],
            "audio_rows": 0,
        }
        refused = self.executor.start(
            [
                self._request(1, "t2va", []),
                self._request(2, "fl2va", [keyframe]),
            ]
        )
        assert refused == []

    def _request(self, slot: int, task: str, conditions: list) -> dict:
        """A started request of ``task`` in ``slot`` as the engine sends it."""
        return {
            "request_key": _key(slot),
            "request_pool_idx": slot,
            "prompt_token_ids": list(range(1000, 1000 + TOKENS)),
            "diffusion": {
                "width": self.width,
                "height": self.height,
                "num_frames": self.frames,
                "seed": 10 + slot,
            },
            "video": {
                "conditioning": json.dumps(
                    {
                        "task": task,
                        "text_tags": [1] * TOKENS,
                        "conditions": conditions,
                    }
                )
            },
        }

    def features(self, seed: int) -> torch.Tensor:
        """Prompt features of the text encoder's width, drawn from ``seed``."""
        values = torch.randn(
            (TOKENS, self.feature_width),
            generator=torch.Generator().manual_seed(seed),
        )
        return values.to(device=DEVICE, dtype=self.feature_dtype)

    def keyframe(self, seed: int) -> torch.Tensor:
        """A keyframe's encoded latent rows, drawn from ``seed``."""
        values = torch.randn(
            (self.keyframe_rows, self.latent_width),
            generator=torch.Generator().manual_seed(seed),
        )
        return values.to(device=DEVICE, dtype=self.latent_dtype)

    def bank(self, index: int, slot: int) -> torch.Tensor:
        """Slot ``slot``'s pages of bank ``index``, once the stream drained."""
        self.stream.synchronize()
        pages = slice(
            1 + (slot - 1) * self.slot_pages, 1 + slot * self.slot_pages
        )
        return self.banks[index][pages].clone()

    def run(
        self,
        kind: str,
        slot: int,
        latent: dict,
        reads: list[torch.Tensor],
        predicate: bool | None,
    ) -> None:
        """Run one call of ``kind``, gated by a device predicate if given.

        The worker appends a predicated call's ``uint8 [1]`` predicate after
        the batch's other tensors and names its position in the call's entry.
        """
        entry = {
            "index": 0,
            "refused": False,
            "active": True,
            "slot": slot,
            "call": {
                "kind": kind,
                "request_key": _key(slot),
                "inputs": [{} for _ in reads],
                "outputs": [],
            },
            "output_names": [],
            "latent": latent,
        }
        tensors = list(reads)
        if predicate is not None:
            entry["predicate"] = len(tensors)
            tensors.append(
                torch.tensor([int(predicate)], dtype=torch.uint8, device=DEVICE)
            )
        step = {"calls": [entry]}
        step["prepared"] = self.executor.prepare(step)
        assert step["prepared"][0]["reads"] == list(range(len(reads)))
        self.executor.execute(step, tensors)

    def prepare_latents(self, slot, write, reads, predicate=None) -> None:
        self.run("latent_preparation", slot, {"write": write}, reads, predicate)

    def denoise(self, slot, read, write, step, predicate=None) -> None:
        latent = {"read": read, "write": write, "step": step}
        self.run("denoising", slot, latent, [], predicate)

    def close(self) -> None:
        self.stream.synchronize()
        self.executor.release(list(range(1, SLOTS + 1)))
        self.executor.close(False)


@pytest.fixture(scope="module")
def worker():
    checkpoint = os.environ.get("UNISERVE_H3_MODEL", "")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail(
            "UNISERVE_H3_MODEL must name the supported full FastH3 VSA "
            "checkpoint directory"
        )
    worker = Worker(checkpoint)
    yield worker
    worker.close()


@pytest.mark.parametrize("slot", [1, 2], ids=["text_to_video", "first_frame"])
@torch.inference_mode()
def test_false_predicates_leave_the_trajectory_as_it_was(worker, slot):
    # The request's own preparation inputs, and others a skipped
    # preparation reads: other prompt features and, for the first-frame
    # request, another keyframe.
    own = [worker.features(1)]
    other = [worker.features(2)]
    if slot == 2:
        own.append(worker.keyframe(3))
        other.append(worker.keyframe(4))

    # The reference trajectory: the request's noise in bank 0, then two
    # ungated steps alternating the banks.
    worker.prepare_latents(slot, 0, own)
    noise = worker.bank(0, slot)
    assert noise.abs().sum() > 0
    worker.denoise(slot, 0, 1, 0)
    first = worker.bank(1, slot)
    worker.denoise(slot, 1, 0, 1)
    second = worker.bank(0, slot)
    assert not torch.equal(first, noise)
    assert not torch.equal(second, first)

    # A skipped preparation of other inputs keeps the bank it would write,
    # the slot's tables and its conditioning.
    worker.prepare_latents(slot, 1, other, predicate=False)
    assert torch.equal(worker.bank(1, slot), first)
    # A skipped step writes the samples it read, unadvanced.
    worker.denoise(slot, 1, 0, 1, predicate=False)
    assert torch.equal(worker.bank(0, slot), first)
    # A true predicate steps as the ungated step did, from the state the
    # skipped calls left: the conditioning of the request's own inputs.
    worker.denoise(slot, 1, 0, 1, predicate=True)
    assert torch.equal(worker.bank(0, slot), second)
    # A true preparation writes the request's noise as an ungated one does.
    worker.prepare_latents(slot, 1, own, predicate=True)
    assert torch.equal(worker.bank(1, slot), noise)
