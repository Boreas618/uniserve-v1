"""A video encoder's output in the host encoder pool, over the FastH3 text
encoder on one GPU.

The tests play the worker's half of a text encoding call whose output lies
in the host encoder pool (``Call::host_output``): the worker binds a view of
each pool page the output's rows fall on and names, for each, the byte
offset in the call's output, the byte offset in the page and the bytes it
holds (``crates/worker/src/executor/products.rs::host_copies``). A call whose
output the pool does not hold computes it and writes its rows to the
pages; a call whose output the pool holds reads its rows from the pages
instead of encoding, and its output is the computed one, byte for byte.
The pages here are page-locked host tensors whose size makes their
boundaries fall inside rows, as the pool's pages do.
"""  # noqa: D205

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
ENCODER = "text_encoder"
# A 64-token prompt, below the one text capacity the rank prepares.
TOKENS = 64
# Bytes of a page: not a multiple of a row's bytes, so rows straddle pages.
PAGE_BYTES = 4099
# The worker's names of the element types a manifest states.
DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}


def _copies(rows: int, row_bytes: int) -> list[tuple[int, int, int, int]]:
    """The copies of an output's ``rows`` rows of ``row_bytes`` bytes
    stored from row zero: ``(output offset, page, page offset, bytes)``,
    split at every page boundary.
    """  # noqa: D205
    copies, start, end = [], 0, rows * row_bytes
    while start < end:
        page, offset = divmod(start, PAGE_BYTES)
        size = min(PAGE_BYTES - offset, end - start)
        copies.append((start, page, offset, size))
        start += size
    return copies


class Encoder:
    """The worker's half of a rank holding the text encoder alone."""

    def __init__(self, checkpoint: str) -> None:
        # The executor reads the denoiser's facts from its loaded module,
        # which the rank does not execute.
        config = models.read_config(
            checkpoint, modules=frozenset({DENOISER, ENCODER})
        )
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
        placed = {
            "name": ENCODER,
            "host": False,
            "calls": [
                {"capability": tag, "path": call.path}
                for call in calls[ENCODER]
                for tag in capability_tags(call.module)
            ],
            "kinds": ["text_encoding"],
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
                    "transient_layouts": 1,
                },
            },
        )
        manifest = self.executor.manifest()
        self.output = next(
            output
            for outputs in manifest["outputs"].values()
            for output in outputs
            if output["name"] == "conditioning"
        )
        self.width = int(self.output["shape_bound"]["dims"][1]["static"])
        self.dtype = DTYPES[self.output["dtype"]]
        self.stream = torch.cuda.Stream(DEVICE)
        scratch = Scratch()
        self.executor.bind(
            {
                "slots": 1,
                "reports": None,
                "ffmpeg": "ffmpeg",
                "transient_layouts": 1,
            },
            {"text_encoding": self.stream},
            {},
            {"text_encoding": scratch},
            None,
        )
        canvas = manifest["denoiser"]["canvases"][0]["canvas"]
        frames = min(int(frames) for frames in manifest["frame_counts"])
        refused = self.executor.start(
            [
                {
                    "request_key": self.key,
                    "request_pool_idx": 1,
                    "prompt_token_ids": list(range(1000, 1000 + TOKENS)),
                    "diffusion": {
                        "width": int(canvas["width"]),
                        "height": int(canvas["height"]),
                        "num_frames": frames,
                        "seed": 11,
                    },
                    "video": {
                        "conditioning": json.dumps(
                            {
                                "task": "t2va",
                                "text_tags": [1] * TOKENS,
                                "conditions": [],
                            }
                        )
                    },
                }
            ]
        )
        assert refused == []

    key = {"engine_id": 1, "request_id": 1, "request_epoch": 1}

    @property
    def row_bytes(self) -> int:
        return self.width * torch.tensor([], dtype=self.dtype).element_size()

    def encode(
        self, pages: list[torch.Tensor] | None, imported: bool
    ) -> torch.Tensor:
        """Run the request's text encoding into a new output, whose rows
        lie in ``pages`` when given: read from them when ``imported``,
        written to them otherwise. Returns the output once the stream
        drained.
        """  # noqa: D205
        entry = {
            "index": 0,
            "refused": False,
            "active": True,
            "slot": 1,
            "call": {
                "kind": "text_encoding",
                "request_key": self.key,
                "inputs": [],
                "outputs": [self.output],
            },
            "output_names": ["conditioning"],
        }
        views = []
        if pages is not None:
            copies = _copies(TOKENS, self.row_bytes)
            entry["host"] = {
                "import": imported,
                "copies": [
                    [start, offset, size] for start, _, offset, size in copies
                ],
            }
            views = [pages[page] for _, page, _, _ in copies]
        step = {"calls": [entry]}
        step["prepared"] = self.executor.prepare(step)
        [write] = step["prepared"][0]["writes"]
        assert [int(extent) for extent in write["shape"]] == [
            TOKENS,
            self.width,
        ]
        # An output an import must fill whole starts as no encoding.
        output = torch.full(
            (TOKENS, self.width), float("nan"), dtype=self.dtype, device=DEVICE
        )
        self.executor.execute(step, [output, *views])
        self.stream.synchronize()
        return output

    def close(self) -> None:
        self.stream.synchronize()
        self.executor.release([1])
        self.executor.close(False)


@pytest.fixture(scope="module")
def encoder():
    checkpoint = os.environ.get("UNISERVE_H3_MODEL", "")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail(
            "UNISERVE_H3_MODEL must name the supported full FastH3 VSA "
            "checkpoint directory"
        )
    encoder = Encoder(checkpoint)
    yield encoder
    encoder.close()


@torch.inference_mode()
def test_a_missing_output_is_computed_and_stored_then_read_byte_for_byte(
    encoder,
):
    copies = _copies(TOKENS, encoder.row_bytes)
    pages = [
        torch.zeros(PAGE_BYTES, dtype=torch.uint8).pin_memory()
        for _ in range(copies[-1][1] + 1)
    ]

    # The pool holds no output: the call encodes the prompt and writes the
    # output's bytes to the pages, in order.
    computed = encoder.encode(pages, imported=False)
    assert not computed.isnan().any()
    stored = torch.cat(pages)[: TOKENS * encoder.row_bytes]
    assert torch.equal(stored, computed.cpu().view(-1).view(torch.uint8))

    # The pool holds it: the call reads the computed output, byte for byte.
    read = encoder.encode(pages, imported=True)
    assert torch.equal(
        read.view(-1).view(torch.uint8), computed.view(-1).view(torch.uint8)
    )
