"""An image encoding's features in the host encoder pool, over the stub
model's vision and latent encoders on one GPU.

The tests play the worker's half of an image encoding whose features lie
in the host encoder pool (``Call::host_output``): the worker binds a view of
each pool page the features' rows fall on and names, for each, the byte
offset in the features, the byte offset in the page and the bytes it holds
(``crates/worker/src/executor/image.rs::host_copies``). An encoding whose
features the pool does not hold encodes the image and writes the features'
rows to the pages; an encoding whose features the pool holds reads them
from the pages instead of encoding, so its features are the stored ones,
byte for byte, whatever image it was handed. The pages are page-locked
host tensors whose size makes their boundaries fall inside rows, as the
pool's pages do. The rank states each encoder's row bytes and whether its
features follow the request's draw, which the engine keys and sizes the
stored features by.
"""  # noqa: D205

from __future__ import annotations

import json

import pytest
import torch

from tests.python.fixtures.stub import LOCAL, load_model, read_config
from uniserve.bindings.workspace import Scratch
from uniserve.executor.image import ImageBinding, ImageExecutor, StreamBinding

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="the image computations run on a GPU",
    ),
]

DEVICE = torch.device("cuda", 0)
KEY = {"engine_id": 1, "request_id": 3, "request_epoch": 1}
# Bytes of a page: not a multiple of a feature row's bytes, so rows
# straddle pages.
PAGE_BYTES = 1031
# The image each encoding reads: 32 x 32 pixels of a resident product.
IMAGE = {"shape": [3, 32, 32], "height": 32, "width": 32, "value_range": "unit"}


def _settings() -> dict:
    return {
        "graphs": False,
        "model_dtype": "bfloat16",
        "components": [
            {
                "name": "model",
                "host": False,
                "calls": [
                    {
                        "capability": "patch_autoencoder",
                        "path": "latent_encoder",
                    }
                ],
                "kinds": ["latent_encoding"],
            },
            {
                "name": "vision_encoder",
                "host": False,
                "calls": [
                    {"capability": "patch_encoder", "path": "vision_encoder"}
                ],
                "kinds": ["vision_encoding"],
            },
        ],
        "image": {
            "max_batch_calls": 1,
            "max_batch_tokens": 64,
            "max_request_slots": 2,
            "max_model_len": 64,
            "flow_shapes": [[32, 32]],
        },
    }


def _copies(rows_bytes: int) -> list[tuple[int, int, int, int]]:
    """The copies of features of ``rows_bytes`` bytes stored from row zero:
    ``(features offset, page, page offset, bytes)``, split at every page
    boundary.
    """  # noqa: D205
    copies, start = [], 0
    while start < rows_bytes:
        page, offset = divmod(start, PAGE_BYTES)
        size = min(PAGE_BYTES - offset, rows_bytes - start)
        copies.append((start, page, offset, size))
        start += size
    return copies


class _Encoders:
    """The worker's half of a rank holding the stub's two encoders."""

    def __init__(self) -> None:
        stub = load_model(
            read_config(
                components=json.dumps(
                    {"model": LOCAL, "vision_encoder": LOCAL}
                ),
                device="cuda:0",
            ),
            device="cuda:0",
        )
        self.executor = ImageExecutor(stub, _settings())
        self.manifest = self.executor.manifest()
        self.stream = torch.cuda.Stream(DEVICE)
        computations = self.executor.computations()
        self.computations = {
            kind: index
            for index, entry in enumerate(computations)
            for kind in entry["kinds"]
        }
        self.executor.bind(
            ImageBinding(
                slots=2,
                computations=tuple(
                    StreamBinding(stream=self.stream, scratch=Scratch())
                    for _ in computations
                ),
            )
        )
        self.executor.start(
            [
                {
                    "request_key": KEY,
                    "request_pool_idx": 1,
                    "generation": {"negative_token_ids": []},
                    "image": None,
                    "input_images": 0,
                }
            ]
        )

    def encode(
        self,
        kind: str,
        pixels: torch.Tensor,
        pages: list[torch.Tensor] | None,
        imported: bool,
    ) -> torch.Tensor:
        """Run one ``kind`` encoding of ``pixels`` into new features, whose
        rows lie in ``pages`` when given: read from them when ``imported``,
        written to them otherwise. Returns the features once the stream
        drained.
        """  # noqa: D205
        computation = self.computations[kind]
        call = {"kind": kind, "request_key": KEY}
        if kind == "latent_encoding":
            call["latent_draw"] = 0
        entry = {
            "index": 0,
            "slot": 1,
            "active": True,
            "refused": False,
            "call": call,
            "output_names": [],
            "latent": None,
            "image": {"input": {"resident": IMAGE}, "input_images": 0},
        }
        step = {"batch_id": 1, "computation": computation, "calls": [entry]}
        prepared = self.executor.prepare(computation, step)
        [write] = prepared[0]["image"]["writes"]
        rows, width = (int(extent) for extent in write["shape"])
        # Features an import must fill whole start as no encoding.
        features = torch.full(
            (rows, width), float("nan"), dtype=torch.bfloat16, device=DEVICE
        )
        views = []
        if pages is not None:
            copies = _copies(features.numel() * features.element_size())
            entry["host"] = {
                "import": imported,
                "copies": [
                    [start, offset, size] for start, _, offset, size in copies
                ],
            }
            views = [pages[page] for _, page, _, _ in copies]
        self.executor.execute(
            computation,
            {**step, "prepared": prepared},
            [pixels, features, *views],
        )
        self.stream.synchronize()
        return features

    def close(self) -> None:
        self.stream.synchronize()
        self.executor.release([1])
        self.executor.close(False)


@pytest.fixture(scope="module")
def encoders():
    encoders = _Encoders()
    yield encoders
    encoders.close()


def _pixels(start: float) -> torch.Tensor:
    pixels = torch.linspace(start, 1, 3 * 32 * 32, device=DEVICE)
    return pixels.reshape(3, 32, 32).to(torch.bfloat16)


@torch.inference_mode()
@pytest.mark.parametrize(
    ("kind", "output"),
    [
        ("vision_encoding", "vision_features"),
        ("latent_encoding", "latent_features"),
    ],
)
def test_missing_features_are_encoded_and_stored_then_read_byte_for_byte(
    encoders, kind, output
):
    # The rank states the bytes of the encoder's feature rows; the stub's
    # encoders follow the image alone, its latent encoder taking its
    # posterior's mean.
    stated = encoders.manifest["encoders"][output]
    assert stated["draws"] is False

    # The pool holds no features: the encoding encodes the image and writes
    # the features' bytes to the pages, in order.
    pixels = _pixels(0.0)
    computed = encoders.encode(kind, pixels, None, imported=False)
    assert computed.shape[1] * computed.element_size() == stated["row_bytes"]
    total = computed.numel() * computed.element_size()
    pages = [
        torch.zeros(PAGE_BYTES, dtype=torch.uint8).pin_memory()
        for _ in range(_copies(total)[-1][1] + 1)
    ]
    exported = encoders.encode(kind, pixels, pages, imported=False)
    assert torch.equal(exported, computed)
    stored = torch.cat(pages)[:total]
    assert torch.equal(stored, computed.cpu().view(-1).view(torch.uint8))

    # The pool holds them: an encoding handed another image reads the
    # stored features instead of encoding it, byte for byte.
    other = _pixels(0.5)
    assert not torch.equal(
        encoders.encode(kind, other, None, imported=False), computed
    )
    read = encoders.encode(kind, other, pages, imported=True)
    assert torch.equal(
        read.view(-1).view(torch.uint8), computed.view(-1).view(torch.uint8)
    )
