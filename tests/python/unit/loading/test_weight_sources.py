"""A plugin's weight source reads as the checkpoint files it stands for.

A source whose tensors a plugin streams (``Source.stream``) lists the same
names, shapes and element types and reads the same values as a source of
the safetensors files holding those tensors, its encoded weights paired
with their scales as a file's are.
"""

import pytest
import torch
from safetensors.torch import save_file

from uniserve import plugins
from uniserve.loading import checkpoint
from uniserve.loading.config import Config

pytestmark = pytest.mark.unit


class _Stream:
    """The tensors of one checkpoint source, held in storage."""

    def __init__(self, tensors):
        self.values = tensors
        self.reads = []

    def identity(self):
        return "storage:test"

    def tensors(self, source):
        return [
            (name, tuple(value.shape), value.dtype)
            for name, value in self.values.items()
        ]

    def read(self, source, name):
        self.reads.append((source, name))
        return self.values[name]


def _tensors():
    generator = torch.Generator().manual_seed(7)
    return {
        "model.norm.weight": torch.randn(8, generator=generator).bfloat16(),
        "model.embed.weight": torch.randn(16, 8, generator=generator),
        "model.proj.weight": torch.randn(4, 8, generator=generator).to(
            torch.float8_e4m3fn
        ),
        "model.proj.weight_scale": torch.tensor([0.5]),
    }


def test_a_streamed_source_reads_as_its_files(tmp_path):
    tensors = _tensors()
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    files = checkpoint.Source("model", tmp_path, (path,))
    stream = _Stream(tensors)
    streamed = checkpoint.Source(
        "model", tmp_path, (), stream=stream, identity=stream.identity()
    )
    io = Config()
    with files.open(io=io) as expected, streamed.open(io=io) as reader:
        assert reader.names() == expected.names()
        for name in expected.names():
            want, got = expected.get(name), reader.get(name)
            assert (got.shape, got.dtype) == (want.shape, want.dtype), name
            assert type(got) is type(want), name
        # The FP8 weight pairs with its scale.
        weight = reader.get("model.proj.weight")
        assert isinstance(weight, checkpoint.FP8Weight)
        for name in ("model.norm.weight", "model.embed.weight"):
            assert torch.equal(
                reader.get(name).read(), expected.get(name).read()
            )
        region = (slice(2, 6), slice(1, 3))
        assert torch.equal(
            reader.get("model.embed.weight").read(region),
            expected.get("model.embed.weight").read(region),
        )
    assert ("model", "model.norm.weight") in stream.reads


def test_a_weight_source_is_named_by_its_scheme(monkeypatch):
    added = plugins.Registry()
    added.weight_source("storage", lambda uri: ("streamed", uri))
    monkeypatch.setattr(plugins, "_registry", added)
    assert plugins.weight_source("storage://run/7") == (
        "streamed",
        "storage://run/7",
    )
    with pytest.raises(ValueError, match="no plugin streams"):
        plugins.weight_source("ipc://handles")
    with pytest.raises(ValueError, match="names no scheme"):
        plugins.weight_source("/checkpoints/next")
    with pytest.raises(ValueError, match="two plugins stream"):
        added.weight_source("storage", lambda uri: None)
