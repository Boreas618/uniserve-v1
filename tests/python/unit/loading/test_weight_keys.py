"""Retained tensors depend on their own checkpoint and numerical inputs."""

import hashlib
from dataclasses import dataclass, replace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from uniserve import loading
from uniserve.loading import checkpoint, weights
from uniserve.loading.retained import fingerprints
from uniserve.models import registry
from uniserve.nn.linear import Linear
from uniserve.quantization import QuantizationConfig, Quantizer

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class Config:
    eps: float = 1e-6
    frequency: float = 1.0
    offset_function: str = "tests.offset/1"
    projection_source: str = "projection"
    tied: bool = False
    preserve_dtype: bool = False


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.projection = Linear(8, 8, bias=False)
        self.head = Linear(8, 8, bias=False)
        if config.tied:
            self.head.weight = self.projection.weight
        self.norm = nn.LayerNorm(8, eps=config.eps, elementwise_affine=False)
        self.register_buffer(
            "frequency", torch.tensor(config.frequency, device="cpu")
        )
        self.register_buffer("offset", torch.empty(8))


def mappings(model):
    def derive(reader):
        model.offset = reader.get("offset").read().square()

    return (
        weights.ModuleMapping(
            model,
            "primary",
            lambda reader: (
                weights.Assignment(
                    model.projection.weight,
                    reader.get(model.config.projection_source),
                    preserve_dtype=model.config.preserve_dtype,
                ),
                weights.Assignment(
                    model.head.weight,
                    reader.get(
                        model.config.projection_source
                        if model.config.tied
                        else "head"
                    ),
                    preserve_dtype=model.config.preserve_dtype,
                ),
            ),
            frozenset(("projection.weight", "head.weight")),
            derivations=(
                weights.Derivation(
                    model.config.offset_function,
                    ("offset",),
                    ("offset",),
                    {},
                    derive,
                ),
            ),
        ),
    )


@pytest.fixture
def source(tmp_path):
    values = {
        "projection": torch.arange(64, dtype=torch.float32).reshape(8, 8),
        "head": torch.eye(8),
        "offset": torch.arange(8, dtype=torch.float32),
    }
    files = tuple(
        _save_blob(tmp_path, name, tensor) for name, tensor in values.items()
    )
    io = loading.Config()
    return registry.Config(
        model=Config(),
        model_class=Model,
        checkpoint=(checkpoint.Source("primary", tmp_path, files),),
        mapping=mappings,
        entry_points={},
        weights=weights.Config(dtype=torch.float32),
        precisions={},
        checkpoint_format=None,
        io=io,
        tokenizer=None,
        image_processor=None,
        flow_prompt=None,
        modules=None,
    )


def _save_blob(root, name, tensor):
    saved = root / f"{name}.safetensors"
    if saved.is_symlink():
        saved.unlink()
    save_file({name: tensor}, saved)
    # Immutable Hub-style blobs have stable identities immediately after
    # creation; the named source follows the blob just as a snapshot does.
    blob = root / "blobs" / hashlib.sha256(saved.read_bytes()).hexdigest()
    blob.parent.mkdir(exist_ok=True)
    saved.rename(blob)
    saved.symlink_to(blob)
    return saved


def keys(source):
    return fingerprints(source, source.io, "local")["tensors"]


def test_numerical_options_that_load_no_weights_keep_weight_keys(source):
    before = keys(source)
    assert (
        keys(replace(source, model=replace(source.model, eps=1e-5))) == before
    )


@pytest.mark.parametrize(
    ("change", "affected"),
    (
        ({"frequency": 2.0}, "b:frequency"),
        ({"offset_function": "tests.offset/2"}, "b:offset"),
        ({"projection_source": "head"}, "p:projection.weight"),
    ),
)
def test_a_tensor_change_preserves_every_other_tensor_key(
    source, change, affected
):
    before = keys(source)
    after = keys(replace(source, model=replace(source.model, **change)))
    assert {name for name in before if before[name] != after[name]} == {
        affected
    }


def test_precision_choices_are_local_to_their_numerical_module(source):
    before = keys(source)
    quantizer = Quantizer("fp8", axis=0)
    selected = replace(
        source,
        weights=weights.Config(
            dtype=torch.float32,
            quantization={"projection": QuantizationConfig(quantizer, None)},
        ),
    )
    after = keys(selected)
    assert {name for name in before if before[name] != after[name]} == {
        "p:projection.weight"
    }


def test_tied_parameters_have_keys_under_every_resident_alias(source):
    result = keys(replace(source, model=replace(source.model, tied=True)))
    assert result["p:projection.weight"] == result["p:head.weight"]


def test_preserved_checkpoint_dtype_ignores_the_default_weight_dtype(source):
    source = replace(source, model=replace(source.model, preserve_dtype=True))
    before = keys(source)
    after = keys(replace(source, weights=weights.Config(dtype=torch.bfloat16)))
    assert before == after


def test_changing_one_checkpoint_shard_keeps_unrelated_tensors(source):
    before = keys(source)
    _save_blob(source.checkpoint[0].root, "head", torch.eye(8) * 2)
    after = keys(source)
    assert {name for name in before if before[name] != after[name]} == {
        "p:head.weight"
    }


def test_a_rank_of_some_modules_keys_every_tensor_of_the_model_alike(source):
    # Ranks that receive weights together select other worlds' retained
    # fields by the model's tensor identities, so a rank holding only some
    # modules must state the same identities as a rank holding them all,
    # while its own identities name only the tensors it holds.
    whole = fingerprints(source, source.io, "local")
    narrowed = fingerprints(
        replace(source, modules=frozenset(("projection",))),
        source.io,
        "local",
    )
    assert narrowed["tensors"] == {
        "p:projection.weight": whole["tensors"]["p:projection.weight"]
    }
    assert narrowed["model_tensors"] == whole["model_tensors"]
    assert whole["model_tensors"] == whole["tensors"]
