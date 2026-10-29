"""The identities that key what a model's components compute and load."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from uniserve import numerics
from uniserve.errors import SetupError
from uniserve.nn.linear import Linear
from uniserve.nn.norm import RMSNorm

pytestmark = pytest.mark.unit


class _Block(nn.Module):
    """A component of a test model."""

    function = "tests.block/1"

    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.norm = RMSNorm(8, dtype=dtype)
        self.projection = Linear(8, 8, dtype=dtype)
        self.layers = nn.ModuleList([RMSNorm(8, dtype=dtype)])


def test_equal_models_have_one_identity_and_a_revision_or_format_changes_it(
    monkeypatch,
):
    identity = numerics.identity(_Block())
    assert identity == numerics.identity(_Block())

    # Another revision of a component's function, however deep in the tree.
    monkeypatch.setattr(RMSNorm, "function", "nn.norm.RMSNorm/2")
    revised = numerics.identity(_Block())
    assert revised != identity
    monkeypatch.undo()

    # Another element type of the tensors a component holds.
    assert numerics.identity(_Block(torch.bfloat16)) != identity


def test_an_undeclared_component_has_no_identity():
    class Undeclared(nn.Module):
        """A module that states no function."""

    model = _Block()
    model.extra = Undeclared()
    with pytest.raises(SetupError, match="declares no function"):
        numerics.identity(model)


def test_pytorch_modules_compute_their_documented_function():
    assert numerics.function(nn.ModuleList) == "torch.nn.ModuleList/1"
    assert numerics.function(nn.Embedding) == "torch.nn.Embedding/1"


def test_a_dependencys_modules_compute_its_installed_versions_function():
    from importlib import metadata

    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (
        MiniMaxH3AudioBigVGANDecoder,
    )

    assert numerics.function(MiniMaxH3AudioBigVGANDecoder) == (
        "diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio."
        f"MiniMaxH3AudioBigVGANDecoder/{metadata.version('diffusers')}"
    )
