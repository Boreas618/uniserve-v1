"""SenseNova U1 capability composition with one shared MoT backbone."""

from __future__ import annotations

from types import MappingProxyType

import torch
from torch import nn

from uniserve.models import (
    DEFAULT_COMPONENT,
    CausalLM,
    ComponentEntry,
    EntryPoint,
    ImageDecoder,
    PatchEncoder,
)
from uniserve.nn.linear import VocabParallelHead
from uniserve.nn.vae.patch import RGBDecoder

from . import vision
from .config import Config
from .denoiser import Denoiser
from .processing import MAX_IMAGE_PIXELS
from .transformer import Transformer

# Images one packed vision call holds, each in a slot of the largest image's
# patch rows. The largest packed graph's activations (this many slots)
# bound the graph pool every slot count shares; a larger batch replays that
# graph once per group.
VISION_SLOTS = 4


class Model(nn.Module):
    """Compose text, denoising, and vision over one shared MoT backbone.

    The text LM and the image denoiser share the same transformer instance:
    text calls run through its ``text`` experts and denoising through its
    ``flow`` experts, over one attention prefix cache. The vision encoder turns
    input images into NEO patch features that the worker prefills into the
    text model as token embeddings, and the image decoder unfolds the
    denoiser's pixel-space patch rows back into RGB images. The denoiser owns
    a second ``vision.Encoder`` with separate checkpoint weights for its own
    image input.
    """

    function = "models.sensenova_u1.model.Model/1"

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        backbone = Transformer(config.text)
        self.text = CausalLM(
            backbone,
            VocabParallelHead(config.text.hidden_size, config.text.vocab_size),
        )
        # Construction precedes pipeline binding, so both modules are present.
        if config.text.tie_word_embeddings:
            embedding, head = backbone.embedding, self.text.lm_head
            if embedding is None or head is None:
                raise ValueError(
                    "tied embeddings require the embedding and head"
                )
            head.weight = embedding.weight
        self.denoiser = Denoiser(config, backbone)

        # A slot holds the largest image the ViT transform prepares.
        self.vision_encoder = PatchEncoder(
            vision.Encoder(config.vision),
            nn.Identity(),
            patch_size=config.vision.patch_size,
            downsample=round(1 / config.vision.downsample_ratio),
            output_size=config.text.hidden_size,
            output_dtype=torch.bfloat16,
            max_patches=MAX_IMAGE_PIXELS // config.vision.patch_size**2,
            max_slots=VISION_SLOTS,
        )
        self.image_decoder = ImageDecoder(RGBDecoder(self.denoiser.patch_size))


def entry_points(config: Config):
    """Declare the model's single IPC component and its callable methods.

    Token embedding runs on the first pipeline stage and logits on the last;
    the vision encoder and image decoder join no communication group.
    """
    return MappingProxyType(
        {
            DEFAULT_COMPONENT: ComponentEntry(
                "",
                (
                    EntryPoint("text.forward", groups=("tp", "sp", "pp")),
                    EntryPoint("text.embed_input_ids", "first", ("tp",)),
                    EntryPoint("text.compute_logits", "last", ("tp",)),
                    EntryPoint("denoiser.forward", groups=("tp", "sp", "pp")),
                    EntryPoint("vision_encoder.encode"),
                    EntryPoint("image_decoder.decode"),
                ),
            ),
        }
    )
