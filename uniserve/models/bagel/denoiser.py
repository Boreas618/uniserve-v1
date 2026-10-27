"""BAGEL latent prediction, guidance and diffusion schedules."""

from __future__ import annotations

import torch

from uniserve.diffusion import (
    EulerSolver,
    NestedGuidance,
    NoiseScale,
    Renorm,
    make_schedule,
)
from uniserve.media import image
from uniserve.models import ImageDenoiser
from uniserve.models.inputs import PackedImageInput, PackedImageRows, TokenKind
from uniserve.nn.linear import Linear
from uniserve.nn.routing import IndexedRoutes, SparseRoute
from uniserve.nn.timestep import TimestepEmbedding
from uniserve.nn.vision import PositionEmbedding
from uniserve.processing import BranchSource
from uniserve.tensors import OutputLayout, TensorOutput

from .config import Config
from .transformer import Transformer


class Denoiser(ImageDenoiser):
    """Predict image-latent velocity through the shared text/flow backbone.

    Each row of a packed forward is one image's framed sequence: the
    start-of-image marker, one token per latent patch in raster order, and
    the end-of-image marker. Every token runs through the flow expert and
    the two markers of each row through the text expert, whose results
    replace the flow expert's before attention mixes tokens; each sequence
    attends to its guidance branch's cached token prefix through the
    attention layers it shares with the text LM. Every operation outside
    attention acts on one token, so the forward is row-local. With pipeline
    parallelism, the first stage builds the input embeddings and only the
    last stage returns predictions.
    """

    function = "models.bagel.denoiser.Denoiser/1"

    # BAGEL frames each generated image with its start and end marker tokens,
    # and its image-unconditional branch reads the conditioning prefix.
    framing_tokens = 2
    image_unconditional = BranchSource.CONDITIONING
    # The start marker sits at the position before the image, the patches
    # share the next one and the end marker takes the one after.
    rope_advance = 2

    markers: torch.Tensor

    def __init__(self, config: Config, backbone: Transformer):
        super().__init__(
            patch_size=config.latent_patch_size,
            latent_channels=config.vae.latent_channels,
            downsample=config.vae.downsample * config.latent_patch_size,
            noise_scale=NoiseScale(1.0, "constant", 1.0, 1.0),
            prediction_dtype=torch.bfloat16,
            solver=EulerSolver("velocity"),
        )
        self.config, self.backbone = config, backbone
        width = config.latent_patch_size**2 * config.vae.latent_channels
        self.input = Linear(width, config.text.hidden_size)
        self.time_embedding = TimestepEmbedding(config.text.hidden_size)
        self.position = PositionEmbedding(
            (config.max_latent_size,) * 2, config.text.hidden_size
        )
        self.prediction = Linear(config.text.hidden_size, width)

        # The marker IDs derive from the config, not the checkpoint. Explicit
        # CPU construction keeps the buffer materialized when the model is
        # built on the meta device; loading then moves it to the module's
        # placement device.
        self.register_buffer(
            "markers",
            torch.tensor(
                (config.start_of_image_id, config.end_of_image_id),
                dtype=torch.long,
                device="cpu",
            ),
            persistent=False,
        )

    @property
    def max_sequence_tokens(self) -> int:
        """Latent rows the learned position grid covers; markers not counted."""
        return self.config.max_latent_size**2

    def noise_shape(self, modality: str, size: image.Config):
        # BAGEL draws noise directly in the canonical [patch rows, patch
        # values] shape, so ``prepare_latents`` scales it without patchifying.
        return self.latent_shape(modality, size)

    def make_schedules(self, steps, *, shift, device):
        # Descending network time equals sigma, from one (noise) to zero.
        return {
            "image": make_schedule(
                steps,
                shift=self.config.timestep_shift if shift is None else shift,
                direction="descending",
                shift_domain="time",
                device=device,
            )
        }

    def make_guidance(
        self,
        *,
        text_scale: float,
        image_scale: float,
        interval: tuple[float, float],
        renorm: Renorm,
        renorm_min: float,
    ):
        return NestedGuidance(
            text_scale, image_scale, interval, renorm, renorm_min
        )

    def _routes(self, rows: PackedImageRows) -> IndexedRoutes:
        """Every token takes the flow expert; each row's markers the text one.

        The text route lists each row's opening token, then each row's
        closing token: ``2 * rows`` entries, of which rows without markers
        (padding and empty rows) keep none. The list is derived on the
        device from the token kinds, so its shape is static.
        """
        tokens = torch.arange(rows.tokens, device=rows.kind.device)
        absent = torch.full_like(tokens, -1)
        markers = []
        for kind in (TokenKind.OPEN, TokenKind.CLOSE):
            # A row holds at most one marker of each kind, which the maximum
            # selects over the row's other tokens' -1.
            row_marker = torch.full(
                (rows.rows,), -1, dtype=torch.int64, device=tokens.device
            )
            markers.append(
                row_marker.scatter_reduce(
                    0,
                    rows.token_row,
                    torch.where(rows.kind == kind, tokens, absent),
                    reduce="amax",
                )
            )
        return IndexedRoutes(
            "flow", (SparseRoute.of("text", torch.cat(markers)),)
        )

    def _embed(
        self, source: torch.Tensor, rows: PackedImageRows
    ) -> torch.Tensor:
        """The packed ``[tokens, hidden]`` BF16 input embeddings.

        A latent token embeds its source row with its row's time embedding
        and its learned grid position; a marker token is its marker's token
        embedding; a padding token is zero.
        """
        # Tokens other than latent ones read source row 0, whose embedding
        # the selection below discards.
        samples = source.index_select(0, rows.sample_index.clamp_min(0))
        # Latent tokens index the learned grid row-major. Rows and columns
        # must stay below max_latent_size, which this method does not check;
        # a column past it selects another grid row's embedding whenever the
        # flat index stays inside the table. Serving admission (the server's
        # Bagel profile) rejects a canvas side beyond max_latent_size latent
        # patches. Other tokens sit at the grid origin.
        coordinates = (
            rows.positions[1] * self.config.max_latent_size + rows.positions[2]
        )
        features = (
            self.input(samples.to(torch.bfloat16))
            + self.time_embedding(rows.timesteps)
            + self.position(coordinates)
        ).to(torch.bfloat16)

        marker = self.backbone.embed_input_ids(self.markers).to(torch.bfloat16)
        markers = torch.where(
            (rows.kind == TokenKind.OPEN)[:, None], marker[0], marker[1]
        )
        framing = (rows.kind == TokenKind.OPEN) | (rows.kind == TokenKind.CLOSE)
        return torch.where(
            (rows.kind == TokenKind.LATENT)[:, None],
            features,
            torch.where(framing[:, None], markers, features.new_zeros(())),
        )

    def forward(self, inputs: PackedImageInput, *, state, constants, workspace):
        rows = inputs.rows
        width = self.patch_size**2 * self.latent_channels
        if inputs.source.shape[1] != width:
            raise ValueError("BAGEL reads canonical latent patch rows")

        # Without a pipeline axis the local group makes this rank both the
        # first and the last stage. Only the first stage embeds inputs;
        # later stages receive hidden states from their predecessor but
        # still need the routes.
        group = self.mesh.get_group("pp" if "pp" in self.mesh.axes else ())
        hidden = self.backbone(
            self._embed(inputs.source, rows) if group.rank == 0 else None,
            rows.positions,
            inputs.attention,
            routes=self._routes(rows),
        )
        # Only the last stage holds final hidden states.
        if group.rank != group.size - 1:
            return {"image": (None,)}

        # The backbone gathers every token on the last stage, so the output's
        # local slice spans its complete shape.
        prediction = self.prediction(hidden.to(torch.bfloat16))
        shape = (rows.tokens, width)
        return {
            "image": (
                TensorOutput(
                    prediction,
                    OutputLayout(
                        shape,
                        prediction.dtype,
                        tuple(slice(0, n) for n in shape),
                    ),
                ),
            )
        }
