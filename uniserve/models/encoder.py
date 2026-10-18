"""Encoders with homogeneous feature batching and numerical boundaries.

The text, patch and condition encoders, and the windowed video and audio
encoders.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import prod
from typing import Generic, TypeVar

import torch
from torch import nn

from uniserve.media import image, video
from uniserve.nn.attention import (
    AttentionBatch,
    SequenceLengths,
    VarlenInput,
)
from uniserve.nn.vae import LatentEncoder
from uniserve.tensors import OutputLayout, TensorOutput

from .inputs import EmbeddingReplacement, VisionInput
from .transformer import TransformerDecoder

InputT = TypeVar("InputT")


class Encoder(nn.Module, Generic[InputT]):
    """Stack equal-shaped feature samples and restore their original order.

    The network preserves its leading sample dimension. It must apply token
    calls independently within each sample, as ordinary batched modules do.
    """

    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def encode(self, inputs: InputT) -> tuple[torch.Tensor, ...] | None:
        """Encode each sample, or return None on a non-final pipeline stage.

        This default batches indexable tensor samples; encoders of structured
        inputs override it.
        """
        if not isinstance(inputs, (Sequence, torch.Tensor)):
            raise TypeError("the default encoder batches indexable samples")
        groups: defaultdict[
            tuple[torch.Size, torch.dtype, torch.device], list[int]
        ] = defaultdict(list)
        for index, value in enumerate(inputs):
            groups[(value.shape, value.dtype, value.device)].append(index)

        result: dict[int, torch.Tensor] = {}
        for indices in groups.values():
            values = self.network(
                torch.stack(tuple(inputs[index] for index in indices))
            )
            if values.shape[0] != len(indices):
                raise ValueError(
                    "encoder network must preserve the sample dimension"
                )
            for index, value in zip(indices, values.unbind(), strict=True):
                result[index] = value
        return tuple(result[index] for index in range(len(inputs)))


class TextConditioner(Encoder[tuple[torch.Tensor, ...]]):
    """Refine the feature rows of text samples, padded or exact.

    ``encode`` takes ``[rows, width]`` samples. Without ``lengths`` every row
    is text, and samples of equal shape are refined together. With
    ``lengths``, an int32 device tensor holding each sample's text row count,
    every sample has the same shape and holds its text in its leading rows.
    Rows past a sample's length are padding: they may hold any finite
    values, and the network keeps every text row's output independent of
    them, as masking them out of attention does. Padding every text length
    to a shared row count lets one prepared or captured call serve all of
    them; the outputs of padding rows are unspecified. The network receives
    stacked samples ``[samples, rows, width]`` and their lengths or None.
    """

    def encode(
        self,
        inputs: tuple[torch.Tensor, ...],
        *,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if any(value.ndim != 2 for value in inputs):
            raise ValueError("text conditioning samples are [rows, width]")
        if lengths is None:
            groups: defaultdict[torch.Size, list[int]] = defaultdict(list)
            for index, value in enumerate(inputs):
                groups[value.shape].append(index)
        else:
            if (
                lengths.shape != (len(inputs),)
                or lengths.dtype != torch.int32
                or any(value.shape != inputs[0].shape for value in inputs)
            ):
                raise ValueError(
                    "padded text conditioning requires samples of one shape "
                    "and one int32 length per sample"
                )
            groups = defaultdict(
                list, {inputs[0].shape: list(range(len(inputs)))}
            )

        result: dict[int, torch.Tensor] = {}
        for indices in groups.values():
            values = self.network(
                torch.stack(tuple(inputs[index] for index in indices)),
                lengths,
            )
            if values.shape[0] != len(indices):
                raise ValueError(
                    "encoder network must preserve the sample dimension"
                )
            for index, value in zip(indices, values.unbind(), strict=True):
                result[index] = value
        return tuple(result[index] for index in range(len(inputs)))


class PatchEncoder(Encoder[VisionInput]):
    """Pack image samples and split their spatially downsampled features.

    downsample counts input patches per output feature on each spatial axis.
    Patch serialization for an architecture is defined by its network;
    complete CHW images are stacked without altering their pixel order.
    A packed sample with a ``(time, height, width)`` grid keeps its time
    axis: each output feature merges patches of one time step, so the sample
    yields ``time * height * width / downsample**2`` features.

    With ``max_patches``, the most patch rows one image can have, and
    ``max_slots``, the most slots one packed call holds, the encoder also
    encodes images packed into fixed slots of ``max_patches`` rows
    (``encode_packed``). The network must then read image grids only as
    device values, accept ``grid_shapes=None``, and keep every image's
    output independent of the other segments its grids describe, as
    attention restricted to each segment and pooling within it do. One
    prepared or captured call then serves every packing of a slot count.
    ``max_slots`` bounds the activations a packed graph retains; a larger
    batch runs in groups of that many.
    """

    def __init__(
        self,
        network: nn.Module,
        connector: nn.Module,
        *,
        patch_size: int,
        downsample: int,
        output_size: int,
        output_dtype: torch.dtype,
        max_patches: int | None = None,
        max_slots: int | None = None,
    ):
        super().__init__(network)
        if any(
            type(value) is not int or value < 1
            for value in (patch_size, downsample, output_size)
        ):
            raise ValueError(
                "patch encoding requires positive spatial and feature "
                "dimensions"
            )
        if max_patches is not None and (
            type(max_patches) is not int
            or max_patches < 1
            or max_patches % downsample**2
        ):
            raise ValueError(
                "packed image slots require a positive patch capacity that "
                "whole output features divide"
            )
        if (max_slots is None) != (max_patches is None) or (
            max_slots is not None
            and (type(max_slots) is not int or max_slots < 1)
        ):
            raise ValueError(
                "packed image slots require both a patch capacity and a "
                "positive slot count per call"
            )
        self.connector, self.patch_size, self.downsample = (
            connector,
            patch_size,
            downsample,
        )
        self._output_size, self._output_dtype = output_size, output_dtype
        self.max_patches = max_patches
        self.max_slots = max_slots

    def packed_grids(
        self, shapes: Sequence[tuple[int, int]], slots: int
    ) -> tuple[tuple[int, int], ...]:
        """Lay out ``shapes`` in ``slots`` fixed slots for ``encode_packed``.

        Each slot holds ``max_patches`` rows: one image's patch rows first,
        then padding. It is described by two ``(rows, columns)`` grids: the
        image's patch grid, ``(0, 0)`` for an empty slot, followed by a
        ``downsample`` rows high grid covering the slot's remaining rows, or
        ``(0, 0)`` when the image fills it. Images take the leading slots in
        order.

        Raises:
            ValueError: When there are more images than slots, an image has
                more than ``max_patches`` patches, or a side is not a
                positive multiple of ``downsample``.
        """
        capacity = self.max_patches
        if capacity is None:
            raise ValueError("this patch encoder packs no image slots")
        if len(shapes) > slots:
            raise ValueError("packed images exceed their slots")

        grids: list[tuple[int, int]] = []
        for index in range(slots):
            shape = shapes[index] if index < len(shapes) else (0, 0)
            rows = shape[0] * shape[1]
            if index < len(shapes) and (
                min(shape) < 1
                or any(side % self.downsample for side in shape)
                or rows > capacity
            ):
                raise ValueError(
                    "packed image grids must align with spatial "
                    "downsampling within the slot capacity"
                )
            padding = capacity - rows
            grids.append(shape)
            grids.append(
                (self.downsample, padding // self.downsample)
                if padding
                else (0, 0)
            )
        return tuple(grids)

    def encode_packed(
        self, pixels: torch.Tensor, grids: torch.Tensor
    ) -> torch.Tensor:
        """Encode patch rows packed in fixed slots (see ``packed_grids``).

        ``pixels`` is ``[slots * max_patches, row]`` patch rows and
        ``grids`` the ``[2 * slots, 2]`` int64 device grids of their
        images and padding. Padding rows may hold any finite values.

        Returns:
            ``[slots * max_patches // downsample**2, output_size]`` features
            in row order: each slot's image features first, then features of
            its padding, which are unspecified.

        Raises:
            ValueError: When the encoder packs no slots or the tensors do not
                span whole slots.
        """
        capacity = self.max_patches
        if capacity is None:
            raise ValueError("this patch encoder packs no image slots")
        slots = grids.shape[0] // 2
        if (
            pixels.ndim != 2
            or pixels.shape[0] != slots * capacity
            or grids.shape != (2 * slots, 2)
        ):
            raise ValueError("packed patch rows must span whole image slots")
        features = self.network(pixels, grids, None)
        return self.connector(features).to(self._output_dtype)

    def encode(self, inputs: VisionInput) -> tuple[torch.Tensor, ...]:
        groups = defaultdict(list)
        for index, value in enumerate(inputs.images):
            # Packed samples share one network call when their patch width
            # and grid rank agree, so their grids concatenate.
            key = (
                value.ndim,
                value.shape[1:] if value.ndim == 2 else value.shape,
                len(inputs.grid_shapes[index] or ()),
                value.dtype,
                value.device,
            )
            groups[key].append(index)

        result: dict[int, torch.Tensor] = {}
        for indices in groups.values():
            pixels, grids, shapes = [], [], []
            for index in indices:
                value, grid, shape = (
                    inputs.images[index],
                    inputs.grids[index],
                    inputs.grid_shapes[index],
                )
                if value.ndim == 3:
                    if any(axis % self.patch_size for axis in value.shape[-2:]):
                        raise ValueError(
                            "image pixels must contain complete input patches"
                        )
                    shape = (
                        value.shape[-2] // self.patch_size,
                        value.shape[-1] // self.patch_size,
                    )
                    grid = value.new_empty((1, 2), dtype=torch.int64)
                    grid[:, 0].fill_(shape[0])
                    grid[:, 1].fill_(shape[1])
                elif value.ndim != 2 or grid is None or shape is None:
                    raise ValueError(
                        "packed vision samples require their grid and host "
                        "shape"
                    )
                elif value.shape[0] != prod(shape):
                    raise ValueError(
                        "vision grid dimensions must cover their patch rows"
                    )
                if (
                    len(shape) not in (2, 3)
                    or grid.shape != (1, len(shape))
                    or min(shape) < 1
                    or any(axis % self.downsample for axis in shape[-2:])
                ):
                    raise ValueError(
                        "vision grids must align with spatial downsampling"
                    )
                pixels.append(value)
                grids.append(grid)
                shapes.append(shape)

            values = (
                torch.stack(pixels)
                if pixels[0].ndim == 3
                else torch.cat(pixels)
            )
            features = self.network(values, torch.cat(grids), tuple(shapes))
            features = self.connector(features).to(self._output_dtype)

            counts = tuple(
                prod(shape) // self.downsample**2 for shape in shapes
            )
            if features.shape != (sum(counts), self._output_size):
                raise ValueError(
                    "vision features must cover the declared spatial output "
                    "grids"
                )
            for index, features in zip(
                indices, features.split(counts), strict=True
            ):
                result[index] = features
        return tuple(result[index] for index in range(inputs.batch_size))

    def pack_pixels(
        self, frames: torch.Tensor, grid: tuple[int, int, int]
    ) -> torch.Tensor:
        """Return one image's or video's packed patch rows for ``encode``.

        ``frames`` holds ``[frames, height, width, 3]`` uint8 RGB on the
        host, an image being one frame; ``grid`` is the ``(time, height,
        width)`` patch grid they are resized to. The result is the sample
        ``encode`` takes with ``grid`` as its host shape. An encoder whose
        network reads packed patch rows defines this conversion; it reads no
        parameter, so a host rank holding the module's description runs it.
        """
        raise NotImplementedError

    def pixels_layout(self, num_tokens: int) -> OutputLayout:
        """Describe the packed patch rows of ``num_tokens`` merged tokens.

        These are the rows ``pack_pixels`` produces, ``downsample**2`` per
        merged token. An encoder that defines ``pack_pixels`` defines it.
        """
        raise NotImplementedError

    def features_layout(self, num_tokens: int) -> Mapping[str, OutputLayout]:
        """Describe ``encode``'s features of ``num_tokens`` merged tokens.

        The packed samples' rows are described under ``features``.
        """
        shape = (num_tokens, self._output_size)
        return {
            "features": OutputLayout(
                shape,
                self._output_dtype,
                tuple(slice(0, n) for n in shape),
                variable_axes=(0,),
            )
        }

    def output_layout(self, size: image.Config):
        stride = self.patch_size * self.downsample
        shape = (
            size.height // stride * (size.width // stride),
            self._output_size,
        )
        return {
            "features": OutputLayout(
                shape,
                self._output_dtype,
                tuple(slice(0, n) for n in shape),
                variable_axes=(0,),
            )
        }


class TextEncoder(Encoder[tuple[torch.Tensor, ...]]):
    """Encode independent token sequences through the retained decoder layers.

    retained_layers names the checkpoint layers participating in the numerical
    encoder, in execution order. Output normalization belongs to the supplied
    network; an Identity norm exposes its unnormalized final residual stream.
    """

    network: TransformerDecoder

    def __init__(
        self, network: TransformerDecoder, retained_layers: tuple[int, ...]
    ):
        super().__init__(network)
        if (
            not retained_layers
            or tuple(sorted(set(retained_layers))) != retained_layers
        ):
            raise ValueError(
                "retained decoder layers must be a nonempty increasing tuple"
            )
        if any(str(index) not in network.layers for index in retained_layers):
            raise ValueError(
                "retained decoder layers must exist in the supplied network"
            )
        self.retained_layers = retained_layers
        network.layers = nn.ModuleDict(
            {
                str(index): network.layers[str(index)]
                for index in retained_layers
            }
        )

    def encode(
        self,
        tokens: tuple[torch.Tensor, ...],
        *,
        positions: tuple[torch.Tensor, ...] | None = None,
        embeddings: tuple[EmbeddingReplacement | None, ...] | None = None,
        deepstack: tuple[tuple[torch.Tensor, ...] | None, ...] | None = None,
    ) -> tuple[torch.Tensor, ...] | None:
        """Return complete sequence features on the final pipeline stage.

        All participating stages execute the retained network. Earlier stages
        return None after forwarding their activations; sequence partitions
        are gathered by the final decoder stage before restoring sample bounds.

        The optional inputs hold one entry per sample of ``tokens``:

        - ``positions``: rotary coordinates, ``[length]`` or, for
          multimodal rotary positions, ``[axes, length]`` with the same axes
          in every sample. Without them a sample's tokens take positions
          ``0..length-1``.
        - ``embeddings``: an ``EmbeddingReplacement`` whose dense
          ``[length, hidden]`` values replace the token embeddings where its
          ``[length]`` mask is set, or None to keep every token embedding.
        - ``deepstack``: dense ``[length, hidden]`` features added to the
          residual stream after the leading retained layers, the ``j``-th
          after retained layer ``j`` (DeepStack), or None. Rows without
          features hold zeros. Every sample that has features has the same
          number of them, at most one per retained layer.

        Raises:
            ValueError: An optional input does not match the samples.
        """
        if not tokens:
            return (
                ()
                if self.network._pipeline.rank
                == self.network._pipeline.size - 1
                else None
            )
        if any(value.ndim != 1 for value in tokens):
            raise ValueError(
                "text encoding requires one-dimensional token sequences"
            )
        counts = tuple(value.numel() for value in tokens)
        packed = torch.cat(tokens)
        if positions is None:
            packed_positions = torch.cat(
                tuple(
                    torch.arange(count, device=packed.device)
                    for count in counts
                )
            )
        else:
            if (
                len(positions) != len(counts)
                or any(
                    value.shape[-1] != count
                    for value, count in zip(positions, counts, strict=True)
                )
                or len({value.shape[:-1] for value in positions}) != 1
            ):
                raise ValueError(
                    "positions must cover every sample's tokens with one "
                    "common axis layout"
                )
            packed_positions = torch.cat(positions, dim=-1)
        additions = self._deepstack(counts, deepstack)
        values = torch.cat(
            tuple(
                packed.new_full((1,), count, dtype=torch.int32)
                for count in counts
            )
        )
        offsets = torch.cat(
            (values.new_zeros(1), values.cumsum(0, dtype=torch.int32))
        )
        lengths = SequenceLengths(host=counts, values=values, offsets=offsets)
        attention = AttentionBatch.single(
            VarlenInput(lengths, lengths, (True,) * len(counts))
        )

        embedded = None
        if self.network._pipeline.rank == 0:
            embedded = self._embed(packed, counts, embeddings)
        features = self.network(
            embedded, packed_positions, attention, deepstack=additions
        )
        if self.network._pipeline.rank != self.network._pipeline.size - 1:
            return None
        return tuple(features.split(counts))

    def _embed(
        self,
        packed: torch.Tensor,
        counts: tuple[int, ...],
        embeddings: tuple[EmbeddingReplacement | None, ...] | None,
    ) -> torch.Tensor:
        """Embed packed tokens, splicing each sample's replaced rows."""
        embedded = self.network.embed_input_ids(packed)
        if embeddings is None:
            return embedded
        if len(embeddings) != len(counts):
            raise ValueError("embedding replacements must align with samples")

        parts = list(embedded.split(counts))
        for index, replacement in enumerate(embeddings):
            if replacement is None:
                continue
            if replacement.values.shape != parts[index].shape or (
                replacement.mask.shape != (counts[index],)
            ):
                raise ValueError(
                    "embedding replacements must cover their sample's rows"
                )
            parts[index] = torch.where(
                replacement.mask.reshape(-1, 1),
                replacement.values.to(embedded.dtype),
                parts[index],
            )
        return torch.cat(parts)

    def _deepstack(
        self,
        counts: tuple[int, ...],
        deepstack: tuple[tuple[torch.Tensor, ...] | None, ...] | None,
    ) -> dict[str, torch.Tensor] | None:
        """Pack per-sample DeepStack features by the layer receiving them.

        Returns packed ``[tokens, hidden]`` additions keyed by the retained
        layer's network key, or None when no sample has features.
        """
        if deepstack is None:
            return None
        if len(deepstack) != len(counts):
            raise ValueError("DeepStack features must align with samples")
        provided = [values for values in deepstack if values]
        if not provided:
            return None
        depth, reference = len(provided[0]), provided[0][0]
        if (
            depth > len(self.retained_layers)
            or any(
                len(values) != depth
                for values in deepstack
                if values is not None
            )
            or any(
                feature.shape != (count, self.network.hidden_size)
                for values, count in zip(deepstack, counts, strict=True)
                if values is not None
                for feature in values
            )
        ):
            raise ValueError(
                "DeepStack features must give every sample the same leading "
                "retained layers and cover its rows"
            )

        additions = {}
        for layer in range(depth):
            additions[str(self.retained_layers[layer])] = torch.cat(
                tuple(
                    reference.new_zeros((count, self.network.hidden_size))
                    if values is None
                    else values[layer].to(reference.dtype)
                    for values, count in zip(deepstack, counts, strict=True)
                )
            )
        return additions

    def positions(
        self,
        token_ids: Sequence[int],
        *,
        image_grids: Sequence[tuple[int, int, int]] = (),
        video_grids: Sequence[tuple[int, int, int]] = (),
    ) -> torch.Tensor:
        """Return the rotary coordinates of a prompt holding vision blocks.

        ``image_grids`` and ``video_grids`` are the patch grids of the
        prompt's image and video blocks in prompt order. The result is the
        CPU int64 ``positions`` sample ``encode`` takes for the prompt. An
        encoder that splices vision tokens into prompts defines it.
        """
        raise NotImplementedError

    def output_layout(self, num_tokens: int, dtype: torch.dtype):
        """Describe the encoded conditioning this encoder emits.

        ``dtype`` is the numerical dtype the deployment runs the encoder at,
        which is what its hidden states carry. The caller supplies it because
        every rank of a deployment declares this product, including the ranks
        that never materialize the encoder's parameters and could read no
        representation from them.
        """
        if type(num_tokens) is not int or num_tokens < 0:
            raise ValueError("text output length must be a nonnegative integer")
        if not dtype.is_floating_point:
            raise ValueError("encoded conditioning requires a float dtype")
        shape = (num_tokens, self.network.hidden_size)
        return {
            "conditioning": OutputLayout(
                shape,
                dtype,
                tuple(slice(0, n) for n in shape),
                variable_axes=(0,),
            )
        }


class VideoEncoder(nn.Module):
    """Encode temporal units of RGB24 frames into latent rows.

    A unit is a contiguous run of a video's frames whose latents depend on no
    frame outside it. ``frame_slices`` lists a video's units in order and
    ``latent_slices`` the contiguous latent frames each one produces, so the
    units of one video may be encoded by different ranks, in any order, and
    their rows assemble the whole-video encoding exactly.

    Subclasses define both partitions; ``output_layout``, which describes a
    video's complete latent as frame-major rows along its leading axis, each
    latent frame owning the same number of consecutive rows;
    ``unpack_pixels``, the conversion of one unit's frames into the native
    encoder input; and ``pack_latents``, the conversion of that unit's native
    NCTHW latents into its output rows. An encoder whose posterior is sampled
    defines ``posterior_shape`` and ``draw_noise``: the draw spans the
    video's complete latent and depends on the video's size alone, so a
    caller draws it once per size and passes it with every unit of that
    size; each unit takes the share of its own latent frames, which
    reproduces the draw of a whole-video encoding. Inputs remain borrowed;
    results are new tensors.
    """

    def __init__(self, encoder: LatentEncoder):
        super().__init__()
        self.encoder = encoder

    def frame_slices(self, num_frames: int) -> tuple[slice, ...]:
        """Partition a video of ``num_frames`` frames into its units."""
        raise NotImplementedError

    def latent_slices(self, num_frames: int) -> tuple[slice, ...]:
        """Return the latent frames each unit produces, in unit order."""
        raise NotImplementedError

    def output_layout(self, size: video.Config) -> Mapping[str, OutputLayout]:
        """Describe the complete latent rows of one video under ``video``."""
        raise NotImplementedError

    def load_kernels(self) -> None:
        """Build or load the native kernels ``encode`` launches.

        The kernels compile for the current CUDA device on first use; the
        rank calls this once before serving so no encoding compiles one. An
        encoder whose forward launches no natively built kernel does
        nothing.
        """

    def posterior_shape(self, size: video.Config) -> tuple[int, ...] | None:
        """Return the NCTHW shape of one video's complete posterior draw.

        None when ``encode`` takes no draw, leaving the latent to the latent
        encoder's own posterior: its mean, or no posterior at all.
        """
        return None

    def draw_noise(self, size: video.Config, out: torch.Tensor) -> None:
        """Fill ``out`` with one video's complete posterior draw.

        ``out`` is FP32 of ``posterior_shape(size)``, on any device; the draw
        has the same values wherever it is stored. Defined by an encoder
        whose ``posterior_shape`` is not None.
        """
        raise ValueError(f"{type(self).__qualname__} takes no posterior draw")

    def unpack_pixels(
        self, pixels: torch.Tensor, frames: slice, num_frames: int
    ) -> torch.Tensor:
        """Return the native encoder input of one unit's frames.

        ``pixels`` holds exactly the frames ``frames`` selects, as
        ``[frames, height, width, 3]`` uint8 RGB.
        """
        raise NotImplementedError

    def pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Return the output rows of one unit's native NCTHW latents."""
        raise NotImplementedError

    @torch.inference_mode()
    def encode(
        self,
        pixels: tuple[torch.Tensor, ...],
        *,
        frames: tuple[slice, ...],
        num_frames: tuple[int, ...],
        noise: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[TensorOutput, ...]:
        """Encode units of RGB24 videos into their latent rows.

        ``pixels[i]`` holds exactly the frames that ``frames[i]`` selects from
        a video of ``num_frames[i]`` frames, as ``[frames, height, width, 3]``
        uint8, and ``frames[i]`` must be one of that video's units. An
        encoder that samples its posterior takes ``noise[i]``, the complete
        draw of unit ``i``'s video (``draw_noise``), which units of one video
        share; other encoders take no ``noise``. Each result holds its unit's
        rows at their place in the video's complete latent.
        """
        if (
            not pixels
            or len(pixels) != len(frames)
            or len(pixels) != len(num_frames)
            or (noise is not None and len(noise) != len(pixels))
        ):
            raise ValueError(
                "video frames, unit slices, durations and draws must align"
            )

        units = []
        for value, interval, count in zip(
            pixels, frames, num_frames, strict=True
        ):
            legal = self.frame_slices(count)
            if interval not in legal:
                raise ValueError(
                    "video frame slice must select one complete encoding unit"
                )
            if (
                value.ndim != 4
                or value.dtype != torch.uint8
                or value.shape[-1] != 3
                or value.shape[0] != interval.stop - interval.start
            ):
                raise ValueError(
                    "video units must be uint8 RGB frames "
                    "[frames, height, width, 3] covering their frame slice"
                )
            units.append(legal.index(interval))

        outputs = []
        for index, (value, interval, count, unit) in enumerate(
            zip(pixels, frames, num_frames, units, strict=True)
        ):
            size = video.Config(
                count, image.Config(int(value.shape[1]), int(value.shape[2]))
            )
            layout = self.output_layout(size)["video"]
            windows = self.latent_slices(count)
            window = windows[unit]
            extent = window.stop - window.start
            if layout.shape[0] % windows[-1].stop:
                raise ValueError(
                    "video latent rows must divide evenly among latent frames"
                )
            rows_per_frame = layout.shape[0] // windows[-1].stop

            shape = self.posterior_shape(size)
            if (shape is None) != (noise is None):
                raise ValueError(
                    "a sampled posterior takes its video's draw, and only it"
                )
            draw = None
            if noise is not None:
                if tuple(noise[index].shape) != shape:
                    raise ValueError(
                        "a posterior draw must span its video's latent"
                    )
                # The unit's latent frames of its video's draw. A caller's
                # draw stays unchanged while the copy reads it.
                draw = noise[index][:, :, window].to(
                    value.device, non_blocking=True
                )

            # A unit whose input was padded to the encoder's temporal extent
            # yields trailing latent frames beyond its own; the window keeps
            # the leading ones, so the posterior never samples the others.
            latents = self.encoder(
                self.unpack_pixels(value, interval, count),
                window=(slice(None), slice(None), slice(0, extent)),
                noise=draw,
            )
            rows = self.pack_latents(latents)
            if (
                rows.shape != (extent * rows_per_frame, *layout.shape[1:])
                or rows.dtype != layout.dtype
            ):
                raise ValueError(
                    "packed video latents must match the declared output rows"
                )
            outputs.append(
                TensorOutput(
                    rows,
                    OutputLayout(
                        layout.shape,
                        layout.dtype,
                        (
                            slice(
                                window.start * rows_per_frame,
                                window.stop * rows_per_frame,
                            ),
                            *layout.local_slice[1:],
                        ),
                        variable_axes=layout.variable_axes,
                        value_range=layout.value_range,
                    ),
                )
            )
        return tuple(outputs)


class AudioEncoder(nn.Module):
    """Encode sample-major PCM tracks into latent rows.

    Tracks are encoded whole rather than by media unit. Subclasses define
    ``latent_frames`` and ``latent_rate`` from their codec's compression;
    ``output_layout``; ``unpack_samples``, the conversion of one track's
    ``[samples, channels]`` PCM into the native encoder input; and
    ``pack_latents``, the conversion of the track's native latents into its
    output rows. Inputs carry ``sample_rate`` samples per second and remain
    borrowed; results are new tensors.
    """

    def __init__(self, encoder: LatentEncoder, *, sample_rate: int):
        super().__init__()
        if type(sample_rate) is not int or sample_rate < 1:
            raise ValueError("audio sample rate must be a positive integer")
        self.encoder, self.sample_rate = encoder, sample_rate

    def latent_frames(self, num_samples: int) -> int:
        """Return the latent frames that encode ``num_samples`` samples."""
        raise NotImplementedError

    @property
    def latent_rate(self) -> int:
        """Input samples one latent frame encodes."""
        raise NotImplementedError

    def output_layout(self, num_samples: int) -> Mapping[str, OutputLayout]:
        """Describe the latent rows of one track under ``audio``."""
        raise NotImplementedError

    def unpack_samples(self, samples: torch.Tensor) -> torch.Tensor:
        """Return the native encoder input of one PCM track."""
        raise NotImplementedError

    def pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Return the output rows of one track's native latents."""
        raise NotImplementedError

    @torch.inference_mode()
    def encode(
        self, samples: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        """Encode ``[samples, channels]`` floating-point PCM tracks.

        Values lie in ``[-1, 1]``. Each result holds one track's complete
        latent rows as ``output_layout`` describes them.
        """
        if not samples:
            raise ValueError("audio encoding requires at least one track")
        outputs = []
        for track in samples:
            if (
                track.ndim != 2
                or track.shape[0] < 1
                or track.shape[1] < 1
                or not track.is_floating_point()
            ):
                raise ValueError(
                    "audio tracks must be nonempty floating-point PCM "
                    "[samples, channels]"
                )
            layout = self.output_layout(int(track.shape[0]))["audio"]
            rows = self.pack_latents(self.encoder(self.unpack_samples(track)))
            if rows.shape != layout.shape or rows.dtype != layout.dtype:
                raise ValueError(
                    "packed audio latents must match the declared output rows"
                )
            outputs.append(rows)
        return tuple(outputs)
