"""An image model's latent codec: latent encodes and image decodes.

The latent encoder (a ``PatchAutoencoder``) turns an image's prepared
pixels into its canonical latent rows; the image decoder turns a
trajectory's latent rows into pixels, which it publishes as an image product
and quantizes into the RGB bytes the worker encodes as a PNG. Their
convolutions run at the image's spatial size, so padding an image would
change its border: each size keeps a graph of its own, captured at startup
for every configured image size, and under graphs only images of those
sizes are served (the engine refuses others).

Each codec copies an image's inputs into fixed buffers, replays its size's
graph and copies the results out of fixed buffers; without graphs it runs
the same computation eagerly on the image's own tensors.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from math import prod
from typing import Any

import torch

from uniserve import cuda_graph
from uniserve.diffusion import normal_noise
from uniserve.executor.text.graphs import GraphTable
from uniserve.executor.text.protocol import GraphMode
from uniserve.media import image
from uniserve.media.pixels import quantize_image_hwc
from uniserve.models import ImageDecoder
from uniserve.nn.rng import posterior_noise_seed
from uniserve.nn.vae import PatchAutoencoder
from uniserve.runner.context import execution_context, kernel_records

__all__ = ["LatentEncoder", "PixelDecoder"]


class _Codec:
    """What both codecs share: a context, a stream and a graph table."""

    def __init__(
        self,
        *,
        module,
        owner: str,
        stream: torch.cuda.Stream | None,
        device: int,
        scratch,
        attention,
        pool: Any | None,
        peers: Sequence[tuple[int, Any]],
    ) -> None:
        self.module = module
        self.owner = owner
        self.stream = stream
        self.pool = pool
        self.peers = tuple(peers)
        self.context = execution_context(
            module,
            0 if stream is None else stream.cuda_stream,
            attention,
            scratch,
            device=int(device),
            derive_host_lengths=True,
        )
        self.context.prepare(None)
        self.device = self.context.device
        self.graphs = GraphTable(owner)

    def _launch(self, key: Hashable, body) -> GraphMode:
        """Replay ``key``'s graph, or run ``body`` eagerly and capture it.

        Without a pool the body runs eagerly. After the codec is sealed, a
        key without a graph is an error.
        """
        if self.pool is None or self.stream is None:
            body()
            return GraphMode.EAGER
        graph = self.graphs.lookup(key)
        if graph is not None:
            with torch.cuda.stream(self.stream):
                graph.replay()
            return GraphMode.REPLAY
        body()
        captured = cuda_graph.capture(
            body, stream=self.stream, pool=self.pool, peers=self.peers
        )
        self.graphs.insert(key, None, captured)
        return GraphMode.CAPTURE

    def seal(self) -> None:
        self.graphs.seal()

    def kernels(self) -> list[str]:
        return kernel_records(self.context.kernels())

    def close(self, aborted: bool) -> None:
        try:
            self.graphs.close(aborted)
        finally:
            self.context.close(aborted=bool(aborted))


class LatentEncoder(_Codec):
    """Latent encodes of prepared images, one graph per captured size.

    Args:
        module: The latent encoder.
        owner: Its component and path, naming it.
        stream: The primary stream.
        device: The CUDA ordinal ``stream`` belongs to, -1 for the host.
        scratch: The stream's shared work areas.
        attention: The attention backend the model loaded with.
        sizes: The ``(height, width)`` prepared pixel sizes captured under
            graphs: those of the configured generated image sizes.
        pixel_dtype: The dtype of the pixels the image policy prepares.
        pool: The primary stream's graph pool, ``None`` without graphs.
        peers: ``(ordinal, pool)`` of each other device it computes on.
    """

    def __init__(
        self,
        *,
        module: PatchAutoencoder,
        owner: str,
        stream: torch.cuda.Stream | None,
        device: int,
        scratch,
        attention,
        sizes: Sequence[tuple[int, int]],
        pixel_dtype: torch.dtype,
        pool: Any | None,
        peers: Sequence[tuple[int, Any]],
    ) -> None:
        super().__init__(
            module=module,
            owner=owner,
            stream=stream,
            device=device,
            scratch=scratch,
            attention=attention,
            pool=pool,
            peers=peers,
        )
        # Largest first: captures in one pool keep addressing its largest
        # activations, and every rank captures in this order.
        self.sizes = sorted(
            {(int(height), int(width)) for height, width in sizes},
            key=lambda size: (-size[0] * size[1], size[0], size[1]),
        )
        self.width = module.patch_size**2 * module.latent_channels
        self.pixel_dtype = pixel_dtype
        self.pixels = self.noise = self.rows = None
        if pool is not None and self.sizes:
            pixels = max(3 * height * width for height, width in self.sizes)
            noise = max(
                prod(module.posterior_shape(height, width))
                for height, width in self.sizes
            )
            units = max(self.latent_rows(*size) for size in self.sizes)
            with torch.cuda.stream(self.stream):
                self.pixels = torch.zeros(
                    pixels, dtype=pixel_dtype, device=self.device
                )
                self.noise = torch.zeros(
                    noise, dtype=torch.float32, device=self.device
                )
                self.rows = torch.zeros(
                    (units, self.width),
                    dtype=module.latent_dtype,
                    device=self.device,
                )

    def latent_rows(self, height: int, width: int) -> int:
        """Latent rows of an image prepared at ``height`` x ``width``."""
        side = self.module.downsample
        return (int(height) // side) * (int(width) // side)

    def _draw(
        self, noise: torch.Tensor, seed: int, draw: int, height: int, width: int
    ) -> torch.Tensor:
        """Draw an image's posterior noise into ``noise``'s leading elements.

        The draw is seeded by the request's ``seed`` and ``draw``, the
        request's latent encodings before this one.
        """
        shape = (1, *self.module.posterior_shape(height, width))
        view = noise.reshape(-1)[: prod(shape)].view(shape)
        normal_noise((posterior_noise_seed(int(seed), int(draw)),), out=(view,))
        return view

    def encode(
        self,
        pixels: torch.Tensor,
        out: torch.Tensor,
        *,
        seed: int,
        draw: int,
    ) -> GraphMode:
        """Encode one prepared ``[3, height, width]`` image into ``out``.

        ``pixels`` are page-locked host storage the worker prepared, or a
        device tensor. A sampled posterior draws its noise at ``seed`` and
        ``draw``.

        Raises:
            ValueError: Under graphs, the image's size was not captured.
        """
        module = self.module
        height, width = (int(extent) for extent in pixels.shape[-2:])
        units = self.latent_rows(height, width)
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            if self.pool is None:
                with self.context.activate():
                    noise = None
                    if module.samples_posterior:
                        noise = self._draw(
                            torch.empty(
                                prod(module.posterior_shape(height, width)),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                            seed,
                            draw,
                            height,
                            width,
                        )
                    value = module.encode(
                        pixels.to(self.device, non_blocking=True).unsqueeze(0),
                        noise=noise,
                    )[0]
                    _write(out, value)
                return GraphMode.EAGER
            if (height, width) not in self.sizes:
                raise ValueError(
                    f"{self.owner}: a {width}x{height} image's latent encode "
                    "was not captured"
                )
            assert self.pixels is not None and self.noise is not None
            count = 3 * height * width
            self.pixels[:count].view(3, height, width).copy_(
                pixels.to(
                    device=self.device,
                    dtype=self.pixels.dtype,
                    non_blocking=True,
                )
            )
            if module.samples_posterior:
                self._draw(self.noise, seed, draw, height, width)
        mode = self._launch(
            ("latent", height, width),
            lambda: self._forward(height, width),
        )
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            assert self.rows is not None
            _write(out, self.rows[:units])
        return mode

    def _forward(self, height: int, width: int) -> None:
        """The encode a size's graph captures, on the fixed buffers."""
        module = self.module
        assert self.pixels is not None and self.noise is not None
        assert self.rows is not None
        units = self.latent_rows(height, width)
        shape = (1, *module.posterior_shape(height, width))
        noise = (
            self.noise[: prod(shape)].view(shape)
            if module.samples_posterior
            else None
        )
        with torch.inference_mode(), self.context.activate():
            pixels = self.pixels[: 3 * height * width].view(1, 3, height, width)
            self.rows[:units].copy_(module.encode(pixels, noise=noise)[0])

    def warm(self, side: int) -> None:
        """Select the encoder's kernels, then capture every size's encode.

        A black ``side`` x ``side`` image encodes eagerly first, with zero
        posterior noise, so the kernels are selected before serving with or
        without graphs; under graphs every configured size's encode is then
        captured, largest first.
        """
        module = self.module
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            with self.context.activate():
                pixels = torch.zeros(
                    (1, 3, side, side),
                    dtype=self.pixel_dtype,
                    device=self.device,
                )
                noise = None
                if module.samples_posterior:
                    noise = torch.zeros(
                        (1, *module.posterior_shape(side, side)),
                        device=self.device,
                    )
                module.encode(pixels, noise=noise)
        if self.pool is None:
            return
        for height, width in self.sizes:
            self._launch(
                ("latent", height, width),
                lambda height=height, width=width: self._forward(height, width),
            )

    def keys(self) -> list[tuple[int, int]]:
        """The graph keys in capture order (empty without graphs)."""
        return [] if self.pool is None else list(self.sizes)


class PixelDecoder(_Codec):
    """Image decodes and quantizations, one graph per captured size.

    Args:
        module: The image decoder.
        owner: Its component and path, naming it.
        stream: The primary stream.
        device: The CUDA ordinal ``stream`` belongs to, -1 for the host.
        scratch: The stream's shared work areas.
        attention: The attention backend the model loaded with.
        sizes: The ``(height, width, latent rows)`` of every image size
            captured under graphs.
        latent_width: Values per latent row.
        latent_dtype: The dtype of the latent rows.
        pool: The primary stream's graph pool, ``None`` without graphs.
        peers: ``(ordinal, pool)`` of each other device it computes on.
    """

    def __init__(
        self,
        *,
        module: ImageDecoder,
        owner: str,
        stream: torch.cuda.Stream | None,
        device: int,
        scratch,
        attention,
        sizes: Sequence[tuple[int, int, int]],
        latent_width: int,
        latent_dtype: torch.dtype,
        pool: Any | None,
        peers: Sequence[tuple[int, Any]],
    ) -> None:
        super().__init__(
            module=module,
            owner=owner,
            stream=stream,
            device=device,
            scratch=scratch,
            attention=attention,
            pool=pool,
            peers=peers,
        )
        low, high = module.decoder.value_range
        self.value_range = (float(low), float(high))
        self.sizes = {
            (int(height), int(width)): int(units)
            for height, width, units in sizes
        }
        self.latent_width = int(latent_width)
        self.latent_dtype = latent_dtype
        self.inputs = self.values = self.staging = None
        if pool is not None and self.sizes:
            units = max(self.sizes.values())
            pixels = max(3 * height * width for height, width in self.sizes)
            # Every captured decode reads one image's latent rows from the
            # inputs and writes its values and quantized bytes; every
            # captured quantization reads the values.
            with torch.cuda.stream(self.stream):
                self.inputs = torch.zeros(
                    (units, latent_width),
                    dtype=latent_dtype,
                    device=self.device,
                )
                self.values = torch.zeros(
                    pixels, dtype=torch.float32, device=self.device
                )
                self.staging = torch.zeros(
                    pixels, dtype=torch.uint8, device=self.device
                )

    def decode(
        self,
        rows: torch.Tensor,
        height: int,
        width: int,
        product: torch.Tensor | None,
        out: torch.Tensor,
    ) -> GraphMode:
        """Decode one image's latent ``rows`` into ``product`` and ``out``.

        ``product`` receives the ``[3, height, width]`` decoded values in its
        dtype (``None`` for a call that publishes none) and ``out``, the
        worker's lent host bytes, the image's quantized HWC RGB bytes.

        Raises:
            ValueError: Under graphs, the image's size was not captured.
        """
        height, width = int(height), int(width)
        count = 3 * height * width
        if self.pool is None:
            with torch.inference_mode(), torch.cuda.stream(self.stream):
                with self.context.activate():
                    value = self.module.decode(
                        (rows.to(self.device, non_blocking=True),),
                        sizes=(image.Config(height, width),),
                    )[0]
                    pixels = quantize_image_hwc(
                        value, value_range=self.value_range
                    )
                if product is not None:
                    _write(product, value)
                out[:count].copy_(pixels.reshape(-1), non_blocking=True)
            return GraphMode.EAGER
        units = self.sizes.get((height, width))
        if units is None:
            raise ValueError(
                f"{self.owner}: a {width}x{height} image's decode was not "
                "captured"
            )
        assert self.inputs is not None and self.values is not None
        assert self.staging is not None
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            self.inputs[:units].copy_(
                rows[:units].to(
                    device=self.device,
                    dtype=self.inputs.dtype,
                    non_blocking=True,
                )
            )
        mode = self._launch(
            ("decode", height, width),
            lambda: self._decode(units, height, width),
        )
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            if product is not None:
                _write(product, self.values[:count].view(3, height, width))
            out[:count].copy_(self.staging[:count], non_blocking=True)
        return mode

    def _decode(self, units: int, height: int, width: int) -> None:
        """The decode a size's graph captures, on the fixed buffers."""
        assert self.inputs is not None and self.values is not None
        assert self.staging is not None
        count = 3 * height * width
        with torch.inference_mode(), self.context.activate():
            value = self.module.decode(
                (self.inputs[:units],), sizes=(image.Config(height, width),)
            )[0]
            self.values[:count].view(3, height, width).copy_(value)
            self.staging[:count].copy_(
                quantize_image_hwc(value, value_range=self.value_range).reshape(
                    -1
                )
            )

    def quantize(
        self,
        pixels: torch.Tensor,
        value_range: tuple[float, float],
        out: torch.Tensor,
    ) -> GraphMode:
        """Quantize a resident ``[3, height, width]`` image into ``out``.

        Its values lie in ``value_range``. Under graphs a resident image of
        a captured size and of the decoder's range replays its size's
        quantization.

        Raises:
            ValueError: Under graphs, no quantization of the image's size
                and range was captured.
        """
        if pixels.ndim == 4:
            pixels = pixels[0]
        height, width = (int(extent) for extent in pixels.shape[-2:])
        count = 3 * height * width
        value_range = (float(value_range[0]), float(value_range[1]))
        if self.pool is None:
            with torch.inference_mode(), torch.cuda.stream(self.stream):
                quantized = quantize_image_hwc(
                    pixels.to(self.device, non_blocking=True),
                    value_range=value_range,
                )
                out[:count].copy_(quantized.reshape(-1), non_blocking=True)
            return GraphMode.EAGER
        if (height, width) not in self.sizes or value_range != self.value_range:
            raise ValueError(
                f"{self.owner}: a {width}x{height} image of range "
                f"{value_range} has no captured quantization"
            )
        assert self.values is not None and self.staging is not None
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            self.values[:count].view(3, height, width).copy_(
                pixels.to(self.device, non_blocking=True)
            )
        mode = self._launch(
            ("quantize", height, width),
            lambda: self._quantize(height, width),
        )
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            out[:count].copy_(self.staging[:count], non_blocking=True)
        return mode

    def _quantize(self, height: int, width: int) -> None:
        """The quantization a size's graph captures."""
        assert self.values is not None and self.staging is not None
        count = 3 * height * width
        with torch.inference_mode(), self.context.activate():
            self.staging[:count].copy_(
                quantize_image_hwc(
                    self.values[:count].view(3, height, width),
                    value_range=self.value_range,
                ).reshape(-1)
            )

    def warm(self, side: int, units: int) -> None:
        """Select the decoder's kernels, then capture every size's decode.

        A zero latent of a ``side`` x ``side`` image (``units`` latent rows)
        decodes eagerly first, so the kernels are selected before serving
        with or without graphs; under graphs every configured size's decode
        and quantization is then captured, largest first.
        """
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            with self.context.activate():
                rows = torch.zeros(
                    (units, self.latent_width),
                    dtype=self.latent_dtype,
                    device=self.device,
                )
                (value,) = self.module.decode(
                    (rows,), sizes=(image.Config(side, side),)
                )
                quantize_image_hwc(value, value_range=self.value_range)
        if self.pool is None:
            return
        for height, width in self.keys():
            units = self.sizes[(height, width)]
            self._launch(
                ("decode", height, width),
                lambda units=units, height=height, width=width: self._decode(
                    units, height, width
                ),
            )
            self._launch(
                ("quantize", height, width),
                lambda height=height, width=width: self._quantize(
                    height, width
                ),
            )

    def keys(self) -> list[tuple[int, int]]:
        """The captured sizes in capture order (empty without graphs)."""
        if self.pool is None:
            return []
        return sorted(
            self.sizes,
            key=lambda size: (-size[0] * size[1], size[0], size[1]),
        )


def _write(target: torch.Tensor, value: torch.Tensor) -> None:
    """Copy ``value`` into ``target``'s leading elements in its dtype."""
    if value.numel() > target.numel():
        raise ValueError("a result exceeds its product")
    target.reshape(-1)[: value.numel()].view(value.shape).copy_(
        value.to(device=target.device, dtype=target.dtype, non_blocking=True)
    )
