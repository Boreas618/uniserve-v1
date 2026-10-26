"""Vision encodes of an image model's patch tower on patch-row buckets.

A vision encoding turns each input image (or a generated image fed back as
context) into the features a prompt places in its vision block. A tower
that packs images (``PatchEncoder.max_patches``) encodes every image of a
forward in one call: their patch rows back to back, then one padding
segment, described by device segment grids (``PatchEncoder.
encode_segments``), so the call's key is a bucket ``(P, S)`` of patch rows
and image slots (:mod:`.shapes`) and one graph per bucket serves every
packing that fits it. Images beyond the largest bucket split into
consecutive chunks; each image's features are independent of the others',
so a chunk is a forward of its own.

A tower that does not pack encodes images of one pixel shape per call, in
first-appearance order (:func:`~.plan.shape_groups`), eagerly; it is not
served under CUDA graphs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from uniserve import cuda_graph
from uniserve.errors import SetupError
from uniserve.executor.text.graphs import GraphTable
from uniserve.executor.text.protocol import GraphMode
from uniserve.models import PatchEncoder, TextSize, VisionInput
from uniserve.runner.context import execution_context, kernel_records

from .plan import chunk_rows, segment_grids, shape_groups
from .shapes import (
    VisionLimits,
    VisionShape,
    VisionShapes,
    patch_sizes,
    slot_sizes,
    vision_shapes,
)

__all__ = ["VisionEncoder", "VisionImage", "features", "vision_limits"]


@dataclass(frozen=True, slots=True)
class VisionImage:
    """One image a vision encoding encodes.

    Attributes:
        pixels: Its prepared ``[patches, row]`` patch rows: page-locked
            host memory the worker prepared, or a device tensor.
        grid: Its ``(rows, columns)`` patch grid.
        out: The ``[features, output_size]`` product its features fill.
    """

    pixels: torch.Tensor
    grid: tuple[int, int]
    out: torch.Tensor

    @property
    def patches(self) -> int:
        return self.grid[0] * self.grid[1]


def features(module: PatchEncoder) -> tuple[int, torch.dtype]:
    """The ``(width, dtype)`` of a patch tower's output features."""
    layout = module.features_layout(1)["features"]
    return int(layout.shape[1]), layout.dtype


def vision_limits(module: PatchEncoder) -> VisionLimits | None:
    """The encodes a packing patch tower admits; ``None`` if it packs none."""
    if module.max_patches is None or module.max_slots is None:
        return None
    return VisionLimits(
        downsample=int(module.downsample),
        max_patches=int(module.max_patches),
        max_slots=int(module.max_slots),
    )


class VisionEncoder:
    """A patch tower's encodes on one stream.

    Args:
        module: The patch tower.
        owner: Its component and path, naming it.
        stream: The lane stream.
        device: The CUDA ordinal ``stream`` belongs to, -1 for the host.
        scratch: The stream's shared work areas.
        attention: The attention backend the model loaded with.
        graphs: Whether encodes replay captured graphs.
        pool: The lane stream's graph pool, ``None`` without graphs.
        peers: ``(ordinal, pool)`` of each other device the tower computes
            on.
        pixel_dtype: The dtype of the pixels the image policy prepares.
        row_budget: Patch rows one call of a tower that does not pack holds.
        max_images: Images one call of such a tower holds.

    Raises:
        SetupError: A tower that does not pack serves under graphs.
    """

    def __init__(
        self,
        *,
        module: PatchEncoder,
        owner: str,
        stream: torch.cuda.Stream | None,
        device: int,
        scratch,
        attention,
        graphs: bool,
        pool: Any | None,
        peers: Sequence[tuple[int, Any]],
        pixel_dtype: torch.dtype,
        row_budget: int,
        max_images: int,
    ) -> None:
        self.module = module
        self.owner = owner
        self.stream = stream
        self.pool = pool
        self.peers = tuple(peers)
        self.limits = vision_limits(module)
        if graphs and self.limits is None:
            raise SetupError(
                f"{owner}: under CUDA graphs a vision encoder packs its images "
                "(PatchEncoder max_patches), which this one does not; serve "
                'it with cuda_graph_config.policy = "off"'
            )
        self.shapes: VisionShapes | None = None
        self.row_budget = int(row_budget)
        self.max_images = max(int(max_images), 1)
        self.pixel_dtype = pixel_dtype
        self.context = execution_context(
            module,
            0 if stream is None else stream.cuda_stream,
            attention,
            scratch,
            device=int(device),
            derive_host_lengths=True,
        )
        self.device = self.context.device
        self.graphs = GraphTable(owner)
        limits = self.limits
        if limits is None:
            self.context.prepare(TextSize(self.row_budget, self.max_images))
            return
        if graphs:
            self.shapes = vision_shapes(
                patch_sizes(limits), slot_sizes(limits), limits
            )
            patches = max(shape.patches for shape in self.shapes.shapes)
            slots = max(shape.slots for shape in self.shapes.shapes)
        else:
            patches, slots = limits.context
            slots -= 1
        # The extent of the buffers: the largest packing. Under graphs it is
        # one bucket's, the bucket of the most slots and patch rows, which
        # ``vision_shapes`` keeps for every slot of the largest image.
        self.extent = VisionShape(patches, slots)
        self.context.prepare(TextSize(patches, slots + 1))
        # Captured encodes address these buffers: every packing's patch rows,
        # its segment grids and its features.
        row = 3 * int(module.patch_size) ** 2
        width, dtype = features(module)
        with torch.cuda.stream(self.stream):
            self.pixels = torch.zeros(
                (patches, row), dtype=pixel_dtype, device=self.device
            )
            self.grids = torch.zeros(
                (slots + 1, 2), dtype=torch.int64, device=self.device
            )
            self.features = torch.zeros(
                (patches // limits.downsample**2, width),
                dtype=dtype,
                device=self.device,
            )

    def encode(self, images: Sequence[VisionImage]) -> GraphMode:
        """Encode ``images``, each into its product; how the forwards ran.

        Raises:
            ValueError: An image exceeds the tower's patch rows, or the
                encoder is sealed and holds no graph of a chunk's bucket.
        """
        if not images:
            return GraphMode.NONE
        if self.limits is None:
            return self._unpacked(images)
        unit = self.limits.downsample**2
        if any(
            image.patches > self.limits.max_patches
            or image.patches % unit
            or image.pixels.shape[0] != image.patches
            for image in images
        ):
            raise ValueError(
                f"{self.owner}: a vision image exceeds the tower's "
                f"{self.limits.max_patches} patch rows or its grid disagrees "
                "with its pixels"
            )
        # The largest packing, which every chunk fits.
        chunks = chunk_rows(
            range(len(images)),
            self.extent.patches,
            lambda index: images[index].patches,
            members=self.extent.slots,
        )
        mode = GraphMode.NONE
        for chunk in chunks:
            ran = self._packed([images[index] for index in chunk])
            mode = max(mode, ran)
        return mode

    def _packed(self, images: Sequence[VisionImage]) -> GraphMode:
        """Encode one chunk of images in one packed forward."""
        limits = self.limits
        assert limits is not None
        unit = limits.downsample**2
        used = sum(image.patches for image in images)
        shapes = [image.grid for image in images]
        if self.shapes is None:
            # Without buckets the forward packs exactly the images.
            key = None
            patches, grids = used, shapes
        else:
            bucket = self.shapes.select(len(images), used)
            if bucket is None:
                raise ValueError(
                    f"{self.owner}: no vision bucket holds {len(images)} "
                    f"images of {used} patch rows"
                )
            key = bucket
            patches = bucket.patches
            grids = segment_grids(
                shapes, bucket.slots, bucket.patches, limits.downsample
            )
        segments = len(grids)
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            offset = 0
            for image in images:
                self.pixels[offset : offset + image.patches].copy_(
                    image.pixels.to(
                        device=self.device,
                        dtype=self.pixels.dtype,
                        non_blocking=True,
                    )
                )
                offset += image.patches
            self.grids[:segments].copy_(
                torch.tensor(grids, dtype=torch.int64), non_blocking=True
            )
        mode = self._run(key, patches, segments)
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            offset = 0
            for image in images:
                count = image.patches // unit
                image.out.reshape(-1)[: count * self.features.shape[1]].view(
                    count, -1
                ).copy_(
                    self.features[offset : offset + count].to(
                        device=image.out.device,
                        dtype=image.out.dtype,
                        non_blocking=True,
                    )
                )
                offset += count
        return mode

    def _run(
        self, key: VisionShape | None, patches: int, segments: int
    ) -> GraphMode:
        """Encode the buffers' leading patch rows and segments."""
        graph = None if key is None else self.graphs.lookup(key)
        if graph is not None:
            with torch.cuda.stream(self.stream):
                graph.replay()
            return GraphMode.REPLAY
        self._forward(patches, segments)
        if key is None or self.pool is None or self.stream is None:
            return GraphMode.EAGER
        captured = cuda_graph.capture(
            lambda: self._forward(patches, segments),
            stream=self.stream,
            pool=self.pool,
            peers=self.peers,
        )
        self.graphs.insert(key, None, captured)
        return GraphMode.CAPTURE

    def _forward(self, patches: int, segments: int) -> None:
        """The encode a bucket's graph captures."""
        unit = self.module.downsample**2
        with torch.inference_mode(), self.context.activate():
            self.features[: patches // unit].copy_(
                self.module.encode_segments(
                    self.pixels[:patches], self.grids[:segments]
                )
            )

    def _unpacked(self, images: Sequence[VisionImage]) -> GraphMode:
        """Encode images of one pixel shape per call, eagerly."""
        groups = [
            chunk
            for group in shape_groups(
                (index, tuple(image.pixels.shape))
                for index, image in enumerate(images)
            )
            for chunk in chunk_rows(
                group,
                self.row_budget,
                lambda index: images[index].patches,
                members=self.max_images,
            )
        ]
        with torch.inference_mode(), self.context.activate():
            for group in groups:
                members = [images[index] for index in group]
                values = self.module.encode(
                    VisionInput(
                        tuple(
                            image.pixels.to(self.device, non_blocking=True)
                            for image in members
                        ),
                        tuple(
                            torch.tensor([image.grid], dtype=torch.int64).to(
                                self.device, non_blocking=True
                            )
                            for image in members
                        ),
                        tuple(image.grid for image in members),
                    )
                )
                for image, value in zip(members, values, strict=True):
                    image.out.reshape(-1)[: value.numel()].view(
                        value.shape
                    ).copy_(
                        value.to(
                            device=image.out.device,
                            dtype=image.out.dtype,
                            non_blocking=True,
                        )
                    )
        return GraphMode.EAGER

    def warm(self) -> None:
        """Warm and capture every bucket in capture order, largest first.

        Without graphs, encode the smallest packing once so the tower's
        kernels are selected before serving; a tower that does not pack
        encodes one zero image of one feature. The buffers hold zero pixels
        and the padding segments cover them.
        """
        limits = self.limits
        if limits is None:
            side = self.module.downsample
            row = 3 * self.module.patch_size**2
            with torch.inference_mode(), self.context.activate():
                self.module.encode(
                    VisionInput(
                        (
                            torch.zeros(
                                (side * side, row),
                                dtype=self.pixel_dtype,
                                device=self.device,
                            ),
                        ),
                        (
                            torch.tensor(
                                [[side, side]],
                                dtype=torch.int64,
                                device=self.device,
                            ),
                        ),
                        ((side, side),),
                    )
                )
            return
        if self.shapes is None:
            unit = limits.downsample**2
            self._fill_grids([(limits.downsample, limits.downsample)])
            self._run(None, unit, 1)
            return
        for bucket in self.shapes.capture_order():
            grids = segment_grids(
                [], bucket.slots, bucket.patches, limits.downsample
            )
            self._fill_grids(grids)
            self._run(bucket, bucket.patches, len(grids))

    def _fill_grids(self, grids: Sequence[tuple[int, int]]) -> None:
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            self.grids[: len(grids)].copy_(
                torch.tensor(list(grids), dtype=torch.int64), non_blocking=True
            )

    def capacity(self) -> tuple[int, int]:
        """``(patch rows, images)`` one packed encode holds; zero unpacked."""
        if self.limits is None:
            return 0, 0
        return self.extent.patches, self.extent.slots

    def keys(self) -> list[tuple[int, int]]:
        """The graph keys in capture order (empty without graphs)."""
        if self.shapes is None:
            return []
        return [bucket.key for bucket in self.shapes.capture_order()]

    def seal(self) -> None:
        self.graphs.seal()

    def kernels(self) -> list[str]:
        return kernel_records(self.context.kernels())

    def close(self, aborted: bool) -> None:
        try:
            self.graphs.close(aborted)
        finally:
            self.context.close(aborted=bool(aborted))
