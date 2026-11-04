"""The seam between two decoded tiles: the ``cross_fade`` operator, and why this package refuses.

The reference blends the shared band of two decoded video tiles, over the leading ``extent``
voxels of one axis of a ``[channels, frames, height, width]`` tile:

```text
for i < extent along `axis`:
    out[..., i, ...] = previous[..., −extent + i, ...] · (1 − i/extent)
                     + current [..., i,          ...] · (i/extent)
elsewhere along that axis:
    out[..., i, ...] = current[..., i, ...]
```

The weights are the reference's own integer ratios, so the operator's tolerance is exact and one
implementation produces its bits — but only if the kernel can say *which* position `i` a voxel is.

It cannot, from the launch it receives. `cross_fade.rs`'s argument pack is
``previous, current, out, extent, axis, channels, frames, block, pdl``: the height and the width
are absent, and the blend needs each of them separately. So there is no kernel here — the build
script refuses with that reason, and this module stands where the kernel will stand.

When the operator passes ``dims[2]`` and ``dims[3]``, this file gains the kernel and
``src/build.py`` becomes the usual Triton build:

```python
@triton.jit
def _cross_fade_kernel(previous_ptr, current_ptr, out_ptr, extent, axis, channels, frames,
                      height, width, block, pdl, BLOCK: tl.constexpr): ...
```

with the grid walking the output, each lane decoding its voxel into (channel, frame, row, column),
blending inside the band and copying outside it — which is also what lets ``out`` alias ``current``.
"""
