"""The diffusion solver's scaled step: the ``scaled_add`` operator's Triton source.

One element of the step is ``target ← β · target + α · source``, applied in place: the latent
scaled by the schedule's coefficient and the model's velocity added at its own. The two operands
are read in their own element types (a BF16 accumulator stepped by an FP32 source is the case the
operator exists for) and the products and the sum are computed in FP32, so the store into the
target's type is the only rounding.

Both coefficients are per-call host scalars: no upload, no view.
"""

import triton
import triton.language as tl


@triton.jit
def _scaled_add_kernel(
    target_ptr,
    source_ptr,
    alpha,
    beta,
    elements,
    BLOCK: tl.constexpr,
):
    """Update one block of ``target`` from ``source``.

    The grid covers the flattened tensors. ``alpha`` and ``beta`` arrive as FP32 scalars in the
    low word of their ABI slots; the arithmetic is FP32, so a narrower operand type costs only its
    own representation.
    """
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    live = offsets < elements
    # A zero `beta` is a step that replaces the target rather than moving it, and the source may be
    # widening into a target nothing has written yet: an uninitialised word can hold a NaN or an
    # Infinity, and `0 * NaN` is a NaN the caller never asked to read. So the target is not loaded
    # at all when its coefficient is zero, and `other=0.0` keeps the arithmetic below unchanged.
    reads_target = beta != 0.0
    target = tl.load(target_ptr + offsets, mask=live & reads_target, other=0.0).to(tl.float32)
    source = tl.load(source_ptr + offsets, mask=live, other=0.0).to(tl.float32)
    value = beta * target + alpha * source
    tl.store(
        target_ptr + offsets,
        value.to(target_ptr.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=live,
    )
