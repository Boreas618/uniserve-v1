//! The kernel argument an operator's launch code is written in: a device address, an element
//! type, and the extents and strides of a tensor's layout.
//!
//! [`Operand`] is what a kernel package's tensor parameters are made of. Its address is
//! `platform`'s, its element type is the tensor library's `DType`, whose number and spelling for
//! a package [`crate::abi`] states, and its extents and strides are a tensor library [`Layout`]
//! whose start is the address itself. Everything here is plain data: no storage is owned and
//! nothing synchronizes. An operand borrows device memory its owner keeps alive until the stream
//! work that reads it completes.
//!
//! An operand takes as many axes as the tensor library's strided operations do,
//! [`MAX_DIMS`](tensor::nditer::MAX_DIMS): a tensor is a kernel argument whenever the CPU backend
//! can compute on it. Launch code works in the 64-bit words a kernel reads, so the extents and
//! strides are answered as `u64`.

use std::fmt;

use platform::DevicePtr;
use tensor::nditer::MAX_DIMS;
use tensor::{DType, Layout, Shape};

use crate::{Error, Result, abi};

/// A borrowed view of device memory: an address, an element type, extents and element strides.
///
/// The address is the first element; a kernel indexes from it with the strides, so a view of a
/// larger tensor needs no copy to be handed to an operator. Nothing here owns the memory: the
/// caller keeps it alive until the stream work reading it completes, including the replays of a
/// captured graph.
#[derive(Clone)]
pub struct Operand {
    /// The address of the first element.
    pub ptr: DevicePtr,
    pub dtype: DType,
    /// The extents and element strides, starting at `ptr`: the start offset is always zero.
    layout: Layout,
}

impl Operand {
    /// A row-major view of `dims` elements of `dtype` at `ptr`.
    pub fn new(ptr: DevicePtr, dtype: DType, dims: &[u64]) -> Self {
        Self {
            ptr,
            dtype,
            layout: Layout::contiguous(words(dims)),
        }
    }

    /// A view of `dims` elements of `dtype` at `ptr`, `strides` elements apart along each axis.
    ///
    /// # Panics
    ///
    /// When `strides` has another number of axes than `dims`.
    #[track_caller]
    pub fn with_strides(ptr: DevicePtr, dtype: DType, dims: &[u64], strides: &[u64]) -> Self {
        assert_eq!(dims.len(), strides.len(), "one stride per axis");
        Self {
            ptr,
            dtype,
            layout: Layout::new(Shape::from(words(dims)), words(strides), 0),
        }
    }

    /// The view `layout` states of the elements at `ptr`: `ptr` is the address of the element at
    /// the layout's start, and the view keeps the layout's extents and strides.
    pub fn with_layout(ptr: DevicePtr, dtype: DType, layout: &Layout) -> Self {
        Self {
            ptr,
            dtype,
            layout: Layout::new(layout.shape().clone(), layout.stride().to_vec(), 0),
        }
    }

    /// The argument of an operand a call does not read: no elements of `U8` at address zero.
    ///
    /// An operator whose mode leaves an operand unused (the combine of an expert exchange reads
    /// no routes) still takes one; a kernel never dereferences it.
    pub fn unread() -> Self {
        Self::new(DevicePtr::from_address(0), DType::U8, &[0])
    }

    /// The extents and strides, starting at the view's address.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The extents.
    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    /// The extents, in order.
    pub fn dims(&self) -> &[usize] {
        self.layout.dims()
    }

    /// The element strides, in order.
    pub fn strides(&self) -> &[usize] {
        self.layout.stride()
    }

    /// The number of axes.
    pub fn rank(&self) -> usize {
        self.dims().len()
    }

    /// The extent of `axis`; zero past the last axis.
    ///
    /// An operator's check reads the axes it expects before it compares the view's shape with
    /// them, so a view of fewer axes reads zero there and is refused by the comparison rather
    /// than ending the process.
    pub fn dim(&self, axis: usize) -> u64 {
        self.dims().get(axis).map_or(0, |&dim| dim as u64)
    }

    /// The element stride of `axis`; zero past the last axis, as [`Operand::dim`].
    pub fn stride(&self, axis: usize) -> u64 {
        self.strides().get(axis).map_or(0, |&stride| stride as u64)
    }

    /// The extent of the last axis; zero for a view of no axis.
    pub fn last(&self) -> u64 {
        self.dims().last().map_or(0, |&dim| dim as u64)
    }

    /// The stride of the last axis; zero for a view of no axis.
    pub fn last_stride(&self) -> u64 {
        self.strides().last().map_or(0, |&stride| stride as u64)
    }

    /// The number of elements the view names, the product of the extents; one for a view of no
    /// axis.
    pub fn numel(&self) -> u64 {
        self.dims().iter().map(|&dim| dim as u64).product()
    }

    /// The number of rows the view names, the product of every extent but the last.
    pub fn rows(&self) -> u64 {
        self.dims().split_last().map_or(1, |(_, leading)| {
            leading.iter().map(|&dim| dim as u64).product()
        })
    }

    /// The same view at another address, e.g. its shadow buffer.
    pub fn at(&self, ptr: DevicePtr) -> Self {
        Self {
            ptr,
            ..self.clone()
        }
    }

    /// The same elements under the extents `dims`, with the same element type and address.
    ///
    /// A view whose axes are row-major packed reshapes into the row-major packed view of the new
    /// extents: `[tokens, heads * head_dim]` refined to `[tokens, heads, head_dim]` strides
    /// `(heads * head_dim, head_dim, 1)`. A view that is not row-major cannot be reshaped without
    /// moving bytes, so its strides are kept and only the extents change - correct when the new
    /// extents refine or merge the last axis, which is the only reshape a strided view can
    /// express.
    pub fn reshape(&self, dims: &[u64]) -> Self {
        let layout = if self.is_row_major() {
            Layout::contiguous(words(dims))
        } else {
            Layout::new(Shape::from(words(dims)), self.strides().to_vec(), 0)
        };
        Self {
            layout,
            ..self.clone()
        }
    }

    /// Whether the axes are packed row-major: the last axis's stride is one and every axis's
    /// stride is the extent of the axis after it times that axis's stride.
    pub fn is_row_major(&self) -> bool {
        let mut expected = 1;
        for (&dim, &stride) in self.dims().iter().zip(self.strides()).rev() {
            if stride != expected {
                return false;
            }
            expected *= dim;
        }
        true
    }

    /// The bytes the view's elements occupy, when the element type has a whole-byte size.
    pub fn byte_len(&self) -> Result<u64> {
        let bytes = abi::bytes(self.dtype).ok_or_else(|| {
            Error::Shape(format!(
                "{} elements have no whole-byte size; its storage layout names its bytes",
                abi::name(self.dtype)
            ))
        })?;
        Ok(self.numel() * bytes)
    }

    /// The address of the element at `index`.
    pub fn element(&self, index: &[u64]) -> DevicePtr {
        let element = index
            .iter()
            .zip(self.strides())
            .map(|(value, &stride)| value * stride as u64)
            .sum::<u64>();
        self.offset_elements(element)
    }

    /// The `length` entries of `axis` starting at `start`, at the view's own strides: torch's
    /// `narrow`. Only that axis's extent and the address change, so the result names a subset of
    /// the same elements.
    ///
    /// # Errors
    ///
    /// An axis past the view's rank, or entries past the axis's extent.
    pub fn narrow(&self, axis: usize, start: u64, length: u64) -> Result<Self> {
        if axis >= self.rank() || start + length > self.dim(axis) {
            return Err(Error::Shape(format!(
                "entries {start}..{} of axis {axis} lie outside the view {:?}",
                start + length,
                self.dims()
            )));
        }
        let mut dims = self.dims().to_vec();
        dims[axis] = length as usize;

        Ok(Self {
            ptr: self.offset_elements(start * self.stride(axis)),
            dtype: self.dtype,
            layout: Layout::new(Shape::from(dims), self.strides().to_vec(), 0),
        })
    }

    /// The address `elements` elements further, when the element type has a whole-byte size.
    pub fn offset_elements(&self, elements: u64) -> DevicePtr {
        match abi::bytes(self.dtype) {
            Some(bytes) => DevicePtr::from_address(self.ptr.address() + elements * bytes),
            None => self.ptr,
        }
    }

    /// Check that this view is `dtype` with `last` elements on its last axis and a unit last
    /// stride, as every row-wise operator here reads it.
    ///
    /// A refusal names the view it refused: the operator alone does not say *which* of its operands
    /// disagrees with the descriptor, and an operator with several same-typed operands (a residual, an
    /// activation's two halves, a projection's input and output) leaves the reader to guess.
    pub fn require_row(&self, operator: &'static str, dtype: DType, last: u64) -> Result<()> {
        if self.dtype != dtype {
            return Err(Error::argument(
                operator,
                format!(
                    "expected {} elements, found {} in the view {:?}",
                    abi::name(dtype),
                    abi::name(self.dtype),
                    self.dims()
                ),
            ));
        }
        if self.last() != last {
            return Err(Error::argument(
                operator,
                format!(
                    "expected {last} elements on the last axis, found {} in the view {:?}",
                    self.last(),
                    self.dims()
                ),
            ));
        }
        if self.last_stride() != 1 {
            return Err(Error::argument(
                operator,
                format!(
                    "expected a unit stride on the last axis, found {} in the view {:?}",
                    self.last_stride(),
                    self.dims()
                ),
            ));
        }
        Ok(())
    }

    /// Check that this view has the extents `dims`.
    pub fn require_shape(&self, operator: &'static str, dims: &[u64]) -> Result<()> {
        if !self.has_dims(dims) {
            return Err(Error::argument(
                operator,
                format!("expected shape {dims:?}, found {:?}", self.dims()),
            ));
        }
        Ok(())
    }

    /// Check that this view has the extents `dims`, another view's.
    pub fn require_dims(&self, operator: &'static str, dims: &[usize]) -> Result<()> {
        if self.dims() != dims {
            return Err(Error::argument(
                operator,
                format!("expected shape {dims:?}, found {:?}", self.dims()),
            ));
        }
        Ok(())
    }

    /// Whether this view has the extents `dims`.
    pub fn has_dims(&self, dims: &[u64]) -> bool {
        self.rank() == dims.len()
            && self
                .dims()
                .iter()
                .zip(dims)
                .all(|(&ours, &theirs)| ours as u64 == theirs)
    }

    /// Check that this view has `rank` axes.
    pub fn require_rank(&self, operator: &'static str, rank: usize) -> Result<()> {
        if self.rank() != rank {
            return Err(Error::argument(
                operator,
                format!("expected {rank} axes, found {}", self.rank()),
            ));
        }
        Ok(())
    }
}

impl fmt::Debug for Operand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Operand")
            .field("ptr", &format_args!("{:#x}", self.ptr.address()))
            .field("dtype", &self.dtype)
            .field("shape", &self.dims())
            .field("strides", &self.strides())
            .finish()
    }
}

/// The extents or strides `values` as the tensor library's words.
fn words(values: &[u64]) -> Vec<usize> {
    values.iter().map(|&value| value as usize).collect()
}

/// Check that `out` can be written one row at a time, and answer its outer strides.
///
/// A decoder writes each tile into its own sub-region of one bigger raster, so a destination is a
/// strided block by construction. What a row-at-a-time write needs of it, and no more: the innermost
/// axis is unit-strided — so a row is written contiguously and the inner loop vectorizes — and no two
/// rows overlap, so each outer stride is at least the sweep of the axes inside it. A transposed
/// destination is refused: nobody synthesizes one, and allowing it would put a gather in the inner
/// loop for no caller.
///
/// The outer strides are answered in axis order (`[channel, frame, row]` for a four-axis raster) so
/// the operator can pass them as launch parameters; the innermost stride is 1 by the rule and never
/// travels.
pub(crate) fn row_writable(out: &Operand, op: &'static str) -> Result<[u64; MAX_DIMS]> {
    row_layout(out, op, "destination")
}

/// The outer strides of a **source** read a row at a time.
///
/// The same rule as [`row_writable`] — a unit-strided last axis and outer strides that do not let
/// one row overlap the next — stated for an operand rather than a destination. A source may be a
/// view inside a larger buffer (a tile of a raster, or the seam band between two of them), so the
/// operator needs its strides; a source whose last axis is not contiguous cannot be read a row at a
/// time at all and is refused by name.
pub(crate) fn row_readable(source: &Operand, op: &'static str) -> Result<[u64; MAX_DIMS]> {
    row_layout(source, op, "source")
}

fn row_layout(view: &Operand, op: &'static str, role: &str) -> Result<[u64; MAX_DIMS]> {
    if view.last_stride() != 1 || view.rank() == 0 {
        return Err(Error::argument(
            op,
            format!(
                "a {role} read or written a row at a time must be unit-strided along its last axis; \
                 found strides {:?}",
                view.strides()
            ),
        ));
    }
    for axis in (0..view.rank() - 1).rev() {
        let inner = view.dim(axis + 1) * view.stride(axis + 1);
        if view.stride(axis) < inner {
            return Err(Error::argument(
                op,
                format!(
                    "axis {axis} of the {role} advances by {} but its inner axes span {inner}: \
                     the rows would overlap",
                    view.stride(axis)
                ),
            ));
        }
    }
    Ok(outer_strides(view))
}

/// The outer strides of a view [`row_writable`] or [`row_readable`] accepted, in axis order.
///
/// An operator's `check` states the row rule and its `launch` reads the strides here, so the rule
/// is checked once per call.
pub(crate) fn outer_strides(view: &Operand) -> [u64; MAX_DIMS] {
    let mut outer = [0_u64; MAX_DIMS];
    let axes = view.rank().saturating_sub(1);
    for (axis, stride) in outer[..axes].iter_mut().enumerate() {
        *stride = view.stride(axis);
    }
    outer
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(shape: &[u64], strides: &[u64]) -> Operand {
        Operand::with_strides(DevicePtr::from_address(0x1000), DType::BF16, shape, strides)
    }

    #[test]
    fn a_reshape_of_a_packed_view_is_packed_at_its_new_extents() {
        // `[tokens, heads * head_dim]` refined to `[tokens, heads, head_dim]`: the elements do not
        // move, so the refined axes are row-major at the new extents. Carrying the packed pair over
        // left the head axis at stride one, which read a token's four heads as one element.
        let packed = view(&[15, 512], &[512, 1]);
        let refined = packed.reshape(&[15, 4, 128]);
        assert_eq!(refined.stride(0), 512);
        assert_eq!(refined.stride(1), 128);
        assert_eq!(refined.last_stride(), 1);
    }

    #[test]
    fn a_narrowed_view_moves_its_address_and_keeps_its_strides() {
        // Frames 1..3 of a `[3, 4, 8, 16]` raster: one frame is 128 BF16 elements, 256 bytes.
        let raster = view(&[3, 4, 8, 16], &[512, 128, 16, 1]);
        let frames = raster.narrow(1, 1, 2).unwrap();
        assert_eq!(frames.dims(), &[3, 2, 8, 16]);
        assert_eq!(frames.strides(), raster.strides());
        assert_eq!(frames.ptr.address(), 0x1000 + 256);
        // Rows of a matrix are the leading axis.
        let rows = view(&[6, 4], &[4, 1]).narrow(0, 5, 1).unwrap();
        assert_eq!(rows.ptr.address(), 0x1000 + 40);
        assert!(raster.narrow(1, 3, 2).is_err());
        assert!(raster.narrow(4, 0, 1).is_err());
    }

    #[test]
    fn a_reshape_of_a_strided_view_keeps_the_layout_it_cannot_move() {
        // A last axis that is not unit-strided cannot be repacked without a copy, so the strides are
        // the caller's and only the extents change.
        let strided = view(&[8, 16], &[512, 8]);
        let reshaped = strided.reshape(&[8, 2, 8]);
        assert_eq!(reshaped.stride(0), 512);
        assert_eq!(reshaped.stride(1), 8);
    }

    #[test]
    fn a_row_major_destination_is_writable_and_so_is_a_block_of_a_bigger_one() {
        // A contiguous `[2, 3, 8, 16]` raster: the outer strides are the row-major ones.
        let contiguous = &[384, 128, 16, 1];
        assert_eq!(
            row_writable(&view(&[2, 3, 8, 16], contiguous), "test").unwrap()[..3],
            [384, 128, 16]
        );

        // The same tile placed at an origin inside a `[2, 6, 64, 128]` parent: the parent's strides
        // are the tile's, and only the base address moved — which is exactly the decoder's case.
        let placed = &[49152, 8192, 128, 1];
        assert_eq!(
            row_writable(&view(&[2, 3, 8, 16], placed), "test").unwrap()[..3],
            [49152, 8192, 128]
        );
    }

    #[test]
    fn a_transposed_or_overlapping_destination_is_refused() {
        // The last axis is not unit-strided: a row would be a gather, and no caller synthesizes one.
        assert!(row_writable(&view(&[2, 3, 8, 16], &[384, 128, 1, 8]), "test").is_err());
        // The rows overlap: the row stride is shorter than a row.
        assert!(row_writable(&view(&[2, 3, 8, 16], &[384, 128, 8, 1]), "test").is_err());
        // The planes overlap, which a row-size check alone would miss.
        assert!(row_writable(&view(&[2, 3, 8, 16], &[384, 64, 16, 1]), "test").is_err());
    }

    #[test]
    fn a_view_of_the_most_axes_a_tensor_takes_is_written_a_row_at_a_time() {
        // Eight axes, the tensor library's bound: every outer stride is answered.
        let dims = [2, 2, 2, 2, 2, 2, 2, 4];
        let packed = Operand::new(DevicePtr::from_address(0x1000), DType::F32, &dims);
        let outer = row_writable(&packed, "test").unwrap();
        assert_eq!(outer[..7], [256, 128, 64, 32, 16, 8, 4]);
    }
}
