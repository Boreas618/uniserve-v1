//! The kernel argument an operator's launch code is written in: a device address, an element
//! type, and inline extents and strides.
//!
//! [`Operand`] is what a kernel package's tensor parameters are made of. Its element type is the
//! tensor library's `DType`, whose number and spelling for a package [`crate::abi`] states; its
//! [`Shape`] and [`Strides`] are inline arrays of at most [`MAX_RANK`] axes, so that an operand
//! is `Copy` and an operator's argument struct can be rebound without allocating. Everything
//! here is plain data: no storage is owned, no allocation happens and nothing synchronizes. An
//! operand borrows device memory its owner keeps alive until the stream work that reads it
//! completes.

use std::fmt;

use tensor::DType;

use crate::{Error, Result, abi, platform};

/// A device address: an opaque `u64` the driver uses (`CUdeviceptr`).
///
/// `ops` carries addresses in its own type rather than as bare integers, so that a view's address
/// cannot be confused with a scalar parameter and a byte offset cannot be added to an element
/// count by accident. Arithmetic on it is in bytes ([`DevicePtr::offset`]); the element offset of a
/// view goes through [`Operand::offset_elements`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DevicePtr(u64);

impl DevicePtr {
    /// The address of `address`.
    pub const fn new(address: u64) -> Self {
        Self(address)
    }

    /// The address as the driver sees it.
    pub const fn address(self) -> u64 {
        self.0
    }

    /// The address `bytes` bytes further.
    pub const fn offset(self, bytes: u64) -> Self {
        Self(self.0 + bytes)
    }
}

impl std::fmt::Debug for DevicePtr {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "0x{:x}", self.0)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, dim) in self.dims().iter().enumerate() {
            if index > 0 {
                formatter.write_str("x")?;
            }
            write!(formatter, "{dim}")?;
        }
        Ok(())
    }
}

impl fmt::Display for Strides {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let strides = self.as_slice();
        for (index, stride) in strides.iter().enumerate() {
            if index > 0 {
                formatter.write_str("x")?;
            }
            write!(formatter, "{stride}")?;
        }
        Ok(())
    }
}

impl std::fmt::Display for DevicePtr {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "0x{:x}", self.0)
    }
}

/// The most axes a [`Shape`] or [`Strides`] holds.
///
/// Views are inline values, not slices, so an operator's argument struct is `Copy` and can be
/// rebound without allocating. Six covers every operator here (a 5-d convolution, a 4-d paged
/// cache); an operator that needs more states its own geometry instead.
pub const MAX_RANK: usize = 6;

/// The extents of each axis of a view.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: [u64; MAX_RANK],
    rank: u8,
}

impl Shape {
    /// The shape of `dims`, in order.
    ///
    /// # Panics
    ///
    /// When `dims` has more than [`MAX_RANK`] axes.
    #[track_caller]
    pub const fn new(dims: &[u64]) -> Self {
        assert!(
            dims.len() <= MAX_RANK,
            "a shape has at most MAX_RANK (6) axes"
        );
        let mut buffer = [0_u64; MAX_RANK];
        let mut axis = 0;
        while axis < dims.len() {
            buffer[axis] = dims[axis];
            axis += 1;
        }
        Self {
            dims: buffer,
            rank: dims.len() as u8,
        }
    }

    /// The extents, in order.
    pub fn dims(&self) -> &[u64] {
        &self.dims[..usize::from(self.rank)]
    }

    /// The number of axes.
    pub fn rank(&self) -> usize {
        usize::from(self.rank)
    }

    /// The extent of `axis`.
    ///
    /// # Panics
    ///
    /// When `axis` is past the last one.
    #[track_caller]
    pub fn dim(&self, axis: usize) -> u64 {
        self.dims[axis]
    }

    /// The extent of the last axis.
    pub fn last(&self) -> u64 {
        self.dims[usize::from(last_axis(self.rank))]
    }

    /// The number of elements, the product of the extents; one for a rank-0 shape.
    pub fn numel(&self) -> u64 {
        self.dims().iter().product()
    }

    /// The number of rows: the product of every extent but the last.
    pub fn rows(&self) -> u64 {
        self.dims()
            .split_last()
            .map_or(1, |(_, leading)| leading.iter().product())
    }

    /// This shape with its last extent replaced.
    pub fn with_last(&self, last: u64) -> Self {
        let mut dims = *self;
        dims.dims[usize::from(last_axis(self.rank))] = last;
        dims
    }

    /// Whether this shape and `other` have the same extents.
    pub fn matches(&self, other: &Shape) -> bool {
        self.dims() == other.dims()
    }
}

/// A last axis index that is valid whether or not the shape has any axis, for writes that are
/// discarded on a rank-0 shape.
const fn last_axis(rank: u8) -> u8 {
    if rank == 0 { 0 } else { rank - 1 }
}

impl fmt::Debug for Shape {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.dims()).finish()
    }
}

/// The stride of each axis of a view, in elements.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Strides {
    strides: [u64; MAX_RANK],
    rank: u8,
}

impl Strides {
    /// The strides of `strides`, in order.
    ///
    /// # Panics
    ///
    /// When `strides` has more than [`MAX_RANK`] axes.
    #[track_caller]
    pub const fn new(strides: &[u64]) -> Self {
        assert!(
            strides.len() <= MAX_RANK,
            "strides have at most MAX_RANK (6) axes"
        );
        let mut buffer = [0_u64; MAX_RANK];
        let mut axis = 0;
        while axis < strides.len() {
            buffer[axis] = strides[axis];
            axis += 1;
        }
        Self {
            strides: buffer,
            rank: strides.len() as u8,
        }
    }

    /// The row-major strides of `shape`: the last axis is contiguous.
    pub fn row_major(shape: Shape) -> Self {
        let mut buffer = [0_u64; MAX_RANK];
        let mut stride = 1;
        for axis in (0..shape.rank()).rev() {
            buffer[axis] = stride;
            stride *= shape.dim(axis);
        }
        Self {
            strides: buffer,
            rank: shape.rank() as u8,
        }
    }

    /// The strides, in order.
    pub fn as_slice(&self) -> &[u64] {
        &self.strides[..usize::from(self.rank)]
    }

    /// The number of axes.
    pub fn rank(&self) -> usize {
        usize::from(self.rank)
    }

    /// The stride of `axis`.
    ///
    /// # Panics
    ///
    /// When `axis` is past the last one.
    #[track_caller]
    pub fn get(&self, axis: usize) -> u64 {
        self.strides[axis]
    }

    /// The stride of the last axis.
    pub fn last(&self) -> u64 {
        self.strides[usize::from(last_axis(self.rank))]
    }
}

impl fmt::Debug for Strides {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.as_slice()).finish()
    }
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
pub fn row_writable(out: &Operand, op: &'static str) -> Result<[u64; MAX_RANK]> {
    row_layout(out, op, "destination")
}

/// The outer strides of a **source** read a row at a time.
///
/// The same rule as [`row_writable`] — a unit-strided last axis and outer strides that do not let
/// one row overlap the next — stated for an operand rather than a destination. A source may be a
/// view inside a larger buffer (a tile of a raster, or the seam band between two of them), so the
/// operator needs its strides; a source whose last axis is not contiguous cannot be read a row at a
/// time at all and is refused by name.
pub fn row_readable(source: &Operand, op: &'static str) -> Result<[u64; MAX_RANK]> {
    row_layout(source, op, "source")
}

fn row_layout(view: &Operand, op: &'static str, role: &str) -> Result<[u64; MAX_RANK]> {
    let out = view;
    let strides = out.strides.as_slice();
    if strides.last() != Some(&1) {
        return Err(Error::argument(
            op,
            format!(
                "a {role} read or written a row at a time must be unit-strided along its last axis; \
                 found strides {:?}",
                out.strides
            ),
        ));
    }
    for axis in (0..strides.len() - 1).rev() {
        let inner = out.shape.dim(axis + 1) * strides[axis + 1];
        if strides[axis] < inner {
            return Err(Error::argument(
                op,
                format!(
                    "axis {axis} of the {role} advances by {} but its inner axes span {inner}: \
                     the rows would overlap",
                    strides[axis]
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
pub fn outer_strides(view: &Operand) -> [u64; MAX_RANK] {
    let strides = view.strides.as_slice();
    let mut outer = [0_u64; MAX_RANK];
    let axes = strides.len().saturating_sub(1);
    outer[..axes].copy_from_slice(&strides[..axes]);
    outer
}

/// A borrowed view of device memory: an address, an element type, extents and element strides.
///
/// The address is the first element; a kernel indexes from it with the strides, so a view of a
/// larger tensor needs no copy to be handed to an operator. Nothing here owns the memory: the
/// caller keeps it alive until the stream work reading it completes, including the replays of a
/// captured graph.
#[derive(Clone, Copy)]
pub struct Operand {
    pub ptr: DevicePtr,
    pub dtype: DType,
    pub shape: Shape,
    pub strides: Strides,
}

impl Operand {
    /// A row-major view of `shape` elements of `dtype` at `ptr`.
    pub fn new(ptr: DevicePtr, dtype: DType, shape: Shape) -> Self {
        Self {
            ptr,
            dtype,
            shape,
            strides: Strides::row_major(shape),
        }
    }

    /// A view with explicit strides, in elements.
    pub fn with_strides(ptr: DevicePtr, dtype: DType, shape: Shape, strides: Strides) -> Self {
        Self {
            ptr,
            dtype,
            shape,
            strides,
        }
    }

    /// The same view at another address, e.g. its shadow buffer.
    pub fn at(&self, ptr: DevicePtr) -> Self {
        Self { ptr, ..*self }
    }

    /// The same elements under other extents, with the same element type and address.
    ///
    /// A view whose axes are row-major packed reshapes into the row-major packed view of the new
    /// shape, so its strides are the new shape's own: `[tokens, heads * head_dim]` refined to
    /// `[tokens, heads, head_dim]` strides `(heads * head_dim, head_dim, 1)`. A view that is not
    /// row-major cannot be reshaped without moving bytes, so its strides are kept and only the
    /// shape changes - correct when the new extents refine or merge the last axis, which is the
    /// only reshape a strided view can express.
    pub fn reshape(&self, shape: Shape) -> Self {
        let strides = if self.is_row_major() {
            Strides::row_major(shape)
        } else {
            self.strides
        };
        Self {
            shape,
            strides,
            ..*self
        }
    }

    /// Whether the axes are packed row-major: the last axis's stride is one and every axis's
    /// stride is the extent of the axis after it times that axis's stride.
    pub fn is_row_major(&self) -> bool {
        let mut expected = 1_u64;
        for axis in (0..self.strides.rank()).rev() {
            if self.strides.get(axis) != expected {
                return false;
            }
            expected *= self.shape.dim(axis);
        }
        true
    }

    /// The number of elements the view names.
    pub fn numel(&self) -> u64 {
        self.shape.numel()
    }

    /// The number of rows the view names, the product of every extent but the last.
    pub fn rows(&self) -> u64 {
        self.shape.rows()
    }

    /// The extent of the last axis.
    pub fn last(&self) -> u64 {
        self.shape.last()
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
            .zip(self.strides.as_slice())
            .map(|(value, stride)| value * stride)
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
        if axis >= self.shape.rank() || start + length > self.shape.dim(axis) {
            return Err(Error::Shape(format!(
                "entries {start}..{} of axis {axis} lie outside the view {:?}",
                start + length,
                self.shape
            )));
        }
        let mut dims = self.shape.dims().to_vec();
        dims[axis] = length;
        Ok(Self {
            ptr: self.offset_elements(start * self.strides.get(axis)),
            shape: Shape::new(&dims),
            ..*self
        })
    }

    /// The address `elements` elements further, when the element type has a whole-byte size.
    pub fn offset_elements(&self, elements: u64) -> DevicePtr {
        match abi::bytes(self.dtype) {
            Some(bytes) => platform::offset(self.ptr, elements * bytes),
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
                    self.shape
                ),
            ));
        }
        if self.last() != last {
            return Err(Error::argument(
                operator,
                format!(
                    "expected {last} elements on the last axis, found {} in the view {:?}",
                    self.last(),
                    self.shape
                ),
            ));
        }
        if self.strides.last() != 1 {
            return Err(Error::argument(
                operator,
                format!(
                    "expected a unit stride on the last axis, found {} in the view {:?}",
                    self.strides.last(),
                    self.shape
                ),
            ));
        }
        Ok(())
    }

    /// Check that this view has the extents of `shape`.
    pub fn require_shape(&self, operator: &'static str, shape: Shape) -> Result<()> {
        if !self.shape.matches(&shape) {
            return Err(Error::argument(
                operator,
                format!(
                    "expected shape {:?}, found {:?}",
                    shape.dims(),
                    self.shape.dims()
                ),
            ));
        }
        Ok(())
    }

    /// Check that this view has `rank` axes.
    pub fn require_rank(&self, operator: &'static str, rank: usize) -> Result<()> {
        if self.shape.rank() != rank {
            return Err(Error::argument(
                operator,
                format!("expected {rank} axes, found {}", self.shape.rank()),
            ));
        }
        Ok(())
    }
}

impl fmt::Debug for Operand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Operand")
            .field("ptr", &self.ptr)
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(shape: &[u64], strides: &[u64]) -> Operand {
        Operand {
            ptr: DevicePtr::new(0x1000),
            dtype: DType::BF16,
            shape: Shape::new(shape),
            strides: Strides::new(strides),
        }
    }

    #[test]
    fn a_reshape_of_a_packed_view_is_packed_at_its_new_extents() {
        // `[tokens, heads * head_dim]` refined to `[tokens, heads, head_dim]`: the elements do not
        // move, so the refined axes are row-major at the new extents. Carrying the packed pair over
        // left the head axis at stride one, which read a token's four heads as one element.
        let packed = view(&[15, 512], &[512, 1]);
        let refined = packed.reshape(Shape::new(&[15, 4, 128]));
        assert_eq!(refined.strides.get(0), 512);
        assert_eq!(refined.strides.get(1), 128);
        assert_eq!(refined.strides.last(), 1);
    }

    #[test]
    fn a_narrowed_view_moves_its_address_and_keeps_its_strides() {
        // Frames 1..3 of a `[3, 4, 8, 16]` raster: one frame is 128 BF16 elements, 256 bytes.
        let raster = view(&[3, 4, 8, 16], &[512, 128, 16, 1]);
        let frames = raster.narrow(1, 1, 2).unwrap();
        assert_eq!(frames.shape.dims(), &[3, 2, 8, 16]);
        assert_eq!(frames.strides, raster.strides);
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
        let reshaped = strided.reshape(Shape::new(&[8, 2, 8]));
        assert_eq!(reshaped.strides.get(0), 512);
        assert_eq!(reshaped.strides.get(1), 8);
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
}
