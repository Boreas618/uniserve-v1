//! The numerical ABI an operator's launch code is written in: element types, shapes, strides and
//! borrowed tensor views.
//!
//! These are the types a kernel program's parameters are made of (specs/worker.md 4.2, 4.3). They
//! live in `ops` because `ops` depends on `platform` alone and both `ops` and `nn` need them; `nn`
//! re-exports them. Everything here is plain data: no storage is owned, no allocation happens and
//! nothing synchronizes. A view borrows device memory its owner keeps alive until the stream work
//! that reads it completes.

use std::fmt;

use crate::{Error, Result, platform};

/// A device address: an opaque `u64` the driver uses (`CUdeviceptr`).
///
/// `ops` carries addresses in its own type rather than as bare integers, so that a view's address
/// cannot be confused with a scalar parameter and a byte offset cannot be added to an element
/// count by accident. Arithmetic on it is in bytes ([`DevicePtr::offset`]); the element offset of a
/// view goes through [`TensorView::offset_elements`].
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

/// An element type.
///
/// The numbering is part of the package ABI: a package's C header carries the same values, so a
/// kernel program reads the element type the operator launched it with. Do not renumber.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i64)]
pub enum DType {
    U8 = 0,
    I32 = 1,
    I64 = 2,
    F16 = 3,
    BF16 = 4,
    F32 = 5,
    /// FP8 E4M3 (`.e4m3fn`), the FP8 paged-KV element type.
    F8E4M3 = 6,
    /// NVFP4 E2M1, two values in one byte.
    F4E2M1 = 7,
}

impl DType {
    /// The ABI numbering a package header must carry.
    pub const fn as_i64(self) -> i64 {
        self as i64
    }

    /// The element type with ABI number `value`.
    pub const fn from_i64(value: i64) -> Option<Self> {
        Some(match value {
            0 => Self::U8,
            1 => Self::I32,
            2 => Self::I64,
            3 => Self::F16,
            4 => Self::BF16,
            5 => Self::F32,
            6 => Self::F8E4M3,
            7 => Self::F4E2M1,
            _ => return None,
        })
    }

    /// The number of bits one element occupies in memory.
    pub const fn bits(self) -> u32 {
        match self {
            Self::F4E2M1 => 4,
            Self::U8 | Self::F8E4M3 => 8,
            Self::F16 | Self::BF16 => 16,
            Self::I32 | Self::F32 => 32,
            Self::I64 => 64,
        }
    }

    /// The number of bytes one element occupies, when an element is a whole number of bytes.
    ///
    /// A sub-byte element type is addressed by its encoded storage layout, which the operators that
    /// read it (`gemm` on an NVFP4 weight, for one) name; everything that hands out bytes fails for
    /// it rather than rounding.
    pub const fn bytes(self) -> Option<u64> {
        match self.bits() {
            4 => None,
            bits => Some((bits / 8) as u64),
        }
    }

    /// Whether the element type holds real numbers.
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            Self::F16 | Self::BF16 | Self::F32 | Self::F8E4M3 | Self::F4E2M1
        )
    }

    /// The unit round-off `u` of the element type: half the gap between one and the next value
    /// above it, which is the factor every rounding bound is written with (Higham, *Accuracy and
    /// Stability of Numerical Algorithms*, 2nd ed., §2.2).
    pub const fn unit_roundoff(self) -> f64 {
        match self {
            Self::F32 => 5.960_464_477_539_063e-8, // 2^-24
            Self::BF16 => 3.906_25e-3,             // 2^-8
            Self::F16 => 4.882_812_5e-4,           // 2^-11
            Self::F8E4M3 => 6.25e-2,               // 2^-4
            Self::F4E2M1 => 2.5e-1,                // 2^-2
            Self::U8 | Self::I32 | Self::I64 => 0.0,
        }
    }

    /// The element type a reduction over this one accumulates in.
    ///
    /// Every operator here accumulates a floating reduction in FP32 (specs/worker.md 9), so the
    /// bounds are written with `u = 2^-24` whatever the input dtype is.
    pub const fn accumulate(self) -> Self {
        Self::F32
    }
}

impl fmt::Display for DType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::U8 => "u8",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F32 => "f32",
            Self::F8E4M3 => "f8e4m3",
            Self::F4E2M1 => "f4e2m1",
        };
        formatter.write_str(name)
    }
}

impl fmt::Debug for DType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

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

    /// This shape with one axis replaced.
    ///
    /// # Panics
    ///
    /// When `axis` is past the last one.
    #[track_caller]
    pub fn with_dim(&self, axis: usize, extent: u64) -> Self {
        let mut changed = *self;
        changed.dims[axis] = extent;
        changed
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

/// A borrowed view of device memory: an address, an element type, extents and element strides.
///
/// The address is the first element; a kernel indexes from it with the strides, so a view of a
/// larger tensor needs no copy to be handed to an operator. Nothing here owns the memory: the
/// caller keeps it alive until the stream work reading it completes, including the replays of a
/// captured graph.
#[derive(Clone, Copy)]
pub struct TensorView {
    pub ptr: DevicePtr,
    pub dtype: DType,
    pub shape: Shape,
    pub strides: Strides,
}

impl TensorView {
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

    /// The same view of other extents, with the same element type and strides.
    pub fn reshape(&self, shape: Shape) -> Self {
        Self { shape, ..*self }
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
        let bytes = self.dtype.bytes().ok_or_else(|| {
            Error::Shape(format!(
                "{:?} elements have no whole-byte size; its storage layout names its bytes",
                self.dtype
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

    /// The address `elements` elements further, when the element type has a whole-byte size.
    pub fn offset_elements(&self, elements: u64) -> DevicePtr {
        match self.dtype.bytes() {
            Some(bytes) => platform::offset(self.ptr, elements * bytes),
            None => self.ptr,
        }
    }

    /// Check that this view is `dtype` with `last` elements on its last axis and a unit last
    /// stride, as every row-wise operator here reads it.
    pub fn require_row(&self, operator: &'static str, dtype: DType, last: u64) -> Result<()> {
        if self.dtype != dtype {
            return Err(Error::argument(
                operator,
                format!("expected {} elements, found {}", dtype, self.dtype),
            ));
        }
        if self.last() != last {
            return Err(Error::argument(
                operator,
                format!(
                    "expected {last} elements on the last axis, found {}",
                    self.last()
                ),
            ));
        }
        if self.strides.last() != 1 {
            return Err(Error::argument(
                operator,
                format!(
                    "expected a unit stride on the last axis, found {}",
                    self.strides.last()
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

impl fmt::Debug for TensorView {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TensorView")
            .field("ptr", &self.ptr)
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .finish()
    }
}
