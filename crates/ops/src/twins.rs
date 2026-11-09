//! Shadow buffers for shadow execution.
//!
//! During a program-layer update, the new kernel of an operator runs beside the old one on the same
//! inputs: the old kernel writes the real outputs, the new one writes twins of them, and the two are
//! compared within the operator's [`Tolerance`] (specs/worker.md 7.3). Because the
//! new kernel's result is never fed on, the difference of one operator does not accumulate through
//! the rest of the forward.
//!
//! A [`Twins`] is one region of device memory from which [`Op::shadow`](crate::Op::shadow) carves the
//! twins of one paired execution. It comes from the worker's scratch pool, so shadow buffers never
//! touch the resident memory of the serving model and never enter the graph pool.
//!
//! ## Why the carving does not return a result
//!
//! `Op::shadow` returns the operator's arguments, not a result: shadow execution happens inside an
//! operator, which cannot fail the caller. So a carving that cannot fit records the failure in the
//! region and hands back the real view — the launch is still well defined, the model still runs, and
//! the paired-execution harness reads [`Twins::failure`] after the forward and fails the precopy
//! there. A region that is too small therefore cannot turn into a comparison of a buffer against
//! itself.

use crate::platform::{Allocator, Buffer, PoolKind, Stream};
use crate::tensor::{DType, DevicePtr, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Result, platform};

/// The alignment every twin buffer starts at.
///
/// Kernels read their operands through vector loads, so a twin must be at least as aligned as the
/// tensor it shadows; 256 bytes covers every element type and vector width here.
pub const ALIGNMENT: u64 = 256;

/// One shadow buffer and the real buffer it shadows.
#[derive(Clone, Copy, Debug)]
pub struct Twin {
    /// Where the new implementation writes.
    pub shadow: TensorView,
    /// Where the old implementation writes, and what the comparison reads as the reference.
    pub real: TensorView,
}

/// The shadow buffers of one paired execution.
///
/// The region is the worker's scratch pool (specs/worker.md 4.5, 7.3), never the graph pool: a
/// paired forward writes resident memory not at all, and `Twins` only carves views of the bytes the
/// worker gave it.
pub struct Twins {
    buffer: Buffer,
    cursor: u64,
    pairs: Vec<Twin>,
    /// The pairs whose real side has been copied out (`Self::freeze`).
    frozen: usize,
    tolerance: Option<Tolerance>,
    failure: Option<Error>,
}

impl Twins {
    /// A twin region of `bytes` from `allocator`'s scratch pool.
    ///
    /// The size is the largest shadow set of the buckets the runner shadows; a paired execution
    /// whose operator needs more records the failure in the region rather than allocating during a
    /// step. The allocation is the worker's, made once per runner and owned by the `Twins`.
    pub fn new(allocator: &mut Allocator, bytes: u64) -> Result<Self> {
        let bytes = usize::try_from(bytes.max(ALIGNMENT))
            .map_err(|_| Error::Shape("a twin region larger than usize::MAX".to_owned()))?;
        let buffer = allocator.allocate(PoolKind::Scratch, bytes)?;
        Ok(Self {
            buffer,
            cursor: 0,
            pairs: Vec::new(),
            frozen: 0,
            tolerance: None,
            failure: None,
        })
    }

    /// The address of the region's first byte.
    fn base(&self) -> DevicePtr {
        DevicePtr::new(self.buffer.address())
    }

    /// The number of bytes this region holds.
    pub fn capacity(&self) -> u64 {
        self.buffer.len() as u64
    }

    /// The bytes not yet handed out in this execution.
    pub fn available(&self) -> u64 {
        self.capacity().saturating_sub(self.cursor)
    }


    /// A shadow buffer of the same geometry as `real`, recorded as its twin.
    pub fn shadow_of(&mut self, real: TensorView) -> TensorView {
        match self.carve(real.shape, real.dtype) {
            Ok(address) => {
                let shadow =
                    TensorView::with_strides(address, real.dtype, real.shape, real.strides);
                self.pairs.push(Twin { shadow, real });
                shadow
            }
            Err(error) => {
                self.fail(error);
                real
            }
        }
    }

    /// A shadow buffer of an operand this operator modifies in place, recorded as its twin, with the
    /// operand's current bytes copied in first.
    ///
    /// The copy is asynchronous on `stream`: every launch after it on that stream sees the bytes, and
    /// a capture records the copy as a node like any other.
    pub fn in_place(&mut self, real: TensorView, stream: &Stream) -> TensorView {
        match self.carve(real.shape, real.dtype) {
            Ok(address) => {
                match real.byte_len() {
                    Ok(bytes) => {
                        if let Err(error) = platform::copy_bytes(stream, address, real.ptr, bytes) {
                            self.fail(error);
                            return real;
                        }
                    }
                    Err(error) => {
                        self.fail(error);
                        return real;
                    }
                }
                let shadow =
                    TensorView::with_strides(address, real.dtype, real.shape, real.strides);
                self.pairs.push(Twin { shadow, real });
                shadow
            }
            Err(error) => {
                self.fail(error);
                real
            }
        }
    }

    /// The pairs of this execution, for the comparison.
    pub fn pairs(&self) -> &[Twin] {
        &self.pairs
    }

    /// Copies the real side of every pair recorded since the last call into this region.
    ///
    /// The comparison runs after the walk, and a real output lives in the bucket's arena, where a
    /// later layer writes over it. So the old kernel's output is copied here the moment it is
    /// written, on `stream`, and the pair compares that copy: both sides then hold what the two
    /// kernels produced from one input, which is the comparison 7.3 asks for.
    pub fn freeze(&mut self, stream: &Stream) {
        for index in self.frozen..self.pairs.len() {
            let real = self.pairs[index].real;
            let copied = self.carve(real.shape, real.dtype).and_then(|address| {
                let bytes = real.byte_len()?;
                platform::copy_bytes(stream, address, real.ptr, bytes)?;
                Ok(address)
            });
            match copied {
                Ok(address) => {
                    self.pairs[index].real =
                        TensorView::with_strides(address, real.dtype, real.shape, real.strides);
                }
                Err(error) => self.fail(error),
            }
        }
        self.frozen = self.pairs.len();
    }

    /// The first carving that did not fit, if any; the harness fails the precopy with it.
    pub fn failure(&self) -> Option<&Error> {
        self.failure.as_ref()
    }

    /// Start the next execution, handing every buffer out again.
    pub fn reset(&mut self) {
        self.cursor = 0;
        self.pairs.clear();
        self.frozen = 0;
        self.tolerance = None;
        self.failure = None;
    }

    /// Record how far the shadowed launch's real and shadow results may differ.
    ///
    /// The operator's own type is the only place its tolerance is known (`Op::tolerance`), so the
    /// layer that shadows the launch records it here and the harness that compares the twins reads
    /// it back: the comparison then never needs the operator's type.
    pub fn set_tolerance(&mut self, tolerance: Tolerance) {
        self.tolerance = Some(tolerance);
    }

    /// The tolerance of the launch whose twins this region holds, if one was recorded.
    pub fn tolerance(&self) -> Option<Tolerance> {
        self.tolerance
    }

    /// A buffer of `shape` elements of `dtype`, from the region.
    fn carve(&mut self, shape: Shape, dtype: DType) -> Result<DevicePtr> {
        let bytes = dtype.bytes().ok_or_else(|| {
            Error::Shape(format!(
                "{dtype} elements have no whole-byte size; a twin cannot address them"
            ))
        })?;
        let bytes = shape.numel() * bytes;
        let start = self.cursor.div_ceil(ALIGNMENT) * ALIGNMENT;
        let end = start.checked_add(bytes).ok_or_else(|| Error::Twins {
            operator: "twins",
            needed: bytes,
            available: self.available(),
        })?;
        if end > self.capacity() {
            return Err(Error::Twins {
                operator: "twins",
                needed: end - self.cursor.min(end),
                available: self.available(),
            });
        }
        self.cursor = end;
        Ok(platform::offset(self.base(), start))
    }

    fn fail(&mut self, error: Error) {
        if self.failure.is_none() {
            self.failure = Some(error);
        }
    }
}

impl std::fmt::Debug for Twins {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Twins")
            .field("capacity", &self.capacity())
            .field("used", &self.cursor)
            .field("pairs", &self.pairs.len())
            .field("failure", &self.failure)
            .finish()
    }
}
