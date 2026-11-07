//! The columns a runner stages a batch into: a host buffer whose device address never moves, and
//! the encoders that turn a mechanism's values into its bytes.
//!
//! Every runner stages into these, whatever its class: a column is written on the host, uploaded
//! once on the runner's stream, and read by the captured graph through a view of the device buffer's
//! address. Nothing here names a token or a cache.

use ops::{DType, TensorView};
use platform::DeviceId;

use crate::{Error, Result};

/// One host column of a batch, uploaded into a device buffer whose address never moves.
///
/// A captured graph addresses these buffers directly, so they are allocated once when the runner
/// binds, sized for the largest bucket, and every step rewrites their contents instead of
/// allocating: what 4.5 forbids is an allocation inside a step, not a copy into a fixed buffer.
pub(crate) struct Column {
    /// The device buffer's view: what a captured graph addresses.
    pub(crate) device: TensorView,
    /// The pinned host side, and the allocation whose address `device` names until the column is
    /// [placed](Column::place).
    pub(crate) host: platform::PinnedBuffer,
    /// The device storage of a placed column.
    placed: Option<Placed>,
}

/// What a column keeps once it has device storage of its own.
///
/// A graph reading host memory reads it while it runs, so a batch staged behind a running graph
/// would rewrite what that graph reads. A placed column takes its values by a copy on the stream
/// instead, and the two host halves take turns: the next batch writes one while the copy of the
/// other waits behind the graph in flight.
struct Placed {
    /// Kept for the addresses a captured graph holds.
    _storage: platform::Buffer,
    /// The host half the next write does not touch.
    other: platform::PinnedBuffer,
    /// How many bytes of `host` the next [`Column::push`] carries.
    written: usize,
}

impl Column {
    /// A column of `numel` elements of `dtype` on `device`.
    pub(crate) fn new(device: DeviceId, dtype: DType, numel: u64) -> Result<Self> {
        let bytes = dtype
            .bytes()
            .ok_or_else(|| Error::Invalid(format!("{dtype} has no whole-byte size")))?
            .checked_mul(numel)
            .ok_or_else(|| Error::Invalid("a batch column's extent overflows".into()))?;
        let host = platform::PinnedBuffer::new(device, bytes as usize)?;
        let device = TensorView::new(
            ops::DevicePtr::new(host.address()),
            dtype,
            ops::Shape::new(&[numel]),
        );
        Ok(Self {
            device,
            host,
            placed: None,
        })
    }

    /// Moves the column to device storage from `pools`, with the values its host side holds.
    ///
    /// The graphs captured afterwards address the storage, so this runs before the first capture.
    pub(crate) fn place(
        &mut self,
        device: DeviceId,
        pools: &mut crate::arena::Pools,
        stream: &platform::Stream,
    ) -> Result<()> {
        if self.placed.is_some() {
            return Ok(());
        }
        let bytes = self.host.len();
        let storage = pools.allocate(platform::PoolKind::Resident, bytes as u64)?;
        // SAFETY: the destination is `bytes` of storage just carved for this column, the source its
        // own host side of the same length, and the caller waits before either is touched again.
        unsafe {
            platform::memcpy_async(storage.address(), self.host.address(), bytes, stream)?;
        }
        self.device = TensorView::new(
            ops::DevicePtr::new(storage.address()),
            self.device.dtype,
            self.device.shape,
        );
        self.placed = Some(Placed {
            _storage: storage,
            other: platform::PinnedBuffer::new(device, bytes)?,
            written: 0,
        });
        Ok(())
    }

    /// Hands what was written since the last push to the device, on `stream`.
    ///
    /// Only a placed column has anything to carry; the graph reads any other from the host.
    pub(crate) fn push(&mut self, stream: &platform::Stream) -> Result<()> {
        let Some(placed) = self.placed.as_mut() else {
            return Ok(());
        };
        if placed.written == 0 {
            return Ok(());
        }
        // SAFETY: the destination is this column's device storage and the source its pinned host
        // half; neither is shorter than `written`, and the other half takes the next write.
        unsafe {
            platform::memcpy_async(
                self.device.ptr.address(),
                self.host.address(),
                placed.written,
                stream,
            )?;
        }
        placed.written = 0;
        std::mem::swap(&mut self.host, &mut placed.other);
        Ok(())
    }

    /// Writes `bytes` into the host side; the device copy follows in [`Column::upload`].
    pub(crate) fn write(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() > self.host.len() {
            return Err(Error::Invariant(format!(
                "a column of {} bytes cannot hold {}",
                self.host.len(),
                bytes.len()
            )));
        }
        // SAFETY: the caller wrote `bytes.len()` bytes of one element type into a host buffer of at
        // least that size, allocated for exactly this purpose; the buffer is not read concurrently.
        unsafe {
            self.host.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        }
        if let Some(placed) = self.placed.as_mut() {
            placed.written = bytes.len();
        }
        Ok(())
    }

    /// Copies the host side into the device column; the caller's graph reads it afterwards.
    ///
    /// A column whose device view is its own host storage has nothing to copy, and a copy of pinned
    /// host memory onto itself is the driver's host copy, which waits for the stream: it held every
    /// step's launch behind the step before it.
    pub(crate) fn upload(&self, stream: &platform::Stream) -> Result<()> {
        if self.device.ptr.address() == self.host.address() {
            return Ok(());
        }
        // SAFETY: source and destination are this column's own host and device buffers, of the
        // same length, and the copy is stream-ordered before the graph that reads the device one.
        unsafe {
            platform::memcpy_async(
                self.device.ptr.address(),
                self.host.address(),
                self.host.len(),
                stream,
            )?;
        }
        Ok(())
    }

    /// The first `elements` of the column, as a view the batch reads.
    pub(crate) fn extent(&self, elements: std::ops::Range<u64>) -> TensorView {
        TensorView::new(
            self.device.ptr,
            self.device.dtype,
            ops::Shape::new(&[elements.end - elements.start]),
        )
    }

    /// The column's head, in `shape`, as a view the batch reads.
    ///
    /// A batch's own state is not a flat extent: the sampler's parameters are
    /// `[rows, PARAMETER_COLUMNS]`, and a bucket narrower than the largest reads its rows from the
    /// column's head. Both views name the same base pointer, which is what a captured graph
    /// addresses, so the two differ in shape alone.
    pub(crate) fn shaped(&self, shape: &[u64]) -> TensorView {
        debug_assert!(
            shape.iter().product::<u64>() <= self.device.shape.dim(0),
            "a shaped view of {} elements does not fit a column of {}",
            shape.iter().product::<u64>(),
            self.device.shape.dim(0)
        );
        TensorView::new(self.device.ptr, self.device.dtype, ops::Shape::new(shape))
    }
}

/// A region a graph reads and writes on the device alone, whose address never moves.
///
/// A [`Column`] is the host's way in: its device view is its pinned buffer, so every read and write
/// of it crosses to host memory. That is right for what the host stages, and wrong for what a
/// forward works in: a residual stream read and written by every layer ran thirty times slower
/// there than in device memory.
pub(crate) struct Region {
    /// What a captured graph addresses.
    pub(crate) device: TensorView,
    /// The allocation `device` names, held as long as a graph may address it.
    _storage: platform::DeviceAllocation,
}

impl Region {
    /// A region of `numel` elements of `dtype` on `device`.
    pub(crate) fn new(device: DeviceId, dtype: DType, numel: u64) -> Result<Self> {
        let bytes = dtype
            .bytes()
            .ok_or_else(|| Error::Invalid(format!("{dtype} has no whole-byte size")))?
            .checked_mul(numel)
            .ok_or_else(|| Error::Invalid("a region's extent overflows".into()))?;
        let bytes = usize::try_from(bytes)
            .map_err(|_| Error::Invalid("a region is larger than memory".into()))?;
        let storage = platform::DeviceAllocation::linear(device, bytes)?;
        Ok(Self {
            device: TensorView::new(
                ops::DevicePtr::new(storage.address()),
                dtype,
                ops::Shape::new(&[numel]),
            ),
            _storage: storage,
        })
    }

    /// The first `elements` of the region, as a view the batch reads.
    pub(crate) fn extent(&self, elements: std::ops::Range<u64>) -> TensorView {
        TensorView::new(
            self.device.ptr,
            self.device.dtype,
            ops::Shape::new(&[elements.end - elements.start]),
        )
    }
}

/// `values` as the bytes of the column's element type, in the host's byte order.
pub(crate) fn encode_i32(buffer: &mut Vec<u8>, values: &[i32]) {
    buffer.clear();
    for value in values {
        buffer.extend_from_slice(&value.to_ne_bytes());
    }
}

/// The same for f32 columns.
pub(crate) fn encode_f32(buffer: &mut Vec<u8>, values: &[f32]) {
    buffer.clear();
    for value in values {
        buffer.extend_from_slice(&value.to_ne_bytes());
    }
}
