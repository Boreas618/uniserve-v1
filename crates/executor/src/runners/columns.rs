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
    /// The pinned host side, and the allocation whose address `device` names.
    pub(crate) host: platform::PinnedBuffer,
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
        Ok(Self { device, host })
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
        Ok(())
    }

    /// Copies the host side into the device column; the caller's graph reads it afterwards.
    pub(crate) fn upload(&self, stream: &platform::Stream) -> Result<()> {
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
