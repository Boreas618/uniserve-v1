//! Pitched copies: layers of rows at a byte pitch in each operand, submitted
//! as one `cuMemcpy3DBatchAsync` call.

use super::{SourceOrder, require_batch_stream};
use crate::driver::{
    CU_MEMCPY_OPERAND_TYPE_POINTER, Handle, MemLocation, Memcpy3DBatchOp, Memcpy3DOperand, driver,
};
use crate::{Error, Result, Stream};

/// One pitched copy of `depth` layers of `height` rows of `width` bytes.
///
/// Row `y` of layer `z` starts at `base + (z * layer + y) * pitch` in each
/// operand: `pitch` is the byte distance between rows and `layer` the number
/// of rows between layer starts. With `height == 1` the pitch is unused, and
/// with `depth == 1` the layer is unused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pitched {
    pub destination: u64,
    pub destination_pitch: usize,
    pub destination_layer: usize,
    pub source: u64,
    pub source_pitch: usize,
    pub source_layer: usize,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

impl Pitched {
    fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 || self.depth == 0 {
            return Err(Error::invalid("pitched copy extents must be positive"));
        }
        if self.height > 1
            && (self.destination_pitch < self.width || self.source_pitch < self.width)
        {
            return Err(Error::invalid(format!(
                "pitched copy rows of {} bytes overlap with pitches {} and {}",
                self.width, self.destination_pitch, self.source_pitch
            )));
        }
        if self.depth > 1
            && (self.destination_layer < self.height || self.source_layer < self.height)
        {
            return Err(Error::invalid(format!(
                "pitched copy layers of {} rows overlap with layer heights {} and {}",
                self.height, self.destination_layer, self.source_layer
            )));
        }
        Ok(())
    }

    fn operand(&self, pointer: u64, pitch: usize, layer: usize) -> Memcpy3DOperand {
        // For pointer operands the driver measures rows in bytes (the element
        // of an untyped pointer is one byte) and layers in rows.
        Memcpy3DOperand {
            kind: CU_MEMCPY_OPERAND_TYPE_POINTER,
            pointer,
            row_length: if self.height > 1 { pitch } else { self.width },
            layer_height: if self.depth > 1 { layer } else { self.height },
            hint: MemLocation::default(),
        }
    }
}

/// Pitched copies submitted with one `cuMemcpy3DBatchAsync` call. The same
/// ordering rules as [`CopyBatch`](super::CopyBatch) apply.
#[derive(Clone, Debug)]
pub struct PitchedBatch {
    copies: Vec<Pitched>,
    order: SourceOrder,
}

impl PitchedBatch {
    pub fn new(order: SourceOrder) -> Self {
        Self {
            copies: Vec::new(),
            order,
        }
    }

    /// Adds one copy after checking that its rows and layers do not overlap.
    pub fn push(&mut self, copy: Pitched) -> Result<()> {
        copy.validate()?;
        self.copies.push(copy);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.copies.len()
    }

    pub fn is_empty(&self) -> bool {
        self.copies.is_empty()
    }

    pub fn clear(&mut self) {
        self.copies.clear();
    }

    pub fn copies(&self) -> &[Pitched] {
        &self.copies
    }

    /// Submits every copy on `stream`; an empty batch submits nothing.
    /// Reads sources and fails like [`CopyBatch::submit`](super::CopyBatch::submit).
    ///
    /// # Safety
    /// Every addressed row stays valid until the stream reaches the copies.
    pub unsafe fn submit(&self, stream: &Stream) -> Result<()> {
        if self.is_empty() {
            return Ok(());
        }
        require_batch_stream(stream, PITCHED_BATCH_CALL)?;
        let _scope = stream.enter()?;

        // SAFETY: forwarded from the caller.
        unsafe { submit_pitched_batch(&self.copies, self.order, stream) }
    }
}

const PITCHED_BATCH_CALL: &str = "cuMemcpy3DBatchAsync_v2";

/// Submits pitched copies with one batched call.
///
/// # Safety
/// As for [`PitchedBatch::submit`]; the stream's context is current.
unsafe fn submit_pitched_batch(
    copies: &[Pitched],
    order: SourceOrder,
    stream: &Stream,
) -> Result<()> {
    let driver = driver()?;
    let copy = driver.memcpy_3d_batch_async.clone()?;

    let mut operations: Vec<Memcpy3DBatchOp> = copies
        .iter()
        .map(|copy| Memcpy3DBatchOp {
            source: copy.operand(copy.source, copy.source_pitch, copy.source_layer),
            destination: copy.operand(
                copy.destination,
                copy.destination_pitch,
                copy.destination_layer,
            ),
            extent: [copy.width, copy.height, copy.depth],
            source_order: order.raw(),
            flags: 0,
        })
        .collect();

    // SAFETY: the driver reads the operation array during the call only; the
    // caller guarantees the addressed rows.
    driver.check(
        unsafe {
            copy(
                operations.len(),
                operations.as_mut_ptr(),
                0,
                stream.handle() as Handle,
            )
        },
        PITCHED_BATCH_CALL,
    )
}
