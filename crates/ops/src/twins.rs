//! Shadow buffers for shadow execution.
//!
//! During a generation update, the new kernel of an operator runs beside the old one on the same
//! inputs: the old kernel writes the real outputs, the new one writes twins of them, and the two are
//! compared within the operator's [`Tolerance`]. Because the
//! new kernel's result is never fed on, the difference of one operator does not accumulate through
//! the rest of the forward.
//!
//! A [`Twins`] is one region of device memory from which [`Op::shadow`](crate::Op::shadow) carves the
//! twins of one paired call. It comes from the worker's scratch pool, so shadow buffers never
//! touch the resident memory of the serving model and never enter the graph pool. The region holds
//! one call's twins at a time: the caller compares a call's pairs before the next paired call
//! carves the region again ([`Twins::reset`]), while the real outputs are still where the serving
//! kernel wrote them.
//!
//! ## Why the carving does not return a result
//!
//! `Op::shadow` returns the operator's arguments, not a result: shadow execution happens inside an
//! operator, which cannot fail the caller. So a carving that cannot fit records the failure in the
//! region and hands back the real view; the paired launch then launches no new kernel
//! ([`Call::launch_redirected`](crate::Call::launch_redirected)) and the harness reads
//! [`Twins::failure`] after the call. A region that is too small therefore cannot turn into a
//! comparison of a buffer against itself, nor into a second write of an operand updated in place.
//!
//! ## Counting
//!
//! A region made by [`Twins::counting`] holds no memory: it carves addresses from zero and states
//! how many bytes a call's shadow takes ([`Twins::used`]). A profile run counts every launch this
//! way, which is what sizes the region a paired call is carved from.

use platform::{Allocator, Buffer, DevicePtr, PoolKind, Stream};

use crate::abi;
use crate::allocator::ALIGNMENT;
use crate::operand::Operand;
use crate::{Error, Result};

/// One shadow buffer and the real buffer it shadows.
#[derive(Clone, Debug)]
pub struct Twin {
    /// Where the new implementation writes.
    pub shadow: Operand,
    /// Where the old implementation writes, and what the comparison reads as the reference.
    pub real: Operand,
}

/// The shadow buffers of one paired call.
///
/// The region is the worker's scratch pool, never the graph pool: a paired call writes resident
/// memory not at all, and `Twins` only carves views of the bytes the worker gave it.
pub struct Twins {
    /// The region's memory; `None` for a region that only counts ([`Twins::counting`]).
    buffer: Option<Buffer>,
    /// The bytes the region holds.
    capacity: u64,
    /// The first byte not handed out in this call.
    cursor: u64,
    pairs: Vec<Twin>,
    /// `(twin, real, bytes)` of each operand the call modifies in place, whose current bytes are
    /// copied into its twin before the new kernel runs ([`Twins::copy_in`]).
    copies: Vec<(DevicePtr, DevicePtr, u64)>,
    failure: Option<Error>,
}

impl Twins {
    /// A twin region of `bytes` from `allocator`'s scratch pool.
    ///
    /// The size is the most one paired call of the runner's forwards carves; a call whose shadow
    /// needs more records the failure in the region rather than allocating during a step. The
    /// allocation is the worker's, made once per runner and owned by the `Twins`.
    pub fn new(allocator: &mut Allocator, bytes: u64) -> Result<Self> {
        let bytes = usize::try_from(bytes.max(ALIGNMENT))
            .map_err(|_| Error::Shape("a twin region larger than usize::MAX".to_owned()))?;
        let buffer = allocator.allocate(PoolKind::Scratch, bytes)?;
        Ok(Self {
            capacity: buffer.len() as u64,
            buffer: Some(buffer),
            cursor: 0,
            pairs: Vec::new(),
            copies: Vec::new(),
            failure: None,
        })
    }

    /// A region that holds no memory and every carving: what a call's shadow takes is read from
    /// [`Twins::used`]. Its views address nothing and are never launched.
    pub fn counting() -> Self {
        Self {
            buffer: None,
            capacity: u64::MAX,
            cursor: 0,
            pairs: Vec::new(),
            copies: Vec::new(),
            failure: None,
        }
    }

    /// The address of the region's first byte; zero for a counting region.
    fn base(&self) -> u64 {
        self.buffer.as_ref().map_or(0, Buffer::address)
    }

    /// The number of bytes this region holds.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// The bytes not yet handed out in this call.
    pub fn available(&self) -> u64 {
        self.capacity.saturating_sub(self.cursor)
    }

    /// The bytes handed out in this call, each twin from an [`ALIGNMENT`] boundary.
    pub fn used(&self) -> u64 {
        self.cursor
    }

    /// A shadow buffer of the same geometry as `real`, recorded as its twin.
    pub fn shadow_of(&mut self, real: Operand) -> Operand {
        // The buffer carved below holds the operand's elements back to back. A strided operand
        // reaches further than that, so a kernel writing through its strides would write past
        // the buffer: such an output is shadowed through the storage it reaches
        // (`strided::shadow_window`), and one that arrives here is refused by name.
        if !crate::strided::is_contiguous(&real) {
            self.fail(Error::Shape(format!(
                "a strided output of {:?} elements has no shadow buffer of its own extent: its \
                 operator shadows it through the storage it reaches",
                real.dims()
            )));
            return real;
        }

        match self.carve(&real) {
            Ok(address) => {
                let shadow = real.at(address);
                self.pairs.push(Twin {
                    shadow: shadow.clone(),
                    real,
                });
                shadow
            }
            Err(error) => {
                self.fail(error);
                real
            }
        }
    }

    /// A shadow buffer of an operand this operator modifies in place, recorded as its twin. The
    /// operand's current bytes are copied into it by [`Twins::copy_in`], before the new kernel
    /// launches and after every launch before the call.
    pub fn in_place(&mut self, real: Operand) -> Operand {
        let copied = real
            .byte_len()
            .and_then(|bytes| Ok((self.carve(&real)?, bytes)));

        match copied {
            Ok((address, bytes)) => {
                self.copies.push((address, real.ptr, bytes));
                let shadow = real.at(address);
                self.pairs.push(Twin {
                    shadow: shadow.clone(),
                    real,
                });
                shadow
            }
            Err(error) => {
                self.fail(error);
                real
            }
        }
    }

    /// Issues on `stream` the copies of the in-place operands carved since the last call into
    /// their twins, so the new kernel reads the bytes the serving kernel will.
    ///
    /// # Errors
    ///
    /// For a counting region, which has no memory to copy into, or when the driver refuses a
    /// copy.
    pub fn copy_in(&mut self, stream: &Stream) -> Result<()> {
        if self.buffer.is_none() && !self.copies.is_empty() {
            return Err(Error::Shape(
                "a counting twin region holds no memory to copy an operand into".to_owned(),
            ));
        }

        for (twin, real, bytes) in std::mem::take(&mut self.copies) {
            copy_bytes(stream, twin, real, bytes)?;
        }
        Ok(())
    }

    /// The pairs of this call, for the comparison.
    pub fn pairs(&self) -> &[Twin] {
        &self.pairs
    }

    /// The first carving that did not fit, if any; the harness reports the call with it.
    pub fn failure(&self) -> Option<&Error> {
        self.failure.as_ref()
    }

    /// Start the next call, handing every buffer out again.
    ///
    /// The caller resets once the device has read the pairs of the call before: the next call's
    /// launches, issued behind those reads on one stream, write the same bytes.
    pub fn reset(&mut self) {
        self.cursor = 0;
        self.pairs.clear();
        self.copies.clear();
        self.failure = None;
    }

    /// A buffer of as many elements as `real` names, of its element type, from the region.
    fn carve(&mut self, real: &Operand) -> Result<DevicePtr> {
        let bytes = abi::bytes(real.dtype).ok_or_else(|| {
            Error::Shape(format!(
                "{} elements have no whole-byte size; a twin cannot address them",
                abi::name(real.dtype)
            ))
        })?;
        let bytes = real.numel() * bytes;
        let start = self.cursor.div_ceil(ALIGNMENT) * ALIGNMENT;
        let end = start.checked_add(bytes).ok_or_else(|| Error::Twins {
            operator: "twins",
            needed: bytes,
            available: self.available(),
        })?;
        if end > self.capacity {
            return Err(Error::Twins {
                operator: "twins",
                needed: end - self.cursor.min(end),
                available: self.available(),
            });
        }
        self.cursor = end;
        Ok(DevicePtr::from_address(self.base() + start))
    }

    fn fail(&mut self, error: Error) {
        if self.failure.is_none() {
            self.failure = Some(error);
        }
    }
}

/// Copies `bytes` device bytes from `source` to `destination` on `stream`.
fn copy_bytes(
    stream: &Stream,
    destination: DevicePtr,
    source: DevicePtr,
    bytes: u64,
) -> Result<()> {
    let bytes = usize::try_from(bytes)
        .map_err(|_| Error::Shape(format!("a {bytes}-byte copy is larger than usize::MAX")))?;
    // SAFETY: both ranges address device memory the caller keeps alive until the copy completes
    // and until the stream work after it finishes.
    unsafe { platform::memcpy_async(destination.address(), source.address(), bytes, stream) }?;
    Ok(())
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

#[cfg(test)]
mod tests {
    use platform::DevicePtr;
    use tensor::DType;

    use super::{ALIGNMENT, Twins};
    use crate::Operand;

    /// A counting region states the bytes a call's shadow carves: each twin from a 256-byte
    /// boundary, an output and an operand written in place alike.
    #[test]
    fn a_counting_region_states_the_aligned_bytes_of_a_calls_twins() {
        let mut twins = Twins::counting();
        let output = Operand::new(DevicePtr::from_address(0x1000), DType::BF16, &[3, 50]);
        let state = Operand::new(DevicePtr::from_address(0x8000), DType::I32, &[129]);

        twins.shadow_of(output);
        assert_eq!(twins.used(), 300);
        twins.in_place(state);
        assert_eq!(twins.used(), ALIGNMENT * 2 + 129 * 4);
        assert_eq!(twins.pairs().len(), 2);
        assert!(twins.failure().is_none());

        // The next call carves from the start again.
        twins.reset();
        assert_eq!(twins.used(), 0);
        assert!(twins.pairs().is_empty());
    }
}
