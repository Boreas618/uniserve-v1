//! The page-locked regions a runner's forwards upload their host values from.
//!
//! A model computes its host-side inputs (positions, packing indices, schedule coefficients) inside
//! its forward and places them on the device with `nn::Forward::upload`, which writes the values
//! into a staging region and, in a launching forward, issues a copy from it. A graph captured from
//! that forward keeps the copies and reads the same addresses on every replay.
//!
//! A runner therefore holds two kinds of region, all of one size (the extent its recorded walks'
//! uploads reached):
//!
//! - **The captured region**, which every graph of the runner was captured over and whose
//!   addresses its copy nodes read. Only a capture writes it from this thread, and before each
//!   replay a host function on the stream fills it from a slot ([`StagingRegion::publish`]): the
//!   driver runs the function after the previous replay completed and before this one starts, so
//!   the region a graph is reading is never overwritten, and the executor never waits for it.
//! - **A ring of slots**, which an eager forward uploads from directly and a staging walk
//!   (`nn::Forward::staging`) writes a replay's values into. A slot is reused once the stream has
//!   consumed it, which an event recorded after its use states; the executor waits only when the
//!   ring wraps onto a slot the stream has not reached yet.
//!
//! A runner whose forwards upload nothing holds no memory, and its replays run no staging walk.

use std::cell::{Cell, RefCell};

use platform::{DeviceId, Event, EventFlags, PinnedBuffer, Stream};

use crate::{Error, Result};

/// The slots a runner rotates through: a replay's values and an eager forward's are written ahead
/// of the stream by at most this many uses before the executor waits.
const SLOTS: usize = 2;

/// One page-locked region and the staging view that writes through it.
struct Region {
    /// The memory `staging` addresses, held so it lives exactly as long as that address.
    buffer: PinnedBuffer,
    staging: nn::forward::Staging,
    /// Recorded on the stream after the region's last use was issued; a slot is reused once it
    /// completed. Unrecorded for the captured region, which the stream order protects instead.
    consumed: Event,
}

impl Region {
    fn new(device: DeviceId, bytes: u64) -> Result<Self> {
        let length = usize::try_from(bytes)
            .map_err(|_| Error::Invalid(format!("a {bytes}-byte staging region exceeds usize")))?;
        let buffer = PinnedBuffer::new(device, length)?;
        // SAFETY: the buffer is page-locked, holds `bytes` bytes, and lives as long as this region,
        // which lives as long as the runner whose graphs and forwards read it; only forwards over
        // this region and the publishing host function write it.
        let staging = unsafe { nn::forward::Staging::new(buffer.as_ptr(), bytes) };
        Ok(Self {
            buffer,
            staging,
            consumed: Event::new(EventFlags::default()),
        })
    }
}

/// The captured region and the ring of slots of a runner that uploads.
struct Regions {
    captured: Region,
    slots: Vec<Region>,
    /// The slot the next use takes.
    next: Cell<usize>,
    /// The slot the current use writes, which [`StagingRegion::release`] or
    /// [`StagingRegion::publish`] ends.
    current: RefCell<Option<usize>>,
}

/// One runner's staging regions: the captured region and the ring of slots.
pub struct StagingRegion {
    /// `None` when the runner's forwards upload nothing.
    regions: Option<Regions>,
    /// Stands for the regions of a runner that uploads nothing: forwards over it only count, and
    /// a launching forward that uploads through it is refused (`nn::Forward::upload`).
    counting: nn::forward::Staging,
}

impl StagingRegion {
    /// The regions of a runner whose forwards upload nothing, until its recorded walk sizes them.
    pub fn empty() -> Self {
        Self {
            regions: None,
            counting: nn::forward::Staging::counting(),
        }
    }

    /// Regions of `bytes` page-locked bytes each on `device`, the extent the runner's recorded
    /// walks reached ([`crate::runners::recording::Footprint::staging`]).
    ///
    /// # Errors
    ///
    /// When a page-locked allocation fails.
    pub fn new(device: DeviceId, bytes: u64) -> Result<Self> {
        if bytes == 0 {
            return Ok(Self::empty());
        }
        let slots = (0..SLOTS)
            .map(|_| Region::new(device, bytes))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            regions: Some(Regions {
                captured: Region::new(device, bytes)?,
                slots,
                next: Cell::new(0),
                current: RefCell::new(None),
            }),
            counting: nn::forward::Staging::counting(),
        })
    }

    /// Whether any forward of the runner uploads values, so a replay needs a staging walk.
    pub fn is_used(&self) -> bool {
        self.regions.is_some()
    }

    /// The region a capture walk uploads from, reset: its addresses are the ones the captured
    /// copies read on every replay.
    ///
    /// The stream is drained first, so no publishing host function of an earlier replay writes
    /// the region while the capture's eager walk does.
    ///
    /// # Errors
    ///
    /// When the drain fails.
    pub fn captured(&self, stream: &Stream) -> Result<&nn::forward::Staging> {
        let Some(regions) = &self.regions else {
            return Ok(&self.counting);
        };
        if stream.capture_status()? == platform::CaptureStatus::None {
            stream.synchronize()?;
        }
        regions.captured.staging.reset();
        Ok(&regions.captured.staging)
    }

    /// The next slot, reset, once the stream has consumed its previous use: what an eager forward
    /// uploads from, or what a staging walk writes a replay's values into.
    ///
    /// The caller ends the use with [`Self::release`] (eager) or [`Self::publish`] (replay).
    ///
    /// # Errors
    ///
    /// When waiting on the slot's previous use fails.
    pub fn slot(&self) -> Result<&nn::forward::Staging> {
        let Some(regions) = &self.regions else {
            return Ok(&self.counting);
        };
        let index = regions.next.get();
        regions.next.set((index + 1) % regions.slots.len());
        let slot = &regions.slots[index];
        slot.consumed.synchronize()?;
        slot.staging.reset();
        *regions.current.borrow_mut() = Some(index);
        Ok(&slot.staging)
    }

    /// Ends an eager use of the current slot: it is consumed once `stream` reaches this point.
    ///
    /// # Errors
    ///
    /// When the record cannot be made.
    pub fn release(&self, stream: &Stream) -> Result<()> {
        let Some(regions) = &self.regions else {
            return Ok(());
        };
        if let Some(index) = regions.current.borrow_mut().take() {
            regions.slots[index].consumed.record(stream)?;
        }
        Ok(())
    }

    /// Ends a staging walk: a host function on `stream` copies the current slot into the captured
    /// region once the stream's earlier work (the previous replay included) has completed, and the
    /// slot is consumed after it. The replay launched next reads this walk's values.
    ///
    /// # Errors
    ///
    /// When no slot is in use, or the host function or the record cannot be issued.
    pub fn publish(&self, stream: &Stream) -> Result<()> {
        let Some(regions) = &self.regions else {
            return Ok(());
        };
        let index = regions.current.borrow_mut().take().ok_or_else(|| {
            Error::Invariant("a staging walk was published with no slot in use".into())
        })?;
        let slot = &regions.slots[index];
        let source = slot.buffer.address();
        let destination = regions.captured.buffer.address();
        let bytes = slot.buffer.len();
        stream.launch_host_fn(move || {
            // SAFETY: both regions are page-locked buffers of `bytes` bytes owned by the runner,
            // which outlives every function queued on its stream (a runner's stream is drained
            // before the runner is dropped). The driver runs this after the stream's earlier work,
            // so no graph reads the captured region now, and the slot is not reused before the
            // event recorded after this function has completed.
            unsafe {
                std::ptr::copy_nonoverlapping(source as *const u8, destination as *mut u8, bytes);
            }
        })?;
        slot.consumed.record(stream)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runners::lifecycle::walk_capture;
    use ops::Shape;
    use platform::DeviceAllocation;

    const DEVICE: DeviceId = DeviceId(0);
    const WORDS: usize = 8;

    /// The words a forward over `staging` uploads into the arena's first tensor.
    fn upload(forward: &nn::Forward<'_>, value: i32) -> crate::Result<ops::Operand> {
        Ok(forward.upload(&[value; WORDS], Shape::new(&[WORDS as u64]))?)
    }

    /// Copies `view`'s words to the host behind the stream's earlier work.
    fn read_back(view: ops::Operand, host: &PinnedBuffer, stream: &Stream) {
        // SAFETY: both ranges hold `WORDS` i32 and outlive the stream's drain below.
        unsafe {
            platform::memcpy_async(host.address(), view.ptr.address(), WORDS * 4, stream)
                .expect("a read-back copy");
        }
    }

    /// A graph captured over the captured region reads, on each replay, the values the staging walk
    /// before that replay wrote into a slot - even when two replays are queued back to back and the
    /// executor never waits between them: the second walk's values reach the captured region only
    /// after the first replay has read it.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn each_replay_reads_the_values_its_own_staging_walk_wrote() {
        let stream = Stream::new(DEVICE).expect("a stream");
        let memory = DeviceAllocation::linear(DEVICE, 4096).expect("an arena region");
        let arena = nn::Arena::new(ops::DevicePtr::new(memory.address()), 4096);
        let kernels = ops::Generation::new(0, []);
        let regions = StagingRegion::new(DEVICE, (WORDS * 4) as u64).expect("staging regions");

        // The capture: one upload, which becomes a copy node reading the captured region.
        let captured = regions.captured(&stream).expect("the captured region");
        let mut uploaded = None;
        let walked = walk_capture(&stream, || {
            arena.reset();
            let forward = nn::Forward::new(&stream, &kernels, &arena).with_staging(captured);
            uploaded = Some(upload(&forward, -1)?);
            Ok(())
        })
        .expect("a capture of one upload");
        let graph = crate::captured::Captured::new(walked.template, walked.nodes, Vec::new())
            .expect("an executable");
        let view = uploaded.expect("the upload's tensor");

        // Two replays queued back to back, each read back behind itself.
        let first = PinnedBuffer::new(DEVICE, WORDS * 4).expect("a read-back buffer");
        let second = PinnedBuffer::new(DEVICE, WORDS * 4).expect("a read-back buffer");
        for (value, host) in [(7, &first), (9, &second)] {
            let slot = regions.slot().expect("a slot");
            arena.reset();
            let staged =
                upload(&nn::Forward::staging(&arena, slot), value).expect("a staging walk");
            assert_eq!(
                staged.ptr, view.ptr,
                "a staging walk reserves what the capture did"
            );
            regions
                .publish(&stream)
                .expect("the publishing host function");
            graph.launch(&stream).expect("a replay");
            read_back(view, host, &stream);
        }
        stream.synchronize().expect("the replays complete");

        let words = |host: &PinnedBuffer| -> Vec<i32> {
            // SAFETY: the stream drained, so no copy writes the buffer.
            let bytes = unsafe { host.as_slice() };
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| i32::from_ne_bytes(*word))
                .collect()
        };
        assert_eq!(words(&first), vec![7; WORDS]);
        assert_eq!(words(&second), vec![9; WORDS]);
    }
}
