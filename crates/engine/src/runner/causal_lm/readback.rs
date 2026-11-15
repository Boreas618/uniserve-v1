//! The read of a step's outputs to the host, issued on the runner's stream behind the step.
//!
//! A replay's outputs lie in its graph's memory, which the next replay of the bucket writes
//! again. Each step's outputs are therefore copied to page-locked host memory by one batched copy
//! on the stream, right behind the step's last work and before anything of a later step is
//! issued, and an event is recorded behind the copy. The host reads the step's outputs once that
//! event has completed, whether or not later steps were issued on the stream meanwhile: this is
//! what lets a step land after the next one was launched ([`Runner::start`](crate::Runner::start)).
//!
//! The memory is a set of turns, one per step whose outputs are issued and not yet read. A turn is
//! taken when a step issues its copy and given back when the host has read it; a step that finds
//! every turn taken gets a new one. The turns are therefore as many as the most steps of the
//! runner ever in flight at once, which the scheduler's queue depth bounds, and none is allocated
//! on the step path once the runner has served its largest outputs.

use std::sync::Arc;

use platform::{CopyBatch, DeviceId, Event, EventFlags, PinnedBuffer, SourceOrder, Stream};
use tensor::{DType, Device, Tensor};

use crate::worker::{Error, Result};

/// The alignment of each output in a turn: the widest element type's, and a copy's preferred one.
const ALIGNMENT: usize = 256;

/// The page-locked turns a runner's steps read their outputs into.
pub(crate) struct Readback {
    device: DeviceId,
    turns: Vec<Turn>,
}

/// One turn: its memory, sized for the largest outputs it has held, and the event recorded behind
/// the copy into it.
struct Turn {
    memory: Option<PinnedBuffer>,
    event: Arc<Event>,
    /// Whether a step's outputs are in flight to this turn or not yet read from it.
    taken: bool,
}

/// One step's outputs in flight to the host: the turn that receives them, and the element type,
/// dimensions, offset and bytes of each output, in the step's order.
pub(crate) struct Pending {
    turn: usize,
    outputs: Vec<(DType, Vec<usize>, usize, usize)>,
}

impl Readback {
    /// No turn yet: the first step allocates one.
    pub(crate) fn new(device: DeviceId) -> Self {
        Self {
            device,
            turns: Vec::new(),
        }
    }

    /// Issues the copy of `outputs`, device tensors in contiguous memory, to a free turn on
    /// `stream`, behind everything the stream holds, and records the turn's event behind it.
    ///
    /// # Errors
    ///
    /// [`Error::Invariant`] for an output that is not a contiguous device tensor; the driver's
    /// refusal of the allocation, the copy or the event. A turn is taken only once its copy and
    /// event were issued.
    pub(crate) fn issue(&mut self, outputs: &[Tensor], stream: &Stream) -> Result<Pending> {
        // Where each output lies in the turn.
        let mut layout = Vec::with_capacity(outputs.len());
        let mut sources = Vec::with_capacity(outputs.len());
        let mut bytes = 0;
        for (index, output) in outputs.iter().enumerate() {
            if !output.is_contiguous() {
                return Err(Error::Invariant(format!(
                    "output {index} of the step is not contiguous, and is read back as stored"
                )));
            }
            let size = output
                .dtype()
                .storage_bytes(output.elem_count())
                .ok_or_else(|| {
                    Error::Invariant(format!(
                        "output {index} of the step holds more bytes than the address space"
                    ))
                })?;
            let source = ops::operand(output)?.ptr.address();

            layout.push((output.dtype(), output.dims().to_vec(), bytes, size));
            sources.push(source);
            bytes += size.next_multiple_of(ALIGNMENT);
        }

        let turn = self.free_turn(bytes)?;
        let Turn { memory, event, .. } = &self.turns[turn];
        let base = memory.as_ref().map_or(0, PinnedBuffer::address);

        // The destinations are disjoint ranges of the turn, which no copy in flight writes: the
        // turn is free.
        let mut copies = CopyBatch::new(SourceOrder::Stream);
        for (&(_, _, offset, size), &source) in layout.iter().zip(&sources) {
            if size > 0 {
                copies.push(base + offset as u64, source, size);
            }
        }
        if !copies.is_empty() {
            // SAFETY: each source is an output of the step on the stream's device, read in stream
            // order behind the step; each destination lies inside the turn's page-locked memory,
            // sized above, which is not read before the event recorded behind the copy completed
            // (`Readback::read`) nor written by another copy while the turn is taken.
            unsafe { copies.submit(stream) }?;
        }
        event.record(stream)?;

        self.turns[turn].taken = true;
        Ok(Pending {
            turn,
            outputs: layout,
        })
    }

    /// The event recorded behind the copy of `pending`: complete once the step and the copy of
    /// its outputs have completed on the device.
    pub(crate) fn event(&self, pending: &Pending) -> Arc<Event> {
        Arc::clone(&self.turns[pending.turn].event)
    }

    /// Waits for the copy of `pending` and returns its outputs as host tensors of the outputs'
    /// element types and dimensions, in the step's order. The turn is free again afterwards,
    /// also when the wait fails.
    ///
    /// # Errors
    ///
    /// The device's failure, which the wait reports, or a host tensor that cannot be built.
    pub(crate) fn read(&mut self, pending: Pending) -> Result<Vec<Tensor>> {
        let turn = &mut self.turns[pending.turn];
        let waited = turn.event.synchronize();
        turn.taken = false;
        waited?;

        let base = turn
            .memory
            .as_ref()
            .map_or(std::ptr::null(), |memory| memory.as_ptr());
        pending
            .outputs
            .iter()
            .map(|(dtype, dims, offset, size)| {
                let bytes: &[u8] = if *size == 0 {
                    &[]
                } else {
                    // SAFETY: the range lies inside the turn's memory (`Readback::issue`), and
                    // the copy into it completed: the event recorded behind it was waited for.
                    unsafe { std::slice::from_raw_parts(base.add(*offset), *size) }
                };
                Ok(Tensor::from_raw_buffer(bytes, *dtype, dims, &Device::Cpu)?)
            })
            .collect()
    }

    /// A turn no step holds, with at least `bytes` of memory: the first free one, its memory grown
    /// when it is too small, or a new one.
    fn free_turn(&mut self, bytes: usize) -> Result<usize> {
        let turn = match self.turns.iter().position(|turn| !turn.taken) {
            Some(turn) => turn,
            None => {
                self.turns.push(Turn {
                    memory: None,
                    event: Arc::new(Event::new(EventFlags::default())),
                    taken: false,
                });
                self.turns.len() - 1
            }
        };

        let held = self.turns[turn]
            .memory
            .as_ref()
            .map_or(0, PinnedBuffer::len);
        if bytes > held {
            // A free turn's last copy completed before its outputs were read, so its memory is
            // released with nothing in flight to it.
            self.turns[turn].memory =
                Some(PinnedBuffer::new(self.device, bytes.next_power_of_two())?);
        }
        Ok(turn)
    }
}
