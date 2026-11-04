//! Work recorded in a batch's order and issued when the batch is submitted.
//!
//! A caller that must order device work behind operations it has not issued
//! yet — a cache export's copies behind the producer's kernels, a page
//! offload behind the batch that writes the pages — records the work into a
//! [`Batch`] and the submission performs it ([`Batch::perform`]) while it
//! issues the batch, so the work lands after everything recorded before it
//! and before everything recorded after it.
//!
//! A recording keeps what each operation needs and nothing else: no thread, no
//! channel, no completion of its own. The contract on the storage it addresses
//! is the caller's, unchanged: an operation's operands stay valid until the
//! batch performed it. A recording is not a job: it carries no result and no
//! completion, and the submission that performs it decides what waits on the
//! batch.

use std::cell::RefCell;
use std::sync::Arc;

use platform::{CopyBatch, Event, Stream, StridedCopy};

use crate::stream::events::{Deferred, EventPool};
use crate::stream::issue::{
    Fill, HostFunction, Issue, RowFill, copy_host, fill_host, fill_host_rows,
};
use crate::{Error, Result};

/// One operation a recording holds. It runs when the batch is performed,
/// with the batch's stream (`None` for host storage).
type Step = Box<dyn FnOnce(Option<&Stream>) -> Result<()>>;

/// A batch's recorded work.
struct Recorded {
    steps: Vec<Step>,
    performed: bool,
}

/// Work recorded in a batch's order.
///
/// A batch belongs to the thread that records into it: the operations it
/// holds address storage that thread owns.
pub struct Batch {
    steps: RefCell<Recorded>,
    stream: Option<Stream>,
    events: Arc<EventPool>,
}

impl Batch {
    /// An empty batch for `stream` (`None` for host storage, whose work is
    /// plain storage writes in the recording's order) that defers through
    /// `events`, the pool of the rank the batch serves.
    pub fn new(stream: Option<&Stream>, events: Arc<EventPool>) -> Self {
        Self {
            steps: RefCell::new(Recorded {
                steps: Vec::new(),
                performed: false,
            }),
            stream: stream.cloned(),
            events,
        }
    }

    /// The stream the recorded work is issued on.
    pub fn stream(&self) -> Option<&Stream> {
        self.stream.as_ref()
    }

    /// Record work from this point on.
    pub fn on(&self) -> Recording<'_> {
        Recording { batch: self }
    }

    /// Whether nothing was recorded.
    pub fn is_empty(&self) -> bool {
        self.steps.borrow().steps.is_empty()
    }

    /// Issue everything recorded so far, in order.
    ///
    /// # Errors
    ///
    /// Fails when an operation fails, which stops the batch: the operations
    /// after it are not performed. A batch is performed at most once.
    pub fn perform(&self) -> Result<()> {
        let steps = {
            let mut recorded = self.steps.borrow_mut();
            if recorded.performed {
                return Err(Error::Invariant("a batch is performed once".into()));
            }
            recorded.performed = true;
            std::mem::take(&mut recorded.steps)
        };
        for step in steps {
            step(self.stream.as_ref())?;
        }
        Ok(())
    }

    fn push(&self, step: Step) {
        self.steps.borrow_mut().steps.push(step);
    }
}

/// The recorder of one [`Batch`]: an [`Issue`] whose work is issued where the
/// batch is performed.
#[derive(Clone, Copy)]
pub struct Recording<'a> {
    batch: &'a Batch,
}

/// The host writes of a recorder, in the recording's order.
#[derive(Clone, Copy)]
pub struct RecordingHost<'a> {
    batch: &'a Batch,
}

impl<'a> Issue for Recording<'a> {
    type Host = RecordingHost<'a>;

    fn stream(&self) -> Option<&Stream> {
        self.batch.stream()
    }

    fn host(&self) -> Self::Host {
        RecordingHost { batch: self.batch }
    }

    unsafe fn copy(&self, copies: &CopyBatch) -> Result<()> {
        // The batch holds the copies it was given: their operands' contract
        // is the caller's, unchanged.
        let copies = copies.clone();
        self.batch.push(Box::new(move |stream| match stream {
            Some(stream) => {
                // SAFETY: forwarded from the caller.
                unsafe { stream.copy(&copies) }
            }
            None => {
                // SAFETY: forwarded from the caller.
                unsafe { copy_host(&copies) };
                Ok(())
            }
        }));
        Ok(())
    }

    unsafe fn copy_strided(&self, copies: &StridedCopy) -> Result<()> {
        let copies = copies.clone();
        self.batch.push(Box::new(move |stream| {
            let Some(stream) = stream else {
                return Err(Error::Invariant(
                    "a strided copy is issued on a stream".into(),
                ));
            };
            // SAFETY: forwarded from the caller.
            unsafe { stream.copy_strided(&copies) }
        }));
        Ok(())
    }

    unsafe fn fill(&self, fill: Fill) -> Result<()> {
        self.batch.push(Box::new(move |stream| match stream {
            // SAFETY: forwarded from the caller.
            Some(stream) => unsafe { stream.fill(fill) },
            // SAFETY: forwarded from the caller.
            None => {
                unsafe { fill_host(fill) };
                Ok(())
            }
        }));
        Ok(())
    }

    unsafe fn memset_2d(&self, fill: RowFill) -> Result<()> {
        self.batch.push(Box::new(move |stream| match stream {
            // SAFETY: forwarded from the caller.
            Some(stream) => unsafe { stream.memset_2d(fill) },
            // SAFETY: forwarded from the caller.
            None => {
                unsafe { fill_host_rows(fill) };
                Ok(())
            }
        }));
        Ok(())
    }

    fn record(&self, event: &Arc<Event>) -> Result<()> {
        let event = Arc::clone(event);
        self.batch.push(Box::new(move |stream| {
            let Some(stream) = stream else {
                return Err(Error::Invariant("an event is recorded on a stream".into()));
            };
            stream.record(&event)
        }));
        Ok(())
    }

    fn wait(&self, event: &Arc<Event>) -> Result<()> {
        let event = Arc::clone(event);
        self.batch.push(Box::new(move |stream| {
            let Some(stream) = stream else {
                return Err(Error::Invariant(
                    "an event is waited for on a stream".into(),
                ));
            };
            stream.wait(&event)
        }));
        Ok(())
    }

    fn defer(&self, events: &EventPool, fences: Vec<Arc<Event>>, owner: Deferred) -> Result<()> {
        if !std::ptr::eq(events, self.batch.events.as_ref()) {
            return Err(Error::Invariant(
                "a batch defers through the pool it was made for".into(),
            ));
        }
        let pool = Arc::clone(&self.batch.events);
        self.batch
            .push(Box::new(move |_| pool.defer(fences, owner)));
        Ok(())
    }

    fn launch_host_fn(&self, function: HostFunction) -> Result<()> {
        self.batch.push(Box::new(move |_| {
            function();
            Ok(())
        }));
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        self.batch.push(Box::new(move |stream| match stream {
            Some(stream) => stream.synchronize().map_err(|error| error.into()),
            None => Ok(()),
        }));
        Ok(())
    }
}

impl Issue for RecordingHost<'_> {
    type Host = Self;

    fn stream(&self) -> Option<&Stream> {
        None
    }

    fn host(&self) -> Self::Host {
        *self
    }

    unsafe fn copy(&self, copies: &CopyBatch) -> Result<()> {
        let copies = copies.clone();
        self.batch.push(Box::new(move |_| {
            // SAFETY: forwarded from the caller.
            unsafe { copy_host(&copies) };
            Ok(())
        }));
        Ok(())
    }

    unsafe fn copy_strided(&self, _copies: &StridedCopy) -> Result<()> {
        Err(Error::Invariant(
            "a strided copy is issued on a stream".into(),
        ))
    }

    unsafe fn fill(&self, fill: Fill) -> Result<()> {
        self.batch.push(Box::new(move |_| {
            // SAFETY: forwarded from the caller.
            unsafe { fill_host(fill) };
            Ok(())
        }));
        Ok(())
    }

    unsafe fn memset_2d(&self, fill: RowFill) -> Result<()> {
        self.batch.push(Box::new(move |_| {
            // SAFETY: forwarded from the caller.
            unsafe { fill_host_rows(fill) };
            Ok(())
        }));
        Ok(())
    }

    fn record(&self, _event: &Arc<Event>) -> Result<()> {
        Err(Error::Invariant("host storage records no event".into()))
    }

    fn wait(&self, _event: &Arc<Event>) -> Result<()> {
        Err(Error::Invariant("host storage waits for no event".into()))
    }

    fn defer(&self, events: &EventPool, fences: Vec<Arc<Event>>, owner: Deferred) -> Result<()> {
        Recording { batch: self.batch }.defer(events, fences, owner)
    }

    fn launch_host_fn(&self, function: HostFunction) -> Result<()> {
        self.batch.push(Box::new(move |_| {
            function();
            Ok(())
        }));
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// A pool a batch defers through, without device work.
    fn pool() -> Arc<EventPool> {
        EventPool::new()
    }

    #[test]
    fn a_batch_performs_what_it_recorded_in_order() -> Result<()> {
        let events = pool();
        let batch = Batch::new(None, Arc::clone(&events));
        let order = Arc::new(AtomicUsize::new(0));

        let first = Arc::clone(&order);
        // SAFETY: four bytes of host storage this test owns.
        unsafe {
            let mut bytes = [0_u8; 4];
            let mut copies = CopyBatch::new(platform::SourceOrder::DuringCall);
            copies.push(
                bytes.as_mut_ptr() as u64,
                [1_u8, 2, 3, 4].as_ptr() as u64,
                4,
            );
            batch.on().copy(&copies)?;
            std::mem::forget(bytes);
        }
        batch.on().launch_host_fn(Box::new(move || {
            first.fetch_add(1, Ordering::SeqCst);
        }));

        assert!(!batch.is_empty());
        assert_eq!(
            order.load(Ordering::SeqCst),
            0,
            "nothing runs before the batch"
        );
        batch.perform()?;
        assert_eq!(order.load(Ordering::SeqCst), 1);
        assert!(batch.perform().is_err(), "a batch is performed once");
        Ok(())
    }

    #[test]
    fn an_empty_batch_performs_nothing() -> Result<()> {
        let batch = Batch::new(None, pool());
        assert!(batch.is_empty());
        batch.perform()?;
        Ok(())
    }

    #[test]
    fn a_batch_defers_through_its_own_pool_only() {
        let batch = Batch::new(None, pool());
        let other = pool();
        let error = batch
            .on()
            .defer(&other, Vec::new(), Box::new(|| Ok(())))
            .expect_err("another rank's pool is not this batch's");
        assert!(error.to_string().contains("the pool it was made for"));
    }
}
