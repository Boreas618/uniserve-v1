//! One read's ticket: its admission, its consumable result and its
//! physical retirement.

use std::sync::Arc;

use uniserve_cuda::{DeviceGuard, Event, Stream};

use super::{Notify, Retirement, Spans};
use crate::stream::Issue;
use crate::{Error, EventPool, Outcome, Result};

enum ReadAccess {
    Copy {
        handoff: Option<Arc<Event>>,
    },
    Borrowed {
        /// Consumer streams not yet fenced, one handle per stream.
        streams: Vec<Stream>,
        events: Vec<Arc<Event>>,
    },
}

/// A read's result and physical lifetime. The transport serializes mutations
/// and dispatches returned callbacks after releasing its ownership lock.
///
/// A consumable view can be followed by a device failure, so result readiness
/// is separate from the one-shot physical retirement completion.
pub(crate) struct TransferTicket {
    value: Option<Arc<Spans>>,
    error: Option<Arc<Error>>,
    cancelled: bool,
    access: ReadAccess,
    producer: Option<Arc<Event>>,
    copy_stream: Option<Arc<Stream>>,
    source_ready: Option<Arc<Event>>,
    closed: bool,
    undrained: bool,
    callbacks: Vec<Notify>,
    pub retirement: Arc<Retirement>,
}

impl TransferTicket {
    pub(crate) fn new(borrowed: bool) -> Self {
        Self {
            value: None,
            error: None,
            cancelled: false,
            access: if borrowed {
                ReadAccess::Borrowed {
                    streams: Vec::new(),
                    events: Vec::new(),
                }
            } else {
                ReadAccess::Copy { handoff: None }
            },
            producer: None,
            copy_stream: None,
            source_ready: None,
            closed: false,
            undrained: false,
            callbacks: Vec::new(),
            retirement: Arc::new(Retirement::default()),
        }
    }

    pub(crate) fn ready(&self) -> bool {
        self.value.is_some() || self.error.is_some()
    }

    pub(crate) fn retired(&self) -> bool {
        self.retirement.succeeded()
    }

    pub(crate) fn undrained(&self) -> bool {
        self.undrained
    }

    pub(crate) fn retirement_ready(&self) -> Result<bool> {
        if self.undrained {
            return Err(Error::Resource("transfer physical completion is unknown"));
        }

        Ok(self.retired())
    }

    pub(crate) fn add_done_callback(&mut self, callback: Notify) -> Option<Notify> {
        if self.ready() {
            return Some(callback);
        }

        self.callbacks.push(callback);
        None
    }

    /// Cancellation revokes consumption, but does not end a submitted access.
    pub(crate) fn cancel(&mut self, error: Error) -> Vec<Notify> {
        self.cancelled = true;
        if self.error.is_none() {
            self.error = Some(Arc::new(error));
        }

        std::mem::take(&mut self.callbacks)
    }

    pub(crate) fn cancellation_error(&self) -> Option<Arc<Error>> {
        if self.cancelled {
            self.error.clone()
        } else {
            None
        }
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    /// Expose a completed view without replacing an earlier error or result.
    pub(crate) fn complete(&mut self, value: Spans, producer: Option<Arc<Event>>) -> Vec<Notify> {
        self.producer = producer;
        if self.ready() {
            return Vec::new();
        }

        self.value = Some(Arc::new(value));
        std::mem::take(&mut self.callbacks)
    }

    pub(crate) fn fail_shared(&mut self, error: Arc<Error>) -> (bool, Vec<Notify>) {
        let late = self.value.is_some();
        self.error = Some(error);
        (late, std::mem::take(&mut self.callbacks))
    }

    pub(crate) fn result(&self) -> Result<Outcome<Arc<Spans>>> {
        if let Some(error) = &self.error {
            return Ok(Outcome::Failed(Arc::clone(error)));
        }

        let value = self.value.as_ref().ok_or(Error::State(
            "transfer ticket was observed before readiness",
        ))?;
        if self.closed {
            return Err(Error::State("transfer consumption has already closed"));
        }

        Ok(Outcome::Success(Arc::clone(value)))
    }

    /// End borrowed consumption once. Copy destinations retain their ordinary
    /// value ownership and remain readable after close.
    pub(crate) fn close(&mut self) -> bool {
        if !matches!(self.access, ReadAccess::Borrowed { .. }) || self.closed {
            return false;
        }

        self.closed = true;
        true
    }

    /// Capture writes made before submission. Later work on the submitting
    /// stream must not become a dependency of the transport copy.
    pub(crate) fn prepare_copy(&mut self, pool: &EventPool, stream: impl Issue) -> Result<()> {
        let ReadAccess::Copy { handoff } = &mut self.access else {
            return Err(Error::State("a borrowed transfer cannot copy into storage"));
        };
        *handoff = Some(pool.fence(stream, false)?);
        Ok(())
    }

    /// Order a transport stream after the destination's submitting stream.
    pub(crate) fn wait_for_copy(&self, stream: &Stream) -> Result<()> {
        let ReadAccess::Copy { handoff } = &self.access else {
            return Err(Error::State("a borrowed transfer cannot copy into storage"));
        };
        if let Some(event) = handoff {
            stream.wait_event(event)?;
        }
        Ok(())
    }

    /// Order a consuming stream after the producer and, for borrowed reads,
    /// remember it so close can fence its access to the source.
    pub(crate) fn consume(&mut self, stream: impl Issue) -> Result<()> {
        if let Some(producer) = &self.producer {
            stream.wait(producer)?;
            if let (ReadAccess::Borrowed { streams, .. }, Some(consumer)) =
                (&mut self.access, stream.stream())
                && !streams.iter().any(|known| known.raw() == consumer.raw())
            {
                streams.push(consumer.clone());
            }
        }
        Ok(())
    }

    pub(crate) fn producer(&self) -> Option<&Arc<Event>> {
        self.producer.as_ref()
    }

    /// Fence each consumer at the end of borrowed access. The caller transfers
    /// these retained event references to deferred source release.
    pub(crate) fn record_consumers(&mut self, pool: &EventPool) -> Result<()> {
        if let ReadAccess::Borrowed { streams, events } = &mut self.access {
            for stream in streams.iter() {
                let _device = DeviceGuard::new(stream.device())?;
                events.push(pool.fence(stream, false)?);
            }
            streams.clear();
        }
        Ok(())
    }

    pub(crate) fn consumer_events(&self) -> &[Arc<Event>] {
        match &self.access {
            ReadAccess::Borrowed { events, .. } => events,
            ReadAccess::Copy { .. } => &[],
        }
    }

    pub(crate) fn release_consumers(&mut self) {
        if let ReadAccess::Borrowed { events, .. } = &mut self.access {
            events.clear();
        }
    }

    /// Move this ticket's retained readiness fences to deferred release.
    pub(crate) fn take_events(&mut self) -> Vec<Arc<Event>> {
        let mut events: Vec<_> = self.producer.take().into_iter().collect();
        if let ReadAccess::Copy { handoff } = &mut self.access {
            events.extend(handoff.take());
        }
        events
    }

    /// The backend could not drain device access. It must retain its backing
    /// allocations; neither cancellation nor result failure permits reuse.
    pub(crate) fn mark_undrained(
        &mut self,
        stream: Option<Arc<Stream>>,
        source_ready: Option<Arc<Event>>,
    ) {
        self.undrained = true;
        self.copy_stream = stream;
        self.source_ready = source_ready;
    }

    pub(crate) fn retire(&self) -> Result<Vec<Notify>> {
        if self.undrained {
            return Err(Error::Invariant(
                "transfer cannot retire with unknown physical completion".into(),
            ));
        }

        self.retirement.complete(Ok(()))
    }

    pub(crate) fn value(&self) -> Option<&Spans> {
        self.value.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::ffi::{Backing, DType};
    use crate::storage::memory::{Memory, StorageView};
    use crate::transfer::{TransferCapacity, notify};

    /// A host value of `elements` bytes.
    fn value(elements: i64) -> Result<Spans> {
        Spans::single(StorageView::contiguous(
            Memory::host(elements as usize) as Arc<dyn Backing>,
            0,
            DType::U8,
            &[elements],
        )?)
    }

    /// An observer and the number of times it ran.
    fn observer() -> (Notify, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&calls);
        let observer: Notify = Arc::new(move || {
            counted.fetch_add(1, Ordering::Relaxed);
        });
        (observer, calls)
    }

    #[test]
    fn cancelling_a_read_does_not_return_its_physical_credit() -> Result<()> {
        let capacity = Arc::new(TransferCapacity::new(8, 1)?);
        capacity.take_reads(1)?;
        let mut ticket = TransferTicket::new(false);
        let (done, calls) = observer();
        assert!(ticket.add_done_callback(done).is_none());

        let released = Arc::clone(&capacity);
        let returned: Notify = Arc::new(move || {
            if let Err(error) = released.return_reads(1) {
                panic!("{error}");
            }
        });
        assert!(ticket.retirement.subscribe(returned).is_none());
        notify(ticket.cancel(Error::Resource("transfer read was cancelled")));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        // A copy completing after the cancellation leaves the cancellation
        // as the result.
        notify(ticket.complete(value(3)?, None));
        assert!(matches!(ticket.result()?, Outcome::Failed(_)));
        assert!(ticket.cancellation_error().is_some());
        assert!(!ticket.retirement_ready()?);
        assert!(matches!(
            capacity.take_reads(1),
            Err(Error::ReadBackpressure { .. })
        ));

        notify(ticket.retire()?);
        assert!(ticket.retirement_ready()?);
        capacity.take_reads(1)?;
        capacity.return_reads(1)
    }

    #[test]
    fn a_late_read_failure_preserves_observed_views_but_prevents_more_consumption() -> Result<()> {
        let mut ticket = TransferTicket::new(false);
        let (done, calls) = observer();
        assert!(ticket.add_done_callback(done).is_none());
        notify(ticket.complete(value(3)?, None));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        let Outcome::Success(observed) = ticket.result()? else {
            return Err(Error::State("a completed read has no value"));
        };

        let (late, callbacks) =
            ticket.fail_shared(Arc::new(Error::Transport("device copy failed".into())));
        notify(callbacks);
        assert!(late);
        assert_eq!(observed.shape(), vec![3]);
        assert!(matches!(ticket.result()?, Outcome::Failed(_)));
        let (later, _) = observer();
        assert!(ticket.add_done_callback(later).is_some());

        ticket.mark_undrained(None, None);
        assert!(ticket.retirement_ready().is_err());
        assert!(ticket.retire().is_err());
        assert!(!ticket.retirement.done());
        notify(ticket.cancel(Error::Resource("transfer read was cancelled")));
        let failure = ticket
            .cancellation_error()
            .ok_or(Error::State("the cancelled read lost its error"))?;
        assert!(failure.to_string().contains("device copy failed"));
        assert!(!ticket.retired());
        Ok(())
    }

    #[test]
    fn borrowed_consumption_closes_before_its_shared_retirement_completes() -> Result<()> {
        let mut borrowed = TransferTicket::new(true);
        assert!(borrowed.result().is_err());
        notify(borrowed.complete(value(1)?, None));
        assert!(borrowed.close());
        assert!(!borrowed.close());
        assert!(borrowed.result().is_err());
        assert!(!borrowed.retirement_ready()?);

        let retirement = Arc::clone(&borrowed.retirement);
        drop(borrowed);
        // A transport can finish the consumer fences after the ticket is
        // dropped.
        notify(retirement.complete(Ok(()))?);
        assert!(retirement.succeeded());
        let (late, _) = observer();
        assert!(retirement.subscribe(late).is_some());

        let mut copied = TransferTicket::new(false);
        notify(copied.complete(value(1)?, None));
        assert!(!copied.close());
        assert!(matches!(copied.result()?, Outcome::Success(_)));
        Ok(())
    }
}
