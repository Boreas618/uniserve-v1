//! One read's ticket: its admission, its consumable result and its
//! physical retirement.

use std::sync::Arc;

use uniserve_cuda::{DeviceGuard, Event, Stream};

use crate::stream::Issue;
use crate::{Completion, Error, EventPool, Outcome, Result};

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
pub struct TransferTicket<V, E, C> {
    value: Option<Arc<V>>,
    error: Option<Arc<E>>,
    cancelled: bool,
    access: ReadAccess,
    producer: Option<Arc<Event>>,
    copy_stream: Option<Arc<Stream>>,
    source_ready: Option<Arc<Event>>,
    closed: bool,
    undrained: bool,
    callbacks: Vec<C>,
    pub retirement: Arc<Completion<E, C>>,
}

impl<V, E, C> TransferTicket<V, E, C> {
    pub fn new(borrowed: bool) -> Self {
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
            retirement: Arc::new(Completion::default()),
        }
    }

    pub fn ready(&self) -> bool {
        self.value.is_some() || self.error.is_some()
    }

    pub fn retired(&self) -> bool {
        self.retirement.succeeded()
    }

    pub fn undrained(&self) -> bool {
        self.undrained
    }

    pub fn closed(&self) -> bool {
        self.closed
    }

    pub fn retirement_ready(&self) -> Result<bool> {
        if self.undrained {
            return Err(Error::Resource("transfer physical completion is unknown"));
        }

        Ok(self.retired())
    }

    pub fn add_done_callback(&mut self, callback: C) -> Option<C> {
        if self.ready() {
            return Some(callback);
        }

        self.callbacks.push(callback);
        None
    }

    /// Cancellation revokes consumption, but does not end a submitted access.
    pub fn cancel(&mut self, error: E) -> Vec<C> {
        self.cancelled = true;
        if self.error.is_none() {
            self.error = Some(Arc::new(error));
        }

        std::mem::take(&mut self.callbacks)
    }

    pub fn cancellation_error(&self) -> Option<Arc<E>> {
        if self.cancelled {
            self.error.clone()
        } else {
            None
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    /// Expose a completed view without replacing an earlier error or result.
    pub fn complete(&mut self, value: V, producer: Option<Arc<Event>>) -> Vec<C> {
        self.producer = producer;
        if self.ready() {
            return Vec::new();
        }

        self.value = Some(Arc::new(value));
        std::mem::take(&mut self.callbacks)
    }

    /// Return whether consumers could already have observed a successful view.
    /// Such a failure also invalidates the transport pool, not just this read.
    pub fn fail(&mut self, error: E) -> (bool, Vec<C>) {
        self.fail_shared(Arc::new(error))
    }

    pub fn fail_shared(&mut self, error: Arc<E>) -> (bool, Vec<C>) {
        let late = self.value.is_some();
        self.error = Some(error);
        (late, std::mem::take(&mut self.callbacks))
    }

    pub fn result(&self) -> Result<Outcome<E, Arc<V>>> {
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
    pub fn close(&mut self) -> bool {
        if !matches!(self.access, ReadAccess::Borrowed { .. }) || self.closed {
            return false;
        }

        self.closed = true;
        true
    }

    /// Capture writes made before submission. Later work on the submitting
    /// stream must not become a dependency of the transport copy.
    pub fn prepare_copy(&mut self, pool: &EventPool, stream: impl Issue) -> Result<()> {
        let ReadAccess::Copy { handoff } = &mut self.access else {
            return Err(Error::State("a borrowed transfer cannot copy into storage"));
        };
        *handoff = Some(pool.fence(stream, false)?);
        Ok(())
    }

    /// Order a transport stream after the destination's submitting stream.
    pub fn wait_for_copy(&self, stream: &Stream) -> Result<()> {
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
    pub fn consume(&mut self, stream: impl Issue) -> Result<()> {
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

    pub fn producer(&self) -> Option<&Arc<Event>> {
        self.producer.as_ref()
    }

    /// Fence each consumer at the end of borrowed access. The caller transfers
    /// these retained event references to deferred source release.
    pub fn record_consumers(&mut self, pool: &EventPool) -> Result<()> {
        if let ReadAccess::Borrowed { streams, events } = &mut self.access {
            for stream in streams.iter() {
                let _device = DeviceGuard::new(stream.device())?;
                events.push(pool.fence(stream, false)?);
            }
            streams.clear();
        }
        Ok(())
    }

    pub fn consumer_events(&self) -> &[Arc<Event>] {
        match &self.access {
            ReadAccess::Borrowed { events, .. } => events,
            ReadAccess::Copy { .. } => &[],
        }
    }

    pub fn release_consumers(&mut self) {
        if let ReadAccess::Borrowed { events, .. } = &mut self.access {
            events.clear();
        }
    }

    /// Move this ticket's retained readiness fences to deferred release.
    pub fn take_events(&mut self) -> Vec<Arc<Event>> {
        let mut events: Vec<_> = self.producer.take().into_iter().collect();
        if let ReadAccess::Copy { handoff } = &mut self.access {
            events.extend(handoff.take());
        }
        events
    }

    /// The backend could not drain device access. It must retain its backing
    /// allocations; neither cancellation nor result failure permits reuse.
    pub fn mark_undrained(
        &mut self,
        stream: Option<Arc<Stream>>,
        source_ready: Option<Arc<Event>>,
    ) {
        self.undrained = true;
        self.copy_stream = stream;
        self.source_ready = source_ready;
    }

    pub fn retire(&self) -> Result<Vec<C>> {
        if self.undrained {
            return Err(Error::Invariant(
                "transfer cannot retire with unknown physical completion".into(),
            ));
        }

        self.retirement.complete(Ok(()))
    }

    pub fn value(&self) -> Option<&V> {
        self.value.as_deref()
    }

    pub fn error(&self) -> Option<&E> {
        self.error.as_deref()
    }
}
